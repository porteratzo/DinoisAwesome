# %% [markdown]
# # Fundamental: Augmented Prototypes — Does Perturbing the Exemplar Crop Improve
# # Oracle-IoU Localization?
#
# Third experiment in `experiments/fundamental/`. `scale_crop_similarity.py` asked how
# crop tightness moves an object's own embedding; `augmentation_sensitivity.py` asked how
# far six common perturbations move a *fixed* crop's embedding from its own unperturbed
# baseline. This one asks the practical question behind both: if the reference crop used
# to build a matching prototype is perturbed (rotated, relit, blurred, ...) before being
# pooled into a masked-mean exemplar, does scoring a *different* image with that perturbed
# prototype localize the object better or worse than the plain, unaugmented prototype —
# i.e. does augmenting a training-time exemplar behave like ordinary data augmentation
# (broadening what the prototype matches) or does it just walk the prototype away from a
# real match?
#
# "Oracle IoU" here means: given one raw cosine-similarity heatmap (prototype vs. every
# patch of the query image), sweep every candidate threshold and keep the best patch-mask
# IoU against the query's *own* ground-truth mask — the same `iou_threshold_curve` search
# `object_detection/multiscale_crop_ablation.py` uses to *tune* a threshold, but applied
# directly to the query instead of fit-on-ref/apply-to-query. It isolates "how separable
# is object from background in this score map" from "how would we pick a threshold in
# practice", so an augmentation's effect on localizability can be read off on its own.
#
# Steps:
#   1. Load one abc3 reference image + one annotated instance's mask (same object as the
#      other two fundamental experiments), and a *different* abc3 image of the same part
#      type as the query — mirrors `multiscale_crop_ablation.py`'s ref/query split.
#   2. Build two prototype source crops from the reference instance — "close" (tight bbox
#      + padding) and "mid" (halfway to the full image) — the same geometry as
#      `multiscale_crop_ablation.py`'s `scale_crop_box`.
#   3. Encode the query image once. For each scale's *unaugmented* crop, build a
#      masked-mean prototype, score every query patch (cosine similarity -> raw heatmap),
#      and compute its oracle IoU against the query's own GT mask — the baseline.
#   4. For each augmentation family (rotation, illumination/gamma, color jitter, Gaussian
#      blur, Gaussian noise, JPEG compression) and severity sweep (same families/values as
#      `augmentation_sensitivity.py`, for direct comparability), apply it to the scale's
#      crop (reprojecting the mask for rotation, the only geometry-moving family), rebuild
#      the prototype, rescore the *same* query image, and recompute oracle IoU.
#   5. Compare oracle-IoU-vs-severity curves for "close" vs. "mid", plus a summary of each
#      family's best achievable oracle IoU (max over severities) against the
#      no-augmentation baseline, for both scales.
#   6. Visualize the augmented crop grids (per scale) and the oracle-IoU curves/summary.
#   7. Compose an ensemble prototype per (scale, family) by averaging L2-normalised
#      per-severity prototypes and re-normalising, swept cumulatively (k=1 original alone
#      through k=N all severities pooled) to see whether the benefit — if any — builds up
#      gradually or only shows up once every severity is folded in, then compare the k=N
#      endpoint against the original baseline and the best single-severity prototype.
#   8. Visualize the baseline vs. best single (family, severity) combo's actual similarity
#      heatmap, one row per scale, so a qualitative look at *where* the score map changed
#      backs up (or undercuts) the oracle-IoU number.
#   9. Flip the roles: hold each scale's prototype clean (the plain, unaugmented baseline)
#      and instead perturb the *query* image (reprojecting its own GT mask the same way),
#      then rescore and compare against the crop-augmentation sensitivity ranking above —
#      the more realistic "clean exemplar, noisy factory frame" scenario.

# %% Logging — must be before torch import
import io
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("augmented_prototype_oracle_iou")

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image, ImageFilter
from torchvision import transforms

from dinoisawesome import DinoEncoder, compute_exemplar_features, load_annotations
from dinoisawesome.instance_detection import extract_patch_tokens

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

data_dir = _REPO_ROOT / "data" / "abc3"

# Same object/class as scale_crop_similarity.py and augmentation_sensitivity.py, for
# comparability. QUERY_STEM is a *different* abc3 image of the same part type — mirrors
# multiscale_crop_ablation.py's REF_NUMBER=1 / QUERY_NUMBER=2 ref/query split.
REF_STEM = "LHa_1"
QUERY_STEM = "LHa_2"
TARGET_CLASS = "donut foam single"
INSTANCE_INDEX = 0

DINO_VERSION = "v3"
DINO_SIZE = "large"
IMG_SIZE = 1024  # must be divisible by patch_size (16 for v3)
LAYER_IDX = 23  # penultimate/last block of ViT-L/16 (depth 24)
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")

MASK_PATCH_THRESHOLD = 0.3  # patch-grid cell counts as "object" once this fraction is masked
CROP_PADDING_FRACTION = 1.0  # close/mid crop padding, fraction of the mask bbox's own extent
MIN_CROP_SIZE = 128  # "close" crop must be at least this many native px on each side

ORACLE_THRESHOLD_STEPS = 25  # candidate thresholds searched per oracle-IoU evaluation

SCALES: list[str] = ["close", "mid"]
SCALE_COLOR: dict[str, str] = {"close": "#e74c3c", "mid": "#f39c12"}

SEED = 0
torch.manual_seed(SEED)

OUTPUT_DIR = _REPO_ROOT / "outputs" / "fundamental" / "augmented_prototype_oracle_iou"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log.info(
    "ref=%s query=%s class=%r  |  DINO%s-%s img_size=%d layer=%d",
    REF_STEM,
    QUERY_STEM,
    TARGET_CLASS,
    DINO_VERSION,
    DINO_SIZE,
    IMG_SIZE,
    LAYER_IDX,
)

# %% Mask / geometry / IoU helpers (same as multiscale_crop_ablation.py)


def mask_bbox_px(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of the True region as (rmin, rmax, cmin, cmax)."""
    rows, cols = np.where(mask)
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def pixel_mask_to_patch_mask(
    pixel_mask: np.ndarray, grid_h: int, grid_w: int, img_size: int, threshold: float
) -> np.ndarray:
    """Resize a pixel-space mask to patch-grid resolution, (grid_h, grid_w) bool."""
    mask_pil = Image.fromarray(pixel_mask.astype(np.uint8) * 255)
    mask_resized = np.array(mask_pil.resize((img_size, img_size), Image.NEAREST)) > 0
    ph = img_size // grid_h
    pw = img_size // grid_w
    tiled = mask_resized.reshape(grid_h, ph, grid_w, pw)
    patch_density = tiled.mean(axis=(1, 3))
    return patch_density >= threshold


def scale_crop_box(
    pixel_mask: np.ndarray, scale: str, padding_frac: float
) -> tuple[int, int, int, int]:
    """PIL-style crop box (x0, y0, x1, y1) for the "close" or "mid" prototype scale.

    close — tight bbox around the mask, padded by *padding_frac* of its own extent.
    mid   — midpoint between the close box and the full image.
    """
    H, W = pixel_mask.shape
    rmin, rmax, cmin, cmax = mask_bbox_px(pixel_mask)
    pad_r = int((rmax - rmin) * padding_frac)
    pad_c = int((cmax - cmin) * padding_frac)
    close_box = (
        max(0, cmin - pad_c),
        max(0, rmin - pad_r),
        min(W, cmax + pad_c),
        min(H, rmax + pad_r),
    )
    if scale == "close":
        return close_box
    if scale == "mid":
        global_box = (0, 0, W, H)
        return (
            (close_box[0] + global_box[0]) // 2,
            (close_box[1] + global_box[1]) // 2,
            (close_box[2] + global_box[2]) // 2,
            (close_box[3] + global_box[3]) // 2,
        )
    raise ValueError(f"Unknown scale: {scale!r}")


def mean_color(img: Image.Image) -> tuple[int, int, int]:
    return tuple(int(c) for c in np.array(img).reshape(-1, 3).mean(axis=0))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / (union + 1e-8)


def iou_threshold_curve(
    raw: np.ndarray, gt_mask: np.ndarray, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Patch-mask IoU of `(raw > c)` vs. `gt_mask` for `steps` candidates spanning raw's range."""
    candidates = np.linspace(raw.min(), raw.max(), steps)
    ious = np.array([mask_iou(raw > c, gt_mask) for c in candidates])
    return candidates, ious


def oracle_iou(raw: np.ndarray, gt_mask: np.ndarray, steps: int) -> float:
    """Best patch-mask IoU any single global threshold on *raw* could achieve against
    *gt_mask* — "oracle" because the threshold is picked with direct knowledge of the
    query's own ground truth, the upper bound a real (ref-tuned) threshold could reach."""
    _, ious = iou_threshold_curve(raw, gt_mask, steps)
    return float(ious.max())


# %% Load reference image + instance mask, build close/mid crops
ref_anns = load_annotations(data_dir / "annotations" / REF_STEM)
ref_instances = [a for a in ref_anns if a["class"] == TARGET_CLASS]
if not ref_instances:
    raise ValueError(f"No instances of class {TARGET_CLASS!r} in {REF_STEM}")
ref_mask = ref_instances[INSTANCE_INDEX]["mask"]  # (H, W) bool, full native resolution

ref_img = Image.open(data_dir / f"{REF_STEM}.jpg").convert("RGB")

base_crops: dict[str, Image.Image] = {}
base_masks_px: dict[str, np.ndarray] = {}
fills: dict[str, tuple[int, int, int]] = {}
for scale in SCALES:
    box = scale_crop_box(ref_mask, scale, CROP_PADDING_FRACTION)
    x0, y0, x1, y1 = box
    if x1 - x0 < MIN_CROP_SIZE or y1 - y0 < MIN_CROP_SIZE:
        raise ValueError(
            f"scale={scale!r} crop {box} is below MIN_CROP_SIZE={MIN_CROP_SIZE}px — "
            "reduce CROP_PADDING_FRACTION or pick a larger instance."
        )
    base_crops[scale] = ref_img.crop(box)
    base_masks_px[scale] = ref_mask[y0:y1, x0:x1]
    fills[scale] = mean_color(base_crops[scale])
    log.info("scale=%-5s box=%s size=%dx%dpx", scale, box, x1 - x0, y1 - y0)

# %% Load query image + its own GT mask for the same class
query_anns = load_annotations(data_dir / "annotations" / QUERY_STEM)
query_instances = [a for a in query_anns if a["class"] == TARGET_CLASS]
if not query_instances:
    raise ValueError(f"No instances of class {TARGET_CLASS!r} in {QUERY_STEM}")
query_pixel_mask = np.stack([a["mask"] for a in query_instances]).any(axis=0)
query_img = Image.open(data_dir / f"{QUERY_STEM}.jpg").convert("RGB")

# %% Encode the query image once
encoder = DinoEncoder(
    version=DINO_VERSION,
    size=DINO_SIZE,
    img_size=IMG_SIZE,
    layers=[LAYER_IDX],
    weights_dir=DINO_WEIGHTS_DIR,
    amp=True,
)
q_tokens, q_h, q_w = extract_patch_tokens(encoder, query_img, LAYER_IDX, debias=True)  # (N, D)
gt_patch_mask = pixel_mask_to_patch_mask(query_pixel_mask, q_h, q_w, IMG_SIZE, MASK_PATCH_THRESHOLD)
log.info(
    "query grid=%dx%d  GT patches=%d/%d", q_h, q_w, int(gt_patch_mask.sum()), gt_patch_mask.size
)

# %% Augmentation families — same families/values as augmentation_sensitivity.py. Each
# sweep starts at a literal no-op value (angle=0, gamma=1.0, jitter magnitude=0, ...), so
# severity index 0 is pixel-identical to the scale's *base_crop* — one no-augmentation
# baseline per scale, shared by every family's severity=0 entry.

AugmentFn = Callable[
    [Image.Image, np.ndarray, float, tuple[int, int, int]], tuple[Image.Image, np.ndarray]
]


def _pixel_only(fn: Callable[[Image.Image, float], Image.Image]) -> AugmentFn:
    """Wrap a pixel-value-only augmentation to the (img, mask, val, fill) -> (img, mask) shape."""

    def wrapped(
        img: Image.Image, mask_px: np.ndarray, val: float, fill: tuple[int, int, int]
    ) -> tuple[Image.Image, np.ndarray]:
        return fn(img, val), mask_px

    return wrapped


def _apply_rotation(
    img: Image.Image, mask_px: np.ndarray, angle: float, fill: tuple[int, int, int]
) -> tuple[Image.Image, np.ndarray]:
    if angle == 0:
        return img, mask_px
    rotated_img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=fill)
    mask_pil = Image.fromarray(mask_px.astype(np.uint8) * 255)
    rotated_mask = (
        np.array(mask_pil.rotate(angle, resample=Image.NEAREST, expand=False, fillcolor=0)) > 0
    )
    return rotated_img, rotated_mask


def _apply_gamma(img: Image.Image, gamma: float) -> Image.Image:
    if gamma == 1.0:
        return img
    arr = (np.array(img).astype(np.float32) / 255.0) ** gamma
    return Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))


def _apply_color_jitter(img: Image.Image, magnitude: float) -> Image.Image:
    if magnitude == 0:
        return img
    torch.manual_seed(SEED)  # deterministic draw per severity level
    jitter = transforms.ColorJitter(
        brightness=magnitude,
        contrast=magnitude,
        saturation=magnitude,
        hue=min(magnitude * 0.5, 0.5),
    )
    return jitter(img)


def _apply_blur(img: Image.Image, radius: float) -> Image.Image:
    if radius == 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _apply_noise(img: Image.Image, sigma: float) -> Image.Image:
    if sigma == 0:
        return img
    rng = np.random.default_rng(SEED)
    arr = np.array(img).astype(np.float32)
    noisy = np.clip(arr + rng.normal(0.0, sigma, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def _apply_jpeg(img: Image.Image, quality: float) -> Image.Image:
    if quality >= 100:
        return img
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


AUGMENTATIONS: dict[str, dict] = {
    "rotation": {
        "values": [0, 8, 16, 30, 50, 75],
        "unit": "deg",
        "apply": _apply_rotation,
    },
    "illumination (gamma)": {
        "values": [1.0, 1.3, 1.7, 2.2, 2.8, 3.5],
        "unit": "gamma",
        "apply": _pixel_only(_apply_gamma),
    },
    "color jitter": {
        "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "unit": "magnitude",
        "apply": _pixel_only(_apply_color_jitter),
    },
    "gaussian blur": {
        "values": [0, 1, 2, 4, 7, 11],
        "unit": "px radius",
        "apply": _pixel_only(_apply_blur),
    },
    "gaussian noise": {
        "values": [0, 8, 16, 28, 45, 70],
        "unit": "sigma (0-255)",
        "apply": _pixel_only(_apply_noise),
    },
    "jpeg compression": {
        "values": [100, 80, 60, 35, 15, 3],
        "unit": "quality",
        "apply": _pixel_only(_apply_jpeg),
    },
}

# %% Build every (scale, family, severity) augmented crop up front
entries: list[dict] = []
for scale in SCALES:
    for family, spec in AUGMENTATIONS.items():
        for val in spec["values"]:
            img, mask_px = spec["apply"](base_crops[scale], base_masks_px[scale], val, fills[scale])
            entries.append(
                {"scale": scale, "family": family, "value": val, "img": img, "mask_px": mask_px}
            )

log.info(
    "Built %d augmented crops across %d scales x %d families",
    len(entries),
    len(SCALES),
    len(AUGMENTATIONS),
)

# %% Encode every augmented crop in a single batched forward pass
out = encoder([e["img"] for e in entries], layers=[LAYER_IDX], debias=True)
patches = out.patches[:, 0]  # (N, grid_h, grid_w, D)
_, grid_h, grid_w, D = patches.shape

# %% Per-crop prototype -> score the query -> oracle IoU
for entry, patch_tokens in zip(entries, patches):
    patch_mask = pixel_mask_to_patch_mask(
        entry["mask_px"], grid_h, grid_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )
    tokens = F.normalize(patch_tokens.reshape(grid_h * grid_w, D), p=2, dim=-1)
    patch_flat = torch.from_numpy(patch_mask.reshape(-1)).to(tokens.device)

    masked = tokens[patch_flat]
    if masked.shape[0] == 0:
        log.warning(
            "scale=%s family=%s value=%s: mask empty after projection — using all crop patches",
            entry["scale"],
            entry["family"],
            entry["value"],
        )
        masked = tokens
    prototype = compute_exemplar_features(masked, mode="mean")  # (1, D)

    raw = (q_tokens @ prototype.T).reshape(q_h, q_w).cpu().float().numpy()
    entry["prototype"] = prototype
    entry["raw"] = raw
    entry["oracle_iou"] = oracle_iou(raw, gt_patch_mask, ORACLE_THRESHOLD_STEPS)

# %% Aggregate: oracle-IoU-vs-severity per scale/family, and the shared no-aug baseline
baseline_oracle_iou: dict[str, float] = {
    scale: next(
        e["oracle_iou"]
        for e in entries
        if e["scale"] == scale
        and e["family"] == next(iter(AUGMENTATIONS))
        and e["value"] == AUGMENTATIONS[next(iter(AUGMENTATIONS))]["values"][0]
    )
    for scale in SCALES
}

curves: dict[str, dict[str, list[float]]] = {scale: {} for scale in SCALES}
best_over_baseline: list[dict] = []
for scale in SCALES:
    for family, spec in AUGMENTATIONS.items():
        values = spec["values"]
        ious = [
            next(
                e["oracle_iou"]
                for e in entries
                if e["scale"] == scale and e["family"] == family and e["value"] == v
            )
            for v in values
        ]
        curves[scale][family] = ious
        best_idx = int(np.argmax(ious))
        best_over_baseline.append(
            {
                "scale": scale,
                "family": family,
                "best_value": values[best_idx],
                "best_iou": ious[best_idx],
                "delta_vs_baseline": ious[best_idx] - baseline_oracle_iou[scale],
            }
        )
        log.info(
            "scale=%-5s %-22s values=%s  oracle_iou=%s",
            scale,
            family,
            values,
            np.round(ious, 3),
        )

log.info(
    "Baseline (no augmentation) oracle IoU: %s",
    {k: round(v, 3) for k, v in baseline_oracle_iou.items()},
)
for row in sorted(best_over_baseline, key=lambda r: -r["delta_vs_baseline"]):
    log.info(
        "BEST scale=%-5s %-22s value=%-6s oracle_iou=%.3f (%+.3f vs. baseline)",
        row["scale"],
        row["family"],
        row["best_value"],
        row["best_iou"],
        row["delta_vs_baseline"],
    )

# %% Visualization — augmented crop grid, one figure per scale, one row per family
for scale in SCALES:
    n_families = len(AUGMENTATIONS)
    n_cols = max(len(spec["values"]) for spec in AUGMENTATIONS.values())
    fig, axes = plt.subplots(n_families, n_cols, figsize=(2.6 * n_cols, 2.9 * n_families))
    for row, (family, spec) in enumerate(AUGMENTATIONS.items()):
        for col, val in enumerate(spec["values"]):
            ax = axes[row, col]
            entry = next(
                e
                for e in entries
                if e["scale"] == scale and e["family"] == family and e["value"] == val
            )
            ax.imshow(entry["img"])
            ax.set_title(f"{val} {spec['unit']}\noracle_iou={entry['oracle_iou']:.3f}", fontsize=8)
            ax.axis("off")
        for col in range(len(spec["values"]), n_cols):
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(family, fontsize=9)
    fig.suptitle(
        f"Augmented '{scale}' prototype crops — {REF_STEM} -> {QUERY_STEM} / {TARGET_CLASS!r}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUTPUT_DIR / f"augmented_crops_{scale}.png", dpi=150, bbox_inches="tight")

# %% Visualization — oracle-IoU curves, one panel per scale (x normalized to severity fraction)
fig, axes = plt.subplots(1, len(SCALES), figsize=(7.5 * len(SCALES), 5.5), sharey=True)
colors = plt.get_cmap("tab10").colors
for ax, scale in zip(axes, SCALES):
    for i, (family, spec) in enumerate(AUGMENTATIONS.items()):
        values = np.array(spec["values"], dtype=float)
        frac = (values - values[0]) / (values[-1] - values[0])
        ax.plot(
            frac, curves[scale][family], marker="o", label=family, color=colors[i % len(colors)]
        )
    ax.axhline(
        baseline_oracle_iou[scale],
        color="k",
        linestyle="--",
        linewidth=1,
        label="no-augmentation baseline",
    )
    ax.set_xlabel("severity fraction (0 = no-op, 1 = strongest tested)")
    ax.set_title(f"scale = {scale}")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("oracle IoU (best threshold vs. query GT)")
axes[0].legend(fontsize=8)
fig.suptitle(f"Oracle-IoU vs. prototype augmentation severity — {REF_STEM} -> {QUERY_STEM}")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "oracle_iou_curves.png", dpi=150, bbox_inches="tight")

# %% Visualization — best achievable oracle IoU per family vs. baseline, grouped by scale
fig, ax = plt.subplots(figsize=(9, 5))
family_names = list(AUGMENTATIONS.keys())
x = np.arange(len(family_names))
width = 0.8 / len(SCALES)
for i, scale in enumerate(SCALES):
    bests = [
        next(r["best_iou"] for r in best_over_baseline if r["scale"] == scale and r["family"] == f)
        for f in family_names
    ]
    ax.bar(
        x + i * width,
        bests,
        width=width,
        label=f"{scale} (best over severities)",
        color=SCALE_COLOR[scale],
    )
    ax.axhline(
        baseline_oracle_iou[scale], color=SCALE_COLOR[scale], linestyle="--", linewidth=1, alpha=0.7
    )
ax.set_xticks(x + width * (len(SCALES) - 1) / 2, family_names, rotation=30, ha="right")
ax.set_ylabel("oracle IoU")
ax.set_title("Best achievable oracle IoU per augmentation family (dashed = that scale's baseline)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "best_vs_baseline.png", dpi=150, bbox_inches="tight")

log.info("Saved figures to %s", OUTPUT_DIR)

# %% [markdown]
# ## Composed prototype augmentation
#
# The severity sweep above scores one augmented crop's prototype at a time. This section
# asks the ensemble question instead: per (scale, family), pool every severity's prototype
# — the no-op included — into one "composed" prototype by averaging the L2-normalised
# per-crop vectors (`compute_exemplar_features(mode="mean")` already returns unit vectors)
# and re-normalising the mean. It's a cheap, training-free stand-in for the multiple
# augmented views a real training pipeline would show a model.
#
# To see whether any benefit builds up gradually or only appears once every severity is
# folded in, each family's composed prototype is swept cumulatively: k=1 uses just the
# original (no-op) crop, k=2 averages original+severity-2, ... up to k=N (all severities
# pooled) — the same sweep the line plots below chart, with the k=N endpoint reused as
# "composed" in the summary bar chart that follows.

# %% Composed prototype augmentation — cumulative severity-averaged prototypes, k=1..N
composed_curves: dict[str, dict[str, list[float]]] = {scale: {} for scale in SCALES}
composed_results: list[dict] = []
for scale in SCALES:
    for family in AUGMENTATIONS:
        family_entries = [e for e in entries if e["scale"] == scale and e["family"] == family]
        cumulative_ious: list[float] = []
        for k in range(1, len(family_entries) + 1):
            composed_prototype = F.normalize(
                torch.cat([e["prototype"] for e in family_entries[:k]], dim=0).mean(
                    dim=0, keepdim=True
                ),
                p=2,
                dim=-1,
            )
            composed_raw = (q_tokens @ composed_prototype.T).reshape(q_h, q_w).cpu().float().numpy()
            cumulative_ious.append(oracle_iou(composed_raw, gt_patch_mask, ORACLE_THRESHOLD_STEPS))
        composed_curves[scale][family] = cumulative_ious

        composed_oracle_iou = cumulative_ious[-1]  # k = N, every severity pooled
        best_single_severity_oracle_iou = next(
            r["best_iou"]
            for r in best_over_baseline
            if r["scale"] == scale and r["family"] == family
        )
        composed_results.append(
            {
                "scale": scale,
                "family": family,
                "n_severities_composed": len(family_entries),
                "original_oracle_iou": baseline_oracle_iou[scale],
                "composed_oracle_iou": composed_oracle_iou,
                "best_single_severity_oracle_iou": best_single_severity_oracle_iou,
                "delta_vs_original": composed_oracle_iou - baseline_oracle_iou[scale],
                "delta_vs_best_single": composed_oracle_iou - best_single_severity_oracle_iou,
            }
        )

log.info("Composed (all-severities-averaged) prototype vs. original vs. best single severity:")
for row in sorted(composed_results, key=lambda r: -r["delta_vs_original"]):
    log.info(
        "scale=%-5s %-22s composed=%.3f  original=%.3f (%+.3f)  best_single=%.3f (%+.3f)",
        row["scale"],
        row["family"],
        row["composed_oracle_iou"],
        row["original_oracle_iou"],
        row["delta_vs_original"],
        row["best_single_severity_oracle_iou"],
        row["delta_vs_best_single"],
    )

# %% Visualization — composed oracle-IoU curves (cumulative severities), one panel per scale
fig, axes = plt.subplots(1, len(SCALES), figsize=(7.5 * len(SCALES), 5.5), sharey=True)
for ax, scale in zip(axes, SCALES):
    for i, family in enumerate(AUGMENTATIONS):
        ious = composed_curves[scale][family]
        ax.plot(
            range(1, len(ious) + 1), ious, marker="o", label=family, color=colors[i % len(colors)]
        )
    ax.axhline(
        baseline_oracle_iou[scale],
        color="k",
        linestyle="--",
        linewidth=1,
        label="no-augmentation baseline",
    )
    ax.set_xlabel("severities composed (cumulative, k=1 is the original alone)")
    ax.set_title(f"scale = {scale}")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("oracle IoU (composed prototype vs. query GT)")
axes[0].legend(fontsize=8)
fig.suptitle(f"Composed-prototype oracle-IoU vs. severities averaged — {REF_STEM} -> {QUERY_STEM}")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "composed_oracle_iou_curves.png", dpi=150, bbox_inches="tight")

log.info("Saved composed oracle-IoU curves to %s", OUTPUT_DIR / "composed_oracle_iou_curves.png")

# %% Visualization — composed (all-severities) vs. original vs. best-single-severity
fig, ax = plt.subplots(figsize=(9, 5))
family_names = list(AUGMENTATIONS.keys())
x = np.arange(len(family_names))
width = 0.8 / len(SCALES)
for i, scale in enumerate(SCALES):
    composed_ious = [
        next(
            r["composed_oracle_iou"]
            for r in composed_results
            if r["scale"] == scale and r["family"] == f
        )
        for f in family_names
    ]
    ax.bar(
        x + i * width,
        composed_ious,
        width=width,
        label=f"{scale} (composed, all severities)",
        color=SCALE_COLOR[scale],
    )
    ax.axhline(
        baseline_oracle_iou[scale], color=SCALE_COLOR[scale], linestyle="--", linewidth=1, alpha=0.7
    )
ax.set_xticks(x + width * (len(SCALES) - 1) / 2, family_names, rotation=30, ha="right")
ax.set_ylabel("oracle IoU")
ax.set_title(
    "Composed (all-severities-averaged) prototype oracle IoU "
    "(dashed = that scale's original baseline)"
)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "composed_vs_baseline.png", dpi=150, bbox_inches="tight")

log.info("Saved composed-prototype comparison to %s", OUTPUT_DIR / "composed_vs_baseline.png")

# %% [markdown]
# ## Baseline vs. best-combo heatmaps
#
# Every result above is an IoU number. This section turns the single best result per scale
# — whichever (family, severity) combo scored highest, found across every family, not one
# at a time — back into a picture: the raw cosine-similarity heatmap itself, baseline vs.
# best, with the query's own GT mask outlined on both. Both scales always have a
# well-defined baseline and best entry, so the comparison applies to each one tested here.
# See "Reading the results" at the end of this file for how to tell a heatmap-level
# improvement apart from an IoU-level one.

# %% Visualization — baseline vs. best (family, severity) combo heatmaps, one row per scale
query_gt_disp = (
    np.array(
        Image.fromarray(query_pixel_mask.astype(np.uint8) * 255).resize(
            query_img.size, Image.NEAREST
        )
    )
    > 0
)

fig, axes = plt.subplots(len(SCALES), 3, figsize=(15, 5 * len(SCALES)))
if len(SCALES) == 1:
    axes = axes.reshape(1, 3)

first_family = next(iter(AUGMENTATIONS))
for row, scale in enumerate(SCALES):
    scale_entries = [e for e in entries if e["scale"] == scale]
    baseline_entry = next(
        e
        for e in scale_entries
        if e["family"] == first_family and e["value"] == AUGMENTATIONS[first_family]["values"][0]
    )
    best_entry = max(scale_entries, key=lambda e: e["oracle_iou"])
    best_unit = AUGMENTATIONS[best_entry["family"]]["unit"]
    delta = best_entry["oracle_iou"] - baseline_entry["oracle_iou"]

    axes[row, 0].imshow(query_img)
    axes[row, 0].imshow(query_gt_disp, cmap="Reds", alpha=0.35)
    axes[row, 0].set_title(f"scale={scale}  query image + GT mask", fontsize=10)
    axes[row, 0].axis("off")

    im1 = axes[row, 1].imshow(baseline_entry["raw"], cmap="jet", aspect="auto")
    axes[row, 1].contour(gt_patch_mask.astype(float), levels=[0.5], colors="lime", linewidths=1.2)
    axes[row, 1].set_title(
        f"baseline (no aug) — oracle_iou={baseline_entry['oracle_iou']:.3f}", fontsize=10
    )
    axes[row, 1].axis("off")
    plt.colorbar(im1, ax=axes[row, 1], shrink=0.75, pad=0.02)

    im2 = axes[row, 2].imshow(best_entry["raw"], cmap="jet", aspect="auto")
    axes[row, 2].contour(gt_patch_mask.astype(float), levels=[0.5], colors="lime", linewidths=1.2)
    axes[row, 2].set_title(
        f"best: {best_entry['family']} value={best_entry['value']} {best_unit} — "
        f"oracle_iou={best_entry['oracle_iou']:.3f} ({delta:+.3f} vs. baseline)",
        fontsize=10,
    )
    axes[row, 2].axis("off")
    plt.colorbar(im2, ax=axes[row, 2], shrink=0.75, pad=0.02)

fig.suptitle(f"Baseline vs. best augmentation+severity heatmap — {REF_STEM} -> {QUERY_STEM}")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUTPUT_DIR / "baseline_vs_best_heatmap.png", dpi=150, bbox_inches="tight")

log.info(
    "Saved baseline-vs-best heatmap comparison to %s", OUTPUT_DIR / "baseline_vs_best_heatmap.png"
)

# %% [markdown]
# ## Augment the query instead of the prototype
#
# Every experiment above perturbs the *reference* crop and scores a clean, fixed query —
# convenient for isolating "what does augmenting the exemplar do to matching", but backwards
# from the common real-world case: a clean reference exemplar captured once under good
# conditions, scored against noisy factory-floor frames (bad lighting, motion blur, a part
# rotated in-frame, JPEG artifacts off a compressed camera feed). This section flips the
# roles: the prototype is held clean — the plain, unaugmented "close"/"mid" prototype
# already built for the severity-sweep baseline above — and the *query image* is perturbed
# instead, using the same augmentation families/severities/apply functions as everywhere
# else in this file (so rotation reprojects the query's own GT mask, exactly as it does for
# the reference crop). The augmented query doesn't depend on which prototype scale scores
# it, so each one is built and encoded once and then scored against *both* scales' clean
# prototypes — half the encoder work the crop-augmentation experiment needed for the same
# number of (scale, family, severity) combos.

# %% Build every (family, severity) augmented query image up front — scale-independent,
# since the query image itself doesn't change per scale, only which prototype later scores
# it does.
query_fill = mean_color(query_img)
query_aug_entries: list[dict] = []
for family, spec in AUGMENTATIONS.items():
    for val in spec["values"]:
        aug_img, aug_pixel_mask = spec["apply"](query_img, query_pixel_mask, val, query_fill)
        query_aug_entries.append(
            {"family": family, "value": val, "img": aug_img, "pixel_mask": aug_pixel_mask}
        )

log.info(
    "Built %d augmented query images across %d families",
    len(query_aug_entries),
    len(AUGMENTATIONS),
)

# %% Encode every augmented query image in a single batched forward pass
query_aug_out = encoder([e["img"] for e in query_aug_entries], layers=[LAYER_IDX], debias=True)
query_aug_patches = query_aug_out.patches[:, 0]  # (N, grid_h, grid_w, D)

for entry, patch_tokens in zip(query_aug_entries, query_aug_patches):
    entry["tokens"] = F.normalize(patch_tokens.reshape(q_h * q_w, D), p=2, dim=-1)
    entry["gt_patch_mask"] = pixel_mask_to_patch_mask(
        entry["pixel_mask"], q_h, q_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )

# %% Score each scale's clean (unaugmented) prototype against every augmented query
clean_prototype: dict[str, torch.Tensor] = {
    scale: next(
        e["prototype"]
        for e in entries
        if e["scale"] == scale
        and e["family"] == first_family
        and e["value"] == AUGMENTATIONS[first_family]["values"][0]
    )
    for scale in SCALES
}

query_aug_results: list[dict] = []
for scale in SCALES:
    prototype = clean_prototype[scale]
    for entry in query_aug_entries:
        raw = (entry["tokens"] @ prototype.T).reshape(q_h, q_w).cpu().float().numpy()
        query_aug_results.append(
            {
                "scale": scale,
                "family": entry["family"],
                "value": entry["value"],
                "img": entry["img"],
                "raw": raw,
                "oracle_iou": oracle_iou(raw, entry["gt_patch_mask"], ORACLE_THRESHOLD_STEPS),
            }
        )

# %% Aggregate: oracle-IoU-vs-severity per scale/family, and the shared no-aug baseline
query_aug_baseline_oracle_iou: dict[str, float] = {
    scale: next(
        r["oracle_iou"]
        for r in query_aug_results
        if r["scale"] == scale
        and r["family"] == first_family
        and r["value"] == AUGMENTATIONS[first_family]["values"][0]
    )
    for scale in SCALES
}

query_aug_curves: dict[str, dict[str, list[float]]] = {scale: {} for scale in SCALES}
query_aug_best_over_baseline: list[dict] = []
for scale in SCALES:
    for family, spec in AUGMENTATIONS.items():
        values = spec["values"]
        ious = [
            next(
                r["oracle_iou"]
                for r in query_aug_results
                if r["scale"] == scale and r["family"] == family and r["value"] == v
            )
            for v in values
        ]
        query_aug_curves[scale][family] = ious
        best_idx = int(np.argmax(ious))
        query_aug_best_over_baseline.append(
            {
                "scale": scale,
                "family": family,
                "best_value": values[best_idx],
                "best_iou": ious[best_idx],
                "delta_vs_baseline": ious[best_idx] - query_aug_baseline_oracle_iou[scale],
            }
        )
        log.info(
            "[query-aug] scale=%-5s %-22s values=%s  oracle_iou=%s",
            scale,
            family,
            values,
            np.round(ious, 3),
        )

log.info(
    "[query-aug] Baseline (no augmentation) oracle IoU: %s",
    {k: round(v, 3) for k, v in query_aug_baseline_oracle_iou.items()},
)
for row in sorted(query_aug_best_over_baseline, key=lambda r: -r["delta_vs_baseline"]):
    log.info(
        "[query-aug] BEST scale=%-5s %-22s value=%-6s oracle_iou=%.3f (%+.3f vs. baseline)",
        row["scale"],
        row["family"],
        row["best_value"],
        row["best_iou"],
        row["delta_vs_baseline"],
    )

# %% Visualization — augmented query image grid, one row per family. Unlike the crop grids
# above, this is a single figure (the query doesn't vary by scale); each title shows both
# scales' oracle IoU since the same augmented query is scored against two different clean
# prototypes.
n_families = len(AUGMENTATIONS)
n_cols = max(len(spec["values"]) for spec in AUGMENTATIONS.values())
fig, axes = plt.subplots(n_families, n_cols, figsize=(2.6 * n_cols, 2.9 * n_families))
for row, (family, spec) in enumerate(AUGMENTATIONS.items()):
    for col, val in enumerate(spec["values"]):
        ax = axes[row, col]
        entry = next(e for e in query_aug_entries if e["family"] == family and e["value"] == val)
        ious_by_scale = {
            r["scale"]: r["oracle_iou"]
            for r in query_aug_results
            if r["family"] == family and r["value"] == val
        }
        iou_str = "  ".join(f"{s}={v:.3f}" for s, v in ious_by_scale.items())
        ax.imshow(entry["img"])
        ax.set_title(f"{val} {spec['unit']}\n{iou_str}", fontsize=8)
        ax.axis("off")
    for col in range(len(spec["values"]), n_cols):
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(family, fontsize=9)
fig.suptitle(
    f"Augmented query images, clean prototype — {REF_STEM} -> {QUERY_STEM} / {TARGET_CLASS!r}"
)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUTPUT_DIR / "augmented_query_grid.png", dpi=150, bbox_inches="tight")

# %% Visualization — query-augmentation oracle-IoU curves, one panel per scale
fig, axes = plt.subplots(1, len(SCALES), figsize=(7.5 * len(SCALES), 5.5), sharey=True)
for ax, scale in zip(axes, SCALES):
    for i, (family, spec) in enumerate(AUGMENTATIONS.items()):
        values = np.array(spec["values"], dtype=float)
        frac = (values - values[0]) / (values[-1] - values[0])
        ax.plot(
            frac,
            query_aug_curves[scale][family],
            marker="o",
            label=family,
            color=colors[i % len(colors)],
        )
    ax.axhline(
        query_aug_baseline_oracle_iou[scale],
        color="k",
        linestyle="--",
        linewidth=1,
        label="no-augmentation baseline",
    )
    ax.set_xlabel("severity fraction (0 = no-op, 1 = strongest tested)")
    ax.set_title(f"scale = {scale}")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("oracle IoU (clean prototype vs. augmented-query GT)")
axes[0].legend(fontsize=8)
fig.suptitle(f"Oracle-IoU vs. query augmentation severity — {REF_STEM} -> {QUERY_STEM}")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "query_aug_oracle_iou_curves.png", dpi=150, bbox_inches="tight")

# %% Visualization — best achievable query-augmentation oracle IoU per family vs. baseline
fig, ax = plt.subplots(figsize=(9, 5))
for i, scale in enumerate(SCALES):
    bests = [
        next(
            r["best_iou"]
            for r in query_aug_best_over_baseline
            if r["scale"] == scale and r["family"] == f
        )
        for f in family_names
    ]
    ax.bar(
        x + i * width,
        bests,
        width=width,
        label=f"{scale} (best over severities)",
        color=SCALE_COLOR[scale],
    )
    ax.axhline(
        query_aug_baseline_oracle_iou[scale],
        color=SCALE_COLOR[scale],
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )
ax.set_xticks(x + width * (len(SCALES) - 1) / 2, family_names, rotation=30, ha="right")
ax.set_ylabel("oracle IoU")
ax.set_title(
    "Best achievable query-augmentation oracle IoU per family (dashed = that scale's baseline)"
)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "query_aug_best_vs_baseline.png", dpi=150, bbox_inches="tight")

log.info("Saved query-augmentation figures to %s", OUTPUT_DIR)

# %% [markdown]
# ## Reading the results
#
# A family/scale combination whose curve rises *above* the dashed no-augmentation
# baseline is genuinely useful signal: perturbing that exemplar crop before pooling it
# into a prototype made the resulting heatmap *more* separable on an unseen query image,
# not less — evidence that some augmentation of the reference crop generalizes the same
# way augmenting training data usually does. A family that only ever drops below baseline
# says the opposite for this backbone/layer/object: perturbing the exemplar just moves the
# prototype off the real object's manifold, and any resemblance to helpful "invariance"
# is coincidental. Comparing `close` against `mid` also separates two different
# failure/benefit modes — `close` prototypes are built from mostly-object pixels, so
# augmentation there mostly perturbs texture/appearance, while `mid` prototypes include a
# lot of the object's own surrounding context, so augmentation there also perturbs how
# much of that context factors into the masked-mean.
#
# The composed-prototype ensemble (severity-averaged, every value for a family pooled into
# one prototype) is a different bet than the single-severity sweep above: it can never land
# on the *single* highest-scoring perturbation, but it also can't overfit to one lucky
# severity draw. If `composed_oracle_iou` beats `original_oracle_iou` but trails
# `best_single_severity_oracle_iou`, averaging is buying back some of the single-best
# severity's benefit without having to search for or commit to it. If it beats *both*, the
# ensemble is smoothing out per-severity noise that the oracle threshold search was
# otherwise exploiting. If it's worse than the plain original, this family's severities pull
# the prototype in inconsistent directions and averaging just cancels out whatever any one
# of them offered. The cumulative curves (k=1..N severities composed) show *how* it gets
# there: a curve that climbs steadily says the ensemble effect compounds as views are
# added; a curve that jumps on one k and flattens says a single severity is doing all the
# work and the rest are just dilution; a curve that's flat throughout says composing
# doesn't move this family's prototype much either way.
#
# The baseline-vs-best heatmap panels turn the IoU number back into a picture: a `jet`
# heatmap that visibly tightens around the green GT contour (less hot signal on background,
# more contiguous hot signal on the object) is a heatmap-level explanation for a positive
# delta. A heatmap that looks much like the baseline despite a nonzero IoU delta is a sign
# the gain came from the oracle threshold search finding a marginally better cut point, not
# from a genuinely more separable score map — worth treating with more skepticism.
#
# The query-augmentation results answer a different question than the crop-augmentation
# ones above: not "does perturbing the exemplar help matching" but "how much does the clean
# prototype's own oracle IoU degrade as the *query* gets noisier" — the realistic deployment
# failure mode (a clean exemplar captured once under good conditions, scored against
# whatever a factory-floor camera hands it). Comparing the two sensitivity rankings
# family-by-family is the point: a family whose query-augmentation curve drops sharply while
# its crop-augmentation curve stayed flat (or vice versa) says this backbone/layer is more
# sensitive to that perturbation on one side of the match than the other — worth knowing
# when only one side of a real pipeline is controllable.
#
# ## Other augmented-prototype experiments worth running in this dir
#
# - **Multiple query images** — this file uses one ref/query pair; repeating across every
#   abc3 part type (like `multiscale_crop_ablation.py`'s `RUN_PART_TYPES` loop) would show
#   whether a helpful augmentation family here is a general effect or specific to this
#   object/lighting pair.
# - **Real threshold instead of oracle** — pair this with `iou_tuned_threshold`'s
#   fit-on-ref pattern (tune the threshold on the *augmented* ref crop's own mask, not the
#   query's GT) to see how much of any oracle-IoU gain survives once the threshold has to
#   be picked without seeing the query's ground truth.

# %%
