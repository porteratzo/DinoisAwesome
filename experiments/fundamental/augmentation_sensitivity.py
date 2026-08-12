# %% [markdown]
# # Fundamental: Augmentation — How Robust Are DINOv3 Patch Embeddings to Common
# # Image Perturbations?
#
# Second experiment in `experiments/fundamental/` (see `scale_crop_similarity.py` for
# the first). That one asked how *crop tightness* moves an object's patch embedding;
# this one holds the crop fixed and asks how much each of several common perturbations
# moves it instead — the ones `scale_crop_similarity.py`'s closing markdown flagged as
# follow-ups (rotation, lighting) plus a few more that matter for real factory-floor
# imagery (abc3 is uncontrolled shop-floor lighting, not a studio).
#
# Steps:
#   1. Load one abc3 image + the mask of a single annotated instance (same
#      image/class as `scale_crop_similarity.py`, for comparability).
#   2. Build one fixed "mid" crop — padded around the instance bbox, partway between
#      the tight bbox and the whole image (no scale sweep here; scale is the other
#      experiment's axis).
#   3. For each augmentation family (rotation, illumination/gamma, color jitter,
#      Gaussian blur, Gaussian noise, JPEG compression), apply a severity sweep from
#      "no-op" up to a visibly strong perturbation.
#   4. Re-encode every augmented crop, pool the object's own patch tokens (mask
#      projected into that crop's grid — reprojected per-rotation, since rotation is
#      the only family that moves the object within the frame) into one masked-mean
#      embedding per crop.
#   5. Compare each severity level's embedding to the unperturbed (severity=0)
#      embedding via cosine similarity, and plot drift curves for all six families on
#      one axis (x-axis normalized to a 0..1 "severity fraction" per family so they're
#      comparable despite different native units).
#   6. Visualize the augmented crop grid alongside the drift plot.

# %% Logging — must be before torch import
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("augmentation_sensitivity")

import io
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

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

data_dir = _REPO_ROOT / "data" / "abc3"

# Same object as scale_crop_similarity.py — one instance per abc3 image, small bbox
# relative to the full frame. Swap IMAGE_STEM / TARGET_CLASS to try others.
IMAGE_STEM = "LHa_1"
TARGET_CLASS = "donut foam single"
INSTANCE_INDEX = 0

DINO_VERSION = "v3"
DINO_SIZE = "large"
IMG_SIZE = 1024  # must be divisible by patch_size (16 for v3)
LAYER_IDX = 23  # penultimate/last block of ViT-L/16 (depth 24)
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")

MASK_PATCH_THRESHOLD = 0.3  # patch-grid cell counts as "object" once this fraction is masked
MID_PADDING_FRACTION = 1.0  # mid-crop padding around the mask bbox, fraction of its extent

SEED = 0
torch.manual_seed(SEED)

OUTPUT_DIR = _REPO_ROOT / "outputs" / "fundamental" / "augmentation_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log.info(
    "image=%s class=%r  |  DINO%s-%s img_size=%d layer=%d",
    IMAGE_STEM,
    TARGET_CLASS,
    DINO_VERSION,
    DINO_SIZE,
    IMG_SIZE,
    LAYER_IDX,
)

# %% Mask / geometry helpers (same as scale_crop_similarity.py)


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


def mid_bbox_crop(pixel_mask: np.ndarray, padding_frac: float) -> tuple[int, int, int, int]:
    """PIL-style box (x0, y0, x1, y1): the mask's bbox padded by *padding_frac* of its
    own extent on each side, clipped to the image."""
    H, W = pixel_mask.shape
    rmin, rmax, cmin, cmax = mask_bbox_px(pixel_mask)
    pad_r = int((rmax - rmin) * padding_frac)
    pad_c = int((cmax - cmin) * padding_frac)
    return (
        max(0, cmin - pad_c),
        max(0, rmin - pad_r),
        min(W, cmax + pad_c),
        min(H, rmax + pad_r),
    )


def mean_color(img: Image.Image) -> tuple[int, int, int]:
    return tuple(int(c) for c in np.array(img).reshape(-1, 3).mean(axis=0))


# %% Load image + single instance mask, build the fixed mid crop
anns = load_annotations(data_dir / "annotations" / IMAGE_STEM)
instances = [a for a in anns if a["class"] == TARGET_CLASS]
if not instances:
    raise ValueError(f"No instances of class {TARGET_CLASS!r} in {IMAGE_STEM}")
instance = instances[INSTANCE_INDEX]
mask = instance["mask"]  # (H, W) bool, full native resolution

ref_img = Image.open(data_dir / f"{IMAGE_STEM}.jpg").convert("RGB")
mid_box = mid_bbox_crop(mask, MID_PADDING_FRACTION)
x0, y0, x1, y1 = mid_box
base_crop = ref_img.crop(mid_box)
base_mask_px = mask[y0:y1, x0:x1]
fill = mean_color(base_crop)

log.info("Mid crop box=%s size=%dx%dpx", mid_box, x1 - x0, y1 - y0)

# %% Augmentation families
#
# Each family is a severity sweep starting at a literal no-op value (angle=0,
# gamma=1.0, jitter magnitude=0, ...) so severity=0 is pixel-identical to *base_crop*
# — a built-in sanity check that every family's drift curve starts at similarity 1.0.
# All families are pixel-only (mask unchanged) except rotation, which moves the object
# within the frame and rotates the mask by the same angle.

AugmentFn = Callable[[Image.Image, np.ndarray, float], tuple[Image.Image, np.ndarray]]


def _pixel_only(fn: Callable[[Image.Image, float], Image.Image]) -> AugmentFn:
    """Wrap a pixel-value-only augmentation to the (img, mask, val) -> (img, mask) shape."""

    def wrapped(
        img: Image.Image, mask_px: np.ndarray, val: float
    ) -> tuple[Image.Image, np.ndarray]:
        return fn(img, val), mask_px

    return wrapped


def _apply_rotation(
    img: Image.Image, mask_px: np.ndarray, angle: float
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

# %% Build every augmented crop up front (base_crop / base_mask_px for severity=0)
entries: list[dict] = []
for family, spec in AUGMENTATIONS.items():
    for val in spec["values"]:
        img, mask_px = spec["apply"](base_crop, base_mask_px, val)
        entries.append({"family": family, "value": val, "img": img, "mask_px": mask_px})

log.info("Built %d augmented crops across %d families", len(entries), len(AUGMENTATIONS))

# %% Encode every augmented crop in a single batched forward pass
encoder = DinoEncoder(
    version=DINO_VERSION,
    size=DINO_SIZE,
    img_size=IMG_SIZE,
    layers=[LAYER_IDX],
    weights_dir=DINO_WEIGHTS_DIR,
    amp=True
)
out = encoder([e["img"] for e in entries], layers=[LAYER_IDX], debias=True)
patches = out.patches[:, 0]  # (N, grid_h, grid_w, D)
_, grid_h, grid_w, D = patches.shape

# %% Per-crop masked-mean object embedding
for entry, patch_tokens in zip(entries, patches):
    patch_mask = pixel_mask_to_patch_mask(
        entry["mask_px"], grid_h, grid_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )
    tokens = F.normalize(patch_tokens.reshape(grid_h * grid_w, D), p=2, dim=-1)
    patch_flat = torch.from_numpy(patch_mask.reshape(-1)).to(tokens.device)

    masked = tokens[patch_flat]
    if masked.shape[0] == 0:
        log.warning(
            "family=%s value=%s: mask empty after patch-grid projection — using all crop patches",
            entry["family"],
            entry["value"],
        )
        masked = tokens
    entry["embedding"] = compute_exemplar_features(masked, mode="mean")  # (1, D)
    entry["n_masked_patches"] = int(patch_flat.sum())

# %% Similarity vs. each family's own severity=0 (unperturbed) embedding
baseline_by_family = {
    family: next(
        e["embedding"] for e in entries if e["family"] == family and e["value"] == spec["values"][0]
    )
    for family, spec in AUGMENTATIONS.items()
}

for entry in entries:
    baseline = baseline_by_family[entry["family"]]
    entry["similarity"] = float((entry["embedding"] @ baseline.T).item())

drift_summary = {}
for family, spec in AUGMENTATIONS.items():
    values = spec["values"]
    sims = [
        next(e["similarity"] for e in entries if e["family"] == family and e["value"] == v)
        for v in values
    ]
    drift_summary[family] = sims
    log.info("%-22s values=%s  sim=%s", family, values, np.round(sims, 3))

# %% Visualization — augmented crop grid, one row per family
n_families = len(AUGMENTATIONS)
n_cols = max(len(spec["values"]) for spec in AUGMENTATIONS.values())
fig, axes = plt.subplots(n_families, n_cols, figsize=(2.6 * n_cols, 2.9 * n_families))
for row, (family, spec) in enumerate(AUGMENTATIONS.items()):
    for col, val in enumerate(spec["values"]):
        ax = axes[row, col]
        entry = next(e for e in entries if e["family"] == family and e["value"] == val)
        ax.imshow(entry["img"])
        ax.set_title(f"{val} {spec['unit']}\nsim={entry['similarity']:.3f}", fontsize=8)
        ax.axis("off")
    for col in range(len(spec["values"]), n_cols):
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(family, fontsize=9)
fig.suptitle(f"Augmentation sweeps — {IMAGE_STEM} / {TARGET_CLASS!r} (mid crop)")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUTPUT_DIR / "augmented_crops.png", dpi=150, bbox_inches="tight")

# %% Visualization — combined drift curves (x normalized to 0..1 severity fraction)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
colors = plt.get_cmap("tab10").colors
for i, (family, spec) in enumerate(AUGMENTATIONS.items()):
    values = np.array(spec["values"], dtype=float)
    frac = (values - values[0]) / (values[-1] - values[0])
    ax.plot(frac, drift_summary[family], marker="o", label=family, color=colors[i % len(colors)])
ax.set_xlabel("severity fraction (0 = no-op, 1 = strongest tested)")
ax.set_ylabel("cosine similarity to severity=0")
ax.set_ylim(min(0.0, min(min(v) for v in drift_summary.values()) - 0.05), 1.02)
ax.legend(fontsize=8)
ax.set_title("Masked-object embedding drift per augmentation family")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "drift_curves.png", dpi=150, bbox_inches="tight")

log.info("Saved figures to %s", OUTPUT_DIR)

# %% [markdown]
# ## Reading the results
#
# The family whose curve drops fastest (lowest similarity at severity fraction 1.0) is
# the perturbation this backbone/layer is *least* invariant to — worth knowing before
# relying on masked-mean exemplars for matching under that kind of real-world
# variation (e.g. if `gaussian blur` drops fast, a slightly out-of-focus camera frame
# is a bigger risk to a prototype-matching pipeline than `jpeg compression` at typical
# stream-quality settings).
#
# ## Other augmentation/robustness experiments worth running in this dir
#
# - **Composed perturbations** — this script applies each family independently from
#   the clean crop; real frames stack several at once (blur *and* low light *and* jpeg
#   artifacts). Worth checking whether drift is roughly additive or whether some
#   combinations compound non-linearly.
# - **Layer-wise robustness** — repeat the sweep across several `LAYER_IDX` values;
#   early blocks are closer to raw texture and likely far more blur/noise-sensitive
#   than late, more semantic blocks.
# - **Per-augmentation randomness spread** — color jitter and noise here use one fixed
#   seed per severity; running several seeds per level and plotting a band (not just a
#   line) would separate "this augmentation's typical effect" from "this one draw's
#   effect."
# - **Occlusion** — progressively mask out a growing fraction of the instance's own
#   patches (independent of any pixel-level augmentation) — flagged in
#   `scale_crop_similarity.py` too, and complements this file's perturbation set.

# %%
