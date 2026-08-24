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
#   1. Load every abc3 image and every annotated instance of every class in it (not
#      just one image/instance) — more instances and more part types means the drift
#      curves reflect the dataset instead of one lucky/unlucky crop.
#   2. Build one fixed "mid" crop per instance — padded around its bbox, partway
#      between the tight bbox and the whole image (no scale sweep here; scale is the
#      other experiment's axis).
#   3. For each augmentation family (rotation, illumination/gamma, color jitter,
#      Gaussian blur, Gaussian noise, JPEG compression), apply a severity sweep from
#      "no-op" up to a visibly strong perturbation, on every instance's crop.
#   4. Re-encode every augmented crop, pool the object's own patch tokens (mask
#      projected into that crop's grid — reprojected per-rotation, since rotation is
#      the only family that moves the object within the frame) into one masked-mean
#      embedding per crop.
#   5. Compare each severity level's embedding to that instance's own unperturbed
#      (severity=0) embedding via cosine similarity, then aggregate across all
#      instances per family/severity (mean ± std band) and plot drift curves for all
#      six families on one axis (x-axis normalized to a 0..1 "severity fraction" per
#      family so they're comparable despite different native units).
#   6. Visualize the augmented crop grid for one representative instance (same
#      image/class as `scale_crop_similarity.py`, for comparability) alongside the
#      aggregated drift plot.

# %% Logging — must be before torch import
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("augmentation_sensitivity")

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from dinoisawesome import DinoEncoder, compute_exemplar_features, load_annotations

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared.augmentations import (  # noqa: E402
    apply_blur,
    apply_color_jitter,
    apply_gamma,
    apply_jpeg,
    apply_noise,
    apply_rotation,
    mean_color,
    pixel_only,
)
from _shared.mask_geometry import mask_bbox_px, pixel_mask_to_patch_mask  # noqa: E402

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

data_dir = _REPO_ROOT / "data" / "abc3"

# Every abc3 image, every annotated instance of every class in it — not just one
# image/instance. IMAGE_STEMS is discovered from disk so a new abc3 capture is picked
# up automatically.
IMAGE_STEMS = sorted(p.stem for p in data_dir.glob("*.jpg"))

# Reference instance used only for the augmented-crop-grid visualization (a full grid
# across all instances would be unreadable) — same object as scale_crop_similarity.py,
# for comparability. Drift curves aggregate every instance, not just this one.
REFERENCE_IMAGE_STEM = "LHa_1"
REFERENCE_TARGET_CLASS = "donut foam single"
REFERENCE_INSTANCE_ID = 1  # annotation "instance_id" (1-based, per class per image)

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
    "images=%s  |  DINO%s-%s img_size=%d layer=%d",
    IMAGE_STEMS,
    DINO_VERSION,
    DINO_SIZE,
    IMG_SIZE,
    LAYER_IDX,
)

# %% Mask / geometry helpers (same as scale_crop_similarity.py)


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


# %% Load every abc3 image + every annotated instance, build each one's fixed mid crop
instances: list[dict] = []
for image_stem in tqdm(IMAGE_STEMS, desc="Loading images/annotations"):
    anns = load_annotations(data_dir / "annotations" / image_stem)
    ref_img = Image.open(data_dir / f"{image_stem}.jpg").convert("RGB")
    for ann in anns:
        mask = ann["mask"]  # (H, W) bool, full native resolution
        mid_box = mid_bbox_crop(mask, MID_PADDING_FRACTION)
        x0, y0, x1, y1 = mid_box
        base_crop = ref_img.crop(mid_box)
        base_mask_px = mask[y0:y1, x0:x1]
        instances.append(
            {
                "image_stem": image_stem,
                "class": ann["class"],
                "instance_id": ann["instance_id"],
                "base_crop": base_crop,
                "base_mask_px": base_mask_px,
                "fill": mean_color(base_crop),
            }
        )

reference_instance = next(
    (
        inst
        for inst in instances
        if inst["image_stem"] == REFERENCE_IMAGE_STEM
        and inst["class"] == REFERENCE_TARGET_CLASS
        and inst["instance_id"] == REFERENCE_INSTANCE_ID
    ),
    None,
)
if reference_instance is None:
    raise ValueError(
        f"Reference instance image={REFERENCE_IMAGE_STEM!r} class={REFERENCE_TARGET_CLASS!r} "
        f"instance_id={REFERENCE_INSTANCE_ID} not found among loaded instances"
    )

log.info("Loaded %d instances across %d images", len(instances), len(IMAGE_STEMS))

# %% Augmentation families
#
# Each family is a severity sweep starting at a literal no-op value (angle=0,
# gamma=1.0, jitter magnitude=0, ...) so severity=0 is pixel-identical to *base_crop*
# — a built-in sanity check that every family's drift curve starts at similarity 1.0.
# All families are pixel-only (mask unchanged) except rotation, which moves the object
# within the frame and rotates the mask by the same angle.

AUGMENTATIONS: dict[str, dict] = {
    "rotation": {
        "values": [0, 8, 16, 30, 50, 75],
        "unit": "deg",
        "apply": apply_rotation,
    },
    "illumination (gamma)": {
        "values": [1.0, 1.3, 1.7, 2.2, 2.8, 3.5],
        "unit": "gamma",
        "apply": pixel_only(apply_gamma),
    },
    "color jitter": {
        "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "unit": "magnitude",
        "apply": pixel_only(partial(apply_color_jitter, seed=SEED)),
    },
    "gaussian blur": {
        "values": [0, 1, 2, 4, 7, 11],
        "unit": "px radius",
        "apply": pixel_only(apply_blur),
    },
    "gaussian noise": {
        "values": [0, 8, 16, 28, 45, 70],
        "unit": "sigma (0-255)",
        "apply": pixel_only(partial(apply_noise, seed=SEED)),
    },
    "jpeg compression": {
        "values": [100, 80, 60, 35, 15, 3],
        "unit": "quality",
        "apply": pixel_only(apply_jpeg),
    },
}

# %% Build every augmented crop up front (severity=0 == that instance's own base_crop)
entries: list[dict] = []
for inst in tqdm(instances, desc="Building augmented crops"):
    for family, spec in AUGMENTATIONS.items():
        for val in spec["values"]:
            img, mask_px = spec["apply"](inst["base_crop"], inst["base_mask_px"], val, inst["fill"])
            entries.append(
                {
                    "image_stem": inst["image_stem"],
                    "class": inst["class"],
                    "instance_id": inst["instance_id"],
                    "family": family,
                    "value": val,
                    "img": img,
                    "mask_px": mask_px,
                }
            )

log.info(
    "Built %d augmented crops across %d instances x %d families",
    len(entries),
    len(instances),
    len(AUGMENTATIONS),
)

# %% Encode every augmented crop, chunked to encoder.max_batch_size with a progress bar
encoder = DinoEncoder(
    version=DINO_VERSION,
    size=DINO_SIZE,
    img_size=IMG_SIZE,
    layers=[LAYER_IDX],
    weights_dir=DINO_WEIGHTS_DIR,
    amp=True,
)
chunk_size = encoder.max_batch_size
patch_chunks = []
for i in tqdm(range(0, len(entries), chunk_size), desc="Encoding crops"):
    chunk = entries[i : i + chunk_size]
    out = encoder([e["img"] for e in chunk], layers=[LAYER_IDX], debias=True)
    patch_chunks.append(out.patches[:, 0].cpu())  # (chunk, grid_h, grid_w, D)
patches = torch.cat(patch_chunks, dim=0)  # (N, grid_h, grid_w, D)
_, grid_h, grid_w, D = patches.shape

# %% Per-crop masked-mean object embedding
for entry, patch_tokens in tqdm(
    zip(entries, patches), total=len(entries), desc="Pooling object embeddings"
):
    patch_mask = pixel_mask_to_patch_mask(
        entry["mask_px"], grid_h, grid_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )
    tokens = F.normalize(patch_tokens.reshape(grid_h * grid_w, D), p=2, dim=-1)
    patch_flat = torch.from_numpy(patch_mask.reshape(-1)).to(tokens.device)

    masked = tokens[patch_flat]
    if masked.shape[0] == 0:
        log.warning(
            "image=%s class=%s instance=%s family=%s value=%s: mask empty after patch-grid "
            "projection — using all crop patches",
            entry["image_stem"],
            entry["class"],
            entry["instance_id"],
            entry["family"],
            entry["value"],
        )
        masked = tokens
    entry["embedding"] = compute_exemplar_features(masked, mode="mean")  # (1, D)
    entry["n_masked_patches"] = int(patch_flat.sum())

# %% Similarity vs. each instance's own severity=0 (unperturbed) embedding, then
# aggregated (mean + std) across all instances per family/severity
baseline_by_instance_family: dict[tuple, torch.Tensor] = {}
for family, spec in AUGMENTATIONS.items():
    baseline_val = spec["values"][0]
    for entry in entries:
        if entry["family"] == family and entry["value"] == baseline_val:
            key = (entry["image_stem"], entry["class"], entry["instance_id"], family)
            baseline_by_instance_family[key] = entry["embedding"]

for entry in entries:
    key = (entry["image_stem"], entry["class"], entry["instance_id"], entry["family"])
    baseline = baseline_by_instance_family[key]
    entry["similarity"] = float((entry["embedding"] @ baseline.T).item())

drift_summary: dict[str, dict[str, np.ndarray]] = {}
for family, spec in AUGMENTATIONS.items():
    mean_sim = []
    std_sim = []
    for val in spec["values"]:
        sims = np.array(
            [e["similarity"] for e in entries if e["family"] == family and e["value"] == val]
        )
        mean_sim.append(float(sims.mean()))
        std_sim.append(float(sims.std()))
    drift_summary[family] = {"mean": np.array(mean_sim), "std": np.array(std_sim)}
    log.info(
        "%-22s values=%s  mean_sim=%s  std=%s",
        family,
        spec["values"],
        np.round(mean_sim, 3),
        np.round(std_sim, 3),
    )

# %% Visualization — augmented crop grid, one row per family (reference instance only;
# a grid across all instances would be unreadable, so this is a qualitative sample —
# the drift-curve plot below is the one aggregated across every instance)
ref_entries = [
    e
    for e in entries
    if e["image_stem"] == reference_instance["image_stem"]
    and e["class"] == reference_instance["class"]
    and e["instance_id"] == reference_instance["instance_id"]
]
n_families = len(AUGMENTATIONS)
n_cols = max(len(spec["values"]) for spec in AUGMENTATIONS.values())
fig, axes = plt.subplots(n_families, n_cols, figsize=(2.6 * n_cols, 2.9 * n_families))
for row, (family, spec) in enumerate(AUGMENTATIONS.items()):
    for col, val in enumerate(spec["values"]):
        ax = axes[row, col]
        entry = next(e for e in ref_entries if e["family"] == family and e["value"] == val)
        ax.imshow(entry["img"])
        ax.set_title(f"{val} {spec['unit']}\nsim={entry['similarity']:.3f}", fontsize=8)
        ax.axis("off")
    for col in range(len(spec["values"]), n_cols):
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(family, fontsize=9)
fig.suptitle(
    f"Augmentation sweeps — {reference_instance['image_stem']} / "
    f"{reference_instance['class']!r} instance {reference_instance['instance_id']} (mid crop)"
)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUTPUT_DIR / "augmented_crops.png", dpi=150, bbox_inches="tight")

# %% Visualization — combined drift curves, aggregated across all instances
# (x normalized to 0..1 severity fraction; shaded band = ±1 std across instances)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
colors = plt.get_cmap("tab10").colors
for i, (family, spec) in enumerate(AUGMENTATIONS.items()):
    values = np.array(spec["values"], dtype=float)
    frac = (values - values[0]) / (values[-1] - values[0])
    mean_sim = drift_summary[family]["mean"]
    std_sim = drift_summary[family]["std"]
    color = colors[i % len(colors)]
    ax.plot(frac, mean_sim, marker="o", label=family, color=color)
    ax.fill_between(frac, mean_sim - std_sim, mean_sim + std_sim, color=color, alpha=0.15)
ax.set_xlabel("severity fraction (0 = no-op, 1 = strongest tested)")
ax.set_ylabel(f"cosine similarity to severity=0 (mean ± std, n={len(instances)} instances)")
lowest = min(
    float((drift_summary[f]["mean"] - drift_summary[f]["std"]).min()) for f in AUGMENTATIONS
)
ax.set_ylim(min(0.0, lowest - 0.05), 1.02)
ax.legend(fontsize=8)
ax.set_title(
    f"Masked-object embedding drift per augmentation family (n={len(instances)} instances)"
)
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
# - **Per-augmentation randomness spread** — the shaded band in `drift_curves.png` is
#   spread *across instances* at one fixed seed; color jitter and noise still use one
#   fixed seed per severity, so it says nothing about *this one draw's* variance.
#   Running several seeds per level and plotting a second band would separate "this
#   augmentation's typical effect" from "this one draw's effect."
# - **Occlusion** — progressively mask out a growing fraction of the instance's own
#   patches (independent of any pixel-level augmentation) — flagged in
#   `scale_crop_similarity.py` too, and complements this file's perturbation set.


# %%
