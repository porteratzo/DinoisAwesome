# %% [markdown]
# # Fundamental: Scale — How Cropping Closer Changes DINOv3 Patch Embeddings
#
# First experiment in `experiments/fundamental/` — a series meant to probe DINOv3's
# *basic* representational properties (scale, rotation, lighting, occlusion, ...) one
# axis at a time, independent of any downstream task (unlike `object_detection/`, which
# is all about detection quality).
#
# This one asks a single question: as we crop progressively closer to one object
# instance — same pixels of that object, shrinking field of view around it — how much
# does its patch-embedding representation actually move in DINOv3 feature space? A
# training-free exemplar/detection pipeline (like `object_detection/multiscale_crop_
# ablation.py`) implicitly assumes "close" and "global" crops of the same object encode
# to *similar* vectors. This experiment measures that assumption directly instead of
# taking it on faith.
#
# Steps:
#   1. Load one abc3 image + the mask of a single annotated instance.
#   2. Build N progressively tighter crops, interpolating from the whole image (t=0)
#      down to a tight, padded bbox around the instance (t=1).
#   3. Re-encode each crop at native resolution (so "closer" really means more pixels
#      on the object, not the same tokens subset from one global pass).
#   4. At each scale, pool the object's own patch tokens (mask projected into that
#      crop's own grid) into one masked-mean embedding.
#   5. Compare embeddings across scales: pairwise cosine-similarity matrix, drift
#      curves relative to the global and closest views, and the same analysis for the
#      whole-crop CLS token as a reference point.
#   6. Visualize the crop sequence (with the instance bbox overlaid) and the
#      similarity results.
#
# See the bottom markdown cell for other "fundamental" experiments this style of setup
# would suit well.

# %% Logging — must be before torch import
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("scale_crop_similarity")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from matplotlib.patches import Rectangle
from PIL import Image

from dinoisawesome import DinoEncoder, compute_exemplar_features, load_annotations

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

data_dir = _REPO_ROOT / "data" / "abc3"

# "donut foam single" has exactly one instance in every abc3 image and a small bbox
# (~100x90px in a 3840x2160 frame) relative to the full image — a good, dramatic case
# for a scale study. Swap IMAGE_STEM / TARGET_CLASS to try others (see classes.json).
IMAGE_STEM = "LHa_1"
TARGET_CLASS = "donut foam single"
INSTANCE_INDEX = 0  # which matching instance, if TARGET_CLASS has more than one

DINO_VERSION = "v3"
DINO_SIZE = "large"
IMG_SIZE = 1024  # must be divisible by patch_size (16 for v3)
LAYER_IDX = 23  # penultimate/last block of ViT-L/16 (depth 24)
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")

MASK_PATCH_THRESHOLD = 0.3  # patch-grid cell counts as "object" once this fraction is masked

N_SCALES = 7  # crop steps, t = 0 (global) .. 1 (closest), evenly spaced
CLOSE_PADDING_FRACTION = 0.5  # closest crop's padding around the mask bbox, fraction of its extent
MIN_CROP_SIZE = 64  # closest crop must be at least this many native px on each side

OUTPUT_DIR = _REPO_ROOT / "outputs" / "fundamental" / "scale_crop_similarity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
torch.manual_seed(SEED)

log.info(
    "image=%s class=%r  |  DINO%s-%s img_size=%d layer=%d  |  n_scales=%d",
    IMAGE_STEM,
    TARGET_CLASS,
    DINO_VERSION,
    DINO_SIZE,
    IMG_SIZE,
    LAYER_IDX,
    N_SCALES,
)

# %% Mask / geometry helpers


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


def scale_crop_boxes(
    pixel_mask: np.ndarray, n_scales: int, padding_frac: float, min_crop_size: int
) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    """PIL-style crop boxes (x0, y0, x1, y1) linearly interpolated from the whole image
    (t=0) to a tight, padded bbox around *pixel_mask* (t=1)."""
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
    if close_box[2] - close_box[0] < min_crop_size or close_box[3] - close_box[1] < min_crop_size:
        raise ValueError(
            f"Closest crop {close_box} is below MIN_CROP_SIZE={min_crop_size}px — "
            "reduce N_SCALES's padding or pick a larger instance."
        )
    global_box = (0, 0, W, H)

    t_values = np.linspace(0.0, 1.0, n_scales)
    boxes = [
        tuple(int(round(a + (b - a) * t)) for a, b in zip(global_box, close_box)) for t in t_values
    ]
    return boxes, t_values


# %% Load image + single instance mask, build the crop sequence
anns = load_annotations(data_dir / "annotations" / IMAGE_STEM)
instances = [a for a in anns if a["class"] == TARGET_CLASS]
if not instances:
    raise ValueError(f"No instances of class {TARGET_CLASS!r} in {IMAGE_STEM}")
instance = instances[INSTANCE_INDEX]
mask = instance["mask"]  # (H, W) bool, full native resolution

ref_img = Image.open(data_dir / f"{IMAGE_STEM}.jpg").convert("RGB")
boxes, t_values = scale_crop_boxes(mask, N_SCALES, CLOSE_PADDING_FRACTION, MIN_CROP_SIZE)
crops = [ref_img.crop(box) for box in boxes]

log.info(
    "Crop sizes (native px): %s",
    [(box[2] - box[0], box[3] - box[1]) for box in boxes],
)

# %% Encode every scale's crop in a single batched forward pass
encoder = DinoEncoder(
    version=DINO_VERSION,
    size=DINO_SIZE,
    img_size=IMG_SIZE,
    layers=[LAYER_IDX],
    weights_dir=DINO_WEIGHTS_DIR,
    amp=True,
)
out = encoder(crops, layers=[LAYER_IDX], debias=True)
patches = out.patches[:, 0]  # (N_SCALES, grid_h, grid_w, D)
cls = out.cls[:, 0]  # (N_SCALES, D)
_, grid_h, grid_w, D = patches.shape

# %% Per-scale masked-mean object embedding + how many patches the mask covers
object_embeddings = []
n_masked_patches = []
for i, box in enumerate(boxes):
    x0, y0, x1, y1 = box
    crop_mask_px = mask[y0:y1, x0:x1]
    patch_mask = pixel_mask_to_patch_mask(
        crop_mask_px, grid_h, grid_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )
    tokens = F.normalize(patches[i].reshape(grid_h * grid_w, D), p=2, dim=-1)
    patch_flat = torch.from_numpy(patch_mask.reshape(-1)).to(tokens.device)

    masked = tokens[patch_flat]
    if masked.shape[0] == 0:
        log.warning(
            "t=%.2f: mask empty after patch-grid projection — using all crop patches", t_values[i]
        )
        masked = tokens
    object_embeddings.append(compute_exemplar_features(masked, mode="mean"))  # (1, D)
    n_masked_patches.append(int(patch_flat.sum()))

object_embeddings = torch.cat(object_embeddings, dim=0)  # (N_SCALES, D)
cls_embeddings = F.normalize(cls, p=2, dim=-1)  # (N_SCALES, D)

log.info("Masked patch count per scale: %s", n_masked_patches)

# %% Similarity: pairwise matrices + drift curves relative to the global/closest views
sim_matrix = (object_embeddings @ object_embeddings.T).cpu().float().numpy()  # (N_SCALES, N_SCALES)
cls_sim_matrix = (cls_embeddings @ cls_embeddings.T).cpu().float().numpy()

sim_to_global = sim_matrix[0]  # each scale's masked-object embedding vs. t=0 (whole image)
sim_to_closest = sim_matrix[-1]  # ... vs. t=1 (tightest crop)
cls_sim_to_global = cls_sim_matrix[0]  # whole-crop CLS token, same comparison

log.info("Object-embedding similarity to global (t=0): %s", np.round(sim_to_global, 3))
log.info("Object-embedding similarity to closest (t=1): %s", np.round(sim_to_closest, 3))

# %% Visualization — crop sequence with the instance bbox overlaid
rmin, rmax, cmin, cmax = mask_bbox_px(mask)
fig, axes = plt.subplots(1, N_SCALES, figsize=(3.0 * N_SCALES, 3.6))
for ax, crop, box, t, n_patches in zip(axes, crops, boxes, t_values, n_masked_patches):
    x0, y0, x1, y1 = box
    ax.imshow(crop)
    ax.add_patch(
        Rectangle(
            (cmin - x0, rmin - y0),
            cmax - cmin,
            rmax - rmin,
            fill=False,
            edgecolor="#e74c3c",
            linewidth=1.5,
        )
    )
    ax.set_title(f"t={t:.2f}\n{x1 - x0}x{y1 - y0}px\n{n_patches} masked patches", fontsize=9)
    ax.axis("off")
fig.suptitle(f"Progressive crop — {IMAGE_STEM} / {TARGET_CLASS!r} (red = instance bbox)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "crop_sequence.png", dpi=150, bbox_inches="tight")

# %% Visualization — similarity heatmap + drift curves
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

im = axes[0].imshow(sim_matrix, cmap="viridis", vmin=float(sim_matrix.min()), vmax=1.0)
tick_labels = [f"{t:.2f}" for t in t_values]
axes[0].set_xticks(range(N_SCALES), tick_labels, rotation=45)
axes[0].set_yticks(range(N_SCALES), tick_labels)
axes[0].set_xlabel("crop tightness t")
axes[0].set_ylabel("crop tightness t")
axes[0].set_title("Masked-object embedding\ncosine similarity (scale x scale)")
plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

axes[1].plot(
    t_values, sim_to_global, marker="o", label="object emb. vs. t=0 (global)", color="#2ecc71"
)
axes[1].plot(
    t_values, sim_to_closest, marker="o", label="object emb. vs. t=1 (closest)", color="#e74c3c"
)
axes[1].plot(
    t_values,
    cls_sim_to_global,
    marker="s",
    label="whole-crop CLS vs. t=0",
    color="#3498db",
    linestyle="--",
)
axes[1].set_xlabel("crop tightness t (0 = global, 1 = closest)")
axes[1].set_ylabel("cosine similarity")
axes[1].set_ylim(min(0.0, float(sim_matrix.min()) - 0.05), 1.02)
axes[1].legend(fontsize=8)
axes[1].set_title("Embedding drift vs. crop tightness")
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "similarity.png", dpi=150, bbox_inches="tight")

log.info("Saved figures to %s", OUTPUT_DIR)

# %% [markdown]
# ## Aspect ratio — crop-conserving vs. resize-distorting
#
# `DinoEncoder.preprocess` always does a non-isotropic `Resize((img_size, img_size))`:
# any crop box that isn't already square gets squashed/stretched to fit. The t-sweep
# above doesn't control for this — the box's own aspect ratio drifts as t changes, so
# "zoom level" and "aspect distortion" are tangled together in that data.
#
# This section isolates aspect ratio on its own: starting from the closest crop's own
# box (`boxes[-1]`, same centre and area), build a series of boxes at fixed area but
# increasing width:height ratio, and for each one make two variants:
#   - **distort** — crop the elongated box as-is; the encoder's resize squashes it to
#     square (today's default behaviour, same as every crop above).
#   - **conserve** — letterbox-pad the same box to a square canvas (centred, filled
#     with the crop's own mean colour) *before* resizing, so the resize is isotropic
#     and the object's true shape is preserved (at the cost of a smaller in-frame
#     object / more padding as the ratio grows).
# At aspect ratio 1.0 the two variants are pixel-identical (no padding needed), which
# doubles as a sanity check on the pipeline below.

# %% Aspect-ratio parameters
ASPECT_RATIOS: list[float] = [1.0, 1.5, 2.0, 3.0, 5.0]  # width:height, area held constant

# %% Aspect-ratio helpers


def aspect_ratio_boxes(
    base_box: tuple[int, int, int, int], ratios: list[float], img_w: int, img_h: int
) -> list[tuple[int, int, int, int]]:
    """Boxes of area equal to *base_box*, centred the same, at each width:height in *ratios*."""
    x0, y0, x1, y1 = base_box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    area = (x1 - x0) * (y1 - y0)
    boxes = []
    for ratio in ratios:
        w = (area * ratio) ** 0.5
        h = (area / ratio) ** 0.5
        bx0, bx1 = cx - w / 2, cx + w / 2
        by0, by1 = cy - h / 2, cy + h / 2
        if bx0 < 0 or by0 < 0 or bx1 > img_w or by1 > img_h:
            log.warning(
                "aspect ratio %.2f: box (%.0f,%.0f,%.0f,%.0f) exceeds image bounds — clipping",
                ratio,
                bx0,
                by0,
                bx1,
                by1,
            )
        boxes.append(
            (
                max(0, int(round(bx0))),
                max(0, int(round(by0))),
                min(img_w, int(round(bx1))),
                min(img_h, int(round(by1))),
            )
        )
    return boxes


def letterbox_pad(img: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
    """Pad *img* to a square canvas (centred, filled with its own mean colour).

    Returns the padded image and the (left, top) offset the original content was
    placed at, so a matching pixel mask can be padded the same way.
    """
    w, h = img.size
    side = max(w, h)
    fill = tuple(int(c) for c in np.array(img).reshape(-1, 3).mean(axis=0))
    canvas = Image.new("RGB", (side, side), fill)
    left, top = (side - w) // 2, (side - h) // 2
    canvas.paste(img, (left, top))
    return canvas, (left, top)


def pad_pixel_mask(crop_mask_px: np.ndarray, left: int, top: int, side: int) -> np.ndarray:
    """Place *crop_mask_px* into a (side, side) canvas at (left, top); rest is False."""
    canvas_mask = np.zeros((side, side), dtype=bool)
    h, w = crop_mask_px.shape
    canvas_mask[top : top + h, left : left + w] = crop_mask_px
    return canvas_mask


# %% Build distort/conserve crop pairs and encode them in one batch
img_w, img_h = ref_img.size
ar_boxes = aspect_ratio_boxes(boxes[-1], ASPECT_RATIOS, img_w, img_h)

ar_pending: list[dict] = []
for ratio, box in zip(ASPECT_RATIOS, ar_boxes):
    x0, y0, x1, y1 = box
    raw_crop = ref_img.crop(box)
    crop_mask_px = mask[y0:y1, x0:x1]

    padded_crop, (left, top) = letterbox_pad(raw_crop)
    padded_mask_px = pad_pixel_mask(crop_mask_px, left, top, padded_crop.size[0])

    ar_pending.append(
        {"ratio": ratio, "variant": "distort", "img": raw_crop, "mask_px": crop_mask_px}
    )
    ar_pending.append(
        {"ratio": ratio, "variant": "conserve", "img": padded_crop, "mask_px": padded_mask_px}
    )

ar_out = encoder([p["img"] for p in ar_pending], layers=[LAYER_IDX], debias=True)
ar_patches = ar_out.patches[:, 0]  # (2*N_AR, ar_grid_h, ar_grid_w, D)
_, ar_grid_h, ar_grid_w, _ = ar_patches.shape

# %% Aspect-ratio masked-object embeddings + distort-vs-conserve similarity
ar_embeddings: dict[tuple[float, str], torch.Tensor] = {}
ar_masked_patches: dict[tuple[float, str], int] = {}
for entry, patch_tokens in zip(ar_pending, ar_patches):
    patch_mask = pixel_mask_to_patch_mask(
        entry["mask_px"], ar_grid_h, ar_grid_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )
    tokens = F.normalize(patch_tokens.reshape(ar_grid_h * ar_grid_w, D), p=2, dim=-1)
    patch_flat = torch.from_numpy(patch_mask.reshape(-1)).to(tokens.device)

    masked = tokens[patch_flat]
    if masked.shape[0] == 0:
        log.warning(
            "ratio=%.2f variant=%s: mask empty after projection — using all crop patches",
            entry["ratio"],
            entry["variant"],
        )
        masked = tokens
    key = (entry["ratio"], entry["variant"])
    ar_embeddings[key] = compute_exemplar_features(masked, mode="mean")  # (1, D)
    ar_masked_patches[key] = int(patch_flat.sum())

sim_distort_vs_conserve = np.array(
    [
        float((ar_embeddings[(r, "distort")] @ ar_embeddings[(r, "conserve")].T).item())
        for r in ASPECT_RATIOS
    ]
)
distort_anchor = ar_embeddings[(1.0, "distort")]  # == conserve anchor at ratio 1.0 (no padding)
sim_distort_vs_anchor = np.array(
    [float((ar_embeddings[(r, "distort")] @ distort_anchor.T).item()) for r in ASPECT_RATIOS]
)
sim_conserve_vs_anchor = np.array(
    [float((ar_embeddings[(r, "conserve")] @ distort_anchor.T).item()) for r in ASPECT_RATIOS]
)

log.info("Aspect ratios: %s", ASPECT_RATIOS)
log.info("distort vs. conserve similarity, per ratio: %s", np.round(sim_distort_vs_conserve, 3))
log.info("distort vs. ratio=1.0 anchor, per ratio:    %s", np.round(sim_distort_vs_anchor, 3))
log.info("conserve vs. ratio=1.0 anchor, per ratio:   %s", np.round(sim_conserve_vs_anchor, 3))

# %% Visualization — distort vs. conserve crops, one column per aspect ratio
fig, axes = plt.subplots(2, len(ASPECT_RATIOS), figsize=(3.0 * len(ASPECT_RATIOS), 6.4))
for col, (ratio, box) in enumerate(zip(ASPECT_RATIOS, ar_boxes)):
    distort_entry = next(p for p in ar_pending if p["ratio"] == ratio and p["variant"] == "distort")
    conserve_entry = next(
        p for p in ar_pending if p["ratio"] == ratio and p["variant"] == "conserve"
    )
    axes[0, col].imshow(distort_entry["img"])
    axes[0, col].set_title(
        f"ratio={ratio:.1f} — distort\n{ar_masked_patches[(ratio, 'distort')]} masked patches",
        fontsize=9,
    )
    axes[0, col].axis("off")

    axes[1, col].imshow(conserve_entry["img"])
    axes[1, col].set_title(
        f"ratio={ratio:.1f} — conserve\n{ar_masked_patches[(ratio, 'conserve')]} masked patches",
        fontsize=9,
    )
    axes[1, col].axis("off")
fig.suptitle(
    f"Aspect-ratio distortion — {IMAGE_STEM} / {TARGET_CLASS!r}\n"
    "(top: squashed by resize, bottom: letterbox-padded)"
)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.subplots_adjust(hspace=0.5)
fig.savefig(OUTPUT_DIR / "aspect_ratio_crops.png", dpi=150, bbox_inches="tight")

# %% Visualization — aspect-ratio distortion sensitivity curves
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(
    ASPECT_RATIOS,
    sim_distort_vs_conserve,
    marker="o",
    label="distort vs. conserve (same ratio)",
    color="#9b59b6",
)
ax.plot(
    ASPECT_RATIOS, sim_distort_vs_anchor, marker="o", label="distort vs. ratio=1.0", color="#e74c3c"
)
ax.plot(
    ASPECT_RATIOS,
    sim_conserve_vs_anchor,
    marker="o",
    label="conserve vs. ratio=1.0",
    color="#2ecc71",
)
ax.set_xlabel("width:height ratio (area held constant)")
ax.set_ylabel("cosine similarity")
ax.set_ylim(min(0.0, float(sim_distort_vs_conserve.min()) - 0.05), 1.02)
ax.legend(fontsize=8)
ax.set_title("Aspect-ratio distortion sensitivity")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "aspect_ratio_similarity.png", dpi=150, bbox_inches="tight")

log.info("Saved aspect-ratio figures to %s", OUTPUT_DIR)

# %% [markdown]
# ## Other "fundamental" scale/robustness experiments worth running in this dir
#
# - **Crop vs. pure zoom** — repeat the same t-sweep but instead of cropping-then-
#   resizing, just resize (upsample) the *whole* image so the object subtends the same
#   angle, with no crop. Separates "more native pixels sampled" from "object appears
#   bigger in the resized input" — two different reasons embeddings could drift.
# - **Layer-wise scale sensitivity** — repeat the whole sweep across several
#   `LAYER_IDX` values (early / mid / late blocks). Hypothesis: early blocks are closer
#   to raw texture (scale-sensitive), late blocks are closer to semantics (more
#   scale-stable) — worth checking against this specific backbone/dataset instead of
#   assuming it.
# - **Absolute vs. relative drift budget** — instead of only comparing to t=0 and t=1,
#   fit how much cosine similarity is "used up" per unit of bbox-area change, to get a
#   single scalar sensitivity number per layer/model size, comparable across objects
#   and images.
# - **Model-size sweep** — same t-sweep for small/base/large/giant DINOv3 backbones on
#   the same crops, to see whether scale sensitivity is a capacity artifact or
#   architecture-inherent (patch size is fixed across sizes, so this isolates capacity).
# - **Rotation / in-plane orientation** — same masked-mean-embedding-drift protocol,
#   but sweeping rotation angle around the instance centroid instead of crop tightness.
# - **Lighting / exposure** — sweep brightness/contrast/gamma on the same tight crop
#   and measure drift the same way; relevant here since abc3 is real factory-floor
#   imagery with uncontrolled lighting.
# - **Occlusion** — progressively mask out a growing fraction of the instance's own
#   patches (independent of the crop box) and see how fast the masked-mean embedding
#   degrades — a proxy for how much partial occlusion a prototype-matching pipeline can
#   tolerate before the exemplar becomes unrepresentative.
# - **Register/artifact tokens** — DINOv3 background patches are known to carry
#   high-norm "register" artifacts; plotting per-patch token norm (not just cosine
#   similarity) across scales could show whether the object's own patches are exempt or
#   also affected as the crop tightens and background patches disappear.

# %%
