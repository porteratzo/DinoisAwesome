# %% [markdown]
# # Instance Detection Head
#
# Training-free instance detection via DINOv3 patch-token cosine similarity.
# Given one exemplar image and a set of query images the pipeline:
#
#   1. Extracts L2-normalised patch tokens from an intermediate transformer block.
#   2. Masks the exemplar tokens to the object region using an accompanying .npy mask.
#   3. Aggregates masked tokens into a descriptor (mean or K-means centroids).
#   4. Computes per-patch cosine similarity → density map.
#   5. Applies max-pool NMS to find instance centres.

# %% Logging — must be before torch import
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("instance_detection")

from glob import glob
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

from dinoisawesome import DinoEncoder
from dinoisawesome.annotation_utils import load_annotations
from dinoisawesome.instance_detection import (
    compute_density_map,
    compute_exemplar_features,
    extract_patch_tokens,
    extract_peaks,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared.gt_utils import (
    gt_instance_patch_sizes as _shared_gt_instance_patch_sizes,  # noqa: E402
)
from _shared.mask_geometry import mask_iou, pixel_mask_to_patch_mask  # noqa: E402

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

part_type = "LHa"
ref_number = 1
data_dir = _REPO_ROOT / "data" / "abc3"

all_images = glob(str(data_dir / f"{part_type}_*.jpg"))
REF_IMAGE_PATH: str = [i for i in all_images if f"{part_type}_{ref_number}.jpg" in i][0]
TAR_IMAGE_PATH: list[str] = [i for i in all_images if f"{part_type}_{ref_number}.jpg" not in i]

EXEMPLAR_PATH = REF_IMAGE_PATH
QUERY_PATHS = TAR_IMAGE_PATH
EXEMPLAR_MASK_PATH: Path | None = data_dir / "annotations" / f"{part_type}_{ref_number}"
#EXEMPLAR_CLASS: list[str] | None = ['donut foam single', 'donut foam']  # None = use all masks; e.g. ["donut foam"]
EXEMPLAR_CLASS: list[str] | None = ['donut foam single', 'donut foam']  # None = use all masks; e.g. ["donut foam"]

DINO_VERSION = "v3"
DINO_SIZE = "large"
IMG_SIZE = 512 * 3
LAYER_IDX = 23
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")

MASK_PATCH_THRESHOLD = 0.3
EXEMPLAR_MODE = "mean"  # "mean" or "kmeans"
EXEMPLAR_K = 3

DENSITY_THRESHOLD = 0.3
PEAK_KERNEL_SIZE = 5
MIN_PEAK_THRESHOLD = 0.5

log.info("Reference images: 1  |  Target images: %d", len(TAR_IMAGE_PATH))

# %% Mask helpers


def load_pixel_mask(path: str) -> np.ndarray:
    """Load a .npy / .npz mask, union all channels → (H, W) bool."""
    if path.endswith(".npy"):
        seg = np.load(path)  # (N, H, W) or (H, W, N)
        if seg.ndim == 3 and seg.shape[0] < seg.shape[1]:
            seg = seg.transpose(1, 2, 0)
    else:
        seg = np.load(path)["segmaps"]  # (H, W, N)
    return seg.any(axis=2)


# %% Visualisation helpers


def upsample_map(arr: np.ndarray, size: int) -> np.ndarray:
    norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    pil = Image.fromarray((norm * 255).astype(np.uint8))
    return np.array(pil.resize((size, size), Image.NEAREST)) / 255.0


def heat_overlay(bg: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    colored = plt.get_cmap("jet")(heat)[..., :3]
    return np.clip(bg / 255.0 * (1 - alpha) + colored * alpha, 0, 1)


# %% Load encoder
encoder = DinoEncoder(
    version=DINO_VERSION,
    size=DINO_SIZE,
    img_size=IMG_SIZE,
    weights_dir=DINO_WEIGHTS_DIR,
)
log.info(
    "DINOv%s-%s | patch_size=%d | grid=%dx%d",
    DINO_VERSION[1],
    DINO_SIZE,
    encoder.patch_size,
    encoder.grid_h,
    encoder.grid_w,
)
GRID_H, GRID_W = encoder.grid_h, encoder.grid_w
PATCH_SIZE = encoder.patch_size

# %% Step 0 — Input images and exemplar mask
exemplar_img = Image.open(EXEMPLAR_PATH).convert("RGB")
query_imgs = [Image.open(p).convert("RGB") for p in QUERY_PATHS]

log.info("Exemplar: %s  (%dx%d px)", EXEMPLAR_PATH, *exemplar_img.size)
for p, qi in zip(QUERY_PATHS, query_imgs):
    log.info("Query:    %s  (%dx%d px)", p, *qi.size)

pixel_mask: np.ndarray | None = None
fg_mask: np.ndarray | None = None
n_instances = 0
exemplar_img_orig = exemplar_img
query_imgs_orig = list(query_imgs)
if EXEMPLAR_MASK_PATH is not None:
    try:
        anns = load_annotations(EXEMPLAR_MASK_PATH)
        detection_masks = [
            a for a in anns if EXEMPLAR_CLASS is None or a["class"] in EXEMPLAR_CLASS
        ]
        if detection_masks:
            pixel_mask = np.stack([a["mask"] for a in detection_masks]).any(axis=0)
            n_instances = len(detection_masks)
            log.info(
                "Instance masks loaded: %d  coverage=%.1f%%",
                n_instances,
                100.0 * pixel_mask.mean(),
            )
        else:
            log.warning("No masks matched classes %s — running without mask.", EXEMPLAR_CLASS)
    except FileNotFoundError:
        log.warning("Mask not found at %s — running without mask.", EXEMPLAR_MASK_PATH)

display_ex = np.array(exemplar_img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))

ncols = 2 + len(query_imgs)
fig, axes = plt.subplots(1, ncols, figsize=(ncols * 4, 5))

axes[0].imshow(exemplar_img_orig)
if fg_mask is not None:
    fg_disp = (
        np.array(
            Image.fromarray(fg_mask.astype(np.uint8) * 255).resize(
                exemplar_img_orig.size, Image.NEAREST
            )
        )
        > 0
    )
    fg_overlay = np.zeros((*fg_disp.shape, 4), dtype=np.float32)
    fg_overlay[fg_disp] = [0.9, 0.4, 0.1, 0.5]
    axes[0].imshow(fg_overlay)
    axes[0].set_title(
        f"Exemplar (original) + foreground mask\n{100 * fg_mask.mean():.1f}% coverage",
        fontsize=10,
    )
else:
    axes[0].set_title("Exemplar (original)", fontsize=10)
#axes[0].axis("off")

axes[1].imshow(display_ex)
if pixel_mask is not None:
    mask_disp = (
        np.array(
            Image.fromarray(pixel_mask.astype(np.uint8) * 255).resize(
                (IMG_SIZE, IMG_SIZE), Image.NEAREST
            )
        )
        > 0
    )
    green_overlay = np.zeros((*mask_disp.shape, 4), dtype=np.float32)
    green_overlay[mask_disp] = [0.2, 0.9, 0.2, 0.5]
    axes[1].imshow(green_overlay)
    axes[1].set_title(
        f"Exemplar (cropped to fg) + instance mask\n"
        f"{n_instances} instance(s) | {100 * pixel_mask.mean():.1f}% coverage",
        fontsize=10,
    )
else:
    axes[1].set_title("Exemplar (no mask)", fontsize=10)
axes[1].axis("off")

for i, (p, qi_orig, qi) in enumerate(zip(QUERY_PATHS, query_imgs_orig, query_imgs), start=2):
    axes[i].imshow(qi_orig)
    axes[i].set_title(f"Query {i - 1} (original)\n{Path(p).name}", fontsize=9)
    axes[i].axis("off")

plt.suptitle("Step 0: Input images and exemplar mask", fontsize=12)
plt.tight_layout()
plt.show()

# %% Step 1 — Patch token extraction + mask projection
exemplar_tokens, ex_h, ex_w = extract_patch_tokens(encoder, exemplar_img, LAYER_IDX, debias=True)
query_tokens, q_h, q_w = extract_patch_tokens(encoder, query_imgs[0], LAYER_IDX, debias=True)

log.info(
    "Exemplar tokens: %s  (grid %dx%d)  device=%s",
    exemplar_tokens.shape,
    ex_h,
    ex_w,
    exemplar_tokens.device,
)
log.info("Query tokens:    %s  (grid %dx%d)", query_tokens.shape, q_h, q_w)

if pixel_mask is not None:
    patch_mask = pixel_mask_to_patch_mask(pixel_mask, ex_h, ex_w, IMG_SIZE, MASK_PATCH_THRESHOLD)
    patch_mask_flat = torch.from_numpy(patch_mask.reshape(-1)).to(exemplar_tokens.device)
    exemplar_tokens_masked = exemplar_tokens[patch_mask_flat]
    log.info(
        "Patch mask: %d / %d patches  (threshold=%.2f)",
        patch_mask_flat.sum().item(),
        ex_h * ex_w,
        MASK_PATCH_THRESHOLD,
    )
else:
    patch_mask = np.ones((ex_h, ex_w), dtype=bool)
    patch_mask_flat = torch.ones(ex_h * ex_w, dtype=torch.bool, device=exemplar_tokens.device)
    exemplar_tokens_masked = exemplar_tokens

# %% PCA visualisation — joint embedding
all_tokens_np = torch.cat([exemplar_tokens, query_tokens], dim=0).cpu().float().numpy()
pca = PCA(n_components=3)
pca.fit(all_tokens_np)
log.info(
    "PCA explained variance: %.1f%% / %.1f%% / %.1f%%",
    pca.explained_variance_ratio_[0] * 100,
    pca.explained_variance_ratio_[1] * 100,
    pca.explained_variance_ratio_[2] * 100,
)


def tokens_to_pca_rgb(tokens: torch.Tensor, h: int, w: int) -> np.ndarray:
    proj = pca.transform(tokens.cpu().float().numpy())
    for c in range(3):
        lo, hi = proj[:, c].min(), proj[:, c].max()
        proj[:, c] = (proj[:, c] - lo) / (hi - lo + 1e-8)
    return proj.reshape(h, w, 3)


def to_display(rgb_hw3: np.ndarray, size: int) -> np.ndarray:
    return (
        np.array(
            Image.fromarray((rgb_hw3 * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
        )
        / 255.0
    )


ex_rgb = tokens_to_pca_rgb(exemplar_tokens, ex_h, ex_w)
q_rgb = tokens_to_pca_rgb(query_tokens, q_h, q_w)
ex_rgb_up = to_display(ex_rgb, IMG_SIZE)
q_rgb_up = to_display(q_rgb, IMG_SIZE)

display_q0 = np.array(query_imgs[0].resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))
mask_border = (
    np.array(
        Image.fromarray(patch_mask.astype(np.uint8) * 255).resize(
            (IMG_SIZE, IMG_SIZE), Image.NEAREST
        )
    )
    > 0
)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(display_ex)
axes[0, 0].set_title("Exemplar — original", fontsize=10)
axes[0, 0].axis("off")

axes[0, 1].imshow(ex_rgb_up)
axes[0, 1].set_title(
    f"Exemplar — PCA(tokens) → RGB\nblock {LAYER_IDX}  |  grid {ex_h}×{ex_w}", fontsize=10
)
axes[0, 1].axis("off")

axes[0, 2].imshow(ex_rgb_up)
green_ov = np.zeros((IMG_SIZE, IMG_SIZE, 4), dtype=np.float32)
green_ov[mask_border] = [0.2, 0.9, 0.2, 0.45]
green_ov[~mask_border] = [0.0, 0.0, 0.0, 0.35]
axes[0, 2].imshow(green_ov)
axes[0, 2].set_title(
    f"Exemplar PCA — masked patches highlighted\n"
    f"{patch_mask.sum()} / {ex_h * ex_w} patches used  (threshold={MASK_PATCH_THRESHOLD})",
    fontsize=10,
)
axes[0, 2].axis("off")

axes[1, 0].imshow(display_q0)
axes[1, 0].set_title("Query 1 — original", fontsize=10)
axes[1, 0].axis("off")

axes[1, 1].imshow(q_rgb_up)
axes[1, 1].set_title(
    f"Query 1 — PCA(tokens) → RGB\nblock {LAYER_IDX}  |  grid {q_h}×{q_w}", fontsize=10
)
axes[1, 1].axis("off")
axes[1, 2].axis("off")

plt.suptitle(
    f"Step 1: Patch token extraction + mask projection  |  DINOv{DINO_VERSION[1]}-{DINO_SIZE}",
    fontsize=12,
    y=1.01,
)
plt.tight_layout()
plt.show()

# %% Step 2 — Exemplar feature aggregation
feat_mean = compute_exemplar_features(exemplar_tokens_masked, mode="mean")
sim_ex_to_mean = (exemplar_tokens @ feat_mean.T).squeeze(1).reshape(ex_h, ex_w).cpu().numpy()

feat_kmeans = compute_exemplar_features(exemplar_tokens_masked, mode="kmeans", k=EXEMPLAR_K)
sim_ex_km = (exemplar_tokens @ feat_kmeans.T).cpu().numpy()
assignments = sim_ex_km.argmax(axis=1).reshape(ex_h, ex_w)
max_sim_km = sim_ex_km.max(axis=1).reshape(ex_h, ex_w)

CMAP_KM = plt.get_cmap("tab10")
fig, axes = plt.subplots(1, 4, figsize=(25, 5))

axes[0].imshow(display_ex)
axes[0].imshow(green_ov)
axes[0].set_title("Exemplar + patch mask", fontsize=10)
axes[0].axis("off")

im1 = axes[1].imshow(
    sim_ex_to_mean, cmap="viridis", vmin=sim_ex_to_mean.min(), vmax=1.0, aspect="auto"
)
axes[1].set_title("Exemplar patches: sim to mean descriptor\n(masked-token mean)", fontsize=10)
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], shrink=0.75, pad=0.02)

axes[2].imshow(assignments, cmap="tab10", vmin=0, vmax=9, aspect="auto")
legend_km = [mpatches.Patch(color=CMAP_KM(i), label=f"centroid {i}") for i in range(EXEMPLAR_K)]
axes[2].legend(handles=legend_km, loc="lower right", fontsize=8, framealpha=0.85)
axes[2].set_title(
    f"K-means centroid assignment (K={EXEMPLAR_K})\nmasked-token K-means", fontsize=10
)
axes[2].axis("off")

im3 = axes[3].imshow(max_sim_km, cmap="viridis", vmin=max_sim_km.min(), vmax=1.0, aspect="auto")
axes[3].set_title(f"Max sim to nearest centroid (K={EXEMPLAR_K})", fontsize=10)
axes[3].axis("off")
plt.colorbar(im3, ax=axes[3], shrink=0.75, pad=0.02)

plt.suptitle("Step 2: Exemplar feature aggregation (masked tokens)", fontsize=12, y=1.01)
plt.tight_layout()
plt.show()

# %% Step 3 — Density maps: masked vs unmasked
feat_masked = compute_exemplar_features(exemplar_tokens_masked, mode=EXEMPLAR_MODE, k=EXEMPLAR_K)
feat_unmasked = compute_exemplar_features(exemplar_tokens, mode=EXEMPLAR_MODE, k=EXEMPLAR_K)


def raw_sim_2d(q_tok: torch.Tensor, feat: torch.Tensor) -> np.ndarray:
    return (q_tok @ feat.T).mean(dim=-1).reshape(q_h, q_w).cpu().numpy()

PEAK_KERNEL_SIZE = 7
MIN_PEAK_THRESHOLD = 0.1
DENSITY_THRESHOLD = 0.7
sim_masked_raw = raw_sim_2d(query_tokens, feat_masked)
sim_unmasked_raw = raw_sim_2d(query_tokens, feat_unmasked)
dm_masked = compute_density_map(query_tokens, feat_masked, q_h, q_w, DENSITY_THRESHOLD)
dm_unmasked = compute_density_map(query_tokens, feat_unmasked, q_h, q_w, DENSITY_THRESHOLD)

log.info(
    "Masked   density: max=%.3f  non-zero=%d/%d",
    dm_masked.max().item(),
    (dm_masked > 0).sum().item(),
    q_h * q_w,
)
log.info(
    "Unmasked density: max=%.3f  non-zero=%d/%d",
    dm_unmasked.max().item(),
    (dm_unmasked > 0).sum().item(),
    q_h * q_w,
)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for row, (label, sim_raw, dm, feat) in enumerate(
    [
        ("masked exemplar", sim_masked_raw, dm_masked, feat_masked),
        ("unmasked exemplar", sim_unmasked_raw, dm_unmasked, feat_unmasked),
    ]
):
    dm_np = dm.cpu().numpy()
    im0 = axes[row, 0].imshow(sim_raw, cmap="jet", aspect="auto")
    axes[row, 0].set_title(f"[{label}]\nRaw cosine similarity", fontsize=10)
    axes[row, 0].axis("off")
    plt.colorbar(im0, ax=axes[row, 0], shrink=0.75, pad=0.02)

    im1 = axes[row, 1].imshow(dm_np, cmap="jet", aspect="auto")
    axes[row, 1].set_title(f"[{label}]\nDensity map (threshold={DENSITY_THRESHOLD})", fontsize=10)
    axes[row, 1].axis("off")
    plt.colorbar(im1, ax=axes[row, 1], shrink=0.75, pad=0.02)

    axes[row, 2].imshow(display_q0)
    axes[row, 2].imshow(heat_overlay(display_q0, upsample_map(dm_np, IMG_SIZE)))
    axes[row, 2].set_title(f"[{label}]\nDensity overlay on query", fontsize=10)
    axes[row, 2].axis("off")

plt.suptitle(
    f"Step 3: Density map — masked vs unmasked  |  mode={EXEMPLAR_MODE}", fontsize=12, y=1.01
)
plt.tight_layout()
plt.show()

# %% Step 4 — Max-pool NMS peak extraction
density_map = dm_masked
dm_4d = density_map.unsqueeze(0).unsqueeze(0)
padding = PEAK_KERNEL_SIZE // 2
pooled = F.max_pool2d(dm_4d, kernel_size=PEAK_KERNEL_SIZE, stride=1, padding=padding).squeeze()

is_local_max = density_map == pooled
is_peak = is_local_max & (density_map > MIN_PEAK_THRESHOLD)

peaks = extract_peaks(density_map, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)
peaks_sim = density_map[peaks[:, 1], peaks[:, 0]]
log.info(
    "Peaks: %d found  (kernel=%d  min_threshold=%.4f)",
    len(peaks),
    PEAK_KERNEL_SIZE,
    MIN_PEAK_THRESHOLD,
)
if len(peaks):
    log.info("Peak patch-grid (x,y): %s", list(zip(peaks[:, 0].tolist(), peaks[:, 1].tolist())))
    log.info("Peak similarities: %s", peaks_sim.cpu().numpy().tolist())

density_np = density_map.cpu().numpy()
pooled_np = pooled.cpu().numpy()
diff_np = (density_map - pooled).cpu().numpy()
is_peak_np = is_peak.cpu().numpy()

fig, axes = plt.subplots(1, 4, figsize=(24, 6))
im0 = axes[0].imshow(density_np, cmap="jet", aspect="auto")
axes[0].set_title("Density map (input to NMS)", fontsize=10)
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], shrink=0.75, pad=0.02)

im1 = axes[1].imshow(pooled_np, cmap="jet", aspect="auto")
axes[1].set_title(f"Max-pooled map (kernel {PEAK_KERNEL_SIZE}×{PEAK_KERNEL_SIZE})", fontsize=10)
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], shrink=0.75, pad=0.02)

im2 = axes[2].imshow(diff_np, cmap="coolwarm", aspect="auto")
axes[2].set_title("density − pooled\n(= 0 at local maxima)", fontsize=10)
axes[2].axis("off")
plt.colorbar(im2, ax=axes[2], shrink=0.75, pad=0.02)

axes[3].imshow(is_peak_np.astype(np.float32), cmap="Greys_r", aspect="auto")
if len(peaks):
    axes[3].scatter(
        peaks[:, 0].cpu().numpy(),
        peaks[:, 1].cpu().numpy(),
        c="red",
        s=120,
        marker="x",
        linewidths=2,
        zorder=5,
    )
axes[3].set_title(f"Peak mask  ({len(peaks)} peaks)", fontsize=10)
axes[3].axis("off")

plt.suptitle(
    f"Step 4: Max-pool NMS  |  kernel={PEAK_KERNEL_SIZE}  min_threshold={MIN_PEAK_THRESHOLD}",
    fontsize=12,
    y=1.01,
)
plt.tight_layout()
plt.show()

# %% Step 5 — Final detection result
scale = [i / IMG_SIZE for i in query_imgs[0].size]
peaks_np = peaks.cpu().numpy()
px_x_vis = (peaks_np[:, 0] + 0.5) * PATCH_SIZE * scale[0] if len(peaks_np) else []
px_y_vis = (peaks_np[:, 1] + 0.5) * PATCH_SIZE * scale[1] if len(peaks_np) else []

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    f"Instance Detection  |  DINOv{DINO_VERSION[1]}-{DINO_SIZE}  block {LAYER_IDX}  "
    f"mode={EXEMPLAR_MODE}  threshold={DENSITY_THRESHOLD}  masked={pixel_mask is not None}",
    fontsize=10,
)

axes[0].imshow(exemplar_img)
if pixel_mask is not None:
    mask_big = (
        np.array(
            Image.fromarray(pixel_mask.astype(np.uint8) * 255).resize(
                exemplar_img.size, Image.NEAREST
            )
        )
        > 0
    )
    g_ov = np.zeros((*mask_big.shape, 4), dtype=np.float32)
    g_ov[mask_big] = [0.2, 0.9, 0.2, 0.4]
    axes[0].imshow(g_ov)
axes[0].set_title("Exemplar  (green = masked region used)", fontsize=10)
axes[0].axis("off")

im = axes[1].imshow(density_np, cmap="jet", interpolation="nearest")
axes[1].set_title("Density map (patch grid)", fontsize=10)
axes[1].axis("off")
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

axes[2].imshow(query_imgs[0])
if len(peaks_np):
    axes[2].scatter(
        px_x_vis, px_y_vis, c="red", s=120, marker="o", linewidths=1.5, edgecolors="white", zorder=5
    )
axes[2].set_title(f"Detections — {len(peaks_np)} found", fontsize=10)
axes[2].axis("off")

plt.tight_layout()
plt.show()

# %% Sweep 2 — Density threshold
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
fig, axes = plt.subplots(2, len(THRESHOLDS), figsize=(len(THRESHOLDS) * 4, 9))

for col, thr in enumerate(THRESHOLDS):
    dm_thr = compute_density_map(query_tokens, feat_masked, q_h, q_w, threshold=thr)
    pk_thr = extract_peaks(dm_thr, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)
    dm_np = dm_thr.cpu().numpy()
    pk_np = pk_thr.cpu().numpy()

    im = axes[0, col].imshow(dm_np, cmap="jet", aspect="auto")
    axes[0, col].set_title(f"threshold={thr}\n{(dm_np > 0).sum()} active", fontsize=9)
    axes[0, col].axis("off")
    plt.colorbar(im, ax=axes[0, col], shrink=0.75, pad=0.02)

    axes[1, col].imshow(display_q0)
    axes[1, col].imshow(heat_overlay(display_q0, upsample_map(dm_np, IMG_SIZE)))
    if len(pk_np):
        axes[1, col].scatter(
            (pk_np[:, 0] + 0.5) * PATCH_SIZE,
            (pk_np[:, 1] + 0.5) * PATCH_SIZE,
            c="red",
            s=80,
            marker="o",
            linewidths=1.5,
            edgecolors="white",
            zorder=5,
        )
    axes[1, col].set_title(f"{len(pk_np)} peak(s)", fontsize=9)
    axes[1, col].axis("off")

plt.suptitle(
    f"Sweep: density_threshold  |  mode={EXEMPLAR_MODE}  masked exemplar", fontsize=12, y=1.01
)
plt.tight_layout()
plt.show()

# %% Sweep 3 — Transformer block (layer index)
LAYER_INDICES = [6, 12, 16, 20, 22, 23]
fig, axes = plt.subplots(2, len(LAYER_INDICES), figsize=(len(LAYER_INDICES) * 4, 9))

for col, layer in enumerate(LAYER_INDICES):
    ex_tok_l, ex_h_l, ex_w_l = extract_patch_tokens(encoder, exemplar_img, layer, debias=True)
    q_tok_l, q_h_l, q_w_l = extract_patch_tokens(encoder, query_imgs[0], layer, debias=True)

    if pixel_mask is not None:
        pm_l = pixel_mask_to_patch_mask(pixel_mask, ex_h_l, ex_w_l, IMG_SIZE, MASK_PATCH_THRESHOLD)
        pm_flat_l = torch.from_numpy(pm_l.reshape(-1)).to(ex_tok_l.device)
        ex_tok_l_m = ex_tok_l[pm_flat_l] if pm_flat_l.any() else ex_tok_l
    else:
        ex_tok_l_m = ex_tok_l

    feat_l = compute_exemplar_features(ex_tok_l_m, mode=EXEMPLAR_MODE, k=EXEMPLAR_K)
    dm_l = compute_density_map(q_tok_l, feat_l, q_h_l, q_w_l, DENSITY_THRESHOLD)
    pk_l = extract_peaks(dm_l, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)
    dm_l_np = dm_l.cpu().numpy()

    im = axes[0, col].imshow(dm_l_np, cmap="jet", aspect="auto")
    axes[0, col].set_title(f"block {layer}", fontsize=9)
    axes[0, col].axis("off")
    plt.colorbar(im, ax=axes[0, col], shrink=0.75, pad=0.02)

    axes[1, col].imshow(display_q0)
    axes[1, col].imshow(heat_overlay(display_q0, upsample_map(dm_l_np, IMG_SIZE)))
    if len(pk_l):
        axes[1, col].scatter(
            (pk_l[:, 0].float().cpu() + 0.5) * PATCH_SIZE,
            (pk_l[:, 1].float().cpu() + 0.5) * PATCH_SIZE,
            c="red",
            s=80,
            marker="o",
            linewidths=1.5,
            edgecolors="white",
            zorder=5,
        )
    axes[1, col].set_title(f"{len(pk_l)} peak(s)", fontsize=9)
    axes[1, col].axis("off")

plt.suptitle(
    f"Sweep: layer_idx  |  mode={EXEMPLAR_MODE}  masked  density_thr={DENSITY_THRESHOLD}",
    fontsize=12,
    y=1.01,
)
plt.tight_layout()
plt.show()

# %% Sweep 4 — Aggregation mode (mean vs K-means)
MODE_CONFIGS: list[tuple[str, int]] = [("mean", 1), ("kmeans", 2), ("kmeans", 3), ("kmeans", 5)]
fig, axes = plt.subplots(2, len(MODE_CONFIGS), figsize=(len(MODE_CONFIGS) * 5, 10))

for col, (mode, k) in enumerate(MODE_CONFIGS):
    feat_m = compute_exemplar_features(exemplar_tokens_masked, mode=mode, k=k)
    dm_m = compute_density_map(query_tokens, feat_m, q_h, q_w, DENSITY_THRESHOLD)
    pk_m = extract_peaks(dm_m, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)
    dm_m_np = dm_m.cpu().numpy()
    label = "mean" if mode == "mean" else f"kmeans K={k}"

    im = axes[0, col].imshow(dm_m_np, cmap="jet", aspect="auto")
    axes[0, col].set_title(label, fontsize=9)
    axes[0, col].axis("off")
    plt.colorbar(im, ax=axes[0, col], shrink=0.75, pad=0.02)

    axes[1, col].imshow(display_q0)
    axes[1, col].imshow(heat_overlay(display_q0, upsample_map(dm_m_np, IMG_SIZE)))
    if len(pk_m):
        axes[1, col].scatter(
            (pk_m[:, 0].float().cpu() + 0.5) * PATCH_SIZE,
            (pk_m[:, 1].float().cpu() + 0.5) * PATCH_SIZE,
            c="red",
            s=80,
            marker="o",
            linewidths=1.5,
            edgecolors="white",
            zorder=5,
        )
    axes[1, col].set_title(f"{len(pk_m)} peak(s)", fontsize=9)
    axes[1, col].axis("off")

plt.suptitle(
    f"Sweep: aggregation mode  |  block={LAYER_IDX}  density_thr={DENSITY_THRESHOLD}  masked",
    fontsize=12,
    y=1.01,
)
plt.tight_layout()
plt.show()

# %% Multi-query detection
query_results: list[dict] = []
for path, q_img in zip(QUERY_PATHS, query_imgs):
    q_tok_f, q_h_f, q_w_f = extract_patch_tokens(encoder, q_img, LAYER_IDX, debias=True)
    dm_f = compute_density_map(q_tok_f, feat_masked, q_h_f, q_w_f, DENSITY_THRESHOLD)
    pk_f = extract_peaks(dm_f, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)
    query_results.append({"path": path, "img": q_img, "density": dm_f, "peaks": pk_f})
    log.info("%s → %d peak(s)  density max=%.3f", Path(path).name, len(pk_f), dm_f.max().item())

nq = len(query_results)
fig, axes = plt.subplots(3, nq, figsize=(nq * 5, 15))
if nq == 1:
    axes = axes.reshape(3, 1)

for col, qr in enumerate(query_results):
    dm_np_f = qr["density"].cpu().numpy()
    pk_np_f = qr["peaks"].cpu().numpy()

    axes[0, col].imshow(qr["img"])
    axes[0, col].set_title(Path(qr["path"]).name, fontsize=8)
    axes[0, col].axis("off")

    im = axes[1, col].imshow(dm_np_f, cmap="jet", aspect="auto")
    axes[1, col].set_title(f"density max={dm_np_f.max():.3f}", fontsize=9)
    axes[1, col].axis("off")
    plt.colorbar(im, ax=axes[1, col], shrink=0.75, pad=0.02)

    axes[2, col].imshow(qr["img"])
    if len(pk_np_f):
        axes[2, col].scatter(
            (pk_np_f[:, 0] + 0.5) * PATCH_SIZE,
            (pk_np_f[:, 1] + 0.5) * PATCH_SIZE,
            c="red",
            s=120,
            marker="o",
            linewidths=1.5,
            edgecolors="white",
            zorder=5,
        )
    axes[2, col].set_title(f"{len(pk_np_f)} peak(s)", fontsize=9)
    axes[2, col].axis("off")

plt.suptitle(
    f"Multi-query  |  DINOv{DINO_VERSION[1]}-{DINO_SIZE}  block {LAYER_IDX}  "
    f"mode={EXEMPLAR_MODE}  masked",
    fontsize=11,
    y=1.01,
)
plt.tight_layout()
plt.show()

# %% Peak self-similarity heatmap drilldown
if len(peaks) == 0:
    log.warning("No peaks in query 1 — lower DENSITY_THRESHOLD to see heatmaps.")
else:
    n_peaks = len(peaks)
    disp_q_d = np.array(query_imgs[0].resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))

    fig, axes = plt.subplots(2, n_peaks, figsize=(n_peaks * 4, 8))
    if n_peaks == 1:
        axes = axes.reshape(2, 1)

    for col, peak in enumerate(peaks):
        px, py = peak[0].item(), peak[1].item()
        peak_feat = query_tokens[py * q_w + px]

        sim_self = (query_tokens @ peak_feat).reshape(q_h, q_w).cpu().numpy()
        heat_self = upsample_map(sim_self, IMG_SIZE)
        blended = np.clip(
            disp_q_d / 255.0 * 0.45 + plt.get_cmap("jet")(heat_self)[..., :3] * 0.55, 0, 1
        )

        im = axes[0, col].imshow(sim_self, cmap="jet", vmin=sim_self.min(), vmax=1.0, aspect="auto")
        axes[0, col].set_title(
            f"Peak ({px},{py}) self-sim\nscore={density_np[py, px]:.3f}", fontsize=9
        )
        axes[0, col].axis("off")
        plt.colorbar(im, ax=axes[0, col], shrink=0.75, pad=0.02)

        axes[1, col].imshow(blended)
        axes[1, col].scatter(
            [(px + 0.5) * PATCH_SIZE],
            [(py + 0.5) * PATCH_SIZE],
            c="red",
            s=120,
            marker="o",
            linewidths=1.5,
            edgecolors="white",
            zorder=5,
        )
        axes[1, col].set_title("Overlay (heat + query)", fontsize=9)
        axes[1, col].axis("off")

    plt.suptitle(
        f"Peak self-similarity heatmaps  |  query 1  |  block {LAYER_IDX}", fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.show()

# %% Experiment — Region-growing IoU NMS
# Pipeline:
#   1. Extract raw peak candidates with a relaxed score threshold.
#   2. Flood-fill the density map from each peak (4-connected, >= GROW_THRESHOLD)
#      to get a per-instance binary mask.
#   3. Sort candidates by score, then greedily suppress any peak whose mask
#      overlaps a higher-scoring kept mask above IOU_THRESHOLD.
#   4. Visualise: grown masks | original NMS result | IoU-NMS result.

RELAXED_MIN_PEAK = 0.3   # lower than MIN_PEAK_THRESHOLD to get more candidates
GROW_THRESHOLD = DENSITY_THRESHOLD  # flood-fill boundary — same as density cutoff
IOU_THRESHOLD = 0.3      # suppress if intersection/union exceeds this


def region_grow(
    density_arr: np.ndarray, seed_y: int, seed_x: int, threshold: float
) -> np.ndarray:
    """4-connected flood-fill from (seed_y, seed_x) into patches >= threshold."""
    h, w = density_arr.shape
    visited = np.zeros((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=bool)
    stack = [(seed_y, seed_x)]
    visited[seed_y, seed_x] = True
    while stack:
        y, x = stack.pop()
        if density_arr[y, x] >= threshold:
            mask[y, x] = True
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
    return mask


# 1. Relaxed candidate peaks
RELAXED_MIN_PEAK = 0.1
raw_peaks = extract_peaks(density_map, PEAK_KERNEL_SIZE, RELAXED_MIN_PEAK)
if len(raw_peaks) == 0:
    log.warning("No candidates found at relaxed threshold=%.2f — skipping experiment.", RELAXED_MIN_PEAK)
else:
    raw_peaks_np = raw_peaks.cpu().numpy()  # (N, 2): col 0 = x, col 1 = y
    raw_scores = density_np[raw_peaks_np[:, 1], raw_peaks_np[:, 0]]
    log.info("Candidates (relaxed thr=%.2f): %d peaks", RELAXED_MIN_PEAK, len(raw_peaks_np))

    # 2. Grow a region mask from each candidate
    instance_masks = [
        region_grow(density_np, int(p[1]), int(p[0]), GROW_THRESHOLD)
        for p in raw_peaks_np
    ]
    log.info("Grown region areas (patches): %s", [int(m.sum()) for m in instance_masks])

    # 3. Greedy IoU suppression (highest score kept first)
    order = np.argsort(raw_scores)[::-1]
    keep: list[int] = []
    suppressed: set[int] = set()

    for idx in order:
        if idx in suppressed:
            continue
        keep.append(idx)
        for other in order:
            if other == idx or other in suppressed:
                continue
            iou = mask_iou(instance_masks[idx], instance_masks[other])
            if iou > IOU_THRESHOLD:
                suppressed.add(other)
                log.info(
                    "  suppressed peak %d (score=%.3f)  IoU=%.3f  with kept peak %d (score=%.3f)",
                    other, raw_scores[other], iou, idx, raw_scores[idx],
                )

    kept_np = raw_peaks_np[keep]
    log.info(
        "After IoU-NMS: %d / %d candidates kept  (iou_thr=%.2f)",
        len(keep), len(raw_peaks_np), IOU_THRESHOLD,
    )

    # 4. Visualise
    CMAP_INST = plt.get_cmap("tab10")

    # Build coloured mask canvases (patch-grid resolution)
    all_canvas = np.zeros((*density_np.shape, 3), dtype=np.float32)
    kept_canvas = np.zeros((*density_np.shape, 3), dtype=np.float32)

    for rank, idx in enumerate(order):
        color = np.array(CMAP_INST(rank % 10)[:3], dtype=np.float32)
        alpha = 1.0 if idx not in suppressed else 0.35
        all_canvas[instance_masks[idx]] = color * alpha

    for rank, idx in enumerate(keep):
        color = np.array(CMAP_INST(rank % 10)[:3], dtype=np.float32)
        kept_canvas[instance_masks[idx]] = color

    # Pixel-space coordinates for overlays on the original query image
    img_scale = [s / IMG_SIZE for s in query_imgs[0].size]  # (w_scale, h_scale)

    def patches_to_px(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        px_x = (arr[:, 0] + 0.5) * PATCH_SIZE * img_scale[0]
        px_y = (arr[:, 1] + 0.5) * PATCH_SIZE * img_scale[1]
        return px_x, px_y

    disp_q = np.array(query_imgs[0])
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))

    # Row 0: patch-grid views ---------------------------------------------------
    im0 = axes[0, 0].imshow(density_np, cmap="jet", aspect="auto")
    axes[0, 0].set_title("Density map (input)", fontsize=10)
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.75, pad=0.02)

    axes[0, 1].imshow(all_canvas, aspect="auto")
    for i, (px, py) in enumerate(zip(raw_peaks_np[:, 0], raw_peaks_np[:, 1])):
        marker, color = ("x", "white") if i in suppressed else ("o", "red")
        axes[0, 1].scatter(px, py, c=color, s=80, marker=marker, linewidths=1.5, zorder=5)
        axes[0, 1].text(px + 0.3, py - 0.4, f"{raw_scores[i]:.2f}", fontsize=6, color="white")
    axes[0, 1].set_title(
        f"All grown regions  (white × = suppressed)\n"
        f"{len(raw_peaks_np)} candidates  →  {len(keep)} kept  iou_thr={IOU_THRESHOLD}",
        fontsize=9,
    )
    axes[0, 1].axis("off")

    axes[0, 2].imshow(kept_canvas, aspect="auto")
    axes[0, 2].scatter(
        kept_np[:, 0], kept_np[:, 1],
        c="red", s=100, marker="o", linewidths=1.5, edgecolors="white", zorder=5,
    )
    axes[0, 2].set_title(f"Kept regions after IoU-NMS  ({len(keep)} instances)", fontsize=10)
    axes[0, 2].axis("off")

    # Row 1: overlaid on original query image ------------------------------------
    axes[1, 0].imshow(disp_q)
    axes[1, 0].set_title("Query image", fontsize=10)
    axes[1, 0].axis("off")

    # Original max-pool NMS (Step 4)
    axes[1, 1].imshow(disp_q)
    if len(peaks_np):
        ox, oy = patches_to_px(peaks_np)
        axes[1, 1].scatter(ox, oy, c="red", s=120, marker="o", linewidths=1.5, edgecolors="white", zorder=5)
    axes[1, 1].set_title(f"Original max-pool NMS  ({len(peaks_np)} peaks)", fontsize=10)
    axes[1, 1].axis("off")

    # IoU-NMS result
    axes[1, 2].imshow(disp_q)
    if len(kept_np):
        kx, ky = patches_to_px(kept_np)
        axes[1, 2].scatter(kx, ky, c="lime", s=120, marker="o", linewidths=1.5, edgecolors="white", zorder=5)
    axes[1, 2].set_title(
        f"Region-grow IoU-NMS  ({len(keep)} peaks)\n"
        f"grow_thr={GROW_THRESHOLD}  iou_thr={IOU_THRESHOLD}",
        fontsize=10,
    )
    axes[1, 2].axis("off")

    plt.suptitle(
        f"Experiment: Region-growing IoU NMS  |  relaxed_min_peak={RELAXED_MIN_PEAK}  "
        f"iou_thr={IOU_THRESHOLD}  grow_thr={GROW_THRESHOLD}",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    plt.show()

# %% Experiment — HDBSCAN clustering on density map (GT-driven cluster-size bounds)
# Pipeline:
#   1. Threshold the density map to keep only patches above CLUSTER_DENSITY_THRESHOLD.
#      Only the *spatial* position (x, y) of each kept patch is used as a clustering
#      feature — the density/similarity score is used solely for this thresholding
#      step, never as an HDBSCAN input dimension.
#   2. Load the ground-truth instance masks for the query image being clustered
#      (query_imgs[0]), project each instance mask onto the query patch grid, and
#      measure its area in patches → `gt_sizes`, one value per GT instance.
#   3. Derive (min_cluster_size, max_cluster_size) from `gt_sizes` via one of several
#      methods (median ± margin, min/max ± margin, mean ± k·std, IQR ± margin).
#   4. Run sklearn HDBSCAN with those bounds, once per method.
#   5. Visualise, per method: normalised-coordinate scatter | cluster overlay on query image.
import numpy as np

CLUSTER_DENSITY_THRESHOLD = 0.8
HDBSCAN_MIN_CLUSTER_SIZE = 3  # fallback floor; also reused by the DBSCAN sweep below

CLUSTER_SIZE_MARGIN = 3  # patches; used by the *_margin methods
CLUSTER_SIZE_STD_K = 1.0  # std multiplier; used by the mean_std method
GT_BOUND_METHODS = ["median_margin", "minmax_margin", "mean_std", "quantile_margin"]


def gt_instance_patch_sizes(
    query_path: str,
    class_filter: list[str] | None,
    grid_h: int,
    grid_w: int,
    img_size: int,
    patch_threshold: float,
) -> np.ndarray:
    """GT instance sizes (in patches) for the query image, using its own annotation set."""
    sizes = _shared_gt_instance_patch_sizes(
        Path(query_path).stem, class_filter, grid_h, grid_w, img_size, patch_threshold, data_dir
    )
    log.info("GT instance patch-sizes for %s: %s", Path(query_path).name, sizes.tolist())
    return sizes


def gt_cluster_size_bounds(
    sizes: np.ndarray, method: str, margin: float, std_k: float
) -> tuple[int, int]:
    """Derive (min_cluster_size, max_cluster_size) from GT instance patch-sizes.

    - "median_margin":   [median - margin,        median + margin]
    - "minmax_margin":   [min(sizes) - margin,     max(sizes) + margin]
    - "mean_std":        [mean - std_k*std,        mean + std_k*std]
    - "quantile_margin": [p25 - margin,             p75 + margin]
    """
    if method == "median_margin":
        center = np.median(sizes)
        lo, hi = center - margin, center + margin
    elif method == "minmax_margin":
        lo, hi = sizes.min() - margin, sizes.max() + margin
    elif method == "mean_std":
        mean, std = sizes.mean(), sizes.std()
        lo, hi = mean - std_k * std, mean + std_k * std
    elif method == "quantile_margin":
        p25, p75 = np.percentile(sizes, [25, 75])
        lo, hi = p25 - margin, p75 + margin
    else:
        raise ValueError(f"Unknown GT bound method: {method}")

    min_cluster_size = max(2, int(round(lo)))
    max_cluster_size = max(min_cluster_size, int(round(hi)))
    return min_cluster_size, max_cluster_size


redone_density_np = density_np + DENSITY_THRESHOLD
ys_grid, xs_grid = np.where(redone_density_np > CLUSTER_DENSITY_THRESHOLD)
scores_grid = redone_density_np[ys_grid, xs_grid]
log.info(
    "HDBSCAN input: %d / %d patches above threshold=%.2f",
    len(xs_grid),
    density_np.size,
    CLUSTER_DENSITY_THRESHOLD,
)

gt_sizes = gt_instance_patch_sizes(
    QUERY_PATHS[0], EXEMPLAR_CLASS, q_h, q_w, IMG_SIZE, MASK_PATCH_THRESHOLD
)

if len(xs_grid) < HDBSCAN_MIN_CLUSTER_SIZE:
    log.warning("Too few points above threshold — skipping HDBSCAN experiment.")
elif len(gt_sizes) == 0:
    log.warning("No GT instance sizes available — skipping GT-driven HDBSCAN experiment.")
else:
    xs_norm = xs_grid / max(q_w - 1, 1)
    ys_norm = ys_grid / max(q_h - 1, 1)
    features = np.stack([xs_norm, ys_norm], axis=1)  # spatial-only — no similarity feature

    disp_q_c = np.array(query_imgs[0])
    img_scale_c = [s / IMG_SIZE for s in query_imgs[0].size]

    def patch_to_px(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (x + 0.5) * PATCH_SIZE * img_scale_c[0], (y + 0.5) * PATCH_SIZE * img_scale_c[1]

    CMAP_HDB = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, len(GT_BOUND_METHODS), figsize=(len(GT_BOUND_METHODS) * 4.5, 9))

    for col, method in enumerate(GT_BOUND_METHODS):
        min_cs, max_cs = gt_cluster_size_bounds(
            gt_sizes, method, CLUSTER_SIZE_MARGIN, CLUSTER_SIZE_STD_K
        )
        labels = HDBSCAN(min_cluster_size=min_cs, max_cluster_size=max_cs).fit_predict(features)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        log.info(
            "method=%s -> min_cluster_size=%d max_cluster_size=%d -> %d cluster(s) (%d noise)",
            method,
            min_cs,
            max_cs,
            n_clusters,
            n_noise,
        )

        colors = np.array(
            [CMAP_HDB(lbl % 10) if lbl >= 0 else (0.6, 0.6, 0.6, 1.0) for lbl in labels]
        )

        axes[0, col].scatter(xs_norm, ys_norm, c=colors, s=25)
        axes[0, col].set_xlim(-0.05, 1.05)
        axes[0, col].set_ylim(1.05, -0.05)
        axes[0, col].set_aspect("equal")
        axes[0, col].set_title(
            f"{method}\nsize=[{min_cs},{max_cs}]  {n_clusters} cluster(s)", fontsize=9
        )

        axes[1, col].imshow(disp_q_c)
        px_x, px_y = patch_to_px(xs_grid, ys_grid)
        axes[1, col].scatter(px_x, px_y, c=colors, s=30, edgecolors="white", linewidths=0.3)
        axes[1, col].set_title(f"{n_noise} noise pt(s)", fontsize=9)
        axes[1, col].axis("off")

    plt.suptitle(
        f"Experiment: HDBSCAN on density map  |  density_thr={CLUSTER_DENSITY_THRESHOLD}  "
        f"gt_sizes(n={len(gt_sizes)})={gt_sizes.tolist()}  margin={CLUSTER_SIZE_MARGIN}  "
        f"std_k={CLUSTER_SIZE_STD_K}",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()
    plt.show()

# %% Experiment — HDBSCAN with a programmatic density threshold (minmax_margin sizing)
# Pipeline:
#   1. Recompute a *raw*, unthresholded cosine-similarity density map (compute_density_map()
#      clamps below its own threshold, which would hide most of the background distribution).
#   2. Project the query's GT instance masks onto the patch grid, then split every patch's
#      raw density value into two populations: "true" (inside a GT instance) and
#      "false" (outside all GT instances).
#   3. Derive the density threshold *from those two distributions* instead of a hand-tuned
#      constant, via one of several strategies (see `gt_density_threshold` below).
#   4. Threshold the raw density map at the derived value -> candidate patches -> spatial-only
#      features (density is never a clustering feature, only a filter).
#   5. Size HDBSCAN with the "minmax_margin" cluster-size bounds derived from `gt_sizes`
#      (reused from the previous cell).
#   6. Visualise, per threshold strategy: true/false score histogram with the chosen cutoff |
#      cluster overlay on the query image.


def gt_patch_true_false_values(
    query_path: str,
    class_filter: list[str] | None,
    density_map: np.ndarray,
    grid_h: int,
    grid_w: int,
    img_size: int,
    patch_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(true_values, false_values, patch_mask) — density_map values split by GT membership."""
    ann_stem = data_dir / "annotations" / Path(query_path).stem
    empty_mask = np.zeros(density_map.shape, dtype=bool)
    try:
        anns = load_annotations(ann_stem)
    except FileNotFoundError:
        log.warning("No GT annotations at %s — cannot derive a density threshold.", ann_stem)
        return np.array([]), np.array([]), empty_mask

    instances = [a for a in anns if class_filter is None or a["class"] in class_filter]
    if not instances:
        log.warning("No GT instances matched classes %s at %s.", class_filter, ann_stem)
        return np.array([]), np.array([]), empty_mask

    patch_mask = empty_mask.copy()
    for a in instances:
        patch_mask |= pixel_mask_to_patch_mask(a["mask"], grid_h, grid_w, img_size, patch_threshold)

    return density_map[patch_mask], density_map[~patch_mask], patch_mask


def gt_density_threshold(
    true_vals: np.ndarray, false_vals: np.ndarray, method: str, margin: float
) -> float:
    """Derive a density-map cutoff from GT true/false patch-score distributions.

    - "mean_minus_margin":     mean(true) - margin
    - "median_minus_margin":   median(true) - margin
    - "min_minus_margin":      min(true) - margin
    - "max_false_plus_margin": max(false) + margin  (push above the hardest negatives)
    - "midpoint":              halfway between min(true) and max(false) — margin-free
    - "youden_optimal":        threshold maximising TPR - FPR over both distributions
                                (Youden's J statistic) — margin-free, ROC-style
    """
    if method == "mean_minus_margin":
        thr = true_vals.mean() - margin
    elif method == "median_minus_margin":
        thr = np.median(true_vals) - margin
    elif method == "min_minus_margin":
        thr = true_vals.min() - margin
    elif method == "max_false_plus_margin":
        thr = (false_vals.max() if len(false_vals) else true_vals.min()) + margin
    elif method == "midpoint":
        false_max = false_vals.max() if len(false_vals) else true_vals.min()
        thr = (true_vals.min() + false_max) / 2.0
    elif method == "youden_optimal":
        candidates = np.unique(np.concatenate([true_vals, false_vals]))
        best_thr, best_j = float(candidates.min()), -np.inf
        for c in candidates:
            tpr = (true_vals >= c).mean()
            fpr = (false_vals >= c).mean() if len(false_vals) else 0.0
            j = tpr - fpr
            if j > best_j:
                best_j, best_thr = j, float(c)
        thr = best_thr
    else:
        raise ValueError(f"Unknown density threshold method: {method}")
    return float(thr)


DENSITY_THRESHOLD_MARGIN = 0.05
DENSITY_THRESHOLD_METHODS = [
    "mean_minus_margin",
    "median_minus_margin",
    "min_minus_margin",
    "max_false_plus_margin",
    "midpoint",
    "youden_optimal",
]

# compute_density_map() clamps everything below its own `threshold` arg to zero, which would
# collapse most of the background distribution to a single spike at 0 — useless for searching a
# threshold. Recompute the raw, unclamped cosine-similarity map so every patch keeps its true value.
raw_density_np = raw_sim_2d(query_tokens, feat_masked)

true_vals, false_vals, gt_patch_mask_grid = gt_patch_true_false_values(
    QUERY_PATHS[0], EXEMPLAR_CLASS, raw_density_np, q_h, q_w, IMG_SIZE, MASK_PATCH_THRESHOLD
)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

im0 = axes[0].imshow(raw_density_np, cmap="jet", interpolation="nearest")
axes[0].set_title("Raw (unthresholded) cosine-similarity density map — query 1", fontsize=10)
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

axes[1].imshow(gt_patch_mask_grid, cmap="Greys_r", interpolation="nearest")
axes[1].set_title("GT binary mask (patch grid)", fontsize=10)
axes[1].axis("off")

im2 = axes[2].imshow(raw_density_np, cmap="jet", interpolation="nearest")
axes[2].contour(gt_patch_mask_grid.astype(float), levels=[0.5], colors="white", linewidths=1.5)
axes[2].set_title("Density map + GT mask overlay", fontsize=10)
axes[2].axis("off")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

log.info("GT true patches: %d  false patches: %d", len(true_vals), len(false_vals))
if len(true_vals):
    log.info(
        "true  density  min=%.3f  mean=%.3f  median=%.3f  max=%.3f",
        true_vals.min(),
        true_vals.mean(),
        np.median(true_vals),
        true_vals.max(),
    )
if len(false_vals):
    log.info(
        "false density  min=%.3f  mean=%.3f  median=%.3f  max=%.3f",
        false_vals.min(),
        false_vals.mean(),
        np.median(false_vals),
        false_vals.max(),
    )

if len(true_vals) == 0:
    log.warning("No GT-true patches — skipping programmatic-threshold HDBSCAN experiment.")
else:
    min_cs, max_cs = gt_cluster_size_bounds(
        gt_sizes, "minmax_margin", CLUSTER_SIZE_MARGIN, CLUSTER_SIZE_STD_K
    )
    log.info("HDBSCAN size bounds (minmax_margin): [%d, %d]", min_cs, max_cs)

    fig, axes = plt.subplots(
        3, len(DENSITY_THRESHOLD_METHODS), figsize=(len(DENSITY_THRESHOLD_METHODS) * 4.5, 13.5)
    )

    for col, method in enumerate(DENSITY_THRESHOLD_METHODS):
        thr = gt_density_threshold(true_vals, false_vals, method, DENSITY_THRESHOLD_MARGIN)
        binary_map = raw_density_np > thr
        ys_grid_t, xs_grid_t = np.where(binary_map)
        n_points = len(xs_grid_t)

        ax_false = axes[0, col]
        ax_true = ax_false.twinx()
        ax_false.hist(false_vals, bins=20, alpha=0.5, color="gray", label="false (outside GT)")
        ax_true.hist(true_vals, bins=20, alpha=0.5, color="green", label="true (inside GT)")
        ax_false.tick_params(axis="y", labelcolor="gray", labelsize=7)
        ax_true.tick_params(axis="y", labelcolor="green", labelsize=7)
        ax_false.axvline(thr, color="red", linestyle="--", linewidth=1.5)
        ax_false.set_title(f"{method}\nthr={thr:.3f}  n={n_points} patches", fontsize=9)
        if col == 0:
            lines_false, labels_false = ax_false.get_legend_handles_labels()
            lines_true, labels_true = ax_true.get_legend_handles_labels()
            ax_false.legend(
                lines_false + lines_true, labels_false + labels_true, fontsize=7, loc="upper left"
            )

        axes[1, col].imshow(binary_map, cmap="Greys_r", aspect="auto")
        axes[1, col].set_title(f"binary map (density > {thr:.3f})", fontsize=9)
        axes[1, col].axis("off")

        if n_points < min_cs:
            axes[2, col].imshow(np.array(query_imgs[0]))
            axes[2, col].set_title("too few points for min_cluster_size", fontsize=9)
            axes[2, col].axis("off")
            log.warning(
                "method=%s thr=%.3f -> only %d point(s) (< min_cluster_size=%d) — skipping HDBSCAN",
                method,
                thr,
                n_points,
                min_cs,
            )
            continue

        xs_norm_t = xs_grid_t / max(q_w - 1, 1)
        ys_norm_t = ys_grid_t / max(q_h - 1, 1)
        features_t = np.stack([xs_norm_t, ys_norm_t], axis=1)
        labels_t = HDBSCAN(min_cluster_size=min_cs, max_cluster_size=max_cs).fit_predict(features_t)
        n_clusters_t = len(set(labels_t)) - (1 if -1 in labels_t else 0)
        n_noise_t = int((labels_t == -1).sum())
        log.info(
            "method=%s thr=%.3f -> %d patch(es) -> %d cluster(s)  (%d noise)",
            method,
            thr,
            n_points,
            n_clusters_t,
            n_noise_t,
        )

        colors_t = np.array(
            [CMAP_HDB(lbl % 10) if lbl >= 0 else (0.6, 0.6, 0.6, 1.0) for lbl in labels_t]
        )
        axes[2, col].imshow(disp_q_c)
        px_x_t, px_y_t = patch_to_px(xs_grid_t, ys_grid_t)
        axes[2, col].scatter(px_x_t, px_y_t, c=colors_t, s=30, edgecolors="white", linewidths=0.3)
        axes[2, col].set_title(f"{n_clusters_t} cluster(s), {n_noise_t} noise", fontsize=9)
        axes[2, col].axis("off")

    plt.suptitle(
        f"Experiment: HDBSCAN  |  programmatic density threshold  |  "
        f"size_bounds(minmax_margin)=[{min_cs},{max_cs}]  thr_margin={DENSITY_THRESHOLD_MARGIN}",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()
    plt.show()


# %%
