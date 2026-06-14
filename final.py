# %% [markdown]
# # Instance Detection Head
# 
# Training-free instance detection via DINOv3 patch-token cosine similarity.
# Given one **exemplar** image (what the object looks like) and a **query** image
# (the scene to search), the pipeline:
# 
# 1. Extracts L2-normalised patch tokens from an intermediate transformer block.
# 2. **Masks** the exemplar tokens to the object region using the accompanying `_mask_good.npz`.
# 3. Aggregates the *masked* exemplar tokens into a compact descriptor (mean **or** K-means centroids).
# 4. Computes per-patch cosine similarity between query and descriptor → density map.
# 5. Applies background suppression (threshold) and max-pool NMS to find instance centres.
# 
# | Step | Function |
# |------|----------|
# | Token extraction | `extract_patch_tokens` |
# | **Mask → patch grid** | `pixel_mask_to_patch_mask` (notebook helper) |
# | Exemplar aggregation | `compute_exemplar_features` |
# | Density map | `compute_density_map` |
# | Peak extraction (NMS) | `extract_peaks` |

# %%
# Logging must be configured before torch is imported.
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("instance_detection")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from dotenv import load_dotenv

from dinoisawesome import DinoEncoder
from dinoisawesome.instance_detection import (
    compute_density_map,
    compute_exemplar_features,
    extract_patch_tokens,
    extract_peaks,
)

# %%
# ── Parameters ────────────────────────────────────────────────────────────────
load_dotenv()

# Exemplar image + its instance mask (_mask_good.npz).
# The mask npz contains segmaps (H, W, N_instances) bool.
# All instance channels are unioned → single foreground mask.
EXEMPLAR_PATH:      str       = "DinoisAwesome/data/custom_slim/frame_spectacles3_0100.jpg"
EXEMPLAR_MASK_PATH: str | None = "DinoisAwesome/data/custom_slim/frame_spectacles3_0100_mask_good.npz"

# Queries — no masks needed on the query side.
QUERY_PATHS: list[str] = [
    "DinoisAwesome/data/custom_slim/frame_20250731_130415000_iOS_1650.jpg",
    "DinoisAwesome/data/custom_slim/frame_20250731_130415000_iOS_0200.jpg",
    "DinoisAwesome/data/custom_slim/frame_spectacles3_0300.jpg",
]

# Model settings
DINO_VERSION  = "v3"    # "v2" or "v3"
DINO_SIZE     = "large" # "small" | "base" | "large" | "giant"
IMG_SIZE      = 512     # must be divisible by patch_size (16 for v3, 14 for v2)
LAYER_IDX     = 23      # 0-based transformer block index

# Mask → patch-grid: a patch is "in mask" if this fraction of its pixels are True.
MASK_PATCH_THRESHOLD = 0.3

# Exemplar aggregation
EXEMPLAR_MODE = "kmeans"  # "mean" or "kmeans"
EXEMPLAR_K    = 8       # number of centroids (only used when mode="kmeans")

# Density map / NMS
DENSITY_THRESHOLD  = 0.3
PEAK_KERNEL_SIZE   = 5
MIN_PEAK_THRESHOLD = 0.01

# Local weights directory
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")

# %%
# ── Mask helpers ──────────────────────────────────────────────────────────────

def load_pixel_mask(npz_path: str) -> np.ndarray:
    """Load a _mask_good.npz, union all instance channels → (H, W) bool."""
    seg = np.load(npz_path)["segmaps"]  # (H, W, N)
    return seg.any(axis=2)              # (H, W)


def pixel_mask_to_patch_mask(
    pixel_mask: np.ndarray,
    grid_h: int,
    grid_w: int,
    img_size: int,
    threshold: float = 0.3,
) -> np.ndarray:
    """Resize a pixel-space boolean mask to patch-grid resolution.

    Steps
    -----
    1. Resize the pixel mask to ``img_size × img_size`` (nearest-neighbour).
    2. Partition into ``grid_h × grid_w`` patch windows.
    3. A patch is True if >= ``threshold`` fraction of its pixels are True.

    Returns
    -------
    (grid_h, grid_w) bool ndarray
    """
    # Step 1 — resize to encoder input resolution
    mask_pil = Image.fromarray(pixel_mask.astype(np.uint8) * 255)
    mask_resized = np.array(
        mask_pil.resize((img_size, img_size), Image.NEAREST)
    ) > 0  # (img_size, img_size) bool

    # Step 2 — pool to patch grid
    ph = img_size // grid_h  # patch height in pixels
    pw = img_size // grid_w
    # Reshape to (grid_h, ph, grid_w, pw) then average over the two patch axes
    tiled = mask_resized.reshape(grid_h, ph, grid_w, pw)
    patch_density = tiled.mean(axis=(1, 3))  # (grid_h, grid_w) float
    return patch_density >= threshold        # (grid_h, grid_w) bool

# %%
# ── Shared visualisation helpers ───────────────────────────────────────────────

def upsample_map(arr: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour upsample a (H,W) float map, normalised to [0,1]."""
    norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    pil  = Image.fromarray((norm * 255).astype(np.uint8))
    return np.array(pil.resize((size, size), Image.NEAREST)) / 255.0


def heat_overlay(bg: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend a [0,1] normalised jet heatmap over an uint8 RGB image."""
    colored = plt.get_cmap("jet")(heat)[..., :3]
    return np.clip(bg / 255.0 * (1 - alpha) + colored * alpha, 0, 1)

# %%
# ── Load encoder ───────────────────────────────────────────────────────────────
encoder = DinoEncoder(
    version=DINO_VERSION,
    size=DINO_SIZE,
    img_size=IMG_SIZE,
    weights_dir=DINO_WEIGHTS_DIR,
)
log.info(
    "DINOv%s-%s | patch_size=%d | grid=%dx%d",
    DINO_VERSION[1], DINO_SIZE,
    encoder.patch_size, encoder.grid_h, encoder.grid_w,
)

GRID_H, GRID_W = encoder.grid_h, encoder.grid_w
PATCH_SIZE = encoder.patch_size

# %% [markdown]
# ## Step 0 — Input Images and Exemplar Mask
# 
# Show the exemplar with its instance mask overlaid, alongside the query images.
# The mask selects which part of the exemplar we actually want to match — only those
# patch tokens feed into the descriptor.

# %%
# ── Load images ────────────────────────────────────────────────────────────────
exemplar_img = Image.open(EXEMPLAR_PATH).convert("RGB")
query_imgs   = [Image.open(p).convert("RGB") for p in QUERY_PATHS]

log.info("Exemplar: %s  (%dx%d px)", EXEMPLAR_PATH, *exemplar_img.size)
for p, qi in zip(QUERY_PATHS, query_imgs):
    log.info("Query:    %s  (%dx%d px)", p, *qi.size)

# ── Load and inspect pixel mask ────────────────────────────────────────────────
pixel_mask: np.ndarray | None = None
if EXEMPLAR_MASK_PATH is not None:
    raw_seg    = np.load(EXEMPLAR_MASK_PATH)["segmaps"]  # (H, W, N)
    pixel_mask = raw_seg.any(axis=2)                      # (H, W) union
    n_instances = raw_seg.shape[2]
    log.info(
        "Mask: %s  shape=%s  instances=%d  coverage=%.1f%%",
        EXEMPLAR_MASK_PATH, raw_seg.shape,
        n_instances,
        100.0 * pixel_mask.mean(),
    )

# ── Visualise exemplar + mask + queries ───────────────────────────────────────
display_ex = np.array(exemplar_img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))

ncols = 2 + len(query_imgs)  # raw exemplar | masked exemplar | queries
fig, axes = plt.subplots(1, ncols, figsize=(ncols * 4, 5))

# Panel 0: raw exemplar
axes[0].imshow(exemplar_img)
axes[0].set_title("Exemplar (original)", fontsize=10)
axes[0].axis("off")

# Panel 1: exemplar with mask overlay
axes[1].imshow(display_ex)
if pixel_mask is not None:
    # Resize mask to IMG_SIZE for display
    mask_disp = np.array(
        Image.fromarray(pixel_mask.astype(np.uint8) * 255)
        .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    ) > 0
    # Green overlay inside mask
    green_overlay = np.zeros((*mask_disp.shape, 4), dtype=np.float32)
    green_overlay[mask_disp] = [0.2, 0.9, 0.2, 0.5]
    axes[1].imshow(green_overlay)
    axes[1].set_title(
        f"Exemplar + instance mask\n"
        f"{n_instances} instance(s) | {100*pixel_mask.mean():.1f}% coverage",
        fontsize=10,
    )
else:
    axes[1].set_title("Exemplar (no mask)", fontsize=10)
axes[1].axis("off")

# Remaining panels: queries
for i, (p, qi) in enumerate(zip(QUERY_PATHS, query_imgs), start=2):
    axes[i].imshow(qi)
    axes[i].set_title(f"Query {i-1}\n{p.split('/')[-1]}", fontsize=9)
    axes[i].axis("off")

plt.suptitle("Step 0: Input images and exemplar mask", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 1 — Patch Token Extraction + Mask Projection
# 
# `extract_patch_tokens` gives us `(H×W, D)` L2-normalised tokens for each image.
# 
# The pixel-space mask (768×768) is then:
# 1. **Resized** to `IMG_SIZE × IMG_SIZE` (nearest-neighbour).
# 2. **Pooled** to the patch grid: a patch is *inside* the mask if ≥ `MASK_PATCH_THRESHOLD`
#    of its pixels are True.
# 
# Only the **masked** tokens feed into the exemplar descriptor — background patches are discarded.
# 
# We project tokens into 3D via PCA (fitted jointly on exemplar + query) and colour them
# to show the feature landscape; masked patches are highlighted with a border overlay.

# %%
# ── Extract tokens ─────────────────────────────────────────────────────────────
exemplar_tokens, ex_h, ex_w = extract_patch_tokens(encoder, exemplar_img, LAYER_IDX)
query_tokens,    q_h,  q_w  = extract_patch_tokens(encoder, query_imgs[0], LAYER_IDX)

log.info(
    "Exemplar tokens: %s  (grid %dx%d)  device=%s",
    exemplar_tokens.shape, ex_h, ex_w, exemplar_tokens.device,
)
log.info("Query tokens:    %s  (grid %dx%d)", query_tokens.shape, q_h, q_w)

# ── Build patch-grid mask ──────────────────────────────────────────────────────
if pixel_mask is not None:
    patch_mask = pixel_mask_to_patch_mask(
        pixel_mask, ex_h, ex_w, IMG_SIZE, threshold=MASK_PATCH_THRESHOLD
    )  # (ex_h, ex_w) bool
    patch_mask_flat = torch.from_numpy(patch_mask.reshape(-1)).to(exemplar_tokens.device)  # (N,)
    exemplar_tokens_masked = exemplar_tokens[patch_mask_flat]  # (M, D)
    log.info(
        "Patch mask: %d / %d patches inside mask  (threshold=%.2f)",
        patch_mask_flat.sum().item(), ex_h * ex_w, MASK_PATCH_THRESHOLD,
    )
else:
    patch_mask      = np.ones((ex_h, ex_w), dtype=bool)
    patch_mask_flat = torch.ones(ex_h * ex_w, dtype=torch.bool, device=exemplar_tokens.device)
    exemplar_tokens_masked = exemplar_tokens
    log.info("No mask — using all %d exemplar patches.", ex_h * ex_w)

# %%
# ── PCA visualisation: joint embedding ────────────────────────────────────────
all_tokens_np = exemplar_tokens_masked.cpu().float().numpy()
pca = PCA(n_components=3)
pca.fit(all_tokens_np)
log.info(
    "PCA explained variance: %.1f%% / %.1f%% / %.1f%%",
    pca.explained_variance_ratio_[0] * 100,
    pca.explained_variance_ratio_[1] * 100,
    pca.explained_variance_ratio_[2] * 100,
)


def tokens_to_pca_rgb(tokens: torch.Tensor, h: int, w: int) -> np.ndarray:
    proj = pca.transform(tokens.cpu().float().numpy())  # (N, 3)
    for c in range(3):
        lo, hi = proj[:, c].min(), proj[:, c].max()
        proj[:, c] = (proj[:, c] - lo) / (hi - lo + 1e-8)
    return proj.reshape(h, w, 3)


ex_rgb  = tokens_to_pca_rgb(exemplar_tokens, ex_h, ex_w)  # (H, W, 3)
q_rgb   = tokens_to_pca_rgb(query_tokens,    q_h,  q_w)

# Upscale PCA maps to IMG_SIZE (nearest so patch boundaries stay sharp)
def to_display(rgb_hw3: np.ndarray, size: int) -> np.ndarray:
    return np.array(
        Image.fromarray((rgb_hw3 * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
    ) / 255.0

ex_rgb_up = to_display(ex_rgb, IMG_SIZE)
q_rgb_up  = to_display(q_rgb,  IMG_SIZE)

display_q0 = np.array(query_imgs[0].resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))

# Patch-grid mask upscaled for overlay
mask_border = np.array(
    Image.fromarray(patch_mask.astype(np.uint8) * 255).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
) > 0

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 0 — exemplar
axes[0, 0].imshow(display_ex)
axes[0, 0].set_title("Exemplar — original", fontsize=10)
axes[0, 0].axis("off")

axes[0, 1].imshow(ex_rgb_up)
axes[0, 1].set_title(
    f"Exemplar — PCA(tokens) → RGB\nblock {LAYER_IDX}  |  grid {ex_h}×{ex_w}",
    fontsize=10,
)
axes[0, 1].axis("off")

# PCA map with mask region highlighted
axes[0, 2].imshow(ex_rgb_up)
green_ov = np.zeros((IMG_SIZE, IMG_SIZE, 4), dtype=np.float32)
green_ov[mask_border]  = [0.2, 0.9, 0.2, 0.45]
green_ov[~mask_border] = [0.0, 0.0, 0.0, 0.35]   # darken excluded patches
axes[0, 2].imshow(green_ov)
axes[0, 2].set_title(
    f"Exemplar PCA — masked patches highlighted\n"
    f"{patch_mask.sum()} / {ex_h*ex_w} patches used  (threshold={MASK_PATCH_THRESHOLD})",
    fontsize=10,
)
axes[0, 2].axis("off")

# Row 1 — query (reference, no mask)
axes[1, 0].imshow(display_q0)
axes[1, 0].set_title("Query 1 — original", fontsize=10)
axes[1, 0].axis("off")

axes[1, 1].imshow(q_rgb_up)
axes[1, 1].set_title(
    f"Query 1 — PCA(tokens) → RGB\nblock {LAYER_IDX}  |  grid {q_h}×{q_w}",
    fontsize=10,
)
axes[1, 1].axis("off")

plt.suptitle(
    f"Step 1: Patch token extraction + mask projection  |  DINOv{DINO_VERSION[1]}-{DINO_SIZE}",
    fontsize=12, y=1.01,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 2 — Exemplar Feature Aggregation (masked tokens only)
# 
# `compute_exemplar_features` now receives only the **M masked tokens** instead of all H×W.
# 
# The visualisation compares:
# - **Mean descriptor** — single centroid of the masked tokens.
# - **K-means** — K centroids; each centroid captures a distinct part of the object.
# 
# The similarity map shows how similar each *query* patch is to each centroid —
# letting you see which parts of the descriptor fire where in the scene.

# %%
# ── Mean descriptor (masked) ──────────────────────────────────────────────────
feat_mean = compute_exemplar_features(exemplar_tokens_masked, mode="mean")
log.info("Mean descriptor shape: %s", feat_mean.shape)

# Per-exemplar-patch similarity to the mean — visualise how compact the masked region is
sim_ex_to_mean = (exemplar_tokens @ feat_mean.T).squeeze(1).reshape(ex_h, ex_w).cpu().numpy()

# ── K-means descriptors (masked) ──────────────────────────────────────────────
feat_kmeans = compute_exemplar_features(exemplar_tokens_masked, mode="kmeans", k=EXEMPLAR_K)
log.info("K-means descriptors shape: %s  (K=%d)", feat_kmeans.shape, EXEMPLAR_K)

# Centroid assignment map across ALL exemplar patches (masked + unmasked, for visualisation)
sim_ex_km   = (exemplar_tokens @ feat_kmeans.T).cpu().numpy()  # (N, K)
assignments = sim_ex_km.argmax(axis=1).reshape(ex_h, ex_w)
max_sim_km  = sim_ex_km.max(axis=1).reshape(ex_h, ex_w)

# ── Visualise ─────────────────────────────────────────────────────────────────
CMAP_KM = plt.get_cmap("tab10")

fig, axes = plt.subplots(1, 5, figsize=(25, 5))

# 0: exemplar with mask
axes[0].imshow(display_ex)
axes[0].imshow(green_ov)
axes[0].set_title("Exemplar + patch mask", fontsize=10)
axes[0].axis("off")

# 1: similarity to mean (on exemplar)
im1 = axes[1].imshow(sim_ex_to_mean, cmap="viridis",
                      vmin=sim_ex_to_mean.min(), vmax=1.0, aspect="auto")
axes[1].set_title("Exemplar patches: sim to mean descriptor\n(masked-token mean)", fontsize=10)
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], shrink=0.75, pad=0.02)

# 2: mask overlaid on sim-to-mean
sim_mean_masked = sim_ex_to_mean.copy()
sim_mean_masked[~patch_mask] = np.nan
im2 = axes[2].imshow(sim_mean_masked, cmap="viridis", vmin=0, vmax=1.0, aspect="auto")
axes[2].set_title("Sim to mean — masked region only", fontsize=10)
axes[2].axis("off")
plt.colorbar(im2, ax=axes[2], shrink=0.75, pad=0.02)

# 3: K-means centroid assignment
axes[3].imshow(assignments, cmap="tab10", vmin=0, vmax=9, aspect="auto")
legend_km = [
    mpatches.Patch(color=CMAP_KM(i), label=f"centroid {i}") for i in range(EXEMPLAR_K)
]
axes[3].legend(handles=legend_km, loc="lower right", fontsize=8, framealpha=0.85)
axes[3].set_title(f"K-means centroid assignment (K={EXEMPLAR_K})\nmasked-token K-means", fontsize=10)
axes[3].axis("off")

# 4: max similarity to nearest centroid
im4 = axes[4].imshow(max_sim_km, cmap="viridis",
                      vmin=max_sim_km.min(), vmax=1.0, aspect="auto")
axes[4].set_title(f"Max sim to nearest centroid (K={EXEMPLAR_K})", fontsize=10)
axes[4].axis("off")
plt.colorbar(im4, ax=axes[4], shrink=0.75, pad=0.02)

plt.suptitle("Step 2: Exemplar feature aggregation (masked tokens)", fontsize=12, y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 3 — Cosine Similarity Density Map
# 
# `compute_density_map` compares every query patch to the (masked) exemplar descriptor(s).
# 
# We also show a **masked vs unmasked** comparison: the same pipeline run without the mask
# so you can see how background tokens in the descriptor pollute the density map.

# %%
# ── Compute density maps: masked vs unmasked ──────────────────────────────────
feat_masked   = compute_exemplar_features(exemplar_tokens_masked, mode=EXEMPLAR_MODE, k=EXEMPLAR_K)
feat_unmasked = compute_exemplar_features(exemplar_tokens,        mode=EXEMPLAR_MODE, k=EXEMPLAR_K)

# Raw similarity (before threshold) for comparison
def raw_sim_2d(q_tok, feat):
    return (q_tok @ feat.T).mean(dim=-1).reshape(q_h, q_w).cpu().numpy()

sim_masked_raw   = raw_sim_2d(query_tokens, feat_masked)
sim_unmasked_raw = raw_sim_2d(query_tokens, feat_unmasked)

dm_masked   = compute_density_map(query_tokens, feat_masked,   q_h, q_w, DENSITY_THRESHOLD)
dm_unmasked = compute_density_map(query_tokens, feat_unmasked, q_h, q_w, DENSITY_THRESHOLD)

log.info(
    "Masked   density: max=%.3f  non-zero=%d/%d patches",
    dm_masked.max().item(), (dm_masked > 0).sum().item(), q_h * q_w,
)
log.info(
    "Unmasked density: max=%.3f  non-zero=%d/%d patches",
    dm_unmasked.max().item(), (dm_unmasked > 0).sum().item(), q_h * q_w,
)

# ── Figure: 2 rows (masked | unmasked) × 3 cols (raw sim | density | overlay) ─
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for row, (label, sim_raw, dm, feat) in enumerate([
    ("masked exemplar",   sim_masked_raw,   dm_masked,   feat_masked),
    ("unmasked exemplar", sim_unmasked_raw, dm_unmasked, feat_unmasked),
]):
    dm_np = dm.cpu().numpy()

    im0 = axes[row, 0].imshow(sim_raw, cmap="jet", aspect="auto")
    axes[row, 0].set_title(
        f"[{label}]\nRaw cosine similarity", fontsize=10
    )
    axes[row, 0].axis("off")
    plt.colorbar(im0, ax=axes[row, 0], shrink=0.75, pad=0.02)

    im1 = axes[row, 1].imshow(dm_np, cmap="jet", aspect="auto")
    axes[row, 1].set_title(
        f"[{label}]\nDensity map (threshold={DENSITY_THRESHOLD})", fontsize=10
    )
    axes[row, 1].axis("off")
    plt.colorbar(im1, ax=axes[row, 1], shrink=0.75, pad=0.02)

    axes[row, 2].imshow(display_q0)
    axes[row, 2].imshow(heat_overlay(display_q0, upsample_map(dm_np, IMG_SIZE)))
    axes[row, 2].set_title(
        f"[{label}]\nDensity overlay on query", fontsize=10
    )
    axes[row, 2].axis("off")

plt.suptitle(
    f"Step 3: Density map — masked vs unmasked exemplar  |  mode={EXEMPLAR_MODE}",
    fontsize=12, y=1.01,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 4 — Max-Pool NMS Peak Extraction
# 
# `extract_peaks` locates local maxima via max-pool NMS:
# 
# 1. Run `max_pool2d` with a `kernel_size × kernel_size` window.
# 2. A patch is a **peak** if its density equals the pooled value (local max)
#    AND density > `min_peak_threshold`.
# 
# We expose all intermediate tensors so you can see exactly which patches survive.

# %%
# ── Reproduce NMS internals ────────────────────────────────────────────────────
density_map = dm_masked  # use the masked-exemplar density map from here on

dm_4d   = density_map.unsqueeze(0).unsqueeze(0)
padding = PEAK_KERNEL_SIZE // 2
pooled  = F.max_pool2d(dm_4d, kernel_size=PEAK_KERNEL_SIZE, stride=1, padding=padding).squeeze()

is_local_max = (density_map == pooled)
is_peak      = is_local_max & (density_map > MIN_PEAK_THRESHOLD)

peaks = extract_peaks(density_map, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)
log.info(
    "Peaks: %d found  (kernel=%d  min_threshold=%.4f)",
    len(peaks), PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD,
)

if len(peaks):
    log.info(
        "Peak patch-grid (x,y): %s",
        list(zip(peaks[:, 0].tolist(), peaks[:, 1].tolist())),
    )

# ── Figure ────────────────────────────────────────────────────────────────────
density_np = density_map.cpu().numpy()
pooled_np  = pooled.cpu().numpy()
diff_np    = (density_map - pooled).cpu().numpy()
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
        peaks[:, 0].cpu().numpy(), peaks[:, 1].cpu().numpy(),
        c="red", s=120, marker="x", linewidths=2, zorder=5,
    )
axes[3].set_title(f"Peak mask  ({len(peaks)} peaks)", fontsize=10)
axes[3].axis("off")

plt.suptitle(
    f"Step 4: Max-pool NMS  |  kernel={PEAK_KERNEL_SIZE}  min_threshold={MIN_PEAK_THRESHOLD}",
    fontsize=12, y=1.01,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 5 — Final Detection Result

# %%
# ── Three-panel detection figure ──────────────────────────────────────────────
peaks_np = peaks.cpu().numpy()
px_x_vis = (peaks_np[:, 0] + 0.5) * PATCH_SIZE if len(peaks_np) else []
px_y_vis = (peaks_np[:, 1] + 0.5) * PATCH_SIZE if len(peaks_np) else []

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    f"Instance Detection  |  DINOv{DINO_VERSION[1]}-{DINO_SIZE}  block {LAYER_IDX}  "
    f"mode={EXEMPLAR_MODE}  threshold={DENSITY_THRESHOLD}  masked={EXEMPLAR_MASK_PATH is not None}",
    fontsize=10,
)

# Exemplar with mask
axes[0].imshow(exemplar_img)
if pixel_mask is not None:
    axes[0].imshow(
        np.array(
            Image.fromarray((pixel_mask.astype(np.uint8) * 255))
            .resize(exemplar_img.size, Image.NEAREST)
        ),
        alpha=0,  # invisible; just shows structure
    )
    # Draw mask contour via imshow with alpha
    mask_big = np.array(
        Image.fromarray(pixel_mask.astype(np.uint8) * 255)
        .resize(exemplar_img.size, Image.NEAREST)
    ) > 0
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
        px_x_vis, px_y_vis,
        c="red", s=120, marker="o",
        linewidths=1.5, edgecolors="white", zorder=5,
    )
axes[2].set_title(f"Detections — {len(peaks_np)} found", fontsize=10)
axes[2].axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Parameter Sweep 1 — Mask Patch Threshold
# 
# The `MASK_PATCH_THRESHOLD` controls how much of a patch's pixels must be True for the
# patch to be included in the exemplar.  Low values include borderline patches; high values
# restrict to the object core.  We sweep this and show how the number of masked patches
# and the resulting density map change.

# %%
# ── Mask threshold sweep ───────────────────────────────────────────────────────
MASK_THRESHOLDS = [0.05, 0.15, 0.30, 0.50, 0.75, 0.95]

fig, axes = plt.subplots(3, len(MASK_THRESHOLDS), figsize=(len(MASK_THRESHOLDS) * 4, 13))

for col, mthr in enumerate(MASK_THRESHOLDS):
    if pixel_mask is not None:
        pm = pixel_mask_to_patch_mask(pixel_mask, ex_h, ex_w, IMG_SIZE, threshold=mthr)
        pm_flat = torch.from_numpy(pm.reshape(-1)).to(exemplar_tokens.device)
        ex_tok_m = exemplar_tokens[pm_flat] if pm_flat.any() else exemplar_tokens
    else:
        pm = np.ones((ex_h, ex_w), dtype=bool)
        ex_tok_m = exemplar_tokens

    feat_m = compute_exemplar_features(ex_tok_m, mode=EXEMPLAR_MODE, k=EXEMPLAR_K)
    dm_m   = compute_density_map(query_tokens, feat_m, q_h, q_w, DENSITY_THRESHOLD)
    pk_m   = extract_peaks(dm_m, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)
    dm_np_m = dm_m.cpu().numpy()

    # Row 0: patch mask
    axes[0, col].imshow(pm.astype(np.float32), cmap="Greens", vmin=0, vmax=1, aspect="auto")
    axes[0, col].set_title(
        f"mask_thr={mthr}\n{pm.sum()}/{ex_h*ex_w} patches", fontsize=9
    )
    axes[0, col].axis("off")

    # Row 1: density map
    im = axes[1, col].imshow(dm_np_m, cmap="jet", aspect="auto")
    axes[1, col].set_title(f"{(dm_np_m > 0).sum()} active patches", fontsize=9)
    axes[1, col].axis("off")
    plt.colorbar(im, ax=axes[1, col], shrink=0.75, pad=0.02)

    # Row 2: detection overlay
    axes[2, col].imshow(display_q0)
    axes[2, col].imshow(heat_overlay(display_q0, upsample_map(dm_np_m, IMG_SIZE)))
    if len(pk_m):
        px_m = (pk_m[:, 0].float() + 0.5) * PATCH_SIZE
        py_m = (pk_m[:, 1].float() + 0.5) * PATCH_SIZE
        axes[2, col].scatter(
            px_m.cpu(), py_m.cpu(),
            c="red", s=80, marker="o", linewidths=1.5, edgecolors="white", zorder=5,
        )
    axes[2, col].set_title(f"{len(pk_m)} peak(s)", fontsize=9)
    axes[2, col].axis("off")

plt.suptitle(
    f"Sweep: mask_patch_threshold  |  mode={EXEMPLAR_MODE}  density_thr={DENSITY_THRESHOLD}",
    fontsize=12, y=1.01,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Parameter Sweep 2 — Density Threshold

# %%
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

fig, axes = plt.subplots(2, len(THRESHOLDS), figsize=(len(THRESHOLDS) * 4, 9))

for col, thr in enumerate(THRESHOLDS):
    dm_thr = compute_density_map(query_tokens, feat_masked, q_h, q_w, threshold=thr)
    pk_thr = extract_peaks(dm_thr, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)
    dm_np  = dm_thr.cpu().numpy()
    pk_np  = pk_thr.cpu().numpy()

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
            c="red", s=80, marker="o", linewidths=1.5, edgecolors="white", zorder=5,
        )
    axes[1, col].set_title(f"{len(pk_np)} peak(s)", fontsize=9)
    axes[1, col].axis("off")

plt.suptitle(
    f"Sweep: density_threshold  |  mode={EXEMPLAR_MODE}  masked exemplar",
    fontsize=12, y=1.01,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Parameter Sweep 3 — Transformer Block (Layer Index)

# %%
LAYER_INDICES = [6, 12, 16, 20, 22, 23]

fig, axes = plt.subplots(2, len(LAYER_INDICES), figsize=(len(LAYER_INDICES) * 4, 9))

for col, layer in enumerate(LAYER_INDICES):
    ex_tok_l, ex_h_l, ex_w_l = extract_patch_tokens(encoder, exemplar_img, layer)
    q_tok_l,  q_h_l,  q_w_l  = extract_patch_tokens(encoder, query_imgs[0], layer)

    # Re-project mask to potentially different grid (same size here, kept for correctness)
    if pixel_mask is not None:
        pm_l     = pixel_mask_to_patch_mask(pixel_mask, ex_h_l, ex_w_l, IMG_SIZE, MASK_PATCH_THRESHOLD)
        pm_flat_l = torch.from_numpy(pm_l.reshape(-1)).to(ex_tok_l.device)
        ex_tok_l_m = ex_tok_l[pm_flat_l] if pm_flat_l.any() else ex_tok_l
    else:
        ex_tok_l_m = ex_tok_l

    feat_l = compute_exemplar_features(ex_tok_l_m, mode=EXEMPLAR_MODE, k=EXEMPLAR_K)
    dm_l   = compute_density_map(q_tok_l, feat_l, q_h_l, q_w_l, DENSITY_THRESHOLD)
    pk_l   = extract_peaks(dm_l, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)
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
            c="red", s=80, marker="o", linewidths=1.5, edgecolors="white", zorder=5,
        )
    axes[1, col].set_title(f"{len(pk_l)} peak(s)", fontsize=9)
    axes[1, col].axis("off")

plt.suptitle(
    f"Sweep: layer_idx  |  mode={EXEMPLAR_MODE}  masked exemplar  density_thr={DENSITY_THRESHOLD}",
    fontsize=12, y=1.01,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Parameter Sweep 4 — Aggregation Mode (mean vs K-means)

# %%
MODE_CONFIGS: list[tuple[str, int]] = [
    ("mean",   1),
    ("kmeans", 2),
    ("kmeans", 3),
    ("kmeans", 5),
]

fig, axes = plt.subplots(2, len(MODE_CONFIGS), figsize=(len(MODE_CONFIGS) * 5, 10))

for col, (mode, k) in enumerate(MODE_CONFIGS):
    feat_m = compute_exemplar_features(exemplar_tokens_masked, mode=mode, k=k)
    dm_m   = compute_density_map(query_tokens, feat_m, q_h, q_w, DENSITY_THRESHOLD)
    pk_m   = extract_peaks(dm_m, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)
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
            c="red", s=80, marker="o", linewidths=1.5, edgecolors="white", zorder=5,
        )
    axes[1, col].set_title(f"{len(pk_m)} peak(s)", fontsize=9)
    axes[1, col].axis("off")

plt.suptitle(
    f"Sweep: aggregation mode  |  block={LAYER_IDX}  density_thr={DENSITY_THRESHOLD}  masked exemplar",
    fontsize=12, y=1.01,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Multi-Query Detection
# 
# Run the full (masked) pipeline across all query images using the same exemplar.

# %%
# ── Run on all queries ─────────────────────────────────────────────────────────
query_results: list[dict] = []
for path, q_img in zip(QUERY_PATHS, query_imgs):
    q_tok_f, q_h_f, q_w_f = extract_patch_tokens(encoder, q_img, LAYER_IDX)
    dm_f = compute_density_map(q_tok_f, feat_masked, q_h_f, q_w_f, DENSITY_THRESHOLD)
    pk_f = extract_peaks(dm_f, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)
    query_results.append({"path": path, "img": q_img, "density": dm_f, "peaks": pk_f})
    log.info("%s → %d peak(s)  density max=%.3f",
             path.split("/")[-1], len(pk_f), dm_f.max().item())

nq = len(query_results)
fig, axes = plt.subplots(3, nq, figsize=(nq * 5, 15))
if nq == 1:
    axes = axes.reshape(3, 1)

for col, qr in enumerate(query_results):
    dm_np_f = qr["density"].cpu().numpy()
    pk_np_f = qr["peaks"].cpu().numpy()
    disp_qf = np.array(qr["img"].resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))

    axes[0, col].imshow(qr["img"])
    axes[0, col].set_title(qr["path"].split("/")[-1], fontsize=8)
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
            c="red", s=120, marker="o", linewidths=1.5, edgecolors="white", zorder=5,
        )
    axes[2, col].set_title(f"{len(pk_np_f)} peak(s)", fontsize=9)
    axes[2, col].axis("off")

plt.suptitle(
    f"Multi-query  |  DINOv{DINO_VERSION[1]}-{DINO_SIZE}  block {LAYER_IDX}  "
    f"mode={EXEMPLAR_MODE}  masked exemplar",
    fontsize=11, y=1.01,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Peak Self-Similarity Heatmap Drilldown
# 
# For each detected peak in query 1, compute the cosine similarity of that peak's
# feature vector against every other query patch.  This shows what the model is
# "matching" against — the instance-detection analogue of the keypoint heatmap.

# %%
# ── Drilldown on query 1 ───────────────────────────────────────────────────────
if len(peaks) == 0:
    log.warning("No peaks in query 1 — lower DENSITY_THRESHOLD to see heatmaps.")
else:
    n_peaks   = len(peaks)
    disp_q_d  = np.array(query_imgs[0].resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))

    fig, axes = plt.subplots(2, n_peaks, figsize=(n_peaks * 4, 8))
    if n_peaks == 1:
        axes = axes.reshape(2, 1)

    for col, peak in enumerate(peaks):
        px, py    = peak[0].item(), peak[1].item()
        peak_feat = query_tokens[py * q_w + px]  # (D,)

        sim_self  = (query_tokens @ peak_feat).reshape(q_h, q_w).cpu().numpy()
        heat_self = upsample_map(sim_self, IMG_SIZE)
        blended   = np.clip(disp_q_d / 255.0 * 0.45 + plt.get_cmap("jet")(heat_self)[..., :3] * 0.55, 0, 1)

        im = axes[0, col].imshow(sim_self, cmap="jet", vmin=sim_self.min(), vmax=1.0, aspect="auto")
        axes[0, col].set_title(
            f"Peak ({px},{py}) self-sim\nscore={density_np[py, px]:.3f}", fontsize=9
        )
        axes[0, col].axis("off")
        plt.colorbar(im, ax=axes[0, col], shrink=0.75, pad=0.02)

        axes[1, col].imshow(blended)
        axes[1, col].scatter(
            [(px + 0.5) * PATCH_SIZE], [(py + 0.5) * PATCH_SIZE],
            c="red", s=120, marker="o", linewidths=1.5, edgecolors="white", zorder=5,
        )
        axes[1, col].set_title("Overlay (heat + query)", fontsize=9)
        axes[1, col].axis("off")

    plt.suptitle(
        f"Peak self-similarity heatmaps  |  query 1  |  block {LAYER_IDX}",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.show()

# %%


# %%



