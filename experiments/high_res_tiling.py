# %% [markdown]
# # High-Resolution Tiling Experiment
#
# Process images at 2× the encoder's native resolution (base=1024) by splitting
# into a 2×2 grid of tiles, encoding each separately, and stitching the resulting
# feature maps.  Explores the effect of tile overlap and blending strategy on
# feature quality, seam artefacts, and downstream instance detection.
#
# Experiments covered:
#   1.  Baseline single-pass at 1024 vs tiled 2×2 at 2048 (no overlap)
#   2.  PCA feature visualisation — baseline vs tiled
#   3.  Seam quality analysis — cosine-sim continuity across tile boundaries
#   4.  Overlap sweep — vary overlap_px from 0 to 256
#   5.  Blending method comparison — hard / linear / gaussian (fixed overlap)
#   6.  Self-similarity comparison — how consistent are features across scales?
#   7.  Instance detection — baseline vs tiled vs full-size 1024 encoder density map comparison
#   8.  Throughput — wall time for n baseline / n tiled (batched) / n full-size runs
#   9.  encode_tiled breakdown — wall time per major step (resize, crop, encoder, accumulate, normalize)

# %% Logging — must be before torch import
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("high_res_tiling")

import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from sklearn.decomposition import PCA

from dinoisawesome import DinoEncoder
from dinoisawesome.annotation_utils import load_annotations
from dinoisawesome.instance_detection import (
    compute_density_map,
    compute_exemplar_features,
    extract_patch_tokens,
    extract_peaks,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared.mask_geometry import pixel_mask_to_patch_mask  # noqa: E402

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".env")

# Image to experiment on — set to a local path or leave None to download
IMAGE_PATH: Path | None = _REPO_ROOT / "data" / "abc3" / "LHa_1.jpg"

# Optional exemplar mask for instance-detection comparison (.npy / .npz)
MASK_PATH: Path | None = _REPO_ROOT / "data" / "abc3" / "annotations" / "LHa_1"

# Exp 7.1 — inference image (LHa_1 remains the reference/exemplar)
LHA_2_PATH: Path | None = _REPO_ROOT / "data" / "abc3" / "LHa_2.jpg"

# Encoder
DINO_VERSION = "v3"  # "v2" or "v3"
DINO_SIZE = "large"  # "small" | "base" | "large" | "giant"
BASE_IMG_SIZE = 512  # encoder's fixed input size; "1× resolution" baseline
LAYER_IDX = 23  # transformer block for patch-token extraction
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")

# Tiling
N_TILES = 3  # n_tiles × n_tiles grid (2 → 4 tiles, 3 → 9 tiles)
EFFECTIVE_SIZE = BASE_IMG_SIZE * N_TILES  # 2048 — image is scaled to this before tiling

# Overlap sweep
OVERLAP_VALUES = [0, 16, 32, 64, 128]  # pixels of context added to each tile boundary
DEFAULT_OVERLAP = 64  # overlap used for blending comparison

# Blending methods to compare
BLEND_METHODS = ["hard", "linear", "gaussian"]

# Instance detection (for section 7)
DENSITY_THRESHOLD = 0.5
PEAK_KERNEL_SIZE = 5
MIN_PEAK_THRESHOLD = 0.5
MASK_PATCH_THRESHOLD = 0.3
EXEMPLAR_MODE = "mean"

# Timing (Experiment 8)
N_TIMING_RUNS = 30  # timed forward passes per method
N_TIMING_WARMUP = 10  # warmup passes (discarded) before timing starts

OUTPUT_DIR = _REPO_ROOT / "results" / "high_res_tiling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% Core tiling function

# ImageNet normalisation constants for the GPU preprocess path
_IMAGENET_MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD_T = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def encode_tiled(
    image: "Image.Image | np.ndarray | torch.Tensor",
    encoder: DinoEncoder,
    effective_size: int,
    n_tiles: int = 2,
    overlap_px: int = 0,
    blend: str = "hard",
    debias: bool = True,
    resize_backend: str = "pil",
) -> torch.Tensor:
    """Encode image at `effective_size` resolution via n_tiles×n_tiles tiling.

    Each tile is extracted as a `(stride + 2*overlap_px)` × `(stride + 2*overlap_px)`
    crop from the effective-size image and fed to the encoder (which resizes to
    `img_size`).  The resulting patch grids are scatter-accumulated into a
    `(n_tiles*H_grid, n_tiles*W_grid, D)` output with per-tile blending weights.

    Blending weights:
      - "hard"     : uniform (1.0 everywhere); dominated by whatever tile lands last
                     when multiple tiles map to the same output patch.
      - "linear"   : distance-from-tile-centre triangular falloff.
      - "gaussian" : Gaussian falloff with σ = tile_size / 4.

    resize_backend controls how the full image is scaled to `effective_size`:
      - "pil"       : PIL BICUBIC (original behaviour).
      - "cv2_cubic" : OpenCV INTER_CUBIC (~2–4× faster than PIL, same quality).
      - "cv2_linear": OpenCV INTER_LINEAR (fastest CPU path).
      - "gpu"       : GPU bilinear via F.interpolate; also resizes/normalises
                      each crop on GPU, bypassing the encoder's internal preprocess.
                      Accepts a (3, H, W) float [0, 1] GPU tensor, a PIL Image,
                      or a (H, W, 3) uint8 ndarray.

    Returns (n_tiles*H_grid, n_tiles*W_grid, D) float32 CPU tensor, L2-normalised.
    """
    H_t = encoder.grid_h
    W_t = encoder.grid_w
    stride_px = effective_size // n_tiles
    D = encoder.backbone.embed_dim
    H_out = n_tiles * H_t
    W_out = n_tiles * W_t
    _dev = encoder.device

    # --- Tile coordinate metadata (identical for every backend) ---
    tile_meta: list[tuple[int, int, int, int, int, int]] = []
    for row in range(n_tiles):
        for col in range(n_tiles):
            y0 = max(0, row * stride_px - overlap_px)
            y1 = min(effective_size, (row + 1) * stride_px + overlap_px)
            x0 = max(0, col * stride_px - overlap_px)
            x1 = min(effective_size, (col + 1) * stride_px + overlap_px)
            tile_meta.append((y0, y1, x0, x1, y1 - y0, x1 - x0))

    # --- Backend-specific: resize full image + build encoder input ---
    if resize_backend in ("pil", "cv2_cubic", "cv2_linear"):
        if resize_backend == "pil":
            if not isinstance(image, Image.Image):
                if isinstance(image, torch.Tensor):
                    image = Image.fromarray(
                        (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    )
                else:
                    image = Image.fromarray(image)
            img_arr = np.array(image.resize((effective_size, effective_size), Image.BICUBIC))
        else:
            if isinstance(image, Image.Image):
                img_arr = np.array(image)
            elif isinstance(image, torch.Tensor):
                img_arr = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            else:
                img_arr = image
            interp = cv2.INTER_CUBIC if resize_backend == "cv2_cubic" else cv2.INTER_LINEAR
            img_arr = cv2.resize(img_arr, (effective_size, effective_size), interpolation=interp)
        crops: list[Image.Image] = [
            Image.fromarray(img_arr[y0:y1, x0:x1]) for y0, y1, x0, x1, _, _ in tile_meta
        ]
        out = encoder(crops, debias=debias)

    elif resize_backend == "gpu":
        # Convert to (3, H, W) float [0, 1] on device
        if isinstance(image, Image.Image):
            img_t = torch.from_numpy(
                np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
            ).to(_dev)
        elif isinstance(image, np.ndarray):
            img_t = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0).to(_dev)
        else:
            img_t = image.float().to(_dev)
        # Resize full image on GPU
        img_big_t = F.interpolate(
            img_t.unsqueeze(0),
            size=(effective_size, effective_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # (3, effective_size, effective_size)
        # Per-crop: slice → resize to encoder.img_size → ImageNet-normalise → stack
        _mean = _IMAGENET_MEAN_T.to(_dev)
        _std = _IMAGENET_STD_T.to(_dev)
        batch_t = torch.stack(
            [
                (
                    F.interpolate(
                        img_big_t[:, y0:y1, x0:x1].unsqueeze(0),
                        size=(encoder.img_size, encoder.img_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    - _mean
                )
                / _std
                for y0, y1, x0, x1, _, _ in tile_meta
            ]
        )  # (B, 3, img_size, img_size) — pre-processed, bypasses encoder.preprocess
        out = encoder(batch_t, debias=debias)

    else:
        raise ValueError(f"Unknown resize_backend: {resize_backend!r}")

    tile_feats = out.patches.cpu().float()  # (B, H_t, W_t, D)

    # --- Output accumulator and blend weight map (same for every backend) ---
    acc = torch.zeros(H_out, W_out, D, dtype=torch.float32)
    wgt = torch.zeros(H_out, W_out, dtype=torch.float32)

    ty_idx = torch.arange(H_t, dtype=torch.float32)
    tx_idx = torch.arange(W_t, dtype=torch.float32)
    TY, TX = torch.meshgrid(ty_idx, tx_idx, indexing="ij")  # (H_t, W_t)

    if blend == "hard":
        W_map = torch.ones(H_t, W_t)
    elif blend == "linear":
        bdy = 1.0 - (TY - (H_t - 1) / 2.0).abs() / (H_t / 2.0)
        bdx = 1.0 - (TX - (W_t - 1) / 2.0).abs() / (W_t / 2.0)
        W_map = (bdy * bdx).clamp(min=0.0)
    elif blend == "gaussian":
        sigma_y = H_t / 4.0
        sigma_x = W_t / 4.0
        bdy = torch.exp(-0.5 * ((TY - (H_t - 1) / 2.0) / sigma_y) ** 2)
        bdx = torch.exp(-0.5 * ((TX - (W_t - 1) / 2.0) / sigma_x) ** 2)
        W_map = bdy * bdx
    else:
        raise ValueError(f"Unknown blend method: {blend!r}")

    # --- Pass 2: accumulate each tile's patches into the output grid ---
    for tile_feat, (y0, y1, x0, x1, tile_h, tile_w) in zip(tile_feats, tile_meta):
        # Map each tile patch to its canonical output position (float)
        gc_y = y0 + (TY + 0.5) * tile_h / H_t  # global pixel centre (H_t, W_t)
        gc_x = x0 + (TX + 0.5) * tile_w / W_t
        oy_f = (gc_y * H_out / effective_size).clamp(0, H_out - 1)
        ox_f = (gc_x * W_out / effective_size).clamp(0, W_out - 1)
        flat_w = W_map.reshape(-1)
        flat_f = tile_feat.reshape(H_t * W_t, D)  # (H_t*W_t, D)
        # Bilinear splatting: distribute each patch to its 4 surrounding output
        # positions.  This prevents "holes" (zero-weight positions) that appear
        # when tile_h > stride_px (overlap > 0) and the floor-based nearest-
        # neighbour mapping skips output rows/columns periodically.
        oy_lo = oy_f.long().clamp(0, H_out - 1)
        oy_hi = (oy_lo + 1).clamp(0, H_out - 1)
        ox_lo = ox_f.long().clamp(0, W_out - 1)
        ox_hi = (ox_lo + 1).clamp(0, W_out - 1)
        frac_y = (oy_f - oy_lo.float()).clamp(0.0, 1.0)
        frac_x = (ox_f - ox_lo.float()).clamp(0.0, 1.0)
        for OY, OX, corner_w in [
            (oy_lo, ox_lo, (1.0 - frac_y) * (1.0 - frac_x)),
            (oy_lo, ox_hi, (1.0 - frac_y) * frac_x),
            (oy_hi, ox_lo, frac_y * (1.0 - frac_x)),
            (oy_hi, ox_hi, frac_y * frac_x),
        ]:
            flat_oy = OY.reshape(-1)
            flat_ox = OX.reshape(-1)
            combined_w = flat_w * corner_w.reshape(-1)
            flat_idx = flat_oy * W_out + flat_ox
            wgt.reshape(-1).scatter_add_(0, flat_idx, combined_w)
            acc.reshape(-1, D).scatter_add_(
                0,
                flat_idx.unsqueeze(1).expand(-1, D),
                combined_w.unsqueeze(1) * flat_f,
            )

    # Normalise by accumulated weights then L2-normalise each patch vector
    mask = wgt > 0
    acc[mask] /= wgt[mask, None]
    return F.normalize(acc, p=2, dim=-1)


def pca_rgb(feat_map: torch.Tensor, pca_fit: PCA | None = None) -> tuple[np.ndarray, PCA]:
    """Project (H, W, D) feature map to (H, W, 3) RGB via PCA."""
    H, W, D = feat_map.shape
    flat = feat_map.reshape(-1, D).cpu().float().numpy()
    if pca_fit is None:
        pca_fit = PCA(n_components=3)
        pca_fit.fit(flat)
    proj = pca_fit.transform(flat)
    for c in range(3):
        lo, hi = proj[:, c].min(), proj[:, c].max()
        proj[:, c] = (proj[:, c] - lo) / (hi - lo + 1e-8)
    return proj.reshape(H, W, 3), pca_fit


def to_img(arr: np.ndarray, size: int | None = None) -> np.ndarray:
    """Nearest-neighbour upsample a (H, W, C) or (H, W) float array to `size×size`."""
    if arr.ndim == 2:
        pil = Image.fromarray((arr * 255).astype(np.uint8))
    else:
        pil = Image.fromarray((arr * 255).astype(np.uint8))
    if size is not None and (arr.shape[0] != size or arr.shape[1] != size):
        pil = pil.resize((size, size), Image.NEAREST)
    return np.array(pil) / 255.0


# %% Load encoder and image
encoder = DinoEncoder(
    version=DINO_VERSION,
    size=DINO_SIZE,
    img_size=BASE_IMG_SIZE,
    weights_dir=DINO_WEIGHTS_DIR,
    amp=True,
)
H_GRID = encoder.grid_h  # 64 for v3 at 1024
W_GRID = encoder.grid_w
PATCH_SIZE = encoder.patch_size
D = encoder.backbone.embed_dim

log.info(
    "DINOv%s-%s | patch_size=%d | grid=%dx%d | D=%d",
    DINO_VERSION[1],
    DINO_SIZE,
    PATCH_SIZE,
    H_GRID,
    W_GRID,
    D,
)
log.info(
    "Tiling: %d×%d tiles | effective size %dx%d → %dx%d patch grid",
    N_TILES,
    N_TILES,
    EFFECTIVE_SIZE,
    EFFECTIVE_SIZE,
    N_TILES * H_GRID,
    N_TILES * W_GRID,
)

img = Image.open(IMAGE_PATH).convert("RGB")
log.info("Loaded: %s  (%dx%d px)", IMAGE_PATH, *img.size)

display_base = np.array(img.resize((BASE_IMG_SIZE, BASE_IMG_SIZE), Image.BICUBIC))
display_2x = np.array(img.resize((EFFECTIVE_SIZE, EFFECTIVE_SIZE), Image.BICUBIC))

fig, axes = plt.subplots(1, 2, figsize=(15, 10))
axes[0].imshow(display_base)
axes[0].set_title(f"Baseline input  ({BASE_IMG_SIZE}×{BASE_IMG_SIZE})")
axes[0].axis("off")
axes[1].imshow(display_2x)
axes[1].set_title(f"Effective 2× input  ({EFFECTIVE_SIZE}×{EFFECTIVE_SIZE})")
axes[1].axis("off")
plt.suptitle("Input image at both resolutions", fontsize=12)
plt.tight_layout()
plt.show()

# %% Experiment 1 — Baseline vs tiled (no overlap) feature maps
log.info("Experiment 1: Baseline 1× vs Tiled 2× (overlap=0) …")

# Baseline: single forward pass at BASE_IMG_SIZE
# v3 large has some bias in later layers that can dominate PCA; set to False to disable
with_debias = True
out_base = encoder(img, debias=with_debias)
feat_base = out_base.patches[0].cpu().float()  # (H_GRID, W_GRID, D)
feat_base = F.normalize(feat_base, p=2, dim=-1)

# Tiled 2×: 4 tiles, no overlap
feat_tiled = encode_tiled(
    img, encoder, EFFECTIVE_SIZE, n_tiles=N_TILES, overlap_px=0, blend="hard", debias=with_debias
)
# feat_tiled: (2*H_GRID, 2*W_GRID, D)

log.info("Baseline feature map : %s", tuple(feat_base.shape))
log.info("Tiled feature map    : %s", tuple(feat_tiled.shape))

# PCA fitted on baseline; applied to both (same feature space since same model)
rgb_base, pca_obj = pca_rgb(feat_base)
rgb_tiled, _ = pca_rgb(feat_tiled, pca_obj)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(display_base)
axes[0, 0].set_title(f"Image @ {BASE_IMG_SIZE}×{BASE_IMG_SIZE}", fontsize=10)
axes[0, 0].axis("off")

axes[0, 1].imshow(rgb_base)
axes[0, 1].set_title(f"Baseline PCA  ({H_GRID}×{W_GRID} patches)", fontsize=10)
axes[0, 1].axis("off")

axes[0, 2].imshow(to_img(rgb_base, BASE_IMG_SIZE))
axes[0, 2].set_title(f"Baseline PCA upsampled to {BASE_IMG_SIZE}px", fontsize=10)
axes[0, 2].axis("off")

axes[1, 0].imshow(display_2x)
axes[1, 0].set_title(f"Image @ {EFFECTIVE_SIZE}×{EFFECTIVE_SIZE}", fontsize=10)
axes[1, 0].axis("off")

axes[1, 1].imshow(rgb_tiled)
axes[1, 1].set_title(f"Tiled 2× PCA  ({N_TILES * H_GRID}×{N_TILES * W_GRID} patches)", fontsize=10)
axes[1, 1].axis("off")

axes[1, 2].imshow(to_img(rgb_tiled, EFFECTIVE_SIZE))
axes[1, 2].set_title(f"Tiled 2× PCA upsampled to {EFFECTIVE_SIZE}px", fontsize=10)
axes[1, 2].axis("off")

plt.suptitle(
    f"Exp 1: Baseline vs Tiled  |  DINOv{DINO_VERSION[1]}-{DINO_SIZE}  block {LAYER_IDX}",
    fontsize=12,
    y=1.01,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exp1_baseline_vs_tiled.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Experiment 2 — Tile boundary grid overlay
log.info("Experiment 2: Tile grid overlay …")

tile_grid_overlay = display_2x.copy()
stride_show = EFFECTIVE_SIZE // N_TILES
for i in range(1, N_TILES):
    # Vertical seam
    tile_grid_overlay[:, stride_show * i - 1 : stride_show * i + 1] = [255, 0, 0]
    # Horizontal seam
    tile_grid_overlay[stride_show * i - 1 : stride_show * i + 1, :] = [255, 0, 0]

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(tile_grid_overlay)
axes[0].set_title(f"Tile grid ({N_TILES}×{N_TILES}) — seams in red", fontsize=10)
axes[0].axis("off")

# Show which patches belong to which tile
tile_label_map = np.zeros((N_TILES * H_GRID, N_TILES * W_GRID), dtype=int)
for r in range(N_TILES):
    for c in range(N_TILES):
        tile_label_map[r * H_GRID : (r + 1) * H_GRID, c * W_GRID : (c + 1) * W_GRID] = (
            r * N_TILES + c
        )

cmap_tiles = plt.get_cmap("tab10")
tile_rgb = cmap_tiles(tile_label_map / (N_TILES * N_TILES))[..., :3]
axes[1].imshow(to_img(tile_rgb, EFFECTIVE_SIZE))
axes[1].set_title(f"Patch ownership map ({N_TILES * H_GRID}×{N_TILES * W_GRID})", fontsize=10)
axes[1].axis("off")

plt.suptitle("Exp 2: Tile structure", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exp2_tile_structure.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Experiment 3 — Seam quality analysis (no overlap baseline)
log.info("Experiment 3: Seam quality analysis …")

feat_no_ov = feat_tiled  # (2*H_GRID, 2*W_GRID, D) — already computed above

# Measure cosine similarity between adjacent patches across each seam vs. within tiles
H_OUT, W_OUT = N_TILES * H_GRID, N_TILES * W_GRID


def seam_sim_stats(feat: torch.Tensor, axis: str) -> dict[str, float]:
    """Compare patch similarity across tile seams vs within tiles along `axis`."""
    H, W, _ = feat.shape
    seam_sims = []
    within_sims = []

    if axis == "vertical":
        # Vertical seam: adjacent patches in the column dimension at tile boundary
        seam_cols = [c * W_GRID for c in range(1, N_TILES)]
        for col in seam_cols:
            # sim between col-1 and col for all rows
            left = feat[:, col - 1, :]  # (H, D)
            right = feat[:, col, :]
            sims = (left * right).sum(dim=-1).cpu().numpy()
            seam_sims.extend(sims.tolist())
        # Within-tile: sample same distance but not at seam
        for row in range(H):
            for c in range(1, W - 1):
                seam_cols = [c * W_GRID for c in range(1, N_TILES)]
                is_seam = any(c == sc or c - 1 == sc for sc in seam_cols)
                if not is_seam:
                    s = float((feat[row, c - 1, :] * feat[row, c, :]).sum())
                    within_sims.append(s)
    else:  # horizontal
        seam_rows = [r * H_GRID for r in range(1, N_TILES)]
        for row in seam_rows:
            top = feat[row - 1, :, :]
            bottom = feat[row, :, :]
            sims = (top * bottom).sum(dim=-1).cpu().numpy()
            seam_sims.extend(sims.tolist())
        for col in range(W):
            for r in range(1, H - 1):
                seam_rows = [r * H_GRID for r in range(1, N_TILES)]
                is_seam = any(r == sr or r - 1 == sr for sr in seam_rows)
                if not is_seam:
                    s = float((feat[r - 1, col, :] * feat[r, col, :]).sum())
                    within_sims.append(s)

    return {
        "seam_mean": float(np.mean(seam_sims)) if seam_sims else float("nan"),
        "seam_std": float(np.std(seam_sims)) if seam_sims else float("nan"),
        "within_mean": float(np.mean(within_sims[:10000])) if within_sims else float("nan"),
        "within_std": float(np.std(within_sims[:10000])) if within_sims else float("nan"),
    }


stats_v = seam_sim_stats(feat_no_ov, "vertical")
stats_h = seam_sim_stats(feat_no_ov, "horizontal")

log.info("Seam quality (no overlap):")
log.info(
    "  Vertical seam:   seam=%.3f±%.3f  within=%.3f±%.3f",
    stats_v["seam_mean"],
    stats_v["seam_std"],
    stats_v["within_mean"],
    stats_v["within_std"],
)
log.info(
    "  Horizontal seam: seam=%.3f±%.3f  within=%.3f±%.3f",
    stats_h["seam_mean"],
    stats_h["seam_std"],
    stats_h["within_mean"],
    stats_h["within_std"],
)

pd.DataFrame([{"axis": "vertical", **stats_v}, {"axis": "horizontal", **stats_h}]).to_csv(
    OUTPUT_DIR / "exp3_seam_quality.csv", index=False
)
log.info("Wrote %s", OUTPUT_DIR / "exp3_seam_quality.csv")

# Visualise the cross-patch similarity map
H_F, W_F, _ = feat_no_ov.shape
sim_across_h = torch.zeros(H_F, W_F - 1)
for col in range(W_F - 1):
    sim_across_h[:, col] = (feat_no_ov[:, col, :] * feat_no_ov[:, col + 1, :]).sum(dim=-1).cpu()

sim_across_v = torch.zeros(H_F - 1, W_F)
for row in range(H_F - 1):
    sim_across_v[row, :] = (feat_no_ov[row, :, :] * feat_no_ov[row + 1, :, :]).sum(dim=-1).cpu()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

im0 = axes[0].imshow(sim_across_h.numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
for col in [c * W_GRID - 1 for c in range(1, N_TILES)]:
    axes[0].axvline(col, color="red", linewidth=1.5, linestyle="--", alpha=0.8, label="tile seam")
axes[0].set_title("Horizontal adjacent-patch similarity\n(seams = dashed red)", fontsize=10)
axes[0].set_xlabel("column")
axes[0].set_ylabel("row")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(sim_across_v.numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
for row in [r * H_GRID - 1 for r in range(1, N_TILES)]:
    axes[1].axhline(row, color="red", linewidth=1.5, linestyle="--", alpha=0.8, label="tile seam")
axes[1].set_title("Vertical adjacent-patch similarity\n(seams = dashed red)", fontsize=10)
axes[1].set_xlabel("column")
plt.colorbar(im1, ax=axes[1], shrink=0.8)

plt.suptitle("Exp 3: Seam quality — are tile boundaries visible as discontinuities?", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exp3_seam_quality.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Experiment 4 — Overlap sweep
log.info("Experiment 4: Overlap sweep %s px …", OVERLAP_VALUES)

# Effective receptive field per patch as a function of overlap.
# The image goes through three resizes:
#   original → effective_size (upscale) → tile crop (clamped) → encoder native size.
# Each patch token covers (tile_crop_px / H_t) pixels of the effective image,
# which maps back to (tile_crop_px / H_t) * (orig_h / effective_size) original pixels.
# Edge tiles only get overlap on one side (max(0,...) / min(eff,...) clamps), so the
# actual crop size varies by position — we simulate all tile positions to find the range.
_orig_w, _orig_h = img.size
_H_t = encoder.grid_h  # patch rows per tile (e.g. 73 for v3-large at 1024)
_stride_px = EFFECTIVE_SIZE // N_TILES
log.info(
    "  ERF breakdown — orig=%dx%d  effective=%d  n_tiles=%d  stride_px=%d  patches_per_tile=%dx%d",
    _orig_w,
    _orig_h,
    EFFECTIVE_SIZE,
    N_TILES,
    _stride_px,
    _H_t,
    encoder.grid_w,
)
log.info(
    "  %8s  %14s  %14s  %12s  %12s",
    "overlap",
    "crop_min_px",
    "crop_max_px",
    "erf_min_px",
    "erf_max_px",
)
for _ov in OVERLAP_VALUES:
    _crops: list[int] = []
    for _r in range(N_TILES):
        for _c in range(N_TILES):
            _y0 = max(0, _r * _stride_px - _ov)
            _y1 = min(EFFECTIVE_SIZE, (_r + 1) * _stride_px + _ov)
            _x0 = max(0, _c * _stride_px - _ov)
            _x1 = min(EFFECTIVE_SIZE, (_c + 1) * _stride_px + _ov)
            _crops.append(_y1 - _y0)
            _crops.append(_x1 - _x0)
    _crop_min, _crop_max = min(_crops), max(_crops)
    _erf_min = _crop_min / _H_t * (_orig_h / EFFECTIVE_SIZE)
    _erf_max = _crop_max / _H_t * (_orig_h / EFFECTIVE_SIZE)
    log.info("  %8d  %14d  %14d  %12.1f  %12.1f", _ov, _crop_min, _crop_max, _erf_min, _erf_max)

overlap_results: list[dict] = []
for ov in OVERLAP_VALUES:
    log.info("  overlap_px=%d …", ov)
    feat_ov = encode_tiled(
        img,
        encoder,
        EFFECTIVE_SIZE,
        n_tiles=N_TILES,
        overlap_px=ov,
        blend="linear",
        debias=with_debias,
    )
    stats_v_ov = seam_sim_stats(feat_ov, "vertical")
    stats_h_ov = seam_sim_stats(feat_ov, "horizontal")
    rgb_ov, _ = pca_rgb(feat_ov, pca_obj)
    overlap_results.append(
        {
            "overlap_px": ov,
            "feat": feat_ov,
            "rgb": rgb_ov,
            "stats_v": stats_v_ov,
            "stats_h": stats_h_ov,
        }
    )
    log.info(
        "    V seam: %.3f±%.3f  H seam: %.3f±%.3f  (within: %.3f)",
        stats_v_ov["seam_mean"],
        stats_v_ov["seam_std"],
        stats_h_ov["seam_mean"],
        stats_h_ov["seam_std"],
        stats_v_ov["within_mean"],
    )

# Plot: PCA row per overlap value
fig, axes = plt.subplots(len(OVERLAP_VALUES), 3, figsize=(16, 4 * len(OVERLAP_VALUES)))
for row, res in enumerate(overlap_results):
    ov = res["overlap_px"]

    axes[row, 0].imshow(to_img(res["rgb"], EFFECTIVE_SIZE))
    axes[row, 0].set_title(f"overlap={ov}px  PCA", fontsize=9)
    axes[row, 0].axis("off")

    # Horizontal sim map for this overlap
    feat_ov = res["feat"]
    H_o, W_o, _ = feat_ov.shape
    sim_h = torch.zeros(H_o, W_o - 1)
    for col in range(W_o - 1):
        sim_h[:, col] = (feat_ov[:, col, :] * feat_ov[:, col + 1, :]).sum(dim=-1).cpu()

    im = axes[row, 1].imshow(sim_h.numpy(), cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    for col in [c * W_GRID - 1 for c in range(1, N_TILES)]:
        axes[row, 1].axvline(col, color="red", linewidth=1.5, linestyle="--", alpha=0.7)
    axes[row, 1].set_title(
        f"Horiz. adj. sim  seam={res['stats_v']['seam_mean']:.3f}"
        f"  within={res['stats_v']['within_mean']:.3f}",
        fontsize=9,
    )
    axes[row, 1].axis("off")
    plt.colorbar(im, ax=axes[row, 1], shrink=0.8)

    # Bar chart: seam vs within
    cats = ["seam V", "seam H", "within"]
    vals = [
        res["stats_v"]["seam_mean"],
        res["stats_h"]["seam_mean"],
        res["stats_v"]["within_mean"],
    ]
    errs = [
        res["stats_v"]["seam_std"],
        res["stats_h"]["seam_std"],
        res["stats_v"]["within_std"],
    ]
    colors = ["#e74c3c", "#e67e22", "#2ecc71"]
    axes[row, 2].bar(cats, vals, yerr=errs, color=colors, capsize=4)
    axes[row, 2].set_ylim(0, 1)
    axes[row, 2].set_ylabel("cosine similarity")
    axes[row, 2].set_title(f"Similarity stats  overlap={ov}px", fontsize=9)
    axes[row, 2].axhline(res["stats_v"]["within_mean"], color="#2ecc71", linestyle="--", alpha=0.5)
    axes[row, 2].grid(axis="y", alpha=0.3)

plt.suptitle(
    f"Exp 4: Overlap sweep  |  blend=linear  |  DINOv{DINO_VERSION[1]}-{DINO_SIZE}",
    fontsize=13,
    y=1.01,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exp4_overlap_sweep.png", dpi=150, bbox_inches="tight")
plt.show()

# Summary: seam sim vs overlap
ov_vals = [r["overlap_px"] for r in overlap_results]
seam_v_mu = [r["stats_v"]["seam_mean"] for r in overlap_results]
seam_h_mu = [r["stats_h"]["seam_mean"] for r in overlap_results]
within_mu = [r["stats_v"]["within_mean"] for r in overlap_results]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ov_vals, seam_v_mu, "o-", color="#e74c3c", label="seam (vertical)")
ax.plot(ov_vals, seam_h_mu, "s-", color="#e67e22", label="seam (horizontal)")
ax.plot(ov_vals, within_mu, "--", color="#2ecc71", label="within-tile")
ax.set_xlabel("overlap_px")
ax.set_ylabel("Mean cosine similarity (adjacent patches)")
ax.set_title("Seam quality vs overlap")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "exp4_seam_vs_overlap.png", dpi=150)
plt.show()

overlap_summary_df = pd.DataFrame(
    [
        {
            "overlap_px": r["overlap_px"],
            "seam_v_mean": r["stats_v"]["seam_mean"],
            "seam_v_std": r["stats_v"]["seam_std"],
            "seam_h_mean": r["stats_h"]["seam_mean"],
            "seam_h_std": r["stats_h"]["seam_std"],
            "within_mean": r["stats_v"]["within_mean"],
            "within_std": r["stats_v"]["within_std"],
        }
        for r in overlap_results
    ]
)
overlap_summary_df.to_csv(OUTPUT_DIR / "exp4_overlap_sweep.csv", index=False)
log.info("Wrote %s", OUTPUT_DIR / "exp4_overlap_sweep.csv")

log.info("Exp 4 complete.")

# %% Experiment 5 — Blending method comparison (fixed overlap)
log.info("Experiment 5: Blending comparison at overlap=%d px …", DEFAULT_OVERLAP)

blend_results: list[dict] = {}
for blend in BLEND_METHODS:
    log.info("  blend=%s …", blend)
    feat_bl = encode_tiled(
        img,
        encoder,
        EFFECTIVE_SIZE,
        n_tiles=N_TILES,
        overlap_px=DEFAULT_OVERLAP,
        blend=blend,
    )
    stats_v_bl = seam_sim_stats(feat_bl, "vertical")
    stats_h_bl = seam_sim_stats(feat_bl, "horizontal")
    rgb_bl, _ = pca_rgb(feat_bl, pca_obj)
    blend_results[blend] = {
        "feat": feat_bl,
        "rgb": rgb_bl,
        "stats_v": stats_v_bl,
        "stats_h": stats_h_bl,
    }
    log.info(
        "    V seam: %.3f±%.3f  H seam: %.3f±%.3f",
        stats_v_bl["seam_mean"],
        stats_v_bl["seam_std"],
        stats_h_bl["seam_mean"],
        stats_h_bl["seam_std"],
    )

fig, axes = plt.subplots(len(BLEND_METHODS), 3, figsize=(16, 4 * len(BLEND_METHODS)))
for row, blend in enumerate(BLEND_METHODS):
    res = blend_results[blend]
    feat_bl = res["feat"]
    H_b, W_b, _ = feat_bl.shape

    axes[row, 0].imshow(to_img(res["rgb"], EFFECTIVE_SIZE))
    axes[row, 0].set_title(f"blend={blend}  PCA", fontsize=9)
    axes[row, 0].axis("off")

    sim_h_bl = torch.zeros(H_b, W_b - 1)
    for col in range(W_b - 1):
        sim_h_bl[:, col] = (feat_bl[:, col, :] * feat_bl[:, col + 1, :]).sum(dim=-1).cpu()

    im = axes[row, 1].imshow(sim_h_bl.numpy(), cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    for col in [c * W_GRID - 1 for c in range(1, N_TILES)]:
        axes[row, 1].axvline(col, color="red", linewidth=1.5, linestyle="--", alpha=0.7)
    axes[row, 1].set_title(
        f"Horiz. adj. sim  seam={res['stats_v']['seam_mean']:.3f}",
        fontsize=9,
    )
    axes[row, 1].axis("off")
    plt.colorbar(im, ax=axes[row, 1], shrink=0.8)

    cats = ["seam V", "seam H", "within"]
    vals = [
        res["stats_v"]["seam_mean"],
        res["stats_h"]["seam_mean"],
        res["stats_v"]["within_mean"],
    ]
    errs = [
        res["stats_v"]["seam_std"],
        res["stats_h"]["seam_std"],
        res["stats_v"]["within_std"],
    ]
    colors = ["#e74c3c", "#e67e22", "#2ecc71"]
    axes[row, 2].bar(cats, vals, yerr=errs, color=colors, capsize=4)
    axes[row, 2].set_ylim(0, 1)
    axes[row, 2].set_title(f"Sim stats  blend={blend}", fontsize=9)
    axes[row, 2].axhline(res["stats_v"]["within_mean"], color="#2ecc71", linestyle="--", alpha=0.5)
    axes[row, 2].grid(axis="y", alpha=0.3)

plt.suptitle(
    f"Exp 5: Blending method comparison  |  overlap={DEFAULT_OVERLAP}px"
    f"  |  DINOv{DINO_VERSION[1]}-{DINO_SIZE}",
    fontsize=13,
    y=1.01,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exp5_blend_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

blend_summary_df = pd.DataFrame(
    [
        {
            "blend": blend,
            "seam_v_mean": res["stats_v"]["seam_mean"],
            "seam_v_std": res["stats_v"]["seam_std"],
            "seam_h_mean": res["stats_h"]["seam_mean"],
            "seam_h_std": res["stats_h"]["seam_std"],
            "within_mean": res["stats_v"]["within_mean"],
            "within_std": res["stats_v"]["within_std"],
        }
        for blend, res in blend_results.items()
    ]
)
blend_summary_df.to_csv(OUTPUT_DIR / "exp5_blend_comparison.csv", index=False)
log.info("Wrote %s", OUTPUT_DIR / "exp5_blend_comparison.csv")

# %% Experiment 6 — Cross-scale self-similarity
log.info("Experiment 6: Cross-scale self-similarity …")

# How similar are corresponding patches between baseline and tiled?
# Baseline: (H_GRID, W_GRID, D) → each patch covers 1 patch of 1024×1024
# Tiled:    (2*H_GRID, 2*W_GRID, D) → each patch covers 1 patch of 2048×2048
# Corresponding: baseline patch (i,j) ↔ average of 4 tiled patches (2i:2i+2, 2j:2j+2)
feat_tiled_best = blend_results["gaussian"]["feat"]  # use gaussian blend for smoothest

# Average 2×2 tiled patches to match baseline resolution
tiled_down = feat_tiled_best.reshape(H_GRID, N_TILES, W_GRID, N_TILES, D).mean(
    dim=(1, 3)
)  # (H_GRID, W_GRID, D)
tiled_down = F.normalize(tiled_down, p=2, dim=-1)

cross_sim = (feat_base * tiled_down).sum(dim=-1).cpu().numpy()  # (H_GRID, W_GRID)

log.info(
    "Cross-scale similarity: mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
    cross_sim.mean(),
    cross_sim.std(),
    cross_sim.min(),
    cross_sim.max(),
)
pd.DataFrame(
    [
        {
            "mean": float(cross_sim.mean()),
            "std": float(cross_sim.std()),
            "min": float(cross_sim.min()),
            "max": float(cross_sim.max()),
        }
    ]
).to_csv(OUTPUT_DIR / "exp6_cross_scale_sim.csv", index=False)
log.info("Wrote %s", OUTPUT_DIR / "exp6_cross_scale_sim.csv")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].imshow(img.resize((BASE_IMG_SIZE, BASE_IMG_SIZE), Image.BICUBIC))
axes[0].set_title("Original image", fontsize=10)
axes[0].axis("off")

im = axes[1].imshow(cross_sim, cmap="jet", vmin=0, vmax=1, aspect="auto")
axes[1].set_title(
    f"Cross-scale cosine sim\nmean={cross_sim.mean():.3f}  std={cross_sim.std():.3f}",
    fontsize=10,
)
axes[1].axis("off")
plt.colorbar(im, ax=axes[1], shrink=0.85)

axes[2].hist(cross_sim.ravel(), bins=50, color="steelblue", edgecolor="white")
axes[2].set_xlabel("Cosine similarity (baseline vs. downsampled-tiled)")
axes[2].set_ylabel("Patch count")
axes[2].set_title("Distribution of cross-scale patch similarity", fontsize=10)
axes[2].grid(axis="y", alpha=0.3)

plt.suptitle(
    "Exp 6: Cross-scale patch consistency (baseline 1× vs tiled 2× downsampled to 1×)",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exp6_cross_scale_sim.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Instantiate full-size encoder — used in Exp 7 and Exp 8
log.info("Instantiating full-size encoder (img_size=%d) …", EFFECTIVE_SIZE)
encoder_full = DinoEncoder(
    version=DINO_VERSION,
    size=DINO_SIZE,
    img_size=EFFECTIVE_SIZE,
    weights_dir=DINO_WEIGHTS_DIR,
    amp=True,
)
log.info(
    "Full-size encoder ready  grid=%dx%d  D=%d",
    encoder_full.grid_h,
    encoder_full.grid_w,
    encoder_full.backbone.embed_dim,
)

# %% Experiment 7 — Instance detection comparison
log.info("Experiment 7: Instance detection baseline vs tiled vs full-size …")

try:
    anns = load_annotations(MASK_PATH)
except FileNotFoundError as e:
    log.info("No mask provided — using full image as exemplar.")
    log.info("error: %s", e)
    pixel_mask_det = None

# %%


def extract_masked_exemplar(
    feat_map: torch.Tensor,  # (H, W, D)
    pixel_mask: np.ndarray | None,
    grid_h: int,
    grid_w: int,
    img_size: int,
    threshold: float,
) -> torch.Tensor:
    """Return masked exemplar tokens (M, D) or full tokens if no mask."""
    flat = feat_map.reshape(-1, feat_map.shape[-1])
    if pixel_mask is None:
        return flat
    patch_mask_2d = pixel_mask_to_patch_mask(pixel_mask, grid_h, grid_w, img_size, threshold)
    sel = torch.from_numpy(patch_mask_2d.reshape(-1))
    return flat[sel] if sel.any() else flat


detection_class = ["donut foam", "donut foam single"]
detection_masks = [i for i in anns if i["class"] in detection_class]
pixel_mask_det = (
    np.stack([i["mask"] for i in detection_masks]).any(axis=0) if detection_masks else None
)
canvas = np.array(img.copy())
mask_overlay = (np.stack([pixel_mask_det] * 3, axis=-1) * 255).astype(np.uint8)
canvas = cv2.addWeighted(canvas, 0.5, mask_overlay, 0.5, 0)
plt.imshow(canvas)

# %%
# Baseline: use encode to get tokens from encoder forward (which is at BASE_IMG_SIZE)
MIN_PEAK_THRESHOLD = 0.2
PEAK_KERNEL_SIZE = 9
exemplar_tokens_base, ex_h_base, ex_w_base = extract_patch_tokens(
    encoder, img, LAYER_IDX, debias=with_debias
)
exemplar_masked_base = extract_masked_exemplar(
    exemplar_tokens_base.reshape(ex_h_base, ex_w_base, -1),
    pixel_mask_det,
    ex_h_base,
    ex_w_base,
    BASE_IMG_SIZE,
    MASK_PATCH_THRESHOLD,
)
feat_ex_base = compute_exemplar_features(exemplar_masked_base, mode=EXEMPLAR_MODE)

# Query = same image (self-detection, just to compare density map detail)
query_tokens_base, q_h_b, q_w_b = extract_patch_tokens(encoder, img, LAYER_IDX, debias=with_debias)
dm_base = compute_density_map(query_tokens_base, feat_ex_base, q_h_b, q_w_b, DENSITY_THRESHOLD)
peaks_base = extract_peaks(dm_base, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)

# Tiled: use the gaussian-blended tiled feature map as both exemplar and query
feat_tiled_det = blend_results["linear"]["feat"]  # (2*H, 2*W, D)
H_2x, W_2x, _ = feat_tiled_det.shape

# Build exemplar from tiled map
exemplar_masked_tiled = extract_masked_exemplar(
    feat_tiled_det,
    pixel_mask_det,
    H_2x,
    W_2x,
    EFFECTIVE_SIZE,
    MASK_PATCH_THRESHOLD,
)
feat_ex_tiled = compute_exemplar_features(exemplar_masked_tiled, mode=EXEMPLAR_MODE)

# Compute density map at tiled resolution
query_tiled = feat_tiled_det.reshape(-1, D)
dm_tiled = compute_density_map(query_tiled, feat_ex_tiled, H_2x, W_2x, DENSITY_THRESHOLD)
peaks_tiled = extract_peaks(dm_tiled, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)

log.info("Baseline peaks: %d", len(peaks_base))
log.info("Tiled peaks:    %d", len(peaks_tiled))

# Full-size: single forward pass at EFFECTIVE_SIZE via encoder_full
exemplar_tokens_full, ex_h_full, ex_w_full = extract_patch_tokens(
    encoder_full, img, LAYER_IDX, debias=with_debias
)
exemplar_masked_full = extract_masked_exemplar(
    exemplar_tokens_full.reshape(ex_h_full, ex_w_full, -1),
    pixel_mask_det,
    ex_h_full,
    ex_w_full,
    EFFECTIVE_SIZE,
    MASK_PATCH_THRESHOLD,
)
feat_ex_full = compute_exemplar_features(exemplar_masked_full, mode=EXEMPLAR_MODE)

query_tokens_full, q_h_full, q_w_full = extract_patch_tokens(
    encoder_full, img, LAYER_IDX, debias=with_debias
)
dm_full = compute_density_map(
    query_tokens_full, feat_ex_full, q_h_full, q_w_full, DENSITY_THRESHOLD
)
peaks_full = extract_peaks(dm_full, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)

log.info("Full-size peaks: %d", len(peaks_full))

dm_base_np = dm_base.cpu().numpy()
dm_tiled_np = dm_tiled.cpu().numpy()
dm_full_np = dm_full.cpu().numpy()


def heat_overlay(bg: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    colored = plt.get_cmap("jet")(norm)[..., :3]
    # Upsample heatmap to match background spatial size if needed
    if colored.shape[:2] != bg.shape[:2]:
        colored = (
            np.array(
                Image.fromarray((colored * 255).astype(np.uint8)).resize(
                    (bg.shape[1], bg.shape[0]), Image.NEAREST
                )
            )
            / 255.0
        )
    return np.clip(bg / 255.0 * (1 - alpha) + colored * alpha, 0, 1)


display_q_base = np.array(img.resize((BASE_IMG_SIZE, BASE_IMG_SIZE), Image.BICUBIC))
display_q_2x = np.array(img.resize((EFFECTIVE_SIZE, EFFECTIVE_SIZE), Image.BICUBIC))

fig, axes = plt.subplots(3, 3, figsize=(18, 18))

axes[0, 0].imshow(display_q_base)
axes[0, 0].set_title(f"Baseline image ({BASE_IMG_SIZE}px)", fontsize=10)
axes[0, 0].axis("off")

im0 = axes[0, 1].imshow(dm_base_np, cmap="jet", interpolation="nearest")
axes[0, 1].set_title(f"Baseline density map ({H_GRID}×{W_GRID})", fontsize=10)
axes[0, 1].axis("off")
plt.colorbar(im0, ax=axes[0, 1], shrink=0.85)

axes[0, 2].imshow(heat_overlay(display_q_base, dm_base_np))
if len(peaks_base):
    peaks_b_np = peaks_base.cpu().numpy()
    axes[0, 2].scatter(
        (peaks_b_np[:, 0] + 0.5) * PATCH_SIZE,
        (peaks_b_np[:, 1] + 0.5) * PATCH_SIZE,
        c="red",
        s=120,
        marker="o",
        linewidths=1.5,
        edgecolors="white",
        zorder=5,
    )
axes[0, 2].set_title(f"Baseline detections — {len(peaks_base)} found", fontsize=10)
axes[0, 2].axis("off")

axes[1, 0].imshow(display_q_2x)
axes[1, 0].set_title(f"Tiled {N_TILES}×{N_TILES} image ({EFFECTIVE_SIZE}px)", fontsize=10)
axes[1, 0].axis("off")

im1 = axes[1, 1].imshow(dm_tiled_np, cmap="jet", interpolation="nearest")
axes[1, 1].set_title(f"Tiled density map ({H_2x}×{W_2x})", fontsize=10)
axes[1, 1].axis("off")
plt.colorbar(im1, ax=axes[1, 1], shrink=0.85)

px_per_patch_2x = EFFECTIVE_SIZE / H_2x
axes[1, 2].imshow(heat_overlay(display_q_2x, dm_tiled_np))
if len(peaks_tiled):
    peaks_t_np = peaks_tiled.cpu().numpy()
    axes[1, 2].scatter(
        (peaks_t_np[:, 0] + 0.5) * px_per_patch_2x,
        (peaks_t_np[:, 1] + 0.5) * px_per_patch_2x,
        c="red",
        s=120,
        marker="o",
        linewidths=1.5,
        edgecolors="white",
        zorder=5,
    )
axes[1, 2].set_title(f"Tiled detections — {len(peaks_tiled)} found", fontsize=10)
axes[1, 2].axis("off")

# Full-size row: single forward pass at EFFECTIVE_SIZE
axes[2, 0].imshow(display_q_2x)
axes[2, 0].set_title(f"Full-size image ({EFFECTIVE_SIZE}px, single pass)", fontsize=10)
axes[2, 0].axis("off")

im2 = axes[2, 1].imshow(dm_full_np, cmap="jet", interpolation="nearest")
axes[2, 1].set_title(f"Full-size density map ({q_h_full}×{q_w_full})", fontsize=10)
axes[2, 1].axis("off")
plt.colorbar(im2, ax=axes[2, 1], shrink=0.85)

px_per_patch_full = EFFECTIVE_SIZE / q_h_full
axes[2, 2].imshow(heat_overlay(display_q_2x, dm_full_np))
if len(peaks_full):
    peaks_full_np = peaks_full.cpu().numpy()
    axes[2, 2].scatter(
        (peaks_full_np[:, 0] + 0.5) * px_per_patch_full,
        (peaks_full_np[:, 1] + 0.5) * px_per_patch_full,
        c="red",
        s=120,
        marker="o",
        linewidths=1.5,
        edgecolors="white",
        zorder=5,
    )
axes[2, 2].set_title(f"Full-size detections — {len(peaks_full)} found", fontsize=10)
axes[2, 2].axis("off")

plt.suptitle(
    f"Exp 7: Instance detection  |  DINOv{DINO_VERSION[1]}-{DINO_SIZE}  block {LAYER_IDX}\n"
    f"baseline {BASE_IMG_SIZE}px  vs  tiled {N_TILES}×{N_TILES} {EFFECTIVE_SIZE}px (overlap={DEFAULT_OVERLAP}px)  "
    f"vs  full-size {EFFECTIVE_SIZE}px",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exp7_detection_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

pd.DataFrame(
    [
        {"approach": "baseline", "grid": f"{H_GRID}x{W_GRID}", "n_peaks": len(peaks_base)},
        {"approach": "tiled", "grid": f"{H_2x}x{W_2x}", "n_peaks": len(peaks_tiled)},
        {"approach": "full-size", "grid": f"{ex_h_full}x{ex_w_full}", "n_peaks": len(peaks_full)},
    ]
).to_csv(OUTPUT_DIR / "exp7_detection_comparison.csv", index=False)
log.info("Wrote %s", OUTPUT_DIR / "exp7_detection_comparison.csv")

# %% Experiment 7.1 — Instance detection: LHa_1 as reference, LHa_2 as query
log.info("Experiment 7.1: Instance detection — reference=LHa_1, query=LHa_2 …")

img2 = Image.open(LHA_2_PATH).convert("RGB")
log.info("Loaded query image: %s  (%dx%d px)", LHA_2_PATH, *img2.size)

display_q2_base = np.array(img2.resize((BASE_IMG_SIZE, BASE_IMG_SIZE), Image.BICUBIC))
display_q2_2x = np.array(img2.resize((EFFECTIVE_SIZE, EFFECTIVE_SIZE), Image.BICUBIC))

# Baseline — reference exemplar from LHa_1 (feat_ex_base already computed), query from LHa_2
query_tokens_base2, q_h_b2, q_w_b2 = extract_patch_tokens(
    encoder, img2, LAYER_IDX, debias=with_debias
)
dm_base2 = compute_density_map(query_tokens_base2, feat_ex_base, q_h_b2, q_w_b2, DENSITY_THRESHOLD)
peaks_base2 = extract_peaks(dm_base2, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)

# Tiled — reference exemplar from LHa_1 (feat_ex_tiled already computed), query from LHa_2
feat_tiled_det2 = encode_tiled(
    img2,
    encoder,
    EFFECTIVE_SIZE,
    n_tiles=N_TILES,
    overlap_px=DEFAULT_OVERLAP,
    blend="linear",
    debias=with_debias,
)
H_2x2, W_2x2, _ = feat_tiled_det2.shape
query_tiled2 = feat_tiled_det2.reshape(-1, D)
dm_tiled2 = compute_density_map(query_tiled2, feat_ex_tiled, H_2x2, W_2x2, DENSITY_THRESHOLD)
peaks_tiled2 = extract_peaks(dm_tiled2, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)

# Full-size — reference exemplar from LHa_1 (feat_ex_full already computed), query from LHa_2
query_tokens_full2, q_h_full2, q_w_full2 = extract_patch_tokens(
    encoder_full, img2, LAYER_IDX, debias=with_debias
)
dm_full2 = compute_density_map(
    query_tokens_full2, feat_ex_full, q_h_full2, q_w_full2, DENSITY_THRESHOLD
)
peaks_full2 = extract_peaks(dm_full2, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)

log.info("Exp 7.1  Baseline peaks : %d", len(peaks_base2))
log.info("Exp 7.1  Tiled peaks    : %d", len(peaks_tiled2))
log.info("Exp 7.1  Full-size peaks: %d", len(peaks_full2))

dm_base2_np = dm_base2.cpu().numpy()
dm_tiled2_np = dm_tiled2.cpu().numpy()
dm_full2_np = dm_full2.cpu().numpy()

fig, axes = plt.subplots(3, 3, figsize=(18, 18))

axes[0, 0].imshow(display_q2_base)
axes[0, 0].set_title(f"Query (LHa_2) @ {BASE_IMG_SIZE}px", fontsize=10)
axes[0, 0].axis("off")

im0 = axes[0, 1].imshow(dm_base2_np, cmap="jet", interpolation="nearest")
axes[0, 1].set_title(f"Baseline density map ({H_GRID}×{W_GRID})", fontsize=10)
axes[0, 1].axis("off")
plt.colorbar(im0, ax=axes[0, 1], shrink=0.85)

axes[0, 2].imshow(heat_overlay(display_q2_base, dm_base2_np))
if len(peaks_base2):
    pb2_np = peaks_base2.cpu().numpy()
    axes[0, 2].scatter(
        (pb2_np[:, 0] + 0.5) * PATCH_SIZE,
        (pb2_np[:, 1] + 0.5) * PATCH_SIZE,
        c="red",
        s=120,
        marker="o",
        linewidths=1.5,
        edgecolors="white",
        zorder=5,
    )
axes[0, 2].set_title(f"Baseline detections — {len(peaks_base2)} found", fontsize=10)
axes[0, 2].axis("off")

axes[1, 0].imshow(display_q2_2x)
axes[1, 0].set_title(f"Query (LHa_2) tiled {N_TILES}×{N_TILES} ({EFFECTIVE_SIZE}px)", fontsize=10)
axes[1, 0].axis("off")

im1 = axes[1, 1].imshow(dm_tiled2_np, cmap="jet", interpolation="nearest")
axes[1, 1].set_title(f"Tiled density map ({H_2x2}×{W_2x2})", fontsize=10)
axes[1, 1].axis("off")
plt.colorbar(im1, ax=axes[1, 1], shrink=0.85)

px_per_patch_2x2 = EFFECTIVE_SIZE / H_2x2
axes[1, 2].imshow(heat_overlay(display_q2_2x, dm_tiled2_np))
if len(peaks_tiled2):
    pt2_np = peaks_tiled2.cpu().numpy()
    axes[1, 2].scatter(
        (pt2_np[:, 0] + 0.5) * px_per_patch_2x2,
        (pt2_np[:, 1] + 0.5) * px_per_patch_2x2,
        c="red",
        s=120,
        marker="o",
        linewidths=1.5,
        edgecolors="white",
        zorder=5,
    )
axes[1, 2].set_title(f"Tiled detections — {len(peaks_tiled2)} found", fontsize=10)
axes[1, 2].axis("off")

axes[2, 0].imshow(display_q2_2x)
axes[2, 0].set_title(f"Query (LHa_2) full-size ({EFFECTIVE_SIZE}px, single pass)", fontsize=10)
axes[2, 0].axis("off")

im2 = axes[2, 1].imshow(dm_full2_np, cmap="jet", interpolation="nearest")
axes[2, 1].set_title(f"Full-size density map ({q_h_full2}×{q_w_full2})", fontsize=10)
axes[2, 1].axis("off")
plt.colorbar(im2, ax=axes[2, 1], shrink=0.85)

px_per_patch_full2 = EFFECTIVE_SIZE / q_h_full2
axes[2, 2].imshow(heat_overlay(display_q2_2x, dm_full2_np))
if len(peaks_full2):
    pf2_np = peaks_full2.cpu().numpy()
    axes[2, 2].scatter(
        (pf2_np[:, 0] + 0.5) * px_per_patch_full2,
        (pf2_np[:, 1] + 0.5) * px_per_patch_full2,
        c="red",
        s=120,
        marker="o",
        linewidths=1.5,
        edgecolors="white",
        zorder=5,
    )
axes[2, 2].set_title(f"Full-size detections — {len(peaks_full2)} found", fontsize=10)
axes[2, 2].axis("off")

plt.suptitle(
    f"Exp 7.1: Cross-image detection  |  reference=LHa_1  query=LHa_2\n"
    f"DINOv{DINO_VERSION[1]}-{DINO_SIZE}  block {LAYER_IDX}  |  "
    f"baseline {BASE_IMG_SIZE}px  vs  tiled {N_TILES}×{N_TILES} {EFFECTIVE_SIZE}px  "
    f"vs  full-size {EFFECTIVE_SIZE}px",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exp7_1_cross_image_detection.png", dpi=150, bbox_inches="tight")
plt.show()

pd.DataFrame(
    [
        {"approach": "baseline", "grid": f"{H_GRID}x{W_GRID}", "n_peaks": len(peaks_base2)},
        {"approach": "tiled", "grid": f"{H_2x2}x{W_2x2}", "n_peaks": len(peaks_tiled2)},
        {
            "approach": "full-size",
            "grid": f"{q_h_full2}x{q_w_full2}",
            "n_peaks": len(peaks_full2),
        },
    ]
).to_csv(OUTPUT_DIR / "exp7_1_cross_image_detection.csv", index=False)
log.info("Wrote %s", OUTPUT_DIR / "exp7_1_cross_image_detection.csv")

# %% Summary
log.info("── High-Res Tiling Summary ─────────────────────────────────────────")
log.info("  Encoder           : DINOv%s-%s  block %d", DINO_VERSION[1], DINO_SIZE, LAYER_IDX)
log.info(
    "  Base resolution   : %d×%d  →  %dx%d patches",
    BASE_IMG_SIZE,
    BASE_IMG_SIZE,
    H_GRID,
    W_GRID,
)
log.info(
    "  Tiled resolution  : %d×%d  →  %dx%d patches",
    EFFECTIVE_SIZE,
    EFFECTIVE_SIZE,
    N_TILES * H_GRID,
    N_TILES * W_GRID,
)
log.info("  ")
log.info(
    "  Seam similarity (no overlap / within-tile baseline: %.3f):",
    overlap_results[0]["stats_v"]["within_mean"],
)
for res in overlap_results:
    log.info(
        "    overlap=%3dpx  →  seam_V=%.3f  seam_H=%.3f",
        res["overlap_px"],
        res["stats_v"]["seam_mean"],
        res["stats_h"]["seam_mean"],
    )
log.info("  ")
log.info("  Blending comparison (overlap=%d px):", DEFAULT_OVERLAP)
for blend, res in blend_results.items():
    log.info(
        "    %s  →  seam_V=%.3f  seam_H=%.3f",
        blend,
        res["stats_v"]["seam_mean"],
        res["stats_h"]["seam_mean"],
    )
log.info("  ")
log.info("  Cross-scale patch consistency: mean=%.3f", cross_sim.mean())
log.info("  ")
log.info("  Instance detection:")
log.info("    Baseline  (%dx%d): %d peaks", H_GRID, W_GRID, len(peaks_base))
log.info("    Tiled     (%dx%d): %d peaks", H_2x, W_2x, len(peaks_tiled))
log.info("    Full-size (%dx%d): %d peaks", ex_h_full, ex_w_full, len(peaks_full))
log.info("Outputs saved to %s", OUTPUT_DIR)

# %% Experiment 8 — Throughput: baseline vs tiled (batched) vs full-size single-pass
#
# Three approaches: N_TIMING_WARMUP warmup passes (discarded), then N_TIMING_RUNS timed passes:
#   baseline  — encoder at BASE_IMG_SIZE   → (H_GRID, W_GRID) patch grid
#   tiled     — encode_tiled at EFFECTIVE_SIZE via N_TILES² batched crops
#               → (N_TILES*H_GRID, N_TILES*W_GRID) grid
#   full-size — fresh encoder at EFFECTIVE_SIZE, single forward pass
#               → same (N_TILES*H_GRID, N_TILES*W_GRID) grid, no tiling overhead
log.setLevel(logging.DEBUG)
log.info("Experiment 8: Throughput comparison (%d runs each) …", N_TIMING_RUNS)

# Full-size encoder: same weights, img_size = EFFECTIVE_SIZE.
# This is the apples-to-apples single-pass reference at the tiled output resolution.
# (encoder_full was already instantiated before Exp 7)
log.info(
    "  Full-size encoder  grid=%dx%d  D=%d",
    encoder_full.grid_h,
    encoder_full.grid_w,
    encoder_full.backbone.embed_dim,
)


def _sync(dev: torch.device) -> None:
    """Flush the GPU queue so wall-clock time reflects actual compute."""
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _time_runs(fn, dev: torch.device) -> np.ndarray:
    """Warm up N_TIMING_WARMUP times, then return N_TIMING_RUNS wall-clock times in seconds."""
    for _ in range(N_TIMING_WARMUP):
        fn()
    _sync(dev)
    times: list[float] = []
    with torch.no_grad():
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            for _ in range(N_TIMING_RUNS):
                _sync(dev)
                t0 = time.perf_counter()
                fn()
                _sync(dev)
                times.append(time.perf_counter() - t0)
    return np.array(times)


log.info("  Timing baseline …")
times_base = _time_runs(lambda: encoder(img, debias=with_debias), encoder.device)

log.info("  Timing tiled (batched) …")
times_tiled = _time_runs(
    lambda: encode_tiled(
        img,
        encoder,
        EFFECTIVE_SIZE,
        n_tiles=N_TILES,
        overlap_px=DEFAULT_OVERLAP,
        blend="linear",
        debias=with_debias,
        resize_backend="cv2_linear",
    ),
    encoder.device,
)

log.info("  Timing full-size …")
times_full = _time_runs(lambda: encoder_full(img, debias=with_debias), encoder_full.device)

for label, t_arr in zip(
    ["Baseline", f"Tiled {N_TILES}×{N_TILES} (batched)", "Full-size"],
    [times_base, times_tiled, times_full],
):
    log.info(
        "  %-28s  mean=%6.1f ms  std=%5.1f ms  min=%6.1f ms  max=%6.1f ms",
        label,
        t_arr.mean() * 1e3,
        t_arr.std() * 1e3,
        t_arr.min() * 1e3,
        t_arr.max() * 1e3,
    )

labels_8 = [
    f"Baseline\n({BASE_IMG_SIZE}px  {H_GRID}×{W_GRID})",
    f"Tiled {N_TILES}×{N_TILES}\n({EFFECTIVE_SIZE}px  {N_TILES * H_GRID}×{N_TILES * W_GRID})",
    f"Full-size\n({EFFECTIVE_SIZE}px  {encoder_full.grid_h}×{encoder_full.grid_w})",
]
means_8 = np.array([times_base.mean(), times_tiled.mean(), times_full.mean()]) * 1e3
stds_8 = np.array([times_base.std(), times_tiled.std(), times_full.std()]) * 1e3
colors_8 = ["#3498db", "#e67e22", "#2ecc71"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

bars = axes[0].bar(labels_8, means_8, yerr=stds_8, color=colors_8, capsize=6, width=0.5)
axes[0].set_ylabel("Wall time (ms)")
axes[0].set_title(f"Mean ± std  ({N_TIMING_RUNS} runs, {N_TIMING_WARMUP} warmups)")
axes[0].grid(axis="y", alpha=0.3)
for bar, mean_ms, std_ms in zip(bars, means_8, stds_8):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2.0,
        mean_ms + std_ms + stds_8.max() * 0.05,
        f"{mean_ms:.1f} ms",
        ha="center",
        va="bottom",
        fontsize=9,
    )

axes[1].boxplot(
    [times_base * 1e3, times_tiled * 1e3, times_full * 1e3],
    patch_artist=True,
    boxprops={"facecolor": "none"},
    medianprops={"color": "black", "linewidth": 2},
)
axes[1].set_xticks([1, 2, 3])
axes[1].set_xticklabels(labels_8)
axes[1].set_ylabel("Wall time (ms)")
axes[1].set_title("Per-run distribution")
axes[1].grid(axis="y", alpha=0.3)

plt.suptitle(
    f"Exp 8: Throughput  |  DINOv{DINO_VERSION[1]}-{DINO_SIZE}  |  {N_TIMING_RUNS} runs each",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exp8_throughput.png", dpi=150, bbox_inches="tight")
plt.show()

pd.DataFrame(
    {
        "run": np.arange(N_TIMING_RUNS),
        "baseline_ms": times_base * 1e3,
        "tiled_ms": times_tiled * 1e3,
        "full_size_ms": times_full * 1e3,
    }
).to_csv(OUTPUT_DIR / "exp8_throughput.csv", index=False)
log.info("Wrote %s", OUTPUT_DIR / "exp8_throughput.csv")

# %% Experiment 9 — encode_tiled step-level breakdown across resize backends
#
# Compares four resize backends:
#   pil        — PIL BICUBIC (baseline)
#   cv2_cubic  — OpenCV INTER_CUBIC
#   cv2_linear — OpenCV INTER_LINEAR
#   gpu        — GPU bilinear via F.interpolate; crops also resized/normalised on GPU
#
# Steps timed per backend:
#   resize    — full-image resize to effective_size
#   crop      — tile region extraction
#               (gpu: also includes per-crop resize + ImageNet normalise + stack)
#   encoder   — batched forward pass
#               (pil/cv2: includes encoder's internal preprocess; gpu: forward only)
#   accumulate— bilinear-splatting loop (identical for all backends)
#   normalize — weight normalize + L2 normalize (identical for all backends)
#
# N_TIMING_WARMUP warmup passes (discarded), then N_TIMING_RUNS timed passes each.
log.setLevel(logging.DEBUG)
log.info("Experiment 9: encode_tiled step breakdown (%d runs) …", N_TIMING_RUNS)


def _encode_tiled_instrumented(
    image: "Image.Image | torch.Tensor",
    encoder: DinoEncoder,
    effective_size: int,
    n_tiles: int,
    overlap_px: int,
    blend: str,
    debias: bool,
    dev: torch.device,
    resize_backend: str = "pil",
) -> dict[str, float]:
    """Run encode_tiled with per-step timing.  Returns wall-time in seconds per step.

    For PIL/cv2 backends the 'encoder' step includes the encoder's internal
    per-crop preprocess (resize + normalise).  For the GPU backend that work is
    done in the 'crop' step instead, so 'encoder' is the bare forward pass only.
    The sum crop+encoder covers the same total work for every backend.
    """
    H_t = encoder.grid_h
    W_t = encoder.grid_w
    stride_px = effective_size // n_tiles
    D = encoder.backbone.embed_dim
    H_out = n_tiles * H_t
    W_out = n_tiles * W_t

    def _sync() -> None:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    # Blending weight map — computed once, same for all backends and all tiles
    ty_idx = torch.arange(H_t, dtype=torch.float32)
    tx_idx = torch.arange(W_t, dtype=torch.float32)
    TY, TX = torch.meshgrid(ty_idx, tx_idx, indexing="ij")
    if blend == "hard":
        W_map_base = torch.ones(H_t, W_t)
    elif blend == "linear":
        bdy = 1.0 - (TY - (H_t - 1) / 2.0).abs() / (H_t / 2.0)
        bdx = 1.0 - (TX - (W_t - 1) / 2.0).abs() / (W_t / 2.0)
        W_map_base = (bdy * bdx).clamp(min=0.0)
    elif blend == "gaussian":
        sigma_y = H_t / 4.0
        sigma_x = W_t / 4.0
        bdy = torch.exp(-0.5 * ((TY - (H_t - 1) / 2.0) / sigma_y) ** 2)
        bdx = torch.exp(-0.5 * ((TX - (W_t - 1) / 2.0) / sigma_x) ** 2)
        W_map_base = bdy * bdx
    else:
        raise ValueError(f"Unknown blend method: {blend!r}")

    # -------------------------------------------------------------------------
    # Steps 1–3: backend-specific
    # -------------------------------------------------------------------------

    if resize_backend in ("pil", "cv2_cubic", "cv2_linear"):
        # --- Step 1: resize ---
        _sync()
        t0 = time.perf_counter()
        if resize_backend == "pil":
            if not isinstance(image, Image.Image):
                _pil = Image.fromarray(image)
            else:
                _pil = image
            img_arr = np.array(_pil.resize((effective_size, effective_size), Image.BICUBIC))
        else:
            _np = np.array(image) if isinstance(image, Image.Image) else image
            interp = cv2.INTER_CUBIC if resize_backend == "cv2_cubic" else cv2.INTER_LINEAR
            img_arr = cv2.resize(_np, (effective_size, effective_size), interpolation=interp)
        _sync()
        t_resize = time.perf_counter() - t0

        # --- Step 2: crop extraction ---
        _sync()
        t0 = time.perf_counter()
        tile_meta: list[tuple[int, int, int, int, int, int]] = []
        crops: list[Image.Image] = []
        for row in range(n_tiles):
            for col in range(n_tiles):
                y0 = max(0, row * stride_px - overlap_px)
                y1 = min(effective_size, (row + 1) * stride_px + overlap_px)
                x0 = max(0, col * stride_px - overlap_px)
                x1 = min(effective_size, (col + 1) * stride_px + overlap_px)
                crops.append(Image.fromarray(img_arr[y0:y1, x0:x1]))
                tile_meta.append((y0, y1, x0, x1, y1 - y0, x1 - x0))
        _sync()
        t_crop = time.perf_counter() - t0

        # --- Step 3: encoder (includes internal per-crop preprocess) ---
        _sync()
        t0 = time.perf_counter()
        out = encoder(crops, debias=debias)
        tile_feats = out.patches.cpu().float()
        _sync()
        t_encoder = time.perf_counter() - t0

    else:  # "gpu"
        # --- Step 1: convert + GPU resize of full image ---
        _sync()
        t0 = time.perf_counter()
        if isinstance(image, Image.Image):
            _img_t = torch.from_numpy(
                np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
            ).to(dev)
        elif isinstance(image, np.ndarray):
            _img_t = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0).to(dev)
        else:
            _img_t = image.float().to(dev)
        img_big_t = F.interpolate(
            _img_t.unsqueeze(0),
            size=(effective_size, effective_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        _sync()
        t_resize = time.perf_counter() - t0

        # --- Step 2: tile coords + GPU per-crop resize / normalise / stack ---
        _sync()
        t0 = time.perf_counter()
        tile_meta = []
        for row in range(n_tiles):
            for col in range(n_tiles):
                y0 = max(0, row * stride_px - overlap_px)
                y1 = min(effective_size, (row + 1) * stride_px + overlap_px)
                x0 = max(0, col * stride_px - overlap_px)
                x1 = min(effective_size, (col + 1) * stride_px + overlap_px)
                tile_meta.append((y0, y1, x0, x1, y1 - y0, x1 - x0))
        _mean = _IMAGENET_MEAN_T.to(dev)
        _std = _IMAGENET_STD_T.to(dev)
        batch_t = torch.stack(
            [
                (
                    F.interpolate(
                        img_big_t[:, y0:y1, x0:x1].unsqueeze(0),
                        size=(encoder.img_size, encoder.img_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    - _mean
                )
                / _std
                for y0, y1, x0, x1, _, _ in tile_meta
            ]
        )
        _sync()
        t_crop = time.perf_counter() - t0

        # --- Step 3: encoder forward only (preprocess already done above) ---
        _sync()
        t0 = time.perf_counter()
        out = encoder(batch_t, debias=debias)
        tile_feats = out.patches.cpu().float()
        _sync()
        t_encoder = time.perf_counter() - t0

    # -------------------------------------------------------------------------
    # Steps 4–5: identical for all backends
    # -------------------------------------------------------------------------

    acc = torch.zeros(H_out, W_out, D, dtype=torch.float32)
    wgt = torch.zeros(H_out, W_out, dtype=torch.float32)

    # --- Step 4: accumulate ---
    _sync()
    t0 = time.perf_counter()
    for tile_feat, (y0, y1, x0, x1, tile_h, tile_w) in zip(tile_feats, tile_meta):
        gc_y = y0 + (TY + 0.5) * tile_h / H_t
        gc_x = x0 + (TX + 0.5) * tile_w / W_t
        oy_f = (gc_y * H_out / effective_size).clamp(0, H_out - 1)
        ox_f = (gc_x * W_out / effective_size).clamp(0, W_out - 1)
        flat_w = W_map_base.reshape(-1)
        flat_f = tile_feat.reshape(H_t * W_t, D)
        oy_lo = oy_f.long().clamp(0, H_out - 1)
        oy_hi = (oy_lo + 1).clamp(0, H_out - 1)
        ox_lo = ox_f.long().clamp(0, W_out - 1)
        ox_hi = (ox_lo + 1).clamp(0, W_out - 1)
        frac_y = (oy_f - oy_lo.float()).clamp(0.0, 1.0)
        frac_x = (ox_f - ox_lo.float()).clamp(0.0, 1.0)
        for OY, OX, corner_w in [
            (oy_lo, ox_lo, (1.0 - frac_y) * (1.0 - frac_x)),
            (oy_lo, ox_hi, (1.0 - frac_y) * frac_x),
            (oy_hi, ox_lo, frac_y * (1.0 - frac_x)),
            (oy_hi, ox_hi, frac_y * frac_x),
        ]:
            flat_oy = OY.reshape(-1)
            flat_ox = OX.reshape(-1)
            combined_w = flat_w * corner_w.reshape(-1)
            flat_idx = flat_oy * W_out + flat_ox
            wgt.reshape(-1).scatter_add_(0, flat_idx, combined_w)
            acc.reshape(-1, D).scatter_add_(
                0,
                flat_idx.unsqueeze(1).expand(-1, D),
                combined_w.unsqueeze(1) * flat_f,
            )
    _sync()
    t_accumulate = time.perf_counter() - t0

    # --- Step 5: normalize ---
    _sync()
    t0 = time.perf_counter()
    mask = wgt > 0
    acc[mask] /= wgt[mask, None]
    F.normalize(acc, p=2, dim=-1)
    _sync()
    t_normalize = time.perf_counter() - t0

    return {
        "resize": t_resize,
        "crop": t_crop,
        "encoder": t_encoder,
        "accumulate": t_accumulate,
        "normalize": t_normalize,
    }


# Pre-build GPU tensor once — used for the "gpu" backend
_img_gpu = torch.from_numpy(np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0).to(
    encoder.device
)

# Backends to compare: (label, resize_backend, image_input)
_backends: list[tuple[str, str, "Image.Image | torch.Tensor"]] = [
    ("PIL\n(BICUBIC)", "pil", img),
    ("cv2\n(CUBIC)", "cv2_cubic", img),
    ("cv2\n(LINEAR)", "cv2_linear", img),
    ("GPU\n(bilinear)", "gpu", _img_gpu),
]
_step_names = ["resize", "crop", "encoder", "accumulate", "normalize"]
_colors_9 = ["#3498db", "#9b59b6", "#e67e22", "#e74c3c", "#2ecc71"]

# Per-backend timing results: {backend_label: {step: np.ndarray (ms)}}
_all_results: dict[str, dict[str, np.ndarray]] = {}

for _blabel, _bname, _bimg in _backends:
    log.info("  [%s] Warming up (%d passes) …", _bname, N_TIMING_WARMUP)
    for _ in range(N_TIMING_WARMUP):
        encode_tiled(
            _bimg,
            encoder,
            EFFECTIVE_SIZE,
            n_tiles=N_TILES,
            overlap_px=DEFAULT_OVERLAP,
            blend="linear",
            debias=with_debias,
            resize_backend=_bname,
        )
    if encoder.device.type == "cuda":
        torch.cuda.synchronize(encoder.device)

    _times: dict[str, list[float]] = {s: [] for s in _step_names}
    log.info("  [%s] Timing %d runs …", _bname, N_TIMING_RUNS)
    with torch.no_grad():
        with torch.autocast(encoder.device.type, dtype=torch.bfloat16):
            for _i in range(N_TIMING_RUNS):
                _run = _encode_tiled_instrumented(
                    _bimg,
                    encoder,
                    EFFECTIVE_SIZE,
                    n_tiles=N_TILES,
                    overlap_px=DEFAULT_OVERLAP,
                    blend="linear",
                    debias=with_debias,
                    dev=encoder.device,
                    resize_backend=_bname,
                )
                for _s in _step_names:
                    _times[_s].append(_run[_s])

    _all_results[_blabel] = {s: np.array(_times[s]) * 1e3 for s in _step_names}
    _total = sum(v.mean() for v in _all_results[_blabel].values())
    log.info("  [%s] step breakdown (mean ms):", _bname)
    for _s in _step_names:
        _arr = _all_results[_blabel][_s]
        log.info(
            "    %-12s  mean=%6.1f ms  std=%5.1f ms  (%4.1f%%)",
            _s,
            _arr.mean(),
            _arr.std(),
            _arr.mean() / _total * 100,
        )
    log.info("    %-12s  mean=%6.1f ms", "TOTAL", _total)

# --- Plot ---
_b_labels = [b[0] for b in _backends]
_n_backends = len(_b_labels)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: stacked bars — one per backend, each stacked by step
_bar_w = 0.55
_x_pos = np.arange(_n_backends)
for _si, (_s, _color) in enumerate(zip(_step_names, _colors_9)):
    _bottoms_arr = np.array(
        [sum(_all_results[bl][ss].mean() for ss in _step_names[:_si]) for bl in _b_labels]
    )
    _means_arr = np.array([_all_results[bl][_s].mean() for bl in _b_labels])
    axes[0].bar(
        _x_pos,
        _means_arr,
        bottom=_bottoms_arr,
        color=_color,
        width=_bar_w,
        label=_step_names[_si].capitalize(),
    )

# Annotate total on top of each bar
for _xi, _bl in zip(_x_pos, _b_labels):
    _tot = sum(_all_results[_bl][_s].mean() for _s in _step_names)
    axes[0].text(
        _xi,
        _tot + _tot * 0.01,
        f"{_tot:.1f} ms",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )

axes[0].set_xticks(_x_pos)
axes[0].set_xticklabels(_b_labels, fontsize=9)
axes[0].set_ylabel("Wall time (ms)")
axes[0].set_title(f"Step breakdown per backend (mean, {N_TIMING_RUNS} runs)")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].grid(axis="y", alpha=0.3)

# Right: box plot of total time per backend
_totals_per_backend = [sum(_all_results[bl][_s] for _s in _step_names) for bl in _b_labels]
axes[1].boxplot(
    _totals_per_backend,
    patch_artist=True,
    boxprops={"facecolor": "none"},
    medianprops={"color": "black", "linewidth": 2},
)
axes[1].set_xticks(range(1, _n_backends + 1))
axes[1].set_xticklabels(_b_labels, fontsize=9)
axes[1].set_ylabel("Total wall time (ms)")
axes[1].set_title("Total time distribution per backend")
axes[1].grid(axis="y", alpha=0.3)

plt.suptitle(
    f"Exp 9: encode_tiled breakdown  |  DINOv{DINO_VERSION[1]}-{DINO_SIZE}"
    f"  |  {N_TILES}×{N_TILES} tiles  overlap={DEFAULT_OVERLAP}px  |  {N_TIMING_RUNS} runs",
    fontsize=11,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exp9_encode_tiled_breakdown.png", dpi=150, bbox_inches="tight")
plt.show()

pd.DataFrame(
    [
        {
            "backend": _bname,
            "step": _s,
            "mean_ms": float(_all_results[_blabel][_s].mean()),
            "std_ms": float(_all_results[_blabel][_s].std()),
        }
        for _blabel, _bname, _ in _backends
        for _s in _step_names
    ]
).to_csv(OUTPUT_DIR / "exp9_encode_tiled_breakdown.csv", index=False)
log.info("Wrote %s", OUTPUT_DIR / "exp9_encode_tiled_breakdown.csv")

# %%
