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
#   7.  Instance detection — baseline vs tiled density map comparison

# %% Logging — must be before torch import
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("high_res_tiling")

from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from sklearn.decomposition import PCA

from dinoisawesome import DinoEncoder
from dinoisawesome.instance_detection import (
    compute_density_map,
    compute_exemplar_features,
    extract_patch_tokens,
    extract_peaks,
)

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".env")

# Image to experiment on — set to a local path or leave None to download
IMAGE_PATH: Path | None = _REPO_ROOT / "data" / "abc2" / "LHa_1.jpg"

# Optional exemplar mask for instance-detection comparison (.npy / .npz)
MASK_PATH: Path | None = None

# Encoder
DINO_VERSION = "v3"  # "v2" or "v3"
DINO_SIZE = "large"  # "small" | "base" | "large" | "giant"
BASE_IMG_SIZE = 1024  # encoder's fixed input size; "1× resolution" baseline
LAYER_IDX = 23  # transformer block for patch-token extraction
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")

# Tiling
N_TILES = 2  # n_tiles × n_tiles grid (2 → 4 tiles, 3 → 9 tiles)
EFFECTIVE_SIZE = BASE_IMG_SIZE * N_TILES  # 2048 — image is scaled to this before tiling

# Overlap sweep
OVERLAP_VALUES = [0, 16, 32, 64, 128]  # pixels of context added to each tile boundary
DEFAULT_OVERLAP = 32  # overlap used for blending comparison

# Blending methods to compare
BLEND_METHODS = ["hard", "linear", "gaussian"]

# Instance detection (for section 7)
DENSITY_THRESHOLD = 0.3
PEAK_KERNEL_SIZE = 5
MIN_PEAK_THRESHOLD = 0.3
MASK_PATCH_THRESHOLD = 0.3
EXEMPLAR_MODE = "mean"

OUTPUT_DIR = _REPO_ROOT / "results" / "high_res_tiling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% Core tiling function


def encode_tiled(
    image: Image.Image,
    encoder: DinoEncoder,
    effective_size: int,
    n_tiles: int = 2,
    overlap_px: int = 0,
    blend: str = "hard",
    debias: bool = True,
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

    Returns (n_tiles*H_grid, n_tiles*W_grid, D) float32 CPU tensor, L2-normalised.
    """
    H_t = encoder.grid_h  # patches per tile dimension
    W_t = encoder.grid_w
    stride_px = effective_size // n_tiles

    img_big = image.resize((effective_size, effective_size), Image.BICUBIC)
    img_arr = np.array(img_big)

    D = encoder.backbone.embed_dim
    H_out = n_tiles * H_t
    W_out = n_tiles * W_t
    acc = torch.zeros(H_out, W_out, D, dtype=torch.float32)
    wgt = torch.zeros(H_out, W_out, dtype=torch.float32)

    ty_idx = torch.arange(H_t, dtype=torch.float32)
    tx_idx = torch.arange(W_t, dtype=torch.float32)
    TY, TX = torch.meshgrid(ty_idx, tx_idx, indexing="ij")  # (H_t, W_t)

    for row in range(n_tiles):
        for col in range(n_tiles):
            y0 = max(0, row * stride_px - overlap_px)
            y1 = min(effective_size, (row + 1) * stride_px + overlap_px)
            x0 = max(0, col * stride_px - overlap_px)
            x1 = min(effective_size, (col + 1) * stride_px + overlap_px)

            tile_h = y1 - y0
            tile_w = x1 - x0

            crop = Image.fromarray(img_arr[y0:y1, x0:x1])
            out = encoder([crop], debias=debias)
            tile_feat = out.patches[0].cpu().float()  # (H_t, W_t, D)

            # Map each tile patch to its canonical output position (float)
            gc_y = y0 + (TY + 0.5) * tile_h / H_t  # global pixel centre (H_t, W_t)
            gc_x = x0 + (TX + 0.5) * tile_w / W_t
            oy_f = (gc_y * H_out / effective_size).clamp(0, H_out - 1)  # float
            ox_f = (gc_x * W_out / effective_size).clamp(0, W_out - 1)

            # Per-patch blending weights
            if blend == "hard":
                W_map = torch.ones(H_t, W_t)
            elif blend == "linear":
                # Triangle: peak = 1 at tile centre, 0 at outer edge
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

            bilinear_corners = [
                (oy_lo, ox_lo, (1.0 - frac_y) * (1.0 - frac_x)),
                (oy_lo, ox_hi, (1.0 - frac_y) * frac_x),
                (oy_hi, ox_lo, frac_y * (1.0 - frac_x)),
                (oy_hi, ox_hi, frac_y * frac_x),
            ]
            for OY, OX, corner_w in bilinear_corners:
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

    # Normalise by accumulated weights
    mask = wgt > 0
    acc[mask] /= wgt[mask, None]

    # L2-normalise each patch vector
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
with_debias = True  # v3 large has some bias in later layers that can dominate PCA; set to False to disable
out_base = encoder(img, debias=with_debias)
feat_base = out_base.patches[0].cpu().float()  # (H_GRID, W_GRID, D)
feat_base = F.normalize(feat_base, p=2, dim=-1)

# Tiled 2×: 4 tiles, no overlap
feat_tiled = encode_tiled(img, encoder, EFFECTIVE_SIZE, n_tiles=N_TILES, overlap_px=0, blend="hard", debias=with_debias)
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

overlap_results: list[dict] = []
for ov in OVERLAP_VALUES:
    log.info("  overlap_px=%d …", ov)
    feat_ov = encode_tiled(
        img, encoder, EFFECTIVE_SIZE, n_tiles=N_TILES, overlap_px=ov, blend="linear", debias=with_debias
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

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].imshow(to_img(rgb_base, BASE_IMG_SIZE))
axes[0].set_title("Baseline PCA (1024)", fontsize=10)
axes[0].axis("off")

im = axes[1].imshow(cross_sim, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
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

# %% Experiment 7 — Instance detection comparison
log.info("Experiment 7: Instance detection baseline vs tiled …")

if MASK_PATH is not None and Path(MASK_PATH).exists():
    if str(MASK_PATH).endswith(".npy"):
        raw_seg = np.load(MASK_PATH)
        if raw_seg.ndim == 3 and raw_seg.shape[0] < raw_seg.shape[1]:
            raw_seg = raw_seg.transpose(1, 2, 0)
    else:
        raw_seg = np.load(MASK_PATH)["segmaps"]
    pixel_mask_det = raw_seg.any(axis=2)
    log.info("Mask loaded: coverage=%.1f%%", 100 * pixel_mask_det.mean())
else:
    log.info("No mask provided — using full image as exemplar.")
    pixel_mask_det = None


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
    mask_pil = Image.fromarray(pixel_mask.astype(np.uint8) * 255)
    mask_res = np.array(mask_pil.resize((img_size, img_size), Image.NEAREST)) > 0
    ph = img_size // grid_h
    pw = img_size // grid_w
    tiled_m = mask_res.reshape(grid_h, ph, grid_w, pw)
    patch_den = tiled_m.mean(axis=(1, 3))
    patch_mask_2d = patch_den >= threshold
    sel = torch.from_numpy(patch_mask_2d.reshape(-1))
    return flat[sel] if sel.any() else flat


# Baseline: use encode to get tokens from encoder forward (which is at BASE_IMG_SIZE)
exemplar_tokens_base, ex_h_base, ex_w_base = extract_patch_tokens(encoder, img, LAYER_IDX)
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
query_tokens_base, q_h_b, q_w_b = extract_patch_tokens(encoder, img, LAYER_IDX)
dm_base = compute_density_map(query_tokens_base, feat_ex_base, q_h_b, q_w_b, DENSITY_THRESHOLD)
peaks_base = extract_peaks(dm_base, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)

# Tiled: use the gaussian-blended tiled feature map as both exemplar and query
feat_tiled_det = blend_results["gaussian"]["feat"]  # (2*H, 2*W, D)
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

dm_base_np = dm_base.cpu().numpy()
dm_tiled_np = dm_tiled.cpu().numpy()


def heat_overlay(bg: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    colored = plt.get_cmap("jet")(norm)[..., :3]
    # Upsample heatmap to match background spatial size if needed
    if colored.shape[:2] != bg.shape[:2]:
        colored = np.array(
            Image.fromarray((colored * 255).astype(np.uint8)).resize(
                (bg.shape[1], bg.shape[0]), Image.NEAREST
            )
        ) / 255.0
    return np.clip(bg / 255.0 * (1 - alpha) + colored * alpha, 0, 1)


display_q_base = np.array(img.resize((BASE_IMG_SIZE, BASE_IMG_SIZE), Image.BICUBIC))
display_q_2x = np.array(img.resize((EFFECTIVE_SIZE, EFFECTIVE_SIZE), Image.BICUBIC))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(display_q_base)
axes[0, 0].set_title(f"Baseline image ({BASE_IMG_SIZE}px)", fontsize=10)
axes[0, 0].axis("off")

im0 = axes[0, 1].imshow(dm_base_np, cmap="jet", interpolation="nearest")
axes[0, 1].set_title(f"Baseline density map ({H_GRID}×{W_GRID})", fontsize=10)
axes[0, 1].axis("off")
plt.colorbar(im0, ax=axes[0, 1], shrink=0.85)

axes[0, 2].imshow(display_q_base)
axes[0, 2].imshow(heat_overlay(display_q_base, dm_base_np))
if len(peaks_base):
    peaks_b_np = peaks_base.cpu().numpy()
    axes[0, 2].scatter(
        (peaks_b_np[:, 0] + 0.5) * PATCH_SIZE * (BASE_IMG_SIZE / BASE_IMG_SIZE),
        (peaks_b_np[:, 1] + 0.5) * PATCH_SIZE * (BASE_IMG_SIZE / BASE_IMG_SIZE),
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
axes[1, 0].set_title(f"Tiled image ({EFFECTIVE_SIZE}px)", fontsize=10)
axes[1, 0].axis("off")

im1 = axes[1, 1].imshow(dm_tiled_np, cmap="jet", interpolation="nearest")
axes[1, 1].set_title(f"Tiled density map ({H_2x}×{W_2x})", fontsize=10)
axes[1, 1].axis("off")
plt.colorbar(im1, ax=axes[1, 1], shrink=0.85)

px_per_patch_2x = EFFECTIVE_SIZE / H_2x
axes[1, 2].imshow(display_q_2x)
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

plt.suptitle(
    f"Exp 7: Instance detection  |  DINOv{DINO_VERSION[1]}-{DINO_SIZE}  block {LAYER_IDX}  "
    f"blend=gaussian  overlap={DEFAULT_OVERLAP}px",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exp7_detection_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

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
log.info("    Baseline (%dx%d): %d peaks", H_GRID, W_GRID, len(peaks_base))
log.info("    Tiled    (%dx%d): %d peaks", H_2x, W_2x, len(peaks_tiled))
log.info("Outputs saved to %s", OUTPUT_DIR)

# %%
