#!/usr/bin/env python3
"""Multi-scale exemplar gallery experiment for instance detection.

Tests detection quality when the exemplar descriptor is built from the
reference image cropped at three spatial scales:

  close  – tight crop around the annotated object (+ configurable padding)
  mid    – halfway between the close crop and the full image
  full   – entire 4K reference image

Detection always runs on the full-resolution query images, so we can see
how the granularity of the exemplar features affects what the model finds.

Usage
-----
    python scripts/multiscale_detection.py \\
        --data-dir ~/DinoisAwesome/data/abc2 \\
        --part-type LHb \\
        --ref-number 1 \\
        --out-dir scripts/multiscale_results
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Literal

# Logging must be configured before torch is imported.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("multiscale_detection")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from dinoisawesome import DinoEncoder  # noqa: E402
from dinoisawesome.instance_detection import (  # noqa: E402
    compute_density_map,
    compute_exemplar_features,
    extract_patch_tokens,
    extract_peaks,
)

# ── Experiment parameters ──────────────────────────────────────────────────────
DINO_VERSION: Literal["v2", "v3"] = "v3"
DINO_SIZE: Literal["small", "base", "large", "giant"] = "large"
IMG_SIZE = 1024  # must be divisible by patch_size (16 for v3)
LAYER_IDX = 23  # 0-based transformer block index

MASK_PATCH_THRESHOLD = 0.3  # fraction of patch pixels that must be in mask
CLOSE_PADDING_FRAC = 0.3  # expand close crop by this fraction of object extent

EXEMPLAR_MODE: Literal["mean", "kmeans"] = "mean"
EXEMPLAR_K = 3
DENSITY_THRESHOLD = 0.3
PEAK_KERNEL_SIZE = 5
MIN_PEAK_THRESHOLD = 0.3

SCALES: list[str] = ["close", "mid", "full"]
SCALE_COLOR = {"close": "#e74c3c", "mid": "#f39c12", "full": "#2ecc71"}
DISPLAY_WIDTH = 900  # pixels wide for display thumbnails


# ── Geometry helpers ───────────────────────────────────────────────────────────


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of the True region as (rmin, rmax, cmin, cmax)."""
    rows, cols = np.where(mask)
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def scale_crop_box(
    mask: np.ndarray,
    scale: str,
    padding_frac: float = CLOSE_PADDING_FRAC,
) -> tuple[int, int, int, int]:
    """PIL crop box (x0, y0, x1, y1) for the requested scale.

    close – tight bbox around mask + padding
    mid   – midpoint between close box and full image extents
    full  – entire image
    """
    H, W = mask.shape
    rmin, rmax, cmin, cmax = mask_bbox(mask)

    pad_r = int((rmax - rmin) * padding_frac)
    pad_c = int((cmax - cmin) * padding_frac)

    close: tuple[int, int, int, int] = (
        max(0, cmin - pad_c),
        max(0, rmin - pad_r),
        min(W, cmax + pad_c),
        min(H, rmax + pad_r),
    )
    full: tuple[int, int, int, int] = (0, 0, W, H)

    if scale == "close":
        return close
    if scale == "full":
        return full
    if scale == "mid":
        return (
            (close[0] + full[0]) // 2,
            (close[1] + full[1]) // 2,
            (close[2] + full[2]) // 2,
            (close[3] + full[3]) // 2,
        )
    raise ValueError(f"Unknown scale: {scale!r}")


def crop_mask(pixel_mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop pixel mask to box (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    return pixel_mask[y0:y1, x0:x1]


def pixel_mask_to_patch_mask(
    pixel_mask: np.ndarray,
    grid_h: int,
    grid_w: int,
    img_size: int,
    threshold: float = MASK_PATCH_THRESHOLD,
) -> np.ndarray:
    """Resize a (H, W) bool pixel mask to (grid_h, grid_w) patch-grid bool array."""
    mask_pil = Image.fromarray(pixel_mask.astype(np.uint8) * 255)
    resized = np.array(mask_pil.resize((img_size, img_size), Image.NEAREST)) > 0
    ph = img_size // grid_h
    pw = img_size // grid_w
    tiled = resized.reshape(grid_h, ph, grid_w, pw)
    return tiled.mean(axis=(1, 3)) >= threshold  # (grid_h, grid_w) bool


# ── Visualisation helpers ──────────────────────────────────────────────────────


def thumb(img: Image.Image, width: int = DISPLAY_WIDTH) -> np.ndarray:
    """Resize PIL image to `width` keeping aspect ratio, return uint8 array."""
    h, w = img.height, img.width
    new_h = int(h * width / w)
    return np.array(img.resize((width, new_h), Image.BICUBIC))


def upsample_map(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Normalise to [0, 1] and upsample a (H, W) map to (target_h, target_w)."""
    norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    pil = Image.fromarray((norm * 255).astype(np.uint8))
    return np.array(pil.resize((target_w, target_h), Image.NEAREST)) / 255.0


def heat_overlay(
    bg_uint8: np.ndarray,
    heat: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """Blend a jet heatmap over an uint8 RGB image."""
    colored = plt.get_cmap("jet")(heat)[..., :3]
    return np.clip(bg_uint8 / 255.0 * (1 - alpha) + colored * alpha, 0, 1)


def mask_overlay(
    base_uint8: np.ndarray, mask_bool: np.ndarray, color_rgba=(0.2, 0.9, 0.2, 0.45)
) -> None:
    """Imshow a semi-transparent colour overlay for a boolean mask on the current axes."""
    ov = np.zeros((*mask_bool.shape, 4), dtype=np.float32)
    ov[mask_bool] = color_rgba
    plt.gca().imshow(ov)


def draw_box(ax: plt.Axes, box: tuple[int, int, int, int], color: str, label: str) -> None:
    """Draw a PIL-style (x0,y0,x1,y1) rectangle on an axes."""
    x0, y0, x1, y1 = box
    rect = mpatches.Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        linewidth=2,
        edgecolor=color,
        facecolor="none",
        label=label,
    )
    ax.add_patch(rect)


# ── Data loading ───────────────────────────────────────────────────────────────


def load_pixel_mask(path: Path) -> np.ndarray:
    """Load .npy annotation (N, H, W) bool → (H, W) union mask."""
    raw = np.load(path)  # (N, H, W)
    raw = raw.transpose(1, 2, 0)  # (H, W, N)
    return raw.any(axis=2)  # (H, W) bool


def find_images(data_dir: Path, part_type: str, ref_number: int) -> tuple[Path, list[Path]]:
    """Return (exemplar_path, [query_path, ...])."""
    all_imgs = sorted(data_dir.glob(f"{part_type}_*.jpg"))
    ref_name = f"{part_type}_{ref_number}.jpg"
    exemplar = next((p for p in all_imgs if p.name == ref_name), None)
    if exemplar is None:
        raise FileNotFoundError(f"Exemplar {ref_name} not found in {data_dir}")
    queries = [p for p in all_imgs if p.name != ref_name]
    if not queries:
        raise FileNotFoundError(
            f"No query images found in {data_dir} (expected other {part_type}_*.jpg)"
        )
    return exemplar, queries


# ── Per-scale feature extraction ───────────────────────────────────────────────


def build_scale_descriptor(
    encoder: DinoEncoder,
    exemplar_img: Image.Image,
    pixel_mask: np.ndarray,
    scale: str,
) -> dict:
    """Crop exemplar at `scale`, extract tokens, return descriptor dict."""
    box = scale_crop_box(pixel_mask, scale)
    x0, y0, x1, y1 = box
    crop_w, crop_h = x1 - x0, y1 - y0

    exemplar_crop = exemplar_img.crop(box)
    mask_crop = crop_mask(pixel_mask, box)

    tokens, grid_h, grid_w = extract_patch_tokens(encoder, exemplar_crop, LAYER_IDX)

    patch_mask = pixel_mask_to_patch_mask(mask_crop, grid_h, grid_w, IMG_SIZE)
    patch_mask_flat = torch.from_numpy(patch_mask.reshape(-1)).to(tokens.device)

    n_masked = int(patch_mask_flat.sum().item())
    n_total = grid_h * grid_w
    log.info(
        "[%s] crop %dx%d | masked patches %d / %d (%.1f%%)",
        scale,
        crop_w,
        crop_h,
        n_masked,
        n_total,
        100.0 * n_masked / n_total,
    )

    if not patch_mask_flat.any():
        log.warning("[%s] mask is empty after projection; using all patches", scale)
        tokens_masked = tokens
    else:
        tokens_masked = tokens[patch_mask_flat]

    feat = compute_exemplar_features(tokens_masked, mode=EXEMPLAR_MODE, k=EXEMPLAR_K)

    return {
        "scale": scale,
        "box": box,
        "crop_img": exemplar_crop,
        "mask_crop": mask_crop,
        "patch_mask": patch_mask,
        "feat": feat,
        "n_masked": n_masked,
        "n_total": n_total,
    }


# ── Figure: crop box overview ──────────────────────────────────────────────────


def fig_crop_overview(
    exemplar_img: Image.Image,
    pixel_mask: np.ndarray,
    scale_infos: list[dict],
    out_path: Path,
) -> None:
    """Show all three crop bounding boxes on the full exemplar."""
    ncols = 1 + len(scale_infos)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, 5))

    # Full exemplar with all boxes
    ex_arr = thumb(exemplar_img)
    H_full, W_full = exemplar_img.height, exemplar_img.width
    scale_x = ex_arr.shape[1] / W_full
    scale_y = ex_arr.shape[0] / H_full

    axes[0].imshow(ex_arr)
    # Also overlay mask in green
    mask_resized = (
        np.array(
            Image.fromarray(pixel_mask.astype(np.uint8) * 255).resize(
                (ex_arr.shape[1], ex_arr.shape[0]), Image.NEAREST
            )
        )
        > 0
    )
    ov = np.zeros((*mask_resized.shape, 4), dtype=np.float32)
    ov[mask_resized] = [0.2, 0.9, 0.2, 0.4]
    axes[0].imshow(ov)
    for info in scale_infos:
        x0, y0, x1, y1 = info["box"]
        scaled_box = (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)
        draw_box(axes[0], scaled_box, SCALE_COLOR[info["scale"]], info["scale"])
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Crop boxes on full exemplar\n(green = annotated mask)", fontsize=9)
    axes[0].axis("off")

    # One panel per crop scale
    for ax, info in zip(axes[1:], scale_infos):
        crop_arr = thumb(info["crop_img"])
        mask_c_resized = (
            np.array(
                Image.fromarray(info["mask_crop"].astype(np.uint8) * 255).resize(
                    (crop_arr.shape[1], crop_arr.shape[0]), Image.NEAREST
                )
            )
            > 0
        )
        ax.imshow(crop_arr)
        ov2 = np.zeros((*mask_c_resized.shape, 4), dtype=np.float32)
        ov2[mask_c_resized] = [0.2, 0.9, 0.2, 0.45]
        ax.imshow(ov2)
        x0, y0, x1, y1 = info["box"]
        ax.set_title(
            f"Scale: {info['scale']}\n"
            f"crop {x1 - x0}×{y1 - y0} px  |  "
            f"{info['n_masked']}/{info['n_total']} patches masked",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle("Multi-scale crops with annotated mask", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


# ── Figure: per-query detection comparison ─────────────────────────────────────


def fig_detection_comparison(
    query_img: Image.Image,
    query_tokens: torch.Tensor,
    query_grid_h: int,
    query_grid_w: int,
    scale_infos: list[dict],
    query_name: str,
    out_path: Path,
) -> None:
    """Rows = scales, cols = [exemplar crop | density map | detection overlay]."""
    n_rows = len(scale_infos)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    q_arr = np.array(query_img)

    for row, info in enumerate(scale_infos):
        scale = info["scale"]
        color = SCALE_COLOR[scale]

        # Density map + peaks
        dm = compute_density_map(
            query_tokens, info["feat"], query_grid_h, query_grid_w, DENSITY_THRESHOLD
        )
        peaks = extract_peaks(dm, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)
        dm_np = dm.cpu().numpy()
        peaks_np = peaks.cpu().numpy()

        log.info(
            "[%s] query=%s  peaks=%d  density_max=%.3f",
            scale,
            query_name,
            len(peaks_np),
            float(dm_np.max()),
        )

        # Col 0: exemplar crop with mask
        crop_arr = thumb(info["crop_img"])
        mask_c_resized = (
            np.array(
                Image.fromarray(info["mask_crop"].astype(np.uint8) * 255).resize(
                    (crop_arr.shape[1], crop_arr.shape[0]), Image.NEAREST
                )
            )
            > 0
        )
        axes[row, 0].imshow(crop_arr)
        ov = np.zeros((*mask_c_resized.shape, 4), dtype=np.float32)
        ov[mask_c_resized] = [0.2, 0.9, 0.2, 0.45]
        axes[row, 0].imshow(ov)
        x0, y0, x1, y1 = info["box"]
        axes[row, 0].set_title(
            f"[{scale}] exemplar crop {x1 - x0}×{y1 - y0} px\n"
            f"{info['n_masked']}/{info['n_total']} patches used",
            fontsize=9,
            color=color,
        )
        axes[row, 0].axis("off")

        # Col 1: density map
        im1 = axes[row, 1].imshow(dm_np, cmap="jet", aspect="auto", vmin=0, vmax=dm_np.max())
        axes[row, 1].set_title(
            f"[{scale}] density map\n"
            f"max={dm_np.max():.3f}  active={int((dm_np > 0).sum())}/{query_grid_h * query_grid_w}",
            fontsize=9,
            color=color,
        )
        axes[row, 1].axis("off")
        plt.colorbar(im1, ax=axes[row, 1], shrink=0.75, pad=0.02)

        # Col 2: detection overlay on full query
        patch_size = IMG_SIZE // query_grid_h
        heat = upsample_map(dm_np, q_arr.shape[0], q_arr.shape[1])
        blended = heat_overlay(q_arr, heat)
        axes[row, 2].imshow(blended)
        if len(peaks_np):
            # Scale peak patch-grid coords back to original image pixels
            px_x = (peaks_np[:, 0] + 0.5) * patch_size * q_arr.shape[1] / IMG_SIZE
            px_y = (peaks_np[:, 1] + 0.5) * patch_size * q_arr.shape[0] / IMG_SIZE
            axes[row, 2].scatter(
                px_x,
                px_y,
                c=color,
                s=150,
                marker="o",
                linewidths=1.5,
                edgecolors="white",
                zorder=5,
            )
        axes[row, 2].set_title(
            f"[{scale}] {len(peaks_np)} detection(s) on query",
            fontsize=9,
            color=color,
        )
        axes[row, 2].axis("off")

    fig.suptitle(
        f"Multi-scale detection comparison | query: {query_name}\n"
        f"DINOv{DINO_VERSION[1]}-{DINO_SIZE}  block {LAYER_IDX}  "
        f"mode={EXEMPLAR_MODE}  density_thr={DENSITY_THRESHOLD}",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


# ── Figure: combined detections from all scales on one image ───────────────────


def fig_combined_detections(
    query_img: Image.Image,
    query_tokens: torch.Tensor,
    query_grid_h: int,
    query_grid_w: int,
    scale_infos: list[dict],
    query_name: str,
    out_path: Path,
) -> None:
    """All scale detections overlaid on a single query image view."""
    fig, axes = plt.subplots(1, len(scale_infos) + 1, figsize=((len(scale_infos) + 1) * 6, 6))

    q_arr = np.array(query_img)
    patch_size = IMG_SIZE // query_grid_h

    # Right panel: all peaks together
    axes[-1].imshow(q_arr)

    for ax_idx, info in enumerate(scale_infos):
        scale = info["scale"]
        color = SCALE_COLOR[scale]

        dm = compute_density_map(
            query_tokens, info["feat"], query_grid_h, query_grid_w, DENSITY_THRESHOLD
        )
        peaks = extract_peaks(dm, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)
        peaks_np = peaks.cpu().numpy()

        # Individual panel
        heat = upsample_map(dm.cpu().numpy(), q_arr.shape[0], q_arr.shape[1])
        blended = heat_overlay(q_arr, heat)
        axes[ax_idx].imshow(blended)
        if len(peaks_np):
            px_x = (peaks_np[:, 0] + 0.5) * patch_size * q_arr.shape[1] / IMG_SIZE
            px_y = (peaks_np[:, 1] + 0.5) * patch_size * q_arr.shape[0] / IMG_SIZE
            axes[ax_idx].scatter(
                px_x,
                px_y,
                c=color,
                s=150,
                marker="o",
                linewidths=1.5,
                edgecolors="white",
                zorder=5,
            )
            axes[-1].scatter(
                px_x,
                px_y,
                c=color,
                s=150,
                marker="o",
                linewidths=1.5,
                edgecolors="white",
                zorder=5,
                label=f"{scale} ({len(peaks_np)})",
            )
        axes[ax_idx].set_title(
            f"Scale: {scale}  |  {len(peaks_np)} peak(s)", fontsize=9, color=color
        )
        axes[ax_idx].axis("off")

    axes[-1].legend(loc="upper right", fontsize=9, framealpha=0.85)
    axes[-1].set_title("All scales combined", fontsize=9)
    axes[-1].axis("off")

    fig.suptitle(
        f"Combined detections | query: {query_name}  |  "
        f"DINOv{DINO_VERSION[1]}-{DINO_SIZE}  block {LAYER_IDX}",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


# ── Figure: density map strip across scales at patch-grid resolution ───────────


def fig_density_strip(
    query_tokens: torch.Tensor,
    query_grid_h: int,
    query_grid_w: int,
    scale_infos: list[dict],
    query_name: str,
    out_path: Path,
) -> None:
    """Small grid-resolution density maps for all scales — useful for comparing fine structure."""
    n = len(scale_infos)
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 4))

    for ax, info in zip(axes, scale_infos):
        dm = compute_density_map(
            query_tokens, info["feat"], query_grid_h, query_grid_w, DENSITY_THRESHOLD
        )
        dm_np = dm.cpu().numpy()
        peaks = extract_peaks(dm, PEAK_KERNEL_SIZE, MIN_PEAK_THRESHOLD)
        peaks_np = peaks.cpu().numpy()

        im = ax.imshow(dm_np, cmap="jet", aspect="auto", vmin=0)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        if len(peaks_np):
            ax.scatter(
                peaks_np[:, 0],
                peaks_np[:, 1],
                c=SCALE_COLOR[info["scale"]],
                s=60,
                marker="x",
                linewidths=2,
                zorder=5,
            )
        ax.set_title(
            f"scale={info['scale']}\n{len(peaks_np)} peak(s)  max={dm_np.max():.3f}",
            fontsize=9,
            color=SCALE_COLOR[info["scale"]],
        )

    fig.suptitle(
        f"Density maps at patch-grid resolution | query: {query_name}"
        f"  ({query_grid_h}×{query_grid_w} grid)",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-scale detection quality experiment on abc2 data."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing {part_type}_*.jpg images and annotations/ folder.",
    )
    p.add_argument("--part-type", default="LHb", help="Image prefix, e.g. LHb")
    p.add_argument("--ref-number", type=int, default=1, help="Exemplar index (LHb_1.jpg → 1)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "multiscale_results",
        help="Directory where output figures are saved.",
    )
    p.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Local DINO weights directory (overrides DINO_WEIGHTS_DIR env var).",
    )
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = args.weights_dir or (
        Path(os.environ["DINO_WEIGHTS_DIR"]) if "DINO_WEIGHTS_DIR" in os.environ else None
    )

    # ── Load encoder ───────────────────────────────────────────────────────────
    encoder = DinoEncoder(
        version=DINO_VERSION,
        size=DINO_SIZE,
        img_size=IMG_SIZE,
        weights_dir=weights_dir,
    )
    log.info(
        "DINOv%s-%s | patch_size=%d | grid=%dx%d",
        DINO_VERSION[1],
        DINO_SIZE,
        encoder.patch_size,
        encoder.grid_h,
        encoder.grid_w,
    )

    # ── Load images and mask ───────────────────────────────────────────────────
    exemplar_path, query_paths = find_images(args.data_dir, args.part_type, args.ref_number)
    mask_path = args.data_dir / "annotations" / f"{args.part_type}_{args.ref_number}.npy"

    exemplar_img = Image.open(exemplar_path).convert("RGB")
    query_imgs = [(p, Image.open(p).convert("RGB")) for p in query_paths]

    log.info("Exemplar: %s  (%dx%d)", exemplar_path.name, *exemplar_img.size)
    for p, qi in query_imgs:
        log.info("Query:    %s  (%dx%d)", p.name, *qi.size)

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    pixel_mask = load_pixel_mask(mask_path)
    n_instances = np.load(mask_path).shape[0]
    log.info(
        "Mask: %s  shape=%s  instances=%d  coverage=%.2f%%",
        mask_path.name,
        pixel_mask.shape,
        n_instances,
        100.0 * pixel_mask.mean(),
    )

    rmin, rmax, cmin, cmax = mask_bbox(pixel_mask)
    log.info(
        "Mask bounding box: rows %d–%d  cols %d–%d  (%dx%d px)",
        rmin,
        rmax,
        cmin,
        cmax,
        cmax - cmin,
        rmax - rmin,
    )

    # ── Build exemplar descriptors for each scale ──────────────────────────────
    log.info("Building exemplar descriptors at %d scales…", len(SCALES))
    scale_infos: list[dict] = []
    for scale in SCALES:
        info = build_scale_descriptor(encoder, exemplar_img, pixel_mask, scale)
        scale_infos.append(info)

    # ── Overview figure: crop boxes on full exemplar ───────────────────────────
    fig_crop_overview(
        exemplar_img,
        pixel_mask,
        scale_infos,
        out_dir / "01_crop_overview.png",
    )

    # ── Per-query figures ──────────────────────────────────────────────────────
    for q_path, q_img in query_imgs:
        q_name = q_path.stem
        log.info("Processing query: %s", q_name)

        q_tokens, q_grid_h, q_grid_w = extract_patch_tokens(encoder, q_img, LAYER_IDX)
        log.info("Query grid: %dx%d  tokens: %s", q_grid_h, q_grid_w, tuple(q_tokens.shape))

        fig_detection_comparison(
            q_img,
            q_tokens,
            q_grid_h,
            q_grid_w,
            scale_infos,
            q_name,
            out_dir / f"02_detection_comparison_{q_name}.png",
        )

        fig_combined_detections(
            q_img,
            q_tokens,
            q_grid_h,
            q_grid_w,
            scale_infos,
            q_name,
            out_dir / f"03_combined_detections_{q_name}.png",
        )

        fig_density_strip(
            q_tokens,
            q_grid_h,
            q_grid_w,
            scale_infos,
            q_name,
            out_dir / f"04_density_strip_{q_name}.png",
        )

    log.info("All figures saved to %s", out_dir)


if __name__ == "__main__":
    main()
