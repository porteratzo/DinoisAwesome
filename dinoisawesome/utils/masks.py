"""Instance-mask I/O and mask <-> patch-grid conversions.

These helpers assume the "stretch" resize DinoEncoder actually applies
(``transforms.Resize((img_size, img_size))``, no center crop). Pipelines that
pre-process crops with an aspect-preserving Resize + CenterCrop (see
``scripts/eval_sam_dino.py``) use a different mask->patch-grid projection and
are intentionally not merged into these functions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_instance_mask(path: str | Path, key: str = "segmaps") -> np.ndarray:
    """Load an instance segmentation mask file, unioned to a single ``(H, W)`` bool mask.

    Supports the three formats used across this repo's datasets:
      - ``.npy``: ``(N, H, W)`` array, one channel per instance.
      - ``.npz``: dict-like archive with an ``(H, W, N)`` array under *key*.
      - any other image extension: single-channel mask, thresholded ``> 0``.
    """
    path = Path(path)
    if path.suffix == ".npy":
        raw = np.load(path)  # (N, H, W)
        raw = raw.transpose(1, 2, 0)  # (H, W, N)
    elif path.suffix == ".npz":
        raw = np.load(path)[key]  # (H, W, N)
    else:
        raw = np.asarray(Image.open(path).convert("L"))[:, :, None] > 0
    return raw.any(axis=2)


def pixel_mask_to_patch_mask(
    pixel_mask: np.ndarray,
    grid_h: int,
    grid_w: int,
    img_size: int,
    threshold: float = 0.3,
) -> np.ndarray:
    """Resize a pixel-space boolean mask to ``(grid_h, grid_w)`` patch-grid resolution.

    Steps: resize to ``img_size x img_size`` (nearest-neighbour), then a patch
    is True if at least *threshold* fraction of its pixels are True.
    """
    mask_pil = Image.fromarray(pixel_mask.astype(np.uint8) * 255)
    resized = np.array(mask_pil.resize((img_size, img_size), Image.NEAREST)) > 0
    ph = img_size // grid_h
    pw = img_size // grid_w
    tiled = resized.reshape(grid_h, ph, grid_w, pw)
    return tiled.mean(axis=(1, 3)) >= threshold  # (grid_h, grid_w) bool


def mask_bbox_rc(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of the True region as ``(rmin, rmax, cmin, cmax)``, inclusive."""
    rows, cols = np.where(mask)
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def mask_bbox_xywh(mask: np.ndarray) -> list[int]:
    """Bounding box of the True region as ``[x, y, w, h]``."""
    rmin, rmax, cmin, cmax = mask_bbox_rc(mask)
    return [cmin, rmin, cmax - cmin + 1, rmax - rmin + 1]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-Union between two boolean masks of the same shape."""
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection) / (float(union) + 1e-8)
