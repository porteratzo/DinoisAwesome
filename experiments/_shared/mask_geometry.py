"""Pixel/patch mask geometry: patch-grid projection, bounding boxes, IoU, crop-scale
boxes, and connected-component blob helpers shared across the object_detection and
fundamental experiment scripts.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage


def pixel_mask_to_patch_mask(
    pixel_mask: np.ndarray,
    grid_h: int,
    grid_w: int,
    img_size: int,
    threshold: float = 0.3,
) -> np.ndarray:
    """Resize a pixel-space mask to patch-grid resolution, (grid_h, grid_w) bool."""
    mask_pil = Image.fromarray(pixel_mask.astype(np.uint8) * 255)
    mask_resized = np.array(mask_pil.resize((img_size, img_size), Image.NEAREST)) > 0
    ph = img_size // grid_h
    pw = img_size // grid_w
    tiled = mask_resized.reshape(grid_h, ph, grid_w, pw)
    patch_density = tiled.mean(axis=(1, 3))
    return patch_density >= threshold


def mask_bbox_px(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of the True region as (rmin, rmax, cmin, cmax)."""
    rows, cols = np.where(mask)
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / (union + 1e-8)


def scale_crop_box(
    pixel_mask: np.ndarray, scale: str, padding_frac: float
) -> tuple[int, int, int, int]:
    """PIL-style crop box (x0, y0, x1, y1) for one of the three prototype scales.

    close  — tight bbox around the mask, padded by *padding_frac* of its own extent.
    mid    — midpoint between the close box and the full image.
    global — the entire image (no crop).
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
    global_box = (0, 0, W, H)

    if scale == "close":
        return close_box
    if scale == "global":
        return global_box
    if scale == "mid":
        return (
            (close_box[0] + global_box[0]) // 2,
            (close_box[1] + global_box[1]) // 2,
            (close_box[2] + global_box[2]) // 2,
            (close_box[3] + global_box[3]) // 2,
        )
    raise ValueError(f"Unknown scale: {scale!r}")


def connected_component_blobs(mask: np.ndarray) -> list[dict]:
    """8-connected components of a boolean patch mask -> list of {'mask': ...}."""
    structure = ndimage.generate_binary_structure(2, 2)
    labeled, n = ndimage.label(mask, structure=structure)
    return [{"mask": labeled == lbl} for lbl in range(1, n + 1)]


def blob_patch_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Half-open patch-grid bbox (y0, y1, x0, x1) of a boolean blob mask."""
    ys, xs = np.where(mask)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def patch_bbox_to_native_px(
    y0: int, y1: int, x0: int, x1: int, patch_size: int, scale_x: float, scale_y: float
) -> tuple[float, float, float, float]:
    """Patch-grid bbox -> native (original, full-resolution image) pixel bbox."""
    px0 = x0 * patch_size * scale_x
    px1 = x1 * patch_size * scale_x
    py0 = y0 * patch_size * scale_y
    py1 = y1 * patch_size * scale_y
    return px0, py0, px1, py1
