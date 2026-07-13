"""Shared helpers for experiment notebooks and scripts.

Note: logging setup is intentionally *not* provided here. Configure
``logging.basicConfig(...)`` inline, before any ``dinoisawesome``/``torch``
import — importing this subpackage imports ``dinoisawesome``, which imports
``torch``, which may register its own handlers first.
"""

from .images import thumb, to_pil
from .masks import (
    load_instance_mask,
    mask_bbox_rc,
    mask_bbox_xywh,
    mask_iou,
    pixel_mask_to_patch_mask,
)
from .visualization import (
    draw_box,
    draw_points,
    heat_overlay,
    mask_overlay_rgba,
    pca_project_to_rgb,
    to_display_upscale,
    upsample_map,
)

__all__ = [
    "to_pil",
    "thumb",
    "load_instance_mask",
    "pixel_mask_to_patch_mask",
    "mask_bbox_rc",
    "mask_bbox_xywh",
    "mask_iou",
    "upsample_map",
    "heat_overlay",
    "mask_overlay_rgba",
    "draw_box",
    "draw_points",
    "pca_project_to_rgb",
    "to_display_upscale",
]
