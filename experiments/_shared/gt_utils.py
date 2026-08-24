"""Ground-truth instance sizing, shared across the object_detection experiment scripts."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from dinoisawesome.annotation_utils import load_annotations

from .mask_geometry import pixel_mask_to_patch_mask

log = logging.getLogger(__name__)


def gt_instance_patch_sizes(
    stem: str,
    class_filter: list[str] | None,
    grid_h: int,
    grid_w: int,
    img_size: int,
    patch_threshold: float,
    data_dir: Path,
) -> np.ndarray:
    """Per-instance GT sizes in patches, using that image's own annotation set."""
    ann_stem = data_dir / "annotations" / stem
    try:
        anns = load_annotations(ann_stem)
    except FileNotFoundError:
        log.warning("No GT annotations at %s — cannot derive cluster-size bounds.", ann_stem)
        return np.array([])
    instances = [a for a in anns if class_filter is None or a["class"] in class_filter]
    if not instances:
        log.warning("No GT instances matched classes %s at %s.", class_filter, ann_stem)
        return np.array([])
    return np.array(
        [
            pixel_mask_to_patch_mask(a["mask"], grid_h, grid_w, img_size, patch_threshold).sum()
            for a in instances
        ]
    )
