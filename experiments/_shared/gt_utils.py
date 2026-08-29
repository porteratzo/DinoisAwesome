"""Ground-truth instance sizing, shared across the object_detection experiment scripts."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from dinoisawesome.annotation_utils import load_annotations

from .mask_geometry import pixel_mask_to_patch_mask

log = logging.getLogger(__name__)


def _gt_instances(stem: str, class_filter: list[str] | None, data_dir: Path) -> list[dict]:
    """Annotated instances (each with its own ``mask``) matching *class_filter*."""
    ann_stem = data_dir / "annotations" / stem
    try:
        anns = load_annotations(ann_stem)
    except FileNotFoundError:
        log.warning("No GT annotations at %s.", ann_stem)
        return []
    instances = [a for a in anns if class_filter is None or a["class"] in class_filter]
    if not instances:
        log.warning("No GT instances matched classes %s at %s.", class_filter, ann_stem)
    return instances


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
    instances = _gt_instances(stem, class_filter, data_dir)
    if not instances:
        return np.array([])
    return np.array(
        [
            pixel_mask_to_patch_mask(a["mask"], grid_h, grid_w, img_size, patch_threshold).sum()
            for a in instances
        ]
    )


def gt_instance_patch_masks(
    stem: str,
    class_filter: list[str] | None,
    grid_h: int,
    grid_w: int,
    img_size: int,
    patch_threshold: float,
    data_dir: Path,
) -> list[np.ndarray]:
    """Per-instance GT patch masks, one per annotated instance (never unioned/re-merged).

    Unlike deriving "instances" by DBSCAN-clustering a unioned GT mask, this preserves the
    true annotation boundary even when two distinct instances are touching or adjacent in
    patch space — a union-then-recluster approach would silently merge them into one blob.

    Instances that round down to zero patches under *patch_threshold* are dropped (they'd
    otherwise contribute a permanently-unmatched, always-empty GT cluster and inflate FN
    for an instance too small to ever be detected at this resolution).
    """
    instances = _gt_instances(stem, class_filter, data_dir)
    masks = [
        pixel_mask_to_patch_mask(a["mask"], grid_h, grid_w, img_size, patch_threshold)
        for a in instances
    ]
    return [m for m in masks if m.any()]
