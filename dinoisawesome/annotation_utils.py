"""Utilities for loading paired mask / metadata annotations.

Each annotation set is stored as two files with the same stem:
  - ``<stem>.npy``  — boolean array of shape ``(N, H, W)``; one mask per instance
  - ``<stem>.json`` — list of N dicts with keys: class, instance_id, prompt_type,
                      prompt_content, image_path

:func:`load_annotations` loads both, applies morphological opening to each mask
(removes speckle noise while preserving large blobs), and returns a list of
per-instance dicts that merge the JSON metadata with a ``mask`` field.
"""

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)


def load_annotations(
    annotation_stem: str | Path,
    morph_kernel_size: int = 20,
) -> list[dict[str, Any]]:
    """Load a paired ``.npy`` / ``.json`` annotation and return cleaned instances.

    Parameters
    ----------
    annotation_stem:
        Path to the annotation files **without extension**, e.g.
        ``"data/abc3/annotations/LHa_1"``.  Both ``<stem>.npy`` and
        ``<stem>.json`` must exist.
    morph_kernel_size:
        Side length (pixels) of the square structuring element used for
        morphological opening.  Larger values remove bigger specks but may
        trim thin true-positive regions.  Default ``20`` works well for
        4K-resolution masks.

    Returns
    -------
    list[dict]
        One dict per instance, in the same order as the JSON list.  Each dict
        contains all JSON fields (``class``, ``instance_id``, ``prompt_type``,
        ``prompt_content``, ``image_path``) plus:

        ``mask`` : np.ndarray, shape ``(H, W)``, dtype ``bool``
            The cleaned binary mask for this instance.
    """
    stem = Path(annotation_stem)
    npy_path = stem.with_suffix(".npy")
    json_path = stem.with_suffix(".json")

    if not npy_path.exists():
        raise FileNotFoundError(f"Mask file not found: {npy_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {json_path}")

    masks: np.ndarray = np.load(npy_path, allow_pickle=False)  # (N, H, W) bool
    with open(json_path) as f:
        metadata: list[dict[str, Any]] = json.load(f)

    if masks.ndim != 3:
        raise ValueError(f"Expected (N, H, W) mask array, got shape {masks.shape}")
    if len(masks) != len(metadata):
        raise ValueError(
            f"Mask count ({len(masks)}) does not match annotation count ({len(metadata)})"
        )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    log.debug(
        "Loading %d annotations from %s; opening kernel %dx%d",
        len(masks),
        stem,
        morph_kernel_size,
        morph_kernel_size,
    )

    result = []
    for i, (mask, meta) in enumerate(zip(masks, metadata)):
        mask_u8 = mask.astype(np.uint8) * 255
        cleaned = (
            cv2.morphologyEx(
                mask_u8, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
            )
            > 0
        )
        entry = dict(meta)
        entry["mask"] = cleaned
        log.debug(
            "  [%d] class=%r  pixels before=%d  after=%d",
            i,
            meta.get("class"),
            int(mask.sum()),
            int(cleaned.sum()),
        )
        result.append(entry)

    return result
