"""Score-map thresholding: Otsu, IoU-tuned threshold sweeps, and unsupervised ROI
binarization (used to find two-stage ROI blobs before re-encoding at native resolution).
"""

from __future__ import annotations

import cv2
import numpy as np

from .mask_geometry import mask_iou


def otsu_threshold(raw: np.ndarray) -> float:
    """Otsu's threshold on *raw*, computed via cv2 on a rescaled 8-bit copy."""
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo < 1e-8:
        return lo
    scaled = ((raw - lo) / (hi - lo) * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return lo + (float(otsu_val) / 255.0) * (hi - lo)


def iou_threshold_curve(
    raw: np.ndarray, gt_mask: np.ndarray, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Patch-mask IoU of `(raw > c)` vs. `gt_mask` for `steps` candidates spanning raw's range."""
    candidates = np.linspace(raw.min(), raw.max(), steps)
    ious = np.array([mask_iou(raw > c, gt_mask) for c in candidates])
    return candidates, ious


def iou_tuned_threshold(raw: np.ndarray, gt_mask: np.ndarray, steps: int) -> float:
    """Threshold that maximises patch-mask IoU against *gt_mask*.

    Meant to be fit on a reference/exemplar's own score map + GT mask (self-supervised,
    no dependence on the query's own GT), then reused as-is to threshold queries.
    """
    candidates, ious = iou_threshold_curve(raw, gt_mask, steps)
    return float(candidates[int(np.argmax(ious))])


def roi_binary_mask(
    raw: np.ndarray, method: str, single_threshold: float, percentile: float | None = None
) -> np.ndarray:
    """Unsupervised (no-GT) foreground/ROI mask used to decide where to zoom in.

    method: "otsu" | "single" | "percentile". *percentile* is required when
    method == "percentile" (keeps the top (100 - percentile)% of raw).
    """
    if method == "otsu":
        thr = otsu_threshold(raw)
    elif method == "single":
        thr = single_threshold
    elif method == "percentile":
        if percentile is None:
            raise ValueError("percentile is required when method == 'percentile'")
        thr = float(np.percentile(raw, percentile))
    else:
        raise ValueError(f"Unknown roi_binarize_method: {method!r}")
    return raw > thr
