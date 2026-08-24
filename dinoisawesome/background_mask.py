"""PCA-based foreground/background patch masking.

Ported from anomalib's ``AnomalyDINOModel.compute_background_masks`` (the
background-suppression trick from the AnomalyDINO paper) so the same masking
behaviour can be reused outside anomalib's DINOv2-only model — e.g. by
dinoisawesome's DINOv3-backed :class:`~dinoisawesome.anomaly_head.AnomalyHead`
and :class:`~dinoisawesome.prototype_head.PrototypeAnomalyHead`.

Morphological cleanup uses ``cv2.dilate`` / ``cv2.morphologyEx`` directly (matching
anomalib's own implementation), now that opencv is a declared project dependency
(see ``dinoisawesome.annotation_utils``, which also uses it — scipy's dense-footprint
``binary_*`` ops are dramatically slower than cv2's for the same rectangular kernel).
``borderType=cv2.BORDER_CONSTANT, borderValue=0`` is set explicitly to match scipy's
``border_value=0`` default (cv2's own default border value is effectively "treat the
border as foreground", which erodes edge-touching regions differently). With that set,
cv2 and scipy agree exactly for odd `kernel_size`; even sizes can differ by ~1px at a
mask's edge due to a center-pixel convention difference between the two libraries for
structuring elements with no single center — negligible for this heuristic cleanup.
"""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.decomposition import PCA


def compute_foreground_mask(
    patches: np.ndarray,
    grid_size: tuple[int, int],
    threshold: float = 10.0,
    kernel_size: int = 3,
    border: float = 0.2,
) -> np.ndarray:
    """Estimate which patches belong to the foreground object vs. background.

    Fits 1-component PCA over *patches* (raw, un-normalised patch embeddings of
    a single image) and thresholds the first principal component. PCA's sign is
    ambiguous, so it's auto-flipped whenever the image center — assumed to be
    foreground — would otherwise end up mostly unmasked. A dilate-then-close
    pass cleans up speckle noise in the resulting mask.

    Args:
        patches: ``(N, D)`` raw patch embeddings, where ``N == grid_size[0] *
            grid_size[1]``, flattened in row-major ``(H, W)`` order.
        grid_size: ``(H, W)`` patch grid dimensions.
        threshold: PCA threshold separating foreground from background.
        kernel_size: Side length of the square structuring element used for
            morphological cleanup.
        border: Fraction of the grid excluded from each edge when checking
            whether the center region is foreground-majority.

    Returns:
        Boolean ``(N,)`` mask, ``True`` == foreground patch. Falls back to an
        all-``True`` mask if the PCA threshold masks out every patch (e.g. a
        near-uniform texture with no separable foreground), so callers never
        see an empty foreground set.
    """
    h, w = grid_size
    pca = PCA(n_components=1, svd_solver="randomized")
    first_pc = pca.fit_transform(patches.astype(np.float32))
    mask_2d = (first_pc > threshold).reshape(h, w)

    y0, y1 = int(h * border), int(h * (1 - border))
    x0, x1 = int(w * border), int(w * (1 - border))
    center_crop = mask_2d[y0:y1, x0:x1]
    if center_crop.sum() <= center_crop.size * 0.35:
        mask_2d = (-first_pc > threshold).reshape(h, w)

    if not mask_2d.any():
        return np.ones(h * w, dtype=bool)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask_u8 = mask_2d.astype(np.uint8) * 255
    mask_u8 = cv2.dilate(mask_u8, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    mask_u8 = cv2.morphologyEx(
        mask_u8, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
    )
    mask_2d = mask_u8 > 0

    return mask_2d.reshape(-1)
