"""PCA-based foreground/background patch masking.

Ported from anomalib's ``AnomalyDINOModel.compute_background_masks`` (the
background-suppression trick from the AnomalyDINO paper) so the same masking
behaviour can be reused outside anomalib's DINOv2-only model — e.g. by
dinoisawesome's DINOv3-backed :class:`~dinoisawesome.anomaly_head.AnomalyHead`
and :class:`~dinoisawesome.prototype_head.PrototypeAnomalyHead`.

Unlike anomalib's version (``cv2.dilate`` / ``cv2.morphologyEx``), morphological
cleanup here uses ``scipy.ndimage`` with an equivalent 3x3 structuring element —
scipy is already pulled in transitively by scikit-learn, so this avoids adding
opencv as a new dependency.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation
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

    structure = np.ones((kernel_size, kernel_size), dtype=bool)
    mask_2d = binary_dilation(mask_2d, structure=structure)
    mask_2d = binary_closing(mask_2d, structure=structure)

    return mask_2d.reshape(-1)
