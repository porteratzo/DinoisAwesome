"""DBSCAN-based foreground clustering and greedy score-ordered IoU matching against
ground-truth clusters.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.cluster import DBSCAN

from .mask_geometry import mask_iou


def patch_radius_to_eps(n_pixels: float) -> float:
    """Convert an 8-connected pixel/patch radius to a DBSCAN eps (Euclidean)."""
    return n_pixels * math.sqrt(2)


def dbscan_clusters_from_mask(
    patch_mask: np.ndarray,
    max_px_separation: float,
    min_samples: int,
    crop_w_px: float | None = None,
    crop_h_px: float | None = None,
    raw: np.ndarray | None = None,
) -> list[dict]:
    """Cluster foreground patch coords with plain DBSCAN (patch-index space by default,
    or native-pixel space when *crop_w_px*/*crop_h_px* are given).

    ``max_px_separation`` is interpreted two ways depending on whether the crop's absolute
    size is given:

    - ``crop_w_px``/``crop_h_px`` omitted (default) — patch-index-space eps. Patch
      coordinates are used as-is, so e.g. 1.5 means "adjacent or diagonal patch in this
      mask's own grid" regardless of that grid's physical scale. Correct for recovering
      connected components from a *known* mask (GT clustering with ``min_samples=1``)
      since grid adjacency, not physical distance, is what defines one instance there.
    - ``crop_w_px``/``crop_h_px`` given — patch coordinates are first projected to real
      native-pixel coordinates via the crop's own absolute size (normalized grid position
      x crop size), then ``max_px_separation`` is a genuine native-pixel linking distance.
      Needed when clustering *predicted* foreground points into instances: a fixed
      patch-count eps means a different physical distance depending on how zoomed-in the
      crop is, so the same numeric eps behaves inconsistently across scales unless
      converted like this.

    ``raw`` is optional and only needed when the result will be used as *predicted*
    clusters (i.e. passed to ``match_and_score`` as ``pred_clusters``, which ranks by
    score) — it attaches each cluster's own max raw score. GT clusters never need it.
    """
    ys, xs = np.where(patch_mask)
    if len(xs) == 0:
        return []
    if crop_w_px is not None and crop_h_px is not None:
        grid_h, grid_w = patch_mask.shape
        xs_c = xs / max(grid_w - 1, 1) * crop_w_px
        ys_c = ys / max(grid_h - 1, 1) * crop_h_px
    else:
        xs_c, ys_c = xs.astype(float), ys.astype(float)
    coords = np.stack([xs_c, ys_c], axis=1)
    labels = DBSCAN(eps=max_px_separation, min_samples=min_samples).fit_predict(coords)
    clusters = []
    for lbl in sorted(set(labels)):
        if lbl == -1:
            continue
        sel = labels == lbl
        mask = np.zeros(patch_mask.shape, dtype=bool)
        mask[ys[sel], xs[sel]] = True
        cluster: dict = {"mask": mask}
        if raw is not None:
            cluster["score"] = float(raw[ys[sel], xs[sel]].max())
        clusters.append(cluster)
    return clusters


def dbscan_clusters(
    xs: np.ndarray,
    ys: np.ndarray,
    grid_h: int,
    grid_w: int,
    raw: np.ndarray,
    eps: float,
    min_samples: int,
    min_cs: int | None = None,
) -> list[dict]:
    """Cluster foreground patch coords (already thresholded, given as xs/ys) with plain
    DBSCAN (patch-index space), post-filtered by minimum cluster size.

    Thin wrapper around ``dbscan_clusters_from_mask`` — builds the boolean mask from
    xs/ys, clusters it, and drops any cluster smaller than *min_cs* (DBSCAN has no
    notion of a cluster-size bound the way HDBSCAN's ``min_cluster_size`` did, so an
    undersized cluster here is dropped rather than merged into a neighbour).
    """
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    mask[ys, xs] = True
    clusters = dbscan_clusters_from_mask(mask, eps, min_samples, raw=raw)
    if min_cs is not None:
        clusters = [c for c in clusters if c["mask"].sum() >= min_cs]
    return clusters


def min_cluster_size_bound(sizes: np.ndarray, margin: int | float, floor: int) -> int:
    """Minimum cluster size from the smallest GT instance size, margin-relaxed downward.

    ``margin`` as an int is a flat patch-count offset: k_min - margin. As a float it's
    read as a fraction of the bound: k_min - margin*k_min.
    """
    if len(sizes) == 0:
        return floor
    k_min = float(sizes.min())
    lo = k_min - margin * k_min if isinstance(margin, float) else k_min - margin
    return max(floor, int(round(lo)))


def match_and_score(pred_clusters: list[dict], gt_clusters: list[dict], iou_thr: float) -> dict:
    """Greedy score-ordered IoU matching of predicted clusters against GT clusters ->
    precision/recall/F1/mean_iou/tp/fp/fn/count_error.
    """
    n_pred, n_gt = len(pred_clusters), len(gt_clusters)
    if n_gt == 0:
        return {
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "mean_iou": np.nan,
            "tp": 0,
            "fp": n_pred,
            "fn": 0,
            "count_error": n_pred,
            "n_pred": n_pred,
            "n_gt": 0,
        }
    order = sorted(range(n_pred), key=lambda i: -pred_clusters[i]["score"])
    matched_gt: set[int] = set()
    tp = 0
    ious: list[float] = []
    for i in order:
        best_j, best_iou = -1, 0.0
        for j in range(n_gt):
            if j in matched_gt:
                continue
            iou = mask_iou(pred_clusters[i]["mask"], gt_clusters[j]["mask"])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thr:
            matched_gt.add(best_j)
            tp += 1
            ious.append(best_iou)
    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "count_error": abs(n_pred - n_gt),
        "n_pred": n_pred,
        "n_gt": n_gt,
    }
