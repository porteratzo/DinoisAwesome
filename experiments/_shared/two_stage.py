"""Two-stage ROI-blob rescoring / cross-scale similarity machinery, shared by the
multi-scale-crop exemplar pipelines (engine.py and multiscale_crop_ablation.py).

These functions never call a method's ``score_method`` directly — each caller passes
its own scoring callable (``query_tokens -> raw_scores``, state already bound) as
*score_fn*, since the two pipelines' ``MethodState``/``score_method`` shapes have a
real representation difference (concatenated vs. separate fg/bg tensors for the
k-means fg-bg variant) that isn't just a config-plumbing difference.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from .clustering import dbscan_clusters, dbscan_clusters_from_mask
from .mask_geometry import mask_iou, pixel_mask_to_patch_mask

# State already bound by the caller (each pipeline's own score_method, possibly with
# method-specific kwargs like knn_k pre-applied via a closure/partial).
ScoreFn = Callable[[torch.Tensor], np.ndarray]


def cluster_mean_patch_score(
    cluster_mask: np.ndarray, tokens: torch.Tensor, mean_patch_prototype: torch.Tensor
) -> float:
    """Cosine similarity between one predicted cluster's own mean patch token (mean-pooled
    over the cluster's member patches, in *tokens*' own embedding space) and the training-time
    ``mean_patch_prototype``.
    """
    flat = torch.from_numpy(cluster_mask.reshape(-1)).to(tokens.device)
    cluster_tokens = tokens[flat]
    if cluster_tokens.shape[0] == 0:
        return float("-inf")
    mean_tok = F.normalize(cluster_tokens.mean(dim=0, keepdim=True), p=2, dim=-1)
    return float((mean_tok @ mean_patch_prototype.T).item())


def annotate_cluster_rejection(
    clusters: list[dict],
    tokens: torch.Tensor,
    mean_patch_prototype: torch.Tensor,
    threshold: float,
) -> None:
    """Mutates *clusters* in place: adds ``mean_patch_score`` and ``rejected`` (score below
    *threshold*) to every cluster dict, using *tokens*' own embedding space (query- or
    crop-space — cluster masks are always local to whichever grid *tokens* came from).
    """
    for cl in clusters:
        cl["mean_patch_score"] = cluster_mean_patch_score(cl["mask"], tokens, mean_patch_prototype)
        cl["rejected"] = cl["mean_patch_score"] < threshold


def tune_cluster_reject_threshold(
    ref_crops: list[Any],
    score_fn: ScoreFn,
    mean_patch_prototype: torch.Tensor,
    patch_thr: float,
    min_cs: int,
    iou_thr: float,
    pred_dbscan_eps: float,
    pred_dbscan_min_samples: int,
    gt_dbscan_eps: float,
    gt_dbscan_min_samples: int,
    cluster_reject_margin_fraction: float,
    min_points_floor: int,
    include_crop_viz_fields: bool = False,
) -> tuple[float, list[dict]]:
    """Self-supervised mean-patch cosine-similarity cutoff that separates predicted clusters
    that genuinely match a GT instance ("good") from spurious ones ("bad") — fit on the
    reference's own predicted clusters, found the same way query clusters are (score each
    *ref_crops* crop with *score_fn*, threshold at *patch_thr*, then DBSCAN with the same
    *min_cs*), reused as-is on query clusters.

    *ref_crops* elements need ``.tokens``, ``.grid_h``, ``.grid_w``, ``.exclude_patch_mask``,
    ``.cluster_idx`` attributes (plus ``.crop_img`` when *include_crop_viz_fields* is set —
    only ``multiscale_crop_ablation.py``'s ``ClusterCrop`` carries that field, used by its
    own inline plotting cells).

    Returns ``(threshold, ref_clusters)`` — each ref cluster dict gains ``mean_patch_score``,
    ``gt_iou`` (best IoU against any single GT instance in its own source crop), ``gt_good``
    (``gt_iou >= iou_thr``), and ``crop_idx``. If no ref cluster is "good" (nothing to anchor
    a margin to), falls back to a cutoff just below the global min score. Otherwise the
    cutoff is *cluster_reject_margin_fraction* below the smallest good score, clamped so it
    never drops to or below the largest bad score (when one exists).
    """
    ref_clusters: list[dict] = []
    for cc in ref_crops:
        raw = score_fn(cc.tokens).reshape(cc.grid_h, cc.grid_w)
        ys, xs = np.where(raw > patch_thr)
        if len(xs) < max(min_points_floor, min_cs):
            continue
        crop_clusters = dbscan_clusters(
            xs, ys, cc.grid_h, cc.grid_w, raw, pred_dbscan_eps, pred_dbscan_min_samples, min_cs
        )
        gt_instances = dbscan_clusters_from_mask(
            cc.exclude_patch_mask, gt_dbscan_eps, gt_dbscan_min_samples
        )
        for cl in crop_clusters:
            cl["mean_patch_score"] = cluster_mean_patch_score(
                cl["mask"], cc.tokens, mean_patch_prototype
            )
            cl["gt_iou"] = max(
                (mask_iou(cl["mask"], gt_inst["mask"]) for gt_inst in gt_instances), default=0.0
            )
            cl["gt_good"] = cl["gt_iou"] >= iou_thr
            cl["crop_idx"] = cc.cluster_idx
            if include_crop_viz_fields:
                cl["crop_img"] = cc.crop_img
                cl["crop_gt_mask"] = cc.exclude_patch_mask
        ref_clusters.extend(crop_clusters)

    if not ref_clusters:
        return float("-inf"), ref_clusters

    scores = np.array([cl["mean_patch_score"] for cl in ref_clusters])
    good = np.array([cl["gt_good"] for cl in ref_clusters])
    if not good.any():
        return float(scores.min()) - 1e-3, ref_clusters

    min_good = float(scores[good].min())
    margin_thr = min_good - cluster_reject_margin_fraction * abs(min_good)
    if (~good).any():
        max_bad = float(scores[~good].max())
        margin_thr = max(margin_thr, max_bad + 1e-3)
    return margin_thr, ref_clusters


def blob_crop_gt_mask(
    blob: dict, q_pixel_mask: np.ndarray | None, img_size: int, mask_patch_threshold: float
) -> np.ndarray:
    """Patch-grid GT mask for one ROI blob crop, aligned to its own re-encoded grid."""
    px0, py0, px1, py1 = blob["px_bbox"]
    c_h, c_w = blob["c_h"], blob["c_w"]
    crop_pixel_mask = q_pixel_mask[py0:py1, px0:px1] if q_pixel_mask is not None else None
    if crop_pixel_mask is None or not crop_pixel_mask.any():
        return np.zeros((c_h, c_w), dtype=bool)
    return pixel_mask_to_patch_mask(crop_pixel_mask, c_h, c_w, img_size, mask_patch_threshold)


def blob_crop_gt_instance_masks(
    blob: dict,
    q_instance_pixel_masks: list[np.ndarray] | None,
    img_size: int,
    mask_patch_threshold: float,
) -> list[np.ndarray]:
    """Per-instance patch-grid GT masks for one ROI blob crop, aligned to its own
    re-encoded grid — one entry per real annotated query instance that overlaps this
    blob's pixel bbox.

    Unlike ``blob_crop_gt_mask`` (which unions every instance into one mask for
    display), this keeps each instance separate so a caller doing per-instance
    TP/FP/FN cluster matching never has to reconstruct instances by DBSCAN-clustering
    a unioned mask — two distinct instances that are touching/adjacent within this
    blob would otherwise get silently re-merged into one (see
    ``_shared.gt_utils.gt_instance_patch_masks`` for the same fix at the pair level).
    """
    if not q_instance_pixel_masks:
        return []
    px0, py0, px1, py1 = blob["px_bbox"]
    c_h, c_w = blob["c_h"], blob["c_w"]
    masks = []
    for inst_pixel_mask in q_instance_pixel_masks:
        crop_pixel_mask = inst_pixel_mask[py0:py1, px0:px1]
        if not crop_pixel_mask.any():
            continue
        patch_mask = pixel_mask_to_patch_mask(
            crop_pixel_mask, c_h, c_w, img_size, mask_patch_threshold
        )
        if patch_mask.any():
            masks.append(patch_mask)
    return masks


def crop_patch_centers_to_native_px(
    xs: np.ndarray, ys: np.ndarray, px_bbox: tuple[int, int, int, int], c_h: int, c_w: int
) -> tuple[np.ndarray, np.ndarray]:
    """Crop-grid patch centers -> pixel coords in the *original* (uncropped) query image."""
    px0, py0, px1, py1 = px_bbox
    cell_w = (px1 - px0) / c_w
    cell_h = (py1 - py0) / c_h
    return px0 + (xs + 0.5) * cell_w, py0 + (ys + 0.5) * cell_h


def _project_area_weighted(
    cluster_mask: np.ndarray,
    px_bbox: tuple[int, int, int, int],
    c_h: int,
    c_w: int,
    q_h: int,
    q_w: int,
    patch_size: int,
    scale_x: float,
    scale_y: float,
    area_threshold: float,
) -> np.ndarray:
    """Area-weighted variant of ``project_crop_mask_to_query_grid``'s "nearest_union" mode:
    accumulates each foreground crop patch's real native-pixel rectangle overlap area into
    every query patch it actually intersects, then thresholds each query cell by the same
    area-fraction rule ``pixel_mask_to_patch_mask`` uses to build GT masks — instead of
    nearest_union's point-sample-then-OR, which isn't that rule and can silently over- or
    under-cover a query cell depending on the crop/query grid's relative resolution and
    the (generally non-aligned) offset between the crop's box and the query grid.
    """
    px0, py0, px1, py1 = px_bbox
    cell_w = (px1 - px0) / c_w
    cell_h = (py1 - py0) / c_h
    dx = patch_size * scale_x  # native px per query patch, x
    dy = patch_size * scale_y  # native px per query patch, y

    ys, xs = np.where(cluster_mask)
    area_acc = np.zeros((q_h, q_w), dtype=np.float64)
    for y, x in zip(ys.tolist(), xs.tolist()):
        fx0, fx1 = px0 + x * cell_w, px0 + (x + 1) * cell_w
        fy0, fy1 = py0 + y * cell_h, py0 + (y + 1) * cell_h

        col_lo = max(0, int(fx0 // dx))
        col_hi = min(q_w - 1, int(fx1 // dx))
        row_lo = max(0, int(fy0 // dy))
        row_hi = min(q_h - 1, int(fy1 // dy))
        for row in range(row_lo, row_hi + 1):
            overlap_h = min(fy1, (row + 1) * dy) - max(fy0, row * dy)
            if overlap_h <= 0:
                continue
            for col in range(col_lo, col_hi + 1):
                overlap_w = min(fx1, (col + 1) * dx) - max(fx0, col * dx)
                if overlap_w <= 0:
                    continue
                area_acc[row, col] += overlap_w * overlap_h

    cell_area = dx * dy
    if cell_area <= 0:
        return np.zeros((q_h, q_w), dtype=bool)
    return (area_acc / cell_area) >= area_threshold


def project_crop_mask_to_query_grid(
    cluster_mask: np.ndarray,
    px_bbox: tuple[int, int, int, int],
    c_h: int,
    c_w: int,
    q_h: int,
    q_w: int,
    patch_size: int,
    scale_x: float,
    scale_y: float,
    mode: Literal["nearest_union", "area_weighted"] = "nearest_union",
    area_threshold: float = 0.3,
) -> np.ndarray:
    """Project one crop-local cluster mask onto the full query's own (q_h, q_w) patch grid.

    ``mode="nearest_union"`` (default, legacy behaviour): each foreground crop patch is
    placed at its own center's native-pixel position (see
    ``crop_patch_centers_to_native_px``) and snapped to the nearest full-query patch —
    a point-sample union, not an area rule, so it can over- or under-cover a query cell
    relative to how GT masks are built.

    ``mode="area_weighted"``: see ``_project_area_weighted`` — uses the same area-fraction
    rule GT masks are built with (``pixel_mask_to_patch_mask``'s ``>= area_threshold``),
    so a two-stage prediction and its GT counterpart are computed by the same rule.
    """
    if mode == "area_weighted":
        return _project_area_weighted(
            cluster_mask, px_bbox, c_h, c_w, q_h, q_w, patch_size, scale_x, scale_y, area_threshold
        )
    if mode != "nearest_union":
        raise ValueError(f"Unknown projection mode: {mode!r}")

    ys, xs = np.where(cluster_mask)
    mask = np.zeros((q_h, q_w), dtype=bool)
    if len(xs) == 0:
        return mask
    native_x, native_y = crop_patch_centers_to_native_px(xs, ys, px_bbox, c_h, c_w)
    cols = np.clip((native_x / (patch_size * scale_x)).astype(int), 0, q_w - 1)
    rows = np.clip((native_y / (patch_size * scale_y)).astype(int), 0, q_h - 1)
    mask[rows, cols] = True
    return mask


def merge_overlapping_clusters(clusters: list[dict]) -> list[dict]:
    """Greedy largest-first merge: drop any cluster whose (already-projected) ``mask`` shares
    a patch with a bigger cluster already kept.
    """
    ordered = sorted(clusters, key=lambda c: -c["size"])
    kept: list[dict] = []
    for cl in ordered:
        if any((cl["mask"] & k["mask"]).any() for k in kept):
            continue
        kept.append(cl)
    return kept


def cross_score_blobs(
    blobs: list[dict],
    scale_states: dict[str, Any],
    q_pixel_mask: np.ndarray | None,
    per_method: dict[str, dict],
    score_fn_for: Callable[[Any], ScoreFn],
    img_size: int,
    mask_patch_threshold: float,
) -> list[dict]:
    """Score every blob crop with every candidate *scale_states* prototype.

    Binarises each crop's score map with *that prototype's own* tuned threshold and
    measures IoU against the crop's own ground-truth patch mask. *score_fn_for(state)*
    returns the scoring callable for one method state (each caller's own ``score_method``,
    possibly with method-specific kwargs like ``knn_k`` already bound).
    """
    rows: list[dict] = []
    for blob in blobs:
        c_h, c_w = blob["c_h"], blob["c_w"]
        crop_gt_mask = blob_crop_gt_mask(blob, q_pixel_mask, img_size, mask_patch_threshold)
        for score_scale, state in scale_states.items():
            score_fn = score_fn_for(state)
            crop_raw = score_fn(blob["crop_tokens"]).reshape(c_h, c_w)
            thr = per_method[score_scale]["threshold"]
            iou = mask_iou(crop_raw > thr, crop_gt_mask)
            rows.append(
                {"score_scale": score_scale, "iou": iou, "gt_present": bool(crop_gt_mask.any())}
            )
    return rows


def two_stage_predicted_clusters(
    blobs: list[dict],
    score_fn: ScoreFn,
    threshold: float,
    mean_patch_prototype: torch.Tensor,
    cluster_reject_thr: float,
    min_cs: int,
    q_h: int,
    q_w: int,
    patch_size: int,
    scale_x: float,
    scale_y: float,
    pred_dbscan_eps: float,
    pred_dbscan_min_samples: int,
    min_points_floor: int,
    q_pixel_mask: np.ndarray | None = None,
    q_instance_pixel_masks: list[np.ndarray] | None = None,
    img_size: int | None = None,
    mask_patch_threshold: float | None = None,
    projection_mode: Literal["nearest_union", "area_weighted"] = "nearest_union",
    collect_diagnostics: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Predicted clusters for the two-stage pipeline's P/R/F1: one method state used for both
    stage-1 ROI discovery (which found *blobs*) and stage-2 re-scoring (*score_fn*/*threshold*,
    that same state's own tuned prototype/threshold).

    Each crop's own re-scored patches are thresholded, DBSCAN-clustered, and mean-patch-rejected
    against *cluster_reject_thr* exactly like ``annotate_cluster_rejection`` does for
    single-step. Surviving clusters are projected onto the shared (q_h, q_w) query grid (see
    ``project_crop_mask_to_query_grid``'s *mode* — pass *projection_mode* to opt into
    "area_weighted" instead of the legacy "nearest_union" point-sample; *mask_patch_threshold*
    doubles as that mode's area-fraction cutoff, defaulting to 0.3 when not given even if
    *collect_diagnostics* is off) and merged across every blob, keeping the biggest of any that
    collide.

    Returns ``(merged_predicted_clusters, blob_diagnostics)``. ``merged_predicted_clusters``
    (used for P/R/F1 via ``match_and_score``) is computed independent of *collect_diagnostics*.
    When *collect_diagnostics* is True (requires *q_pixel_mask*/*img_size*/*mask_patch_threshold*),
    ``blob_diagnostics`` gets one entry per blob — its own raw score map, threshold, every
    DBSCAN cluster found in it (including rejected ones), its own (unioned) GT patch mask for
    display, and — when *q_instance_pixel_masks* is also given — that same GT split into one
    patch mask per real annotated query instance (``gt_instance_masks``), for callers doing
    per-instance TP/FP/FN matching (see ``blob_crop_gt_instance_masks``). Otherwise
    ``blob_diagnostics`` is ``[]``.
    """
    area_threshold = 0.3 if mask_patch_threshold is None else mask_patch_threshold
    candidates: list[dict] = []
    diagnostics: list[dict] = []
    for blob in blobs:
        c_h, c_w = blob["c_h"], blob["c_w"]
        crop_raw = score_fn(blob["crop_tokens"]).reshape(c_h, c_w)
        ys, xs = np.where(crop_raw > threshold)
        if len(xs) < max(min_points_floor, min_cs):
            crop_clusters: list[dict] = []
        else:
            crop_clusters = dbscan_clusters(
                xs, ys, c_h, c_w, crop_raw, pred_dbscan_eps, pred_dbscan_min_samples, min_cs
            )
            annotate_cluster_rejection(
                crop_clusters, blob["crop_tokens"], mean_patch_prototype, cluster_reject_thr
            )
            for cl in crop_clusters:
                if cl["rejected"]:
                    continue
                projected = project_crop_mask_to_query_grid(
                    cl["mask"],
                    blob["px_bbox"],
                    c_h,
                    c_w,
                    q_h,
                    q_w,
                    patch_size,
                    scale_x,
                    scale_y,
                    mode=projection_mode,
                    area_threshold=area_threshold,
                )
                if not projected.any():
                    continue
                candidates.append(
                    {"mask": projected, "score": cl["score"], "size": int(cl["mask"].sum())}
                )
        if collect_diagnostics:
            assert q_pixel_mask is not None and img_size is not None
            assert mask_patch_threshold is not None
            diagnostics.append(
                {
                    "raw": crop_raw.astype(np.float32),
                    "threshold": threshold,
                    "clusters": [
                        {
                            "mask": cl["mask"],
                            "score": cl["score"],
                            "rejected": cl["rejected"],
                            "mean_patch_score": cl["mean_patch_score"],
                        }
                        for cl in crop_clusters
                    ],
                    "gt_mask": blob_crop_gt_mask(
                        blob, q_pixel_mask, img_size, mask_patch_threshold
                    ),
                    "gt_instance_masks": blob_crop_gt_instance_masks(
                        blob, q_instance_pixel_masks, img_size, mask_patch_threshold
                    ),
                    "px_bbox": blob["px_bbox"],
                    "c_h": c_h,
                    "c_w": c_w,
                }
            )
    return merge_overlapping_clusters(candidates), diagnostics
