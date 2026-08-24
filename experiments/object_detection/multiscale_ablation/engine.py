"""Domain engine: crop geometry, multi-scale exemplar prototypes, scoring, DBSCAN
clustering + GT matching, and the two-stage ROI-blob re-scoring / cross-scale
similarity mechanism.

Ported from ``object_detection/multiscale_crop_ablation.py`` (see that file's module
docstring for the experiment background) and parameterised on :class:`CropConfig` /
:class:`ScoringConfig` instead of module-level constants, so ``run_experiments.py``
can cache by config hash. This module has no notion of "which method" beyond the
generic :class:`MethodState` (a set of L2-normalised prototype rows plus a "kind"
that says how to turn per-patch cosine similarities into one score) — the specific
methods (mean, k-means, fg-bg, ...) are built in ``methods.py``.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from common import DATA_DIR, CropConfig, ScoringConfig
from PIL import Image
from scipy import ndimage

from dinoisawesome import DinoEncoder
from dinoisawesome.instance_detection import compute_exemplar_features

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# dbscan_clusters_from_mask / min_cluster_size_bound / annotate_cluster_rejection /
# match_and_score / iou_tuned_threshold are re-exported: run_experiments.py imports all
# five straight from this module rather than from _shared.
from _shared.clustering import (  # noqa: E402
    dbscan_clusters as _shared_dbscan_clusters,
)
from _shared.clustering import (
    dbscan_clusters_from_mask,  # noqa: F401
    match_and_score,  # noqa: F401
    min_cluster_size_bound,  # noqa: F401
    patch_radius_to_eps,
)
from _shared.crop_utils import pad_and_floor_crop_box  # noqa: E402
from _shared.gt_utils import (  # noqa: E402
    gt_instance_patch_sizes as _shared_gt_instance_patch_sizes,
)
from _shared.mask_geometry import (  # noqa: E402
    blob_patch_bbox,
    connected_component_blobs,
    patch_bbox_to_native_px,
    pixel_mask_to_patch_mask,
    scale_crop_box,
)
from _shared.prototype_ops import extract_patch_tokens_batch, knn_fgbg_score  # noqa: E402
from _shared.thresholding import iou_tuned_threshold  # noqa: E402, F401
from _shared.thresholding import roi_binary_mask as _shared_roi_binary_mask  # noqa: E402
from _shared.two_stage import (  # noqa: E402
    annotate_cluster_rejection,  # noqa: F401
)
from _shared.two_stage import (
    blob_crop_gt_mask as _shared_blob_crop_gt_mask,
)
from _shared.two_stage import (
    cross_score_blobs as _shared_cross_score_blobs,
)
from _shared.two_stage import (
    tune_cluster_reject_threshold as _shared_tune_cluster_reject_threshold,
)
from _shared.two_stage import (
    two_stage_predicted_clusters as _shared_two_stage_predicted_clusters,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mask / geometry helpers
# ---------------------------------------------------------------------------


def gt_instance_patch_sizes(
    stem: str,
    class_filter: list[str] | None,
    grid_h: int,
    grid_w: int,
    crop_cfg: CropConfig,
) -> np.ndarray:
    """Per-instance GT sizes in patches, using that image's own annotation set."""
    return _shared_gt_instance_patch_sizes(
        stem,
        class_filter,
        grid_h,
        grid_w,
        crop_cfg.img_size,
        crop_cfg.mask_patch_threshold,
        DATA_DIR,
    )


# ---------------------------------------------------------------------------
# Multi-scale exemplar prototype builder
# ---------------------------------------------------------------------------


@dataclass
class ClusterCrop:
    """One instance's own crop at one scale (mid/close), pre-aggregation."""

    cluster_idx: int
    box: tuple[int, int, int, int]  # (x0, y0, x1, y1) in the reference image
    tokens: torch.Tensor  # (H*W, C) L2-normalised, from this cluster's own crop
    grid_h: int
    grid_w: int
    patch_mask: np.ndarray  # (grid_h, grid_w) bool — this instance's mask within the crop
    exclude_patch_mask: np.ndarray  # union of every instance's mask within this crop


@dataclass
class ScalePrototype:
    scale: str
    box: tuple[int, int, int, int]  # representative crop's box (largest cluster for mid/close)
    tokens: torch.Tensor  # (H*W, C) L2-normalised, from the representative crop
    grid_h: int
    grid_w: int
    patch_mask: np.ndarray  # (grid_h, grid_w) bool — object mask within the representative crop
    mean_prototype: torch.Tensor  # (1, C) L2-normalised — mean of all cluster prototypes
    bg_prototype: torch.Tensor  # (1, C) L2-normalised — mean of all cluster backgrounds
    cluster_crops: list[ClusterCrop] | None = None  # per-instance crops; None for "global"


def build_all_scale_prototypes(
    encoder: DinoEncoder,
    ref_img: Image.Image,
    instance_masks: list[np.ndarray],
    crop_cfg: CropConfig,
) -> tuple[dict[str, ScalePrototype], torch.Tensor]:
    """Build one exemplar prototype per scale (mean-collapsed) plus every cluster's own
    tokens/masks, encoding all kept crops in one batch.

    Method-agnostic: only ever computes ``mean_prototype``/``bg_prototype`` via a plain
    masked mean — methods that want a different aggregation (k-means, ...) read
    ``cluster_crops``' raw tokens straight off the returned :class:`ScalePrototype` and
    re-aggregate themselves (see ``methods.py``), so this function only needs to run once
    per (image, crop_cfg) regardless of which methods are being compared.
    """
    union_mask = np.stack(instance_masks).any(axis=0)
    H, W = union_mask.shape
    global_box = (0, 0, W, H)

    pending: list[dict] = [
        {
            "scale": "global",
            "cluster_idx": -1,
            "box": global_box,
            "crop_img": ref_img.crop(global_box),
            "mask_crop": union_mask,
            "union_mask_crop": union_mask,
        }
    ]
    for cluster_idx, inst_mask in enumerate(instance_masks):
        for scale in ("mid", "close"):
            if scale not in crop_cfg.scales:
                continue
            box = scale_crop_box(inst_mask, scale, crop_cfg.exemplar_close_padding_fraction)
            x0, y0, x1, y1 = box
            if scale == "close" and (
                x1 - x0 < crop_cfg.min_crop_size or y1 - y0 < crop_cfg.min_crop_size
            ):
                log.warning(
                    "scale=close cluster=%d: crop %dx%dpx below min_crop_size=%dpx — dropping",
                    cluster_idx,
                    x1 - x0,
                    y1 - y0,
                    crop_cfg.min_crop_size,
                )
                continue
            pending.append(
                {
                    "scale": scale,
                    "cluster_idx": cluster_idx,
                    "box": box,
                    "crop_img": ref_img.crop(box),
                    "mask_crop": inst_mask[y0:y1, x0:x1],
                    "union_mask_crop": union_mask[y0:y1, x0:x1],
                }
            )

    tokens_batch = extract_patch_tokens_batch(
        encoder, [p["crop_img"] for p in pending], crop_cfg.layer_idx, crop_cfg.debias
    )

    global_crop: ClusterCrop | None = None
    clusters_by_scale: dict[str, list[ClusterCrop]] = {"mid": [], "close": []}
    for entry, (tokens, grid_h, grid_w) in zip(pending, tokens_batch):
        patch_mask = pixel_mask_to_patch_mask(
            entry["mask_crop"], grid_h, grid_w, crop_cfg.img_size, crop_cfg.mask_patch_threshold
        )
        exclude_patch_mask = pixel_mask_to_patch_mask(
            entry["union_mask_crop"],
            grid_h,
            grid_w,
            crop_cfg.img_size,
            crop_cfg.mask_patch_threshold,
        )

        cc = ClusterCrop(
            entry["cluster_idx"],
            entry["box"],
            tokens,
            grid_h,
            grid_w,
            patch_mask,
            exclude_patch_mask,
        )
        if entry["scale"] == "global":
            global_crop = cc
        else:
            clusters_by_scale[entry["scale"]].append(cc)

    def _masked_mean(cc: ClusterCrop, want_fg: bool) -> torch.Tensor:
        flat = torch.from_numpy((cc.patch_mask if want_fg else cc.exclude_patch_mask).reshape(-1))
        flat = flat.to(cc.tokens.device)
        sel = cc.tokens[flat] if want_fg else cc.tokens[~flat]
        if sel.shape[0] == 0:
            log.warning("empty %s selection — using all crop patches", "fg" if want_fg else "bg")
            sel = cc.tokens
        return compute_exemplar_features(sel, mode="mean")

    protos: dict[str, ScalePrototype] = {}
    assert global_crop is not None
    protos["global"] = ScalePrototype(
        "global",
        global_crop.box,
        global_crop.tokens,
        global_crop.grid_h,
        global_crop.grid_w,
        global_crop.patch_mask,
        _masked_mean(global_crop, want_fg=True),
        _masked_mean(global_crop, want_fg=False),
        cluster_crops=None,
    )

    for scale in ("mid", "close"):
        if scale not in crop_cfg.scales:
            continue
        clusters = clusters_by_scale[scale]
        if not clusters:
            log.warning("scale=%s: every cluster crop was dropped — dropping this scale", scale)
            continue
        avg = F.normalize(
            torch.cat([_masked_mean(c, True) for c in clusters], dim=0).mean(dim=0, keepdim=True),
            p=2,
            dim=-1,
        )
        bg_avg = F.normalize(
            torch.cat([_masked_mean(c, False) for c in clusters], dim=0).mean(dim=0, keepdim=True),
            p=2,
            dim=-1,
        )
        rep = max(clusters, key=lambda c: int(c.patch_mask.sum()))
        protos[scale] = ScalePrototype(
            scale,
            rep.box,
            rep.tokens,
            rep.grid_h,
            rep.grid_w,
            rep.patch_mask,
            avg,
            bg_avg,
            cluster_crops=clusters,
        )

    all_instance_protos = clusters_by_scale["mid"] + clusters_by_scale["close"]
    if all_instance_protos:
        mean_patch_prototype = F.normalize(
            torch.cat([_masked_mean(c, True) for c in all_instance_protos], dim=0).mean(
                dim=0, keepdim=True
            ),
            p=2,
            dim=-1,
        )
    else:
        # mid/close both disabled in crop_cfg.scales — fall back to global's own fg mean.
        mean_patch_prototype = protos["global"].mean_prototype
    return protos, mean_patch_prototype


def pool_scale_patches(
    scale_protos: dict[str, ScalePrototype], scales: list[str], want_fg: bool
) -> torch.Tensor:
    """Pool masked (want_fg=True) or unmasked (False) patch tokens across *scales*.

    Pools across every cluster crop within each scale. For want_fg=False, a
    ClusterCrop pools patches outside its ``exclude_patch_mask`` (union of every
    instance's mask within that crop), not just outside its own ``patch_mask`` — a
    neighbouring instance inside the same crop box must not leak into the bg pool.
    """
    chunks: list[torch.Tensor] = []
    for s in scales:
        proto = scale_protos[s]
        cells = proto.cluster_crops if proto.cluster_crops is not None else [proto]
        for cell in cells:
            if want_fg:
                sel_mask = cell.patch_mask
            else:
                sel_mask = (
                    cell.exclude_patch_mask if isinstance(cell, ClusterCrop) else cell.patch_mask
                )
            sel_flat = torch.from_numpy(sel_mask.reshape(-1)).to(cell.tokens.device)
            chunks.append(cell.tokens[sel_flat] if want_fg else cell.tokens[~sel_flat])
    return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# Method scoring — generic given a MethodState (see methods.py for construction)
# ---------------------------------------------------------------------------


@dataclass
class MethodState:
    name: str
    kind: Literal["single", "multi", "fgbg", "knn_fgbg", "fgbg_multi"]
    # (K, C) L2-normalised prototype(s). "fgbg": [fg, bg] (2, C). "fgbg_multi": fg rows
    # followed by one bg row (K+1, C). None for "knn_fgbg" (uses fg_bank/bg_bank).
    payload: torch.Tensor | None = None
    fg_bank: torch.Tensor | None = None  # (Nfg, C) L2-normalised patch gallery
    bg_bank: torch.Tensor | None = None  # (Nbg, C) L2-normalised patch gallery
    # Name of the method (in the same per-method dict, possibly itself) whose own raw
    # map should locate two-stage ROI blobs for this method (see run_experiments.py).
    # A method is "self-anchored" when this equals its own name (e.g. every plain
    # single-scale method); fg-bg / multi-scale-combine methods point at whichever
    # single-scale method their foreground side is actually built from.
    roi_source_method: str | None = None


def score_method(state: MethodState, query_tokens: torch.Tensor, knn_k: int = 10) -> np.ndarray:
    """Similarity score per query patch.

    "single"/"multi": cosine similarity to the state's prototype(s), max'd over multiple.
    "fgbg": fg cosine similarity minus bg cosine similarity (both mean-collapsed).
    "fgbg_multi": max fg-row cosine similarity minus the (single) bg cosine similarity.
    "knn_fgbg": patch-gallery contrastive kNN (see knn_fgbg_score).
    """
    if state.kind == "knn_fgbg":
        assert state.fg_bank is not None and state.bg_bank is not None
        return knn_fgbg_score(query_tokens, state.fg_bank, state.bg_bank, knn_k)
    assert state.payload is not None
    sim = query_tokens @ state.payload.T  # (N, K)
    if state.kind == "single":
        return sim.squeeze(-1).cpu().float().numpy()
    if state.kind == "fgbg":
        return (sim[:, 0] - sim[:, 1]).cpu().float().numpy()
    if state.kind == "fgbg_multi":
        fg_max = sim[:, :-1].max(dim=-1).values
        return (fg_max - sim[:, -1]).cpu().float().numpy()
    return sim.max(dim=-1).values.cpu().float().numpy()  # "multi"


# ---------------------------------------------------------------------------
# Threshold + clustering helpers
# ---------------------------------------------------------------------------


def dbscan_clusters(
    xs: np.ndarray,
    ys: np.ndarray,
    grid_h: int,
    grid_w: int,
    raw: np.ndarray,
    scoring_cfg: ScoringConfig,
    min_cs: int | None = None,
) -> list[dict]:
    """Thin wrapper around ``dbscan_clusters_from_mask`` for predicted (noisy) clusters."""
    eps = patch_radius_to_eps(scoring_cfg.pred_dbscan_eps_patches)
    return _shared_dbscan_clusters(
        xs, ys, grid_h, grid_w, raw, eps, scoring_cfg.pred_dbscan_min_samples, min_cs
    )


# ---------------------------------------------------------------------------
# Cluster mean-patch reject
# ---------------------------------------------------------------------------


def tune_cluster_reject_threshold(
    ref_crops: list[ClusterCrop],
    state: MethodState,
    mean_patch_prototype: torch.Tensor,
    patch_thr: float,
    min_cs: int,
    iou_thr: float,
    scoring_cfg: ScoringConfig,
) -> tuple[float, list[dict]]:
    """Self-supervised mean-patch cosine-similarity cutoff — see the original
    ``multiscale_crop_ablation.py::tune_cluster_reject_threshold`` for full rationale.
    """
    eps = patch_radius_to_eps(scoring_cfg.pred_dbscan_eps_patches)
    return _shared_tune_cluster_reject_threshold(
        ref_crops,
        lambda tokens: score_method(state, tokens, knn_k=scoring_cfg.knn_fgbg_num_neighbours),
        mean_patch_prototype,
        patch_thr,
        min_cs,
        iou_thr,
        eps,
        scoring_cfg.pred_dbscan_min_samples,
        scoring_cfg.gt_dbscan_eps,
        scoring_cfg.gt_dbscan_min_samples,
        scoring_cfg.cluster_reject_margin_fraction,
        scoring_cfg.min_points_floor,
    )


# ---------------------------------------------------------------------------
# ROI blob helpers — unsupervised (no-GT) crop discovery for two-stage / cross-scale
# ---------------------------------------------------------------------------


def roi_binary_mask(raw: np.ndarray, scoring_cfg: ScoringConfig) -> np.ndarray:
    return _shared_roi_binary_mask(
        raw,
        scoring_cfg.roi_binarize_method,
        scoring_cfg.roi_single_threshold,
        percentile=scoring_cfg.roi_percentile,
    )


def find_roi_blobs(
    raw: np.ndarray,
    query_img: Image.Image,
    encoder: DinoEncoder,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    patch_size: int,
) -> tuple[list[dict], np.ndarray]:
    """Unsupervised ROI blobs from *raw*, cropped at native resolution and re-encoded.

    Returns ``(crops, roi_mask)``; each crop dict carries ``mask`` (patch-grid, for
    building two-stage predictions), ``patch_bbox``, ``px_bbox``, ``crop_tokens``.
    """
    native_w, native_h = query_img.size
    scale_x = native_w / crop_cfg.img_size
    scale_y = native_h / crop_cfg.img_size

    roi_mask = roi_binary_mask(raw, scoring_cfg)
    if scoring_cfg.roi_morph_close_iterations > 0:
        roi_mask = ndimage.binary_closing(
            roi_mask,
            structure=ndimage.generate_binary_structure(2, 2),
            iterations=scoring_cfg.roi_morph_close_iterations,
        )
    if scoring_cfg.roi_crop_mode == "per_blob":
        blobs = connected_component_blobs(roi_mask)
    elif scoring_cfg.roi_crop_mode == "single_bbox":
        blobs = [{"mask": roi_mask}] if roi_mask.any() else []
    else:
        raise ValueError(f"Unknown roi_crop_mode: {scoring_cfg.roi_crop_mode!r}")

    floor_side_px = crop_cfg.img_size / scoring_cfg.max_upscale_factor
    pending: list[dict] = []
    for blob in blobs:
        y0, y1, x0, x1 = blob_patch_bbox(blob["mask"])
        raw_px0, raw_py0, raw_px1, raw_py1 = patch_bbox_to_native_px(
            y0, y1, x0, x1, patch_size, scale_x, scale_y
        )
        px0, py0, px1, py1 = pad_and_floor_crop_box(
            raw_px0,
            raw_py0,
            raw_px1,
            raw_py1,
            scoring_cfg.blob_crop_padding_fraction,
            floor_side_px,
            floor_side_px,
            native_w,
            native_h,
        )
        if px1 - px0 < 2 or py1 - py0 < 2:
            continue
        pending.append(
            {
                "mask": blob["mask"],
                "patch_bbox": (y0, y1, x0, x1),
                "px_bbox": (px0, py0, px1, py1),
                "crop_img": query_img.crop((px0, py0, px1, py1)),
            }
        )

    if not pending:
        return [], roi_mask

    tokens_batch = extract_patch_tokens_batch(
        encoder, [p["crop_img"] for p in pending], crop_cfg.layer_idx, crop_cfg.debias
    )
    crops: list[dict] = [
        {
            k: v
            for k, v in entry.items()
            if k != "crop_img"  # not cached — visualisation re-crops from the query image
        }
        | {"crop_tokens": crop_tokens, "c_h": c_h, "c_w": c_w}
        for entry, (crop_tokens, c_h, c_w) in zip(pending, tokens_batch)
    ]
    return crops, roi_mask


def blob_crop_gt_mask(
    blob: dict, q_pixel_mask: np.ndarray | None, crop_cfg: CropConfig
) -> np.ndarray:
    return _shared_blob_crop_gt_mask(
        blob, q_pixel_mask, crop_cfg.img_size, crop_cfg.mask_patch_threshold
    )


def cross_score_blobs(
    blobs: list[dict],
    scale_states: dict[str, MethodState],
    q_pixel_mask: np.ndarray | None,
    per_method: dict[str, dict],
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
) -> list[dict]:
    """Score every blob crop with every candidate *scale_states* prototype.

    Binarises each crop's score map with *that prototype's own* tuned threshold and
    measures IoU against the crop's own ground-truth patch mask.
    """
    return _shared_cross_score_blobs(
        blobs,
        scale_states,
        q_pixel_mask,
        per_method,
        score_fn_for=lambda state: (
            lambda tokens: score_method(state, tokens, knn_k=scoring_cfg.knn_fgbg_num_neighbours)
        ),
        img_size=crop_cfg.img_size,
        mask_patch_threshold=crop_cfg.mask_patch_threshold,
    )


def two_stage_predicted_clusters(
    blobs: list[dict],
    state: MethodState,
    threshold: float,
    mean_patch_prototype: torch.Tensor,
    cluster_reject_thr: float,
    min_cs: int,
    q_h: int,
    q_w: int,
    patch_size: int,
    scale_x: float,
    scale_y: float,
    scoring_cfg: ScoringConfig,
    q_pixel_mask: np.ndarray | None = None,
    crop_cfg: CropConfig | None = None,
    collect_diagnostics: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Predicted clusters for the two-stage pipeline's P/R/F1 (see the original
    ``multiscale_crop_ablation.py::two_stage_predicted_clusters`` docstring).

    Returns ``(merged_predicted_clusters, blob_diagnostics)``. ``merged_predicted_clusters``
    (used for P/R/F1 via ``match_and_score``) is computed exactly as before, independent of
    *collect_diagnostics*. When *collect_diagnostics* is True (requires *q_pixel_mask* and
    *crop_cfg*), ``blob_diagnostics`` gets one entry per blob — its own raw score map,
    threshold, every DBSCAN cluster found in it (including rejected ones, unlike the merged
    predictions which drop rejected clusters), and its own GT patch mask — everything
    ``visualize_results.py`` needs to render a per-blob heatmap/binary/histogram/cluster
    breakdown (mirrors the original notebook's ``plot_crop_breakdown``) without needing the
    encoder or blob tokens again. Otherwise ``blob_diagnostics`` is ``[]``.
    """
    eps = patch_radius_to_eps(scoring_cfg.pred_dbscan_eps_patches)
    return _shared_two_stage_predicted_clusters(
        blobs,
        lambda tokens: score_method(state, tokens, knn_k=scoring_cfg.knn_fgbg_num_neighbours),
        threshold,
        mean_patch_prototype,
        cluster_reject_thr,
        min_cs,
        q_h,
        q_w,
        patch_size,
        scale_x,
        scale_y,
        eps,
        scoring_cfg.pred_dbscan_min_samples,
        scoring_cfg.min_points_floor,
        q_pixel_mask=q_pixel_mask,
        img_size=crop_cfg.img_size if crop_cfg is not None else None,
        mask_patch_threshold=crop_cfg.mask_patch_threshold if crop_cfg is not None else None,
        collect_diagnostics=collect_diagnostics,
    )
