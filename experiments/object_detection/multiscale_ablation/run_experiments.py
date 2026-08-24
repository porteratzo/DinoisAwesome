"""Stage 1: encode reference/query images, build every requested method's prototype(s),
score + threshold + DBSCAN-cluster + GT-match, and run the two-stage ROI-blob rescoring
and cross-scale similarity experiments — the encoding/scoring half of
``multiscale_crop_ablation.py``'s pipeline, split out so it can be cached and so new
methods (see ``methods.py``) never require touching this file.

Two-tier cache (see ``common.py``'s module docstring):
    cache/crops/<crop_hash>/<pair>/tokens.pt
        DINO tokens for every scale's exemplar crops + the query image. Shared by
        every method that uses the same crop geometry/encoder settings — adding a
        method (even one that scores differently, like k-means) never re-triggers an
        encoder pass here. torch.save'd; internal to this script only.
    cache/blobs/<crop_hash>__<scoring_hash>/<pair>/<roi_source>.pt
        Two-stage ROI blob crop tokens, keyed by which method's raw map located them.
        torch.save'd; internal to this script only.
    cache/methods/<crop_hash>__<scoring_hash>/<pair>/
        pair_meta.pkl           geometry + GT clusters, shared across methods
        <method>.pkl            raw score map, clusters, metrics
        two-stage(<method>).pkl same schema, two-stage variant
        blobs__<roi_source>.pkl blob boxes/masks (no tokens) for detailed viz
        cross_scale.pkl         roi_source x score_scale IoU rows
    These are pure numpy/python (pickle) — ``visualize_results.py`` reads only these,
    never cache/crops or cache/blobs, so it never needs torch/the encoder.

Usage:
    python run_experiments.py                                   # every pair, every method
    python run_experiments.py --part-types RHa --limit-pairs 1   # smoke test
    python run_experiments.py --methods global mid close mid-kmeans3
    python run_experiments.py --include-kmeans                   # also run the k-means families
    python run_experiments.py --force                            # ignore existing cache

k-means methods (``<scale>-kmeans<k>``, ``fg-bg-kmeans<k>(...)``) consistently underperform
the mean/proto/knn families (see ``methods.py``), so they're skipped by default; pass
``--include-kmeans`` to run them too, or name one explicitly via ``--methods``.
"""

# Logging — must be before torch import
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("run_experiments")

import argparse
import os
import pickle
import time
import traceback

import numpy as np
import torch
from common import (
    DATA_DIR,
    DEFAULT_CROP_CONFIG,
    DEFAULT_SCORING_CONFIG,
    CropConfig,
    PairKey,
    ScoringConfig,
    all_pairs,
    blob_cache_path,
    crop_cache_path,
    instance_classes_for,
    method_cache_dir,
    method_cache_path,
)
from engine import (
    ScalePrototype,
    annotate_cluster_rejection,
    build_all_scale_prototypes,
    cross_score_blobs,
    dbscan_clusters,
    dbscan_clusters_from_mask,
    find_roi_blobs,
    gt_instance_patch_sizes,
    iou_tuned_threshold,
    match_and_score,
    min_cluster_size_bound,
    pixel_mask_to_patch_mask,
    score_method,
    tune_cluster_reject_threshold,
    two_stage_predicted_clusters,
)
from methods import all_method_names, build_method_states
from PIL import Image

from dinoisawesome import DinoEncoder
from dinoisawesome.abc3 import load_instance_pixel_mask, load_instance_pixel_masks
from dinoisawesome.instance_detection import extract_patch_tokens

DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part-types", nargs="+", default=None, help="Restrict to these part types"
    )
    parser.add_argument(
        "--instance-types", nargs="+", default=None, help="Restrict to these instance-type groups"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        # include_kmeans=True here just so the (default-off) k-means names remain valid
        # --methods choices; whether they're actually built by default is controlled by
        # --include-kmeans / build_method_states, not by this list.
        choices=all_method_names(
            list(DEFAULT_CROP_CONFIG.scales), DEFAULT_SCORING_CONFIG.kmeans_ks, include_kmeans=True
        ),
        help="Restrict to these methods (default: every registered method except k-means)",
    )
    parser.add_argument(
        "--limit-pairs", type=int, default=None, help="Cap number of pairs (smoke test)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Ignore existing cache, recompute everything"
    )
    parser.add_argument(
        "--include-kmeans",
        action="store_true",
        help="Also build the k-means method families (skipped by default — see methods.py)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Tier A/B — crop cache (expensive: encoder forward passes)
# ---------------------------------------------------------------------------


def _load_pair_images_and_masks(
    pair: PairKey,
) -> tuple[Image.Image, Image.Image, list, np.ndarray, np.ndarray | None]:
    exemplar_class = instance_classes_for(pair.instance_type)
    ref_stem = f"{pair.part_type}_{pair.ref_number}"
    query_stem = f"{pair.part_type}_{pair.query_number}"

    ref_instance_masks = load_instance_pixel_masks(
        DATA_DIR / "annotations" / ref_stem, exemplar_class
    )
    ref_pixel_mask = np.stack(ref_instance_masks).any(axis=0) if ref_instance_masks else None
    ref_img = Image.open(DATA_DIR / f"{ref_stem}.jpg").convert("RGB")
    query_img = Image.open(DATA_DIR / f"{query_stem}.jpg").convert("RGB")
    q_pixel_mask = load_instance_pixel_mask(DATA_DIR / "annotations" / query_stem, exemplar_class)
    return ref_img, query_img, ref_instance_masks, ref_pixel_mask, q_pixel_mask


def _get_or_build_crop_cache(
    pair: PairKey,
    crop_cfg: CropConfig,
    encoder: DinoEncoder,
    ref_img: Image.Image,
    query_img: Image.Image,
    ref_instance_masks: list,
    force: bool,
) -> dict:
    cache_path = crop_cache_path(pair, crop_cfg)
    if cache_path.exists() and not force:
        log.info("[%s] crop cache hit: %s", pair.slug, cache_path)
        return torch.load(cache_path, weights_only=False)

    scale_protos, mean_patch_prototype = build_all_scale_prototypes(
        encoder, ref_img, ref_instance_masks, crop_cfg
    )
    q_tokens, q_h, q_w = extract_patch_tokens(
        encoder, query_img, crop_cfg.layer_idx, debias=crop_cfg.debias
    )
    payload = {
        "scale_protos": scale_protos,
        "mean_patch_prototype": mean_patch_prototype,
        "q_tokens": q_tokens,
        "q_h": q_h,
        "q_w": q_w,
    }
    torch.save(payload, cache_path)
    log.info("[%s] crop cache written: %s", pair.slug, cache_path)
    return payload


# ---------------------------------------------------------------------------
# Pair-level metadata (GT clusters, geometry) — cheap, no encoder involved
# ---------------------------------------------------------------------------


def _compute_pair_meta(
    pair: PairKey,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    scale_protos: dict[str, ScalePrototype],
    q_h: int,
    q_w: int,
    ref_pixel_mask: np.ndarray | None,
    q_pixel_mask: np.ndarray | None,
) -> dict:
    query_stem = f"{pair.part_type}_{pair.query_number}"
    gt_patch_mask = (
        pixel_mask_to_patch_mask(
            q_pixel_mask, q_h, q_w, crop_cfg.img_size, crop_cfg.mask_patch_threshold
        )
        if q_pixel_mask is not None
        else np.zeros((q_h, q_w), dtype=bool)
    )
    exemplar_class = instance_classes_for(pair.instance_type)
    gt_sizes = gt_instance_patch_sizes(query_stem, exemplar_class, q_h, q_w, crop_cfg)
    min_cs = min_cluster_size_bound(
        gt_sizes, scoring_cfg.cluster_size_margin, scoring_cfg.min_points_floor
    )
    gt_clusters = dbscan_clusters_from_mask(
        gt_patch_mask, scoring_cfg.gt_dbscan_eps, scoring_cfg.gt_dbscan_min_samples
    )
    log.info(
        "[%s] GT instances=%d sizes=%s -> min_cluster_size=%d -> GT-DBSCAN clusters=%d",
        pair.slug,
        len(gt_sizes),
        gt_sizes.tolist(),
        min_cs,
        len(gt_clusters),
    )
    return {
        "part_type": pair.part_type,
        "instance_type": pair.instance_type,
        "ref_number": pair.ref_number,
        "query_number": pair.query_number,
        "scale_boxes": {s: p.box for s, p in scale_protos.items()},
        "scale_patch_masks": {s: p.patch_mask for s, p in scale_protos.items()},
        "cluster_boxes": {
            s: [(c.cluster_idx, c.box, c.patch_mask) for c in p.cluster_crops]
            for s, p in scale_protos.items()
            if p.cluster_crops is not None
        },
        "q_h": q_h,
        "q_w": q_w,
        "q_pixel_mask": q_pixel_mask,
        "gt_patch_mask": gt_patch_mask,
        "gt_clusters": gt_clusters,
        "min_cs": min_cs,
    }


# ---------------------------------------------------------------------------
# Tier C — per-method scoring/clustering/matching (cheap given cached tokens)
# ---------------------------------------------------------------------------


def _run_method(
    state,
    q_tokens: torch.Tensor,
    q_h: int,
    q_w: int,
    ref_mid: ScalePrototype,
    ref_mid_gt_mask: np.ndarray,
    mean_patch_prototype: torch.Tensor,
    min_cs: int,
    gt_clusters: list[dict],
    scoring_cfg: ScoringConfig,
) -> dict:
    raw = score_method(state, q_tokens, knn_k=scoring_cfg.knn_fgbg_num_neighbours).reshape(q_h, q_w)
    ref_raw = score_method(
        state, ref_mid.tokens, knn_k=scoring_cfg.knn_fgbg_num_neighbours
    ).reshape(ref_mid.grid_h, ref_mid.grid_w)
    thr = iou_tuned_threshold(ref_raw, ref_mid_gt_mask, scoring_cfg.ref_threshold_steps)

    assert ref_mid.cluster_crops, "mid scale is never dropped by default crop configs"
    cluster_reject_thr, ref_tuning_clusters = tune_cluster_reject_threshold(
        ref_mid.cluster_crops,
        state,
        mean_patch_prototype,
        thr,
        min_cs,
        scoring_cfg.iou_match_threshold,
        scoring_cfg,
    )

    ys, xs = np.where(raw > thr)
    if len(xs) < max(scoring_cfg.min_points_floor, min_cs):
        pred_clusters: list[dict] = []
    else:
        pred_clusters = dbscan_clusters(xs, ys, q_h, q_w, raw, scoring_cfg, min_cs)
    annotate_cluster_rejection(pred_clusters, q_tokens, mean_patch_prototype, cluster_reject_thr)
    kept_clusters = [c for c in pred_clusters if not c["rejected"]]

    metrics = match_and_score(kept_clusters, gt_clusters, scoring_cfg.iou_match_threshold)
    return {
        "raw": raw.astype(np.float32),
        "threshold": thr,
        "cluster_reject_thr": cluster_reject_thr,
        "pred_clusters": pred_clusters,
        "ref_tuning_clusters": ref_tuning_clusters,
        "metrics": metrics,
    }


def _write_method_cache(
    pair: PairKey, crop_cfg: CropConfig, scoring_cfg: ScoringConfig, method_name: str, payload: dict
) -> None:
    # The on-disk filename sanitises "/" (e.g. "fg-bg-mean(mid/all)") for path-safety,
    # which is lossy to invert — callers that need the true method name (e.g.
    # visualize_results.py's summary aggregation) must read it back from here, never
    # reconstruct it from the filename stem.
    path = method_cache_path(pair, crop_cfg, scoring_cfg, method_name)
    path.write_bytes(pickle.dumps({"method": method_name, **payload}))


def _method_cache_exists(
    pair: PairKey, crop_cfg: CropConfig, scoring_cfg: ScoringConfig, method_name: str
) -> bool:
    return method_cache_path(pair, crop_cfg, scoring_cfg, method_name).exists()


# ---------------------------------------------------------------------------
# Two-stage ROI blobs (own encoder pass, cached per roi-source method)
# ---------------------------------------------------------------------------


def _get_or_build_blobs(
    pair: PairKey,
    roi_source: str,
    raw: np.ndarray,
    query_img: Image.Image,
    encoder: DinoEncoder,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    patch_size: int,
    force: bool,
) -> tuple[list[dict], np.ndarray]:
    cache_path = blob_cache_path(pair, crop_cfg, scoring_cfg, roi_source)
    if cache_path.exists() and not force:
        payload = torch.load(cache_path, weights_only=False)
        return payload["blobs"], payload["roi_mask"]

    blobs, roi_mask = find_roi_blobs(raw, query_img, encoder, crop_cfg, scoring_cfg, patch_size)
    torch.save({"blobs": blobs, "roi_mask": roi_mask}, cache_path)

    light_blobs = [{k: v for k, v in b.items() if k != "crop_tokens"} for b in blobs]
    light_path = method_cache_dir(pair, crop_cfg, scoring_cfg) / f"blobs__{roi_source}.pkl"
    light_path.write_bytes(pickle.dumps({"blobs": light_blobs, "roi_mask": roi_mask}))
    return blobs, roi_mask


# ---------------------------------------------------------------------------
# Per-pair driver
# ---------------------------------------------------------------------------


def run_pair(
    pair: PairKey,
    encoder: DinoEncoder,
    patch_size: int,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    method_names: list[str] | None,
    force: bool,
    include_kmeans: bool = False,
) -> None:
    ref_img, query_img, ref_instance_masks, ref_pixel_mask, q_pixel_mask = (
        _load_pair_images_and_masks(pair)
    )
    if not ref_instance_masks:
        log.warning("[%s] no exemplar instances — skipping", pair.slug)
        return

    crop_cache = _get_or_build_crop_cache(
        pair, crop_cfg, encoder, ref_img, query_img, ref_instance_masks, force
    )
    scale_protos: dict[str, ScalePrototype] = crop_cache["scale_protos"]
    mean_patch_prototype = crop_cache["mean_patch_prototype"]
    q_tokens, q_h, q_w = crop_cache["q_tokens"], crop_cache["q_h"], crop_cache["q_w"]

    pair_meta = _compute_pair_meta(
        pair, crop_cfg, scoring_cfg, scale_protos, q_h, q_w, ref_pixel_mask, q_pixel_mask
    )
    method_cache_dir(pair, crop_cfg, scoring_cfg).joinpath("pair_meta.pkl").write_bytes(
        pickle.dumps(pair_meta)
    )

    states = build_method_states(
        scale_protos, scoring_cfg, list(scale_protos.keys()), method_names, include_kmeans
    )
    if not states:
        log.warning("[%s] no methods buildable for this pair's available scales", pair.slug)
        return

    ref_mid = scale_protos["mid"]
    x0, y0, x1, y1 = ref_mid.box
    ref_mid_gt_mask = pixel_mask_to_patch_mask(
        ref_pixel_mask[y0:y1, x0:x1],
        ref_mid.grid_h,
        ref_mid.grid_w,
        crop_cfg.img_size,
        crop_cfg.mask_patch_threshold,
    )

    per_method: dict[str, dict] = {}
    for name, state in states.items():
        if not force and _method_cache_exists(pair, crop_cfg, scoring_cfg, name):
            with method_cache_path(pair, crop_cfg, scoring_cfg, name).open("rb") as f:
                per_method[name] = pickle.load(f)
            continue
        result = _run_method(
            state,
            q_tokens,
            q_h,
            q_w,
            ref_mid,
            ref_mid_gt_mask,
            mean_patch_prototype,
            pair_meta["min_cs"],
            pair_meta["gt_clusters"],
            scoring_cfg,
        )
        m = result["metrics"]
        log.info(
            "[%s/%s] thr=%.3f pred=%d P=%.2f R=%.2f F1=%.2f mIoU=%.2f",
            pair.slug,
            name,
            result["threshold"],
            len(result["pred_clusters"]),
            m["precision"],
            m["recall"],
            m["f1"],
            m["mean_iou"],
        )
        _write_method_cache(pair, crop_cfg, scoring_cfg, name, result)
        per_method[name] = result

    # Self-anchored methods (roi_source_method == own name) generate their own ROI
    # blobs; every other method reuses whichever self-anchored method its foreground
    # side is built from (see methods.py's roi_source_method assignment).
    self_anchored = [n for n, s in states.items() if s.roi_source_method == n]
    native_w, native_h = query_img.size
    scale_x, scale_y = native_w / crop_cfg.img_size, native_h / crop_cfg.img_size

    blobs_by_source: dict[str, list[dict]] = {}
    for name in self_anchored:
        blobs, _roi_mask = _get_or_build_blobs(
            pair,
            name,
            per_method[name]["raw"],
            query_img,
            encoder,
            crop_cfg,
            scoring_cfg,
            patch_size,
            force,
        )
        blobs_by_source[name] = blobs
        log.info("[%s] roi_source=%s -> %d blob(s)", pair.slug, name, len(blobs))

    cross_cache_path = method_cache_dir(pair, crop_cfg, scoring_cfg) / "cross_scale.pkl"
    if force or not cross_cache_path.exists():
        cross_rows: list[dict] = []
        anchor_states = {n: states[n] for n in self_anchored}
        for roi_source, blobs in blobs_by_source.items():
            scored = cross_score_blobs(
                blobs, anchor_states, q_pixel_mask, per_method, crop_cfg, scoring_cfg
            )
            for r in scored:
                cross_rows.append({"roi_source": roi_source, **r})
        cross_cache_path.write_bytes(pickle.dumps(cross_rows))

    for name, state in states.items():
        two_stage_name = f"two-stage({name})"
        if not force and _method_cache_exists(pair, crop_cfg, scoring_cfg, two_stage_name):
            continue
        roi_source = state.roi_source_method
        if roi_source is None or roi_source not in blobs_by_source:
            continue
        blobs = blobs_by_source[roi_source]
        method_info = per_method[name]
        pred_clusters, blob_diagnostics = two_stage_predicted_clusters(
            blobs,
            state,
            method_info["threshold"],
            mean_patch_prototype,
            method_info["cluster_reject_thr"],
            pair_meta["min_cs"],
            q_h,
            q_w,
            patch_size,
            scale_x,
            scale_y,
            scoring_cfg,
            q_pixel_mask=q_pixel_mask,
            crop_cfg=crop_cfg,
            collect_diagnostics=True,
        )
        metrics = match_and_score(
            pred_clusters, pair_meta["gt_clusters"], scoring_cfg.iou_match_threshold
        )
        log.info(
            "[%s/%s] two-stage(roi=%s) pred=%d P=%.2f R=%.2f F1=%.2f mIoU=%.2f",
            pair.slug,
            name,
            roi_source,
            len(pred_clusters),
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["mean_iou"],
        )
        _write_method_cache(
            pair,
            crop_cfg,
            scoring_cfg,
            two_stage_name,
            {
                "pred_clusters": pred_clusters,
                "metrics": metrics,
                "roi_source": roi_source,
                "blob_diagnostics": blob_diagnostics,
            },
        )


def main() -> None:
    args = _parse_args()
    crop_cfg = DEFAULT_CROP_CONFIG
    scoring_cfg = DEFAULT_SCORING_CONFIG

    encoder = DinoEncoder(
        version=crop_cfg.dino_version,
        size=crop_cfg.dino_size,
        img_size=crop_cfg.img_size,
        weights_dir=DINO_WEIGHTS_DIR,
        amp=True,
    )
    patch_size = encoder.patch_size
    log.info(
        "DINOv%s-%s | patch_size=%d | grid=%dx%d | crop_hash=%s | scoring_hash=%s",
        crop_cfg.dino_version[1],
        crop_cfg.dino_size,
        patch_size,
        encoder.grid_h,
        encoder.grid_w,
        crop_cfg.hash(),
        scoring_cfg.hash(),
    )

    pairs = all_pairs()
    if args.part_types:
        pairs = [p for p in pairs if p.part_type in args.part_types]
    if args.instance_types:
        pairs = [p for p in pairs if p.instance_type in args.instance_types]
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]
    if not pairs:
        raise RuntimeError("No pairs matched --part-types/--instance-types filters")

    log.info("Running %d pair(s): %s", len(pairs), [p.slug for p in pairs])

    t_start = time.time()
    failures: list[tuple[str, str]] = []
    for pair in pairs:
        try:
            run_pair(
                pair,
                encoder,
                patch_size,
                crop_cfg,
                scoring_cfg,
                args.methods,
                args.force,
                args.include_kmeans,
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch going, report at the end
            log.error("[%s] FAILED: %s", pair.slug, exc)
            traceback.print_exc()
            failures.append((pair.slug, str(exc)))

    elapsed = time.time() - t_start
    log.info("Done in %.1fs. %d failure(s).", elapsed, len(failures))
    for slug, msg in failures:
        log.error("  %s: %s", slug, msg)


if __name__ == "__main__":
    main()
