"""Foreground/background gallery cleaning (Phase 2): mixed-patch rejection ("step1"), an
independent-appearance attention check ("step2"), and cross-pair HDBSCAN + kNN-consensus
voting ("step3"), ported from ``experiments/fundamental/noisy_fgbg_cleaning.py`` (see that
file's module docstring for the full ablation and its rationale) into a form that plugs into
``engine.build_all_scale_prototypes``'s output, gated by ``ScoringConfig.fg_clean_stage``.
"step3" pools every mid/close cluster's raw foreground tokens across every ``PairKey`` sharing
an instance-type group (e.g. every part type annotated with "donut foam"), runs one HDBSCAN +
kNN-consensus pass over that pool (see :func:`hdbscan_knn_consensus_keep`), and caches the
resulting per-cluster keep-masks under ``group_cache_path`` so every pair in the group reuses
the same pass instead of recomputing it once per pair.

Only mid/close scales are cleaned — "global"'s foreground gallery already spans the whole
image, so boundary patches are a tiny fraction of its fg pool; the noise-cleaning problem
``noisy_fgbg_cleaning.py`` targets is specific to the tight mid/close crops. This applies to
"step3" too: unlike the original (which pooled global + mid + close together), this port pools
only mid/close, for the same reason.

``ClusterCrop.patch_mask``/``exclude_patch_mask`` (the true GT extent, used elsewhere for
visualisation and GT diagnostics) are never modified — cleaning only ever produces
``fg_select_mask``/``bg_select_mask``, a separate, gallery-only selection that
``engine._masked_mean`` (baked into the crop cache's "raw" prototype) and
``engine.pool_scale_patches`` prefer when present. Because this only re-filters already-cached
crop tokens (no new encoder pass), it's a ``ScoringConfig`` knob, applied once per
``run_experiments.py`` pair via :func:`apply_fg_cleaning`, not baked into the crop cache.
"""

from __future__ import annotations

import dataclasses
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from common import CropConfig, PairKey, ScoringConfig, all_pairs, group_cache_path
from engine import ClusterCrop, ScalePrototype
from scipy import ndimage
from sklearn.cluster import HDBSCAN
from tqdm import tqdm

from dinoisawesome import DinoEncoder

# Self-sufficient rather than relying on import order elsewhere having already patched
# sys.path (see bg_enrichment.py's identical comment) — cheap insurance either way, since
# `from engine import ...` above already does this as a side effect today.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _shared.mask_geometry import patch_fg_fraction  # noqa: E402

log = logging.getLogger(__name__)


def _keep_top_fraction_by_similarity(
    fg_tokens: torch.Tensor, reference: torch.Tensor, keep_fraction: float
) -> torch.Tensor:
    """Boolean keep-mask over *fg_tokens* (Nfg, C): keeps the *keep_fraction* most similar
    (cosine) to *reference* ((1, C) or (C,)), dropping the least-similar tail as suspected
    boundary/occlusion leakage. Keeps everything when Nfg < 4 (too small for a percentile cut
    to be meaningful).
    """
    n = fg_tokens.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.bool)
    if n < 4:
        return torch.ones(n, dtype=torch.bool)
    sims = (fg_tokens @ reference.reshape(1, -1).T).squeeze(-1)
    cutoff = torch.quantile(sims.float(), 1.0 - keep_fraction)
    return sims >= cutoff


def _center_prototype(
    c: ClusterCrop, crop_cfg: CropConfig, scoring_cfg: ScoringConfig
) -> torch.Tensor | None:
    """Masked-mean prototype over only the instance mask's innermost "core" pixels — the ones
    farthest from the mask boundary by Euclidean distance transform, above
    ``scoring_cfg.fg_clean_center_core_percentile`` of the in-mask distance distribution — a
    second, independent appearance reference for "step2_center". None if the mask/core is too
    degenerate to project onto any patch.
    """
    if c.own_mask_px is None or not c.own_mask_px.any():
        return None
    dist = ndimage.distance_transform_edt(c.own_mask_px)
    cutoff = np.percentile(dist[c.own_mask_px], scoring_cfg.fg_clean_center_core_percentile)
    core_px = (dist >= cutoff) & c.own_mask_px
    # mask_patch_threshold, not fg_clean_high: core_px is already the innermost slice of the
    # mask (by distance-from-edge) — requiring a patch to *also* clear the tighter step1 bar
    # left this empty for almost every crop in practice (see noisy_fgbg_cleaning.py).
    core_patch = (
        patch_fg_fraction(core_px, c.grid_h, c.grid_w, crop_cfg.img_size)
        >= crop_cfg.mask_patch_threshold
    )
    flat = torch.from_numpy(core_patch.reshape(-1)).to(c.tokens.device)
    if int(flat.sum().item()) == 0:
        return None
    return F.normalize(c.tokens[flat].mean(dim=0, keepdim=True), p=2, dim=-1)


def _clean_cluster_masks(
    c: ClusterCrop,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    close_cls: torch.Tensor | None,
) -> tuple[np.ndarray, np.ndarray]:
    """(fg_select, bg_select) boolean grids for one cluster crop, per
    ``scoring_cfg.fg_clean_stage``.

    "step1": tightened high/low thresholds on each side, ambiguous boundary band dropped from
    both. "step2_cls"/"step2_center": start from the *raw* (mask_patch_threshold) fg
    selection and further drop the least-similar tail vs. an independent appearance reference
    — bg is left as *raw* (untouched), since step2 is a foreground-only cross-check (see this
    module's docstring). Falls back to raw fg/bg (logged) whenever a stage would otherwise
    leave a side empty.
    """
    stage = scoring_cfg.fg_clean_stage
    raw_fg, raw_bg = c.patch_mask, ~c.exclude_patch_mask

    if stage == "step1":
        assert c.own_frac is not None and c.excl_frac is not None
        fg = c.own_frac >= scoring_cfg.fg_clean_high
        bg = c.excl_frac <= scoring_cfg.fg_clean_low
        if not fg.any():
            log.warning(
                "cluster=%d: step1 spatial filter left zero fg patches — falling back to raw fg",
                c.cluster_idx,
            )
            fg = raw_fg
        if not bg.any():
            log.warning(
                "cluster=%d: step1 spatial filter left zero bg patches — falling back to raw bg",
                c.cluster_idx,
            )
            bg = raw_bg
        return fg, bg

    if stage in ("step2_cls", "step2_center"):
        reference = (
            close_cls if stage == "step2_cls" else _center_prototype(c, crop_cfg, scoring_cfg)
        )
        if reference is None or not raw_fg.any():
            return raw_fg, raw_bg
        flat = raw_fg.reshape(-1)
        idx = np.flatnonzero(flat)
        fg_tokens = c.tokens[torch.from_numpy(flat).to(c.tokens.device)]
        keep = _keep_top_fraction_by_similarity(
            fg_tokens, reference, scoring_cfg.fg_clean_attention_keep_fraction
        )
        fg = np.zeros_like(flat)
        fg[idx[keep.cpu().numpy()]] = True
        fg = fg.reshape(raw_fg.shape)
        if not fg.any():
            log.warning(
                "cluster=%d stage=%s: attention check left zero fg patches — falling back to "
                "raw fg",
                c.cluster_idx,
                stage,
            )
            fg = raw_fg
        return fg, raw_bg

    raise ValueError(f"Unknown fg_clean_stage: {stage!r}")


def _masked_mean_select(tokens: torch.Tensor, select: np.ndarray) -> torch.Tensor:
    flat = torch.from_numpy(select.reshape(-1)).to(tokens.device)
    sel = tokens[flat]
    if sel.shape[0] == 0:
        sel = tokens
    return F.normalize(sel.mean(dim=0, keepdim=True), p=2, dim=-1)


# ---------------------------------------------------------------------------
# Step 3 — HDBSCAN + kNN consensus voting, pooled across every pair sharing an
# instance-type group
# ---------------------------------------------------------------------------


def hdbscan_knn_consensus_keep(
    tokens: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    knn_k: int,
    min_agreement: float,
) -> tuple[np.ndarray, np.ndarray]:
    """HDBSCAN-cluster *tokens* (N, C), L2-normalised, then keep a point only if (a) HDBSCAN
    placed it in a real cluster (label != -1) and (b) a majority (>= min_agreement) of its
    knn_k nearest neighbours in this same set share that label — HDBSCAN's own noise flag
    catches sparse outliers, the kNN vote catches points HDBSCAN happened to assign to a
    cluster despite sitting on that cluster's own ragged boundary. Tokens are L2-normalised,
    so plain Euclidean distance (HDBSCAN's default metric) is already a monotonic transform
    of cosine similarity, hence no custom metric is needed for either step.

    Returns (keep, hdbscan_labels). Too few points to cluster meaningfully (fewer than
    max(min_cluster_size, knn_k + 1)) short-circuits to "keep everything" — there isn't
    enough data for HDBSCAN's density estimate to mean anything.
    """
    n = tokens.shape[0]
    if n < max(min_cluster_size, knn_k + 1):
        return np.ones(n, dtype=bool), np.zeros(n, dtype=int)
    labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(tokens)
    sims = tokens @ tokens.T
    np.fill_diagonal(sims, -np.inf)
    k = min(knn_k, n - 1)
    knn_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    keep = np.zeros(n, dtype=bool)
    for i in range(n):
        if labels[i] == -1:
            continue
        agreement = float(np.mean(labels[knn_idx[i]] == labels[i]))
        keep[i] = agreement >= min_agreement
    return keep, labels


def _build_group_step3_masks(
    instance_type: str,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    encoder: DinoEncoder,
    force: bool,
) -> dict[tuple[str, str, int], np.ndarray]:
    """Pools every mid/close ``ClusterCrop``'s *raw* foreground tokens across every ``PairKey``
    sharing *instance_type* (abc3's instance-type groups aren't per-part-type — see
    ``common.all_pairs``/``dinoisawesome.abc3.INSTANCE_TYPE_GROUPS``), runs one
    :func:`hdbscan_knn_consensus_keep` pass over the pool, and returns each surviving/rejected
    cluster's own ``fg_select_mask`` keyed by ``(pair.slug, scale, cluster_idx)``.

    Cached under ``group_cache_path`` (keyed by instance_type/crop_cfg/scoring_cfg) so every
    pair in the group reuses the same pass rather than recomputing it once per pair.
    """
    cache_path = group_cache_path(instance_type, crop_cfg, scoring_cfg)
    if cache_path.exists() and not force:
        cache_path.touch()  # mark as just-used for scripts/prune_cache.py
        return pickle.loads(cache_path.read_bytes())

    # Local import to avoid a circular import: cleaning.py is imported by run_experiments.py
    # (`from cleaning import apply_fg_cleaning`), so a top-level `from run_experiments import
    # ...` here would fail at module load time. By the time this function is actually called,
    # run_experiments is already fully imported, so the import resolves fine.
    from run_experiments import _get_or_build_crop_cache, _load_pair_images_and_masks

    group_pairs = [p for p in all_pairs() if p.instance_type == instance_type]

    chunks: list[torch.Tensor] = []
    # (pair_slug, scale, cluster_idx, idx, start, end) — idx is the flat-index array the
    # segment's tokens were gathered from (np.flatnonzero(c.patch_mask.reshape(-1))), needed to
    # scatter the kept subset back onto that cluster's own (grid_h, grid_w) grid.
    segments: list[tuple[str, str, int, np.ndarray, int, int]] = []
    shapes: dict[tuple[str, str, int], tuple[int, int]] = {}
    pooled_pair_slugs: set[str] = set()
    offset = 0

    for pair in tqdm(group_pairs, desc=f"step3 pool[{instance_type}]"):
        ref_img, query_img, ref_instance_masks, _ref_pixel_mask, _q_pixel_mask, _q_inst_masks = (
            _load_pair_images_and_masks(pair)
        )
        if not ref_instance_masks:
            log.warning("[%s] no exemplar instances — skipping in step3 group pool", pair.slug)
            continue
        # Reuses the same per-pair crop cache run_pair itself builds/reads (keyed only on
        # pair/crop_cfg) — pooling a pair whose own run already populated it costs nothing extra.
        crop_cache = _get_or_build_crop_cache(
            pair, crop_cfg, encoder, ref_img, query_img, ref_instance_masks, force
        )
        scale_protos: dict[str, ScalePrototype] = crop_cache["scale_protos"]
        for scale in ("mid", "close"):
            proto = scale_protos.get(scale)
            if proto is None or proto.cluster_crops is None:
                continue
            for c in proto.cluster_crops:
                if c is None:
                    continue
                shapes[(pair.slug, scale, c.cluster_idx)] = (c.grid_h, c.grid_w)
                idx = np.flatnonzero(c.patch_mask.reshape(-1))
                if idx.size == 0:
                    continue
                fg_tokens = c.tokens[torch.from_numpy(idx).to(c.tokens.device)]
                chunks.append(fg_tokens)
                segments.append((pair.slug, scale, c.cluster_idx, idx, offset, offset + idx.size))
                pooled_pair_slugs.add(pair.slug)
                offset += idx.size

    fg_masks: dict[tuple[str, str, int], np.ndarray] = {}
    if not chunks:
        cache_path.write_bytes(pickle.dumps(fg_masks))
        log.info(
            "[group=%s] step3 pool: 0 pairs, 0 clusters, 0 tokens -> cache written: %s",
            instance_type,
            cache_path,
        )
        return fg_masks

    pooled = torch.cat(chunks, dim=0)
    keep, _labels = hdbscan_knn_consensus_keep(
        pooled.cpu().numpy(),
        scoring_cfg.fg_clean_step3_hdbscan_min_cluster_size,
        scoring_cfg.fg_clean_step3_hdbscan_min_samples,
        scoring_cfg.fg_clean_step3_knn_k,
        scoring_cfg.fg_clean_step3_min_agreement,
    )

    for pair_slug, scale, cluster_idx, idx, start, end in segments:
        key = (pair_slug, scale, cluster_idx)
        grid_h, grid_w = shapes[key]
        local_keep = keep[start:end]
        fg_select_mask = np.zeros((grid_h, grid_w), dtype=bool)
        fg_select_mask.flat[idx[local_keep]] = True
        if not fg_select_mask.any():
            log.warning(
                "%s scale=%s cluster=%d stage=step3: HDBSCAN + kNN consensus rejected every "
                "patch — falling back to raw fg",
                pair_slug,
                scale,
                cluster_idx,
            )
            fg_select_mask = np.zeros((grid_h, grid_w), dtype=bool)
            fg_select_mask.flat[idx] = True
        fg_masks[key] = fg_select_mask

    cache_path.write_bytes(pickle.dumps(fg_masks))
    log.info(
        "[group=%s] step3 pool: %d pairs, %d clusters, %d tokens -> cache written: %s",
        instance_type,
        len(pooled_pair_slugs),
        len(segments),
        offset,
        cache_path,
    )
    return fg_masks


def apply_fg_cleaning(
    pair: PairKey,
    scale_protos: dict[str, ScalePrototype],
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    encoder: DinoEncoder,
    force: bool = False,
) -> dict[str, ScalePrototype]:
    """Rebuild each mid/close scale's fg/bg gallery selection per
    ``scoring_cfg.fg_clean_stage``. Identity (returns *scale_protos* unchanged, no cost) when
    the stage is "raw" — the default. "global" is never cleaned (see this module's
    docstring).

    ``pair``/``encoder``/``force`` are only used by the "step3" branch (to pool raw fg tokens
    across every pair sharing ``pair.instance_type`` — see :func:`_build_group_step3_masks`);
    every other stage ignores them, so a plain run's cost is unaffected.

    Recomputes ``mean_prototype``/``bg_prototype`` from the cleaned selection, folding in any
    ``extra_bg_crops`` (Phase 1 background enrichment) unchanged so the two techniques compose
    — enabling both flags together cleans the fg/bg pools *and* keeps the enriched bg crops.
    """
    if scoring_cfg.fg_clean_stage == "raw":
        return scale_protos

    close_cls_by_cluster: dict[int, torch.Tensor] = {}
    close_proto = scale_protos.get("close")
    if close_proto is not None and close_proto.cluster_crops is not None:
        for c in close_proto.cluster_crops:
            if c.cls is not None:
                close_cls_by_cluster[c.cluster_idx] = c.cls

    group_masks: dict[tuple[str, str, int], np.ndarray] | None = None
    if scoring_cfg.fg_clean_stage == "step3":
        group_masks = _build_group_step3_masks(
            pair.instance_type, crop_cfg, scoring_cfg, encoder, force
        )

    new_protos: dict[str, ScalePrototype] = dict(scale_protos)
    for scale in ("mid", "close"):
        proto = scale_protos.get(scale)
        if proto is None or proto.cluster_crops is None:
            continue
        new_clusters = []
        for c in proto.cluster_crops:
            if group_masks is not None:
                key = (pair.slug, scale, c.cluster_idx)
                fg_sel = group_masks.get(key)
                if fg_sel is None:
                    # Shouldn't happen — every mid/close cluster crop this pair builds should
                    # have been pooled by _build_group_step3_masks too. Cheap insurance in case
                    # the group cache was built under a different pair count (e.g. --limit-pairs
                    # / --part-types narrowed one run but not the other).
                    log.warning(
                        "%s scale=%s cluster=%d stage=step3: missing from group pool — "
                        "falling back to raw fg",
                        pair.slug,
                        scale,
                        c.cluster_idx,
                    )
                    fg_sel = c.patch_mask
                bg_sel = ~c.exclude_patch_mask
            else:
                fg_sel, bg_sel = _clean_cluster_masks(
                    c, crop_cfg, scoring_cfg, close_cls_by_cluster.get(c.cluster_idx)
                )
            new_clusters.append(
                dataclasses.replace(c, fg_select_mask=fg_sel, bg_select_mask=bg_sel)
            )
        fg_means = [_masked_mean_select(c.tokens, c.fg_select_mask) for c in new_clusters]
        bg_means = [_masked_mean_select(c.tokens, c.bg_select_mask) for c in new_clusters]
        bg_means += [e.mean_token for e in (proto.extra_bg_crops or [])]
        avg = F.normalize(torch.cat(fg_means, dim=0).mean(dim=0, keepdim=True), p=2, dim=-1)
        bg_avg = F.normalize(torch.cat(bg_means, dim=0).mean(dim=0, keepdim=True), p=2, dim=-1)
        new_protos[scale] = dataclasses.replace(
            proto, cluster_crops=new_clusters, mean_prototype=avg, bg_prototype=bg_avg
        )
    return new_protos
