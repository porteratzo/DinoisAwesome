"""Foreground/background gallery cleaning (Phase 2): mixed-patch rejection ("step1") and an
independent-appearance attention check ("step2"), ported from ``experiments/fundamental/
noisy_fgbg_cleaning.py`` (see that file's module docstring for the full ablation and its
rationale) into a form that plugs into ``engine.build_all_scale_prototypes``'s output, gated
by ``ScoringConfig.fg_clean_stage``.

Only mid/close scales are cleaned — "global"'s foreground gallery already spans the whole
image, so boundary patches are a tiny fraction of its fg pool; the noise-cleaning problem
``noisy_fgbg_cleaning.py`` targets is specific to the tight mid/close crops. ``step3``
(HDBSCAN + kNN consensus, pooled across every part type sharing an instance-type group) isn't
ported here — it needs cross-pair pooling that doesn't fit this pipeline's per-pair cache
model; it's deferred pending a separate design pass.

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
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from common import CropConfig, ScoringConfig
from engine import ClusterCrop, ScalePrototype
from scipy import ndimage

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


def apply_fg_cleaning(
    scale_protos: dict[str, ScalePrototype], crop_cfg: CropConfig, scoring_cfg: ScoringConfig
) -> dict[str, ScalePrototype]:
    """Rebuild each mid/close scale's fg/bg gallery selection per
    ``scoring_cfg.fg_clean_stage``. Identity (returns *scale_protos* unchanged, no cost) when
    the stage is "raw" — the default. "global" is never cleaned (see this module's
    docstring).

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

    new_protos: dict[str, ScalePrototype] = dict(scale_protos)
    for scale in ("mid", "close"):
        proto = scale_protos.get(scale)
        if proto is None or proto.cluster_crops is None:
            continue
        new_clusters = []
        for c in proto.cluster_crops:
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
