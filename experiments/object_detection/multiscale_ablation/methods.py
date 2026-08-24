"""Method registry: every way of turning a pair's :class:`~engine.ScalePrototype`\\ s
into a :class:`~engine.MethodState` that ``engine.score_method`` can score a query
with.

This is the extension point for "we will make new methods" — a new method is a new
``build_*_states`` function (reading tokens straight off ``scale_protos``/
``pool_scale_patches``, never touching the encoder) plus a registration in
``build_method_states``/``all_method_names``. Nothing in ``run_experiments.py`` or
``visualize_results.py`` needs to change to add one, as long as it doesn't need a
different crop (see ``common.CropConfig`` if it does).

Method families:
    - Mean, single/multi scale (``global``, ``mid``, ``close``, and combos of them) —
      the original ablation's baseline: one masked-mean prototype per scale.
    - fg-bg-mean — contrastive foreground-minus-background, single fg scale, both
      sides collapsed to a mean vector.
    - fg-bg-proto — contrastive foreground-minus-background using *multiple* fg
      datasources: one combo per multi-scale fg source (e.g. ``global+mid``,
      ``mid+close``, ``global+mid+close``), keeping each scale's own mean prototype as
      a separate row (not averaged into one) so scoring can take the best-matching
      scale per query patch. See ``FGBG_SOURCE_COMBOS`` and
      ``build_fgbg_multiproto_states``.
    - fg-bg-knn — contrastive, patch-gallery kNN instead of mean vectors on either
      side; the fg gallery can likewise be pooled from multiple scales (same
      ``FGBG_SOURCE_COMBOS`` combos as fg-bg-proto).
    - k-means, single scale (``<scale>-kmeans<k>``), and fg-bg-kmeans (contrastive,
      k-means centroids on the fg side) — both consistently underperform the mean/
      proto/knn families in practice, so they're **skipped by default**; pass
      ``include_kmeans=True`` (``build_method_states``/``all_method_names``) or
      ``--include-kmeans`` (``run_experiments.py``) to opt back in.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from common import ScoringConfig
from engine import MethodState, ScalePrototype, pool_scale_patches

from dinoisawesome.instance_detection import compute_exemplar_features

# Single-scale combos are the always-on baseline; multi-scale combos are opt-in (add
# entries here to compare e.g. "mid+close-max" against the single scales — kept empty
# by default since the original ablation found combining scales didn't help).
MEAN_SCALE_COMBOS: dict[str, list[str]] = {
    "global": ["global"],
    "mid": ["mid"],
    "close": ["close"],
}
MULTI_SCALE_MEAN_COMBOS: dict[str, list[str]] = {}
MULTI_SCALE_COMBINE: list[Literal["max", "mean"]] = ["max", "mean"]

# fg source scales / bg source scales for every contrastive method (fg-bg-mean,
# fg-bg-proto, fg-bg-knn, fg-bg-kmeans all share this combo table, one MethodState per
# combo per family). "bg" always spans every scale so background sees far-field
# context, not just a narrow local neighbourhood — see engine.pool_scale_patches.
# Single-fg-scale combos are built by build_fgbg_mean_states ("fg-bg-mean(...)");
# multi-fg-scale combos are built by build_fgbg_multiproto_states ("fg-bg-proto(...)")
# instead — see that function for why they aren't also averaged into a single vector.
FGBG_SOURCE_COMBOS: dict[str, dict[str, list[str]]] = {
    "global/all": {"fg": ["global"], "bg": ["global", "mid", "close"]},
    "mid/all": {"fg": ["mid"], "bg": ["global", "mid", "close"]},
    "global+mid/all": {"fg": ["global", "mid"], "bg": ["global", "mid", "close"]},
    "mid+close/all": {"fg": ["mid", "close"], "bg": ["global", "mid", "close"]},
    "global+mid+close/all": {
        "fg": ["global", "mid", "close"],
        "bg": ["global", "mid", "close"],
    },
}


def build_mean_states(
    scale_protos: dict[str, ScalePrototype],
    combos: dict[str, list[str]],
    combine_modes: list[Literal["max", "mean"]],
) -> dict[str, MethodState]:
    """Single- and multi-scale mean-prototype methods (the original ablation baseline)."""
    states: dict[str, MethodState] = {}
    for combo_name, members in combos.items():
        if not all(m in scale_protos for m in members):
            continue
        protos = [scale_protos[m].mean_prototype for m in members]
        stacked = protos[0] if len(protos) == 1 else torch.cat(protos, dim=0)
        if len(members) == 1:
            states[combo_name] = MethodState(
                combo_name, "single", stacked, roi_source_method=combo_name
            )
            continue
        for mode in combine_modes:
            name = f"{combo_name}-{mode}"
            if mode == "mean":
                mean_proto = F.normalize(stacked.mean(dim=0, keepdim=True), p=2, dim=-1)
                states[name] = MethodState(name, "single", mean_proto, roi_source_method=members[0])
            else:
                states[name] = MethodState(name, "multi", stacked, roi_source_method=members[0])
    return states


def build_kmeans_states(
    scale_protos: dict[str, ScalePrototype], scales: list[str], ks: tuple[int, ...]
) -> dict[str, MethodState]:
    """One k-means(k) method per (scale, k): same pooled foreground patches as the mean
    method for that scale, but k centroids instead of a single mean vector.
    """
    states: dict[str, MethodState] = {}
    for scale in scales:
        if scale not in scale_protos:
            continue
        fg_patches = pool_scale_patches(scale_protos, [scale], want_fg=True)
        for k in ks:
            kk = min(k, fg_patches.shape[0])
            centroids = compute_exemplar_features(fg_patches, mode="kmeans", k=kk)  # (kk, C)
            name = f"{scale}-kmeans{k}"
            states[name] = MethodState(name, "multi", centroids, roi_source_method=name)
    return states


def build_fgbg_mean_states(
    scale_protos: dict[str, ScalePrototype], combos: dict[str, dict[str, list[str]]]
) -> dict[str, MethodState]:
    """Contrastive fg-bg with both sides collapsed to a single mean vector.

    Only built for combos with a single fg scale — a multi-scale fg combo keeps its
    scales as separate prototype rows instead (see ``build_fgbg_multiproto_states``),
    rather than also being averaged down to one vector here.
    """
    states: dict[str, MethodState] = {}
    for combo_name, sources in combos.items():
        fg_scales, bg_scales = sources["fg"], sources["bg"]
        if len(fg_scales) != 1 or not all(s in scale_protos for s in [*fg_scales, *bg_scales]):
            continue
        fg = F.normalize(
            torch.cat([scale_protos[s].mean_prototype for s in fg_scales], dim=0).mean(
                dim=0, keepdim=True
            ),
            p=2,
            dim=-1,
        )
        bg = F.normalize(
            torch.cat([scale_protos[s].bg_prototype for s in bg_scales], dim=0).mean(
                dim=0, keepdim=True
            ),
            p=2,
            dim=-1,
        )
        payload = torch.cat([fg, bg], dim=0)  # (2, C): row 0 = fg, row 1 = bg
        name = f"fg-bg-mean({combo_name})"
        states[name] = MethodState(name, "fgbg", payload, roi_source_method=fg_scales[0])
    return states


def build_fgbg_multiproto_states(
    scale_protos: dict[str, ScalePrototype], combos: dict[str, dict[str, list[str]]]
) -> dict[str, MethodState]:
    """Multi-datasource contrastive fg-bg: one prototype row per fg scale, kept
    separate (not averaged into one vector like ``build_fgbg_mean_states``) so scoring
    (the "fgbg_multi" kind) can take the best-matching scale's similarity per query
    patch, minus the (still single-mean) bg similarity — the same mechanism
    ``build_fgbg_kmeans_states`` uses for its k-means centroids, just fed each fg
    scale's own mean prototype instead of learned centroids.

    Only built for combos with more than one fg scale — a single fg scale has nothing
    to keep separate, so it's just ``build_fgbg_mean_states``'s "fg-bg-mean(...)".
    """
    states: dict[str, MethodState] = {}
    for combo_name, sources in combos.items():
        fg_scales, bg_scales = sources["fg"], sources["bg"]
        if len(fg_scales) < 2 or not all(s in scale_protos for s in [*fg_scales, *bg_scales]):
            continue
        fg_protos = torch.cat([scale_protos[s].mean_prototype for s in fg_scales], dim=0)  # (K, C)
        bg = F.normalize(
            torch.cat([scale_protos[s].bg_prototype for s in bg_scales], dim=0).mean(
                dim=0, keepdim=True
            ),
            p=2,
            dim=-1,
        )
        payload = torch.cat([fg_protos, bg], dim=0)  # (K + 1, C), last row = bg
        name = f"fg-bg-proto({combo_name})"
        states[name] = MethodState(name, "fgbg_multi", payload, roi_source_method=fg_scales[0])
    return states


def build_fgbg_knn_states(
    scale_protos: dict[str, ScalePrototype], combos: dict[str, dict[str, list[str]]]
) -> dict[str, MethodState]:
    """Contrastive fg-bg where each side is a per-patch kNN gallery, not a mean vector."""
    states: dict[str, MethodState] = {}
    for combo_name, sources in combos.items():
        fg_scales, bg_scales = sources["fg"], sources["bg"]
        if not all(s in scale_protos for s in [*fg_scales, *bg_scales]):
            continue
        fg_bank = pool_scale_patches(scale_protos, fg_scales, want_fg=True)
        bg_bank = pool_scale_patches(scale_protos, bg_scales, want_fg=False)
        name = f"fg-bg-knn({combo_name})"
        states[name] = MethodState(
            name, "knn_fgbg", fg_bank=fg_bank, bg_bank=bg_bank, roi_source_method=fg_scales[0]
        )
    return states


def build_fgbg_kmeans_states(
    scale_protos: dict[str, ScalePrototype],
    combos: dict[str, dict[str, list[str]]],
    ks: tuple[int, ...],
) -> dict[str, MethodState]:
    """Contrastive fg-bg where the foreground side is k k-means centroids (max-cosine
    over them) instead of one mean vector; background stays a single mean. One method
    per (combo, k) in *ks*.
    """
    states: dict[str, MethodState] = {}
    for combo_name, sources in combos.items():
        fg_scales, bg_scales = sources["fg"], sources["bg"]
        if not all(s in scale_protos for s in [*fg_scales, *bg_scales]):
            continue
        fg_patches = pool_scale_patches(scale_protos, fg_scales, want_fg=True)
        bg = F.normalize(
            torch.cat([scale_protos[s].bg_prototype for s in bg_scales], dim=0).mean(
                dim=0, keepdim=True
            ),
            p=2,
            dim=-1,
        )
        for k in ks:
            kk = min(k, fg_patches.shape[0])
            fg_centroids = compute_exemplar_features(fg_patches, mode="kmeans", k=kk)  # (kk, C)
            payload = torch.cat([fg_centroids, bg], dim=0)  # (kk + 1, C), last row = bg
            name = f"fg-bg-kmeans{k}({combo_name})"
            states[name] = MethodState(name, "fgbg_multi", payload, roi_source_method=fg_scales[0])
    return states


def build_method_states(
    scale_protos: dict[str, ScalePrototype],
    scoring_cfg: ScoringConfig,
    scales: list[str],
    method_names: list[str] | None = None,
    include_kmeans: bool = False,
) -> dict[str, MethodState]:
    """Build every registered method's :class:`~engine.MethodState`, skipping combos
    whose scale(s) were dropped for this pair (e.g. "close" below ``min_crop_size``).

    *method_names*, if given, restricts the result to those names (still built from
    the same registry, so adding a new family only ever requires a new
    ``build_*_states`` call below — never a change to callers).

    The k-means families (``build_kmeans_states``, ``build_fgbg_kmeans_states``)
    consistently underperform the mean/proto/knn families, so they're only built when
    *include_kmeans* is True, or when *method_names* explicitly names one (so
    ``--methods mid-kmeans3`` still works without also passing ``--include-kmeans``).
    """
    all_combos = {**MEAN_SCALE_COMBOS, **MULTI_SCALE_MEAN_COMBOS}
    states: dict[str, MethodState] = {}
    states.update(build_mean_states(scale_protos, all_combos, MULTI_SCALE_COMBINE))
    states.update(build_fgbg_mean_states(scale_protos, FGBG_SOURCE_COMBOS))
    states.update(build_fgbg_multiproto_states(scale_protos, FGBG_SOURCE_COMBOS))
    states.update(build_fgbg_knn_states(scale_protos, FGBG_SOURCE_COMBOS))
    include_kmeans = include_kmeans or (
        method_names is not None and any("kmeans" in name for name in method_names)
    )
    if include_kmeans:
        states.update(build_kmeans_states(scale_protos, scales, scoring_cfg.kmeans_ks))
        states.update(
            build_fgbg_kmeans_states(scale_protos, FGBG_SOURCE_COMBOS, scoring_cfg.kmeans_ks)
        )
    if method_names is not None:
        states = {name: state for name, state in states.items() if name in method_names}
    return states


def all_method_names(
    scales: list[str], ks: tuple[int, ...] = (3, 8), include_kmeans: bool = False
) -> list[str]:
    """Every base (stage-1) method name the registry can produce, in display order.

    Two-stage variants aren't listed here — they're named ``two-stage(<method>)`` at
    run time, once per method that is (or points at) a self-anchored single-scale
    method; see ``run_experiments.py``. *include_kmeans* mirrors
    ``build_method_states``'s flag of the same name — pass True to also list the
    (default-off) k-means method names, e.g. for an argparse ``choices=`` list that
    should still accept them explicitly.
    """
    names: list[str] = []
    for combo_name, members in {**MEAN_SCALE_COMBOS, **MULTI_SCALE_MEAN_COMBOS}.items():
        if len(members) == 1:
            names.append(combo_name)
        else:
            names.extend(f"{combo_name}-{mode}" for mode in MULTI_SCALE_COMBINE)
    names += [
        f"fg-bg-mean({name})"
        for name, sources in FGBG_SOURCE_COMBOS.items()
        if len(sources["fg"]) == 1
    ]
    names += [
        f"fg-bg-proto({name})"
        for name, sources in FGBG_SOURCE_COMBOS.items()
        if len(sources["fg"]) > 1
    ]
    names += [f"fg-bg-knn({name})" for name in FGBG_SOURCE_COMBOS]
    if include_kmeans:
        names += [f"{scale}-kmeans{k}" for scale in scales for k in ks]
        names += [f"fg-bg-kmeans{k}({name})" for name in FGBG_SOURCE_COMBOS for k in ks]
    return names
