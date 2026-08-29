"""Stage 2: render results cached by ``run_experiments.py``.

Reads only ``cache/methods/<crop_hash>__<scoring_hash>/**`` — pure numpy/python pickles,
no torch/encoder needed (mirrors ``experiments/anomaly_detection/analyze_results.py``'s
split: stage 1 does the AI work, stage 2 is plotting).

``--mode`` controls how much gets rendered on top of the summary, which always runs:

    summary   Aggregate every cached pair into a metrics table + a handful of
              comparison figures (bar chart; method x part-type and method x
              instance-type heatmaps for IoU, precision, and recall; cross-scale
              similarity heatmap; and one bar-chart breakdown per instance_type /
              "class" under figures/by_instance_type/) — for "how do the methods
              compare overall", not "what happened on this one image". Runs
              regardless of ``--mode``.
    detailed  Everything from summary, plus rich per-case figures for every
              (part_type, instance_type) pair by default (narrow with
              --part-type/--instance-type): exemplar crop overview, per-method
              score/cluster/GT breakdown, cluster-reject threshold tuning,
              two-stage ROI blob overlay. Written flat under
              figures/<instance_type>/, one file per (kind, part-instance case)
              — and, for the per-method two-stage breakdown, per (method, case)
              with the method name first in the filename so files for the same
              method across parts sort together.

Usage:
    python visualize_results.py --mode summary
    python visualize_results.py --mode detailed --part-type RHa --instance-type foam
    python visualize_results.py --methods mid mid-kmeans3   # applies to summary too
    python visualize_results.py --include-kmeans --include-classifiers
    python visualize_results.py --resolution 768   # match a non-default run_experiments.py run
    python visualize_results.py --model base       # match a non-default run_experiments.py run
    python visualize_results.py --offset 0.05      # match a non-default run_experiments.py run
"""

# Logging first, matches the rest of this experiment's scripts (see run_experiments.py)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("visualize_results")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import (  # noqa: E402
    CACHE_ROOT,
    DATA_DIR,
    DEFAULT_CROP_CONFIG,
    DEFAULT_SCORING_CONFIG,
    FIGURES_ROOT,
    RESULTS_ROOT,
    CropConfig,
    PairKey,
    ScoringConfig,
)
from methods import DEFAULT_SKIP_FGBG_COMBOS, all_method_names  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _shared.clustering import dbscan_clusters_from_mask  # noqa: E402
from _shared.mask_geometry import mask_iou  # noqa: E402

CROP_CFG = DEFAULT_CROP_CONFIG
SCORING_CFG = DEFAULT_SCORING_CONFIG


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["summary", "detailed"], default="summary")
    parser.add_argument("--part-type", default=None, help="Detailed mode: focal part type")
    parser.add_argument(
        "--instance-type", default=None, help="Detailed mode: focal instance-type group"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        # include_kmeans=True/include_classifiers=True here just so the (default-off)
        # k-means/linear-probe/svm names remain valid --methods choices, mirroring
        # run_experiments.py's own --methods; whether they're actually charted/rendered
        # by default is controlled by --include-kmeans/--include-classifiers.
        choices=all_method_names(
            list(CROP_CFG.scales),
            SCORING_CFG.kmeans_ks,
            include_kmeans=True,
            include_classifiers=True,
        ),
        help=(
            "Restrict to these methods (summary charts and, if --mode detailed, "
            "rendered figures too; default: every registered method except "
            "k-means/svm/linear-probe)"
        ),
    )
    parser.add_argument(
        "--include-kmeans",
        action="store_true",
        help="Also chart/render the k-means method families (skipped by default — see methods.py)",
    )
    parser.add_argument(
        "--include-classifiers",
        action="store_true",
        help=(
            "Also chart/render the linear-probe/svm method families "
            "(skipped by default — see methods.py)"
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=(
            "Override CropConfig.img_size to match a run_experiments.py run made with "
            f"--resolution (default: {DEFAULT_CROP_CONFIG.img_size})"
        ),
    )
    parser.add_argument(
        "--model",
        choices=["small", "base", "large", "giant"],
        default=None,
        help=(
            "Override CropConfig.dino_size to match a run_experiments.py run made with "
            f"--model (default: {DEFAULT_CROP_CONFIG.dino_size!r})"
        ),
    )
    parser.add_argument(
        "--bg-enrich-crops",
        type=int,
        default=None,
        help=(
            "Match a run_experiments.py run made with --bg-enrich-crops (default: "
            f"{DEFAULT_CROP_CONFIG.bg_enrich_crops_per_scale}, i.e. off). When set to a "
            "non-default value, also renders vanilla-vs-flagged comparison figures against "
            "the plain (--bg-enrich-crops 0) cache — see comparison_bg_enrich.png."
        ),
    )
    parser.add_argument(
        "--fg-clean",
        choices=["raw", "step1", "step2_cls", "step2_center"],
        default=None,
        help=(
            "Match a run_experiments.py run made with --fg-clean (default: "
            f"{DEFAULT_SCORING_CONFIG.fg_clean_stage!r}, i.e. off). When set to a non-default "
            "value, also renders vanilla-vs-flagged comparison figures against the plain "
            "(--fg-clean raw) cache — see comparison_fg_clean.png."
        ),
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help=(
            "Match a run_experiments.py run made with --offset (default: "
            f"{DEFAULT_SCORING_CONFIG.threshold_offset}, i.e. off). When non-zero, also "
            "renders vanilla-vs-flagged comparison figures against the plain (--offset 0) "
            "cache — see comparison_offset.png."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Cache readers
# ---------------------------------------------------------------------------


def _pairs_root(
    crop_cfg: CropConfig | None = None, scoring_cfg: ScoringConfig | None = None
) -> Path:
    crop_cfg = crop_cfg if crop_cfg is not None else CROP_CFG
    scoring_cfg = scoring_cfg if scoring_cfg is not None else SCORING_CFG
    return CACHE_ROOT / "methods" / f"{crop_cfg.hash()}__{scoring_cfg.hash()}"


def discover_cached_pairs(
    crop_cfg: CropConfig | None = None, scoring_cfg: ScoringConfig | None = None
) -> list[tuple[PairKey, Path]]:
    """Every cached pair under ``(crop_cfg, scoring_cfg)``'s cache namespace — the current
    (possibly flagged) run by default, or an explicit namespace (e.g. the vanilla baseline
    for a --bg-enrich-crops/--fg-clean comparison, see build_comparison_df)."""
    root = _pairs_root(crop_cfg, scoring_cfg)
    if not root.exists():
        return []
    out = []
    for pair_dir in sorted(root.iterdir()):
        meta_path = pair_dir / "pair_meta.pkl"
        if not meta_path.exists():
            continue
        meta = pickle.loads(meta_path.read_bytes())
        pair = PairKey(
            meta["part_type"], meta["instance_type"], meta["ref_number"], meta["query_number"]
        )
        out.append((pair, pair_dir))
    return out


def load_pair_meta(pair_dir: Path) -> dict:
    return pickle.loads((pair_dir / "pair_meta.pkl").read_bytes())


def load_method_result(pair_dir: Path, method_name: str) -> dict | None:
    path = pair_dir / f"{method_name.replace('/', '-')}.pkl"
    if not path.exists():
        return None
    return pickle.loads(path.read_bytes())


def load_cross_scale(pair_dir: Path) -> list[dict]:
    path = pair_dir / "cross_scale.pkl"
    if not path.exists():
        return []
    return pickle.loads(path.read_bytes())


def load_blobs_light(pair_dir: Path, roi_source: str) -> dict | None:
    path = pair_dir / f"blobs__{roi_source}.pkl"
    if not path.exists():
        return None
    return pickle.loads(path.read_bytes())


def method_names_in(pair_dir: Path) -> list[str]:
    """True method names for every cached result in *pair_dir*.

    Reads the ``method`` field each payload was written with rather than the on-disk
    filename stem — the filename sanitises "/" (e.g. "fg-bg-proto(mid/all)" ->
    "fg-bg-proto(mid-all)"), which is lossy to invert.
    """
    names = []
    for p in pair_dir.glob("*.pkl"):
        if p.stem in ("pair_meta", "cross_scale") or p.stem.startswith("blobs__"):
            continue
        names.append(pickle.loads(p.read_bytes())["method"])
    return names


def _is_default_skipped(name: str) -> bool:
    """Mirrors ``methods.build_method_states``'s default-skip filter (see
    ``DEFAULT_SKIP_FGBG_COMBOS``), so stale cached results for opt-in-only combos —
    left behind by an old run predating that skip, or a run that named them explicitly
    via ``--methods`` — don't leak into figures just because they're still on disk.
    """
    return any(f"({combo})" in name for combo in DEFAULT_SKIP_FGBG_COMBOS)


def _base_method_order(
    method_names: list[str] | None, include_kmeans: bool, include_classifiers: bool
) -> list[str]:
    """Base (stage-1) method order for one CLI invocation — *method_names* if given
    (explicit opt-in, matching ``build_method_states``), else every registered method
    except the default-off families (k-means/svm/linear-probe unless
    *include_kmeans*/*include_classifiers*, plus ``DEFAULT_SKIP_FGBG_COMBOS``).
    """
    if method_names is not None:
        return method_names
    return [
        m
        for m in all_method_names(
            list(CROP_CFG.scales), SCORING_CFG.kmeans_ks, include_kmeans, include_classifiers
        )
        if not _is_default_skipped(m)
    ]


def _display_order(
    method_names: list[str] | None, include_kmeans: bool, include_classifiers: bool
) -> list[str]:
    """``_base_method_order`` plus each base method's ``two-stage(...)`` variant."""
    base = _base_method_order(method_names, include_kmeans, include_classifiers)
    return base + [f"two-stage({m})" for m in base]


# Plain (no CLI args) defaults — still used as the module-wide fallback wherever a
# caller doesn't have per-invocation args handy (e.g. plot helpers below).
BASE_METHOD_ORDER = _base_method_order(None, False, False)
DISPLAY_ORDER = _display_order(None, False, False)


# ---------------------------------------------------------------------------
# Pure-numpy matching helpers, imported from experiments/_shared — mask_iou and
# dbscan_clusters_from_mask are plain numpy/sklearn (no torch), matching this script's
# deliberately torch-free design (see module docstring).
# ---------------------------------------------------------------------------


def match_clusters_labeled(
    pred_clusters: list[dict], gt_clusters: list[dict], iou_thr: float
) -> tuple[list[bool], set[int]]:
    """Greedy score-order IoU matching, mirroring ``engine.match_and_score``, but returns
    per-cluster TP/FP labels and the set of unmatched GT indices (FN) instead of only
    aggregate counts — what the colored final-cluster panels need.

    Returns ``(tp_flags, fn_gt_indices)`` where ``tp_flags[i]`` says whether
    ``pred_clusters[i]`` matched a (previously unmatched) GT cluster at IoU >= *iou_thr*.
    """
    n_pred, n_gt = len(pred_clusters), len(gt_clusters)
    tp_flags = [False] * n_pred
    if n_gt == 0 or n_pred == 0:
        return tp_flags, set(range(n_gt))
    order = sorted(range(n_pred), key=lambda i: -pred_clusters[i]["score"])
    matched_gt: set[int] = set()
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
            tp_flags[i] = True
    return tp_flags, set(range(n_gt)) - matched_gt


# Cluster-outcome colors, shared by the method and two-stage breakdown panels.
COLOR_TP = "green"
COLOR_FP = "yellow"
COLOR_FN = "blue"
COLOR_REJECTED = "orange"


def _draw_scored_clusters(
    ax: plt.Axes,
    kept_clusters: list[dict],
    rejected_clusters: list[dict],
    gt_clusters: list[dict],
    iou_thr: float,
) -> tuple[int, int, int, int]:
    """Scatter *kept_clusters* (green=TP/yellow=FP), *rejected_clusters* (orange), and
    unmatched *gt_clusters* (blue=FN) onto *ax*, already holding the background image.
    Returns ``(tp, fp, fn, rejected)`` counts for the caller's title.
    """
    tp_flags, fn_idx = match_clusters_labeled(kept_clusters, gt_clusters, iou_thr)
    for cl, is_tp in zip(kept_clusters, tp_flags):
        ys_c, xs_c = np.where(cl["mask"])
        ax.scatter(xs_c, ys_c, s=14, color=COLOR_TP if is_tp else COLOR_FP, marker="s")
    for cl in rejected_clusters:
        ys_c, xs_c = np.where(cl["mask"])
        ax.scatter(xs_c, ys_c, s=14, color=COLOR_REJECTED, marker="s")
    for j in fn_idx:
        ys_c, xs_c = np.where(gt_clusters[j]["mask"])
        ax.scatter(xs_c, ys_c, s=14, color=COLOR_FN, marker="s")
    n_tp = sum(tp_flags)
    return n_tp, len(kept_clusters) - n_tp, len(fn_idx), len(rejected_clusters)


def _plot_score_histogram(
    ax: plt.Axes, raw: np.ndarray, gt_mask: np.ndarray, thr: float, legend: bool = True
) -> None:
    """GT-true vs. GT-false score histogram, GT-true on its own twin y-axis (it's vastly
    outnumbered by GT-false/background patches) — matches
    ``multiscale_crop_ablation.py``'s per-method score histogram panel.
    """
    true_vals = raw[gt_mask]
    false_vals = raw[~gt_mask]
    bins = np.linspace(raw.min(), raw.max(), 40)
    ax_true = ax.twinx()
    h_false = ax.hist(false_vals, bins=bins, alpha=0.6, color="tab:gray", label="GT-false")
    h_true = ax_true.hist(true_vals, bins=bins, alpha=0.6, color="tab:red", label="GT-true")
    ax.set_yscale("log")
    ax.tick_params(axis="y", labelcolor="tab:gray", labelsize=7)
    ax_true.tick_params(axis="y", labelcolor="tab:red", labelsize=7)
    thr_line = ax.axvline(thr, color="black", linestyle="--", linewidth=1.5)
    ax.set_title(f"score histogram (thr={thr:.3f})", fontsize=9)
    if legend:
        ax.legend(
            [h_false[2][0], h_true[2][0], thr_line],
            ["GT-false", "GT-true", "threshold"],
            fontsize=7,
        )


# ---------------------------------------------------------------------------
# Summary mode
# ---------------------------------------------------------------------------


def build_metrics_df(
    crop_cfg: CropConfig | None = None, scoring_cfg: ScoringConfig | None = None
) -> pd.DataFrame:
    rows = []
    for pair, pair_dir in discover_cached_pairs(crop_cfg, scoring_cfg):
        for name in method_names_in(pair_dir):
            result = load_method_result(pair_dir, name)
            if result is None:
                continue
            rows.append(
                {
                    "part_type": pair.part_type,
                    "instance_type": pair.instance_type,
                    "method": name,
                    **result["metrics"],
                }
            )
    return pd.DataFrame(rows)


def build_cross_scale_df() -> pd.DataFrame:
    rows = []
    for pair, pair_dir in discover_cached_pairs():
        for r in load_cross_scale(pair_dir):
            rows.append({"part_type": pair.part_type, "instance_type": pair.instance_type, **r})
    return pd.DataFrame(rows)


def _warn_incomplete_coverage(metrics_df: pd.DataFrame, counts: pd.Series) -> None:
    """Flag methods averaged over fewer pairs than the best-covered method — a bar
    computed from 1 pair looks identical to one computed from 12 unless someone checks,
    so surface the gap in the log instead of only in a chart annotation."""
    max_n = int(counts.max())
    short = counts[counts < max_n]
    if short.empty:
        return
    detail = ", ".join(f"{m}={n}/{max_n}" for m, n in short.items())
    log.warning(
        "%d method(s) have incomplete pair coverage — their summary bars average over "
        "far fewer pairs than the rest and should not be compared at face value: %s",
        len(short),
        detail,
    )


def plot_summary_bars(
    metrics_df: pd.DataFrame, order: list[str], out_path: Path, title_suffix: str = ""
) -> None:
    counts = metrics_df.groupby("method").size().reindex(order)
    _warn_incomplete_coverage(metrics_df, counts)
    summary_df = (
        metrics_df.groupby("method")[["precision", "recall", "f1", "mean_iou", "count_error"]]
        .mean()
        .reindex(order)
    )
    # Each panel is sorted independently by the metric it displays, best-first, rather than
    # sharing DISPLAY_ORDER — makes the strongest methods for that particular metric readable
    # at a glance instead of scattered across the bar chart.
    pr_order = summary_df[["precision", "recall"]].mean(axis=1).sort_values(ascending=False).index
    iou_order = summary_df["mean_iou"].sort_values(ascending=False).index

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 0.5 * len(order)), 5), constrained_layout=True)
    summary_df.loc[pr_order, ["precision", "recall", "f1"]].plot.bar(ax=axes[0])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"Precision / Recall / F1{title_suffix}")
    axes[0].set_xticklabels([f"{m} (n={counts[m]})" for m in pr_order], rotation=90, ha="center")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    summary_df.loc[iou_order, "mean_iou"].plot.bar(ax=axes[1], color="teal")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title(f"Mean matched IoU{title_suffix}")
    axes[1].set_xticklabels([f"{m} (n={counts[m]})" for m in iou_order], rotation=90, ha="center")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


def plot_summary_bars_by_instance_type(
    metrics_df: pd.DataFrame, order: list[str], out_dir: Path
) -> None:
    """One `plot_summary_bars` figure per `instance_type` ("class"), for side-by-side
    comparison against the overall aggregate (which can mask a method that's strong on
    one class and weak on another)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for instance_type in sorted(metrics_df["instance_type"].unique()):
        class_df = metrics_df[metrics_df["instance_type"] == instance_type]
        class_order = [m for m in order if m in class_df["method"].unique()]
        n_pairs = class_df["part_type"].nunique()
        plot_summary_bars(
            class_df,
            class_order,
            out_dir / f"summary_bar__{instance_type}.png",
            title_suffix=f" — instance_type={instance_type!r} (n={n_pairs} part types)",
        )


def plot_metric_heatmap(
    metrics_df: pd.DataFrame,
    order: list[str],
    out_path: Path,
    metric_col: str = "mean_iou",
    metric_label: str = "mean matched IoU",
    group_col: str = "part_type",
) -> None:
    pivot = (
        metrics_df.groupby(["method", group_col])[metric_col]
        .mean()
        .unstack(group_col)
        .reindex(index=order)
    )
    fig, ax = plt.subplots(
        figsize=(1.7 * len(pivot.columns) + 2.5, 0.4 * len(pivot) + 2), constrained_layout=True
    )
    im = ax.imshow(pivot.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if val < 0.6 else "black",
                )
    fig.colorbar(im, ax=ax, label=metric_label)
    ax.set_title(f"{metric_label} per method x {group_col}")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


# (metric_col, metric_label, output-filename stem) for every method x {part_type,
# instance_type} heatmap `run_summary` renders — one IoU heatmap plus the same grid for
# precision and recall.
METRIC_HEATMAPS = [
    ("mean_iou", "mean matched IoU", "iou_heatmap"),
    ("precision", "mean precision", "precision_heatmap"),
    ("recall", "mean recall", "recall_heatmap"),
]


def plot_cross_scale_heatmap(cross_df: pd.DataFrame, out_path: Path, base_order: list[str]) -> None:
    if cross_df.empty:
        return
    gt = cross_df[cross_df["gt_present"]]
    if gt.empty:
        return
    matrix = gt.groupby(["roi_source", "score_scale"])["iou"].mean().unstack("score_scale")
    order = [m for m in base_order if m in matrix.index or m in matrix.columns]
    matrix = matrix.reindex(index=order, columns=order)
    fig, ax = plt.subplots(
        figsize=(0.6 * len(order) + 3, 0.6 * len(order) + 3), constrained_layout=True
    )
    vmax = max(float(np.nanmax(matrix.values)), 1e-6)
    im = ax.imshow(matrix.values, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, ha="center", fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("prototype used to score the crop")
    ax.set_ylabel("method whose raw map located the crop (ROI source)")
    for i in range(len(order)):
        for j in range(len(order)):
            val = matrix.values[i, j]
            label = f"{val:.2f}" if np.isfinite(val) else "n/a"
            color = "white" if (np.isfinite(val) and val < vmax * 0.6) else "black"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=7)
    fig.colorbar(im, ax=ax, label="mean IoU vs. crop GT")
    ax.set_title(f"Cross-scale similarity (n={len(gt)} GT-overlapping blobs)", fontsize=10)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


def run_summary(
    method_names: list[str] | None = None,
    include_kmeans: bool = False,
    include_classifiers: bool = False,
) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    metrics_df = build_metrics_df()
    if metrics_df.empty:
        raise RuntimeError(
            f"No cached results under {_pairs_root()} — run run_experiments.py first."
        )
    metrics_df.to_csv(RESULTS_ROOT / "metrics_records.csv", index=False)
    log.info("wrote %s (%d rows)", RESULTS_ROOT / "metrics_records.csv", len(metrics_df))

    n_pairs = metrics_df[["part_type", "instance_type"]].drop_duplicates().shape[0]
    if n_pairs <= 1:
        log.warning(
            "Only %d cached pair(s) total — every method's precision/recall is a single "
            "win/loss (0.0 or 1.0), not a meaningful average. Run run_experiments.py "
            "without --part-types/--limit-pairs for a real comparison.",
            n_pairs,
        )

    base_order = _base_method_order(method_names, include_kmeans, include_classifiers)
    order = [
        m
        for m in _display_order(method_names, include_kmeans, include_classifiers)
        if m in metrics_df["method"].unique()
    ]
    plot_summary_bars(
        metrics_df,
        order,
        FIGURES_ROOT / "summary_bar.png",
        title_suffix=" (mean across every cached pair)",
    )
    for metric_col, metric_label, filename_stem in METRIC_HEATMAPS:
        for group_col in ("part_type", "instance_type"):
            plot_metric_heatmap(
                metrics_df,
                order,
                FIGURES_ROOT / f"{filename_stem}_by_{group_col}.png",
                metric_col=metric_col,
                metric_label=metric_label,
                group_col=group_col,
            )
    plot_summary_bars_by_instance_type(metrics_df, order, FIGURES_ROOT / "by_instance_type")

    cross_df = build_cross_scale_df()
    if not cross_df.empty:
        cross_df.to_csv(RESULTS_ROOT / "cross_scale_records.csv", index=False)
        plot_cross_scale_heatmap(cross_df, FIGURES_ROOT / "cross_scale_heatmap.png", base_order)

    n_classes = metrics_df["instance_type"].nunique()
    log.info(
        "Summary done: %d pairs, %d methods, %d instance-type breakdowns.",
        n_pairs,
        len(order),
        n_classes,
    )


# ---------------------------------------------------------------------------
# Vanilla-vs-flagged comparison (Phase 1 bg-enrich / Phase 2 fg-clean)
#
# Both flags live in their own crop/scoring-hash cache namespace (see common.py's two-tier
# cache docstring), so "does this flag help" means loading the vanilla (flag off) namespace
# alongside the current (flagged) one and comparing per-method metrics side by side. Only
# triggered from main() when the respective CLI flag differs from its default — a plain run
# costs nothing extra and renders no comparison figures.
# ---------------------------------------------------------------------------


def build_comparison_df(
    vanilla_crop_cfg: CropConfig,
    vanilla_scoring_cfg: ScoringConfig,
    flagged_crop_cfg: CropConfig,
    flagged_scoring_cfg: ScoringConfig,
    flagged_label: str,
) -> pd.DataFrame | None:
    """Long-form (method, variant, metrics...) dataframe pairing the vanilla cache namespace
    against the current flagged one. Returns None (logged) if the vanilla namespace has no
    cached results — the flag comparison needs a plain ``run_experiments.py`` run first.
    """
    vanilla_root = _pairs_root(vanilla_crop_cfg, vanilla_scoring_cfg)
    vanilla_df = build_metrics_df(vanilla_crop_cfg, vanilla_scoring_cfg)
    if vanilla_df.empty:
        log.warning(
            "No cached vanilla-baseline results under %s — run run_experiments.py with "
            "defaults (no --bg-enrich-crops/--fg-clean) first to enable the vanilla-vs-"
            "flagged comparison. Skipping.",
            vanilla_root,
        )
        return None
    flagged_df = build_metrics_df(flagged_crop_cfg, flagged_scoring_cfg)
    return pd.concat(
        [vanilla_df.assign(variant="vanilla"), flagged_df.assign(variant=flagged_label)],
        ignore_index=True,
    )


def plot_variant_comparison_bars(
    comparison_df: pd.DataFrame, order: list[str], flagged_label: str, out_path: Path, title: str
) -> None:
    """Grouped (method x variant) bar chart — vanilla vs *flagged_label* — for F1 and mean
    IoU, the two-panel analogue of ``plot_summary_bars`` with variant as the hue instead of
    a single series."""
    variants = ["vanilla", flagged_label]
    summary = comparison_df.groupby(["method", "variant"])[["f1", "mean_iou"]].mean()

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 0.6 * len(order)), 5), constrained_layout=True)
    for ax, metric, color in [(axes[0], "f1", None), (axes[1], "mean_iou", "teal")]:
        pivot = summary[metric].unstack("variant").reindex(index=order, columns=variants)
        pivot.plot.bar(ax=ax, color=(["#b0b0b0", color] if color else None))
        ax.set_ylim(0, 1.0 if metric == "mean_iou" else 1.05)
        ax.set_title(f"{'Mean matched IoU' if metric == 'mean_iou' else 'F1'}{title}")
        ax.set_xticklabels(order, rotation=90, ha="center")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


def write_comparison_csv(
    comparison_df: pd.DataFrame, order: list[str], flagged_label: str, out_path: Path
) -> None:
    """Per-method vanilla/flagged metrics side by side, plus flagged-minus-vanilla deltas —
    the numeric backing for ``plot_variant_comparison_bars``' bar chart."""
    variants = ["vanilla", flagged_label]
    wide = (
        comparison_df.groupby(["method", "variant"])[["precision", "recall", "f1", "mean_iou"]]
        .mean()
        .unstack("variant")
        .reindex(
            index=order,
            columns=pd.MultiIndex.from_product(
                [["precision", "recall", "f1", "mean_iou"], variants]
            ),
        )
    )
    for metric in ("precision", "recall", "f1", "mean_iou"):
        wide[(metric, f"delta_{flagged_label}_minus_vanilla")] = (
            wide[(metric, flagged_label)] - wide[(metric, "vanilla")]
        )
    wide.to_csv(out_path)
    log.info("wrote %s", out_path)


def run_variant_comparison(
    flag_name: str,
    flagged_label: str,
    vanilla_crop_cfg: CropConfig,
    vanilla_scoring_cfg: ScoringConfig,
    order: list[str],
) -> None:
    """Vanilla-vs-flagged comparison figure + CSV for one Phase 1/2 flag — called from
    main() only when that flag differs from its default (see this section's docstring)."""
    comparison_df = build_comparison_df(
        vanilla_crop_cfg, vanilla_scoring_cfg, CROP_CFG, SCORING_CFG, flagged_label
    )
    if comparison_df is None:
        return
    comparison_order = [m for m in order if m in comparison_df["method"].unique()]
    plot_variant_comparison_bars(
        comparison_df,
        comparison_order,
        flagged_label,
        FIGURES_ROOT / f"comparison_{flag_name}.png",
        title=f" — vanilla vs {flagged_label} (mean across every cached pair)",
    )
    write_comparison_csv(
        comparison_df, comparison_order, flagged_label, RESULTS_ROOT / f"comparison_{flag_name}.csv"
    )
    log.info(
        "Comparison done for %s: %d method(s), vanilla vs %s.",
        flag_name,
        len(comparison_order),
        flagged_label,
    )


# ---------------------------------------------------------------------------
# Detailed mode
# ---------------------------------------------------------------------------


def _resolve_focal_pairs(
    part_type: str | None, instance_type: str | None
) -> list[tuple[PairKey, Path]]:
    """Every cached pair matching the optional filters — all pairs when both are None."""
    pairs = discover_cached_pairs()
    if not pairs:
        raise RuntimeError(
            f"No cached results under {_pairs_root()} — run run_experiments.py first."
        )
    if part_type is not None:
        pairs = [p for p in pairs if p[0].part_type == part_type]
    if instance_type is not None:
        pairs = [p for p in pairs if p[0].instance_type == instance_type]
    if not pairs:
        raise RuntimeError(
            f"No cached pair matches part_type={part_type!r} instance_type={instance_type!r}"
        )
    return pairs


def _load_images(pair: PairKey) -> tuple[Image.Image, Image.Image]:
    ref_img = Image.open(DATA_DIR / f"{pair.part_type}_{pair.ref_number}.jpg").convert("RGB")
    query_img = Image.open(DATA_DIR / f"{pair.part_type}_{pair.query_number}.jpg").convert("RGB")
    return ref_img, query_img


def render_exemplar_overview(
    pair: PairKey, meta: dict, ref_img: Image.Image, out_dir: Path
) -> None:
    scales = list(meta["scale_boxes"].keys())
    fig, axes = plt.subplots(1, len(scales), figsize=(4.5 * len(scales), 4.5))
    axes = np.atleast_1d(axes)
    for ax, scale in zip(axes, scales):
        box = meta["scale_boxes"][scale]
        crop = ref_img.crop(box)
        mask = meta["scale_patch_masks"][scale]
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(crop.size, Image.NEAREST)
        ax.imshow(crop)
        ax.imshow(np.array(mask_img), cmap="Reds", alpha=0.35 * (np.array(mask_img) > 0))
        ax.set_title(f"{scale}  ({crop.size[0]}x{crop.size[1]}px)")
        ax.axis("off")
    fig.suptitle(f"{pair.part_type} / {pair.instance_type} — exemplar crops per scale")
    fig.tight_layout()
    out_path = out_dir / f"exemplar_overview__{pair.case_slug}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


def render_method_breakdown(
    pair: PairKey,
    meta: dict,
    pair_dir: Path,
    query_img: Image.Image,
    method_names: list[str],
    out_dir: Path,
) -> None:
    """One row per method: raw score heatmap, thresholded binary + GT outline, score
    histogram, and the final clusters actually fed into P/R/F1 — colored green=TP,
    yellow=FP, blue=FN, orange=rejected by the mean-patch cluster filter. Mirrors
    ``multiscale_crop_ablation.py``'s per-method breakdown panel layout/coloring more
    closely than the previous 3-panel (query+GT / raw overlay / kept-vs-rejected) version.
    """
    q_h, q_w = meta["q_h"], meta["q_w"]
    gt_patch_mask = meta["gt_patch_mask"]
    gt_clusters = meta["gt_clusters"]

    rows = []
    for name in method_names:
        result = load_method_result(pair_dir, name)
        if result is None or "raw" not in result:
            continue
        rows.append((name, result))
    if not rows:
        log.warning("[%s] no base method results found for detailed breakdown", pair.slug)
        return

    fig, axes = plt.subplots(len(rows), 4, figsize=(22, 4.5 * len(rows)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    query_arr = np.array(query_img.resize((q_w, q_h)))
    for r, (name, result) in enumerate(rows):
        raw = result["raw"]
        thr = result["threshold"]
        pred_clusters = result["pred_clusters"]
        m = result["metrics"]

        ax = axes[r, 0]
        im0 = ax.imshow(raw, cmap="jet", aspect="auto")
        ax.set_title(f"[{name}] raw score heatmap" if r == 0 else name, fontsize=10)
        ax.set_ylabel(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im0, ax=ax, shrink=0.75, pad=0.02)

        ax = axes[r, 1]
        binary = raw > thr
        ax.imshow(binary, cmap="Greys_r", aspect="auto")
        ax.contour(gt_patch_mask.astype(float), levels=[0.5], colors="lime", linewidths=1.2)
        binary_iou = mask_iou(binary, gt_patch_mask)
        ax.set_title(f"binary (thr={thr:.3f}) + GT outline\nIoU={binary_iou:.2f}", fontsize=10)
        ax.axis("off")

        _plot_score_histogram(axes[r, 2], raw, gt_patch_mask, thr, legend=(r == 0))

        kept_clusters = [c for c in pred_clusters if not c.get("rejected")]
        rejected_clusters = [c for c in pred_clusters if c.get("rejected")]
        ax = axes[r, 3]
        ax.imshow(query_arr)
        tp, fp, fn, rejected = _draw_scored_clusters(
            ax, kept_clusters, rejected_clusters, gt_clusters, SCORING_CFG.iou_match_threshold
        )
        title = "final clusters (green=TP yellow=FP blue=FN orange=rejected)\n" if r == 0 else ""
        title += (
            f"tp={tp} fp={fp} fn={fn} rejected={rejected}\n"
            f"P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} mIoU={m['mean_iou']:.2f}"
        )
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    fig.suptitle(f"{pair.part_type} / {pair.instance_type} — per-method breakdown")
    out_path = out_dir / f"method_breakdown__{pair.case_slug}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


def render_cluster_reject_tuning(
    pair: PairKey, pair_dir: Path, method_names: list[str], out_dir: Path
) -> None:
    rows = []
    for name in method_names:
        result = load_method_result(pair_dir, name)
        if result is None or not result.get("ref_tuning_clusters"):
            continue
        rows.append((name, result))
    if not rows:
        return

    ncols = min(len(rows), 4)
    nrows = -(-len(rows) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
    for i, (name, result) in enumerate(rows):
        ax = axes[i // ncols][i % ncols]
        clusters = result["ref_tuning_clusters"]
        scores = np.array([c["mean_patch_score"] for c in clusters])
        good = np.array([c["gt_good"] for c in clusters])
        ax.scatter(
            np.where(good, 1, 0) + np.random.uniform(-0.05, 0.05, len(scores)),
            scores,
            c=["lime" if g else "red" for g in good],
            s=25,
        )
        ax.axhline(result["cluster_reject_thr"], color="black", linestyle="--", linewidth=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["bad (no GT match)", "good (GT match)"], fontsize=8)
        ax.set_ylabel("mean-patch cosine sim")
        ax.set_title(f"{name}\ncutoff={result['cluster_reject_thr']:.3f}", fontsize=9)
    for j in range(len(rows), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"{pair.part_type} / {pair.instance_type} — cluster-reject threshold tuning")
    fig.tight_layout()
    out_path = out_dir / f"cluster_reject_tuning__{pair.case_slug}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


def render_two_stage_overview(
    pair: PairKey, pair_dir: Path, query_img: Image.Image, out_dir: Path
) -> None:
    roi_sources = [p.stem.removeprefix("blobs__") for p in pair_dir.glob("blobs__*.pkl")]
    if not roi_sources:
        return
    ncols = min(len(roi_sources), 3)
    nrows = -(-len(roi_sources) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows), squeeze=False)
    for i, roi_source in enumerate(sorted(roi_sources)):
        ax = axes[i // ncols][i % ncols]
        payload = load_blobs_light(pair_dir, roi_source)
        if payload is None:
            continue
        ax.imshow(query_img)
        for blob in payload["blobs"]:
            x0, y0, x1, y1 = blob["px_bbox"]
            ax.add_patch(
                plt.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=1.5
                )
            )
        two_stage = load_method_result(pair_dir, f"two-stage({roi_source})")
        title = f"roi_source={roi_source} ({len(payload['blobs'])} blobs)"
        if two_stage is not None:
            m = two_stage["metrics"]
            title += (
                f"\nP={m['precision']:.2f} R={m['recall']:.2f} "
                f"F1={m['f1']:.2f} mIoU={m['mean_iou']:.2f}"
            )
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for j in range(len(roi_sources), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"{pair.part_type} / {pair.instance_type} — two-stage ROI blobs")
    fig.tight_layout()
    out_path = out_dir / f"two_stage_overview__{pair.case_slug}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


def render_two_stage_breakdown(
    pair: PairKey, pair_dir: Path, query_img: Image.Image, method_names: list[str], out_dir: Path
) -> None:
    """One figure per two-stage method, one row per ROI blob: raw score heatmap, binary +
    GT outline, score histogram, and final clusters colored green=TP/yellow=FP/blue=FN/
    orange=rejected — the blob-local analogue of ``render_method_breakdown``, using the
    ``blob_diagnostics`` ``run_experiments.py`` now caches per two-stage result (see
    ``engine.two_stage_predicted_clusters``). Methods whose two-stage cache predates that
    (no ``blob_diagnostics``) are skipped — rerun ``run_experiments.py --force`` to backfill.
    TP/FP/FN matching uses each blob's ``gt_instance_masks`` (one mask per real annotated
    query instance) when cached; older caches without it fall back to reconstructing GT
    clusters via DBSCAN on the unioned mask, with a warning — rerun with ``--force`` to
    get the more accurate per-instance panel.
    """
    for name in method_names:
        two_stage_name = f"two-stage({name})"
        result = load_method_result(pair_dir, two_stage_name)
        if result is None:
            continue
        diagnostics = result.get("blob_diagnostics")
        if not diagnostics:
            log.warning(
                "[%s/%s] no blob_diagnostics cached — rerun run_experiments.py --force",
                pair.slug,
                two_stage_name,
            )
            continue

        n_blobs = len(diagnostics)
        fig, axes = plt.subplots(n_blobs, 4, figsize=(22, 4.5 * n_blobs), constrained_layout=True)
        axes = np.atleast_2d(axes)
        for row, diag in enumerate(diagnostics):
            raw, thr, gt_mask = diag["raw"], diag["threshold"], diag["gt_mask"]
            px0, py0, px1, py1 = diag["px_bbox"]
            crop_arr = np.array(
                query_img.crop((px0, py0, px1, py1)).resize((diag["c_w"], diag["c_h"]))
            )

            ax = axes[row, 0]
            im0 = ax.imshow(raw, cmap="jet", aspect="auto")
            ax.set_title(f"blob {row} — raw score heatmap" if row == 0 else f"blob {row}")
            ax.set_ylabel(f"blob {row}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im0, ax=ax, shrink=0.75, pad=0.02)

            ax = axes[row, 1]
            binary = raw > thr
            ax.imshow(binary, cmap="Greys_r", aspect="auto")
            if gt_mask.any():
                ax.contour(gt_mask.astype(float), levels=[0.5], colors="lime", linewidths=1.2)
            ax.set_title(f"binary (thr={thr:.3f})", fontsize=10)
            ax.axis("off")

            _plot_score_histogram(axes[row, 2], raw, gt_mask, thr, legend=(row == 0))

            clusters = diag["clusters"]
            kept_clusters = [c for c in clusters if not c["rejected"]]
            rejected_clusters = [c for c in clusters if c["rejected"]]
            if "gt_instance_masks" in diag:
                # One cluster per real annotated query instance overlapping this blob —
                # never reconstructed by DBSCAN-clustering the unioned gt_mask, which
                # would silently re-merge touching-but-distinct instances into one and
                # make a correctly-split prediction score as a spurious FP.
                gt_clusters = [{"mask": m} for m in diag["gt_instance_masks"]]
            else:
                log.warning(
                    "[%s] blob diagnostics predate gt_instance_masks — falling back to "
                    "DBSCAN-on-union GT reconstruction for this panel (may over-merge "
                    "touching instances); rerun run_experiments.py --force to backfill",
                    pair.slug,
                )
                gt_clusters = dbscan_clusters_from_mask(
                    gt_mask, SCORING_CFG.gt_dbscan_eps, SCORING_CFG.gt_dbscan_min_samples
                )
            ax = axes[row, 3]
            ax.imshow(crop_arr)
            tp, fp, fn, rejected = _draw_scored_clusters(
                ax, kept_clusters, rejected_clusters, gt_clusters, SCORING_CFG.iou_match_threshold
            )
            title = (
                "final clusters (green=TP yellow=FP blue=FN orange=rejected)\n" if row == 0 else ""
            )
            title += f"tp={tp} fp={fp} fn={fn} rejected={rejected}"
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        m = result["metrics"]
        fig.suptitle(
            f"{pair.part_type} / {pair.instance_type} — two-stage({name}) per-blob breakdown "
            f"(roi_source={result['roi_source']})\n"
            f"overall P={m['precision']:.2f} R={m['recall']:.2f} "
            f"F1={m['f1']:.2f} mIoU={m['mean_iou']:.2f}"
        )
        safe_name = name.replace("/", "-")
        out_path = out_dir / f"two_stage_breakdown__{safe_name}__{pair.case_slug}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("wrote %s", out_path)


def run_detailed(
    part_type: str | None,
    instance_type: str | None,
    methods: list[str] | None,
    include_kmeans: bool = False,
    include_classifiers: bool = False,
) -> None:
    pairs = _resolve_focal_pairs(part_type, instance_type)
    log.info("[detailed] rendering %d pair(s)", len(pairs))
    default_base_order = _base_method_order(methods, include_kmeans, include_classifiers)
    for pair, pair_dir in pairs:
        log.info("[detailed] focal pair: %s", pair.slug)
        meta = load_pair_meta(pair_dir)
        ref_img, query_img = _load_images(pair)

        out_dir = FIGURES_ROOT / pair.safe_instance_type
        out_dir.mkdir(parents=True, exist_ok=True)

        render_exemplar_overview(pair, meta, ref_img, out_dir)

        available = method_names_in(pair_dir)
        base_available = [m for m in available if not m.startswith("two-stage(")]
        selected = (
            methods
            if methods is not None
            else [m for m in default_base_order if m in base_available]
        )
        render_method_breakdown(pair, meta, pair_dir, query_img, selected, out_dir)
        render_cluster_reject_tuning(pair, pair_dir, selected, out_dir)
        render_two_stage_overview(pair, pair_dir, query_img, out_dir)
        render_two_stage_breakdown(pair, pair_dir, query_img, selected, out_dir)
        log.info("[detailed] figures written to %s", out_dir)
    log.info("[detailed] done: %d pair(s) rendered", len(pairs))


def main() -> None:
    global CROP_CFG, SCORING_CFG
    args = _parse_args()
    if args.resolution is not None:
        CROP_CFG = dataclasses.replace(CROP_CFG, img_size=args.resolution)
    if args.model is not None:
        # layer_idx=None re-triggers CropConfig.__post_init__'s "last block of this
        # size" derivation — see run_experiments.py's matching comment.
        CROP_CFG = dataclasses.replace(CROP_CFG, dino_size=args.model, layer_idx=None)
    if args.bg_enrich_crops is not None:
        CROP_CFG = dataclasses.replace(CROP_CFG, bg_enrich_crops_per_scale=args.bg_enrich_crops)
    if args.fg_clean is not None:
        SCORING_CFG = dataclasses.replace(SCORING_CFG, fg_clean_stage=args.fg_clean)
    if args.offset:
        SCORING_CFG = dataclasses.replace(SCORING_CFG, threshold_offset=args.offset)

    run_summary(args.methods, args.include_kmeans, args.include_classifiers)
    if args.mode == "detailed":
        run_detailed(
            args.part_type,
            args.instance_type,
            args.methods,
            args.include_kmeans,
            args.include_classifiers,
        )

    # Every comparison's "vanilla" side resets only the augmentation flags (bg_enrich,
    # fg_clean, threshold_offset) to their defaults, while keeping every other CROP_CFG/
    # SCORING_CFG field — resolution, model, etc. — exactly as passed on this invocation.
    # Using DEFAULT_CROP_CONFIG/DEFAULT_SCORING_CONFIG outright would ignore --resolution/
    # --model and look up a baseline cache for the wrong model size entirely. Shared across
    # all three comparisons so running multiple flags together (e.g. --bg-enrich-crops +
    # --fg-clean) compares each against the same plain baseline instead of needing a
    # separate cached run per leave-one-out combination. Requires one baseline
    # `run_experiments.py` run (same --resolution/--model, no augmentation flags) to
    # populate that baseline cache.
    vanilla_crop_cfg = dataclasses.replace(
        CROP_CFG,
        bg_enrich_crops_per_scale=DEFAULT_CROP_CONFIG.bg_enrich_crops_per_scale,
        bg_enrich_max_overlap_fraction=DEFAULT_CROP_CONFIG.bg_enrich_max_overlap_fraction,
        bg_enrich_seed=DEFAULT_CROP_CONFIG.bg_enrich_seed,
    )
    vanilla_scoring_cfg = dataclasses.replace(
        SCORING_CFG,
        fg_clean_stage=DEFAULT_SCORING_CONFIG.fg_clean_stage,
        threshold_offset=DEFAULT_SCORING_CONFIG.threshold_offset,
    )

    display_order = _display_order(args.methods, args.include_kmeans, args.include_classifiers)
    if CROP_CFG.bg_enrich_crops_per_scale != DEFAULT_CROP_CONFIG.bg_enrich_crops_per_scale:
        run_variant_comparison(
            "bg_enrich",
            f"bg_enrich={CROP_CFG.bg_enrich_crops_per_scale}",
            vanilla_crop_cfg,
            vanilla_scoring_cfg,
            display_order,
        )
    if SCORING_CFG.fg_clean_stage != DEFAULT_SCORING_CONFIG.fg_clean_stage:
        run_variant_comparison(
            "fg_clean",
            f"fg_clean={SCORING_CFG.fg_clean_stage}",
            vanilla_crop_cfg,
            vanilla_scoring_cfg,
            display_order,
        )
    if SCORING_CFG.threshold_offset != DEFAULT_SCORING_CONFIG.threshold_offset:
        run_variant_comparison(
            "offset",
            f"offset={SCORING_CFG.threshold_offset}",
            vanilla_crop_cfg,
            vanilla_scoring_cfg,
            display_order,
        )


if __name__ == "__main__":
    main()
