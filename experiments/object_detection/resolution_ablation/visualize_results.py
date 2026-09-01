"""Stage 2 for the resolution + model-size ablation: aggregate & plot metrics across
(encoder resolution, DINOv3 backbone size) combos, reading straight from
``multiscale_ablation``'s cache tree.

Each swept combo is just a different ``CropConfig(img_size=..., dino_size=...,
layer_idx=...)``, and ``multiscale_ablation``'s cache is already keyed by
``crop_cfg.hash()`` (see that module's ``common.py``), so this script never touches the
encoder — it re-derives each combo's cache path the same way ``run_experiments.py``
(this dir) wrote it, tags every cached metrics row with its resolution and size, and
reuses ``multiscale_ablation/visualize_results.py``'s pure-pandas plotting helpers
(which are already generic over an arbitrary "group by" column) to compare methods
across resolutions/sizes the same way that script compares them across
part_type/instance_type.

``layer_idx`` isn't swept directly — it's derived from ``dino_size`` via
``LAYER_IDX_BY_SIZE`` below, which must match whatever ``run_experiments.py`` actually
derived (live, from the loaded backbone's block count) when it wrote the cache for that
size. See that script's module docstring for why this is hardcoded here instead of
loaded live.

Usage:
    python visualize_results.py
    python visualize_results.py --resolutions 512 1024 --sizes large
    python visualize_results.py --methods global mid close
    python visualize_results.py --include-kmeans --include-classifiers
"""

# Logging first, matches the rest of this experiment's scripts (see run_experiments.py)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("resolution_ablation.visualize_results")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "multiscale_ablation"))
from common import (  # noqa: E402
    DEFAULT_CROP_CONFIG,
    DEFAULT_SCORING_CONFIG,
    REPO_ROOT,
)
from methods import all_method_names  # noqa: E402
from visualize_results import (  # noqa: E402
    _pairs_root,
    discover_cached_pairs,
    load_method_result,
    method_names_in,
    plot_metric_heatmap,
    plot_summary_bars,
)

# Kept in sync with run_experiments.py's own constants of the same names in this dir —
# not imported from it: sys.path has multiscale_ablation inserted ahead of this
# directory (see above), so a plain `from run_experiments import ...` here would
# silently resolve to multiscale_ablation's run_experiments.py instead of the sibling
# script in this directory.
DEFAULT_RESOLUTIONS = (256, 512, 768, 1024, 1536)
DEFAULT_SIZES = ("small", "base", "large")

# depth (transformer block count) - 1, i.e. the "last block" every CropConfig.layer_idx
# in this ablation actually uses (see run_experiments.py's module docstring). Verified
# by instantiating each DINOv3 backbone: small=12 blocks, base=12, large=24.
LAYER_IDX_BY_SIZE = {"small": 11, "base": 11, "large": 23}

SCORING_CFG = DEFAULT_SCORING_CONFIG
OUTPUT_DIR = REPO_ROOT / "outputs" / "object_detection" / "resolution_ablation"
RESULTS_ROOT = OUTPUT_DIR / "results"
FIGURES_ROOT = RESULTS_ROOT / "figures"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
        help="img_size values to compare (default: 256 512 768 1024 1536)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=list(DEFAULT_SIZES),
        choices=list(LAYER_IDX_BY_SIZE),
        help="DINOv3 backbone sizes to compare (default: small base large)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        # include_kmeans=True/include_classifiers=True here just so the (default-off)
        # k-means/linear-probe/svm names remain valid --methods choices, mirroring
        # run_experiments.py's own --methods; whether they're actually charted is
        # controlled by --include-kmeans/--include-classifiers, not by this list.
        choices=all_method_names(
            list(DEFAULT_CROP_CONFIG.scales),
            DEFAULT_SCORING_CONFIG.kmeans_ks,
            include_kmeans=True,
            include_classifiers=True,
        ),
        help=(
            "Restrict to these methods (default: every registered method except "
            "k-means/svm/linear-probe)"
        ),
    )
    parser.add_argument(
        "--include-kmeans",
        action="store_true",
        help=(
            "Also chart the k-means method families (skipped by default — see methods.py); "
            "only has an effect if run_experiments.py was also run with --include-kmeans"
        ),
    )
    parser.add_argument(
        "--include-classifiers",
        action="store_true",
        help=(
            "Also chart the linear-probe/svm method families (skipped by default — see "
            "methods.py); only has an effect if run_experiments.py was also run with "
            "--include-classifiers"
        ),
    )
    return parser.parse_args()


def _display_order(
    method_names: list[str] | None, include_kmeans: bool, include_classifiers: bool
) -> list[str]:
    """Method display order, mirroring ``multiscale_ablation/visualize_results.py``'s
    module-level ``DISPLAY_ORDER`` but responsive to
    ``--methods``/``--include-kmeans``/``--include-classifiers`` instead of hardcoding
    ``include_kmeans=False``/``include_classifiers=False`` — otherwise k-means/svm/
    linear-probe results cached via ``run_experiments.py --include-kmeans``/
    ``--include-classifiers`` would never be chartable here.
    """
    base = method_names or all_method_names(
        list(DEFAULT_CROP_CONFIG.scales),
        DEFAULT_SCORING_CONFIG.kmeans_ks,
        include_kmeans,
        include_classifiers,
    )
    return base + [f"two-stage({m})" for m in base]


def build_metrics_df(resolutions: list[int], sizes: list[str]) -> pd.DataFrame:
    rows = []
    for size in sizes:
        for resolution in resolutions:
            crop_cfg = dataclasses.replace(
                DEFAULT_CROP_CONFIG,
                dino_size=size,
                img_size=resolution,
                layer_idx=LAYER_IDX_BY_SIZE[size],
            )
            pairs = discover_cached_pairs(crop_cfg, SCORING_CFG)
            if not pairs:
                log.warning(
                    "size=%s resolution=%dpx: no cached results under %s — "
                    "run run_experiments.py first",
                    size,
                    resolution,
                    _pairs_root(crop_cfg, SCORING_CFG),
                )
                continue
            for pair, pair_dir in pairs:
                for name in method_names_in(pair_dir):
                    result = load_method_result(pair_dir, name)
                    if result is None:
                        continue
                    rows.append(
                        {
                            "dino_size": size,
                            "resolution": resolution,
                            "config": f"{size}@{resolution}px",
                            "part_type": pair.part_type,
                            "instance_type": pair.instance_type,
                            "method": name,
                            **result["metrics"],
                        }
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Resolution-specific views on top of the reused multiscale_ablation plotting helpers
# ---------------------------------------------------------------------------


def plot_summary_bars_by_config(metrics_df: pd.DataFrame, order: list[str], out_dir: Path) -> None:
    """One `plot_summary_bars` figure per (size, resolution) combo, nested under
    ``<out_dir>/<size>/``, for side-by-side comparison."""
    for size in sorted(metrics_df["dino_size"].unique()):
        size_df = metrics_df[metrics_df["dino_size"] == size]
        size_dir = out_dir / size
        size_dir.mkdir(parents=True, exist_ok=True)
        for resolution in sorted(size_df["resolution"].unique()):
            res_df = size_df[size_df["resolution"] == resolution]
            res_order = [m for m in order if m in res_df["method"].unique()]
            plot_summary_bars(
                res_df,
                res_order,
                size_dir / f"summary_bar__{resolution}px.png",
                title_suffix=f" — size={size}, resolution={resolution}px",
            )


def plot_metric_vs_resolution(
    metrics_df: pd.DataFrame, order: list[str], out_path: Path, size: str
) -> None:
    """Trend chart for one backbone *size*: one line per method, x=resolution, for each
    of P/R/F1/mIoU — the view a resolution *sweep* actually wants (part_type/
    instance_type aren't ordinal, resolution is), which is why it's new here rather
    than reused from ``multiscale_ablation/visualize_results.py``. Faceted per size
    rather than mixed into one chart since size is a distinct architecture, not a point
    on the same axis as resolution.
    """
    size_df = metrics_df[metrics_df["dino_size"] == size]
    metric_cols = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("mean_iou", "Mean IoU"),
    ]
    pivot_mean = size_df.groupby(["method", "resolution"])[[c for c, _ in metric_cols]].mean()

    fig, axes = plt.subplots(
        1, len(metric_cols), figsize=(5 * len(metric_cols), 4.5), constrained_layout=True
    )
    for ax, (col, label) in zip(axes, metric_cols):
        for method in order:
            if method not in pivot_mean.index.get_level_values("method"):
                continue
            series = pivot_mean.loc[method, col].sort_index()
            ax.plot(series.index, series.values, marker="o", label=method)
        ax.set_xlabel("resolution (px)")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.05)
        ax.set_title(label)
        ax.set_xticks(sorted(size_df["resolution"].unique()))
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=8)
    fig.suptitle(f"Metric vs. resolution — size={size} (mean across every cached pair)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


def run_summary(
    resolutions: list[int],
    sizes: list[str],
    method_names: list[str] | None = None,
    include_kmeans: bool = False,
    include_classifiers: bool = False,
) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    metrics_df = build_metrics_df(resolutions, sizes)
    if metrics_df.empty:
        raise RuntimeError(
            f"No cached results for resolutions={resolutions} sizes={sizes} — "
            "run run_experiments.py first."
        )
    metrics_df.to_csv(RESULTS_ROOT / "metrics_records.csv", index=False)
    log.info("wrote %s (%d rows)", RESULTS_ROOT / "metrics_records.csv", len(metrics_df))

    order = [
        m
        for m in _display_order(method_names, include_kmeans, include_classifiers)
        if m in metrics_df["method"].unique()
    ]
    by_size_dir = FIGURES_ROOT / "by_size"
    for size in sorted(metrics_df["dino_size"].unique()):
        size_dir = by_size_dir / size
        size_dir.mkdir(parents=True, exist_ok=True)
        plot_metric_vs_resolution(metrics_df, order, size_dir / "metric_vs_resolution.png", size)
    plot_summary_bars_by_config(metrics_df, order, by_size_dir)

    # method x resolution (aggregated over size), method x size (aggregated over
    # resolution), and method x the full (size, resolution) grid — three complementary
    # slices through the same 3D (method, size, resolution) result cube.
    for metric_col, metric_label, filename_stem in [
        ("mean_iou", "mean matched IoU", "iou_heatmap"),
        ("precision", "mean precision", "precision_heatmap"),
        ("recall", "mean recall", "recall_heatmap"),
    ]:
        for group_col in ("resolution", "dino_size", "config"):
            plot_metric_heatmap(
                metrics_df,
                order,
                FIGURES_ROOT / f"{filename_stem}_by_{group_col}.png",
                metric_col=metric_col,
                metric_label=metric_label,
                group_col=group_col,
            )

    n_pairs = metrics_df[["part_type", "instance_type"]].drop_duplicates().shape[0]
    log.info(
        "Summary done: %d size(s) x %d resolution(s), %d pairs, %d methods.",
        metrics_df["dino_size"].nunique(),
        metrics_df["resolution"].nunique(),
        n_pairs,
        len(order),
    )


def main() -> None:
    args = _parse_args()
    run_summary(
        args.resolutions,
        args.sizes,
        args.methods,
        args.include_kmeans,
        args.include_classifiers,
    )


if __name__ == "__main__":
    main()
