"""Stage 2: load the cache from run_experiments.py, compute metrics with a single
shared implementation across all four methods, and render comparison figures.

Usage:
    python analyze_results.py
    python analyze_results.py --categories bottle carpet

Reads outputs/anomaly_detection/cache/<category>/<method>/{scores.parquet,
anomaly_maps.npz, meta.json}. Writes outputs/anomaly_detection/results/{metrics.csv,
summary.md, figures/*.png}.

Metrics use `anomalib.metrics.{AUROC, AUPR, F1Max, AUPRO}` directly (bypassing
Engine/Batch) by feeding them a `types.SimpleNamespace` with the fields they expect —
`AnomalibMetric.update()` only duck-types on attribute access (`getattr(batch, key)`),
so this works without constructing anomalib's actual `ImageBatch`/`Batch` dataclass.
"""

# Logging — must be before torch import
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("analyze_results")

import argparse
import json
import types

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from common import (
    CATEGORIES,
    FIGURES_ROOT,
    METHODS,
    RESULTS_ROOT,
    anomaly_maps_path,
    cache_dir,
    scores_path,
)
from PIL import Image

# Fixed evaluation resolution for pixel-level metrics — every method's cached
# anomaly map (and every mask) is resized here so shapes always match, regardless
# of each method's own native input resolution (252 / 256 / ...).
_EVAL_SIZE = (256, 256)  # (W, H) for PIL


def _load_cached(category: str, method: str) -> tuple[pd.DataFrame, dict, dict] | None:
    if not scores_path(category, method).exists():
        return None
    df = pd.read_parquet(scores_path(category, method))
    npz = np.load(anomaly_maps_path(category, method))
    maps_by_id = dict(zip(npz["image_id"], npz["maps"]))
    meta = json.loads((cache_dir(category, method) / "meta.json").read_text())
    return df, maps_by_id, meta


def _resized_map(maps_by_id: dict, image_id: str) -> np.ndarray:
    m = maps_by_id[image_id].astype(np.float32)
    return np.array(Image.fromarray(m).resize(_EVAL_SIZE, Image.Resampling.BILINEAR))


def _resized_mask(mask_path: str | float | None) -> np.ndarray:
    # A parquet round-trip through pyarrow turns None in an all-object "mask_path"
    # column into NaN (a float) for normal images that never had a mask — pd.isna
    # catches both that and an actual None uniformly.
    if pd.isna(mask_path):
        h, w = _EVAL_SIZE[1], _EVAL_SIZE[0]
        return np.zeros((h, w), dtype=np.int64)
    mask = Image.open(mask_path).convert("L").resize(_EVAL_SIZE, Image.Resampling.NEAREST)
    return (np.array(mask) > 127).astype(np.int64)


def compute_metrics(df: pd.DataFrame, maps_by_id: dict) -> dict:
    from anomalib.metrics import AUPR, AUPRO, AUROC, F1Max

    pred_score = torch.tensor(df["image_score"].to_numpy(), dtype=torch.float32)
    gt_label = torch.tensor(df["label_index"].to_numpy(), dtype=torch.int64)
    image_ns = types.SimpleNamespace(pred_score=pred_score, gt_label=gt_label)

    image_auroc = AUROC(fields=["pred_score", "gt_label"])
    image_auroc.update(image_ns)
    image_aupr = AUPR(fields=["pred_score", "gt_label"])
    image_aupr.update(image_ns)
    image_f1max = F1Max(fields=["pred_score", "gt_label"])
    image_f1max.update(image_ns)

    pred_maps = np.stack([_resized_map(maps_by_id, iid) for iid in df["image_id"]])
    gt_masks = np.stack([_resized_mask(mp) for mp in df["mask_path"]])
    pixel_ns = types.SimpleNamespace(
        anomaly_map=torch.tensor(pred_maps, dtype=torch.float32),
        gt_mask=torch.tensor(gt_masks, dtype=torch.int64),
    )
    pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"])
    pixel_auroc.update(pixel_ns)
    aupro = AUPRO(fields=["anomaly_map", "gt_mask"])
    aupro.update(pixel_ns)

    return {
        "image_auroc": float(image_auroc.compute()),
        "image_aupr": float(image_aupr.compute()),
        "image_f1max": float(image_f1max.compute()),
        "pixel_auroc": float(pixel_auroc.compute()),
        "aupro": float(aupro.compute()),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-flavored markdown table.

    Avoids adding a `tabulate` dependency just for `DataFrame.to_markdown()`.
    """
    header = ["category", *[str(c) for c in df.columns]]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for idx, row in df.iterrows():
        cells = [str(idx), *[("" if pd.isna(v) else str(v)) for v in row]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


_METHOD_COLORS = {
    "patchcore": "#4c72b0",
    "anomalydino_v2": "#dd8452",
    "anomalydino_v3": "#55a868",
    "anomalydino_v3_smooth9": "#c44e52",
}


def plot_metric_bars(metrics_df: pd.DataFrame, metric: str) -> None:
    categories = sorted(metrics_df["category"].unique())
    methods = [m for m in METHODS if m in metrics_df["method"].unique()]
    x = np.arange(len(categories))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(1.6 * len(categories) + 2, 5))
    for i, method in enumerate(methods):
        sub = metrics_df[metrics_df["method"] == method].set_index("category")[metric]
        values = [sub.get(cat, np.nan) for cat in categories]
        ax.bar(x + i * width, values, width, label=method, color=_METHOD_COLORS.get(method))

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{metric} by category and method")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_ROOT / f"metric_bars_{metric}.png", dpi=150)
    plt.close(fig)


def _overlay(image: Image.Image, anomaly_map: np.ndarray) -> np.ndarray:
    """Resize *image* to _EVAL_SIZE and alpha-blend a red heatmap over it."""
    base = np.array(image.convert("RGB").resize(_EVAL_SIZE)).astype(np.float32) / 255.0
    m = anomaly_map.astype(np.float32)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    heat = plt.get_cmap("inferno")(m)[..., :3]
    return np.clip(0.55 * base + 0.45 * heat, 0, 1)


def plot_best_worst(
    category: str, method: str, df: pd.DataFrame, maps_by_id: dict, n: int = 3
) -> None:
    anomalous = df[df["label_index"] == 1].sort_values("image_score")
    if len(anomalous) == 0:
        return
    worst = anomalous.head(n)  # missed: low score despite being anomalous
    best = anomalous.tail(n)  # correctly flagged: high score

    rows = [("worst (missed)", worst), ("best (detected)", best)]
    ncols = max(len(worst), len(best), 1)
    fig, axes = plt.subplots(2, ncols, figsize=(3 * ncols, 6.5))
    axes = np.atleast_2d(axes)
    for r, (title, subset) in enumerate(rows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(subset):
                continue
            row = subset.iloc[c]
            image = Image.open(row["image_path"])
            overlay = _overlay(image, _resized_map(maps_by_id, row["image_id"]))
            ax.imshow(overlay)
            ax.set_title(f"{row['label_name']}\nscore={row['image_score']:.3f}", fontsize=9)
        axes[r, 0].set_ylabel(title, fontsize=10)
    fig.suptitle(f"{category} / {method}: best vs. worst detections")
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / f"best_worst_{category}_{method}.png", dpi=150)
    plt.close(fig)


def plot_side_by_side(category: str, cached: dict[str, tuple], n_samples: int = 4) -> None:
    """One row per sampled test image; columns = GT mask + every method's heatmap."""
    methods = [m for m in METHODS if m in cached]
    if not methods:
        return
    ref_df = cached[methods[0]][0]
    anomalous_ids = ref_df.loc[ref_df["label_index"] == 1, "image_id"]
    if len(anomalous_ids) == 0:
        return
    sample_ids = list(anomalous_ids.head(n_samples))

    ncols = 1 + len(methods)
    fig, axes = plt.subplots(len(sample_ids), ncols, figsize=(3 * ncols, 3 * len(sample_ids)))
    axes = np.atleast_2d(axes)

    for r, image_id in enumerate(sample_ids):
        row0 = ref_df[ref_df["image_id"] == image_id].iloc[0]
        image = Image.open(row0["image_path"])
        mask = _resized_mask(row0["mask_path"])

        ax = axes[r, 0]
        ax.imshow(np.array(image.convert("RGB").resize(_EVAL_SIZE)))
        ax.imshow(mask, cmap="Reds", alpha=0.4 * (mask > 0))
        ax.set_ylabel(image_id, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            ax.set_title("image + GT mask", fontsize=9)

        for c, method in enumerate(methods, start=1):
            df, maps_by_id, _ = cached[method]
            match = df[df["image_id"] == image_id]
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if match.empty:
                ax.axis("off")
                continue
            row = match.iloc[0]
            overlay = _overlay(image, _resized_map(maps_by_id, image_id))
            ax.imshow(overlay)
            if r == 0:
                ax.set_title(method, fontsize=9)
            ax.set_xlabel(f"{row['image_score']:.3f}", fontsize=8)

    fig.suptitle(f"{category}: side-by-side method comparison")
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / f"side_by_side_{category}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=CATEGORIES, choices=CATEGORIES)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    for category in args.categories:
        cached: dict[str, tuple] = {}
        for method in args.methods:
            loaded = _load_cached(category, method)
            if loaded is None:
                log.warning("[%s/%s] no cache found, skipping", category, method)
                continue
            df, maps_by_id, meta = loaded
            cached[method] = (df, maps_by_id, meta)

            log.info("[%s/%s] computing metrics", category, method)
            metrics = compute_metrics(df, maps_by_id)
            metric_rows.append(
                {
                    "category": category,
                    "method": method,
                    **metrics,
                    "fit_time_s": meta["fit_time_s"],
                    "mean_latency_ms": float(df["latency_ms"].mean()),
                    "n_test": len(df),
                }
            )

            plot_best_worst(category, method, df, maps_by_id)

        if cached:
            plot_side_by_side(category, cached)

    metrics_df = pd.DataFrame(metric_rows)
    if metrics_df.empty:
        log.error("No cached results found under %s — run run_experiments.py first.", cache_dir)
        return

    metrics_df.to_csv(RESULTS_ROOT / "metrics.csv", index=False)
    log.info("Wrote %s", RESULTS_ROOT / "metrics.csv")

    for metric in ("image_auroc", "pixel_auroc", "aupro", "image_f1max"):
        plot_metric_bars(metrics_df, metric)

    summary_lines = ["# Anomaly detection method comparison\n"]
    for metric in (
        "image_auroc",
        "pixel_auroc",
        "aupro",
        "image_f1max",
        "fit_time_s",
        "mean_latency_ms",
    ):
        pivot = metrics_df.pivot(index="category", columns="method", values=metric)
        pivot = pivot.reindex(columns=[m for m in METHODS if m in pivot.columns])
        summary_lines.append(f"\n## {metric}\n")
        summary_lines.append(_df_to_markdown(pivot.round(4)))
    (RESULTS_ROOT / "summary.md").write_text("\n".join(summary_lines) + "\n")
    log.info("Wrote %s", RESULTS_ROOT / "summary.md")
    log.info("Figures written to %s", FIGURES_ROOT)


if __name__ == "__main__":
    main()
