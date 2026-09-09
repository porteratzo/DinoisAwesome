"""Localization quality: oracle-IoU/Otsu-IoU bounds and a defect-size/contrast error
breakdown, computed from `run_experiments.py`'s existing cache.

`analyze_results.py` reports AUROC/AUPR/F1Max/AUPRO per (category, method) and a 3-image
best/worst eyeball check -- neither tells you *why* a method's pixel-level score is what it
is. This script adds two things `_shared/thresholding.py` (Otsu, oracle-IoU) already
provides but `anomaly_detection/` never uses:

1. **Oracle vs. Otsu IoU**: `oracle_iou` is the best patch-mask IoU any single global
   threshold on the anomaly map could achieve against the GT mask (a representation-quality
   ceiling); `otsu_iou` applies the same map's own Otsu threshold (no GT peeking, what a real
   deployment would actually use). The gap between them separates "the representation caps
   out here" from "the threshold/aggregation choice is leaving IoU on the table" -- the same
   distinction `fundamental/`'s oracle-IoU experiments draw for detection.
2. **Error breakdown by defect size/contrast**: buckets every anomalous test image into
   size/contrast terciles (computed once per (category, image_id) from its own GT mask/image,
   shared across every method being compared) and reports mean oracle/Otsu IoU per bucket,
   plus each bucket's own image-level AUROC (that bucket's anomalous images + every normal
   image) -- replacing `plot_best_worst`'s 3-image spot-check with an aggregate answer to "does
   this method's detection/localization degrade on small or low-contrast defects?"

Reads-only against `outputs/anomaly_detection/cache/` -- run `run_experiments.py` first for
whatever (category, method) pairs you want analyzed here.

Usage:
    python localization_analysis.py
    python localization_analysis.py --categories bottle carpet --methods patchcore anomalydino_v3
"""

# Logging — must be before torch import
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("localization_analysis")

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_results import _load_cached, _resized_map, _resized_mask
from common import ALL_METHODS, CATEGORIES, RESULTS_ROOT
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared.mask_geometry import mask_iou  # noqa: E402
from _shared.thresholding import oracle_iou, otsu_threshold  # noqa: E402

_OUT_DIR = RESULTS_ROOT / "localization_analysis"
_ORACLE_STEPS = 50
_SIZE_LABELS = ("small", "medium", "large")
_CONTRAST_LABELS = ("low", "medium", "high")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=CATEGORIES, choices=CATEGORIES)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    return parser.parse_args()


def _tercile_bucket(value: float, edges: np.ndarray, labels: tuple[str, str, str]) -> str:
    if value < edges[0]:
        return labels[0]
    if value < edges[1]:
        return labels[1]
    return labels[2]


def _difficulty_covariates(ref_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """{image_id: (defect_area_frac, contrast)} for every anomalous row in *ref_df*.

    Computed once per image (shared GT mask/image across every method for that category),
    not per (image, method) -- these are properties of the dataset, not of any one method's
    scores.
    """
    out: dict[str, tuple[float, float]] = {}
    for _, row in ref_df[ref_df["label_index"] == 1].iterrows():
        mask = _resized_mask(row["mask_path"]).astype(bool)
        if not mask.any() or mask.all():
            continue
        gray = np.array(Image.open(row["image_path"]).convert("L").resize(mask.shape[::-1]))
        area_frac = float(mask.sum()) / mask.size
        contrast = abs(float(gray[mask].mean()) - float(gray[~mask].mean())) / 255.0
        out[row["image_id"]] = (area_frac, contrast)
    return out


def _bucketed_auroc(df: pd.DataFrame, bucket_image_ids: set[str]) -> float:
    """Image AUROC restricted to *bucket_image_ids* (anomalous) plus every normal image in df."""
    from anomalib.metrics import AUROC

    keep = df[(df["label_index"] == 0) | (df["image_id"].isin(bucket_image_ids))]
    if keep["label_index"].nunique() < 2:
        return float("nan")
    ns = types.SimpleNamespace(
        pred_score=torch.tensor(keep["image_score"].to_numpy(), dtype=torch.float32),
        gt_label=torch.tensor(keep["label_index"].to_numpy(), dtype=torch.int64),
    )
    metric = AUROC(fields=["pred_score", "gt_label"])
    metric.update(ns)
    return float(metric.compute())


def main() -> None:
    args = _parse_args()
    per_image_rows: list[dict] = []
    bucket_rows: list[dict] = []

    for category in tqdm(args.categories, desc="categories"):
        cached: dict[str, tuple] = {}
        for method in args.methods:
            loaded = _load_cached(category, method)
            if loaded is not None:
                cached[method] = loaded
        if not cached:
            continue

        ref_df = next(iter(cached.values()))[0]
        difficulty = _difficulty_covariates(ref_df)
        if not difficulty:
            log.warning("[%s] no usable GT masks, skipping", category)
            continue

        areas = np.array([v[0] for v in difficulty.values()])
        contrasts = np.array([v[1] for v in difficulty.values()])
        area_edges = np.quantile(areas, [1 / 3, 2 / 3])
        contrast_edges = np.quantile(contrasts, [1 / 3, 2 / 3])

        for method, (df, maps_by_id, _meta) in cached.items():
            anomalous = df[(df["label_index"] == 1) & (df["image_id"].isin(difficulty))]
            method_rows: list[dict] = []
            for _, row in anomalous.iterrows():
                area_frac, contrast = difficulty[row["image_id"]]
                raw_map = _resized_map(maps_by_id, row["image_id"])
                mask = _resized_mask(row["mask_path"]).astype(bool)
                o_iou = oracle_iou(raw_map, mask, steps=_ORACLE_STEPS)
                ot_iou = mask_iou(raw_map > otsu_threshold(raw_map), mask)
                method_rows.append(
                    {
                        "category": category,
                        "method": method,
                        "image_id": row["image_id"],
                        "defect_area_frac": area_frac,
                        "contrast": contrast,
                        "oracle_iou": o_iou,
                        "otsu_iou": ot_iou,
                        "size_bucket": _tercile_bucket(area_frac, area_edges, _SIZE_LABELS),
                        "contrast_bucket": _tercile_bucket(
                            contrast, contrast_edges, _CONTRAST_LABELS
                        ),
                    }
                )
            per_image_rows.extend(method_rows)
            if not method_rows:
                continue

            image_df = pd.DataFrame(method_rows)
            axes = (("size_bucket", _SIZE_LABELS), ("contrast_bucket", _CONTRAST_LABELS))
            for axis, labels in axes:
                for label in labels:
                    bucket_ids = set(image_df.loc[image_df[axis] == label, "image_id"])
                    if not bucket_ids:
                        continue
                    bucket_rows.append(
                        {
                            "category": category,
                            "method": method,
                            "axis": axis,
                            "bucket": label,
                            "n_images": len(bucket_ids),
                            "mean_oracle_iou": image_df.loc[
                                image_df[axis] == label, "oracle_iou"
                            ].mean(),
                            "mean_otsu_iou": image_df.loc[
                                image_df[axis] == label, "otsu_iou"
                            ].mean(),
                            "bucketed_auroc": _bucketed_auroc(df, bucket_ids),
                        }
                    )

    if not per_image_rows:
        log.error("No cached results found -- run run_experiments.py first.")
        return

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_image_df = pd.DataFrame(per_image_rows)
    bucket_df = pd.DataFrame(bucket_rows)
    per_image_df.to_csv(_OUT_DIR / "per_image.csv", index=False)
    bucket_df.to_csv(_OUT_DIR / "buckets.csv", index=False)
    log.info("Wrote %s and %s", _OUT_DIR / "per_image.csv", _OUT_DIR / "buckets.csv")

    _write_summary(per_image_df, bucket_df)
    _plot(bucket_df)


def _flat_df_to_markdown(df: pd.DataFrame) -> str:
    """Render a flat (no meaningful index) DataFrame as a GitHub-flavored markdown table.

    Avoids adding a `tabulate` dependency just for `DataFrame.to_markdown()` -- same
    reasoning as `analyze_results._df_to_markdown`, generalized to a table with no
    meaningful index column (that helper always prepends one named "category").
    """
    header = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for _, row in df.iterrows():
        cells = ["" if pd.isna(v) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_summary(per_image_df: pd.DataFrame, bucket_df: pd.DataFrame) -> None:
    lines = ["# Localization quality analysis\n", "\n## Mean oracle IoU vs. Otsu IoU by method\n"]
    overall = per_image_df.groupby(["category", "method"])[["oracle_iou", "otsu_iou"]].mean()
    overall["gap"] = overall["oracle_iou"] - overall["otsu_iou"]
    lines.append(_flat_df_to_markdown(overall.round(4).reset_index()))
    lines.append("\n\n## AUROC by defect-size / contrast tercile\n")
    lines.append(_flat_df_to_markdown(bucket_df.round(4)))
    (_OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    log.info("Wrote %s", _OUT_DIR / "summary.md")


def _plot(bucket_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    figures_dir = _OUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for axis, labels in (("size_bucket", _SIZE_LABELS), ("contrast_bucket", _CONTRAST_LABELS)):
        sub_axis = bucket_df[bucket_df["axis"] == axis]
        for category in sub_axis["category"].unique():
            sub = sub_axis[sub_axis["category"] == category]
            methods = sorted(sub["method"].unique())
            x = np.arange(len(labels))
            width = 0.8 / max(len(methods), 1)
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            for i, method in enumerate(methods):
                m = sub[sub["method"] == method].set_index("bucket")
                values = [m["bucketed_auroc"].get(label, np.nan) for label in labels]
                ax.bar(x + i * width, values, width, label=method)
            ax.set_xticks(x + width * (len(methods) - 1) / 2)
            ax.set_xticklabels(labels)
            ax.set_ylabel("image AUROC")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{category}: AUROC by {axis}")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(figures_dir / f"{axis}_{category}.png", dpi=150)
            plt.close(fig)
    log.info("Figures written to %s", figures_dir)


if __name__ == "__main__":
    main()
