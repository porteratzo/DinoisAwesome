"""Training-set-size ablation: how many normal images does each method actually need
before AUROC saturates?

`run_experiments.py` always fits on the category's full "good" train split. This script
answers the question that leaves open: is that necessary, or does e.g. PatchCore/AnomalyDINO
saturate at 10-25 normal images for a given category, while another method keeps improving?
Answering this matters for real deployment (fewer reference images to collect/label) and for
telling "this method needs more data" apart from "this method has hit its representational
ceiling" when comparing AUROC numbers in `analyze_results.py`'s summary.

Reuses `methods.build_method` and `common.resolve_paths` completely unmodified — every
train-size point is just `build_method(name, category).fit(shuffled_train_paths[:n])` followed
by `.predict()` on the same fixed held-out test set, so results are directly comparable to
`run_experiments.py`'s own cache. Train-set subsampling takes a *prefix* of one fixed random
shuffle per category (not an independent resample per size) so smaller sizes are proper subsets
of larger ones -- a size effect can't be confounded by which specific images happened to be
drawn at each point.

DINOv3-backed methods (`anomalydino_v3`, `dinov3_proto_*`) build a Gallery on disk keyed by
category name; to avoid clobbering `run_experiments.py`'s own full-train-set cache at
`outputs/anomaly_detection/cache/<category>/<method>/`, every point here is fit under a
synthetic category name (`<category>__ts<n>`) so its gallery lands in a disjoint cache
directory. `common.cache_dir` never validates its `category` argument against `CATEGORIES`, so
this is a safe, zero-modification reuse of the existing cache-path helpers.

Not cached across reruns the way `run_experiments.py` is (no `scores.parquet` skip-if-exists
check) -- this is a one-off sweep, run it when you need the curve, not incrementally.

Usage:
    python train_size_ablation.py --categories bottle carpet --limit 40
    python train_size_ablation.py --methods patchcore anomalydino_v3 --train-sizes 5 10 25 all
"""

# Logging — must be before torch import
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("train_size_ablation")

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_results import compute_metrics
from common import ALL_METHODS, CATEGORIES, RESULTS_ROOT, image_id_for, resolve_paths
from methods import ScoreResult, build_method
from tqdm import tqdm

_DEFAULT_METHODS: list[str] = [
    "patchcore",
    "anomalydino_v2",
    "anomalydino_v3",
    "dinov3_proto_k1_n1_max",
]
_DEFAULT_TRAIN_SIZES: list[str] = ["5", "10", "25", "all"]
_OUT_DIR = RESULTS_ROOT / "train_size_ablation"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=CATEGORIES, choices=CATEGORIES)
    parser.add_argument("--methods", nargs="+", default=_DEFAULT_METHODS, choices=ALL_METHODS)
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        default=_DEFAULT_TRAIN_SIZES,
        help="Normal-image counts to sweep, or the literal 'all' for the full train split.",
    )
    parser.add_argument(
        "--limit", type=int, default=40, help="Cap the (fixed, shared-across-sizes) test set."
    )
    parser.add_argument("--seed", type=int, default=0, help="Train-shuffle seed.")
    return parser.parse_args()


def _resolve_sizes(train_sizes: list[str], n_available: int) -> list[int]:
    sizes = sorted({n_available if s == "all" else int(s) for s in train_sizes})
    return [n for n in sizes if n <= n_available]


def _run_one(
    category: str, method_name: str, train_paths: list[Path], test_df: pd.DataFrame, n_train: int
) -> dict:
    synthetic_category = f"{category}__ts{n_train}"
    method = build_method(method_name, synthetic_category)
    fit_time_s = method.fit(train_paths[:n_train])

    records: list[dict] = []
    maps: dict[str, np.ndarray] = {}
    for _, row in test_df.iterrows():
        image_id = image_id_for(row["label"], row["image_path"])
        result: ScoreResult = method.predict(row["image_path"])
        records.append(
            {
                "image_id": image_id,
                "label_index": int(row["label_index"]),
                "image_score": result.score,
                "mask_path": row["mask_path"],
            }
        )
        maps[image_id] = result.anomaly_map.astype(np.float16)

    df = pd.DataFrame.from_records(records)
    metrics = compute_metrics(df, maps)

    del method
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "category": category,
        "method": method_name,
        "n_train": n_train,
        "fit_time_s": fit_time_s,
        **metrics,
    }


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []

    for category in tqdm(args.categories, desc="categories"):
        # train_limit is passed explicitly (rather than left None) so resolve_paths' own
        # smoke-test behaviour -- capping the train pool to max(limit, 5) whenever train_limit
        # is left unset -- doesn't quietly make "all" mean "capped to the test-set limit"
        # instead of the category's true full training pool.
        train_paths, test_df = resolve_paths(category, limit=args.limit, train_limit=1_000_000)
        shuffled = list(train_paths)
        rng.shuffle(shuffled)
        sizes = _resolve_sizes(args.train_sizes, len(shuffled))
        if not sizes:
            log.warning(
                "[%s] no valid train sizes (only %d train images available)",
                category,
                len(shuffled),
            )
            continue

        for method_name in tqdm(args.methods, desc=f"{category} methods", leave=False):
            for n_train in tqdm(sizes, desc=f"{category}/{method_name} sizes", leave=False):
                try:
                    rows.append(_run_one(category, method_name, shuffled, test_df, n_train))
                except Exception:
                    log.exception("[%s/%s/n=%d] FAILED, skipping", category, method_name, n_train)

    if not rows:
        log.error("No results produced.")
        return

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(rows)
    results_df.to_csv(_OUT_DIR / "metrics.csv", index=False)
    log.info("Wrote %s", _OUT_DIR / "metrics.csv")

    _plot_learning_curves(results_df)


def _plot_learning_curves(results_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    figures_dir = _OUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for category in results_df["category"].unique():
        sub = results_df[results_df["category"] == category]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for method_name in sub["method"].unique():
            m = sub[sub["method"] == method_name].sort_values("n_train")
            ax.plot(m["n_train"], m["image_auroc"], marker="o", label=method_name)
        ax.set_xlabel("normal training images")
        ax.set_ylabel("image AUROC")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{category}: AUROC vs. training-set size")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / f"learning_curve_{category}.png", dpi=150)
        plt.close(fig)
    log.info("Figures written to %s", figures_dir)


if __name__ == "__main__":
    main()
