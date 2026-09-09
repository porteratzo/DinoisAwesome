"""Resolution x backbone-size ablation for the DINOv3-backed anomaly methods.

`run_experiments.py` hardcodes `img_size=256, size="small"` for `anomalydino_v3` and every
`dinov3_proto_*` variant. `object_detection/resolution_ablation/` and
`fundamental/resolution_ablation.py` both found resolution/backbone-size effects worth
measuring for detection and localization; this is the anomaly-detection analogue, answering
whether a bigger/higher-resolution DINOv3 backbone actually buys better AUROC/AUPRO per
category, or whether 256px/small already saturates it (in which case the extra compute the
main benchmark already pays for those methods is wasted).

Constructs `AnomalyDINOv3Method`/`PrototypeMethod` directly (bypassing `build_method`'s
name-based dispatch, which has no resolution/size axis) under a synthetic category name
(`<category>__r<resolution>_<size>`) so its Gallery cache never collides with
`run_experiments.py`'s own `anomalydino_v3`/`dinov3_proto_*` cache at the real category path.

**Cost**: every (category, resolution, size, method) point re-encodes the full train set
from scratch (a different `img_size`/`size` means a genuinely different encoder, not just a
different scoring pass) -- unlike `layer_ablation.py`, there's no shared encoding to reuse
across points. `base`/`large` and >512px are opt-in for that reason; each point is wrapped in
its own CUDA-OOM guard (mirroring `fundamental/resolution_ablation.py`'s own guard) so one
bad combination doesn't abort a long sweep.

Usage:
    python resolution_size_ablation.py --categories bottle carpet --resolutions 256 512
    python resolution_size_ablation.py --sizes small base --methods anomalydino_v3
"""

# Logging — must be before torch import
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("resolution_size_ablation")

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_results import compute_metrics
from common import CATEGORIES, RESULTS_ROOT, image_id_for, resolve_paths
from methods import AnomalyDINOv3Method, PrototypeMethod
from tqdm import tqdm

_OUT_DIR = RESULTS_ROOT / "resolution_size_ablation"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=CATEGORIES, choices=CATEGORIES)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["anomalydino_v3", "dinov3_proto"],
        choices=["anomalydino_v3", "dinov3_proto"],
    )
    parser.add_argument("--resolutions", nargs="+", type=int, default=[256, 512])
    parser.add_argument(
        "--sizes", nargs="+", default=["small"], choices=["small", "base", "large", "giant"]
    )
    parser.add_argument("--limit", type=int, default=30, help="Cap test set size")
    parser.add_argument("--train-limit", type=int, default=None)
    return parser.parse_args()


def _build_method(
    method: str, category: str, resolution: int, size: str
) -> AnomalyDINOv3Method | PrototypeMethod:
    if method == "anomalydino_v3":
        return AnomalyDINOv3Method(category=category, img_size=resolution, size=size)
    return PrototypeMethod(
        category=category,
        n_prototypes=1,
        retrieval_k=1,
        aggregation="max",
        masking=False,
        img_size=resolution,
        size=size,
    )


def _run_one(
    method_name: str,
    category: str,
    resolution: int,
    size: str,
    train_paths: list[Path],
    test_df: pd.DataFrame,
) -> dict:
    synthetic_category = f"{category}__r{resolution}_{size}"
    method = _build_method(method_name, synthetic_category, resolution, size)
    fit_time_s = method.fit(train_paths)

    records: list[dict] = []
    maps: dict[str, np.ndarray] = {}
    for _, row in test_df.iterrows():
        image_id = image_id_for(row["label"], row["image_path"])
        result = method.predict(row["image_path"])
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
    return {
        "category": category,
        "method": method_name,
        "resolution": resolution,
        "size": size,
        "fit_time_s": fit_time_s,
        **metrics,
    }


def main() -> None:
    args = _parse_args()
    rows: list[dict] = []

    for category in tqdm(args.categories, desc="categories"):
        train_paths, test_df = resolve_paths(
            category, limit=args.limit, train_limit=args.train_limit
        )

        for method_name in args.methods:
            for size in args.sizes:
                for resolution in args.resolutions:
                    label = f"{category}/{method_name}/{size}/{resolution}px"
                    try:
                        rows.append(
                            _run_one(method_name, category, resolution, size, train_paths, test_df)
                        )
                        log.info("[%s] done", label)
                    except torch.cuda.OutOfMemoryError:
                        log.error("[%s] CUDA OOM, skipping", label)
                    except Exception:
                        log.exception("[%s] FAILED, skipping", label)
                    finally:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    if not rows:
        log.error("No results produced.")
        return

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(rows)
    results_df.to_csv(_OUT_DIR / "metrics.csv", index=False)
    log.info("Wrote %s", _OUT_DIR / "metrics.csv")

    _plot(results_df)


def _plot(results_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    figures_dir = _OUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for method_name in results_df["method"].unique():
        sub_method = results_df[results_df["method"] == method_name]
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for (category, size), sub in sub_method.groupby(["category", "size"]):
            sub = sub.sort_values("resolution")
            ax.plot(sub["resolution"], sub["image_auroc"], marker="o", label=f"{category}/{size}")
        ax.set_xlabel("resolution (px)")
        ax.set_ylabel("image AUROC")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{method_name}: AUROC vs. resolution/size")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures_dir / f"resolution_size_{method_name}.png", dpi=150)
        plt.close(fig)
    log.info("Figures written to %s", figures_dir)


if __name__ == "__main__":
    main()
