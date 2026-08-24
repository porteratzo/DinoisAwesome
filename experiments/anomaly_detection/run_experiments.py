"""Stage 1: fit + score every (category, method) pair and cache everything.

Usage:
    python run_experiments.py                                  # full run (VAD auto-capped)
    python run_experiments.py --categories bottle --limit 10   # smoke test
    python run_experiments.py --categories vad --limit 2000 --train-limit 2000  # full VAD
    python run_experiments.py --methods patchcore anomalydino_v3
    python run_experiments.py --force                          # overwrite existing cache

VAD's train/good split (2000 images) is slow to fit, so `common.DEFAULT_LIMITS` caps it by
default to 200 train / 200 normal + 200 anomalous test unless --limit/--train-limit override it.

Writes, per (category, method), under outputs/anomaly_detection/cache/<category>/<method>/:
    scores.parquet      one row per test image (common.ScoreRecord)
    anomaly_maps.npz    stacked (N, H, W) float16 anomaly maps + image_id array
    meta.json           fit_time_s and run hyperparameters
    gallery/            (methods 3 & 4 only) dinoisawesome.Gallery — persisted patch
                         embeddings for train + test images
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
import gc
import json
import time
import traceback

import numpy as np
import pandas as pd
import torch
from common import (
    ALL_METHODS,
    CATEGORIES,
    METHODS,
    anomaly_maps_path,
    cache_dir,
    image_id_for,
    resolve_paths,
    scores_path,
)
from methods import ScoreResult, build_method


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=CATEGORIES, choices=CATEGORIES)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap test set size for a fast smoke test"
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Cap train set size independently of --limit",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache")
    return parser.parse_args()


def _run_one(
    category: str, method_name: str, limit: int | None, train_limit: int | None, force: bool
) -> None:
    out_dir = cache_dir(category, method_name)
    if not force and scores_path(category, method_name).exists():
        log.info("[%s/%s] cache exists, skipping (use --force to overwrite)", category, method_name)
        return

    log.info("[%s/%s] resolving dataset paths", category, method_name)
    train_paths, test_df = resolve_paths(category, limit=limit, train_limit=train_limit)

    method = build_method(method_name, category)
    log.info("[%s/%s] fitting on %d normal images", category, method_name, len(train_paths))
    fit_time_s = method.fit(train_paths)
    log.info("[%s/%s] fit done in %.1fs", category, method_name, fit_time_s)

    records: list[dict] = []
    maps: list[np.ndarray] = []
    image_ids: list[str] = []

    for i, row in test_df.iterrows():
        image_path = row["image_path"]
        image_id = image_id_for(row["label"], image_path)
        result: ScoreResult = method.predict(image_path)

        records.append(
            {
                "image_id": image_id,
                "split": "test",
                "label_index": int(row["label_index"]),
                "label_name": row["label"],
                "image_score": result.score,
                "latency_ms": result.latency_ms,
                "image_path": str(image_path),
                "mask_path": str(row["mask_path"]) if row["mask_path"] else None,
            }
        )
        image_ids.append(image_id)
        maps.append(result.anomaly_map.astype(np.float16))

        if (i + 1) % 25 == 0 or (i + 1) == len(test_df):
            log.info("[%s/%s] scored %d/%d", category, method_name, i + 1, len(test_df))

    pd.DataFrame.from_records(records).to_parquet(scores_path(category, method_name), index=False)
    np.savez_compressed(
        anomaly_maps_path(category, method_name),
        image_id=np.array(image_ids),
        maps=np.stack(maps),
    )
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "category": category,
                "method": method_name,
                "fit_time_s": fit_time_s,
                "n_train": len(train_paths),
                "n_test": len(test_df),
                "limit": limit,
                "train_limit": train_limit,
            },
            indent=2,
        )
    )
    log.info("[%s/%s] cache written to %s", category, method_name, out_dir)

    del method
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = _parse_args()
    t_start = time.time()
    failures: list[tuple[str, str, str]] = []

    for category in args.categories:
        for method_name in args.methods:
            try:
                _run_one(category, method_name, args.limit, args.train_limit, args.force)
            except Exception as exc:  # noqa: BLE001 - keep the batch going, report at the end
                log.error("[%s/%s] FAILED: %s", category, method_name, exc)
                traceback.print_exc()
                failures.append((category, method_name, str(exc)))

    elapsed = time.time() - t_start
    log.info("Done in %.1fs. %d failure(s).", elapsed, len(failures))
    for category, method_name, msg in failures:
        log.error("  %s/%s: %s", category, method_name, msg)


if __name__ == "__main__":
    main()
