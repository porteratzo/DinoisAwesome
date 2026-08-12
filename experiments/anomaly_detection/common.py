"""Shared paths, dataset resolution, and cache-path helpers for the anomaly-detection
method comparison (see the plan this experiment implements for context)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

DATA_ROOT = REPO_ROOT / "data" / "mvtec_ad"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "anomaly_detection"
CACHE_ROOT = OUTPUT_ROOT / "cache"
RESULTS_ROOT = OUTPUT_ROOT / "results"
FIGURES_ROOT = RESULTS_ROOT / "figures"

# Curated MVTec AD subset: mixes texture (carpet) and object categories, and
# mask-relevant (capsule, hazelnut — masking is expected to help) with
# mask-irrelevant (bottle, cable, screw) cases.
CATEGORIES: list[str] = ["bottle", "cable", "capsule", "carpet", "hazelnut", "screw"]

# Canonical method order — reused by both scripts so cache layout and plot
# ordering always agree.
METHODS: list[str] = ["patchcore", "anomalydino_v2", "anomalydino_v3", "anomalydino_v3_smooth9"]


@dataclass
class ScoreRecord:
    """One row of `scores.parquet` — one test image's result for one method."""

    image_id: str
    split: str  # always "test" (only test images are scored)
    label_index: int  # 0 = normal, 1 = anomalous
    label_name: str  # MVTec defect-type subfolder, e.g. "good" / "broken_large"
    image_score: float
    latency_ms: float
    image_path: str
    mask_path: str | None  # None for normal images


# ---------------------------------------------------------------------------
# Dataset resolution (anomalib vendor code — called, not modified)
# ---------------------------------------------------------------------------


def ensure_mvtec_category(category: str) -> None:
    """Download MVTec AD into DATA_ROOT if not already present (no-op otherwise).

    The upstream archive bundles all 15 categories together, so this fetches
    the full dataset on first call regardless of which category is requested.
    """
    from anomalib.data import MVTecAD

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    datamodule = MVTecAD(root=DATA_ROOT, category=category)
    datamodule.prepare_data()


def load_samples(category: str) -> pd.DataFrame:
    """Return the raw MVTec AD sample table for *category* (train + test rows).

    Columns (from `anomalib.data.datasets.image.mvtecad.MVTecADDataset`):
    path, split, label, image_path, label_index, mask_path.
    """
    from anomalib.data.datasets.image.mvtecad import MVTecADDataset

    ensure_mvtec_category(category)
    dataset = MVTecADDataset(root=DATA_ROOT, category=category, split=None)
    return dataset.samples


def image_id_for(label: str, image_path: str) -> str:
    """Filesystem-safe unique ID within one category: defect-subfolder + stem.

    Needed because MVTec re-uses filenames like "000.png" across the "good"
    and each defect-type test subfolder.
    """
    return f"{label}_{Path(image_path).stem}"


def resolve_paths(category: str, limit: int | None = None) -> tuple[list[Path], pd.DataFrame]:
    """Return (train image paths, test sample rows) for *category*.

    Args:
        category: MVTec AD category name.
        limit: If given, caps the test set to roughly `limit` images — split
            evenly between normal and anomalous so a small smoke-test run
            still exercises both classes — and caps the train set to
            `max(limit, 5)` images so fitting stays fast during smoke tests.
    """
    samples = load_samples(category)
    train_paths = [Path(p) for p in samples.loc[samples.split == "train", "image_path"]]
    test_df = samples.loc[samples.split == "test"].reset_index(drop=True)

    if limit is not None:
        normal = test_df[test_df.label_index == 0]
        anomalous = test_df[test_df.label_index == 1]
        n_normal = max(1, limit // 2)
        n_anomalous = max(1, limit - n_normal)
        test_df = pd.concat([normal.head(n_normal), anomalous.head(n_anomalous)]).reset_index(
            drop=True
        )
        train_paths = train_paths[: max(limit, 5)]

    logger.info(
        "%s: %d train images, %d test images (%d normal, %d anomalous)",
        category,
        len(train_paths),
        len(test_df),
        int((test_df.label_index == 0).sum()),
        int((test_df.label_index == 1).sum()),
    )
    return train_paths, test_df


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------


def cache_dir(category: str, method: str) -> Path:
    d = CACHE_ROOT / category / method
    d.mkdir(parents=True, exist_ok=True)
    return d


def scores_path(category: str, method: str) -> Path:
    return cache_dir(category, method) / "scores.parquet"


def anomaly_maps_path(category: str, method: str) -> Path:
    return cache_dir(category, method) / "anomaly_maps.npz"


def gallery_dir(category: str, method: str) -> Path:
    return cache_dir(category, method) / "gallery"


def has_cache(category: str, method: str) -> bool:
    return scores_path(category, method).exists() and anomaly_maps_path(category, method).exists()
