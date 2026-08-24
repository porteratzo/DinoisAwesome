"""Shared paths, dataset resolution, and cache-path helpers for the anomaly-detection
method comparison (see the plan this experiment implements for context).

Four dataset families are supported, dispatched by category name:
    - MVTec AD (`_MVTEC_CATEGORIES`) — downloaded on first use via anomalib.
    - Auto6 / AutoVI (`_AUTO6_CATEGORIES`) — local only, under `data/auto6/<category>`.
      Ground-truth masks live one level deeper than MVTec AD's own layout
      (`ground_truth/<label>/<image_stem>/*.png`, one file per defect instance), so
      they're parsed by a dedicated loader instead of anomalib's `MVTecADDataset`.
    - VAD (`_VAD_CATEGORIES`) — local only, under `data/vad`; single category "vad".
      Its on-disk layout already matches MVTec AD's convention, so it's loaded
      through the same `MVTecADDataset` class as MVTec AD, just without the
      download step.
    - MDTecAD2 / MVTec AD 2 (`_MDTECAD2_CATEGORIES`) — local only, under
      `data/mdtecad2/<category>/<category>` (double-nested, as shipped). Only
      `train/good` and `test_public/{good,bad}` are usable for scoring —
      `test_private`/`test_private_mixed` are unlabeled leaderboard holdouts and
      `validation` is normal-only, so a dedicated loader maps `test_public` to
      the `test` split anomalib's convention expects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from PIL import Image

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

DATA_ROOT = REPO_ROOT / "data" / "mvtec_ad"
AUTO6_ROOT = REPO_ROOT / "data" / "auto6"
VAD_ROOT = REPO_ROOT / "data" / "vad"
MDTECAD2_ROOT = REPO_ROOT / "data" / "mdtecad2"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "anomaly_detection"
CACHE_ROOT = OUTPUT_ROOT / "cache"
RESULTS_ROOT = OUTPUT_ROOT / "results"
FIGURES_ROOT = RESULTS_ROOT / "figures"
AUTO6_MERGED_MASKS_ROOT = OUTPUT_ROOT / "auto6_merged_masks"

# Curated MVTec AD subset: mixes texture (carpet) and object categories, and
# mask-relevant (capsule, hazelnut — masking is expected to help) with
# mask-irrelevant (bottle, cable, screw) cases.
_MVTEC_CATEGORIES: list[str] = ["bottle", "cable", "capsule", "carpet", "hazelnut", "screw"]

# All six AutoVI categories (folder names under data/auto6).
_AUTO6_CATEGORIES: list[str] = [
    "pipe_staple",
    "tank_screw",
    "pipe_clip",
    "engine_wiring",
    "underbody_pipes",
    "underbody_screw",
]

# VAD ships as a single category.
_VAD_CATEGORIES: list[str] = ["vad"]

# MDTecAD2 (MVTec AD 2) categories present under data/mdtecad2.
_MDTECAD2_CATEGORIES: list[str] = ["can", "fruit_jelly", "sheet_metal", "vial", "wallplugs"]

# Canonical category order — reused by both scripts as the default --categories list.
CATEGORIES: list[str] = [
    *_MVTEC_CATEGORIES,
    *_AUTO6_CATEGORIES,
    *_VAD_CATEGORIES,
    *_MDTECAD2_CATEGORIES,
]

# Canonical method order — reused by both scripts so cache layout and plot
# ordering always agree. This is the *default* sweep (fast); the PCA-masked
# baselines and the full prototype-method grid (see below) are opt-in via
# `--methods` since they multiply the run count considerably.
METHODS: list[str] = ["patchcore", "anomalydino_v2", "anomalydino_v3", "anomalydino_v3_smooth9"]

# PCA background-suppression variants of the two DINO memory-bank baselines
# (ported from anomalib's AnomalyDINO — see dinoisawesome.background_mask).
MASKED_METHODS: list[str] = ["anomalydino_v2_masked", "anomalydino_v3_masked"]

# ---------------------------------------------------------------------------
# Prototype method (dinoisawesome.PrototypeAnomalyHead) ablation grid:
# CLS-retrieval breadth (k) x k-means prototype count (n) x score aggregation
# x PCA background masking. Method names are generated, not hand-enumerated,
# so `common.py` stays the single source of truth for the naming scheme;
# `methods.py::build_method` parses a name back into build kwargs via
# `parse_prototype_name`.
# ---------------------------------------------------------------------------
_PROTO_RETRIEVAL_K: tuple[int, ...] = (1, 3)
_PROTO_N_PROTOTYPES: tuple[int, ...] = (1, 2, 5)
_PROTO_AGGREGATION: tuple[str, ...] = ("max", "avg")
_PROTO_MASKING: tuple[bool, ...] = (False, True)


def prototype_method_name(
    n_prototypes: int, retrieval_k: int, aggregation: str, masking: bool
) -> str:
    suffix = "_masked" if masking else ""
    return f"dinov3_proto_k{retrieval_k}_n{n_prototypes}_{aggregation}{suffix}"


def parse_prototype_name(name: str) -> dict[str, int | str | bool] | None:
    """Parse a ``dinov3_proto_k{k}_n{n}_{agg}[_masked]`` method name back into
    :class:`~dinoisawesome.PrototypeAnomalyHead` build kwargs.

    Returns ``None`` if *name* isn't a prototype-method name (so callers can
    fall through to their own explicit method dispatch).
    """
    if not name.startswith("dinov3_proto_"):
        return None
    masking = name.endswith("_masked")
    body = name[: -len("_masked")] if masking else name
    _, _, k_part, n_part, aggregation = body.split("_", 4)
    return {
        "retrieval_k": int(k_part[1:]),
        "n_prototypes": int(n_part[1:]),
        "aggregation": aggregation,
        "masking": masking,
    }


PROTOTYPE_METHODS: list[str] = [
    prototype_method_name(n, k, agg, masking)
    for k in _PROTO_RETRIEVAL_K
    for n in _PROTO_N_PROTOTYPES
    for agg in _PROTO_AGGREGATION
    for masking in _PROTO_MASKING
]

# Every method name build_method() can construct — the full `--methods` choices set.
ALL_METHODS: list[str] = [*METHODS, *MASKED_METHODS, *PROTOTYPE_METHODS]


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
# Dataset resolution (anomalib vendor code — called, not modified — for MVTec AD
# and VAD; a dedicated loader below for Auto6, whose ground-truth layout anomalib's
# MVTecADDataset can't parse)
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


def _merge_mask_instances(mask_files: list[Path], out_path: Path) -> Path:
    """Union multiple per-defect-instance mask files into one binary mask PNG.

    Auto6's "multiple" defect-type folders hold one file per defect instance
    (each flagged with its own `defects_config.json` pixel value); pixel-level
    metrics need a single combined mask per image. Cached under
    AUTO6_MERGED_MASKS_ROOT so repeat runs don't redo the merge.
    """
    if out_path.exists():
        return out_path
    merged = None
    for f in mask_files:
        arr = np.array(Image.open(f).convert("L"))
        merged = arr if merged is None else np.maximum(merged, arr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((merged > 0).astype(np.uint8) * 255).save(out_path)
    return out_path


def _load_auto6_samples(category: str) -> pd.DataFrame:
    """Parse Auto6's on-disk layout directly (train/test/ground_truth, MVTec-style,
    but with ground-truth masks nested one level deeper as
    `ground_truth/<label>/<image_stem>/*.png`, one file per defect instance).
    """
    category_root = AUTO6_ROOT / category
    records: list[dict] = []
    mask_paths: list[str | None] = []
    for split in ("train", "test"):
        for label_dir in sorted((category_root / split).iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for image_path in sorted(label_dir.glob("*.png")):
                mask_path: Path | None = None
                if label != "good":
                    mask_files = sorted(
                        (category_root / "ground_truth" / label / image_path.stem).glob("*.png")
                    )
                    if len(mask_files) == 1:
                        mask_path = mask_files[0]
                    elif len(mask_files) > 1:
                        mask_path = _merge_mask_instances(
                            mask_files,
                            AUTO6_MERGED_MASKS_ROOT / category / label / f"{image_path.stem}.png",
                        )
                records.append(
                    {
                        "path": str(category_root),
                        "split": split,
                        "label": label,
                        "image_path": str(image_path),
                        "label_index": 0 if label == "good" else 1,
                    }
                )
                mask_paths.append(str(mask_path) if mask_path else None)
    samples = pd.DataFrame.from_records(records)
    # Assign as an explicit object-dtype Series (matches anomalib's own
    # MVTecADDataset column) rather than folding it into the from_records() call:
    # pandas 3.0's default string-dtype inference otherwise silently turns this
    # mixed None/str column into its `str` extension dtype, whose missing-value
    # sentinel `.iterrows()` promotes to a float NaN — and `str(nan)` downstream
    # serializes to the literal string "nan" instead of a null.
    samples["mask_path"] = pd.Series(mask_paths, dtype=object)
    samples.attrs["task"] = "segmentation"
    return samples


def _load_mdtecad2_samples(category: str) -> pd.DataFrame:
    """Parse MDTecAD2's on-disk layout directly (double-nested `<category>/<category>`,
    `train`/`test_public` are the only usable splits, ground-truth masks named
    `ground_truth/<label>/<image_stem>_mask.png` — MVTec AD's own flat convention,
    so no per-instance merge is needed like Auto6).
    """
    category_root = MDTECAD2_ROOT / category / category
    split_dirs = {"train": category_root / "train", "test": category_root / "test_public"}
    records: list[dict] = []
    mask_paths: list[str | None] = []
    for split, split_root in split_dirs.items():
        for label_dir in sorted(split_root.iterdir()):
            if not label_dir.is_dir() or label_dir.name == "ground_truth":
                continue
            label = label_dir.name
            for image_path in sorted(label_dir.glob("*.png")):
                mask_path: Path | None = None
                if label != "good":
                    candidate = split_root / "ground_truth" / label / f"{image_path.stem}_mask.png"
                    if candidate.exists():
                        mask_path = candidate
                records.append(
                    {
                        "path": str(category_root),
                        "split": split,
                        "label": label,
                        "image_path": str(image_path),
                        "label_index": 0 if label == "good" else 1,
                    }
                )
                mask_paths.append(str(mask_path) if mask_path else None)
    samples = pd.DataFrame.from_records(records)
    # See _load_auto6_samples for why this is an explicit object-dtype Series.
    samples["mask_path"] = pd.Series(mask_paths, dtype=object)
    samples.attrs["task"] = "segmentation"
    return samples


def load_samples(category: str) -> pd.DataFrame:
    """Return the raw sample table for *category* (train + test rows), dispatched
    to the right dataset family by category name.

    Columns (from `anomalib.data.datasets.image.mvtecad.MVTecADDataset` for MVTec
    AD/VAD, mirrored by `_load_auto6_samples` for Auto6 and `_load_mdtecad2_samples`
    for MDTecAD2): path, split, label, image_path, label_index, mask_path.
    """
    if category in _AUTO6_CATEGORIES:
        return _load_auto6_samples(category)

    if category in _MDTECAD2_CATEGORIES:
        return _load_mdtecad2_samples(category)

    from anomalib.data.datasets.image.mvtecad import MVTecADDataset

    if category in _VAD_CATEGORIES:
        return MVTecADDataset(root=VAD_ROOT, category=category, split=None).samples

    ensure_mvtec_category(category)
    return MVTecADDataset(root=DATA_ROOT, category=category, split=None).samples


def image_id_for(label: str, image_path: str) -> str:
    """Filesystem-safe unique ID within one category: defect-subfolder + stem.

    Needed because MVTec re-uses filenames like "000.png" across the "good"
    and each defect-type test subfolder.
    """
    return f"{label}_{Path(image_path).stem}"


# Per-category (limit, train_limit) defaults applied when the caller passes neither
# explicitly. VAD's train/good split (2000 images) is slow to fit on every run, so it
# defaults to 200 train / 200 normal + 200 anomalous test instead of the full 2000/1835.
DEFAULT_LIMITS: dict[str, tuple[int | None, int | None]] = {
    "vad": (400, 200),
}


def resolve_paths(
    category: str, limit: int | None = None, train_limit: int | None = None
) -> tuple[list[Path], pd.DataFrame]:
    """Return (train image paths, test sample rows) for *category*.

    Args:
        category: MVTec AD category name.
        limit: If given, caps the test set to roughly `limit` images — split
            evenly between normal and anomalous so a small smoke-test run
            still exercises both classes — and, when `train_limit` is not
            given, also caps the train set to `max(limit, 5)` images so
            fitting stays fast during smoke tests.
        train_limit: If given, caps the train set independently of `limit`
            (e.g. a slow "good" split that needs a smaller cap than the test
            split).

    `limit`/`train_limit` left as `None` fall back to `DEFAULT_LIMITS[category]`
    when the category has one (currently just VAD); pass an explicit value to
    override.
    """
    default_limit, default_train_limit = DEFAULT_LIMITS.get(category, (None, None))
    if limit is None:
        limit = default_limit
    if train_limit is None:
        train_limit = default_train_limit

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
        if train_limit is None:
            train_paths = train_paths[: max(limit, 5)]
    if train_limit is not None:
        train_paths = train_paths[:train_limit]

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
