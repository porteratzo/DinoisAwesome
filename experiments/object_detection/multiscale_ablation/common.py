"""Paths, configs, and cache-path helpers for the multi-scale-crop method comparison.

Mirrors the split used by ``experiments/anomaly_detection`` (``common.py`` /
``methods.py`` / ``run_experiments.py`` / ``analyze_results.py``), adapted to this
experiment's domain: instead of one image-level anomaly score, each "pair" (a part
type + an instance-type group, e.g. ``RHa`` / ``foam``) has a reference image (build
exemplar prototypes from it) and a query image (scored against those prototypes),
matched against GT instance clusters for P/R/F1/mIoU.

Two config objects gate two independent cache tiers so a new method never forces a
re-encode of images it doesn't need a different crop for (see module docstring in
``run_experiments.py``):

- :class:`CropConfig` — encoder + crop-geometry parameters. Governs
  ``cache/crops/<crop_hash>/...``, which holds the *expensive* artifact: DINO patch
  tokens for the reference exemplar crops (per scale) and the query image. Every
  method that reuses the same scales/padding/encoder settings shares one cache entry
  here, regardless of how it turns those tokens into a prototype (mean, k-means, ...).
- :class:`ScoringConfig` — threshold/clustering/ROI parameters. Governs
  ``cache/methods/<crop_hash>__<scoring_hash>/...`` (per-method score maps, clusters,
  metrics) and ``cache/blobs/<crop_hash>__<scoring_hash>/...`` (two-stage ROI blob
  crops + tokens, keyed additionally by which method's raw map located them).

A method that needs different crops (e.g. a new scale, different padding) gets its
own ``CropConfig`` and therefore its own crop cache namespace, without invalidating
anything existing methods rely on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env")

DATA_DIR = REPO_ROOT / "data" / "abc3"
OUTPUT_DIR = REPO_ROOT / "outputs" / "object_detection" / "multiscale_ablation"
CACHE_ROOT = OUTPUT_DIR / "cache"
RESULTS_ROOT = OUTPUT_DIR / "results"
FIGURES_ROOT = RESULTS_ROOT / "figures"

REF_NUMBER = 1
QUERY_NUMBER = 2


def _stable_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


@dataclass(frozen=True)
class CropConfig:
    """Everything that changes what gets encoded (crops + query tokens)."""

    dino_version: str = "v3"
    dino_size: str = "large"
    img_size: int = 1024
    layer_idx: int = 23
    debias: bool = True
    mask_patch_threshold: float = 0.3
    scales: tuple[str, ...] = ("global", "mid", "close")
    exemplar_close_padding_fraction: float = 1.0
    min_crop_size: int = 128

    def hash(self) -> str:
        return _stable_hash(asdict(self))


@dataclass(frozen=True)
class ScoringConfig:
    """Threshold/clustering/ROI parameters — cheap to recompute from cached tokens,
    but still cached (Tier C) so ``visualize_results.py`` never needs the encoder.
    """

    ref_threshold_steps: int = 25
    cluster_size_margin: float = 0.5
    gt_dbscan_eps: float = 1.5
    gt_dbscan_min_samples: int = 1
    iou_match_threshold: float = 0.3
    min_points_floor: int = 2
    cluster_reject_margin_fraction: float = 0.2
    pred_dbscan_eps_patches: int = 2
    pred_dbscan_min_samples: int = 2
    knn_fgbg_num_neighbours: int = 10
    kmeans_ks: tuple[int, ...] = (3, 8)
    roi_binarize_method: Literal["otsu", "single", "percentile"] = "percentile"
    roi_single_threshold: float = 0.5
    roi_percentile: float = 98.0
    roi_morph_close_iterations: int = 2
    roi_crop_mode: Literal["per_blob", "single_bbox"] = "per_blob"
    blob_crop_padding_fraction: float = 0.2
    max_upscale_factor: float = 2.0

    def hash(self) -> str:
        return _stable_hash(asdict(self))


DEFAULT_CROP_CONFIG = CropConfig()
DEFAULT_SCORING_CONFIG = ScoringConfig()


@dataclass(frozen=True)
class PairKey:
    """One (part type, instance-type group) exemplar/query pair — the experiment's
    unit of work, mirroring ``run_pair`` in the original notebook script.
    """

    part_type: str
    instance_type: str
    ref_number: int = REF_NUMBER
    query_number: int = QUERY_NUMBER

    @property
    def slug(self) -> str:
        safe_instance = self.instance_type.replace(" ", "-").replace("/", "-")
        return f"{self.part_type}__{safe_instance}__{self.ref_number}-{self.query_number}"


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------


def crop_cache_dir(pair: PairKey, crop_cfg: CropConfig) -> Path:
    d = CACHE_ROOT / "crops" / crop_cfg.hash() / pair.slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def crop_cache_path(pair: PairKey, crop_cfg: CropConfig) -> Path:
    return crop_cache_dir(pair, crop_cfg) / "tokens.pt"


def method_cache_dir(pair: PairKey, crop_cfg: CropConfig, scoring_cfg: ScoringConfig) -> Path:
    d = CACHE_ROOT / "methods" / f"{crop_cfg.hash()}__{scoring_cfg.hash()}" / pair.slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def method_cache_path(
    pair: PairKey, crop_cfg: CropConfig, scoring_cfg: ScoringConfig, method_name: str
) -> Path:
    safe_name = method_name.replace("/", "-")
    return method_cache_dir(pair, crop_cfg, scoring_cfg) / f"{safe_name}.pkl"


def blob_cache_dir(pair: PairKey, crop_cfg: CropConfig, scoring_cfg: ScoringConfig) -> Path:
    d = CACHE_ROOT / "blobs" / f"{crop_cfg.hash()}__{scoring_cfg.hash()}" / pair.slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def blob_cache_path(
    pair: PairKey, crop_cfg: CropConfig, scoring_cfg: ScoringConfig, roi_source: str
) -> Path:
    safe_name = roi_source.replace("/", "-")
    return blob_cache_dir(pair, crop_cfg, scoring_cfg) / f"{safe_name}.pt"


# ---------------------------------------------------------------------------
# Pair enumeration (abc3-specific — mirrors BATCH_MODE's loop in the original script)
# ---------------------------------------------------------------------------


def all_pairs(ref_number: int = REF_NUMBER, query_number: int = QUERY_NUMBER) -> list[PairKey]:
    """Every (part_type, instance_type group) combo actually present in data/abc3."""
    from dinoisawesome.abc3 import INSTANCE_TYPE_GROUPS, PART_TYPES, available_instance_groups

    pairs: list[PairKey] = []
    for part_type in PART_TYPES:
        ref_stem = f"{part_type}_{ref_number}"
        ref_ann_stem = DATA_DIR / "annotations" / ref_stem
        for group_name in available_instance_groups(ref_ann_stem, INSTANCE_TYPE_GROUPS):
            pairs.append(PairKey(part_type, group_name, ref_number, query_number))
    return pairs


def instance_classes_for(instance_type: str) -> list[str] | None:
    from dinoisawesome.abc3 import INSTANCE_TYPE_GROUPS

    return INSTANCE_TYPE_GROUPS[instance_type]
