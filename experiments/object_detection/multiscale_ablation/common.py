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


# Transformer block counts per (dino_version, dino_size), from the vendored torch-hub
# source (dinov2/models/vision_transformer.py's vit_small/base/large/giant2 `depth=`,
# dinov3/hub/backbones.py's dinov3_vits16/vitb16/vitl16/vit7b16 `depth=`). Used to derive
# CropConfig.layer_idx's "last block" default so it stays valid across --model sizes
# instead of the old hardcoded 23 (large-only: base/small only have 12 blocks, so
# layers=[23] on them fails get_intermediate_layers's "only 0/1 blocks found" assert).
_NUM_BLOCKS: dict[tuple[str, str], int] = {
    ("v2", "small"): 12,
    ("v2", "base"): 12,
    ("v2", "large"): 24,
    ("v2", "giant"): 40,
    ("v3", "small"): 12,
    ("v3", "base"): 12,
    ("v3", "large"): 24,
    ("v3", "giant"): 40,
}


def last_block_idx(dino_version: str, dino_size: str) -> int:
    return _NUM_BLOCKS[(dino_version, dino_size)] - 1


@dataclass(frozen=True)
class CropConfig:
    """Everything that changes what gets encoded (crops + query tokens)."""

    dino_version: str = "v3"
    dino_size: str = "large"
    img_size: int = 1024
    # None -> last transformer block of (dino_version, dino_size), resolved in
    # __post_init__ so it stays correct when dino_size changes (e.g. via --model);
    # pass an explicit int to pin a specific block regardless of size.
    layer_idx: int | None = None
    debias: bool = True
    mask_patch_threshold: float = 0.3
    scales: tuple[str, ...] = ("global", "mid", "close")
    exemplar_close_padding_fraction: float = 1.0
    min_crop_size: int = 128

    # --- Background-gallery enrichment (Phase 1, ported from bg_gallery_enrichment.py) ---
    # Rejection-sampled extra mid/close-scale crops elsewhere in the reference image (no
    # foreground, low overlap with real/other-extra crops of the same scale), folded into
    # that scale's bg mean/gallery. 0 (default) is a no-op — identical to today's behaviour.
    # A CropConfig field (not ScoringConfig): sampling these needs its own encoder passes, so
    # a non-zero value gets its own crop-cache namespace (see this module's docstring).
    bg_enrich_crops_per_scale: int = 0
    bg_enrich_max_overlap_fraction: float = 0.35
    bg_enrich_seed: int = 0

    def __post_init__(self) -> None:
        if self.layer_idx is None:
            object.__setattr__(self, "layer_idx", last_block_idx(self.dino_version, self.dino_size))

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
    pred_dbscan_eps_patches: int = 1
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

    # --- Two-stage IoU-degradation fixes (toggleable — flip either back if it regresses;
    # see multiscale_crop_ablation.py's PROJECTION_MODE / TUNE_THRESHOLD_PER_SCALE for the
    # full rationale, ported here 1:1) ---
    #
    # "nearest_union" (legacy): project_crop_mask_to_query_grid snaps each foreground crop
    # patch's center to the nearest query cell and ORs it in — a point-sample, not the
    # area-fraction rule GT masks are built with, so it can over/under-cover a query cell.
    # "area_weighted": accumulates each foreground crop patch's real native-pixel overlap
    # area into every query cell it intersects, then thresholds with that same GT rule.
    projection_mode: Literal["nearest_union", "area_weighted"] = "area_weighted"
    # Tune the "close" method's threshold/cluster-reject cutoff against its own
    # representative ref crop instead of always ref_mid — ref_mid is a domain mismatch
    # that two-stage's stage-2 rescoring (native "close" crops) actually experiences
    # directly. Scoped only to "close" (see run_experiments.py's _run_method).
    tune_threshold_per_scale: bool = True

    # --- Fg/bg gallery cleaning (Phase 2, ported from noisy_fgbg_cleaning.py) ---
    # "raw" (default) is a no-op — identical to today's behaviour. "step1" (spatial filter)
    # tightens the single mask_patch_threshold cut into a high/low band, dropping the
    # ambiguous boundary band from both fg and bg. "step2_cls"/"step2_center" keep step1's
    # own raw (mask_patch_threshold) fg selection and further drop the least-similar tail vs.
    # an independent appearance reference (the close crop's own [CLS] token, or a masked-mean
    # over the mask's innermost "core" pixels) — bg is untouched, a foreground-only check.
    # Only mid/close are cleaned; "global" is unaffected (see cleaning.py's module docstring).
    # A ScoringConfig field, not CropConfig: it only re-filters already-cached crop tokens, no
    # new encoder pass needed, so flipping it never invalidates the crop cache.
    fg_clean_stage: Literal["raw", "step1", "step2_cls", "step2_center"] = "raw"
    fg_clean_high: float = 0.85
    fg_clean_low: float = 0.15
    fg_clean_attention_keep_fraction: float = 0.75
    fg_clean_center_core_percentile: float = 70.0

    # Added to the tuned per-method pixel-selection threshold (raw > thr) before it's
    # applied — lets an ablation sweep stricter/looser thresholding without re-tuning.
    # Does not affect cluster_reject_thr (a separate cosine cutoff on whole-cluster
    # embeddings, auto-tuned independently — see tune_cluster_reject_threshold).
    # 0.0 (default) is a no-op. A ScoringConfig field so each offset gets its own
    # method-cache namespace.
    threshold_offset: float = 0.0

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
    def safe_instance_type(self) -> str:
        return self.instance_type.replace(" ", "-").replace("/", "-")

    @property
    def slug(self) -> str:
        return f"{self.part_type}__{self.safe_instance_type}__{self.ref_number}-{self.query_number}"

    @property
    def case_slug(self) -> str:
        """Identity within an instance-type group — same as `slug` minus the instance-type
        segment, for when instance_type is already the parent directory."""
        return f"{self.part_type}__{self.ref_number}-{self.query_number}"


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
