# %% [markdown]
# # Fundamental: Feature-Space Transformations — Does Reshaping Raw DINOv3 Patch
# # Geometry Improve FG/BG Separability Under kNN/Cosine Matching?
#
# Every other fundamental experiment in this directory changes *what* goes into the fg/bg
# galleries (augmentation severity in `augmented_prototype_oracle_iou_knn_fgbg.py`,
# gallery-cleaning stages in `noisy_fgbg_cleaning.py`). This one holds the galleries fixed —
# the same close+mid+global multiscale pooling those files already use — and instead sweeps
# *how the raw embedding geometry itself is reshaped* before matching: mean-centering,
# ZCA whitening, PCA truncation, a supervised LDA direction, and a Mahalanobis-distance
# variant, each fit **per combo** (per reference instance) from that combo's own pooled
# fg/bg patches. Scored the same way as every sibling file: oracle IoU, the best patch-mask
# IoU any single global threshold on the raw score map could achieve against GT
# (`_shared.thresholding.iou_threshold_curve`).
#
# ## Pipelines
#
# | # | Pipeline | Fit source | Swept? | Matching method(s) |
# |---|---|---|---|---|
# | 1 | `raw` | — | no | single_proto, knn_fgbg (cosine) |
# | 2 | `global_center` | mean(fg u bg) | no | single_proto, knn_fgbg |
# | 3 | `bg_center` | mean(bg) | no | single_proto, knn_fgbg |
# | 4 | `global_zca` | eigh(cov(fg u bg)) | eps | single_proto, knn_fgbg |
# | 5 | `bg_zca` | eigh(cov(bg)) | eps | single_proto, knn_fgbg |
# | 6 | `lda` | shrinkage LDA, fg vs bg | no | lda_direct (own scalar score) |
# | 7 | `pca_truncate` | reuses #4's eigh, top-k | k | single_proto, knn_fgbg (k-dim space) |
# | 8 | `mahalanobis` | reuses #5's eigh, no final L2-norm | eps (shared w/ #5) | mahalanobis_knn |
#
# Every pipeline except `lda` and `mahalanobis` ends with an L2-normalize before scoring —
# matches the "+ L2 Norm" suffix on every row of the original spec's experimental-plan
# table. `lda` and `mahalanobis` deliberately skip it, and both are real deviations from a
# literal reading of that table, not oversights:
#
# - **`lda`**: a 2-class LDA has exactly one discriminant direction (`rank(S_B) <= 1`), so
#   its "projection" is already a 1-D scalar per patch. Cosine similarity on a 1-D vector is
#   degenerate (every point is +-1 after normalizing a scalar) — the signed projection is
#   used directly as the score map instead.
# - **`mahalanobis`**: reuses `bg_zca`'s fit (same mean/eigh/eps grid) but keeps the
#   whitened residual's *magnitude* instead of discarding it via a final L2-normalize, then
#   scores with a Euclidean-distance contrastive kNN (`knn_fgbg_score_euclidean`) instead of
#   cosine similarity. `bg_zca` and `mahalanobis` sharing a fit is deliberate: their IoU gap
#   directly answers "does keeping distance-from-background magnitude help", isolated from
#   any difference in how the background is whitened.
#
# ## Fit granularity
#
# Per-combo pooled patch counts (a few hundred to ~1500, from close+mid fg and
# close+mid+global bg) are typically below DINOv3-large's C=1024 feature dimension, so the
# per-combo covariance is rank-deficient and the LDA within-class scatter is singular.
# `_shared.feature_transforms.zca_matrix`'s `eps` and `fit_lda_direction`'s Ledoit-Wolf
# shrinkage are what make these fits well-defined despite that, not optional regularization
# — see the closing markdown cell for how to read results in light of this.

# %% Logging — must be before torch import
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("feature_transform_oracle_iou")

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from matplotlib.patches import Patch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from dinoisawesome import DinoEncoder, EncoderWithCache, compute_exemplar_features, load_annotations
from dinoisawesome.abc3 import INSTANCE_TYPE_GROUPS, PART_TYPES, available_instance_groups

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared.abc3_combos import combo_key  # noqa: E402
from _shared.dataset_pairs import REF_QUERY_PAIRS, RefQueryPair  # noqa: E402
from _shared.latency import cuda_timer, images_per_sec  # noqa: E402
from _shared.feature_transforms import (  # noqa: E402
    apply_affine,
    fit_cov_eigh,
    fit_lda_direction,
    fit_mean,
    knn_fgbg_score_euclidean,
    lda_score,
    pca_truncate,
    zca_matrix,
)
from _shared.mask_geometry import pixel_mask_to_patch_mask, scale_crop_box  # noqa: E402
from _shared.pooled_gallery_cv import (  # noqa: E402
    MAX_BANK_SIZE_TRANSFORM_53,
    N_FOLDS_53,
    cap_bank_size,
    discover_all_instances,
    make_fold_role_splits,
)
from _shared.prototype_ops import knn_score_heatmap, score_heatmap  # noqa: E402
from _shared.qualitative_gallery import ScoredExample, save_score_gallery  # noqa: E402
from _shared.stats import bootstrap_ci, bootstrap_prob_greater  # noqa: E402
from _shared.thresholding import achievable_iou, oracle_iou  # noqa: E402

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

DATA_ROOT = _REPO_ROOT / "data"

# abc5 has four ref/query pairs per part type — (1,2)/(3,4)/(5,6)/(7,8) (see
# _shared/dataset_pairs.py) — narrow for fast iteration, e.g. REF_QUERY_PAIRS[:1].
RUN_PAIRS: list[RefQueryPair] = REF_QUERY_PAIRS

# Same focus combo as the sibling fundamental scripts, for direct comparability of figures.
FOCUS_UNIT = "LHa_1-2"
FOCUS_CLASS = "donut foam single"
FOCUS_INSTANCE_ID = 1

DINO_VERSION = "v3"
DINO_SIZE = "base"
IMG_SIZE = 768
LAYER_IDX = 11  # last block of ViT-B/16 (depth 12)
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")
DINO_ENCODING_CACHE_DIR: str | None = os.environ.get("DINO_ENCODING_CACHE_DIR")

MASK_PATCH_THRESHOLD = 0.3
CROP_PADDING_FRACTION = 1.0
MIN_CROP_SIZE = 128

ORACLE_THRESHOLD_STEPS = 25
SCALES: list[str] = ["close", "mid"]  # crop scales; "global" (full ref image) is bg-only, Part 3.6

# Which scales feed each variant's fg gallery, swept as an outer axis over every pipeline x
# method x param cell below. bg stays fixed at close+mid+global ("all") for every scale combo,
# mirroring multiscale_crop_ablation.py's FGBG_SOURCE_COMBOS convention (bg is always the full
# "all" set regardless of which fg combo is scored) — isolates "does broadening the fg gallery's
# own scales help" from any bg-side effect. "close+mid" is the combo every pipeline comparison
# in this file used before this sweep existed, kept here as the reference point.
SCALE_COMBOS: dict[str, list[str]] = {
    "close+mid": ["close", "mid"],
    "global+mid": ["global", "mid"],
    "global+mid+close": ["global", "mid", "close"],
}

KNN_FGBG_NUM_NEIGHBOURS = 10

# Matches the taxonomy's stated epsilon range; shared by global_zca, bg_zca, and mahalanobis
# (mahalanobis reuses bg_zca's own eigh, see the module docstring).
EPS_SWEEP: list[float] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# Fractions of DINOv3-large's C=1024.
PCA_K_SWEEP: list[int] = [32, 64, 128, 256, 512]

PIPELINES: list[str] = [
    "raw",
    "global_center",
    "bg_center",
    "global_zca",
    "bg_zca",
    "lda",
    "pca_truncate",
    "mahalanobis",
]
METHODS_BY_PIPELINE: dict[str, list[str]] = {
    "raw": ["single_proto", "knn_fgbg"],
    "global_center": ["single_proto", "knn_fgbg"],
    "bg_center": ["single_proto", "knn_fgbg"],
    "global_zca": ["single_proto", "knn_fgbg"],
    "bg_zca": ["single_proto", "knn_fgbg"],
    "lda": ["lda_direct"],
    "pca_truncate": ["single_proto", "knn_fgbg"],
    "mahalanobis": ["mahalanobis_knn"],
}
# Which swept-parameter grid each pipeline's fit is repeated over; unswept pipelines get a
# single `[None]` "sweep".
SWEPT_PIPELINES: dict[str, list[Any]] = {
    "global_zca": EPS_SWEEP,
    "bg_zca": EPS_SWEEP,
    "pca_truncate": PCA_K_SWEEP,
    "mahalanobis": EPS_SWEEP,
}
METHOD_COLOR: dict[str, str] = {
    "single_proto": "#7f8c8d",
    "knn_fgbg": "#2ecc71",
    "lda_direct": "#9b59b6",
    "mahalanobis_knn": "#e67e22",
}

# One representative (scale_combo, pipeline, param, method) point whose individual per-combo
# queries get kept as PIL images + raw score maps for the worst/best-N qualitative gallery
# below — collecting this for every (scale_combo, pipeline, param) cell would multiply
# memory/disk cost, so only this one point is captured. "close+mid" is this file's own
# pre-sweep reference scale combo (see SCALE_COMBOS' docstring); bg_zca/knn_fgbg at a mid-range
# epsilon is a representative non-trivial transform (not the degenerate raw/lda extremes).
QUALITATIVE_SCALE_COMBO = "close+mid"
QUALITATIVE_PIPELINE = "bg_zca"
QUALITATIVE_PARAM: float = EPS_SWEEP[2]
QUALITATIVE_METHOD = "knn_fgbg"
QUALITATIVE_MAX_EXAMPLES = 60  # capped so the gallery figure itself stays a readable size

# Bootstrap settings for the headline CI (added to every pipeline_method_summary row) and the
# PCA_K_SWEEP lowest-vs-highest-k significance check (see the closing Parts below).
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

SEED = 0
torch.manual_seed(SEED)

OUTPUT_DIR = _REPO_ROOT / "outputs" / "fundamental_abc5" / "feature_transform_oracle_iou"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log.info(
    "RUN_PAIRS=%d units (%s)  |  DINO%s-%s img_size=%d layer=%d  |  "
    "pipelines=%s eps_sweep=%s pca_k_sweep=%s scale_combos=%s",
    len(RUN_PAIRS),
    [p.unit for p in RUN_PAIRS],
    DINO_VERSION,
    DINO_SIZE,
    IMG_SIZE,
    LAYER_IDX,
    PIPELINES,
    EPS_SWEEP,
    PCA_K_SWEEP,
    SCALE_COMBOS,
)

# %% Helpers shared across discovery / scoring / aggregation / plotting


def split_fg_bg_patches_raw(
    patch_tokens: torch.Tensor,
    mask_px: np.ndarray,
    grid_h: int,
    grid_w: int,
    label: str,
    *,
    bg_exclude_mask_px: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a crop's RAW (non-L2-normalised) patch tokens into (fg, bg) — identical in
    spirit to augmented_prototype_oracle_iou_knn_fgbg.py's local `split_fg_bg_patches`,
    except it skips that helper's opening `F.normalize`: every transform pipeline below
    needs raw embedding geometry as its input and L2-normalizes (or deliberately doesn't,
    see the module docstring) as its own final step, never before centering/whitening."""
    if bg_exclude_mask_px is None:
        bg_exclude_mask_px = mask_px
    tokens = patch_tokens.reshape(grid_h * grid_w, -1).float()

    fg_patch_mask = pixel_mask_to_patch_mask(
        mask_px, grid_h, grid_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )
    fg_flat = torch.from_numpy(fg_patch_mask.reshape(-1)).to(tokens.device)
    fg = tokens[fg_flat]
    if fg.shape[0] == 0:
        log.warning("%s: fg mask empty after patch-grid projection — using all patches", label)
        fg = tokens

    bg_exclude_patch_mask = pixel_mask_to_patch_mask(
        bg_exclude_mask_px, grid_h, grid_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )
    bg_exclude_flat = torch.from_numpy(bg_exclude_patch_mask.reshape(-1)).to(tokens.device)
    bg = tokens[~bg_exclude_flat]
    if bg.shape[0] == 0:
        log.warning("%s: bg mask empty after patch-grid projection — using all patches", label)
        bg = tokens

    return fg, bg


def knn_euclid_score_heatmap(
    tokens: torch.Tensor, fg_bank: torch.Tensor, bg_bank: torch.Tensor, k: int, h: int, w: int
) -> np.ndarray:
    return knn_fgbg_score_euclidean(tokens, fg_bank, bg_bank, k).reshape(h, w)


def lda_score_heatmap(
    tokens: torch.Tensor, direction: torch.Tensor, mu: torch.Tensor, h: int, w: int
) -> np.ndarray:
    return lda_score(tokens, direction, mu).reshape(h, w).cpu().float().numpy()


IouLookup = dict[str, dict[Any, dict[str, dict[tuple, float]]]]


def score_combo(
    ck: tuple,
    fg_raw: torch.Tensor,
    bg_raw: torch.Tensor,
    q_raw: torch.Tensor,
    q_h: int,
    q_w: int,
    gt: np.ndarray,
    lookup: IouLookup,
    *,
    ref_raw: torch.Tensor | None = None,
    ref_gt: np.ndarray | None = None,
    ref_h: int | None = None,
    ref_w: int | None = None,
    achievable_lookup: IouLookup | None = None,
) -> dict[tuple[str, Any, str], np.ndarray]:
    """Fit every pipeline's parameters from this combo's own (fg_raw, bg_raw), score the
    query against each pipeline x method x swept-param combination, store each oracle IoU in
    `lookup[pipeline][param][method][ck]`, and return every raw score map keyed by
    (pipeline, param, method) — reused by the qualitative figures below so the focus combo
    never needs rescoring from scratch. Runs `fit_cov_eigh` exactly once for the global pool
    and once for the bg-only pool; every swept eps/k value below reuses those two
    eigendecompositions (see `_shared.feature_transforms.fit_cov_eigh`).

    Achievable IoU (Addition 1): when `ref_raw`/`ref_gt` (the exemplar/ref image's own raw
    tokens + its own GT patch mask, at its own grid `ref_h`/`ref_w`) and `achievable_lookup`
    are all given, every fitted pipeline/param/method is *also* scored against the reference
    with the exact same fit (mu/w/eigvecs/proto/fg_t/bg_t — never refit), then
    `_shared.thresholding.achievable_iou` tunes a threshold on that reference score map/GT and
    transfers it as-is to the query's own score map. This never looks at the query's own GT to
    pick a threshold — `oracle_iou` on the same query is always >= this value. Left absent
    (achievable_lookup untouched for this ck) when no reference is available."""
    raw_maps: dict[tuple[str, Any, str], np.ndarray] = {}
    has_ref = ref_raw is not None and ref_gt is not None and achievable_lookup is not None

    def cosine_variant(
        pipeline: str,
        param: Any,
        mu: torch.Tensor,
        w: torch.Tensor | None = None,
        eigvecs_for_pca: torch.Tensor | None = None,
        k_trunc: int | None = None,
    ) -> None:
        if k_trunc is not None:
            assert eigvecs_for_pca is not None
            fg_t = F.normalize(pca_truncate(fg_raw, mu, eigvecs_for_pca, k_trunc), p=2, dim=-1)
            bg_t = F.normalize(pca_truncate(bg_raw, mu, eigvecs_for_pca, k_trunc), p=2, dim=-1)
            q_t = F.normalize(pca_truncate(q_raw, mu, eigvecs_for_pca, k_trunc), p=2, dim=-1)
        else:
            fg_t = F.normalize(apply_affine(fg_raw, mu, w), p=2, dim=-1)
            bg_t = F.normalize(apply_affine(bg_raw, mu, w), p=2, dim=-1)
            q_t = F.normalize(apply_affine(q_raw, mu, w), p=2, dim=-1)

        proto = compute_exemplar_features(fg_t, mode="mean")
        raw_proto = score_heatmap(q_t, proto, q_h, q_w)
        raw_maps[(pipeline, param, "single_proto")] = raw_proto
        lookup[pipeline][param]["single_proto"][ck] = oracle_iou(
            raw_proto, gt, ORACLE_THRESHOLD_STEPS
        )

        raw_knn = knn_score_heatmap(q_t, fg_t, bg_t, KNN_FGBG_NUM_NEIGHBOURS, q_h, q_w)
        raw_maps[(pipeline, param, "knn_fgbg")] = raw_knn
        lookup[pipeline][param]["knn_fgbg"][ck] = oracle_iou(raw_knn, gt, ORACLE_THRESHOLD_STEPS)

        if has_ref:
            if k_trunc is not None:
                ref_t = F.normalize(pca_truncate(ref_raw, mu, eigvecs_for_pca, k_trunc), p=2, dim=-1)
            else:
                ref_t = F.normalize(apply_affine(ref_raw, mu, w), p=2, dim=-1)
            ref_raw_proto = score_heatmap(ref_t, proto, ref_h, ref_w)
            achievable_lookup[pipeline][param]["single_proto"][ck] = achievable_iou(
                ref_raw_proto, ref_gt, raw_proto, gt, ORACLE_THRESHOLD_STEPS
            )
            ref_raw_knn = knn_score_heatmap(
                ref_t, fg_t, bg_t, KNN_FGBG_NUM_NEIGHBOURS, ref_h, ref_w
            )
            achievable_lookup[pipeline][param]["knn_fgbg"][ck] = achievable_iou(
                ref_raw_knn, ref_gt, raw_knn, gt, ORACLE_THRESHOLD_STEPS
            )

    all_raw = torch.cat([fg_raw, bg_raw], dim=0)
    mu_all = fit_mean(all_raw)
    v_all, ev_all = fit_cov_eigh(all_raw, mu_all)
    mu_bg = fit_mean(bg_raw)
    v_bg, ev_bg = fit_cov_eigh(bg_raw, mu_bg)

    cosine_variant("raw", None, torch.zeros_like(mu_all))
    cosine_variant("global_center", None, mu_all)
    cosine_variant("bg_center", None, mu_bg)
    for eps in EPS_SWEEP:
        cosine_variant("global_zca", eps, mu_all, w=zca_matrix(v_all, ev_all, eps))
    for eps in EPS_SWEEP:
        cosine_variant("bg_zca", eps, mu_bg, w=zca_matrix(v_bg, ev_bg, eps))
    for k in PCA_K_SWEEP:
        cosine_variant("pca_truncate", k, mu_all, eigvecs_for_pca=v_all, k_trunc=k)

    direction, mu_lda = fit_lda_direction(fg_raw, bg_raw)
    raw_lda = lda_score_heatmap(q_raw, direction, mu_lda, q_h, q_w)
    raw_maps[("lda", None, "lda_direct")] = raw_lda
    lookup["lda"][None]["lda_direct"][ck] = oracle_iou(raw_lda, gt, ORACLE_THRESHOLD_STEPS)
    if has_ref:
        ref_raw_lda = lda_score_heatmap(ref_raw, direction, mu_lda, ref_h, ref_w)
        achievable_lookup["lda"][None]["lda_direct"][ck] = achievable_iou(
            ref_raw_lda, ref_gt, raw_lda, gt, ORACLE_THRESHOLD_STEPS
        )

    for eps in EPS_SWEEP:
        w = zca_matrix(v_bg, ev_bg, eps)
        fg_t = apply_affine(fg_raw, mu_bg, w)
        bg_t = apply_affine(bg_raw, mu_bg, w)
        q_t = apply_affine(q_raw, mu_bg, w)
        raw_maha = knn_euclid_score_heatmap(q_t, fg_t, bg_t, KNN_FGBG_NUM_NEIGHBOURS, q_h, q_w)
        raw_maps[("mahalanobis", eps, "mahalanobis_knn")] = raw_maha
        if has_ref:
            ref_t = apply_affine(ref_raw, mu_bg, w)
            ref_raw_maha = knn_euclid_score_heatmap(
                ref_t, fg_t, bg_t, KNN_FGBG_NUM_NEIGHBOURS, ref_h, ref_w
            )
            achievable_lookup["mahalanobis"][eps]["mahalanobis_knn"][ck] = achievable_iou(
                ref_raw_maha, ref_gt, raw_maha, gt, ORACLE_THRESHOLD_STEPS
            )
        lookup["mahalanobis"][eps]["mahalanobis_knn"][ck] = oracle_iou(
            raw_maha, gt, ORACLE_THRESHOLD_STEPS
        )

    return raw_maps


# %% Part 1 — discover every (part_type, instance-type group, ref instance) combo
combos: list[dict] = []
group_query_masks: dict[tuple[str, str], np.ndarray] = {}
group_ref_masks: dict[tuple[str, str], np.ndarray] = {}
ref_images: dict[str, Image.Image] = {}
query_images: dict[str, Image.Image] = {}

for pair in tqdm(RUN_PAIRS, desc="Discovering combos"):
    unit = pair.unit
    pair_data_dir = DATA_ROOT / pair.dataset
    ref_stem = f"{pair.part_type}_{pair.ref}"
    query_stem = f"{pair.part_type}_{pair.query}"
    ref_ann_path = pair_data_dir / "annotations" / ref_stem
    query_ann_path = pair_data_dir / "annotations" / query_stem

    groups = available_instance_groups(ref_ann_path)
    if not groups:
        log.warning("unit=%s: no instance-type groups annotated — skipping", unit)
        continue

    ref_anns = load_annotations(ref_ann_path)
    query_anns = load_annotations(query_ann_path)
    ref_images[unit] = Image.open(pair_data_dir / f"{ref_stem}.jpg").convert("RGB")
    query_images[unit] = Image.open(pair_data_dir / f"{query_stem}.jpg").convert("RGB")

    for group in groups:
        classes = INSTANCE_TYPE_GROUPS[group]
        ref_group_anns = [a for a in ref_anns if a["class"] in classes]
        query_group_anns = [a for a in query_anns if a["class"] in classes]
        if not query_group_anns:
            log.warning("unit=%s group=%s: no query GT instances — skipping", unit, group)
            continue
        group_query_masks[(unit, group)] = np.stack([a["mask"] for a in query_group_anns]).any(
            axis=0
        )
        if ref_group_anns:
            group_ref_masks[(unit, group)] = np.stack([a["mask"] for a in ref_group_anns]).any(
                axis=0
            )
        for ref_ann in ref_group_anns:
            combos.append(
                {
                    "unit": unit,
                    "dataset": pair.dataset,
                    "part_type": pair.part_type,
                    "group": group,
                    "class": ref_ann["class"],
                    "instance_id": ref_ann["instance_id"],
                    "ref_mask": ref_ann["mask"],
                }
            )

if not combos:
    raise RuntimeError(
        f"No combos discovered for RUN_PAIRS={[p.unit for p in RUN_PAIRS]} — every pair was "
        "skipped (no annotated instance-type groups or no query GT)."
    )
log.info(
    "Discovered %d (unit, group, instance) combos across %d units (%d part types)",
    len(combos),
    len({c["unit"] for c in combos}),
    len({c["part_type"] for c in combos}),
)

# %% Part 2 — build close/mid crops per combo
scales_by_ck: dict[tuple, list[str]] = defaultdict(list)
for combo in tqdm(combos, desc="Building close/mid crops"):
    ref_img = ref_images[combo["unit"]]
    group_mask = group_ref_masks.get((combo["unit"], combo["group"]), combo["ref_mask"])
    combo["crops"] = {}
    for scale in SCALES:
        box = scale_crop_box(combo["ref_mask"], scale, CROP_PADDING_FRACTION)
        x0, y0, x1, y1 = box
        if x1 - x0 < MIN_CROP_SIZE or y1 - y0 < MIN_CROP_SIZE:
            log.warning(
                "combo=%s scale=%-5s crop %s below MIN_CROP_SIZE=%dpx — skipping this scale",
                combo_key(combo),
                scale,
                box,
                MIN_CROP_SIZE,
            )
            continue
        crop_img = ref_img.crop(box)
        combo["crops"][scale] = {
            "img": crop_img,
            "mask_px": combo["ref_mask"][y0:y1, x0:x1],
            "bg_exclude_mask_px": group_mask[y0:y1, x0:x1],
        }
        scales_by_ck[combo_key(combo)].append(scale)

log.info("Combos with at least one usable scale: %d/%d", len(scales_by_ck), len(combos))

# %% Part 3 — encoder + RAW query-image patch tokens + GT patch masks
# Deliberately bypasses `dinoisawesome.extract_patch_tokens` (which force-L2-normalizes):
# every pipeline above needs raw embedding geometry as its input, see the module docstring.
encoder = DinoEncoder(
    version=DINO_VERSION,
    size=DINO_SIZE,
    img_size=IMG_SIZE,
    layers=[LAYER_IDX],
    weights_dir=DINO_WEIGHTS_DIR,
    amp=True,
)
encoder = EncoderWithCache(encoder, cache_dir=DINO_ENCODING_CACHE_DIR)
chunk_size = encoder.max_batch_size

# Latency (Addition 2): every phase not already timed gets wrapped in `cuda_timer()` (GPU-
# synchronized, see `_shared/latency.py`) and appended here; written to latency.csv near the
# end of the script alongside a phase-by-phase bar chart.
latency_rows: list[dict] = []

query_raw_encodings: dict[str, tuple[torch.Tensor, int, int]] = {}
with cuda_timer() as t_query_encode:
    for unit in tqdm(sorted(query_images), desc="Encoding query images (raw)"):
        out = encoder(query_images[unit], layers=[LAYER_IDX], debias=True)
        patches = out.patches[:, 0]  # (1, H, W, D)
        q_h, q_w = patches.shape[1], patches.shape[2]
        query_raw_encodings[unit] = (patches[0].reshape(q_h * q_w, -1).float(), q_h, q_w)
latency_rows.append(
    {
        "endpoint": "1-1",
        "phase": "query_image_encode",
        "elapsed_s": t_query_encode["elapsed_s"],
        "n_units": len(query_images),
        "units_per_sec": images_per_sec(len(query_images), t_query_encode["elapsed_s"]),
    }
)

gt_patch_masks: dict[tuple[str, str], np.ndarray] = {}
for (unit, group), pixel_mask in group_query_masks.items():
    _, q_h, q_w = query_raw_encodings[unit]
    gt_patch_masks[(unit, group)] = pixel_mask_to_patch_mask(
        pixel_mask, q_h, q_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )

# Object-size correlation (Addition 3): the query GT's own patch-mask coverage, keyed by
# (unit, group) — every combo sharing a (unit, group) shares this same value, appended to
# `per_combo_rows` in Part 6 below.
gt_area_frac_by_unit_group: dict[tuple[str, str], float] = {
    key: float(gt.sum()) / gt.size for key, gt in gt_patch_masks.items()
}

# %% Part 3.6 — RAW full-reference-image patch tokens: the "global" scale source
# One full-image encode per part type, reused below (Part 4) as every one of that part
# type's combos' "global" scale — both the always-on bg source and, for scale combos whose
# SCALE_COMBOS entry includes "global", an fg source too. Same role as the sibling
# fundamental scripts' "global" scale, just kept raw here instead of normalized.
ref_raw_encodings: dict[str, tuple[torch.Tensor, int, int]] = {}
with cuda_timer() as t_ref_encode:
    for unit in tqdm(sorted(ref_images), desc="Encoding ref images (raw, global bg source)"):
        out = encoder(ref_images[unit], layers=[LAYER_IDX], debias=True)
        patches = out.patches[:, 0]
        r_h, r_w = patches.shape[1], patches.shape[2]
        ref_raw_encodings[unit] = (patches[0].reshape(r_h * r_w, -1).float(), r_h, r_w)
latency_rows.append(
    {
        "endpoint": "1-1",
        "phase": "ref_image_encode",
        "elapsed_s": t_ref_encode["elapsed_s"],
        "n_units": len(ref_images),
        "units_per_sec": images_per_sec(len(ref_images), t_ref_encode["elapsed_s"]),
    }
)

# Achievable IoU (Addition 1): the reference (exemplar) image's own GT, projected onto its own
# patch grid — the fixed-pair natural choice per the module's own ref/query asymmetry (the
# exemplar's own mask is what built the gallery in the first place). Absent for a (unit,
# group) whose ref image had no annotated instances of that group (`group_ref_masks` already
# tracks this — see Part 1/2's own identical `.get(..., combo["ref_mask"])` fallback pattern).
ref_gt_patch_masks: dict[tuple[str, str], np.ndarray] = {}
for (unit, group), pixel_mask in group_ref_masks.items():
    _, r_h, r_w = ref_raw_encodings[unit]
    ref_gt_patch_masks[(unit, group)] = pixel_mask_to_patch_mask(
        pixel_mask, r_h, r_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )

# %% Part 4 — encode every combo's close/mid crops (RAW), pool per-combo fg/bg galleries
# bg = close+mid bg patches + this combo's own "global" bg (Part 3.6's full ref-image tokens,
# minus the whole instance-type group's ref mask) — the same multiscale pooling
# augmented_prototype_oracle_iou_knn_fgbg.py's Part 3.5 builds, kept raw here instead of
# L2-normalized, and always "all" scales regardless of which fg SCALE_COMBOS entry is being
# scored (see SCALE_COMBOS' docstring above). fg is built per (combo, scale) below and
# assembled into whichever scale combo a given sweep cell asks for, further down in this part.
fg_by_scale_raw: dict[tuple, torch.Tensor] = {}
bg_by_scale_raw: dict[tuple, torch.Tensor] = {}

clean_items: list[tuple] = []
for combo in combos:
    ck = combo_key(combo)
    for scale, crop in combo["crops"].items():
        clean_items.append((ck, scale, crop["img"], crop["mask_px"], crop["bg_exclude_mask_px"]))

with cuda_timer() as t_crop_encode:
    for i in tqdm(range(0, len(clean_items), chunk_size), desc="Encoding combo crops (raw)"):
        chunk = clean_items[i : i + chunk_size]
        out = encoder([c[2] for c in chunk], layers=[LAYER_IDX], debias=True)
        chunk_patches = out.patches[:, 0]
        grid_h, grid_w = chunk_patches.shape[1], chunk_patches.shape[2]
        for (ck, scale, _, mask_px, bg_exclude_mask_px), patch_tokens in zip(chunk, chunk_patches):
            fg, bg = split_fg_bg_patches_raw(
                patch_tokens,
                mask_px,
                grid_h,
                grid_w,
                f"{ck} scale={scale}",
                bg_exclude_mask_px=bg_exclude_mask_px,
            )
            # Kept on CPU: with REF_QUERY_PAIRS spanning 16 ref/query pairs, every combo's raw
            # (non-L2-normalised, float32) fg/bg bank held on GPU simultaneously no longer fits
            # (abc3-only fit in ~12GB, the combined pool doesn't) — moved back to the query's
            # device per combo in Part 5.
            fg_by_scale_raw[(ck, scale)] = fg.cpu()
            bg_by_scale_raw[(ck, scale)] = bg.cpu()
latency_rows.append(
    {
        "endpoint": "1-1",
        "phase": "gallery_crop_encode",
        "elapsed_s": t_crop_encode["elapsed_s"],
        "n_units": len(clean_items),
        "units_per_sec": images_per_sec(len(clean_items), t_crop_encode["elapsed_s"]),
    }
)

# "global" scale, per combo: fg is this combo's own ref_mask projected onto the full
# ref-image patch grid (the object's own patches, at the encoder's whole-image resolution);
# bg is the same full-image tokens with the whole instance-type group's ref mask excluded
# (unchanged from before this sweep existed — previously its own combo_bg_global_raw dict).
for combo in combos:
    ck = combo_key(combo)
    unit, group = ck[0], ck[1]
    r_tokens_raw, r_h, r_w = ref_raw_encodings[unit]

    fg_patch_mask = pixel_mask_to_patch_mask(
        combo["ref_mask"], r_h, r_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )
    fg_flat = torch.from_numpy(fg_patch_mask.reshape(-1)).to(r_tokens_raw.device)
    global_fg = r_tokens_raw[fg_flat]
    if global_fg.shape[0] == 0:
        log.warning("%s: global fg mask empty after patch-grid projection — using all patches", ck)
        global_fg = r_tokens_raw
    fg_by_scale_raw[(ck, "global")] = global_fg.cpu()

    exclude_mask_px = group_ref_masks.get((unit, group), combo["ref_mask"])
    exclude_patch_mask = pixel_mask_to_patch_mask(
        exclude_mask_px, r_h, r_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )
    exclude_flat = torch.from_numpy(exclude_patch_mask.reshape(-1)).to(r_tokens_raw.device)
    global_bg = r_tokens_raw[~exclude_flat]
    if global_bg.shape[0] == 0:
        log.warning("%s: global bg mask empty after patch-grid projection — using all patches", ck)
        global_bg = r_tokens_raw
    bg_by_scale_raw[(ck, "global")] = global_bg.cpu()

bg_raw_lookup: dict[tuple, torch.Tensor] = {}
for ck, scales in scales_by_ck.items():
    bg_raw_lookup[ck] = torch.cat(
        [bg_by_scale_raw[(ck, s)] for s in scales] + [bg_by_scale_raw[(ck, "global")]], dim=0
    )

# fg gallery, per (combo, scale-combo name): built for every SCALE_COMBOS entry whose scales
# are all present for this combo. "close"/"mid" can be missing (dropped below MIN_CROP_SIZE,
# see scales_by_ck in Part 2); "global" is always present since it's just the full ref image.
fg_raw_lookup: dict[tuple[tuple, str], torch.Tensor] = {}
for ck, scales in scales_by_ck.items():
    available = {*scales, "global"}
    for combo_name, combo_scales in SCALE_COMBOS.items():
        if not all(s in available for s in combo_scales):
            log.warning(
                "%s: scale combo %r needs %s, only %s available — skipping this combo",
                ck,
                combo_name,
                combo_scales,
                sorted(available),
            )
            continue
        fg_raw_lookup[(ck, combo_name)] = torch.cat(
            [fg_by_scale_raw[(ck, s)] for s in combo_scales], dim=0
        )

log.info(
    "Built raw bg galleries for %d combos and raw fg galleries for %d (combo, scale-combo) cells",
    len(bg_raw_lookup),
    len(fg_raw_lookup),
)

# %% Part 5 — main per-combo fit + score sweep, repeated once per SCALE_COMBOS entry


def new_iou_lookup() -> IouLookup:
    lookup: IouLookup = {}
    for pipeline in PIPELINES:
        params = SWEPT_PIPELINES.get(pipeline, [None])
        lookup[pipeline] = {
            param: {method: {} for method in METHODS_BY_PIPELINE[pipeline]} for param in params
        }
    return lookup


iou_lookup_by_combo: dict[str, IouLookup] = {name: new_iou_lookup() for name in SCALE_COMBOS}
# Achievable IoU (Addition 1): same shape as iou_lookup_by_combo, populated alongside it inside
# score_combo whenever a ref (exemplar) GT is available for that combo's (unit, group).
achievable_lookup_by_combo: dict[str, IouLookup] = {
    name: new_iou_lookup() for name in SCALE_COMBOS
}
# Qualitative worst/best-N gallery (Addition 5): collected only at QUALITATIVE_SCALE_COMBO /
# QUALITATIVE_PIPELINE / QUALITATIVE_PARAM / QUALITATIVE_METHOD (see Parameters above) —
# collecting every (scale_combo, pipeline, param) cell would multiply memory/disk cost.
qualitative_examples: list[ScoredExample] = []
n_score_combos = 0

with cuda_timer() as t_scoring:
    for combo in tqdm(combos, desc="Part 5: fitting + scoring"):
        ck = combo_key(combo)
        unit, group = ck[0], ck[1]
        if ck not in bg_raw_lookup:
            continue
        gt = gt_patch_masks.get((unit, group))
        if gt is None:
            continue
        q_raw, q_h, q_w = query_raw_encodings[unit]
        bg_raw = bg_raw_lookup[ck].to(q_raw.device)
        if bg_raw.shape[0] == 0:
            log.warning("%s: empty bg raw gallery — skipping", ck)
            continue
        # Achievable IoU's reference: the exemplar's own raw tokens + own GT patch mask
        # (Part 3.6/above) — absent (achievable_iou left NaN for this ck) if the ref image
        # had no annotated instances of this group.
        ref_gt = ref_gt_patch_masks.get((unit, group))
        ref_raw_tokens = ref_h = ref_w = None
        if ref_gt is not None:
            ref_raw_tokens, ref_h, ref_w = ref_raw_encodings[unit]
        else:
            log.warning(
                "%s: no reference (exemplar) GT for %s/%s — achievable_iou left NaN", ck, unit, group
            )
        for scale_combo_name in SCALE_COMBOS:
            fg_raw = fg_raw_lookup.get((ck, scale_combo_name))
            if fg_raw is None or fg_raw.shape[0] == 0:
                continue
            fg_raw = fg_raw.to(q_raw.device)
            raw_maps = score_combo(
                ck,
                fg_raw,
                bg_raw,
                q_raw,
                q_h,
                q_w,
                gt,
                iou_lookup_by_combo[scale_combo_name],
                ref_raw=ref_raw_tokens,
                ref_gt=ref_gt,
                ref_h=ref_h,
                ref_w=ref_w,
                achievable_lookup=achievable_lookup_by_combo[scale_combo_name],
            )
            n_score_combos += 1
            if (
                scale_combo_name == QUALITATIVE_SCALE_COMBO
                and len(qualitative_examples) < QUALITATIVE_MAX_EXAMPLES
            ):
                qual_key = (QUALITATIVE_PIPELINE, QUALITATIVE_PARAM, QUALITATIVE_METHOD)
                if qual_key in raw_maps:
                    qualitative_examples.append(
                        ScoredExample(
                            label=f"{ck[0]}/{ck[1]}/{ck[2]}#{ck[3]}",
                            image=query_images[unit],
                            raw=raw_maps[qual_key],
                            gt=gt,
                            score=iou_lookup_by_combo[scale_combo_name][QUALITATIVE_PIPELINE][
                                QUALITATIVE_PARAM
                            ][QUALITATIVE_METHOD][ck],
                        )
                    )
latency_rows.append(
    {
        "endpoint": "1-1",
        "phase": "scoring",
        "elapsed_s": t_scoring["elapsed_s"],
        "n_units": n_score_combos,
        "units_per_sec": images_per_sec(n_score_combos, t_scoring["elapsed_s"]),
    }
)
cache_hits, cache_misses = encoder.total_hits, encoder.total_misses
cache_total = cache_hits + cache_misses
latency_rows.append(
    {
        "endpoint": "1-1",
        "phase": "total",
        "elapsed_s": (
            t_query_encode["elapsed_s"]
            + t_ref_encode["elapsed_s"]
            + t_crop_encode["elapsed_s"]
            + t_scoring["elapsed_s"]
        ),
        "n_units": len(query_images) + len(ref_images) + len(clean_items) + n_score_combos,
        "units_per_sec": float("nan"),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": cache_hits / cache_total if cache_total > 0 else float("nan"),
    }
)

log.info(
    "Scoring complete: %d combos x %d pipelines x %d scale combos",
    len(bg_raw_lookup),
    len(PIPELINES),
    len(SCALE_COMBOS),
)

# %% Part 6 — aggregate + headline bar chart


def mean_std_iou(
    lookup: IouLookup, pipeline: str, param: Any, method: str, combo_keys: set[tuple] | None = None
) -> tuple[float, float, int]:
    vals = [
        v
        for ck, v in lookup[pipeline][param][method].items()
        if combo_keys is None or ck in combo_keys
    ]
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def best_param(lookup: IouLookup, pipeline: str, method: str) -> Any:
    """Swept-parameter value with the highest dataset-wide mean oracle IoU for
    (pipeline, method); the sole `None` entry for unswept pipelines. Used everywhere a chart
    needs one fixed value per pipeline (headline bars, per-group breakdown, the qualitative
    figure) so they stay apples-to-apples with each other instead of each independently
    re-optimizing its own eps/k."""
    candidates = SWEPT_PIPELINES.get(pipeline, [None])
    scored = [(p, mean_std_iou(lookup, pipeline, p, method)[0]) for p in candidates]
    scored = [(p, m) for p, m in scored if not np.isnan(m)]
    return max(scored, key=lambda t: t[1])[0] if scored else candidates[0]


def pipeline_method_summary(
    lookup: IouLookup,
    achievable_lookup: IouLookup | None = None,
    combo_keys: set[tuple] | None = None,
) -> pd.DataFrame:
    rows = []
    for pipeline in PIPELINES:
        for method in METHODS_BY_PIPELINE[pipeline]:
            param = best_param(lookup, pipeline, method)
            mean, std, n = mean_std_iou(lookup, pipeline, param, method, combo_keys)
            vals = [
                v
                for ck, v in lookup[pipeline][param][method].items()
                if combo_keys is None or ck in combo_keys
            ]
            # Bootstrap CI (Addition 4): percentile bootstrap CI on the mean, alongside the
            # plain std this script already reported — std alone doesn't say whether two
            # pipelines' means are actually distinguishable or both plausible draws from the
            # same distribution; see _shared/stats.py, mirroring resolution_ablation.py Part 4
            # / training_set_size_ablation.py Part 6's own identical addition. mean/std/n
            # themselves are untouched, still from the unchanged mean_std_iou() above.
            _, ci_lo, ci_hi = bootstrap_ci(np.asarray(vals), n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
            # Achievable IoU (Addition 1): mean/std across the same (pipeline, param, method)
            # cell's achievable_iou values (NaN entries — combos with no reference GT —
            # excluded), plus the oracle-minus-achievable gap.
            achievable_vals: list[float] = []
            if achievable_lookup is not None:
                achievable_vals = [
                    v
                    for ck, v in achievable_lookup[pipeline][param][method].items()
                    if (combo_keys is None or ck in combo_keys) and not np.isnan(v)
                ]
            mean_ach = float(np.mean(achievable_vals)) if achievable_vals else float("nan")
            std_ach = float(np.std(achievable_vals)) if achievable_vals else float("nan")
            rows.append(
                {
                    "pipeline": pipeline,
                    "method": method,
                    "param": param,
                    "mean_iou": mean,
                    "std_iou": std,
                    "ci95_lo": ci_lo,
                    "ci95_hi": ci_hi,
                    "n_combos": n,
                    "mean_achievable_iou": mean_ach,
                    "std_achievable_iou": std_ach,
                    "oracle_minus_achievable_gap": (
                        float(mean - mean_ach)
                        if not (np.isnan(mean) or np.isnan(mean_ach))
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_pipeline_bar_chart(summary_df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(summary_df))
    bars = ax.bar(
        x,
        summary_df["mean_iou"],
        yerr=summary_df["std_iou"],
        capsize=3,
        color=[METHOD_COLOR[m] for m in summary_df["method"]],
    )
    for bar, mean in zip(bars, summary_df["mean_iou"]):
        if not np.isnan(mean):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(
        x,
        [f"{p}\n({m})" for p, m in zip(summary_df["pipeline"], summary_df["method"])],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("oracle IoU (mean +/- std across combos)")
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(
        handles=[Patch(color=c, label=m) for m, c in METHOD_COLOR.items()],
        fontsize=8,
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_oracle_vs_achievable(summary_df: pd.DataFrame, title: str, out_path: Path) -> None:
    """Oracle IoU (upper bound, tunes the threshold against the query's own GT — solid marker)
    vs. achievable IoU (a threshold tuned on the exemplar's own GT, transferred as-is to the
    query — dashed marker of the same color), per (pipeline, method) at the same x positions
    `plot_pipeline_bar_chart`'s headline bars use."""
    x = np.arange(len(summary_df))
    fig, ax = plt.subplots(figsize=(14, 6))
    for method, color in METHOD_COLOR.items():
        mask = (summary_df["method"] == method).to_numpy()
        if not mask.any():
            continue
        ax.plot(
            x[mask], summary_df.loc[mask, "mean_iou"], marker="o", linestyle="-",
            color=color, label=f"{method} oracle",
        )
        ax.plot(
            x[mask], summary_df.loc[mask, "mean_achievable_iou"], marker="^", linestyle="--",
            color=color, alpha=0.6, label=f"{method} achievable",
        )
    ax.set_xticks(
        x,
        [f"{p}\n({m})" for p, m in zip(summary_df["pipeline"], summary_df["method"])],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("IoU (mean across combos)")
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


summary_by_combo: dict[str, pd.DataFrame] = {}
for scale_combo_name, lookup in iou_lookup_by_combo.items():
    summary_df = pipeline_method_summary(lookup, achievable_lookup_by_combo[scale_combo_name])
    summary_by_combo[scale_combo_name] = summary_df
    log.info(
        "[scale_combo=%s] Oracle-IoU summary (mean +/- std, best swept param per pipeline):",
        scale_combo_name,
    )
    for _, row in summary_df.iterrows():
        log.info(
            "  %-16s %-14s param=%-10s iou=%.3f+/-%.3f (n=%d) ci95=[%.3f,%.3f] achievable=%.3f",
            row.pipeline,
            row.method,
            row.param,
            row.mean_iou,
            row.std_iou,
            row.n_combos,
            row.ci95_lo,
            row.ci95_hi,
            row.mean_achievable_iou,
        )
    _headline_path = OUTPUT_DIR / f"oracle_iou_by_pipeline__{scale_combo_name}.png"
    plot_pipeline_bar_chart(
        summary_df,
        f"Feature-space transforms — 1-1 (single ref/query pair) — oracle IoU by pipeline, "
        f"fg scales={scale_combo_name} ({len(RUN_PAIRS)} part types)",
        _headline_path,
    )
    log.info("Saved headline bar chart to %s", _headline_path)
    _oracle_vs_achievable_path = OUTPUT_DIR / f"oracle_vs_achievable__{scale_combo_name}.png"
    plot_oracle_vs_achievable(
        summary_df,
        f"Oracle (query's own GT) vs. achievable (exemplar-tuned threshold) IoU, "
        f"fg scales={scale_combo_name}",
        _oracle_vs_achievable_path,
    )
    log.info("Saved oracle-vs-achievable chart to %s", _oracle_vs_achievable_path)

combined_summary_df = pd.concat(
    [df.assign(scale_combo=name) for name, df in summary_by_combo.items()], ignore_index=True
)
combined_summary_df.to_csv(OUTPUT_DIR / "oracle_iou_by_pipeline.csv", index=False)

per_combo_rows = [
    {
        "scale_combo": scale_combo_name,
        "pipeline": pipeline,
        "param": param,
        "method": method,
        "unit": ck[0],
        "group": ck[1],
        "class": ck[2],
        "instance_id": ck[3],
        "oracle_iou": iou,
        # Object-size correlation (Addition 3): the query GT's own area fraction, shared by
        # every combo/pipeline/param/method row with the same (unit, group).
        "gt_area_frac": gt_area_frac_by_unit_group.get((ck[0], ck[1]), float("nan")),
    }
    for scale_combo_name, lookup in iou_lookup_by_combo.items()
    for pipeline, by_param in lookup.items()
    for param, by_method in by_param.items()
    for method, by_ck in by_method.items()
    for ck, iou in by_ck.items()
]
per_combo_df = pd.DataFrame(per_combo_rows)
per_combo_df.to_csv(OUTPUT_DIR / "oracle_iou_per_combo.csv", index=False)
log.info(
    "Wrote %s and %s (%d per-combo rows)",
    OUTPUT_DIR / "oracle_iou_by_pipeline.csv",
    OUTPUT_DIR / "oracle_iou_per_combo.csv",
    len(per_combo_rows),
)

# %% Part 6b — does oracle IoU correlate with object size? An aggregate mean (every chart
# above) can hide "this pipeline only helps small/large instances" — `gt_area_frac` (the query
# GT's own patch-mask coverage, already joined onto every row of `oracle_iou_per_combo.csv`
# above) lets us check per (scale_combo, pipeline, method), mirroring the pearson/spearman
# correlation pattern `scale_composition_adaptive_oracle.py` already established for instance
# size vs. optimal scale (and resolution_ablation.py/training_set_size_ablation.py's own
# copies of it).
size_correlation_rows = []
for scale_combo_name in SCALE_COMBOS:
    for pipeline in PIPELINES:
        for method in METHODS_BY_PIPELINE[pipeline]:
            sub = per_combo_df[
                (per_combo_df.scale_combo == scale_combo_name)
                & (per_combo_df.pipeline == pipeline)
                & (per_combo_df.method == method)
            ]
            if len(sub) < 3:
                continue
            pearson_r, pearson_p = pearsonr(sub["gt_area_frac"], sub["oracle_iou"])
            spearman_r, spearman_p = spearmanr(sub["gt_area_frac"], sub["oracle_iou"])
            size_correlation_rows.append(
                {
                    "scale_combo": scale_combo_name,
                    "pipeline": pipeline,
                    "method": method,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "n_samples": len(sub),
                }
            )
size_correlation_df = pd.DataFrame(size_correlation_rows)
size_correlation_df.to_csv(OUTPUT_DIR / "size_correlation.csv", index=False)

# Object-size terciles (global, computed once so the same size cutoffs apply everywhere) x
# oracle IoU, faceted by scale combo, pooled across pipelines per method — the more
# interpretable companion to the raw correlation coefficients above.
try:
    per_combo_df["size_tercile"] = pd.qcut(
        per_combo_df["gt_area_frac"], 3, labels=["small", "medium", "large"]
    )
except ValueError:
    log.warning(
        "gt_area_frac has too few distinct values for 3 clean terciles — falling back to "
        "qcut's own duplicate-safe binning (labels become numeric ranges, not small/medium/large)"
    )
    per_combo_df["size_tercile"] = pd.qcut(per_combo_df["gt_area_frac"], 3, duplicates="drop")

fig, axes = plt.subplots(1, len(SCALE_COMBOS), figsize=(6 * len(SCALE_COMBOS), 5), sharey=True)
for ax, scale_combo_name in zip(axes, SCALE_COMBOS):
    tercile_means = (
        per_combo_df[per_combo_df.scale_combo == scale_combo_name]
        .groupby(["size_tercile", "method"], observed=True)["oracle_iou"]
        .mean()
        .unstack("method")
    )
    tercile_means.plot(
        kind="bar", ax=ax, color=[METHOD_COLOR.get(m, "#333333") for m in tercile_means.columns]
    )
    ax.set_title(f"fg scales={scale_combo_name}")
    ax.set_xlabel("object-size tercile (by GT patch-mask area fraction)")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("mean oracle IoU (pooled across pipelines)")
fig.suptitle("Does object size predict oracle IoU?")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "size_correlation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(
    "Wrote %s and %s", OUTPUT_DIR / "size_correlation.csv", OUTPUT_DIR / "size_correlation.png"
)


def plot_scale_combo_comparison(combined_df: pd.DataFrame, title: str, out_path: Path) -> None:
    """Grouped bar chart: one x-position per (pipeline, method), one bar per scale combo —
    the direct answer to "does broadening the fg gallery's scales help, and does it help every
    pipeline/method the same way" (vs. plot_pipeline_bar_chart's per-combo view, which answers
    "which pipeline is best" for a single fixed scale combo)."""
    cell_order = list(
        combined_df[["pipeline", "method"]].drop_duplicates().itertuples(index=False, name=None)
    )
    combo_names = list(SCALE_COMBOS)
    combo_colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(cell_order))
    width = 0.8 / len(combo_names)
    indexed = combined_df.set_index(["scale_combo", "pipeline", "method"])["mean_iou"]
    for i, combo_name in enumerate(combo_names):
        means = [indexed.get((combo_name, p, m), float("nan")) for p, m in cell_order]
        ax.bar(
            x + i * width,
            means,
            width=width,
            label=combo_name,
            color=combo_colors[i % len(combo_colors)],
        )
    ax.set_xticks(
        x + width * (len(combo_names) - 1) / 2,
        [f"{p}\n({m})" for p, m in cell_order],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("oracle IoU (mean across combos)")
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


_comparison_path = OUTPUT_DIR / "oracle_iou_by_scale_combo_comparison.png"
plot_scale_combo_comparison(
    combined_summary_df,
    f"Feature-space transforms — 1-1 (single ref/query pair) — fg scale combo comparison "
    f"({len(RUN_PAIRS)} part types)",
    _comparison_path,
)
log.info("Saved scale-combo comparison chart to %s", _comparison_path)

overall_best_param_by_combo: dict[str, dict[str, dict[str, Any]]] = {
    scale_combo_name: {
        pipeline: {
            method: best_param(lookup, pipeline, method) for method in METHODS_BY_PIPELINE[pipeline]
        }
        for pipeline in PIPELINES
    }
    for scale_combo_name, lookup in iou_lookup_by_combo.items()
}

# %% Part 7 — sweep curves: epsilon (global/bg ZCA + Mahalanobis), k (PCA truncation), per
# scale combo. These explore each transform's own hyperparameter, orthogonal to which fg
# scales feed it, so every SCALE_COMBOS entry gets its own pair of CSV/PNG outputs.
ZCA_MAHALANOBIS_CELLS = [
    ("global_zca", "single_proto"),
    ("global_zca", "knn_fgbg"),
    ("bg_zca", "single_proto"),
    ("bg_zca", "knn_fgbg"),
    ("mahalanobis", "mahalanobis_knn"),
]

eps_sweep_rows = []
for scale_combo_name, lookup in iou_lookup_by_combo.items():
    for pipeline, method in ZCA_MAHALANOBIS_CELLS:
        for eps in EPS_SWEEP:
            mean, std, n = mean_std_iou(lookup, pipeline, eps, method)
            eps_sweep_rows.append(
                {
                    "scale_combo": scale_combo_name,
                    "pipeline": pipeline,
                    "method": method,
                    "eps": eps,
                    "mean_iou": mean,
                    "std_iou": std,
                    "n_combos": n,
                }
            )
eps_sweep_df = pd.DataFrame(eps_sweep_rows)
eps_sweep_df.to_csv(OUTPUT_DIR / "eps_sweep.csv", index=False)
log.info("Wrote %s", OUTPUT_DIR / "eps_sweep.csv")

for scale_combo_name, lookup in iou_lookup_by_combo.items():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for pipeline, method in ZCA_MAHALANOBIS_CELLS:
        means = [mean_std_iou(lookup, pipeline, eps, method)[0] for eps in EPS_SWEEP]
        ax.plot(EPS_SWEEP, means, marker="o", label=f"{pipeline}/{method}")
    ax.set_xscale("log")
    ax.set_xlabel("epsilon (ZCA regularization)")
    ax.set_ylabel("oracle IoU (mean across combos)")
    ax.set_title(
        f"Epsilon sweep — 1-1 (single ref/query pair) — global/bg ZCA whitening and "
        f"Mahalanobis, fg scales={scale_combo_name}"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _eps_path = OUTPUT_DIR / f"eps_sweep__{scale_combo_name}.png"
    fig.savefig(_eps_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved epsilon-sweep curve to %s", _eps_path)

pca_sweep_rows = []
for scale_combo_name, lookup in iou_lookup_by_combo.items():
    for method in METHODS_BY_PIPELINE["pca_truncate"]:
        for k in PCA_K_SWEEP:
            mean, std, n = mean_std_iou(lookup, "pca_truncate", k, method)
            pca_sweep_rows.append(
                {
                    "scale_combo": scale_combo_name,
                    "method": method,
                    "k": k,
                    "mean_iou": mean,
                    "std_iou": std,
                    "n_combos": n,
                }
            )
pca_sweep_df = pd.DataFrame(pca_sweep_rows)
pca_sweep_df.to_csv(OUTPUT_DIR / "pca_k_sweep.csv", index=False)
log.info("Wrote %s", OUTPUT_DIR / "pca_k_sweep.csv")

for scale_combo_name, lookup in iou_lookup_by_combo.items():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for method in METHODS_BY_PIPELINE["pca_truncate"]:
        means = [mean_std_iou(lookup, "pca_truncate", k, method)[0] for k in PCA_K_SWEEP]
        ax.plot(PCA_K_SWEEP, means, marker="o", label=method)
    ax.set_xlabel("k (retained principal components, of C=1024)")
    ax.set_ylabel("oracle IoU (mean across combos)")
    ax.set_title(
        f"PCA truncation — 1-1 (single ref/query pair) — dimensionality sweep, "
        f"fg scales={scale_combo_name}"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _pca_path = OUTPUT_DIR / f"pca_k_sweep__{scale_combo_name}.png"
    fig.savefig(_pca_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved PCA k-sweep curve to %s", _pca_path)

# %% Part 7b — is the PCA_K_SWEEP effect visible in `pca_k_sweep.png` real, or combo-to-combo
# noise? An unpaired bootstrap comparison (see `_shared/stats.py`) of the lowest-vs-highest k's
# per-combo oracle_iou arrays, per (scale_combo, method) — referenced in this file's own
# `N_BOOTSTRAP`/`BOOTSTRAP_SEED` parameter comment above ("the PCA_K_SWEEP lowest-vs-highest-k
# significance check"), mirroring resolution_ablation.py Part 9 / training_set_size_ablation.py
# Part 6d.
pca_significance_rows = []
for scale_combo_name, lookup in iou_lookup_by_combo.items():
    for method in METHODS_BY_PIPELINE["pca_truncate"]:
        lo_vals = np.array(list(lookup["pca_truncate"][PCA_K_SWEEP[0]][method].values()))
        hi_vals = np.array(list(lookup["pca_truncate"][PCA_K_SWEEP[-1]][method].values()))
        if lo_vals.size == 0 or hi_vals.size == 0:
            continue
        prob_hi_greater = bootstrap_prob_greater(
            hi_vals, lo_vals, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
        )
        pca_significance_rows.append(
            {
                "scale_combo": scale_combo_name,
                "method": method,
                "k_lo": PCA_K_SWEEP[0],
                "k_hi": PCA_K_SWEEP[-1],
                "prob_hi_beats_lo": prob_hi_greater,
                "n_lo": int(lo_vals.size),
                "n_hi": int(hi_vals.size),
            }
        )
pca_significance_df = pd.DataFrame(pca_significance_rows)
pca_significance_df.to_csv(OUTPUT_DIR / "pca_k_sweep_significance.csv", index=False)
log.info(
    "PCA-k effect significance (P(k=%d mean > k=%d mean) under %d-resample bootstrap; near 0.5 "
    "= indistinguishable from noise):",
    PCA_K_SWEEP[-1],
    PCA_K_SWEEP[0],
    N_BOOTSTRAP,
)
for _, row in pca_significance_df.iterrows():
    log.info(
        "  scale_combo=%-16s method=%-14s P(k=%d beats k=%d)=%.3f (n=%d vs n=%d)",
        row.scale_combo,
        row.method,
        row.k_hi,
        row.k_lo,
        row.prob_hi_beats_lo,
        row.n_hi,
        row.n_lo,
    )
log.info("Wrote %s", OUTPUT_DIR / "pca_k_sweep_significance.csv")

# %% Part 8 — per-instance-type (group) breakdown, per scale combo. The headline chart above
# pools every group together, which can hide a group-specific effect — see
# noisy_fgbg_cleaning.py's identical rationale for its own per-group breakdown.
combos_by_group: dict[str, list[tuple]] = defaultdict(list)
for combo in combos:
    ck = combo_key(combo)
    if ck in bg_raw_lookup:
        combos_by_group[ck[1]].append(ck)

group_summary_frames = []
for scale_combo_name, lookup in iou_lookup_by_combo.items():
    for group, cks in combos_by_group.items():
        group_df = pipeline_method_summary(lookup, combo_keys=set(cks))
        group_df.insert(0, "group", group)
        group_df["scale_combo"] = scale_combo_name
        group_summary_frames.append(group_df)
        _group_slug = group.replace(" ", "_")
        _group_path = OUTPUT_DIR / f"oracle_iou_by_pipeline__{_group_slug}__{scale_combo_name}.png"
        plot_pipeline_bar_chart(
            group_df,
            f"Feature-space transforms — 1-1 (single ref/query pair) — oracle IoU, "
            f"group={group}, fg scales={scale_combo_name} (n={len(cks)} combos)",
            _group_path,
        )
pd.concat(group_summary_frames, ignore_index=True).to_csv(
    OUTPUT_DIR / "oracle_iou_by_pipeline_per_group.csv", index=False
)
log.info(
    "Saved %d per-group breakdown charts and %s",
    len(combos_by_group) * len(SCALE_COMBOS),
    OUTPUT_DIR / "oracle_iou_by_pipeline_per_group.csv",
)

# %% Part 9 — qualitative figure: every pipeline's score map for one focus combo, per scale
# combo (fg gallery differs by scale combo, so the score maps do too).
focus_combo = next(
    (
        c
        for c in combos
        if c["unit"] == FOCUS_UNIT
        and c["class"] == FOCUS_CLASS
        and c["instance_id"] == FOCUS_INSTANCE_ID
    ),
    combos[0],
)
focus_ck = combo_key(focus_combo)
if (focus_combo["unit"], focus_combo["class"], focus_combo["instance_id"]) != (
    FOCUS_UNIT,
    FOCUS_CLASS,
    FOCUS_INSTANCE_ID,
):
    log.warning(
        "Focus combo %s not found under RUN_PAIRS — falling back to %s",
        (FOCUS_UNIT, FOCUS_CLASS, FOCUS_INSTANCE_ID),
        focus_ck,
    )

if focus_ck not in bg_raw_lookup:
    log.warning("Focus combo %s has no usable bg gallery — skipping qualitative figure", focus_ck)
else:
    focus_unit, focus_group = focus_ck[0], focus_ck[1]
    focus_gt = gt_patch_masks[(focus_unit, focus_group)]
    focus_q_raw, focus_q_h, focus_q_w = query_raw_encodings[focus_unit]

    for scale_combo_name in SCALE_COMBOS:
        if (focus_ck, scale_combo_name) not in fg_raw_lookup:
            log.warning(
                "Focus combo %s has no usable fg gallery for scale combo %r — skipping",
                focus_ck,
                scale_combo_name,
            )
            continue
        # Re-scores the focus combo a second time (Part 5 already covered it) — deterministic
        # given the same inputs, and keeps this section self-contained without threading every
        # combo's raw maps through the whole run just for one figure.
        focus_raw_maps = score_combo(
            focus_ck,
            fg_raw_lookup[(focus_ck, scale_combo_name)].to(focus_q_raw.device),
            bg_raw_lookup[focus_ck].to(focus_q_raw.device),
            focus_q_raw,
            focus_q_h,
            focus_q_w,
            focus_gt,
            iou_lookup_by_combo[scale_combo_name],
        )

        query_img = query_images[focus_unit]
        n_panels = 1 + len(PIPELINES)
        n_cols = 3
        n_rows = -(-n_panels // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.2 * n_rows))
        axes_flat = axes.reshape(-1)

        axes_flat[0].imshow(query_img)
        gt_overlay = np.zeros((*focus_gt.shape, 4))
        gt_overlay[focus_gt] = (0.2, 0.8, 0.2, 0.45)
        axes_flat[0].imshow(gt_overlay, extent=(0, query_img.width, query_img.height, 0))
        axes_flat[0].set_title("query + GT")
        axes_flat[0].axis("off")

        for i, pipeline in enumerate(PIPELINES, start=1):
            # The last method in each pipeline's list is the one the pipeline is actually
            # about (single_proto is every cosine pipeline's simpler baseline, already its
            # own bar).
            method = METHODS_BY_PIPELINE[pipeline][-1]
            param = overall_best_param_by_combo[scale_combo_name][pipeline][method]
            raw = focus_raw_maps[(pipeline, param, method)]
            im = axes_flat[i].imshow(raw, cmap="magma")
            if isinstance(param, float):
                param_str = f"\neps={param:.0e}"
            elif isinstance(param, int):
                param_str = f"\nk={param}"
            else:
                param_str = ""
            axes_flat[i].set_title(f"{pipeline} ({method}){param_str}", fontsize=9)
            axes_flat[i].axis("off")
            plt.colorbar(im, ax=axes_flat[i], fraction=0.046)

        for j in range(n_panels, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(
            f"Feature-transform score maps — 1-1 (single ref/query pair) — focus combo "
            f"{focus_ck}, fg scales={scale_combo_name}"
        )
        fig.tight_layout()
        _focus_path = OUTPUT_DIR / f"focus_combo_pipeline_grid__{scale_combo_name}.png"
        fig.savefig(_focus_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved qualitative focus-combo grid to %s", _focus_path)

# %% Part 10 — 5-3 pooled gallery, cross-validated: does ZCA/PCA/LDA/Mahalanobis's ranking
# hold when the gallery is built from 5 pooled training images instead of one reference
# image? Reuses `score_combo`/`split_fg_bg_patches_raw`/`best_param`/`mean_std_iou` unchanged
# — only discovery, pooling, and fold/role assignment are new (see
# `_shared/pooled_gallery_cv.py` for why folds use a fresh random shuffle rather than a fixed
# image order, which was a real dataset-of-origin confound in `training_set_size_ablation.py`'s
# first version). The existing 1-1 combos/results above are untouched by this section.
PART_TYPES_53 = PART_TYPES
discovery_53 = discover_all_instances(DATA_ROOT, "abc5", PART_TYPES_53)

usable_instances_53: list = []
for inst in tqdm(discovery_53.instances, desc="5-3: building close/mid crops"):
    img = discovery_53.images[(inst.part_type, inst.image_number)]
    crops: dict = {}
    ok = True
    for scale in SCALES:  # ["close", "mid"] — "global" is derived per-instance below
        x0, y0, x1, y1 = scale_crop_box(inst.mask, scale, CROP_PADDING_FRACTION)
        if x1 - x0 < MIN_CROP_SIZE or y1 - y0 < MIN_CROP_SIZE:
            ok = False
            break
        crops[scale] = {
            "img": img.crop((x0, y0, x1, y1)),
            "mask_px": inst.mask[y0:y1, x0:x1],
            "bg_exclude_mask_px": inst.bg_exclude_mask[y0:y1, x0:x1],
        }
    if not ok:
        continue
    inst.crops = crops
    usable_instances_53.append(inst)
log.info("5-3: usable instances %d/%d", len(usable_instances_53), len(discovery_53.instances))

# Raw full-image encodings — reused both as each instance's own "global" fg/bg source (same
# role as Part 3.6's ref_raw_encodings) and as every image's query/eval tokens (Part 3's role).
image_raw_encodings_53: dict[tuple[str, int], tuple[torch.Tensor, int, int]] = {}
for key in tqdm(sorted(discovery_53.images), desc="5-3: encoding images (raw)"):
    out = encoder(discovery_53.images[key], layers=[LAYER_IDX], debias=True)
    patches = out.patches[:, 0]
    h, w = patches.shape[1], patches.shape[2]
    image_raw_encodings_53[key] = (patches[0].reshape(h * w, -1).float(), h, w)

gt_patch_masks_53: dict[tuple[str, str, int], np.ndarray] = {}
for (part_type, group, n), pixel_mask in discovery_53.gt_masks.items():
    _, h, w = image_raw_encodings_53[(part_type, n)]
    gt_patch_masks_53[(part_type, group, n)] = pixel_mask_to_patch_mask(
        pixel_mask, h, w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )

# Encode close/mid crops (raw) + derive each instance's own "global" fg/bg from its own
# image's full raw encoding — same per-instance/scale bank pattern as Part 4, generalized off
# combos onto every discovered instance.
fg_by_inst_scale_53: dict[tuple[int, str], torch.Tensor] = {}
bg_by_inst_scale_53: dict[tuple[int, str], torch.Tensor] = {}

clean_items_53: list[tuple] = []
for i, inst in enumerate(usable_instances_53):
    for scale, crop in inst.crops.items():
        clean_items_53.append((i, scale, crop["img"], crop["mask_px"], crop["bg_exclude_mask_px"]))

for i in tqdm(range(0, len(clean_items_53), chunk_size), desc="5-3: encoding crops (raw)"):
    chunk = clean_items_53[i : i + chunk_size]
    out = encoder([c[2] for c in chunk], layers=[LAYER_IDX], debias=True)
    chunk_patches = out.patches[:, 0]
    grid_h, grid_w = chunk_patches.shape[1], chunk_patches.shape[2]
    for (idx, scale, _, mask_px, bg_exclude_mask_px), patch_tokens in zip(chunk, chunk_patches):
        inst = usable_instances_53[idx]
        fg, bg = split_fg_bg_patches_raw(
            patch_tokens,
            mask_px,
            grid_h,
            grid_w,
            f"5-3/{inst.part_type}/{inst.group}/img#{inst.image_number}/"
            f"inst{inst.instance_id}/{scale}",
            bg_exclude_mask_px=bg_exclude_mask_px,
        )
        fg_by_inst_scale_53[(idx, scale)] = fg.cpu()
        bg_by_inst_scale_53[(idx, scale)] = bg.cpu()

for i, inst in enumerate(usable_instances_53):
    r_tokens, r_h, r_w = image_raw_encodings_53[(inst.part_type, inst.image_number)]
    fg_patch_mask = pixel_mask_to_patch_mask(inst.mask, r_h, r_w, IMG_SIZE, MASK_PATCH_THRESHOLD)
    fg_flat = torch.from_numpy(fg_patch_mask.reshape(-1)).to(r_tokens.device)
    global_fg = r_tokens[fg_flat]
    if global_fg.shape[0] == 0:
        global_fg = r_tokens
    fg_by_inst_scale_53[(i, "global")] = global_fg.cpu()

    exclude_patch_mask = pixel_mask_to_patch_mask(
        inst.bg_exclude_mask, r_h, r_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )
    exclude_flat = torch.from_numpy(exclude_patch_mask.reshape(-1)).to(r_tokens.device)
    global_bg = r_tokens[~exclude_flat]
    if global_bg.shape[0] == 0:
        global_bg = r_tokens
    bg_by_inst_scale_53[(i, "global")] = global_bg.cpu()

instances_by_pg_53: dict[tuple[str, str], list[int]] = defaultdict(list)
for i, inst in enumerate(usable_instances_53):
    instances_by_pg_53[(inst.part_type, inst.group)].append(i)
groups_by_pt_53: dict[str, list[str]] = defaultdict(list)
for pt, g in instances_by_pg_53:
    groups_by_pt_53[pt].append(g)
log.info(
    "5-3: built raw fg/bg banks for %d instances across %d (part_type, group) pairs",
    len(usable_instances_53),
    len(instances_by_pg_53),
)

# Sweep: for each fold x part_type x group, pool the fold's training instances into one
# gallery, score with the same score_combo() every real combo uses. Restricted to this
# file's own report-headline scale combo (global+mid — bg_zca's 0.848) rather than
# re-sweeping all 3 SCALE_COMBOS: with pooled galleries already ~5x a single-image
# gallery's size, sweeping every scale combo here would triple an already-expensive stage
# for a question (which scale combo wins) this section isn't asking — see MAX_BANK_SIZE_TRANSFORM_53
# below for the other half of keeping this section's cost bounded.
HEADLINE_SCALE_COMBO_53 = "global+mid"
fold_splits_53 = make_fold_role_splits(PART_TYPES_53)  # truly randomized, not SEED-reproducible
iou_lookup_53 = new_iou_lookup()
n_pooled_samples_53 = 0

n_units_53 = N_FOLDS_53 * len(PART_TYPES_53)
with tqdm(total=n_units_53, desc="5-3: cross-validated fit + score") as pbar:
    for fold_idx, split in enumerate(fold_splits_53):
        for part_type in PART_TYPES_53:
            train_numbers, eval_numbers = split[part_type]
            for group in groups_by_pt_53.get(part_type, []):
                idxs = instances_by_pg_53[(part_type, group)]
                pool_idxs = [
                    i for i in idxs if usable_instances_53[i].image_number in train_numbers
                ]
                if not pool_idxs:
                    continue
                bg_raw_53 = cap_bank_size(
                    torch.cat(
                        [
                            bg_by_inst_scale_53[(i, s)]
                            for i in pool_idxs
                            for s in (*SCALES, "global")
                        ],
                        dim=0,
                    ),
                    MAX_BANK_SIZE_TRANSFORM_53,
                    SEED,
                )
                fg_raw_53 = cap_bank_size(
                    torch.cat(
                        [
                            fg_by_inst_scale_53[(i, s)]
                            for i in pool_idxs
                            for s in SCALE_COMBOS[HEADLINE_SCALE_COMBO_53]
                        ],
                        dim=0,
                    ),
                    MAX_BANK_SIZE_TRANSFORM_53,
                    SEED,
                )
                if fg_raw_53.shape[0] == 0 or bg_raw_53.shape[0] == 0:
                    continue
                for eval_number in eval_numbers:
                    key = (part_type, group, eval_number)
                    if key not in gt_patch_masks_53:
                        continue
                    q_raw, q_h, q_w = image_raw_encodings_53[(part_type, eval_number)]
                    gt = gt_patch_masks_53[key]
                    pseudo_ck = (f"fold{fold_idx}", part_type, group, eval_number)
                    score_combo(
                        pseudo_ck,
                        fg_raw_53.to(q_raw.device),
                        bg_raw_53.to(q_raw.device),
                        q_raw,
                        q_h,
                        q_w,
                        gt,
                        iou_lookup_53,
                    )
                    n_pooled_samples_53 += 1
            pbar.update(1)

log.info("5-3: scoring complete, %d pooled-gallery samples scored", n_pooled_samples_53)

# Headline comparison: 1-1 (existing combos) vs 5-3 (pooled), per pipeline/method, at
# HEADLINE_SCALE_COMBO_53.
comparison_53_rows = []
for pipeline in PIPELINES:
    for method in METHODS_BY_PIPELINE[pipeline]:
        param_11 = best_param(iou_lookup_by_combo[HEADLINE_SCALE_COMBO_53], pipeline, method)
        mean_11, std_11, n_11 = mean_std_iou(
            iou_lookup_by_combo[HEADLINE_SCALE_COMBO_53], pipeline, param_11, method
        )
        param_53 = best_param(iou_lookup_53, pipeline, method)
        mean_53, std_53, n_53 = mean_std_iou(iou_lookup_53, pipeline, param_53, method)
        delta = mean_53 - mean_11 if not (np.isnan(mean_11) or np.isnan(mean_53)) else float("nan")
        comparison_53_rows.append(
            {
                "pipeline": pipeline,
                "method": method,
                "iou_1_1": mean_11,
                "std_1_1": std_11,
                "n_1_1": n_11,
                "param_1_1": param_11,
                "iou_5_3": mean_53,
                "std_5_3": std_53,
                "n_5_3": n_53,
                "param_5_3": param_53,
                "delta": delta,
            }
        )
comparison_53_df = pd.DataFrame(comparison_53_rows)
_comparison_53_csv = OUTPUT_DIR / f"comparison_1_1_vs_5_3__{HEADLINE_SCALE_COMBO_53}.csv"
comparison_53_df.to_csv(_comparison_53_csv, index=False)
log.info("1-1 vs 5-3 comparison (fg scales=%s):", HEADLINE_SCALE_COMBO_53)
for _, row in comparison_53_df.iterrows():
    log.info(
        "  %-16s %-14s 1-1=%.3f+/-%.3f (n=%d)  5-3=%.3f+/-%.3f (n=%d)  delta=%+.3f",
        row.pipeline,
        row.method,
        row.iou_1_1,
        row.std_1_1,
        row.n_1_1,
        row.iou_5_3,
        row.std_5_3,
        row.n_5_3,
        row.delta,
    )

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(comparison_53_df))
width = 0.35
ax.bar(
    x - width / 2,
    comparison_53_df["iou_1_1"],
    width,
    yerr=comparison_53_df["std_1_1"],
    capsize=3,
    label="1-1 (existing)",
    color="#7f8c8d",
)
ax.bar(
    x + width / 2,
    comparison_53_df["iou_5_3"],
    width,
    yerr=comparison_53_df["std_5_3"],
    capsize=3,
    label=f"5-3 (pooled, {N_FOLDS_53}-fold CV)",
    color="#2ecc71",
)
ax.set_xticks(
    x,
    [f"{p}\n({m})" for p, m in zip(comparison_53_df["pipeline"], comparison_53_df["method"])],
    rotation=30,
    ha="right",
    fontsize=8,
)
ax.set_ylabel("oracle IoU (mean +/- std)")
ax.set_title(f"1-1 vs. 5-3 pooled gallery, fg scales={HEADLINE_SCALE_COMBO_53}")
ax.set_ylim(0, 1.0)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
_comparison_53_png = OUTPUT_DIR / f"comparison_1_1_vs_5_3__{HEADLINE_SCALE_COMBO_53}.png"
fig.savefig(_comparison_53_png, dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved %s and %s", _comparison_53_csv, _comparison_53_png)

# %% Part 11 — latency/throughput summary: every phase timed above (query/ref image encode,
# gallery crop encode, scoring) was already collected into `latency_rows` as it ran (see the
# "Latency (Addition 2)" comment near Part 3); written out here as a CSV. No curve plot — the
# 1-1 section's phases each run once rather than sweeping a resolution/size/N_train axis (the
# `endpoint` column distinguishes the 1-1 rows collected here from a future 5-3 timing pass), so
# a per-phase table is the useful artifact, not a curve.
latency_df = pd.DataFrame(latency_rows)
latency_df.to_csv(OUTPUT_DIR / "latency.csv", index=False)
log.info("Wrote %s", OUTPUT_DIR / "latency.csv")
for _, row in latency_df.iterrows():
    log.info("  phase=%-24s elapsed=%.1fs n_units=%d", row.phase, row.elapsed_s, row.n_units)

# %% Part 12 — worst/best-N qualitative gallery at one representative point (see
# QUALITATIVE_SCALE_COMBO/QUALITATIVE_PIPELINE/QUALITATIVE_PARAM/QUALITATIVE_METHOD above) —
# every other figure in this script averages across combos; this shows actual individual query
# images so a failure mode is visible instead of washed out by the mean.
if qualitative_examples:
    save_score_gallery(
        qualitative_examples,
        OUTPUT_DIR / "qualitative_worst_best.png",
        n=5,
        score_name="oracle_iou",
        title=(
            f"Worst/best oracle_iou examples: scale_combo={QUALITATIVE_SCALE_COMBO} "
            f"pipeline={QUALITATIVE_PIPELINE} param={QUALITATIVE_PARAM} "
            f"method={QUALITATIVE_METHOD}"
        ),
    )
    log.info(
        "Wrote %s (%d examples)",
        OUTPUT_DIR / "qualitative_worst_best.png",
        len(qualitative_examples),
    )
else:
    log.warning(
        "No qualitative examples collected for the representative point "
        "(scale_combo=%s, pipeline=%s, param=%s, method=%s)",
        QUALITATIVE_SCALE_COMBO,
        QUALITATIVE_PIPELINE,
        QUALITATIVE_PARAM,
        QUALITATIVE_METHOD,
    )

# %% [markdown]
# ## Reading the results
#
# - **LDA's 1-D degeneracy**: `lda`'s score is a single signed scalar per patch, not a
#   cosine similarity — it isn't directly comparable in scale to the other pipelines' scores,
#   only in the oracle-IoU it achieves. A strong `lda` result says the fg/bg boundary is
#   well-approximated by *one* linear direction found from this combo's own patches; a weak
#   one doesn't rule out a better nonlinear or higher-rank boundary existing.
# - **`bg_zca` vs. `mahalanobis`**: these two pipelines share the same fit (mean, eigh, and
#   epsilon grid) and differ only in whether the final L2-normalize is applied. Their IoU gap
#   at matched epsilon is the cleanest read in this experiment of whether keeping
#   distance-from-background *magnitude* (Mahalanobis) beats discarding it for pure direction
#   (cosine similarity after whitening).
# - **Per-combo rank deficiency**: pooled patch counts here (a few hundred to ~1500) are
#   typically well below C=1024. Every ZCA/PCA/LDA fit above is therefore working with a
#   covariance whose eigenvalues below that rank are ~0 numerically — `eps` and LDA's
#   shrinkage are what keep those directions from dominating the result, not real learned
#   structure. A pipeline that only wins at the largest epsilon in `EPS_SWEEP` is likely
#   benefiting mostly from this regularization smoothing out noise, not from whitening real
#   signal — read the full `eps_sweep.png` curve, not just the headline bar, before trusting
#   a ZCA/Mahalanobis result.
# - As with every other fundamental experiment here, the headline chart pools every
#   part-type/group/instance combo together — check `oracle_iou_by_pipeline__<group>.png`
#   before concluding a pipeline's aggregate win holds for every instance-type group, and use
#   `focus_combo_pipeline_grid.png` only as one qualitative example, not as the dataset.
# - **`size_correlation.csv`/`.png`** — does oracle IoU correlate with the query GT's own area
#   fraction (`gt_area_frac`, already a column of `oracle_iou_per_combo.csv`), per (scale_combo,
#   pipeline, method)? Same aggregation-can-hide-an-effect caveat as the per-group breakdown,
#   but for object size instead of instance-type group.
# - **`pca_k_sweep_significance.csv`** — an unpaired bootstrap comparison (2000 resamples) of
#   the lowest- vs. highest-k per-combo oracle_iou arrays from `pca_k_sweep.png`, per
#   (scale_combo, method): `prob_hi_beats_lo` near 0.5 means the apparent PCA-k trend is not
#   distinguishable from combo-to-combo noise.
# - **`latency.csv`** — GPU-synchronized wall-clock cost (see `_shared/latency.py`) of the 1-1
#   section's phases (query/ref image encode, gallery crop encode, scoring), logged per phase
#   since each runs once rather than sweeping a resolution/size/N_train axis.
# - **`qualitative_worst_best.png`** — actual worst-5/best-5 query images (crop, raw score map,
#   GT mask) at one representative (scale_combo, pipeline, param, method) point, not an
#   average — `focus_combo_pipeline_grid.png` shows every pipeline for one combo; this shows
#   one pipeline/method across many combos, so a failure mode specific to one query image is
#   visible instead of washed out by the mean.
