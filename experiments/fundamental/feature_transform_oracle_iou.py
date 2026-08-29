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
from tqdm import tqdm

from dinoisawesome import DinoEncoder, EncoderWithCache, compute_exemplar_features, load_annotations
from dinoisawesome.abc3 import INSTANCE_TYPE_GROUPS, PART_TYPES, available_instance_groups

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
from _shared.prototype_ops import knn_fgbg_score  # noqa: E402
from _shared.thresholding import iou_threshold_curve  # noqa: E402

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

data_dir = _REPO_ROOT / "data" / "abc3"

REF_NUMBER = 1
QUERY_NUMBER = 2

# Narrow for fast iteration, e.g. ["LHa"] — same knob as the sibling fundamental scripts.
RUN_PART_TYPES: list[str] = PART_TYPES

# Same focus combo as the sibling fundamental scripts, for direct comparability of figures.
FOCUS_PART_TYPE = "LHa"
FOCUS_CLASS = "donut foam single"
FOCUS_INSTANCE_ID = 1

DINO_VERSION = "v3"
DINO_SIZE = "large"
IMG_SIZE = 768
LAYER_IDX = 23
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")
DINO_ENCODING_CACHE_DIR: str | None = os.environ.get("DINO_ENCODING_CACHE_DIR")

MASK_PATCH_THRESHOLD = 0.3
CROP_PADDING_FRACTION = 1.0
MIN_CROP_SIZE = 128

ORACLE_THRESHOLD_STEPS = 25
SCALES: list[str] = ["close", "mid"]  # crop scales; "global" (full ref image) is bg-only, Part 3.6

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

SEED = 0
torch.manual_seed(SEED)

OUTPUT_DIR = _REPO_ROOT / "outputs" / "fundamental" / "feature_transform_oracle_iou"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log.info(
    "RUN_PART_TYPES=%s ref_number=%d query_number=%d  |  DINO%s-%s img_size=%d layer=%d  |  "
    "pipelines=%s eps_sweep=%s pca_k_sweep=%s",
    RUN_PART_TYPES,
    REF_NUMBER,
    QUERY_NUMBER,
    DINO_VERSION,
    DINO_SIZE,
    IMG_SIZE,
    LAYER_IDX,
    PIPELINES,
    EPS_SWEEP,
    PCA_K_SWEEP,
)

# %% Helpers shared across discovery / scoring / aggregation / plotting


def combo_key(d: dict) -> tuple[str, str, str, int]:
    """(part_type, instance_type group, class, instance_id) — a combo's stable identity."""
    return (d["part_type"], d["group"], d["class"], d["instance_id"])


def oracle_iou(raw: np.ndarray, gt_mask: np.ndarray, steps: int) -> float:
    """Best patch-mask IoU any single global threshold on *raw* could achieve against
    *gt_mask* — see augmented_prototype_oracle_iou.py's identical helper for rationale."""
    _, ious = iou_threshold_curve(raw, gt_mask, steps)
    return float(ious.max())


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


def score_heatmap(tokens: torch.Tensor, prototype: torch.Tensor, h: int, w: int) -> np.ndarray:
    """single_proto: cosine-similarity heatmap, prototype vs. every query patch."""
    return (tokens @ prototype.T).reshape(h, w).cpu().float().numpy()


def knn_score_heatmap(
    tokens: torch.Tensor, fg_bank: torch.Tensor, bg_bank: torch.Tensor, k: int, h: int, w: int
) -> np.ndarray:
    return knn_fgbg_score(tokens, fg_bank, bg_bank, k).reshape(h, w)


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
) -> dict[tuple[str, Any, str], np.ndarray]:
    """Fit every pipeline's parameters from this combo's own (fg_raw, bg_raw), score the
    query against each pipeline x method x swept-param combination, store each oracle IoU in
    `lookup[pipeline][param][method][ck]`, and return every raw score map keyed by
    (pipeline, param, method) — reused by the qualitative figures below so the focus combo
    never needs rescoring from scratch. Runs `fit_cov_eigh` exactly once for the global pool
    and once for the bg-only pool; every swept eps/k value below reuses those two
    eigendecompositions (see `_shared.feature_transforms.fit_cov_eigh`)."""
    raw_maps: dict[tuple[str, Any, str], np.ndarray] = {}

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

    for eps in EPS_SWEEP:
        w = zca_matrix(v_bg, ev_bg, eps)
        fg_t = apply_affine(fg_raw, mu_bg, w)
        bg_t = apply_affine(bg_raw, mu_bg, w)
        q_t = apply_affine(q_raw, mu_bg, w)
        raw_maha = knn_euclid_score_heatmap(q_t, fg_t, bg_t, KNN_FGBG_NUM_NEIGHBOURS, q_h, q_w)
        raw_maps[("mahalanobis", eps, "mahalanobis_knn")] = raw_maha
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

for part_type in tqdm(RUN_PART_TYPES, desc="Discovering combos"):
    ref_stem = f"{part_type}_{REF_NUMBER}"
    query_stem = f"{part_type}_{QUERY_NUMBER}"
    ref_ann_path = data_dir / "annotations" / ref_stem
    query_ann_path = data_dir / "annotations" / query_stem

    groups = available_instance_groups(ref_ann_path)
    if not groups:
        log.warning("part_type=%s: no instance-type groups annotated — skipping", part_type)
        continue

    ref_anns = load_annotations(ref_ann_path)
    query_anns = load_annotations(query_ann_path)
    ref_images[part_type] = Image.open(data_dir / f"{ref_stem}.jpg").convert("RGB")
    query_images[part_type] = Image.open(data_dir / f"{query_stem}.jpg").convert("RGB")

    for group in groups:
        classes = INSTANCE_TYPE_GROUPS[group]
        ref_group_anns = [a for a in ref_anns if a["class"] in classes]
        query_group_anns = [a for a in query_anns if a["class"] in classes]
        if not query_group_anns:
            log.warning("part_type=%s group=%s: no query GT instances — skipping", part_type, group)
            continue
        group_query_masks[(part_type, group)] = np.stack([a["mask"] for a in query_group_anns]).any(
            axis=0
        )
        if ref_group_anns:
            group_ref_masks[(part_type, group)] = np.stack([a["mask"] for a in ref_group_anns]).any(
                axis=0
            )
        for ref_ann in ref_group_anns:
            combos.append(
                {
                    "part_type": part_type,
                    "group": group,
                    "class": ref_ann["class"],
                    "instance_id": ref_ann["instance_id"],
                    "ref_mask": ref_ann["mask"],
                }
            )

if not combos:
    raise RuntimeError(
        f"No combos discovered for RUN_PART_TYPES={RUN_PART_TYPES} — every part type was "
        "skipped (no annotated instance-type groups or no query GT)."
    )
log.info(
    "Discovered %d (part_type, group, instance) combos across %d part types",
    len(combos),
    len({c["part_type"] for c in combos}),
)

# %% Part 2 — build close/mid crops per combo
scales_by_ck: dict[tuple, list[str]] = defaultdict(list)
for combo in tqdm(combos, desc="Building close/mid crops"):
    ref_img = ref_images[combo["part_type"]]
    group_mask = group_ref_masks.get((combo["part_type"], combo["group"]), combo["ref_mask"])
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

query_raw_encodings: dict[str, tuple[torch.Tensor, int, int]] = {}
for part_type in tqdm(sorted(query_images), desc="Encoding query images (raw)"):
    out = encoder(query_images[part_type], layers=[LAYER_IDX], debias=True)
    patches = out.patches[:, 0]  # (1, H, W, D)
    q_h, q_w = patches.shape[1], patches.shape[2]
    query_raw_encodings[part_type] = (patches[0].reshape(q_h * q_w, -1).float(), q_h, q_w)

gt_patch_masks: dict[tuple[str, str], np.ndarray] = {}
for (part_type, group), pixel_mask in group_query_masks.items():
    _, q_h, q_w = query_raw_encodings[part_type]
    gt_patch_masks[(part_type, group)] = pixel_mask_to_patch_mask(
        pixel_mask, q_h, q_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )

# %% Part 3.6 — RAW full-reference-image patch tokens: the "global" bg source
# One full-image encode per part type, reused as every one of that part type's combos'
# extra bg source below (Part 4) — same role as the sibling fundamental scripts' "global"
# scale, just kept raw here instead of normalized.
ref_raw_encodings: dict[str, tuple[torch.Tensor, int, int]] = {}
for part_type in tqdm(sorted(ref_images), desc="Encoding ref images (raw, global bg source)"):
    out = encoder(ref_images[part_type], layers=[LAYER_IDX], debias=True)
    patches = out.patches[:, 0]
    r_h, r_w = patches.shape[1], patches.shape[2]
    ref_raw_encodings[part_type] = (patches[0].reshape(r_h * r_w, -1).float(), r_h, r_w)

# %% Part 4 — encode every combo's close/mid crops (RAW), pool per-combo fg/bg galleries
# fg = close+mid fg patches; bg = close+mid bg patches + this combo's own "global" bg
# (Part 3.6's full ref-image tokens, minus the whole instance-type group's ref mask) — the
# same multiscale pooling augmented_prototype_oracle_iou_knn_fgbg.py's Part 3.5 builds, kept
# raw here instead of L2-normalized.
fg_by_scale_raw: dict[tuple, torch.Tensor] = {}
bg_by_scale_raw: dict[tuple, torch.Tensor] = {}

clean_items: list[tuple] = []
for combo in combos:
    ck = combo_key(combo)
    for scale, crop in combo["crops"].items():
        clean_items.append((ck, scale, crop["img"], crop["mask_px"], crop["bg_exclude_mask_px"]))

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
        fg_by_scale_raw[(ck, scale)] = fg
        bg_by_scale_raw[(ck, scale)] = bg

combo_bg_global_raw: dict[tuple, torch.Tensor] = {}
for combo in combos:
    ck = combo_key(combo)
    part_type, group = ck[0], ck[1]
    r_tokens_raw, r_h, r_w = ref_raw_encodings[part_type]
    exclude_mask_px = group_ref_masks.get((part_type, group), combo["ref_mask"])
    exclude_patch_mask = pixel_mask_to_patch_mask(
        exclude_mask_px, r_h, r_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )
    exclude_flat = torch.from_numpy(exclude_patch_mask.reshape(-1)).to(r_tokens_raw.device)
    global_bg = r_tokens_raw[~exclude_flat]
    if global_bg.shape[0] == 0:
        log.warning("%s: global bg mask empty after patch-grid projection — using all patches", ck)
        global_bg = r_tokens_raw
    combo_bg_global_raw[ck] = global_bg

fg_raw_lookup: dict[tuple, torch.Tensor] = {}
bg_raw_lookup: dict[tuple, torch.Tensor] = {}
for ck, scales in scales_by_ck.items():
    fg_raw_lookup[ck] = torch.cat([fg_by_scale_raw[(ck, s)] for s in scales], dim=0)
    bg_raw_lookup[ck] = torch.cat(
        [bg_by_scale_raw[(ck, s)] for s in scales] + [combo_bg_global_raw[ck]], dim=0
    )

log.info("Built raw fg/bg galleries for %d combos", len(fg_raw_lookup))

# %% Part 5 — main per-combo fit + score sweep
iou_lookup: IouLookup = {}
for pipeline in PIPELINES:
    params = SWEPT_PIPELINES.get(pipeline, [None])
    iou_lookup[pipeline] = {
        param: {method: {} for method in METHODS_BY_PIPELINE[pipeline]} for param in params
    }

for combo in tqdm(combos, desc="Part 5: fitting + scoring"):
    ck = combo_key(combo)
    part_type, group = ck[0], ck[1]
    if ck not in fg_raw_lookup:
        continue
    fg_raw, bg_raw = fg_raw_lookup[ck], bg_raw_lookup[ck]
    if fg_raw.shape[0] == 0 or bg_raw.shape[0] == 0:
        log.warning("%s: empty fg/bg raw gallery — skipping", ck)
        continue
    gt = gt_patch_masks.get((part_type, group))
    if gt is None:
        continue
    q_raw, q_h, q_w = query_raw_encodings[part_type]
    score_combo(ck, fg_raw, bg_raw, q_raw, q_h, q_w, gt, iou_lookup)

log.info("Scoring complete: %d combos x %d pipelines", len(fg_raw_lookup), len(PIPELINES))

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
    lookup: IouLookup, combo_keys: set[tuple] | None = None
) -> pd.DataFrame:
    rows = []
    for pipeline in PIPELINES:
        for method in METHODS_BY_PIPELINE[pipeline]:
            param = best_param(lookup, pipeline, method)
            mean, std, n = mean_std_iou(lookup, pipeline, param, method, combo_keys)
            rows.append(
                {
                    "pipeline": pipeline,
                    "method": method,
                    "param": param,
                    "mean_iou": mean,
                    "std_iou": std,
                    "n_combos": n,
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


summary_df = pipeline_method_summary(iou_lookup)
log.info(
    "Oracle-IoU summary (mean +/- std across %d combos, best swept param per pipeline):",
    len(fg_raw_lookup),
)
for _, row in summary_df.iterrows():
    log.info(
        "  %-16s %-14s param=%-10s iou=%.3f+/-%.3f (n=%d)",
        row.pipeline,
        row.method,
        row.param,
        row.mean_iou,
        row.std_iou,
        row.n_combos,
    )

_headline_path = OUTPUT_DIR / "oracle_iou_by_pipeline.png"
plot_pipeline_bar_chart(
    summary_df,
    f"Feature-space transforms — oracle IoU by pipeline "
    f"({len(fg_raw_lookup)} combos across {len(RUN_PART_TYPES)} part types)",
    _headline_path,
)
log.info("Saved headline bar chart to %s", _headline_path)

overall_best_param: dict[str, dict[str, Any]] = {
    pipeline: {
        method: best_param(iou_lookup, pipeline, method) for method in METHODS_BY_PIPELINE[pipeline]
    }
    for pipeline in PIPELINES
}

# %% Part 7 — sweep curves: epsilon (global/bg ZCA + Mahalanobis), k (PCA truncation)
fig, ax = plt.subplots(figsize=(8, 5.5))
for pipeline, method in [
    ("global_zca", "single_proto"),
    ("global_zca", "knn_fgbg"),
    ("bg_zca", "single_proto"),
    ("bg_zca", "knn_fgbg"),
    ("mahalanobis", "mahalanobis_knn"),
]:
    means = [mean_std_iou(iou_lookup, pipeline, eps, method)[0] for eps in EPS_SWEEP]
    ax.plot(EPS_SWEEP, means, marker="o", label=f"{pipeline}/{method}")
ax.set_xscale("log")
ax.set_xlabel("epsilon (ZCA regularization)")
ax.set_ylabel("oracle IoU (mean across combos)")
ax.set_title("Epsilon sweep — global/bg ZCA whitening and Mahalanobis")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
_eps_path = OUTPUT_DIR / "eps_sweep.png"
fig.savefig(_eps_path, dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved epsilon-sweep curve to %s", _eps_path)

fig, ax = plt.subplots(figsize=(7, 5.5))
for method in METHODS_BY_PIPELINE["pca_truncate"]:
    means = [mean_std_iou(iou_lookup, "pca_truncate", k, method)[0] for k in PCA_K_SWEEP]
    ax.plot(PCA_K_SWEEP, means, marker="o", label=method)
ax.set_xlabel("k (retained principal components, of C=1024)")
ax.set_ylabel("oracle IoU (mean across combos)")
ax.set_title("PCA truncation — dimensionality sweep")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
_pca_path = OUTPUT_DIR / "pca_k_sweep.png"
fig.savefig(_pca_path, dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved PCA k-sweep curve to %s", _pca_path)

# %% Part 8 — per-instance-type (group) breakdown. The headline chart above pools every
# group together, which can hide a group-specific effect — see noisy_fgbg_cleaning.py's
# identical rationale for its own per-group breakdown.
combos_by_group: dict[str, list[tuple]] = defaultdict(list)
for combo in combos:
    ck = combo_key(combo)
    if ck in fg_raw_lookup:
        combos_by_group[ck[1]].append(ck)

for group, cks in combos_by_group.items():
    group_df = pipeline_method_summary(iou_lookup, combo_keys=set(cks))
    _group_path = OUTPUT_DIR / f"oracle_iou_by_pipeline__{group.replace(' ', '_')}.png"
    plot_pipeline_bar_chart(
        group_df,
        f"Feature-space transforms — oracle IoU, group={group} (n={len(cks)} combos)",
        _group_path,
    )
log.info("Saved %d per-group breakdown charts", len(combos_by_group))

# %% Part 9 — qualitative figure: every pipeline's score map for one focus combo
focus_combo = next(
    (
        c
        for c in combos
        if c["part_type"] == FOCUS_PART_TYPE
        and c["class"] == FOCUS_CLASS
        and c["instance_id"] == FOCUS_INSTANCE_ID
    ),
    combos[0],
)
focus_ck = combo_key(focus_combo)
if (focus_combo["part_type"], focus_combo["class"], focus_combo["instance_id"]) != (
    FOCUS_PART_TYPE,
    FOCUS_CLASS,
    FOCUS_INSTANCE_ID,
):
    log.warning(
        "Focus combo %s not found under RUN_PART_TYPES — falling back to %s",
        (FOCUS_PART_TYPE, FOCUS_CLASS, FOCUS_INSTANCE_ID),
        focus_ck,
    )

if focus_ck not in fg_raw_lookup:
    log.warning(
        "Focus combo %s has no usable fg/bg gallery — skipping qualitative figure", focus_ck
    )
else:
    focus_part_type, focus_group = focus_ck[0], focus_ck[1]
    focus_gt = gt_patch_masks[(focus_part_type, focus_group)]
    focus_q_raw, focus_q_h, focus_q_w = query_raw_encodings[focus_part_type]
    # Re-scores the focus combo a second time (Part 5 already covered it) — deterministic
    # given the same inputs, and keeps this section self-contained without threading every
    # combo's raw maps through the whole run just for one figure.
    focus_raw_maps = score_combo(
        focus_ck,
        fg_raw_lookup[focus_ck],
        bg_raw_lookup[focus_ck],
        focus_q_raw,
        focus_q_h,
        focus_q_w,
        focus_gt,
        iou_lookup,
    )

    query_img = query_images[focus_part_type]
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
        # The last method in each pipeline's list is the one the pipeline is actually about
        # (single_proto is every cosine pipeline's simpler baseline, already its own bar).
        method = METHODS_BY_PIPELINE[pipeline][-1]
        param = overall_best_param[pipeline][method]
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

    fig.suptitle(f"Feature-transform score maps — focus combo {focus_ck}")
    fig.tight_layout()
    _focus_path = OUTPUT_DIR / "focus_combo_pipeline_grid.png"
    fig.savefig(_focus_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved qualitative focus-combo grid to %s", _focus_path)

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
