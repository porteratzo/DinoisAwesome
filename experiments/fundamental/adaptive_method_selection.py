# %% [markdown]
# # Fundamental: Adaptive Method-Selection Logic for the Multi-Scale Ablation
#
# `experiments/object_detection/multiscale_ablation/` currently picks scale, method,
# single-vs-two-stage, and denoising **manually**: `run_experiments.py` computes every
# registered method (and its two-stage variant) for every pair, and a human reads the
# cached P/R/F1/mIoU table or `visualize_results.py`'s heatmaps to decide which config
# is "best" per (part_type, instance_type). This script replaces that manual reading
# with adaptive selection logic that makes the same decisions itself, in order:
#
#   1. **Scale** — global / mid / close / combinations.
#   2. **Method** — prototype (mean) / knn (fg-bg-knn) / kmeans.
#   2.5. **Feature-space transform** (`select_transform`) — no transform vs. `bg_zca`
#      whitening, fit from the train pool's own RAW (pre-L2-normalise) bg tokens and
#      swept over `FEATURE_TRANSFORM_EPS_SWEEP` — the recipe `feature_transform_
#      oracle_iou.py` validated as the single biggest lever across the whole
#      `fundamental/` series (bg_zca + knn_fgbg, +0.047 oracle IoU, eps~=1e-3). Searched
#      with scale/method already locked, against the same single-stage val signal those
#      two steps use — see `select_transform`'s own docstring for why it deliberately
#      does not compose with two-stage's blob rescoring.
#   3. **Single vs. two-stage** — decided from precision/recall of detections (a
#      genuine two-stage improvement) or, in the GT-free branch, from a geometric
#      "objects too small / too close together" proxy — two-stage carries a real
#      latency/complexity cost, so it only wins when it's actually needed.
#   4. **Denoising** — raw fg/bg galleries vs. the best of `cleaning.py`'s cleaning
#      stages (step1 / step2_cls / step2_center).
#   5. **Augmentation** (off by default — `ENABLE_AUGMENTATION_SEARCH`) — no augmentation
#      vs. one family's severities pooled together (rotation / blur / noise); "all"
#      (every family composed at once) is excluded from the search — `[[project-
#      dinoisawesome-fundamental-robustness]]` finding #3 already found that composing
#      every family together doesn't stack, only costs more. By far the most expensive
#      of the 6 steps for a modest, not-always-present gain, so it's skipped (locked to
#      "none", at zero extra cost) unless explicitly enabled.
#
# Two parallel branches make these same decisions with different scoring signals:
#
#   - **GT-calibrated** — scores every candidate with `oracle_iou` (steps 1/2/2.5/4/5) or
#     real precision/recall (step 3), against the held-out *val* set's ground truth.
#   - **GT-free** — never consults any annotation to score a candidate (reference
#     instance masks are still used structurally to build exemplar crops, as they
#     always have been — that's not what "GT-free" judges here). Candidates are scored
#     with unsupervised proxies instead: Otsu separability of the raw score map
#     (steps 1/2/2.5/4), a size/fill-ratio heuristic against the training pool's own
#     median instance size (step 3), and prediction stability under nuisance
#     perturbation (step 5).
#
# **Data: a real train/val/test split, not a single ref/query pair.** Every
# (part_type, instance_type) combo pools *every* annotated image for that combo across
# `data/abc5` (abc3+abc4 merged — see `scripts/build_abc5_dataset.py` — 7-8 images per
# combo) and splits it three ways, as evenly as the pool size allows
# (`split_train_val_test`):
#
#   - **train** — every image's exemplar crops are pooled into *one* gallery per scale
#     (`pool_train_pool_into_gallery`) — a real multi-image train set, not one image's
#     crops. mid/close pool their per-instance crops directly; "global" has no
#     per-instance list, so each image's own whole-image crop is wrapped as its own
#     pooled "cell" instead (see that function's docstring).
#   - **val** — every one of the 5 decision steps scores each candidate against
#     *every* val image and averages (`_score_over_eval_roles` and friends), instead
#     of a single eval image.
#   - **test** — held out for the whole run, touched only once per fold by
#     `evaluate_on_test`, after every decision is locked in. Final metrics are the
#     mean across every test image, then averaged again across CV folds.
#
# Cross-validation reshuffles which images are train vs. val: `build_cv_folds` shuffles
# the combined train+val pool once (seeded per pair) and partitions it into
# `NUM_CV_FOLDS` chunks, each taking a turn as val (standard K-fold with
# `shuffle=True`) — the test pool never enters this, so nothing the report calls "best"
# was ever scored against it before the final, single evaluation.
#
# This is a standalone research script, not a change to `multiscale_ablation`'s
# production pipeline — wiring the winning logic in as a new method there is a
# deliberately separate, later step.

# %% Logging — must be before torch import
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("adaptive_method_selection")

import dataclasses
import os
import sys
import zlib
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from dinoisawesome import DinoEncoder
from dinoisawesome.abc3 import (
    INSTANCE_TYPE_GROUPS,
    PART_TYPES,
    available_instance_groups,
    load_instance_pixel_masks,
)
from dinoisawesome.instance_detection import compute_exemplar_features

_EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_EXPERIMENTS_ROOT))
sys.path.insert(0, str(_EXPERIMENTS_ROOT / "object_detection" / "multiscale_ablation"))

from _shared.augmentations import (  # noqa: E402
    apply_blur,
    apply_noise,
    apply_rotation,
    mean_color,
    pixel_only,
)
from _shared.feature_transforms import (  # noqa: E402
    apply_affine,
    fit_cov_eigh,
    fit_mean,
    zca_matrix,
)
from _shared.mask_geometry import (
    mask_iou,  # noqa: E402
    patch_fg_fraction,  # noqa: E402
)
from _shared.prototype_ops import extract_patch_tokens_batch_with_cls  # noqa: E402
from _shared.thresholding import oracle_iou, otsu_threshold  # noqa: E402
from cleaning import apply_fg_cleaning  # noqa: E402
from common import (  # noqa: E402
    DEFAULT_CROP_CONFIG,
    DEFAULT_SCORING_CONFIG,
    CropConfig,
    ScoringConfig,
    instance_classes_for,
)
from engine import (  # noqa: E402
    ClusterCrop,
    ScalePrototype,
    annotate_cluster_rejection,
    build_all_scale_prototypes,
    dbscan_clusters,
    find_roi_blobs,
    iou_tuned_threshold,
    match_and_score,
    min_cluster_size_bound,
    pixel_mask_to_patch_mask,
    pool_scale_patches,
    score_method,
    tune_cluster_reject_threshold,
    two_stage_predicted_clusters,
)
from methods import MethodState  # noqa: E402

DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")

# %% Parameters
_REPO_ROOT = _EXPERIMENTS_ROOT.parent
ABC5_DATA_DIR = _REPO_ROOT / "data" / "abc5"
OUTPUT_DIR = _REPO_ROOT / "outputs" / "fundamental_abc5" / "adaptive_method_selection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Runnable on one combo or every combo — see main() at the bottom.
FOCUS_PART_TYPE = "LHa"
FOCUS_INSTANCE_TYPE = "donut foam"
RUN_ALL_PAIRS = True

SEED = 0

# A real train/val/test split needs at least 2 images per split; abc5 gives every combo
# 7-8 (see scripts/build_abc5_dataset.py), comfortably above this.
MIN_POOL_SIZE = 6
# (train, val) partitions of the combined train+val pool per pair — see build_cv_folds.
# Higher = more robust selection but scales cost linearly (every one of the 5 decision
# steps' candidate search re-runs per fold); 2 keeps this close to the old script's cost.
NUM_CV_FOLDS = 2

# knn's bg gallery pools essentially "every non-object patch" across every train-pool
# image, instance, and (for step 5's "all" candidate) augmented severity — at full
# img_size=1024 resolution every crop (global AND mid/close) contributes a full
# grid_h*grid_w-patch grid, so this can reach hundreds of thousands of rows and blow both
# the concatenation itself and the query x bg_bank similarity matrix past GPU memory
# (engine.py's pool_scale_patches / knn_fgbg_score's bg_sim). Unlike fg (kept exhaustive
# — every real instance patch matters), bg has enormous natural redundancy, so
# method_state_for pools bg itself (_bg_gallery_capped below), capped at this budget.
MAX_BG_BANK_SIZE = 20_000

# Step "2.5" — feature-space transform (see select_transform). "none" vs. bg_zca whitening
# fit from the train pool's own raw bg tokens, swept over this epsilon grid — matches
# feature_transform_oracle_iou.py's own EPS_SWEEP, the recipe validated there (bg_zca +
# knn_fgbg, +0.047 oracle IoU, eps~=1e-3) as the single biggest lever across the whole
# fundamental/ series.
FEATURE_TRANSFORM_EPS_SWEEP: list[float] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

# Two-stage carries a real latency/complexity cost — only worth it for a genuine F1 gain.
TWO_STAGE_MIN_GAIN = 0.05
# GT-free single-vs-two-stage geometric heuristic thresholds (see geometric_two_stage_decision).
SMALL_OBJECT_FRACTION = 0.5
MERGE_FILL_RATIO_THRESHOLD = 0.3

# Step 5 candidates: each family's severities are pooled together when that family is
# chosen (e.g. "rotation" = rotation@5 AND rotation@10 pooled). "all" (every family
# composed at once) is excluded from the search — see ENABLE_AUGMENTATION_SEARCH below.
AUGMENTATIONS: dict[str, dict] = {
    "rotation": {"values": [5.0, 10.0], "apply": apply_rotation},
    "blur": {"values": [1.0, 2.0], "apply": pixel_only(apply_blur)},
    "noise": {"values": [10.0, 20.0], "apply": pixel_only(partial(apply_noise, seed=SEED))},
}

# Step 5 (augmentation) is by far the most expensive of the 6 decision steps (measured:
# ~15-35s/fold vs. <5s for everything else combined) for a modest, not-always-present
# gain (+0.03 to +0.07 oracle IoU per [[project-dinoisawesome-fundamental-robustness]]
# finding #3) — not worth paying every fold by default. Off by default (augmentation is
# always "none", at zero extra cost — protos_with_augmentation short-circuits on "none");
# flip to True to search it.
ENABLE_AUGMENTATION_SEARCH: bool = False

# GT-free step 5 stability proxy: mild single-severity nuisance perturbations applied to
# the val-image, image; a candidate gallery is "stable" if its predicted fg mask barely
# changes across these.
NUISANCE_PERTURBATIONS = [
    ("rotate", 5.0, apply_rotation),
    ("blur", 1.0, pixel_only(apply_blur)),
    ("noise", 10.0, pixel_only(partial(apply_noise, seed=SEED))),
]

SCALE_COMBOS: dict[str, list[str]] = {
    "global": ["global"],
    "mid": ["mid"],
    "close": ["close"],
    "mid+close": ["mid", "close"],
    "global+mid+close": ["global", "mid", "close"],
}


# %% GT-free proxy signal: Otsu separability
def otsu_separability(raw: np.ndarray) -> float:
    """Otsu between-class variance normalised by total variance — a GT-free proxy for how
    confidently *raw* splits into two populations (fg/bg). ~0 = unimodal/flat, close to 1
    = strongly bimodal. Pure numpy, no new dependency.
    """
    values = raw.astype(np.float64).ravel()
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-8:
        return 0.0
    hist, edges = np.histogram(values, bins=256, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = float(hist.sum())
    total_mean = (hist * centers).sum() / total
    total_var = (hist * (centers - total_mean) ** 2).sum() / total
    if total_var < 1e-12:
        return 0.0
    w0 = np.cumsum(hist).astype(np.float64)
    w1 = total - w0
    sum0 = np.cumsum(hist * centers)
    sum_total = sum0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean0 = np.where(w0 > 0, sum0 / w0, 0.0)
        mean1 = np.where(w1 > 0, (sum_total - sum0) / w1, 0.0)
        between = w0 * w1 * (mean0 - mean1) ** 2 / (total**2)
    return float(np.nanmax(between) / total_var)


def raw_score(
    branch: str, raw: np.ndarray, gt_mask: np.ndarray | None, scoring_cfg: ScoringConfig
) -> float:
    """The swappable step 1/2/4 scorer — oracle_iou (GT-calibrated) or otsu_separability
    (GT-free). Swap this one function for a different metric (spec: "keep it swappable")."""
    if branch == "gt_calibrated":
        assert gt_mask is not None
        return oracle_iou(raw, gt_mask, scoring_cfg.ref_threshold_steps)
    return otsu_separability(raw)


# %% Pool images (train / val / test) — one physical image + its own GT
@dataclass
class RoleImage:
    tag: str
    image: Image.Image
    instance_masks: list[np.ndarray]
    tokens: torch.Tensor
    raw_tokens: torch.Tensor
    grid_h: int
    grid_w: int
    union_pixel_mask: np.ndarray | None
    gt_patch_mask: np.ndarray
    gt_clusters: list[dict]


def _extract_patch_tokens_with_raw(
    encoder: DinoEncoder, image: Image.Image, layer_idx: int, debias: bool = False
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Local mirror of ``dinoisawesome.instance_detection.extract_patch_tokens``, additionally
    keeping the RAW (pre-L2-normalise) token grid from the same forward pass — needed by
    select_transform's bg_zca candidate (see MAX_BG_BANK_SIZE's neighbour,
    FEATURE_TRANSFORM_EPS_SWEEP, and _shared/feature_transforms.py for why raw, not
    normalised). Duplicated locally rather than extending the core library function, which
    is used well outside experiments/.
    """
    out = encoder(image, layers=[layer_idx], debias=debias)
    patches = out.patches[:, 0]  # (B, H, W, D)
    _, h, w, d = patches.shape
    raw = patches[0].reshape(h * w, d)
    tokens = F.normalize(raw, p=2, dim=-1)
    return tokens, raw, h, w


def make_role_image(
    tag: str,
    image: Image.Image,
    instance_masks: list[np.ndarray],
    encoder: DinoEncoder,
    crop_cfg: CropConfig,
) -> RoleImage:
    tokens, raw_tokens, h, w = _extract_patch_tokens_with_raw(
        encoder, image, crop_cfg.layer_idx, debias=crop_cfg.debias
    )
    union = np.stack(instance_masks).any(axis=0) if instance_masks else None
    gt_patch_mask = (
        pixel_mask_to_patch_mask(union, h, w, crop_cfg.img_size, crop_cfg.mask_patch_threshold)
        if union is not None
        else np.zeros((h, w), dtype=bool)
    )
    gt_clusters = [
        {
            "mask": pixel_mask_to_patch_mask(
                m, h, w, crop_cfg.img_size, crop_cfg.mask_patch_threshold
            )
        }
        for m in instance_masks
    ]
    return RoleImage(
        tag, image, instance_masks, tokens, raw_tokens, h, w, union, gt_patch_mask, gt_clusters
    )


def raw_map_for(
    state: MethodState, tokens: torch.Tensor, h: int, w: int, scoring_cfg: ScoringConfig
) -> np.ndarray:
    return score_method(state, tokens, knn_k=scoring_cfg.knn_fgbg_num_neighbours).reshape(h, w)


# %% Data loading — every abc5 image annotated for a (part_type, instance_type) combo
def all_combos() -> list[tuple[str, str]]:
    """Every (part_type, instance-type group) combo actually present in data/abc5.

    Checked against image ``_1`` only (abc5's abc3-sourced image), like
    ``multiscale_ablation/common.py``'s ``all_pairs`` does for abc3 — every abc5 image
    within a part type was validated to agree on which instance-type groups are
    annotated when the dataset was built (see scripts/build_abc5_dataset.py).
    """
    combos: list[tuple[str, str]] = []
    for part_type in PART_TYPES:
        ann_stem = ABC5_DATA_DIR / "annotations" / f"{part_type}_1"
        for group_name in available_instance_groups(ann_stem, INSTANCE_TYPE_GROUPS):
            combos.append((part_type, group_name))
    return combos


def build_pool_role_images(
    part_type: str, instance_type: str, encoder: DinoEncoder, crop_cfg: CropConfig
) -> list[RoleImage]:
    """Every abc5 image (1..8) annotated for *(part_type, instance_type)*, as RoleImages —
    the full pool `split_train_val_test`/`build_cv_folds` draw train/val/test from."""
    classes = instance_classes_for(instance_type)
    roles: list[RoleImage] = []
    for n in range(1, 9):
        stem = f"{part_type}_{n}"
        img_path = ABC5_DATA_DIR / f"{stem}.jpg"
        if not img_path.exists():
            continue
        masks = load_instance_pixel_masks(ABC5_DATA_DIR / "annotations" / stem, classes)
        if not masks:
            continue
        img = Image.open(img_path).convert("RGB")
        roles.append(make_role_image(stem, img, masks, encoder, crop_cfg))
    return roles


def split_train_val_test(
    roles: list[RoleImage], seed: int
) -> tuple[list[RoleImage], list[RoleImage], list[RoleImage]]:
    """As-even-as-possible 3-way split of *roles*. Any remainder (pool size not divisible
    by 3) goes to train, then val, so test — held out for the whole run, see
    build_cv_folds — is never the largest split."""
    n = len(roles)
    base, rem = divmod(n, 3)
    sizes = [base + (1 if i < rem else 0) for i in range(3)]
    train_size, val_size, _test_size = sizes
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    train = [roles[i] for i in order[:train_size]]
    val = [roles[i] for i in order[train_size : train_size + val_size]]
    test = [roles[i] for i in order[train_size + val_size :]]
    return train, val, test


def build_cv_folds(
    train_roles: list[RoleImage], val_roles: list[RoleImage], num_folds: int, seed: int
) -> list[tuple[list[RoleImage], list[RoleImage]]]:
    """*num_folds* (train, val) partitions of the combined train+val pool — shuffled once,
    then cut into num_folds contiguous chunks, each taking a turn as val (standard K-fold
    with shuffle=True). The test pool (see split_train_val_test) never enters this — it's
    held out for the whole run, touched only once per fold, after selection is done."""
    combined = train_roles + val_roles
    n = len(combined)
    if not (2 <= num_folds <= n):
        raise ValueError(f"NUM_CV_FOLDS={num_folds} invalid for a combined train+val pool of {n}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    shuffled = [combined[i] for i in order]
    chunks = np.array_split(np.arange(n), num_folds)
    folds: list[tuple[list[RoleImage], list[RoleImage]]] = []
    for chunk in chunks:
        chunk_set = set(chunk.tolist())
        val = [shuffled[i] for i in chunk]
        train = [shuffled[i] for i in range(n) if i not in chunk_set]
        folds.append((train, val))
    return folds


def _bg_gallery_capped(
    scale_protos: dict[str, ScalePrototype],
    bg_scales: list[str],
    max_size: int,
    seed: int,
    use_raw: bool = False,
) -> torch.Tensor:
    """Same per-cell bg-token selection as engine.py's ``pool_scale_patches(want_fg=False)``,
    but shuffles cells and stops once *max_size* patches are collected, instead of
    concatenating every cell first and discarding the excess after — pool_train_pool_
    into_gallery's multi-image pooling combined with step 5's augmented crops can push a
    scale's bg pool into the hundreds of thousands of patches, and materializing that just
    to subsample it already OOMs (see MAX_BG_BANK_SIZE's comment). Every cell here is a
    ``ClusterCrop`` (mid/close's real per-instance crops, and pool_train_pool_into_gallery's
    synthetic per-image "global" cells) — no bare-``ScalePrototype``-as-cell case to handle,
    unlike the general engine.py version. Does not pool ``extra_bg_crops`` (background
    enrichment) — always empty in this script's ``CropConfig`` usage.

    *use_raw* pools each cell's RAW (pre-L2-normalise) tokens instead of ``tokens`` — used by
    ``fit_bg_zca`` to fit whitening from a real (magnitude-preserving) bg covariance; requires
    every cell's ``raw_tokens`` to be populated (``pool_train_pool_into_gallery`` always
    requests this).
    """
    cells: list[ClusterCrop] = []
    for s in bg_scales:
        cells.extend(scale_protos[s].cluster_crops)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cells))
    chunks: list[torch.Tensor] = []
    total = 0
    for i in order:
        cell = cells[int(i)]
        sel_mask = (
            cell.bg_select_mask if cell.bg_select_mask is not None else ~cell.exclude_patch_mask
        )
        source = cell.raw_tokens if use_raw else cell.tokens
        sel = torch.from_numpy(sel_mask.reshape(-1)).to(source.device)
        tokens = source[sel]
        if tokens.shape[0] == 0:
            continue
        chunks.append(tokens)
        total += tokens.shape[0]
        if total >= max_size:
            break
    return torch.cat(chunks, dim=0)


# %% Method-family construction — direct primitives so augmentation injection (below)
# transparently flows through to every family without needing methods.py's rigid,
# fixed-combo registry to be re-derived per candidate.
def method_state_for(
    scale_protos: dict[str, ScalePrototype],
    members: list[str],
    method: str,
    kmeans_k: int | None = None,
    name: str | None = None,
) -> MethodState:
    name = name or f"{'+'.join(members)}:{method}"
    if method == "prototype":
        fg = pool_scale_patches(scale_protos, members, want_fg=True)
        proto = F.normalize(fg.mean(dim=0, keepdim=True), p=2, dim=-1)
        return MethodState(name, "single", proto, roi_source_method=name)
    if method == "knn":
        fg_bank = pool_scale_patches(scale_protos, members, want_fg=True)
        bg_scales = [s for s in ("global", "mid", "close") if s in scale_protos]
        bg_bank = _bg_gallery_capped(scale_protos, bg_scales, MAX_BG_BANK_SIZE, SEED)
        return MethodState(
            name, "knn_fgbg", fg_bank=fg_bank, bg_bank=bg_bank, roi_source_method=name
        )
    if method == "kmeans":
        fg = pool_scale_patches(scale_protos, members, want_fg=True)
        k = min(kmeans_k or 8, fg.shape[0])
        centroids = compute_exemplar_features(fg, mode="kmeans", k=k)
        return MethodState(name, "multi", centroids, roi_source_method=name)
    raise ValueError(f"Unknown method: {method!r}")


# %% Multi-image gallery pooling (the real "train set") — merges every train-pool image's
# own single-image ScalePrototype (engine.py's build_all_scale_prototypes) into one gallery
# per scale.
def _cell_masked_mean(cell: ClusterCrop, want_fg: bool) -> torch.Tensor:
    mask = cell.patch_mask if want_fg else cell.exclude_patch_mask
    flat = torch.from_numpy(mask.reshape(-1)).to(cell.tokens.device)
    sel = cell.tokens[flat] if want_fg else cell.tokens[~flat]
    if sel.shape[0] == 0:
        sel = cell.tokens
    return compute_exemplar_features(sel, mode="mean")


def _mean_patch_prototype_from(scale_protos: dict[str, ScalePrototype]) -> torch.Tensor:
    """Recomputes the fg-patch mean prototype from *scale_protos*' own mid/close cells —
    engine.py's own build_all_scale_prototypes fallback logic, factored out so
    select_transform can recompute it after transforming (mean_prototype must be refit in
    the transformed space, not carried over from the original)."""
    mid_cells = scale_protos["mid"].cluster_crops if "mid" in scale_protos else None
    close_cells = scale_protos["close"].cluster_crops if "close" in scale_protos else None
    all_instance_cells = (mid_cells or []) + (close_cells or [])
    if all_instance_cells:
        return F.normalize(
            torch.cat([_cell_masked_mean(c, True) for c in all_instance_cells], dim=0).mean(
                dim=0, keepdim=True
            ),
            p=2,
            dim=-1,
        )
    return scale_protos["global"].mean_prototype


def pool_train_pool_into_gallery(
    encoder: DinoEncoder, train_pool: list[RoleImage], crop_cfg: CropConfig
) -> tuple[dict[str, ScalePrototype], torch.Tensor, dict[str, list[Image.Image]]]:
    """Builds one merged gallery per scale from every image in *train_pool* — a real
    multi-image train set, not one image's crops.

    mid/close pool the plain concatenation of every image's own per-instance
    ``cluster_crops``. "global" has no per-instance list (one whole-image crop per
    ScalePrototype — see its docstring), so each image's own global crop is wrapped as
    its own synthetic single-cell ``ClusterCrop`` instead, letting engine.py's
    ``pool_scale_patches`` pool every image's global fg/bg exactly like it already pools
    per-instance cells for mid/close.

    The representative scalar fields every ``ScalePrototype`` carries (box/tokens/
    grid_h/grid_w/patch_mask) always come from ``train_pool[0]`` — used only by
    ``evaluate_single_stage``'s threshold-tuning trick, which inherently needs one
    reference image's own geometry, not a pool.

    Also returns *crop_sources_by_scale*: for each scale, the source ``Image.Image`` for
    every entry in that scale's merged ``cluster_crops`` (same order/length) — step 5's
    augmentation injection needs to know which physical image to re-crop from, since a
    pooled gallery's crops no longer all come from one image.

    Every cell's raw (pre-L2-normalise) tokens are always kept (``keep_raw=True``) — needed
    every fold by ``select_transform``'s bg_zca candidate.
    """
    per_image = [
        build_all_scale_prototypes(
            encoder, role.image, role.instance_masks, crop_cfg, keep_raw=True
        )
        for role in train_pool
    ]
    protos_list = [p for p, _ in per_image]

    merged: dict[str, ScalePrototype] = {}
    crop_sources_by_scale: dict[str, list[Image.Image]] = {}
    for scale in protos_list[0]:
        first = protos_list[0][scale]
        if first.cluster_crops is not None:
            merged_crops = [cc for protos in protos_list for cc in protos[scale].cluster_crops]
            crop_sources = [
                role.image
                for protos, role in zip(protos_list, train_pool)
                for _ in protos[scale].cluster_crops
            ]
        else:
            # "global" — one synthetic cell per train image (see docstring above).
            merged_crops = [
                ClusterCrop(
                    -1,
                    protos[scale].box,
                    protos[scale].tokens,
                    protos[scale].grid_h,
                    protos[scale].grid_w,
                    protos[scale].patch_mask,
                    protos[scale].patch_mask,
                    raw_tokens=protos[scale].raw_tokens,
                )
                for protos in protos_list
            ]
            crop_sources = [role.image for role in train_pool]
        crop_sources_by_scale[scale] = crop_sources

        fg_means = [_cell_masked_mean(c, want_fg=True) for c in merged_crops]
        bg_means = [_cell_masked_mean(c, want_fg=False) for c in merged_crops]
        extra_bg = [e for protos in protos_list for e in (protos[scale].extra_bg_crops or [])]
        bg_means += [e.mean_token for e in extra_bg]
        avg = F.normalize(torch.cat(fg_means, dim=0).mean(dim=0, keepdim=True), p=2, dim=-1)
        bg_avg = F.normalize(torch.cat(bg_means, dim=0).mean(dim=0, keepdim=True), p=2, dim=-1)

        target_size_frac = first.target_size_frac
        if scale == "mid":
            # Median across every pooled crop (not per-image then averaged) — more crops,
            # more robust than any single image's own median.
            fracs = [
                ((cc.box[2] - cc.box[0]) / src.width, (cc.box[3] - cc.box[1]) / src.height)
                for cc, src in zip(merged_crops, crop_sources)
            ]
            target_size_frac = (
                float(np.median([f[0] for f in fracs])),
                float(np.median([f[1] for f in fracs])),
            )

        merged[scale] = dataclasses.replace(
            first,
            mean_prototype=avg,
            bg_prototype=bg_avg,
            cluster_crops=merged_crops,
            target_size_frac=target_size_frac,
            extra_bg_crops=extra_bg or None,
        )

    mean_patch_prototype = _mean_patch_prototype_from(merged)
    return merged, mean_patch_prototype, crop_sources_by_scale


# %% Feature-space transform (step "2.5") — bg_zca whitening, fit from the train pool's own
# raw bg tokens and applied everywhere a gallery or a query image's tokens get compared by
# cosine similarity. See select_transform's docstring for the two-stage exception (blob
# rescoring, done by engine.py's own internal re-encoding, stays untransformed).
Transform = tuple[torch.Tensor, torch.Tensor]  # (mu, zca whitening matrix)


def apply_transform_to_tokens(tokens: torch.Tensor, transform: Transform) -> torch.Tensor:
    """Applies a fitted (mu, w) affine transform then re-L2-normalises — matches every
    non-lda/mahalanobis pipeline in feature_transform_oracle_iou.py (fit on raw tokens,
    "+ L2 Norm" as the final step, never before centering/whitening)."""
    mu, w = transform
    return F.normalize(apply_affine(tokens, mu, w), p=2, dim=-1)


def fit_bg_zca(
    scale_protos: dict[str, ScalePrototype], bg_scales: list[str], eps: float
) -> Transform:
    """Fits ZCA whitening from the train pool's own raw bg tokens, pooled across
    *bg_scales* the same way knn's bg gallery is (_bg_gallery_capped, capped at
    MAX_BG_BANK_SIZE, use_raw=True) — the real-train-set analogue of
    feature_transform_oracle_iou.py's per-combo ``bg_zca`` fit (``eigh(cov(bg))``)."""
    bg_raw = _bg_gallery_capped(scale_protos, bg_scales, MAX_BG_BANK_SIZE, SEED, use_raw=True)
    mu = fit_mean(bg_raw)
    eigvecs, eigvals = fit_cov_eigh(bg_raw, mu)
    return mu, zca_matrix(eigvecs, eigvals, eps)


def transform_scale_protos(
    scale_protos: dict[str, ScalePrototype], transform: Transform
) -> dict[str, ScalePrototype]:
    """Returns a copy of *scale_protos* where every scale's cluster_crops (and the scale's
    own representative ``tokens``) have their raw tokens passed through *transform* — so
    every existing cosine-similarity function downstream (method_state_for, raw_map_for,
    cleaning, ...) works completely unchanged on the transformed embedding space, exactly
    as it did on the original one."""
    return {
        scale: dataclasses.replace(
            proto,
            cluster_crops=[
                dataclasses.replace(cc, tokens=apply_transform_to_tokens(cc.raw_tokens, transform))
                for cc in proto.cluster_crops
            ],
            tokens=apply_transform_to_tokens(proto.raw_tokens, transform),
        )
        for scale, proto in scale_protos.items()
    }


def transform_role_image(role: RoleImage, transform: Transform) -> RoleImage:
    """Same substitution as transform_scale_protos, for one query image's own token grid."""
    return dataclasses.replace(role, tokens=apply_transform_to_tokens(role.raw_tokens, transform))


# %% Augmentation injection (step 5) — appends perturbed ClusterCrops re-cropped from each
# crop's own source image (crop_sources, parallel to base_crops — see
# pool_train_pool_into_gallery), so method_state_for's pooling picks them up transparently.
def augmented_extra_crops(
    base_crops: list[ClusterCrop],
    crop_sources: list[Image.Image],
    family_specs: list[tuple[str, float, object]],
    encoder: DinoEncoder,
    crop_cfg: CropConfig,
    transform: Transform | None = None,
) -> list[ClusterCrop]:
    """Builds every (crop, family/severity) augmented image up front, then encodes all of
    them in ONE batched forward pass — extract_patch_tokens_batch_with_cls is built for
    exactly this, but calling it once per image (a batch of 1) inside this loop paid full
    per-call GPU/Python overhead for every one of them, dominating select_augmentation's
    cost (measured: ~50s/fold, by far the most expensive of the 6 decision steps).
    """
    aug_images: list[Image.Image] = []
    aug_meta: list[tuple[ClusterCrop, np.ndarray]] = []
    for cc, src_img in zip(base_crops, crop_sources):
        crop_img = src_img.crop(cc.box)
        fill = mean_color(src_img)
        for _family, val, fn in family_specs:
            aug_img, aug_mask_px = fn(crop_img, cc.own_mask_px, val, fill)
            aug_images.append(aug_img)
            aug_meta.append((cc, aug_mask_px))

    token_results = extract_patch_tokens_batch_with_cls(
        encoder, aug_images, crop_cfg.layer_idx, crop_cfg.debias, return_raw=True
    )
    extra: list[ClusterCrop] = []
    for (cc, aug_mask_px), (tokens, cls, gh, gw, raw) in zip(aug_meta, token_results):
        if transform is not None:
            # A transformed gallery's every cell must live in the same transformed space
            # — an untransformed augmented crop pooled alongside transformed originals
            # would silently corrupt the fg/bg gallery, not just misscore a query.
            tokens = apply_transform_to_tokens(raw, transform)
        own_frac = patch_fg_fraction(aug_mask_px, gh, gw, crop_cfg.img_size)
        patch_mask = own_frac >= crop_cfg.mask_patch_threshold
        extra.append(
            ClusterCrop(
                cc.cluster_idx,
                cc.box,
                tokens,
                gh,
                gw,
                patch_mask,
                cc.exclude_patch_mask,
                own_frac=own_frac,
                own_mask_px=aug_mask_px,
                cls=cls,
            )
        )
    return extra


def protos_with_augmentation(
    base_protos: dict[str, ScalePrototype],
    crop_sources_by_scale: dict[str, list[Image.Image]],
    members: list[str],
    candidate: str,
    encoder: DinoEncoder,
    crop_cfg: CropConfig,
    transform: Transform | None = None,
) -> dict[str, ScalePrototype]:
    if candidate == "none":
        return base_protos
    if candidate == "all":
        family_specs = [
            (fam, v, spec["apply"]) for fam, spec in AUGMENTATIONS.items() for v in spec["values"]
        ]
    else:
        spec = AUGMENTATIONS[candidate]
        family_specs = [(candidate, v, spec["apply"]) for v in spec["values"]]
    new_protos = dict(base_protos)
    for scale in members:
        # "global"'s cluster_crops are pool_train_pool_into_gallery's synthetic per-image
        # cells (whole-image crops, not real instances) — augmentation only ever grows
        # mid/close's real per-instance galleries, same rule as the single-image script.
        if scale == "global":
            continue
        extra = augmented_extra_crops(
            base_protos[scale].cluster_crops,
            crop_sources_by_scale[scale],
            family_specs,
            encoder,
            crop_cfg,
            transform=transform,
        )
        new_protos[scale] = dataclasses.replace(
            base_protos[scale], cluster_crops=base_protos[scale].cluster_crops + extra
        )
    return new_protos


def stability_score(
    state: MethodState,
    eval_role: RoleImage,
    encoder: DinoEncoder,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    transform: Transform | None = None,
) -> float:
    """GT-free step 5 scorer: mean IoU of the (Otsu-binarised) predicted fg mask between
    the clean val image and each mildly-perturbed view -- higher = more stable.

    *eval_role.tokens* is already in *transform*'s space when a transform is active (every
    eval/test RoleImage gets replaced via transform_role_image once, upstream of this
    call) — but each nuisance-perturbed view is freshly re-encoded right here, so it needs
    the same transform applied explicitly to stay consistent with *state*'s own (possibly
    transformed) galleries.
    """
    clean_raw = raw_map_for(
        state, eval_role.tokens, eval_role.grid_h, eval_role.grid_w, scoring_cfg
    )
    clean_mask = clean_raw > otsu_threshold(clean_raw)
    fill = mean_color(eval_role.image)
    dummy_mask = np.zeros((eval_role.image.height, eval_role.image.width), dtype=bool)
    ious = []
    for _tag, val, fn in NUISANCE_PERTURBATIONS:
        pert_img, _ = fn(eval_role.image, dummy_mask, val, fill)
        tokens, raw_tokens, h, w = _extract_patch_tokens_with_raw(
            encoder, pert_img, crop_cfg.layer_idx, debias=crop_cfg.debias
        )
        if transform is not None:
            tokens = apply_transform_to_tokens(raw_tokens, transform)
        raw_map = score_method(state, tokens, knn_k=scoring_cfg.knn_fgbg_num_neighbours).reshape(
            h, w
        )
        mask = raw_map > otsu_threshold(raw_map)
        if mask.shape != clean_mask.shape:
            continue
        ious.append(mask_iou(clean_mask, mask))
    return float(np.mean(ious)) if ious else 0.0


# %% Shared helper — score one MethodState across every val-set image and average
def _score_over_eval_roles(
    state: MethodState, eval_roles: list[RoleImage], branch: str, scoring_cfg: ScoringConfig
) -> tuple[np.ndarray, float]:
    """Mean raw_score for *state* across every val-set image, plus the first image's raw
    map (diagnostics/visualization only render one fold's first val image)."""
    scores = []
    rep_raw: np.ndarray | None = None
    for eval_role in eval_roles:
        raw = raw_map_for(state, eval_role.tokens, eval_role.grid_h, eval_role.grid_w, scoring_cfg)
        if rep_raw is None:
            rep_raw = raw
        scores.append(raw_score(branch, raw, eval_role.gt_patch_mask, scoring_cfg))
    assert rep_raw is not None
    return rep_raw, float(np.mean(scores))


# %% Decision step 1 — scale
def select_scale(
    scale_protos: dict[str, ScalePrototype],
    eval_roles: list[RoleImage],
    branch: str,
    scoring_cfg: ScoringConfig,
) -> tuple[str, list[str], dict[str, np.ndarray], dict[str, float]]:
    candidates = {
        name: members
        for name, members in SCALE_COMBOS.items()
        if all(m in scale_protos for m in members)
    }
    raws, scores = {}, {}
    for name, members in candidates.items():
        state = method_state_for(scale_protos, members, "prototype", name=name)
        raws[name], scores[name] = _score_over_eval_roles(state, eval_roles, branch, scoring_cfg)
    best = max(scores, key=scores.get)
    return best, candidates[best], raws, scores


# %% Decision step 2 — method
def select_method(
    scale_protos: dict[str, ScalePrototype],
    members: list[str],
    eval_roles: list[RoleImage],
    branch: str,
    scoring_cfg: ScoringConfig,
) -> tuple[str, int | None, dict[str, np.ndarray], dict[str, float]]:
    raws, scores = {}, {}
    best_kmeans_k = None
    for method in ("prototype", "knn", "kmeans"):
        if method == "kmeans":
            best_k, best_raw, best_s = None, None, -np.inf
            for k in scoring_cfg.kmeans_ks:
                state = method_state_for(
                    scale_protos, members, "kmeans", kmeans_k=k, name=f"kmeans{k}"
                )
                raw, s = _score_over_eval_roles(state, eval_roles, branch, scoring_cfg)
                if s > best_s:
                    best_k, best_raw, best_s = k, raw, s
            raws["kmeans"], scores["kmeans"] = best_raw, best_s
            best_kmeans_k = best_k
        else:
            state = method_state_for(scale_protos, members, method, name=method)
            raws[method], scores[method] = _score_over_eval_roles(
                state, eval_roles, branch, scoring_cfg
            )
    best = max(scores, key=scores.get)
    return best, (best_kmeans_k if best == "kmeans" else None), raws, scores


# %% Decision step 3 — single vs. two-stage
def geometric_two_stage_decision(
    pred_clusters: list[dict], expected_size: float
) -> tuple[str, dict]:
    """GT-free proxy: flag "too small" (predicted clusters undersized vs. the training
    pool's own median instance size) or "too close" (a cluster's bbox fill-ratio is
    anomalously low, suggesting several instances merged into one blob)."""
    if not pred_clusters:
        return "two_stage", {"reason": "no single-stage detections", "expected_size": expected_size}
    sizes = np.array([c["mask"].sum() for c in pred_clusters], dtype=float)
    fill_ratios = []
    for c in pred_clusters:
        ys, xs = np.where(c["mask"])
        bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        fill_ratios.append(c["mask"].sum() / bbox_area)
    too_small = bool(np.median(sizes) < SMALL_OBJECT_FRACTION * expected_size)
    too_close = bool(np.min(fill_ratios) < MERGE_FILL_RATIO_THRESHOLD)
    stage = "two_stage" if (too_small or too_close) else "single"
    return stage, {
        "too_small": too_small,
        "too_close": too_close,
        "median_size": float(np.median(sizes)),
        "expected_size": expected_size,
        "min_fill_ratio": float(np.min(fill_ratios)),
    }


def evaluate_single_stage(
    state: MethodState,
    locked_scale_name: str,
    train_scale_protos: dict[str, ScalePrototype],
    train_role: RoleImage,
    eval_role: RoleImage,
    branch: str,
    mean_patch_prototype: torch.Tensor,
    min_cs: int,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    compute_metrics: bool | None = None,
) -> dict:
    if compute_metrics is None:
        compute_metrics = branch == "gt_calibrated"
    if (
        scoring_cfg.tune_threshold_per_scale
        and locked_scale_name == "close"
        and "close" in train_scale_protos
    ):
        ref = train_scale_protos["close"]
    else:
        ref = train_scale_protos["mid"]
    x0, y0, x1, y1 = ref.box
    train_union = np.stack(train_role.instance_masks).any(axis=0)
    ref_gt_mask = pixel_mask_to_patch_mask(
        train_union[y0:y1, x0:x1],
        ref.grid_h,
        ref.grid_w,
        crop_cfg.img_size,
        crop_cfg.mask_patch_threshold,
    )
    ref_raw = raw_map_for(state, ref.tokens, ref.grid_h, ref.grid_w, scoring_cfg)

    if branch == "gt_calibrated":
        thr = iou_tuned_threshold(ref_raw, ref_gt_mask, scoring_cfg.ref_threshold_steps)
        cluster_reject_thr, _ = tune_cluster_reject_threshold(
            ref.cluster_crops,
            state,
            mean_patch_prototype,
            thr,
            min_cs,
            scoring_cfg.iou_match_threshold,
            scoring_cfg,
        )
    else:
        thr = otsu_threshold(ref_raw)
        ref_cos = (ref.tokens @ mean_patch_prototype.T).squeeze(-1).cpu().float().numpy()
        cluster_reject_thr = otsu_threshold(ref_cos)

    eval_raw = raw_map_for(state, eval_role.tokens, eval_role.grid_h, eval_role.grid_w, scoring_cfg)
    ys, xs = np.where(eval_raw > thr)
    if len(xs) < max(scoring_cfg.min_points_floor, min_cs):
        pred_clusters: list[dict] = []
    else:
        pred_clusters = dbscan_clusters(
            xs, ys, eval_role.grid_h, eval_role.grid_w, eval_raw, scoring_cfg, min_cs
        )
    annotate_cluster_rejection(
        pred_clusters, eval_role.tokens, mean_patch_prototype, cluster_reject_thr
    )
    kept = [c for c in pred_clusters if not c["rejected"]]
    metrics = (
        match_and_score(kept, eval_role.gt_clusters, scoring_cfg.iou_match_threshold)
        if compute_metrics
        else None
    )
    return {
        "raw": eval_raw,
        "threshold": thr,
        "cluster_reject_thr": cluster_reject_thr,
        "pred_clusters": kept,
        "metrics": metrics,
    }


def evaluate_two_stage(
    state: MethodState,
    roi_source_raw: np.ndarray,
    role: RoleImage,
    encoder: DinoEncoder,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    mean_patch_prototype: torch.Tensor,
    min_cs: int,
    single_result: dict,
    patch_size: int,
    target_size_frac: tuple[float, float] | None,
    branch: str,
    compute_metrics: bool | None = None,
) -> dict:
    if compute_metrics is None:
        compute_metrics = branch == "gt_calibrated"
    native_w, native_h = role.image.size
    scale_x, scale_y = native_w / crop_cfg.img_size, native_h / crop_cfg.img_size
    blobs, _roi_mask = find_roi_blobs(
        roi_source_raw, role.image, encoder, crop_cfg, scoring_cfg, patch_size, target_size_frac
    )
    pred_clusters, diagnostics = two_stage_predicted_clusters(
        blobs,
        state,
        single_result["threshold"],
        mean_patch_prototype,
        single_result["cluster_reject_thr"],
        min_cs,
        role.grid_h,
        role.grid_w,
        patch_size,
        scale_x,
        scale_y,
        scoring_cfg,
        q_pixel_mask=role.union_pixel_mask,
        q_instance_pixel_masks=role.instance_masks,
        crop_cfg=crop_cfg,
        collect_diagnostics=True,
    )
    metrics = (
        match_and_score(pred_clusters, role.gt_clusters, scoring_cfg.iou_match_threshold)
        if compute_metrics
        else None
    )
    return {
        "pred_clusters": pred_clusters,
        "metrics": metrics,
        "blobs": blobs,
        "diagnostics": diagnostics,
    }


def select_stage(
    state: MethodState,
    two_stage_state: MethodState,
    two_stage_scale_protos: dict[str, ScalePrototype],
    two_stage_mean_patch_prototype: torch.Tensor,
    locked_scale_name: str,
    train_scale_protos: dict[str, ScalePrototype],
    train_role: RoleImage,
    eval_roles: list[RoleImage],
    encoder: DinoEncoder,
    mean_patch_prototype: torch.Tensor,
    min_cs: int,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    branch: str,
    patch_size: int,
) -> tuple[str, dict, dict, dict]:
    """*state*/*train_scale_protos*/*mean_patch_prototype* are the (possibly transformed —
    see select_transform) galleries used for the "official" single-stage result and to seed
    ROI blob discovery. *two_stage_state*/*two_stage_scale_protos*/*two_stage_mean_patch_
    prototype* are always the ORIGINAL, untransformed ones — two-stage's blob rescoring
    (engine.py's two_stage_predicted_clusters) re-encodes ROI blobs internally with no
    transform hook, so it must be compared against an untransformed gallery with its own
    untransformed threshold/cluster_reject_thr (recomputed here via a second
    evaluate_single_stage call when a transform is active — cheap, no re-encoding, just
    re-run cosine-similarity math) — never the transformed ones' threshold, which was tuned
    for a different embedding space. When no transform is active, both triples are the same
    objects and the second call is skipped."""
    target_size_frac = (
        train_scale_protos[locked_scale_name].target_size_frac
        if locked_scale_name in train_scale_protos
        else None
    )
    same = state is two_stage_state
    per_image = []
    for eval_role in eval_roles:
        single = evaluate_single_stage(
            state,
            locked_scale_name,
            train_scale_protos,
            train_role,
            eval_role,
            branch,
            mean_patch_prototype,
            min_cs,
            crop_cfg,
            scoring_cfg,
        )
        single_for_two_stage = (
            single
            if same
            else evaluate_single_stage(
                two_stage_state,
                locked_scale_name,
                two_stage_scale_protos,
                train_role,
                eval_role,
                branch,
                two_stage_mean_patch_prototype,
                min_cs,
                crop_cfg,
                scoring_cfg,
                compute_metrics=False,
            )
        )
        two_stage = evaluate_two_stage(
            two_stage_state,
            single["raw"],
            eval_role,
            encoder,
            crop_cfg,
            scoring_cfg,
            two_stage_mean_patch_prototype,
            min_cs,
            single_for_two_stage,
            patch_size,
            target_size_frac,
            branch,
        )
        per_image.append((single, two_stage))

    if branch == "gt_calibrated":
        gains = [
            (
                (two_stage["metrics"]["f1"] - single["metrics"]["f1"])
                if (two_stage["metrics"] is not None and single["metrics"] is not None)
                else -1.0
            )
            for single, two_stage in per_image
        ]
        gain = float(np.mean(gains))
        stage = "two_stage" if gain > TWO_STAGE_MIN_GAIN else "single"
        decision_info = {"f1_gain": gain, "per_image_f1_gain": gains}
    else:
        mid_crops = train_scale_protos["mid"].cluster_crops
        expected_size = (
            float(np.median([cc.patch_mask.sum() for cc in mid_crops]))
            if mid_crops
            else float(min_cs)
        )
        decisions = [
            geometric_two_stage_decision(single["pred_clusters"], expected_size)
            for single, _ in per_image
        ]
        votes = [d for d, _ in decisions]
        stage = "two_stage" if votes.count("two_stage") > len(votes) / 2 else "single"
        decision_info = {
            "votes": votes,
            "details": [info for _, info in decisions],
            "expected_size": expected_size,
        }
    single0, two_stage0 = per_image[0]
    return stage, single0, two_stage0, decision_info


# %% Decision step 4 — denoising
def select_denoising(
    base_scale_protos: dict[str, ScalePrototype],
    members: list[str],
    method: str,
    kmeans_k: int | None,
    eval_roles: list[RoleImage],
    branch: str,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
) -> tuple[str, dict[str, dict[str, ScalePrototype]], dict[str, np.ndarray], dict[str, float]]:
    variants = ["raw", "step1", "step2_cls", "step2_center"]
    protos_by_variant, raws, scores = {}, {}, {}
    for v in variants:
        protos = (
            base_scale_protos
            if v == "raw"
            else apply_fg_cleaning(
                base_scale_protos, crop_cfg, dataclasses.replace(scoring_cfg, fg_clean_stage=v)
            )
        )
        protos_by_variant[v] = protos
        state = method_state_for(protos, members, method, kmeans_k=kmeans_k, name=v)
        raws[v], scores[v] = _score_over_eval_roles(state, eval_roles, branch, scoring_cfg)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    best_clean = max(("step1", "step2_cls", "step2_center"), key=scores.get)
    chosen = "raw" if scores["raw"] >= scores[best_clean] else best_clean
    return chosen, protos_by_variant, raws, scores


# %% Decision step 5 — augmentation
def select_augmentation(
    base_protos: dict[str, ScalePrototype],
    crop_sources_by_scale: dict[str, list[Image.Image]],
    members: list[str],
    method: str,
    kmeans_k: int | None,
    eval_roles: list[RoleImage],
    branch: str,
    encoder: DinoEncoder,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    transform: Transform | None = None,
) -> tuple[str, dict[str, float]]:
    # "all" (every family composed at once) is deliberately excluded: [[project-
    # dinoisawesome-fundamental-robustness]] finding #3 (augmented_prototype_oracle_iou_
    # knn_fgbg.py) already established composing every augmentation family together is
    # flat-to-negative and never the winning choice ("augment conservatively — one family,
    # one severity — or not at all; never compose the whole catalog into the gallery"). It
    # was also the single most expensive candidate here (6 severities vs. 2 per family —
    # half of step 5's total augmented-crop volume) for a config that was never going to
    # win. protos_with_augmentation still supports candidate="all" if ever wanted directly.
    candidates = ["none", *AUGMENTATIONS.keys()]
    scores = {}
    for cand in candidates:
        protos = protos_with_augmentation(
            base_protos,
            crop_sources_by_scale,
            members,
            cand,
            encoder,
            crop_cfg,
            transform=transform,
        )
        state = method_state_for(protos, members, method, kmeans_k=kmeans_k, name=cand)
        if branch == "gt_calibrated":
            per_image = []
            for eval_role in eval_roles:
                raw = raw_map_for(
                    state, eval_role.tokens, eval_role.grid_h, eval_role.grid_w, scoring_cfg
                )
                per_image.append(
                    oracle_iou(raw, eval_role.gt_patch_mask, scoring_cfg.ref_threshold_steps)
                )
        else:
            per_image = [
                stability_score(
                    state, eval_role, encoder, crop_cfg, scoring_cfg, transform=transform
                )
                for eval_role in eval_roles
            ]
        scores[cand] = float(np.mean(per_image))
        del protos, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    best = max(scores, key=scores.get)
    return best, scores


# %% Decision step "2.5" — feature-space transform
def select_transform(
    base_protos: dict[str, ScalePrototype],
    scale_members: list[str],
    method: str,
    kmeans_k: int | None,
    eval_roles: list[RoleImage],
    branch: str,
    scoring_cfg: ScoringConfig,
) -> tuple[Transform | None, float | None, dict[str, float]]:
    """Searched with scale/method already locked (steps 1-2), against the same single-stage
    val signal those two steps already use. Candidates: no transform vs. bg_zca at each
    FEATURE_TRANSFORM_EPS_SWEEP value, fit from the train pool's own raw bg tokens
    (fit_bg_zca) — the raw-token, "+ L2 Norm"-last recipe feature_transform_oracle_iou.py
    validated as the single biggest lever across the whole fundamental/ series.

    Does not compose with two-stage's blob rescoring: engine.py's find_roi_blobs/
    two_stage_predicted_clusters do their own internal re-encoding of ROI blob crops with
    no transform hook, so applying a fitted (mu, w) there would compare a transformed
    gallery against untransformed blob tokens — silently wrong, not just weaker. See
    run_fold/evaluate_on_test: two-stage's blob-rescoring state is always built from the
    *original*, untransformed galleries, even when this step picks bg_zca — only single-
    stage scoring (and the score map that seeds ROI blob discovery — a spatial signal, not
    an embedding comparison, so the transform still helps there) is transform-aware.

    Returns (transform_or_None, chosen_eps_or_None, scores).
    """
    bg_scales = [s for s in ("global", "mid", "close") if s in base_protos]
    scores: dict[str, float] = {}

    state_none = method_state_for(
        base_protos, scale_members, method, kmeans_k=kmeans_k, name="transform_none"
    )
    _, scores["none"] = _score_over_eval_roles(state_none, eval_roles, branch, scoring_cfg)

    best_transform, best_eps, best_score = None, None, scores["none"]
    for eps in FEATURE_TRANSFORM_EPS_SWEEP:
        transform = fit_bg_zca(base_protos, bg_scales, eps)
        transformed_protos = transform_scale_protos(base_protos, transform)
        state = method_state_for(
            transformed_protos, scale_members, method, kmeans_k=kmeans_k, name=f"bg_zca_{eps}"
        )
        per_image = []
        for eval_role in eval_roles:
            q_tokens = apply_transform_to_tokens(eval_role.raw_tokens, transform)
            raw = raw_map_for(state, q_tokens, eval_role.grid_h, eval_role.grid_w, scoring_cfg)
            per_image.append(raw_score(branch, raw, eval_role.gt_patch_mask, scoring_cfg))
        s = float(np.mean(per_image))
        scores[f"bg_zca_eps={eps}"] = s
        if s > best_score:
            best_transform, best_eps, best_score = transform, eps, s
        del transformed_protos, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_transform, best_eps, scores


# %% Fold / branch orchestration
@dataclass
class SelectionConfig:
    scale_name: str
    scale_members: list[str]
    method: str
    kmeans_k: int | None
    transform: str
    transform_eps: float | None
    stage: str
    clean_stage: str
    augmentation: str


def run_fold(
    train_pool: list[RoleImage],
    eval_roles: list[RoleImage],
    encoder: DinoEncoder,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    branch: str,
    patch_size: int,
) -> tuple[SelectionConfig, dict]:
    base_protos_original, mean_patch_prototype_original, crop_sources_by_scale = (
        pool_train_pool_into_gallery(encoder, train_pool, crop_cfg)
    )
    mid_sizes = np.array([cc.patch_mask.sum() for cc in base_protos_original["mid"].cluster_crops])
    min_cs = min_cluster_size_bound(
        mid_sizes, scoring_cfg.cluster_size_margin, scoring_cfg.min_points_floor
    )

    scale_name, scale_members, scale_raws, scale_scores = select_scale(
        base_protos_original, eval_roles, branch, scoring_cfg
    )
    method, kmeans_k, method_raws, method_scores = select_method(
        base_protos_original, scale_members, eval_roles, branch, scoring_cfg
    )

    transform, transform_eps, transform_scores = select_transform(
        base_protos_original, scale_members, method, kmeans_k, eval_roles, branch, scoring_cfg
    )
    if transform is not None:
        base_protos = transform_scale_protos(base_protos_original, transform)
        eval_roles_t = [transform_role_image(r, transform) for r in eval_roles]
        mean_patch_prototype = _mean_patch_prototype_from(base_protos)
        transform_name = "bg_zca"
    else:
        base_protos = base_protos_original
        eval_roles_t = eval_roles
        mean_patch_prototype = mean_patch_prototype_original
        transform_name = "none"

    locked_state = method_state_for(
        base_protos, scale_members, method, kmeans_k=kmeans_k, name="locked"
    )
    two_stage_state = (
        locked_state
        if transform is None
        else method_state_for(
            base_protos_original, scale_members, method, kmeans_k=kmeans_k, name="locked_orig"
        )
    )
    train_role = train_pool[0]
    stage, single_result, two_stage_result, stage_decision = select_stage(
        locked_state,
        two_stage_state,
        base_protos_original,
        mean_patch_prototype_original,
        scale_name,
        base_protos,
        train_role,
        eval_roles_t,
        encoder,
        mean_patch_prototype,
        min_cs,
        crop_cfg,
        scoring_cfg,
        branch,
        patch_size,
    )
    clean_stage, clean_protos_by_variant, clean_raws, clean_scores = select_denoising(
        base_protos, scale_members, method, kmeans_k, eval_roles_t, branch, crop_cfg, scoring_cfg
    )
    final_protos = clean_protos_by_variant[clean_stage]
    if ENABLE_AUGMENTATION_SEARCH:
        augmentation, aug_scores = select_augmentation(
            final_protos,
            crop_sources_by_scale,
            scale_members,
            method,
            kmeans_k,
            eval_roles_t,
            branch,
            encoder,
            crop_cfg,
            scoring_cfg,
            transform=transform,
        )
    else:
        augmentation, aug_scores = "none", {}

    cfg = SelectionConfig(
        scale_name,
        scale_members,
        method,
        kmeans_k,
        transform_name,
        transform_eps,
        stage,
        clean_stage,
        augmentation,
    )
    diagnostics = {
        "scale_raws": scale_raws,
        "scale_scores": scale_scores,
        "method_raws": method_raws,
        "method_scores": method_scores,
        "transform_scores": transform_scores,
        "single_result": single_result,
        "two_stage_result": two_stage_result,
        "stage_decision": stage_decision,
        "clean_raws": clean_raws,
        "clean_scores": clean_scores,
        "clean_protos_by_variant": clean_protos_by_variant,
        "aug_scores": aug_scores,
        "min_cs": min_cs,
        "mean_patch_prototype": mean_patch_prototype,
        "base_protos": base_protos,
    }
    return cfg, diagnostics


def evaluate_on_test(
    cfg: SelectionConfig,
    branch: str,
    train_pool: list[RoleImage],
    test_roles: list[RoleImage],
    encoder: DinoEncoder,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    patch_size: int,
) -> dict:
    """Applies *cfg* (chosen by *branch*, using only train_pool/val) to every held-out test
    image, faithfully: the GT-free branch's own Otsu-based thresholding/rejection is used
    end-to-end here too (a true zero-annotation config never gets to call the
    GT-tuned threshold at deployment either) — GT is used only afterwards, by
    match_and_score, purely to grade the outcome for reporting. Returns metrics averaged
    across every test image.

    Re-fits ``cfg.transform``/``cfg.transform_eps`` from the train pool (SelectionConfig only
    stores the choice, not the fitted tensors — same pattern cfg.clean_stage/cfg.augmentation
    already use). Two-stage evaluation always falls back to the untransformed galleries for
    its blob rescoring — see select_stage's docstring for why.
    """
    base_protos_original, mean_patch_prototype_original, crop_sources_by_scale = (
        pool_train_pool_into_gallery(encoder, train_pool, crop_cfg)
    )

    transform: Transform | None = None
    if cfg.transform == "bg_zca":
        bg_scales = [s for s in ("global", "mid", "close") if s in base_protos_original]
        transform = fit_bg_zca(base_protos_original, bg_scales, cfg.transform_eps)

    def _build(
        base_protos: dict[str, ScalePrototype], transform_: Transform | None
    ) -> tuple[dict[str, ScalePrototype], MethodState]:
        protos = (
            base_protos
            if cfg.clean_stage == "raw"
            else apply_fg_cleaning(
                base_protos,
                crop_cfg,
                dataclasses.replace(scoring_cfg, fg_clean_stage=cfg.clean_stage),
            )
        )
        protos = protos_with_augmentation(
            protos,
            crop_sources_by_scale,
            cfg.scale_members,
            cfg.augmentation,
            encoder,
            crop_cfg,
            transform=transform_,
        )
        state = method_state_for(
            protos, cfg.scale_members, cfg.method, kmeans_k=cfg.kmeans_k, name="final"
        )
        return protos, state

    if transform is not None:
        base_protos = transform_scale_protos(base_protos_original, transform)
        mean_patch_prototype = _mean_patch_prototype_from(base_protos)
        protos, state = _build(base_protos, transform)
        protos_orig, state_orig = _build(base_protos_original, None)
    else:
        mean_patch_prototype = mean_patch_prototype_original
        protos, state = _build(base_protos_original, None)
        protos_orig, state_orig = protos, state

    mid_sizes = np.array([cc.patch_mask.sum() for cc in protos["mid"].cluster_crops])
    min_cs = min_cluster_size_bound(
        mid_sizes, scoring_cfg.cluster_size_margin, scoring_cfg.min_points_floor
    )

    train_role = train_pool[0]
    target_size_frac = protos[cfg.scale_name].target_size_frac if cfg.scale_name in protos else None
    test_roles_t = (
        [transform_role_image(r, transform) for r in test_roles]
        if transform is not None
        else test_roles
    )

    per_image_metrics = []
    for test_role, test_role_t in zip(test_roles, test_roles_t):
        single = evaluate_single_stage(
            state,
            cfg.scale_name,
            protos,
            train_role,
            test_role_t,
            branch,
            mean_patch_prototype,
            min_cs,
            crop_cfg,
            scoring_cfg,
            compute_metrics=True,
        )
        if cfg.stage == "single":
            per_image_metrics.append(single["metrics"])
            continue
        single_for_two_stage = (
            single
            if transform is None
            else evaluate_single_stage(
                state_orig,
                cfg.scale_name,
                protos_orig,
                train_role,
                test_role,
                branch,
                mean_patch_prototype_original,
                min_cs,
                crop_cfg,
                scoring_cfg,
                compute_metrics=False,
            )
        )
        two_stage = evaluate_two_stage(
            state_orig,
            single["raw"],
            test_role,
            encoder,
            crop_cfg,
            scoring_cfg,
            mean_patch_prototype_original,
            min_cs,
            single_for_two_stage,
            patch_size,
            target_size_frac,
            branch,
            compute_metrics=True,
        )
        per_image_metrics.append(two_stage["metrics"])

    return {
        k: float(np.nanmean([m[k] for m in per_image_metrics]))
        for k in ("precision", "recall", "f1", "mean_iou")
    }


def run_branch(
    pair_label: str,
    branch: str,
    folds: list[tuple[list[RoleImage], list[RoleImage]]],
    test_roles: list[RoleImage],
    encoder: DinoEncoder,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    patch_size: int,
) -> dict:
    fold_results = []
    for i, (train_pool, eval_roles) in enumerate(folds):
        cfg, diagnostics = run_fold(
            train_pool, eval_roles, encoder, crop_cfg, scoring_cfg, branch, patch_size
        )
        test_metrics = evaluate_on_test(
            cfg, branch, train_pool, test_roles, encoder, crop_cfg, scoring_cfg, patch_size
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fold_results.append(
            {
                "fold": i,
                "config": cfg,
                "diagnostics": diagnostics,
                "test_metrics": test_metrics,
                "train_pool": train_pool,
                "eval_roles": eval_roles,
            }
        )
        transform_label = (
            f"{cfg.transform}(eps={cfg.transform_eps})"
            if cfg.transform != "none"
            else cfg.transform
        )
        log.info(
            "[%s/%s] fold%d train=%s val=%s -> scale=%s method=%s transform=%s stage=%s "
            "clean=%s aug=%s | test P=%.2f R=%.2f F1=%.2f mIoU=%.2f",
            pair_label,
            branch,
            i,
            [r.tag for r in train_pool],
            [r.tag for r in eval_roles],
            cfg.scale_name,
            cfg.method,
            transform_label,
            cfg.stage,
            cfg.clean_stage,
            cfg.augmentation,
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["f1"],
            test_metrics["mean_iou"],
        )
    avg_metrics = {
        k: float(np.nanmean([fr["test_metrics"][k] for fr in fold_results]))
        for k in ("precision", "recall", "f1", "mean_iou")
    }
    return {"branch": branch, "folds": fold_results, "avg_test_metrics": avg_metrics}


# %% Visualizations
def _overlay_gt_contour(ax, gt_mask: np.ndarray | None) -> None:
    if gt_mask is not None and gt_mask.any():
        ax.contour(gt_mask.astype(float), levels=[0.5], colors="lime", linewidths=1.5)


def plot_heatmap_grid(
    path: Path,
    title: str,
    raws: dict[str, np.ndarray],
    scores: dict[str, float],
    gt_mask: np.ndarray | None,
    chosen: str,
) -> None:
    n = len(raws)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    axes = [axes] if n == 1 else list(axes)
    for ax, (name, raw) in zip(axes, raws.items()):
        ax.imshow(raw, cmap="jet")
        _overlay_gt_contour(ax, gt_mask)
        marker = " *" if name == chosen else ""
        ax.set_title(f"{name}{marker}\nscore={scores[name]:.3f}")
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)


def plot_stage_diagnostic(
    path: Path,
    single_result: dict,
    two_stage_result: dict | None,
    decision_info: dict,
    chosen_stage: str,
    branch: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(single_result["raw"], cmap="jet")
    for c in single_result["pred_clusters"]:
        ys, xs = np.where(c["mask"])
        if len(xs) == 0:
            continue
        axes[0].add_patch(
            plt.Rectangle(
                (xs.min(), ys.min()),
                xs.max() - xs.min() + 1,
                ys.max() - ys.min() + 1,
                fill=False,
                edgecolor="orange",
                linewidth=1.5,
            )
        )
    axes[0].set_title(f"single-stage ({len(single_result['pred_clusters'])} clusters)")
    axes[0].axis("off")

    n_two = len(two_stage_result["pred_clusters"]) if two_stage_result else 0
    axes[1].imshow(single_result["raw"], cmap="jet")
    if two_stage_result:
        for c in two_stage_result["pred_clusters"]:
            mask = c.get("mask")
            if mask is None or mask.sum() == 0:
                continue
            ys, xs = np.where(mask)
            axes[1].add_patch(
                plt.Rectangle(
                    (xs.min(), ys.min()),
                    xs.max() - xs.min() + 1,
                    ys.max() - ys.min() + 1,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.5,
                )
            )
    axes[1].set_title(f"two-stage ({n_two} clusters)")
    axes[1].axis("off")
    fig.suptitle(f"[{branch}] stage decision: {chosen_stage} | {decision_info}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)


def plot_denoise_discarded(
    path: Path,
    protos_by_variant: dict[str, dict[str, ScalePrototype]],
    members: list[str],
    chosen: str,
) -> None:
    variants = ["raw", "step1", "step2_cls", "step2_center"]
    # cleaning.py only ever cleans mid/close (global has no per-instance cluster_crops to
    # show discarded patches for) -- pick the first member scale that actually has some.
    scale = next(
        (s for s in members if protos_by_variant["raw"][s].cluster_crops is not None), None
    )
    if scale is None:
        log.info("skipping %s -- locked scale(s) %s have no per-instance crops", path, members)
        return
    fig, axes = plt.subplots(1, len(variants), figsize=(4 * len(variants), 4))
    for ax, v in zip(axes, variants):
        cc = max(protos_by_variant[v][scale].cluster_crops, key=lambda c: int(c.patch_mask.sum()))
        raw_fg = cc.patch_mask
        kept_fg = cc.fg_select_mask if cc.fg_select_mask is not None else cc.patch_mask
        discarded = raw_fg & ~kept_fg
        img = np.zeros((*raw_fg.shape, 3))
        img[kept_fg] = [0, 1, 0]
        img[discarded] = [1, 0, 0]
        ax.imshow(img)
        marker = " *" if v == chosen else ""
        ax.set_title(f"{v}{marker}\ndiscarded={int(discarded.sum())}")
        ax.axis("off")
    fig.suptitle(f"scale={scale}: fg patches kept (green) vs. discarded by denoising (red)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)


def plot_augmentation_scores(
    path: Path, scores: dict[str, float], chosen: str, branch: str
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(scores.keys())
    values = [scores[n] for n in names]
    colors = ["tab:green" if n == chosen else "tab:blue" for n in names]
    ax.bar(names, values, color=colors)
    ax.set_ylabel("score")
    ax.set_title(f"[{branch}] augmentation comparison (chosen: {chosen})")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)


def render_pair_visualizations(pair_dir: Path, branch_results: dict[str, dict]) -> None:
    for branch, res in branch_results.items():
        fold0 = res["folds"][0]
        diag = fold0["diagnostics"]
        cfg = fold0["config"]
        eval_role = fold0["eval_roles"][0]
        gt_mask = eval_role.gt_patch_mask if branch == "gt_calibrated" else None
        plot_heatmap_grid(
            pair_dir / f"{branch}_scale_heatmaps.png",
            f"[{branch}] scale candidates",
            diag["scale_raws"],
            diag["scale_scores"],
            gt_mask,
            cfg.scale_name,
        )
        plot_heatmap_grid(
            pair_dir / f"{branch}_method_heatmaps.png",
            f"[{branch}] method candidates (scale={cfg.scale_name})",
            diag["method_raws"],
            diag["method_scores"],
            gt_mask,
            cfg.method,
        )
        plot_stage_diagnostic(
            pair_dir / f"{branch}_stage_diagnostic.png",
            diag["single_result"],
            diag["two_stage_result"],
            diag["stage_decision"],
            cfg.stage,
            branch,
        )
        plot_denoise_discarded(
            pair_dir / f"{branch}_denoise_discarded.png",
            diag["clean_protos_by_variant"],
            cfg.scale_members,
            cfg.clean_stage,
        )
        if diag["aug_scores"]:
            plot_augmentation_scores(
                pair_dir / f"{branch}_augmentation_scores.png",
                diag["aug_scores"],
                cfg.augmentation,
                branch,
            )
        else:
            log.info(
                "skipping %s_augmentation_scores.png -- ENABLE_AUGMENTATION_SEARCH=False",
                branch,
            )


def summarize_and_plot(
    all_results: list[tuple[str, dict[str, dict]]], output_dir: Path
) -> pd.DataFrame:
    rows = []
    for pair_label, branch_results in all_results:
        for branch, res in branch_results.items():
            cfg0 = res["folds"][0]["config"]
            rows.append(
                {
                    "pair": pair_label,
                    "branch": branch,
                    "scale": cfg0.scale_name,
                    "method": cfg0.method,
                    "transform": cfg0.transform,
                    "transform_eps": cfg0.transform_eps,
                    "stage": cfg0.stage,
                    "clean_stage": cfg0.clean_stage,
                    "augmentation": cfg0.augmentation,
                    **res["avg_test_metrics"],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "adaptive_selection_summary.csv", index=False)
    log.info("wrote %s", output_dir / "adaptive_selection_summary.csv")

    fig, ax = plt.subplots(figsize=(max(8, len(df["pair"].unique()) * 1.2), 5))
    pivot = df.pivot(index="pair", columns="branch", values="f1")
    pivot.plot.bar(ax=ax)
    ax.set_ylabel("test F1")
    ax.set_title("Adaptive selection: GT-calibrated vs. GT-free (test F1 per pair)")
    fig.savefig(output_dir / "summary_f1_by_pair.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", output_dir / "summary_f1_by_pair.png")
    return df


# %% Per-pair driver
def run_pair(
    part_type: str,
    instance_type: str,
    encoder: DinoEncoder,
    patch_size: int,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    output_dir: Path,
) -> dict | None:
    pair_label = f"{part_type}__{instance_type.replace(' ', '-')}"
    roles = build_pool_role_images(part_type, instance_type, encoder, crop_cfg)
    if len(roles) < MIN_POOL_SIZE:
        log.warning(
            "[%s] pool too small for a real train/val/test split (%d images, need >=%d) "
            "-- skipping",
            pair_label,
            len(roles),
            MIN_POOL_SIZE,
        )
        return None

    pair_seed = SEED + zlib.crc32(pair_label.encode()) % 10_000
    train_roles, val_roles, test_roles = split_train_val_test(roles, seed=pair_seed)
    cv_folds = build_cv_folds(train_roles, val_roles, NUM_CV_FOLDS, seed=pair_seed)

    pair_dir = output_dir / pair_label
    pair_dir.mkdir(parents=True, exist_ok=True)

    branch_results = {}
    for branch in ("gt_calibrated", "gt_free"):
        branch_results[branch] = run_branch(
            pair_label, branch, cv_folds, test_roles, encoder, crop_cfg, scoring_cfg, patch_size
        )

    render_pair_visualizations(pair_dir, branch_results)

    # main() keeps every pair's returned dict alive in all_results until the very end (for
    # summarize_and_plot's cross-pair summary), but that function only reads each fold's
    # "config" (small dataclass) and each branch's "avg_test_metrics" (scalars) — everything
    # else here (train_pool/eval_roles' encoded patch-token tensors, diagnostics' score
    # heatmaps/prototype states) was already consumed by render_pair_visualizations above and
    # is otherwise dead weight. Left in place, running every pair in one process accumulates
    # GPU memory pair over pair until a later pair's own allocation OOMs (observed at pair 8/12
    # on an 11.47GB GPU pre-abc5) — stripping it here is the fix, not a bigger GPU.
    for res in branch_results.values():
        for fold in res["folds"]:
            fold.pop("diagnostics", None)
            fold.pop("train_pool", None)
            fold.pop("eval_roles", None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return branch_results


# %% Main
def main() -> None:
    crop_cfg = DEFAULT_CROP_CONFIG
    scoring_cfg = DEFAULT_SCORING_CONFIG

    encoder = DinoEncoder(
        version=crop_cfg.dino_version,
        size=crop_cfg.dino_size,
        img_size=crop_cfg.img_size,
        weights_dir=DINO_WEIGHTS_DIR,
        amp=True,
    )
    patch_size = encoder.patch_size
    log.info(
        "DINOv%s-%s | patch_size=%d | img_size=%d | NUM_CV_FOLDS=%d",
        crop_cfg.dino_version[1],
        crop_cfg.dino_size,
        patch_size,
        crop_cfg.img_size,
        NUM_CV_FOLDS,
    )

    if RUN_ALL_PAIRS:
        combos = all_combos()
    else:
        combos = [(FOCUS_PART_TYPE, FOCUS_INSTANCE_TYPE)]
        if combos[0] not in all_combos():
            raise RuntimeError(f"No combo found for {FOCUS_PART_TYPE}/{FOCUS_INSTANCE_TYPE}")
    log.info("Running %d combo(s): %s", len(combos), combos)

    all_results: list[tuple[str, dict]] = []
    for part_type, instance_type in combos:
        pair_label = f"{part_type}__{instance_type.replace(' ', '-')}"
        try:
            res = run_pair(
                part_type, instance_type, encoder, patch_size, crop_cfg, scoring_cfg, OUTPUT_DIR
            )
            if res is not None:
                all_results.append((pair_label, res))
        except Exception:
            log.exception("[%s] FAILED", pair_label)

    if all_results:
        summarize_and_plot(all_results, OUTPUT_DIR)
    else:
        log.warning("No pairs completed -- nothing to summarize")


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Reading the results
#
# `outputs/fundamental_abc5/adaptive_method_selection/adaptive_selection_summary.csv` has
# one row per (pair, branch): the chosen scale/method/transform (+eps)/stage/denoising/
# augmentation and the resulting cross-validated test P/R/F1/mIoU — each already a mean
# across every held-out test image, then averaged again across `NUM_CV_FOLDS` train/val
# folds.
# `summary_f1_by_pair.png` plots GT-calibrated vs. GT-free test F1 side by side per pair.
# Per-pair figures under `outputs/fundamental_abc5/adaptive_method_selection/<pair_label>/`
# show the diagnostic behind each decision for that pair's first CV fold's first val image:
# which scale/method scored best and why (heatmaps + scores, averaged across the fold's
# whole val set even though only one image is drawn), what drove the single-vs-two-stage
# call (mean F1 gain across val images for GT-calibrated, or a majority vote of the
# size/fill-ratio proxy for GT-free), which fg/bg patches each denoising stage discarded,
# and how the augmentation candidates compared.
#
# The two branches are not expected to agree on every decision — GT-free is a genuinely
# different, weaker information regime (no annotation, ever, at either calibration or
# deployment time). What's worth checking is whether GT-free's choices are *defensible*
# given only its own proxy signals, and how much test F1/mIoU it gives up relative to
# GT-calibrated for that independence. A wide, consistent gap on an otherwise "easy" pair
# would suggest one of the GT-free proxies (Otsu separability, the size/fill-ratio
# heuristic, or the stability score) needs a better replacement — which is exactly why
# every one of them is a single, swappable function here.
