# %% [markdown]
# # Fundamental: Adaptive Method-Selection Logic for the Multi-Scale Ablation
#
# `experiments/object_detection/multiscale_ablation/` currently picks scale, method,
# single-vs-two-stage, and denoising **manually**: `run_experiments.py` computes every
# registered method (and its two-stage variant) for every pair, and a human reads the
# cached P/R/F1/mIoU table or `visualize_results.py`'s heatmaps to decide which config
# is "best" per (part_type, instance_type). This script replaces that manual reading
# with adaptive selection logic that makes the same 5 decisions itself, in order:
#
#   1. **Scale** — global / mid / close / combinations.
#   2. **Method** — prototype (mean) / knn (fg-bg-knn) / kmeans.
#   3. **Single vs. two-stage** — decided from precision/recall of detections (a
#      genuine two-stage improvement) or, in the GT-free branch, from a geometric
#      "objects too small / too close together" proxy — two-stage carries a real
#      latency/complexity cost, so it only wins when it's actually needed.
#   4. **Denoising** — raw fg/bg galleries vs. the best of `cleaning.py`'s cleaning
#      stages (step1 / step2_cls / step2_center).
#   5. **Augmentation** — no augmentation vs. one family's cumulative-severity
#      composition (e.g. rotation at severity 1, then 1+2, ... pooled together) vs.
#      every family composed at once.
#
# Two parallel branches make these same 5 decisions with different scoring signals:
#
#   - **GT-calibrated** — scores every candidate with `oracle_iou` (steps 1/2/4/5) or
#     real precision/recall (step 3), against a held-out *eval* fold's ground truth.
#   - **GT-free** — never consults any annotation to score a candidate (reference
#     instance masks are still used structurally to build exemplar crops, as they
#     always have been — that's not what "GT-free" judges here). Candidates are scored
#     with unsupervised proxies instead: Otsu separability of the raw score map
#     (steps 1/2/4), a size/fill-ratio heuristic against the reference instance's own
#     known size (step 3), and prediction stability under nuisance perturbation
#     (step 5).
#
# Both branches share one calibration harness (train/eval/test, with cross-validation
# by swapping train/eval and averaging test metrics) — only the scoring signal at each
# step differs, so the comparison is apples-to-apples. `data/abc3` has exactly one real
# reference image and one real query image per part type (see `dinoisawesome/abc3.py`),
# so the query is always the test set and the *reference* image is the entire
# calibration pool — with only one real calibration image, train/eval are two
# synthetically perturbed views of it (a fixed mild rotation+blur, deliberately
# distinct from step 5's own augmentation sweep), swapped for the second
# cross-validation fold. `split_calibration_pool` defines the general rule for when
# more real images become available.
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
from dinoisawesome.abc3 import load_instance_pixel_masks
from dinoisawesome.instance_detection import compute_exemplar_features, extract_patch_tokens

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
from _shared.mask_geometry import (
    mask_iou,  # noqa: E402
    patch_fg_fraction,  # noqa: E402
)
from _shared.prototype_ops import extract_patch_tokens_batch_with_cls  # noqa: E402
from _shared.thresholding import oracle_iou, otsu_threshold  # noqa: E402
from cleaning import apply_fg_cleaning  # noqa: E402
from common import (  # noqa: E402
    DATA_DIR,
    DEFAULT_CROP_CONFIG,
    DEFAULT_SCORING_CONFIG,
    CropConfig,
    PairKey,
    ScoringConfig,
    all_pairs,
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
OUTPUT_DIR = _EXPERIMENTS_ROOT.parent / "outputs" / "fundamental" / "adaptive_method_selection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Runnable on one pair or every pair — see main() at the bottom.
FOCUS_PART_TYPE = "LHa"
FOCUS_INSTANCE_TYPE = "donut foam"
RUN_ALL_PAIRS = False

SEED = 0

# Two-stage carries a real latency/complexity cost — only worth it for a genuine F1 gain.
TWO_STAGE_MIN_GAIN = 0.05
# GT-free single-vs-two-stage geometric heuristic thresholds (see geometric_two_stage_decision).
SMALL_OBJECT_FRACTION = 0.5
MERGE_FILL_RATIO_THRESHOLD = 0.3

# Synthetic train/eval fold padding when only one real calibration image exists (today,
# always) — deliberately mild and distinct from AUGMENTATIONS (step 5's own sweep) so the
# two uses of augmentation never get conflated: this one exists purely to get two folds.
SYNTHETIC_FOLD_ANGLE = 8.0
SYNTHETIC_FOLD_BLUR_RADIUS = 1.5

# Step 5 candidates: "individual composed" (cumulative severities within one family, e.g.
# rotation@[1], @[1,2], ... pooled together — here just "pool every non-zero severity of
# this family at once") vs. "all composed" (every family's severities pooled together).
AUGMENTATIONS: dict[str, dict] = {
    "rotation": {"values": [5.0, 10.0], "apply": apply_rotation},
    "blur": {"values": [1.0, 2.0], "apply": pixel_only(apply_blur)},
    "noise": {"values": [10.0, 20.0], "apply": pixel_only(partial(apply_noise, seed=SEED))},
}

# GT-free step 5 stability proxy: mild single-severity nuisance perturbations applied to
# the eval-fold image; a candidate gallery is "stable" if its predicted fg mask barely
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


# %% Role images (train / eval / test) — one physical or synthetic image + its own GT
@dataclass
class RoleImage:
    tag: str
    image: Image.Image
    instance_masks: list[np.ndarray]
    tokens: torch.Tensor
    grid_h: int
    grid_w: int
    union_pixel_mask: np.ndarray | None
    gt_patch_mask: np.ndarray
    gt_clusters: list[dict]


def make_role_image(
    tag: str,
    image: Image.Image,
    instance_masks: list[np.ndarray],
    encoder: DinoEncoder,
    crop_cfg: CropConfig,
) -> RoleImage:
    tokens, h, w = extract_patch_tokens(encoder, image, crop_cfg.layer_idx, debias=crop_cfg.debias)
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
    return RoleImage(tag, image, instance_masks, tokens, h, w, union, gt_patch_mask, gt_clusters)


def raw_map_for(
    state: MethodState, tokens: torch.Tensor, h: int, w: int, scoring_cfg: ScoringConfig
) -> np.ndarray:
    return score_method(state, tokens, knn_k=scoring_cfg.knn_fgbg_num_neighbours).reshape(h, w)


# %% Data loading + synthetic calibration folds
def load_pair_images_and_masks(
    pair: PairKey,
) -> tuple[Image.Image, Image.Image, list[np.ndarray], list[np.ndarray]]:
    exemplar_class = instance_classes_for(pair.instance_type)
    ref_stem = f"{pair.part_type}_{pair.ref_number}"
    query_stem = f"{pair.part_type}_{pair.query_number}"
    ref_instance_masks = load_instance_pixel_masks(
        DATA_DIR / "annotations" / ref_stem, exemplar_class
    )
    ref_img = Image.open(DATA_DIR / f"{ref_stem}.jpg").convert("RGB")
    query_img = Image.open(DATA_DIR / f"{query_stem}.jpg").convert("RGB")
    q_instance_pixel_masks = load_instance_pixel_masks(
        DATA_DIR / "annotations" / query_stem, exemplar_class
    )
    return ref_img, query_img, ref_instance_masks, q_instance_pixel_masks


def synthetic_second_view(
    image: Image.Image, instance_masks: list[np.ndarray]
) -> tuple[Image.Image, list[np.ndarray]]:
    """A mild rotate+blur view of *image* — used only to pad a size-1 calibration pool
    into two synthetic CV folds (see SYNTHETIC_FOLD_ANGLE/RADIUS's docstring above)."""
    fill = mean_color(image)
    rotated_img = image
    rotated_masks = []
    for m in instance_masks:
        rotated_img, rm = apply_rotation(image, m, SYNTHETIC_FOLD_ANGLE, fill)
        rotated_masks.append(rm)
    rotated_img = apply_blur(rotated_img, SYNTHETIC_FOLD_BLUR_RADIUS)
    return rotated_img, rotated_masks


def split_calibration_pool(
    pool: list[tuple[str, Image.Image, list[np.ndarray]]],
) -> tuple[list, list]:
    """General train/eval split rule for the calibration pool (test image excluded).

    N==1 (abc3's real state today): no split is possible from real images alone — callers
    fall back to synthetic folds (see build_folds). N>=2: split roughly in half, e.g. the
    spec's illustrative train(2)/eval(2)/test(2) once 6 real images exist per pair.
    """
    n = len(pool)
    if n < 2:
        raise ValueError("split_calibration_pool needs >=2 images; use synthetic folds for N==1")
    half = n // 2
    return pool[:half], pool[half:]


def build_folds(pool: list[tuple[str, Image.Image, list[np.ndarray]]]) -> list[tuple]:
    """Two (train, eval) fold assignments, swapped, for cross-validation."""
    if len(pool) == 1:
        tag, img, masks = pool[0]
        synth_img, synth_masks = synthetic_second_view(img, masks)
        log.info(
            "N=1 real calibration image (%s) -- using synthetic CV folds (rot=%.0f, blur=%.1f)",
            tag,
            SYNTHETIC_FOLD_ANGLE,
            SYNTHETIC_FOLD_BLUR_RADIUS,
        )
        return [
            (tag, img, masks, "synthetic", synth_img, synth_masks),
            ("synthetic", synth_img, synth_masks, tag, img, masks),
        ]
    train_pool, eval_pool = split_calibration_pool(pool)
    if len(train_pool) > 1 or len(eval_pool) > 1:
        log.warning(
            "train/eval pool has >1 image (%d/%d) -- using only the first of each; true "
            "multi-image gallery pooling is future work once more real images exist",
            len(train_pool),
            len(eval_pool),
        )
    t_tag, t_img, t_masks = train_pool[0]
    e_tag, e_img, e_masks = eval_pool[0]
    return [
        (t_tag, t_img, t_masks, e_tag, e_img, e_masks),
        (e_tag, e_img, e_masks, t_tag, t_img, t_masks),
    ]


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
        bg_bank = pool_scale_patches(scale_protos, bg_scales, want_fg=False)
        return MethodState(
            name, "knn_fgbg", fg_bank=fg_bank, bg_bank=bg_bank, roi_source_method=name
        )
    if method == "kmeans":
        fg = pool_scale_patches(scale_protos, members, want_fg=True)
        k = min(kmeans_k or 8, fg.shape[0])
        centroids = compute_exemplar_features(fg, mode="kmeans", k=k)
        return MethodState(name, "multi", centroids, roi_source_method=name)
    raise ValueError(f"Unknown method: {method!r}")


# %% Augmentation injection (step 5) — appends perturbed ClusterCrops built from the
# TRAIN role's own crops, so method_state_for's pooling picks them up transparently.
def augmented_extra_crops(
    base_crops: list[ClusterCrop],
    train_img: Image.Image,
    family_specs: list[tuple[str, float, object]],
    encoder: DinoEncoder,
    crop_cfg: CropConfig,
) -> list[ClusterCrop]:
    extra: list[ClusterCrop] = []
    fill = mean_color(train_img)
    for cc in base_crops:
        crop_img = train_img.crop(cc.box)
        for _family, val, fn in family_specs:
            aug_img, aug_mask_px = fn(crop_img, cc.own_mask_px, val, fill)
            ((tokens, cls, gh, gw),) = extract_patch_tokens_batch_with_cls(
                encoder, [aug_img], crop_cfg.layer_idx, crop_cfg.debias
            )
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
    members: list[str],
    candidate: str,
    train_img: Image.Image,
    encoder: DinoEncoder,
    crop_cfg: CropConfig,
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
        # "global" has no per-instance cluster_crops (a single whole-image crop, not a
        # per-instance list — see ScalePrototype's docstring) — nothing to extend there;
        # augmentation only ever grows the mid/close per-instance galleries.
        if base_protos[scale].cluster_crops is None:
            continue
        extra = augmented_extra_crops(
            base_protos[scale].cluster_crops, train_img, family_specs, encoder, crop_cfg
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
) -> float:
    """GT-free step 5 scorer: mean IoU of the (Otsu-binarised) predicted fg mask between
    the clean eval image and each mildly-perturbed view -- higher = more stable."""
    clean_raw = raw_map_for(
        state, eval_role.tokens, eval_role.grid_h, eval_role.grid_w, scoring_cfg
    )
    clean_mask = clean_raw > otsu_threshold(clean_raw)
    fill = mean_color(eval_role.image)
    dummy_mask = np.zeros((eval_role.image.height, eval_role.image.width), dtype=bool)
    ious = []
    for _tag, val, fn in NUISANCE_PERTURBATIONS:
        pert_img, _ = fn(eval_role.image, dummy_mask, val, fill)
        tokens, h, w = extract_patch_tokens(
            encoder, pert_img, crop_cfg.layer_idx, debias=crop_cfg.debias
        )
        raw = score_method(state, tokens, knn_k=scoring_cfg.knn_fgbg_num_neighbours).reshape(h, w)
        mask = raw > otsu_threshold(raw)
        if mask.shape != clean_mask.shape:
            continue
        ious.append(mask_iou(clean_mask, mask))
    return float(np.mean(ious)) if ious else 0.0


# %% Decision step 1 — scale
def select_scale(
    scale_protos: dict[str, ScalePrototype],
    eval_role: RoleImage,
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
        raw = raw_map_for(state, eval_role.tokens, eval_role.grid_h, eval_role.grid_w, scoring_cfg)
        raws[name] = raw
        scores[name] = raw_score(branch, raw, eval_role.gt_patch_mask, scoring_cfg)
    best = max(scores, key=scores.get)
    return best, candidates[best], raws, scores


# %% Decision step 2 — method
def select_method(
    scale_protos: dict[str, ScalePrototype],
    members: list[str],
    eval_role: RoleImage,
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
                raw = raw_map_for(
                    state, eval_role.tokens, eval_role.grid_h, eval_role.grid_w, scoring_cfg
                )
                s = raw_score(branch, raw, eval_role.gt_patch_mask, scoring_cfg)
                if s > best_s:
                    best_k, best_raw, best_s = k, raw, s
            raws["kmeans"], scores["kmeans"] = best_raw, best_s
            best_kmeans_k = best_k
        else:
            state = method_state_for(scale_protos, members, method, name=method)
            raw = raw_map_for(
                state, eval_role.tokens, eval_role.grid_h, eval_role.grid_w, scoring_cfg
            )
            raws[method] = raw
            scores[method] = raw_score(branch, raw, eval_role.gt_patch_mask, scoring_cfg)
    best = max(scores, key=scores.get)
    return best, (best_kmeans_k if best == "kmeans" else None), raws, scores


# %% Decision step 3 — single vs. two-stage
def geometric_two_stage_decision(
    pred_clusters: list[dict], expected_size: float
) -> tuple[str, dict]:
    """GT-free proxy: flag "too small" (predicted clusters undersized vs. the reference
    instance's own known size) or "too close" (a cluster's bbox fill-ratio is anomalously
    low, suggesting several instances merged into one blob)."""
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
    locked_scale_name: str,
    train_scale_protos: dict[str, ScalePrototype],
    train_role: RoleImage,
    eval_role: RoleImage,
    encoder: DinoEncoder,
    mean_patch_prototype: torch.Tensor,
    min_cs: int,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    branch: str,
    patch_size: int,
) -> tuple[str, dict, dict, dict]:
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
    target_size_frac = (
        train_scale_protos[locked_scale_name].target_size_frac
        if locked_scale_name in train_scale_protos
        else None
    )
    two_stage = evaluate_two_stage(
        state,
        single["raw"],
        eval_role,
        encoder,
        crop_cfg,
        scoring_cfg,
        mean_patch_prototype,
        min_cs,
        single,
        patch_size,
        target_size_frac,
        branch,
    )
    if branch == "gt_calibrated":
        gain = (
            two_stage["metrics"]["f1"] - single["metrics"]["f1"]
            if (two_stage["metrics"] is not None and single["metrics"] is not None)
            else -1.0
        )
        stage = "two_stage" if gain > TWO_STAGE_MIN_GAIN else "single"
        decision_info = {"f1_gain": gain}
    else:
        mid_crops = train_scale_protos["mid"].cluster_crops
        expected_size = (
            float(np.median([cc.patch_mask.sum() for cc in mid_crops]))
            if mid_crops
            else float(min_cs)
        )
        stage, decision_info = geometric_two_stage_decision(single["pred_clusters"], expected_size)
    return stage, single, two_stage, decision_info


# %% Decision step 4 — denoising
def select_denoising(
    base_scale_protos: dict[str, ScalePrototype],
    members: list[str],
    method: str,
    kmeans_k: int | None,
    eval_role: RoleImage,
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
        raw = raw_map_for(state, eval_role.tokens, eval_role.grid_h, eval_role.grid_w, scoring_cfg)
        raws[v] = raw
        scores[v] = raw_score(branch, raw, eval_role.gt_patch_mask, scoring_cfg)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    best_clean = max(("step1", "step2_cls", "step2_center"), key=scores.get)
    chosen = "raw" if scores["raw"] >= scores[best_clean] else best_clean
    return chosen, protos_by_variant, raws, scores


# %% Decision step 5 — augmentation
def select_augmentation(
    base_protos: dict[str, ScalePrototype],
    members: list[str],
    method: str,
    kmeans_k: int | None,
    train_img: Image.Image,
    eval_role: RoleImage,
    branch: str,
    encoder: DinoEncoder,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
) -> tuple[str, dict[str, float]]:
    candidates = ["none", *AUGMENTATIONS.keys(), "all"]
    scores = {}
    for cand in candidates:
        protos = protos_with_augmentation(base_protos, members, cand, train_img, encoder, crop_cfg)
        state = method_state_for(protos, members, method, kmeans_k=kmeans_k, name=cand)
        if branch == "gt_calibrated":
            raw = raw_map_for(
                state, eval_role.tokens, eval_role.grid_h, eval_role.grid_w, scoring_cfg
            )
            scores[cand] = oracle_iou(raw, eval_role.gt_patch_mask, scoring_cfg.ref_threshold_steps)
        else:
            scores[cand] = stability_score(state, eval_role, encoder, crop_cfg, scoring_cfg)
        del protos, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    best = max(scores, key=scores.get)
    return best, scores


# %% Fold / branch orchestration
@dataclass
class SelectionConfig:
    scale_name: str
    scale_members: list[str]
    method: str
    kmeans_k: int | None
    stage: str
    clean_stage: str
    augmentation: str


def run_fold(
    train_role: RoleImage,
    eval_role: RoleImage,
    encoder: DinoEncoder,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    branch: str,
    patch_size: int,
) -> tuple[SelectionConfig, dict]:
    base_protos, mean_patch_prototype = build_all_scale_prototypes(
        encoder, train_role.image, train_role.instance_masks, crop_cfg
    )
    mid_sizes = np.array([cc.patch_mask.sum() for cc in base_protos["mid"].cluster_crops])
    min_cs = min_cluster_size_bound(
        mid_sizes, scoring_cfg.cluster_size_margin, scoring_cfg.min_points_floor
    )

    scale_name, scale_members, scale_raws, scale_scores = select_scale(
        base_protos, eval_role, branch, scoring_cfg
    )
    method, kmeans_k, method_raws, method_scores = select_method(
        base_protos, scale_members, eval_role, branch, scoring_cfg
    )
    locked_state = method_state_for(
        base_protos, scale_members, method, kmeans_k=kmeans_k, name="locked"
    )
    stage, single_result, two_stage_result, stage_decision = select_stage(
        locked_state,
        scale_name,
        base_protos,
        train_role,
        eval_role,
        encoder,
        mean_patch_prototype,
        min_cs,
        crop_cfg,
        scoring_cfg,
        branch,
        patch_size,
    )
    clean_stage, clean_protos_by_variant, clean_raws, clean_scores = select_denoising(
        base_protos, scale_members, method, kmeans_k, eval_role, branch, crop_cfg, scoring_cfg
    )
    final_protos = clean_protos_by_variant[clean_stage]
    augmentation, aug_scores = select_augmentation(
        final_protos,
        scale_members,
        method,
        kmeans_k,
        train_role.image,
        eval_role,
        branch,
        encoder,
        crop_cfg,
        scoring_cfg,
    )

    cfg = SelectionConfig(
        scale_name, scale_members, method, kmeans_k, stage, clean_stage, augmentation
    )
    diagnostics = {
        "scale_raws": scale_raws,
        "scale_scores": scale_scores,
        "method_raws": method_raws,
        "method_scores": method_scores,
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
    train_role: RoleImage,
    test_role: RoleImage,
    encoder: DinoEncoder,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    patch_size: int,
) -> tuple[dict, dict, dict | None]:
    """Applies *cfg* (chosen by *branch*, using only train_role) to the held-out test
    image, faithfully: the GT-free branch's own Otsu-based thresholding/rejection is used
    end-to-end here too (a true zero-annotation config never gets to call the
    GT-tuned threshold at deployment either) — GT is used only afterwards, by
    match_and_score, purely to grade the outcome for reporting.
    """
    base_protos, mean_patch_prototype = build_all_scale_prototypes(
        encoder, train_role.image, train_role.instance_masks, crop_cfg
    )
    protos = (
        base_protos
        if cfg.clean_stage == "raw"
        else apply_fg_cleaning(
            base_protos, crop_cfg, dataclasses.replace(scoring_cfg, fg_clean_stage=cfg.clean_stage)
        )
    )
    protos = protos_with_augmentation(
        protos, cfg.scale_members, cfg.augmentation, train_role.image, encoder, crop_cfg
    )
    state = method_state_for(
        protos, cfg.scale_members, cfg.method, kmeans_k=cfg.kmeans_k, name="final"
    )
    mid_sizes = np.array([cc.patch_mask.sum() for cc in protos["mid"].cluster_crops])
    min_cs = min_cluster_size_bound(
        mid_sizes, scoring_cfg.cluster_size_margin, scoring_cfg.min_points_floor
    )

    single = evaluate_single_stage(
        state,
        cfg.scale_name,
        protos,
        train_role,
        test_role,
        branch,
        mean_patch_prototype,
        min_cs,
        crop_cfg,
        scoring_cfg,
        compute_metrics=True,
    )
    if cfg.stage == "single":
        return single["metrics"], single, None
    target_size_frac = protos[cfg.scale_name].target_size_frac if cfg.scale_name in protos else None
    two_stage = evaluate_two_stage(
        state,
        single["raw"],
        test_role,
        encoder,
        crop_cfg,
        scoring_cfg,
        mean_patch_prototype,
        min_cs,
        single,
        patch_size,
        target_size_frac,
        branch,
        compute_metrics=True,
    )
    return two_stage["metrics"], single, two_stage


def run_branch(
    pair: PairKey,
    branch: str,
    folds: list[tuple],
    test_role: RoleImage,
    encoder: DinoEncoder,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    patch_size: int,
) -> dict:
    fold_results = []
    for i, (t_tag, t_img, t_masks, e_tag, e_img, e_masks) in enumerate(folds):
        train_role = make_role_image(f"train[{t_tag}]", t_img, t_masks, encoder, crop_cfg)
        eval_role = make_role_image(f"eval[{e_tag}]", e_img, e_masks, encoder, crop_cfg)
        cfg, diagnostics = run_fold(
            train_role, eval_role, encoder, crop_cfg, scoring_cfg, branch, patch_size
        )
        test_metrics, _test_single, _test_two_stage = evaluate_on_test(
            cfg, branch, train_role, test_role, encoder, crop_cfg, scoring_cfg, patch_size
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fold_results.append(
            {
                "fold": i,
                "config": cfg,
                "diagnostics": diagnostics,
                "test_metrics": test_metrics,
                "train_role": train_role,
                "eval_role": eval_role,
            }
        )
        log.info(
            "[%s/%s] fold%d train=%s eval=%s -> scale=%s method=%s stage=%s clean=%s aug=%s "
            "| test P=%.2f R=%.2f F1=%.2f mIoU=%.2f",
            pair.slug,
            branch,
            i,
            t_tag,
            e_tag,
            cfg.scale_name,
            cfg.method,
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
        eval_role = fold0["eval_role"]
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
        plot_augmentation_scores(
            pair_dir / f"{branch}_augmentation_scores.png",
            diag["aug_scores"],
            cfg.augmentation,
            branch,
        )


def summarize_and_plot(
    all_results: list[tuple[str, dict[str, dict]]], output_dir: Path
) -> pd.DataFrame:
    rows = []
    for pair_slug, branch_results in all_results:
        for branch, res in branch_results.items():
            cfg0 = res["folds"][0]["config"]
            rows.append(
                {
                    "pair": pair_slug,
                    "branch": branch,
                    "scale": cfg0.scale_name,
                    "method": cfg0.method,
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
    pair: PairKey,
    encoder: DinoEncoder,
    patch_size: int,
    crop_cfg: CropConfig,
    scoring_cfg: ScoringConfig,
    output_dir: Path,
) -> dict | None:
    ref_img, query_img, ref_instance_masks, q_instance_pixel_masks = load_pair_images_and_masks(
        pair
    )
    if not ref_instance_masks:
        log.warning("[%s] no exemplar instances -- skipping", pair.slug)
        return None
    if not q_instance_pixel_masks:
        log.warning(
            "[%s] no query instances -- skipping (no test-set GT to report against)", pair.slug
        )
        return None

    pool = [("ref", ref_img, ref_instance_masks)]
    folds = build_folds(pool)
    test_role = make_role_image("test", query_img, q_instance_pixel_masks, encoder, crop_cfg)

    pair_dir = output_dir / pair.slug
    pair_dir.mkdir(parents=True, exist_ok=True)

    branch_results = {}
    for branch in ("gt_calibrated", "gt_free"):
        branch_results[branch] = run_branch(
            pair, branch, folds, test_role, encoder, crop_cfg, scoring_cfg, patch_size
        )

    render_pair_visualizations(pair_dir, branch_results)
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
        "DINOv%s-%s | patch_size=%d | img_size=%d",
        crop_cfg.dino_version[1],
        crop_cfg.dino_size,
        patch_size,
        crop_cfg.img_size,
    )

    if RUN_ALL_PAIRS:
        pairs = all_pairs()
    else:
        pairs = [
            p
            for p in all_pairs()
            if p.part_type == FOCUS_PART_TYPE and p.instance_type == FOCUS_INSTANCE_TYPE
        ]
        if not pairs:
            raise RuntimeError(f"No pair found for {FOCUS_PART_TYPE}/{FOCUS_INSTANCE_TYPE}")
    log.info("Running %d pair(s): %s", len(pairs), [p.slug for p in pairs])

    all_results: list[tuple[str, dict]] = []
    for pair in pairs:
        try:
            res = run_pair(pair, encoder, patch_size, crop_cfg, scoring_cfg, OUTPUT_DIR)
            if res is not None:
                all_results.append((pair.slug, res))
        except Exception:
            log.exception("[%s] FAILED", pair.slug)

    if all_results:
        summarize_and_plot(all_results, OUTPUT_DIR)
    else:
        log.warning("No pairs completed -- nothing to summarize")


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Reading the results
#
# `outputs/fundamental/adaptive_method_selection/adaptive_selection_summary.csv` has one
# row per (pair, branch): the chosen scale/method/stage/denoising/augmentation and the
# resulting cross-validated test P/R/F1/mIoU. `summary_f1_by_pair.png` plots GT-calibrated
# vs. GT-free test F1 side by side per pair. Per-pair figures under
# `outputs/fundamental/adaptive_method_selection/<pair_slug>/` show the diagnostic behind
# each decision for that pair's first CV fold: which scale/method scored best and why
# (heatmaps + scores), what drove the single-vs-two-stage call (real F1 gain for
# GT-calibrated, or the size/fill-ratio symptoms for GT-free), which fg/bg patches each
# denoising stage discarded, and how the augmentation candidates compared.
#
# The two branches are not expected to agree on every decision — GT-free is a genuinely
# different, weaker information regime (no annotation, ever, at either calibration or
# deployment time). What's worth checking is whether GT-free's choices are *defensible*
# given only its own proxy signals, and how much test F1/mIoU it gives up relative to
# GT-calibrated for that independence. A wide, consistent gap on an otherwise "easy" pair
# would suggest one of the GT-free proxies (Otsu separability, the size/fill-ratio
# heuristic, or the stability score) needs a better replacement — which is exactly why
# every one of them is a single, swappable function here.
