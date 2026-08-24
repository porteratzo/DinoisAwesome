# %% [markdown]
# # Multi-Scale Crop Prototype Ablation + Cross-Scale Similarity
#
# Builds three exemplar prototypes from the *same* instance mask, at three spatial
# scales (mirrors ``scripts/multiscale_detection.py``'s close/mid/full crops, but
# adds ground-truth IoU evaluation and a cross-scale similarity study on top):
#
#   - **global** — whole reference image, masked-mean prototype (today's baseline).
#   - **mid**    — crop halfway between the tight mask bbox and the full image.
#   - **close**  — tight crop around the mask bbox (+ padding), i.e. a close-up.
#
# Each scale is re-encoded at native crop resolution (like the stage-2 refinement in
# ``density_map_methods.py``) so "close" really is a higher-detail view of the object,
# not just the same tokens spatially subset from the global pass.
#
# Two experiments share this exemplar setup:
#
# 1. **Ablation** — score the full query image with every combination of the three
#    scale prototypes (single-scale and multi-scale max-similarity), then run the
#    same threshold -> DBSCAN -> IoU-match pipeline ``density_map_methods.py`` uses for its
#    GT clusters (this file defaults to DBSCAN for prediction clusters too, see
#    ``dbscan_clusters``), get precision/recall/F1/mIoU per method. Results are shown as a
#    bar chart and as
#    an IoU heatmap (method x part type).
#
# 2. **Cross-scale similarity** — for each single scale, find ROI blobs in *that
#    scale's own* full-query raw score map (Otsu + connected components, exactly the
#    stage-2 ROI mechanism used elsewhere in this repo), crop+re-encode each blob at
#    native resolution, then score every blob with *all three* prototypes (not just
#    the one that found it). The diagonal of the resulting scale x scale matrix is
#    the existing two-stage behaviour (global finds+scores its own crop); the
#    off-diagonal cells are the actual cross-similarity question: how well does a
#    global prototype score a close-up crop, and how well does a close-up prototype
#    score generalise back onto a coarser view?

# %% Logging — must be before torch import
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from matplotlib.patches import Rectangle
from PIL import Image
from scipy import ndimage

from dinoisawesome import DinoEncoder
from dinoisawesome.abc3 import PART_TYPES, load_instance_pixel_mask, load_instance_pixel_masks
from dinoisawesome.instance_detection import compute_exemplar_features, extract_patch_tokens

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared.clustering import (  # noqa: E402
    dbscan_clusters as _shared_dbscan_clusters,
)
from _shared.clustering import (
    dbscan_clusters_from_mask,
    match_and_score,
    patch_radius_to_eps,
)
from _shared.clustering import (
    min_cluster_size_bound as _shared_min_cluster_size_bound,
)
from _shared.crop_utils import pad_and_floor_crop_box  # noqa: E402
from _shared.gt_utils import (
    gt_instance_patch_sizes as _shared_gt_instance_patch_sizes,  # noqa: E402
)
from _shared.mask_geometry import (  # noqa: E402
    blob_patch_bbox,
    connected_component_blobs,
    mask_iou,
    patch_bbox_to_native_px,
    pixel_mask_to_patch_mask,
    scale_crop_box,
)
from _shared.prototype_ops import extract_patch_tokens_batch, knn_fgbg_score  # noqa: E402
from _shared.thresholding import iou_threshold_curve, iou_tuned_threshold  # noqa: E402
from _shared.thresholding import roi_binary_mask as _shared_roi_binary_mask  # noqa: E402
from _shared.two_stage import (  # noqa: E402
    annotate_cluster_rejection,
    crop_patch_centers_to_native_px,
    merge_overlapping_clusters,
    project_crop_mask_to_query_grid,
)
from _shared.two_stage import (
    blob_crop_gt_mask as _shared_blob_crop_gt_mask,
)
from _shared.two_stage import (
    cross_score_blobs as _shared_cross_score_blobs,
)
from _shared.two_stage import (
    tune_cluster_reject_threshold as _shared_tune_cluster_reject_threshold,
)
from _shared.two_stage import (
    two_stage_predicted_clusters as _shared_two_stage_predicted_clusters,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("multiscale_crop_ablation")

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

data_dir = _REPO_ROOT / "data" / "abc3"

# PART_TYPES imported from dinoisawesome.abc3 (shared with augmented_prototype_oracle_iou.py's
# batch mode)
REF_NUMBER = 1
QUERY_NUMBER = 2
FOCUS_PART_TYPE = "RHa"  # part type used for the detailed per-method / crop figures
# Annotated classes per part type in data/abc3 — EXEMPLAR_CLASS must be present in whichever
# part type(s) RUN_PART_TYPES selects, or run_pair skips them (no exemplar instances) and
# results/metrics_rows end up empty:
#   LHa, RHa — foam, donut foam, donut foam single           (no velcro, no white clips)
#   LHb, RHb — foam, donut foam single, velcro, white clips
# EXEMPLAR_CLASS: list[str] | None = ["white clips"]  # None = all classes
# EXEMPLAR_CLASS: list[str] | None = ["donut foam single", "donut foam"]  # None = all classes
EXEMPLAR_CLASS: list[str] | None = ["foam"]  # None = all classes
# EXEMPLAR_CLASS: list[str] | None = ["velcro"]  # None = all classes

# run_pair does several encoder forward passes per part type (ref multi-scale crops, the query,
# and each scale's ROI blob crops) — the dominant cost of a full notebook re-run. All the
# mid-scale detail cells (threshold tuning, crop-making, per-crop breakdown, DBSCAN
# tuning) only ever look at results[FOCUS_PART_TYPE], so set this to just [FOCUS_PART_TYPE] for
# fast iteration on those; the ablation bar chart/heatmap and cross-scale similarity cells above
# them will simply summarise fewer part types. Must include FOCUS_PART_TYPE.
# RUN_PART_TYPES: list[str] = PART_TYPES
RUN_PART_TYPES: list[str] = [FOCUS_PART_TYPE]

DINO_VERSION = "v3"
DINO_SIZE = "large"
IMG_SIZE = 1024
LAYER_IDX = 23
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")
DEBIAS = True  # whether to apply positional debiasing when extracting patch tokens

MASK_PATCH_THRESHOLD = 0.3

SCALES: list[str] = ["global", "mid", "close"]
SCALE_COLOR: dict[str, str] = {"global": "#2ecc71", "mid": "#f39c12", "close": "#e74c3c"}
EXEMPLAR_CLOSE_PADDING_FRACTION = 1.0  # exemplar "close" crop pad, fraction of mask bbox extent

# Below this many pixels on either side, a "close" crop is too degenerate to encode
# usefully (the encoder would just upsample a sliver to img_size). Dropped when too
# small; "global" and "mid" are never dropped, so "mid" becomes the last usable scale.
MIN_CROP_SIZE = 128

ABLATION_COMBOS: dict[str, list[str]] = {
    # global is pretty bad and combining it doesn't help, calculate it anyway for completeness
    "global": ["global"],
    "mid": ["mid"],
    "close": ["close"],
    # "global+mid": ["global", "mid"],
    # "global+close": ["global", "close"],
    # "mid+close": ["mid", "close"], mid+close close seams not as good as mid, so lets drop it
    # "global+mid+close": ["global", "mid", "close"],
}

# fg-bg-mean, fg-bg-proto and fg-bg-knn are built separately from the single-scale combos above
# (they need per-scale foreground/background prototypes or galleries, not just
# scale_protos[m].prototype — see build_fgbg_states, build_fgbg_multiproto_states and
# build_knn_fgbg_states) and are configurable the same way ABLATION_COMBOS is: each key names a
# (fg source scales, bg source scales) pair. Single-fg-scale combos build a "fg-bg-mean(...)"
# (both sides collapsed to one mean vector); multi-fg-scale combos build a "fg-bg-proto(...)"
# instead (each fg scale's own mean prototype kept as a separate row, not averaged into one —
# see build_fgbg_multiproto_states for why). "fg-bg-knn(...)" is built for every combo either
# way, since it never collapses to a mean on either side.
FGBG_SOURCE_COMBOS: dict[str, dict[str, list[str]]] = {
    "global/all": {"fg": ["global"], "bg": ["global", "mid", "close"]},
    "mid/all": {"fg": ["mid"], "bg": ["global", "mid", "close"]},
    "global+mid/all": {"fg": ["global", "mid"], "bg": ["global", "mid", "close"]},
    "mid+close/all": {"fg": ["mid", "close"], "bg": ["global", "mid", "close"]},
    "global+mid+close/all": {
        "fg": ["global", "mid", "close"],
        "bg": ["global", "mid", "close"],
    },
}

# How a multi-scale combo's per-scale prototypes are combined (single-scale entries are
# unaffected either way — there's nothing to combine). "max" keeps each scale's prototype
# separate and takes the per-patch max cosine similarity across them (an ensemble — different
# patches can be "won" by different scales). "mean" instead renormalises the combo's scale
# prototypes into one vector up front, so scoring is one cosine similarity per patch, same as
# any single-scale method. A list, not a single mode: every mode listed here is built for every
# multi-scale combo, so e.g. "mid+close-max" and "mid+close-mean" can both be compared in the
# same ablation run. See build_ablation_states / score_method / ablation_method_names.
MULTI_SCALE_COMBINE: list[Literal["max", "mean"]] = ["max", "mean"]

# k-means variant of the single-scale mean methods above: same pooled foreground patches
# per scale, but k centroids (multi-row, max-cosine per query patch) instead of one mean
# vector. One "<scale>-kmeans<k>" method per (scale in SCALES, k in KMEANS_KS) — see
# build_kmeans_states.
KMEANS_KS: tuple[int, ...] = (3, 8)

# The k-means families (build_kmeans_states' "<scale>-kmeans<k>" and build_kmeans_fgbg_states'
# "fg-bg-kmeans<k>(...)") consistently score worse than the mean/proto/knn families in these
# runs, so they're skipped by default. Flip to True to re-enable them for a comparison run —
# nothing else needs to change, both builders stay registered below.
INCLUDE_KMEANS_METHODS: bool = False


def ablation_method_names(
    combos: dict[str, list[str]], combine_modes: list[Literal["max", "mean"]]
) -> list[str]:
    """Expand each combo into the method name(s) build_ablation_states will produce for it.

    Single-scale combos are unaffected by combine_modes — there's only one possible state, named
    after the combo itself. Multi-scale combos get one name per mode in combine_modes, suffixed
    "-<mode>" (e.g. "mid+close-max", "mid+close-mean"), matching the states build_ablation_states
    actually builds. Shared by METHOD_DISPLAY_ORDER (computed before scale_protos exists, so it
    can only use the static combo definitions) and build_ablation_states (computed per reference
    image, from the scales actually present) so the two never drift apart.
    """
    names = []
    for combo_name, members in combos.items():
        if len(members) == 1:
            names.append(combo_name)
        else:
            names.extend(f"{combo_name}-{mode}" for mode in combine_modes)
    return names


# "fg-bg-mean" and "fg-bg-knn" are built separately (need background prototypes/galleries, not
# just the scale_protos[m].prototype the combos above key off — see build_fgbg_states and
# build_knn_fgbg_states) but share display ordering with the combos. "two-stage(<scale>)" is
# separate again — it's the diagonal of the cross-scale matrix (that scale finds its own ROI
# blobs *and* re-scores them), added per scale in SCALES rather than as an ABLATION_COMBOS entry
# since it isn't a single-pass prototype method — see two_stage_predicted_clusters.
_KMEANS_SCALE_NAMES = [f"{scale}-kmeans{k}" for scale in SCALES for k in KMEANS_KS]

METHOD_DISPLAY_ORDER: list[str] = (
    ablation_method_names(ABLATION_COMBOS, MULTI_SCALE_COMBINE)
    + (_KMEANS_SCALE_NAMES if INCLUDE_KMEANS_METHODS else [])
    + [f"fg-bg-mean({n})" for n, c in FGBG_SOURCE_COMBOS.items() if len(c["fg"]) == 1]
    + [f"fg-bg-proto({n})" for n, c in FGBG_SOURCE_COMBOS.items() if len(c["fg"]) > 1]
    + [f"fg-bg-knn({name})" for name in FGBG_SOURCE_COMBOS]
    + (
        [f"fg-bg-kmeans{k}({name})" for name in FGBG_SOURCE_COMBOS for k in KMEANS_KS]
        if INCLUDE_KMEANS_METHODS
        else []
    )
    + [f"two-stage({s})" for s in SCALES]
    + [f"two-stage(fg-bg-mean({n}))" for n, c in FGBG_SOURCE_COMBOS.items() if len(c["fg"]) == 1]
    + [f"two-stage(fg-bg-proto({n}))" for n, c in FGBG_SOURCE_COMBOS.items() if len(c["fg"]) > 1]
    + [f"two-stage(fg-bg-knn({name}))" for name in FGBG_SOURCE_COMBOS]
    + (
        [f"two-stage(fg-bg-kmeans{k}({name}))" for name in FGBG_SOURCE_COMBOS for k in KMEANS_KS]
        if INCLUDE_KMEANS_METHODS
        else []
    )
)

# fg-bg-knn: k for the per-patch kNN gallery lookup (see build_knn_fgbg_states / knn_fgbg_score).
# Same idea as num_neighbours in dinoisawesome.anomaly_head.AnomalyHead, but contrastive
# (fg gallery top-k minus bg gallery top-k) instead of a pure nearest-neighbour distance.
KNN_FGBG_NUM_NEIGHBOURS = 10

REF_THRESHOLD_STEPS = 25  # ref_iou: number of candidate thresholds searched


# patch_radius_to_eps (imported from _shared.clustering above): DBSCAN's eps is a Euclidean
# radius, but "n pixels away, counting diagonals" is a Chebyshev (chessboard) distance — the
# two only agree at the block's corner. A point n pixels out on the diagonal sits at Euclidean
# distance n*sqrt(2), the largest gap DBSCAN must bridge to treat an n-pixel step as
# "connected". n=1 -> eps spans the immediate 8-neighborhood (matches GT_DBSCAN_EPS's own
# ~1.4142, rounded up to 1.5). Only exact for n<=2 — at n>=3 the corner distance n*sqrt(2)
# exceeds n+1 (the nearest excluded axis-aligned point), so eps starts leaking in points
# beyond the intended square block; treat larger n as "bridges gaps up to n pixels", not a
# hard radius.

# CLUSTER_SIZE_MARGIN: int -> flat patch-count offset (k_min - margin);
# float -> fraction of the bound (k_min - margin*k_min). See min_cluster_size_bound.
CLUSTER_SIZE_MARGIN: int | float = 0.5
GT_DBSCAN_EPS = 1.5  # patch units — ~8-connectivity on the grid
GT_DBSCAN_MIN_SAMPLES = 1  # every point is core -> connected components
IOU_MATCH_THRESHOLD = 0.3
MIN_POINTS_FLOOR = 2  # fewer foreground points than this -> don't bother clustering

# CLUSTER_REJECT_MARGIN_FRACTION: how far below the smallest "good" ref cluster's mean-patch
# score to place the cluster-reject cutoff, as a fraction of the gap down to the largest "wrong"
# (bad) ref cluster's score — gives borderline query clusters some slack instead of cutting
# exactly at the weakest good example, while the clamp in tune_cluster_reject_threshold keeps the
# cutoff from ever dropping to or below a known-bad cluster's score.
CLUSTER_REJECT_MARGIN_FRACTION = 0.2

# Predicted (noisy) foreground clusters, unlike GT_DBSCAN_EPS's exact known mask, benefit from
# bridging a slightly wider gap than immediate 8-connectivity so one instance doesn't fragment
# into several clusters around isolated missed patches — patch_radius_to_eps(2) rather than (1).
# Tune by changing PRED_DBSCAN_EPS_PATCHES (try 1, 2, 5 — see patch_radius_to_eps's docstring for
# what each radius means in Euclidean terms).
PRED_DBSCAN_EPS_PATCHES = 2
PRED_DBSCAN_EPS = patch_radius_to_eps(PRED_DBSCAN_EPS_PATCHES)
PRED_DBSCAN_MIN_SAMPLES = 2

# Cross-scale similarity: ROI blobs are found (unsupervised) in a single scale's own
# full-query raw score map, then their native-resolution crops are re-scored with
# every scale's prototype. Mirrors the stage-2 ROI mechanism in density_map_methods.py.
ROI_BINARIZE_METHOD: Literal["otsu", "single", "percentile"] = "percentile"
ROI_SINGLE_THRESHOLD = 0.5  # used when ROI_BINARIZE_METHOD == "single"
ROI_PERCENTILE = 98.0  # used when ROI_BINARIZE_METHOD == "percentile"; keep top (100-p)% of raw
ROI_MORPH_CLOSE_ITERATIONS = (
    2  # binary_closing passes on roi_mask before blob extraction; 0 disables
)

# ROI_CROP_MODE: "per_blob" crops each connected ROI component separately (current
# stage-2 behaviour, one re-scored crop per blob); "single_bbox" merges every ROI
# pixel into one bounding box and produces a single crop covering all segmented parts.
ROI_CROP_MODE: Literal["per_blob", "single_bbox"] = "per_blob"
BLOB_CROP_PADDING_FRACTION = 0.2  # padding around a blob bbox, fraction of its own width/height
MAX_UPSCALE_FACTOR = 2.0  # crop native-pixel side floor = IMG_SIZE / MAX_UPSCALE_FACTOR

SEED = 0
torch.manual_seed(SEED)

log.info(
    "Part types: %s  |  ref_number=%d  query_number=%d  |  ablation methods=%s  |  "
    "multi_scale_combine=%s",
    PART_TYPES,
    REF_NUMBER,
    QUERY_NUMBER,
    METHOD_DISPLAY_ORDER,
    MULTI_SCALE_COMBINE,
)

# %% Mask / geometry helpers
# load_instance_pixel_masks / load_instance_pixel_mask now live in dinoisawesome.abc3 (shared
# with augmented_prototype_oracle_iou.py's batch mode) — they take an already-composed
# annotation_stem, unlike this file's own bare `stem` convention.


def gt_instance_patch_sizes(
    stem: str,
    class_filter: list[str] | None,
    grid_h: int,
    grid_w: int,
    img_size: int,
    patch_threshold: float,
) -> np.ndarray:
    """Per-instance GT sizes in patches, using that image's own annotation set."""
    return _shared_gt_instance_patch_sizes(
        stem, class_filter, grid_h, grid_w, img_size, patch_threshold, data_dir
    )


# %% Multi-scale exemplar prototype builder


@dataclass
class ClusterCrop:
    """One instance's own crop at one scale (mid/close), pre-averaging."""

    cluster_idx: int
    box: tuple[int, int, int, int]  # (x0, y0, x1, y1) in the reference image
    crop_img: Image.Image
    tokens: torch.Tensor  # (H*W, C) L2-normalised, from this cluster's own crop
    grid_h: int
    grid_w: int
    patch_mask: np.ndarray  # (grid_h, grid_w) bool — this instance's mask within the crop
    # (grid_h, grid_w) bool — union of every instance's mask within this crop (this instance's
    # own patch_mask plus any neighbouring instance that happens to fall inside the same crop
    # box). Used to keep bg_prototype's exclusion in step with fg: bg is "not foreground of any
    # instance", not just "not this instance", so a neighbour's fg patches never leak into bg.
    exclude_patch_mask: np.ndarray
    prototype: torch.Tensor  # (1, C) L2-normalised masked-mean descriptor (foreground)
    # (1, C) L2-normalised masked-mean descriptor over patches outside exclude_patch_mask —
    # i.e. excluding this instance's own mask *and* any other instance's mask in the same crop.
    bg_prototype: torch.Tensor


@dataclass
class ScalePrototype:
    scale: str
    box: tuple[int, int, int, int]  # representative crop's box (largest cluster for mid/close)
    crop_img: Image.Image  # representative crop (global's only crop, or largest cluster's)
    tokens: torch.Tensor  # (H*W, C) L2-normalised, from the representative crop
    grid_h: int
    grid_w: int
    patch_mask: np.ndarray  # (grid_h, grid_w) bool — object mask within the representative crop
    prototype: torch.Tensor  # (1, C) L2-normalised — mean of all cluster prototypes for mid/close
    bg_prototype: torch.Tensor  # (1, C) L2-normalised — same mean, over each cluster's background
    cluster_crops: list[ClusterCrop] | None = None  # per-instance crops; None for "global"
    # (width_frac, height_frac): median training-time crop size for this scale, as a fraction of
    # the reference image's own width/height. Only populated for "mid" (see
    # build_all_scale_prototypes) — used at inference to pull stage-2 ROI blob crops toward the
    # same field of view the mid prototype was actually built from, instead of the scale-agnostic
    # fixed floor every scale used before (see find_roi_blobs). None for "close"/"global".
    target_size_frac: tuple[float, float] | None = None


def build_all_scale_prototypes(
    encoder: DinoEncoder, ref_img: Image.Image, instance_masks: list[np.ndarray]
) -> tuple[dict[str, ScalePrototype], torch.Tensor]:
    """Build one exemplar prototype per scale, encoding all kept crops in one batch.

    "global" is a single whole-image crop, masked-mean over the union of every instance
    in *instance_masks* (unaffected by how many instances there are). "mid" and "close"
    are built per instance ("cluster"): each instance gets its own crop (see
    ``scale_crop_box``, applied to that instance's own mask) and its own masked-mean
    prototype; the scale's final ``prototype`` is the mean of all per-instance
    prototypes, re-normalised. This avoids one crop spanning every instance of a
    multi-instance class (e.g. several "white clips" scattered across the image), which
    would mostly capture background between them rather than a close-up of any one part.

    A cluster's "close" crop is dropped when it's below ``MIN_CROP_SIZE`` on either side
    — a tight bbox around a small instance can shrink to a sliver the encoder would
    otherwise just upsample to ``img_size``. "mid" crops are never dropped. If every
    cluster's "close" crop is dropped, the "close" scale itself is dropped for this
    reference image. Each ``ScalePrototype``'s single-crop fields (box/crop_img/tokens/
    patch_mask) point at the *largest* surviving cluster for that scale, as a
    representative for single-crop visualisations/deep-dives elsewhere in this file.

    Also returns ``mean_patch_prototype`` (1, C): the L2-renormalised mean of every
    individual instance cluster's own masked-mean descriptor, pooled across "mid" and
    "close" — i.e. the mean of the mean patches of every cluster. This is a single
    embedding-space prototype, independent of scale/method, used at detection time to
    sanity-check a *predicted* cluster's own mean patch against what a real instance
    looks like on average (see ``cluster_mean_patch_score`` /
    ``tune_cluster_reject_threshold``), on top of the existing per-patch score map.
    """
    union_mask = np.stack(instance_masks).any(axis=0)
    H, W = union_mask.shape
    global_box = (0, 0, W, H)

    pending: list[dict] = [
        {
            "scale": "global",
            "cluster_idx": -1,
            "box": global_box,
            "crop_img": ref_img.crop(global_box),
            "mask_crop": union_mask,
            # global's own mask already *is* the union, so exclusion == mask here.
            "union_mask_crop": union_mask,
        }
    ]
    for cluster_idx, inst_mask in enumerate(instance_masks):
        for scale in ("mid", "close"):
            box = scale_crop_box(inst_mask, scale, EXEMPLAR_CLOSE_PADDING_FRACTION)
            x0, y0, x1, y1 = box
            if scale == "close" and (x1 - x0 < MIN_CROP_SIZE or y1 - y0 < MIN_CROP_SIZE):
                log.warning(
                    "scale=close cluster=%d: crop %dx%dpx below MIN_CROP_SIZE=%dpx — "
                    "dropping this cluster",
                    cluster_idx,
                    x1 - x0,
                    y1 - y0,
                    MIN_CROP_SIZE,
                )
                continue
            pending.append(
                {
                    "scale": scale,
                    "cluster_idx": cluster_idx,
                    "box": box,
                    "crop_img": ref_img.crop(box),
                    "mask_crop": inst_mask[y0:y1, x0:x1],
                    # Union of *every* instance within this cluster's crop box, not just this
                    # instance's own mask — a neighbouring instance can fall inside the same
                    # box (close crops pad by EXEMPLAR_CLOSE_PADDING_FRACTION of their own
                    # extent; mid crops span halfway to the full image). bg must exclude all
                    # of it, or a neighbour's foreground patches get averaged into this
                    # cluster's background prototype.
                    "union_mask_crop": union_mask[y0:y1, x0:x1],
                }
            )

    tokens_batch = extract_patch_tokens_batch(
        encoder, [p["crop_img"] for p in pending], LAYER_IDX, debias=DEBIAS
    )

    global_crop: ClusterCrop | None = None
    clusters_by_scale: dict[str, list[ClusterCrop]] = {"mid": [], "close": []}
    for entry, (tokens, grid_h, grid_w) in zip(pending, tokens_batch):
        patch_mask = pixel_mask_to_patch_mask(
            entry["mask_crop"], grid_h, grid_w, IMG_SIZE, MASK_PATCH_THRESHOLD
        )
        patch_flat = torch.from_numpy(patch_mask.reshape(-1)).to(tokens.device)

        exclude_patch_mask = pixel_mask_to_patch_mask(
            entry["union_mask_crop"], grid_h, grid_w, IMG_SIZE, MASK_PATCH_THRESHOLD
        )
        exclude_flat = torch.from_numpy(exclude_patch_mask.reshape(-1)).to(tokens.device)

        masked = tokens[patch_flat]
        if masked.shape[0] == 0:
            log.warning(
                "scale=%s cluster=%s: mask empty after projection — using all crop patches",
                entry["scale"],
                entry["cluster_idx"],
            )
            masked = tokens
        prototype = compute_exemplar_features(masked, mode="mean")  # (1, C)

        bg_masked = tokens[~exclude_flat]
        if bg_masked.shape[0] == 0:
            log.warning(
                "scale=%s cluster=%s: background empty after projection — using all crop patches",
                entry["scale"],
                entry["cluster_idx"],
            )
            bg_masked = tokens
        bg_prototype = compute_exemplar_features(bg_masked, mode="mean")  # (1, C)

        cc = ClusterCrop(
            entry["cluster_idx"],
            entry["box"],
            entry["crop_img"],
            tokens,
            grid_h,
            grid_w,
            patch_mask,
            exclude_patch_mask,
            prototype,
            bg_prototype,
        )
        if entry["scale"] == "global":
            global_crop = cc
        else:
            clusters_by_scale[entry["scale"]].append(cc)

    protos: dict[str, ScalePrototype] = {}
    assert global_crop is not None
    protos["global"] = ScalePrototype(
        "global",
        global_crop.box,
        global_crop.crop_img,
        global_crop.tokens,
        global_crop.grid_h,
        global_crop.grid_w,
        global_crop.patch_mask,
        global_crop.prototype,
        global_crop.bg_prototype,
        cluster_crops=None,
    )

    for scale in ("mid", "close"):
        clusters = clusters_by_scale[scale]
        if not clusters:
            log.warning(
                "scale=%s: every cluster crop was below MIN_CROP_SIZE — dropping this scale",
                scale,
            )
            continue
        avg = F.normalize(
            torch.cat([c.prototype for c in clusters], dim=0).mean(dim=0, keepdim=True),
            p=2,
            dim=-1,
        )
        bg_avg = F.normalize(
            torch.cat([c.bg_prototype for c in clusters], dim=0).mean(dim=0, keepdim=True),
            p=2,
            dim=-1,
        )
        rep = max(clusters, key=lambda c: int(c.patch_mask.sum()))

        target_size_frac = None
        if scale == "mid":
            # Median, not mean — robust to one outlier-sized instance skewing the target that
            # every query blob then gets pulled toward.
            widths = np.array([c.box[2] - c.box[0] for c in clusters], dtype=float)
            heights = np.array([c.box[3] - c.box[1] for c in clusters], dtype=float)
            target_size_frac = (float(np.median(widths)) / W, float(np.median(heights)) / H)

        protos[scale] = ScalePrototype(
            scale,
            rep.box,
            rep.crop_img,
            rep.tokens,
            rep.grid_h,
            rep.grid_w,
            rep.patch_mask,
            avg,
            bg_avg,
            cluster_crops=clusters,
            target_size_frac=target_size_frac,
        )

    all_instance_protos = clusters_by_scale["mid"] + clusters_by_scale["close"]
    assert all_instance_protos, "mid clusters are never dropped, so this can't be empty"
    mean_patch_prototype = F.normalize(
        torch.cat([c.prototype for c in all_instance_protos], dim=0).mean(dim=0, keepdim=True),
        p=2,
        dim=-1,
    )
    return protos, mean_patch_prototype


# %% Ablation method states — single-scale prototypes and their combinations


@dataclass
class MethodState:
    name: str
    kind: Literal["single", "multi", "fgbg", "knn_fgbg", "kmeans_fgbg"]
    # (K, C) L2-normalised prototype(s); "fgbg" is (2, C) = [fg, bg]; None for "knn_fgbg"/
    # "kmeans_fgbg" (both use fg_bank/bg_bank instead — see build_knn_fgbg_states/
    # build_kmeans_fgbg_states / knn_fgbg_score / kmeans_fgbg_score).
    payload: torch.Tensor | None = None
    # "knn_fgbg": (Nfg, C) raw patch gallery. "kmeans_fgbg": (k, C) k-means centroids.
    fg_bank: torch.Tensor | None = None
    # "knn_fgbg": (Nbg, C) raw patch gallery. "kmeans_fgbg": (1, C) collapsed mean bg vector
    # (same bg side as "fgbg"/fg-bg-mean) — only the fg side is k-means for this kind.
    bg_bank: torch.Tensor | None = None


def build_ablation_states(
    scale_protos: dict[str, ScalePrototype],
    combos: dict[str, list[str]],
    combine_modes: list[Literal["max", "mean"]],
) -> dict[str, MethodState]:
    """Build states for every combo whose scales are all present in *scale_protos*.

    Combos referencing a dropped scale (e.g. "close" below MIN_CROP_SIZE) are
    skipped rather than raising, so a too-small close-up doesn't fail the whole run.

    A single-scale combo produces exactly one state, named after the combo. A multi-scale combo
    produces one state per mode in *combine_modes*, named "<combo>-<mode>": "max" keeps every
    scale's prototype as its own row (kind="multi") for score_method's per-patch max similarity;
    "mean" renormalises them into a single vector here (kind="single"), so scoring is one cosine
    similarity per patch like any single-scale method. Passing both modes builds both variants
    (e.g. "mid+close-max" and "mid+close-mean") so they can be compared in the same ablation run.
    See ablation_method_names, which mirrors this naming for METHOD_DISPLAY_ORDER.
    """
    states = {}
    for combo_name, members in combos.items():
        if not all(m in scale_protos for m in members):
            continue
        protos = torch.cat([scale_protos[m].prototype for m in members], dim=0)  # (K, C)
        if len(members) == 1:
            states[combo_name] = MethodState(combo_name, "single", protos)
            continue
        for mode in combine_modes:
            name = f"{combo_name}-{mode}"
            if mode == "mean":
                mean_proto = F.normalize(protos.mean(dim=0, keepdim=True), p=2, dim=-1)  # (1, C)
                states[name] = MethodState(name, "single", mean_proto)
            else:
                states[name] = MethodState(name, "multi", protos)
    return states


def build_fgbg_states(
    scale_protos: dict[str, ScalePrototype], combos: dict[str, dict[str, list[str]]]
) -> dict[str, MethodState]:
    """Mean-collapsed contrastive fg-bg: score = cos(query, fg) - cos(query, bg).

    One MethodState per *combos* entry with a single fg scale, named
    "fg-bg-mean(<combo_name>)" — the "-mean" makes explicit that both sides are collapsed to a
    single mean vector, unlike the "-max"/"-mean" combine modes on multi-scale single-side
    combos (see MULTI_SCALE_COMBINE), and unlike fg-bg-knn which never collapses. fg is the
    L2-renormalised (here, trivially, since there's only one scale) foreground (masked-mean)
    prototype of the named "fg" scale, and bg is the L2-renormalised average of the named "bg"
    scales' own background prototypes (the masked mean of the *non*-mask patches within that
    scale's own crop(s)). A scale in "bg" that only appears there (e.g. "global") contributes
    far-field background from the rest of the reference image — without it, bg only ever sees a
    narrow local neighborhood and fails to discriminate once the query has background content
    outside that neighborhood.

    Combos with more than one fg scale are built by build_fgbg_multiproto_states instead (kept
    as separate prototype rows, not averaged here) — see that function. A combo is skipped (not
    included in the returned dict) if any of its fg/bg scales was dropped for this reference
    image (e.g. "close" below MIN_CROP_SIZE) — build_ablation_states skips combos the same way.
    """
    states = {}
    for combo_name, sources in combos.items():
        fg_scales, bg_scales = sources["fg"], sources["bg"]
        if len(fg_scales) != 1 or not all(s in scale_protos for s in [*fg_scales, *bg_scales]):
            continue
        fg = F.normalize(
            torch.cat([scale_protos[s].prototype for s in fg_scales], dim=0).mean(
                dim=0, keepdim=True
            ),
            p=2,
            dim=-1,
        )
        bg = F.normalize(
            torch.cat([scale_protos[s].bg_prototype for s in bg_scales], dim=0).mean(
                dim=0, keepdim=True
            ),
            p=2,
            dim=-1,
        )
        payload = torch.cat([fg, bg], dim=0)  # (2, C): row 0 = fg, row 1 = bg
        name = f"fg-bg-mean({combo_name})"
        states[name] = MethodState(name, "fgbg", payload)
    return states


def build_fgbg_multiproto_states(
    scale_protos: dict[str, ScalePrototype], combos: dict[str, dict[str, list[str]]]
) -> dict[str, MethodState]:
    """Multi-datasource contrastive fg-bg: one prototype row per fg scale, kept separate
    instead of averaged into one vector like build_fgbg_states.

    Reuses the "kmeans_fgbg" MethodState shape/scoring (fg_bank of (K, C) rows scored via
    per-patch max cosine similarity, minus a single collapsed bg vector — see
    kmeans_fgbg_score) — the mechanism is identical to build_kmeans_fgbg_states' k-means
    centroids, just fed each fg scale's own mean prototype instead of learned centroids, so a
    query patch can match whichever scale's prototype fits best rather than an average of them.

    One MethodState per *combos* entry with more than one fg scale, named
    "fg-bg-proto(<combo_name>)" — a single fg scale has nothing to keep separate, so it's just
    build_fgbg_states' "fg-bg-mean(...)". Combos are skipped the same way build_fgbg_states
    skips them (a dropped fg/bg scale for this reference image).
    """
    states: dict[str, MethodState] = {}
    for combo_name, sources in combos.items():
        fg_scales, bg_scales = sources["fg"], sources["bg"]
        if len(fg_scales) < 2 or not all(s in scale_protos for s in [*fg_scales, *bg_scales]):
            continue
        fg_protos = torch.cat([scale_protos[s].prototype for s in fg_scales], dim=0)  # (K, C)
        bg = F.normalize(
            torch.cat([scale_protos[s].bg_prototype for s in bg_scales], dim=0).mean(
                dim=0, keepdim=True
            ),
            p=2,
            dim=-1,
        )
        name = f"fg-bg-proto({combo_name})"
        states[name] = MethodState(name, "kmeans_fgbg", fg_bank=fg_protos, bg_bank=bg)
    return states


def _pool_scale_patches(
    scale_protos: dict[str, ScalePrototype], scales: list[str], want_fg: bool
) -> torch.Tensor:
    """Pool masked (want_fg=True) or unmasked (False) patch tokens across *scales*.

    Pools across every cluster crop within each scale, matching build_ablation_states'
    per-scale prototype pooling. For want_fg=False, a ClusterCrop pools patches outside its
    ``exclude_patch_mask`` (union of every instance's mask within that crop), not just outside
    its own ``patch_mask`` — otherwise a neighbouring instance that falls inside the same crop
    box would contribute its foreground patches to the bg gallery. The plain "global" cell (a
    ScalePrototype, not a ClusterCrop) has no neighbours to exclude beyond its own mask, which
    already *is* the union of every instance.
    """
    chunks: list[torch.Tensor] = []
    for s in scales:
        proto = scale_protos[s]
        cells = proto.cluster_crops if proto.cluster_crops is not None else [proto]
        for cell in cells:
            if want_fg:
                sel_mask = cell.patch_mask
            else:
                sel_mask = (
                    cell.exclude_patch_mask if isinstance(cell, ClusterCrop) else cell.patch_mask
                )
            sel_flat = torch.from_numpy(sel_mask.reshape(-1)).to(cell.tokens.device)
            chunks.append(cell.tokens[sel_flat] if want_fg else cell.tokens[~sel_flat])
    return torch.cat(chunks, dim=0)


def build_kmeans_states(
    scale_protos: dict[str, ScalePrototype], scales: list[str], ks: tuple[int, ...]
) -> dict[str, MethodState]:
    """One k-means(k) method per (scale, k): same pooled foreground patches as that
    scale's single-scale mean method (see ``_pool_scale_patches``), but *k* centroids
    instead of one mean vector. Scoring reuses the "multi" kind's per-patch max-cosine
    (see ``score_method``), same as a "max"-combined multi-scale combo.

    A scale dropped for this reference image (e.g. "close" below MIN_CROP_SIZE) is
    skipped, same as ``build_ablation_states``. *k* is clamped to the pooled patch
    count so a scale with fewer foreground patches than *k* doesn't error out of
    ``compute_exemplar_features``'s k-means.
    """
    states: dict[str, MethodState] = {}
    for scale in scales:
        if scale not in scale_protos:
            continue
        fg_patches = _pool_scale_patches(scale_protos, [scale], want_fg=True)
        for k in ks:
            kk = min(k, fg_patches.shape[0])
            centroids = compute_exemplar_features(fg_patches, mode="kmeans", k=kk)  # (kk, C)
            name = f"{scale}-kmeans{k}"
            states[name] = MethodState(name, "multi", centroids)
    return states


def build_knn_fgbg_states(
    scale_protos: dict[str, ScalePrototype], combos: dict[str, dict[str, list[str]]]
) -> dict[str, MethodState]:
    """kNN-gallery variant of build_fgbg_states: contrastive fg-bg without collapsing to a mean.

    build_fgbg_states (the "fg-bg-mean" methods) reduces each scale's masked patches down to a
    single masked-mean vector per side (fg, bg) before ever comparing to the query. This method
    instead keeps every individual patch token as its own gallery entry: for each *combos* entry,
    the named "fg" scales' foreground-masked patches are pooled into one fg gallery and the named
    "bg" scales' background-masked patches into one bg gallery, across every cluster crop. Scoring
    (see knn_fgbg_score) then does a per-query-patch kNN lookup against each gallery rather than one
    cosine similarity to a mean — closer in spirit to dinoisawesome.anomaly_head.AnomalyHead's
    memory-bank matching than to a prototype method, but contrastive (fg top-k minus bg top-k)
    instead of a pure nearest-neighbour distance.

    One MethodState per *combos* entry, named "fg-bg-knn(<combo_name>)". A combo is skipped if
    any of its fg/bg scales was dropped for this reference image, exactly like
    build_fgbg_states.
    """
    states = {}
    for combo_name, sources in combos.items():
        fg_scales, bg_scales = sources["fg"], sources["bg"]
        if not all(s in scale_protos for s in [*fg_scales, *bg_scales]):
            continue
        fg_bank = _pool_scale_patches(scale_protos, fg_scales, want_fg=True)
        bg_bank = _pool_scale_patches(scale_protos, bg_scales, want_fg=False)
        name = f"fg-bg-knn({combo_name})"
        log.info(
            "%s: gallery sizes fg=%d bg=%d patches (k=%d)",
            name,
            fg_bank.shape[0],
            bg_bank.shape[0],
            KNN_FGBG_NUM_NEIGHBOURS,
        )
        states[name] = MethodState(name, "knn_fgbg", fg_bank=fg_bank, bg_bank=bg_bank)
    return states


def build_kmeans_fgbg_states(
    scale_protos: dict[str, ScalePrototype],
    combos: dict[str, dict[str, list[str]]],
    ks: tuple[int, ...],
) -> dict[str, MethodState]:
    """Background-contrastive variant of build_kmeans_states.

    build_kmeans_states' "<scale>-kmeans<k>" methods score a query patch as the max cosine
    similarity to any of k foreground centroids, with nothing pulling background patches back
    down — unlike every other multi-vector method here (fg-bg-mean, fg-bg-knn), which
    contrasts against a background side. This builds that missing bg term: the fg side is
    still k learned centroids (see compute_exemplar_features's kmeans mode, pooled the same
    way build_knn_fgbg_states pools its fg gallery), scored with a per-query-patch max
    (kmeans_fgbg_score) — a query patch only needs to match ONE learned appearance mode, not
    average well across all of them, so this deliberately isn't knn_fgbg_score's mean-of-top-k.
    The bg side stays a single collapsed mean, identical to fg-bg-mean's bg — the point is
    only to give kmeans' foreground gallery a contrast term, not to redesign the bg side too.

    One MethodState per (combo, k) in *combos* x *ks*, named "fg-bg-kmeans<k>(<combo_name>)".
    A combo is skipped for this reference image if any of its fg/bg scales was dropped,
    exactly like build_fgbg_states/build_knn_fgbg_states. *k* is clamped to the pooled fg
    patch count, same as build_kmeans_states.
    """
    states: dict[str, MethodState] = {}
    for combo_name, sources in combos.items():
        fg_scales, bg_scales = sources["fg"], sources["bg"]
        if not all(s in scale_protos for s in [*fg_scales, *bg_scales]):
            continue
        fg_patches = _pool_scale_patches(scale_protos, fg_scales, want_fg=True)
        bg = F.normalize(
            torch.cat([scale_protos[s].bg_prototype for s in bg_scales], dim=0).mean(
                dim=0, keepdim=True
            ),
            p=2,
            dim=-1,
        )
        for k in ks:
            kk = min(k, fg_patches.shape[0])
            centroids = compute_exemplar_features(fg_patches, mode="kmeans", k=kk)  # (kk, C)
            name = f"fg-bg-kmeans{k}({combo_name})"
            states[name] = MethodState(name, "kmeans_fgbg", fg_bank=centroids, bg_bank=bg)
    return states


def kmeans_fgbg_score(
    query_tokens: torch.Tensor, fg_centroids: torch.Tensor, bg_vec: torch.Tensor
) -> np.ndarray:
    """Per-query-patch contrastive score for "kmeans_fgbg": max cosine similarity to any of
    the k fg centroids, minus cosine similarity to the single collapsed bg vector.

    Deliberately a max on the fg side (like build_kmeans_states' plain kmeans methods),
    not knn_fgbg_score's mean-of-top-k — a query patch should count as foreground if it
    matches ANY one learned appearance mode strongly, not if it matches all k on average.
    """
    fg_sim = query_tokens @ fg_centroids.T  # (N, k)
    fg_max = fg_sim.max(dim=1).values
    bg_sim = (query_tokens @ bg_vec.T).squeeze(-1)  # (N,)
    return (fg_max - bg_sim).cpu().float().numpy()


def score_method(state: MethodState, query_tokens: torch.Tensor) -> np.ndarray:
    """Similarity score per query patch.

    "single"/"multi": cosine similarity to the state's prototype(s), max'd over multiple.
    "fgbg": fg cosine similarity minus bg cosine similarity (contrastive, mean-based).
    "knn_fgbg": same contrastive idea, but each side is a patch gallery scored via kNN
    (see knn_fgbg_score) instead of a single mean vector.
    "kmeans_fgbg": fg side is k learned centroids scored via per-patch max, bg side is a
    single collapsed mean (see kmeans_fgbg_score).
    """
    if state.kind == "knn_fgbg":
        assert state.fg_bank is not None and state.bg_bank is not None
        return knn_fgbg_score(query_tokens, state.fg_bank, state.bg_bank, KNN_FGBG_NUM_NEIGHBOURS)
    if state.kind == "kmeans_fgbg":
        assert state.fg_bank is not None and state.bg_bank is not None
        return kmeans_fgbg_score(query_tokens, state.fg_bank, state.bg_bank)
    assert state.payload is not None
    sim = query_tokens @ state.payload.T  # (N, K)
    if state.kind == "single":
        return sim.squeeze(-1).cpu().float().numpy()
    if state.kind == "fgbg":
        return (sim[:, 0] - sim[:, 1]).cpu().float().numpy()
    return sim.max(dim=-1).values.cpu().float().numpy()


# %% Threshold + clustering helpers (ref_iou threshold, DBSCAN prediction clusters,
# DBSCAN GT clusters, greedy IoU matching). density_map_methods.py still uses HDBSCAN for
# prediction clusters; this file defaults to DBSCAN throughout instead (see dbscan_clusters).


def min_cluster_size_bound(sizes: np.ndarray, margin: int | float) -> int:
    """Minimum cluster size from the smallest GT instance size, margin-relaxed downward.

    ``margin`` as an int is a flat patch-count offset: k_min - margin. As a float it's
    read as a fraction of the bound: k_min - margin*k_min.
    """
    return _shared_min_cluster_size_bound(sizes, margin, MIN_POINTS_FLOOR)


def dbscan_clusters(
    xs: np.ndarray,
    ys: np.ndarray,
    grid_h: int,
    grid_w: int,
    raw: np.ndarray,
    min_cs: int | None = None,
    eps: float = PRED_DBSCAN_EPS,
    min_samples: int = PRED_DBSCAN_MIN_SAMPLES,
) -> list[dict]:
    """Cluster foreground patch coords with plain DBSCAN (patch-index space).

    Thin wrapper around ``dbscan_clusters_from_mask`` that keeps the old
    ``hdbscan_clusters`` call signature (xs/ys/grid_h/grid_w/min_cs) so call sites
    didn't need to change shape when this file dropped HDBSCAN in favour of DBSCAN
    everywhere. ``min_cs`` is applied as a post-hoc size filter on the resulting
    clusters — DBSCAN has no notion of a cluster-size bound the way HDBSCAN's
    ``min_cluster_size`` did (that split/merged *during* clustering itself), so an
    undersized DBSCAN cluster here is dropped rather than merged into a neighbour.
    """
    return _shared_dbscan_clusters(xs, ys, grid_h, grid_w, raw, eps, min_samples, min_cs)


# %% Cluster mean-patch reject — a cluster-level sanity check on top of the per-patch score
# map + DBSCAN pipeline above. A predicted cluster can pass the per-patch threshold and still
# not look like a real instance on average (e.g. it straddles an edge, or sits on background
# that happens to score just above threshold); this compares each predicted cluster's own mean
# patch token against ``mean_patch_prototype`` (built once, at training time, from the mean of
# every ref instance cluster's own mean patch — see ``build_all_scale_prototypes``) and rejects
# clusters that fall below a tuned cosine-similarity cutoff.


def tune_cluster_reject_threshold(
    ref_crops: list[ClusterCrop],
    state: MethodState,
    mean_patch_prototype: torch.Tensor,
    patch_thr: float,
    min_cs: int,
    iou_thr: float,
) -> tuple[float, list[dict]]:
    """Tune the mean-patch cosine-similarity cutoff that separates predicted clusters that
    genuinely match a GT instance ("good") from spurious ones ("bad") — fit on the reference's
    own predicted clusters, found the same way query clusters are (score each ``ref_crops`` crop
    with *state*, threshold at *patch_thr*, then DBSCAN with the same ``min_cs``), reused as-is on
    query clusters. Self-supervised, mirrors ``iou_tuned_threshold``'s fit-on-ref / apply-to-query
    pattern.

    Pools across *every* mid-scale instance crop in ``ref_crops`` (``ScalePrototype.
    cluster_crops``), not just the single representative/biggest one — mirrors the "pooled
    accumulation across mid crops" idea used for the patch-level threshold.

    Returns ``(threshold, ref_clusters)`` — each ref cluster dict gains ``mean_patch_score``,
    ``gt_iou`` (best IoU against any single GT instance in its own source crop), ``gt_good``
    (``gt_iou >= iou_thr``), and ``crop_idx``/``crop_img``/``crop_gt_mask`` identifying which
    ``ref_crops`` entry it came from. If no ref cluster is "good" (nothing to anchor a margin
    to), falls back to a cutoff just below the global min score — permissive rather than
    rejecting everything. Otherwise the cutoff is ``CLUSTER_REJECT_MARGIN_FRACTION`` below the
    smallest good score, clamped so it never drops to or below the largest bad score.
    """
    return _shared_tune_cluster_reject_threshold(
        ref_crops,
        lambda tokens: score_method(state, tokens),
        mean_patch_prototype,
        patch_thr,
        min_cs,
        iou_thr,
        PRED_DBSCAN_EPS,
        PRED_DBSCAN_MIN_SAMPLES,
        GT_DBSCAN_EPS,
        GT_DBSCAN_MIN_SAMPLES,
        CLUSTER_REJECT_MARGIN_FRACTION,
        MIN_POINTS_FLOOR,
        include_crop_viz_fields=True,
    )


# %% ROI blob helpers — unsupervised (no-GT) crop discovery, reused for the
# cross-scale similarity experiment below.


def roi_binary_mask(
    raw: np.ndarray, method: str, single_threshold: float, percentile: float = ROI_PERCENTILE
) -> np.ndarray:
    """Unsupervised (no-GT) foreground/ROI mask used to decide where to zoom in."""
    return _shared_roi_binary_mask(raw, method, single_threshold, percentile=percentile)


def find_roi_blobs(
    raw: np.ndarray,
    query_img: Image.Image,
    encoder: DinoEncoder,
    crop_mode: str = ROI_CROP_MODE,
    target_size_frac: tuple[float, float] | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Unsupervised ROI blobs from *raw* (binarised, morphologically closed, then
    connected components), each cropped at native resolution and re-encoded — the
    stage-2 crop mechanism from density_map_methods.py, factored out so any scale's
    raw map can supply the ROIs.

    ``crop_mode="per_blob"`` (default) crops each connected ROI component separately.
    ``crop_mode="single_bbox"`` merges every ROI pixel into one blob, so exactly one
    crop (bounding all segmented parts) is produced instead of one per part.

    ``target_size_frac`` (width_frac, height_frac), when given (currently only for "mid" —
    see ``ScalePrototype.target_size_frac``), is the median crop size that scale's
    prototype was actually built from at reference time, as a fraction of the reference
    image's own width/height. Projected onto *this* query image's resolution, it becomes a
    per-axis floor (on top of the existing ``MAX_UPSCALE_FACTOR`` floor) that
    ``pad_and_floor_crop_box`` pulls undersized blob crops up toward — so a "mid" query
    crop reproduces the field of view the mid prototype was trained on, rather than
    whatever the fixed ``MAX_UPSCALE_FACTOR`` floor and a small blob happen to produce.
    Like that floor, it only ever expands a crop — a blob already bigger than the target
    (e.g. several nearby instances merged into one blob) is left as-is. None (the default)
    keeps the old scale-agnostic behaviour.

    Returns ``(crops, roi_mask)``; each crop dict also carries ``mask`` (the blob's own
    connected-component mask, in the full query patch grid — used to build two-stage
    "predictions" for P/R/F1 matching against ``gt_clusters``, see
    ``two_stage_predicted_clusters``), ``patch_bbox`` and ``raw_px_bbox`` (the un-padded
    blob bbox, for the "step" visualisations). Scoring only needs ``px_bbox`` / ``crop_tokens``.
    """
    native_w, native_h = query_img.size
    scale_x = native_w / IMG_SIZE
    scale_y = native_h / IMG_SIZE

    roi_mask = roi_binary_mask(raw, ROI_BINARIZE_METHOD, ROI_SINGLE_THRESHOLD)
    if ROI_MORPH_CLOSE_ITERATIONS > 0:
        roi_mask = ndimage.binary_closing(
            roi_mask,
            structure=ndimage.generate_binary_structure(2, 2),
            iterations=ROI_MORPH_CLOSE_ITERATIONS,
        )
    if crop_mode == "per_blob":
        blobs = connected_component_blobs(roi_mask)
    elif crop_mode == "single_bbox":
        blobs = [{"mask": roi_mask}] if roi_mask.any() else []
    else:
        raise ValueError(f"Unknown ROI_CROP_MODE: {crop_mode!r}")

    base_floor_px = IMG_SIZE / MAX_UPSCALE_FACTOR
    if target_size_frac is not None:
        floor_w_px = max(base_floor_px, target_size_frac[0] * native_w)
        floor_h_px = max(base_floor_px, target_size_frac[1] * native_h)
    else:
        floor_w_px = floor_h_px = base_floor_px

    pending: list[dict] = []
    for blob in blobs:
        y0, y1, x0, x1 = blob_patch_bbox(blob["mask"])
        raw_px0, raw_py0, raw_px1, raw_py1 = patch_bbox_to_native_px(
            y0, y1, x0, x1, PATCH_SIZE, scale_x, scale_y
        )
        px0, py0, px1, py1 = pad_and_floor_crop_box(
            raw_px0,
            raw_py0,
            raw_px1,
            raw_py1,
            BLOB_CROP_PADDING_FRACTION,
            floor_w_px,
            floor_h_px,
            native_w,
            native_h,
        )
        if px1 - px0 < 2 or py1 - py0 < 2:
            continue

        pending.append(
            {
                "mask": blob["mask"],
                "patch_bbox": (y0, y1, x0, x1),
                "raw_px_bbox": (raw_px0, raw_py0, raw_px1, raw_py1),
                "px_bbox": (px0, py0, px1, py1),
                "crop_img": query_img.crop((px0, py0, px1, py1)),
            }
        )

    if not pending:
        return [], roi_mask

    tokens_batch = extract_patch_tokens_batch(
        encoder, [p["crop_img"] for p in pending], LAYER_IDX, debias=DEBIAS
    )
    crops: list[dict] = [
        {**entry, "crop_tokens": crop_tokens, "c_h": c_h, "c_w": c_w}
        for entry, (crop_tokens, c_h, c_w) in zip(pending, tokens_batch)
    ]
    return crops, roi_mask


def blob_crop_gt_mask(blob: dict, q_pixel_mask: np.ndarray | None) -> np.ndarray:
    """Patch-grid GT mask for one ROI blob crop, aligned to its own re-encoded grid."""
    return _shared_blob_crop_gt_mask(blob, q_pixel_mask, IMG_SIZE, MASK_PATCH_THRESHOLD)


def crop_patch_centers_to_px(
    xs: np.ndarray, ys: np.ndarray, crop_img: Image.Image, c_h: int, c_w: int
) -> tuple[np.ndarray, np.ndarray]:
    """Crop-grid patch centers -> pixel coords within *crop_img* itself (for overlay)."""
    cw, ch = crop_img.size
    return (xs + 0.5) * cw / c_w, (ys + 0.5) * ch / c_h


def cross_score_blobs(
    blobs: list[dict],
    ablation_states: dict[str, MethodState],
    q_pixel_mask: np.ndarray | None,
    per_method: dict[str, dict],
) -> list[dict]:
    """Score every blob crop with every single-scale prototype.

    Binarises each crop's score map with *that prototype's own* tuned threshold
    (the one fit during the ablation pass, against the exemplar's global-scale GT
    mask) and measures IoU against the crop's own ground-truth patch mask.
    """
    scale_states = {s: ablation_states[s] for s in SCALES}
    return _shared_cross_score_blobs(
        blobs,
        scale_states,
        q_pixel_mask,
        per_method,
        score_fn_for=lambda state: (lambda tokens: score_method(state, tokens)),
        img_size=IMG_SIZE,
        mask_patch_threshold=MASK_PATCH_THRESHOLD,
    )


def two_stage_predicted_clusters(
    blobs: list[dict],
    state: MethodState,
    threshold: float,
    mean_patch_prototype: torch.Tensor,
    cluster_reject_thr: float,
    min_cs: int,
    q_h: int,
    q_w: int,
    scale_x: float,
    scale_y: float,
) -> list[dict]:
    """Predicted clusters for the two-stage pipeline's P/R/F1 — one scale used for both
    stage-1 ROI discovery (which found *blobs*) and stage-2 re-scoring (*state*/*threshold*,
    that same scale's own tuned prototype/threshold): the diagonal of the cross-scale matrix.

    Mirrors the single-step pipeline per crop rather than a single pass/fail per blob: each
    crop's own re-scored patches are thresholded, DBSCAN-clustered (same params as the
    single-step methods — a crop can hold more than one instance, so one blob can yield
    several clusters), and mean-patch-rejected against *cluster_reject_thr* exactly like
    ``annotate_cluster_rejection`` does for single-step. Surviving clusters are projected onto
    the shared (q_h, q_w) query grid (see ``project_crop_mask_to_query_grid``) and merged
    across every blob, keeping the biggest of any that collide (``merge_overlapping_clusters``)
    — collisions happen both within one crop (several clusters snapping onto the same coarse
    patches) and across crops (padded blob boxes can overlap in native pixels). The merged list
    is what ``match_and_score`` matches against ``gt_clusters``, exactly like every other
    ablation method's DBSCAN clusters.
    """
    merged, _diagnostics = _shared_two_stage_predicted_clusters(
        blobs,
        lambda tokens: score_method(state, tokens),
        threshold,
        mean_patch_prototype,
        cluster_reject_thr,
        min_cs,
        q_h,
        q_w,
        PATCH_SIZE,
        scale_x,
        scale_y,
        PRED_DBSCAN_EPS,
        PRED_DBSCAN_MIN_SAMPLES,
        MIN_POINTS_FLOOR,
    )
    return merged


# %% Load encoder (shared across all pairs)
encoder = DinoEncoder(
    version=DINO_VERSION,
    size=DINO_SIZE,
    img_size=IMG_SIZE,
    weights_dir=DINO_WEIGHTS_DIR,
    amp=True,
)
PATCH_SIZE = encoder.patch_size
log.info(
    "DINOv%s-%s | patch_size=%d | grid=%dx%d",
    DINO_VERSION[1],
    DINO_SIZE,
    PATCH_SIZE,
    encoder.grid_h,
    encoder.grid_w,
)

# %% Per-pair pipeline


def run_pair(part_type: str, ref_number: int, query_number: int) -> dict | None:
    ref_stem = f"{part_type}_{ref_number}"
    query_stem = f"{part_type}_{query_number}"

    ref_instance_masks = load_instance_pixel_masks(
        data_dir / "annotations" / ref_stem, EXEMPLAR_CLASS
    )
    if not ref_instance_masks:
        log.warning(
            "[%s] no exemplar instances for classes %s — skipping part type.",
            part_type,
            EXEMPLAR_CLASS,
        )
        return None
    ref_pixel_mask = np.stack(ref_instance_masks).any(axis=0)

    ref_img = Image.open(data_dir / f"{ref_stem}.jpg").convert("RGB")
    query_img = Image.open(data_dir / f"{query_stem}.jpg").convert("RGB")

    scale_protos, mean_patch_prototype = build_all_scale_prototypes(
        encoder, ref_img, ref_instance_masks
    )
    for scale, proto in scale_protos.items():
        n_clusters = len(proto.cluster_crops) if proto.cluster_crops is not None else 1
        log.info(
            "[%s] scale=%s clusters=%d representative_crop=%dx%dpx grid=%dx%d masked_patches=%d/%d",
            part_type,
            scale,
            n_clusters,
            proto.box[2] - proto.box[0],
            proto.box[3] - proto.box[1],
            proto.grid_h,
            proto.grid_w,
            int(proto.patch_mask.sum()),
            proto.patch_mask.size,
        )

    ablation_states = build_ablation_states(scale_protos, ABLATION_COMBOS, MULTI_SCALE_COMBINE)
    ablation_states.update(build_fgbg_states(scale_protos, FGBG_SOURCE_COMBOS))
    ablation_states.update(build_fgbg_multiproto_states(scale_protos, FGBG_SOURCE_COMBOS))
    ablation_states.update(build_knn_fgbg_states(scale_protos, FGBG_SOURCE_COMBOS))
    if INCLUDE_KMEANS_METHODS:
        ablation_states.update(build_kmeans_states(scale_protos, SCALES, KMEANS_KS))
        ablation_states.update(
            build_kmeans_fgbg_states(scale_protos, FGBG_SOURCE_COMBOS, KMEANS_KS)
        )

    q_tokens, q_h, q_w = extract_patch_tokens(encoder, query_img, LAYER_IDX, debias=DEBIAS)

    q_pixel_mask = load_instance_pixel_mask(data_dir / "annotations" / query_stem, EXEMPLAR_CLASS)
    gt_patch_mask = (
        pixel_mask_to_patch_mask(q_pixel_mask, q_h, q_w, IMG_SIZE, MASK_PATCH_THRESHOLD)
        if q_pixel_mask is not None
        else np.zeros((q_h, q_w), dtype=bool)
    )
    gt_sizes = gt_instance_patch_sizes(
        query_stem, EXEMPLAR_CLASS, q_h, q_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )
    min_cs = min_cluster_size_bound(gt_sizes, CLUSTER_SIZE_MARGIN)
    gt_clusters = dbscan_clusters_from_mask(gt_patch_mask, GT_DBSCAN_EPS, GT_DBSCAN_MIN_SAMPLES)
    log.info(
        "[%s] GT instances=%d sizes=%s -> min_cluster_size=%d -> GT-DBSCAN clusters=%d",
        part_type,
        len(gt_sizes),
        gt_sizes.tolist(),
        min_cs,
        len(gt_clusters),
    )

    # Tuning source for every method's threshold: the ref's own mid crop (never dropped,
    # unlike "close"), scored against the union of every GT instance overlapping that crop's
    # box — not just the single representative instance scale_protos["mid"].patch_mask
    # carries (see build_all_scale_prototypes). Using a per-instance mask here would silently
    # ignore other in-frame instances when tuning.
    ref_mid = scale_protos["mid"]
    ref_mid_x0, ref_mid_y0, ref_mid_x1, ref_mid_y1 = ref_mid.box
    ref_mid_gt_mask = pixel_mask_to_patch_mask(
        ref_pixel_mask[ref_mid_y0:ref_mid_y1, ref_mid_x0:ref_mid_x1],
        ref_mid.grid_h,
        ref_mid.grid_w,
        IMG_SIZE,
        MASK_PATCH_THRESHOLD,
    )

    per_method: dict[str, dict] = {}
    for name, state in ablation_states.items():
        raw = score_method(state, q_tokens).reshape(q_h, q_w)

        ref_raw = score_method(state, ref_mid.tokens).reshape(ref_mid.grid_h, ref_mid.grid_w)
        thr = iou_tuned_threshold(ref_raw, ref_mid_gt_mask, REF_THRESHOLD_STEPS)

        assert ref_mid.cluster_crops, "mid scale is never dropped, so this can't be empty"
        cluster_reject_thr, ref_tuning_clusters = tune_cluster_reject_threshold(
            ref_mid.cluster_crops,
            state,
            mean_patch_prototype,
            thr,
            min_cs,
            IOU_MATCH_THRESHOLD,
        )

        binary = raw > thr
        ys, xs = np.where(binary)
        if len(xs) < max(MIN_POINTS_FLOOR, min_cs):
            pred_clusters: list[dict] = []
        else:
            pred_clusters = dbscan_clusters(xs, ys, q_h, q_w, raw, min_cs)
        annotate_cluster_rejection(
            pred_clusters, q_tokens, mean_patch_prototype, cluster_reject_thr
        )
        kept_clusters = [c for c in pred_clusters if not c["rejected"]]

        metrics = match_and_score(kept_clusters, gt_clusters, IOU_MATCH_THRESHOLD)
        log.info(
            "[%s/%s] thr=%.3f cluster_reject_thr=%.3f pred_clusters=%d kept=%d "
            "P=%.2f R=%.2f F1=%.2f mIoU=%.2f",
            part_type,
            name,
            thr,
            cluster_reject_thr,
            len(pred_clusters),
            len(kept_clusters),
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["mean_iou"],
        )
        per_method[name] = {
            "raw": raw,
            "threshold": thr,
            "binary": binary,
            "clusters": pred_clusters,
            "kept_clusters": kept_clusters,
            "cluster_reject_thr": cluster_reject_thr,
            "ref_tuning_clusters": ref_tuning_clusters,
            "metrics": metrics,
        }

    # Stage-2 ROI blobs, one set per single-scale prototype's own raw map — the
    # inputs to the cross-scale similarity experiment below.
    blobs_by_scale: dict[str, list[dict]] = {}
    roi_mask_by_scale: dict[str, np.ndarray] = {}
    for scale in SCALES:
        if scale not in per_method:
            continue  # dropped scale (e.g. "close" below MIN_CROP_SIZE) — nothing to do
        # scale_protos[scale].target_size_frac is None for every scale except "mid" (see
        # build_all_scale_prototypes), so this naturally only pulls "mid" blob crops toward
        # their training-time field of view — other scales keep the old fixed floor.
        blobs, roi_mask = find_roi_blobs(
            per_method[scale]["raw"],
            query_img,
            encoder,
            target_size_frac=scale_protos[scale].target_size_frac,
        )
        blobs_by_scale[scale] = blobs
        roi_mask_by_scale[scale] = roi_mask
        log.info("[%s] roi_source=%s -> %d blob(s)", part_type, scale, len(blobs))

    return {
        "part_type": part_type,
        "ref_img": ref_img,
        "query_img": query_img,
        "ref_pixel_mask": ref_pixel_mask,
        "q_pixel_mask": q_pixel_mask,
        "scale_protos": scale_protos,
        "mean_patch_prototype": mean_patch_prototype,
        "gt_patch_mask": gt_patch_mask,
        "gt_clusters": gt_clusters,
        "min_cs": min_cs,
        "per_method": per_method,
        "blobs_by_scale": blobs_by_scale,
        "roi_mask_by_scale": roi_mask_by_scale,
        "ablation_states": ablation_states,
        "q_h": q_h,
        "q_w": q_w,
    }


# %% Run across all part types, collect ablation + two-stage metrics, and cross-scale similarity
# (score every ROI-source scale's blobs with every prototype scale). The two-stage pipeline's
# metrics are computed here, per ROI-source scale, from the *diagonal* of that same cross-scale
# scoring pass (roi_source == score_scale) — see two_stage_predicted_clusters.
results: dict[str, dict] = {}
metrics_rows: list[dict] = []
cross_rows: list[dict] = []
for pt in RUN_PART_TYPES:
    result = run_pair(pt, REF_NUMBER, QUERY_NUMBER)
    if result is None:
        continue
    results[pt] = result
    for method_name, m in result["per_method"].items():
        metrics_rows.append({"part_type": pt, "method": method_name, **m["metrics"]})

    # Native-px-per-IMG_SIZE-px scale for this pair's query image — shared by every
    # two_stage_predicted_clusters call below to project crop-local clusters back onto this
    # query's own (q_h, q_w) grid (see project_crop_mask_to_query_grid).
    native_w, native_h = result["query_img"].size
    scale_x, scale_y = native_w / IMG_SIZE, native_h / IMG_SIZE

    for roi_source, blobs in result["blobs_by_scale"].items():
        scored = cross_score_blobs(
            blobs, result["ablation_states"], result["q_pixel_mask"], result["per_method"]
        )
        for r in scored:
            cross_rows.append({"part_type": pt, "roi_source": roi_source, **r})

        own_scale_state = result["ablation_states"][roi_source]
        own_scale_method = result["per_method"][roi_source]
        pred_clusters = two_stage_predicted_clusters(
            blobs,
            own_scale_state,
            own_scale_method["threshold"],
            result["mean_patch_prototype"],
            own_scale_method["cluster_reject_thr"],
            result["min_cs"],
            result["q_h"],
            result["q_w"],
            scale_x,
            scale_y,
        )
        two_stage_metrics = match_and_score(
            pred_clusters, result["gt_clusters"], IOU_MATCH_THRESHOLD
        )
        metrics_rows.append(
            {"part_type": pt, "method": f"two-stage({roi_source})", **two_stage_metrics}
        )

    # fg-bg-mean/fg-bg-knn/fg-bg-kmeans two-stage variants: not scale-keyed like the SCALES loop
    # above. ROI blobs come from the combo's first "fg" scale (the scale that state's foreground
    # side is actually built from).
    for combo_name, combo in FGBG_SOURCE_COMBOS.items():
        roi_source_scale = combo["fg"][0]
        if roi_source_scale not in result["blobs_by_scale"]:
            continue  # dropped scale (e.g. "close" below MIN_CROP_SIZE)
        blobs = result["blobs_by_scale"][roi_source_scale]
        fgbg_method_names = (
            f"fg-bg-mean({combo_name})",
            f"fg-bg-proto({combo_name})",
            f"fg-bg-knn({combo_name})",
            *(f"fg-bg-kmeans{k}({combo_name})" for k in KMEANS_KS),
        )
        for method_name in fgbg_method_names:
            if method_name not in result["ablation_states"]:
                continue  # combo skipped for this ref image (see build_fgbg_states)
            state = result["ablation_states"][method_name]
            method_info = result["per_method"][method_name]
            pred_clusters = two_stage_predicted_clusters(
                blobs,
                state,
                method_info["threshold"],
                result["mean_patch_prototype"],
                method_info["cluster_reject_thr"],
                result["min_cs"],
                result["q_h"],
                result["q_w"],
                scale_x,
                scale_y,
            )
            two_stage_metrics = match_and_score(
                pred_clusters, result["gt_clusters"], IOU_MATCH_THRESHOLD
            )
            metrics_rows.append(
                {"part_type": pt, "method": f"two-stage({method_name})", **two_stage_metrics}
            )

if not metrics_rows:
    raise RuntimeError(
        f"No results for RUN_PART_TYPES={RUN_PART_TYPES} with EXEMPLAR_CLASS={EXEMPLAR_CLASS} — "
        "every part type was skipped (see the 'no exemplar instances' warnings above). Pick a "
        "RUN_PART_TYPES entry whose annotations actually contain one of EXEMPLAR_CLASS, or "
        "widen EXEMPLAR_CLASS."
    )
metrics_df = pd.DataFrame(metrics_rows)
log.info("\n%s", metrics_df.to_string(index=False))

summary_df = (
    metrics_df.groupby("method")[["precision", "recall", "f1", "mean_iou", "count_error"]]
    .mean()
    .reindex(METHOD_DISPLAY_ORDER)
)
log.info("Summary (mean across part types):\n%s", summary_df.to_string())

cross_df = pd.DataFrame(cross_rows)
cross_gt = cross_df[cross_df["gt_present"]]
cross_matrix = (
    cross_gt.groupby(["roi_source", "score_scale"])["iou"]
    .mean()
    .unstack("score_scale")
    .reindex(index=SCALES, columns=SCALES)
)
log.info(
    "Cross-scale mean IoU — rows=ROI source scale, cols=scoring prototype scale "
    "(n=%d GT-overlapping blobs / %d total):\n%s",
    len(cross_gt),
    len(cross_df),
    cross_matrix.to_string(),
)

# %% [markdown]
# ## Ablation summary — bar chart + IoU heatmap
# Precision/recall/F1 and mean matched IoU per method (averaged across `RUN_PART_TYPES`), plus
# a method x part-type IoU heatmap.

# %% Visualisation — ablation summary (bar chart + IoU heatmap by part type)
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
summary_df[["precision", "recall", "f1"]].plot.bar(ax=axes[0])
axes[0].set_ylim(0, 1.05)
axes[0].set_title("Precision / Recall / F1 (mean across part types)")
axes[0].tick_params(axis="x", rotation=90)

summary_df["mean_iou"].plot.bar(ax=axes[1], color="teal")
axes[1].set_ylim(0, 1.0)
axes[1].set_title("Mean matched IoU (mean across part types)")
axes[1].tick_params(axis="x", rotation=90)

plt.suptitle(
    f"Multi-scale crop prototype ablation | {len(results)} part types | "
    f"iou_match={IOU_MATCH_THRESHOLD}",
    fontsize=11,
)
plt.show()

iou_pivot = metrics_df.pivot(index="method", columns="part_type", values="mean_iou").reindex(
    index=METHOD_DISPLAY_ORDER
)
fig, ax = plt.subplots(
    figsize=(1.7 * len(iou_pivot.columns) + 2.5, 0.6 * len(iou_pivot) + 2),
    constrained_layout=True,
)
im = ax.imshow(iou_pivot.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(iou_pivot.columns)))
ax.set_xticklabels(iou_pivot.columns)
ax.set_yticks(range(len(iou_pivot)))
ax.set_yticklabels(iou_pivot.index)
for i in range(len(iou_pivot)):
    for j in range(len(iou_pivot.columns)):
        val = iou_pivot.values[i, j]
        if np.isfinite(val):
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if val < 0.6 else "black",
            )
plt.colorbar(im, ax=ax, label="mean matched IoU")
ax.set_title(
    f"Ablation — mean matched IoU per method x part type  (iou_match={IOU_MATCH_THRESHOLD})"
)
plt.show()

# %% [markdown]
# ## Cross-scale similarity heatmap
# Mean IoU when a blob found by one scale's ROI mask (rows) is re-scored with another scale's
# prototype (columns) — the diagonal is today's matched two-stage behaviour, off-diagonal cells
# are the cross-scale generalisation question.

# %% Visualisation — cross-scale similarity heatmap
fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
vmax = max(float(np.nanmax(cross_matrix.values)), 1e-6)
im = ax.imshow(cross_matrix.values, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(SCALES)))
ax.set_xticklabels(SCALES)
ax.set_yticks(range(len(SCALES)))
ax.set_yticklabels(SCALES)
ax.set_xlabel("prototype scale used to score the crop")
ax.set_ylabel("scale whose raw map located the crop (ROI source)")
for i in range(len(SCALES)):
    for j in range(len(SCALES)):
        val = cross_matrix.values[i, j]
        label = f"{val:.2f}" if np.isfinite(val) else "n/a"
        color = "white" if (np.isfinite(val) and val < vmax * 0.6) else "black"
        ax.text(j, i, label, ha="center", va="center", color=color, fontsize=11)
plt.colorbar(im, ax=ax, label="mean IoU vs. crop GT (patch grid)")
ax.set_title(
    "Cross-scale similarity — prototype scored on crops found by another scale\n"
    "diagonal = matched two-stage refinement (today's behaviour)  |  "
    f"off-diagonal = cross-scale  (n={len(cross_gt)} GT-overlapping blobs, "
    f"{len(results)} part types)",
    fontsize=9,
)
plt.show()

# %% [markdown]
# ## Exemplar multi-scale crop overview (`FOCUS_PART_TYPE`)
# The reference image with its GT mask and every scale's crop box overlaid, then a grid of every
# individual crop (one row per scale, one column per cluster) with its own patch mask.

# %% Visualisation — exemplar multi-scale crop overview (FOCUS_PART_TYPE)
if FOCUS_PART_TYPE not in results:
    raise RuntimeError(
        f"FOCUS_PART_TYPE={FOCUS_PART_TYPE!r} was skipped (no exemplar instances for "
        f"classes {EXEMPLAR_CLASS}) — pick a part type present in {sorted(results)}."
    )
focus = results[FOCUS_PART_TYPE]
focus_scale_protos = focus["scale_protos"]
present_scales = [s for s in SCALES if s in focus_scale_protos]  # "close" may be dropped


def _crop_cells(proto: ScalePrototype) -> list[ClusterCrop | ScalePrototype]:
    """The crops to display for one scale: the single global crop, or every cluster."""
    if proto.cluster_crops is None:
        return [proto]
    return [c for c in proto.cluster_crops]


# Reference image + mask + every crop box (global's single box, every cluster's mid/close box).
fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
disp_ref = np.array(focus["ref_img"])
ax.imshow(disp_ref)
ref_mask_disp = (
    np.array(
        Image.fromarray(focus["ref_pixel_mask"].astype(np.uint8) * 255).resize(
            focus["ref_img"].size, Image.NEAREST
        )
    )
    > 0
)
ov = np.zeros((*ref_mask_disp.shape, 4), dtype=np.float32)
ov[ref_mask_disp] = [0.2, 0.9, 0.2, 0.4]
ax.imshow(ov)
for scale in present_scales:
    cells = _crop_cells(focus_scale_protos[scale])
    for i, cell in enumerate(cells):
        x0, y0, x1, y1 = cell.box
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=2,
                edgecolor=SCALE_COLOR[scale],
                facecolor="none",
                label=scale if i == 0 else None,
            )
        )
ax.legend(loc="upper right", fontsize=8)
ax.set_title("reference image + mask + per-cluster crop boxes", fontsize=10)
ax.axis("off")
plt.suptitle(f"Multi-scale exemplar crops — part_type={FOCUS_PART_TYPE}", fontsize=12)
plt.show()

# Grid of every crop: one row per scale, one column per cluster (global has a single
# column). "mid"/"close" prototypes are the mean of every cluster crop shown in their row.
n_cols = max(len(_crop_cells(focus_scale_protos[s])) for s in present_scales)
fig, axes = plt.subplots(
    len(present_scales),
    n_cols,
    figsize=(n_cols * 4, len(present_scales) * 4),
    squeeze=False,
    constrained_layout=True,
)
for row, scale in enumerate(present_scales):
    proto = focus_scale_protos[scale]
    cells = _crop_cells(proto)
    for col in range(n_cols):
        ax = axes[row, col]
        if col >= len(cells):
            ax.axis("off")
            continue
        cell = cells[col]
        crop_arr = np.array(cell.crop_img)
        ax.imshow(crop_arr)
        mask_up = (
            np.array(
                Image.fromarray(cell.patch_mask.astype(np.uint8) * 255).resize(
                    (crop_arr.shape[1], crop_arr.shape[0]), Image.NEAREST
                )
            )
            > 0
        )
        ov2 = np.zeros((*mask_up.shape, 4), dtype=np.float32)
        ov2[mask_up] = [0.9, 0.2, 0.2, 0.4]  # this cluster's own mask (fg -> prototype)
        ax.imshow(ov2)

        x0, y0, x1, y1 = cell.box
        # Other instances' fg that happen to fall inside *this* cluster's own crop box.
        # build_all_scale_prototypes now excludes cell.exclude_patch_mask (union of every
        # instance in the crop) from bg_prototype, so these patches are correctly kept out of
        # bg rather than leaking into it. Still highlighted (yellow) so it's visible which
        # crops actually have a neighbouring instance in-frame; "global"'s cell already covers
        # the union mask so it never has a secondary region by construction.
        n_secondary = 0
        if isinstance(cell, ClusterCrop):
            secondary_patch_mask = cell.exclude_patch_mask & ~cell.patch_mask
            n_secondary = int(secondary_patch_mask.sum())
            if n_secondary:
                sec_up = (
                    np.array(
                        Image.fromarray(secondary_patch_mask.astype(np.uint8) * 255).resize(
                            (crop_arr.shape[1], crop_arr.shape[0]), Image.NEAREST
                        )
                    )
                    > 0
                )
                ov3 = np.zeros((*sec_up.shape, 4), dtype=np.float32)
                ov3[sec_up] = [1.0, 0.85, 0.0, 0.45]  # other instances' fg, excluded from bg
                ax.imshow(ov3)

        title = f"scale={scale}"
        if isinstance(cell, ClusterCrop):
            title += f"  cluster {cell.cluster_idx}"
            title += "  (representative)" if cell.box == proto.box else ""
        title += (
            f"\ncrop {x1 - x0}x{y1 - y0}px\n"
            f"{int(cell.patch_mask.sum())}/{cell.patch_mask.size} patches masked"
        )
        if n_secondary:
            title += f"\n{n_secondary} secondary-instance patches excluded from bg (yellow)"
        ax.set_title(title, fontsize=9)
        ax.axis("off")

plt.suptitle(
    f"Multi-scale exemplar crops per cluster — part_type={FOCUS_PART_TYPE}  "
    "(mid/close prototypes = mean over each row; red=own mask, "
    "yellow=other instances' fg excluded from this cluster's bg)",
    fontsize=12,
)
plt.show()

# %% [markdown]
# ## Detailed per-method breakdown (`FOCUS_PART_TYPE`)
# One row per ablation method, scored on the full query: raw score map, thresholded binary +
# GT outline, DBSCAN clusters, GT-DBSCAN clusters, score histogram, and the mean-patch
# cluster-reject result.

# %% Visualisation — detailed per-method breakdown (FOCUS_PART_TYPE)
methods_present = [m for m in METHOD_DISPLAY_ORDER if m in focus["per_method"]]
disp_q = np.array(focus["query_img"])
scale_x = focus["query_img"].size[0] / IMG_SIZE
scale_y = focus["query_img"].size[1] / IMG_SIZE


def patch_centers_to_px(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (xs + 0.5) * PATCH_SIZE * scale_x, (ys + 0.5) * PATCH_SIZE * scale_y


CMAP = plt.get_cmap("tab10")
n_rows = len(methods_present)
fig, axes = plt.subplots(n_rows, 6, figsize=(28, 5 * n_rows), constrained_layout=True)
if n_rows == 1:
    axes = axes.reshape(1, 6)

for row, name in enumerate(methods_present):
    m = focus["per_method"][name]
    raw_np = m["raw"]

    im0 = axes[row, 0].imshow(raw_np, cmap="jet", aspect="auto")
    axes[row, 0].set_title(f"[{name}] raw score map", fontsize=10)
    axes[row, 0].axis("off")
    plt.colorbar(im0, ax=axes[row, 0], shrink=0.75, pad=0.02)

    axes[row, 1].imshow(m["binary"], cmap="Greys_r", aspect="auto")
    axes[row, 1].contour(
        focus["gt_patch_mask"].astype(float), levels=[0.5], colors="lime", linewidths=1.2
    )
    binary_iou = mask_iou(m["binary"], focus["gt_patch_mask"])
    axes[row, 1].set_title(
        f"binary (thr={m['threshold']:.3f}) + GT outline\nIoU={binary_iou:.2f}", fontsize=10
    )
    axes[row, 1].axis("off")

    axes[row, 2].imshow(disp_q)
    for i, cl in enumerate(m["clusters"]):
        ys_c, xs_c = np.where(cl["mask"])
        px_x, px_y = patch_centers_to_px(xs_c, ys_c)
        axes[row, 2].scatter(px_x, px_y, s=14, color=CMAP(i % 10), label=f"c{i}")
    metrics = m["metrics"]
    axes[row, 2].set_title(
        f"DBSCAN clusters ({len(m['clusters'])})\n"
        f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} "
        f"F1={metrics['f1']:.2f} mIoU={metrics['mean_iou']:.2f}",
        fontsize=9,
    )
    axes[row, 2].axis("off")

    axes[row, 3].imshow(disp_q)
    for i, cl in enumerate(focus["gt_clusters"]):
        ys_c, xs_c = np.where(cl["mask"])
        px_x, px_y = patch_centers_to_px(xs_c, ys_c)
        axes[row, 3].scatter(px_x, px_y, s=14, color=CMAP(i % 10))
    axes[row, 3].set_title(
        f"GT-DBSCAN clusters ({len(focus['gt_clusters'])})\nmin_cluster_size={focus['min_cs']}",
        fontsize=9,
    )
    axes[row, 3].axis("off")

    true_vals = raw_np[focus["gt_patch_mask"]]
    false_vals = raw_np[~focus["gt_patch_mask"]]
    bins = np.linspace(raw_np.min(), raw_np.max(), 40)

    # GT-false patches vastly outnumber GT-true ones (background vs. object), so GT-true
    # gets its own right-hand y-axis/scale — on a shared axis its bars would be invisible.
    ax_false = axes[row, 4]
    ax_true = ax_false.twinx()
    h_false = ax_false.hist(false_vals, bins=bins, alpha=0.6, color="tab:gray", label="GT-false")
    h_true = ax_true.hist(true_vals, bins=bins, alpha=0.6, color="tab:red", label="GT-true")
    ax_false.set_yscale("log")
    ax_false.set_ylabel("GT-false count (log)", color="tab:gray", fontsize=8)
    ax_true.set_ylabel("GT-true count", color="tab:red", fontsize=8)
    ax_false.tick_params(axis="y", labelcolor="tab:gray")
    ax_true.tick_params(axis="y", labelcolor="tab:red")
    thr_line = ax_false.axvline(
        m["threshold"], color="black", linestyle="--", linewidth=1.5, label="threshold"
    )
    ax_false.set_title(f"[{name}] score histogram (thr={m['threshold']:.3f})", fontsize=10)
    handles = [h_false[2][0], h_true[2][0], thr_line]
    labels = ["GT-false", "GT-true", "threshold"]
    ax_false.legend(handles, labels, fontsize=8)

    # Mean-patch reject is disregarded in this view only — shows every raw DBSCAN cluster
    # (m["clusters"]), not m["kept_clusters"]; the real pipeline (run_pair, cluster_reject_thr,
    # per_method's P/R/F1/mIoU in column 2 above) still applies the reject step as normal.
    axes[row, 5].imshow(disp_q)
    for cl in m["clusters"]:
        ys_c, xs_c = np.where(cl["mask"])
        px_x, px_y = patch_centers_to_px(xs_c, ys_c)
        good = mask_iou(cl["mask"], focus["gt_patch_mask"]) >= IOU_MATCH_THRESHOLD
        axes[row, 5].scatter(px_x, px_y, s=14, color="lime" if good else "gray")
    axes[row, 5].set_title(
        f"[{name}] raw DBSCAN clusters, mean-patch reject disregarded\n"
        f"({len(m['clusters'])} clusters, lime=matches GT, gray=no GT overlap)",
        fontsize=8,
    )
    axes[row, 5].axis("off")

plt.suptitle(
    f"Multi-scale crop prototype ablation — detailed view  |  part_type={FOCUS_PART_TYPE}  "
    f"block={LAYER_IDX}",
    fontsize=12,
)
plt.show()


# %% [markdown]
# ## Threshold tuning — pooling all mid crops (`FOCUS_PART_TYPE`)
# Self-supervised: pools the score map + GT mask from *every* mid-scale instance crop (not a
# single representative crop) before running the IoU sweep, so the tuning sample isn't capped at
# whatever one crop happens to contain. That cutoff (`pooled_thr`) is reused as-is on every
# query-side crop below — it is never re-tuned per query. Purely a comparison visualization —
# does not feed back into `per_method` or anything upstream in the real pipeline.

# %% Visualisation — per-crop maps/binary/histogram for every mid crop (FOCUS_PART_TYPE). GT per
# crop is exclude_patch_mask (union of every instance's mask within that crop) — the same "don't
# hide neighbouring instances" mask ClusterCrop already carries.
# prototype_to_use = "fg-bg-knn(mid/all)"
prototype_to_use = "fg-bg-mean(mid/all)"
mid_state = focus["ablation_states"][prototype_to_use]

mid_clusters = focus_scale_protos["mid"].cluster_crops
assert mid_clusters, "mid scale is never dropped, so this can't be empty"

per_crop_raw: list[np.ndarray] = []
per_crop_gt: list[np.ndarray] = []
for cc in mid_clusters:
    per_crop_raw.append(score_method(mid_state, cc.tokens).reshape(cc.grid_h, cc.grid_w))
    per_crop_gt.append(cc.exclude_patch_mask)

pooled_raw = np.concatenate([r.reshape(-1) for r in per_crop_raw])
pooled_gt = np.concatenate([g.reshape(-1) for g in per_crop_gt])
pooled_candidates, pooled_ious = iou_threshold_curve(pooled_raw, pooled_gt, REF_THRESHOLD_STEPS)
pooled_best_idx = int(np.argmax(pooled_ious))
pooled_thr = float(pooled_candidates[pooled_best_idx])

log.info(
    "[%s] pooled all-mid-crop threshold tuning (%d crops, %d patches): thr=%.3f best_iou=%.3f",
    FOCUS_PART_TYPE,
    len(mid_clusters),
    pooled_raw.size,
    pooled_thr,
    pooled_ious[pooled_best_idx],
)

# Cluster-size bound, tuned the same self-supervised way: the query-side gt_sizes used for the
# ablation's own min_cs are measured on the *full-image* grid (see run_pair), which under-counts
# how many patches one instance spans in the much denser mid-crop / ROI-blob grids used below.
# Re-derive the bound from every mid-crop instance's own size instead, pooled the same way as
# pooled_thr, margin-expanded the same way as the ablation.
mid_gt_sizes = np.array([float(cc.patch_mask.sum()) for cc in mid_clusters])
mid_min_cs = min_cluster_size_bound(mid_gt_sizes, CLUSTER_SIZE_MARGIN)
log.info(
    "[%s] pooled mid-crop cluster-size tuning: sizes=%s -> min_cluster_size=%d",
    FOCUS_PART_TYPE,
    mid_gt_sizes.astype(int).tolist(),
    mid_min_cs,
)

n_rows = len(mid_clusters)
fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4.6 * n_rows), constrained_layout=True)
if n_rows == 1:
    axes = axes.reshape(1, 4)

for row, (cc, raw_i, gt_i) in enumerate(zip(mid_clusters, per_crop_raw, per_crop_gt)):
    axes[row, 0].imshow(np.array(cc.crop_img))
    mask_up_i = (
        np.array(
            Image.fromarray(gt_i.astype(np.uint8) * 255).resize(cc.crop_img.size, Image.NEAREST)
        )
        > 0
    )
    ov_i = np.zeros((*mask_up_i.shape, 4), dtype=np.float32)
    ov_i[mask_up_i] = [0.2, 0.9, 0.2, 0.4]
    axes[row, 0].imshow(ov_i)
    axes[row, 0].set_title(
        f"cluster {cc.cluster_idx} mid crop + GT ({int(gt_i.sum())}/{gt_i.size} patches)",
        fontsize=9,
    )
    axes[row, 0].axis("off")

    im1 = axes[row, 1].imshow(raw_i, cmap="jet", aspect="auto")
    axes[row, 1].contour(gt_i.astype(float), levels=[0.5], colors="lime", linewidths=1.2)
    axes[row, 1].set_title(f"cluster {cc.cluster_idx} score map", fontsize=9)
    axes[row, 1].axis("off")
    plt.colorbar(im1, ax=axes[row, 1], shrink=0.75, pad=0.02)

    binary_i = raw_i > pooled_thr
    axes[row, 2].imshow(binary_i, cmap="Greys_r", aspect="auto")
    axes[row, 2].contour(gt_i.astype(float), levels=[0.5], colors="lime", linewidths=1.2)
    axes[row, 2].set_title(
        f"binary (pooled thr={pooled_thr:.3f})\n{int(binary_i.sum())}/{binary_i.size} px kept",
        fontsize=9,
    )
    axes[row, 2].axis("off")

    true_vals_i, false_vals_i = raw_i[gt_i], raw_i[~gt_i]
    bins_i = np.linspace(raw_i.min(), raw_i.max(), 40)
    ax_false_i = axes[row, 3]
    ax_true_i = ax_false_i.twinx()
    h_false_i = ax_false_i.hist(
        false_vals_i, bins=bins_i, alpha=0.6, color="tab:gray", label="GT-false"
    )
    h_true_i = ax_true_i.hist(true_vals_i, bins=bins_i, alpha=0.6, color="tab:red", label="GT-true")
    ax_false_i.set_yscale("log")
    ax_false_i.tick_params(axis="y", labelcolor="tab:gray")
    ax_true_i.tick_params(axis="y", labelcolor="tab:red")
    thr_line_i = ax_false_i.axvline(pooled_thr, color="black", linestyle="--", linewidth=1.5)
    ax_false_i.set_title(f"cluster {cc.cluster_idx} score histogram", fontsize=9)
    ax_false_i.legend(
        [h_false_i[2][0], h_true_i[2][0], thr_line_i],
        ["GT-false", "GT-true", "threshold"],
        fontsize=7,
    )

plt.suptitle(
    f"Threshold tuning source — all mid crops (experiment) | part_type={FOCUS_PART_TYPE}",
    fontsize=12,
)
plt.show()

# %% Visualisation — pooled accumulation histogram + IoU-vs-threshold curve, all mid crops
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

pooled_true, pooled_false = pooled_raw[pooled_gt], pooled_raw[~pooled_gt]
pooled_bins = np.linspace(pooled_raw.min(), pooled_raw.max(), 40)
ax_false = axes[0]
ax_true = ax_false.twinx()
h_false = ax_false.hist(
    pooled_false, bins=pooled_bins, alpha=0.6, color="tab:gray", label="GT-false"
)
h_true = ax_true.hist(pooled_true, bins=pooled_bins, alpha=0.6, color="tab:red", label="GT-true")
ax_false.set_yscale("log")
ax_false.set_ylabel("GT-false count (log)", color="tab:gray", fontsize=8)
ax_true.set_ylabel("GT-true count", color="tab:red", fontsize=8)
ax_false.tick_params(axis="y", labelcolor="tab:gray")
ax_true.tick_params(axis="y", labelcolor="tab:red")
thr_line = ax_false.axvline(pooled_thr, color="black", linestyle="--", linewidth=1.5)
ax_false.set_title(
    f"pooled score histogram, all {len(mid_clusters)} mid crops (thr={pooled_thr:.3f})",
    fontsize=10,
)
ax_false.legend(
    [h_false[2][0], h_true[2][0], thr_line], ["GT-false", "GT-true", "threshold"], fontsize=8
)

axes[1].plot(pooled_candidates, pooled_ious, color="steelblue", label="IoU(threshold)")
axes[1].axvline(
    pooled_thr, color="black", linestyle="--", linewidth=1.5, label=f"pooled thr={pooled_thr:.3f}"
)
axes[1].scatter([pooled_thr], [pooled_ious[pooled_best_idx]], color="black", zorder=5)
axes[1].set_xlabel("candidate threshold")
axes[1].set_ylabel("pooled IoU")
axes[1].set_title("IoU vs threshold — pooled across all mid crops", fontsize=10)
axes[1].legend(fontsize=8)

plt.suptitle(
    f"Threshold tuning — pooled accumulation across mid crops | part_type={FOCUS_PART_TYPE}",
    fontsize=12,
)
plt.show()

# %% [markdown]
# ## Mean-patch cluster-reject threshold tuning (`FOCUS_PART_TYPE`)
# The reference's own predicted clusters (same threshold -> DBSCAN pipeline as the query),
# labelled "good"/"bad" by GT overlap, and the mean-patch cosine-similarity cutoff
# (`cluster_reject_thr`) that best separates them — reused as-is to reject query-side clusters,
# same fit-on-ref/apply-to-query pattern as `pooled_thr` above.

# %% Visualisation — mean-patch cluster-reject threshold tuning (FOCUS_PART_TYPE). Self-supervised
# on the reference's own predicted clusters (same threshold -> DBSCAN pipeline as the query),
# labelled "good" if they overlap the exemplar's own GT mask, "bad" otherwise. The cosine cutoff
# that best separates the two (tune_cluster_reject_threshold, computed once per method in
# run_pair) is reused as-is to reject query-side predicted clusters — same fit-on-ref,
# apply-to-query pattern as pooled_thr above.
cluster_reject_thr = focus["per_method"][prototype_to_use]["cluster_reject_thr"]
ref_tuning_clusters = focus["per_method"][prototype_to_use]["ref_tuning_clusters"]

# tune_cluster_reject_threshold now pools DBSCAN clusters across every mid-scale instance
# crop (not just the single representative one), so group back by source crop for display —
# each cluster carries crop_idx/crop_img/crop_gt_mask identifying which crop it came from.
tuning_by_crop: dict[int, list[dict]] = {}
for cl in ref_tuning_clusters:
    tuning_by_crop.setdefault(cl["crop_idx"], []).append(cl)

# Per-cluster diagnostics: gt_good comes from cl["gt_iou"] >= IOU_MATCH_THRESHOLD, where
# gt_iou is each cluster's *best* IoU against any single GT instance inside its own source
# crop's GT mask (tune_cluster_reject_threshold splits that possibly-multi-instance mask via
# dbscan_clusters_from_mask before matching — comparing against the flat union mask instead
# would deflate IoU toward 1/n_instances for any crop with more than one GT object).
log.info(
    "[%s] ref cluster tuning diagnostics — %d crops, %d clusters, iou_thr=%.2f "
    "cluster_reject_thr=%.3f",
    prototype_to_use,
    len(tuning_by_crop),
    len(ref_tuning_clusters),
    IOU_MATCH_THRESHOLD,
    cluster_reject_thr,
)
for crop_idx, crop_clusters in tuning_by_crop.items():
    for cl in crop_clusters:
        size = int(cl["mask"].sum())
        union_overlap = int((cl["mask"] & cl["crop_gt_mask"]).sum())
        would_keep = cl["mean_patch_score"] >= cluster_reject_thr
        mismatch = (
            " <-- mean-patch score disagrees with gt_good label"
            if would_keep != cl["gt_good"]
            else ""
        )
        log.info(
            "  crop %d cluster: size=%d patches union_overlap=%d/%d best_instance_iou=%.3f "
            "(%s %.2f) -> gt_good=%s | mean_patch_score=%.3f (%s %.3f) -> would_keep=%s%s",
            crop_idx,
            size,
            union_overlap,
            size,
            cl["gt_iou"],
            ">=" if cl["gt_iou"] >= IOU_MATCH_THRESHOLD else "<",
            IOU_MATCH_THRESHOLD,
            cl["gt_good"],
            cl["mean_patch_score"],
            ">=" if would_keep else "<",
            cluster_reject_thr,
            would_keep,
            mismatch,
        )

n_tuning_rows = max(len(tuning_by_crop), 1)
fig, axes = plt.subplots(
    n_tuning_rows, 1, figsize=(6, 4.5 * n_tuning_rows), constrained_layout=True
)
axes = np.atleast_1d(axes)

for row, (crop_idx, crop_clusters) in enumerate(tuning_by_crop.items()):
    crop_img = crop_clusters[0]["crop_img"]
    crop_gt_mask = crop_clusters[0]["crop_gt_mask"]
    ax = axes[row]
    ax.imshow(np.array(crop_img))
    mask_up = (
        np.array(
            Image.fromarray(crop_gt_mask.astype(np.uint8) * 255).resize(
                crop_img.size, Image.NEAREST
            )
        )
        > 0
    )
    ov = np.zeros((*mask_up.shape, 4), dtype=np.float32)
    ov[mask_up] = [0.2, 0.9, 0.2, 0.4]
    ax.imshow(ov)
    for cl in crop_clusters:
        ys_c, xs_c = np.where(cl["mask"])
        grid_h, grid_w = cl["mask"].shape
        px_x, px_y = crop_patch_centers_to_px(xs_c, ys_c, crop_img, grid_h, grid_w)
        ax.scatter(px_x, px_y, s=16, color="lime" if cl["gt_good"] else "red")
    ax.set_title(f"crop {crop_idx} ({len(crop_clusters)} clusters)", fontsize=9)
    ax.axis("off")

if not tuning_by_crop:
    axes[0].text(0.5, 0.5, "no ref clusters found", ha="center", va="center")
    axes[0].axis("off")

plt.suptitle(
    f"Ref clusters used to tune the reject threshold — pooled across mid crops\n"
    f"(green fill=GT mask; green dot=good/GT-overlapping, red dot=bad/spurious, "
    f"n={len(ref_tuning_clusters)})",
    fontsize=11,
)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
if ref_tuning_clusters:
    tuning_scores = np.array([cl["mean_patch_score"] for cl in ref_tuning_clusters])
    tuning_good = np.array([cl["gt_good"] for cl in ref_tuning_clusters])
    tuning_bins = (
        np.linspace(tuning_scores.min(), tuning_scores.max(), 20)
        if tuning_scores.max() > tuning_scores.min()
        else 10
    )
    ax.hist(
        tuning_scores[tuning_good], bins=tuning_bins, alpha=0.6, color="tab:green", label="good"
    )
    ax.hist(tuning_scores[~tuning_good], bins=tuning_bins, alpha=0.6, color="tab:red", label="bad")
    ax.axvline(cluster_reject_thr, color="black", linestyle="--", linewidth=1.5, label="threshold")
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "no ref clusters found", ha="center", va="center")
ax.set_title(
    f"mean-patch score by label, pooled across {len(tuning_by_crop)} mid crops "
    f"(thr={cluster_reject_thr:.3f})",
    fontsize=9,
)

plt.suptitle(
    f"Cluster mean-patch reject — threshold tuning | scale={prototype_to_use} | "
    f"part_type={FOCUS_PART_TYPE}",
    fontsize=12,
)
plt.show()

# %% [markdown]
# ## Crop-making step 1 — ROI segmentation on the full query (`FOCUS_PART_TYPE`)
# Scores the full query with the mid prototype, percentile-thresholds that raw map into an ROI
# mask (unsupervised — no GT involved), then turns each connected ROI blob into one
# native-resolution crop. Step 2 (next cell) scores those crops independently and never touches
# this raw map again.

# %% Visualisation — crop-making step (FOCUS_PART_TYPE): step 1 scores the full query with the
# mid prototype, percentile-thresholds that raw map into an ROI mask (unsupervised — no GT
# involved), then each connected ROI blob becomes one native-resolution crop. Step 2 (next cell)
# scores those crops independently and never touches this raw map again — no stitching back.
stage1_raw = focus["per_method"][prototype_to_use]["raw"]
mid_roi_mask = focus["roi_mask_by_scale"]["mid"]
mid_blobs = focus["blobs_by_scale"]["mid"]

seg_label = ROI_BINARIZE_METHOD
if ROI_BINARIZE_METHOD == "percentile":
    seg_label += f" p={ROI_PERCENTILE}"

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)

im0 = axes[0].imshow(stage1_raw, cmap="jet", aspect="auto")
axes[0].set_title("[mid] step 1 — raw score map (full query)", fontsize=10)
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], shrink=0.75, pad=0.02)

axes[1].imshow(mid_roi_mask, cmap="Greys_r", aspect="auto")
axes[1].set_title(
    f"segmentation ({seg_label}) — {int(mid_roi_mask.sum())}/{mid_roi_mask.size} patches kept",
    fontsize=10,
)
axes[1].axis("off")

axes[2].imshow(disp_q)
for i, blob in enumerate(mid_blobs):
    px0, py0, px1, py1 = blob["px_bbox"]
    axes[2].add_patch(
        Rectangle(
            (px0, py0), px1 - px0, py1 - py0, linewidth=1.5, edgecolor="yellow", facecolor="none"
        )
    )
    axes[2].text(px0, py0 - 6, f"b{i}", color="yellow", fontsize=9)
axes[2].set_title(f"crops generated ({len(mid_blobs)})", fontsize=10)
axes[2].axis("off")

plt.suptitle(
    f"Crop-making step — scale={prototype_to_use} | part_type={FOCUS_PART_TYPE}", fontsize=12
)
plt.show()

# %% [markdown]
# ## Crop-making step 2 — per-crop scoring breakdown (`FOCUS_PART_TYPE`)
# For each crop generated in step 1: its own score map, thresholded binary, DBSCAN clusters
# (shown both in crop-space and projected back onto the original query image), GT-DBSCAN
# clusters, score histogram, and mean-patch cluster-reject result — entirely in the crop's own
# coordinate frame. Thresholded with `pooled_thr` (tuned on foreground pooled across every mid
# crop — see the "Threshold tuning — pooled accumulation" cell above).

# %% Visualisation — detailed per-crop breakdown (FOCUS_PART_TYPE): step 2 scores each crop
# independently — heatmap, then threshold + DBSCAN clustering, entirely in the crop's own
# coordinate frame. Clusters are also projected back onto the original query image purely for
# display (crop-space and projected views are both shown, per request). Factored into a function
# so callers can pass whichever threshold/label they need without duplicating the plotting logic.


def plot_crop_breakdown(
    blobs: list[dict],
    state: MethodState,
    threshold: float,
    threshold_label: str,
    min_cs: int,
    mean_patch_prototype: torch.Tensor,
    cluster_reject_thr: float,
    q_pixel_mask: np.ndarray | None,
    disp_q: np.ndarray,
    scale_label: str,
    part_type: str,
) -> None:
    """Detailed per-crop breakdown grid — one row per blob, entirely in each crop's own
    coordinate frame. *threshold*/*threshold_label* let callers compare different
    threshold-tuning strategies (e.g. single-crop vs. pooled) on the same blobs without
    duplicating this plotting logic.
    """
    if len(blobs) == 0:
        log.warning(
            "[%s] scale=mid produced no crops — nothing to show in the detailed view.", part_type
        )
        return

    n_blobs = len(blobs)
    fig, axes = plt.subplots(n_blobs, 8, figsize=(40, 4.5 * n_blobs), constrained_layout=True)
    if n_blobs == 1:
        axes = axes.reshape(1, 8)

    for row, blob in enumerate(blobs):
        c_h, c_w = blob["c_h"], blob["c_w"]
        crop_arr = np.array(blob["crop_img"])
        crop_raw = score_method(state, blob["crop_tokens"]).reshape(c_h, c_w)
        crop_gt_mask = blob_crop_gt_mask(blob, q_pixel_mask)
        crop_binary = crop_raw > threshold
        n_kept = int(crop_binary.sum())

        ys_bin, xs_bin = np.where(crop_binary)
        if len(xs_bin) < max(MIN_POINTS_FLOOR, min_cs):
            crop_pred_clusters: list[dict] = []
        else:
            crop_pred_clusters = dbscan_clusters(xs_bin, ys_bin, c_h, c_w, crop_raw, min_cs)
        annotate_cluster_rejection(
            crop_pred_clusters,
            blob["crop_tokens"],
            mean_patch_prototype,
            cluster_reject_thr,
        )
        crop_kept_clusters = [c for c in crop_pred_clusters if not c["rejected"]]
        crop_gt_clusters = dbscan_clusters_from_mask(
            crop_gt_mask, GT_DBSCAN_EPS, GT_DBSCAN_MIN_SAMPLES
        )
        crop_metrics = match_and_score(crop_kept_clusters, crop_gt_clusters, IOU_MATCH_THRESHOLD)

        im0 = axes[row, 0].imshow(crop_raw, cmap="jet", aspect="auto")
        axes[row, 0].set_title(f"blob {row} — step 2 score map (mid, on crop)", fontsize=10)
        axes[row, 0].axis("off")
        plt.colorbar(im0, ax=axes[row, 0], shrink=0.75, pad=0.02)

        axes[row, 1].imshow(crop_binary, cmap="Greys_r", aspect="auto")
        if crop_gt_mask.any():
            axes[row, 1].contour(
                crop_gt_mask.astype(float), levels=[0.5], colors="lime", linewidths=1.2
            )
        axes[row, 1].set_title(
            f"binary (thr={threshold:.3f})\n{n_kept}/{crop_binary.size} px kept", fontsize=9
        )
        axes[row, 1].axis("off")

        axes[row, 2].imshow(crop_arr)
        for i, cl in enumerate(crop_pred_clusters):
            ys_c, xs_c = np.where(cl["mask"])
            px_x, px_y = crop_patch_centers_to_px(xs_c, ys_c, blob["crop_img"], c_h, c_w)
            axes[row, 2].scatter(px_x, px_y, s=14, color=CMAP(i % 10))
        axes[row, 2].set_title(
            f"DBSCAN clusters, crop-space ({len(crop_pred_clusters)})\n"
            f"P={crop_metrics['precision']:.2f} R={crop_metrics['recall']:.2f} "
            f"F1={crop_metrics['f1']:.2f}",
            fontsize=9,
        )
        axes[row, 2].axis("off")

        axes[row, 3].imshow(disp_q)
        px0, py0, px1, py1 = blob["px_bbox"]
        axes[row, 3].add_patch(
            Rectangle(
                (px0, py0),
                px1 - px0,
                py1 - py0,
                linewidth=1.2,
                edgecolor="yellow",
                facecolor="none",
            )
        )
        for i, cl in enumerate(crop_pred_clusters):
            ys_c, xs_c = np.where(cl["mask"])
            px_x, px_y = crop_patch_centers_to_native_px(xs_c, ys_c, blob["px_bbox"], c_h, c_w)
            axes[row, 3].scatter(px_x, px_y, s=14, color=CMAP(i % 10))
        axes[row, 3].set_title("same clusters, projected to original image (optional)", fontsize=9)
        axes[row, 3].axis("off")

        axes[row, 4].imshow(crop_arr)
        for i, cl in enumerate(crop_gt_clusters):
            ys_c, xs_c = np.where(cl["mask"])
            px_x, px_y = crop_patch_centers_to_px(xs_c, ys_c, blob["crop_img"], c_h, c_w)
            axes[row, 4].scatter(px_x, px_y, s=14, color=CMAP(i % 10))
        axes[row, 4].set_title(
            f"GT-DBSCAN clusters, crop-space ({len(crop_gt_clusters)})", fontsize=9
        )
        axes[row, 4].axis("off")

        true_vals = crop_raw[crop_gt_mask]
        false_vals = crop_raw[~crop_gt_mask]
        bins = np.linspace(crop_raw.min(), crop_raw.max(), 40)
        ax_false = axes[row, 5]
        ax_true = ax_false.twinx()
        h_false = ax_false.hist(
            false_vals, bins=bins, alpha=0.6, color="tab:gray", label="GT-false"
        )
        h_true = ax_true.hist(true_vals, bins=bins, alpha=0.6, color="tab:red", label="GT-true")
        ax_false.set_yscale("log")
        ax_false.tick_params(axis="y", labelcolor="tab:gray", labelsize=7)
        ax_true.tick_params(axis="y", labelcolor="tab:red", labelsize=7)
        thr_line = ax_false.axvline(threshold, color="black", linestyle="--", linewidth=1.5)
        ax_false.set_title(f"score histogram (thr={threshold:.3f})", fontsize=9)
        if row == 0:
            ax_false.legend(
                [h_false[2][0], h_true[2][0], thr_line],
                ["GT-false", "GT-true", "threshold"],
                fontsize=7,
            )

        mp_scores = np.array([cl["mean_patch_score"] for cl in crop_pred_clusters])
        mp_rejected = np.array([cl["rejected"] for cl in crop_pred_clusters])
        ax_mp = axes[row, 6]
        if len(mp_scores):
            mp_bins = (
                np.linspace(mp_scores.min(), mp_scores.max(), 20)
                if mp_scores.max() > mp_scores.min()
                else 10
            )
            ax_mp.hist(
                mp_scores[~mp_rejected], bins=mp_bins, alpha=0.7, color="tab:green", label="kept"
            )
            ax_mp.hist(
                mp_scores[mp_rejected], bins=mp_bins, alpha=0.7, color="tab:red", label="rejected"
            )
            ax_mp.axvline(
                cluster_reject_thr, color="black", linestyle="--", linewidth=1.5, label="threshold"
            )
            if row == 0:
                ax_mp.legend(fontsize=7)
        else:
            ax_mp.text(0.5, 0.5, "no clusters", ha="center", va="center")
        ax_mp.set_title(f"cluster mean-patch similarity (thr={cluster_reject_thr:.3f})", fontsize=9)

        axes[row, 7].imshow(crop_arr)
        for cl in crop_kept_clusters:
            ys_c, xs_c = np.where(cl["mask"])
            px_x, px_y = crop_patch_centers_to_px(xs_c, ys_c, blob["crop_img"], c_h, c_w)
            good = mask_iou(cl["mask"], crop_gt_mask) >= IOU_MATCH_THRESHOLD
            axes[row, 7].scatter(px_x, px_y, s=14, color="lime" if good else "gray")
        for cl in crop_pred_clusters:
            if not cl["rejected"]:
                continue
            ys_c, xs_c = np.where(cl["mask"])
            px_x, px_y = crop_patch_centers_to_px(xs_c, ys_c, blob["crop_img"], c_h, c_w)
            axes[row, 7].scatter(px_x, px_y, s=14, color="red", marker="x")
        n_rejected = len(crop_pred_clusters) - len(crop_kept_clusters)
        axes[row, 7].set_title(
            f"mean-patch reject (thr={cluster_reject_thr:.3f})\n"
            f"kept={len(crop_kept_clusters)} rejected={n_rejected}",
            fontsize=9,
        )
        axes[row, 7].axis("off")

    plt.suptitle(
        f"Detailed per-crop breakdown (step 2) — scale={scale_label} | "
        f"threshold={threshold_label} | part_type={part_type}",
        fontsize=12,
    )
    plt.show()


_blob_mean_patch_prototype = focus["mean_patch_prototype"]
_blob_cluster_reject_thr = focus["per_method"][prototype_to_use]["cluster_reject_thr"]

plot_crop_breakdown(
    mid_blobs,
    mid_state,
    pooled_thr,
    "pooled (pooled_thr)",
    mid_min_cs,
    _blob_mean_patch_prototype,
    _blob_cluster_reject_thr,
    focus["q_pixel_mask"],
    disp_q,
    prototype_to_use,
    FOCUS_PART_TYPE,
)

# %% [markdown]
# ## IoU calculation — clusters reprojected from crop to original image (`FOCUS_PART_TYPE`)
# Each mid-crop's surviving clusters (thresholded at `pooled_thr`, DBSCAN'd, mean-patch-rejected
# — identical to `two_stage_predicted_clusters`) are projected from their own crop-local patch
# grid onto the full query's own `(q_h, q_w)` patch grid via `project_crop_mask_to_query_grid`,
# merged across blobs (`merge_overlapping_clusters`), then matched against the full-image GT
# clusters (`focus["gt_clusters"]`) with `match_and_score` — the exact reprojection + matching the
# two-stage pipeline's P/R/F1 is built on. That match is otherwise only reported as an aggregate
# metric; this cell draws it directly on the original image.

# %% Visualisation — IoU calculation via crop-to-original reprojection (FOCUS_PART_TYPE)
q_h, q_w = focus["q_h"], focus["q_w"]

reproj_candidates: list[dict] = []
for blob in mid_blobs:
    c_h, c_w = blob["c_h"], blob["c_w"]
    crop_raw = score_method(mid_state, blob["crop_tokens"]).reshape(c_h, c_w)
    ys_bin, xs_bin = np.where(crop_raw > pooled_thr)
    if len(xs_bin) < max(MIN_POINTS_FLOOR, mid_min_cs):
        continue
    crop_clusters = dbscan_clusters(xs_bin, ys_bin, c_h, c_w, crop_raw, mid_min_cs)
    annotate_cluster_rejection(
        crop_clusters, blob["crop_tokens"], _blob_mean_patch_prototype, _blob_cluster_reject_thr
    )
    for cl in crop_clusters:
        if cl["rejected"]:
            continue
        projected = project_crop_mask_to_query_grid(
            cl["mask"], blob["px_bbox"], c_h, c_w, q_h, q_w, PATCH_SIZE, scale_x, scale_y
        )
        if not projected.any():
            continue
        reproj_candidates.append(
            {"mask": projected, "score": cl["score"], "size": int(cl["mask"].sum())}
        )

reproj_clusters = merge_overlapping_clusters(reproj_candidates)
reproj_gt_clusters = focus["gt_clusters"]
reproj_metrics = match_and_score(reproj_clusters, reproj_gt_clusters, IOU_MATCH_THRESHOLD)

# Recover which predicted cluster matched which GT cluster (mirrors match_and_score's own
# greedy score-ranked pass) purely for coloring below — match_and_score itself only returns
# aggregate counts, not the per-cluster assignment.
_order = sorted(range(len(reproj_clusters)), key=lambda i: -reproj_clusters[i]["score"])
matched_gt: set[int] = set()
pred_is_tp = [False] * len(reproj_clusters)
for i in _order:
    best_j, best_iou = -1, 0.0
    for j, gt in enumerate(reproj_gt_clusters):
        if j in matched_gt:
            continue
        iou = mask_iou(reproj_clusters[i]["mask"], gt["mask"])
        if iou > best_iou:
            best_iou, best_j = iou, j
    if best_j >= 0 and best_iou >= IOU_MATCH_THRESHOLD:
        matched_gt.add(best_j)
        pred_is_tp[i] = True

fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
ax.imshow(disp_q)
for j, gt in enumerate(reproj_gt_clusters):
    ys_g, xs_g = np.where(gt["mask"])
    px_x, px_y = patch_centers_to_px(xs_g, ys_g)
    gt_color = "lime" if j in matched_gt else "orange"
    ax.scatter(px_x, px_y, s=70, facecolors="none", edgecolors=gt_color, linewidths=1.5)
for i, cl in enumerate(reproj_clusters):
    ys_c, xs_c = np.where(cl["mask"])
    px_x, px_y = patch_centers_to_px(xs_c, ys_c)
    color, marker = ("lime", "o") if pred_is_tp[i] else ("red", "x")
    ax.scatter(px_x, px_y, s=18, color=color, marker=marker)
ax.legend(
    handles=[
        plt.Line2D([0], [0], marker="o", color="lime", linestyle="", markersize=7, label="pred TP"),
        plt.Line2D([0], [0], marker="x", color="red", linestyle="", markersize=7, label="pred FP"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="lime",
            linestyle="",
            markersize=10,
            label="GT matched",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="orange",
            linestyle="",
            markersize=10,
            label="GT unmatched (FN)",
        ),
    ],
    loc="upper right",
    fontsize=8,
)
ax.set_title(
    f"IoU calculation — crop clusters reprojected onto original image | scale={prototype_to_use} | "
    f"part_type={FOCUS_PART_TYPE}\n"
    f"pred={len(reproj_clusters)} gt={len(reproj_gt_clusters)}  "
    f"TP={reproj_metrics['tp']} FP={reproj_metrics['fp']} FN={reproj_metrics['fn']}  "
    f"P={reproj_metrics['precision']:.2f} R={reproj_metrics['recall']:.2f} "
    f"F1={reproj_metrics['f1']:.2f} mean_IoU={reproj_metrics['mean_iou']:.2f}",
    fontsize=10,
)
ax.axis("off")
plt.show()


# %%
