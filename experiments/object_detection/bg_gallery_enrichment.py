# %% [markdown]
# # Background-Gallery Enrichment Ablation
#
# Baselines: **fg-bg-proto(global+mid+close/all)** and **fg-bg-knn(global+mid+close/all)** —
# the best-performing contrastive methods from ``multiscale_crop_ablation.py``'s ablation
# (foreground = each of the three exemplar scales' own prototype, kept as a separate row;
# background = the mean/gallery of every scale's own non-foreground patches).
#
# Their background side only ever sees patches *inside* the mid/close crops that were built
# around real instances — background just outside those crops (but still somewhere in the
# reference image) is never sampled. This experiment tests whether a richer background
# gallery, drawn from parts of the reference image that were never inside any mid/close
# instance crop, improves discrimination:
#
# For each reference image, sample up to ``MAX_EXTRA_BG_CROPS`` additional crops at **mid**
# scale and ``MAX_EXTRA_BG_CROPS`` more at **close** scale (crop sizes drawn from that scale's
# own existing crop-size distribution), each rejected if it contains *any* foreground pixel or
# overlaps an existing crop *of the same scale* (original or already-accepted-extra — a new
# close crop only avoids other close crops, a new mid crop only avoids other mid crops; the
# two scales are free to overlap each other) by more than ``EXTRA_BG_MAX_OVERLAP_FRACTION``.
# Each accepted crop is encoded once; its *entire* token set counts as background (there's no
# foreground to exclude).
#
# The first ``n`` extra crops per scale (n = 0..``MAX_EXTRA_BG_CROPS``, n=0 is the plain
# baseline) are folded into the mid/close background pools (mean-collapsed for fg-bg-proto,
# raw patch gallery for fg-bg-knn) — "global"'s background is untouched, since only mid/close
# have a well-defined per-instance crop geometry to sample "not considered before" bg from.
# Every n is scored end-to-end exactly like ``multiscale_crop_ablation.py``'s ablation: ref-tuned
# threshold -> DBSCAN -> mean-patch cluster reject -> greedy IoU match against GT clusters ->
# precision/recall/F1/mean IoU.
#
# Runs every (part_type, instance_type) pair actually annotated in ``data/abc3`` (see
# ``dinoisawesome.abc3.available_instance_groups``) — restrict via RUN_PART_TYPES /
# RUN_INSTANCE_TYPES for fast iteration. No figures are shown interactively (this is meant to
# run as ``python bg_gallery_enrichment.py``, not just as notebook cells); everything is written
# to ``outputs/object_detection/bg_gallery_enrichment/``.

# %% Logging — must be before torch import
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from matplotlib.patches import Patch, Rectangle
from PIL import Image

from dinoisawesome import DinoEncoder, EncoderWithCache
from dinoisawesome.abc3 import (
    INSTANCE_TYPE_GROUPS,
    PART_TYPES,
    available_instance_groups,
    load_instance_pixel_mask,
    load_instance_pixel_masks,
)
from dinoisawesome.instance_detection import compute_exemplar_features, extract_patch_tokens

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared.clustering import (  # noqa: E402
    dbscan_clusters,
    dbscan_clusters_from_mask,
    match_and_score,
    min_cluster_size_bound,
    patch_radius_to_eps,
)
from _shared.gt_utils import (
    gt_instance_patch_sizes as _shared_gt_instance_patch_sizes,  # noqa: E402
)
from _shared.mask_geometry import pixel_mask_to_patch_mask, scale_crop_box  # noqa: E402
from _shared.prototype_ops import extract_patch_tokens_batch, knn_fgbg_score  # noqa: E402
from _shared.thresholding import iou_tuned_threshold  # noqa: E402
from _shared.two_stage import annotate_cluster_rejection  # noqa: E402
from _shared.two_stage import tune_cluster_reject_threshold as _shared_tune_cluster_reject_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("bg_gallery_enrichment")

# %% Setup — repo paths + env vars
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")
data_dir = _REPO_ROOT / "data" / "abc3"

OUTPUT_DIR = _REPO_ROOT / "outputs" / "object_detection" / "bg_gallery_enrichment"
CROPS_DIR = OUTPUT_DIR / "crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# %% Parameters
REF_NUMBER = 1
QUERY_NUMBER = 2

# None = every part type / every instance-type group actually annotated for that part type
# (see dinoisawesome.abc3.available_instance_groups) — restrict either for fast iteration.
RUN_PART_TYPES: list[str] | None = None
RUN_INSTANCE_TYPES: list[str] | None = None

DINO_VERSION = "v3"
DINO_SIZE = "large"
# 1024, not multiscale_crop_ablation.py's 768: at 768 the ref crop's own (coarser-grid)
# IoU-vs-threshold curve can be razor-thin — the chosen threshold sits right at the score
# range's edge, so a threshold tuned near-perfectly on the ref crop turns out too strict for
# the query and yields zero detections. Matches multiscale_ablation/common.py's
# DEFAULT_CROP_CONFIG (the current golden implementation of these methods), which was
# verified (against its own cached results) to reproduce exactly at this value.
IMG_SIZE = 1024
LAYER_IDX = 23
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")
DINO_ENCODING_CACHE_DIR: str | None = os.environ.get("DINO_ENCODING_CACHE_DIR")
DEBIAS = True

MASK_PATCH_THRESHOLD = 0.3
EXEMPLAR_CLOSE_PADDING_FRACTION = 1.0
MIN_CROP_SIZE = 128

# --- Background enrichment ---
MAX_EXTRA_BG_CROPS = 10  # per scale (mid, close) — n sweeps 0..MAX_EXTRA_BG_CROPS
# "Overlap" is measured as intersection-area / candidate-crop-area, checked only against
# existing crops of the *same* scale (original + already-accepted-extra) — an extra "close"
# crop only avoids other close crops, an extra "mid" crop only avoids other mid crops. Mid
# crops are large (often ~1/3 of the image each), so a same-scale-only budget matters: cross-
# scale blocking left some images with no legal spot for another mid crop at all.
EXTRA_BG_MAX_OVERLAP_FRACTION = 0.35
EXTRA_BG_MAX_ATTEMPTS_PER_CROP = 200
EXTRA_BG_SEED = 0

KNN_FGBG_NUM_NEIGHBOURS = 10
REF_THRESHOLD_STEPS = 25

CLUSTER_SIZE_MARGIN: int | float = 0.5
GT_DBSCAN_EPS = 1.5
GT_DBSCAN_MIN_SAMPLES = 1
IOU_MATCH_THRESHOLD = 0.3
MIN_POINTS_FLOOR = 2
CLUSTER_REJECT_MARGIN_FRACTION = 0.2
# 1, not multiscale_crop_ablation.py's 2 — matches multiscale_ablation/common.py's
# DEFAULT_SCORING_CONFIG (the golden implementation).
PRED_DBSCAN_EPS_PATCHES = 1
PRED_DBSCAN_MIN_SAMPLES = 2

SEED = 0

# %% Derived parameters
PRED_DBSCAN_EPS = patch_radius_to_eps(PRED_DBSCAN_EPS_PATCHES)
torch.manual_seed(SEED)

log.info(
    "RUN_PART_TYPES=%s RUN_INSTANCE_TYPES=%s | ref=%d query=%d | MAX_EXTRA_BG_CROPS=%d | "
    "max_overlap=%.0f%%",
    RUN_PART_TYPES,
    RUN_INSTANCE_TYPES,
    REF_NUMBER,
    QUERY_NUMBER,
    MAX_EXTRA_BG_CROPS,
    EXTRA_BG_MAX_OVERLAP_FRACTION * 100,
)


def all_pairs() -> list[tuple[str, str]]:
    """Every (part_type, instance_type) actually annotated in data/abc3, honoring the
    RUN_PART_TYPES / RUN_INSTANCE_TYPES filters (None = no filter, i.e. run everything)."""
    part_types = RUN_PART_TYPES if RUN_PART_TYPES is not None else PART_TYPES
    pairs: list[tuple[str, str]] = []
    for pt in part_types:
        ref_ann_stem = data_dir / "annotations" / f"{pt}_{REF_NUMBER}"
        groups = available_instance_groups(ref_ann_stem, INSTANCE_TYPE_GROUPS)
        if RUN_INSTANCE_TYPES is not None:
            groups = [g for g in groups if g in RUN_INSTANCE_TYPES]
        pairs.extend((pt, g) for g in groups)
    return pairs


def gt_instance_patch_sizes(
    stem: str, class_filter: list[str] | None, grid_h: int, grid_w: int
) -> np.ndarray:
    return _shared_gt_instance_patch_sizes(
        stem, class_filter, grid_h, grid_w, IMG_SIZE, MASK_PATCH_THRESHOLD, data_dir
    )


def tune_cluster_reject_threshold(
    ref_crops: list["ClusterCrop"],
    score_fn,
    mean_patch_prototype: torch.Tensor,
    patch_thr: float,
    min_cs: int,
    iou_thr: float,
) -> tuple[float, list[dict]]:
    return _shared_tune_cluster_reject_threshold(
        ref_crops,
        score_fn,
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
    )


# %% Multi-scale exemplar prototype builder (global/mid/close, mean fg + mean bg per scale)
# — same construction as multiscale_crop_ablation.py's build_all_scale_prototypes, kept here
# standalone (not imported) since this experiment only needs the mean-prototype variant, never
# the k-means/multi-scale-combine machinery that file also builds.


@dataclass
class ClusterCrop:
    """One real instance's own crop at one scale (mid/close)."""

    cluster_idx: int
    box: tuple[int, int, int, int]
    tokens: torch.Tensor  # (H*W, C) L2-normalised
    grid_h: int
    grid_w: int
    patch_mask: np.ndarray  # this instance's own mask within the crop
    exclude_patch_mask: np.ndarray  # union of every instance's mask within the crop
    prototype: torch.Tensor  # (1, C) masked-mean over patch_mask
    bg_prototype: torch.Tensor  # (1, C) masked-mean over ~exclude_patch_mask


@dataclass
class ScalePrototype:
    scale: str
    box: tuple[int, int, int, int]
    tokens: torch.Tensor
    grid_h: int
    grid_w: int
    patch_mask: np.ndarray
    prototype: torch.Tensor  # (1, C) — mean of per-cluster prototypes ("global": its own mean)
    bg_prototype: torch.Tensor  # (1, C) — mean of per-cluster bg prototypes
    cluster_crops: list[ClusterCrop] | None = None  # None for "global"


def _masked_mean(tokens: torch.Tensor, mask: np.ndarray, what: str) -> torch.Tensor:
    flat = torch.from_numpy(mask.reshape(-1)).to(tokens.device)
    sel = tokens[flat]
    if sel.shape[0] == 0:
        log.warning("empty %s selection — using all crop patches", what)
        sel = tokens
    return compute_exemplar_features(sel, mode="mean")


def build_all_scale_prototypes(
    encoder: DinoEncoder, ref_img: Image.Image, instance_masks: list[np.ndarray]
) -> tuple[dict[str, ScalePrototype], torch.Tensor]:
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
            "union_mask_crop": union_mask,
        }
    ]
    for cluster_idx, inst_mask in enumerate(instance_masks):
        for scale in ("mid", "close"):
            box = scale_crop_box(inst_mask, scale, EXEMPLAR_CLOSE_PADDING_FRACTION)
            x0, y0, x1, y1 = box
            if scale == "close" and (x1 - x0 < MIN_CROP_SIZE or y1 - y0 < MIN_CROP_SIZE):
                log.warning(
                    "scale=close cluster=%d: crop %dx%dpx below MIN_CROP_SIZE=%dpx — dropping",
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
        exclude_patch_mask = pixel_mask_to_patch_mask(
            entry["union_mask_crop"], grid_h, grid_w, IMG_SIZE, MASK_PATCH_THRESHOLD
        )
        prototype = _masked_mean(tokens, patch_mask, "foreground")
        bg_prototype = _masked_mean(tokens, ~exclude_patch_mask, "background")
        cc = ClusterCrop(
            entry["cluster_idx"],
            entry["box"],
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
            log.warning("scale=%s: every cluster crop was dropped — dropping this scale", scale)
            continue
        avg = F.normalize(
            torch.cat([c.prototype for c in clusters], dim=0).mean(dim=0, keepdim=True), p=2, dim=-1
        )
        bg_avg = F.normalize(
            torch.cat([c.bg_prototype for c in clusters], dim=0).mean(dim=0, keepdim=True),
            p=2,
            dim=-1,
        )
        rep = max(clusters, key=lambda c: int(c.patch_mask.sum()))
        protos[scale] = ScalePrototype(
            scale,
            rep.box,
            rep.tokens,
            rep.grid_h,
            rep.grid_w,
            rep.patch_mask,
            avg,
            bg_avg,
            clusters,
        )

    all_instance_protos = clusters_by_scale["mid"] + clusters_by_scale["close"]
    assert all_instance_protos, "mid clusters are never dropped when present"
    mean_patch_prototype = F.normalize(
        torch.cat([c.prototype for c in all_instance_protos], dim=0).mean(dim=0, keepdim=True),
        p=2,
        dim=-1,
    )
    return protos, mean_patch_prototype


# %% Extra background-only crop sampling
#
# Each accepted crop has zero foreground pixels (it's rejected outright if it overlaps the
# union instance mask at all), so its *entire* token set is background — no exclude mask
# needed, unlike a real ClusterCrop.


@dataclass
class ExtraBgCrop:
    scale: str
    box: tuple[int, int, int, int]
    tokens: torch.Tensor  # (H*W, C) L2-normalised — every patch counts as background
    grid_h: int
    grid_w: int
    mean_token: torch.Tensor  # (1, C) plain mean over every patch


def _box_overlap_fraction(
    candidate: tuple[int, int, int, int], other: tuple[int, int, int, int]
) -> float:
    """Intersection area as a fraction of *candidate*'s own area (not IoU) — "how much of the
    new crop was already covered by an existing one", which is what a 20% overlap budget means.
    """
    cx0, cy0, cx1, cy1 = candidate
    ox0, oy0, ox1, oy1 = other
    ix0, iy0 = max(cx0, ox0), max(cy0, oy0)
    ix1, iy1 = min(cx1, ox1), min(cy1, oy1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    cand_area = (cx1 - cx0) * (cy1 - cy0)
    if cand_area <= 0:
        return 0.0
    return ((ix1 - ix0) * (iy1 - iy0)) / cand_area


def sample_extra_bg_crops(
    rng: np.random.Generator,
    ref_img: Image.Image,
    ref_pixel_mask: np.ndarray,
    scale_protos: dict[str, ScalePrototype],
    scale: str,
    count: int,
    encoder: DinoEncoder,
) -> list[ExtraBgCrop]:
    """Rejection-sample up to *count* crops at *scale*'s own size distribution that contain no
    foreground and overlap no existing crop *of the same scale* (original or already-accepted-
    extra) by more than EXTRA_BG_MAX_OVERLAP_FRACTION. The other scale's crops are ignored —
    an extra "close" crop nested inside a "mid" crop's footprint is still new background
    territory at the close scale.
    """
    target_clusters = scale_protos[scale].cluster_crops if scale in scale_protos else None
    if target_clusters is None:
        return []
    sizes = [(c.box[2] - c.box[0], c.box[3] - c.box[1]) for c in target_clusters]
    H, W = ref_pixel_mask.shape

    blocked_boxes: list[tuple[int, int, int, int]] = [c.box for c in target_clusters]

    pending: list[dict] = []
    for _ in range(count):
        accepted_box = None
        for _attempt in range(EXTRA_BG_MAX_ATTEMPTS_PER_CROP):
            w, h = sizes[int(rng.integers(len(sizes)))]
            w, h = min(w, W), min(h, H)
            x0 = int(rng.integers(0, W - w + 1)) if W > w else 0
            y0 = int(rng.integers(0, H - h + 1)) if H > h else 0
            x1, y1 = x0 + w, y0 + h
            if ref_pixel_mask[y0:y1, x0:x1].any():
                continue
            box = (x0, y0, x1, y1)
            if any(
                _box_overlap_fraction(box, other) > EXTRA_BG_MAX_OVERLAP_FRACTION
                for other in blocked_boxes
            ):
                continue
            accepted_box = box
            break
        if accepted_box is None:
            log.warning(
                "scale=%s: could only sample %d/%d extra bg crops (overlap/attempt budget "
                "exhausted)",
                scale,
                len(pending),
                count,
            )
            break
        blocked_boxes.append(accepted_box)
        pending.append({"box": accepted_box, "crop_img": ref_img.crop(accepted_box)})

    if not pending:
        return []
    tokens_batch = extract_patch_tokens_batch(
        encoder, [p["crop_img"] for p in pending], LAYER_IDX, debias=DEBIAS
    )
    extras: list[ExtraBgCrop] = []
    for entry, (tokens, grid_h, grid_w) in zip(pending, tokens_batch):
        mean_token = compute_exemplar_features(tokens, mode="mean")
        extras.append(ExtraBgCrop(scale, entry["box"], tokens, grid_h, grid_w, mean_token))
    return extras


# %% Method states — fg-bg-proto / fg-bg-knn over (global, mid, close), background enriched
# with the first n extra crops per scale.

FG_SCALES = ["global", "mid", "close"]
BG_SCALES = ["global", "mid", "close"]


@dataclass
class MethodState:
    name: str
    kind: str  # "fgbg_multi" | "knn_fgbg"
    payload: torch.Tensor | None = None  # (K+1, C): K fg rows + 1 bg row
    fg_bank: torch.Tensor | None = None
    bg_bank: torch.Tensor | None = None


def _pool_fg_tokens(proto: ScalePrototype) -> torch.Tensor:
    if proto.cluster_crops is None:
        flat = torch.from_numpy(proto.patch_mask.reshape(-1)).to(proto.tokens.device)
        sel = proto.tokens[flat]
        return sel if sel.shape[0] > 0 else proto.tokens
    chunks = []
    for c in proto.cluster_crops:
        flat = torch.from_numpy(c.patch_mask.reshape(-1)).to(c.tokens.device)
        chunks.append(c.tokens[flat])
    return torch.cat(chunks, dim=0)


def _scale_bg_mean_and_gallery(
    proto: ScalePrototype, extras: list[ExtraBgCrop], n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(bg_mean (1, C), bg_gallery (Nbg, C)) for one scale, folding in extras[:n]."""
    if proto.cluster_crops is None:  # "global" — never enriched
        flat = torch.from_numpy(proto.patch_mask.reshape(-1)).to(proto.tokens.device)
        gallery = proto.tokens[~flat]
        if gallery.shape[0] == 0:
            gallery = proto.tokens
        return proto.bg_prototype, gallery

    extra_n = extras[:n]
    means = [c.bg_prototype for c in proto.cluster_crops] + [e.mean_token for e in extra_n]
    gallery_chunks = []
    for c in proto.cluster_crops:
        flat = torch.from_numpy(c.exclude_patch_mask.reshape(-1)).to(c.tokens.device)
        gallery_chunks.append(c.tokens[~flat])
    gallery_chunks.extend(e.tokens for e in extra_n)
    bg_mean = F.normalize(torch.cat(means, dim=0).mean(dim=0, keepdim=True), p=2, dim=-1)
    return bg_mean, torch.cat(gallery_chunks, dim=0)


def build_states_for_n(
    scale_protos: dict[str, ScalePrototype],
    extra_bg_by_scale: dict[str, list[ExtraBgCrop]],
    n: int,
) -> dict[str, MethodState]:
    if not all(s in scale_protos for s in FG_SCALES):
        return {}

    fg_protos = torch.cat([scale_protos[s].prototype for s in FG_SCALES], dim=0)  # (3, C)
    fg_gallery = torch.cat([_pool_fg_tokens(scale_protos[s]) for s in FG_SCALES], dim=0)

    bg_means, bg_gallery_chunks = [], []
    for s in BG_SCALES:
        mean_vec, gallery = _scale_bg_mean_and_gallery(
            scale_protos[s], extra_bg_by_scale.get(s, []), n
        )
        bg_means.append(mean_vec)
        bg_gallery_chunks.append(gallery)
    bg_mean = F.normalize(torch.cat(bg_means, dim=0).mean(dim=0, keepdim=True), p=2, dim=-1)
    bg_gallery = torch.cat(bg_gallery_chunks, dim=0)

    proto_payload = torch.cat([fg_protos, bg_mean], dim=0)  # (4, C), last row = bg
    return {
        "fg-bg-proto": MethodState("fg-bg-proto", "fgbg_multi", payload=proto_payload),
        "fg-bg-knn": MethodState("fg-bg-knn", "knn_fgbg", fg_bank=fg_gallery, bg_bank=bg_gallery),
    }


def score_method(state: MethodState, query_tokens: torch.Tensor) -> np.ndarray:
    if state.kind == "knn_fgbg":
        assert state.fg_bank is not None and state.bg_bank is not None
        return knn_fgbg_score(query_tokens, state.fg_bank, state.bg_bank, KNN_FGBG_NUM_NEIGHBOURS)
    assert state.payload is not None
    sim = query_tokens @ state.payload.T  # (N, K+1)
    fg_max = sim[:, :-1].max(dim=-1).values
    return (fg_max - sim[:, -1]).cpu().float().numpy()


# %% Load encoder (shared across all pairs)
encoder = DinoEncoder(
    version=DINO_VERSION, size=DINO_SIZE, img_size=IMG_SIZE, weights_dir=DINO_WEIGHTS_DIR, amp=True
)
encoder = EncoderWithCache(encoder, cache_dir=DINO_ENCODING_CACHE_DIR)
log.info(
    "DINOv%s-%s | patch_size=%d | grid=%dx%d",
    DINO_VERSION[1],
    DINO_SIZE,
    encoder.patch_size,
    encoder.grid_h,
    encoder.grid_w,
)


# %% Per-method evaluation (threshold tune -> DBSCAN -> cluster reject -> GT match)


def evaluate_method(
    state: MethodState,
    q_tokens: torch.Tensor,
    q_h: int,
    q_w: int,
    ref_mid: ScalePrototype,
    ref_mid_gt_mask: np.ndarray,
    mean_patch_prototype: torch.Tensor,
    min_cs: int,
    gt_clusters: list[dict],
) -> dict:
    raw = score_method(state, q_tokens).reshape(q_h, q_w)
    ref_raw = score_method(state, ref_mid.tokens).reshape(ref_mid.grid_h, ref_mid.grid_w)
    thr = iou_tuned_threshold(ref_raw, ref_mid_gt_mask, REF_THRESHOLD_STEPS)

    assert ref_mid.cluster_crops
    cluster_reject_thr, _ref_tuning_clusters = tune_cluster_reject_threshold(
        ref_mid.cluster_crops,
        lambda tokens: score_method(state, tokens),
        mean_patch_prototype,
        thr,
        min_cs,
        IOU_MATCH_THRESHOLD,
    )

    ys, xs = np.where(raw > thr)
    if len(xs) < max(MIN_POINTS_FLOOR, min_cs):
        pred_clusters: list[dict] = []
    else:
        pred_clusters = dbscan_clusters(
            xs, ys, q_h, q_w, raw, PRED_DBSCAN_EPS, PRED_DBSCAN_MIN_SAMPLES, min_cs
        )
    annotate_cluster_rejection(pred_clusters, q_tokens, mean_patch_prototype, cluster_reject_thr)
    kept = [c for c in pred_clusters if not c["rejected"]]
    metrics = match_and_score(kept, gt_clusters, IOU_MATCH_THRESHOLD)
    return {"raw": raw, "threshold": thr, "metrics": metrics}


# %% Sanity-check visualisation — where did the extra bg crops land, per pair
#
# Green/orange outlines are the original mid/close instance crops; cyan/blue are the sampled
# extra bg crops. The union foreground mask is shaded red. Saved (never shown interactively —
# see module docstring) for every pair actually run, not just one.

_CROP_STYLE = {
    "mid": {"orig": "#f39c12", "extra": "#00e5ff"},
    "close": {"orig": "#e74c3c", "extra": "#3949ab"},
}


def save_crop_placement_plot(
    pair_slug: str,
    ref_img: Image.Image,
    ref_pixel_mask: np.ndarray,
    scale_protos: dict[str, ScalePrototype],
    extra_bg_by_scale: dict[str, list[ExtraBgCrop]],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(ref_img)
    fg_overlay = np.zeros((*ref_pixel_mask.shape, 4))
    fg_overlay[ref_pixel_mask] = [1, 0, 0, 0.25]
    ax.imshow(fg_overlay)

    for scale in ("mid", "close"):
        for c in scale_protos[scale].cluster_crops:
            x0, y0, x1, y1 = c.box
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor=_CROP_STYLE[scale]["orig"],
                    linewidth=2,
                )
            )
        for e in extra_bg_by_scale[scale]:
            x0, y0, x1, y1 = e.box
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor=_CROP_STYLE[scale]["extra"],
                    linewidth=1.5,
                    linestyle="--",
                )
            )
    ax.legend(
        handles=[
            Patch(edgecolor=_CROP_STYLE["mid"]["orig"], facecolor="none", label="mid (original)"),
            Patch(edgecolor=_CROP_STYLE["mid"]["extra"], facecolor="none", label="mid (extra bg)"),
            Patch(
                edgecolor=_CROP_STYLE["close"]["orig"], facecolor="none", label="close (original)"
            ),
            Patch(
                edgecolor=_CROP_STYLE["close"]["extra"], facecolor="none", label="close (extra bg)"
            ),
        ],
        loc="upper right",
    )
    ax.set_title(f"{pair_slug} — original vs. extra bg crop placement")
    ax.axis("off")
    fig.tight_layout()
    out_path = CROPS_DIR / f"{pair_slug}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("[%s] saved crop placement figure -> %s", pair_slug, out_path)


# %% Per-pair pipeline


def run_pair(part_type: str, instance_type: str, ref_number: int, query_number: int) -> dict | None:
    exemplar_class = INSTANCE_TYPE_GROUPS[instance_type]
    safe_instance_type = instance_type.replace(" ", "-").replace("/", "-")
    pair_slug = f"{part_type}__{safe_instance_type}"

    ref_stem = f"{part_type}_{ref_number}"
    query_stem = f"{part_type}_{query_number}"

    ref_instance_masks = load_instance_pixel_masks(
        data_dir / "annotations" / ref_stem, exemplar_class
    )
    if not ref_instance_masks:
        log.warning(
            "[%s] no exemplar instances for classes %s — skipping", pair_slug, exemplar_class
        )
        return None
    ref_pixel_mask = np.stack(ref_instance_masks).any(axis=0)
    ref_img = Image.open(data_dir / f"{ref_stem}.jpg").convert("RGB")
    query_img = Image.open(data_dir / f"{query_stem}.jpg").convert("RGB")

    scale_protos, mean_patch_prototype = build_all_scale_prototypes(
        encoder, ref_img, ref_instance_masks
    )
    if not all(s in scale_protos for s in FG_SCALES):
        missing = set(FG_SCALES) - set(scale_protos)
        log.warning(
            "[%s] scale(s) %s dropped (below MIN_CROP_SIZE) — this experiment needs "
            "global+mid+close, skipping",
            pair_slug,
            missing,
        )
        return None

    rng = np.random.default_rng(EXTRA_BG_SEED)
    extra_bg_by_scale = {
        scale: sample_extra_bg_crops(
            rng, ref_img, ref_pixel_mask, scale_protos, scale, MAX_EXTRA_BG_CROPS, encoder
        )
        for scale in ("mid", "close")
    }
    for scale, extras in extra_bg_by_scale.items():
        log.info(
            "[%s] scale=%s sampled %d/%d extra bg crops",
            pair_slug,
            scale,
            len(extras),
            MAX_EXTRA_BG_CROPS,
        )
    save_crop_placement_plot(pair_slug, ref_img, ref_pixel_mask, scale_protos, extra_bg_by_scale)

    q_tokens, q_h, q_w = extract_patch_tokens(encoder, query_img, LAYER_IDX, debias=DEBIAS)
    q_pixel_mask = load_instance_pixel_mask(data_dir / "annotations" / query_stem, exemplar_class)
    gt_patch_mask = (
        pixel_mask_to_patch_mask(q_pixel_mask, q_h, q_w, IMG_SIZE, MASK_PATCH_THRESHOLD)
        if q_pixel_mask is not None
        else np.zeros((q_h, q_w), dtype=bool)
    )
    gt_sizes = gt_instance_patch_sizes(query_stem, exemplar_class, q_h, q_w)
    min_cs = min_cluster_size_bound(gt_sizes, CLUSTER_SIZE_MARGIN, MIN_POINTS_FLOOR)
    gt_clusters = dbscan_clusters_from_mask(gt_patch_mask, GT_DBSCAN_EPS, GT_DBSCAN_MIN_SAMPLES)
    log.info(
        "[%s] GT instances=%d sizes=%s -> min_cluster_size=%d -> GT-DBSCAN clusters=%d",
        pair_slug,
        len(gt_sizes),
        gt_sizes.tolist(),
        min_cs,
        len(gt_clusters),
    )

    ref_mid = scale_protos["mid"]
    mx0, my0, mx1, my1 = ref_mid.box
    ref_mid_gt_mask = pixel_mask_to_patch_mask(
        ref_pixel_mask[my0:my1, mx0:mx1],
        ref_mid.grid_h,
        ref_mid.grid_w,
        IMG_SIZE,
        MASK_PATCH_THRESHOLD,
    )

    rows: list[dict] = []
    for n in range(0, MAX_EXTRA_BG_CROPS + 1):
        # Each scale saturates independently at however many crops it actually sampled — e.g. if
        # "mid" only found 3/10 (large mid crops leave little overlap-budget headroom) while
        # "close" found all 10, mid flat-lines past n=3 but close keeps varying up to n=10;
        # _scale_bg_mean_and_gallery's extras[:n] slicing handles this per scale on its own.
        states = build_states_for_n(scale_protos, extra_bg_by_scale, n)
        n_eff = {s: min(n, len(extra_bg_by_scale[s])) for s in ("mid", "close")}
        for family, state in states.items():
            result = evaluate_method(
                state,
                q_tokens,
                q_h,
                q_w,
                ref_mid,
                ref_mid_gt_mask,
                mean_patch_prototype,
                min_cs,
                gt_clusters,
            )
            m = result["metrics"]
            log.info(
                "[%s/%s n=%d] thr=%.3f P=%.2f R=%.2f F1=%.2f mIoU=%.2f",
                part_type,
                family,
                n,
                result["threshold"],
                m["precision"],
                m["recall"],
                m["f1"],
                m["mean_iou"],
            )
            rows.append(
                {
                    "part_type": part_type,
                    "instance_type": instance_type,
                    "family": family,
                    "n": n,
                    "n_effective_mid": n_eff["mid"],
                    "n_effective_close": n_eff["close"],
                    **m,
                }
            )

    return {"metrics_rows": rows}


# %% Run across every (part_type, instance_type) pair — figures are saved, never shown
# interactively (see module docstring); run `python bg_gallery_enrichment.py` for a full sweep.
metrics_rows: list[dict] = []
for part_type, instance_type in all_pairs():
    result = run_pair(part_type, instance_type, REF_NUMBER, QUERY_NUMBER)
    if result is None:
        continue
    metrics_rows.extend(result["metrics_rows"])

if not metrics_rows:
    raise RuntimeError(
        "No results — every (part_type, instance_type) pair was skipped "
        f"(RUN_PART_TYPES={RUN_PART_TYPES}, RUN_INSTANCE_TYPES={RUN_INSTANCE_TYPES}; "
        "see warnings above)."
    )
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
log.info(
    "wrote %s (%d rows)\n%s",
    OUTPUT_DIR / "metrics.csv",
    len(metrics_df),
    metrics_df.to_string(index=False),
)

summary_df = (
    metrics_df.groupby(["family", "n"])[["precision", "recall", "f1", "mean_iou", "count_error"]]
    .mean()
    .reset_index()
)
summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
log.info("Summary (mean across every pair):\n%s", summary_df.to_string(index=False))

# %% [markdown]
# ## Metric vs. n — does a richer bg gallery help?

# %% Visualisation — precision/recall/F1/mIoU vs n, one line per family, mean over every pair
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharex=True)
for ax, metric in zip(axes, ["precision", "recall", "f1", "mean_iou"]):
    for family, color in [("fg-bg-proto", "#2ecc71"), ("fg-bg-knn", "#e74c3c")]:
        sub = summary_df[summary_df["family"] == family].sort_values("n")
        ax.plot(sub["n"], sub[metric], marker="o", label=family, color=color)
    ax.set_title(metric)
    ax.set_xlabel("n extra bg crops (per scale)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("score")
axes[0].legend()
fig.suptitle("fg-bg-proto / fg-bg-knn vs. background-gallery enrichment (n=0 is the baseline)")
fig.tight_layout()
out_path = OUTPUT_DIR / "metrics_vs_n.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("saved %s", out_path)

# %% Visualisation — per-pair mean IoU vs n (does enrichment help consistently across pairs?)
metrics_df["pair"] = metrics_df["part_type"] + " / " + metrics_df["instance_type"]
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True, sharey=True)
for ax, family in zip(axes, ["fg-bg-proto", "fg-bg-knn"]):
    for pair, sub in metrics_df[metrics_df["family"] == family].groupby("pair"):
        sub = sub.sort_values("n")
        ax.plot(sub["n"], sub["mean_iou"], marker=".", alpha=0.7, label=pair)
    ax.set_title(family)
    ax.set_xlabel("n extra bg crops (per scale)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("mean IoU")
axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
fig.suptitle("Per-pair mean IoU vs. background-gallery enrichment")
fig.tight_layout()
out_path = OUTPUT_DIR / "metrics_vs_n_by_pair.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("saved %s", out_path)
