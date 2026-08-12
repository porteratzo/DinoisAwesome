# %% [markdown]
# # Density Map Generation — Method Comparison
#
# Compares several ways of turning one exemplar image + its instance mask into a
# per-patch "density" score for a query image, all built on the same DINOv3
# patch-token extraction:
#
#   - **mean**       — single L2-normalised mean descriptor (existing baseline)
#   - **kmeans**     — K centroids over masked exemplar tokens, max-sim aggregation
#   - **knn**        — raw exemplar tokens kept as a memory bank (PatchCore-style),
#                      score = mean of top-K nearest-neighbour similarities
#   - **whitening**  — PCA-whitening fit on the exemplar's own tokens, then mean
#                      descriptor + cosine sim in whitened space
#   - **mlp**        — small MLP classifier trained on masked (fg) vs. unmasked
#                      (bg) exemplar tokens; score = P(foreground)
#
# For every method's raw score map:
#   1. A threshold is derived from the *query's own* GT true/false patch-score
#      split via **median_minus_margin** (median(true) - margin).
#   2. Patches above threshold are clustered spatially (xy only) with **HDBSCAN**,
#      sized via **minmax_margin** cluster-size bounds derived from GT instance
#      patch-areas.
#   3. GT instances are *also* re-derived by running **DBSCAN** on the (unioned)
#      GT patch mask, so predictions and ground truth are clustered the same way —
#      this bounds the comparison by what patch-grid clustering can achieve at all,
#      rather than comparing against idealised per-instance GT masks.
#   4. Predicted vs. GT clusters are greedily matched by IoU → precision, recall,
#      F1, mean matched IoU, and instance count-error, aggregated across four
#      exemplar/query pairs (LHa, LHb, RHa, RHb).

# %% Logging — must be before torch import
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from matplotlib.patches import Rectangle
from PIL import Image
from scipy import ndimage
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neural_network import MLPClassifier

from dinoisawesome import DinoEncoder
from dinoisawesome.annotation_utils import load_annotations
from dinoisawesome.instance_detection import compute_exemplar_features, extract_patch_tokens

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("density_map_methods")

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

data_dir = _REPO_ROOT / "data" / "abc3"

PART_TYPES = ["LHa", "LHb", "RHa", "RHb"]
REF_NUMBER = 1
QUERY_NUMBER = 2
FOCUS_PART_TYPE = "LHb"  # part type used for the detailed per-method figure
# foam -- donut foam single -- donut foam -- velcro -- white clips
# EXEMPLAR_CLASS: list[str] | None = ["donut foam single", "donut foam"]
EXEMPLAR_CLASS: list[str] | None = ["velcro"]  # None = all classes

DINO_VERSION = "v3"
DINO_SIZE = "large"
IMG_SIZE = 768
LAYER_IDX = 23
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")

MASK_PATCH_THRESHOLD = 0.3

EXEMPLAR_K = 3  # kmeans centroids
KNN_BANK_MAX = 300  # memory-bank subsample cap
KNN_TOPK = 5  # top-K neighbour similarities averaged
KNN_BG_BANK_MAX = 3000  # background-token subsample cap for the knn_bg memory bank
WHITEN_DIM = 64  # PCA-whitened dimensionality
WHITEN_FIT_SAMPLES = 3000  # subsample cap for the whitening SVD fit
MLP_HIDDEN: tuple[int, ...] = (16,)
MLP_NEG_MAX = 3000  # background-token subsample cap for MLP training

THRESHOLD_METHOD: Literal["median_minus_margin", "ref_iou"] = "ref_iou"
DENSITY_THRESHOLD_MARGIN = 0.05  # median_minus_margin
REF_THRESHOLD_STEPS = 50  # ref_iou: number of candidate thresholds searched

# minmax_margin: int -> flat patch-count offset (k - margin, k + margin);
# float -> fraction of each bound (k - margin*k, k + margin*k).
CLUSTER_SIZE_MARGIN: int | float = 0.2
GT_DBSCAN_EPS = 1.5  # patch units — ~8-connectivity on the grid
GT_DBSCAN_MIN_SAMPLES = 1  # every point is core -> connected components
IOU_MATCH_THRESHOLD = 0.3
HDBSCAN_MIN_POINTS_FLOOR = 2

# Two-stage ROI refinement: find coarse ROI blobs in the full-image raw score map,
# re-encode each blob's (padded) crop at native resolution, and splat the finer
# per-crop scores back onto the coarse grid before thresholding/clustering. The
# rest of the pipeline (threshold, HDBSCAN, match_and_score) is unchanged.
TWO_STAGE_ENABLED = True
ROI_BINARIZE_METHOD: Literal["otsu", "single"] = "otsu"
ROI_SINGLE_THRESHOLD = 0.5  # used when ROI_BINARIZE_METHOD == "single"
BLOB_METHOD: Literal["connected_components", "hdbscan"] = "connected_components"
CROP_PADDING_FRACTION = 0.15  # 15% of the blob bbox's own width/height, per side
MAX_UPSCALE_FACTOR = 2.0  # crop native-pixel side floor = IMG_SIZE / MAX_UPSCALE_FACTOR

SEED = 0
torch.manual_seed(SEED)

ALL_METHOD_IDS = ["mean", "kmeans", "knn", "knn_bg", "whitening", "mlp"]
SELECTED_METHODS: list[str] | None = [
    "mean",
    "kmeans",
    "knn",
]  # None = all; e.g. ["mean", "knn", "mlp"]


def resolve_selected_methods(selected: list[str] | None) -> list[str]:
    if selected is None:
        return list(ALL_METHOD_IDS)
    unknown = set(selected) - set(ALL_METHOD_IDS)
    if unknown:
        raise ValueError(f"Unknown method id(s) {sorted(unknown)} — choose from {ALL_METHOD_IDS}")
    return [m for m in ALL_METHOD_IDS if m in selected]


def method_display_name(method_id: str) -> str:
    return f"kmeans_k{EXEMPLAR_K}" if method_id == "kmeans" else method_id


SELECTED = resolve_selected_methods(SELECTED_METHODS)
METHOD_ORDER = [method_display_name(m) for m in SELECTED]

log.info(
    "Part types: %s  |  ref_number=%d  query_number=%d  |  methods=%s",
    PART_TYPES,
    REF_NUMBER,
    QUERY_NUMBER,
    METHOD_ORDER,
)

# %% Mask helpers


def load_instance_pixel_mask(stem: str, class_filter: list[str] | None) -> np.ndarray | None:
    """Union of instance masks matching *class_filter* for annotation *stem*, or None."""
    ann_stem = data_dir / "annotations" / stem
    try:
        anns = load_annotations(ann_stem)
    except FileNotFoundError:
        log.warning("Mask not found at %s", ann_stem)
        return None
    instances = [a for a in anns if class_filter is None or a["class"] in class_filter]
    if not instances:
        log.warning("No instances matched classes %s at %s", class_filter, ann_stem)
        return None
    return np.stack([a["mask"] for a in instances]).any(axis=0)


def pixel_mask_to_patch_mask(
    pixel_mask: np.ndarray,
    grid_h: int,
    grid_w: int,
    img_size: int,
    threshold: float = 0.3,
) -> np.ndarray:
    """Resize a pixel-space mask to patch-grid resolution, (grid_h, grid_w) bool."""
    mask_pil = Image.fromarray(pixel_mask.astype(np.uint8) * 255)
    mask_resized = np.array(mask_pil.resize((img_size, img_size), Image.NEAREST)) > 0
    ph = img_size // grid_h
    pw = img_size // grid_w
    tiled = mask_resized.reshape(grid_h, ph, grid_w, pw)
    patch_density = tiled.mean(axis=(1, 3))
    return patch_density >= threshold


def gt_instance_patch_sizes(
    stem: str,
    class_filter: list[str] | None,
    grid_h: int,
    grid_w: int,
    img_size: int,
    patch_threshold: float,
) -> np.ndarray:
    """Per-instance GT sizes in patches, using that image's own annotation set."""
    ann_stem = data_dir / "annotations" / stem
    try:
        anns = load_annotations(ann_stem)
    except FileNotFoundError:
        return np.array([])
    instances = [a for a in anns if class_filter is None or a["class"] in class_filter]
    if not instances:
        return np.array([])
    return np.array(
        [
            pixel_mask_to_patch_mask(a["mask"], grid_h, grid_w, img_size, patch_threshold).sum()
            for a in instances
        ]
    )


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / (union + 1e-8)


# %% Descriptor-method registry
#
# Every method reduces to two functions: fit(exemplar tokens) -> MethodState, and
# score(state, query_tokens) -> raw (H*W,) score array, so the rest of the pipeline
# (thresholding, clustering, matching) is written once and reused for all five.


@dataclass
class MethodState:
    name: str
    kind: Literal["single", "multi", "bank", "bank_bg", "whiten", "clf"]
    payload: Any


def fit_mean(ex_masked: torch.Tensor) -> MethodState:
    proto = compute_exemplar_features(ex_masked, mode="mean")  # (1, C)
    return MethodState("mean", "single", proto)


def fit_kmeans(ex_masked: torch.Tensor, k: int) -> MethodState:
    protos = compute_exemplar_features(ex_masked, mode="kmeans", k=k)  # (K, C)
    return MethodState(f"kmeans_k{k}", "multi", protos)


def fit_knn_bank(ex_masked: torch.Tensor, max_bank: int, seed: int = SEED) -> MethodState:
    n = ex_masked.shape[0]
    if n > max_bank:
        gen = torch.Generator(device=ex_masked.device).manual_seed(seed)
        idx = torch.randperm(n, generator=gen, device=ex_masked.device)[:max_bank]
        bank = ex_masked[idx]
    else:
        bank = ex_masked
    return MethodState("knn", "bank", bank)


def fit_knn_bank_bg(
    ex_masked: torch.Tensor,
    ex_bg: torch.Tensor,
    max_bank_fg: int,
    max_bank_bg: int,
    seed: int = SEED,
) -> MethodState:
    """PatchCore-style bank over BOTH fg and bg exemplar tokens.

    Nearest-neighbour hits that land on a bg bank entry are dropped before
    averaging (see the "bank_bg" branch of score_method), so a bg token can
    "steal" a top-K slot away from a weak fg match instead of silently
    inflating the plain knn score.
    """

    def subsample(t: torch.Tensor, cap: int, salt: int) -> torch.Tensor:
        n = t.shape[0]
        if n <= cap:
            return t
        gen = torch.Generator(device=t.device).manual_seed(seed + salt)
        idx = torch.randperm(n, generator=gen, device=t.device)[:cap]
        return t[idx]

    fg_bank = subsample(ex_masked, max_bank_fg, salt=0)
    bg_bank = subsample(ex_bg, max_bank_bg, salt=1)
    bank = torch.cat([fg_bank, bg_bank], dim=0)
    is_fg = torch.zeros(bank.shape[0], dtype=torch.bool, device=bank.device)
    is_fg[: fg_bank.shape[0]] = True
    return MethodState("knn_bg", "bank_bg", {"bank": bank, "is_fg": is_fg})


def fit_whitening(
    ex_tokens: torch.Tensor,
    ex_masked: torch.Tensor,
    dim: int,
    fit_samples: int,
    eps: float = 1e-5,
    seed: int = SEED,
) -> MethodState:
    n = ex_tokens.shape[0]
    if n > fit_samples:
        gen = torch.Generator(device=ex_tokens.device).manual_seed(seed)
        idx = torch.randperm(n, generator=gen, device=ex_tokens.device)[:fit_samples]
        fit_tokens = ex_tokens[idx]
    else:
        fit_tokens = ex_tokens

    x = fit_tokens.cpu().float().numpy()
    mean = x.mean(axis=0)
    xc = x - mean
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    d = min(dim, vt.shape[0])
    components = vt[:d]  # (d, C)
    eigvals = (s[:d] ** 2) / max(x.shape[0] - 1, 1)
    scale = 1.0 / np.sqrt(eigvals + eps)

    def transform(t: torch.Tensor) -> torch.Tensor:
        arr = t.detach().cpu().float().numpy()
        proj = (arr - mean) @ components.T * scale
        out = torch.from_numpy(proj.astype(np.float32)).to(t.device)
        return F.normalize(out, p=2, dim=-1)

    ex_masked_w = transform(ex_masked)
    proto = F.normalize(ex_masked_w.mean(dim=0, keepdim=True), p=2, dim=-1)
    return MethodState("whitening", "whiten", {"transform": transform, "proto": proto})


def fit_mlp(
    ex_masked: torch.Tensor,
    ex_bg: torch.Tensor,
    hidden: tuple[int, ...],
    neg_max: int,
    seed: int = SEED,
) -> MethodState:
    pos = ex_masked.cpu().float().numpy()
    neg_all = ex_bg.cpu().float().numpy()
    if len(neg_all) > neg_max:
        rng = np.random.default_rng(seed)
        neg = neg_all[rng.choice(len(neg_all), neg_max, replace=False)]
    else:
        neg = neg_all
    X = np.concatenate([pos, neg], axis=0)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    clf = MLPClassifier(
        hidden_layer_sizes=hidden, max_iter=500, random_state=seed, early_stopping=True
    )
    clf.fit(X, y)
    return MethodState("mlp", "clf", clf)


def build_methods(
    ex_tokens: torch.Tensor,
    ex_masked: torch.Tensor,
    ex_bg: torch.Tensor,
    selected: list[str],
) -> dict[str, MethodState]:
    """Fit only the methods in *selected* (see ALL_METHOD_IDS / SELECTED_METHODS)."""
    methods: dict[str, MethodState] = {}
    if "mean" in selected:
        methods["mean"] = fit_mean(ex_masked)
    if "kmeans" in selected:
        methods[f"kmeans_k{EXEMPLAR_K}"] = fit_kmeans(ex_masked, EXEMPLAR_K)
    if "knn" in selected:
        methods["knn"] = fit_knn_bank(ex_masked, KNN_BANK_MAX)
    if "whitening" in selected:
        methods["whitening"] = fit_whitening(ex_tokens, ex_masked, WHITEN_DIM, WHITEN_FIT_SAMPLES)

    needs_bg = "knn_bg" in selected or "mlp" in selected
    if needs_bg and len(ex_bg) == 0:
        log.warning("No background tokens available — skipping knn_bg and mlp methods.")
    elif needs_bg:
        if "knn_bg" in selected:
            methods["knn_bg"] = fit_knn_bank_bg(ex_masked, ex_bg, KNN_BANK_MAX, KNN_BG_BANK_MAX)
        if "mlp" in selected:
            methods["mlp"] = fit_mlp(ex_masked, ex_bg, MLP_HIDDEN, MLP_NEG_MAX)
    return methods


def score_method(state: MethodState, query_tokens: torch.Tensor) -> np.ndarray:
    if state.kind == "single":
        sim = (query_tokens @ state.payload.T).squeeze(-1)
        return sim.cpu().float().numpy()
    if state.kind == "multi":
        sim = query_tokens @ state.payload.T
        return sim.max(dim=-1).values.cpu().float().numpy()
    if state.kind == "bank":
        sim = query_tokens @ state.payload.T
        topk = min(KNN_TOPK, sim.shape[1])
        vals, _ = sim.topk(topk, dim=-1)
        return vals.mean(dim=-1).cpu().float().numpy()
    if state.kind == "bank_bg":
        bank = state.payload["bank"]
        is_fg = state.payload["is_fg"]
        sim = query_tokens @ bank.T
        topk = min(KNN_TOPK, sim.shape[1])
        vals, idx = sim.topk(topk, dim=-1)
        fg_hits = is_fg[idx]
        vals_fg = vals.masked_fill(~fg_hits, 0.0)
        fg_count = fg_hits.sum(dim=-1).clamp(min=1)
        score = vals_fg.sum(dim=-1) / fg_count
        return score.cpu().float().numpy()
    if state.kind == "whiten":
        qw = state.payload["transform"](query_tokens)
        sim = (qw @ state.payload["proto"].T).squeeze(-1)
        return sim.cpu().float().numpy()
    if state.kind == "clf":
        x = query_tokens.detach().cpu().float().numpy()
        return state.payload.predict_proba(x)[:, 1]
    raise ValueError(f"Unknown method kind: {state.kind}")


def median_minus_margin_threshold(true_vals: np.ndarray, margin: float) -> float:
    return float(np.median(true_vals) - margin)


def iou_threshold_curve(
    raw: np.ndarray, gt_mask: np.ndarray, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Patch-mask IoU of `(raw > c)` vs. `gt_mask` for `steps` candidates spanning raw's range."""
    candidates = np.linspace(raw.min(), raw.max(), steps)
    ious = np.array([mask_iou(raw > c, gt_mask) for c in candidates])
    return candidates, ious


def iou_tuned_threshold(raw: np.ndarray, gt_mask: np.ndarray, steps: int) -> float:
    """Threshold that maximises patch-mask IoU against *gt_mask*.

    Sweeps `steps` candidates spanning the raw score range and keeps the one
    with the highest IoU between the thresholded mask and GT — meant to be fit
    on the exemplar's own score map + GT mask (self-supervised, no dependence
    on the query's GT), then reused as-is to threshold queries.
    """
    candidates, ious = iou_threshold_curve(raw, gt_mask, steps)
    best_idx = int(np.argmax(ious))
    return float(candidates[best_idx])


def minmax_margin_cluster_bounds(sizes: np.ndarray, margin: int | float) -> tuple[int, int]:
    """Cluster-size bounds from GT instance sizes ± margin.

    ``margin`` as an int is a flat patch-count offset: k - margin, k + margin.
    As a float it's read as a fraction of each bound: k - margin*k, k + margin*k.
    """
    if len(sizes) == 0:
        return HDBSCAN_MIN_POINTS_FLOOR, HDBSCAN_MIN_POINTS_FLOOR * 4
    k_min, k_max = float(sizes.min()), float(sizes.max())
    if isinstance(margin, float):
        lo, hi = k_min - margin * k_min, k_max + margin * k_max
    else:
        lo, hi = k_min - margin, k_max + margin
    min_cs = max(HDBSCAN_MIN_POINTS_FLOOR, int(round(lo)))
    max_cs = max(min_cs, int(round(hi)))
    return min_cs, max_cs


def hdbscan_clusters(
    xs: np.ndarray,
    ys: np.ndarray,
    grid_h: int,
    grid_w: int,
    raw: np.ndarray,
    min_cs: int | None = None,
    max_cs: int | None = None,
) -> list[dict]:
    """Cluster foreground patch coords with HDBSCAN.

    min_cs/max_cs=None means no size bound is passed to HDBSCAN — used for the
    "unrestricted" fallback pass that just wants to see what's out there.
    """
    xs_norm = xs / max(grid_w - 1, 1)
    ys_norm = ys / max(grid_h - 1, 1)
    features = np.stack([xs_norm, ys_norm], axis=1)
    # HDBSCAN defaults min_cluster_size (and thus min_samples) to 5 when unset,
    # which errors out if fewer than 5 points are passed in. Cap it to the
    # number of available points so the "unrestricted" fallback pass never
    # requests more samples than it has.
    effective_min_cs = min_cs if min_cs is not None else 2
    effective_min_cs = max(2, min(effective_min_cs, len(xs)))
    size_kwargs: dict[str, int] = {"min_cluster_size": effective_min_cs}
    if max_cs is not None:
        size_kwargs["max_cluster_size"] = max_cs
    labels = HDBSCAN(copy=False, **size_kwargs).fit_predict(features)
    clusters = []
    for lbl in sorted(set(labels)):
        if lbl == -1:
            continue
        sel = labels == lbl
        mask = np.zeros((grid_h, grid_w), dtype=bool)
        mask[ys[sel], xs[sel]] = True
        clusters.append({"mask": mask, "score": float(raw[ys[sel], xs[sel]].max())})
    return clusters


def dbscan_clusters_from_mask(patch_mask: np.ndarray, eps: float, min_samples: int) -> list[dict]:
    ys, xs = np.where(patch_mask)
    if len(xs) == 0:
        return []
    coords = np.stack([xs, ys], axis=1).astype(float)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    clusters = []
    for lbl in sorted(set(labels)):
        if lbl == -1:
            continue
        sel = labels == lbl
        mask = np.zeros(patch_mask.shape, dtype=bool)
        mask[ys[sel], xs[sel]] = True
        clusters.append({"mask": mask})
    return clusters


# %% Two-stage ROI refinement — binarize, blob, crop, re-encode, splat back


def otsu_threshold(raw: np.ndarray) -> float:
    """Otsu's threshold on *raw*, computed via cv2 on a rescaled 8-bit copy."""
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo < 1e-8:
        return lo
    scaled = ((raw - lo) / (hi - lo) * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return lo + (float(otsu_val) / 255.0) * (hi - lo)


def roi_binary_mask(raw: np.ndarray, method: str, single_threshold: float) -> np.ndarray:
    """Unsupervised (no-GT) foreground/ROI mask used to decide where to zoom in."""
    if method == "otsu":
        thr = otsu_threshold(raw)
    elif method == "single":
        thr = single_threshold
    else:
        raise ValueError(f"Unknown ROI_BINARIZE_METHOD: {method!r}")
    return raw > thr


def connected_component_blobs(mask: np.ndarray) -> list[dict]:
    """8-connected components of a boolean patch mask -> list of {'mask': ...}."""
    structure = ndimage.generate_binary_structure(2, 2)
    labeled, n = ndimage.label(mask, structure=structure)
    return [{"mask": labeled == lbl} for lbl in range(1, n + 1)]


def blob_patch_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Half-open patch-grid bbox (y0, y1, x0, x1) of a boolean blob mask."""
    ys, xs = np.where(mask)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def patch_bbox_to_native_px(
    y0: int, y1: int, x0: int, x1: int, patch_size: int, scale_x: float, scale_y: float
) -> tuple[float, float, float, float]:
    """Patch-grid bbox -> native (original, full-resolution image) pixel bbox."""
    px0 = x0 * patch_size * scale_x
    px1 = x1 * patch_size * scale_x
    py0 = y0 * patch_size * scale_y
    py1 = y1 * patch_size * scale_y
    return px0, py0, px1, py1


def pad_and_floor_crop_box(
    px0: float,
    py0: float,
    px1: float,
    py1: float,
    pad_fraction: float,
    floor_side_px: float,
    native_w: int,
    native_h: int,
) -> tuple[int, int, int, int]:
    """Pad a native-pixel bbox by *pad_fraction* per side (of its own width/height),
    then enforce a centred minimum side length of *floor_side_px* — this bounds how
    much the encoder's resize-to-IMG_SIZE can upscale the crop — then clip to bounds.
    """
    w, h = px1 - px0, py1 - py0
    pad_w, pad_h = w * pad_fraction, h * pad_fraction
    px0, px1 = px0 - pad_w, px1 + pad_w
    py0, py1 = py0 - pad_h, py1 + pad_h

    w, h = px1 - px0, py1 - py0
    if w < floor_side_px:
        cx = (px0 + px1) / 2
        px0, px1 = cx - floor_side_px / 2, cx + floor_side_px / 2
    if h < floor_side_px:
        cy = (py0 + py1) / 2
        py0, py1 = cy - floor_side_px / 2, cy + floor_side_px / 2

    px0_i = max(0, int(round(px0)))
    py0_i = max(0, int(round(py0)))
    px1_i = min(native_w, int(round(px1)))
    py1_i = min(native_h, int(round(py1)))
    return px0_i, py0_i, px1_i, py1_i


def refine_raw_with_crops(
    state: MethodState,
    raw: np.ndarray,
    query_img: Image.Image,
    grid_h: int,
    grid_w: int,
) -> tuple[np.ndarray, dict]:
    """Stage 2: find coarse ROI blobs in *raw*, re-encode each blob's padded native
    crop, and max-splat the finer per-crop scores back onto the coarse grid.

    Returns ``(refined, debug)``: *refined* is a (grid_h, grid_w) raw map the caller
    thresholds/clusters exactly as it would the original *raw* map, so no downstream
    code needs to change; *debug* carries the intermediate ROI mask and per-blob
    bbox/crop/raw_crop info needed for the stage-2 diagnostic plots.
    """
    native_w, native_h = query_img.size
    scale_x = native_w / IMG_SIZE
    scale_y = native_h / IMG_SIZE

    roi_mask = roi_binary_mask(raw, ROI_BINARIZE_METHOD, ROI_SINGLE_THRESHOLD)
    if BLOB_METHOD == "connected_components":
        blobs = connected_component_blobs(roi_mask)
    elif BLOB_METHOD == "hdbscan":
        ys, xs = np.where(roi_mask)
        blobs = (
            hdbscan_clusters(xs, ys, grid_h, grid_w, raw)
            if len(xs) >= HDBSCAN_MIN_POINTS_FLOOR
            else []
        )
    else:
        raise ValueError(f"Unknown BLOB_METHOD: {BLOB_METHOD!r}")

    refined = raw.copy()
    floor_side_px = IMG_SIZE / MAX_UPSCALE_FACTOR
    blob_debug: list[dict] = []

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
            CROP_PADDING_FRACTION,
            floor_side_px,
            native_w,
            native_h,
        )
        if px1 - px0 < 2 or py1 - py0 < 2:
            continue

        crop_img = query_img.crop((px0, py0, px1, py1))
        crop_tokens, c_h, c_w = extract_patch_tokens(encoder, crop_img, LAYER_IDX, debias=True)
        raw_crop = score_method(state, crop_tokens).reshape(c_h, c_w)

        crop_w_native, crop_h_native = px1 - px0, py1 - py0
        centers_x = px0 + (np.arange(c_w) + 0.5) * crop_w_native / c_w
        centers_y = py0 + (np.arange(c_h) + 0.5) * crop_h_native / c_h
        coarse_j = np.clip((centers_x / (native_w / grid_w)).astype(int), 0, grid_w - 1)
        coarse_i = np.clip((centers_y / (native_h / grid_h)).astype(int), 0, grid_h - 1)

        flat_idx = (coarse_i[:, None] * grid_w + coarse_j[None, :]).ravel()
        refined_flat = refined.reshape(-1)
        np.maximum.at(refined_flat, flat_idx, raw_crop.ravel())

        blob_debug.append(
            {
                "patch_bbox": (y0, y1, x0, x1),
                "raw_px_bbox": (raw_px0, raw_py0, raw_px1, raw_py1),
                "crop_px_bbox": (px0, py0, px1, py1),
                "crop_img": crop_img,
                "raw_crop": raw_crop,
            }
        )

    debug = {"raw_stage1": raw, "roi_mask": roi_mask, "blobs": blob_debug}
    return refined, debug


def match_and_score(pred_clusters: list[dict], gt_clusters: list[dict], iou_thr: float) -> dict:
    n_pred, n_gt = len(pred_clusters), len(gt_clusters)
    if n_gt == 0:
        return {
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "mean_iou": np.nan,
            "tp": 0,
            "fp": n_pred,
            "fn": 0,
            "count_error": n_pred,
            "n_pred": n_pred,
            "n_gt": 0,
        }
    order = sorted(range(n_pred), key=lambda i: -pred_clusters[i]["score"])
    matched_gt: set[int] = set()
    tp = 0
    ious: list[float] = []
    for i in order:
        best_j, best_iou = -1, 0.0
        for j in range(n_gt):
            if j in matched_gt:
                continue
            iou = mask_iou(pred_clusters[i]["mask"], gt_clusters[j]["mask"])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thr:
            matched_gt.add(best_j)
            tp += 1
            ious.append(best_iou)
    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "count_error": abs(n_pred - n_gt),
        "n_pred": n_pred,
        "n_gt": n_gt,
    }


# %% Load encoder (shared across all pairs)
encoder = DinoEncoder(
    version=DINO_VERSION,
    size=DINO_SIZE,
    img_size=IMG_SIZE,
    weights_dir=DINO_WEIGHTS_DIR,
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

    # Check exemplar-class availability before running the (expensive) encoder —
    # not every part type has instances of every EXEMPLAR_CLASS.
    ex_pixel_mask = load_instance_pixel_mask(ref_stem, EXEMPLAR_CLASS)
    if ex_pixel_mask is None:
        log.warning(
            "[%s] no exemplar instances for classes %s — skipping part type.",
            part_type,
            EXEMPLAR_CLASS,
        )
        return None

    ref_img = Image.open(data_dir / f"{ref_stem}.jpg").convert("RGB")
    query_img = Image.open(data_dir / f"{query_stem}.jpg").convert("RGB")

    ex_tokens, ex_h, ex_w = extract_patch_tokens(encoder, ref_img, LAYER_IDX, debias=True)
    q_tokens, q_h, q_w = extract_patch_tokens(encoder, query_img, LAYER_IDX, debias=True)

    ex_patch_mask = pixel_mask_to_patch_mask(
        ex_pixel_mask, ex_h, ex_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )
    ex_patch_flat = torch.from_numpy(ex_patch_mask.reshape(-1)).to(ex_tokens.device)
    ex_masked = ex_tokens[ex_patch_flat]
    ex_bg = ex_tokens[~ex_patch_flat]
    log.info("[%s] exemplar patches: %d fg / %d bg", part_type, ex_masked.shape[0], ex_bg.shape[0])

    methods = build_methods(ex_tokens, ex_masked, ex_bg, SELECTED)

    q_pixel_mask = load_instance_pixel_mask(query_stem, EXEMPLAR_CLASS)
    gt_patch_mask = (
        pixel_mask_to_patch_mask(q_pixel_mask, q_h, q_w, IMG_SIZE, MASK_PATCH_THRESHOLD)
        if q_pixel_mask is not None
        else np.zeros((q_h, q_w), dtype=bool)
    )
    gt_sizes = gt_instance_patch_sizes(
        query_stem, EXEMPLAR_CLASS, q_h, q_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )
    min_cs, max_cs = minmax_margin_cluster_bounds(gt_sizes, CLUSTER_SIZE_MARGIN)
    gt_clusters = dbscan_clusters_from_mask(gt_patch_mask, GT_DBSCAN_EPS, GT_DBSCAN_MIN_SAMPLES)
    log.info(
        "[%s] GT instances=%d sizes=%s -> cluster bounds=[%d,%d] -> GT-DBSCAN clusters=%d",
        part_type,
        len(gt_sizes),
        gt_sizes.tolist(),
        min_cs,
        max_cs,
        len(gt_clusters),
    )

    per_method: dict[str, dict] = {}
    for name, state in methods.items():
        raw = score_method(state, q_tokens).reshape(q_h, q_w)
        stage2_debug: dict | None = None
        if TWO_STAGE_ENABLED:
            raw, stage2_debug = refine_raw_with_crops(state, raw, query_img, q_h, q_w)
            for blob in stage2_debug["blobs"]:
                px0, py0, px1, py1 = blob["crop_px_bbox"]
                c_h, c_w = blob["raw_crop"].shape
                crop_pixel_mask = (
                    q_pixel_mask[py0:py1, px0:px1] if q_pixel_mask is not None else None
                )
                blob["crop_gt_patch_mask"] = (
                    pixel_mask_to_patch_mask(
                        crop_pixel_mask, c_h, c_w, IMG_SIZE, MASK_PATCH_THRESHOLD
                    )
                    if crop_pixel_mask is not None and crop_pixel_mask.any()
                    else np.zeros((c_h, c_w), dtype=bool)
                )
        true_vals = raw[gt_patch_mask]
        if len(true_vals) == 0:
            log.warning("[%s/%s] no GT-true patches — skipping.", part_type, name)
            continue
        if THRESHOLD_METHOD == "ref_iou":
            ref_raw = score_method(state, ex_tokens).reshape(ex_h, ex_w)
            thr = iou_tuned_threshold(ref_raw, ex_patch_mask, REF_THRESHOLD_STEPS)
        else:
            thr = median_minus_margin_threshold(true_vals, DENSITY_THRESHOLD_MARGIN)
        binary = raw > thr
        ys, xs = np.where(binary)
        if len(xs) < max(HDBSCAN_MIN_POINTS_FLOOR, min_cs):
            pred_clusters: list[dict] = []
        else:
            pred_clusters = hdbscan_clusters(xs, ys, q_h, q_w, raw, min_cs, max_cs)

        # If size-bounded HDBSCAN found nothing, rerun unrestricted (no size bounds)
        # purely so the detailed view has something to show — these don't count as
        # predictions and are excluded from match_and_score.
        rejected_clusters: list[dict] = []
        if not pred_clusters and len(xs) >= HDBSCAN_MIN_POINTS_FLOOR:
            rejected_clusters = hdbscan_clusters(xs, ys, q_h, q_w, raw)
            if rejected_clusters:
                log.info(
                    "[%s/%s] size-bounded HDBSCAN found nothing (bounds=[%d,%d]) — "
                    "%d unrestricted cluster(s) shown as rejected.",
                    part_type,
                    name,
                    min_cs,
                    max_cs,
                    len(rejected_clusters),
                )

        metrics = match_and_score(pred_clusters, gt_clusters, IOU_MATCH_THRESHOLD)
        log.info(
            "[%s/%s] thr=%.3f  pred_clusters=%d  P=%.2f R=%.2f F1=%.2f mIoU=%.2f",
            part_type,
            name,
            thr,
            len(pred_clusters),
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
            "rejected_clusters": rejected_clusters,
            "metrics": metrics,
            "stage2": stage2_debug,
        }

    return {
        "part_type": part_type,
        "ref_img": ref_img,
        "query_img": query_img,
        "ref_pixel_mask": ex_pixel_mask,
        "ex_patch_mask": ex_patch_mask,
        "ex_h": ex_h,
        "ex_w": ex_w,
        "gt_patch_mask": gt_patch_mask,
        "gt_clusters": gt_clusters,
        "gt_sizes": gt_sizes,
        "min_cs": min_cs,
        "max_cs": max_cs,
        "per_method": per_method,
        "q_h": q_h,
        "q_w": q_w,
    }


# %% Visualisation — ref_iou threshold sweep (chosen vs. random under/over thresholds)
if THRESHOLD_METHOD == "ref_iou":
    diag_stem = f"{FOCUS_PART_TYPE}_{REF_NUMBER}"
    diag_img = Image.open(data_dir / f"{diag_stem}.jpg").convert("RGB")
    diag_tokens, diag_h, diag_w = extract_patch_tokens(encoder, diag_img, LAYER_IDX, debias=True)

    diag_pixel_mask = load_instance_pixel_mask(diag_stem, EXEMPLAR_CLASS)
    if diag_pixel_mask is None:
        raise RuntimeError(f"No exemplar mask for {diag_stem} — cannot fit descriptors.")
    diag_patch_mask = pixel_mask_to_patch_mask(
        diag_pixel_mask, diag_h, diag_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )
    diag_patch_flat = torch.from_numpy(diag_patch_mask.reshape(-1)).to(diag_tokens.device)
    diag_masked = diag_tokens[diag_patch_flat]
    diag_bg = diag_tokens[~diag_patch_flat]

    diag_method_id = SELECTED[0]
    diag_method_name = method_display_name(diag_method_id)
    diag_methods = build_methods(diag_tokens, diag_masked, diag_bg, [diag_method_id])
    diag_state = diag_methods[diag_method_name]
    diag_raw = score_method(diag_state, diag_tokens).reshape(diag_h, diag_w)

    diag_candidates, diag_ious = iou_threshold_curve(diag_raw, diag_patch_mask, REF_THRESHOLD_STEPS)
    diag_best_idx = int(np.argmax(diag_ious))

    diag_rng = np.random.default_rng(SEED)
    diag_under_idx = (
        int(diag_rng.integers(0, diag_best_idx)) if diag_best_idx > 0 else diag_best_idx
    )
    diag_over_idx = (
        int(diag_rng.integers(diag_best_idx + 1, len(diag_candidates)))
        if diag_best_idx < len(diag_candidates) - 1
        else diag_best_idx
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(diag_candidates, diag_ious, color="steelblue", label="IoU(threshold)")
    ax.axvline(
        diag_candidates[diag_best_idx],
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"chosen thr={diag_candidates[diag_best_idx]:.3f} "
        f"(iou={diag_ious[diag_best_idx]:.3f})",
    )
    ax.axvline(
        diag_candidates[diag_under_idx],
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label=f"random under-thr={diag_candidates[diag_under_idx]:.3f} "
        f"(iou={diag_ious[diag_under_idx]:.3f})",
    )
    ax.axvline(
        diag_candidates[diag_over_idx],
        color="red",
        linestyle=":",
        linewidth=1.5,
        label=f"random over-thr={diag_candidates[diag_over_idx]:.3f} "
        f"(iou={diag_ious[diag_over_idx]:.3f})",
    )
    ax.set_xlabel("candidate threshold")
    ax.set_ylabel("patch-mask IoU vs. exemplar GT mask")
    ax.set_title(
        f"ref_iou threshold sweep — [{FOCUS_PART_TYPE}] {diag_method_name} "
        f"(fit on exemplar {diag_stem} against its own GT mask)"
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

# %% Run across all part types, collect metrics
results: dict[str, dict] = {}
rows: list[dict] = []
for pt in PART_TYPES:
    result = run_pair(pt, REF_NUMBER, QUERY_NUMBER)
    if result is None:
        continue
    results[pt] = result
    for method_name, m in result["per_method"].items():
        rows.append({"part_type": pt, "method": method_name, **m["metrics"]})

metrics_df = pd.DataFrame(rows)
log.info("\n%s", metrics_df.to_string(index=False))

summary_df = (
    metrics_df.groupby("method")[["precision", "recall", "f1", "mean_iou", "count_error"]]
    .mean()
    .reindex(METHOD_ORDER)
)
log.info("Summary (mean across part types):\n%s", summary_df.to_string())

# %% Visualisation — summary comparison across methods
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
summary_df[["precision", "recall", "f1"]].plot.bar(ax=axes[0])
axes[0].set_ylim(0, 1.05)
axes[0].set_title("Precision / Recall / F1 (mean across part types)")
axes[0].tick_params(axis="x", rotation=30)

summary_df["mean_iou"].plot.bar(ax=axes[1], color="teal")
axes[1].set_ylim(0, 1.0)
axes[1].set_title("Mean matched IoU (mean across part types)")
axes[1].tick_params(axis="x", rotation=30)

plt.suptitle(
    f"Density-map method comparison  |  {len(PART_TYPES)} part types  |  "
    f"thr=median_minus_margin  bounds=minmax_margin  iou_match={IOU_MATCH_THRESHOLD}",
    fontsize=11,
)
plt.tight_layout()
plt.show()

# %% Visualisation — reference exemplar + its instance mask (precedes the detailed view)
if FOCUS_PART_TYPE not in results:
    raise RuntimeError(
        f"FOCUS_PART_TYPE={FOCUS_PART_TYPE!r} was skipped (no exemplar instances for "
        f"classes {EXEMPLAR_CLASS}) — pick a part type present in {sorted(results)}."
    )
focus = results[FOCUS_PART_TYPE]
ref_img = focus["ref_img"]
disp_ref = np.array(ref_img)
ref_pixel_mask_disp = (
    np.array(
        Image.fromarray(focus["ref_pixel_mask"].astype(np.uint8) * 255).resize(
            ref_img.size, Image.NEAREST
        )
    )
    > 0
)
ref_grid_img = np.array(ref_img.resize((focus["ex_w"], focus["ex_h"]), Image.BOX))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(disp_ref)
axes[0].set_title(f"[{FOCUS_PART_TYPE}] reference image", fontsize=10)
axes[0].axis("off")

axes[1].imshow(disp_ref)
axes[1].imshow(ref_pixel_mask_disp, cmap="Reds", alpha=0.4)
axes[1].set_title("reference image + instance mask", fontsize=10)
axes[1].axis("off")

axes[2].imshow(ref_grid_img)
axes[2].imshow(focus["ex_patch_mask"], cmap="Reds", alpha=0.5, interpolation="nearest")
axes[2].set_title(
    f"mask overlay @ DINOv3 patch size ({focus['ex_h']}x{focus['ex_w']})", fontsize=10
)
axes[2].axis("off")

plt.suptitle(f"Reference exemplar — part_type={FOCUS_PART_TYPE}", fontsize=12)
plt.tight_layout()
plt.show()

# %% Visualisation — detailed per-method breakdown for one part type
methods_present = [m for m in METHOD_ORDER if m in focus["per_method"]]
disp_q = np.array(focus["query_img"])
scale_x = focus["query_img"].size[0] / IMG_SIZE
scale_y = focus["query_img"].size[1] / IMG_SIZE


def patch_centers_to_px(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (xs + 0.5) * PATCH_SIZE * scale_x, (ys + 0.5) * PATCH_SIZE * scale_y


CMAP = plt.get_cmap("tab10")
n_rows = len(methods_present)
fig, axes = plt.subplots(n_rows, 5, figsize=(24, 5 * n_rows))
if n_rows == 1:
    axes = axes.reshape(1, 5)

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
    axes[row, 1].set_title(f"binary (thr={m['threshold']:.3f}) + GT outline", fontsize=10)
    axes[row, 1].axis("off")

    axes[row, 2].imshow(disp_q)
    for i, cl in enumerate(m["clusters"]):
        ys_c, xs_c = np.where(cl["mask"])
        px_x, px_y = patch_centers_to_px(xs_c, ys_c)
        axes[row, 2].scatter(px_x, px_y, s=14, color=CMAP(i % 10), label=f"c{i}")
    rejected = m.get("rejected_clusters", [])
    for i, cl in enumerate(rejected):
        ys_c, xs_c = np.where(cl["mask"])
        px_x, px_y = patch_centers_to_px(xs_c, ys_c)
        axes[row, 2].scatter(
            px_x,
            px_y,
            s=14,
            marker="x",
            color="0.4",
            label="rejected" if i == 0 else None,
        )
    metrics = m["metrics"]
    title = (
        f"HDBSCAN clusters ({len(m['clusters'])})\n"
        f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} "
        f"F1={metrics['f1']:.2f} mIoU={metrics['mean_iou']:.2f}"
    )
    if rejected:
        title += f"\n{len(rejected)} rejected (outside size bounds, unrestricted HDBSCAN)"
        axes[row, 2].legend(fontsize=7, loc="lower right")
    axes[row, 2].set_title(title, fontsize=9)
    axes[row, 2].axis("off")

    axes[row, 3].imshow(disp_q)
    for i, cl in enumerate(focus["gt_clusters"]):
        ys_c, xs_c = np.where(cl["mask"])
        px_x, px_y = patch_centers_to_px(xs_c, ys_c)
        axes[row, 3].scatter(px_x, px_y, s=14, color=CMAP(i % 10))
    axes[row, 3].set_title(
        f"GT-DBSCAN clusters ({len(focus['gt_clusters'])})\n"
        f"size_bounds=[{focus['min_cs']},{focus['max_cs']}]",
        fontsize=9,
    )
    axes[row, 3].axis("off")

    true_vals = raw_np[focus["gt_patch_mask"]]
    false_vals = raw_np[~focus["gt_patch_mask"]]
    bins = np.linspace(raw_np.min(), raw_np.max(), 40)
    axes[row, 4].hist(false_vals, bins=bins, alpha=0.6, color="tab:gray", label="GT-false")
    axes[row, 4].hist(true_vals, bins=bins, alpha=0.6, color="tab:red", label="GT-true")
    axes[row, 4].axvline(
        m["threshold"], color="black", linestyle="--", linewidth=1.5, label="threshold"
    )
    axes[row, 4].set_title(f"[{name}] score histogram (thr={m['threshold']:.3f})", fontsize=10)
    axes[row, 4].legend(fontsize=8)

plt.suptitle(
    f"Density-map methods — detailed view  |  part_type={FOCUS_PART_TYPE}  block={LAYER_IDX}",
    fontsize=12,
    y=1.005,
)
plt.tight_layout()
plt.show()

# %% Visualisation — two-stage: ROI binarization + blobs + crops used
if not TWO_STAGE_ENABLED:
    log.info("TWO_STAGE_ENABLED is False — skipping stage-2 diagnostic visualisations.")
else:
    stage2_methods = [m for m in methods_present if focus["per_method"][m].get("stage2")]
    if not stage2_methods:
        log.warning("[%s] no stage-2 debug info to plot.", FOCUS_PART_TYPE)

    n_rows = len(stage2_methods)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 3)

    for row, name in enumerate(stage2_methods):
        m = focus["per_method"][name]
        s2 = m["stage2"]
        raw_stage1 = s2["raw_stage1"]
        roi_mask = s2["roi_mask"]
        blobs = s2["blobs"]

        im0 = axes[row, 0].imshow(raw_stage1, cmap="jet", aspect="auto")
        axes[row, 0].set_title(f"[{name}] stage-1 raw score map (full image)", fontsize=10)
        axes[row, 0].axis("off")
        plt.colorbar(im0, ax=axes[row, 0], shrink=0.75, pad=0.02)

        axes[row, 1].imshow(roi_mask, cmap="Greys_r", aspect="auto")
        roi_thr_label = (
            f"otsu={otsu_threshold(raw_stage1):.3f}"
            if ROI_BINARIZE_METHOD == "otsu"
            else f"single={ROI_SINGLE_THRESHOLD:.3f}"
        )
        axes[row, 1].set_title(
            f"ROI binary mask ({ROI_BINARIZE_METHOD}, {roi_thr_label})\n"
            f"{BLOB_METHOD} -> {len(blobs)} blob(s)",
            fontsize=10,
        )
        axes[row, 1].axis("off")

        axes[row, 2].imshow(disp_q)
        for i, blob in enumerate(blobs):
            color = CMAP(i % 10)
            rx0, ry0, rx1, ry1 = blob["raw_px_bbox"]
            axes[row, 2].add_patch(
                Rectangle(
                    (rx0, ry0),
                    rx1 - rx0,
                    ry1 - ry0,
                    linewidth=1.2,
                    linestyle="--",
                    edgecolor=color,
                    facecolor="none",
                )
            )
            cx0, cy0, cx1, cy1 = blob["crop_px_bbox"]
            axes[row, 2].add_patch(
                Rectangle(
                    (cx0, cy0),
                    cx1 - cx0,
                    cy1 - cy0,
                    linewidth=1.8,
                    edgecolor=color,
                    facecolor="none",
                )
            )
            axes[row, 2].text(
                cx0, cy0 - 4, f"b{i}", color=color, fontsize=9, fontweight="bold"
            )
        axes[row, 2].set_title(
            f"blob bbox (dashed) -> padded+floored crop (solid)\n"
            f"pad={CROP_PADDING_FRACTION:.0%}  max_upscale={MAX_UPSCALE_FACTOR:.1f}x",
            fontsize=10,
        )
        axes[row, 2].axis("off")

    plt.suptitle(
        f"Two-stage ROI refinement — binarize + blob + crop  |  part_type={FOCUS_PART_TYPE}",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    plt.show()

    # %% Visualisation — two-stage: density-map methods on the detailed view of the crops
    for name in stage2_methods:
        m = focus["per_method"][name]
        blobs = m["stage2"]["blobs"]
        if not blobs:
            continue

        n_rows = len(blobs)
        fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4.5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, 4)

        for row, blob in enumerate(blobs):
            crop_img = blob["crop_img"]
            raw_crop = blob["raw_crop"]
            gt_crop = blob["crop_gt_patch_mask"]
            thr = m["threshold"]
            binary_crop = raw_crop > thr

            axes[row, 0].imshow(crop_img)
            axes[row, 0].set_title(
                f"[{name}] blob b{row} crop  ({crop_img.size[0]}x{crop_img.size[1]}px)",
                fontsize=10,
            )
            axes[row, 0].axis("off")

            im = axes[row, 1].imshow(raw_crop, cmap="jet", aspect="auto")
            axes[row, 1].set_title("crop raw score map", fontsize=10)
            axes[row, 1].axis("off")
            plt.colorbar(im, ax=axes[row, 1], shrink=0.75, pad=0.02)

            axes[row, 2].imshow(binary_crop, cmap="Greys_r", aspect="auto")
            if gt_crop.any():
                axes[row, 2].contour(
                    gt_crop.astype(float), levels=[0.5], colors="lime", linewidths=1.2
                )
            axes[row, 2].set_title(f"crop binary (thr={thr:.3f}) + GT outline", fontsize=10)
            axes[row, 2].axis("off")

            if gt_crop.any():
                true_vals_c = raw_crop[gt_crop]
                false_vals_c = raw_crop[~gt_crop]
                bins_c = np.linspace(raw_crop.min(), raw_crop.max(), 40)
                axes[row, 3].hist(
                    false_vals_c, bins=bins_c, alpha=0.6, color="tab:gray", label="GT-false"
                )
                axes[row, 3].hist(
                    true_vals_c, bins=bins_c, alpha=0.6, color="tab:red", label="GT-true"
                )
                axes[row, 3].legend(fontsize=8)
            else:
                axes[row, 3].hist(raw_crop.ravel(), bins=40, alpha=0.6, color="tab:gray")
            axes[row, 3].axvline(
                thr, color="black", linestyle="--", linewidth=1.5, label="threshold"
            )
            axes[row, 3].set_title("crop score histogram", fontsize=10)

        plt.suptitle(
            f"Two-stage crop detail — [{name}]  part_type={FOCUS_PART_TYPE}  "
            f"{len(blobs)} crop(s)",
            fontsize=12,
            y=1.01,
        )
        plt.tight_layout()
        plt.show()

# %%
