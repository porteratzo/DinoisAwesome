# %% [markdown]
# # Fundamental: Noisy FG/BG Cleaning — Does Discarding Ambiguous Exemplar Patches
# # Improve Oracle-IoU Localization?
#
# `multiscale_crop_ablation.py` and this folder's `augmented_prototype_oracle_iou*.py` build
# exemplar foreground/background galleries with a single bool threshold on patch-mask overlap
# (`pixel_mask_to_patch_mask`, `MASK_PATCH_THRESHOLD=0.3`) — every 16x16 patch that's at least
# 30% covered by the instance mask counts as "foreground", everything else "background". A
# patch straddling the object's edge is neither: it's part background pixels, part object
# pixels, and the resulting token is a blend of both appearances — noise in whichever gallery
# it lands in. This file asks whether removing that noise, with three cheap, unsupervised
# techniques, improves localization on a *different* query image:
#
#   - **Step 1 (Spatial Filter)** — mixed-patch rejection. A patch counts as foreground only
#     above `FG_HIGH=0.85` mask coverage, background only below `FG_LOW=0.15`; the ambiguous
#     `0.15 < Pfg < 0.85` band (mostly boundary patches) is dropped from *both* galleries
#     rather than assigned to either.
#   - **Step 2 (DINO Attention Check)** — cross-checks the **raw** (0.3-threshold) foreground
#     patches against an independent appearance reference, dropping the least-similar tail as
#     suspected boundary/occlusion leakage that a purely geometric filter couldn't catch. Two
#     references are compared side by side (the file header wasn't sure which would work
#     better, so both are kept as separate branches all the way through):
#       - `cls`    — the [CLS] token of that instance's own **close** crop (the tightest,
#         least-background-contaminated view available), L2-normalised, used as-is (CLS is
#         *not* L2-normalised by `DinoEncoder` itself — see `dinoisawesome.encoder.
#         ExtractorOutput`). Reused as the *same* reference at every scale (global/mid/close)
#         being cleaned, since the close crop's CLS is the purest available appearance signal.
#       - `center` — a masked-mean prototype built only from the mask's own innermost "core"
#         pixels (Euclidean distance-transform, keep the farthest-from-edge `CENTER_CORE_
#         PERCENTILE`), computed independently per scale from that scale's own crop — a
#         second, training-free way to ask "what does this look like, ignoring the edges".
#   - **Step 3 (Feature Clean)** — HDBSCAN + kNN consensus voting, run once per instance-type
#     *group* (`dinoisawesome.abc3.INSTANCE_TYPE_GROUPS` — e.g. "donut foam" merges the
#     multi- and single-instance annotation variants of the same physical object) pooling
#     every instance of that group's **raw** foreground patches across every part type at
#     once, so HDBSCAN sees as many real examples of "what this object looks like" as the
#     dataset has. A patch survives only if HDBSCAN placed it in a real cluster (not noise,
#     label -1) *and* a majority of its `KNN_CONSENSUS_K` nearest neighbours in that same
#     pooled set share its label — HDBSCAN's own density-based label can be locally unstable
#     right at a cluster's edge, so the kNN vote is a second, independent check on exactly the
#     "residual boundary leakage" this step is meant to catch. Tokens are L2-normalised
#     throughout, so plain Euclidean distance (`sklearn.cluster.HDBSCAN`'s default metric) is
#     already a monotonic transform of cosine similarity (`||a-b||^2 = 2 - 2*cos_sim` for unit
#     vectors) — no custom metric needed.
#
# **This is an ablation, not a pipeline** — each step is applied directly to the same `raw`
# gallery, not to the previous step's output. `step2_cls`/`step2_center` cross-check `raw`'s
# own foreground patches, not `step1`'s; `step3` clusters `raw`'s own pooled foreground
# patches, not `step2`'s. Nothing compounds. This deliberately isolates each technique's own
# marginal effect against the same fixed baseline, rather than measuring a cumulative
# pipeline where a later step's apparent contribution is confounded with whatever the step
# before it already removed. Background is only ever cleaned by Step 1's own spatial filter
# (the mixed-patch band is ambiguous for either side) — Steps 2-3 are explicitly about
# *foreground* purity ("cross-check FG patches", "per class"; background has no class to
# cluster against), so `step2_cls`, `step2_center`, and `step3` all reuse `raw`'s background
# gallery unchanged. Every non-`raw` stage therefore differs from `raw` by exactly one
# change: its own foreground-cleaning technique, nothing else.
#
# Five stages are compared this way: `raw` (today's single-threshold baseline), `step1`
# (spatial filter alone), `step2_cls` / `step2_center` (both branches of the attention check,
# each applied to raw fg independently), `step3` (HDBSCAN + kNN consensus, applied to raw fg
# independently). Every stage is scored two ways — `proto` (single masked-mean prototype,
# cosine similarity) and `knn_fgbg` (raw fg/bg patch galleries, contrastive kNN — see
# `_shared.prototype_ops.knn_fgbg_score`) — the same two families `augmented_prototype_
# oracle_iou_knn_fgbg.py` compares, so a "does cleaning help the mean-collapsed method,
# the raw-gallery method, or both" question can be read off directly. Every stage/method
# pools **all three scales at once** ("global+mid+close/all", `multiscale_crop_ablation.py`'s
# best-performing `FGBG_SOURCE_COMBOS` entry) — the *only* scale combo scored here, since
# sweeping every combo x every stage x both methods would multiply this file's already
# five-stage sweep well past what a "does cleaning help" question needs.
#
# **Evaluation metric**: "oracle IoU" — sweep every candidate threshold on the raw
# cosine-similarity (or contrastive-kNN) score map and keep the best patch-mask IoU against
# the query's own GT mask (`_shared.thresholding.iou_threshold_curve`, the same helper
# `multiscale_crop_ablation.py` uses to *tune* its own threshold, applied directly here
# instead). This is a deliberate scope choice, not an oversight: `multiscale_crop_ablation.py`
# itself additionally clusters (DBSCAN) and greedily matches predicted-vs-GT *instances* for
# precision/recall/count-error — machinery this file doesn't need to answer "does the score
# map separate object from background better", and reproducing it here would roughly double
# the file for a question this experiment doesn't ask. Oracle IoU isolates exactly the
# localizability question a cleaner gallery should move, the same choice this folder's
# `augmented_prototype_oracle_iou*.py` siblings already made for an analogous "does changing
# how the gallery is built help" question.
#
# Every (part_type, instance-type group, ref instance) combo actually annotated in `data/abc3`
# is swept for the quantitative oracle-IoU results (not one hand-picked pair) — Step 3 in
# particular is *only* meaningful pooled across many instances of the same group. The
# qualitative figures (spatial-filter/attention-check/feature-clean/pipeline-summary grids)
# show one representative "focus" combo only, at all three scales, mirroring `augmented_
# prototype_oracle_iou_knn_fgbg.py`'s FOCUS_* convention.

# %% Logging — must be before torch import
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("noisy_fgbg_cleaning")

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from scipy import ndimage
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from tqdm import tqdm

from dinoisawesome import DinoEncoder, EncoderWithCache, compute_exemplar_features, load_annotations
from dinoisawesome.abc3 import INSTANCE_TYPE_GROUPS, PART_TYPES, available_instance_groups
from dinoisawesome.instance_detection import extract_patch_tokens

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared.mask_geometry import patch_fg_fraction, scale_crop_box  # noqa: E402
from _shared.prototype_ops import knn_fgbg_score  # noqa: E402
from _shared.thresholding import iou_threshold_curve  # noqa: E402

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

data_dir = _REPO_ROOT / "data" / "abc3"

REF_NUMBER = 1
QUERY_NUMBER = 2

# Every part type/instance is swept for the quantitative oracle-IoU sweep. Narrow this for
# fast iteration while developing the script, e.g. RUN_PART_TYPES = ["LHa"].
RUN_PART_TYPES: list[str] = PART_TYPES

# Focus combos for every qualitative (spatial-filter/attention-check/feature-clean/pipeline)
# figure — one of them (the first) is the same object as augmented_prototype_oracle_iou_
# knn_fgbg.py's FOCUS_*, for comparability; the other two span the dataset's other
# instance-type groups (see module docstring's "Reading the results" for why one instance
# alone is a bad stand-in for the dataset). Each entry falls back to the first discovered
# combo (with a warning) if not present under RUN_PART_TYPES.
FOCUS_COMBOS_SPEC: list[tuple[str, str, int]] = [
    ("LHa", "donut foam single", 1),  # group "donut foam" (original default)
    ("LHb", "velcro", 1),  # group "velcro" — single annotated instance
    ("RHb", "white clips", 2),  # group "white clips" — multi-instance class, 2nd instance
]

DINO_VERSION = "v3"
DINO_SIZE = "large"
IMG_SIZE = 768
LAYER_IDX = 23
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")
DINO_ENCODING_CACHE_DIR: str | None = os.environ.get("DINO_ENCODING_CACHE_DIR")
DEBIAS = True

MASK_PATCH_THRESHOLD = 0.3  # "raw" baseline single-threshold split, matches the rest of the repo
CROP_PADDING_FRACTION = 1.0  # close/mid crop padding, fraction of the mask bbox's own extent
MIN_CROP_SIZE = 128  # "close" is dropped (not the whole combo) below this native px size

SCALES: list[str] = ["global", "mid", "close"]
CROP_SCALES: list[str] = ["mid", "close"]  # scales needing their own crop; "global" reuses the
# full ref image already encoded once per part type (see Part 3.5)
SCALE_COLOR: dict[str, str] = {"global": "#2ecc71", "mid": "#f39c12", "close": "#e74c3c"}

# Step 1 — mixed-patch rejection: a patch below FG_LOW is confidently background, above
# FG_HIGH confidently foreground; the band between is dropped from both galleries.
FG_HIGH = 0.85
FG_LOW = 0.15

# Step 2 — DINO attention check: keep only the top ATTENTION_KEEP_FRACTION of step-1 fg
# patches by cosine similarity to the reference (close-crop CLS, or the center prototype).
ATTENTION_KEEP_FRACTION = 0.75

# Step 2's "center" branch: the innermost (100 - CENTER_CORE_PERCENTILE) percent of mask
# pixels by distance-from-edge (Euclidean distance transform) define the "core" region used
# to build the center prototype.
CENTER_CORE_PERCENTILE = 70.0

# Step 3 — HDBSCAN + kNN consensus, run once per instance-type group (pooling every part
# type's instances of that group at once).
HDBSCAN_MIN_CLUSTER_SIZE = 8
HDBSCAN_MIN_SAMPLES = 3
KNN_CONSENSUS_K = 10
KNN_CONSENSUS_MIN_AGREEMENT = 0.6

# fg-bg-knn scoring (Part 7): k for the per-patch kNN gallery lookup, same default
# multiscale_crop_ablation.py uses for its own "fg-bg-knn(...)" methods.
KNN_FGBG_NUM_NEIGHBOURS = 10

ORACLE_THRESHOLD_STEPS = 25

STAGES: list[str] = ["raw", "step1", "step2_cls", "step2_center", "step3"]
STAGE_LABELS: dict[str, str] = {
    "raw": "raw (0.3 threshold)",
    "step1": "step1 (spatial filter)",
    "step2_cls": "step2 (CLS check)",
    "step2_center": "step2 (center check)",
    "step3": "step3 (HDBSCAN + kNN)",
}
METHODS: list[str] = ["proto", "knn_fgbg"]
METHOD_COLOR: dict[str, str] = {"proto": "#7f8c8d", "knn_fgbg": "#2ecc71"}

SEED = 0
torch.manual_seed(SEED)

OUTPUT_DIR = _REPO_ROOT / "outputs" / "fundamental" / "noisy_fgbg_cleaning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log.info(
    "RUN_PART_TYPES=%s ref_number=%d query_number=%d  |  DINO%s-%s img_size=%d layer=%d  |  "
    "stages=%s methods=%s",
    RUN_PART_TYPES,
    REF_NUMBER,
    QUERY_NUMBER,
    DINO_VERSION,
    DINO_SIZE,
    IMG_SIZE,
    LAYER_IDX,
    STAGES,
    METHODS,
)

# %% Core helpers — combo identity, scoring, oracle IoU


def combo_key(d: dict) -> tuple[str, str, str, int]:
    """(part_type, instance_type group, class, instance_id) — a combo's stable identity."""
    return (d["part_type"], d["group"], d["class"], d["instance_id"])


def oracle_iou(raw: np.ndarray, gt_mask: np.ndarray, steps: int) -> float:
    """Best patch-mask IoU any single global threshold on *raw* could achieve against
    *gt_mask* — see augmented_prototype_oracle_iou.py's identical helper for rationale."""
    _, ious = iou_threshold_curve(raw, gt_mask, steps)
    return float(ious.max())


def annotate_bar_values(ax, bars, fmt: str = "%.3f") -> None:
    """Print each bar's own height above it — bar-chart differences in this file are often
    a few hundredths of oracle IoU, too small to read reliably off the y-axis alone."""
    ax.bar_label(bars, fmt=fmt, fontsize=7, padding=2)


def score_heatmap(tokens: torch.Tensor, prototype: torch.Tensor, h: int, w: int) -> np.ndarray:
    """ "proto" method: cosine-similarity heatmap, prototype vs. every query patch."""
    return (tokens @ prototype.T).reshape(h, w).cpu().float().numpy()


def knn_score_heatmap(
    tokens: torch.Tensor, fg_bank: torch.Tensor, bg_bank: torch.Tensor, k: int, h: int, w: int
) -> np.ndarray:
    """ "knn_fgbg" method heatmap — see _shared.prototype_ops.knn_fgbg_score."""
    return knn_fgbg_score(tokens, fg_bank, bg_bank, k).reshape(h, w)


def extract_tokens_batch_with_cls(
    encoder: DinoEncoder, images: list[Image.Image], layer_idx: int, debias: bool = False
) -> list[tuple[torch.Tensor, torch.Tensor, int, int]]:
    """Batched patch-token + [CLS]-token extraction — _shared.prototype_ops.
    extract_patch_tokens_batch's sibling, also returning each image's own L2-normalised
    [CLS] token (that helper discards it, matching extract_patch_tokens). CLS is *not*
    L2-normalised by DinoEncoder itself (unlike patch tokens), so it's normalised here.
    debias only ever affects patch tokens (see dinoisawesome.encoder), never CLS.
    """
    out = encoder(images, layers=[layer_idx], debias=debias)
    patches = out.patches[:, 0]  # (B, H, W, D)
    cls = F.normalize(out.cls[:, 0], p=2, dim=-1)  # (B, D)
    _, grid_h, grid_w, D = patches.shape
    return [
        (
            F.normalize(patches[b].reshape(grid_h * grid_w, D), p=2, dim=-1),
            cls[b],
            grid_h,
            grid_w,
        )
        for b in range(patches.shape[0])
    ]


# %% Step 1/2 helpers — spatial filter + attention check


def keep_top_fraction_by_similarity(
    fg_tokens: torch.Tensor, reference: torch.Tensor, keep_fraction: float
) -> torch.Tensor:
    """Boolean keep-mask over *fg_tokens* (Nfg, C): keeps the *keep_fraction* most similar
    (cosine) to *reference* ((1, C) or (C,)), dropping the least-similar tail as suspected
    boundary/occlusion leakage Step 1's purely geometric filter couldn't catch. Keeps
    everything when Nfg is too small (<4) for a percentile cut to be meaningful."""
    n = fg_tokens.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.bool)
    if n < 4:
        return torch.ones(n, dtype=torch.bool)
    sims = (fg_tokens @ reference.reshape(1, -1).T).squeeze(-1)
    cutoff = torch.quantile(sims.float(), 1.0 - keep_fraction)
    return sims >= cutoff


def center_prototype(
    tokens: torch.Tensor,
    mask_px: np.ndarray,
    grid_h: int,
    grid_w: int,
    core_percentile: float,
    label: str,
) -> torch.Tensor | None:
    """Masked-mean prototype over only the instance mask's innermost "core" pixels — the
    ones farthest from the mask boundary by Euclidean distance transform, above
    *core_percentile* of the in-mask distance distribution — a second, independent
    appearance reference for Step 2's attention check (alongside the close crop's [CLS]
    token). Returns None if the mask/core is too degenerate to project onto any patch.
    """
    if not mask_px.any():
        return None
    dist = ndimage.distance_transform_edt(mask_px)
    cutoff = np.percentile(dist[mask_px], core_percentile)
    core_px = (dist >= cutoff) & mask_px
    # MASK_PATCH_THRESHOLD, not FG_HIGH: core_px is already the innermost slice of the mask
    # (by distance-from-edge) — requiring a patch to *also* be 85%+ covered by that already-
    # eroded region left this empty for almost every combo/scale in practice, silencing the
    # center branch into a near-total no-op. A patch predominantly inside the core is enough.
    core_patch = patch_fg_fraction(core_px, grid_h, grid_w, IMG_SIZE) >= MASK_PATCH_THRESHOLD
    flat = torch.from_numpy(core_patch.reshape(-1)).to(tokens.device)
    if int(flat.sum().item()) == 0:
        log.warning("%s: core-region projection empty — skipping center prototype", label)
        return None
    return compute_exemplar_features(tokens[flat], mode="mean")  # (1, C)


def process_scale(
    scale: str,
    crop_img: Image.Image,
    tokens: torch.Tensor,
    grid_h: int,
    grid_w: int,
    own_mask_px: np.ndarray,
    excl_mask_px: np.ndarray,
    close_cls: torch.Tensor | None,
    label: str,
) -> dict:
    """One (combo, scale)'s Step 1 (spatial filter) computation, plus Step 2's (attention
    check) cls/center keep-masks — both are independent ablations against this scale's
    **raw** (0.3-threshold) foreground, not a cascade: Step 2 cross-checks `raw_fg_tokens`
    directly, never `step1`'s output (see the file header). Step 3 is a cross-combo,
    per-group operation and is handled separately (see pool_and_clean_group), also pooling
    `raw` fg directly — this function only ever looks at this one crop.

    *own_mask_px* is this instance's own mask (defines foreground); *excl_mask_px* is the
    union of every instance in this combo's group that falls inside this crop (defines
    background — "not any instance", not just "not this one", mirroring multiscale_crop_
    ablation.py's exclude_patch_mask so a neighbouring same-group instance never leaks into
    the background gallery).

    Returns every intermediate array needed both to build this combo's stage galleries and,
    for the focus combo only, to drive the qualitative figures — most fields are numpy
    arrays over this scale's own flat (grid_h*grid_w,) patch-index space, so a fallback
    (empty step-1 fg/bg) or a downstream slice can always be traced back to real patches.
    """
    own_frac = patch_fg_fraction(own_mask_px, grid_h, grid_w, IMG_SIZE)
    excl_frac = patch_fg_fraction(excl_mask_px, grid_h, grid_w, IMG_SIZE)

    raw_fg_flat = (own_frac >= MASK_PATCH_THRESHOLD).reshape(-1)
    raw_bg_flat = (excl_frac < MASK_PATCH_THRESHOLD).reshape(-1)
    step1_fg_flat = (own_frac >= FG_HIGH).reshape(-1)
    step1_bg_flat = (excl_frac <= FG_LOW).reshape(-1)

    if not step1_fg_flat.any():
        log.warning(
            "%s scale=%-6s: spatial filter left zero fg patches — falling back to raw fg",
            label,
            scale,
        )
        step1_fg_flat = raw_fg_flat
    if not step1_bg_flat.any():
        log.warning(
            "%s scale=%-6s: spatial filter left zero bg patches — falling back to raw bg",
            label,
            scale,
        )
        step1_bg_flat = raw_bg_flat

    raw_fg_idx = np.flatnonzero(raw_fg_flat)
    raw_fg_tokens = tokens[torch.from_numpy(raw_fg_flat).to(tokens.device)]
    raw_bg_tokens = tokens[torch.from_numpy(raw_bg_flat).to(tokens.device)]
    step1_fg_idx = np.flatnonzero(step1_fg_flat)
    step1_fg_tokens = tokens[torch.from_numpy(step1_fg_flat).to(tokens.device)]
    step1_bg_tokens = tokens[torch.from_numpy(step1_bg_flat).to(tokens.device)]

    cls_keep = cls_sims = None
    if close_cls is not None and raw_fg_tokens.shape[0] > 0:
        sims = (raw_fg_tokens @ close_cls.reshape(1, -1).T).squeeze(-1)
        cls_sims = sims.cpu().float().numpy()
        cls_keep = (
            keep_top_fraction_by_similarity(raw_fg_tokens, close_cls, ATTENTION_KEEP_FRACTION)
            .cpu()
            .numpy()
        )

    center = center_prototype(tokens, own_mask_px, grid_h, grid_w, CENTER_CORE_PERCENTILE, label)
    center_keep = center_sims = None
    if center is not None and raw_fg_tokens.shape[0] > 0:
        sims = (raw_fg_tokens @ center.T).squeeze(-1)
        center_sims = sims.cpu().float().numpy()
        center_keep = (
            keep_top_fraction_by_similarity(raw_fg_tokens, center, ATTENTION_KEEP_FRACTION)
            .cpu()
            .numpy()
        )

    return {
        "scale": scale,
        "img": crop_img,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "own_frac": own_frac,
        "excl_frac": excl_frac,
        "raw_fg_flat": raw_fg_flat,
        "raw_bg_flat": raw_bg_flat,
        "step1_fg_flat": step1_fg_flat,
        "step1_bg_flat": step1_bg_flat,
        "raw_fg_idx": raw_fg_idx,
        "step1_fg_idx": step1_fg_idx,
        "raw_fg_tokens": raw_fg_tokens,
        "raw_bg_tokens": raw_bg_tokens,
        "step1_fg_tokens": step1_fg_tokens,
        "step1_bg_tokens": step1_bg_tokens,
        "cls_keep": cls_keep,
        "cls_sims": cls_sims,
        "center_keep": center_keep,
        "center_sims": center_sims,
        "center_proto": center,
    }


# %% Step 3 helper — HDBSCAN + kNN consensus voting, per instance-type group


def hdbscan_knn_consensus_keep(
    tokens: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    knn_k: int,
    min_agreement: float,
) -> tuple[np.ndarray, np.ndarray]:
    """HDBSCAN-cluster *tokens* (N, C), L2-normalised, then keep a point only if (a) HDBSCAN
    placed it in a real cluster (label != -1) and (b) a majority (>= min_agreement) of its
    knn_k nearest neighbours in this same set share that label — HDBSCAN's own noise flag
    catches sparse outliers, the kNN vote catches points HDBSCAN happened to assign to a
    cluster despite sitting on that cluster's own ragged boundary. Tokens are L2-normalised,
    so plain Euclidean distance (HDBSCAN's default metric) is already a monotonic transform
    of cosine similarity, hence no custom metric is needed for either step.

    Returns (keep, hdbscan_labels). Too few points to cluster meaningfully (fewer than
    max(min_cluster_size, knn_k + 1)) short-circuits to "keep everything" — there isn't
    enough data for HDBSCAN's density estimate to mean anything.
    """
    n = tokens.shape[0]
    if n < max(min_cluster_size, knn_k + 1):
        return np.ones(n, dtype=bool), np.zeros(n, dtype=int)
    labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(tokens)
    sims = tokens @ tokens.T
    np.fill_diagonal(sims, -np.inf)
    k = min(knn_k, n - 1)
    knn_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    keep = np.zeros(n, dtype=bool)
    for i in range(n):
        if labels[i] == -1:
            continue
        agreement = float(np.mean(labels[knn_idx[i]] == labels[i]))
        keep[i] = agreement >= min_agreement
    return keep, labels


def pool_and_clean_group(
    group_combos: list[dict],
    combo_galleries: dict[tuple, dict],
    stage_source: str,
    capture: bool,
) -> tuple[dict[tuple, torch.Tensor], dict | None]:
    """Pools *stage_source*'s (here always "raw" — Step 3 clusters the raw foreground
    gallery directly, not a previous step's output, see the file header) fg tokens —
    already pooled across scales per combo — across every combo in one instance-type group
    (every part type at once, abc3's groups aren't per-part-type), runs one HDBSCAN +
    kNN-consensus pass over the pooled set, and splits the surviving tokens back out per
    combo.

    Diagnostics (pooled numpy tokens, HDBSCAN labels, keep mask, and the {combo_key: (start,
    end)} slice map) are only captured when *capture* is True — used for Part 9's Step-3
    visualization of the focus combo's own group. A combo whose gallery collapsed to zero
    tokens is skipped from the pool but simply absent from the returned dict; callers fall
    back to that combo's own stage_source gallery unfiltered (logged) rather than erroring.
    """
    chunks: list[torch.Tensor] = []
    slices: list[tuple[tuple, int, int]] = []
    offset = 0
    for combo in group_combos:
        ck = combo_key(combo)
        fg = combo_galleries[ck][stage_source]["fg"]
        if fg.shape[0] == 0:
            continue
        chunks.append(fg)
        slices.append((ck, offset, offset + fg.shape[0]))
        offset += fg.shape[0]
    if not chunks:
        return {}, None

    pooled = torch.cat(chunks, dim=0)
    keep, labels = hdbscan_knn_consensus_keep(
        pooled.cpu().numpy(),
        HDBSCAN_MIN_CLUSTER_SIZE,
        HDBSCAN_MIN_SAMPLES,
        KNN_CONSENSUS_K,
        KNN_CONSENSUS_MIN_AGREEMENT,
    )
    keep_t = torch.from_numpy(keep)
    result: dict[tuple, torch.Tensor] = {}
    for ck, start, end in slices:
        kept = pooled[start:end][keep_t[start:end]]
        if kept.shape[0] == 0:
            log.warning(
                "%s stage=%s: HDBSCAN + kNN consensus rejected every patch — falling back to "
                "the unfiltered %s gallery",
                ck,
                stage_source,
                stage_source,
            )
            kept = pooled[start:end]
        result[ck] = kept

    diagnostics = None
    if capture:
        diagnostics = {
            "pooled": pooled.cpu().numpy(),
            "labels": labels,
            "keep": keep,
            "slices": slices,
        }
    return result, diagnostics


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
combos_by_key: dict[tuple, dict] = {combo_key(c): c for c in combos}
combos_by_group: dict[str, list[dict]] = defaultdict(list)
for c in combos:
    combos_by_group[c["group"]].append(c)
log.info(
    "Discovered %d (part_type, group, instance) combos across %d part types, %d groups",
    len(combos),
    len({c["part_type"] for c in combos}),
    len(combos_by_group),
)

# %% Focus combos — used for every qualitative figure below (chosen before the main sweep so
# Part 5/6 know when to capture extra diagnostics for them)


def resolve_focus_combo(part_type: str, cls: str, instance_id: int) -> dict:
    combo = next(
        (
            c
            for c in combos
            if c["part_type"] == part_type and c["class"] == cls and c["instance_id"] == instance_id
        ),
        None,
    )
    if combo is None:
        combo = combos[0]
        log.warning(
            "Focus combo part_type=%s class=%r instance_id=%d not found under "
            "RUN_PART_TYPES=%s — falling back to %s",
            part_type,
            cls,
            instance_id,
            RUN_PART_TYPES,
            combo_key(combo),
        )
    return combo


focus_combos: list[dict] = []
focus_keys: set[tuple] = set()
for _part_type, _cls, _instance_id in FOCUS_COMBOS_SPEC:
    _combo = resolve_focus_combo(_part_type, _cls, _instance_id)
    _ck = combo_key(_combo)
    if _ck in focus_keys:
        log.warning(
            "Focus combo %s already selected (duplicate spec or fallback collision) — skipping "
            "the repeat",
            _ck,
        )
        continue
    focus_keys.add(_ck)
    focus_combos.append(_combo)
log.info("Focus combos for qualitative figures: %s", [combo_key(c) for c in focus_combos])

# %% Part 2 — build mid/close crops per combo ("global" reuses the full ref image, Part 3.5)
combo_keys_by_scale: dict[str, list[tuple]] = defaultdict(list)
for combo in tqdm(combos, desc="Building mid/close crops"):
    ref_img = ref_images[combo["part_type"]]
    group_mask = group_ref_masks.get((combo["part_type"], combo["group"]), combo["ref_mask"])
    combo["crops"] = {}
    for scale in CROP_SCALES:
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
        combo["crops"][scale] = {
            "img": ref_img.crop(box),
            "mask_px": combo["ref_mask"][y0:y1, x0:x1],
            "exclude_mask_px": group_mask[y0:y1, x0:x1],
        }
        combo_keys_by_scale[scale].append(combo_key(combo))

for scale in CROP_SCALES:
    log.info("scale=%-5s usable combos=%d", scale, len(combo_keys_by_scale[scale]))

# %% Part 3 — encode each part type's query image once; per-(part_type, group) GT patch mask
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

query_encodings: dict[str, dict] = {}
for part_type in tqdm(sorted(query_images), desc="Encoding query images"):
    q_tokens, q_h, q_w = extract_patch_tokens(
        encoder, query_images[part_type], LAYER_IDX, debias=DEBIAS
    )
    query_encodings[part_type] = {"q_tokens": q_tokens, "q_h": q_h, "q_w": q_w}

gt_patch_masks: dict[tuple[str, str], np.ndarray] = {}
for (part_type, group), pixel_mask in group_query_masks.items():
    q = query_encodings[part_type]
    gt_patch_masks[(part_type, group)] = (
        patch_fg_fraction(pixel_mask, q["q_h"], q["q_w"], IMG_SIZE) >= MASK_PATCH_THRESHOLD
    )

# %% Part 3.5 — encode each part type's full, uncropped reference image once: the "global"
# scale for every combo sharing that part type.
ref_encodings: dict[str, dict] = {}
for part_type in tqdm(sorted(ref_images), desc="Encoding ref images (global scale)"):
    r_tokens, r_h, r_w = extract_patch_tokens(
        encoder, ref_images[part_type], LAYER_IDX, debias=DEBIAS
    )
    ref_encodings[part_type] = {"r_tokens": r_tokens, "r_h": r_h, "r_w": r_w}

# %% Part 4 — batched-encode every combo's mid/close crops (patch tokens + [CLS] token)
crop_items: list[tuple[tuple, str]] = [
    (combo_key(c), scale) for c in combos for scale in c["crops"]
]
for i in tqdm(range(0, len(crop_items), chunk_size), desc="Encoding mid/close crops"):
    chunk = crop_items[i : i + chunk_size]
    images = [combos_by_key[ck]["crops"][scale]["img"] for ck, scale in chunk]
    encoded = extract_tokens_batch_with_cls(encoder, images, LAYER_IDX, debias=DEBIAS)
    for (ck, scale), (tokens, cls, grid_h, grid_w) in zip(chunk, encoded):
        crop = combos_by_key[ck]["crops"][scale]
        crop["tokens"], crop["grid_h"], crop["grid_w"] = tokens, grid_h, grid_w
        if scale == "close":
            crop["cls"] = cls

log.info("Encoded %d (combo, scale) crops with patch tokens + [CLS]", len(crop_items))

# %% Part 5 — main combo-major sweep: Step 1 (spatial filter) + Step 2 (attention check),
# per combo per scale, building every combo's raw/step1/step2_cls/step2_center galleries.
# step2_cls/step2_center filter `raw`'s own foreground directly (not step1's output — see
# the file header; this is an ablation, not a cascade). Step 3 (cross-combo, per-group,
# also sourced from `raw`) is handled separately in Part 6 below.
combo_galleries: dict[tuple, dict] = {}
focus_scale_diag: dict[tuple, dict[str, dict]] = {}

for combo in tqdm(combos, desc="Part 5: spatial filter + attention check"):
    ck = combo_key(combo)
    part_type = combo["part_type"]
    is_focus = ck in focus_keys

    procs: list[dict] = []
    r = ref_encodings[part_type]
    group_mask = group_ref_masks.get((part_type, combo["group"]), combo["ref_mask"])
    procs.append(
        process_scale(
            "global",
            ref_images[part_type],
            r["r_tokens"],
            r["r_h"],
            r["r_w"],
            combo["ref_mask"],
            group_mask,
            None,  # step2_cls's reference always comes from this combo's own close crop
            str(ck),
        )
    )
    for scale, crop in combo["crops"].items():
        close_cls = combo["crops"]["close"]["cls"] if "close" in combo["crops"] else None
        procs.append(
            process_scale(
                scale,
                crop["img"],
                crop["tokens"],
                crop["grid_h"],
                crop["grid_w"],
                crop["mask_px"],
                crop["exclude_mask_px"],
                close_cls,
                str(ck),
            )
        )
    if "close" not in combo["crops"]:
        log.warning(
            "%s: 'close' scale dropped (below MIN_CROP_SIZE) — step2_cls has no [CLS] "
            "reference for this combo, falls back to raw fg unfiltered",
            ck,
        )

    galleries: dict[str, dict] = {
        "raw": {
            "fg": torch.cat([p["raw_fg_tokens"] for p in procs], dim=0),
            "bg": torch.cat([p["raw_bg_tokens"] for p in procs], dim=0),
        },
        "step1": {
            "fg": torch.cat([p["step1_fg_tokens"] for p in procs], dim=0),
            "bg": torch.cat([p["step1_bg_tokens"] for p in procs], dim=0),
        },
    }
    # step2_cls/step2_center filter `raw`'s own fg directly (not step1's — see the file
    # header) and reuse `raw`'s own bg unchanged, so each differs from `raw` by exactly one
    # change. raw_sizes (per-scale raw-fg patch counts, same concatenation order as
    # galleries["raw"]["fg"]) is recorded once here — Part 6 pools this combo's raw fg
    # per group for Step 3, and the focus-combo visualizations use raw_sizes to trace a
    # Step-3 survivor back to its own (scale, patch) location.
    for branch in ("cls", "center"):
        chunks = []
        for p in procs:
            keep = p[f"{branch}_keep"]
            fg_kept = p["raw_fg_tokens"] if keep is None else p["raw_fg_tokens"][keep]
            chunks.append(fg_kept)
        fg_cat = torch.cat(chunks, dim=0)
        if fg_cat.shape[0] == 0:
            log.warning(
                "%s: step2_%s left zero fg patches across every scale — falling back to raw",
                ck,
                branch,
            )
            fg_cat = galleries["raw"]["fg"]
        galleries[f"step2_{branch}"] = {"fg": fg_cat, "bg": galleries["raw"]["bg"]}
    galleries["raw_sizes"] = [(p["scale"], p["raw_fg_tokens"].shape[0]) for p in procs]  # type: ignore[assignment]

    combo_galleries[ck] = galleries
    if is_focus:
        focus_scale_diag[ck] = {p["scale"]: p for p in procs}

# %% Part 6 — Step 3: HDBSCAN + kNN consensus voting, once per instance-type group, pooling
# every combo's own **raw** fg gallery directly (not step 1/2's output — see the file
# header: this is an ablation against a fixed baseline, not a cascade). Populates
# combo_galleries[ck]["step3"]; a combo absent from the pooled result (its raw fg gallery
# was empty) falls back to that combo's own raw fg unfiltered, logged, same fallback
# pattern as every other stage in this file.
# group_diagnostics captures every group's (not just the focus one's) pooled tokens/labels/
# keep-mask/slices — cheap for this dataset (a handful of groups, at most a couple thousand
# points each) and needed by the per-scale survival check right below, which exists because
# a *single* focus instance's Step-3 behaviour (see step3_feature_clean.png/
# pipeline_summary.png) is not a reliable stand-in for what Step 3 does dataset-wide.
group_diagnostics: dict[str, dict | None] = {}
for group, group_combos in tqdm(combos_by_group.items(), desc="Part 6: HDBSCAN + kNN consensus"):
    result, diag = pool_and_clean_group(group_combos, combo_galleries, "raw", capture=True)
    group_diagnostics[group] = diag
    for combo in group_combos:
        ck = combo_key(combo)
        if ck in result:
            fg = result[ck]
        else:
            log.warning(
                "%s: absent from group=%s pooled result (empty raw fg gallery) "
                "— step3 falls back to raw fg unfiltered",
                ck,
                group,
            )
            fg = combo_galleries[ck]["raw"]["fg"]
        combo_galleries[ck]["step3"] = {
            "fg": fg,
            "bg": combo_galleries[ck]["raw"]["bg"],
        }
step3_diagnostics_by_focus: dict[tuple, dict | None] = {
    combo_key(c): group_diagnostics.get(c["group"]) for c in focus_combos
}

log.info("Step 3 complete for %d instance-type groups", len(combos_by_group))

# Per-scale Step-3 survival, aggregated across *every* combo (not just the focus one) — see
# the module docstring's "Reading the results" section for why this matters: the focus
# combo can (and here, for one specific instance, does) look like Step 3 wipes out an
# entire scale, when the dataset-wide picture is much less dramatic.
_scale_totals: dict[str, list[int]] = {s: [0, 0] for s in SCALES}
for group, diag in group_diagnostics.items():
    if diag is None:
        continue
    for ck, start, end in diag["slices"]:
        keep_this_combo = diag["keep"][start:end]
        offset = 0
        for s, n in combo_galleries[ck]["raw_sizes"]:
            _scale_totals[s][0] += int(keep_this_combo[offset : offset + n].sum())
            _scale_totals[s][1] += n
            offset += n
log.info("Step 3 survival by scale, aggregated across every combo:")
for s in SCALES:
    k, n = _scale_totals[s]
    log.info("  scale=%-6s kept=%d/%d (%.1f%%)", s, k, n, 100 * k / n if n else float("nan"))

# %% Part 7 — score every combo x every stage x every method (oracle IoU)
iou_lookup: dict[str, dict[str, dict[tuple, float]]] = {m: {s: {} for s in STAGES} for m in METHODS}

for combo in tqdm(combos, desc="Part 7: scoring"):
    ck = combo_key(combo)
    part_type, group = combo["part_type"], combo["group"]
    q = query_encodings[part_type]
    gt = gt_patch_masks.get((part_type, group))
    if gt is None:
        continue
    for stage in STAGES:
        fg, bg = combo_galleries[ck][stage]["fg"], combo_galleries[ck][stage]["bg"]
        if fg.shape[0] == 0 or bg.shape[0] == 0:
            log.warning(
                "%s stage=%s: empty fg/bg gallery — skipping this (combo, stage)", ck, stage
            )
            continue
        proto = compute_exemplar_features(fg, mode="mean")
        raw_proto = score_heatmap(q["q_tokens"], proto, q["q_h"], q["q_w"])
        iou_lookup["proto"][stage][ck] = oracle_iou(raw_proto, gt, ORACLE_THRESHOLD_STEPS)

        raw_knn = knn_score_heatmap(
            q["q_tokens"], fg, bg, KNN_FGBG_NUM_NEIGHBOURS, q["q_h"], q["q_w"]
        )
        iou_lookup["knn_fgbg"][stage][ck] = oracle_iou(raw_knn, gt, ORACLE_THRESHOLD_STEPS)

log.info(
    "Scoring complete: %d combos x %d stages x %d methods", len(combos), len(STAGES), len(METHODS)
)

# %% Part 8 — aggregate + bar chart


def stage_method_summary(combo_keys: set[tuple] | None = None) -> pd.DataFrame:
    """Mean/std oracle IoU per (method, stage), pooled over *combo_keys* (None = every
    combo). Shared by the aggregate chart below and Part 8b's per-instance-type breakdown —
    same (method, stage) shape, just a different combo subset feeding each row's vals."""
    rows: list[dict] = []
    for method in METHODS:
        for stage in STAGES:
            vals = [
                v
                for ck, v in iou_lookup[method][stage].items()
                if combo_keys is None or ck in combo_keys
            ]
            rows.append(
                {
                    "method": method,
                    "stage": stage,
                    "mean_iou": float(np.mean(vals)) if vals else float("nan"),
                    "std_iou": float(np.std(vals)) if vals else float("nan"),
                    "n_combos": len(vals),
                }
            )
    return pd.DataFrame(rows)


def log_stage_method_summary(summary_df: pd.DataFrame) -> None:
    """Logs one line per stage, method scores side by side — caller logs its own header
    (aggregate vs. a specific group) immediately before calling this."""
    for stage in STAGES:
        parts = []
        for method in METHODS:
            row = summary_df[(summary_df.stage == stage) & (summary_df.method == method)].iloc[0]
            parts.append(f"{method}={row.mean_iou:.3f}+/-{row.std_iou:.3f} (n={row.n_combos})")
        log.info("  %-14s  %s", stage, "  ".join(parts))


def plot_oracle_iou_bar_chart(summary_df: pd.DataFrame, title: str, out_path: Path) -> None:
    """Grouped (method x stage) bar chart with error bars — the aggregate chart (Part 8)
    and every per-instance-type chart (Part 8b) share this renderer; only *summary_df*'s
    combo subset, *title*, and *out_path* differ between calls."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(STAGES))
    width = 0.8 / len(METHODS)
    for i, method in enumerate(METHODS):
        means = [
            summary_df[(summary_df.stage == s) & (summary_df.method == method)]["mean_iou"].iloc[0]
            for s in STAGES
        ]
        stds = [
            summary_df[(summary_df.stage == s) & (summary_df.method == method)]["std_iou"].iloc[0]
            for s in STAGES
        ]
        bars = ax.bar(
            x + i * width,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            label=method,
            color=METHOD_COLOR[method],
        )
        annotate_bar_values(ax, bars)
    ax.set_xticks(
        x + width * (len(METHODS) - 1) / 2,
        [STAGE_LABELS[s] for s in STAGES],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("oracle IoU (mean +/- std across combos)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


summary_df = stage_method_summary()
log.info("Oracle-IoU summary (mean +/- std across %d combos):", len(combos))
log_stage_method_summary(summary_df)

_aggregate_chart_path = OUTPUT_DIR / "oracle_iou_by_stage.png"
plot_oracle_iou_bar_chart(
    summary_df,
    f"Noisy fg/bg cleaning — oracle IoU per stage, global+mid+close/all scale combo "
    f"({len(combos)} combos across {len(RUN_PART_TYPES)} part types)",
    _aggregate_chart_path,
)
log.info("Saved oracle-IoU bar chart to %s", _aggregate_chart_path)

# %% Part 8b — per-instance-type (group) breakdown. The aggregate chart above pools every
# group together, which can hide a group-specific effect the same way a single focus
# combo's Step-3 figures can misrepresent the dataset-wide average (see "Reading the
# results" below) — one bar chart per instance-type group makes that visible directly in
# the oracle-IoU numbers, rather than relying on the qualitative figures plus the Part 6
# per-scale survival log to catch it.
for _group in sorted(combos_by_group):
    _group_keys = {combo_key(c) for c in combos_by_group[_group]}
    _group_summary_df = stage_method_summary(_group_keys)

    log.info(
        "Oracle-IoU summary for group=%r (mean +/- std across %d combos):",
        _group,
        len(_group_keys),
    )
    log_stage_method_summary(_group_summary_df)

    _group_chart_path = OUTPUT_DIR / f"oracle_iou_by_stage__{_group.replace(' ', '_')}.png"
    plot_oracle_iou_bar_chart(
        _group_summary_df,
        f"Noisy fg/bg cleaning — oracle IoU per stage, group={_group!r} "
        f"({len(_group_keys)} combos)",
        _group_chart_path,
    )
    log.info("Saved oracle-IoU bar chart for group=%r to %s", _group, _group_chart_path)

# %% [markdown]
# ## Qualitative figures — one figure set per focus combo, all three scales
#
# Everything below visualizes each of `focus_combos` (`FOCUS_COMBOS_SPEC`) at
# global/mid/close, showing Steps 1-3 one at a time — each computed independently against
# `raw`, not against the step before it (see the file header) — then a pipeline-summary grid
# showing every stage side by side for direct comparison. Three instance types are shown
# (not just one) so the qualitative read isn't drawn from a single object's idiosyncrasies —
# see "Reading the results" below for a case where that would have been actively misleading.
# Each combo's figures are saved with its own `__<part_type>_<class>_<instance_id>` filename
# suffix.


# %% Visualization helpers
def flat_idx_to_bool_grid(idx: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    grid = np.zeros(grid_h * grid_w, dtype=bool)
    grid[idx] = True
    return grid.reshape(grid_h, grid_w)


def sims_to_grid(idx: np.ndarray, sims: np.ndarray | None, grid_h: int, grid_w: int) -> np.ndarray:
    """NaN-filled (grid_h, grid_w) grid with *sims* placed at *idx* — for imshow with a
    NaN-aware colormap, restricting a similarity heatmap to only the patches it was
    computed for (step-1 foreground)."""
    grid = np.full(grid_h * grid_w, np.nan)
    if sims is not None:
        grid[idx] = sims
    return grid.reshape(grid_h, grid_w)


def step3_kept_flat_idx(
    focus_key: tuple,
    scale_diag: dict[str, dict],
    diag: dict | None,
    scale: str,
) -> np.ndarray:
    """For one focus combo (*focus_key*, with its own per-scale diagnostics *scale_diag* and
    its group's Step-3 diagnostics *diag*): which flat patch-grid indices (into that scale's
    own grid_h*grid_w layout) survived Step 3 — reconstructed by walking back through the
    per-scale raw-fg segment sizes recorded in Part 5 (``raw_sizes``) and the group-level
    keep mask captured in Part 6. Step 3 clusters `raw` fg directly (not step 1/2's output
    — see the file header), so the base index here is `raw_fg_idx`, not a filtered subset.
    """
    sizes = combo_galleries[focus_key]["raw_sizes"]
    if diag is None:
        return np.array([], dtype=int)
    ck_start, ck_end = next((s, e) for ck, s, e in diag["slices"] if ck == focus_key)
    keep_this_combo = diag["keep"][ck_start:ck_end]
    offset = 0
    for s, n in sizes:
        if s == scale:
            scale_keep = keep_this_combo[offset : offset + n]
            return scale_diag[scale]["raw_fg_idx"][scale_keep]
        offset += n
    return np.array([], dtype=int)


def overlay_patches(
    ax, img: Image.Image, idx: np.ndarray, grid_h: int, grid_w: int, color: str
) -> None:
    """Draw a translucent *color* square over every patch in *idx* on top of *img*."""
    ax.imshow(img)
    w, h = img.size
    ph, pw = h / grid_h, w / grid_w
    grid = flat_idx_to_bool_grid(idx, grid_h, grid_w)
    ys, xs = np.where(grid)
    for y, x in zip(ys, xs):
        ax.add_patch(
            plt.Rectangle((x * pw, y * ph), pw, ph, facecolor=color, edgecolor="none", alpha=0.45)
        )
    ax.axis("off")


def focus_combo_slug(combo: dict) -> str:
    """Filesystem-safe identity string for one focus combo's figure filenames."""
    return f"{combo['part_type']}_{combo['class']}_{combo['instance_id']}".replace(" ", "_")


def render_focus_qualitative_figures(
    focus_combo: dict,
    scale_diag: dict[str, dict],
    step3_diag: dict | None,
) -> None:
    """Visualizations 1-4 (spatial filter, attention check, feature clean, pipeline
    summary) for one focus combo — everything the module header calls "qualitative
    figures". Called once per entry in `focus_combos` (see FOCUS_COMBOS_SPEC); each combo's
    files get their own `__<slug>` suffix so they don't overwrite each other."""
    focus_key = combo_key(focus_combo)
    slug = focus_combo_slug(focus_combo)

    def out_path(name: str) -> Path:
        return OUTPUT_DIR / f"{name}__{slug}.png"

    # Visualization 1 — Step 1 (spatial filter), all three scales
    fig, axes = plt.subplots(len(SCALES), 3, figsize=(13, 4.3 * len(SCALES)))
    for row, scale in enumerate(SCALES):
        diag = scale_diag[scale]
        axes[row, 0].imshow(diag["img"])
        axes[row, 0].set_title(f"scale={scale}: crop")
        axes[row, 0].axis("off")

        im = axes[row, 1].imshow(diag["own_frac"], cmap="magma", vmin=0, vmax=1)
        axes[row, 1].set_title("Pfg (per-patch fg fraction)")
        axes[row, 1].axis("off")
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046)

        cat = np.zeros((*diag["own_frac"].shape, 3))
        cat[diag["step1_fg_flat"].reshape(diag["own_frac"].shape)] = (0.2, 0.8, 0.2)  # fg = green
        cat[diag["step1_bg_flat"].reshape(diag["own_frac"].shape)] = (0.8, 0.2, 0.2)  # bg = red
        rejected = ~diag["step1_fg_flat"] & ~diag["step1_bg_flat"]
        cat[rejected.reshape(diag["own_frac"].shape)] = (0.5, 0.5, 0.5)  # mixed/rejected = gray
        axes[row, 2].imshow(cat)
        n_fg, n_bg = diag["step1_fg_flat"].sum(), diag["step1_bg_flat"].sum()
        n_rej = rejected.sum()
        axes[row, 2].set_title(f"fg={n_fg} bg={n_bg} rejected={n_rej}")
        axes[row, 2].axis("off")
    fig.suptitle(f"Step 1: spatial filter (mixed-patch rejection) — focus combo {focus_key}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path("step1_spatial_filter"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Visualization 2 — Step 2 (attention check), all three scales, both branches. Applied
    # directly to `raw` fg (not step 1's output — see the file header), so *idx* here is
    # raw_fg_idx, not step1_fg_idx.
    fig, axes = plt.subplots(len(SCALES), 4, figsize=(17, 4.3 * len(SCALES)))
    for row, scale in enumerate(SCALES):
        diag = scale_diag[scale]
        gh, gw = diag["grid_h"], diag["grid_w"]
        idx = diag["raw_fg_idx"]

        cls_grid = sims_to_grid(idx, diag["cls_sims"], gh, gw)
        im = axes[row, 0].imshow(cls_grid, cmap="viridis", vmin=-1, vmax=1)
        axes[row, 0].set_title(f"scale={scale}: cos-sim to close-crop [CLS]")
        axes[row, 0].axis("off")
        plt.colorbar(im, ax=axes[row, 0], fraction=0.046)

        cls_keep_idx = idx if diag["cls_keep"] is None else idx[diag["cls_keep"]]
        overlay_patches(axes[row, 1], diag["img"], cls_keep_idx, gh, gw, "#2ecc71")
        axes[row, 1].set_title(f"retained by CLS check (n={len(cls_keep_idx)})")

        center_grid = sims_to_grid(idx, diag["center_sims"], gh, gw)
        im = axes[row, 2].imshow(center_grid, cmap="viridis", vmin=-1, vmax=1)
        axes[row, 2].set_title("cos-sim to center prototype")
        axes[row, 2].axis("off")
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046)

        center_keep_idx = idx if diag["center_keep"] is None else idx[diag["center_keep"]]
        overlay_patches(axes[row, 3], diag["img"], center_keep_idx, gh, gw, "#3498db")
        axes[row, 3].set_title(f"retained by center check (n={len(center_keep_idx)})")
    fig.suptitle(f"Step 2: DINO attention check — focus combo {focus_key}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path("step2_attention_check"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Histogram of similarity distributions + the ATTENTION_KEEP_FRACTION cutoff, per scale
    fig, axes = plt.subplots(1, len(SCALES), figsize=(6 * len(SCALES), 4.5), sharey=True)
    for ax, scale in zip(axes, SCALES):
        diag = scale_diag[scale]
        if diag["cls_sims"] is not None:
            cutoff = np.quantile(diag["cls_sims"], 1.0 - ATTENTION_KEEP_FRACTION)
            ax.hist(diag["cls_sims"], bins=20, alpha=0.6, label="cls", color="#2ecc71")
            ax.axvline(cutoff, color="#2ecc71", linestyle="--", linewidth=1)
        if diag["center_sims"] is not None:
            cutoff = np.quantile(diag["center_sims"], 1.0 - ATTENTION_KEEP_FRACTION)
            ax.hist(diag["center_sims"], bins=20, alpha=0.6, label="center", color="#3498db")
            ax.axvline(cutoff, color="#3498db", linestyle="--", linewidth=1)
        ax.set_title(f"scale={scale}")
        ax.set_xlabel("cosine similarity (dashed = keep-fraction cutoff)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("raw fg patch count")
    fig.suptitle(f"Step 2: attention-check similarity distributions — focus combo {focus_key}")
    fig.tight_layout()
    fig.savefig(out_path("step2_similarity_histograms"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Visualization 3 — Step 3 (HDBSCAN + kNN consensus), all three scales, applied
    # directly to each scale's own `raw` fg patches (not step 1/2's output — see the file
    # header), plus a group-level PCA scatter showing where the focus combo's own raw fg
    # patches sit relative to the rest of its instance-type group's pooled cloud.
    fig, axes = plt.subplots(1, len(SCALES), figsize=(6 * len(SCALES), 5.5))
    for ax, scale in zip(axes, SCALES):
        diag = scale_diag[scale]
        gh, gw = diag["grid_h"], diag["grid_w"]
        base_idx = diag["raw_fg_idx"]
        kept_idx = step3_kept_flat_idx(focus_key, scale_diag, step3_diag, scale)
        dropped_idx = np.setdiff1d(base_idx, kept_idx)
        ax.imshow(diag["img"])
        w, h = diag["img"].size
        ph, pw = h / gh, w / gw
        for idx_set, color in ((kept_idx, "#2ecc71"), (dropped_idx, "#e74c3c")):
            grid = flat_idx_to_bool_grid(idx_set, gh, gw)
            ys, xs = np.where(grid)
            for y, x in zip(ys, xs):
                ax.add_patch(
                    plt.Rectangle(
                        (x * pw, y * ph), pw, ph, facecolor=color, edgecolor="none", alpha=0.45
                    )
                )
        ax.axis("off")
        ax.set_title(f"scale={scale}: kept={len(kept_idx)} dropped={len(dropped_idx)}")
    fig.suptitle(
        f"Step 3: HDBSCAN + kNN consensus on raw fg (green=kept, red=dropped) — focus combo "
        f"{focus_key}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path("step3_feature_clean"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    if step3_diag is None:
        ax.set_title("no diagnostics captured")
        ax.axis("off")
    else:
        pooled_2d = PCA(n_components=2, random_state=SEED).fit_transform(step3_diag["pooled"])
        keep = step3_diag["keep"]
        ax.scatter(
            pooled_2d[~keep, 0],
            pooled_2d[~keep, 1],
            s=10,
            alpha=0.4,
            color="#e74c3c",
            label="dropped",
        )
        ax.scatter(
            pooled_2d[keep, 0], pooled_2d[keep, 1], s=10, alpha=0.4, color="#2ecc71", label="kept"
        )
        ck_start, ck_end = next((s, e) for ck, s, e in step3_diag["slices"] if ck == focus_key)
        ax.scatter(
            pooled_2d[ck_start:ck_end, 0],
            pooled_2d[ck_start:ck_end, 1],
            s=60,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            label="focus combo",
        )
        ax.set_title(f"group={focus_combo['group']} pooled raw-fg tokens (PCA)")
        ax.legend(fontsize=8)
    fig.suptitle(f"Step 3: per-group pooled feature space — group={focus_combo['group']}")
    fig.tight_layout()
    fig.savefig(out_path("step3_group_pca"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Visualization 4 — pipeline summary: every stage, all three scales, one grid
    fig, axes = plt.subplots(
        len(SCALES), len(STAGES), figsize=(3.1 * len(STAGES), 3.6 * len(SCALES))
    )
    for row, scale in enumerate(SCALES):
        diag = scale_diag[scale]
        gh, gw = diag["grid_h"], diag["grid_w"]
        for col, stage in enumerate(STAGES):
            ax = axes[row, col]
            if stage == "raw":
                idx = diag["raw_fg_idx"]
            elif stage == "step1":
                idx = diag["step1_fg_idx"]
            elif stage in ("step2_cls", "step2_center"):
                branch = stage.split("_")[1]
                base_idx = diag["raw_fg_idx"]
                branch_keep = diag[f"{branch}_keep"]
                idx = base_idx if branch_keep is None else base_idx[branch_keep]
            else:  # "step3"
                idx = step3_kept_flat_idx(focus_key, scale_diag, step3_diag, scale)
            overlay_patches(ax, diag["img"], idx, gh, gw, "#2ecc71")
            if row == 0:
                ax.set_title(f"{STAGE_LABELS[stage]}\nn={len(idx)}", fontsize=9)
            else:
                ax.set_title(f"n={len(idx)}", fontsize=9)
        axes[row, 0].text(
            -0.15,
            0.5,
            scale,
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
        )
    fig.suptitle(f"Pipeline summary: surviving fg patches per stage — focus combo {focus_key}")
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))
    fig.savefig(out_path("pipeline_summary"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# %% Render qualitative figures for every focus combo
for _focus_combo in focus_combos:
    render_focus_qualitative_figures(
        _focus_combo,
        focus_scale_diag[combo_key(_focus_combo)],
        step3_diagnostics_by_focus[combo_key(_focus_combo)],
    )
log.info("Saved every qualitative figure for %d focus combos to %s", len(focus_combos), OUTPUT_DIR)

# %% [markdown]
# ## Reading the results
#
# `oracle_iou_by_stage.png` / the logged summary answer the quantitative question: does
# each technique, applied *on its own* to the `raw` gallery, move the needle on
# localization, for `proto` and `knn_fgbg` independently? Because this is an ablation
# against a fixed `raw` baseline rather than a cascade, every stage's bar is directly
# comparable back to `raw` — a stage below `raw` removed real signal along with noise (or
# over-filtered a gallery down to too few patches for a stable mean/kNN estimate — check
# `n_combos` and the Part 5/6/7 warning logs for how often a fallback fired); a stage above
# `raw` is genuinely cleaning noise the 0.3-threshold baseline was picking up. The stages
# are *not* directly comparable to each other in a "step2 built on step1" sense — there is
# no such dependency to read into the chart. `step1_spatial_filter__<slug>.png`,
# `step2_attention_check__<slug>.png` / `step2_similarity_histograms__<slug>.png`,
# `step3_feature_clean__<slug>.png` / `step3_group_pca__<slug>.png`, and
# `pipeline_summary__<slug>.png` (one set per focus combo — see FOCUS_COMBOS_SPEC) show
# *where* each technique's cleaning actually happened on that instance (each computed
# independently from `raw`, side by side), which is the qualitative half of that same
# question — a technique can look aggressive in the summary grid while barely moving oracle
# IoU (its removed patches weren't hurting localization much) or vice versa.
#
# **One instance is not the dataset — this is why three focus combos are rendered, not
# one.** Part 6 logs "Step 3 survival by scale, aggregated across every combo" right after
# the per-group HDBSCAN + kNN-consensus pass — read that alongside the Step-3 figures, not
# instead of them. With the first FOCUS_COMBOS_SPEC entry (`LHa`/`donut foam single`/1),
# `step3_feature_clean__LHa_donut_foam_single_1.png`/`pipeline_summary__LHa_donut_foam_
# single_1.png` show global and mid *completely* wiped out (0/3 and 0/9 patches survive) —
# which looks like Step 3 systematically can't be trusted below "close" scale. The aggregate
# log line says otherwise: across all 30 combos, global keeps ~83%, mid ~89%, close ~94% —
# and the *same* object class on a different part type (`LHb`/`donut foam single`/1) keeps
# 100% of its global and mid patches at nearly the same raw patch counts (3/3, 8/8). The
# `velcro` and `white clips` focus combos added alongside the original give two more
# concrete data points on this same question without re-running the whole sweep. The
# `LHa`/`donut foam single`/1 combo shown in most detail above is a genuine outlier, not
# representative of what Step 3 does to global/mid scales generally — which is also the
# reason `oracle_iou_by_stage.png`'s `step3` bar isn't noticeably worse than `raw`:
# dataset-wide, `step3`'s pooled fg gallery is nowhere near as close-scale-only as any single
# combo's figures alone would suggest.
#
# Natural follow-ups this file doesn't attempt: tuning ATTENTION_KEEP_FRACTION/
# CENTER_CORE_PERCENTILE/HDBSCAN_MIN_CLUSTER_SIZE/KNN_CONSENSUS_MIN_AGREEMENT against a
# held-out split rather than the fixed defaults used throughout; combining the `cls` and
# `center` branches (e.g. requiring both to agree) instead of comparing them independently;
# and reusing `multiscale_crop_ablation.py`'s full DBSCAN + greedy-match pipeline on top of
# whichever stage wins here, to see whether a cleaner gallery also improves instance-level
# precision/recall/count-error, not just oracle IoU. Part 10 below picks up the other
# natural follow-up — actually chaining the techniques into a real cascade — that this
# section's ablation deliberately left untested.

# %% [markdown]
# ## Part 10 — composed pipeline: does chaining the steps help, and which step matters?
#
# Every result above is an **ablation against a fixed `raw` baseline**: step1/step2_cls/
# step2_center/step3 each filter `raw`'s own foreground independently, so none of them
# compound (see the file header). That isolates each technique's own marginal effect, but it
# can't answer a different, equally natural question: if you actually build a real cleaning
# *pipeline* — apply spatial filter, then attention check, then feature clean, each stage
# consuming the previous stage's output — does the composition help, and does every stage in
# it earn its place?
#
# This section builds exactly that pipeline (`PIPELINE_STEPS`, in order: step1 -> step2 ->
# step3) and then runs a **leave-one-out evaluation**: the full 3-step pipeline, plus three
# variants each omitting exactly one step (keeping the other two in the same relative
# order), scored the same way as Part 7/8 (oracle IoU, `proto` and `knn_fgbg`,
# global+mid+close/all). Comparing a leave-one-out variant against the full pipeline
# isolates that step's marginal contribution *in composition* — which can differ from its
# isolated contribution against `raw` above if the steps interact (e.g. step1 removing
# boundary patches before step3 pools across combos changes what HDBSCAN sees, in a way the
# isolated step3 ablation never exercises since it pools `raw` fg directly). `raw` (no
# steps at all) is included in the chart as the ground reference, reusing Part 7's
# already-computed scores rather than recomputing them.
#
# Step 2 has two independent reference branches ("cls" and "center", see the file header)
# that scored similarly *in isolation* (Part 8) — but that doesn't guarantee they behave the
# same once step2 is consuming step1's already-thinned foreground instead of `raw`'s, so
# `PIPELINE_STEP2_BRANCHES = ["cls", "center"]` runs the entire leave-one-out sweep once per
# branch, producing two independent charts (`cascade_leave_one_out__cls.png`,
# `cascade_leave_one_out__center.png`) rather than picking one branch as "the" pipeline.
# Each bar is labeled with its own oracle-IoU score, and each variant is additionally
# labeled (once, above both method bars) with the total number of foreground patches that
# survived into its gallery, summed across every combo — the two numbers together show not
# just whether a step's removal changed the score, but how much gallery it was keeping or
# discarding to get there.

# %% Part 10a — composed-pipeline helpers: recompute each scale's (tokens, own_frac) and
# Step 2's reference directly from persisted encodings (Parts 2-4), independently of Part
# 5's (deliberately-uncascaded) `procs` list, so the cascade below has no hidden dependency
# on the ablation's own intermediate state.


def get_scale_tokens_and_frac(
    combo: dict, scale: str
) -> tuple[torch.Tensor, int, int, np.ndarray] | None:
    """This combo's own (tokens, grid_h, grid_w, own_frac) at *scale* — None if this combo
    doesn't have that scale (e.g. 'close' dropped below MIN_CROP_SIZE, see Part 2)."""
    part_type = combo["part_type"]
    if scale == "global":
        r = ref_encodings[part_type]
        own_frac = patch_fg_fraction(combo["ref_mask"], r["r_h"], r["r_w"], IMG_SIZE)
        return r["r_tokens"], r["r_h"], r["r_w"], own_frac
    crop = combo["crops"].get(scale)
    if crop is None:
        return None
    own_frac = patch_fg_fraction(crop["mask_px"], crop["grid_h"], crop["grid_w"], IMG_SIZE)
    return crop["tokens"], crop["grid_h"], crop["grid_w"], own_frac


def get_step2_reference(combo: dict, scale: str, branch: str) -> torch.Tensor | None:
    """Step 2's reference embedding at *scale* for *branch* ("cls" or "center") — the same
    two references process_scale (Part 5) builds, recomputed here so Part 10's cascade
    doesn't depend on Part 5's discarded per-scale diagnostics."""
    if branch == "cls":
        close_crop = combo["crops"].get("close")
        return close_crop["cls"] if close_crop is not None else None
    inputs = get_scale_tokens_and_frac(combo, scale)
    if inputs is None:
        return None
    tokens, grid_h, grid_w, _ = inputs
    mask_px = combo["ref_mask"] if scale == "global" else combo["crops"][scale]["mask_px"]
    return center_prototype(
        tokens,
        mask_px,
        grid_h,
        grid_w,
        CENTER_CORE_PERCENTILE,
        f"{combo_key(combo)}/{scale}/cascade",
    )


def apply_cascade_step(
    step: str,
    idx: np.ndarray,
    tokens: torch.Tensor,
    own_frac: np.ndarray,
    reference: torch.Tensor | None,
) -> tuple[np.ndarray, torch.Tensor]:
    """Apply one named step ("step1" or "step2") to a (idx, tokens) pair already filtered by
    any earlier step in the cascade — idx indexes this scale's flat own_frac grid, tokens are
    the L2-normalised patch tokens at those same positions, an invariant threaded through the
    whole cascade so any step works regardless of its position in PIPELINE_VARIANTS' order.
    Step 3 is cross-combo and per-group, so it isn't handled here (see build_cascade_fg and
    the pool_and_clean_group call in Part 10b).
    """
    if idx.shape[0] == 0:
        return idx, tokens
    if step == "step1":
        keep = own_frac.reshape(-1)[idx] >= FG_HIGH
    elif step == "step2":
        if reference is None:
            return idx, tokens
        keep = (
            keep_top_fraction_by_similarity(tokens, reference, ATTENTION_KEEP_FRACTION)
            .cpu()
            .numpy()
        )
    else:
        raise ValueError(f"apply_cascade_step: unsupported step {step!r}")
    if not keep.any():
        # A step emptying the cascade mid-pipeline would silently kill every later step too
        # — same "don't let one stage zero everything out" fallback used throughout this
        # file (Part 5's raw-fallback, Part 6's pooled-fallback): keep the pre-step set.
        return idx, tokens
    return idx[keep], tokens[keep]


def build_cascade_fg(combo: dict, steps: list[str], branch: str) -> torch.Tensor:
    """This combo's fg gallery after applying *steps* (a subset of PIPELINE_STEPS, minus
    "step3") to `raw` fg in sequence, scale by scale, then concatenated across scales — the
    cascaded analogue of Steps 1/2's independent-against-raw logic in process_scale."""
    chunks: list[torch.Tensor] = []
    for scale in SCALES:
        inputs = get_scale_tokens_and_frac(combo, scale)
        if inputs is None:
            continue
        tokens, grid_h, grid_w, own_frac = inputs
        raw_flat = (own_frac >= MASK_PATCH_THRESHOLD).reshape(-1)
        idx = np.flatnonzero(raw_flat)
        cur_tokens = tokens[torch.from_numpy(raw_flat).to(tokens.device)]
        for step in steps:
            if step == "step3":
                continue  # handled per-group, after every combo's steps 1/2 run (Part 10b)
            reference = get_step2_reference(combo, scale, branch) if step == "step2" else None
            idx, cur_tokens = apply_cascade_step(step, idx, cur_tokens, own_frac, reference)
        chunks.append(cur_tokens)
    return torch.cat(chunks, dim=0)


# %% Part 10b — build & score every pipeline variant (full + 3 leave-one-out ablations),
# once per Step-2 branch — "cls" and "center" scored similarly in Part 8's isolated
# ablation, but that doesn't guarantee they behave the same *in composition* (step2 sees
# step1's already-thinned fg here, not raw's), so each branch gets its own run and its own
# chart rather than picking one as "the" default.
PIPELINE_STEP2_BRANCHES: list[str] = ["cls", "center"]

PIPELINE_STEPS: list[str] = ["step1", "step2", "step3"]  # canonical order every variant
# below respects (a variant only ever drops a step, never reorders the ones it keeps)
PIPELINE_VARIANTS: dict[str, list[str]] = {
    "full": ["step1", "step2", "step3"],
    "no_step1": ["step2", "step3"],
    "no_step2": ["step1", "step3"],
    "no_step3": ["step1", "step2"],
}
PIPELINE_VARIANT_LABELS: dict[str, str] = {
    "full": "full (1+2+3)",
    "no_step1": "no step1 (2+3)",
    "no_step2": "no step2 (1+3)",
    "no_step3": "no step3 (1+2)",
}
CASCADE_CHART_ORDER: list[str] = ["raw", "full", "no_step1", "no_step2", "no_step3"]
CASCADE_CHART_LABELS: dict[str, str] = {"raw": "raw (no steps)", **PIPELINE_VARIANT_LABELS}


def run_composed_pipeline(branch: str) -> pd.DataFrame:
    """Build, score, and plot the full leave-one-out sweep (Part 10b+10c) for one Step-2
    branch. Returns the per-(method, variant) summary dataframe (mean/std oracle IoU,
    n_combos) — kept so a later cell could compare branches without re-running everything.
    """
    log.info(
        "Part 10 (branch=%s): composed pipeline steps=%s, variants=%s",
        branch,
        PIPELINE_STEPS,
        list(PIPELINE_VARIANTS),
    )

    # Every variant's fg gallery *before* Step 3 (Step 3 needs cross-combo pooling per
    # group, so it's applied separately, right below, exactly like Part 6 does for the
    # isolated ablation).
    cascade_pre_step3: dict[str, dict[tuple, torch.Tensor]] = {
        variant: {
            combo_key(c): build_cascade_fg(c, [s for s in steps if s != "step3"], branch)
            for c in combos
        }
        for variant, steps in tqdm(
            PIPELINE_VARIANTS.items(), desc=f"Part 10 ({branch}): steps 1-2 per variant"
        )
    }

    # Step 3, per group, for every variant that includes it — reuses pool_and_clean_group
    # (Part 6's own helper) by staging each variant's pre-step3 fg into combo_galleries
    # under a private stage key, exactly the interface pool_and_clean_group already
    # expects; the key is removed again once that variant's pooling is done so
    # combo_galleries doesn't accumulate scratch state across variants or branches.
    cascade_final_fg: dict[str, dict[tuple, torch.Tensor]] = {}
    for variant, steps in PIPELINE_VARIANTS.items():
        if "step3" not in steps:
            cascade_final_fg[variant] = cascade_pre_step3[variant]
            continue
        stage_key = f"_cascade_pre3_{branch}_{variant}"
        for c in combos:
            ck = combo_key(c)
            combo_galleries[ck][stage_key] = {"fg": cascade_pre_step3[variant][ck]}
        variant_result: dict[tuple, torch.Tensor] = {}
        for group, group_combos in combos_by_group.items():
            pooled_result, _ = pool_and_clean_group(
                group_combos, combo_galleries, stage_key, capture=False
            )
            variant_result.update(pooled_result)
        for c in combos:
            ck = combo_key(c)
            if ck not in variant_result:
                log.warning(
                    "%s branch=%s variant=%s: absent from pooled result (empty pre-step3 "
                    "fg) — falling back to that combo's own pre-step3 fg unfiltered",
                    ck,
                    branch,
                    variant,
                )
                variant_result[ck] = cascade_pre_step3[variant][ck]
            del combo_galleries[ck][stage_key]
        cascade_final_fg[variant] = variant_result

    # Score every (combo, variant, method) the same way Part 7 scores every (combo, stage,
    # method) — bg comes from combo_galleries' already-computed "step1"/"raw" bg gallery
    # (Steps 2-3 never touch bg, see the file header), matching whichever of those this
    # variant's own steps include.
    cascade_iou_lookup: dict[str, dict[str, dict[tuple, float]]] = {
        m: {v: {} for v in PIPELINE_VARIANTS} for m in METHODS
    }
    for combo in tqdm(combos, desc=f"Part 10 ({branch}): scoring composed pipeline variants"):
        ck = combo_key(combo)
        part_type, group = combo["part_type"], combo["group"]
        q = query_encodings[part_type]
        gt = gt_patch_masks.get((part_type, group))
        if gt is None:
            continue
        for variant, steps in PIPELINE_VARIANTS.items():
            fg = cascade_final_fg[variant][ck]
            bg = (
                combo_galleries[ck]["step1"]["bg"]
                if "step1" in steps
                else combo_galleries[ck]["raw"]["bg"]
            )
            if fg.shape[0] == 0 or bg.shape[0] == 0:
                log.warning(
                    "%s branch=%s variant=%s: empty fg/bg gallery — skipping this (combo, variant)",
                    ck,
                    branch,
                    variant,
                )
                continue
            proto = compute_exemplar_features(fg, mode="mean")
            raw_proto = score_heatmap(q["q_tokens"], proto, q["q_h"], q["q_w"])
            cascade_iou_lookup["proto"][variant][ck] = oracle_iou(
                raw_proto, gt, ORACLE_THRESHOLD_STEPS
            )

            raw_knn = knn_score_heatmap(
                q["q_tokens"], fg, bg, KNN_FGBG_NUM_NEIGHBOURS, q["q_h"], q["q_w"]
            )
            cascade_iou_lookup["knn_fgbg"][variant][ck] = oracle_iou(
                raw_knn, gt, ORACLE_THRESHOLD_STEPS
            )

    log.info(
        "Part 10 (branch=%s) scoring complete: %d combos x %d variants x %d methods",
        branch,
        len(combos),
        len(PIPELINE_VARIANTS),
        len(METHODS),
    )

    # Total surviving fg patch count per variant, summed across every combo actually
    # scored (pooled across all 3 scales, same combos the "proto" mean_iou above is
    # averaged over) — the raw byproduct of how aggressively each variant's own steps
    # filtered the gallery, independent of whether that filtering helped or hurt oracle IoU.
    def total_patches(chart_key: str) -> int:
        if chart_key == "raw":
            return int(
                sum(combo_galleries[ck]["raw"]["fg"].shape[0] for ck in iou_lookup["proto"]["raw"])
            )
        return int(
            sum(
                cascade_final_fg[chart_key][ck].shape[0]
                for ck in cascade_iou_lookup["proto"][chart_key]
            )
        )

    patch_counts: dict[str, int] = {ck: total_patches(ck) for ck in CASCADE_CHART_ORDER}

    # Part 10c — aggregate, bar chart (annotated with score + total surviving patches), and
    # the leave-one-out deltas
    cascade_summary_rows: list[dict] = []
    for method in METHODS:
        for chart_key in CASCADE_CHART_ORDER:
            vals = (
                list(iou_lookup[method]["raw"].values())
                if chart_key == "raw"
                else list(cascade_iou_lookup[method][chart_key].values())
            )
            cascade_summary_rows.append(
                {
                    "method": method,
                    "variant": chart_key,
                    "mean_iou": float(np.mean(vals)) if vals else float("nan"),
                    "std_iou": float(np.std(vals)) if vals else float("nan"),
                    "n_combos": len(vals),
                }
            )
    cascade_summary_df = pd.DataFrame(cascade_summary_rows)

    log.info(
        "Composed-pipeline (branch=%s) oracle-IoU summary (mean +/- std across %d combos):",
        branch,
        len(combos),
    )
    for chart_key in CASCADE_CHART_ORDER:
        parts = []
        for method in METHODS:
            row = cascade_summary_df[
                (cascade_summary_df.variant == chart_key) & (cascade_summary_df.method == method)
            ].iloc[0]
            parts.append(f"{method}={row.mean_iou:.3f}+/-{row.std_iou:.3f} (n={row.n_combos})")
        log.info(
            "  %-16s  patches=%-6d %s",
            CASCADE_CHART_LABELS[chart_key],
            patch_counts[chart_key],
            "  ".join(parts),
        )

    log.info(
        "Branch=%s leave-one-out deltas vs. full pipeline (full_mean - variant_mean; "
        "positive means removing that step *hurt* -> the step helps in composition; "
        "negative means removing it *helped* -> the step hurts in composition):",
        branch,
    )
    for method in METHODS:
        full_mean = cascade_summary_df[
            (cascade_summary_df.variant == "full") & (cascade_summary_df.method == method)
        ]["mean_iou"].iloc[0]
        for variant in ("no_step1", "no_step2", "no_step3"):
            variant_mean = cascade_summary_df[
                (cascade_summary_df.variant == variant) & (cascade_summary_df.method == method)
            ]["mean_iou"].iloc[0]
            delta = full_mean - variant_mean
            log.info(
                "  method=%-9s %-16s delta=%+.3f (full=%.3f, without=%.3f)",
                method,
                PIPELINE_VARIANT_LABELS[variant],
                delta,
                full_mean,
                variant_mean,
            )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(CASCADE_CHART_ORDER))
    width = 0.8 / len(METHODS)
    for i, method in enumerate(METHODS):
        means = [
            cascade_summary_df[
                (cascade_summary_df.variant == v) & (cascade_summary_df.method == method)
            ]["mean_iou"].iloc[0]
            for v in CASCADE_CHART_ORDER
        ]
        stds = [
            cascade_summary_df[
                (cascade_summary_df.variant == v) & (cascade_summary_df.method == method)
            ]["std_iou"].iloc[0]
            for v in CASCADE_CHART_ORDER
        ]
        bars = ax.bar(
            x + i * width,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            label=method,
            color=METHOD_COLOR[method],
        )
        # 4 decimals, not annotate_bar_values' usual 3 — the leave-one-out deltas this
        # chart exists to show are often in the thousandths, so the default precision
        # would print identical-looking labels on bars the log's delta table shows are
        # meaningfully different.
        annotate_bar_values(ax, bars, fmt="%.4f")

    # Total surviving patch count doesn't depend on method (proto/knn_fgbg share the same
    # fg/bg galleries), so it's printed once per variant, above both of that variant's bars
    # rather than duplicated on each one.
    for vi, chart_key in enumerate(CASCADE_CHART_ORDER):
        top = max(
            cascade_summary_df[
                (cascade_summary_df.variant == chart_key) & (cascade_summary_df.method == method)
            ]["mean_iou"].iloc[0]
            + cascade_summary_df[
                (cascade_summary_df.variant == chart_key) & (cascade_summary_df.method == method)
            ]["std_iou"].iloc[0]
            for method in METHODS
        )
        ax.text(
            x[vi] + width * (len(METHODS) - 1) / 2,
            top + 0.03,
            f"n={patch_counts[chart_key]}\npatches",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#444444",
        )

    ax.set_xticks(
        x + width * (len(METHODS) - 1) / 2,
        [CASCADE_CHART_LABELS[v] for v in CASCADE_CHART_ORDER],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("oracle IoU (mean +/- std across combos)")
    ax.set_title(
        f"Composed pipeline — leave-one-out oracle IoU, global+mid+close/all scale combo "
        f"({len(combos)} combos across {len(RUN_PART_TYPES)} part types, step2 branch="
        f"{branch!r})"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 1.12)  # headroom above 1.0 for the per-variant patch-count annotation
    fig.tight_layout()
    out_path = OUTPUT_DIR / f"cascade_leave_one_out__{branch}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved leave-one-out bar chart (branch=%s) to %s", branch, out_path)

    return cascade_summary_df


cascade_results_by_branch: dict[str, pd.DataFrame] = {
    branch: run_composed_pipeline(branch) for branch in PIPELINE_STEP2_BRANCHES
}

# %% [markdown]
# ## Reading the composed-pipeline results
#
# `cascade_leave_one_out__cls.png` / `cascade_leave_one_out__center.png` (one per Step-2
# branch) and the logged deltas answer the follow-up question the ablation above (Parts
# 7-9) can't: not "does step X help against raw in isolation" but "does step X earn its
# place inside an actual pipeline". A positive delta for a step means removing it from the
# full pipeline *hurt* oracle IoU — that step is pulling its weight in composition. A
# negative delta means removing it *helped* — that step is net-harmful once the other two
# have already run (e.g. it might be over-filtering a gallery step1/step2 already thinned
# down, or removing patches step3's pooled clustering needed). Compare each step's
# leave-one-out delta here against its own isolated bar in `oracle_iou_by_stage.png` — a step
# that helped in isolation but shows a near-zero or negative delta here is one whose benefit
# doesn't survive composition, most likely because a later step in the pipeline already
# removes the same noise it targets.
#
# The `n=... patches` annotation above each variant's bars is the other half of that
# reading: a step with a near-zero score delta but a large drop in surviving patch count
# was net-neutral on localization while still discarding most of the gallery — a much
# more aggressive (and riskier, on a dataset with fewer combos or a noisier query) filter
# than the oracle-IoU bar alone would suggest. Comparing the two branches' patch counts for
# the same variant also shows whether "cls" and "center" disagree about *how much* to keep,
# not just *how well* what they keep scores.

# %%
