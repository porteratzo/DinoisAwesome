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
#     vectors) — no custom metric needed. **Off by default** (`ENABLE_STEP3=False`) — it
#     never reliably beat `raw` in this file's own results, and its per-fold cost in the 5-3
#     section below (O(N^2), uncached) dominates this file's runtime; flip it on to
#     re-validate rather than to get a faster run.
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
# Every (part_type, instance-type group, ref instance) combo actually annotated in `data/abc5`
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
from dotenv import load_dotenv
from matplotlib.colors import to_rgb
from PIL import Image
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from tqdm import tqdm

from dinoisawesome import DinoEncoder, EncoderWithCache, compute_exemplar_features, load_annotations
from dinoisawesome.abc3 import INSTANCE_TYPE_GROUPS, available_instance_groups
from dinoisawesome.instance_detection import extract_patch_tokens

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared.abc3_combos import combo_key  # noqa: E402
from _shared.dataset_pairs import REF_QUERY_PAIRS, RefQueryPair  # noqa: E402
from _shared.latency import cuda_timer, images_per_sec  # noqa: E402
from _shared.mask_geometry import patch_fg_fraction, scale_crop_box  # noqa: E402
from _shared.pooled_gallery_cv import (  # noqa: E402
    MAX_BANK_SIZE_DENOISE_53,
    N_FOLDS_53,
    cap_bank_size,
    discover_all_instances,
    make_fold_role_splits,
)
from _shared.prototype_ops import (  # noqa: E402
    extract_patch_tokens_batch_with_cls,
    knn_score_heatmap,
    score_heatmap,
)
from _shared.qualitative_gallery import ScoredExample, save_score_gallery  # noqa: E402
from _shared.run_config import apply_overrides, load_run_config, resolve_output_dir  # noqa: E402
from _shared.stats import bootstrap_ci, bootstrap_prob_greater  # noqa: E402
from _shared.thresholding import achievable_iou, oracle_iou  # noqa: E402

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

DATA_ROOT = _REPO_ROOT / "data"

# abc5 has four ref/query pairs per part type — (1,2)/(3,4)/(5,6)/(7,8) (see
# _shared/dataset_pairs.py). Every unit/instance is swept for the quantitative oracle-IoU
# sweep. Narrow this for fast iteration while developing the script, e.g. REF_QUERY_PAIRS[:1].
RUN_PAIRS: list[RefQueryPair] = REF_QUERY_PAIRS

# Focus combos for every qualitative (spatial-filter/attention-check/feature-clean/pipeline)
# figure — one of them (the first) is the same object as augmented_prototype_oracle_iou_
# knn_fgbg.py's FOCUS_*, for comparability; the other two span the dataset's other
# instance-type groups (see module docstring's "Reading the results" for why one instance
# alone is a bad stand-in for the dataset). Each entry falls back to the first discovered
# combo (with a warning) if not present under RUN_PAIRS.
FOCUS_COMBOS_SPEC: list[tuple[str, str, int]] = [
    ("LHa_1-2", "donut foam single", 1),  # group "donut foam" (original default)
    ("LHb_1-2", "velcro", 1),  # group "velcro" — single annotated instance
    ("RHb_1-2", "white clips", 2),  # group "white clips" — multi-instance class, 2nd instance
]

DINO_VERSION = "v3"
DINO_SIZE = "base"
IMG_SIZE = 768
LAYER_IDX = 11  # last block of ViT-B/16 (depth 12)
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
# This file's own ablation (Parts 7-8) and the 5-3 extension below never showed Step 3
# reliably beating `raw` — gallery cleaning via clustering is a dead end on this dataset —
# while its per-fold HDBSCAN + kNN consensus pass (O(N^2) up to MAX_BANK_SIZE_DENOISE_53
# points) is by far the most expensive part of the 5-3 section: unlike DINO encoding, it
# re-clusters from scratch every fold (n_units_53 = N_FOLDS_53 * (part_type, group) pairs),
# so nothing here is cache-eligible. Off by default; flip to True to re-validate.
ENABLE_STEP3: bool = False
HDBSCAN_MIN_CLUSTER_SIZE = 8
HDBSCAN_MIN_SAMPLES = 3
KNN_CONSENSUS_K = 10
KNN_CONSENSUS_MIN_AGREEMENT = 0.6

# fg-bg-knn scoring (Part 7): k for the per-patch kNN gallery lookup, same default
# multiscale_crop_ablation.py uses for its own "fg-bg-knn(...)" methods.
KNN_FGBG_NUM_NEIGHBOURS = 10

ORACLE_THRESHOLD_STEPS = 25

STAGES: list[str] = ["raw", "step1", "step2_cls", "step2_center"] + (
    ["step3"] if ENABLE_STEP3 else []
)
STAGE_LABELS: dict[str, str] = {
    "raw": "raw (0.3 threshold)",
    "step1": "step1 (spatial filter)",
    "step2_cls": "step2 (CLS check)",
    "step2_center": "step2 (center check)",
    "step3": "step3 (HDBSCAN + kNN)",
}
METHODS: list[str] = ["proto", "knn_fgbg"]
METHOD_COLOR: dict[str, str] = {"proto": "#7f8c8d", "knn_fgbg": "#2ecc71"}

# One representative stage whose individual per-combo scoring results (query crop, raw
# knn_fgbg score map, GT mask, oracle IoU) get kept for the worst/best-N qualitative gallery
# in Part 7 below — collecting this for every stage would multiply memory/disk cost, so only
# "raw" (today's single-threshold baseline, the most decision-relevant single stage to see
# failure modes for) is captured.
QUALITATIVE_STAGE = "raw"
QUALITATIVE_MAX_EXAMPLES = 60  # capped so the gallery figure itself stays a readable size

# Bootstrap settings for the headline CI (Part 8) and the 1-1-vs-5-3 significance check
# (Part 11).
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

SEED = 0

apply_overrides(globals(), load_run_config(__file__))
torch.manual_seed(SEED)

OUTPUT_DIR = resolve_output_dir(_REPO_ROOT / "outputs" / "fundamental_abc5" / "noisy_fgbg_cleaning")

log.info(
    "RUN_PAIRS=%d units (%s)  |  DINO%s-%s img_size=%d layer=%d  |  stages=%s methods=%s",
    len(RUN_PAIRS),
    [p.unit for p in RUN_PAIRS],
    DINO_VERSION,
    DINO_SIZE,
    IMG_SIZE,
    LAYER_IDX,
    STAGES,
    METHODS,
)

# %% Core helpers — combo identity, scoring, oracle IoU


def annotate_bar_values(ax, bars, fmt: str = "%.3f") -> None:
    """Print each bar's own height above it — bar-chart differences in this file are often
    a few hundredths of oracle IoU, too small to read reliably off the y-axis alone."""
    ax.bar_label(bars, fmt=fmt, fontsize=7, padding=2)


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
combos_by_key: dict[tuple, dict] = {combo_key(c): c for c in combos}
combos_by_group: dict[str, list[dict]] = defaultdict(list)
for c in combos:
    combos_by_group[c["group"]].append(c)
log.info(
    "Discovered %d (unit, group, instance) combos across %d units (%d part types), %d groups",
    len(combos),
    len({c["unit"] for c in combos}),
    len({c["part_type"] for c in combos}),
    len(combos_by_group),
)

# %% Focus combos — used for every qualitative figure below (chosen before the main sweep so
# Part 5/6 know when to capture extra diagnostics for them)


def resolve_focus_combo(unit: str, cls: str, instance_id: int) -> dict:
    combo = next(
        (
            c
            for c in combos
            if c["unit"] == unit and c["class"] == cls and c["instance_id"] == instance_id
        ),
        None,
    )
    if combo is None:
        combo = combos[0]
        log.warning(
            "Focus combo unit=%s class=%r instance_id=%d not found under "
            "RUN_PAIRS=%s — falling back to %s",
            unit,
            cls,
            instance_id,
            [p.unit for p in RUN_PAIRS],
            combo_key(combo),
        )
    return combo


focus_combos: list[dict] = []
focus_keys: set[tuple] = set()
for _unit, _cls, _instance_id in FOCUS_COMBOS_SPEC:
    _combo = resolve_focus_combo(_unit, _cls, _instance_id)
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
    ref_img = ref_images[combo["unit"]]
    group_mask = group_ref_masks.get((combo["unit"], combo["group"]), combo["ref_mask"])
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

latency_rows: list[dict] = []

query_encodings: dict[str, dict] = {}
with cuda_timer() as t_query_encode:
    for unit in tqdm(sorted(query_images), desc="Encoding query images"):
        q_tokens, q_h, q_w = extract_patch_tokens(
            encoder, query_images[unit], LAYER_IDX, debias=DEBIAS
        )
        query_encodings[unit] = {"q_tokens": q_tokens, "q_h": q_h, "q_w": q_w}
latency_rows.append(
    {
        "phase": "query_image_encode",
        "elapsed_s": t_query_encode["elapsed_s"],
        "n_units": len(query_images),
        "units_per_sec": images_per_sec(len(query_images), t_query_encode["elapsed_s"]),
    }
)

gt_patch_masks: dict[tuple[str, str], np.ndarray] = {}
for (unit, group), pixel_mask in group_query_masks.items():
    q = query_encodings[unit]
    gt_patch_masks[(unit, group)] = (
        patch_fg_fraction(pixel_mask, q["q_h"], q["q_w"], IMG_SIZE) >= MASK_PATCH_THRESHOLD
    )

# %% Part 3.5 — encode each ref/query unit's full, uncropped reference image once: the
# "global" scale for every combo sharing that unit.
ref_encodings: dict[str, dict] = {}
with cuda_timer() as t_ref_encode:
    for unit in tqdm(sorted(ref_images), desc="Encoding ref images (global scale)"):
        r_tokens, r_h, r_w = extract_patch_tokens(
            encoder, ref_images[unit], LAYER_IDX, debias=DEBIAS
        )
        ref_encodings[unit] = {"r_tokens": r_tokens, "r_h": r_h, "r_w": r_w}
latency_rows.append(
    {
        "phase": "ref_image_encode",
        "elapsed_s": t_ref_encode["elapsed_s"],
        "n_units": len(ref_images),
        "units_per_sec": images_per_sec(len(ref_images), t_ref_encode["elapsed_s"]),
    }
)

# Achievable IoU (Part 7 below) needs the reference/exemplar image's own GT projected to
# patch space, mirroring gt_patch_masks above but for the image that *built* each combo's
# gallery rather than the query being scored — the same "own-GT reference" pattern
# scale_composition_bg_ablation.py's own Part 3 already established for a fixed-pair script
# (the exemplar already has its own GT, so it can stand in for a query-time-labeled
# reference without needing the query's own GT to tune a threshold).
ref_gt_patch_masks: dict[tuple[str, str], np.ndarray] = {}
for (unit, group), pixel_mask in group_ref_masks.items():
    r = ref_encodings[unit]
    ref_gt_patch_masks[(unit, group)] = (
        patch_fg_fraction(pixel_mask, r["r_h"], r["r_w"], IMG_SIZE) >= MASK_PATCH_THRESHOLD
    )

# %% Part 4 — batched-encode every combo's mid/close crops (patch tokens + [CLS] token)
crop_items: list[tuple[tuple, str]] = [
    (combo_key(c), scale) for c in combos for scale in c["crops"]
]
with cuda_timer() as t_crop_encode:
    for i in tqdm(range(0, len(crop_items), chunk_size), desc="Encoding mid/close crops"):
        chunk = crop_items[i : i + chunk_size]
        images = [combos_by_key[ck]["crops"][scale]["img"] for ck, scale in chunk]
        encoded = extract_patch_tokens_batch_with_cls(encoder, images, LAYER_IDX, debias=DEBIAS)
        for (ck, scale), (tokens, cls, grid_h, grid_w) in zip(chunk, encoded):
            crop = combos_by_key[ck]["crops"][scale]
            crop["tokens"], crop["grid_h"], crop["grid_w"] = tokens, grid_h, grid_w
            if scale == "close":
                crop["cls"] = cls
latency_rows.append(
    {
        "phase": "gallery_crop_encode",
        "elapsed_s": t_crop_encode["elapsed_s"],
        "n_units": len(crop_items),
        "units_per_sec": images_per_sec(len(crop_items), t_crop_encode["elapsed_s"]),
    }
)

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
    unit = combo["unit"]
    is_focus = ck in focus_keys

    procs: list[dict] = []
    r = ref_encodings[unit]
    group_mask = group_ref_masks.get((unit, combo["group"]), combo["ref_mask"])
    procs.append(
        process_scale(
            "global",
            ref_images[unit],
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
if ENABLE_STEP3:
    for group, group_combos in tqdm(
        combos_by_group.items(), desc="Part 6: HDBSCAN + kNN consensus"
    ):
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

    # Per-scale Step-3 survival, aggregated across *every* combo (not just the focus one) —
    # see the module docstring's "Reading the results" section for why this matters: the
    # focus combo can (and here, for one specific instance, does) look like Step 3 wipes out
    # an entire scale, when the dataset-wide picture is much less dramatic.
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
else:
    step3_diagnostics_by_focus = {combo_key(c): None for c in focus_combos}
    log.info("Step 3 (HDBSCAN + kNN consensus) disabled — ENABLE_STEP3=False, skipping")

# %% Part 7 — score every combo x every stage x every method (oracle IoU). Alongside oracle_iou
# (tunes its threshold against the query's own GT — an upper bound), also scores the *same*
# gallery against the reference/exemplar image's own full extent + its own GT
# (ref_encodings/ref_gt_patch_masks from Part 3.5) to get achievable_iou: a threshold tuned on
# the exemplar, transferred as-is to the query — the number a deployed pipeline without
# query-time labels would actually see. One extra score_heatmap/knn_score_heatmap call per
# (combo, stage), not per query, since there is exactly one query per combo in this fixed-pair
# script (mirrors scale_composition_bg_ablation.py's own Part 5).
iou_lookup: dict[str, dict[str, dict[tuple, float]]] = {m: {s: {} for s in STAGES} for m in METHODS}
achievable_iou_lookup: dict[str, dict[str, dict[tuple, float]]] = {
    m: {s: {} for s in STAGES} for m in METHODS
}
gt_area_frac_by_key: dict[tuple[str, str], float] = {}
qualitative_examples: list[ScoredExample] = []

with cuda_timer() as t_scoring_1_1:
    for combo in tqdm(combos, desc="Part 7: scoring"):
        ck = combo_key(combo)
        unit, group = combo["unit"], combo["group"]
        q = query_encodings[unit]
        gt = gt_patch_masks.get((unit, group))
        if gt is None:
            continue
        gt_area_frac_by_key[(unit, group)] = float(gt.sum()) / gt.size
        r = ref_encodings[unit]
        ref_gt = ref_gt_patch_masks.get((unit, group))
        if ref_gt is None:
            log.warning("%s: no reference-image GT — achievable_iou left NaN", ck)
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

            if ref_gt is not None:
                ref_raw_proto = score_heatmap(r["r_tokens"], proto, r["r_h"], r["r_w"])
                achievable_iou_lookup["proto"][stage][ck] = achievable_iou(
                    ref_raw_proto, ref_gt, raw_proto, gt, ORACLE_THRESHOLD_STEPS
                )
                ref_raw_knn = knn_score_heatmap(
                    r["r_tokens"], fg, bg, KNN_FGBG_NUM_NEIGHBOURS, r["r_h"], r["r_w"]
                )
                achievable_iou_lookup["knn_fgbg"][stage][ck] = achievable_iou(
                    ref_raw_knn, ref_gt, raw_knn, gt, ORACLE_THRESHOLD_STEPS
                )

            if stage == QUALITATIVE_STAGE and len(qualitative_examples) < QUALITATIVE_MAX_EXAMPLES:
                qualitative_examples.append(
                    ScoredExample(
                        label=f"{unit}/{group}/{combo['class']}#{combo['instance_id']}",
                        image=query_images[unit],
                        raw=raw_knn,
                        gt=gt,
                        score=iou_lookup["knn_fgbg"][stage][ck],
                    )
                )
latency_rows.append(
    {
        "phase": "scoring_1_1",
        "elapsed_s": t_scoring_1_1["elapsed_s"],
        "n_units": len(combos),
        "units_per_sec": images_per_sec(len(combos), t_scoring_1_1["elapsed_s"]),
    }
)

log.info(
    "Scoring complete: %d combos x %d stages x %d methods", len(combos), len(STAGES), len(METHODS)
)

_per_combo_rows = [
    {
        "method": method,
        "stage": stage,
        "unit": ck[0],
        "group": ck[1],
        "class": ck[2],
        "instance_id": ck[3],
        "oracle_iou": iou,
        "achievable_iou": achievable_iou_lookup[method][stage].get(ck, float("nan")),
        "gt_area_frac": gt_area_frac_by_key.get((ck[0], ck[1]), float("nan")),
    }
    for method, by_stage in iou_lookup.items()
    for stage, by_ck in by_stage.items()
    for ck, iou in by_ck.items()
]
pd.DataFrame(_per_combo_rows).to_csv(OUTPUT_DIR / "oracle_iou_per_combo.csv", index=False)
log.info("Wrote %s (%d rows)", OUTPUT_DIR / "oracle_iou_per_combo.csv", len(_per_combo_rows))

# Worst/best-N qualitative gallery at one representative stage (QUALITATIVE_STAGE, knn_fgbg
# method) — every figure above averages across combos; this shows actual individual query
# images so a failure mode is visible instead of washed out by the mean.
if qualitative_examples:
    save_score_gallery(
        qualitative_examples,
        OUTPUT_DIR / "qualitative_worst_best.png",
        n=5,
        score_name="oracle_iou (knn_fgbg)",
        title=f"Worst/best oracle_iou examples: stage={QUALITATIVE_STAGE} method=knn_fgbg (1-1)",
    )
    log.info(
        "Wrote %s (%d examples)",
        OUTPUT_DIR / "qualitative_worst_best.png",
        len(qualitative_examples),
    )
else:
    log.warning("No qualitative examples collected for stage=%s", QUALITATIVE_STAGE)

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
            achievable_vals = [
                v
                for ck, v in achievable_iou_lookup[method][stage].items()
                if combo_keys is None or ck in combo_keys
            ]
            # Percentile bootstrap CI on the mean, alongside the plain std this script
            # already reported — std alone doesn't say whether e.g. `raw`'s and a cleaned
            # stage's means are actually distinguishable or both plausible draws from the
            # same underlying distribution; see _shared/stats.py.
            if vals:
                _, ci_lo, ci_hi = bootstrap_ci(
                    np.array(vals), n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
                )
            else:
                ci_lo = ci_hi = float("nan")
            rows.append(
                {
                    "method": method,
                    "stage": stage,
                    "mean_iou": float(np.mean(vals)) if vals else float("nan"),
                    "std_iou": float(np.std(vals)) if vals else float("nan"),
                    "ci95_lo": ci_lo,
                    "ci95_hi": ci_hi,
                    "mean_achievable_iou": (
                        float(np.mean(achievable_vals)) if achievable_vals else float("nan")
                    ),
                    "std_achievable_iou": (
                        float(np.std(achievable_vals)) if achievable_vals else float("nan")
                    ),
                    "oracle_minus_achievable_gap": (
                        float(np.mean(vals) - np.mean(achievable_vals))
                        if vals and achievable_vals
                        else float("nan")
                    ),
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
summary_df.to_csv(OUTPUT_DIR / "oracle_iou_by_stage.csv", index=False)
log.info("Wrote %s", OUTPUT_DIR / "oracle_iou_by_stage.csv")

_aggregate_chart_path = OUTPUT_DIR / "oracle_iou_by_stage.png"
plot_oracle_iou_bar_chart(
    summary_df,
    f"Noisy fg/bg cleaning — 1-1 (single ref/query pair) — oracle IoU per stage, "
    f"global+mid+close/all scale combo ({len(combos)} combos across {len(RUN_PAIRS)} "
    f"ref/query units)",
    _aggregate_chart_path,
)
log.info("Saved oracle-IoU bar chart to %s", _aggregate_chart_path)

# %% Part 8b — per-instance-type (group) breakdown. The aggregate chart above pools every
# group together, which can hide a group-specific effect the same way a single focus
# combo's Step-3 figures can misrepresent the dataset-wide average (see "Reading the
# results" below) — one bar chart per instance-type group makes that visible directly in
# the oracle-IoU numbers, rather than relying on the qualitative figures plus the Part 6
# per-scale survival log to catch it.
_group_summary_frames = []
for _group in sorted(combos_by_group):
    _group_keys = {combo_key(c) for c in combos_by_group[_group]}
    _group_summary_df = stage_method_summary(_group_keys)

    log.info(
        "Oracle-IoU summary for group=%r (mean +/- std across %d combos):",
        _group,
        len(_group_keys),
    )
    log_stage_method_summary(_group_summary_df)
    _group_summary_df.insert(0, "group", _group)
    _group_summary_frames.append(_group_summary_df)

    _group_chart_path = OUTPUT_DIR / f"oracle_iou_by_stage__{_group.replace(' ', '_')}.png"
    plot_oracle_iou_bar_chart(
        _group_summary_df,
        f"Noisy fg/bg cleaning — 1-1 (single ref/query pair) — oracle IoU per stage, "
        f"group={_group!r} ({len(_group_keys)} combos)",
        _group_chart_path,
    )
    log.info("Saved oracle-IoU bar chart for group=%r to %s", _group, _group_chart_path)

pd.concat(_group_summary_frames, ignore_index=True).to_csv(
    OUTPUT_DIR / "oracle_iou_by_stage_per_group.csv", index=False
)
log.info("Wrote %s", OUTPUT_DIR / "oracle_iou_by_stage_per_group.csv")

# %% Part 8c — oracle IoU (upper bound, tunes the threshold against the query's own GT) vs.
# achievable IoU (a threshold tuned on the reference/exemplar image's own GT, transferred
# as-is to the query — what a deployed pipeline without query-time labels would actually
# get). Every figure through Part 8b plots oracle_iou only; this is the gap between "what's
# the best any threshold could do" and "what a realistic fixed threshold does," per stage.
_stage_x = np.arange(len(STAGES))
_stage_labels = [STAGE_LABELS[s] for s in STAGES]
fig, ax = plt.subplots(figsize=(11, 5.5))
for method in METHODS:
    oracle_means = [
        summary_df[(summary_df.stage == s) & (summary_df.method == method)]["mean_iou"].iloc[0]
        for s in STAGES
    ]
    achievable_means = [
        summary_df[(summary_df.stage == s) & (summary_df.method == method)][
            "mean_achievable_iou"
        ].iloc[0]
        for s in STAGES
    ]
    ax.plot(
        _stage_x,
        oracle_means,
        marker="o",
        linestyle="-",
        color=METHOD_COLOR[method],
        label=f"{method} oracle",
    )
    ax.plot(
        _stage_x,
        achievable_means,
        marker="^",
        linestyle="--",
        color=METHOD_COLOR[method],
        alpha=0.6,
        label=f"{method} achievable",
    )
ax.set_xticks(_stage_x)
ax.set_xticklabels(_stage_labels, rotation=20, ha="right")
ax.set_xlabel("stage")
ax.set_ylabel("mean IoU across combos")
ax.set_ylim(0, 1.0)
ax.set_title(
    "1-1 (single ref/query pair) — Oracle (upper bound) vs. achievable "
    "(exemplar-tuned threshold) IoU, per stage"
)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "oracle_vs_achievable.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Oracle-vs-achievable gap by stage (mean oracle_iou - mean achievable_iou):")
for _, row in summary_df.iterrows():
    log.info(
        "  stage=%-14s method=%-9s gap=%.3f (oracle=%.3f achievable=%.3f)",
        row.stage,
        row.method,
        row.oracle_minus_achievable_gap,
        row.mean_iou,
        row.mean_achievable_iou,
    )
log.info("Saved %s", OUTPUT_DIR / "oracle_vs_achievable.png")

# %% Part 8d — does oracle IoU correlate with object size? An aggregate mean (every figure
# above) can hide "cleaning only helps small/large instances" — `gt_area_frac` (the query
# GT's own patch-mask coverage, added to every row of oracle_iou_per_combo.csv in Part 7)
# lets us check, mirroring the pearson/spearman correlation pattern
# `scale_composition_adaptive_oracle.py` already established for instance size vs. optimal
# scale.
_per_combo_df = pd.DataFrame(_per_combo_rows)
size_correlation_rows = []
for method in METHODS:
    for stage in STAGES:
        sub = _per_combo_df[(_per_combo_df.method == method) & (_per_combo_df.stage == stage)]
        if len(sub) < 3:
            continue
        pearson_r, pearson_p = pearsonr(sub["gt_area_frac"], sub["oracle_iou"])
        spearman_r, spearman_p = spearmanr(sub["gt_area_frac"], sub["oracle_iou"])
        size_correlation_rows.append(
            {
                "method": method,
                "stage": stage,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "n_samples": len(sub),
            }
        )
size_correlation_df = pd.DataFrame(size_correlation_rows)
size_correlation_df.to_csv(OUTPUT_DIR / "size_correlation.csv", index=False)

# Object-size terciles (global, computed once across every row so the same size cutoffs
# apply everywhere) x oracle IoU, faceted per stage like the per-group charts above: does the
# smallest third of instances systematically score worse, and does that gap close or widen
# for a cleaned stage vs. `raw`?
try:
    _per_combo_df["size_tercile"] = pd.qcut(
        _per_combo_df["gt_area_frac"], 3, labels=["small", "medium", "large"]
    )
except ValueError:
    log.warning(
        "gt_area_frac has too few distinct values for 3 clean terciles — falling back to "
        "qcut's own duplicate-safe binning (labels become numeric ranges, not small/medium/large)"
    )
    _per_combo_df["size_tercile"] = pd.qcut(_per_combo_df["gt_area_frac"], 3, duplicates="drop")

fig, axes = plt.subplots(1, len(STAGES), figsize=(3.2 * len(STAGES), 5), sharey=True)
for ax, stage in zip(axes, STAGES):
    tercile_means = (
        _per_combo_df[_per_combo_df.stage == stage]
        .groupby(["size_tercile", "method"], observed=True)["oracle_iou"]
        .mean()
        .unstack("method")
    )
    tercile_means.plot(kind="bar", ax=ax, color=[METHOD_COLOR[m] for m in tercile_means.columns])
    ax.set_title(STAGE_LABELS[stage], fontsize=9)
    ax.set_xlabel("size tercile")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("mean oracle IoU")
fig.suptitle("Does object size predict oracle IoU, per stage?")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "size_correlation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Wrote %s and %s", OUTPUT_DIR / "size_correlation.csv", OUTPUT_DIR / "size_correlation.png")

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


MAX_PCA_SCATTER_POINTS = 3000


def subsample_rows(coords: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    """Randomly subsample rows of *coords* down to at most *max_points*.

    For `step3_group_pca`'s dense, alpha-blended background cloud (thousands of pooled
    patch tokens per instance-type group): past a few thousand points at alpha=0.4,
    additional points are already overplotted into indistinguishable density rather than
    adding visible information, so drawing all of them just costs render time for no
    visual difference. The focus-combo's own highlighted points are drawn separately from
    the unsubsampled array and are unaffected by this.
    """
    if coords.shape[0] <= max_points:
        return coords
    rng = np.random.default_rng(seed)
    sel = rng.choice(coords.shape[0], size=max_points, replace=False)
    return coords[sel]


def overlay_patches(
    ax, img: Image.Image, idx: np.ndarray, grid_h: int, grid_w: int, color: str
) -> None:
    """Draw a translucent *color* square over every patch in *idx* on top of *img*.

    Rendered as one rasterized (grid_h, grid_w, 4) RGBA overlay via `imshow` rather than
    one `Rectangle` artist per patch — with global-scale grids running into the thousands
    of foreground patches, per-patch `add_patch` was the dominant cost of this script's
    qualitative figures (each Rectangle is a separate Artist with its own transform).
    Same pixels on screen, no data dropped, an order of magnitude+ faster to render.
    """
    ax.imshow(img)
    w, h = img.size
    grid = flat_idx_to_bool_grid(idx, grid_h, grid_w)
    rgba = np.zeros((grid_h, grid_w, 4), dtype=np.float32)
    rgba[grid] = (*to_rgb(color), 0.45)
    ax.imshow(rgba, extent=(0, w, h, 0), interpolation="nearest")
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
    # Skipped entirely when ENABLE_STEP3=False — step3_diag is always None then, so these
    # figures would just show "everything dropped" rather than anything meaningful.
    if ENABLE_STEP3:
        fig, axes = plt.subplots(1, len(SCALES), figsize=(6 * len(SCALES), 5.5))
        for ax, scale in zip(axes, SCALES):
            diag = scale_diag[scale]
            gh, gw = diag["grid_h"], diag["grid_w"]
            base_idx = diag["raw_fg_idx"]
            kept_idx = step3_kept_flat_idx(focus_key, scale_diag, step3_diag, scale)
            dropped_idx = np.setdiff1d(base_idx, kept_idx)
            ax.imshow(diag["img"])
            w, h = diag["img"].size
            # Rasterized RGBA overlay, not one Rectangle per patch — see `overlay_patches`.
            rgba = np.zeros((gh, gw, 4), dtype=np.float32)
            for idx_set, color in ((kept_idx, "#2ecc71"), (dropped_idx, "#e74c3c")):
                grid = flat_idx_to_bool_grid(idx_set, gh, gw)
                rgba[grid] = (*to_rgb(color), 0.45)
            ax.imshow(rgba, extent=(0, w, h, 0), interpolation="nearest")
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
            dropped_2d = subsample_rows(pooled_2d[~keep], MAX_PCA_SCATTER_POINTS, SEED)
            kept_2d = subsample_rows(pooled_2d[keep], MAX_PCA_SCATTER_POINTS, SEED)
            ax.scatter(
                dropped_2d[:, 0],
                dropped_2d[:, 1],
                s=10,
                alpha=0.4,
                color="#e74c3c",
                label="dropped",
            )
            ax.scatter(
                kept_2d[:, 0], kept_2d[:, 1], s=10, alpha=0.4, color="#2ecc71", label="kept"
            )
            ck_start, ck_end = next(
                (s, e) for ck, s, e in step3_diag["slices"] if ck == focus_key
            )
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
    unit = combo["unit"]
    if scale == "global":
        r = ref_encodings[unit]
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

# canonical order every variant below respects (a variant only ever drops a step, never
# reorders the ones it keeps). step3 is excluded from every variant when ENABLE_STEP3=False
# — "no_step3" would then be identical to "full", so it's dropped from the variant set
# entirely rather than kept as a redundant duplicate.
PIPELINE_STEPS: list[str] = ["step1", "step2"] + (["step3"] if ENABLE_STEP3 else [])
PIPELINE_VARIANTS: dict[str, list[str]] = {
    "full": ["step1", "step2"] + (["step3"] if ENABLE_STEP3 else []),
    "no_step1": ["step2"] + (["step3"] if ENABLE_STEP3 else []),
    "no_step2": ["step1"] + (["step3"] if ENABLE_STEP3 else []),
    **({"no_step3": ["step1", "step2"]} if ENABLE_STEP3 else {}),
}
PIPELINE_VARIANT_LABELS: dict[str, str] = {
    "full": "full (1+2+3)" if ENABLE_STEP3 else "full (1+2)",
    "no_step1": "no step1 (2+3)" if ENABLE_STEP3 else "no step1 (2)",
    "no_step2": "no step2 (1+3)" if ENABLE_STEP3 else "no step2 (1)",
    "no_step3": "no step3 (1+2)",
}
CASCADE_CHART_ORDER: list[str] = ["raw", "full", "no_step1", "no_step2"] + (
    ["no_step3"] if ENABLE_STEP3 else []
)
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
        unit, group = combo["unit"], combo["group"]
        q = query_encodings[unit]
        gt = gt_patch_masks.get((unit, group))
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
    cascade_summary_df["n_patches"] = cascade_summary_df["variant"].map(patch_counts)

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
    leave_one_out_rows: list[dict] = []
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
            leave_one_out_rows.append(
                {
                    "method": method,
                    "removed_step": variant,
                    "full_mean_iou": full_mean,
                    "without_mean_iou": variant_mean,
                    "delta": delta,
                }
            )

    cascade_summary_df.to_csv(OUTPUT_DIR / f"cascade_summary__{branch}.csv", index=False)
    pd.DataFrame(leave_one_out_rows).to_csv(
        OUTPUT_DIR / f"cascade_leave_one_out__{branch}.csv", index=False
    )
    log.info(
        "Wrote %s and %s",
        OUTPUT_DIR / f"cascade_summary__{branch}.csv",
        OUTPUT_DIR / f"cascade_leave_one_out__{branch}.csv",
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
        f"Composed pipeline — 1-1 (single ref/query pair) — leave-one-out oracle IoU, "
        f"global+mid+close/all scale combo ({len(combos)} combos across {len(RUN_PAIRS)} "
        f"ref/query units, step2 branch={branch!r})"
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

# %% Part 11 — 5-3 pooled gallery, cross-validated: does the "no single cleaning step
# carries real signal" finding (Parts 7-8) hold when each stage's gallery is built from 5
# pooled training images instead of one reference image? Reuses `process_scale`,
# `hdbscan_knn_consensus_keep`, `score_heatmap`/`knn_score_heatmap`/`oracle_iou` unchanged —
# only discovery, per-instance crop-building, and fold/role assignment are new (see
# `_shared/pooled_gallery_cv.py` for why folds use a fresh random shuffle rather than a fixed
# image order). Scoped to the isolated per-stage ablation above (Parts 7-8's own question),
# not Part 10's composed leave-one-out pipeline — replicating that too would roughly double
# this file's newest section for a secondary question. The existing 1-1 combos/results above
# are untouched by this section.
#
# Step 3's HDBSCAN + kNN pool is intentionally scoped to just each fold's own pooled training
# instances here, not the whole dataset the way Part 6's 1-1 baseline pools every ref/query
# unit's instances of a group at once — that dataset-wide pool was never really "built from 1
# training image" even in the 1-1 case, so comparing it against a properly-scoped 5-image
# pool keeps this an apples-to-apples "does more training data change what each stage does"
# question, not a mix of that question and "does dataset-wide pooling help".
#
# `combo_galleries` (113 combos x up to 5 stages x fg+bg, never CPU-offloaded in this file —
# see Bug 1's own note in the abc4-merge history for why this script alone gets away with
# staying GPU-resident) is the single largest GPU-resident structure left over from Parts
# 5-10 and nothing below this point reads it again; freeing it before this section's own
# encoding starts is what keeps this section's peak memory close to "this section's data"
# instead of "that data plus the entire 1-1 pipeline's", which is what actually blew the GPU
# budget here on a box also running an unrelated 610MB external process.
# Reassigned rather than `del`d: both names are still referenced inside
# `run_composed_pipeline`'s already-completed calls above (a static lint pass can't see that
# those calls are already done), and dropping the only reference this way still lets Python's
# refcounting free the underlying GPU tensors before `empty_cache()` below.
combo_galleries = None
group_diagnostics = None
torch.cuda.empty_cache()

discovery_53 = discover_all_instances(DATA_ROOT, "abc5", sorted({c["part_type"] for c in combos}))

usable_instances_53: list[dict] = []
for inst in tqdm(discovery_53.instances, desc="5-3: building mid/close crops"):
    img = discovery_53.images[(inst.part_type, inst.image_number)]
    crops: dict = {}
    for scale in CROP_SCALES:
        x0, y0, x1, y1 = scale_crop_box(inst.mask, scale, CROP_PADDING_FRACTION)
        if x1 - x0 < MIN_CROP_SIZE or y1 - y0 < MIN_CROP_SIZE:
            continue
        crops[scale] = {
            "img": img.crop((x0, y0, x1, y1)),
            "mask_px": inst.mask[y0:y1, x0:x1],
            "exclude_mask_px": inst.bg_exclude_mask[y0:y1, x0:x1],
        }
    usable_instances_53.append(
        {
            "part_type": inst.part_type,
            "group": inst.group,
            "image_number": inst.image_number,
            "ref_mask": inst.mask,
            "bg_exclude_mask": inst.bg_exclude_mask,
            "crops": crops,
        }
    )
log.info("5-3: instances %d", len(usable_instances_53))

# Full-image encodings — each instance's own "global" scale source (Part 3.5's per-unit role,
# generalized to per-image), and every image's own query/eval tokens (Part 3's role).
image_encodings_53: dict[tuple[str, int], dict] = {}
with cuda_timer() as t_image_encode_53:
    for key in tqdm(sorted(discovery_53.images), desc="5-3: encoding images"):
        tokens, h, w = extract_patch_tokens(
            encoder, discovery_53.images[key], LAYER_IDX, debias=DEBIAS
        )
        image_encodings_53[key] = {"tokens": tokens, "h": h, "w": w}
latency_rows.append(
    {
        "phase": "image_encode_5_3",
        "elapsed_s": t_image_encode_53["elapsed_s"],
        "n_units": len(discovery_53.images),
        "units_per_sec": images_per_sec(len(discovery_53.images), t_image_encode_53["elapsed_s"]),
    }
)

gt_patch_masks_53: dict[tuple[str, str, int], np.ndarray] = {}
for (part_type, group, n), pixel_mask in discovery_53.gt_masks.items():
    img_enc = image_encodings_53[(part_type, n)]
    gt_patch_masks_53[(part_type, group, n)] = (
        patch_fg_fraction(pixel_mask, img_enc["h"], img_enc["w"], IMG_SIZE) >= MASK_PATCH_THRESHOLD
    )

# Encode every instance's mid/close crops (patch tokens + [CLS]) — same batched pattern as
# Part 4, generalized off combos onto every discovered instance.
crop_items_53: list[tuple[int, str]] = [
    (i, scale) for i, inst in enumerate(usable_instances_53) for scale in inst["crops"]
]
with cuda_timer() as t_crop_encode_53:
    for i in tqdm(
        range(0, len(crop_items_53), chunk_size), desc="5-3: encoding mid/close crops"
    ):
        chunk = crop_items_53[i : i + chunk_size]
        images_chunk = [usable_instances_53[idx]["crops"][scale]["img"] for idx, scale in chunk]
        encoded = extract_patch_tokens_batch_with_cls(
            encoder, images_chunk, LAYER_IDX, debias=DEBIAS
        )
        for (idx, scale), (tokens, cls, grid_h, grid_w) in zip(chunk, encoded):
            crop = usable_instances_53[idx]["crops"][scale]
            crop["tokens"], crop["grid_h"], crop["grid_w"] = tokens, grid_h, grid_w
            if scale == "close":
                crop["cls"] = cls
latency_rows.append(
    {
        "phase": "gallery_crop_encode_5_3",
        "elapsed_s": t_crop_encode_53["elapsed_s"],
        "n_units": len(crop_items_53),
        "units_per_sec": images_per_sec(len(crop_items_53), t_crop_encode_53["elapsed_s"]),
    }
)

# Build each instance's own raw/step1/step2_cls/step2_center galleries — identical logic to
# Part 5, run per discovered instance instead of per ref/query-pair combo.
inst_galleries_53: list[dict] = []
for inst in tqdm(usable_instances_53, desc="5-3: spatial filter + attention check"):
    img_key = (inst["part_type"], inst["image_number"])
    r = image_encodings_53[img_key]
    label = f"5-3/{inst['part_type']}/{inst['group']}/img#{inst['image_number']}"
    procs: list[dict] = [
        process_scale(
            "global",
            discovery_53.images[img_key],
            r["tokens"],
            r["h"],
            r["w"],
            inst["ref_mask"],
            inst["bg_exclude_mask"],
            None,
            label,
        )
    ]
    close_cls = inst["crops"].get("close", {}).get("cls")
    for scale, crop in inst["crops"].items():
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
                label,
            )
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
    for branch in ("cls", "center"):
        chunks = [
            p["raw_fg_tokens"]
            if p[f"{branch}_keep"] is None
            else p["raw_fg_tokens"][p[f"{branch}_keep"]]
            for p in procs
        ]
        fg_cat = torch.cat(chunks, dim=0)
        if fg_cat.shape[0] == 0:
            fg_cat = galleries["raw"]["fg"]
        galleries[f"step2_{branch}"] = {"fg": fg_cat, "bg": galleries["raw"]["bg"]}
    # Moved to CPU here, not kept resident on GPU: with 227 discovered instances (roughly
    # double this file's own 113-combo 1-1 pipeline, since 5-3 discovers every image instead
    # of just RUN_PAIRS' ref images), holding every instance's full stage galleries on GPU
    # simultaneously is exactly the peak-memory mistake `combo_galleries` above never had to
    # make at only 113 combos — moved back to the query's device only at pooling time below.
    for stage_dict in galleries.values():
        stage_dict["fg"] = stage_dict["fg"].cpu()
        stage_dict["bg"] = stage_dict["bg"].cpu()
    inst_galleries_53.append(galleries)

instances_by_pg_53: dict[tuple[str, str], list[int]] = defaultdict(list)
for i, inst in enumerate(usable_instances_53):
    instances_by_pg_53[(inst["part_type"], inst["group"])].append(i)
groups_by_pt_53: dict[str, list[str]] = defaultdict(list)
for pt, g in instances_by_pg_53:
    groups_by_pt_53[pt].append(g)
log.info(
    "5-3: built stage galleries for %d instances across %d (part_type, group) pairs",
    len(usable_instances_53),
    len(instances_by_pg_53),
)

# Cross-validated sweep: for each fold x part_type x group, pool the fold's training
# instances' raw/step1/step2_* galleries, run Step 3's HDBSCAN + kNN consensus scoped to
# just that pool, and score every stage against every eval image in the fold with GT.
fold_splits_53 = make_fold_role_splits(
    sorted(groups_by_pt_53)
)  # truly randomized, not SEED-reproducible
iou_lookup_53: dict[str, dict[str, list[float]]] = {m: {s: [] for s in STAGES} for m in METHODS}

n_units_53 = N_FOLDS_53 * len(groups_by_pt_53)
with cuda_timer() as t_scoring_53, tqdm(
    total=n_units_53, desc="5-3: cross-validated fit + score"
) as pbar:
    for fold_idx, split in enumerate(fold_splits_53):
        for part_type in groups_by_pt_53:
            train_numbers, eval_numbers = split[part_type]
            for group in groups_by_pt_53[part_type]:
                idxs = instances_by_pg_53[(part_type, group)]
                pool_idxs = [
                    i for i in idxs if usable_instances_53[i]["image_number"] in train_numbers
                ]
                if not pool_idxs:
                    continue

                # Capped right after pooling (not per-instance, before) — a 5-image pool has
                # far more patches than a single reference image, and both the per-stage
                # cosine scoring and (especially) Step 3's O(N^2) HDBSCAN + kNN consensus
                # below scale with patch count for no real benefit past MAX_BANK_SIZE_DENOISE_53; see
                # `_shared/pooled_gallery_cv.py`'s docstring.
                pooled: dict[str, dict[str, torch.Tensor]] = {}
                for stage in ("raw", "step1", "step2_cls", "step2_center"):
                    pooled[stage] = {
                        "fg": cap_bank_size(
                            torch.cat(
                                [inst_galleries_53[i][stage]["fg"] for i in pool_idxs], dim=0
                            ),
                            MAX_BANK_SIZE_DENOISE_53,
                            SEED,
                        ),
                        "bg": cap_bank_size(
                            torch.cat(
                                [inst_galleries_53[i][stage]["bg"] for i in pool_idxs], dim=0
                            ),
                            MAX_BANK_SIZE_DENOISE_53,
                            SEED,
                        ),
                    }

                if ENABLE_STEP3:
                    raw_fg_pool = pooled["raw"]["fg"]
                    step3_fg = raw_fg_pool
                    if raw_fg_pool.shape[0] > 0:
                        keep, _ = hdbscan_knn_consensus_keep(
                            raw_fg_pool.cpu().numpy(),
                            HDBSCAN_MIN_CLUSTER_SIZE,
                            HDBSCAN_MIN_SAMPLES,
                            KNN_CONSENSUS_K,
                            KNN_CONSENSUS_MIN_AGREEMENT,
                        )
                        kept = raw_fg_pool[torch.from_numpy(keep)]
                        if kept.shape[0] > 0:
                            step3_fg = kept
                    pooled["step3"] = {"fg": step3_fg, "bg": pooled["raw"]["bg"]}

                for eval_number in eval_numbers:
                    key = (part_type, group, eval_number)
                    if key not in gt_patch_masks_53:
                        continue
                    q = image_encodings_53[(part_type, eval_number)]
                    gt = gt_patch_masks_53[key]
                    for stage in STAGES:
                        fg, bg = pooled[stage]["fg"], pooled[stage]["bg"]
                        if fg.shape[0] == 0 or bg.shape[0] == 0:
                            continue
                        fg_dev, bg_dev = fg.to(q["tokens"].device), bg.to(q["tokens"].device)
                        proto = compute_exemplar_features(fg_dev, mode="mean")
                        raw_proto = score_heatmap(q["tokens"], proto, q["h"], q["w"])
                        iou_lookup_53["proto"][stage].append(
                            oracle_iou(raw_proto, gt, ORACLE_THRESHOLD_STEPS)
                        )
                        raw_knn = knn_score_heatmap(
                            q["tokens"], fg_dev, bg_dev, KNN_FGBG_NUM_NEIGHBOURS, q["h"], q["w"]
                        )
                        iou_lookup_53["knn_fgbg"][stage].append(
                            oracle_iou(raw_knn, gt, ORACLE_THRESHOLD_STEPS)
                        )
            pbar.update(1)
latency_rows.append(
    {
        "phase": "scoring_5_3",
        "elapsed_s": t_scoring_53["elapsed_s"],
        "n_units": n_units_53,
        "units_per_sec": images_per_sec(n_units_53, t_scoring_53["elapsed_s"]),
    }
)

summary_53_rows = []
for method in METHODS:
    for stage in STAGES:
        vals = iou_lookup_53[method][stage]
        summary_53_rows.append(
            {
                "method": method,
                "stage": stage,
                "mean_iou": float(np.mean(vals)) if vals else float("nan"),
                "std_iou": float(np.std(vals)) if vals else float("nan"),
                "n_combos": len(vals),
            }
        )
summary_53_df = pd.DataFrame(summary_53_rows)
summary_53_df.to_csv(OUTPUT_DIR / "oracle_iou_by_stage__5_3.csv", index=False)
log.info("5-3 oracle-IoU summary (mean +/- std, %d-fold CV):", N_FOLDS_53)
log_stage_method_summary(summary_53_df)

comparison_53_rows = []
for method in METHODS:
    for stage in STAGES:
        row_11 = summary_df[(summary_df.stage == stage) & (summary_df.method == method)].iloc[0]
        row_53 = summary_53_df[
            (summary_53_df.stage == stage) & (summary_53_df.method == method)
        ].iloc[0]
        delta = (
            row_53.mean_iou - row_11.mean_iou
            if not (np.isnan(row_11.mean_iou) or np.isnan(row_53.mean_iou))
            else float("nan")
        )
        # Is the 1-1-vs-5-3 gap (this file's most central comparison — is a pooled 5-image
        # gallery actually better than a single reference image, for the same cleaning
        # stage?) real, or combo/fold-to-fold noise? An unpaired bootstrap comparison (see
        # _shared/stats.py) of the two regimes' own per-sample oracle_iou arrays.
        vals_11 = np.array(list(iou_lookup[method][stage].values()))
        vals_53 = np.array(iou_lookup_53[method][stage])
        prob_53_beats_11 = (
            bootstrap_prob_greater(vals_53, vals_11, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
            if len(vals_11) and len(vals_53)
            else float("nan")
        )
        comparison_53_rows.append(
            {
                "method": method,
                "stage": stage,
                "iou_1_1": row_11.mean_iou,
                "std_1_1": row_11.std_iou,
                "iou_5_3": row_53.mean_iou,
                "std_5_3": row_53.std_iou,
                "delta": delta,
                "prob_5_3_beats_1_1": prob_53_beats_11,
                "n_1_1": len(vals_11),
                "n_5_3": len(vals_53),
            }
        )
        log.info(
            "  %-14s %-10s 1-1=%.3f+/-%.3f  5-3=%.3f+/-%.3f  delta=%+.3f  "
            "P(5-3 beats 1-1)=%.3f",
            stage,
            method,
            row_11.mean_iou,
            row_11.std_iou,
            row_53.mean_iou,
            row_53.std_iou,
            delta,
            prob_53_beats_11,
        )
comparison_53_df = pd.DataFrame(comparison_53_rows)
comparison_53_df.to_csv(OUTPUT_DIR / "comparison_1_1_vs_5_3.csv", index=False)

fig, axes = plt.subplots(1, len(METHODS), figsize=(11 * len(METHODS), 5.5), sharey=True)
for ax, method in zip(axes, METHODS):
    sub = comparison_53_df[comparison_53_df.method == method]
    x = np.arange(len(STAGES))
    width = 0.35
    ax.bar(
        x - width / 2,
        sub["iou_1_1"],
        width,
        yerr=sub["std_1_1"],
        capsize=3,
        label="1-1 (existing)",
        color="#7f8c8d",
    )
    ax.bar(
        x + width / 2,
        sub["iou_5_3"],
        width,
        yerr=sub["std_5_3"],
        capsize=3,
        label=f"5-3 (pooled, {N_FOLDS_53}-fold CV)",
        color="#2ecc71",
    )
    ax.set_xticks(x, [STAGE_LABELS[s] for s in STAGES], rotation=20, ha="right")
    ax.set_title(method)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("oracle IoU (mean +/- std)")
fig.suptitle("Noisy fg/bg cleaning — 1-1 vs. 5-3 pooled gallery, per stage")
fig.tight_layout()
_comparison_53_path = OUTPUT_DIR / "comparison_1_1_vs_5_3.png"
fig.savefig(_comparison_53_path, dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved %s and %s", OUTPUT_DIR / "comparison_1_1_vs_5_3.csv", _comparison_53_path)

# %% Part 12 — latency/throughput: every phase above traded off against wall-clock cost,
# which no figure in this script reported before now. `torch.cuda.synchronize()` is called
# around every timed block (see `_shared/latency.py`) so GPU-async dispatch doesn't
# understate elapsed time. Scoped to this file's own headline sections (Parts 3-4's
# encoding, Part 7's 1-1 scoring, and the 5-3 section's encoding/scoring) rather than every
# intermediate stage's own sub-loop (Part 5's spatial-filter/attention-check pass, Part 6's
# Step-3 clustering, Part 10's composed-pipeline sweep) — those are covered by the encoding
# phases they reuse and by Part 7/the 5-3 section's own scoring timers, and separately timing
# every one of them would multiply this already-large file's latency bookkeeping for a
# question ("which of Steps 1-3 is slow") this file's own module docstring already answers
# qualitatively (Step 3's HDBSCAN + kNN pass is by far the most expensive part of the 5-3
# section, hence ENABLE_STEP3=False by default).
cache_hits, cache_misses = encoder.total_hits, encoder.total_misses
cache_total = cache_hits + cache_misses
latency_rows.append(
    {
        "phase": "total",
        "elapsed_s": sum(r["elapsed_s"] for r in latency_rows),
        "n_units": float("nan"),
        "units_per_sec": float("nan"),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": cache_hits / cache_total if cache_total > 0 else float("nan"),
    }
)
latency_df = pd.DataFrame(latency_rows)
latency_df.to_csv(OUTPUT_DIR / "latency.csv", index=False)
log.info("Latency by phase (GPU-synchronized wall-clock time):")
for _, row in latency_df.iterrows():
    log.info("  phase=%-24s elapsed=%.1fs n_units=%s", row.phase, row.elapsed_s, row.n_units)
log.info("Wrote %s", OUTPUT_DIR / "latency.csv")

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
#
# - **`oracle_vs_achievable.png`/`oracle_iou_by_stage.csv`'s `mean_achievable_iou`/
#   `oracle_minus_achievable_gap` columns** — `oracle_iou` everywhere else in Parts 7-9 is an
#   upper bound (tunes its threshold against the query's own GT); achievable_iou tunes the
#   threshold on the *reference/exemplar* image's own GT instead (the same image that built
#   the gallery) and transfers it as-is to the query — the number a deployed pipeline without
#   query-time labels would actually see. A cleaning stage that beats `raw` for oracle but
#   not achievable IoU means it's a trend in "how separable the scores could be," not in what
#   a real threshold captures — check both before trusting `oracle_iou_by_stage.png` alone.
# - **`latency.csv`** — GPU-synchronized wall-clock cost (see `_shared/latency.py`) of this
#   file's own headline phases: query/reference image encode, gallery-crop encode, 1-1
#   scoring (Part 7), and the 5-3 section's own image encode/crop encode/scoring, plus cache
#   hit-rate on the final "total" row. Scoped to these phases rather than every intermediate
#   stage's own sub-loop — see Part 12's own comment for why.
# - **`size_correlation.csv`/`.png`** — does oracle IoU correlate with the query GT's own area
#   fraction (`gt_area_frac`, added to every row of `oracle_iou_per_combo.csv` in Part 7)?
#   `.csv` covers every (method, stage); `.png`'s tercile bars facet by stage, mirroring
#   `oracle_iou_by_stage.png`'s own per-stage breakdown — check whether a cleaning stage's
#   benefit concentrates on small objects specifically before generalizing it.
# - **`qualitative_worst_best.png`** — actual worst-5/best-5 query images (crop, raw knn_fgbg
#   score map, GT mask) at the `raw` stage across every combo, not an average — unlike the
#   per-focus-combo figures above (which show *where* a technique cleaned one instance), this
#   shows *which whole combos* the fixed `raw` baseline fails or succeeds on hardest.
# - **`comparison_1_1_vs_5_3.csv`'s `prob_5_3_beats_1_1` column** — an unpaired bootstrap
#   comparison (2000 resamples, see `_shared/stats.py`) of this file's most central
#   comparison: for each (method, stage), is the pooled 5-3 gallery's oracle IoU actually
#   distinguishable from the single-reference 1-1 gallery's, or within combo/fold-to-fold
#   noise? Near 0.5 means indistinguishable — a quantitative version of
#   `comparison_1_1_vs_5_3.png`'s own eyeball comparison.

# %%
