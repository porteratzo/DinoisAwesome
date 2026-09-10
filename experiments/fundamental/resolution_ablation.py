# %% [markdown]
# # Fundamental: Resolution + Backbone-Size Ablation — 1-1 vs. 5-3 Cross-Validated
#
# `training_set_size_ablation.py` found that a single reshuffle can swing oracle IoU
# substantially at any gallery size, and that its own first (uncross-validated, fixed image
# order) sweep showed a spurious training-set-size effect that vanished once a cross-validated
# **1-1 vs. 5-3** check (1 train/1 eval image vs. 5 train/3 eval images, fresh random shuffle
# per fold) was run instead. `object_detection/resolution_ablation/` sweeps DINOv3 resolution
# (256/512/768/1024/1536px) and backbone size (small/base/large), but — like every *other*
# fundamental-family sibling before `training_set_size_ablation.py` — scores a handful of fixed
# `(exemplar, query)` pairs, never cross-validated. A resolution or size effect measured that
# way is exactly the shape of confound `training_set_size_ablation.py` found: it could be real,
# or it could be one noisy draw of which images happened to play which role.
#
# This script applies that same 1-1/5-3 two-endpoint CV check — reusing `_shared.
# pooled_gallery_cv`'s discovery/fold-role-assignment helpers, the same shared module the other
# 8 `fundamental/` scripts' own "5-3 pooled gallery" sections already use, rather than
# re-deriving that bookkeeping locally — to the resolution and backbone-size axes instead of
# training-set size: at *every* (backbone size, resolution) combination in the sweep, run both
# the 1-train/1-eval and 5-train/3-eval regimes, **5-fold CV each**. Fold role assignment
# (which images train/eval this fold) is drawn **once per endpoint, not once per (size,
# resolution)** — the same fold splits are reused at every point so that size/resolution are the
# only things varying between points; re-shuffling per point would let fold-to-fold noise
# masquerade as a size/resolution effect, exactly the confound this whole exercise exists to
# rule out. That gives two independent, cross-validated datapoints per (size, resolution)
# instead of one fixed-pair number — enough to tell a real trend from fold-to-fold noise, and to
# check whether it holds up the same way at both gallery sizes.
#
# Per (size, resolution, endpoint, fold, part_type, instance-type group): pool every training
# instance's fg/bg tokens (foreground = its own mask, background = excludes every instance of
# that group in its own image — same convention every sibling script uses) from the classic
# 3-point `global+mid+close` crop scales (not itself under test here) into one gallery, score it
# against every one of that fold's held-out eval images with GT for that group. Scored both ways
# every sibling script uses: `single_proto` (masked-mean cosine similarity) and `knn_fgbg`
# (per-patch contrastive kNN), oracle IoU per sample.
#
# **GPU memory**: sweeping up to 1536px already needed a script-local cap on the kNN gallery
# bank size (see `MAX_BANK_SIZE_KNN` below — the shared module's own default is tuned for its
# 768px-fixed callers and overflowed an 11.47GB card at 1536px on a first attempt). Adding
# `large` (24 blocks, C=1024 vs. `base`'s 12 blocks/C=768) raises the resident baseline (bigger
# weights, bigger activations) on top of that, so the cap here is tightened further and each
# (size, resolution) unit's encode+score block is wrapped in its own OOM guard: on
# `torch.cuda.OutOfMemoryError` it logs the failure, frees what it can, and moves on to the next
# point rather than losing the rest of an hours-long sweep to one bad combination.

# %% Logging — must be before torch import
import gc
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("resolution_ablation")

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from dinoisawesome import DinoEncoder, EncoderWithCache, compute_exemplar_features
from dinoisawesome.abc3 import PART_TYPES
from dinoisawesome.instance_detection import extract_patch_tokens

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared.latency import cuda_timer, images_per_sec  # noqa: E402
from _shared.mask_geometry import pixel_mask_to_patch_mask, scale_crop_box  # noqa: E402
from _shared.pooled_gallery_cv import (  # noqa: E402
    N_EVAL_53,
    N_TRAIN_53,
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
DATASET = "abc5"

# The two-endpoint CV check from training_set_size_ablation.py's own original pre-check,
# crossed with resolution/size instead of held fixed. Both endpoints now use 5-fold CV (the
# user's explicit choice, more folds than either of that script's original 3-for-1-1/2-for-5-3
# counts — more datapoints per point, at the cost of more compute). (n_train, n_eval, n_folds).
N_FOLDS = 5
N_TRAIN_11, N_EVAL_11 = 1, 1
ENDPOINTS: list[tuple[str, int, int, int]] = [
    ("1-1", N_TRAIN_11, N_EVAL_11, N_FOLDS),
    ("5-3", N_TRAIN_53, N_EVAL_53, N_FOLDS),
]

# DINOv3 img_size values to sweep — same defaults as object_detection/resolution_ablation/ for
# a like-for-like comparison against its uncross-validated numbers.
RESOLUTION_SWEEP: list[int] = [256, 512, 768, 1024, 1536]

# The classic 3-point baseline every sibling script defaults to — not the axis under test
# here, so it's held fixed rather than swept (see scale_composition_oracle_iou.py for that).
GALLERY_SCALES: list[str] = ["global", "mid", "close"]

DINO_VERSION = "v3"
# Backbone sizes to sweep, crossed with RESOLUTION_SWEEP — same three sizes
# object_detection/resolution_ablation/ sweeps, for a like-for-like comparison.
DINO_SIZES: list[str] = ["small", "base", "large"]
# layer_idx is architecture- (not resolution-) dependent, but *is* size-dependent (small/base
# have 12 blocks, large has 24) — so unlike the single-size version of this script, it can't be
# a fixed constant any more. Derived live from each just-built encoder's own block count inside
# the sweep loop below, matching object_detection/resolution_ablation/run_experiments.py's own
# reasoning for doing the same rather than hardcoding a per-size table that could drift from the
# actual loaded checkpoint.
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")
DINO_ENCODING_CACHE_DIR: str | None = os.environ.get("DINO_ENCODING_CACHE_DIR")

MASK_PATCH_THRESHOLD = 0.3
CROP_PADDING_FRACTION = 1.0
MIN_CROP_SIZE = 128

ORACLE_THRESHOLD_STEPS = 25
KNN_FGBG_NUM_NEIGHBOURS = 10

# _shared.pooled_gallery_cv's own MAX_BANK_SIZE_KNN_53 (100_000) is tuned for its siblings,
# which all fix IMG_SIZE=768 (query grid ~48x48=2304 patches) and never sweep backbone size.
# knn_fgbg_score's `query_tokens @ bg_bank.T` matmul costs Q x N_bg — at this script's top
# resolution (1536px, query grid 96x96=9216 patches, 4x that), a bg gallery anywhere near that
# cap made this matmul alone briefly demand ~3.4GB on an 11.47GB card that already had ~6GB
# resident (base-size encoder + all 32 full-image encodings kept on GPU across folds). Adding
# `large` (24 blocks, C=1024 vs. base's 12/768) raises that resident baseline further, so this
# cap is tightened again from this script's own first (base-only) fix (20_000): at 10_000, the
# same worst-case matmul costs ~350MB, comfortable headroom even with large's bigger baseline.
# Applied uniformly across every (size, resolution) point, not just the biggest, so the cap
# itself never becomes a confound in the comparison.
MAX_BANK_SIZE_KNN = 10_000

METHODS: list[str] = ["single_proto", "knn_fgbg"]
METHOD_COLOR: dict[str, str] = {"single_proto": "#7f8c8d", "knn_fgbg": "#2ecc71"}
ENDPOINT_LINESTYLE: dict[str, str] = {"1-1": "--", "5-3": "-"}
ENDPOINT_MARKER: dict[str, str] = {"1-1": "s", "5-3": "o"}

# One representative (size, resolution, endpoint, method, fold) point whose individual eval
# samples get kept as PIL images + raw score maps for the worst/best-N qualitative gallery in
# Part 8 below — collecting this for every sweep point would multiply memory/disk cost by
# n_sweep_units, so only this one point (the sweep's middle resolution, a mid-size backbone, the
# stronger "5-3"/knn_fgbg regime) is captured. Must each be a member of the sweep lists above.
QUALITATIVE_SIZE = "base"
QUALITATIVE_RESOLUTION = 768
QUALITATIVE_ENDPOINT = "5-3"
QUALITATIVE_METHOD = "knn_fgbg"
QUALITATIVE_FOLD = 0
QUALITATIVE_MAX_EXAMPLES = 60  # capped so the gallery figure itself stays a readable size

# Bootstrap settings for Part 7 (CI on headline means) and Part 9 (is the resolution effect
# distinguishable from fold/sample noise, or within it).
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

SEED = 0
torch.manual_seed(SEED)

OUTPUT_DIR = _REPO_ROOT / "outputs" / "fundamental_abc5" / "resolution_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log.info(
    "dataset=%s part_types=%s resolutions=%s sizes=%s endpoints=%s  |  DINO%s  |  "
    "gallery_scales=%s",
    DATASET,
    PART_TYPES,
    RESOLUTION_SWEEP,
    DINO_SIZES,
    [(label, n_train, n_eval, n_folds) for label, n_train, n_eval, n_folds in ENDPOINTS],
    DINO_VERSION,
    GALLERY_SCALES,
)

# Fold role assignment: drawn once per endpoint here, *before* the resolution loop, and reused
# unchanged at every resolution — see module docstring for why (resolution must be the only
# thing varying between points).
fold_splits_by_endpoint: dict[str, list[dict[str, tuple[set[int], list[int]]]]] = {
    label: make_fold_role_splits(
        PART_TYPES, seed=SEED, n_train=n_train, n_eval=n_eval, n_folds=n_folds
    )
    for label, n_train, n_eval, n_folds in ENDPOINTS
}


# %% Helper: split one crop's patch tokens into (fg, bg), L2-normalised — same local pattern
# every sibling script keeps (self-contained per-file, not shared). Takes img_size explicitly
# (unlike every sibling's module-level constant) since resolution is the axis under test here.
def split_fg_bg_patches(
    patch_tokens: torch.Tensor,
    mask_px: np.ndarray,
    grid_h: int,
    grid_w: int,
    img_size: int,
    label: str,
    *,
    bg_exclude_mask_px: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bg_exclude_mask_px is None:
        bg_exclude_mask_px = mask_px
    tokens = F.normalize(patch_tokens.reshape(grid_h * grid_w, -1), p=2, dim=-1)

    fg_patch_mask = pixel_mask_to_patch_mask(
        mask_px, grid_h, grid_w, img_size, MASK_PATCH_THRESHOLD
    )
    fg_flat = torch.from_numpy(fg_patch_mask.reshape(-1)).to(tokens.device)
    fg = tokens[fg_flat]
    if fg.shape[0] == 0:
        log.warning("%s: fg mask empty after patch-grid projection — using all patches", label)
        fg = tokens

    bg_exclude_patch_mask = pixel_mask_to_patch_mask(
        bg_exclude_mask_px, grid_h, grid_w, img_size, MASK_PATCH_THRESHOLD
    )
    bg_exclude_flat = torch.from_numpy(bg_exclude_patch_mask.reshape(-1)).to(tokens.device)
    bg = tokens[~bg_exclude_flat]
    if bg.shape[0] == 0:
        log.warning("%s: bg mask empty after patch-grid projection — using all patches", label)
        bg = tokens

    return fg, bg


# %% Part 1 — discover every annotated instance across all 8 abc5 images per part type, plus
# each image's per-group GT mask (`_shared.pooled_gallery_cv.discover_all_instances`, the same
# generic discovery every other fundamental script's own 5-3 section already reuses).
# Resolution-independent (raw PIL images, pixel-space masks) — done once, outside the
# resolution loop.
discovery = discover_all_instances(DATA_ROOT, DATASET, PART_TYPES)
if not discovery.instances:
    raise RuntimeError(f"No instances discovered under data/{DATASET} — check the data.")
log.info(
    "Discovered %d instances across %d part types, %d (part_type, group, image) GT masks",
    len(discovery.instances),
    len({i.part_type for i in discovery.instances}),
    len(discovery.gt_masks),
)

# %% Part 2 — build each instance's 3 gallery-scale crops in pixel space. Resolution-independent
# (crop boxes come from the mask's own pixel extent, not the encoder's input size) — done once,
# outside the resolution loop. All-or-nothing per instance: skip it entirely if its tightest
# ("close") crop is below MIN_CROP_SIZE (matches every sibling script's convention).
usable_instances: list[dict] = []
for inst in tqdm(discovery.instances, desc="Building gallery-scale crops"):
    img = discovery.images[(inst.part_type, inst.image_number)]
    close_box = scale_crop_box(inst.mask, "close", CROP_PADDING_FRACTION)
    if close_box[2] - close_box[0] < MIN_CROP_SIZE or close_box[3] - close_box[1] < MIN_CROP_SIZE:
        log.warning(
            "part_type=%s group=%s image#%d instance=%d: close crop below MIN_CROP_SIZE=%dpx "
            "— skipping",
            inst.part_type,
            inst.group,
            inst.image_number,
            inst.instance_id,
            MIN_CROP_SIZE,
        )
        continue
    crops: dict = {}
    for scale in GALLERY_SCALES:
        x0, y0, x1, y1 = scale_crop_box(inst.mask, scale, CROP_PADDING_FRACTION)
        crops[scale] = {
            "img": img.crop((x0, y0, x1, y1)),
            "mask_px": inst.mask[y0:y1, x0:x1],
            "bg_exclude_mask_px": inst.bg_exclude_mask[y0:y1, x0:x1],
        }
    usable_instances.append(
        {
            "part_type": inst.part_type,
            "group": inst.group,
            "image_number": inst.image_number,
            "instance_id": inst.instance_id,
            "crops": crops,
        }
    )
log.info("Usable instances: %d/%d", len(usable_instances), len(discovery.instances))

instances_by_part_group: dict[tuple[str, str], list[int]] = defaultdict(list)
for i, inst in enumerate(usable_instances):
    instances_by_part_group[(inst["part_type"], inst["group"])].append(i)
groups_by_part_type: dict[str, list[str]] = defaultdict(list)
for pt, group in instances_by_part_group:
    groups_by_part_type[pt].append(group)


# %% Part 3 — per-resolution encode + score. Everything downstream of the encoder (image
# tokens, GT patch masks, gallery fg/bg banks, and the 1-1/5-3 cross-validated scoring itself)
# depends on img_size, so it all lives inside this loop; Parts 1-2 above (pixel-space discovery
# and cropping) do not and were done once.
results: list[dict] = []
latency_rows: list[dict] = []
qualitative_examples: list[ScoredExample] = []
units_per_point = sum(n_folds for _, _, _, n_folds in ENDPOINTS) * len(PART_TYPES)
n_sweep_units = len(DINO_SIZES) * len(RESOLUTION_SWEEP) * units_per_point
oom_failures: list[tuple[str, str]] = []


def pick_ref_number(
    train_numbers: set[int], part_type: str, group: str, gt_patch_masks: dict
) -> int | None:
    """The training image (of this fold's `train_numbers`) whose own GT is available, used
    to *tune* (not oracle-search) an achievable-IoU threshold — see `score_point` below. Picks
    the lowest image number deterministically rather than e.g. `train_numbers`'s arbitrary set
    iteration order, so reruns pick the same reference image."""
    for n in sorted(train_numbers):
        if (part_type, group, n) in gt_patch_masks:
            return n
    return None


def build_gallery_bank(
    pool_idxs: list[int],
    fg_by_instance_scale: dict[tuple, torch.Tensor],
    bg_by_instance_scale: dict[tuple, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fg_bank = cap_bank_size(
        torch.cat(
            [fg_by_instance_scale[(i, scale)] for i in pool_idxs for scale in GALLERY_SCALES], dim=0
        ),
        MAX_BANK_SIZE_KNN,
        SEED,
    ).to(device)
    bg_bank = cap_bank_size(
        torch.cat(
            [bg_by_instance_scale[(i, scale)] for i in pool_idxs for scale in GALLERY_SCALES], dim=0
        ),
        MAX_BANK_SIZE_KNN,
        SEED,
    ).to(device)
    proto = compute_exemplar_features(fg_bank, mode="mean")
    return fg_bank, bg_bank, proto


def score_raws(
    fg_bank: torch.Tensor,
    bg_bank: torch.Tensor,
    proto: torch.Tensor,
    tokens: torch.Tensor,
    h: int,
    w: int,
) -> dict[str, np.ndarray]:
    return {
        "single_proto": score_heatmap(tokens, proto, h, w),
        "knn_fgbg": knn_score_heatmap(tokens, fg_bank, bg_bank, KNN_FGBG_NUM_NEIGHBOURS, h, w),
    }

with tqdm(total=n_sweep_units, desc="Part 3: per-(size,resolution) 1-1/5-3 CV sweep") as pbar:
    for dino_size in DINO_SIZES:
        for resolution in RESOLUTION_SWEEP:
            point_tag = f"size={dino_size}/res={resolution}"
            units_done = 0
            # Pre-bound to None so the `del` in `finally` below never NameErrors, whether this
            # point failed before or after each was actually built.
            raw_encoder = encoder = None
            image_encodings = gt_patch_masks = None
            fg_by_instance_scale = bg_by_instance_scale = None
            try:
                raw_encoder = DinoEncoder(
                    version=DINO_VERSION,
                    size=dino_size,
                    img_size=resolution,
                    weights_dir=DINO_WEIGHTS_DIR,
                    amp=True,
                )
                # layer_idx derived live from the just-built backbone's own block count — see
                # the DINO_SIZES parameter comment above for why this can't be a constant here.
                layer_idx = len(raw_encoder.backbone.blocks) - 1
                log.info(
                    "=== %s | depth=%d layer_idx=%d | grid=%dx%d ===",
                    point_tag,
                    layer_idx + 1,
                    layer_idx,
                    raw_encoder.grid_h,
                    raw_encoder.grid_w,
                )
                encoder = EncoderWithCache(raw_encoder, cache_dir=DINO_ENCODING_CACHE_DIR)
                chunk_size = encoder.max_batch_size

                image_encodings = {}
                with cuda_timer() as t_full_encode:
                    for img_key in sorted(discovery.images):
                        tokens, q_h, q_w = extract_patch_tokens(
                            encoder, discovery.images[img_key], layer_idx, debias=True
                        )
                        image_encodings[img_key] = (tokens, q_h, q_w)
                latency_rows.append(
                    {
                        "dino_size": dino_size,
                        "resolution": resolution,
                        "phase": "full_image_encode",
                        "elapsed_s": t_full_encode["elapsed_s"],
                        "n_units": len(discovery.images),
                        "units_per_sec": images_per_sec(
                            len(discovery.images), t_full_encode["elapsed_s"]
                        ),
                    }
                )

                gt_patch_masks = {}
                for (part_type, group, n), pixel_mask in discovery.gt_masks.items():
                    _, q_h, q_w = image_encodings[(part_type, n)]
                    gt_patch_masks[(part_type, group, n)] = pixel_mask_to_patch_mask(
                        pixel_mask, q_h, q_w, resolution, MASK_PATCH_THRESHOLD
                    )

                fg_by_instance_scale = {}
                bg_by_instance_scale = {}
                clean_items: list[tuple] = []
                for i, inst in enumerate(usable_instances):
                    for scale, crop in inst["crops"].items():
                        clean_items.append(
                            (i, scale, crop["img"], crop["mask_px"], crop["bg_exclude_mask_px"])
                        )
                with cuda_timer() as t_crop_encode:
                    for i in range(0, len(clean_items), chunk_size):
                        chunk = clean_items[i : i + chunk_size]
                        out = encoder([c[2] for c in chunk], layers=[layer_idx], debias=True)
                        chunk_patches = out.patches[:, 0]
                        grid_h, grid_w = chunk_patches.shape[1], chunk_patches.shape[2]
                        for (idx, scale, _, mask_px, bg_exclude_mask_px), patch_tokens in zip(
                            chunk, chunk_patches
                        ):
                            inst = usable_instances[idx]
                            fg, bg = split_fg_bg_patches(
                                patch_tokens,
                                mask_px,
                                grid_h,
                                grid_w,
                                resolution,
                                f"{point_tag}/{inst['part_type']}/{inst['group']}/"
                                f"image#{inst['image_number']}/inst{inst['instance_id']}/{scale}",
                                bg_exclude_mask_px=bg_exclude_mask_px,
                            )
                            fg_by_instance_scale[(idx, scale)] = fg.cpu()
                            bg_by_instance_scale[(idx, scale)] = bg.cpu()
                latency_rows.append(
                    {
                        "dino_size": dino_size,
                        "resolution": resolution,
                        "phase": "gallery_crop_encode",
                        "elapsed_s": t_crop_encode["elapsed_s"],
                        "n_units": len(clean_items),
                        "units_per_sec": images_per_sec(
                            len(clean_items), t_crop_encode["elapsed_s"]
                        ),
                    }
                )

                n_score_evals = 0
                with cuda_timer() as t_scoring:
                    for endpoint_label, n_train, n_eval, n_folds in ENDPOINTS:
                        for fold_idx, split in enumerate(fold_splits_by_endpoint[endpoint_label]):
                            for part_type in PART_TYPES:
                                train_numbers, eval_numbers = split[part_type]

                                for group in groups_by_part_type.get(part_type, []):
                                    idxs = instances_by_part_group[(part_type, group)]
                                    pool_idxs = [
                                        i
                                        for i in idxs
                                        if usable_instances[i]["image_number"] in train_numbers
                                    ]
                                    if not pool_idxs:
                                        continue
                                    fg_bank, bg_bank, proto = build_gallery_bank(
                                        pool_idxs,
                                        fg_by_instance_scale,
                                        bg_by_instance_scale,
                                        encoder.device,
                                    )
                                    # Achievable IoU (see _shared.thresholding.achievable_iou)
                                    # needs a threshold fit on a reference image this gallery
                                    # never gets to see the query's own GT for — one pooled
                                    # training image, scored once per (fold, part_type, group)
                                    # rather than once per eval_number since it doesn't depend
                                    # on eval_number, to keep this addition's extra cost to one
                                    # extra score_heatmap/knn_score_heatmap call per gallery
                                    # instead of one per (gallery, eval_number) pair.
                                    ref_number = pick_ref_number(
                                        train_numbers, part_type, group, gt_patch_masks
                                    )
                                    ref_raws = ref_gt = None
                                    if ref_number is not None:
                                        ref_tokens, ref_h, ref_w = image_encodings[
                                            (part_type, ref_number)
                                        ]
                                        ref_raws = score_raws(
                                            fg_bank, bg_bank, proto, ref_tokens, ref_h, ref_w
                                        )
                                        ref_gt = gt_patch_masks[(part_type, group, ref_number)]
                                    else:
                                        log.warning(
                                            "%s: no training image with its own GT for "
                                            "%s/%s — achievable_iou left NaN this fold",
                                            point_tag,
                                            part_type,
                                            group,
                                        )
                                    for eval_number in eval_numbers:
                                        gt_key = (part_type, group, eval_number)
                                        if gt_key not in gt_patch_masks:
                                            continue
                                        q_tokens, q_h, q_w = image_encodings[
                                            (part_type, eval_number)
                                        ]
                                        gt = gt_patch_masks[gt_key]
                                        query_raws = score_raws(
                                            fg_bank, bg_bank, proto, q_tokens, q_h, q_w
                                        )
                                        n_score_evals += 1
                                        gt_area_frac = float(gt.sum()) / gt.size
                                        for method in METHODS:
                                            oi = oracle_iou(
                                                query_raws[method], gt, ORACLE_THRESHOLD_STEPS
                                            )
                                            ai = (
                                                achievable_iou(
                                                    ref_raws[method],
                                                    ref_gt,
                                                    query_raws[method],
                                                    gt,
                                                    ORACLE_THRESHOLD_STEPS,
                                                )
                                                if ref_raws is not None
                                                else float("nan")
                                            )
                                            results.append(
                                                {
                                                    "dino_size": dino_size,
                                                    "resolution": resolution,
                                                    "endpoint": endpoint_label,
                                                    "n_train": n_train,
                                                    "n_eval": n_eval,
                                                    "fold": fold_idx,
                                                    "part_type": part_type,
                                                    "group": group,
                                                    "train_numbers": "+".join(
                                                        map(str, sorted(train_numbers))
                                                    ),
                                                    "eval_number": eval_number,
                                                    "n_train_instances": len(pool_idxs),
                                                    "method": method,
                                                    "oracle_iou": oi,
                                                    "achievable_iou": ai,
                                                    "gt_area_frac": gt_area_frac,
                                                }
                                            )
                                            if (
                                                dino_size == QUALITATIVE_SIZE
                                                and resolution == QUALITATIVE_RESOLUTION
                                                and endpoint_label == QUALITATIVE_ENDPOINT
                                                and method == QUALITATIVE_METHOD
                                                and fold_idx == QUALITATIVE_FOLD
                                                and len(qualitative_examples)
                                                < QUALITATIVE_MAX_EXAMPLES
                                            ):
                                                qualitative_examples.append(
                                                    ScoredExample(
                                                        label=f"{part_type}/{group}/"
                                                        f"img{eval_number}",
                                                        image=discovery.images[
                                                            (part_type, eval_number)
                                                        ],
                                                        raw=query_raws[method],
                                                        gt=gt,
                                                        score=oi,
                                                    )
                                                )
                                pbar.update(1)
                                units_done += 1
                cache_hits, cache_misses = encoder.total_hits, encoder.total_misses
                cache_total = cache_hits + cache_misses
                latency_rows.append(
                    {
                        "dino_size": dino_size,
                        "resolution": resolution,
                        "phase": "scoring",
                        "elapsed_s": t_scoring["elapsed_s"],
                        "n_units": n_score_evals,
                        "units_per_sec": images_per_sec(n_score_evals, t_scoring["elapsed_s"]),
                    }
                )
                latency_rows.append(
                    {
                        "dino_size": dino_size,
                        "resolution": resolution,
                        "phase": "total",
                        "elapsed_s": (
                            t_full_encode["elapsed_s"]
                            + t_crop_encode["elapsed_s"]
                            + t_scoring["elapsed_s"]
                        ),
                        "n_units": len(discovery.images) + len(clean_items),
                        "units_per_sec": float("nan"),
                        "cache_hits": cache_hits,
                        "cache_misses": cache_misses,
                        "cache_hit_rate": (
                            cache_hits / cache_total if cache_total > 0 else float("nan")
                        ),
                    }
                )
            except torch.OutOfMemoryError as exc:
                log.error(
                    "%s: FAILED (CUDA OOM) — skipping remaining folds for this point: %s",
                    point_tag,
                    exc,
                )
                oom_failures.append((point_tag, str(exc)))
            finally:
                remaining = units_per_point - units_done
                if remaining > 0:
                    pbar.update(remaining)
                del raw_encoder, encoder
                del image_encodings, gt_patch_masks
                del fg_by_instance_scale, bg_by_instance_scale
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "oracle_iou_per_sample.csv", index=False)
log.info(
    "Scoring complete: %d size x resolution x endpoint x fold x part_type units swept, "
    "%d scored rows, %d point(s) failed with OOM",
    n_sweep_units,
    len(results_df),
    len(oom_failures),
)
for tag, msg in oom_failures:
    log.error("  %s: %s", tag, msg)

# %% Part 4 — headline curves: mean +/- std oracle IoU vs. resolution, one line per
# (method, endpoint), faceted by backbone size, pooled across every fold/part_type/group/
# eval_image sample at that point.
ENDPOINT_LABELS = [label for label, *_ in ENDPOINTS]

headline_rows = []
for dino_size in DINO_SIZES:
    for resolution in RESOLUTION_SWEEP:
        for endpoint_label, n_train, n_eval, n_folds in ENDPOINTS:
            for method in METHODS:
                row_mask = (
                    (results_df.dino_size == dino_size)
                    & (results_df.resolution == resolution)
                    & (results_df.endpoint == endpoint_label)
                    & (results_df.method == method)
                )
                vals = results_df.loc[row_mask, "oracle_iou"]
                achievable_vals = results_df.loc[row_mask, "achievable_iou"]
                # Percentile bootstrap CI on the mean, alongside the plain std every sibling
                # script already reports — std alone doesn't say whether e.g. resolution=256's
                # and resolution=1536's means are actually distinguishable or both plausible
                # draws from the same underlying distribution; see _shared/stats.py.
                mean_iou, ci_lo, ci_hi = bootstrap_ci(
                    vals.to_numpy(), n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
                )
                headline_rows.append(
                    {
                        "dino_size": dino_size,
                        "resolution": resolution,
                        "endpoint": endpoint_label,
                        "n_train": n_train,
                        "n_eval": n_eval,
                        "n_folds": n_folds,
                        "method": method,
                        "mean_iou": float(vals.mean()) if len(vals) else float("nan"),
                        "std_iou": float(vals.std()) if len(vals) else float("nan"),
                        "ci95_lo": ci_lo,
                        "ci95_hi": ci_hi,
                        "mean_achievable_iou": (
                            float(achievable_vals.mean()) if len(achievable_vals) else float("nan")
                        ),
                        "std_achievable_iou": (
                            float(achievable_vals.std()) if len(achievable_vals) else float("nan")
                        ),
                        "oracle_minus_achievable_gap": (
                            float(vals.mean() - achievable_vals.mean())
                            if len(vals) and len(achievable_vals)
                            else float("nan")
                        ),
                        "n_samples": len(vals),
                    }
                )
headline_df = pd.DataFrame(headline_rows)
headline_df.to_csv(OUTPUT_DIR / "resolution_curve.csv", index=False)

log.info("Resolution/size ablation (mean +/- std oracle IoU, cross-validated per point):")
for _, row in headline_df.iterrows():
    log.info(
        "  size=%-5s resolution=%-4d endpoint=%-3s method=%-13s iou=%.3f+/-%.3f (n=%d, %d folds)",
        row.dino_size,
        row.resolution,
        row.endpoint,
        row.method,
        row.mean_iou,
        row.std_iou,
        row.n_samples,
        row.n_folds,
    )
for dino_size in DINO_SIZES:
    for endpoint_label in ENDPOINT_LABELS:
        for method in METHODS:
            sub = headline_df[
                (headline_df.dino_size == dino_size)
                & (headline_df.endpoint == endpoint_label)
                & (headline_df.method == method)
            ].sort_values("resolution")
            delta = sub["mean_iou"].iloc[-1] - sub["mean_iou"].iloc[0]
            log.info(
                "  size=%s endpoint=%s %s: resolution=%d -> resolution=%d delta=%+.3f "
                "(%.3f -> %.3f)",
                dino_size,
                endpoint_label,
                method,
                RESOLUTION_SWEEP[0],
                RESOLUTION_SWEEP[-1],
                delta,
                sub["mean_iou"].iloc[0],
                sub["mean_iou"].iloc[-1],
            )

fig, axes = plt.subplots(1, len(DINO_SIZES), figsize=(7 * len(DINO_SIZES), 5.5), sharey=True)
for ax, dino_size in zip(axes, DINO_SIZES):
    for method in METHODS:
        for endpoint_label in ENDPOINT_LABELS:
            sub = headline_df[
                (headline_df.dino_size == dino_size)
                & (headline_df.method == method)
                & (headline_df.endpoint == endpoint_label)
            ].sort_values("resolution")
            ax.errorbar(
                sub["resolution"],
                sub["mean_iou"],
                yerr=sub["std_iou"],
                marker=ENDPOINT_MARKER[endpoint_label],
                linestyle=ENDPOINT_LINESTYLE[endpoint_label],
                capsize=3,
                label=f"{method} ({endpoint_label})",
                color=METHOD_COLOR[method],
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks(RESOLUTION_SWEEP)
    ax.set_xticklabels([str(r) for r in RESOLUTION_SWEEP])
    ax.set_xlabel("DINOv3 img_size (px)")
    ax.set_title(f"size={dino_size}")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("oracle IoU on held-out eval images (mean +/- std)")
fig.suptitle("Does resolution help? (1-1 vs. 5-3 cross-validated, not a fixed pair)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "resolution_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(
    "Saved %s and %s", OUTPUT_DIR / "resolution_curve.csv", OUTPUT_DIR / "resolution_curve.png"
)

# %% Part 4b — oracle IoU (upper bound, tunes the threshold against the query's own GT) vs.
# achievable IoU (a threshold tuned on one pooled training image, transferred as-is to the
# query — what a deployed pipeline without query-time labels would actually get). Every other
# figure in this script plots oracle_iou only; this is the gap between "what's the best any
# threshold could do" and "what a realistic fixed threshold does," at the "5-3" endpoint (the
# larger, more representative gallery size) since achievable_iou needs a training image with its
# own GT, which is more often available with more pooled training images.
fig, axes = plt.subplots(1, len(DINO_SIZES), figsize=(7 * len(DINO_SIZES), 5.5), sharey=True)
for ax, dino_size in zip(axes, DINO_SIZES):
    for method in METHODS:
        sub = headline_df[
            (headline_df.dino_size == dino_size)
            & (headline_df.method == method)
            & (headline_df.endpoint == "5-3")
        ].sort_values("resolution")
        ax.plot(
            sub["resolution"],
            sub["mean_iou"],
            marker="o",
            linestyle="-",
            label=f"{method} oracle",
            color=METHOD_COLOR[method],
        )
        ax.plot(
            sub["resolution"],
            sub["mean_achievable_iou"],
            marker="^",
            linestyle="--",
            label=f"{method} achievable",
            color=METHOD_COLOR[method],
            alpha=0.6,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(RESOLUTION_SWEEP)
    ax.set_xticklabels([str(r) for r in RESOLUTION_SWEEP])
    ax.set_xlabel("DINOv3 img_size (px)")
    ax.set_title(f"size={dino_size}")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("mean IoU on held-out eval images (5-3 endpoint)")
fig.suptitle("Oracle (upper bound) vs. achievable (transferred threshold) IoU by resolution")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "oracle_vs_achievable.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(
    "Oracle-vs-achievable gap (5-3 endpoint, mean oracle_iou - mean achievable_iou):"
)
for _, row in headline_df[headline_df.endpoint == "5-3"].iterrows():
    log.info(
        "  size=%-5s resolution=%-4d method=%-13s gap=%.3f (oracle=%.3f achievable=%.3f)",
        row.dino_size,
        row.resolution,
        row.method,
        row.oracle_minus_achievable_gap,
        row.mean_iou,
        row.mean_achievable_iou,
    )
log.info("Saved %s", OUTPUT_DIR / "oracle_vs_achievable.png")

# %% Part 5 — per-fold breakdown: each fold's own mean oracle IoU at each (size, resolution,
# endpoint), pooled across every part_type/group/eval_image sample in that fold. Direct
# evidence for how much a single reshuffle can swing the result at any given point — including
# the 1-1 endpoint, matching every sibling script's own uncross-validated single-ref-image
# paradigm.
fold_rows = []
for dino_size in DINO_SIZES:
    for resolution in RESOLUTION_SWEEP:
        for endpoint_label, _n_train, _n_eval, n_folds in ENDPOINTS:
            for fold_idx in range(n_folds):
                for method in METHODS:
                    vals = results_df.loc[
                        (results_df.dino_size == dino_size)
                        & (results_df.resolution == resolution)
                        & (results_df.endpoint == endpoint_label)
                        & (results_df.fold == fold_idx)
                        & (results_df.method == method),
                        "oracle_iou",
                    ]
                    if len(vals) == 0:
                        continue
                    fold_rows.append(
                        {
                            "dino_size": dino_size,
                            "resolution": resolution,
                            "endpoint": endpoint_label,
                            "fold": fold_idx,
                            "method": method,
                            "mean_iou": float(vals.mean()),
                            "std_iou": float(vals.std()),
                            "n_samples": len(vals),
                        }
                    )
fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(OUTPUT_DIR / "fold_breakdown.csv", index=False)

fig, axes = plt.subplots(
    len(METHODS), len(DINO_SIZES), figsize=(7 * len(DINO_SIZES), 5 * len(METHODS)), sharey=True
)
x_pos = {res: i for i, res in enumerate(RESOLUTION_SWEEP)}
for row, method in enumerate(METHODS):
    for col, dino_size in enumerate(DINO_SIZES):
        ax = axes[row, col]
        for endpoint_label in ENDPOINT_LABELS:
            offset = -0.12 if endpoint_label == "1-1" else 0.12
            sub = fold_df[
                (fold_df.method == method)
                & (fold_df.dino_size == dino_size)
                & (fold_df.endpoint == endpoint_label)
            ]
            ax.scatter(
                [x_pos[r] + offset for r in sub["resolution"]],
                sub["mean_iou"],
                s=70,
                marker=ENDPOINT_MARKER[endpoint_label],
                color=METHOD_COLOR[method],
                edgecolors="black",
                zorder=3,
                label=endpoint_label,
            )
        ax.set_xticks(list(x_pos.values()))
        ax.set_xticklabels([str(r) for r in RESOLUTION_SWEEP])
        ax.set_xlabel("DINOv3 img_size (px)")
        ax.set_title(f"{method}, size={dino_size}")
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, title="endpoint")
    axes[row, 0].set_ylabel("one fold's mean oracle IoU")
fig.suptitle("Fold-to-fold spread at each (size, resolution), for both the 1-1 and 5-3 endpoints")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fold_variance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved %s and %s", OUTPUT_DIR / "fold_breakdown.csv", OUTPUT_DIR / "fold_variance.png")

for dino_size in DINO_SIZES:
    for method in METHODS:
        for resolution in RESOLUTION_SWEEP:
            for endpoint_label in ENDPOINT_LABELS:
                sub = fold_df[
                    (fold_df.dino_size == dino_size)
                    & (fold_df.method == method)
                    & (fold_df.resolution == resolution)
                    & (fold_df.endpoint == endpoint_label)
                ]
                if len(sub) < 2:
                    continue
                log.info(
                    "  size=%s %s resolution=%d endpoint=%s: fold means range=%.3f "
                    "(min=%.3f, max=%.3f across %d folds)",
                    dino_size,
                    method,
                    resolution,
                    endpoint_label,
                    sub["mean_iou"].max() - sub["mean_iou"].min(),
                    sub["mean_iou"].min(),
                    sub["mean_iou"].max(),
                    len(sub),
                )

# %% Part 6 — per-(part_type, group) breakdown across the sweep — same rationale as every
# sibling script's own per-group breakdown: the aggregate can hide a group-specific effect.
per_group_rows = []
for part_type, group in instances_by_part_group:
    for dino_size in DINO_SIZES:
        for resolution in RESOLUTION_SWEEP:
            for endpoint_label in ENDPOINT_LABELS:
                for method in METHODS:
                    vals = results_df.loc[
                        (results_df.part_type == part_type)
                        & (results_df.group == group)
                        & (results_df.dino_size == dino_size)
                        & (results_df.resolution == resolution)
                        & (results_df.endpoint == endpoint_label)
                        & (results_df.method == method),
                        "oracle_iou",
                    ]
                    if len(vals) == 0:
                        continue
                    per_group_rows.append(
                        {
                            "part_type": part_type,
                            "group": group,
                            "dino_size": dino_size,
                            "resolution": resolution,
                            "endpoint": endpoint_label,
                            "method": method,
                            "mean_iou": float(vals.mean()),
                            "std_iou": float(vals.std()),
                            "n_samples": len(vals),
                        }
                    )
pd.DataFrame(per_group_rows).to_csv(OUTPUT_DIR / "per_group_breakdown.csv", index=False)
log.info("Wrote %s", OUTPUT_DIR / "per_group_breakdown.csv")

# %% Part 7 — does oracle/achievable IoU correlate with object size? An aggregate mean (every
# figure above) can hide "this only helps small/large instances" — `gt_area_frac` (the query
# GT's own patch-mask coverage, added to every results_df row in Part 3) lets us check, mirroring
# the pearson/spearman correlation pattern `scale_composition_adaptive_oracle.py` already
# established for instance size vs. optimal scale.
size_correlation_rows = []
for dino_size in DINO_SIZES:
    for resolution in RESOLUTION_SWEEP:
        for endpoint_label in ENDPOINT_LABELS:
            for method in METHODS:
                sub = results_df[
                    (results_df.dino_size == dino_size)
                    & (results_df.resolution == resolution)
                    & (results_df.endpoint == endpoint_label)
                    & (results_df.method == method)
                ]
                if len(sub) < 3:
                    continue
                pearson_r, pearson_p = pearsonr(sub["gt_area_frac"], sub["oracle_iou"])
                spearman_r, spearman_p = spearmanr(sub["gt_area_frac"], sub["oracle_iou"])
                size_correlation_rows.append(
                    {
                        "dino_size": dino_size,
                        "resolution": resolution,
                        "endpoint": endpoint_label,
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

# Object-size terciles (global, computed once across every row so the same size cutoffs apply
# everywhere) x oracle IoU, pooled across resolution/endpoint per (dino_size, method) — the more
# interpretable companion to the raw correlation coefficients above: does the smallest third of
# instances systematically score worse, and does that gap close or widen with a bigger backbone?
try:
    results_df["size_tercile"] = pd.qcut(
        results_df["gt_area_frac"], 3, labels=["small", "medium", "large"]
    )
except ValueError:
    log.warning(
        "gt_area_frac has too few distinct values for 3 clean terciles — falling back to "
        "qcut's own duplicate-safe binning (labels become numeric ranges, not small/medium/large)"
    )
    results_df["size_tercile"] = pd.qcut(results_df["gt_area_frac"], 3, duplicates="drop")
fig, axes = plt.subplots(1, len(DINO_SIZES), figsize=(6 * len(DINO_SIZES), 5), sharey=True)
for ax, dino_size in zip(axes, DINO_SIZES):
    tercile_means = (
        results_df[results_df.dino_size == dino_size]
        .groupby(["size_tercile", "method"], observed=True)["oracle_iou"]
        .mean()
        .unstack("method")
    )
    tercile_means.plot(kind="bar", ax=ax, color=[METHOD_COLOR[m] for m in tercile_means.columns])
    ax.set_title(f"size={dino_size}")
    ax.set_xlabel("object-size tercile (by GT patch-mask area fraction)")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("mean oracle IoU (pooled across resolution/endpoint)")
fig.suptitle("Does object size predict oracle IoU?")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "size_correlation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Wrote %s and %s", OUTPUT_DIR / "size_correlation.csv", OUTPUT_DIR / "size_correlation.png")

# %% Part 8 — latency/throughput: every point above traded off against wall-clock cost, which no
# figure in this script reported before now. `torch.cuda.synchronize()` is called around every
# timed block (see `_shared/latency.py`) so GPU-async dispatch doesn't understate elapsed time.
latency_df = pd.DataFrame(latency_rows)
latency_df.to_csv(OUTPUT_DIR / "latency.csv", index=False)

fig, axes = plt.subplots(1, len(DINO_SIZES), figsize=(7 * len(DINO_SIZES), 5.5), sharey=True)
total_latency = latency_df[latency_df.phase == "total"]
for ax, dino_size in zip(axes, DINO_SIZES):
    sub = total_latency[total_latency.dino_size == dino_size].sort_values("resolution")
    ax.plot(sub["resolution"], sub["elapsed_s"], marker="o", color="#e74c3c")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(RESOLUTION_SWEEP)
    ax.set_xticklabels([str(r) for r in RESOLUTION_SWEEP])
    ax.set_xlabel("DINOv3 img_size (px)")
    ax.set_title(f"size={dino_size}")
    ax.grid(alpha=0.3, which="both")
axes[0].set_ylabel("total wall-clock time per point (s, log scale)")
fig.suptitle("Latency cost of the resolution/size sweep (encode + score, this point only)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "latency.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# The actual decision-relevant plot: is a resolution/size bump worth its latency cost? One point
# per (size, resolution) at the 5-3 endpoint's knn_fgbg mean oracle IoU against that point's total
# latency — a config in the upper-left (high IoU, low latency) dominates one to its lower-right.
fig, ax = plt.subplots(figsize=(8, 6))
tradeoff = headline_df[
    (headline_df.endpoint == "5-3") & (headline_df.method == "knn_fgbg")
].merge(
    total_latency[["dino_size", "resolution", "elapsed_s"]], on=["dino_size", "resolution"]
)
for dino_size, marker in zip(DINO_SIZES, ["o", "s", "^"]):
    sub = tradeoff[tradeoff.dino_size == dino_size].sort_values("resolution")
    ax.plot(sub["elapsed_s"], sub["mean_iou"], marker=marker, label=f"size={dino_size}")
    for _, row in sub.iterrows():
        ax.annotate(str(row.resolution), (row.elapsed_s, row.mean_iou), fontsize=7)
ax.set_xscale("log")
ax.set_xlabel("total wall-clock time per point (s, log scale)")
ax.set_ylabel("mean oracle IoU (knn_fgbg, 5-3 endpoint)")
ax.set_title("Accuracy vs. latency tradeoff (point labels are resolution in px)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "accuracy_vs_latency.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(
    "Wrote %s, %s, and %s",
    OUTPUT_DIR / "latency.csv",
    OUTPUT_DIR / "latency.png",
    OUTPUT_DIR / "accuracy_vs_latency.png",
)
for _, row in total_latency.iterrows():
    log.info(
        "  size=%-5s resolution=%-4d total=%.1fs cache_hit_rate=%.2f",
        row.dino_size,
        row.resolution,
        row.elapsed_s,
        row.cache_hit_rate,
    )

# %% Part 9 — is the resolution effect within each size panel real, or fold/sample noise? An
# unpaired bootstrap comparison (see _shared/stats.py) of the lowest vs. highest resolution's
# per-sample oracle_iou arrays, at the 5-3 endpoint (more samples per point than 1-1) — this is
# the significance check `fold_variance.png` (Part 5) leaves the reader to eyeball.
significance_rows = []
for dino_size in DINO_SIZES:
    for method in METHODS:
        lo_vals = results_df.loc[
            (results_df.dino_size == dino_size)
            & (results_df.resolution == RESOLUTION_SWEEP[0])
            & (results_df.endpoint == "5-3")
            & (results_df.method == method),
            "oracle_iou",
        ].to_numpy()
        hi_vals = results_df.loc[
            (results_df.dino_size == dino_size)
            & (results_df.resolution == RESOLUTION_SWEEP[-1])
            & (results_df.endpoint == "5-3")
            & (results_df.method == method),
            "oracle_iou",
        ].to_numpy()
        if len(lo_vals) == 0 or len(hi_vals) == 0:
            continue
        prob_hi_greater = bootstrap_prob_greater(
            hi_vals, lo_vals, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
        )
        significance_rows.append(
            {
                "dino_size": dino_size,
                "method": method,
                "resolution_lo": RESOLUTION_SWEEP[0],
                "resolution_hi": RESOLUTION_SWEEP[-1],
                "prob_hi_beats_lo": prob_hi_greater,
                "n_lo": len(lo_vals),
                "n_hi": len(hi_vals),
            }
        )
significance_df = pd.DataFrame(significance_rows)
significance_df.to_csv(OUTPUT_DIR / "resolution_effect_significance.csv", index=False)
log.info(
    "Resolution effect significance (5-3 endpoint, P(highest-resolution mean > lowest-resolution "
    "mean) under 2000-resample bootstrap; near 0.5 = indistinguishable from noise):"
)
for _, row in significance_df.iterrows():
    log.info(
        "  size=%-5s method=%-13s P(res=%d beats res=%d)=%.3f (n=%d vs n=%d)",
        row.dino_size,
        row.method,
        row.resolution_hi,
        row.resolution_lo,
        row.prob_hi_beats_lo,
        row.n_hi,
        row.n_lo,
    )
log.info("Wrote %s", OUTPUT_DIR / "resolution_effect_significance.csv")

# %% Part 10 — worst/best-N qualitative gallery at one representative point (see
# QUALITATIVE_SIZE/QUALITATIVE_RESOLUTION/QUALITATIVE_ENDPOINT/QUALITATIVE_METHOD above) — every
# other figure in this script averages across instances; this shows actual individual query
# images so a failure mode (one orientation, one lighting condition) is visible instead of
# washed out by the mean.
if qualitative_examples:
    save_score_gallery(
        qualitative_examples,
        OUTPUT_DIR / "qualitative_worst_best.png",
        n=5,
        score_name="oracle_iou",
        title=(
            f"Worst/best oracle_iou examples: size={QUALITATIVE_SIZE} "
            f"resolution={QUALITATIVE_RESOLUTION} endpoint={QUALITATIVE_ENDPOINT} "
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
        "No qualitative examples collected — the representative point "
        "(size=%s, resolution=%d, endpoint=%s, method=%s) never scored, likely because it "
        "failed with OOM (see oom_failures above)",
        QUALITATIVE_SIZE,
        QUALITATIVE_RESOLUTION,
        QUALITATIVE_ENDPOINT,
        QUALITATIVE_METHOD,
    )

# %% [markdown]
# ## Reading the results
#
# - **`resolution_curve.png`/`.csv`** answer the headline question directly, one panel per
#   backbone size: does oracle IoU on *fresh, randomly-drawn* eval images improve with DINOv3
#   input resolution, averaged across folds at every point — and does that trend look the same
#   at both the 1-1 and 5-3 endpoints, and across small/base/large? If the two endpoints
#   disagree (e.g. resolution helps at 5-3 but not 1-1, or vice versa), that's itself the
#   finding: a resolution effect that depends on gallery size isn't a clean resolution effect —
#   check this independently within each size panel, since a confound could in principle affect
#   one size and not another. If both endpoints track together within a size, that's much
#   stronger evidence than either alone — and than `object_detection/resolution_ablation/`'s
#   fixed-pair numbers, which can't distinguish a real trend from which pair happened to be used.
# - **`fold_variance.png`/`fold_breakdown.csv`** are the direct check on how much a single
#   reshuffle can swing the result at *any* (size, resolution) point, including the 1-1
#   endpoint — which is exactly `object_detection/resolution_ablation/`'s own uncross-validated
#   paradigm (one fixed pair per part type). A wide spread here means that script's numbers for
#   the corresponding size/resolution are one noisy draw, not a stable estimate.
# - **`per_group_breakdown.csv`** — same aggregation-can-hide-a-group-effect caveat as every
#   sibling script; check before generalizing the resolution curve's shape to every
#   instance-type group.
# - **If any point failed with a logged CUDA OOM** (see the "Part 3" completion log line and the
#   `oom_failures` list it reports), that (size, resolution) combination is simply missing from
#   every CSV/plot here, not zero-filled — check the log before concluding "large gets worse at
#   high resolution" versus "large's high-resolution points never finished."
# - Every gallery here still uses the classic `global+mid+close` 3-point crop scale, held fixed
#   throughout — this experiment isolates *input resolution and backbone size*, not crop
#   composition (see `scale_composition_oracle_iou.py` for that axis).
# - **`oracle_vs_achievable.png`/`resolution_curve.csv`'s `mean_achievable_iou`/
#   `oracle_minus_achievable_gap` columns** — `oracle_iou` everywhere else in this script is an
#   upper bound (it tunes its threshold against the query's own GT); achievable_iou tunes the
#   threshold on one pooled training image only and transfers it as-is, the number a deployed
#   pipeline without query-time labels would actually see. A resolution/size trend that holds for
#   oracle but not achievable IoU means it's a trend in "how separable the scores could be," not
#   in what a real threshold captures — check both before trusting `resolution_curve.png` alone.
# - **`latency.csv`/`latency.png`/`accuracy_vs_latency.png`** — wall-clock cost (GPU-synchronized,
#   see `_shared/latency.py`) per (size, resolution) point, split by phase (full-image encode,
#   gallery-crop encode, scoring) plus cache hit-rate. `accuracy_vs_latency.png` is the actual
#   tradeoff plot: an IoU gain from a resolution/size bump that costs 5x the latency reads very
#   differently once you can see both axes at once.
# - **`size_correlation.csv`/`.png`** — does oracle IoU correlate with the query GT's own area
#   fraction (`gt_area_frac`, added to every row of `oracle_iou_per_sample.csv`)? Same
#   aggregation-can-hide-an-effect caveat as `per_group_breakdown.csv`, but for object size
#   instead of instance-type group — check whether a resolution/size benefit concentrates on
#   small objects specifically before generalizing it.
# - **`resolution_effect_significance.csv`** — an unpaired bootstrap comparison (2000 resamples)
#   of the lowest- vs. highest-resolution per-sample oracle_iou arrays at the 5-3 endpoint, for
#   each (size, method): `prob_hi_beats_lo` near 0.5 means the apparent trend in
#   `resolution_curve.png` is not distinguishable from fold/sample noise at that size/method: a
#   quantitative version of the `fold_variance.png` eyeball check.
# - **`qualitative_worst_best.png`** — actual worst-5/best-5 query images (crop, raw score map,
#   GT mask) at one representative sweep point, not an average — every other figure here plots a
#   mean or an averaged heatmap, which can't show *why* a config fails on a specific image.

# %%
