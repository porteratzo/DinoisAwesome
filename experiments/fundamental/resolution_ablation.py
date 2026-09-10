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
from tqdm import tqdm

from dinoisawesome import DinoEncoder, EncoderWithCache, compute_exemplar_features
from dinoisawesome.abc3 import PART_TYPES
from dinoisawesome.instance_detection import extract_patch_tokens

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared.mask_geometry import pixel_mask_to_patch_mask, scale_crop_box  # noqa: E402
from _shared.pooled_gallery_cv import (  # noqa: E402
    N_EVAL_53,
    N_TRAIN_53,
    cap_bank_size,
    discover_all_instances,
    make_fold_role_splits,
)
from _shared.prototype_ops import knn_score_heatmap, score_heatmap  # noqa: E402
from _shared.thresholding import oracle_iou  # noqa: E402

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
def build_gallery(
    pool_idxs: list[int],
    fg_by_instance_scale: dict[tuple, torch.Tensor],
    bg_by_instance_scale: dict[tuple, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool + cap + move to device + build the mean prototype for one (fold, part_type, group)'s
    training pool. Depends only on `pool_idxs` — not on which eval image is being scored — so
    callers build this once per pool and reuse it across every eval_number in that fold (it used
    to be rebuilt from scratch per eval_number inside score_gallery, 3x redundant work for every
    5-3 fold since N_EVAL_53=3)."""
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


def score_gallery(
    fg_bank: torch.Tensor,
    bg_bank: torch.Tensor,
    proto: torch.Tensor,
    q_tokens: torch.Tensor,
    q_h: int,
    q_w: int,
    gt: np.ndarray,
) -> dict[str, float]:
    raw_proto = score_heatmap(q_tokens, proto, q_h, q_w)
    raw_knn = knn_score_heatmap(q_tokens, fg_bank, bg_bank, KNN_FGBG_NUM_NEIGHBOURS, q_h, q_w)
    return {
        "single_proto": oracle_iou(raw_proto, gt, ORACLE_THRESHOLD_STEPS),
        "knn_fgbg": oracle_iou(raw_knn, gt, ORACLE_THRESHOLD_STEPS),
    }


results: list[dict] = []
units_per_point = sum(n_folds for _, _, _, n_folds in ENDPOINTS) * len(PART_TYPES)
n_sweep_units = len(DINO_SIZES) * len(RESOLUTION_SWEEP) * units_per_point
oom_failures: list[tuple[str, str]] = []

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
                for img_key in sorted(discovery.images):
                    tokens, q_h, q_w = extract_patch_tokens(
                        encoder, discovery.images[img_key], layer_idx, debias=True
                    )
                    image_encodings[img_key] = (tokens, q_h, q_w)

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
                                valid_evals = [
                                    n
                                    for n in eval_numbers
                                    if (part_type, group, n) in gt_patch_masks
                                ]
                                if not valid_evals:
                                    continue
                                fg_bank, bg_bank, proto = build_gallery(
                                    pool_idxs,
                                    fg_by_instance_scale,
                                    bg_by_instance_scale,
                                    encoder.device,
                                )
                                for eval_number in valid_evals:
                                    gt_key = (part_type, group, eval_number)
                                    q_tokens, q_h, q_w = image_encodings[(part_type, eval_number)]
                                    gt = gt_patch_masks[gt_key]
                                    ious = score_gallery(
                                        fg_bank, bg_bank, proto, q_tokens, q_h, q_w, gt
                                    )
                                    for method, iou in ious.items():
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
                                                "oracle_iou": iou,
                                            }
                                        )
                            pbar.update(1)
                            units_done += 1
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
                vals = results_df.loc[
                    (results_df.dino_size == dino_size)
                    & (results_df.resolution == resolution)
                    & (results_df.endpoint == endpoint_label)
                    & (results_df.method == method),
                    "oracle_iou",
                ]
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

# %%
