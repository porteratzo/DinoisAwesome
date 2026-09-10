# %% [markdown]
# # Fundamental: Training Set Size — Full N=1..5 Sweep, Properly Cross-Validated
#
# Every sibling script in `experiments/fundamental/` builds its gallery from exactly **one**
# reference image, scored against exactly one query image — a 1-train/1-eval split, but
# picked once (a handful of fixed `(ref, query)` pairs from `_shared/dataset_pairs.py`), never
# cross-validated. This script's very first version swept a gallery pooled from a *fixed*
# image order (N=1..5, eval fixed at images 6-8) and found pooling more training images
# clearly helped (`knn_fgbg`: 0.815 -> 0.877). That didn't survive a follow-up cross-validated
# check comparing only the two endpoints (1-1 vs. 5-3): the fixed order was itself a
# dataset-of-origin confound (image 1 is abc3's original capture, images 3-8 are abc4's, and
# the fixed eval set happened to be all abc4-origin) — once role assignment was randomized per
# fold, the effect vanished.
#
# This version restores the full growth curve (every step from 1 to 5 training images, not
# just the two endpoints) while keeping the fix that mattered: **every point in the sweep uses
# a fresh random shuffle of that part type's 8 images, 5-fold cross-validated, always against
# 3 held-out eval images** — `N_TRAIN_SWEEP = [1, 2, 3, 4, 5]`, `N_EVAL = 3`, `N_FOLDS = 5`
# for every N (bounded deliberately — an exhaustive combinatorial sweep over every possible
# image subset would multiply every downstream cost for marginal extra confidence). Holding
# `n_eval` fixed at 3 for every N (rather than 1 eval image at N=1 like the two-endpoint
# version) also removes an eval-set-size mismatch that made N=1 and N=5 not quite
# apples-to-apples.
#
# Per (part_type, instance-type group, N_train, fold): pool every training instance's fg/bg
# tokens (foreground = its own mask, background = excludes every instance of that group in its
# own image — same convention every sibling script uses) from the classic 3-point
# `global+mid+close` crop scales (the already-settled baseline from the scale-composition
# experiments, not itself under test here) into one gallery, score it against every one of that
# fold's 3 eval images with GT for that group. Scored both ways every sibling script uses:
# `single_proto` (masked-mean cosine similarity) and `knn_fgbg` (per-patch contrastive kNN),
# oracle IoU per sample, pooled into a per-N mean +/- std growth curve and a per-fold
# breakdown (the direct evidence for how much a single reshuffle can swing the result at any
# given N — including N=1, matching every sibling script's own uncross-validated paradigm).

# %% Logging — must be before torch import
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("training_set_size_ablation")

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from dinoisawesome import DinoEncoder, EncoderWithCache, compute_exemplar_features, load_annotations
from dinoisawesome.abc3 import INSTANCE_TYPE_GROUPS, PART_TYPES, available_instance_groups
from dinoisawesome.instance_detection import extract_patch_tokens

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared.latency import cuda_timer, images_per_sec  # noqa: E402
from _shared.mask_geometry import pixel_mask_to_patch_mask, scale_crop_box  # noqa: E402
from _shared.prototype_ops import knn_score_heatmap, score_heatmap  # noqa: E402
from _shared.qualitative_gallery import ScoredExample, save_score_gallery  # noqa: E402
from _shared.stats import bootstrap_ci, bootstrap_prob_greater  # noqa: E402
from _shared.thresholding import achievable_iou, oracle_iou  # noqa: E402

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

DATA_ROOT = _REPO_ROOT / "data"
DATASET = "abc5"

# abc5 numbers 1-8 per part type (abc3's original (1,2) renumbered, then abc4's (1..6) as
# (3..8) — see _shared/dataset_pairs.py's docstring). Every image is a candidate for either
# role; which role it plays is decided per-fold below, not fixed up front.
ALL_NUMBERS: list[int] = [1, 2, 3, 4, 5, 6, 7, 8]

N_TRAIN_SWEEP: list[int] = [1, 2, 3, 4, 5]
N_EVAL: int = 3
N_FOLDS: int = 5

# The classic 3-point baseline every sibling script defaults to — not the axis under test
# here, so it's held fixed rather than swept (see scale_composition_oracle_iou.py for that).
GALLERY_SCALES: list[str] = ["global", "mid", "close"]

DINO_VERSION = "v3"
DINO_SIZE = "base"
IMG_SIZE = 768
LAYER_IDX = 11  # last block of ViT-B/16 (depth 12)
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")
DINO_ENCODING_CACHE_DIR: str | None = os.environ.get("DINO_ENCODING_CACHE_DIR")

MASK_PATCH_THRESHOLD = 0.3
CROP_PADDING_FRACTION = 1.0
MIN_CROP_SIZE = 128

ORACLE_THRESHOLD_STEPS = 25
KNN_FGBG_NUM_NEIGHBOURS = 10

METHODS: list[str] = ["single_proto", "knn_fgbg"]
METHOD_COLOR: dict[str, str] = {"single_proto": "#7f8c8d", "knn_fgbg": "#2ecc71"}

# One representative (N_train, fold, method) point whose individual eval samples get kept as
# PIL images + raw score maps for the worst/best-N qualitative gallery — collecting this for
# every sweep point would be needless disk/memory cost, so only the largest gallery (where a
# failure is most informative — it's the "should be easiest" endpoint) is captured.
QUALITATIVE_N_TRAIN = N_TRAIN_SWEEP[-1]
QUALITATIVE_METHOD = "knn_fgbg"
QUALITATIVE_FOLD = 0
QUALITATIVE_MAX_EXAMPLES = 60

# Bootstrap settings for the headline CI and the N_train=1-vs-largest significance check.
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

# Fold role-assignment RNG — one shared generator, advanced in a fixed (N_train, fold,
# part_type) order. Seeded from OS entropy (no fixed seed), not from SEED below: folds should
# be genuinely randomized on every run, not the same replayed permutations forever.
SEED = 0
torch.manual_seed(SEED)
fold_rng = np.random.default_rng()

OUTPUT_DIR = _REPO_ROOT / "outputs" / "fundamental_abc5" / "training_set_size_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log.info(
    "dataset=%s part_types=%s all_numbers=%s N_TRAIN_SWEEP=%s n_eval=%d n_folds=%d  |  "
    "DINO%s-%s img_size=%d layer=%d  |  gallery_scales=%s",
    DATASET,
    PART_TYPES,
    ALL_NUMBERS,
    N_TRAIN_SWEEP,
    N_EVAL,
    N_FOLDS,
    DINO_VERSION,
    DINO_SIZE,
    IMG_SIZE,
    LAYER_IDX,
    GALLERY_SCALES,
)


# %% Helper: split one crop's patch tokens into (fg, bg), L2-normalised — same local pattern
# every sibling script keeps (self-contained per-file, not shared).
def split_fg_bg_patches(
    patch_tokens: torch.Tensor,
    mask_px: np.ndarray,
    grid_h: int,
    grid_w: int,
    label: str,
    *,
    bg_exclude_mask_px: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bg_exclude_mask_px is None:
        bg_exclude_mask_px = mask_px
    tokens = F.normalize(patch_tokens.reshape(grid_h * grid_w, -1), p=2, dim=-1)

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


# %% Part 1 — discover every (part_type, group, image_number, instance) instance and every
# image's per-group GT mask, across all 8 abc5 images per part type. No train/eval split here
# — every image is preprocessed identically; which images play which role is decided per-fold
# in Part 5.
all_instances: list[dict] = []  # one entry per (part_type, group, image_number, instance)
gt_masks: dict[tuple[str, str, int], np.ndarray] = {}  # (part_type, group, image_number) -> mask
images: dict[tuple[str, int], Image.Image] = {}

for part_type in tqdm(PART_TYPES, desc="Discovering combos"):
    for n in ALL_NUMBERS:
        stem = f"{part_type}_{n}"
        ann_path = DATA_ROOT / DATASET / "annotations" / stem
        groups = available_instance_groups(ann_path)
        if not groups:
            continue
        anns = load_annotations(ann_path)
        img = None
        for group in groups:
            classes = INSTANCE_TYPE_GROUPS[group]
            group_anns = [a for a in anns if a["class"] in classes]
            if not group_anns:
                continue
            if img is None:
                img = Image.open(DATA_ROOT / DATASET / f"{stem}.jpg").convert("RGB")
                images[(part_type, n)] = img
            group_mask = np.stack([a["mask"] for a in group_anns]).any(axis=0)
            gt_masks[(part_type, group, n)] = group_mask
            for ann in group_anns:
                all_instances.append(
                    {
                        "part_type": part_type,
                        "group": group,
                        "image_number": n,
                        "class": ann["class"],
                        "instance_id": ann["instance_id"],
                        "mask": ann["mask"],
                        "bg_exclude_mask": group_mask,
                    }
                )

if not all_instances:
    raise RuntimeError(f"No instances discovered under data/{DATASET} — check the data.")
log.info(
    "Discovered %d instances across %d part types, %d (part_type, group, image) GT masks",
    len(all_instances),
    len({i["part_type"] for i in all_instances}),
    len(gt_masks),
)

# %% Part 2 — build each instance's 3 gallery-scale crops. All-or-nothing per instance: skip
# it entirely if its tightest ("close") crop is below MIN_CROP_SIZE (matches every sibling
# script's convention — see scale_composition_oracle_iou.py).
usable_instances: list[dict] = []
for inst in tqdm(all_instances, desc="Building gallery-scale crops"):
    img = images[(inst["part_type"], inst["image_number"])]
    close_box = scale_crop_box(inst["mask"], "close", CROP_PADDING_FRACTION)
    if close_box[2] - close_box[0] < MIN_CROP_SIZE or close_box[3] - close_box[1] < MIN_CROP_SIZE:
        log.warning(
            "part_type=%s group=%s image#%d instance=%d: close crop below MIN_CROP_SIZE=%dpx "
            "— skipping",
            inst["part_type"],
            inst["group"],
            inst["image_number"],
            inst["instance_id"],
            MIN_CROP_SIZE,
        )
        continue
    inst["crops"] = {}
    for scale in GALLERY_SCALES:
        x0, y0, x1, y1 = scale_crop_box(inst["mask"], scale, CROP_PADDING_FRACTION)
        inst["crops"][scale] = {
            "img": img.crop((x0, y0, x1, y1)),
            "mask_px": inst["mask"][y0:y1, x0:x1],
            "bg_exclude_mask_px": inst["bg_exclude_mask"][y0:y1, x0:x1],
        }
    usable_instances.append(inst)
log.info("Usable instances: %d/%d", len(usable_instances), len(all_instances))

# %% Part 3 — encoder + every image's patch tokens + GT patch masks. Every image is encoded
# once regardless of which role(s) it ends up playing across folds.
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

image_encodings: dict[tuple[str, int], tuple[torch.Tensor, int, int]] = {}
latency_rows: list[dict] = []
with cuda_timer() as t_full_encode:
    for key in tqdm(sorted(images), desc="Encoding images"):
        tokens, q_h, q_w = extract_patch_tokens(encoder, images[key], LAYER_IDX, debias=True)
        image_encodings[key] = (tokens, q_h, q_w)
latency_rows.append(
    {
        "phase": "full_image_encode",
        "elapsed_s": t_full_encode["elapsed_s"],
        "n_units": len(images),
        "units_per_sec": images_per_sec(len(images), t_full_encode["elapsed_s"]),
    }
)

gt_patch_masks: dict[tuple[str, str, int], np.ndarray] = {}
for (part_type, group, n), pixel_mask in gt_masks.items():
    _, q_h, q_w = image_encodings[(part_type, n)]
    gt_patch_masks[(part_type, group, n)] = pixel_mask_to_patch_mask(
        pixel_mask, q_h, q_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )

# %% Part 4 — encode every usable instance's 3 gallery-scale crops, split into fg/bg patch
# banks. Kept on CPU per instance/scale (same rationale as every sibling script once combo
# counts grew past abc3-alone: peak GPU memory is "one gallery's tensors" not "every
# instance's tensors, always"), pooled onto the query's device only at scoring time.
fg_by_instance_scale: dict[tuple, torch.Tensor] = {}  # (instance idx, scale) -> (Nfg, C)
bg_by_instance_scale: dict[tuple, torch.Tensor] = {}  # (instance idx, scale) -> (Nbg, C)

clean_items: list[tuple] = []
for i, inst in enumerate(usable_instances):
    for scale, crop in inst["crops"].items():
        clean_items.append((i, scale, crop["img"], crop["mask_px"], crop["bg_exclude_mask_px"]))

with cuda_timer() as t_crop_encode:
    for i in tqdm(range(0, len(clean_items), chunk_size), desc="Encoding gallery crops"):
        chunk = clean_items[i : i + chunk_size]
        out = encoder([c[2] for c in chunk], layers=[LAYER_IDX], debias=True)
        chunk_patches = out.patches[:, 0]
        grid_h, grid_w = chunk_patches.shape[1], chunk_patches.shape[2]
        for (idx, scale, _, mask_px, bg_exclude_mask_px), patch_tokens in zip(chunk, chunk_patches):
            inst = usable_instances[idx]
            fg, bg = split_fg_bg_patches(
                patch_tokens,
                mask_px,
                grid_h,
                grid_w,
                f"{inst['part_type']}/{inst['group']}/image#{inst['image_number']}/"
                f"inst{inst['instance_id']}/{scale}",
                bg_exclude_mask_px=bg_exclude_mask_px,
            )
            fg_by_instance_scale[(idx, scale)] = fg.cpu()
            bg_by_instance_scale[(idx, scale)] = bg.cpu()
latency_rows.append(
    {
        "phase": "gallery_crop_encode",
        "elapsed_s": t_crop_encode["elapsed_s"],
        "n_units": len(clean_items),
        "units_per_sec": images_per_sec(len(clean_items), t_crop_encode["elapsed_s"]),
    }
)
cache_hits, cache_misses = encoder.total_hits, encoder.total_misses
cache_total = cache_hits + cache_misses
log.info(
    "Encoding cache: %d hits / %d misses (hit_rate=%.2f) across image + gallery-crop encoding",
    cache_hits,
    cache_misses,
    cache_hits / cache_total if cache_total > 0 else float("nan"),
)

instances_by_part_group: dict[tuple[str, str], list[int]] = defaultdict(list)
for i, inst in enumerate(usable_instances):
    instances_by_part_group[(inst["part_type"], inst["group"])].append(i)
sweep_keys = list(instances_by_part_group.keys())
groups_by_part_type: dict[str, list[str]] = defaultdict(list)
for pt, group in sweep_keys:
    groups_by_part_type[pt].append(group)
log.info(
    "Built gallery-scale fg/bg banks for %d instances across %d (part_type, group) pairs",
    len(usable_instances),
    len(sweep_keys),
)


# %% Part 5 — the cross-validated sweep. For every N_train x fold x part_type: shuffle that
# part type's 8 images, take the first N_train as the training pool and the next N_EVAL as the
# held-out eval set for this fold, then for every group with training instances in the pool,
# pool them into one gallery and score against every eval image in the fold with GT for that
# group. A fresh shuffle per (N_train, fold, part_type) — folds are independent random
# resamples (their eval sets can and do overlap across folds), not a non-overlapping partition;
# that's deliberate, matching "shuffle once more" rather than a strict k-fold split.
def pick_ref_number(train_numbers: set[int], part_type: str, group: str) -> int | None:
    """The training image (of this fold's `train_numbers`) whose own GT is available, used to
    *tune* (not oracle-search) an achievable-IoU threshold — see `achievable_iou` below. Picks
    the lowest image number deterministically, not `train_numbers`'s arbitrary set order."""
    for n in sorted(train_numbers):
        if (part_type, group, n) in gt_patch_masks:
            return n
    return None


def build_gallery_bank(pool_idxs: list[int], device: torch.device) -> tuple:
    fg_bank = torch.cat(
        [fg_by_instance_scale[(i, scale)] for i in pool_idxs for scale in GALLERY_SCALES], dim=0
    ).to(device)
    bg_bank = torch.cat(
        [bg_by_instance_scale[(i, scale)] for i in pool_idxs for scale in GALLERY_SCALES], dim=0
    ).to(device)
    proto = compute_exemplar_features(fg_bank, mode="mean")
    return fg_bank, bg_bank, proto


def score_raws(fg_bank, bg_bank, proto, tokens, h, w) -> dict[str, np.ndarray]:
    return {
        "single_proto": score_heatmap(tokens, proto, h, w),
        "knn_fgbg": knn_score_heatmap(tokens, fg_bank, bg_bank, KNN_FGBG_NUM_NEIGHBOURS, h, w),
    }


results: list[dict] = []
qualitative_examples: list[ScoredExample] = []
n_sweep_units = len(N_TRAIN_SWEEP) * N_FOLDS * len(PART_TYPES)
with tqdm(total=n_sweep_units, desc="Part 5: cross-validated sweep") as pbar:
    for n_train in N_TRAIN_SWEEP:
        n_score_evals = 0
        with cuda_timer() as t_scoring:
            for fold_idx in range(N_FOLDS):
                for part_type in PART_TYPES:
                    perm = fold_rng.permutation(ALL_NUMBERS)
                    train_numbers = set(perm[:n_train].tolist())
                    eval_numbers = perm[n_train : n_train + N_EVAL].tolist()

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
                            pool_idxs, encoder.device
                        )
                        # Achievable IoU (see _shared.thresholding.achievable_iou) needs a
                        # threshold fit on a reference image this gallery never gets to see
                        # the query's own GT for — one pooled training image, scored once per
                        # (n_train, fold, part_type, group) since it doesn't depend on
                        # eval_number.
                        ref_number = pick_ref_number(train_numbers, part_type, group)
                        ref_raws = ref_gt = None
                        if ref_number is not None:
                            ref_tokens, ref_h, ref_w = image_encodings[(part_type, ref_number)]
                            ref_raws = score_raws(
                                fg_bank, bg_bank, proto, ref_tokens, ref_h, ref_w
                            )
                            ref_gt = gt_patch_masks[(part_type, group, ref_number)]
                        for eval_number in eval_numbers:
                            key = (part_type, group, eval_number)
                            if key not in gt_patch_masks:
                                continue
                            q_tokens, q_h, q_w = image_encodings[(part_type, eval_number)]
                            gt = gt_patch_masks[key]
                            query_raws = score_raws(fg_bank, bg_bank, proto, q_tokens, q_h, q_w)
                            n_score_evals += 1
                            gt_area_frac = float(gt.sum()) / gt.size
                            for method in METHODS:
                                oi = oracle_iou(query_raws[method], gt, ORACLE_THRESHOLD_STEPS)
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
                                        "n_train": n_train,
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
                                    n_train == QUALITATIVE_N_TRAIN
                                    and method == QUALITATIVE_METHOD
                                    and fold_idx == QUALITATIVE_FOLD
                                    and len(qualitative_examples) < QUALITATIVE_MAX_EXAMPLES
                                ):
                                    qualitative_examples.append(
                                        ScoredExample(
                                            label=f"{part_type}/{group}/img{eval_number}",
                                            image=images[(part_type, eval_number)],
                                            raw=query_raws[method],
                                            gt=gt,
                                            score=oi,
                                        )
                                    )
                    pbar.update(1)
        latency_rows.append(
            {
                "phase": "scoring",
                "n_train": n_train,
                "elapsed_s": t_scoring["elapsed_s"],
                "n_units": n_score_evals,
                "units_per_sec": images_per_sec(n_score_evals, t_scoring["elapsed_s"]),
            }
        )

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "oracle_iou_per_sample.csv", index=False)
log.info(
    "Scoring complete: %d N_train x fold x part_type units swept, %d scored rows",
    n_sweep_units,
    len(results_df),
)

# %% Part 6 — headline growth curve: mean +/- std oracle IoU vs. N_train, pooled across every
# fold/part_type/group/eval_image sample at that N
headline_rows = []
for n_train in N_TRAIN_SWEEP:
    for method in METHODS:
        row_mask = (results_df.n_train == n_train) & (results_df.method == method)
        vals = results_df.loc[row_mask, "oracle_iou"]
        achievable_vals = results_df.loc[row_mask, "achievable_iou"]
        # Percentile bootstrap CI on the mean, alongside the plain std this script already
        # reported — std alone doesn't say whether N_train=1's and N_train=5's means are
        # actually distinguishable or both plausible draws from the same distribution.
        mean_iou, ci_lo, ci_hi = bootstrap_ci(vals.to_numpy(), n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
        headline_rows.append(
            {
                "n_train": n_train,
                "n_eval": N_EVAL,
                "n_folds": N_FOLDS,
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
headline_df.to_csv(OUTPUT_DIR / "growth_curve.csv", index=False)

log.info("Training-set-size growth curve (mean +/- std oracle IoU, %d-fold CV per point):", N_FOLDS)
for _, row in headline_df.iterrows():
    log.info(
        "  N_train=%d method=%-13s iou=%.3f+/-%.3f (n=%d)",
        row.n_train,
        row.method,
        row.mean_iou,
        row.std_iou,
        row.n_samples,
    )
for method in METHODS:
    sub = headline_df[headline_df.method == method].sort_values("n_train")
    delta = sub["mean_iou"].iloc[-1] - sub["mean_iou"].iloc[0]
    log.info(
        "  %s: N_train=1 -> N_train=%d delta=%+.3f (%.3f -> %.3f)",
        method,
        N_TRAIN_SWEEP[-1],
        delta,
        sub["mean_iou"].iloc[0],
        sub["mean_iou"].iloc[-1],
    )

fig, ax = plt.subplots(figsize=(7.5, 5.5))
for method in METHODS:
    sub = headline_df[headline_df.method == method].sort_values("n_train")
    ax.errorbar(
        sub["n_train"],
        sub["mean_iou"],
        yerr=sub["std_iou"],
        marker="o",
        capsize=3,
        label=method,
        color=METHOD_COLOR[method],
    )
ax.set_xticks(N_TRAIN_SWEEP)
ax.set_xlabel(
    f"N training images pooled into the gallery ({N_FOLDS}-fold CV, random shuffle per fold)"
)
ax.set_ylabel(f"oracle IoU on {N_EVAL} held-out eval images (mean +/- std)")
ax.set_ylim(0, 1.0)
ax.set_title("Does more training data help? (cross-validated, not a fixed image order)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "growth_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved %s and %s", OUTPUT_DIR / "growth_curve.csv", OUTPUT_DIR / "growth_curve.png")

# %% Part 6b — oracle IoU (upper bound, tunes the threshold against the query's own GT) vs.
# achievable IoU (a threshold tuned on one pooled training image, transferred as-is to the
# query — what a deployed pipeline without query-time labels would actually get). Every other
# figure in this script plots oracle_iou only.
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for method in METHODS:
    sub = headline_df[headline_df.method == method].sort_values("n_train")
    ax.plot(
        sub["n_train"], sub["mean_iou"], marker="o", linestyle="-",
        label=f"{method} oracle", color=METHOD_COLOR[method],
    )
    ax.plot(
        sub["n_train"], sub["mean_achievable_iou"], marker="^", linestyle="--",
        label=f"{method} achievable", color=METHOD_COLOR[method], alpha=0.6,
    )
ax.set_xticks(N_TRAIN_SWEEP)
ax.set_xlabel("N training images pooled into the gallery")
ax.set_ylabel(f"mean IoU on {N_EVAL} held-out eval images")
ax.set_ylim(0, 1.0)
ax.set_title("Oracle (upper bound) vs. achievable (transferred threshold) IoU vs. gallery size")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "oracle_vs_achievable.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Oracle-vs-achievable gap by N_train:")
for _, row in headline_df.iterrows():
    log.info(
        "  N_train=%d method=%-13s gap=%.3f (oracle=%.3f achievable=%.3f)",
        row.n_train, row.method, row.oracle_minus_achievable_gap, row.mean_iou,
        row.mean_achievable_iou,
    )
log.info("Saved %s", OUTPUT_DIR / "oracle_vs_achievable.png")

# %% Part 6c — latency: does pooling more training images into the gallery cost meaningfully
# more scoring time (bigger fg/bg banks -> bigger knn_fgbg matmuls)? Image + gallery-crop
# encoding happen once regardless of N_train (same 8-images-per-part-type set every time), so
# only the per-N_train scoring phase is a genuine latency-vs-N_train tradeoff; both are still
# reported for completeness. GPU-synchronized timing, see _shared/latency.py.
latency_df = pd.DataFrame(latency_rows)
latency_df.to_csv(OUTPUT_DIR / "latency.csv", index=False)

scoring_latency = latency_df[latency_df.phase == "scoring"].sort_values("n_train")
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.plot(scoring_latency["n_train"], scoring_latency["elapsed_s"], marker="o", color="#e74c3c")
ax.set_xticks(N_TRAIN_SWEEP)
ax.set_xlabel("N training images pooled into the gallery")
ax.set_ylabel(f"total scoring wall-clock time across all {N_FOLDS} folds x part types (s)")
ax.set_title("Latency cost of a bigger gallery (scoring phase only)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "latency.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(
    "Wrote %s and %s (full_image_encode/gallery_crop_encode elapsed once, scoring per N_train)",
    OUTPUT_DIR / "latency.csv", OUTPUT_DIR / "latency.png",
)
for _, row in latency_df.iterrows():
    log.info("  phase=%-20s elapsed=%.1fs n_units=%d", row.phase, row.elapsed_s, row.n_units)

# %% Part 6d — is the N_train=1-vs-largest effect real, or fold/sample noise? An unpaired
# bootstrap comparison (see _shared/stats.py) of the per-sample oracle_iou arrays — the
# significance check `fold_variance.png` (Part 7 below) leaves the reader to eyeball.
significance_rows = []
for method in METHODS:
    lo_vals = results_df.loc[
        (results_df.n_train == N_TRAIN_SWEEP[0]) & (results_df.method == method), "oracle_iou"
    ].to_numpy()
    hi_vals = results_df.loc[
        (results_df.n_train == N_TRAIN_SWEEP[-1]) & (results_df.method == method), "oracle_iou"
    ].to_numpy()
    prob_hi_greater = bootstrap_prob_greater(hi_vals, lo_vals, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
    significance_rows.append(
        {
            "method": method,
            "n_train_lo": N_TRAIN_SWEEP[0],
            "n_train_hi": N_TRAIN_SWEEP[-1],
            "prob_hi_beats_lo": prob_hi_greater,
            "n_lo": len(lo_vals),
            "n_hi": len(hi_vals),
        }
    )
significance_df = pd.DataFrame(significance_rows)
significance_df.to_csv(OUTPUT_DIR / "growth_effect_significance.csv", index=False)
log.info(
    "Growth-curve effect significance (P(N_train=%d mean > N_train=%d mean) under %d-resample "
    "bootstrap; near 0.5 = indistinguishable from noise):",
    N_TRAIN_SWEEP[-1], N_TRAIN_SWEEP[0], N_BOOTSTRAP,
)
for _, row in significance_df.iterrows():
    log.info(
        "  method=%-13s P(N_train=%d beats N_train=%d)=%.3f (n=%d vs n=%d)",
        row.method, row.n_train_hi, row.n_train_lo, row.prob_hi_beats_lo, row.n_hi, row.n_lo,
    )
log.info("Wrote %s", OUTPUT_DIR / "growth_effect_significance.csv")

# %% Part 7 — per-fold breakdown: each fold's own mean oracle IoU at each N_train, pooled
# across every part_type/group/eval_image sample in that fold. Direct evidence for how much a
# single reshuffle can swing the result at any given N — including N=1, matching every
# sibling script's own uncross-validated single-ref-image paradigm.
fold_rows = []
for n_train in N_TRAIN_SWEEP:
    for fold_idx in range(N_FOLDS):
        for method in METHODS:
            vals = results_df.loc[
                (results_df.n_train == n_train)
                & (results_df.fold == fold_idx)
                & (results_df.method == method),
                "oracle_iou",
            ]
            if len(vals) == 0:
                continue
            fold_rows.append(
                {
                    "n_train": n_train,
                    "fold": fold_idx,
                    "method": method,
                    "mean_iou": float(vals.mean()),
                    "std_iou": float(vals.std()),
                    "n_samples": len(vals),
                }
            )
fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(OUTPUT_DIR / "fold_breakdown.csv", index=False)

fig, axes = plt.subplots(1, len(METHODS), figsize=(7 * len(METHODS), 5.5), sharey=True)
for ax, method in zip(axes, METHODS):
    for n_train in N_TRAIN_SWEEP:
        sub = fold_df[(fold_df.n_train == n_train) & (fold_df.method == method)]
        jitter = np.linspace(-0.12, 0.12, max(len(sub), 1))
        ax.scatter(
            n_train + jitter[: len(sub)],
            sub["mean_iou"],
            s=70,
            color=METHOD_COLOR[method],
            edgecolors="black",
            zorder=3,
        )
    ax.set_xticks(N_TRAIN_SWEEP)
    ax.set_xlabel("N training images")
    ax.set_title(method)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel(f"one fold's mean oracle IoU ({N_FOLDS} folds per N_train)")
fig.suptitle("Fold-to-fold spread at each training-set size")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fold_variance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved %s and %s", OUTPUT_DIR / "fold_breakdown.csv", OUTPUT_DIR / "fold_variance.png")

for method in METHODS:
    for n_train in N_TRAIN_SWEEP:
        sub = fold_df[(fold_df.n_train == n_train) & (fold_df.method == method)]
        if len(sub) < 2:
            continue
        log.info(
            "  %s N_train=%d: fold means range=%.3f (min=%.3f, max=%.3f across %d folds)",
            method,
            n_train,
            sub["mean_iou"].max() - sub["mean_iou"].min(),
            sub["mean_iou"].min(),
            sub["mean_iou"].max(),
            len(sub),
        )

# %% Part 8 — per-(part_type, group) breakdown across the sweep — same rationale as every
# sibling script's own per-group breakdown: the aggregate can hide a group-specific effect.
per_group_rows = []
for part_type, group in sweep_keys:
    for n_train in N_TRAIN_SWEEP:
        for method in METHODS:
            vals = results_df.loc[
                (results_df.part_type == part_type)
                & (results_df.group == group)
                & (results_df.n_train == n_train)
                & (results_df.method == method),
                "oracle_iou",
            ]
            if len(vals) == 0:
                continue
            per_group_rows.append(
                {
                    "part_type": part_type,
                    "group": group,
                    "n_train": n_train,
                    "method": method,
                    "mean_iou": float(vals.mean()),
                    "std_iou": float(vals.std()),
                    "n_samples": len(vals),
                }
            )
pd.DataFrame(per_group_rows).to_csv(OUTPUT_DIR / "per_group_breakdown.csv", index=False)
log.info("Wrote %s", OUTPUT_DIR / "per_group_breakdown.csv")

# %% Part 9 — does oracle/achievable IoU correlate with object size? An aggregate mean (every
# figure above) can hide "pooling more images only helps small/large instances" —
# `gt_area_frac` (the query GT's own patch-mask coverage, added to every results_df row in Part
# 5) lets us check, mirroring the pearson/spearman pattern `scale_composition_adaptive_oracle.py`
# already established for instance size vs. optimal scale.
size_correlation_rows = []
for n_train in N_TRAIN_SWEEP:
    for method in METHODS:
        sub = results_df[(results_df.n_train == n_train) & (results_df.method == method)]
        if len(sub) < 3:
            continue
        pearson_r, pearson_p = pearsonr(sub["gt_area_frac"], sub["oracle_iou"])
        spearman_r, spearman_p = spearmanr(sub["gt_area_frac"], sub["oracle_iou"])
        size_correlation_rows.append(
            {
                "n_train": n_train,
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
# everywhere) x oracle IoU, faceted by N_train: does the smallest third of instances
# systematically score worse, and does pooling more training images close or widen that gap?
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

fig, axes = plt.subplots(1, len(N_TRAIN_SWEEP), figsize=(4 * len(N_TRAIN_SWEEP), 5), sharey=True)
for ax, n_train in zip(axes, N_TRAIN_SWEEP):
    tercile_means = (
        results_df[results_df.n_train == n_train]
        .groupby(["size_tercile", "method"], observed=True)["oracle_iou"]
        .mean()
        .unstack("method")
    )
    tercile_means.plot(kind="bar", ax=ax, color=[METHOD_COLOR[m] for m in tercile_means.columns])
    ax.set_title(f"N_train={n_train}")
    ax.set_xlabel("object-size tercile")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("mean oracle IoU")
fig.suptitle("Does object size predict oracle IoU, and does it change with gallery size?")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "size_correlation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Wrote %s and %s", OUTPUT_DIR / "size_correlation.csv", OUTPUT_DIR / "size_correlation.png")

# %% Part 10 — worst/best-N qualitative gallery at one representative point (largest gallery,
# knn_fgbg, fold 0 — see QUALITATIVE_* above). Every other figure in this script averages across
# instances; this shows actual individual query images so a failure mode is visible instead of
# washed out by the mean.
if qualitative_examples:
    save_score_gallery(
        qualitative_examples,
        OUTPUT_DIR / "qualitative_worst_best.png",
        n=5,
        score_name="oracle_iou",
        title=(
            f"Worst/best oracle_iou examples: N_train={QUALITATIVE_N_TRAIN} "
            f"method={QUALITATIVE_METHOD} fold={QUALITATIVE_FOLD}"
        ),
    )
    log.info(
        "Wrote %s (%d examples)",
        OUTPUT_DIR / "qualitative_worst_best.png",
        len(qualitative_examples),
    )
else:
    log.warning("No qualitative examples collected for the representative point")

# %% [markdown]
# ## Reading the results
#
# - **`growth_curve.png`/`.csv`** answer the headline question directly: does oracle IoU on
#   a *fresh, randomly-drawn* 3-image eval set improve as the gallery pools more training
#   images (1-5), averaged over `N_FOLDS` independent reshuffles at every point? A flat or
#   noisy line puts training-set size in the same bucket as every other composition axis this
#   series has tested (multi-scale fg/bg composition, exemplar-augmentation composition,
#   gallery-cleaning composition) — real gains came from *which* single example/scale/
#   transform was used, not from pooling more of them. A clearly rising line, this time
#   actually surviving cross-validation, would be the first such axis where "more" holds up.
# - **`fold_variance.png`/`fold_breakdown.csv`** are the direct check on how much a single
#   reshuffle can swing the result at *any* N, including N=1 — which is exactly every sibling
#   script's own uncross-validated paradigm (one fixed `(ref, query)` pair). A wide spread at
#   N=1 means those scripts' headline numbers are one noisy draw, not a stable estimate.
# - **`per_group_breakdown.csv`** — same aggregation-can-hide-a-group-effect caveat as every
#   sibling script; check before generalizing the growth curve's shape to every instance-type
#   group.
# - Every gallery here still uses the classic `global+mid+close` 3-point crop scale, the
#   already-settled baseline — this experiment isolates *how many images build the gallery*,
#   not *how they're cropped*.
# - **`oracle_vs_achievable.png`/`growth_curve.csv`'s `mean_achievable_iou`/
#   `oracle_minus_achievable_gap` columns** — `oracle_iou` everywhere else is an upper bound
#   (tunes its threshold against the query's own GT); achievable_iou tunes on one pooled
#   training image only and transfers the threshold as-is — the number a deployed pipeline
#   without query-time labels would actually see. If the growth curve rises for oracle but not
#   achievable IoU, pooling more training images is making scores more *separable* without
#   making a *fixed* threshold any better — check both before trusting `growth_curve.png` alone.
# - **`latency.csv`/`latency.png`** — GPU-synchronized wall-clock cost (see `_shared/latency.py`)
#   of a bigger gallery: does N_train=5's larger fg/bg bank cost meaningfully more scoring time
#   than N_train=1's, especially for `knn_fgbg`'s per-patch matmul against the bank?
# - **`size_correlation.csv`/`.png`** — does oracle IoU correlate with the query GT's own area
#   fraction (`gt_area_frac`)? Same aggregation-can-hide-an-effect caveat as
#   `per_group_breakdown.csv`, but for object size instead of instance-type group — check
#   whether pooling more training images helps small objects specifically, or the growth curve
#   is being driven entirely by already-easy large ones.
# - **`growth_effect_significance.csv`** — an unpaired bootstrap comparison (2000 resamples) of
#   N_train=1's vs. N_train=5's per-sample oracle_iou arrays: `prob_hi_beats_lo` near 0.5 means
#   the apparent rise in `growth_curve.png` is not distinguishable from fold/sample noise — a
#   quantitative version of the `fold_variance.png` eyeball check.
# - **`qualitative_worst_best.png`** — actual worst-5/best-5 query images (crop, raw score map,
#   GT mask) at the largest gallery size, not an average — no other figure here shows *why* a
#   specific image fails.

# %%
