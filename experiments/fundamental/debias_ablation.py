# %% [markdown]
# # Fundamental: Positional Debiasing — Does `debias=True` Really Help Oracle IoU?
#
# Every downstream script in this repo — every `fundamental/` sibling, `object_detection/`,
# `anomaly_detection/`, the basic tutorials — passes `debias=True` to `DinoEncoder.forward()`
# (or leaves it as the head classes' own default) as an unquestioned convention. `debias=True`
# projects patch embeddings onto the orthogonal complement of a "positional subspace" estimated
# via SVD on a blank image (`DinoEncoder._build_positional_basis` / `_debias_features`, derived
# from INSID3 — see `dinoisawesome/encoder.py`), the idea being that DINOv3 patch tokens carry a
# spurious position-dependent component that hurts appearance-based matching. That idea has never
# actually been ablated against `debias=False` anywhere in this repo: it was adopted wholesale
# and never measured.
#
# `training_set_size_ablation.py` and `resolution_ablation.py` already found that a fixed
# `(ref, query)` pair — this repo's original per-script paradigm — can show an effect that a
# proper 1-1-vs-5-3 cross-validated check makes vanish (a dataset-of-origin confound in the
# first case). Positional debiasing is exactly the kind of binary on/off switch that same
# confound could fake a "helps" or "hurts" verdict for, so this script applies the identical
# check — reusing `_shared.pooled_gallery_cv`'s discovery/fold-role-assignment helpers, the same
# shared module every other `fundamental/` script's own "5-3 pooled gallery" section already
# uses — to `debias` instead of resolution or training-set size: at **both** the 1-train/1-eval
# and 5-train/3-eval regimes, 5-fold CV each, score every fold with `debias=False` and
# `debias=True` and compare. Fold role assignment (which images train/eval this fold) is drawn
# **once per endpoint**, shared across both `debias` values — the same reasoning as
# `resolution_ablation.py`: `debias` must be the only thing varying between the two arms of the
# comparison, or fold-to-fold noise could masquerade as a debiasing effect.
#
# Unlike the resolution/size sweep, `debias` doesn't change the backbone or its input
# resolution — only which forward-pass output is used — so there's no need to rebuild the
# encoder or guard against OOM per point; one `DinoEncoder` (fixed `DINO_SIZE`/`IMG_SIZE`, the
# same defaults `training_set_size_ablation.py` uses) is built once and reused for both
# `debias` values, and the ground-truth patch masks (which only depend on the patch grid shape,
# not on `debias`) are computed once rather than duplicated per arm.
#
# Per (debias, endpoint, fold, part_type, instance-type group): pool every training instance's
# fg/bg tokens (foreground = its own mask, background = excludes every instance of that group in
# its own image — same convention every sibling script uses) from the classic 3-point
# `global+mid+close` crop scales (not itself under test here) into one gallery, score it against
# every one of that fold's held-out eval images with GT for that group. Scored both ways every
# sibling script uses: `single_proto` (masked-mean cosine similarity) and `knn_fgbg` (per-patch
# contrastive kNN), oracle IoU per sample. Beyond the headline mean-IoU comparison every sibling
# script reports, this script also computes a **paired delta** (`debias=True` minus
# `debias=False`, matched on the exact same fold/pool/eval-image/method) — a much more sensitive
# check than comparing two independent means, since it cancels out per-sample difficulty rather
# than averaging over it.

# %% Logging — must be before torch import
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("debias_ablation")

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

# The axis under test: does DinoEncoder's positional-debiasing projection help oracle IoU?
DEBIAS_SWEEP: list[bool] = [False, True]

# Same 1-1-vs-5-3 two-endpoint CV check `resolution_ablation.py` runs, both at 5-fold CV.
N_FOLDS = 5
N_TRAIN_11, N_EVAL_11 = 1, 1
ENDPOINTS: list[tuple[str, int, int, int]] = [
    ("1-1", N_TRAIN_11, N_EVAL_11, N_FOLDS),
    ("5-3", N_TRAIN_53, N_EVAL_53, N_FOLDS),
]

# The classic 3-point baseline every sibling script defaults to — not the axis under test
# here, so it's held fixed rather than swept (see scale_composition_oracle_iou.py for that).
GALLERY_SCALES: list[str] = ["global", "mid", "close"]

# Fixed encoder config — same defaults training_set_size_ablation.py uses. Unlike
# resolution_ablation.py, `debias` doesn't change the backbone or its input size, so a single
# encoder is built once below rather than rebuilt per sweep point.
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
DEBIAS_COLOR: dict[bool, str] = {False: "#e74c3c", True: "#2980b9"}
ENDPOINT_LABELS = [label for label, *_ in ENDPOINTS]

SEED = 0
torch.manual_seed(SEED)

OUTPUT_DIR = _REPO_ROOT / "outputs" / "fundamental_abc5" / "debias_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log.info(
    "dataset=%s part_types=%s debias_sweep=%s endpoints=%s  |  DINO%s-%s img_size=%d layer=%d  |  "
    "gallery_scales=%s",
    DATASET,
    PART_TYPES,
    DEBIAS_SWEEP,
    [(label, n_train, n_eval, n_folds) for label, n_train, n_eval, n_folds in ENDPOINTS],
    DINO_VERSION,
    DINO_SIZE,
    IMG_SIZE,
    LAYER_IDX,
    GALLERY_SCALES,
)

# Fold role assignment: drawn once per endpoint here, shared across both debias=False and
# debias=True below — see module docstring for why (debias must be the only thing varying
# between the two arms of the comparison).
fold_splits_by_endpoint: dict[str, list[dict[str, tuple[set[int], list[int]]]]] = {
    label: make_fold_role_splits(
        PART_TYPES, seed=SEED, n_train=n_train, n_eval=n_eval, n_folds=n_folds
    )
    for label, n_train, n_eval, n_folds in ENDPOINTS
}


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


# %% Part 1 — discover every annotated instance across all 8 abc5 images per part type, plus
# each image's per-group GT mask (`_shared.pooled_gallery_cv.discover_all_instances`, the same
# generic discovery every other fundamental script's own 5-3 section already reuses).
discovery = discover_all_instances(DATA_ROOT, DATASET, PART_TYPES)
if not discovery.instances:
    raise RuntimeError(f"No instances discovered under data/{DATASET} — check the data.")
log.info(
    "Discovered %d instances across %d part types, %d (part_type, group, image) GT masks",
    len(discovery.instances),
    len({i.part_type for i in discovery.instances}),
    len(discovery.gt_masks),
)

# %% Part 2 — build each instance's 3 gallery-scale crops. All-or-nothing per instance: skip it
# entirely if its tightest ("close") crop is below MIN_CROP_SIZE (matches every sibling script's
# convention).
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

# %% Part 3 — one encoder, built once (debias doesn't change the backbone or its input size).
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

# GT patch masks only depend on the patch grid shape (constant across debias values, since
# debias never changes IMG_SIZE or the backbone), so computed once rather than per debias arm.
# Encoding one image is enough to learn the grid shape.
_probe_key = sorted(discovery.images)[0]
_probe_tokens, grid_h, grid_w = extract_patch_tokens(
    encoder, discovery.images[_probe_key], LAYER_IDX, debias=False
)
gt_patch_masks: dict[tuple[str, str, int], np.ndarray] = {}
for (part_type, group, n), pixel_mask in discovery.gt_masks.items():
    gt_patch_masks[(part_type, group, n)] = pixel_mask_to_patch_mask(
        pixel_mask, grid_h, grid_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )

# %% Part 4 — per-debias-value encode: every image's patch tokens, plus every usable instance's
# 3 gallery-scale crops split into fg/bg patch banks. Encoded twice (once per debias value) since
# `debias` changes which forward-pass output is used, unlike everything in Parts 1-3 above.
image_encodings: dict[bool, dict[tuple[str, int], tuple[torch.Tensor, int, int]]] = {}
fg_by_debias_instance_scale: dict[bool, dict[tuple, torch.Tensor]] = {}
bg_by_debias_instance_scale: dict[bool, dict[tuple, torch.Tensor]] = {}

clean_items: list[tuple] = []
for i, inst in enumerate(usable_instances):
    for scale, crop in inst["crops"].items():
        clean_items.append((i, scale, crop["img"], crop["mask_px"], crop["bg_exclude_mask_px"]))

for debias_value in DEBIAS_SWEEP:
    image_encodings[debias_value] = {}
    for img_key in tqdm(sorted(discovery.images), desc=f"Encoding images (debias={debias_value})"):
        tokens, q_h, q_w = extract_patch_tokens(
            encoder, discovery.images[img_key], LAYER_IDX, debias=debias_value
        )
        image_encodings[debias_value][img_key] = (tokens, q_h, q_w)

    fg_by_instance_scale: dict[tuple, torch.Tensor] = {}
    bg_by_instance_scale: dict[tuple, torch.Tensor] = {}
    for i in tqdm(
        range(0, len(clean_items), chunk_size),
        desc=f"Encoding gallery crops (debias={debias_value})",
    ):
        chunk = clean_items[i : i + chunk_size]
        out = encoder([c[2] for c in chunk], layers=[LAYER_IDX], debias=debias_value)
        chunk_patches = out.patches[:, 0]
        c_grid_h, c_grid_w = chunk_patches.shape[1], chunk_patches.shape[2]
        for (idx, scale, _, mask_px, bg_exclude_mask_px), patch_tokens in zip(chunk, chunk_patches):
            inst = usable_instances[idx]
            fg, bg = split_fg_bg_patches(
                patch_tokens,
                mask_px,
                c_grid_h,
                c_grid_w,
                f"debias={debias_value}/{inst['part_type']}/{inst['group']}/"
                f"image#{inst['image_number']}/inst{inst['instance_id']}/{scale}",
                bg_exclude_mask_px=bg_exclude_mask_px,
            )
            fg_by_instance_scale[(idx, scale)] = fg.cpu()
            bg_by_instance_scale[(idx, scale)] = bg.cpu()
    fg_by_debias_instance_scale[debias_value] = fg_by_instance_scale
    bg_by_debias_instance_scale[debias_value] = bg_by_instance_scale

log.info(
    "Built gallery-scale fg/bg banks for %d instances across %d (part_type, group) pairs, "
    "for both debias=False and debias=True",
    len(usable_instances),
    len(instances_by_part_group),
)


# %% Part 5 — the cross-validated sweep: for every debias value x endpoint x fold x part_type,
# pool the fold's training instances into one gallery and score it against every held-out eval
# image with GT for that group.
def score_gallery(
    pool_idxs: list[int],
    fg_by_instance_scale: dict[tuple, torch.Tensor],
    bg_by_instance_scale: dict[tuple, torch.Tensor],
    q_tokens: torch.Tensor,
    q_h: int,
    q_w: int,
    gt: np.ndarray,
) -> dict[str, float]:
    fg_bank = torch.cat(
        [fg_by_instance_scale[(i, scale)] for i in pool_idxs for scale in GALLERY_SCALES], dim=0
    ).to(q_tokens.device)
    bg_bank = torch.cat(
        [bg_by_instance_scale[(i, scale)] for i in pool_idxs for scale in GALLERY_SCALES], dim=0
    ).to(q_tokens.device)

    proto = compute_exemplar_features(fg_bank, mode="mean")
    raw_proto = score_heatmap(q_tokens, proto, q_h, q_w)
    raw_knn = knn_score_heatmap(q_tokens, fg_bank, bg_bank, KNN_FGBG_NUM_NEIGHBOURS, q_h, q_w)
    return {
        "single_proto": oracle_iou(raw_proto, gt, ORACLE_THRESHOLD_STEPS),
        "knn_fgbg": oracle_iou(raw_knn, gt, ORACLE_THRESHOLD_STEPS),
    }


results: list[dict] = []
n_sweep_units = len(DEBIAS_SWEEP) * sum(n_folds for _, _, _, n_folds in ENDPOINTS) * len(PART_TYPES)
with tqdm(total=n_sweep_units, desc="Part 5: debias x endpoint x fold CV sweep") as pbar:
    for debias_value in DEBIAS_SWEEP:
        fg_by_instance_scale = fg_by_debias_instance_scale[debias_value]
        bg_by_instance_scale = bg_by_debias_instance_scale[debias_value]
        for endpoint_label, n_train, n_eval, n_folds in ENDPOINTS:
            for fold_idx, split in enumerate(fold_splits_by_endpoint[endpoint_label]):
                for part_type in PART_TYPES:
                    train_numbers, eval_numbers = split[part_type]

                    for group in groups_by_part_type.get(part_type, []):
                        idxs = instances_by_part_group[(part_type, group)]
                        pool_idxs = [
                            i for i in idxs if usable_instances[i]["image_number"] in train_numbers
                        ]
                        if not pool_idxs:
                            continue
                        for eval_number in eval_numbers:
                            gt_key = (part_type, group, eval_number)
                            if gt_key not in gt_patch_masks:
                                continue
                            q_tokens, q_h, q_w = image_encodings[debias_value][
                                (part_type, eval_number)
                            ]
                            gt = gt_patch_masks[gt_key]
                            ious = score_gallery(
                                pool_idxs,
                                fg_by_instance_scale,
                                bg_by_instance_scale,
                                q_tokens,
                                q_h,
                                q_w,
                                gt,
                            )
                            for method, iou in ious.items():
                                results.append(
                                    {
                                        "debias": debias_value,
                                        "endpoint": endpoint_label,
                                        "n_train": n_train,
                                        "n_eval": n_eval,
                                        "fold": fold_idx,
                                        "part_type": part_type,
                                        "group": group,
                                        "train_numbers": "+".join(map(str, sorted(train_numbers))),
                                        "eval_number": eval_number,
                                        "n_train_instances": len(pool_idxs),
                                        "method": method,
                                        "oracle_iou": iou,
                                    }
                                )
                    pbar.update(1)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "oracle_iou_per_sample.csv", index=False)
log.info(
    "Scoring complete: %d debias x endpoint x fold x part_type units swept, %d scored rows",
    n_sweep_units,
    len(results_df),
)

# %% Part 6 — headline comparison: mean +/- std oracle IoU for debias=False vs. debias=True, one
# grouped-bar panel per method, faceted by endpoint.
headline_rows = []
for debias_value in DEBIAS_SWEEP:
    for endpoint_label, n_train, n_eval, n_folds in ENDPOINTS:
        for method in METHODS:
            vals = results_df.loc[
                (results_df.debias == debias_value)
                & (results_df.endpoint == endpoint_label)
                & (results_df.method == method),
                "oracle_iou",
            ]
            headline_rows.append(
                {
                    "debias": debias_value,
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
headline_df.to_csv(OUTPUT_DIR / "debias_comparison.csv", index=False)

log.info("Debiasing ablation (mean +/- std oracle IoU, %d-fold CV per point):", N_FOLDS)
for _, row in headline_df.iterrows():
    log.info(
        "  debias=%-5s endpoint=%-3s method=%-13s iou=%.3f+/-%.3f (n=%d)",
        row.debias,
        row.endpoint,
        row.method,
        row.mean_iou,
        row.std_iou,
        row.n_samples,
    )
for endpoint_label in ENDPOINT_LABELS:
    for method in METHODS:
        sub = headline_df[(headline_df.endpoint == endpoint_label) & (headline_df.method == method)]
        iou_false = sub.loc[~sub.debias, "mean_iou"].iloc[0]
        iou_true = sub.loc[sub.debias, "mean_iou"].iloc[0]
        log.info(
            "  endpoint=%s %s: debias=False -> debias=True delta=%+.3f (%.3f -> %.3f)",
            endpoint_label,
            method,
            iou_true - iou_false,
            iou_false,
            iou_true,
        )

fig, axes = plt.subplots(1, len(METHODS), figsize=(6.5 * len(METHODS), 5.5), sharey=True)
bar_width = 0.32
x_pos = {label: i for i, label in enumerate(ENDPOINT_LABELS)}
for ax, method in zip(axes, METHODS):
    for j, debias_value in enumerate(DEBIAS_SWEEP):
        sub = headline_df[(headline_df.method == method) & (headline_df.debias == debias_value)]
        sub = sub.set_index("endpoint").loc[ENDPOINT_LABELS].reset_index()
        offset = (j - 0.5) * bar_width
        ax.bar(
            [x_pos[e] + offset for e in sub["endpoint"]],
            sub["mean_iou"],
            yerr=sub["std_iou"],
            width=bar_width,
            capsize=4,
            color=DEBIAS_COLOR[debias_value],
            label=f"debias={debias_value}",
        )
    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels(ENDPOINT_LABELS)
    ax.set_xlabel("endpoint (n_train-n_eval)")
    ax.set_title(method)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("oracle IoU on held-out eval images (mean +/- std)")
fig.suptitle("Does positional debiasing help? (1-1 vs. 5-3 cross-validated, not a fixed pair)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "debias_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(
    "Saved %s and %s", OUTPUT_DIR / "debias_comparison.csv", OUTPUT_DIR / "debias_comparison.png"
)

# %% Part 7 — paired delta: debias=True minus debias=False oracle IoU, matched on the exact same
# (endpoint, fold, part_type, group, eval_number, method) sample. Cancels out per-sample
# difficulty instead of averaging over it — a much more sensitive check than Part 6's comparison
# of two independent means.
join_keys = ["endpoint", "fold", "part_type", "group", "eval_number", "method"]
false_df = results_df.loc[~results_df.debias, [*join_keys, "oracle_iou"]]
false_df = false_df.rename(columns={"oracle_iou": "iou_false"})
true_df = results_df.loc[results_df.debias, [*join_keys, "oracle_iou"]]
true_df = true_df.rename(columns={"oracle_iou": "iou_true"})
paired_df = false_df.merge(true_df, on=join_keys)
paired_df["delta"] = paired_df["iou_true"] - paired_df["iou_false"]
paired_df.to_csv(OUTPUT_DIR / "paired_delta.csv", index=False)

paired_summary_rows = []
for endpoint_label in ENDPOINT_LABELS:
    for method in METHODS:
        sub = paired_df[(paired_df.endpoint == endpoint_label) & (paired_df.method == method)]
        if len(sub) == 0:
            continue
        paired_summary_rows.append(
            {
                "endpoint": endpoint_label,
                "method": method,
                "mean_delta": float(sub["delta"].mean()),
                "std_delta": float(sub["delta"].std()),
                "frac_positive": float((sub["delta"] > 0).mean()),
                "frac_negative": float((sub["delta"] < 0).mean()),
                "n_samples": len(sub),
            }
        )
paired_summary_df = pd.DataFrame(paired_summary_rows)
paired_summary_df.to_csv(OUTPUT_DIR / "paired_delta_summary.csv", index=False)

log.info("Paired delta (debias=True - debias=False oracle IoU, matched per sample):")
for _, row in paired_summary_df.iterrows():
    log.info(
        "  endpoint=%-3s method=%-13s mean_delta=%+.3f+/-%.3f  frac_positive=%.2f (n=%d)",
        row.endpoint,
        row.method,
        row.mean_delta,
        row.std_delta,
        row.frac_positive,
        row.n_samples,
    )

fig, axes = plt.subplots(1, len(METHODS), figsize=(6.5 * len(METHODS), 5.5), sharey=True)
for ax, method in zip(axes, METHODS):
    data = [
        paired_df.loc[(paired_df.endpoint == e) & (paired_df.method == method), "delta"].to_numpy()
        for e in ENDPOINT_LABELS
    ]
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.boxplot(data, tick_labels=ENDPOINT_LABELS, showmeans=True)
    ax.set_xlabel("endpoint (n_train-n_eval)")
    ax.set_title(method)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("per-sample oracle IoU delta (debias=True - debias=False)")
fig.suptitle("Paired debiasing effect — above zero means debiasing helps, per matched sample")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "paired_delta.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved %s and %s", OUTPUT_DIR / "paired_delta.csv", OUTPUT_DIR / "paired_delta.png")

# %% Part 8 — per-fold breakdown: each fold's own mean oracle IoU at each (debias, endpoint),
# pooled across every part_type/group/eval_image sample in that fold. Direct evidence for how
# much a single reshuffle can swing the result at any given point — including the 1-1 endpoint,
# matching every sibling script's own uncross-validated single-ref-image paradigm.
fold_rows = []
for debias_value in DEBIAS_SWEEP:
    for endpoint_label, _n_train, _n_eval, n_folds in ENDPOINTS:
        for fold_idx in range(n_folds):
            for method in METHODS:
                vals = results_df.loc[
                    (results_df.debias == debias_value)
                    & (results_df.endpoint == endpoint_label)
                    & (results_df.fold == fold_idx)
                    & (results_df.method == method),
                    "oracle_iou",
                ]
                if len(vals) == 0:
                    continue
                fold_rows.append(
                    {
                        "debias": debias_value,
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

fig, axes = plt.subplots(1, len(METHODS), figsize=(7 * len(METHODS), 5.5), sharey=True)
for ax, method in zip(axes, METHODS):
    for debias_value in DEBIAS_SWEEP:
        for endpoint_label in ENDPOINT_LABELS:
            offset = -0.12 if endpoint_label == "1-1" else 0.12
            base_x = 0 if not debias_value else 1
            sub = fold_df[
                (fold_df.method == method)
                & (fold_df.debias == debias_value)
                & (fold_df.endpoint == endpoint_label)
            ]
            jitter = np.linspace(-0.05, 0.05, max(len(sub), 1))
            ax.scatter(
                [base_x + offset + j for j in jitter[: len(sub)]],
                sub["mean_iou"],
                s=70,
                marker="s" if endpoint_label == "1-1" else "o",
                color=DEBIAS_COLOR[debias_value],
                edgecolors="black",
                zorder=3,
                label=f"debias={debias_value}, {endpoint_label}",
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["debias=False", "debias=True"])
    ax.set_title(method)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7)
axes[0].set_ylabel(f"one fold's mean oracle IoU ({N_FOLDS} folds per point)")
fig.suptitle("Fold-to-fold spread at each (debias, endpoint)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fold_variance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved %s and %s", OUTPUT_DIR / "fold_breakdown.csv", OUTPUT_DIR / "fold_variance.png")

# %% Part 9 — per-(part_type, group) breakdown across the sweep — same rationale as every
# sibling script's own per-group breakdown: the aggregate can hide a group-specific effect.
per_group_rows = []
for part_type, group in instances_by_part_group:
    for debias_value in DEBIAS_SWEEP:
        for endpoint_label in ENDPOINT_LABELS:
            for method in METHODS:
                vals = results_df.loc[
                    (results_df.part_type == part_type)
                    & (results_df.group == group)
                    & (results_df.debias == debias_value)
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
                        "debias": debias_value,
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
# - **`debias_comparison.png`/`.csv`** answer the headline question directly, one panel per
#   method: is mean oracle IoU on *fresh, randomly-drawn* eval images higher with `debias=True`
#   than `debias=False`, at both the 1-1 and 5-3 endpoints? If the two endpoints disagree (helps
#   at 5-3 but not 1-1, or vice versa), that's itself the finding — a debiasing effect that
#   depends on gallery size isn't a clean effect.
# - **`paired_delta.png`/`.csv`/`paired_delta_summary.csv`** are the more sensitive check: the
#   per-sample IoU delta (`debias=True - debias=False`), matched on the exact same fold/pool/
#   eval-image/method rather than compared as two independent means. A box centered clearly
#   above zero (and `frac_positive` well above 0.5) is real evidence debiasing helps; a box
#   straddling zero means the headline comparison in `debias_comparison.png` could easily be
#   noise even if its two means differ.
# - **`fold_variance.png`/`fold_breakdown.csv`** are the direct check on how much a single
#   reshuffle can swing the result at *any* (debias, endpoint) point, including the 1-1
#   endpoint — which is every other script in this repo's own uncross-validated paradigm (one
#   fixed pair per part type, always with `debias=True`, never checked against `debias=False`).
#   A wide spread here means this repo's existing `debias=True`-only numbers are one noisy draw,
#   not a stable estimate of what debiasing buys.
# - **`per_group_breakdown.csv`** — same aggregation-can-hide-a-group-effect caveat as every
#   sibling script; check before generalizing the headline comparison to every instance-type
#   group.
# - Every gallery here still uses the classic `global+mid+close` 3-point crop scale and a fixed
#   DINOv3-base/768px encoder — this experiment isolates *positional debiasing on vs. off*, not
#   crop composition or resolution/size (see `scale_composition_oracle_iou.py` and
#   `resolution_ablation.py` for those axes).

# %%
