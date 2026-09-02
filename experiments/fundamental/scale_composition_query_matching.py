# %% [markdown]
# # Fundamental: Scale Composition — Reference/Query Scale Matching
#
# Every scale-composition script so far only crops the *reference* image; the query is always
# scored at full, native resolution. That leaves a confound in `scale_composition_oracle_iou.py`'s
# finding that `close` underperforms `global`/`mid`: is a tightly-cropped exemplar prototype
# genuinely worse at representing the object, or is it being penalized for a **scale mismatch**
# against a query that's always scored zoomed all the way out?
#
# This script isolates that by also GT-cropping the *query* through the same
# `N_SCALE_STEPS + 1` t-sweep (using each instance-type group's own query-side GT mask, the same
# way the reference side is cropped from its own GT), then scoring every (ref scale, query scale)
# pair. **Bounded deliberately to a 7x7 grid, not a combinatorial search**: `SCALE_NAMES` has
# `N_SCALE_STEPS + 1 = 7` entries, so the full ref x query cross product is exactly
# `(N_SCALE_STEPS + 1)^2 = 49` cells per method — one heatmap each, not 49 separate figures or a
# per-instance search over anything larger. No composition (this script never combines scales,
# ref or query) and no per-group breakdown, to keep the output to two heatmaps plus one summary
# table.
#
# **Important caveat on IoU scale**: scoring against a tight, GT-centered query crop is a
# fundamentally easier/less-imbalanced task than scoring the whole query image (foreground is a
# much larger fraction of the patch grid once the query itself is cropped near the object) — the
# IoU numbers here are **not on the same scale** as `scale_composition_oracle_iou.py`'s
# whole-image numbers and must not be compared to them directly. Only the *relative* pattern
# within this script's own 7x7 grid (does the diagonal — matched ref/query scale — outperform
# off-diagonal cells?) is the actual question being asked.

# %% Logging — must be before torch import
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("scale_composition_query_matching")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from dinoisawesome import DinoEncoder, EncoderWithCache, compute_exemplar_features, load_annotations
from dinoisawesome.abc3 import INSTANCE_TYPE_GROUPS, PART_TYPES, available_instance_groups

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared.abc3_combos import combo_key  # noqa: E402
from _shared.mask_geometry import pixel_mask_to_patch_mask, scale_crop_box  # noqa: E402
from _shared.prototype_ops import knn_score_heatmap, score_heatmap  # noqa: E402
from _shared.thresholding import oracle_iou  # noqa: E402

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

data_dir = _REPO_ROOT / "data" / "abc3"

REF_NUMBER = 1
QUERY_NUMBER = 2

RUN_PART_TYPES: list[str] = PART_TYPES

DINO_VERSION = "v3"
DINO_SIZE = "large"
IMG_SIZE = 768
LAYER_IDX = 23
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")
DINO_ENCODING_CACHE_DIR: str | None = os.environ.get("DINO_ENCODING_CACHE_DIR")

MASK_PATCH_THRESHOLD = 0.3
CROP_PADDING_FRACTION = 1.0
MIN_CROP_SIZE = 128

ORACLE_THRESHOLD_STEPS = 25
KNN_FGBG_NUM_NEIGHBOURS = 10

N_SCALE_STEPS = 6

METHODS: list[str] = ["single_proto", "knn_fgbg"]

SEED = 0
torch.manual_seed(SEED)

OUTPUT_DIR = _REPO_ROOT / "outputs" / "fundamental" / "scale_composition_query_matching"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log.info(
    "RUN_PART_TYPES=%s ref_number=%d query_number=%d  |  DINO%s-%s img_size=%d layer=%d  |  "
    "n_scale_steps=%d (7x7=49 ref x query cells per method)",
    RUN_PART_TYPES,
    REF_NUMBER,
    QUERY_NUMBER,
    DINO_VERSION,
    DINO_SIZE,
    IMG_SIZE,
    LAYER_IDX,
    N_SCALE_STEPS,
)

# %% Scale-step naming + crop-box geometry (identical to scale_composition_oracle_iou.py)
T_VALUES: np.ndarray = np.linspace(0.0, 1.0, N_SCALE_STEPS + 1)


def scale_step_name(i: int, n: int) -> str:
    if i == 0:
        return "global"
    if i == n:
        return "close"
    if n % 2 == 0 and i == n // 2:
        return "mid"
    return f"{i}/{n}"


SCALE_NAMES: list[str] = [scale_step_name(i, N_SCALE_STEPS) for i in range(N_SCALE_STEPS + 1)]
log.info("Scale steps (global -> close): %s", SCALE_NAMES)


def scale_step_boxes(
    pixel_mask: np.ndarray, t_values: np.ndarray, padding_frac: float
) -> list[tuple[int, int, int, int]]:
    H, W = pixel_mask.shape
    close_box = scale_crop_box(pixel_mask, "close", padding_frac)
    global_box = (0, 0, W, H)
    return [
        tuple(int(round(a + (b - a) * t)) for a, b in zip(global_box, close_box)) for t in t_values
    ]


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
log.info(
    "Discovered %d (part_type, group, instance) combos across %d part types",
    len(combos),
    len({c["part_type"] for c in combos}),
)

# %% Part 2 — build the N_SCALE_STEPS + 1 REFERENCE crops per combo (fg/bg source, same as
# scale_composition_oracle_iou.py), AND the N_SCALE_STEPS + 1 QUERY crops per (part_type, group)
# — the query side only needs one crop set per group (its own GT mask), not per ref instance.
usable_combo_keys: set[tuple] = set()
for combo in tqdm(combos, desc="Building reference scale-step crops"):
    ref_img = ref_images[combo["part_type"]]
    group_mask = group_ref_masks.get((combo["part_type"], combo["group"]), combo["ref_mask"])
    boxes = scale_step_boxes(combo["ref_mask"], T_VALUES, CROP_PADDING_FRACTION)
    close_box = boxes[-1]
    if close_box[2] - close_box[0] < MIN_CROP_SIZE or close_box[3] - close_box[1] < MIN_CROP_SIZE:
        log.warning(
            "combo=%s: closest ref crop %s below MIN_CROP_SIZE=%dpx — skipping every scale step",
            combo_key(combo),
            close_box,
            MIN_CROP_SIZE,
        )
        combo["crops"] = {}
        continue
    combo["crops"] = {}
    for name, box in zip(SCALE_NAMES, boxes):
        x0, y0, x1, y1 = box
        combo["crops"][name] = {
            "img": ref_img.crop(box),
            "mask_px": combo["ref_mask"][y0:y1, x0:x1],
            "bg_exclude_mask_px": group_mask[y0:y1, x0:x1],
        }
    usable_combo_keys.add(combo_key(combo))
log.info("Combos with every ref scale step usable: %d/%d", len(usable_combo_keys), len(combos))

usable_query_groups: set[tuple[str, str]] = set()
query_scale_items: list[tuple] = []  # (part_type, group, scale_name, img, gt_mask_px)
for (part_type, group), pixel_mask in tqdm(
    group_query_masks.items(), desc="Building query scale-step crops"
):
    query_img = query_images[part_type]
    boxes = scale_step_boxes(pixel_mask, T_VALUES, CROP_PADDING_FRACTION)
    close_box = boxes[-1]
    if close_box[2] - close_box[0] < MIN_CROP_SIZE or close_box[3] - close_box[1] < MIN_CROP_SIZE:
        log.warning(
            "part_type=%s group=%s: closest query crop %s below MIN_CROP_SIZE=%dpx — skipping",
            part_type,
            group,
            close_box,
            MIN_CROP_SIZE,
        )
        continue
    usable_query_groups.add((part_type, group))
    for name, box in zip(SCALE_NAMES, boxes):
        x0, y0, x1, y1 = box
        query_scale_items.append(
            (part_type, group, name, query_img.crop(box), pixel_mask[y0:y1, x0:x1])
        )
log.info(
    "Groups with every query scale step usable: %d/%d",
    len(usable_query_groups),
    len(group_query_masks),
)

# %% Part 3 — encoder
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

# %% Part 4 — encode reference scale-step crops -> per-scale fg/bg token banks (same as
# scale_composition_oracle_iou.py; bg fixed at "every ref scale" throughout, matching that
# script's design — this experiment isolates ref/query scale matching only, not bg composition,
# which scale_composition_bg_ablation.py already covers).
fg_by_scale: dict[tuple, torch.Tensor] = {}
bg_by_scale: dict[tuple, torch.Tensor] = {}

clean_items: list[tuple] = []
for combo in combos:
    ck = combo_key(combo)
    if ck not in usable_combo_keys:
        continue
    for name, crop in combo["crops"].items():
        clean_items.append((ck, name, crop["img"], crop["mask_px"], crop["bg_exclude_mask_px"]))

for i in tqdm(range(0, len(clean_items), chunk_size), desc="Encoding reference crops"):
    chunk = clean_items[i : i + chunk_size]
    out = encoder([c[2] for c in chunk], layers=[LAYER_IDX], debias=True)
    chunk_patches = out.patches[:, 0]
    grid_h, grid_w = chunk_patches.shape[1], chunk_patches.shape[2]
    for (ck, name, _, mask_px, bg_exclude_mask_px), patch_tokens in zip(chunk, chunk_patches):
        fg, bg = split_fg_bg_patches(
            patch_tokens,
            mask_px,
            grid_h,
            grid_w,
            f"{ck} scale={name}",
            bg_exclude_mask_px=bg_exclude_mask_px,
        )
        fg_by_scale[(ck, name)] = fg
        bg_by_scale[(ck, name)] = bg

bg_all_lookup: dict[tuple, torch.Tensor] = {
    ck: torch.cat([bg_by_scale[(ck, name)] for name in SCALE_NAMES], dim=0)
    for ck in usable_combo_keys
}
log.info("Built per-scale reference fg/bg galleries for %d combos", len(usable_combo_keys))

# %% Part 5 — encode query scale-step crops -> tokens + local GT patch mask, one grid per crop
# (each query crop has its own (grid_h, grid_w), unlike the whole-image query every other script
# uses — a local, GT-centered patch grid, per the module docstring's IoU-scale caveat).
query_scale_encodings: dict[tuple[str, str, str], tuple[torch.Tensor, int, int]] = {}
query_scale_gt: dict[tuple[str, str, str], np.ndarray] = {}

for i in tqdm(range(0, len(query_scale_items), chunk_size), desc="Encoding query crops"):
    chunk = query_scale_items[i : i + chunk_size]
    out = encoder([c[3] for c in chunk], layers=[LAYER_IDX], debias=True)
    chunk_patches = out.patches[:, 0]
    grid_h, grid_w = chunk_patches.shape[1], chunk_patches.shape[2]
    for (part_type, group, name, _, gt_mask_px), patch_tokens in zip(chunk, chunk_patches):
        tokens = F.normalize(patch_tokens.reshape(grid_h * grid_w, -1), p=2, dim=-1)
        key = (part_type, group, name)
        query_scale_encodings[key] = (tokens, grid_h, grid_w)
        query_scale_gt[key] = pixel_mask_to_patch_mask(
            gt_mask_px, grid_h, grid_w, IMG_SIZE, MASK_PATCH_THRESHOLD
        )
log.info("Built per-scale query crop tokens for %d groups", len(usable_query_groups))

# %% Part 6 — main scoring: every (ref scale, query scale) pair, 7x7=49 cells per method
# matrix_iou[method][t_ref][t_query][ck] -> oracle IoU (local, GT-cropped-query regime)
MatrixIou = dict[str, dict[str, dict[str, dict[tuple, float]]]]
matrix_iou: MatrixIou = {
    method: {tr: {tq: {} for tq in SCALE_NAMES} for tr in SCALE_NAMES} for method in METHODS
}

for combo in tqdm(combos, desc="Part 6: scoring ref x query scale matrix"):
    ck = combo_key(combo)
    if ck not in usable_combo_keys:
        continue
    part_type, group = ck[0], ck[1]
    if (part_type, group) not in usable_query_groups:
        continue
    bg_bank = bg_all_lookup[ck]

    for t_ref in SCALE_NAMES:
        fg_bank = fg_by_scale[(ck, t_ref)]
        proto = compute_exemplar_features(fg_bank, mode="mean")

        for t_query in SCALE_NAMES:
            q_tokens, q_h, q_w = query_scale_encodings[(part_type, group, t_query)]
            gt_local = query_scale_gt[(part_type, group, t_query)]

            raw_proto = score_heatmap(q_tokens, proto, q_h, q_w)
            matrix_iou["single_proto"][t_ref][t_query][ck] = oracle_iou(
                raw_proto, gt_local, ORACLE_THRESHOLD_STEPS
            )

            raw_knn = knn_score_heatmap(
                q_tokens, fg_bank, bg_bank, KNN_FGBG_NUM_NEIGHBOURS, q_h, q_w
            )
            matrix_iou["knn_fgbg"][t_ref][t_query][ck] = oracle_iou(
                raw_knn, gt_local, ORACLE_THRESHOLD_STEPS
            )

n_scored_combos = len(
    {ck for combo in combos for ck in [combo_key(combo)] if combo_key(combo) in usable_combo_keys}
    & {ck for ck in usable_combo_keys if (ck[0], ck[1]) in usable_query_groups}
)
log.info(
    "Scoring complete: %d combos x 7x7=49 (ref,query) cells x %d methods",
    n_scored_combos,
    len(METHODS),
)


# %% Part 7 — heatmaps + diagonal-vs-row-best summary
def mean_iou(lookup: dict[tuple, float]) -> tuple[float, int]:
    vals = list(lookup.values())
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


matrix_rows = []
for method in METHODS:
    for t_ref in SCALE_NAMES:
        for t_query in SCALE_NAMES:
            m, n = mean_iou(matrix_iou[method][t_ref][t_query])
            matrix_rows.append(
                {
                    "method": method,
                    "ref_scale": t_ref,
                    "query_scale": t_query,
                    "mean_iou": m,
                    "n_combos": n,
                }
            )
matrix_df = pd.DataFrame(matrix_rows)
matrix_df.to_csv(OUTPUT_DIR / "ref_query_matrix_iou.csv", index=False)
log.info("Wrote %s (%d rows)", OUTPUT_DIR / "ref_query_matrix_iou.csv", len(matrix_df))

fig, axes = plt.subplots(1, len(METHODS), figsize=(7.5 * len(METHODS), 6.5))
for ax, method in zip(axes, METHODS):
    grid = np.array(
        [[mean_iou(matrix_iou[method][tr][tq])[0] for tq in SCALE_NAMES] for tr in SCALE_NAMES]
    )
    im = ax.imshow(grid, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(SCALE_NAMES)), SCALE_NAMES, rotation=45)
    ax.set_yticks(range(len(SCALE_NAMES)), SCALE_NAMES)
    ax.set_xlabel("query crop scale")
    ax.set_ylabel("reference crop scale")
    ax.set_title(method)
    for r in range(len(SCALE_NAMES)):
        for c in range(len(SCALE_NAMES)):
            ax.text(
                c,
                r,
                f"{grid[r, c]:.2f}",
                ha="center",
                va="center",
                color="white" if grid[r, c] < 0.6 else "black",
                fontsize=7,
            )
        ax.add_patch(
            plt.Rectangle((r - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2)
        )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle(
    f"Reference x query scale matching (local, GT-cropped-query IoU regime — "
    f"{n_scored_combos} combos; red = matched-scale diagonal)"
)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "ref_query_matrix_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved %s", OUTPUT_DIR / "ref_query_matrix_heatmap.png")

diagonal_rows = []
for method in METHODS:
    for t_ref in SCALE_NAMES:
        row_means = {tq: mean_iou(matrix_iou[method][t_ref][tq])[0] for tq in SCALE_NAMES}
        diag = row_means[t_ref]
        best_tq = max(row_means, key=lambda k: row_means[k])
        diagonal_rows.append(
            {
                "method": method,
                "ref_scale": t_ref,
                "diagonal_iou": diag,
                "row_best_query_scale": best_tq,
                "row_best_iou": row_means[best_tq],
                "diagonal_is_row_best": best_tq == t_ref,
            }
        )
diagonal_df = pd.DataFrame(diagonal_rows)
diagonal_df.to_csv(OUTPUT_DIR / "diagonal_vs_row_best.csv", index=False)
log.info("Matched-scale (diagonal) vs. best-query-scale-for-that-ref, per ref scale:")
for _, row in diagonal_df.iterrows():
    log.info(
        "  %-13s ref=%-8s diagonal=%.3f  row_best=%s (%.3f)  matched_is_best=%s",
        row.method,
        row.ref_scale,
        row.diagonal_iou,
        row.row_best_query_scale,
        row.row_best_iou,
        row.diagonal_is_row_best,
    )
n_matched_best = diagonal_df["diagonal_is_row_best"].sum()
log.info(
    "Matched (ref==query) scale is the row-best query scale in %d/%d rows across both methods",
    n_matched_best,
    len(diagonal_df),
)

# %% [markdown]
# ## Reading the results
#
# - **`ref_query_matrix_heatmap.png`** is the main output — two 7x7 heatmaps, one per method,
#   red outline on the diagonal (matched ref/query scale). If the diagonal is visibly the
#   brightest cell in each row, matching ref and query scale genuinely matters and every other
#   script's whole-image-query design was systematically penalizing tight `close` ref crops for
#   a reason unrelated to the crop's own representational quality. If the brightest cell in each
#   row is off-diagonal (e.g. every row's best column is `global`, regardless of ref scale),
#   query scale barely matters here and the ref-scale effect found elsewhere is a property of
#   the reference crop alone.
# - **`diagonal_vs_row_best.csv`**'s `diagonal_is_row_best` column is that same read as a
#   boolean per row — the logged count out of 14 rows (7 ref scales x 2 methods) is the
#   headline number.
# - **Do not compare this script's `mean_iou` values to `per_scale_iou.csv`'s** — see the module
#   docstring's caveat: this regime scores within a small, GT-centered query crop (foreground-
#   dense), not the whole query image (foreground-sparse), so absolute IoU here runs much higher
#   across the board regardless of scale matching.

# %%
