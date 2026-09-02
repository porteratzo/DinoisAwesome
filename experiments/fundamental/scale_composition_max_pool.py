# %% [markdown]
# # Fundamental: Scale Composition — Max-Pooling vs. Concatenation-Pooling
#
# `scale_composition_oracle_iou.py` found that composing multiple scales' foreground patches
# into one bank never beat the single best scale, for either method, across all 28 curated
# combos. That script only tested one way of combining scales: **concatenate** every member
# scale's fg patches into one bank, then collapse to a single mean prototype (`single_proto`) or
# score against the pooled bank as one gallery (`knn_fgbg`) — every member scale's patches are
# treated as equally relevant everywhere.
#
# `object_detection/multiscale_ablation/methods.py`'s `build_mean_states` already supports a
# different combine mode for its 3-point `MULTI_SCALE_MEAN_COMBOS`: **max** — score each member
# scale's own prototype/gallery *separately* against the query, then take the per-query-patch
# maximum across scales, instead of pooling patches before scoring. A query patch effectively
# picks whichever scale's view of the object it agrees with best, rather than being scored
# against one prototype diluted by every scale's contribution. This script asks whether that
# untested-at-fine-granularity combine mode changes the finding: is composition's failure a
# property of combining scale information at all, or specifically an artifact of concatenation
# pooling?
#
# Bg is held fixed at "every scale step" throughout (same as `scale_composition_oracle_iou.py`'s
# design) — this script isolates the *pooling mode* question only;
# `scale_composition_bg_ablation.py` already isolates the background-composition question
# separately.

# %% Logging — must be before torch import
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("scale_composition_max_pool")

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
from dinoisawesome.instance_detection import extract_patch_tokens

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
METHOD_COLOR: dict[str, str] = {"single_proto": "#7f8c8d", "knn_fgbg": "#2ecc71"}
POOL_STYLE: dict[str, str] = {"concat": "-", "max": "--"}

SEED = 0
torch.manual_seed(SEED)

OUTPUT_DIR = _REPO_ROOT / "outputs" / "fundamental" / "scale_composition_max_pool"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log.info(
    "RUN_PART_TYPES=%s ref_number=%d query_number=%d  |  DINO%s-%s img_size=%d layer=%d  |  "
    "n_scale_steps=%d",
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


# %% Composition combo table — identical construction to scale_composition_oracle_iou.py's
# COMPOSITION_COMBOS (single scale, prefix-from-global, suffix-from-close, classic 3-point,
# anchored-inward/outward). Reused verbatim (not imported — every fundamental script here is
# self-contained), scored below under both pooling modes.
COMPOSITION_COMBOS: dict[str, list[str]] = {}
for _name in SCALE_NAMES:
    COMPOSITION_COMBOS[_name] = [_name]
PREFIX_NAMES: list[str] = []
for _i in range(2, len(SCALE_NAMES) + 1):
    _members = SCALE_NAMES[:_i]
    _key = "+".join(_members)
    COMPOSITION_COMBOS[_key] = _members
    PREFIX_NAMES.append(_key)
SUFFIX_NAMES: list[str] = []
for _i in range(2, len(SCALE_NAMES) + 1):
    _members = SCALE_NAMES[-_i:]
    _key = "+".join(_members)
    if _key not in COMPOSITION_COMBOS:
        COMPOSITION_COMBOS[_key] = _members
    SUFFIX_NAMES.append(_key)
if "mid" in SCALE_NAMES:
    COMPOSITION_COMBOS["global+mid+close"] = ["global", "mid", "close"]
_MIDDLE_NAMES = SCALE_NAMES[1:-1]
ANCHORED_INWARD_NAMES: list[str] = []
for _i in range(0, len(_MIDDLE_NAMES) + 1):
    _members = ["global", *_MIDDLE_NAMES[:_i], "close"]
    _key = "+".join(_members)
    COMPOSITION_COMBOS[_key] = _members
    ANCHORED_INWARD_NAMES.append(_key)
ANCHORED_OUTWARD_NAMES: list[str] = []
for _i in range(0, len(_MIDDLE_NAMES) + 1):
    _members = ["global", *_MIDDLE_NAMES[len(_MIDDLE_NAMES) - _i :], "close"]
    _key = "+".join(_members)
    if _key not in COMPOSITION_COMBOS:
        COMPOSITION_COMBOS[_key] = _members
    ANCHORED_OUTWARD_NAMES.append(_key)
log.info(
    "Composition combos: %d unique (single/prefix/suffix/classic/anchored)", len(COMPOSITION_COMBOS)
)


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

# %% Part 2 — build the N_SCALE_STEPS + 1 crops per combo
usable_combo_keys: set[tuple] = set()
for combo in tqdm(combos, desc="Building scale-step crops"):
    ref_img = ref_images[combo["part_type"]]
    group_mask = group_ref_masks.get((combo["part_type"], combo["group"]), combo["ref_mask"])
    boxes = scale_step_boxes(combo["ref_mask"], T_VALUES, CROP_PADDING_FRACTION)
    close_box = boxes[-1]
    if close_box[2] - close_box[0] < MIN_CROP_SIZE or close_box[3] - close_box[1] < MIN_CROP_SIZE:
        log.warning(
            "combo=%s: closest crop %s below MIN_CROP_SIZE=%dpx — skipping every scale step",
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

log.info("Combos with every scale step usable: %d/%d", len(usable_combo_keys), len(combos))

# %% Part 3 — encoder + query-image patch tokens + GT patch masks
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

query_encodings: dict[str, tuple[torch.Tensor, int, int]] = {}
for part_type in tqdm(sorted(query_images), desc="Encoding query images"):
    tokens, q_h, q_w = extract_patch_tokens(
        encoder, query_images[part_type], LAYER_IDX, debias=True
    )
    query_encodings[part_type] = (tokens, q_h, q_w)

gt_patch_masks: dict[tuple[str, str], np.ndarray] = {}
for (part_type, group), pixel_mask in group_query_masks.items():
    _, q_h, q_w = query_encodings[part_type]
    gt_patch_masks[(part_type, group)] = pixel_mask_to_patch_mask(
        pixel_mask, q_h, q_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )

# %% Part 4 — encode every combo's scale-step crops, split into per-scale fg/bg token banks
fg_by_scale: dict[tuple, torch.Tensor] = {}
bg_by_scale: dict[tuple, torch.Tensor] = {}

clean_items: list[tuple] = []
for combo in combos:
    ck = combo_key(combo)
    if ck not in usable_combo_keys:
        continue
    for name, crop in combo["crops"].items():
        clean_items.append((ck, name, crop["img"], crop["mask_px"], crop["bg_exclude_mask_px"]))

for i in tqdm(range(0, len(clean_items), chunk_size), desc="Encoding scale-step crops"):
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
log.info("Built per-scale fg/bg galleries for %d combos", len(usable_combo_keys))

# %% Part 5 — score every composition entry under BOTH pooling modes: concat (today's design,
# one bank/prototype from every member scale's pooled patches) and max (score each member
# scale separately, take the per-query-patch max across scales before oracle-thresholding).
# pool_iou_lookup[pool_style][composition_name][method][ck] -> oracle IoU
PoolIouLookup = dict[str, dict[str, dict[str, dict[tuple, float]]]]
pool_iou_lookup: PoolIouLookup = {
    style: {name: {method: {} for method in METHODS} for name in COMPOSITION_COMBOS}
    for style in ["concat", "max"]
}

for combo in tqdm(combos, desc="Part 5: scoring concat vs max pooling"):
    ck = combo_key(combo)
    if ck not in usable_combo_keys:
        continue
    part_type, group = ck[0], ck[1]
    gt = gt_patch_masks.get((part_type, group))
    if gt is None:
        continue
    q_tokens, q_h, q_w = query_encodings[part_type]
    bg_bank = bg_all_lookup[ck]

    for composition_name, members in COMPOSITION_COMBOS.items():
        # concat: one bank/prototype pooled from every member scale (== the sibling script)
        fg_concat = torch.cat([fg_by_scale[(ck, m)] for m in members], dim=0)
        proto_concat = compute_exemplar_features(fg_concat, mode="mean")
        raw_proto_concat = score_heatmap(q_tokens, proto_concat, q_h, q_w)
        pool_iou_lookup["concat"][composition_name]["single_proto"][ck] = oracle_iou(
            raw_proto_concat, gt, ORACLE_THRESHOLD_STEPS
        )
        raw_knn_concat = knn_score_heatmap(
            q_tokens, fg_concat, bg_bank, KNN_FGBG_NUM_NEIGHBOURS, q_h, q_w
        )
        pool_iou_lookup["concat"][composition_name]["knn_fgbg"][ck] = oracle_iou(
            raw_knn_concat, gt, ORACLE_THRESHOLD_STEPS
        )

        # max: score each member scale separately, take the per-query-patch max across scales
        proto_maps = np.stack(
            [
                score_heatmap(
                    q_tokens,
                    compute_exemplar_features(fg_by_scale[(ck, m)], mode="mean"),
                    q_h,
                    q_w,
                )
                for m in members
            ]
        )
        raw_proto_max = proto_maps.max(axis=0)
        pool_iou_lookup["max"][composition_name]["single_proto"][ck] = oracle_iou(
            raw_proto_max, gt, ORACLE_THRESHOLD_STEPS
        )
        knn_maps = np.stack(
            [
                knn_score_heatmap(
                    q_tokens, fg_by_scale[(ck, m)], bg_bank, KNN_FGBG_NUM_NEIGHBOURS, q_h, q_w
                )
                for m in members
            ]
        )
        raw_knn_max = knn_maps.max(axis=0)
        pool_iou_lookup["max"][composition_name]["knn_fgbg"][ck] = oracle_iou(
            raw_knn_max, gt, ORACLE_THRESHOLD_STEPS
        )

log.info(
    "Scoring complete: %d combos x %d composition entries x %d methods x 2 pooling modes",
    len(usable_combo_keys),
    len(COMPOSITION_COMBOS),
    len(METHODS),
)


# %% Part 6 — full comparison table + headline: does max recover what concat lost?
def mean_std_iou(
    lookup: dict[tuple, float], combo_keys: set[tuple] | None = None
) -> tuple[float, float, int]:
    vals = [v for ck, v in lookup.items() if combo_keys is None or ck in combo_keys]
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


comparison_rows = []
for name, members in COMPOSITION_COMBOS.items():
    for method in METHODS:
        concat_mean, concat_std, n = mean_std_iou(pool_iou_lookup["concat"][name][method])
        max_mean, max_std = mean_std_iou(pool_iou_lookup["max"][name][method])[:2]
        comparison_rows.append(
            {
                "composition": name,
                "n_scales": len(members),
                "method": method,
                "concat_mean_iou": concat_mean,
                "concat_std_iou": concat_std,
                "max_mean_iou": max_mean,
                "max_std_iou": max_std,
                "max_minus_concat": max_mean - concat_mean,
                "n_combos": n,
            }
        )
comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(OUTPUT_DIR / "pooling_comparison.csv", index=False)
log.info("Wrote %s (%d rows)", OUTPUT_DIR / "pooling_comparison.csv", len(comparison_df))

# Single best FIXED scale per method (from concat's own single-scale rows — identical to
# scale_composition_oracle_iou.py's fixed-best-scale, recomputed here for self-containment).
best_single_scale_iou = {
    method: max(mean_std_iou(pool_iou_lookup["concat"][name][method])[0] for name in SCALE_NAMES)
    for method in METHODS
}
best_max_composition = {
    method: comparison_df.loc[comparison_df.method == method]
    .sort_values("max_mean_iou", ascending=False)
    .iloc[0]
    for method in METHODS
}
for method in METHODS:
    row = best_max_composition[method]
    log.info(
        "%-13s best single scale=%.3f | best MAX composition=%s (%d scales) iou=%.3f "
        "(vs. its own concat=%.3f, delta=%+.3f)",
        method,
        best_single_scale_iou[method],
        row.composition,
        row.n_scales,
        row.max_mean_iou,
        row.concat_mean_iou,
        row.max_minus_concat,
    )
    beats_single = row.max_mean_iou > best_single_scale_iou[method]
    log.info(
        "  -> max-pooled composition %s the best single scale",
        "BEATS" if beats_single else "does NOT beat",
    )

# %% Part 7 — growth curves: concat (solid) vs max (dashed), per method, prefix-from-global only
# (the clearest single view of "does the gap between concat and max widen as more scales are
# added" — full 3-panel x 2-pooling-mode grid would be 6 panels per method; kept to one
# representative growth direction per method here, full numbers are in pooling_comparison.csv).
fig, axes = plt.subplots(1, len(METHODS), figsize=(7 * len(METHODS), 5.5), sharey=True)
growth_names = SCALE_NAMES[:1] + PREFIX_NAMES
xs = [len(COMPOSITION_COMBOS[name]) for name in growth_names]
for ax, method in zip(axes, METHODS):
    for style in ["concat", "max"]:
        means = [mean_std_iou(pool_iou_lookup[style][name][method])[0] for name in growth_names]
        stds = [mean_std_iou(pool_iou_lookup[style][name][method])[1] for name in growth_names]
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            label=style,
            color=METHOD_COLOR[method],
            linestyle=POOL_STYLE[style],
            alpha=1.0 if style == "max" else 0.6,
        )
    ax.axhline(
        best_single_scale_iou[method],
        linestyle=":",
        color="black",
        alpha=0.4,
        label="best single scale",
    )
    ax.set_xticks(range(1, len(SCALE_NAMES) + 1), [str(n) for n in range(1, len(SCALE_NAMES) + 1)])
    ax.set_xlabel("number of scale steps composed (growing inward from global)")
    ax.set_title(method)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("oracle IoU (mean +/- std across combos)")
fig.suptitle(
    f"Concat- vs. max-pooled composition, n={N_SCALE_STEPS} steps ({len(usable_combo_keys)} combos)"
)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "pooling_growth_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved %s", OUTPUT_DIR / "pooling_growth_comparison.png")

# %% [markdown]
# ## Reading the results
#
# - **`pooling_comparison.csv`**'s `max_minus_concat` column is the core result: positive means
#   scoring each scale separately and taking the per-query-patch max recovers some of what
#   concatenation pooling diluted away; near-zero or negative means the two composition entries'
#   *members* just aren't complementary regardless of how they're combined — the failure in
#   `scale_composition_oracle_iou.py` was about which scales were combined, not how.
# - **The `BEATS`/`does NOT beat` log line** is the headline: does *any* max-pooled composition
#   finally beat the single best fixed scale that concat pooling couldn't? If max-pooling still
#   can't beat the single best scale either, that's a much stronger version of the original
#   negative result — composing scale information doesn't help this task at all, under either
#   combination strategy tried so far.
# - **`pooling_growth_comparison.png`** only shows the prefix-from-global growth direction for
#   readability (six pooling-mode/growth-direction panels per method would be hard to read at
#   once) — check `pooling_comparison.csv` directly for suffix-from-close or anchored entries.

# %%
