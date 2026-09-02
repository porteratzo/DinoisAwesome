# %% [markdown]
# # Fundamental: Scale Composition — Adaptive-Scale Oracle + Instance-Size Correlation
#
# `scale_composition_oracle_iou.py` reported one dataset-wide winner per method (`4/6` for both)
# — a single fixed scale applied to every instance. Two follow-up questions that script can't
# answer from its own saved CSVs (only per-scale *means* were persisted, not per-instance raw
# IoU):
#
#   1. **Adaptive-scale oracle**: if you could pick the best of the `N_SCALE_STEPS + 1` scales
#      *per instance* instead of one fixed scale for the whole dataset, how much headroom is
#      there over the fixed-best-scale baseline? `max` over each instance's own 7 scale IoUs,
#      averaged across instances, vs. the fixed scale that maximizes the dataset-wide mean.
#   2. **Does the optimal scale correlate with instance size?** Hypothesis: a large instance
#      (bbox already covers a big fraction of the reference image) needs less "zoom" than a tiny
#      one to reach the same effective magnification, so its optimal t should sit lower (closer
#      to `global`). Checked via each instance's own bbox-area fraction vs. its per-instance
#      argmax-t, both methods.
#
# Parts 1-4 are copied verbatim from `scale_composition_oracle_iou.py` (see that script's module
# docstring for why every fundamental script here is self-contained).

# %% Logging — must be before torch import
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("scale_composition_adaptive_oracle")

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

SEED = 0
torch.manual_seed(SEED)

OUTPUT_DIR = _REPO_ROOT / "outputs" / "fundamental" / "scale_composition_adaptive_oracle"
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

# %% Part 2 — build the N_SCALE_STEPS + 1 crops per combo, plus each instance's own bbox-area
# fraction of the full reference image (for the size-vs-optimal-t correlation).
usable_combo_keys: set[tuple] = set()
instance_size_fraction: dict[tuple, float] = {}
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
    ck = combo_key(combo)
    usable_combo_keys.add(ck)
    H, W = combo["ref_mask"].shape
    instance_size_fraction[ck] = float(combo["ref_mask"].sum()) / float(H * W)

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

# %% Part 5 — per-combo, per-scale raw IoU (kept per-instance, not just aggregated — needed for
# both the adaptive oracle and the size-correlation below).
# per_instance_iou[method][scale][ck] -> oracle IoU
PerInstanceIou = dict[str, dict[str, dict[tuple, float]]]
per_instance_iou: PerInstanceIou = {
    method: {name: {} for name in SCALE_NAMES} for method in METHODS
}

for combo in tqdm(combos, desc="Part 5: scoring per scale"):
    ck = combo_key(combo)
    if ck not in usable_combo_keys:
        continue
    part_type, group = ck[0], ck[1]
    gt = gt_patch_masks.get((part_type, group))
    if gt is None:
        continue
    q_tokens, q_h, q_w = query_encodings[part_type]
    bg_bank = bg_all_lookup[ck]

    for name in SCALE_NAMES:
        fg_bank = fg_by_scale[(ck, name)]

        proto = compute_exemplar_features(fg_bank, mode="mean")
        raw_proto = score_heatmap(q_tokens, proto, q_h, q_w)
        per_instance_iou["single_proto"][name][ck] = oracle_iou(
            raw_proto, gt, ORACLE_THRESHOLD_STEPS
        )

        raw_knn = knn_score_heatmap(q_tokens, fg_bank, bg_bank, KNN_FGBG_NUM_NEIGHBOURS, q_h, q_w)
        per_instance_iou["knn_fgbg"][name][ck] = oracle_iou(raw_knn, gt, ORACLE_THRESHOLD_STEPS)

log.info(
    "Scoring complete: %d combos x %d scales x %d methods",
    len(usable_combo_keys),
    len(SCALE_NAMES),
    len(METHODS),
)

# %% Part 6 — adaptive-scale oracle: per-instance max over scales vs. dataset-wide fixed-best-scale
per_instance_rows = []
for ck in sorted(usable_combo_keys):
    row = {
        "part_type": ck[0],
        "group": ck[1],
        "class": ck[2],
        "instance_id": ck[3],
        "size_fraction": instance_size_fraction[ck],
    }
    for method in METHODS:
        ious = [per_instance_iou[method][name][ck] for name in SCALE_NAMES]
        best_i = int(np.argmax(ious))
        row[f"{method}_best_scale"] = SCALE_NAMES[best_i]
        row[f"{method}_best_t"] = float(T_VALUES[best_i])
        row[f"{method}_adaptive_iou"] = ious[best_i]
    per_instance_rows.append(row)
per_instance_df = pd.DataFrame(per_instance_rows)
per_instance_df.to_csv(OUTPUT_DIR / "per_instance_adaptive.csv", index=False)
log.info("Wrote %s (%d rows)", OUTPUT_DIR / "per_instance_adaptive.csv", len(per_instance_df))

adaptive_summary_rows = []
for method in METHODS:
    fixed_means = {
        name: np.mean(list(per_instance_iou[method][name].values())) for name in SCALE_NAMES
    }
    fixed_best_scale = max(fixed_means, key=lambda n: fixed_means[n])
    fixed_best_mean = fixed_means[fixed_best_scale]
    adaptive_mean = float(per_instance_df[f"{method}_adaptive_iou"].mean())
    adaptive_summary_rows.append(
        {
            "method": method,
            "fixed_best_scale": fixed_best_scale,
            "fixed_best_mean_iou": fixed_best_mean,
            "adaptive_oracle_mean_iou": adaptive_mean,
            "headroom": adaptive_mean - fixed_best_mean,
        }
    )
adaptive_summary_df = pd.DataFrame(adaptive_summary_rows)
adaptive_summary_df.to_csv(OUTPUT_DIR / "adaptive_oracle_summary.csv", index=False)
log.info("Adaptive-scale oracle vs. fixed-best-scale:")
for _, row in adaptive_summary_df.iterrows():
    log.info(
        "  %-13s fixed_best=%s (%.3f)  adaptive_oracle=%.3f  headroom=+%.3f",
        row.method,
        row.fixed_best_scale,
        row.fixed_best_mean_iou,
        row.adaptive_oracle_mean_iou,
        row.headroom,
    )

fig, ax = plt.subplots(figsize=(7, 5.5))
x = np.arange(len(METHODS))
width = 0.35
ax.bar(
    x - width / 2,
    adaptive_summary_df["fixed_best_mean_iou"],
    width,
    label="fixed best scale (dataset-wide)",
    color=[METHOD_COLOR[m] for m in METHODS],
    alpha=0.5,
)
ax.bar(
    x + width / 2,
    adaptive_summary_df["adaptive_oracle_mean_iou"],
    width,
    label="adaptive oracle (best per instance)",
    color=[METHOD_COLOR[m] for m in METHODS],
)
for i, row in adaptive_summary_df.iterrows():
    ax.text(
        i + width / 2,
        row.adaptive_oracle_mean_iou + 0.01,
        f"+{row.headroom:.3f}",
        ha="center",
        fontsize=9,
    )
ax.set_xticks(x, METHODS)
ax.set_ylabel("oracle IoU (mean across combos)")
ax.set_ylim(0, 1.0)
ax.set_title(
    f"Adaptive per-instance scale oracle vs. fixed best scale ({len(usable_combo_keys)} combos)"
)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "adaptive_oracle_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved %s", OUTPUT_DIR / "adaptive_oracle_summary.png")

# %% Part 7 — instance size vs. optimal scale correlation
correlation_rows = []
fig, axes = plt.subplots(1, len(METHODS), figsize=(6.5 * len(METHODS), 5.5), sharey=True)
for ax, method in zip(axes, METHODS):
    sizes = per_instance_df["size_fraction"].to_numpy()
    best_t = per_instance_df[f"{method}_best_t"].to_numpy()
    pearson_r, pearson_p = pearsonr(sizes, best_t)
    spearman_r, spearman_p = spearmanr(sizes, best_t)
    correlation_rows.append(
        {
            "method": method,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "n": len(sizes),
        }
    )
    ax.scatter(sizes, best_t, color=METHOD_COLOR[method], alpha=0.7)
    ax.set_xlabel("instance bbox-mask area / full ref image area")
    ax.set_title(
        f"{method}\npearson r={pearson_r:.2f} (p={pearson_p:.3f}), n={len(sizes)}", fontsize=9
    )
    ax.grid(alpha=0.3)
axes[0].set_ylabel("optimal t (0=global, 1=close)")
fig.suptitle("Instance size vs. per-instance optimal scale")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "size_vs_optimal_t.png", dpi=150, bbox_inches="tight")
plt.close(fig)

correlation_df = pd.DataFrame(correlation_rows)
correlation_df.to_csv(OUTPUT_DIR / "size_vs_optimal_t_correlation.csv", index=False)
log.info("Instance-size vs. optimal-t correlation:")
for _, row in correlation_df.iterrows():
    log.info(
        "  %-13s pearson_r=%.3f (p=%.3f)  spearman_r=%.3f (p=%.3f)  n=%d",
        row.method,
        row.pearson_r,
        row.pearson_p,
        row.spearman_r,
        row.spearman_p,
        row.n,
    )
log.info(
    "Saved %s and %s",
    OUTPUT_DIR / "size_vs_optimal_t.png",
    OUTPUT_DIR / "size_vs_optimal_t_correlation.csv",
)

# %% [markdown]
# ## Reading the results
#
# - **`adaptive_oracle_summary.png`/`.csv`**: the "+headroom" number is the ceiling available
#   from a smarter, per-instance scale-selection policy over what a single hardcoded scale
#   already captures. A small headroom means the fixed-best-scale choice from
#   `scale_composition_oracle_iou.py` is already close to as good as it gets; a large one means
#   an adaptive policy (e.g. picking scale from instance size, see Part 7) is worth building.
# - **`size_vs_optimal_t.png`/`size_vs_optimal_t_correlation.csv`**: tests the specific
#   hypothesis that bigger instances want a lower (more global) optimal t. A weak/insignificant
#   correlation means whatever *is* driving per-instance optimal-scale variation, it isn't
#   simply "how big is the object in the reference image" — worth checking other per-instance
#   properties (aspect ratio, texture uniformity, occlusion) instead.
# - Both analyses pool every instance-type group together; a per-group correlation could differ
#   if, e.g., `donut foam`'s size range barely varies but `white_clips`'s does — worth splitting
#   by group if the pooled correlation looks weak but the per-group scatter looks structured.

# %%
