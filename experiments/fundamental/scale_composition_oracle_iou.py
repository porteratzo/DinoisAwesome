# %% [markdown]
# # Fundamental: Scale Composition — Finer Scale Steps + Multi-Scale Fg/Bg Composition
#
# Every sibling script in `experiments/fundamental/` and `object_detection/multiscale_ablation/`
# builds `single_proto`/`fg-bg-knn` galleries from exactly three named crop scales — `close`,
# `mid`, `global` (`_shared.mask_geometry.scale_crop_box`), where `mid` is literally the t=0.5
# midpoint between `close` (t=1) and `global` (t=0). This experiment generalizes that fixed
# 3-point sweep into `N_SCALE_STEPS + 1` evenly-spaced crop scales (t = 0, 1/n, 2/n, ..., 1 —
# same linear interpolation `scale_crop_similarity.py` already uses for its single-instance case
# study, applied here across the whole abc3 dataset for real fg/bg IoU) and asks two questions:
#
#   1. **Per-scale**: how does oracle IoU (best patch-mask IoU any single threshold on the raw
#      score map achieves against GT, `_shared.thresholding.oracle_iou`) for `single_proto` and
#      `fg-bg-knn` change as the crop tightens step by step, and what's the average IoU across
#      all scale steps?
#   2. **Composition**: does *combining* several scale steps' foreground patches into one bank —
#      e.g. `global`, `global+1/6`, `global+1/6+2/6`, ..., all the way to every scale, and the
#      mirror-image sweep growing outward from `close` (`close`, `close+5/6`, `close+5/6+4/6`,
#      ...) — beat any single scale alone? Background is always pooled from *every* scale step
#      (mirrors `object_detection/multiscale_ablation/methods.py`'s `FGBG_SOURCE_COMBOS`, whose
#      bg side always spans `global+mid+close` regardless of which scale(s) the fg side uses),
#      so composition only changes what's on the foreground side.
#
# Composition is evaluated over a curated table, not every subset of scales (2^(n+1) - 1 = 127
# combos for n=6 would multiply every per-combo scoring pass a hundred-fold for figures nobody
# would read one at a time). Five families, ~4n+3 combos instead of 2^(n+1):
#   - single scale (n+1) — each scale step alone (Part 6's per-scale curve).
#   - prefix-from-global / suffix-from-close (2n, `PREFIX_NAMES`/`SUFFIX_NAMES`) — open-ended
#     growth from one endpoint, dropping the other until the very last step.
#   - the classic 3-point `global+mid+close` baseline — every sibling script's fixed combo,
#     included once for direct comparison against the finer-grained sweeps.
#   - anchored-inward / anchored-outward (2(n-1), `ANCHORED_INWARD_NAMES`/`ANCHORED_OUTWARD_NAMES`)
#     — keep BOTH `global` and `close` in every entry (matches the classic combo's own logic of
#     "always cover both extremes") and grow the middle from one side or the other.
#
# Scored exactly like `augmented_prototype_oracle_iou_knn_fgbg.py` /
# `feature_transform_oracle_iou.py`: `single_proto` (masked-mean cosine similarity) and
# `fg-bg-knn` (per-patch contrastive kNN, `_shared.prototype_ops.knn_fgbg_score`), oracle IoU
# per (part_type, group, instance) combo, pooled into a dataset-wide mean +/- std.

# %% Logging — must be before torch import
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("scale_composition_oracle_iou")

from collections import defaultdict
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

# Same knob as the sibling fundamental scripts — narrow for fast iteration, e.g. ["LHa"].
RUN_PART_TYPES: list[str] = PART_TYPES

# Same focus combo as the sibling fundamental scripts, for direct comparability of figures.
FOCUS_PART_TYPE = "LHa"
FOCUS_CLASS = "donut foam single"
FOCUS_INSTANCE_ID = 1

DINO_VERSION = "v3"
DINO_SIZE = "large"
IMG_SIZE = 768
LAYER_IDX = 23
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")
DINO_ENCODING_CACHE_DIR: str | None = os.environ.get("DINO_ENCODING_CACHE_DIR")

MASK_PATCH_THRESHOLD = 0.3
CROP_PADDING_FRACTION = 1.0  # matches close/mid's own padding in every sibling script
MIN_CROP_SIZE = 128

ORACLE_THRESHOLD_STEPS = 25
KNN_FGBG_NUM_NEIGHBOURS = 10  # same default multiscale_crop_ablation.py's fg-bg-knn uses

# n in "global-.../close, n=6" — number of steps from global (t=0) to close (t=1); gives
# N_SCALE_STEPS + 1 scale points. n=6 reproduces today's "mid" exactly at t=3/6=0.5.
N_SCALE_STEPS = 6

METHODS: list[str] = ["single_proto", "knn_fgbg"]
METHOD_COLOR: dict[str, str] = {"single_proto": "#7f8c8d", "knn_fgbg": "#2ecc71"}

SEED = 0
torch.manual_seed(SEED)

OUTPUT_DIR = _REPO_ROOT / "outputs" / "fundamental" / "scale_composition_oracle_iou"
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

# %% Scale-step naming + crop-box geometry
T_VALUES: np.ndarray = np.linspace(0.0, 1.0, N_SCALE_STEPS + 1)


def scale_step_name(i: int, n: int) -> str:
    """t=0 -> "global", t=1 -> "close", the exact halfway point -> "mid" (matches today's
    naming when n is even), everything else -> its own fraction, e.g. "2/6"."""
    if i == 0:
        return "global"
    if i == n:
        return "close"
    if n % 2 == 0 and i == n // 2:
        return "mid"
    return f"{i}/{n}"


SCALE_NAMES: list[str] = [scale_step_name(i, N_SCALE_STEPS) for i in range(N_SCALE_STEPS + 1)]
SCALE_COLOR: dict[str, str] = {
    name: plt.get_cmap("viridis")(t) for name, t in zip(SCALE_NAMES, T_VALUES)
}
log.info("Scale steps (global -> close): %s", SCALE_NAMES)


def scale_step_boxes(
    pixel_mask: np.ndarray, t_values: np.ndarray, padding_frac: float
) -> list[tuple[int, int, int, int]]:
    """PIL-style crop boxes linearly interpolated from the whole image (t=0) to `close`'s own
    tight, padded bbox (t=1) — same interpolation scale_crop_similarity.py's own
    `scale_crop_boxes` uses, generalizing `scale_crop_box`'s fixed global/mid/close named
    points to arbitrary t. Boxes shrink monotonically as t grows, so `close` (t=1, the
    smallest) meeting MIN_CROP_SIZE guarantees every other t does too."""
    H, W = pixel_mask.shape
    close_box = scale_crop_box(pixel_mask, "close", padding_frac)
    global_box = (0, 0, W, H)
    return [
        tuple(int(round(a + (b - a) * t)) for a, b in zip(global_box, close_box)) for t in t_values
    ]


# %% Composition combo table — not the full power set (see module docstring), but more than
# just the two open-ended growth sweeps: also the classic 3-point `global+mid+close` baseline
# (the fixed combo every sibling script's FGBG_SOURCE_COMBOS already uses), and two "anchored"
# growth sweeps that keep BOTH endpoints in every entry and grow the middle from one side —
# "anchored_inward" adds middle scales moving away from global (mirrors PREFIX_NAMES but never
# drops `close`), "anchored_outward" adds them moving away from close (mirrors SUFFIX_NAMES but
# never drops `global`). ~4n+3 combos instead of 2^(n+1) - 1.
COMPOSITION_COMBOS: dict[str, list[str]] = {}
for _name in SCALE_NAMES:
    COMPOSITION_COMBOS[_name] = [_name]  # every single scale, on its own
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
    if _key not in COMPOSITION_COMBOS:  # i == len(SCALE_NAMES) duplicates the full prefix
        COMPOSITION_COMBOS[_key] = _members
    SUFFIX_NAMES.append(_key)

if "mid" in SCALE_NAMES:
    COMPOSITION_COMBOS["global+mid+close"] = ["global", "mid", "close"]

_MIDDLE_NAMES = SCALE_NAMES[1:-1]  # every scale strictly between global and close
ANCHORED_INWARD_NAMES: list[str] = []
for _i in range(0, len(_MIDDLE_NAMES) + 1):
    _members = ["global", *_MIDDLE_NAMES[:_i], "close"]
    _key = "+".join(_members)
    COMPOSITION_COMBOS[_key] = _members  # i=0 -> "global+close"; i=len(_MIDDLE_NAMES) -> full set
    ANCHORED_INWARD_NAMES.append(_key)
ANCHORED_OUTWARD_NAMES: list[str] = []
for _i in range(0, len(_MIDDLE_NAMES) + 1):
    _members = ["global", *_MIDDLE_NAMES[len(_MIDDLE_NAMES) - _i :], "close"]
    _key = "+".join(_members)
    if _key not in COMPOSITION_COMBOS:  # i=0 and i=len(_MIDDLE_NAMES) duplicate inward's ends
        COMPOSITION_COMBOS[_key] = _members
    ANCHORED_OUTWARD_NAMES.append(_key)

log.info(
    "Composition combos: %d single-scale + %d prefix-from-global + %d suffix-from-close + "
    "1 classic 3-point + %d anchored-inward + %d anchored-outward (%d unique total)",
    len(SCALE_NAMES),
    len(PREFIX_NAMES),
    len(SUFFIX_NAMES),
    len(ANCHORED_INWARD_NAMES),
    len(ANCHORED_OUTWARD_NAMES),
    len(COMPOSITION_COMBOS),
)

# %% Helper: split one crop's patch tokens into (fg, bg), L2-normalised. Identical in spirit to
# every sibling script's local `split_fg_bg_patches` — kept self-contained per-file rather than
# shared, matching this directory's existing convention.


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

# %% Part 2 — build the N_SCALE_STEPS + 1 crops per combo. All-or-nothing per combo: boxes
# shrink monotonically from global to close, so if the closest one clears MIN_CROP_SIZE every
# other step does too (see scale_step_boxes's docstring) — a combo either gets every scale or
# is skipped entirely, no partial scale availability to special-case downstream.
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
fg_by_scale: dict[tuple, torch.Tensor] = {}  # (ck, scale_name) -> (Nfg, C)
bg_by_scale: dict[tuple, torch.Tensor] = {}  # (ck, scale_name) -> (Nbg, C)

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

# Background is scale-composition-invariant: every combo's every composition, single-scale or
# multi-scale, is scored against the SAME pooled bg gallery spanning every scale step (mirrors
# FGBG_SOURCE_COMBOS's bg=["global","mid","close"] always-full-span default).
bg_all_lookup: dict[tuple, torch.Tensor] = {
    ck: torch.cat([bg_by_scale[(ck, name)] for name in SCALE_NAMES], dim=0)
    for ck in usable_combo_keys
}
log.info("Built per-scale fg/bg galleries for %d combos", len(usable_combo_keys))

# %% Part 5 — main per-combo, per-composition-combo scoring
IouLookup = dict[str, dict[str, dict[tuple, float]]]  # composition_name -> method -> ck -> iou
iou_lookup: IouLookup = {name: {method: {} for method in METHODS} for name in COMPOSITION_COMBOS}

for combo in tqdm(combos, desc="Part 5: scoring compositions"):
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
        fg_bank = torch.cat([fg_by_scale[(ck, name)] for name in members], dim=0)

        proto = compute_exemplar_features(fg_bank, mode="mean")
        raw_proto = score_heatmap(q_tokens, proto, q_h, q_w)
        iou_lookup[composition_name]["single_proto"][ck] = oracle_iou(
            raw_proto, gt, ORACLE_THRESHOLD_STEPS
        )

        raw_knn = knn_score_heatmap(q_tokens, fg_bank, bg_bank, KNN_FGBG_NUM_NEIGHBOURS, q_h, q_w)
        iou_lookup[composition_name]["knn_fgbg"][ck] = oracle_iou(
            raw_knn, gt, ORACLE_THRESHOLD_STEPS
        )

log.info(
    "Scoring complete: %d combos x %d composition entries x %d methods",
    len(usable_combo_keys),
    len(COMPOSITION_COMBOS),
    len(METHODS),
)


# %% Part 6 — per-scale results: table + line chart over t, average IoU across scales
def mean_std_iou(
    lookup: IouLookup, composition_name: str, method: str, combo_keys: set[tuple] | None = None
) -> tuple[float, float, int]:
    vals = [
        v
        for ck, v in lookup[composition_name][method].items()
        if combo_keys is None or ck in combo_keys
    ]
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


per_scale_rows = []
for name, t in zip(SCALE_NAMES, T_VALUES):
    for method in METHODS:
        mean, std, n = mean_std_iou(iou_lookup, name, method)
        per_scale_rows.append(
            {
                "scale": name,
                "t": t,
                "method": method,
                "mean_iou": mean,
                "std_iou": std,
                "n_combos": n,
            }
        )
per_scale_df = pd.DataFrame(per_scale_rows)
per_scale_df.to_csv(OUTPUT_DIR / "per_scale_iou.csv", index=False)

average_across_scales = {
    method: float(per_scale_df.loc[per_scale_df.method == method, "mean_iou"].mean())
    for method in METHODS
}
log.info("Per-scale oracle IoU (mean +/- std across %d combos):", len(usable_combo_keys))
for _, row in per_scale_df.iterrows():
    log.info(
        "  scale=%-8s t=%.2f method=%-13s iou=%.3f+/-%.3f (n=%d)",
        row.scale,
        row.t,
        row.method,
        row.mean_iou,
        row.std_iou,
        row.n_combos,
    )
for method, avg in average_across_scales.items():
    log.info(
        "  average IoU across all %d scale steps, method=%s: %.3f", len(SCALE_NAMES), method, avg
    )
pd.DataFrame(
    [{"method": m, "average_iou_across_scales": v} for m, v in average_across_scales.items()]
).to_csv(OUTPUT_DIR / "average_iou_across_scales.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5.5))
for method in METHODS:
    sub = per_scale_df[per_scale_df.method == method]
    ax.errorbar(
        sub["t"],
        sub["mean_iou"],
        yerr=sub["std_iou"],
        marker="o",
        capsize=3,
        label=method,
        color=METHOD_COLOR[method],
    )
    ax.axhline(average_across_scales[method], linestyle="--", color=METHOD_COLOR[method], alpha=0.5)
ax.set_xticks(T_VALUES, SCALE_NAMES, rotation=45)
ax.set_xlabel("crop tightness t (0 = global, 1 = close)")
ax.set_ylabel("oracle IoU (mean +/- std across combos)")
ax.set_ylim(0, 1.0)
ax.set_title(
    f"Per-scale oracle IoU, n={N_SCALE_STEPS} steps ({len(usable_combo_keys)} combos)\n"
    "dashed line = average IoU across all scale steps"
)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "per_scale_iou.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved %s and %s", OUTPUT_DIR / "per_scale_iou.csv", OUTPUT_DIR / "per_scale_iou.png")

# %% Part 7 — composition growth curves (open-ended + anchored-at-both-ends) + full table
composition_rows = []
for name, members in COMPOSITION_COMBOS.items():
    for method in METHODS:
        mean, std, n = mean_std_iou(iou_lookup, name, method)
        composition_rows.append(
            {
                "composition": name,
                "n_scales": len(members),
                "members": "+".join(members),
                "method": method,
                "mean_iou": mean,
                "std_iou": std,
                "n_combos": n,
            }
        )
composition_df = pd.DataFrame(composition_rows)
composition_df.to_csv(OUTPUT_DIR / "composition_iou.csv", index=False)
log.info("Wrote %s (%d rows)", OUTPUT_DIR / "composition_iou.csv", len(composition_df))

fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), sharey=True, sharex=True)
for ax, growth_names, direction in [
    (axes[0], SCALE_NAMES[:1] + PREFIX_NAMES, "growing inward from global"),
    (axes[1], SCALE_NAMES[-1:] + SUFFIX_NAMES, "growing outward from close"),
    (axes[2], ANCHORED_INWARD_NAMES, "anchored at global+close, growing inward"),
]:
    # x = actual scale count (not list position) so all three panels share the same axis —
    # matters for panel 3, whose shortest entry ("global+close") already has 2 scales, not 1.
    xs = [len(COMPOSITION_COMBOS[name]) for name in growth_names]
    for method in METHODS:
        means = [mean_std_iou(iou_lookup, name, method)[0] for name in growth_names]
        stds = [mean_std_iou(iou_lookup, name, method)[1] for name in growth_names]
        ax.errorbar(
            xs, means, yerr=stds, marker="o", capsize=3, label=method, color=METHOD_COLOR[method]
        )
    ax.set_xticks(range(1, len(SCALE_NAMES) + 1), [str(n) for n in range(1, len(SCALE_NAMES) + 1)])
    ax.set_xlabel("number of scale steps composed")
    ax.set_title(direction)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)

# Third panel also gets the outward-from-close anchored sweep (dashed — same x meaning as
# anchored-inward, just grown from the opposite side) and the classic 3-point
# "global+mid+close" baseline as a standalone marker at x=3 — all three share axes[2] since
# they're all "keep both endpoints, vary the middle" variants.
ax2 = axes[2]
for method in METHODS:
    xs_out = [len(COMPOSITION_COMBOS[name]) for name in ANCHORED_OUTWARD_NAMES]
    means = [mean_std_iou(iou_lookup, name, method)[0] for name in ANCHORED_OUTWARD_NAMES]
    stds = [mean_std_iou(iou_lookup, name, method)[1] for name in ANCHORED_OUTWARD_NAMES]
    ax2.errorbar(
        xs_out,
        means,
        yerr=stds,
        marker="s",
        linestyle="--",
        capsize=3,
        alpha=0.6,
        color=METHOD_COLOR[method],
    )
if "global+mid+close" in COMPOSITION_COMBOS:
    for method in METHODS:
        mean = mean_std_iou(iou_lookup, "global+mid+close", method)[0]
        ax2.scatter(
            [3], [mean], marker="D", s=90, color=METHOD_COLOR[method], zorder=5, edgecolors="black"
        )
ax2.set_title(
    "anchored at global+close\n"
    "(o/solid=grow from global, s/dashed=grow from close, diamond=global+mid+close)",
    fontsize=8,
)

axes[0].set_ylabel("oracle IoU (mean +/- std across combos)")
axes[0].legend(fontsize=8)
fig.suptitle(
    f"Scale composition growth curves, n={N_SCALE_STEPS} steps ({len(usable_combo_keys)} combos)"
)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "composition_growth.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved %s", OUTPUT_DIR / "composition_growth.png")

best_composition = {
    method: composition_df.loc[composition_df.method == method]
    .sort_values("mean_iou", ascending=False)
    .iloc[0]
    for method in METHODS
}
for method, row in best_composition.items():
    log.info(
        "Best composition for %s: %s (%d scales) iou=%.3f+/-%.3f, vs. best single scale=%.3f",
        method,
        row.composition,
        row.n_scales,
        row.mean_iou,
        row.std_iou,
        per_scale_df.loc[per_scale_df.method == method, "mean_iou"].max(),
    )

# %% Part 8 — per-group breakdown of the per-scale curve (Part 6). The headline pools every
# instance-type group together, which can hide a group-specific effect — same rationale as
# every sibling fundamental script's own per-group breakdown. Scoped to the per-scale curve
# only (not every composition combo) to keep the number of figures this script produces
# proportional to N_SCALE_STEPS, not to N_SCALE_STEPS times the group count.
combos_by_group: dict[str, list[tuple]] = defaultdict(list)
for combo in combos:
    ck = combo_key(combo)
    if ck in usable_combo_keys:
        combos_by_group[ck[1]].append(ck)

for group, cks in combos_by_group.items():
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in METHODS:
        means = [
            mean_std_iou(iou_lookup, name, method, combo_keys=set(cks))[0] for name in SCALE_NAMES
        ]
        stds = [
            mean_std_iou(iou_lookup, name, method, combo_keys=set(cks))[1] for name in SCALE_NAMES
        ]
        ax.errorbar(
            T_VALUES,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            label=method,
            color=METHOD_COLOR[method],
        )
    ax.set_xticks(T_VALUES, SCALE_NAMES, rotation=45)
    ax.set_xlabel("crop tightness t (0 = global, 1 = close)")
    ax.set_ylabel("oracle IoU")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Per-scale oracle IoU, group={group} (n={len(cks)} combos)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"per_scale_iou__{group.replace(' ', '_')}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
log.info("Saved %d per-group per-scale breakdown charts", len(combos_by_group))

# %% Part 9 — qualitative figure: focus combo's score maps across every scale step, both methods
focus_combo = next(
    (
        c
        for c in combos
        if c["part_type"] == FOCUS_PART_TYPE
        and c["class"] == FOCUS_CLASS
        and c["instance_id"] == FOCUS_INSTANCE_ID
    ),
    combos[0],
)
focus_ck = combo_key(focus_combo)
if focus_ck not in usable_combo_keys:
    log.warning("Focus combo %s has no usable scale steps — skipping qualitative figure", focus_ck)
else:
    focus_part_type, focus_group = focus_ck[0], focus_ck[1]
    focus_gt = gt_patch_masks[(focus_part_type, focus_group)]
    focus_q_tokens, focus_q_h, focus_q_w = query_encodings[focus_part_type]
    focus_bg = bg_all_lookup[focus_ck]

    query_img = query_images[focus_part_type]
    n_panels = 1 + 2 * len(SCALE_NAMES)
    n_cols = len(SCALE_NAMES)
    fig, axes = plt.subplots(3, n_cols, figsize=(2.6 * n_cols, 8.4))

    axes[0, 0].imshow(query_img)
    gt_overlay = np.zeros((*focus_gt.shape, 4))
    gt_overlay[focus_gt] = (0.2, 0.8, 0.2, 0.45)
    axes[0, 0].imshow(gt_overlay, extent=(0, query_img.width, query_img.height, 0))
    axes[0, 0].set_title("query + GT")
    for col in range(1, n_cols):
        axes[0, col].axis("off")
    for col in range(n_cols):
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

    for col, name in enumerate(SCALE_NAMES):
        fg_bank = fg_by_scale[(focus_ck, name)]
        proto = compute_exemplar_features(fg_bank, mode="mean")
        raw_proto = score_heatmap(focus_q_tokens, proto, focus_q_h, focus_q_w)
        axes[1, col].imshow(raw_proto, cmap="magma")
        axes[1, col].set_title(f"{name}\nsingle_proto", fontsize=8)
        axes[1, col].axis("off")

        raw_knn = knn_score_heatmap(
            focus_q_tokens, fg_bank, focus_bg, KNN_FGBG_NUM_NEIGHBOURS, focus_q_h, focus_q_w
        )
        axes[2, col].imshow(raw_knn, cmap="magma")
        axes[2, col].set_title("knn_fgbg", fontsize=8)
        axes[2, col].axis("off")

    fig.suptitle(f"Per-scale score maps — focus combo {focus_ck}")
    fig.tight_layout()
    _focus_path = OUTPUT_DIR / "focus_combo_scale_grid.png"
    fig.savefig(_focus_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved qualitative focus-combo grid to %s", _focus_path)

# %% [markdown]
# ## Reading the results
#
# - **`per_scale_iou.png`/`per_scale_iou.csv`** answer the first question directly: oracle IoU
#   at each of the `N_SCALE_STEPS + 1` crop tightness levels, plus the dashed "average across
#   scales" line. If that curve isn't monotonic, some interior scale beats both endpoints —
#   worth checking against `composition_growth.png` below, since a single winning interior
#   scale doesn't by itself mean *combining* scales helps.
# - **`composition_growth.png`/`composition_iou.csv`** answer the second: two growth curves
#   (adding scales inward from `global`, adding scales outward from `close`), each point built
#   from strictly more foreground patches than the last, background held fixed and full-span.
#   A curve that peaks partway through and then declines means the extra scales' fg patches are
#   diluting the prototype/gallery, not enriching it — a real result, not a failure of the
#   composition to "add up".
# - **Combining always widens the fg gallery, never subsets it** — this experiment doesn't test
#   whether a *specific* subset (e.g. skip every other scale) beats the prefix/suffix sweep;
#   `composition_iou.csv`'s `members` column is limited to the growth table's combos, not a
#   power-set search (see the module docstring's runtime rationale).
# - As with every other fundamental experiment here, the headline charts pool every
#   part-type/group/instance combo together — check `per_scale_iou__<group>.png` before
#   concluding a scale or composition's aggregate win holds for every instance-type group, and
#   use `focus_combo_scale_grid.png` only as one qualitative example, not as the dataset.


# %%
