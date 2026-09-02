# %% [markdown]
# # Fundamental: Scale Composition — Isolating the Background Side
#
# `scale_composition_oracle_iou.py` swept the *foreground* side of the fg/bg gallery through
# `N_SCALE_STEPS + 1` crop scales and a curated composition table, but held the background side
# fixed at "every scale step, always" for every single entry (mirrors
# `object_detection/multiscale_ablation/methods.py`'s `FGBG_SOURCE_COMBOS`, whose bg side always
# spans `global+mid+close` regardless of what the fg side uses). That means the sibling script
# never actually tested whether a multi-scale *background* helps at all — "bg=all scales" was
# never compared against any alternative.
#
# This script isolates that: foreground is held to **each single scale on its own** (`single_proto`
# is bg-invariant by construction — its score map never reads `bg_bank` — so this whole file is
# really a `fg-bg-knn`-only ablation; `single_proto`'s per-fg-scale IoU is recorded once as a
# bg-invariant reference line, not swept), and **background is grown through the exact same
# composition families** `scale_composition_oracle_iou.py` used for foreground: single scale,
# prefix-from-global, suffix-from-close, the classic `global+mid+close` triple, and the two
# anchored-at-both-ends sweeps. Same `N_SCALE_STEPS=6`, same curated-not-power-set rationale (see
# that script's module docstring), same `_shared` scoring primitives.
#
# Parts 1-4 (combo discovery, scale-step crop building, encoder, per-scale fg/bg token banks) are
# copied verbatim from `scale_composition_oracle_iou.py` — every fundamental script in this
# directory is self-contained (no cross-script imports), and this one needs the identical crop
# geometry and per-scale fg/bg banks as its sibling to make a fair comparison.

# %% Logging — must be before torch import
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("scale_composition_bg_ablation")

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

# Same n as scale_composition_oracle_iou.py — n=6 reproduces today's "mid" at t=3/6=0.5.
N_SCALE_STEPS = 6

KNN_COLOR = "#2ecc71"  # matches METHOD_COLOR["knn_fgbg"] in the sibling script
SINGLE_PROTO_COLOR = "#7f8c8d"  # matches METHOD_COLOR["single_proto"]; bg-invariant reference

SEED = 0
torch.manual_seed(SEED)

OUTPUT_DIR = _REPO_ROOT / "outputs" / "fundamental" / "scale_composition_bg_ablation"
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
log.info("Scale steps (global -> close): %s", SCALE_NAMES)


def scale_step_boxes(
    pixel_mask: np.ndarray, t_values: np.ndarray, padding_frac: float
) -> list[tuple[int, int, int, int]]:
    """PIL-style crop boxes linearly interpolated from the whole image (t=0) to `close`'s own
    tight, padded bbox (t=1). Boxes shrink monotonically as t grows, so `close` (t=1, the
    smallest) meeting MIN_CROP_SIZE guarantees every other t does too."""
    H, W = pixel_mask.shape
    close_box = scale_crop_box(pixel_mask, "close", padding_frac)
    global_box = (0, 0, W, H)
    return [
        tuple(int(round(a + (b - a) * t)) for a, b in zip(global_box, close_box)) for t in t_values
    ]


# %% Background composition table — same construction as scale_composition_oracle_iou.py's
# COMPOSITION_COMBOS, applied here to the BACKGROUND side instead of foreground: single scale,
# prefix-from-global, suffix-from-close, classic 3-point, and the two anchored-at-both-ends
# sweeps. Reused verbatim (not imported — see module docstring) so the two scripts' growth
# tables are guaranteed identical.
BG_COMPOSITION_COMBOS: dict[str, list[str]] = {}
for _name in SCALE_NAMES:
    BG_COMPOSITION_COMBOS[_name] = [_name]
BG_PREFIX_NAMES: list[str] = []
for _i in range(2, len(SCALE_NAMES) + 1):
    _members = SCALE_NAMES[:_i]
    _key = "+".join(_members)
    BG_COMPOSITION_COMBOS[_key] = _members
    BG_PREFIX_NAMES.append(_key)
BG_SUFFIX_NAMES: list[str] = []
for _i in range(2, len(SCALE_NAMES) + 1):
    _members = SCALE_NAMES[-_i:]
    _key = "+".join(_members)
    if _key not in BG_COMPOSITION_COMBOS:
        BG_COMPOSITION_COMBOS[_key] = _members
    BG_SUFFIX_NAMES.append(_key)
if "mid" in SCALE_NAMES:
    BG_COMPOSITION_COMBOS["global+mid+close"] = ["global", "mid", "close"]
_MIDDLE_NAMES = SCALE_NAMES[1:-1]
BG_ANCHORED_INWARD_NAMES: list[str] = []
for _i in range(0, len(_MIDDLE_NAMES) + 1):
    _members = ["global", *_MIDDLE_NAMES[:_i], "close"]
    _key = "+".join(_members)
    BG_COMPOSITION_COMBOS[_key] = _members
    BG_ANCHORED_INWARD_NAMES.append(_key)
BG_ANCHORED_OUTWARD_NAMES: list[str] = []
for _i in range(0, len(_MIDDLE_NAMES) + 1):
    _members = ["global", *_MIDDLE_NAMES[len(_MIDDLE_NAMES) - _i :], "close"]
    _key = "+".join(_members)
    if _key not in BG_COMPOSITION_COMBOS:
        BG_COMPOSITION_COMBOS[_key] = _members
    BG_ANCHORED_OUTWARD_NAMES.append(_key)
FULL_BG_NAME = "+".join(SCALE_NAMES)  # "bg=every scale step" — the sibling script's fixed choice

log.info(
    "Background composition combos: %d single-scale + %d prefix-from-global + "
    "%d suffix-from-close + 1 classic 3-point + %d anchored-inward + %d anchored-outward "
    "(%d unique total)",
    len(SCALE_NAMES),
    len(BG_PREFIX_NAMES),
    len(BG_SUFFIX_NAMES),
    len(BG_ANCHORED_INWARD_NAMES),
    len(BG_ANCHORED_OUTWARD_NAMES),
    len(BG_COMPOSITION_COMBOS),
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

# %% Part 2 — build the N_SCALE_STEPS + 1 crops per combo. All-or-nothing per combo (see
# scale_step_boxes's docstring: boxes shrink monotonically, so `close` clearing MIN_CROP_SIZE
# guarantees every other step does too).
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

log.info("Built per-scale fg/bg galleries for %d combos", len(usable_combo_keys))

# %% Part 5 — main scoring: fg fixed to each single scale, bg swept through BG_COMPOSITION_COMBOS.
# single_proto never reads bg_bank (see module docstring), so it's scored once per (combo,
# fg_scale) as a bg-invariant reference, not swept.
# bg_iou_lookup[fg_scale][bg_composition_name][ck] -> knn_fgbg oracle IoU
BgIouLookup = dict[str, dict[str, dict[tuple, float]]]
bg_iou_lookup: BgIouLookup = {
    fg_scale: {bg_name: {} for bg_name in BG_COMPOSITION_COMBOS} for fg_scale in SCALE_NAMES
}
# fg_only_iou[fg_scale][ck] -> single_proto oracle IoU (bg-invariant reference)
FgOnlyIou = dict[str, dict[tuple, float]]
fg_only_iou: FgOnlyIou = {fg_scale: {} for fg_scale in SCALE_NAMES}

for combo in tqdm(combos, desc="Part 5: scoring bg ablation"):
    ck = combo_key(combo)
    if ck not in usable_combo_keys:
        continue
    part_type, group = ck[0], ck[1]
    gt = gt_patch_masks.get((part_type, group))
    if gt is None:
        continue
    q_tokens, q_h, q_w = query_encodings[part_type]

    for fg_scale in SCALE_NAMES:
        fg_bank = fg_by_scale[(ck, fg_scale)]

        proto = compute_exemplar_features(fg_bank, mode="mean")
        raw_proto = score_heatmap(q_tokens, proto, q_h, q_w)
        fg_only_iou[fg_scale][ck] = oracle_iou(raw_proto, gt, ORACLE_THRESHOLD_STEPS)

        for bg_name, bg_members in BG_COMPOSITION_COMBOS.items():
            bg_bank = torch.cat([bg_by_scale[(ck, m)] for m in bg_members], dim=0)
            raw_knn = knn_score_heatmap(
                q_tokens, fg_bank, bg_bank, KNN_FGBG_NUM_NEIGHBOURS, q_h, q_w
            )
            bg_iou_lookup[fg_scale][bg_name][ck] = oracle_iou(raw_knn, gt, ORACLE_THRESHOLD_STEPS)

log.info(
    "Scoring complete: %d combos x %d fg scales x %d bg compositions (knn_fgbg) + "
    "%d combos x %d fg scales (single_proto, bg-invariant reference)",
    len(usable_combo_keys),
    len(SCALE_NAMES),
    len(BG_COMPOSITION_COMBOS),
    len(usable_combo_keys),
    len(SCALE_NAMES),
)


# %% Part 6 — full table + headline comparison: own-scale-only bg vs. global-only bg vs.
# bg=every-scale (the sibling script's fixed choice) vs. the best bg composition found, per fg
# scale.
def mean_std_iou(
    lookup: dict[tuple, float], combo_keys: set[tuple] | None = None
) -> tuple[float, float, int]:
    vals = [v for ck, v in lookup.items() if combo_keys is None or ck in combo_keys]
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


bg_ablation_rows = []
for fg_scale in SCALE_NAMES:
    for bg_name, bg_members in BG_COMPOSITION_COMBOS.items():
        mean, std, n = mean_std_iou(bg_iou_lookup[fg_scale][bg_name])
        bg_ablation_rows.append(
            {
                "fg_scale": fg_scale,
                "bg_composition": bg_name,
                "n_bg_scales": len(bg_members),
                "bg_members": "+".join(bg_members),
                "mean_iou": mean,
                "std_iou": std,
                "n_combos": n,
            }
        )
bg_ablation_df = pd.DataFrame(bg_ablation_rows)
bg_ablation_df.to_csv(OUTPUT_DIR / "bg_ablation_iou.csv", index=False)
log.info("Wrote %s (%d rows)", OUTPUT_DIR / "bg_ablation_iou.csv", len(bg_ablation_df))

fg_only_rows = [
    {"fg_scale": fg_scale, "mean_iou": mean_std_iou(fg_only_iou[fg_scale])[0]}
    for fg_scale in SCALE_NAMES
]
pd.DataFrame(fg_only_rows).to_csv(OUTPUT_DIR / "fg_only_iou_reference.csv", index=False)

STRATEGY_LABELS = ["own scale only", "global only", "every scale (all)", "best found"]
STRATEGY_COLORS = ["#e74c3c", "#3498db", "#7f8c8d", "#2ecc71"]

headline_rows = []
for fg_scale in SCALE_NAMES:
    own_iou = mean_std_iou(bg_iou_lookup[fg_scale][fg_scale])[0]
    global_iou = mean_std_iou(bg_iou_lookup[fg_scale]["global"])[0]
    all_iou = mean_std_iou(bg_iou_lookup[fg_scale][FULL_BG_NAME])[0]
    best_name = max(
        BG_COMPOSITION_COMBOS, key=lambda n: mean_std_iou(bg_iou_lookup[fg_scale][n])[0]
    )
    best_iou = mean_std_iou(bg_iou_lookup[fg_scale][best_name])[0]
    headline_rows.append(
        {
            "fg_scale": fg_scale,
            "own_scale_only": own_iou,
            "global_only": global_iou,
            "every_scale_all": all_iou,
            "best_found": best_iou,
            "best_bg_composition": best_name,
        }
    )
headline_df = pd.DataFrame(headline_rows)
headline_df.to_csv(OUTPUT_DIR / "bg_headline_comparison.csv", index=False)
log.info("Background-strategy headline comparison, per fg scale:")
for _, row in headline_df.iterrows():
    log.info(
        "  fg=%-8s own=%.3f global=%.3f all=%.3f best=%.3f (%s) | single_proto ref=%.3f",
        row.fg_scale,
        row.own_scale_only,
        row.global_only,
        row.every_scale_all,
        row.best_found,
        row.best_bg_composition,
        mean_std_iou(fg_only_iou[row.fg_scale])[0],
    )

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(SCALE_NAMES))
width = 0.2
for i, (strategy, color) in enumerate(
    zip(
        ["own_scale_only", "global_only", "every_scale_all", "best_found"],
        STRATEGY_COLORS,
    )
):
    ax.bar(
        x + (i - 1.5) * width, headline_df[strategy], width, label=STRATEGY_LABELS[i], color=color
    )
ax.scatter(
    x,
    [mean_std_iou(fg_only_iou[s])[0] for s in SCALE_NAMES],
    marker="_",
    s=400,
    color=SINGLE_PROTO_COLOR,
    label="single_proto (bg-invariant ref)",
    zorder=5,
)
ax.set_xticks(x, SCALE_NAMES)
ax.set_xlabel("foreground scale (fixed, single scale only)")
ax.set_ylabel("oracle IoU (mean across combos)")
ax.set_ylim(0, 1.0)
ax.set_title(
    f"Background-composition strategies per fixed fg scale, n={N_SCALE_STEPS} steps "
    f"({len(usable_combo_keys)} combos)"
)
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "bg_headline_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved %s", OUTPUT_DIR / "bg_headline_comparison.png")

# %% Part 7 — bg growth curves per fixed fg scale (mirrors scale_composition_oracle_iou.py's
# composition_growth.png, one file per fg scale since 7 fg scales x 3 panels in one figure would
# be unreadable).
for fg_scale in SCALE_NAMES:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), sharey=True, sharex=True)
    for ax, growth_names, direction in [
        (axes[0], SCALE_NAMES[:1] + BG_PREFIX_NAMES, "bg growing inward from global"),
        (axes[1], SCALE_NAMES[-1:] + BG_SUFFIX_NAMES, "bg growing outward from close"),
        (axes[2], BG_ANCHORED_INWARD_NAMES, "bg anchored at global+close, growing inward"),
    ]:
        xs = [len(BG_COMPOSITION_COMBOS[name]) for name in growth_names]
        means = [mean_std_iou(bg_iou_lookup[fg_scale][name])[0] for name in growth_names]
        stds = [mean_std_iou(bg_iou_lookup[fg_scale][name])[1] for name in growth_names]
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, color=KNN_COLOR)
        ax.axhline(
            mean_std_iou(fg_only_iou[fg_scale])[0],
            linestyle=":",
            color=SINGLE_PROTO_COLOR,
            alpha=0.7,
            label="single_proto (bg-invariant ref)",
        )
        ax.set_xticks(
            range(1, len(SCALE_NAMES) + 1), [str(n) for n in range(1, len(SCALE_NAMES) + 1)]
        )
        ax.set_xlabel("number of bg scale steps composed")
        ax.set_title(direction, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)

    ax2 = axes[2]
    xs_out = [len(BG_COMPOSITION_COMBOS[name]) for name in BG_ANCHORED_OUTWARD_NAMES]
    means_out = [
        mean_std_iou(bg_iou_lookup[fg_scale][name])[0] for name in BG_ANCHORED_OUTWARD_NAMES
    ]
    stds_out = [
        mean_std_iou(bg_iou_lookup[fg_scale][name])[1] for name in BG_ANCHORED_OUTWARD_NAMES
    ]
    ax2.errorbar(
        xs_out,
        means_out,
        yerr=stds_out,
        marker="s",
        linestyle="--",
        capsize=3,
        alpha=0.6,
        color=KNN_COLOR,
    )
    if "global+mid+close" in BG_COMPOSITION_COMBOS:
        classic_mean = mean_std_iou(bg_iou_lookup[fg_scale]["global+mid+close"])[0]
        ax2.scatter(
            [3], [classic_mean], marker="D", s=90, color=KNN_COLOR, zorder=5, edgecolors="black"
        )
    ax2.set_title(
        "bg anchored at global+close\n"
        "(o/solid=grow from global, s/dashed=grow from close, diamond=global+mid+close)",
        fontsize=8,
    )

    axes[0].set_ylabel("knn_fgbg oracle IoU (mean +/- std across combos)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Background-composition growth curves, fg fixed at '{fg_scale}'")
    fig.tight_layout()
    _safe_name = fg_scale.replace("/", "-")
    fig.savefig(OUTPUT_DIR / f"bg_growth__fg_{_safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
log.info("Saved %d per-fg-scale bg-growth-curve figures", len(SCALE_NAMES))

# %% [markdown]
# ## Reading the results
#
# - **`bg_headline_comparison.png`/`.csv`** is the main answer: for each fixed fg scale, four bg
#   strategies side by side — bg drawn only from that same crop ("own scale only", the no-multi-
#   scale-context baseline), bg drawn only from the full uncropped image ("global only", pure
#   far-field), bg pooled from every scale step ("every scale (all)", what
#   `scale_composition_oracle_iou.py` used throughout), and the best of all 28
#   `BG_COMPOSITION_COMBOS` for that fg scale. If "every scale (all)" sits close to "best found"
#   everywhere, the sibling script's fixed bg choice was already close to optimal and this whole
#   ablation is a null result in the useful sense — multi-scale bg helps, just not much beyond
#   "just use all of it". If "own scale only" is competitive with "every scale (all)", background
#   scale diversity isn't actually doing much work and the real driver of `knn_fgbg`'s advantage
#   over `single_proto` (see `scale_composition_oracle_iou.py`'s findings) is the per-patch
#   gallery mechanism itself, not the multi-scale bg pooling specifically.
# - **`bg_growth__fg_<scale>.png`** (one per fg scale) is the same 3-panel growth-curve design as
#   the sibling script's `composition_growth.png`, but with bg on the x-axis instead of fg, fg
#   held fixed, and a dotted reference line for `single_proto`'s bg-invariant IoU at that same fg
#   scale (a floor `knn_fgbg` should always clear, if a rich enough bg gallery is doing its job).
# - **This script never grows fg** — every row uses exactly one fg scale. Combining both
#   dimensions (composed fg x composed bg) is a 28x28 matrix per combo and was deliberately left
#   out here to keep each script answering one question; see `scale_composition_oracle_iou.py`
#   for the fg-composition-only sweep (bg fixed at "every scale") this one complements.

# %%
