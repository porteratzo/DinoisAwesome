# %% [markdown]
# # Fundamental: Augmented kNN Fg Prototypes — Does Perturbing Foreground
# # Gallery Patches Improve Oracle-IoU Localization?
#
# `augmented_prototype_oracle_iou.py` asks whether averaging one augmented view into a
# single masked-mean exemplar prototype helps localization. This file asks the same
# question for the "fg-bg-knn" scoring method from `multiscale_crop_ablation.py`
# (`_shared.prototype_ops.knn_fgbg_score`): instead of collapsing the exemplar to one mean
# vector, keep every masked ("foreground") patch as its own gallery entry and every
# unmasked ("background") patch as its own gallery entry, then score a query patch as
# (mean top-k cosine similarity to the fg gallery) minus (mean top-k similarity to the bg
# gallery).
#
# Because fg and bg are two independent galleries instead of one vector, a perturbed view's
# patches could in principle be folded into the fg gallery, the bg gallery, or both. This
# file originally tested all three: **fg-aug** (broaden what counts as "the object", bg
# stays the clean reference's own surroundings), **bg-aug** (broaden what counts as "not the
# object", fg stays the clean reference's own masked patches), and **both-aug** (both sides
# see the same perturbed view at once). **`bg-aug` and `both-aug` were dropped** — across
# every family and scale they scored completely worse than both `single_proto` and `fg-aug`,
# so augmenting the bg gallery never helped and only dragged `both-aug` down with it. Only
# the fg-side hypothesis remains below, now compared two ways:
#
#   - **`knn_fg`** — the fg gallery keeps every raw patch token (augmented + clean) as its
#     own entry; bg stays the raw, clean, multiscale patch gallery. Score = mean top-k fg
#     similarity minus mean top-k bg similarity.
#   - **`proto_fgbg`** — both sides are collapsed to a single mean vector instead of a raw
#     gallery: the fg side is the *same* augmented masked-mean prototype `single_proto` uses,
#     the bg side is one clean masked-mean prototype over all its patches. Score is the same
#     fg-similarity-minus-bg-similarity formula, just with k=1 prototypes standing in for
#     `knn_fg`'s full galleries — a cheap way to ask whether the fg-bg *contrast* itself (not
#     the raw-gallery kNN machinery) is what's buying `knn_fg` its improvement over
#     `single_proto`.
#
# The **single masked-mean prototype method is recomputed here too, unchanged, as the
# baseline for comparison** — every aggregate chart below plots `single_proto` alongside
# `knn_fg` and `proto_fgbg` so "does going contrastive beat one mean vector, and does the
# contrast need a full kNN gallery or just a second mean vector" can be read off directly,
# not inferred by cross-referencing separate scripts/runs.
#
# Parity with `augmented_prototype_oracle_iou.py`'s crop-augmentation experiment layers, now
# run for all three methods (`single_proto`, `knn_fg`, `proto_fgbg`). (The original file's
# fourth layer — augmenting the *query* image instead of the exemplar — was tried here too
# and dropped: it isn't a practical scenario for this pipeline and it never surfaced a
# result worth acting on, so it added run time and output files without payoff.)
#
#   1. **Single-severity sweep** — one augmentation family/severity's own patches folded
#      into the clean fg gallery/prototype (or averaged into the clean prototype), oracle
#      IoU vs. severity, per scale.
#   2. **Composed ensemble** — per family, cumulative k=1..N pooling of every severity's
#      own patches (or mean vectors) into one fg gallery/prototype.
#   3. **All-augmentations composed, leave-one-out** — cumulative pooling across every
#      *included* family at once, swept once with nothing held out and once per family
#      held out, to see which family the grand ensemble is better or worse without.
#
# One deliberate deviation from the original file's structure: everything here is built
# **combo-major** (one reference instance at a time, all its scales/families/severities
# encoded, scored, composed, and leave-one-out'd immediately, then discarded) rather than
# accumulating every crop's raw patch tokens in one global list first. A masked-mean
# prototype is a single (1, C) vector per crop — cheap to keep for thousands of crops at
# once, which is what the original file does. A kNN gallery needs the *raw* patch tokens
# themselves (hundreds per crop), and with every (part_type, group, instance, scale,
# family, severity) combination in play that would be tens of GB of host RAM held live
# simultaneously. Processing one reference instance's crops at a time keeps peak memory to
# one instance's own patch tokens, discarded before the next.
#
# `bg-aug`/`both-aug` (above) were about *augmenting* the bg gallery and made things worse.
# Separately, the *clean* baseline bg gallery both remaining methods still read every bg
# score from (`clean_bg_bank_lookup`/`clean_bg_proto_lookup`, Part 3.5) was itself weak: it
# only ever pooled patches from the padded fringe immediately around a combo's own
# close/mid crop, never anything genuinely far-field. `multiscale_crop_ablation.py`'s
# fg-bg-knn methods, by contrast, always include a "global" (whole uncropped reference
# image) source on the bg side — see its `CropConfig.scales` default and
# `FGBG_SOURCE_COMBOS`'s `"bg": ["global", "mid", "close"]` — precisely so the bg gallery
# gets far-field context, not just a narrow local neighbourhood. Part 3.6/3.5 below now
# build and fold in that same "global" bg source per combo (excluding every ref instance
# in the combo's own group, not just itself, from what counts as background).
#
# Severity index 0 is a literal no-op for every family (angle=0, gamma=1.0, ...), so it's
# pixel-identical to the plain, unaugmented crop. For the single-severity sweep this is
# special-cased to score the *plain* clean galleries directly rather than concatenating the
# severity-0 view onto itself — duplicating a patch set changes a kNN top-k mean (the
# duplicated points can crowd out other neighbours) even though it's a no-op for a mean
# vector, so without the special case the `knn_fg` and `proto_fgbg` curves wouldn't all start
# from the same "no augmentation" value as `single_proto` and each other. The composed and
# leave-one-out sections don't need this special case: they pool severities' own patches directly
# (severity 0's own patches already *are* the clean gallery), never pooling the clean
# gallery a second time on top.

# %% Logging — must be before torch import
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("augmented_prototype_oracle_iou_knn_fgbg")

from collections import defaultdict
from functools import partial
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
from _shared.augmentations import (  # noqa: E402
    apply_blur,
    apply_color_jitter,
    apply_gamma,
    apply_jpeg,
    apply_noise,
    apply_rotation,
    mean_color,
    pixel_only,
)
from _shared.mask_geometry import pixel_mask_to_patch_mask, scale_crop_box  # noqa: E402
from _shared.prototype_ops import knn_score_heatmap, score_heatmap  # noqa: E402
from _shared.thresholding import oracle_iou  # noqa: E402

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

data_dir = _REPO_ROOT / "data" / "abc3"

REF_NUMBER = 1
QUERY_NUMBER = 2

# Same knob as augmented_prototype_oracle_iou.py — narrow for fast iteration, e.g. ["LHa"].
RUN_PART_TYPES: list[str] = PART_TYPES

# Same focus combo as augmented_prototype_oracle_iou.py, for direct comparability of the
# qualitative figures.
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
CROP_PADDING_FRACTION = 1.0
MIN_CROP_SIZE = 128

ORACLE_THRESHOLD_STEPS = 25

SCALES: list[str] = ["close", "mid"]
SCALE_COLOR: dict[str, str] = {"close": "#e74c3c", "mid": "#f39c12"}

# k for the per-patch kNN gallery lookup — same default multiscale_crop_ablation.py uses
# for its "fg-bg-knn(...)" methods.
KNN_FGBG_NUM_NEIGHBOURS = 10

# The three methods compared on every aggregate chart below. "single_proto" is the
# unmodified augmented_prototype_oracle_iou.py method, recomputed here as the baseline.
# "knn_bg"/"knn_both" (bg-side augmentation) were dropped — see the file header — they
# scored completely worse than single_proto and knn_fg across every family/scale.
METHOD_LABELS: list[str] = ["single_proto", "knn_fg", "proto_fgbg"]
METHOD_TITLES: dict[str, str] = {
    "single_proto": "single masked-mean prototype (baseline)",
    "knn_fg": "fg-bg-knn, raw galleries, augment fg only",
    "proto_fgbg": "fg-bg prototype, mean vectors, augment fg only",
}
SEED = 0
torch.manual_seed(SEED)

OUTPUT_DIR = _REPO_ROOT / "outputs" / "fundamental" / "augmented_prototype_oracle_iou_knn_fgbg"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log.info(
    "RUN_PART_TYPES=%s ref_number=%d query_number=%d  |  DINO%s-%s img_size=%d layer=%d  |  "
    "knn_k=%d methods=%s",
    RUN_PART_TYPES,
    REF_NUMBER,
    QUERY_NUMBER,
    DINO_VERSION,
    DINO_SIZE,
    IMG_SIZE,
    LAYER_IDX,
    KNN_FGBG_NUM_NEIGHBOURS,
    METHOD_LABELS,
)

# %% Helpers shared across discovery / scoring / aggregation / plotting


def split_fg_bg_patches(
    patch_tokens: torch.Tensor,
    mask_px: np.ndarray,
    grid_h: int,
    grid_w: int,
    label: str,
    *,
    bg_exclude_mask_px: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-normalise a crop's patch tokens and split them into (fg, bg). fg is *mask_px*
    projected to the patch grid. bg is everything outside *bg_exclude_mask_px* (defaults to
    *mask_px* itself when omitted) projected the same way — pass the union of every
    same-group ref instance's mask within this crop's box as *bg_exclude_mask_px* so a
    neighbouring instance that falls inside the crop (multi-instance groups like "white
    clips") doesn't leak its own foreground patches into the bg gallery, mirroring
    multiscale_crop_ablation.py's union_mask_crop/exclude_patch_mask. Falls back to "every
    patch" (logged) for whichever side projects to nothing — mirrors masked_mean_prototype's
    fallback in the sibling script, applied independently to each side since a degenerate
    crop can lose either one."""
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


def augmented_fg_gallery(clean_fg: torch.Tensor, fg_extra: list[torch.Tensor]) -> torch.Tensor:
    """knn_fg's fg gallery: the clean baseline gallery with whatever augmented-patch tensors
    *fg_extra* wants folded in. Empty fg_extra returns the clean gallery unchanged (no
    torch.cat). bg-side augmentation was dropped (see the file header), so every remaining
    method's bg gallery/prototype stays the clean baseline everywhere it's used below."""
    return torch.cat([clean_fg, *fg_extra], dim=0) if fg_extra else clean_fg


def score_and_store(
    lookup: dict[str, dict[tuple, float]],
    method: str,
    key: tuple,
    query_tokens: torch.Tensor,
    h: int,
    w: int,
    gt: np.ndarray,
    *,
    proto: torch.Tensor | None = None,
    fg_bank: torch.Tensor | None = None,
    bg_bank: torch.Tensor | None = None,
) -> tuple[np.ndarray, float]:
    """Score one (method, key) heatmap, compute its oracle IoU, store it in lookup[method],
    and return (raw, iou) so callers needing the heatmap for qualitative figures don't have
    to rescore it."""
    if method == "single_proto":
        assert proto is not None
        raw = score_heatmap(query_tokens, proto, h, w)
    else:
        assert fg_bank is not None and bg_bank is not None
        raw = knn_score_heatmap(query_tokens, fg_bank, bg_bank, KNN_FGBG_NUM_NEIGHBOURS, h, w)
    iou = oracle_iou(raw, gt, ORACLE_THRESHOLD_STEPS)
    lookup[method][key] = iou
    return raw, iou


colors = plt.get_cmap("tab10").colors

# %% Part 1 — discover every (part_type, instance-type group, ref instance) combo
# Identical to augmented_prototype_oracle_iou.py's Part 1.
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
            # Union of every ref instance in this group — excluded wholesale from the
            # "global" bg gallery below (Part 3.6/3.5) so a neighbouring same-type
            # instance elsewhere in the ref image never leaks into a combo's own bg
            # patches, mirroring multiscale_ablation's exclude_patch_mask.
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

# %% Part 2 — build close/mid crops per combo
# Identical to augmented_prototype_oracle_iou.py's Part 2, plus scales_by_ck (which scales
# survived MIN_CROP_SIZE for a given combo) — needed below to know which scales' clean bg
# patches belong in that combo's pooled bg gallery. Also stores each crop's
# bg_exclude_mask_px — the combo's own group_ref_masks union (every ref instance in the
# group, not just this one) sliced to the crop's box — so Part 3.5's bg split can exclude a
# neighbouring same-group instance that falls inside this crop (e.g. "white clips", which can
# have several instances close together), mirroring multiscale_crop_ablation.py's
# union_mask_crop/exclude_patch_mask. Without this, a neighbour's own foreground patches leak
# into the bg gallery as false "background", unlike the global bg source (Part 3.6/3.5), which
# already excludes the whole group correctly.
combo_keys_by_scale: dict[str, list[tuple]] = defaultdict(list)
scales_by_ck: dict[tuple, list[str]] = defaultdict(list)
for combo in tqdm(combos, desc="Building close/mid crops"):
    ref_img = ref_images[combo["part_type"]]
    group_mask = group_ref_masks.get((combo["part_type"], combo["group"]), combo["ref_mask"])
    combo["crops"] = {}
    for scale in SCALES:
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
        crop_img = ref_img.crop(box)
        combo["crops"][scale] = {
            "img": crop_img,
            "mask_px": combo["ref_mask"][y0:y1, x0:x1],
            "bg_exclude_mask_px": group_mask[y0:y1, x0:x1],
            "fill": mean_color(crop_img),
        }
        combo_keys_by_scale[scale].append(combo_key(combo))
        scales_by_ck[combo_key(combo)].append(scale)

for scale in SCALES:
    log.info("scale=%-5s usable combos=%d", scale, len(combo_keys_by_scale[scale]))

# %% Part 3 — encode each part type's query image once; per-(part_type, group) GT patch mask
# Identical to augmented_prototype_oracle_iou.py's Part 3.
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

query_encodings: dict[str, dict] = {}
for part_type in tqdm(sorted(query_images), desc="Encoding query images"):
    q_tokens, q_h, q_w = extract_patch_tokens(
        encoder, query_images[part_type], LAYER_IDX, debias=True
    )
    # Kept on the encoder's own device (GPU, when available), not moved to CPU: there are
    # only as many entries here as part types (a handful), so the memory-safety concern
    # that drives the combo-major restructuring elsewhere in this file doesn't apply — and
    # every knn_fgbg_score call below multiplies this tensor against a gallery, so keeping
    # it on GPU is what actually makes those matmuls fast instead of CPU-bound.
    query_encodings[part_type] = {"q_tokens": q_tokens, "q_h": q_h, "q_w": q_w}

gt_patch_masks: dict[tuple[str, str], np.ndarray] = {}
for (part_type, group), pixel_mask in group_query_masks.items():
    q = query_encodings[part_type]
    gt_patch_masks[(part_type, group)] = pixel_mask_to_patch_mask(
        pixel_mask, q["q_h"], q["q_w"], IMG_SIZE, MASK_PATCH_THRESHOLD
    )

# %% Part 3.6 — encode each part type's *full, uncropped* reference image once: the
# "global" scale background source. Every close/mid bg gallery below is drawn only from
# the padded fringe immediately around a combo's own crop (CROP_PADDING_FRACTION=1.0) —
# at scale=close that fringe is a thin border sharing the object's own surface/lighting,
# a weak negative set for the knn_fg/proto_fgbg contrastive methods even though it never
# hurts single_proto (which has no bg side to be starved). multiscale_ablation avoids
# this by always spanning CropConfig.scales=("global","mid","close") on the bg side (see
# its engine.build_all_scale_prototypes/methods.py FGBG_SOURCE_COMBOS) — "global" there is
# the whole uncropped reference image, giving the bg gallery far-field context instead of
# just the local crop border. This mirrors that: one full-image encode per part type,
# reused as every combo's extra bg source below.
ref_encodings: dict[str, dict] = {}
for part_type in tqdm(sorted(ref_images), desc="Encoding ref images (global bg scale)"):
    r_tokens, r_h, r_w = extract_patch_tokens(
        encoder, ref_images[part_type], LAYER_IDX, debias=True
    )
    ref_encodings[part_type] = {"r_tokens": r_tokens, "r_h": r_h, "r_w": r_w}

# %% Part 3.5 — encode every combo's *clean* (unaugmented) crops once, up front, to build
# the baseline prototype (single_proto), fg gallery (knn_fg), and fg/bg prototypes
# (proto_fgbg) every later section folds augmented patches into. Batched across every
# (combo, scale) at once since there are only as many of these as (combo, scale) pairs
# (tens, not thousands) — kept on the encoder's own device (GPU, when available) rather than
# moved to CPU, so every downstream knn_fgbg_score matmul against them runs on GPU. Worst
# case (every crop's bg gallery at full 48x48-ish patch-grid size) is under ~1 GB for a few
# dozen (combo, scale) pairs at ViT-L's 1024-dim features — negligible next to the backbone
# weights already resident.
clean_prototype_lookup: dict[tuple, torch.Tensor] = {}
clean_fg_bank_lookup: dict[tuple, torch.Tensor] = {}
clean_bg_by_scale_lookup: dict[tuple, torch.Tensor] = {}
# Per-combo "global" bg source (Part 3.6's full ref-image tokens, minus this combo's
# group), folded into clean_bg_bank_lookup below alongside the per-scale bg patches.
clean_bg_global_lookup: dict[tuple, torch.Tensor] = {}
clean_bg_bank_lookup: dict[tuple, torch.Tensor] = {}
# proto_fgbg's bg side: the clean, multiscale bg gallery collapsed to one mean vector — never
# augmented (bg-side augmentation was dropped, see the file header).
clean_bg_proto_lookup: dict[tuple, torch.Tensor] = {}

clean_items: list[tuple] = []
for combo in combos:
    ck = combo_key(combo)
    for scale, crop in combo["crops"].items():
        clean_items.append((ck, scale, crop["img"], crop["mask_px"], crop["bg_exclude_mask_px"]))

for i in tqdm(range(0, len(clean_items), chunk_size), desc="Encoding clean baseline crops"):
    chunk = clean_items[i : i + chunk_size]
    out = encoder([c[2] for c in chunk], layers=[LAYER_IDX], debias=True)
    chunk_patches = out.patches[:, 0]
    grid_h, grid_w = chunk_patches.shape[1], chunk_patches.shape[2]
    for (ck, scale, _, mask_px, bg_exclude_mask_px), patch_tokens in tqdm(
        zip(chunk, chunk_patches), desc="Building clean baseline galleries", total=len(chunk)
    ):
        fg, bg = split_fg_bg_patches(
            patch_tokens,
            mask_px,
            grid_h,
            grid_w,
            f"{ck} scale={scale} clean",
            bg_exclude_mask_px=bg_exclude_mask_px,
        )
        clean_fg_bank_lookup[(ck, scale)] = fg
        clean_bg_by_scale_lookup[(ck, scale)] = bg
        clean_prototype_lookup[(ck, scale)] = compute_exemplar_features(fg, mode="mean")

for combo in combos:
    ck = combo_key(combo)
    part_type, group = ck[0], ck[1]
    r = ref_encodings[part_type]
    # Exclude every ref instance in this combo's group (not just this one) from the
    # global bg gallery — a neighbouring same-type instance elsewhere in the ref image
    # must not leak into "background", mirroring multiscale_ablation's exclude_patch_mask.
    exclude_mask_px = group_ref_masks.get((part_type, group), combo["ref_mask"])
    exclude_patch_mask = pixel_mask_to_patch_mask(
        exclude_mask_px, r["r_h"], r["r_w"], IMG_SIZE, MASK_PATCH_THRESHOLD
    )
    exclude_flat = torch.from_numpy(exclude_patch_mask.reshape(-1)).to(r["r_tokens"].device)
    global_bg = r["r_tokens"][~exclude_flat]
    if global_bg.shape[0] == 0:
        log.warning("%s: global bg mask empty after patch-grid projection — using all patches", ck)
        global_bg = r["r_tokens"]
    clean_bg_global_lookup[ck] = global_bg

for ck, scales in scales_by_ck.items():
    clean_bg_bank_lookup[ck] = torch.cat(
        [clean_bg_by_scale_lookup[(ck, s)] for s in scales] + [clean_bg_global_lookup[ck]], dim=0
    )
    clean_bg_proto_lookup[ck] = compute_exemplar_features(clean_bg_bank_lookup[ck], mode="mean")

log.info(
    "Built clean single_proto/fg-bank/bg-bank/bg-proto baselines for %d (combo, scale) pairs "
    "(bg-bank now includes each combo's global/far-field patches)",
    len(clean_items),
)

# %% Augmentation families — same families/values as augmented_prototype_oracle_iou.py, for
# direct comparability.
AUGMENTATIONS: dict[str, dict] = {
    "rotation": {"values": [0, 8, 16, 30, 50, 75], "unit": "deg", "apply": apply_rotation},
    "illumination (gamma)": {
        "values": [1.0, 1.3, 1.7, 2.2, 2.8, 3.5],
        "unit": "gamma",
        "apply": pixel_only(apply_gamma),
    },
    "color jitter": {
        "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "unit": "magnitude",
        "apply": pixel_only(partial(apply_color_jitter, seed=SEED)),
    },
    "gaussian blur": {
        "values": [0, 1, 2, 4, 7, 11],
        "unit": "px radius",
        "apply": pixel_only(apply_blur),
    },
    "gaussian noise": {
        "values": [0, 8, 16, 28, 45, 70],
        "unit": "sigma (0-255)",
        "apply": pixel_only(partial(apply_noise, seed=SEED)),
    },
    "jpeg compression": {
        "values": [100, 80, 60, 35, 15, 3],
        "unit": "quality",
        "apply": pixel_only(apply_jpeg),
    },
}
first_family = next(iter(AUGMENTATIONS))
first_val = AUGMENTATIONS[first_family]["values"][0]
N_SEVERITY_LEVELS = len(AUGMENTATIONS[first_family]["values"])
HELD_OUT_LABELS: list[str] = ["none", *AUGMENTATIONS]

# %% Focus combo — used for every qualitative (crop-grid / heatmap) figure below. Same
# fallback logic as augmented_prototype_oracle_iou.py.
focus_combo = next(
    (
        c
        for c in combos
        if c["part_type"] == FOCUS_PART_TYPE
        and c["class"] == FOCUS_CLASS
        and c["instance_id"] == FOCUS_INSTANCE_ID
    ),
    None,
)
if focus_combo is None:
    focus_combo = combos[0]
    log.warning(
        "Focus combo part_type=%s class=%r instance_id=%d not found under RUN_PART_TYPES=%s "
        "— falling back to %s",
        FOCUS_PART_TYPE,
        FOCUS_CLASS,
        FOCUS_INSTANCE_ID,
        RUN_PART_TYPES,
        combo_key(focus_combo),
    )
focus_key = combo_key(focus_combo)
log.info("Focus combo for qualitative figures: %s", focus_key)

# %% Part 4/5 — main combo-major sweep: single-severity crop-augmentation, per-family
# composed ensemble, and cross-family leave-one-out ensemble, for all four methods at once.
# See the file header for why this is combo-major (one reference instance's crops encoded,
# scored, and discarded before the next) rather than one global flat list of every crop.
iou_lookup: dict[str, dict[tuple, float]] = {m: {} for m in METHOD_LABELS}
composed_iou_lookup: dict[str, dict[tuple, float]] = {m: {} for m in METHOD_LABELS}
all_aug_composed_iou_lookup: dict[str, dict[tuple, float]] = {m: {} for m in METHOD_LABELS}

# Qualitative stash for the focus combo only: crop-grid images (single_proto oracle_iou in
# the title, as in the sibling script) and, per method, the baseline vs. best-severity raw
# heatmap for the baseline-vs-best figure below.
focus_crop_grid: dict[str, list[dict]] = {}  # scale -> list of {family, value, img, oracle_iou}
focus_heatmaps: dict[str, dict[str, dict]] = {}  # scale -> method -> {"baseline":.., "best":..}

for combo in tqdm(combos, desc="Main per-combo sweep"):
    ck = combo_key(combo)
    part_type, group = ck[0], ck[1]
    q = query_encodings[part_type]
    q_tokens, q_h, q_w = q["q_tokens"], q["q_h"], q["q_w"]
    gt = gt_patch_masks[(part_type, group)]
    is_focus = ck == focus_key
    if is_focus:
        focus_crop_grid.clear()
        focus_heatmaps.clear()

    for scale, crop in combo["crops"].items():
        clean_proto = clean_prototype_lookup[(ck, scale)]
        clean_fg = clean_fg_bank_lookup[(ck, scale)]
        clean_bg_full = clean_bg_bank_lookup[ck]
        clean_bg_proto = clean_bg_proto_lookup[ck]

        # Build + encode every (family, severity) augmented crop for this (combo, scale).
        scale_entries: list[dict] = []
        for family, spec in AUGMENTATIONS.items():
            for val in spec["values"]:
                img, mask_px = spec["apply"](crop["img"], crop["mask_px"], val, crop["fill"])
                scale_entries.append(
                    {"family": family, "value": val, "img": img, "mask_px": mask_px}
                )

        # Kept on GPU (not .cpu()'d) for the duration of this (combo, scale) iteration only —
        # every entry here feeds hundreds of knn_fgbg_score matmuls below (single-severity
        # sweep + composed + leave-one-out), so this is the actual hot path; scale_entries
        # (and every entry's fg_patches/prototype) still goes out of scope at the end of this
        # iteration, so peak memory stays "one combo's crops", same as before. Only the fg
        # side of split_fg_bg_patches's output is kept — bg-side augmentation was dropped, so
        # every augmented crop's own bg patches are never folded into anything.
        for i in range(0, len(scale_entries), chunk_size):
            chunk = scale_entries[i : i + chunk_size]
            out = encoder([e["img"] for e in chunk], layers=[LAYER_IDX], debias=True)
            chunk_patches = out.patches[:, 0]
            g_h, g_w = chunk_patches.shape[1], chunk_patches.shape[2]
            for e, patch_tokens in zip(chunk, chunk_patches):
                label = f"{ck} scale={scale} family={e['family']} value={e['value']}"
                fg, _bg = split_fg_bg_patches(patch_tokens, e["mask_px"], g_h, g_w, label)
                e["fg_patches"] = fg
                e["prototype"] = compute_exemplar_features(fg, mode="mean")

        if is_focus:
            focus_crop_grid[scale] = []

        # -- Layer 1: single-severity sweep (baseline paired/pooled with this one severity) --
        for e in scale_entries:
            key = (ck, scale, e["family"], e["value"])
            is_baseline = e["value"] == AUGMENTATIONS[e["family"]]["values"][0]

            paired_proto = F.normalize(
                torch.cat([clean_proto, e["prototype"]], dim=0).mean(dim=0, keepdim=True),
                p=2,
                dim=-1,
            )
            raw_sp, iou_sp = score_and_store(
                iou_lookup, "single_proto", key, q_tokens, q_h, q_w, gt, proto=paired_proto
            )

            fg_extra = [] if is_baseline else [e["fg_patches"]]
            fg_bank = augmented_fg_gallery(clean_fg, fg_extra)
            raw_fg, iou_fg = score_and_store(
                iou_lookup,
                "knn_fg",
                key,
                q_tokens,
                q_h,
                q_w,
                gt,
                fg_bank=fg_bank,
                bg_bank=clean_bg_full,
            )
            # proto_fgbg's fg side reuses paired_proto — the same augmented masked-mean
            # prototype single_proto just scored — contrasted against the clean bg prototype.
            raw_pfb, iou_pfb = score_and_store(
                iou_lookup,
                "proto_fgbg",
                key,
                q_tokens,
                q_h,
                q_w,
                gt,
                fg_bank=paired_proto,
                bg_bank=clean_bg_proto,
            )

            raws_this_entry = {"single_proto": raw_sp, "knn_fg": raw_fg, "proto_fgbg": raw_pfb}
            ious_this_entry = {"single_proto": iou_sp, "knn_fg": iou_fg, "proto_fgbg": iou_pfb}

            if is_focus:
                focus_crop_grid[scale].append(
                    {
                        "family": e["family"],
                        "value": e["value"],
                        "img": e["img"],
                        "oracle_iou": iou_sp,
                    }
                )
                # Gated on the *global* baseline (first family's first value), not the
                # per-family is_baseline used above — every family has its own severity-0
                # no-op entry (all pixel-identical to the clean crop), and resetting the
                # stash on each of those would wipe out "best" accumulated from families
                # processed earlier in this scale's loop.
                is_global_baseline = e["family"] == first_family and e["value"] == first_val
                if is_global_baseline:
                    focus_heatmaps.setdefault(scale, {})
                    for m in METHOD_LABELS:
                        focus_heatmaps[scale][m] = {
                            "baseline": (raws_this_entry[m], ious_this_entry[m])
                        }
                else:
                    for m in METHOD_LABELS:
                        cur = focus_heatmaps[scale][m].get("best")
                        if cur is None or ious_this_entry[m] > cur[1]:
                            focus_heatmaps[scale][m]["best"] = (
                                raws_this_entry[m],
                                ious_this_entry[m],
                                e["family"],
                                e["value"],
                            )

        # -- Layer 2: per-family composed ensemble (k=1..N severities pooled) --
        entries_by_family = {
            family: [e for e in scale_entries if e["family"] == family] for family in AUGMENTATIONS
        }
        for family, fam_entries in entries_by_family.items():
            n = len(fam_entries)
            sp_pool: list[torch.Tensor] = []
            for k in range(1, n + 1):
                key = (ck, scale, family, k)
                sp_pool.append(fam_entries[k - 1]["prototype"])
                comp_proto = F.normalize(
                    torch.cat(sp_pool, dim=0).mean(dim=0, keepdim=True), p=2, dim=-1
                )
                score_and_store(
                    composed_iou_lookup,
                    "single_proto",
                    key,
                    q_tokens,
                    q_h,
                    q_w,
                    gt,
                    proto=comp_proto,
                )

                fg_pool = torch.cat([fam_entries[i]["fg_patches"] for i in range(k)], dim=0)

                score_and_store(
                    composed_iou_lookup,
                    "knn_fg",
                    key,
                    q_tokens,
                    q_h,
                    q_w,
                    gt,
                    fg_bank=fg_pool,
                    bg_bank=clean_bg_full,
                )
                # proto_fgbg reuses comp_proto (same cumulative fg pooling as single_proto)
                # against the clean bg prototype — bg side is never composed/augmented.
                score_and_store(
                    composed_iou_lookup,
                    "proto_fgbg",
                    key,
                    q_tokens,
                    q_h,
                    q_w,
                    gt,
                    fg_bank=comp_proto,
                    bg_bank=clean_bg_proto,
                )

        # -- Layer 3: all-augmentations composed, leave-one-out across families --
        for held_out in HELD_OUT_LABELS:
            included = (
                [f for f in AUGMENTATIONS if f != held_out]
                if held_out != "none"
                else list(AUGMENTATIONS)
            )
            sp_pool = [clean_proto]
            fg_pool_list = [clean_fg]
            for k in range(1, N_SEVERITY_LEVELS + 1):
                if k > 1:
                    level = k - 1
                    for family in included:
                        fam_entries = entries_by_family[family]
                        if len(fam_entries) > level:
                            e = fam_entries[level]
                            sp_pool.append(e["prototype"])
                            fg_pool_list.append(e["fg_patches"])
                key = (ck, scale, held_out, k)
                comp_proto = F.normalize(
                    torch.cat(sp_pool, dim=0).mean(dim=0, keepdim=True), p=2, dim=-1
                )
                score_and_store(
                    all_aug_composed_iou_lookup,
                    "single_proto",
                    key,
                    q_tokens,
                    q_h,
                    q_w,
                    gt,
                    proto=comp_proto,
                )
                fg_pool = torch.cat(fg_pool_list, dim=0)
                score_and_store(
                    all_aug_composed_iou_lookup,
                    "knn_fg",
                    key,
                    q_tokens,
                    q_h,
                    q_w,
                    gt,
                    fg_bank=fg_pool,
                    bg_bank=clean_bg_full,
                )
                # proto_fgbg reuses comp_proto against the clean bg prototype — bg side is
                # never composed/augmented (dropped, see the file header).
                score_and_store(
                    all_aug_composed_iou_lookup,
                    "proto_fgbg",
                    key,
                    q_tokens,
                    q_h,
                    q_w,
                    gt,
                    fg_bank=comp_proto,
                    bg_bank=clean_bg_proto,
                )
        # scale_entries (and every entry's fg_patches/prototype) goes out of scope here —
        # discarded before the next scale/combo, per the file header's memory rationale.

log.info("Main per-combo sweep complete: %d combos x %d scales", len(combos), len(SCALES))

# %% Generic aggregation + plotting helpers — reused for every method x every layer below.


def aggregate_curves(lookup: dict[tuple, float]) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    curves: dict[str, dict[str, dict[str, np.ndarray]]] = {scale: {} for scale in SCALES}
    for scale in SCALES:
        cks = combo_keys_by_scale[scale]
        for family, spec in AUGMENTATIONS.items():
            means, stds = [], []
            for v in spec["values"]:
                vals = [
                    lookup[(ck, scale, family, v)] for ck in cks if (ck, scale, family, v) in lookup
                ]
                means.append(float(np.mean(vals)) if vals else float("nan"))
                stds.append(float(np.std(vals)) if vals else float("nan"))
            curves[scale][family] = {"mean": np.array(means), "std": np.array(stds)}
    return curves


def aggregate_baseline(lookup: dict[tuple, float]) -> dict[str, float]:
    baseline: dict[str, float] = {}
    for scale in SCALES:
        cks = combo_keys_by_scale[scale]
        vals = [
            lookup[(ck, scale, first_family, first_val)]
            for ck in cks
            if (ck, scale, first_family, first_val) in lookup
        ]
        baseline[scale] = float(np.mean(vals)) if vals else float("nan")
    return baseline


def aggregate_best_vs_baseline(lookup: dict[tuple, float]) -> list[dict]:
    rows: list[dict] = []
    for scale in SCALES:
        cks = combo_keys_by_scale[scale]
        for family, spec in AUGMENTATIONS.items():
            best_list, delta_list = [], []
            for ck in cks:
                vals = {
                    v: lookup[(ck, scale, family, v)]
                    for v in spec["values"]
                    if (ck, scale, family, v) in lookup
                }
                if not vals:
                    continue
                best_iou = max(vals.values())
                best_list.append(best_iou)
                baseline_iou = lookup.get((ck, scale, first_family, first_val))
                if baseline_iou is not None:
                    delta_list.append(best_iou - baseline_iou)
            rows.append(
                {
                    "scale": scale,
                    "family": family,
                    "mean_best_iou": float(np.mean(best_list)) if best_list else float("nan"),
                    "std_best_iou": float(np.std(best_list)) if best_list else float("nan"),
                    "mean_delta_vs_baseline": (
                        float(np.mean(delta_list)) if delta_list else float("nan")
                    ),
                    "n_combos": len(best_list),
                }
            )
    return rows


def aggregate_k_endpoint(
    lookup_by_k: dict[tuple, float], k_endpoint: int, baseline_lookup: dict[tuple, float]
) -> list[dict]:
    """Same row shape as aggregate_best_vs_baseline, but sourced from a composed/leave-one-out
    lookup's k=k_endpoint slice instead of a best-over-severities max — feeds the composed-
    ensemble log lines below with the same (scale, family, mean/std/delta) row shape."""
    rows: list[dict] = []
    for scale in SCALES:
        cks = combo_keys_by_scale[scale]
        for family in AUGMENTATIONS:
            vals = [
                lookup_by_k[(ck, scale, family, k_endpoint)]
                for ck in cks
                if (ck, scale, family, k_endpoint) in lookup_by_k
            ]
            base_vals = [
                baseline_lookup[(ck, scale, first_family, first_val)]
                for ck in cks
                if (ck, scale, first_family, first_val) in baseline_lookup
            ]
            rows.append(
                {
                    "scale": scale,
                    "family": family,
                    "mean_best_iou": float(np.mean(vals)) if vals else float("nan"),
                    "std_best_iou": float(np.std(vals)) if vals else float("nan"),
                    "mean_delta_vs_baseline": (
                        float(np.mean(vals)) - float(np.mean(base_vals))
                        if vals and base_vals
                        else float("nan")
                    ),
                    "n_combos": len(vals),
                }
            )
    return rows


def plot_severity_curves(
    curves: dict, baseline: dict[str, float], title: str, out_path: Path, ylabel: str
) -> None:
    fig, axes = plt.subplots(1, len(SCALES), figsize=(7.5 * len(SCALES), 5.5), sharey=True)
    for ax, scale in zip(axes, SCALES):
        n = len(combo_keys_by_scale[scale])
        for i, (family, spec) in enumerate(AUGMENTATIONS.items()):
            values = np.array(spec["values"], dtype=float)
            frac = (values - values[0]) / (values[-1] - values[0])
            ax.plot(
                frac,
                curves[scale][family]["mean"],
                marker="o",
                label=family,
                color=colors[i % len(colors)],
            )
        ax.axhline(
            baseline[scale],
            color="k",
            linestyle="--",
            linewidth=1,
            label="no-augmentation baseline",
        )
        ax.set_xlabel("severity fraction (0 = no-op, 1 = strongest tested)")
        ax.set_title(f"scale = {scale}  (n={n} combos)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_composed_curves(
    lookup_by_k: dict[tuple, float],
    baseline: dict[str, float],
    max_k: int,
    title: str,
    out_path: Path,
    ylabel: str,
) -> None:
    fig, axes = plt.subplots(1, len(SCALES), figsize=(7.5 * len(SCALES), 5.5), sharey=True)
    for ax, scale in zip(axes, SCALES):
        n = len(combo_keys_by_scale[scale])
        for i, family in enumerate(AUGMENTATIONS):
            means = []
            for k in range(1, max_k + 1):
                vals = [
                    lookup_by_k[(ck, scale, family, k)]
                    for ck in combo_keys_by_scale[scale]
                    if (ck, scale, family, k) in lookup_by_k
                ]
                means.append(float(np.mean(vals)) if vals else float("nan"))
            ax.plot(
                range(1, max_k + 1), means, marker="o", label=family, color=colors[i % len(colors)]
            )
        ax.axhline(
            baseline[scale],
            color="k",
            linestyle="--",
            linewidth=1,
            label="no-augmentation baseline",
        )
        ax.set_xlabel("severities composed (cumulative, k=1 is the original alone)")
        ax.set_title(f"scale = {scale}  (n={n} combos)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_leaveoneout_curves(
    lookup: dict[tuple, float], title: str, out_path: Path, ylabel: str
) -> None:
    fig, axes = plt.subplots(1, len(SCALES), figsize=(7.5 * len(SCALES), 5.5), sharey=True)
    ks = range(1, N_SEVERITY_LEVELS + 1)
    for ax, scale in zip(axes, SCALES):
        n = len(combo_keys_by_scale[scale])
        none_means = [
            float(
                np.mean(
                    [
                        lookup[(ck, scale, "none", k)]
                        for ck in combo_keys_by_scale[scale]
                        if (ck, scale, "none", k) in lookup
                    ]
                )
            )
            for k in ks
        ]
        ax.plot(ks, none_means, marker="o", color="black", linewidth=2, label="none held out")
        for i, family in enumerate(AUGMENTATIONS):
            means = [
                float(
                    np.mean(
                        [
                            lookup[(ck, scale, family, k)]
                            for ck in combo_keys_by_scale[scale]
                            if (ck, scale, family, k) in lookup
                        ]
                    )
                )
                for k in ks
            ]
            ax.plot(
                ks,
                means,
                marker="o",
                markersize=4,
                linestyle="--",
                label=f"hold out {family}",
                color=colors[i % len(colors)],
            )
        ax.set_xlabel("severity levels folded in (k=1 is the baseline alone)")
        ax.set_title(f"scale = {scale}  (n={n} combos)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_leaveoneout_bar(
    lookup: dict[tuple, float], baseline: dict[str, float], title: str, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(HELD_OUT_LABELS))
    width = 0.8 / len(SCALES)
    for i, scale in enumerate(SCALES):
        means = [
            float(
                np.mean(
                    [
                        lookup[(ck, scale, h, N_SEVERITY_LEVELS)]
                        for ck in combo_keys_by_scale[scale]
                        if (ck, scale, h, N_SEVERITY_LEVELS) in lookup
                    ]
                )
            )
            for h in HELD_OUT_LABELS
        ]
        ax.bar(x + i * width, means, width=width, label=f"{scale}", color=SCALE_COLOR[scale])
        ax.axhline(
            baseline[scale], color=SCALE_COLOR[scale], linestyle="--", linewidth=1, alpha=0.7
        )
    ax.set_xticks(
        x + width * (len(SCALES) - 1) / 2,
        ["none" if h == "none" else f"w/o {h}" for h in HELD_OUT_LABELS],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("oracle IoU (mean across combos)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# %% Layer 1 figures — single-severity sweep, per method + cross-method comparison
baseline_oracle_iou: dict[str, dict[str, float]] = {}
best_over_baseline: dict[str, list[dict]] = {}
for method in METHOD_LABELS:
    curves = aggregate_curves(iou_lookup[method])
    baseline_oracle_iou[method] = aggregate_baseline(iou_lookup[method])
    best_over_baseline[method] = aggregate_best_vs_baseline(iou_lookup[method])
    log.info(
        "[%s] baseline oracle IoU: %s",
        method,
        {k: round(v, 3) for k, v in baseline_oracle_iou[method].items()},
    )
    for row in sorted(best_over_baseline[method], key=lambda r: -r["mean_delta_vs_baseline"]):
        log.info(
            "[%s] BEST scale=%-5s %-22s mean_best_iou=%.3f±%.3f (%+.3f vs. baseline, n=%d combos)",
            method,
            row["scale"],
            row["family"],
            row["mean_best_iou"],
            row["std_best_iou"],
            row["mean_delta_vs_baseline"],
            row["n_combos"],
        )
    plot_severity_curves(
        curves,
        baseline_oracle_iou[method],
        f"[{METHOD_TITLES[method]}] Oracle-IoU vs. severity — {len(combos)} combos",
        OUTPUT_DIR / f"oracle_iou_curves__{method}.png",
        "oracle IoU (mean across combos)",
    )

baseline_rows = [
    {"method": method, "scale": scale, "baseline_oracle_iou": val}
    for method, by_scale in baseline_oracle_iou.items()
    for scale, val in by_scale.items()
]
pd.DataFrame(baseline_rows).to_csv(OUTPUT_DIR / "baseline_oracle_iou.csv", index=False)

best_over_baseline_df = pd.DataFrame(
    [{"method": method, **row} for method, rows in best_over_baseline.items() for row in rows]
)
best_over_baseline_df.to_csv(OUTPUT_DIR / "best_over_baseline.csv", index=False)
log.info(
    "Wrote %s and %s",
    OUTPUT_DIR / "baseline_oracle_iou.csv",
    OUTPUT_DIR / "best_over_baseline.csv",
)

log.info("Saved single-severity-sweep curve figures to %s", OUTPUT_DIR)

# %% Layer 1 — qualitative figures for the focus combo (crop grid + baseline-vs-best heatmap)
for scale in focus_crop_grid:
    grid = focus_crop_grid[scale]
    n_families = len(AUGMENTATIONS)
    n_cols = max(len(spec["values"]) for spec in AUGMENTATIONS.values())
    fig, axes = plt.subplots(n_families, n_cols, figsize=(2.6 * n_cols, 2.9 * n_families))
    for row, (family, spec) in enumerate(AUGMENTATIONS.items()):
        for col, val in enumerate(spec["values"]):
            ax = axes[row, col]
            entry = next(e for e in grid if e["family"] == family and e["value"] == val)
            ax.imshow(entry["img"])
            ax.set_title(
                f"{val} {spec['unit']}\nsingle_proto oracle_iou={entry['oracle_iou']:.3f}",
                fontsize=8,
            )
            ax.axis("off")
        for col in range(len(spec["values"]), n_cols):
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(family, fontsize=9)
    fig.suptitle(f"Augmented '{scale}' prototype crops — focus combo {focus_key}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUTPUT_DIR / f"augmented_crops_{scale}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

query_pixel_mask = group_query_masks[(focus_key[0], focus_key[1])]
focus_query_img = query_images[focus_key[0]]
query_gt_disp = (
    np.array(
        Image.fromarray(query_pixel_mask.astype(np.uint8) * 255).resize(
            focus_query_img.size, Image.NEAREST
        )
    )
    > 0
)
for scale in focus_heatmaps:
    gt_patch_mask = gt_patch_masks[(focus_key[0], focus_key[1])]
    fig, axes = plt.subplots(len(METHOD_LABELS), 3, figsize=(15, 5 * len(METHOD_LABELS)))
    for row, method in enumerate(METHOD_LABELS):
        baseline_raw, baseline_iou = focus_heatmaps[scale][method]["baseline"]
        best_raw, best_iou, best_family, best_value = focus_heatmaps[scale][method]["best"]
        delta = best_iou - baseline_iou

        axes[row, 0].imshow(focus_query_img)
        axes[row, 0].imshow(query_gt_disp, cmap="Reds", alpha=0.35)
        axes[row, 0].set_title(f"[{method}] query image + GT mask", fontsize=10)
        axes[row, 0].axis("off")

        im1 = axes[row, 1].imshow(baseline_raw, cmap="jet", aspect="auto")
        axes[row, 1].contour(
            gt_patch_mask.astype(float), levels=[0.5], colors="lime", linewidths=1.2
        )
        axes[row, 1].set_title(f"baseline (no aug) — oracle_iou={baseline_iou:.3f}", fontsize=10)
        axes[row, 1].axis("off")
        plt.colorbar(im1, ax=axes[row, 1], shrink=0.75, pad=0.02)

        im2 = axes[row, 2].imshow(best_raw, cmap="jet", aspect="auto")
        axes[row, 2].contour(
            gt_patch_mask.astype(float), levels=[0.5], colors="lime", linewidths=1.2
        )
        axes[row, 2].set_title(
            f"+ {best_family} value={best_value} — oracle_iou={best_iou:.3f} ({delta:+.3f})",
            fontsize=10,
        )
        axes[row, 2].axis("off")
        plt.colorbar(im2, ax=axes[row, 2], shrink=0.75, pad=0.02)

    fig.suptitle(
        f"scale={scale} — baseline vs. best-severity heatmap, all methods — focus combo {focus_key}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUTPUT_DIR / f"baseline_vs_best_heatmap_{scale}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

log.info("Saved focus-combo qualitative figures to %s", OUTPUT_DIR)

# %% Layer 2 figures — composed ensemble (cumulative severities within one family)
composed_endpoint_rows: dict[str, list[dict]] = {}
for method in METHOD_LABELS:
    plot_composed_curves(
        composed_iou_lookup[method],
        baseline_oracle_iou[method],
        N_SEVERITY_LEVELS,
        f"[{METHOD_TITLES[method]}] Composed-prototype oracle-IoU vs. severities averaged "
        f"— {len(combos)} combos",
        OUTPUT_DIR / f"composed_oracle_iou_curves__{method}.png",
        "oracle IoU (composed, mean across combos)",
    )
    composed_endpoint_rows[method] = aggregate_k_endpoint(
        composed_iou_lookup[method], N_SEVERITY_LEVELS, iou_lookup[method]
    )
    for row in sorted(composed_endpoint_rows[method], key=lambda r: -r["mean_delta_vs_baseline"]):
        log.info(
            "[%s composed] scale=%-5s %-22s composed=%.3f±%.3f (%+.3f vs. original, n=%d combos)",
            method,
            row["scale"],
            row["family"],
            row["mean_best_iou"],
            row["std_best_iou"],
            row["mean_delta_vs_baseline"],
            row["n_combos"],
        )

composed_endpoint_df = pd.DataFrame(
    [{"method": method, **row} for method, rows in composed_endpoint_rows.items() for row in rows]
)
composed_endpoint_df.to_csv(OUTPUT_DIR / "composed_endpoint.csv", index=False)
log.info("Wrote %s", OUTPUT_DIR / "composed_endpoint.csv")

log.info("Saved composed-ensemble curve figures to %s", OUTPUT_DIR)

# %% Layer 3 figures — all-augmentations composed, leave-one-out across families
leaveoneout_endpoint_rows: dict[str, list[dict]] = {}
for method in METHOD_LABELS:
    plot_leaveoneout_curves(
        all_aug_composed_iou_lookup[method],
        f"[{METHOD_TITLES[method]}] All-augmentations composed, leave-one-out "
        f"— {len(combos)} combos",
        OUTPUT_DIR / f"all_augmentations_composed_curves__{method}.png",
        "oracle IoU (all-augmentations composed, mean across combos)",
    )
    plot_leaveoneout_bar(
        all_aug_composed_iou_lookup[method],
        baseline_oracle_iou[method],
        f"[{METHOD_TITLES[method]}] All-augmentations composed, leave-one-out, "
        f"k={N_SEVERITY_LEVELS} endpoint (dashed = baseline)",
        OUTPUT_DIR / f"all_augmentations_composed_vs_baseline__{method}.png",
    )
    # Reuse aggregate_k_endpoint by aliasing held_out labels as "family" — the row shape
    # (scale, family/held_out, mean_best_iou, ...) is identical, only the label meaning differs.
    rows = []
    for scale in SCALES:
        cks = combo_keys_by_scale[scale]
        for held_out in HELD_OUT_LABELS:
            vals = [
                all_aug_composed_iou_lookup[method][(ck, scale, held_out, N_SEVERITY_LEVELS)]
                for ck in cks
                if (ck, scale, held_out, N_SEVERITY_LEVELS) in all_aug_composed_iou_lookup[method]
            ]
            base_vals = [
                iou_lookup[method][(ck, scale, first_family, first_val)]
                for ck in cks
                if (ck, scale, first_family, first_val) in iou_lookup[method]
            ]
            rows.append(
                {
                    "scale": scale,
                    "held_out": held_out,
                    "mean_iou": float(np.mean(vals)) if vals else float("nan"),
                    "std_iou": float(np.std(vals)) if vals else float("nan"),
                    "mean_delta_vs_baseline": (
                        float(np.mean(vals)) - float(np.mean(base_vals))
                        if vals and base_vals
                        else float("nan")
                    ),
                    "n_combos": len(vals),
                }
            )
    leaveoneout_endpoint_rows[method] = rows
    none_rows = [r for r in rows if r["held_out"] == "none"]
    for r in none_rows:
        log.info(
            "[%s leave-one-out] scale=%-5s held_out=none all_aug_composed=%.3f±%.3f "
            "(%+.3f vs. baseline)",
            method,
            r["scale"],
            r["mean_iou"],
            r["std_iou"],
            r["mean_delta_vs_baseline"],
        )

leaveoneout_endpoint_df = pd.DataFrame(
    [
        {"method": method, **row}
        for method, rows in leaveoneout_endpoint_rows.items()
        for row in rows
    ]
)
leaveoneout_endpoint_df.to_csv(OUTPUT_DIR / "leave_one_out_endpoint.csv", index=False)
log.info("Wrote %s", OUTPUT_DIR / "leave_one_out_endpoint.csv")

log.info("Saved leave-one-out figures (per method) to %s", OUTPUT_DIR)

# %% [markdown]
# ## Reading the results
#
# `single_proto` is the exact method from `augmented_prototype_oracle_iou.py`, recomputed
# here so every chart above has a same-run baseline to compare `knn_fg`/`proto_fgbg` against
# — not a number pulled from a separate script's log. (`knn_bg`/`knn_both` — augmenting the
# bg side — were dropped: across every family and scale they scored completely worse than
# `single_proto` and `knn_fg`, so this run only ever augments the fg side.) The headline
# question each curve figure answers: for each family/scale, does going contrastive (either
# knn variant) beat `single_proto`'s own baseline by more than `single_proto` manages on its
# own, and does the contrast need `knn_fg`'s full raw gallery or does `proto_fgbg`'s single
# bg mean vector capture the same effect for a fraction of the memory/compute? (The
# best-vs-baseline and method-comparison bar charts that used to summarize this were dropped
# — not useful; read the per-method oracle-IoU curve figures directly instead.)
#
# A family/scale where `knn_fg` clearly beats `proto_fgbg` says the fg-bg contrast benefits
# from keeping every raw patch as its own gallery entry — the kNN top-k mean is doing real
# work beyond what one collapsed bg mean vector can express, plausible when the bg gallery is
# heterogeneous enough (multiple surfaces/textures/lighting across the combo's scales) that
# a single mean washes out the useful negative signal. `proto_fgbg` matching or beating
# `knn_fg` says the reverse: the *contrast itself* (having any bg reference to subtract at
# all) is what buys the improvement over `single_proto`, and the extra raw-gallery machinery
# `knn_fg` carries isn't earning its cost. Either result vs. `single_proto` alone (no bg
# contrast at all) says whether subtracting a background reference helps localization in the
# first place, independent of how that reference is represented.
#
# The composed and leave-one-out sections ask the same "does it build up gradually or spike
# on one severity/family" questions `augmented_prototype_oracle_iou.py` asks for single_proto,
# now for both contrastive methods — a family where `knn_fg`'s leave-one-out curve pulls away
# from `proto_fgbg`'s as more severities compose in says that family's value compounds better
# in a growing raw gallery than in a growing-then-collapsed mean vector.

# %%
