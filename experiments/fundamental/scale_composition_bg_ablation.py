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

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from dinoisawesome import DinoEncoder, EncoderWithCache, compute_exemplar_features, load_annotations
from dinoisawesome.abc3 import INSTANCE_TYPE_GROUPS, available_instance_groups
from dinoisawesome.instance_detection import extract_patch_tokens

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _scale_composition_common import (  # noqa: E402
    build_composition_combos,
    scale_step_boxes,
    scale_step_name,
    split_fg_bg_patches,
)
from _shared.abc3_combos import combo_key  # noqa: E402
from _shared.dataset_pairs import REF_QUERY_PAIRS, RefQueryPair  # noqa: E402
from _shared.latency import cuda_timer, images_per_sec  # noqa: E402
from _shared.mask_geometry import pixel_mask_to_patch_mask, scale_crop_box  # noqa: E402
from _shared.pooled_gallery_cv import (  # noqa: E402
    MAX_BANK_SIZE_KNN_53,
    N_FOLDS_53,
    cap_bank_size,
    discover_all_instances,
    make_fold_role_splits,
)
from _shared.prototype_ops import knn_score_heatmap, score_heatmap  # noqa: E402
from _shared.qualitative_gallery import ScoredExample, save_score_gallery  # noqa: E402
from _shared.run_config import apply_overrides, load_run_config, resolve_output_dir  # noqa: E402
from _shared.stats import bootstrap_ci, bootstrap_prob_greater  # noqa: E402
from _shared.thresholding import achievable_iou, oracle_iou  # noqa: E402

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

DATA_ROOT = _REPO_ROOT / "data"

# abc5 has four ref/query pairs per part type — (1,2)/(3,4)/(5,6)/(7,8) (see
# _shared/dataset_pairs.py) — narrow for fast iteration, e.g. REF_QUERY_PAIRS[:1].
RUN_PAIRS: list[RefQueryPair] = REF_QUERY_PAIRS

DINO_VERSION = "v3"
DINO_SIZE = "base"
IMG_SIZE = 768
LAYER_IDX = 11  # last block of ViT-B/16 (depth 12)
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

# Bootstrap settings for the headline CI (Part 6) and the every_scale_all-vs-best_found
# significance check (Part 10).
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

SEED = 0

apply_overrides(globals(), load_run_config(__file__))
torch.manual_seed(SEED)

OUTPUT_DIR = resolve_output_dir(
    _REPO_ROOT / "outputs" / "fundamental_abc5" / "scale_composition_bg_ablation"
)

log.info(
    "RUN_PAIRS=%d units (%s)  |  DINO%s-%s img_size=%d layer=%d  |  n_scale_steps=%d",
    len(RUN_PAIRS),
    [p.unit for p in RUN_PAIRS],
    DINO_VERSION,
    DINO_SIZE,
    IMG_SIZE,
    LAYER_IDX,
    N_SCALE_STEPS,
)

# %% Scale-step naming + crop-box geometry (identical to scale_composition_oracle_iou.py)
T_VALUES: np.ndarray = np.linspace(0.0, 1.0, N_SCALE_STEPS + 1)

SCALE_NAMES: list[str] = [scale_step_name(i, N_SCALE_STEPS) for i in range(N_SCALE_STEPS + 1)]
log.info("Scale steps (global -> close): %s", SCALE_NAMES)

# %% Background composition table — same construction as scale_composition_oracle_iou.py's
# COMPOSITION_COMBOS, applied here to the BACKGROUND side instead of foreground: single scale,
# prefix-from-global, suffix-from-close, classic 3-point, and the two anchored-at-both-ends
# sweeps. Reused verbatim (not imported — see module docstring) so the two scripts' growth
# tables are guaranteed identical.
(
    BG_COMPOSITION_COMBOS,
    BG_PREFIX_NAMES,
    BG_SUFFIX_NAMES,
    BG_ANCHORED_INWARD_NAMES,
    BG_ANCHORED_OUTWARD_NAMES,
) = build_composition_combos(SCALE_NAMES)
FULL_BG_NAME = "+".join(SCALE_NAMES)  # "bg=every scale step" — the sibling script's fixed choice

# One representative (fg_scale, bg_composition) point whose individual combos get kept as PIL
# images + raw score maps for the worst/best-N qualitative gallery in Part 12 below — collecting
# this for every (fg_scale, bg_composition) pair would multiply memory/disk cost, so only this
# one point (a mid fg scale, bg pooled from every scale step — the sibling script's own fixed
# choice) is captured. Must each be a member of SCALE_NAMES / BG_COMPOSITION_COMBOS.
QUALITATIVE_FG_SCALE = "mid"
QUALITATIVE_BG_COMPOSITION = FULL_BG_NAME
QUALITATIVE_MAX_EXAMPLES = 60  # capped so the gallery figure itself stays a readable size

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

# %% Part 1 — discover every (part_type, instance-type group, ref instance) combo
combos: list[dict] = []
group_query_masks: dict[tuple[str, str], np.ndarray] = {}
group_ref_masks: dict[tuple[str, str], np.ndarray] = {}
ref_images: dict[str, Image.Image] = {}
query_images: dict[str, Image.Image] = {}

for pair in tqdm(RUN_PAIRS, desc="Discovering combos"):
    unit = pair.unit
    pair_data_dir = DATA_ROOT / pair.dataset
    ref_stem = f"{pair.part_type}_{pair.ref}"
    query_stem = f"{pair.part_type}_{pair.query}"
    ref_ann_path = pair_data_dir / "annotations" / ref_stem
    query_ann_path = pair_data_dir / "annotations" / query_stem

    groups = available_instance_groups(ref_ann_path)
    if not groups:
        log.warning("unit=%s: no instance-type groups annotated — skipping", unit)
        continue

    ref_anns = load_annotations(ref_ann_path)
    query_anns = load_annotations(query_ann_path)
    ref_images[unit] = Image.open(pair_data_dir / f"{ref_stem}.jpg").convert("RGB")
    query_images[unit] = Image.open(pair_data_dir / f"{query_stem}.jpg").convert("RGB")

    for group in groups:
        classes = INSTANCE_TYPE_GROUPS[group]
        ref_group_anns = [a for a in ref_anns if a["class"] in classes]
        query_group_anns = [a for a in query_anns if a["class"] in classes]
        if not query_group_anns:
            log.warning("unit=%s group=%s: no query GT instances — skipping", unit, group)
            continue
        group_query_masks[(unit, group)] = np.stack([a["mask"] for a in query_group_anns]).any(
            axis=0
        )
        if ref_group_anns:
            group_ref_masks[(unit, group)] = np.stack([a["mask"] for a in ref_group_anns]).any(
                axis=0
            )
        for ref_ann in ref_group_anns:
            combos.append(
                {
                    "unit": unit,
                    "dataset": pair.dataset,
                    "part_type": pair.part_type,
                    "group": group,
                    "class": ref_ann["class"],
                    "instance_id": ref_ann["instance_id"],
                    "ref_mask": ref_ann["mask"],
                }
            )

if not combos:
    raise RuntimeError(
        f"No combos discovered for RUN_PAIRS={[p.unit for p in RUN_PAIRS]} — every pair was "
        "skipped (no annotated instance-type groups or no query GT)."
    )
log.info(
    "Discovered %d (unit, group, instance) combos across %d units (%d part types)",
    len(combos),
    len({c["unit"] for c in combos}),
    len({c["part_type"] for c in combos}),
)

# %% Part 2 — build the N_SCALE_STEPS + 1 crops per combo. All-or-nothing per combo (see
# scale_step_boxes's docstring: boxes shrink monotonically, so `close` clearing MIN_CROP_SIZE
# guarantees every other step does too).
usable_combo_keys: set[tuple] = set()
for combo in tqdm(combos, desc="Building scale-step crops"):
    ref_img = ref_images[combo["unit"]]
    group_mask = group_ref_masks.get((combo["unit"], combo["group"]), combo["ref_mask"])
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

# %% Part 3 — encoder + query-image patch tokens + GT patch masks. Also encodes each unit's
# *reference* image full-extent (the "global" scale step's own crop box is the whole ref image —
# see scale_step_boxes) and its own GT patch mask, used by achievable_iou below (Part 5): the
# exemplar image already has its own GT, so it can stand in for a query-time-labeled reference
# without needing the query's own GT to tune a threshold.
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

latency_rows: list[dict] = []

query_encodings: dict[str, tuple[torch.Tensor, int, int]] = {}
with cuda_timer() as t_query_encode:
    for unit in tqdm(sorted(query_images), desc="Encoding query images"):
        tokens, q_h, q_w = extract_patch_tokens(encoder, query_images[unit], LAYER_IDX, debias=True)
        query_encodings[unit] = (tokens, q_h, q_w)
latency_rows.append(
    {
        "phase": "query_image_encode",
        "elapsed_s": t_query_encode["elapsed_s"],
        "n_units": len(query_images),
        "units_per_sec": images_per_sec(len(query_images), t_query_encode["elapsed_s"]),
    }
)

ref_encodings: dict[str, tuple[torch.Tensor, int, int]] = {}
with cuda_timer() as t_ref_encode:
    for unit in tqdm(sorted(ref_images), desc="Encoding reference images (achievable_iou)"):
        tokens, r_h, r_w = extract_patch_tokens(encoder, ref_images[unit], LAYER_IDX, debias=True)
        ref_encodings[unit] = (tokens, r_h, r_w)
latency_rows.append(
    {
        "phase": "ref_image_encode",
        "elapsed_s": t_ref_encode["elapsed_s"],
        "n_units": len(ref_images),
        "units_per_sec": images_per_sec(len(ref_images), t_ref_encode["elapsed_s"]),
    }
)

gt_patch_masks: dict[tuple[str, str], np.ndarray] = {}
for (unit, group), pixel_mask in group_query_masks.items():
    _, q_h, q_w = query_encodings[unit]
    gt_patch_masks[(unit, group)] = pixel_mask_to_patch_mask(
        pixel_mask, q_h, q_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )

ref_patch_masks: dict[tuple[str, str], np.ndarray] = {}
for (unit, group), pixel_mask in group_ref_masks.items():
    _, r_h, r_w = ref_encodings[unit]
    ref_patch_masks[(unit, group)] = pixel_mask_to_patch_mask(
        pixel_mask, r_h, r_w, IMG_SIZE, MASK_PATCH_THRESHOLD
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

with cuda_timer() as t_crop_encode:
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
                IMG_SIZE,
                MASK_PATCH_THRESHOLD,
                bg_exclude_mask_px=bg_exclude_mask_px,
            )
            # Kept on CPU: with REF_QUERY_PAIRS spanning 16 ref/query pairs, every combo's
            # every-scale fg/bg bank held on GPU simultaneously no longer fits (abc3-only fit in
            # ~12GB, the combined pool doesn't) — moved back to the query's device per combo in
            # Part 5.
            fg_by_scale[(ck, name)] = fg.cpu()
            bg_by_scale[(ck, name)] = bg.cpu()
latency_rows.append(
    {
        "phase": "gallery_crop_encode",
        "elapsed_s": t_crop_encode["elapsed_s"],
        "n_units": len(clean_items),
        "units_per_sec": images_per_sec(len(clean_items), t_crop_encode["elapsed_s"]),
    }
)

log.info("Built per-scale fg/bg galleries for %d combos", len(usable_combo_keys))

# %% Part 5 — main scoring: fg fixed to each single scale, bg swept through BG_COMPOSITION_COMBOS.
# single_proto never reads bg_bank (see module docstring), so it's scored once per (combo,
# fg_scale) as a bg-invariant reference, not swept. Alongside oracle_iou (tunes its threshold
# against the query's own GT — an upper bound), also scores the *same* gallery against the
# reference/exemplar image's own full extent + its own GT (ref_encodings/ref_patch_masks from
# Part 3) to get achievable_iou: a threshold tuned on the exemplar, transferred as-is to the
# query — the number a deployed pipeline without query-time labels would actually see. One extra
# score_heatmap/knn_score_heatmap call per gallery, not per query, since there is exactly one
# query per combo in this fixed-pair script.
# bg_iou_lookup[fg_scale][bg_composition_name][ck] -> knn_fgbg oracle IoU
BgIouLookup = dict[str, dict[str, dict[tuple, float]]]
bg_iou_lookup: BgIouLookup = {
    fg_scale: {bg_name: {} for bg_name in BG_COMPOSITION_COMBOS} for fg_scale in SCALE_NAMES
}
# bg_ai_lookup[fg_scale][bg_composition_name][ck] -> knn_fgbg achievable IoU
bg_ai_lookup: BgIouLookup = {
    fg_scale: {bg_name: {} for bg_name in BG_COMPOSITION_COMBOS} for fg_scale in SCALE_NAMES
}
# fg_only_iou[fg_scale][ck] -> single_proto oracle IoU (bg-invariant reference)
FgOnlyIou = dict[str, dict[tuple, float]]
fg_only_iou: FgOnlyIou = {fg_scale: {} for fg_scale in SCALE_NAMES}
fg_only_ai: FgOnlyIou = {fg_scale: {} for fg_scale in SCALE_NAMES}

# gt_area_frac_by_key[(unit, group)] -> query GT's own patch-mask coverage — same for every
# fg_scale/bg_composition since it only depends on the query, computed once for Part 11's
# size-correlation analysis rather than recomputed per (fg_scale, bg_name).
gt_area_frac_by_key: dict[tuple[str, str], float] = {
    key: float(gt.sum()) / gt.size for key, gt in gt_patch_masks.items()
}

qualitative_examples: list[ScoredExample] = []

with cuda_timer() as t_scoring_1_1:
    for combo in tqdm(combos, desc="Part 5: scoring bg ablation"):
        ck = combo_key(combo)
        if ck not in usable_combo_keys:
            continue
        unit, group = ck[0], ck[1]
        gt = gt_patch_masks.get((unit, group))
        if gt is None:
            continue
        q_tokens, q_h, q_w = query_encodings[unit]
        ref_tokens, ref_h, ref_w = ref_encodings[unit]
        ref_gt = ref_patch_masks.get((unit, group))
        if ref_gt is None:
            log.warning(
                "unit=%s group=%s: no reference-image GT — achievable_iou left NaN", unit, group
            )

        for fg_scale in SCALE_NAMES:
            fg_bank = fg_by_scale[(ck, fg_scale)].to(q_tokens.device)

            proto = compute_exemplar_features(fg_bank, mode="mean")
            raw_proto = score_heatmap(q_tokens, proto, q_h, q_w)
            fg_only_iou[fg_scale][ck] = oracle_iou(raw_proto, gt, ORACLE_THRESHOLD_STEPS)
            if ref_gt is not None:
                ref_raw_proto = score_heatmap(ref_tokens, proto, ref_h, ref_w)
                fg_only_ai[fg_scale][ck] = achievable_iou(
                    ref_raw_proto, ref_gt, raw_proto, gt, ORACLE_THRESHOLD_STEPS
                )
            else:
                fg_only_ai[fg_scale][ck] = float("nan")

            for bg_name, bg_members in BG_COMPOSITION_COMBOS.items():
                bg_bank = torch.cat([bg_by_scale[(ck, m)] for m in bg_members], dim=0).to(
                    q_tokens.device
                )
                raw_knn = knn_score_heatmap(
                    q_tokens, fg_bank, bg_bank, KNN_FGBG_NUM_NEIGHBOURS, q_h, q_w
                )
                bg_iou_lookup[fg_scale][bg_name][ck] = oracle_iou(
                    raw_knn, gt, ORACLE_THRESHOLD_STEPS
                )
                if ref_gt is not None:
                    ref_raw_knn = knn_score_heatmap(
                        ref_tokens, fg_bank, bg_bank, KNN_FGBG_NUM_NEIGHBOURS, ref_h, ref_w
                    )
                    bg_ai_lookup[fg_scale][bg_name][ck] = achievable_iou(
                        ref_raw_knn, ref_gt, raw_knn, gt, ORACLE_THRESHOLD_STEPS
                    )
                else:
                    bg_ai_lookup[fg_scale][bg_name][ck] = float("nan")

                if (
                    fg_scale == QUALITATIVE_FG_SCALE
                    and bg_name == QUALITATIVE_BG_COMPOSITION
                    and len(qualitative_examples) < QUALITATIVE_MAX_EXAMPLES
                ):
                    qualitative_examples.append(
                        ScoredExample(
                            label=f"{unit}/{group}",
                            image=query_images[unit],
                            raw=raw_knn,
                            gt=gt,
                            score=bg_iou_lookup[fg_scale][bg_name][ck],
                        )
                    )
latency_rows.append(
    {
        "phase": "scoring_1_1",
        "elapsed_s": t_scoring_1_1["elapsed_s"],
        "n_units": len(usable_combo_keys),
        "units_per_sec": images_per_sec(len(usable_combo_keys), t_scoring_1_1["elapsed_s"]),
    }
)

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
        vals = np.array(list(bg_iou_lookup[fg_scale][bg_name].values()))
        mean, std, n = mean_std_iou(bg_iou_lookup[fg_scale][bg_name])
        # Percentile bootstrap CI on the mean, alongside the plain std this script already
        # reported — std alone doesn't say whether two bg compositions' means are actually
        # distinguishable or both plausible draws from the same underlying distribution.
        if n > 0:
            _, ci_lo, ci_hi = bootstrap_ci(vals, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
        else:
            ci_lo = ci_hi = float("nan")
        achievable_vals = np.array(list(bg_ai_lookup[fg_scale][bg_name].values()))
        mean_ai_val = float(np.nanmean(achievable_vals)) if len(achievable_vals) else float("nan")
        std_ai_val = float(np.nanstd(achievable_vals)) if len(achievable_vals) else float("nan")
        bg_ablation_rows.append(
            {
                "fg_scale": fg_scale,
                "bg_composition": bg_name,
                "n_bg_scales": len(bg_members),
                "bg_members": "+".join(bg_members),
                "mean_iou": mean,
                "std_iou": std,
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
                "mean_achievable_iou": mean_ai_val,
                "std_achievable_iou": std_ai_val,
                "oracle_minus_achievable_gap": (
                    mean - mean_ai_val
                    if not (np.isnan(mean) or np.isnan(mean_ai_val))
                    else float("nan")
                ),
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

def mean_ai(fg_scale: str, bg_name: str) -> float:
    vals = np.array(list(bg_ai_lookup[fg_scale][bg_name].values()))
    return float(np.nanmean(vals)) if len(vals) else float("nan")


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
            # achievable_iou (threshold tuned on the exemplar's own GT, transferred as-is) for
            # the same four strategies — see Part 5's module comment and _shared/thresholding.py.
            "own_scale_only_achievable": mean_ai(fg_scale, fg_scale),
            "global_only_achievable": mean_ai(fg_scale, "global"),
            "every_scale_all_achievable": mean_ai(fg_scale, FULL_BG_NAME),
            "best_found_achievable": mean_ai(fg_scale, best_name),
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
    f"1-1 (single ref/query pair) — Background-composition strategies per fixed fg scale, "
    f"n={N_SCALE_STEPS} steps ({len(usable_combo_keys)} combos)"
)
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "bg_headline_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved %s", OUTPUT_DIR / "bg_headline_comparison.png")

# %% Part 6b — oracle IoU (upper bound, tunes the threshold against the query's own GT) vs.
# achievable IoU (a threshold tuned on the exemplar's own GT, transferred as-is to the query —
# what a deployed pipeline without query-time labels would actually get), same four strategies
# and same per-fg-scale x-axis as bg_headline_comparison.png above.
fig, ax = plt.subplots(figsize=(11, 6))
STRATEGY_KEYS = ["own_scale_only", "global_only", "every_scale_all", "best_found"]
for strategy, label, color in zip(STRATEGY_KEYS, STRATEGY_LABELS, STRATEGY_COLORS):
    ax.plot(
        SCALE_NAMES,
        headline_df[strategy],
        marker="o",
        linestyle="-",
        color=color,
        label=f"{label} (oracle)",
    )
    ax.plot(
        SCALE_NAMES,
        headline_df[f"{strategy}_achievable"],
        marker="^",
        linestyle="--",
        color=color,
        alpha=0.6,
        label=f"{label} (achievable)",
    )
ax.set_xlabel("foreground scale (fixed, single scale only)")
ax.set_ylabel("mean IoU across combos")
ax.set_ylim(0, 1.0)
ax.set_title(
    "1-1 (single ref/query pair) — Oracle (upper bound) vs. achievable "
    "(exemplar-tuned threshold) IoU, per bg strategy"
)
ax.legend(fontsize=7, loc="lower right", ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "oracle_vs_achievable.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(
    "Oracle-vs-achievable gap (mean oracle_iou - mean achievable_iou), every_scale_all strategy:"
)
for _, row in headline_df.iterrows():
    log.info(
        "  fg=%-8s gap=%.3f (oracle=%.3f achievable=%.3f)",
        row.fg_scale,
        row.every_scale_all - row.every_scale_all_achievable,
        row.every_scale_all,
        row.every_scale_all_achievable,
    )
log.info("Saved %s", OUTPUT_DIR / "oracle_vs_achievable.png")

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
    fig.suptitle(
        f"1-1 (single ref/query pair) — Background-composition growth curves, "
        f"fg fixed at '{fg_scale}'"
    )
    fig.tight_layout()
    _safe_name = fg_scale.replace("/", "-")
    fig.savefig(OUTPUT_DIR / f"bg_growth__fg_{_safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
log.info("Saved %d per-fg-scale bg-growth-curve figures", len(SCALE_NAMES))

# %% Part 8 — 5-3 pooled gallery, cross-validated: does "own-scale-only bg collapses at
# tight crops, global-only bg captures most of every-scale bg's benefit" (Part 6's finding)
# hold when the gallery is pooled from 5 training images instead of one? Reuses
# `split_fg_bg_patches`/`knn_score_heatmap`/`oracle_iou` unchanged — only discovery,
# per-instance crop-building, and fold/role assignment are new (see
# `_shared/pooled_gallery_cv.py`). Scoped to 3 representative fg scales (global/mid/close,
# not all 7 N_SCALE_STEPS points) x the 3 named bg strategies from Part 6's own headline
# table (not the full 28-entry BG_COMPOSITION_COMBOS growth-curve sweep).
PART_TYPES_53 = sorted({c["part_type"] for c in combos})
discovery_53 = discover_all_instances(DATA_ROOT, "abc5", PART_TYPES_53)
POOL_SCALES_53 = ["global", "mid", "close"]

usable_instances_53: list[dict] = []
for inst in tqdm(discovery_53.instances, desc="5-3: building global/mid/close crops"):
    img = discovery_53.images[(inst.part_type, inst.image_number)]
    close_box = scale_crop_box(inst.mask, "close", CROP_PADDING_FRACTION)
    if close_box[2] - close_box[0] < MIN_CROP_SIZE or close_box[3] - close_box[1] < MIN_CROP_SIZE:
        continue
    crops: dict = {}
    for scale in POOL_SCALES_53:
        x0, y0, x1, y1 = scale_crop_box(inst.mask, scale, CROP_PADDING_FRACTION)
        crops[scale] = {
            "img": img.crop((x0, y0, x1, y1)),
            "mask_px": inst.mask[y0:y1, x0:x1],
            "bg_exclude_mask_px": inst.bg_exclude_mask[y0:y1, x0:x1],
        }
    usable_instances_53.append(
        {
            "part_type": inst.part_type,
            "group": inst.group,
            "image_number": inst.image_number,
            "crops": crops,
        }
    )
log.info("5-3: usable instances %d/%d", len(usable_instances_53), len(discovery_53.instances))

image_encodings_53: dict[tuple[str, int], dict] = {}
for key in tqdm(sorted(discovery_53.images), desc="5-3: encoding images"):
    tokens, h, w = extract_patch_tokens(encoder, discovery_53.images[key], LAYER_IDX, debias=True)
    image_encodings_53[key] = {"tokens": tokens, "h": h, "w": w}

gt_patch_masks_53: dict[tuple[str, str, int], np.ndarray] = {}
for (part_type, group, n), pixel_mask in discovery_53.gt_masks.items():
    q = image_encodings_53[(part_type, n)]
    gt_patch_masks_53[(part_type, group, n)] = pixel_mask_to_patch_mask(
        pixel_mask, q["h"], q["w"], IMG_SIZE, MASK_PATCH_THRESHOLD
    )

fg_by_inst_scale_53: dict[tuple[int, str], torch.Tensor] = {}
bg_by_inst_scale_53: dict[tuple[int, str], torch.Tensor] = {}
clean_items_53: list[tuple] = []
for i, inst in enumerate(usable_instances_53):
    for scale, crop in inst["crops"].items():
        clean_items_53.append((i, scale, crop["img"], crop["mask_px"], crop["bg_exclude_mask_px"]))

for i in tqdm(range(0, len(clean_items_53), chunk_size), desc="5-3: encoding crops"):
    chunk = clean_items_53[i : i + chunk_size]
    out = encoder([c[2] for c in chunk], layers=[LAYER_IDX], debias=True)
    chunk_patches = out.patches[:, 0]
    grid_h, grid_w = chunk_patches.shape[1], chunk_patches.shape[2]
    for (idx, scale, _, mask_px, bg_exclude_mask_px), patch_tokens in zip(chunk, chunk_patches):
        fg, bg = split_fg_bg_patches(
            patch_tokens,
            mask_px,
            grid_h,
            grid_w,
            f"5-3 inst{idx} scale={scale}",
            IMG_SIZE,
            MASK_PATCH_THRESHOLD,
            bg_exclude_mask_px=bg_exclude_mask_px,
        )
        fg_by_inst_scale_53[(idx, scale)] = fg.cpu()
        bg_by_inst_scale_53[(idx, scale)] = bg.cpu()

instances_by_pg_53: dict[tuple[str, str], list[int]] = defaultdict(list)
for i, inst in enumerate(usable_instances_53):
    instances_by_pg_53[(inst["part_type"], inst["group"])].append(i)
groups_by_pt_53: dict[str, list[str]] = defaultdict(list)
for pt, g in instances_by_pg_53:
    groups_by_pt_53[pt].append(g)
log.info(
    "5-3: built per-scale fg/bg banks for %d instances across %d (part_type, group) pairs",
    len(usable_instances_53),
    len(instances_by_pg_53),
)

fold_splits_53 = make_fold_role_splits(PART_TYPES_53)  # truly randomized, not SEED-reproducible
results_53: list[dict] = []
n_units_53 = N_FOLDS_53 * len(PART_TYPES_53)

with cuda_timer() as t_scoring_5_3, tqdm(
    total=n_units_53, desc="5-3: cross-validated fit + score"
) as pbar:
    for fold_idx, split in enumerate(fold_splits_53):
        for part_type in PART_TYPES_53:
            train_numbers, eval_numbers = split[part_type]
            for group in groups_by_pt_53.get(part_type, []):
                idxs = instances_by_pg_53[(part_type, group)]
                pool_idxs = [
                    i for i in idxs if usable_instances_53[i]["image_number"] in train_numbers
                ]
                if not pool_idxs:
                    continue
                for eval_number in eval_numbers:
                    key = (part_type, group, eval_number)
                    if key not in gt_patch_masks_53:
                        continue
                    q = image_encodings_53[(part_type, eval_number)]
                    gt = gt_patch_masks_53[key]
                    for fg_scale in POOL_SCALES_53:
                        fg_bank = cap_bank_size(
                            torch.cat(
                                [fg_by_inst_scale_53[(i, fg_scale)] for i in pool_idxs], dim=0
                            ),
                            MAX_BANK_SIZE_KNN_53,
                            SEED,
                        ).to(q["tokens"].device)
                        if fg_bank.shape[0] == 0:
                            continue
                        for strategy, members in {
                            "own_scale_only": [fg_scale],
                            "global_only": ["global"],
                            "every_scale_all": POOL_SCALES_53,
                        }.items():
                            bg_bank = cap_bank_size(
                                torch.cat(
                                    [
                                        bg_by_inst_scale_53[(i, m)]
                                        for i in pool_idxs
                                        for m in members
                                    ],
                                    dim=0,
                                ),
                                MAX_BANK_SIZE_KNN_53,
                                SEED,
                            ).to(q["tokens"].device)
                            if bg_bank.shape[0] == 0:
                                continue
                            raw_knn = knn_score_heatmap(
                                q["tokens"],
                                fg_bank,
                                bg_bank,
                                KNN_FGBG_NUM_NEIGHBOURS,
                                q["h"],
                                q["w"],
                            )
                            results_53.append(
                                {
                                    "fg_scale": fg_scale,
                                    "bg_strategy": strategy,
                                    "oracle_iou": oracle_iou(raw_knn, gt, ORACLE_THRESHOLD_STEPS),
                                }
                            )
            pbar.update(1)
latency_rows.append(
    {
        "phase": "scoring_5_3",
        "elapsed_s": t_scoring_5_3["elapsed_s"],
        "n_units": len(results_53),
        "units_per_sec": images_per_sec(len(results_53), t_scoring_5_3["elapsed_s"]),
    }
)

results_53_df = pd.DataFrame(results_53)
summary_53_rows = []
for fg_scale in POOL_SCALES_53:
    row_11 = headline_df[headline_df.fg_scale == fg_scale].iloc[0]
    row = {
        "fg_scale": fg_scale,
        "own_scale_only_1_1": row_11.own_scale_only,
        "global_only_1_1": row_11.global_only,
        "every_scale_all_1_1": row_11.every_scale_all,
    }
    for strategy in ["own_scale_only", "global_only", "every_scale_all"]:
        vals = results_53_df.loc[
            (results_53_df.fg_scale == fg_scale) & (results_53_df.bg_strategy == strategy),
            "oracle_iou",
        ]
        row[f"{strategy}_5_3"] = float(vals.mean()) if len(vals) else float("nan")
        row[f"{strategy}_n_5_3"] = len(vals)
    summary_53_rows.append(row)
summary_53_df = pd.DataFrame(summary_53_rows)
summary_53_df.to_csv(OUTPUT_DIR / "comparison_1_1_vs_5_3.csv", index=False)
log.info("1-1 vs 5-3 bg-strategy comparison:")
for _, row in summary_53_df.iterrows():
    log.info(
        "  fg=%-8s own: 1-1=%.3f 5-3=%.3f | global: 1-1=%.3f 5-3=%.3f | all: 1-1=%.3f 5-3=%.3f",
        row.fg_scale,
        row.own_scale_only_1_1,
        row.own_scale_only_5_3,
        row.global_only_1_1,
        row.global_only_5_3,
        row.every_scale_all_1_1,
        row.every_scale_all_5_3,
    )
log.info("Wrote %s", OUTPUT_DIR / "comparison_1_1_vs_5_3.csv")

# %% Part 9 — latency/throughput: every phase above traded off against wall-clock cost, which no
# figure in this script reported before now. `torch.cuda.synchronize()` is called around every
# timed block (see `_shared/latency.py`) so GPU-async dispatch doesn't understate elapsed time.
# This script has no single clean sweep axis to plot latency against (Part 5's scoring loop
# scores every (fg_scale, bg_composition) pair in one untimed-per-point pass, and restructuring
# that loop just to isolate per-point timing is out of scope for a purely additive change) — so
# unlike resolution_ablation.py/training_set_size_ablation.py's latency.png, this just logs and
# tabulates per-phase totals.
cache_hits, cache_misses = encoder.total_hits, encoder.total_misses
cache_total = cache_hits + cache_misses
latency_rows.append(
    {
        "phase": "total",
        "elapsed_s": sum(r["elapsed_s"] for r in latency_rows),
        "n_units": float("nan"),
        "units_per_sec": float("nan"),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": cache_hits / cache_total if cache_total > 0 else float("nan"),
    }
)
latency_df = pd.DataFrame(latency_rows)
latency_df.to_csv(OUTPUT_DIR / "latency.csv", index=False)
log.info("Latency by phase (GPU-synchronized wall-clock time):")
for _, row in latency_df.iterrows():
    log.info("  phase=%-20s elapsed=%.1fs n_units=%s", row.phase, row.elapsed_s, row.n_units)
log.info("Wrote %s", OUTPUT_DIR / "latency.csv")

# %% Part 10 — is the "every_scale_all vs. best_found" gap (Part 6's headline finding) real, or
# combo-to-combo noise? An unpaired bootstrap comparison (see _shared/stats.py) of the sibling
# script's fixed bg choice ("every scale, always") against the best bg composition this script
# actually found, per fg scale — the significance check `bg_headline_comparison.png` leaves the
# reader to eyeball.
significance_rows = []
for fg_scale in SCALE_NAMES:
    best_name = headline_df.loc[headline_df.fg_scale == fg_scale, "best_bg_composition"].iloc[0]
    all_vals = np.array(list(bg_iou_lookup[fg_scale][FULL_BG_NAME].values()))
    best_vals = np.array(list(bg_iou_lookup[fg_scale][best_name].values()))
    if len(all_vals) == 0 or len(best_vals) == 0:
        continue
    prob_best_greater = bootstrap_prob_greater(
        best_vals, all_vals, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
    )
    significance_rows.append(
        {
            "fg_scale": fg_scale,
            "best_bg_composition": best_name,
            "prob_best_beats_every_scale_all": prob_best_greater,
            "n_every_scale_all": len(all_vals),
            "n_best": len(best_vals),
        }
    )
significance_df = pd.DataFrame(significance_rows)
significance_df.to_csv(OUTPUT_DIR / "bg_significance.csv", index=False)
log.info(
    "Best-found-vs-every-scale-all significance (P(best mean > every_scale_all mean) under "
    "%d-resample bootstrap; near 0.5 = indistinguishable from noise):",
    N_BOOTSTRAP,
)
for _, row in significance_df.iterrows():
    log.info(
        "  fg=%-8s best=%-30s P(best beats every_scale_all)=%.3f (n=%d vs n=%d)",
        row.fg_scale,
        row.best_bg_composition,
        row.prob_best_beats_every_scale_all,
        row.n_best,
        row.n_every_scale_all,
    )
log.info("Wrote %s", OUTPUT_DIR / "bg_significance.csv")

# %% Part 11 — does oracle IoU correlate with object size? An aggregate mean (every figure above)
# can hide "multi-scale bg only helps small/large instances" — `gt_area_frac_by_key` (the query
# GT's own patch-mask coverage, computed once in Part 5) lets us check, mirroring the
# pearson/spearman correlation pattern `scale_composition_adaptive_oracle.py` already established
# for instance size vs. optimal scale, grouped the same way this script's own headline breakdown
# (Part 6) already is: per fg_scale, across the four named bg strategies.
correlation_rows = []
for fg_scale in SCALE_NAMES:
    best_name = headline_df.loc[headline_df.fg_scale == fg_scale, "best_bg_composition"].iloc[0]
    for strategy_label, bg_name in [
        ("own_scale_only", fg_scale),
        ("global_only", "global"),
        ("every_scale_all", FULL_BG_NAME),
        ("best_found", best_name),
    ]:
        lookup = bg_iou_lookup[fg_scale][bg_name]
        cks = [ck for ck in lookup if (ck[0], ck[1]) in gt_area_frac_by_key]
        if len(cks) < 3:
            continue
        area_fracs = [gt_area_frac_by_key[(ck[0], ck[1])] for ck in cks]
        ious = [lookup[ck] for ck in cks]
        pearson_r, pearson_p = pearsonr(area_fracs, ious)
        spearman_r, spearman_p = spearmanr(area_fracs, ious)
        correlation_rows.append(
            {
                "fg_scale": fg_scale,
                "bg_strategy": strategy_label,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "n_samples": len(cks),
            }
        )
size_correlation_df = pd.DataFrame(correlation_rows)
size_correlation_df.to_csv(OUTPUT_DIR / "size_correlation.csv", index=False)

# Object-size terciles (global, computed once across every combo so the same size cutoffs apply
# everywhere) x oracle IoU for the every_scale_all strategy (the sibling script's own fixed
# choice — the most representative single strategy to check), faceted per fg_scale like
# bg_growth__fg_<scale>.png above.
corr_flat_rows = [
    {"fg_scale": fg_scale, "ck": ck, "gt_area_frac": gt_area_frac_by_key[(ck[0], ck[1])], "oracle_iou": v}
    for fg_scale in SCALE_NAMES
    for ck, v in bg_iou_lookup[fg_scale][FULL_BG_NAME].items()
    if (ck[0], ck[1]) in gt_area_frac_by_key
]
corr_flat_df = pd.DataFrame(corr_flat_rows)
try:
    corr_flat_df["size_tercile"] = pd.qcut(
        corr_flat_df["gt_area_frac"], 3, labels=["small", "medium", "large"]
    )
except ValueError:
    log.warning(
        "gt_area_frac has too few distinct values for 3 clean terciles — falling back to "
        "qcut's own duplicate-safe binning (labels become numeric ranges, not small/medium/large)"
    )
    corr_flat_df["size_tercile"] = pd.qcut(corr_flat_df["gt_area_frac"], 3, duplicates="drop")

fig, axes = plt.subplots(1, len(SCALE_NAMES), figsize=(3.2 * len(SCALE_NAMES), 5), sharey=True)
for ax, fg_scale in zip(axes, SCALE_NAMES):
    tercile_means = (
        corr_flat_df[corr_flat_df.fg_scale == fg_scale].groupby("size_tercile", observed=True)[
            "oracle_iou"
        ]
        .mean()
    )
    tercile_means.plot(kind="bar", ax=ax, color=KNN_COLOR)
    ax.set_title(fg_scale, fontsize=9)
    ax.set_xlabel("size tercile")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("mean oracle IoU (bg=every scale, knn_fgbg)")
fig.suptitle("Does object size predict oracle IoU, per fg scale?")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "size_correlation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Wrote %s and %s", OUTPUT_DIR / "size_correlation.csv", OUTPUT_DIR / "size_correlation.png")

# %% Part 12 — worst/best-N qualitative gallery at one representative (fg_scale, bg_composition)
# point (see QUALITATIVE_FG_SCALE/QUALITATIVE_BG_COMPOSITION above) — every other figure in this
# script averages across combos; this shows actual individual query images so a failure mode is
# visible instead of washed out by the mean.
if qualitative_examples:
    save_score_gallery(
        qualitative_examples,
        OUTPUT_DIR / "qualitative_worst_best.png",
        n=5,
        score_name="oracle_iou",
        title=(
            f"Worst/best oracle_iou examples: fg_scale={QUALITATIVE_FG_SCALE} "
            f"bg_composition={QUALITATIVE_BG_COMPOSITION}"
        ),
    )
    log.info(
        "Wrote %s (%d examples)", OUTPUT_DIR / "qualitative_worst_best.png", len(qualitative_examples)
    )
else:
    log.warning("No qualitative examples collected for the representative point")

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
# - **`oracle_vs_achievable.png`/`bg_headline_comparison.csv`'s `*_achievable` columns and
#   `bg_ablation_iou.csv`'s `mean_achievable_iou`/`oracle_minus_achievable_gap` columns** —
#   `oracle_iou` everywhere else in this script is an upper bound (tunes its threshold against the
#   query's own GT); achievable_iou tunes the threshold on the *exemplar's* own GT instead (the
#   same image that built the gallery) and transfers it as-is to the query — the number a deployed
#   pipeline without query-time labels would actually see. A bg-strategy ranking that holds for
#   oracle but not achievable IoU means it's a trend in "how separable the scores could be," not
#   in what a real threshold captures — check both before trusting `bg_headline_comparison.png`
#   alone.
# - **`latency.csv`** — GPU-synchronized wall-clock cost (see `_shared/latency.py`) per phase
#   (reference/query image encode, gallery-crop encode, 1-1 scoring, 5-3 scoring) plus cache
#   hit-rate on the final "total" row. No `latency.png`/`accuracy_vs_latency.png` here — Part 5's
#   scoring loop doesn't isolate per-fg-scale or per-bg-composition timing, so there's no natural
#   per-point axis to plot latency against without restructuring that loop.
# - **`bg_significance.csv`** — an unpaired bootstrap comparison (2000 resamples, see
#   `_shared/stats.py`) of the sibling script's fixed "every scale, always" bg choice against the
#   best bg composition actually found, per fg scale: `prob_best_beats_every_scale_all` near 0.5
#   means the apparent edge in `bg_headline_comparison.png` is not distinguishable from
#   combo-to-combo noise — a quantitative version of that figure's eyeball comparison.
# - **`size_correlation.csv`/`.png`** — does oracle IoU correlate with the query GT's own area
#   fraction (`gt_area_frac`)? `.csv` covers all four named bg strategies per fg scale;
#   `.png`'s tercile bars use only the "every scale (all)" strategy (the sibling script's own
#   fixed choice) per fg scale, for readability — check whether a bg-composition benefit
#   concentrates on small objects specifically before generalizing it.
# - **`qualitative_worst_best.png`** — actual worst-5/best-5 query images (crop, raw score map,
#   GT mask) at one representative (fg_scale, bg_composition) point, not an average — every other
#   figure here plots a mean or an averaged heatmap, which can't show *why* a config fails on a
#   specific image.

# %%
