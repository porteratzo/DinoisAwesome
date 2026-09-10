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
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from dinoisawesome import DinoEncoder, EncoderWithCache, compute_exemplar_features, load_annotations
from dinoisawesome.abc3 import INSTANCE_TYPE_GROUPS, available_instance_groups

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _scale_composition_common import (  # noqa: E402
    scale_step_boxes,
    scale_step_name,
    split_fg_bg_patches,
)
from _shared.abc3_combos import combo_key  # noqa: E402
from _shared.dataset_pairs import REF_QUERY_PAIRS, RefQueryPair  # noqa: E402
from _shared.latency import cuda_timer, images_per_sec  # noqa: E402
from _shared.mask_geometry import pixel_mask_to_patch_mask  # noqa: E402
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
CROP_PADDING_FRACTION = 1.0
MIN_CROP_SIZE = 128

ORACLE_THRESHOLD_STEPS = 25
KNN_FGBG_NUM_NEIGHBOURS = 10

N_SCALE_STEPS = 6

METHODS: list[str] = ["single_proto", "knn_fgbg"]
METHOD_COLOR: dict[str, str] = {"single_proto": "#7f8c8d", "knn_fgbg": "#2ecc71"}

# One representative (ref_scale, method) row whose individual per-query-scale-cell samples get
# kept as PIL crops + raw score maps for the worst/best-N qualitative gallery — collecting every
# row would multiply memory/disk cost 7x, so only the midpoint ref scale (QUALITATIVE_REF_SCALE,
# set below once SCALE_NAMES exists) and the stronger knn_fgbg method are captured.
QUALITATIVE_METHOD = "knn_fgbg"
QUALITATIVE_MAX_EXAMPLES = 60  # capped so the gallery figure itself stays a readable size

# Bootstrap settings for the per-cell CI added to the headline matrix CSV and for the
# diagonal-vs-off-diagonal significance check below.
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

SEED = 0

apply_overrides(globals(), load_run_config(__file__))
torch.manual_seed(SEED)

OUTPUT_DIR = resolve_output_dir(
    _REPO_ROOT / "outputs" / "fundamental_abc5" / "scale_composition_query_matching"
)

log.info(
    "RUN_PAIRS=%d units (%s)  |  DINO%s-%s img_size=%d layer=%d  |  "
    "n_scale_steps=%d (7x7=49 ref x query cells per method)",
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

# Depends on SCALE_NAMES, so set here rather than in the Parameters section above — the
# midpoint scale (not global or close, both edge cases) for the qualitative gallery's one
# representative row (see QUALITATIVE_METHOD's comment above).
QUALITATIVE_REF_SCALE: str = SCALE_NAMES[len(SCALE_NAMES) // 2]
log.info("Qualitative gallery representative row: ref_scale=%s", QUALITATIVE_REF_SCALE)


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

# %% Part 2 — build the N_SCALE_STEPS + 1 REFERENCE crops per combo (fg/bg source, same as
# scale_composition_oracle_iou.py), AND the N_SCALE_STEPS + 1 QUERY crops per (part_type, group)
# — the query side only needs one crop set per group (its own GT mask), not per ref instance.
usable_combo_keys: set[tuple] = set()
for combo in tqdm(combos, desc="Building reference scale-step crops"):
    ref_img = ref_images[combo["unit"]]
    group_mask = group_ref_masks.get((combo["unit"], combo["group"]), combo["ref_mask"])
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
query_scale_items: list[tuple] = []  # (unit, group, scale_name, img, gt_mask_px)
for (unit, group), pixel_mask in tqdm(
    group_query_masks.items(), desc="Building query scale-step crops"
):
    query_img = query_images[unit]
    boxes = scale_step_boxes(pixel_mask, T_VALUES, CROP_PADDING_FRACTION)
    close_box = boxes[-1]
    if close_box[2] - close_box[0] < MIN_CROP_SIZE or close_box[3] - close_box[1] < MIN_CROP_SIZE:
        log.warning(
            "unit=%s group=%s: closest query crop %s below MIN_CROP_SIZE=%dpx — skipping",
            unit,
            group,
            close_box,
            MIN_CROP_SIZE,
        )
        continue
    usable_query_groups.add((unit, group))
    for name, box in zip(SCALE_NAMES, boxes):
        x0, y0, x1, y1 = box
        query_scale_items.append((unit, group, name, query_img.crop(box), pixel_mask[y0:y1, x0:x1]))
log.info(
    "Groups with every query scale step usable: %d/%d",
    len(usable_query_groups),
    len(group_query_masks),
)

# Lookup of each query scale-step's own PIL crop, keyed the same way as query_scale_items below
# — needed by the qualitative gallery (Part 9) to show the actual query crop a raw score map and
# GT mask came from, not just an average.
query_scale_images: dict[tuple[str, str, str], Image.Image] = {
    (unit, group, name): img for unit, group, name, img, _gt_mask_px in query_scale_items
}

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

latency_rows: list[dict] = []
with cuda_timer() as t_ref_crop_encode:
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
                IMG_SIZE,
                MASK_PATCH_THRESHOLD,
                bg_exclude_mask_px=bg_exclude_mask_px,
            )
            # Kept on CPU: with REF_QUERY_PAIRS spanning 16 ref/query pairs, every combo's
            # every-scale fg/bg bank held on GPU simultaneously no longer fits (abc3-only fit in
            # ~12GB, the combined pool doesn't) — moved back to the query's device per combo in
            # Part 6.
            fg_by_scale[(ck, name)] = fg.cpu()
            bg_by_scale[(ck, name)] = bg.cpu()
latency_rows.append(
    {
        "phase": "reference_crop_encode",
        "elapsed_s": t_ref_crop_encode["elapsed_s"],
        "n_units": len(clean_items),
        "units_per_sec": images_per_sec(len(clean_items), t_ref_crop_encode["elapsed_s"]),
    }
)

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

with cuda_timer() as t_query_crop_encode:
    for i in tqdm(range(0, len(query_scale_items), chunk_size), desc="Encoding query crops"):
        chunk = query_scale_items[i : i + chunk_size]
        out = encoder([c[3] for c in chunk], layers=[LAYER_IDX], debias=True)
        chunk_patches = out.patches[:, 0]
        grid_h, grid_w = chunk_patches.shape[1], chunk_patches.shape[2]
        for (unit, group, name, _, gt_mask_px), patch_tokens in zip(chunk, chunk_patches):
            tokens = F.normalize(patch_tokens.reshape(grid_h * grid_w, -1), p=2, dim=-1)
            key = (unit, group, name)
            query_scale_encodings[key] = (tokens, grid_h, grid_w)
            query_scale_gt[key] = pixel_mask_to_patch_mask(
                gt_mask_px, grid_h, grid_w, IMG_SIZE, MASK_PATCH_THRESHOLD
            )
latency_rows.append(
    {
        "phase": "query_crop_encode",
        "elapsed_s": t_query_crop_encode["elapsed_s"],
        "n_units": len(query_scale_items),
        "units_per_sec": images_per_sec(len(query_scale_items), t_query_crop_encode["elapsed_s"]),
    }
)
log.info("Built per-scale query crop tokens for %d groups", len(usable_query_groups))
_QUERY_DEVICE = next(iter(query_scale_encodings.values()))[0].device

# %% Part 6 — main scoring: every (ref scale, query scale) pair, 7x7=49 cells per method
# matrix_iou[method][t_ref][t_query][ck] -> oracle IoU (local, GT-cropped-query regime)
MatrixIou = dict[str, dict[str, dict[str, dict[tuple, float]]]]
matrix_iou: MatrixIou = {
    method: {tr: {tq: {} for tq in SCALE_NAMES} for tr in SCALE_NAMES} for method in METHODS
}
# matrix_achievable_iou mirrors matrix_iou's shape — achievable (non-oracle) IoU instead of the
# oracle upper bound; see the ACHIEVABLE-IOU REFERENCE CHOICE comment inside the loop below.
matrix_achievable_iou: MatrixIou = {
    method: {tr: {tq: {} for tq in SCALE_NAMES} for tr in SCALE_NAMES} for method in METHODS
}
# Per-(method, ref_scale, query_scale, combo) sample rows — needed for the object-size
# correlation in Part 7c below, which needs each cell's individual gt_area_frac/oracle_iou
# pairs, not just the per-cell mean matrix_iou already collapses them to.
matrix_sample_rows: list[dict] = []
# One representative row's worth of ScoredExamples for the qualitative gallery (Part 9).
qualitative_examples: list[ScoredExample] = []

n_score_evals = 0
with cuda_timer() as t_scoring:
    for combo in tqdm(combos, desc="Part 6: scoring ref x query scale matrix"):
        ck = combo_key(combo)
        if ck not in usable_combo_keys:
            continue
        unit, group = ck[0], ck[1]
        if (unit, group) not in usable_query_groups:
            continue
        bg_bank = bg_all_lookup[ck].to(_QUERY_DEVICE)

        for t_ref in SCALE_NAMES:
            fg_bank = fg_by_scale[(ck, t_ref)].to(_QUERY_DEVICE)
            proto = compute_exemplar_features(fg_bank, mode="mean")

            # ACHIEVABLE-IOU REFERENCE CHOICE: this script has no train/eval split (every combo
            # is one fixed ref/query pair, scored at every scale) and no separate "own GT" image
            # beyond the query crops already being scored here — so the natural reference for
            # achievable_iou is this row's own diagonal cell (t_query == t_ref, the query crop
            # cropped to the *same* scale as this row's reference gallery). It's the one
            # query-side GT this ref-scale gallery could plausibly be threshold-tuned against
            # without leaking any *other* cell's own GT. Scored once per (combo, t_ref) here and
            # reused as achievable_iou's reference for every t_query in the row below — including
            # the diagonal cell itself, where achievable_iou trivially equals oracle_iou (the
            # threshold is tuned on the same raw score map/GT it's then applied to).
            ref_q_tokens, ref_q_h, ref_q_w = query_scale_encodings[(unit, group, t_ref)]
            ref_gt = query_scale_gt[(unit, group, t_ref)]
            ref_raw_proto = score_heatmap(ref_q_tokens, proto, ref_q_h, ref_q_w)
            ref_raw_knn = knn_score_heatmap(
                ref_q_tokens, fg_bank, bg_bank, KNN_FGBG_NUM_NEIGHBOURS, ref_q_h, ref_q_w
            )

            for t_query in SCALE_NAMES:
                if t_query == t_ref:
                    # Reuse the row's own diagonal scoring above instead of recomputing it.
                    raw_proto, raw_knn, gt_local = ref_raw_proto, ref_raw_knn, ref_gt
                else:
                    q_tokens, q_h, q_w = query_scale_encodings[(unit, group, t_query)]
                    gt_local = query_scale_gt[(unit, group, t_query)]
                    raw_proto = score_heatmap(q_tokens, proto, q_h, q_w)
                    raw_knn = knn_score_heatmap(
                        q_tokens, fg_bank, bg_bank, KNN_FGBG_NUM_NEIGHBOURS, q_h, q_w
                    )
                n_score_evals += 1
                gt_area_frac = float(gt_local.sum()) / gt_local.size

                oi_proto = oracle_iou(raw_proto, gt_local, ORACLE_THRESHOLD_STEPS)
                ai_proto = achievable_iou(
                    ref_raw_proto, ref_gt, raw_proto, gt_local, ORACLE_THRESHOLD_STEPS
                )
                matrix_iou["single_proto"][t_ref][t_query][ck] = oi_proto
                matrix_achievable_iou["single_proto"][t_ref][t_query][ck] = ai_proto

                oi_knn = oracle_iou(raw_knn, gt_local, ORACLE_THRESHOLD_STEPS)
                ai_knn = achievable_iou(
                    ref_raw_knn, ref_gt, raw_knn, gt_local, ORACLE_THRESHOLD_STEPS
                )
                matrix_iou["knn_fgbg"][t_ref][t_query][ck] = oi_knn
                matrix_achievable_iou["knn_fgbg"][t_ref][t_query][ck] = ai_knn

                for method, oi, ai in (
                    ("single_proto", oi_proto, ai_proto),
                    ("knn_fgbg", oi_knn, ai_knn),
                ):
                    matrix_sample_rows.append(
                        {
                            "method": method,
                            "ref_scale": t_ref,
                            "query_scale": t_query,
                            "unit": unit,
                            "group": group,
                            "class": combo["class"],
                            "instance_id": combo["instance_id"],
                            "oracle_iou": oi,
                            "achievable_iou": ai,
                            "gt_area_frac": gt_area_frac,
                        }
                    )

                if (
                    t_ref == QUALITATIVE_REF_SCALE
                    and len(qualitative_examples) < QUALITATIVE_MAX_EXAMPLES
                ):
                    q_raw = raw_proto if QUALITATIVE_METHOD == "single_proto" else raw_knn
                    q_oi = oi_proto if QUALITATIVE_METHOD == "single_proto" else oi_knn
                    qualitative_examples.append(
                        ScoredExample(
                            label=f"{unit}/{group}/{combo['class']}#{combo['instance_id']}/"
                            f"tref={t_ref}/tquery={t_query}",
                            image=query_scale_images[(unit, group, t_query)],
                            raw=q_raw,
                            gt=gt_local,
                            score=q_oi,
                        )
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

# %% Part 6b — latency/throughput: every phase above (reference-crop encode, query-crop encode,
# scoring) traded off against wall-clock cost, which no figure in this script reported before
# now. `torch.cuda.synchronize()` is called around every timed block (see `_shared/latency.py`)
# so GPU-async dispatch doesn't understate elapsed time. Like scale_composition_adaptive_oracle.py
# (its own Part 5b), this script has no per-point sweep loop that re-runs each phase at multiple
# configs — every combo/scale pair is encoded and scored once, in one pass — so there's no sweep
# axis to plot latency against, and no latency.png/accuracy_vs_latency.png; a per-phase log +
# latency.csv is enough.
latency_rows.append(
    {
        "phase": "scoring",
        "elapsed_s": t_scoring["elapsed_s"],
        "n_units": n_score_evals,
        "units_per_sec": images_per_sec(n_score_evals, t_scoring["elapsed_s"]),
    }
)
cache_hits, cache_misses = encoder.total_hits, encoder.total_misses
cache_total = cache_hits + cache_misses
latency_rows.append(
    {
        "phase": "total",
        "elapsed_s": (
            t_ref_crop_encode["elapsed_s"]
            + t_query_crop_encode["elapsed_s"]
            + t_scoring["elapsed_s"]
        ),
        "n_units": len(clean_items) + len(query_scale_items) + n_score_evals,
        "units_per_sec": float("nan"),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": cache_hits / cache_total if cache_total > 0 else float("nan"),
    }
)
latency_df = pd.DataFrame(latency_rows)
latency_df.to_csv(OUTPUT_DIR / "latency.csv", index=False)
log.info("Latency by phase:")
for _, row in latency_df.iterrows():
    log.info("  phase=%-24s elapsed=%.1fs n_units=%d", row.phase, row.elapsed_s, row.n_units)
total_row = latency_df[latency_df.phase == "total"].iloc[0]
log.info(
    "  cache_hits=%d cache_misses=%d cache_hit_rate=%.2f",
    total_row.cache_hits,
    total_row.cache_misses,
    total_row.cache_hit_rate,
)
log.info("Wrote %s", OUTPUT_DIR / "latency.csv")

# %% Part 7 — heatmaps + diagonal-vs-row-best summary
def mean_iou(lookup: dict[tuple, float]) -> tuple[float, int]:
    vals = list(lookup.values())
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


def cell_stats(lookup: dict[tuple, float]) -> tuple[float, float, float]:
    """(std, ci95_lo, ci95_hi) via a percentile bootstrap on one matrix cell's per-combo
    values (see _shared/stats.py) — a plain mean/n (mean_iou above) doesn't say whether two
    cells' means are actually distinguishable from combo-to-combo resampling noise."""
    vals = np.array(list(lookup.values()), dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    _, ci_lo, ci_hi = bootstrap_ci(vals, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
    return float(vals.std()), ci_lo, ci_hi


matrix_rows = []
for method in METHODS:
    for t_ref in SCALE_NAMES:
        for t_query in SCALE_NAMES:
            m, n = mean_iou(matrix_iou[method][t_ref][t_query])
            std_iou, ci_lo, ci_hi = cell_stats(matrix_iou[method][t_ref][t_query])
            am, _an = mean_iou(matrix_achievable_iou[method][t_ref][t_query])
            std_achievable_iou, _ai_lo, _ai_hi = cell_stats(
                matrix_achievable_iou[method][t_ref][t_query]
            )
            matrix_rows.append(
                {
                    "method": method,
                    "ref_scale": t_ref,
                    "query_scale": t_query,
                    "mean_iou": m,
                    "n_combos": n,
                    "std_iou": std_iou,
                    "ci95_lo": ci_lo,
                    "ci95_hi": ci_hi,
                    "mean_achievable_iou": am,
                    "std_achievable_iou": std_achievable_iou,
                    "oracle_minus_achievable_gap": (
                        float(m - am) if not (np.isnan(m) or np.isnan(am)) else float("nan")
                    ),
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

# %% Part 7b — oracle (upper bound, tunes threshold against the query's own GT) vs. achievable
# (threshold tuned on this row's own diagonal cell, transferred as-is — see the ACHIEVABLE-IOU
# REFERENCE CHOICE comment in Part 6) IoU, one line per reference scale, faceted by method — the
# same facet as `ref_query_matrix_heatmap.png` above. Where a t_ref line's oracle (solid) and
# achievable (dashed, same color) markers coincide is exactly the diagonal cell (t_query ==
# t_ref) — the achievable-IoU reference itself, where the two are tautologically equal; the gap
# that opens up moving away from that point along the row is the actual answer to "how much of
# the oracle upper bound would a scale-mismatched query actually get with a threshold tuned at
# the matched scale."
ref_scale_colors = plt.cm.viridis(np.linspace(0, 1, len(SCALE_NAMES)))
fig, axes = plt.subplots(1, len(METHODS), figsize=(7.5 * len(METHODS), 6), sharey=True)
for ax, method in zip(axes, METHODS):
    for t_ref, color in zip(SCALE_NAMES, ref_scale_colors):
        oracle_line = [mean_iou(matrix_iou[method][t_ref][tq])[0] for tq in SCALE_NAMES]
        achievable_line = [
            mean_iou(matrix_achievable_iou[method][t_ref][tq])[0] for tq in SCALE_NAMES
        ]
        ax.plot(SCALE_NAMES, oracle_line, marker="o", linestyle="-", color=color, alpha=0.85)
        ax.plot(SCALE_NAMES, achievable_line, marker="^", linestyle="--", color=color, alpha=0.85)
    ax.set_xticks(range(len(SCALE_NAMES)), SCALE_NAMES, rotation=45)
    ax.set_xlabel("query crop scale")
    ax.set_title(method)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("mean IoU across combos (solid=oracle, dashed=achievable; color=ref scale)")
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=len(SCALE_NAMES) - 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02, ticks=range(len(SCALE_NAMES)))
cbar.ax.set_yticklabels(SCALE_NAMES)
cbar.set_label("reference scale")
fig.suptitle("Oracle vs. achievable IoU across the ref x query matrix (one line per reference scale)")
fig.savefig(OUTPUT_DIR / "oracle_vs_achievable.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved %s", OUTPUT_DIR / "oracle_vs_achievable.png")

# %% Part 7c — does oracle IoU correlate with the query GT's own object size (gt_area_frac,
# added to every matrix_sample_rows row in Part 6)? Mirrors scale_composition_adaptive_oracle.py's
# own Part 7 correlation_rows/pearson_r/pearson_p/spearman_r/spearman_p convention (that script
# set this precedent first), grouped by (method, ref_scale, query_scale) — the same granularity
# as this script's own headline matrix_rows breakdown above. gt_area_frac genuinely varies
# combo-to-combo within one cell (different (unit, group) pairs crop to different absolute
# object sizes) even though every combo in a cell shares the same query crop scale.
matrix_sample_df = pd.DataFrame(matrix_sample_rows)
size_correlation_rows = []
for method in METHODS:
    for t_ref in SCALE_NAMES:
        for t_query in SCALE_NAMES:
            sub = matrix_sample_df[
                (matrix_sample_df.method == method)
                & (matrix_sample_df.ref_scale == t_ref)
                & (matrix_sample_df.query_scale == t_query)
            ]
            if len(sub) < 3:
                continue
            pearson_r, pearson_p = pearsonr(sub["gt_area_frac"], sub["oracle_iou"])
            spearman_r, spearman_p = spearmanr(sub["gt_area_frac"], sub["oracle_iou"])
            size_correlation_rows.append(
                {
                    "method": method,
                    "ref_scale": t_ref,
                    "query_scale": t_query,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "n_samples": len(sub),
                }
            )
size_correlation_df = pd.DataFrame(size_correlation_rows)
size_correlation_df.to_csv(OUTPUT_DIR / "size_correlation.csv", index=False)
log.info("Wrote %s (%d rows)", OUTPUT_DIR / "size_correlation.csv", len(size_correlation_df))

# Object-size terciles (global cutoffs, computed once across every sample row so they're
# consistent everywhere) x oracle IoU, faceted by method — same facet as
# `ref_query_matrix_heatmap.png` above, pooled across every (ref_scale, query_scale) cell.
try:
    matrix_sample_df["size_tercile"] = pd.qcut(
        matrix_sample_df["gt_area_frac"], 3, labels=["small", "medium", "large"]
    )
except ValueError:
    log.warning(
        "gt_area_frac has too few distinct values for 3 clean terciles — falling back to "
        "qcut's own duplicate-safe binning (labels become numeric ranges, not small/medium/large)"
    )
    matrix_sample_df["size_tercile"] = pd.qcut(matrix_sample_df["gt_area_frac"], 3, duplicates="drop")

fig, axes = plt.subplots(1, len(METHODS), figsize=(6.5 * len(METHODS), 5.5), sharey=True)
for ax, method in zip(axes, METHODS):
    tercile_means = (
        matrix_sample_df[matrix_sample_df.method == method]
        .groupby("size_tercile", observed=True)["oracle_iou"]
        .mean()
    )
    tercile_means.plot(kind="bar", ax=ax, color=METHOD_COLOR[method])
    ax.set_title(method)
    ax.set_xlabel("object-size tercile (by query GT patch-mask area fraction)")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("mean oracle IoU (pooled across every ref/query scale cell)")
fig.suptitle("Does object size predict oracle IoU in the ref x query scale matrix?")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "size_correlation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Wrote %s", OUTPUT_DIR / "size_correlation.png")

# %% Part 8 — is the "matched scale beats mismatched scale" pattern real, or combo-to-combo
# noise? An unpaired bootstrap comparison (see _shared/stats.py) of the matrix's two natural
# groups of cells, per method: every diagonal (matched ref/query scale) cell's oracle_iou values
# pooled across ref scales vs. every off-diagonal (mismatched) cell's. This is the direct
# significance check `diagonal_vs_row_best.csv`'s `diagonal_is_row_best` column above (Part 7)
# otherwise leaves the reader to eyeball as a plain win-count.
significance_rows = []
for method in METHODS:
    diagonal_vals = matrix_sample_df.loc[
        (matrix_sample_df.method == method)
        & (matrix_sample_df.ref_scale == matrix_sample_df.query_scale),
        "oracle_iou",
    ].to_numpy()
    off_diagonal_vals = matrix_sample_df.loc[
        (matrix_sample_df.method == method)
        & (matrix_sample_df.ref_scale != matrix_sample_df.query_scale),
        "oracle_iou",
    ].to_numpy()
    prob_diag_greater = bootstrap_prob_greater(
        diagonal_vals, off_diagonal_vals, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
    )
    significance_rows.append(
        {
            "method": method,
            "prob_diagonal_beats_off_diagonal": prob_diag_greater,
            "n_diagonal": len(diagonal_vals),
            "n_off_diagonal": len(off_diagonal_vals),
        }
    )
significance_df = pd.DataFrame(significance_rows)
significance_df.to_csv(OUTPUT_DIR / "matrix_diagonal_significance.csv", index=False)
log.info(
    "Matched-vs-mismatched scale significance (P(diagonal mean > off-diagonal mean) under "
    "%d-resample bootstrap; near 0.5 = indistinguishable from noise):",
    N_BOOTSTRAP,
)
for _, row in significance_df.iterrows():
    log.info(
        "  method=%-13s P(diagonal beats off-diagonal)=%.3f (n=%d vs n=%d)",
        row.method,
        row.prob_diagonal_beats_off_diagonal,
        row.n_diagonal,
        row.n_off_diagonal,
    )
log.info("Wrote %s", OUTPUT_DIR / "matrix_diagonal_significance.csv")

# %% Part 9 — worst/best-N qualitative gallery at one representative row (see
# QUALITATIVE_REF_SCALE/QUALITATIVE_METHOD above) — every other figure in this script averages
# across combos (heatmap cells, tercile bars); this shows actual individual query crops so a
# failure mode (one orientation, one lighting condition) is visible instead of washed out by
# the mean.
if qualitative_examples:
    save_score_gallery(
        qualitative_examples,
        OUTPUT_DIR / "qualitative_worst_best.png",
        n=5,
        score_name="oracle_iou",
        title=(
            f"Worst/best oracle_iou examples: ref_scale={QUALITATIVE_REF_SCALE} "
            f"method={QUALITATIVE_METHOD} (across every query scale)"
        ),
    )
    log.info(
        "Wrote %s (%d examples)",
        OUTPUT_DIR / "qualitative_worst_best.png",
        len(qualitative_examples),
    )
else:
    log.warning(
        "No qualitative examples collected for the representative row (ref_scale=%s, method=%s)",
        QUALITATIVE_REF_SCALE,
        QUALITATIVE_METHOD,
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
# - **`ref_query_matrix_iou.csv`'s `mean_achievable_iou`/`std_achievable_iou`/
#   `oracle_minus_achievable_gap` columns**, and **`oracle_vs_achievable.png`** — `mean_iou`
#   everywhere else is an oracle upper bound (tunes its threshold against the query's own GT);
#   `achievable_iou` tunes the threshold on this row's own diagonal (matched-scale) cell only and
#   transfers it as-is to every other cell in the row — the number a deployed pipeline without
#   query-time labels would actually see (see the ACHIEVABLE-IOU REFERENCE CHOICE comment in
#   Part 6 for why the diagonal was chosen as the reference). Achievable == oracle exactly on the
#   diagonal itself (tautologically, since the threshold is tuned on that very cell); the gap
#   that opens up moving away from the diagonal is the real cost of not having query-time labels.
#   `ref_query_matrix_iou.csv`'s `ci95_lo`/`ci95_hi` columns are a percentile bootstrap CI (2000
#   resamples) on each cell's mean.
# - **`size_correlation.csv`/`.png`** — does oracle IoU correlate with the query GT's own area
#   fraction (`gt_area_frac`), per (method, ref_scale, query_scale) cell? A resolution here that
#   `size_vs_optimal_t_correlation.csv` (a *different* script, `scale_composition_adaptive_
#   oracle.py`) doesn't test: whether small/large query crops are simply harder or easier to
#   score at any given ref/query scale pairing, not which scale is optimal for them.
# - **`matrix_diagonal_significance.csv`** — an unpaired bootstrap comparison (2000 resamples) of
#   every diagonal cell's oracle_iou values (pooled across ref scales) vs. every off-diagonal
#   cell's, per method: `prob_diagonal_beats_off_diagonal` near 0.5 means the apparent
#   matched-scale advantage in `ref_query_matrix_heatmap.png`/`diagonal_vs_row_best.csv` is not
#   distinguishable from combo-to-combo noise — the quantitative version of that CSV's win-count.
# - **`latency.csv`** — GPU-synchronized wall-clock cost (see `_shared/latency.py`) per phase
#   (reference-crop encode, query-crop encode, scoring) plus encoding-cache hit rate. No sweep
#   axis exists in this script to plot latency against (every combo/scale pair is encoded and
#   scored once, not per-config like `resolution_ablation.py`'s sweep) — see the per-phase log
#   lines instead.
# - **`qualitative_worst_best.png`** — actual worst-5/best-5 query crops (crop, raw score map,
#   GT mask) at one representative reference scale (`QUALITATIVE_REF_SCALE`, the sweep's
#   midpoint) across every query scale, not an average — no other figure here shows *why* a
#   specific (ref, query) scale pairing fails on a specific instance.

# %%
