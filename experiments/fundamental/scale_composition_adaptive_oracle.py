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
from dotenv import load_dotenv
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from dinoisawesome import DinoEncoder, EncoderWithCache, compute_exemplar_features, load_annotations
from dinoisawesome.abc3 import INSTANCE_TYPE_GROUPS, available_instance_groups
from dinoisawesome.instance_detection import extract_patch_tokens

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

# One representative (scale, method) point whose individual query samples get kept as PIL
# images + raw score maps for the worst/best-N qualitative gallery — collecting this for every
# scale would multiply memory/disk cost, so only the midpoint scale (the one place both a
# global-context confound and a close-crop noise confound are least likely to dominate) and the
# stronger knn_fgbg method are captured.
QUALITATIVE_SCALE = "mid"
QUALITATIVE_METHOD = "knn_fgbg"
QUALITATIVE_MAX_EXAMPLES = 60  # capped so the gallery figure itself stays a readable size

# Bootstrap settings for the achievable-IoU CI added to Part 6's headline summary and for the
# global-vs-close scale-effect significance check in Part 8.
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

SEED = 0

apply_overrides(globals(), load_run_config(__file__))
torch.manual_seed(SEED)

OUTPUT_DIR = resolve_output_dir(
    _REPO_ROOT / "outputs" / "fundamental_abc5" / "scale_composition_adaptive_oracle"
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

# %% Part 2 — build the N_SCALE_STEPS + 1 crops per combo, plus each instance's own bbox-area
# fraction of the full reference image (for the size-vs-optimal-t correlation).
usable_combo_keys: set[tuple] = set()
instance_size_fraction: dict[tuple, float] = {}
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
    ck = combo_key(combo)
    usable_combo_keys.add(ck)
    H, W = combo["ref_mask"].shape
    instance_size_fraction[ck] = float(combo["ref_mask"].sum()) / float(H * W)

log.info("Combos with every scale step usable: %d/%d", len(usable_combo_keys), len(combos))
combo_by_key: dict[tuple, dict] = {
    combo_key(c): c for c in combos if combo_key(c) in usable_combo_keys
}

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

latency_rows: list[dict] = []

query_encodings: dict[str, tuple[torch.Tensor, int, int]] = {}
with cuda_timer() as t_query_encode:
    for unit in tqdm(sorted(query_images), desc="Encoding query images"):
        tokens, q_h, q_w = extract_patch_tokens(encoder, query_images[unit], LAYER_IDX, debias=True)
        query_encodings[unit] = (tokens, q_h, q_w)
latency_rows.append(
    {
        "phase": "query_full_image_encode",
        "elapsed_s": t_query_encode["elapsed_s"],
        "n_units": len(query_images),
        "units_per_sec": images_per_sec(len(query_images), t_query_encode["elapsed_s"]),
    }
)

gt_patch_masks: dict[tuple[str, str], np.ndarray] = {}
for (unit, group), pixel_mask in group_query_masks.items():
    _, q_h, q_w = query_encodings[unit]
    gt_patch_masks[(unit, group)] = pixel_mask_to_patch_mask(
        pixel_mask, q_h, q_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )

# Achievable IoU (see _shared.thresholding.achievable_iou) needs a "reference" raw score map +
# its own GT that the query being scored never sees. This script has no train/eval split (every
# combo is one fixed ref/query pair) — the natural reference is the exemplar/ref image itself:
# its own `ref_mask` is the same GT that built the gallery, so a threshold can be tuned on it and
# transferred to the query, mirroring resolution_ablation.py's `pick_ref_number`/`ref_raws`
# pattern for its own train/eval split. Whole ref images are encoded here (once per unit,
# alongside the query encodings above) so Part 5 below can score each gallery against its own
# ref image the same way it scores the query.
ref_encodings: dict[str, tuple[torch.Tensor, int, int]] = {}
with cuda_timer() as t_ref_encode:
    for unit in tqdm(
        sorted(ref_images), desc="Encoding reference images (whole, for achievable_iou)"
    ):
        tokens, r_h, r_w = extract_patch_tokens(encoder, ref_images[unit], LAYER_IDX, debias=True)
        ref_encodings[unit] = (tokens, r_h, r_w)
latency_rows.append(
    {
        "phase": "ref_full_image_encode",
        "elapsed_s": t_ref_encode["elapsed_s"],
        "n_units": len(ref_images),
        "units_per_sec": images_per_sec(len(ref_images), t_ref_encode["elapsed_s"]),
    }
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

bg_all_lookup: dict[tuple, torch.Tensor] = {
    ck: torch.cat([bg_by_scale[(ck, name)] for name in SCALE_NAMES], dim=0)
    for ck in usable_combo_keys
}
log.info("Built per-scale fg/bg galleries for %d combos", len(usable_combo_keys))

# Each combo's ref-image GT reprojected onto its own whole-ref-image patch grid (Part 3's
# ref_encodings) — the "own GT" achievable_iou tunes its threshold against in Part 5 below.
ref_gt_patch_masks: dict[tuple, np.ndarray] = {}
for ck in usable_combo_keys:
    unit = ck[0]
    _, r_h, r_w = ref_encodings[unit]
    ref_gt_patch_masks[ck] = pixel_mask_to_patch_mask(
        combo_by_key[ck]["ref_mask"], r_h, r_w, IMG_SIZE, MASK_PATCH_THRESHOLD
    )

# %% Part 5 — per-combo, per-scale raw IoU (kept per-instance, not just aggregated — needed for
# both the adaptive oracle and the size-correlation below).
# per_instance_iou[method][scale][ck] -> oracle IoU
PerInstanceIou = dict[str, dict[str, dict[tuple, float]]]
per_instance_iou: PerInstanceIou = {
    method: {name: {} for name in SCALE_NAMES} for method in METHODS
}
# per_instance_achievable_iou[method][scale][ck] -> achievable IoU (threshold tuned on the
# combo's own ref image, transferred as-is to the query — see achievable_iou's docstring).
per_instance_achievable_iou: PerInstanceIou = {
    method: {name: {} for name in SCALE_NAMES} for method in METHODS
}

qualitative_examples: list[ScoredExample] = []
n_score_evals = 0
with cuda_timer() as t_scoring:
    for combo in tqdm(combos, desc="Part 5: scoring per scale"):
        ck = combo_key(combo)
        if ck not in usable_combo_keys:
            continue
        unit, group = ck[0], ck[1]
        gt = gt_patch_masks.get((unit, group))
        if gt is None:
            continue
        q_tokens, q_h, q_w = query_encodings[unit]
        bg_bank = bg_all_lookup[ck].to(q_tokens.device)
        ref_tokens, r_h, r_w = ref_encodings[unit]
        ref_gt = ref_gt_patch_masks[ck]

        for name in SCALE_NAMES:
            fg_bank = fg_by_scale[(ck, name)].to(q_tokens.device)

            proto = compute_exemplar_features(fg_bank, mode="mean")
            raw_proto = score_heatmap(q_tokens, proto, q_h, q_w)
            oi_proto = oracle_iou(raw_proto, gt, ORACLE_THRESHOLD_STEPS)
            per_instance_iou["single_proto"][name][ck] = oi_proto
            # One extra score_heatmap call per (combo, scale) — the same gallery/prototype this
            # cell just scored the query with, applied to the combo's own ref image instead.
            ref_raw_proto = score_heatmap(ref_tokens, proto, r_h, r_w)
            ai_proto = achievable_iou(
                ref_raw_proto, ref_gt, raw_proto, gt, ORACLE_THRESHOLD_STEPS
            )
            per_instance_achievable_iou["single_proto"][name][ck] = ai_proto

            raw_knn = knn_score_heatmap(q_tokens, fg_bank, bg_bank, KNN_FGBG_NUM_NEIGHBOURS, q_h, q_w)
            oi_knn = oracle_iou(raw_knn, gt, ORACLE_THRESHOLD_STEPS)
            per_instance_iou["knn_fgbg"][name][ck] = oi_knn
            ref_raw_knn = knn_score_heatmap(
                ref_tokens, fg_bank, bg_bank, KNN_FGBG_NUM_NEIGHBOURS, r_h, r_w
            )
            ai_knn = achievable_iou(ref_raw_knn, ref_gt, raw_knn, gt, ORACLE_THRESHOLD_STEPS)
            per_instance_achievable_iou["knn_fgbg"][name][ck] = ai_knn

            n_score_evals += 1
            if name == QUALITATIVE_SCALE and len(qualitative_examples) < QUALITATIVE_MAX_EXAMPLES:
                raw_by_method = {"single_proto": (raw_proto, oi_proto), "knn_fgbg": (raw_knn, oi_knn)}
                q_raw, q_oi = raw_by_method[QUALITATIVE_METHOD]
                qualitative_examples.append(
                    ScoredExample(
                        label=f"{unit}/{group}/{combo['class']}#{combo['instance_id']}",
                        image=query_images[unit],
                        raw=q_raw,
                        gt=gt,
                        score=q_oi,
                    )
                )
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
            t_query_encode["elapsed_s"]
            + t_ref_encode["elapsed_s"]
            + t_crop_encode["elapsed_s"]
            + t_scoring["elapsed_s"]
        ),
        "n_units": len(query_images) + len(ref_images) + len(clean_items) + n_score_evals,
        "units_per_sec": float("nan"),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": cache_hits / cache_total if cache_total > 0 else float("nan"),
    }
)

log.info(
    "Scoring complete: %d combos x %d scales x %d methods",
    len(usable_combo_keys),
    len(SCALE_NAMES),
    len(METHODS),
)

# %% Part 5b — latency/throughput: every phase above (query/ref whole-image encode, gallery-crop
# encode, scoring) traded off against wall-clock cost, which no figure in this script reported
# before now. `torch.cuda.synchronize()` is called around every timed block (see
# `_shared/latency.py`) so GPU-async dispatch doesn't understate elapsed time. Unlike
# resolution_ablation.py/training_set_size_ablation.py, this script has no per-point sweep loop
# that re-runs each phase at multiple configs (every combo/scale is encoded and scored once, in
# one pass) — so there's no sweep axis to plot latency against, and no accuracy_vs_latency.png;
# a per-phase log + latency.csv is enough.
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

# %% Part 6 — adaptive-scale oracle: per-instance max over scales vs. dataset-wide fixed-best-scale
per_instance_rows = []
for ck in sorted(usable_combo_keys):
    row = {
        "unit": ck[0],
        "dataset": combo_by_key[ck]["dataset"],
        "part_type": combo_by_key[ck]["part_type"],
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
    # Percentile bootstrap CI on both headline means, alongside the plain values this script
    # already reported — a CI says whether fixed-best's and adaptive-oracle's means are actually
    # distinguishable or both plausible draws from the same underlying per-instance distribution
    # (see _shared/stats.py).
    _, fixed_best_ci_lo, fixed_best_ci_hi = bootstrap_ci(
        np.array(list(per_instance_iou[method][fixed_best_scale].values())),
        n_boot=N_BOOTSTRAP,
        seed=BOOTSTRAP_SEED,
    )
    _, adaptive_ci_lo, adaptive_ci_hi = bootstrap_ci(
        per_instance_df[f"{method}_adaptive_iou"].to_numpy(), n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
    )
    # Achievable IoU (see _shared.thresholding.achievable_iou / Part 5 above) at this method's
    # fixed-best scale: `oracle_iou` everywhere else in this script is an upper bound (tunes its
    # threshold against the query's own GT); achievable_iou tunes on the combo's own ref image
    # only and transfers the threshold as-is — the number a deployed pipeline without query-time
    # labels would actually see at that scale.
    achievable_vals = np.array(list(per_instance_achievable_iou[method][fixed_best_scale].values()))
    mean_achievable_iou = float(achievable_vals.mean()) if len(achievable_vals) else float("nan")
    std_achievable_iou = float(achievable_vals.std()) if len(achievable_vals) else float("nan")
    adaptive_summary_rows.append(
        {
            "method": method,
            "fixed_best_scale": fixed_best_scale,
            "fixed_best_mean_iou": fixed_best_mean,
            "fixed_best_ci95_lo": fixed_best_ci_lo,
            "fixed_best_ci95_hi": fixed_best_ci_hi,
            "adaptive_oracle_mean_iou": adaptive_mean,
            "adaptive_oracle_ci95_lo": adaptive_ci_lo,
            "adaptive_oracle_ci95_hi": adaptive_ci_hi,
            "headroom": adaptive_mean - fixed_best_mean,
            "mean_achievable_iou": mean_achievable_iou,
            "std_achievable_iou": std_achievable_iou,
            "oracle_minus_achievable_gap": fixed_best_mean - mean_achievable_iou,
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

# %% Part 6b — oracle IoU (upper bound, tunes the threshold against the query's own GT) vs.
# achievable IoU (a threshold tuned on the combo's own ref image, transferred as-is to the
# query — what a deployed pipeline without query-time labels would actually get), across every
# reference scale. Every other figure in this script plots oracle_iou only. A single, un-faceted
# axes — the same facet-shape as `adaptive_oracle_summary.png` above (one panel, not one per
# extra sweep dimension), with the reference-scale sweep (SCALE_NAMES/T_VALUES) on the x-axis.
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for method in METHODS:
    oracle_means = [
        float(np.mean(list(per_instance_iou[method][name].values()))) for name in SCALE_NAMES
    ]
    achievable_means = [
        float(np.mean(list(per_instance_achievable_iou[method][name].values())))
        if per_instance_achievable_iou[method][name]
        else float("nan")
        for name in SCALE_NAMES
    ]
    ax.plot(
        T_VALUES, oracle_means, marker="o", linestyle="-",
        label=f"{method} oracle", color=METHOD_COLOR[method],
    )
    ax.plot(
        T_VALUES, achievable_means, marker="^", linestyle="--",
        label=f"{method} achievable", color=METHOD_COLOR[method], alpha=0.6,
    )
ax.set_xticks(T_VALUES, SCALE_NAMES, rotation=45)
ax.set_xlabel("reference crop scale (global -> close)")
ax.set_ylabel("mean IoU across combos")
ax.set_ylim(0, 1.0)
ax.set_title("Oracle (upper bound) vs. achievable (transferred threshold) IoU by reference scale")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "oracle_vs_achievable.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved %s", OUTPUT_DIR / "oracle_vs_achievable.png")

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

# %% Part 7b — object-size correlation, take two: does oracle_iou *itself* (the IoU achieved at
# a given scale, not which scale is optimal) correlate with instance size? Separate from Part 7
# above, which correlates instance size against per-instance *optimal scale* t — this instead
# asks whether small/large instances are simply harder or easier at any given scale. Mirrors
# Part 7's own correlation_rows/pearson_r/pearson_p/spearman_r/spearman_p convention (the
# precedent this script set for every other `fundamental/` sibling), extended with `scale` as
# the grouping key since (unlike Part 7's per-instance best-t) oracle_iou varies by scale too.
# Uses `gt_area_frac` (the *query*-side GT's own patch-mask coverage) rather than Part 7's
# `size_fraction` (the *ref*-side bbox-mask fraction) — the two need not agree instance-to-
# instance (ref and query are different photos of possibly-different-sized instances).
t_by_scale: dict[str, float] = dict(zip(SCALE_NAMES, T_VALUES))
gt_area_frac_by_group: dict[tuple[str, str], float] = {
    (unit, group): float(mask.sum()) / mask.size for (unit, group), mask in gt_patch_masks.items()
}

size_iou_rows = []
for method in METHODS:
    for name in SCALE_NAMES:
        for ck, iou in per_instance_iou[method][name].items():
            unit, group = ck[0], ck[1]
            size_iou_rows.append(
                {
                    "method": method,
                    "scale": name,
                    "t": t_by_scale[name],
                    "unit": unit,
                    "group": group,
                    "class": ck[2],
                    "instance_id": ck[3],
                    "oracle_iou": iou,
                    "gt_area_frac": gt_area_frac_by_group.get((unit, group), float("nan")),
                }
            )
size_iou_df = pd.DataFrame(size_iou_rows)

size_correlation_rows = []
for method in METHODS:
    for name in SCALE_NAMES:
        sub = size_iou_df[(size_iou_df.method == method) & (size_iou_df.scale == name)]
        if len(sub) < 3:
            continue
        pearson_r, pearson_p = pearsonr(sub["gt_area_frac"], sub["oracle_iou"])
        spearman_r, spearman_p = spearmanr(sub["gt_area_frac"], sub["oracle_iou"])
        size_correlation_rows.append(
            {
                "method": method,
                "scale": name,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "n": len(sub),
            }
        )
size_correlation_df = pd.DataFrame(size_correlation_rows)
size_correlation_df.to_csv(OUTPUT_DIR / "size_correlation.csv", index=False)
log.info("Object-size (gt_area_frac) vs. oracle_iou correlation, per (method, scale):")
for _, row in size_correlation_df.iterrows():
    log.info(
        "  %-13s scale=%-8s pearson_r=%.3f (p=%.3f)  spearman_r=%.3f (p=%.3f)  n=%d",
        row.method,
        row.scale,
        row.pearson_r,
        row.pearson_p,
        row.spearman_r,
        row.spearman_p,
        row.n,
    )

# Object-size terciles (global, computed once across every row so the same size cutoffs apply
# everywhere) x oracle IoU, faceted by method (Part 7's own facet convention) and pooled across
# every scale.
try:
    size_iou_df["size_tercile"] = pd.qcut(
        size_iou_df["gt_area_frac"], 3, labels=["small", "medium", "large"]
    )
except ValueError:
    log.warning(
        "gt_area_frac has too few distinct values for 3 clean terciles — falling back to "
        "qcut's own duplicate-safe binning (labels become numeric ranges, not small/medium/large)"
    )
    size_iou_df["size_tercile"] = pd.qcut(size_iou_df["gt_area_frac"], 3, duplicates="drop")

fig, axes = plt.subplots(1, len(METHODS), figsize=(6 * len(METHODS), 5), sharey=True)
for ax, method in zip(axes, METHODS):
    tercile_means = (
        size_iou_df[size_iou_df.method == method]
        .groupby("size_tercile", observed=True)["oracle_iou"]
        .mean()
    )
    tercile_means.plot(kind="bar", ax=ax, color=METHOD_COLOR[method])
    ax.set_title(method)
    ax.set_xlabel("object-size tercile (by query GT patch-mask area fraction)")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("mean oracle IoU (pooled across scales)")
fig.suptitle("Does object size predict oracle IoU? (separate from Part 7's size-vs-optimal-t check)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "size_correlation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(
    "Wrote %s and %s", OUTPUT_DIR / "size_correlation.csv", OUTPUT_DIR / "size_correlation.png"
)

# %% Part 8 — is the global-vs-close scale effect real, or combo-to-combo noise? An unpaired
# bootstrap comparison (see _shared/stats.py) of the two scale-sweep endpoints' per-combo
# oracle_iou arrays — the significance check no figure in this script previously quantified.
significance_rows = []
for method in METHODS:
    lo_vals = np.array(list(per_instance_iou[method][SCALE_NAMES[0]].values()))
    hi_vals = np.array(list(per_instance_iou[method][SCALE_NAMES[-1]].values()))
    prob_hi_greater = bootstrap_prob_greater(hi_vals, lo_vals, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
    significance_rows.append(
        {
            "method": method,
            "scale_lo": SCALE_NAMES[0],
            "scale_hi": SCALE_NAMES[-1],
            "prob_hi_beats_lo": prob_hi_greater,
            "n_lo": len(lo_vals),
            "n_hi": len(hi_vals),
        }
    )
significance_df = pd.DataFrame(significance_rows)
significance_df.to_csv(OUTPUT_DIR / "scale_effect_significance.csv", index=False)
log.info(
    "Scale-effect significance (P(%s mean > %s mean) under %d-resample bootstrap; near 0.5 = "
    "indistinguishable from noise):",
    SCALE_NAMES[-1],
    SCALE_NAMES[0],
    N_BOOTSTRAP,
)
for _, row in significance_df.iterrows():
    log.info(
        "  method=%-13s P(%s beats %s)=%.3f (n=%d vs n=%d)",
        row.method,
        row.scale_hi,
        row.scale_lo,
        row.prob_hi_beats_lo,
        row.n_hi,
        row.n_lo,
    )
log.info("Wrote %s", OUTPUT_DIR / "scale_effect_significance.csv")

# %% Part 9 — worst/best-N qualitative gallery at one representative point (see
# QUALITATIVE_SCALE/QUALITATIVE_METHOD above) — every other figure in this script averages
# across instances; this shows actual individual query images so a failure mode (one
# orientation, one lighting condition) is visible instead of washed out by the mean.
if qualitative_examples:
    save_score_gallery(
        qualitative_examples,
        OUTPUT_DIR / "qualitative_worst_best.png",
        n=5,
        score_name="oracle_iou",
        title=f"Worst/best oracle_iou examples: scale={QUALITATIVE_SCALE} method={QUALITATIVE_METHOD}",
    )
    log.info(
        "Wrote %s (%d examples)",
        OUTPUT_DIR / "qualitative_worst_best.png",
        len(qualitative_examples),
    )
else:
    log.warning(
        "No qualitative examples collected for the representative point (scale=%s, method=%s)",
        QUALITATIVE_SCALE,
        QUALITATIVE_METHOD,
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
# - **`oracle_vs_achievable.png`/`adaptive_oracle_summary.csv`'s `mean_achievable_iou`/
#   `oracle_minus_achievable_gap` columns** — `oracle_iou` everywhere else in this script is an
#   upper bound (tunes its threshold against the query's own GT); achievable_iou tunes the
#   threshold on the combo's own ref image only (the same image that built the gallery) and
#   transfers it as-is — the number a deployed pipeline without query-time labels would actually
#   see. A scale that helps oracle but not achievable IoU is a scale that makes scores more
#   *separable*, not one a fixed real-world threshold can actually exploit.
# - **`adaptive_oracle_summary.csv`'s `fixed_best_ci95_lo`/`_hi` and `adaptive_oracle_ci95_lo`/
#   `_hi` columns** — a percentile bootstrap CI (2000 resamples) on each headline mean; if the
#   two means' CIs overlap heavily, the "+headroom" number is less trustworthy than the point
#   estimate alone suggests.
# - **`latency.csv`** — GPU-synchronized wall-clock cost (see `_shared/latency.py`) per phase
#   (query/ref whole-image encode, gallery-crop encode, scoring) plus encoding-cache hit rate.
#   No sweep axis exists in this script to plot latency against (every combo/scale is encoded
#   and scored once, not per-config like `resolution_ablation.py`'s sweep) — see the per-phase
#   log lines instead.
# - **`size_correlation.csv`/`.png`** — a second, separate object-size check from Part 7's own
#   size-vs-optimal-t correlation: does the IoU actually *achieved* at a given scale (not which
#   scale is best) correlate with the query GT's own area fraction (`gt_area_frac`)? A resolution
#   here that Part 7's correlation misses (or vice versa) means "does size predict which scale
#   wins" and "does size predict how well any scale does" are genuinely different questions.
# - **`scale_effect_significance.csv`** — an unpaired bootstrap comparison (2000 resamples) of
#   the `global` vs. `close` scale endpoints' per-combo oracle_iou arrays: `prob_hi_beats_lo`
#   near 0.5 means the apparent scale effect in `adaptive_oracle_summary.png`/the per-scale means
#   inside it is not distinguishable from combo-to-combo noise.
# - **`qualitative_worst_best.png`** — actual worst-5/best-5 query images (crop, raw score map,
#   GT mask) at one representative (scale, method) point, not an average — no other figure here
#   shows *why* a specific instance fails.

# %%
