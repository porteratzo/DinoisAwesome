# %% [markdown]
# # Fundamental: Scale Composition — Finer Scale Steps + Multi-Scale Fg/Bg Composition
#
# Every sibling script in `experiments/fundamental/` and `object_detection/multiscale_ablation/`
# builds `single_proto`/`fg-bg-knn` galleries from exactly three named crop scales — `close`,
# `mid`, `global` (`_shared.mask_geometry.scale_crop_box`), where `mid` is literally the t=0.5
# midpoint between `close` (t=1) and `global` (t=0). This experiment generalizes that fixed
# 3-point sweep into `N_SCALE_STEPS + 1` evenly-spaced crop scales (t = 0, 1/n, 2/n, ..., 1 —
# same linear interpolation `scale_crop_similarity.py` already uses for its single-instance case
# study, applied here across the whole abc5 dataset for real fg/bg IoU) and asks two questions:
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

# Same focus combo as the sibling fundamental scripts, for direct comparability of figures.
FOCUS_UNIT = "LHa_1-2"
FOCUS_CLASS = "donut foam single"
FOCUS_INSTANCE_ID = 1

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

# n in "global-.../close, n=6" — number of steps from global (t=0) to close (t=1); gives
# N_SCALE_STEPS + 1 scale points. n=6 reproduces today's "mid" exactly at t=3/6=0.5.
N_SCALE_STEPS = 6

METHODS: list[str] = ["single_proto", "knn_fgbg"]
METHOD_COLOR: dict[str, str] = {"single_proto": "#7f8c8d", "knn_fgbg": "#2ecc71"}

# One representative composition (the classic 3-point baseline, already every sibling script's
# own default gallery) whose individual combo score maps get kept as PIL images + raw score maps
# for the worst/best-N qualitative gallery in Part 9b below — collecting this for every
# composition x combo would multiply memory/disk cost by len(COMPOSITION_COMBOS), so only this
# one composition (and only knn_fgbg, the stronger method) is captured.
QUALITATIVE_COMPOSITION = "global+mid+close"
QUALITATIVE_METHOD = "knn_fgbg"
QUALITATIVE_MAX_EXAMPLES = 60  # capped so the gallery figure itself stays a readable size

# Bootstrap settings for the headline CIs (Parts 6/7/10) and the scale-effect significance check
# (Parts 7b/10c): is the global-vs-close scale effect actually distinguishable from combo/fold/
# sample noise, or within it?
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

SEED = 0

apply_overrides(globals(), load_run_config(__file__))
torch.manual_seed(SEED)

OUTPUT_DIR = resolve_output_dir(
    _REPO_ROOT / "outputs" / "fundamental_abc5" / "scale_composition_oracle_iou"
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

# %% Scale-step naming + crop-box geometry
T_VALUES: np.ndarray = np.linspace(0.0, 1.0, N_SCALE_STEPS + 1)

SCALE_NAMES: list[str] = [scale_step_name(i, N_SCALE_STEPS) for i in range(N_SCALE_STEPS + 1)]
SCALE_COLOR: dict[str, str] = {
    name: plt.get_cmap("viridis")(t) for name, t in zip(SCALE_NAMES, T_VALUES)
}
log.info("Scale steps (global -> close): %s", SCALE_NAMES)

# %% Composition combo table — not the full power set (see module docstring), but more than
# just the two open-ended growth sweeps: also the classic 3-point `global+mid+close` baseline
# (the fixed combo every sibling script's FGBG_SOURCE_COMBOS already uses), and two "anchored"
# growth sweeps that keep BOTH endpoints in every entry and grow the middle from one side —
# "anchored_inward" adds middle scales moving away from global (mirrors PREFIX_NAMES but never
# drops `close`), "anchored_outward" adds them moving away from close (mirrors SUFFIX_NAMES but
# never drops `global`). ~4n+3 combos instead of 2^(n+1) - 1.
COMPOSITION_COMBOS, PREFIX_NAMES, SUFFIX_NAMES, ANCHORED_INWARD_NAMES, ANCHORED_OUTWARD_NAMES = (
    build_composition_combos(SCALE_NAMES)
)

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

# %% Part 2 — build the N_SCALE_STEPS + 1 crops per combo. All-or-nothing per combo: boxes
# shrink monotonically from global to close, so if the closest one clears MIN_CROP_SIZE every
# other step does too (see scale_step_boxes's docstring) — a combo either gets every scale or
# is skipped entirely, no partial scale availability to special-case downstream.
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

# %% Part 3 — encoder + query-image patch tokens + GT patch masks, plus (for achievable_iou,
# see Part 5) ref-image patch tokens + GT patch masks — the ref image already has its own GT
# (group_ref_masks, used to build the gallery in the first place), so it's the natural
# achievable-IoU reference for this fixed ref/query-pair script (see _shared/thresholding.py's
# achievable_iou docstring: fit the threshold on a reference raw map/GT the query's own GT never
# touches).
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
        "run": "1-1",
        "phase": "query_image_encode",
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

ref_encodings: dict[str, tuple[torch.Tensor, int, int]] = {}
with cuda_timer() as t_ref_encode:
    for unit in tqdm(sorted(ref_images), desc="Encoding ref images (for achievable_iou)"):
        tokens, r_h, r_w = extract_patch_tokens(encoder, ref_images[unit], LAYER_IDX, debias=True)
        ref_encodings[unit] = (tokens, r_h, r_w)
latency_rows.append(
    {
        "run": "1-1",
        "phase": "ref_image_encode",
        "elapsed_s": t_ref_encode["elapsed_s"],
        "n_units": len(ref_images),
        "units_per_sec": images_per_sec(len(ref_images), t_ref_encode["elapsed_s"]),
    }
)

ref_gt_patch_masks: dict[tuple[str, str], np.ndarray] = {}
for (unit, group), pixel_mask in group_ref_masks.items():
    _, r_h, r_w = ref_encodings[unit]
    ref_gt_patch_masks[(unit, group)] = pixel_mask_to_patch_mask(
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
            # Part 5/9 instead.
            fg_by_scale[(ck, name)] = fg.cpu()
            bg_by_scale[(ck, name)] = bg.cpu()
latency_rows.append(
    {
        "run": "1-1",
        "phase": "gallery_crop_encode",
        "elapsed_s": t_crop_encode["elapsed_s"],
        "n_units": len(clean_items),
        "units_per_sec": images_per_sec(len(clean_items), t_crop_encode["elapsed_s"]),
    }
)

# Background is scale-composition-invariant: every combo's every composition, single-scale or
# multi-scale, is scored against the SAME pooled bg gallery spanning every scale step (mirrors
# FGBG_SOURCE_COMBOS's bg=["global","mid","close"] always-full-span default).
bg_all_lookup: dict[tuple, torch.Tensor] = {
    ck: torch.cat([bg_by_scale[(ck, name)] for name in SCALE_NAMES], dim=0)
    for ck in usable_combo_keys
}
log.info("Built per-scale fg/bg galleries for %d combos", len(usable_combo_keys))

# %% Part 5 — main per-combo, per-composition-combo scoring. Alongside oracle_iou (tunes its
# threshold against the query's own GT — an upper bound), also scores the *same* gallery against
# the reference/exemplar image's own full extent + its own GT (ref_encodings/ref_gt_patch_masks
# from Part 3) to get achievable_iou: a threshold tuned on the exemplar, transferred as-is to the
# query — the number a deployed pipeline without query-time labels would actually see. One extra
# score_heatmap/knn_score_heatmap call per (combo, composition), not per query, since there is
# exactly one query per combo in this fixed-pair script.
IouLookup = dict[str, dict[str, dict[tuple, float]]]  # composition_name -> method -> ck -> iou
iou_lookup: IouLookup = {name: {method: {} for method in METHODS} for name in COMPOSITION_COMBOS}
# ai_lookup mirrors iou_lookup exactly, holding achievable_iou instead of oracle_iou.
ai_lookup: IouLookup = {name: {method: {} for method in METHODS} for name in COMPOSITION_COMBOS}

# gt_area_frac_by_key[(unit, group)] -> query GT's own patch-mask coverage — same for every
# composition/method since it only depends on the query, computed once here for Part 12's
# size-correlation analysis rather than recomputed per composition.
gt_area_frac_by_key: dict[tuple[str, str], float] = {
    key: float(gt.sum()) / gt.size for key, gt in gt_patch_masks.items()
}

qualitative_examples: list[ScoredExample] = []

with cuda_timer() as t_scoring_1_1:
    for combo in tqdm(combos, desc="Part 5: scoring compositions"):
        ck = combo_key(combo)
        if ck not in usable_combo_keys:
            continue
        unit, group = ck[0], ck[1]
        gt = gt_patch_masks.get((unit, group))
        if gt is None:
            continue
        q_tokens, q_h, q_w = query_encodings[unit]
        bg_bank = bg_all_lookup[ck].to(q_tokens.device)
        ref_tokens, ref_h, ref_w = ref_encodings[unit]
        ref_gt = ref_gt_patch_masks.get((unit, group))
        if ref_gt is None:
            log.warning(
                "unit=%s group=%s: no reference-image GT — achievable_iou left NaN", unit, group
            )

        for composition_name, members in COMPOSITION_COMBOS.items():
            fg_bank = torch.cat([fg_by_scale[(ck, name)] for name in members], dim=0).to(
                q_tokens.device
            )

            proto = compute_exemplar_features(fg_bank, mode="mean")
            raw_proto = score_heatmap(q_tokens, proto, q_h, q_w)
            iou_lookup[composition_name]["single_proto"][ck] = oracle_iou(
                raw_proto, gt, ORACLE_THRESHOLD_STEPS
            )
            if ref_gt is not None:
                ref_raw_proto = score_heatmap(ref_tokens, proto, ref_h, ref_w)
                ai_lookup[composition_name]["single_proto"][ck] = achievable_iou(
                    ref_raw_proto, ref_gt, raw_proto, gt, ORACLE_THRESHOLD_STEPS
                )
            else:
                ai_lookup[composition_name]["single_proto"][ck] = float("nan")

            raw_knn = knn_score_heatmap(
                q_tokens, fg_bank, bg_bank, KNN_FGBG_NUM_NEIGHBOURS, q_h, q_w
            )
            iou_lookup[composition_name]["knn_fgbg"][ck] = oracle_iou(
                raw_knn, gt, ORACLE_THRESHOLD_STEPS
            )
            if ref_gt is not None:
                ref_raw_knn = knn_score_heatmap(
                    ref_tokens, fg_bank, bg_bank, KNN_FGBG_NUM_NEIGHBOURS, ref_h, ref_w
                )
                ai_lookup[composition_name]["knn_fgbg"][ck] = achievable_iou(
                    ref_raw_knn, ref_gt, raw_knn, gt, ORACLE_THRESHOLD_STEPS
                )
            else:
                ai_lookup[composition_name]["knn_fgbg"][ck] = float("nan")

            if (
                composition_name == QUALITATIVE_COMPOSITION
                and QUALITATIVE_METHOD == "knn_fgbg"
                and len(qualitative_examples) < QUALITATIVE_MAX_EXAMPLES
            ):
                qualitative_examples.append(
                    ScoredExample(
                        label=f"{unit}/{group}",
                        image=query_images[unit],
                        raw=raw_knn,
                        gt=gt,
                        score=iou_lookup[composition_name]["knn_fgbg"][ck],
                    )
                )
latency_rows.append(
    {
        "run": "1-1",
        "phase": "scoring",
        "elapsed_s": t_scoring_1_1["elapsed_s"],
        "n_units": len(usable_combo_keys),
        "units_per_sec": images_per_sec(len(usable_combo_keys), t_scoring_1_1["elapsed_s"]),
    }
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


def mean_std_ai(composition_name: str, method: str) -> tuple[float, float, int]:
    """Same as mean_std_iou but against ai_lookup and NaN-aware (achievable_iou is left NaN for
    combos whose reference image has no own GT — see Part 5 — and those must not poison the
    mean/std of the combos that do)."""
    vals = np.array(list(ai_lookup[composition_name][method].values()))
    valid = vals[~np.isnan(vals)]
    if len(valid) == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(valid)), float(np.std(valid)), len(valid)


per_scale_rows = []
for name, t in zip(SCALE_NAMES, T_VALUES):
    for method in METHODS:
        mean, std, n = mean_std_iou(iou_lookup, name, method)
        # Percentile bootstrap CI on the mean, alongside the plain std this script already
        # reported — std alone doesn't say whether e.g. "global"'s and "close"'s means are
        # actually distinguishable or both plausible draws from the same underlying distribution.
        if n > 0:
            _, ci_lo, ci_hi = bootstrap_ci(
                np.array(list(iou_lookup[name][method].values())),
                n_boot=N_BOOTSTRAP,
                seed=BOOTSTRAP_SEED,
            )
        else:
            ci_lo = ci_hi = float("nan")
        mean_ai, std_ai, _ = mean_std_ai(name, method)
        per_scale_rows.append(
            {
                "scale": name,
                "t": t,
                "method": method,
                "mean_iou": mean,
                "std_iou": std,
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
                "mean_achievable_iou": mean_ai,
                "std_achievable_iou": std_ai,
                "oracle_minus_achievable_gap": (
                    mean - mean_ai if not (np.isnan(mean) or np.isnan(mean_ai)) else float("nan")
                ),
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
    f"1-1 (single ref/query pair) — Per-scale oracle IoU, n={N_SCALE_STEPS} steps "
    f"({len(usable_combo_keys)} combos)\ndashed line = average IoU across all scale steps"
)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "per_scale_iou.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved %s and %s", OUTPUT_DIR / "per_scale_iou.csv", OUTPUT_DIR / "per_scale_iou.png")

# %% Part 6b — oracle IoU (upper bound, tunes the threshold against the query's own GT) vs.
# achievable IoU (a threshold tuned on the reference/exemplar image's own GT, transferred as-is
# to the query — what a deployed pipeline without query-time labels would actually get), per
# scale step. Every other figure in this script plots oracle_iou only.
fig, ax = plt.subplots(figsize=(8, 5.5))
for method in METHODS:
    sub = per_scale_df[per_scale_df.method == method]
    ax.plot(
        sub["t"],
        sub["mean_iou"],
        marker="o",
        linestyle="-",
        label=f"{method} oracle",
        color=METHOD_COLOR[method],
    )
    ax.plot(
        sub["t"],
        sub["mean_achievable_iou"],
        marker="^",
        linestyle="--",
        label=f"{method} achievable",
        color=METHOD_COLOR[method],
        alpha=0.6,
    )
ax.set_xticks(T_VALUES, SCALE_NAMES, rotation=45)
ax.set_xlabel("crop tightness t (0 = global, 1 = close)")
ax.set_ylabel("mean IoU across combos")
ax.set_ylim(0, 1.0)
ax.set_title(
    "1-1 (single ref/query pair) — Oracle (upper bound) vs. achievable (exemplar-tuned "
    f"threshold) IoU per scale, n={N_SCALE_STEPS} steps"
)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "oracle_vs_achievable.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Oracle-vs-achievable gap per scale (mean oracle_iou - mean achievable_iou):")
for _, row in per_scale_df.iterrows():
    log.info(
        "  scale=%-8s method=%-13s gap=%.3f (oracle=%.3f achievable=%.3f)",
        row.scale,
        row.method,
        row.oracle_minus_achievable_gap,
        row.mean_iou,
        row.mean_achievable_iou,
    )
log.info("Saved %s", OUTPUT_DIR / "oracle_vs_achievable.png")

# %% Part 7 — composition growth curves (open-ended + anchored-at-both-ends) + full table
composition_rows = []
for name, members in COMPOSITION_COMBOS.items():
    for method in METHODS:
        mean, std, n = mean_std_iou(iou_lookup, name, method)
        if n > 0:
            _, ci_lo, ci_hi = bootstrap_ci(
                np.array(list(iou_lookup[name][method].values())),
                n_boot=N_BOOTSTRAP,
                seed=BOOTSTRAP_SEED,
            )
        else:
            ci_lo = ci_hi = float("nan")
        mean_ai, std_ai, _ = mean_std_ai(name, method)
        composition_rows.append(
            {
                "composition": name,
                "n_scales": len(members),
                "members": "+".join(members),
                "method": method,
                "mean_iou": mean,
                "std_iou": std,
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
                "mean_achievable_iou": mean_ai,
                "std_achievable_iou": std_ai,
                "oracle_minus_achievable_gap": (
                    mean - mean_ai if not (np.isnan(mean) or np.isnan(mean_ai)) else float("nan")
                ),
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
    f"1-1 (single ref/query pair) — Scale composition growth curves, n={N_SCALE_STEPS} "
    f"steps ({len(usable_combo_keys)} combos)"
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
    ax.set_title(
        f"1-1 (single ref/query pair) — Per-scale oracle IoU, group={group} "
        f"(n={len(cks)} combos)"
    )
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
        if c["unit"] == FOCUS_UNIT
        and c["class"] == FOCUS_CLASS
        and c["instance_id"] == FOCUS_INSTANCE_ID
    ),
    combos[0],
)
focus_ck = combo_key(focus_combo)
if focus_ck not in usable_combo_keys:
    log.warning("Focus combo %s has no usable scale steps — skipping qualitative figure", focus_ck)
else:
    focus_unit, focus_group = focus_ck[0], focus_ck[1]
    focus_gt = gt_patch_masks[(focus_unit, focus_group)]
    focus_q_tokens, focus_q_h, focus_q_w = query_encodings[focus_unit]
    focus_bg = bg_all_lookup[focus_ck].to(focus_q_tokens.device)

    query_img = query_images[focus_unit]
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
        fg_bank = fg_by_scale[(focus_ck, name)].to(focus_q_tokens.device)
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

# %% Part 9b — worst/best-N qualitative gallery at one representative composition (see
# QUALITATIVE_COMPOSITION/QUALITATIVE_METHOD above) — every other figure in this script averages
# across combos; this shows actual individual query images so a failure mode (one orientation,
# one lighting condition) is visible instead of washed out by the mean.
if qualitative_examples:
    save_score_gallery(
        qualitative_examples,
        OUTPUT_DIR / "qualitative_worst_best.png",
        n=5,
        score_name="oracle_iou",
        title=(
            f"Worst/best oracle_iou examples: composition={QUALITATIVE_COMPOSITION} "
            f"method={QUALITATIVE_METHOD}"
        ),
    )
    log.info(
        "Wrote %s (%d examples)",
        OUTPUT_DIR / "qualitative_worst_best.png",
        len(qualitative_examples),
    )
else:
    log.warning("No qualitative examples collected for the representative point")

# %% Part 10 — 5-3 pooled gallery, cross-validated: does "both methods peak mid-sweep, not
# at either extreme" (Part 6's finding) hold when the gallery is pooled from 5 training
# images instead of one? Reuses `split_fg_bg_patches`/`score_heatmap`/`knn_score_heatmap`/
# `oracle_iou` unchanged — only discovery, per-instance crop-building, and fold/role
# assignment are new (see `_shared/pooled_gallery_cv.py`). Scoped to the classic 3-point
# global/mid/close (not the full N_SCALE_STEPS=6 fine sweep) — every sibling script in the
# suite already uses this as its own baseline, and re-sweeping all 7 points here for a
# pooled gallery would multiply this section's cost for resolution the headline finding
# ("peaks mid-sweep, not at either extreme") doesn't need.
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
with cuda_timer() as t_image_encode_53:
    for key in tqdm(sorted(discovery_53.images), desc="5-3: encoding images"):
        tokens, h, w = extract_patch_tokens(
            encoder, discovery_53.images[key], LAYER_IDX, debias=True
        )
        image_encodings_53[key] = {"tokens": tokens, "h": h, "w": w}
latency_rows.append(
    {
        "run": "5-3",
        "phase": "full_image_encode",
        "elapsed_s": t_image_encode_53["elapsed_s"],
        "n_units": len(discovery_53.images),
        "units_per_sec": images_per_sec(len(discovery_53.images), t_image_encode_53["elapsed_s"]),
    }
)

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

with cuda_timer() as t_crop_encode_53:
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
latency_rows.append(
    {
        "run": "5-3",
        "phase": "gallery_crop_encode",
        "elapsed_s": t_crop_encode_53["elapsed_s"],
        "n_units": len(clean_items_53),
        "units_per_sec": images_per_sec(len(clean_items_53), t_crop_encode_53["elapsed_s"]),
    }
)

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

# Sweep: for each fold x part_type x group x scale, pool the fold's training instances' fg at
# that one scale (bg always pooled across every POOL_SCALES_53 entry, matching this file's
# own "bg is scale-composition-invariant" convention), score against every eval image.
fold_splits_53 = make_fold_role_splits(PART_TYPES_53)  # truly randomized, not SEED-reproducible
results_53: list[dict] = []
n_units_53 = N_FOLDS_53 * len(PART_TYPES_53)

with cuda_timer() as t_scoring_53, tqdm(
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
                bg_bank = cap_bank_size(
                    torch.cat(
                        [bg_by_inst_scale_53[(i, s)] for i in pool_idxs for s in POOL_SCALES_53],
                        dim=0,
                    ),
                    MAX_BANK_SIZE_KNN_53,
                    SEED,
                )
                if bg_bank.shape[0] == 0:
                    continue
                for eval_number in eval_numbers:
                    key = (part_type, group, eval_number)
                    if key not in gt_patch_masks_53:
                        continue
                    q = image_encodings_53[(part_type, eval_number)]
                    gt = gt_patch_masks_53[key]
                    bg_dev = bg_bank.to(q["tokens"].device)
                    for scale in POOL_SCALES_53:
                        fg_bank = cap_bank_size(
                            torch.cat([fg_by_inst_scale_53[(i, scale)] for i in pool_idxs], dim=0),
                            MAX_BANK_SIZE_KNN_53,
                            SEED,
                        ).to(q["tokens"].device)
                        if fg_bank.shape[0] == 0:
                            continue
                        proto = compute_exemplar_features(fg_bank, mode="mean")
                        raw_proto = score_heatmap(q["tokens"], proto, q["h"], q["w"])
                        results_53.append(
                            {
                                "scale": scale,
                                "method": "single_proto",
                                "oracle_iou": oracle_iou(raw_proto, gt, ORACLE_THRESHOLD_STEPS),
                            }
                        )
                        raw_knn = knn_score_heatmap(
                            q["tokens"], fg_bank, bg_dev, KNN_FGBG_NUM_NEIGHBOURS, q["h"], q["w"]
                        )
                        results_53.append(
                            {
                                "scale": scale,
                                "method": "knn_fgbg",
                                "oracle_iou": oracle_iou(raw_knn, gt, ORACLE_THRESHOLD_STEPS),
                            }
                        )
            pbar.update(1)
latency_rows.append(
    {
        "run": "5-3",
        "phase": "scoring",
        "elapsed_s": t_scoring_53["elapsed_s"],
        "n_units": len(results_53),
        "units_per_sec": images_per_sec(len(results_53), t_scoring_53["elapsed_s"]),
    }
)

results_53_df = pd.DataFrame(results_53)
summary_53_rows = []
for scale in POOL_SCALES_53:
    for method in METHODS:
        vals = results_53_df.loc[
            (results_53_df.scale == scale) & (results_53_df.method == method), "oracle_iou"
        ]
        row_11 = per_scale_df[(per_scale_df.scale == scale) & (per_scale_df.method == method)].iloc[
            0
        ]
        summary_53_rows.append(
            {
                "scale": scale,
                "method": method,
                "mean_iou_1_1": row_11.mean_iou,
                "std_iou_1_1": row_11.std_iou,
                "mean_iou_5_3": float(vals.mean()) if len(vals) else float("nan"),
                "std_iou_5_3": float(vals.std()) if len(vals) else float("nan"),
                "n_samples_5_3": len(vals),
            }
        )
summary_53_df = pd.DataFrame(summary_53_rows)
summary_53_df.to_csv(OUTPUT_DIR / "comparison_1_1_vs_5_3.csv", index=False)
log.info("1-1 vs 5-3 comparison (global/mid/close):")
for _, row in summary_53_df.iterrows():
    log.info(
        "  scale=%-6s %-13s 1-1=%.3f+/-%.3f  5-3=%.3f+/-%.3f (n=%d)",
        row.scale,
        row.method,
        row.mean_iou_1_1,
        row.std_iou_1_1,
        row.mean_iou_5_3,
        row.std_iou_5_3,
        row.n_samples_5_3,
    )
log.info("Wrote %s", OUTPUT_DIR / "comparison_1_1_vs_5_3.csv")

# %% Part 10b — latency/throughput: every phase above traded off against wall-clock cost, which
# no figure in this script reported before now. `torch.cuda.synchronize()` is called around every
# timed block (see `_shared/latency.py`) so GPU-async dispatch doesn't understate elapsed time.
# This script has no single clean sweep axis to plot latency against (Part 5's scoring loop
# scores every composition combo in one untimed-per-point pass, and restructuring that loop just
# to isolate per-point timing is out of scope for a purely additive change) — so unlike
# resolution_ablation.py/training_set_size_ablation.py's latency.png, this just logs and
# tabulates per-phase totals (matches scale_composition_bg_ablation.py's own choice for the same
# reason).
cache_hits, cache_misses = encoder.total_hits, encoder.total_misses
cache_total = cache_hits + cache_misses
latency_rows.append(
    {
        "run": "total",
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
    log.info(
        "  run=%-5s phase=%-20s elapsed=%.1fs n_units=%s",
        row.run,
        row.phase,
        row.elapsed_s,
        row.n_units,
    )
log.info("Wrote %s", OUTPUT_DIR / "latency.csv")

# %% Part 11 — is the global-vs-close scale effect (Part 6's per-scale curve, the script's own
# main swept axis) real, or combo-to-combo noise? An unpaired bootstrap comparison (see
# _shared/stats.py) of the two most extreme scale steps' per-combo oracle_iou arrays — the
# significance check `per_scale_iou.png` leaves the reader to eyeball.
significance_rows = []
for method in METHODS:
    global_vals = np.array(list(iou_lookup["global"][method].values()))
    close_vals = np.array(list(iou_lookup["close"][method].values()))
    if len(global_vals) == 0 or len(close_vals) == 0:
        continue
    prob_close_greater = bootstrap_prob_greater(
        close_vals, global_vals, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
    )
    significance_rows.append(
        {
            "method": method,
            "scale_lo": "global",
            "scale_hi": "close",
            "prob_hi_beats_lo": prob_close_greater,
            "n_lo": len(global_vals),
            "n_hi": len(close_vals),
        }
    )
significance_df = pd.DataFrame(significance_rows)
significance_df.to_csv(OUTPUT_DIR / "scale_effect_significance.csv", index=False)
log.info(
    "Scale effect significance (P(close mean > global mean) under %d-resample bootstrap; near "
    "0.5 = indistinguishable from noise):",
    N_BOOTSTRAP,
)
for _, row in significance_df.iterrows():
    log.info(
        "  method=%-13s P(close beats global)=%.3f (n=%d vs n=%d)",
        row.method,
        row.prob_hi_beats_lo,
        row.n_hi,
        row.n_lo,
    )
log.info("Wrote %s", OUTPUT_DIR / "scale_effect_significance.csv")

# %% Part 12 — does oracle IoU correlate with object size? An aggregate mean (every figure above)
# can hide "tighter/looser crops only help small/large instances" — `gt_area_frac_by_key` (the
# query GT's own patch-mask coverage, computed once in Part 5) lets us check, mirroring the
# pearson/spearman correlation pattern `scale_composition_adaptive_oracle.py` already established
# for instance size vs. optimal scale, grouped the same way this script's own headline breakdown
# (Part 6) already is: per (scale, method).
correlation_rows = []
for name in SCALE_NAMES:
    for method in METHODS:
        lookup = iou_lookup[name][method]
        cks = [ck for ck in lookup if (ck[0], ck[1]) in gt_area_frac_by_key]
        if len(cks) < 3:
            continue
        area_fracs = [gt_area_frac_by_key[(ck[0], ck[1])] for ck in cks]
        ious = [lookup[ck] for ck in cks]
        pearson_r, pearson_p = pearsonr(area_fracs, ious)
        spearman_r, spearman_p = spearmanr(area_fracs, ious)
        correlation_rows.append(
            {
                "scale": name,
                "method": method,
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
# everywhere) x oracle IoU for the knn_fgbg method (the stronger, more representative method),
# faceted per scale step like per_scale_iou__<group>.png above.
corr_flat_rows = [
    {"scale": name, "ck": ck, "gt_area_frac": gt_area_frac_by_key[(ck[0], ck[1])], "oracle_iou": v}
    for name in SCALE_NAMES
    for ck, v in iou_lookup[name]["knn_fgbg"].items()
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
for ax, name in zip(axes, SCALE_NAMES):
    tercile_means = (
        corr_flat_df[corr_flat_df.scale == name]
        .groupby("size_tercile", observed=True)["oracle_iou"]
        .mean()
    )
    tercile_means.plot(kind="bar", ax=ax, color=METHOD_COLOR["knn_fgbg"])
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("size tercile")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("mean oracle IoU (knn_fgbg)")
fig.suptitle("Does object size predict oracle IoU, per scale step?")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "size_correlation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(
    "Wrote %s and %s", OUTPUT_DIR / "size_correlation.csv", OUTPUT_DIR / "size_correlation.png"
)

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
# - **`oracle_vs_achievable.png`/`per_scale_iou.csv`'s and `composition_iou.csv`'s
#   `mean_achievable_iou`/`oracle_minus_achievable_gap` columns** — `oracle_iou` everywhere else
#   in this script is an upper bound (tunes its threshold against the query's own GT);
#   achievable_iou tunes the threshold on the reference/exemplar image's own GT instead (the same
#   image that built the gallery) and transfers it as-is to the query — the number a deployed
#   pipeline without query-time labels would actually see. A scale or composition ranking that
#   holds for oracle but not achievable IoU means it's a trend in "how separable the scores could
#   be," not in what a real threshold captures — check both before trusting `per_scale_iou.png`/
#   `composition_growth.png` alone.
# - **`latency.csv`** — GPU-synchronized wall-clock cost (see `_shared/latency.py`) per phase
#   (query/reference image encode, gallery-crop encode, 1-1 scoring, 5-3 image encode/crop
#   encode/scoring) plus cache hit-rate on the final "total" row. No `latency.png`/
#   `accuracy_vs_latency.png` here — Part 5's scoring loop doesn't isolate per-scale or
#   per-composition timing, so there's no natural per-point axis to plot latency against without
#   restructuring that loop (same reasoning `scale_composition_bg_ablation.py` already used).
# - **`scale_effect_significance.csv`** — an unpaired bootstrap comparison (2000 resamples, see
#   `_shared/stats.py`) of the two scale-sweep endpoints' (`global` vs. `close`) per-combo
#   oracle_iou arrays, per method: `prob_hi_beats_lo` near 0.5 means the apparent shape of
#   `per_scale_iou.png` is not distinguishable from combo-to-combo noise — a quantitative version
#   of that figure's eyeball comparison.
# - **`size_correlation.csv`/`.png`** — does oracle IoU correlate with the query GT's own area
#   fraction (`gt_area_frac_by_key`)? `.csv` covers both methods per scale step; `.png`'s tercile
#   bars use only `knn_fgbg` (the stronger method) per scale step, for readability — check
#   whether a tighter/looser crop's benefit concentrates on small objects specifically before
#   generalizing it.
# - **`qualitative_worst_best.png`** — actual worst-5/best-5 query images (crop, raw score map,
#   GT mask) at one representative composition (`global+mid+close`, `knn_fgbg`), not an average —
#   every other figure here plots a mean or an averaged heatmap, which can't show *why* a
#   specific image fails.


# %%
