# %% [markdown]
# # Fundamental: Augmented Prototypes — Does Perturbing the Exemplar Crop Improve
# # Oracle-IoU Localization?
#
# Third experiment in `experiments/fundamental/`. `scale_crop_similarity.py` asked how
# crop tightness moves an object's own embedding; `augmentation_sensitivity.py` asked how
# far six common perturbations move a *fixed* crop's embedding from its own unperturbed
# baseline. This one asks the practical question behind both: if one extra, perturbed view
# of the reference crop (rotated, relit, blurred, ...) is averaged into the plain,
# unaugmented masked-mean exemplar, does scoring a *different* image with that broadened
# prototype localize the object better or worse than the plain prototype alone — i.e. does
# adding a single augmented view behave like ordinary data augmentation (broadening what
# the prototype matches) or does it just walk the prototype away from a real match? Each
# severity is tested by itself, averaged with the clean baseline alone — not accumulated
# with the severities before it — so a curve singles out that one perturbation's own
# marginal effect rather than conflating it with everything already folded in ahead of it
# (that's what the composed-prototype sweep further down is for).
#
# "Oracle IoU" here means: given one raw cosine-similarity heatmap (prototype vs. every
# patch of the query image), sweep every candidate threshold and keep the best patch-mask
# IoU against the query's *own* ground-truth mask — the same `iou_threshold_curve` search
# `multiscale_crop_ablation.py` uses to *tune* a threshold, but applied directly to the
# query instead of fit-on-ref/apply-to-query. It isolates "how separable is object from
# background in this score map" from "how would we pick a threshold in practice", so an
# augmentation's effect on localizability can be read off on its own.
#
# Every (part type, instance-type group, instance) combination actually annotated in
# `data/abc3` is swept — not one hand-picked ref/query pair — so every quantitative curve
# and bar chart below is a mean *across combos*, not one lucky (or unlucky) object/lighting
# draw (the per-combo std is still computed and logged alongside each mean, just not
# plotted). `dinoisawesome.abc3` provides the discovery machinery
# (`PART_TYPES`, `INSTANCE_TYPE_GROUPS`, `available_instance_groups`) already used by
# `multiscale_crop_ablation.py`'s `RUN_PART_TYPES` sweep. The qualitative figures (crop
# grids, heatmaps) still show just one representative "focus" combo — a grid or heatmap
# panel per combo would be unreadable at this scale — mirroring how
# `augmentation_sensitivity.py` picks one `REFERENCE_*` instance for its crop grid while
# aggregating its drift curves across every instance.
#
# Steps:
#   1. Discover every (part_type, instance-type group, ref instance) combo in `data/abc3`
#      — one ref/query image pair per part type, one GT mask (union of that group's
#      classes) per (part_type, group), one exemplar crop source per annotated instance.
#   2. Build two prototype source crops per combo — "close" (tight bbox + padding) and
#      "mid" (halfway to the full image) — the same geometry as
#      `multiscale_crop_ablation.py`'s `scale_crop_box`.
#   3. Encode each part type's query image once. For each combo's scale's *unaugmented*
#      crop, build a masked-mean prototype, score every query patch (cosine similarity ->
#      raw heatmap), and compute its oracle IoU against that group's query GT mask — the
#      baseline.
#   4. For each augmentation family (rotation, illumination/gamma, color jitter, Gaussian
#      blur, Gaussian noise, JPEG compression) and severity sweep (same families/values as
#      `augmentation_sensitivity.py`, for direct comparability), apply it to the combo's
#      scale's crop (reprojecting the mask for rotation, the only geometry-moving family),
#      build *that severity's own* prototype, average it with the combo's clean (severity=0)
#      prototype, rescore the same query image with the pair, and recompute oracle IoU — so
#      each point isolates one severity's own contribution on top of the baseline.
#   5. Aggregate oracle-IoU-vs-severity curves (mean across every combo) for "close"
#      vs. "mid", plus each family's best achievable oracle IoU (mean over combos of the
#      per-combo max-over-severities) against the no-augmentation baseline, for both
#      scales.
#   6. Visualize the focus combo's augmented crop grids (per scale) and the aggregated
#      oracle-IoU curves/summary.
#   7. Compose an ensemble prototype per (combo, scale, family) by averaging L2-normalised
#      per-severity prototypes and re-normalising, swept cumulatively (k=1 original alone
#      through k=N all severities pooled), aggregated the same way, to see whether the
#      benefit — if any — builds up gradually or only shows up once every severity is
#      folded in, then compare the k=N endpoint against the original baseline and the best
#      single-severity prototype.
#   8. Run the same cumulative-composition idea across families instead of within one, swept
#      leave-one-out: at step k, every *included* family's entry at severity level k-1 is
#      folded into the running prototype alongside the shared baseline, k=1..N, once with
#      every family included ("none" held out) and once per family with that family's
#      crops left out entirely — to see whether folding in more *kinds* of perturbation
#      keeps helping the way more severity of one kind does, and which family (if any) the
#      grand ensemble would be better off without.
#   9. Visualize the focus combo's baseline vs. best single (family, severity) combo's
#      actual similarity heatmap, one row per scale, so a qualitative look at *where* the
#      score map changed backs up (or undercuts) the oracle-IoU number.
#  10. Flip the roles: hold each combo's scale's prototype clean (the plain, unaugmented
#      baseline) and instead perturb the *query* image (reprojecting its own GT mask the
#      same way — built once per (part_type, group), reused by every ref instance sharing
#      that group), then rescore and compare against the crop-augmentation sensitivity
#      ranking above — the more realistic "clean exemplar, noisy factory frame" scenario.

# %% Logging — must be before torch import
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("augmented_prototype_oracle_iou")

from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from dinoisawesome import DinoEncoder, compute_exemplar_features, load_annotations
from dinoisawesome.abc3 import INSTANCE_TYPE_GROUPS, PART_TYPES, available_instance_groups
from dinoisawesome.instance_detection import extract_patch_tokens

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
from _shared.thresholding import iou_threshold_curve  # noqa: E402

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

data_dir = _REPO_ROOT / "data" / "abc3"

REF_NUMBER = 1
QUERY_NUMBER = 2

# Every part type is swept by default — "load all instances and parts" means every
# quantitative curve/bar chart below aggregates across every (part_type, instance-type
# group, ref instance) combo actually annotated in data/abc3. Narrow this for fast
# iteration while developing the script, e.g. RUN_PART_TYPES = ["LHa"].
RUN_PART_TYPES: list[str] = PART_TYPES

# Combo used for the qualitative figures (augmented crop grid, baseline-vs-best heatmap,
# augmented query grid) — a figure per combo would be unreadable at this scale. Same
# object as scale_crop_similarity.py and augmentation_sensitivity.py's REFERENCE_*, for
# comparability. Falls back to the first discovered combo (with a warning) if this exact
# (part_type, class, instance_id) isn't present under RUN_PART_TYPES.
FOCUS_PART_TYPE = "LHa"
FOCUS_CLASS = "donut foam single"
FOCUS_INSTANCE_ID = 1

DINO_VERSION = "v3"
DINO_SIZE = "large"
IMG_SIZE = 768  # must be divisible by patch_size (16 for v3)
LAYER_IDX = 23  # penultimate/last block of ViT-L/16 (depth 24)
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")

MASK_PATCH_THRESHOLD = 0.3  # patch-grid cell counts as "object" once this fraction is masked
CROP_PADDING_FRACTION = 1.0  # close/mid crop padding, fraction of the mask bbox's own extent
MIN_CROP_SIZE = 128  # a combo's scale is skipped (not a hard error) below this native px size

ORACLE_THRESHOLD_STEPS = 25  # candidate thresholds searched per oracle-IoU evaluation

SCALES: list[str] = ["close", "mid"]
SCALE_COLOR: dict[str, str] = {"close": "#e74c3c", "mid": "#f39c12"}

SEED = 0
torch.manual_seed(SEED)

OUTPUT_DIR = _REPO_ROOT / "outputs" / "fundamental" / "augmented_prototype_oracle_iou"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log.info(
    "RUN_PART_TYPES=%s ref_number=%d query_number=%d  |  DINO%s-%s img_size=%d layer=%d",
    RUN_PART_TYPES,
    REF_NUMBER,
    QUERY_NUMBER,
    DINO_VERSION,
    DINO_SIZE,
    IMG_SIZE,
    LAYER_IDX,
)

# %% Helpers shared across the discovery / aggregation / plotting sections below


def combo_key(d: dict) -> tuple[str, str, str, int]:
    """(part_type, instance_type group, class, instance_id) — a combo's identity, stable
    across the "combos" list and every "entries"-style row built from it."""
    return (d["part_type"], d["group"], d["class"], d["instance_id"])


def oracle_iou(raw: np.ndarray, gt_mask: np.ndarray, steps: int) -> float:
    """Best patch-mask IoU any single global threshold on *raw* could achieve against
    *gt_mask* — "oracle" because the threshold is picked with direct knowledge of the
    query's own ground truth, the upper bound a real (ref-tuned) threshold could reach."""
    _, ious = iou_threshold_curve(raw, gt_mask, steps)
    return float(ious.max())


def masked_mean_prototype(
    patch_tokens: torch.Tensor, mask_px: np.ndarray, grid_h: int, grid_w: int, label: str
) -> torch.Tensor:
    """L2-normalise a crop's patch tokens and mean-pool the ones under *mask_px* (projected
    to the patch grid) into one prototype vector — falls back to every patch in the crop
    (logged) if the mask projects to nothing. Device-agnostic: runs whatever device
    *patch_tokens* is already on, so call it on a CPU tensor for CPU-side pooling."""
    patch_mask = pixel_mask_to_patch_mask(mask_px, grid_h, grid_w, IMG_SIZE, MASK_PATCH_THRESHOLD)
    tokens = F.normalize(patch_tokens.reshape(grid_h * grid_w, -1), p=2, dim=-1)
    patch_flat = torch.from_numpy(patch_mask.reshape(-1)).to(tokens.device)
    masked = tokens[patch_flat]
    if masked.shape[0] == 0:
        log.warning("%s: mask empty after patch-grid projection — using all crop patches", label)
        masked = tokens
    return compute_exemplar_features(masked, mode="mean")  # (1, D)


def score_heatmap(tokens: torch.Tensor, prototype: torch.Tensor, h: int, w: int) -> np.ndarray:
    """Cosine-similarity heatmap (prototype vs. every token), moved off the GPU as an
    (h, w) float32 numpy array — the shape iou_threshold_curve/oracle_iou expect."""
    return (tokens @ prototype.T).reshape(h, w).cpu().float().numpy()


# %% Part 1 — discover every (part_type, instance-type group, ref instance) combo
combos: list[dict] = []
group_query_masks: dict[tuple[str, str], np.ndarray] = {}
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

# %% Part 2 — build close/mid crops per combo (skip a combo's scale, not the whole run,
# if the crop falls below MIN_CROP_SIZE)
combo_keys_by_scale: dict[str, list[tuple]] = defaultdict(list)
for combo in tqdm(combos, desc="Building close/mid crops"):
    ref_img = ref_images[combo["part_type"]]
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
            "fill": mean_color(crop_img),
        }
        combo_keys_by_scale[scale].append(combo_key(combo))

for scale in SCALES:
    log.info("scale=%-5s usable combos=%d", scale, len(combo_keys_by_scale[scale]))

# %% Part 3 — encode each part type's query image once; per-(part_type, group) GT patch mask
encoder = DinoEncoder(
    version=DINO_VERSION,
    size=DINO_SIZE,
    img_size=IMG_SIZE,
    layers=[LAYER_IDX],
    weights_dir=DINO_WEIGHTS_DIR,
    amp=True,
)

query_encodings: dict[str, dict] = {}
for part_type in tqdm(sorted(query_images), desc="Encoding query images"):
    q_tokens, q_h, q_w = extract_patch_tokens(
        encoder, query_images[part_type], LAYER_IDX, debias=True
    )
    # Moved to CPU immediately: everything downstream (masked-mean pooling, cosine
    # scoring) is tiny relative to the encoder forward pass, so there's no reason to keep
    # patch grids resident in GPU memory once the encoder is done with them.
    query_encodings[part_type] = {"q_tokens": q_tokens.cpu(), "q_h": q_h, "q_w": q_w}

gt_patch_masks: dict[tuple[str, str], np.ndarray] = {}
for (part_type, group), pixel_mask in group_query_masks.items():
    q = query_encodings[part_type]
    gt_patch_masks[(part_type, group)] = pixel_mask_to_patch_mask(
        pixel_mask, q["q_h"], q["q_w"], IMG_SIZE, MASK_PATCH_THRESHOLD
    )

# %% Augmentation families — same families/values as augmentation_sensitivity.py. Each
# sweep starts at a literal no-op value (angle=0, gamma=1.0, jitter magnitude=0, ...), so
# severity index 0 is pixel-identical to the combo's *base crop* — one no-augmentation
# baseline per (combo, scale), shared by every family's severity=0 entry.

AUGMENTATIONS: dict[str, dict] = {
    "rotation": {
        "values": [0, 8, 16, 30, 50, 75],
        "unit": "deg",
        "apply": apply_rotation,
    },
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

# %% Part 4 — build every (combo, scale, family, severity) augmented crop up front
entries: list[dict] = []
for combo in tqdm(combos, desc="Building augmented crops"):
    for scale, crop in combo["crops"].items():
        for family, spec in AUGMENTATIONS.items():
            for val in spec["values"]:
                img, mask_px = spec["apply"](crop["img"], crop["mask_px"], val, crop["fill"])
                entries.append(
                    {
                        "part_type": combo["part_type"],
                        "group": combo["group"],
                        "class": combo["class"],
                        "instance_id": combo["instance_id"],
                        "scale": scale,
                        "family": family,
                        "value": val,
                        "img": img,
                        "mask_px": mask_px,
                    }
                )

log.info(
    "Built %d augmented crops across %d combos x %d scales x %d families",
    len(entries),
    len(combos),
    len(SCALES),
    len(AUGMENTATIONS),
)

# %% Part 5 — encode every augmented crop, chunked to encoder.max_batch_size, and score it
# immediately. Scoring here means the clean baseline (severity=0) prototype AVERAGED with
# this one crop's own embedding, not this crop's embedding alone — so oracle_iou_curves.png
# / best_vs_baseline.png below answer "what does adding just this one augmented view do to
# the clean exemplar?" (singling out one severity's marginal contribution), not "what if
# this one perturbed crop replaced the exemplar outright?" `entry["prototype"]` still holds
# each crop's own (unpaired) embedding — the composed-prototype sweep in Part 10 needs those
# to build its own N-way cumulative ensembles independently of this pairing. The severity=0
# point is unaffected either way: averaging the baseline with itself is the baseline.
#
# Each chunk's patch grid is moved to CPU right after the forward pass and pooled into its
# masked-mean prototype there, so the run never holds more than one chunk's raw patch grid
# in GPU memory. At this scale (thousands of crops, each a (grid_h, grid_w, D) grid)
# accumulating every chunk on GPU before scoring — the way a single-combo run comfortably
# could — would mean tens of GB of VRAM just for patch grids immediately reduced to a
# handful of floats each.
chunk_size = encoder.max_batch_size
clean_prototype_lookup: dict[tuple, torch.Tensor] = {}
for i in tqdm(range(0, len(entries), chunk_size), desc="Encoding + scoring crops"):
    chunk = entries[i : i + chunk_size]
    out = encoder([e["img"] for e in chunk], layers=[LAYER_IDX], debias=True)
    chunk_patches = out.patches[:, 0].cpu()  # (chunk, grid_h, grid_w, D)
    grid_h, grid_w = chunk_patches.shape[1], chunk_patches.shape[2]

    for entry, patch_tokens in zip(chunk, chunk_patches):
        label = (
            f"{combo_key(entry)} scale={entry['scale']} "
            f"family={entry['family']} value={entry['value']}"
        )
        prototype = masked_mean_prototype(patch_tokens, entry["mask_px"], grid_h, grid_w, label)
        entry["prototype"] = prototype

        # Entries are built (Part 4) and therefore encoded here in (combo, scale, family,
        # value) order with value=first_val first per family, so the baseline entry for a
        # given (combo, scale) is always encoded before any other entry that needs it.
        cs_key = (combo_key(entry), entry["scale"])
        if entry["family"] == first_family and entry["value"] == first_val:
            clean_prototype_lookup[cs_key] = prototype
        baseline_prototype = clean_prototype_lookup[cs_key]
        paired_prototype = F.normalize(
            torch.cat([baseline_prototype, prototype], dim=0).mean(dim=0, keepdim=True),
            p=2,
            dim=-1,
        )

        q = query_encodings[entry["part_type"]]
        raw = score_heatmap(q["q_tokens"], paired_prototype, q["q_h"], q["q_w"])
        gt = gt_patch_masks[(entry["part_type"], entry["group"])]
        entry["raw"] = raw
        entry["oracle_iou"] = oracle_iou(raw, gt, ORACLE_THRESHOLD_STEPS)

# %% Part 7 — lookup structures reused by aggregation, the composed-prototype sweep, and
# the query-augmentation section below (clean_prototype_lookup was already built in Part 5)
iou_lookup: dict[tuple, float] = {
    (combo_key(e), e["scale"], e["family"], e["value"]): e["oracle_iou"] for e in entries
}
entries_by_csf: dict[tuple, list[dict]] = defaultdict(list)
for e in entries:
    entries_by_csf[(combo_key(e), e["scale"], e["family"])].append(e)

# %% Generic aggregation helpers — reused for both the crop-augmentation sweep and the
# query-augmentation sweep (Part 9 below), since both produce a {(combo_key, scale,
# family, value): oracle_iou} lookup over the same combo_keys_by_scale.


def aggregate_curves(
    lookup: dict[tuple, float],
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Per-scale, per-family mean/std oracle IoU across every combo, one point per severity."""
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
    """Per-scale mean no-augmentation-baseline oracle IoU across every combo."""
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
    """Per (scale, family): mean/std of each combo's best-over-severities oracle IoU, plus
    the mean per-combo delta against that combo's own no-augmentation baseline."""
    rows: list[dict] = []
    for scale in SCALES:
        cks = combo_keys_by_scale[scale]
        for family, spec in AUGMENTATIONS.items():
            best_list: list[float] = []
            delta_list: list[float] = []
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


colors = plt.get_cmap("tab10").colors


def plot_severity_curves(
    curves: dict, baseline: dict[str, float], title: str, out_path: Path, ylabel: str
) -> None:
    fig, axes = plt.subplots(1, len(SCALES), figsize=(7.5 * len(SCALES), 5.5), sharey=True)
    for ax, scale in zip(axes, SCALES):
        n = len(combo_keys_by_scale[scale])
        for i, (family, spec) in enumerate(AUGMENTATIONS.items()):
            values = np.array(spec["values"], dtype=float)
            frac = (values - values[0]) / (values[-1] - values[0])
            mean = curves[scale][family]["mean"]
            color = colors[i % len(colors)]
            ax.plot(frac, mean, marker="o", label=family, color=color)
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


def plot_best_bar(rows: list[dict], baseline: dict[str, float], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    family_names = list(AUGMENTATIONS.keys())
    x = np.arange(len(family_names))
    width = 0.8 / len(SCALES)
    for i, scale in enumerate(SCALES):
        means = [
            next(r["mean_best_iou"] for r in rows if r["scale"] == scale and r["family"] == f)
            for f in family_names
        ]
        ax.bar(
            x + i * width,
            means,
            width=width,
            label=f"{scale} (best over severities, mean across combos)",
            color=SCALE_COLOR[scale],
        )
        ax.axhline(
            baseline[scale], color=SCALE_COLOR[scale], linestyle="--", linewidth=1, alpha=0.7
        )
    ax.set_xticks(x + width * (len(SCALES) - 1) / 2, family_names, rotation=30, ha="right")
    ax.set_ylabel("oracle IoU")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


# %% Part 8 — aggregate + visualize the crop-augmentation sweep
curves = aggregate_curves(iou_lookup)
baseline_oracle_iou = aggregate_baseline(iou_lookup)
best_over_baseline = aggregate_best_vs_baseline(iou_lookup)

log.info(
    "Baseline (no augmentation) oracle IoU: %s",
    {k: round(v, 3) for k, v in baseline_oracle_iou.items()},
)
for row in sorted(best_over_baseline, key=lambda r: -r["mean_delta_vs_baseline"]):
    log.info(
        "BEST scale=%-5s %-22s mean_best_iou=%.3f±%.3f (%+.3f vs. baseline, n=%d combos)",
        row["scale"],
        row["family"],
        row["mean_best_iou"],
        row["std_best_iou"],
        row["mean_delta_vs_baseline"],
        row["n_combos"],
    )

plot_severity_curves(
    curves,
    baseline_oracle_iou,
    f"Oracle-IoU vs. severity, baseline + one augmented view — {len(combos)} combos "
    f"across {len(RUN_PART_TYPES)} part types",
    OUTPUT_DIR / "oracle_iou_curves.png",
    "oracle IoU, baseline averaged with this one severity (mean across combos)",
)
plot_best_bar(
    best_over_baseline,
    baseline_oracle_iou,
    "Best single severity's contribution when averaged into the baseline "
    "(dashed = that scale's baseline)",
    OUTPUT_DIR / "best_vs_baseline.png",
)
log.info("Saved oracle-IoU curves and best-vs-baseline figures to %s", OUTPUT_DIR)

# %% Part 9 — focus combo used for every qualitative (crop-grid / heatmap) figure below
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

# %% Visualization — augmented crop grid, one figure per scale, one row per family
# (focus combo only — one grid per combo would be unreadable at this scale). The pictured
# crop is this severity alone; the oracle_iou printed under it is for baseline+this-severity
# averaged together (Part 5), not for this crop's prototype used by itself.
for scale in focus_combo["crops"]:
    focus_entries = [e for e in entries if combo_key(e) == focus_key and e["scale"] == scale]
    n_families = len(AUGMENTATIONS)
    n_cols = max(len(spec["values"]) for spec in AUGMENTATIONS.values())
    fig, axes = plt.subplots(n_families, n_cols, figsize=(2.6 * n_cols, 2.9 * n_families))
    for row, (family, spec) in enumerate(AUGMENTATIONS.items()):
        for col, val in enumerate(spec["values"]):
            ax = axes[row, col]
            entry = next(e for e in focus_entries if e["family"] == family and e["value"] == val)
            ax.imshow(entry["img"])
            ax.set_title(
                f"{val} {spec['unit']}\nbaseline+this: oracle_iou={entry['oracle_iou']:.3f}",
                fontsize=8,
            )
            ax.axis("off")
        for col in range(len(spec["values"]), n_cols):
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(family, fontsize=9)
    fig.suptitle(f"Augmented '{scale}' prototype crops — focus combo {focus_key}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUTPUT_DIR / f"augmented_crops_{scale}.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Composed prototype augmentation
#
# The severity sweep above scores one augmented crop's prototype at a time. This section
# asks the ensemble question instead: per (combo, scale, family), pool every severity's
# prototype — the no-op included — into one "composed" prototype by averaging the
# L2-normalised per-crop vectors (`compute_exemplar_features(mode="mean")` already returns
# unit vectors) and re-normalising the mean. It's a cheap, training-free stand-in for the
# multiple augmented views a real training pipeline would show a model.
#
# To see whether any benefit builds up gradually or only appears once every severity is
# folded in, each combo's composed prototype is swept cumulatively: k=1 uses just the
# original (no-op) crop, k=2 averages original+severity-2, ... up to k=N (all severities
# pooled) — the same sweep the line plots below chart, aggregated as mean ± std across
# every combo, with the k=N endpoint reused as "composed" in the summary bar chart that
# follows.

# %% Part 10 — composed prototype augmentation, cumulative severity-averaged, k=1..N
composed_curves: dict[str, dict[str, dict[str, np.ndarray]]] = {scale: {} for scale in SCALES}
composed_iou_lookup: dict[tuple, float] = {}
for scale in SCALES:
    for family in tqdm(AUGMENTATIONS, desc=f"Composing prototypes ({scale})"):
        n = len(AUGMENTATIONS[family]["values"])
        per_k_values: list[list[float]] = [[] for _ in range(n)]
        for ck in combo_keys_by_scale[scale]:
            family_entries = entries_by_csf.get((ck, scale, family), [])
            if len(family_entries) != n:
                continue
            part_type, group = ck[0], ck[1]
            q = query_encodings[part_type]
            gt = gt_patch_masks[(part_type, group)]
            for k in range(1, n + 1):
                composed_prototype = F.normalize(
                    torch.cat([e["prototype"] for e in family_entries[:k]], dim=0).mean(
                        dim=0, keepdim=True
                    ),
                    p=2,
                    dim=-1,
                )
                composed_raw = score_heatmap(q["q_tokens"], composed_prototype, q["q_h"], q["q_w"])
                iou = oracle_iou(composed_raw, gt, ORACLE_THRESHOLD_STEPS)
                composed_iou_lookup[(ck, scale, family, k)] = iou
                per_k_values[k - 1].append(iou)
        composed_curves[scale][family] = {
            "mean": np.array([float(np.mean(v)) if v else float("nan") for v in per_k_values]),
            "std": np.array([float(np.std(v)) if v else float("nan") for v in per_k_values]),
        }

composed_results: list[dict] = []
for scale in SCALES:
    for family in AUGMENTATIONS:
        n = len(AUGMENTATIONS[family]["values"])
        per_combo_composed, per_combo_delta_orig, per_combo_delta_best = [], [], []
        for ck in combo_keys_by_scale[scale]:
            composed_iou = composed_iou_lookup.get((ck, scale, family, n))
            if composed_iou is None:
                continue
            single_vals = [
                iou_lookup[(ck, scale, family, v)]
                for v in AUGMENTATIONS[family]["values"]
                if (ck, scale, family, v) in iou_lookup
            ]
            if not single_vals:
                continue
            per_combo_composed.append(composed_iou)
            baseline_iou = iou_lookup.get((ck, scale, first_family, first_val))
            if baseline_iou is not None:
                per_combo_delta_orig.append(composed_iou - baseline_iou)
            per_combo_delta_best.append(composed_iou - max(single_vals))
        composed_results.append(
            {
                "scale": scale,
                "family": family,
                "mean_composed_iou": (
                    float(np.mean(per_combo_composed)) if per_combo_composed else float("nan")
                ),
                "std_composed_iou": (
                    float(np.std(per_combo_composed)) if per_combo_composed else float("nan")
                ),
                "mean_delta_vs_original": (
                    float(np.mean(per_combo_delta_orig)) if per_combo_delta_orig else float("nan")
                ),
                "mean_delta_vs_best_single": (
                    float(np.mean(per_combo_delta_best)) if per_combo_delta_best else float("nan")
                ),
                "n_combos": len(per_combo_composed),
            }
        )

log.info("Composed (all-severities-averaged) prototype vs. original vs. best single severity:")
for row in sorted(composed_results, key=lambda r: -r["mean_delta_vs_original"]):
    log.info(
        "scale=%-5s %-22s composed=%.3f±%.3f  "
        "(%+.3f vs. original, %+.3f vs. best single, n=%d combos)",
        row["scale"],
        row["family"],
        row["mean_composed_iou"],
        row["std_composed_iou"],
        row["mean_delta_vs_original"],
        row["mean_delta_vs_best_single"],
        row["n_combos"],
    )

# %% Visualization — composed oracle-IoU curves (cumulative severities), one panel per
# scale, mean across every combo
fig, axes = plt.subplots(1, len(SCALES), figsize=(7.5 * len(SCALES), 5.5), sharey=True)
for ax, scale in zip(axes, SCALES):
    n_combos = len(combo_keys_by_scale[scale])
    for i, family in enumerate(AUGMENTATIONS):
        mean = composed_curves[scale][family]["mean"]
        color = colors[i % len(colors)]
        ax.plot(range(1, len(mean) + 1), mean, marker="o", label=family, color=color)
    ax.axhline(
        baseline_oracle_iou[scale],
        color="k",
        linestyle="--",
        linewidth=1,
        label="no-augmentation baseline",
    )
    ax.set_xlabel("severities composed (cumulative, k=1 is the original alone)")
    ax.set_title(f"scale = {scale}  (n={n_combos} combos)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("oracle IoU (composed prototype vs. query GT, mean across combos)")
axes[0].legend(fontsize=8)
fig.suptitle(f"Composed-prototype oracle-IoU vs. severities averaged — {len(combos)} combos")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "composed_oracle_iou_curves.png", dpi=150, bbox_inches="tight")

log.info("Saved composed oracle-IoU curves to %s", OUTPUT_DIR / "composed_oracle_iou_curves.png")

# %% Visualization — composed (all-severities) vs. original vs. best-single-severity
fig, ax = plt.subplots(figsize=(9, 5))
family_names = list(AUGMENTATIONS.keys())
x = np.arange(len(family_names))
width = 0.8 / len(SCALES)
for i, scale in enumerate(SCALES):
    means = [
        next(
            r["mean_composed_iou"]
            for r in composed_results
            if r["scale"] == scale and r["family"] == f
        )
        for f in family_names
    ]
    ax.bar(
        x + i * width,
        means,
        width=width,
        label=f"{scale} (composed, all severities, mean across combos)",
        color=SCALE_COLOR[scale],
    )
    ax.axhline(
        baseline_oracle_iou[scale], color=SCALE_COLOR[scale], linestyle="--", linewidth=1, alpha=0.7
    )
ax.set_xticks(x + width * (len(SCALES) - 1) / 2, family_names, rotation=30, ha="right")
ax.set_ylabel("oracle IoU")
ax.set_title(
    "Composed (all-severities-averaged) prototype oracle IoU "
    "(dashed = that scale's original baseline)"
)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "composed_vs_baseline.png", dpi=150, bbox_inches="tight")

log.info("Saved composed-prototype comparison to %s", OUTPUT_DIR / "composed_vs_baseline.png")

# %% [markdown]
# ## All-augmentations composed prototype (leave-one-out)
#
# The composed sweep above pools severities *within* one augmentation family at a time,
# cumulatively — six separate ensembles, one per family. This section runs the same
# cumulative-severity-level idea *across* families instead, swept leave-one-out style: one
# curve/bar with every family included ("none" held out — the full grand ensemble), plus
# one curve/bar per family with *that one family's* crops left out of the pool entirely. At
# step k, every *included* family's entry at severity index k-1 (one entry per included
# family, not just one family's own crop) is folded into the running prototype, on top of
# the shared clean baseline; k=1 is the baseline alone, k=N has every severity level of
# every included family folded in. The baseline itself is folded in once regardless of
# which family is held out: every family's severity=0 is pixel-identical to the same base
# crop, so pooling it in once per family would silently give the un-augmented view N-1x the
# weight of any one augmented view.
#
# Comparing each "leave family X out" curve/bar against the "none held out" reference
# isolates family X's own marginal contribution to the *grand ensemble specifically* — a
# different question than the single-severity sweep or the per-family composed sweep above,
# both of which measure a family's contribution against the plain baseline alone. If
# leaving X out makes the ensemble *worse* (drops below "none"), X was pulling its weight
# in the mix; if leaving it out makes the ensemble *better* (rises above "none"), X was
# diluting the other five families' signal and the grand ensemble is better off without it.

# %% Part 10b — all-augmentations composed prototype, leave-one-out: cumulative across
# severity level (as above), swept once with every family included ("none" held out) and
# once per family with that family excluded
N_SEVERITY_LEVELS = len(AUGMENTATIONS[first_family]["values"])  # 6; same length for every family
HELD_OUT_LABELS: list[str] = ["none", *AUGMENTATIONS]  # "none" = every family included


def included_families(held_out: str) -> list[str]:
    if held_out == "none":
        return list(AUGMENTATIONS)
    return [f for f in AUGMENTATIONS if f != held_out]


all_aug_composed_iou_lookup: dict[tuple, float] = {}  # (combo_key, scale, held_out, k) -> iou
for scale in SCALES:
    for ck in tqdm(combo_keys_by_scale[scale], desc=f"Leave-one-out composing ({scale})"):
        part_type, group = ck[0], ck[1]
        q = query_encodings[part_type]
        gt = gt_patch_masks[(part_type, group)]

        for held_out in HELD_OUT_LABELS:
            pooled_prototypes = [clean_prototype_lookup[(ck, scale)]]  # baseline, once
            for k in range(1, N_SEVERITY_LEVELS + 1):
                if k > 1:
                    level = k - 1  # severity index just folded in at this step
                    for family in included_families(held_out):
                        family_entries = entries_by_csf.get((ck, scale, family), [])
                        if len(family_entries) > level:
                            pooled_prototypes.append(family_entries[level]["prototype"])
                composed_prototype = F.normalize(
                    torch.cat(pooled_prototypes, dim=0).mean(dim=0, keepdim=True), p=2, dim=-1
                )
                raw = score_heatmap(q["q_tokens"], composed_prototype, q["q_h"], q["q_w"])
                all_aug_composed_iou_lookup[(ck, scale, held_out, k)] = oracle_iou(
                    raw, gt, ORACLE_THRESHOLD_STEPS
                )

all_aug_composed_curves: dict[str, dict[str, np.ndarray]] = {scale: {} for scale in SCALES}
for scale in SCALES:
    for held_out in HELD_OUT_LABELS:
        means = []
        for k in range(1, N_SEVERITY_LEVELS + 1):
            vals = [
                all_aug_composed_iou_lookup[(ck, scale, held_out, k)]
                for ck in combo_keys_by_scale[scale]
                if (ck, scale, held_out, k) in all_aug_composed_iou_lookup
            ]
            means.append(float(np.mean(vals)) if vals else float("nan"))
        all_aug_composed_curves[scale][held_out] = np.array(means)

all_aug_composed_results: list[dict] = []
for scale in SCALES:
    for held_out in HELD_OUT_LABELS:
        per_combo_iou: list[float] = []
        per_combo_delta_baseline: list[float] = []
        per_combo_delta_full: list[float] = []
        for ck in combo_keys_by_scale[scale]:
            iou = all_aug_composed_iou_lookup.get((ck, scale, held_out, N_SEVERITY_LEVELS))
            if iou is None:
                continue
            per_combo_iou.append(iou)

            baseline_iou = iou_lookup.get((ck, scale, first_family, first_val))
            if baseline_iou is not None:
                per_combo_delta_baseline.append(iou - baseline_iou)

            if held_out != "none":
                full_iou = all_aug_composed_iou_lookup.get((ck, scale, "none", N_SEVERITY_LEVELS))
                if full_iou is not None:
                    per_combo_delta_full.append(iou - full_iou)

        all_aug_composed_results.append(
            {
                "scale": scale,
                "held_out": held_out,
                "mean_iou": float(np.mean(per_combo_iou)) if per_combo_iou else float("nan"),
                "std_iou": float(np.std(per_combo_iou)) if per_combo_iou else float("nan"),
                "mean_delta_vs_baseline": (
                    float(np.mean(per_combo_delta_baseline))
                    if per_combo_delta_baseline
                    else float("nan")
                ),
                "mean_delta_vs_full": (
                    float(np.mean(per_combo_delta_full)) if per_combo_delta_full else float("nan")
                ),
                "n_combos": len(per_combo_iou),
            }
        )

log.info("All-augmentations composed prototype, k=%d, leave-one-out:", N_SEVERITY_LEVELS)
for scale in SCALES:
    none_row = next(
        r for r in all_aug_composed_results if r["scale"] == scale and r["held_out"] == "none"
    )
    log.info(
        "scale=%-5s held_out=%-22s all_aug_composed=%.3f±%.3f  (%+.3f vs. baseline, n=%d combos)",
        scale,
        "none",
        none_row["mean_iou"],
        none_row["std_iou"],
        none_row["mean_delta_vs_baseline"],
        none_row["n_combos"],
    )
    loo_rows = [
        r for r in all_aug_composed_results if r["scale"] == scale and r["held_out"] != "none"
    ]
    for row in sorted(loo_rows, key=lambda r: -r["mean_delta_vs_full"]):
        log.info(
            "scale=%-5s held_out=%-22s all_aug_composed=%.3f±%.3f  "
            "(%+.3f vs. baseline, %+.3f vs. none-held-out, n=%d combos)",
            scale,
            row["held_out"],
            row["mean_iou"],
            row["std_iou"],
            row["mean_delta_vs_baseline"],
            row["mean_delta_vs_full"],
            row["n_combos"],
        )

# %% Visualization — all-augmentations composed oracle-IoU vs. severity levels folded in,
# one panel per scale, one line per held-out family plus the "none held out" reference
fig, axes = plt.subplots(1, len(SCALES), figsize=(7.5 * len(SCALES), 5.5), sharey=True)
ks = range(1, N_SEVERITY_LEVELS + 1)
for ax, scale in zip(axes, SCALES):
    n_combos = len(combo_keys_by_scale[scale])
    ax.plot(
        ks,
        all_aug_composed_curves[scale]["none"],
        marker="o",
        label="none held out",
        color="black",
        linewidth=2,
    )
    for i, family in enumerate(AUGMENTATIONS):
        ax.plot(
            ks,
            all_aug_composed_curves[scale][family],
            marker="o",
            markersize=4,
            linestyle="--",
            label=f"hold out {family}",
            color=colors[i % len(colors)],
        )
    ax.axhline(
        baseline_oracle_iou[scale],
        color="0.6",
        linestyle=":",
        linewidth=1,
        label="no-augmentation baseline",
    )
    ax.set_xlabel("severity levels folded in (k=1 is the baseline alone)")
    ax.set_title(f"scale = {scale}  (n={n_combos} combos)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("oracle IoU (all-augmentations composed prototype, mean across combos)")
axes[0].legend(fontsize=7, ncol=2)
fig.suptitle(f"All-augmentations composed prototype, leave-one-out — {len(combos)} combos")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "all_augmentations_composed_curves.png", dpi=150, bbox_inches="tight")

log.info(
    "Saved all-augmentations composed curves to %s",
    OUTPUT_DIR / "all_augmentations_composed_curves.png",
)

# %% Visualization — all-augmentations composed prototype (k=N endpoint), leave-one-out,
# one group of bars per scale, one bar per held-out family plus "none held out"
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(HELD_OUT_LABELS))
width = 0.8 / len(SCALES)
for i, scale in enumerate(SCALES):
    means = [
        next(
            r["mean_iou"]
            for r in all_aug_composed_results
            if r["scale"] == scale and r["held_out"] == h
        )
        for h in HELD_OUT_LABELS
    ]
    ax.bar(
        x + i * width,
        means,
        width=width,
        label=f"{scale} (k={N_SEVERITY_LEVELS}, mean across combos)",
        color=SCALE_COLOR[scale],
    )
    ax.axhline(
        baseline_oracle_iou[scale], color=SCALE_COLOR[scale], linestyle="--", linewidth=1, alpha=0.7
    )
ax.set_xticks(
    x + width * (len(SCALES) - 1) / 2,
    ["none" if h == "none" else f"w/o {h}" for h in HELD_OUT_LABELS],
    rotation=30,
    ha="right",
)
ax.set_ylabel("oracle IoU (mean across combos)")
ax.set_title(
    f"All-augmentations composed prototype, leave-one-out, k={N_SEVERITY_LEVELS} endpoint "
    "(dashed = that scale's baseline)"
)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "all_augmentations_composed_vs_baseline.png", dpi=150, bbox_inches="tight")

log.info(
    "Saved all-augmentations composed comparison to %s",
    OUTPUT_DIR / "all_augmentations_composed_vs_baseline.png",
)

# %% [markdown]
# ## Baseline vs. best-combo heatmaps
#
# Every result above is an IoU number. This section turns the focus combo's single best
# result per scale — whichever (family, severity) scored highest when averaged into the
# baseline, found across every family, not one at a time — back into a picture: the raw
# cosine-similarity heatmap itself, plain baseline vs. baseline+that-one-severity, with the
# query's own GT mask outlined on both. See "Reading the results" at the end of this file
# for how to tell a heatmap-level improvement apart from an IoU-level one.

# %% Visualization — baseline vs. best (family, severity) combo heatmaps, one row per
# scale, focus combo only
focus_scales = list(focus_combo["crops"])
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

fig, axes = plt.subplots(len(focus_scales), 3, figsize=(15, 5 * len(focus_scales)))
if len(focus_scales) == 1:
    axes = axes.reshape(1, 3)

for row, scale in enumerate(focus_scales):
    scale_entries = [e for e in entries if combo_key(e) == focus_key and e["scale"] == scale]
    baseline_entry = next(
        e for e in scale_entries if e["family"] == first_family and e["value"] == first_val
    )
    best_entry = max(scale_entries, key=lambda e: e["oracle_iou"])
    best_unit = AUGMENTATIONS[best_entry["family"]]["unit"]
    delta = best_entry["oracle_iou"] - baseline_entry["oracle_iou"]
    gt_patch_mask = gt_patch_masks[(focus_key[0], focus_key[1])]

    axes[row, 0].imshow(focus_query_img)
    axes[row, 0].imshow(query_gt_disp, cmap="Reds", alpha=0.35)
    axes[row, 0].set_title(f"scale={scale}  query image + GT mask", fontsize=10)
    axes[row, 0].axis("off")

    im1 = axes[row, 1].imshow(baseline_entry["raw"], cmap="jet", aspect="auto")
    axes[row, 1].contour(gt_patch_mask.astype(float), levels=[0.5], colors="lime", linewidths=1.2)
    axes[row, 1].set_title(
        f"baseline (no aug) — oracle_iou={baseline_entry['oracle_iou']:.3f}", fontsize=10
    )
    axes[row, 1].axis("off")
    plt.colorbar(im1, ax=axes[row, 1], shrink=0.75, pad=0.02)

    im2 = axes[row, 2].imshow(best_entry["raw"], cmap="jet", aspect="auto")
    axes[row, 2].contour(gt_patch_mask.astype(float), levels=[0.5], colors="lime", linewidths=1.2)
    axes[row, 2].set_title(
        f"baseline + {best_entry['family']} value={best_entry['value']} {best_unit} — "
        f"oracle_iou={best_entry['oracle_iou']:.3f} ({delta:+.3f} vs. plain baseline)",
        fontsize=10,
    )
    axes[row, 2].axis("off")
    plt.colorbar(im2, ax=axes[row, 2], shrink=0.75, pad=0.02)

fig.suptitle(f"Baseline vs. baseline+best-severity heatmap — focus combo {focus_key}")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUTPUT_DIR / "baseline_vs_best_heatmap.png", dpi=150, bbox_inches="tight")

log.info(
    "Saved baseline-vs-best heatmap comparison to %s", OUTPUT_DIR / "baseline_vs_best_heatmap.png"
)

# %% [markdown]
# ## Augment the query instead of the prototype
#
# Every experiment above perturbs the *reference* crop and scores a clean, fixed query —
# convenient for isolating "what does augmenting the exemplar do to matching", but
# backwards from the common real-world case: a clean reference exemplar captured once
# under good conditions, scored against noisy factory-floor frames (bad lighting, motion
# blur, a part rotated in-frame, JPEG artifacts off a compressed camera feed). This
# section flips the roles: the prototype is held clean — the plain, unaugmented
# "close"/"mid" prototype already built for the severity-sweep baseline above — and the
# *query image* is perturbed instead, using the same augmentation families/severities/
# apply functions as everywhere else in this file (so rotation reprojects the query's own
# GT mask, exactly as it does for the reference crop). The augmented query doesn't depend
# on which combo/scale scores it — only on (part_type, group), since that's what fixes the
# query image and its GT mask — so each one is built and encoded once per (part_type,
# group) and then scored against every combo sharing that group.

# %% Part 11 — build every (part_type, group, family, severity) augmented query image up
# front, then encode once per unique image (chunked, with a progress bar)
query_aug_entries: list[dict] = []
for (part_type, group), pixel_mask in tqdm(
    group_query_masks.items(), desc="Building augmented query images"
):
    fill = mean_color(query_images[part_type])
    for family, spec in AUGMENTATIONS.items():
        for val in spec["values"]:
            aug_img, aug_pixel_mask = spec["apply"](query_images[part_type], pixel_mask, val, fill)
            query_aug_entries.append(
                {
                    "part_type": part_type,
                    "group": group,
                    "family": family,
                    "value": val,
                    "img": aug_img,
                    "pixel_mask": aug_pixel_mask,
                }
            )

log.info(
    "Built %d augmented query images across %d (part_type, group) pairs x %d families",
    len(query_aug_entries),
    len(group_query_masks),
    len(AUGMENTATIONS),
)

# Each chunk is moved to CPU right after encoding, same reasoning as the crop-augmentation
# loop above — these tokens are kept for the rest of the run (every combo sharing a
# (part_type, group) scores against them later), but on CPU rather than GPU: cheap to hold
# at this size, and every downstream use (normalize, the scoring matmul in Part 12) is
# device-agnostic.
for i in tqdm(range(0, len(query_aug_entries), chunk_size), desc="Encoding augmented query images"):
    chunk = query_aug_entries[i : i + chunk_size]
    out = encoder([e["img"] for e in chunk], layers=[LAYER_IDX], debias=True)
    chunk_patches = out.patches[:, 0].cpu()
    for entry, patch_tokens in zip(chunk, chunk_patches):
        q = query_encodings[entry["part_type"]]
        entry["tokens"] = F.normalize(patch_tokens.reshape(q["q_h"] * q["q_w"], -1), p=2, dim=-1)
        entry["gt_patch_mask"] = pixel_mask_to_patch_mask(
            entry["pixel_mask"], q["q_h"], q["q_w"], IMG_SIZE, MASK_PATCH_THRESHOLD
        )

query_aug_entries_by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
for e in query_aug_entries:
    query_aug_entries_by_group[(e["part_type"], e["group"])].append(e)

# %% Part 12 — score each combo's clean (unaugmented) prototype against every augmented
# query sharing its (part_type, group)
query_aug_iou_lookup: dict[tuple, float] = {}
for scale in SCALES:
    for ck in tqdm(combo_keys_by_scale[scale], desc=f"Scoring query-aug ({scale})"):
        part_type, group = ck[0], ck[1]
        prototype = clean_prototype_lookup[(ck, scale)]
        q = query_encodings[part_type]
        for qe in query_aug_entries_by_group[(part_type, group)]:
            raw = score_heatmap(qe["tokens"], prototype, q["q_h"], q["q_w"])
            query_aug_iou_lookup[(ck, scale, qe["family"], qe["value"])] = oracle_iou(
                raw, qe["gt_patch_mask"], ORACLE_THRESHOLD_STEPS
            )

# %% Part 13 — aggregate + visualize the query-augmentation sweep (same helpers as Part 8,
# since query_aug_iou_lookup has the same {(combo_key, scale, family, value): oracle_iou} shape)
query_aug_curves = aggregate_curves(query_aug_iou_lookup)
query_aug_baseline_oracle_iou = aggregate_baseline(query_aug_iou_lookup)
query_aug_best_over_baseline = aggregate_best_vs_baseline(query_aug_iou_lookup)

log.info(
    "[query-aug] Baseline (no augmentation) oracle IoU: %s",
    {k: round(v, 3) for k, v in query_aug_baseline_oracle_iou.items()},
)
for row in sorted(query_aug_best_over_baseline, key=lambda r: -r["mean_delta_vs_baseline"]):
    log.info(
        "[query-aug] BEST scale=%-5s %-22s mean_best_iou=%.3f±%.3f "
        "(%+.3f vs. baseline, n=%d combos)",
        row["scale"],
        row["family"],
        row["mean_best_iou"],
        row["std_best_iou"],
        row["mean_delta_vs_baseline"],
        row["n_combos"],
    )

# %% Visualization — augmented query image grid, one row per family (focus combo's
# (part_type, group) only; each title shows both scales' oracle IoU since the same
# augmented query is scored against two different clean prototypes)
focus_group_entries = query_aug_entries_by_group[(focus_key[0], focus_key[1])]
n_families = len(AUGMENTATIONS)
n_cols = max(len(spec["values"]) for spec in AUGMENTATIONS.values())
fig, axes = plt.subplots(n_families, n_cols, figsize=(2.6 * n_cols, 2.9 * n_families))
for row, (family, spec) in enumerate(AUGMENTATIONS.items()):
    for col, val in enumerate(spec["values"]):
        ax = axes[row, col]
        entry = next(e for e in focus_group_entries if e["family"] == family and e["value"] == val)
        ious_by_scale = {
            scale: query_aug_iou_lookup[(focus_key, scale, family, val)]
            for scale in focus_scales
            if (focus_key, scale, family, val) in query_aug_iou_lookup
        }
        iou_str = "  ".join(f"{s}={v:.3f}" for s, v in ious_by_scale.items())
        ax.imshow(entry["img"])
        ax.set_title(f"{val} {spec['unit']}\n{iou_str}", fontsize=8)
        ax.axis("off")
    for col in range(len(spec["values"]), n_cols):
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(family, fontsize=9)
fig.suptitle(f"Augmented query images, clean prototype — focus combo {focus_key}")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUTPUT_DIR / "augmented_query_grid.png", dpi=150, bbox_inches="tight")

plot_severity_curves(
    query_aug_curves,
    query_aug_baseline_oracle_iou,
    f"Oracle-IoU vs. query augmentation severity — {len(combos)} combos "
    f"across {len(RUN_PART_TYPES)} part types",
    OUTPUT_DIR / "query_aug_oracle_iou_curves.png",
    "oracle IoU (clean prototype vs. augmented-query GT, mean across combos)",
)
plot_best_bar(
    query_aug_best_over_baseline,
    query_aug_baseline_oracle_iou,
    "Best achievable query-augmentation oracle IoU per family (dashed = that scale's baseline)",
    OUTPUT_DIR / "query_aug_best_vs_baseline.png",
)

log.info("Saved query-augmentation figures to %s", OUTPUT_DIR)

# %% [markdown]
# ## Reading the results
#
# A family/scale combination whose mean curve rises *above* the dashed no-augmentation
# baseline is genuinely useful signal: averaging that one perturbed view into the clean
# exemplar made the resulting heatmap *more* separable on unseen query images *on average
# across every combo*, not less — evidence that this one augmented view, added to the
# baseline, generalizes the same way augmenting training data usually does, not just a
# quirk of one object/lighting pair. Because every point pairs a single severity with the
# unperturbed baseline rather than accumulating severities in sequence, the curve isolates
# that one perturbation's own marginal contribution — a spike at one severity and a dip at
# its neighbors means that specific magnitude helps and the rest don't, not that signal is
# building up gradually (that reading belongs to the composed curves below instead). The
# per-combo std logged alongside each mean (the plots below chart mean only) says how
# consistent the effect is: a family with a high mean but a wide std helps some
# part-type/object combos a lot and others little or not at all; a narrow std says the
# effect is uniform across the dataset. A family that only ever drops below baseline says
# the opposite for this backbone/layer: even one
# perturbed view averaged in pulls the prototype off the real object's manifold, and any
# resemblance to helpful "invariance" is coincidental. Comparing `close` against `mid` also
# separates two different failure/benefit modes — `close` prototypes are built from
# mostly-object pixels, so augmentation there mostly perturbs texture/appearance, while
# `mid` prototypes include a lot of the object's own surrounding context, so augmentation
# there also perturbs how much of that context factors into the masked-mean.
#
# The composed-prototype ensemble (severity-averaged, every value for a family pooled into
# one prototype) is a different bet than the single-severity sweep above: it can never
# land on the *single* highest-scoring perturbation for any one combo, but it also can't
# overfit to one lucky severity draw. If `mean_composed_iou` beats `mean_original` but
# trails `mean_best_single`, averaging is buying back some of the single-best severity's
# benefit without having to search for or commit to it — on average, across combos. If it
# beats both, the ensemble is smoothing out per-severity noise the oracle threshold search
# was otherwise exploiting. If it's worse than the plain original, this family's severities
# pull the prototype in inconsistent directions and averaging just cancels out whatever
# any one of them offered. The cumulative curves (k=1..N severities composed) show *how*
# it gets there, aggregated the same way: a curve that climbs steadily says the ensemble
# effect compounds as views are added; a curve that jumps on one k and flattens says a
# single severity is doing all the work and the rest are just dilution.
#
# The all-augmentations composed prototype asks the same "does it build up gradually"
# question as the per-family curves, but along a different axis: not more severity of one
# perturbation, but more *kinds* of perturbation folded in together, one severity level at
# a time, across every included family at once. The "none held out" curve (solid black)
# climbing through k=N says diversity across perturbation types keeps paying off; climbing
# early and then flattening or dipping says a couple of severity levels' worth of diversity
# captures most of the benefit and folding in the rest is dilution, not signal. The six
# leave-one-out curves (dashed, one per family) are where the ablation answer lives: a
# family whose leave-one-out curve sits *below* "none held out" (negative
# `mean_delta_vs_full`) was pulling its weight — the grand ensemble is worse without it. A
# family whose leave-one-out curve sits *above* "none held out" (positive
# `mean_delta_vs_full`) was diluting the other five families' signal — the ensemble is
# better off without it. The bar chart's "w/o {family}" bars are the same comparison at the
# k=N endpoint only, easier to rank at a glance than reading six overlapping curves.
#
# The focus combo's baseline-vs-best heatmap panels turn one representative IoU number
# back into a picture: a `jet` heatmap that visibly tightens around the green GT contour
# (less hot signal on background, more contiguous hot signal on the object) is a
# heatmap-level explanation for a positive delta *for that combo*. A heatmap that looks
# much like the baseline despite a nonzero IoU delta is a sign the gain came from the
# oracle threshold search finding a marginally better cut point rather than a genuinely
# more separable score map — worth checking against the aggregate mean±std before
# generalizing from one combo's picture.
#
# The query-augmentation results answer a different question than the crop-augmentation
# ones above: not "does perturbing the exemplar help matching" but "how much does the
# clean prototype's own oracle IoU degrade as the *query* gets noisier" — the realistic
# deployment failure mode (a clean exemplar captured once under good conditions, scored
# against whatever a factory-floor camera hands it). Comparing the two aggregated
# sensitivity rankings family-by-family is the point: a family whose query-augmentation
# curve drops sharply while its crop-augmentation curve stayed flat (or vice versa) says
# this backbone/layer is more sensitive to that perturbation on one side of the match than
# the other — worth knowing when only one side of a real pipeline is controllable.
#
# ## Other augmented-prototype experiments worth running in this dir
#
# - **Real threshold instead of oracle** — pair this with `iou_tuned_threshold`'s
#   fit-on-ref pattern (tune the threshold on the *augmented* ref crop's own mask, not the
#   query's GT) to see how much of any oracle-IoU gain survives once the threshold has to
#   be picked without seeing the query's ground truth.
# - **Per-part-type breakdown** — the aggregates here pool every part type together;
#   grouping `iou_lookup`/`query_aug_iou_lookup` by `part_type` (the first element of each
#   combo key) instead of pooling everything would show whether a helpful family is
#   general or specific to one part type's lighting/geometry.

# %%
