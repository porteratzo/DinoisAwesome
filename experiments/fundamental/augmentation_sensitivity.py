# %% [markdown]
# # Fundamental: Augmentation — How Robust Are DINOv3 Patch Embeddings to Common
# # Image Perturbations?
#
# Second experiment in `experiments/fundamental/` (see `scale_crop_similarity.py` for
# the first). That one asked how *crop tightness* moves an object's patch embedding;
# this one holds the crop fixed and asks how much each of several common perturbations
# moves it instead — the ones `scale_crop_similarity.py`'s closing markdown flagged as
# follow-ups (rotation, lighting) plus a few more that matter for real factory-floor
# imagery (abc3 is uncontrolled shop-floor lighting, not a studio).
#
# Steps:
#   1. Load every abc3 image and every annotated instance of every class in it (not
#      just one image/instance) — more instances and more part types means the drift
#      curves reflect the dataset instead of one lucky/unlucky crop.
#   2. Build one fixed "mid" crop per instance — padded around its bbox, partway
#      between the tight bbox and the whole image (no scale sweep here; scale is the
#      other experiment's axis).
#   3. For each augmentation family (rotation, illumination/gamma, color jitter,
#      Gaussian blur, Gaussian noise, JPEG compression), apply a severity sweep from
#      "no-op" up to a visibly strong perturbation, on every instance's crop.
#   4. Re-encode every augmented crop, pool the object's own patch tokens (mask
#      projected into that crop's grid — reprojected per-rotation, since rotation is
#      the only family that moves the object within the frame) into one masked-mean
#      embedding per crop.
#   5. Compare each severity level's embedding to that instance's own unperturbed
#      (severity=0) embedding via cosine similarity, then aggregate across all
#      instances per family/severity (mean ± std band) and plot drift curves for all
#      six families on one axis (x-axis normalized to a 0..1 "severity fraction" per
#      family so they're comparable despite different native units).
#   6. Visualize the augmented crop grid for one representative instance (same
#      image/class as `scale_crop_similarity.py`, for comparability) alongside the
#      aggregated drift plot.
#
# Not a per-dataset-update rerun: this measures how robust the backbone's own
# masked-mean pooling is to synthetic perturbations — a property of the encoder
# (model/layer/img_size) and the crop/pooling method, not of which reference/query
# images happen to be in the dataset. Confirmed empirically: results replicated to
# 3 decimals across abc3 (n=60) -> abc3+abc4 (n=227) -> abc5's physical merge of the
# same images (n=227, bit-identical crops). Only rerun this when the backbone, layer,
# img_size, or crop/pooling method changes — a routine dataset-merge/consolidation
# rerun of the `fundamental/` suite doesn't need to include it (it's also the
# second-most expensive script in the suite at ~13 min, ~21% of a full run).

# %% Logging — must be before torch import
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("augmentation_sensitivity")

from functools import partial
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
from _shared.latency import cuda_timer, images_per_sec  # noqa: E402
from _shared.mask_geometry import pixel_mask_to_patch_mask, scale_crop_box  # noqa: E402
from _shared.stats import bootstrap_ci, bootstrap_prob_greater  # noqa: E402

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

DATA_ROOT = _REPO_ROOT / "data"

# Every abc5 image, every annotated instance of every class in it — not just one
# image/instance. IMAGE_STEMS is discovered from disk so a new capture is picked up
# automatically. DATASETS/the (dataset, stem) tupling is a holdover from when this ran
# across abc3+abc4 separately (see _shared/dataset_pairs.py's docstring) — abc5 merged
# those into one dataset, but the plumbing still works unchanged with a single entry.
DATASETS: list[str] = ["abc5"]
IMAGE_STEMS: list[tuple[str, str]] = sorted(
    (dataset, p.stem) for dataset in DATASETS for p in (DATA_ROOT / dataset).glob("*.jpg")
)

# Reference instance used only for the augmented-crop-grid visualization (a full grid
# across all instances would be unreadable) — same object as scale_crop_similarity.py,
# for comparability. Drift curves aggregate every instance, not just this one.
REFERENCE_DATASET = "abc5"
REFERENCE_IMAGE_STEM = "LHa_1"
REFERENCE_TARGET_CLASS = "donut foam single"
REFERENCE_INSTANCE_ID = 1  # annotation "instance_id" (1-based, per class per image)

DINO_VERSION = "v3"
DINO_SIZE = "base"
IMG_SIZE = 768  # must be divisible by patch_size (16 for v3)
LAYER_IDX = 11  # last block of ViT-B/16 (depth 12)
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")
DINO_ENCODING_CACHE_DIR: str | None = os.environ.get("DINO_ENCODING_CACHE_DIR")

MASK_PATCH_THRESHOLD = 0.3  # patch-grid cell counts as "object" once this fraction is masked
MID_PADDING_FRACTION = 1.0  # mid-crop padding around the mask bbox, fraction of its extent

# One representative (family, severity) point whose individual crops get kept as PIL images
# for the worst/best-N qualitative gallery below — collecting every instance's image at every
# family/severity would multiply memory cost by n_instances x n_families x n_severities, so
# only this one point (the strongest tested severity of "gaussian blur" — a real-world-relevant
# failure mode per the module docstring's own factory-floor framing) is captured, capped at
# QUALITATIVE_MAX_EXAMPLES. Must be a member of AUGMENTATIONS above.
QUALITATIVE_FAMILY = "gaussian blur"
QUALITATIVE_SEVERITY_INDEX = -1  # index into AUGMENTATIONS[QUALITATIVE_FAMILY]["values"]
QUALITATIVE_MAX_EXAMPLES = 60  # capped so the gallery figure itself stays a readable size

# Bootstrap settings for the drift-summary CI and the mildest-vs-strongest significance check.
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

SEED = 0
torch.manual_seed(SEED)

OUTPUT_DIR = _REPO_ROOT / "outputs" / "fundamental_abc5" / "augmentation_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log.info(
    "images=%s  |  DINO%s-%s img_size=%d layer=%d",
    [f"{d}/{s}" for d, s in IMAGE_STEMS],
    DINO_VERSION,
    DINO_SIZE,
    IMG_SIZE,
    LAYER_IDX,
)

# %% Load every abc3 image + every annotated instance, build each one's fixed mid crop.
# "mid" here is scale_crop_box's actual mid geometry — the midpoint between the tight
# bbox and the full image edges — the same helper augmented_prototype_oracle_iou.py uses
# for its "mid" scale, so the two scripts' "mid crop" means the same thing. (An earlier
# version of this file reimplemented its own mid_bbox_crop, which despite the name/docs
# actually computed the *close*-scale formula — a padded-bbox crop, much smaller than a
# real mid crop — silently understating drift for this "mid" setting; see scale_crop_box
# in _shared/mask_geometry.py for the close/mid/global definitions.)
instances: list[dict] = []
for dataset, image_stem in tqdm(IMAGE_STEMS, desc="Loading images/annotations"):
    dataset_dir = DATA_ROOT / dataset
    anns = load_annotations(dataset_dir / "annotations" / image_stem)
    ref_img = Image.open(dataset_dir / f"{image_stem}.jpg").convert("RGB")
    for ann in anns:
        mask = ann["mask"]  # (H, W) bool, full native resolution
        mid_box = scale_crop_box(mask, "mid", MID_PADDING_FRACTION)
        x0, y0, x1, y1 = mid_box
        base_crop = ref_img.crop(mid_box)
        base_mask_px = mask[y0:y1, x0:x1]
        instances.append(
            {
                "dataset": dataset,
                "image_stem": image_stem,
                "class": ann["class"],
                "instance_id": ann["instance_id"],
                "base_crop": base_crop,
                "base_mask_px": base_mask_px,
                "fill": mean_color(base_crop),
            }
        )

reference_instance = next(
    (
        inst
        for inst in instances
        if inst["dataset"] == REFERENCE_DATASET
        and inst["image_stem"] == REFERENCE_IMAGE_STEM
        and inst["class"] == REFERENCE_TARGET_CLASS
        and inst["instance_id"] == REFERENCE_INSTANCE_ID
    ),
    None,
)
if reference_instance is None:
    raise ValueError(
        f"Reference instance dataset={REFERENCE_DATASET!r} image={REFERENCE_IMAGE_STEM!r} "
        f"class={REFERENCE_TARGET_CLASS!r} instance_id={REFERENCE_INSTANCE_ID} not found "
        "among loaded instances"
    )

log.info("Loaded %d instances across %d images", len(instances), len(IMAGE_STEMS))

# %% Augmentation families
#
# Each family is a severity sweep starting at a literal no-op value (angle=0,
# gamma=1.0, jitter magnitude=0, ...) so severity=0 is pixel-identical to *base_crop*
# — a built-in sanity check that every family's drift curve starts at similarity 1.0.
# All families are pixel-only (mask unchanged) except rotation, which moves the object
# within the frame and rotates the mask by the same angle.

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

# %% Build + encode every instance's augmented crops together, one instance at a time.
# Building all ~36 x N_instances augmented PIL images up front (a separate build-everything-
# then-encode-everything pass) exhausted host RAM once abc4 pushed the instance count from
# abc3-alone's ~60 to ~230 — only one instance's own 36 crops are ever resident now, chunked
# to encoder.max_batch_size same as before; only the reference instance's own images are kept
# afterward (for the qualitative crop-grid figure below), every other instance's img/mask_px
# is dropped right after encoding so the persisted `entries` list stays small (embeddings only).
encoder = DinoEncoder(
    version=DINO_VERSION,
    size=DINO_SIZE,
    img_size=IMG_SIZE,
    layers=[LAYER_IDX],
    weights_dir=DINO_WEIGHTS_DIR,
    max_batch_size=16,
    amp=True,
)
encoder = EncoderWithCache(encoder, cache_dir=DINO_ENCODING_CACHE_DIR)
chunk_size = encoder.max_batch_size

qualitative_severity_value = AUGMENTATIONS[QUALITATIVE_FAMILY]["values"][QUALITATIVE_SEVERITY_INDEX]
n_qualitative_imgs_kept = 0  # bounds how many non-reference crops keep their PIL image below

entries: list[dict] = []
latency_rows: list[dict] = []
with cuda_timer() as t_crop_encode:
    for inst in tqdm(instances, desc="Building + encoding augmented crops"):
        is_reference = inst is reference_instance
        inst_entries: list[dict] = []
        for family, spec in AUGMENTATIONS.items():
            for val in spec["values"]:
                img, mask_px = spec["apply"](
                    inst["base_crop"], inst["base_mask_px"], val, inst["fill"]
                )
                inst_entries.append(
                    {
                        "dataset": inst["dataset"],
                        "image_stem": inst["image_stem"],
                        "class": inst["class"],
                        "instance_id": inst["instance_id"],
                        "family": family,
                        "value": val,
                        "img": img,
                        "mask_px": mask_px,
                    }
                )

        for i in range(0, len(inst_entries), chunk_size):
            chunk = inst_entries[i : i + chunk_size]
            out = encoder([e["img"] for e in chunk], layers=[LAYER_IDX], debias=True)
            chunk_patches = out.patches[:, 0].cpu()  # (chunk, grid_h, grid_w, D) — freed at loop end
            D = chunk_patches.shape[-1]

            for entry, patch_tokens in zip(chunk, chunk_patches):
                patch_mask = pixel_mask_to_patch_mask(
                    entry["mask_px"], encoder.grid_h, encoder.grid_w, IMG_SIZE, MASK_PATCH_THRESHOLD
                )
                tokens = F.normalize(
                    patch_tokens.reshape(encoder.grid_h * encoder.grid_w, D), p=2, dim=-1
                )
                patch_flat = torch.from_numpy(patch_mask.reshape(-1)).to(tokens.device)

                masked = tokens[patch_flat]
                if masked.shape[0] == 0:
                    log.warning(
                        "image=%s class=%s instance=%s family=%s value=%s: mask empty after "
                        "patch-grid projection — using all crop patches",
                        entry["image_stem"],
                        entry["class"],
                        entry["instance_id"],
                        entry["family"],
                        entry["value"],
                    )
                    masked = tokens
                entry["embedding"] = compute_exemplar_features(masked, mode="mean")  # (1, D)
                entry["n_masked_patches"] = int(patch_flat.sum())
                # Object-size proxy for Part "size correlation" below: this crop's own object
                # patch-mask coverage, not a scored ground-truth region (this script has no
                # IoU/oracle-threshold step) — see that Part's docstring for why.
                entry["mask_area_frac"] = float(patch_flat.sum()) / patch_flat.numel()

        # Every non-reference instance's own PIL img/mask_px is dropped right after encoding to
        # keep `entries` small (see module docstring) — except the one representative
        # (QUALITATIVE_FAMILY, QUALITATIVE_SEVERITY_INDEX) point's crops, kept (capped at
        # QUALITATIVE_MAX_EXAMPLES total across all instances) for the worst/best-N qualitative
        # gallery below, which needs the actual images to show.
        if not is_reference:
            for e in inst_entries:
                if (
                    e["family"] == QUALITATIVE_FAMILY
                    and e["value"] == qualitative_severity_value
                    and n_qualitative_imgs_kept < QUALITATIVE_MAX_EXAMPLES
                ):
                    n_qualitative_imgs_kept += 1
                    continue
                e.pop("img", None)
                e.pop("mask_px", None)
        entries.extend(inst_entries)

latency_rows.append(
    {
        "phase": "augmented_crop_encode",
        "elapsed_s": t_crop_encode["elapsed_s"],
        "n_units": len(entries),
        "units_per_sec": images_per_sec(len(entries), t_crop_encode["elapsed_s"]),
    }
)

log.info(
    "Built + encoded %d augmented crops across %d instances x %d families",
    len(entries),
    len(instances),
    len(AUGMENTATIONS),
)

# %% Similarity vs. each instance's own severity=0 (unperturbed) embedding, then
# aggregated (mean + std) across all instances per family/severity
baseline_by_instance_family: dict[tuple, torch.Tensor] = {}
for family, spec in AUGMENTATIONS.items():
    baseline_val = spec["values"][0]
    for entry in entries:
        if entry["family"] == family and entry["value"] == baseline_val:
            key = (
                entry["dataset"],
                entry["image_stem"],
                entry["class"],
                entry["instance_id"],
                family,
            )
            baseline_by_instance_family[key] = entry["embedding"]

with cuda_timer() as t_scoring:
    for entry in entries:
        key = (
            entry["dataset"],
            entry["image_stem"],
            entry["class"],
            entry["instance_id"],
            entry["family"],
        )
        baseline = baseline_by_instance_family[key]
        entry["similarity"] = float((entry["embedding"] @ baseline.T).item())
latency_rows.append(
    {
        "phase": "similarity_scoring",
        "elapsed_s": t_scoring["elapsed_s"],
        "n_units": len(entries),
        "units_per_sec": images_per_sec(len(entries), t_scoring["elapsed_s"]),
    }
)
cache_hits, cache_misses = encoder.total_hits, encoder.total_misses
cache_total = cache_hits + cache_misses
latency_rows.append(
    {
        "phase": "total",
        "elapsed_s": t_crop_encode["elapsed_s"] + t_scoring["elapsed_s"],
        "n_units": len(entries),
        "units_per_sec": float("nan"),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": cache_hits / cache_total if cache_total > 0 else float("nan"),
    }
)
latency_df = pd.DataFrame(latency_rows)
latency_df.to_csv(OUTPUT_DIR / "latency.csv", index=False)
log.info(
    "Latency: augmented_crop_encode=%.1fs similarity_scoring=%.1fs total=%.1fs "
    "cache_hit_rate=%.2f (%d hits / %d misses) — no sweep axis in this script, so no "
    "latency.png (see latency.csv for the per-phase breakdown)",
    t_crop_encode["elapsed_s"],
    t_scoring["elapsed_s"],
    t_crop_encode["elapsed_s"] + t_scoring["elapsed_s"],
    cache_hits / cache_total if cache_total > 0 else float("nan"),
    cache_hits,
    cache_misses,
)

drift_summary: dict[str, dict[str, np.ndarray]] = {}
for family, spec in AUGMENTATIONS.items():
    mean_sim = []
    std_sim = []
    ci_lo_sim = []
    ci_hi_sim = []
    for val in spec["values"]:
        sims = np.array(
            [e["similarity"] for e in entries if e["family"] == family and e["value"] == val]
        )
        mean_sim.append(float(sims.mean()))
        std_sim.append(float(sims.std()))
        # Percentile bootstrap CI on the mean, alongside the plain std this script already
        # reported — std alone doesn't say whether e.g. this severity's and the no-op severity's
        # mean similarity are actually distinguishable across instances or both plausible draws
        # from the same distribution; see _shared/stats.py.
        _, ci_lo, ci_hi = bootstrap_ci(sims, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
        ci_lo_sim.append(ci_lo)
        ci_hi_sim.append(ci_hi)
    drift_summary[family] = {
        "mean": np.array(mean_sim),
        "std": np.array(std_sim),
        "ci_lo": np.array(ci_lo_sim),
        "ci_hi": np.array(ci_hi_sim),
    }
    log.info(
        "%-22s values=%s  mean_sim=%s  std=%s",
        family,
        spec["values"],
        np.round(mean_sim, 3),
        np.round(std_sim, 3),
    )

# %% Numeric results — per-crop table + per-family/severity aggregate
entries_df = pd.DataFrame(
    [
        {
            "dataset": e["dataset"],
            "image_stem": e["image_stem"],
            "class": e["class"],
            "instance_id": e["instance_id"],
            "family": e["family"],
            "value": e["value"],
            "n_masked_patches": e["n_masked_patches"],
            "mask_area_frac": e["mask_area_frac"],
            "similarity": e["similarity"],
        }
        for e in entries
    ]
)
entries_df.to_csv(OUTPUT_DIR / "per_crop_similarity.csv", index=False)

summary_rows = [
    {
        "family": family,
        "value": val,
        "mean_similarity": mean,
        "std_similarity": std,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
    }
    for family, spec in AUGMENTATIONS.items()
    for val, mean, std, ci_lo, ci_hi in zip(
        spec["values"],
        drift_summary[family]["mean"],
        drift_summary[family]["std"],
        drift_summary[family]["ci_lo"],
        drift_summary[family]["ci_hi"],
    )
]
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / "drift_summary.csv", index=False)
log.info(
    "Wrote %s (%d rows) and %s\n%s",
    OUTPUT_DIR / "per_crop_similarity.csv",
    len(entries_df),
    OUTPUT_DIR / "drift_summary.csv",
    summary_df.to_string(index=False),
)

# %% Is the mildest-vs-strongest severity drop within each family real, or just per-instance
# noise? An unpaired bootstrap comparison (see _shared/stats.py) of the mildest tested
# severity's vs. the strongest tested severity's per-instance similarity arrays, per family —
# the significance check `drift_curves.png` below leaves the reader to eyeball. "Mildest" here
# is `spec["values"][1]` (the first real perturbation), not `spec["values"][0]` (the literal
# no-op, whose similarity is always exactly 1.0 for every instance by construction — a
# degenerate, zero-variance comparison that would tell us nothing).
significance_rows = []
for family, spec in AUGMENTATIONS.items():
    if len(spec["values"]) < 2:
        continue
    mild_val, strong_val = spec["values"][1], spec["values"][-1]
    mild_vals = np.array(
        [e["similarity"] for e in entries if e["family"] == family and e["value"] == mild_val]
    )
    strong_vals = np.array(
        [e["similarity"] for e in entries if e["family"] == family and e["value"] == strong_val]
    )
    prob_mild_greater = bootstrap_prob_greater(
        mild_vals, strong_vals, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
    )
    significance_rows.append(
        {
            "family": family,
            "value_mild": mild_val,
            "value_strong": strong_val,
            "prob_mild_beats_strong": prob_mild_greater,
            "n_mild": len(mild_vals),
            "n_strong": len(strong_vals),
        }
    )
significance_df = pd.DataFrame(significance_rows)
significance_df.to_csv(OUTPUT_DIR / "augmentation_effect_significance.csv", index=False)
log.info(
    "Mildest-vs-strongest severity significance (P(mild mean similarity > strong mean "
    "similarity) under %d-resample bootstrap; near 0.5 = indistinguishable from noise):",
    N_BOOTSTRAP,
)
for _, row in significance_df.iterrows():
    log.info(
        "  %-22s P(value=%s beats value=%s)=%.3f (n=%d vs n=%d)",
        row.family,
        row.value_mild,
        row.value_strong,
        row.prob_mild_beats_strong,
        row.n_mild,
        row.n_strong,
    )
log.info("Wrote %s", OUTPUT_DIR / "augmentation_effect_significance.csv")

# %% Does object size correlate with how much an augmentation moves the embedding? This script
# has no GT/IoU scoring (it measures embedding drift via cosine similarity, not localization),
# so `mask_area_frac` (this crop's own object patch-mask coverage, added to every entries_df row
# above) is correlated against `similarity` directly rather than against an oracle_iou — the
# adapted version of the size-correlation check `scale_composition_adaptive_oracle.py`'s own
# Part 7 established for instance size vs. optimal scale. Grouped by family, the same grouping
# this script's own headline breakdown (`drift_summary.csv`) already uses.
size_correlation_rows = []
for family in AUGMENTATIONS:
    sub = entries_df[entries_df.family == family]
    if len(sub) < 3:
        continue
    pearson_r, pearson_p = pearsonr(sub["mask_area_frac"], sub["similarity"])
    spearman_r, spearman_p = spearmanr(sub["mask_area_frac"], sub["similarity"])
    size_correlation_rows.append(
        {
            "family": family,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "n_samples": len(sub),
        }
    )
size_correlation_df = pd.DataFrame(size_correlation_rows)
size_correlation_df.to_csv(OUTPUT_DIR / "size_correlation.csv", index=False)

# Object-size terciles (global, computed once across every row so the same size cutoffs apply
# everywhere) x similarity, faceted by family — does the smallest third of instances lose more
# similarity than the largest third under the same augmentation?
try:
    entries_df["size_tercile"] = pd.qcut(
        entries_df["mask_area_frac"], 3, labels=["small", "medium", "large"]
    )
except ValueError:
    log.warning(
        "mask_area_frac has too few distinct values for 3 clean terciles — falling back to "
        "qcut's own duplicate-safe binning (labels become numeric ranges, not small/medium/large)"
    )
    entries_df["size_tercile"] = pd.qcut(entries_df["mask_area_frac"], 3, duplicates="drop")

n_families = len(AUGMENTATIONS)
fig, axes = plt.subplots(1, n_families, figsize=(4 * n_families, 5), sharey=True)
for ax, family in zip(axes, AUGMENTATIONS):
    tercile_means = (
        entries_df[entries_df.family == family].groupby("size_tercile", observed=True)["similarity"]
        .mean()
    )
    tercile_means.plot(kind="bar", ax=ax, color="#2ecc71")
    ax.set_title(family, fontsize=9)
    ax.set_xlabel("object-size tercile")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("mean similarity to severity=0 (pooled across severities)")
fig.suptitle("Does object size predict augmentation robustness?")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "size_correlation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(
    "Wrote %s and %s", OUTPUT_DIR / "size_correlation.csv", OUTPUT_DIR / "size_correlation.png"
)
for _, row in size_correlation_df.iterrows():
    log.info(
        "  %-22s pearson_r=%.3f (p=%.3f)  spearman_r=%.3f (p=%.3f)  n=%d",
        row.family,
        row.pearson_r,
        row.pearson_p,
        row.spearman_r,
        row.spearman_p,
        row.n_samples,
    )

# %% Visualization — augmented crop grid, one row per family (reference instance only;
# a grid across all instances would be unreadable, so this is a qualitative sample —
# the drift-curve plot below is the one aggregated across every instance)
ref_entries = [
    e
    for e in entries
    if e["dataset"] == reference_instance["dataset"]
    and e["image_stem"] == reference_instance["image_stem"]
    and e["class"] == reference_instance["class"]
    and e["instance_id"] == reference_instance["instance_id"]
]
n_families = len(AUGMENTATIONS)
n_cols = max(len(spec["values"]) for spec in AUGMENTATIONS.values())
fig, axes = plt.subplots(n_families, n_cols, figsize=(2.6 * n_cols, 2.9 * n_families))
for row, (family, spec) in enumerate(AUGMENTATIONS.items()):
    for col, val in enumerate(spec["values"]):
        ax = axes[row, col]
        entry = next(e for e in ref_entries if e["family"] == family and e["value"] == val)
        ax.imshow(entry["img"])
        ax.set_title(f"{val} {spec['unit']}\nsim={entry['similarity']:.3f}", fontsize=8)
        ax.axis("off")
    for col in range(len(spec["values"]), n_cols):
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(family, fontsize=9)
fig.suptitle(
    f"Augmentation sweeps — {reference_instance['dataset']}/{reference_instance['image_stem']} / "
    f"{reference_instance['class']!r} instance {reference_instance['instance_id']} (mid crop)"
)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUTPUT_DIR / "augmented_crops.png", dpi=150, bbox_inches="tight")

# %% Visualization — combined drift curves, aggregated across all instances
# (x normalized to 0..1 severity fraction; shaded band = ±1 std across instances)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
colors = plt.get_cmap("tab10").colors
for i, (family, spec) in enumerate(AUGMENTATIONS.items()):
    values = np.array(spec["values"], dtype=float)
    frac = (values - values[0]) / (values[-1] - values[0])
    mean_sim = drift_summary[family]["mean"]
    std_sim = drift_summary[family]["std"]
    color = colors[i % len(colors)]
    ax.plot(frac, mean_sim, marker="o", label=family, color=color)
    ax.fill_between(frac, mean_sim - std_sim, mean_sim + std_sim, color=color, alpha=0.15)
ax.set_xlabel("severity fraction (0 = no-op, 1 = strongest tested)")
ax.set_ylabel(f"cosine similarity to severity=0 (mean ± std, n={len(instances)} instances)")
lowest = min(
    float((drift_summary[f]["mean"] - drift_summary[f]["std"]).min()) for f in AUGMENTATIONS
)
ax.set_ylim(min(0.0, lowest - 0.05), 1.02)
ax.legend(fontsize=8)
ax.set_title(
    f"Masked-object embedding drift per augmentation family (n={len(instances)} instances)"
)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "drift_curves.png", dpi=150, bbox_inches="tight")

log.info("Saved figures to %s", OUTPUT_DIR)

# %% Worst/best-N qualitative gallery — actual individual crops at one representative
# (QUALITATIVE_FAMILY, strongest severity) point, ranked by *similarity drop* rather than an
# IoU score (this script has no GT/oracle-threshold step). Uses matplotlib imshow directly
# rather than `_shared/qualitative_gallery.py`'s `save_score_gallery` helper, since that helper
# expects a GT mask this drift-based use case doesn't have.
qualitative_candidates = [
    e
    for e in entries
    if e["family"] == QUALITATIVE_FAMILY
    and e["value"] == qualitative_severity_value
    and "img" in e
]
if qualitative_candidates:
    ranked = sorted(qualitative_candidates, key=lambda e: e["similarity"])  # worst (lowest) first
    n_show = min(5, len(ranked) // 2) or 1
    worst = ranked[:n_show]
    best = ranked[-n_show:][::-1]
    fig, axes = plt.subplots(2, n_show, figsize=(2.8 * n_show, 6.2), squeeze=False)
    for col, e in enumerate(worst):
        axes[0, col].imshow(e["img"])
        axes[0, col].set_title(
            f"{e['image_stem']}/{e['class']}#{e['instance_id']}\nsim={e['similarity']:.3f}",
            fontsize=8,
        )
        axes[0, col].axis("off")
    for col in range(len(worst), n_show):
        axes[0, col].axis("off")
    for col, e in enumerate(best):
        axes[1, col].imshow(e["img"])
        axes[1, col].set_title(
            f"{e['image_stem']}/{e['class']}#{e['instance_id']}\nsim={e['similarity']:.3f}",
            fontsize=8,
        )
        axes[1, col].axis("off")
    for col in range(len(best), n_show):
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("worst (lowest sim)", fontsize=9)
    axes[1, 0].set_ylabel("best (highest sim)", fontsize=9)
    fig.suptitle(
        f"Worst/best similarity examples: family={QUALITATIVE_FAMILY!r} "
        f"value={qualitative_severity_value} (n={len(qualitative_candidates)} candidates)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "qualitative_worst_best.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(
        "Wrote %s (%d candidates, showing worst/best %d)",
        OUTPUT_DIR / "qualitative_worst_best.png",
        len(qualitative_candidates),
        n_show,
    )
else:
    log.warning(
        "No qualitative candidates collected for family=%s value=%s — check "
        "QUALITATIVE_FAMILY/QUALITATIVE_SEVERITY_INDEX",
        QUALITATIVE_FAMILY,
        qualitative_severity_value,
    )

# %% [markdown]
# ## Reading the results
#
# The family whose curve drops fastest (lowest similarity at severity fraction 1.0) is
# the perturbation this backbone/layer is *least* invariant to — worth knowing before
# relying on masked-mean exemplars for matching under that kind of real-world
# variation (e.g. if `gaussian blur` drops fast, a slightly out-of-focus camera frame
# is a bigger risk to a prototype-matching pipeline than `jpeg compression` at typical
# stream-quality settings).
#
# This script measures embedding *drift* (cosine similarity vs. the instance's own
# severity=0 embedding), not localization IoU, so the achievable-IoU addition from the
# other `fundamental/` scripts doesn't apply here — there's no oracle-threshold step to
# give a realistic counterpart to.
#
# - **`latency.csv`** — GPU-synchronized wall-clock cost (see `_shared/latency.py`) of the
#   augmented-crop encoding pass and the similarity-scoring pass, plus encoding-cache
#   hit/miss counts and hit rate. No sweep-axis plot (`latency.png`) — this script has one
#   fixed model config, not a sweep of separate configs — see the logged total instead.
# - **`drift_summary.csv`'s `ci95_lo`/`ci95_hi` columns** — a percentile bootstrap CI (2000
#   resamples across instances) on each family/severity's mean similarity, alongside the
#   plain std already reported — std alone doesn't say whether two severities' means are
#   actually distinguishable or both plausible draws from the same distribution.
# - **`augmentation_effect_significance.csv`** — an unpaired bootstrap comparison of the
#   mildest tested (non-no-op) severity's vs. the strongest tested severity's per-instance
#   similarity arrays, per family: `prob_mild_beats_strong` near 0.5 means that family's
#   apparent drop in `drift_curves.png` is not distinguishable from per-instance noise.
# - **`size_correlation.csv`/`.png`** — does an instance's own object-mask patch-coverage
#   (`mask_area_frac`, added to every `per_crop_similarity.csv` row) correlate with how much
#   similarity it loses under each augmentation family (pearson/spearman, mirroring
#   `scale_composition_adaptive_oracle.py`'s own instance-size-vs-optimal-scale check, here
#   against similarity directly since this script has no oracle_iou to correlate against)?
#   Same aggregation-can-hide-an-effect caveat as every sibling script's own size checks —
#   worth knowing whether a family that "drops fast" on average is actually a small-object
#   problem specifically.
# - **`qualitative_worst_best.png`** — actual worst-N/best-N individual crops (not an
#   average) at one representative point (`QUALITATIVE_FAMILY`'s strongest tested severity),
#   ranked by similarity drop rather than IoU (this script scores drift, not localization) —
#   built with matplotlib `imshow` directly rather than `_shared/qualitative_gallery.py`'s
#   `save_score_gallery`, which expects a GT mask this use case doesn't have.
#
# ## Other augmentation/robustness experiments worth running in this dir
#
# - **Composed perturbations** — this script applies each family independently from
#   the clean crop; real frames stack several at once (blur *and* low light *and* jpeg
#   artifacts). Worth checking whether drift is roughly additive or whether some
#   combinations compound non-linearly.
# - **Layer-wise robustness** — repeat the sweep across several `LAYER_IDX` values;
#   early blocks are closer to raw texture and likely far more blur/noise-sensitive
#   than late, more semantic blocks.
# - **Per-augmentation randomness spread** — the shaded band in `drift_curves.png` is
#   spread *across instances* at one fixed seed; color jitter and noise still use one
#   fixed seed per severity, so it says nothing about *this one draw's* variance.
#   Running several seeds per level and plotting a second band would separate "this
#   augmentation's typical effect" from "this one draw's effect."
# - **Occlusion** — progressively mask out a growing fraction of the instance's own
#   patches (independent of any pixel-level augmentation) — flagged in
#   `scale_crop_similarity.py` too, and complements this file's perturbation set.


# %%
