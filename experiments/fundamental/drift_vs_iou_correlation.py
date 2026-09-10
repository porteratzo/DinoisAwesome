# %% [markdown]
# # Fundamental: Does Embedding Drift Predict Downstream IoU Loss?
#
# `scale_crop_similarity.py` -> `augmentation_sensitivity.py` -> `augmented_prototype_oracle_iou_
# knn_fgbg.py` is explicitly framed (see `experiments/README.md`) as a three-script progression:
# does perturbing an exemplar move its embedding (`augmentation_sensitivity.py`'s
# `drift_summary.csv`, one row per augmentation family x severity, cosine similarity to the
# unperturbed crop), and — separately — does perturbing the exemplar *before* pooling it into a
# gallery change downstream localization (`augmented_prototype_oracle_iou_knn_fgbg.py`'s
# `composed_endpoint.csv`, one row per method x scale x family, oracle IoU delta vs. an
# unaugmented baseline)? Until now, nothing actually joins those two CSVs and checks whether the
# first number *predicts* the second — the reader has been left to eyeball two PNGs in different
# output folders and judge for themselves whether "more drift" lines up with "worse IoU."
#
# This script joins them on `family` (rotation, illumination/gamma, color jitter, blur, noise,
# jpeg — the same six augmentation families both scripts share) and computes a Pearson/Spearman
# correlation between each family's worst-case embedding drift and its accuracy impact, for every
# (method, scale) combination in `composed_endpoint.csv`. **Caveat up front: there are only 6
# augmentation families, so every correlation here has n=6 — treat the p-values as suggestive,
# not confirmatory; the scatter plot itself (which family is the outlier, if any) is more
# informative than the correlation coefficient alone at this sample size.**
#
# Needs both scripts already run at least once: `outputs/fundamental_abc5/augmentation_
# sensitivity/drift_summary.csv` and `outputs/fundamental_abc5/augmented_prototype_oracle_iou_
# knn_fgbg/composed_endpoint.csv`.

# %% Logging — must be before torch import (kept for CLAUDE.md consistency across this
# directory's scripts even though this one never imports torch itself — pure CSV post-analysis)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("drift_vs_iou_correlation")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

DRIFT_CSV = (
    _REPO_ROOT / "outputs" / "fundamental_abc5" / "augmentation_sensitivity" / "drift_summary.csv"
)
IOU_CSV = (
    _REPO_ROOT
    / "outputs"
    / "fundamental_abc5"
    / "augmented_prototype_oracle_iou_knn_fgbg"
    / "composed_endpoint.csv"
)

OUTPUT_DIR = _REPO_ROOT / "outputs" / "fundamental_abc5" / "drift_vs_iou_correlation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for csv_path, script_name in (
    (DRIFT_CSV, "augmentation_sensitivity.py"),
    (IOU_CSV, "augmented_prototype_oracle_iou_knn_fgbg.py"),
):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} does not exist — run {script_name} at least once first "
            "(this script is pure post-hoc analysis over its saved CSV, not a fresh sweep)."
        )

# %% Part 1 — per-family worst-case drift. drift_summary.csv is one row per (family, severity);
# reduce each family to a single "how much did the worst severity move the embedding" number
# (1 - the lowest mean cosine similarity across that family's severities) so it can join against
# composed_endpoint.csv's one-row-per-family granularity.
drift_df = pd.read_csv(DRIFT_CSV)
drift_by_family = (
    drift_df.groupby("family")["mean_similarity"]
    .min()
    .rename("min_mean_similarity")
    .reset_index()
)
drift_by_family["drift_magnitude"] = 1.0 - drift_by_family["min_mean_similarity"]
log.info(
    "Per-family worst-case drift (1 - min mean cosine similarity across severities):\n%s",
    drift_by_family.to_string(index=False),
)

# %% Part 2 — join against composed_endpoint.csv's per-(method, scale, family) accuracy impact,
# correlate drift_magnitude against mean_delta_vs_baseline for every (method, scale) slice.
iou_df = pd.read_csv(IOU_CSV)
merged = iou_df.merge(drift_by_family, on="family", how="inner")
if merged["family"].nunique() < iou_df["family"].nunique():
    log.warning(
        "Only %d/%d augmentation families matched between the two CSVs by name — check both "
        "scripts still define AUGMENTATIONS identically before trusting this join.",
        merged["family"].nunique(),
        iou_df["family"].nunique(),
    )
merged.to_csv(OUTPUT_DIR / "drift_vs_iou_joined.csv", index=False)

correlation_rows = []
for method in sorted(merged["method"].unique()):
    for scale in sorted(merged["scale"].unique()):
        sub = merged[(merged.method == method) & (merged.scale == scale)]
        if len(sub) < 3:
            continue
        pearson_r, pearson_p = pearsonr(sub["drift_magnitude"], sub["mean_delta_vs_baseline"])
        spearman_r, spearman_p = spearmanr(sub["drift_magnitude"], sub["mean_delta_vs_baseline"])
        correlation_rows.append(
            {
                "method": method,
                "scale": scale,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "n_families": len(sub),
            }
        )
correlation_df = pd.DataFrame(correlation_rows)
correlation_df.to_csv(OUTPUT_DIR / "drift_vs_iou_correlation.csv", index=False)
log.info(
    "Drift-vs-IoU-loss correlation (n=%d augmentation families per slice, so treat p-values as "
    "suggestive):",
    merged["family"].nunique(),
)
for _, row in correlation_df.iterrows():
    log.info(
        "  method=%-13s scale=%-8s pearson_r=%+.3f (p=%.3f)  spearman_r=%+.3f (p=%.3f)",
        row.method,
        row.scale,
        row.pearson_r,
        row.pearson_p,
        row.spearman_r,
        row.spearman_p,
    )
log.info("Wrote %s and %s", OUTPUT_DIR / "drift_vs_iou_joined.csv", OUTPUT_DIR / "drift_vs_iou_correlation.csv")

# %% Part 3 — scatter plot, one panel per method, points colored/labeled by family, one series
# per scale. A family that sits far off the trend line the other five suggest is itself a
# finding: drift magnitude doesn't uniformly predict IoU loss for that perturbation type.
methods = sorted(merged["method"].unique())
scales = sorted(merged["scale"].unique())
scale_markers = {scale: marker for scale, marker in zip(scales, ["o", "s", "^", "D", "v", "P"])}

fig, axes = plt.subplots(1, len(methods), figsize=(6.5 * len(methods), 5.5), sharey=True)
if len(methods) == 1:
    axes = [axes]
for ax, method in zip(axes, methods):
    sub = merged[merged.method == method]
    for scale in scales:
        scale_sub = sub[sub.scale == scale]
        ax.scatter(
            scale_sub["drift_magnitude"],
            scale_sub["mean_delta_vs_baseline"],
            marker=scale_markers[scale],
            s=70,
            label=scale,
            edgecolors="black",
        )
        for _, row in scale_sub.iterrows():
            ax.annotate(row["family"], (row["drift_magnitude"], row["mean_delta_vs_baseline"]), fontsize=7)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("embedding drift magnitude (1 - min mean cosine similarity)")
    ax.set_title(method)
    ax.legend(fontsize=8, title="scale")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("mean oracle IoU delta vs. unaugmented baseline")
fig.suptitle("Does more embedding drift predict worse downstream localization?")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "drift_vs_iou_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Wrote %s", OUTPUT_DIR / "drift_vs_iou_scatter.png")

# %% [markdown]
# ## Reading the results
#
# - **`drift_vs_iou_correlation.csv`/`.png`** answer the question the three-script README
#   progression poses but never itself computes: within each (method, scale) slice, does a
#   family's worst-case embedding drift (`augmentation_sensitivity.py`) predict its IoU cost
#   under augmented-prototype matching (`augmented_prototype_oracle_iou_knn_fgbg.py`)? A strong
#   negative `pearson_r`/`spearman_r` (more drift -> more negative `mean_delta_vs_baseline`)
#   would validate embedding drift as a cheap proxy for localization risk, worth checking before
#   running the full oracle-IoU pipeline on a new augmentation family. A weak or inconsistent
#   correlation means drift magnitude alone doesn't predict downstream harm — some perturbation
#   types could move the embedding a lot but barely hurt matching (or the reverse).
# - **n=6 families per correlation** — this is a small-sample correlation by construction (there
#   are only 6 augmentation families in `AUGMENTATIONS`); a single outlier family can swing
#   `pearson_r` substantially. Read the scatter plot's actual point layout, not just the
#   correlation coefficient, before concluding "drift predicts IoU loss" one way or the other.
# - **`drift_vs_iou_joined.csv`** is the underlying per-(method, scale, family) join if you want
#   to re-aggregate differently (e.g. pool across scale, or restrict to one method).

# %%
