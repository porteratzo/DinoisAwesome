# %% [markdown]
# # Fundamental: Classification-Head Architecture x Layer-Depth Ablation
#
# Every scoring method used elsewhere in `fundamental/` is training-free: cosine similarity
# against a mean prototype, or a contrastive kNN. This script asks a different question —
# **given labeled patches (foreground vs. background, abc5's own instance masks), does a
# *fitted* classification head beat plain cosine similarity, and does that answer change with
# which transformer block(s) the patch tokens come from?**
#
# Four heads, same frozen DINOv3-base patch tokens as input:
#
# | Head | What it is | Fit source |
# |---|---|---|
# | `cosine` | nearest fg/bg centroid, L2-normalized train-patch means | centroids, no fitting |
# | `linear_probe` | `LogisticRegression`, `class_weight="balanced"` (bg outnumbers fg) | fit/fold |
# | `svm` | `LinearSVC`, linear kernel (tests head architecture, not kernel choice) | supervised |
# | `knn_fgbg` | contrastive kNN vs. pooled fg/bg patch banks (this repo's best method) | no fit |
#
# **`knn_fgbg` is scored differently from the other three, and that's deliberate, not an
# oversight:** it's a continuous score, not a hard label, so — matching every other
# training-free method's convention throughout this directory — it's scored as **oracle
# IoU**: the best patch IoU any single global threshold on that score could achieve against
# *this eval image's own GT* (`_shared.thresholding.iou_tuned_threshold`/`oracle_iou`). That
# means `knn_fgbg`'s numbers are an upper bound that legitimately gets to see the eval labels
# to pick its operating point, while `cosine`/`linear_probe`/`svm` never see eval labels at
# all (their decision boundary is fixed at fit time). Read `knn_fgbg` as "the best this
# repo's existing method could ever do here," not as a directly comparable deployment-time
# operating point next to the other three.
#
# Four transformer blocks of DINOv3-base (depth 12), roughly evenly spaced across the
# beginning/middle/near-end/last of the stack:
#
# | Name | Block index |
# |---|---|
# | `early` | 2 |
# | `mid` | 5 |
# | `late` | 9 |
# | `last` | 11 |
#
# `DinoEncoder.forward(layers=[...])` already returns every requested block in one pass
# (`ExtractorOutput.patches` shape `(B, L, H, W, D)`), so all four blocks are extracted from a
# single encode per image — no re-encoding per layer or per combo. Each layer's patch-token
# grid is L2-normalized independently (so no single layer's raw magnitude dominates), and
# **layer combinations are formed by concatenating** those per-layer-normalized vectors along
# the feature dimension — every non-empty subset of the 4 named layers (15 combos: 4 singles +
# 6 pairs + 4 triples + 1 all-four).
#
# ## Task and evaluation
#
# The label is the same fg/bg convention every sibling `fundamental/` script already uses
# (`MASK_PATCH_THRESHOLD=0.3` on the instance-type-group mask, via
# `_shared.mask_geometry.pixel_mask_to_patch_mask`) — here read as a **per-patch binary
# classification** target, one task per (part_type, instance-type group), rather than as a
# gallery to match a query heatmap against.
#
# abc5 has only 8 images per part type, but each image contributes its *whole patch grid*
# (48x48 = 2304 patches at 768px), so a fitted head sees thousands of labeled patches per fold
# even though the image count is tiny — unlike an image-level classification probe, which
# would only get 5 labeled examples per fold. Still cross-validated the same way every other
# `fundamental/` 5-3 addition is (`_shared/pooled_gallery_cv.py`: 5 images/part type pooled
# into the fit, scored against the 3 held out, 5 folds with a fresh random shuffle each) —
# partly for direct comparability with those scripts, partly because a *single* fixed split
# would still risk reporting one lucky/unlucky shuffle as if it were a stable result.
#
# Metrics are hard-decision (accuracy/precision/recall/F1/IoU) rather than the oracle-tuned
# continuous-score IoU other scripts report: all three heads here produce a discrete label per
# patch (cosine included — argmax of the two centroid similarities), so there is no
# threshold left to tune against ground truth.

# %% Logging — must be before torch import
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("head_layer_ablation")

from collections.abc import Callable
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from tqdm import tqdm

from dinoisawesome import DinoEncoder, EncoderWithCache
from dinoisawesome.abc3 import PART_TYPES

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared.mask_geometry import pixel_mask_to_patch_mask  # noqa: E402
from _shared.pooled_gallery_cv import (  # noqa: E402
    N_FOLDS_53,
    discover_all_instances,
    make_fold_role_splits,
)
from _shared.prototype_ops import knn_fgbg_score  # noqa: E402
from _shared.run_config import apply_overrides, load_run_config, resolve_output_dir  # noqa: E402
from _shared.stats import bootstrap_ci  # noqa: E402
from _shared.thresholding import iou_tuned_threshold  # noqa: E402

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

DATA_ROOT = _REPO_ROOT / "data"
DATASET = "abc5"

DINO_VERSION = "v3"
DINO_SIZE = "base"
IMG_SIZE = 768
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")
DINO_ENCODING_CACHE_DIR: str | None = os.environ.get("DINO_ENCODING_CACHE_DIR")

# Beginning/middle/near-end/last of DINOv3-base's 12-block stack (see module docstring).
# Re-deriving these for a different DINO_SIZE means re-picking indices by hand — block count
# isn't queried from the encoder here because layer names/roles (not raw indices) are the
# axis under test.
LAYER_IDX_BY_NAME: dict[str, int] = {"early": 2, "mid": 5, "late": 9, "last": 11}
LAYER_NAMES: list[str] = list(LAYER_IDX_BY_NAME)
LAYER_INDICES: list[int] = list(LAYER_IDX_BY_NAME.values())

# Every non-empty subset of the 4 named layers, concatenated in LAYER_NAMES order.
LAYER_COMBOS: dict[str, tuple[str, ...]] = {
    "+".join(combo): combo
    for r in range(1, len(LAYER_NAMES) + 1)
    for combo in combinations(LAYER_NAMES, r)
}

# The 3 fitted heads (blind at eval time) and the "knn_fgbg" training-free baseline (scored
# as oracle IoU, which legitimately sees eval labels — see module docstring) are swept
# together but scored differently below; FITTED_HEADS drives the shared fit/predict loop,
# HEADS (superset, used for summary/plotting) fixes their on-chart order.
FITTED_HEADS: list[str] = ["cosine", "linear_probe", "svm"]
BASELINE_HEAD = "knn_fgbg"
HEADS: list[str] = [*FITTED_HEADS, BASELINE_HEAD]
HEAD_COLOR: dict[str, str] = {
    "cosine": "#7f8c8d",
    "linear_probe": "#3498db",
    "svm": "#e67e22",
    "knn_fgbg": "#2ecc71",  # same green feature_transform_oracle_iou.py's own METHOD_COLOR uses
}

MASK_PATCH_THRESHOLD = 0.3
DEBIAS = True  # matches every sibling fundamental script's default (see debias_ablation.py)

# knn_fgbg baseline: same defaults every sibling fundamental script uses.
KNN_FGBG_NUM_NEIGHBOURS = 10
ORACLE_THRESHOLD_STEPS = 25

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0
SEED = 0

apply_overrides(globals(), load_run_config(__file__))
torch.manual_seed(SEED)

OUTPUT_DIR = resolve_output_dir(_REPO_ROOT / "outputs" / "fundamental_abc5" / "head_layer_ablation")

log.info(
    "dataset=%s part_types=%s  |  DINO%s-%s img_size=%d layers=%s  |  heads=%s combos=%d",
    DATASET,
    PART_TYPES,
    DINO_VERSION,
    DINO_SIZE,
    IMG_SIZE,
    LAYER_IDX_BY_NAME,
    HEADS,
    len(LAYER_COMBOS),
)

fold_splits = make_fold_role_splits(PART_TYPES, seed=SEED, n_folds=N_FOLDS_53)

# %% Part 1 — discover every abc5 instance + each image's per-group GT mask
# (`_shared.pooled_gallery_cv.discover_all_instances`, the same generic discovery every other
# fundamental script's own 5-3 section reuses). Only the images/gt_masks are needed here —
# whole-image patch grids are the unit of classification, not per-instance crops.
discovery = discover_all_instances(DATA_ROOT, DATASET, PART_TYPES)
if not discovery.instances:
    raise RuntimeError(f"No instances discovered under data/{DATASET} — check the data.")

groups_by_part_type: dict[str, set[str]] = {pt: set() for pt in PART_TYPES}
for part_type, group, _n in discovery.gt_masks:
    groups_by_part_type[part_type].add(group)

log.info(
    "Discovered %d instances, %d images, %d (part_type, group, image) GT masks across %d groups",
    len(discovery.instances),
    len(discovery.images),
    len(discovery.gt_masks),
    len({g for gs in groups_by_part_type.values() for g in gs}),
)

# %% Part 2 — encode every image once with every swept layer in one forward pass, L2-normalize
# each layer's patch grid independently (so concatenated combos aren't dominated by whichever
# layer happens to have larger raw magnitude), keep on CPU (patch counts here are large enough
# that holding every image x every layer on GPU at once isn't worth it for a one-off encode).
encoder = DinoEncoder(
    version=DINO_VERSION,
    size=DINO_SIZE,
    img_size=IMG_SIZE,
    layers=LAYER_INDICES,
    weights_dir=DINO_WEIGHTS_DIR,
    amp=True,
)
encoder = EncoderWithCache(encoder, cache_dir=DINO_ENCODING_CACHE_DIR)
chunk_size = encoder.max_batch_size

image_keys: list[tuple[str, int]] = sorted(discovery.images)

# Probe the patch grid shape from one image (constant across abc5 images at a fixed IMG_SIZE)
# so the main encode loop below never has to deal with an Optional grid_h/grid_w.
_probe_out = encoder(discovery.images[image_keys[0]], layers=LAYER_INDICES, debias=DEBIAS)
grid_h, grid_w = int(_probe_out.patches.shape[2]), int(_probe_out.patches.shape[3])

image_layer_tokens: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
for i in tqdm(range(0, len(image_keys), chunk_size), desc="Encoding abc5 images (all layers)"):
    chunk_keys = image_keys[i : i + chunk_size]
    out = encoder([discovery.images[k] for k in chunk_keys], layers=LAYER_INDICES, debias=DEBIAS)
    for b, key in enumerate(chunk_keys):
        image_layer_tokens[key] = {
            name: F.normalize(
                out.patches[b, li].reshape(grid_h * grid_w, -1).float(), p=2, dim=-1
            ).cpu()
            for li, name in enumerate(LAYER_NAMES)
        }

# GT patch masks, flattened to match the (grid_h * grid_w,)-shaped patch tokens above.
gt_patch_masks: dict[tuple[str, str, int], np.ndarray] = {
    key: pixel_mask_to_patch_mask(mask, grid_h, grid_w, IMG_SIZE, MASK_PATCH_THRESHOLD).reshape(-1)
    for key, mask in discovery.gt_masks.items()
}

log.info("Encoded %d images at patch grid %dx%d", len(image_layer_tokens), grid_h, grid_w)


# %% Part 3 — feature assembly + the three classification heads
def combo_features(image_key: tuple[str, int], combo: tuple[str, ...]) -> np.ndarray:
    """Concatenate this image's per-layer-normalized patch tokens for *combo*'s layers along
    the feature dim -> (grid_h * grid_w, D_per_layer * len(combo))."""
    tokens = image_layer_tokens[image_key]
    return torch.cat([tokens[name] for name in combo], dim=-1).numpy()


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def fit_predict_cosine(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    """Nearest-centroid: classify each eval patch by whichever of the (L2-normalized) mean
    fg/bg train centroids it's more cosine-similar to."""
    fg_centroid = _l2_normalize_rows(X_train[y_train].mean(axis=0, keepdims=True))[0]
    bg_centroid = _l2_normalize_rows(X_train[~y_train].mean(axis=0, keepdims=True))[0]
    X_eval_n = _l2_normalize_rows(X_eval)
    return (X_eval_n @ fg_centroid) > (X_eval_n @ bg_centroid)


def fit_predict_linear_probe(
    X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray
) -> np.ndarray:
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)
    return clf.predict(X_eval).astype(bool)


def fit_predict_svm(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    # dual=False: the primal formulation is the standard choice once n_samples > n_features,
    # true here for every LAYER_COMBOS entry (thousands of pooled train patches vs. at most
    # 4 * 768 = 3072 concatenated dims for the all-four combo).
    clf = LinearSVC(class_weight="balanced", max_iter=5000, dual=False)
    clf.fit(X_train, y_train)
    return clf.predict(X_eval).astype(bool)


HEAD_FUNCS: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {
    "cosine": fit_predict_cosine,
    "linear_probe": fit_predict_linear_probe,
    "svm": fit_predict_svm,
}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    denom_iou = tp + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0
        else float("nan")
    )
    return {
        "iou": tp / denom_iou if denom_iou > 0 else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(y_true) if len(y_true) > 0 else float("nan"),
    }


def knn_fgbg_oracle_metrics(
    X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray
) -> dict[str, float]:
    """The `knn_fgbg` baseline: continuous per-patch contrastive-kNN score against the pooled
    train fg/bg patch banks (not centroids — `_shared.prototype_ops.knn_fgbg_score`), scored
    as oracle IoU (`_shared.thresholding.iou_tuned_threshold` picks the threshold that
    maximizes IoU against *this eval image's own* `y_eval`, then `classification_metrics`
    reports every metric — including IoU — from that same threshold's hard decision, so all
    of them agree with the oracle IoU value). Unlike `HEAD_FUNCS`' fitted heads, this
    legitimately depends on `y_eval` — see the module docstring for why that's this repo's
    own established convention for training-free continuous-score methods, not a leak."""
    fg_bank = torch.from_numpy(X_train[y_train])
    bg_bank = torch.from_numpy(X_train[~y_train])
    query = torch.from_numpy(X_eval)
    score = knn_fgbg_score(query, fg_bank, bg_bank, KNN_FGBG_NUM_NEIGHBOURS)
    thr = iou_tuned_threshold(score, y_eval, ORACLE_THRESHOLD_STEPS)
    return classification_metrics(y_eval, score > thr)


# %% Part 4 — the cross-validated sweep: for every fold x part_type x group, pool the fold's
# training images' whole patch grids into (X_train, y_train), fit every (layer_combo, head) on
# that same pool, and score each held-out eval image separately.
results: list[dict] = []
n_sweep_units = N_FOLDS_53 * len(PART_TYPES)
_desc = "Part 4: fold x part_type x group x combo x head sweep"
with tqdm(total=n_sweep_units, desc=_desc) as pbar:
    for fold_idx, split in enumerate(fold_splits):
        for part_type in PART_TYPES:
            train_numbers, eval_numbers = split[part_type]
            for group in sorted(groups_by_part_type.get(part_type, [])):
                train_ns = [
                    n
                    for n in train_numbers
                    if (part_type, n) in image_layer_tokens
                    and (part_type, group, n) in gt_patch_masks
                ]
                eval_ns = [
                    n
                    for n in eval_numbers
                    if (part_type, n) in image_layer_tokens
                    and (part_type, group, n) in gt_patch_masks
                ]
                if len(train_ns) < 2 or not eval_ns:
                    log.warning(
                        "fold=%d part_type=%s group=%s: %d usable train / %d usable eval images "
                        "— skipping",
                        fold_idx,
                        part_type,
                        group,
                        len(train_ns),
                        len(eval_ns),
                    )
                    continue

                y_train_full = np.concatenate(
                    [gt_patch_masks[(part_type, group, n)] for n in train_ns], axis=0
                )
                if y_train_full.all() or not y_train_full.any():
                    log.warning(
                        "fold=%d part_type=%s group=%s: training pool has only one class "
                        "— skipping",
                        fold_idx,
                        part_type,
                        group,
                    )
                    continue

                def _record(head: str, combo_name: str, n: int, metrics: dict[str, float]) -> None:
                    results.append(
                        {
                            "fold": fold_idx,
                            "part_type": part_type,
                            "group": group,
                            "layer_combo": combo_name,
                            "head": head,
                            "eval_number": n,
                            "n_train_images": len(train_ns),
                            **metrics,
                        }
                    )

                for combo_name, combo in LAYER_COMBOS.items():
                    X_train = np.concatenate(
                        [combo_features((part_type, n), combo) for n in train_ns], axis=0
                    )
                    for head in FITTED_HEADS:
                        try:
                            for n in eval_ns:
                                y_eval = gt_patch_masks[(part_type, group, n)]
                                X_eval = combo_features((part_type, n), combo)
                                y_pred = HEAD_FUNCS[head](X_train, y_train_full, X_eval)
                                _record(head, combo_name, n, classification_metrics(y_eval, y_pred))
                        except Exception:
                            log.exception(
                                "fold=%d part_type=%s group=%s combo=%s head=%s: FAILED, skipping",
                                fold_idx,
                                part_type,
                                group,
                                combo_name,
                                head,
                            )
                    try:
                        for n in eval_ns:
                            y_eval = gt_patch_masks[(part_type, group, n)]
                            X_eval = combo_features((part_type, n), combo)
                            metrics = knn_fgbg_oracle_metrics(X_train, y_train_full, X_eval, y_eval)
                            _record(BASELINE_HEAD, combo_name, n, metrics)
                    except Exception:
                        log.exception(
                            "fold=%d part_type=%s group=%s combo=%s head=%s: FAILED, skipping",
                            fold_idx,
                            part_type,
                            group,
                            combo_name,
                            BASELINE_HEAD,
                        )
            pbar.update(1)

if not results:
    raise RuntimeError("No (fold, part_type, group, layer_combo, head) cell produced a result.")

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "metrics_per_sample.csv", index=False)
log.info(
    "Scoring complete: %d scored (fold, part_type, group, combo, head, eval_image) rows",
    len(results_df),
)

# %% Part 5 — headline summary: mean +/- std +/- bootstrap CI IoU by (layer_combo, head),
# pooled across every fold/part_type/group/eval_image.
summary_rows = []
for combo_name in LAYER_COMBOS:
    for head in HEADS:
        vals = results_df.loc[
            (results_df.layer_combo == combo_name) & (results_df.head == head), "iou"
        ].to_numpy()
        mean, ci_lo, ci_hi = bootstrap_ci(vals, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
        summary_rows.append(
            {
                "layer_combo": combo_name,
                "n_layers": len(LAYER_COMBOS[combo_name]),
                "head": head,
                "mean_iou": mean,
                "std_iou": float(np.nanstd(vals)) if len(vals) else float("nan"),
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
                "mean_f1": float(
                    np.nanmean(
                        results_df.loc[
                            (results_df.layer_combo == combo_name) & (results_df.head == head), "f1"
                        ]
                    )
                ),
                "n_samples": len(vals),
            }
        )
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / "iou_summary_by_combo_head.csv", index=False)

log.info("Headline IoU summary (mean +/- std, bootstrap 95%% CI), best-to-worst per combo:")
for combo_name in LAYER_COMBOS:
    sub = summary_df[summary_df.layer_combo == combo_name].sort_values("mean_iou", ascending=False)
    for _, row in sub.iterrows():
        log.info(
            "  combo=%-16s head=%-13s iou=%.3f+/-%.3f ci95=[%.3f,%.3f] f1=%.3f (n=%d)",
            row.layer_combo,
            row.head,
            row.mean_iou,
            row.std_iou,
            row.ci95_lo,
            row.ci95_hi,
            row.mean_f1,
            row.n_samples,
        )

fig, ax = plt.subplots(figsize=(max(14, 1.1 * len(LAYER_COMBOS)), 6.5))
combo_order = list(LAYER_COMBOS)
x = np.arange(len(combo_order))
bar_width = 0.8 / len(HEADS)
for j, head in enumerate(HEADS):
    sub = summary_df[summary_df.head == head].set_index("layer_combo").loc[combo_order]
    offset = (j - (len(HEADS) - 1) / 2) * bar_width
    ax.bar(
        x + offset,
        sub["mean_iou"],
        width=bar_width,
        yerr=sub["std_iou"],
        capsize=2,
        color=HEAD_COLOR[head],
        label=head,
    )
ax.set_xticks(x, combo_order, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("patch-level fg/bg IoU (mean +/- std, 5-3 pooled CV)")
ax.set_title(
    f"Head architecture x layer-combo ablation — DINO{DINO_VERSION}-{DINO_SIZE} "
    f"({len(PART_TYPES)} part types, {N_FOLDS_53}-fold CV)"
)
ax.set_ylim(0, 1.0)
ax.grid(alpha=0.3, axis="y")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "iou_by_combo_head.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(
    "Saved %s and %s",
    OUTPUT_DIR / "iou_summary_by_combo_head.csv",
    OUTPUT_DIR / "iou_by_combo_head.png",
)

# %% Part 6 — per-(part_type, group) breakdown: the aggregate above can hide a group-specific
# effect (same rationale as every sibling fundamental script's own per-group breakdown).
per_group_rows = []
for part_type in PART_TYPES:
    for group in sorted(groups_by_part_type.get(part_type, [])):
        sub_pg = results_df[(results_df.part_type == part_type) & (results_df.group == group)]
        if sub_pg.empty:
            continue
        for combo_name in LAYER_COMBOS:
            for head in HEADS:
                vals = sub_pg.loc[(sub_pg.layer_combo == combo_name) & (sub_pg.head == head), "iou"]
                if len(vals) == 0:
                    continue
                per_group_rows.append(
                    {
                        "part_type": part_type,
                        "group": group,
                        "layer_combo": combo_name,
                        "head": head,
                        "mean_iou": float(vals.mean()),
                        "std_iou": float(vals.std()),
                        "n_samples": len(vals),
                    }
                )
pd.DataFrame(per_group_rows).to_csv(OUTPUT_DIR / "per_group_breakdown.csv", index=False)
log.info("Wrote %s", OUTPUT_DIR / "per_group_breakdown.csv")

# %% [markdown]
# ## Reading the results
#
# - **`iou_by_combo_head.png`/`iou_summary_by_combo_head.csv`** are the headline comparison:
#   for each of the 15 layer combos, which head wins, and does any head's ranking flip between
#   a single early/mid/late/last layer and a multi-layer concatenation? A head that only wins
#   at `last` but loses everywhere else isn't "the best head" — it's tied to one layer's
#   representation.
# - Class imbalance (bg patches vastly outnumber fg patches per image) is why `class_weight=
#   "balanced"` is used for both supervised heads and why **IoU**, not accuracy, is the primary
#   metric — a trivial always-predict-background classifier would still score well over 90% on
#   accuracy while scoring 0 on IoU.
# - **`knn_fgbg` is an oracle upper bound, not a deployment-time number** — its IoU comes from
#   a threshold tuned against each eval image's own GT (this repo's standard convention for
#   every training-free continuous-score method), while `cosine`/`linear_probe`/`svm` never
#   see eval labels at all. A fitted head that *still* beats `knn_fgbg` despite that handicap
#   is a strong result; one that merely comes close is not yet a clear win.
# - **`per_group_breakdown.csv`** — check before generalizing the headline ranking to every
#   instance-type group; the same aggregation-can-hide-a-group-effect caveat every sibling
#   script's own per-group breakdown exists for.
# - This script fixes DINOv3-base at 768px and the fg/bg mask-threshold convention every
#   sibling `fundamental/` script uses — it isolates *head architecture x layer depth*, not
#   resolution, backbone size, or label threshold (see `resolution_ablation.py` and
#   `feature_transform_oracle_iou.py` for those axes).

# %%
