# %% [markdown]
# # Open-Vocabulary Generalized Hough Transform
#
# A training-free instance-localization pipeline that swaps the classic Generalized
# Hough Transform's gradient-orientation R-table for dense semantic feature tokens
# (as would come from a DINOv3 patch encoder). Everything here runs on synthetic
# feature tensors so the voting geometry can be inspected in isolation, with no
# model weights or images required. Pipeline:
#
#   1. Synthesize a template object and a cluttered scene as L2-normalised D=384
#      patch-token grids, standing in for real DINOv3 features.
#   2. Dense cosine-similarity matching between every template patch and every
#      scene patch, thresholded to keep only confident correspondences.
#   3. Build an R-table: each template patch's displacement to the object center
#      ("semantic compass"), independent of the scene.
#   4. Cast votes from every confident match into a 2D accumulator, using the
#      R-table displacement to project a candidate object center.
#   5. Detect peaks in the accumulator (thresholding + DBSCAN) to recover the
#      discovered instance centers.

# %% Logging — must be before torch import
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("open_vocab_hough_transform")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# %% Parameters
SEED = 42

D = 384  # feature dim, standardized to match a DINOv3-small token width
N_CLASSES = 5  # size of the synthetic "semantic vocabulary" bank
NOISE_SIGMA = 0.08  # unit-norm noise magnitude mixed into each token before re-normalizing

TEMPLATE_H, TEMPLATE_W = 6, 6
SCENE_H, SCENE_W = 30, 40

# Top-left (row, col) anchors for pasted template instances (kept non-overlapping).
INSTANCE_ANCHORS = [(3, 4), (12, 20), (20, 30)]

MATCH_THRESHOLD = 0.75  # cosine similarity cutoff for a "confident" dense match

ACCUMULATOR_PAD = max(TEMPLATE_H, TEMPLATE_W)  # margin so off-scene votes don't clip
PEAK_FRACTION = 0.35  # accumulator bins above this fraction of the global max seed peak search
DBSCAN_EPS = 2.5  # patches; spatial radius for grouping accumulator hits into one peak
DBSCAN_MIN_SAMPLES = 3

torch.manual_seed(SEED)

# %% [markdown]
# ## 1. Setup & Synthetic Data
#
# The template is a 6×6 concentric-ring pattern built from a small bank of random
# unit "semantic class" vectors — the border ring, inner ring, and center square each
# get their own class. Repeating a class across many template patches deliberately
# mirrors how classical GHT bins many edge points into the same gradient-orientation
# bucket, so later steps have to cope with ambiguous, many-to-many matches. The scene
# is mostly independent random clutter with three noisy copies of the template pasted
# in at known anchors (kept aside as ground truth for later comparison).

# %% Build the semantic class bank and template
class_bank = F.normalize(torch.randn(N_CLASSES, D), dim=-1)

template_classes = np.zeros((TEMPLATE_H, TEMPLATE_W), dtype=int)
rows, cols = np.indices((TEMPLATE_H, TEMPLATE_W))
is_border = (rows == 0) | (rows == TEMPLATE_H - 1) | (cols == 0) | (cols == TEMPLATE_W - 1)
is_inner_ring = ~is_border & (
    (rows <= 1) | (rows >= TEMPLATE_H - 2) | (cols <= 1) | (cols >= TEMPLATE_W - 2)
)
template_classes[is_border] = 0
template_classes[is_inner_ring] = 1
template_classes[~(is_border | is_inner_ring)] = 2

template = torch.empty(TEMPLATE_H, TEMPLATE_W, D)
for i in range(TEMPLATE_H):
    for j in range(TEMPLATE_W):
        base = class_bank[template_classes[i, j]]
        template[i, j] = base + NOISE_SIGMA * F.normalize(torch.randn(D), dim=0)
template = F.normalize(template, dim=-1)
log.info(
    "Template: shape=%s  classes used=%s", tuple(template.shape), sorted(set(template_classes.flat))
)

# %% Build the cluttered scene and paste in template instances
scene = F.normalize(torch.randn(SCENE_H, SCENE_W, D), dim=-1)

ground_truth_centers: list[tuple[float, float]] = []
for r0, c0 in INSTANCE_ANCHORS:
    for i in range(TEMPLATE_H):
        for j in range(TEMPLATE_W):
            base = class_bank[template_classes[i, j]]
            scene[r0 + i, c0 + j] = base + NOISE_SIGMA * F.normalize(torch.randn(D), dim=0)
    ground_truth_centers.append((r0 + (TEMPLATE_H - 1) / 2, c0 + (TEMPLATE_W - 1) / 2))
scene = F.normalize(scene, dim=-1)
log.info(
    "Scene: shape=%s  |  %d template instances pasted at %s",
    tuple(scene.shape),
    len(INSTANCE_ANCHORS),
    INSTANCE_ANCHORS,
)
log.info(
    "Ground-truth centers (row, col): %s",
    [(round(r, 1), round(c, 1)) for r, c in ground_truth_centers],
)

# %% Visualize the feature maps (PCA → RGB, since D=384 can't be viewed directly)
template_flat = template.reshape(-1, D)
scene_flat = scene.reshape(-1, D)

pca = PCA(n_components=3)
pca.fit(torch.cat([template_flat, scene_flat], dim=0).numpy())


def pca_to_rgb(tokens: torch.Tensor, h: int, w: int) -> np.ndarray:
    proj = pca.transform(tokens.numpy())
    for c in range(3):
        lo, hi = proj[:, c].min(), proj[:, c].max()
        proj[:, c] = (proj[:, c] - lo) / (hi - lo + 1e-8)
    return proj.reshape(h, w, 3)


template_rgb = pca_to_rgb(template_flat, TEMPLATE_H, TEMPLATE_W)
scene_rgb = pca_to_rgb(scene_flat, SCENE_H, SCENE_W)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].imshow(template_classes, cmap="tab10", vmin=0, vmax=9)
axes[0].set_title("Template — ground-truth class id", fontsize=10)
axes[0].set_xticks(range(TEMPLATE_W))
axes[0].set_yticks(range(TEMPLATE_H))

axes[1].imshow(template_rgb)
axes[1].set_title(
    f"Template — PCA(tokens) → RGB\n{TEMPLATE_H}×{TEMPLATE_W} patches, D={D}", fontsize=10
)
axes[1].set_xticks(range(TEMPLATE_W))
axes[1].set_yticks(range(TEMPLATE_H))

axes[2].imshow(scene_rgb)
for (r0, c0), (cr, cc) in zip(INSTANCE_ANCHORS, ground_truth_centers):
    axes[2].add_patch(
        mpatches.Rectangle(
            (c0 - 0.5, r0 - 0.5),
            TEMPLATE_W,
            TEMPLATE_H,
            fill=False,
            edgecolor="white",
            linestyle="--",
            linewidth=1.5,
        )
    )
    axes[2].scatter([cc], [cr], c="white", marker="x", s=60, linewidths=2)
axes[2].set_title(
    f"Scene — PCA(tokens) → RGB\n{SCENE_H}×{SCENE_W} patches (dashed = ground truth)", fontsize=10
)

plt.suptitle("Step 1: Synthetic feature maps", fontsize=13, y=1.03)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Dense Semantic Matching
#
# Every template patch is compared against every scene patch with cosine similarity.
# Since both sides are L2-normalised, this is a single matrix multiply. Matches above
# `MATCH_THRESHOLD` are kept as candidate correspondences — deliberately many-to-many,
# since several template patches share a class.

# %% Pairwise cosine similarity and thresholded matches
similarity = template_flat @ scene_flat.T  # (N_template, N_scene), cosine similarity
match_mask = similarity > MATCH_THRESHOLD
match_pairs = torch.nonzero(
    match_mask, as_tuple=False
)  # (num_matches, 2) → (template_idx, scene_idx)

log.info(
    "Similarity matrix: %s  |  confident matches (> %.2f): %d",
    tuple(similarity.shape),
    MATCH_THRESHOLD,
    match_pairs.shape[0],
)
matches_per_template = match_mask.sum(dim=1)
log.info(
    "Matches per template patch: min=%d  max=%d  mean=%.1f",
    matches_per_template.min().item(),
    matches_per_template.max().item(),
    matches_per_template.float().mean().item(),
)

fig, axes = plt.subplots(1, 2, figsize=(16, 4))
im0 = axes[0].imshow(similarity.numpy(), cmap="viridis", vmin=-1, vmax=1, aspect="auto")
axes[0].set_title(
    f"Cosine similarity  ({TEMPLATE_H * TEMPLATE_W}×{SCENE_H * SCENE_W} patches)", fontsize=10
)
axes[0].set_xlabel("scene patch index")
axes[0].set_ylabel("template patch index")
plt.colorbar(im0, ax=axes[0], shrink=0.85)

im1 = axes[1].imshow(match_mask.numpy(), cmap="Greys", vmin=0, vmax=1, aspect="auto")
axes[1].set_title(f"Confident matches (> {MATCH_THRESHOLD})", fontsize=10)
axes[1].set_xlabel("scene patch index")
plt.colorbar(im1, ax=axes[1], shrink=0.85)

plt.suptitle("Step 2: Dense semantic matching — affinity matrix", fontsize=13, y=1.05)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. R-Table & The Semantic Compass
#
# The R-table maps each template patch to the displacement vector that would carry it
# to the object center, computed once from the template geometry alone (no scene
# involved). At vote time, that same displacement is applied starting from wherever
# the matching scene patch sits — exactly the classic GHT R-table, indexed by
# semantic identity instead of gradient orientation.

# %% Build the R-table (template-patch idx → displacement to object center)
center_row = (TEMPLATE_H - 1) / 2
center_col = (TEMPLATE_W - 1) / 2

template_rows = torch.arange(TEMPLATE_H * TEMPLATE_W) // TEMPLATE_W
template_cols = torch.arange(TEMPLATE_H * TEMPLATE_W) % TEMPLATE_W
r_table_dy = center_row - template_rows.float()  # (N_template,)
r_table_dx = center_col - template_cols.float()

for t in range(0, TEMPLATE_H * TEMPLATE_W, 7):  # a sparse sample across the grid
    log.info(
        "R-table  patch (row=%d, col=%d, class=%d) → displacement (dy=%.1f, dx=%.1f)",
        template_rows[t].item(),
        template_cols[t].item(),
        template_classes[template_rows[t].item(), template_cols[t].item()],
        r_table_dy[t].item(),
        r_table_dx[t].item(),
    )

# %% Verify the math on the true correspondences for one known instance
# (True correspondence = the exact template patch that was pasted at each scene position;
# the many *other* above-threshold matches from shared classes are the noise votes that
# Steps 4-5 have to see through.)
verify_r0, verify_c0 = INSTANCE_ANCHORS[0]
verify_center = ground_truth_centers[0]
scene_rows_all = match_pairs[:, 1] // SCENE_W
scene_cols_all = match_pairs[:, 1] % SCENE_W
in_instance_0 = (
    (scene_rows_all >= verify_r0)
    & (scene_rows_all < verify_r0 + TEMPLATE_H)
    & (scene_cols_all >= verify_c0)
    & (scene_cols_all < verify_c0 + TEMPLATE_W)
)
log.info(
    "Verifying against instance 0, ground-truth center=(%.1f, %.1f)  |  %d candidate matches "
    "land inside its footprint (true + ambiguous noise votes):",
    *verify_center,
    int(in_instance_0.sum()),
)
for t in range(0, TEMPLATE_H * TEMPLATE_W, 7):  # same sparse sample as the R-table above
    ti, tj = template_rows[t].item(), template_cols[t].item()
    sy, sx = verify_r0 + ti, verify_c0 + tj
    dy, dx = r_table_dy[t].item(), r_table_dx[t].item()
    pred_row, pred_col = sy + dy, sx + dx
    log.info(
        "  scene patch (row=%d, col=%d) + displacement (dy=%.1f, dx=%.1f) → predicted center "
        "(%.1f, %.1f)",
        sy,
        sx,
        dy,
        dx,
        pred_row,
        pred_col,
    )

# %% Visualize the R-table as a "semantic compass" over the template
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.imshow(template_classes, cmap="tab10", vmin=0, vmax=9, alpha=0.35)
colors = plt.get_cmap("tab10")(template_classes.reshape(-1) / 9)
ax.quiver(
    template_cols.numpy(),
    template_rows.numpy(),
    r_table_dx.numpy(),
    r_table_dy.numpy(),
    color=colors,
    angles="xy",
    scale_units="xy",
    scale=1,
    width=0.006,
)
ax.scatter(
    [center_col], [center_row], c="black", marker="*", s=200, zorder=5, label="object center"
)
ax.set_xlim(-1, TEMPLATE_W)
ax.set_ylim(TEMPLATE_H, -1)
ax.set_xlabel("column")
ax.set_ylabel("row")
ax.legend(loc="upper right", fontsize=8)
ax.set_title("Step 3: R-table — each patch's vote direction to center", fontsize=11)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. 2D Accumulator Voting Space
#
# Every confident match casts one weighted vote (weight = cosine similarity) into a
# 2D accumulator, landing at `scene_coord + R-table displacement`. A padded margin
# absorbs any vote that overshoots the scene bounds. Correct matches from the same
# instance reinforce the true center; class-sharing mismatches scatter votes
# elsewhere, so we expect a few sharp peaks against a diffuse noise floor.

# %% Cast votes into the accumulator
pad = ACCUMULATOR_PAD
acc = torch.zeros(SCENE_H + 2 * pad, SCENE_W + 2 * pad)

t_idx, s_idx = match_pairs[:, 0], match_pairs[:, 1]
scene_rows = (s_idx // SCENE_W).float()
scene_cols = (s_idx % SCENE_W).float()
vote_rows = torch.round(scene_rows + r_table_dy[t_idx]).long() + pad
vote_cols = torch.round(scene_cols + r_table_dx[t_idx]).long() + pad
vote_rows = vote_rows.clamp(0, acc.shape[0] - 1)
vote_cols = vote_cols.clamp(0, acc.shape[1] - 1)
vote_weights = similarity[t_idx, s_idx]

acc.index_put_((vote_rows, vote_cols), vote_weights, accumulate=True)
log.info(
    "Accumulator: shape=%s (padded by %d)  |  %d votes cast  |  max=%.2f",
    tuple(acc.shape),
    pad,
    match_pairs.shape[0],
    acc.max().item(),
)

# %% Visualize the accumulator
acc_np = acc.numpy()
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(acc_np, cmap="hot", aspect="auto")
for cr, cc in ground_truth_centers:
    ax.scatter([cc + pad], [cr + pad], c="cyan", marker="x", s=80, linewidths=2)
ax.set_title(
    f"Step 4: Accumulator (hot) — cyan × = ground-truth centers  |  {match_pairs.shape[0]} votes",
    fontsize=11,
)
ax.set_xlabel("padded column")
ax.set_ylabel("padded row")
plt.colorbar(im, ax=ax, shrink=0.85, label="accumulated vote weight")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Peak Detection & Extraction
#
# Accumulator bins above `PEAK_FRACTION` of the global max are treated as peak
# candidates. DBSCAN then groups spatially-nearby candidates (weighted by vote
# strength) into single detections, discarding scattered low-density noise votes as
# outliers — turning the diffuse accumulator into a short list of discovered centers.

# %% Threshold + DBSCAN clustering to extract peaks
thresh_val = PEAK_FRACTION * acc_np.max()
ys, xs = np.where(acc_np >= thresh_val)
weights = acc_np[ys, xs]
candidates = np.stack([ys, xs], axis=1).astype(float)
log.info(
    "Peak candidates above %.2f (%.0f%% of max): %d bins",
    thresh_val,
    PEAK_FRACTION * 100,
    len(candidates),
)

detections: list[tuple[float, float, float]] = []  # (row, col) in *scene* coords, vote strength
if len(candidates):
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(
        candidates, sample_weight=weights
    )
    n_clusters = len(set(labels) - {-1})
    log.info(
        "DBSCAN: %d peak cluster(s)  |  %d noise bin(s)", n_clusters, int((labels == -1).sum())
    )
    for cid in sorted(set(labels) - {-1}):
        m = labels == cid
        w = weights[m]
        centroid = (candidates[m] * w[:, None]).sum(axis=0) / w.sum()
        detections.append((centroid[0] - pad, centroid[1] - pad, float(w.sum())))
    detections.sort(key=lambda d: -d[2])

for i, (row, col, strength) in enumerate(detections):
    nearest_gt = min(ground_truth_centers, key=lambda gt: (gt[0] - row) ** 2 + (gt[1] - col) ** 2)
    err = float(np.hypot(nearest_gt[0] - row, nearest_gt[1] - col))
    log.info(
        "Detection P%d: center=(%.1f, %.1f)  vote strength=%.1f  |  nearest ground truth "
        "(%.1f, %.1f), error=%.2f patches",
        i + 1,
        row,
        col,
        strength,
        *nearest_gt,
        err,
    )
if not detections:
    log.warning("No peaks survived clustering — lower PEAK_FRACTION or MATCH_THRESHOLD.")

# %% Visualize final detections on the accumulator and on the scene
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].imshow(acc_np, cmap="hot", aspect="auto")
for cr, cc in ground_truth_centers:
    axes[0].scatter([cc + pad], [cr + pad], c="cyan", marker="x", s=80, linewidths=2)
for i, (row, col, _) in enumerate(detections):
    axes[0].scatter([col + pad], [row + pad], c="lime", marker="+", s=200, linewidths=2.5)
    axes[0].annotate(
        f"P{i + 1}",
        (col + pad, row + pad),
        color="lime",
        fontsize=11,
        weight="bold",
        xytext=(6, 6),
        textcoords="offset points",
    )
axes[0].set_title("Accumulator + detected peaks", fontsize=11)
axes[0].set_xlabel("padded column")
axes[0].set_ylabel("padded row")

axes[1].imshow(scene_rgb)
for i, (row, col, _) in enumerate(detections):
    axes[1].scatter([col], [row], c="lime", marker="+", s=200, linewidths=2.5)
    axes[1].annotate(
        f"P{i + 1}",
        (col, row),
        color="lime",
        fontsize=11,
        weight="bold",
        xytext=(6, 6),
        textcoords="offset points",
    )
for cr, cc in ground_truth_centers:
    axes[1].scatter([cc], [cr], c="cyan", marker="x", s=60, linewidths=2)
axes[1].set_title(
    f"Scene — {len(detections)} instance(s) detected (lime) vs ground truth (cyan)", fontsize=11
)

plt.suptitle("Step 5: Peak detection via thresholding + DBSCAN", fontsize=13, y=1.03)
plt.tight_layout()
plt.show()

# %%
