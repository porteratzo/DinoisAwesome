# %% [markdown]
# # Patch Token Clustering
#
# Extract DINO patch tokens from an image, build a pairwise cosine similarity matrix,
# and compare clustering algorithms by overlaying assignments on the original image.

# %% Imports and logging
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("patch_clustering")

from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.preprocessing import normalize as sk_normalize

from dinoisawesome import DinoEncoder

# %% Parameters
_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".env")

# Set IMAGE_PATH to a local file path, or leave as None to download the example.
IMAGE_PATH: Path | None = _REPO_ROOT / "data" / "tiger.jpeg"
SAMPLE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"

# Model settings
DINO_VERSION = "v3"  # "v2" or "v3"
DINO_SIZE = "large"  # "small" | "base" | "large" | "giant"
IMG_SIZE = 1024  # square input side (divisible by patch_size: 14 for v2, 16 for v3)

# Local weights dir: set DINO_WEIGHTS_DIR env var to load from disk instead of torch hub.
DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")

# Clustering parameters
N_CLUSTERS = 6  # k for KMeans, Spectral
AGGLOMERATIVE_THRESHOLD = 0.3  # cosine distance threshold for AgglomerativeClustering
DBSCAN_EPS = 0.40  # cosine distance; tune if clusters look wrong

# %% Load model
encoder = DinoEncoder(
    version=DINO_VERSION,
    size=DINO_SIZE,
    img_size=IMG_SIZE,
    weights_dir=DINO_WEIGHTS_DIR,
)
H_GRID, W_GRID = encoder.grid_h, encoder.grid_w
N_PATCHES = H_GRID * W_GRID
log.info(
    "DINOv%s-%s | patch_size=%d | grid=%dx%d (%d patches)",
    DINO_VERSION[1],
    DINO_SIZE,
    encoder.patch_size,
    H_GRID,
    W_GRID,
    N_PATCHES,
)

# %% Load image
if IMAGE_PATH is None or not Path(IMAGE_PATH).exists():
    _dl = _REPO_ROOT / "data" / "sample.jpg"
    _dl.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading sample image to %s …", _dl)
    urlretrieve(SAMPLE_URL, _dl)
    IMAGE_PATH = _dl

img = Image.open(IMAGE_PATH).convert("RGB")
log.info("Loaded image from %s  (%dx%d px)", IMAGE_PATH, *img.size)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(img)
ax.set_title("Input image")
ax.axis("off")
plt.tight_layout()
plt.show()

# %% Extract patch tokens
output = encoder(img, debias=True)
# output.patches shape: (1, H_GRID, W_GRID, D) for single-layer extraction
patches_grid = output.patches[0]  # (H, W, D)
H, W, D = patches_grid.shape
tokens = patches_grid.reshape(N_PATCHES, D)  # (N, D)
log.info("Tokens: shape=%s  dtype=%s  device=%s", tokens.shape, tokens.dtype, tokens.device)

# %% Cross-similarity and distance matrices
tokens_norm = F.normalize(tokens.float(), dim=-1)  # unit vectors
similarity = (tokens_norm @ tokens_norm.T).cpu().numpy()  # (N, N)  in [-1, 1]
distance = np.clip(1.0 - similarity, 0.0, 2.0)  # cosine distance in [0, 2]
affinity = (similarity + 1.0) / 2.0  # non-negative, for SpectralClustering

sim_offdiag = similarity.copy()
np.fill_diagonal(sim_offdiag, np.nan)
log.info(
    "Off-diagonal similarity: min=%.3f  max=%.3f  mean=%.3f",
    float(np.nanmin(sim_offdiag)),
    float(np.nanmax(sim_offdiag)),
    float(np.nanmean(sim_offdiag)),
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
im0 = axes[0].imshow(similarity, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
axes[0].set_title(f"Cosine similarity  ({N_PATCHES}×{N_PATCHES} patches)")
axes[0].set_xlabel("patch index")
axes[0].set_ylabel("patch index")
plt.colorbar(im0, ax=axes[0], shrink=0.85)

im1 = axes[1].imshow(distance, cmap="viridis", vmin=0, vmax=2, aspect="auto")
axes[1].set_title(f"Cosine distance  ({N_PATCHES}×{N_PATCHES} patches)")
axes[1].set_xlabel("patch index")
plt.colorbar(im1, ax=axes[1], shrink=0.85)
plt.tight_layout()
plt.show()

# %% Define and run clustering algorithms
tokens_l2 = sk_normalize(tokens.cpu().float().numpy())
AGGLOMERATIVE_THRESHOLD = 0.5

algo_configs: list[tuple[str, object, np.ndarray]] = [
    (
        f"DBSCAN\neps={DBSCAN_EPS}",
        DBSCAN(eps=DBSCAN_EPS, min_samples=3, metric="precomputed"),
        distance,
    ),
    (
        f"Agglomerative\naverage linkage\nthresh={AGGLOMERATIVE_THRESHOLD}",
        AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=AGGLOMERATIVE_THRESHOLD,
            metric="precomputed",
            linkage="average",
        ),
        distance,
    ),
    (
        f"Agglomerative\ncomplete linkage\nthresh={AGGLOMERATIVE_THRESHOLD}",
        AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=AGGLOMERATIVE_THRESHOLD,
            metric="precomputed",
            linkage="complete",
        ),
        distance,
    ),
    (
        f"Agglomerative\nsingle linkage\nthresh={AGGLOMERATIVE_THRESHOLD}",
        AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=AGGLOMERATIVE_THRESHOLD,
            metric="precomputed",
            linkage="single",
        ),
        distance,
    ),
    (
        f"KMeans  k={N_CLUSTERS}",
        KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42),
        tokens_l2,
    ),
    (
        f"Spectral  k={N_CLUSTERS}",
        SpectralClustering(
            n_clusters=N_CLUSTERS, affinity="precomputed", random_state=42, n_init=10
        ),
        affinity,
    ),
]

try:
    from sklearn.cluster import HDBSCAN  # noqa: PLC0415

    algo_configs.append(
        (
            "HDBSCAN\nmin_cluster_size=5",
            HDBSCAN(min_cluster_size=5, metric="precomputed"),
            distance,
        )
    )
    log.info("HDBSCAN available — added to experiment")
except ImportError:
    log.warning("HDBSCAN not available (requires scikit-learn >= 1.3)")

cluster_results: list[tuple[str, np.ndarray]] = []
for name, algo, X in algo_configs:
    labels = algo.fit_predict(X)
    n_found = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    log.info("%-44s  %d clusters  %d noise", name.replace("\n", " "), n_found, n_noise)
    cluster_results.append((name, labels))

# %% Visualise cluster assignments (full-image patches)
OVERLAY_ALPHA = 0.60
CMAP = plt.get_cmap("tab20")

display_img = np.array(img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))


def _labels_to_rgba_full(labels: np.ndarray) -> np.ndarray:
    labels_2d = labels.reshape(H_GRID, W_GRID)
    unique_clusters = sorted(c for c in set(labels_2d.flat) if c != -1)
    rgba = np.zeros((H_GRID, W_GRID, 4), dtype=np.float32)
    for i, cid in enumerate(unique_clusters):
        color = np.array(CMAP(i % 20), dtype=np.float32)
        color[3] = OVERLAY_ALPHA
        rgba[labels_2d == cid] = color
    if -1 in labels_2d:
        rgba[labels_2d == -1] = [0.05, 0.05, 0.05, OVERLAY_ALPHA]
    return rgba


def _draw_result_full(ax: plt.Axes, labels: np.ndarray, title: str) -> None:
    labels_2d = labels.reshape(H_GRID, W_GRID)
    unique_clusters = sorted(c for c in set(labels_2d.flat) if c != -1)
    n_clusters = len(unique_clusters)
    n_noise = int((labels == -1).sum())

    overlay_small = _labels_to_rgba_full(labels)
    overlay_pil = Image.fromarray((overlay_small * 255).astype(np.uint8), mode="RGBA")
    overlay_arr = np.array(overlay_pil.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) / 255.0

    ax.imshow(display_img)
    ax.imshow(overlay_arr, interpolation="nearest")

    legend_handles = [
        mpatches.Patch(color=CMAP(i % 20), label=f"cluster {cid}")
        for i, cid in enumerate(unique_clusters)
    ]
    if n_noise:
        legend_handles.append(
            mpatches.Patch(facecolor=(0.05, 0.05, 0.05), label=f"noise ({n_noise})")
        )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=6, framealpha=0.85)
    suffix = f", {n_noise} noise" if n_noise else ""
    ax.set_title(f"{title}\n{n_clusters} clusters{suffix}", fontsize=9)
    ax.axis("off")


NCOLS = 3
n_algos = len(cluster_results)
n_algo_rows = (n_algos + NCOLS - 1) // NCOLS
nrows = 1 + n_algo_rows

fig, axes = plt.subplots(nrows, NCOLS, figsize=(NCOLS * 5, nrows * 5))
axes = axes.flatten()

axes[0].imshow(display_img)
axes[0].set_title("Original image", fontsize=11)
axes[0].axis("off")

for idx, (name, labels) in enumerate(cluster_results):
    _draw_result_full(axes[NCOLS + idx], labels, name)

for i in range(NCOLS + n_algos, len(axes)):
    axes[i].axis("off")

plt.suptitle(
    f"DINO{DINO_VERSION}-{DINO_SIZE}  |  {N_PATCHES} patches  |  {IMG_SIZE}×{IMG_SIZE}px",
    fontsize=13,
    y=1.01,
)
plt.tight_layout()
plt.show()

# %% Foreground-only clustering
from dinoisawesome.foreground_head import ForegroundHead  # noqa: E402

fg_head = ForegroundHead(encoder, n_components=3)
fg_result = fg_head.segment(img, threshold=0.3, debias=True)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(fg_result["foreground_mask"])
ax.set_title("Foreground mask")
ax.axis("off")
plt.tight_layout()
plt.show()

# %% Extract foreground patch tokens
output = encoder(img, debias=True)
patches_grid_fg = output.patches[0][fg_result["foreground_mask_feature"]]  # (M, D)
N_fg, D_fg = patches_grid_fg.shape
tokens_fg = patches_grid_fg.reshape(N_fg, D_fg)
log.info("Foreground tokens: shape=%s", tokens_fg.shape)

# %% Similarity matrix for foreground tokens only
tokens_fg_norm = F.normalize(tokens_fg.float(), dim=-1)
similarity_fg = (tokens_fg_norm @ tokens_fg_norm.T).cpu().numpy()
distance_fg = np.clip(1.0 - similarity_fg, 0.0, 2.0)
affinity_fg = (similarity_fg + 1.0) / 2.0

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
im0 = axes[0].imshow(similarity_fg, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
axes[0].set_title(f"Cosine similarity (foreground, {N_fg} patches)")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], shrink=0.85)
im1 = axes[1].imshow(distance_fg, cmap="viridis", vmin=0, vmax=2, aspect="auto")
axes[1].set_title("Cosine distance (foreground)")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], shrink=0.85)
plt.tight_layout()
plt.show()

# %% Clustering on foreground tokens
tokens_fg_l2 = sk_normalize(tokens_fg.cpu().float().numpy())
DBSCAN_EPS_FG = 10.0
N_CLUSTERS_FG = 8
AGGLOMERATIVE_THRESHOLD_FG = 0.5

algo_configs_fg: list[tuple[str, object, np.ndarray]] = [
    (
        f"DBSCAN\neps={DBSCAN_EPS_FG}",
        DBSCAN(eps=DBSCAN_EPS_FG, min_samples=3, metric="precomputed"),
        distance_fg,
    ),
    (
        f"Agglomerative avg\nthresh={AGGLOMERATIVE_THRESHOLD_FG}",
        AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=AGGLOMERATIVE_THRESHOLD_FG,
            metric="precomputed",
            linkage="average",
        ),
        distance_fg,
    ),
    (
        f"Agglomerative complete\nthresh={AGGLOMERATIVE_THRESHOLD_FG}",
        AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=AGGLOMERATIVE_THRESHOLD_FG,
            metric="precomputed",
            linkage="complete",
        ),
        distance_fg,
    ),
    (
        f"KMeans  k={N_CLUSTERS_FG}",
        KMeans(n_clusters=N_CLUSTERS_FG, n_init=10, random_state=42),
        tokens_fg_l2,
    ),
    (
        f"Spectral  k={N_CLUSTERS_FG}",
        SpectralClustering(
            n_clusters=N_CLUSTERS_FG, affinity="precomputed", random_state=42, n_init=10
        ),
        affinity_fg,
    ),
]

cluster_results_fg: list[tuple[str, np.ndarray]] = []
for name, algo, X in algo_configs_fg:
    labels = algo.fit_predict(X)
    n_found = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    log.info("FG %-40s  %d clusters  %d noise", name.replace("\n", " "), n_found, n_noise)
    cluster_results_fg.append((name, labels))

# %% Visualise foreground-only clustering
fg_mask_2d = fg_result["foreground_mask_feature"]  # (H_GRID, W_GRID) bool


def _labels_to_rgba_fg(labels: np.ndarray) -> np.ndarray:
    labels_2d = np.zeros((H_GRID, W_GRID))
    labels_2d[fg_mask_2d] = labels
    unique_clusters = sorted(c for c in set(labels_2d.flat) if c != -1)
    rgba = np.zeros((H_GRID, W_GRID, 4), dtype=np.float32)
    for i, cid in enumerate(unique_clusters):
        color = np.array(CMAP(i % 20), dtype=np.float32)
        color[3] = OVERLAY_ALPHA
        rgba[labels_2d == cid] = color
    if -1 in labels_2d:
        rgba[labels_2d == -1] = [0.05, 0.05, 0.05, OVERLAY_ALPHA]
    return rgba


def _draw_result_fg(ax: plt.Axes, labels: np.ndarray, title: str) -> None:
    labels_2d = np.zeros((H_GRID, W_GRID))
    labels_2d[fg_mask_2d] = labels
    unique_clusters = sorted(c for c in set(labels_2d.flat) if c != -1)
    n_clusters = len(unique_clusters)
    n_noise = int((labels == -1).sum())

    overlay_small = _labels_to_rgba_fg(labels)
    overlay_pil = Image.fromarray((overlay_small * 255).astype(np.uint8), mode="RGBA")
    overlay_arr = np.array(overlay_pil.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) / 255.0

    ax.imshow(display_img)
    ax.imshow(overlay_arr, interpolation="nearest")

    legend_handles = [
        mpatches.Patch(color=CMAP(i % 20), label=f"cluster {cid}")
        for i, cid in enumerate(unique_clusters)
    ]
    if n_noise:
        legend_handles.append(
            mpatches.Patch(facecolor=(0.05, 0.05, 0.05), label=f"noise ({n_noise})")
        )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=6, framealpha=0.85)
    suffix = f", {n_noise} noise" if n_noise else ""
    ax.set_title(f"{title}\n{n_clusters} clusters{suffix}", fontsize=9)
    ax.axis("off")


NCOLS_FG = 3
n_algos_fg = len(cluster_results_fg)
n_algo_rows_fg = (n_algos_fg + NCOLS_FG - 1) // NCOLS_FG
nrows_fg = 1 + n_algo_rows_fg

fig, axes = plt.subplots(nrows_fg, NCOLS_FG, figsize=(NCOLS_FG * 5, nrows_fg * 5))
axes = axes.flatten()

axes[0].imshow(display_img)
axes[0].set_title("Original image", fontsize=11)
axes[0].axis("off")

for idx, (name, labels) in enumerate(cluster_results_fg):
    _draw_result_fg(axes[NCOLS_FG + idx], labels, name)

for i in range(NCOLS_FG + n_algos_fg, len(axes)):
    axes[i].axis("off")

plt.suptitle(
    f"Foreground-only clustering  |  DINO{DINO_VERSION}-{DINO_SIZE}  |  {N_fg} patches",
    fontsize=13,
    y=1.01,
)
plt.tight_layout()
plt.show()
