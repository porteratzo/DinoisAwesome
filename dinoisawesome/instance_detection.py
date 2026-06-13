"""Training-free instance detection via DINOv3 patch-token cosine similarity.

Pipeline
--------
1. Load exemplar and query images (PIL → encoder preprocessing).
2. Extract L2-normalised patch tokens from an intermediate transformer block
   (default: block 22 of 24 in ViT-L/16, i.e. ``dinov3_vitl16``).
3. Aggregate exemplar tokens into a compact descriptor (mean or K-means centroids).
4. Compute per-patch cosine similarity between query and descriptor → density map.
5. Apply max-pool NMS to extract instance centre coordinates.
6. Render a three-panel figure: query image | density map | annotated detections.

Note on patch size
------------------
DINOv3 in this project uses 16×16 patches (``dinov3_vitl16``).  The default
``img_size=448`` yields a 28×28 patch grid (448 / 16 = 28).  DINOv2 uses
14×14 patches; the encoder enforces divisibility at construction time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from PIL import Image

from .encoder import DinoEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------


def extract_patch_tokens(
    encoder: DinoEncoder,
    image: Image.Image,
    layer_idx: int,
) -> tuple[torch.Tensor, int, int]:
    """Return L2-normalised patch tokens from a single transformer block.

    Args:
        encoder:   Initialised :class:`DinoEncoder`.
        image:     PIL image (resized/normalised by the encoder's own transform).
        layer_idx: 0-based transformer block index.

    Returns:
        tokens:  Float tensor ``(H*W, C)`` on the encoder's device.
        grid_h:  Number of patch rows.
        grid_w:  Number of patch columns.
    """
    # layers=[layer_idx] → raw output shape (B, L=1, H, W, D)
    out = encoder(image, layers=[layer_idx])
    patches = out.patches[:, 0]  # (B, H, W, D) — drop the L=1 dimension
    _, H, W, D = patches.shape
    tokens = patches[0].reshape(H * W, D)  # (H*W, D)
    tokens = F.normalize(tokens, p=2, dim=-1)
    return tokens, H, W


# ---------------------------------------------------------------------------
# Exemplar aggregation
# ---------------------------------------------------------------------------


def compute_exemplar_features(
    tokens: torch.Tensor,
    mode: Literal["mean", "kmeans"] = "mean",
    k: int = 1,
) -> torch.Tensor:
    """Aggregate exemplar patch tokens into a compact descriptor.

    Args:
        tokens: ``(H*W, C)`` L2-normalised exemplar tokens.
        mode:   ``"mean"`` — single averaged descriptor (K=1);
                ``"kmeans"`` — K centroids via Lloyd's algorithm (10 iterations).
        k:      Number of centroids; used only when ``mode="kmeans"``.

    Returns:
        ``(K, C)`` L2-normalised feature matrix where K=1 for mean mode.
    """
    if mode == "mean":
        mean = tokens.mean(dim=0, keepdim=True)  # (1, C)
        return F.normalize(mean, p=2, dim=-1)

    # Mini k-means — pure PyTorch, no extra dependencies
    indices = torch.randperm(tokens.shape[0], device=tokens.device)[:k]
    centroids = tokens[indices].clone()  # (K, C), already L2-normalised

    for _ in range(10):
        sim = tokens @ centroids.T  # (N, K) — cosine sim (tokens are L2-normed)
        assignments = sim.argmax(dim=1)  # (N,)

        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=tokens.device)
        new_centroids.scatter_add_(0, assignments.unsqueeze(1).expand_as(tokens), tokens)
        counts.scatter_add_(0, assignments, torch.ones(tokens.shape[0], device=tokens.device))

        valid = counts > 0
        new_centroids[valid] = new_centroids[valid] / counts[valid].unsqueeze(1)
        # Centroids for empty clusters keep their previous value (no reinit needed here)
        centroids = F.normalize(new_centroids, p=2, dim=-1)

    return centroids  # (K, C)


# ---------------------------------------------------------------------------
# Density map
# ---------------------------------------------------------------------------


def compute_density_map(
    query_tokens: torch.Tensor,
    exemplar_features: torch.Tensor,
    grid_h: int,
    grid_w: int,
    threshold: float = 0.3,
) -> torch.Tensor:
    """Compute a thresholded cosine-similarity density map.

    Both *query_tokens* and *exemplar_features* must be L2-normalised, so
    ``query_tokens @ exemplar_features.T`` equals cosine similarity directly.

    Args:
        query_tokens:      ``(H*W, C)`` L2-normalised query patch tokens.
        exemplar_features: ``(K, C)`` L2-normalised exemplar descriptor(s).
        grid_h:            Patch-grid height of the query image.
        grid_w:            Patch-grid width of the query image.
        threshold:         Background-suppression offset applied before clamp.
                           Values below *threshold* are zeroed out.

    Returns:
        ``(H, W)`` float32 density map; background regions are clamped to zero.
    """
    sim = query_tokens @ exemplar_features.T  # (H*W, K)
    sim_mean = sim.mean(dim=-1)  # (H*W,)  — average across exemplar descriptors
    sim_2d = sim_mean.reshape(grid_h, grid_w)  # (H, W)
    return torch.clamp(sim_2d - threshold, min=0.0)


# ---------------------------------------------------------------------------
# Peak extraction
# ---------------------------------------------------------------------------


def extract_peaks(
    density_map: torch.Tensor,
    kernel_size: int = 5,
    min_peak_threshold: float = 0.01,
) -> torch.Tensor:
    """Locate local maxima via max-pool non-maximum suppression (NMS).

    A patch qualifies as a peak when its density equals the local neighbourhood
    maximum *and* exceeds *min_peak_threshold*.

    Args:
        density_map:        ``(H, W)`` thresholded density map.
        kernel_size:        NMS window size in patch units.
        min_peak_threshold: Minimum post-threshold density to accept as a peak.

    Returns:
        ``(N, 2)`` integer tensor of ``(col, row)`` = ``(x, y)`` coordinates
        in patch-grid space, one row per detected instance centre.
    """
    dm = density_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    padding = kernel_size // 2
    pooled = F.max_pool2d(dm, kernel_size=kernel_size, stride=1, padding=padding)
    pooled = pooled.squeeze()  # (H, W)

    is_peak = (density_map == pooled) & (density_map > min_peak_threshold)
    rows, cols = torch.where(is_peak)  # y-axis, x-axis
    return torch.stack([cols, rows], dim=1)  # (N, 2) as (x, y)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def visualize(
    query_image: Image.Image,
    density_map: torch.Tensor,
    peaks: torch.Tensor,
    patch_size: int,
    title: str = "Instance Detection",
    save_path: str | Path | None = None,
) -> Any:  # -> plt.Figure — deferred to avoid hard matplotlib dependency at import
    """Three-panel figure: query image | density map | annotated detections.

    Peak coordinates (patch-grid space) are scaled to pixel space by multiplying
    by *patch_size* and centring within each patch (offset +0.5).

    Args:
        query_image:  Original PIL query image.
        density_map:  ``(H, W)`` float tensor.
        peaks:        ``(N, 2)`` tensor of ``(x, y)`` patch-grid coordinates.
        patch_size:   Pixels per patch edge; scales grid coords to pixel coords.
        title:        Figure suptitle.
        save_path:    Optional path to write the figure (PNG/PDF/SVG).

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt  # deferred: optional at import time

    dm_np = density_map.cpu().float().numpy()
    peaks_np = peaks.cpu().numpy()  # (N, 2): (x_patch, y_patch)

    # Patch-grid → pixel centres
    px_x = (peaks_np[:, 0] + 0.5) * patch_size if len(peaks_np) else []
    px_y = (peaks_np[:, 1] + 0.5) * patch_size if len(peaks_np) else []

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)

    axes[0].imshow(query_image)
    axes[0].set_title("Query Image")
    axes[0].axis("off")

    im = axes[1].imshow(dm_np, cmap="jet", interpolation="nearest")
    axes[1].set_title("Density Map")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(query_image)
    if len(peaks_np) > 0:
        axes[2].scatter(
            px_x,
            px_y,
            c="red",
            s=80,
            marker="o",
            linewidths=1.5,
            edgecolors="white",
            zorder=5,
        )
    axes[2].set_title(f"Detections — {len(peaks_np)} found")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info("Saved visualization to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# End-to-end entry point
# ---------------------------------------------------------------------------


def detect_instances(
    exemplar_path: str | Path,
    query_path: str | Path,
    *,
    version: Literal["v2", "v3"] = "v3",
    size: Literal["small", "base", "large", "giant"] = "large",
    img_size: int = 448,
    layer_idx: int = 22,
    exemplar_mode: Literal["mean", "kmeans"] = "mean",
    exemplar_k: int = 1,
    density_threshold: float = 0.3,
    peak_kernel_size: int = 5,
    min_peak_threshold: float = 0.01,
    device: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    """End-to-end training-free instance detection using DINO patch tokens.

    Loads an exemplar and a query image, extracts L2-normalised patch tokens
    from a chosen intermediate transformer block, builds a cosine-similarity
    density map, and returns detected instance centres together with a figure.

    Args:
        exemplar_path:      Path to the reference (exemplar) image.
        query_path:         Path to the query image to search in.
        version:            DINO version (``"v2"`` or ``"v3"``).
        size:               Model size (``"small"`` | ``"base"`` | ``"large"`` | ``"giant"``).
        img_size:           Square input resolution; must be divisible by the patch size
                            (14 for v2, 16 for v3).  Default 448 satisfies both.
        layer_idx:          0-based transformer block index for feature extraction.
                            Default 22 is the penultimate block of ViT-L/24.
        exemplar_mode:      Aggregation strategy: ``"mean"`` (single descriptor) or
                            ``"kmeans"`` (K centroids for multi-modal exemplars).
        exemplar_k:         Number of K-means centroids (ignored for ``"mean"`` mode).
        density_threshold:  Background-suppression offset; patches below this cosine
                            similarity are zeroed in the density map.
        peak_kernel_size:   NMS window size in patch units (odd number recommended).
        min_peak_threshold: Minimum post-threshold density to report as a peak.
        device:             Torch device string or ``None`` for auto-detection.
        save_path:          If provided, the figure is saved to this path.

    Returns:
        density_map: ``(H_q, W_q)`` float32 tensor (patch-grid resolution).
        peaks:       ``(N, 2)`` integer tensor of ``(x, y)`` patch-grid coordinates.
        fig:         :class:`matplotlib.figure.Figure` with the three-panel visualisation.
    """
    logger.info(
        "Loading DINO%s-%s encoder (img_size=%d, layer_idx=%d)",
        version,
        size,
        img_size,
        layer_idx,
    )
    encoder = DinoEncoder(
        version=version,
        size=size,
        img_size=img_size,
        layers=[layer_idx],
        device=device,
    )

    exemplar_img = Image.open(Path(exemplar_path)).convert("RGB")
    query_img = Image.open(Path(query_path)).convert("RGB")

    logger.info("Extracting exemplar tokens from block %d", layer_idx)
    exemplar_tokens, _, _ = extract_patch_tokens(encoder, exemplar_img, layer_idx)

    logger.info("Extracting query tokens from block %d", layer_idx)
    query_tokens, grid_h, grid_w = extract_patch_tokens(encoder, query_img, layer_idx)

    exemplar_features = compute_exemplar_features(exemplar_tokens, mode=exemplar_mode, k=exemplar_k)

    logger.info("Computing density map (threshold=%.3f)", density_threshold)
    density_map = compute_density_map(
        query_tokens,
        exemplar_features,
        grid_h,
        grid_w,
        density_threshold,
    )

    logger.info(
        "Extracting peaks (kernel_size=%d, min_peak_threshold=%.4f)",
        peak_kernel_size,
        min_peak_threshold,
    )
    peaks = extract_peaks(density_map, peak_kernel_size, min_peak_threshold)
    logger.info("Found %d instance candidate(s)", len(peaks))

    fig = visualize(
        query_img,
        density_map,
        peaks,
        encoder.patch_size,
        title=f"Instance Detection — DINO{version}-{size} block {layer_idx}",
        save_path=save_path,
    )
    return density_map, peaks, fig
