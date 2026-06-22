"""Sub-patch-accurate keypoint localization via Gaussian-suppressed soft-argmax.

Given a raw cosine-similarity heatmap between a reference keypoint token and
all DINOv3 patch tokens in a target image, the pipeline:

  1. Extracts the coarse peak via argmax.
  2. Multiplies the map by a 2-D Gaussian centred on that peak (suppresses
     distant background activations without zeroing the entire field).
  3. Converts the suppressed map to a sharp probability distribution via
     temperature-scaled softmax.
  4. Computes the centre of mass of the distribution to obtain a
     sub-patch-accurate, floating-point (x, y) coordinate.

All functions accept an arbitrary leading batch shape ``(..., H, W)`` so they
compose naturally with single heatmaps ``(H, W)``, per-image batches
``(B, H, W)``, or multi-keypoint batches ``(B, N, H, W)``.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coordinate grid
# ---------------------------------------------------------------------------


def make_coordinate_grid(
    H: int,
    W: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a spatial coordinate grid for a patch map of shape ``(H, W)``.

    Each cell ``[h, w]`` contains the ``(x, y)`` coordinate of that patch in
    patch-grid units, where ``x`` is the column index and ``y`` is the row
    index.

    Args:
        H:      Number of patch rows.
        W:      Number of patch columns.
        device: Target device for the output tensor.
        dtype:  Floating-point dtype for coordinates.

    Returns:
        Tensor of shape ``(H, W, 2)`` where the last dimension is ``(x, y)``.
    """
    rows = torch.arange(H, device=device, dtype=dtype)
    cols = torch.arange(W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")  # (H, W) each
    return torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)


# ---------------------------------------------------------------------------
# Gaussian suppression
# ---------------------------------------------------------------------------


def apply_gaussian_suppression(
    heatmap: torch.Tensor,
    peak_row: torch.Tensor,
    peak_col: torch.Tensor,
    sigma: float = 7.0,
) -> torch.Tensor:
    """Suppress a similarity heatmap with a Gaussian centred on a coarse peak.

    A 2-D Gaussian mask is generated for each element in the leading batch
    dimensions and multiplied element-wise with the heatmap.  Activations far
    from the peak are attenuated towards zero, preventing distant background
    patches from dragging the subsequent soft-argmax estimate.

    Args:
        heatmap:  Similarity map of shape ``(..., H, W)``.
        peak_row: Row index of the coarse peak, shape ``(...,)``.  Must be
                  broadcastable against the leading dims of *heatmap*.
        peak_col: Column index of the coarse peak, shape ``(...,)``.  Same
                  broadcast rules as *peak_row*.
        sigma:    Standard deviation of the Gaussian in patch units.
                  Larger values retain a wider neighbourhood.

    Returns:
        Suppressed heatmap, same shape as *heatmap*.
    """
    H, W = heatmap.shape[-2], heatmap.shape[-1]
    device, dtype = heatmap.device, heatmap.dtype

    rows = torch.arange(H, device=device, dtype=dtype)
    cols = torch.arange(W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")  # (H, W)

    # Expand peak indices so they broadcast over the (H, W) spatial dims.
    pr = peak_row.unsqueeze(-1).unsqueeze(-1).to(dtype)  # (..., 1, 1)
    pc = peak_col.unsqueeze(-1).unsqueeze(-1).to(dtype)  # (..., 1, 1)

    dist_sq = (grid_y - pr) ** 2 + (grid_x - pc) ** 2  # (..., H, W)
    gaussian = torch.exp(-dist_sq / (2.0 * sigma**2))  # (..., H, W)

    return heatmap * gaussian


# ---------------------------------------------------------------------------
# Temperature-scaled softmax
# ---------------------------------------------------------------------------


def temperature_softmax(
    heatmap: torch.Tensor,
    beta: float = 0.04,
) -> torch.Tensor:
    """Convert a (suppressed) similarity map to a normalised probability map.

    The spatial map is flattened, divided by *beta*, and passed through
    softmax before being restored to its original shape.  A small *beta*
    produces a very sharp (near one-hot) distribution; a large *beta* gives
    a diffuse one.

    Args:
        heatmap: Similarity map of shape ``(..., H, W)``.  May contain
                 negative values (cosine similarities span ``[-1, 1]``).
        beta:    Temperature parameter.  Must be positive.

    Returns:
        Probability map of shape ``(..., H, W)`` that sums to 1 over the
        spatial dimensions.

    Raises:
        ValueError: If *beta* is not positive.
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta!r}")

    H, W = heatmap.shape[-2], heatmap.shape[-1]
    flat = heatmap.reshape(*heatmap.shape[:-2], H * W)  # (..., H*W)
    probs = F.softmax(flat / beta, dim=-1)  # (..., H*W)
    return probs.reshape(*heatmap.shape[:-2], H, W)  # (..., H, W)


# ---------------------------------------------------------------------------
# Sub-patch-accurate localization
# ---------------------------------------------------------------------------


def localize_keypoint(
    heatmap: torch.Tensor,
    sigma: float = 7.0,
    beta: float = 0.04,
) -> torch.Tensor:
    """Localize a keypoint in patch-grid coordinates via Gaussian-suppressed soft-argmax.

    Runs the full four-step pipeline:

    1. **Coarse peak** — argmax over the raw similarity map.
    2. **Gaussian suppression** — multiply by a 2-D Gaussian centred on the
       peak to attenuate background activations.
    3. **Temperature softmax** — sharpen the suppressed map into a probability
       distribution.
    4. **Weighted centroid** — compute the centre of mass to obtain a
       continuous, sub-patch-accurate location.

    Args:
        heatmap: Raw cosine-similarity map, shape ``(..., H, W)``.  Values
                 should span ``[-1, 1]`` (L2-normalised dot products), but the
                 function is not restricted to this range.
        sigma:   Gaussian suppression radius in patch units.
        beta:    Softmax temperature.  Smaller values sharpen the peak.

    Returns:
        Tensor of shape ``(..., 2)`` containing the ``(x, y)`` coordinate of
        each localised keypoint in **patch-grid space** (fractional patch
        indices, where ``x`` is the column direction and ``y`` the row
        direction).  Pass the result to :func:`rescale_coords_to_image` to
        obtain pixel coordinates.
    """
    H, W = heatmap.shape[-2], heatmap.shape[-1]

    # Step 1 – coarse argmax peak.
    flat = heatmap.reshape(*heatmap.shape[:-2], H * W)  # (..., H*W)
    peak_idx = flat.argmax(dim=-1)  # (...,)  int64
    peak_row = peak_idx // W  # (...,)
    peak_col = peak_idx % W  # (...,)

    _log.debug("Coarse peak: row=%s col=%s", peak_row.tolist(), peak_col.tolist())

    # Step 2 – Gaussian suppression.
    masked = apply_gaussian_suppression(heatmap, peak_row.float(), peak_col.float(), sigma)

    # Step 3 – temperature-scaled softmax → probability map.
    probs = temperature_softmax(masked, beta)  # (..., H, W)

    # Step 4 – weighted centroid (soft-argmax).
    grid = make_coordinate_grid(H, W, device=heatmap.device, dtype=heatmap.dtype)
    # probs: (..., H, W), grid: (H, W, 2)
    # Multiply and sum over the spatial (H, W) dimensions.
    coords = (probs.unsqueeze(-1) * grid).sum(dim=(-3, -2))  # (..., 2)

    return coords


# ---------------------------------------------------------------------------
# Coordinate rescaling
# ---------------------------------------------------------------------------


def rescale_coords_to_image(
    coords: torch.Tensor,
    grid_hw: tuple[int, int],
    image_hw: tuple[int, int],
) -> torch.Tensor:
    """Map sub-patch coordinates from patch-grid space to image pixel space.

    Each patch of index ``(col, row)`` occupies one cell of the grid, so its
    pixel centre is at ``(col + 0.5) * (W_img / W_grid)``.  Fractional patch
    indices (as produced by :func:`localize_keypoint`) are mapped with the
    same linear transform, yielding sub-pixel-accurate pixel coordinates.

    Args:
        coords:   Patch-grid coordinates of shape ``(..., 2)`` where the last
                  dimension is ``(x, y)`` (column, row).
        grid_hw:  ``(H_grid, W_grid)`` — height and width of the patch grid.
        image_hw: ``(H_img, W_img)`` — height and width of the target image in
                  pixels.

    Returns:
        Tensor of shape ``(..., 2)`` containing ``(x_px, y_px)`` pixel
        coordinates (floating point, origin at the top-left corner).
    """
    H_grid, W_grid = grid_hw
    H_img, W_img = image_hw
    scale = coords.new_tensor([W_img / W_grid, H_img / H_grid])  # (2,)
    return (coords + 0.5) * scale
