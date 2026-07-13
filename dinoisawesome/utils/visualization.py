"""Plotting/overlay helpers shared across the experiment notebooks and scripts.

Requires matplotlib (the ``vis``/``dev`` extras). ``draw_points`` additionally
requires ``opencv-python``, imported lazily so the rest of this module works
without it.
"""

from __future__ import annotations

import numpy as np
import torch
from matplotlib.axes import Axes
from PIL import Image
from sklearn.decomposition import PCA


def upsample_map(arr: np.ndarray, target_h: int, target_w: int | None = None) -> np.ndarray:
    """Normalise a ``(H, W)`` float map to [0, 1] and nearest-upsample to target resolution."""
    if target_w is None:
        target_w = target_h
    norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    pil = Image.fromarray((norm * 255).astype(np.uint8))
    return np.array(pil.resize((target_w, target_h), Image.NEAREST)) / 255.0


def heat_overlay(bg_uint8: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend a ``[0, 1]``-normalised jet heatmap over a uint8 RGB image."""
    from matplotlib import colormaps

    colored = colormaps["jet"](heat)[..., :3]
    return np.clip(bg_uint8 / 255.0 * (1 - alpha) + colored * alpha, 0, 1)


def mask_overlay_rgba(
    mask_bool: np.ndarray, color: tuple[float, float, float, float] = (0.2, 0.9, 0.2, 0.45)
) -> np.ndarray:
    """Build an ``(H, W, 4)`` RGBA overlay for a boolean mask; ``ax.imshow()`` the result."""
    ov = np.zeros((*mask_bool.shape, 4), dtype=np.float32)
    ov[mask_bool] = color
    return ov


def draw_box(ax: Axes, box_xyxy: tuple[float, float, float, float], color: str, label: str) -> None:
    """Draw a PIL-style ``(x0, y0, x1, y1)`` rectangle on an axes."""
    import matplotlib.patches as mpatches

    x0, y0, x1, y1 = box_xyxy
    rect = mpatches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor=color, facecolor="none", label=label
    )
    ax.add_patch(rect)


def draw_points(
    image: Image.Image,
    points: list[tuple[int, int]],
    labels: list[str],
    label_colors: dict[str, tuple[int, int, int]],
    marker_radius: int = 18,
    show_labels: bool = True,
) -> np.ndarray:
    """Draw labelled circles on *image*; returns a uint8 RGB canvas.

    Requires ``opencv-python`` (imported lazily).
    """
    import cv2

    canvas = np.array(image.convert("RGB"))
    for (x, y), lbl in zip(points, labels):
        color = label_colors[lbl]
        cv2.circle(canvas, (int(x), int(y)), marker_radius, color, thickness=-1)
        cv2.circle(canvas, (int(x), int(y)), marker_radius, (255, 255, 255), thickness=2)
        if show_labels:
            cv2.putText(
                canvas,
                lbl,
                (int(x) + marker_radius + 4, int(y) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                5,
                cv2.LINE_AA,
            )
    return canvas


def pca_project_to_rgb(
    tokens: torch.Tensor, h: int, w: int, pca: PCA | None = None
) -> tuple[np.ndarray, PCA]:
    """Project ``(H*W, D)`` patch tokens to 3 PCA components, min-max normalised to ``(h, w, 3)``.

    Pass a *pca* fitted on a joint token set (e.g. exemplar + query) to keep
    color assignments comparable across images; otherwise a new PCA is fitted
    on *tokens*.
    """
    tokens_np = tokens.cpu().float().numpy()
    if pca is None:
        pca = PCA(n_components=3)
        pca.fit(tokens_np)
    proj = pca.transform(tokens_np)  # (N, 3)
    for c in range(3):
        lo, hi = proj[:, c].min(), proj[:, c].max()
        proj[:, c] = (proj[:, c] - lo) / (hi - lo + 1e-8)
    return proj.reshape(h, w, 3), pca


def to_display_upscale(rgb_hw3: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour upscale a ``[0, 1]`` float ``(h, w, 3)`` array to ``size x size``."""
    img = Image.fromarray((rgb_hw3 * 255).astype(np.uint8))
    return np.array(img.resize((size, size), Image.NEAREST)) / 255.0
