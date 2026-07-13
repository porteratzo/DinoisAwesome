"""Image loading/resizing helpers shared across heads, scripts, and notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def to_pil(image: Image.Image | np.ndarray | str | Path) -> Image.Image:
    """Normalise a PIL image, numpy array, or file path to an RGB ``PIL.Image``."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return Image.open(image).convert("RGB")


def thumb(image: Image.Image, width: int) -> np.ndarray:
    """Resize *image* to *width* pixels wide, preserving aspect ratio.

    Returns a uint8 ``(H, W, 3)`` array.
    """
    new_h = int(image.height * width / image.width)
    return np.array(image.resize((width, new_h), Image.BICUBIC))
