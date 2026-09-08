"""Shared image-input coercion for the head classes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def to_pil(image: Image.Image | np.ndarray | str | Path) -> Image.Image:
    """Coerce a PIL Image, numpy (H, W, 3) array, or file path to an RGB PIL Image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return Image.open(image).convert("RGB")
