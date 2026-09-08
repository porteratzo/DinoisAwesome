"""Synthetic test image shared by the numbered tutorial scripts.

Not a package __init__ — these tutorials are meant to be run directly
(``python basic_tutorials/0N_*.py``), which puts this directory on
``sys.path[0]``, so siblings import it as a plain top-level module
(``import _common``) rather than a relative import.
"""

from __future__ import annotations

import numpy as np


def make_disc_image(h: int = 224, w: int = 224) -> np.ndarray:
    """Synthetic uint8 RGB image: dark background with a bright green disc."""
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    cy, cx = h // 2, w // 2
    ys, xs = np.ogrid[:h, :w]
    disc = (ys - cy) ** 2 + (xs - cx) ** 2 <= (min(h, w) // 3) ** 2
    img[disc] = [50, 200, 50]
    return img


def make_disc_image_with_geometry(
    h: int = 224, w: int = 224
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Same disc image as :func:`make_disc_image`, plus its (center_x, center_y, radius)."""
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 3
    return make_disc_image(h, w), (cx, cy, radius)
