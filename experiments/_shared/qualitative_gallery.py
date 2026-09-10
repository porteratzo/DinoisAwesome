"""Worst-N / best-N qualitative failure galleries.

Every `fundamental/` script's own figures average across instances (mean +/- std
curves, or heatmaps averaged pixel-wise across many instances) — none of them show an
actual individual failure case. An averaged number can't say *why* a config fails (one
orientation, one lighting condition, one occluded instance); this module fills that gap
with a simple, uniform "sort by score, plot the extremes" gallery any sweep script can
call once on its own headline comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class ScoredExample(TypedDict):
    label: str
    image: Image.Image
    raw: np.ndarray
    gt: np.ndarray
    score: float


def save_score_gallery(
    examples: list[ScoredExample],
    output_path: Path,
    n: int = 5,
    score_name: str = "IoU",
    title: str = "",
) -> None:
    """Saves a worst-`n`/best-`n` grid (by ascending `score`) with one row per example:
    the query crop, its raw score map, and its GT patch mask. `examples` shorter than
    `2 * n` just shows everything once, unsorted-duplicate-free (no row repeated).
    """
    if not examples:
        return
    ordered = sorted(examples, key=lambda e: e["score"])
    if len(ordered) <= 2 * n:
        picked = [(f"#{i + 1} worst-to-best", e) for i, e in enumerate(ordered)]
    else:
        picked = [(f"worst #{i + 1}", e) for i, e in enumerate(ordered[:n])] + [
            (f"best #{i + 1}", e) for i, e in enumerate(reversed(ordered[-n:]))
        ]

    n_rows = len(picked)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3.1 * n_rows), squeeze=False)
    for row, (rank_label, ex) in enumerate(picked):
        axes[row, 0].imshow(ex["image"])
        axes[row, 0].set_title(f"{rank_label}: {ex['label']}\n{score_name}={ex['score']:.3f}")
        axes[row, 0].axis("off")

        im = axes[row, 1].imshow(ex["raw"], cmap="viridis")
        axes[row, 1].set_title("raw score map")
        axes[row, 1].axis("off")
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)

        axes[row, 2].imshow(ex["gt"], cmap="gray")
        axes[row, 2].set_title("GT patch mask")
        axes[row, 2].axis("off")
    fig.suptitle(title or f"Worst {n} / best {n} by {score_name}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
