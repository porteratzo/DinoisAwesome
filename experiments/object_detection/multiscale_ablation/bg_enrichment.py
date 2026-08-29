"""Background-gallery enrichment (Phase 1): sample extra crops from parts of the reference
image that were never inside any real mid/close instance crop, so a scale's background
gallery isn't limited to the padded fringe immediately around each instance.

Ported from ``experiments/object_detection/bg_gallery_enrichment.py`` (see that file's module
docstring for the original ablation and its findings) into a form ``engine.
build_all_scale_prototypes`` can call directly, gated by ``CropConfig.
bg_enrich_crops_per_scale``. Every non-zero setting is a genuinely new set of encoder passes,
which is why it's a ``CropConfig`` field (its own crop-cache namespace) rather than a
``ScoringConfig`` one — see ``common.py``'s module docstring on the two-tier cache split.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dinoisawesome import DinoEncoder
from dinoisawesome.instance_detection import compute_exemplar_features

# Self-sufficient rather than relying on engine.py's own sys.path.insert having already run —
# engine.py imports this module before it patches sys.path, so this module needs its own.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _shared.prototype_ops import extract_patch_tokens_batch  # noqa: E402

log = logging.getLogger(__name__)


@dataclass
class ExtraBgCrop:
    """One rejection-sampled background-only crop. Its *entire* token set counts as
    background (it's rejected outright if it overlaps the union instance mask at all), so
    there's no exclude mask to track, unlike a real ``engine.ClusterCrop``."""

    scale: str
    box: tuple[int, int, int, int]
    tokens: torch.Tensor  # (H*W, C) L2-normalised
    grid_h: int
    grid_w: int
    mean_token: torch.Tensor  # (1, C) plain mean over every patch


def _box_overlap_fraction(
    candidate: tuple[int, int, int, int], other: tuple[int, int, int, int]
) -> float:
    """Intersection area as a fraction of *candidate*'s own area (not IoU) — "how much of the
    new crop was already covered by an existing one", which is what an overlap budget means.
    """
    cx0, cy0, cx1, cy1 = candidate
    ox0, oy0, ox1, oy1 = other
    ix0, iy0 = max(cx0, ox0), max(cy0, oy0)
    ix1, iy1 = min(cx1, ox1), min(cy1, oy1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    cand_area = (cx1 - cx0) * (cy1 - cy0)
    if cand_area <= 0:
        return 0.0
    return ((ix1 - ix0) * (iy1 - iy0)) / cand_area


def sample_extra_bg_crops(
    rng: np.random.Generator,
    ref_img: Image.Image,
    union_mask: np.ndarray,
    scale: str,
    existing_boxes: list[tuple[int, int, int, int]],
    count: int,
    max_overlap_fraction: float,
    encoder: DinoEncoder,
    layer_idx: int,
    debias: bool,
    max_attempts_per_crop: int = 200,
) -> list[ExtraBgCrop]:
    """Rejection-sample up to *count* crops at *existing_boxes*'s own size distribution
    (that scale's real instance crops) that contain no foreground pixel and overlap no
    *existing_boxes* entry (real or already-accepted-extra, all the same scale) by more than
    *max_overlap_fraction*. Returns fewer than *count* (logged) if the attempt budget is
    exhausted before enough legal spots are found — large crops can leave little headroom.
    """
    if not existing_boxes:
        return []
    sizes = [(x1 - x0, y1 - y0) for x0, y0, x1, y1 in existing_boxes]
    H, W = union_mask.shape
    blocked_boxes = list(existing_boxes)

    pending: list[dict] = []
    for _ in range(count):
        accepted_box = None
        for _attempt in range(max_attempts_per_crop):
            w, h = sizes[int(rng.integers(len(sizes)))]
            w, h = min(w, W), min(h, H)
            x0 = int(rng.integers(0, W - w + 1)) if W > w else 0
            y0 = int(rng.integers(0, H - h + 1)) if H > h else 0
            x1, y1 = x0 + w, y0 + h
            if union_mask[y0:y1, x0:x1].any():
                continue
            box = (x0, y0, x1, y1)
            if any(
                _box_overlap_fraction(box, other) > max_overlap_fraction for other in blocked_boxes
            ):
                continue
            accepted_box = box
            break
        if accepted_box is None:
            log.warning(
                "scale=%s: could only sample %d/%d extra bg crops (overlap/attempt budget "
                "exhausted)",
                scale,
                len(pending),
                count,
            )
            break
        blocked_boxes.append(accepted_box)
        pending.append({"box": accepted_box, "crop_img": ref_img.crop(accepted_box)})

    if not pending:
        return []
    tokens_batch = extract_patch_tokens_batch(
        encoder, [p["crop_img"] for p in pending], layer_idx, debias
    )
    extras: list[ExtraBgCrop] = []
    for entry, (tokens, grid_h, grid_w) in zip(pending, tokens_batch):
        mean_token = compute_exemplar_features(tokens, mode="mean")
        extras.append(ExtraBgCrop(scale, entry["box"], tokens, grid_h, grid_w, mean_token))
    return extras
