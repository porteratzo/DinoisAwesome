"""Shared setup logic factored out of the `scale_composition_*.py` sibling ablation scripts in
this directory (`scale_composition_oracle_iou.py`, `scale_composition_bg_ablation.py`,
`scale_composition_adaptive_oracle.py`, `scale_composition_max_pool.py`,
`scale_composition_query_matching.py`) to avoid drift between copies each script's own comments
used to describe as "identical to scale_composition_oracle_iou.py" / "copied verbatim". Only
pieces confirmed code-identical (ignoring per-script docstring/comment wording and, for the
composition-combo builder, variable naming) were extracted here; each script still owns the rest
of its logic, which genuinely differs by design.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from _shared.mask_geometry import pixel_mask_to_patch_mask, scale_crop_box

log = logging.getLogger(__name__)


def scale_step_name(i: int, n: int) -> str:
    """t=0 -> "global", t=1 -> "close", the exact halfway point -> "mid" (matches today's
    naming when n is even), everything else -> its own fraction, e.g. "2/6"."""
    if i == 0:
        return "global"
    if i == n:
        return "close"
    if n % 2 == 0 and i == n // 2:
        return "mid"
    return f"{i}/{n}"


def scale_step_boxes(
    pixel_mask: np.ndarray, t_values: np.ndarray, padding_frac: float
) -> list[tuple[int, int, int, int]]:
    """PIL-style crop boxes linearly interpolated from the whole image (t=0) to `close`'s own
    tight, padded bbox (t=1) — same interpolation scale_crop_similarity.py's own
    `scale_crop_boxes` uses, generalizing `scale_crop_box`'s fixed global/mid/close named
    points to arbitrary t. Boxes shrink monotonically as t grows, so `close` (t=1, the
    smallest) meeting MIN_CROP_SIZE guarantees every other t does too."""
    H, W = pixel_mask.shape
    close_box = scale_crop_box(pixel_mask, "close", padding_frac)
    global_box = (0, 0, W, H)
    return [
        tuple(int(round(a + (b - a) * t)) for a, b in zip(global_box, close_box)) for t in t_values
    ]


def split_fg_bg_patches(
    patch_tokens: torch.Tensor,
    mask_px: np.ndarray,
    grid_h: int,
    grid_w: int,
    label: str,
    img_size: int,
    mask_patch_threshold: float,
    *,
    bg_exclude_mask_px: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split one crop's patch tokens into (fg, bg), L2-normalised. `img_size`/`mask_patch_threshold`
    are passed in explicitly (each caller's own `IMG_SIZE`/`MASK_PATCH_THRESHOLD` module
    constant) rather than closed over, since a shared module can't rely on an importing script's
    own global namespace."""
    if bg_exclude_mask_px is None:
        bg_exclude_mask_px = mask_px
    tokens = F.normalize(patch_tokens.reshape(grid_h * grid_w, -1), p=2, dim=-1)

    fg_patch_mask = pixel_mask_to_patch_mask(
        mask_px, grid_h, grid_w, img_size, mask_patch_threshold
    )
    fg_flat = torch.from_numpy(fg_patch_mask.reshape(-1)).to(tokens.device)
    fg = tokens[fg_flat]
    if fg.shape[0] == 0:
        log.warning("%s: fg mask empty after patch-grid projection — using all patches", label)
        fg = tokens

    bg_exclude_patch_mask = pixel_mask_to_patch_mask(
        bg_exclude_mask_px, grid_h, grid_w, img_size, mask_patch_threshold
    )
    bg_exclude_flat = torch.from_numpy(bg_exclude_patch_mask.reshape(-1)).to(tokens.device)
    bg = tokens[~bg_exclude_flat]
    if bg.shape[0] == 0:
        log.warning("%s: bg mask empty after patch-grid projection — using all patches", label)
        bg = tokens

    return fg, bg


def build_composition_combos(
    scale_names: list[str],
) -> tuple[dict[str, list[str]], list[str], list[str], list[str], list[str]]:
    """Returns (combos, prefix_names, suffix_names, anchored_inward_names, anchored_outward_names).

    Not the full power set of scales, but more than just the two open-ended growth sweeps.
    Five families, ~4n+3 combos instead of 2^(n+1) - 1:
      - single scale (n+1) — each scale step alone.
      - prefix-from-global / suffix-from-close (2n) — open-ended growth from one endpoint,
        dropping the other until the very last step.
      - the classic 3-point `global+mid+close` baseline — every sibling script's fixed combo,
        included once for direct comparison against the finer-grained sweeps.
      - anchored-inward / anchored-outward (2(n-1)) — keep BOTH `global` and `close` in every
        entry (matches the classic combo's own logic of "always cover both extremes") and grow
        the middle from one side or the other: "anchored_inward" adds middle scales moving away
        from global (mirrors the prefix sweep but never drops `close`), "anchored_outward" adds
        them moving away from close (mirrors the suffix sweep but never drops `global`).
    """
    combos: dict[str, list[str]] = {}
    for _name in scale_names:
        combos[_name] = [_name]  # every single scale, on its own
    prefix_names: list[str] = []
    for _i in range(2, len(scale_names) + 1):
        _members = scale_names[:_i]
        _key = "+".join(_members)
        combos[_key] = _members
        prefix_names.append(_key)
    suffix_names: list[str] = []
    for _i in range(2, len(scale_names) + 1):
        _members = scale_names[-_i:]
        _key = "+".join(_members)
        if _key not in combos:  # i == len(scale_names) duplicates the full prefix
            combos[_key] = _members
        suffix_names.append(_key)

    if "mid" in scale_names:
        combos["global+mid+close"] = ["global", "mid", "close"]

    _middle_names = scale_names[1:-1]  # every scale strictly between global and close
    anchored_inward_names: list[str] = []
    for _i in range(0, len(_middle_names) + 1):
        _members = ["global", *_middle_names[:_i], "close"]
        _key = "+".join(_members)
        combos[_key] = _members  # i=0 -> "global+close"; i=len(_middle_names) -> full set
        anchored_inward_names.append(_key)
    anchored_outward_names: list[str] = []
    for _i in range(0, len(_middle_names) + 1):
        _members = ["global", *_middle_names[len(_middle_names) - _i :], "close"]
        _key = "+".join(_members)
        if _key not in combos:  # i=0 and i=len(_middle_names) duplicate inward's ends
            combos[_key] = _members
        anchored_outward_names.append(_key)

    return combos, prefix_names, suffix_names, anchored_inward_names, anchored_outward_names
