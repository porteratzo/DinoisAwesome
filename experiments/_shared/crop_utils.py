"""Native-pixel crop-box padding/flooring for ROI blob crops."""

from __future__ import annotations


def pad_and_floor_crop_box(
    px0: float,
    py0: float,
    px1: float,
    py1: float,
    pad_fraction: float,
    floor_w_px: float,
    floor_h_px: float,
    native_w: int,
    native_h: int,
) -> tuple[int, int, int, int]:
    """Pad a native-pixel bbox by *pad_fraction* per side (of its own width/height), then
    enforce a centred minimum width of *floor_w_px* and minimum height of *floor_h_px* —
    bounds how much the encoder's resize-to-img_size can upscale the crop — then clip.

    Per-axis rather than one shared floor so a scale's own (possibly non-square) crop
    size can be reproduced exactly, not just approximated by a single side length. Either
    floor only ever expands the box (never shrinks it below its own padded size), so a
    blob already bigger than the floor is left untouched.
    """
    w, h = px1 - px0, py1 - py0
    pad_w, pad_h = w * pad_fraction, h * pad_fraction
    px0, px1 = px0 - pad_w, px1 + pad_w
    py0, py1 = py0 - pad_h, py1 + pad_h

    w, h = px1 - px0, py1 - py0
    if w < floor_w_px:
        cx = (px0 + px1) / 2
        px0, px1 = cx - floor_w_px / 2, cx + floor_w_px / 2
    if h < floor_h_px:
        cy = (py0 + py1) / 2
        py0, py1 = cy - floor_h_px / 2, cy + floor_h_px / 2

    px0_i = max(0, int(round(px0)))
    py0_i = max(0, int(round(py0)))
    px1_i = min(native_w, int(round(px1)))
    py1_i = min(native_h, int(round(py1)))
    return px0_i, py0_i, px1_i, py1_i
