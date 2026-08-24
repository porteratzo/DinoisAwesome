"""Severity-swept image corruptions used by the fundamental/ augmentation-robustness
experiments (rotation, illumination, color jitter, blur, noise, JPEG), plus the
mean-color fill helper rotation needs to pad without introducing a black border.

Each family sweeps from a literal no-op value (angle=0, gamma=1.0, jitter magnitude=0,
...) so severity=0 is pixel-identical to the unperturbed crop. All families are
pixel-only (mask unchanged) except rotation, which moves the object within the frame and
rotates the mask by the same angle — so every ``apply_*`` function shares the uniform
``(img, mask_px, val, fill) -> (img, mask_px)`` shape (see ``AugmentFn``); ``pixel_only``
adapts a plain ``(img, val) -> img`` function to it.

``fill`` and ``seed`` are explicit parameters, not module globals — the exemplar/query
image supplies its own ``mean_color(...)`` fill, and callers bind a fixed seed (e.g. via
``functools.partial``) so a severity level's random draw (color jitter, noise) is
deterministic per run.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

AugmentFn = Callable[
    [Image.Image, np.ndarray, float, tuple[int, int, int]], tuple[Image.Image, np.ndarray]
]


def mean_color(img: Image.Image) -> tuple[int, int, int]:
    return tuple(int(c) for c in np.array(img).reshape(-1, 3).mean(axis=0))


def pixel_only(fn: Callable[[Image.Image, float], Image.Image]) -> AugmentFn:
    """Wrap a pixel-value-only augmentation to the (img, mask, val, fill) -> (img, mask) shape."""

    def wrapped(
        img: Image.Image, mask_px: np.ndarray, val: float, fill: tuple[int, int, int]
    ) -> tuple[Image.Image, np.ndarray]:
        return fn(img, val), mask_px

    return wrapped


def apply_rotation(
    img: Image.Image, mask_px: np.ndarray, angle: float, fill: tuple[int, int, int]
) -> tuple[Image.Image, np.ndarray]:
    if angle == 0:
        return img, mask_px
    rotated_img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=fill)
    mask_pil = Image.fromarray(mask_px.astype(np.uint8) * 255)
    rotated_mask = (
        np.array(mask_pil.rotate(angle, resample=Image.NEAREST, expand=False, fillcolor=0)) > 0
    )
    return rotated_img, rotated_mask


def apply_gamma(img: Image.Image, gamma: float) -> Image.Image:
    if gamma == 1.0:
        return img
    arr = (np.array(img).astype(np.float32) / 255.0) ** gamma
    return Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))


def apply_color_jitter(img: Image.Image, magnitude: float, seed: int) -> Image.Image:
    if magnitude == 0:
        return img
    torch.manual_seed(seed)  # deterministic draw per severity level
    jitter = transforms.ColorJitter(
        brightness=magnitude,
        contrast=magnitude,
        saturation=magnitude,
        hue=min(magnitude * 0.5, 0.5),
    )
    return jitter(img)


def apply_blur(img: Image.Image, radius: float) -> Image.Image:
    if radius == 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_noise(img: Image.Image, sigma: float, seed: int) -> Image.Image:
    if sigma == 0:
        return img
    rng = np.random.default_rng(seed)
    arr = np.array(img).astype(np.float32)
    noisy = np.clip(arr + rng.normal(0.0, sigma, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_jpeg(img: Image.Image, quality: float) -> Image.Image:
    if quality >= 100:
        return img
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")
