from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_PATCH_SIZES: dict[str, int] = {"v2": 14, "v3": 16}

_HUB_REPOS: dict[str, str] = {
    "v2": "facebookresearch/dinov2",
    "v3": "facebookresearch/dinov3",
}

_MODEL_NAMES: dict[tuple[str, str], str] = {
    ("v2", "small"): "dinov2_vits14",
    ("v2", "base"):  "dinov2_vitb14",
    ("v2", "large"): "dinov2_vitl14",
    ("v2", "giant"): "dinov2_vitg14",
    ("v3", "small"): "dinov3_vits16",
    ("v3", "base"):  "dinov3_vitb16",
    ("v3", "large"): "dinov3_vitl16",
    ("v3", "giant"): "dinov3_vitg16",
}


@dataclass
class ExtractorOutput:
    """Output of a single forward pass through DinoEncoder.

    Single layer (layers=1):
        cls:     (B, D)           — CLS token embedding
        patches: (B, H, W, D)    — patch embeddings, y-first (height × width)

    Multiple layers (layers=N>1 or layers=[i,j,...]):
        cls:     (B, L, D)
        patches: (B, L, H, W, D)
    """

    cls: torch.Tensor
    patches: torch.Tensor


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


class DinoEncoder(nn.Module):
    """Thin wrapper around a DINO ViT loaded from torch hub.

    Args:
        version:  "v2" or "v3".
        size:     "small" | "base" | "large" | "giant".
        img_size: Square input resolution. Must be divisible by the patch size
                  (14 for v2, 16 for v3).
        layers:   Which transformer blocks to extract.
                  int  → last n blocks (e.g. 1 = last block only).
                  list → explicit 0-based block indices (e.g. [8, 9, 10, 11]).
                  Can be overridden per-call in forward().
        device:   "cuda" / "cuda:N", "xpu" / "xpu:N", "cpu", or None for auto.
        amp:      Enable torch.autocast (bfloat16) during inference.
        dtype:    Cast model weights to this dtype. None keeps float32.
    """

    def __init__(
        self,
        version: Literal["v2", "v3"] = "v2",
        size: Literal["small", "base", "large", "giant"] = "base",
        img_size: int = 224,
        layers: int | list[int] = 1,
        device: str | torch.device | None = None,
        amp: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        patch_size = _PATCH_SIZES[version]
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size must be divisible by {patch_size} for DINO {version}, got {img_size}"
            )

        self.version    = version
        self.size       = size
        self.img_size   = img_size
        self.layers     = layers
        self.amp        = amp
        self.model_dtype = dtype
        self.device     = _resolve_device(device)
        self.patch_size = patch_size
        self.grid_h     = img_size // patch_size
        self.grid_w     = img_size // patch_size

        backbone = torch.hub.load(_HUB_REPOS[version], _MODEL_NAMES[(version, size)])
        backbone.eval()
        if dtype is not None:
            backbone = backbone.to(dtype=dtype)
        self.backbone = backbone.to(self.device)

        self.preprocess = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    def _autocast_ctx(self):
        if self.amp:
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _to_tensor_batch(
        self, images: torch.Tensor | Image.Image | np.ndarray | list
    ) -> torch.Tensor:
        """Normalise varied image inputs to a (B, 3, H, W) float tensor on device."""
        if isinstance(images, torch.Tensor):
            x = images.to(device=self.device)
        else:
            if not isinstance(images, (list, tuple)):
                images = [images]
            pil = [img if isinstance(img, Image.Image) else Image.fromarray(img)
                   for img in images]
            x = torch.stack([self.preprocess(img) for img in pil]).to(self.device)

        if self.model_dtype is not None and not self.amp:
            x = x.to(dtype=self.model_dtype)
        return x

    @torch.inference_mode()
    def forward(
        self,
        images: torch.Tensor | Image.Image | np.ndarray | list,
        layers: int | list[int] | None = None,
    ) -> ExtractorOutput:
        """Extract CLS and patch features via get_intermediate_layers.

        Args:
            images: PIL Image / list of PIL Images, numpy (H,W,3) uint8 array /
                    list thereof, or a pre-processed float tensor (B, 3, H, W).
            layers: Override the encoder's default layer selection for this call.
                    int → last n blocks; list[int] → explicit block indices.

        Returns:
            ExtractorOutput with shapes as documented on that class.
            Patches are in (y, x) / (height, width) spatial order.
        """
        n      = layers if layers is not None else self.layers
        single = isinstance(n, int) and n == 1
        x      = self._to_tensor_batch(images)

        with self._autocast_ctx():
            # Each element: (patch_tokens (B, N, D), cls_token (B, D))
            raw = self.backbone.get_intermediate_layers(
                x, n=n, return_class_token=True, norm=True
            )

        cls     = torch.stack([r[1] for r in raw], dim=1)          # (B, L, D)
        patches = (torch.stack([r[0] for r in raw], dim=1)          # (B, L, N, D)
                   .reshape(x.shape[0], len(raw), self.grid_h, self.grid_w, -1))

        if single:
            cls     = cls.squeeze(1)      # (B, D)
            patches = patches.squeeze(1)  # (B, H, W, D)

        return ExtractorOutput(cls=cls, patches=patches)
