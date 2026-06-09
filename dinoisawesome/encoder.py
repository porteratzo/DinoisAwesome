import contextlib
from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Patch size is 14 for all DINO ViT variants
_PATCH_SIZE = 14

# Hub repos — add v3 entry once it lands on torch hub
_HUB_REPOS: dict[str, str] = {
    "v2": "facebookresearch/dinov2",
    "v3": "facebookresearch/dinov3",
}

# torch hub model names per (version, size)
_MODEL_NAMES: dict[tuple[str, str], str] = {
    ("v2", "small"): "dinov2_vits14",
    ("v2", "base"):  "dinov2_vitb14",
    ("v2", "large"): "dinov2_vitl14",
    ("v2", "giant"): "dinov2_vitg14",
    ("v3", "small"): "dinov3_vits14",
    ("v3", "base"):  "dinov3_vitb14",
    ("v3", "large"): "dinov3_vitl14",
    ("v3", "giant"): "dinov3_vitg14",
}


@dataclass
class ExtractorOutput:
    """Output of a single forward pass through DinoEncoder.

    cls:     (B, D)        — CLS token embedding
    patches: (B, H, W, D)  — patch embeddings in (y, x) / (height, width) order
    """

    cls: torch.Tensor
    patches: torch.Tensor


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
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
        img_size: Square input resolution (must be divisible by 14).
        device:   "cuda" / "cuda:N", "xpu" / "xpu:N", "cpu", or None for auto.
        amp:      Enable torch.autocast during inference (bfloat16 on the
                  resolved device type).
        dtype:    Cast model weights to this dtype (e.g. torch.bfloat16 or
                  torch.float16).  None keeps default float32.
    """

    def __init__(
        self,
        version: Literal["v2", "v3"] = "v2",
        size: Literal["small", "base", "large", "giant"] = "base",
        img_size: int = 224,
        device: Optional[Union[str, torch.device]] = None,
        amp: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        if img_size % _PATCH_SIZE != 0:
            raise ValueError(f"img_size must be divisible by {_PATCH_SIZE}, got {img_size}")

        self.version = version
        self.size = size
        self.img_size = img_size
        self.amp = amp
        self.model_dtype = dtype
        self.device = _resolve_device(device)

        hub_repo = _HUB_REPOS[version]
        model_name = _MODEL_NAMES[(version, size)]
        backbone = torch.hub.load(hub_repo, model_name)
        backbone.eval()
        if dtype is not None:
            backbone = backbone.to(dtype=dtype)
        self.backbone = backbone.to(self.device)

        self.grid_h = img_size // _PATCH_SIZE
        self.grid_w = img_size // _PATCH_SIZE

        self.preprocess = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _autocast_ctx(self):
        if self.amp:
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _to_tensor_batch(
        self,
        images: Union[torch.Tensor, Image.Image, np.ndarray, list],
    ) -> torch.Tensor:
        """Normalise varied image inputs to a (B, 3, H, W) float tensor on device."""
        if isinstance(images, torch.Tensor):
            x = images.to(device=self.device)
            if self.model_dtype is not None and not self.amp:
                x = x.to(dtype=self.model_dtype)
            return x

        if not isinstance(images, (list, tuple)):
            images = [images]

        pil_images = [
            img if isinstance(img, Image.Image) else Image.fromarray(img)
            for img in images
        ]
        x = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
        if self.model_dtype is not None and not self.amp:
            x = x.to(dtype=self.model_dtype)
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        images: Union[torch.Tensor, Image.Image, np.ndarray, list],
    ) -> ExtractorOutput:
        """Extract CLS and patch features.

        Args:
            images: One of:
                - PIL Image or list of PIL Images
                - numpy array (H, W, 3) uint8 or list thereof
                - pre-processed float tensor (B, 3, H, W) already on any device

        Returns:
            ExtractorOutput with:
                cls     — (B, D)
                patches — (B, H, W, D) in (y, x) / (height, width) order
        """
        x = self._to_tensor_batch(images)

        with self._autocast_ctx():
            out = self.backbone.forward_features(x)

        cls: torch.Tensor = out["x_norm_clstoken"]           # (B, D)
        patch_tokens: torch.Tensor = out["x_norm_patchtokens"]  # (B, N, D)

        B, _N, D = patch_tokens.shape
        # reshape to spatial grid — row-major so dim1=y(height), dim2=x(width)
        patches = patch_tokens.reshape(B, self.grid_h, self.grid_w, D)

        return ExtractorOutput(cls=cls, patches=patches)
