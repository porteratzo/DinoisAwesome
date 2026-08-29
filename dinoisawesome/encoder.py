"""DINO ViT encoder wrapper: loads a pretrained backbone from torch hub and
exposes CLS and patch-level intermediate-layer features."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    ("v2", "base"): "dinov2_vitb14",
    ("v2", "large"): "dinov2_vitl14",
    ("v2", "giant"): "dinov2_vitg14",
    ("v3", "small"): "dinov3_vits16",
    ("v3", "base"): "dinov3_vitb16",
    ("v3", "large"): "dinov3_vitl16",
    ("v3", "giant"): "dinov3_vitg16",
}

_ALLOWED_DTYPES: frozenset[torch.dtype] = frozenset(
    {torch.float32, torch.float16, torch.bfloat16, torch.float64}
)

_log = logging.getLogger(__name__)


def _find_weights_file(weights_dir: Path, model_name: str) -> Path:
    """Return the unique ``{model_name}_*.pth`` file inside *weights_dir*.

    Args:
        weights_dir: Directory that contains the ``.pth`` weight files.
        model_name:  Hub model name (e.g. ``"dinov3_vitb16"``).

    Raises:
        FileNotFoundError: No matching file found.
        RuntimeError:      More than one matching file found.
    """
    matches = sorted(weights_dir.glob(f"{model_name}_*.pth"))
    if not matches:
        raise FileNotFoundError(
            f"No weight file matching '{model_name}_*.pth' found in {weights_dir}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple weight files match '{model_name}_*.pth' in {weights_dir}: "
            + ", ".join(p.name for p in matches)
        )
    return matches[0]


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
    """Resolve a device specifier to a concrete ``torch.device``.

    If *device* is ``None``, auto-selects the first available accelerator in
    priority order: CUDA → Intel XPU → CPU.

    Args:
        device: A device string (``"cuda"``, ``"cuda:1"``, ``"cpu"``), an
                existing ``torch.device``, or ``None`` for auto-detection.

    Returns:
        A ``torch.device`` instance.
    """
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
        device:      "cuda" / "cuda:N", "xpu" / "xpu:N", "cpu", or None for auto.
        amp:         Enable torch.autocast during inference. When `dtype` is not given,
                     amp also defaults the backbone's weight dtype to bfloat16 (an
                     explicit `dtype` always takes precedence over amp's default).
        dtype:       Cast backbone weights (and, when amp=False, the input tensor) to
                     this dtype. One of torch.float32, torch.float16, torch.bfloat16,
                     torch.float64. None keeps float32, unless amp=True defaults it to
                     bfloat16.
        weights_dir: Directory containing local ``.pth`` weight files named
                     ``{model_name}_*.pth`` (e.g. ``dinov3_vitb16_pretrain_*.pth``).
                     When set, the architecture is still fetched from torch hub
                     (code only, no large download) and the weights are loaded from
                     this directory instead.  If ``None``, torch hub downloads both.
        max_batch_size: Maximum images per forward pass. `forward()` splits any larger
                     input list/tensor into chunks of at most this size and concatenates
                     the results, so callers can pass an arbitrarily long list without
                     batching manually or risking a CUDA OOM from one oversized pass.
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
        svd_components: int = 8,
        weights_dir: str | Path | None = None,
        max_batch_size: int = 16,
    ) -> None:
        super().__init__()

        patch_size = _PATCH_SIZES[version]
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size must be divisible by {patch_size} for DINO {version}, got {img_size}"
            )

        if dtype is not None and dtype not in _ALLOWED_DTYPES:
            raise ValueError(
                f"dtype must be one of {sorted(_ALLOWED_DTYPES, key=str)}, got {dtype}"
            )

        self.version = version
        self.size = size
        self.img_size = img_size
        self.layers = layers
        self.amp = amp
        self.device = _resolve_device(device)
        self.patch_size = patch_size
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.max_batch_size = max_batch_size

        # An explicit dtype always wins; otherwise amp defaults the effective weight/
        # compute dtype to bfloat16 so amp=True actually halves backbone memory instead
        # of just wrapping fp32 weights in an autocast region.
        self.model_dtype = dtype if dtype is not None else (torch.bfloat16 if amp else None)
        if self.model_dtype is torch.float16 and self.device.type == "cpu":
            _log.warning(
                "dtype=torch.float16 on CPU has limited PyTorch kernel support; "
                "prefer torch.bfloat16 or torch.float32 for CPU inference."
            )

        model_name = _MODEL_NAMES[(version, size)]
        if weights_dir is not None:
            weights_path = _find_weights_file(Path(weights_dir), model_name)
            self.weights_signature = f"{weights_path}:{weights_path.stat().st_mtime_ns}"
            _log.info("Loading %s architecture from hub (no pretrained weights)", model_name)
            backbone = torch.hub.load(_HUB_REPOS[version], model_name, pretrained=False)
            _log.info("Loading weights from %s", weights_path)
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            # some checkpoints nest the state dict under a "model" key
            if (
                isinstance(state, dict)
                and "model" in state
                and not any(k.startswith("blocks.") for k in state)
            ):
                state = state["model"]
            backbone.load_state_dict(state)
        else:
            backbone = torch.hub.load(_HUB_REPOS[version], model_name)
            self.weights_signature = f"hub:{model_name}"
        backbone.eval()
        if self.model_dtype is not None:
            backbone = backbone.to(dtype=self.model_dtype)
        self.backbone = backbone.to(self.device)

        self.svd_components = svd_components
        self._positional_basis: torch.Tensor | None = None
        self._p_perp: torch.Tensor | None = None
        self._p_perp_key: tuple[torch.device, torch.dtype] | None = None

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (img_size, img_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                # transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    def _autocast_ctx(self) -> contextlib.AbstractContextManager:
        """Return an autocast context manager using `self.model_dtype` when AMP is
        enabled, or a no-op. `__init__` guarantees `model_dtype` is set whenever amp=True."""
        if self.amp:
            return torch.autocast(device_type=self.device.type, dtype=self.model_dtype)
        return contextlib.nullcontext()

    def _to_tensor_batch(
        self, images: torch.Tensor | Image.Image | np.ndarray | list
    ) -> torch.Tensor:
        """Normalise varied image inputs to a ``(B, 3, H, W)`` float tensor on device.

        Accepts a pre-processed batch tensor, a single unbatched tensor/PIL/numpy
        image, a batched numpy array, or a list of PIL/numpy images. When ``amp=False``
        and a ``model_dtype`` is set, the tensor is cast to that dtype before returning.

        Args:
            images: One of:
                - ``torch.Tensor`` of shape ``(B, 3, H, W)`` — used as-is (moved to device).
                - ``torch.Tensor`` of shape ``(3, H, W)`` — a single image, unsqueezed to ``B=1``.
                - ``np.ndarray`` of shape ``(B, H, W, 3)`` uint8 — a batch, split into ``B``
                  single images before preprocessing.
                - ``PIL.Image.Image`` or ``np.ndarray`` (H, W, 3) uint8 — a single image.
                - ``list``/``tuple`` of the single-image types above.

        Returns:
            Float tensor of shape ``(B, 3, H, W)`` on ``self.device``.
        """
        if isinstance(images, torch.Tensor):
            x = images.to(device=self.device)
            if x.ndim == 3:
                x = x.unsqueeze(0)
            elif x.ndim != 4:
                raise ValueError(
                    f"Tensor input must be (3, H, W) or (B, 3, H, W), got shape {tuple(x.shape)}"
                )
        else:
            if isinstance(images, np.ndarray) and images.ndim == 4:
                images = list(images)  # (B, H, W, 3) -> B individual (H, W, 3) arrays
            elif not isinstance(images, (list, tuple)):
                images = [images]
            pil = [img if isinstance(img, Image.Image) else Image.fromarray(img) for img in images]
            x = torch.stack([self.preprocess(img) for img in pil]).to(self.device)

        if self.model_dtype is not None and not self.amp:
            x = x.to(dtype=self.model_dtype)
        return x

    # ---------------------------------------------------------------------------
    # The following methods (_build_positional_basis, _debias_features, and the
    # positional_basis property) are derived from INSID3
    # (https://github.com/visinf/INSID3).
    # Copyright 2026 Claudia Cuttano, Gabriele Trivigno, Christoph Reich,
    # Daniel Cremers, Carlo Masone, Stefan Roth.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    #
    # Adapted from models/insid3.py for DinoEncoder.
    # ---------------------------------------------------------------------------

    @property
    def positional_basis(self) -> torch.Tensor:
        """(D, svd_components) orthonormal basis of the positional subspace.

        Built lazily from a blank image on first access and cached for the
        lifetime of the encoder.
        """
        if self._positional_basis is None:
            self._positional_basis = self._build_positional_basis()
        return self._positional_basis

    @torch.inference_mode()
    def _build_positional_basis(self) -> torch.Tensor:
        """Estimate the positional subspace of the backbone via SVD on a blank image.

        Feeds an all-zeros image through the backbone, L2-normalises the patch
        tokens (which then carry only position information), centres them, and
        returns the top-``svd_components`` left singular vectors as an orthonormal
        basis of shape ``(D, svd_components)``.
        """
        blank = torch.zeros(1, 3, self.img_size, self.img_size, device=self.device)
        if self.model_dtype is not None:
            blank = blank.to(dtype=self.model_dtype)

        with self._autocast_ctx():
            raw = self.backbone.get_intermediate_layers(
                blank, n=1, return_class_token=False, norm=True
            )

        # raw[0]: (1, N, D) patch tokens; N = H*W
        patches = raw[0][0]  # (N, D)
        patches = F.normalize(patches, p=2, dim=-1)

        E = patches.T  # (D, N)
        E = E - E.mean(dim=1, keepdim=True)
        # SVD always in float32: torch.linalg.svd has no CPU kernel for bfloat16 at all
        # (crashes outright when amp forces model_dtype to bfloat16 on a CPU-resolved
        # device), and bfloat16 SVD is numerically weak even where it does run — this
        # basis is reused for the encoder's lifetime, so it's worth computing precisely
        # once. _debias_features casts the result back to each caller's own device/dtype.
        U, _, _ = torch.linalg.svd(E.float(), full_matrices=False)  # U: (D, min(D, N))
        return U[:, : self.svd_components].contiguous()

    def _debias_features(self, patches: torch.Tensor) -> torch.Tensor:
        """Project patch embeddings onto the orthogonal complement of the positional subspace.

        Handles both single-layer ``(B, H, W, D)`` and multi-layer
        ``(B, L, H, W, D)`` inputs; the output has the same shape.

        Args:
            patches: L2-normalised patch embeddings from ``forward()``.

        Returns:
            Debiased, L2-renormalised patches with the same shape as the input.
        """
        shape = patches.shape
        D = shape[-1]
        # Flatten all dims except the last into (N, H*W, D)
        x = patches.reshape(-1, self.grid_h * self.grid_w, D)

        # P_perp only depends on positional_basis (cached for the encoder's lifetime) and the
        # (device, dtype) of the incoming patches, so it's cached here too, keyed on that pair —
        # otherwise this (D, D) eye + matmul gets rebuilt on every single forward() call.
        key = (x.device, x.dtype)
        if self._p_perp is None or self._p_perp_key != key:
            basis = self.positional_basis.to(device=x.device, dtype=x.dtype)  # (D, k)
            self._p_perp = torch.eye(D, device=x.device, dtype=x.dtype) - basis @ basis.T
            self._p_perp_key = key

        x_deb = x @ self._p_perp.T  # (N, H*W, D)
        return F.normalize(x_deb, p=2, dim=-1).reshape(shape)

    def _split_into_chunks(
        self,
        images: torch.Tensor | Image.Image | np.ndarray | list,
        chunk_size: int | None = None,
    ) -> list:
        """Split *images* into chunks of at most `chunk_size` items each.

        Mirrors `_to_tensor_batch`'s shape handling so slicing always happens along the
        batch axis: a single unbatched tensor/PIL/numpy image becomes one chunk of size
        1 (never sliced along its channel/height axis), while a ``(B, ...)`` tensor,
        ``(B, H, W, 3)`` ndarray, or list/tuple is sliced along its leading dimension.

        Args:
            chunk_size: Items per chunk. Defaults to `self.max_batch_size`; pass
                        `chunk_size=1` to get one chunk per individual item (e.g. for
                        per-item cache-key hashing).
        """
        if chunk_size is None:
            chunk_size = self.max_batch_size
        items: torch.Tensor | np.ndarray | list | tuple
        if isinstance(images, torch.Tensor):
            items = images.unsqueeze(0) if images.ndim == 3 else images
        elif isinstance(images, np.ndarray) and images.ndim == 4:
            items = images
        elif isinstance(images, (list, tuple)):
            items = images
        else:
            items = [images]  # single PIL Image or (H, W, 3) ndarray
        return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    @torch.inference_mode()
    def forward(
        self,
        images: torch.Tensor | Image.Image | np.ndarray | list,
        layers: int | list[int] | None = None,
        debias: bool = False,
    ) -> ExtractorOutput:
        """Extract CLS and patch features via get_intermediate_layers.

        Internally splits *images* into chunks of at most `self.max_batch_size` and runs
        one forward pass per chunk, concatenating the results — callers can pass an
        arbitrarily long list without batching manually or risking a CUDA OOM from one
        oversized forward pass.

        Args:
            images: PIL Image / list of PIL Images, numpy (H,W,3) uint8 array /
                    list thereof, or a pre-processed float tensor (B, 3, H, W).
            layers: Override the encoder's default layer selection for this call.
                    int → last n blocks; list[int] → explicit block indices.
            debias: If True, remove the positional subspace from patch embeddings
                    via ``_debias_features()`` before returning.  The positional
                    basis is computed lazily from ``positional_basis`` and cached.

        Returns:
            ExtractorOutput with shapes as documented on that class.
            Patches are in (y, x) / (height, width) spatial order.
        """
        chunks = self._split_into_chunks(images)
        if len(chunks) == 1:
            return self._forward_single_batch(chunks[0], layers, debias)

        outputs = [self._forward_single_batch(chunk, layers, debias) for chunk in chunks]
        return ExtractorOutput(
            cls=torch.cat([o.cls for o in outputs], dim=0),
            patches=torch.cat([o.patches for o in outputs], dim=0),
        )

    def _forward_single_batch(
        self,
        images: torch.Tensor | Image.Image | np.ndarray | list,
        layers: int | list[int] | None,
        debias: bool,
    ) -> ExtractorOutput:
        """Single un-chunked forward pass. See `forward()` for the public, chunked entry point."""
        n = layers if layers is not None else self.layers
        single = isinstance(n, int) and n == 1
        x = self._to_tensor_batch(images)

        with self._autocast_ctx():
            # Each element: (patch_tokens (B, N, D), cls_token (B, D))
            raw = self.backbone.get_intermediate_layers(x, n=n, return_class_token=True, norm=True)

        cls = torch.stack([r[1] for r in raw], dim=1)  # (B, L, D)
        patches = torch.stack([r[0] for r in raw], dim=1).reshape(  # (B, L, N, D)
            x.shape[0], len(raw), self.grid_h, self.grid_w, -1
        )

        if single:
            cls = cls.squeeze(1)  # (B, D)
            patches = patches.squeeze(1)  # (B, H, W, D)

        if debias:
            patches = self._debias_features(patches)

        return ExtractorOutput(cls=cls, patches=patches)
