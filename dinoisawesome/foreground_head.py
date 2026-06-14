"""ForegroundHead: PCA-based foreground/background segmentation using DINO patch embeddings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .encoder import DinoEncoder

logger = logging.getLogger(__name__)


def _to_pil(image: Image.Image | np.ndarray | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return Image.open(image).convert("RGB")


class ForegroundHead:
    """PCA-based foreground/background segmentation from a single image's patch embeddings.

    No reference gallery is needed.  PCA is computed directly on the query image's
    own patch tokens — the first principal component captures the highest-variance
    direction in that image's patch space, which for natural images typically
    aligns with the foreground/background boundary.

    Usage::

        head = ForegroundHead(encoder)

        result = head.segment(image)
        result["foreground_mask"]   # bool ndarray [orig_H, orig_W]
        result["foreground_map"]    # float32 ndarray [orig_H, orig_W]  (PC-1 score)
        result["patch_scores"]      # float32 ndarray [N]               (per-patch PC-1)
        result["pca_projections"]   # float32 ndarray [N, n_components] (all components)

    Args:
        encoder:      DinoEncoder used to extract patch features.
        block_idx:    Transformer block to use.  ``None`` uses the encoder's last
                      requested layer (layers=-1 convention).
        n_components: Number of PCA components to compute per image (default 3).
    """

    def __init__(
        self,
        encoder: DinoEncoder,
        block_idx: int | None = None,
        n_components: int = 3,
    ) -> None:
        self.encoder = encoder
        self._block_idx = block_idx
        self.n_components = n_components

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def block_idx(self) -> int:
        """Resolved transformer block index.

        Falls back to the last block the encoder is configured to extract when
        ``block_idx`` was not specified at construction time.
        """
        if self._block_idx is not None:
            return self._block_idx
        layers = self.encoder.layers
        if isinstance(layers, int):
            # encoder.layers=n means the last n blocks; last one is backbone depth - 1
            depth = len(self.encoder.backbone.blocks)
            return depth - 1
        return layers[-1]

    # ------------------------------------------------------------------
    # PCA helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_pca(patches: torch.Tensor, n_components: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit PCA on *patches* via SVD and return (mean, components).

        Args:
            patches:      L2-normalised patch embeddings, shape ``(N, D)``.
            n_components: Number of principal components to return.

        Returns:
            mean:       ``(D,)`` mean of *patches*.
            components: ``(n_components, D)`` principal directions, sorted by
                        descending explained variance.
        """
        mean = patches.mean(dim=0)  # (D,)
        centred = patches - mean  # (N, D)

        # SVD on (D, N): left singular vectors are the principal directions in D-space.
        # This convention matches DinoEncoder._build_positional_basis.
        k = min(n_components, centred.shape[0], centred.shape[1])
        U, _, _ = torch.linalg.svd(centred.T, full_matrices=False)  # U: (D, min(D,N))
        components = U[:, :k].T.contiguous()  # (k, D)

        return mean, components

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def segment(
        self,
        image: Image.Image | np.ndarray | str | Path,
        debias: bool = False,
        threshold: float = 0.0,
    ) -> dict[str, Any]:
        """Segment *image* into foreground and background using per-image PCA.

        PCA is computed on the image's own patch embeddings.  The first
        principal component captures the highest-variance direction — for scenes
        with a dominant subject this typically separates foreground from
        background.

        The sign of PC-1 is determined by the data and can be either polarity.
        If the returned mask is inverted relative to expectations, negate the
        ``threshold`` (e.g. ``threshold=-1e-9``) or flip the sign of
        ``foreground_map`` before thresholding downstream.

        Args:
            image:     PIL Image, numpy (H,W,3) uint8 array, or a file path.
            debias:    Remove the positional subspace from patch embeddings before
                       PCA (see :meth:`DinoEncoder._debias_features`).  Useful to
                       suppress grid-layout artifacts in the segmentation.
            threshold: PC-1 score cut-off for the foreground decision.  Patches
                       with score > *threshold* are labelled foreground.
                       Defaults to 0.0 (split at the zero-crossing of the
                       mean-centred projection).

        Returns:
            {
                ``"foreground_mask"``:  bool ndarray ``[orig_H, orig_W]`` — True = foreground
                ``"foreground_map"``:   float32 ndarray ``[orig_H, orig_W]`` — PC-1 score map
                ``"patch_scores"``:     float32 ndarray ``[N]`` — per-patch PC-1 scores
                ``"pca_projections"``:  float32 ndarray ``[N, n_components]`` — all components
            }
        """
        pil_img = _to_pil(image)
        orig_w, orig_h = pil_img.size

        out = self.encoder([pil_img], layers=[self.block_idx], debias=debias)
        patches = out.patches[0, 0]  # (H, W, D)
        H, W, D = patches.shape

        flat = F.normalize(patches.reshape(H * W, D), p=2, dim=1)  # (N, D)

        mean, components = self._fit_pca(flat, self.n_components)  # (D,), (k, D)

        centred = flat - mean  # (N, D)
        projections = centred @ components.T  # (N, k)
        pc1_scores = projections[:, 0]  # (N,)

        foreground_flat = pc1_scores > threshold  # (N,) bool

        scores_np = pc1_scores.cpu().float().numpy()  # (N,)
        proj_np = projections.cpu().float().numpy()  # (N, k)
        fg_uint8 = foreground_flat.cpu().numpy().astype(np.uint8) * 255  # (N,) uint8

        # Resize patch-resolution maps to original image size
        score_pil = Image.fromarray(scores_np.reshape(H, W)).resize(
            (orig_w, orig_h), Image.Resampling.BILINEAR
        )
        fg_pil = Image.fromarray(fg_uint8.reshape(H, W)).resize(
            (orig_w, orig_h), Image.Resampling.NEAREST
        )

        return {
            "foreground_mask_feature": fg_uint8.reshape(H, W) > 127,
            "foreground_mask": np.array(fg_pil) > 127,
            "foreground_map": np.array(score_pil, dtype=np.float32),
            "patch_scores": scores_np,
            "pca_projections": proj_np,
        }
