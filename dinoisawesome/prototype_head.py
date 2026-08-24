"""PrototypeAnomalyHead: CLS-retrieval + k-means prototype anomaly detector."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans

from .background_mask import compute_foreground_mask
from .encoder import DinoEncoder
from .gallery import Gallery

logger = logging.getLogger(__name__)


def _to_pil(image: Image.Image | np.ndarray | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return Image.open(image).convert("RGB")


class PrototypeAnomalyHead:
    """CLS-retrieval + k-means-prototype anomaly detector backed by a Gallery.

    Unlike :class:`~dinoisawesome.anomaly_head.AnomalyHead` — which scores every
    query patch against one cross-image memory bank built from *all* normal
    training patches — this head builds a tiny, query-specific reference on the
    fly:

    1. Retrieve the ``retrieval_k`` normal training images whose CLS token is
       most cosine-similar to the query's CLS token
       (:meth:`Gallery.retrieve_images`).
    2. Pool their patch tokens (optionally PCA-background-masked, see
       :func:`~dinoisawesome.background_mask.compute_foreground_mask`) and
       collapse them into ``n_prototypes`` centroids via k-means.
    3. Score every query patch against those centroids via cosine distance,
       aggregated per :attr:`aggregation`.

    Args:
        gallery:        Gallery of normal reference images (CLS tokens + patch grids).
        encoder:        DinoEncoder used to extract query features at predict time.
        n_prototypes:   Number of k-means centroids computed per query (``k`` in k-means).
        retrieval_k:    Number of nearest normal images (by CLS similarity) whose
                        patches are pooled before clustering.
        aggregation:    ``"max"`` — patch score = cosine distance to the single
                        nearest prototype (standard nearest-centroid scoring).
                        ``"avg"`` — patch score = mean cosine distance across all
                        ``n_prototypes`` centroids (penalises patches unlike *any*
                        prototype, not just the closest one).
        masking:        If True, drop background patches (both the pooled reference
                        patches used for clustering and the query patches being
                        scored) via PCA-based foreground masking. Background query
                        patches score 0 in the returned anomaly map.
        block_idx:      Transformer block to score against. Defaults to the last
                        stored block in the gallery config.
        split:          Restrict CLS retrieval and prototype pooling to this split.
    """

    def __init__(
        self,
        gallery: Gallery,
        encoder: DinoEncoder,
        n_prototypes: int = 1,
        retrieval_k: int = 1,
        aggregation: Literal["max", "avg"] = "max",
        masking: bool = False,
        block_idx: int | None = None,
        split: str | None = "train",
    ) -> None:
        if aggregation not in ("max", "avg"):
            raise ValueError(f"aggregation must be 'max' or 'avg', got {aggregation!r}")
        self.gallery = gallery
        self.encoder = encoder
        self.n_prototypes = n_prototypes
        self.retrieval_k = retrieval_k
        self.aggregation = aggregation
        self.masking = masking
        self._block_idx = block_idx
        self._filter_split = split

    @property
    def block_idx(self) -> int:
        """Resolved transformer block index (last stored block if not specified)."""
        if self._block_idx is None:
            return self.gallery.config.block_indices[-1]
        return self._block_idx

    def _foreground_features(self, grid: np.ndarray) -> np.ndarray:
        """Flatten a ``(H, W, D)`` grid to ``(N, D)``, optionally dropping background patches."""
        h, w, d = grid.shape
        flat = grid.reshape(h * w, d)
        if not self.masking:
            return flat
        return flat[compute_foreground_mask(flat, (h, w))]

    def _build_prototypes(self, image_ids: list[str]) -> torch.Tensor:
        """Pool patch features from *image_ids* and k-means them into prototypes."""
        pooled = np.concatenate(
            [
                self._foreground_features(self.gallery.load_image_grid(img_id, self.block_idx))
                for img_id in image_ids
            ],
            axis=0,
        )
        n_clusters = min(self.n_prototypes, len(pooled))
        if n_clusters < self.n_prototypes:
            logger.warning(
                "Only %d foreground patch(es) available across %d retrieved image(s); "
                "reducing n_prototypes %d -> %d",
                len(pooled),
                len(image_ids),
                self.n_prototypes,
                n_clusters,
            )
        centers = (
            KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(pooled).cluster_centers_
        )
        return F.normalize(torch.from_numpy(centers.astype(np.float32)), p=2, dim=1)

    def predict(self, image: Image.Image | np.ndarray | str | Path) -> dict[str, Any]:
        """Compute patch-level anomaly scores for *image*.

        Returns:
            {
                ``"score"``:        float — image-level score (mean of top-1% patch distances)
                ``"anomaly_map"``:  float32 ndarray of shape ``[orig_H, orig_W]``
                ``"patch_scores"``: float32 ndarray of shape ``[N]``
            }
        """
        pil_img = _to_pil(image)
        orig_w, orig_h = pil_img.size

        # layers=[...] keeps multi-layer form → patches: (B, 1, H, W, D), cls: (B, 1, D)
        out = self.encoder([pil_img], layers=[self.block_idx])
        query_cls = out.cls[0, 0].cpu().float().numpy()
        patches = out.patches[0, 0]  # (H, W, D)
        H, W, D = patches.shape

        top_images = self.gallery.retrieve_images(
            query_cls, k=self.retrieval_k, block_idx=self.block_idx, split=self._filter_split
        )
        if len(top_images) == 0:
            raise ValueError("No normal reference images available for CLS retrieval.")
        prototypes = self._build_prototypes(top_images["image_id"].tolist())

        flat_all = F.normalize(patches.reshape(H * W, D), p=2, dim=1)
        if self.masking:
            fg_mask = compute_foreground_mask(patches.reshape(H * W, D).cpu().numpy(), (H, W))
            mask_idx: np.ndarray | None = np.nonzero(fg_mask)[0]
            flat = flat_all[mask_idx]
        else:
            mask_idx = None
            flat = flat_all

        proto = prototypes.to(flat.device)
        sim = flat @ proto.T  # (N, n_prototypes)
        dists = (1.0 - sim).clamp(0.0, 2.0)

        scored = dists.min(dim=1).values if self.aggregation == "max" else dists.mean(dim=1)

        if mask_idx is not None:
            patch_scores = torch.zeros(H * W, dtype=scored.dtype, device=scored.device)
            patch_scores[mask_idx] = scored
        else:
            patch_scores = scored

        num_top = max(1, int(patch_scores.shape[0] * 0.01))
        top_vals, _ = torch.topk(patch_scores, num_top, largest=True)
        image_score = float(top_vals.mean())

        scores_np = patch_scores.cpu().float().numpy()
        score_pil = Image.fromarray(scores_np.reshape(H, W)).resize(
            (orig_w, orig_h), Image.Resampling.BILINEAR
        )

        return {
            "score": image_score,
            "anomaly_map": np.array(score_pil, dtype=np.float32),
            "patch_scores": scores_np,
        }
