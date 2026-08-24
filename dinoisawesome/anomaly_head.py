"""AnomalyHead: kNN memory-bank anomaly detector backed by a Gallery."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .background_mask import compute_foreground_mask
from .encoder import DinoEncoder
from .gallery import Gallery

logger = logging.getLogger(__name__)


def _box_smooth(grid: torch.Tensor) -> torch.Tensor:
    """3x3 box average over a ``(H, W, D)`` patch grid, center included.

    Ports PatchCore's own ``AvgPool2d(3, 1, 1)`` local-smoothing trick (applied
    there to CNN feature maps) onto ViT patch tokens. At the grid borders, only
    in-bounds neighbors are averaged (``count_include_pad=False``) so edge/corner
    patches aren't biased toward zero by implicit zero-padding.
    """
    x = grid.permute(2, 0, 1).unsqueeze(0)  # (1, D, H, W)
    smoothed = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
    return smoothed.squeeze(0).permute(1, 2, 0)  # (H, W, D)


def _greedy_coreset(embeddings: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Greedy k-center coreset on L2-normalised embeddings (cosine distance).

    Returns a 1-D index tensor of *n_samples* representative rows.
    """
    n = len(embeddings)
    if n_samples >= n:
        return torch.arange(n)
    first = int(torch.randint(n, (1,)).item())
    selected = [first]
    min_dists = torch.full((n,), float("inf"), device=embeddings.device)
    for _ in range(n_samples - 1):
        last = embeddings[selected[-1]].unsqueeze(0)
        cos_sim = (embeddings @ last.T).squeeze(1)
        new_dists = (1.0 - cos_sim).clamp(min=0.0)
        min_dists = torch.minimum(min_dists, new_dists)
        selected.append(int(min_dists.argmax().item()))
    return torch.tensor(selected, device=embeddings.device)


def _to_pil(image: Image.Image | np.ndarray | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return Image.open(image).convert("RGB")


class AnomalyHead:
    """kNN memory-bank anomaly detector backed by a Gallery for persistence.

    The gallery stores reference patch embeddings on disk; the head keeps a
    copy in RAM for fast kNN scoring.  The in-RAM bank is never written back
    automatically — call :meth:`save` explicitly to flush label/split changes.

    Fit phase::

        head = AnomalyHead.build(encoder, normal_images, image_ids, out_dir)
        # or from an existing gallery:
        head = AnomalyHead(gallery, encoder)

    Inference::

        result = head.predict(query_image)
        result["score"]        # float  — image-level anomaly score
        result["anomaly_map"]  # ndarray [orig_H, orig_W] float32
        result["patch_scores"] # ndarray [N] per-patch kNN distance

    Args:
        gallery:        Pre-built Gallery of normal reference images.
        encoder:        DinoEncoder used to extract query features at predict time.
        block_idx:      Transformer block to score against.  Defaults to the last
                        stored block in the gallery config.
        num_neighbours: k for kNN scoring.
        coreset_ratio:  If given (0–1), subsample the memory bank via greedy
                        k-center after loading.
        split:          Pre-filter — only load gallery patches from this split.
        has_labels:     Pre-filter — only load patches that carry all of these labels.
        smoothing:      If True, replace every patch feature (bank and query alike)
                        with its 3x3 box average (center included, edge-safe) over
                        the patch grid before scoring — see :func:`_box_smooth`.
        masking:        If True, drop background patches (bank and query alike) via
                        PCA-based foreground masking before scoring — see
                        :func:`~dinoisawesome.background_mask.compute_foreground_mask`.
                        Background query patches score 0 in the returned anomaly map,
                        matching anomalib's own AnomalyDINO masking behaviour.
    """

    def __init__(
        self,
        gallery: Gallery,
        encoder: DinoEncoder,
        block_idx: int | None = None,
        num_neighbours: int = 1,
        coreset_ratio: float | None = None,
        split: str | None = None,
        has_labels: list[str] | None = None,
        smoothing: bool = False,
        masking: bool = False,
    ) -> None:
        self.gallery = gallery
        self.encoder = encoder
        self._block_idx = block_idx
        self.num_neighbours = num_neighbours
        self._filter_split = split
        self._filter_has_labels = has_labels
        self._coreset_ratio = coreset_ratio
        self.smoothing = smoothing
        self.masking = masking

        self._memory_bank: torch.Tensor = self._build_memory_bank()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        encoder: DinoEncoder,
        images: list,
        image_ids: list[str],
        gallery_dir: Path | str,
        split: str | list[str] = "train",
        image_labels: dict[str, list[str]] | None = None,
        patch_labels: dict[tuple[str, int, int], list[str]] | None = None,
        batch_size: int = 1,
        block_idx: int | None = None,
        num_neighbours: int = 1,
        coreset_ratio: float | None = None,
        filter_split: str | None = "train",
        filter_has_labels: list[str] | None = None,
        smoothing: bool = False,
        masking: bool = False,
    ) -> AnomalyHead:
        """Build a Gallery from *images* and return a ready-to-use AnomalyHead.

        Args:
            encoder:            DinoEncoder with ``layers > 1``.
            images:             PIL Images, numpy arrays (H,W,3), or file paths.
            image_ids:          Filesystem-safe unique ID per image.
            gallery_dir:        Output directory for the gallery.
            split:              Single split string or one per image.
            image_labels:       ``{image_id: [label, ...]}`` applied to all patches.
            patch_labels:       ``{(image_id, row, col): [label, ...]}`` overrides.
            batch_size:         Images per encoder forward pass.
            block_idx:          Transformer block for inference scoring.
            num_neighbours:     k for kNN.
            coreset_ratio:      Memory-bank subsampling ratio (None = keep all).
            filter_split:       Only load patches from this split into the bank.
            filter_has_labels:  Only load patches with all of these labels into the bank.
            smoothing:          See :class:`AnomalyHead`.
            masking:            See :class:`AnomalyHead`.
        """
        gallery = Gallery.build(
            encoder=encoder,
            images=images,
            image_ids=image_ids,
            out_dir=gallery_dir,
            split=split,
            image_labels=image_labels,
            patch_labels=patch_labels,
            batch_size=batch_size,
        )
        return cls(
            gallery=gallery,
            encoder=encoder,
            block_idx=block_idx,
            num_neighbours=num_neighbours,
            coreset_ratio=coreset_ratio,
            split=filter_split,
            has_labels=filter_has_labels,
            smoothing=smoothing,
            masking=masking,
        )

    # ------------------------------------------------------------------
    # Memory bank
    # ------------------------------------------------------------------

    @property
    def block_idx(self) -> int:
        """Resolved transformer block index (last stored block if not specified)."""
        if self._block_idx is None:
            return self.gallery.config.block_indices[-1]
        return self._block_idx

    def _build_memory_bank(self) -> torch.Tensor:
        df = self.gallery.filter(
            has_labels=self._filter_has_labels,
            split=self._filter_split,
        )
        if len(df) == 0:
            raise ValueError(
                "No gallery patches match the filter — check split/has_labels arguments."
            )
        if self.smoothing or self.masking:
            embs = self._load_processed_embeddings(
                df
            )  # (N', D) float32 — N' <= len(df) when masking
        else:
            embs = self.gallery.load_embeddings(df, self.block_idx)  # (N, D) float32
        bank = F.normalize(torch.from_numpy(embs), p=2, dim=1)

        if self._coreset_ratio is not None:
            n_keep = max(1, int(len(bank) * self._coreset_ratio))
            idx = _greedy_coreset(bank, n_keep)
            bank = bank[idx]
            logger.info(
                "Coreset: %d → %d patches (ratio=%.3f)",
                len(embs),
                len(bank),
                self._coreset_ratio,
            )

        logger.info("Memory bank ready: %d patches, D=%d", len(bank), bank.shape[1])
        return bank

    def _load_processed_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """Load patch embeddings for the rows in *df*, applying smoothing and/or
        background masking (whichever is enabled) per source image.

        Unlike :meth:`Gallery.load_embeddings`, this is not guaranteed to return one
        row per input row: masked-out background patches are dropped entirely, since
        a memory bank has no notion of "position" to preserve.
        """
        chunks: list[np.ndarray] = []
        for img_id, group in df.groupby("image_id"):
            grid = self.gallery.load_image_grid(img_id, self.block_idx)  # (H, W, D)
            if self.smoothing:
                grid = _box_smooth(torch.from_numpy(np.array(grid))).numpy()
            rows = group["row"].values.astype(int)
            cols = group["col"].values.astype(int)
            feats = grid[rows, cols]  # (n_i, D)
            if self.masking:
                h, w, d = grid.shape
                fg_2d = compute_foreground_mask(grid.reshape(h * w, d), (h, w)).reshape(h, w)
                feats = feats[fg_2d[rows, cols]]
            chunks.append(feats)
        return np.concatenate(chunks, axis=0)

    def reload(self) -> None:
        """Rebuild the in-RAM memory bank from the gallery on disk.

        Useful after the gallery metadata has been updated externally (e.g. new
        labels applied, images added via another Gallery instance).
        """
        self._memory_bank = self._build_memory_bank()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        image: Image.Image | np.ndarray | str | Path,
        debias: bool = False,
    ) -> dict[str, Any]:
        """Compute patch-level kNN anomaly scores for *image*.

        Args:
            image:  PIL Image, numpy (H,W,3) uint8 array, or a file path.
            debias: Remove the positional subspace from query patch embeddings
                    before scoring (see :meth:`DinoEncoder._debias_features`).

        Returns:
            {
                ``"score"``:        float — image-level score (mean of top-1% patch distances)
                ``"anomaly_map"``:  float32 ndarray of shape ``[orig_H, orig_W]``
                ``"patch_scores"``: float32 ndarray of shape ``[N]``
            }
        """
        pil_img = _to_pil(image)
        orig_w, orig_h = pil_img.size

        # layers=[...] keeps multi-layer form → patches: (B, 1, H, W, D)
        out = self.encoder([pil_img], layers=[self.block_idx], debias=debias)
        patches = out.patches[0, 0]  # (H, W, D)
        if self.smoothing:
            patches = _box_smooth(patches)
        H, W, D = patches.shape

        flat_all = F.normalize(patches.reshape(H * W, D), p=2, dim=1)
        if self.masking:
            fg_mask = compute_foreground_mask(patches.reshape(H * W, D).cpu().numpy(), (H, W))
            mask_idx: np.ndarray | None = np.nonzero(fg_mask)[0]
            flat = flat_all[mask_idx]
        else:
            mask_idx = None
            flat = flat_all
        bank = self._memory_bank.to(flat.device)

        sim = flat @ bank.T  # (N, M)
        dists = (1.0 - sim).clamp(0.0, 2.0)

        k = max(1, self.num_neighbours)
        topk, _ = torch.topk(dists, k=k, dim=1, largest=False)
        scored = topk.mean(dim=1) if k > 1 else topk.squeeze(1)

        if mask_idx is not None:
            # Background patches score 0 — excluded from the memory bank and from
            # scoring, matching anomalib's own AnomalyDINO masking behaviour.
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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Flush gallery metadata (labels, splits) to disk.

        The ``.npy`` embedding files are written once at build time and do not
        change.  Only the parquet metadata DataFrames need explicit saving.
        """
        self.gallery._save_metadata()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def gallery_size(self) -> int:
        """Number of patches in the in-RAM memory bank."""
        return len(self._memory_bank)

    @property
    def embed_dim(self) -> int:
        """Embedding dimension D."""
        return self._memory_bank.shape[1]
