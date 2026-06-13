"""KeypointHead: gallery-backed keypoint matcher using DINO patch embeddings."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .encoder import DinoEncoder
from .gallery import Gallery

logger = logging.getLogger(__name__)


def _to_pil(image: Image.Image | np.ndarray | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return Image.open(image).convert("RGB")


class KeypointHead:
    """Gallery-backed keypoint matcher using DINO patch embeddings.

    Reference keypoints are stored as labeled patches in a pre-built Gallery.
    At find-time, query features are extracted once and each keypoint label's
    stored embedding is matched against all query patches.

    The label-to-embedding cache is held in RAM and invalidated whenever a
    new keypoint is registered.  The gallery metadata is not written to disk
    until :meth:`save` is called explicitly.

    Register keypoints::

        head = KeypointHead(gallery, encoder)
        head.register(
            image_id="ref_001",
            points=[[x0, y0], [x1, y1]],
            labels=["bolt_left", "bolt_right"],
            orig_size=(orig_w, orig_h),
        )
        head.save()

    Find in a new image::

        matches = head.find(query_image, labels=["bolt_left", "bolt_right"])
        # [{"label": "bolt_left", "point": (px, py), "similarity": 0.93}, ...]

    Args:
        gallery:   Pre-built Gallery containing the reference image(s).
        encoder:   DinoEncoder used to extract query features at find time.
        block_idx: Transformer block to use.  Defaults to the last stored block.
    """

    def __init__(
        self,
        gallery: Gallery,
        encoder: DinoEncoder,
        block_idx: int | None = None,
    ) -> None:
        self.gallery = gallery
        self.encoder = encoder
        self._block_idx = block_idx
        self._kp_cache: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def block_idx(self) -> int:
        """Resolved transformer block index (last stored block if not specified)."""
        if self._block_idx is None:
            return self.gallery.config.block_indices[-1]
        return self._block_idx

    @property
    def registered_labels(self) -> list[str]:
        """Sorted list of all unique labels currently in the gallery patch metadata."""
        labels: set[str] = set()
        for ls in self.gallery.patches["labels"]:
            labels.update(ls)
        return sorted(labels)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        image_id: str,
        points: list[list[int]],
        labels: list[str],
        orig_size: tuple[int, int],
    ) -> None:
        """Label patches at *points* in *image_id* with the given *labels*.

        The gallery's patch metadata is updated in RAM immediately.  Call
        :meth:`save` to persist to disk.

        Args:
            image_id:  Image ID as stored in the gallery (must exist).
            points:    ``[[x, y], ...]`` pixel coordinates in the **original**
                       image coordinate space (before resize to ``image_size``).
            labels:    Per-point string label, e.g. ``"bolt_left"``.  Must have
                       the same length as *points*.
            orig_size: ``(orig_w, orig_h)`` of the original image, used to
                       map pixel coordinates to the gallery's patch grid.

        Raises:
            ValueError: If *image_id* is not in the gallery.
            ValueError: If *points* and *labels* have different lengths.
        """
        if len(points) != len(labels):
            raise ValueError("points and labels must have the same length")

        img_patches = self.gallery.patches[self.gallery.patches["image_id"] == image_id]
        if img_patches.empty:
            raise ValueError(f"image_id {image_id!r} not found in gallery")

        img_size = self.gallery.config.image_size
        ps = self.gallery.config.patch_size
        n_patches = img_size // ps
        orig_w, orig_h = orig_size

        for (x, y), label in zip(points, labels):
            col = min(int(x * img_size / orig_w) // ps, n_patches - 1)
            row = min(int(y * img_size / orig_h) // ps, n_patches - 1)
            patch_row = img_patches[
                (img_patches["row"] == row) & (img_patches["col"] == col)
            ]
            if patch_row.empty:
                logger.warning(
                    "No patch at (row=%d, col=%d) for image_id=%r — skipping label %r",
                    row,
                    col,
                    image_id,
                    label,
                )
                continue
            self.gallery.add_labels(patch_row, [label], save=False)
            self._kp_cache.pop(label, None)

        logger.info(
            "Registered %d keypoint(s) for image_id=%r",
            len(points),
            image_id,
        )

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _get_label_emb(self, label: str) -> torch.Tensor | None:
        """Return the cached ``(D,)`` L2-normalised embedding for *label*.

        Loads from the gallery on first access and caches the result.  Returns
        ``None`` if no patches carry *label*.
        """
        if label not in self._kp_cache:
            df = self.gallery.filter(has_labels=[label])
            if len(df) == 0:
                return None
            embs = self.gallery.load_embeddings(df, self.block_idx)  # (K, D) float32
            # Average over all registered patches for this label, then re-normalise.
            emb = torch.from_numpy(embs).float().mean(dim=0)
            self._kp_cache[label] = F.normalize(emb, p=2, dim=0)
        return self._kp_cache[label]

    def find(
        self,
        image: Image.Image | np.ndarray | str | Path,
        labels: list[str] | None = None,
        debias: bool = False,
    ) -> list[dict]:
        """Locate registered keypoints in *image* via nearest-patch cosine similarity.

        A single encoder forward pass is made for the query image; all requested
        labels are matched against the resulting feature map.

        Args:
            image:   Query PIL Image, numpy (H,W,3) uint8 array, or file path.
            labels:  Restrict to these keypoint labels.  ``None`` = all registered.
            debias:  Remove the positional subspace from query patches before matching.

        Returns:
            List of dicts, one per matched keypoint, in the same order as *labels*
            (or alphabetical if *labels* is ``None``):

            - ``"label"``:      keypoint label string
            - ``"point"``:      ``(x, y)`` pixel centre of the best-matching patch
            - ``"similarity"``: cosine similarity score (higher = more similar)
        """
        target_labels = labels if labels is not None else self.registered_labels
        if not target_labels:
            return []

        ref_embs = {lbl: self._get_label_emb(lbl) for lbl in target_labels}
        ref_embs = {lbl: emb for lbl, emb in ref_embs.items() if emb is not None}
        if not ref_embs:
            return []

        pil_img = _to_pil(image)
        orig_w, orig_h = pil_img.size

        # layers=[...] keeps multi-layer form → patches: (B, 1, H, W, D)
        out = self.encoder([pil_img], layers=[self.block_idx], debias=debias)
        patches = out.patches[0, 0]  # (H, W, D)
        H, W, D = patches.shape
        flat = F.normalize(patches.reshape(H * W, D), p=2, dim=1)  # (N, D)

        results: list[dict] = []
        for lbl, ref_emb in ref_embs.items():
            ref_emb = ref_emb.to(flat.device)
            sims = flat @ ref_emb  # (N,)
            best_idx = int(sims.argmax())
            row, col = divmod(best_idx, W)
            px = int((col + 0.5) * orig_w / W)
            py = int((row + 0.5) * orig_h / H)
            results.append(
                {
                    "label": lbl,
                    "point": (px, py),
                    "similarity": float(sims[best_idx]),
                }
            )

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Flush the gallery's patch label metadata to disk."""
        self.gallery._save_metadata()
