from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image

from .encoder import DinoEncoder, ExtractorOutput, _MODEL_NAMES


@dataclass
class GalleryConfig:
    """Immutable record of the model and settings that produced this gallery."""

    model_name: str           # e.g. "dinov2_vitb14"
    version: str              # "v2" or "v3"
    size: str                 # "small" / "base" / "large" / "giant"
    patch_size: int           # 14 (v2) or 16 (v3)
    image_size: int           # e.g. 518
    block_indices: list[int]  # transformer block numbers stored, in L-axis order
    embed_dim: int            # embedding dimension D
    created_at: str           # ISO-8601 UTC timestamp
    schema_version: str = "1.0"

    def layer_idx(self, block_idx: int) -> int:
        return self.block_indices.index(block_idx)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> GalleryConfig:
        return cls(**json.loads(path.read_text()))

    @classmethod
    def from_encoder(cls, encoder: DinoEncoder) -> GalleryConfig:
        layers = encoder.layers
        if isinstance(layers, int):
            n = len(encoder.backbone.blocks)
            block_indices = list(range(n - layers, n))
        else:
            block_indices = list(layers)
        return cls(
            model_name=_MODEL_NAMES[(encoder.version, encoder.size)],
            version=encoder.version,
            size=encoder.size,
            patch_size=encoder.patch_size,
            image_size=encoder.img_size,
            block_indices=block_indices,
            embed_dim=int(encoder.backbone.embed_dim),
            created_at=datetime.now(timezone.utc).isoformat(),
        )


class Gallery:
    """
    Feature gallery: pandas DataFrames as the index, numpy files as storage.

    Disk layout::

        <root>/
            gallery_config.json
            patches.parquet        # one row per (image_id, row, col)
            cls_tokens.parquet     # one row per image_id; row i == cls_tokens.npy[i]
            cls_tokens.npy         # float32 (N, L, D) — all CLS embeddings
            embeddings/<id>.npy    # float32 (L, H, W, D) — patch embeddings per image

    patches.parquet columns:
        image_id  str       unique image identifier
        row       int16     patch-grid row  (0-indexed)
        col       int16     patch-grid column (0-indexed)
        y_center  int16     pixel y of patch centre
        x_center  int16     pixel x of patch centre
        labels    object    list[str] — arbitrary multi-labels
        split     str       "train" / "val" / "test" / "unlabeled"

    Labels are spatial (image_id, row, col), not layer-specific.  Choose which
    block to pull when loading or searching; the parquet files stay the same.

    Two-stage retrieval::

        top_images = gallery.retrieve_images(query_cls, k=20, split="train")
        matches = gallery.retrieve(
            query_patch, k=10,
            image_ids=top_images["image_id"].tolist(),
            has_labels=["keypoint"],
        )
    """

    _CONFIG_FILE  = "gallery_config.json"
    _PATCHES_FILE = "patches.parquet"
    _CLS_META     = "cls_tokens.parquet"
    _CLS_NPY      = "cls_tokens.npy"
    _EMB_DIR      = "embeddings"

    def __init__(self, root: Path | str) -> None:
        self.root       = Path(root)
        self.config     = GalleryConfig.load(self.root / self._CONFIG_FILE)
        self.patches    = pd.read_parquet(self.root / self._PATCHES_FILE)
        self.cls_tokens = pd.read_parquet(self.root / self._CLS_META)
        self._cls_array = np.load(self.root / self._CLS_NPY, mmap_mode="r")  # (N, L, D)

    @classmethod
    def build(
        cls,
        encoder: DinoEncoder,
        images: list,
        image_ids: list[str],
        out_dir: Path | str,
        split: str | list[str] = "train",
        image_labels: dict[str, list[str]] | None = None,
        patch_labels: dict[tuple[str, int, int], list[str]] | None = None,
        batch_size: int = 1,
    ) -> Gallery:
        """Extract features for all images and write the gallery to disk.

        Args:
            encoder:       DinoEncoder with layers > 1.
            images:        PIL Images, numpy arrays (H,W,3), or file paths.
            image_ids:     Filesystem-safe unique ID per image.
            out_dir:       Output directory (created if absent).
            split:         Single split string or one per image.
            image_labels:  {image_id: [label, ...]} applied to all patches and
                           the CLS token of that image.
            patch_labels:  {(image_id, row, col): [label, ...]} for specific patches.
            batch_size:    Images per forward pass.
        """
        if len(images) != len(image_ids):
            raise ValueError("images and image_ids must have the same length")
        if isinstance(encoder.layers, int) and encoder.layers == 1:
            raise ValueError(
                "Gallery requires multi-layer output. "
                "Initialise DinoEncoder with layers > 1 or an explicit list."
            )

        out_dir = Path(out_dir)
        (out_dir / cls._EMB_DIR).mkdir(parents=True, exist_ok=True)

        config       = GalleryConfig.from_encoder(encoder)
        splits       = [split] * len(images) if isinstance(split, str) else list(split)
        image_labels = image_labels or {}
        patch_labels = patch_labels or {}

        patch_rows: list[dict] = []
        cls_rows:   list[dict] = []
        cls_arrays: list[np.ndarray] = []

        for start in range(0, len(images), batch_size):
            batch_imgs = images[start : start + batch_size]
            batch_ids  = image_ids[start : start + batch_size]
            batch_spls = splits[start : start + batch_size]

            loaded = [
                Image.open(img).convert("RGB") if isinstance(img, (str, Path)) else img
                for img in batch_imgs
            ]

            out: ExtractorOutput = encoder(loaded)
            patches_np = out.patches.cpu().float().numpy()  # (B, L, H, W, D)
            cls_np     = out.cls.cpu().float().numpy()      # (B, L, D)
            _, L, H, W, _ = patches_np.shape

            for b, (img_id, spl) in enumerate(zip(batch_ids, batch_spls)):
                np.save(out_dir / cls._EMB_DIR / f"{img_id}.npy", patches_np[b])
                cls_arrays.append(cls_np[b])

                img_lbl = set(image_labels.get(img_id, []))
                for row in range(H):
                    for col in range(W):
                        extra = set(patch_labels.get((img_id, row, col), []))
                        patch_rows.append({
                            "image_id": img_id,
                            "row":      row,
                            "col":      col,
                            "y_center": row * config.patch_size + config.patch_size // 2,
                            "x_center": col * config.patch_size + config.patch_size // 2,
                            "labels":   list(img_lbl | extra),
                            "split":    spl,
                        })
                cls_rows.append({"image_id": img_id, "labels": list(img_lbl), "split": spl})

        np.save(out_dir / cls._CLS_NPY, np.stack(cls_arrays))

        patches_df = pd.DataFrame(patch_rows)
        for c in ("row", "col", "y_center", "x_center"):
            patches_df[c] = patches_df[c].astype("int16")
        patches_df.to_parquet(out_dir / cls._PATCHES_FILE, index=False)
        pd.DataFrame(cls_rows).to_parquet(out_dir / cls._CLS_META, index=False)
        config.save(out_dir / cls._CONFIG_FILE)

        return cls(out_dir)

    def filter(
        self,
        has_labels: list[str] | None = None,
        any_labels: list[str] | None = None,
        image_ids: list[str] | None = None,
        split: str | None = None,
    ) -> pd.DataFrame:
        """Return a filtered view of patches, preserving original index."""
        df = self.patches
        if has_labels:
            for lbl in has_labels:
                df = df[df["labels"].apply(lambda ls: lbl in ls)]
        if any_labels:
            df = df[df["labels"].apply(lambda ls: any(l in ls for l in any_labels))]
        if image_ids is not None:
            df = df[df["image_id"].isin(image_ids)]
        if split is not None:
            df = df[df["split"] == split]
        return df

    def load_embeddings(self, df: pd.DataFrame, block_idx: int | None = None) -> np.ndarray:
        """Load patch embeddings for rows in df as float32 (N, D).

        Opens each image's npy file once via mmap and does a vectorised spatial
        lookup — no full-file reads for sparse patch subsets.
        """
        if block_idx is None:
            block_idx = self.config.block_indices[-1]
        layer_idx = self.config.layer_idx(block_idx)

        result  = np.empty((len(df), self.config.embed_dim), dtype=np.float32)
        pos_map = {idx: i for i, idx in enumerate(df.index)}

        for img_id, group in df.groupby("image_id"):
            npy     = np.load(self.root / self._EMB_DIR / f"{img_id}.npy", mmap_mode="r")
            layer   = np.asarray(npy[layer_idx])  # (H, W, D)
            out_pos = [pos_map[idx] for idx in group.index]
            result[out_pos] = layer[group["row"].values.astype(int),
                                    group["col"].values.astype(int)]
        return result

    def load_cls_embeddings(
        self, image_ids: list[str] | None = None, block_idx: int | None = None
    ) -> tuple[list[str], np.ndarray]:
        """Return (ids, float32 (N, D)) CLS embeddings from the single npy array."""
        df = self.cls_tokens
        if image_ids is not None:
            df = df[df["image_id"].isin(image_ids)]
        return df["image_id"].tolist(), self._get_cls_embs(df, block_idx)

    def retrieve_images(
        self,
        query: np.ndarray,
        k: int = 10,
        block_idx: int | None = None,
        image_ids: list[str] | None = None,
        split: str | None = None,
    ) -> pd.DataFrame:
        """Top-k images by CLS cosine similarity. No patch files are opened.

        Returns a sub-DataFrame of cls_tokens with a "similarity" column added.
        """
        df = self.cls_tokens
        if split is not None:
            df = df[df["split"] == split]
        if image_ids is not None:
            df = df[df["image_id"].isin(image_ids)]
        if len(df) == 0:
            return df.assign(similarity=pd.Series(dtype=float))

        top_pos, top_sims = _cosine_topk(self._get_cls_embs(df, block_idx), query, k)
        result = df.iloc[top_pos].copy()
        result["similarity"] = top_sims
        return result.reset_index(drop=True)

    def retrieve(
        self,
        query: np.ndarray,
        k: int = 10,
        block_idx: int | None = None,
        has_labels: list[str] | None = None,
        any_labels: list[str] | None = None,
        image_ids: list[str] | None = None,
        split: str | None = None,
    ) -> pd.DataFrame:
        """Top-k patches by cosine similarity, with optional pre-filtering.

        Returns a sub-DataFrame of patches with a "similarity" column added.
        """
        df = self.filter(has_labels=has_labels, any_labels=any_labels,
                         image_ids=image_ids, split=split)
        if len(df) == 0:
            return df.assign(similarity=pd.Series(dtype=float))

        top_pos, top_sims = _cosine_topk(self.load_embeddings(df, block_idx), query, k)
        result = df.iloc[top_pos].copy()
        result["similarity"] = top_sims
        return result.reset_index(drop=True)

    def add_labels(self, subset: pd.DataFrame, labels: list[str], save: bool = True) -> None:
        """Add labels to patches in subset (index must align with self.patches).

        Example::

            kp = gallery.filter(image_ids=["img_001"])
            gallery.add_labels(kp[(kp.row == 5) & (kp.col == 7)], ["keypoint"])
        """
        new = set(labels)
        for idx in subset.index:
            self.patches.at[idx, "labels"] = list(set(self.patches.at[idx, "labels"]) | new)
        if save:
            self._save_metadata()

    def remove_labels(self, subset: pd.DataFrame, labels: list[str], save: bool = True) -> None:
        """Remove labels from patches in subset."""
        remove = set(labels)
        for idx in subset.index:
            self.patches.at[idx, "labels"] = [
                l for l in self.patches.at[idx, "labels"] if l not in remove
            ]
        if save:
            self._save_metadata()

    def label_from_mask(
        self,
        image_id: str,
        mask: np.ndarray,
        label_map: dict[int, list[str]],
        reduce: Literal["center", "majority"] = "center",
        save: bool = True,
    ) -> None:
        """Label patches of one image using a spatial mask.

        mask can be ``(image_size, image_size)`` pixel-space (reduced to patch
        grid via ``reduce``) or ``(grid_h, grid_w)`` already at patch resolution.

        label_map maps each integer mask value to the labels to assign::

            {0: ["background"], 1: ["foreground", "class_cat"]}

        ``reduce="center"`` samples each patch's centre pixel (fast);
        ``reduce="majority"`` uses the most common value in the patch region.
        Pass ``save=False`` when labeling many images; use :meth:`label_from_masks`
        to batch them with a single parquet write.
        """
        ps     = self.config.patch_size
        grid_h = self.config.image_size // ps
        grid_w = self.config.image_size // ps

        mask = np.asarray(mask)
        if mask.dtype == bool:
            mask = mask.view(np.uint8)

        H, W = mask.shape[:2]
        if (H, W) == (grid_h, grid_w):
            patch_mask = mask
        elif (H, W) == (self.config.image_size, self.config.image_size):
            patch_mask = _reduce_mask_to_grid(mask, grid_h, grid_w, ps, reduce)
        else:
            raise ValueError(
                f"mask shape {mask.shape} must be "
                f"({self.config.image_size}, {self.config.image_size}) or ({grid_h}, {grid_w})"
            )

        img_patches = self.patches[self.patches["image_id"] == image_id]
        if img_patches.empty:
            raise ValueError(f"image_id {image_id!r} not found in gallery")

        patch_values = patch_mask[
            img_patches["row"].values.astype(int),
            img_patches["col"].values.astype(int),
        ]
        for mask_val, labels in label_map.items():
            hits = np.where(patch_values == mask_val)[0]
            if hits.size:
                self.add_labels(img_patches.iloc[hits], labels, save=False)

        if save:
            self._save_metadata()

    def label_from_masks(
        self,
        masks: dict[str, np.ndarray],
        label_map: dict[int, list[str]],
        reduce: Literal["center", "majority"] = "center",
    ) -> None:
        """Apply label_from_mask across many images, writing parquet once."""
        for image_id, mask in masks.items():
            self.label_from_mask(image_id, mask, label_map, reduce=reduce, save=False)
        self._save_metadata()

    # private helpers

    def _get_cls_embs(self, df: pd.DataFrame, block_idx: int | None) -> np.ndarray:
        if block_idx is None:
            block_idx = self.config.block_indices[-1]
        layer = np.asarray(self._cls_array[:, self.config.layer_idx(block_idx), :])
        return layer[df.index.to_numpy()]

    def _save_metadata(self) -> None:
        self.patches.to_parquet(self.root / self._PATCHES_FILE, index=False)
        self.cls_tokens.to_parquet(self.root / self._CLS_META, index=False)


def _cosine_topk(
    embs: np.ndarray, query: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_positions, similarities) for the top-k rows in embs."""
    q    = query.ravel().astype(np.float32)
    sims = (embs @ q) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q) + 1e-8)
    k    = min(k, len(embs))
    top  = np.argpartition(sims, -k)[-k:]
    top  = top[np.argsort(sims[top])[::-1]]
    return top, sims[top]


def _reduce_mask_to_grid(
    mask: np.ndarray, grid_h: int, grid_w: int, patch_size: int, reduce: str
) -> np.ndarray:
    """Reduce a pixel-space mask to (grid_h, grid_w) patch resolution."""
    if reduce == "center":
        ys = np.arange(grid_h) * patch_size + patch_size // 2
        xs = np.arange(grid_w) * patch_size + patch_size // 2
        return mask[np.ix_(ys, xs)]
    if reduce == "majority":
        cropped = mask[: grid_h * patch_size, : grid_w * patch_size]
        blocks  = cropped.reshape(grid_h, patch_size, grid_w, patch_size)
        out     = np.empty((grid_h, grid_w), dtype=mask.dtype)
        for r in range(grid_h):
            for c in range(grid_w):
                out[r, c] = np.bincount(blocks[r, :, c, :].ravel().astype(np.intp)).argmax()
        return out
    raise ValueError(f"reduce must be 'center' or 'majority', got {reduce!r}")
