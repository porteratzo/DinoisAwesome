from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

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
    block_indices: list[int]  # transformer block numbers stored, in order
    embed_dim: int            # embedding dimension D
    created_at: str           # ISO-8601 UTC timestamp
    schema_version: str = "1.0"

    @property
    def n_layers(self) -> int:
        return len(self.block_indices)

    def layer_idx(self, block_idx: int) -> int:
        """Return the L-axis index for a given transformer block number."""
        return self.block_indices.index(block_idx)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "GalleryConfig":
        return cls(**json.loads(path.read_text()))

    @classmethod
    def from_encoder(cls, encoder: DinoEncoder) -> "GalleryConfig":
        model_name = _MODEL_NAMES[(encoder.version, encoder.size)]
        layers = encoder.layers
        if isinstance(layers, int):
            n_blocks = len(encoder.backbone.blocks)
            block_indices = list(range(n_blocks - layers, n_blocks))
        else:
            block_indices = list(layers)
        return cls(
            model_name=model_name,
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
    Feature gallery: a pandas DataFrame as the index, numpy files as storage.

    Disk layout::

        <root>/
            gallery_config.json
            patches.parquet        # one row per (image_id, row, col)
            cls_tokens.parquet     # one row per image_id; row i aligns with cls_tokens.npy[i]
            cls_tokens.npy         # float32 (N, L, D) — all CLS embeddings in one array
            embeddings/<id>.npy    # float32 (L, H, W, D) — patch embeddings per image

    CLS tokens are small enough to live in a single file (N×L×D) that can be
    mmapped and searched globally without opening per-image files.  Patch
    embeddings stay per-image so large galleries remain feasible with sparse
    mmap access.

    patches.parquet columns:

        image_id  str       unique image identifier
        row       int16     patch-grid row  (0-indexed, y-axis)
        col       int16     patch-grid column (0-indexed, x-axis)
        y_center  int16     pixel y of patch centre
        x_center  int16     pixel x of patch centre
        labels    object    list[str] — arbitrary multi-labels
        split     str       "train" / "val" / "test" / "unlabeled"

    cls_tokens.parquet columns:

        image_id  str
        labels    object    list[str] — image-level labels
        split     str

    Labels live on the *spatial patch* (image_id, row, col), not on a specific
    layer.  When loading embeddings you choose which block to pull from the
    (L, H, W, D) patch array or the (N, L, D) CLS array; the parquet files are
    the same regardless of layer choice.

    Typical two-stage retrieval::

        # 1. coarse: find relevant images via CLS similarity (no file opens)
        top_images = gallery.retrieve_images(query_cls, k=20, split="train")

        # 2. fine: patch search within those images only
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
        self.root = Path(root)
        self.config     = GalleryConfig.load(self.root / self._CONFIG_FILE)
        self.patches    = pd.read_parquet(self.root / self._PATCHES_FILE)
        self.cls_tokens = pd.read_parquet(self.root / self._CLS_META)
        # mmap the whole CLS array; row i aligns with cls_tokens.iloc[i]
        self._cls_array: np.ndarray = np.load(
            self.root / self._CLS_NPY, mmap_mode="r"
        )  # (N, L, D)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        encoder: DinoEncoder,
        images: list,
        image_ids: list[str],
        out_dir: Path | str,
        split: Union[str, list[str]] = "train",
        image_labels: Optional[dict[str, list[str]]] = None,
        patch_labels: Optional[dict[tuple[str, int, int], list[str]]] = None,
        batch_size: int = 1,
    ) -> "Gallery":
        """Extract features and write the gallery to disk.

        Args:
            encoder:       DinoEncoder configured with layers > 1.
            images:        PIL Images, numpy arrays (H,W,3), or file paths.
            image_ids:     Unique string ID for each image (must be filesystem-safe).
            out_dir:       Output directory (created if absent).
            split:         "train"/"val"/"test"/"unlabeled", or a per-image list.
            image_labels:  {image_id: [label, ...]} applied to every patch and the
                           CLS token of that image (e.g. class membership).
            patch_labels:  {(image_id, row, col): [label, ...]} for specific patches
                           (e.g. annotated keypoints).
            batch_size:    Images per forward pass.
        """
        if len(images) != len(image_ids):
            raise ValueError("images and image_ids must have the same length")

        if isinstance(encoder.layers, int) and encoder.layers == 1:
            raise ValueError(
                "Gallery requires multi-layer output. "
                "Initialise DinoEncoder with layers > 1 or an explicit list of block indices."
            )

        out_dir = Path(out_dir)
        (out_dir / cls._EMB_DIR).mkdir(parents=True, exist_ok=True)

        config       = GalleryConfig.from_encoder(encoder)
        splits       = [split] * len(images) if isinstance(split, str) else list(split)
        image_labels = image_labels or {}
        patch_labels = patch_labels or {}

        patch_rows: list[dict] = []
        cls_rows:   list[dict] = []
        cls_arrays: list[np.ndarray] = []   # accumulate (L, D) per image

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
                cls_arrays.append(cls_np[b])  # (L, D) — gathered into one array below

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

                cls_rows.append({
                    "image_id": img_id,
                    "labels":   list(image_labels.get(img_id, [])),
                    "split":    spl,
                })

        # Single CLS array: (N, L, D) — row order matches cls_tokens.parquet
        np.save(out_dir / cls._CLS_NPY, np.stack(cls_arrays, axis=0))

        patches_df = pd.DataFrame(patch_rows)
        cls_df     = pd.DataFrame(cls_rows)

        for col_name in ("row", "col", "y_center", "x_center"):
            patches_df[col_name] = patches_df[col_name].astype("int16")

        patches_df.to_parquet(out_dir / cls._PATCHES_FILE, index=False)
        cls_df.to_parquet(out_dir / cls._CLS_META, index=False)
        config.save(out_dir / cls._CONFIG_FILE)

        return cls(out_dir)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter(
        self,
        has_labels: Optional[list[str]] = None,
        any_labels: Optional[list[str]] = None,
        image_ids: Optional[list[str]] = None,
        split: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return a filtered view of patches.parquet.

        Args:
            has_labels:  Keep patches that have ALL of these labels.
            any_labels:  Keep patches that have ANY of these labels.
            image_ids:   Restrict to these images.
            split:       "train", "val", "test", or "unlabeled".

        Returns:
            Sub-DataFrame of self.patches sharing its index (pass back to
            load_embeddings, retrieve, or add_labels without re-indexing).
        """
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

    # ------------------------------------------------------------------
    # Loading embeddings
    # ------------------------------------------------------------------

    def load_embeddings(
        self,
        df: pd.DataFrame,
        block_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Load patch embeddings for the rows in df.

        Groups by image, mmaps each per-image npy once, then does a vectorised
        spatial lookup — no full-file reads for sparse patch subsets.

        Args:
            df:         Sub-DataFrame from .filter() (index must align with self.patches).
            block_idx:  Transformer block to use; defaults to the last stored block.

        Returns:
            float32 ndarray of shape (N, D), rows aligned with df.
        """
        if block_idx is None:
            block_idx = self.config.block_indices[-1]
        layer_idx = self.config.layer_idx(block_idx)

        result  = np.empty((len(df), self.config.embed_dim), dtype=np.float32)
        pos_map = {idx: i for i, idx in enumerate(df.index)}

        for img_id, group in df.groupby("image_id"):
            npy   = np.load(self.root / self._EMB_DIR / f"{img_id}.npy", mmap_mode="r")
            layer = np.asarray(npy[layer_idx])  # (H, W, D) — single sequential page-in
            rows  = group["row"].values.astype(int)
            cols  = group["col"].values.astype(int)
            out_pos = [pos_map[idx] for idx in group.index]
            result[out_pos] = layer[rows, cols]

        return result

    def load_cls_embeddings(
        self,
        image_ids: Optional[list[str]] = None,
        block_idx: Optional[int] = None,
    ) -> tuple[list[str], np.ndarray]:
        """Load CLS token embeddings from the single cls_tokens.npy array.

        Args:
            image_ids:  Restrict to these images; None returns all.
            block_idx:  Transformer block; defaults to the last stored block.

        Returns:
            (ids, embeddings) — ids: list[str], embeddings: float32 (N, D).
        """
        if block_idx is None:
            block_idx = self.config.block_indices[-1]
        layer_idx = self.config.layer_idx(block_idx)

        df = self.cls_tokens
        if image_ids is not None:
            df = df[df["image_id"].isin(image_ids)]

        # df.index values are row positions in _cls_array (RangeIndex after parquet load)
        positions = df.index.to_numpy()
        # Load the full layer slice — small enough to be held in RAM for any realistic gallery
        layer_all = np.asarray(self._cls_array[:, layer_idx, :])  # (N_total, D)
        return df["image_id"].tolist(), layer_all[positions]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_images(
        self,
        query: np.ndarray,
        k: int = 10,
        block_idx: Optional[int] = None,
        image_ids: Optional[list[str]] = None,
        split: Optional[str] = None,
    ) -> pd.DataFrame:
        """Find the top-k most similar images to a query CLS embedding.

        Operates entirely on cls_tokens.npy — no per-image patch files are
        opened.  Use the returned image_ids to narrow a subsequent
        patch-level retrieve() call.

        Args:
            query:      (D,) CLS embedding to search against.
            k:          Number of images to return.
            block_idx:  Which block's CLS to use; defaults to the last stored.
            image_ids:  Optional pre-filter to a subset of images.
            split:      "train", "val", "test", or "unlabeled".

        Returns:
            Sub-DataFrame of self.cls_tokens with a "similarity" column added,
            sorted descending, index reset.
        """
        if block_idx is None:
            block_idx = self.config.block_indices[-1]
        layer_idx = self.config.layer_idx(block_idx)

        df = self.cls_tokens
        if split is not None:
            df = df[df["split"] == split]
        if image_ids is not None:
            df = df[df["image_id"].isin(image_ids)]

        if len(df) == 0:
            return df.assign(similarity=pd.Series(dtype=float))

        positions  = df.index.to_numpy()
        layer_all  = np.asarray(self._cls_array[:, layer_idx, :])  # (N_total, D)
        embs       = layer_all[positions]                            # (N_filtered, D)

        q      = query.ravel().astype(np.float32)
        norms  = np.linalg.norm(embs, axis=1)
        q_norm = np.linalg.norm(q)
        sims   = (embs @ q) / (norms * q_norm + 1e-8)

        k       = min(k, len(df))
        top_pos = np.argpartition(sims, -k)[-k:]
        top_pos = top_pos[np.argsort(sims[top_pos])[::-1]]

        result = df.iloc[top_pos].copy()
        result["similarity"] = sims[top_pos]
        return result.reset_index(drop=True)

    def retrieve(
        self,
        query: np.ndarray,
        k: int = 10,
        block_idx: Optional[int] = None,
        has_labels: Optional[list[str]] = None,
        any_labels: Optional[list[str]] = None,
        image_ids: Optional[list[str]] = None,
        split: Optional[str] = None,
    ) -> pd.DataFrame:
        """Find the top-k most cosine-similar patches to a query embedding.

        Filter kwargs narrow the candidate set before loading any patch files.

        Returns:
            Sub-DataFrame of self.patches with a "similarity" column appended,
            sorted descending by similarity, index reset.
        """
        df = self.filter(
            has_labels=has_labels,
            any_labels=any_labels,
            image_ids=image_ids,
            split=split,
        )
        if len(df) == 0:
            return df.assign(similarity=pd.Series(dtype=float))

        embs   = self.load_embeddings(df, block_idx=block_idx)
        q      = query.ravel().astype(np.float32)
        norms  = np.linalg.norm(embs, axis=1)
        q_norm = np.linalg.norm(q)
        sims   = (embs @ q) / (norms * q_norm + 1e-8)

        k       = min(k, len(df))
        top_pos = np.argpartition(sims, -k)[-k:]
        top_pos = top_pos[np.argsort(sims[top_pos])[::-1]]

        result = df.iloc[top_pos].copy()
        result["similarity"] = sims[top_pos]
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Label mutation
    # ------------------------------------------------------------------

    def add_labels(
        self,
        subset: pd.DataFrame,
        labels: list[str],
        save: bool = True,
    ) -> None:
        """Add labels to patches identified by subset's index in self.patches.

        Typical workflow::

            kp = gallery.filter(image_ids=["img_001"])
            kp = kp[(kp.row == 5) & (kp.col == 7)]
            gallery.add_labels(kp, ["keypoint", "left_eye"])

        Args:
            subset:  Sub-DataFrame whose index aligns with self.patches.
            labels:  Labels to add (existing labels are preserved).
            save:    Write updated patches.parquet to disk immediately.
        """
        new = set(labels)
        for idx in subset.index:
            self.patches.at[idx, "labels"] = list(set(self.patches.at[idx, "labels"]) | new)
        if save:
            self._save_metadata()

    def remove_labels(
        self,
        subset: pd.DataFrame,
        labels: list[str],
        save: bool = True,
    ) -> None:
        """Remove labels from patches identified by subset's index."""
        remove = set(labels)
        for idx in subset.index:
            self.patches.at[idx, "labels"] = [
                l for l in self.patches.at[idx, "labels"] if l not in remove
            ]
        if save:
            self._save_metadata()

    def _save_metadata(self) -> None:
        self.patches.to_parquet(self.root / self._PATCHES_FILE, index=False)
        self.cls_tokens.to_parquet(self.root / self._CLS_META, index=False)
