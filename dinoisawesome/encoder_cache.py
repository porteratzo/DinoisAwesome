"""Transparent, content-addressed disk cache for `DinoEncoder`.

Research scripts under `experiments/` repeatedly re-encode the same images: the
same reference/query images across different scripts, and — since augmentations
are seeded — byte-identical augmented crops across reruns of the same script.
`EncoderWithCache` wraps a `DinoEncoder` and skips inference for anything already
encoded under the same model/config, keyed on the actual pixel content rather
than a file path (so in-memory-only crops are cacheable too).

Storage follows the same convention as `gallery.py`: a pandas/parquet index plus
one numpy file per cached item, under `<cache_dir>/<fingerprint digest>/`.

Caveats:
    - Cached tensors are always returned as float32, even when the wrapped
      encoder's own `amp`/`model_dtype` would otherwise produce bfloat16/float16.
      This is necessary because numpy has no native bfloat16 storage, and it
      keeps cache-hit and cache-miss entries consistently stackable within the
      same batch.
    - bfloat16/autocast matmuls are not guaranteed bit-exact across runs or
      hardware, so a cache hit reuses one earlier numeric realization rather
      than guaranteeing what a fresh run would produce bit-for-bit.
    - Only PIL Image / (H, W, 3) numpy array inputs (single, or a list/batch of
      them) are cached per-item. A pre-processed `torch.Tensor` batch is passed
      straight through to the wrapped encoder, uncached, since `DinoEncoder` has
      no supported way to reassemble a list of raw per-item tensors.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .encoder import _MODEL_NAMES, DinoEncoder, ExtractorOutput

_log = logging.getLogger(__name__)

_INDEX_COLUMNS = [
    "key",
    "image_hash",
    "layers_key",
    "debias",
    "dtype",
    "cls_shape",
    "patches_shape",
    "created_at",
]


@dataclass
class EncoderFingerprint:
    """Everything about a `DinoEncoder` that can change its numeric output.

    Used to namespace the on-disk cache: two encoders with the same fingerprint
    are guaranteed to produce the same features for the same image + call params,
    so their cached entries can be shared; encoders that differ in any of these
    fields get separate cache subdirectories.
    """

    model_name: str  # e.g. "dinov3_vitl16"
    version: str  # "v2" or "v3"
    size: str  # "small" / "base" / "large" / "giant"
    img_size: int
    model_dtype: str  # str(torch.dtype) or "None"
    svd_components: int
    weights_signature: str  # resolved weights path+mtime, or "hub:<model_name>"
    schema_version: str = "1.0"

    @property
    def digest(self) -> str:
        """Short, stable hash of every field — names this fingerprint's cache dir."""
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> EncoderFingerprint:
        return cls(**json.loads(path.read_text()))

    @classmethod
    def from_encoder(cls, encoder: DinoEncoder) -> EncoderFingerprint:
        return cls(
            model_name=_MODEL_NAMES[(encoder.version, encoder.size)],
            version=encoder.version,
            size=encoder.size,
            img_size=encoder.img_size,
            model_dtype=str(encoder.model_dtype),
            svd_components=encoder.svd_components,
            weights_signature=encoder.weights_signature,
        )


def image_content_hash(item: Image.Image | np.ndarray | torch.Tensor) -> str:
    """Hash one image's actual pixel content (not a file path).

    Args:
        item: A single PIL Image, an (H, W, 3) numpy array, or an unbatched
              (3, H, W) torch tensor.

    Returns:
        A 32-character hex digest.
    """
    if isinstance(item, Image.Image):
        rgb = item.convert("RGB")
        payload = rgb.tobytes() + str(rgb.size).encode()
    elif isinstance(item, np.ndarray):
        payload = item.tobytes() + str(item.shape).encode() + str(item.dtype).encode()
    elif isinstance(item, torch.Tensor):
        arr = item.detach().float().cpu().numpy()
        payload = arr.tobytes() + str(arr.shape).encode() + str(arr.dtype).encode()
    else:
        raise TypeError(f"Unsupported image type for hashing: {type(item)!r}")
    return hashlib.sha256(payload).hexdigest()[:32]


def _layers_key(layers: int | list[int]) -> str:
    """Canonicalise the effective `layers` value into a stable cache-key fragment.

    Order is preserved (not sorted) since the output's layer axis follows the
    order `layers` was given in, not sorted order.
    """
    if isinstance(layers, int):
        return f"n{layers}"
    return ",".join(str(i) for i in layers)


class EncoderWithCache:
    """Drop-in wrapper around a `DinoEncoder` that caches per-image encodings to disk.

    Usage::

        encoder = DinoEncoder(version="v3", size="large", img_size=768, layers=[23])
        cached = EncoderWithCache(encoder, cache_dir="data/encoding_cache")
        out = cached(images, layers=[23], debias=True)  # same call shape as DinoEncoder

    Any attribute not defined on this wrapper (e.g. `.max_batch_size`, `.device`)
    falls through to the wrapped encoder, so existing scripts that read those
    attributes keep working unmodified.
    """

    def __init__(self, encoder: DinoEncoder, cache_dir: str | Path | None = None) -> None:
        resolved = cache_dir if cache_dir is not None else os.environ.get("DINO_ENCODING_CACHE_DIR")
        if resolved is None:
            raise ValueError(
                "cache_dir must be given, or DINO_ENCODING_CACHE_DIR set in the environment"
            )

        self._encoder = encoder
        self._fingerprint = EncoderFingerprint.from_encoder(encoder)
        self._fingerprint_dir = Path(resolved) / self._fingerprint.digest
        self._embeddings_dir = self._fingerprint_dir / "embeddings"
        self._embeddings_dir.mkdir(parents=True, exist_ok=True)

        fp_path = self._fingerprint_dir / "fingerprint.json"
        if not fp_path.exists():
            self._fingerprint.save(fp_path)

        self._index_path = self._fingerprint_dir / "index.parquet"
        if self._index_path.exists():
            self._index = pd.read_parquet(self._index_path)
        else:
            self._index = pd.DataFrame(columns=_INDEX_COLUMNS)

        _log.info(
            "EncoderWithCache: fingerprint=%s dir=%s (%d entries already cached)",
            self._fingerprint.digest,
            self._fingerprint_dir,
            len(self._index),
        )

    def __getattr__(self, name: str):
        if name == "_encoder":
            raise AttributeError(name)
        return getattr(self._encoder, name)

    def __call__(
        self,
        images: torch.Tensor | Image.Image | np.ndarray | list,
        layers: int | list[int] | None = None,
        debias: bool = False,
    ) -> ExtractorOutput:
        return self.forward(images, layers=layers, debias=debias)

    def forward(
        self,
        images: torch.Tensor | Image.Image | np.ndarray | list,
        layers: int | list[int] | None = None,
        debias: bool = False,
    ) -> ExtractorOutput:
        encoder = self._encoder

        if isinstance(images, torch.Tensor):
            _log.debug("EncoderWithCache: tensor input bypasses caching")
            return encoder(images, layers=layers, debias=debias)

        n = layers if layers is not None else encoder.layers
        layers_key = _layers_key(n)

        chunks = encoder._split_into_chunks(images, chunk_size=1)
        items = [chunk[0] for chunk in chunks]  # one raw single-image item per position

        keys = [f"{image_content_hash(item)}__{layers_key}__{int(debias)}" for item in items]

        cls_list: list[torch.Tensor | None] = [None] * len(items)
        patches_list: list[torch.Tensor | None] = [None] * len(items)
        miss_positions: list[int] = []

        for i, key in enumerate(keys):
            npz_path = self._embeddings_dir / f"{key}.npz"
            if npz_path.exists():
                data = np.load(npz_path)
                cls_list[i] = torch.from_numpy(data["cls"]).to(device=encoder.device)
                patches_list[i] = torch.from_numpy(data["patches"]).to(device=encoder.device)
            else:
                miss_positions.append(i)

        _log.info(
            "EncoderWithCache: %d/%d hits (layers=%s debias=%s)",
            len(items) - len(miss_positions),
            len(items),
            layers_key,
            debias,
        )

        if miss_positions:
            miss_items = [items[i] for i in miss_positions]
            out = encoder(miss_items, layers=layers, debias=debias)

            new_rows = []
            for j, i in enumerate(miss_positions):
                cls_i = out.cls[j].float()
                patches_i = out.patches[j].float()
                cls_list[i] = cls_i
                patches_list[i] = patches_i

                key = keys[i]
                np.savez(
                    self._embeddings_dir / f"{key}.npz",
                    cls=cls_i.cpu().numpy(),
                    patches=patches_i.cpu().numpy(),
                )
                new_rows.append(
                    {
                        "key": key,
                        "image_hash": key.split("__")[0],
                        "layers_key": layers_key,
                        "debias": debias,
                        "dtype": str(cls_i.dtype),
                        "cls_shape": str(tuple(cls_i.shape)),
                        "patches_shape": str(tuple(patches_i.shape)),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

            self._index = pd.concat([self._index, pd.DataFrame(new_rows)], ignore_index=True)
            self._index.to_parquet(self._index_path, index=False)

        cls = torch.stack(cast("list[torch.Tensor]", cls_list), dim=0)
        patches = torch.stack(cast("list[torch.Tensor]", patches_list), dim=0)
        return ExtractorOutput(cls=cls, patches=patches)


def inspect_cache(cache_dir: str | Path) -> pd.DataFrame:
    """Summarise every encoder fingerprint cached under *cache_dir*.

    Answers "what have I already encoded" at a glance: one row per distinct
    model/config combination that has ever been cached here.

    Args:
        cache_dir: Root cache directory (same value passed to `EncoderWithCache`).

    Returns:
        DataFrame with columns: digest, model_name, version, size, img_size,
        model_dtype, weights_signature, n_entries, total_bytes.
    """
    cache_dir = Path(cache_dir)
    columns = [
        "digest",
        "model_name",
        "version",
        "size",
        "img_size",
        "model_dtype",
        "weights_signature",
        "n_entries",
        "total_bytes",
    ]
    if not cache_dir.exists():
        return pd.DataFrame(columns=columns)

    rows = []
    for fp_dir in sorted(p for p in cache_dir.iterdir() if p.is_dir()):
        fp_file = fp_dir / "fingerprint.json"
        if not fp_file.exists():
            continue
        fingerprint = EncoderFingerprint.load(fp_file)

        index_file = fp_dir / "index.parquet"
        n_entries = len(pd.read_parquet(index_file)) if index_file.exists() else 0

        emb_dir = fp_dir / "embeddings"
        total_bytes = (
            sum(f.stat().st_size for f in emb_dir.glob("*.npz")) if emb_dir.exists() else 0
        )

        rows.append(
            {
                "digest": fp_dir.name,
                "model_name": fingerprint.model_name,
                "version": fingerprint.version,
                "size": fingerprint.size,
                "img_size": fingerprint.img_size,
                "model_dtype": fingerprint.model_dtype,
                "weights_signature": fingerprint.weights_signature,
                "n_entries": n_entries,
                "total_bytes": total_bytes,
            }
        )
    return pd.DataFrame(rows, columns=columns)
