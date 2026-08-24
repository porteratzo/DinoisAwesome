"""The four anomaly-detection methods under comparison.

Each method implements the same `AnomalyMethod` interface: `fit(train_paths)` builds
a memory bank from normal images and returns the fit wall-clock time; `predict(path)`
scores one image and returns an image-level score, a full-resolution anomaly map, and
inference latency.

Methods 1-2 (`patchcore`, `anomalydino_v2`) wrap anomalib's own `Patchcore` /
`AnomalyDINO` Lightning modules **unmodified** — no `Engine`/datamodule, the torch
submodule (`model.model`) is driven directly: forward passes in train mode populate
its `embedding_store`, `model.fit()` consolidates the memory bank (this mirrors what
`MemoryBankMixin.on_train_epoch_end` does automatically inside a real training loop —
see anomalib's `models/components/base/memory_bank_module.py`), then forward passes
in eval mode return `InferenceBatch(pred_score, anomaly_map)`. `tep_ai`'s own
inference code (`tep_ai/tep_ai/services/model_service.py::_run_inference_common`)
uses this same "call the model directly" pattern for scoring.

Methods 3-4 (`anomalydino_v3`, `anomalydino_v3_smooth9`) are `dinoisawesome`'s own
`AnomalyHead` + `DinoEncoder(version="v3")` — see `dinoisawesome/anomaly_head.py`.

Masked variants of methods 1-4 (`anomalydino_v2_masked`, `anomalydino_v3_masked`)
turn on PCA-based background suppression (see `dinoisawesome/background_mask.py`,
ported from anomalib's own `AnomalyDINOModel.compute_background_masks`).

The `dinov3_proto_*` family (built by `PrototypeMethod`) is a fifth, different
method: `dinoisawesome`'s `PrototypeAnomalyHead` — CLS-token retrieval of the
nearest normal training image(s) followed by k-means prototype clustering,
instead of one big cross-image patch memory bank. See
`dinoisawesome/prototype_head.py` and `common.parse_prototype_name` for the
naming scheme and parameter grid.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from common import gallery_dir, parse_prototype_name, prototype_method_name
from PIL import Image

from dinoisawesome import AnomalyHead, DinoEncoder, Gallery, PrototypeAnomalyHead

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINO_WEIGHTS_DIR = os.environ.get("DINO_WEIGHTS_DIR")


@dataclass
class ScoreResult:
    score: float
    anomaly_map: np.ndarray  # (H, W) float32
    latency_ms: float


class AnomalyMethod(ABC):
    name: str

    @abstractmethod
    def fit(self, train_paths: list[Path]) -> float:
        """Fit on normal training images. Returns wall-clock fit time in seconds."""

    @abstractmethod
    def predict(self, image_path: Path) -> ScoreResult:
        """Score one test image."""


# ---------------------------------------------------------------------------
# Methods 1-2: anomalib vendor models, driven directly (vendor code untouched)
# ---------------------------------------------------------------------------


def _load_raw_tensor(image_path: Path) -> torch.Tensor:
    """PIL -> (1, 3, H, W) float tensor in [0, 1], full original resolution."""
    image = Image.open(image_path).convert("RGB")
    arr = np.array(image)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0


class _VendorAnomalibMethod(AnomalyMethod):
    """Shared driver for anomalib Lightning modules, called without Engine/Trainer.

    Subclasses only need to build `self.model` (an `AnomalibModule` instance) in
    `__init__`. Preprocessing always goes through the model's own configured
    `pre_processor.transform`, so it matches upstream's intended preprocessing
    exactly.
    """

    model: Any  # anomalib AnomalibModule instance, set by subclass __init__

    def _preprocess(self, image_path: Path) -> torch.Tensor:
        tensor = _load_raw_tensor(image_path).to(DEVICE)
        return self.model.pre_processor.transform(tensor)

    def fit(self, train_paths: list[Path]) -> float:
        self.model.to(DEVICE)
        self.model.model.train()
        t_start = time.perf_counter()
        with torch.no_grad():
            for path in train_paths:
                self.model.model(self._preprocess(path))
        self.model.fit()  # consolidates memory bank (+ coreset subsample for PatchCore)
        return time.perf_counter() - t_start

    def predict(self, image_path: Path) -> ScoreResult:
        self.model.model.eval()
        t_start = time.perf_counter()
        with torch.no_grad():
            output = self.model.model(self._preprocess(image_path))
        latency_ms = (time.perf_counter() - t_start) * 1000
        anomaly_map = output.anomaly_map.squeeze().detach().cpu().float().numpy()
        score = float(output.pred_score.squeeze().detach().cpu())
        return ScoreResult(score=score, anomaly_map=anomaly_map, latency_ms=latency_ms)


class PatchCoreMethod(_VendorAnomalibMethod):
    name = "patchcore"

    def __init__(self, image_size: tuple[int, int] = (256, 256)) -> None:
        from anomalib.models import Patchcore

        self.model = Patchcore(
            pre_processor=Patchcore.configure_pre_processor(image_size=image_size),
            num_neighbors=9,  # PatchCore literature default
        )


class AnomalyDINOv2Method(_VendorAnomalibMethod):
    def __init__(self, image_size: tuple[int, int] = (252, 252), masking: bool = False) -> None:
        from anomalib.models import AnomalyDINO

        self.name = "anomalydino_v2_masked" if masking else "anomalydino_v2"
        self.model = AnomalyDINO(
            pre_processor=AnomalyDINO.configure_pre_processor(image_size=image_size),
            encoder_name="dinov2_vit_small_14",
            num_neighbours=1,
            masking=masking,
        )


# ---------------------------------------------------------------------------
# Methods 3-4: dinoisawesome's own AnomalyHead (DINOv3-backed)
# ---------------------------------------------------------------------------


class _AnomalyDINOv3Method(AnomalyMethod):
    """Shared driver for `dinoisawesome.AnomalyHead` over a DINOv3 encoder.

    `category` and `method` select the on-disk Gallery cache directory so the
    caller controls where embeddings are persisted (see `common.gallery_dir`).
    """

    def __init__(
        self,
        category: str,
        smoothing: bool,
        masking: bool = False,
        img_size: int = 256,
        size: str = "small",
    ) -> None:
        if smoothing and masking:
            self.name = "anomalydino_v3_smooth9_masked"
        elif smoothing:
            self.name = "anomalydino_v3_smooth9"
        elif masking:
            self.name = "anomalydino_v3_masked"
        else:
            self.name = "anomalydino_v3"
        self._category = category
        self._smoothing = smoothing
        self._masking = masking
        self._encoder = DinoEncoder(
            version="v3",
            size=size,
            img_size=img_size,
            layers=1,
            device=DEVICE,
            weights_dir=DINO_WEIGHTS_DIR,
        )
        self._head: AnomalyHead | None = None

    def fit(self, train_paths: list[Path]) -> float:
        t_start = time.perf_counter()
        self._head = AnomalyHead.build(
            encoder=self._encoder,
            images=train_paths,
            image_ids=[p.stem for p in train_paths],
            gallery_dir=gallery_dir(self._category, self.name),
            split="train",
            num_neighbours=1,
            smoothing=self._smoothing,
            masking=self._masking,
        )
        return time.perf_counter() - t_start

    def predict(self, image_path: Path) -> ScoreResult:
        assert self._head is not None, "fit() must be called before predict()"
        t_start = time.perf_counter()
        result = self._head.predict(image_path)
        latency_ms = (time.perf_counter() - t_start) * 1000
        return ScoreResult(
            score=result["score"], anomaly_map=result["anomaly_map"], latency_ms=latency_ms
        )


class AnomalyDINOv3Method(_AnomalyDINOv3Method):
    def __init__(self, category: str, masking: bool = False) -> None:
        super().__init__(category=category, smoothing=False, masking=masking)


class AnomalyDINOv3SmoothMethod(_AnomalyDINOv3Method):
    def __init__(self, category: str) -> None:
        super().__init__(category=category, smoothing=True)


# ---------------------------------------------------------------------------
# Method 5: dinoisawesome's PrototypeAnomalyHead — CLS retrieval + k-means
# prototypes (DINOv3-backed), instead of one big cross-image patch memory bank.
# ---------------------------------------------------------------------------


class PrototypeMethod(AnomalyMethod):
    """Driver for `dinoisawesome.PrototypeAnomalyHead` over a DINOv3 encoder.

    Builds the same kind of Gallery as `_AnomalyDINOv3Method` (CLS tokens + patch
    grids for every normal training image), but scores each query against a
    small set of k-means prototypes drawn from the nearest retrieved normal
    image(s) instead of the full cross-image patch bank.
    """

    def __init__(
        self,
        category: str,
        n_prototypes: int,
        retrieval_k: int,
        aggregation: Literal["max", "avg"],
        masking: bool,
        img_size: int = 256,
        size: str = "small",
    ) -> None:
        self.name = prototype_method_name(n_prototypes, retrieval_k, aggregation, masking)
        self._category = category
        self._n_prototypes = n_prototypes
        self._retrieval_k = retrieval_k
        self._aggregation = aggregation
        self._masking = masking
        self._encoder = DinoEncoder(
            version="v3",
            size=size,
            img_size=img_size,
            layers=1,
            device=DEVICE,
            weights_dir=DINO_WEIGHTS_DIR,
        )
        self._head: PrototypeAnomalyHead | None = None

    def fit(self, train_paths: list[Path]) -> float:
        t_start = time.perf_counter()
        gallery = Gallery.build(
            encoder=self._encoder,
            images=train_paths,
            image_ids=[p.stem for p in train_paths],
            out_dir=gallery_dir(self._category, self.name),
            split="train",
        )
        self._head = PrototypeAnomalyHead(
            gallery=gallery,
            encoder=self._encoder,
            n_prototypes=self._n_prototypes,
            retrieval_k=self._retrieval_k,
            aggregation=self._aggregation,
            masking=self._masking,
            split="train",
        )
        return time.perf_counter() - t_start

    def predict(self, image_path: Path) -> ScoreResult:
        assert self._head is not None, "fit() must be called before predict()"
        t_start = time.perf_counter()
        result = self._head.predict(image_path)
        latency_ms = (time.perf_counter() - t_start) * 1000
        return ScoreResult(
            score=result["score"], anomaly_map=result["anomaly_map"], latency_ms=latency_ms
        )


def build_method(name: str, category: str) -> AnomalyMethod:
    """Factory: construct a fresh method instance for one (category, method) run."""
    if name == "patchcore":
        return PatchCoreMethod()
    if name == "anomalydino_v2":
        return AnomalyDINOv2Method()
    if name == "anomalydino_v2_masked":
        return AnomalyDINOv2Method(masking=True)
    if name == "anomalydino_v3":
        return AnomalyDINOv3Method(category=category)
    if name == "anomalydino_v3_masked":
        return AnomalyDINOv3Method(category=category, masking=True)
    if name == "anomalydino_v3_smooth9":
        return AnomalyDINOv3SmoothMethod(category=category)
    proto_kwargs = parse_prototype_name(name)
    if proto_kwargs is not None:
        return PrototypeMethod(category=category, **proto_kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unknown method: {name!r}")
