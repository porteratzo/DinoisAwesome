"""SAM 3 segmentation service backed by HuggingFace transformers.

Supports text, box, and mixed prompts natively via Sam3Processor.
Uses a non-blocking threading.Lock so concurrent requests receive HTTP 429
rather than queueing up and exhausting GPU memory.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np
from PIL import Image

# Do NOT call logging.basicConfig here — app.py owns that.
_log = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "facebook/sam3"


class SAM3Service:
    """Thread-safe wrapper around Sam3Model / Sam3Processor.

    Model weights are loaded once at construction time.  If a segment*() call
    arrives while inference is already running the lock acquisition fails
    immediately and RuntimeError("busy") is raised; the Flask route returns 429.
    """

    def __init__(self, model_id: str | None = None) -> None:
        self._model_id: str = model_id or os.environ.get("SAM_MODEL_ID", _DEFAULT_MODEL_ID)
        self._lock = threading.Lock()
        self._processing = False
        self._processor: Any = None
        self._model: Any = None
        self._load()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _load(self) -> None:
        # torch and transformers are imported here so that app.py can call
        # logging.basicConfig() before this module is imported.
        from transformers import Sam3Model, Sam3Processor  # type: ignore[import-untyped]

        _log.info(
            "Loading SAM 3 model '%s' — first run may download several GB of weights",
            self._model_id,
        )
        self._processor = Sam3Processor.from_pretrained(self._model_id)
        self._model = Sam3Model.from_pretrained(self._model_id, device_map="auto")
        self._model.eval()
        _log.info("SAM 3 ready on device map: %s", getattr(self._model, "hf_device_map", "N/A"))

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def is_processing(self) -> bool:
        """True while an inference call is executing."""
        return self._processing

    # ------------------------------------------------------------------
    # Public segment methods
    # ------------------------------------------------------------------

    def segment_with_text(self, image: Image.Image, text: str) -> list[np.ndarray]:
        """Segment using a free-text concept prompt."""
        inputs = self._processor(images=image, text=text, return_tensors="pt")
        return self._run(image, inputs)

    def segment_with_boxes(
        self,
        image: Image.Image,
        boxes: list[list[int]],
        labels: list[int],
    ) -> list[np.ndarray]:
        """Segment using one or more bounding-box visual prompts.

        Args:
            boxes:  List of [x1, y1, x2, y2] boxes in pixel coordinates.
            labels: Per-box label (1 = include, 0 = exclude).
        """
        inputs = self._processor(
            images=image,
            input_boxes=[boxes],
            input_boxes_labels=[labels],
            return_tensors="pt",
        )
        return self._run(image, inputs)

    def segment_mixed(
        self,
        image: Image.Image,
        text: str,
        boxes: list[list[int]],
        labels: list[int],
    ) -> list[np.ndarray]:
        """Segment using text + box prompts together."""
        inputs = self._processor(
            images=image,
            text=text,
            input_boxes=[boxes],
            input_boxes_labels=[labels],
            return_tensors="pt",
        )
        return self._run(image, inputs)

    # ------------------------------------------------------------------
    # Shared inference core
    # ------------------------------------------------------------------

    def _run(self, image: Image.Image, inputs: dict[str, Any]) -> list[np.ndarray]:
        """Run forward pass and post-process; raises RuntimeError('busy') if locked."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("busy")
        self._processing = True
        try:
            import torch

            # Move tensor inputs to the model's primary device.
            device = next(self._model.parameters()).device
            inputs_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
            }

            with torch.inference_mode():
                outputs = self._model(**inputs_dev)

            # post_process_instance_segmentation expects target_sizes as list of (H, W).
            target_sizes = [(image.height, image.width)]
            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=target_sizes,
            )

            masks: list[np.ndarray] = []
            if results and "masks" in results[0]:
                masks = [m.cpu().numpy().astype(bool) for m in results[0]["masks"]]

            _log.info("Inference complete — %d mask(s) returned", len(masks))
            return masks

        finally:
            self._processing = False
            self._lock.release()
