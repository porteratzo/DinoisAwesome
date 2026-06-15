"""SAM 3 segmentation service backed by HuggingFace transformers.

Two backends are supported, selected by the USE_TRACKER environment variable:

* **Sam3Model** (default, ``USE_TRACKER`` unset / "0"):
  Supports text, box, and mixed prompts via Sam3Processor.

* **Sam3TrackerModel** (``USE_TRACKER=1``):
  Supports text, box, and point prompts via Sam3TrackerProcessor.
  Uses a different post-processing path (``post_process_masks``).

Both backends use a non-blocking threading.Lock so concurrent requests receive
HTTP 429 rather than queueing up and exhausting GPU memory.
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

    def __init__(self, model_id: str | None = None, use_tracker: bool | None = None) -> None:
        self._model_id: str = model_id or os.environ.get("SAM_MODEL_ID", _DEFAULT_MODEL_ID)
        self._use_tracker: bool = (
            use_tracker
            if use_tracker is not None
            else os.environ.get("USE_TRACKER", "0").strip().lower() in ("1", "true", "yes")
        )
        self._lock = threading.Lock()
        self._processing = False
        self._processor: Any = None
        self._model: Any = None
        self._load()

    # ------------------------------------------------------------------
    # Prompt-overlap filtering helpers
    # ------------------------------------------------------------------

    def _build_prompt_region(
        self,
        image_size: tuple[int, int],
        fg_boxes: list[list[int]] | None = None,
        fg_points: list[list[int]] | None = None,
        point_radius: int | None = None,
    ) -> np.ndarray | None:
        """Return a binary (H, W) mask covering the foreground prompt region.

        Args:
            image_size: (H, W) of the image in pixels.
            fg_boxes:   List of [x1, y1, x2, y2] positive-label boxes.
            fg_points:  List of [x, y] positive-label points.
            point_radius: Radius around each point (px). Defaults to
                          ``max(8, min(H, W) // 100)``.

        Returns:
            Binary ndarray or None if no foreground prompts were given.
        """
        H, W = image_size
        region = np.zeros((H, W), dtype=bool)
        has_content = False

        if fg_boxes:
            for x1, y1, x2, y2 in fg_boxes:
                region[max(0, y1) : min(H, y2), max(0, x1) : min(W, x2)] = True
                has_content = True

        if fg_points:
            r = point_radius if point_radius is not None else max(8, min(H, W) // 100)
            yy, xx = np.ogrid[:H, :W]
            for x, y in fg_points:
                region |= (xx - x) ** 2 + (yy - y) ** 2 <= r**2
                has_content = True

        return region if has_content else None

    def _filter_masks_by_overlap(
        self,
        masks: list[np.ndarray],
        prompt_region: np.ndarray,
    ) -> list[np.ndarray]:
        """Return the single mask that overlaps most with *prompt_region*.

        The overlap score is the fraction of the prompt region covered by the
        mask.  Always returns at least one mask even if overlap is zero.
        """
        prompt_area = int(prompt_region.sum())
        if prompt_area == 0 or not masks:
            return masks[:1]

        best = max(masks, key=lambda m: int(np.logical_and(m, prompt_region).sum()))
        return [best]

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def use_tracker(self) -> bool:
        """True when Sam3TrackerModel is active; False for Sam3Model."""
        return self._use_tracker

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _load(self) -> None:
        # torch and transformers are imported here so that app.py can call
        # logging.basicConfig() before this module is imported.
        if self._use_tracker:
            from transformers import (  # type: ignore[import-untyped]
                Sam3TrackerModel,
                Sam3TrackerProcessor,
            )

            _log.info(
                "Loading SAM 3 Tracker model '%s' (USE_TRACKER=1) — "
                "first run may download several GB of weights",
                self._model_id,
            )
            self._processor = Sam3TrackerProcessor.from_pretrained(self._model_id)
            self._model = Sam3TrackerModel.from_pretrained(self._model_id, device_map="auto")
        else:
            from transformers import Sam3Model, Sam3Processor  # type: ignore[import-untyped]

            _log.info(
                "Loading SAM 3 model '%s' — first run may download several GB of weights",
                self._model_id,
            )
            self._processor = Sam3Processor.from_pretrained(self._model_id)
            self._model = Sam3Model.from_pretrained(self._model_id, device_map="auto")

        self._model.eval()
        _log.info(
            "SAM 3 %s ready on device map: %s",
            "Tracker" if self._use_tracker else "",
            getattr(self._model, "hf_device_map", "N/A"),
        )

    @property
    def is_processing(self) -> bool:
        """True while an inference call is executing."""
        return self._processing

    # ------------------------------------------------------------------
    # Public segment methods
    # ------------------------------------------------------------------

    def segment_with_text(self, image: Image.Image, text: str) -> list[np.ndarray]:
        """Segment using a free-text concept prompt.

        Supported by both Sam3Model and Sam3TrackerModel.
        """
        inputs = self._processor(images=image, text=text, return_tensors="pt")
        return self._run(image, inputs)

    # ------------------------------------------------------------------
    # Embedding-reuse methods (Sam3Model only)
    # ------------------------------------------------------------------

    def get_vision_embeddings(self, image: Image.Image) -> Any:
        """Pre-compute vision embeddings for reuse across multiple prompts.

        Only supported by Sam3Model (``use_tracker=False``).  Call this once per
        image, then pass the returned object to :meth:`segment_with_embeddings`
        for each text or box concept query.  Avoids re-running the heavy vision
        backbone on every prompt::

            embeds = service.get_vision_embeddings(img)
            masks_cat = service.segment_with_embeddings(img, "cat", embeds)
            masks_dog = service.segment_with_embeddings(img, "dog", embeds)

        Raises:
            RuntimeError: If ``use_tracker`` is True or the model is busy.
        """
        if self._use_tracker:
            raise RuntimeError("get_vision_embeddings requires Sam3Model (use_tracker=False)")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("busy")
        self._processing = True
        try:
            import torch

            device = next(self._model.parameters()).device
            pixel_inputs = self._processor(images=image, return_tensors="pt")
            pixel_values = pixel_inputs["pixel_values"].to(device)
            with torch.inference_mode():
                vision_embeds = self._model.get_vision_features(pixel_values=pixel_values)
            _log.info("Vision embeddings computed for image %s", image.size)
            return vision_embeds
        finally:
            self._processing = False
            self._lock.release()

    def segment_with_embeddings(
        self,
        image: Image.Image,
        text: str,
        vision_embeds: Any,
    ) -> list[np.ndarray]:
        """Segment using pre-computed vision embeddings + a text prompt.

        Use together with :meth:`get_vision_embeddings` to query the same image
        with multiple text concepts without re-running the vision backbone each
        time.  Only supported by Sam3Model (``use_tracker=False``).

        Args:
            image:        The same PIL image that produced ``vision_embeds``.
            text:         The concept string to segment (e.g. ``"cat"``).
            vision_embeds: Output of :meth:`get_vision_embeddings`.

        Raises:
            RuntimeError: If ``use_tracker`` is True or the model is busy.
        """
        if self._use_tracker:
            raise RuntimeError("segment_with_embeddings requires Sam3Model (use_tracker=False)")

        import torch

        device = next(self._model.parameters()).device
        text_inputs = self._processor(text=text, return_tensors="pt")
        # Exclude pixel_values — vision_embeds replaces the backbone pass.
        inputs_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in text_inputs.items()
            if k != "pixel_values"
        }
        return self._run(image, inputs_dev, extra_model_kwargs={"vision_embeds": vision_embeds})

    # ------------------------------------------------------------------
    # Iterative mask refinement (Sam3TrackerModel only)
    # ------------------------------------------------------------------

    def refine_mask(
        self,
        image: Image.Image,
        mask: np.ndarray,
        points: list[list[int]],
        labels: list[int],
    ) -> list[np.ndarray]:
        """Iteratively refine a prior mask using additional click prompts.

        Passes the existing mask as ``input_masks`` alongside fresh point
        clicks so the tracker can tighten or expand the segmentation boundary
        without starting from scratch.  Only supported by Sam3TrackerModel
        (``use_tracker=True``).

        Args:
            image:  The same PIL image that produced ``mask``.
            mask:   (H, W) bool array — the mask to refine.
            points: Additional [x, y] refinement click coordinates.
            labels: Per-point label (1 = include, 0 = exclude).

        Raises:
            RuntimeError: If ``use_tracker`` is False or the model is busy.
        """
        if not self._use_tracker:
            raise RuntimeError("refine_mask requires Sam3TrackerModel (use_tracker=True)")

        import torch

        # Treat all points as a single object: [1, 1, N, 2] / [1, 1, N]
        input_points = [[[[x, y] for x, y in points]]]
        input_labels = [[[lbl for lbl in labels]]]
        inputs = self._processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        )
        # input_masks expected shape: [batch, num_objects, H, W]
        mask_tensor = torch.from_numpy(mask.astype("float32")).unsqueeze(0).unsqueeze(0)
        return self._run(image, inputs, extra_model_kwargs={"input_masks": mask_tensor})

    def segment_with_points(
        self,
        image: Image.Image,
        points: list[list[int]],
        labels: list[int],
        filter_by_prompt: bool = False,
    ) -> list[np.ndarray]:
        """Segment using click-point prompts.

        * **Sam3Model**: points are converted to small bounding boxes centred on
          each click (Sam3Processor has no native input_points argument).
        * **Sam3TrackerModel**: points are passed directly as ``input_points`` /
          ``input_labels`` in the 4-D / 3-D nesting the tracker expects.

        Args:
            points:           List of [x, y] pixel coordinates.
            labels:           Per-point label (1 = foreground, 0 = background).
            filter_by_prompt: If True, return only the mask that overlaps most
                              with the foreground point region.
        """
        H, W = image.height, image.width
        prompt_region = None

        if self._use_tracker:
            # Sam3TrackerProcessor expects one object slot per instance:
            #   input_points: (image, object, points_per_object, coord)
            #   input_labels: (image, object, points_per_object)
            # Each point becomes its own object → N masks back.
            input_points = [[[[x, y]] for x, y in points]]
            input_labels = [[[lbl] for lbl in labels]]
            inputs = self._processor(
                images=image,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt",
            )
            if filter_by_prompt:
                fg_points = [p for p, lbl in zip(points, labels) if lbl == 1]
                prompt_region = self._build_prompt_region((H, W), fg_points=fg_points)
        else:
            # Sam3Processor: convert points to tiny boxes.
            half = max(8, min(W, H) // 100)
            boxes: list[list[int]] = []
            for x, y in points:
                boxes.append([max(0, x - half), max(0, y - half), min(W, x + half), min(H, y + half)])
            inputs = self._processor(
                images=image,
                input_boxes=[boxes],
                input_boxes_labels=[labels],
                return_tensors="pt",
            )
            if filter_by_prompt:
                fg_points = [p for p, lbl in zip(points, labels) if lbl == 1]
                prompt_region = self._build_prompt_region(
                    (H, W), fg_points=fg_points, point_radius=half
                )

        return self._run(image, inputs, prompt_region=prompt_region)

    def segment_with_boxes(
        self,
        image: Image.Image,
        boxes: list[list[int]],
        labels: list[int],
        filter_by_prompt: bool = False,
    ) -> list[np.ndarray]:
        """Segment using one or more bounding-box visual prompts.

        * **Sam3Model**: boxes passed directly.
        * **Sam3TrackerModel**: boxes are converted to their centre points
          (the tracker has no native box input).

        Args:
            boxes:            List of [x1, y1, x2, y2] boxes in pixel coords.
            labels:           Per-box label (1 = include, 0 = exclude).
            filter_by_prompt: If True, return only the mask that overlaps most
                              with the union of foreground (label=1) boxes.
        """
        H, W = image.height, image.width
        prompt_region = None

        if self._use_tracker:
            # Tracker uses input_boxes natively: shape (batch, num_objects, 4).
            # It has no box-label concept — only positive (label=1) boxes are passed.
            fg_boxes = [b for b, lbl in zip(boxes, labels) if lbl == 1]
            inputs = self._processor(
                images=image,
                input_boxes=[fg_boxes],
                return_tensors="pt",
            )
        else:
            inputs = self._processor(
                images=image,
                input_boxes=[boxes],
                input_boxes_labels=[labels],
                return_tensors="pt",
            )

        if filter_by_prompt:
            fg_boxes = [b for b, lbl in zip(boxes, labels) if lbl == 1]
            prompt_region = self._build_prompt_region((H, W), fg_boxes=fg_boxes)

        return self._run(image, inputs, prompt_region=prompt_region)

    def segment_with_points_and_boxes(
        self,
        image: Image.Image,
        points: list[list[int]],
        point_labels: list[int],
        boxes: list[list[int]],
        box_labels: list[int],
        filter_by_prompt: bool = False,
    ) -> list[np.ndarray]:
        """Segment using a native combination of click-point and bounding-box prompts.

        **Sam3TrackerModel** (``use_tracker=True``):
            Both prompt types are forwarded in a single processor call.
            ``input_points`` / ``input_labels`` carry all click prompts as a
            single-object slot (shape ``[1, 1, N, 2]`` / ``[1, 1, N]``).
            ``input_boxes`` carries the axis-aligned union of all positive-label
            boxes (shape ``[1, 1, 4]``) so the ``num_objects`` dimension matches.
            Negative-label boxes are omitted because the tracker has no box-label
            concept; they are expressed instead through negative click labels.

        **Sam3Model** (``use_tracker=False``):
            ``Sam3Processor`` has no native ``input_points`` argument, so each
            click is converted to a tiny bounding box centred on the click
            coordinate (same radius as :meth:`segment_with_points`) and merged
            with the explicit boxes before calling the processor.

        Args:
            points:           List of [x, y] pixel coordinates (click prompts).
            point_labels:     Per-point label (1 = foreground, 0 = background).
            boxes:            List of [x1, y1, x2, y2] explicit bounding boxes.
            box_labels:       Per-box label (1 = include, 0 = exclude).
            filter_by_prompt: If True, return only the mask that overlaps most
                              with the union of all foreground prompts.
        """
        H, W = image.height, image.width
        prompt_region = None

        if self._use_tracker:
            # ── Sam3TrackerModel: native multi-modal prompt ───────────────
            fg_boxes = [b for b, lbl in zip(boxes, box_labels) if lbl == 1]

            processor_kwargs: dict = {"images": image, "return_tensors": "pt"}

            if points:
                # All clicks in one object slot:
                # input_points  → [batch=1, objects=1, pts_per_obj=N, coords=2]
                # input_labels  → [batch=1, objects=1, pts_per_obj=N]
                processor_kwargs["input_points"] = [[points]]
                processor_kwargs["input_labels"] = [[point_labels]]

            if fg_boxes:
                if points:
                    # Merge all positive boxes into their axis-aligned union so
                    # the num_objects dimension equals 1 on both sides.
                    x1 = min(b[0] for b in fg_boxes)
                    y1 = min(b[1] for b in fg_boxes)
                    x2 = max(b[2] for b in fg_boxes)
                    y2 = max(b[3] for b in fg_boxes)
                    # input_boxes → [batch=1, objects=1, coords=4]
                    processor_kwargs["input_boxes"] = [[[x1, y1, x2, y2]]]
                else:
                    # No points — pass each box as its own object (standard tracker behaviour).
                    processor_kwargs["input_boxes"] = [fg_boxes]

            inputs = self._processor(**processor_kwargs)

        else:
            # ── Sam3Model: convert clicks to micro-boxes and merge ────────
            half = max(8, min(W, H) // 100)
            all_boxes: list[list[int]] = list(boxes)
            all_labels: list[int] = list(box_labels)
            for (x, y), lbl in zip(points, point_labels):
                all_boxes.append(
                    [max(0, x - half), max(0, y - half), min(W, x + half), min(H, y + half)]
                )
                all_labels.append(lbl)
            inputs = self._processor(
                images=image,
                input_boxes=[all_boxes],
                input_boxes_labels=[all_labels],
                return_tensors="pt",
            )

        if filter_by_prompt:
            fg_boxes_filter = [b for b, lbl in zip(boxes, box_labels) if lbl == 1]
            fg_pts_filter = [p for p, lbl in zip(points, point_labels) if lbl == 1]
            prompt_region = self._build_prompt_region(
                (H, W),
                fg_boxes=fg_boxes_filter or None,
                fg_points=fg_pts_filter or None,
            )

        return self._run(image, inputs, prompt_region=prompt_region)

    def segment_mixed(
        self,
        image: Image.Image,
        text: str,
        boxes: list[list[int]],
        labels: list[int],
        filter_by_prompt: bool = False,
    ) -> list[np.ndarray]:
        """Segment using text + box prompts together.

        Supported by both Sam3Model and Sam3TrackerModel.  For the tracker,
        only positive-label (label=1) boxes are forwarded since it has no
        box-label concept.
        """
        fg_boxes = [b for b, lbl in zip(boxes, labels) if lbl == 1]

        if self._use_tracker:
            inputs = self._processor(
                images=image,
                text=text,
                input_boxes=[fg_boxes],
                return_tensors="pt",
            )
        else:
            inputs = self._processor(
                images=image,
                text=text,
                input_boxes=[boxes],
                input_boxes_labels=[labels],
                return_tensors="pt",
            )

        prompt_region = None
        if filter_by_prompt:
            prompt_region = self._build_prompt_region(
                (image.height, image.width), fg_boxes=fg_boxes
            )

        return self._run(image, inputs, prompt_region=prompt_region)

    # ------------------------------------------------------------------
    # Shared inference core
    # ------------------------------------------------------------------

    def _run(
        self,
        image: Image.Image,
        inputs: dict[str, Any],
        prompt_region: np.ndarray | None = None,
        extra_model_kwargs: dict[str, Any] | None = None,
    ) -> list[np.ndarray]:
        """Run forward pass and post-process; raises RuntimeError('busy') if locked.

        Dispatches to the appropriate post-processing path depending on whether
        Sam3Model or Sam3TrackerModel is active.

        Args:
            prompt_region:      Optional (H, W) binary mask.  When provided, the
                                returned list is narrowed to the single mask that
                                overlaps most with this region.
            extra_model_kwargs: Additional keyword arguments forwarded directly to
                                the model call (e.g. ``vision_embeds``,
                                ``input_masks``, ``image_embeddings``).  Tensor
                                values are moved to the model's primary device
                                automatically.
        """
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
                kwargs: dict[str, Any] = {"multimask_output": False} if self._use_tracker else {}
                if extra_model_kwargs:
                    kwargs.update(
                        {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in extra_model_kwargs.items()
                        }
                    )
                outputs = self._model(**inputs_dev, **kwargs)

            if self._use_tracker:
                masks = self._postprocess_tracker(outputs, inputs_dev)
            else:
                masks = self._postprocess_sam3(outputs, image)

            if prompt_region is not None and masks:
                masks = self._filter_masks_by_overlap(masks, prompt_region)

            _log.info(
                "Inference complete — %d mask(s) returned (filter_by_prompt=%s)",
                len(masks),
                prompt_region is not None,
            )
            return masks

        finally:
            self._processing = False
            self._lock.release()

    def _postprocess_sam3(self, outputs: Any, image: Image.Image) -> list[np.ndarray]:
        """Post-process Sam3Model outputs via post_process_instance_segmentation."""
        target_sizes = [(image.height, image.width)]
        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=target_sizes,
        )
        if results and "masks" in results[0]:
            return [m.cpu().numpy().astype(bool) for m in results[0]["masks"]]
        return []

    def _postprocess_tracker(self, outputs: Any, inputs_dev: dict[str, Any]) -> list[np.ndarray]:
        """Post-process Sam3TrackerModel outputs via post_process_masks.

        With multimask_output=False the tracker returns pred_masks of shape
        (num_objects, 1, H, W) — one mask per object/point.  We return them as
        a flat list of (H, W) bool arrays, one entry per instance.
        """
        original_sizes = inputs_dev.get("original_sizes")
        masks_tensor = self._processor.post_process_masks(
            outputs.pred_masks.cpu(), original_sizes
        )[0]
        # masks_tensor shape: (num_objects, 1, H, W)
        return [masks_tensor[obj_idx, 0].numpy().astype(bool) for obj_idx in range(masks_tensor.shape[0])]
