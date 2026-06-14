"""Evaluation on the custom_slim dataset: SAM+DINO pipeline with ground-truth metrics.

Loads every image and its ``_mask_good.npz`` ground-truth mask from
``data/custom_slim``.  One sample is used as the exemplar (withheld from
inference); the rest are scored by the SAM+DINO pipeline.

Metrics computed
----------------
* Per-candidate IoU between the SAM-proposal mask and the GT mask.
* Candidate-level Precision-Recall curves for each scoring method (M1/M2/M3).
* Instance-level ROC (TPR/FPR) curves: a candidate is TP when its IoU with the
  GT mask exceeds ``--iou-threshold``; FPR is relative to all non-matching
  candidates.
* IoU and score distribution histograms.
* Summary JSON with AP (area under PR curve) per method and instance-level
  stats at the given IoU threshold.

The SAM mask generator uses the HuggingFace ``transformers`` SAM2 backend
(same as ``notebooks/sam_segmenter.ipynb``) — no local checkpoint files are
required.  The model is downloaded from HuggingFace Hub on first use.

Usage
-----
python scripts/eval_custom_slim.py \\
    --data-dir      DinoisAwesome/data/custom_slim \\
    --exemplar-stem frame_spectacles1_0100 \\
    --text-prompt   "spectacle frame" \\
    --output-dir    results/custom_slim_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from PIL import Image

# ── load .env before anything reads environment variables ──────────────────
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# ── logging first (before torch / sam imports) ─────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── reuse helpers from eval_sam_dino ───────────────────────────────────────
_SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS))
from eval_sam_dino import (  # noqa: E402
    ExemplarFeatures,
    MaskGenerator,
    _bbox_from_mask,
    _cosine_sim,
    _crop_to_bbox,
    _mask_to_patch_grid,
    _masked_prototype,
    _preprocess_mask,
    process_exemplar,
    score_method1,
    score_method2,
    score_method3,
)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from dinoisawesome import DinoEncoder  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace SAM2 mask generator (mirrors notebooks/sam_segmenter.ipynb)
# ─────────────────────────────────────────────────────────────────────────────


def _to_pil_rgb(image: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


class SAMSegmenter:
    """Prompted segmenter backed by HuggingFace transformers SAM3 (or SAM2).

    No local checkpoint file is required — the model is downloaded from
    HuggingFace Hub on first use and cached in ``~/.cache/huggingface``.

    SAM3 is tried first (``Sam3Model`` / ``Sam3Processor``); SAM2 classes are
    used as a fallback so the same class works with both model families.

    Args:
        model_id: HuggingFace model ID.  Default: ``"facebook/sam3"``.
        device:   Torch device string; auto-detected if ``None``.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # bfloat16 on CUDA halves VRAM with negligible quality loss; keep float32 on CPU.
        self._dtype = torch.bfloat16 if self._device.startswith("cuda") else torch.float32
        self._model = None
        self._processor = None

    def _to_device(self, inputs: dict) -> dict:
        """Move inputs to device, casting floating-point tensors to the model dtype."""
        return {
            k: (v.to(device=self._device, dtype=self._dtype) if torch.is_floating_point(v) else v.to(self._device))
            for k, v in inputs.items()
        }

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import os

        token = os.environ.get("HF_TOKEN") or None
        log.info("Loading %s on %s (dtype=%s) …", self._model_id, self._device, self._dtype)
        try:
            from transformers import Sam3Model, Sam3Processor  # type: ignore[import]

            self._processor = Sam3Processor.from_pretrained(self._model_id, token=token)
            self._model = Sam3Model.from_pretrained(
                self._model_id, token=token, torch_dtype=self._dtype
            ).to(self._device)
            log.info("SAM3 ready.")
        except ImportError:
            from transformers import Sam2Model, Sam2Processor  # type: ignore[import]

            log.warning("Sam3Model not found in transformers — falling back to Sam2Model.")
            self._processor = Sam2Processor.from_pretrained(self._model_id, token=token)
            self._model = Sam2Model.from_pretrained(
                self._model_id, token=token, torch_dtype=self._dtype
            ).to(self._device)
            log.info("SAM2 ready (fallback).")
        self._model.eval()

    def _run(self, pil_img: Image.Image, proc_kwargs: dict) -> list[dict]:
        """Single forward pass; returns masks sorted by IoU score descending."""
        self._ensure_loaded()
        inputs = self._processor(images=pil_img, return_tensors="pt", **proc_kwargs)
        original_sizes = inputs.pop("original_sizes")
        inputs.pop("reshaped_input_sizes", None)
        model_inputs = self._to_device(inputs)

        with torch.no_grad():
            outputs = self._model(**model_inputs)  # type: ignore[misc]

        orig_h = int(original_sizes[0, 0])
        orig_w = int(original_sizes[0, 1])
        b, obj, nm, lh, lw = outputs.pred_masks.shape
        logits = outputs.pred_masks.cpu().reshape(b * obj * nm, 1, lh, lw).float()
        upscaled = F.interpolate(logits, (orig_h, orig_w), mode="bilinear", align_corners=False)
        masks = (upscaled.squeeze(1) > 0).numpy()  # (N, H, W) bool
        scores = outputs.iou_scores.cpu().reshape(b * obj * nm).numpy()

        return sorted(
            [{"mask": masks[i], "score": float(scores[i])} for i in range(len(scores))],
            key=lambda r: r["score"],
            reverse=True,
        )

    def segment_with_points(
        self,
        image: np.ndarray | Image.Image,
        points: list[list[int]],
        labels: list[int],
        existing_mask: np.ndarray | None = None,
    ) -> list[dict]:
        """Segment with foreground/background click points."""
        pil = _to_pil_rgb(image)
        kw: dict = {"input_points": [[points]], "input_labels": [[labels]]}
        if existing_mask is not None:
            kw["input_masks"] = [[existing_mask.astype(np.float32)]]
        return self._run(pil, kw)

    def segment_with_text(
        self,
        image: np.ndarray | Image.Image,
        text: str,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> list[dict]:
        """Segment all instances matching *text* using SAM3's text-prompt API.

        Uses the processor's ``post_process_instance_segmentation`` to decode
        the model output — the correct path for SAM3 text prompts.  Returns
        ``[{"mask": bool[H,W], "score": float}, …]`` sorted by score descending.
        """
        self._ensure_loaded()
        pil = _to_pil_rgb(image)
        raw_inputs = self._processor(images=pil, text=text, return_tensors="pt")
        inputs = self._to_device(raw_inputs)

        with torch.no_grad():
            outputs = self._model(**inputs)  # type: ignore[misc]

        target_sizes = inputs.get("original_sizes").tolist()
        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=score_threshold,
            mask_threshold=mask_threshold,
            target_sizes=target_sizes,
        )[0]

        entries = [
            {"mask": results["masks"][i].cpu().numpy().astype(bool), "score": float(results["scores"][i])}
            for i in range(len(results["masks"]))
        ]
        return sorted(entries, key=lambda r: r["score"], reverse=True)


class HFSAMTextGenerator:
    """Automatic mask generator that uses SAM3's text-prompt interface.

    A single forward pass with the text prompt returns all candidate instance
    masks for the image.  Tiny masks are discarded; greedy mask-IoU NMS removes
    duplicates among the survivors.

    Implements the ``MaskGenerator`` protocol expected by the evaluation loop:
    ``generate(image) -> list[dict]`` where each dict contains
    ``"segmentation"`` (bool [H, W]) and ``"bbox"`` ([x, y, w, h]).

    Args:
        segmenter:         An initialised ``SAMSegmenter`` instance (SAM3).
        text_prompt:       Free-form description of the object to find,
                           e.g. ``"spectacle frame"``.
        score_threshold:   Minimum confidence score for a detected instance.
        mask_threshold:    Logit threshold for converting soft masks to binary.
        nms_iou_threshold: Masks with IoU above this are considered duplicates.
        min_area_frac:     Discard masks smaller than this fraction of the image.
        sam_input_size:    If set, images are downscaled so their longest side
                           equals this value before being passed to SAM3.
                           Returned masks are upsampled back to the original
                           resolution.  Reduces peak VRAM significantly for
                           high-resolution inputs (e.g. 1024 → ~4× less VRAM
                           than passing a 3840×2160 image directly).
    """

    def __init__(
        self,
        segmenter: SAMSegmenter,
        text_prompt: str,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        nms_iou_threshold: float = 0.70,
        min_area_frac: float = 0.001,
        sam_input_size: int | None = None,
    ) -> None:
        self._seg = segmenter
        self._text_prompt = text_prompt
        self._score_threshold = score_threshold
        self._mask_threshold = mask_threshold
        self._nms_iou_threshold = nms_iou_threshold
        self._min_area_frac = min_area_frac
        self._sam_input_size = sam_input_size

    @staticmethod
    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = int((a & b).sum())
        union = int((a | b).sum())
        return inter / union if union else 0.0

    def generate(self, image: np.ndarray) -> list[dict]:
        H, W = image.shape[:2]
        min_area = int(H * W * self._min_area_frac)

        # Optionally downscale for SAM3 to cut peak VRAM on large images.
        if self._sam_input_size is not None and max(H, W) > self._sam_input_size:
            scale = self._sam_input_size / max(H, W)
            sam_h, sam_w = int(H * scale), int(W * scale)
            sam_image = np.array(
                Image.fromarray(image).resize((sam_w, sam_h), Image.BILINEAR)
            )
            log.info("  SAM input resized %dx%d → %dx%d", W, H, sam_w, sam_h)
        else:
            sam_image = image
            sam_h, sam_w = H, W

        raw_results = self._seg.segment_with_text(
            sam_image,
            self._text_prompt,
            score_threshold=self._score_threshold,
            mask_threshold=self._mask_threshold,
        )

        # Upsample masks back to the original image resolution.
        if sam_image is not image:
            upsampled = []
            for r in raw_results:
                m_pil = Image.fromarray(r["mask"].astype(np.uint8) * 255).resize(
                    (W, H), Image.NEAREST
                )
                upsampled.append({"mask": np.array(m_pil) > 127, "score": r["score"]})
            raw_results = upsampled

        # Filter by minimum area.
        raw: list[tuple[np.ndarray, float]] = [
            (r["mask"], r["score"])
            for r in raw_results
            if r["mask"].sum() >= min_area
        ]

        if not raw:
            log.warning("No masks above min_area_frac=%.4f for prompt %r", self._min_area_frac, self._text_prompt)
            return []

        # Greedy NMS: keep masks in score order; drop those overlapping a kept mask.
        raw.sort(key=lambda t: t[1], reverse=True)
        kept: list[tuple[np.ndarray, float]] = []
        for mask, score in raw:
            if all(self._mask_iou(mask, k[0]) < self._nms_iou_threshold for k in kept):
                kept.append((mask, score))

        log.info(
            "HFSAMTextGenerator [%r]: %d raw → %d after NMS (iou_thresh=%.2f)",
            self._text_prompt,
            len(raw),
            len(kept),
            self._nms_iou_threshold,
        )
        return [
            {"segmentation": mask, "bbox": _bbox_from_mask(mask), "predicted_iou": score}
            for mask, score in kept
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Sample:
    """One image + GT mask pair from the dataset."""

    stem: str
    image: np.ndarray  # (H, W, 3) uint8 RGB  — original resolution
    gt_mask: np.ndarray  # (H, W) bool         — resized to match image


@dataclass
class CandidateRecord:
    """A single SAM proposal with its DINO scores and IoU vs. GT."""

    image_id: str
    bbox_xywh: list[int]
    seg: np.ndarray  # (H, W) bool — original image coordinates
    prefilt_score: float
    score_m1: float | None
    score_m2: float | None
    score_m3: float | None
    iou_gt: float  # IoU with ground-truth mask (0.0 if no GT)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_dataset(data_dir: Path) -> list[Sample]:
    """Load all image+mask pairs from *data_dir*.

    Only images that have a corresponding ``<stem>_mask_good.npz`` are kept.
    The GT mask (stored at 768×768) is rescaled to the image's native
    resolution using nearest-neighbour interpolation.
    """
    samples: list[Sample] = []
    for img_path in sorted(data_dir.iterdir()):
        if img_path.suffix.lower() not in _IMAGE_EXTS:
            continue
        mask_path = data_dir / f"{img_path.stem}_mask_good.npz"
        if not mask_path.exists():
            log.warning("No GT mask for %s — skipping", img_path.name)
            continue

        image = np.asarray(Image.open(img_path).convert("RGB"))
        H, W = image.shape[:2]

        raw = np.load(mask_path)["segmaps"]  # (768, 768, 1) bool
        gt_pil = Image.fromarray(raw[:, :, 0].astype(np.uint8) * 255)
        gt_pil = gt_pil.resize((W, H), Image.NEAREST)
        gt_mask = np.asarray(gt_pil) > 127

        samples.append(Sample(stem=img_path.stem, image=image, gt_mask=gt_mask))

    log.info("Loaded %d samples from %s", len(samples), data_dir)
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# IoU helper
# ─────────────────────────────────────────────────────────────────────────────


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-Union between two boolean masks of the same shape."""
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection) / (float(union) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline evaluation (mirrors SAMDinoEvaluator but keeps segmentations)
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_image(
    image_id: str,
    image: np.ndarray,
    gt_mask: np.ndarray,
    exemplar: ExemplarFeatures,
    encoder: DinoEncoder,
    mask_generator: MaskGenerator,
    prefilt_threshold: float,
) -> list[CandidateRecord]:
    """Run SAM+DINO on *image* and return one CandidateRecord per surviving mask."""
    t0 = time.perf_counter()

    sam_masks: list[dict[str, Any]] = mask_generator.generate(image)
    log.info("  %s: SAM → %d masks", image_id, len(sam_masks))

    # Full-image DINO pass for pre-filtering.
    full_patches = encoder([image]).patches[0].cpu().numpy()

    # Pre-filter all masks before crop encoding so we can free the full tensor.
    prefilt_pairs: list[tuple[np.ndarray, list[int], float]] = []  # (seg, bbox, score)
    for m in sam_masks:
        seg: np.ndarray = m["segmentation"]
        proc_mask = _preprocess_mask(seg, encoder.img_size)
        proto = _masked_prototype(full_patches, proc_mask, encoder.patch_size)
        if proto is None:
            continue
        prefilt_score = _cosine_sim(proto, exemplar.prototype)
        if prefilt_score < prefilt_threshold:
            continue
        bbox_raw = m.get("bbox")
        bbox = [int(v) for v in bbox_raw] if bbox_raw is not None else _bbox_from_mask(seg)
        prefilt_pairs.append((seg, bbox, prefilt_score))

    # Release full-image patches — no longer needed.
    del full_patches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    candidates: list[CandidateRecord] = []
    passed = len(prefilt_pairs)

    for seg, bbox, prefilt_score in prefilt_pairs:
        # Crop-level DINO scoring.
        crop, crop_mask_arr = _crop_to_bbox(image, seg, bbox)
        if crop.size == 0:
            continue

        crop_patches = encoder([crop]).patches[0].cpu().numpy()
        H_g, W_g, D = crop_patches.shape
        proc_crop_mask = _preprocess_mask(crop_mask_arr, encoder.img_size)
        grid = _mask_to_patch_grid(proc_crop_mask, H_g, W_g, encoder.patch_size)
        cand_patches = crop_patches[grid]
        if len(cand_patches) == 0:
            cand_patches = crop_patches.reshape(-1, D)

        s1 = score_method1(cand_patches, exemplar.prototype)
        s2 = score_method2(cand_patches, exemplar.patches)
        s3 = (
            score_method3(cand_patches, exemplar.cluster_prototypes)
            if exemplar.cluster_prototypes is not None
            else None
        )

        iou = mask_iou(seg, gt_mask)
        candidates.append(
            CandidateRecord(
                image_id=image_id,
                bbox_xywh=bbox,
                seg=seg,
                prefilt_score=prefilt_score,
                score_m1=s1,
                score_m2=s2,
                score_m3=s3,
                iou_gt=iou,
            )
        )

    log.info(
        "  %s: pre-filter passed %d/%d  |  elapsed %.2fs",
        image_id,
        passed,
        len(sam_masks),
        time.perf_counter() - t0,
    )
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def compute_pr_curve(
    scores: np.ndarray,
    iou_gt: np.ndarray,
    iou_threshold: float,
    n_images: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Candidate-level Precision-Recall curve.

    Each candidate is TP if its IoU with GT ≥ *iou_threshold*; FP otherwise.
    Recall is normalised by the total number of ground-truth instances
    (*n_images* — one per inference image).

    Returns
    -------
    precisions, recalls, thresholds (same length; monotone in threshold)
    """
    order = np.argsort(scores)[::-1]
    scores_sorted = scores[order]
    labels = (iou_gt[order] >= iou_threshold).astype(int)

    tp_cum = np.cumsum(labels)
    fp_cum = np.cumsum(1 - labels)

    precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
    recalls = tp_cum / (n_images + 1e-8)
    return precisions, recalls, scores_sorted


def compute_roc_curve(
    scores: np.ndarray,
    iou_gt: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Candidate-level ROC: TP rate vs. FP rate.

    * TP candidates : IoU ≥ *iou_threshold*
    * FP candidates : IoU <  *iou_threshold*

    Returns
    -------
    tprs, fprs, thresholds
    """
    order = np.argsort(scores)[::-1]
    scores_sorted = scores[order]
    labels = (iou_gt[order] >= iou_threshold).astype(int)

    total_pos = labels.sum()
    total_neg = len(labels) - total_pos
    if total_pos == 0 or total_neg == 0:
        log.warning("ROC curve: degenerate split (P=%d, N=%d)", total_pos, total_neg)

    tp_cum = np.cumsum(labels)
    fp_cum = np.cumsum(1 - labels)
    tprs = tp_cum / (total_pos + 1e-8)
    fprs = fp_cum / (total_neg + 1e-8)
    return tprs, fprs, scores_sorted


def average_precision(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """Area under PR curve via trapezoidal integration (sklearn-style)."""
    # Prepend sentinel so the curve starts at recall=0, precision=1
    r = np.concatenate([[0.0], recalls, [recalls[-1]]])
    p = np.concatenate([[1.0], precisions, [0.0]])
    return float(np.trapezoid(p, r))


def instance_stats(
    all_records: list[CandidateRecord],
    image_ids: list[str],
    score_attr: str,
    score_threshold: float,
    iou_threshold: float,
) -> dict[str, float]:
    """Instance-level TP/FP/FN stats.

    For each image, the best-scoring candidate (above *score_threshold*) is
    considered the detection.  If its IoU ≥ *iou_threshold* → TP; if the
    image has no passing candidate → FN.  FP when the image has passing
    candidates but none match GT.
    """
    per_image: dict[str, list[CandidateRecord]] = {i: [] for i in image_ids}
    for rec in all_records:
        s = getattr(rec, score_attr)
        if s is not None and s >= score_threshold:
            per_image[rec.image_id].append(rec)

    tp = fn = fp = 0
    for img_id in image_ids:
        cands = per_image[img_id]
        if not cands:
            fn += 1
        else:
            best_iou = max(r.iou_gt for r in cands)
            if best_iou >= iou_threshold:
                tp += 1
            else:
                fp += 1

    tpr = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    return {"TP": tp, "FP": fp, "FN": fn, "TPR": tpr, "Precision": precision}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

_METHODS = [
    ("score_m1", "M1 Global", "tab:blue"),
    ("score_m2", "M2 PatchCross", "tab:orange"),
    ("score_m3", "M3 Cluster", "tab:green"),
]


def plot_pr_curves(
    all_records: list[CandidateRecord],
    n_images: int,
    iou_threshold: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for attr, label, color in _METHODS:
        scores = np.array([getattr(r, attr) for r in all_records if getattr(r, attr) is not None])
        ious = np.array([r.iou_gt for r in all_records if getattr(r, attr) is not None])
        if len(scores) == 0:
            continue
        prec, rec, _ = compute_pr_curve(scores, ious, iou_threshold, n_images)
        ap = average_precision(prec, rec)
        ax.plot(rec, prec, label=f"{label}  AP={ap:.3f}", color=color)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall  (IoU ≥ {iou_threshold})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved PR curve → %s", out_path)


def plot_roc_curves(
    all_records: list[CandidateRecord],
    iou_threshold: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random")
    for attr, label, color in _METHODS:
        scores = np.array([getattr(r, attr) for r in all_records if getattr(r, attr) is not None])
        ious = np.array([r.iou_gt for r in all_records if getattr(r, attr) is not None])
        if len(scores) == 0:
            continue
        tprs, fprs, _ = compute_roc_curve(scores, ious, iou_threshold)
        auc = float(np.trapezoid(tprs, fprs))
        ax.plot(fprs, tprs, label=f"{label}  AUC={auc:.3f}", color=color)

    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curve  (IoU ≥ {iou_threshold})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved ROC curve → %s", out_path)


def plot_iou_histogram(
    all_records: list[CandidateRecord],
    iou_threshold: float,
    out_path: Path,
) -> None:
    ious = np.array([r.iou_gt for r in all_records])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ious, bins=50, color="steelblue", edgecolor="white")
    ax.axvline(iou_threshold, color="red", linestyle="--", label=f"IoU threshold {iou_threshold}")
    ax.set_xlabel("IoU with GT mask")
    ax.set_ylabel("Number of candidates")
    ax.set_title("IoU distribution (all pre-filter-passing candidates)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved IoU histogram → %s", out_path)


def plot_score_distributions(
    all_records: list[CandidateRecord],
    iou_threshold: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, (attr, label, color) in zip(axes, _METHODS):
        pos_scores = [getattr(r, attr) for r in all_records
                      if getattr(r, attr) is not None and r.iou_gt >= iou_threshold]
        neg_scores = [getattr(r, attr) for r in all_records
                      if getattr(r, attr) is not None and r.iou_gt < iou_threshold]
        bins = np.linspace(0, 1, 30)
        ax.hist(neg_scores, bins=bins, alpha=0.6, color="salmon", label="IoU < thr (FP)")
        ax.hist(pos_scores, bins=bins, alpha=0.7, color="steelblue", label="IoU ≥ thr (TP)")
        ax.set_title(label)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(f"Score distributions  (IoU threshold={iou_threshold})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved score distributions → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SAM+DINO on custom_slim with GT mask metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "custom_slim",
        help="Directory containing images and _mask_good.npz files.",
    )
    parser.add_argument(
        "--exemplar-stem",
        type=str,
        default=None,
        help="Stem of the exemplar image (e.g. frame_spectacles1_0100). "
        "If omitted the first sample (alphabetically) is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/custom_slim_eval"),
        help="Directory for plots and JSON summary.",
    )
    # Encoder
    parser.add_argument("--version", default="v3", choices=["v2", "v3"])
    parser.add_argument("--size", default="base", choices=["small", "base", "large", "giant"])
    parser.add_argument("--img-size", type=int, default=448)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--no-amp", dest="amp", action="store_false",
        help="Disable bfloat16 autocast in DinoEncoder (default: enabled).",
    )
    parser.set_defaults(amp=True)
    # Pipeline
    parser.add_argument("--threshold", type=float, default=0.3, help="Pre-filter threshold.")
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for TP/FP assignment.")
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="Score threshold used for instance-level stats.")
    # SAM3 (HuggingFace transformers backend — no local checkpoint needed)
    parser.add_argument(
        "--sam-model-id",
        default="facebook/sam3",
        help="HuggingFace model ID for SAM3 (or SAM2 as fallback).",
    )
    parser.add_argument(
        "--text-prompt",
        default="object",
        help="Free-form text description of the object to find, e.g. 'spectacle frame'.",
    )
    parser.add_argument(
        "--sam-score-threshold",
        type=float,
        default=0.5,
        help="Minimum SAM3 instance confidence score (passed to post_process_instance_segmentation).",
    )
    parser.add_argument(
        "--sam-mask-threshold",
        type=float,
        default=0.5,
        help="Logit threshold for binarising SAM3 soft masks.",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.70,
        help="Mask-IoU threshold for duplicate removal in the text mask generator.",
    )
    parser.add_argument(
        "--min-area-frac",
        type=float,
        default=0.001,
        help="Minimum mask area as a fraction of image area; smaller masks are dropped.",
    )
    parser.add_argument(
        "--sam-input-size",
        type=int,
        default=None,
        metavar="PX",
        help=(
            "Downscale images to this max side length before SAM3 (e.g. 1024). "
            "Masks are upsampled back to original resolution for IoU. "
            "Most impactful memory knob for high-res inputs (3840×2160 → 1024 saves ~10× VRAM). "
            "Pair with --img-size 224 and --size small to minimise total footprint."
        ),
    )

    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── load dataset ─────────────────────────────────────────────────────────
    samples = load_dataset(args.data_dir)
    if not samples:
        log.error("No valid samples found in %s", args.data_dir)
        return

    if args.exemplar_stem is not None:
        try:
            ex_idx = next(i for i, s in enumerate(samples) if s.stem == args.exemplar_stem)
        except StopIteration:
            log.error("Exemplar stem '%s' not found in dataset.", args.exemplar_stem)
            return
    else:
        ex_idx = 0

    exemplar_sample = samples[ex_idx]
    inference_samples = [s for i, s in enumerate(samples) if i != ex_idx]
    log.info(
        "Exemplar: %s  |  Inference set: %d images",
        exemplar_sample.stem,
        len(inference_samples),
    )

    # ── build encoder & mask generator ───────────────────────────────────────
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = DinoEncoder(
        version=args.version,
        size=args.size,
        img_size=args.img_size,
        layers=1,
        device=device,
        amp=args.amp,
        weights_dir=Path(""),
    )
    log.info("Encoder: DINO%s-%s  img_size=%d  device=%s",
             args.version, args.size, args.img_size, device)

    segmenter = SAMSegmenter(model_id=args.sam_model_id, device=device)
    mask_generator = HFSAMTextGenerator(
        segmenter,
        text_prompt=args.text_prompt,
        score_threshold=args.sam_score_threshold,
        mask_threshold=args.sam_mask_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        min_area_frac=args.min_area_frac,
        sam_input_size=args.sam_input_size,
    )
    log.info(
        "HFSAMTextGenerator: model=%s  prompt=%r  score_thr=%.2f  mask_thr=%.2f  "
        "nms_iou=%.2f  sam_input_size=%s",
        args.sam_model_id,
        args.text_prompt,
        args.sam_score_threshold,
        args.sam_mask_threshold,
        args.nms_iou_threshold,
        args.sam_input_size,
    )

    n_clusters: int | None = args.n_clusters if args.n_clusters > 1 else None

    # ── process exemplar ─────────────────────────────────────────────────────
    log.info("Processing exemplar …")
    exemplar_feats = process_exemplar(
        encoder,
        exemplar_sample.image,
        exemplar_sample.gt_mask,
        n_clusters=n_clusters,
    )

    # ── evaluate inference set ────────────────────────────────────────────────
    all_records: list[CandidateRecord] = []
    image_ids: list[str] = []

    t_total = time.perf_counter()
    for sample in inference_samples:
        image_ids.append(sample.stem)
        records = evaluate_image(
            image_id=sample.stem,
            image=sample.image,
            gt_mask=sample.gt_mask,
            exemplar=exemplar_feats,
            encoder=encoder,
            mask_generator=mask_generator,
            prefilt_threshold=args.threshold,
        )
        all_records.extend(records)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log.info(
        "Evaluation done in %.1fs  |  %d candidates across %d images",
        time.perf_counter() - t_total,
        len(all_records),
        len(inference_samples),
    )

    if not all_records:
        log.error("No candidates survived pre-filtering — cannot compute metrics.")
        return

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_pr_curves(
        all_records, len(inference_samples), args.iou_threshold,
        args.output_dir / "pr_curves.png",
    )
    plot_roc_curves(
        all_records, args.iou_threshold,
        args.output_dir / "roc_curves.png",
    )
    plot_iou_histogram(
        all_records, args.iou_threshold,
        args.output_dir / "iou_histogram.png",
    )
    plot_score_distributions(
        all_records, args.iou_threshold,
        args.output_dir / "score_distributions.png",
    )

    # ── JSON summary ──────────────────────────────────────────────────────────
    summary: dict[str, Any] = {
        "exemplar": exemplar_sample.stem,
        "n_inference_images": len(inference_samples),
        "n_candidates_total": len(all_records),
        "iou_threshold": args.iou_threshold,
        "score_threshold": args.score_threshold,
        "iou_stats": {
            "mean": float(np.mean([r.iou_gt for r in all_records])),
            "median": float(np.median([r.iou_gt for r in all_records])),
            "max": float(np.max([r.iou_gt for r in all_records])),
            "pct_above_threshold": float(
                np.mean([r.iou_gt >= args.iou_threshold for r in all_records])
            ),
        },
        "methods": {},
    }

    for attr, label, _ in _METHODS:
        scores = np.array([getattr(r, attr) for r in all_records if getattr(r, attr) is not None])
        ious = np.array([r.iou_gt for r in all_records if getattr(r, attr) is not None])
        if len(scores) == 0:
            continue
        prec, rec, _ = compute_pr_curve(scores, ious, args.iou_threshold, len(inference_samples))
        tprs, fprs, _ = compute_roc_curve(scores, ious, args.iou_threshold)
        ap = average_precision(prec, rec)
        auc = float(np.trapezoid(tprs, fprs))
        inst = instance_stats(
            all_records, image_ids, attr, args.score_threshold, args.iou_threshold
        )
        summary["methods"][label] = {
            "AP": round(ap, 4),
            "AUC_ROC": round(auc, 4),
            "instance_stats_at_score_threshold": inst,
        }

    out_json = args.output_dir / "metrics_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log.info("Metrics summary → %s", out_json)

    # Print table to stdout.
    print("\n── Metrics Summary ──────────────────────────────────────────────")
    print(f"  Exemplar          : {exemplar_sample.stem}")
    print(f"  Inference images  : {len(inference_samples)}")
    print(f"  Total candidates  : {len(all_records)}")
    iou_vals = [r.iou_gt for r in all_records]
    print(f"  IoU  mean/median  : {np.mean(iou_vals):.3f} / {np.median(iou_vals):.3f}")
    print(f"  IoU > {args.iou_threshold:.2f} fraction  : "
          f"{np.mean([v >= args.iou_threshold for v in iou_vals]):.3f}")
    print()
    print(f"  {'Method':<18}  {'AP':>6}  {'AUC-ROC':>8}  {'TP':>4}  {'FP':>4}  {'FN':>4}  "
          f"{'TPR':>6}  {'Prec':>6}")
    print("  " + "-" * 68)
    for attr, label, _ in _METHODS:
        if label not in summary["methods"]:
            continue
        m = summary["methods"][label]
        ist = m["instance_stats_at_score_threshold"]
        print(f"  {label:<18}  {m['AP']:>6.3f}  {m['AUC_ROC']:>8.3f}  "
              f"{ist['TP']:>4}  {ist['FP']:>4}  {ist['FN']:>4}  "
              f"{ist['TPR']:>6.3f}  {ist['Precision']:>6.3f}")
    print("─────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
