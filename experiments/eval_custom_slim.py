# %% [markdown]
# # SAM3 + DINO Custom Slim Evaluation
#
# End-to-end walkthrough of the `eval_custom_slim` pipeline:
# SAM3 (text-prompted) proposes candidate masks; DINO patch features rank them
# against an exemplar; ground-truth IoU measures quality.
#
# Steps:
#   0. Load `custom_slim` images and `_mask_good.npz` GT masks
#   1. Extract DINO features from the exemplar image + GT mask
#   2. SAM3 text-prompt generates candidate instance masks per image
#   3. DINO pre-filter (cosine similarity) discards low-confidence candidates
#   4. Three scoring methods (M1 global / M2 patch-cross / M3 cluster) rank survivors
#   5. Precision-Recall, ROC, IoU histogram, and score-distribution plots

# %% Logging — must be before any torch / transformers import
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger(__name__)

# %% Imports
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from PIL import Image

_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".env")
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import torch
from eval_sam_dino import (
    ExemplarFeatures,
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

from dinoisawesome import DinoEncoder

log.info("Imports OK — torch %s", torch.__version__)

# %% Parameters
DATA_DIR = _REPO_ROOT / "data" / "custom_slim"
EXEMPLAR_STEM: str | None = None  # None → first sample alphabetically
OUTPUT_DIR = _REPO_ROOT / "results" / "custom_slim_nb"

DINO_VERSION = "v3"
DINO_SIZE = "base"
IMG_SIZE = 448
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = True

PREFILT_THRESHOLD = 0.3
N_CLUSTERS = 4
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5

SAM_MODEL_ID = "facebook/sam3"
TEXT_PROMPT = "object"
SAM_SCORE_THRESHOLD = 0.5
SAM_MASK_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.70
MIN_AREA_FRAC = 0.001
SAM_INPUT_SIZE: int | None = None

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log.info("Output → %s", OUTPUT_DIR.resolve())
log.info("Device  : %s", DEVICE)

# %% SAMSegmenter definition


def _to_pil_rgb(image: Any) -> Image.Image:
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


class SAMSegmenter:
    """HuggingFace SAM3 (SAM2 fallback) wrapper. Model is downloaded on first use."""

    def __init__(self, model_id: str = SAM_MODEL_ID, device: str | None = None) -> None:
        self._model_id = model_id
        self._device = device or DEVICE
        self._dtype = torch.bfloat16 if self._device.startswith("cuda") else torch.float32
        self._model = None
        self._processor = None

    def _to_device(self, inputs: dict) -> dict:
        return {
            k: (
                v.to(device=self._device, dtype=self._dtype)
                if torch.is_floating_point(v)
                else v.to(self._device)
            )
            for k, v in inputs.items()
        }

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
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

            log.warning("Sam3Model not found — falling back to Sam2Model.")
            self._processor = Sam2Processor.from_pretrained(self._model_id, token=token)
            self._model = Sam2Model.from_pretrained(
                self._model_id, token=token, torch_dtype=self._dtype
            ).to(self._device)
            log.info("SAM2 ready (fallback).")
        self._model.eval()

    def segment_with_text(
        self,
        image: Any,
        text: str,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> list[dict]:
        """Return [{mask: bool[H,W], score: float}, …] sorted by score desc."""
        self._ensure_loaded()
        pil = _to_pil_rgb(image)
        raw = self._processor(images=pil, text=text, return_tensors="pt")
        inputs = self._to_device(raw)
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
            {
                "mask": results["masks"][i].cpu().numpy().astype(bool),
                "score": float(results["scores"][i]),
            }
            for i in range(len(results["masks"]))
        ]
        return sorted(entries, key=lambda r: r["score"], reverse=True)


log.info("SAMSegmenter defined.")

# %% HFSAMTextGenerator definition


class HFSAMTextGenerator:
    """Wraps SAMSegmenter text-prompt mode with area filtering and greedy mask-IoU NMS.

    Produces [{segmentation: bool[H,W], bbox: [x,y,w,h], predicted_iou: float}, …].
    """

    def __init__(
        self,
        segmenter: SAMSegmenter,
        text_prompt: str = TEXT_PROMPT,
        score_threshold: float = SAM_SCORE_THRESHOLD,
        mask_threshold: float = SAM_MASK_THRESHOLD,
        nms_iou_threshold: float = NMS_IOU_THRESHOLD,
        min_area_frac: float = MIN_AREA_FRAC,
        sam_input_size: int | None = SAM_INPUT_SIZE,
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

        if self._sam_input_size is not None and max(H, W) > self._sam_input_size:
            scale = self._sam_input_size / max(H, W)
            sam_h, sam_w = int(H * scale), int(W * scale)
            sam_image = np.array(Image.fromarray(image).resize((sam_w, sam_h), Image.BILINEAR))
            log.info("  SAM input resized %dx%d → %dx%d", W, H, sam_w, sam_h)
        else:
            sam_image = image

        raw_results = self._seg.segment_with_text(
            sam_image,
            self._text_prompt,
            score_threshold=self._score_threshold,
            mask_threshold=self._mask_threshold,
        )

        if sam_image is not image:
            upsampled = []
            for r in raw_results:
                m_pil = Image.fromarray(r["mask"].astype(np.uint8) * 255).resize(
                    (W, H), Image.NEAREST
                )
                upsampled.append({"mask": np.array(m_pil) > 127, "score": r["score"]})
            raw_results = upsampled

        raw = [(r["mask"], r["score"]) for r in raw_results if r["mask"].sum() >= min_area]
        if not raw:
            log.warning(
                "No masks above min_area_frac=%.4f for prompt %r",
                self._min_area_frac,
                self._text_prompt,
            )
            return []

        raw.sort(key=lambda t: t[1], reverse=True)
        kept = []
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


log.info("HFSAMTextGenerator defined.")

# %% Step 0 — Dataset


@dataclass
class Sample:
    stem: str
    image: np.ndarray  # (H, W, 3) uint8 RGB
    gt_mask: np.ndarray  # (H, W) bool


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_dataset(data_dir: Path) -> list[Sample]:
    """Load all image + `_mask_good.npz` pairs from *data_dir*."""
    samples = []
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


samples = load_dataset(DATA_DIR)

if EXEMPLAR_STEM is not None:
    ex_idx = next(i for i, s in enumerate(samples) if s.stem == EXEMPLAR_STEM)
else:
    ex_idx = 0
exemplar_sample = samples[ex_idx]
inference_samples = [s for i, s in enumerate(samples) if i != ex_idx]
log.info("Exemplar: %s  |  Inference set: %d images", exemplar_sample.stem, len(inference_samples))

# %% Visualise dataset sample
n_show = min(6, len(samples))
fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
if n_show == 1:
    axes = np.array(axes).reshape(2, 1)

for col, s in enumerate(samples[:n_show]):
    axes[0, col].imshow(s.image)
    axes[0, col].set_title(s.stem, fontsize=7)
    axes[0, col].axis("off")
    overlay = s.image.copy()
    overlay[s.gt_mask] = (
        (overlay[s.gt_mask] * 0.4 + np.array([0, 200, 80]) * 0.6).clip(0, 255).astype(np.uint8)
    )
    axes[1, col].imshow(overlay)
    axes[1, col].set_title("GT mask" if col == 0 else "", fontsize=7)
    axes[1, col].axis("off")

fig.suptitle(f"Dataset — {len(samples)} samples  (exemplar: {exemplar_sample.stem})", fontsize=9)
fig.tight_layout()
plt.show()

# %% Step 1 — DINO Encoder + Exemplar Features
encoder = DinoEncoder(
    version=DINO_VERSION,
    size=DINO_SIZE,
    img_size=IMG_SIZE,
    layers=1,
    device=DEVICE,
    amp=AMP,
    weights_dir=os.environ.get("DINO_WEIGHTS_DIR"),
)
log.info("Encoder: DINO%s-%s  img_size=%d  device=%s", DINO_VERSION, DINO_SIZE, IMG_SIZE, DEVICE)

# %% Compute exemplar features
n_clusters_arg = N_CLUSTERS if N_CLUSTERS > 1 else None

log.info("Processing exemplar …")
exemplar_feats: ExemplarFeatures = process_exemplar(
    encoder,
    exemplar_sample.image,
    exemplar_sample.gt_mask,
    n_clusters=n_clusters_arg,
)
log.info("Exemplar  prototype shape : %s", exemplar_feats.prototype.shape)
log.info("Exemplar  patches shape   : %s", exemplar_feats.patches.shape)
if exemplar_feats.cluster_prototypes is not None:
    log.info("Cluster prototypes shape  : %s", exemplar_feats.cluster_prototypes.shape)

# %% Visualise exemplar
ex_patches = encoder([exemplar_sample.image]).patches[0].cpu().numpy()  # (Hg, Wg, D)
Hg, Wg, D = ex_patches.shape
flat = ex_patches.reshape(-1, D)
proto = exemplar_feats.prototype
norms = np.linalg.norm(flat, axis=1, keepdims=True) * np.linalg.norm(proto) + 1e-8
sim_flat = flat @ proto / norms
sim_map = sim_flat.reshape(Hg, Wg)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(exemplar_sample.image)
axes[0].set_title("Exemplar image")
axes[0].axis("off")

overlay = exemplar_sample.image.copy()
overlay[exemplar_sample.gt_mask] = (
    (overlay[exemplar_sample.gt_mask] * 0.4 + np.array([0, 200, 80]) * 0.6)
    .clip(0, 255)
    .astype(np.uint8)
)
axes[1].imshow(overlay)
axes[1].set_title("GT mask (green)")
axes[1].axis("off")

im = axes[2].imshow(sim_map, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
plt.colorbar(im, ax=axes[2], fraction=0.046)
axes[2].set_title("Patch cosine sim vs. exemplar prototype")
axes[2].axis("off")

fig.suptitle(f"Exemplar: {exemplar_sample.stem}", fontsize=10)
fig.tight_layout()
plt.show()

# %% Step 2 — SAM3 Mask Generator
segmenter = SAMSegmenter(model_id=SAM_MODEL_ID, device=DEVICE)
mask_generator = HFSAMTextGenerator(
    segmenter,
    text_prompt=TEXT_PROMPT,
    score_threshold=SAM_SCORE_THRESHOLD,
    mask_threshold=SAM_MASK_THRESHOLD,
    nms_iou_threshold=NMS_IOU_THRESHOLD,
    min_area_frac=MIN_AREA_FRAC,
    sam_input_size=SAM_INPUT_SIZE,
)
log.info(
    "HFSAMTextGenerator: model=%s  prompt=%r  score_thr=%.2f  nms_iou=%.2f",
    SAM_MODEL_ID,
    TEXT_PROMPT,
    SAM_SCORE_THRESHOLD,
    NMS_IOU_THRESHOLD,
)

# %% Demo: SAM3 on first inference image
demo_sample = inference_samples[0]
log.info("Demo SAM3 on: %s", demo_sample.stem)

t0 = time.perf_counter()
demo_masks = mask_generator.generate(demo_sample.image)
log.info("SAM3 → %d candidates (%.2fs)", len(demo_masks), time.perf_counter() - t0)

n_show_masks = min(8, len(demo_masks))
cols = min(4, n_show_masks)
rows = max(1, (n_show_masks + cols - 1) // cols)

fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
axes_flat = np.array(axes).flatten()
cmap_demo = plt.get_cmap("tab10")

for idx, m in enumerate(demo_masks[:n_show_masks]):
    ax = axes_flat[idx]
    ovl = demo_sample.image.copy()
    colour = (np.array(cmap_demo(idx % 10)[:3]) * 200).astype(np.uint8)
    ovl[m["segmentation"]] = (
        (ovl[m["segmentation"]] * 0.4 + colour * 0.6).clip(0, 255).astype(np.uint8)
    )
    ax.imshow(ovl)
    ax.set_title(f"candidate {idx}  score={m['predicted_iou']:.2f}", fontsize=8)
    ax.axis("off")

for ax in axes_flat[n_show_masks:]:
    ax.axis("off")

fig.suptitle(f"SAM3 candidates — {demo_sample.stem}  (prompt: {TEXT_PROMPT!r})", fontsize=9)
fig.tight_layout()
plt.show()

# %% Step 3 — DINO Scoring helpers


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.logical_and(a, b).sum()) / (float(np.logical_or(a, b).sum()) + 1e-8)


@dataclass
class CandidateRecord:
    image_id: str
    bbox_xywh: list
    seg: np.ndarray
    prefilt_score: float
    score_m1: "float | None"
    score_m2: "float | None"
    score_m3: "float | None"
    iou_gt: float


def evaluate_image(
    image_id: str,
    image: np.ndarray,
    gt_mask: np.ndarray,
    exemplar: ExemplarFeatures,
    encoder_: DinoEncoder,
    mask_gen: HFSAMTextGenerator,
    prefilt_thr: float,
) -> list[CandidateRecord]:
    t0 = time.perf_counter()
    sam_masks = mask_gen.generate(image)
    log.info("  %s: SAM → %d masks", image_id, len(sam_masks))

    full_patches = encoder_([image]).patches[0].cpu().numpy()
    prefilt_pairs = []
    for m in sam_masks:
        seg = m["segmentation"]
        proc_mask = _preprocess_mask(seg, encoder_.img_size)
        proto = _masked_prototype(full_patches, proc_mask, encoder_.patch_size)
        if proto is None:
            continue
        score = _cosine_sim(proto, exemplar.prototype)
        if score < prefilt_thr:
            continue
        bbox_raw = m.get("bbox")
        bbox = [int(v) for v in bbox_raw] if bbox_raw is not None else _bbox_from_mask(seg)
        prefilt_pairs.append((seg, bbox, score))

    del full_patches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    candidates = []
    for seg, bbox, prefilt_score in prefilt_pairs:
        crop, crop_mask_arr = _crop_to_bbox(image, seg, bbox)
        if crop.size == 0:
            continue
        crop_patches = encoder_([crop]).patches[0].cpu().numpy()
        H_g, W_g, D = crop_patches.shape
        proc_crop_mask = _preprocess_mask(crop_mask_arr, encoder_.img_size)
        grid = _mask_to_patch_grid(proc_crop_mask, H_g, W_g, encoder_.patch_size)
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
        len(prefilt_pairs),
        len(sam_masks),
        time.perf_counter() - t0,
    )
    return candidates


log.info("evaluate_image defined.")

# %% Demo scoring on first inference image
demo_records = evaluate_image(
    image_id=demo_sample.stem,
    image=demo_sample.image,
    gt_mask=demo_sample.gt_mask,
    exemplar=exemplar_feats,
    encoder_=encoder,
    mask_gen=mask_generator,
    prefilt_thr=PREFILT_THRESHOLD,
)
log.info("%d candidates passed pre-filter", len(demo_records))
for r in demo_records:
    log.info(
        "  bbox=%s  prefilt=%.3f  M1=%.3f  M2=%.3f  M3=%s  IoU=%.3f",
        r.bbox_xywh,
        r.prefilt_score,
        r.score_m1 or 0,
        r.score_m2 or 0,
        f"{r.score_m3:.3f}" if r.score_m3 is not None else "N/A",
        r.iou_gt,
    )

# %% Visualise scored candidates
if demo_records:
    _METHODS_DEMO = [
        ("score_m1", "M1 Global"),
        ("score_m2", "M2 PatchCross"),
        ("score_m3", "M3 Cluster"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    gt_overlay = demo_sample.image.copy()
    gt_overlay[demo_sample.gt_mask] = (
        (gt_overlay[demo_sample.gt_mask] * 0.4 + np.array([0, 200, 80]) * 0.6)
        .clip(0, 255)
        .astype(np.uint8)
    )
    axes[0].imshow(gt_overlay)
    axes[0].set_title("GT mask (green)")
    axes[0].axis("off")

    for ax, (attr, label) in zip(axes[1:], _METHODS_DEMO):
        valid = [r for r in demo_records if getattr(r, attr) is not None]
        if not valid:
            ax.axis("off")
            ax.set_title(f"{label}\n(no candidates)")
            continue
        best = max(valid, key=lambda r: getattr(r, attr))
        ovl = demo_sample.image.copy()
        colour = np.array([0, 200, 80]) if best.iou_gt >= IOU_THRESHOLD else np.array([220, 50, 50])
        ovl[best.seg] = (ovl[best.seg] * 0.35 + colour * 0.65).clip(0, 255).astype(np.uint8)
        ax.imshow(ovl)
        score_val = getattr(best, attr)
        ax.set_title(
            f"{label}\nscore={score_val:.3f}  IoU={best.iou_gt:.3f}"
            + ("  TP" if best.iou_gt >= IOU_THRESHOLD else "  FP"),
            fontsize=8,
        )
        ax.axis("off")

    fig.suptitle(f"Best candidate per method — {demo_sample.stem}", fontsize=9)
    fig.tight_layout()
    plt.show()
else:
    log.warning("No candidates to visualise for %s", demo_sample.stem)

# %% Step 4 — Full Evaluation Loop
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
        encoder_=encoder,
        mask_gen=mask_generator,
        prefilt_thr=PREFILT_THRESHOLD,
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

# %% Step 5 — Precision-Recall and ROC helpers


def compute_pr_curve(
    scores: np.ndarray, iou_gt: np.ndarray, iou_threshold: float, n_images: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(scores)[::-1]
    labels = (iou_gt[order] >= iou_threshold).astype(int)
    tp_cum = np.cumsum(labels)
    fp_cum = np.cumsum(1 - labels)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
    recalls = tp_cum / (n_images + 1e-8)
    return precisions, recalls, scores[order]


def compute_roc_curve(
    scores: np.ndarray, iou_gt: np.ndarray, iou_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(scores)[::-1]
    labels = (iou_gt[order] >= iou_threshold).astype(int)
    total_pos = labels.sum()
    total_neg = len(labels) - total_pos
    if total_pos == 0 or total_neg == 0:
        log.warning("ROC curve: degenerate split (P=%d, N=%d)", total_pos, total_neg)
    tp_cum = np.cumsum(labels)
    fp_cum = np.cumsum(1 - labels)
    tprs = tp_cum / (total_pos + 1e-8)
    fprs = fp_cum / (total_neg + 1e-8)
    return tprs, fprs, scores[order]


def average_precision(precisions: np.ndarray, recalls: np.ndarray) -> float:
    r = np.concatenate([[0.0], recalls, [recalls[-1]]])
    p = np.concatenate([[1.0], precisions, [0.0]])
    return float(np.trapezoid(p, r))


_METHODS = [
    ("score_m1", "M1 Global", "tab:blue"),
    ("score_m2", "M2 PatchCross", "tab:orange"),
    ("score_m3", "M3 Cluster", "tab:green"),
]
log.info("Metric helpers defined.")

# %% Precision-Recall curves
fig, ax = plt.subplots(figsize=(7, 5))
for attr, label, color in _METHODS:
    scores = np.array([getattr(r, attr) for r in all_records if getattr(r, attr) is not None])
    ious = np.array([r.iou_gt for r in all_records if getattr(r, attr) is not None])
    if len(scores) == 0:
        continue
    prec, rec, _ = compute_pr_curve(scores, ious, IOU_THRESHOLD, len(inference_samples))
    ap = average_precision(prec, rec)
    ax.plot(rec, prec, label=f"{label}  AP={ap:.3f}", color=color)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"Precision-Recall  (IoU ≥ {IOU_THRESHOLD})")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "pr_curves.png", dpi=150)
plt.show()

# %% ROC curves
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random")
for attr, label, color in _METHODS:
    scores = np.array([getattr(r, attr) for r in all_records if getattr(r, attr) is not None])
    ious = np.array([r.iou_gt for r in all_records if getattr(r, attr) is not None])
    if len(scores) == 0:
        continue
    tprs, fprs, _ = compute_roc_curve(scores, ious, IOU_THRESHOLD)
    auc = float(np.trapezoid(tprs, fprs))
    ax.plot(fprs, tprs, label=f"{label}  AUC={auc:.3f}", color=color)

ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_title(f"ROC Curve  (IoU ≥ {IOU_THRESHOLD})")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "roc_curves.png", dpi=150)
plt.show()

# %% Step 6 — IoU Distribution and Score Separability
ious_all = np.array([r.iou_gt for r in all_records])
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(ious_all, bins=50, color="steelblue", edgecolor="white")
ax.axvline(IOU_THRESHOLD, color="red", linestyle="--", label=f"IoU threshold {IOU_THRESHOLD}")
ax.set_xlabel("IoU with GT mask")
ax.set_ylabel("Number of candidates")
ax.set_title("IoU distribution — all pre-filter-passing candidates")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "iou_histogram.png", dpi=150)
plt.show()

log.info(
    "IoU  mean=%.3f  median=%.3f  max=%.3f  pct_above_thr=%.3f",
    float(np.mean(ious_all)),
    float(np.median(ious_all)),
    float(np.max(ious_all)),
    float(np.mean(ious_all >= IOU_THRESHOLD)),
)

# %% Score distributions — TP vs FP
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
for ax, (attr, label, color) in zip(axes, _METHODS):
    pos_scores = [
        getattr(r, attr)
        for r in all_records
        if getattr(r, attr) is not None and r.iou_gt >= IOU_THRESHOLD
    ]
    neg_scores = [
        getattr(r, attr)
        for r in all_records
        if getattr(r, attr) is not None and r.iou_gt < IOU_THRESHOLD
    ]
    bins = np.linspace(0, 1, 30)
    ax.hist(neg_scores, bins=bins, alpha=0.6, color="salmon", label=f"IoU < {IOU_THRESHOLD} (FP)")
    ax.hist(
        pos_scores, bins=bins, alpha=0.7, color="steelblue", label=f"IoU ≥ {IOU_THRESHOLD} (TP)"
    )
    ax.set_title(label)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

fig.suptitle(f"Score distributions  (IoU threshold={IOU_THRESHOLD})", fontsize=10)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "score_distributions.png", dpi=150)
plt.show()

# %% Summary table


def instance_stats(
    all_recs: list[CandidateRecord],
    img_ids: list[str],
    score_attr: str,
    score_thr: float,
    iou_thr: float,
) -> dict:
    per_image: dict[str, list] = {i: [] for i in img_ids}
    for rec in all_recs:
        s = getattr(rec, score_attr)
        if s is not None and s >= score_thr:
            per_image[rec.image_id].append(rec)
    tp = fn = fp = 0
    for img_id in img_ids:
        cands = per_image[img_id]
        if not cands:
            fn += 1
        else:
            if max(r.iou_gt for r in cands) >= iou_thr:
                tp += 1
            else:
                fp += 1
    tpr = tp / (tp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    return {"TP": tp, "FP": fp, "FN": fn, "TPR": tpr, "Precision": prec}


iou_vals = [r.iou_gt for r in all_records]
log.info("── Metrics Summary ──────────────────────────────────────────────")
log.info("  Exemplar          : %s", exemplar_sample.stem)
log.info("  Inference images  : %d", len(inference_samples))
log.info("  Total candidates  : %d", len(all_records))
log.info("  IoU  mean/median  : %.3f / %.3f", np.mean(iou_vals), np.median(iou_vals))
log.info(
    "  IoU > %.2f fraction  : %.3f",
    IOU_THRESHOLD,
    np.mean([v >= IOU_THRESHOLD for v in iou_vals]),
)
log.info(
    "  %-18s  %6s  %8s  %4s  %4s  %4s  %6s  %6s",
    "Method",
    "AP",
    "AUC-ROC",
    "TP",
    "FP",
    "FN",
    "TPR",
    "Prec",
)

for attr, label, _ in _METHODS:
    scores = np.array([getattr(r, attr) for r in all_records if getattr(r, attr) is not None])
    ious = np.array([r.iou_gt for r in all_records if getattr(r, attr) is not None])
    if len(scores) == 0:
        log.info("  %-18s  %6s  %8s  (no data)", label, "—", "—")
        continue
    prec, rec, _ = compute_pr_curve(scores, ious, IOU_THRESHOLD, len(inference_samples))
    tprs, fprs, _ = compute_roc_curve(scores, ious, IOU_THRESHOLD)
    ap = average_precision(prec, rec)
    auc = float(np.trapezoid(tprs, fprs))
    ist = instance_stats(all_records, image_ids, attr, SCORE_THRESHOLD, IOU_THRESHOLD)
    log.info(
        "  %-18s  %6.3f  %8.3f  %4d  %4d  %4d  %6.3f  %6.3f",
        label,
        ap,
        auc,
        ist["TP"],
        ist["FP"],
        ist["FN"],
        ist["TPR"],
        ist["Precision"],
    )

log.info("Plots saved to %s", OUTPUT_DIR.resolve())
