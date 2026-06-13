"""Evaluation script: SAM3 + DINOv3 few-shot object detection pipeline.

Given:
  - An exemplar image with a binary mask of the object of interest.
  - A set of inference images.

Pipeline:
  1. DINOv3 encodes the exemplar; a prototype is computed from the masked
     patch region.  Exemplar patches are also clustered (Method 3 prep).
  2. For each inference image:
     a. SAM3 generates all candidate masks.
     b. DINOv3 encodes the full image.
     c. Pre-filter: masked patch mean vs exemplar prototype; drop below threshold.
  3. For each surviving mask, DINOv3 re-encodes the tight bounding-box crop.
  4. Score each crop with three methods:
       M1 – Global similarity : mean of masked crop patches vs exemplar prototype.
       M2 – Patch cross-sim   : mean over exemplar patches of max sim to any
                                masked crop patch.
       M3 – Cluster prototype : per-crop-patch max sim across cluster prototypes,
                                then mean over crop patches.

Usage
-----
python scripts/eval_sam_dino.py \\
    --exemplar-image exemplar.jpg \\
    --exemplar-mask  mask.png \\
    --inference-dir  images/ \\
    --output         results.json \\
    --sam-checkpoint path/to/sam3.pt \\
    --sam-config     path/to/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image

# Logging must be configured before torch is imported (torch may register
# handlers at import time on some builds).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# isort: off
import torch  # noqa: E402
from dinoisawesome import DinoEncoder  # noqa: E402
# isort: on


# ---------------------------------------------------------------------------
# SAM3 interface protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MaskGenerator(Protocol):
    """Minimal interface required from SAM3 (or any compatible mask generator).

    The ``generate`` return value must be a list of dicts, each containing:
      ``'segmentation'``: ``np.ndarray`` bool, shape (H, W)
      ``'bbox'``        : ``[x, y, w, h]`` in XYWH pixel coordinates (optional;
                          derived from the mask when absent)
    Additional keys (``'predicted_iou'``, ``'stability_score'``, …) are ignored.
    """

    def generate(self, image: np.ndarray) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ExemplarFeatures:
    """Processed exemplar: prototype vectors for all three scoring methods."""

    patches: np.ndarray  # (N_masked, D) — un-normalised masked patch embeddings
    prototype: np.ndarray  # (D,)         — L2-normalised mean of masked patches
    cluster_prototypes: np.ndarray | None = None  # (K, D) — L2-normalised centroids
    cluster_labels: np.ndarray | None = None  # (N_masked,) — int cluster assignment


@dataclass
class CandidateMask:
    """A single SAM3 mask that survived pre-filtering, with per-method scores."""

    bbox_xywh: list[int]  # [x, y, w, h] in original image coordinates
    prefilt_score: float = 0.0
    score_global: float | None = None  # Method 1
    score_patch_cross: float | None = None  # Method 2
    score_cluster: float | None = None  # Method 3


@dataclass
class ImageResult:
    """Detection results for a single inference image."""

    image_id: str
    total_masks: int
    passed_prefilt: int
    candidates: list[CandidateMask] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------


def _preprocess_mask(mask: np.ndarray, img_size: int) -> np.ndarray:
    """Apply the encoder's Resize(img_size) + CenterCrop(img_size) to a mask.

    Mirrors ``torchvision.transforms.Resize`` (shorter-side rescale) followed by
    ``CenterCrop``, using nearest-neighbour interpolation to preserve binary values.
    """
    mask_pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    w, h = mask_pil.size
    scale = img_size / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    mask_pil = mask_pil.resize((new_w, new_h), Image.NEAREST)
    left = (new_w - img_size) // 2
    top = (new_h - img_size) // 2
    mask_pil = mask_pil.crop((left, top, left + img_size, top + img_size))
    return np.asarray(mask_pil) > 127


def _mask_to_patch_grid(mask: np.ndarray, grid_h: int, grid_w: int, patch_size: int) -> np.ndarray:
    """Downsample a boolean pixel mask to the patch grid via majority vote.

    ``mask`` must already be at the encoder's image resolution (img_size × img_size).
    """
    ph = mask.shape[0] // grid_h
    pw = mask.shape[1] // grid_w
    grid = np.zeros((grid_h, grid_w), dtype=bool)
    for r in range(grid_h):
        for c in range(grid_w):
            block = mask[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw]
            grid[r, c] = block.mean() >= 0.5
    return grid


def _crop_to_bbox(
    image: np.ndarray, mask: np.ndarray, bbox_xywh: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (image_crop, mask_crop) for a given XYWH bounding box."""
    x, y, w, h = bbox_xywh
    return image[y : y + h, x : x + w], mask[y : y + h, x : x + w]


def _bbox_from_mask(seg: np.ndarray) -> list[int]:
    """Compute [x, y, w, h] bounding box from a boolean mask."""
    rows = np.where(np.any(seg, axis=1))[0]
    cols = np.where(np.any(seg, axis=0))[0]
    rmin, rmax = int(rows[0]), int(rows[-1])
    cmin, cmax = int(cols[0]), int(cols[-1])
    return [cmin, rmin, cmax - cmin + 1, rmax - rmin + 1]


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (safe against zero-norm rows)."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norms + 1e-8)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _masked_prototype(
    patches_spatial: np.ndarray,  # (H_grid, W_grid, D)
    proc_mask: np.ndarray,  # (img_size, img_size) bool — at encoder resolution
    patch_size: int,
) -> np.ndarray | None:
    """L2-normalised mean of patch embeddings within the mask, or None if empty."""
    H_grid, W_grid, D = patches_spatial.shape
    grid = _mask_to_patch_grid(proc_mask, H_grid, W_grid, patch_size)
    selected = patches_spatial[grid]
    if len(selected) == 0:
        return None
    mean = selected.mean(axis=0)
    return mean / (np.linalg.norm(mean) + 1e-8)


# ---------------------------------------------------------------------------
# Scoring functions (one per method)
# ---------------------------------------------------------------------------


def score_method1(
    candidate_patches: np.ndarray,  # (N_c, D)
    prototype: np.ndarray,  # (D,) L2-normalised exemplar prototype
) -> float:
    """Method 1 — global similarity: normalised mean of candidate patches vs prototype."""
    mean = candidate_patches.mean(axis=0)
    return _cosine_sim(mean, prototype)


def score_method2(
    candidate_patches: np.ndarray,  # (N_c, D)
    exemplar_patches: np.ndarray,  # (N_e, D)
) -> float:
    """Method 2 — patch cross-similarity.

    For each exemplar patch, find its maximum cosine similarity to any candidate
    patch, then average those per-exemplar-patch maxima.
    """
    cand_n = _l2_normalize(candidate_patches)  # (N_c, D)
    ex_n = _l2_normalize(exemplar_patches)  # (N_e, D)
    sim = ex_n @ cand_n.T  # (N_e, N_c)
    return float(sim.max(axis=1).mean())


def score_method3(
    candidate_patches: np.ndarray,  # (N_c, D)
    cluster_prototypes: np.ndarray,  # (K, D) L2-normalised
) -> float:
    """Method 3 — cluster prototype similarity.

    For each candidate patch, find its maximum cosine similarity across all cluster
    prototypes, then average those per-candidate-patch maxima.
    """
    cand_n = _l2_normalize(candidate_patches)  # (N_c, D)
    sim = cand_n @ cluster_prototypes.T  # (N_c, K)
    return float(sim.max(axis=1).mean())


# ---------------------------------------------------------------------------
# Exemplar processing
# ---------------------------------------------------------------------------


def process_exemplar(
    encoder: DinoEncoder,
    image: np.ndarray,
    mask: np.ndarray,
    n_clusters: int | None = None,
) -> ExemplarFeatures:
    """Extract DINOv3 features for the exemplar and build cluster prototypes.

    The exemplar mask is preprocessed with the same Resize+CenterCrop applied to
    images so it aligns with the patch grid.  Agglomerative clustering operates on
    the cosine distance matrix of exemplar patches (``metric='precomputed'``).

    Args:
        encoder:    DinoEncoder instance (version="v3" recommended).
        image:      (H, W, 3) uint8 RGB exemplar image.
        mask:       (H, W) bool or uint8 — non-zero pixels belong to the object.
        n_clusters: Number of clusters for Method 3.  None or ≤1 disables clustering.

    Returns:
        ExemplarFeatures ready for all three scoring methods.
    """
    proc_mask = _preprocess_mask(mask, encoder.img_size)

    out = encoder([image])
    patches_spatial = out.patches[0].cpu().numpy()  # (H_grid, W_grid, D)
    H_grid, W_grid, D = patches_spatial.shape

    grid = _mask_to_patch_grid(proc_mask, H_grid, W_grid, encoder.patch_size)
    masked_patches = patches_spatial[grid]  # (N_masked, D)

    if len(masked_patches) == 0:
        raise ValueError(
            f"Exemplar mask maps to 0 patches on a {H_grid}×{W_grid} grid. "
            "Verify that the mask is non-empty and aligned with the image."
        )

    prototype = masked_patches.mean(axis=0)
    prototype = prototype / (np.linalg.norm(prototype) + 1e-8)

    cluster_prototypes: np.ndarray | None = None
    cluster_labels: np.ndarray | None = None

    if n_clusters is not None and n_clusters > 1:
        if len(masked_patches) < n_clusters:
            log.warning(
                "Only %d exemplar patches but n_clusters=%d — skipping clustering",
                len(masked_patches),
                n_clusters,
            )
        else:
            from sklearn.cluster import AgglomerativeClustering

            ex_n = _l2_normalize(masked_patches)
            # Cosine distance for precomputed agglomerative clustering.
            dist_matrix = np.clip(1.0 - ex_n @ ex_n.T, 0.0, 2.0)
            ac = AgglomerativeClustering(
                n_clusters=n_clusters, metric="precomputed", linkage="average"
            )
            cluster_labels = ac.fit_predict(dist_matrix)
            cluster_prototypes = np.stack(
                [masked_patches[cluster_labels == k].mean(axis=0) for k in range(n_clusters)]
            )
            cluster_prototypes = _l2_normalize(cluster_prototypes)

    log.info(
        "Exemplar: %d masked patches, dim=%d%s",
        len(masked_patches),
        D,
        f", {n_clusters} cluster prototypes" if cluster_prototypes is not None else "",
    )
    return ExemplarFeatures(
        patches=masked_patches,
        prototype=prototype,
        cluster_prototypes=cluster_prototypes,
        cluster_labels=cluster_labels,
    )


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


class SAMDinoEvaluator:
    """Few-shot object detection: SAM3 mask proposals → DINOv3 scoring."""

    def __init__(
        self,
        encoder: DinoEncoder,
        mask_generator: MaskGenerator,
        prefilt_threshold: float = 0.3,
        n_clusters: int | None = 4,
    ) -> None:
        self.encoder = encoder
        self.mask_generator = mask_generator
        self.prefilt_threshold = prefilt_threshold
        self.n_clusters = n_clusters

    # ------------------------------------------------------------------
    # Exemplar
    # ------------------------------------------------------------------

    def prepare_exemplar(self, image: np.ndarray, mask: np.ndarray) -> ExemplarFeatures:
        t0 = time.perf_counter()
        feats = process_exemplar(self.encoder, image, mask, self.n_clusters)
        log.info("Exemplar processed in %.2fs", time.perf_counter() - t0)
        return feats

    # ------------------------------------------------------------------
    # Pre-filter
    # ------------------------------------------------------------------

    def _prefilt(
        self,
        full_patches: np.ndarray,  # (H_grid, W_grid, D) from full-image DINOv3 pass
        sam_masks: list[dict[str, Any]],
        prototype: np.ndarray,
    ) -> list[tuple[CandidateMask, np.ndarray]]:
        """Score each SAM mask by global patch similarity; keep those ≥ threshold.

        Returns (CandidateMask, original_bool_seg) pairs.
        """
        results: list[tuple[CandidateMask, np.ndarray]] = []

        for m in sam_masks:
            seg: np.ndarray = m["segmentation"]
            proc_mask = _preprocess_mask(seg, self.encoder.img_size)

            proto = _masked_prototype(full_patches, proc_mask, self.encoder.patch_size)
            if proto is None:
                continue

            score = _cosine_sim(proto, prototype)
            if score < self.prefilt_threshold:
                continue

            bbox_raw = m.get("bbox")
            bbox = [int(v) for v in bbox_raw] if bbox_raw is not None else _bbox_from_mask(seg)
            results.append((CandidateMask(bbox_xywh=bbox, prefilt_score=score), seg))

        return results

    # ------------------------------------------------------------------
    # Crop scoring
    # ------------------------------------------------------------------

    def _score_crop(
        self,
        image: np.ndarray,
        seg: np.ndarray,
        bbox_xywh: list[int],
        exemplar: ExemplarFeatures,
    ) -> tuple[float | None, float | None, float | None]:
        """DINOv3-encode the crop and return (M1, M2, M3) scores."""
        crop, crop_mask = _crop_to_bbox(image, seg, bbox_xywh)
        if crop.size == 0:
            return None, None, None

        crop_out = self.encoder([crop])
        crop_patches = crop_out.patches[0].cpu().numpy()  # (H, W, D)
        H_g, W_g, D = crop_patches.shape

        proc_crop_mask = _preprocess_mask(crop_mask, self.encoder.img_size)
        grid = _mask_to_patch_grid(proc_crop_mask, H_g, W_g, self.encoder.patch_size)
        cand_patches = crop_patches[grid]
        if len(cand_patches) == 0:
            # Fallback: use all crop patches for very small crops.
            cand_patches = crop_patches.reshape(-1, D)

        s1 = score_method1(cand_patches, exemplar.prototype)
        s2 = score_method2(cand_patches, exemplar.patches)
        s3 = (
            score_method3(cand_patches, exemplar.cluster_prototypes)
            if exemplar.cluster_prototypes is not None
            else None
        )
        return s1, s2, s3

    # ------------------------------------------------------------------
    # Per-image and full evaluation
    # ------------------------------------------------------------------

    def evaluate_image(
        self,
        image_id: str,
        image: np.ndarray,
        exemplar: ExemplarFeatures,
    ) -> ImageResult:
        """Run the full detection pipeline on a single inference image."""
        log.info("Processing '%s'", image_id)
        t0 = time.perf_counter()

        sam_masks = self.mask_generator.generate(image)
        log.info("  SAM3 → %d masks", len(sam_masks))

        # Full-image DINOv3 pass for pre-filtering.
        full_out = self.encoder([image])
        full_patches = full_out.patches[0].cpu().numpy()

        candidates_with_seg = self._prefilt(full_patches, sam_masks, exemplar.prototype)
        log.info(
            "  Pre-filter (≥%.2f): %d/%d passed",
            self.prefilt_threshold,
            len(candidates_with_seg),
            len(sam_masks),
        )

        candidates: list[CandidateMask] = []
        for cand, seg in candidates_with_seg:
            s1, s2, s3 = self._score_crop(image, seg, cand.bbox_xywh, exemplar)
            cand.score_global = s1
            cand.score_patch_cross = s2
            cand.score_cluster = s3
            candidates.append(cand)

        log.info("  Finished in %.2fs", time.perf_counter() - t0)
        return ImageResult(
            image_id=image_id,
            total_masks=len(sam_masks),
            passed_prefilt=len(candidates),
            candidates=candidates,
        )

    def evaluate(
        self,
        exemplar_image: np.ndarray,
        exemplar_mask: np.ndarray,
        inference_images: list[tuple[str, np.ndarray]],
    ) -> list[ImageResult]:
        """End-to-end evaluation: process exemplar once, then score all inference images."""
        exemplar = self.prepare_exemplar(exemplar_image, exemplar_mask)
        return [self.evaluate_image(img_id, img, exemplar) for img_id, img in inference_images]


# ---------------------------------------------------------------------------
# Output serialisation
# ---------------------------------------------------------------------------


def _round_opt(v: float | None) -> float | None:
    return round(v, 4) if v is not None else None


def results_to_dict(results: list[ImageResult]) -> list[dict[str, Any]]:
    """Convert ImageResult list to a JSON-serialisable structure."""
    out = []
    for r in results:
        candidates = []
        for c in r.candidates:
            candidates.append(
                {
                    "bbox_xywh": c.bbox_xywh,
                    "prefilt_score": round(c.prefilt_score, 4),
                    "score_global": _round_opt(c.score_global),
                    "score_patch_cross": _round_opt(c.score_patch_cross),
                    "score_cluster": _round_opt(c.score_cluster),
                }
            )
        out.append(
            {
                "image_id": r.image_id,
                "total_masks": r.total_masks,
                "passed_prefilt": r.passed_prefilt,
                "candidates": candidates,
            }
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def _build_sam_generator(checkpoint: Path, config: Path, device: str) -> MaskGenerator:
    """Attempt to instantiate a SAM3 or SAM2 automatic mask generator."""
    # Try SAM3 first.
    try:
        # isort: off
        from sam3.automatic_mask_generator import SAM3AutomaticMaskGenerator
        from sam3.build_sam import build_sam3
        # isort: on

        model = build_sam3(str(config), str(checkpoint), device=device)
        return SAM3AutomaticMaskGenerator(model)  # type: ignore[return-value]
    except ImportError:
        pass

    # Fall back to SAM2.
    try:
        # isort: off
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
        # isort: on

        model = build_sam2(str(config), str(checkpoint), device=device)
        return SAM2AutomaticMaskGenerator(model)  # type: ignore[return-value]
    except ImportError as exc:
        raise ImportError(
            "Neither sam3 nor sam2 is installed.\n"
            "Install one of:\n"
            "  pip install sam3   # Meta SAM 3\n"
            "  pip install sam2   # Meta SAM 2\n"
            f"(original error: {exc})"
        ) from exc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="SAM3 + DINOv3 few-shot object detection evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- inputs ---
    parser.add_argument("--exemplar-image", type=Path, required=True, help="Exemplar RGB image.")
    parser.add_argument(
        "--exemplar-mask",
        type=Path,
        required=True,
        help="Binary mask image (white = object of interest).",
    )
    parser.add_argument(
        "--inference-dir",
        type=Path,
        required=True,
        help="Directory containing inference images (jpg/png/…).",
    )
    # --- output ---
    parser.add_argument(
        "--output", type=Path, default=Path("results.json"), help="JSON output path."
    )
    # --- encoder ---
    parser.add_argument("--version", default="v3", choices=["v2", "v3"])
    parser.add_argument("--size", default="base", choices=["small", "base", "large", "giant"])
    parser.add_argument(
        "--img-size",
        type=int,
        default=448,
        help="Square resolution fed to DINOv3 (must be divisible by patch size).",
    )
    parser.add_argument("--device", default=None, help="PyTorch device (default: auto).")
    parser.add_argument("--amp", action="store_true", help="Enable bfloat16 autocast.")
    # --- pipeline ---
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Pre-filter cosine similarity threshold.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="Number of exemplar clusters for Method 3 (0 = disable).",
    )
    # --- SAM ---
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        required=True,
        help="SAM3/SAM2 model checkpoint file.",
    )
    parser.add_argument(
        "--sam-config",
        type=Path,
        required=True,
        help="SAM3/SAM2 config file.",
    )

    args = parser.parse_args(argv)

    n_clusters: int | None = args.n_clusters if args.n_clusters > 0 else None

    exemplar_image = _load_rgb(args.exemplar_image)
    exemplar_mask = _load_mask(args.exemplar_mask)
    log.info("Exemplar: %s | mask: %s", args.exemplar_image, args.exemplar_mask)

    image_paths = sorted(p for p in args.inference_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not image_paths:
        log.error("No images found in %s", args.inference_dir)
        return
    inference_images = [(p.stem, _load_rgb(p)) for p in image_paths]
    log.info("Loaded %d inference images from %s", len(inference_images), args.inference_dir)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = DinoEncoder(
        version=args.version,
        size=args.size,
        img_size=args.img_size,
        layers=1,
        device=device,
        amp=args.amp,
    )
    log.info(
        "Encoder: DINO%s-%s  img_size=%d  device=%s",
        args.version,
        args.size,
        args.img_size,
        device,
    )

    mask_generator = _build_sam_generator(args.sam_checkpoint, args.sam_config, device)

    evaluator = SAMDinoEvaluator(
        encoder=encoder,
        mask_generator=mask_generator,
        prefilt_threshold=args.threshold,
        n_clusters=n_clusters,
    )

    t_total = time.perf_counter()
    results = evaluator.evaluate(exemplar_image, exemplar_mask, inference_images)
    log.info("Total evaluation time: %.1fs", time.perf_counter() - t_total)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results_to_dict(results), indent=2))
    log.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
