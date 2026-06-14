"""Step 2 of 2: Score pre-extracted SAM masks with DINOv3.

Loads the ``.npz`` mask files produced by ``extract_sam_masks.py``, encodes
the exemplar and all inference-image crops with DINOv3, then writes
detection scores to a JSON file.  SAM is never imported or loaded.

Usage
-----
python scripts/score_dino.py \\
    --exemplar-image exemplar.jpg \\
    --exemplar-mask  mask.png \\
    --inference-dir  images/ \\
    --masks-dir      masks/ \\
    --output         results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

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

# Pull shared pipeline helpers from eval_sam_dino without importing SAM.
sys.path.insert(0, str(Path(__file__).parent))
from eval_sam_dino import (  # noqa: E402
    CandidateMask,
    ExemplarFeatures,
    ImageResult,
    SAMDinoEvaluator,
    results_to_dict,
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _load_mask_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


# ---------------------------------------------------------------------------
# Mask file I/O
# ---------------------------------------------------------------------------


def load_sam_masks(masks_dir: Path, image_id: str) -> list[dict]:
    """Load SAM masks from the .npz file written by extract_sam_masks.py."""
    npz_path = masks_dir / f"{image_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"No mask file found for '{image_id}'. "
            f"Expected: {npz_path}. "
            "Run extract_sam_masks.py first."
        )
    data = np.load(npz_path)
    segmentations: np.ndarray = data["segmentations"]  # (N, H, W) bool
    bboxes: np.ndarray = data["bboxes"]  # (N, 4) int32
    has_bbox: np.ndarray = data["has_bbox"]  # (N,) bool

    masks = []
    for i in range(len(segmentations)):
        m: dict = {"segmentation": segmentations[i]}
        if has_bbox[i]:
            m["bbox"] = bboxes[i].tolist()
        masks.append(m)
    return masks


# ---------------------------------------------------------------------------
# Evaluator: disk-backed masks instead of live SAM
# ---------------------------------------------------------------------------


class _NullMaskGenerator:
    """Satisfies the MaskGenerator protocol; never actually called."""

    def generate(self, image: np.ndarray) -> list[dict]:  # pragma: no cover
        return []


class MaskFileSAMDinoEvaluator(SAMDinoEvaluator):
    """SAMDinoEvaluator that loads pre-computed masks from disk instead of running SAM.

    All scoring logic (_prefilt, _score_crop, prepare_exemplar) is inherited
    unchanged.  Only evaluate_image is overridden to swap the live SAM call for
    a disk load.
    """

    def __init__(
        self,
        encoder: DinoEncoder,
        masks_dir: Path,
        prefilt_threshold: float = 0.3,
        n_clusters: int | None = 4,
    ) -> None:
        super().__init__(
            encoder=encoder,
            mask_generator=_NullMaskGenerator(),
            prefilt_threshold=prefilt_threshold,
            n_clusters=n_clusters,
        )
        self.masks_dir = masks_dir

    def evaluate_image(
        self,
        image_id: str,
        image: np.ndarray,
        exemplar: ExemplarFeatures,
    ) -> ImageResult:
        log.info("Processing '%s'", image_id)
        t0 = time.perf_counter()

        sam_masks = load_sam_masks(self.masks_dir, image_id)
        log.info("  Loaded %d SAM masks from disk", len(sam_masks))

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Score pre-extracted SAM masks with DINOv3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- exemplar ---
    parser.add_argument("--exemplar-image", type=Path, required=True, help="Exemplar RGB image.")
    parser.add_argument(
        "--exemplar-mask",
        type=Path,
        required=True,
        help="Binary mask image (white = object of interest).",
    )
    # --- images ---
    parser.add_argument(
        "--inference-dir",
        type=Path,
        required=True,
        help="Directory containing inference images (jpg/png/…). "
        "Images are needed for crop-level DINO encoding.",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        required=True,
        help="Directory with .npz mask files produced by extract_sam_masks.py.",
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

    args = parser.parse_args(argv)

    n_clusters: int | None = args.n_clusters if args.n_clusters > 0 else None

    exemplar_image = _load_rgb(args.exemplar_image)
    exemplar_mask = _load_mask_image(args.exemplar_mask)
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

    evaluator = MaskFileSAMDinoEvaluator(
        encoder=encoder,
        masks_dir=args.masks_dir,
        prefilt_threshold=args.threshold,
        n_clusters=n_clusters,
    )

    t_total = time.perf_counter()
    results = evaluator.evaluate(exemplar_image, exemplar_mask, inference_images)
    log.info("Total scoring time: %.1fs", time.perf_counter() - t_total)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results_to_dict(results), indent=2))
    log.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
