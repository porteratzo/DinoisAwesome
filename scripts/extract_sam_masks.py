"""Step 1 of 2: Extract SAM masks for all inference images and save to disk.

Run this on a machine / GPU that can fit SAM3 or SAM2 in memory, then
transfer the output directory to a machine with DINOv3 installed and run
``score_dino.py`` to complete the pipeline.

Output format
-------------
One ``<image_stem>.npz`` file per image in ``--output-dir``, containing:
  ``segmentations``: (N, H, W) bool — one boolean mask per proposal
  ``bboxes``       : (N, 4)  int32 — [x, y, w, h] (filled with -1 when SAM
                                      did not provide a bbox)
  ``has_bbox``     : (N,)    bool  — True when SAM reported a valid bbox

Usage
-----
python scripts/extract_sam_masks.py \\
    --inference-dir  images/ \\
    --output-dir     masks/ \\
    --sam-checkpoint path/to/sam3.pt \\
    --sam-config     path/to/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

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
# isort: on


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _build_sam_generator(checkpoint: Path, config: Path, device: str) -> Any:
    """Instantiate a SAM3 or SAM2 automatic mask generator."""
    try:
        from sam3.automatic_mask_generator import SAM3AutomaticMaskGenerator
        from sam3.build_sam import build_sam3

        model = build_sam3(str(config), str(checkpoint), device=device)
        log.info("Loaded SAM3 from %s", checkpoint)
        return SAM3AutomaticMaskGenerator(model)
    except ImportError:
        pass

    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        model = build_sam2(str(config), str(checkpoint), device=device)
        log.info("Loaded SAM2 from %s", checkpoint)
        return SAM2AutomaticMaskGenerator(model)
    except ImportError as exc:
        raise ImportError(
            "Neither sam3 nor sam2 is installed.\n"
            "Install one of:\n"
            "  pip install sam3\n"
            "  pip install sam2\n"
            f"(original error: {exc})"
        ) from exc


def save_masks(output_dir: Path, image_id: str, masks: list[dict[str, Any]]) -> None:
    """Serialise a SAM mask list to ``<output_dir>/<image_id>.npz``."""
    if not masks:
        np.savez(
            output_dir / f"{image_id}.npz",
            segmentations=np.zeros((0, 1, 1), dtype=bool),
            bboxes=np.zeros((0, 4), dtype=np.int32),
            has_bbox=np.zeros(0, dtype=bool),
        )
        return

    segmentations = np.stack([m["segmentation"].astype(bool) for m in masks])  # (N, H, W)
    has_bbox = np.array(["bbox" in m for m in masks], dtype=bool)
    bboxes = np.array(
        [[int(v) for v in m["bbox"]] if "bbox" in m else [-1, -1, -1, -1] for m in masks],
        dtype=np.int32,
    )  # (N, 4)
    np.savez(
        output_dir / f"{image_id}.npz",
        segmentations=segmentations,
        bboxes=bboxes,
        has_bbox=has_bbox,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract SAM3/SAM2 masks for all images in a directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inference-dir",
        type=Path,
        required=True,
        help="Directory containing inference images (jpg/png/…).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where per-image .npz mask files will be written.",
    )
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
        help="SAM3/SAM2 model config file.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device (default: cuda if available, else cpu).",
    )

    args = parser.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Building SAM generator on device=%s", device)
    mask_generator = _build_sam_generator(args.sam_checkpoint, args.sam_config, device)

    image_paths = sorted(p for p in args.inference_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not image_paths:
        log.error("No images found in %s", args.inference_dir)
        return
    log.info("Found %d inference images in %s", len(image_paths), args.inference_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()
    for path in image_paths:
        image_id = path.stem
        image = _load_rgb(path)
        t0 = time.perf_counter()
        masks = mask_generator.generate(image)
        log.info("  %s → %d masks in %.2fs", image_id, len(masks), time.perf_counter() - t0)
        save_masks(args.output_dir, image_id, masks)

    log.info(
        "Saved masks for %d images to %s  (total %.1fs)",
        len(image_paths),
        args.output_dir,
        time.perf_counter() - t_total,
    )


if __name__ == "__main__":
    main()
