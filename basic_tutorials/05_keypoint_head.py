"""Tutorial 05 — KeypointHead: gallery-backed keypoint registration and matching.

Run from the repository root:

    python basic_tutorials/05_keypoint_head.py [--gallery-dir /tmp/kp_gallery]

KeypointHead stores named keypoints as labelled patches inside a Gallery.
Registration maps pixel coordinates on a reference image to the gallery's
patch grid.  At find-time a single encoder forward pass is made, then each
registered label's stored embedding is matched against all query patches.

Workflow
--------
1. Build a Gallery that contains your reference image(s).
2. Create KeypointHead(gallery, encoder).
3. register(image_id, points, labels, orig_size) — tag reference patches.
4. head.save() — flush label changes to disk.
5. head.find(query_image) — returns [(label, (px, py), similarity), ...].
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# isort: off
from dinoisawesome import DinoEncoder, Gallery, KeypointHead  # noqa: E402

# isort: on

# Keypoint names used throughout this tutorial
_KP_LABELS = ["disc_centre", "disc_top", "disc_left"]


def _make_image(h: int = 224, w: int = 224) -> np.ndarray:
    """Synthetic uint8 RGB image: dark background with a bright green disc."""
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 3
    ys, xs = np.ogrid[:h, :w]
    disc = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius**2
    img[disc] = [50, 200, 50]
    return img, (cx, cy, radius)  # type: ignore[return-value]


def demo(encoder: DinoEncoder, gallery_dir: Path) -> None:
    h = w = encoder.img_size

    # ------------------------------------------------------------------
    # 1. Build a gallery that contains the reference image
    # ------------------------------------------------------------------
    ref_image, (cx, cy, radius) = _make_image(h, w)
    ref_id = "reference"

    gallery = Gallery.build(
        encoder=encoder,
        images=[ref_image],
        image_ids=[ref_id],
        out_dir=gallery_dir,
        split="train",
    )
    log.info("Gallery built: %d image, %d patches", len(gallery.cls_tokens), len(gallery.patches))

    # ------------------------------------------------------------------
    # 2. Create KeypointHead
    # ------------------------------------------------------------------
    head = KeypointHead(gallery, encoder)
    log.info("Registered labels (before registration): %s", head.registered_labels)

    # ------------------------------------------------------------------
    # 3. Register keypoints
    # ------------------------------------------------------------------
    # points are (x, y) pixel coordinates in the ORIGINAL image space.
    # The head maps them to the nearest patch-grid cell automatically.
    points = [
        [cx, cy],  # centre of the disc
        [cx, cy - radius],  # top edge of the disc
        [cx - radius, cy],  # left edge of the disc
    ]
    head.register(
        image_id=ref_id,
        points=points,
        labels=_KP_LABELS,
        orig_size=(w, h),  # (orig_w, orig_h)
    )

    log.info("Registered labels (after registration): %s", head.registered_labels)

    # ------------------------------------------------------------------
    # 4. Persist label changes to disk
    # ------------------------------------------------------------------
    # register() only updates the in-memory DataFrame.
    # save() flushes it so the labels survive a Gallery reload.
    head.save()
    log.info("Labels saved to disk.")

    # ------------------------------------------------------------------
    # 5. Find keypoints in a query image (identical to reference)
    # ------------------------------------------------------------------
    query_image, _ = _make_image(h, w)
    matches = head.find(query_image)

    log.info("Keypoint matches (query == reference):")
    for m in matches:
        log.info(
            "  %-14s  point=(%3d, %3d)  similarity=%.4f",
            m["label"],
            m["point"][0],
            m["point"][1],
            m["similarity"],
        )

    # ------------------------------------------------------------------
    # 6. Find a specific subset of labels
    # ------------------------------------------------------------------
    partial = head.find(query_image, labels=["disc_centre", "disc_top"])
    log.info("Partial find (2 labels): %d results", len(partial))

    # ------------------------------------------------------------------
    # 7. Find with positional debiasing
    # ------------------------------------------------------------------
    # debias=True removes the positional subspace from query patches before
    # matching.  Improves robustness when the object can appear at different
    # positions in the query image.
    matches_deb = head.find(query_image, debias=True)
    log.info("With debias=True:")
    for m in matches_deb:
        log.info(
            "  %-14s  point=(%3d, %3d)  similarity=%.4f",
            m["label"],
            m["point"][0],
            m["point"][1],
            m["similarity"],
        )

    # ------------------------------------------------------------------
    # 8. Reload from disk and verify labels persist
    # ------------------------------------------------------------------
    gallery2 = Gallery(gallery_dir)
    head2 = KeypointHead(gallery2, encoder)
    log.info("Reloaded registered_labels: %s", head2.registered_labels)

    matches2 = head2.find(query_image)
    log.info("Matches after reload: %d  (expect %d)", len(matches2), len(_KP_LABELS))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="KeypointHead tutorial",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", default="v2", choices=["v2", "v3"])
    parser.add_argument("--size", default="small", choices=["small", "base", "large", "giant"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--gallery-dir",
        type=Path,
        default=None,
        help="Where to write gallery files. Defaults to a temp directory.",
    )
    args = parser.parse_args(argv)

    encoder = DinoEncoder(
        version=args.version,
        size=args.size,
        img_size=args.img_size,
        layers=1,
        device=args.device,
    )

    if args.gallery_dir is not None:
        args.gallery_dir.mkdir(parents=True, exist_ok=True)
        demo(encoder, args.gallery_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="dinoisawesome_keypoints_") as tmpdir:
            demo(encoder, Path(tmpdir) / "gallery")

    log.info("Tutorial 05 complete.")


if __name__ == "__main__":
    main()
