"""Tutorial 04 — AnomalyHead: kNN anomaly detection backed by a Gallery.

Run from the repository root:

    python basic_tutorials/04_anomaly_head.py [--gallery-dir /tmp/anomaly]

AnomalyHead maintains a memory bank of "normal" patch embeddings.  At
inference it computes the k-nearest-neighbour distance from each query patch
to the bank.  Patches far from all reference patches get high anomaly scores.

The image-level score is the mean of the top-1 % patch scores — a robust
aggregate that is sensitive to localised anomalies without being dominated by
a single outlier.

Fit  → AnomalyHead.build(encoder, normal_images, ids, gallery_dir)
     or AnomalyHead(existing_gallery, encoder)   if gallery already exists

Score → result = head.predict(image)
        result["score"]        float  — image-level anomaly score
        result["anomaly_map"]  ndarray [orig_H, orig_W] float32
        result["patch_scores"] ndarray [N] float32
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
from dinoisawesome import AnomalyHead, DinoEncoder, Gallery  # noqa: E402

# isort: on


def _make_normal(h: int = 224, w: int = 224) -> np.ndarray:
    """Synthetic 'normal' image: dark background with a bright green disc."""
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    cy, cx = h // 2, w // 2
    ys, xs = np.ogrid[:h, :w]
    disc = (ys - cy) ** 2 + (xs - cx) ** 2 <= (min(h, w) // 3) ** 2
    img[disc] = [50, 200, 50]
    return img


def _make_anomalous(normal: np.ndarray) -> np.ndarray:
    """Invert colours to produce an out-of-distribution image."""
    return (255 - normal).astype(np.uint8)


def demo(encoder: DinoEncoder, gallery_dir: Path) -> None:
    # ------------------------------------------------------------------
    # 1. Build from scratch — wraps Gallery.build then loads the memory bank
    # ------------------------------------------------------------------
    normal_images = [_make_normal(encoder.img_size, encoder.img_size) for _ in range(6)]
    normal_ids = [f"normal_{i:03d}" for i in range(len(normal_images))]

    head = AnomalyHead.build(
        encoder=encoder,
        images=normal_images,
        image_ids=normal_ids,
        gallery_dir=gallery_dir,
        split="train",
        filter_split="train",  # only train patches enter the memory bank
        num_neighbours=1,
    )
    log.info("Memory bank: %d patches  embed_dim=%d", head.gallery_size, head.embed_dim)

    # ------------------------------------------------------------------
    # 2. Score a normal query (in-distribution)
    # ------------------------------------------------------------------
    query_normal = _make_normal(encoder.img_size, encoder.img_size)
    r_normal = head.predict(query_normal)

    log.info("Normal query:")
    log.info("  image score:        %.6f  (lower = more normal)", r_normal["score"])
    log.info("  anomaly_map shape:  %s", r_normal["anomaly_map"].shape)
    log.info("  patch_scores shape: %s", r_normal["patch_scores"].shape)
    ps = r_normal["patch_scores"]
    log.info("  patch score range:  [%.4f, %.4f]", ps.min(), ps.max())

    # ------------------------------------------------------------------
    # 3. Score an anomalous query (out-of-distribution)
    # ------------------------------------------------------------------
    query_anom = _make_anomalous(query_normal)
    r_anom = head.predict(query_anom)

    log.info("Anomalous query (inverted image):")
    log.info("  image score:        %.6f", r_anom["score"])
    log.info("  correctly ranked higher: %s", r_anom["score"] > r_normal["score"])

    # ------------------------------------------------------------------
    # 4. Use positional debiasing at inference
    # ------------------------------------------------------------------
    # debias=True removes positional information from the query patches before
    # kNN scoring.  Useful when normal images were captured at varying positions.
    r_deb = head.predict(query_normal, debias=True)
    log.info("With debias=True — normal query score: %.6f", r_deb["score"])

    # ------------------------------------------------------------------
    # 5. Coreset subsampling — reduce memory bank size
    # ------------------------------------------------------------------
    # For large galleries the memory bank can be huge.  coreset_ratio=0.5 keeps
    # the 50 % most representative patches via greedy k-center.
    head_cs = AnomalyHead.build(
        encoder=encoder,
        images=normal_images,
        image_ids=normal_ids,
        gallery_dir=gallery_dir / "coreset",
        split="train",
        filter_split="train",
        num_neighbours=1,
        coreset_ratio=0.5,
    )
    log.info(
        "Coreset (ratio=0.5): bank %d → %d patches",
        head.gallery_size,
        head_cs.gallery_size,
    )

    # ------------------------------------------------------------------
    # 6. Load from an existing gallery
    # ------------------------------------------------------------------
    # If the gallery is already on disk, skip AnomalyHead.build and
    # construct directly.  The memory bank is rebuilt from the stored embeddings.
    gallery = Gallery(gallery_dir)
    head_reload = AnomalyHead(
        gallery=gallery,
        encoder=encoder,
        num_neighbours=1,
        split="train",
    )
    r_reload = head_reload.predict(query_normal)
    log.info("Re-loaded head — normal query score: %.6f", r_reload["score"])

    # ------------------------------------------------------------------
    # 7. kNN with k > 1
    # ------------------------------------------------------------------
    # num_neighbours=3 averages over the 3 nearest neighbours per patch,
    # producing smoother anomaly maps.
    head_k3 = AnomalyHead(gallery=gallery, encoder=encoder, num_neighbours=3)
    r_k3 = head_k3.predict(query_normal)
    log.info("k=3 neighbours — normal query score: %.6f", r_k3["score"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="AnomalyHead tutorial",
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
        help="Parent directory for gallery output. Defaults to a temp directory.",
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
        with tempfile.TemporaryDirectory(prefix="dinoisawesome_anomaly_") as tmpdir:
            demo(encoder, Path(tmpdir))

    log.info("Tutorial 04 complete.")


if __name__ == "__main__":
    main()
