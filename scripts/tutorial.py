"""Tutorial: feature extraction, gallery building, and heads with DINOisAwesome.

Run from the repository root (package must be installed via ``pip install -e .``):

    python scripts/tutorial.py [--version v2] [--size small] [--img-size 224]

Sections
--------
1. DinoEncoder    — single-image and batch forward passes; CLS and patch embeddings
2. Gallery        — build from synthetic images, image-level and patch-level retrieval
3. ForegroundHead — per-image PCA foreground segmentation (no gallery required)
4. AnomalyHead    — kNN anomaly detection backed by a Gallery memory bank
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np

# Logging must be configured before torch is imported; torch registers its own
# handlers at import time on some builds which can interfere with basicConfig.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# isort: off
from dinoisawesome import (  # noqa: E402
    AnomalyHead,
    DinoEncoder,
    ForegroundHead,
    Gallery,
)

# isort: on


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------


def _solid_disc(
    h: int = 224, w: int = 224, *, bg: tuple[int, int, int] = (30, 30, 30)
) -> np.ndarray:
    """Return a uint8 (H, W, 3) image with a bright green disc on a dark background.

    The disc acts as a distinct "foreground" subject that PCA and kNN heads can
    latch onto.
    """
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 3
    ys, xs = np.ogrid[:h, :w]
    disc = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius**2
    img[disc] = [50, 200, 50]
    return img


# ---------------------------------------------------------------------------
# Section 1 — DinoEncoder
# ---------------------------------------------------------------------------


def demo_encoder(encoder: DinoEncoder) -> None:
    """Show CLS and patch shapes for single images, batches, and multi-layer extraction."""
    log.info("=" * 60)
    log.info("SECTION 1: DinoEncoder")
    log.info("=" * 60)

    image = _solid_disc()  # numpy (H, W, 3) uint8

    # ------------------------------------------------------------------
    # 1a. Single image, single layer (encoder was built with layers=1)
    # ------------------------------------------------------------------
    out = encoder(image)
    # cls:     (B=1, D)         — one CLS token per image
    # patches: (B=1, H, W, D)  — spatial patch grid, H*W = num_patches
    log.info("1a. Single image, single layer:")
    log.info("    cls shape:     %s", tuple(out.cls.shape))
    log.info("    patches shape: %s", tuple(out.patches.shape))

    # ------------------------------------------------------------------
    # 1b. Batch of images
    # ------------------------------------------------------------------
    batch = [_solid_disc() for _ in range(4)]
    out_batch = encoder(batch)
    log.info("1b. Batch of 4 images, single layer:")
    log.info("    cls shape:     %s", tuple(out_batch.cls.shape))
    log.info("    patches shape: %s", tuple(out_batch.patches.shape))

    # ------------------------------------------------------------------
    # 1c. Multi-layer extraction — last 2 transformer blocks on the fly
    # ------------------------------------------------------------------
    # Passing layers=N at call time overrides the encoder's default.
    # The L axis is added when more than one layer is requested.
    out_multi = encoder(image, layers=2)
    # cls:     (B=1, L=2, D)
    # patches: (B=1, L=2, H, W, D)
    log.info("1c. Single image, last 2 layers (layers=2 override):")
    log.info("    cls shape:     %s  — (B, L, D)", tuple(out_multi.cls.shape))
    log.info("    patches shape: %s  — (B, L, H, W, D)", tuple(out_multi.patches.shape))

    # ------------------------------------------------------------------
    # 1d. Positional debiasing
    # ------------------------------------------------------------------
    # debias=True projects out the positional subspace estimated from a blank image.
    # Useful when patch identity (texture/colour) matters more than patch position.
    out_deb = encoder(image, debias=True)
    mean_norm = out_deb.patches.norm(dim=-1).mean().item()
    log.info("1d. Debiased patches L2-norm mean: %.4f  (expect ≈ 1.0)", mean_norm)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info(
        "Encoder summary: %s-%s  patch_size=%d  grid=%dx%d  embed_dim=%d  device=%s",
        encoder.version,
        encoder.size,
        encoder.patch_size,
        encoder.grid_h,
        encoder.grid_w,
        encoder.backbone.embed_dim,
        encoder.device,
    )


# ---------------------------------------------------------------------------
# Section 2 — Gallery
# ---------------------------------------------------------------------------


def demo_gallery(encoder: DinoEncoder, gallery_dir: Path) -> Gallery:
    """Build a gallery from synthetic images then demonstrate image and patch retrieval."""
    log.info("=" * 60)
    log.info("SECTION 2: Gallery")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 2a. Build — extract features and write to disk
    # ------------------------------------------------------------------
    n_train, n_val = 6, 2
    all_images = [_solid_disc() for _ in range(n_train + n_val)]
    all_ids = [f"img_{i:03d}" for i in range(len(all_images))]
    all_splits = ["train"] * n_train + ["val"] * n_val

    # Attach an "example_label" to the first two training images so we can
    # demonstrate label-based filtering later.
    image_labels = {img_id: ["example_label"] for img_id in all_ids[:2]}

    gallery = Gallery.build(
        encoder=encoder,
        images=all_images,
        image_ids=all_ids,
        out_dir=gallery_dir,
        split=all_splits,
        image_labels=image_labels,
    )
    log.info("2a. Gallery built at %s", gallery_dir)
    log.info("    Images indexed: %d", len(gallery.cls_tokens))
    log.info("    Patches indexed: %d", len(gallery.patches))
    log.info("    Stored blocks: %s", gallery.config.block_indices)

    # ------------------------------------------------------------------
    # 2b. Image-level retrieval — top-k by CLS cosine similarity
    # ------------------------------------------------------------------
    query_image = _solid_disc()
    query_out = encoder([query_image])
    query_cls = query_out.cls[0].cpu().float().numpy()  # (D,)

    top_images = gallery.retrieve_images(query_cls, k=3, split="train")
    log.info("2b. Top-3 training images by CLS similarity:")
    for _, row in top_images.iterrows():
        log.info("    image_id=%-10s  similarity=%.4f", row["image_id"], row["similarity"])

    # ------------------------------------------------------------------
    # 2c. Patch-level retrieval — top-k patches from the training split
    # ------------------------------------------------------------------
    # Pull the embedding at the centre of the query's patch grid.
    cy, cx = encoder.grid_h // 2, encoder.grid_w // 2
    query_patch = query_out.patches[0, cy, cx].cpu().float().numpy()  # (D,)

    top_patches = gallery.retrieve(query_patch, k=5, split="train")
    log.info("2c. Top-5 patches by patch cosine similarity (train split):")
    for _, row in top_patches.iterrows():
        log.info(
            "    image_id=%-10s  (row=%2d, col=%2d)  similarity=%.4f",
            row["image_id"],
            row["row"],
            row["col"],
            row["similarity"],
        )

    # ------------------------------------------------------------------
    # 2d. Label-based filtering
    # ------------------------------------------------------------------
    labelled_patches = gallery.filter(has_labels=["example_label"])
    log.info(
        "2d. Patches with 'example_label': %d  (expected ~%d)",
        len(labelled_patches),
        2 * encoder.grid_h * encoder.grid_w,
    )

    # Two-stage retrieval: narrow by CLS first, then search patch space
    candidate_ids = gallery.retrieve_images(query_cls, k=3)["image_id"].tolist()
    refined = gallery.retrieve(query_patch, k=3, image_ids=candidate_ids)
    log.info("2e. Two-stage retrieval (CLS→patch) — %d matches:", len(refined))
    for _, row in refined.iterrows():
        log.info("    image_id=%-10s  similarity=%.4f", row["image_id"], row["similarity"])

    return gallery


# ---------------------------------------------------------------------------
# Section 3 — ForegroundHead
# ---------------------------------------------------------------------------


def demo_foreground(encoder: DinoEncoder) -> None:
    """Segment a single image into foreground / background via per-image PCA."""
    log.info("=" * 60)
    log.info("SECTION 3: ForegroundHead")
    log.info("=" * 60)

    head = ForegroundHead(encoder, n_components=3)

    image = _solid_disc()
    result = head.segment(image, debias=False, threshold=0.0)

    # foreground_mask  — bool ndarray [orig_H, orig_W]  (resized from patch grid)
    # foreground_map   — float32 ndarray [orig_H, orig_W]  (PC-1 score map)
    # patch_scores     — float32 ndarray [N]  (per-patch PC-1 value)
    # pca_projections  — float32 ndarray [N, n_components]
    mask = result["foreground_mask"]
    fg_map = result["foreground_map"]
    patches = result["patch_scores"]

    log.info("Foreground pixel ratio:   %.1f%%", mask.mean() * 100)
    log.info("PC-1 score range:         [%.4f, %.4f]", patches.min(), patches.max())
    log.info("foreground_map shape:     %s", fg_map.shape)
    log.info("patch_scores shape:       %s", patches.shape)
    log.info("pca_projections shape:    %s", result["pca_projections"].shape)

    log.info(
        "Tip: if the mask looks inverted (background captured), "
        "negate the threshold — e.g. head.segment(image, threshold=-1e-9)"
    )


# ---------------------------------------------------------------------------
# Section 4 — AnomalyHead
# ---------------------------------------------------------------------------


def demo_anomaly(encoder: DinoEncoder, anomaly_dir: Path) -> None:
    """Build a kNN memory bank from normal images and score queries."""
    log.info("=" * 60)
    log.info("SECTION 4: AnomalyHead")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 4a. Build — wraps Gallery.build then loads the memory bank into RAM
    # ------------------------------------------------------------------
    normal_images = [_solid_disc() for _ in range(5)]
    normal_ids = [f"normal_{i:03d}" for i in range(len(normal_images))]

    head = AnomalyHead.build(
        encoder=encoder,
        images=normal_images,
        image_ids=normal_ids,
        gallery_dir=anomaly_dir,
        split="train",
        filter_split="train",  # only load train patches into the memory bank
        num_neighbours=1,
    )
    log.info("4a. AnomalyHead ready:")
    log.info("    Memory bank: %d patches  embed_dim=%d", head.gallery_size, head.embed_dim)

    # ------------------------------------------------------------------
    # 4b. Score a normal-looking query (low anomaly score expected)
    # ------------------------------------------------------------------
    normal_query = _solid_disc()
    r_normal = head.predict(normal_query)

    log.info("4b. Normal query (similar to training images):")
    log.info("    image score: %.6f", r_normal["score"])
    log.info("    anomaly_map shape: %s", r_normal["anomaly_map"].shape)
    log.info("    patch_scores shape: %s", r_normal["patch_scores"].shape)

    # ------------------------------------------------------------------
    # 4c. Score an anomalous query (inverted colours — very different)
    # ------------------------------------------------------------------
    anomalous_query = 255 - normal_query
    r_anom = head.predict(anomalous_query)

    log.info("4c. Anomalous query (inverted image — out-of-distribution):")
    log.info("    image score: %.6f  (expect > normal score)", r_anom["score"])

    higher = r_anom["score"] > r_normal["score"]
    log.info("    Anomaly correctly ranked higher than normal: %s", higher)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="DINOisAwesome tutorial: encoder, gallery, and heads",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", default="v2", choices=["v2", "v3"])
    parser.add_argument("--size", default="small", choices=["small", "base", "large", "giant"])
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Square input resolution fed to the encoder.",
    )
    parser.add_argument("--device", default=None, help="PyTorch device (default: auto)")
    parser.add_argument(
        "--gallery-dir",
        type=Path,
        default=None,
        help=(
            "Parent directory for tutorial gallery output. "
            "Defaults to a temporary directory that is cleaned up on exit."
        ),
    )
    args = parser.parse_args(argv)

    log.info(
        "Tutorial config — version=%s  size=%s  img_size=%d",
        args.version,
        args.size,
        args.img_size,
    )

    encoder = DinoEncoder(
        version=args.version,
        size=args.size,
        img_size=args.img_size,
        layers=1,
        device=args.device,
        amp=False,
    )

    def _run(base: Path) -> None:
        demo_encoder(encoder)
        gallery = demo_gallery(encoder, base / "gallery")
        _ = gallery  # kept in scope; remove if unused downstream
        demo_foreground(encoder)
        demo_anomaly(encoder, base / "anomaly_gallery")

    if args.gallery_dir is not None:
        args.gallery_dir.mkdir(parents=True, exist_ok=True)
        _run(args.gallery_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="dinoisawesome_tutorial_") as tmpdir:
            _run(Path(tmpdir))

    log.info("Tutorial complete.")


if __name__ == "__main__":
    main()
