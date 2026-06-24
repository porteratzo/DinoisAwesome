"""Tutorial 02 — Gallery: building, indexing, and retrieving from a feature store.

Run from the repository root:

    python basic_tutorials/02_gallery.py [--gallery-dir /tmp/my_gallery]

The Gallery pairs two storage layers:

* Parquet DataFrames  — lightweight metadata (image_id, row, col, labels, split).
* Memory-mapped .npy  — float32 patch embeddings, never fully loaded into RAM.

Disk layout::

    <root>/
        gallery_config.json          # model metadata
        patches.parquet              # one row per (image_id, patch_row, patch_col)
        cls_tokens.parquet           # one row per image_id
        cls_tokens.npy               # (N_images, L, D) mmap'd CLS array
        embeddings/<image_id>.npy    # (L, H, W, D) per-image patch embeddings
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
from dinoisawesome import DinoEncoder, Gallery  # noqa: E402

# isort: on


def _make_image(h: int = 224, w: int = 224) -> np.ndarray:
    """Synthetic uint8 RGB image: dark background with a bright green disc."""
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    cy, cx = h // 2, w // 2
    ys, xs = np.ogrid[:h, :w]
    disc = (ys - cy) ** 2 + (xs - cx) ** 2 <= (min(h, w) // 3) ** 2
    img[disc] = [50, 200, 50]
    return img


def demo(encoder: DinoEncoder, gallery_dir: Path) -> None:
    # ------------------------------------------------------------------
    # 1. Build — extract features and write everything to disk
    # ------------------------------------------------------------------
    n_train, n_val = 6, 2
    images = [_make_image(encoder.img_size, encoder.img_size) for _ in range(n_train + n_val)]
    ids = [f"img_{i:03d}" for i in range(len(images))]
    splits = ["train"] * n_train + ["val"] * n_val

    # image_labels: propagated to every patch and the CLS token of that image.
    # patch_labels: override for individual patches (image_id, row, col).
    image_labels = {
        "img_000": ["category_a"],
        "img_001": ["category_a"],
        "img_002": ["category_b"],
    }
    patch_labels = {
        ("img_000", encoder.grid_h // 2, encoder.grid_w // 2): ["centre"],
    }

    gallery = Gallery.build(
        encoder=encoder,
        images=images,
        image_ids=ids,
        out_dir=gallery_dir,
        split=splits,
        image_labels=image_labels,
        patch_labels=patch_labels,
    )

    log.info("Gallery built at %s", gallery_dir)
    log.info("  Images:  %d", len(gallery.cls_tokens))
    log.info("  Patches: %d", len(gallery.patches))
    log.info("  Stored transformer blocks: %s", gallery.config.block_indices)
    log.info("  Embed dim: %d", gallery.config.embed_dim)

    # ------------------------------------------------------------------
    # 2. Inspect the DataFrames
    # ------------------------------------------------------------------
    log.info("patches.parquet columns: %s", list(gallery.patches.columns))
    log.info("First patch row:\n  %s", gallery.patches.iloc[0].to_dict())

    # ------------------------------------------------------------------
    # 3. Image-level retrieval — top-k by CLS cosine similarity
    # ------------------------------------------------------------------
    query_img = _make_image(encoder.img_size, encoder.img_size)
    q_out = encoder([query_img])
    query_cls = q_out.cls[0].cpu().float().numpy()  # (D,)

    # Narrow to the training split and return the top 3 most similar images.
    top_images = gallery.retrieve_images(query_cls, k=3, split="train")
    log.info("Top-3 images by CLS similarity (train split):")
    for _, row in top_images.iterrows():
        log.info("  image_id=%-10s  similarity=%.4f", row["image_id"], row["similarity"])

    # ------------------------------------------------------------------
    # 4. Patch-level retrieval — top-k patches
    # ------------------------------------------------------------------
    cy, cx = encoder.grid_h // 2, encoder.grid_w // 2
    query_patch = q_out.patches[0, cy, cx].cpu().float().numpy()  # (D,)

    top_patches = gallery.retrieve(query_patch, k=5, split="train")
    log.info("Top-5 patches by cosine similarity (train split):")
    for _, row in top_patches.iterrows():
        log.info(
            "  image_id=%-10s  (row=%2d, col=%2d)  similarity=%.4f",
            row["image_id"],
            row["row"],
            row["col"],
            row["similarity"],
        )

    # ------------------------------------------------------------------
    # 5. Filtering the patch index
    # ------------------------------------------------------------------
    # filter() returns a sub-DataFrame; no embeddings are loaded.
    cat_a_patches = gallery.filter(has_labels=["category_a"])
    train_patches = gallery.filter(split="train")
    centre_patches = gallery.filter(has_labels=["centre"])

    log.info("Patches with 'category_a' label: %d", len(cat_a_patches))
    log.info("Patches in train split:           %d", len(train_patches))
    log.info("Patches labelled 'centre':        %d", len(centre_patches))

    # Combine filters — all conditions are ANDed.
    cat_a_train = gallery.filter(has_labels=["category_a"], split="train")
    log.info("category_a AND train:             %d", len(cat_a_train))

    # ------------------------------------------------------------------
    # 6. Load raw patch embeddings into a float32 array
    # ------------------------------------------------------------------
    # load_embeddings opens per-image .npy files via mmap and copies only the
    # requested patches — never loads the full gallery into RAM.
    cat_a_embs = gallery.load_embeddings(cat_a_patches)  # (N, D) float32
    log.info("category_a embeddings shape: %s  dtype=%s", cat_a_embs.shape, cat_a_embs.dtype)

    # Specify a block index explicitly when the gallery stores multiple layers.
    block = gallery.config.block_indices[-1]
    embs_last = gallery.load_embeddings(cat_a_patches, block_idx=block)
    log.info("Same embeddings, explicit block_idx=%d: %s", block, embs_last.shape)

    # CLS embeddings come from the single mmap'd cls_tokens.npy — no per-image I/O.
    ids_out, cls_embs = gallery.load_cls_embeddings()
    log.info("CLS embeddings shape: %s  (all images)", cls_embs.shape)

    # ------------------------------------------------------------------
    # 7. Adding labels after the fact
    # ------------------------------------------------------------------
    # Labelling is lazy — add_labels updates the in-memory DataFrame.
    # Pass save=True (default) to flush to disk immediately, or call
    # gallery._save_metadata() once at the end to batch the writes.
    unlabelled = gallery.filter(split="val")
    gallery.add_labels(unlabelled, ["val_reviewed"], save=True)
    reviewed = gallery.filter(has_labels=["val_reviewed"])
    log.info("After add_labels: patches with 'val_reviewed': %d", len(reviewed))

    # ------------------------------------------------------------------
    # 8. Two-stage retrieval — CLS pre-select → patch re-rank
    # ------------------------------------------------------------------
    # Step 1: cheap CLS scan narrows the candidate set.
    candidate_ids = gallery.retrieve_images(query_cls, k=3)["image_id"].tolist()
    log.info("Two-stage retrieval — CLS stage candidate ids: %s", candidate_ids)

    # Step 2: patch-level search within the candidate images only.
    refined = gallery.retrieve(query_patch, k=3, image_ids=candidate_ids)
    log.info("Patch-level re-rank results:")
    for _, row in refined.iterrows():
        log.info(
            "  image_id=%-10s  (row=%2d, col=%2d)  similarity=%.4f",
            row["image_id"],
            row["row"],
            row["col"],
            row["similarity"],
        )

    # ------------------------------------------------------------------
    # 9. Load an existing gallery from disk
    # ------------------------------------------------------------------
    # Gallery(root) re-opens the parquet files and mmap's cls_tokens.npy.
    # No embeddings are loaded until you call load_embeddings or retrieve.
    reloaded = Gallery(gallery_dir)
    log.info("Re-loaded gallery from disk — images: %d", len(reloaded.cls_tokens))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Gallery tutorial",
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
        with tempfile.TemporaryDirectory(prefix="dinoisawesome_gallery_") as tmpdir:
            demo(encoder, Path(tmpdir) / "gallery")

    log.info("Tutorial 02 complete.")


if __name__ == "__main__":
    main()
