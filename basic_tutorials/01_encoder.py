"""Tutorial 01 — DinoEncoder: extracting CLS tokens and patch embeddings.

Run from the repository root:

    python basic_tutorials/01_encoder.py [--version v2] [--size small]

The DinoEncoder wraps a pretrained DINO ViT (v2 or v3) and exposes two kinds
of features per image:

* CLS token  — one vector per image; captures global semantics.
* Patch grid — a (H, W, D) spatial map; each cell covers one patch of pixels.

For DINOv2 the patch size is 14 px; for DINOv3 it is 16 px.  An input of
224 × 224 pixels therefore yields a 16 × 16 (v2) or 14 × 14 (v3) patch grid.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# isort: off
from dinoisawesome import DinoEncoder  # noqa: E402

# isort: on


def _make_image(h: int = 224, w: int = 224) -> np.ndarray:
    """Synthetic uint8 RGB image: dark background with a bright green disc."""
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    cy, cx = h // 2, w // 2
    ys, xs = np.ogrid[:h, :w]
    disc = (ys - cy) ** 2 + (xs - cx) ** 2 <= (min(h, w) // 3) ** 2
    img[disc] = [50, 200, 50]
    return img


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="DinoEncoder tutorial",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", default="v2", choices=["v2", "v3"])
    parser.add_argument("--size", default="small", choices=["small", "base", "large", "giant"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # 1. Construct an encoder
    # ------------------------------------------------------------------
    # layers=1 → extract only the last transformer block (the default for most tasks).
    encoder = DinoEncoder(
        version=args.version,
        size=args.size,
        img_size=args.img_size,
        layers=1,
        device=args.device,
        amp=False,
    )

    depth = len(encoder.backbone.blocks)
    log.info("Loaded DINO%s-%s", args.version, args.size)
    log.info(
        "  patch_size=%d  img_size=%d  grid=%dx%d  embed_dim=%d  depth=%d  device=%s",
        encoder.patch_size,
        encoder.img_size,
        encoder.grid_h,
        encoder.grid_w,
        encoder.backbone.embed_dim,
        depth,
        encoder.device,
    )

    image = _make_image(args.img_size, args.img_size)

    # ------------------------------------------------------------------
    # 2. Single image, single layer
    # ------------------------------------------------------------------
    # Pass a numpy (H, W, 3) uint8 array, a PIL Image, or a list of either.
    # With layers=1, the L (layer) axis is squeezed away.
    out = encoder(image)
    log.info("Single image, single layer:")
    log.info("  cls     shape: %s  — (B, D)", tuple(out.cls.shape))
    log.info("  patches shape: %s  — (B, H, W, D)", tuple(out.patches.shape))

    # Access the CLS token for the first (only) image in the batch
    cls_vec: np.ndarray = out.cls[0].cpu().float().numpy()  # (D,)
    log.info("  CLS vector norm: %.4f", float(np.linalg.norm(cls_vec)))

    # Access the patch at grid position (row=4, col=4)
    patch_vec: np.ndarray = out.patches[0, 4, 4].cpu().float().numpy()  # (D,)
    log.info("  Patch (row=4, col=4) norm: %.4f", float(np.linalg.norm(patch_vec)))

    # ------------------------------------------------------------------
    # 3. Batch of images
    # ------------------------------------------------------------------
    batch = [_make_image(args.img_size, args.img_size) for _ in range(4)]
    out_batch = encoder(batch)
    log.info("Batch of %d images, single layer:", len(batch))
    log.info("  cls     shape: %s  — (B, D)", tuple(out_batch.cls.shape))
    log.info("  patches shape: %s  — (B, H, W, D)", tuple(out_batch.patches.shape))

    # ------------------------------------------------------------------
    # 4. Multi-layer extraction
    # ------------------------------------------------------------------
    # Pass layers=N to get the last N blocks; the L axis is kept.
    out_multi = encoder(image, layers=2)
    log.info("Single image, last 2 blocks (layers=2):")
    log.info("  cls     shape: %s  — (B, L, D)", tuple(out_multi.cls.shape))
    log.info("  patches shape: %s  — (B, L, H, W, D)", tuple(out_multi.patches.shape))

    # Explicit block indices: pass a list.  Requires knowing the model depth.
    last_two = [depth - 2, depth - 1]
    out_explicit = encoder(image, layers=last_two)
    log.info("Same with explicit indices %s:", last_two)
    log.info("  cls     shape: %s", tuple(out_explicit.cls.shape))
    log.info("  patches shape: %s", tuple(out_explicit.patches.shape))

    # Extract a specific layer from the multi-layer output (L axis = 0 → earlier block)
    layer0_patches = out_multi.patches[0, 0]  # (H, W, D) — second-to-last block
    layer1_patches = out_multi.patches[0, 1]  # (H, W, D) — last block
    log.info("  Layer 0 patches shape: %s", tuple(layer0_patches.shape))
    log.info("  Layer 1 patches shape: %s", tuple(layer1_patches.shape))

    # ------------------------------------------------------------------
    # 5. Positional debiasing
    # ------------------------------------------------------------------
    # DINO patch tokens carry both content and position information.
    # debias=True projects out the positional subspace (estimated from a blank
    # image via SVD) so that patches are compared purely by content.
    # The debiased patches are re-L2-normalised; their norm stays ≈ 1.0.
    out_deb = encoder(image, debias=True)
    norm_before = out.patches.norm(dim=-1).mean().item()
    norm_after = out_deb.patches.norm(dim=-1).mean().item()
    log.info("Positional debiasing:")
    log.info("  Mean patch norm before debiasing: %.4f", norm_before)
    log.info("  Mean patch norm after  debiasing: %.4f  (expect ≈ 1.0)", norm_after)

    # The positional basis is computed lazily on first access and cached.
    log.info(
        "  Positional basis shape: %s  (D × svd_components)",
        tuple(encoder.positional_basis.shape),
    )

    # ------------------------------------------------------------------
    # 6. Cosine similarity between two images
    # ------------------------------------------------------------------
    # CLS tokens are L2-normalised by the backbone, so dot product = cosine sim.
    image_a = _make_image(args.img_size, args.img_size)
    image_b = 255 - image_a  # inverted — very different
    out_ab = encoder([image_a, image_b])
    cls_a = out_ab.cls[0].cpu().float()
    cls_b = out_ab.cls[1].cpu().float()
    sim = float((cls_a @ cls_b).item())
    log.info("Cosine similarity — same vs inverted image: %.4f", sim)

    log.info("Tutorial 01 complete.")


if __name__ == "__main__":
    main()
