"""Tutorial 03 — ForegroundHead: PCA-based foreground segmentation.

Run from the repository root:

    python basic_tutorials/03_foreground_head.py [--version v2] [--size small]

ForegroundHead works on a single image with no gallery or training data.
It fits PCA on that image's own patch embeddings and uses the first principal
component (the direction of maximum variance) to separate foreground from
background.

For natural images with a dominant subject the first PC reliably aligns with
the foreground/background boundary.  For texturally uniform images it may not.

Output keys from segment()
--------------------------
foreground_mask         bool ndarray [orig_H, orig_W]    — per-pixel decision
foreground_map          float32 ndarray [orig_H, orig_W] — PC-1 score map (bilinear)
patch_scores            float32 ndarray [N]              — per-patch PC-1 score
foreground_mask_feature bool ndarray [grid_H, grid_W]   — decision at patch resolution
pca_projections         float32 ndarray [N, n_components]— all PCA projections
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
from dinoisawesome import DinoEncoder, ForegroundHead  # noqa: E402

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
        description="ForegroundHead tutorial",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", default="v2", choices=["v2", "v3"])
    parser.add_argument("--size", default="small", choices=["small", "base", "large", "giant"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    encoder = DinoEncoder(
        version=args.version,
        size=args.size,
        img_size=args.img_size,
        layers=1,
        device=args.device,
    )

    # ------------------------------------------------------------------
    # 1. Create the head
    # ------------------------------------------------------------------
    # n_components controls how many PCA components are computed.
    # Only the first is used for the foreground/background decision, but
    # all components are returned in pca_projections for downstream use.
    head = ForegroundHead(encoder, n_components=3)
    log.info(
        "ForegroundHead — block_idx=%d  n_components=%d",
        head.block_idx,
        head.n_components,
    )

    # ------------------------------------------------------------------
    # 2. Segment an image
    # ------------------------------------------------------------------
    image = _make_image(args.img_size, args.img_size)
    result = head.segment(image)

    mask = result["foreground_mask"]  # bool [orig_H, orig_W]
    fg_map = result["foreground_map"]  # float32 [orig_H, orig_W]
    patch_scores = result["patch_scores"]  # float32 [N]
    pca_proj = result["pca_projections"]  # float32 [N, n_components]
    mask_feat = result["foreground_mask_feature"]  # bool [grid_H, grid_W]

    log.info("Output shapes:")
    log.info("  foreground_mask         %s  dtype=%s", mask.shape, mask.dtype)
    log.info("  foreground_map          %s  dtype=%s", fg_map.shape, fg_map.dtype)
    log.info("  patch_scores            %s  dtype=%s", patch_scores.shape, patch_scores.dtype)
    log.info("  foreground_mask_feature %s  dtype=%s", mask_feat.shape, mask_feat.dtype)
    log.info("  pca_projections         %s  dtype=%s", pca_proj.shape, pca_proj.dtype)

    fg_pct = mask.mean() * 100
    log.info("Foreground pixel ratio: %.1f%%", fg_pct)
    log.info("PC-1 score range: [%.4f, %.4f]", patch_scores.min(), patch_scores.max())

    # ------------------------------------------------------------------
    # 3. Adjust the threshold
    # ------------------------------------------------------------------
    # threshold=0.0 (default) splits at the zero-crossing of the mean-centred
    # projection.  Raise it to demand a stronger foreground signal; lower to
    # be more permissive.  Negate to flip the polarity if the mask is inverted.
    result_strict = head.segment(image, threshold=0.1)
    result_lenient = head.segment(image, threshold=-0.05)
    log.info("threshold=0.0   → fg ratio: %.1f%%", mask.mean() * 100)
    log.info("threshold=0.1   → fg ratio: %.1f%%", result_strict["foreground_mask"].mean() * 100)
    log.info("threshold=-0.05 → fg ratio: %.1f%%", result_lenient["foreground_mask"].mean() * 100)

    # ------------------------------------------------------------------
    # 4. Positional debiasing
    # ------------------------------------------------------------------
    # debias=True removes the positional subspace from patch embeddings before
    # PCA.  This suppresses grid-layout artifacts in the segmentation map and
    # makes the head respond more to appearance rather than location.
    result_deb = head.segment(image, debias=True)
    log.info(
        "With debias=True → fg ratio: %.1f%%",
        result_deb["foreground_mask"].mean() * 100,
    )

    # ------------------------------------------------------------------
    # 5. Choosing a specific transformer block
    # ------------------------------------------------------------------
    # Different blocks capture features at different abstraction levels.
    # By default the head uses the last block the encoder is configured for.
    # Override with block_idx to pick a different one.
    depth = len(encoder.backbone.blocks)
    earlier_head = ForegroundHead(encoder, block_idx=depth - 4, n_components=3)
    result_early = earlier_head.segment(image)
    log.info(
        "block_idx=%d (earlier) → fg ratio: %.1f%%",
        earlier_head.block_idx,
        result_early["foreground_mask"].mean() * 100,
    )

    # ------------------------------------------------------------------
    # 6. Using PCA projections for downstream tasks
    # ------------------------------------------------------------------
    # pca_projections[:, 0] = PC-1 scores (same as patch_scores)
    # pca_projections[:, 1] = PC-2 scores (second-largest variance direction)
    pc1 = pca_proj[:, 0]
    pc2 = pca_proj[:, 1]
    log.info("PC-1 variance captured (proxy: score std): %.4f", float(pc1.std()))
    log.info("PC-2 variance captured (proxy: score std): %.4f", float(pc2.std()))

    log.info("Tutorial 03 complete.")


if __name__ == "__main__":
    main()
