"""Resolution + model-size ablation: sweep ``CropConfig.img_size``
(256 / 512 / 768 / 1024 / 1536)
and ``CropConfig.dino_size`` (small / base / large) over their full cross product,
reusing ``multiscale_ablation``'s pipeline unchanged.

Both fields are already part of the crop-cache hash key (see
``multiscale_ablation/common.py``'s module docstring), so each (size, resolution)
combo gets its own cache namespace under the *same* ``multiscale_ablation`` cache tree
automatically -- no new caching mechanism needed, and the existing large/1024px cache
(the multiscale ablation's default) is reused as-is if present. This script is a thin
driver: for each requested (size, resolution) pair, build a ``CropConfig`` and call
``multiscale_ablation.run_experiments.run_pair`` for every data pair, exactly as
``multiscale_ablation/run_experiments.py``'s own ``main()`` does for its single fixed
config.

``layer_idx`` (which transformer block to read patch tokens from) is architecture-
dependent, not resolution-dependent: DINOv3 small/base have 12 blocks, large has 24
(verified by instantiating each backbone; ``CropConfig``'s own default, ``layer_idx=23``,
is exactly "last block" for the large model it was tuned on). Rather than hardcode a
per-size table here and risk it drifting from the actual loaded checkpoint, this script
always asks the just-built encoder for its real block count and uses ``depth - 1``
(last block) for every size. ``visualize_results.py`` (stage 2, deliberately torch-free)
*does* hardcode that resulting small/base/large mapping since it never loads an encoder
— see its ``LAYER_IDX_BY_SIZE``, which must stay in sync with what a run here actually
produces for any size added in the future.

Usage:
    python run_experiments.py                                   # 3 sizes x 5 res x every pair
    python run_experiments.py --sizes large --resolutions 512 1024
    python run_experiments.py --part-types RHa --limit-pairs 1   # smoke test
    python run_experiments.py --methods global mid close
    python run_experiments.py --include-kmeans --include-classifiers
    python run_experiments.py --force                            # ignore existing cache
"""

# Logging — must be before torch import
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("resolution_ablation.run_experiments")

import argparse
import dataclasses
import gc
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Literal, cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "multiscale_ablation"))
from common import DEFAULT_CROP_CONFIG, DEFAULT_SCORING_CONFIG, all_pairs  # noqa: E402
from methods import all_method_names  # noqa: E402
from run_experiments import run_pair  # noqa: E402

from dinoisawesome import DinoEncoder  # noqa: E402

DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")
DEFAULT_RESOLUTIONS = (256, 512, 768, 1024, 1536)
DEFAULT_SIZES = ("small", "base", "large")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
        help="img_size values to sweep (default: 256 512 768 1024 1536)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=list(DEFAULT_SIZES),
        choices=["small", "base", "large"],
        help="DINOv3 backbone sizes to sweep (default: small base large)",
    )
    parser.add_argument(
        "--part-types", nargs="+", default=None, help="Restrict to these part types"
    )
    parser.add_argument(
        "--instance-types", nargs="+", default=None, help="Restrict to these instance-type groups"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        # include_kmeans=True/include_classifiers=True here just so the (default-off)
        # k-means/linear-probe/svm names remain valid --methods choices; whether they're
        # actually built by default is controlled by --include-kmeans/--include-classifiers
        # / build_method_states, not by this list.
        choices=all_method_names(
            list(DEFAULT_CROP_CONFIG.scales),
            DEFAULT_SCORING_CONFIG.kmeans_ks,
            include_kmeans=True,
            include_classifiers=True,
        ),
        help=(
            "Restrict to these methods (default: every registered method except "
            "k-means/svm/linear-probe)"
        ),
    )
    parser.add_argument(
        "--limit-pairs", type=int, default=None, help="Cap number of pairs (smoke test)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Ignore existing cache, recompute everything"
    )
    parser.add_argument(
        "--include-kmeans",
        action="store_true",
        help="Also build the k-means method families (skipped by default — see methods.py)",
    )
    parser.add_argument(
        "--include-classifiers",
        action="store_true",
        help=(
            "Also build the linear-probe/svm method families (skipped by default — see methods.py)"
        ),
    )
    return parser.parse_args()


def run_sweep_point(
    size: str,
    resolution: int,
    pairs: list,
    method_names: list[str] | None,
    force: bool,
    include_kmeans: bool,
    include_classifiers: bool,
) -> list[tuple[str, str]]:
    """Run every *pairs* at one (size, resolution) combo. Returns (tag, error) failures."""
    scoring_cfg = DEFAULT_SCORING_CONFIG

    encoder = DinoEncoder(
        version=DEFAULT_CROP_CONFIG.dino_version,
        size=cast(Literal["small", "base", "large", "giant"], size),
        img_size=resolution,
        weights_dir=DINO_WEIGHTS_DIR,
        amp=True,
    )
    patch_size = encoder.patch_size
    # See module docstring: layer_idx is architecture- (not resolution-) dependent, so
    # it's derived live from the actual loaded backbone rather than guessed per size.
    layer_idx = len(encoder.backbone.blocks) - 1
    crop_cfg = dataclasses.replace(
        DEFAULT_CROP_CONFIG, dino_size=size, img_size=resolution, layer_idx=layer_idx
    )
    log.info(
        "=== size=%s resolution=%dpx | DINOv%s-%s | depth=%d layer_idx=%d | "
        "patch_size=%d | grid=%dx%d | crop_hash=%s ===",
        size,
        resolution,
        crop_cfg.dino_version[1],
        size,
        layer_idx + 1,
        layer_idx,
        patch_size,
        encoder.grid_h,
        encoder.grid_w,
        crop_cfg.hash(),
    )

    tag = f"size={size}/res={resolution}"
    failures: list[tuple[str, str]] = []
    for pair in pairs:
        try:
            run_pair(
                pair,
                encoder,
                patch_size,
                crop_cfg,
                scoring_cfg,
                method_names,
                force,
                include_kmeans,
                include_classifiers,
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch going, report at the end
            log.error("[%s/%s] FAILED: %s", tag, pair.slug, exc)
            traceback.print_exc()
            failures.append((f"{tag}/{pair.slug}", str(exc)))

    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return failures


def main() -> None:
    args = _parse_args()

    pairs = all_pairs()
    if args.part_types:
        pairs = [p for p in pairs if p.part_type in args.part_types]
    if args.instance_types:
        pairs = [p for p in pairs if p.instance_type in args.instance_types]
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]
    if not pairs:
        raise RuntimeError("No pairs matched --part-types/--instance-types filters")

    log.info(
        "Sweeping sizes=%s resolutions=%s over %d pair(s): %s",
        args.sizes,
        args.resolutions,
        len(pairs),
        [p.slug for p in pairs],
    )

    t_start = time.time()
    all_failures: list[tuple[str, str]] = []
    for size in args.sizes:
        for resolution in args.resolutions:
            all_failures.extend(
                run_sweep_point(
                    size,
                    resolution,
                    pairs,
                    args.methods,
                    args.force,
                    args.include_kmeans,
                    args.include_classifiers,
                )
            )

    elapsed = time.time() - t_start
    log.info("Done in %.1fs. %d failure(s).", elapsed, len(all_failures))
    for slug, msg in all_failures:
        log.error("  %s: %s", slug, msg)


if __name__ == "__main__":
    main()
