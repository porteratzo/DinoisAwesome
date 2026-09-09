"""Layer-index ablation: which DINOv3 transformer block gives the best anomaly
localization, and is it the same block across categories?

`AnomalyHead`/`PrototypeAnomalyHead` both accept a `block_idx` argument, but every method
in `methods.py` builds its Gallery with `layers=1` (last block only) and never overrides
`block_idx` -- so `anomalydino_v3`/`dinov3_proto_*`'s "last block" choice in
`run_experiments.py` has never actually been checked against any alternative. This mirrors
`fundamental/`'s repeated finding that layer choice matters a lot for localization quality,
applied here to image/pixel AUROC and AUPRO instead of oracle IoU.

One Gallery is built per category with *every* transformer block stored (`encoder.layers`
set to the full block-index list before `Gallery.build()`), so training images are encoded
once regardless of how many blocks get swept -- only the per-block kNN/prototype scoring
is repeated. Query images are still re-encoded once per block per test image inside
`AnomalyHead.predict()`/`PrototypeAnomalyHead.predict()` (they always request a single
explicit block), so wall-clock cost scales with `n_blocks * n_test_images` at inference time --
keep `--limit` modest for a first pass.

Gallery is cached under a synthetic method name (`layer_ablation_<method>`) so it never
collides with `run_experiments.py`'s own `anomalydino_v3`/`dinov3_proto_*` cache.

Usage:
    python layer_ablation.py --categories bottle carpet --limit 20
    python layer_ablation.py --method dinov3_proto --size base --stride 2
"""

# Logging — must be before torch import
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("layer_ablation")

import argparse
import gc
import os

import numpy as np
import pandas as pd
import torch
from analyze_results import compute_metrics
from common import CATEGORIES, RESULTS_ROOT, gallery_dir, image_id_for, resolve_paths

from dinoisawesome import AnomalyHead, DinoEncoder, Gallery, PrototypeAnomalyHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINO_WEIGHTS_DIR = os.environ.get("DINO_WEIGHTS_DIR")
_OUT_DIR = RESULTS_ROOT / "layer_ablation"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=CATEGORIES, choices=CATEGORIES)
    parser.add_argument(
        "--method",
        default="anomalydino_v3",
        choices=["anomalydino_v3", "dinov3_proto"],
    )
    parser.add_argument("--size", default="small", choices=["small", "base", "large", "giant"])
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=20, help="Cap test set size")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sweep every Nth block instead of all of them (the last block is always kept).",
    )
    return parser.parse_args()


def _block_indices(n_blocks: int, stride: int) -> list[int]:
    indices = set(range(0, n_blocks, stride))
    indices.add(n_blocks - 1)
    return sorted(indices)


def _build_head(
    method: str,
    gallery: Gallery,
    encoder: DinoEncoder,
    block_idx: int,
) -> AnomalyHead | PrototypeAnomalyHead:
    if method == "anomalydino_v3":
        return AnomalyHead(gallery=gallery, encoder=encoder, block_idx=block_idx, split="train")
    return PrototypeAnomalyHead(
        gallery=gallery,
        encoder=encoder,
        n_prototypes=1,
        retrieval_k=1,
        aggregation="max",
        masking=False,
        block_idx=block_idx,
        split="train",
    )


def _score_block(
    head: AnomalyHead | PrototypeAnomalyHead, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    records: list[dict] = []
    maps: dict[str, np.ndarray] = {}
    for _, row in test_df.iterrows():
        image_id = image_id_for(row["label"], row["image_path"])
        result = head.predict(row["image_path"])
        records.append(
            {
                "image_id": image_id,
                "label_index": int(row["label_index"]),
                "image_score": result["score"],
                "mask_path": row["mask_path"],
            }
        )
        maps[image_id] = result["anomaly_map"].astype(np.float16)
    return pd.DataFrame.from_records(records), maps


def main() -> None:
    args = _parse_args()

    encoder = DinoEncoder(
        version="v3",
        size=args.size,
        img_size=args.img_size,
        layers=1,
        device=DEVICE,
        weights_dir=DINO_WEIGHTS_DIR,
    )
    n_blocks = len(encoder.backbone.blocks)
    block_indices = _block_indices(n_blocks, args.stride)
    log.info("Backbone has %d blocks; sweeping %s", n_blocks, block_indices)
    encoder.layers = block_indices  # store every swept block, one encoder pass per train image

    rows: list[dict] = []
    for category in args.categories:
        train_paths, test_df = resolve_paths(
            category, limit=args.limit, train_limit=args.train_limit
        )
        gallery = Gallery.build(
            encoder=encoder,
            images=train_paths,
            image_ids=[p.stem for p in train_paths],
            out_dir=gallery_dir(category, f"layer_ablation_{args.method}"),
            split="train",
        )

        for block_idx in block_indices:
            try:
                head = _build_head(args.method, gallery, encoder, block_idx)
                df, maps = _score_block(head, test_df)
                metrics = compute_metrics(df, maps)
            except Exception:
                log.exception("[%s/block=%d] FAILED, skipping", category, block_idx)
                continue
            rows.append(
                {"category": category, "method": args.method, "block_idx": block_idx, **metrics}
            )
            log.info(
                "[%s/block=%d] image_auroc=%.4f aupro=%.4f",
                category,
                block_idx,
                metrics["image_auroc"],
                metrics["aupro"],
            )

        del gallery
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        log.error("No results produced.")
        return

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(rows)
    results_df.to_csv(_OUT_DIR / f"metrics_{args.method}.csv", index=False)
    log.info("Wrote %s", _OUT_DIR / f"metrics_{args.method}.csv")

    _plot(results_df, args.method)


def _plot(results_df: pd.DataFrame, method: str) -> None:
    import matplotlib.pyplot as plt

    figures_dir = _OUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for category in results_df["category"].unique():
        sub = results_df[results_df["category"] == category].sort_values("block_idx")
        axes[0].plot(sub["block_idx"], sub["image_auroc"], marker="o", label=category)
        axes[1].plot(sub["block_idx"], sub["aupro"], marker="o", label=category)
    axes[0].set_title("image AUROC vs. block index")
    axes[1].set_title("AUPRO vs. block index")
    for ax in axes:
        ax.set_xlabel("block index")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
    fig.suptitle(f"layer ablation: {method}")
    fig.tight_layout()
    fig.savefig(figures_dir / f"layer_ablation_{method}.png", dpi=150)
    plt.close(fig)
    log.info("Figure written to %s", figures_dir / f"layer_ablation_{method}.png")


if __name__ == "__main__":
    main()
