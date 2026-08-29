"""Latency benchmark: per-image DINOv3 encoder inference latency across backbone size x
resolution x batch size.

Unlike ``run_experiments.py``/``visualize_results.py`` (accuracy on the abc3 exemplar/
query pairs, cached per config), this is a pure throughput/latency micro-benchmark of
``DinoEncoder.forward()`` — the same call every frame goes through in production (see
``dinoisawesome/instance_detection.py::extract_patch_tokens``). No dataset, no crop
cache: synthetic random images are used since forward-pass latency depends only on
tensor shape, not pixel content. A "small test" by design — a handful of warmup +
timed reps per (size, resolution, batch_size) cell, not a statistically heavy benchmark
suite.

``max_batch_size`` is set to the largest requested batch size so every batch size runs
as one true forward pass rather than being internally chunked (see ``DinoEncoder.
forward()``'s chunking docstring) — otherwise a "batch_size=32" test with the default
``max_batch_size=16`` would silently measure two chunked passes of 16, not one pass of 32.

Usage:
    python latency_benchmark.py
    python latency_benchmark.py --sizes large --resolutions 1024 --batch-sizes 1 8 16 32
    python latency_benchmark.py --n-reps 20 --n-warmup 5
"""

# Logging — must be before torch import
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("resolution_ablation.latency_benchmark")

import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Literal, cast  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "multiscale_ablation"))
from common import REPO_ROOT  # noqa: E402

from dinoisawesome import DinoEncoder  # noqa: E402

DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")
DEFAULT_SIZES = ("small", "base", "large")
DEFAULT_RESOLUTIONS = (256, 512, 768, 1024, 1536)
DEFAULT_BATCH_SIZES = (1, 8, 16, 32)
# (size, resolution) cells known to OOM past this batch size — skip rather than let one
# cell's CUDA OOM abort the whole sweep. large@1536px is the only cell hit so far.
MAX_BATCH_SIZE_OVERRIDES: dict[tuple[str, int], int] = {("large", 1536): 8}
# Arbitrary "camera frame"-like native size — content and native size don't affect
# forward-pass latency (DinoEncoder resizes to img_size regardless), only batch shape
# after resize does, so one fixed synthetic image is reused for every cell.
NATIVE_SIZE = (1280, 960)

OUTPUT_DIR = REPO_ROOT / "outputs" / "object_detection" / "resolution_ablation"
RESULTS_ROOT = OUTPUT_DIR / "results"
FIGURES_ROOT = RESULTS_ROOT / "figures"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=list(DEFAULT_SIZES),
        choices=["small", "base", "large"],
        help="DINOv3 backbone sizes to benchmark (default: small base large)",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
        help="img_size values to benchmark (default: 256 512 768 1024 1536)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="Batch sizes to benchmark (default: 1 8 16 32)",
    )
    parser.add_argument(
        "--n-reps", type=int, default=10, help="Timed repetitions per cell (default: 10)"
    )
    parser.add_argument(
        "--n-warmup", type=int, default=3, help="Untimed warmup reps per cell (default: 3)"
    )
    parser.add_argument(
        "--skip-plots", action="store_true", help="Only write the CSV, skip figures"
    )
    return parser.parse_args()


def _make_batch(batch_size: int) -> list[Image.Image]:
    rng = np.random.default_rng(0)
    w, h = NATIVE_SIZE
    img = Image.fromarray(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8))
    return [img] * batch_size


def _time_forward(
    encoder: DinoEncoder, batch: list[Image.Image], layer_idx: int, n_reps: int
) -> np.ndarray:
    """Returns *n_reps* wall-clock latencies (seconds) for one full ``forward()`` call."""
    is_cuda = encoder.device.type == "cuda"
    latencies = np.empty(n_reps, dtype=np.float64)
    for i in range(n_reps):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        encoder.forward(batch, layers=[layer_idx])
        if is_cuda:
            torch.cuda.synchronize()
        latencies[i] = time.perf_counter() - t0
    return latencies


def run_benchmark(
    sizes: list[str],
    resolutions: list[int],
    batch_sizes: list[int],
    n_reps: int,
    n_warmup: int,
) -> pd.DataFrame:
    rows = []
    max_bs = max(batch_sizes)
    for size in sizes:
        for resolution in resolutions:
            encoder = DinoEncoder(
                version="v3",
                size=cast(Literal["small", "base", "large", "giant"], size),
                img_size=resolution,
                weights_dir=DINO_WEIGHTS_DIR,
                amp=True,
                max_batch_size=max_bs,
            )
            layer_idx = len(encoder.backbone.blocks) - 1
            device_name = torch.cuda.get_device_name(0) if encoder.device.type == "cuda" else "cpu"
            log.info(
                "=== size=%s resolution=%dpx | layer_idx=%d | device=%s ===",
                size,
                resolution,
                layer_idx,
                device_name,
            )
            max_bs_for_cell = MAX_BATCH_SIZE_OVERRIDES.get((size, resolution))
            for batch_size in batch_sizes:
                if max_bs_for_cell is not None and batch_size > max_bs_for_cell:
                    log.info(
                        "  batch_size=%-3d skipped (size=%s resolution=%dpx capped at "
                        "batch_size=%d — known OOM cell)",
                        batch_size,
                        size,
                        resolution,
                        max_bs_for_cell,
                    )
                    continue
                batch = _make_batch(batch_size)
                _time_forward(encoder, batch, layer_idx, n_warmup)  # warmup, discarded
                latencies_s = _time_forward(encoder, batch, layer_idx, n_reps)
                latencies_ms = latencies_s * 1000
                per_image_ms = latencies_ms / batch_size
                row = {
                    "dino_size": size,
                    "resolution": resolution,
                    "config": f"{size}@{resolution}px",
                    "batch_size": batch_size,
                    "device": device_name,
                    "mean_batch_latency_ms": float(latencies_ms.mean()),
                    "std_batch_latency_ms": float(latencies_ms.std()),
                    "mean_ms_per_image": float(per_image_ms.mean()),
                    "std_ms_per_image": float(per_image_ms.std()),
                    "throughput_img_per_sec": float(1000.0 / per_image_ms.mean()),
                }
                rows.append(row)
                log.info(
                    "  batch_size=%-3d mean=%7.2fms/image (%.1f img/s), batch_latency=%.1fms",
                    batch_size,
                    row["mean_ms_per_image"],
                    row["throughput_img_per_sec"],
                    row["mean_batch_latency_ms"],
                )
            del encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_latency_vs_batch_size(df: pd.DataFrame, out_path: Path) -> None:
    resolutions = sorted(df["resolution"].unique())
    sizes = sorted(df["dino_size"].unique(), key=lambda s: ["small", "base", "large"].index(s))
    fig, axes = plt.subplots(
        1,
        len(resolutions),
        figsize=(5 * len(resolutions), 4.5),
        constrained_layout=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, resolution in zip(axes, resolutions):
        cell = df[df["resolution"] == resolution]
        for size in sizes:
            series = cell[cell["dino_size"] == size].sort_values("batch_size")
            if series.empty:
                continue
            ax.errorbar(
                series["batch_size"],
                series["mean_ms_per_image"],
                yerr=series["std_ms_per_image"],
                marker="o",
                label=size,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df["batch_size"].unique()))
        ax.set_xticklabels(sorted(df["batch_size"].unique()))
        ax.set_xlabel("batch size")
        ax.set_ylabel("ms / image")
        ax.set_title(f"resolution={resolution}px")
    axes[-1].legend(title="size")
    fig.suptitle("Per-image latency vs. batch size")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


def plot_throughput_vs_batch_size(df: pd.DataFrame, out_path: Path) -> None:
    resolutions = sorted(df["resolution"].unique())
    sizes = sorted(df["dino_size"].unique(), key=lambda s: ["small", "base", "large"].index(s))
    fig, axes = plt.subplots(
        1,
        len(resolutions),
        figsize=(5 * len(resolutions), 4.5),
        constrained_layout=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, resolution in zip(axes, resolutions):
        cell = df[df["resolution"] == resolution]
        for size in sizes:
            series = cell[cell["dino_size"] == size].sort_values("batch_size")
            if series.empty:
                continue
            ax.plot(series["batch_size"], series["throughput_img_per_sec"], marker="o", label=size)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df["batch_size"].unique()))
        ax.set_xticklabels(sorted(df["batch_size"].unique()))
        ax.set_xlabel("batch size")
        ax.set_ylabel("images / sec")
        ax.set_title(f"resolution={resolution}px")
    axes[-1].legend(title="size")
    fig.suptitle("Throughput vs. batch size")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


def plot_latency_heatmap(df: pd.DataFrame, order: list[str], out_path: Path) -> None:
    """config x batch_size heatmap of ms/image. Not reused from ``multiscale_ablation/
    visualize_results.py``'s ``plot_metric_heatmap`` — that one hardcodes
    ``vmin=0, vmax=1`` for its 0-1 accuracy metrics, which would clip every latency
    value (tens to hundreds of ms) to the same saturated color.
    """
    pivot = (
        df.groupby(["config", "batch_size"])["mean_ms_per_image"]
        .mean()
        .unstack("batch_size")
        .reindex(index=order)
    )
    vmax = float(np.nanmax(pivot.values))
    fig, ax = plt.subplots(
        figsize=(1.5 * len(pivot.columns) + 2.5, 0.4 * len(pivot) + 2), constrained_layout=True
    )
    im = ax.imshow(pivot.values, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if val < vmax * 0.6 else "black",
                )
    fig.colorbar(im, ax=ax, label="ms / image")
    ax.set_xlabel("batch size")
    ax.set_title("Per-image latency (ms) per config x batch_size")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


def main() -> None:
    args = _parse_args()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    df = run_benchmark(args.sizes, args.resolutions, args.batch_sizes, args.n_reps, args.n_warmup)
    log.info("Benchmark done in %.1fs.", time.time() - t_start)

    csv_path = RESULTS_ROOT / "latency_records.csv"
    df.to_csv(csv_path, index=False)
    log.info("wrote %s (%d rows)", csv_path, len(df))

    if args.skip_plots:
        return

    plot_latency_vs_batch_size(df, FIGURES_ROOT / "latency_vs_batch_size.png")
    plot_throughput_vs_batch_size(df, FIGURES_ROOT / "throughput_vs_batch_size.png")

    order = sorted(df["config"].unique())
    plot_latency_heatmap(df, order, FIGURES_ROOT / "latency_heatmap_by_batch_size.png")


if __name__ == "__main__":
    main()
