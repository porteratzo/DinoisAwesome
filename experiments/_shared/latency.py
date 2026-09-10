"""Wall-clock timing for latency/throughput measurement.

GPU kernels queue asynchronously, so a bare `time.perf_counter()` around a CUDA call
measures dispatch time, not device compute time. Every helper here calls
`torch.cuda.synchronize()` (when CUDA is available) immediately before starting and
before stopping the clock, so elapsed time reflects actual device work — the same
correctness requirement every `fundamental/` sweep script's own resolution/size/scale
axis needs before a latency number is trustworthy.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def cuda_timer() -> Iterator[dict[str, float]]:
    """Times the wrapped block, synchronizing around it. The yielded dict gets
    `elapsed_s` set once the block exits.

    Usage::

        with cuda_timer() as t:
            out = encoder(images, layers=[layer_idx])
        t["elapsed_s"]
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    timing: dict[str, float] = {}
    try:
        yield timing
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing["elapsed_s"] = time.perf_counter() - start


def images_per_sec(n_images: int, elapsed_s: float) -> float:
    """Throughput in images/sec; `nan` if elapsed_s is non-positive (e.g. a cache-only
    call that did no device work) rather than raising a ZeroDivisionError."""
    if elapsed_s <= 0:
        return float("nan")
    return n_images / elapsed_s
