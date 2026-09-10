"""Bootstrap confidence intervals for comparing scores across configs/methods.

Every `fundamental/` sweep script already reports mean +/- std across folds or samples,
but std alone doesn't say whether two configs' means are distinguishable from
resampling noise — exactly the concern the 1-1-vs-5-3 cross-validation framework in
`pooled_gallery_cv.py` exists to address for fold-to-fold noise specifically. These
helpers give the same per-sample arrays a percentile bootstrap CI, and an (unpaired)
bootstrap comparison for "is A's mean actually above B's, or is this within noise."
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the mean of *values*. Returns (mean, lo, hi); all
    `nan` if *values* has no finite entries."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    resampled = rng.choice(values, size=(n_boot, len(values)), replace=True)
    boot_means = resampled.mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return float(values.mean()), float(lo), float(hi)


def bootstrap_prob_greater(
    values_a: np.ndarray, values_b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> float:
    """Fraction of bootstrap resamples where mean(A) > mean(B) — an informal
    significance proxy from two independent (not paired, possibly different-sized)
    sample arrays. Near 0.5 means "indistinguishable"; near 0 or 1 means one config's
    mean is consistently above the other's across resamples. `nan` if either array has
    no finite entries."""
    a = np.asarray(values_a, dtype=float)
    a = a[np.isfinite(a)]
    b = np.asarray(values_b, dtype=float)
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    boot_a = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    boot_b = rng.choice(b, size=(n_boot, len(b)), replace=True).mean(axis=1)
    return float((boot_a > boot_b).mean())
