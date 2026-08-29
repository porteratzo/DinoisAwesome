"""Feature-space transform fit/apply helpers: mean-centering, ZCA whitening, PCA
truncation, LDA projection, and a Euclidean/Mahalanobis-contrastive kNN score — the
linear-algebra primitives behind
``experiments/fundamental/feature_transform_oracle_iou.py``.

Every ``fit_*`` function takes RAW (non-L2-normalised) patch tokens and only needs to run
once per (combo, source); the corresponding apply/transform step is cheap and meant to be
repeated across an epsilon or k sweep without refitting (see ``zca_matrix``/``pca_truncate``,
which both reuse one ``fit_cov_eigh`` result). All tensors are handled in float32 regardless
of the encoder's own dtype (bf16 under ``amp=True``) — the 1024x1024 covariance eigh this
module centers on needs more precision than bf16 offers, and mixed-dtype matmuls between a
bf16 gallery and a float32 transform matrix would otherwise fail.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def fit_mean(x: torch.Tensor) -> torch.Tensor:
    """(1, C) mean of *x*'s raw patch tokens."""
    return x.float().mean(dim=0, keepdim=True)


def fit_cov_eigh(x: torch.Tensor, mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigendecomposition of *x*'s centered covariance: ``(eigvecs (C, C), eigvals (C,))``,
    ascending eigenvalue order (``torch.linalg.eigh``'s own convention — the top-variance
    directions are the last columns, see ``pca_truncate``). Meant to be computed once per
    (combo, source) and reused across an epsilon/k sweep.
    """
    centered = x.float() - mu.float()
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    return eigvecs, eigvals


def zca_matrix(eigvecs: torch.Tensor, eigvals: torch.Tensor, eps: float) -> torch.Tensor:
    """ZCA whitening matrix ``W = V (Lambda + eps*I)^-1/2 V^T`` from a ``fit_cov_eigh``
    result. Per-combo pooled patch counts are typically well below C=1024, so most
    eigenvalues are ~0 (numerically, sometimes slightly negative) — ``eps`` is what keeps
    those directions from exploding under the inverse sqrt, not an optional nicety.
    """
    inv_sqrt = torch.clamp(eigvals, min=0.0).add(eps).pow(-0.5)
    return (eigvecs * inv_sqrt.unsqueeze(0)) @ eigvecs.T


def apply_affine(x: torch.Tensor, mu: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    """``(x - mu)``, optionally left-multiplied by a transform matrix *w* (e.g.
    ``zca_matrix``'s output, or a truncated eigenvector basis)."""
    centered = x.float() - mu.float()
    return centered @ w.float().T if w is not None else centered


def pca_truncate(x: torch.Tensor, mu: torch.Tensor, eigvecs: torch.Tensor, k: int) -> torch.Tensor:
    """Project onto the top-*k* principal components of a ``fit_cov_eigh`` result — eigh
    returns ascending eigenvalues, so the highest-variance directions are the last ``k``
    columns of *eigvecs*."""
    top_k = eigvecs[:, -k:]
    return apply_affine(x, mu, top_k.T)


def fit_lda_direction(fg: torch.Tensor, bg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Shrinkage LDA direction (unit vector) separating *fg* (label 1) from *bg* (label 0),
    plus the pooled mean used to center before projecting. Per-combo pooled patch counts (a
    few hundred to ~1500) are typically below C=1024, so the raw within-class scatter is
    singular — ``shrinkage="auto"`` (Ledoit-Wolf) is required for a well-defined solution,
    not optional regularization. A 2-class LDA has exactly one discriminant direction
    (``rank(S_B) <= 1``), hence the single returned vector rather than a projection matrix.
    """
    pooled = torch.cat([fg, bg], dim=0).float()
    x = pooled.cpu().numpy()
    y = np.concatenate([np.ones(fg.shape[0]), np.zeros(bg.shape[0])])
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    lda.fit(x, y)
    # solver="eigen" doesn't expose an `xbar_` (that's an SVD-solver-only attribute) — center
    # on the pooled mean directly instead, matching every other pipeline's `fit_mean`.
    device, dtype = fg.device, torch.float32
    direction = torch.from_numpy(lda.scalings_[:, 0]).to(device=device, dtype=dtype)
    direction = direction / direction.norm()
    mu = pooled.mean(dim=0, keepdim=True).to(device=device, dtype=dtype)
    return direction, mu


def lda_score(x: torch.Tensor, direction: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Signed 1-D projection of *x* onto a ``fit_lda_direction`` direction, meant to be used
    directly as an ``oracle_iou`` score map — no L2-norm/cosine step, since cosine
    similarity on a 1-D projection is degenerate (every point is +-1 after normalizing a
    scalar)."""
    return apply_affine(x, mu) @ direction


def knn_fgbg_score_euclidean(
    query_tokens: torch.Tensor, fg_bank: torch.Tensor, bg_bank: torch.Tensor, k: int
) -> np.ndarray:
    """Euclidean-distance analogue of ``_shared.prototype_ops.knn_fgbg_score``: mean top-k
    distance to the bg bank minus mean top-k distance to the fg bank (far from bg *and*
    close to fg => high score). Meant for un-normalised, magnitude-preserving tokens — e.g.
    the Mahalanobis pipeline's bg-whitened residuals — where cosine similarity would discard
    exactly the magnitude information this metric is built on.
    """
    query_tokens, fg_bank, bg_bank = query_tokens.float(), fg_bank.float(), bg_bank.float()
    fg_dist = torch.cdist(query_tokens, fg_bank)
    bg_dist = torch.cdist(query_tokens, bg_bank)
    fg_topk = fg_dist.topk(min(k, fg_dist.shape[1]), dim=1, largest=False).values.mean(dim=1)
    bg_topk = bg_dist.topk(min(k, bg_dist.shape[1]), dim=1, largest=False).values.mean(dim=1)
    return (bg_topk - fg_topk).cpu().numpy()
