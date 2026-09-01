"""Batched patch-token extraction and contrastive kNN scoring, shared by the
multi-scale-crop exemplar pipelines (engine.py and multiscale_crop_ablation.py).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dinoisawesome import DinoEncoder


def extract_patch_tokens_batch(
    encoder: DinoEncoder, images: list[Image.Image], layer_idx: int, debias: bool = False
) -> list[tuple[torch.Tensor, int, int]]:
    """Batched patch-token extraction: encodes all *images* in a single forward pass."""
    out = encoder(images, layers=[layer_idx], debias=debias)
    patches = out.patches[:, 0]  # (B, H, W, D)
    _, grid_h, grid_w, D = patches.shape
    return [
        (F.normalize(patches[b].reshape(grid_h * grid_w, D), p=2, dim=-1), grid_h, grid_w)
        for b in range(patches.shape[0])
    ]


def extract_patch_tokens_batch_with_cls(
    encoder: DinoEncoder, images: list[Image.Image], layer_idx: int, debias: bool = False
) -> list[tuple[torch.Tensor, torch.Tensor, int, int]]:
    """``extract_patch_tokens_batch``'s sibling, additionally returning each image's own
    L2-normalised [CLS] token (that helper discards it). CLS is *not* L2-normalised by
    ``DinoEncoder`` itself (unlike patch tokens), so it's normalised here.
    """
    out = encoder(images, layers=[layer_idx], debias=debias)
    patches = out.patches[:, 0]  # (B, H, W, D)
    cls = F.normalize(out.cls[:, 0], p=2, dim=-1)  # (B, D)
    _, grid_h, grid_w, D = patches.shape
    return [
        (
            F.normalize(patches[b].reshape(grid_h * grid_w, D), p=2, dim=-1),
            cls[b],
            grid_h,
            grid_w,
        )
        for b in range(patches.shape[0])
    ]


def knn_fgbg_score(
    query_tokens: torch.Tensor, fg_bank: torch.Tensor, bg_bank: torch.Tensor, k: int
) -> np.ndarray:
    """Per-query-patch contrastive kNN score: mean top-k fg similarity minus mean top-k bg.

    ``fg_bank``/``bg_bank`` are patch-level galleries (Nfg, C) / (Nbg, C), not collapsed
    prototypes — the kNN analogue of a single-mean contrastive fg-bg score.
    """
    fg_sim = query_tokens @ fg_bank.T
    bg_sim = query_tokens @ bg_bank.T
    fg_topk = fg_sim.topk(min(k, fg_sim.shape[1]), dim=1).values.mean(dim=1)
    bg_topk = bg_sim.topk(min(k, bg_sim.shape[1]), dim=1).values.mean(dim=1)
    return (fg_topk - bg_topk).cpu().float().numpy()


def score_heatmap(tokens: torch.Tensor, prototype: torch.Tensor, h: int, w: int) -> np.ndarray:
    """Single-prototype cosine-similarity heatmap: *prototype* vs. every query patch."""
    return (tokens @ prototype.T).reshape(h, w).cpu().float().numpy()


def knn_score_heatmap(
    tokens: torch.Tensor, fg_bank: torch.Tensor, bg_bank: torch.Tensor, k: int, h: int, w: int
) -> np.ndarray:
    """``knn_fgbg_score`` reshaped to the query's own (h, w) patch grid."""
    return knn_fgbg_score(tokens, fg_bank, bg_bank, k).reshape(h, w)
