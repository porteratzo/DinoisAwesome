"""Shared 5-3 pooled-gallery cross-validation for `fundamental/*.py` scripts.

Generalizes `_shared/dataset_pairs.py`'s single ref/query pair (a 1-train/1-eval split) into
a "5 training images pooled into one gallery, scored against 3 held-out eval images, 2-fold
cross-validated" companion mode any script can add without touching its own existing 1-1
pipeline. See `training_set_size_ablation.py` for where this pattern — and its methodological
pitfall — was first worked out: a *fixed* image order (e.g. always images 1-5 for training,
6-8 for eval) is a dataset-of-origin confound, since abc5's image 1 is abc3's original
capture and images 3-8 are abc4's (see `scripts/build_abc5_dataset.py`). Every fold here uses
a fresh random shuffle instead, for exactly that reason.

Discovery and fold/role assignment are generic across scripts (same "abc5's 8 images per
part type" data model, same annotation format); each calling script keeps its own crop
scale(s), fg/bg-split convention, encoder, and scoring logic, and only uses this module for
"which images, which role, this fold" bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dinoisawesome import load_annotations
from dinoisawesome.abc3 import INSTANCE_TYPE_GROUPS, available_instance_groups

ALL_IMAGE_NUMBERS: list[int] = [1, 2, 3, 4, 5, 6, 7, 8]
N_TRAIN_53: int = 5
N_EVAL_53: int = 3
N_FOLDS_53: int = 2

# A gallery pooled from 5 images has far more patches than a single reference image's, and
# three different downstream costs scale with pool size — but only some of them have a real
# "more data past this point doesn't add information" ceiling, so the cap is split by which
# cost is actually being bounded rather than one size for everything:
#
#   - MAX_BANK_SIZE_TRANSFORM_53 (feature_transform_oracle_iou.py): feeds ZCA/PCA/LDA/
#     Mahalanobis eigendecomposition, which has a real mathematical ceiling at DINOv3-large's
#     C=1024 feature dim — feature_transform_oracle_iou.py's own docstring notes per-combo
#     pooled counts of a few hundred to ~1500 patches are already comparable to or below that
#     rank, so more patches past ~4000 mostly cost compute (O(N*C^2) covariance) rather than
#     improve the fit.
#   - MAX_BANK_SIZE_DENOISE_53 (noisy_fgbg_cleaning.py): feeds Step 3's HDBSCAN + kNN
#     consensus, O(N^2) in pool size — a similar diminishing-returns argument (density
#     estimates saturate), weaker than the rank ceiling above but still real.
#   - MAX_BANK_SIZE_KNN_53 (augmented_prototype_oracle_iou_knn_fgbg.py, every
#     scale_composition_*.py 5-3 addition): feeds plain `knn_fgbg` scoring only — a
#     non-parametric retrieval method with no ceiling analogous to the above; more gallery
#     patches can genuinely change (usually improve) which top-k matches a query finds, so
#     capping this tightly risks quietly discarding the exact benefit "does pooling help"
#     experiments are trying to measure. The per-call cost here is only O(N) (one matmul),
#     not O(N^2) or O(N*C^2), so a much larger value is affordable — this cap exists as a
#     safety ceiling against a pathologically large pool, not as an active constraint.
MAX_BANK_SIZE_TRANSFORM_53: int = 4000
MAX_BANK_SIZE_DENOISE_53: int = 4000
MAX_BANK_SIZE_KNN_53: int = 100_000


def cap_bank_size(tokens: torch.Tensor, max_size: int, seed: int) -> torch.Tensor:
    """Randomly subsample *tokens* (N, C) down to at most *max_size* rows, deterministically
    given *seed*. A no-op when already at or below *max_size*."""
    n = tokens.shape[0]
    if n <= max_size:
        return tokens
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_size, replace=False)
    return tokens[torch.from_numpy(idx).to(tokens.device)]


@dataclass
class Instance:
    """One annotated instance, identified by which image it came from (not a ref/query
    pair) — the unit every 5-3 gallery pools over. `crops` is left for the calling script to
    populate with its own scale(s)/padding convention."""

    part_type: str
    group: str
    image_number: int
    cls: str
    instance_id: int
    mask: np.ndarray
    bg_exclude_mask: np.ndarray
    crops: dict = field(default_factory=dict)


@dataclass
class Discovery:
    instances: list[Instance]
    images: dict[tuple[str, int], Image.Image]
    gt_masks: dict[tuple[str, str, int], np.ndarray]  # (part_type, group, image_number) -> mask


def discover_all_instances(data_root: Path, dataset: str, part_types: list[str]) -> Discovery:
    """Every annotated instance across all 8 `dataset` images per part type, plus each
    image's own per-group GT mask (union of that image's instances of the group) — the same
    discovery every sibling script's Part 1 already does per ref/query pair, generalized to
    every image instead of just the two in one pair.
    """
    instances: list[Instance] = []
    images: dict[tuple[str, int], Image.Image] = {}
    gt_masks: dict[tuple[str, str, int], np.ndarray] = {}

    for part_type in part_types:
        for n in ALL_IMAGE_NUMBERS:
            stem = f"{part_type}_{n}"
            ann_path = data_root / dataset / "annotations" / stem
            groups = available_instance_groups(ann_path)
            if not groups:
                continue
            anns = load_annotations(ann_path)
            img: Image.Image | None = None
            for group in groups:
                classes = INSTANCE_TYPE_GROUPS[group]
                group_anns = [a for a in anns if a["class"] in classes]
                if not group_anns:
                    continue
                if img is None:
                    img = Image.open(data_root / dataset / f"{stem}.jpg").convert("RGB")
                    images[(part_type, n)] = img
                group_mask = np.stack([a["mask"] for a in group_anns]).any(axis=0)
                gt_masks[(part_type, group, n)] = group_mask
                for ann in group_anns:
                    instances.append(
                        Instance(
                            part_type=part_type,
                            group=group,
                            image_number=n,
                            cls=ann["class"],
                            instance_id=ann["instance_id"],
                            mask=ann["mask"],
                            bg_exclude_mask=group_mask,
                        )
                    )
    return Discovery(instances=instances, images=images, gt_masks=gt_masks)


def make_fold_role_splits(
    part_types: list[str],
    seed: int,
    n_train: int = N_TRAIN_53,
    n_eval: int = N_EVAL_53,
    n_folds: int = N_FOLDS_53,
) -> list[dict[str, tuple[set[int], list[int]]]]:
    """*n_folds* independent random role assignments, one dict per fold mapping
    `part_type -> (train_numbers_set, eval_numbers_list)`.

    Each fold freshly shuffles that part type's 8 images — folds are independent resamples
    (their eval sets can and do overlap across folds), not a non-overlapping k-fold
    partition; that's deliberate, see the module docstring.
    """
    rng = np.random.default_rng(seed)
    folds: list[dict[str, tuple[set[int], list[int]]]] = []
    for _ in range(n_folds):
        split: dict[str, tuple[set[int], list[int]]] = {}
        for part_type in part_types:
            perm = rng.permutation(ALL_IMAGE_NUMBERS)
            split[part_type] = (
                set(perm[:n_train].tolist()),
                perm[n_train : n_train + n_eval].tolist(),
            )
        folds.append(split)
    return folds
