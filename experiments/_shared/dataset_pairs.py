"""Ref/query pair catalog for ``data/abc5``, used by every ``fundamental/*.py`` script
instead of a single hardcoded ``data/abc3`` + ``REF_NUMBER=1``/``QUERY_NUMBER=2`` pair.

abc5 merges abc3 (2 images/part type) and abc4 (6 images/part type) into one physical
dataset, renumbered ``1..8`` per part type — see ``scripts/build_abc5_dataset.py``. Images
were *captured* in four natural pairs per part type — (1,2) is abc3's original shot-twice
pair, (3,4)/(5,6)/(7,8) are abc4's three — but pairing ref/query along those same capture
sessions every time is a dataset-of-origin confound, the same one
``_shared.pooled_gallery_cv`` was already fixed for on the 5-3 side: (1,2) is always an
abc3-only pair and (3,4)/(5,6)/(7,8) are always abc4-only, so every "1-1" score pools
ref/query images from the *same* capture session and never mixes abc3 with abc4. That both
makes matching artificially easy (same session = near-duplicate lighting/setup, not a real
test of generalizing to an unseen instance) and ties results to which dataset an instance
happened to come from rather than to the method being tested.

``REF_QUERY_PAIRS`` instead shuffles each part type's 8 images and pairs them up freely —
pairs can and do cross abc3/abc4 — drawn fresh from OS entropy on every import, not from a
stored seed, so re-running a script doesn't keep replaying the same fixed pairing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dinoisawesome.abc3 import PART_TYPES

ALL_IMAGE_NUMBERS: list[int] = [1, 2, 3, 4, 5, 6, 7, 8]


@dataclass(frozen=True)
class RefQueryPair:
    part_type: str
    ref: int
    query: int
    dataset: str = "abc5"

    @property
    def unit(self) -> str:
        """Stable string key identifying this pair, e.g. ``LHa_3-4``."""
        return f"{self.part_type}_{self.ref}-{self.query}"


def random_ref_query_pairs(seed: int | None = None) -> list[RefQueryPair]:
    """Shuffles each part type's 8 abc5 images and groups them into 4 ref/query pairs,
    instead of the fixed (1,2)/(3,4)/(5,6)/(7,8) capture-session grouping (see module
    docstring for the dataset-of-origin bias that grouping has). Every image is used
    exactly once, as either ref or query, same as the fixed grouping — only which two
    images end up paired, and which one plays which role, is randomized.

    seed=None (default) seeds from OS entropy; pass an explicit int only for one-off
    reproduction of a specific run's pairing.
    """
    rng = np.random.default_rng(seed)
    pairs: list[RefQueryPair] = []
    for part_type in PART_TYPES:
        perm = rng.permutation(ALL_IMAGE_NUMBERS)
        for i in range(0, len(ALL_IMAGE_NUMBERS), 2):
            pairs.append(RefQueryPair(part_type, int(perm[i]), int(perm[i + 1])))
    return pairs


REF_QUERY_PAIRS: list[RefQueryPair] = random_ref_query_pairs()
