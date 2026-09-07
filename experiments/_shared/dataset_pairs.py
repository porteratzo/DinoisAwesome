"""Ref/query pair catalog for ``data/abc5``, used by every ``fundamental/*.py`` script
instead of a single hardcoded ``data/abc3`` + ``REF_NUMBER=1``/``QUERY_NUMBER=2`` pair.

abc5 merges abc3 (2 images/part type) and abc4 (6 images/part type) into one physical
dataset, renumbered ``1..8`` per part type — see ``scripts/build_abc5_dataset.py``. Images
are annotated in four natural ref/query pairs per part type: (1,2) is abc3's original
pair, (3,4)/(5,6)/(7,8) are abc4's three shot-twice pairs continuing the sequence — matching
how the images were actually captured (each pair is one part instance shot twice).
"""

from __future__ import annotations

from dataclasses import dataclass

from dinoisawesome.abc3 import PART_TYPES

ABC5_PAIRS: list[tuple[int, int]] = [(1, 2), (3, 4), (5, 6), (7, 8)]


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


REF_QUERY_PAIRS: list[RefQueryPair] = [
    RefQueryPair(pt, ref, query) for pt in PART_TYPES for ref, query in ABC5_PAIRS
]
