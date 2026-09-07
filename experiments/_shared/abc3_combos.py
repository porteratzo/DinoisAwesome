"""Shared identity/keying helpers for scripts that iterate dataset "combos" (one annotated
instance = one ref/query pair x instance-type group x class x instance_id), used across the
fundamental/ oracle-IoU scripts.

A combo's identity is keyed by ``unit`` (see ``dataset_pairs.RefQueryPair.unit``, e.g.
``"LHa_3-4"``) rather than by bare ``part_type`` — with multiple ref/query pairs per
part type (abc5 has four: (1,2)/(3,4)/(5,6)/(7,8)), ``part_type`` alone is no longer
unique enough to identify which pair a combo came from.
"""

from __future__ import annotations


def combo_key(d: dict) -> tuple[str, str, str, int]:
    """(unit, instance-type group, class, instance_id) — a combo's stable identity."""
    return (d["unit"], d["group"], d["class"], d["instance_id"])
