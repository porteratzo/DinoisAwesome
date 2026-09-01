"""Shared identity/keying helpers for scripts that iterate abc3 "combos" (one annotated
instance = one part_type x instance-type group x class x instance_id), used across the
fundamental/ oracle-IoU scripts.
"""

from __future__ import annotations


def combo_key(d: dict) -> tuple[str, str, str, int]:
    """(part_type, instance_type group, class, instance_id) — a combo's stable identity."""
    return (d["part_type"], d["group"], d["class"], d["instance_id"])
