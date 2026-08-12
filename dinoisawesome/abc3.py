"""Facts and batch-iteration helpers for the ``data/abc3`` dataset.

abc3 has 4 part types (``LHa``/``LHb``/``RHa``/``RHb``), each with a ref (``_1``) and
query (``_2``) image. Not every class is annotated on every part type, and some classes
are the same physical object annotated two ways (a multi-instance class plus a
single-instance variant) — :data:`INSTANCE_TYPE_GROUPS` and
:func:`available_instance_groups` exist to iterate "every part x every distinct object
type actually present" without hardcoding which variant has which classes.
"""

import logging
from pathlib import Path

import numpy as np

from .annotation_utils import load_annotations

log = logging.getLogger(__name__)

PART_TYPES: list[str] = ["LHa", "LHb", "RHa", "RHb"]

# "donut foam" (multi-instance) and "donut foam single" (the same object, annotated as
# one instance) are grouped together — they're not distinct object types.
INSTANCE_TYPE_GROUPS: dict[str, list[str]] = {
    "foam": ["foam"],
    "donut foam": ["donut foam single", "donut foam"],
    "velcro": ["velcro"],
    "white clips": ["white clips"],
}


def load_instance_pixel_masks(
    annotation_stem: str | Path, class_filter: list[str] | None
) -> list[np.ndarray]:
    """Individual instance masks matching *class_filter* for *annotation_stem* (possibly empty).

    Kept separate per instance (not unioned) so callers can treat each annotated instance
    as its own unit if needed. *class_filter* of ``None`` matches every class.
    """
    try:
        anns = load_annotations(annotation_stem)
    except FileNotFoundError:
        log.warning("Annotation not found at %s", annotation_stem)
        return []
    instances = [a for a in anns if class_filter is None or a["class"] in class_filter]
    if not instances:
        log.warning("No instances matched classes %s at %s", class_filter, annotation_stem)
        return []
    return [a["mask"] for a in instances]


def load_instance_pixel_mask(
    annotation_stem: str | Path, class_filter: list[str] | None
) -> np.ndarray | None:
    """Union of instance masks matching *class_filter* for *annotation_stem*, or None."""
    masks = load_instance_pixel_masks(annotation_stem, class_filter)
    if not masks:
        return None
    return np.asarray(np.stack(masks).any(axis=0))


def classes_present(annotation_stem: str | Path) -> set[str]:
    """Distinct class names annotated at *annotation_stem*."""
    try:
        anns = load_annotations(annotation_stem)
    except FileNotFoundError:
        return set()
    return {a["class"] for a in anns}


def available_instance_groups(
    annotation_stem: str | Path,
    groups: dict[str, list[str]] = INSTANCE_TYPE_GROUPS,
) -> list[str]:
    """Group names from *groups* whose class list intersects what's annotated at *annotation_stem*.

    Lets a batch loop ask "which instance-type groups actually apply to this image"
    instead of hardcoding which part-type variant has which classes.
    """
    present = classes_present(annotation_stem)
    return [name for name, classes in groups.items() if present & set(classes)]
