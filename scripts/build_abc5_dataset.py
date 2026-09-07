"""Builds ``data/abc5`` by merging ``data/abc3`` (2 images/part type) and ``data/abc4``
(6 images/part type) into one physical dataset, renumbered ``1..8`` per part type —
abc3 first, then abc4 — so every ``fundamental/*.py`` script can treat the data as one
pool instead of juggling two datasets via ``_shared/dataset_pairs.py``'s ``dataset``
field.

Renumbering per part type::

    abc3 _1, _2   -> abc5 _1, _2
    abc4 _1.._6   -> abc5 _3.._8

Each abc4 shot-twice pair keeps its pairing under the new numbers: abc4's (1,2)/(3,4)/
(5,6) become abc5's (3,4)/(5,6)/(7,8), alongside abc3's own (1,2) — so
``ABC5_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8)]`` in ``_shared/dataset_pairs.py``.

Run: ``python scripts/build_abc5_dataset.py``
"""

import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("build_abc5_dataset")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
ABC5_DIR = DATA_ROOT / "abc5"

PART_TYPES = ["LHa", "LHb", "RHa", "RHb"]

# (source dataset, source image number) for abc5 numbers 1..8, in order.
SOURCE_MAP: list[tuple[str, int]] = [("abc3", 1), ("abc3", 2)] + [("abc4", n) for n in range(1, 7)]


def merge_classes(abc3_classes: list[str], abc4_classes: list[str]) -> list[str]:
    merged = list(abc3_classes)
    for c in abc4_classes:
        if c not in merged:
            merged.append(c)
    return merged


def build_image_and_annotation(
    part_type: str, src_dataset: str, src_num: int, dst_num: int
) -> None:
    src_dir = DATA_ROOT / src_dataset
    src_stem = f"{part_type}_{src_num}"
    dst_stem = f"{part_type}_{dst_num}"

    shutil.copy2(src_dir / f"{src_stem}.jpg", ABC5_DIR / f"{dst_stem}.jpg")

    npy_src = src_dir / "annotations" / f"{src_stem}.npy"
    json_src = src_dir / "annotations" / f"{src_stem}.json"
    shutil.copy2(npy_src, ABC5_DIR / "annotations" / f"{dst_stem}.npy")

    with open(json_src) as f:
        metadata = json.load(f)
    for entry in metadata:
        entry["image_path"] = f"{dst_stem}.jpg"
    with open(ABC5_DIR / "annotations" / f"{dst_stem}.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    (ABC5_DIR / "annotations").mkdir(parents=True, exist_ok=True)

    for part_type in PART_TYPES:
        for dst_num, (src_dataset, src_num) in enumerate(SOURCE_MAP, start=1):
            build_image_and_annotation(part_type, src_dataset, src_num, dst_num)
        log.info("%s: merged %d images", part_type, len(SOURCE_MAP))

    with open(DATA_ROOT / "abc3" / "classes.json") as f:
        abc3_classes = json.load(f)["classes"]
    with open(DATA_ROOT / "abc4" / "classes.json") as f:
        abc4_classes = json.load(f)["classes"]
    merged_classes = merge_classes(abc3_classes, abc4_classes)
    with open(ABC5_DIR / "classes.json", "w") as f:
        json.dump({"classes": merged_classes}, f, indent=2)
    log.info("wrote classes.json: %s", merged_classes)

    total = len(PART_TYPES) * len(SOURCE_MAP)
    log.info(
        "abc5 build complete: %d part types x %d images = %d images",
        len(PART_TYPES),
        len(SOURCE_MAP),
        total,
    )


if __name__ == "__main__":
    main()
