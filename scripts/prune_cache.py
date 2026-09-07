"""Delete cache entries unused for N days across DinoisAwesome's DINO feature caches.

Three cache trees, three different "last used" signals (none of them support reads
without recording one, so a fresh checkout never gets pruned before its caches have
even been touched once):

    encoding    ``$DINO_ENCODING_CACHE_DIR`` (default ``data/encoding_cache``) — the
                shared ``EncoderWithCache`` disk cache. Per-image ``.npz`` files;
                mtime is bumped on every cache hit (see encoder_cache.py), so a stale
                file really means "not read in N days". Pruned at file granularity,
                with matching rows dropped from that fingerprint's ``index.parquet``.

    anomaly     ``outputs/anomaly_detection/cache/<category>/<method>/`` — pruned as
                a whole method directory, keyed by the newest mtime among
                ``scores.parquet``/``anomaly_maps.npz`` (touched by ``has_cache()``,
                see experiments/anomaly_detection/common.py) and ``gallery/.last_used``
                (touched by ``Gallery.__init__``, see dinoisawesome/gallery.py) when
                present.

    objdet      ``outputs/object_detection/multiscale_ablation*/cache/`` — crops/ and
                blobs/ are pruned at file granularity (each touched on hit, see
                multiscale_ablation/run_experiments.py); methods/ is pruned per
                ``<pair>/`` directory, keyed by the newest mtime among its files. Only
                the main ``<method>.pkl`` is explicitly touched on hit today, so a
                directory stays "warm" as long as at least one of its files was
                recently read — conservative in the safe direction (never deletes
                something still in active use), but a directory whose only reads are
                of files that aren't independently touched (pair_meta.pkl,
                cross_scale.pkl, blobs__*.pkl) could in principle survive on an old
                mtime alone rather than a genuine recent read.

Usage:
    python scripts/prune_cache.py --days 30                 # dry run (default) — report only
    python scripts/prune_cache.py --days 30 --apply         # actually delete
    python scripts/prune_cache.py --days 14 --targets encoding
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)
log = logging.getLogger("prune_cache")

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

DEFAULT_ENCODING_CACHE_DIR = REPO_ROOT / "data" / "encoding_cache"
ANOMALY_DETECTION_CACHE_ROOT = REPO_ROOT / "outputs" / "anomaly_detection" / "cache"
# Every multiscale_ablation* output tree (current + resolution/model sweep variants —
# see multiscale_ablation/run_experiments.py's --resolution/--model flags and
# resolution_ablation/'s own separate output dirs) shares this same cache layout.
OBJECT_DETECTION_CACHE_ROOTS = [
    p / "cache"
    for p in (REPO_ROOT / "outputs" / "object_detection").glob("multiscale_ablation*")
    if (p / "cache").is_dir()
]


@dataclass
class PruneStats:
    scanned: int = 0
    stale: int = 0
    bytes_freed: int = 0

    def __iadd__(self, other: PruneStats) -> PruneStats:
        self.scanned += other.scanned
        self.stale += other.stale
        self.bytes_freed += other.bytes_freed
        return self


def _is_stale(path: Path, cutoff: float) -> bool:
    return path.stat().st_mtime < cutoff


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _delete_file(path: Path, apply: bool) -> int:
    size = path.stat().st_size
    if apply:
        path.unlink()
    return size


def _delete_dir(path: Path, apply: bool) -> int:
    size = _dir_size(path)
    if apply:
        shutil.rmtree(path)
    return size


def prune_encoding_cache(root: Path, cutoff: float, apply: bool) -> PruneStats:
    stats = PruneStats()
    if not root.is_dir():
        log.info("encoding: %s does not exist, skipping", root)
        return stats

    for fp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        emb_dir = fp_dir / "embeddings"
        if not emb_dir.is_dir():
            continue
        index_path = fp_dir / "index.parquet"
        index = pd.read_parquet(index_path) if index_path.exists() else None
        stale_keys: list[str] = []

        for npz_path in tqdm(sorted(emb_dir.glob("*.npz")), desc=f"encoding/{fp_dir.name}"):
            stats.scanned += 1
            if _is_stale(npz_path, cutoff):
                stats.stale += 1
                stats.bytes_freed += _delete_file(npz_path, apply)
                stale_keys.append(npz_path.stem)

        if apply and stale_keys and index is not None:
            index = index[~index["key"].isin(stale_keys)]
            index.to_parquet(index_path, index=False)

    return stats


def prune_anomaly_detection_cache(root: Path, cutoff: float, apply: bool) -> PruneStats:
    stats = PruneStats()
    if not root.is_dir():
        log.info("anomaly: %s does not exist, skipping", root)
        return stats

    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for method_dir in tqdm(
            sorted(p for p in category_dir.iterdir() if p.is_dir()),
            desc=f"anomaly/{category_dir.name}",
        ):
            stats.scanned += 1
            marker = method_dir / "gallery" / ".last_used"
            candidates = [
                f
                for f in (method_dir / "scores.parquet", method_dir / "anomaly_maps.npz", marker)
                if f.exists()
            ]
            last_used = max(
                (f.stat().st_mtime for f in candidates), default=method_dir.stat().st_mtime
            )
            if last_used < cutoff:
                stats.stale += 1
                stats.bytes_freed += _delete_dir(method_dir, apply)

    return stats


def prune_object_detection_cache(root: Path, cutoff: float, apply: bool) -> PruneStats:
    stats = PruneStats()

    for tier in ("crops", "blobs"):
        tier_dir = root / tier
        if not tier_dir.is_dir():
            continue
        for cache_file in tqdm(
            sorted(tier_dir.rglob("*.pt")), desc=f"objdet/{root.parent.name}/{tier}"
        ):
            stats.scanned += 1
            if _is_stale(cache_file, cutoff):
                stats.stale += 1
                stats.bytes_freed += _delete_file(cache_file, apply)

    methods_dir = root / "methods"
    if methods_dir.is_dir():
        for config_dir in sorted(p for p in methods_dir.iterdir() if p.is_dir()):
            for pair_dir in tqdm(
                sorted(p for p in config_dir.iterdir() if p.is_dir()),
                desc=f"objdet/{root.parent.name}/methods/{config_dir.name}",
            ):
                stats.scanned += 1
                files = [f for f in pair_dir.iterdir() if f.is_file()]
                last_used = max((f.stat().st_mtime for f in files), default=0.0)
                if last_used < cutoff:
                    stats.stale += 1
                    stats.bytes_freed += _delete_dir(pair_dir, apply)

    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--days", type=int, default=30, help="Evict entries unused for this many days (default: 30)"
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["encoding", "anomaly", "objdet"],
        choices=["encoding", "anomaly", "objdet"],
        help="Which cache trees to prune (default: all three)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete stale entries (default: dry run, report only).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cutoff = time.time() - args.days * 86400
    mode = "APPLY" if args.apply else "DRY RUN"
    log.info("mode=%s days=%d cutoff=%s", mode, args.days, time.ctime(cutoff))

    total = PruneStats()

    if "encoding" in args.targets:
        encoding_dir = Path(
            os.environ.get("DINO_ENCODING_CACHE_DIR", str(DEFAULT_ENCODING_CACHE_DIR))
        )
        s = prune_encoding_cache(encoding_dir, cutoff, args.apply)
        log.info("encoding: %d/%d stale, %.1f GB freed", s.stale, s.scanned, s.bytes_freed / 1e9)
        total += s

    if "anomaly" in args.targets:
        s = prune_anomaly_detection_cache(ANOMALY_DETECTION_CACHE_ROOT, cutoff, args.apply)
        log.info("anomaly: %d/%d stale, %.1f GB freed", s.stale, s.scanned, s.bytes_freed / 1e9)
        total += s

    if "objdet" in args.targets:
        for root in OBJECT_DETECTION_CACHE_ROOTS:
            s = prune_object_detection_cache(root, cutoff, args.apply)
            log.info(
                "objdet(%s): %d/%d stale, %.1f GB freed",
                root.parent.name,
                s.stale,
                s.scanned,
                s.bytes_freed / 1e9,
            )
            total += s

    log.info(
        "TOTAL: %d/%d entries %s, %.1f GB %s",
        total.stale,
        total.scanned,
        "deleted" if args.apply else "would be deleted",
        total.bytes_freed / 1e9,
        "freed" if args.apply else "would be freed",
    )
    if not args.apply:
        log.info("Dry run only — pass --apply to actually delete.")


if __name__ == "__main__":
    main()
