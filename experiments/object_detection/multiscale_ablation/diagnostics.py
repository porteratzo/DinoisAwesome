"""One-stage DBSCAN clustering deep-dive for a single (pair, method) case.

Ported from ``experiments/object_detection/multiscale_crop_ablation.py``'s "One-stage
DBSCAN clustering deep-dive" section (that file's lines 2534-2896; see that file's own
module docstring / git history for the original ablation this was written against). It
was the one piece of that now-superseded notebook script never carried over when the
rest of the pipeline was ported into this package (see ``engine.py``/``run_experiments.
py``/``visualize_results.py``'s own "Ported from multiscale_crop_ablation.py" notes) —
genuinely unique diagnostic tooling for debugging DBSCAN clustering quality on one
(part_type, ref_number, query_number) pair + one scoring method, not duplicated by
anything in ``visualize_results.py``'s summary/detailed figures.

Diagnoses the *one-stage* pipeline specifically: a method's full-query raw score map,
thresholded and DBSCAN-clustered directly on the coarse full-image patch grid (no ROI/
crop step) — the harder clustering case, and the one worth this level of scrutiny:

1. **k-distance plot** — sort every foreground point's distance to its
   ``scoring_cfg.pred_dbscan_min_samples``-th nearest neighbor and look for the "knee",
   compared against the actually-configured eps.
2. **Merge/split diagnosis** — a predicted cluster overlapping >1 GT instance is a
   merge; a GT instance overlapped by >1 predicted cluster is a split, by 0 is a miss.
   Logged per-cluster and as a summary, plus a 3-panel figure: predicted clusters, GT
   instances, and the raw (pre-size-filter) sklearn DBSCAN labels.
3. **Nearest-patch gap analysis** — the minimum patch-grid distance between every pair
   of GT instances (is eps already bigger than genuine inter-instance spacing, baking
   in merge risk regardless of score quality?) and, per GT instance, the largest
   internal nearest-neighbor gap among its own predicted foreground patches (would a
   perfect score map still fragment it, because the map drops patches mid-object?).
4. **eps / min_samples sweep** — holding the threshold fixed, recompute DBSCAN at a
   grid of eps and min_samples values and plot P/R/F1/mIoU/cluster-count against each,
   to see whether the configured operating point sits in a stable region or right on a
   merge/split cliff.

Reuses ``run_experiments.py``'s on-disk caches (``pair_meta.pkl`` + ``<method>.pkl``
under ``cache/methods/<crop_hash>__<scoring_hash>/<pair>/``) whenever they already exist
for the requested pair/method, exactly as ``visualize_results.py`` prefers reading
caches over recomputing — an encoder is only built, and ``run_experiments.run_pair``
only called, as a fallback when nothing is cached yet (or ``--force``).

Figures are saved to disk (never ``plt.show()``'d) — this is a batch/CLI script, not a
notebook, matching every other script in this package.

Usage:
    python diagnostics.py --part-type RHa --method mid
    python diagnostics.py --part-type RHa --instance-type foam --method global
    python diagnostics.py --part-type RHa --method mid --force   # recompute, ignore cache
    python diagnostics.py --part-type RHa --method mid --resolution 768  # match a non-
                                                                          # default run
"""

# Logging — must be before torch import
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("diagnostics")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import (  # noqa: E402
    DATA_DIR,
    DEFAULT_CROP_CONFIG,
    DEFAULT_SCORING_CONFIG,
    FIGURES_ROOT,
    QUERY_NUMBER,
    REF_NUMBER,
    CropConfig,
    PairKey,
    ScoringConfig,
    all_pairs,
    method_cache_dir,
    method_cache_path,
)
from engine import match_and_score, patch_radius_to_eps
from methods import all_method_names
from PIL import Image
from run_experiments import run_pair
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN as SKDBSCAN
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from dinoisawesome import DinoEncoder, EncoderWithCache

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _shared.clustering import dbscan_clusters as _shared_dbscan_clusters  # noqa: E402

DINO_WEIGHTS_DIR: str | None = os.environ.get("DINO_WEIGHTS_DIR")
DINO_ENCODING_CACHE_DIR: str | None = os.environ.get("DINO_ENCODING_CACHE_DIR")

CMAP = plt.get_cmap("tab10")

# Fixed diagnostic sweep points (not tunable config, see diagnostic #4 above) — ported
# 1:1 from multiscale_crop_ablation.py's EPS_SWEEP_PATCHES / MIN_SAMPLES_SWEEP.
EPS_SWEEP_PATCHES = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
MIN_SAMPLES_SWEEP = [1, 2, 3, 4, 6]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--part-type", required=True, help="Focal part type")
    parser.add_argument(
        "--instance-type",
        default=None,
        help=(
            "Focal instance-type group. Only needed to disambiguate when --part-type "
            "has more than one available instance-type group (see common.all_pairs)."
        ),
    )
    parser.add_argument("--ref-number", type=int, default=REF_NUMBER)
    parser.add_argument("--query-number", type=int, default=QUERY_NUMBER)
    parser.add_argument(
        "--method",
        required=True,
        choices=all_method_names(
            list(DEFAULT_CROP_CONFIG.scales),
            DEFAULT_SCORING_CONFIG.kmeans_ks,
            include_kmeans=True,
            include_classifiers=True,
        ),
        help="Which method's cached one-stage (full-query) result to diagnose",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=(
            "Override CropConfig.img_size to match a run_experiments.py run made with "
            f"--resolution (default: {DEFAULT_CROP_CONFIG.img_size})"
        ),
    )
    parser.add_argument(
        "--model",
        choices=["small", "base", "large", "giant"],
        default=None,
        help=(
            "Override CropConfig.dino_size to match a run_experiments.py run made with "
            f"--model (default: {DEFAULT_CROP_CONFIG.dino_size!r})"
        ),
    )
    parser.add_argument(
        "--bg-enrich-crops",
        type=int,
        default=None,
        help=(
            "Match a run_experiments.py run made with --bg-enrich-crops (default: "
            f"{DEFAULT_CROP_CONFIG.bg_enrich_crops_per_scale}, i.e. off)."
        ),
    )
    parser.add_argument(
        "--fg-clean",
        choices=["raw", "step1", "step2_cls", "step2_center", "step3"],
        default=None,
        help=(
            "Match a run_experiments.py run made with --fg-clean (default: "
            f"{DEFAULT_SCORING_CONFIG.fg_clean_stage!r}, i.e. off)."
        ),
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help=(
            "Match a run_experiments.py run made with --offset (default: "
            f"{DEFAULT_SCORING_CONFIG.threshold_offset}, i.e. off)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing cache, recompute this pair/method via run_experiments.run_pair",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Where figures/CSVs are written (default: {FIGURES_ROOT / 'diagnostics'})",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pair resolution + cache read/compute
# ---------------------------------------------------------------------------


def _resolve_pair(
    part_type: str, instance_type: str | None, ref_number: int, query_number: int
) -> PairKey:
    candidates = [
        p
        for p in all_pairs(ref_number, query_number)
        if p.part_type == part_type and (instance_type is None or p.instance_type == instance_type)
    ]
    if not candidates:
        raise RuntimeError(
            f"No pair found for part_type={part_type!r} instance_type={instance_type!r} "
            f"ref_number={ref_number} query_number={query_number}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"part_type={part_type!r} matches >1 instance-type group "
            f"({[c.instance_type for c in candidates]}) — pass --instance-type to disambiguate"
        )
    return candidates[0]


def _load_or_compute(
    pair: PairKey, crop_cfg: CropConfig, scoring_cfg: ScoringConfig, method: str, force: bool
) -> tuple[dict, dict]:
    """(pair_meta, method_result) for *pair*/*method*, reusing run_experiments.py's cache
    when it already exists and only calling ``run_pair`` (which needs an encoder) as a
    fallback — mirrors ``visualize_results.py``'s cache-first, never-recompute-by-default
    convention, except this script *can* fall back to computing since (unlike
    visualize_results.py) it isn't meant to run encoder-free.
    """
    meta_path = method_cache_dir(pair, crop_cfg, scoring_cfg) / "pair_meta.pkl"
    method_path = method_cache_path(pair, crop_cfg, scoring_cfg, method)
    if not force and meta_path.exists() and method_path.exists():
        log.info("[%s/%s] cache hit — reusing %s and %s", pair.slug, method, meta_path, method_path)
        return pickle.loads(meta_path.read_bytes()), pickle.loads(method_path.read_bytes())

    log.info(
        "[%s/%s] cache miss (or --force) — building an encoder and running run_pair",
        pair.slug,
        method,
    )
    encoder = DinoEncoder(
        version=crop_cfg.dino_version,
        size=crop_cfg.dino_size,
        img_size=crop_cfg.img_size,
        weights_dir=DINO_WEIGHTS_DIR,
        amp=True,
    )
    encoder = EncoderWithCache(encoder, cache_dir=DINO_ENCODING_CACHE_DIR)
    run_pair(pair, encoder, encoder.patch_size, crop_cfg, scoring_cfg, [method], force)
    if not meta_path.exists() or not method_path.exists():
        raise RuntimeError(
            f"[{pair.slug}/{method}] run_pair completed but no cache was written — this pair/"
            "method combination may not be buildable (e.g. a scale dropped for this pair)"
        )
    return pickle.loads(meta_path.read_bytes()), pickle.loads(method_path.read_bytes())


# ---------------------------------------------------------------------------
# Diagnostic 1 — k-distance plot
# ---------------------------------------------------------------------------


def plot_kdistance(
    pair: PairKey, method: str, fg_coords: np.ndarray, k: int, eps: float, out_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    if len(fg_coords) > k:
        nn = NearestNeighbors(n_neighbors=k).fit(fg_coords)
        dists, _ = nn.kneighbors(fg_coords)
        kth_dist_sorted = np.sort(dists[:, -1])
        ax.plot(kth_dist_sorted, color="steelblue")
        ax.axhline(eps, color="red", linestyle="--", label=f"eps={eps:.3f}")
        ax.set_xlabel("foreground points, sorted by k-NN distance")
        ax.set_ylabel(f"distance to {k}-th nearest neighbor")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "too few foreground points", ha="center", va="center")
    ax.set_title(
        f"k-distance plot (k={k}) — {method} | part_type={pair.part_type}\n"
        "look for the 'knee' — eps well past it means far-apart points get bridged anyway",
        fontsize=10,
    )
    out_path = out_dir / f"kdistance__{pair.case_slug}__{_method_slug(method)}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# Diagnostic 2 — merge/split diagnosis + predicted/GT/noise visualization
# ---------------------------------------------------------------------------


def _merge_split_diagnosis(
    pair: PairKey, method: str, pred_clusters: list[dict], gt_clusters: list[dict]
) -> tuple[list[list[int]], list[int]]:
    """Logs the per-cluster and summary merge/split diagnosis; returns
    (pred_overlap, gt_frag_counts) for the visualization below."""
    pred_overlap = [
        [int((cl["mask"] & gt["mask"]).sum()) for gt in gt_clusters] for cl in pred_clusters
    ]
    gt_frag_counts = [0] * len(gt_clusters)
    n_merged = 0
    for cl, overlaps in tqdm(
        list(zip(pred_clusters, pred_overlap)), desc=f"[{pair.slug}/{method}] merge/split diagnosis"
    ):
        hit_gts = [j for j, n in enumerate(overlaps) if n > 0]
        for j in hit_gts:
            gt_frag_counts[j] += 1
        is_merge = len(hit_gts) > 1
        n_merged += int(is_merge)
        log.info(
            "  pred cluster size=%d score=%.3f mean_patch=%.3f rejected=%s -> overlaps GT %s%s",
            int(cl["mask"].sum()),
            cl["score"],
            cl["mean_patch_score"],
            cl["rejected"],
            hit_gts if hit_gts else "none",
            "  <-- MERGE (spans >1 GT instance)" if is_merge else "",
        )

    split_gts = [j for j, n in enumerate(gt_frag_counts) if n > 1]
    missed_gts = [j for j, n in enumerate(gt_frag_counts) if n == 0]
    log.info(
        "[%s] merge/split summary: %d/%d pred clusters merge >=2 GT instances | "
        "%d/%d GT instances split across >=2 pred clusters %s | "
        "%d/%d GT instances have zero overlapping pred cluster %s",
        pair.slug,
        n_merged,
        len(pred_clusters),
        len(split_gts),
        len(gt_clusters),
        split_gts,
        len(missed_gts),
        len(gt_clusters),
        missed_gts,
    )
    return pred_overlap, gt_frag_counts


def plot_cluster_assignment(
    pair: PairKey,
    method: str,
    disp_q: np.ndarray,
    pred_clusters: list[dict],
    gt_clusters: list[dict],
    pred_overlap: list[list[int]],
    gt_frag_counts: list[int],
    fg_coords: np.ndarray,
    xs_fg: np.ndarray,
    ys_fg: np.ndarray,
    eps: float,
    min_samples: int,
    out_dir: Path,
) -> None:
    sk_labels = (
        SKDBSCAN(eps=eps, min_samples=min_samples).fit_predict(fg_coords)
        if len(fg_coords)
        else np.array([])
    )
    n_noise = int((sk_labels == -1).sum()) if len(sk_labels) else 0
    n_raw_clusters = len(set(sk_labels.tolist()) - {-1}) if len(sk_labels) else 0

    fig, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)

    axes[0].imshow(disp_q)
    for i, cl in enumerate(pred_clusters):
        ys_c, xs_c = np.where(cl["mask"])
        n_hit = sum(1 for n in pred_overlap[i] if n > 0)
        marker = "x" if cl["rejected"] else "o"
        edge = "red" if n_hit > 1 else "none"
        axes[0].scatter(
            xs_c,
            ys_c,
            s=16,
            color=CMAP(i % 10),
            marker=marker,
            edgecolors=edge,
            linewidths=1.2 if edge != "none" else 0,
        )
    axes[0].set_title(
        f"predicted DBSCAN clusters ({len(pred_clusters)})\n"
        "x=mean-patch-rejected, red ring=spans >1 GT instance",
        fontsize=9,
    )
    axes[0].axis("off")

    axes[1].imshow(disp_q)
    for j, gt in enumerate(gt_clusters):
        ys_c, xs_c = np.where(gt["mask"])
        edge = "red" if gt_frag_counts[j] > 1 else ("orange" if gt_frag_counts[j] == 0 else "none")
        axes[1].scatter(
            xs_c,
            ys_c,
            s=16,
            color=CMAP(j % 10),
            edgecolors=edge,
            linewidths=1.2 if edge != "none" else 0,
        )
    axes[1].set_title(
        f"GT-DBSCAN instances ({len(gt_clusters)})\n"
        "red ring=split across >1 pred cluster, orange ring=zero pred overlap",
        fontsize=9,
    )
    axes[1].axis("off")

    axes[2].imshow(disp_q)
    noise_sel = sk_labels == -1
    if noise_sel.any():
        axes[2].scatter(
            xs_fg[noise_sel],
            ys_fg[noise_sel],
            s=20,
            color="black",
            marker="x",
            label="DBSCAN noise",
        )
    core_sel = ~noise_sel
    if core_sel.any():
        axes[2].scatter(
            xs_fg[core_sel], ys_fg[core_sel], s=10, color="steelblue", alpha=0.5, label="clustered"
        )
    axes[2].legend(fontsize=8)
    axes[2].set_title(
        f"raw DBSCAN labels, before size filter ({n_raw_clusters} clusters)\n"
        f"{n_noise}/{len(xs_fg)} foreground patches are noise (too isolated to join any cluster)",
        fontsize=9,
    )
    axes[2].axis("off")

    plt.suptitle(
        f"One-stage clustering diagnosis — {method} | part_type={pair.part_type}", fontsize=12
    )
    out_path = out_dir / f"cluster_assignment__{pair.case_slug}__{_method_slug(method)}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# Diagnostic 3 — nearest-patch gap analysis
# ---------------------------------------------------------------------------


def plot_instance_pair_gaps(
    pair: PairKey, method: str, gt_clusters: list[dict], eps: float, out_dir: Path
) -> None:
    if len(gt_clusters) < 2:
        log.info("[%s] fewer than 2 GT instances — skipping inter-instance gap analysis", pair.slug)
        return

    gt_coords_list = []
    for gt in gt_clusters:
        ys_g, xs_g = np.where(gt["mask"])
        gt_coords_list.append(np.stack([xs_g, ys_g], axis=1).astype(float))

    cross_gaps = [
        float(cdist(gt_coords_list[i], gt_coords_list[j]).min())
        for i in range(len(gt_coords_list))
        for j in range(i + 1, len(gt_coords_list))
    ]

    n_at_risk = sum(g <= eps for g in cross_gaps)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.hist(cross_gaps, bins=min(15, max(3, len(cross_gaps))), color="tab:purple", alpha=0.7)
    ax.axvline(eps, color="red", linestyle="--", label=f"eps={eps:.3f}")
    ax.set_xlabel("min patch-grid distance between the two GT instances")
    ax.legend(fontsize=8)
    ax.set_title(
        f"nearest-patch gap between every GT instance pair ({len(cross_gaps)} pairs)\n"
        f"{n_at_risk}/{len(cross_gaps)} pairs closer than eps -> merge risk on a perfect mask too",
        fontsize=9,
    )
    out_path = out_dir / f"instance_pair_gaps__{pair.case_slug}__{_method_slug(method)}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)
    log.info(
        "[%s] GT instance-pair gaps: %s (eps=%.3f, %d/%d pairs at/under eps)",
        pair.slug,
        [round(g, 2) for g in cross_gaps],
        eps,
        n_at_risk,
        len(cross_gaps),
    )


def log_within_instance_gaps(
    pair: PairKey, gt_clusters: list[dict], binary: np.ndarray, eps: float
) -> None:
    for j, gt in enumerate(tqdm(gt_clusters, desc=f"[{pair.slug}] within-instance gaps")):
        inside = binary & gt["mask"]
        ys_in, xs_in = np.where(inside)
        if len(xs_in) < 2:
            log.info(
                "  GT instance %d: only %d foreground patch(es) inside its own mask — too "
                "sparse to assess (likely a recall miss, not a clustering split)",
                j,
                len(xs_in),
            )
            continue
        coords_in = np.stack([xs_in, ys_in], axis=1).astype(float)
        nn_in = NearestNeighbors(n_neighbors=2).fit(coords_in)
        dists_in, _ = nn_in.kneighbors(coords_in)
        max_gap = float(dists_in[:, 1].max())
        log.info(
            "  GT instance %d: %d/%d patches thresholded as foreground, max internal NN gap="
            "%.3f (%s eps=%.3f)",
            j,
            len(xs_in),
            int(gt["mask"].sum()),
            max_gap,
            "EXCEEDS" if max_gap > eps else "within",
            eps,
        )


# ---------------------------------------------------------------------------
# Diagnostic 4 — eps / min_samples sweep
# ---------------------------------------------------------------------------


def _sweep_metrics(
    xs_fg: np.ndarray,
    ys_fg: np.ndarray,
    q_h: int,
    q_w: int,
    raw: np.ndarray,
    eps: float,
    min_samples: int,
    min_cs: int,
    min_points_floor: int,
    gt_clusters: list[dict],
    iou_thr: float,
) -> dict:
    if len(xs_fg) < max(min_points_floor, min_cs):
        clusters: list[dict] = []
    else:
        clusters = _shared_dbscan_clusters(xs_fg, ys_fg, q_h, q_w, raw, eps, min_samples, min_cs)
    m = match_and_score(clusters, gt_clusters, iou_thr)
    return {"eps": eps, "min_samples": min_samples, "n_clusters": len(clusters), **m}


def run_param_sweep(
    pair: PairKey,
    method: str,
    xs_fg: np.ndarray,
    ys_fg: np.ndarray,
    q_h: int,
    q_w: int,
    raw: np.ndarray,
    eps: float,
    min_samples: int,
    min_cs: int,
    min_points_floor: int,
    gt_clusters: list[dict],
    iou_thr: float,
    out_dir: Path,
) -> None:
    eps_sweep_df = pd.DataFrame(
        [
            _sweep_metrics(
                xs_fg,
                ys_fg,
                q_h,
                q_w,
                raw,
                patch_radius_to_eps(r),
                min_samples,
                min_cs,
                min_points_floor,
                gt_clusters,
                iou_thr,
            )
            for r in EPS_SWEEP_PATCHES
        ]
    )
    log.info(
        "[%s] eps sweep (min_samples=%d fixed):\n%s",
        pair.slug,
        min_samples,
        eps_sweep_df.to_string(index=False),
    )

    ms_sweep_df = pd.DataFrame(
        [
            _sweep_metrics(
                xs_fg, ys_fg, q_h, q_w, raw, eps, ms, min_cs, min_points_floor, gt_clusters, iou_thr
            )
            for ms in MIN_SAMPLES_SWEEP
        ]
    )
    log.info(
        "[%s] min_samples sweep (eps=%.3f fixed):\n%s",
        pair.slug,
        eps,
        ms_sweep_df.to_string(index=False),
    )

    method_slug = _method_slug(method)
    eps_csv_path = out_dir / f"eps_sweep__{pair.case_slug}__{method_slug}.csv"
    ms_csv_path = out_dir / f"min_samples_sweep__{pair.case_slug}__{method_slug}.csv"
    eps_sweep_df.to_csv(eps_csv_path, index=False)
    ms_sweep_df.to_csv(ms_csv_path, index=False)
    log.info("Wrote %s and %s", eps_csv_path, ms_csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, sweep_df, x_col, current_val, title in (
        (axes[0], eps_sweep_df, "eps", eps, f"eps sweep (min_samples={min_samples} fixed)"),
        (
            axes[1],
            ms_sweep_df,
            "min_samples",
            min_samples,
            f"min_samples sweep (eps={eps:.3f} fixed)",
        ),
    ):
        for col, style in (("precision", "o-"), ("recall", "o-"), ("f1", "o-"), ("mean_iou", "o-")):
            ax.plot(sweep_df[x_col], sweep_df[col], style, label=col)
        ax_n = ax.twinx()
        ax_n.plot(sweep_df[x_col], sweep_df["n_clusters"], "s:", color="gray", label="n_clusters")
        ax_n.axhline(len(gt_clusters), color="black", linestyle="--", linewidth=1, label="n_GT")
        ax.axvline(current_val, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel(x_col)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_n.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="center right")

    plt.suptitle(
        f"DBSCAN parameter sensitivity — {method} | part_type={pair.part_type}\n"
        "red dashed line = currently configured value",
        fontsize=12,
    )
    out_path = out_dir / f"param_sweep__{pair.case_slug}__{method_slug}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _method_slug(method: str) -> str:
    return method.replace("/", "-")


def run_diagnostics(
    pair: PairKey, method: str, crop_cfg: CropConfig, scoring_cfg: ScoringConfig, out_dir: Path
) -> None:
    meta = pickle.loads(
        method_cache_dir(pair, crop_cfg, scoring_cfg).joinpath("pair_meta.pkl").read_bytes()
    )
    result = pickle.loads(method_cache_path(pair, crop_cfg, scoring_cfg, method).read_bytes())

    q_h, q_w = meta["q_h"], meta["q_w"]
    gt_clusters = meta["gt_clusters"]
    min_cs = meta["min_cs"]
    raw = result["raw"]
    thr = result["threshold"]
    binary = raw > thr
    pred_clusters = result["pred_clusters"]

    eps = patch_radius_to_eps(scoring_cfg.pred_dbscan_eps_patches)
    min_samples = scoring_cfg.pred_dbscan_min_samples

    ys_fg, xs_fg = np.where(binary)
    fg_coords = np.stack([xs_fg, ys_fg], axis=1).astype(float)

    log.info(
        "[%s] one-stage deep-dive: method=%s thr=%.3f foreground_patches=%d/%d pred_clusters=%d "
        "(kept=%d) gt_clusters=%d eps=%.3f min_samples=%d min_cs=%d",
        pair.slug,
        method,
        thr,
        len(xs_fg),
        binary.size,
        len(pred_clusters),
        sum(not c["rejected"] for c in pred_clusters),
        len(gt_clusters),
        eps,
        min_samples,
        min_cs,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Diagnostic 1
    plot_kdistance(pair, method, fg_coords, min_samples, eps, out_dir)

    # Diagnostic 2
    pred_overlap, gt_frag_counts = _merge_split_diagnosis(pair, method, pred_clusters, gt_clusters)
    query_img = Image.open(DATA_DIR / f"{pair.part_type}_{pair.query_number}.jpg").convert("RGB")
    disp_q = np.array(query_img.resize((q_w, q_h)))
    plot_cluster_assignment(
        pair,
        method,
        disp_q,
        pred_clusters,
        gt_clusters,
        pred_overlap,
        gt_frag_counts,
        fg_coords,
        xs_fg,
        ys_fg,
        eps,
        min_samples,
        out_dir,
    )

    # Diagnostic 3
    plot_instance_pair_gaps(pair, method, gt_clusters, eps, out_dir)
    log_within_instance_gaps(pair, gt_clusters, binary, eps)

    # Diagnostic 4
    run_param_sweep(
        pair,
        method,
        xs_fg,
        ys_fg,
        q_h,
        q_w,
        raw,
        eps,
        min_samples,
        min_cs,
        scoring_cfg.min_points_floor,
        gt_clusters,
        scoring_cfg.iou_match_threshold,
        out_dir,
    )


def main() -> None:
    args = _parse_args()
    crop_cfg = DEFAULT_CROP_CONFIG
    if args.resolution is not None:
        crop_cfg = dataclasses.replace(crop_cfg, img_size=args.resolution)
    if args.model is not None:
        crop_cfg = dataclasses.replace(crop_cfg, dino_size=args.model, layer_idx=None)
    if args.bg_enrich_crops is not None:
        crop_cfg = dataclasses.replace(crop_cfg, bg_enrich_crops_per_scale=args.bg_enrich_crops)
    scoring_cfg = DEFAULT_SCORING_CONFIG
    if args.fg_clean is not None:
        scoring_cfg = dataclasses.replace(scoring_cfg, fg_clean_stage=args.fg_clean)
    if args.offset:
        scoring_cfg = dataclasses.replace(scoring_cfg, threshold_offset=args.offset)

    pair = _resolve_pair(args.part_type, args.instance_type, args.ref_number, args.query_number)
    out_dir = args.output_dir if args.output_dir is not None else FIGURES_ROOT / "diagnostics"

    _load_or_compute(pair, crop_cfg, scoring_cfg, args.method, args.force)
    run_diagnostics(pair, args.method, crop_cfg, scoring_cfg, out_dir)


if __name__ == "__main__":
    main()
