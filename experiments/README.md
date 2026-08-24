# experiments/

Exploratory scripts probing DINOv2/v3 patch-embedding behavior and building
training-free detection/anomaly/alignment pipelines on top of them. This is a lab
notebook, not a library — see [`../dinoisawesome/`](../dinoisawesome/) for the
reusable code these scripts exercise.

Setup: `pip install -e "..[dev]"` from the repo root, and a repo-root `.env` if you
use one (every script calls `load_dotenv()` on it automatically).

## Two ways these scripts run — read this first

- **`anomaly_detection/run_experiments.py` and `analyze_results.py`** are real CLI
  scripts: `argparse`, a `if __name__ == "__main__"` entry point, no `plt.show()`.
  Run them with `python <script>.py [flags]` like any normal script.
- **Everything else** is written as a **Jupyter-style cell script** (`# %%` /
  `# %% [markdown]` cell markers, no argparse, ends in `plt.show()` calls). These are
  meant to be run **cell-by-cell in VS Code's Python Interactive Window or a Jupyter
  session** (via `jupytext`), not invoked from the shell — each cell corresponds to
  one numbered "experiment" with markdown commentary above it, and the point is to
  step through and inspect plots inline. You *can* `python <script>.py` them top to
  bottom, but you'll only get whatever gets `savefig`'d (not all of them save — see
  below) or a blocking `plt.show()` window per figure.

Data dependencies you'll hit:
- **`data/abc3`** — the current annotated part-inspection dataset (4 part types:
  `LHa`/`LHb`/`RHa`/`RHb`). Used by most scripts below.
- **`data/abc2`** — a legacy predecessor to `abc3`. Only `keypoint_matching.py` still
  points at it (see below).
- **`data/mvtec_ad`** — the public MVTec AD benchmark, not included in the repo; download
  it separately before running `anomaly_detection/`.
- **`data/custom_slim`** — a SAM3-annotated slim dataset with `*_mask_good.npz` GT
  masks, used only by `object_detection/eval_custom_slim.py`.

---

## `anomaly_detection/` — PatchCore vs. AnomalyDINO benchmark (actively maintained)

A two-stage CLI pipeline comparing 4 anomaly-detection methods
(`patchcore`, `anomalydino_v2`, `anomalydino_v3`, `anomalydino_v3_smooth9`) across 6
MVTec AD categories. Run stage 1 before stage 2.

- **`common.py`** — shared paths, `CATEGORIES`/`METHODS` constants, cache-path
  helpers, `ScoreRecord` dataclass. Not run directly; imported by the other three.
- **`methods.py`** — the `AnomalyMethod` interface and its 4 implementations. Not run
  directly; imported by `run_experiments.py`.
- **`run_experiments.py`** — **Stage 1.** Fits + scores every (category, method) pair
  and caches results.
  ```bash
  python run_experiments.py                                  # full run
  python run_experiments.py --categories bottle --limit 10   # smoke test
  python run_experiments.py --methods patchcore anomalydino_v3
  python run_experiments.py --force                          # overwrite existing cache
  ```
  Requires `data/mvtec_ad/` locally (not present in this checkout as of writing —
  download it first). Writes to `outputs/anomaly_detection/cache/<category>/<method>/`.
- **`analyze_results.py`** — **Stage 2.** Reads stage-1 cache, computes
  AUROC/AUPR/F1Max/AUPRO via `anomalib.metrics`, writes
  `outputs/anomaly_detection/results/{metrics.csv,summary.md,figures/*.png}`.
  ```bash
  python analyze_results.py
  python analyze_results.py --categories bottle carpet
  ```

## `fundamental/` — DINOv3's basic representational properties (actively maintained)

A three-part progressive series, **meant to be read/run in that order** — each
script's docstring explicitly builds on the previous one's finding. All three are
cell-based, need `data/abc3`, and save figures under
`outputs/fundamental/<script_name>/`.

1. **`scale_crop_similarity.py`** — as you crop progressively tighter around one
   instance, how far does its patch embedding move in feature space?
2. **`augmentation_sensitivity.py`** — holding the crop fixed, how far do six
   perturbation families (rotation, gamma/illumination, color jitter, Gaussian blur,
   Gaussian noise, JPEG compression) move the embedding?
3. **`augmented_prototype_oracle_iou.py`** — if you perturb the *exemplar* crop
   before pooling it into a matching prototype, does scoring a different query image
   with that prototype localize the object better or worse (oracle IoU)? The biggest
   file of the three (1333 lines) — it reuses the same augmentation families as #2 and
   the same oracle-IoU search `object_detection/multiscale_crop_ablation.py` uses.

## `object_detection/` — training-free instance detection (actively maintained)

- **`instance_detection.py`** — the baseline pipeline: one exemplar + its instance
  mask → masked patch tokens → cosine-similarity density map → max-pool NMS instance
  centers. Cell-based, `data/abc3`, **interactive only — no figures are saved**, so
  you need the Interactive Window/Jupyter to see anything.
- **`density_map_methods.py`** — compares 5 ways to turn the exemplar into a density
  map (mean / k-means / k-NN memory bank / PCA-whitening / MLP classifier), each run
  through a full threshold → HDBSCAN clustering → IoU-match evaluation across 4
  exemplar/query pairs (`LHa`/`LHb`/`RHa`/`RHb`). Cell-based, interactive only.
- **`multiscale_crop_ablation.py`** — the most complete script in this directory
  (2730 lines). Builds `global`/`mid`/`close` prototypes from one instance, ablates
  every scale combination (single- and multi-scale max-similarity) against GT IoU,
  then runs a cross-scale similarity study (does a global prototype still score a
  close-up crop well, and vice versa?). Mirrors and extends
  `../../scripts/multiscale_detection.py`. Cell-based, interactive only.
- **`eval_custom_slim.py`** — **runs differently from its siblings.** It imports
  directly from `../../scripts/eval_sam_dino.py` via `sys.path.insert(0, repo_root /
  "scripts")`, so that import only resolves if `scripts/eval_sam_dino.py` still
  exists at the repo root (it does, as of writing) — this is a real cross-directory
  dependency, not a self-contained experiment. Walks through SAM3 (text-prompted)
  proposing candidate masks, DINO ranking them against an exemplar, and GT-IoU
  scoring. Needs `data/custom_slim/` with `*_mask_good.npz` masks. Cell-based, saves
  figures under `results/custom_slim_nb/`.

## Top-level scripts

- **`eval_coarse_to_fine_alignment.py`** — DINOv3 + ECC coarse-to-fine image
  alignment, 6 numbered experiments (synthetic-view augmentation → MNN keypoint
  consensus → coarse homography → ECC refinement → quality metrics). Cell-based,
  saves to `outputs/coarse_to_fine/`. **Not fully wired up**: the file's own header
  flags `KeypointMatcherHead.match()` as a soft-argmax placeholder stub to swap for a
  trained head before trusting results on real data.
- **`high_res_tiling.py`** — the largest script here (1748 lines, 9 numbered
  experiments): 2×2 tiling at 2× resolution vs. single-pass baseline, seam-quality
  and overlap/blend sweeps, cross-scale self-similarity, and downstream instance
  detection + throughput comparisons. Cell-based, `data/abc3`, saves to
  `results/high_res_tiling/`.
- **`patch_clustering.py`** — **oldest script in the directory (last touched
  2026-06-21), predates the Aug-12 restructuring that added everything above.**
  Minimal demo: extract patch tokens from a single image (defaults to auto-downloading
  a sample cat photo if `data/tiger.jpeg` isn't present — the only script here with no
  real data dependency), compare KMeans/DBSCAN/Agglomerative/Spectral clustering by
  overlaying assignments. Cell-based, interactive only, nothing saved. Conceptually
  superseded by `object_detection/density_map_methods.py`'s much more rigorous method
  comparison — keep this one around only as a quick intro/smoke-test example, not as
  a reference implementation.
- **`keypoint_matching.py`** — **likely stale.** Also predates the Aug-12
  restructuring (last touched 2026-06-24) and is the only script in this directory
  still pointing at `data/abc2` instead of the current `data/abc3`. Registers named
  keypoints on a reference image (`Gallery` + `KeypointHead`), finds them in query
  images by nearest-patch cosine similarity, and estimates a RANSAC homography.
  Cell-based, interactive only, nothing saved. If you need keypoint-based alignment
  today, prefer `eval_coarse_to_fine_alignment.py`'s Exp 3/4 (MNN consensus +
  homography) — it's the newer, actively-maintained approach to the same problem.
