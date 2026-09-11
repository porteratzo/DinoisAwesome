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
- **`data/abc5`** — merges `data/abc3` (2 images/part type) with `data/abc4` (6
  images/part type) into 8 images/part type, so the cross-validated `fundamental/`
  scripts (below) have enough images per part type to hold out a real eval fold.
  Built by `scripts/build_abc5_dataset.py`.
- **`data/mvtec_ad`** — the public MVTec AD benchmark, not included in the repo; download
  it separately before running `anomaly_detection/`.
- **`data/custom_slim`** — a SAM3-annotated slim dataset with `*_mask_good.npz` GT
  masks, used only by `object_detection/eval_custom_slim.py`.

### Cross-validation methodology used across this directory

Most scripts here score exactly **one fixed (reference, query) pair** per part
type — no CV, just a documented example. A smaller set of scripts, tagged
**"cross-validated"** in the table below, instead use `_shared/pooled_gallery_cv.py`'s
**5-3 pooled-gallery CV**: 5 of `data/abc5`'s 8 images per part type are pooled into
one training gallery, scored against the 3 held-out images, repeated for 5 folds with
a **fresh random shuffle of image order per fold** (not a fixed seed) — deliberately,
because `training_set_size_ablation.py`'s own docstring documents a real methodological
trap: a fixed image order that always puts `abc3`-origin images in training and
`abc4`-origin images in eval is a dataset-of-origin confound that produced a spurious
effect in that script's first version, one only a randomized-fold check caught.

---

## At a glance

One row per runnable experiment (helper modules like `common.py`/`methods.py` that
are only ever imported, not run, are omitted). "CV / eval" says whether a script
cross-validates or just scores one fixed pair — see above.

| Script | What it tests | Key assumption / caveat | CV / eval | Status |
|---|---|---|---|---|
| `anomaly_detection/run_experiments.py` | Fits + scores 4 anomaly methods (PatchCore, AnomalyDINO v2/v3/v3-smooth9) across 6 MVTec categories | Requires `data/mvtec_ad/` downloaded locally | MVTec's own fixed train/test split, no CV | active |
| `anomaly_detection/analyze_results.py` | Computes AUROC/AUPR/F1Max/AUPRO from stage-1 cache | Depends on `run_experiments.py` having run first | aggregation only | active |
| `anomaly_detection/layer_ablation.py` | Which DINOv3 block gives best AUROC/AUPRO, and is it stable across categories? | Every method in `methods.py` hardcodes "last block" | one MVTec split per block, no CV | active |
| `anomaly_detection/resolution_size_ablation.py` | Sweeps `img_size` x backbone size for `anomalydino_v3`/`dinov3_proto_*` | 256px/small is hardcoded everywhere else in the directory | full re-encode per point, no CV | active |
| `anomaly_detection/train_size_ablation.py` | Learning curve of AUROC vs. normal-image count | One fixed random shuffle's *prefix* per size (nested subsets, not independent resamples) | no CV (single shuffle) | active |
| `anomaly_detection/localization_analysis.py` | Oracle-IoU / Otsu-IoU bounds + per-bucket (defect size/contrast) AUROC breakdown | Reads `run_experiments.py`'s existing cache, no re-fitting | none — reads existing cache | active |
| `fundamental/scale_crop_similarity.py` | How far a patch embedding moves in feature space as you crop progressively tighter | Single-instance case study | fixed pair, qualitative | active |
| `fundamental/augmentation_sensitivity.py` | How 6 perturbation families (rotation, gamma, color jitter, blur, noise, JPEG) move the embedding, crop held fixed | — | fixed pair, qualitative | active |
| `fundamental/augmented_prototype_oracle_iou_knn_fgbg.py` | Does perturbing the exemplar crop before pooling into a prototype change oracle-IoU localization on a different query? | Reuses aug. families of `augmentation_sensitivity.py` + the oracle-IoU search from `object_detection/multiscale_ablation/engine.py` | fixed pair, oracle IoU | active — largest of the first three (1333 lines) |
| `fundamental/training_set_size_ablation.py` | Does pooling N=1..5 training images into the gallery improve oracle IoU on held-out eval images? | An earlier fixed-image-order version showed a spurious effect from a dataset-of-origin confound — see own docstring | **cross-validated**: 5-fold, fresh shuffle/fold | active |
| `fundamental/resolution_ablation.py` | DINOv3 input resolution (256-1536px) effect on oracle IoU | Checked against `object_detection/resolution_ablation/`'s own fixed-pair sweep at the same resolutions | **cross-validated**: 1-1 and 5-3 regimes, 2-fold CV each, to separate real trend from fold noise | active |
| `fundamental/feature_transform_oracle_iou.py` | Does reshaping raw patch geometry (centering / ZCA / PCA / LDA / Mahalanobis) before matching improve fg/bg separability? | Galleries held fixed (close+mid+global); only the transform is swept, fit per combo | fixed pair; `pooled_gallery_cv` used for a 5-3 addition too | active |
| `fundamental/head_layer_ablation.py` | Does a *fitted* classification head (linear probe / linear SVM) beat plain cosine-centroid matching — and this directory's own best training-free method, contrastive kNN — for per-patch fg/bg segmentation, and does the answer depend on which DINOv3-base block(s) feed it? | The 3 fitted heads report hard-decision IoU/F1/accuracy; the `knn_fgbg` baseline reports oracle IoU (threshold tuned against each eval image's own GT) — not directly comparable operating points, see script docstring | **cross-validated**: 5-3 pooled CV only (no fixed-pair mode) | active |
| `fundamental/noisy_fgbg_cleaning.py` | Does discarding ambiguous exemplar patches (spatial filter -> DINO-attention check -> HDBSCAN+kNN consensus) improve oracle IoU on a different query? | The 0.3 mask-overlap threshold used elsewhere creates noisy boundary patches; this asks if removing them helps | fixed pair; `pooled_gallery_cv` used for a 5-3 addition too | active |
| `fundamental/scale_composition_oracle_iou.py` | Finer scale steps (7, not 3) + multi-scale fg/bg composition via a curated combo table (not the full 2^n power set) | `mid` = exact midpoint of `close`/`global` is itself an arbitrary convention every sibling script inherits | fixed pair, oracle IoU, dataset-wide mean +/- std | active |
| `fundamental/scale_composition_adaptive_oracle.py` | Per-instance adaptive-scale oracle headroom over the fixed-best-scale baseline; does optimal scale correlate with instance size? | Reuses `scale_composition_oracle_iou.py`'s Parts 1-4 verbatim | post-hoc analysis of that script's per-instance IoUs, no new scoring | active |
| `fundamental/scale_composition_bg_ablation.py` | Isolates whether a multi-scale *background* actually helps (every sibling always used "all scales" for bg, untested) | `single_proto` is bg-invariant by construction, so this is really a `fg-bg-knn`-only ablation | same curated composition families as `scale_composition_oracle_iou.py`, applied to the bg side | active |
| `fundamental/scale_composition_max_pool.py` | Max-pool vs. concatenate-pool when combining scales into one bank | Background held fixed at all-scales throughout | reuses `scale_composition_oracle_iou.py`'s scoring, pooling mode is the only variable | active |
| `fundamental/scale_composition_query_matching.py` | Does a ref/query scale mismatch explain why `close` underperforms — bounded 7x7 grid, not a combinatorial search | **IoU is not on the same scale as `scale_composition_oracle_iou.py`'s** — the query is also GT-cropped here, making the task easier; only the diagonal-vs-off-diagonal pattern is comparable | 7x7 grid heatmap, own dataset only | active |
| `pipeline/adaptive_method_selection.py` | Replaces `object_detection/multiscale_ablation/`'s manual "read the table, pick a config" step with adaptive selection logic (scale -> method -> feature transform -> single/two-stage -> denoising -> optional augmentation) | Locks in `feature_transform_oracle_iou.py`'s `bg_zca` finding as a default; augmentation search is off by default (previously found not to stack) | same single-stage val signal used at each decision step, no independent CV | active |
| `object_detection/instance_detection.py` | Baseline pipeline: 1 exemplar + its instance mask -> masked patch tokens -> cosine-similarity density map -> max-pool NMS instance centers | — | none — interactive only, no figures saved | active |
| `object_detection/multiscale_ablation/` (package) | Builds global/mid/close prototypes from one instance, ablates every scale combination vs. GT IoU, then a cross-scale similarity study | Mirrors/extends `../../scripts/multiscale_detection.py` | fixed pairs, P/R/F1/mIoU, no CV | active |
| `object_detection/resolution_ablation/` (package) | Resolution x model-size ablation, reusing `multiscale_ablation`'s pipeline unchanged over the full cross product | `layer_idx` is architecture-dependent (12 blocks small/base, 24 large); `run_experiments.py` asks the built encoder for its real depth, but `visualize_results.py` (deliberately torch-free) hardcodes the same mapping separately and must be kept in sync by hand | fixed pairs (same as `multiscale_ablation`), no CV | active |
| `object_detection/eval_custom_slim.py` | SAM3 text-prompted mask proposals -> DINO ranking vs. exemplar -> GT-IoU scoring | Imports `../../scripts/eval_sam_dino.py` directly via a `sys.path` hack — a real cross-directory dependency, only resolves if that file still exists at the repo root | single dataset pass, no CV | active, fragile dependency |
| `eval_coarse_to_fine_alignment.py` | DINOv3 + ECC coarse-to-fine alignment, 6 numbered experiments (synthetic-view augmentation -> MNN keypoint consensus -> coarse homography -> ECC refinement -> quality metrics) | File's own header flags `KeypointMatcherHead.match()` as a soft-argmax placeholder stub | qualitative, not cross-validated | **not fully wired up** — own docstring says so |
| `high_res_tiling.py` | 2x2 tiling at 2x resolution vs. single-pass baseline: seam quality, overlap/blend sweeps, cross-scale self-similarity, downstream detection + throughput | — | qualitative sweeps, no CV | active — largest script (1748 lines) |
| `patch_clustering.py` | Minimal demo: extract patch tokens from one image, compare KMeans/DBSCAN/Agglomerative/Spectral clustering | Only script here with no real data dependency (auto-downloads a sample cat photo if `data/tiger.jpeg` is absent) | single-image demo, no CV | oldest script (last touched 2026-06-21); superseded by `object_detection/multiscale_ablation/`'s much more rigorous method comparison — keep only as a smoke-test example |
| `keypoint_matching.py` | Named-keypoint registration (`Gallery` + `KeypointHead`) + RANSAC homography | Still points at legacy `data/abc2`, not `data/abc3` | fixed pair, qualitative | **likely stale** — predates the Aug-12 restructuring; prefer `eval_coarse_to_fine_alignment.py`'s Exp 3/4 for keypoint-based alignment today |

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

Four ablation/analysis scripts extend this benchmark with the cross-validated-ablation
rigor `fundamental/` established (resolution/size/layer/training-set-size sweeps, oracle
bounds) but this harness never applied to anomaly detection specifically. Each writes to its
own `outputs/anomaly_detection/results/<name>/` and, where it fits new normal images, its own
synthetic-category-namespaced Gallery cache under `outputs/anomaly_detection/cache/` (e.g.
`bottle__ts10`, `bottle__r512_small`) so it never collides with `run_experiments.py`'s own
per-category cache.

- **`layer_ablation.py`** — which DINOv3 transformer block gives the best AUROC/AUPRO, and is
  it the same block across categories? `AnomalyHead`/`PrototypeAnomalyHead` both take a
  `block_idx` (every method in `methods.py` hardcodes "last block" and never sweeps it). Builds
  one Gallery per category with every block stored, so training images are encoded once
  regardless of how many blocks are swept; query images are still re-encoded once per block per
  test image, so keep `--limit` modest for a first pass.
  ```bash
  python layer_ablation.py --categories bottle carpet --limit 20
  python layer_ablation.py --method dinov3_proto --size base --stride 2
  ```
- **`resolution_size_ablation.py`** — sweeps DINOv3 `img_size` x backbone `size` for
  `anomalydino_v3`/`dinov3_proto_*` (hardcoded to 256px/small everywhere else in this
  directory). Every point re-encodes the full train set from scratch (unlike the layer
  ablation, there's no shared encoding to reuse across resolutions/sizes) and is wrapped in its
  own CUDA-OOM guard; `base`/`large` and >512px are opt-in.
  ```bash
  python resolution_size_ablation.py --categories bottle carpet --resolutions 256 512
  python resolution_size_ablation.py --sizes small base --methods anomalydino_v3
  ```
- **`train_size_ablation.py`** — learning curve of AUROC vs. normal-image count, for any
  method in `common.ALL_METHODS` (`build_method(...).fit()` is reused unmodified). Subsamples
  one fixed random shuffle's *prefix* per category (not an independent resample per size) so
  smaller sizes are proper subsets of larger ones.
  ```bash
  python train_size_ablation.py --categories bottle carpet --limit 40
  python train_size_ablation.py --methods patchcore anomalydino_v3 --train-sizes 5 10 25 all
  ```
- **`localization_analysis.py`** — reads `run_experiments.py`'s existing cache (no
  re-fitting); adds the oracle-IoU / Otsu-IoU bounds `_shared/thresholding.py` already
  provides but this directory never used, plus an aggregate error breakdown by defect-size and
  -contrast tercile (image AUROC per bucket) in place of `analyze_results.py`'s 3-image
  best/worst eyeball check. Run `run_experiments.py` first for whatever pairs you want analyzed.
  ```bash
  python localization_analysis.py
  python localization_analysis.py --categories bottle carpet --methods patchcore anomalydino_v3
  ```

## `fundamental/` — DINOv3's basic representational properties (actively maintained)

A three-part progressive series, **meant to be read/run in that order** — each
script's docstring explicitly builds on the previous one's finding. All three are
cell-based, need `data/abc3`, and save figures under
`outputs/fundamental/<script_name>/`.

**Running the whole suite from one cfg:** every script below still works exactly as
described here — cell-by-cell in an Interactive Window, hardcoded parameters — but
`scripts/run_fundamental_suite.py` can also drive some or all of them from a single
YAML config (`experiments/fundamental/suite_config.example.yaml` is a starting point),
one subprocess per script. Each run gets its own
`outputs/fundamental_runs/<timestamp>_<name>/` directory (never overwrites a previous
run), with a copy of the exact cfg used, a per-script log, and a `manifest.json`
summary. See that script's own docstring/`--help` for the cfg format (shared
`defaults:` + per-script override sections, keyed by lowercase snake_case names of
each script's own UPPERCASE constants) and `experiments/_shared/run_config.py` for how
a script picks the overrides up.

1. **`scale_crop_similarity.py`** — as you crop progressively tighter around one
   instance, how far does its patch embedding move in feature space?
2. **`augmentation_sensitivity.py`** — holding the crop fixed, how far do six
   perturbation families (rotation, gamma/illumination, color jitter, Gaussian blur,
   Gaussian noise, JPEG compression) move the embedding?
3. **`augmented_prototype_oracle_iou_knn_fgbg.py`** — if you perturb the *exemplar*
   crop before pooling it into a matching prototype, does scoring a different query
   image with that prototype localize the object better or worse (oracle IoU)? The
   biggest file of the three (1333 lines) — it reuses the same augmentation families
   as #2 and the same oracle-IoU search `object_detection/multiscale_ablation/engine.py`
   uses.

Two later additions in the same directory depart from the fixed-pair paradigm above by
cross-validating instead (data `abc5`, save under `outputs/fundamental_abc5/<script_name>/`):

- **`training_set_size_ablation.py`** — does pooling more training images into the
  gallery (N=1..5) improve oracle IoU on held-out eval images? 5-fold CV, fresh random
  shuffle per fold — its own docstring explains why this replaced an earlier
  uncross-validated version that showed a spurious effect.
- **`resolution_ablation.py`** — applies that same script's 1-1-vs-5-3 two-endpoint CV
  check to DINOv3 input resolution instead of training-set size: at every resolution in
  `RESOLUTION_SWEEP` (256/512/768/1024/1536px), runs both the 1-train/1-eval and
  5-train/3-eval regimes (2-fold CV each), to tell a real resolution trend from
  fold-to-fold noise and check it against `object_detection/resolution_ablation/`'s
  own fixed-pair (uncross-validated) resolution sweep.
- **`debias_ablation.py`** — applies the same 1-1-vs-5-3 CV check to `DinoEncoder`'s
  `debias=True` positional-debiasing flag (every other script in the repo passes it
  unconditionally, never ablated against `debias=False`): does projecting out the
  INSID3-derived positional subspace actually improve oracle IoU, at both endpoints and
  for both `single_proto`/`knn_fgbg` scoring? Also reports a paired per-sample IoU delta
  (matched on the same fold/pool/eval-image), a more sensitive check than comparing the
  two arms' independent means.

Two more additions probe the fg/bg gallery itself rather than the crop/resolution
sweep — same fixed-pair paradigm as scripts 1-3, with an optional `pooled_gallery_cv`
5-3 addition each:

- **`feature_transform_oracle_iou.py`** — every sibling script changes *what* goes
  into the fg/bg galleries; this one holds the galleries fixed (close+mid+global) and
  instead sweeps *how the raw embedding geometry is reshaped* before matching:
  mean-centering, ZCA whitening, PCA truncation, a supervised LDA direction, and a
  Mahalanobis-distance variant, each fit per combo from that combo's own pooled fg/bg
  patches. `lda` and `mahalanobis` deliberately skip the final L2-normalize every other
  pipeline uses — the docstring explains why (a 2-class LDA projection is already a
  1-D scalar; Mahalanobis needs the whitened residual's magnitude, not just direction).
- **`noisy_fgbg_cleaning.py`** — the standard `MASK_PATCH_THRESHOLD=0.3` bool
  threshold used to build every fg/bg gallery in this directory leaves boundary
  patches that are part background, part object — noise in whichever gallery they
  land in. This asks whether removing that noise with three cheap, unsupervised
  techniques (spatial filter on mask coverage; a DINO-attention cross-check against
  either a close-crop CLS token or a masked-core-pixel prototype; HDBSCAN + kNN
  consensus voting pooled per instance-type group) improves localization on a
  *different* query image.
- **`head_layer_ablation.py`** — every scoring method elsewhere in this directory is
  training-free (cosine similarity or contrastive kNN against a gallery). This fits three
  classification heads (nearest fg/bg centroid, a logistic-regression linear probe, a linear
  SVM) directly on labeled patch tokens (the same `MASK_PATCH_THRESHOLD` fg/bg convention,
  read as a per-patch binary label), alongside a fourth `knn_fgbg` entry that's this
  directory's own best-performing training-free method (contrastive kNN against the pooled
  fg/bg patch banks, unchanged) included as the baseline the fitted heads should beat — and
  sweeps which DINOv3-base block(s) feed all four: four named blocks spanning the depth
  (`early`=2, `mid`=5, `late`=9, `last`=11) plus every non-empty concatenation combination of
  them (15 combos total). 5-3 pooled CV only — abc5's 8 images/part type contribute thousands
  of labeled patches per fold even though the image count is tiny, but a single fixed split
  would still risk reporting one lucky/unlucky shuffle as if it were stable. The 3 fitted
  heads report hard-decision IoU/F1/accuracy (no threshold to tune, since they emit discrete
  labels); `knn_fgbg` instead reports oracle IoU (threshold tuned against each eval image's
  own GT, this directory's usual convention for continuous-score methods) — an upper bound
  that legitimately sees eval labels the other three never do, so it isn't a directly
  comparable deployment-time operating point, only a reference ceiling.

The newest set of additions generalizes the fixed 3-point `close`/`mid`/`global` crop
scale used everywhere above into a finer, N-step sweep and a curated (not exhaustive)
composition search — all five are self-contained, no cross-script imports, and reuse
`scale_composition_oracle_iou.py`'s Parts 1-4 (combo discovery, crop building, encoder,
per-scale fg/bg token banks) where noted:

- **`scale_composition_oracle_iou.py`** — generalizes the 3-point scale sweep into
  `N_SCALE_STEPS + 1` (7) evenly-spaced crop scales and asks two questions: how does
  oracle IoU change step by step, and does *combining* several scale steps' foreground
  patches into one bank beat any single scale alone? Composition is evaluated over a
  curated table of ~4n+3 combos (single scale, prefix-from-global, suffix-from-close,
  the classic 3-point baseline, anchored-inward/outward), not the full 2^7-1 = 127
  subsets.
- **`scale_composition_adaptive_oracle.py`** — two follow-ups the fixed-scale result
  above can't answer from its saved CSVs alone: how much headroom is there if you pick
  the best scale *per instance* instead of one dataset-wide fixed scale, and does the
  optimal scale correlate with instance size (a hypothesis that a larger instance needs
  less "zoom")?
- **`scale_composition_bg_ablation.py`** — the fixed-scale script above always pooled
  *every* scale step into the background side; this isolates whether a multi-scale
  background helps at all by growing bg through the same composition families while fg
  is held to one scale at a time.
- **`scale_composition_max_pool.py`** — the fixed-scale script's composition never beat
  a single best scale, but it only tested one combine mode (concatenate patches, then
  score once). This tries **max**-pooling instead: score each member scale separately,
  take the per-query-patch maximum — does composition's apparent failure survive a
  different way of combining scales?
- **`scale_composition_query_matching.py`** — every composition script above only crops
  the *reference* image; the query is always scored at full resolution, leaving a
  confound: is a tight exemplar crop worse, or is it penalized for a scale mismatch
  against an always-full-resolution query? GT-crops the query through the same 7-step
  sweep and scores the full 7x7 (ref scale, query scale) grid. **The resulting IoU
  numbers are on a different scale than the other `scale_composition_*.py` scripts'**
  (a GT-cropped query is an easier task) — only the diagonal-vs-off-diagonal pattern
  within this script's own grid is meaningful.

## `pipeline/` — turning fundamental findings into an actual decision procedure (actively maintained)

- **`adaptive_method_selection.py`** — moved out of `fundamental/` because, unlike its
  former siblings, it isn't probing a basic representational property — it's a decision
  pipeline that replaces `object_detection/multiscale_ablation/`'s manual "read the
  P/R/F1/mIoU table, pick a config" step with adaptive selection logic making the same
  decisions itself, in order: scale -> method -> feature-space transform (locks in
  `feature_transform_oracle_iou.py`'s `bg_zca` finding, the single biggest lever found
  across `fundamental/`) -> single-vs-two-stage -> denoising -> optional augmentation
  (off by default; composing every augmentation family was already found not to stack,
  only to cost more).

## `object_detection/` — training-free instance detection (actively maintained)

- **`instance_detection.py`** — the baseline pipeline: one exemplar + its instance
  mask → masked patch tokens → cosine-similarity density map → max-pool NMS instance
  centers. Cell-based, `data/abc3`, **interactive only — no figures are saved**, so
  you need the Interactive Window/Jupyter to see anything.
- **`multiscale_ablation/`** — a proper package (`common.py` configs + cache paths,
  `engine.py` domain logic, `methods.py` method registry, `run_experiments.py` argparse
  CLI + per-pair driver, `visualize_results.py`, `diagnostics.py` for the one-stage
  DBSCAN clustering deep-dive). Builds `global`/`mid`/`close` prototypes from one
  instance, ablates every scale combination (single- and multi-scale max-similarity)
  against GT IoU, then runs a cross-scale similarity study (does a global prototype
  still score a close-up crop well, and vice versa?). Mirrors and extends
  `../../scripts/multiscale_detection.py`. Batch/CLI, results cached and figures saved
  to disk, no cross-validation — fixed pairs, same as `anomaly_detection/run_experiments.py`.
- **`resolution_ablation/`** — resolution + model-size ablation reusing
  `multiscale_ablation`'s pipeline unchanged over the full `(size, resolution)` cross
  product. Both fields are already part of the crop-cache hash key, so every combo
  gets its own cache namespace under the same cache tree automatically. `layer_idx`
  (which transformer block to read) is architecture-dependent (12 blocks for
  small/base, 24 for large): `run_experiments.py` asks the just-built encoder for its
  real depth rather than hardcoding a table, but `visualize_results.py` (deliberately
  torch-free, so it never loads an encoder) hardcodes the same small/base/large mapping
  separately as `LAYER_IDX_BY_SIZE` — the two must be kept in sync by hand if a new
  size is ever added.
  ```bash
  python run_experiments.py                                   # 3 sizes x 5 res x every pair
  python run_experiments.py --sizes large --resolutions 512 1024
  python run_experiments.py --part-types RHa --limit-pairs 1   # smoke test
  ```
- **`eval_custom_slim.py`** — **runs differently from its siblings.** It imports
  directly from `../../scripts/eval_sam_dino.py` via `sys.path.insert(0, repo_root /
  "scripts")`, so that import only resolves if `scripts/eval_sam_dino.py` still
  exists at the repo root (it does, as of writing) — this is a real cross-directory
  dependency, not a self-contained experiment. Walks through SAM3 (text-prompted)
  proposing candidate masks, DINO ranking them against an exemplar, and GT-IoU
  scoring. Needs `data/custom_slim/` with `*_mask_good.npz` masks. Cell-based, saves
  figures under `results/custom_slim_nb/`.

Note: an earlier `density_map_methods.py` (comparing 5 ways to build a density map —
mean / k-means / k-NN memory bank / PCA-whitening / MLP classifier) has since been
**removed from the repo**; `multiscale_ablation/` is the current, actively maintained
home for method-comparison work in this directory.

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
  superseded by `object_detection/multiscale_ablation/`'s much more rigorous method
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
