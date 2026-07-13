---
name: dino-experiment
description: Use when starting a new DINO vision-transformer experiment in this repo, or asked to "try a new approach", "add an experiment", "prototype X with DINO", "explore/investigate patch features", or to scaffold a new notebook under notebooks/. Also use when iterating on an existing experiment notebook (adjusting parameters, re-running, and re-visualising results). Encodes this repo's setup -> run -> visualize -> tweak -> repeat workflow and the shared dinoisawesome.utils helpers, so new experiments don't reimplement mask/patch-grid/heatmap/PCA plumbing that already exists.
---

# DINO experiment workflow

This repo's experiments (`notebooks/*.ipynb`) all follow the same interactive
loop: **set up once, iterate cheaply.** The model load and data load are
expensive and go in their own cells; everything downstream is cheap to
re-run and cheap to re-visualise. Optimize the notebook structure for that
loop, not for a single top-to-bottom run.

## The loop

1. **Setup** — load the encoder, load the input image(s)/mask(s), show them
   once. This cell is slow (model weights, forward pass) — never fold
   parameter tuning into it.
2. **Experiment** — the actual computation under test (similarity matrix,
   clustering, density map, keypoint match, …), driven entirely by constants
   defined in a `# ── Parameters ──` cell above it.
3. **Visualize** — a dedicated cell (or cells) that only plots what step 2
   produced. Never recompute inside a viz cell.
4. **Tweak and repeat** — change one constant in the Parameters cell, re-run
   from the Experiment cell down (not from Setup). Prefer sweep cells (loop
   over a small list of parameter values, one subplot per value) over manually
   re-running the same cell repeatedly — see `notebooks/instance_detection.ipynb`
   ("Parameter Sweep" sections) for the pattern.

Keep cells small and single-purpose so re-running a downstream cell after a
tweak doesn't require re-running an expensive one.

## Before writing a helper, check dinoisawesome.utils first

`dinoisawesome/utils/` exists specifically so experiments stop reimplementing
the same plumbing. Before writing any of the following inline, import it:

| Need | Use |
|---|---|
| Load a `.npy`/`.npz`/image instance mask, union channels | `dinoisawesome.utils.load_instance_mask` |
| Resize a pixel mask onto the patch grid | `dinoisawesome.utils.pixel_mask_to_patch_mask` |
| Bounding box of a boolean mask | `dinoisawesome.utils.mask_bbox_rc` / `mask_bbox_xywh` |
| IoU between two boolean masks | `dinoisawesome.utils.mask_iou` |
| Normalize + nearest-upsample a `(H,W)` map | `dinoisawesome.utils.upsample_map` |
| Blend a jet heatmap over an image | `dinoisawesome.utils.heat_overlay` |
| Semi-transparent mask overlay for `ax.imshow` | `dinoisawesome.utils.mask_overlay_rgba` |
| Draw a labelled box / labelled points on an image | `dinoisawesome.utils.draw_box` / `draw_points` |
| Project patch tokens to an RGB PCA map | `dinoisawesome.utils.pca_project_to_rgb` / `to_display_upscale` |
| Normalize any image input to PIL, or a display thumbnail | `dinoisawesome.utils.to_pil` / `thumb` |

If a genuinely new helper is needed and it's likely to be reused (not a
one-off plot tweak), add it to the appropriate module in `dinoisawesome/utils/`
instead of defining it inline in the notebook — that's the whole point of the
directory. One-off, notebook-specific plotting closures (e.g. a bespoke
multi-panel figure layout used exactly once) can stay inline.

Existing model-facing building blocks already live in the package and are
not duplicated in `dinoisawesome.utils` — reuse them directly:
`DinoEncoder` (`dinoisawesome.encoder`), `Gallery`/`GalleryConfig`
(patch retrieval), `KeypointHead`, `ForegroundHead`, `AnomalyHead`, and
`extract_patch_tokens` / `compute_exemplar_features` / `compute_density_map`
/ `extract_peaks` (`dinoisawesome.instance_detection`).

## Non-negotiables (from CLAUDE.md — do not skip these)

- **Logging before torch.** The very first code cell configures
  `logging.basicConfig(...)` and only *then* imports anything that pulls in
  torch (`dinoisawesome`, `torch` itself). This must stay inline in the
  notebook/script — it cannot be moved into `dinoisawesome.utils`, because
  importing anything from the `dinoisawesome` package (including its `utils`
  subpackage) runs `dinoisawesome/__init__.py`, which imports `torch`.
- **No `print()`** — use `log.info(...)` / `log.warning(...)`.
- **No `os.getcwd()`** — anchor paths to `Path(__file__).parent` (scripts) or
  a config-provided path; notebooks reference `data/` relative to the repo.
- **Gallery arrays stay memory-mapped** — don't load a full gallery's vectors
  into RAM with `np.load(..., mmap_mode=None)` or `.copy()` across all rows.

## Notebook skeleton

New experiment notebooks follow this cell order (see any of
`notebooks/patch_clustering.ipynb`, `instance_detection.ipynb`,
`keypoint_matching.ipynb` for real examples):

1. **Markdown** — one-paragraph description of what's being tested and why.
2. **Logging + imports** — logging config first, then imports, ending with
   the `dinoisawesome` / `dinoisawesome.utils` imports.
3. **Parameters** — every tunable constant (model version/size, image size,
   thresholds, layer index, paths) as UPPER_CASE module-level constants, with
   inline comments for units/valid ranges. `DINO_WEIGHTS_DIR` always read
   from the environment via `os.environ.get("DINO_WEIGHTS_DIR")`, never
   hardcoded.
4. **Load model** — construct `DinoEncoder(...)`, log its grid shape.
5. **Load data + preview** — open image(s)/mask(s) from `data/`, show them
   once so a bad path or misaligned mask is obvious immediately.
6. **Core experiment** — the thing being tested. Use existing package
   functions/utils; keep this cell focused on computation, not plotting.
7. **Visualize results** — plot what step 6 produced.
8. **Repeat / sweep** — either a small parameter sweep (list of values ->
   one subplot per value) or a note on which constant to change next.

Use `notebooks/instance_detection.ipynb` as the fullest reference: it
threads model-load -> per-scale/per-threshold sweeps -> multi-panel
visualisation -> a second-stage pipeline reusing the same primitives.

## Data

Reuse the datasets already referenced by existing notebooks
(`data/`, and the `abc`/`abc2`/`custom_slim`-style directories notebooks
`glob()` for) — don't fetch or fabricate new sample data unless asked.

## When the right approach is ambiguous

Per CLAUDE.md: don't guess model size, layer index, or storage format from
whatever the nearest existing notebook happens to use if the new experiment
is meaningfully different — ask. Do infer conventions (parameter block
style, logging setup, cell ordering) from the existing notebooks without
asking, since those are structural, not scientific, choices.
