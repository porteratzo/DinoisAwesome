# CLAUDE.md

Experiments with DINO vision transformer encoders (v2 / v3). Provides feature extraction and patch-level retrieval galleries backed by pandas + memory-mapped NumPy arrays.

**This is primarily an experiments repository. Do not write or run tests.**

## Common Commands

```bash
# Install in editable mode with dev deps
pip install -e ".[dev]"

# Lint
ruff check .

# Format
ruff format .

# Type-check
mypy dinoisawesome/

```

## Non-Negotiables

- **No `print()`** — use Python's `logging` module so level/handler control is preserved.
- **No hardcoded paths via `os.getcwd()`** — anchor to `Path(__file__).parent` or a config-provided storage path.
- **No unbounded array loads** — gallery vectors are memory-mapped; keep it that way.
- **Initialize logging before importing torch** — torch registers handlers at import time on some builds.
- **Wrap long-running loops in `tqdm`** — encoding passes, per-pair/per-image processing, batch inference, etc. need visible progress feedback; bare `for` loops over more than a handful of items should use `tqdm(...)` (or `tqdm.write()` if logging inside the loop).

## DinoEncoder Performance

`DinoEncoder` (`dinoisawesome/encoder.py`) defaults to the safe, general-purpose config:
`amp=False` (plain fp32), `resize_workers=0`, `gpu_normalize=False`. TF32 is the one
optimization already on unconditionally for any CUDA device — it's a near-free ~1.7x on
the backbone with negligible accuracy cost, verified via cosine similarity on both v2 and
v3 (see git history on `encoder.py` for the benchmark notes).

`amp` defaults to `False` to match the usual convention in other projects (explicit
opt-in for mixed precision rather than a silent default), not because it's risky here.
It's generally fine to use — backbone weights/compute move to bfloat16, a real numerical
change (not free like TF32), with the most sensitive case measured being a dinov2
patch-token worst-case cosine similarity down to ~0.50 vs. fp32 (dinov3 stayed much
tighter, ~0.991 worst-case) — worth knowing about, but a normal bf16 precision tradeoff,
not a reason to avoid it. Enable it whenever exact fp32 reproducibility isn't the point,
which is most experiments.

For a long-running job (a full gallery build, a sweep over many images, anything that'll
run for more than a couple minutes unattended) where the same encoder config gets reused
across many calls, it's worth tuning rather than accepting every default:

- **Enable `amp=True`** if the experiment doesn't depend on exact fp32 behavior — the
  backbone is the dominant cost, and bf16 autocast is the biggest lever available.
- **Pick a batch size deliberately.** The preprocessing optimizations below only pay off
  once there's enough work per call to amortize their overhead, and that threshold depends
  on how fast the backbone already is.
- **`resize_workers=4`** is a reasonable starting point if batches are 16+ images — more
  threads did not measure faster on this project's hybrid P-core/E-core dev machine, and
  smaller batches saw a net loss from thread-dispatch overhead (worse under `amp=True`,
  where the backbone is fast enough that the fixed overhead is a bigger fraction of the
  total). Don't enable it for small-batch or interactive use.
- **`gpu_normalize=True`** is worth it once `amp=True` and batches are 8+ — this was the
  bigger, more consistent win of the two (up to ~5x end-to-end at batch 32 under
  amp+TF32), but only once the backbone is fast enough for preprocessing to matter; under
  a slow (fp32) backbone it's noise-level.
- Both preprocessing flags are opt-in and compose with each other; neither changes output
  beyond float-rounding-level amounts (cosine similarity >=0.999 in every configuration
  tested) — they're dispatch/placement changes, not algorithm changes.

In short: defaults are for correctness-first, ad-hoc, or small-scale use; for a long
unattended run, turn on `amp`, pick a batch size worth the overhead, and enable
`resize_workers`/`gpu_normalize` to match it.

## Working Assumptions

- Don't infer the intended approach from file presence alone; files may be leftover experiments.
- When the right model size, layer index, or storage format is ambiguous, ask before implementing.

## Critical Thinking

Evaluate requests on their technical merits before acting. If you spot a flaw, a simpler path, or a hidden cost, say so and explain why. When a plan is sound, confirm and proceed.
