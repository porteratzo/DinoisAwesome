# CLAUDE.md

Experiments with DINO vision transformer encoders (v2 / v3). Provides feature extraction and patch-level retrieval galleries backed by pandas + memory-mapped NumPy arrays.

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

# Run tests (once a test suite exists)
pytest
```

## Non-Negotiables

- **No `print()`** — use Python's `logging` module so level/handler control is preserved.
- **No hardcoded paths via `os.getcwd()`** — anchor to `Path(__file__).parent` or a config-provided storage path.
- **No unbounded array loads** — gallery vectors are memory-mapped; keep it that way.
- **Initialize logging before importing torch** — torch registers handlers at import time on some builds.

## Working Assumptions

- Don't infer the intended approach from file presence alone; files may be leftover experiments.
- When the right model size, layer index, or storage format is ambiguous, ask before implementing.

## Critical Thinking

Evaluate requests on their technical merits before acting. If you spot a flaw, a simpler path, or a hidden cost, say so and explain why. When a plan is sound, confirm and proceed.

Every bug fix should include a regression test (or an explanation of why one isn't practical).

## Pre-Push Checklist

Before every `git push`:
1. `ruff check .` — fix all errors
2. `ruff format .` — apply formatting
3. `mypy dinoisawesome/` — resolve type errors
