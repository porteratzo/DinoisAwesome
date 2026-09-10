"""Config plumbing for `scripts/run_fundamental_suite.py`.

The suite runner launches each `experiments/fundamental/*.py` script as its own
subprocess with `RUN_CFG` (path to the suite's YAML config) and `RUN_DIR` (this
run's own output directory) set in its environment. A script picks those up by
calling `load_run_config(__file__)` + `apply_overrides(globals(), ...)` right after
its own "# %% Parameters" cell, and `resolve_output_dir(...)` in place of its usual
`OUTPUT_DIR = ...; OUTPUT_DIR.mkdir(...)` lines.

Neither env var is set when a script is run standalone or cell-by-cell in an
Interactive Window/Jupyter session, so `load_run_config` returns `{}`, `apply_overrides`
is a no-op, and `resolve_output_dir` returns the script's own default path unchanged —
existing interactive usage is untouched.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def load_run_config(script_path: str | Path) -> dict[str, Any]:
    """This run's parameter overrides for the script at `script_path`.

    Reads the YAML file at `$RUN_CFG` (empty dict if unset) and merges its top-level
    `defaults` mapping with the mapping under this script's own stem (e.g.
    `debias_ablation:`), the latter taking precedence.
    """
    cfg_path = os.environ.get("RUN_CFG")
    if not cfg_path:
        return {}

    import yaml

    with open(cfg_path) as f:
        full_cfg = yaml.safe_load(f) or {}

    script_name = Path(script_path).stem
    merged = dict(full_cfg.get("defaults") or {})
    merged.update(full_cfg.get(script_name) or {})
    return merged


def apply_overrides(script_globals: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Apply `overrides` (lowercase/snake_case YAML keys) onto the matching UPPERCASE
    constants already present in `script_globals` (pass `globals()`).

    Only overwrites constants the script already defines — an unrecognized key is a
    likely config typo, so it's logged and skipped rather than silently injected as a
    new global.
    """
    for key, value in overrides.items():
        const_name = key.upper()
        if const_name not in script_globals:
            log.warning(
                "Config key %r has no matching %s constant in this script; ignoring.",
                key,
                const_name,
            )
            continue
        script_globals[const_name] = value


def resolve_output_dir(default_dir: Path) -> Path:
    """`$RUN_DIR/<default_dir.name>` when the suite runner set `RUN_DIR`, else
    `default_dir` unchanged. Creates the directory either way."""
    run_dir = os.environ.get("RUN_DIR")
    out_dir = Path(run_dir) / default_dir.name if run_dir else default_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
