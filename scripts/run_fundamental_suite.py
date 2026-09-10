"""Run all (or a subset) of `experiments/fundamental/`'s scripts from one shared cfg.

Each script in that directory is a standalone, cell-based experiment with its own
hardcoded parameters (DINO size, image resolution, layer index, ...) — today, trying a
different setting means hand-editing constants inside the script itself, and every run
overwrites the previous one's `outputs/fundamental*/<script>/` figures. This runner
gives them a single YAML entry point instead: shared/per-script parameter overrides,
and every run written under its own timestamped, named directory (which also gets a
copy of the exact cfg used), so nothing overwrites a previous run and you can tell runs
apart by intent.

Each target script picks up `RUN_CFG`/`RUN_DIR` (set below in the child's environment)
via `experiments/_shared/run_config.py`; see that module's docstring. A script run
standalone (no env vars set) is completely unaffected — this is opt-in.

Usage:
    python scripts/run_fundamental_suite.py \
        --config experiments/fundamental/suite_config.example.yaml --name layer-sweep-v2
    python scripts/run_fundamental_suite.py \
        --config my_cfg.yaml --scripts scale_crop_similarity debias_ablation
    python scripts/run_fundamental_suite.py --config my_cfg.yaml --name smoke-test --dry-run
    python scripts/run_fundamental_suite.py --list
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)
log = logging.getLogger("run_fundamental_suite")

REPO_ROOT = Path(__file__).resolve().parents[1]
FUNDAMENTAL_DIR = REPO_ROOT / "experiments" / "fundamental"
RUNS_ROOT = REPO_ROOT / "outputs" / "fundamental_runs"

# Progressive-series order (see experiments/README.md's `fundamental/` section for the
# first six); the rest follow directory order. `_scale_composition_common.py` is
# shared plumbing imported by the scale_composition_* scripts below, not a runnable
# experiment itself, so it's excluded here — same convention as `_shared/`.
SUITE_SCRIPTS: list[str] = [
    "scale_crop_similarity",
    "augmentation_sensitivity",
    "augmented_prototype_oracle_iou_knn_fgbg",
    "training_set_size_ablation",
    "resolution_ablation",
    "debias_ablation",
    "feature_transform_oracle_iou",
    "noisy_fgbg_cleaning",
    "scale_composition_oracle_iou",
    "scale_composition_adaptive_oracle",
    "scale_composition_bg_ablation",
    "scale_composition_max_pool",
    "scale_composition_query_matching",
]


@dataclass
class ScriptResult:
    name: str
    status: str  # "ok" | "failed"
    returncode: int
    duration_s: float
    log_path: str


@dataclass
class SuiteResult:
    results: list[ScriptResult] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(r.status == "ok" for r in self.results)


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="Path to the suite's YAML config file.")
    parser.add_argument(
        "--name",
        help="Run name, used to make the output dir identifiable (e.g. 'layer-sweep-v2'). "
        "Overrides the cfg's own top-level `name:`, if any. Defaults to 'run'.",
    )
    parser.add_argument(
        "--scripts",
        nargs="+",
        metavar="SCRIPT",
        help="Subset of SUITE_SCRIPTS to run (by stem), in the order given. "
        "Overrides the cfg's own top-level `scripts:` list, if any. Defaults to all.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the suite on the first script failure instead of continuing to the rest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved run dir and script list without creating anything or running.",
    )
    parser.add_argument(
        "--list", action="store_true", help="Print the available script stems and exit."
    )
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict:
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def resolve_script_list(cli_scripts: list[str] | None, cfg: dict) -> list[str]:
    names = cli_scripts or cfg.get("scripts") or SUITE_SCRIPTS
    unknown = [n for n in names if n not in SUITE_SCRIPTS]
    if unknown:
        raise ValueError(
            f"Unknown script(s) {unknown!r}; choose from {SUITE_SCRIPTS} (see --list)."
        )
    return list(names)


def run_one_script(script_name: str, run_dir: Path, config_path: Path) -> ScriptResult:
    script_path = FUNDAMENTAL_DIR / f"{script_name}.py"
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{script_name}.log"

    env = {
        **os.environ,
        "RUN_CFG": str(config_path.resolve()),
        "RUN_DIR": str(run_dir.resolve()),
    }

    start = time.monotonic()
    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=REPO_ROOT,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
    duration_s = time.monotonic() - start

    status = "ok" if proc.returncode == 0 else "failed"
    if status == "failed":
        log.error(
            "%s failed (exit %d) after %.1fs — see %s",
            script_name,
            proc.returncode,
            duration_s,
            log_path,
        )
    else:
        log.info("%s finished (%.1fs)", script_name, duration_s)

    return ScriptResult(
        name=script_name,
        status=status,
        returncode=proc.returncode,
        duration_s=duration_s,
        log_path=str(log_path.relative_to(run_dir)),
    )


def write_manifest(run_dir: Path, run_name: str, config_path: Path, result: SuiteResult) -> None:
    manifest = {
        "run_name": run_name,
        "config_file": config_path.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scripts": [r.__dict__ for r in result.results],
        "all_ok": result.all_ok,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> int:
    args = parse_args()

    if args.list:
        print("\n".join(SUITE_SCRIPTS))
        return 0

    if not args.config:
        log.error("--config is required (or pass --list to see available scripts).")
        return 2
    if not args.config.is_file():
        log.error("Config file not found: %s", args.config)
        return 2

    cfg = load_yaml_config(args.config)
    run_name = args.name or cfg.get("name") or "run"
    scripts = resolve_script_list(args.scripts, cfg)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_ROOT / f"{timestamp}_{slugify(run_name)}"

    if args.dry_run:
        log.info("[dry-run] run_dir would be: %s", run_dir)
        log.info("[dry-run] scripts to run, in order: %s", scripts)
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    config_copy_path = run_dir / "config.yaml"
    shutil.copy2(args.config, config_copy_path)
    log.info("Run %r → %s (cfg copied to %s)", run_name, run_dir, config_copy_path.name)

    suite_result = SuiteResult()
    for script_name in tqdm(scripts, desc=f"fundamental suite: {run_name}"):
        result = run_one_script(script_name, run_dir, config_copy_path)
        suite_result.results.append(result)
        if result.status == "failed" and args.stop_on_error:
            log.error("Stopping suite early (--stop-on-error) after %s failed.", script_name)
            break

    write_manifest(run_dir, run_name, args.config, suite_result)

    n_ok = sum(1 for r in suite_result.results if r.status == "ok")
    n_total = len(suite_result.results)
    log.info("Suite finished: %d/%d scripts ok. Results in %s", n_ok, n_total, run_dir)

    return 0 if suite_result.all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
