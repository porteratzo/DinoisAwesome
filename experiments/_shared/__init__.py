"""Shared eval-harness utilities for scripts under experiments/.

Not part of the installed ``dinoisawesome`` package — this is experiment-plumbing
(crop/patch/annotation conventions specific to how the scripts in this directory are
written), not general-purpose DINO library code. Each experiment script adds
``experiments/`` to ``sys.path`` before importing from here (see any script's
``sys.path.insert(0, str(_REPO_ROOT / "experiments"))`` line), the same convention
``eval_custom_slim.py`` already used for ``scripts/eval_sam_dino.py``.
"""
