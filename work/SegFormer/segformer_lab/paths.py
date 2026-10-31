"""Path conventions for the SegFormer PCB1 experiment.

Two roots matter and they are deliberately separate:

``project_root``
    ``E:\\...\\AI_Try\\work\\SegFormer`` - source code, configs, manifests,
    stage documents. Tracked by Git.

``resource_root``
    ``E:\\...\\AI\\segformer_resources`` - Conda prefix, caches, temp files,
    extracted data, model snapshots, runs. Outside Git.

Every module resolves paths through this file so a moved checkout only needs
one override (``SEGFORMER_RESOURCE``) instead of edits in many places.
"""

from __future__ import annotations

import os
from pathlib import Path

# .../AI_Try/work/SegFormer/segformer_lab/paths.py -> parents[1] is the project dir
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


def _resource_root() -> Path:
    """Resource root: explicit override first, otherwise the documented layout."""
    override = os.environ.get("SEGFORMER_RESOURCE")
    if override:
        return Path(override).expanduser().resolve()
    # PROJECT_ROOT = <workspace>/AI_Try/work/SegFormer
    return PROJECT_ROOT.parents[2] / "segformer_resources"


RESOURCE_ROOT: Path = _resource_root()

ENV_DIR: Path = RESOURCE_ROOT / "env"
CACHE_DIR: Path = RESOURCE_ROOT / "cache"
TMP_DIR: Path = RESOURCE_ROOT / "tmp"
AUDIT_DIR: Path = RESOURCE_ROOT / "audit"
DATA_DIR: Path = RESOURCE_ROOT / "data"
MODELS_DIR: Path = RESOURCE_ROOT / "models"
RUNS_DIR: Path = RESOURCE_ROOT / "runs"
CONFIG_DIR: Path = RESOURCE_ROOT / "config"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
MANIFESTS_DIR: Path = PROJECT_ROOT / "manifests"
DOCS_DIR: Path = PROJECT_ROOT / "docs"


def resolve(path_like: str | os.PathLike[str]) -> Path:
    """Resolve *path_like* against the project root unless it is absolute.

    Relative paths inside configs and CLI examples are always meant relative to
    the experiment project, never relative to whatever directory the shell
    happens to be in. Absolute paths are returned unchanged (resolved).
    """
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def ensure_parent(path_like: str | os.PathLike[str]) -> Path:
    """Resolve *path_like* and create its parent directory, returning the path."""
    path = resolve(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
