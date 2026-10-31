"""Configuration loading for the SegFormer PCB1 experiment.

One job only: turn ``configs/*.yaml`` into a dictionary where every path is an
absolute ``Path``, and fail loudly when a required field is missing or a fixed
protocol value has drifted.

Why a dedicated module this early: the audit and preview commands must read the
same ``data.root`` / ``input_size`` / ``ignore_index`` that the training stage
will use later. If each script parsed YAML on its own, the numbers could drift
apart without anyone noticing.

Path rules
----------
* Relative paths resolve against the project root (``paths.PROJECT_ROOT``).
* ``~`` is expanded; environment variables are not, so a config can never
  depend on whatever a particular shell happens to export.
* Absolute paths are kept as-is (resolved for ``..``/symlinks).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import paths

# Fields that define the experiment protocol. Changing one changes the identity
# of the run, so they are validated rather than silently accepted.
FIXED_VALUES = {
    ("data", "input_size"): 512,
    ("data", "resize_mode"): "letterbox",
    ("data", "ignore_index"): 255,
    ("model", "num_labels"): 2,
    ("eval", "metric_space"): "original",
}

PATH_FIELDS = (
    ("data", "root"),
    ("data", "train_manifest"),
    ("data", "val_manifest"),
    ("data", "test_manifest"),
    ("data", "split_meta"),
    ("data", "train_ids_file"),
    ("experiment", "output_root"),
    ("model", "pretrained_dir"),
    ("logging", "fixed_preview_manifest"),
)

REQUIRED_SECTIONS = ("experiment", "data", "model", "train", "loss", "eval", "logging")


class ConfigError(ValueError):
    """Raised when a configuration file cannot be used as written."""


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = paths.PROJECT_ROOT / path
    return path.resolve()


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load, validate and path-resolve a configuration file."""
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = paths.PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    missing = [section for section in REQUIRED_SECTIONS if section not in raw]
    if missing:
        raise ConfigError(f"missing config section(s): {missing}")

    for (section, key), expected in FIXED_VALUES.items():
        actual = raw.get(section, {}).get(key)
        if actual != expected:
            raise ConfigError(
                f"{section}.{key} must be {expected!r} for this protocol, found {actual!r}"
            )

    resolved: dict[str, Path] = {}
    for section, key in PATH_FIELDS:
        value = raw.get(section, {}).get(key)
        if value in (None, ""):
            resolved[f"{section}.{key}"] = None  # type: ignore[assignment]
            continue
        resolved[f"{section}.{key}"] = _resolve(str(value))

    data = raw["data"]
    if int(data.get("input_size", 0)) <= 0:
        raise ConfigError("data.input_size must be positive")
    ignore_index = int(data["ignore_index"])
    if ignore_index in (0, 1):
        raise ConfigError("data.ignore_index must not collide with the 0/1 label space")
    pad = data.get("image_pad_rgb")
    if not (isinstance(pad, list) and len(pad) == 3 and all(isinstance(v, int) and 0 <= v <= 255 for v in pad)):
        raise ConfigError("data.image_pad_rgb must be three integers in 0..255")

    return {
        "path": path,
        "raw": raw,
        "paths": resolved,
        "data_root": resolved["data.root"],
        "input_size": int(data["input_size"]),
        "ignore_index": ignore_index,
        "image_pad_rgb": tuple(int(v) for v in pad),
        "category": str(data.get("category", "pcb1")),
        "num_labels": int(raw["model"]["num_labels"]),
    }


def resolved_yaml(cfg: dict[str, Any]) -> str:
    """Render the effective configuration (absolute paths) as YAML text."""
    document = {
        "config_path": str(cfg["path"]),
        "category": cfg["category"],
        "input_size": cfg["input_size"],
        "ignore_index": cfg["ignore_index"],
        "image_pad_rgb": list(cfg["image_pad_rgb"]),
        "num_labels": cfg["num_labels"],
        "paths": {key: (str(value) if value is not None else None) for key, value in cfg["paths"].items()},
        "raw": cfg["raw"],
    }
    return yaml.safe_dump(document, allow_unicode=True, sort_keys=False)


def write_resolved_config(cfg: dict[str, Any], out_path: str | Path) -> Path:
    """Write the resolved configuration next to the run's evidence."""
    target = paths.ensure_parent(out_path)
    target.write_text(resolved_yaml(cfg), encoding="utf-8")
    return target
