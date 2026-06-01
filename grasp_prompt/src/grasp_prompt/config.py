from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, resolving optional relative `base` files."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised only without PyYAML
        raise RuntimeError("PyYAML is required to load YAML config files.") from exc

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is None:
        return cfg

    merged: dict[str, Any] = {}
    base_files = base if isinstance(base, list) else [base]
    for base_file in base_files:
        base_path = Path(base_file)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = merge_overrides(merged, load_config(base_path))
    return merge_overrides(merged, cfg)


def merge_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge a small override dictionary into a config."""
    merged = deepcopy(config)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_overrides(merged[key], value)
        else:
            merged[key] = value
    return merged
