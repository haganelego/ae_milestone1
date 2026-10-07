"""Configuration loading and path resolution."""

import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


CONFIG_OVERLAY_ENV = "MILESTONE1_CONFIG_OVERLAY"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``overlay`` merged in (nested dicts merged, rest replaced)."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load the YAML configuration file.

    If the environment variable ``MILESTONE1_CONFIG_OVERLAY`` names a YAML file,
    it is deep-merged on top (e.g. configs/config_heldout.yaml redirects all
    MIRU paths to a held-out image set without touching the MIRU outputs).
    """
    if config_path is None:
        config_path = str(REPO_ROOT / "configs" / "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    overlay_path = os.environ.get(CONFIG_OVERLAY_ENV)
    if overlay_path:
        with open(resolve(overlay_path)) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
        cfg["_overlay"] = str(overlay_path)
    return cfg


def resolve(path: str) -> Path:
    """Resolve a config-relative path against REPO_ROOT."""
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def setup_egtr_path() -> None:
    """Add EGTR submodule to sys.path for imports."""
    egtr_path = str(REPO_ROOT / "egtr")
    if egtr_path not in sys.path:
        sys.path.insert(0, egtr_path)
