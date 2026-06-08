"""Configuration loading and path resolution."""

import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load the YAML configuration file."""
    if config_path is None:
        config_path = str(REPO_ROOT / "configs" / "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


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
