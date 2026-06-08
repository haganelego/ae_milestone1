"""Load object and predicate vocabularies from Visual Genome JSON files."""

import json
from pathlib import Path


def load_obj_names(vg_root: str | Path) -> dict[int, str]:
    """Load object category names from test.json.

    Returns a dict mapping 0-indexed class id (0..149) to name.
    Model output pred_classes uses 0-indexed ids.
    """
    with open(Path(vg_root) / "test.json") as f:
        vg_data = json.load(f)
    return {
        c["id"] - 1: c["name"]
        for c in sorted(vg_data["categories"], key=lambda x: x["id"])
    }


def load_pred_names(vg_root: str | Path) -> list[str]:
    """Load predicate names from rel.json.

    Returns a list of 50 predicate names, 0-indexed matching model output.
    """
    with open(Path(vg_root) / "rel.json") as f:
        rel_data = json.load(f)
    return rel_data["rel_categories"][1:]  # skip __background__
