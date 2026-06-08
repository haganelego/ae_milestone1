"""Validate EGTR output schema (03 already writes final format)."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import load_config, resolve

REQUIRED_KEYS = {"image_id", "level", "objects", "relationships"}
OBJ_KEYS = {"object_id", "label", "confidence", "bbox"}
REL_KEYS = {
    "subject_id", "subject_label", "predicate",
    "object_id", "object_label", "confidence",
}


def main() -> None:
    cfg = load_config()

    selected_path = resolve(cfg["paths"]["selected_images"])
    with open(selected_path) as f:
        image_ids = json.load(f)

    pred_root = resolve(cfg["paths"]["egtr_predictions"])
    num_levels = cfg["resolution_levels"]["num_levels"]

    errors = 0
    checked = 0

    for img_id in image_ids:
        for level_idx in range(1, num_levels + 1):
            level_name = f"level_{level_idx:02d}"
            path = pred_root / str(img_id) / f"{level_name}.json"

            if not path.exists():
                print(f"  MISSING: {path}")
                errors += 1
                continue

            with open(path) as f:
                data = json.load(f)

            # Check top-level keys
            missing = REQUIRED_KEYS - set(data.keys())
            if missing:
                print(f"  {path}: missing keys {missing}")
                errors += 1
                continue

            # Check object schema
            for obj in data["objects"]:
                obj_missing = OBJ_KEYS - set(obj.keys())
                if obj_missing:
                    print(f"  {path}: object missing keys {obj_missing}")
                    errors += 1
                    break

            # Check relationship schema
            for rel in data["relationships"]:
                rel_missing = REL_KEYS - set(rel.keys())
                if rel_missing:
                    print(f"  {path}: relationship missing keys {rel_missing}")
                    errors += 1
                    break

            checked += 1

    print(f"Validated {checked} files, {errors} errors.")
    if errors == 0:
        print("All outputs conform to expected schema.")


if __name__ == "__main__":
    main()
