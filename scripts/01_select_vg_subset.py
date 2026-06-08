"""Select a subset of Visual Genome images with sufficient GT annotations."""

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import load_config, resolve
from milestone1.gt import count_gt_objects_and_relationships


def main() -> None:
    cfg = load_config()
    vg_root = cfg["paths"]["vg_root"]
    split = cfg["paths"]["vg_split"]
    num_images = cfg["dataset"]["num_images"]
    min_obj = cfg["dataset"]["min_gt_objects"]
    min_rel = cfg["dataset"]["min_gt_relationships"]
    seed = cfg["dataset"]["seed"]

    print(f"Loading GT counts from {vg_root}/{split}.json ...")
    counts = count_gt_objects_and_relationships(vg_root, split)
    print(f"Total images in {split}: {len(counts)}")

    # Filter by minimum counts
    eligible = [
        c for c in counts
        if c["num_objects"] >= min_obj and c["num_relationships"] >= min_rel
    ]
    print(f"Eligible (>={min_obj} objects, >={min_rel} rels): {len(eligible)}")

    # Deterministic selection
    random.seed(seed)
    random.shuffle(eligible)
    selected = eligible[:num_images]

    # Sort by image_id for consistency
    selected_ids = sorted([s["image_id"] for s in selected])

    # Save
    out_path = resolve(cfg["paths"]["selected_images"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(selected_ids, f, indent=2)

    print(f"Selected {len(selected_ids)} images: {selected_ids}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
