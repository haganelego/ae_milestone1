"""Convert Visual Genome GT annotations to standard scene graph format."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import load_config, resolve
from milestone1.gt import load_gt_scene_graph


def main() -> None:
    cfg = load_config()
    vg_root = cfg["paths"]["vg_root"]
    split = cfg["paths"]["vg_split"]

    selected_path = resolve(cfg["paths"]["selected_images"])
    with open(selected_path) as f:
        image_ids = json.load(f)

    out_root = resolve(cfg["paths"]["gt_scene_graphs"])
    out_root.mkdir(parents=True, exist_ok=True)

    for img_id in image_ids:
        sg = load_gt_scene_graph(vg_root, split, img_id)
        out_path = out_root / f"{img_id}.json"
        with open(out_path, "w") as f:
            json.dump(sg, f, indent=2)
        print(
            f"  {img_id}: {len(sg['objects'])} objects, "
            f"{len(sg['relationships'])} relationships"
        )

    print(f"Saved {len(image_ids)} GT scene graphs to {out_root}")


if __name__ == "__main__":
    main()
