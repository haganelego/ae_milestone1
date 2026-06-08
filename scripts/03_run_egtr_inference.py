"""Run EGTR inference on all degraded images."""

import json
import sys
import time
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "egtr"))

from milestone1.config import load_config, resolve
from milestone1.egtr_infer import EgtrModel


def main() -> None:
    cfg = load_config()

    # Load selected images
    selected_path = resolve(cfg["paths"]["selected_images"])
    with open(selected_path) as f:
        image_ids = json.load(f)

    scales = cfg["resolution_levels"]["scales"]
    num_levels = cfg["resolution_levels"]["num_levels"]
    degraded_root = resolve(cfg["paths"]["degraded_images"])
    out_root = resolve(cfg["paths"]["egtr_predictions"])

    # Load model ONCE
    print("Loading EGTR model ...")
    t0 = time.time()
    egtr = EgtrModel(
        artifact_path=str(resolve(cfg["paths"]["egtr_artifact"])),
        architecture=cfg["paths"]["egtr_architecture"],
        vg_root=cfg["paths"]["vg_root"],
        object_conf_threshold=cfg["egtr"]["object_conf_threshold"],
        relation_conf_threshold=cfg["egtr"]["relation_conf_threshold"],
        top_k_objects=cfg["egtr"]["top_k_objects"],
        top_k_relationships=cfg["egtr"]["top_k_relationships"],
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    total = len(image_ids) * num_levels
    done = 0

    for img_id in image_ids:
        img_out_dir = out_root / str(img_id)
        img_out_dir.mkdir(parents=True, exist_ok=True)

        for level_idx in range(1, num_levels + 1):
            level_name = f"level_{level_idx:02d}"
            img_path = degraded_root / str(img_id) / f"{level_name}.jpg"

            out_path = img_out_dir / f"{level_name}.json"

            # Skip if already exists (idempotent reuse)
            if out_path.exists():
                done += 1
                continue

            if not img_path.exists():
                print(f"  WARNING: {img_path} not found, skipping")
                done += 1
                continue

            pil_img = Image.open(img_path).convert("RGB")
            result = egtr.predict(pil_img)

            output = {
                "image_id": img_id,
                "level": level_idx,
                "scale": scales[level_idx - 1],
                "objects": result["objects"],
                "relationships": result["relationships"],
            }

            with open(out_path, "w") as f:
                json.dump(output, f, indent=2)

            done += 1

        n_obj = len(result["objects"])
        n_rel = len(result["relationships"])
        print(
            f"  {img_id}: {num_levels} levels done "
            f"(last level: {n_obj} objects, {n_rel} rels) "
            f"[{done}/{total}]"
        )

    print("Inference complete.")


if __name__ == "__main__":
    main()
