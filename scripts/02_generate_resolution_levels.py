"""Generate degraded images at multiple resolution levels."""

import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import load_config, resolve

# Map config interpolation name -> PIL resample filter.
_RESAMPLE_FILTERS = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}


def main() -> None:
    cfg = load_config()
    vg_root = Path(cfg["paths"]["vg_root"])
    scales = cfg["resolution_levels"]["scales"]
    num_levels = cfg["resolution_levels"]["num_levels"]
    interp_name = cfg["resolution_levels"].get("interpolation", "bilinear")
    resample = _RESAMPLE_FILTERS[interp_name]
    print(f"Interpolation: {interp_name}")

    # Load selected images
    selected_path = resolve(cfg["paths"]["selected_images"])
    with open(selected_path) as f:
        image_ids = json.load(f)

    out_root = resolve(cfg["paths"]["degraded_images"])
    print(f"Generating {num_levels} resolution levels for {len(image_ids)} images")

    for img_id in image_ids:
        img_path = vg_root / "images" / f"{img_id}.jpg"
        if not img_path.exists():
            print(f"  WARNING: {img_path} not found, skipping")
            continue

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        img_out_dir = out_root / str(img_id)
        img_out_dir.mkdir(parents=True, exist_ok=True)

        for level_idx, scale in enumerate(scales, start=1):
            level_name = f"level_{level_idx:02d}"
            out_path = img_out_dir / f"{level_name}.jpg"

            # Skip if already exists (idempotent reuse)
            if out_path.exists():
                continue

            if scale >= 1.0:
                # Original resolution — just save as-is
                degraded = img.copy()
            else:
                # Downscale then upscale back to original size
                small_w = max(1, int(orig_w * scale))
                small_h = max(1, int(orig_h * scale))
                small = img.resize((small_w, small_h), resample)
                degraded = small.resize((orig_w, orig_h), resample)

            degraded.save(out_path, quality=95)

        print(f"  {img_id}: {num_levels} levels generated ({orig_w}x{orig_h})")

    print("Done.")


if __name__ == "__main__":
    main()
