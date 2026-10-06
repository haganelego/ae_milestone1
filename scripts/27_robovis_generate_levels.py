"""ROBOVIS2027: pre-shrink to max side 384, then generate 20 resolution levels.

Pipeline per image (all levels, including level_20 at scale 1.0):
  original RGB -> bilinear pre-shrink so max(W,H) == base_max_side
  -> downscale by scale (int(w*s), int(h*s), min 1 px) -> upscale to base size
  -> JPEG quality 95 at data/processed/robovis2027/degraded_images/<id>/level_XX.jpg

The degradation step mirrors scripts/02_generate_resolution_levels.py
(via milestone1.robovis_data.degrade). Writes levels_manifest.json.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.robovis_data import (  # noqa: E402
    RESAMPLE_FILTERS,
    degrade,
    effective_size,
    level_name,
    load_miru_scales,
    load_robovis_config,
    make_base_image,
    scales_equal,
    sha256_file,
)

PROTECTED_DIRS = ("data/processed/degraded_images", "outputs")


def check_output_root(out_root: Path) -> None:
    """Refuse to write into the MIRU result directories."""
    for protected in PROTECTED_DIRS:
        if out_root.resolve().is_relative_to(resolve(protected).resolve()):
            raise ValueError(f"Refusing to write into protected dir {protected}")


def process_image(
    rec: dict[str, Any],
    images_dir: Path,
    out_root: Path,
    scales: list[float],
    base_max_side: int,
    resample: int,
    jpeg_quality: int,
    output_format: str,
) -> dict[str, Any]:
    """Generate all levels for one image; return its manifest entry."""
    with Image.open(images_dir / rec["file_name"]) as raw:
        img = raw.convert("RGB")
    orig_w, orig_h = img.size
    base = make_base_image(img, base_max_side, resample)
    base_w, base_h = base.size

    img_out_dir = out_root / str(rec["image_id"])
    img_out_dir.mkdir(parents=True, exist_ok=True)
    level_sha: dict[str, str] = {}
    for level_idx, scale in enumerate(scales, start=1):
        out_path = img_out_dir / f"{level_name(level_idx)}.jpg"
        degrade(base, scale, resample).save(
            out_path, format=output_format, quality=jpeg_quality
        )
        level_sha[level_name(level_idx)] = sha256_file(out_path)

    return {
        "image_id": rec["image_id"],
        "split": rec["split"],
        "orig_width": orig_w,
        "orig_height": orig_h,
        "base_width": base_w,
        "base_height": base_h,
        "coord_scale": max(orig_w, orig_h) / base_max_side,
        "coord_scale_x": orig_w / base_w,
        "coord_scale_y": orig_h / base_h,
        "level_sha256": level_sha,
    }


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_robovis_config()
    ds, res, paths = cfg["dataset"], cfg["resolution"], cfg["paths"]
    scales = [float(s) for s in res["scales"]]
    miru_scales = load_miru_scales()
    assert scales_equal(scales, miru_scales), "robovis scales != configs/config.yaml"

    base_max_side = int(ds["base_max_side"])
    interp = res["interpolation"]
    resample = RESAMPLE_FILTERS[interp]
    jpeg_quality = int(res["jpeg_quality"])
    output_format = res["output_format"]
    images_dir = Path(ds["vg_root"]) / "images"
    out_root = resolve(paths["degraded_images"])
    check_output_root(out_root)

    with open(resolve(paths["selected_images"])) as f:
        records = json.load(f)
    print(f"Generating {len(scales)} levels for {len(records)} images -> {out_root}")

    entries = []
    for rec in records:
        entry = process_image(
            rec,
            images_dir,
            out_root,
            scales,
            base_max_side,
            resample,
            jpeg_quality,
            output_format,
        )
        entries.append(entry)
        print(
            f"  {rec['image_id']} [{rec['split']}] "
            f"{entry['orig_width']}x{entry['orig_height']} -> "
            f"{entry['base_width']}x{entry['base_height']}"
        )

    ex = entries[0]
    example_sizes = [
        {
            "level": level_name(i),
            "scale": s,
            "effective_size": list(
                effective_size(ex["base_width"], ex["base_height"], s)
            ),
        }
        for i, s in enumerate(scales, start=1)
    ]
    n_upscaled = sum(
        max(e["orig_width"], e["orig_height"]) < base_max_side for e in entries
    )
    manifest = {
        "spec_version": cfg["spec_version"],
        "script": "scripts/27_robovis_generate_levels.py",
        "selected_images": paths["selected_images"],
        "output_root": paths["degraded_images"],
        "naming": "<image_id>/level_XX.jpg (XX = 01..20, same as script 02)",
        "base_max_side": base_max_side,
        "pre_shrink": {
            "interpolation": interp,
            "rule": "aspect-preserving, longer side == base_max_side, shorter side rounded",
            "n_images_upscaled_in_pre_shrink": n_upscaled,
        },
        "coord_scale_definition": (
            "orig_max_side / base_max_side; multiply predicted boxes in base/level "
            "coordinates by this to get original GT coordinates "
            "(per-axis exact factors: coord_scale_x/coord_scale_y)"
        ),
        "scales": scales,
        "scales_match_configs_config_yaml": True,
        "interpolation": interp,
        "output_format": output_format,
        "jpeg_quality": jpeg_quality,
        "level_20_reencoded": True,
        "example_effective_sizes": {
            "image_id": ex["image_id"],
            "base_size": [ex["base_width"], ex["base_height"]],
            "levels": example_sizes,
        },
        "images": entries,
    }
    manifest_path = resolve(paths["levels_manifest"])
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote {manifest_path}")
    print(f"Elapsed: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
