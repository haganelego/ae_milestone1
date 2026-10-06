"""ROBOVIS2027: select test(100) + dev(20) VG-150 test images with GT-only criteria.

Outputs (never touches the MIRU files):
  data/processed/robovis2027/selected_images.json
  data/processed/robovis2027/eligible_ids.json
  data/processed/robovis2027/selection_manifest.json
"""

import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.gt import count_gt_objects_and_relationships  # noqa: E402
from milestone1.robovis_data import (  # noqa: E402
    load_robovis_config,
    meets_gt_criteria,
    sha256_file,
    sha256_ids,
    shuffle_and_split,
)

LIMITATIONS = [
    "Deduplication is exact only (image id and sha256 of image bytes); "
    "no near-duplicate (perceptual-hash) check was performed.",
    "No train-leak check beyond asserting every selected id is in the VG-150 "
    "test.json; PGSG's test split is the same VG-150 test split, so test images "
    "were not seen in PGSG training.",
    "Relation count is the raw length of rel.json[test][image_id] (may include "
    "duplicate triplets), matching the MIRU selection script.",
]


def git_head() -> str:
    """Return the current git commit (or 'unknown')."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def load_image_meta(vg_root: Path, split: str) -> dict[int, dict[str, Any]]:
    """Map image_id -> COCO image record from <split>.json."""
    with open(vg_root / f"{split}.json") as f:
        images = json.load(f)["images"]
    return {int(rec["id"]): rec for rec in images}


def filter_eligible(
    counts: list[dict[str, Any]],
    meta: dict[int, dict[str, Any]],
    images_dir: Path,
    min_obj: int,
    min_rel: int,
) -> tuple[list[int], Counter, dict[int, str]]:
    """Apply GT criteria, file checks and exact dedupe.

    Returns (eligible_ids, excluded_by_reason, sha256_by_id).
    """
    excluded: Counter = Counter()
    seen_ids: set[int] = set()
    candidates: list[int] = []
    for c in counts:
        image_id = int(c["image_id"])
        if image_id in seen_ids:
            excluded["duplicate_image_id"] += 1
            continue
        seen_ids.add(image_id)
        if c["num_objects"] < min_obj:
            excluded["too_few_gt_objects"] += 1
            continue
        if not meets_gt_criteria(
            c["num_objects"], c["num_relationships"], min_obj, min_rel
        ):
            excluded["too_few_gt_relations"] += 1
            continue
        path = images_dir / meta[image_id]["file_name"]
        if not path.exists():
            excluded["missing_image_file"] += 1
            continue
        if path.stat().st_size == 0:
            excluded["zero_byte_image_file"] += 1
            continue
        candidates.append(image_id)

    # Exact-duplicate removal by image bytes: keep the smallest id per digest.
    sha_by_id: dict[int, str] = {}
    first_id_by_sha: dict[str, int] = {}
    eligible: list[int] = []
    for image_id in sorted(candidates):
        digest = sha256_file(images_dir / meta[image_id]["file_name"])
        if digest in first_id_by_sha:
            excluded["duplicate_image_sha256"] += 1
            continue
        first_id_by_sha[digest] = image_id
        sha_by_id[image_id] = digest
        eligible.append(image_id)
    return eligible, excluded, sha_by_id


def build_records(
    split_ids: dict[str, list[int]],
    meta: dict[int, dict[str, Any]],
    counts_by_id: dict[int, dict[str, Any]],
    sha_by_id: dict[int, str],
    images_dir: Path,
) -> list[dict[str, Any]]:
    """Per-image selection records (verifies true pixel size against JSON)."""
    records = []
    for split_name, ids in split_ids.items():
        for rank, image_id in enumerate(ids):
            rec = meta[image_id]
            with Image.open(images_dir / rec["file_name"]) as img:
                width, height = img.size
            if (width, height) != (rec["width"], rec["height"]):
                raise ValueError(f"{image_id}: file size {img.size} != test.json")
            records.append(
                {
                    "image_id": image_id,
                    "split": split_name,
                    "rank": rank,
                    "file_name": rec["file_name"],
                    "orig_width": width,
                    "orig_height": height,
                    "sha256": sha_by_id[image_id],
                    "n_gt_objects": counts_by_id[image_id]["num_objects"],
                    "n_gt_relations": counts_by_id[image_id]["num_relationships"],
                }
            )
    return records


def size_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Orientation / max-side summary of the selected images."""
    max_sides = Counter(max(r["orig_width"], r["orig_height"]) for r in records)
    return {
        "n_portrait": sum(r["orig_height"] > r["orig_width"] for r in records),
        "n_landscape": sum(r["orig_width"] > r["orig_height"] for r in records),
        "n_square": sum(r["orig_width"] == r["orig_height"] for r in records),
        "max_side_counts": {str(k): v for k, v in sorted(max_sides.items())},
    }


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_robovis_config()
    ds = cfg["dataset"]
    paths = cfg["paths"]
    vg_root = Path(ds["vg_root"])
    split = ds["source_split"]
    images_dir = vg_root / "images"

    meta = load_image_meta(vg_root, split)
    counts = count_gt_objects_and_relationships(vg_root, split)
    counts_by_id = {int(c["image_id"]): c for c in counts}
    print(f"{split}.json images: {len(meta)}")

    eligible, excluded, sha_by_id = filter_eligible(
        counts, meta, images_dir, ds["min_gt_objects"], ds["min_gt_relations"]
    )
    print(f"Eligible: {len(eligible)}  excluded: {dict(excluded)}")

    _, test_ids, dev_ids = shuffle_and_split(
        eligible, ds["seed"], ds["test_count"], ds["dev_count"]
    )
    assert not set(test_ids) & set(dev_ids), "test/dev overlap"
    selected = test_ids + dev_ids
    assert all(i in meta for i in selected), "selected id not in source test.json"

    records = build_records(
        {"test": test_ids, "dev": dev_ids}, meta, counts_by_id, sha_by_id, images_dir
    )

    with open(resolve(paths["miru_selected_images"])) as f:
        miru_ids = {int(i) for i in json.load(f)}

    manifest = {
        "spec_version": cfg["spec_version"],
        "script": "scripts/26_robovis_select_images.py",
        "git_commit": git_head(),
        "source": {
            "vg_root": str(vg_root),
            "annotations": f"{split}.json",
            "relations": f"rel.json[{split!r}]",
            "split": split,
            "n_images_in_split": len(meta),
            "all_selected_in_source_split": True,
            "pgsg_test_split_is_same_vg150_test": True,
        },
        "criteria": {
            "min_gt_objects": ds["min_gt_objects"],
            "min_gt_relations": ds["min_gt_relations"],
            "exclude_zero_byte_files": True,
            "dedupe": ["image_id", "image_bytes_sha256 (keep smallest id)"],
            "observer_dependent_filters": None,
        },
        "ordering": {
            "procedure": "sorted(eligible_ids) then random.Random(seed).shuffle",
            "seed": ds["seed"],
            "test": f"shuffled[0:{ds['test_count']}]",
            "dev": f"shuffled[{ds['test_count']}:{ds['test_count'] + ds['dev_count']}]",
            "test_truncatable_to_prefix": True,
        },
        "counts": {
            "eligible": len(eligible),
            "excluded_by_reason": dict(sorted(excluded.items())),
            "test": len(test_ids),
            "dev": len(dev_ids),
        },
        "eligible_ids_sha256": sha256_ids(sorted(eligible)),
        "overlap_with_miru_100": {
            "miru_file": paths["miru_selected_images"],
            "miru_count": len(miru_ids),
            "test_overlap": len(miru_ids & set(test_ids)),
            "dev_overlap": len(miru_ids & set(dev_ids)),
            "total_overlap": len(miru_ids & set(selected)),
        },
        "selected_size_stats": size_stats(records),
        "limitations": LIMITATIONS,
    }

    write_json(resolve(paths["selected_images"]), records)
    write_json(resolve(paths["eligible_ids"]), sorted(eligible))
    write_json(resolve(paths["selection_manifest"]), manifest)
    print(f"Overlap with MIRU 100: {manifest['overlap_with_miru_100']}")
    print(f"Size stats: {manifest['selected_size_stats']}")
    print(f"Wrote {paths['selected_images']} ({len(records)} images)")
    print(f"Elapsed: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
