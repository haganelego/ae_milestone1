"""ROBOVIS2027 recovery plan §6.1: freeze the new 60-image test set (selection only, no runs).

VG-150 official test split (test.json + rel.json), GT-only criteria (objects >= 5,
relations >= 3), excluding EVERY image id used so far (all id sets recorded in
<run_dir>/data_manifest.json by scripts/43, i.e. dev20, test100, held-out96, the MIRU
100 incl. its git history, PGSG pilot/validation ids, smoke and schedule-compare ids).
Remaining ids are sorted ascending, shuffled with random.Random(seed), first N kept.
No predictions / images / stop times are looked at.

Usage:
  uv run --project egtr python scripts/44_recovery_test_selection.py --run-id rec_r0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.gt import count_gt_objects_and_relationships  # noqa: E402
from milestone1.robovis_data import (  # noqa: E402
    meets_gt_criteria,
    sha256_file,
    sha256_ids,
)

OUT_ROOT = REPO_ROOT / "outputs/robovis2027_recovery"
VG_ROOT = Path("/home/kanaoka/usr/dataset/vg")
SPLIT = "test"
SEED = 20261008
N_SELECT = 60
MIN_OBJECTS = 5
MIN_RELATIONS = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-id", default="rec_r0")
    args = parser.parse_args()
    run_dir = OUT_ROOT / args.run_id
    manifest = json.loads((run_dir / "data_manifest.json").read_text())
    excluded_by_set = {name: s["ids"] for name, s in manifest["id_sets"].items()}
    excluded = set().union(*excluded_by_set.values())

    counts = count_gt_objects_and_relationships(VG_ROOT, SPLIT)
    eligible = sorted(
        c["image_id"]
        for c in counts
        if meets_gt_criteria(
            c["num_objects"], c["num_relationships"], MIN_OBJECTS, MIN_RELATIONS
        )
    )
    pool = sorted(i for i in eligible if i not in excluded)
    shuffled = list(pool)
    random.Random(SEED).shuffle(shuffled)
    selected = shuffled[:N_SELECT]
    by_id = {c["image_id"]: c for c in counts}

    record = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/44_recovery_test_selection.py",
        "source": {
            "vg_root": str(VG_ROOT),
            "split": SPLIT,
            "test.json_sha256": sha256_file(VG_ROOT / "test.json"),
            "rel.json_sha256": sha256_file(VG_ROOT / "rel.json"),
        },
        "criteria": {"min_gt_objects": MIN_OBJECTS, "min_gt_relations": MIN_RELATIONS},
        "procedure": "eligible ids minus all excluded ids, sorted ascending, random.Random(seed).shuffle, first n",
        "seed": SEED,
        "n_test_images": len(counts),
        "n_eligible": len(eligible),
        "n_excluded_total": len(excluded),
        "n_excluded_eligible": len(set(eligible) & excluded),
        "n_pool": len(pool),
        "excluded_sets": {
            name: {
                "n": len(ids),
                "n_in_eligible": len(set(ids) & set(eligible)),
                "source": manifest["id_sets"][name]["source"],
            }
            for name, ids in excluded_by_set.items()
        },
        "selected_ids": selected,
        "selected_sha256_ordered": sha256_ids(selected),
        "selected_gt_counts": {
            str(i): {
                "objects": by_id[i]["num_objects"],
                "relations": by_id[i]["num_relationships"],
            }
            for i in selected
        },
        "training_split_independence": {
            "assertable": True,
            "evidence": [
                "Local VG-150 splits are disjoint: train.json (57,723) / val.json (5,000) / test.json (26,446) share 0 image ids.",
                "Pix2Grp DATASET.MD: VG follows the BGNN / Tang 'Unbiased SGG' protocol; its loader "
                "(lavis/datasets/datasets/vg_rel_detection.py) trains on h5 split flag 0 and tests on split flag 2.",
                "The standard VG-150 test split has 26,446 images, matching the local test.json used here.",
            ],
            "caveat": "The authors' VG-SGG h5 files are not available locally, so the identity of their train split with "
            "ours is inferred from the published protocol and the matching test-split size, not byte-verified.",
        },
    }
    out = run_dir / "test_selection.json"
    out.write_text(json.dumps(record, indent=1))
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "n_test_images",
                    "n_eligible",
                    "n_excluded_total",
                    "n_excluded_eligible",
                    "n_pool",
                )
            }
        ),
        "selected",
        len(selected),
        selected[:5],
        "->",
        out,
    )


if __name__ == "__main__":
    main()
