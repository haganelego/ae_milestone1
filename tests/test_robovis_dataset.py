"""Tests for the ROBOVIS2027 dataset selection and resolution levels.

Run:  uv run --project egtr --with pytest python -m pytest tests/test_robovis_dataset.py -q
Fallback without pytest:  uv run --project egtr python tests/test_robovis_dataset.py --self-test
Data-dependent tests skip if scripts 26/27 have not been run.
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.gt import count_gt_objects_and_relationships  # noqa: E402
from milestone1.robovis_data import (  # noqa: E402
    RESAMPLE_FILTERS,
    degrade,
    effective_size,
    load_miru_scales,
    load_robovis_config,
    make_base_image,
    meets_gt_criteria,
    scales_equal,
    sha256_ids,
    shuffle_and_split,
)

KNOWN_SCALES = [
    0.05, 0.0585, 0.0685, 0.0802, 0.0940, 0.1100, 0.1288, 0.1508, 0.1765, 0.2066,
    0.2419, 0.2832, 0.3316, 0.3882, 0.4545, 0.5321, 0.6229, 0.7293, 0.8538, 1.0,
]  # fmt: skip
NUM_LEVELS = 20
BASE_MAX_SIDE = 384
SEED = 42
CFG = load_robovis_config()
PATHS = CFG["paths"]
DS = CFG["dataset"]


class SkipTestError(Exception):
    """Raised to skip a data-dependent test when pytest is unavailable."""


def _skip(reason: str) -> None:
    try:
        import pytest
    except ImportError as exc:
        raise SkipTestError(reason) from exc
    pytest.skip(reason)


def _load_json(key: str) -> Any:
    path = resolve(PATHS[key])
    if not path.exists():
        _skip(f"{path} missing; run scripts 26/27 first")
    with open(path) as f:
        return json.load(f)


# ── Pure / config tests ──


def test_scale_array_equals_known_values() -> None:
    scales = [float(s) for s in CFG["resolution"]["scales"]]
    assert len(scales) == NUM_LEVELS
    assert scales_equal(scales, KNOWN_SCALES)
    assert scales_equal(load_miru_scales(), KNOWN_SCALES)


def test_shuffle_and_split_is_deterministic_and_disjoint() -> None:
    ids = list(range(1000, 1500))
    first = shuffle_and_split(reversed(ids), SEED, 100, 20)
    second = shuffle_and_split(ids, SEED, 100, 20)
    assert first == second
    _, test_ids, dev_ids = first
    assert len(test_ids) == 100 and len(dev_ids) == 20
    assert not set(test_ids) & set(dev_ids)
    assert shuffle_and_split(ids, SEED + 1, 100, 20)[1] != test_ids


def test_meets_gt_criteria_boundaries() -> None:
    assert meets_gt_criteria(5, 3, 5, 3)
    assert not meets_gt_criteria(4, 3, 5, 3)
    assert not meets_gt_criteria(5, 2, 5, 3)


def test_degraded_dims_equal_base_and_level01_effective_size() -> None:
    img = Image.new("RGB", (500, 375), (120, 60, 30))
    resample = RESAMPLE_FILTERS["bilinear"]
    base = make_base_image(img, BASE_MAX_SIDE, resample)
    assert base.size == (384, 288)
    for scale in KNOWN_SCALES:
        assert degrade(base, scale, resample).size == base.size
    expected = max(1, int(BASE_MAX_SIDE * KNOWN_SCALES[0]))
    assert effective_size(*base.size, KNOWN_SCALES[0])[0] == expected == 19


# ── Data tests (outputs of scripts 26/27) ──


def test_selection_matches_seeded_shuffle_of_eligible() -> None:
    records = _load_json("selected_images")
    eligible = _load_json("eligible_ids")
    manifest = _load_json("selection_manifest")
    assert sha256_ids(eligible) == manifest["eligible_ids_sha256"]
    _, test_ids, dev_ids = shuffle_and_split(
        eligible, DS["seed"], DS["test_count"], DS["dev_count"]
    )
    by_split = {"test": [], "dev": []}
    for rec in sorted(records, key=lambda r: (r["split"], r["rank"])):
        by_split[rec["split"]].append(rec["image_id"])
    assert by_split["test"] == test_ids
    assert by_split["dev"] == dev_ids


def test_dev_test_disjoint_and_sizes() -> None:
    records = _load_json("selected_images")
    test_ids = {r["image_id"] for r in records if r["split"] == "test"}
    dev_ids = {r["image_id"] for r in records if r["split"] == "dev"}
    assert len(test_ids) == DS["test_count"]
    assert len(dev_ids) == DS["dev_count"]
    assert not test_ids & dev_ids
    assert len({r["sha256"] for r in records}) == len(records)


def test_all_selected_meet_gt_criteria() -> None:
    records = _load_json("selected_images")
    counts = {
        c["image_id"]: c
        for c in count_gt_objects_and_relationships(DS["vg_root"], DS["source_split"])
    }
    for rec in records:
        c = counts[rec["image_id"]]  # also asserts id is in the source split
        assert c["num_objects"] == rec["n_gt_objects"]
        assert c["num_relationships"] == rec["n_gt_relations"]
        assert meets_gt_criteria(
            rec["n_gt_objects"],
            rec["n_gt_relations"],
            DS["min_gt_objects"],
            DS["min_gt_relations"],
        )


def test_level_files_have_base_dims() -> None:
    manifest = _load_json("levels_manifest")
    out_root = resolve(PATHS["degraded_images"])
    assert len(manifest["images"]) == DS["test_count"] + DS["dev_count"]
    for entry in manifest["images"]:
        assert max(entry["base_width"], entry["base_height"]) == BASE_MAX_SIDE
        assert len(entry["level_sha256"]) == NUM_LEVELS
    for entry in manifest["images"][:2]:
        for level in entry["level_sha256"]:
            with Image.open(out_root / str(entry["image_id"]) / f"{level}.jpg") as im:
                assert im.size == (entry["base_width"], entry["base_height"])
                assert im.format == "JPEG"


def test_manifest_level01_effective_size() -> None:
    manifest = _load_json("levels_manifest")
    example = manifest["example_effective_sizes"]
    level01 = example["levels"][0]
    assert level01["level"] == "level_01"
    long_side = max(level01["effective_size"])
    assert long_side == max(1, int(BASE_MAX_SIDE * KNOWN_SCALES[0]))


def _self_test() -> int:
    tests: list[tuple[str, Callable[[], None]]] = [
        (name, fn) for name, fn in globals().items() if name.startswith("test_")
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except SkipTestError as exc:
            print(f"SKIP {name}: {exc}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc!r}")
    print(f"{len(tests) - failures}/{len(tests)} passed or skipped")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    print("Usage: python tests/test_robovis_dataset.py --self-test")
