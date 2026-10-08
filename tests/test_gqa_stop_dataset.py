"""Eligibility, duplicates, deterministic stratified scan, adoption (plan §3, §16 A1/A2).

Mandatory tests 1 (duplicates never cross splits) and 2 (fixed seed -> same order
and same adopted set) plus the §16 A2 quota / shortfall-reallocation rules.
"""

from __future__ import annotations

import json
from pathlib import Path

from gqa_stop_helpers import (
    CONFIG_PATH,
    DUPLICATE_PAIR,
    make_questions,
    raw_question,
    write_images,
)

from milestone1.gqa_stop import dataset, sampling

CFG = json.loads(CONFIG_PATH.read_text())
FILTER = CFG["dataset"]["filter"]
STRATA = sampling.parse_strata(CFG["dataset"]["stratification"])
SEED = 20261008


def pool_and_groups(tmp_path: Path, seed: int = SEED):
    questions = make_questions()
    write_images(tmp_path)
    pool = dataset.build_eligible_pool(questions, FILTER)
    hashes = dataset.compute_image_hashes([r.image_id for r in pool.records], tmp_path)
    groups = sampling.build_groups(pool.records, dataset.duplicate_groups(hashes), seed)
    return pool, groups


def scan_both(groups, dev_q=24, test_q=40, dev_cap=None, test_cap=None):
    """Same protocol as `prepare`: A2-bis quotas, dev first, test from dev cursor."""
    dev_quota, test_quota = sampling.split_quotas(
        STRATA, sampling.cell_counts(groups), dev_q, test_q, 100, 200
    )
    dev = sampling.scan_split(groups, "dev", "main", dev_quota, dev_cap, 2, set())
    test = sampling.scan_split(
        groups,
        "test",
        "main",
        test_quota,
        test_cap,
        2,
        set(dev.groups_used),
        start_index=dev.end_index,
    )
    return dev, test


# ---- eligibility (§16 A1) -------------------------------------------------------
def test_classify_uses_final_query_argument_not_detailed_type() -> None:
    raw = raw_question("1", "What color is the cup?", "red", "color", 0, "cup")
    raw["types"]["detailed"] = "directWhich"
    assert dataset.classify(raw, FILTER) == dataset.ELIGIBLE
    raw["semantic"][-1]["argument"] = "name"
    assert dataset.classify(raw, FILTER) == "final_argument_not_target_attribute"


def test_classify_exclusion_reasons() -> None:
    base = raw_question("1", "What color is the cup?", "red", "color", 0, "cup")
    negated = json.loads(json.dumps(base))
    negated["semantic"].insert(
        1, {"operation": "filter", "dependencies": [0], "argument": "not(small)"}
    )
    yes = {**base, "answer": "yes"}
    numeric = {**base, "answer": "3"}
    empty = {**base, "answer": " "}
    mismatch = {**base, "question": "What is the boy doing?"}
    evident = {**base, "question": "What color is the red cup?"}
    forbidden = json.loads(json.dumps(base))
    forbidden["semantic"].insert(
        1, {"operation": "exist", "dependencies": [0], "argument": "?"}
    )
    assert dataset.classify(negated, FILTER) == "negation"
    assert dataset.classify(yes, FILTER) == "answer_yes_no"
    assert dataset.classify(numeric, FILTER) == "answer_numeric"
    assert dataset.classify(empty, FILTER) == "answer_empty"
    assert dataset.classify(mismatch, FILTER) == "question_text_attribute_mismatch"
    assert dataset.classify(evident, FILTER) == "answer_word_in_question"
    assert dataset.classify(forbidden, FILTER) == "forbidden_operation"


def test_material_text_variants_are_accepted() -> None:
    raw = raw_question("1", "What makes up the fence?", "wood", "material", 0, "fence")
    assert dataset.classify(raw, FILTER) == dataset.ELIGIBLE


def test_schema_mapping_records_counts_and_string_image_ids() -> None:
    questions = make_questions()
    questions["777"] = {**questions["100000"], "imageId": 1000}
    pool = dataset.build_eligible_pool(questions, FILTER)
    mapping = pool.schema_mapping
    assert mapping["total_questions"] == len(questions)
    assert sum(mapping["exclusion_counts"].values()) == len(questions)
    assert mapping["eligible_questions"] == len(pool.records)
    assert all(isinstance(r.image_id, str) for r in pool.records)


# ---- duplicates (test 1) ------------------------------------------------------------
def test_duplicate_bytes_share_group_and_split(tmp_path: Path) -> None:
    _, groups = pool_and_groups(tmp_path)
    by_image = {i: g.group_id for g in groups for i in g.image_ids}
    assert by_image[DUPLICATE_PAIR[0]] == by_image[DUPLICATE_PAIR[1]]
    dev, test = scan_both(groups)
    dev_images = {i for c in dev.candidates for i in c["group_image_ids"]}
    test_images = {i for c in test.candidates for i in c["group_image_ids"]}
    assert not dev_images & test_images
    pair = set(DUPLICATE_PAIR)
    assert (
        pair <= dev_images
        or pair <= test_images
        or not pair & (dev_images | test_images)
    )


def test_same_image_questions_never_cross_splits(tmp_path: Path) -> None:
    _, groups = pool_and_groups(tmp_path)
    dev, test = scan_both(groups)
    assert not {c["image_id"] for c in dev.candidates} & {
        c["image_id"] for c in test.candidates
    }
    assert set(dev.groups_used).isdisjoint(test.groups_used)


# ---- determinism (test 2) -------------------------------------------------------------
def test_fixed_seed_gives_same_candidates_and_adoption(tmp_path: Path) -> None:
    _, groups_a = pool_and_groups(tmp_path / "a")
    _, groups_b = pool_and_groups(tmp_path / "b")
    dev_a, test_a = scan_both(groups_a)
    dev_b, test_b = scan_both(groups_b)
    assert dev_a.candidates == dev_b.candidates
    assert test_a.candidates == test_b.candidates
    passed = {c["question_id"] for c in dev_a.candidates[::2]}
    adopted_a, _ = sampling.adopt(dev_a.candidates, passed, 8, STRATA, SEED)
    adopted_b, _ = sampling.adopt(dev_b.candidates, passed, 8, STRATA, SEED)
    assert adopted_a == adopted_b


def test_different_seed_changes_order(tmp_path: Path) -> None:
    _, groups_a = pool_and_groups(tmp_path / "a", seed=SEED)
    _, groups_b = pool_and_groups(tmp_path / "b", seed=SEED + 1)
    assert [g.group_id for g in groups_a] != [g.group_id for g in groups_b]


def test_groups_follow_seeded_hash_order(tmp_path: Path) -> None:
    _, groups = pool_and_groups(tmp_path)
    keys = [g.order_key for g in groups]
    assert keys == sorted(keys)
    for group in groups:
        qkeys = [sampling.question_order_key(SEED, q) for q in group.questions]
        assert qkeys == sorted(qkeys)


# ---- scan rules -----------------------------------------------------------------------
def test_scan_caps_images_and_questions_per_image(tmp_path: Path) -> None:
    _, groups = pool_and_groups(tmp_path)
    dev, _ = scan_both(groups, dev_q=100, dev_cap=5)
    assert len(dev.groups_used) <= 5
    per_image = {}
    for c in dev.candidates:
        per_image[c["group_id"]] = per_image.get(c["group_id"], 0) + 1
    assert max(per_image.values()) <= 2


def test_pick_prefers_different_attributes() -> None:
    qs = [
        dataset.QuestionRecord(
            "q1", "1", "What color a?", "color", 0, 2, "", "red", ""
        ),
        dataset.QuestionRecord(
            "q2", "1", "What color b?", "color", 0, 2, "", "red", ""
        ),
        dataset.QuestionRecord(
            "q3", "1", "What material c?", "material", 0, 2, "", "wood", ""
        ),
    ]
    group = sampling.ImageGroup("1", ("1",), tuple(qs), "k")
    picked = sampling.pick_from_group(group, {"color|r0": 5, "material|r0": 5}, 2)
    assert [p.question_id for p in picked] == ["q1", "q3"]
    same = sampling.pick_from_group(group, {"color|r0": 5, "material|r0": 0}, 2)
    assert [p.question_id for p in same] == ["q1", "q2"]


def test_scan_respects_cell_quotas(tmp_path: Path) -> None:
    _, groups = pool_and_groups(tmp_path)
    dev, _ = scan_both(groups)
    for cell, filled in dev.filled().items():
        assert filled <= dev.quotas.targets[cell]


# ---- allocation / shortfall reallocation (§16 A2) -----------------------------------
def test_allocate_initial_shares() -> None:
    plenty = dict.fromkeys(STRATA.cells, 1000)
    alloc = sampling.allocate(100, STRATA, plenty)
    t = alloc.targets
    assert t["color|r0"] == t["color|r1"] == 20
    assert t["material|r0"] == t["material|r1"] == 20
    assert t["shape|r0"] == t["shape|r1"] == 10
    assert alloc.unfilled == 0


def test_shortfall_goes_to_same_attribute_first() -> None:
    avail = dict.fromkeys(STRATA.cells, 1000)
    avail["material|r1"] = 5
    t = sampling.allocate(100, STRATA, avail).targets
    assert t["material|r1"] == 5
    assert t["material|r0"] == 35
    assert t["color|r0"] == t["color|r1"] == 20


def test_shortfall_then_other_attributes_proportional_and_shape_capped() -> None:
    avail = dict.fromkeys(STRATA.cells, 1000)
    avail.update({"shape|r0": 3, "shape|r1": 1})
    alloc = sampling.allocate(100, STRATA, avail)
    t = alloc.targets
    assert t["shape|r0"] == 3 and t["shape|r1"] == 1
    assert t["color|r0"] + t["color|r1"] == 48  # 40 + half of the 16 shortfall
    assert t["material|r0"] + t["material|r1"] == 48
    assert sum(t.values()) == 100
    avail = dict.fromkeys(STRATA.cells, 1000)
    avail.update({"color|r0": 0, "color|r1": 0})
    t2 = sampling.allocate(100, STRATA, avail).targets
    assert t2["shape|r0"] + t2["shape|r1"] == 20  # capped attribute never receives
    assert t2["material|r0"] + t2["material|r1"] == 80


def test_allocation_reports_unfilled_when_supply_exhausted() -> None:
    avail = dict.fromkeys(STRATA.cells, 2)
    alloc = sampling.allocate(100, STRATA, avail)
    assert sum(alloc.targets.values()) == 12
    assert alloc.unfilled == 88
    assert any("unfilled" in line for line in alloc.log)


def test_adopt_takes_hash_order_per_cell_with_reallocation(tmp_path: Path) -> None:
    _, groups = pool_and_groups(tmp_path)
    dev, _ = scan_both(groups)
    passed = {c["question_id"] for c in dev.candidates if c["cell"] != "material|r1"}
    adopted, alloc = sampling.adopt(dev.candidates, passed, 10, STRATA, SEED)
    assert all(r["question_id"] in passed for r in adopted)
    assert alloc.targets["material|r1"] == 0
    keys = [sampling.seeded_hash(SEED, r["question_id"]) for r in adopted]
    assert keys == sorted(keys)
    for cell, n in alloc.targets.items():
        in_cell = sorted(
            (
                c
                for c in dev.candidates
                if c["cell"] == cell and c["question_id"] in passed
            ),
            key=lambda c: sampling.seeded_hash(SEED, c["question_id"]),
        )
        chosen = {r["question_id"] for r in adopted if r["cell"] == cell}
        assert chosen == {c["question_id"] for c in in_cell[:n]}


def test_meets_minimum_counts_images() -> None:
    rows = [{"group_id": "a"}, {"group_id": "a"}, {"group_id": "b"}]
    assert sampling.meets_minimum(rows, 3, 2)
    assert not sampling.meets_minimum(rows, 3, 3)


# ---- A2-bis: split-aware quotas, no image cap ------------------------------------------
REAL_SIZES = {
    "color|r0": 4750,
    "color|r1": 2909,
    "material|r0": 1231,
    "material|r1": 146,
    "shape|r0": 76,
    "shape|r1": 36,
}


def test_rare_cells_split_one_third_dev_two_thirds_test() -> None:
    dev, test = sampling.split_quotas(STRATA, REAL_SIZES, 300, 800, 100, 200)
    assert dev.targets == {
        "color|r0": 60,
        "color|r1": 60,
        "material|r0": 60,
        "material|r1": 48,  # floor(146 / 3)
        "shape|r0": 25,  # floor(76 / 3)
        "shape|r1": 12,  # 36 / 3 exactly (integer arithmetic)
    }
    assert test.targets == {
        "color|r0": 160,
        "color|r1": 160,
        "material|r0": 160,
        "material|r1": 98,
        "shape|r0": 51,
        "shape|r1": 24,
    }
    for cell in ("material|r1", "shape|r0", "shape|r1"):
        assert test.targets[cell] >= 2 * REAL_SIZES[cell] // 3
        assert dev.targets[cell] + test.targets[cell] <= REAL_SIZES[cell]
    assert any("A2-bis" in line for line in dev.log)


def test_dense_cells_bound_by_configured_share() -> None:
    dev, test = sampling.split_quotas(STRATA, REAL_SIZES, 300, 800, 100, 200)
    share_dev = sampling.share_quotas(300, STRATA)
    share_test = sampling.share_quotas(800, STRATA)
    for cell in ("color|r0", "color|r1", "material|r0"):
        assert dev.targets[cell] == share_dev[cell]
        assert test.targets[cell] == share_test[cell]


def test_split_fraction_comes_from_targets() -> None:
    sizes = dict.fromkeys(STRATA.cells, 40)
    dev, _ = sampling.split_quotas(STRATA, sizes, 1000, 1000, 100, 100)
    assert set(dev.targets.values()) == {20}  # fraction 1/2 binds every cell


def test_no_image_cap_scans_until_quotas_filled(tmp_path: Path) -> None:
    _, groups = pool_and_groups(tmp_path)
    dev, _ = scan_both(groups)
    assert dev.filled() == {c: q for c, q in dev.quotas.targets.items() if q}
    capped, _ = scan_both(groups, dev_cap=3)
    assert len(capped.groups_used) == 3 and len(dev.groups_used) > 3


def test_no_image_cap_runs_until_pool_exhausted(tmp_path: Path) -> None:
    _, groups = pool_and_groups(tmp_path)
    dev, test = scan_both(groups, dev_q=24, test_q=10_000)
    assert test.end_index == len(groups)  # quotas unfillable -> whole pool walked
    assert dev.groups_scanned + test.groups_scanned == len(groups)


def test_test_scan_continues_from_dev_cursor(tmp_path: Path) -> None:
    _, groups = pool_and_groups(tmp_path)
    dev, test = scan_both(groups)
    assert test.start_index == dev.end_index
    position = {g.group_id: i for i, g in enumerate(groups)}
    assert all(position[g] >= dev.end_index for g in test.groups_used)
    assert all(position[g] < dev.end_index for g in dev.groups_used)
    assert set(dev.groups_used).isdisjoint(test.groups_used)
