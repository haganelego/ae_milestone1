"""v2 consistency-stop (plan §17): planner_v2 schema, deterministic
sub-questions, token match, a+b+c score, consecutive-k policy, GT leak for
the new roles, and a synthetic v1 -> v2 dev run ending in `devgrid`."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from gqa_stop_helpers import (
    PROMPTS_DIR,
    FakeTransport,
    answers_by_text,
    fake_tags_get,
    make_context,
    make_world,
    raw_question,
)

from milestone1.gqa_stop import stages_data, stages_model
from milestone1.gqa_stop.client import RequestEngine, load_prompts, settings_from_config
from milestone1.gqa_stop.common import StageError, read_id_csv, read_jsonl
from milestone1.gqa_stop.consistency import (
    LIST_OBJECTS_QUESTION,
    PlanV2Outcome,
    build_subquestions,
    consistency_row,
    image_spec,
    planner_v2_spec,
    recompose_spec,
    target_match,
    validate_plan_v2,
)
from milestone1.gqa_stop.dataset import to_record
from milestone1.gqa_stop.policy import (
    ConsistencyVariant,
    consistency_variants,
    online_stop_consecutive,
    replay_stop_consecutive,
)

V2_ROLES = ["answer_only", "planner_v2", "identify", "attr_blind", "recompose"]


def plan(**kw: Any) -> dict:
    base = {
        "target": "car",
        "synonyms": ["vehicle"],
        "attribute": "color",
        "referring_phrase": "the object that is on the road",
    }
    return {**base, **kw}


# ---- planner_v2 schema -------------------------------------------------------------
def test_plan_v2_valid_and_null_phrase() -> None:
    ok, errors = validate_plan_v2(plan())
    assert errors == [] and ok["referring_phrase"] == "the object that is on the road"
    ok, errors = validate_plan_v2(plan(referring_phrase=None, synonyms=[]))
    assert errors == [] and ok["referring_phrase"] is None


@pytest.mark.parametrize(
    ("override", "needle"),
    [
        ({"target": ""}, "target"),
        ({"synonyms": ["a", "b", "c", "d"]}, "at most"),
        ({"synonyms": "car"}, "synonyms"),
        ({"attribute": "size"}, "attribute"),
        ({"referring_phrase": ""}, "referring_phrase"),
    ],
)
def test_plan_v2_errors(override: dict, needle: str) -> None:
    parsed, errors = validate_plan_v2(plan(**override))
    assert parsed is None and any(needle in e for e in errors), errors


def test_phrase_naming_target_is_flagged_not_rejected() -> None:
    ok, errors = validate_plan_v2(
        plan(
            target="cup",
            synonyms=["mug"],
            referring_phrase="the object that is to the right of the other cup",
        )
    )
    assert errors == [] and ok["phrase_names_target"] is True
    assert validate_plan_v2(plan())[0]["phrase_names_target"] is False


def test_planner_v2_schema_version_in_cache_key(tmp_path: Path) -> None:
    from milestone1.gqa_stop import client

    settings = settings_from_config(
        json.loads(
            (PROMPTS_DIR.parents[1] / "configs" / "gqa_stop_v2.json").read_text()
        )
    )
    engine = RequestEngine(
        settings, load_prompts(PROMPTS_DIR, V2_ROLES), None, tmp_path
    )
    spec = planner_v2_spec(QUESTION)
    reader = image_spec("identify", QUESTION, 1, b"x", "What is it?")
    before, reader_before = engine.cache_key(spec), engine.cache_key(reader)
    client.SCHEMA_VERSIONS["planner_v2"] = "99"
    try:
        assert engine.cache_key(spec) != before
        assert engine.cache_key(reader) == reader_before  # other roles unaffected
    finally:
        client.SCHEMA_VERSIONS["planner_v2"] = "2"


def test_plan_v2_rejects_non_object() -> None:
    assert validate_plan_v2(["car"])[0] is None


# ---- deterministic sub-questions ----------------------------------------------------------
def test_subquestions_with_referring_phrase() -> None:
    sub = build_subquestions(plan())
    assert sub.identify_question == (
        "What is the object that is on the road? Answer with the object name."
    )
    assert sub.attr_question == "What color is the object that is on the road?"
    assert sub.blind_degenerate is False
    assert build_subquestions(plan()) == sub  # deterministic


def test_subquestions_relate0_fallback_is_degenerate() -> None:
    sub = build_subquestions(plan(referring_phrase=None, attribute="material"))
    assert sub.identify_question == LIST_OBJECTS_QUESTION
    assert sub.identify_question == (
        "List the main objects visible in this image, up to five, as single nouns."
    )
    assert sub.attr_question == "What material is the car?"
    assert sub.blind_degenerate is True


# ---- token-level target match ---------------------------------------------------------------
@pytest.mark.parametrize(
    ("answer", "target", "synonyms", "expected"),
    [
        ("cars", "car", [], True),
        ("carpet", "car", [], False),
        ("A Car.", "car", [], True),
        ("tree, cars, road", "car", [], True),
        ("traffic lights", "traffic light", [], True),
        ("traffic", "traffic light", [], False),
        ("glasses", "glass", [], True),
        ("boxes", "box", [], True),
        ("shoes", "shoe", [], True),
        ("automobile", "car", ["automobile"], True),
        ("scar", "car", [], False),
        ("", "car", [], False),
    ],
)
def test_target_match(answer: str, target: str, synonyms: list, expected: bool) -> None:
    assert target_match(answer, target, synonyms) is expected


# ---- score a + b + c -------------------------------------------------------------------------
def rec(answer: str | None, status: str = "valid") -> dict | None:
    if answer is None:
        return None
    return {
        "status": status,
        "parsed": {"answer": answer} if status == "valid" else None,
        "attempt": 1,
        "initial_valid": status == "valid",
    }


def outcome() -> PlanV2Outcome:
    p = validate_plan_v2(plan())[0]
    return PlanV2Outcome("q", "valid", p, True, 1, build_subquestions(p))


QUESTION = {"question_id": "q", "image_id": "1", "question": "What color is the car?"}


@pytest.mark.parametrize(
    ("identify", "attr", "recompose", "reader", "abc"),
    [
        ("car", "Red", "red ", "red", (True, True, True)),
        ("tree", "red", "blue", "red", (False, True, False)),
        ("cars", "blue", "red", "red", (True, False, True)),
        ("blob", "blue", "blue", "red", (False, False, False)),
    ],
)
def test_score_is_sum_of_components(identify, attr, recompose, reader, abc) -> None:
    row = consistency_row(
        QUESTION, 3, outcome(), rec(reader), rec(identify), rec(attr), rec(recompose)
    )
    assert (row["a"], row["b"], row["c"]) == abc
    assert row["consistency_score"] == sum(abc)


def test_failed_calls_score_zero() -> None:
    row = consistency_row(
        QUESTION,
        3,
        outcome(),
        rec("red", "schema_failure"),
        rec("car"),
        rec("red"),
        None,
    )
    assert (row["a"], row["b"], row["c"]) == (True, False, False)
    assert row["recompose_status"] == "skipped"
    failed = PlanV2Outcome("q", "planner_failure")
    row = consistency_row(QUESTION, 3, failed, rec("red"), None, None, None)
    assert row["consistency_score"] == 0 and row["identify_status"] == "planner_failure"


# ---- consecutive-k policy on prefixes ----------------------------------------------------------
def test_consecutive_k_requires_a_run() -> None:
    scores = [0, 2, 0, 2, 2, 3] + [3] * 14
    assert replay_stop_consecutive(scores, ConsistencyVariant(2, 1)).level == 2
    assert replay_stop_consecutive(scores, ConsistencyVariant(2, 2)).level == 5
    assert replay_stop_consecutive(scores, ConsistencyVariant(3, 2)).level == 7
    d = replay_stop_consecutive([1] * 20, ConsistencyVariant(2, 1))
    assert d.level == 20 and d.never_met


def test_consecutive_policy_reads_prefix_only() -> None:
    seen: list[int] = []
    scores = [0, 3, 3, 0] + [3] * 16

    def at(t: int) -> int:
        seen.append(t)
        return scores[t - 1]

    assert online_stop_consecutive(at, 3, 2, 20).level == 3
    assert seen == [1, 2, 3]


def test_k2_can_stop_at_last_level_naturally() -> None:
    scores = [0] * 18 + [3, 3]
    d = replay_stop_consecutive(scores, ConsistencyVariant(3, 2))
    assert d.level == 20 and not d.never_met and d.passed_at_last


def test_variant_grid_validation() -> None:
    cfg = {
        "start_level": 1,
        "force_stop_level": 20,
        "thetas": [1, 2, 3],
        "consecutive_k_values": [1, 2],
    }
    labels = [v.label for v in consistency_variants(cfg, 20)]
    assert labels == [f"consistency@theta{t}_k{k}" for t in (1, 2, 3) for k in (1, 2)]
    for bad in (
        {"consecutive_k_values": [3]},
        {"thetas": [4]},
        {"start_level": 2},
        {"force_stop_level": 19},
    ):
        with pytest.raises(ValueError):
            consistency_variants({**cfg, **bad}, 20)


# ---- GT leak for the new roles -----------------------------------------------------------------
def test_new_roles_receive_no_gt_and_question_only_where_allowed(
    tmp_path: Path,
) -> None:
    raw = raw_question(
        "4242", "What color is the mug on the table?", "chartreuse", "color", 1, "mug"
    )
    raw["fullAnswer"] = "The mug is definitely chartreuse."
    record = to_record("q1", raw)
    full = {
        **record.to_dict(),
        "semantic": raw["semantic"],
        "imageId": "4242",
        "annotations": {"answer": {"0": "ANNOT9"}},
        "fullAnswer": raw["fullAnswer"],
    }
    sub = build_subquestions(
        plan(
            target="mug",
            synonyms=[],
            referring_phrase="the object that is on the table",
        )
    )
    specs = {
        "planner_v2": planner_v2_spec(full),
        "identify": image_spec("identify", full, 5, b"img", sub.identify_question),
        "attr_blind": image_spec("attr_blind", full, 5, b"img", sub.attr_question),
        "recompose": recompose_spec(full, 5, sub, "mug", "blue"),
    }
    settings = settings_from_config(
        json.loads(
            (PROMPTS_DIR.parents[1] / "configs" / "gqa_stop_v2.json").read_text()
        )
    )
    engine = RequestEngine(
        settings, load_prompts(PROMPTS_DIR, V2_ROLES), None, tmp_path
    )
    forbidden = [
        "chartreuse",
        "definitely",
        "ANNOT9",
        "4242",
        "table,on,s",
        "select",
        "directOf",
        "gqa_val_balanced",
        "q1",
    ]
    for role, spec in specs.items():
        text = json.dumps(engine.build_payload(spec, engine.build_messages(spec)))
        for word in forbidden:
            assert word not in text, (role, word)
        has_question = "What color is the mug on the table?" in text
        assert has_question is (role in {"planner_v2", "recompose"}), role
    assert specs["planner_v2"].image_bytes is None
    assert specs["recompose"].image_bytes is None
    assert set(specs["recompose"].input_data) == {"question", "observations"}


# ---- synthetic v1 -> v2 dev run ------------------------------------------------------------------
def v2_config(v1_cfg: dict) -> dict:
    cfg = copy.deepcopy(v1_cfg)
    cfg["run_id"] = "synthetic_v2"
    cfg["paths"].update(
        {
            "output_root": "outputs/run_v2",
            "selection_root": "data/run_v2",
            "inherit_split_from": "outputs/run",
            "reuse_records_from": {
                "output_root": "outputs/run",
                "roles": ["answer_only"],
            },
        }
    )
    cfg["prompts"] = {"roles": V2_ROLES}
    cfg["model"]["num_predict"].update(
        {"planner_v2": 512, "identify": 128, "attr_blind": 128, "recompose": 128}
    )
    cfg["policy"] = {
        "family": "consistency",
        "methods": ["consistency"],
        "thetas": [1, 2, 3],
        "consecutive_k_values": [1, 2],
        "start_level": 1,
        "consecutive_k": 1,
        "force_stop_level": 20,
        "primary_dev_accuracy_target": 0.95,
        "secondary_dev_accuracy_targets": [0.9, 0.97, 0.99],
    }
    cfg["comparison"] = {"v1_output_root": "outputs/run", "v1_method": "checklist_min"}
    return cfg


@pytest.fixture(scope="module")
def v2_run(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("v2")
    config_path, cfg, questions = make_world(tmp)
    t1 = FakeTransport(None, answers_by_text(questions), questions)
    v1 = make_context(config_path, cfg, t1)
    t1.paths = v1.paths
    stages_model.preflight(v1, get=fake_tags_get)
    stages_data.prepare(v1)
    stages_data.screen(v1, "dev")
    stages_model.pilot(v1, [1, 5, 10, 15, 20], calibration=True, use_fallback=False)
    stages_model.infer(v1, "dev", list(range(1, 21)))
    stages_model.freeze(v1)
    v1_manifest = v1.paths.data("selected_manifest.jsonl").read_text()
    v2_path = tmp / "gqa_stop_v2.json"
    v2_path.write_text(json.dumps(v2_config(cfg)))
    t2 = FakeTransport(None, answers_by_text(questions), questions)
    v2 = make_context(v2_path, {}, t2)
    t2.paths = v2.paths
    stages_model.preflight(v2, get=fake_tags_get)
    stages_data.screen(v2, "dev")
    reader_calls_after_screen = t2.calls["answer_only"]
    stages_model.pilot(v2, [1, 5, 10, 15, 20], calibration=False, use_fallback=False)
    stages_model.infer(v2, "dev", list(range(1, 21)))
    summary = stages_model.devgrid(v2)
    return v1, v2, t2, summary, v1_manifest, reader_calls_after_screen


def test_v2_dev_set_equals_v1_and_shared_data_untouched(v2_run) -> None:
    v1, v2, _, _, v1_manifest, _ = v2_run
    assert [r["question_id"] for r in v2.selected("dev")] == [
        r["question_id"] for r in v1.selected("dev")
    ]
    assert v2.state.get("screening")["dev"]["reference_match"] is True
    assert v1.paths.data("selected_manifest.jsonl").read_text() == v1_manifest
    assert v2.paths.selection("selected_manifest.jsonl").exists()


def test_v2_reuses_identical_v1_reader_records(v2_run) -> None:
    _, v2, t2, _, _, after_screen = v2_run
    assert after_screen == 0 and t2.calls["answer_only"] == 0  # all copied from v1
    copied = list(v2.paths.out("reader").rglob("*.json"))
    assert copied and all(json.loads(p.read_text()).get("copied_from") for p in copied)
    assert "reuse_root:" in v2.paths.out("worklog.md").read_text()


def test_v2_devgrid_outputs(v2_run) -> None:
    _, v2, t2, summary, _, _ = v2_run
    grid = pd.read_csv(v2.paths.out("dev_threshold_grid.csv"))
    assert list(grid["variant"]) == [
        f"consistency@theta{t}_k{k}" for t in (1, 2, 3) for k in (1, 2)
    ]
    for column in (
        "accuracy",
        "mean_level",
        "median_level",
        "stop20_rate",
        "stopped_but_wrong",
    ):
        assert column in grid
    hist = pd.read_csv(v2.paths.out("dev_stop_hist.csv"))
    assert hist.groupby("variant")["n_stopped"].sum().eq(len(v2.selected("dev"))).all()
    comp = pd.read_csv(v2.paths.out("dev_components.csv"))
    assert list(comp["level"]) == list(range(1, 21))
    for column in (
        "a_rate",
        "b_rate",
        "c_rate",
        "abc_rate",
        "p_a_given_wrong",
        "p_s_ge_1_and_wrong",
        "p_s_ge_3_and_wrong",
    ):
        assert column in comp
    comparison = read_id_csv(v2.paths.out("dev_comparison.csv"))
    assert comparison["method"].str.startswith("v1 checklist_min@0.95").any()
    curve = pd.read_csv(v2.paths.out("dev_fixed_curve.csv"))
    assert (curve["v1_reader_accuracy"] == curve["v2_reader_accuracy"]).all()
    assert summary["frozen_policy_written"] is False
    assert not v2.paths.out("frozen_policy.json").exists()
    assert set(t2.calls) == {"planner_v2", "identify", "attr_blind", "recompose"}


def test_v2_pilot_examples_and_format_rates(v2_run) -> None:
    _, v2, _, _, _, _ = v2_run
    summary = json.loads(v2.paths.out("pilot_summary.json").read_text())
    assert set(summary["valid_rate_initial"]) == {
        "reader",
        "identify",
        "attr_blind",
        "recompose",
        "planner",
    }
    examples = json.loads(v2.paths.out("pilot/pilot_examples.json").read_text())
    pilot_n = v2.cfg["execution"]["pilot_questions"]  # 4 in the synthetic world
    assert len(examples) == min(5, pilot_n) and {"L1", "L20"} <= set(examples[0])


def test_v2_test_side_and_freeze_commands_refuse(v2_run) -> None:
    _, v2, _, _, _, _ = v2_run
    with pytest.raises(StageError):
        stages_model.freeze(v2)
    with pytest.raises(StageError):
        stages_data.screen(v2, "test")
    with pytest.raises(StageError):
        stages_model.infer(v2, "test", [1])
    with pytest.raises(StageError):
        stages_model.run_evaluate(v2)
    with pytest.raises(StageError):
        stages_data.prepare(v2)


def test_v2_reference_mismatch_stops(v2_run) -> None:
    _, v2, _, _, _, _ = v2_run
    adopted = v2.selected("dev")
    allocation = json.loads((v2.inherit_root / "adoption_dev.json").read_text())
    assert stages_data.reference_mismatch(v2, "dev", adopted, allocation) is None
    assert stages_data.reference_mismatch(v2, "dev", adopted[1:], allocation)
    changed = {**allocation, "targets": {**allocation["targets"], "color|r0": 0}}
    assert stages_data.reference_mismatch(v2, "dev", adopted, changed)


def test_v1_selected_manifest_rows(v2_run) -> None:
    v1, _, _, _, _, _ = v2_run
    assert read_jsonl(v1.paths.data("selected_manifest.jsonl"))
