"""Oracle checklist plans from GQA programs and their per-level aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from milestone1.gqa_stop.oracle_plan import (
    EXISTENCE,
    FINAL,
    answer_leak,
    build_oracle_plan,
    join_options,
    option_order,
    strip_ref,
)
from milestone1.gqa_stop.oracle_score import gated_scores, object_absent, score_level

SELECTION = (
    Path(__file__).resolve().parents[1]
    / "outputs/gqa_stop/probe_multi_rel/selection.jsonl"
)


def op(operation: str, argument: str = "", deps: list[int] | None = None) -> dict:
    return {"operation": operation, "argument": argument, "dependencies": deps or []}


def record(question: str, program: list[dict], qid: str = "q1") -> dict[str, Any]:
    return {"question_id": qid, "question": question, "semantic": program}


def texts(plan: dict[str, Any]) -> list[str]:
    return [i["question"] for i in plan["items"]]


def sig(answer: str, pans: float, resolved: bool = True, valid: bool = True) -> dict:
    return {"valid": valid, "answer": answer, "resolved": resolved, "pans": pans}


# ---- plan builder ----------------------------------------------------------------------------
def test_build_relo_query_checks_anchor_and_related_object() -> None:
    # Arrange
    rec = record(
        "What is the man wearing?",
        [
            op("select", "man (884411)"),
            op("relate", "_,wearing,o (884423)", [0]),
            op("query", "name", [1]),
        ],
    )
    # Act
    plan = build_oracle_plan(rec)
    # Assert
    assert texts(plan) == [
        "Is there a man in the image?",
        "Is the man wearing something?",
        "What is the man wearing?",
    ]
    assert [i["role"] for i in plan["items"]] == [EXISTENCE, EXISTENCE, FINAL]
    assert [i["required_answer"] for i in plan["items"]] == ["yes", "yes", None]
    assert [i["depends_on"] for i in plan["items"]] == [[], ["o1"], ["o2"]]
    assert plan["sensible"] is True


def test_build_rels_query_with_position_filter() -> None:
    rec = record(
        "What is on the ski on the left?",
        [
            op("select", "ski (188288)"),
            op("filter hposition", "left", [0]),
            op("relate", "_,on,s (188289)", [1]),
            op("query", "name", [2]),
        ],
    )
    plan = build_oracle_plan(rec)
    assert texts(plan) == [
        "Is there a ski on the left in the image?",
        "Is there something on the ski on the left?",
        "What is on the ski on the left?",
    ]


def test_build_person_and_place_targets_use_who_and_where() -> None:
    who = build_oracle_plan(
        record(
            "The crowd is watching who?",
            [
                op("select", "crowd (1)"),
                op("relate", "person,watching,o (2)", [0]),
                op("query", "name", [1]),
            ],
        )
    )
    where = build_oracle_plan(
        record(
            "Where are the trees?",
            [
                op("select", "trees (1)"),
                op("relate", "place,in,o (2)", [0]),
                op("query", "name", [1]),
            ],
        )
    )
    assert texts(who)[1:] == [
        "Is the crowd watching someone?",
        "Who is the crowd watching?",
    ]
    assert texts(where)[0] == "Are there trees in the image?"
    assert texts(where)[2] == "Where are the trees?"


def test_build_choose_rel_orders_options_by_question_not_program() -> None:
    # GQA lists the correct option first ("below|above"); the question says "above or below"
    rec = record(
        "Is the counter above or below the cabinet the microwave is to the left of?",
        [
            op("select", "microwave (1)"),
            op("relate", "cabinet,to the left of,o (2)", [0]),
            op("choose rel", "counter,below|above,s (3)", [1]),
        ],
    )
    plan = build_oracle_plan(rec)
    assert texts(plan) == [
        "Is there a microwave in the image?",
        "Is the microwave to the left of a cabinet?",
        "Is there a counter in the image?",
        "Is the counter above or below the cabinet that the microwave is to the left of?",
    ]
    assert plan["items"][-1]["depends_on"] == ["o2", "o3"]


def test_build_choose_rel_joins_shared_suffix() -> None:
    rec = record(
        "Is the elephant to the left or to the right of the car?",
        [
            op("select", "car (1)"),
            op("choose rel", "elephant,to the right of|to the left of,s (2)", [0]),
        ],
    )
    plan = build_oracle_plan(rec)
    assert texts(plan)[-1] == "Is the elephant to the left or to the right of the car?"


def test_build_verify_rel_maps_pronoun_and_strips_absence_marker() -> None:
    rec = record(
        "Is he carrying a surfboard?",
        [
            op("select", "he (776863)"),
            op("verify rel", "surfboard,carrying,o (-)", [0]),
        ],
    )
    plan = build_oracle_plan(rec)
    assert texts(plan) == [
        "Is there a man in the image?",
        "Is the man carrying a surfboard?",
    ]
    assert "pronoun: he" in plan["flags"]
    assert not any("(-)" in t or "(" in t for t in texts(plan))


def test_build_exist_after_relate_asserts_the_related_object() -> None:
    rec = record(
        "Are there any books on the floor?",
        [
            op("select", "floor (1)"),
            op("relate", "books,on,s (2)", [0]),
            op("exist", "?", [1]),
        ],
    )
    plan = build_oracle_plan(rec)
    assert texts(plan) == [
        "Is there a floor in the image?",
        "Are there books on the floor?",
    ]
    assert plan["items"][1]["role"] == FINAL
    assert plan["items"][1]["required_answer"] is None


def test_build_or_gives_logic_items_without_required_answer() -> None:
    rec = record(
        "Do you see any doors or cars that are white?",
        [
            op("select", "door (1)"),
            op("filter color", "white", [0]),
            op("exist", "?", [1]),
            op("select", "car (-) "),
            op("filter color", "white", [3]),
            op("exist", "?", [4]),
            op("or", "", [2, 5]),
        ],
    )
    plan = build_oracle_plan(rec)
    assert texts(plan) == [
        "Is there a white door in the image?",
        "Is there a white car in the image?",
    ]
    assert {i["logic"] for i in plan["items"]} == {"or"}
    assert all(
        i["role"] == FINAL and i["required_answer"] is None for i in plan["items"]
    )


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("same material", "Are the barrier and the pole made of the same material?"),
        ("same color", "Do the barrier and the pole have the same color?"),
        ("different color", "Do the barrier and the pole have different colors?"),
        ("common", "What is common to the barrier and the pole?"),
    ],
)
def test_build_compare_ops_check_both_objects(operation: str, expected: str) -> None:
    rec = record(
        "q?",
        [
            op("select", "barrier (1)"),
            op("select", "pole (2)"),
            op(operation, "", [0, 1]),
        ],
    )
    plan = build_oracle_plan(rec)
    assert texts(plan) == [
        "Is there a barrier in the image?",
        "Is there a pole in the image?",
        expected,
    ]
    assert plan["items"][-1]["depends_on"] == ["o1", "o2"]


def test_build_attribute_query_verify_and_choose() -> None:
    base = [op("select", "shirt (1)")]
    query = build_oracle_plan(
        record("What color is the shirt?", [*base, op("query", "color", [0])])
    )
    verify = build_oracle_plan(
        record("Is the shirt red?", [*base, op("verify color", "red", [0])])
    )
    choose = build_oracle_plan(
        record(
            "Is the shirt red or blue?", [*base, op("choose color", "blue|red", [0])]
        )
    )
    assert texts(query)[-1] == "What color is the shirt?"
    assert texts(verify)[-1] == "Is the shirt red?"
    assert texts(choose)[-1] == "Is the shirt red or blue?"


def test_build_unsupported_op_gives_empty_flagged_plan() -> None:
    rec = record(
        "Which is taller?",
        [op("select", "a (1)"), op("select", "b (2)"), op("choose taller", "", [0, 1])],
    )
    plan = build_oracle_plan(rec)
    assert plan["items"] == []
    assert plan["sensible"] is False
    assert any(f.startswith("unsupported") for f in plan["flags"])


def test_strip_ref_and_options_helpers() -> None:
    assert strip_ref("bottle cap (-)  ") == "bottle cap"
    assert strip_ref("man (884411)") == "man"
    assert option_order(["below", "above"], "Is it above or below the x?") == [
        "above",
        "below",
    ]
    assert join_options(["above", "below"]) == "above or below"


def test_answer_leak_detects_gt_word_absent_from_question() -> None:
    plan = {
        "question": "What is the man wearing?",
        "items": [{"question": "Is the man wearing a hat?"}],
    }
    assert answer_leak(plan, "hat") == ["Is the man wearing a hat?"]
    assert (
        answer_leak(
            {**plan, "items": [{"question": "What is the man wearing?"}]}, "hat"
        )
        == []
    )


@pytest.mark.skipif(not SELECTION.exists(), reason="probe selection not present")
def test_probe_questions_all_get_sensible_plans_without_leaks() -> None:
    rows = [json.loads(line) for line in SELECTION.open()]
    plans = [build_oracle_plan(r) for r in rows]
    assert len(plans) == 30
    assert all(p["sensible"] for p in plans)
    assert all(2 <= len(p["items"]) <= 4 for p in plans)
    assert not any(
        answer_leak(p, str(r["answer"])) for p, r in zip(plans, rows, strict=True)
    )


# ---- aggregation -----------------------------------------------------------------------------
def items_of(*specs: tuple[str, str, str | None, str | None, list[str]]) -> list[dict]:
    return [
        {
            "id": i,
            "role": role,
            "required_answer": req,
            "logic": logic,
            "depends_on": deps,
        }
        for i, role, req, logic, deps in specs
    ]


EXIST_THEN_FINAL = items_of(
    ("o1", EXISTENCE, "yes", None, []),
    ("o2", FINAL, None, None, ["o1"]),
)
OR_PAIR = items_of(("o1", FINAL, None, "or", []), ("o2", FINAL, None, "or", []))
AND_PAIR = items_of(("o1", FINAL, None, "and", []), ("o2", FINAL, None, "and", []))


def test_gated_scores_zero_unknown_unresolved_and_failed_dependency() -> None:
    signals = {"o1": sig("unknown", 0.9), "o2": sig("red", 0.8)}
    assert gated_scores(EXIST_THEN_FINAL, signals) == {"o1": 0.0, "o2": 0.0}
    signals = {"o1": sig("yes", 0.9, resolved=False), "o2": sig("red", 0.8)}
    assert gated_scores(EXIST_THEN_FINAL, signals) == {"o1": 0.0, "o2": 0.0}
    signals = {"o1": sig("yes", 0.9), "o2": sig("red", 0.8)}
    assert gated_scores(EXIST_THEN_FINAL, signals) == {"o1": 0.9, "o2": 0.8}


def test_score_level_veto_on_resolved_no_for_required_yes() -> None:
    # Arrange: the presupposed object is reported absent, the final item is confident
    signals = {"o1": sig("no", 0.95), "o2": sig("red", 0.9)}
    # Act
    out = score_level(EXIST_THEN_FINAL, signals)
    # Assert
    assert object_absent(EXIST_THEN_FINAL, signals) is True
    assert out["object_absent"] is True
    assert out["oracle_logic"] == 0.0
    assert out["oracle_kleene"] == 0.0
    assert out["oracle_logic_noveto"] == pytest.approx(0.9)
    assert out["oracle_min"] == pytest.approx(0.9)  # plain min has no veto


def test_score_level_unresolved_no_does_not_veto() -> None:
    signals = {"o1": sig("no", 0.95, resolved=False), "o2": sig("red", 0.9)}
    out = score_level(EXIST_THEN_FINAL, signals)
    assert out["object_absent"] is False
    assert out["oracle_logic"] == 0.0  # the gate already zeroes o1 and o2


def test_score_level_or_takes_max_and_min_takes_min() -> None:
    signals = {"o1": sig("yes", 0.9), "o2": sig("no", 0.6)}
    out = score_level(OR_PAIR, signals)
    assert out["oracle_logic"] == pytest.approx(0.9)
    assert out["oracle_min"] == pytest.approx(0.6)
    assert out["oracle_existence"] == 1.0  # no existence items


def test_score_level_kleene_or_needs_every_disjunct_for_no() -> None:
    no_and_unknown = {"o1": sig("no", 0.99), "o2": sig("unknown", 0.5)}
    both_no = {"o1": sig("no", 0.99), "o2": sig("no", 0.7)}
    yes_and_no = {"o1": sig("yes", 0.6), "o2": sig("no", 0.99)}
    assert score_level(OR_PAIR, no_and_unknown)["oracle_logic"] == pytest.approx(0.99)
    assert score_level(OR_PAIR, no_and_unknown)["oracle_kleene"] == 0.0
    assert score_level(OR_PAIR, both_no)["oracle_kleene"] == pytest.approx(0.7)
    assert score_level(OR_PAIR, yes_and_no)["oracle_kleene"] == pytest.approx(0.6)


def test_score_level_and_is_min_and_kleene_dual() -> None:
    both_yes = {"o1": sig("yes", 0.9), "o2": sig("yes", 0.7)}
    one_no = {"o1": sig("yes", 0.9), "o2": sig("no", 0.8)}
    assert score_level(AND_PAIR, both_yes)["oracle_logic"] == pytest.approx(0.7)
    assert score_level(AND_PAIR, one_no)["oracle_kleene"] == pytest.approx(0.8)


def test_score_level_empty_plan_scores_zero() -> None:
    out = score_level([], {})
    assert out["oracle_logic"] == 0.0 and out["oracle_min"] == 0.0
    assert out["object_absent"] is False


def test_score_level_mixed_logic_raises() -> None:
    mixed = items_of(("o1", FINAL, None, "or", []), ("o2", FINAL, None, None, []))
    with pytest.raises(ValueError):
        score_level(mixed, {"o1": sig("yes", 0.9), "o2": sig("yes", 0.9)})
