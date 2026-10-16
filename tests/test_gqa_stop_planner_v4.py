"""planner_v4: extended-schema validator, request specs and plan-quality helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from milestone1.gqa_stop.client import ROLE_DIRS
from milestone1.gqa_stop.oracle_plan import build_oracle_plan
from milestone1.gqa_stop.oracle_score import score_level
from milestone1.gqa_stop.planner_v4 import (
    OBSERVER_V4_NAMESPACE,
    PLANNER_V4_ROLE,
    PlanV4Settings,
    final_type,
    observer_v4_spec,
    plan_quality,
    plan_v4_items,
    plan_v4_settings,
    planner_v4_spec,
    validate_plan_v4,
)
from milestone1.gqa_stop.sampled import validate_plan_v3

REPO = Path(__file__).resolve().parents[1]
SELECTION = REPO / "outputs/gqa_stop/probe_multi_rel/selection.jsonl"
PROMPT = REPO / "prompts/gqa_stop/planner_v4.txt"
DEFAULTS = PlanV4Settings()


def item(
    item_id: str,
    question: str,
    deps: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {"id": item_id, "question": question, "depends_on": deps or [], **extra}


def op(operation: str, argument: str = "", deps: list[int] | None = None) -> dict:
    return {"operation": operation, "argument": argument, "dependencies": deps or []}


def compare_oracle() -> tuple[dict[str, Any], list[dict]]:
    program = [
        op("select", "van (1)"),
        op("select", "hydrant (2)"),
        op("same color", "", [0, 1]),
    ]
    plan = build_oracle_plan(
        {
            "question_id": "q1",
            "question": "Do the van and the hydrant have the same color?",
            "semantic": program,
        }
    )
    return plan, program


# ---- validator ---------------------------------------------------------------------------
def test_validate_plan_v4_full_schema_returns_items_with_roles() -> None:
    # Arrange
    obj = {
        "items": [
            item(
                "c1",
                "Is there a table in the image?",
                role="existence",
                required_answer="yes",
                logic=None,
            ),
            item(
                "c2",
                "What color is the cup on the table?",
                ["c1"],
                role="final",
                required_answer=None,
                logic=None,
            ),
        ]
    }
    # Act
    parsed, errors = validate_plan_v4(obj, DEFAULTS)
    # Assert
    assert errors == []
    assert [i["role"] for i in parsed["items"]] == ["existence", "final"]
    assert [i["required_answer"] for i in parsed["items"]] == ["yes", None]
    assert parsed["defaults_applied"] == []


def test_validate_plan_v4_missing_fields_get_defaults_and_are_listed() -> None:
    obj = {
        "items": [item("c1", "Is there a cup?"), item("c2", "Is the cup red?", ["c1"])]
    }
    parsed, errors = validate_plan_v4(obj, DEFAULTS)
    assert errors == []
    assert [i["role"] for i in parsed["items"]] == ["existence", "final"]
    assert [i["required_answer"] for i in parsed["items"]] == ["yes", None]
    assert [i["logic"] for i in parsed["items"]] == [None, None]
    assert set(parsed["defaults_applied"]) == {
        f"{i}.{f}" for i in ("c1", "c2") for f in ("role", "logic", "required_answer")
    }


def test_validate_plan_v4_without_defaults_missing_role_is_error() -> None:
    obj = {"items": [item("c1", "Is there a cup?"), item("c2", "Is the cup red?")]}
    parsed, errors = validate_plan_v4(obj, PlanV4Settings(fill_defaults=False))
    assert parsed is None
    assert any("role is required" in e for e in errors)


def test_validate_plan_v4_accepts_or_operands_case_insensitive() -> None:
    obj = {
        "items": [
            item(
                "c1", "Is there a cup?", role="Final", required_answer=None, logic="OR"
            ),
            item(
                "c2",
                "Is there a plate?",
                role="final",
                required_answer=None,
                logic="or",
            ),
        ]
    }
    parsed, errors = validate_plan_v4(obj, DEFAULTS)
    assert errors == []
    assert [i["logic"] for i in parsed["items"]] == ["or", "or"]


@pytest.mark.parametrize(
    ("items", "message"),
    [
        (
            [
                item("c1", "Is there a cup?", role="existence", logic="or"),
                item("c2", "Is the cup red?", role="final"),
            ],
            "only allowed on final",
        ),
        (
            [
                item("c1", "Is there a cup?", role="final", logic="or"),
                item("c2", "Is there a plate?", role="final", logic="and"),
            ],
            "mix different logic",
        ),
        (
            [
                item("c1", "Is there a cup?", role="existence"),
                item("c2", "Is there a plate?", role="existence"),
            ],
            "no final item",
        ),
        (
            [
                item("c1", "Is there a cup?", role="existence"),
                item("c2", "Is the cup red?", role="final", required_answer="yes"),
            ],
            "required_answer null",
        ),
        (
            [
                item("c1", "Is there a cup?", role="object"),
                item("c2", "Is the cup red?", role="final"),
            ],
            "role must be",
        ),
        (
            [
                item("c1", "Is there a cup?", required_answer="no"),
                item("c2", "Is the cup red?"),
            ],
            "required_answer must be",
        ),
        (
            [item("c 1", "Is there a cup?"), item("c2", "Is the cup red?", ["c 1"])],
            "id must match",
        ),
    ],
)
def test_validate_plan_v4_contradictions_are_errors(
    items: list[dict[str, Any]], message: str
) -> None:
    parsed, errors = validate_plan_v4({"items": items}, DEFAULTS)
    assert parsed is None
    assert any(message in e for e in errors), errors


def test_validate_plan_v4_item_count_bounds() -> None:
    five = [item(f"c{k}", f"Is there a thing {k}?") for k in range(1, 6)]
    six = [*five, item("c6", "Is there a thing 6?")]
    assert validate_plan_v4({"items": five}, DEFAULTS)[1] == []
    assert validate_plan_v4({"items": six}, DEFAULTS)[0] is None
    assert validate_plan_v4({"items": five[:1]}, DEFAULTS)[0] is None


def test_plan_v4_settings_reads_config_block_with_defaults() -> None:
    assert plan_v4_settings({}) == PlanV4Settings(2, 5, True)
    cfg = {"planner_v4": {"max_items": 4, "fill_defaults": False}}
    assert plan_v4_settings(cfg) == PlanV4Settings(2, 4, False)


def test_planner_v3_validator_unchanged_by_v4() -> None:
    # v3 keeps its 2-3 item limit and drops unknown keys
    items = [
        item("c1", "Is there a cup in the image?", role="existence"),
        item("c2", "What color is the cup?", ["c1"], role="final"),
    ]
    parsed, errors = validate_plan_v3({"items": items}, 2, 3)
    assert errors == []
    assert "role" not in parsed["items"][0]
    four = [item(f"c{k}", "Is there a cup?") for k in range(1, 5)]
    assert validate_plan_v3({"items": four}, 2, 3)[0] is None


# ---- specs and records -------------------------------------------------------------------------
def test_planner_v4_spec_is_text_only_question() -> None:
    spec = planner_v4_spec(
        {"question_id": "q1", "question": "Is it red?", "answer": "no"}
    )
    assert spec.role == PLANNER_V4_ROLE
    assert spec.input_data == {"question": "Is it red?"}
    assert spec.image_bytes is None and spec.level is None
    assert ROLE_DIRS[PLANNER_V4_ROLE] == "plans_v4"


def test_observer_v4_spec_uses_own_namespace_and_item_only() -> None:
    spec = observer_v4_spec(
        {"question_id": "q1", "question": "orig?"},
        3,
        b"img",
        {"id": "c2", "question": "Is there a cup?"},
    )
    assert spec.role == "observer_v3"
    assert spec.namespace == OBSERVER_V4_NAMESPACE
    assert spec.input_data == {"question": "Is there a cup?"}
    assert spec.suffix == "_c2" and spec.logprobs == 10


def test_plan_v4_items_failure_gives_empty_plan() -> None:
    assert plan_v4_items({"status": "schema_failure", "parsed": None}) == []
    ok = {"status": "valid", "parsed": {"items": [{"id": "c1"}]}}
    assert plan_v4_items(ok) == [{"id": "c1"}]


def test_v4_items_score_with_oracle_aggregator() -> None:
    parsed, _ = validate_plan_v4(
        {
            "items": [
                item("c1", "Is there a cup?", role="final", logic="or"),
                item("c2", "Is there a plate?", role="final", logic="or"),
            ]
        },
        DEFAULTS,
    )
    signals = {
        "c1": {"valid": True, "answer": "no", "resolved": True, "pans": 0.6},
        "c2": {"valid": True, "answer": "yes", "resolved": True, "pans": 0.9},
    }
    scores = score_level(parsed["items"], signals)
    assert scores["oracle_min"] == 0.6
    assert scores["oracle_logic"] == 0.9


# ---- plan quality -------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Do the van and the hydrant have the same color?", "compare"),
        ("What is common to the umbrella and the scarf?", "common"),
        ("Is the lamp to the left or to the right of the sofa?", "choose"),
        ("What is the man wearing?", "query"),
        ("Is the knife on a cutting board?", "verify"),
    ],
)
def test_final_type_classifies_single_final_item(text: str, expected: str) -> None:
    assert final_type([{"role": "final", "logic": None, "question": text}]) == expected


def test_final_type_logic_group_and_no_final() -> None:
    finals = [
        {"role": "final", "logic": "or", "question": "Is there a cup?"},
        {"role": "final", "logic": "or", "question": "Is there a plate?"},
    ]
    assert final_type(finals) == "logic_or"
    assert final_type([{"role": "existence", "logic": None, "question": "x"}]) == "none"


def test_plan_quality_flags_hijacked_compare_and_counts_objects() -> None:
    plan, program = compare_oracle()
    v4 = [
        {
            "id": "c1",
            "question": "Is there a van in the image?",
            "role": "existence",
            "logic": None,
        },
        {
            "id": "c2",
            "question": "What color is the van?",
            "role": "final",
            "logic": None,
        },
    ]
    quality = plan_quality(v4, plan, program)
    assert quality["objects"] == 2
    assert quality["objects_covered"] == 1  # hydrant missing
    assert quality["final_type_preserved"] is False
    assert quality["v4_final_type"] == "query"


def test_plan_quality_relation_check_needs_anchor_and_relation_words() -> None:
    program = [
        op("select", "man (1)"),
        op("relate", "_,wearing,o (2)", [0]),
        op("query", "name", [1]),
    ]
    plan = build_oracle_plan(
        {
            "question_id": "q",
            "question": "What is the man wearing?",
            "semantic": program,
        }
    )
    good = [
        {
            "id": "c1",
            "question": "Is there a man in the image?",
            "role": "existence",
            "logic": None,
        },
        {
            "id": "c2",
            "question": "Is the man wearing something?",
            "role": "existence",
            "logic": None,
        },
        {
            "id": "c3",
            "question": "What is the man wearing?",
            "role": "final",
            "logic": None,
        },
    ]
    q_good = plan_quality(good, plan, program)
    q_bad = plan_quality([good[0], good[2]], plan, program)
    assert (q_good["relation_checks"], q_good["relation_checks_covered"]) == (1, 1)
    assert q_bad["relation_checks_covered"] == 0
    assert q_good["final_type_preserved"] is True


def test_planner_v4_prompt_examples_are_not_probe_questions() -> None:
    text = PROMPT.read_text()
    assert "Return JSON only" in text and "{question}" not in text
    if not SELECTION.exists():
        pytest.skip("probe selection not available")
    questions = [
        json.loads(line)["question"] for line in SELECTION.read_text().splitlines()
    ]
    assert len(questions) == 30
    assert not [q for q in questions if q in text]


def test_validate_plan_v4_null_string_means_null() -> None:
    obj = {
        "items": [
            item(
                "c1",
                "Is there a van?",
                role="existence",
                required_answer="yes",
                logic="null",
            ),
            item(
                "c2",
                "Do the van and the car have the same color?",
                ["c1"],
                role="final",
                required_answer="None",
                logic=None,
            ),
        ]
    }
    parsed, errors = validate_plan_v4(obj, DEFAULTS)
    assert errors == []
    assert [i["logic"] for i in parsed["items"]] == [None, None]
    assert parsed["items"][1]["required_answer"] is None


def test_final_type_contraction_is_query_and_extra_existence_counted() -> None:
    assert (
        final_type(
            [{"role": "final", "logic": None, "question": "What's on the oven?"}]
        )
        == "query"
    )
    program = [
        op("select", "ottoman (1)"),
        op("verify rel", "bed,to the left of,o (-)", [0]),
    ]
    plan = build_oracle_plan(
        {
            "question_id": "q",
            "question": "Is the ottoman to the left of a bed?",
            "semantic": program,
        }
    )
    v4 = [
        {
            "id": "c1",
            "question": "Is there an ottoman in the image?",
            "role": "existence",
            "logic": None,
        },
        {
            "id": "c2",
            "question": "Is there a bed in the image?",
            "role": "existence",
            "logic": None,
        },
        {
            "id": "c3",
            "question": "Is the ottoman to the left of a bed?",
            "role": "final",
            "logic": None,
        },
    ]
    quality = plan_quality(v4, plan, program)
    assert quality["extra_existence"] == 1
    assert quality["objects_covered"] == quality["objects"] == 1
