"""Mandatory test 8: NaN, out-of-range, missing / duplicate / extra ids,
cycles and extra items are detected; nothing is silently coerced."""

from __future__ import annotations

import json
import math

import pytest

from milestone1.gqa_stop.client import parse_and_validate
from milestone1.gqa_stop.schemas import (
    check_confidence,
    validate_checklist,
    validate_direct,
    validate_plan,
    validate_reader,
)

PLAN_IDS = ["c1", "c2"]


def plan(*items: dict) -> dict:
    return {"items": list(items)}


def item(i: str, deps: list[str] | None = None) -> dict:
    return {"id": i, "question": f"q {i}?", "depends_on": deps or []}


def check(i: str, conf: object = 0.5, resolved: object = True) -> dict:
    return {"id": i, "answer": "red", "resolved": resolved, "confidence": conf}


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.01, 1.01, True, "0.9", None])
def test_invalid_confidences_rejected(value: object) -> None:
    number, errors = check_confidence(value, "x")
    assert number is None and errors


@pytest.mark.parametrize("value", [0, 1, 0.0, 0.5, 1.0])
def test_valid_confidences_accepted(value: object) -> None:
    number, errors = check_confidence(value, "x")
    assert errors == [] and number == float(value)


def test_nan_literal_from_json_is_rejected() -> None:
    parsed, errors = parse_and_validate(
        '{"answer": "red", "confidence": NaN}', validate_direct
    )
    assert parsed is None and any("finite" in e for e in errors)


def test_reader_and_direct_validation() -> None:
    assert validate_reader({"answer": "red"}) == ({"answer": "red"}, [])
    assert validate_reader({"answer": ""})[1]
    assert validate_reader({"answer": 3})[1]
    assert validate_reader(["red"])[1]
    assert validate_direct({"answer": "red", "confidence": 0.7})[1] == []
    assert validate_direct({"answer": "red", "confidence": "0.7"})[1]
    assert validate_direct({"answer": "red"})[1]


def test_plan_valid_two_and_three_items() -> None:
    ok, errors = validate_plan(plan(item("c1"), item("c2", ["c1"])), 2, 3)
    assert errors == [] and [i["id"] for i in ok["items"]] == ["c1", "c2"]
    assert (
        validate_plan(plan(item("a"), item("b", ["a"]), item("c", ["a", "b"])), 2, 3)[1]
        == []
    )


def test_plan_item_count_bounds() -> None:
    assert validate_plan(plan(item("c1")), 2, 3)[1]
    four = plan(item("a"), item("b"), item("c"), item("d"))
    assert any("expected 2-3" in e for e in validate_plan(four, 2, 3)[1])


def test_plan_detects_cycles_unknown_and_duplicate_ids() -> None:
    self_dep = plan(item("c1", ["c1"]), item("c2"))
    forward = plan(item("c1", ["c2"]), item("c2", ["c1"]))
    unknown = plan(item("c1"), item("c2", ["zz"]))
    dup = plan(item("c1"), item("c1"))
    assert any("cycle" in e for e in validate_plan(self_dep, 2, 3)[1])
    assert any("cycle" in e for e in validate_plan(forward, 2, 3)[1])
    assert any("unknown" in e for e in validate_plan(unknown, 2, 3)[1])
    assert any("duplicate" in e for e in validate_plan(dup, 2, 3)[1])


def test_plan_rejects_bad_types() -> None:
    bad_deps = plan(item("c1"), {"id": "c2", "question": "x", "depends_on": "c1"})
    no_question = plan(item("c1"), {"id": "c2", "question": "", "depends_on": []})
    assert validate_plan(bad_deps, 2, 3)[1]
    assert validate_plan(no_question, 2, 3)[1]
    assert validate_plan({"items": "c1"}, 2, 3)[1]


def test_checklist_valid() -> None:
    ok, errors = validate_checklist(
        {"items": [check("c1"), check("c2", 0.0, False)]}, PLAN_IDS
    )
    assert errors == [] and ok["items"][1]["resolved"] is False


@pytest.mark.parametrize(
    ("items", "needle"),
    [
        ([check("c1")], "missing"),
        ([check("c1"), check("c1"), check("c2")], "duplicate"),
        ([check("c1"), check("c2"), check("c3")], "extra"),
        ([check("c2"), check("c1")], "order"),
        ([check("c1", math.nan), check("c2")], "finite"),
        ([check("c1", 1.5), check("c2")], "outside"),
        ([check("c1", resolved="true"), check("c2")], "boolean"),
        ([check("c1", conf=True), check("c2")], "number"),
    ],
)
def test_checklist_errors(items: list[dict], needle: str) -> None:
    parsed, errors = validate_checklist({"items": items}, PLAN_IDS)
    assert parsed is None
    assert any(needle in e for e in errors), errors


def test_whole_content_must_be_json() -> None:
    fenced = "```json\n" + json.dumps({"answer": "red"}) + "\n```"
    parsed, errors = parse_and_validate(fenced, validate_reader)
    assert parsed is None and errors
    noisy = 'Sure: {"answer": "red"}'
    assert parse_and_validate(noisy, validate_reader)[0] is None
