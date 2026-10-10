"""Plan §24 v5 helpers: GQA target boxes, box rules, v3 verbalised rebuild, diff spec."""

from __future__ import annotations

import math

import numpy as np

from milestone1.gqa_stop import boxes_v5 as b
from milestone1.gqa_stop import diff_v5 as d
from milestone1.gqa_stop import replays_v5 as r
from milestone1.gqa_stop.client import ROLE_DIRS


def test_target_ids_follow_filters_and_relate() -> None:
    plain = [
        {"operation": "select", "dependencies": [], "argument": "tiles (282360)"},
        {"operation": "filter color", "dependencies": [0], "argument": "white"},
        {"operation": "query", "dependencies": [1], "argument": "shape"},
    ]
    relate = [
        {"operation": "select", "dependencies": [], "argument": "screen (1)"},
        {
            "operation": "relate",
            "dependencies": [0],
            "argument": "cabinet,to the right of,s (2)",
        },
        {"operation": "query", "dependencies": [1], "argument": "material"},
    ]
    assert b.target_ids(plain) == ["282360"]
    assert b.target_ids(relate) == ["2"]
    assert (
        b.target_ids([{"operation": "query", "dependencies": [], "argument": "x"}])
        == []
    )


def test_area_fraction_absent_and_capped() -> None:
    graph = {
        "width": 10,
        "height": 10,
        "objects": {"1": {"w": 5, "h": 2}, "2": {"w": 10, "h": 10}},
    }
    assert b.area_fraction(graph, ["1"]) == 0.1
    assert b.area_fraction(graph, ["1", "2"]) == 1.0
    assert b.area_fraction(graph, ["9"]) is None
    assert b.area_fraction(None, ["1"]) is None


def test_rule_stops_box_only_and_gated() -> None:
    eff = np.array([[1.0, 5.0, 9.0], [np.nan, np.nan, np.nan]])
    attr = np.array(["color", "color"])
    tau = {"color": 4.0, "material": 0.0, "shape": 0.0}
    level, met = b.rule_stops(eff, attr, tau, None, 0.0)
    assert level.tolist() == [2, 3] and met.tolist() == [
        True,
        False,
    ]  # absent never meets
    score = np.array([[0.9, 0.1, 0.8], [0.9, 0.9, 0.9]])
    level, met = b.rule_stops(eff, attr, tau, score, 0.5)
    assert level.tolist() == [3, 3] and met.tolist() == [True, False]


def test_search_respects_target() -> None:
    eff = np.tile(np.array([1.0, 10.0, 100.0]), (6, 1))
    correct = np.array([[False, True, True]] * 6)
    attr = np.array(["color", "material", "shape"] * 2)
    best, pareto = b.search(eff, correct, attr, b.tau_grid(eff), None, [0.0], [0.9])
    assert best["0.90"]["accuracy"] == 1.0 and best["0.90"]["mean_level"] == 2.0
    assert not pareto.empty


def test_verbalised_scores_apply_v1_gate() -> None:
    plan = [
        {"id": "1", "question": "Is there a cup?", "depends_on": []},
        {"id": "2", "question": "What color is the cup?", "depends_on": ["1"]},
    ]
    row = {
        "question_id": "q",
        "item1_question": "Is there a cup?",
        "item1_valid": True,
        "item1_answer": "yes",
        "item1_resolved": True,
        "item1_confidence": 0.9,
        "item2_question": "What color is the cup?",
        "item2_valid": True,
        "item2_answer": "red",
        "item2_resolved": True,
        "item2_confidence": 0.8,
    }
    assert r.verbalised_scores(row, plan) == (0.8, (0.9 + 0.8) / 2)
    unresolved = {**row, "item1_resolved": False}
    assert r.verbalised_scores(unresolved, plan) == (0.0, 0.0)
    missing = {**row, "item2_confidence": math.nan}
    assert r.verbalised_scores(missing, plan)[0] == 0.0


def test_diff_spec_sends_two_images_in_order() -> None:
    spec = d.diff_spec(
        {"question_id": "q", "question": "What color?"}, 5, b"prev", b"cur"
    )
    assert spec.image_bytes == b"prev" and spec.extra_image_bytes == (b"cur",)
    assert spec.raw_text and spec.logprobs > 0 and d.ROLE in ROLE_DIRS


def test_per_attribute_theta_is_box_search_on_scores() -> None:
    score = np.array([[0.1, 0.5, 0.9]] * 4)
    correct = np.array(
        [
            [False, True, True],
            [False, True, True],
            [False, False, True],
            [False, False, True],
        ]
    )
    attr = np.array(["color", "color", "material", "material"])
    grid = np.array([0.0, 0.5, 0.9, np.inf])
    best, _ = b.search(score, correct, attr, grid, None, [0.0], [0.99])
    tau = best["0.99"]["tau"]
    assert tau["color"] == 0.5 and tau["material"] == 0.9
    assert best["0.99"]["mean_level"] == (2 + 2 + 3 + 3) / 4
