"""Exploratory informed-reader probe: input formatting and onset logic."""

from __future__ import annotations

import math

import pandas as pd

from milestone1.gqa_stop.informed_reader import (
    NEVER,
    informed_spec,
    observation_lines,
    onsets,
)

ROW = {
    "item1_question": "Is there a chair?",
    "item1_answer": "yes",
    "item1_resolved": True,
    "item1_rel": 0.876,
    "item2_question": "What color is the chair?",
    "item2_answer": math.nan,
    "item2_resolved": False,
    "item2_rel": math.nan,
    "item3_question": math.nan,
    "direct_answer": "red",
    "reader_answer": "blue",
}


def test_observation_lines_format_and_missing_values() -> None:
    lines = observation_lines(ROW)
    assert lines == [
        "Q: Is there a chair? | A: yes | resolved: true | answerability: 0.88",
        "Q: What color is the chair? | A: (none) | resolved: false | answerability: n/a",
    ]


def test_informed_spec_never_leaks_gt_or_direct_answer() -> None:
    q = {"question_id": "01", "question": "What color is the chair?", "gt_ref": "red"}
    spec = informed_spec(q, 3, b"png", ROW)
    assert set(spec.input_data) == {"question", "auxiliary_observations"}
    assert "red" not in str(spec.input_data) and spec.logprobs > 0
    assert spec.role == "reader_informed"


def test_onsets_stable_correct_from_level() -> None:
    table = pd.DataFrame(
        {
            "question_id": ["a"] * 4 + ["b"] * 4,
            "level": [1, 2, 3, 4] * 2,
            "ok": [True, False, True, True, True, True, True, False],
        }
    )
    assert onsets(table, "ok") == {"a": 3, "b": NEVER}
