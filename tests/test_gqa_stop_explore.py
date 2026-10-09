"""Exploratory replays (orchestrator request; not pre-registered): soft
existence score, k = 2 stopping, best-θ selection, written outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from gqa_stop_helpers import CONFIG_PATH

from milestone1.gqa_stop.explore import (
    SETTINGS,
    best_setting,
    replay,
    run_exploration,
    soft_existence_score,
)

POLICY = json.loads(CONFIG_PATH.read_text())["policy"]
N = 20


def row(n_items: int = 2, **kw) -> dict:
    base = {
        "n_items": n_items,
        "item1_resolved": "False",
        "item1_valid": "True",
        "item1_product": 0.9,
        "item2_resolved": "True",
        "item2_valid": "True",
        "item2_product": 0.42,
        "item3_resolved": "True",
        "item3_valid": "True",
        "item3_product": 0.7,
    }
    return {**base, **kw}


def test_soft_existence_uses_last_item_only() -> None:
    assert soft_existence_score(row()) == pytest.approx(
        0.42
    )  # item 1 unresolved ignored
    assert soft_existence_score(row(n_items=3)) == pytest.approx(0.7)
    assert soft_existence_score(row(item2_resolved="False")) == 0.0
    assert soft_existence_score(row(item2_valid="False")) == 0.0
    assert soft_existence_score(row(n_items=0)) == 0.0


def table(scores: dict[str, list[float]], correct_from: dict[str, int]) -> pd.DataFrame:
    rows = []
    for q, values in scores.items():
        for t, v in enumerate(values, start=1):
            rows.append(
                {
                    "question_id": q,
                    "group_id": q,
                    "level": t,
                    "score": v,
                    "reader_correct": t >= correct_from[q],
                }
            )
    return pd.DataFrame(rows)


def test_k2_needs_two_consecutive_levels() -> None:
    spike = [0.0] * 3 + [0.9] + [0.0] * 4 + [0.9, 0.9] + [0.0] * 10
    frame = table({"a": spike}, {"a": 1})
    assert replay(frame, "score", 0.5, 1)["mean_level"] == 4
    assert replay(frame, "score", 0.5, 2)["mean_level"] == 10


def test_stopped_wrong_ids_and_best_setting() -> None:
    early = [0.95] * 4 + [1.0] * (N - 4)  # passes 0.96 first at L5 (correct)
    late = [0.0] * 15 + [1.0] * 5
    frame = table({"a": early, "b": late}, {"a": 5, "b": 1})
    result = replay(frame, "score", 0.9, 1)
    assert result["stopped_wrong_ids"] == ["a"] and result["accuracy"] == 0.5
    best = best_setting(frame, "score", 1, [0.9, 0.96, float("inf")], 0.95)
    assert best["theta"] == 0.96 and best["accuracy"] == 1.0


def test_run_exploration_writes_all_settings(tmp_path: Path) -> None:
    rows = []
    for q in range(4):
        for t in range(1, N + 1):
            rows.append(
                {
                    "question_id": f"q{q}",
                    "group_id": f"g{q}",
                    "level": t,
                    "reader_correct": t >= 2 + q,
                    "n_items": 2,
                    "item1_resolved": "True",
                    "item1_valid": "True",
                    "item1_product": 0.9,
                    "item2_resolved": str(t >= 3),
                    "item2_valid": "True",
                    "item2_product": min(1.0, t / 10),
                    "checklist_v4_min": min(1.0, t / 12),
                    "direct_v4": min(1.0, t / 8),
                    "checklist_v1_min": 0.9 if t > 15 else 0.0,
                }
            )
    summary = run_exploration(pd.DataFrame(rows), POLICY, tmp_path)
    names = [s["method"] for s in summary["settings"]]
    assert names == [name for name, *_ in SETTINGS]
    assert (tmp_path / "dev_exploratory.csv").exists()
    assert all("stopped_wrong_ids" in s for s in summary["settings"] if s["feasible"])
    assert "not pre-registered" in summary["label"]
