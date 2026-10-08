"""Effective confidence (test 6), cache-only replay of min/mean (test 7),
planner / reader failure handling (test 9)."""

from __future__ import annotations

from pathlib import Path

import pytest
from gqa_stop_helpers import (
    FAIL_PLAN_WORD,
    FakeTransport,
    answers_by_text,
    fake_tags_get,
    make_context,
    make_world,
)

from milestone1.gqa_stop import stages_model
from milestone1.gqa_stop.client import record_tag
from milestone1.gqa_stop.inference import (
    CacheMissError,
    aggregate,
    effective_confidences,
)
from milestone1.gqa_stop.policy import replay_stop
from milestone1.gqa_stop.stages_model import level_table

PLAN3 = [
    {"id": "c1", "question": "a", "depends_on": []},
    {"id": "c2", "question": "b", "depends_on": ["c1"]},
    {"id": "c3", "question": "c", "depends_on": ["c2"]},
]


def obs(i: str, resolved: bool, conf: float, answer: str = "x") -> dict:
    return {"id": i, "answer": answer, "resolved": resolved, "confidence": conf}


# ---- test 6 -----------------------------------------------------------------------
def test_unresolved_ancestor_zeroes_descendants() -> None:
    r = effective_confidences(
        PLAN3, [obs("c1", False, 0.0), obs("c2", True, 0.99), obs("c3", True, 0.98)]
    )
    assert r == {"c1": 0.0, "c2": 0.0, "c3": 0.0}
    r = effective_confidences(
        PLAN3, [obs("c1", True, 0.9), obs("c2", False, 0.7), obs("c3", True, 0.98)]
    )
    assert r == {"c1": 0.9, "c2": 0.0, "c3": 0.0}


def test_unknown_answer_counts_as_unresolved() -> None:
    r = effective_confidences(
        PLAN3[:2], [obs("c1", True, 0.9, "Unknown"), obs("c2", True, 0.9)]
    )
    assert r == {"c1": 0.0, "c2": 0.0}


def test_failed_checklist_call_gives_zero() -> None:
    assert effective_confidences(PLAN3, None) == {"c1": 0.0, "c2": 0.0, "c3": 0.0}


def test_min_and_mean_use_same_effective_values() -> None:
    r = effective_confidences(
        PLAN3, [obs("c1", True, 0.9), obs("c2", True, 0.6), obs("c3", False, 0.8)]
    )
    c_min, c_mean = aggregate(r)
    assert c_min == 0.0
    assert c_mean == pytest.approx((0.9 + 0.6 + 0.0) / 3)
    assert aggregate({}) == (0.0, 0.0)


# ---- tests 7 and 9 on the synthetic world ----------------------------------------------
@pytest.fixture()
def world(tmp_path: Path):
    config_path, cfg, questions = make_world(tmp_path)
    transport = FakeTransport(None, answers_by_text(questions))
    ctx = make_context(config_path, cfg, transport)
    transport.paths = ctx.paths

    stages_model.preflight(ctx, get=fake_tags_get)
    return ctx, transport, questions


def question_dict(qid: str, raw: dict) -> dict:
    return {
        "question_id": qid,
        "image_id": raw["imageId"],
        "question": raw["question"],
        "split": "dev",
    }


def test_replay_from_cache_matches_and_makes_no_calls(world) -> None:
    ctx, transport, questions = world
    qid, raw = next(
        (k, v) for k, v in questions.items() if FAIL_PLAN_WORD not in v["question"]
    )
    q = question_dict(qid, raw)
    levels = list(range(1, 21))
    live = level_table(ctx, [q], levels, cache_only=False)
    calls = sum(transport.calls.values())
    replay = level_table(ctx, [q], levels, cache_only=True)
    assert sum(transport.calls.values()) == calls  # no new calls
    cols = ["level", "reader_answer", "direct_score", "checklist_min", "checklist_mean"]
    assert live[cols].equals(replay[cols])
    for row in replay.itertuples(index=False):
        values = list(row.r_values.values())
        assert row.checklist_min == min(values)
        assert row.checklist_mean == pytest.approx(sum(values) / len(values))


def test_cache_only_replay_refuses_missing_records(world) -> None:
    ctx, _, questions = world
    qid, raw = next(iter(questions.items()))
    with pytest.raises(CacheMissError):
        level_table(ctx, [question_dict(qid, raw)], [1], cache_only=True)


def test_planner_failure_is_kept_and_forces_level_20(world) -> None:
    ctx, _, questions = world
    qid, raw = next(
        (k, v) for k, v in questions.items() if FAIL_PLAN_WORD in v["question"]
    )
    table = level_table(
        ctx, [question_dict(qid, raw)], list(range(1, 21)), cache_only=False
    )
    assert len(table) == 20  # not excluded
    assert set(table["planner_status"]) == {"planner_failure"}
    assert set(table["check_status"]) == {"planner_failure"}
    assert (table["checklist_min"] == 0).all() and (table["checklist_mean"] == 0).all()
    decision = replay_stop(table["checklist_min"].tolist(), 0.01)
    assert decision.level == 20 and decision.never_met
    tag = record_tag(ctx.settings.name, ctx.settings.digest)
    plan_record = ctx.paths.out(f"plans/{tag}/{qid}.json")
    assert plan_record.exists()


def test_reader_failure_is_incorrect(world) -> None:
    ctx, transport, questions = world
    qid, raw = next(
        (k, v) for k, v in questions.items() if FAIL_PLAN_WORD not in v["question"]
    )
    transport.broken.add(("answer_only", 20))
    table = level_table(ctx, [question_dict(qid, raw)], [20], cache_only=False)
    row = table.iloc[0]
    assert row["reader_status"] == "schema_failure"
    assert row["reader_answer"] is None and not row["reader_correct"]
