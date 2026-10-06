"""Tests for the ROBOVIS2027 stop rule (plan §6, acceptance tests 6 and 7).

Run: uv run --project egtr --with pytest python -m pytest tests/test_stop_policy.py -q
"""

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.robovis_data import load_robovis_config  # noqa: E402
from milestone1.stop_policy import (  # noqa: E402
    BENEFIT_GATED,
    BENEFIT_GATED_OBS,
    MAIN_CONDITIONS,
    OBSERVED_ONLY,
    PROPOSED,
    U_RULE_BENEFIT_GATED,
    UNKNOWN_CONSERVATIVE,
    build_conditions,
    fixed_method_name,
    fixed_stop,
    item_utility,
    level_score,
    replay_stop,
    score_level_outputs,
)
from milestone1.structured_io import (  # noqa: E402
    Checks,
    GraphIndex,
    Hypothesis,
    JudgmentItem,
    Judgments,
    ObservedIssue,
)

N = 20
THETA = 2
K = 2
CFG = load_robovis_config()
CONDITIONS = build_conditions(CFG)
GRAPH = GraphIndex(frozenset({"v1", "v2", "v3"}), frozenset({"e1"}))
EMPTY_GRAPH = GraphIndex(frozenset(), frozenset())


def _obs(
    cid: str = "o1", anchors: tuple[str, ...] = ("v1",), region: str | None = "v1"
):
    return ObservedIssue(cid, anchors, "target", region, "evidence")


def _hyp(cid: str = "h1", region: str | None = "v2") -> Hypothesis:
    return Hypothesis(cid, "missing_object", "cup", ("v2",), (), region, "evidence")


def _judge(
    cid: str,
    status: str = "unresolved",
    basis: int = 2,
    benefit: int = 2,
    limitation: str = "low_resolution",
) -> JudgmentItem:
    return JudgmentItem(cid, status, basis, benefit, limitation, "reason")


# --------------------------------------------------------------------------
# (6) replay
# --------------------------------------------------------------------------


def test_replay_two_consecutive_instant_stops_stops_at_t3() -> None:
    res = replay_stop([2, 1, 1] + [2] * (N - 3), THETA, K)
    assert (res.stop_t, res.natural_stop, res.forced_terminal) == (3, True, False)


def test_replay_never_below_theta_forces_terminal_at_t20() -> None:
    res = replay_stop([2] * N, THETA, K)
    assert (res.stop_t, res.natural_stop, res.forced_terminal) == (N, False, True)


def test_replay_only_last_two_levels_is_natural_stop_at_t20() -> None:
    res = replay_stop([2] * (N - 2) + [1, 1], THETA, K)
    assert (res.stop_t, res.natural_stop, res.forced_terminal) == (N, True, False)


def test_replay_missing_level_resets_counter_and_counts_failure() -> None:
    res = replay_stop([2, 1, None, 1, 1] + [2] * (N - 5), THETA, K)
    assert res.stop_t == 5
    assert res.failure_count == 1


def test_replay_failures_after_stop_are_not_counted() -> None:
    res = replay_stop([1, 1] + [None] * (N - 2), THETA, K)
    assert (res.stop_t, res.failure_count) == (2, 0)


def test_replay_all_missing_is_forced_with_all_failures() -> None:
    res = replay_stop([None] * N, THETA, K)
    assert (res.stop_t, res.forced_terminal, res.failure_count) == (N, True, N)


def test_replay_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError):
        replay_stop([], THETA, K)


def test_fixed_stop_and_name() -> None:
    res = fixed_stop(7, N)
    assert (res.stop_t, res.forced_terminal, res.failure_count) == (7, False, 0)
    assert fixed_method_name(7) == "fixed_07"
    with pytest.raises(ValueError):
        fixed_stop(N + 1, N)


# --------------------------------------------------------------------------
# (7) item utility
# --------------------------------------------------------------------------


@pytest.mark.parametrize("limitation", ["out_of_frame", "occluded", "other", "unknown"])
def test_non_low_resolution_limitations_give_zero(limitation: str) -> None:
    assert item_utility(_obs(), _judge("o1", limitation=limitation), GRAPH) == 0


def test_missing_region_anchor_gives_zero() -> None:
    assert item_utility(_obs(region=None), _judge("o1"), GRAPH) == 0


def test_region_anchor_not_in_graph_gives_zero() -> None:
    assert item_utility(_obs(region="v99"), _judge("o1"), GRAPH) == 0


def test_empty_anchor_list_gives_zero() -> None:
    assert item_utility(_obs(anchors=()), _judge("o1"), GRAPH) == 0


@pytest.mark.parametrize("status", ["resolved", "unsupported"])
def test_non_unresolved_status_gives_zero(status: str) -> None:
    assert item_utility(_obs(), _judge("o1", status=status), GRAPH) == 0


@pytest.mark.parametrize(
    ("basis", "benefit", "u"), [(2, 2, 2), (2, 1, 1), (1, 2, 1), (0, 2, 0)]
)
def test_low_resolution_unresolved_is_min_of_basis_and_benefit(
    basis: int, benefit: int, u: int
) -> None:
    assert item_utility(_obs(), _judge("o1", basis=basis, benefit=benefit), GRAPH) == u


def test_unknown_conservative_assigns_one_only_with_basis() -> None:
    unknown = _judge("o1", limitation="unknown", basis=1, benefit=0)
    no_basis = _judge("o1", limitation="unknown", basis=0, benefit=2)
    assert item_utility(_obs(), unknown, GRAPH, unknown_conservative=True) == 1
    assert item_utility(_obs(), no_basis, GRAPH, unknown_conservative=True) == 0
    assert (
        item_utility(_obs(region=None), unknown, GRAPH, unknown_conservative=True) == 0
    )


# --------------------------------------------------------------------------
# Level score / conditions
# --------------------------------------------------------------------------


def test_conditions_from_config() -> None:
    assert CONDITIONS[PROPOSED].theta == CFG["proposal"]["theta"]
    assert CONDITIONS[UNKNOWN_CONSERVATIVE].theta == 1
    assert not CONDITIONS[OBSERVED_ONLY].include_hypotheses


def test_level_score_proposed_vs_observed_only() -> None:
    checks = Checks((_obs(),), (_hyp(),))
    judgments = Judgments((_judge("o1", basis=1), _judge("h1", basis=2)))
    prop = level_score(checks, judgments, GRAPH, CONDITIONS[PROPOSED])
    obs = level_score(checks, judgments, GRAPH, CONDITIONS[OBSERVED_ONLY])
    assert (prop.a, prop.h, prop.c) == (1, 2, 2)
    assert (obs.a, obs.c) == (1, 1)


def test_level_score_no_items_is_zero() -> None:
    score = level_score(Checks((), ()), Judgments(()), GRAPH, CONDITIONS[PROPOSED])
    assert score.c == 0


def test_level_score_empty_graph_forces_a_max() -> None:
    for cond in CONDITIONS.values():
        score = level_score(Checks((), ()), Judgments(()), EMPTY_GRAPH, cond)
        assert score.a == CFG["proposal"]["max_u"] == score.c


# --------------------------------------------------------------------------
# score_level_outputs (raw on-disk objects)
# --------------------------------------------------------------------------


def _graph(entities: bool = True) -> dict[str, Any]:
    ents = [{"id": e} for e in ("v1", "v2", "v3")] if entities else []
    return {"entities": ents, "relations": [{"id": "e1"}] if entities else []}


CHECKS_OBJ = {
    "observed_issues": [
        {"id": "o1", "anchor_ids": ["v1"], "target": "t", "region_anchor_id": "v1",
         "evidence": "e"}
    ],
    "hypotheses": [],
}  # fmt: skip
DECISION_OBJ = {
    "items": [
        {"id": "o1", "status": "unresolved", "basis": 2, "resolution_benefit": 1,
         "limitation": "low_resolution", "reason": "r"}
    ],
    "failure": False,
}  # fmt: skip


def test_score_level_outputs_ok() -> None:
    scores, outcome = score_level_outputs(
        _graph(), CHECKS_OBJ, DECISION_OBJ, CONDITIONS
    )
    assert outcome == "ok"
    assert scores[PROPOSED] == 1


def test_score_level_outputs_empty_graph_needs_no_model_output() -> None:
    scores, outcome = score_level_outputs(_graph(False), None, None, CONDITIONS)
    assert outcome == "empty_graph"
    assert set(scores.values()) == {CFG["proposal"]["max_u"]}


@pytest.mark.parametrize(
    ("graph", "checks", "decision", "outcome"),
    [
        (None, CHECKS_OBJ, DECISION_OBJ, "missing_graph"),
        (_graph(), None, DECISION_OBJ, "missing_checks"),
        (_graph(), {**CHECKS_OBJ, "failure": True}, DECISION_OBJ, "checks_failure"),
        (_graph(), CHECKS_OBJ, None, "missing_decision"),
        (_graph(), CHECKS_OBJ, {**DECISION_OBJ, "failure": True}, "decision_failure"),
        (_graph(), {"observed_issues": [{"id": "o1"}], "hypotheses": []}, DECISION_OBJ,
         "invalid_checks"),
        (_graph(), CHECKS_OBJ, {"items": [], "failure": False}, "invalid_judgments"),
    ],
)  # fmt: skip
def test_score_level_outputs_failures_are_missing(
    graph: Any, checks: Any, decision: Any, outcome: str
) -> None:
    scores, got = score_level_outputs(graph, checks, decision, CONDITIONS)
    assert got == outcome
    assert all(c is None for c in scores.values())


def test_score_level_outputs_accepts_nested_check_generator_record() -> None:
    record = {"model": "m", "failure": False, "error": None, "checks": CHECKS_OBJ}
    scores, outcome = score_level_outputs(_graph(), record, DECISION_OBJ, CONDITIONS)
    assert outcome == "ok"
    assert scores[PROPOSED] == 1


def test_score_level_outputs_nested_null_checks_is_invalid() -> None:
    record = {"failure": False, "checks": None}
    scores, outcome = score_level_outputs(_graph(), record, DECISION_OBJ, CONDITIONS)
    assert outcome == "invalid_checks"
    assert scores[PROPOSED] is None


def test_score_level_outputs_null_items_failure_record_is_missing() -> None:
    record = {"items": None, "failure": True, "skipped_empty": False, "error": "x"}
    scores, outcome = score_level_outputs(_graph(), CHECKS_OBJ, record, CONDITIONS)
    assert outcome == "decision_failure"
    assert scores[PROPOSED] is None


# --------------------------------------------------------------------------
# Exploratory benefit_gated rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("basis", "benefit", "u"), [(1, 2, 2), (2, 1, 1), (2, 0, 0), (0, 2, 0)]
)
def test_benefit_gated_utility(basis: int, benefit: int, u: int) -> None:
    judgment = _judge("o1", basis=basis, benefit=benefit)
    assert item_utility(_obs(), judgment, GRAPH, u_rule=U_RULE_BENEFIT_GATED) == u


@pytest.mark.parametrize(
    "judgment",
    [
        _judge("o1", limitation="unknown"),
        _judge("o1", status="resolved"),
    ],
)
def test_benefit_gated_keeps_the_plan_gate(judgment: JudgmentItem) -> None:
    assert item_utility(_obs(), judgment, GRAPH, u_rule=U_RULE_BENEFIT_GATED) == 0
    assert (
        item_utility(_obs(region=None), _judge("o1"), GRAPH, u_rule="benefit_gated")
        == 0
    )


def test_unknown_u_rule_raises() -> None:
    with pytest.raises(ValueError):
        item_utility(_obs(), _judge("o1"), GRAPH, u_rule="nope")


def test_exploratory_conditions_are_labelled() -> None:
    bg, bg_obs = CONDITIONS[BENEFIT_GATED], CONDITIONS[BENEFIT_GATED_OBS]
    assert bg.include_hypotheses and not bg_obs.include_hypotheses
    assert bg.exploratory and bg_obs.exploratory
    assert bg.theta == CFG["proposal"]["theta"] and bg.consecutive_k == K
    assert "exploratory" in bg.status_label
    assert all(not CONDITIONS[n].exploratory for n in MAIN_CONDITIONS)
    assert set(build_conditions(CFG, include_exploratory=False)) == set(MAIN_CONDITIONS)


def test_benefit_gated_level_score_vs_min_rule() -> None:
    # basis 1 / benefit 2 on both items: min-rule u=1, benefit_gated u=2.
    checks = Checks((_obs(),), (_hyp(),))
    judgments = Judgments((_judge("o1", basis=1), _judge("h1", basis=1)))
    assert level_score(checks, judgments, GRAPH, CONDITIONS[PROPOSED]).c == 1
    bg = level_score(checks, judgments, GRAPH, CONDITIONS[BENEFIT_GATED])
    obs = level_score(checks, judgments, GRAPH, CONDITIONS[BENEFIT_GATED_OBS])
    assert (bg.a, bg.h, bg.c) == (2, 2, 2)
    assert obs.c == 2
