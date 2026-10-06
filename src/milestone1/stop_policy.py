"""Resolution-recovery stop rule (plan §6).

Per check item j (observed issue or hypothesis) with judgment:

    u_j = min(basis_j, resolution_benefit_j)
          if anchors valid AND region anchor present/exists
             AND status == unresolved AND limitation == low_resolution
          else 0

    A_t = max_j over observed issues u_j   (0 if none; max_u if graph empty)
    H_t = max_j over hypotheses u_j        (0 if none)
    C_t = max(A_t, H_t)  [proposed]  |  A_t  [observed_only]
    C_t = None when the model output failed / is missing / is invalid.

Instant stop at t iff C_t < theta. Actual stop when the instant stop holds for
``consecutive_k`` consecutive levels; stop_t is the last of them. A continue or
a missing C_t resets the counter. If never satisfied by the final level,
stop_t = final level and forced_terminal = True.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from milestone1.robovis_data import load_robovis_config
from milestone1.structured_io import (
    Checks,
    ContractLimits,
    GraphIndex,
    Hypothesis,
    JudgmentItem,
    Judgments,
    ObservedIssue,
    validate_checks,
    validate_judgments,
)

STATUS_UNRESOLVED = "unresolved"
LIMITATION_LOW_RES = "low_resolution"
LIMITATION_UNKNOWN = "unknown"
UNKNOWN_CONSERVATIVE_U = 1  # u assigned to unresolved/unknown items in that variant
UNKNOWN_CONSERVATIVE_MIN_BASIS = 1

PROPOSED = "proposed"
OBSERVED_ONLY = "observed_only"
UNKNOWN_CONSERVATIVE = "unknown_conservative"
FIXED_PREFIX = "fixed_"


@dataclass(frozen=True)
class Condition:
    """One adaptive stop condition."""

    name: str
    include_hypotheses: bool
    unknown_conservative: bool
    theta: int
    consecutive_k: int
    max_u: int


@dataclass(frozen=True)
class LevelScore:
    a: int
    h: int
    c: int


@dataclass(frozen=True)
class StopResult:
    stop_t: int
    natural_stop: bool
    forced_terminal: bool
    failure_count: int


def build_conditions(cfg: Mapping[str, Any] | None = None) -> dict[str, Condition]:
    """The three adaptive conditions, parameterised from the robovis config."""
    cfg = cfg if cfg is not None else load_robovis_config()
    proposal = cfg["proposal"]
    theta = int(proposal["theta"])
    k = int(proposal["consecutive_k"])
    max_u = int(proposal["max_u"])
    return {
        PROPOSED: Condition(PROPOSED, True, False, theta, k, max_u),
        OBSERVED_ONLY: Condition(OBSERVED_ONLY, False, False, theta, k, max_u),
        UNKNOWN_CONSERVATIVE: Condition(
            UNKNOWN_CONSERVATIVE,
            True,
            True,
            int(proposal["unknown_conservative_theta"]),
            k,
            max_u,
        ),
    }


def fixed_method_name(level: int) -> str:
    return f"{FIXED_PREFIX}{level:02d}"


def _anchors_valid(item: ObservedIssue | Hypothesis, graph: GraphIndex) -> bool:
    return bool(item.anchor_ids) and all(a in graph.entity_ids for a in item.anchor_ids)


def _region_valid(item: ObservedIssue | Hypothesis, graph: GraphIndex) -> bool:
    return (
        item.region_anchor_id is not None and item.region_anchor_id in graph.entity_ids
    )


def item_utility(
    item: ObservedIssue | Hypothesis,
    judgment: JudgmentItem,
    graph: GraphIndex,
    *,
    unknown_conservative: bool = False,
) -> int:
    """u_j of one check item (plan §6)."""
    if not (_anchors_valid(item, graph) and _region_valid(item, graph)):
        return 0
    if judgment.status != STATUS_UNRESOLVED:
        return 0
    if judgment.limitation == LIMITATION_LOW_RES:
        return min(judgment.basis, judgment.resolution_benefit)
    if (
        unknown_conservative
        and judgment.limitation == LIMITATION_UNKNOWN
        and judgment.basis >= UNKNOWN_CONSERVATIVE_MIN_BASIS
    ):
        return UNKNOWN_CONSERVATIVE_U
    return 0


def level_score(
    checks: Checks, judgments: Judgments, graph: GraphIndex, condition: Condition
) -> LevelScore:
    """A_t, H_t, C_t for one level with valid checks + judgments."""
    by_id = judgments.by_id()

    def best(items: Sequence[ObservedIssue | Hypothesis]) -> int:
        return max(
            (
                item_utility(
                    it,
                    by_id[it.id],
                    graph,
                    unknown_conservative=condition.unknown_conservative,
                )
                for it in items
            ),
            default=0,
        )

    a = condition.max_u if graph.is_empty else best(checks.observed_issues)
    h = best(checks.hypotheses)
    c = max(a, h) if condition.include_hypotheses else a
    return LevelScore(a, h, c)


def empty_graph_score(condition: Condition) -> LevelScore:
    """Score of a level whose observed graph is empty (no model output needed)."""
    a = condition.max_u
    return LevelScore(a, 0, a)


def replay_stop(
    c_seq: Sequence[int | None], theta: int, consecutive_k: int
) -> StopResult:
    """Replay the k-consecutive stop rule over C_1..C_N (None = missing).

    ``failure_count`` counts missing C_t among the levels actually observed
    (1..stop_t), since later levels would not be requested online.
    """
    if not c_seq:
        raise ValueError("c_seq must contain at least one level")
    if consecutive_k < 1:
        raise ValueError("consecutive_k must be >= 1")
    run = 0
    failures = 0
    for t, c in enumerate(c_seq, start=1):
        if c is None:
            failures += 1
            run = 0
            continue
        run = run + 1 if c < theta else 0
        if run >= consecutive_k:
            return StopResult(
                t, natural_stop=True, forced_terminal=False, failure_count=failures
            )
    return StopResult(
        len(c_seq), natural_stop=False, forced_terminal=True, failure_count=failures
    )


def fixed_stop(level: int, num_levels: int) -> StopResult:
    """Fixed-k baseline: always stop at ``level``."""
    if not 1 <= level <= num_levels:
        raise ValueError(f"fixed level {level} outside 1..{num_levels}")
    return StopResult(level, natural_stop=True, forced_terminal=False, failure_count=0)


# Keys of the on-disk payloads that belong to the contracts; anything else in
# the files (e.g. "failure", model metadata) is wrapper information.
CHECKS_PAYLOAD_KEYS = ("observed_issues", "hypotheses")
JUDGMENTS_PAYLOAD_KEYS = ("items",)
FAILURE_KEY = "failure"
# check_generator.py records nest the contract payload under this key.
NESTED_CHECKS_KEY = "checks"


def _payload(obj: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {k: obj[k] for k in keys if k in obj}


def _checks_payload(obj: Mapping[str, Any]) -> Any:
    """Contract payload of a checks file: flat or nested under ``"checks"``."""
    if NESTED_CHECKS_KEY in obj:
        return obj[NESTED_CHECKS_KEY]
    return _payload(obj, CHECKS_PAYLOAD_KEYS)


def score_level_outputs(
    graph: Mapping[str, Any] | None,
    checks_obj: Mapping[str, Any] | None,
    decision_obj: Mapping[str, Any] | None,
    conditions: Mapping[str, Condition],
    limits: ContractLimits | None = None,
) -> tuple[dict[str, int | None], str]:
    """C_t per condition for one level from raw on-disk objects.

    ``checks_obj`` may be the bare contract dict or a check_generator record
    (payload under ``"checks"``, plus ``"failure"`` and metadata);
    ``decision_obj`` is a judgment record (``"items"`` + ``"failure"``).
    Returns (C_t by condition name, outcome tag). Outcome tags:
    ``ok``, ``empty_graph``, ``missing_graph``, ``missing_checks``,
    ``missing_decision``, ``checks_failure``, ``decision_failure``,
    ``invalid_checks``, ``invalid_judgments``. Every outcome other than
    ``ok``/``empty_graph`` yields C_t = None for all conditions.
    """
    missing: dict[str, int | None] = dict.fromkeys(conditions)
    if graph is None:
        return missing, "missing_graph"
    index = GraphIndex.from_graph(graph)
    if index.is_empty:
        # A_t = max_u regardless of model output (plan §6).
        return {n: empty_graph_score(c).c for n, c in conditions.items()}, "empty_graph"
    if checks_obj is None:
        return missing, "missing_checks"
    if checks_obj.get(FAILURE_KEY) is True:
        return missing, "checks_failure"
    if decision_obj is None:
        return missing, "missing_decision"
    if decision_obj.get(FAILURE_KEY) is True:
        return missing, "decision_failure"
    checks = validate_checks(_checks_payload(checks_obj), index, limits)
    if not checks.ok or checks.value is None:
        return missing, "invalid_checks"
    judgments = validate_judgments(
        _payload(decision_obj, JUDGMENTS_PAYLOAD_KEYS), checks.value, limits
    )
    if not judgments.ok or judgments.value is None:
        return missing, "invalid_judgments"
    return {
        name: level_score(checks.value, judgments.value, index, cond).c
        for name, cond in conditions.items()
    }, "ok"
