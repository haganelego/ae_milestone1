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
BENEFIT_GATED = "benefit_gated"
BENEFIT_GATED_OBS = "benefit_gated_obs"
FIXED_PREFIX = "fixed_"
MAIN_CONDITIONS = (PROPOSED, OBSERVED_ONLY, UNKNOWN_CONSERVATIVE)
EXPLORATORY_CONDITIONS = (BENEFIT_GATED, BENEFIT_GATED_OBS)
EXPLORATORY_LABEL = "exploratory (plan deviation, needs sign-off)"
MAIN_LABEL = "main"

# u-rules: "min" = plan §6 min(basis, benefit); "benefit_gated" = benefit if
# basis >= BENEFIT_GATED_MIN_BASIS (exploratory).
U_RULE_MIN = "min"
U_RULE_BENEFIT_GATED = "benefit_gated"
BENEFIT_GATED_MIN_BASIS = 1


@dataclass(frozen=True)
class Condition:
    """One adaptive stop condition."""

    name: str
    include_hypotheses: bool
    unknown_conservative: bool
    theta: int
    consecutive_k: int
    max_u: int
    u_rule: str = U_RULE_MIN
    exploratory: bool = False

    @property
    def status_label(self) -> str:
        return EXPLORATORY_LABEL if self.exploratory else MAIN_LABEL


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


def build_conditions(
    cfg: Mapping[str, Any] | None = None, *, include_exploratory: bool = True
) -> dict[str, Condition]:
    """Adaptive conditions, parameterised from the robovis config.

    Main: proposed, observed_only, unknown_conservative (plan §6).
    Exploratory (plan deviation): benefit_gated, benefit_gated_obs.
    """
    cfg = cfg if cfg is not None else load_robovis_config()
    proposal = cfg["proposal"]
    theta = int(proposal["theta"])
    k = int(proposal["consecutive_k"])
    max_u = int(proposal["max_u"])
    main = {
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
    if not include_exploratory:
        return main
    exploratory = {
        name: Condition(
            name,
            include_hypotheses=name == BENEFIT_GATED,
            unknown_conservative=False,
            theta=theta,
            consecutive_k=k,
            max_u=max_u,
            u_rule=U_RULE_BENEFIT_GATED,
            exploratory=True,
        )
        for name in EXPLORATORY_CONDITIONS
    }
    return {**main, **exploratory}


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
    u_rule: str = U_RULE_MIN,
) -> int:
    """u_j of one check item (plan §6; ``u_rule`` selects the exploratory rule)."""
    if not (_anchors_valid(item, graph) and _region_valid(item, graph)):
        return 0
    if judgment.status != STATUS_UNRESOLVED:
        return 0
    if judgment.limitation == LIMITATION_LOW_RES:
        if u_rule == U_RULE_BENEFIT_GATED:
            if judgment.basis < BENEFIT_GATED_MIN_BASIS:
                return 0
            return judgment.resolution_benefit
        if u_rule != U_RULE_MIN:
            raise ValueError(f"unknown u_rule {u_rule!r}")
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
                    u_rule=condition.u_rule,
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


OUTCOME_OK = "ok"
OUTCOME_EMPTY_GRAPH = "empty_graph"


@dataclass(frozen=True)
class ParsedLevel:
    """Validated model outputs of one state (checks/judgments None if empty graph)."""

    graph: GraphIndex
    checks: Checks | None
    judgments: Judgments | None


def parse_level_outputs(
    graph: Mapping[str, Any] | None,
    checks_obj: Mapping[str, Any] | None,
    decision_obj: Mapping[str, Any] | None,
    limits: ContractLimits | None = None,
) -> tuple[ParsedLevel | None, str]:
    """Validate the raw on-disk objects of one state.

    ``checks_obj`` may be the bare contract dict or a check_generator record
    (payload under ``"checks"``, plus ``"failure"`` and metadata);
    ``decision_obj`` is a judgment record (``"items"`` + ``"failure"``).
    Returns (parsed or None, outcome tag). Outcome tags:
    ``ok``, ``empty_graph``, ``missing_graph``, ``missing_checks``,
    ``missing_decision``, ``checks_failure``, ``decision_failure``,
    ``invalid_checks``, ``invalid_judgments``. Only ``ok``/``empty_graph``
    return a ParsedLevel.
    """
    if graph is None:
        return None, "missing_graph"
    index = GraphIndex.from_graph(graph)
    if index.is_empty:
        # A_t = max_u regardless of model output (plan §6).
        return ParsedLevel(index, None, None), OUTCOME_EMPTY_GRAPH
    if checks_obj is None:
        return None, "missing_checks"
    if checks_obj.get(FAILURE_KEY) is True:
        return None, "checks_failure"
    if decision_obj is None:
        return None, "missing_decision"
    if decision_obj.get(FAILURE_KEY) is True:
        return None, "decision_failure"
    checks = validate_checks(_checks_payload(checks_obj), index, limits)
    if not checks.ok or checks.value is None:
        return None, "invalid_checks"
    judgments = validate_judgments(
        _payload(decision_obj, JUDGMENTS_PAYLOAD_KEYS), checks.value, limits
    )
    if not judgments.ok or judgments.value is None:
        return None, "invalid_judgments"
    return ParsedLevel(index, checks.value, judgments.value), OUTCOME_OK


def parsed_level_score(parsed: ParsedLevel, condition: Condition) -> LevelScore:
    """A_t, H_t, C_t of a validated state (empty graph handled)."""
    if parsed.checks is None or parsed.judgments is None:
        return empty_graph_score(condition)
    return level_score(parsed.checks, parsed.judgments, parsed.graph, condition)


def score_level_outputs(
    graph: Mapping[str, Any] | None,
    checks_obj: Mapping[str, Any] | None,
    decision_obj: Mapping[str, Any] | None,
    conditions: Mapping[str, Condition],
    limits: ContractLimits | None = None,
) -> tuple[dict[str, int | None], str]:
    """C_t per condition for one state; None for every condition unless the
    outcome is ``ok`` or ``empty_graph`` (see :func:`parse_level_outputs`)."""
    parsed, outcome = parse_level_outputs(graph, checks_obj, decision_obj, limits)
    if parsed is None:
        return dict.fromkeys(conditions), outcome
    return {
        name: parsed_level_score(parsed, cond).c for name, cond in conditions.items()
    }, outcome
