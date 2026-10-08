"""Per-level inference (reader / direct / checklist) and stop-score derivation.

Request builders take a question mapping but copy ONLY ``question_id`` and the
``question`` text into the request (plus the fixed planner items for the
checklist); GT, fullAnswer, semantic and annotations never reach a request.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from .client import STATUS_VALID, RequestEngine, RequestSpec
from .dataset import normalize_text
from .images import ImageSeries
from .planner import PLANNER_FAILURE, PlanOutcome, plan_from_record, planner_spec
from .schemas import (
    validate_checklist,
    validate_direct,
    validate_plan,
    validate_reader,
)

READER_ROLE = "answer_only"
DIRECT_ROLE = "direct"
CHECKLIST_ROLE = "checklist"
LEVEL_ROLES = (READER_ROLE, DIRECT_ROLE, CHECKLIST_ROLE)
UNKNOWN_ANSWER = "unknown"
FAILED_SCORE = 0.0
NOT_RUN = "not_run"
SCORE_COLUMNS = {
    "direct": "direct_score",
    "checklist_min": "checklist_min",
    "checklist_mean": "checklist_mean",
}


class CacheMissError(RuntimeError):
    """Raised in cache-only replay when a record is missing."""


def question_text_input(question: Mapping[str, Any]) -> dict[str, Any]:
    return {"question": str(question["question"])}


def checklist_input(
    question: Mapping[str, Any], plan_items: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    items = [
        {"id": i["id"], "question": i["question"], "depends_on": list(i["depends_on"])}
        for i in plan_items
    ]
    return {"question": str(question["question"]), "checklist": items}


def build_level_spec(
    role: str,
    question: Mapping[str, Any],
    level: int,
    image_bytes: bytes,
    plan_items: Sequence[Mapping[str, Any]] | None = None,
    namespace: str = "",
) -> RequestSpec:
    if role == CHECKLIST_ROLE:
        data = checklist_input(question, plan_items or [])
    else:
        data = question_text_input(question)
    return RequestSpec(
        role=role,
        question_id=str(question["question_id"]),
        level=level,
        input_data=data,
        image_bytes=image_bytes,
        namespace=namespace,
    )


def validator_for(role: str, plan: PlanOutcome | None) -> Callable[[Any], Any]:
    if role == READER_ROLE:
        return validate_reader
    if role == DIRECT_ROLE:
        return validate_direct
    ids = plan.ids if plan is not None else []
    return lambda obj: validate_checklist(obj, ids)


def effective_confidences(
    plan_items: Sequence[Mapping[str, Any]],
    check_items: Sequence[Mapping[str, Any]] | None,
) -> dict[str, float]:
    """r_j = confidence_j if j and all its ancestors are resolved, else 0 (§5.4).

    "unknown" answers count as unresolved; a failed checklist call -> all 0.
    """
    if not check_items:
        return {item["id"]: FAILED_SCORE for item in plan_items}
    by_id = {item["id"]: item for item in check_items}
    resolved: dict[str, bool] = {}
    result: dict[str, float] = {}
    for item in plan_items:
        observed = by_id.get(item["id"])
        own = (
            observed is not None
            and observed["resolved"] is True
            and normalize_text(str(observed["answer"])) != UNKNOWN_ANSWER
        )
        ok = own and all(resolved.get(dep, False) for dep in item["depends_on"])
        resolved[item["id"]] = ok
        result[item["id"]] = float(observed["confidence"]) if ok and observed else 0.0
    return result


def aggregate(r_values: Mapping[str, float]) -> tuple[float, float]:
    """(min, mean) of the SAME effective confidences; empty -> (0, 0)."""
    if not r_values:
        return FAILED_SCORE, FAILED_SCORE
    values = list(r_values.values())
    return min(values), sum(values) / len(values)


def level_row(
    question: Mapping[str, Any],
    level: int,
    plan: PlanOutcome,
    reader: Mapping[str, Any] | None,
    direct: Mapping[str, Any] | None,
    checks: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """One (question, level) row of the score table. GT is added later."""
    reader_ok = reader is not None and reader.get("status") == STATUS_VALID
    direct_ok = direct is not None and direct.get("status") == STATUS_VALID
    checks_ok = checks is not None and checks.get("status") == STATUS_VALID
    check_items = checks["parsed"]["items"] if checks_ok and checks else None
    r_values = effective_confidences(plan.items, check_items) if not plan.failed else {}
    c_min, c_mean = aggregate(r_values)
    direct_parsed = direct["parsed"] if direct_ok and direct else {}
    unresolved = sum(1 for v in r_values.values() if v <= 0.0)
    return {
        "question_id": str(question["question_id"]),
        "image_id": str(question["image_id"]),
        "group_id": str(question.get("group_id", question["image_id"])),
        "split": question.get("split", ""),
        "attribute": question.get("attribute", ""),
        "relate_count": question.get("relate_count", -1),
        "cell": question.get("cell", ""),
        "question": str(question.get("question", "")),
        "level": level,
        "reader_answer": reader["parsed"]["answer"] if reader_ok and reader else None,
        "reader_status": _status(reader),
        "direct_answer": direct_parsed.get("answer"),
        "direct_confidence": direct_parsed.get("confidence"),
        "direct_status": _status(direct),
        "direct_score": float(direct_parsed.get("confidence", FAILED_SCORE)),
        "planner_status": plan.status,
        "n_items": len(plan.items),
        "check_status": _check_status(plan, checks),
        "r_values": dict(r_values),
        "unresolved_items": unresolved,
        "checklist_min": c_min,
        "checklist_mean": c_mean,
    }


def _check_status(plan: PlanOutcome, checks: Mapping[str, Any] | None) -> str:
    if plan.status == NOT_RUN:
        return NOT_RUN
    return PLANNER_FAILURE if plan.failed else _status(checks)


def _status(record: Mapping[str, Any] | None) -> str:
    return NOT_RUN if record is None else str(record.get("status"))


class InferenceRunner:
    """Runs (or replays from cache) every role for questions x levels."""

    def __init__(
        self,
        engine: RequestEngine,
        series: ImageSeries,
        min_items: int,
        max_items: int,
        cache_only: bool = False,
    ) -> None:
        self.engine = engine
        self.series = series
        self.min_items = min_items
        self.max_items = max_items
        self.cache_only = cache_only

    def _run(self, spec: RequestSpec, validator: Callable[[Any], Any]) -> dict:
        if not self.cache_only:
            return self.engine.run(spec, validator)
        cached = self.engine.lookup(spec)
        if cached is None:
            raise CacheMissError(f"{spec.role} {spec.question_id} L{spec.level}")
        return cached

    def plan(self, question: Mapping[str, Any]) -> PlanOutcome:
        record = self._run(
            planner_spec(question),
            lambda obj: validate_plan(obj, self.min_items, self.max_items),
        )
        return plan_from_record(str(question["question_id"]), record)

    def role_record(
        self,
        role: str,
        question: Mapping[str, Any],
        level: int,
        plan: PlanOutcome | None = None,
        namespace: str = "",
    ) -> dict[str, Any] | None:
        if role == CHECKLIST_ROLE and (plan is None or plan.failed):
            return None  # planner failure: no checklist call, score 0
        image = self.series.bytes_for(str(question["image_id"]), level)
        items = plan.items if plan is not None else None
        spec = build_level_spec(role, question, level, image, items, namespace)
        return self._run(spec, validator_for(role, plan))

    def run_question(
        self,
        question: Mapping[str, Any],
        levels: Iterable[int],
        roles: Sequence[str] = LEVEL_ROLES,
    ) -> list[dict[str, Any]]:
        plan = self.plan(question) if CHECKLIST_ROLE in roles else _no_plan(question)
        rows = []
        for level in levels:
            records = {r: self.role_record(r, question, level, plan) for r in roles}
            rows.append(
                level_row(
                    question,
                    level,
                    plan,
                    records.get(READER_ROLE),
                    records.get(DIRECT_ROLE),
                    records.get(CHECKLIST_ROLE),
                )
            )
        return rows


def _no_plan(question: Mapping[str, Any]) -> PlanOutcome:
    return PlanOutcome(str(question["question_id"]), NOT_RUN, [])


def planned_request_count(n_questions: int, n_levels: int, roles: Sequence[str]) -> int:
    per_level = sum(1 for r in roles if r in LEVEL_ROLES)
    planner = n_questions if CHECKLIST_ROLE in roles else 0
    return planner + n_questions * n_levels * per_level
