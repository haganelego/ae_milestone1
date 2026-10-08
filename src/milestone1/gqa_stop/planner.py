"""Question-only checklist planner (plan §5.3). One plan per question, fixed
across all levels. A final planner failure is NOT a question exclusion: the
checklist scores become 0 at every level, which forces a stop at Level 20.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .client import STATUS_VALID, RequestSpec

PLANNER_ROLE = "planner"
PLANNER_FAILURE = "planner_failure"


def planner_input(question_text: str) -> dict[str, Any]:
    """The planner sees the question string only (no image, no GT, no program)."""
    return {"question": question_text}


def planner_spec(question: Mapping[str, Any]) -> RequestSpec:
    return RequestSpec(
        role=PLANNER_ROLE,
        question_id=str(question["question_id"]),
        level=None,
        input_data=planner_input(str(question["question"])),
        image_bytes=None,
    )


@dataclass(frozen=True)
class PlanOutcome:
    question_id: str
    status: str
    items: list[dict[str, Any]] = field(default_factory=list)
    initial_valid: bool = False  # valid on the first attempt (no repair)
    attempts: int = 0

    @property
    def failed(self) -> bool:
        return self.status != STATUS_VALID

    @property
    def ids(self) -> list[str]:
        return [item["id"] for item in self.items]


def plan_from_record(question_id: str, record: Mapping[str, Any]) -> PlanOutcome:
    initial = bool(record.get("initial_valid"))
    attempts = int(record.get("attempt", 0))
    if record.get("status") != STATUS_VALID or record.get("parsed") is None:
        return PlanOutcome(question_id, PLANNER_FAILURE, [], initial, attempts)
    items = list(record["parsed"]["items"])
    return PlanOutcome(question_id, STATUS_VALID, items, initial, attempts)
