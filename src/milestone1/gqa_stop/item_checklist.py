"""v3 final (plan §18.6): standalone items observed one call each, N = 1.

planner_v3 (text only) makes 2-3 self-contained items. Per level, each item is
observed by ONE observer_v3 call (temperature 0, v1 seed) that sees only the
image and that item's question. The stop score is v1's: effective confidence
r_j = confidence if j and all its ancestors are resolved (answer not
"unknown"), else 0; checklist_v3_min = min r_j (checklist_v3_mean logged).
Reader and Direct are v1's cached calls (copied by identical cache key).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .client import STATUS_VALID, RequestEngine, RequestSpec
from .dataset import normalize_text
from .images import ImageSeries
from .inference import (
    DIRECT_ROLE,
    READER_ROLE,
    CacheMissError,
    aggregate,
    attempt_columns,
    build_level_spec,
    effective_confidences,
)
from .planner import PlanOutcome, plan_from_record
from .sampled import (
    MAX_ITEMS_COLUMNS,
    OBSERVER_ROLE,
    CallTally,
    planner_v3_spec,
    validate_observation,
    validate_plan_v3,
)
from .schemas import validate_direct, validate_reader

METHOD_COLUMNS = {
    "checklist_v3_min": "checklist_v3_min",
    "checklist_v3_mean": "checklist_v3_mean",
    "direct": "direct_score",
}
PRIMARY_METHOD = "checklist_v3_min"
COUNT_ITEM_RE = re.compile(r"\b(how many|count|number of)\b")


def item_spec(
    question: Mapping[str, Any],
    level: int,
    image: bytes,
    item: Mapping[str, Any],
    namespace: str = "",
) -> RequestSpec:
    """The item question ONLY; default (v1) decoding, one call."""
    return RequestSpec(
        role=OBSERVER_ROLE,
        question_id=str(question["question_id"]),
        level=level,
        input_data={"question": str(item["question"])},
        image_bytes=image,
        namespace=namespace,
        suffix=f"_{item['id']}",
    )


def plan_quality(plan: PlanOutcome, last_has_attribute: bool) -> dict[str, Any]:
    questions = [normalize_text(i["question"]) for i in plan.items]
    return {
        "last_item_has_attribute": bool(last_has_attribute),
        "item_not_standalone_any": any(
            i.get("item_not_standalone") for i in plan.items
        ),
        "count_item_any": any(COUNT_ITEM_RE.search(q) for q in questions),
    }


class ItemChecklistRunner:
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
            raise CacheMissError(
                f"{spec.role} {spec.question_id} L{spec.level}{spec.suffix}"
            )
        return cached

    def plan(self, question: Mapping[str, Any]) -> tuple[PlanOutcome, bool]:
        record = self._run(
            planner_v3_spec(question),
            lambda o: validate_plan_v3(o, self.min_items, self.max_items),
        )
        parsed = record.get("parsed") or {}
        return plan_from_record(str(question["question_id"]), record), bool(
            parsed.get("last_item_has_attribute", False)
        )

    def run_question(
        self, question: Mapping[str, Any], levels: Sequence[int]
    ) -> list[dict[str, Any]]:
        plan, last_ok = self.plan(question)
        return [self.level_row(question, level, plan, last_ok) for level in levels]

    def level_row(
        self, question: Mapping[str, Any], level: int, plan: PlanOutcome, last_ok: bool
    ) -> dict[str, Any]:
        image = self.series.bytes_for(str(question["image_id"]), level)
        reader = self._run(
            build_level_spec(READER_ROLE, question, level, image), validate_reader
        )
        direct = self._run(
            build_level_spec(DIRECT_ROLE, question, level, image), validate_direct
        )
        tally = CallTally()
        tally.add("direct", direct)
        observations: dict[str, dict[str, Any] | None] = {}
        for item in plan.items:
            record = self._run(
                item_spec(question, level, image, item), validate_observation
            )
            tally.add("observer", record)
            ok = record.get("status") == STATUS_VALID
            observations[item["id"]] = (
                {**record["parsed"], "id": item["id"]} if ok else None
            )
        return item_row(
            question, level, plan, last_ok, reader, direct, observations, tally
        )


def item_row(
    question: Mapping[str, Any],
    level: int,
    plan: PlanOutcome,
    last_ok: bool,
    reader: Mapping[str, Any] | None,
    direct: Mapping[str, Any] | None,
    observations: Mapping[str, Mapping[str, Any] | None],
    tally: CallTally,
) -> dict[str, Any]:
    valid_obs = [o for o in observations.values() if o is not None]
    r_values = effective_confidences(plan.items, valid_obs) if not plan.failed else {}
    c_min, c_mean = aggregate(r_values)
    reader_ok = reader is not None and reader.get("status") == STATUS_VALID
    direct_ok = direct is not None and direct.get("status") == STATUS_VALID
    direct_parsed = direct["parsed"] if direct_ok and direct else {}
    row: dict[str, Any] = {
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
        "reader_status": "not_run" if reader is None else str(reader.get("status")),
        **attempt_columns("reader", reader),
        "direct_answer": direct_parsed.get("answer"),
        "direct_confidence": direct_parsed.get("confidence"),
        "direct_status": "not_run" if direct is None else str(direct.get("status")),
        "direct_score": float(direct_parsed.get("confidence", 0.0)),
        "planner_status": plan.status,
        "planner_initial_valid": plan.initial_valid,
        "planner_attempts": plan.attempts,
        "n_items": len(plan.items),
        **plan_quality(plan, last_ok),
        "r_values": dict(r_values),
        "unresolved_items": sum(1 for v in r_values.values() if v <= 0.0),
        "checklist_v3_min": c_min,
        "checklist_v3_mean": c_mean,
        **dict(tally.counts),
    }
    for index, item in enumerate(plan.items[:MAX_ITEMS_COLUMNS], start=1):
        obs = observations.get(item["id"])
        row[f"item{index}_question"] = item["question"]
        row[f"item{index}_answer"] = obs["answer"] if obs else None
        row[f"item{index}_resolved"] = bool(obs["resolved"]) if obs else False
        row[f"item{index}_confidence"] = obs["confidence"] if obs else None
        row[f"item{index}_valid"] = obs is not None
        row[f"item{index}_r"] = r_values.get(item["id"], 0.0)
    return row


def planned_item_requests(n_questions: int, n_levels: int, max_items: int) -> int:
    """Upper bound of NEW calls: planner + items per level (reader/direct copied)."""
    return n_questions + n_questions * n_levels * max_items
