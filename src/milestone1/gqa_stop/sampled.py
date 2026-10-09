"""v3 (plan §18.2): standalone checklist items observed one at a time, N samples.

Per question a text-only planner_v3 makes 2-3 self-contained items. Per level
and item, observer_v3 sees ONLY the image and that item's question, N times
(temperature T, seed base+i). Direct is sampled the same way. Item scores:
  S_mean  = mean over N of effective confidence (unresolved / "unknown" /
            invalid samples contribute 0);
  S_agree = share of the N samples that are resolved and whose answer equals
            the majority answer (normalised string over the valid samples).
            Tie rule: if several answers share the top count, the
            lexicographically smallest normalised answer is the majority.
Dependency gate (pre-registered): a child's score is 0 unless every parent's
gated score of the same type is > 0, i.e. the parent was resolved in at least
one sample. Diagnostic stricter gate: parent S_agree >= 0.5.
Stop score = min over items (mean logged). Direct-sampled: S_mean = mean
confidence, S_agree = share of samples equal to the majority answer.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .client import STATUS_VALID, RequestEngine, RequestSpec
from .dataset import normalize_text
from .images import ImageSeries
from .inference import READER_ROLE, CacheMissError, attempt_columns, build_level_spec
from .planner import PLANNER_FAILURE, PlanOutcome, plan_from_record
from .schemas import (
    check_answer,
    check_confidence,
    validate_direct,
    validate_plan,
    validate_reader,
)

PLANNER_V3_ROLE = "planner_v3"
OBSERVER_ROLE = "observer_v3"
DIRECT_ROLE = "direct"
UNKNOWN = "unknown"
STRICT_PARENT_AGREE = 0.5
METHOD_COLUMNS = {
    "checklist_v3_min_smean": "v3_min_smean",
    "checklist_v3_min_sagree": "v3_min_sagree",
    "direct_sampled_smean": "direct_smean",
    "direct_sampled_sagree": "direct_sagree",
}
# Heuristic only (logged, never rejected): pronouns / cross-item references.
NOT_STANDALONE_RE = re.compile(
    r"\b(it|its|they|them|their|if yes|if so|the above|previous|item|c[1-9])\b"
)


# ---- configuration -------------------------------------------------------------------
@dataclass(frozen=True)
class Sampling:
    n: int
    temperature: float
    seed_base: int

    def override(self, index: int) -> tuple[tuple[str, Any], ...]:
        return (("temperature", self.temperature), ("seed", self.seed_base + index))


def sampling_from_config(cfg: Mapping[str, Any]) -> Sampling:
    s = cfg["sampling"]
    return Sampling(int(s["n"]), float(s["temperature"]), int(s["seed_base"]))


# ---- validation ------------------------------------------------------------------------
ATTRIBUTE_WORD_RE = re.compile(r"\b(colou?r|material|made of|shape|shaped)\b")


def has_attribute_word(question: str) -> bool:
    return bool(ATTRIBUTE_WORD_RE.search(normalize_text(question)))


def item_not_standalone(question: str) -> bool:
    return bool(NOT_STANDALONE_RE.search(normalize_text(question)))


def validate_plan_v3(obj: Any, min_items: int, max_items: int) -> tuple[Any, list[str]]:
    """v1 plan schema; the standalone rule is only flagged per item."""
    parsed, errors = validate_plan(obj, min_items, max_items)
    if errors:
        return None, errors
    items = [
        {**item, "item_not_standalone": item_not_standalone(item["question"])}
        for item in parsed["items"]
    ]
    # Flag only (never rejected): the last item must name the attribute (§18.6).
    last_ok = bool(items) and has_attribute_word(items[-1]["question"])
    return {"items": items, "last_item_has_attribute": last_ok}, []


def validate_observation(obj: Any) -> tuple[Any, list[str]]:
    if not isinstance(obj, dict):
        return None, ["response is not a JSON object"]
    answer, errors = check_answer(obj.get("answer"), "answer")
    resolved = obj.get("resolved")
    if not isinstance(resolved, bool):
        errors.append("resolved must be a boolean")
    confidence, conf_errors = check_confidence(obj.get("confidence"), "confidence")
    errors.extend(conf_errors)
    if errors:
        return None, errors
    return {"answer": answer, "resolved": resolved, "confidence": confidence}, []


# ---- scores ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Sample:
    valid: bool
    answer: str | None = None
    resolved: bool = False
    confidence: float = 0.0

    @property
    def norm(self) -> str:
        return normalize_text(self.answer or "")

    @property
    def effective(self) -> float:
        if not self.valid or not self.resolved or self.norm == UNKNOWN:
            return 0.0
        return self.confidence


def sample_from_record(record: Mapping[str, Any] | None, has_resolved: bool) -> Sample:
    if record is None or record.get("status") != STATUS_VALID:
        return Sample(False)
    parsed = record["parsed"]
    resolved = bool(parsed["resolved"]) if has_resolved else True
    return Sample(True, str(parsed["answer"]), resolved, float(parsed["confidence"]))


def majority_answer(samples: Sequence[Sample]) -> str | None:
    """Most frequent normalised answer of the valid samples; ties -> smallest."""
    counts = Counter(s.norm for s in samples if s.valid)
    if not counts:
        return None
    top = max(counts.values())
    return sorted(a for a, c in counts.items() if c == top)[0]


def item_scores(samples: Sequence[Sample], n: int) -> dict[str, Any]:
    majority = majority_answer(samples)
    agree = sum(
        1
        for s in samples
        if s.valid and s.resolved and s.norm != UNKNOWN and s.norm == majority
    )
    return {
        "smean": sum(s.effective for s in samples) / n,
        "sagree": agree / n,
        "resolved_frac": sum(1 for s in samples if s.valid and s.resolved) / n,
        "valid": sum(1 for s in samples if s.valid),
        "majority": majority,
    }


def gate(
    plan_items: Sequence[Mapping[str, Any]],
    raw: Mapping[str, float],
    parent_ok: Callable[[str], bool],
) -> dict[str, float]:
    """Child score -> 0 unless every parent passes ``parent_ok`` (transitively)."""
    passed: dict[str, bool] = {}
    out: dict[str, float] = {}
    for item in plan_items:
        ok = all(passed.get(dep, False) for dep in item["depends_on"])
        value = raw[item["id"]] if ok else 0.0
        out[item["id"]] = value
        passed[item["id"]] = ok and parent_ok(item["id"])
    return out


def checklist_scores(
    plan_items: Sequence[Mapping[str, Any]], per_item: Mapping[str, Mapping[str, Any]]
) -> dict[str, float]:
    """min/mean stop scores for both types, plus the strict-gate diagnostic."""
    if not plan_items:  # planner failure: every stop score is 0 -> Level 20
        return {
            f"v3_{agg}_{kind}{suffix}": 0.0
            for agg in ("min", "mean")
            for kind in ("smean", "sagree")
            for suffix in ("", "_strict")
        }
    result: dict[str, float] = {}
    sagree = {i: per_item[i]["sagree"] for i in per_item}
    for kind in ("smean", "sagree"):
        raw = {i: per_item[i][kind] for i in per_item}
        loose = gate(plan_items, raw, lambda i, raw=raw: raw[i] > 0)
        strict = gate(plan_items, raw, lambda i: sagree[i] >= STRICT_PARENT_AGREE)
        for suffix, values in (("", loose), ("_strict", strict)):
            vals = list(values.values())
            result[f"v3_min_{kind}{suffix}"] = min(vals)
            result[f"v3_mean_{kind}{suffix}"] = sum(vals) / len(vals)
    return result


def direct_scores(samples: Sequence[Sample], n: int) -> dict[str, Any]:
    majority = majority_answer(samples)
    return {
        "direct_smean": sum(s.confidence for s in samples if s.valid) / n,
        "direct_sagree": sum(1 for s in samples if s.valid and s.norm == majority) / n,
        "direct_majority": majority,
        "direct_valid": sum(1 for s in samples if s.valid),
    }


# ---- request specs ---------------------------------------------------------------------
def planner_v3_spec(question: Mapping[str, Any]) -> RequestSpec:
    return RequestSpec(
        role=PLANNER_V3_ROLE,
        question_id=str(question["question_id"]),
        level=None,
        input_data={"question": str(question["question"])},
    )


def observer_spec(
    question: Mapping[str, Any],
    level: int,
    image: bytes,
    item: Mapping[str, Any],
    sample: int,
    sampling: Sampling,
    namespace: str = "",
) -> RequestSpec:
    """The item question ONLY (no original question, no other items)."""
    return RequestSpec(
        role=OBSERVER_ROLE,
        question_id=str(question["question_id"]),
        level=level,
        input_data={"question": str(item["question"])},
        image_bytes=image,
        namespace=namespace,
        options_override=sampling.override(sample),
        suffix=f"_{item['id']}_s{sample}",
    )


def direct_sample_spec(
    question: Mapping[str, Any],
    level: int,
    image: bytes,
    sample: int,
    sampling: Sampling,
) -> RequestSpec:
    return RequestSpec(
        role=DIRECT_ROLE,
        question_id=str(question["question_id"]),
        level=level,
        input_data={"question": str(question["question"])},
        image_bytes=image,
        options_override=sampling.override(sample),
        suffix=f"_s{sample}",
    )


# ---- runner ----------------------------------------------------------------------------
class CallTally:
    """Per-role call counts for first-attempt / post-repair validity."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def add(self, role: str, record: Mapping[str, Any] | None) -> None:
        if record is None:
            return
        self.counts[f"{role}_calls"] += 1
        self.counts[f"{role}_initial_valid"] += int(bool(record.get("initial_valid")))
        self.counts[f"{role}_valid"] += int(record.get("status") == STATUS_VALID)
        self.counts[f"{role}_repaired"] += int(int(record.get("attempt", 0)) > 1)


class SampledRunner:
    def __init__(
        self,
        engine: RequestEngine,
        series: ImageSeries,
        sampling: Sampling,
        min_items: int,
        max_items: int,
        cache_only: bool = False,
    ) -> None:
        self.engine = engine
        self.series = series
        self.sampling = sampling
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

    def plan(self, question: Mapping[str, Any]) -> PlanOutcome:
        record = self._run(
            planner_v3_spec(question),
            lambda o: validate_plan_v3(o, self.min_items, self.max_items),
        )
        return plan_from_record(str(question["question_id"]), record)

    def run_question(
        self, question: Mapping[str, Any], levels: Sequence[int]
    ) -> list[dict[str, Any]]:
        plan = self.plan(question)
        return [self.level_row(question, level, plan) for level in levels]

    def level_row(
        self, question: Mapping[str, Any], level: int, plan: PlanOutcome
    ) -> dict[str, Any]:
        image = self.series.bytes_for(str(question["image_id"]), level)
        tally = CallTally()
        reader = self._run(
            build_level_spec(READER_ROLE, question, level, image), validate_reader
        )
        n = self.sampling.n
        per_item: dict[str, dict[str, Any]] = {}
        for item in plan.items:
            records = [
                self._run(
                    observer_spec(question, level, image, item, i, self.sampling),
                    validate_observation,
                )
                for i in range(n)
            ]
            for record in records:
                tally.add("observer", record)
            per_item[item["id"]] = item_scores(
                [sample_from_record(r, True) for r in records], n
            )
        direct_records = [
            self._run(
                direct_sample_spec(question, level, image, i, self.sampling),
                validate_direct,
            )
            for i in range(n)
        ]
        for record in direct_records:
            tally.add("direct", record)
        direct = direct_scores(
            [sample_from_record(r, False) for r in direct_records], n
        )
        return sampled_row(question, level, plan, reader, per_item, direct, tally)


MAX_ITEMS_COLUMNS = 3


def sampled_row(
    question: Mapping[str, Any],
    level: int,
    plan: PlanOutcome,
    reader: Mapping[str, Any] | None,
    per_item: Mapping[str, Mapping[str, Any]],
    direct: Mapping[str, Any],
    tally: CallTally,
) -> dict[str, Any]:
    reader_ok = reader is not None and reader.get("status") == STATUS_VALID
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
        "planner_status": plan.status,
        "planner_initial_valid": plan.initial_valid,
        "planner_attempts": plan.attempts,
        "n_items": len(plan.items),
        "item_not_standalone_any": any(
            i.get("item_not_standalone") for i in plan.items
        ),
        **checklist_scores(plan.items, per_item),
        **direct,
        **dict(tally.counts),
    }
    for index, item in enumerate(plan.items[:MAX_ITEMS_COLUMNS], start=1):
        scores = per_item[item["id"]]
        row[f"item{index}_question"] = item["question"]
        row[f"item{index}_smean"] = scores["smean"]
        row[f"item{index}_sagree"] = scores["sagree"]
        row[f"item{index}_resolved_frac"] = scores["resolved_frac"]
        row[f"item{index}_majority"] = scores["majority"]
    if plan.status == PLANNER_FAILURE:
        row["item1_resolved_frac"] = None
    return row


def planned_sampled_requests(
    n_questions: int, n_levels: int, n: int, items: int
) -> int:
    """Upper bound: planner + per level (items x n observer + n direct)."""
    return n_questions + n_questions * n_levels * (items * n + n)
