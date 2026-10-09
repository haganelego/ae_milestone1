"""v4 (plan §19): token-probability x relevance item confidence.

P_ans: probability (exp(logprob), full vocabulary) of the FIRST generated token
of the "answer" value in observer_v3's JSON reply; no merging of case or
tokenisation variants. The token is found on the concatenated token BYTES:
the first token whose byte span contains the first byte after the pattern
`"answer"\\s*:\\s*"` (if that token also carries the opening quote, its
probability is used as is).
Rel: probability of the exact token "Yes" among the first generated token's
top_logprobs of a separate relevance_v4 call (image + question, plain text,
temperature 0); 0 if "Yes" is absent.
Item confidence r_j = P_ans x Rel under v1's unresolved / ancestor gate; stop
score = min (mean logged). P_ans-only and Rel-only variants are logged.
Direct v4 = P_ans(direct answer) x Rel(original question).
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .client import STATUS_VALID, RequestEngine, RequestSpec
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
from .item_checklist import item_spec, plan_quality
from .planner import PlanOutcome, plan_from_record
from .sampled import (
    MAX_ITEMS_COLUMNS,
    CallTally,
    planner_v3_spec,
    validate_observation,
    validate_plan_v3,
)
from .schemas import validate_direct, validate_reader

RELEVANCE_ROLE = "relevance_v4"
TOP_LOGPROBS = 10
LP_NAMESPACE = "lp"
YES = "Yes"
NO = "No"
ANSWER_VALUE_RE = re.compile(rb'"answer"\s*:\s*"')
KINDS = ("v4", "pans", "rel")
METHOD_COLUMNS = {
    "checklist_v4_min": "checklist_v4_min",
    "checklist_v4_mean": "checklist_v4_mean",
    "pans_min": "pans_min",
    "pans_mean": "pans_mean",
    "rel_min": "rel_min",
    "rel_mean": "rel_mean",
    "direct_v4": "direct_v4",
    "direct_pans": "direct_pans",
    "direct_rel": "direct_rel",
    "direct_v1_verbalized": "direct_score",
    "checklist_v3_min": "checklist_v3_min",
}
PRIMARY_METHOD = "checklist_v4_min"


# ---- parsing ---------------------------------------------------------------------------
def token_bytes(entry: Mapping[str, Any]) -> bytes:
    raw = entry.get("bytes")
    if isinstance(raw, list):
        return bytes(raw)
    return str(entry.get("token", "")).encode("utf-8")


@dataclass(frozen=True)
class AnswerToken:
    index: int
    token: str
    prob: float
    top: list[tuple[str, float]]


def locate_answer_token(
    logprobs: Sequence[Mapping[str, Any]] | None,
) -> AnswerToken | None:
    """First generated token of the JSON "answer" value (see module docstring)."""
    if not logprobs:
        return None
    spans: list[tuple[int, int]] = []
    buffer = b""
    for entry in logprobs:
        chunk = token_bytes(entry)
        spans.append((len(buffer), len(buffer) + len(chunk)))
        buffer += chunk
    match = ANSWER_VALUE_RE.search(buffer)
    if match is None:
        return None
    offset = match.end()
    if offset >= len(buffer) or buffer[offset : offset + 1] == b'"':
        return None  # empty answer value
    for index, (start, end) in enumerate(spans):
        if start <= offset < end:
            entry = logprobs[index]
            top = [
                (str(alt.get("token")), math.exp(float(alt["logprob"])))
                for alt in entry.get("top_logprobs", [])
            ]
            return AnswerToken(
                index, str(entry.get("token")), math.exp(float(entry["logprob"])), top
            )
    return None


@dataclass(frozen=True)
class Relevance:
    p_yes: float
    p_no: float
    has_yes_or_no: bool
    first_token: str | None


def relevance_from_logprobs(logprobs: Sequence[Mapping[str, Any]] | None) -> Relevance:
    """P(exact "Yes") among the first token's top_logprobs; 0 when absent."""
    if not logprobs:
        return Relevance(0.0, 0.0, False, None)
    first = logprobs[0]
    alternatives = first.get("top_logprobs") or [first]
    probs = {str(a.get("token")): math.exp(float(a["logprob"])) for a in alternatives}
    p_yes, p_no = probs.get(YES, 0.0), probs.get(NO, 0.0)
    return Relevance(p_yes, p_no, YES in probs or NO in probs, str(first.get("token")))


# ---- request specs (GT never enters: item question or original question text) -------------
def observer_lp_spec(
    question: Mapping[str, Any], level: int, image: bytes, item: Mapping[str, Any]
) -> RequestSpec:
    return dataclasses.replace(
        item_spec(question, level, image, item, namespace=LP_NAMESPACE),
        logprobs=TOP_LOGPROBS,
    )


def direct_lp_spec(
    question: Mapping[str, Any], level: int, image: bytes
) -> RequestSpec:
    spec = build_level_spec(DIRECT_ROLE, question, level, image)
    return dataclasses.replace(spec, namespace=LP_NAMESPACE, logprobs=TOP_LOGPROBS)


def relevance_spec(
    question_id: str, level: int, image: bytes, text: str, target: str
) -> RequestSpec:
    """``target`` = item id or "direct"; plain-text reply, temperature 0."""
    return RequestSpec(
        role=RELEVANCE_ROLE,
        question_id=question_id,
        level=level,
        input_data={"question": text},
        image_bytes=image,
        suffix=f"_{target}",
        logprobs=TOP_LOGPROBS,
        raw_text=True,
    )


def _accept_text(obj: Any) -> tuple[Any, list[str]]:
    return obj, []


# ---- runner -----------------------------------------------------------------------------------
class LogprobRunner:
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
    ) -> list[dict]:
        plan, last_ok = self.plan(question)
        return [self.level_row(question, level, plan, last_ok) for level in levels]

    def level_row(
        self, question: Mapping[str, Any], level: int, plan: PlanOutcome, last_ok: bool
    ) -> dict[str, Any]:
        qid = str(question["question_id"])
        image = self.series.bytes_for(str(question["image_id"]), level)
        tally = CallTally()
        reader = self._run(
            build_level_spec(READER_ROLE, question, level, image), validate_reader
        )
        direct_v1 = self._run(
            build_level_spec(DIRECT_ROLE, question, level, image), validate_direct
        )
        direct_lp = self._run(direct_lp_spec(question, level, image), validate_direct)
        direct_rel = self._run(
            relevance_spec(qid, level, image, str(question["question"]), "direct"),
            _accept_text,
        )
        tally.add("direct_lp", direct_lp)
        tally.add("relevance", direct_rel)
        items: dict[str, dict[str, Any]] = {}
        for item in plan.items:
            obs = self._run(
                observer_lp_spec(question, level, image, item), validate_observation
            )
            rel = self._run(
                relevance_spec(qid, level, image, str(item["question"]), item["id"]),
                _accept_text,
            )
            tally.add("observer", obs)
            tally.add("relevance", rel)
            items[item["id"]] = item_signals(obs, rel)
        return v4_row(
            question,
            level,
            plan,
            last_ok,
            reader,
            direct_v1,
            item_signals(direct_lp, direct_rel),
            items,
            tally,
        )


def item_signals(obs: Mapping[str, Any] | None, rel: Mapping[str, Any] | None) -> dict:
    """Parsed answer + P_ans + Rel for one observation (item or direct)."""
    valid = obs is not None and obs.get("status") == STATUS_VALID
    parsed = obs["parsed"] if valid and obs else {}
    token = locate_answer_token(obs.get("logprobs") if valid and obs else None)
    relevance = relevance_from_logprobs(
        rel.get("logprobs")
        if rel is not None and rel.get("status") == STATUS_VALID
        else None
    )
    p_ans = token.prob if token else 0.0
    return {
        "valid": valid,
        "answer": parsed.get("answer"),
        "resolved": bool(parsed.get("resolved", valid)),
        "confidence": parsed.get("confidence"),
        "located": token is not None,
        "token": token.token if token else None,
        "pans": p_ans,
        "rel": relevance.p_yes,
        "rel_no": relevance.p_no,
        "rel_has_yes_no": relevance.has_yes_or_no,
        "rel_text": (rel.get("parsed") or {}).get("text") if rel else None,
        "product": p_ans * relevance.p_yes,
    }


def gated_scores(
    plan: PlanOutcome, items: Mapping[str, Mapping[str, Any]]
) -> dict[str, float]:
    """v1 gate with confidence := P_ans x Rel / P_ans / Rel; min and mean of each."""
    out: dict[str, float] = {}
    for kind, key in (("checklist_v4", "product"), ("pans", "pans"), ("rel", "rel")):
        observations = [
            {
                "id": i,
                "answer": s["answer"],
                "resolved": s["resolved"],
                "confidence": s[key],
            }
            for i, s in items.items()
            if s["valid"]
        ]
        r = effective_confidences(plan.items, observations) if not plan.failed else {}
        out[f"{kind}_min"], out[f"{kind}_mean"] = aggregate(r)
    return out


def v4_row(
    question: Mapping[str, Any],
    level: int,
    plan: PlanOutcome,
    last_ok: bool,
    reader: Mapping[str, Any] | None,
    direct_v1: Mapping[str, Any] | None,
    direct: Mapping[str, Any],
    items: Mapping[str, Mapping[str, Any]],
    tally: CallTally,
) -> dict[str, Any]:
    reader_ok = reader is not None and reader.get("status") == STATUS_VALID
    v1_ok = direct_v1 is not None and direct_v1.get("status") == STATUS_VALID
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
        "direct_score": float(direct_v1["parsed"]["confidence"])
        if v1_ok and direct_v1
        else 0.0,
        "direct_lp_answer": direct["answer"],
        "direct_located": direct["located"],
        "direct_rel_has_yes_no": direct["rel_has_yes_no"],
        "direct_pans": direct["pans"],
        "direct_rel": direct["rel"],
        "direct_v4": direct["product"],
        "planner_status": plan.status,
        "planner_initial_valid": plan.initial_valid,
        "planner_attempts": plan.attempts,
        "n_items": len(plan.items),
        **plan_quality(plan, last_ok),
        **gated_scores(plan, items),
        **dict(tally.counts),
    }
    for index, item in enumerate(plan.items[:MAX_ITEMS_COLUMNS], start=1):
        sig = items[item["id"]]
        row[f"item{index}_question"] = item["question"]
        for key in (
            "answer",
            "resolved",
            "confidence",
            "valid",
            "located",
            "token",
            "pans",
            "rel",
            "rel_no",
            "rel_has_yes_no",
            "product",
        ):
            row[f"item{index}_{key}"] = sig[key]
    return row


def planned_v4_requests(n_questions: int, n_levels: int, max_items: int) -> int:
    """New calls upper bound: per level (items x (observer + relevance) + direct + rel)."""
    return n_questions * n_levels * (2 * max_items + 2)
