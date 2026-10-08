"""v2 consistency-stop roles and per-level score (plan §17).

Per question a text-only planner_v2 call fixes {target, synonyms, attribute,
referring_phrase}. Per level three independent calls follow, none of which
sees the reader/direct answers:
  identify   (image)  - the referring phrase, or "list the main objects";
  attr_blind (image)  - the attribute of the referring phrase (or of the target
                        when there is no relation: degenerate, flagged);
  recompose  (text)   - the original question + the two sub-QA pairs.
Score s = a + b + c with a = target/synonym token match in the identify answer,
b = attr_blind == reader, c = recompose == reader (normalised exact match).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .client import STATUS_VALID, RequestEngine, RequestSpec
from .dataset import normalize_text
from .images import ImageSeries
from .inference import READER_ROLE, CacheMissError, attempt_columns, build_level_spec
from .schemas import validate_reader

PLANNER_V2_ROLE = "planner_v2"
IDENTIFY_ROLE = "identify"
ATTR_BLIND_ROLE = "attr_blind"
RECOMPOSE_ROLE = "recompose"
CONSISTENCY_LEVEL_ROLES = (READER_ROLE, IDENTIFY_ROLE, ATTR_BLIND_ROLE, RECOMPOSE_ROLE)
ATTRIBUTES = ("color", "material", "shape")
MAX_SYNONYMS = 3
MIN_STEM_CHARS = 2
SCORE_COLUMN = "consistency_score"
PLANNER_FAILURE = "planner_failure"
SKIPPED = "skipped"
NOT_RUN = "not_run"
LIST_OBJECTS_QUESTION = (
    "List the main objects visible in this image, up to five, as single nouns."
)
IDENTIFY_TEMPLATE = "What is {phrase}? Answer with the object name."
ATTR_PHRASE_TEMPLATE = "What {attribute} is {phrase}?"
ATTR_TARGET_TEMPLATE = "What {attribute} is the {target}?"
TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---- token matching ------------------------------------------------------------------
def tokens(text: str) -> list[str]:
    """Normalised (NFKC, lower, strip, collapse) alphanumeric tokens."""
    return TOKEN_RE.findall(normalize_text(text))


def token_forms(token: str) -> frozenset[str]:
    """Naive singularisation: the token, minus trailing "s", minus trailing "es"."""
    forms = {token}
    if token.endswith("es") and len(token) - 2 >= MIN_STEM_CHARS:
        forms.add(token[:-2])
    if token.endswith("s") and len(token) - 1 >= MIN_STEM_CHARS:
        forms.add(token[:-1])
    return frozenset(forms)


def phrase_in_tokens(phrase: Sequence[str], words: Sequence[str]) -> bool:
    """Contiguous token match where each position shares a singular form."""
    if not phrase or len(phrase) > len(words):
        return False
    wanted = [token_forms(t) for t in phrase]
    for start in range(len(words) - len(phrase) + 1):
        window = words[start : start + len(phrase)]
        if all(wanted[i] & token_forms(w) for i, w in enumerate(window)):
            return True
    return False


def target_match(answer: str, target: str, synonyms: Sequence[str]) -> bool:
    words = tokens(answer)
    return any(phrase_in_tokens(tokens(name), words) for name in [target, *synonyms])


def answers_equal(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    return normalize_text(left) == normalize_text(right)


# ---- planner_v2 ------------------------------------------------------------------------
def validate_plan_v2(obj: Any) -> tuple[Any, list[str]]:
    if not isinstance(obj, dict):
        return None, ["response is not a JSON object"]
    errors: list[str] = []
    target = obj.get("target")
    synonyms = obj.get("synonyms")
    attribute = obj.get("attribute")
    phrase = obj.get("referring_phrase")
    if not isinstance(target, str) or not tokens(target):
        errors.append("target must be a non-empty string")
    if not isinstance(synonyms, list) or not all(
        isinstance(x, str) and tokens(x) for x in synonyms
    ):
        errors.append("synonyms must be a list of non-empty strings")
    elif len(synonyms) > MAX_SYNONYMS:
        errors.append(f"at most {MAX_SYNONYMS} synonyms")
    if attribute not in ATTRIBUTES:
        errors.append(f"attribute must be one of {list(ATTRIBUTES)}")
    if phrase is not None and (not isinstance(phrase, str) or not tokens(phrase)):
        errors.append("referring_phrase must be a non-empty string or null")
    if errors:
        return None, errors
    # A landmark may legitimately share the target noun ("the object to the right
    # of the other cup"), so naming it is recorded as a diagnostic, not rejected
    # (schema v2; v1 rejected it and failed such plans).
    names = [target, *synonyms]
    names_target = phrase is not None and any(
        phrase_in_tokens(tokens(n), tokens(phrase)) for n in names
    )
    plan = {
        "target": target.strip(),
        "synonyms": [s.strip() for s in synonyms],
        "attribute": attribute,
        "referring_phrase": phrase.strip() if phrase is not None else None,
        "phrase_names_target": bool(names_target),
    }
    return plan, []


@dataclass(frozen=True)
class SubQuestions:
    identify_question: str
    attr_question: str
    blind_degenerate: bool


def build_subquestions(plan: Mapping[str, Any]) -> SubQuestions:
    """Deterministic §17 wording; relate-0 falls back to a degenerate attr question."""
    phrase = plan.get("referring_phrase")
    attribute = plan["attribute"]
    if phrase:
        return SubQuestions(
            IDENTIFY_TEMPLATE.format(phrase=phrase),
            ATTR_PHRASE_TEMPLATE.format(attribute=attribute, phrase=phrase),
            False,
        )
    return SubQuestions(
        LIST_OBJECTS_QUESTION,
        ATTR_TARGET_TEMPLATE.format(attribute=attribute, target=plan["target"]),
        True,
    )


@dataclass(frozen=True)
class PlanV2Outcome:
    question_id: str
    status: str
    plan: dict[str, Any] | None = None
    initial_valid: bool = False
    attempts: int = 0
    subquestions: SubQuestions | None = field(default=None)

    @property
    def failed(self) -> bool:
        return self.status != STATUS_VALID


def plan_v2_from_record(question_id: str, record: Mapping[str, Any]) -> PlanV2Outcome:
    initial = bool(record.get("initial_valid"))
    attempts = int(record.get("attempt", 0))
    parsed = record.get("parsed")
    if record.get("status") != STATUS_VALID or parsed is None:
        return PlanV2Outcome(question_id, PLANNER_FAILURE, None, initial, attempts)
    sub = build_subquestions(parsed)
    return PlanV2Outcome(
        question_id, STATUS_VALID, dict(parsed), initial, attempts, sub
    )


# ---- request specs (GT never enters: only question text or sub-questions) ----------------
def planner_v2_spec(question: Mapping[str, Any]) -> RequestSpec:
    return RequestSpec(
        role=PLANNER_V2_ROLE,
        question_id=str(question["question_id"]),
        level=None,
        input_data={"question": str(question["question"])},
    )


def image_spec(
    role: str, question: Mapping[str, Any], level: int, image: bytes, text: str
) -> RequestSpec:
    """identify / attr_blind: the sub-question only, never the original question."""
    return RequestSpec(
        role=role,
        question_id=str(question["question_id"]),
        level=level,
        input_data={"question": text},
        image_bytes=image,
    )


def recompose_spec(
    question: Mapping[str, Any],
    level: int,
    sub: SubQuestions,
    identify_answer: str,
    attr_answer: str,
) -> RequestSpec:
    """Text only: original question + the two sub-QA pairs (no reader/direct)."""
    return RequestSpec(
        role=RECOMPOSE_ROLE,
        question_id=str(question["question_id"]),
        level=level,
        input_data={
            "question": str(question["question"]),
            "observations": [
                {"question": sub.identify_question, "answer": identify_answer},
                {"question": sub.attr_question, "answer": attr_answer},
            ],
        },
    )


def _answer(record: Mapping[str, Any] | None) -> str | None:
    if record is None or record.get("status") != STATUS_VALID:
        return None
    return str(record["parsed"]["answer"])


def _status(record: Mapping[str, Any] | None, fallback: str) -> str:
    return fallback if record is None else str(record.get("status"))


class ConsistencyRunner:
    """Runs (or replays from cache) the §17 roles for questions x levels."""

    def __init__(
        self, engine: RequestEngine, series: ImageSeries, cache_only: bool = False
    ) -> None:
        self.engine = engine
        self.series = series
        self.cache_only = cache_only

    def _run(self, spec: RequestSpec, validator: Callable[[Any], Any]) -> dict:
        if not self.cache_only:
            return self.engine.run(spec, validator)
        cached = self.engine.lookup(spec)
        if cached is None:
            raise CacheMissError(f"{spec.role} {spec.question_id} L{spec.level}")
        return cached

    def plan(self, question: Mapping[str, Any]) -> PlanV2Outcome:
        record = self._run(planner_v2_spec(question), validate_plan_v2)
        return plan_v2_from_record(str(question["question_id"]), record)

    def run_question(
        self, question: Mapping[str, Any], levels: Sequence[int]
    ) -> list[dict[str, Any]]:
        plan = self.plan(question)
        return [self.level_row(question, level, plan) for level in levels]

    def level_row(
        self, question: Mapping[str, Any], level: int, plan: PlanV2Outcome
    ) -> dict[str, Any]:
        image = self.series.bytes_for(str(question["image_id"]), level)
        reader = self._run(
            build_level_spec(READER_ROLE, question, level, image), validate_reader
        )
        identify = attr = recompose = None
        sub = plan.subquestions
        if sub is not None:
            identify = self._run(
                image_spec(
                    IDENTIFY_ROLE, question, level, image, sub.identify_question
                ),
                validate_reader,
            )
            attr = self._run(
                image_spec(ATTR_BLIND_ROLE, question, level, image, sub.attr_question),
                validate_reader,
            )
            id_answer, attr_answer = _answer(identify), _answer(attr)
            if id_answer is not None and attr_answer is not None:
                recompose = self._run(
                    recompose_spec(question, level, sub, id_answer, attr_answer),
                    validate_reader,
                )
        return consistency_row(question, level, plan, reader, identify, attr, recompose)


def consistency_row(
    question: Mapping[str, Any],
    level: int,
    plan: PlanV2Outcome,
    reader: Mapping[str, Any] | None,
    identify: Mapping[str, Any] | None,
    attr: Mapping[str, Any] | None,
    recompose: Mapping[str, Any] | None,
) -> dict[str, Any]:
    reader_answer = _answer(reader)
    id_answer = _answer(identify)
    planned = plan.plan or {}
    sub = plan.subquestions
    a = id_answer is not None and target_match(
        id_answer, planned["target"], planned["synonyms"]
    )
    b = answers_equal(_answer(attr), reader_answer)
    c = answers_equal(_answer(recompose), reader_answer)
    no_plan = PLANNER_FAILURE if plan.failed else NOT_RUN
    sub_failed = SKIPPED if not plan.failed else PLANNER_FAILURE
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
        "reader_answer": reader_answer,
        "reader_status": _status(reader, NOT_RUN),
        **attempt_columns("reader", reader),
        "planner_status": plan.status,
        "planner_initial_valid": plan.initial_valid,
        "planner_attempts": plan.attempts,
        "target": planned.get("target"),
        "synonyms": "|".join(planned.get("synonyms", [])),
        "plan_attribute": planned.get("attribute"),
        "referring_phrase": planned.get("referring_phrase"),
        "phrase_names_target": planned.get("phrase_names_target"),
        "blind_degenerate": bool(sub.blind_degenerate) if sub else None,
        "identify_question": sub.identify_question if sub else None,
        "attr_question": sub.attr_question if sub else None,
        "identify_answer": id_answer,
        "identify_status": _status(identify, no_plan),
        **attempt_columns("identify", identify),
        "attr_answer": _answer(attr),
        "attr_status": _status(attr, no_plan),
        **attempt_columns("attr", attr),
        "recompose_answer": _answer(recompose),
        "recompose_status": _status(recompose, sub_failed),
        **attempt_columns("recompose", recompose),
        "a": bool(a),
        "b": bool(b),
        "c": bool(c),
        SCORE_COLUMN: int(a) + int(b) + int(c),
    }


def planned_consistency_requests(n_questions: int, n_levels: int) -> int:
    """planner_v2 once + (reader, identify, attr_blind, recompose) per level."""
    return n_questions + n_questions * n_levels * len(CONSISTENCY_LEVEL_ROLES)
