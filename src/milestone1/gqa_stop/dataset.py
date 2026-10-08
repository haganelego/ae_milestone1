"""GQA eligibility filter, schema mapping and duplicate-image grouping.

Plan §3.1 with the §16 A1 correction: the final operation is always
``query`` and the requested attribute lives in its ``argument``.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .common import read_json, sha256_file

TARGET_ATTRIBUTES = ("color", "material", "shape")
FORBIDDEN_OPERATION_HEADS = frozenset(
    {"exist", "verify", "choose", "or", "and", "same", "different", "common", "compare"}
)
NEGATION_MARKER = "not("
YES_NO_ANSWERS = frozenset({"yes", "no"})
# Question-text consistency (plan §3.1: metadata AND question text must agree).
ATTRIBUTE_TEXT_PATTERNS: dict[str, re.Pattern[str]] = {
    "color": re.compile(r"\bcolou?r(s|ed|ful)?\b"),
    "material": re.compile(r"\b(materials?|made|make|makes)\b"),
    "shape": re.compile(r"\b(shapes?|shaped)\b"),
}
RELATE_OPERATION = "relate"

# Exclusion reasons in evaluation order (first failing reason is recorded).
REASONS = (
    "not_structural_query",
    "not_semantic_attr",
    "final_operation_not_query",
    "final_argument_not_target_attribute",
    "forbidden_operation",
    "negation",
    "too_many_relations",
    "too_many_operations",
    "relate_count_outside_cells",
    "answer_empty",
    "answer_yes_no",
    "answer_numeric",
    "question_text_attribute_mismatch",
    "answer_word_in_question",
)
ELIGIBLE = "eligible"


def normalize_text(text: str) -> str:
    """Same normalisation as the evaluator (NFKC, lower, strip, collapse)."""
    return " ".join(unicodedata.normalize("NFKC", text).lower().strip().split())


@dataclass(frozen=True)
class QuestionRecord:
    """One eligible GQA question. GT fields are kept for selection/scoring only."""

    question_id: str
    image_id: str
    question: str
    attribute: str
    relate_count: int
    operation_count: int
    detailed_type: str
    answer: str
    full_answer: str

    @property
    def cell(self) -> str:
        return cell_name(self.attribute, self.relate_count)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "cell": self.cell}

    def public(self) -> dict[str, Any]:
        """Fields safe to hand to inference code (no GT)."""
        hidden = {"answer", "full_answer"}
        return {k: v for k, v in self.to_dict().items() if k not in hidden}


def cell_name(attribute: str, relate_count: int) -> str:
    return f"{attribute}|r{relate_count}"


def operation_head(operation: str) -> str:
    return operation.split(" ")[0]


def classify(raw: Mapping[str, Any], filter_cfg: Mapping[str, Any]) -> str:
    """Return ``ELIGIBLE`` or the first exclusion reason for one raw question."""
    types = raw.get("types", {})
    semantic = raw.get("semantic") or []
    if types.get("structural") != filter_cfg["structural"]:
        return "not_structural_query"
    if types.get("semantic") != filter_cfg["semantic_type"]:
        return "not_semantic_attr"
    if not semantic or semantic[-1].get("operation") != filter_cfg["final_operation"]:
        return "final_operation_not_query"
    attribute = semantic[-1].get("argument")
    if attribute not in filter_cfg["final_argument_in"]:
        return "final_argument_not_target_attribute"
    heads = [operation_head(op.get("operation", "")) for op in semantic]
    if any(h in FORBIDDEN_OPERATION_HEADS for h in heads):
        return "forbidden_operation"
    negation = filter_cfg.get("exclude_negation", True)
    if negation and any(NEGATION_MARKER in str(op.get("argument")) for op in semantic):
        return "negation"
    relates = heads.count(RELATE_OPERATION)
    if relates > int(filter_cfg["max_relations"]):
        return "too_many_relations"
    if len(semantic) > int(filter_cfg["max_operations"]):
        return "too_many_operations"
    if relates > 1:
        return "relate_count_outside_cells"
    return _classify_answer(raw, attribute, filter_cfg)


def _classify_answer(
    raw: Mapping[str, Any], attribute: str, filter_cfg: Mapping[str, Any]
) -> str:
    answer = normalize_text(str(raw.get("answer", "")))
    question = normalize_text(str(raw.get("question", "")))
    if not answer:
        return "answer_empty"
    if answer in YES_NO_ANSWERS:
        return "answer_yes_no"
    if answer.replace(".", "", 1).isdigit():
        return "answer_numeric"
    text_check = filter_cfg.get("require_question_text_attribute_match", True)
    if text_check and not ATTRIBUTE_TEXT_PATTERNS[attribute].search(question):
        return "question_text_attribute_mismatch"
    if filter_cfg.get("exclude_answer_word_in_question", True) and re.search(
        rf"\b{re.escape(answer)}\b", question
    ):
        return "answer_word_in_question"
    return ELIGIBLE


def to_record(question_id: str, raw: Mapping[str, Any]) -> QuestionRecord:
    semantic = raw["semantic"]
    heads = [operation_head(op["operation"]) for op in semantic]
    return QuestionRecord(
        question_id=str(question_id),
        image_id=str(raw["imageId"]),
        question=str(raw["question"]),
        attribute=str(semantic[-1]["argument"]),
        relate_count=heads.count(RELATE_OPERATION),
        operation_count=len(semantic),
        detailed_type=str(raw.get("types", {}).get("detailed", "")),
        answer=str(raw["answer"]),
        full_answer=str(raw.get("fullAnswer", "")),
    )


@dataclass(frozen=True)
class EligibilityResult:
    records: list[QuestionRecord]
    reason_counts: dict[str, int]
    schema_mapping: dict[str, Any]


def build_eligible_pool(
    questions: Mapping[str, Mapping[str, Any]], filter_cfg: Mapping[str, Any]
) -> EligibilityResult:
    """Classify every question; returns eligible records sorted by question id."""
    reasons: Counter[str] = Counter()
    structural: Counter[str] = Counter()
    semantic_types: Counter[str] = Counter()
    detailed: Counter[str] = Counter()
    final_args: Counter[str] = Counter()
    answers: dict[str, Counter[str]] = defaultdict(Counter)
    records: list[QuestionRecord] = []
    for qid in sorted(questions):
        raw = questions[qid]
        types = raw.get("types", {})
        structural[str(types.get("structural"))] += 1
        semantic_types[str(types.get("semantic"))] += 1
        reason = classify(raw, filter_cfg)
        reasons[reason] += 1
        if reason != ELIGIBLE:
            continue
        record = to_record(qid, raw)
        records.append(record)
        detailed[record.detailed_type] += 1
        final_args[record.attribute] += 1
        answers[record.attribute][normalize_text(record.answer)] += 1
    mapping = {
        "filter_rule": (
            "types.structural == structural and types.semantic == semantic_type and "
            "semantic[-1].operation == final_operation and semantic[-1].argument in "
            "final_argument_in (plan §16 A1); then §3.1 exclusions in REASONS order"
        ),
        "filter_config": dict(filter_cfg),
        "reason_order": list(REASONS),
        "forbidden_operation_heads": sorted(FORBIDDEN_OPERATION_HEADS),
        "attribute_text_patterns": {
            k: v.pattern for k, v in ATTRIBUTE_TEXT_PATTERNS.items()
        },
        "observed_structural_types": dict(structural),
        "observed_semantic_types": dict(semantic_types),
        "eligible_detailed_types_recorded_only": dict(detailed),
        "eligible_final_arguments": dict(final_args),
        "eligible_answer_vocabulary": {k: dict(v) for k, v in answers.items()},
        "exclusion_counts": dict(reasons),
        "total_questions": len(questions),
        "eligible_questions": len(records),
        "eligible_images": len({r.image_id for r in records}),
        "cell_counts": dict(Counter(r.cell for r in records)),
    }
    return EligibilityResult(records, dict(reasons), mapping)


def load_questions(path: str | Path) -> dict[str, Any]:
    return read_json(path)


def image_path(images_root: Path, image_id: str) -> Path:
    return images_root / f"{image_id}.jpg"


def compute_image_hashes(
    image_ids: Iterable[str],
    images_root: Path,
    cache: Mapping[str, str] | None = None,
    hasher: Callable[[Path], str] = sha256_file,
) -> dict[str, str]:
    """sha256 of every source jpg (cached by image id). Missing files -> ''."""
    known = dict(cache or {})
    result: dict[str, str] = {}
    for image_id in sorted(set(image_ids)):
        if image_id in known:
            result[image_id] = known[image_id]
            continue
        path = image_path(images_root, image_id)
        result[image_id] = hasher(path) if path.exists() else ""
    return result


def duplicate_groups(image_hashes: Mapping[str, str]) -> dict[str, str]:
    """Map image id -> group id; identical bytes share the smallest image id."""
    by_hash: dict[str, list[str]] = defaultdict(list)
    for image_id, digest in image_hashes.items():
        key = digest if digest else f"missing:{image_id}"
        by_hash[key].append(image_id)
    group_of: dict[str, str] = {}
    for members in by_hash.values():
        group_id = min(members)
        for image_id in members:
            group_of[image_id] = group_id
    return group_of
