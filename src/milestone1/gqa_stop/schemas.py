"""Strict response validation for every role (plan §5, §9.2, test 8).

Validators take the object produced by ``json.loads`` on the whole response
and return ``(normalized, errors)``; an empty error list means valid. Nothing
is coerced: booleans, numeric strings, NaN or out-of-range confidences fail.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0

Validated = tuple[Any, list[str]]


def check_confidence(value: Any, where: str) -> tuple[float | None, list[str]]:
    """Finite real in [0, 1]. ``bool`` is rejected even though it is an int."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None, [
            f"{where}: confidence must be a number, got {type(value).__name__}"
        ]
    number = float(value)
    if not math.isfinite(number):
        return None, [f"{where}: confidence is not finite"]
    if not CONFIDENCE_MIN <= number <= CONFIDENCE_MAX:
        return None, [f"{where}: confidence {number} outside [0, 1]"]
    return number, []


def check_answer(value: Any, where: str) -> tuple[str | None, list[str]]:
    if not isinstance(value, str):
        return None, [f"{where}: answer must be a string"]
    if not value.strip():
        return None, [f"{where}: answer is empty"]
    return value, []


def _require_object(obj: Any) -> list[str]:
    return [] if isinstance(obj, dict) else ["response is not a JSON object"]


def validate_reader(obj: Any) -> Validated:
    errors = _require_object(obj)
    if errors:
        return None, errors
    answer, errors = check_answer(obj.get("answer"), "answer")
    return ({"answer": answer} if not errors else None), errors


def validate_direct(obj: Any) -> Validated:
    errors = _require_object(obj)
    if errors:
        return None, errors
    answer, answer_errors = check_answer(obj.get("answer"), "answer")
    confidence, conf_errors = check_confidence(obj.get("confidence"), "confidence")
    errors = answer_errors + conf_errors
    if errors:
        return None, errors
    return {"answer": answer, "confidence": confidence}, []


def validate_plan(obj: Any, min_items: int, max_items: int) -> Validated:
    """2-3 items; unique ids; depends_on references earlier ids only (acyclic)."""
    errors = _require_object(obj)
    if errors:
        return None, errors
    items = obj.get("items")
    if not isinstance(items, list):
        return None, ["items must be a list"]
    if not min_items <= len(items) <= max_items:
        errors.append(f"expected {min_items}-{max_items} items, got {len(items)}")
    seen: list[str] = []
    all_ids = {it.get("id") for it in items if isinstance(it, dict)}
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        item_errors, row = _validate_plan_item(item, index, seen, all_ids)
        errors.extend(item_errors)
        if row is not None:
            seen.append(row["id"])
            normalized.append(row)
    if errors:
        return None, errors
    return {"items": normalized}, []


def _validate_plan_item(
    item: Any, index: int, seen: Sequence[str], all_ids: set[Any]
) -> tuple[list[str], dict[str, Any] | None]:
    where = f"items[{index}]"
    if not isinstance(item, dict):
        return [f"{where}: not an object"], None
    item_id = item.get("id")
    question = item.get("question")
    depends = item.get("depends_on")
    errors: list[str] = []
    if not isinstance(item_id, str) or not item_id.strip():
        errors.append(f"{where}: id must be a non-empty string")
    elif item_id in seen:
        errors.append(f"{where}: duplicate id {item_id!r}")
    if not isinstance(question, str) or not question.strip():
        errors.append(f"{where}: question must be a non-empty string")
    if not isinstance(depends, list) or not all(isinstance(d, str) for d in depends):
        errors.append(f"{where}: depends_on must be a list of strings")
        depends = []
    for dep in depends:
        if dep == item_id:
            errors.append(f"{where}: cycle (depends on itself)")
        elif dep in seen:
            continue
        elif dep in all_ids:
            errors.append(f"{where}: cycle/forward dependency on later id {dep!r}")
        else:
            errors.append(f"{where}: unknown dependency {dep!r}")
    if errors:
        return errors, None
    return [], {"id": item_id, "question": question, "depends_on": list(depends)}


def validate_checklist(obj: Any, plan_ids: Sequence[str]) -> Validated:
    """All plan ids exactly once, in input order; answer/resolved/confidence typed."""
    errors = _require_object(obj)
    if errors:
        return None, errors
    items = obj.get("items")
    if not isinstance(items, list):
        return None, ["items must be a list"]
    ids = [it.get("id") if isinstance(it, dict) else None for it in items]
    counts = Counter(ids)
    expected = list(plan_ids)
    errors.extend(f"missing id {i!r}" for i in expected if counts[i] == 0)
    errors.extend(f"duplicate id {i!r}" for i in expected if counts[i] > 1)
    errors.extend(f"extra id {i!r}" for i in counts if i not in expected)
    if not errors and ids != expected:
        errors.append(f"ids out of order: {ids} != {expected}")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        item_errors, row = _validate_check_item(item, index)
        errors.extend(item_errors)
        if row is not None:
            normalized.append(row)
    if errors:
        return None, errors
    return {"items": normalized}, []


def _validate_check_item(
    item: Any, index: int
) -> tuple[list[str], dict[str, Any] | None]:
    where = f"items[{index}]"
    if not isinstance(item, dict):
        return [f"{where}: not an object"], None
    item_id = item.get("id")
    if not isinstance(item_id, str) or not item_id:
        return [f"{where}: id must be a non-empty string"], None
    answer, errors = check_answer(item.get("answer"), where)
    resolved = item.get("resolved")
    if not isinstance(resolved, bool):
        errors.append(f"{where}: resolved must be a boolean")
    confidence, conf_errors = check_confidence(item.get("confidence"), where)
    errors.extend(conf_errors)
    if errors:
        return errors, None
    row = {
        "id": item_id,
        "answer": answer,
        "resolved": resolved,
        "confidence": confidence,
    }
    return [], row
