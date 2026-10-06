"""ROBOVIS2027 judgment step (plan §5/§7): graph + checks -> per-check judgments.

Text-only (no image). The model does not choose the stopping time; a fixed
rule (stop_policy.py) aggregates these judgments.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from milestone1.check_generator import (
    COMMON_PREAMBLE,
    CheckLimits,
    check_ids,
    load_cached,
    outcome_fields,
    render_graph,
    write_json_atomic,
)
from milestone1.ollama_client import (
    ChatClient,
    sha256_json,
    sha256_text,
    structured_chat,
)

STATUSES = ("resolved", "unsupported", "unresolved")
GRADES = (0, 1, 2)
LIMITATIONS = ("low_resolution", "occluded", "out_of_frame", "other", "unknown")

JUDGE_TASK = (
    "For every supplied check, assess whether it remains unresolved and whether "
    "increasing resolution of the same view has a concrete expected benefit. "
    "Use only the graph and evidence records. Do not invent new image evidence. "
    "Return status, basis (0/1/2), resolution_benefit (0/1/2), limitation and a "
    "short reason. Generic co-occurrence alone is basis=1, not 2. Missing "
    "evidence is unknown, not proof. Off-frame and fully occluded content cannot "
    "be resolved by resolution alone. Do not directly choose the stopping time; "
    "a fixed rule will aggregate these judgments."
)

JUDGE_SCHEMA_TEMPLATE = """Return exactly one JSON object (no markdown, no text outside JSON):
{{
  "items": [
    {{"id": "h1", "status": "unresolved", "basis": 1, "resolution_benefit": 2,
      "limitation": "low_resolution", "reason": "<short reason>"}}
  ]
}}
Field rules:
- items: exactly one entry for every supplied check id, each id exactly once.
- status: one of "resolved", "unsupported", "unresolved".
- basis: integer 0, 1 or 2 (0 = no support, 1 = generic co-occurrence or weak
  support, 2 = specific observed relations support it).
- resolution_benefit: integer 0, 1 or 2 (0 = none, 1 = possible, 2 = concrete
  expected benefit from higher resolution of the same view).
- limitation: one of "low_resolution", "occluded", "out_of_frame", "other", "unknown".
- reason: plain string, at most {max_chars} characters."""

CHECKS_HEADER = "Checks to judge (evidence records derived from the graph above):"
IDS_NOTE = "Return one item for each of these check ids, each exactly once: {ids}"


def judge_system_prompt(limits: CheckLimits) -> str:
    schema = JUDGE_SCHEMA_TEMPLATE.format(max_chars=limits.max_string_chars)
    return f"{COMMON_PREAMBLE}\n\n{JUDGE_TASK}\n\n{schema}"


def render_checks(checks: dict[str, Any]) -> str:
    return json.dumps(checks, ensure_ascii=False, indent=1)


def build_judge_messages(
    graph_text: str, checks: dict[str, Any], limits: CheckLimits
) -> list[dict[str, Any]]:
    ids_note = IDS_NOTE.format(ids=", ".join(check_ids(checks)))
    user_text = (
        f"{graph_text}\n\n{CHECKS_HEADER}\n{render_checks(checks)}\n\n{ids_note}"
    )
    return [
        {"role": "system", "content": judge_system_prompt(limits)},
        {"role": "user", "content": user_text},
    ]


def _is_grade(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value in GRADES


def validate_judgment(
    data: Any, expected_ids: list[str], limits: CheckLimits
) -> tuple[dict[str, Any] | None, list[str]]:
    """Every expected id exactly once; enums and grades as in the contract."""
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        return None, ["top level must be an object with an 'items' list"]
    errors: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(data["items"]):
        where = f"items[{idx}]"
        if not isinstance(item, dict):
            errors.append(f"{where} must be an object")
            continue
        item_id = item.get("id")
        if item_id not in expected_ids:
            errors.append(f"{where}.id unknown: {item_id!r}")
            continue
        if item_id in by_id:
            errors.append(f"duplicate id {item_id}")
            continue
        if item.get("status") not in STATUSES:
            errors.append(f"{where}.status invalid: {item.get('status')!r}")
        for key in ("basis", "resolution_benefit"):
            if not _is_grade(item.get(key)):
                errors.append(f"{where}.{key} must be 0, 1 or 2: {item.get(key)!r}")
        if item.get("limitation") not in LIMITATIONS:
            errors.append(f"{where}.limitation invalid: {item.get('limitation')!r}")
        reason = item.get("reason")
        if not isinstance(reason, str):
            errors.append(f"{where}.reason must be a string")
        elif len(reason) > limits.max_string_chars:
            errors.append(f"{where}.reason longer than {limits.max_string_chars} chars")
        by_id[item_id] = {
            k: item.get(k)
            for k in (
                "id",
                "status",
                "basis",
                "resolution_benefit",
                "limitation",
                "reason",
            )
        }
    missing = [i for i in expected_ids if i not in by_id]
    if missing:
        errors.append(f"missing ids: {missing}")
    if errors:
        return None, errors
    return {"items": [by_id[i] for i in expected_ids]}, []


def judge_checks(
    graph: dict[str, Any],
    checks_record: dict[str, Any],
    *,
    client: ChatClient,
    model: str,
    limits: CheckLimits,
    schema_retries: int,
    out_path: Path,
) -> tuple[dict[str, Any], bool]:
    """Judge one state's checks. Returns (record, from_cache).

    - upstream check failure -> failure record, no model call;
    - zero checks -> empty items, no model call (``skipped_empty``).
    """
    base: dict[str, Any] = {
        "model": model,
        "path": checks_record.get("path"),
        "image_id": graph.get("image_id"),
        "level": graph.get("level"),
        "checks_cache_key": checks_record.get("cache_key"),
        "options": client.settings.record(),
        "skipped_empty": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if checks_record.get("failure") or checks_record.get("checks") is None:
        record = {
            **base,
            "items": None,
            "attempts": 0,
            "failure": True,
            "error": "upstream checks failure",
        }
        write_json_atomic(out_path, record)
        return record, False
    checks = checks_record["checks"]
    expected = check_ids(checks)
    if not expected:
        record = {
            **base,
            "items": [],
            "attempts": 0,
            "skipped_empty": True,
            "failure": False,
            "error": None,
        }
        write_json_atomic(out_path, record)
        return record, False

    rendered = render_graph(graph, limits)
    prompt_sha = sha256_text(judge_system_prompt(limits))
    graph_sha = sha256_text(rendered.text)
    cache_key = sha256_json(
        {
            "stage": "judge",
            "model": model,
            "prompt_sha256": prompt_sha,
            "options": base["options"],
            "graph_sha256": graph_sha,
            "checks_sha256": sha256_json(checks),
        }
    )
    cached = load_cached(out_path, cache_key)
    if cached is not None:
        return cached, True

    messages = build_judge_messages(rendered.text, checks, limits)
    outcome = structured_chat(
        client,
        model,
        messages,
        lambda data: validate_judgment(data, expected, limits),
        schema_retries,
    )
    record = {
        **base,
        "prompt_sha256": prompt_sha,
        "graph_sha256": graph_sha,
        "cache_key": cache_key,
        "items": None if outcome.parsed is None else outcome.parsed["items"],
        **outcome_fields(outcome),
    }
    write_json_atomic(out_path, record)
    return record, False
