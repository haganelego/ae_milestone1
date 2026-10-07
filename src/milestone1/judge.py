"""ROBOVIS2027 judgment step (plan §5/§7): graph + checks -> per-check judgments.

Text-only by default; ``image_path`` (variant F1v) additionally sends the
current-state JPEG. The model does not choose the stopping time; a fixed rule
(stop_policy.py) aggregates these judgments.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from milestone1.check_generator import (
    COMMON_PREAMBLE,
    IMAGE_SENT_AS,
    PROMPT_V1,
    PROMPT_V2,
    PROMPT_VERSIONS,
    STRING_HINT_TEMPLATE,
    CheckLimits,
    check_ids,
    load_cached,
    outcome_fields,
    render_graph,
    write_json_atomic,
)
from milestone1.ollama_client import (
    ChatClient,
    encode_image_file,
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

# Image judge (F1v): the text-only sentence is swapped for an image-grounded one.
TEXT_ONLY_EVIDENCE = (
    "Use only the graph and evidence records. Do not invent new image evidence."
)
IMAGE_EVIDENCE = (
    "Use only the graph, the evidence records and the attached image of the "
    "current view. Do not invent evidence that is not visible in the image."
)
JUDGE_IMAGE_NOTE = "The attached image shows the current view described by this graph."
BASIS_RULE_V1 = """- basis: integer 0, 1 or 2 (0 = no support, 1 = generic co-occurrence or weak
  support, 2 = specific observed relations support it)."""
# v2 basis/benefit semantics (team-lead addendum 2026-10-07, from A2 item-level
# diagnosis: v1 4b judge treated the two grades as anti-correlated).
BASIS_RULE_V2 = """- basis: integer 0, 1 or 2: how concrete the evidence is that this check is a
  real open question in THIS view (0 = none; 1 = generic co-occurrence or a
  single weak cue; 2 = specific observed relations or visible partial evidence)."""
BENEFIT_RULE_V1 = """- resolution_benefit: integer 0, 1 or 2 (0 = none, 1 = possible, 2 = concrete
  expected benefit from higher resolution of the same view)."""
BENEFIT_RULE_V2 = """- resolution_benefit: integer 0, 1 or 2: whether more pixels in the same view
  would change the answer (0 = no; 1 = unclear; 2 = yes: the region anchor is
  small, blurry or low-detail in the current view and the target would become
  visible with more detail)."""
AXES_NOTE_V2 = """basis and resolution_benefit are independent axes: judge each on its own. A
high basis does not imply a low resolution_benefit, and an item can score 2 on
both.
Example (synthetic, illustrative only; not about the supplied input):
  check: {"id": "h1", "kind": "missing_object", "candidate": "small object held
  by the man", "anchor_ids": ["v1", "v2"], "relation_ids": ["e1"],
  "region_anchor_id": "v1", "evidence": "man holding an unidentified object near
  the table"}
  judgment: {"id": "h1", "status": "unresolved", "basis": 2,
  "resolution_benefit": 2, "limitation": "low_resolution", "reason": "Specific
  'man holding X' relation supports it; the held object is tiny and blurry, so
  more detail would reveal what it is."}"""
BENEFIT_IMAGE_RULE = (
    "  Judge resolution_benefit from how the region looks in the attached image."
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
{basis_rule}
{benefit_rule}
- limitation: one of "low_resolution", "occluded", "out_of_frame", "other", "unknown".
- reason: plain string, at most {max_chars} characters."""

LIMITATION_ENUM_NOTE_V2 = (
    "limitation must be exactly one of: "
    + ", ".join(f'"{v}"' for v in LIMITATIONS)
    + "."
)

CHECKS_HEADER = "Checks to judge (evidence records derived from the graph above):"
IDS_NOTE = "Return one item for each of these check ids, each exactly once: {ids}"


def judge_system_prompt(
    limits: CheckLimits, prompt_version: str = PROMPT_V1, with_image: bool = False
) -> str:
    """v1 text-only prompt is byte-identical to the original plan §7 version."""
    is_v2 = prompt_version == PROMPT_V2
    basis_rule = BASIS_RULE_V2 if is_v2 else BASIS_RULE_V1
    benefit_rule = BENEFIT_RULE_V2 if is_v2 else BENEFIT_RULE_V1
    task = JUDGE_TASK
    if with_image:
        task = task.replace(TEXT_ONLY_EVIDENCE, IMAGE_EVIDENCE)
        if is_v2:
            benefit_rule = f"{benefit_rule}\n{BENEFIT_IMAGE_RULE}"
    schema = JUDGE_SCHEMA_TEMPLATE.format(
        max_chars=limits.max_string_chars,
        basis_rule=basis_rule,
        benefit_rule=benefit_rule,
    )
    if not is_v2:
        return f"{COMMON_PREAMBLE}\n\n{task}\n\n{schema}"
    hint = STRING_HINT_TEMPLATE.format(chars=limits.string_hint_chars)
    return (
        f"{COMMON_PREAMBLE}\n\n{task}\n\n{AXES_NOTE_V2}\n\n"
        f"{hint}\n{LIMITATION_ENUM_NOTE_V2}\n{schema}"
    )


def render_checks(checks: dict[str, Any]) -> str:
    return json.dumps(checks, ensure_ascii=False, indent=1)


def build_judge_messages(
    graph_text: str,
    checks: dict[str, Any],
    limits: CheckLimits,
    prompt_version: str = PROMPT_V1,
    image_b64: str | None = None,
) -> list[dict[str, Any]]:
    ids_note = IDS_NOTE.format(ids=", ".join(check_ids(checks)))
    user_text = (
        f"{graph_text}\n\n{CHECKS_HEADER}\n{render_checks(checks)}\n\n{ids_note}"
    )
    if image_b64 is not None:
        user_text = f"{JUDGE_IMAGE_NOTE}\n\n{user_text}"
    user: dict[str, Any] = {"role": "user", "content": user_text}
    if image_b64 is not None:
        user["images"] = [image_b64]
    system = judge_system_prompt(limits, prompt_version, image_b64 is not None)
    return [{"role": "system", "content": system}, user]


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
    prompt_version: str = PROMPT_V1,
    image_path: Path | None = None,
    record_extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Judge one state's checks. Returns (record, from_cache).

    ``limits`` and ``prompt_version`` must match those used for the checks
    (same rendering, so anchors resolve). ``image_path`` -> image judge (F1v).

    - upstream check failure -> failure record, no model call;
    - zero checks -> empty items, no model call (``skipped_empty``).
    """
    if prompt_version not in PROMPT_VERSIONS:
        raise ValueError(f"unknown prompt version {prompt_version!r}")
    checks_version = checks_record.get("prompt_version", PROMPT_V1)
    if checks_version != prompt_version:
        raise ValueError(
            f"checks were generated with {checks_version}, judge uses {prompt_version}"
        )
    base: dict[str, Any] = {
        "model": model,
        "prompt_version": prompt_version,
        "judge_saw_image": image_path is not None,
        "path": checks_record.get("path"),
        "image_id": graph.get("image_id"),
        "level": graph.get("level"),
        "checks_cache_key": checks_record.get("cache_key"),
        "options": client.settings.record(),
        "skipped_empty": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(record_extra or {}),
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
    image_b64, image_sha = (None, None)
    if image_path is not None:
        image_b64, image_sha = encode_image_file(image_path)
    system_prompt = judge_system_prompt(limits, prompt_version, image_path is not None)
    prompt_sha = sha256_text(system_prompt)
    graph_sha = sha256_text(rendered.text)
    key_fields: dict[str, Any] = {
        "stage": "judge",
        "model": model,
        "prompt_sha256": prompt_sha,
        "options": base["options"],
        "graph_sha256": graph_sha,
        "checks_sha256": sha256_json(checks),
    }
    if image_sha is not None:  # keep text-only v1 keys identical to earlier runs
        key_fields["image_sha256"] = image_sha
    if prompt_version != PROMPT_V1:
        key_fields["prompt_version"] = prompt_version
    cache_key = sha256_json(key_fields)
    cached = load_cached(out_path, cache_key)
    if cached is not None:
        return cached, True

    messages = build_judge_messages(
        rendered.text, checks, limits, prompt_version, image_b64
    )
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
        "image_sha256": image_sha,
        "image_sent_as": IMAGE_SENT_AS if image_path is not None else None,
        "cache_key": cache_key,
        "items": None if outcome.parsed is None else outcome.parsed["items"],
        **outcome_fields(outcome),
    }
    write_json_atomic(out_path, record)
    return record, False
