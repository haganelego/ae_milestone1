"""Parameterized LLM scoring engine for ablation studies.

Builds scene-graph sequences with configurable features (relation_on,
confidence_on, history_length) and scores them via ollama /api/chat.
"""

import json
from pathlib import Path
from typing import Any

from milestone1.llm import _parse_llm_json

SATURATION_PROMPT_TEMPLATE = """You are a scene graph analysis expert. You are given a sequence of scene graphs extracted from the same image at increasing resolution levels.

Your task: Judge whether the scene graph has reached "semantic saturation" — i.e., whether further increasing image resolution is unlikely to reveal meaningful new objects or relationships.

INPUT: A JSON array of scene graphs at consecutive resolution levels. Each entry has:
- "level": resolution level (higher = better resolution)
- "objects": detected objects with labels{conf_note_objects}
{rel_note}
CRITERIA for saturation:
1. The set of detected object categories has stabilized (few or no new categories appearing)
{rel_criteria}3. The scene graph structure is converging

OUTPUT: Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{{
  "saturation_score": <float 0.0-1.0, where 1.0 = fully saturated>,
  "stop_decision": <boolean, true if saturation_score >= 0.8>,
  "reason": "<brief explanation of your judgment>"
}}

Scene graph sequence:
"""


def build_prompt(relation_on: bool, confidence_on: bool) -> str:
    """Build the saturation prompt based on ablation flags."""
    conf_note_objects = " and confidence scores" if confidence_on else ""

    if relation_on:
        if confidence_on:
            rel_note = '- "relationships": detected relationships (subject -> predicate -> object) with confidence scores'
        else:
            rel_note = '- "relationships": detected relationships (subject -> predicate -> object)'
        rel_criteria = "2. The set of detected relationships has stabilized\n"
    else:
        rel_note = "(No relationship information is provided in this evaluation.)"
        rel_criteria = ""

    if confidence_on and relation_on:
        # Add confidence stability criterion
        rel_criteria += "3. Confidence scores are not significantly changing\n"
        # Renumber the last criterion
        return SATURATION_PROMPT_TEMPLATE.format(
            conf_note_objects=conf_note_objects,
            rel_note=rel_note,
            rel_criteria=rel_criteria.rstrip("\n") + "\n",
        )

    return SATURATION_PROMPT_TEMPLATE.format(
        conf_note_objects=conf_note_objects,
        rel_note=rel_note,
        rel_criteria=rel_criteria,
    )


def build_sequence(
    image_id: int,
    current_level: int,
    history_length: int,
    pred_root: Path,
    relation_on: bool = True,
    confidence_on: bool = True,
    input_obj_threshold: float = 0.3,
    input_rel_threshold: float = 0.005,
    input_top_k_objects: int = 20,
    input_top_k_relationships: int = 30,
) -> list[dict[str, Any]]:
    """Build a scene graph sequence for LLM input with configurable features."""
    start = max(1, current_level - history_length + 1)
    sequence = []

    for lv in range(start, current_level + 1):
        level_name = f"level_{lv:02d}"
        pred_path = pred_root / str(image_id) / f"{level_name}.json"

        if not pred_path.exists():
            continue

        with open(pred_path) as f:
            pred = json.load(f)

        # Filter objects
        objects = [
            o for o in pred["objects"] if o["confidence"] >= input_obj_threshold
        ][:input_top_k_objects]

        # Build object entries
        if confidence_on:
            obj_entries = [
                {
                    "object_id": o["object_id"],
                    "label": o["label"],
                    "confidence": o["confidence"],
                }
                for o in objects
            ]
        else:
            obj_entries = [
                {
                    "object_id": o["object_id"],
                    "label": o["label"],
                }
                for o in objects
            ]

        entry: dict[str, Any] = {
            "level": lv,
            "objects": obj_entries,
        }

        # Add relationships if enabled
        if relation_on:
            relationships = [
                r
                for r in pred["relationships"]
                if r["confidence"] >= input_rel_threshold
            ][:input_top_k_relationships]

            if confidence_on:
                rel_entries = [
                    {
                        "subject_id": r["subject_id"],
                        "subject_label": r["subject_label"],
                        "predicate": r["predicate"],
                        "object_id": r["object_id"],
                        "object_label": r["object_label"],
                        "confidence": r["confidence"],
                    }
                    for r in relationships
                ]
            else:
                rel_entries = [
                    {
                        "subject_id": r["subject_id"],
                        "subject_label": r["subject_label"],
                        "predicate": r["predicate"],
                        "object_id": r["object_id"],
                        "object_label": r["object_label"],
                    }
                    for r in relationships
                ]
            entry["relationships"] = rel_entries

        sequence.append(entry)

    return sequence


def make_config_slug(
    model: str,
    history_length: int,
    relation_on: bool,
    confidence_on: bool,
) -> str:
    """Create a filesystem-safe slug encoding the ablation config."""
    model_tag = model.replace(":", "-").replace("/", "-")
    rel_tag = "rel_on" if relation_on else "rel_off"
    conf_tag = "conf_on" if confidence_on else "conf_off"
    return f"{model_tag}_h{history_length}_{rel_tag}_{conf_tag}"


def _do_ollama_call(
    full_prompt: str,
    model: str,
    chat_endpoint: str,
    socket_timeout: int,
) -> str:
    """Make the actual ollama API call (runs in a thread for wall-clock timeout)."""
    import requests as req

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": full_prompt},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1024,
            # Long history-5 / top-20-30 prompts reach ~9.2k tokens; ollama's
            # default num_ctx=4096 truncates the prompt to 4095 and leaves zero
            # room to generate, producing a 1-token stub ("Based") that fails to
            # parse. Size the context to fit the largest prompt + generation.
            "num_ctx": 12288,
        },
    }
    resp = req.post(chat_endpoint, json=payload, timeout=socket_timeout)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def score_with_config(
    sequence_json: list[dict[str, Any]],
    prompt: str,
    model: str,
    chat_endpoint: str,
    timeout: int = 120,
) -> dict[str, Any]:
    """Score saturation for a sequence using the parameterized prompt.

    Uses ollama /api/chat with think=false, stream=false.
    Enforces a hard wall-clock timeout via ThreadPoolExecutor to handle
    cases where ollama keeps the connection alive with slow token generation.
    Returns {saturation_score, stop_decision, reason}.
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeout

    full_prompt = prompt + json.dumps(sequence_json, indent=2)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                _do_ollama_call, full_prompt, model, chat_endpoint, timeout
            )
            # Hard wall-clock timeout: timeout + 30s grace
            raw = future.result(timeout=timeout + 30)

        result = _parse_llm_json(raw)

        # Ensure required fields
        if "saturation_score" not in result:
            result["saturation_score"] = 0.0
        if "stop_decision" not in result:
            result["stop_decision"] = result["saturation_score"] >= 0.8
        if "reason" not in result:
            result["reason"] = "no reason provided"

        result["saturation_score"] = max(
            0.0, min(1.0, float(result["saturation_score"]))
        )
        result["stop_decision"] = bool(result["stop_decision"])
        return result

    except FuturesTimeout:
        return {
            "saturation_score": 0.0,
            "stop_decision": False,
            "reason": "error: wall-clock timeout exceeded",
        }
    except Exception as e:
        return {
            "saturation_score": 0.0,
            "stop_decision": False,
            "reason": f"error: {str(e)[:200]}",
        }
