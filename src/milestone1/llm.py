"""Ollama LLM client for semantic saturation scoring."""

import json
import re
from typing import Any

import requests

SATURATION_PROMPT = """You are a scene graph analysis expert. You are given a sequence of scene graphs extracted from the same image at increasing resolution levels.

Your task: Judge whether the scene graph has reached "semantic saturation" — i.e., whether further increasing image resolution is unlikely to reveal meaningful new objects or relationships.

INPUT: A JSON array of scene graphs at consecutive resolution levels. Each entry has:
- "level": resolution level (higher = better resolution)
- "objects": detected objects with labels and confidence scores
- "relationships": detected relationships (subject -> predicate -> object) with confidence scores

CRITERIA for saturation:
1. The set of detected object categories has stabilized (few or no new categories appearing)
2. The set of detected relationships has stabilized
3. Confidence scores are not significantly changing
4. The scene graph structure is converging

OUTPUT: Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{
  "saturation_score": <float 0.0-1.0, where 1.0 = fully saturated>,
  "stop_decision": <boolean, true if saturation_score >= 0.8>,
  "reason": "<brief explanation of your judgment>"
}

Scene graph sequence:
"""


def _parse_llm_json(text: str) -> dict[str, Any]:
    """Parse JSON from LLM response, handling markdown fences and noise."""
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    # Try to extract first JSON object
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try parsing the whole text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {
            "saturation_score": 0.0,
            "stop_decision": False,
            "reason": f"parse-error: {text[:200]}",
        }


def _derive_chat_endpoint(generate_endpoint: str) -> str:
    """Convert generate endpoint to chat endpoint."""
    return generate_endpoint.replace("/api/generate", "/api/chat")


def call_ollama(
    prompt: str,
    model: str,
    endpoint: str,
    timeout: int = 120,
) -> str:
    """Call Ollama chat API with thinking disabled and return the response text."""
    chat_endpoint = _derive_chat_endpoint(endpoint)
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1024,
            # Avoid ollama's default num_ctx=4096 truncating long prompts to
            # 4095 tokens (leaves no room to generate -> 1-token "Based" stub
            # that fails JSON parsing). Fits our largest ~9.2k-token prompts.
            "num_ctx": 12288,
        },
    }
    resp = requests.post(chat_endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def smoke_test_ollama(model: str, endpoint: str, timeout: int = 30) -> bool:
    """Quick check if Ollama is reachable and model is loaded."""
    try:
        result = call_ollama("Respond with just: OK", model, endpoint, timeout=timeout)
        return len(result.strip()) > 0
    except Exception as e:
        print(f"Ollama smoke test failed: {e}")
        return False


def score_saturation(
    sequence_json: list[dict[str, Any]],
    model: str,
    endpoint: str,
    timeout: int = 120,
) -> dict[str, Any]:
    """Score semantic saturation for a scene graph sequence.

    Returns {saturation_score, stop_decision, reason}.
    """
    prompt = SATURATION_PROMPT + json.dumps(sequence_json, indent=2)

    try:
        raw_response = call_ollama(prompt, model, endpoint, timeout=timeout)
        result = _parse_llm_json(raw_response)

        # Ensure required fields
        if "saturation_score" not in result:
            result["saturation_score"] = 0.0
        if "stop_decision" not in result:
            result["stop_decision"] = result["saturation_score"] >= 0.8
        if "reason" not in result:
            result["reason"] = "no reason provided"

        # Clamp score to [0, 1]
        result["saturation_score"] = max(
            0.0, min(1.0, float(result["saturation_score"]))
        )
        result["stop_decision"] = bool(result["stop_decision"])

        return result

    except Exception as e:
        return {
            "saturation_score": 0.0,
            "stop_decision": False,
            "reason": f"error: {str(e)[:200]}",
        }
