"""Minimal ollama /api/chat client with retries and a structured-output loop.

Used by the ROBOVIS2027 check generator (F1/F2) and judgment step.

Failure policy (plan §11):
- transient HTTP failure (connection error, timeout, 5xx, 429) -> up to
  ``transient_retries`` retries with exponential backoff;
- schema failure (unparseable JSON or contract violation) -> re-ask the SAME
  messages up to ``schema_retries`` more times with ``seed + i`` (temperature
  unchanged; with temperature 0 an identical seed returns identical bytes);
  the first valid answer wins;
- anything else is returned as an explicit failure, never as an empty result.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import requests

HTTP_TOO_MANY_REQUESTS = 429
SEED_RETRY_INCREMENT = 1  # schema retry i uses seed + i (plan decision 2026-10-07)
# Schema retries also switch to ``retry_temperature`` when configured (team-lead
# decision 2026-10-07): temperature 0 is greedy, so seed alone changes nothing.
HTTP_SERVER_ERROR_MIN = 500
USAGE_KEYS = (
    "prompt_eval_count",
    "eval_count",
    "total_duration",
    "load_duration",
    "prompt_eval_duration",
    "eval_duration",
)
_FENCE_RE = re.compile(r"```(?:json)?")

# A validator maps parsed JSON to (normalized_object, errors). Empty errors = valid.
Validator = Callable[[Any], tuple[Any, list[str]]]


class OllamaError(RuntimeError):
    """Raised when a chat call fails permanently (after transient retries)."""


@dataclass(frozen=True)
class OllamaSettings:
    """Connection and generation settings; ``options`` is sent verbatim."""

    endpoint: str
    think: bool
    json_mode: bool
    options: dict[str, Any]
    timeout_s: float
    transient_retries: int
    backoff_base_s: float
    backoff_max_s: float
    retry_temperature: float | None = None  # schema retries only; None = unchanged

    def record(self) -> dict[str, Any]:
        """Settings that affect model output (stored next to every result)."""
        return {
            "think": self.think,
            "format": "json" if self.json_mode else None,
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class ChatResult:
    """One successful HTTP round trip."""

    content: str
    elapsed_s: float
    http_attempts: int
    usage: dict[str, Any]


class ChatClient(Protocol):
    """Anything that can answer a chat request (real client or test fake)."""

    settings: OllamaSettings

    def chat(
        self,
        model: str,
        messages: Sequence[dict[str, Any]],
        options_override: dict[str, Any] | None = None,
    ) -> ChatResult: ...


PostFn = Callable[..., Any]


class OllamaClient:
    """Thin wrapper over ``POST /api/chat`` with ``stream: false``."""

    def __init__(
        self,
        settings: OllamaSettings,
        post: PostFn | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._post = post if post is not None else requests.post
        self._sleep = sleep

    def build_payload(
        self,
        model: str,
        messages: Sequence[dict[str, Any]],
        options_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "stream": False,
            "think": self.settings.think,
            "options": {**self.settings.options, **(options_override or {})},
        }
        if self.settings.json_mode:
            payload["format"] = "json"
        return payload

    def _backoff(self, attempt: int) -> float:
        delay = self.settings.backoff_base_s * (2**attempt)
        return min(delay, self.settings.backoff_max_s)

    def chat(
        self,
        model: str,
        messages: Sequence[dict[str, Any]],
        options_override: dict[str, Any] | None = None,
    ) -> ChatResult:
        payload = self.build_payload(model, messages, options_override)
        max_attempts = 1 + self.settings.transient_retries
        last_error = ""
        t0 = time.monotonic()
        for attempt in range(max_attempts):
            try:
                resp = self._post(
                    self.settings.endpoint,
                    json=payload,
                    timeout=self.settings.timeout_s,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                status = resp.status_code
                transient = (
                    status == HTTP_TOO_MANY_REQUESTS or status >= HTTP_SERVER_ERROR_MIN
                )
                if transient:
                    last_error = f"HTTP {status}: {resp.text[:200]}"
                elif status != requests.codes.ok:
                    raise OllamaError(f"HTTP {status}: {resp.text[:200]}")
                else:
                    body = resp.json()
                    return ChatResult(
                        content=body.get("message", {}).get("content", ""),
                        elapsed_s=time.monotonic() - t0,
                        http_attempts=attempt + 1,
                        usage={k: body[k] for k in USAGE_KEYS if k in body},
                    )
            if attempt + 1 < max_attempts:
                self._sleep(self._backoff(attempt))
        raise OllamaError(
            f"transient failure after {max_attempts} attempts: {last_error}"
        )


def parse_json_object(text: str) -> Any:
    """Parse a JSON object from model text (tolerates fences / leading noise)."""
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in response")
    return json.loads(cleaned[start : end + 1])


@dataclass
class StructuredOutcome:
    """Result of a structured call: the first valid answer or an explicit failure."""

    parsed: Any | None
    attempts: int
    failure: bool
    error: str | None
    elapsed_s: float
    http_attempts: int
    raw_responses: list[str] = field(default_factory=list)
    attempt_errors: list[list[str]] = field(default_factory=list)
    usage: list[dict[str, Any]] = field(default_factory=list)
    seeds: list[int | None] = field(default_factory=list)
    seed_used: int | None = None
    temperatures: list[float | None] = field(default_factory=list)
    temperature_used: float | None = None


def structured_chat(
    client: ChatClient,
    model: str,
    messages: Sequence[dict[str, Any]],
    validator: Validator,
    schema_retries: int,
) -> StructuredOutcome:
    """Ask once; on schema failure re-ask the same messages with seed + i.

    The first valid answer is kept (never best-of-several). ``seeds`` records
    the seed of every attempt, ``seed_used`` the seed of the accepted answer.
    """
    outcome = StructuredOutcome(
        parsed=None,
        attempts=0,
        failure=True,
        error=None,
        elapsed_s=0.0,
        http_attempts=0,
    )
    base_seed = client.settings.options.get("seed")
    base_temperature = client.settings.options.get("temperature")
    retry_temperature = client.settings.retry_temperature
    for attempt in range(1 + schema_retries):
        outcome.attempts += 1
        seed = None
        temperature = base_temperature
        override: dict[str, Any] = {}
        if base_seed is not None:
            seed = int(base_seed) + attempt * SEED_RETRY_INCREMENT
            override["seed"] = seed
        if attempt > 0 and retry_temperature is not None:
            temperature = retry_temperature
            override["temperature"] = temperature
        outcome.seeds.append(seed)
        outcome.temperatures.append(temperature)
        try:
            result = client.chat(model, messages, override or None)
        except OllamaError as exc:
            outcome.error = f"http: {exc}"
            return outcome
        outcome.elapsed_s += result.elapsed_s
        outcome.http_attempts += result.http_attempts
        outcome.raw_responses.append(result.content)
        outcome.usage.append(result.usage)
        try:
            data = parse_json_object(result.content)
        except (ValueError, json.JSONDecodeError) as exc:
            errors = [f"json: {exc}"]
        else:
            normalized, errors = validator(data)
            if not errors:
                outcome.attempt_errors.append([])
                outcome.parsed = normalized
                outcome.seed_used = seed
                outcome.temperature_used = temperature
                outcome.failure = False
                outcome.error = None
                return outcome
        outcome.attempt_errors.append(errors)
        outcome.error = "schema: " + "; ".join(errors[:5])
    return outcome


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False))


def encode_image_file(path: str | Path) -> tuple[str, str]:
    """Return (base64 of the raw file bytes, sha256 of the bytes). No re-encoding."""
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("ascii"), hashlib.sha256(data).hexdigest()


def settings_from_config(cfg: dict[str, Any]) -> OllamaSettings:
    """Build settings from configs/robovis2027.yaml (`llm`, `checks`, `execution`)."""
    checks = cfg["checks"]
    return OllamaSettings(
        endpoint=cfg["llm"]["endpoint"],
        think=bool(cfg["llm"]["think"]),
        json_mode=bool(checks["json_mode"]),
        options=dict(checks["options"]),
        timeout_s=float(checks["request_timeout_s"]),
        transient_retries=int(cfg["execution"]["transient_retries"]),
        backoff_base_s=float(checks["backoff_base_s"]),
        backoff_max_s=float(checks["backoff_max_s"]),
        retry_temperature=(
            float(checks["retry_temperature"])
            if checks.get("retry_temperature") is not None
            else None
        ),
    )
