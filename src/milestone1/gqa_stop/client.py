"""Ollama /api/chat client with on-disk cache, retries and one schema repair.

Plan §5 / §9: every request is a fresh conversation; JSON is parsed with
``json.loads`` on the whole content (no regex extraction); transport failures
are retried with the same payload; only format failures get one repair
attempt with a short schema-fix instruction (never GT).
"""

from __future__ import annotations

import base64
import json
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import requests

from .common import (
    append_jsonl,
    atomic_write_json,
    canonical_json,
    read_json,
    sha256_bytes,
    sha256_json,
    sha256_text,
    utc_now,
)

ROLES = ("answer_only", "direct", "planner", "checklist")
V2_ROLES = ("planner_v2", "identify", "attr_blind", "recompose")  # plan §17
REPAIR_PROMPT = "schema_repair"
ROLE_DIRS = {
    "answer_only": "reader",
    "direct": "direct",
    "planner": "plans",
    "checklist": "checks",
    "planner_v2": "plans_v2",
    "identify": "identify",
    "attr_blind": "attr_blind",
    "recompose": "recompose",
}
STATUS_VALID = "valid"
STATUS_SCHEMA_FAILURE = "schema_failure"
STATUS_TRANSPORT_FAILURE = "transport_failure"
FINAL_STATUSES = frozenset({STATUS_VALID, STATUS_SCHEMA_FAILURE})
HTTP_OK = 200
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR_MIN = 500
DEFAULT_BACKOFF_BASE_S = 2.0
DEFAULT_BACKOFF_MAX_S = 30.0
MAX_REPAIR_ERRORS_SHOWN = 5
INPUT_DATA_SEPARATOR = "\n\nInput data: "

Validator = Callable[[Any], tuple[Any, list[str]]]


class TransportError(RuntimeError):
    """HTTP-level failure after all retries (or a non-retryable status)."""


class PauseRequired(RuntimeError):
    """Too many technical failures: stop inference and repair (plan §9.2)."""


@dataclass(frozen=True)
class ModelSettings:
    endpoint: str
    name: str
    digest: str | None
    temperature: float
    seed: int
    think: bool
    json_format: bool
    num_ctx: int
    num_predict: dict[str, int]
    timeout_s: float
    transport_retries: int
    schema_repair_attempts: int
    backoff_base_s: float = DEFAULT_BACKOFF_BASE_S
    backoff_max_s: float = DEFAULT_BACKOFF_MAX_S

    def options(self, role: str) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "num_predict": int(self.num_predict[role]),
        }

    def decoding(self, role: str) -> dict[str, Any]:
        """Everything sent besides messages; part of the cache key."""
        return {
            "think": self.think,
            "format": "json" if self.json_format else None,
            "stream": False,
            "options": self.options(role),
        }


def settings_from_config(cfg: Mapping[str, Any]) -> ModelSettings:
    model = cfg["model"]
    return ModelSettings(
        endpoint=str(model["endpoint"]),
        name=str(model["name"]),
        digest=model.get("digest"),
        temperature=float(model["temperature"]),
        seed=int(model["seed"]),
        think=bool(model["think"]),
        json_format=model.get("format") == "json",
        num_ctx=int(model["num_ctx"]),
        num_predict={k: int(v) for k, v in model["num_predict"].items()},
        timeout_s=float(model["request_timeout_seconds"]),
        transport_retries=int(model["transport_retries"]),
        schema_repair_attempts=int(model["schema_repair_attempts"]),
        backoff_base_s=float(model.get("backoff_base_seconds", DEFAULT_BACKOFF_BASE_S)),
        backoff_max_s=float(model.get("backoff_max_seconds", DEFAULT_BACKOFF_MAX_S)),
    )


@dataclass(frozen=True)
class PromptSet:
    texts: dict[str, str]

    @property
    def hashes(self) -> dict[str, str]:
        return {k: sha256_text(v) for k, v in self.texts.items()}


def load_prompts(prompts_dir: Path, roles: Sequence[str] = ROLES) -> PromptSet:
    """Prompt texts for ``roles`` plus the schema-repair prompt."""
    names = (*roles, REPAIR_PROMPT)
    return PromptSet({n: (prompts_dir / f"{n}.txt").read_text("utf-8") for n in names})


def render_prompt(template: str, input_data: Mapping[str, Any]) -> str:
    """Fixed instruction + JSON-escaped input data (question text is data)."""
    return template.rstrip("\n") + INPUT_DATA_SEPARATOR + json.dumps(input_data)


@dataclass(frozen=True)
class RequestSpec:
    """One model call. ``input_data`` must only hold question text / checklist."""

    role: str
    question_id: str
    level: int | None
    input_data: dict[str, Any]
    image_bytes: bytes | None = None
    namespace: str = ""

    @property
    def image_sha256(self) -> str | None:
        return sha256_bytes(self.image_bytes) if self.image_bytes is not None else None


@dataclass(frozen=True)
class TransportResult:
    content: str
    http_attempts: int
    elapsed_s: float
    usage: dict[str, Any] = field(default_factory=dict)


class Transport(Protocol):
    def send(self, payload: dict[str, Any]) -> TransportResult: ...


USAGE_KEYS = ("prompt_eval_count", "eval_count", "total_duration", "eval_duration")


class OllamaTransport:
    """POST /api/chat with bounded retries and exponential backoff."""

    def __init__(
        self,
        settings: ModelSettings,
        post: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._post = post if post is not None else requests.post
        self._sleep = sleep

    def send(self, payload: dict[str, Any]) -> TransportResult:
        attempts = 1 + self.settings.transport_retries
        last_error = ""
        started = time.monotonic()
        for attempt in range(attempts):
            try:
                response = self._post(
                    self.settings.endpoint,
                    json=payload,
                    timeout=self.settings.timeout_s,
                )
                status = int(response.status_code)
                body = response.json() if status == HTTP_OK else None
            except (requests.RequestException, ValueError) as exc:
                # connection/timeout/chunked/content-decoding errors and
                # non-JSON 200 bodies are all retryable transport failures
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if status == HTTP_OK and isinstance(body, dict):
                    return TransportResult(
                        content=str(body.get("message", {}).get("content", "")),
                        http_attempts=attempt + 1,
                        elapsed_s=time.monotonic() - started,
                        usage={k: body[k] for k in USAGE_KEYS if k in body},
                    )
                retryable = (
                    status == HTTP_OK
                    or status == HTTP_TOO_MANY_REQUESTS
                    or status >= HTTP_SERVER_ERROR_MIN
                )
                last_error = f"HTTP {status}" + (
                    " non-object body" if status == HTTP_OK else ""
                )
                if not retryable:
                    raise TransportError(last_error)
            if attempt + 1 < attempts:
                delay = self.settings.backoff_base_s * (2**attempt)
                self._sleep(min(delay, self.settings.backoff_max_s))
        raise TransportError(f"failed after {attempts} attempts: {last_error}")


class FailureMonitor:
    """Pause rule: 10 consecutive technical failures or >5% of the last 100."""

    def __init__(self, max_consecutive: int, window: int, max_fraction: float) -> None:
        self.max_consecutive = max_consecutive
        self.window: deque[bool] = deque(maxlen=window)
        self.max_fraction = max_fraction
        self.consecutive = 0

    def observe(self, technical_failure: bool) -> None:
        self.window.append(technical_failure)
        self.consecutive = self.consecutive + 1 if technical_failure else 0
        if self.consecutive >= self.max_consecutive:
            raise PauseRequired(f"{self.consecutive} consecutive technical failures")
        full = len(self.window) == self.window.maxlen
        if full and sum(self.window) / len(self.window) > self.max_fraction:
            raise PauseRequired("technical failure rate over the recent window")


def monitor_from_config(cfg: Mapping[str, Any]) -> FailureMonitor:
    execution = cfg["execution"]
    return FailureMonitor(
        int(execution["pause_on_consecutive_technical_errors"]),
        int(execution["pause_error_window"]),
        float(execution["pause_error_fraction"]),
    )


class RequestEngine:
    """Builds payloads, consults the cache, calls the transport, writes records."""

    def __init__(
        self,
        settings: ModelSettings,
        prompts: PromptSet,
        transport: Transport,
        output_root: Path,
        monitor: FailureMonitor | None = None,
        reuse_root: Path | None = None,
        reuse_roles: frozenset[str] = frozenset(),
    ) -> None:
        self.reuse_root = (
            reuse_root  # another run whose identical records may be copied
        )
        self.reuse_roles = reuse_roles
        self.copied = 0
        self.settings = settings
        self.prompts = prompts
        self.transport = transport
        self.output_root = output_root
        self.monitor = monitor
        self.calls = 0
        self.cache_hits = 0
        self.stage = ""
        self.reused_from: Counter[str] = Counter()

    @property
    def model_tag(self) -> str:
        """name@short-digest; separates records of different models on disk."""
        return model_tag(self.settings.name, self.settings.digest)

    # ---- identity -------------------------------------------------------
    def cache_key(self, spec: RequestSpec) -> str:
        return sha256_json(
            {
                "role": spec.role,
                "namespace": spec.namespace,
                "model": self.settings.name,
                "digest": self.settings.digest,
                "model_tag": self.model_tag,
                "parse_version": PARSE_VERSION,
                **self._schema_key(spec.role),
                "prompt_hash": self.prompts.hashes[spec.role],
                "repair_prompt_hash": self._repair_hash(),
                "input_hash": sha256_json(spec.input_data),
                "image_sha256": spec.image_sha256,
                "decoding": self.settings.decoding(spec.role),
            }
        )

    def record_path(self, spec: RequestSpec) -> Path:
        role_dir = ROLE_DIRS[spec.role] + (
            f"_{spec.namespace}" if spec.namespace else ""
        )
        base = (
            self.output_root
            / role_dir
            / record_tag(self.settings.name, self.settings.digest)
        )
        if spec.level is None:
            return base / f"{spec.question_id}.json"
        return base / spec.question_id / f"{spec.level:02d}.json"

    def build_messages(self, spec: RequestSpec) -> list[dict[str, Any]]:
        message: dict[str, Any] = {
            "role": "user",
            "content": render_prompt(self.prompts.texts[spec.role], spec.input_data),
        }
        if spec.image_bytes is not None:
            message["images"] = [base64.b64encode(spec.image_bytes).decode("ascii")]
        return [message]

    def build_payload(
        self, spec: RequestSpec, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.name,
            "messages": messages,
            **self.settings.decoding(spec.role),
        }
        if payload["format"] is None:
            payload.pop("format")
        return payload

    # ---- cache ----------------------------------------------------------
    def lookup(self, spec: RequestSpec) -> dict[str, Any] | None:
        path = self.record_path(spec)
        record = self._final_record_at(path, spec)
        if record is not None:
            return record
        return self._copy_from_reuse_root(spec, path)

    def _final_record_at(self, path: Path, spec: RequestSpec) -> dict[str, Any] | None:
        if not path.exists():
            return None
        record = read_json(path)
        if record.get("cache_key") != self.cache_key(spec):
            return None
        if record.get("status") not in FINAL_STATUSES:
            return None
        return record

    def _copy_from_reuse_root(
        self, spec: RequestSpec, path: Path
    ) -> dict[str, Any] | None:
        """Copy an identical-key final record from the reuse run (provenance kept)."""
        if self.reuse_root is None or spec.role not in self.reuse_roles:
            return None
        source = self.reuse_root / path.relative_to(self.output_root)
        record = self._final_record_at(source, spec)
        if record is None:
            return None
        copied = {**record, "copied_from": str(source)}
        atomic_write_json(path, copied)
        self.copied += 1
        return copied

    # ---- execution ------------------------------------------------------
    def run(self, spec: RequestSpec, validator: Validator) -> dict[str, Any]:
        cached = self.lookup(spec)
        if cached is not None:
            self.cache_hits += 1
            origin = str(cached.get("stage", ""))
            if cached.get("copied_from"):
                origin = f"reuse_root:{origin}"
            if origin and origin != self.stage:
                self.reused_from[origin] += 1
            return cached
        record = self._execute(spec, validator)
        atomic_write_json(self.record_path(spec), record)
        if record["status"] != STATUS_VALID:
            append_jsonl(self.output_root / "errors.jsonl", _error_row(record))
        if self.monitor is not None:
            self.monitor.observe(record["status"] != STATUS_VALID)
        return record

    def _execute(self, spec: RequestSpec, validator: Validator) -> dict[str, Any]:
        messages = self.build_messages(spec)
        attempts: list[dict[str, Any]] = []
        status = STATUS_TRANSPORT_FAILURE
        parsed: Any = None
        raw: str | None = None
        max_attempts = 1 + self.settings.schema_repair_attempts
        for attempt in range(1, max_attempts + 1):
            payload = self.build_payload(spec, messages)
            row = self._attempt_row(spec, payload, attempt)
            try:
                result = self.transport.send(payload)
            except TransportError as exc:
                row = {**row, "status": STATUS_TRANSPORT_FAILURE, "error": str(exc)}
                attempts.append(row)
                self._log_request(row)
                status = STATUS_TRANSPORT_FAILURE
                break
            self.calls += 1
            raw = result.content
            result_parse = parse_response(raw, validator)
            parsed, errors = result_parse.parsed, result_parse.errors
            fence_stripped = result_parse.fence_stripped
            status = STATUS_SCHEMA_FAILURE if errors else STATUS_VALID
            row = {
                **row,
                "status": status,
                "errors": errors,
                "raw": raw,
                "fence_stripped": fence_stripped,
                "http_attempts": result.http_attempts,
                "elapsed_s": round(result.elapsed_s, 3),
                "usage": result.usage,
            }
            attempts.append(row)
            self._log_request(row)
            if not errors:
                break
            messages = self._repair_messages(messages, raw, errors)
        return self._final_record(spec, attempts, status, raw, parsed)

    @staticmethod
    def _schema_key(role: str) -> dict[str, str]:
        """Only roles with a non-default validator version change their key."""
        version = SCHEMA_VERSIONS.get(role)
        return {} if version is None else {"schema_version": version}

    def _repair_hash(self) -> str | None:
        if self.settings.schema_repair_attempts <= 0:
            return None
        return self.prompts.hashes[REPAIR_PROMPT]

    def _repair_messages(
        self, messages: list[dict[str, Any]], raw: str, errors: list[str]
    ) -> list[dict[str, Any]]:
        """Original request + model reply + short schema-fix instruction (no GT)."""
        problems = "; ".join(errors[:MAX_REPAIR_ERRORS_SHOWN])
        repair = self.prompts.texts[REPAIR_PROMPT].rstrip("\n") + " " + problems
        return [
            messages[0],
            {"role": "assistant", "content": raw},
            {"role": "user", "content": repair},
        ]

    def _attempt_row(
        self, spec: RequestSpec, payload: dict[str, Any], attempt: int
    ) -> dict[str, Any]:
        return {
            "timestamp": utc_now(),
            "role": spec.role,
            "namespace": spec.namespace,
            "question_id": spec.question_id,
            "level": spec.level,
            "model": self.settings.name,
            "digest": self.settings.digest,
            "attempt": attempt,
            "kind": "initial" if attempt == 1 else "schema_repair",
            "request_hash": request_hash(payload),
            "prompt_hash": self.prompts.hashes[spec.role],
            "repair_prompt_hash": (
                self.prompts.hashes[REPAIR_PROMPT] if attempt > 1 else None
            ),
            "image_sha256": spec.image_sha256,
            "cache_key": self.cache_key(spec),
        }

    def _log_request(self, row: dict[str, Any]) -> None:
        append_jsonl(self.output_root / "requests.jsonl", row)

    def _final_record(
        self,
        spec: RequestSpec,
        attempts: list[dict[str, Any]],
        status: str,
        raw: str | None,
        parsed: Any,
    ) -> dict[str, Any]:
        return {
            "role": spec.role,
            "namespace": spec.namespace,
            "question_id": spec.question_id,
            "level": spec.level,
            "model": self.settings.name,
            "digest": self.settings.digest,
            "cache_key": self.cache_key(spec),
            "request_hash": attempts[-1]["request_hash"] if attempts else None,
            "prompt_hash": self.prompts.hashes[spec.role],
            "input_hash": sha256_json(spec.input_data),
            "input_data": spec.input_data,
            "image_sha256": spec.image_sha256,
            "decoding": self.settings.decoding(spec.role),
            "model_tag": self.model_tag,
            "parse_version": PARSE_VERSION,
            "fence_stripped": bool(attempts and attempts[-1].get("fence_stripped")),
            "initial_valid": bool(
                attempts and attempts[0].get("status") == STATUS_VALID
            ),
            "repair_prompt_hash": self._repair_hash(),
            "stage": self.stage,
            "status": status,
            "validation_status": status,
            "attempt": len(attempts),
            "attempts": attempts,
            "raw": raw,
            "parsed": parsed if status == STATUS_VALID else None,
            "timestamp": utc_now(),
        }


SHORT_DIGEST_CHARS = 12


def model_tag(name: str, digest: str | None) -> str:
    safe = name.replace(":", "-").replace("/", "_")
    return f"{safe}@{(digest or 'nodigest')[:SHORT_DIGEST_CHARS]}"


# Response parser version. Part of the cache key and of the record directory,
# so records parsed under an older version are never reused.
#   v1: json.loads on the whole stripped content.
#   v2 (2026-10-09): if the whole content is exactly one fenced block
#       (```json ... ``` or ``` ... ```), the fence markers are removed and the
#       ENTIRE inner text goes to json.loads. Anything else (text around the
#       fence, two fences, another language tag) is parsed as in v1 and fails.
#       Reason: qwen3.5:9b fenced 737/759 first-attempt checklist replies,
#       pushing nearly every checklist record through the repair path.
PARSE_VERSION = "parse-v2"
# Validator version per role, part of the cache key so that a validator change
# never reuses records judged under the old rule (default "1").
#   planner_v2 "2" (2026-10-09): a referring phrase that names the target noun
#   (e.g. "the object to the right of the other cup") is accepted and flagged.
SCHEMA_VERSIONS = {"planner_v2": "2"}
DEFAULT_SCHEMA_VERSION = "1"
FENCE = "```"
FENCE_LANGUAGE = "json"
FENCES_IN_ONE_BLOCK = 2


@dataclass(frozen=True)
class ParseResult:
    parsed: Any
    errors: list[str]
    fence_stripped: bool


def strip_single_fence(raw: str) -> tuple[str, bool]:
    """Unwrap content that is exactly one fenced block; otherwise unchanged."""
    text = raw.strip()
    single_block = (
        text.startswith(FENCE)
        and text.endswith(FENCE)
        and len(text) >= FENCES_IN_ONE_BLOCK * len(FENCE)
        and text.count(FENCE) == FENCES_IN_ONE_BLOCK
    )
    if not single_block:
        return raw, False
    inner = text[len(FENCE) : -len(FENCE)]
    head, newline, rest = inner.partition("\n")
    head = head.strip()
    if head.lower().startswith(FENCE_LANGUAGE):
        head = head[len(FENCE_LANGUAGE) :]
    elif head and head[0] not in "{[":
        return raw, False  # some other language tag: not our JSON block
    return head + newline + rest, True


def parse_response(raw: str, validator: Validator) -> ParseResult:
    """``json.loads`` on the whole (single-fence-unwrapped) content; no regex."""
    text, fence_stripped = strip_single_fence(raw)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return ParseResult(None, [f"invalid JSON: {exc.msg}"], fence_stripped)
    parsed, errors = validator(obj)
    return ParseResult(parsed, errors, fence_stripped)


def parse_and_validate(raw: str, validator: Validator) -> tuple[Any, list[str]]:
    result = parse_response(raw, validator)
    return result.parsed, result.errors


def record_tag(name: str, digest: str | None) -> str:
    """Record sub-directory: model tag plus parser version."""
    return f"{model_tag(name, digest)}/{PARSE_VERSION}"


def request_hash(payload: Mapping[str, Any]) -> str:
    """Hash of the payload with image base64 replaced by its sha256."""
    messages = []
    for message in payload.get("messages", []):
        images = message.get("images")
        if images:
            message = {
                **message,
                "images": [sha256_bytes(base64.b64decode(i)) for i in images],
            }
        messages.append(message)
    return sha256_text(canonical_json({**payload, "messages": messages}))


def _error_row(record: Mapping[str, Any]) -> dict[str, Any]:
    last = record["attempts"][-1] if record["attempts"] else {}
    return {
        "timestamp": record["timestamp"],
        "role": record["role"],
        "namespace": record["namespace"],
        "question_id": record["question_id"],
        "level": record["level"],
        "status": record["status"],
        "attempt": record["attempt"],
        "error": last.get("error") or last.get("errors"),
        "cache_key": record["cache_key"],
    }
