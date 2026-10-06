"""Structured IO contracts for ROBOVIS2027 checks and judgments (plan §5).

Validation is implemented with frozen dataclasses and explicit checks (pydantic
is not available in the egtr uv environment). Every ``validate_*`` function
returns a :class:`ValidationResult` holding either a typed value or a tuple of
human-readable violations; it never raises on bad model output.

Contracts::

    checks    = {"observed_issues": [ObservedIssue, ...],   # <= max_observed_issues
                 "hypotheses":      [Hypothesis, ...]}      # <= max_hypotheses
    judgments = {"items": [JudgmentItem, ...]}              # every check id exactly once
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from milestone1.robovis_data import load_robovis_config

T = TypeVar("T")

# Enumerations (plan §5).
HYPOTHESIS_KINDS = ("missing_object", "missing_relation")
STATUSES = ("resolved", "unsupported", "unresolved")
LIMITATIONS = ("low_resolution", "occluded", "out_of_frame", "other", "unknown")
SCORE_LEVELS = (0, 1, 2)  # allowed values of basis / resolution_benefit

OBSERVED_KEYS = ("id", "anchor_ids", "target", "region_anchor_id", "evidence")
HYPOTHESIS_KEYS = (
    "id",
    "kind",
    "candidate",
    "anchor_ids",
    "relation_ids",
    "region_anchor_id",
    "evidence",
)
JUDGMENT_KEYS = (
    "id",
    "status",
    "basis",
    "resolution_benefit",
    "limitation",
    "reason",
)
CHECKS_TOP_KEYS = ("observed_issues", "hypotheses")
JUDGMENTS_TOP_KEYS = ("items",)

# Graph fields that change between identical runs (timings) and must not
# influence cache keys.
VOLATILE_GRAPH_KEYS = ("stats",)
CACHE_KEY_VERSION = 1
CODE_FENCE = "```"


# --------------------------------------------------------------------------
# Typed objects
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ContractLimits:
    """Size limits of the check contract (from configs/robovis2027.yaml)."""

    max_observed_issues: int
    max_hypotheses: int
    max_text_chars: int

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None = None) -> ContractLimits:
        """Build limits from the robovis config (loaded if not given)."""
        cfg = cfg if cfg is not None else load_robovis_config()
        proposal = cfg["proposal"]
        return cls(
            max_observed_issues=int(proposal["max_observed_issues"]),
            max_hypotheses=int(proposal["max_hypotheses"]),
            max_text_chars=int(cfg["checks"]["max_string_chars"]),
        )


@dataclass(frozen=True)
class GraphIndex:
    """Id sets of one observed scene graph (only what validation needs)."""

    entity_ids: frozenset[str]
    relation_ids: frozenset[str]

    @classmethod
    def from_graph(cls, graph: Mapping[str, Any]) -> GraphIndex:
        """Index a graph JSON dict (worklog graph contract)."""
        entities = graph.get("entities") or []
        relations = graph.get("relations") or []
        return cls(
            entity_ids=frozenset(str(e["id"]) for e in entities),
            relation_ids=frozenset(str(r["id"]) for r in relations),
        )

    @property
    def is_empty(self) -> bool:
        """True when the observed graph has no entities."""
        return not self.entity_ids


@dataclass(frozen=True)
class ObservedIssue:
    id: str
    anchor_ids: tuple[str, ...]
    target: str
    region_anchor_id: str | None
    evidence: str


@dataclass(frozen=True)
class Hypothesis:
    id: str
    kind: str
    candidate: str
    anchor_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    region_anchor_id: str | None
    evidence: str


@dataclass(frozen=True)
class Checks:
    observed_issues: tuple[ObservedIssue, ...]
    hypotheses: tuple[Hypothesis, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        """All check ids, observed issues first."""
        return tuple(c.id for c in self.observed_issues) + tuple(
            c.id for c in self.hypotheses
        )


@dataclass(frozen=True)
class JudgmentItem:
    id: str
    status: str
    basis: int
    resolution_benefit: int
    limitation: str
    reason: str


@dataclass(frozen=True)
class Judgments:
    items: tuple[JudgmentItem, ...]

    def by_id(self) -> dict[str, JudgmentItem]:
        """Map check id -> judgment item."""
        return {item.id: item for item in self.items}


@dataclass(frozen=True)
class ValidationResult(Generic[T]):
    value: T | None
    violations: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.value is not None and not self.violations


class StructuredOutputError(ValueError):
    """Raised when a model response is not a single JSON object."""


# --------------------------------------------------------------------------
# JSON extraction
# --------------------------------------------------------------------------


def _strip_whole_code_fence(text: str) -> str:
    """Remove a markdown fence only when it wraps the *entire* response."""
    if not (text.startswith(CODE_FENCE) and text.endswith(CODE_FENCE)):
        return text
    if len(text) < 2 * len(CODE_FENCE):
        return text
    inner = text[len(CODE_FENCE) : -len(CODE_FENCE)]
    first_newline = inner.find("\n")
    if first_newline < 0:
        return inner.strip()
    # Drop an optional language tag such as ``json`` on the opening line.
    return inner[first_newline + 1 :].strip()


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a whole model response as one JSON object.

    Leading/trailing whitespace and a single markdown fence wrapping the whole
    response are tolerated. Any other surrounding text is an error (no
    "first {...} in the string" heuristics).
    """
    if not isinstance(text, str):
        raise StructuredOutputError("response is not a string")
    body = _strip_whole_code_fence(text.strip())
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"response is not valid JSON: {exc.msg}") from exc
    if not isinstance(obj, dict):
        raise StructuredOutputError("response JSON is not an object")
    return obj


# --------------------------------------------------------------------------
# Field checkers (append violations, return the coerced value or None)
# --------------------------------------------------------------------------


def _check_keys(
    obj: Any, required: Sequence[str], where: str, errors: list[str]
) -> bool:
    if not isinstance(obj, dict):
        errors.append(f"{where}: expected object, got {type(obj).__name__}")
        return False
    missing = [k for k in required if k not in obj]
    extra = sorted(k for k in obj if k not in required)
    if missing:
        errors.append(f"{where}: missing keys {missing}")
    if extra:
        errors.append(f"{where}: unknown keys {extra}")
    return not missing


def _check_text(
    value: Any, where: str, limits: ContractLimits, errors: list[str], *, nonempty: bool
) -> str | None:
    if not isinstance(value, str):
        errors.append(f"{where}: expected string")
        return None
    if nonempty and not value.strip():
        errors.append(f"{where}: must be non-empty")
        return None
    if len(value) > limits.max_text_chars:
        errors.append(f"{where}: {len(value)} chars > {limits.max_text_chars}")
        return None
    return value


def _check_enum(
    value: Any, allowed: Sequence[str], where: str, errors: list[str]
) -> str | None:
    if not isinstance(value, str) or value not in allowed:
        errors.append(f"{where}: {value!r} not in {list(allowed)}")
        return None
    return value


def _check_score(value: Any, where: str, errors: list[str]) -> int | None:
    # bool is an int subclass; reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{where}: expected integer in {list(SCORE_LEVELS)}")
        return None
    if value not in SCORE_LEVELS:
        errors.append(f"{where}: {value} not in {list(SCORE_LEVELS)}")
        return None
    return value


def _check_id_list(
    value: Any,
    known: frozenset[str],
    kind: str,
    where: str,
    limits: ContractLimits,
    errors: list[str],
) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        errors.append(f"{where}: expected list of {kind} ids")
        return None
    out: list[str] = []
    for i, item in enumerate(value):
        text = _check_text(item, f"{where}[{i}]", limits, errors, nonempty=True)
        if text is None:
            return None
        if text not in known:
            errors.append(f"{where}[{i}]: {kind} {text!r} not in graph")
            return None
        out.append(text)
    return tuple(out)


def _check_region(
    value: Any, graph: GraphIndex, where: str, limits: ContractLimits, errors: list[str]
) -> tuple[bool, str | None]:
    """Return (is_valid, region_anchor_id)."""
    if value is None:
        return True, None
    text = _check_text(value, where, limits, errors, nonempty=True)
    if text is None:
        return False, None
    if text not in graph.entity_ids:
        errors.append(f"{where}: entity {text!r} not in graph")
        return False, None
    return True, text


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def _parse_observed(
    obj: Any, where: str, graph: GraphIndex, limits: ContractLimits, errors: list[str]
) -> ObservedIssue | None:
    n_before = len(errors)
    if not _check_keys(obj, OBSERVED_KEYS, where, errors):
        return None
    cid = _check_text(obj["id"], f"{where}.id", limits, errors, nonempty=True)
    anchors = _check_id_list(
        obj["anchor_ids"],
        graph.entity_ids,
        "entity",
        f"{where}.anchor_ids",
        limits,
        errors,
    )
    target = _check_text(
        obj["target"], f"{where}.target", limits, errors, nonempty=False
    )
    _, region = _check_region(
        obj["region_anchor_id"], graph, f"{where}.region_anchor_id", limits, errors
    )
    evidence = _check_text(
        obj["evidence"], f"{where}.evidence", limits, errors, nonempty=False
    )
    if len(errors) > n_before:
        return None
    assert cid is not None and anchors is not None
    assert target is not None and evidence is not None
    return ObservedIssue(cid, anchors, target, region, evidence)


def _parse_hypothesis(
    obj: Any, where: str, graph: GraphIndex, limits: ContractLimits, errors: list[str]
) -> Hypothesis | None:
    n_before = len(errors)
    if not _check_keys(obj, HYPOTHESIS_KEYS, where, errors):
        return None
    cid = _check_text(obj["id"], f"{where}.id", limits, errors, nonempty=True)
    kind = _check_enum(obj["kind"], HYPOTHESIS_KINDS, f"{where}.kind", errors)
    candidate = _check_text(
        obj["candidate"], f"{where}.candidate", limits, errors, nonempty=False
    )
    anchors = _check_id_list(
        obj["anchor_ids"],
        graph.entity_ids,
        "entity",
        f"{where}.anchor_ids",
        limits,
        errors,
    )
    relations = _check_id_list(
        obj["relation_ids"],
        graph.relation_ids,
        "relation",
        f"{where}.relation_ids",
        limits,
        errors,
    )
    _, region = _check_region(
        obj["region_anchor_id"], graph, f"{where}.region_anchor_id", limits, errors
    )
    evidence = _check_text(
        obj["evidence"], f"{where}.evidence", limits, errors, nonempty=False
    )
    if len(errors) > n_before:
        return None
    assert cid is not None and kind is not None and candidate is not None
    assert anchors is not None and relations is not None and evidence is not None
    return Hypothesis(cid, kind, candidate, anchors, relations, region, evidence)


def validate_checks(
    obj: Any, graph: GraphIndex, limits: ContractLimits | None = None
) -> ValidationResult[Checks]:
    """Validate a checks dict against the observed graph (plan §5)."""
    limits = limits or ContractLimits.from_config()
    errors: list[str] = []
    if not _check_keys(obj, CHECKS_TOP_KEYS, "checks", errors):
        return ValidationResult(None, tuple(errors))

    sections: dict[str, list[Any]] = {}
    caps = {
        "observed_issues": limits.max_observed_issues,
        "hypotheses": limits.max_hypotheses,
    }
    for key, cap in caps.items():
        value = obj[key]
        if not isinstance(value, list):
            errors.append(f"checks.{key}: expected list")
            continue
        if len(value) > cap:
            errors.append(f"checks.{key}: {len(value)} items > max {cap}")
        sections[key] = value
    if errors:
        return ValidationResult(None, tuple(errors))

    observed = [
        _parse_observed(o, f"observed_issues[{i}]", graph, limits, errors)
        for i, o in enumerate(sections["observed_issues"])
    ]
    hypotheses = [
        _parse_hypothesis(h, f"hypotheses[{i}]", graph, limits, errors)
        for i, h in enumerate(sections["hypotheses"])
    ]
    if errors:
        return ValidationResult(None, tuple(errors))

    checks = Checks(
        observed_issues=tuple(o for o in observed if o is not None),
        hypotheses=tuple(h for h in hypotheses if h is not None),
    )
    duplicates = _duplicates(checks.ids)
    if duplicates:
        return ValidationResult(None, (f"checks: duplicate ids {duplicates}",))
    return ValidationResult(checks)


# --------------------------------------------------------------------------
# Judgments
# --------------------------------------------------------------------------


def _parse_judgment(
    obj: Any, where: str, limits: ContractLimits, errors: list[str]
) -> JudgmentItem | None:
    n_before = len(errors)
    if not _check_keys(obj, JUDGMENT_KEYS, where, errors):
        return None
    cid = _check_text(obj["id"], f"{where}.id", limits, errors, nonempty=True)
    status = _check_enum(obj["status"], STATUSES, f"{where}.status", errors)
    basis = _check_score(obj["basis"], f"{where}.basis", errors)
    benefit = _check_score(
        obj["resolution_benefit"], f"{where}.resolution_benefit", errors
    )
    limitation = _check_enum(
        obj["limitation"], LIMITATIONS, f"{where}.limitation", errors
    )
    reason = _check_text(
        obj["reason"], f"{where}.reason", limits, errors, nonempty=False
    )
    if len(errors) > n_before:
        return None
    assert cid is not None and status is not None and limitation is not None
    assert basis is not None and benefit is not None and reason is not None
    return JudgmentItem(cid, status, basis, benefit, limitation, reason)


def validate_judgments(
    obj: Any, checks: Checks, limits: ContractLimits | None = None
) -> ValidationResult[Judgments]:
    """Validate judgments; every check id must appear exactly once."""
    limits = limits or ContractLimits.from_config()
    errors: list[str] = []
    if not _check_keys(obj, JUDGMENTS_TOP_KEYS, "judgments", errors):
        return ValidationResult(None, tuple(errors))
    raw_items = obj["items"]
    if not isinstance(raw_items, list):
        return ValidationResult(None, ("judgments.items: expected list",))

    parsed = [
        _parse_judgment(item, f"items[{i}]", limits, errors)
        for i, item in enumerate(raw_items)
    ]
    if errors:
        return ValidationResult(None, tuple(errors))
    items = tuple(p for p in parsed if p is not None)

    ids = [item.id for item in items]
    expected = set(checks.ids)
    duplicates = _duplicates(ids)
    if duplicates:
        errors.append(f"judgments: duplicate ids {duplicates}")
    missing = sorted(expected - set(ids))
    if missing:
        errors.append(f"judgments: missing check ids {missing}")
    unknown = sorted(set(ids) - expected)
    if unknown:
        errors.append(f"judgments: unknown check ids {unknown}")
    if errors:
        return ValidationResult(None, tuple(errors))
    return ValidationResult(Judgments(items))


def _duplicates(ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    dups: set[str] = set()
    for cid in ids:
        if cid in seen:
            dups.add(cid)
        seen.add(cid)
    return sorted(dups)


# --------------------------------------------------------------------------
# Hashing / cache keys
# --------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialisation (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def graph_sha256(graph: Mapping[str, Any]) -> str:
    """Hash of a graph dict, ignoring volatile fields such as timings."""
    stable = {k: v for k, v in graph.items() if k not in VOLATILE_GRAPH_KEYS}
    return sha256_text(canonical_json(stable))


def cache_key(
    *,
    image_sha256: str | None,
    model_id: str,
    prompt_sha256: str,
    generation: Mapping[str, Any],
    graph_sha256: str | None,
) -> str:
    """Canonical cache key for one model call.

    ``image_sha256`` is the hash of the image *bytes* (None for text-only
    calls); ``graph_sha256`` is the hash of the input graph (None if no graph).
    """
    payload = {
        "version": CACHE_KEY_VERSION,
        "image_sha256": image_sha256,
        "model_id": model_id,
        "prompt_sha256": prompt_sha256,
        "generation": dict(generation),
        "graph_sha256": graph_sha256,
    }
    return sha256_text(canonical_json(payload))
