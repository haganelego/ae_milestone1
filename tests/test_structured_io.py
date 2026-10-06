"""Tests for the ROBOVIS2027 check/judgment contracts (plan §5).

Run: uv run --project egtr --with pytest python -m pytest tests/test_structured_io.py -q
"""

import copy
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.structured_io import (  # noqa: E402
    ContractLimits,
    GraphIndex,
    StructuredOutputError,
    cache_key,
    graph_sha256,
    parse_json_response,
    validate_checks,
    validate_judgments,
)

LIMITS = ContractLimits.from_config()
GRAPH = GraphIndex.from_graph(
    {
        "entities": [{"id": "v1"}, {"id": "v2"}, {"id": "v3"}],
        "relations": [{"id": "e1", "sub": "v1", "obj": "v2"}],
    }
)


def _checks() -> dict[str, Any]:
    return {
        "observed_issues": [
            {"id": "o1", "anchor_ids": ["v1"], "target": "label of v1",
             "region_anchor_id": "v1", "evidence": "tiny box"},
        ],
        "hypotheses": [
            {"id": "h1", "kind": "missing_relation", "candidate": "man holding cup",
             "anchor_ids": ["v1", "v3"], "relation_ids": ["e1"],
             "region_anchor_id": "v3", "evidence": "hand near table"},
        ],
    }  # fmt: skip


def _judgments() -> dict[str, Any]:
    item = {"status": "unresolved", "basis": 2, "resolution_benefit": 1,
            "limitation": "low_resolution", "reason": "blurry"}  # fmt: skip
    return {"items": [{"id": "o1", **item}, {"id": "h1", **item}]}


def _valid_checks():
    res = validate_checks(_checks(), GRAPH, LIMITS)
    assert res.ok and res.value is not None
    return res.value


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def test_valid_checks_pass() -> None:
    checks = _valid_checks()
    assert checks.ids == ("o1", "h1")
    assert checks.hypotheses[0].relation_ids == ("e1",)


def test_null_region_anchor_is_valid() -> None:
    obj = _checks()
    obj["observed_issues"][0]["region_anchor_id"] = None
    assert validate_checks(obj, GRAPH, LIMITS).ok


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("observed_issues", "anchor_ids", ["v99"]),
        ("observed_issues", "region_anchor_id", "v99"),
        ("hypotheses", "anchor_ids", ["v1", "v99"]),
        ("hypotheses", "relation_ids", ["e99"]),
    ],
)
def test_anchor_missing_from_graph_is_invalid(
    section: str, key: str, value: Any
) -> None:
    obj = _checks()
    obj[section][0][key] = value
    res = validate_checks(obj, GRAPH, LIMITS)
    assert not res.ok
    assert any("not in graph" in v for v in res.violations)


def test_duplicate_ids_across_sections_are_invalid() -> None:
    obj = _checks()
    obj["hypotheses"][0]["id"] = "o1"
    res = validate_checks(obj, GRAPH, LIMITS)
    assert not res.ok
    assert any("duplicate" in v for v in res.violations)


def test_duplicate_ids_within_section_are_invalid() -> None:
    obj = _checks()
    obj["observed_issues"].append(copy.deepcopy(obj["observed_issues"][0]))
    assert not validate_checks(obj, GRAPH, LIMITS).ok


def test_evidence_longer_than_limit_is_invalid() -> None:
    obj = _checks()
    obj["observed_issues"][0]["evidence"] = "x" * (LIMITS.max_text_chars + 1)
    assert not validate_checks(obj, GRAPH, LIMITS).ok
    obj["observed_issues"][0]["evidence"] = "x" * LIMITS.max_text_chars
    assert validate_checks(obj, GRAPH, LIMITS).ok


def test_too_many_hypotheses_is_invalid() -> None:
    obj = _checks()
    base = obj["hypotheses"][0]
    obj["hypotheses"] = [
        {**base, "id": f"h{i}"} for i in range(LIMITS.max_hypotheses + 1)
    ]
    assert not validate_checks(obj, GRAPH, LIMITS).ok


@pytest.mark.parametrize(
    "mutate",
    [
        lambda o: o["hypotheses"][0].__setitem__("kind", "missing_attribute"),
        lambda o: o["observed_issues"][0].__setitem__("extra", 1),
        lambda o: o["observed_issues"][0].pop("target"),
        lambda o: o["observed_issues"][0].__setitem__("anchor_ids", "v1"),
        lambda o: o["observed_issues"][0].__setitem__("id", ""),
        lambda o: o.pop("hypotheses"),
    ],
)
def test_schema_violations_are_invalid(mutate: Any) -> None:
    obj = _checks()
    mutate(obj)
    assert not validate_checks(obj, GRAPH, LIMITS).ok


def test_non_dict_checks_is_invalid() -> None:
    assert not validate_checks([], GRAPH, LIMITS).ok


# --------------------------------------------------------------------------
# Judgments
# --------------------------------------------------------------------------


def test_valid_judgments_pass() -> None:
    res = validate_judgments(_judgments(), _valid_checks(), LIMITS)
    assert res.ok and res.value is not None
    assert res.value.by_id()["h1"].resolution_benefit == 1


@pytest.mark.parametrize("value", [3, -1, True, 1.0, "2"])
def test_basis_out_of_range_is_invalid(value: Any) -> None:
    obj = _judgments()
    obj["items"][0]["basis"] = value
    assert not validate_judgments(obj, _valid_checks(), LIMITS).ok


def test_each_check_id_must_appear_exactly_once() -> None:
    checks = _valid_checks()
    missing = _judgments()
    missing["items"].pop()
    duplicate = _judgments()
    duplicate["items"][1]["id"] = "o1"
    unknown = _judgments()
    unknown["items"].append({**unknown["items"][0], "id": "zz"})
    for obj in (missing, duplicate, unknown):
        assert not validate_judgments(obj, checks, LIMITS).ok


@pytest.mark.parametrize(
    ("key", "value"),
    [("status", "maybe"), ("limitation", "blur"), ("reason", "r" * 241)],
)
def test_judgment_enum_and_text_violations(key: str, value: Any) -> None:
    obj = _judgments()
    obj["items"][0][key] = value
    assert not validate_judgments(obj, _valid_checks(), LIMITS).ok


def test_judgments_wrapper_key_is_rejected_by_contract() -> None:
    obj = {**_judgments(), "failure": False}
    assert not validate_judgments(obj, _valid_checks(), LIMITS).ok


# --------------------------------------------------------------------------
# JSON extraction
# --------------------------------------------------------------------------


def test_parse_json_plain_and_fenced() -> None:
    assert parse_json_response(' {"a": 1}\n') == {"a": 1}
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}


@pytest.mark.parametrize(
    "text",
    ['Sure! {"a": 1}', '{"a": 1} trailing', "[1, 2]", "", '{"a": 1}{"b": 2}'],
)
def test_parse_json_rejects_non_whole_object(text: str) -> None:
    with pytest.raises(StructuredOutputError):
        parse_json_response(text)


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------


def _key(**overrides: Any) -> str:
    args = {
        "image_sha256": "img",
        "model_id": "qwen3.5:4b",
        "prompt_sha256": "p",
        "generation": {"temperature": 0, "seed": 1},
        "graph_sha256": "g",
    }
    return cache_key(**{**args, **overrides})


def test_cache_key_is_deterministic_and_order_insensitive() -> None:
    assert _key() == _key(generation={"seed": 1, "temperature": 0})


@pytest.mark.parametrize(
    "override",
    [
        {"image_sha256": None},
        {"model_id": "other"},
        {"prompt_sha256": "q"},
        {"generation": {"temperature": 0, "seed": 2}},
        {"graph_sha256": "h"},
    ],
)
def test_cache_key_changes_with_each_input(override: dict[str, Any]) -> None:
    assert _key(**override) != _key()


def test_graph_hash_ignores_volatile_stats() -> None:
    g = {"entities": [{"id": "v1"}], "stats": {"elapsed_s": 1.0}}
    assert graph_sha256(g) == graph_sha256({**g, "stats": {"elapsed_s": 2.0}})
    assert graph_sha256(g) != graph_sha256({**g, "entities": []})
