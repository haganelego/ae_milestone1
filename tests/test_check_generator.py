"""Tests for the ROBOVIS2027 check generator, validator and ollama retry logic.

Run:  uv run --project egtr --with pytest python -m pytest tests/test_check_generator.py -q
No ollama needed: the chat client is faked.
"""

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from robovis_fixtures import SYNTHETIC_IMAGE_IDS, make_graph  # noqa: E402

from milestone1.check_generator import (  # noqa: E402
    PATH_F1,
    PATH_F2,
    CheckLimits,
    build_check_messages,
    generate_checks,
    limits_from_config,
    render_graph,
    validate_checks,
)
from milestone1.ollama_client import (  # noqa: E402
    ChatResult,
    OllamaClient,
    OllamaError,
    OllamaSettings,
    parse_json_object,
)
from milestone1.robovis_data import load_robovis_config  # noqa: E402

LIMITS = CheckLimits(
    max_observed_issues=3,
    max_hypotheses=3,
    max_string_chars=240,
    max_relations=50,
    max_entities=60,
    box_decimals=2,
    score_decimals=2,
)
SETTINGS = OllamaSettings(
    endpoint="http://localhost:11434/api/chat",
    think=False,
    json_mode=True,
    options={"temperature": 0, "seed": 1},
    timeout_s=10.0,
    transient_retries=5,
    backoff_base_s=0.0,
    backoff_max_s=0.0,
)
IMAGE_ID = SYNTHETIC_IMAGE_IDS[0]
LEVEL = 7
SCALE = 0.1288

VALID_CHECKS = {
    "observed_issues": [
        {
            "id": "o1",
            "anchor_ids": ["v7"],
            "target": "sign text unreadable",
            "region_anchor_id": "v7",
            "evidence": "low score sign",
        }
    ],
    "hypotheses": [
        {
            "id": "h1",
            "kind": "missing_object",
            "candidate": "helmet",
            "anchor_ids": ["v1", "v4"],
            "relation_ids": ["e3"],
            "region_anchor_id": "v4",
            "evidence": "seat on motorcycle",
        }
    ],
}


class FakeClient:
    """Returns queued contents (or raises queued exceptions) in order."""

    def __init__(self, responses: list[Any]) -> None:
        self.settings = SETTINGS
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []
        self.overrides: list[dict[str, Any] | None] = []

    def chat(
        self,
        model: str,
        messages: Sequence[dict[str, Any]],
        options_override: dict[str, Any] | None = None,
    ) -> ChatResult:
        self.calls.append(list(messages))
        self.overrides.append(options_override)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return ChatResult(content=item, elapsed_s=0.01, http_attempts=1, usage={})


def _graph() -> dict[str, Any]:
    return make_graph(IMAGE_ID, level=LEVEL, scale=SCALE)


def _rendered():
    return render_graph(_graph(), LIMITS)


# --- rendering is state-only -------------------------------------------------


def test_render_graph_with_level_and_scale_contains_no_leak() -> None:
    text = "\n".join(
        m["content"] for m in build_check_messages(_rendered(), LIMITS, "AAAA")
    )
    lowered = text.lower()
    assert "level" not in lowered
    assert str(SCALE) not in text
    assert str(IMAGE_ID) not in text
    assert "scale" not in lowered


def test_render_graph_caps_relations_and_notes_cap() -> None:
    limits = CheckLimits(**{**LIMITS.__dict__, "max_relations": 2})
    rendered = render_graph(_graph(), limits)
    assert rendered.relation_ids == frozenset({"e1", "e2"})
    assert "only the top 2" in rendered.text
    assert rendered.n_relations_total == len(_graph()["relations"])


def test_render_graph_keeps_entities_referenced_by_relations() -> None:
    limits = CheckLimits(**{**LIMITS.__dict__, "max_entities": 1})
    rendered = render_graph(_graph(), limits)
    assert {"v1", "v2", "v3"} <= rendered.entity_ids


def test_render_graph_empty_graph_is_valid() -> None:
    graph = make_graph(IMAGE_ID, entities=[], relations=[])
    rendered = render_graph(graph, LIMITS)
    assert rendered.entity_ids == frozenset()
    assert "(none)" in rendered.text


def test_config_limits_load() -> None:
    limits = limits_from_config(load_robovis_config())
    assert limits.max_hypotheses == 3 and limits.max_observed_issues == 3


# --- validator ---------------------------------------------------------------


def test_validate_checks_valid_returns_normalized() -> None:
    out, errors = validate_checks(VALID_CHECKS, _rendered(), LIMITS)
    assert errors == []
    assert out == VALID_CHECKS


def test_validate_checks_empty_lists_are_valid() -> None:
    out, errors = validate_checks(
        {"observed_issues": [], "hypotheses": []}, _rendered(), LIMITS
    )
    assert errors == [] and out == {"observed_issues": [], "hypotheses": []}


def _mutated(**hyp_changes: Any) -> dict[str, Any]:
    data = json.loads(json.dumps(VALID_CHECKS))
    data["hypotheses"][0].update(hyp_changes)
    return data


@pytest.mark.parametrize(
    "data",
    [
        _mutated(anchor_ids=["v99"]),
        _mutated(anchor_ids=[]),
        _mutated(relation_ids=["e99"]),
        _mutated(region_anchor_id="v99"),
        _mutated(kind="missing_attribute"),
        _mutated(candidate="x" * 241),
        _mutated(id="o1"),
        {"observed_issues": [], "hypotheses": [VALID_CHECKS["hypotheses"][0]] * 2},
        {
            "observed_issues": [],
            "hypotheses": [
                {**VALID_CHECKS["hypotheses"][0], "id": f"h{i}"} for i in range(1, 5)
            ],
        },
        {"hypotheses": []},
        [],
    ],
    ids=[
        "missing_anchor",
        "empty_anchor",
        "missing_relation",
        "bad_region",
        "bad_kind",
        "too_long",
        "wrong_id_prefix",
        "dup_id",
        "too_many",
        "missing_key",
        "not_object",
    ],
)
def test_validate_checks_invalid_returns_errors(data: Any) -> None:
    out, errors = validate_checks(data, _rendered(), LIMITS)
    assert out is None
    assert errors


def test_validate_checks_string_null_region_is_normalized() -> None:
    data = json.loads(json.dumps(VALID_CHECKS))
    data["hypotheses"][0]["region_anchor_id"] = "null"
    out, errors = validate_checks(data, _rendered(), LIMITS)
    assert errors == [] and out["hypotheses"][0]["region_anchor_id"] is None


def test_validate_checks_missing_region_key_is_error() -> None:
    data = json.loads(json.dumps(VALID_CHECKS))
    del data["observed_issues"][0]["region_anchor_id"]
    _, errors = validate_checks(data, _rendered(), LIMITS)
    assert any("region_anchor_id missing" in e for e in errors)


def test_validate_checks_anchor_outside_rendered_cap_is_error() -> None:
    limits = CheckLimits(**{**LIMITS.__dict__, "max_relations": 1})
    rendered = render_graph(_graph(), limits)
    _, errors = validate_checks(VALID_CHECKS, rendered, limits)
    assert any("e3" in e for e in errors)  # e3 is no longer shown


# --- retry logic -------------------------------------------------------------


def _run(client: FakeClient, tmp_path: Path, path: str = PATH_F2, image=None):
    return generate_checks(
        _graph(),
        client=client,
        model="fake",
        path=path,
        limits=LIMITS,
        schema_retries=1,
        out_path=tmp_path / "level_07.json",
        image_path=image,
    )


def test_generate_checks_schema_fail_then_success_records_two_attempts(
    tmp_path: Path,
) -> None:
    client = FakeClient(["not json", json.dumps(VALID_CHECKS)])
    record, cached = _run(client, tmp_path)
    assert not cached
    assert record["attempts"] == 2
    assert record["failure"] is False and record["checks"] == VALID_CHECKS
    assert client.calls[0] == client.calls[1]  # same input on retry
    assert client.overrides == [{"seed": 1}, {"seed": 2}]  # seed + 1 on retry
    assert record["seeds"] == [1, 2] and record["seed_used"] == 2


def test_generate_checks_two_schema_fails_records_failure(tmp_path: Path) -> None:
    bad = json.dumps(_mutated(anchor_ids=["v99"]))
    client = FakeClient([bad, bad])
    record, _ = _run(client, tmp_path)
    assert record["failure"] is True
    assert record["checks"] is None
    assert record["attempts"] == 2
    assert record["error"].startswith("schema:")
    assert record["seeds"] == [1, 2] and record["seed_used"] is None


def test_generate_checks_empty_answer_is_valid_not_failure(tmp_path: Path) -> None:
    client = FakeClient(['{"observed_issues": [], "hypotheses": []}'])
    record, _ = _run(client, tmp_path)
    assert record["failure"] is False and record["attempts"] == 1
    assert record["seed_used"] == 1


def test_generate_checks_http_failure_is_recorded(tmp_path: Path) -> None:
    client = FakeClient([OllamaError("transient failure after 6 attempts")])
    record, _ = _run(client, tmp_path)
    assert record["failure"] is True and record["error"].startswith("http:")


def test_generate_checks_cache_hit_skips_model(tmp_path: Path) -> None:
    _run(FakeClient([json.dumps(VALID_CHECKS)]), tmp_path)
    client = FakeClient([])
    record, cached = _run(client, tmp_path)
    assert cached and client.calls == [] and record["checks"] == VALID_CHECKS


def test_generate_checks_failure_is_not_reused_from_cache(tmp_path: Path) -> None:
    _run(FakeClient(["x", "x"]), tmp_path)
    record, cached = _run(FakeClient([json.dumps(VALID_CHECKS)]), tmp_path)
    assert not cached and record["failure"] is False


def test_generate_checks_f1_sends_image_and_hash(tmp_path: Path) -> None:
    image = tmp_path / "img.jpg"
    image.write_bytes(b"\xff\xd8fakejpeg")
    client = FakeClient([json.dumps(VALID_CHECKS)])
    record, _ = _run(client, tmp_path, path=PATH_F1, image=image)
    assert client.calls[0][1]["images"]
    assert record["image_sha256"] is not None


def test_generate_checks_f2_rejects_image(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _run(FakeClient([]), tmp_path, path=PATH_F2, image=tmp_path / "x.jpg")


# --- ollama client -----------------------------------------------------------


class _Resp:
    def __init__(self, status: int, body: dict[str, Any] | None = None) -> None:
        self.status_code = status
        self._body = body or {}
        self.text = json.dumps(self._body)

    def json(self) -> dict[str, Any]:
        return self._body


def test_ollama_client_retries_transient_then_succeeds() -> None:
    seq: list[Any] = [
        requests.ConnectionError("down"),
        _Resp(503),
        _Resp(200, {"message": {"content": "{}"}, "eval_count": 3}),
    ]
    payloads: list[dict[str, Any]] = []

    def post(url: str, json: dict[str, Any], timeout: float) -> Any:
        payloads.append(json)
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    result = OllamaClient(SETTINGS, post=post, sleep=lambda _s: None).chat("m", [])
    assert result.http_attempts == 3 and result.usage == {"eval_count": 3}
    assert payloads[0]["think"] is False and payloads[0]["format"] == "json"
    assert payloads[0]["options"] == {"temperature": 0, "seed": 1}


def test_ollama_client_options_override_merges_without_mutating() -> None:
    client = OllamaClient(SETTINGS)
    payload = client.build_payload("m", [], {"seed": 2})
    assert payload["options"] == {"temperature": 0, "seed": 2}
    assert SETTINGS.options["seed"] == 1


def test_ollama_client_gives_up_after_transient_retries() -> None:
    def post(url: str, json: dict[str, Any], timeout: float) -> Any:
        return _Resp(500)

    client = OllamaClient(SETTINGS, post=post, sleep=lambda _s: None)
    with pytest.raises(OllamaError):
        client.chat("m", [])


def test_ollama_client_client_error_is_not_retried() -> None:
    calls = []

    def post(url: str, json: dict[str, Any], timeout: float) -> Any:
        calls.append(1)
        return _Resp(404)

    with pytest.raises(OllamaError):
        OllamaClient(SETTINGS, post=post, sleep=lambda _s: None).chat("m", [])
    assert len(calls) == 1


def test_parse_json_object_tolerates_fences() -> None:
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_object("no json here")
