"""Tests for the ROBOVIS2027 judgment step.

Run:  uv run --project egtr --with pytest python -m pytest tests/test_judge.py -q
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from robovis_fixtures import SYNTHETIC_IMAGE_IDS, make_graph  # noqa: E402
from test_check_generator import LIMITS, VALID_CHECKS, FakeClient  # noqa: E402

from milestone1.judge import (  # noqa: E402
    build_judge_messages,
    judge_checks,
    validate_judgment,
)

IMAGE_ID = SYNTHETIC_IMAGE_IDS[1]
SCALE = 0.3316
EXPECTED_IDS = ["o1", "h1"]


def _item(item_id: str, **changes: Any) -> dict[str, Any]:
    base = {
        "id": item_id,
        "status": "unresolved",
        "basis": 1,
        "resolution_benefit": 2,
        "limitation": "low_resolution",
        "reason": "small sign region",
    }
    return {**base, **changes}


VALID_JUDGMENT = {"items": [_item("o1"), _item("h1", status="unsupported", basis=0)]}


def _checks_record(checks: dict[str, Any] | None, failure: bool = False) -> dict:
    return {"path": "F2", "cache_key": "k", "checks": checks, "failure": failure}


def _graph() -> dict[str, Any]:
    return make_graph(SYNTHETIC_IMAGE_IDS[0], level=13, scale=SCALE)


def _judge(client: FakeClient, tmp_path: Path, record: dict[str, Any]):
    return judge_checks(
        _graph(),
        record,
        client=client,
        model="fake",
        limits=LIMITS,
        schema_retries=1,
        out_path=tmp_path / "level_13.json",
    )


def test_validate_judgment_valid_orders_by_check_ids() -> None:
    data = {"items": list(reversed(VALID_JUDGMENT["items"]))}
    out, errors = validate_judgment(data, EXPECTED_IDS, LIMITS)
    assert errors == []
    assert [i["id"] for i in out["items"]] == EXPECTED_IDS


@pytest.mark.parametrize(
    "items",
    [
        [_item("o1")],
        [_item("o1"), _item("h1"), _item("h1")],
        [_item("o1"), _item("h9")],
        [_item("o1"), _item("h1", status="maybe")],
        [_item("o1"), _item("h1", basis=3)],
        [_item("o1"), _item("h1", basis=True)],
        [_item("o1"), _item("h1", resolution_benefit="2")],
        [_item("o1"), _item("h1", limitation="blur")],
        [_item("o1"), _item("h1", reason="r" * 241)],
    ],
    ids=[
        "missing",
        "dup",
        "unknown",
        "bad_status",
        "basis_range",
        "basis_bool",
        "benefit_str",
        "bad_limitation",
        "long_reason",
    ],
)
def test_validate_judgment_invalid_returns_errors(items: list[dict]) -> None:
    out, errors = validate_judgment({"items": items}, EXPECTED_IDS, LIMITS)
    assert out is None and errors


def test_judge_prompt_contains_no_level_scale_or_image_id() -> None:
    messages = build_judge_messages("graph text", VALID_CHECKS, LIMITS)
    text = "\n".join(m["content"] for m in messages)
    assert "level" not in text.lower()
    assert str(SCALE) not in text and str(IMAGE_ID) not in text
    assert all("images" not in m for m in messages)
    assert "o1, h1" in messages[1]["content"]


def test_judge_checks_empty_checks_skips_model(tmp_path: Path) -> None:
    client = FakeClient([])
    record, _ = _judge(
        client, tmp_path, _checks_record({"observed_issues": [], "hypotheses": []})
    )
    assert client.calls == []
    assert record["skipped_empty"] is True
    assert record["items"] == [] and record["failure"] is False


def test_judge_checks_upstream_failure_is_failure_without_call(tmp_path: Path) -> None:
    client = FakeClient([])
    record, _ = _judge(client, tmp_path, _checks_record(None, failure=True))
    assert client.calls == [] and record["failure"] is True


def test_judge_checks_schema_fail_then_success(tmp_path: Path) -> None:
    client = FakeClient(['{"items": []}', json.dumps(VALID_JUDGMENT)])
    record, _ = _judge(client, tmp_path, _checks_record(VALID_CHECKS))
    assert record["attempts"] == 2 and record["failure"] is False
    assert record["seeds"] == [1, 2] and record["seed_used"] == 2
    assert [i["id"] for i in record["items"]] == EXPECTED_IDS


def test_judge_checks_two_schema_fails_records_failure(tmp_path: Path) -> None:
    client = FakeClient(['{"items": []}', '{"items": []}'])
    record, _ = _judge(client, tmp_path, _checks_record(VALID_CHECKS))
    assert record["failure"] is True and record["items"] is None


def test_judge_checks_cache_hit(tmp_path: Path) -> None:
    _judge(
        FakeClient([json.dumps(VALID_JUDGMENT)]), tmp_path, _checks_record(VALID_CHECKS)
    )
    client = FakeClient([])
    _, cached = _judge(client, tmp_path, _checks_record(VALID_CHECKS))
    assert cached and client.calls == []


def _load_summary_module():
    import importlib.util

    path = REPO_ROOT / "scripts" / "29b_robovis_checks_summary.py"
    spec = importlib.util.spec_from_file_location("checks_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_decisions_gate_requires_region_and_theta() -> None:
    summary = _load_summary_module()
    checks = json.loads(json.dumps(VALID_CHECKS))
    checks["observed_issues"][0]["region_anchor_id"] = None  # o1 cannot gate
    decision = {
        "items": [
            _item("o1", basis=2, resolution_benefit=2),
            _item("h1", basis=2, resolution_benefit=2),
        ],
        "failure": False,
    }
    entity_ids = {e["id"] for e in _graph()["entities"]}
    out = summary.summarize_decisions(
        [(decision, {"checks": checks}, entity_ids)], theta=2
    )
    assert out["frac_items_gate"] == 0.5
    assert out["frac_items_min_basis_benefit_ge_theta"] == 1.0
    assert out["frac_items_gate_and_min_ge_theta"] == 0.5
    assert out["frac_states_any_gate_and_min_ge_theta"] == 1.0


def test_summarize_checks_counts_empty_and_failures() -> None:
    summary = _load_summary_module()
    records = [
        {"checks": {"observed_issues": [], "hypotheses": []}, "failure": False},
        {"checks": VALID_CHECKS, "failure": False},
        {"checks": None, "failure": True},
    ]
    out = summary.summarize_checks(records)
    assert out["n_failures"] == 1 and out["n_valid"] == 2
    assert out["frac_both_empty"] == 0.5
    assert out["kind"]["counts"] == {"missing_object": 1}
