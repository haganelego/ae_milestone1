"""Tests for prompt v2 rendering (entity merge, strict caps) and the image judge (F1v).

Run:  uv run --project egtr --with pytest python -m pytest tests/test_prompt_v2.py -q
"""

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from robovis_fixtures import SYNTHETIC_IMAGE_IDS, make_graph  # noqa: E402
from test_check_generator import LIMITS, VALID_CHECKS, FakeClient  # noqa: E402

from milestone1.check_generator import (  # noqa: E402
    CHECK_TASK_V2_ADDENDUM,
    PATH_F1,
    PROMPT_V2,
    build_check_messages,
    check_system_prompt,
    generate_checks,
    limits_from_config,
    merge_entities,
    render_graph,
)
from milestone1.judge import (  # noqa: E402
    AXES_NOTE_V2,
    BASIS_RULE_V2,
    IMAGE_EVIDENCE,
    TEXT_ONLY_EVIDENCE,
    build_judge_messages,
    judge_checks,
    judge_system_prompt,
)
from milestone1.robovis_data import load_robovis_config  # noqa: E402

LIMITS_V2 = replace(
    LIMITS, max_relations=20, max_entities=30, merge_iou=0.5, string_hint_chars=200
)
IMAGE_ID = SYNTHETIC_IMAGE_IDS[0]
SCALE = 0.2066

DUP_ENTITIES = [
    ("tree", 0.9, (0.0, 0.0, 100.0, 100.0)),
    ("tree", 0.5, (5.0, 5.0, 100.0, 100.0)),  # IoU ~0.9 with v1 -> merged
    ("tree", 0.4, (200.0, 0.0, 300.0, 100.0)),  # same label, no overlap -> kept
    ("man", 0.8, (0.0, 0.0, 100.0, 100.0)),  # other label -> kept
]
DUP_RELATIONS = [
    (3, "near", 0, 0.5),
    (3, "near", 1, 0.4),  # becomes duplicate of e1 after merge
    (0, "near", 1, 0.3),  # becomes a self-loop after merge
    (3, "near", 2, 0.2),
]


def _dup_graph() -> dict[str, Any]:
    return make_graph(
        IMAGE_ID, level=10, scale=SCALE, entities=DUP_ENTITIES, relations=DUP_RELATIONS
    )


def test_merge_entities_merges_same_label_overlap_only() -> None:
    kept, alias = merge_entities(_dup_graph()["entities"], 0.5)
    assert [e["id"] for e in kept] == ["v1", "v3", "v4"]
    assert alias == {"v2": "v1"}


def test_render_v2_remaps_dedupes_and_drops_self_loops() -> None:
    rendered = render_graph(_dup_graph(), LIMITS_V2)
    assert rendered.entity_ids == frozenset({"v1", "v3", "v4"})
    assert rendered.relation_ids == frozenset({"e1", "e4"})
    assert rendered.relation_alias == {"e2": "e1"}
    assert rendered.entity_alias == {"v2": "v1"}
    assert "merged" in rendered.text and "top 30 entities" in rendered.text


def test_render_v2_strict_entity_cap_keeps_relation_endpoints_consistent() -> None:
    limits = replace(LIMITS_V2, max_entities=2)
    rendered = render_graph(make_graph(IMAGE_ID), limits)
    assert rendered.n_entities_shown <= 2
    graph = make_graph(IMAGE_ID)
    rels = {r["id"]: r for r in graph["relations"]}
    for rid in rendered.relation_ids:
        assert {rels[rid]["sub"], rels[rid]["obj"]} <= rendered.entity_ids


def test_render_v1_has_no_merge_note_and_keeps_duplicates() -> None:
    rendered = render_graph(_dup_graph(), LIMITS)
    assert "merged" not in rendered.text
    assert "v2" in rendered.entity_ids and rendered.entity_alias == {}


def test_limits_from_config_v2_reads_overrides() -> None:
    limits = limits_from_config(load_robovis_config(), PROMPT_V2)
    assert (limits.max_relations, limits.max_entities, limits.merge_iou) == (
        20,
        30,
        0.5,
    )
    assert limits_from_config(load_robovis_config()).merge_iou is None
    assert limits.string_hint_chars == 200 < limits.max_string_chars


def test_v2_prompts_contain_scene_scoping_and_no_leak() -> None:
    system = check_system_prompt(LIMITS_V2, PROMPT_V2)
    assert CHECK_TASK_V2_ADDENDUM in system
    assert CHECK_TASK_V2_ADDENDUM not in check_system_prompt(LIMITS)
    msgs = build_check_messages(
        render_graph(_dup_graph(), LIMITS_V2), LIMITS_V2, "A", PROMPT_V2
    )
    msgs += build_judge_messages("g", VALID_CHECKS, LIMITS_V2, PROMPT_V2, "A")
    text = "\n".join(m["content"] for m in msgs)
    assert "level" not in text.lower() and str(SCALE) not in text
    assert str(IMAGE_ID) not in text


def test_judge_system_prompt_image_and_v2_benefit_definition() -> None:
    v1_text = judge_system_prompt(LIMITS)
    assert TEXT_ONLY_EVIDENCE in v1_text and IMAGE_EVIDENCE not in v1_text
    img = judge_system_prompt(LIMITS, with_image=True)
    assert IMAGE_EVIDENCE in img and TEXT_ONLY_EVIDENCE not in img
    v2 = judge_system_prompt(LIMITS_V2, PROMPT_V2, with_image=True)
    assert "small" in v2 and "attached image" in v2
    assert AXES_NOTE_V2 in v2 and BASIS_RULE_V2 in v2
    assert AXES_NOTE_V2 not in v1_text and AXES_NOTE_V2 not in img
    hint = "under 200 characters"
    assert hint in v2 and hint in check_system_prompt(LIMITS_V2, PROMPT_V2)
    assert hint not in v1_text and hint not in check_system_prompt(LIMITS)
    assert v2.index('"out_of_frame", "other"') < v2.index("Return exactly one JSON")


def _image(tmp_path: Path) -> Path:
    path = tmp_path / "level_05.jpg"
    path.write_bytes(b"\xff\xd8fakejpeg")
    return path


def test_judge_with_image_sends_image_and_records_it(tmp_path: Path) -> None:
    judgment = {
        "items": [
            {"id": i, "status": "unresolved", "basis": 2, "resolution_benefit": 2,
             "limitation": "low_resolution", "reason": "small"}
            for i in ("o1", "h1")
        ]
    }  # fmt: skip
    client = FakeClient([json.dumps(judgment)])
    record, _ = judge_checks(
        make_graph(IMAGE_ID),
        {"path": "F1", "checks": VALID_CHECKS, "failure": False},
        client=client,
        model="fake",
        limits=LIMITS,
        schema_retries=1,
        out_path=tmp_path / "d.json",
        image_path=_image(tmp_path),
    )
    assert client.calls[0][1]["images"]
    assert record["judge_saw_image"] is True and record["image_sha256"]
    assert record["failure"] is False


def test_judge_rejects_prompt_version_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        judge_checks(
            make_graph(IMAGE_ID),
            {"path": "F1", "checks": VALID_CHECKS, "failure": False},
            client=FakeClient([]),
            model="fake",
            limits=LIMITS_V2,
            schema_retries=1,
            out_path=tmp_path / "d.json",
            prompt_version=PROMPT_V2,
        )


def test_generate_checks_v2_records_version_and_alias(tmp_path: Path) -> None:
    client = FakeClient(['{"observed_issues": [], "hypotheses": []}'])
    record, _ = generate_checks(
        _dup_graph(),
        client=client,
        model="fake",
        path=PATH_F1,
        limits=LIMITS_V2,
        schema_retries=1,
        out_path=tmp_path / "c.json",
        image_path=_image(tmp_path),
        prompt_version=PROMPT_V2,
    )
    assert record["prompt_version"] == PROMPT_V2
    assert record["render"]["entity_alias"] == {"v2": "v1"}
    assert CHECK_TASK_V2_ADDENDUM in client.calls[0][0]["content"]


def test_generate_checks_and_judge_record_extra_provenance(tmp_path: Path) -> None:
    extra = {"graphs_subdir": "graphs_clean"}
    rec, _ = generate_checks(
        make_graph(IMAGE_ID),
        client=FakeClient([json.dumps(VALID_CHECKS)]),
        model="fake",
        path=PATH_F1,
        limits=LIMITS,
        schema_retries=1,
        out_path=tmp_path / "c.json",
        image_path=_image(tmp_path),
        record_extra=extra,
    )
    assert rec["graphs_subdir"] == "graphs_clean"
    jrec, _ = judge_checks(
        make_graph(IMAGE_ID),
        {"path": "F1", "checks": {"observed_issues": [], "hypotheses": []}},
        client=FakeClient([]),
        model="fake",
        limits=LIMITS,
        schema_retries=1,
        out_path=tmp_path / "d.json",
        record_extra=extra,
    )
    assert jrec["graphs_subdir"] == "graphs_clean" and jrec["skipped_empty"]
