"""Tests for judge v4 (bbox-grounded visibility).

Run:  uv run --project egtr --with pytest python -m pytest tests/test_grounded_judge.py -q
"""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from robovis_fixtures import SYNTHETIC_IMAGE_IDS, make_graph  # noqa: E402

from milestone1.grounded_judge import (  # noqa: E402
    OverlayStyle,
    build_v4_prompt,
    derive_observed,
    draw_overlay,
    reuse_hypotheses,
    top_entities,
    validate_v4,
)

IMAGE_ID = SYNTHETIC_IMAGE_IDS[0]
SCALE = 0.1288
STYLE = OverlayStyle(line_width=1, color=(255, 0, 0), font_size=10, jpeg_quality=95)


def _graph():
    return make_graph(IMAGE_ID, level=7, scale=SCALE)


def test_top_entities_by_score() -> None:
    ents = top_entities(_graph(), 3)
    assert [e["label"] for e in ents] == ["motorcycle", "wheel", "wheel"]


def test_v4_prompt_lists_ids_and_has_no_level_scale_or_image_id() -> None:
    g = _graph()
    text = build_v4_prompt(top_entities(g, 8), g["base_size"], 2, 200)
    assert "v1: motorcycle, box=[" in text
    assert "level" not in text.lower() and "scale" not in text.lower()
    assert "resolution" not in text.lower()
    assert str(SCALE) not in text and str(IMAGE_ID) not in text


def test_validate_v4_requires_every_id_once_and_grades() -> None:
    ok = {
        "objects": [
            {"id": "v1", "clear": 2, "note": "bike"},
            {"id": "v2", "clear": 0, "note": ""},
        ]
    }
    out, errors = validate_v4(ok, ["v1", "v2"], 240)
    assert errors == [] and [o["id"] for o in out["objects"]] == ["v1", "v2"]
    for bad in (
        {"objects": [{"id": "v1", "clear": 2, "note": "x"}]},
        {
            "objects": [
                {"id": "v1", "clear": 3, "note": "x"},
                {"id": "v2", "clear": 0, "note": ""},
            ]
        },
        {
            "objects": [
                {"id": "v1", "clear": True, "note": "x"},
                {"id": "v2", "clear": 0, "note": ""},
            ]
        },
        {
            "objects": [
                {"id": "v9", "clear": 1, "note": "x"},
                {"id": "v2", "clear": 0, "note": ""},
            ]
        },
        [],
    ):
        assert validate_v4(bad, ["v1", "v2"], 240)[0] is None
    # fmt: skip


def test_derive_observed_caps_least_clear_first_and_maps_grades() -> None:
    objects = [
        {"id": "v1", "clear": 2, "note": "clear bike"},
        {"id": "v2", "clear": 1, "note": "blurry wheel"},
        {"id": "v3", "clear": 0, "note": "nothing"},
        {"id": "v4", "clear": 2, "note": "seat"},
        {"id": "v9", "clear": 1, "note": "street"},
    ]
    ents_by_id = [
        e for e in _graph()["entities"] if e["id"] in {"v1", "v2", "v3", "v4", "v9"}
    ]
    issues, items = derive_observed(objects, ents_by_id, 3)
    assert [i["anchor_ids"][0] for i in issues] == ["v3", "v2", "v9"]
    assert [i["id"] for i in issues] == ["o1", "o2", "o3"]
    assert all(i["region_anchor_id"] == i["anchor_ids"][0] for i in issues)
    assert [(j["status"], j["basis"], j["resolution_benefit"]) for j in items] == [
        ("unresolved", 2, 1),
        ("unresolved", 2, 2),
        ("unresolved", 2, 2),
    ]
    issues2, items2 = derive_observed(objects[:1], ents_by_id, 3)
    assert items2[0]["status"] == "resolved" and items2[0]["resolution_benefit"] == 0
    assert issues2[0]["target"] == "identity/extent of motorcycle"


def test_draw_overlay_returns_same_size_jpeg() -> None:
    buf = io.BytesIO()
    Image.new("RGB", (384, 308), (90, 90, 90)).save(buf, format="JPEG")
    out = draw_overlay(buf.getvalue(), top_entities(_graph(), 8), STYLE)
    with Image.open(io.BytesIO(out)) as img:
        assert img.size == (384, 308) and img.format == "JPEG"


@pytest.mark.parametrize("failure", [True, False])
def test_reuse_hypotheses_drops_unjudged(failure: bool) -> None:
    hyp = {"id": "h1", "kind": "missing_object", "candidate": "x", "anchor_ids": ["v1"],
           "relation_ids": [], "region_anchor_id": None, "evidence": ""}  # fmt: skip
    checks = {"failure": False, "checks": {"observed_issues": [], "hypotheses": [hyp]}}
    decision = {"failure": failure, "items": [{"id": "h1", "status": "unresolved"}]}
    kept, items, dropped = reuse_hypotheses(checks, decision)
    if failure:
        assert kept == [] and dropped == ["h1"]
    else:
        assert kept == [hyp] and items[0]["id"] == "h1" and dropped == []
