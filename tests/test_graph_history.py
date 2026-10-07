"""Tests for --graph-history (earlier graphs as text context for checks and judge).

Run:  uv run --project egtr --with pytest python -m pytest tests/test_graph_history.py -q
"""

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from robovis_fixtures import SYNTHETIC_IMAGE_IDS, make_graph  # noqa: E402
from test_check_generator import VALID_CHECKS, FakeClient  # noqa: E402

from milestone1.check_generator import (  # noqa: E402
    CURRENT_HEADER,
    PATH_F2,
    PROMPT_V3,
    build_check_messages,
    generate_checks,
    limits_from_config,
    render_graph,
    render_history,
)
from milestone1.judge import build_judge_messages, judge_checks  # noqa: E402
from milestone1.robovis_data import load_robovis_config  # noqa: E402

IMAGE_ID = SYNTHETIC_IMAGE_IDS[0]
LIMITS = limits_from_config(load_robovis_config(), PROMPT_V3)
BOX = (0.0, 0.0, 50.0, 50.0)
OLD_SCALE, MID_SCALE, CUR_SCALE = 0.0585, 0.0802, 0.1288


def _history() -> list[dict[str, Any]]:
    oldest = make_graph(
        IMAGE_ID, level=2, scale=OLD_SCALE,
        entities=[("horse", 0.9, BOX), ("field", 0.5, (0.0, 50.0, 384.0, 300.0))],
        relations=[(0, "on", 1, 0.4)],
    )  # fmt: skip
    newer = make_graph(
        IMAGE_ID, level=3, scale=MID_SCALE,
        entities=[("zebra", 0.8, BOX)], relations=[],
    )  # fmt: skip
    return [oldest, newer]


def _current() -> dict[str, Any]:
    return make_graph(IMAGE_ID, level=4, scale=CUR_SCALE)


def test_history_text_has_no_level_scale_ids_or_image_id() -> None:
    text = render_history(_history(), LIMITS)
    lowered = text.lower()
    assert "level" not in lowered and "scale" not in lowered
    for value in (OLD_SCALE, MID_SCALE, CUR_SCALE, IMAGE_ID):
        assert str(value) not in text
    assert "v1" not in text and "e1" not in text  # earlier ids are not citable


def test_history_is_ordered_oldest_to_newest_and_precedes_current() -> None:
    history = render_history(_history(), LIMITS)
    rendered = render_graph(_current(), LIMITS)
    user = build_check_messages(rendered, LIMITS, "IMG", PROMPT_V3, history)[1]
    text = user["content"]
    assert text.index("Earlier observation 1 (oldest)") < text.index("horse")
    assert text.index("horse on field") < text.index("Earlier observation 2")
    assert text.index("Earlier observation 2 (most recent)") < text.index("zebra")
    assert (
        text.index("zebra") < text.index(CURRENT_HEADER) < text.index("v1 motorcycle")
    )
    assert "level" not in text.lower() and str(CUR_SCALE) not in text


def test_no_history_leaves_prompt_unchanged() -> None:
    rendered = render_graph(_current(), LIMITS)
    assert render_history([], LIMITS) == "" and render_history(None, LIMITS) == ""
    plain = build_check_messages(rendered, LIMITS, None, PROMPT_V3)
    explicit = build_check_messages(rendered, LIMITS, None, PROMPT_V3, "")
    assert plain == explicit and CURRENT_HEADER not in plain[1]["content"]


def test_cache_key_includes_history(tmp_path: Path) -> None:
    def run(history: list[dict[str, Any]] | None, name: str) -> dict[str, Any]:
        rec, _ = generate_checks(
            _current(),
            client=FakeClient([json.dumps(VALID_CHECKS)]),
            model="fake",
            path=PATH_F2,
            limits=LIMITS,
            schema_retries=1,
            out_path=tmp_path / name,
            prompt_version=PROMPT_V3,
            history_graphs=history,
        )
        return rec

    plain, hist = run(None, "a.json"), run(_history(), "b.json")
    assert plain["cache_key"] != hist["cache_key"]
    assert plain["graph_history"] == 0 and plain["history_sha256"] is None
    assert hist["graph_history"] == 2 and hist["history_sha256"]


def test_judge_receives_history(tmp_path: Path) -> None:
    history = render_history(_history(), LIMITS)
    msgs = build_judge_messages("graph", VALID_CHECKS, LIMITS, PROMPT_V3, None, history)
    assert "Earlier observation 1 (oldest)" in msgs[1]["content"]
    judgment = {
        "items": [
            {"id": i, "status": "resolved", "basis": 2, "resolution_benefit": 0,
             "limitation": "other", "reason": "ok"}
            for i in ("o1", "h1")
        ]
    }  # fmt: skip
    client = FakeClient([json.dumps(judgment)])
    rec, _ = judge_checks(
        _current(),
        {"path": "F1", "checks": VALID_CHECKS, "prompt_version": "v2"},
        client=client,
        model="fake",
        limits=LIMITS,
        schema_retries=1,
        out_path=tmp_path / "d.json",
        prompt_version=PROMPT_V3,
        history_graphs=_history(),
    )
    assert "Earlier observation 2 (most recent)" in client.calls[0][1]["content"]
    assert rec["graph_history"] == 2 and not rec["failure"]
