"""Tests for scripts/40 (MIRU VLM+SGG scoring): message layout and score validation."""

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
_SPEC = importlib.util.spec_from_file_location(
    "s40", REPO_ROOT / "scripts/40_vlm_sgg_saturation_scoring.py"
)
s40 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(s40)

from milestone1.llm import SATURATION_PROMPT  # noqa: E402

SEQ = [{"level": 5, "objects": [], "relationships": []}]


def test_text_message_is_exactly_script09_content() -> None:
    msgs = s40.build_messages(SEQ, None)
    assert msgs == [
        {"role": "user", "content": SATURATION_PROMPT + json.dumps(SEQ, indent=2)}
    ]


def test_image_message_adds_only_note_and_image() -> None:
    msg = s40.build_messages(SEQ, "QUJD")[0]
    assert msg["images"] == ["QUJD"]
    assert msg["content"].replace(
        s40.IMAGE_NOTE, "", 1
    ) == SATURATION_PROMPT + json.dumps(SEQ, indent=2)
    assert msg["content"].index(s40.IMAGE_NOTE) < msg["content"].index(s40.SCENE_MARKER)


def test_validate_score_clamps_and_fills_defaults() -> None:
    out, err = s40.validate_score({"saturation_score": 1.4, "reason": "ok"})
    assert not err and out["saturation_score"] == 1.0 and out["score_clamped"]
    assert out["stop_decision"] is True  # missing -> score >= 0.8
    out, _ = s40.validate_score({"saturation_score": "0.5", "stop_decision": "false"})
    assert (
        out["saturation_score"] == 0.5
        and out["stop_decision"] is False
        and out["reason"] == ""
    )


def test_validate_score_rejects_invalid() -> None:
    assert s40.validate_score({"stop_decision": True})[1]
    assert s40.validate_score({"saturation_score": "high"})[1]
    assert s40.validate_score({"saturation_score": float("nan")})[1]
    assert s40.validate_score([0.5])[1]


def test_consecutive_k_stop_matches_script13_rule() -> None:
    scores = {5: 0.8, 6: 0.5, 7: 0.75, 8: 0.9}
    assert s40.consecutive_k_stop(scores, 0.7, 1, 5, 20) == 5
    assert s40.consecutive_k_stop(scores, 0.7, 2, 5, 20) == 8
    assert s40.consecutive_k_stop({}, 0.7, 1, 5, 20) == 20
