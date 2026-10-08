"""Parser v2 (single fenced block unwrapped), parse_version namespacing, and
initial-vs-repair validity in the pilot format gate."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest
from gqa_stop_helpers import CONFIG_PATH, PROMPTS_DIR

from milestone1.gqa_stop import client
from milestone1.gqa_stop.client import (
    PARSE_VERSION,
    RequestEngine,
    TransportResult,
    load_prompts,
    parse_response,
    settings_from_config,
    strip_single_fence,
)
from milestone1.gqa_stop.inference import build_level_spec
from milestone1.gqa_stop.schemas import validate_checklist, validate_reader
from milestone1.gqa_stop.stages_model import format_rates

CFG = json.loads(CONFIG_PATH.read_text())
SETTINGS = dataclasses.replace(settings_from_config(CFG), digest="abc")
BODY = json.dumps({"answer": "red"})
PRETTY = json.dumps({"answer": "red"}, indent=2)


@pytest.mark.parametrize(
    "raw",
    [
        f"```json\n{BODY}\n```",
        f"```\n{BODY}\n```",
        f"  \n```json\n{PRETTY}\n```\n\n",
        f"```JSON\n{PRETTY}\n```",
        f"```json\n{PRETTY}```",  # closing fence on the last JSON line (seen once)
        f"```json {BODY}```",  # single-line fence
    ],
)
def test_single_fenced_block_parses(raw: str) -> None:
    result = parse_response(raw, validate_reader)
    assert result.errors == [] and result.parsed == {"answer": "red"}
    assert result.fence_stripped is True


def test_plain_json_is_not_marked_stripped() -> None:
    result = parse_response(f"  {BODY}\n", validate_reader)
    assert result.parsed == {"answer": "red"} and result.fence_stripped is False


@pytest.mark.parametrize(
    "raw",
    [
        f"Here you go:\n```json\n{BODY}\n```",  # text before
        f"```json\n{BODY}\n```\nHope this helps",  # text after
        f"```json\n{BODY}\n```\n```json\n{BODY}\n```",  # two fences
        f"```python\n{BODY}\n```",  # other language tag
        f"```json\n{BODY}\nextra text\n```",  # inner is not pure JSON
        '```json\n{"answer": "red"} {"answer": "blue"}\n```',  # two objects inside
        f"```json\n{BODY}",  # unterminated
    ],
)
def test_anything_but_one_clean_block_still_fails(raw: str) -> None:
    result = parse_response(raw, validate_reader)
    assert result.parsed is None and result.errors


def test_fenced_checklist_like_model_output() -> None:
    raw = (
        "```json\n"
        + json.dumps(
            {
                "items": [
                    {
                        "id": "c1",
                        "answer": "unknown",
                        "resolved": False,
                        "confidence": 0.0,
                    }
                ]
            },
            indent=2,
        )
        + "\n```"
    )
    result = parse_response(raw, lambda o: validate_checklist(o, ["c1"]))
    assert result.errors == [] and result.fence_stripped


def test_strip_returns_raw_unchanged_when_not_single_block() -> None:
    assert strip_single_fence("no fence") == ("no fence", False)


class Scripted:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.n = 0

    def send(self, payload: dict) -> TransportResult:
        self.n += 1
        return TransportResult(self.replies.pop(0), 1, 0.0)


def spec():
    question = {"question_id": "q", "image_id": "1", "question": "What color?"}
    return build_level_spec("answer_only", question, 1, b"img")


def test_fenced_reply_is_valid_on_first_attempt_and_recorded(tmp_path: Path) -> None:
    transport = Scripted([f"```json\n{BODY}\n```"])
    record = RequestEngine(
        SETTINGS, load_prompts(PROMPTS_DIR), transport, tmp_path
    ).run(spec(), validate_reader)
    assert transport.n == 1  # no repair call
    assert record["status"] == "valid" and record["attempt"] == 1
    assert record["initial_valid"] is True and record["fence_stripped"] is True
    assert record["parse_version"] == PARSE_VERSION
    assert PARSE_VERSION in str(
        RequestEngine(
            SETTINGS, load_prompts(PROMPTS_DIR), transport, tmp_path
        ).record_path(spec())
    )


def test_parse_version_change_invalidates_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = load_prompts(PROMPTS_DIR)
    RequestEngine(SETTINGS, prompts, Scripted([BODY]), tmp_path).run(
        spec(), validate_reader
    )
    same = Scripted([])
    RequestEngine(SETTINGS, prompts, same, tmp_path).run(spec(), validate_reader)
    assert same.n == 0
    monkeypatch.setattr(client, "PARSE_VERSION", "parse-v3")
    fresh = Scripted([BODY])
    engine = RequestEngine(SETTINGS, prompts, fresh, tmp_path)
    engine.run(spec(), validate_reader)
    assert fresh.n == 1  # new key and new directory -> re-called
    old_dirs = sorted(
        p.name for p in (tmp_path / "reader" / "qwen3.5-9b@abc").iterdir()
    )
    assert old_dirs == ["parse-v2", "parse-v3"]  # old records kept on disk


def test_repaired_call_is_not_initial_valid(tmp_path: Path) -> None:
    transport = Scripted(["Sure! " + BODY, BODY])
    record = RequestEngine(
        SETTINGS, load_prompts(PROMPTS_DIR), transport, tmp_path
    ).run(spec(), validate_reader)
    assert record["status"] == "valid" and record["attempt"] == 2
    assert record["initial_valid"] is False


def test_format_rates_separate_initial_and_repair() -> None:
    rows = []
    for i in range(10):
        repaired = i < 3
        rows.append(
            {
                "question_id": f"q{i}",
                "level": 1,
                "reader_status": "valid",
                "reader_initial_valid": True,
                "reader_attempts": 1,
                "direct_status": "valid",
                "direct_initial_valid": True,
                "direct_attempts": 1,
                "check_status": "valid" if i != 9 else "schema_failure",
                "check_initial_valid": not repaired and i != 9,
                "check_attempts": 2 if (repaired or i == 9) else 1,
                "planner_status": "valid",
                "planner_initial_valid": True,
                "planner_attempts": 1,
            }
        )
    table = pd.DataFrame(rows)
    initial, after, repair = format_rates(table, table.groupby("question_id").first())
    assert initial["checklist"] == pytest.approx(0.6)
    assert after["checklist"] == pytest.approx(0.9)
    assert repair["checklist"] == pytest.approx(0.4)
    assert initial["reader"] == 1.0 and repair["reader"] == 0.0
