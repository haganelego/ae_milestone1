"""Regression tests for the independent review (gqa-stop-phase-a-review.md).

One section per review item; numbers follow the orchestrator's fix list.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests
from gqa_stop_helpers import (
    CONFIG_PATH,
    PROMPTS_DIR,
    FakeTransport,
    answers_by_text,
    fake_tags_get,
    make_context,
    make_world,
)
from PIL import Image

from milestone1.gqa_stop import report, stages_data, stages_model
from milestone1.gqa_stop.client import (
    OllamaTransport,
    PromptSet,
    RequestEngine,
    TransportError,
    TransportResult,
    load_prompts,
    model_tag,
    settings_from_config,
)
from milestone1.gqa_stop.common import (
    StageError,
    atomic_write_json,
    read_id_csv,
    read_json,
    read_jsonl,
    seeded_hash,
)
from milestone1.gqa_stop.evaluate import build_eval_data, evaluate
from milestone1.gqa_stop.images import ImageSeries
from milestone1.gqa_stop.inference import build_level_spec
from milestone1.gqa_stop.policy import policy_params
from milestone1.gqa_stop.schemas import validate_reader

CFG = json.loads(CONFIG_PATH.read_text())
LEADING_ZERO_ID = "05515938"
N = 20


@pytest.fixture()
def world(tmp_path: Path):
    config_path, cfg, questions = make_world(tmp_path)
    transport = FakeTransport(None, answers_by_text(questions))
    ctx = make_context(config_path, cfg, transport)
    transport.paths = ctx.paths
    stages_model.preflight(ctx, get=fake_tags_get)
    return ctx, transport, questions, tmp_path


def screened(world):
    ctx, transport, questions, tmp = world
    stages_data.prepare(ctx)
    stages_data.screen(ctx, "dev")
    return ctx, transport, questions, tmp


# ---- 1. leading-zero ids survive CSV round trips ----------------------------------
def test_read_id_csv_keeps_leading_zeros(tmp_path: Path) -> None:
    path = tmp_path / "stops.csv"
    pd.DataFrame(
        {"question_id": [LEADING_ZERO_ID], "image_id": ["0012"], "group_id": ["0012"]}
    ).to_csv(path, index=False)
    frame = read_id_csv(path)
    assert frame.loc[0, "question_id"] == LEADING_ZERO_ID
    assert frame.loc[0, "image_id"] == "0012"


def test_report_examples_use_string_ids(tmp_path: Path) -> None:
    other = "15515938"
    stops = pd.DataFrame(
        [
            {
                "method": m,
                "question_id": q,
                "stop_level": lvl,
                "correct": ok,
                "reader_answer": "x",
            }
            for q in (LEADING_ZERO_ID, other)
            for m, lvl, ok in (
                ("checklist_min@0.95", 3, True),
                ("direct@0.95", 9, True),
            )
        ]
    )
    table = pd.DataFrame(
        [
            {"question_id": q, "level": t, "reader_correct": t >= 3}
            for q in (LEADING_ZERO_ID, other)
            for t in range(1, N + 1)
        ]
    )
    stops.to_csv(tmp_path / "stops.csv", index=False)
    table.to_csv(tmp_path / "table.csv", index=False)
    loaded_stops, loaded_table = (
        read_id_csv(tmp_path / "stops.csv"),
        read_id_csv(tmp_path / "table.csv"),
    )
    cats = report.pick_examples(loaded_stops, loaded_table, "0.95", 20261008)
    expected = sorted([LEADING_ZERO_ID, other], key=lambda q: seeded_hash(20261008, q))
    assert cats["min_early_and_correct"] == expected


# ---- 2. reader prompt revision after screening; infeasible targets --------------------
def revise_reader_prompt(ctx) -> None:
    texts = {**ctx.prompts.texts, "answer_only": ctx.prompts.texts["answer_only"] + " "}
    ctx.__dict__["prompts"] = PromptSet(texts)


def test_reader_prompt_change_after_screening_is_refused(world) -> None:
    ctx, *_ = screened(world)
    assert (
        ctx.state.get("screening")["dev"]["reader_prompt_hash"]
        == ctx.prompts.hashes["answer_only"]
    )
    revise_reader_prompt(ctx)
    with pytest.raises(StageError, match="re-run `screen`"):
        stages_model.pilot(ctx, [1], calibration=False, use_fallback=False)
    with pytest.raises(StageError, match="re-run `screen`"):
        stages_model.infer(ctx, "dev", [1])
    ctx.state.set("infer", {"dev": {"complete": True}})
    with pytest.raises(StageError, match="re-run `screen`"):
        stages_model.freeze(ctx)


def eval_table() -> pd.DataFrame:
    rows = []
    for q in range(4):
        for t in range(1, N + 1):
            rows.append(
                {
                    "question_id": f"0{q}",
                    "image_id": f"{q}",
                    "group_id": f"{q}",
                    "level": t,
                    "reader_correct": t >= 3,
                    "reader_answer": "x",
                    "attribute": "color",
                    "relate_count": 0,
                    "direct_score": t / N,
                    "checklist_min": t / N,
                    "checklist_mean": t / N,
                    "reader_status": "valid",
                    "direct_status": "valid",
                    "direct_answer": "x",
                    "check_status": "valid",
                    "planner_status": "valid",
                    "n_items": 2,
                    "unresolved_items": 0,
                }
            )
    return pd.DataFrame(rows)


def test_infeasible_target_is_skipped_and_recorded(tmp_path: Path) -> None:
    frozen = {
        "primary_target": "0.95",
        "methods": ["direct", "checklist_min", "checklist_mean"],
        "selections": {
            "0.95": {
                "direct": {"theta": 0.5},
                "checklist_min": {"theta": 0.5},
                "checklist_mean": {"theta": "inf"},
                "fixed_dev": 3,
            },
            "0.99": {
                "direct": {"theta": None, "feasible": False},
                "checklist_min": {"theta": 0.6},
                "checklist_mean": {"theta": 0.7},
                "fixed_dev": None,
            },
        },
    }
    data = build_eval_data(eval_table(), frozen, N)
    assert "direct@0.99" in data.infeasible and "fixed_dev@0.99" in data.infeasible
    summary = evaluate(
        eval_table(),
        frozen,
        {
            "bootstrap_repeats": 5,
            "bootstrap_seed": 1,
            "accuracy_noninferiority_margin": 0.05,
        },
        N,
        tmp_path,
    )
    assert "direct@0.99" in summary["infeasible_targets"]
    ms = pd.read_csv(tmp_path / "method_summary.csv")
    assert ms.set_index("method").loc["direct@0.99", "status"] == "infeasible_on_dev"


# ---- 3. live digest verification and frozen match --------------------------------------
def test_engine_refuses_when_live_digest_changed(world) -> None:
    ctx, *_ = world

    def changed(url: str, timeout: float) -> Any:
        class R:
            def json(self) -> dict:
                return {"models": [{"name": "qwen3.5:9b", "digest": "ee" * 32}]}

        return R()

    ctx.tags_get = changed
    with pytest.raises(StageError, match="model changed"):
        ctx.engine()


def test_engine_refuses_when_ollama_unreachable(world) -> None:
    ctx, *_ = world

    def down(url: str, timeout: float) -> Any:
        raise requests.ConnectionError("down")

    ctx.tags_get = down
    with pytest.raises(StageError, match="unreachable"):
        ctx.engine()


def test_test_stages_require_frozen_digest_and_prompts(world) -> None:
    ctx, *_ = world
    base = {
        "model": ctx.settings.name,
        "digest": ctx.settings.digest,
        "prompt_hashes": ctx.prompts.hashes,
    }
    atomic_write_json(ctx.paths.out("frozen_policy.json"), base)
    ctx.require_frozen_match()  # consistent -> ok
    atomic_write_json(ctx.paths.out("frozen_policy.json"), {**base, "digest": "old"})
    with pytest.raises(StageError, match="frozen"):
        ctx.require_frozen_match()
    ctx.state.set("prepare", {"frozen": True})
    with pytest.raises(StageError, match="frozen"):
        stages_data.screen(ctx, "test")
    with pytest.raises(StageError, match="frozen"):
        stages_model.run_evaluate(ctx)
    atomic_write_json(
        ctx.paths.out("frozen_policy.json"),
        {**base, "prompt_hashes": {**base["prompt_hashes"], "direct": "x"}},
    )
    with pytest.raises(StageError, match="prompt hashes"):
        ctx.require_frozen_match()


def test_preflight_refuses_digest_change_after_freeze(world) -> None:
    ctx, *_ = world
    atomic_write_json(ctx.paths.out("frozen_policy.json"), {"digest": "old-digest"})
    with pytest.raises(StageError, match="after freeze"):
        stages_model.preflight(ctx, get=fake_tags_get)


# ---- 4. transport errors inside the retry path -----------------------------------------
SETTINGS = dataclasses.replace(settings_from_config(CFG), digest="abc")


class Response:
    def __init__(self, status: int, body: Any = None, bad_json: bool = False) -> None:
        self.status_code = status
        self._body = body
        self._bad = bad_json

    def json(self) -> Any:
        if self._bad:
            raise requests.exceptions.JSONDecodeError("bad", "doc", 0)
        return self._body


@pytest.mark.parametrize(
    "failure",
    [
        requests.exceptions.ChunkedEncodingError("cut"),
        requests.exceptions.ContentDecodingError("gzip"),
        Response(200, bad_json=True),
        Response(200, body=["not", "an", "object"]),
    ],
)
def test_transport_retries_broken_responses(failure: Any) -> None:
    replies = [failure, Response(200, {"message": {"content": "{}"}})]

    def post(*_a: Any, **_k: Any) -> Response:
        item = replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    result = OllamaTransport(SETTINGS, post=post, sleep=lambda _s: None).send({})
    assert result.content == "{}" and result.http_attempts == 2


def test_transport_gives_up_on_persistent_bad_json() -> None:
    transport = OllamaTransport(
        SETTINGS,
        post=lambda *a, **k: Response(200, bad_json=True),
        sleep=lambda _s: None,
    )
    with pytest.raises(TransportError):
        transport.send({})


# ---- 5. schema-repair prompt hash in the cache key --------------------------------------
class Scripted:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.n = 0

    def send(self, payload: dict) -> TransportResult:
        self.n += 1
        return TransportResult(self.replies.pop(0), 1, 0.0)


def test_repair_prompt_change_invalidates_cache(tmp_path: Path) -> None:
    prompts = load_prompts(PROMPTS_DIR)
    spec = build_level_spec(
        "answer_only",
        {"question_id": "q", "image_id": "1", "question": "What color?"},
        1,
        b"img",
    )
    RequestEngine(SETTINGS, prompts, Scripted(['{"answer": "red"}']), tmp_path).run(
        spec, validate_reader
    )
    same = Scripted([])
    RequestEngine(SETTINGS, prompts, same, tmp_path).run(spec, validate_reader)
    assert same.n == 0
    changed = PromptSet({**prompts.texts, "schema_repair": "different repair text"})
    fresh = Scripted(['{"answer": "blue"}'])
    RequestEngine(SETTINGS, changed, fresh, tmp_path).run(spec, validate_reader)
    assert fresh.n == 1


# ---- 6. prepare --revise --------------------------------------------------------------------
def test_revise_dry_run_shows_revised_scan_and_persists_nothing(world) -> None:
    ctx, _, questions, tmp = world
    stages_data.prepare(ctx)
    state_before = ctx.paths.out("state.json").read_text()
    eff_before = ctx.paths.out("effective_config.json").read_text()
    revision = tmp / "rev.json"
    revision.write_text(json.dumps({"dataset": {"candidate_dev_questions": 6}}))
    dry = make_context(ctx.config_path, {}, None, dry_run=True)
    result = stages_data.prepare(dry, revise=True, revision_file=revision, reason="r")
    dev = next(r for r in result["scans"] if r["split"] == "dev")
    assert dev["candidates"] <= 6  # the revised value was used for the scan
    assert ctx.paths.out("state.json").read_text() == state_before
    assert ctx.paths.out("effective_config.json").read_text() == eff_before


def test_revise_failure_persists_nothing(
    world, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx, _, _, tmp = world
    stages_data.prepare(ctx)
    eff_before = ctx.paths.out("effective_config.json").read_text()
    revision = tmp / "rev.json"
    revision.write_text(json.dumps({"dataset": {"candidate_dev_questions": 6}}))

    def boom(_ctx: Any) -> Any:
        raise RuntimeError("scan failed")

    monkeypatch.setattr(stages_data, "build_scan", boom)
    with pytest.raises(RuntimeError):
        stages_data.prepare(ctx, revise=True, revision_file=revision, reason="r")
    assert ctx.state.get("prepare")["revisions"] == 0
    assert ctx.paths.out("effective_config.json").read_text() == eff_before


# ---- 7. one generation per image, one manifest line ----------------------------------------
def test_all_levels_generated_once_with_one_manifest_line(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    Image.new("RGB", (64, 48), (1, 2, 3)).save(tmp_path / "src" / "5.jpg")
    manifest = tmp_path / "m.jsonl"
    series = ImageSeries(CFG["images"], tmp_path / "src", tmp_path / "lv", manifest)
    series.bytes_for("5", 7)
    series.bytes_for("5", 12)
    series.ensure("5", range(1, 21))
    lines = read_jsonl(manifest)
    assert len(lines) == 1 and len(lines[0]["levels"]) == 20
    tampered = series.path("5", 3)
    tampered.write_bytes(b"corrupt")
    data = series.bytes_for("5", 3)  # hash mismatch -> regenerated once
    assert data != b"corrupt"
    assert len(read_jsonl(manifest)) == 2
    reloaded = ImageSeries(CFG["images"], tmp_path / "src", tmp_path / "lv", manifest)
    assert reloaded.row("5", 3)["sha256"] == series.row("5", 3)["sha256"]


# ---- 8. Forced20 per §7.2, stop20_rate, stop_reason --------------------------------------
def test_forced20_includes_natural_pass_and_reasons_recorded(tmp_path: Path) -> None:
    table = eval_table()
    table.loc[table["question_id"] == "00", "direct_score"] = 0.0
    table.loc[
        (table["question_id"] == "01") & (table["level"] == N), "direct_score"
    ] = 1.0
    table.loc[(table["question_id"] == "01") & (table["level"] < N), "direct_score"] = (
        0.0
    )
    frozen = {
        "primary_target": "0.95",
        "methods": ["direct"],
        "selections": {"0.95": {"direct": {"theta": 0.9}, "fixed_dev": 20}},
    }
    evaluate(
        table,
        frozen,
        {
            "bootstrap_repeats": 5,
            "bootstrap_seed": 1,
            "accuracy_noninferiority_margin": 0.05,
        },
        N,
        tmp_path,
    )
    ms = pd.read_csv(tmp_path / "method_summary.csv").set_index("method")
    row = ms.loc["direct@0.95"]
    assert row["forced20_rate"] == 0.5  # q00 never met, q01 met only at L20
    assert row["stop20_rate"] == 0.5
    assert row["never_met_rate"] == 0.25 and row["natural_pass20_rate"] == 0.25
    stops = read_id_csv(tmp_path / "stops.csv")
    reasons = stops[stops["method"] == "direct@0.95"].set_index("question_id")[
        "stop_reason"
    ]
    assert reasons["00"] == "never_met_forced_last_level"
    assert reasons["01"] == "threshold_met_at_last_level"
    assert reasons["02"] == "threshold_met"


# ---- 9. pilot gates enforced in dev infer ------------------------------------------------------
def test_dev_infer_requires_pilot_calibration(world) -> None:
    ctx, *_ = screened(world)
    with pytest.raises(StageError, match="no pilot"):
        stages_model.infer(ctx, "dev", [1])


def write_gate_files(ctx, valid: float, passed: bool) -> None:
    tag = model_tag(ctx.settings.name, ctx.settings.digest)
    atomic_write_json(
        ctx.paths.out(f"pilot/pilot_summary_{tag}.json"),
        {
            "prompt_hashes": ctx.prompts.hashes,
            "valid_rates": {
                "reader": 1.0,
                "direct": valid,
                "checklist": 1.0,
                "planner": 1.0,
            },
        },
    )
    atomic_write_json(
        ctx.paths.out(f"pilot/pilot_calibration_{tag}.json"), {"gate_passed": passed}
    )


def test_dev_infer_enforces_format_and_calibration_gates(world) -> None:
    ctx, *_ = screened(world)
    ctx.engine()  # writes prompt_hashes.json (0 revisions)
    write_gate_files(ctx, valid=0.93, passed=True)
    with pytest.raises(StageError, match="repair the prompt"):
        stages_model.infer(ctx, "dev", [1])
    hashes = read_json(ctx.paths.out("prompt_hashes.json"))
    atomic_write_json(ctx.paths.out("prompt_hashes.json"), {**hashes, "revisions": 1})
    stages_model.enforce_pilot_gates(ctx, False)  # 0.93 after one revision: allowed
    write_gate_files(ctx, valid=0.85, passed=True)
    with pytest.raises(StageError, match="exit branch"):
        stages_model.infer(ctx, "dev", [1])
    write_gate_files(ctx, valid=1.0, passed=False)
    with pytest.raises(StageError, match="use-fallback-model"):
        stages_model.infer(ctx, "dev", [1])
    info = stages_model.infer(ctx, "dev", [1], force_continue=True)
    assert info["levels"] == [1]
    assert "GATE OVERRIDE" in ctx.paths.out("worklog.md").read_text()
    assert ctx.state.get("gate_override")["failures"]


# ---- 10. fallback records coexist with primary records -----------------------------------------
def test_fallback_pilot_does_not_overwrite_primary_records(world) -> None:
    ctx, *_ = screened(world)
    primary_tag = model_tag(ctx.settings.name, ctx.settings.digest)
    question = ctx.selected("dev")[0]
    primary_path = ctx.paths.out(
        f"reader/{primary_tag}/{question['question_id']}/20.json"
    )
    assert primary_path.exists()
    before = primary_path.read_text()
    stages_model.pilot(ctx, [20], calibration=True, use_fallback=True)
    fallback_tag = model_tag(ctx.settings.name, ctx.settings.digest)
    assert fallback_tag != primary_tag
    assert primary_path.read_text() == before
    assert ctx.paths.out(
        f"reader/{fallback_tag}/{question['question_id']}/20.json"
    ).exists()


# ---- 11. policy params from config -------------------------------------------------------------
@pytest.mark.parametrize(
    "override",
    [{"consecutive_k": 2}, {"start_level": 2}, {"force_stop_level": 19}],
)
def test_unsupported_policy_params_raise(override: dict) -> None:
    with pytest.raises(ValueError):
        policy_params({**CFG["policy"], **override}, N)
    params = policy_params(CFG["policy"], N)
    assert (params.start_level, params.consecutive_k, params.max_level) == (1, 1, 20)


# ---- 12. provenance note when probe records are reused ------------------------------------------
def test_probe_record_reuse_is_logged(world) -> None:
    ctx, transport, _, _ = world
    question = stages_data.probe_questions(ctx)[0].public()
    ctx.set_stage("probe")
    ctx.runner().role_record("answer_only", question, 20)
    ctx.set_stage("screen_dev")
    calls = transport.calls["answer_only"]
    ctx.runner().role_record("answer_only", question, 20)
    assert transport.calls["answer_only"] == calls  # reused, not re-run
    ctx.note_reuse()
    assert "probe records" in ctx.paths.out("worklog.md").read_text()
