"""CLI stages on the synthetic world: probe, prepare freeze/revise, dry-runs,
and an end-to-end pipeline (screen -> pilot -> infer -> freeze -> evaluate ->
report) with a fake ollama. Covers test 10 (common reader, L20 screening
reuse) and the protocol guards."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from gqa_stop_helpers import (
    FakeTransport,
    answers_by_text,
    fake_tags_get,
    make_context,
    make_world,
)

from milestone1.gqa_stop import cli, report, stages_data, stages_model
from milestone1.gqa_stop.common import StageError, read_jsonl
from milestone1.gqa_stop.policy import SplitTuningError


def new_ctx(config_path: Path, cfg: dict, questions: dict, dry_run: bool = False):
    transport = FakeTransport(None, answers_by_text(questions))
    ctx = make_context(config_path, cfg, transport, dry_run=dry_run)
    transport.paths = ctx.paths
    return ctx, transport


@pytest.fixture()
def world(tmp_path: Path):
    config_path, cfg, questions = make_world(tmp_path)
    ctx, transport = new_ctx(config_path, cfg, questions)
    stages_model.preflight(ctx, get=fake_tags_get)
    return config_path, cfg, questions, ctx, transport


# ---- dry runs ---------------------------------------------------------------------
def test_dry_runs_make_no_calls_and_no_files(tmp_path: Path) -> None:
    config_path, cfg, questions = make_world(tmp_path)
    for command in ("probe", "prepare"):
        ctx, transport = new_ctx(config_path, cfg, questions, dry_run=True)
        if command == "probe":
            plan = stages_data.probe(ctx)
            assert plan["reader_requests"] == plan["questions"] * 4
        else:
            stages_data.prepare(ctx)
        assert sum(transport.calls.values()) == 0
    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "data").exists()


def test_cli_main_dry_run_exit_codes(tmp_path: Path) -> None:
    config_path, _, _ = make_world(tmp_path)
    base = ["--config", str(config_path)]
    assert cli.main(["prepare", "--dry-run", *base]) == 0
    assert cli.main(["probe", "--dry-run", *base]) == 0
    assert cli.main(["screen", "--dry-run", *base]) == 1  # missing --split


# ---- probe (§16 A3) ----------------------------------------------------------------------
def test_probe_outputs_and_ordering(world) -> None:
    _, _, _, ctx, transport = world
    summary = stages_data.probe(ctx)
    assert transport.calls["answer_only"] == len(stages_data.probe_questions(ctx)) * 4
    assert set(transport.calls) == {"answer_only"}
    data = json.loads(ctx.paths.out("probe_summary.json").read_text())
    assert set(data["levels"]) == {1, 5, 10, 20}
    assert set(summary["cell_level_accuracy"]) <= set(ctx.strata.cells)
    assert ctx.paths.out("probe/probe_cell_level_accuracy.csv").exists()
    manifest = read_jsonl(ctx.paths.data("image_manifest.jsonl"))
    images = {q.image_id for q in stages_data.probe_questions(ctx)}
    assert sorted(e["image_id"] for e in manifest) == sorted(images)  # 1 line/image
    assert all(len(e["levels"]) == 20 for e in manifest)  # all levels at once


def test_probe_refused_after_prepare(world) -> None:
    *_, ctx, _ = world
    stages_data.prepare(ctx)
    with pytest.raises(StageError):
        stages_data.probe(ctx)


# ---- prepare freeze / revise ------------------------------------------------------------
def test_prepare_freezes_and_single_revision(world, tmp_path: Path) -> None:
    *_, ctx, _ = world
    stages_data.prepare(ctx)
    for name in ("schema_mapping.json", "dataset_flow.csv", "scan_summary.json"):
        assert ctx.paths.out(name).exists()
    assert read_jsonl(ctx.paths.data("candidate_manifest.jsonl"))
    assert "A2-bis" in ctx.paths.out("worklog.md").read_text()
    flow = pd.read_csv(ctx.paths.out("dataset_flow.csv"))
    metrics = set(flow[flow["stage"] == "candidates"]["metric"])
    assert {"cell_size", "share_quota", "quota", "filled", "images_scanned"} <= metrics
    assert flow["note"].astype(str).str.contains("A2-bis").any()
    with pytest.raises(StageError):
        stages_data.prepare(ctx)
    revision = tmp_path / "rev.json"
    revision.write_text(json.dumps({"dataset": {"candidate_dev_questions": 18}}))
    stages_data.prepare(
        ctx, revise=True, revision_file=revision, reason="probe: L1 high"
    )
    assert ctx.cfg["dataset"]["candidate_dev_questions"] == 18
    assert "dataset revision 1" in ctx.paths.out("worklog.md").read_text()
    flow = pd.read_csv(ctx.paths.out("dataset_flow.csv"))
    assert (flow["stage"] == "revision").any()
    with pytest.raises(StageError):
        stages_data.prepare(ctx, revise=True, revision_file=revision, reason="again")


def test_revision_must_only_touch_dataset(world, tmp_path: Path) -> None:
    *_, ctx, _ = world
    stages_data.prepare(ctx)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"policy": {"primary_dev_accuracy_target": 0.5}}))
    with pytest.raises(StageError):
        stages_data.prepare(ctx, revise=True, revision_file=bad, reason="x")


# ---- end to end -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory):
    """Runs the whole protocol once; tests below only read or re-run idempotently."""
    config_path, cfg, questions = make_world(tmp_path_factory.mktemp("pipeline"))
    ctx, transport = new_ctx(config_path, cfg, questions)
    stages_model.preflight(ctx, get=fake_tags_get)
    stages_data.prepare(ctx)
    with pytest.raises(StageError):
        stages_data.screen(ctx, "test")  # test screening needs a frozen policy
    stages_data.screen(ctx, "dev")
    reader_l20_after_screen = count_reader_l20(transport)
    stages_model.pilot(ctx, [1, 5, 10, 15, 20], calibration=True, use_fallback=False)
    stages_model.infer(ctx, "dev", list(range(1, 21)))
    stages_model.freeze(ctx)
    stages_data.screen(ctx, "test")
    stages_model.infer(ctx, "test", list(range(1, 21)))
    stages_model.run_evaluate(ctx)
    report.build_report(ctx.cfg, ctx.paths)
    return ctx, transport, reader_l20_after_screen


def count_reader_l20(transport: FakeTransport) -> int:
    return sum(
        1
        for p in transport.payloads
        if p["messages"][0]["content"].startswith("You answer")
        and transport.level_of(p) == 20
        and len(p["messages"]) == 1
    )


def test_pipeline_writes_all_artifacts(pipeline) -> None:
    ctx, _, _ = pipeline
    for name in (
        "preflight.json",
        "effective_config.json",
        "schema_mapping.json",
        "dataset_flow.csv",
        "prompt_hashes.json",
        "dev_threshold_grid.csv",
        "frozen_policy.json",
        "stops.csv",
        "method_summary.csv",
        "paired_comparisons.csv",
        "bootstrap_summary.json",
        "report.md",
        "completion.json",
        "pilot_calibration.json",
        "screening_repeat.json",
        "requests.jsonl",
        "state.json",
        "worklog.md",
        "figures/test_meanlevel_accuracy.png",
    ):
        assert ctx.paths.out(name).exists(), name
    for name in (
        "candidate_manifest.jsonl",
        "selection_results.jsonl",
        "selected_manifest.jsonl",
        "image_manifest.jsonl",
    ):
        assert ctx.paths.data(name).exists(), name
    completion = json.loads(ctx.paths.out("completion.json").read_text())
    assert completion["stages"]["evaluate"]["status"] == "completed"
    assert completion["stages"]["probe"]["status"] == "not_run"
    assert completion["complete"] is False  # probe skipped here -> honestly incomplete


def test_screening_l20_is_reused_as_reader(pipeline) -> None:
    ctx, transport, after_screen = pipeline
    dev_ids = {r["question_id"] for r in ctx.selected("dev")}
    results = {
        r["question_id"]: r
        for r in read_jsonl(ctx.paths.data("selection_results.jsonl"))
    }
    table = pd.read_csv(
        ctx.paths.out("level_table_dev.csv"), dtype={"question_id": str}
    )
    l20 = table[table["level"] == 20].set_index("question_id")
    for qid in dev_ids:
        assert l20.loc[qid, "reader_answer"] == results[qid]["reader_answer"]
        assert results[qid]["outcome"] == "correct"
    test_candidates = len(ctx.candidates("test"))
    assert count_reader_l20(transport) == after_screen + test_candidates


def test_reader_answer_common_to_all_methods(pipeline) -> None:
    ctx, _, _ = pipeline
    stops = pd.read_csv(ctx.paths.out("stops.csv"), dtype={"question_id": str})
    table = pd.read_csv(
        ctx.paths.out("level_table_test.csv"), dtype={"question_id": str}
    )
    answers = {(r.question_id, r.level): r.reader_answer for r in table.itertuples()}
    for row in stops.itertuples():
        assert answers[(row.question_id, row.stop_level)] == row.reader_answer


def test_selected_sets_disjoint_and_adopted_correct(pipeline) -> None:
    ctx, _, _ = pipeline
    dev = {r["group_id"] for r in ctx.selected("dev")}
    test = {r["group_id"] for r in ctx.selected("test")}
    assert dev and test and not dev & test


def test_guards_after_freeze(pipeline) -> None:
    ctx, _, _ = pipeline
    with pytest.raises(StageError):
        stages_data.screen(ctx, "dev")
    with pytest.raises(StageError):
        stages_model.switch_to_fallback(ctx)  # dev infer already started
    with pytest.raises(SplitTuningError):
        stages_model.freeze(ctx)


def test_fenced_checklist_valid_on_first_attempt(pipeline) -> None:
    ctx, _, _ = pipeline
    summary = json.loads(ctx.paths.out("pilot_summary.json").read_text())
    assert summary["valid_rate_initial"]["checklist"] == 1.0
    assert summary["repair_rate"]["checklist"] == 0.0
    assert summary["parse_version"] == "parse-v2"
    checks = list(ctx.paths.out("checks").rglob("*.json"))
    assert checks and all(json.loads(p.read_text())["fence_stripped"] for p in checks)


def test_resume_makes_no_new_calls(pipeline) -> None:
    ctx, transport, _ = pipeline
    before = sum(transport.calls.values())
    stages_model.infer(ctx, "test", list(range(1, 21)))
    assert sum(transport.calls.values()) == before


def test_infer_refuses_when_screening_model_differs(world) -> None:
    *_, ctx, _ = world
    stages_data.prepare(ctx)
    stages_data.screen(ctx, "dev")
    stages_model.switch_to_fallback(ctx)
    assert ctx.settings.name == "qwen3.5:4b"
    with pytest.raises(StageError):
        stages_model.infer(ctx, "dev", [1])
    effective = json.loads(ctx.paths.out("effective_config.json").read_text())
    assert effective["overrides"]["model"]["name"] == "qwen3.5:4b"
