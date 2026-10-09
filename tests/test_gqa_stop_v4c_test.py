"""Plan §21: v4c freeze on dev, one-shot frozen-θ test evaluation, §21.2 extension."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from test_gqa_stop_v4c import v4c_run  # noqa: F401  (module-scoped fixture)

from milestone1.gqa_stop import stages_model, stages_v4c_test
from milestone1.gqa_stop.common import StageError, read_json, read_jsonl
from milestone1.gqa_stop.policy import ConsistencyVariant, replay_stop_consecutive

TARGETS = {"0.95", "0.90", "0.97", "0.99"}


@pytest.fixture(scope="module")
def v4c_test_run(v4c_run):  # noqa: F811
    ctx = v4c_run["ctx"]
    with pytest.raises(StageError, match="freeze before"):
        stages_v4c_test.v1arm_test(ctx)
    frozen = stages_v4c_test.freeze_v4c(ctx)
    stages_v4c_test.v1arm_test(ctx)
    with pytest.raises(StageError, match="test inference must be complete"):
        stages_v4c_test.evaluate_v4c(ctx)
    stages_model.infer(ctx, "test", list(range(1, 21)))
    result = stages_v4c_test.evaluate_v4c(ctx)
    ext = stages_v4c_test.extension(ctx)
    return dict(ctx=ctx, frozen=frozen, result=result, ext=ext)


def test_freeze_writes_all_targets_methods_and_provenance(v4c_test_run) -> None:
    ctx, frozen = v4c_test_run["ctx"], v4c_test_run["frozen"]
    assert set(frozen["selections"]) == TARGETS and frozen["primary_target"] == "0.95"
    primary = frozen["selections"]["0.95"]
    assert set(stages_v4c_test.MAIN) <= set(primary) and "fixed_dev" in primary
    assert primary["checklist_v4_min_k2"]["k"] == 2
    assert primary["pans_min"]["role"] == "logged"
    assert {"model", "digest", "prompt_hashes"} <= set(frozen)
    assert all(
        q in {r["question_id"] for r in ctx.selected("dev")}
        for q in frozen["dev_question_ids"]
    )
    with pytest.raises(StageError, match="frozen once"):
        stages_v4c_test.freeze_v4c(ctx)


def test_evaluation_uses_frozen_theta_on_test_only(v4c_test_run) -> None:
    ctx, frozen = v4c_test_run["ctx"], v4c_test_run["frozen"]
    table = pd.read_csv(
        ctx.paths.out("level_table_test.csv"), dtype={"question_id": str}
    )
    assert set(table["split"]) == {"test"}
    assert set(table["question_id"]) == {r["question_id"] for r in ctx.selected("test")}
    stops = pd.read_csv(
        ctx.paths.out("stops.csv"), dtype={"question_id": str, "target": str}
    )
    entry = frozen["selections"]["0.95"]["checklist_v4_min"]
    theta = math.inf if entry["theta"] == "inf" else float(entry["theta"])
    variant = ConsistencyVariant(theta=theta, k=1)
    expected = {
        q: replay_stop_consecutive(
            list(g.sort_values("level")["checklist_v4_min"]), variant
        ).level
        for q, g in table.groupby("question_id")
    }
    got = stops[(stops["target"] == "0.95") & (stops["method"] == "checklist_v4_min")]
    assert dict(zip(got["question_id"], got["stop_level"], strict=True)) == expected


def test_evaluation_outputs_and_bootstrap(v4c_test_run) -> None:
    ctx, result = v4c_test_run["ctx"], v4c_test_run["result"]
    summary = pd.read_csv(ctx.paths.out("method_summary.csv"))
    assert {
        "accuracy",
        "mean_level",
        "median_level",
        "stop20_rate",
        "accuracy_minus_hull",
    } <= set(summary.columns)
    comps = pd.read_csv(ctx.paths.out("paired_comparisons.csv"))
    assert "checklist_v4_min-direct_v4" in set(comps["comparison"])
    assert (comps["ci_low"] <= comps["ci_high"]).all()
    assert result["bootstrap"]["unit"] == "image group"
    for name in (
        "test_stop_hist.csv",
        "test_attribute_breakdown.csv",
        "figures/test_meanlevel_accuracy.png",
        "figures/test_stop_hist.png",
    ):
        assert ctx.paths.out(name).exists()


def test_extension_runs_reader_wrong_candidates_in_own_namespace(v4c_test_run) -> None:
    ctx, ext = v4c_test_run["ctx"], v4c_test_run["ext"]
    wrong = {
        r["question_id"]
        for r in read_jsonl(ctx.paths.selection("selection_results.jsonl"))
        if r["split"] == "test" and r["outcome"] == "incorrect"
    }
    assert ext["n_questions"] == len(wrong) > 0
    out = ctx.paths.out(stages_v4c_test.EXT_DIR)
    table = pd.read_csv(out / "level_table.csv", dtype={"question_id": str})
    assert set(table["question_id"]) == wrong and len(table) == 20 * len(wrong)
    rows = {r["method"]: r for r in ext["methods"]}
    assert 0.0 <= rows["checklist_v4_min"]["false_stop_rate"] <= 1.0
    assert "checklist_v4_min-direct_v4.false_stop_diff" in ext["paired"]
    assert read_json(out / "ext_summary.json")["primary_target"] == "0.95"


def test_evaluate_extension_false_stop_by_hand(tmp_path) -> None:
    table = pd.DataFrame(
        {
            "question_id": ["a"] * 2,
            "level": [1, 2],
            "group_id": ["g"] * 2,
            "reader_correct": [False, False],
            "s": [0.2, 0.9],
        }
    )
    frozen = {
        "primary_target": "0.95",
        "methods": ["m"],
        "selections": {
            "0.95": {
                "m": {
                    "column": "s",
                    "k": 1,
                    "role": "main",
                    "theta": 0.5,
                    "feasible": True,
                }
            }
        },
    }
    res = stages_v4c_test.evaluate_extension(
        table, frozen, {"bootstrap_repeats": 10, "bootstrap_seed": 1}, 2, tmp_path
    )
    assert res["methods"][0]["false_stop_rate"] == 0.0  # stop at L2 == last level
    frozen["selections"]["0.95"]["m"]["theta"] = 0.1
    res = stages_v4c_test.evaluate_extension(
        table, frozen, {"bootstrap_repeats": 10, "bootstrap_seed": 1}, 2, tmp_path
    )
    assert res["methods"][0]["false_stop_rate"] == 1.0
