"""v22 test protocol: v1-arm opt-out, separate reader-wrong inference, the
descriptive θ sweep (sweep-v22) and the record audit."""

from __future__ import annotations

import math
import shutil

import pandas as pd
import pytest
from test_gqa_stop_v4c import v4c_run  # noqa: F401  (module-scoped fixture)

from milestone1.gqa_stop import audit_v22, stages_model, stages_v22_test, sweep_v22
from milestone1.gqa_stop.common import (
    StageError,
    atomic_write_json,
    read_json,
    sha256_file,
)
from milestone1.gqa_stop.devgrid_v22 import ALL_METHODS
from milestone1.gqa_stop.firing_v23 import wrong_questions

LEVELS = list(range(1, 21))
FROZEN_THETA = 0.5


def frozen_policy(ctx) -> dict:
    entries = {
        name: {"column": column, "k": k, "role": role, "feasible": True}
        for name, column, k, role in ALL_METHODS
    }
    selections = {
        t: {
            **{m: {**e, "theta": FROZEN_THETA} for m, e in entries.items()},
            "fixed_dev": 20,
        }
        for t in ("0.90", "0.95", "0.97", "0.99")
    }
    return {
        "model": ctx.settings.name,
        "digest": ctx.settings.digest,
        "prompt_hashes": ctx.prompts.hashes,
        "primary_target": "0.95",
        "methods": [m[0] for m in ALL_METHODS],
        "main_methods": list(stages_v22_test.MAIN),
        "selections": selections,
    }


@pytest.fixture(scope="module")
def v22_run(v4c_run):  # noqa: F811
    ctx = v4c_run["ctx"]
    path = ctx.paths.out(stages_v22_test.FROZEN_FILE)
    atomic_write_json(path, frozen_policy(ctx))
    ctx.state.set("freeze", {"done": True, "sha256": sha256_file(path)})
    ctx.skip_v1arm = True
    stages_model.infer(ctx, "test", LEVELS)
    ctx.skip_v1arm = False
    with pytest.raises(StageError, match="skip-v1arm"):
        stages_v22_test.evaluate_v22(ctx)
    with pytest.raises(StageError, match="evaluate-v22"):
        sweep_v22.sweep_v22(ctx)
    ctx.skip_v1arm = True
    result = stages_v22_test.evaluate_v22(ctx)
    with pytest.raises(StageError, match="infer-test-wrong"):
        stages_v22_test.firing_test(ctx)
    stages_v22_test.infer_test_wrong(ctx)
    firing = stages_v22_test.firing_test(ctx)
    out = ctx.paths.output_root
    shutil.copy(out / "level_table_test_merged.csv", out / "level_table_dev_merged.csv")
    (out / "ext_dev_wrong").mkdir(exist_ok=True)
    shutil.copy(
        out / "ext_test_wrong/level_table.csv", out / "ext_dev_wrong/level_table.csv"
    )
    sweep = sweep_v22.sweep_v22(ctx)
    return dict(ctx=ctx, result=result, firing=firing, sweep=sweep)


# ---- v1 opt-out ------------------------------------------------------------------------------
def test_without_v1_drops_v1_methods_and_keeps_input() -> None:
    frozen = {
        "methods": ["direct_v4", "checklist_v1_min", "direct_v1_verbalized"],
        "main_methods": ["direct_v4", "checklist_v1_min"],
        "selections": {
            "0.95": {"direct_v4": {}, "checklist_v1_mean": {}, "fixed_dev": 20}
        },
    }
    out = stages_v22_test.without_v1(frozen)
    assert out["methods"] == ["direct_v4"] and out["main_methods"] == ["direct_v4"]
    assert set(out["selections"]["0.95"]) == {"direct_v4", "fixed_dev"}
    assert "checklist_v1_min" in frozen["methods"]  # input not mutated


def test_skip_v1arm_makes_no_verbalised_direct_call_on_test(v22_run) -> None:
    ctx = v22_run["ctx"]
    test_ids = {r["question_id"] for r in ctx.selected("test")}
    test_ids |= {r["question_id"] for r in wrong_questions(ctx, "test")}
    for root in (ctx.paths.output_root, ctx.paths.out("ext_test_wrong")):
        direct = root / "direct"
        made = {p.name for p in direct.glob("*/*/*")} if direct.exists() else set()
        assert not made & test_ids
    table = pd.read_csv(ctx.paths.out("level_table_test_merged.csv"))
    assert table["checklist_v1_min"].isna().all() and table["direct_score"].isna().all()
    assert ctx.state.get("infer")["test"]["skip_v1arm"].startswith("v1 arm skipped")


def test_evaluate_records_skip_and_drops_v1_methods(v22_run) -> None:
    ctx, result = v22_run["ctx"], v22_run["result"]
    assert result["v1_arm"] == stages_v22_test.V1_SKIP_NOTE
    assert result["skipped"] == list(stages_v22_test.V1_SKIPPED)
    assert not set(result["methods_evaluated"]) & set(stages_v22_test.V1_METHODS)
    stops = pd.read_csv(ctx.paths.out("test_stops.csv"), dtype={"target": str})
    assert not set(stops["method"]) & set(stages_v22_test.V1_METHODS)
    assert {"direct_v4", "checklist_v4_min", "direct_pans", "pans_min"} <= set(
        stops["method"]
    )
    frozen = read_json(ctx.paths.out(stages_v22_test.FROZEN_FILE))
    assert "checklist_v1_min" in frozen["methods"]  # frozen file untouched
    with pytest.raises(StageError, match="already run once"):
        stages_v22_test.evaluate_v22(ctx)


def test_firing_test_is_cache_only_and_records_skip(v22_run) -> None:
    ctx, firing = v22_run["ctx"], v22_run["firing"]
    assert firing["v1_arm"] == stages_v22_test.V1_SKIP_NOTE
    assert any("verbalised Direct" in s for s in firing["skipped"])
    out = ctx.paths.out(stages_v22_test.EXT_DIR)
    summary = pd.read_csv(out / "firing_summary.csv")
    assert not set(summary["method"]) & set(stages_v22_test.V1_METHODS)
    assert (summary["firing_ci_low"] <= summary["firing_rate"]).all()
    assert (summary["firing_rate"] <= summary["firing_ci_high"]).all()
    assert firing["n_questions"] == len(wrong_questions(ctx, "test")) > 0
    with pytest.raises(StageError, match="already run once"):
        stages_v22_test.firing_test(ctx)


# ---- sweep -------------------------------------------------------------------------------------
def test_sweep_thetas_are_050_to_100_in_005_steps() -> None:
    assert sweep_v22.SWEEP_THETAS[0] == 0.5 and sweep_v22.SWEEP_THETAS[-1] == 1.0
    assert len(sweep_v22.SWEEP_THETAS) == 11
    assert all(
        math.isclose(b - a, 0.05)
        for a, b in zip(
            sweep_v22.SWEEP_THETAS, sweep_v22.SWEEP_THETAS[1:], strict=False
        )
    )


def hand_table(
    scores: dict[str, list[float]], correct_from: dict[str, int]
) -> pd.DataFrame:
    rows = []
    for q, values in scores.items():
        for level, s in enumerate(values, start=1):
            rows.append(
                {
                    "question_id": q,
                    "group_id": f"g{q}",
                    "level": level,
                    "reader_correct": level >= correct_from[q],
                    "direct_v4": s,
                }
            )
    return pd.DataFrame(rows)


def test_sweep_frame_by_hand() -> None:
    answerable = hand_table(
        {
            "a": [0.6, 0.6, 0.6, 0.6],
            "b": [0.1, 0.7, 0.9, 0.9],
            "c": [0.1, 0.1, 0.1, 0.2],
        },
        {"a": 2, "b": 1, "c": 4},
    )
    wrong = hand_table(
        {
            "x": [0.1, 0.1, 0.1, 0.3],  # never meets θ: forced last level, no firing
            "y": [0.1, 0.8, 0.1, 0.1],  # fires at L2 for both θ
            "w": [0.1, 0.1, 0.1, 0.6],  # meets θ=0.5 at the last level: a firing
        },
        {"x": 9, "y": 9, "w": 9},
    )
    frame = sweep_v22.sweep_frame(
        answerable, wrong, "test", 4, methods=("direct_v4",), thetas=(0.5, 0.75)
    ).set_index("theta")
    low = frame.loc[0.5]  # stops a@1 (wrong), b@2, c@4 (forced)
    assert low["accuracy"] == pytest.approx(2 / 3)
    assert low["mean_level"] == pytest.approx(7 / 3)
    assert (low["q1_level"], low["median_level"], low["q3_level"]) == (1.5, 2.0, 3.0)
    assert low["stop20_rate"] == pytest.approx(1 / 3) and low["stopped_but_wrong"] == 1
    assert low["firing_rate"] == pytest.approx(2 / 3)
    high = frame.loc[0.75]  # a@4, b@3, c@4
    assert high["accuracy"] == 1.0 and high["mean_level"] == pytest.approx(11 / 3)
    assert high["firing_rate"] == pytest.approx(1 / 3) and high["n_wrong_set"] == 3


def test_sweep_outputs_are_labelled_descriptive(v22_run) -> None:
    ctx, meta = v22_run["ctx"], v22_run["sweep"]
    assert meta["label"].startswith("DESCRIPTIVE")
    out = ctx.paths.out(sweep_v22.OUT_DIR)
    for split in ("test", "dev"):
        frame = pd.read_csv(out / f"{split}_sweep.csv")
        assert len(frame) == len(sweep_v22.SWEEP_METHODS) * len(sweep_v22.SWEEP_THETAS)
        assert (frame["q1_level"] <= frame["median_level"]).all()
        assert (frame["median_level"] <= frame["q3_level"]).all()
        assert frame["firing_rate"].between(0, 1).all()
    for metric, _title in sweep_v22.AXES:
        assert (out / "figures" / f"sweep_{metric}.png").exists()


# ---- audit -------------------------------------------------------------------------------------
def lp(token: str, best: str, p: float) -> dict:
    return {
        "token": token,
        "logprob": math.log(p if token == best else 1 - p),
        "top_logprobs": [
            {"token": best, "logprob": math.log(p)},
            {"token": "z", "logprob": math.log(1 - p)},
        ],
    }


def test_audit_record_flags_slow_calls_and_argmax_violations() -> None:
    fast = {"attempts": [{"usage": {"total_duration": 2e9, "eval_duration": 1e9}}]}
    slow = {"attempts": [{"usage": {"total_duration": 9e9, "eval_duration": 1e9}}]}
    assert not audit_v22.audit_record(fast)["slow"]
    assert audit_v22.audit_record(slow)["slow"]
    assert audit_v22.audit_record(slow)["non_generation_s"] == pytest.approx(8.0)
    ok = {**fast, "logprobs": [lp("a", "a", 0.9)]}
    bad = {**fast, "logprobs": [lp("z", "a", 0.9)]}
    assert audit_v22.audit_record(ok)["argmax_violations"] == 0
    assert audit_v22.audit_record(bad)["argmax_violations"] == 1


def test_audit_runs_over_test_records(v22_run) -> None:
    summary = audit_v22.audit(v22_run["ctx"])
    assert summary["records"] > 0 and summary["counts"]["argmax_violations"] == 0
