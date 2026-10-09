"""A13 (plan §20): explicit decoding block on every request, determinism gate,
clean re-run stages (test screening without freeze, reader levels, own textprobe
adoption, v1 arm) and the v4 devgrid with merged v1-arm comparators."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from gqa_stop_helpers import (
    PROMPTS_DIR,
    FakeTransport,
    answers_by_text,
    fake_tags_get,
    make_context,
    run_v1_pipeline,
)

from milestone1.gqa_stop import (
    stages_data,
    stages_model,
    stages_v4,
    stages_v4c,
    textprobe,
)
from milestone1.gqa_stop.client import (
    RequestEngine,
    TransportResult,
    load_prompts,
    settings_from_config,
)
from milestone1.gqa_stop.common import StageError, read_json
from milestone1.gqa_stop.inference import build_level_spec
from milestone1.gqa_stop.item_checklist import item_spec
from milestone1.gqa_stop.logprob_v4 import (
    direct_lp_spec,
    observer_lp_spec,
    relevance_spec,
)
from milestone1.gqa_stop.planner import planner_spec
from milestone1.gqa_stop.stages_v4c import argmax_violations
from milestone1.gqa_stop.textprobe import text_only_spec

CFG = json.loads((PROMPTS_DIR.parents[1] / "configs" / "gqa_stop_v4c.json").read_text())
ROLES = CFG["prompts"]["roles"]
CLEAN = CFG["model"]["decoding_options"]
Q = {"question_id": "q", "image_id": "1", "question": "What color is the cup?"}
ITEM = {"id": "c1", "question": "Is there a cup in the image?"}


def engine(tmp: Path, cfg: dict = CFG, transport: Any = None) -> RequestEngine:
    return RequestEngine(
        settings_from_config(cfg), load_prompts(PROMPTS_DIR, ROLES), transport, tmp
    )


def all_specs():
    return {
        "reader": build_level_spec("answer_only", Q, 3, b"img"),
        "direct": build_level_spec("direct", Q, 3, b"img"),
        "checklist": build_level_spec(
            "checklist", Q, 3, b"img", [{"id": "c1", "question": "a", "depends_on": []}]
        ),
        "planner": planner_spec(Q),
        "text_only": text_only_spec(Q),
        "observer_v3": item_spec(Q, 3, b"img", ITEM),
        "observer_lp": observer_lp_spec(Q, 3, b"img", ITEM),
        "direct_lp": direct_lp_spec(Q, 3, b"img"),
        "relevance": relevance_spec("q", 3, b"img", ITEM["question"], "c1"),
    }


def test_config_decoding_block_is_clean() -> None:
    assert CLEAN == {
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repeat_penalty": 1.0,
        "mirostat": 0,
    }
    assert CFG["model"]["think"] is False


@pytest.mark.parametrize("name", list(all_specs()))
def test_every_request_carries_the_decoding_block(tmp_path: Path, name: str) -> None:
    eng = engine(tmp_path)
    spec = all_specs()[name]
    payload = eng.build_payload(spec, eng.build_messages(spec))
    for key, value in CLEAN.items():
        assert payload["options"][key] == value, (name, key)
    assert payload["options"]["seed"] == 20261008 and payload["think"] is False
    assert "num_ctx" in payload["options"] and "num_predict" in payload["options"]


def test_decoding_block_changes_every_cache_key(tmp_path: Path) -> None:
    legacy = copy.deepcopy(CFG)
    legacy["model"].pop("decoding_options")
    clean, old = engine(tmp_path), engine(tmp_path, legacy)
    for name, spec in all_specs().items():
        assert clean.cache_key(spec) != old.cache_key(spec), name
    tweaked = copy.deepcopy(CFG)
    tweaked["model"]["decoding_options"]["presence_penalty"] = 1.5
    assert engine(tmp_path, tweaked).cache_key(
        all_specs()["reader"]
    ) != clean.cache_key(all_specs()["reader"])


def test_argmax_violations_by_hand() -> None:
    def entry(token: str, lp: float, alts: dict[str, float]) -> dict:
        return {
            "token": token,
            "logprob": lp,
            "top_logprobs": [{"token": t, "logprob": v} for t, v in alts.items()],
        }

    good = [
        entry("yes", -0.1, {"yes": -0.1, "no": -2.4}),
        entry("a", -0.7, {"a": -0.7, "b": -0.7}),
    ]  # tie at the top is fine
    bad = [entry("no", -2.4, {"yes": -0.1, "no": -2.4})]
    assert argmax_violations(good) == (2, 0)
    assert argmax_violations(bad) == (1, 1)
    assert argmax_violations(None) == (0, 0)


# ---- synthetic v1 -> v4c run ------------------------------------------------------------------
def v4c_config(v1_cfg: dict) -> dict:
    cfg = copy.deepcopy(v1_cfg)
    cfg["paths"].update(
        {
            "output_root": "outputs/run_v4c",
            "selection_root": "data/run_v4c",
            "inherit_split_from": "outputs/run",
            "subset_file": "outputs/run_v4c/visual_subset.json",
            "check_reference_adoption": False,
        }
    )
    cfg["model"]["decoding_options"] = dict(CLEAN)
    cfg["model"]["num_predict"].update(
        {"text_only": 128, "planner_v3": 1024, "observer_v3": 256, "relevance_v4": 8}
    )
    cfg["screening"] = {"test_without_freeze": True}
    cfg["determinism"] = {
        "argmax_questions": 50,
        "argmax_levels": [1, 20],
        "relevance_questions": 10,
    }
    cfg["prompts"] = {"roles": ROLES}
    cfg["v1_arm"] = CFG["v1_arm"]
    cfg["policy"].update({"family": "logprob_item", "methods": ["checklist_v4_min"]})
    cfg["comparison"] = {
        "v1_output_root": "outputs/run_v4c/v1arm",
        "v1_method": "checklist_min",
        "v1_fair_frozen": "outputs/run_v4c/v1arm/frozen_policy_subset.json",
        "v3_output_root": "outputs/no_v3_here",
        "merge_tables": [
            {
                "path": "outputs/run_v4c/v1arm/level_table_dev.csv",
                "columns": {"checklist_min": "checklist_v1_min"},
            }
        ],
        "extra_methods": {"checklist_v1_min": "checklist_v1_min"},
    }
    return cfg


@pytest.fixture(scope="module")
def v4c_run(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("v4c")
    _, cfg, questions, v1 = run_v1_pipeline(tmp)
    (tmp / "v4c.json").write_text(json.dumps(v4c_config(cfg)))
    transport = FakeTransport(None, answers_by_text(questions), questions)
    ctx = make_context(tmp / "v4c.json", {}, transport)
    transport.paths = ctx.paths
    stages_model.preflight(ctx, get=fake_tags_get)
    with pytest.raises(StageError, match="detgate"):
        stages_v4c.reader_levels(ctx, "dev", [1])  # gate not passed yet
    gate = stages_v4c.detgate(ctx)
    screen_dev = stages_data.screen(ctx, "dev")
    screen_test = stages_data.screen(ctx, "test")
    stages_v4c.reader_levels(ctx, "dev", list(range(1, 21)))
    stages_v4c.reader_levels(ctx, "test", list(range(1, 21)))
    subset = textprobe.textprobe(ctx)
    stages_v4c.v1arm(ctx)
    pilot = stages_v4.pilot_v4(ctx, [1, 5, 10, 15, 20])
    stages_model.infer(ctx, "dev", list(range(1, 21)))
    grid = stages_v4.devgrid_v4(ctx)
    return dict(
        ctx=ctx,
        v1=v1,
        transport=transport,
        gate=gate,
        screen_dev=screen_dev,
        screen_test=screen_test,
        subset=subset,
        pilot=pilot,
        grid=grid,
    )


def test_determinism_gate_passes_on_deterministic_backend(v4c_run) -> None:
    gate = v4c_run["gate"]
    assert (
        gate["passed"]
        and gate["argmax_calls"] >= 100
        and gate["argmax_match_rate"] == 1.0
    )
    assert all(p["content_identical"] for p in gate["repeat_pairs"].values())
    assert gate["decoding"]["options"]["presence_penalty"] == 0.0


def test_determinism_gate_fails_on_flaky_backend(v4c_run, tmp_path: Path) -> None:
    ctx = v4c_run["ctx"]

    class Flaky:
        def __init__(self) -> None:
            self.n = 0

        def send(self, payload: dict) -> TransportResult:
            self.n += 1
            lp = [
                {
                    "token": "x",
                    "logprob": math.log(0.4),
                    "top_logprobs": [
                        {"token": "y", "logprob": math.log(0.6)},
                        {"token": "x", "logprob": math.log(0.4)},
                    ],
                }
            ]
            return TransportResult(f'{{"answer": "{self.n}"}}', 1, 0.0, logprobs=lp)

    flaky_ctx = make_context(ctx.config_path, {}, None)
    flaky_ctx.transport_factory = lambda _s: Flaky()
    flaky_ctx.paths = dataclasses.replace(ctx.paths, output_root=tmp_path / "flaky")
    with pytest.raises(StageError, match="determinism gate failed"):
        stages_v4c.detgate(flaky_ctx)
    report = read_json(tmp_path / "flaky" / "preflight_determinism.json")
    assert not report["passed"] and report["argmax_match_rate"] < 1.0


def test_screening_both_splits_and_diff_report(v4c_run) -> None:
    ctx = v4c_run["ctx"]
    for split in ("dev", "test"):
        diff = read_json(ctx.paths.out(f"adoption_diff_{split}.json"))
        assert {"outcome_changed", "adopted_kept", "adopted_dropped"} <= set(
            diff["summary"]
        )
        assert ctx.state.get("screening")[split]["reference_diff"] == diff["summary"]
    assert ctx.paths.selection("selected_manifest.jsonl").exists()
    assert "without freeze" in ctx.paths.out("worklog.md").read_text()


def test_reader_levels_and_own_textprobe(v4c_run) -> None:
    ctx, subset = v4c_run["ctx"], v4c_run["subset"]
    for split in ("dev", "test"):
        table = pd.read_csv(
            ctx.paths.out(f"reader_levels_{split}.csv"), dtype={"question_id": str}
        )
        assert len(table) == 20 * len(ctx.selected(split, use_subset=False))
    own = {r["question_id"] for r in ctx.selected("dev", use_subset=False)}
    assert set(subset["dev"]) | set(subset["excluded"]["dev"]) == own


def test_v1_arm_and_devgrid_comparators(v4c_run) -> None:
    ctx, grid = v4c_run["ctx"], v4c_run["grid"]
    assert ctx.paths.out("v1arm/frozen_policy_subset.json").exists()
    frame = pd.read_csv(ctx.paths.out("dev_threshold_grid.csv"))
    methods = set(frame["method"])
    assert {
        "checklist_v4_min",
        "direct_v4",
        "checklist_v1_min",
        "direct_v1_verbalized",
    } <= methods
    assert "checklist_v3_min" not in methods
    assert grid["abort_methods"] == ["checklist_v4_min"]
    assert "checklist_v4_min" in grid["mean_level_diff_vs_fair_v1"]
    assert not ctx.paths.out("frozen_policy.json").exists()


def test_no_method_inference_on_test(v4c_run) -> None:
    ctx = v4c_run["ctx"]
    with pytest.raises(StageError):
        stages_model.infer(ctx, "test", [1])
    with pytest.raises(StageError):
        stages_model.freeze(ctx)
