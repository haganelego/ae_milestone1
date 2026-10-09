"""v3 final (plan §18.6): N = 1 standalone items, v1 scores, attribute flag,
L20-zero cause attribution, and a synthetic v1 -> v3 dev run to devgrid."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
from gqa_stop_helpers import (
    N_IMAGES,
    OBJECTS,
    PROMPTS_DIR,
    FakeTransport,
    answers_by_text,
    fake_tags_get,
    make_context,
    run_v1_pipeline,
)

from milestone1.gqa_stop import (
    fair_subset,
    stages_data,
    stages_model,
    stages_v3,
    textprobe,
)
from milestone1.gqa_stop.client import RequestEngine, load_prompts, settings_from_config
from milestone1.gqa_stop.common import StageError, read_json
from milestone1.gqa_stop.inference import effective_confidences
from milestone1.gqa_stop.item_checklist import item_row, item_spec
from milestone1.gqa_stop.pilot_v3 import l20_zero_causes
from milestone1.gqa_stop.planner import PlanOutcome
from milestone1.gqa_stop.sampled import CallTally, validate_plan_v3

CFG_V3 = json.loads(
    (PROMPTS_DIR.parents[1] / "configs" / "gqa_stop_v3.json").read_text()
)
ROLES = CFG_V3["prompts"]["roles"]


def test_live_v3_config_has_no_sampling() -> None:
    assert "sampling" not in CFG_V3
    assert CFG_V3["policy"]["family"] == "item_checklist"
    assert CFG_V3["policy"]["methods"] == ["checklist_v3_min"]
    assert not any(
        "sagree" in m or "smean" in m for m in json.dumps(CFG_V3["policy"]).split('"')
    )


@pytest.mark.parametrize(
    ("last", "expected"),
    [
        ("What color is the cup on the table?", True),
        ("What is the cup on the table made of?", True),
        ("What material is the plate?", True),
        ("What is the shape of the sign?", True),
        ("Is there another cup to the right of the cup?", False),
        ("What specific types of utensils are inside the box?", False),
    ],
)
def test_last_item_has_attribute_flag(last: str, expected: bool) -> None:
    plan = {
        "items": [
            {"id": "c1", "question": "Is there a cup in the image?", "depends_on": []},
            {"id": "c2", "question": last, "depends_on": ["c1"]},
        ]
    }
    parsed, errors = validate_plan_v3(plan, 2, 3)
    assert errors == [] and parsed["last_item_has_attribute"] is expected


def test_item_spec_is_single_call_with_v1_decoding(tmp_path: Path) -> None:
    eng = RequestEngine(
        settings_from_config(CFG_V3), load_prompts(PROMPTS_DIR, ROLES), None, tmp_path
    )
    q = {"question_id": "q", "image_id": "1", "question": "What color is the cup?"}
    spec = item_spec(q, 4, b"img", {"id": "c2", "question": "What color is the cup?"})
    payload = eng.build_payload(spec, eng.build_messages(spec))
    assert payload["options"]["temperature"] == 0.0
    assert payload["options"]["seed"] == 20261008
    assert eng.record_path(spec).name == "04_c2.json"
    assert json.loads(payload["messages"][0]["content"].split("Input data: ")[1]) == {
        "question": "What color is the cup?"
    }


PLAN = PlanOutcome(
    "q",
    "valid",
    [
        {"id": "c1", "question": "Is there a cup?", "depends_on": []},
        {"id": "c2", "question": "What color is the cup?", "depends_on": ["c1"]},
    ],
    True,
    1,
)
Q = {"question_id": "q", "image_id": "1", "question": "What color is the cup?"}


def obs(item: str, answer: str, resolved: bool, conf: float) -> dict:
    return {"id": item, "answer": answer, "resolved": resolved, "confidence": conf}


def test_item_row_uses_v1_effective_confidence() -> None:
    observations = {
        "c1": obs("c1", "yes", True, 0.0),
        "c2": obs("c2", "red", True, 0.8),
    }
    row = item_row(Q, 5, PLAN, True, None, None, observations, CallTally())
    expected = effective_confidences(PLAN.items, list(observations.values()))
    assert row["r_values"] == expected  # parent resolved with conf 0 still passes (v1)
    assert row["checklist_v3_min"] == 0.0 and row["checklist_v3_mean"] == pytest.approx(
        0.4
    )
    gated = item_row(
        Q,
        5,
        PLAN,
        True,
        None,
        None,
        {"c1": obs("c1", "no", False, 0.0), "c2": obs("c2", "red", True, 0.9)},
        CallTally(),
    )
    assert gated["item2_r"] == 0.0  # unresolved parent zeroes the child
    failed = item_row(
        Q, 5, PLAN, True, None, None, {"c1": None, "c2": None}, CallTally()
    )
    assert failed["checklist_v3_min"] == 0.0 and failed["item1_valid"] is False


def cause_row(
    qid: str,
    item1: str,
    resolved: bool,
    last_attr: bool = True,
    standalone_flag: bool = False,
    count: bool = False,
    score: float = 0.0,
) -> dict:
    return {
        "question_id": qid,
        "level": 20,
        "checklist_v3_min": score,
        "item1_valid": True,
        "item1_answer": item1,
        "item1_resolved": resolved,
        "last_item_has_attribute": last_attr,
        "item_not_standalone_any": standalone_flag,
        "count_item_any": count,
    }


def test_l20_zero_causes_by_hand() -> None:
    table = pd.DataFrame(
        [
            cause_row("a1", "no", True),
            cause_row("a2", "unknown", False, last_attr=False),
            cause_row("c1", "yes", True, last_attr=False),
            cause_row("c2", "yes", True, count=True),
            cause_row("o1", "yes", True),
            cause_row("ok", "yes", True, score=0.9),
        ]
    )
    out = l20_zero_causes(table)
    assert out["n_zero_at_last_level"] == 5
    assert out["primary_counts"] == {
        "a_existence_no_or_unresolved": 2,
        "c_planner_defect": 2,
        "other_later_item_unresolved": 1,
    }
    assert out["flag_counts"]["c_planner_defect"] == 3  # a2 also has a planner defect
    assert out["flag_counts"]["b_wording_variance"] is None


# ---- synthetic v1 -> v3 (item_checklist) run --------------------------------------------
def item_config(v1_cfg: dict, tmp: Path) -> dict:
    cfg = copy.deepcopy(v1_cfg)
    cfg["paths"].update(
        {
            "output_root": "outputs/run_v3",
            "selection_root": "data/run_v3",
            "inherit_split_from": "outputs/run",
            "reuse_records_from": {
                "output_root": "outputs/run",
                "roles": ["answer_only", "direct"],
            },
            "subset_file": "outputs/run_v3/visual_subset.json",
        }
    )
    cfg["prompts"] = {"roles": ROLES}
    cfg["model"]["num_predict"].update(
        {"text_only": 128, "planner_v3": 1024, "observer_v3": 256}
    )
    cfg["policy"].update({"family": "item_checklist", "methods": ["checklist_v3_min"]})
    graphs = {
        str(1000 + i): {
            "objects": {
                "o": {"name": OBJECTS[i % len(OBJECTS)]},
                "t": {"name": "table"},
            }
        }
        for i in range(N_IMAGES)
    }
    graphs["9999"] = {"objects": {"a": {"name": "giraffe"}, "b": {"name": "zebra"}}}
    (tmp / "scene_graphs.json").write_text(json.dumps(graphs))
    cfg["negative_probe"] = {
        **CFG_V3["negative_probe"],
        "scene_graphs": str(tmp / "scene_graphs.json"),
    }
    cfg["comparison"] = {
        "v1_output_root": "outputs/run",
        "v1_method": "checklist_min",
        "v1_fair_frozen": "outputs/run/fair_subset/frozen_policy_subset.json",
    }
    return cfg


@pytest.fixture(scope="module")
def item_run(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("v3item")
    _, cfg, questions, v1 = run_v1_pipeline(tmp)
    path = tmp / "gqa_stop_v3.json"
    path.write_text(json.dumps(item_config(cfg, tmp)))
    transport = FakeTransport(None, answers_by_text(questions), questions)
    v3 = make_context(path, {}, transport)
    transport.paths = v3.paths
    stages_model.preflight(v3, get=fake_tags_get)
    textprobe.textprobe(v3)
    subset = v3.paths.out("visual_subset.json")
    fair_subset.freeze_on_subset(v1, subset)
    stages_data.screen(v3, "dev")
    pilot = stages_v3.pilot_v3(v3, [1, 5, 10, 15, 20])
    stages_model.infer(v3, "dev", list(range(1, 21)))
    grid = stages_v3.devgrid_v3(v3)
    return v3, transport, pilot, grid


def test_item_pilot_metrics(item_run) -> None:
    _, _, pilot, _ = item_run
    assert pilot["family"] == "item_checklist" and pilot["format_valid_ok"] is True
    assert pilot["plan_quality"]["attribute_item_fraction"] == 1.0
    assert set(pilot["by_level"]) == {"1", "5", "10", "15", "20"}
    assert "item1_resolved_rate" in pilot["by_level"]["1"]
    assert "n_zero_at_last_level" in pilot["l20_zero_causes"]
    assert pilot["negative_probe"] and pilot["sampling"].startswith("none")


def test_item_run_reuses_reader_and_direct_and_calls_observer_once(item_run) -> None:
    v3, transport, _, _ = item_run
    assert transport.calls["answer_only"] == 0 and transport.calls["direct"] == 0
    records = list(v3.paths.out("observer_v3").rglob("*.json"))
    assert records and not any("_s" in p.stem.split("_c")[-1] for p in records)
    record = json.loads(records[0].read_text())
    assert record["decoding"]["options"]["temperature"] == 0.0


def test_item_devgrid(item_run) -> None:
    v3, _, _, grid = item_run
    frame = pd.read_csv(v3.paths.out("dev_threshold_grid.csv"))
    assert set(frame["method"]) == {"checklist_v3_min", "checklist_v3_mean", "direct"}
    assert grid["abort_methods"] == ["checklist_v3_min"]
    assert v3.paths.out("dev_l20_zero_causes.json").exists()
    comparison = pd.read_csv(v3.paths.out("dev_comparison.csv"))
    assert comparison["method"].str.startswith("fair v1 checklist_min").any()
    assert not v3.paths.out("frozen_policy.json").exists()
    with pytest.raises(StageError):
        stages_model.freeze(v3)


def test_devgrid_refuses_stale_family(item_run) -> None:
    v3, _, _, _ = item_run
    info = v3.state.get("infer")
    v3.state.set(
        "infer", {**info, "dev": {**info["dev"], "family": "sampled_checklist"}}
    )
    try:
        with pytest.raises(StageError, match="family"):
            stages_v3.devgrid_v3(v3)
    finally:
        v3.state.set("infer", info)
    assert (
        read_json(v3.paths.out("devgrid_summary.json"))["frozen_policy_written"]
        is False
    )
