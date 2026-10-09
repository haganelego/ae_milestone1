"""v3 (plan §18): textprobe + visual subset, fair-v1 subset freeze/evaluate,
standalone-item flag, per-sample caching, S_mean/S_agree/tie rule/gating,
negative probe noun choice, and a synthetic v1 -> v3 dev run to devgrid."""

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
    make_world,
    raw_question,
)

from milestone1.gqa_stop import (
    fair_subset,
    stages_data,
    stages_model,
    stages_v3,
    textprobe,
)
from milestone1.gqa_stop.client import (
    RequestEngine,
    TransportResult,
    load_prompts,
    settings_from_config,
)
from milestone1.gqa_stop.common import StageError, read_json
from milestone1.gqa_stop.dataset import to_record
from milestone1.gqa_stop.pilot_v3 import absent_noun, vocabulary
from milestone1.gqa_stop.sampled import (
    Sample,
    Sampling,
    checklist_scores,
    direct_sample_spec,
    direct_scores,
    item_not_standalone,
    item_scores,
    majority_answer,
    observer_spec,
    validate_observation,
    validate_plan_v3,
)

V3_ROLES = ["answer_only", "text_only", "planner_v3", "observer_v3", "direct"]
CFG_V3 = json.loads(
    (PROMPTS_DIR.parents[1] / "configs" / "gqa_stop_v3.json").read_text()
)
SAMPLING = Sampling(5, 0.7, 20261008)
# The withdrawn N = 5 design (plan §18.2) is kept as the "sampled_checklist" family;
# the live v3 config now uses "item_checklist" (§18.6), so pin its methods here.
SAMPLED_METHODS = [
    "checklist_v3_min_smean",
    "checklist_v3_min_sagree",
    "direct_sampled_smean",
    "direct_sampled_sagree",
]


def engine(tmp_path: Path, transport=None) -> RequestEngine:
    return RequestEngine(
        settings_from_config(CFG_V3),
        load_prompts(PROMPTS_DIR, V3_ROLES),
        transport,
        tmp_path,
    )


# ---- textprobe GT leak ----------------------------------------------------------------
def test_textprobe_request_has_no_gt_and_no_image(tmp_path: Path) -> None:
    raw = raw_question(
        "4242", "What color is the mug on the table?", "chartreuse", "color", 1, "mug"
    )
    raw["fullAnswer"] = "The mug is definitely chartreuse."
    full = {
        **to_record("q1", raw).to_dict(),
        "semantic": raw["semantic"],
        "imageId": "4242",
        "fullAnswer": raw["fullAnswer"],
    }
    spec = textprobe.text_only_spec(full)
    eng = engine(tmp_path)
    text = json.dumps(eng.build_payload(spec, eng.build_messages(spec)))
    for word in ("chartreuse", "definitely", "4242", "table,on,s", "select", "q1"):
        assert word not in text
    assert "What color is the mug on the table?" in text
    assert spec.image_bytes is None and "images" not in text


def test_build_subset_excludes_prior_answerable() -> None:
    rows = [
        {"question_id": "a", "split": "dev", "text_ok": True, "status": "valid"},
        {"question_id": "b", "split": "dev", "text_ok": False, "status": "valid"},
        {
            "question_id": "c",
            "split": "test",
            "text_ok": False,
            "status": "schema_failure",
        },
    ]
    subset = textprobe.build_subset(rows)
    assert subset["dev"] == ["b"] and subset["excluded"]["dev"] == ["a"]
    assert subset["test"] == ["c"] and subset["text_only_failures_kept"]["test"] == [
        "c"
    ]
    assert subset["counts"]["dev"] == {"adopted": 2, "visual": 1, "prior_answerable": 1}


# ---- standalone items ------------------------------------------------------------------
@pytest.mark.parametrize(
    ("question", "flag"),
    [
        ("Is there a cup on the table in the image?", False),
        ("What color is the cup that is on the table?", False),
        ("What color is it?", True),
        ("If yes, what color is the cup?", True),
        ("What material is item c1 made of?", True),
    ],
)
def test_item_not_standalone_heuristic(question: str, flag: bool) -> None:
    assert item_not_standalone(question) is flag


def test_plan_v3_flags_but_does_not_reject() -> None:
    plan = {
        "items": [
            {"id": "c1", "question": "Is there a cup in the image?", "depends_on": []},
            {"id": "c2", "question": "What color is it?", "depends_on": ["c1"]},
        ]
    }
    parsed, errors = validate_plan_v3(plan, 2, 3)
    assert errors == []
    assert [i["item_not_standalone"] for i in parsed["items"]] == [False, True]
    assert validate_plan_v3({"items": plan["items"][:1]}, 2, 3)[
        1
    ]  # schema still applies


def test_observation_schema() -> None:
    assert (
        validate_observation({"answer": "yes", "resolved": True, "confidence": 0.5})[1]
        == []
    )
    assert validate_observation(
        {"answer": "yes", "resolved": "true", "confidence": 0.5}
    )[1]
    assert validate_observation({"items": []})[1]


# ---- per-sample caching ----------------------------------------------------------------
def test_samples_have_distinct_keys_paths_and_decoding(tmp_path: Path) -> None:
    eng = engine(tmp_path)
    q = {"question_id": "q", "image_id": "1", "question": "What color?"}
    item = {"id": "c2", "question": "What color is the cup?", "depends_on": ["c1"]}
    specs = [observer_spec(q, 3, b"img", item, i, SAMPLING) for i in range(5)]
    assert len({eng.cache_key(s) for s in specs}) == 5
    paths = [eng.record_path(s).name for s in specs]
    assert paths == [f"03_c2_s{i}.json" for i in range(5)]
    for i, spec in enumerate(specs):
        payload = eng.build_payload(spec, eng.build_messages(spec))
        assert payload["options"]["temperature"] == 0.7
        assert payload["options"]["seed"] == 20261008 + i
        assert json.loads(
            payload["messages"][0]["content"].split("Input data: ")[1]
        ) == {"question": "What color is the cup?"}  # item question only
    direct = [direct_sample_spec(q, 3, b"img", i, SAMPLING) for i in range(5)]
    assert len({eng.cache_key(s) for s in direct}) == 5


def test_sampled_records_resume_without_calls(tmp_path: Path) -> None:
    class Scripted:
        def __init__(self) -> None:
            self.n = 0

        def send(self, payload: dict) -> TransportResult:
            self.n += 1
            return TransportResult(
                '{"answer":"red","resolved":true,"confidence":0.8}', 1, 0
            )

    q = {"question_id": "q", "image_id": "1", "question": "What color?"}
    item = {"id": "c1", "question": "Is there a cup?", "depends_on": []}
    first = Scripted()
    for i in range(5):
        engine(tmp_path, first).run(
            observer_spec(q, 1, b"x", item, i, SAMPLING), validate_observation
        )
    again = Scripted()
    for i in range(5):
        engine(tmp_path, again).run(
            observer_spec(q, 1, b"x", item, i, SAMPLING), validate_observation
        )
    assert first.n == 5 and again.n == 0


# ---- S_mean / S_agree / tie rule / gating ------------------------------------------------------
def s(
    answer: str, resolved: bool = True, conf: float = 0.8, valid: bool = True
) -> Sample:
    return Sample(valid, answer if valid else None, resolved, conf)


def test_item_scores_by_hand() -> None:
    samples = [
        s("red", conf=0.9),
        s("Red ", conf=0.7),
        s("blue", conf=0.5),
        s("unknown", False, 0.0),
        s("x", valid=False),
    ]
    scores = item_scores(samples, 5)
    assert scores["majority"] == "red"
    assert scores["smean"] == pytest.approx((0.9 + 0.7 + 0.5) / 5)
    assert scores["sagree"] == pytest.approx(2 / 5)
    assert scores["resolved_frac"] == pytest.approx(3 / 5)


def test_tie_rule_picks_lexicographically_smallest() -> None:
    samples = [s("red"), s("blue"), s("red"), s("blue"), s("green")]
    assert majority_answer(samples) == "blue"
    assert item_scores(samples, 5)["sagree"] == pytest.approx(2 / 5)
    assert majority_answer([s("a", valid=False)]) is None


def test_unresolved_majority_gives_zero_agree() -> None:
    samples = [s("unknown", False, 0.0)] * 3 + [s("red"), s("red")]
    assert majority_answer(samples) == "unknown"
    assert item_scores(samples, 5)["sagree"] == 0.0


PLAN = [
    {"id": "c1", "question": "a", "depends_on": []},
    {"id": "c2", "question": "b", "depends_on": ["c1"]},
    {"id": "c3", "question": "c", "depends_on": ["c2"]},
]


def scores(**items: tuple[float, float]) -> dict:
    return {k: {"smean": v[0], "sagree": v[1]} for k, v in items.items()}


def test_gating_parent_resolved_in_at_least_one_sample() -> None:
    out = checklist_scores(PLAN, scores(c1=(0.0, 0.0), c2=(0.9, 1.0), c3=(0.9, 1.0)))
    assert out["v3_min_smean"] == 0.0 and out["v3_mean_smean"] == 0.0
    out = checklist_scores(PLAN, scores(c1=(0.2, 0.2), c2=(0.9, 0.4), c3=(0.8, 1.0)))
    assert out["v3_min_smean"] == pytest.approx(0.2)
    assert out["v3_min_sagree"] == pytest.approx(0.2)
    assert out["v3_mean_sagree"] == pytest.approx((0.2 + 0.4 + 1.0) / 3)
    # strict diagnostic: parent S_agree must be >= 0.5 -> c2 (child of c1=0.2) is 0
    assert out["v3_min_smean_strict"] == 0.0
    assert out["v3_mean_sagree_strict"] == pytest.approx(0.2 / 3)


def test_planner_failure_scores_zero() -> None:
    out = checklist_scores([], {})
    assert set(out.values()) == {0.0} and "v3_min_sagree_strict" in out


def test_direct_scores() -> None:
    out = direct_scores(
        [
            s("red", conf=0.9),
            s("red", conf=0.5),
            s("blue", conf=1.0),
            s("x", valid=False),
            s("red", conf=0.6),
        ],
        5,
    )
    assert out["direct_smean"] == pytest.approx((0.9 + 0.5 + 1.0 + 0.6) / 5)
    assert out["direct_sagree"] == pytest.approx(3 / 5)


# ---- negative probe noun -------------------------------------------------------------------
def test_absent_noun_avoids_present_objects_and_question() -> None:
    graphs = {
        "1": {"objects": {"a": {"name": "cars"}, "b": {"name": "tree"}}},
        "2": {"objects": {"a": {"name": "car"}, "b": {"name": "dog"}}},
        "3": {"objects": {"a": {"name": "dog"}, "b": {"name": "tree"}}},
    }
    vocab = vocabulary(graphs, 10)
    assert vocab == ["dog", "tree", "car", "cars"]
    noun = absent_noun("1", graphs["1"], "What color is the dog?", vocab, 1)
    assert noun is None  # car/cars/tree present, dog in question
    assert absent_noun("3", graphs["3"], "What color is the dog?", vocab, 1) == "car"


# ---- synthetic v1 -> v3 run ----------------------------------------------------------------
def v3_config(v1_cfg: dict, tmp: Path) -> dict:
    cfg = copy.deepcopy(v1_cfg)
    cfg["paths"].update(
        {
            "output_root": "outputs/run_v3",
            "selection_root": "data/run_v3",
            "inherit_split_from": "outputs/run",
            "reuse_records_from": {
                "output_root": "outputs/run",
                "roles": ["answer_only"],
            },
            "subset_file": "outputs/run_v3/visual_subset.json",
        }
    )
    cfg["prompts"] = {"roles": V3_ROLES}
    cfg["model"]["num_predict"].update(
        {"text_only": 128, "planner_v3": 1024, "observer_v3": 256}
    )
    cfg["sampling"] = {"n": 5, "temperature": 0.7, "seed_base": 20261008}
    cfg["policy"].update(
        {"family": "sampled_checklist", "methods": list(SAMPLED_METHODS)}
    )
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
def v3_run(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("v3")
    config_path, cfg, questions = make_world(tmp)
    t1 = FakeTransport(None, answers_by_text(questions), questions)
    v1 = make_context(config_path, cfg, t1)
    t1.paths = v1.paths
    stages_model.preflight(v1, get=fake_tags_get)
    stages_data.prepare(v1)
    stages_data.screen(v1, "dev")
    stages_model.pilot(v1, [1, 5, 10, 15, 20], calibration=True, use_fallback=False)
    stages_model.infer(v1, "dev", list(range(1, 21)))
    stages_model.freeze(v1)
    stages_data.screen(v1, "test")
    stages_model.infer(v1, "test", list(range(1, 21)))
    original_frozen = v1.paths.out("frozen_policy.json").read_text()
    v3_path = tmp / "gqa_stop_v3.json"
    v3_path.write_text(json.dumps(v3_config(cfg, tmp)))
    t3 = FakeTransport(None, answers_by_text(questions), questions)
    v3 = make_context(v3_path, {}, t3)
    t3.paths = v3.paths
    stages_model.preflight(v3, get=fake_tags_get)
    subset = textprobe.textprobe(v3)
    subset_path = v3.paths.out("visual_subset.json")
    v1_calls = sum(t1.calls.values())
    fair_frozen = fair_subset.freeze_on_subset(v1, subset_path)
    fair_summary = fair_subset.evaluate_on_subset(v1, subset_path)
    fair_calls = sum(t1.calls.values()) - v1_calls
    stages_data.screen(v3, "dev")
    pilot = stages_v3.pilot_v3(v3, [1, 5, 10, 15, 20])
    stages_model.infer(v3, "dev", list(range(1, 21)))
    grid = stages_v3.devgrid_v3(v3)
    return dict(
        v1=v1,
        v3=v3,
        t3=t3,
        subset=subset,
        subset_path=subset_path,
        fair_frozen=fair_frozen,
        fair_summary=fair_summary,
        fair_calls=fair_calls,
        original_frozen=original_frozen,
        pilot=pilot,
        grid=grid,
    )


def test_textprobe_subset_written_once(v3_run) -> None:
    v3, subset = v3_run["v3"], v3_run["subset"]
    assert subset["counts"]["dev"]["prior_answerable"] > 0
    assert set(subset["dev"]).isdisjoint(subset["excluded"]["dev"])
    with pytest.raises(StageError, match="only once"):
        textprobe.textprobe(v3)
    assert {r["question_id"] for r in v3.selected("dev")} == set(subset["dev"])


def test_fair_v1_subset_freeze_and_evaluate(v3_run) -> None:
    v1, subset = v3_run["v1"], v3_run["subset"]
    out = v1.paths.out("fair_subset")
    frozen = read_json(out / "frozen_policy_subset.json")
    assert frozen["dev_question_ids"] == sorted(subset["dev"])
    assert "Re-analysis of run v1" in frozen["analysis"]
    assert v3_run["fair_summary"]["n_questions"] == len(subset["test"])
    for name in (
        "method_summary.csv",
        "paired_comparisons.csv",
        "bootstrap_summary.json",
        "figures/test_meanlevel_accuracy.png",
        "analysis_label.json",
    ):
        assert (out / name).exists(), name
    assert v1.paths.out("frozen_policy.json").read_text() == v3_run["original_frozen"]
    assert v3_run["fair_calls"] == 0  # cached records only
    with pytest.raises(StageError):
        fair_subset.freeze_on_subset(v1, v3_run["subset_path"])  # no re-freeze


def test_fair_v1_refuses_changed_subset(v3_run, tmp_path: Path) -> None:
    other = tmp_path / "other.json"
    data = json.loads(v3_run["subset_path"].read_text())
    other.write_text(json.dumps({**data, "test": data["test"][:-1]}))
    with pytest.raises(StageError, match="subset file differs"):
        fair_subset.evaluate_on_subset(v3_run["v1"], other)


def test_v3_pilot_metrics_and_negative_probe(v3_run) -> None:
    pilot = v3_run["pilot"]
    assert set(pilot["valid_rate_initial"]) == {
        "observer",
        "direct",
        "reader",
        "planner",
    }
    assert pilot["format_valid_ok"] is True
    l1 = pilot["by_level"]["1"]
    assert (
        0.0
        <= l1["item1_resolved_sample_rate"]
        <= l1["item1_any_resolved_question_rate"]
    )
    assert sum(l1["histograms_10_bins"]["v3_min_sagree"]) == pilot["questions"]
    assert set(pilot["negative_probe"]) == {"1", "5", "20"}
    negative = pd.read_csv(v3_run["v3"].paths.out("pilot/negative_probe.csv"))
    assert (negative["noun"] == negative["noun"]).all() and not negative.empty
    assert len(pilot["examples"]) == min(5, pilot["questions"])


def test_v3_records_are_sampled_and_namespaced(v3_run) -> None:
    v3 = v3_run["v3"]
    obs = list(v3.paths.out("observer_v3").rglob("*_s4.json"))
    assert obs
    record = json.loads(obs[0].read_text())
    assert record["decoding"]["options"]["seed"] == 20261008 + 4
    assert list(v3.paths.out("observer_v3_negprobe").rglob("*_neg_s0.json"))
    assert v3_run["t3"].calls["answer_only"] == 0  # reader copied from v1


def test_v3_devgrid_outputs_and_guards(v3_run) -> None:
    v3, grid = v3_run["v3"], v3_run["grid"]
    frame = pd.read_csv(v3.paths.out("dev_threshold_grid.csv"))
    assert set(frame["method"]) == set(SAMPLED_METHODS)
    assert frame.groupby("method").size().eq(101).all()
    for name in (
        "dev_stop_hist.csv",
        "dev_components.csv",
        "dev_comparison.csv",
        "dev_fixed_curve.csv",
        "devgrid_summary.json",
        "level_table_dev.csv",
    ):
        assert v3.paths.out(name).exists(), name
    comparison = pd.read_csv(v3.paths.out("dev_comparison.csv"))
    assert comparison["method"].str.startswith("fair v1 checklist_min").any()
    assert comparison["method"].str.contains("Fixed-dev").any()
    assert grid["frozen_policy_written"] is False
    assert not v3.paths.out("frozen_policy.json").exists()
    with pytest.raises(StageError):
        stages_model.freeze(v3)
    with pytest.raises(StageError):
        stages_data.screen(v3, "test")
    with pytest.raises(StageError):
        stages_model.infer(v3, "test", [1])
