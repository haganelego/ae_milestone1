"""v4 (plan §19): answer-token location, Rel parsing, logprob/raw-text requests,
multi-source reuse, P_ans x Rel gating, and a synthetic v1 -> v3 -> v4 dev run."""

from __future__ import annotations

import copy
import json
import math
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
    stages_v4,
    textprobe,
)
from milestone1.gqa_stop.client import (
    RequestEngine,
    TransportResult,
    load_prompts,
    settings_from_config,
)
from milestone1.gqa_stop.common import StageError, atomic_write_json, read_json
from milestone1.gqa_stop.item_checklist import item_spec
from milestone1.gqa_stop.logprob_v4 import (
    direct_lp_spec,
    gated_scores,
    locate_answer_token,
    observer_lp_spec,
    relevance_from_logprobs,
    relevance_spec,
)
from milestone1.gqa_stop.planner import PlanOutcome
from milestone1.gqa_stop.sampled import validate_observation
from milestone1.gqa_stop.schemas import validate_reader

CFG_V4 = json.loads(
    (PROMPTS_DIR.parents[1] / "configs" / "gqa_stop_v4.json").read_text()
)
ROLES = CFG_V4["prompts"]["roles"]


def tok(token: str, p: float = 1.0, top: list | None = None) -> dict:
    entry = {"token": token, "logprob": math.log(p), "bytes": list(token.encode())}
    entry["top_logprobs"] = top or [{"token": token, "logprob": math.log(p)}]
    return entry


# real token split of '{\n"answer": "yellow",\n"resolved": true,\n"confidence": 1.0\n}'
REAL = [
    tok("{"),
    tok("\n"),
    tok('"'),
    tok("answer"),
    tok('":'),
    tok(' "'),
    tok("yellow", 0.803),
    tok('",'),
    tok("\n"),
    tok('"'),
    tok("resolved"),
    tok('":'),
    tok(" true"),
    tok(","),
    tok("\n"),
    tok('"'),
    tok("confidence"),
    tok('":'),
    tok(" "),
    tok("1"),
    tok("."),
    tok("0"),
    tok("\n"),
    tok("}"),
]


# ---- answer token --------------------------------------------------------------------------
def test_locate_on_real_token_split() -> None:
    found = locate_answer_token(REAL)
    assert found is not None and found.index == 6 and found.token == "yellow"
    assert found.prob == pytest.approx(0.803)


def test_locate_in_fenced_json() -> None:
    fenced = [
        tok("```"),
        tok("json"),
        tok("\n"),
        tok("{"),
        tok('"'),
        tok("answer"),
        tok('":'),
        tok(' "'),
        tok("round", 0.6),
        tok('"}'),
        tok("\n```"),
    ]
    found = locate_answer_token(fenced)
    assert found.token == "round" and found.prob == pytest.approx(0.6)


def test_locate_when_quote_and_value_share_a_token() -> None:
    merged = [tok('{"answer":'), tok(' "yel', 0.7), tok("low"), tok('"}')]
    found = locate_answer_token(merged)
    assert found.token == ' "yel' and found.prob == pytest.approx(0.7)


def test_locate_uses_bytes_for_multibyte_tokens() -> None:
    entries = [tok('{"answer": "'), tok("é", 0.5), tok('"}')]
    assert locate_answer_token(entries).token == "é"


@pytest.mark.parametrize(
    "entries",
    [
        None,
        [],
        [tok('{"resolved": true}')],
        [tok('{"answer": ""'), tok("}")],
        [tok('{"answer"'), tok(": 3}")],
    ],
)
def test_locate_returns_none_when_absent(entries) -> None:
    assert locate_answer_token(entries) is None


# ---- relevance -----------------------------------------------------------------------------
def rel_entry(first: str, alts: dict[str, float]) -> list[dict]:
    top = [{"token": t, "logprob": math.log(p)} for t, p in alts.items()]
    return [
        {
            "token": first,
            "logprob": math.log(alts.get(first, 0.01)),
            "top_logprobs": top,
        }
    ]


def test_rel_yes_present() -> None:
    rel = relevance_from_logprobs(
        rel_entry("Yes", {"Yes": 0.85, "No": 0.14, " Yes": 0.01})
    )
    assert rel.p_yes == pytest.approx(0.85) and rel.p_no == pytest.approx(0.14)
    assert rel.has_yes_or_no and rel.first_token == "Yes"


def test_rel_yes_absent_is_zero_and_variants_not_merged() -> None:
    rel = relevance_from_logprobs(rel_entry("No", {"No": 0.7, " Yes": 0.2, "yes": 0.1}))
    assert rel.p_yes == 0.0 and rel.has_yes_or_no


def test_rel_neither_yes_nor_no() -> None:
    rel = relevance_from_logprobs(rel_entry("The", {"The": 0.6, "I": 0.4}))
    assert rel.p_yes == 0.0 and not rel.has_yes_or_no
    assert relevance_from_logprobs(None).p_yes == 0.0


# ---- client: logprobs, raw text, inline template, multi reuse ---------------------------------
def engine(tmp: Path, transport=None, **kw) -> RequestEngine:
    return RequestEngine(
        settings_from_config(CFG_V4),
        load_prompts(PROMPTS_DIR, ROLES),
        transport,
        tmp,
        **kw,
    )


Q = {"question_id": "q", "image_id": "1", "question": "What color is the cup?"}
ITEM = {"id": "c2", "question": "What color is the cup on the table?"}


def test_logprob_and_relevance_payloads(tmp_path: Path) -> None:
    eng = engine(tmp_path)
    obs = observer_lp_spec(Q, 3, b"img", ITEM)
    payload = eng.build_payload(obs, eng.build_messages(obs))
    assert payload["logprobs"] is True and payload["top_logprobs"] == 10
    assert payload["format"] == "json" and payload["options"]["temperature"] == 0.0
    assert eng.cache_key(obs) != eng.cache_key(item_spec(Q, 3, b"img", ITEM))
    assert "observer_v3_lp" in str(eng.record_path(obs))
    rel = relevance_spec("q", 3, b"img", ITEM["question"], "c2")
    payload = eng.build_payload(rel, eng.build_messages(rel))
    assert "format" not in payload and payload["logprobs"] is True
    assert payload["messages"][0]["content"] == (
        "Consider the question: 'What color is the cup on the table?'. Are you confident "
        "about answering the question with the current view? Answer with Yes or No."
    )
    assert payload["messages"][0]["images"]
    direct = direct_lp_spec(Q, 3, b"img")
    assert "direct_lp" in str(eng.record_path(direct))


class Scripted:
    def __init__(self, replies: list[TransportResult]) -> None:
        self.replies = replies
        self.n = 0

    def send(self, payload: dict) -> TransportResult:
        self.n += 1
        return self.replies.pop(0)


def test_raw_text_reply_and_logprobs_are_stored(tmp_path: Path) -> None:
    lp = rel_entry("Yes", {"Yes": 0.8, "No": 0.2})
    transport = Scripted([TransportResult("Yes", 1, 0.0, logprobs=lp)])
    eng = engine(tmp_path, transport)
    record = eng.run(relevance_spec("q", 3, b"img", "Is it?", "c1"), validate_reader)
    assert record["status"] == "valid" and record["parsed"] == {"text": "Yes"}
    assert record["logprobs"] == lp and transport.n == 1
    again = Scripted([])
    assert (
        engine(tmp_path, again).run(
            relevance_spec("q", 3, b"img", "Is it?", "c1"), validate_reader
        )["logprobs"]
        == lp
    )
    assert again.n == 0


def test_multiple_reuse_sources(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    spec = item_spec(Q, 2, b"img", ITEM)
    reply = '{"answer":"red","resolved":true,"confidence":0.9}'
    engine(second, Scripted([TransportResult(reply, 1, 0.0)])).run(
        spec, validate_observation
    )
    target = engine(
        tmp_path / "c",
        Scripted([]),
        reuse_sources=[
            (first, frozenset({"observer_v3"})),
            (second, frozenset({"observer_v3"})),
        ],
    )
    record = target.run(spec, validate_observation)
    assert record["copied_from"].startswith(str(second)) and target.copied == 1


# ---- P_ans x Rel under the v1 gate ------------------------------------------------------------
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


def sig(resolved: bool, pans: float, rel: float, answer: str = "x") -> dict:
    return {
        "valid": True,
        "answer": answer,
        "resolved": resolved,
        "pans": pans,
        "rel": rel,
        "product": pans * rel,
    }


def test_gated_scores_by_hand() -> None:
    out = gated_scores(PLAN, {"c1": sig(True, 0.9, 0.8), "c2": sig(True, 0.5, 0.6)})
    assert out["checklist_v4_min"] == pytest.approx(0.30)
    assert out["checklist_v4_mean"] == pytest.approx((0.72 + 0.30) / 2)
    assert out["pans_min"] == pytest.approx(0.5) and out["rel_min"] == pytest.approx(
        0.6
    )
    gated = gated_scores(PLAN, {"c1": sig(False, 0.9, 0.8), "c2": sig(True, 0.9, 0.9)})
    assert gated["checklist_v4_min"] == 0.0 and gated["checklist_v4_mean"] == 0.0
    unknown = gated_scores(
        PLAN, {"c1": sig(True, 0.9, 0.8), "c2": sig(True, 0.9, 0.9, "unknown")}
    )
    assert unknown["checklist_v4_min"] == 0.0


# ---- synthetic v1 -> v3 -> v4 dev run ---------------------------------------------------------
def scene_graphs(tmp: Path) -> str:
    graphs = {
        str(1000 + i): {"objects": {"o": {"name": OBJECTS[i % len(OBJECTS)]}}}
        for i in range(N_IMAGES)
    }
    (tmp / "scene_graphs.json").write_text(json.dumps(graphs))
    return str(tmp / "scene_graphs.json")


def v3_config(v1_cfg: dict, tmp: Path) -> dict:
    v3 = json.loads(
        (PROMPTS_DIR.parents[1] / "configs" / "gqa_stop_v3.json").read_text()
    )
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
    cfg["prompts"] = v3["prompts"]
    cfg["model"]["num_predict"].update(
        {"text_only": 128, "planner_v3": 1024, "observer_v3": 256, "relevance_v4": 8}
    )
    cfg["policy"].update({"family": "item_checklist", "methods": ["checklist_v3_min"]})
    cfg["negative_probe"] = {**v3["negative_probe"], "scene_graphs": scene_graphs(tmp)}
    cfg["comparison"] = {
        "v1_output_root": "outputs/run",
        "v1_method": "checklist_min",
        "v1_fair_frozen": "outputs/run/fair_subset/frozen_policy_subset.json",
    }
    return cfg


def v4_config(v3_cfg: dict) -> dict:
    cfg = copy.deepcopy(v3_cfg)
    cfg["paths"].update(
        {
            "output_root": "outputs/run_v4",
            "selection_root": "data/run_v4",
            "reuse_records_from": [
                {"output_root": "outputs/run", "roles": ["answer_only", "direct"]},
                {"output_root": "outputs/run_v3", "roles": ["planner_v3"]},
            ],
        }
    )
    cfg["prompts"] = CFG_V4["prompts"]
    cfg["policy"].update({"family": "logprob_item", "methods": ["checklist_v4_min"]})
    cfg.pop("negative_probe")
    cfg["comparison"]["v3_output_root"] = "outputs/run_v3"
    return cfg


@pytest.fixture(scope="module")
def v4_run(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("v4")
    _, cfg, questions, v1 = run_v1_pipeline(tmp)
    v3_cfg = v3_config(cfg, tmp)
    (tmp / "v3.json").write_text(json.dumps(v3_cfg))
    t3 = FakeTransport(None, answers_by_text(questions), questions)
    v3 = make_context(tmp / "v3.json", {}, t3)
    t3.paths = v3.paths
    stages_model.preflight(v3, get=fake_tags_get)
    textprobe.textprobe(v3)
    fair_subset.freeze_on_subset(v1, v3.paths.out("visual_subset.json"))
    stages_data.screen(v3, "dev")
    stages_v3.pilot_v3(v3, [1, 5, 10, 15, 20])
    stages_model.infer(v3, "dev", list(range(1, 21)))
    stages_v3.devgrid_v3(v3)
    (tmp / "v4.json").write_text(json.dumps(v4_config(v3_cfg)))
    t4 = FakeTransport(None, answers_by_text(questions), questions)
    v4 = make_context(tmp / "v4.json", {}, t4)
    t4.paths = v4.paths
    stages_model.preflight(v4, get=fake_tags_get)
    stages_data.screen(v4, "dev")
    pilot = stages_v4.pilot_v4(v4, [1, 5, 10, 15, 20])
    stages_model.infer(v4, "dev", list(range(1, 21)))
    grid = stages_v4.devgrid_v4(v4)
    return v4, t4, pilot, grid


def test_v4_pilot_gates_and_equality(v4_run) -> None:
    _, _, pilot, _ = v4_run
    assert pilot["gates_pass"] and pilot["answer_token_located_rate"] == 1.0
    assert pilot["rel_yes_or_no_present_rate"] == 1.0
    assert pilot["observer_text_vs_v3"]["compared"] > 0
    assert pilot["observer_text_vs_v3"]["parsed_equal_rate"] == 1.0
    level1 = pilot["by_level"]["1"]
    assert {"item_pans", "item_rel", "item_product", "direct_v4"} <= set(level1)
    assert isinstance(pilot["item_rel_mean_monotonic"], bool)


def test_v4_reuses_planner_reader_direct(v4_run) -> None:
    _, t4, _, _ = v4_run
    assert t4.calls["planner_v3"] == 0 and t4.calls["answer_only"] == 0
    assert t4.calls["relevance_v4"] > 0 and t4.calls["observer_v3"] > 0
    assert t4.calls["direct"] > 0  # logprob-enabled direct is a new call


def test_v4_records_store_logprobs(v4_run) -> None:
    v4, _, _, _ = v4_run
    records = list(v4.paths.out("observer_v3_lp").rglob("*.json"))
    assert records and all(read_json(p)["logprobs"] for p in records[:5])
    rel = list(v4.paths.out("relevance_v4").rglob("*.json"))
    assert rel and read_json(rel[0])["parsed"]["text"] in {"Yes", "No"}


def test_v4_devgrid(v4_run) -> None:
    v4, _, _, grid = v4_run
    frame = pd.read_csv(v4.paths.out("dev_threshold_grid.csv"))
    assert {
        "checklist_v4_min",
        "direct_v4",
        "direct_v1_verbalized",
        "checklist_v3_min",
    } <= set(frame["method"])
    assert grid["abort_methods"] == ["checklist_v4_min"]
    sens = read_json(v4.paths.out("dev_sensitivity.json"))
    assert sens["distinct_operating_points"] >= 1 and sens["curve_every_0.05"]
    per_q = pd.read_csv(
        v4.paths.out("dev_l20_per_question.csv"), dtype={"question_id": str}
    )
    assert {"item1_pans", "item1_rel", "checklist_v4_min"} <= set(per_q.columns)
    assert len(per_q) == len(v4.selected("dev"))
    assert not v4.paths.out("frozen_policy.json").exists()
    with pytest.raises(StageError):
        stages_model.freeze(v4)


def test_v4_infer_refuses_when_gates_failed(v4_run) -> None:
    v4, _, _, _ = v4_run
    path = v4.paths.out(f"pilot/pilot_summary_{stages_model.active_tag(v4)}.json")
    original = read_json(path)
    atomic_write_json(path, {**original, "gates_pass": False})
    try:
        with pytest.raises(StageError, match="logprob gates"):
            stages_model.infer(v4, "dev", [1])
    finally:
        atomic_write_json(path, original)
