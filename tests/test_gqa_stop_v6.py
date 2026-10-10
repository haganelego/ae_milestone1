"""Plan §25 v6 helpers: stop rules with gates, θ selection, folds, agreement,
legibility parsing and the v1-gated v6 scores (no model, no dataset)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from milestone1.gqa_stop import v6_agree as va
from milestone1.gqa_stop import v6_eval as ev
from milestone1.gqa_stop import v6_legibility as vl
from milestone1.gqa_stop.client import ROLE_DIRS, PromptSet, RequestEngine
from milestone1.gqa_stop.planner import PlanOutcome

PROMPTS = Path(__file__).resolve().parents[1] / "prompts" / "gqa_stop"


def lp(token: str, logprob: float, alts: list[tuple[str, float]]) -> dict:
    return {
        "token": token,
        "logprob": logprob,
        "top_logprobs": [{"token": t, "logprob": v} for t, v in alts],
    }


# ---- stop rules ---------------------------------------------------------------------------
def test_stops_k1_respects_gate_and_forced_last_level() -> None:
    score = np.array([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]])
    gate = np.array([[False, True, True], [True, True, True]])
    level, fired = ev.stops(score, gate, 0.5)
    assert level.tolist() == [2, 3] and fired.tolist() == [True, False]
    level, fired = ev.stops(score, None, 0.5)
    assert level.tolist() == [1, 3]


def test_stops_k2_both_vs_either_gate_mode() -> None:
    score = np.array([[0.9, 0.9, 0.9, 0.9]])
    gate = np.array([[True, False, True, True]])
    both, _ = ev.stops(score, gate, 0.5, k=2, mode=ev.GATE_BOTH)
    either, _ = ev.stops(score, gate, 0.5, k=2, mode=ev.GATE_EITHER)
    assert both.tolist() == [4]  # levels 3 and 4 both agree
    assert either.tolist() == [2]  # level 1 agrees, level 2 does not
    never, fired = ev.stops(np.array([[0.9, 0.1, 0.9, 0.1]]), None, 0.5, k=2)
    assert never.tolist() == [4] and fired.tolist() == [False]


def test_fired_at_last_level_counts_as_firing() -> None:
    level, fired = ev.stops(np.array([[0.1, 0.1, 0.9]]), None, 0.5)
    assert level.tolist() == [3] and fired.tolist() == [True]


def arrays(score: np.ndarray, correct: np.ndarray, groups: list[str]) -> ev.SetArrays:
    n = score.shape[0]
    return ev.SetArrays(
        qids=[f"q{i}" for i in range(n)],
        groups=groups,
        correct=correct,
        attribute=np.array(["color"] * n),
        relate=np.zeros(n, dtype=int),
        columns={"s": score},
    )


def test_select_index_minimises_mean_level_at_target() -> None:
    levels = ev.N_LEVELS
    score = np.tile(np.round(np.linspace(0.05, 1.0, levels), 2), (4, 1))
    correct = np.zeros((4, levels), dtype=bool)
    correct[:, 9:] = True  # correct from level 10 (score 0.5)
    arr = arrays(score, correct, ["a", "a", "b", "c"])
    gs = ev.grid_stops(arr, ev.Policy("s", "s"))
    i = ev.select_index(gs, 0.95)
    assert i is not None and ev.GRID[i] == 0.5
    assert gs.level[i].tolist() == [10] * 4
    # infeasible targets are impossible here because L20 is correct; θ=∞ stays last
    assert math.isinf(ev.GRID[-1])


def test_union_folds_keep_images_together_across_sets() -> None:
    dev_groups = ["a", "a", "b", "c", "d"]
    wrong_groups = ["a", "e", "c"]
    fd, fw = ev.union_folds(dev_groups, wrong_groups, seed=3, k=2)
    assert fd[0] == fd[1] == fw[0]
    assert fd[3] == fw[2]
    assert set(fd.tolist()) | set(fw.tolist()) <= {0, 1}


def test_pass_rule_needs_all_three_conditions() -> None:
    oof = pd.DataFrame(
        [
            {
                "method": "m",
                "target": 0.95,
                "oof_accuracy": 0.94,
                "oof_mean_level": 14.0,
                "oof_firing_99": 0.3,
            },
            {
                "method": "direct_pans",
                "target": 0.95,
                "oof_accuracy": 0.95,
                "oof_mean_level": 13.6,
                "oof_firing_99": 0.46,
            },
        ]
    )
    pairs = pd.DataFrame(
        [
            {
                "method": "m",
                "vs": "direct_pans",
                "target": 0.95,
                "d_firing_ci": [-0.2, -0.01],
            }
        ]
    )
    ok = ev.pass_rule(oof, pairs, "m", ["direct_pans"])
    assert ok["passed"]
    slow = ev.pass_rule(
        oof.assign(oof_mean_level=[14.2, 13.6]), pairs, "m", ["direct_pans"]
    )
    assert not slow["passed"]
    wide = pairs.assign(d_firing_ci=[[-0.2, 0.01]])
    assert not ev.pass_rule(oof, wide, "m", ["direct_pans"])["passed"]


# ---- V6-A ---------------------------------------------------------------------------------
def test_agreement_flags_normalise_but_do_not_merge_synonyms() -> None:
    flags = va.agreement_flags(
        ["gray", "wood", "gray", None, "red"],
        ["Gray ", "wooden", "grey", None, float("nan")],
    )
    assert flags.tolist() == [True, False, False, False, False]


def test_value_token_entries_skip_grammar_forced_structure() -> None:
    logprobs = [
        lp('{"', -5.7, [("Gray", -0.1), ('{"', -5.7)]),  # grammar-forced, not argmax
        lp("answer", 0.0, [("answer", 0.0)]),
        lp('":"', -0.01, [('":"', -0.01)]),
        lp("gray", -0.1, [("gray", -0.1), ("Gray", -2.8)]),
        lp('"}', 0.0, [('"}', 0.0)]),
    ]
    entries = va.value_token_entries(logprobs)
    assert [e["token"] for e in entries] == ["gray"]
    from milestone1.gqa_stop.stages_v4c import argmax_violations

    assert argmax_violations(entries) == (1, 0)
    assert argmax_violations(logprobs)[1] == 1


def test_union_bootstrap_diff_sign_with_shared_images() -> None:
    a = {"q1": 1.0, "q2": 1.0, "q3": 0.0}
    b = {"q4": 0.0, "q5": 0.0}
    groups = {"q1": "x", "q2": "y", "q3": "z", "q4": "x", "q5": "w"}
    out = va.union_bootstrap_diff(a, b, groups, repeats=200, seed=1)
    assert math.isclose(out["diff"], 2 / 3)
    assert out["ci_low"] <= out["diff"] <= out["ci_high"]


def test_second_reader_records_go_to_their_own_role_dir(tmp_path: Path) -> None:
    from milestone1.gqa_stop.client import ModelSettings

    settings = ModelSettings(
        endpoint="http://x/api/chat",
        name="gemma3:latest",
        digest="a2af6cc3eb7fabcdef",
        temperature=0.0,
        seed=1,
        think=False,
        json_format=True,
        num_ctx=8192,
        num_predict={"answer_only": 128},
        timeout_s=1,
        transport_retries=0,
        schema_repair_attempts=1,
    )
    texts = {"answer_only": "Answer.", "schema_repair": "Fix."}
    engine = RequestEngine(settings, PromptSet(texts), None, tmp_path)  # type: ignore[arg-type]
    spec = va.second_reader_spec(
        {"question_id": "q1", "question": "What color?"}, 5, b"img", "gemma3"
    )
    path = engine.record_path(spec)
    assert path.relative_to(tmp_path).parts[:2] == (
        "reader_gemma3",
        "gemma3-latest@a2af6cc3eb7f",
    )
    assert spec.logprobs == 10 and not spec.raw_text


def test_agree_policies_have_one_primary_and_ungated_comparators() -> None:
    policies = va.agree_policies()
    primary = [p for p in policies if p.role == "primary"]
    assert [p.name for p in primary] == [va.PRIMARY_POLICY]
    comps = va.agree_comparators(policies)
    assert comps[va.PRIMARY_POLICY][:2] == ["direct_pans", "direct_v4"]
    assert "pans_min" in comps["pans_min+agree_qwen4b"]
    assert "direct_pans" not in comps


# ---- V6-B ---------------------------------------------------------------------------------
def test_leg_signal_uses_exact_canonical_tokens() -> None:
    record = {
        "status": "valid",
        "parsed": {"text": "Enough"},
        "logprobs": [
            lp(
                "Enough",
                math.log(0.6),
                [
                    ("Enough", math.log(0.6)),
                    ("Closer", math.log(0.3)),
                    ("enough", math.log(0.05)),
                ],
            )
        ],
    }
    sig = vl.leg_signal(record, "Enough", "Closer")
    assert math.isclose(sig["leg"], 0.6) and math.isclose(sig["p_closer"], 0.3)
    assert sig["has_option"] and sig["starts_with_option"]
    other = vl.leg_signal(record, "Enough_", "Closer_")
    assert other["leg"] == 0.0 and not other["has_option"]
    bad = vl.leg_signal({"status": "schema_failure"}, "Enough", "Closer")
    assert bad["leg"] == 0.0 and not bad["starts_with_option"]


def test_canonical_tokens_and_reparse() -> None:
    frame = pd.DataFrame(
        {
            "text": ["Enough", "Enough.", "Closer", "Hmm"],
            "first_token": ["Enough", "Enough", "Clo", "H"],
            "first_probs": [json.dumps({"Enough": 0.7, "Clo": 0.2})] * 4,
        }
    )
    tokens = vl.canonical_tokens(frame)
    assert tokens["Enough"]["token"] == "Enough" and tokens["Closer"]["token"] == "Clo"
    out = vl.reparse(frame, ("Enough", "Clo"))
    assert out["leg"].tolist() == [0.7] * 4 and out["has_option"].all()


def test_legibility_prompt_is_inline_and_role_dirs_registered(tmp_path: Path) -> None:
    from milestone1.gqa_stop.client import ModelSettings

    assert ROLE_DIRS[vl.ROLE] == "legibility_v6" and vl.SWAP_ROLE in ROLE_DIRS
    template = (PROMPTS / "legibility_v6.txt").read_text("utf-8")
    settings = ModelSettings(
        "http://x/api/chat",
        "qwen3.5:9b",
        "d" * 64,
        0.0,
        1,
        False,
        True,
        8192,
        {vl.ROLE: 8},
        1,
        0,
        1,
    )
    engine = RequestEngine(
        settings, PromptSet({vl.ROLE: template, "schema_repair": "x"}), None, tmp_path
    )  # type: ignore[arg-type]
    spec = vl.leg_spec("q1", 3, b"img", "What color is the cup?", "direct")
    content = engine.build_messages(spec)[0]["content"]
    assert "'What color is the cup?'" in content and "Enough or Closer" in content
    payload = engine.build_payload(spec, engine.build_messages(spec))
    assert "format" not in payload and payload["top_logprobs"] == 10
    assert engine.record_path(spec).name == "03_direct.json"


def plan(*items: tuple[str, list[str]]) -> PlanOutcome:
    return PlanOutcome(
        "q1",
        "valid",
        [{"id": i, "question": f"item {i}?", "depends_on": deps} for i, deps in items],
    )


def test_gated_min_mean_zeroes_unresolved_ancestor() -> None:
    p = plan(("i1", []), ("i2", ["i1"]))
    items = {
        "i1": {"valid": True, "answer": "unknown", "resolved": True, "x": 0.9},
        "i2": {"valid": True, "answer": "red", "resolved": True, "x": 0.8},
    }
    assert vl.gated_min_mean(p, items, "x") == (0.0, 0.0)
    items["i1"]["answer"] = "yes"
    assert vl.gated_min_mean(p, items, "x") == (0.8, (0.9 + 0.8) / 2)


def test_add_v6_scores_products_and_v4_recompute() -> None:
    p = plan(("i1", []), ("i2", ["i1"]))
    table = pd.DataFrame(
        [
            {
                "question_id": "q1",
                "level": 1,
                "direct_pans": 0.5,
                "direct_rel": 0.4,
                "checklist_v4_min": 0.2 * 0.5,
                **{
                    f"item{i}_{k}": v
                    for i, vals in ((1, ("yes", 0.5, 0.5)), (2, ("red", 0.4, 0.5)))
                    for k, v in zip(("answer", "pans", "rel"), vals, strict=True)
                },
                "item1_valid": True,
                "item2_valid": True,
                "item1_resolved": True,
                "item2_resolved": True,
            }
        ]
    )
    leg = pd.DataFrame(
        [
            {"question_id": "q1", "level": 1, "target": "direct", "leg": 0.8},
            {"question_id": "q1", "level": 1, "target": "i1", "leg": 0.9},
            {"question_id": "q1", "level": 1, "target": "i2", "leg": 0.5},
        ]
    )
    out = vl.add_v6_scores(table, leg, {"q1": p}).iloc[0]
    assert math.isclose(out["direct_v6"], 0.4)
    assert math.isclose(out["direct_v6rel"], 0.16)
    assert math.isclose(out["checklist_v6_min"], min(0.5 * 0.9, 0.4 * 0.5))
    assert math.isclose(out["leg_min"], 0.5)
    assert math.isclose(out["checklist_v4_min_recomputed"], min(0.25, 0.2))


# ---- V6-C ---------------------------------------------------------------------------------
from milestone1.gqa_stop import v6_verify as vv  # noqa: E402


def v6c_engine(tmp_path: Path, role: str) -> RequestEngine:
    from milestone1.gqa_stop.client import ModelSettings

    settings = ModelSettings(
        "http://x/api/chat",
        "qwen3.5:9b",
        "d" * 64,
        0.0,
        1,
        False,
        True,
        8192,
        {role: 8},
        1,
        0,
        1,
    )
    template = (PROMPTS / f"{role}.txt").read_text("utf-8")
    return vv.TemplateEngine(
        settings, PromptSet({role: template, "schema_repair": "x"}), None, tmp_path
    )  # type: ignore[arg-type]


def test_template_engine_fills_question_answer_and_attribute(tmp_path: Path) -> None:
    engine = v6c_engine(tmp_path, vv.VERIFY)
    spec = vv.yesno_spec(
        vv.VERIFY,
        "q1",
        4,
        b"img",
        {"question": "What color is the cup?", "answer": "red"},
        "direct",
    )
    content = engine.build_messages(spec)[0]["content"]
    assert (
        "'What color is the cup?'" in content
        and "'red'" in content
        and "{" not in content
    )
    assert engine.record_path(spec).relative_to(tmp_path).parts[0] == "verify_v6"
    vis = v6c_engine(tmp_path, vv.VISIBLE)
    spec = vv.yesno_spec(
        vv.VISIBLE,
        "q1",
        4,
        b"img",
        {"question": "Q?", "attribute": "material"},
        "direct",
    )
    assert "tell its material?" in vis.build_messages(spec)[0]["content"]
    swap = (PROMPTS / "verify_v6_swap.txt").read_text("utf-8")
    assert swap.endswith("Answer with No or Yes.") or swap.strip().endswith(
        "Answer with No or Yes."
    )


def test_level_targets_only_resolved_known_items() -> None:
    p = plan(("i1", []), ("i2", ["i1"]), ("i3", []))
    row = {
        "question_id": "q1",
        "question": "What color is the cup?",
        "direct_lp_answer": " Red ",
        "item1_valid": True,
        "item1_resolved": True,
        "item1_answer": "yes",
        "item1_question": "item i1?",
        "item2_valid": True,
        "item2_resolved": True,
        "item2_answer": "Unknown",
        "item2_question": "item i2?",
        "item3_valid": True,
        "item3_resolved": False,
        "item3_answer": "blue",
        "item3_question": "item i3?",
    }
    targets = vv.level_targets(row, p, vv.VERIFY, "color")
    assert [(t, d["answer"]) for t, d in targets] == [("direct", "red"), ("i1", "yes")]
    vis = vv.level_targets(row, p, vv.VISIBLE, "color")
    assert vis == [
        ("direct", {"question": "What color is the cup?", "attribute": "color"})
    ]


def test_auroc_ranks_and_ties() -> None:
    labels = np.array([True, True, False, False])
    assert vv.auroc(np.array([0.9, 0.8, 0.1, 0.2]), labels) == 1.0
    assert vv.auroc(np.array([0.1, 0.2, 0.9, 0.8]), labels) == 0.0
    assert vv.auroc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == 0.5
    assert np.isnan(vv.auroc(np.array([0.5]), np.array([True])))


def test_add_v6c_scores_uncalled_targets_are_zero() -> None:
    p = plan(("i1", []), ("i2", ["i1"]))
    table = pd.DataFrame(
        [
            {
                "question_id": "q1",
                "level": 2,
                "direct_pans": 0.5,
                "direct_rel": 0.4,
                "item1_valid": True,
                "item1_resolved": True,
                "item1_answer": "yes",
                "item1_pans": 0.8,
                "item1_rel": 0.5,
                "item2_valid": True,
                "item2_resolved": True,
                "item2_answer": "red",
                "item2_pans": 0.6,
                "item2_rel": 0.5,
            }
        ]
    )
    verify = pd.DataFrame(
        [
            {"question_id": "q1", "level": 2, "target": "direct", "p_yes": 0.9},
            {"question_id": "q1", "level": 2, "target": "i1", "p_yes": 0.5},
        ]
    )
    out = vv.add_v6c_scores(table, verify, None, {"q1": p}).iloc[0]
    assert math.isclose(out["direct_v6t"], 0.45) and out["direct_v6v"] == 0.0
    assert math.isclose(out["ptrue_rel_direct"], 0.36)
    assert out["checklist_v6t_min"] == 0.0  # i2 was not verified -> P(True) = 0
    assert math.isclose(out["checklist_v6t_mean"], 0.2)


def test_band_rates_and_bonferroni_ci_key() -> None:
    fired = np.array([True, True, True, False])
    level = np.array([3, 8, 20, 20])
    bands = ev.band_rates(fired, level)
    assert bands["fire_L1_5"] == 0.25 and bands["fire_L6_10"] == 0.25
    assert bands["fire_L6_9"] == 0.25 and bands["fire_L20_20"] == 0.25
    oof = pd.DataFrame(
        [
            {
                "method": "m",
                "target": 0.95,
                "oof_accuracy": 0.95,
                "oof_mean_level": 13.0,
                "oof_firing_99": 0.3,
            },
            {
                "method": "direct_pans",
                "target": 0.95,
                "oof_accuracy": 0.95,
                "oof_mean_level": 13.0,
                "oof_firing_99": 0.4,
            },
        ]
    )
    pairs = pd.DataFrame(
        [
            {
                "method": "m",
                "vs": "direct_pans",
                "target": 0.95,
                "d_firing_ci": [-0.2, -0.01],
                "d_firing_ci975": [-0.22, 0.01],
            }
        ]
    )
    assert ev.pass_rule(oof, pairs, "m", ["direct_pans"])["passed"]
    assert not ev.pass_rule(oof, pairs, "m", ["direct_pans"], ci_key="d_firing_ci975")[
        "passed"
    ]
