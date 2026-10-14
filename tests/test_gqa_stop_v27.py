"""Plan §27 (v27): minimal filter, hash-order selection, mixed strict / unified
scoring, θ*, bootstrap, and the full extract -> screen -> infer -> evaluate flow
on the synthetic world (fake ollama)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gqa_stop_helpers import (
    FakeTransport,
    answers_by_text,
    fake_tags_get,
    make_context,
    make_questions,
    write_images,
)

from milestone1.gqa_stop import (
    dataset,
    sampling,
    stages_model,
    v27_data,
    v27_eval,
    v27_infer,
)
from milestone1.gqa_stop.common import StageError, read_json, read_jsonl
from milestone1.gqa_stop.planner import PlanOutcome

REPO_ROOT = Path(__file__).resolve().parents[1]
V27_CONFIG = REPO_ROOT / "configs" / "gqa_stop_v27.json"
V22_CONFIG = REPO_ROOT / "configs" / "gqa_stop_v22.json"
PROMPTS_DIR = REPO_ROOT / "prompts" / "gqa_stop"
SPLIT_IMAGE = 1024  # images < 1024 -> dev file, >= 1024 -> test file
DEV_QUESTIONS = 30  # enough to include 'glass' (fake L20 reader wrong)


# ---- filter ---------------------------------------------------------------------------------
def raw(question: str, answer: str, semantic: list, structural: str = "query") -> dict:
    return {
        "question": question,
        "answer": answer,
        "imageId": "1",
        "semantic": semantic,
        "types": {"structural": structural, "semantic": "attr", "detailed": "x"},
    }


SEM_NEG = [
    {"operation": "select", "dependencies": [], "argument": "cup (1)"},
    {"operation": "filter size", "dependencies": [0], "argument": "not(small)"},
    {"operation": "relate", "dependencies": [1], "argument": "a,on,s (2)"},
    {"operation": "relate", "dependencies": [2], "argument": "b,on,s (3)"},
    {"operation": "query", "dependencies": [3], "argument": "color"},
]


def test_minimal_filter_keeps_what_v22_excluded() -> None:
    minimal = json.loads(V27_CONFIG.read_text())["dataset"]["filter"]
    v22 = json.loads(V22_CONFIG.read_text())["dataset"]["filter"]
    negation = raw("What color is the cup that is not small?", "red", SEM_NEG)
    answer_in_q = raw("What color is the red cup?", "red", SEM_NEG[-1:])
    no_attr_word = raw("Which is the cup?", "red", SEM_NEG[-1:])
    for case in (negation, answer_in_q, no_attr_word):
        assert dataset.classify(case, minimal) == dataset.ELIGIBLE
    assert dataset.classify(negation, v22) == "negation"
    assert dataset.classify(answer_in_q, v22) == "answer_word_in_question"


def test_minimal_filter_still_requires_query_attr_and_target_attribute() -> None:
    minimal = json.loads(V27_CONFIG.read_text())["dataset"]["filter"]
    verify = raw("Is the cup red?", "yes", SEM_NEG[-1:], structural="verify")
    size = raw(
        "How big is the cup?",
        "small",
        [{"operation": "query", "dependencies": [], "argument": "size"}],
    )
    assert dataset.classify(verify, minimal) == "not_structural_query"
    assert dataset.classify(size, minimal) == "final_argument_not_target_attribute"


# ---- selection -------------------------------------------------------------------------------
def record(qid: str, image: str) -> dataset.QuestionRecord:
    return dataset.QuestionRecord(qid, image, f"q{qid}", "color", 0, 2, "d", "red", "")


def test_select_questions_caps_per_image_and_keeps_hash_order() -> None:
    records = [record(str(i), str(i % 3)) for i in range(9)]  # 3 images x 3 q
    groups = sampling.build_groups(records, {}, seed=7)
    chosen = v27_data.select_questions(groups, max_questions=5, max_per_group=2)
    assert len(chosen) == 5
    per_group: dict[str, int] = {}
    for rec, group, rank in chosen:
        per_group[group.group_id] = per_group.get(group.group_id, 0) + 1
        assert group.questions[rank] == rec
    assert max(per_group.values()) == 2
    assert [g.group_id for _, g, _ in chosen][:2] == [groups[0].group_id] * 2
    everything = v27_data.select_questions(groups, None, None)
    assert len(everything) == 9


# ---- signals ---------------------------------------------------------------------------------
def sig(valid=True, resolved=True, answer="red", pans=0.8) -> dict:
    return {"valid": valid, "resolved": resolved, "answer": answer, "pans": pans}


def test_direct_resolved_score_and_ablation_helpers() -> None:
    assert v27_infer.direct_resolved_score(sig()) == 0.8
    assert v27_infer.direct_resolved_score(sig(resolved=False)) == 0.0
    assert v27_infer.direct_resolved_score(sig(answer=" Unknown ")) == 0.0
    assert v27_infer.direct_resolved_score(sig(valid=False)) == 0.0
    plan = PlanOutcome(
        "q",
        "valid",
        [
            {"id": "c1", "question": "a", "depends_on": []},
            {"id": "c2", "question": "b", "depends_on": ["c1"]},
        ],
    )
    items = {"c1": sig(pans=0.9, resolved=False), "c2": sig(pans=0.4)}
    assert v27_infer.raw_pans_min(plan, items) == 0.4
    assert not v27_infer.all_resolved(plan, items)
    assert v27_infer.all_resolved(plan, {"c1": sig(), "c2": sig()})
    failed = PlanOutcome("q", "planner_failure", [])
    assert v27_infer.raw_pans_min(failed, {}) == 0.0
    assert not v27_infer.all_resolved(failed, {})


def test_identity_rows_compare_raw_bytes() -> None:
    plain = {
        "question_id": "1",
        "level": 1,
        "status": "valid",
        "raw": '{"answer":"red"}',
    }
    plain["parsed"] = {"answer": "red"}
    same = {**plain}
    other = {**plain, "raw": '{"answer": "red"}'}
    rows = v27_infer.identity_rows([(plain, same), (plain, other)])
    assert [r["raw_identical"] for r in rows] == [True, False]
    assert [r["answer_identical"] for r in rows] == [True, True]


# ---- scoring by hand -------------------------------------------------------------------------
def hand_arrays() -> v27_eval.EvalArrays:
    scores = np.array(
        [[0.1, 0.6, 0.9, 0.9], [0.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.7, 0.2]]
    )
    reader = np.array(
        [
            [False, False, True, True],
            [True, True, True, True],
            [False, False, True, False],
        ]
    )
    return v27_eval.EvalArrays(
        ["a", "b", "c"],
        ["g1", "g2", "g3"],
        np.array(["color", "color", "material"]),
        np.array([True, True, False]),
        reader,
        {m: scores for m in v27_eval.METHODS},
    )


def test_stop_levels_first_level_at_or_above_theta() -> None:
    scores = hand_arrays().scores["B1"]
    assert v27_eval.stop_levels(scores, 0.5).tolist() == [2, 0, 3]
    assert v27_eval.stop_levels(scores, 0.0).tolist() == [1, 1, 1]
    assert v27_eval.stop_levels(scores, 0.95).tolist() == [0, 0, 0]


def test_summarize_strict_and_unified_by_hand() -> None:
    arrays = hand_arrays()
    low = v27_eval.summarize(arrays, "B1", 0.5)
    assert low["answerable_acc"] == 0.0  # q a stops at L2 (wrong), q b never stops
    assert low["unanswerable_strict"] == 0.0  # q c stops
    assert low["unanswerable_unified"] == 1.0  # ... where the reader is right
    assert low["balanced_strict"] == 0.0
    assert low["balanced_unified"] == 0.5
    assert low["mean_level_all"] == pytest.approx((2 + 4 + 3) / 3)
    assert low["never_stop_answerable"] == 0.5
    assert low["unans_fire_L1_5"] == 1
    high = v27_eval.summarize(arrays, "B1", 0.8)
    assert high["answerable_acc"] == 0.5
    assert high["unanswerable_strict"] == 1.0
    assert high["balanced_strict"] == 0.75
    assert high["pooled_strict"] == pytest.approx(2 / 3)
    assert high["unans_never"] == 1
    a3 = v27_eval.summarize(arrays, "A3", 0.0)
    assert math.isnan(a3["theta"])
    assert a3["answerable_acc"] == 0.0  # A3 evaluated at θ = 1: nothing reaches it


def test_theta_grid_and_theta_star_ties_to_smaller() -> None:
    grid = v27_eval.theta_grid({"start": 0.0, "stop": 1.0, "step": 0.05})
    assert len(grid) == 21 and grid[0] == 0.0 and grid[-1] == 1.0 and grid[1] == 0.05
    table = pd.DataFrame(
        {
            "method": ["B1"] * 3 + ["A3"],
            "theta": [0.1, 0.2, 0.3, float("nan")],
            "balanced_strict": [0.5, 0.7, 0.7, 0.9],
        }
    )
    stars = v27_eval.theta_star(table)
    assert stars["B1"] == 0.2
    assert math.isnan(stars["A3"])


def test_paired_bootstrap_at_own_thetas_by_hand() -> None:
    arrays = hand_arrays()
    thetas = {"P_min": 0.8, "B1": 0.5}
    boot = v27_eval.paired_bootstrap_at(arrays, [("P_min", "B1")], thetas, 30, 1)
    balanced = boot[boot["quantity"] == "balanced_strict"].iloc[0]
    assert balanced["estimate"] == pytest.approx(0.75 - 0.0)
    assert balanced["theta_a"] == 0.8 and balanced["theta_b"] == 0.5


def test_paired_bootstrap_identical_methods_give_zero_difference() -> None:
    arrays = hand_arrays()
    boot = v27_eval.paired_bootstrap(arrays, [("P_min", "B1")], [0.5, 0.8], 50, 1)
    assert set(boot["quantity"]) == set(v27_eval.DIFF_KEYS)
    assert (boot["estimate"] == 0).all()
    assert (boot["ci_low"] == 0).all() and (boot["ci_high"] == 0).all()


# ---- end to end on the synthetic world --------------------------------------------------------
def v27_world(tmp: Path) -> tuple[Path, dict, dict]:
    # the decoys (900xxx) are covered by the filter unit tests; the fake planner
    # cannot plan their non-attribute wording
    questions = {k: v for k, v in make_questions().items() if not k.startswith("900")}
    dev = {k: v for k, v in questions.items() if int(v["imageId"]) < SPLIT_IMAGE}
    test = {k: v for k, v in questions.items() if int(v["imageId"]) >= SPLIT_IMAGE}
    (tmp / "gqa").mkdir(parents=True)
    (tmp / "gqa" / "dev.json").write_text(json.dumps(dev))
    (tmp / "gqa" / "test.json").write_text(json.dumps(test))
    write_images(tmp / "images")
    cfg = json.loads(V27_CONFIG.read_text())
    cfg["paths"] = {
        "repo_root": str(tmp),
        "gqa_questions": str(tmp / "gqa" / "dev.json"),
        "gqa_questions_test": str(tmp / "gqa" / "test.json"),
        "gqa_images_root": str(tmp / "images"),
        "data_root": "data/v27",
        "selection_root": "data/v27",
        "output_root": "outputs/v27",
        "prompts_dir": str(PROMPTS_DIR),
    }
    cfg["dataset"]["dev_questions"] = DEV_QUESTIONS
    cfg["determinism"].update(argmax_questions=4, direct_resolved_questions=2)
    cfg["v27"]["identity_check"] = {"questions": 3, "levels": [1, 20]}
    cfg["evaluation"]["bootstrap_repeats"] = 20
    cfg["model"]["backoff_base_seconds"] = 0.0
    path = tmp / "v27.json"
    path.write_text(json.dumps(cfg))
    return path, cfg, questions


@pytest.fixture(scope="module")
def v27_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("v27")
    path, cfg, questions = v27_world(tmp)
    transport = FakeTransport(None, answers_by_text(questions), questions)  # type: ignore[arg-type]
    ctx = make_context(path, cfg, transport)
    transport.paths = ctx.paths
    stages_model.preflight(ctx, get=fake_tags_get)
    extract = v27_data.extract(ctx)
    with pytest.raises(StageError, match="envcheck"):
        v27_infer.detgate(ctx)
    ctx.state.set("v27_envcheck", {"passed": True})  # no ollama in tests
    v27_infer.detgate(ctx)
    with pytest.raises(StageError, match="identity"):
        v27_infer.screen(ctx, "dev")
    identity = v27_infer.identity(ctx)
    screen = v27_infer.screen(ctx, "dev")
    partial = v27_infer.infer(ctx, "dev", max_seconds=0.0)
    full = v27_infer.infer(ctx, "dev", max_seconds=None)
    dev_eval = v27_eval.evaluate(ctx, "dev")
    with pytest.raises(StageError, match="confirm-test-eval"):
        v27_eval.evaluate(ctx, "test")
    v27_infer.screen(ctx, "test")
    v27_infer.infer(ctx, "test", max_seconds=None)
    audit = v27_infer.audit(ctx, "test")
    test_eval = v27_eval.evaluate(ctx, "test", confirm_test=True)
    return dict(
        ctx=ctx,
        transport=transport,
        extract=extract,
        identity=identity,
        screen=screen,
        partial=partial,
        full=full,
        dev_eval=dev_eval,
        test_eval=test_eval,
        audit=audit,
    )


def test_extract_dev_capped_test_all_and_disjoint(v27_run) -> None:
    ctx = v27_run["ctx"]
    dev = read_jsonl(v27_data.manifest_path(ctx, "dev"))
    test = read_jsonl(v27_data.manifest_path(ctx, "test"))
    assert len(dev) == DEV_QUESTIONS
    assert max(pd.Series([r["image_id"] for r in dev]).value_counts()) <= 2
    assert len(test) == v27_run["extract"]["test"]["eligible"]["questions"]
    assert not {r["image_id"] for r in dev} & {r["image_id"] for r in test}
    assert all("answer" not in r for r in dev + test)  # GT lives in gt_<split>.json
    gt = read_json(v27_data.gt_path(ctx, "dev"))
    assert set(gt) == {r["question_id"] for r in dev}
    with pytest.raises(StageError, match="frozen"):
        v27_data.extract(ctx)


def test_identity_and_screening_labels(v27_run) -> None:
    assert v27_run["identity"]["all_identical"]
    assert v27_run["identity"]["pairs"] == 6
    labels = v27_run["screen"]["labels"]
    assert sum(labels.values()) == DEV_QUESTIONS
    assert set(labels) == {"answerable", "unanswerable"}


def test_infer_is_resumable_and_completes(v27_run) -> None:
    assert not v27_run["partial"]["complete"]
    assert v27_run["partial"]["questions_done"] == 0
    assert v27_run["full"]["complete"]
    ctx = v27_run["ctx"]
    assert len(v27_infer.load_done(ctx, "dev")) == DEV_QUESTIONS
    again = v27_infer.infer(ctx, "dev", max_seconds=None)  # all skipped
    assert again["complete"] and again["new_calls"] == 0 and again["cache_hits"] == 0
    ctx = v27_run["ctx"]
    payloads = v27_run["transport"].payloads
    contents = [p["messages"][0]["content"] for p in payloads]
    assert not any("confident" in c for c in contents)  # no Rel / relevance calls
    assert not any(c.startswith("Answer the question about") for c in contents)
    table = pd.read_csv(
        ctx.paths.out("level_table_dev.csv"), dtype={"question_id": str}
    )
    assert len(table) == DEV_QUESTIONS * 20
    assert table["reader_located"].all()


def test_reader_lp_records_carry_logprobs_and_direct_resolved_uses_original(
    v27_run,
) -> None:
    ctx = v27_run["ctx"]
    lp = list((ctx.paths.output_root / "reader_lp").rglob("*.json"))
    assert lp and all(read_json(p).get("logprobs") for p in lp[:5])
    direct = list((ctx.paths.output_root / "observer_v3_lp_direct").rglob("*.json"))
    manifest = {
        r["question_id"]: r for r in read_jsonl(v27_data.manifest_path(ctx, "dev"))
    }
    record = read_json(direct[0])
    if record["question_id"] in manifest:
        assert (
            record["input_data"]["question"]
            == manifest[record["question_id"]]["question"]
        )
    assert record["prompt_hash"] == ctx.prompts.hashes["observer_v3"]


def test_evaluation_outputs_and_theta_star(v27_run) -> None:
    ctx = v27_run["ctx"]
    sweep = pd.read_csv(ctx.paths.out("eval/dev/sweep.csv"))
    thresholded = [m for m in v27_eval.METHODS if m not in v27_eval.NO_THRESHOLD]
    assert len(sweep) == len(thresholded) * 21 + 1
    zero = sweep[(sweep["theta"] == 0.0) & (sweep["method"] == "B1")].iloc[0]
    assert zero["mean_level_all"] == 1.0  # θ = 0 stops everything at L1
    assert zero["unanswerable_strict"] in (0.0,) or math.isnan(
        zero["unanswerable_strict"]
    )
    stars = v27_run["dev_eval"]["theta_star"]
    assert set(stars) == set(v27_eval.METHODS)
    boot = pd.read_csv(ctx.paths.out("eval/dev/paired_bootstrap.csv"))
    assert set(boot["pair"]) == {"P_min - B1", "P_min - A1"}
    assert len(boot) == 2 * 21 * len(v27_eval.DIFF_KEYS)
    at_star = pd.read_csv(ctx.paths.out("eval/dev/paired_bootstrap_at_theta_star.csv"))
    assert len(at_star) == 2 * len(v27_eval.DIFF_KEYS)
    assert (at_star["theta_a"] == stars["P_min"]).all()
    assert all(math.isfinite(stars[m]) for m in stars if m != "A3")
    assert math.isnan(stars["A3"])
    test_stars = v27_run["test_eval"]["theta_star"]
    assert all(
        test_stars[m] == stars[m]
        or (math.isnan(test_stars[m]) and math.isnan(stars[m]))
        for m in stars
    )
    for name in (
        "balanced_vs_theta",
        "unanswerable_vs_theta",
        "answerable_acc_vs_level",
    ):
        assert ctx.paths.out(f"figures/dev/{name}.png").exists()


def test_audit_counts_records_and_quarantine_moves_nothing_when_clean(v27_run) -> None:
    audit = v27_run["audit"]
    assert audit["records"] > 0 and audit["flagged"] == []
    assert v27_infer.quarantine(v27_run["ctx"], "test") == {"moved": []}


# ---- post-hoc observer-answer methods (plan §27.12) -------------------------------------------
def posthoc_arrays() -> v27_eval.EvalArrays:
    """hand_arrays plus observer correctness: at θ 0.5 q a stops at L2 (reader
    wrong, last item right) and q c (unanswerable) stops at L3 (reader right,
    last item wrong, direct right)."""
    base = hand_arrays()
    last_item = np.zeros_like(base.reader_correct)
    last_item[0, 1] = True
    direct = np.zeros_like(base.reader_correct)
    direct[2, 2] = True
    return v27_eval.EvalArrays(
        base.question_ids,
        base.group_ids,
        base.attributes,
        base.answerable,
        base.reader_correct,
        base.scores,
        {v27_eval.LAST_ITEM: last_item, v27_eval.DIRECT: direct},
    )


def test_posthoc_method_uses_observer_answer_where_reader_differs() -> None:
    arrays = posthoc_arrays()
    reader = v27_eval.summarize(arrays, "P_min", 0.5)
    observer = v27_eval.summarize(arrays, "P'_min", 0.5)
    assert reader["answerable_acc"] == 0.0  # reader wrong at L2
    assert observer["answerable_acc"] == 0.5  # last item right at L2
    assert observer["mean_level_all"] == reader["mean_level_all"]  # same stop rule
    assert observer["unanswerable_strict"] == reader["unanswerable_strict"] == 0.0
    assert v27_eval.base_method("A1'") == "A1"
    assert v27_eval.answer_source("P'_mean") == v27_eval.LAST_ITEM
    assert v27_eval.answer_source("B1") == v27_eval.READER


def test_posthoc_unified_credit_on_unanswerable_needs_observer_correct() -> None:
    arrays = posthoc_arrays()
    last_item = v27_eval.summarize(arrays, "P'_min", 0.5)
    direct = v27_eval.summarize(arrays, "A1'", 0.5)
    reader = v27_eval.summarize(arrays, "A1", 0.5)
    assert last_item["unanswerable_unified"] == 0.0  # stopped, observer wrong
    assert direct["unanswerable_unified"] == 1.0  # stopped, observer right
    assert reader["unanswerable_unified"] == 1.0  # reader right at L3
    assert direct["unanswerable_strict"] == 0.0  # strict: any stop is wrong
    assert direct["balanced_unified"] == pytest.approx((0.0 + 1.0) / 2)


def test_existing_methods_unchanged_by_observer_arrays() -> None:
    plain, extended = hand_arrays(), posthoc_arrays()
    grid = [0.0, 0.5, 0.8, 1.0]
    before = v27_eval.sweep(plain, grid)
    after = v27_eval.sweep(extended, grid)
    pd.testing.assert_frame_equal(before, after)
    assert list(before["method"].unique()) == list(v27_eval.METHODS)
    full = v27_eval.sweep(extended, grid, v27_eval.ALL_METHODS)
    assert list(full["method"].unique()) == list(v27_eval.ALL_METHODS)
    pd.testing.assert_frame_equal(
        full[full["method"].isin(v27_eval.METHODS)].reset_index(drop=True), before
    )
    with pytest.raises(StageError, match="last_item"):
        v27_eval.summarize(plain, "P'_min", 0.5)  # no observer answers available


def test_theta_star_unified_rule_and_posthoc_subset() -> None:
    table = pd.DataFrame(
        {
            "method": ["B1"] * 3,
            "theta": [0.1, 0.2, 0.3],
            "balanced_strict": [0.5, 0.7, 0.6],
            "balanced_unified": [0.8, 0.6, 0.8],
        }
    )
    assert v27_eval.theta_star(table)["B1"] == 0.2
    assert v27_eval.theta_star(table, "balanced_unified")["B1"] == 0.1
    part = posthoc_arrays().subset(np.array([True, False, True]))
    assert part.answer_correct[v27_eval.LAST_ITEM].shape == (2, 4)
    assert part.answer_correct[v27_eval.DIRECT][1, 2]


def test_arrays_from_table_reads_last_item_and_direct_answers() -> None:
    rows = []
    specs = {  # qid -> (n_items, item answers, dres answer, gt)
        "q1": (2, ("yes", "red"), "blue", "red"),
        "q2": (3, ("yes", "no", "Wood"), "wood", "wood"),
        "q3": (0, (), None, "red"),
    }
    for qid, (n_items, items, dres, gt) in specs.items():
        for level in (1, 2):
            row = {
                "question_id": qid,
                "group_id": f"g{qid}",
                "attribute": "color",
                "level": level,
                "label": v27_infer.ANSWERABLE,
                "reader_correct": False,
                "gt_answer": gt,
                "n_items": n_items,
                "dres_answer": dres,
                **{c: 0.0 for c in v27_eval.METHODS.values()},
            }
            for i in range(1, 4):
                row[f"item{i}_answer"] = items[i - 1] if i <= len(items) else None
            rows.append(row)
    arrays = v27_eval.arrays_from_table(pd.DataFrame(rows), 2)
    assert arrays.answer_correct[v27_eval.LAST_ITEM][:, 0].tolist() == [
        True,
        True,
        False,
    ]
    assert arrays.answer_correct[v27_eval.DIRECT][:, 1].tolist() == [
        False,
        True,
        False,
    ]


def test_posthoc_stage_writes_only_eval_posthoc(v27_run) -> None:
    ctx = v27_run["ctx"]
    confirmatory = sorted(ctx.paths.out("eval").rglob("*"))
    before = {p: p.read_bytes() for p in confirmatory if p.is_file()}
    with pytest.raises(StageError, match="dev --posthoc first"):
        v27_eval.evaluate_posthoc(ctx, "test", confirm_test=True)
    dev = v27_eval.evaluate_posthoc(ctx, "dev")
    with pytest.raises(StageError, match="confirm-test-eval"):
        v27_eval.evaluate_posthoc(ctx, "test")
    test = v27_eval.evaluate_posthoc(ctx, "test", confirm_test=True)
    files = [p for p in sorted(ctx.paths.out("eval").rglob("*")) if p.is_file()]
    assert {p: p.read_bytes() for p in files} == before
    stars = read_json(ctx.paths.out("eval_posthoc/dev/theta_star.json"))
    assert set(stars["theta_star"]) == set(v27_eval.SELECTIONS)
    for sel in v27_eval.SELECTIONS:
        assert set(stars["theta_star"][sel]) == set(v27_eval.ALL_METHODS)
    assert all(dev["strict_theta_star_matches_confirmatory"].values())
    for sel, thetas in stars["theta_star"].items():
        for method, theta in thetas.items():
            used = test["theta_star"][sel][method]
            assert used == theta or (math.isnan(used) and math.isnan(theta))
    sweep = pd.read_csv(ctx.paths.out("eval_posthoc/dev/sweep.csv"))
    assert len(sweep) == (len(v27_eval.ALL_METHODS) - 1) * 21 + 1
    assert set(sweep.loc[sweep["method"] == "A1'", "answer_source"]) == {"direct"}
    at_star = pd.read_csv(ctx.paths.out("eval_posthoc/test/at_theta_star.csv"))
    assert len(at_star) == 2 * len(v27_eval.ALL_METHODS)
    pb = pd.read_csv(
        ctx.paths.out("eval_posthoc/test/paired_bootstrap_at_theta_star.csv")
    )
    assert set(pb["pair"]) == {f"{a} - {b}" for a, b in v27_eval.POSTHOC_PAIRS}
    assert len(pb) == 2 * len(v27_eval.POSTHOC_PAIRS) * len(v27_eval.POSTHOC_DIFF_KEYS)


def test_cli_posthoc_flag_only_for_evaluate(v27_run) -> None:
    from milestone1.gqa_stop import cli

    args = cli.build_parser().parse_args(
        ["v27-screen", "--config", "x.json", "--split", "dev", "--posthoc"]
    )
    with pytest.raises(StageError, match="posthoc"):
        cli.dispatch_v27(v27_run["ctx"], args)
