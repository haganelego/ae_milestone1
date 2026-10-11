"""Plan §26 model-swap replication: the swapped model reuses the reference run's
sets without re-screening or freezing, and the descriptive sweep is written
beside the reference model's sweep."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from gqa_stop_helpers import PROMPTS_DIR, fake_tags_get, make_context
from test_gqa_stop_v4c import v4c_run  # noqa: F401  (module-scoped fixture)

from milestone1.gqa_stop import stages_model, stages_v4c, sweep_swap, sweep_v22
from milestone1.gqa_stop.cli import build_parser
from milestone1.gqa_stop.client import (
    RequestEngine,
    load_prompts,
    parse_response,
    settings_from_config,
    strip_trailing_periods,
)
from milestone1.gqa_stop.common import StageError, read_id_csv, read_json
from milestone1.gqa_stop.inference import build_level_spec
from milestone1.gqa_stop.sampled import planner_v3_spec
from milestone1.gqa_stop.stages_v22_test import infer_test_wrong

LEVELS = list(range(1, 21))
# reference frozen θ for the transfer rows; on the main grid so they can be checked
REF_FROZEN = {
    "direct_v4": 0.5,
    "checklist_v4_min": 0.55,
    "direct_pans": 0.6,
    "pans_min": 0.65,
}
SWAP_MODEL, REF_MODEL = "qwen3.5:4b", "qwen3.5:9b"


# ---- preconditions (stub context) --------------------------------------------------------------
class StubState:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def get(self, key: str) -> Any:
        return self.values.get(key)


def stub_ctx(swap: dict | None, determinism: bool = True) -> SimpleNamespace:
    def frozen() -> None:
        raise StageError("frozen check reached")

    return SimpleNamespace(
        model_swap=swap,
        settings=SimpleNamespace(decoding_options=(("temperature", 0.0),)),
        state=StubState({"determinism": {"passed": determinism}}),
        require_frozen_match=frozen,
    )


def test_model_swap_test_infer_skips_screening_and_frozen_checks() -> None:
    stages_model.check_infer_preconditions(stub_ctx({"plan": "§26"}), "test")


def test_model_swap_still_requires_the_determinism_gate() -> None:
    with pytest.raises(StageError, match="detgate"):
        stages_model.check_infer_preconditions(stub_ctx({"plan": "§26"}, False), "test")


def test_model_swap_refuses_dev_inference() -> None:
    with pytest.raises(StageError, match="test sets only"):
        stages_model.check_infer_preconditions(stub_ctx({"plan": "§26"}), "dev")


def test_ordinary_run_keeps_the_screening_check() -> None:
    with pytest.raises(StageError, match="screen"):
        stages_model.check_infer_preconditions(stub_ctx(None), "test")


# ---- pure frames -------------------------------------------------------------------------------
def sweep_rows(model_values: dict[tuple[str, float], float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"method": m, "theta": t, "accuracy": v} for (m, t), v in model_values.items()]
    )


def test_side_by_side_labels_models_and_keeps_shared_rows() -> None:
    swap = sweep_rows({("direct_v4", 0.5): 0.9, ("direct_v4", 0.55): 0.95})
    ref = sweep_rows(
        {("direct_v4", 0.5): 0.98, ("direct_v4", 0.55): 0.99, ("pans_min", 0.5): 0.8}
    )
    out = sweep_swap.side_by_side(swap, ref, "4b", "9b")
    assert sorted(out["model"].value_counts().items()) == [("4b", 2), ("9b", 2)]
    assert set(out[out["model"] == "9b"]["method"]) == {"direct_v4"}
    assert len(ref) == 3  # input not mutated


def test_side_by_side_requires_every_swap_row_in_the_reference() -> None:
    swap = sweep_rows({("direct_v4", 0.5): 0.9, ("pans_min", 0.5): 0.9})
    ref = sweep_rows({("direct_v4", 0.5): 0.98})
    with pytest.raises(StageError, match="reference sweep lacks"):
        sweep_swap.side_by_side(swap, ref, "4b", "9b")


def test_fixed_frame_by_hand() -> None:
    answerable = pd.DataFrame(
        {"level": [1, 1, 2, 2], "reader_correct": [False, True, True, True]}
    )
    wrong = pd.DataFrame({"level": [1, 2], "reader_correct": [False, True]})
    frame = sweep_swap.fixed_frame(answerable, wrong, "4b").set_index("level")
    assert frame.loc[1, "answerable_reader_accuracy"] == 0.5
    assert frame.loc[2, "answerable_reader_accuracy"] == 1.0
    assert frame.loc[2, "reader_wrong_reader_accuracy"] == 1.0
    assert set(frame["model"]) == {"4b"}


# ---- synthetic end-to-end run ------------------------------------------------------------------
def swap_config(base: dict) -> dict:
    cfg = json.loads(json.dumps(base))
    cfg["paths"]["output_root"] = "outputs/run_swap"
    cfg["paths"]["reuse_records_from"] = []
    cfg["model"]["name"] = SWAP_MODEL
    cfg["model"]["fallback_name"] = SWAP_MODEL
    cfg["model_swap"] = {
        "plan": "§26",
        "reference_output_root": "outputs/ref_run",
        "reference_model": REF_MODEL,
    }
    return cfg


def write_reference(ctx, root: Path) -> pd.DataFrame:
    """A reference run made of the swap run's own tables (same fake model)."""
    out = ctx.paths.output_root
    (root / "ext_test_wrong").mkdir(parents=True, exist_ok=True)
    (root / "sweep_v22").mkdir(parents=True, exist_ok=True)
    shutil.copy(out / "level_table_test.csv", root / "level_table_test_merged.csv")
    shutil.copy(
        out / "ext_test_wrong/level_table.csv", root / "ext_test_wrong/level_table.csv"
    )
    frame = sweep_v22.sweep_frame(
        read_id_csv(root / "level_table_test_merged.csv"),
        read_id_csv(root / "ext_test_wrong/level_table.csv"),
        "test",
        ctx.n_levels,
    )
    frame.to_csv(root / "sweep_v22/test_sweep.csv", index=False)
    selections = {
        "0.95": {m: {"theta": t} for m, t in REF_FROZEN.items()},
    }
    (root / "frozen_policy.json").write_text(json.dumps({"selections": selections}))
    return frame


@pytest.fixture(scope="module")
def swap_run(v4c_run):  # noqa: F811
    base_ctx = v4c_run["ctx"]
    tmp = base_ctx.paths.repo_root
    base = json.loads(Path(base_ctx.config_path).read_text())
    path = tmp / "swap.json"
    path.write_text(json.dumps(swap_config(base)))
    ctx = make_context(path, {}, v4c_run["transport"])
    ctx.skip_v1arm = True
    stages_model.preflight(ctx, get=fake_tags_get)
    with pytest.raises(StageError, match="detgate"):
        stages_model.infer(ctx, "test", LEVELS)
    stages_v4c.detgate(ctx)
    with pytest.raises(StageError, match="swap-provenance"):
        sweep_swap.check_provenance(ctx)
    provenance = sweep_swap.swap_provenance(ctx)
    with pytest.raises(StageError, match="recorded once"):
        sweep_swap.swap_provenance(ctx)
    stages_model.infer(ctx, "test", LEVELS)
    infer_test_wrong(ctx)
    reference = write_reference(ctx, tmp / "outputs/ref_run")
    ctx.skip_v1arm = False
    with pytest.raises(StageError, match="skip-v1arm"):
        sweep_swap.sweep_swap(ctx)
    ctx.skip_v1arm = True
    main = sweep_swap.sweep_swap(ctx)
    main_files = sorted(p.name for p in ctx.paths.out(sweep_swap.OUT_DIR).iterdir())
    ctx.theta_grid = "nope"
    with pytest.raises(StageError, match="theta-grid"):
        sweep_swap.sweep_swap(ctx)
    ctx.theta_grid = "full"
    meta = sweep_swap.sweep_swap(ctx)
    ctx.theta_grid = "main"
    return dict(
        ctx=ctx,
        provenance=provenance,
        reference=reference,
        meta=meta,
        main=main,
        main_files=main_files,
    )


def test_swap_run_uses_the_swapped_model_only(swap_run) -> None:
    ctx = swap_run["ctx"]
    assert ctx.settings.name == SWAP_MODEL
    tags = {p.name for p in ctx.paths.output_root.glob("*/*") if "@" in p.name}
    assert tags and all(t.startswith("qwen3.5-4b@") for t in tags)
    assert not (ctx.paths.output_root / "frozen_policy.json").exists()
    assert swap_run["provenance"]["fallback_name"] == SWAP_MODEL


def test_swap_sweep_matches_reference_built_from_the_same_records(swap_run) -> None:
    ctx = swap_run["ctx"]
    out = ctx.paths.out(sweep_swap.OUT_DIR)
    mine = pd.read_csv(out / "test_sweep.csv")
    cols = ["method", "theta", "accuracy", "mean_level", "median_level", "firing_rate"]
    pd.testing.assert_frame_equal(mine[cols], swap_run["reference"][cols])
    combined = pd.read_csv(out / "test_sweep_vs_reference.csv")
    assert set(combined["model"]) == {SWAP_MODEL, REF_MODEL}
    n = len(sweep_v22.SWEEP_METHODS) * len(sweep_v22.SWEEP_THETAS)
    assert len(combined) == 2 * n


def test_swap_outputs_are_descriptive_and_complete(swap_run) -> None:
    ctx, meta = swap_run["ctx"], swap_run["meta"]
    assert meta["label"].startswith("DESCRIPTIVE")
    out = ctx.paths.out(sweep_swap.OUT_DIR)
    fixed = pd.read_csv(out / "fixed_curve.csv")
    assert set(fixed["model"]) == {SWAP_MODEL, REF_MODEL}
    assert sorted(fixed["level"].unique()) == LEVELS
    l20 = fixed[(fixed["model"] == SWAP_MODEL) & (fixed["level"] == 20)]
    assert (
        meta["l20_reader_accuracy_answerable"]
        == l20["answerable_reader_accuracy"].iloc[0]
    )
    for name in meta["figures"]:
        assert (out / "figures" / name).exists()
    assert read_json(out / "sweep_meta.json")["k"] == 1


def test_swap_sweep_refuses_changed_sets(swap_run) -> None:
    ctx = swap_run["ctx"]
    path = ctx.paths.out(sweep_swap.PROVENANCE_FILE)
    original = path.read_text()
    record = json.loads(original)
    record["sets"]["answerable"]["n"] += 1
    path.write_text(json.dumps(record))
    try:
        with pytest.raises(StageError, match="provenance changed"):
            sweep_swap.check_provenance(ctx)
    finally:
        path.write_text(original)


# ---- plan §26 trailing-period tolerance (config-gated) -------------------------------------
PLAN = '{"items":[{"id":"c1","question":"Is there a cup?","depends_on":[]}]}'


def accept(obj: Any) -> tuple[Any, list[str]]:
    return obj, []


@pytest.mark.parametrize("tail", [".", ". ", "..", ".\n", " .", " .\n", "...  "])
def test_trailing_periods_are_removed_when_enabled(tail: str) -> None:
    result = parse_response(PLAN + tail, accept, trailing_period=True)
    assert result.errors == [] and result.trailing_period_stripped
    assert result.parsed["items"][0]["id"] == "c1"


def test_trailing_period_fails_with_the_strict_parser() -> None:
    result = parse_response(PLAN + ".", accept)
    assert result.errors and not result.trailing_period_stripped


@pytest.mark.parametrize(
    "raw",
    [
        PLAN,  # nothing to strip
        PLAN[:-1] + ".",  # missing closing brace: still a failure
        PLAN + "]",  # other trailing text is not tolerated
        PLAN + ". extra",
        PLAN + "x",
        PLAN + PLAN,  # a second object is not a trailing period
    ],
)
def test_tolerance_strips_only_periods_after_a_closing_brace(raw: str) -> None:
    text, stripped = strip_trailing_periods(raw)
    assert not stripped and text == raw


def settings_with(roles: list[str] | None) -> Any:
    cfg = json.loads(Path("configs/gqa_stop_v22_4b.json").read_text())
    if roles is None:
        cfg.pop("model_swap")
    else:
        cfg["model_swap"]["trailing_period_roles"] = roles
    return settings_from_config(cfg)


def test_tolerance_changes_the_cache_key_of_listed_roles_only(tmp_path: Path) -> None:
    prompts = load_prompts(PROMPTS_DIR, ["planner_v3", "answer_only", "schema_repair"])
    q = {"question_id": "q", "image_id": "1", "question": "What color is the cup?"}
    specs = [planner_v3_spec(q), build_level_spec("answer_only", q, 3, b"img")]
    strict, off, tolerant = (
        RequestEngine(settings_with(r), prompts, None, tmp_path)
        for r in (None, [], ["planner_v3"])
    )
    assert strict.cache_key(specs[0]) == off.cache_key(specs[0])
    assert tolerant.cache_key(specs[0]) != strict.cache_key(specs[0])
    assert tolerant.cache_key(specs[1]) == strict.cache_key(specs[1])


@pytest.mark.parametrize("raw", [PLAN + "x", PLAN + PLAN, PLAN + PLAN + "."])
def test_tolerant_parser_still_rejects_other_trailing_content(raw: str) -> None:
    result = parse_response(raw, accept, trailing_period=True)
    assert result.errors and result.parsed is None


def test_tolerance_leaves_the_parsed_content_unchanged() -> None:
    inner = '{"items":[{"id":"c1","question":"Is it a cup.","depends_on":[]}]}'
    strict = parse_response(inner, accept)
    tolerant = parse_response(inner + ".", accept, trailing_period=True)
    assert tolerant.parsed == strict.parsed and not strict.errors


def test_theta_grid_option_defaults_to_main() -> None:
    parser = build_parser()
    base = ["sweep-swap", "--config", "x.json"]
    assert parser.parse_args(base).theta_grid == "main"
    assert parser.parse_args([*base, "--theta-grid", "full"]).theta_grid == "full"
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--theta-grid", "other"])


def test_main_grid_run_writes_no_full_grid(swap_run) -> None:
    assert swap_run["main"]["theta_grid"] == "main"
    assert "full_grid_meta" not in swap_run["main"]
    assert sweep_swap.FULL_CSV not in swap_run["main_files"]
    assert sweep_swap.TRANSFER_CSV not in swap_run["main_files"]


def test_best_accuracy_by_hand() -> None:
    frame = pd.DataFrame(
        {
            "model": ["4b"] * 3 + ["9b"] * 2,
            "method": ["direct_v4"] * 5,
            "theta": [0.1, 0.2, 0.3, 0.1, 0.2],
            "accuracy": [0.80, 0.88, 0.88, 0.95, 0.90],
            "mean_level": [10.0, 15.0, 17.0, 12.0, 14.0],
        }
    )
    best = sweep_swap.best_accuracy(frame).set_index("model")
    assert best.loc["4b", "theta"] == 0.2 and best.loc["4b", "mean_level"] == 15.0
    assert best.loc["9b", "theta"] == 0.1 and best.loc["9b", "accuracy"] == 0.95


def test_supplementary_full_grid_is_written(swap_run) -> None:
    ctx, meta = swap_run["ctx"], swap_run["meta"]
    out = ctx.paths.out(sweep_swap.OUT_DIR)
    supp = read_json(out / meta["full_grid_meta"])
    assert supp["label"].startswith("SUPPLEMENTARY")
    assert supp["thetas"][0] == 0.05 and supp["thetas"][-1] == 1.0
    assert len(supp["thetas"]) == 20
    frame = pd.read_csv(out / sweep_swap.FULL_CSV)
    assert len(frame) == 2 * len(sweep_v22.SWEEP_METHODS) * 20
    assert set(frame["model"]) == {SWAP_MODEL, REF_MODEL}
    for column in ("accuracy", "mean_level", "q1_level", "median_level", "q3_level"):
        assert column in frame
    assert all((out / "figures" / name).exists() for name in supp["figures"])
    best = pd.read_csv(out / sweep_swap.BEST_CSV)
    assert len(best) == 2 * len(sweep_v22.SWEEP_METHODS)


def test_theta_transfer_rows_match_the_main_sweep(swap_run) -> None:
    ctx = swap_run["ctx"]
    out = ctx.paths.out(sweep_swap.OUT_DIR)
    transfer = pd.read_csv(out / sweep_swap.TRANSFER_CSV)
    assert set(transfer["model"]) == {SWAP_MODEL, REF_MODEL}
    main = pd.read_csv(out / "test_sweep.csv").set_index(["method", "theta"])
    for row in transfer.itertuples():
        assert REF_FROZEN[row.method] == pytest.approx(row.theta)
        ref = main.loc[(row.method, round(row.theta, 2))]
        assert row.accuracy == pytest.approx(ref["accuracy"])
        assert row.firing_rate == pytest.approx(ref["firing_rate"])


def test_parse_tolerance_counts_cover_both_sets(swap_run) -> None:
    meta = swap_run["meta"]
    counts = meta["parse_tolerance"]["counts"]
    assert set(counts) == {
        f"{s}/{r}"
        for s in ("answerable", "ext_test_wrong")
        for r in sweep_swap.TOLERANCE_ROLE_DIRS
    }
