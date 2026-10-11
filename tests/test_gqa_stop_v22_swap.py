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
    meta = sweep_swap.sweep_swap(ctx)
    return dict(ctx=ctx, provenance=provenance, reference=reference, meta=meta)


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


def test_supplementary_full_grid_is_written(swap_run) -> None:
    ctx, meta = swap_run["ctx"], swap_run["meta"]
    supp = meta["supplementary"]
    assert supp["thetas"][0] == 0.05 and supp["thetas"][-1] == 1.0
    assert len(supp["thetas"]) == 20
    frame = pd.read_csv(ctx.paths.out(sweep_swap.OUT_DIR) / supp["csv"])
    assert len(frame) == 2 * len(sweep_v22.SWEEP_METHODS) * 20
    assert set(frame["model"]) == {SWAP_MODEL, REF_MODEL}
    figs = ctx.paths.out(sweep_swap.OUT_DIR) / sweep_swap.SUPP_DIR / "figures"
    assert all((figs / name).exists() for name in supp["figures"])
    counts = meta["parse_tolerance"]["counts"]
    assert set(counts) == {
        f"{s}/{r}"
        for s in ("answerable", "ext_test_wrong")
        for r in sweep_swap.TOLERANCE_ROLE_DIRS
    }
