"""Tests for ROBOVIS2027 evaluation + replay (plan §9, acceptance tests 8 and 9).

Run: uv run --project egtr --with pytest python -m pytest tests/test_robovis_eval.py -q
The end-to-end tests replay ``tests/fixtures/robovis_synthetic`` (regenerate with
``tests/fixtures/make_robovis_synthetic.py``) with all sockets disabled.
"""

import copy
import json
import math
import shutil
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.robovis_data import load_robovis_config  # noqa: E402
from milestone1.robovis_diag import auc, compare_paths, run_signal_diag  # noqa: E402
from milestone1.robovis_eval import (  # noqa: E402
    ORACLE_DEFINED,
    ORACLE_INVALID_LEVELS,
    ORACLE_UNSTABLE_TAIL,
    ORACLE_ZERO_RECOVERY,
    FixedMix,
    OracleParams,
    RecallCurves,
    build_eval_set,
    configured_q_metric,
    equal_observation_mix,
    gain_based_oracle,
    isotonic_increasing,
    load_stage_metrics,
    mean_recall_at_stop,
    mix_recall,
    monotone_stable_oracle,
    paired_bootstrap,
    q_output_dir,
    relative_stable_oracle,
    residual,
    select_figure_examples,
)
from milestone1.robovis_judge_quality import (  # noqa: E402
    add_judge_columns,
    judge_quality,
)
from milestone1.robovis_replay import (  # noqa: E402
    RunLayout,
    is_clean_graph_variant,
    resolve_graphs_subdir,
    run_replay_eval,
)
from milestone1.robovis_sgdet import NUM_PREDICATES, OFFICIAL_MR_EPS  # noqa: E402
from milestone1.stop_policy import build_conditions  # noqa: E402

RATIO = 0.9
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "robovis_synthetic"
FAST_REPEATS = 200


# --------------------------------------------------------------------------
# (8) oracle
# --------------------------------------------------------------------------


def test_oracle_relative_stable_example() -> None:
    res = relative_stable_oracle([0.2, 0.95, 0.7, 0.92, 1.0], None, RATIO)
    assert (res.status, res.t, res.qmax) == (ORACLE_DEFINED, 4, 1.0)


def test_oracle_unstable_tail() -> None:
    res = relative_stable_oracle([0.1, 0.5, 0.4], None, RATIO)
    assert (res.status, res.t) == (ORACLE_UNSTABLE_TAIL, None)


def test_oracle_zero_recovery() -> None:
    res = relative_stable_oracle([0.0, 0.0, 0.0], None, RATIO)
    assert (res.status, res.t) == (ORACLE_ZERO_RECOVERY, None)


def test_oracle_invalid_level_is_undefined_not_filled() -> None:
    res = relative_stable_oracle([0.2, 0.5, 0.5], [True, False, True], RATIO)
    assert (res.status, res.t) == (ORACLE_INVALID_LEVELS, None)
    nan = relative_stable_oracle([0.2, math.nan, 0.5], None, RATIO)
    assert nan.status == ORACLE_INVALID_LEVELS


def test_oracle_threshold_is_inclusive() -> None:
    assert relative_stable_oracle([0.0, 0.45, 0.5], None, RATIO).t == 2


def test_residual() -> None:
    q = [0.2, 0.95, 0.7, 0.92, 1.0]
    assert residual(q, 3) == pytest.approx(0.3)
    assert residual(q, 5) == 0.0


# --------------------------------------------------------------------------
# Equal-observation mix / mR / bootstrap / figure selection
# --------------------------------------------------------------------------


def test_equal_observation_mix() -> None:
    mix = equal_observation_mix(12.25, 20)
    assert (mix.lower, mix.weight_upper) == (12, pytest.approx(0.25))
    assert equal_observation_mix(20.0, 20) == FixedMix(20, 0.0)
    with pytest.raises(ValueError):
        equal_observation_mix(0.5, 20)


def test_mix_recall_interpolates_per_image() -> None:
    qm = np.array([[0.0, 0.4, 1.0], [0.2, 0.2, 0.6]])
    got = mix_recall(qm, FixedMix(2, 0.5))
    assert got == pytest.approx([0.7, 0.4])


def test_mean_recall_at_stop_matches_official_convention() -> None:
    stops = pd.DataFrame({"image_id": [1, 2], "stop_t": [1, 2]})
    hits = pd.DataFrame(
        {
            "image_id": [1, 1, 2, 2],
            "t": [1, 1, 2, 2],
            "pred_id": [0, 1, 0, 1],
            "n_gt": [2, 1, 4, 0],
            "hits": [1, 1, 4, 0],
        }
    )
    # pred 0: mean(1/2, 4/4) = 0.75; pred 1: 1/1 (n_gt=0 row ignored).
    present = mean_recall_at_stop(stops, hits, [1, 2], official=False)
    assert present == pytest.approx(0.875)
    official = mean_recall_at_stop(stops, hits, [1, 2])
    assert official == pytest.approx(1.75 / (NUM_PREDICATES + OFFICIAL_MR_EPS))
    assert math.isnan(mean_recall_at_stop(stops, None, [1, 2]))


def _toy_curves() -> RecallCurves:
    q = {i: np.linspace(0.0, 0.1 * i, 5) for i in range(1, 7)}
    valid = {i: np.ones(5, dtype=bool) for i in q}
    return RecallCurves(q, valid, 5)


def _toy_per_image(curves: RecallCurves) -> pd.DataFrame:
    rows = []
    for i, q in curves.q.items():
        for method, stop in (("a", 3), ("b", 2 + i % 3)):
            rows.append(
                {"image_id": i, "method": method, "stop_t": stop,
                 "q_stop": q[stop - 1], "residual": residual(q, stop)}
            )  # fmt: skip
    return pd.DataFrame(rows)


def test_paired_bootstrap_is_seeded_and_brackets_mean() -> None:
    curves = _toy_curves()
    eval_set = build_eval_set(curves, RATIO)
    per_image = _toy_per_image(curves)
    kwargs: dict[str, Any] = {
        "adaptive_methods": ["a", "b"],
        "comparisons": [("a", "b")],
        "repeats": FAST_REPEATS,
        "seed": 7,
        "ci_level": 0.95,
    }
    first = paired_bootstrap(per_image, curves, eval_set, **kwargs)
    second = paired_bootstrap(per_image, curves, eval_set, **kwargs)
    assert first == second
    diff = first["pairwise"]["a_minus_b"]["q_stop"]
    assert diff["ci_low"] <= diff["mean"] <= diff["ci_high"]
    eq = first["methods"]["a"]["q_minus_eq_fixed"]
    # method "a" stops at a constant level -> identical to fixed_3.
    assert eq["mean"] == pytest.approx(0.0)


def test_select_figure_examples() -> None:
    per_image = pd.DataFrame(
        {
            "image_id": [1, 2, 3, 4] * 2,
            "method": ["p"] * 4 + ["b"] * 4,
            "residual": [0.0, 0.3, 0.1, 0.1, 0.5, 0.1, 0.1, 0.2],
        }
    )
    sel = select_figure_examples(per_image, "p", "b")["examples"]
    assert sel["improvement_max"]["image_id"] == 1  # diff +0.5
    assert sel["degradation_max"]["image_id"] == 2  # diff -0.2
    assert sel["median"]["image_id"] in {3, 4}


# --------------------------------------------------------------------------
# (9) end-to-end replay with no network / model client
# --------------------------------------------------------------------------


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def guard(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access attempted during replay")

    monkeypatch.setattr(socket, "socket", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


@pytest.fixture
def fast_cfg() -> dict[str, Any]:
    cfg = copy.deepcopy(load_robovis_config())
    cfg["evaluation"]["bootstrap_repeats"] = FAST_REPEATS
    return cfg


def _expected() -> dict[str, Any]:
    return json.loads((FIXTURE / "expected_stops.json").read_text())


def test_replay_fixture_end_to_end(
    tmp_path: Path, no_network: None, fast_cfg: dict[str, Any]
) -> None:
    out = tmp_path / "eval"
    digest = run_replay_eval(FIXTURE, out, fast_cfg)
    for name in ("stops.csv", "summary.csv", "bootstrap.json", "eval_set_ids.json"):
        assert (out / name).exists(), name

    stops = pd.read_csv(out / "stops.csv")
    for image_id, spec in _expected().items():
        rows = stops[stops["image_id"] == int(image_id)].set_index("method")
        for method, stop_t in spec["stops"].items():
            assert rows.loc[method, "stop_t"] == stop_t, (image_id, method)
        for method, n_fail in spec["failures"].items():
            assert rows.loc[method, "failure_count"] == n_fail, (image_id, method)
        if spec["forced"] is not None:
            assert bool(rows.loc["proposed", "forced_terminal"]) == spec["forced"]

    oracle = pd.read_csv(out / "oracle.csv").set_index("image_id")
    for image_id, spec in _expected().items():
        got = oracle.loc[int(image_id), "oracle_t"]
        if spec["oracle"] is None:
            assert pd.isna(got), image_id
        else:
            assert int(got) == spec["oracle"], image_id

    eval_ids = json.loads((out / "eval_set_ids.json").read_text())
    expected_eval = sorted(int(i) for i, s in _expected().items() if s["in_eval_set"])
    assert eval_ids["eval_ids"] == expected_eval
    assert digest["n_eval"] == len(expected_eval)

    summary = pd.read_csv(out / "summary.csv").set_index("method")
    n_levels = len(fast_cfg["resolution"]["scales"])
    assert {f"fixed_{k:02d}" for k in range(1, n_levels + 1)} <= set(summary.index)
    assert summary.loc["fixed_05", "mean_stop_t"] == 5
    assert summary.loc["proposed", "n_images"] == len(expected_eval)
    assert not math.isnan(summary.loc["proposed", "mR50_stop"])
    assert summary.loc["proposed", "status"] == "main"
    assert summary.loc["fixed_05", "status"] == "baseline"
    assert "exploratory" in summary.loc["benefit_gated", "status"]
    assert "exploratory" in summary.loc["benefit_gated_obs", "status"]


def test_replay_without_decisions_forces_every_image(
    tmp_path: Path, no_network: None, fast_cfg: dict[str, Any]
) -> None:
    src = tmp_path / "run"
    shutil.copytree(FIXTURE, src, ignore=shutil.ignore_patterns("decisions"))
    run_replay_eval(src, tmp_path / "eval", fast_cfg)
    stops = pd.read_csv(tmp_path / "eval" / "stops.csv")
    adaptive = stops[~stops["method"].str.startswith("fixed_")]
    n_levels = len(fast_cfg["resolution"]["scales"])
    assert (adaptive["stop_t"] == n_levels).all()
    assert adaptive["forced_terminal"].all()


# --------------------------------------------------------------------------
# Signal diagnostic (robovis_diag)
# --------------------------------------------------------------------------


def test_auc_mann_whitney() -> None:
    labels = np.array([True, True, False, False])
    assert auc(np.array([2.0, 1.0, 0.0, 0.0]), labels) == pytest.approx(1.0)
    assert auc(np.array([1.0, 1.0, 1.0, 1.0]), labels) == pytest.approx(0.5)
    assert auc(np.array([0.0, 1.0, 1.0, 0.0]), labels) == pytest.approx(0.5)
    assert math.isnan(auc(np.array([1.0]), np.array([True])))


def test_signal_diag_on_fixture(
    tmp_path: Path, no_network: None, fast_cfg: dict[str, Any]
) -> None:
    res = run_signal_diag(FIXTURE, tmp_path, fast_cfg, RunLayout())
    n_levels = len(fast_cfg["resolution"]["scales"])
    assert res.stats["levels"] == list(range(1, n_levels + 1))
    for name in ("signal_per_state.csv", "signal_by_level.csv", "signal_stats.json"):
        assert (tmp_path / name).exists(), name

    s = res.states.set_index(["image_id", "t"])
    # 1001 t7 = "a1h2": A=1, H=2 -> proposed 2, observed_only 1; two gate items.
    row = s.loc[(1001, 7)]
    assert (row["A"], row["H"], row["C_proposed"], row["C_observed_only"]) == (
        1,
        2,
        2,
        1,
    )
    assert (row["n_gate_items"], row["max_basis"], row["max_benefit"]) == (2, 2, 2)
    # 1001 t1 = empty graph: A = max_u by rule, no items.
    assert bool(s.loc[(1001, 1), "graph_empty"]) and s.loc[(1001, 1), "A"] == 2
    # 1009 t4 = unknown limitation: not a gate item; uc variant gives 1.
    row = s.loc[(1009, 4)]
    assert (row["n_gate_items"], row["C_proposed"], row["C_unknown_conservative"]) == (
        0,
        0,
        1,
    )
    # Failed states are NaN, not zero.
    assert math.isnan(s.loc[(1002, 8), "C_proposed"])
    # gain_left = qmax - q_t (q = R50 by default).
    assert s.loc[(1001, 3), "gain_left"] == pytest.approx(0.78 - 0.05)

    lvl = res.levels.set_index("t")
    assert lvl["n_states"].eq(len(_expected())).all()
    assert lvl.loc[1, "frac_C_proposed_ge_theta"] == pytest.approx(1.0)
    assert {"C_proposed", "C_benefit_gated", "H", "A", "H_bg"} <= set(
        res.stats["spearman"]
    )
    # 1001 t7 = "a1h2": basis 1 / benefit 2 on the observed issue -> A_bg = 2.
    assert s.loc[(1001, 7), "A_bg"] == 2
    assert s.loc[(1001, 7), "C_benefit_gated_obs"] == 2


def test_compare_paths_one_row_per_path_and_condition(
    tmp_path: Path, fast_cfg: dict[str, Any]
) -> None:
    res = run_signal_diag(FIXTURE, tmp_path, fast_cfg, RunLayout())
    conditions = build_conditions(fast_cfg)
    table = compare_paths({"X": res, "Y": res}, conditions, [1, 10, 20])
    assert len(table) == 2 * len(conditions)
    for col in ("frac_ge_theta_t01", "frac_ge_theta_t20", "rho_t", "rho_gain_left",
                "auc_score", "auc_ge_theta", "status"):  # fmt: skip
        assert col in table.columns
    row = table[(table["path"] == "X") & (table["condition"] == "proposed")].iloc[0]
    assert row["frac_ge_theta_t01"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Selectable q metric
# --------------------------------------------------------------------------


def test_q_output_dir_keeps_legacy_layout_for_r50(tmp_path: Path) -> None:
    assert q_output_dir(tmp_path, "R50") == tmp_path
    assert q_output_dir(tmp_path, "relsg_gt") == tmp_path / "q_relsg_gt"


def test_configured_q_metric_override_and_default(fast_cfg: dict[str, Any]) -> None:
    assert configured_q_metric(fast_cfg) == fast_cfg["evaluation"]["q_metric"]
    assert configured_q_metric(fast_cfg, "relsg_gt") == "relsg_gt"
    no_key = {"evaluation": {}}
    assert configured_q_metric(no_key) == "R50"


def test_load_stage_metrics_missing_metric_column_raises() -> None:
    with pytest.raises(ValueError, match="relsg_missing"):
        load_stage_metrics(FIXTURE / "stage_metrics.csv", "relsg_missing")


def test_replay_with_relsg_metric_uses_that_column(
    tmp_path: Path, no_network: None, fast_cfg: dict[str, Any]
) -> None:
    run_replay_eval(FIXTURE, tmp_path / "r50", fast_cfg)
    run_replay_eval(FIXTURE, tmp_path / "relsg", fast_cfg, q_metric="relsg_gt")
    r50 = pd.read_csv(tmp_path / "r50" / "summary.csv").set_index("method")
    rel = pd.read_csv(tmp_path / "relsg" / "summary.csv").set_index("method")
    assert (rel["q_metric"] == "relsg_gt").all() and (r50["q_metric"] == "R50").all()
    # Stops do not depend on q; q(stop) and the fixed curve do (relsg = sqrt(R50)).
    assert rel.loc["proposed", "mean_stop_t"] == r50.loc["proposed", "mean_stop_t"]
    assert rel.loc["fixed_05", "q_stop"] > r50.loc["fixed_05", "q_stop"]
    # mR@50 is metric-independent.
    assert rel.loc["proposed", "mR50_stop"] == pytest.approx(
        r50.loc["proposed", "mR50_stop"]
    )
    boot = json.loads((tmp_path / "relsg" / "bootstrap.json").read_text())
    assert boot["q_metric"] == "relsg_gt"
    # Oracle on relsg: 1001 qmax = sqrt(.78); threshold 0.9*0.8832 = .7949 -> t8
    # (sqrt(.70) = .8367) instead of t9 under R50.
    oracle = pd.read_csv(tmp_path / "relsg" / "oracle.csv").set_index("image_id")
    assert oracle.loc[1001, "oracle_t"] == 8


def test_signal_diag_with_relsg_metric(
    tmp_path: Path, fast_cfg: dict[str, Any]
) -> None:
    res = run_signal_diag(FIXTURE, tmp_path, fast_cfg, RunLayout(), q_metric="relsg_gt")
    s = res.states.set_index(["image_id", "t"])
    assert res.stats["q_metric"] == "relsg_gt"
    assert s.loc[(1001, 3), "q"] == pytest.approx(0.05**0.5, abs=1e-4)
    assert s.loc[(1001, 3), "R50"] == pytest.approx(0.05)
    assert s.loc[(1001, 3), "gain_left"] == pytest.approx(
        0.78**0.5 - 0.05**0.5, abs=1e-4
    )


# --------------------------------------------------------------------------
# Auxiliary oracle definitions
# --------------------------------------------------------------------------

GAIN_DELTA = 0.05


def test_isotonic_increasing_pools_violators() -> None:
    assert isotonic_increasing([3.0, 1.0, 2.0]) == pytest.approx([2.0, 2.0, 2.0])
    assert isotonic_increasing([0.1, 0.2, 0.3]) == pytest.approx([0.1, 0.2, 0.3])
    env = isotonic_increasing([0.2, 0.95, 0.7, 0.92, 1.0])
    assert env == pytest.approx([0.2, 0.825, 0.825, 0.92, 1.0])
    assert np.all(np.diff(env) >= 0)


def test_monotone_stable_on_plan_example() -> None:
    res = monotone_stable_oracle([0.2, 0.95, 0.7, 0.92, 1.0], None, RATIO)
    assert (res.status, res.t) == (ORACLE_DEFINED, 4)


def test_monotone_stable_defines_unstable_tail_case() -> None:
    # Envelope [.1, .45, .45]: threshold .405 -> t2 (relative-stable: unstable_tail).
    res = monotone_stable_oracle([0.1, 0.5, 0.4], None, RATIO)
    assert (res.status, res.t) == (ORACLE_DEFINED, 2)
    assert res.qmax == pytest.approx(0.45)


def test_monotone_stable_is_robust_to_an_early_spike() -> None:
    # A cumulative max would lock in the t2 spike (oracle t2); isotonic does not.
    q = [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.5]
    assert monotone_stable_oracle(q, None, RATIO).t == len(q)


@pytest.mark.parametrize("oracle_fn", [monotone_stable_oracle, gain_based_oracle])
def test_aux_oracles_undefined_cases(oracle_fn: Any) -> None:
    param = RATIO if oracle_fn is monotone_stable_oracle else GAIN_DELTA
    assert oracle_fn([0.0, 0.0, 0.0], None, param).status == ORACLE_ZERO_RECOVERY
    invalid = oracle_fn([0.1, 0.5, 0.5], [True, False, True], param)
    assert (invalid.status, invalid.t) == (ORACLE_INVALID_LEVELS, None)


@pytest.mark.parametrize(
    ("q", "delta", "t"),
    [
        ([0.2, 0.95, 0.7, 0.92, 1.0], GAIN_DELTA, 5),  # t2 gain .05 is not < .05
        ([0.2, 0.95, 0.7, 0.92, 1.0], 0.5, 2),
        ([0.1, 0.5, 0.4], GAIN_DELTA, 2),  # later levels only go down
        ([0.5, 0.52, 0.53, 0.9], GAIN_DELTA, 4),
        ([0.3], GAIN_DELTA, 1),  # final level always qualifies
    ],
)
def test_gain_based_oracle(q: list[float], delta: float, t: int) -> None:
    res = gain_based_oracle(q, None, delta)
    assert (res.status, res.t) == (ORACLE_DEFINED, t)


def test_oracle_params_from_config(fast_cfg: dict[str, Any]) -> None:
    params = OracleParams.from_config(fast_cfg)
    assert params.ratio == fast_cfg["evaluation"]["oracle_ratio"]
    assert params.gain_delta == fast_cfg["evaluation"]["oracle_gain_delta"]
    no_delta = {"evaluation": {"oracle_ratio": RATIO}}
    assert OracleParams.from_config(no_delta).gain_delta == GAIN_DELTA


def test_replay_reports_all_oracle_definitions(
    tmp_path: Path, no_network: None, fast_cfg: dict[str, Any]
) -> None:
    run_replay_eval(FIXTURE, tmp_path, fast_cfg)
    summary = pd.read_csv(tmp_path / "summary.csv").set_index("method")
    # Fixture: 9 eval images, 1 zero_recovery, 1 unstable_tail (1006).
    assert summary.loc["proposed", "oracle_n"] == 7
    assert summary.loc["proposed", "oracle_n_monotone_stable"] == 8
    assert summary.loc["proposed", "oracle_n_gain_based"] == 8
    assert not math.isnan(summary.loc["proposed", "oracle_mae_gain_based"])
    oracle = pd.read_csv(tmp_path / "oracle.csv").set_index("image_id")
    assert oracle.loc[1006, "oracle_status"] == ORACLE_UNSTABLE_TAIL
    assert oracle.loc[1006, "monotone_stable_status"] == ORACLE_DEFINED
    assert oracle.loc[1007, "gain_based_status"] == ORACLE_INVALID_LEVELS
    profile = pd.read_csv(tmp_path / "oracle_profile.csv").set_index("definition")
    assert list(profile.index) == ["relative_stable", "monotone_stable", "gain_based"]
    assert profile.loc["relative_stable", "role"] == "main"
    assert profile.loc["monotone_stable", "defined"] == 8
    ids = json.loads((tmp_path / "eval_set_ids.json").read_text())
    by_def = ids["oracle_status_counts_by_definition"]
    assert by_def["relative_stable"] == ids["oracle_status_counts"]
    assert by_def["gain_based"]["defined"] == 8


def test_q_metric_must_be_numeric_metric_column(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="identifier"):
        load_stage_metrics(FIXTURE / "stage_metrics.csv", "split")
    bad = tmp_path / "stage_metrics.csv"
    bad.write_text("image_id,t,valid,R50\n1,1,1,high\n")
    with pytest.raises(ValueError, match="not numeric"):
        load_stage_metrics(bad, "R50")


# --------------------------------------------------------------------------
# Graph directory selection (raw graphs vs cleaned graph G)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "path", "expected"),
    [
        ("auto", "F1v_v2_G", "graphs_clean"),
        ("auto", "F1v_v2_G__9b", "graphs_clean"),
        ("auto", "F1_v2_G__j9b", "graphs_clean"),
        ("auto", "F1v_v2", "graphs"),
        ("auto", "F2__9b", "graphs"),
        ("auto", "F1_Gx", "graphs"),  # "Gx" is not the G segment
        ("auto", None, "graphs"),
        ("graphs_clean", "F1", "graphs_clean"),
        ("graphs", "F1_v2_G", "graphs"),  # honoured, with a warning
    ],
)
def test_resolve_graphs_subdir(
    requested: str, path: str | None, expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert resolve_graphs_subdir(requested, path) == expected
    warned = "warning" in capsys.readouterr().err
    assert warned == (requested == "graphs" and path == "F1_v2_G")


def test_is_clean_graph_variant() -> None:
    assert is_clean_graph_variant("F1v_v2_G__9b")
    assert not is_clean_graph_variant("G2")
    assert not is_clean_graph_variant(None)


def test_replay_records_graphs_dir_and_treats_missing_graph_as_missing(
    tmp_path: Path, no_network: None, fast_cfg: dict[str, Any]
) -> None:
    run = tmp_path / "run"
    shutil.copytree(FIXTURE, run)
    shutil.copytree(run / "graphs", run / "graphs_clean")
    # Remove one clean-graph level: it must become a missing C_t like any other.
    (run / "graphs_clean" / "1010" / "level_12.json").unlink()
    layout = RunLayout(graphs="graphs_clean")
    run_replay_eval(run, tmp_path / "eval", fast_cfg, layout)
    summary = pd.read_csv(tmp_path / "eval" / "summary.csv")
    assert set(summary["graphs_dir"]) == {"graphs_clean"}
    ids = json.loads((tmp_path / "eval" / "eval_set_ids.json").read_text())
    assert ids["graphs_dir"] == "graphs_clean"
    c = pd.read_csv(tmp_path / "eval" / "c_sequences.csv")
    row = c[(c["image_id"] == 1010) & (c["t"] == 12)]
    assert set(row["outcome"]) == {"missing_graph"} and row["C"].isna().all()
    stops = pd.read_csv(tmp_path / "eval" / "stops.csv").set_index(
        ["image_id", "method"]
    )
    # 1010 proposed: t12 was the first instant stop; now reset -> stop at t14.
    assert stops.loc[(1010, "proposed"), "stop_t"] == 14
    assert stops.loc[(1010, "proposed"), "failure_count"] == 1


# --------------------------------------------------------------------------
# Judge-quality diagnostic
# --------------------------------------------------------------------------


def _judge_states() -> pd.DataFrame:
    base = {"valid": True, "n_items": 2, "qmax": 1.0, "outcome": "ok"}
    spec = [  # image, t, q_ratio, n_resolved, n_lowres, C_proposed
        (1, 1, 0.2, 0, 2, 2), (1, 2, 0.6, 1, 1, 1), (1, 3, 1.0, 2, 0, 0),
        (2, 1, 0.4, 0, 2, 2), (2, 2, 0.95, 2, 0, 0), (2, 3, 0.9, 1, 1, 1),
    ]  # fmt: skip
    rows = [
        {**base, "image_id": i, "t": t, "frac_of_qmax": r, "n_resolved": nr,
         "n_lowres": nl, "C_proposed": c}
        for i, t, r, nr, nl, c in spec
    ]  # fmt: skip
    # Excluded: a failed state and an image that never recovers anything.
    rows.append({**base, "image_id": 3, "t": 1, "frac_of_qmax": 0.5, "n_resolved": 2,
                 "n_lowres": 0, "C_proposed": None, "outcome": "missing_checks"})  # fmt: skip
    rows.append({**base, "image_id": 4, "t": 1, "frac_of_qmax": None, "qmax": 0.0,
                 "n_resolved": 2, "n_lowres": 0, "C_proposed": 0})  # fmt: skip
    return pd.DataFrame(rows)


def test_judge_quality_calibration_and_auc() -> None:
    table, stats = judge_quality(_judge_states(), theta=2, ratio=RATIO,
                                 condition_columns=["C_proposed"])  # fmt: skip
    assert stats["n_states"] == 6 and stats["n_sufficient"] == 3  # 1.0, .95, .9
    # frac_resolved: sufficient {1, 1, .5} vs not {0, .5, 0} -> 8.5 / 9 pairs.
    assert stats["auc_sufficient_resolved"] == pytest.approx(8.5 / 9)
    # 1 - C: sufficient {1, 1, 0} vs not {-1, 0, -1} -> 8.5 / 9.
    assert stats["auc_sufficient_C"]["C_proposed"] == pytest.approx(8.5 / 9)
    assert stats["rho_resolved_qratio"]["rho"] > 0
    # frac_lowres (diagnostic): sufficient {0, 0, .5} vs not {1, .5, 1};
    # score 1 - frac_lowres: {1, 1, .5} vs {0, .5, 0} -> 8.5 / 9.
    assert stats["auc_sufficient_lowres"] == pytest.approx(8.5 / 9)
    assert stats["rho_lowres_qratio"]["rho"] < 0
    assert stats["lowres_role"] == "diagnostic signal, not a stop condition"
    cal = table[table["table"] == "calibration"].set_index("q_ratio_bin")
    assert list(cal.index) == ["[0,0.5)", "[0.5,0.9)", "[0.9,1]"]
    assert list(cal["n"]) == [2, 1, 3]
    assert cal.loc["[0.9,1]", "frac_resolved_mean"] == pytest.approx(2.5 / 3)
    assert cal.loc["[0,0.5)", "any_gate_theta_mean"] == pytest.approx(1.0)
    lvl = table[table["table"] == "level"].set_index("t")
    assert lvl.loc[1, "frac_lowres_mean"] == pytest.approx(1.0)


def test_add_judge_columns_handles_no_items() -> None:
    st = pd.DataFrame({"n_items": [0, 4], "n_resolved": [0, 1], "n_lowres": [0, 2],
                       "C_proposed": [None, 2.0], "frac_of_qmax": [0.1, 0.2]})  # fmt: skip
    out = add_judge_columns(st, theta=2)
    assert (
        math.isnan(out.loc[0, "frac_resolved"]) and out.loc[1, "frac_resolved"] == 0.25
    )
    assert (
        math.isnan(out.loc[0, "any_gate_theta"]) and out.loc[1, "any_gate_theta"] == 1.0
    )


def test_signal_diag_writes_judge_quality(
    tmp_path: Path, fast_cfg: dict[str, Any]
) -> None:
    res = run_signal_diag(FIXTURE, tmp_path, fast_cfg, RunLayout())
    assert (tmp_path / "judge_quality.csv").exists()
    assert "frac_resolved" in res.states.columns
    table = compare_paths({"X": res}, build_conditions(fast_cfg), [1, 10, 20])
    for col in ("rho_resolved_t", "rho_resolved_qratio", "auc_sufficient_C",
                "auc_sufficient_resolved", "rho_lowres_qratio",
                "auc_sufficient_lowres"):  # fmt: skip
        assert col in table.columns
