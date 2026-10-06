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
    RecallCurves,
    build_eval_set,
    configured_q_metric,
    equal_observation_mix,
    load_stage_metrics,
    mean_recall_at_stop,
    mix_recall,
    paired_bootstrap,
    q_output_dir,
    relative_stable_oracle,
    residual,
    select_figure_examples,
)
from milestone1.robovis_replay import RunLayout, run_replay_eval  # noqa: E402
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
