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
from milestone1.robovis_eval import (  # noqa: E402
    ORACLE_DEFINED,
    ORACLE_INVALID_LEVELS,
    ORACLE_UNSTABLE_TAIL,
    ORACLE_ZERO_RECOVERY,
    FixedMix,
    RecallCurves,
    build_eval_set,
    equal_observation_mix,
    mean_recall_at_stop,
    mix_recall,
    paired_bootstrap,
    relative_stable_oracle,
    residual,
    select_figure_examples,
)
from milestone1.robovis_replay import run_replay_eval  # noqa: E402

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


def test_mean_recall_at_stop_is_dataset_level_per_predicate() -> None:
    stops = pd.DataFrame({"image_id": [1, 2], "stop_t": [1, 2]})
    hits = pd.DataFrame(
        {
            "image_id": [1, 1, 2, 2],
            "t": [1, 1, 2, 2],
            "pred_id": [0, 1, 0, 1],
            "n_gt": [2, 1, 2, 0],
            "hits": [1, 1, 2, 0],
        }
    )
    # pred 0: 3/4, pred 1: 1/1 -> mean 0.875
    assert mean_recall_at_stop(stops, hits, [1, 2]) == pytest.approx(0.875)
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
    diff = first["pairwise"]["a_minus_b"]["R50_stop"]
    assert diff["ci_low"] <= diff["mean"] <= diff["ci_high"]
    eq = first["methods"]["a"]["R50_minus_eq_fixed"]
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
