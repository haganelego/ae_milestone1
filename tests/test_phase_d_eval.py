"""Tests for the Phase D evaluation (recovery plan §6.3, §7).

Run: uv run --project egtr --with pytest python -m pytest tests/test_phase_d_eval.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1 import phase_d_eval as pd_eval  # noqa: E402
from milestone1.phase_d_eval import (  # noqa: E402
    VERDICT_NOT_POSITIVE,
    VERDICT_POSITIVE,
    VERDICT_UNCONFIRMED,
    VERDICT_UNSTABLE,
    Arrays,
    combined_verdict,
    level_arrays,
    load_policy,
    policy_minus_mix,
    seed_a_references,
    summarize_seed,
    verdict,
)

L = 20


def _policy(decisions: dict[int, int], failures: tuple[int, ...] = ()) -> pd.DataFrame:
    return load_policy(pd.DataFrame({
        "image_id": list(decisions), "decision_t": list(decisions.values()),
        "extra_cost_failure": [i in failures for i in decisions], "U": 0.5, "valid": True,
    }))  # fmt: skip


def _stage(curves: dict[int, list[float]], levels: tuple[int, ...] = tuple(range(1, L + 1))
           ) -> pd.DataFrame:  # fmt: skip
    rows = [{"image_id": i, "t": t, "R50": c[t - 1], "relsg_gt": c[t - 1], "valid": 1}
            for i, c in curves.items() for t in levels]  # fmt: skip
    return pd.DataFrame(rows)


def _arrays(lo: list[float], hi: list[float], cont: list[bool]) -> Arrays:
    n = len(lo)
    return Arrays(np.array(lo), np.full(n, np.nan), np.array(hi), np.array(cont),
                  np.zeros(n, bool), tuple(range(n)))  # fmt: skip


def test_load_policy_validates() -> None:
    pol = _policy({1: 10, 2: 20}, failures=(2,))
    assert list(pol["failure"]) == [False, True]
    with pytest.raises(ValueError, match="decision_t"):
        _policy({1: 15})
    dup = pd.DataFrame({"image_id": [1, 1], "decision_t": [10, 20]})
    with pytest.raises(ValueError, match="duplicate"):
        load_policy(dup)


def test_policy_minus_mix_same_realized_p() -> None:
    # Policy sends the high-gain image to t20: p = .5.
    a = _arrays([0.1, 0.1], [0.9, 0.1], [True, False])
    policy, mix, diff, p = policy_minus_mix(a, np.arange(2))
    assert p == 0.5
    assert policy == pytest.approx((0.9 + 0.1) / 2)
    assert mix == pytest.approx(np.mean([0.1 + 0.5 * 0.8, 0.1]))
    assert diff == pytest.approx(0.5 - 0.3)


def test_bootstrap_recomputes_p_per_resample(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _arrays([0.1, 0.2, 0.3, 0.0], [0.5, 0.2, 0.6, 0.1], [True, False, True, False])
    seen: list[float] = []
    original = pd_eval.policy_minus_mix

    def spy(arr: Arrays, rows: np.ndarray) -> tuple[float, float, float, float]:
        out = original(arr, rows)
        seen.append(out[3])
        return out

    monkeypatch.setattr(pd_eval, "policy_minus_mix", spy)
    pd_eval.bootstrap_policy_minus_mix(a, repeats=50)
    assert len(seen) == 51  # every resample + the full sample
    assert len(set(seen)) > 1  # realized p differs across resamples


def test_verdict_rules_section7() -> None:
    assert verdict(0.02, 0.001) == VERDICT_POSITIVE
    assert verdict(0.02, -0.01) == VERDICT_UNCONFIRMED
    assert verdict(-0.01, -0.05) == VERDICT_NOT_POSITIVE
    assert combined_verdict(VERDICT_UNCONFIRMED, -0.003) == VERDICT_UNSTABLE
    assert combined_verdict(VERDICT_POSITIVE, 0.01) == VERDICT_POSITIVE
    assert combined_verdict(VERDICT_NOT_POSITIVE, -0.02) == VERDICT_NOT_POSITIVE


def test_missing_q_excluded_not_zero_and_seed_mixing_rejected() -> None:
    stage = _stage({1: [0.1] * L, 2: [0.2] * L})
    stage.loc[(stage["image_id"] == 2) & (stage["t"] == 20), "R50"] = np.nan
    arrays, info = level_arrays(
        stage, "R50", _policy({1: 20, 2: 20}), (10, 15, 20), "A"
    )
    assert arrays.ids == (1,) and info["n_excluded_missing_q"] == 1
    mixed = pd.concat([stage, stage.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        level_arrays(mixed, "R50", _policy({1: 20, 2: 10}), (10, 15, 20), "A")


def test_summary_counts_failures_and_calls_separately() -> None:
    stage = _stage(
        {i: list(np.linspace(0, 0.5, L)) for i in range(1, 5)}, levels=(10, 15, 20)
    )
    pol = _policy({1: 20, 2: 20, 3: 10, 4: 10}, failures=(2,))
    arrays, info = level_arrays(stage, "R50", pol, (10, 15, 20), "B")
    row = summarize_seed(arrays, info, "R50", "B", repeats=20)
    assert row["n_failure_continued"] == 1
    assert row["failure_extra_levels_per_image"] == pytest.approx(10 / 4)
    assert (row["pgsg_states_observed"], row["v4_calls"]) == (6, 4)
    assert row["mean_level"] == 15 and row["fixed15_same_budget"]


def test_seed_a_reference_uses_the_full_curve() -> None:
    curve = [0.0] * L
    curve[4], curve[19] = 0.5, 1.0  # non-concave -> F beats the adjacent mix
    refs = seed_a_references(_stage({1: curve, 2: curve}), "R50", 10.0)
    assert refs["F_at_mean_level"] == pytest.approx(0.5 + 0.5 * 5 / 15)
    assert refs["n_complete_series"] == 2


# --------------------------------------------------------------------------
# MIRU-style comparison
# --------------------------------------------------------------------------


def _levels(q10: list[float], q20: list[float], cont: list[bool]) -> pd_eval.SeedLevels:
    q = np.column_stack([q10, np.add(q10, q20) / 2, q20])
    return pd_eval.SeedLevels(
        {"R50": q, "relsg_gt": q}, np.array(cont), tuple(range(len(cont)))
    )


def test_oracle_mix_and_stop_values() -> None:
    lv = _levels([0.1, 0.1, 0.5, 0.2], [0.9, 0.2, 0.4, 0.3], [True, False, False, True])
    orc = pd_eval.oracle_cont(
        lv.q["R50"], 2
    )  # g = .8, .1, -.1, .1 -> images 0 and 1 (stable)
    assert list(orc) == [True, True, False, False]
    vals = pd_eval.stop_values(lv.q["R50"], np.array([20, 10, 10, 20]), orc)
    assert list(vals["mae"]) == [0, 10, 0, 10] and vals["agree"].mean() == 0.5
    mix = pd_eval.mix_values(lv.q["R50"], 0.5, orc)
    assert mix["mae"].mean() == pytest.approx(5.0) and mix[
        "agree"
    ].mean() == pytest.approx(0.5)
    rows = pd_eval.method_rows(lv, "R50")
    assert rows["GT two-choice oracle (reference)"]["agree"].mean() == 1.0
    assert (
        rows["fixed15"]["agree"].mean() == 0.0 and rows["fixed15"]["mae"].mean() == 5.0
    )


def test_significance_rows_online_comparators_only() -> None:
    rng = np.random.default_rng(0)
    q10 = rng.random(12) * 0.3
    q20 = q10 + rng.random(12) * 0.4
    cont = q20 - q10 > np.median(q20 - q10)  # a good policy: picks the high-gain half
    lv = _levels(list(q10), list(q20), list(cont))
    rows = pd_eval.significance_rows(lv, "R50", "R50", repeats=200, perm_repeats=200)
    assert {(r["comparator"], r["column"]) for r in rows} == {
        ("mix", col) for col in ("mae", "agree", "q")
    }  # no conf decisions -> only the mix; fixed levels are never tested
    mix_q = next(r for r in rows if r["column"] == "q")
    assert mix_q["direction"] == "better" and mix_q["perm_p_better"] < 0.05
    # conf_two_stage that picks the LOW-gain half -> policy significantly better (sign flip)
    lv_conf = pd_eval.SeedLevels(lv.q, lv.cont, lv.ids, conf_cont=~cont)
    rows = pd_eval.significance_rows(
        lv_conf, "R50", "R50", repeats=200, perm_repeats=200
    )
    conf_q = next(
        r for r in rows if r["comparator"] == "conf_two_stage" and r["column"] == "q"
    )
    assert conf_q["perm_test"] == "paired sign flip"
    assert conf_q["diff_policy_minus_comparator"] > 0 and conf_q["significant"]


def test_conf_two_stage_rule_and_fit() -> None:
    fit = pd_eval.fit_conf_threshold([1.0, 2.0, 2.0, 3.0])
    assert fit["tau"] == 2.0 and fit["n_below"] == 1 and fit["n_tie"] == 2
    assert fit["p_tie"] == pytest.approx(0.5)  # (0.5*4 - 1) / 2
    assert (
        pd_eval.conf_decision(1, 1.0, 2.0, 0.5)[0] == 20
    )  # low confidence -> continue
    assert pd_eval.conf_decision(1, 3.0, 2.0, 0.5)[0] == 10
    assert pd_eval.conf_decision(1, float("nan"), 2.0, 0.5) == (20, "failure_continue")
    graph = {"entities": [{"id": f"v{i}", "score": float(i), "box_xyxy_base": [0, 0, 1, 1]}
                          for i in range(10)]}  # fmt: skip
    assert pd_eval.top_k_confidence(graph, 8) == pytest.approx(np.mean(range(2, 10)))
    assert np.isnan(pd_eval.top_k_confidence({"entities": []}))
