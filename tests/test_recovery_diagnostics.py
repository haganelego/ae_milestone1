"""Tests for the recovery-plan diagnostics (plan §9 items 1-6, 9, 10).

Run: uv run --project egtr --with pytest python -m pytest tests/test_recovery_diagnostics.py -q
"""

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1 import robovis_eval  # noqa: E402
from milestone1.recovery_diagnostics import (  # noqa: E402
    best_fixed_mixture,
    budget_oracle,
    cross_seed_oracle,
    legacy_adjacent_mix,
    multiseed_table,
    permuted_stops,
    quality_matrix,
    shuffle_diagnostic,
    spearman_or_undefined,
    two_choice_oracle,
    upper_hull,
)

L = 20
CONCAVE = np.sqrt(np.arange(1, L + 1) / L)  # increasing, concave


# --------------------------------------------------------------------------
# 1. identical curves -> no gain from image-specific assignment
# --------------------------------------------------------------------------


def test_identical_curves_give_zero_assignment_gain() -> None:
    q = np.tile(CONCAVE, (6, 1))
    for b in (10, 12, 15, 18):
        assert budget_oracle(q, b).value == pytest.approx(
            best_fixed_mixture(CONCAVE, b).value
        )
    two = two_choice_oracle(q[:, 9], q[:, 19])
    assert two.gain == pytest.approx(0.0)
    res = shuffle_diagnostic(q, np.array([3, 5, 7, 9, 11, 20]), repeats=50)
    assert res["observed_minus_perm"] == pytest.approx(0.0)
    assert res["perm_sd"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# 2. two-choice oracle picks the largest t10 -> t20 gains
# --------------------------------------------------------------------------


def test_two_choice_oracle_selects_top_gains_including_negative() -> None:
    q10 = np.array([0.5, 0.1, 0.4, 0.3])
    q20 = np.array([0.4, 0.9, 0.45, 0.2])  # g = -.1, .8, .05, -.1
    res = two_choice_oracle(q10, q20, k=2)
    assert set(res.chosen) == {1, 2}
    assert res.value == pytest.approx((0.5 + 0.9 + 0.45 + 0.3) / 4)
    assert res.b_half == pytest.approx(np.mean(q10 + 0.5 * (q20 - q10)))
    # Exactly k images go to t20 even when every gain is negative (least bad first).
    neg = two_choice_oracle(np.array([0.5, 0.3]), np.array([0.2, 0.25]), k=1)
    assert neg.chosen == (1,)
    assert neg.value == pytest.approx((0.5 + 0.25) / 2)


# --------------------------------------------------------------------------
# 3. knapsack DP: budget, one level per image, zero-recovery images kept
# --------------------------------------------------------------------------


def test_budget_oracle_matches_brute_force_and_respects_budget() -> None:
    rng = np.random.default_rng(0)
    levels = 5
    q = rng.random((3, levels))
    q[2] = 0.0  # zero-recovery image stays in the problem
    for b in (1.5, 2.0, 3.4):
        res = budget_oracle(q, b)
        total = math.floor(3 * b)
        best = max(
            (sum(q[i, t - 1] for i, t in enumerate(choice)) / 3, choice)
            for choice in itertools.product(range(1, levels + 1), repeat=3)
            if sum(choice) <= total
        )
        assert res.value == pytest.approx(best[0])
        assert len(res.assignment) == 3 and res.assignment.sum() <= total
        assert all(1 <= t <= levels for t in res.assignment)
        assert res.value == pytest.approx(q[np.arange(3), res.assignment - 1].mean())


# --------------------------------------------------------------------------
# 4. fixed mixture: probabilities, budget, non-concave curve
# --------------------------------------------------------------------------


def test_fixed_mixture_is_valid_and_beats_adjacent_on_non_concave() -> None:
    f = np.zeros(L)
    f[4], f[19] = 0.5, 1.0  # spikes at t5 and t20, flat zero elsewhere
    for b in (10, 12, 15, 18):
        mix = best_fixed_mixture(f, b)
        assert sum(p for _, p in mix.support) == pytest.approx(1.0)
        assert mix.mean_level <= b + 1e-12
        assert mix.value >= legacy_adjacent_mix(f, b) - 1e-12
    mix = best_fixed_mixture(f, 10)
    assert {t for t, _ in mix.support} == {5, 20}
    assert mix.value == pytest.approx(0.5 + 0.5 * 5 / 15)  # p20 = (10-5)/15
    assert legacy_adjacent_mix(f, 10) == 0.0
    # Budget beyond the curve maximum -> the maximum alone.
    rising_then_falling = np.concatenate(
        [np.linspace(0, 1, 10), np.linspace(0.9, 0.5, 10)]
    )
    top = best_fixed_mixture(rising_then_falling, 18)
    assert top.support == ((10, 1.0),) and top.value == pytest.approx(1.0)


def test_upper_hull_drops_points_below_chords() -> None:
    f = np.array([0.0, 0.1, 0.9, 0.5, 1.0])
    assert upper_hull(f) == [0, 2, 4]


# --------------------------------------------------------------------------
# 5. shuffle keeps the multiset of stop levels
# --------------------------------------------------------------------------


def test_permuted_stops_keep_multiset() -> None:
    rng = np.random.default_rng(1)
    stops = np.array([2, 2, 5, 9, 20, 20, 13])
    for _ in range(20):
        assert sorted(permuted_stops(stops, rng)) == sorted(stops)


def test_shuffle_detects_good_assignment() -> None:
    # Image i is only good at level i+1; the matched assignment beats any shuffle.
    q = np.eye(5, L)
    res = shuffle_diagnostic(q, np.arange(1, 6), repeats=200)
    assert res["observed"] == 1.0 and res["observed_minus_perm"] > 0.5
    assert res["perm_p_ge_observed"] < 0.05


# --------------------------------------------------------------------------
# 6. cross-seed: selected on one seed, scored on the other only
# --------------------------------------------------------------------------


def _multiseed(rows: list[tuple[int, int, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["image_id", "seed", "level", "R50"]).assign(
        valid=1
    )


def test_cross_seed_assignment_is_not_reselected() -> None:
    # seed1 gains: image 1 large, image 2 small; seed2 reverses them.
    rows = [
        (1, 1, 10, 0.0), (1, 1, 20, 0.9), (2, 1, 10, 0.0), (2, 1, 20, 0.1),
        (1, 2, 10, 0.0), (1, 2, 20, 0.1), (2, 2, 10, 0.0), (2, 2, 20, 0.9),
    ]  # fmt: skip
    tabs = multiseed_table(_multiseed(rows), "R50")
    res = cross_seed_oracle(tabs, select_seed=1, eval_seed=2, k=1)
    assert res["chosen_ids"] == "1"  # picked by seed1's gain
    assert res["assignment_value"] == pytest.approx((0.1 + 0.0) / 2)  # scored on seed2
    assert res["b_half_eval"] == pytest.approx((0.1 + 0.9) / 4)
    assert res["cross_seed_gain"] < 0 < res["same_seed_oracle_gain"]


def test_spearman_undefined_for_constant_gain() -> None:
    assert math.isnan(spearman_or_undefined(np.ones(5), np.arange(5.0)))
    assert spearman_or_undefined(np.arange(5.0), np.arange(5.0)) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 9. bootstrap recomputes the matched fixed mix in every resample
# --------------------------------------------------------------------------


def test_bootstrap_recomputes_equal_observation_mix_per_resample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(3)
    ids = list(range(1, 9))
    q = {i: np.sort(rng.random(L)) for i in ids}
    curves = robovis_eval.RecallCurves(q, {i: np.ones(L, bool) for i in ids}, L)
    eval_set = robovis_eval.build_eval_set(curves, 0.9)
    stops = rng.integers(2, 19, len(ids))
    rows = [{"image_id": i, "method": "m", "stop_t": int(s), "q_stop": q[i][s - 1],
             "residual": float(q[i].max() - q[i][s - 1])} for i, s in zip(ids, stops, strict=True)]  # fmt: skip
    per_image = pd.DataFrame(rows)
    seen: list[float] = []
    original = robovis_eval.equal_observation_mix

    def spy(mean_stop: float, num_levels: int) -> robovis_eval.FixedMix:
        seen.append(mean_stop)
        return original(mean_stop, num_levels)

    monkeypatch.setattr(robovis_eval, "equal_observation_mix", spy)
    repeats = 25
    robovis_eval.paired_bootstrap(
        per_image, curves, eval_set, ["m"], [], repeats, 0, 0.95
    )
    assert len(seen) == repeats + 1  # one per resample + the full sample
    assert len(set(np.round(seen, 9))) > 1  # mean stop varies across resamples


# --------------------------------------------------------------------------
# 10. missing values are never zero-filled; mixed inputs are rejected
# --------------------------------------------------------------------------


def _stage(rows: list[tuple[int, int, float | None, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["image_id", "t", "R50", "valid"])


def test_quality_matrix_excludes_incomplete_series() -> None:
    rows = [(1, t, 0.1 * t, 1) for t in range(1, L + 1)]
    rows += [(2, t, 0.0, 1) for t in range(1, L + 1) if t != 7]  # missing level 7
    rows += [(3, t, None if t == 4 else 0.2, 1) for t in range(1, L + 1)]  # NaN value
    rows += [(4, t, 0.3, 0 if t == 9 else 1) for t in range(1, L + 1)]  # invalid level
    rows += [(5, t, 0.0, 1) for t in range(1, L + 1)]  # zero recovery: kept
    qm = quality_matrix(_stage(rows), "R50", "toy")
    assert qm.ids == (1, 5) and qm.n_total == 5
    assert qm.coverage == pytest.approx(0.4)
    assert set(qm.excluded) == {2, 3, 4}


def test_duplicate_rows_are_rejected() -> None:
    rows = [(1, t, 0.1, 1) for t in range(1, L + 1)] + [(1, 5, 0.2, 1)]
    with pytest.raises(ValueError, match="duplicate"):
        quality_matrix(_stage(rows), "R50", "toy")
    dup = _multiseed([(1, 1, 10, 0.0), (1, 1, 10, 0.1)])
    with pytest.raises(ValueError, match="duplicate"):
        multiseed_table(dup, "R50")


def test_cross_seed_uses_only_images_complete_in_both_seeds() -> None:
    rows = [
        (1, 1, 10, 0.0), (1, 1, 20, 0.5), (1, 2, 10, 0.0), (1, 2, 20, 0.5),
        (2, 1, 10, 0.0), (2, 1, 20, 0.5), (2, 2, 10, 0.1),  # image 2 lacks seed2 t20
    ]  # fmt: skip
    res = cross_seed_oracle(multiseed_table(_multiseed(rows), "R50"), 1, 2, k=1)
    assert res["n"] == 1


def test_multiseed_wide_format_matches_long_format() -> None:
    wide = pd.DataFrame([
        {"image_id": 1, "seed": 1, "metric": "R50", "q5": 0.0, "q10": 0.0, "q15": 0.2,
         "q20": 0.9, "g": 0.9, "valid": 1},
        {"image_id": 2, "seed": 1, "metric": "R50", "q5": 0.0, "q10": 0.0, "q15": 0.1,
         "q20": 0.1, "g": 0.1, "valid": 1},
        {"image_id": 1, "seed": 2, "metric": "R50", "q5": 0.0, "q10": 0.0, "q15": 0.1,
         "q20": 0.1, "g": 0.1, "valid": 1},
        {"image_id": 2, "seed": 2, "metric": "R50", "q5": 0.0, "q10": 0.0, "q15": None,
         "q20": 0.9, "g": 0.9, "valid": 0},  # q15 blank: other levels still usable
        {"image_id": 1, "seed": 1, "metric": "relsg_gt", "q5": 0.5, "q10": 0.5, "q15": 0.5,
         "q20": 0.5, "g": 0.0, "valid": 1},
    ])  # fmt: skip
    tabs = multiseed_table(wide, "R50")
    assert set(tabs) == {1, 2}
    assert tabs[2].loc[2, 20] == 0.9 and math.isnan(tabs[2].loc[2, 15])
    res = cross_seed_oracle(tabs, 1, 2, k=1)
    assert res["chosen_ids"] == "1" and res["n"] == 2
    with pytest.raises(ValueError, match="duplicate"):
        multiseed_table(pd.concat([wide, wide.iloc[[0]]]), "R50")


def test_bootstrap_reports_difference_to_best_fixed_mixture() -> None:
    # Non-concave mean curve: zero except t5 = .5 and t20 = 1 -> F(10) = .5 + .5/3.
    ids = list(range(1, 5))
    curve = np.zeros(L)
    curve[4], curve[19] = 0.5, 1.0
    q = {i: curve.copy() for i in ids}
    curves = robovis_eval.RecallCurves(q, {i: np.ones(L, bool) for i in ids}, L)
    eval_set = robovis_eval.build_eval_set(curves, 0.9)
    rows = [{"image_id": i, "method": "m", "stop_t": 10, "q_stop": 0.0, "residual": 1.0}
            for i in ids]  # fmt: skip
    boot = robovis_eval.paired_bootstrap(pd.DataFrame(rows), curves, eval_set, ["m"], [],
                                         20, 0, 0.95)  # fmt: skip
    f10 = 0.5 + 0.5 * 5 / 15
    assert boot["methods"]["m"]["q_minus_F"]["mean"] == pytest.approx(-f10)
    # Legacy adjacent mix at 10 is 0 here, so the legacy comparison looks neutral.
    assert boot["methods"]["m"]["q_minus_eq_fixed"]["mean"] == pytest.approx(0.0)
    assert robovis_eval.best_fixed_value(np.tile(curve, (4, 1)), 10) == pytest.approx(
        f10
    )


def test_tie_order_sensitivity_spans_tied_choices() -> None:
    # Two tied zero-gain images compete for the last t20 slot.
    rows = [
        (1, 1, 10, 0.0), (1, 1, 20, 0.5), (2, 1, 10, 0.2), (2, 1, 20, 0.2),
        (3, 1, 10, 0.2), (3, 1, 20, 0.2),
        (1, 2, 10, 0.0), (1, 2, 20, 0.5), (2, 2, 10, 0.0), (2, 2, 20, 0.4),
        (3, 2, 10, 0.2), (3, 2, 20, 0.2),
    ]  # fmt: skip
    res = cross_seed_oracle(multiseed_table(_multiseed(rows), "R50"), 1, 2, k=2)
    assert res["n_gain_pos_select"] == 1 and res["n_gain_zero_select"] == 2
    assert (
        res["tie_random_p05"] < res["tie_random_p95"]
    )  # outcome depends on the tie order
    assert res["tie_random_p05"] <= res["cross_seed_gain"] <= res["tie_random_p95"]
