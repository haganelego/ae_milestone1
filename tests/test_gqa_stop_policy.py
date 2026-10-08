"""Stop rule (test 5), prefix-only policy (test 15), dev-only freeze (test 11)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
from gqa_stop_helpers import CONFIG_PATH

from milestone1.gqa_stop.policy import (
    NEVER_STOP,
    SplitTuningError,
    fixed_dev_level,
    freeze_policy,
    guard_selection,
    online_stop,
    replay_stop,
    select_threshold,
    select_thresholds,
    threshold_grid,
)

POLICY = json.loads(CONFIG_PATH.read_text())["policy"]
N = 20


def scores(values: dict[int, float]) -> list[float]:
    return [values.get(t, 0.0) for t in range(1, N + 1)]


# ---- test 5 ---------------------------------------------------------------------
def test_threshold_is_inclusive() -> None:
    s = scores({4: 0.7, 9: 0.9})
    assert replay_stop(s, 0.7).level == 4
    assert replay_stop(s, 0.71).level == 9


def test_stop_at_level_one() -> None:
    d = replay_stop(scores({1: 0.5}), 0.5)
    assert d.level == 1 and not d.reached_last and d.reason == "threshold_met"


def test_all_unknown_forces_level_20() -> None:
    d = replay_stop(scores({}), 0.01)
    assert d.level == 20 and d.never_met and d.reached_last and not d.passed_at_last
    assert d.reason == "never_met_forced_last_level"


def test_infinite_threshold_always_20() -> None:
    d = replay_stop([1.0] * N, NEVER_STOP)
    assert d.level == 20 and d.never_met and d.reached_last


def test_pass_exactly_at_20_is_not_forced() -> None:
    d = replay_stop(scores({20: 0.9}), 0.9)
    # §7.2 Forced20 counts it (reached L20); the reason is recorded separately
    assert d.level == 20 and d.reached_last and not d.never_met and d.passed_at_last
    assert d.reason == "threshold_met_at_last_level"


def test_grid_contents() -> None:
    grid = threshold_grid(POLICY)
    assert grid[0] == 0.01 and grid[99] == 1.0 and grid[-1] == math.inf
    assert len(grid) == 101 and 0.0 not in grid


# ---- test 15 ---------------------------------------------------------------------
def test_online_policy_never_reads_future_levels() -> None:
    seen: list[int] = []
    full = scores({6: 0.95, 12: 1.0})

    def score_at(t: int) -> float:
        seen.append(t)
        return full[t - 1]

    decision = online_stop(score_at, 0.9, N)
    assert decision.level == 6
    assert max(seen) == 6 and seen == list(range(1, 7))


@pytest.mark.parametrize("theta", [0.05, 0.3, 0.8, 0.95, NEVER_STOP])
def test_prefix_run_equals_offline_replay(theta: float) -> None:
    full = [min(1.0, 0.04 * t + (0.3 if t % 5 == 0 else 0.0)) for t in range(1, N + 1)]
    for cut in range(1, N + 1):
        prefix = full[:cut]
        decision = replay_stop(full, theta)
        if decision.level <= cut:
            assert (
                online_stop(lambda t, p=prefix: p[t - 1], theta, cut).level
                == decision.level
            )


# ---- selection rule ------------------------------------------------------------------
def grid_frame(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"theta": t, "accuracy": a, "mean_level": m} for t, a, m in rows]
    )


def test_selection_min_level_then_accuracy_then_larger_theta() -> None:
    g = grid_frame(
        [
            (0.1, 0.90, 2.0),
            (0.2, 0.95, 5.0),
            (0.3, 0.97, 5.0),
            (0.4, 0.97, 5.0),
            (math.inf, 1.0, 20.0),
        ]
    )
    chosen = select_threshold(g, 0.95)
    assert chosen["theta"] == 0.4 and chosen["feasible"]


def test_selection_falls_back_to_never_stop() -> None:
    g = grid_frame([(0.1, 0.5, 1.0), (math.inf, 1.0, 20.0)])
    assert select_threshold(g, 0.95)["theta"] == math.inf


def test_fixed_dev_is_earliest_feasible_level() -> None:
    curve = {t: (0.8 if t < 7 else 0.96) for t in range(1, 21)}
    assert fixed_dev_level(curve, 0.95) == 7
    assert fixed_dev_level({1: 0.5}, 0.95) is None


# ---- test 11 -------------------------------------------------------------------------
def level_table(split: str = "dev") -> pd.DataFrame:
    rows = []
    for q in range(4):
        for t in range(1, N + 1):
            rows.append(
                {
                    "question_id": f"q{q}",
                    "image_id": f"i{q}",
                    "group_id": f"i{q}",
                    "split": split,
                    "level": t,
                    "reader_correct": t >= 3 + q,
                    "direct_score": min(1.0, t / 10),
                    "checklist_min": min(1.0, t / 15),
                    "checklist_mean": min(1.0, t / 12),
                }
            )
    return pd.DataFrame(rows)


def test_freeze_on_dev_then_test_selection_raises(tmp_path: Path) -> None:
    frozen = freeze_policy(level_table(), POLICY, N, tmp_path, {"model": "m"})
    assert (tmp_path / "frozen_policy.json").exists()
    assert (tmp_path / "dev_threshold_grid.csv").exists()
    assert set(frozen["selections"]) == {"0.95", "0.90", "0.97", "0.99"}
    json.loads((tmp_path / "frozen_policy.json").read_text())  # strict JSON
    with pytest.raises(SplitTuningError):
        select_thresholds(
            level_table("test"), "test", POLICY, N, tmp_path / "frozen_policy.json"
        )
    with pytest.raises(SplitTuningError):
        freeze_policy(level_table(), POLICY, N, tmp_path, {})


def test_selection_rejects_test_split_even_before_freeze(tmp_path: Path) -> None:
    with pytest.raises(SplitTuningError):
        guard_selection(level_table(), "test", tmp_path / "none.json")
    with pytest.raises(SplitTuningError):
        guard_selection(level_table("test"), "dev", tmp_path / "none.json")


def test_frozen_selection_meets_dev_target(tmp_path: Path) -> None:
    frozen = freeze_policy(level_table(), POLICY, N, tmp_path, {})
    for method in ("direct", "checklist_min", "checklist_mean"):
        chosen = frozen["selections"]["0.95"][method]
        assert chosen["accuracy"] >= 0.95
    assert frozen["selections"]["0.95"]["fixed_dev"] == 6
