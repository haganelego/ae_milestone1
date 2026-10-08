"""Threshold stop rule, fixed stops and dev-only threshold freezing (plan §6).

The stop rule is online: ``online_stop`` asks for the score of level t only
after deciding not to stop at t-1, so it can never see future levels. Offline
replay over cached scores goes through the same function (test 15).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import StageError, atomic_write_json, utc_now
from .inference import SCORE_COLUMNS

NEVER_STOP = math.inf
GRID_DECIMALS = 2
ACCURACY_TOLERANCE = 1e-12
DEV_SPLIT = "dev"


class SplitTuningError(StageError):
    """Threshold selection on anything but dev (or after freeze) is forbidden."""


REASON_MET = "threshold_met"
REASON_MET_AT_LAST = "threshold_met_at_last_level"
REASON_NEVER_MET = "never_met_forced_last_level"
REASON_FIXED = "fixed_level"
SUPPORTED_START_LEVEL = 1
SUPPORTED_CONSECUTIVE_K = 1


@dataclass(frozen=True)
class StopDecision:
    level: int
    never_met: bool  # threshold not met even at the last level
    passed_at_last: bool  # met for the first time exactly at the last level
    max_level: int = 20

    @property
    def reached_last(self) -> bool:
        """§7.2 Forced20: threshold not met at t < last, so the last level was reached."""
        return self.level == self.max_level

    @property
    def reason(self) -> str:
        if self.never_met:
            return REASON_NEVER_MET
        return REASON_MET_AT_LAST if self.passed_at_last else REASON_MET


@dataclass(frozen=True)
class PolicyParams:
    start_level: int
    consecutive_k: int
    max_level: int

    def to_dict(self) -> dict[str, int]:
        return {
            "start_level": self.start_level,
            "consecutive_k": self.consecutive_k,
            "force_stop_level": self.max_level,
        }


def policy_params(policy_cfg: Mapping[str, Any], n_levels: int) -> PolicyParams:
    """Read start_level / consecutive_k / force_stop_level; refuse unsupported values."""
    params = PolicyParams(
        start_level=int(policy_cfg["start_level"]),
        consecutive_k=int(policy_cfg["consecutive_k"]),
        max_level=int(policy_cfg["force_stop_level"]),
    )
    if params.start_level != SUPPORTED_START_LEVEL:
        raise ValueError(
            f"start_level must be 1 for spec v1.1, got {params.start_level}"
        )
    if params.consecutive_k != SUPPORTED_CONSECUTIVE_K:
        raise ValueError(
            f"consecutive_k must be 1 for spec v1.1, got {params.consecutive_k}"
        )
    if params.max_level != n_levels:
        raise ValueError(
            f"force_stop_level ({params.max_level}) must equal images.levels ({n_levels})"
        )
    return params


def threshold_grid(policy_cfg: Mapping[str, Any]) -> list[float]:
    grid_cfg = policy_cfg["threshold_grid"]
    start, stop, step = (float(grid_cfg[k]) for k in ("start", "stop", "step"))
    count = int(round((stop - start) / step)) + 1
    grid = [round(start + i * step, GRID_DECIMALS) for i in range(count)]
    if grid_cfg.get("include_never_stop", True):
        grid.append(NEVER_STOP)
    return grid


def online_stop(
    score_at: Callable[[int], float],
    theta: float,
    max_level: int,
    start_level: int = 1,
) -> StopDecision:
    """First t >= start with score(t) >= theta (k=1); otherwise max_level."""
    for level in range(start_level, max_level + 1):
        if score_at(level) >= theta:
            return StopDecision(level, False, level == max_level, max_level)
    return StopDecision(max_level, True, False, max_level)


def replay_stop(
    scores: Sequence[float],
    theta: float,
    start_level: int = 1,
    max_level: int | None = None,
) -> StopDecision:
    """Offline replay over a full score list (index 0 = Level 1)."""
    last = len(scores) if max_level is None else max_level
    if last > len(scores):
        raise ValueError("max_level beyond available scores")
    return online_stop(lambda t: scores[t - 1], theta, last, start_level)


def question_scores(table: pd.DataFrame, column: str) -> dict[str, list[float]]:
    """question_id -> scores ordered by level (requires levels 1..N contiguous)."""
    result: dict[str, list[float]] = {}
    for qid, frame in table.sort_values("level").groupby("question_id", sort=True):
        levels = frame["level"].tolist()
        if levels != list(range(1, len(levels) + 1)):
            raise ValueError(f"question {qid}: levels not contiguous from 1")
        result[str(qid)] = [float(v) for v in frame[column]]
    return result


def correctness_matrix(table: pd.DataFrame) -> dict[str, list[bool]]:
    return {
        str(qid): [bool(v) for v in frame.sort_values("level")["reader_correct"]]
        for qid, frame in table.groupby("question_id", sort=True)
    }


def method_stops(
    table: pd.DataFrame, method: str, theta: float, params: PolicyParams
) -> dict[str, StopDecision]:
    scores = question_scores(table, SCORE_COLUMNS[method])
    return {
        q: replay_stop(s, theta, params.start_level, params.max_level)
        for q, s in scores.items()
    }


def summarize_stops(
    stops: Mapping[str, int], correct: Mapping[str, Sequence[bool]], max_level: int
) -> dict[str, float]:
    levels = [stops[q] for q in correct]
    hits = [correct[q][stops[q] - 1] for q in correct]
    n = len(levels)
    if n == 0:
        return {"n": 0, "accuracy": math.nan, "mean_level": math.nan}
    return {
        "n": n,
        "accuracy": sum(hits) / n,
        "mean_level": sum(levels) / n,
        "median_level": float(pd.Series(levels).median()),
        "stop20_rate": sum(1 for v in levels if v == max_level) / n,
    }


def stop_reason_rates(decisions: Sequence[StopDecision]) -> dict[str, float]:
    """Forced20 per §7.2 (= stop at the last level) plus its two sub-reasons."""
    n = max(len(decisions), 1)
    return {
        "forced20_rate": sum(d.reached_last for d in decisions) / n,
        "never_met_rate": sum(d.never_met for d in decisions) / n,
        "natural_pass20_rate": sum(d.passed_at_last for d in decisions) / n,
    }


def grid_table(
    table: pd.DataFrame, method: str, grid: Sequence[float], params: PolicyParams
) -> pd.DataFrame:
    correct = correctness_matrix(table)
    rows = []
    for theta in grid:
        decisions = method_stops(table, method, theta, params)
        summary = summarize_stops(
            {q: d.level for q, d in decisions.items()}, correct, params.max_level
        )
        rates = stop_reason_rates(list(decisions.values()))
        rows.append({"method": method, "theta": theta, **summary, **rates})
    return pd.DataFrame(rows)


def select_threshold(grid: pd.DataFrame, target: float) -> dict[str, Any]:
    """Accuracy >= target; min mean level; ties -> higher accuracy -> larger theta."""
    feasible = grid[grid["accuracy"] >= target - ACCURACY_TOLERANCE]
    if feasible.empty:
        return {"theta": None, "feasible": False, "target": target}
    ordered = feasible.sort_values(
        ["mean_level", "accuracy", "theta"], ascending=[True, False, False]
    )
    best = ordered.iloc[0].to_dict()
    return {**best, "feasible": True, "target": target}


def fixed_curve(table: pd.DataFrame) -> dict[int, float]:
    return {
        int(level): float(frame["reader_correct"].mean())
        for level, frame in table.groupby("level", sort=True)
    }


def fixed_dev_level(curve: Mapping[int, float], target: float) -> int | None:
    for level in sorted(curve):
        if curve[level] >= target - ACCURACY_TOLERANCE:
            return level
    return None


def guard_selection(table: pd.DataFrame, split: str, frozen_path: Path) -> None:
    """Only dev may choose thresholds, and only before the policy is frozen."""
    if split != DEV_SPLIT:
        raise SplitTuningError(f"threshold selection on split={split!r} is forbidden")
    splits = set(table["split"].unique()) if "split" in table else {DEV_SPLIT}
    if splits - {DEV_SPLIT}:
        raise SplitTuningError(f"selection table contains non-dev rows: {splits}")
    if frozen_path.exists():
        raise SplitTuningError(f"policy already frozen at {frozen_path}")


def select_thresholds(
    table: pd.DataFrame,
    split: str,
    policy_cfg: Mapping[str, Any],
    max_level: int,
    frozen_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Dev grid for every method and every pre-registered accuracy target."""
    guard_selection(table, split, frozen_path)
    params = policy_params(policy_cfg, max_level)
    grid = threshold_grid(policy_cfg)
    targets = [
        float(policy_cfg["primary_dev_accuracy_target"]),
        *map(float, policy_cfg["secondary_dev_accuracy_targets"]),
    ]
    frames = [grid_table(table, m, grid, params) for m in policy_cfg["methods"]]
    full = pd.concat(frames, ignore_index=True)
    curve = fixed_curve(table)
    selections: dict[str, Any] = {}
    for target in targets:
        key = f"{target:.2f}"
        selections[key] = {
            m: select_threshold(full[full["method"] == m], target)
            for m in policy_cfg["methods"]
        }
        selections[key]["fixed_dev"] = fixed_dev_level(curve, target)
    return full, {"targets": targets, "selections": selections, "fixed_curve": curve}


def freeze_policy(
    table: pd.DataFrame,
    policy_cfg: Mapping[str, Any],
    max_level: int,
    output_dir: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Write dev_threshold_grid.csv and frozen_policy.json (refuses to re-freeze)."""
    frozen_path = output_dir / "frozen_policy.json"
    grid, selected = select_thresholds(
        table, DEV_SPLIT, policy_cfg, max_level, frozen_path
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output_dir / "dev_threshold_grid.csv", index=False)
    primary = f"{float(policy_cfg['primary_dev_accuracy_target']):.2f}"
    frozen = {
        "frozen_at": utc_now(),
        "primary_target": primary,
        "methods": list(policy_cfg["methods"]),
        "policy_params": policy_params(policy_cfg, max_level).to_dict(),
        "selections": _json_safe(selected["selections"]),
        "dev_fixed_curve": selected["fixed_curve"],
        "dev_question_ids": sorted(table["question_id"].unique().tolist()),
        "dev_image_ids": sorted(table["image_id"].unique().tolist()),
        **provenance,
    }
    atomic_write_json(frozen_path, frozen)
    return frozen


def _json_safe(obj: Any) -> Any:
    """inf -> "inf", numpy scalars -> Python, so the policy is strict JSON."""
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and math.isinf(obj):
        return "inf"
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def theta_from_json(value: Any) -> float | None:
    """Frozen theta; None marks an infeasible target (no theta met the target)."""
    if value is None:
        return None
    return NEVER_STOP if value == "inf" else float(value)
