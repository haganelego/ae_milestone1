"""Exploratory replays on cached v4c records (orchestrator request, NOT
pre-registered; logged only, never used for a decision). No new inference.

- soft_existence: stop score = P_ans x Rel of the LAST (attribute) item only,
  gated only by that item's own ``resolved`` flag; the existence item is
  excluded from both the min and the ancestor gate.
- k = 2: a stop needs score >= θ at 2 consecutive levels (forced L20 unchanged).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .common import atomic_write_json, read_id_csv
from .devgrid import fixed_curve
from .evaluate import hull_value
from .policy import (
    ACCURACY_TOLERANCE,
    ConsistencyVariant,
    correctness_matrix,
    question_scores,
    replay_stop_consecutive,
    threshold_grid,
)

LABEL = "exploratory (not pre-registered; logged only)"
SOFT_COLUMN = "soft_existence"
MAX_ITEMS = 3


def last_item_index(row: Mapping[str, Any]) -> int:
    n = int(row.get("n_items", 0) or 0)
    return min(n, MAX_ITEMS)


def soft_existence_score(row: Mapping[str, Any]) -> float:
    """P_ans x Rel of the last item if that item is resolved, else 0."""
    index = last_item_index(row)
    if index == 0:
        return 0.0
    resolved = str(row.get(f"item{index}_resolved")) == "True"
    valid = str(row.get(f"item{index}_valid")) == "True"
    product = row.get(f"item{index}_product")
    if not (resolved and valid) or product is None or pd.isna(product):
        return 0.0
    return float(product)


def add_soft_existence(table: pd.DataFrame) -> pd.DataFrame:
    scores = [soft_existence_score(r) for r in table.to_dict("records")]
    return table.assign(**{SOFT_COLUMN: scores})


def replay(table: pd.DataFrame, column: str, theta: float, k: int) -> dict[str, Any]:
    """Accuracy, mean level, stop20 and stopped-and-wrong ids for one setting."""
    correct = correctness_matrix(table)
    variant = ConsistencyVariant(theta=theta, k=k)  # θ is a float here
    decisions = {
        q: replay_stop_consecutive(s, variant)
        for q, s in question_scores(table, column).items()
    }
    n_levels = len(next(iter(correct.values())))
    levels = {q: d.level for q, d in decisions.items()}
    wrong_ids = sorted(
        q
        for q, d in decisions.items()
        if not d.never_met and not correct[q][d.level - 1]
    )
    n = len(levels)
    return {
        "accuracy": sum(correct[q][levels[q] - 1] for q in levels) / n,
        "mean_level": sum(levels.values()) / n,
        "stop20_rate": sum(1 for v in levels.values() if v == n_levels) / n,
        "stopped_but_wrong": len(wrong_ids),
        "stopped_wrong_ids": wrong_ids,
    }


def best_setting(
    table: pd.DataFrame, column: str, k: int, grid: Sequence[float], target: float
) -> dict[str, Any] | None:
    """Min mean level with accuracy >= target; ties: higher accuracy, larger θ."""
    rows = [{"theta": theta, **replay(table, column, theta, k)} for theta in grid]
    feasible = [r for r in rows if r["accuracy"] >= target - ACCURACY_TOLERANCE]
    if not feasible:
        return None
    return min(feasible, key=lambda r: (r["mean_level"], -r["accuracy"], -r["theta"]))


SETTINGS = (
    ("checklist_v4_min", "checklist_v4_min", 1, "pre-registered"),
    ("direct_v4", "direct_v4", 1, "comparator"),
    ("checklist_v1_min (v1 arm)", "checklist_v1_min", 1, "comparator (fair v1)"),
    ("soft_existence", SOFT_COLUMN, 1, LABEL),
    ("checklist_v4_min k=2", "checklist_v4_min", 2, LABEL),
    ("direct_v4 k=2", "direct_v4", 2, LABEL),
    ("soft_existence k=2", SOFT_COLUMN, 2, LABEL),
)


def run_exploration(
    table: pd.DataFrame, policy_cfg: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    target = float(policy_cfg["primary_dev_accuracy_target"])
    grid = threshold_grid(policy_cfg)
    table = add_soft_existence(table)
    curve = fixed_curve(table)
    rows = []
    for name, column, k, role in SETTINGS:
        best = best_setting(table, column, k, grid, target)
        if best is None:
            rows.append({"method": name, "k": k, "role": role, "feasible": False})
            continue
        hull = hull_value(curve, best["mean_level"])
        rows.append(
            {
                "method": name,
                "k": k,
                "role": role,
                "feasible": True,
                **best,
                "accuracy_minus_hull": best["accuracy"] - hull,
            }
        )
    fixed_dev = next(
        (t for t in sorted(curve) if curve[t] >= target - ACCURACY_TOLERANCE),
        max(curve),
    )
    summary = {
        "label": LABEL,
        "target_accuracy": target,
        "n_questions": int(table["question_id"].nunique()),
        "settings": rows,
        "subset_fixed_curve": {str(t): curve[t] for t in sorted(curve)},
        "subset_fixed_dev": fixed_dev,
        "theta_note": "best θ chosen on the same dev questions (in-sample)",
    }
    frame = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "stopped_wrong_ids"} for r in rows]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "dev_exploratory.csv", index=False)
    atomic_write_json(output_dir / "dev_exploratory.json", summary)
    return summary


def merged_table(level_table: Path, v1_table: Path) -> pd.DataFrame:
    table = read_id_csv(level_table)
    if "checklist_v1_min" in table.columns:
        return table
    v1 = read_id_csv(v1_table)[["question_id", "level", "checklist_min"]]
    return table.merge(
        v1.rename(columns={"checklist_min": "checklist_v1_min"}),
        on=["question_id", "level"],
        how="left",
    )
