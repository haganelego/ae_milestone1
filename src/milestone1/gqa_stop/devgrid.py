"""Dev-only grid for the v2 consistency-stop (plan §17). Never freezes.

Outputs (in the v2 output root): dev_threshold_grid.csv, dev_stop_hist.csv,
dev_components.csv, dev_comparison.csv, dev_fixed_curve.csv and
devgrid_summary.json. GT is used only to score the common reader.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .common import atomic_write_json, read_id_csv, read_json
from .consistency import SCORE_COLUMN
from .evaluate import hull_value
from .policy import (
    ACCURACY_TOLERANCE,
    ConsistencyVariant,
    StopDecision,
    correctness_matrix,
    question_scores,
    replay_stop,
    replay_stop_consecutive,
    theta_from_json,
)

ABORT_MEAN_LEVEL = 18.0  # §17: feasible settings averaging >= 18 -> stop the line
V1_METHOD = "checklist_min"


def summarize(
    decisions: Mapping[str, StopDecision],
    correct: Mapping[str, Sequence[bool]],
    curve: Mapping[int, float],
) -> dict[str, Any]:
    levels = [decisions[q].level for q in correct]
    hits = [correct[q][decisions[q].level - 1] for q in correct]
    stopped_wrong = sum(
        1
        for q in correct
        if not decisions[q].never_met and not correct[q][decisions[q].level - 1]
    )
    n = len(levels)
    mean_level = sum(levels) / n
    accuracy = sum(hits) / n
    hull = hull_value(curve, mean_level)
    return {
        "n": n,
        "accuracy": accuracy,
        "mean_level": mean_level,
        "median_level": float(pd.Series(levels).median()),
        "stop20_rate": sum(1 for v in levels if v == len(curve)) / n,
        "never_met_rate": sum(d.never_met for d in decisions.values()) / n,
        "stopped_but_wrong": stopped_wrong,
        "wrong_total": n - sum(hits),
        "hull_accuracy_at_mean_level": hull,
        "accuracy_minus_hull": accuracy - hull,
    }


def fixed_curve(table: pd.DataFrame) -> dict[int, float]:
    return {
        int(level): float(frame["reader_correct"].astype(bool).mean())
        for level, frame in table.groupby("level", sort=True)
    }


def variant_decisions(
    table: pd.DataFrame, variant: ConsistencyVariant
) -> dict[str, StopDecision]:
    scores = question_scores(table, SCORE_COLUMN)
    return {q: replay_stop_consecutive(s, variant) for q, s in scores.items()}


def grid(table: pd.DataFrame, variants: Sequence[ConsistencyVariant]) -> pd.DataFrame:
    correct = correctness_matrix(table)
    curve = fixed_curve(table)
    rows = [
        {
            "variant": v.label,
            "theta": v.theta,
            "k": v.k,
            **summarize(variant_decisions(table, v), correct, curve),
        }
        for v in variants
    ]
    return pd.DataFrame(rows)


def stop_hist(
    table: pd.DataFrame, variants: Sequence[ConsistencyVariant]
) -> pd.DataFrame:
    correct = correctness_matrix(table)
    n_levels = int(table["level"].max())
    rows = []
    for v in variants:
        decisions = variant_decisions(table, v)
        for level in range(1, n_levels + 1):
            at = [q for q, d in decisions.items() if d.level == level]
            rows.append(
                {
                    "variant": v.label,
                    "level": level,
                    "n_stopped": len(at),
                    "n_wrong": sum(1 for q in at if not correct[q][level - 1]),
                }
            )
    return pd.DataFrame(rows)


def components(table: pd.DataFrame, thetas: Sequence[int]) -> pd.DataFrame:
    """Per level: a/b/c rates, a∧b∧c, conditional rates given a wrong reader."""
    rows = []
    for level, frame in table.groupby("level", sort=True):
        a, b, c = (frame[x].astype(bool) for x in ("a", "b", "c"))
        wrong = ~frame["reader_correct"].astype(bool)
        n_wrong = int(wrong.sum())
        row: dict[str, Any] = {
            "level": int(level),
            "n": len(frame),
            "reader_accuracy": float((~wrong).mean()),
            "a_rate": float(a.mean()),
            "b_rate": float(b.mean()),
            "c_rate": float(c.mean()),
            "abc_rate": float((a & b & c).mean()),
            "n_reader_wrong": n_wrong,
            "p_a_given_wrong": float(a[wrong].mean()) if n_wrong else math.nan,
            "p_b_given_wrong": float(b[wrong].mean()) if n_wrong else math.nan,
            "p_c_given_wrong": float(c[wrong].mean()) if n_wrong else math.nan,
            "blind_degenerate_rate": float(
                frame["blind_degenerate"].fillna(False).astype(bool).mean()
            ),
        }
        for theta in thetas:
            hit = frame[SCORE_COLUMN] >= theta
            row[f"p_s_ge_{theta}_and_wrong"] = float((hit & wrong).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def v1_comparison(
    v1_root: Path, n_levels: int
) -> tuple[pd.DataFrame, dict[int, float]]:
    """v1 Checklist-min at its frozen θ, Fixed-dev and every Fixed-t on v1 dev."""
    table = read_id_csv(v1_root / "level_table_dev.csv")
    frozen = read_json(v1_root / "frozen_policy.json")
    primary = frozen["primary_target"]
    theta = theta_from_json(frozen["selections"][primary][V1_METHOD]["theta"])
    correct = correctness_matrix(table)
    curve = fixed_curve(table)
    scores = question_scores(table, V1_METHOD)
    rows = [
        {
            "method": f"v1 {V1_METHOD}@{primary} (theta={theta})",
            **summarize(
                {q: replay_stop(s, theta) for q, s in scores.items()}, correct, curve
            ),
        }
    ]
    fixed_dev = frozen["selections"][primary].get("fixed_dev")
    for t in sorted({*range(1, n_levels + 1)}):
        label = f"v1 fixed_{t}" + (" (Fixed-dev)" if t == fixed_dev else "")
        decisions = {
            q: StopDecision(t, False, t == n_levels, n_levels) for q in correct
        }
        rows.append({"method": label, **summarize(decisions, correct, curve)})
    return pd.DataFrame(rows), curve


def best_feasible(frame: pd.DataFrame, target: float) -> dict[str, Any] | None:
    """Min mean level with accuracy >= target; ties: higher accuracy, larger θ, k."""
    ok = frame[frame["accuracy"] >= target - ACCURACY_TOLERANCE]
    if ok.empty:
        return None
    ordered = ok.sort_values(
        ["mean_level", "accuracy", "theta", "k"], ascending=[True, False, False, False]
    )
    return ordered.iloc[0].to_dict()


def run_devgrid(
    table: pd.DataFrame,
    variants: Sequence[ConsistencyVariant],
    target: float,
    output_dir: Path,
    v1_root: Path | None,
) -> dict[str, Any]:
    n_levels = int(table["level"].max())
    frame = grid(table, variants)
    hist = stop_hist(table, variants)
    comp = components(table, sorted({v.theta for v in variants}))
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "dev_threshold_grid.csv", index=False)
    hist.to_csv(output_dir / "dev_stop_hist.csv", index=False)
    comp.to_csv(output_dir / "dev_components.csv", index=False)
    v2_curve = fixed_curve(table)
    curves = {
        "level": list(range(1, n_levels + 1)),
        "v2_reader_accuracy": [v2_curve[t] for t in range(1, n_levels + 1)],
    }
    comparison = frame.rename(columns={"variant": "method"})
    if v1_root is not None:
        v1_frame, v1_curve = v1_comparison(v1_root, n_levels)
        curves["v1_reader_accuracy"] = [v1_curve[t] for t in range(1, n_levels + 1)]
        comparison = pd.concat([comparison, v1_frame], ignore_index=True)
    comparison.to_csv(output_dir / "dev_comparison.csv", index=False)
    pd.DataFrame(curves).to_csv(output_dir / "dev_fixed_curve.csv", index=False)
    best = best_feasible(frame, target)
    feasible = frame[frame["accuracy"] >= target - ACCURACY_TOLERANCE]
    abort = bool(feasible.empty or (feasible["mean_level"] >= ABORT_MEAN_LEVEL).all())
    summary = {
        "n_questions": int(table["question_id"].nunique()),
        "target_accuracy": target,
        "best_feasible_variant": best,
        "abort_rule": f"abort if every variant with dev accuracy >= {target} "
        f"has mean level >= {ABORT_MEAN_LEVEL} (plan §17)",
        "abort": abort,
        "frozen_policy_written": False,
        "note": "dev-only exploratory grid; thresholds are not frozen (user decision pending)",
    }
    atomic_write_json(output_dir / "devgrid_summary.json", summary)
    return summary
