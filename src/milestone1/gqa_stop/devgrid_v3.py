"""Dev-only grid for v3 (plan §18.4). Never freezes.

Per method (θ grid 0.01..1.00 + inf, k = 1): accuracy, mean/median level,
stop20, stopped-but-wrong; best feasible θ at the 0.95 target (min mean level,
then higher accuracy, then larger θ); stop histogram of that θ; component
trends by level; comparison with fair v1 Checklist-min (its subset-frozen θ
replayed on v1's dev table restricted to the same questions) and the subset
Fixed-t curve / hull. A paired image-cluster bootstrap of the mean-level
difference (v3 best − fair v1) on dev is reported; θ was chosen on the same
dev questions, so this difference is optimistic (in-sample).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import atomic_write_json, read_id_csv, read_json
from .devgrid import fixed_curve, summarize
from .evaluate import cluster_bootstrap_indices
from .policy import (
    ACCURACY_TOLERANCE,
    StopDecision,
    correctness_matrix,
    question_scores,
    replay_stop,
    theta_from_json,
    threshold_grid,
)
from .sampled import METHOD_COLUMNS

PERCENTILES = (2.5, 97.5)


def decisions_for(
    scores: Mapping[str, Sequence[float]], theta: float
) -> dict[str, StopDecision]:
    return {q: replay_stop(s, theta) for q, s in scores.items()}


def method_grid(
    table: pd.DataFrame,
    grid: Sequence[float],
    method_columns: Mapping[str, str] = METHOD_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, dict[str, list[float]]]]:
    correct = correctness_matrix(table)
    curve = fixed_curve(table)
    rows = []
    all_scores = {}
    for method, column in method_columns.items():
        scores = question_scores(table, column)
        all_scores[method] = scores
        for theta in grid:
            rows.append(
                {
                    "method": method,
                    "theta": theta,
                    **summarize(decisions_for(scores, theta), correct, curve),
                }
            )
    return pd.DataFrame(rows), all_scores


def best_per_method(
    grid: pd.DataFrame, target: float
) -> dict[str, dict[str, Any] | None]:
    best: dict[str, dict[str, Any] | None] = {}
    for method, frame in grid.groupby("method", sort=False):
        ok = frame[frame["accuracy"] >= target - ACCURACY_TOLERANCE]
        if ok.empty:
            best[str(method)] = None
            continue
        ordered = ok.sort_values(
            ["mean_level", "accuracy", "theta"], ascending=[True, False, False]
        )
        best[str(method)] = ordered.iloc[0].to_dict()
    return best


def stop_hist(
    decisions: Mapping[str, StopDecision],
    correct: Mapping[str, Sequence[bool]],
    label: str,
) -> list[dict[str, Any]]:
    n_levels = len(next(iter(correct.values())))
    rows = []
    for level in range(1, n_levels + 1):
        at = [q for q, d in decisions.items() if d.level == level]
        rows.append(
            {
                "method": label,
                "level": level,
                "n_stopped": len(at),
                "n_wrong": sum(1 for q in at if not correct[q][level - 1]),
            }
        )
    return rows


def item_components(
    table: pd.DataFrame, best: Mapping[str, Any], method_columns: Mapping[str, str]
) -> pd.DataFrame:
    """v3 final (N = 1) component trends by level."""
    rows = []
    for level, frame in table.groupby("level", sort=True):
        wrong = ~frame["reader_correct"].astype(bool)
        row: dict[str, Any] = {
            "level": int(level),
            "reader_accuracy": float((~wrong).mean()),
            "item1_resolved_rate": float(frame["item1_resolved"].astype(bool).mean()),
            "checklist_min_zero_rate": float((frame["checklist_v3_min"] <= 0).mean()),
        }
        for method, column in method_columns.items():
            row[f"mean_{column}"] = float(frame[column].mean())
            chosen = best.get(method)
            if chosen is not None and math.isfinite(float(chosen["theta"])):
                hit = frame[column] >= float(chosen["theta"])
                row[f"p_{column}_ge_best_and_wrong"] = float((hit & wrong).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def components(
    table: pd.DataFrame,
    best: Mapping[str, Any],
    method_columns: Mapping[str, str] = METHOD_COLUMNS,
) -> pd.DataFrame:
    rows = []
    for level, frame in table.groupby("level", sort=True):
        wrong = ~frame["reader_correct"].astype(bool)
        item1 = frame["item1_resolved_frac"].dropna()
        row: dict[str, Any] = {
            "level": int(level),
            "reader_accuracy": float((~wrong).mean()),
            "item1_resolved_sample_rate": float(item1.mean()),
            "item1_any_resolved_rate": float((item1 > 0).mean()),
            "min_sagree_zero_rate": float((frame["v3_min_sagree"] == 0).mean()),
        }
        for method, column in method_columns.items():
            row[f"mean_{column}"] = float(frame[column].mean())
            chosen = best.get(method)
            if chosen is not None and math.isfinite(float(chosen["theta"])):
                hit = frame[column] >= float(chosen["theta"])
                row[f"p_{column}_ge_best_and_wrong"] = float((hit & wrong).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def fair_v1(
    v1_root: Path, frozen_path: Path, ids: set[str]
) -> tuple[dict[str, StopDecision], dict[str, list[bool]], float]:
    table = read_id_csv(v1_root / "level_table_dev.csv")
    table = table[table["question_id"].isin(ids)]
    frozen = read_json(frozen_path)
    theta = theta_from_json(
        frozen["selections"][frozen["primary_target"]]["checklist_min"]["theta"]
    )
    scores = question_scores(table, "checklist_min")
    return decisions_for(scores, theta), correctness_matrix(table), float(theta)


def mean_level_diff_ci(
    a: Mapping[str, StopDecision],
    b: Mapping[str, StopDecision],
    groups: Mapping[str, str],
    repeats: int,
    seed: int,
) -> dict[str, float]:
    qids = sorted(a)
    diff = np.array([a[q].level - b[q].level for q in qids], dtype=float)
    group_ids = [groups[q] for q in qids]
    draws = [
        diff[idx].mean() for idx in cluster_bootstrap_indices(group_ids, repeats, seed)
    ]
    low, high = np.percentile(draws, PERCENTILES)
    return {
        "estimate": float(diff.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def run_devgrid_v3(
    table: pd.DataFrame,
    policy_cfg: Mapping[str, Any],
    eval_cfg: Mapping[str, Any],
    output_dir: Path,
    v1_root: Path | None,
    fair_frozen: Path | None,
    method_columns: Mapping[str, str] = METHOD_COLUMNS,
    components_fn: Callable[..., pd.DataFrame] = components,
    abort_methods: Sequence[str] | None = None,
) -> dict[str, Any]:
    """``abort_methods``: methods whose significance decides §18.4 (default all)."""
    target = float(policy_cfg["primary_dev_accuracy_target"])
    grid, scores = method_grid(table, threshold_grid(policy_cfg), method_columns)
    best = best_per_method(grid, target)
    correct = correctness_matrix(table)
    curve = fixed_curve(table)
    hist_rows: list[dict[str, Any]] = []
    comparison = []
    best_decisions: dict[str, dict[str, StopDecision]] = {}
    for method, chosen in best.items():
        if chosen is None:
            comparison.append({"method": f"{method} (no θ reaches {target})"})
            continue
        decisions = decisions_for(scores[method], float(chosen["theta"]))
        best_decisions[method] = decisions
        hist_rows += stop_hist(decisions, correct, f"{method}@{chosen['theta']}")
        comparison.append(
            {
                "method": f"{method}@θ={chosen['theta']}",
                **summarize(decisions, correct, curve),
            }
        )
    groups = table.groupby("question_id")["group_id"].first().astype(str).to_dict()
    fair = None
    ci: dict[str, Any] = {}
    if v1_root is not None and fair_frozen is not None and fair_frozen.exists():
        fair, fair_correct, theta = fair_v1(v1_root, fair_frozen, set(correct))
        hist_rows += stop_hist(fair, fair_correct, f"fair_v1_checklist_min@{theta}")
        comparison.append(
            {
                "method": f"fair v1 checklist_min@θ={theta} (subset-frozen)",
                **summarize(fair, fair_correct, curve),
            }
        )
        for method, decisions in best_decisions.items():
            ci[method] = mean_level_diff_ci(
                decisions,
                fair,
                groups,
                int(eval_cfg["bootstrap_repeats"]),
                int(eval_cfg["bootstrap_seed"]),
            )
    n_levels = len(curve)
    fixed_dev = next(
        (t for t in sorted(curve) if curve[t] >= target - ACCURACY_TOLERANCE), n_levels
    )
    for t in sorted({5, 10, 15, fixed_dev, n_levels}):
        fixed = {q: StopDecision(t, False, t == n_levels, n_levels) for q in correct}
        label = f"subset fixed_{t}" + (" (Fixed-dev)" if t == fixed_dev else "")
        comparison.append({"method": label, **summarize(fixed, correct, curve)})
    output_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output_dir / "dev_threshold_grid.csv", index=False)
    pd.DataFrame(hist_rows).to_csv(output_dir / "dev_stop_hist.csv", index=False)
    components_fn(table, best, method_columns).to_csv(
        output_dir / "dev_components.csv", index=False
    )
    pd.DataFrame(comparison).to_csv(output_dir / "dev_comparison.csv", index=False)
    pd.DataFrame(
        {"level": list(curve), "subset_reader_accuracy": list(curve.values())}
    ).to_csv(output_dir / "dev_fixed_curve.csv", index=False)
    deciding = set(abort_methods) if abort_methods is not None else set(ci)
    significant = {m: v["ci_high"] < 0 for m, v in ci.items() if m in deciding}
    summary = {
        "n_questions": len(correct),
        "target_accuracy": target,
        "best_per_method": best,
        "mean_level_diff_vs_fair_v1": ci,
        "mean_level_diff_note": "paired image bootstrap on dev; θ chosen on the same "
        "dev questions, so the difference is optimistic (in-sample)",
        "abort_rule": "§18.4: abort unless a feasible setting's mean level is "
        "significantly below fair v1 (dev CI upper bound < 0)",
        "abort_methods": sorted(deciding),
        "abort": not any(significant.values()),
        "frozen_policy_written": False,
    }
    atomic_write_json(output_dir / "devgrid_summary.json", summary)
    return summary
