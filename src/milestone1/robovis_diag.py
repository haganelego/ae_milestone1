"""Per-state signal diagnostics (no k-consecutive replay, no model calls).

For every (image, level) that has check/decision files, computes the stop-rule
signals (A_t, H_t, C_t per condition) plus item-level summaries, joins them
with recall from ``stage_metrics.csv`` and summarises how the signals relate
to the level and to the recall still left to gain.

Outputs (under ``out_dir``)::

    signal_per_state.csv    one row per state
    signal_by_level.csv     per level: n, mean/std of each signal, frac C>=theta
(q_t = selected metric, ``evaluation.q_metric`` or override; R50 by default)
    signal_stats.json       Spearman correlations and AUCs
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from milestone1.robovis_data import level_name
from milestone1.robovis_eval import FLOAT_TOL, configured_q_metric, load_stage_metrics
from milestone1.robovis_judge_quality import add_judge_columns, judge_quality
from milestone1.robovis_replay import RunLayout, read_json_or_none
from milestone1.robovis_stats import auc, spearman_stat
from milestone1.stop_policy import (
    BENEFIT_GATED,
    LIMITATION_LOW_RES,
    OUTCOME_EMPTY_GRAPH,
    OUTCOME_OK,
    PROPOSED,
    STATUS_UNRESOLVED,
    Condition,
    ParsedLevel,
    build_conditions,
    item_utility,
    parse_level_outputs,
    parsed_level_score,
)
from milestone1.structured_io import ContractLimits, GraphIndex

STATUS_RESOLVED = "resolved"
OK_OUTCOMES = (OUTCOME_OK, OUTCOME_EMPTY_GRAPH)

JUDGE_QUALITY_FILENAME = "judge_quality.csv"
LEVEL_FILE_RE = re.compile(r"^level_(\d+)\.json$")
# Raw recall columns copied to signal_per_state for reference (if present).
RAW_RECALL_COLUMNS = ("R50", "R100")
# Item-level signals besides the per-condition C_t columns.
EXTRA_CORR_SIGNALS = ("H", "A", "H_bg", "A_bg", "max_basis", "max_benefit")
CORR_TARGETS = ("t", "gain_left", "frac_of_qmax")
LEVEL_ITEM_SIGNALS = (
    "A",
    "H",
    "A_bg",
    "H_bg",
    "max_basis",
    "max_benefit",
    "max_u",
    "n_gate_items",
    "n_items",
    "q",
    "gain_left",
)


def c_column(condition: str) -> str:
    return f"C_{condition}"


def corr_signals(conditions: Mapping[str, Condition]) -> list[str]:
    return [c_column(n) for n in conditions] + list(EXTRA_CORR_SIGNALS)


def level_signals(conditions: Mapping[str, Condition]) -> list[str]:
    return [c_column(n) for n in conditions] + list(LEVEL_ITEM_SIGNALS)


# --------------------------------------------------------------------------
# Per-state signals
# --------------------------------------------------------------------------


def _passes_gate(item: Any, status: str, limitation: str, graph: GraphIndex) -> bool:
    """Gate of plan §6 apart from the basis/benefit values."""
    anchors_ok = bool(item.anchor_ids) and all(
        a in graph.entity_ids for a in item.anchor_ids
    )
    region_ok = (
        item.region_anchor_id is not None and item.region_anchor_id in graph.entity_ids
    )
    return (
        anchors_ok
        and region_ok
        and status == STATUS_UNRESOLVED
        and limitation == LIMITATION_LOW_RES
    )


def state_signals(
    parsed: ParsedLevel | None, conditions: Mapping[str, Condition]
) -> dict[str, Any]:
    """Signals of one state; NaN everywhere if the state is missing/failed."""
    names = [c_column(n) for n in conditions]
    keys = ["A", "H", "A_bg", "H_bg", *names, "max_basis", "max_benefit", "max_u"]
    counts = ["n_observed", "n_hypotheses", "n_items", "n_gate_items",
              "n_gate_observed", "n_gate_hypotheses", "n_resolved", "n_lowres"]  # fmt: skip
    if parsed is None:
        return dict.fromkeys(keys + counts, math.nan)

    out: dict[str, Any] = {}
    proposed = parsed_level_score(parsed, conditions[PROPOSED])
    out["A"], out["H"] = proposed.a, proposed.h
    if BENEFIT_GATED in conditions:
        bg = parsed_level_score(parsed, conditions[BENEFIT_GATED])
        out["A_bg"], out["H_bg"] = bg.a, bg.h
    else:
        out["A_bg"], out["H_bg"] = math.nan, math.nan
    for name, cond in conditions.items():
        out[c_column(name)] = parsed_level_score(parsed, cond).c
    if parsed.checks is None or parsed.judgments is None:
        # Empty observed graph: no items were (or need to be) judged.
        return {**out, "max_basis": 0, "max_benefit": 0, "max_u": 0,
                **dict.fromkeys(counts, 0)}  # fmt: skip

    by_id = parsed.judgments.by_id()
    items = [*parsed.checks.observed_issues, *parsed.checks.hypotheses]
    n_obs = len(parsed.checks.observed_issues)
    judged = [by_id[it.id] for it in items]
    gate = [
        _passes_gate(it, j.status, j.limitation, parsed.graph)
        for it, j in zip(items, judged, strict=True)
    ]
    return {
        **out,
        "max_basis": max((j.basis for j in judged), default=0),
        "max_benefit": max((j.resolution_benefit for j in judged), default=0),
        "max_u": max(
            (
                item_utility(it, j, parsed.graph)
                for it, j in zip(items, judged, strict=True)
            ),
            default=0,
        ),
        "n_observed": n_obs,
        "n_hypotheses": len(items) - n_obs,
        "n_items": len(items),
        "n_gate_items": sum(gate),
        "n_gate_observed": sum(gate[:n_obs]),
        "n_gate_hypotheses": sum(gate[n_obs:]),
        "n_resolved": sum(j.status == STATUS_RESOLVED for j in judged),
        "n_lowres": sum(j.limitation == LIMITATION_LOW_RES for j in judged),
    }


def detect_levels(input_dir: Path, layout: RunLayout) -> list[int]:
    """Levels that have at least one checks or decisions file."""
    levels: set[int] = set()
    for sub in (layout.checks, layout.decisions):
        for path in (input_dir / sub).glob("*/level_*.json"):
            match = LEVEL_FILE_RE.match(path.name)
            if match:
                levels.add(int(match.group(1)))
    return sorted(levels)


def collect_states(
    input_dir: Path,
    stage: pd.DataFrame,
    levels: Sequence[int],
    conditions: Mapping[str, Condition],
    limits: ContractLimits,
    layout: RunLayout,
    q_metric: str,
) -> pd.DataFrame:
    """signal_per_state joined with q_t (selected metric), raw R50/R100, qmax."""
    col = q_metric
    raw_cols = tuple(n for n in RAW_RECALL_COLUMNS if n in stage.columns)
    stage = stage.assign(
        **{n: pd.to_numeric(stage[n], errors="coerce") for n in raw_cols}
    )
    valid_stage = stage[stage["valid"]]
    qmax = valid_stage.groupby("image_id")[col].max()
    stage_idx = stage.set_index(["image_id", "t"])
    rows: list[dict[str, Any]] = []
    for image_id in sorted(stage["image_id"].unique()):
        for t in levels:
            fname = f"{level_name(t)}.json"
            graph, _ = read_json_or_none(
                input_dir / layout.graphs / str(image_id) / fname
            )
            checks, _ = read_json_or_none(
                input_dir / layout.checks / str(image_id) / fname
            )
            decision, _ = read_json_or_none(
                input_dir / layout.decisions / str(image_id) / fname
            )
            parsed, outcome = parse_level_outputs(graph, checks, decision, limits)
            key = (int(image_id), int(t))
            srow = stage_idx.loc[key] if key in stage_idx.index else None
            q_t = float(srow[col]) if srow is not None else math.nan
            qm = float(qmax.get(image_id, math.nan))
            rows.append(
                {
                    "image_id": int(image_id),
                    "t": int(t),
                    "outcome": outcome,
                    "graph_empty": outcome == OUTCOME_EMPTY_GRAPH,
                    "n_entities": len(parsed.graph.entity_ids) if parsed else math.nan,
                    **state_signals(parsed, conditions),
                    "valid": bool(srow["valid"]) if srow is not None else False,
                    "q_metric": q_metric,
                    "q": q_t,
                    **{
                        name: float(srow[name]) if srow is not None else math.nan
                        for name in raw_cols
                    },
                    "qmax": qm,
                    "gain_left": qm - q_t,
                    "frac_of_qmax": q_t / qm if qm > 0 else math.nan,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def by_level(states: pd.DataFrame, conditions: Mapping[str, Condition]) -> pd.DataFrame:
    rows = []
    for t, grp in states.groupby("t", sort=True):
        ok = grp[grp["outcome"].isin(OK_OUTCOMES)]
        row: dict[str, Any] = {
            "t": t,
            "n_states": len(grp),
            "n": len(ok),
            "n_missing": len(grp) - len(ok),
            "n_empty_graph": int(grp["graph_empty"].sum()),
        }
        for sig in level_signals(conditions):
            values = ok[sig].astype(float)
            row[f"{sig}_mean"] = float(values.mean()) if len(values) else math.nan
            row[f"{sig}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else math.nan
            )
        for name, cond in conditions.items():
            c = ok[c_column(name)].astype(float)
            row[f"frac_C_{name}_ge_theta"] = (
                float((c >= cond.theta).mean()) if len(c) else math.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def signal_stats(
    states: pd.DataFrame, conditions: Mapping[str, Condition]
) -> dict[str, Any]:
    """Spearman(signal, target) over valid states, and AUCs for gain_left > 0."""
    ok = states[states["outcome"].isin(OK_OUTCOMES) & states["valid"]]
    spearman = {
        sig: {tgt: spearman_stat(ok[sig], ok[tgt]) for tgt in CORR_TARGETS}
        for sig in corr_signals(conditions)
    }
    aucs: dict[str, Any] = {}
    for subset_name, subset in (("all", ok), ("qmax_gt_0", ok[ok["qmax"] > 0])):
        labels = (subset["gain_left"] > FLOAT_TOL).to_numpy()
        entry: dict[str, Any] = {
            "n": len(subset),
            "n_pos": int(labels.sum()),
        }
        for name, cond in conditions.items():
            c = subset[c_column(name)].to_numpy(float)
            entry[f"C_{name}_score"] = auc(c, labels)
            entry[f"C_{name}_ge_theta"] = auc((c >= cond.theta).astype(float), labels)
        aucs[subset_name] = entry
    return {
        "n_states": len(states),
        "n_valid_states": len(ok),
        "outcomes": states["outcome"].value_counts().to_dict(),
        "spearman": spearman,
        "auc_gain_left_pos": aucs,
        "condition_status": {n: c.status_label for n, c in conditions.items()},
        "note": "states pooled over images and levels (not independent); "
        "frac_of_qmax only for qmax > 0",
    }


@dataclass(frozen=True)
class DiagResult:
    states: pd.DataFrame
    levels: pd.DataFrame
    stats: dict[str, Any]


def run_signal_diag(
    input_dir: Path,
    out_dir: Path,
    cfg: Mapping[str, Any],
    layout: RunLayout,
    levels: Sequence[int] | None = None,
    q_metric: str | None = None,
) -> DiagResult:
    """Per-state signals vs the selected q_t metric (gain_left = qmax - q_t)."""
    q_metric = configured_q_metric(cfg, q_metric)
    conditions = build_conditions(cfg)
    limits = ContractLimits.from_config(cfg)
    stage = load_stage_metrics(input_dir / "stage_metrics.csv", q_metric)
    levels = list(levels) if levels else detect_levels(input_dir, layout)
    if not levels:
        raise FileNotFoundError(f"no level files under {input_dir / layout.checks}")
    states = collect_states(
        input_dir, stage, levels, conditions, limits, layout, q_metric
    )
    per_level = by_level(states, conditions)
    theta = int(cfg["proposal"]["theta"])
    ratio = float(cfg["evaluation"]["oracle_ratio"])
    jq_table, jq_stats = judge_quality(
        states, theta, ratio, [c_column(n) for n in conditions]
    )
    states = add_judge_columns(states, theta)
    stats = {
        "q_metric": q_metric,
        "graphs_dir": layout.graphs,
        "levels": levels,
        **signal_stats(states, conditions),
        "judge_quality": jq_stats,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    states.to_csv(out_dir / "signal_per_state.csv", index=False)
    per_level.to_csv(out_dir / "signal_by_level.csv", index=False)
    jq_table.to_csv(out_dir / JUDGE_QUALITY_FILENAME, index=False)
    (out_dir / "signal_stats.json").write_text(json.dumps(stats, indent=2))
    return DiagResult(states, per_level, stats)


def bootstrap_mean_ci(
    values: np.ndarray, repeats: int, seed: int, ci_level: float
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (NaN if fewer than 2 values)."""
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (repeats, len(values)))].mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# Cross-path comparison
# --------------------------------------------------------------------------

COMPARE_AUC_SUBSET = "qmax_gt_0"


def _judge_quality_cells(jq: Mapping[str, Any], condition: str) -> dict[str, Any]:
    """signal_compare columns from the judge-quality diagnostic."""
    if not jq:
        return {}
    return {
        "rho_resolved_t": jq["rho_resolved_t"]["rho"],
        "rho_resolved_qratio": jq["rho_resolved_qratio"]["rho"],
        "auc_sufficient_C": jq["auc_sufficient_C"].get(c_column(condition), math.nan),
        "auc_sufficient_resolved": jq["auc_sufficient_resolved"],
        # Diagnostic only (not a stop condition).
        "rho_lowres_qratio": jq["rho_lowres_qratio"]["rho"],
        "auc_sufficient_lowres": jq["auc_sufficient_lowres"],
        "n_sufficient": jq["n_sufficient"],
    }


def compare_paths(
    results: Mapping[str, DiagResult],
    conditions: Mapping[str, Condition],
    compare_levels: Sequence[int],
) -> pd.DataFrame:
    """One row per (path, condition) for ranking check/judge variants."""
    rows = []
    for path, res in results.items():
        lvl = res.levels.set_index("t")
        st = res.stats
        n_valid = int(st["n_valid_states"])
        for name, cond in conditions.items():
            col = c_column(name)
            row: dict[str, Any] = {
                "path": path,
                "q_metric": st.get("q_metric"),
                "graphs_dir": st.get("graphs_dir"),
                "condition": name,
                "status": cond.status_label,
                "n_states": st["n_states"],
                "n_valid": n_valid,
                "failure_rate": 1.0 - n_valid / st["n_states"]
                if st["n_states"]
                else math.nan,
                "mean_C": float(res.states[col].mean()),
            }
            for t in compare_levels:
                key = f"frac_C_{name}_ge_theta"
                row[f"frac_ge_theta_t{t:02d}"] = (
                    float(lvl.loc[t, key]) if t in lvl.index else math.nan
                )
            for tgt in ("t", "gain_left"):
                sp = st["spearman"][col][tgt]
                row[f"rho_{tgt}"] = sp["rho"]
                row[f"p_{tgt}"] = sp["p"]
            aucs = st["auc_gain_left_pos"][COMPARE_AUC_SUBSET]
            row["auc_score"] = aucs[f"C_{name}_score"]
            row["auc_ge_theta"] = aucs[f"C_{name}_ge_theta"]
            row["auc_subset"] = COMPARE_AUC_SUBSET
            row.update(_judge_quality_cells(st.get("judge_quality") or {}, name))
            rows.append(row)
    return pd.DataFrame(rows)
