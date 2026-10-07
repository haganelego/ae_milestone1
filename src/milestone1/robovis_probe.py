"""Image-only "view sufficiency" probe diagnostic (zero model calls).

Probe records live at ``<run>/probe/<cfg>/<image_id>/level_XX.json``, either
bare or as a run record with the payload under ``"probe"`` and a top-level
``"failure"`` flag::

    {"detail_fraction": 0-100, "sufficient": bool, "next_step_helps": bool,
     "reason": str}

For each probe configuration this module

- validates the records (failed / missing / malformed -> missing state),
- relates ``detail_fraction`` to the level and to the recovery ratio q_t/qmax,
- measures how well the probe predicts ``sufficient_gt = q_t/qmax >= ratio``,
- checks within-image monotonicity of ``detail_fraction`` in t, and
- replays two EXPLORATORY probe-only stop policies through the existing
  k-consecutive replay and evaluation (q_stop vs the equal-observation fixed
  mix, paired bootstrap):

  * ``probe_stop``: instant stop where ``sufficient`` is true;
  * ``probe_stop_next``: instant stop where ``next_step_helps`` is false.

  * ``probe_gate_proposed`` / ``probe_gate_observed_only``: C_t = max_u
    (continue) where the probe says not sufficient, else the judge's C_t of a
    chosen decisions path; a missing/failed probe is a missing C_t.

These policies are diagnostics, not plan conditions.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from milestone1.robovis_data import level_name
from milestone1.robovis_eval import (
    RecallCurves,
    build_curves,
    build_eval_set,
    fixed_stops,
    load_stage_metrics,
    paired_bootstrap,
    per_image_metrics,
    q_output_dir,
    summarize,
)
from milestone1.robovis_replay import EvalParams, RunLayout, score_image
from milestone1.robovis_stats import auc, spearman_stat
from milestone1.stop_policy import (
    OBSERVED_ONLY,
    PROPOSED,
    build_conditions,
    fixed_method_name,
    replay_stop,
)
from milestone1.structured_io import ContractLimits

PROBE_STOP = "probe_stop"
PROBE_STOP_NEXT = "probe_stop_next"
PROBE_POLICIES = (PROBE_STOP, PROBE_STOP_NEXT)
# Combined policies: probe says "not sufficient" -> C = max_u (continue, analogue
# of the plan's empty-graph clause); otherwise C = C_<condition> of a decisions path.
GATE_PREFIX = "probe_gate_"
GATED_CONDITIONS = (PROPOSED, OBSERVED_ONLY)
GATE_POLICIES = tuple(f"{GATE_PREFIX}{c}" for c in GATED_CONDITIONS)
GATE_STATUS = "exploratory (probe gate + judge C, not a plan condition)"
REFERENCE_STATUS = "reference (ungated condition of the decisions path)"
PROBE_STATUS = "exploratory (probe-only policy, not a plan condition)"
FIXED_STATUS = "baseline"
DETAIL_MIN, DETAIL_MAX = 0.0, 100.0
PROBE_KEYS = ("detail_fraction", "sufficient", "next_step_helps", "reason")
# Probe-run records nest the payload under this key (failure flag at top level).
NESTED_PROBE_KEY = "probe"
OUTCOME_OK = "ok"
OUTCOME_MISSING = "missing"
OUTCOME_FAILURE = "failure"
OUTCOME_INVALID = "invalid"
# replay_stop encoding: instant stop <=> C < theta. A "stop" signal maps to 0.
STOP_SIGNAL_C = 0
CONTINUE_C = 1
PROBE_THETA = 1


@dataclass(frozen=True)
class ProbeRecord:
    detail_fraction: float
    sufficient: bool
    next_step_helps: bool


def parse_probe(obj: Any) -> tuple[ProbeRecord | None, str]:
    """Validate one probe record; returns (record or None, outcome tag)."""
    if obj is None:
        return None, OUTCOME_MISSING
    if not isinstance(obj, dict):
        return None, OUTCOME_INVALID
    if obj.get("failure") is True:
        return None, OUTCOME_FAILURE
    if NESTED_PROBE_KEY in obj:  # probe-run record: payload nested + metadata
        obj = obj[NESTED_PROBE_KEY]
        if not isinstance(obj, dict):
            return None, OUTCOME_INVALID
    if any(k not in obj for k in PROBE_KEYS):
        return None, OUTCOME_INVALID
    detail, suff, nxt = (
        obj["detail_fraction"],
        obj["sufficient"],
        obj["next_step_helps"],
    )
    if isinstance(detail, bool) or not isinstance(detail, int | float):
        return None, OUTCOME_INVALID
    if not DETAIL_MIN <= float(detail) <= DETAIL_MAX or math.isnan(float(detail)):
        return None, OUTCOME_INVALID
    if not isinstance(suff, bool) or not isinstance(nxt, bool):
        return None, OUTCOME_INVALID
    return ProbeRecord(float(detail), suff, nxt), OUTCOME_OK


def _read(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "corrupt"  # not a dict -> invalid


def load_probe_states(
    probe_dir: Path, image_ids: Sequence[int], num_levels: int
) -> pd.DataFrame:
    """One row per (image, level) with the parsed probe fields (NaN if missing)."""
    rows = []
    for image_id in image_ids:
        for t in range(1, num_levels + 1):
            path = probe_dir / str(image_id) / f"{level_name(t)}.json"
            rec, outcome = parse_probe(_read(path))
            rows.append({
                "image_id": int(image_id), "t": t, "outcome": outcome,
                "detail_fraction": rec.detail_fraction if rec else math.nan,
                "sufficient": float(rec.sufficient) if rec else math.nan,
                "next_step_helps": float(rec.next_step_helps) if rec else math.nan,
            })  # fmt: skip
    return pd.DataFrame(rows)


def attach_q(states: pd.DataFrame, curves: RecallCurves, ratio: float) -> pd.DataFrame:
    """Add q, qmax, q_ratio and sufficient_gt (q_ratio >= ratio) for one metric."""
    q, qmax, valid = [], [], []
    for r in states[["image_id", "t"]].to_dict("records"):
        qa = curves.q.get(int(r["image_id"]))
        ok = qa is not None and curves.all_valid(int(r["image_id"]))
        q.append(float(qa[int(r["t"]) - 1]) if ok and qa is not None else math.nan)
        qmax.append(float(np.max(qa)) if ok and qa is not None else math.nan)
        valid.append(ok)
    out = states.assign(q=q, qmax=qmax, q_valid=valid)
    q_ratio = (out["q"] / out["qmax"]).where(out["qmax"] > 0)
    return out.assign(
        q_ratio=q_ratio,
        sufficient_gt=(q_ratio >= ratio).astype(float).where(q_ratio.notna()),
    )


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def probe_by_level(states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in sorted(int(x) for x in states["t"].unique()):
        grp = states[states["t"] == t]
        ok = grp[grp["outcome"] == OUTCOME_OK]
        rows.append({
            "t": t, "n_states": len(grp), "n": len(ok),
            "n_failure": int((grp["outcome"] == OUTCOME_FAILURE).sum()),
            "n_missing_or_invalid": int(grp["outcome"].isin([OUTCOME_MISSING, OUTCOME_INVALID]).sum()),
            "detail_fraction_mean": _mean(ok["detail_fraction"]),
            "frac_sufficient": _mean(ok["sufficient"]),
            "frac_next_step_helps": _mean(ok["next_step_helps"]),
            "q_ratio_mean": _mean(ok["q_ratio"]),
            "frac_sufficient_gt": _mean(ok["sufficient_gt"]),
        })  # fmt: skip
    return pd.DataFrame(rows)


def _mean(values: pd.Series) -> float:
    v = values.astype(float).dropna()
    return float(v.mean()) if len(v) else math.nan


def count_inversions(values: Sequence[float]) -> int:
    """Pairs i < j with values[i] > values[j] (NaN entries skipped)."""
    v = [x for x in values if not math.isnan(x)]
    return sum(v[i] > v[j] for i in range(len(v)) for j in range(i + 1, len(v)))


def monotonicity(states: pd.DataFrame) -> dict[str, Any]:
    """Share of images with non-decreasing detail_fraction, mean inversions."""
    ok = states[states["outcome"] == OUTCOME_OK].sort_values(["image_id", "t"])
    inv, mono, complete = [], [], 0
    for _, grp in ok.groupby("image_id"):
        d = grp["detail_fraction"].tolist()
        if len(d) < 2:
            continue
        n_inv = count_inversions(d)
        inv.append(n_inv)
        mono.append(n_inv == 0)
    n_levels = int(states["t"].nunique())
    complete = int(ok.groupby("image_id").size().eq(n_levels).sum()) if len(ok) else 0
    return {
        "n_images": len(inv),
        "n_images_all_levels": complete,
        "frac_monotone": float(np.mean(mono)) if mono else math.nan,
        "mean_inversions": float(np.mean(inv)) if inv else math.nan,
    }


def probe_stats(states: pd.DataFrame) -> dict[str, Any]:
    """Rank statistics on valid, successfully probed states with qmax > 0."""
    use = states[(states["outcome"] == OUTCOME_OK) & states["q_ratio"].notna()]
    gt = use["sufficient_gt"].to_numpy(float) == 1.0
    return {
        "n_states": len(use),
        "n_sufficient_gt": int(gt.sum()),
        "rho_detail_t": spearman_stat(use["detail_fraction"], use["t"]),
        "rho_detail_qratio": spearman_stat(use["detail_fraction"], use["q_ratio"]),
        "auc_detail": auc(use["detail_fraction"].to_numpy(float), gt),
        "auc_not_next": auc(1.0 - use["next_step_helps"].to_numpy(float), gt),
        "auc_sufficient_flag": auc(use["sufficient"].to_numpy(float), gt),
        "probe_sufficient_rate": _mean(use["sufficient"]),
    }


# --------------------------------------------------------------------------
# Probe-only policies through the existing replay / evaluation
# --------------------------------------------------------------------------


def probe_c_sequence(states: pd.DataFrame, image_id: int, policy: str,
                     num_levels: int) -> list[int | None]:  # fmt: skip
    """Per-level C for replay_stop: 0 = stop signal, 1 = continue, None = missing."""
    seq: list[int | None] = [None] * num_levels
    grp = states[(states["image_id"] == image_id) & (states["outcome"] == OUTCOME_OK)]
    for r in grp.to_dict("records"):
        stop = (
            bool(r["sufficient"])
            if policy == PROBE_STOP
            else not bool(r["next_step_helps"])
        )
        seq[int(r["t"]) - 1] = STOP_SIGNAL_C if stop else CONTINUE_C
    return seq


def probe_stops(states: pd.DataFrame, image_ids: Sequence[int], num_levels: int,
                consecutive_k: int) -> pd.DataFrame:  # fmt: skip
    rows = []
    for policy in PROBE_POLICIES:
        for image_id in image_ids:
            seq = probe_c_sequence(states, image_id, policy, num_levels)
            res = replay_stop(seq, PROBE_THETA, consecutive_k)
            rows.append({"image_id": int(image_id), "method": policy, "stop_t": res.stop_t,
                         "natural_stop": res.natural_stop, "forced_terminal": res.forced_terminal,
                         "failure_count": res.failure_count})  # fmt: skip
    return pd.DataFrame(rows)


def gated_c_sequence(
    states: pd.DataFrame,
    image_id: int,
    judge_c: Sequence[int | None],
    max_u: int,
) -> list[int | None]:
    """C_t of the probe-gated policy.

    probe ok & not sufficient -> max_u (keep observing); probe ok & sufficient ->
    the judge's C_t (None if that state failed); probe missing/failed -> None
    (communication failure, plan §6).
    """
    seq: list[int | None] = [None] * len(judge_c)
    grp = states[(states["image_id"] == image_id) & (states["outcome"] == OUTCOME_OK)]
    for r in grp.to_dict("records"):
        t = int(r["t"])
        seq[t - 1] = judge_c[t - 1] if bool(r["sufficient"]) else max_u
    return seq


def judge_c_table(
    input_dir: Path,
    image_ids: Sequence[int],
    num_levels: int,
    cfg: Mapping[str, Any],
    layout: RunLayout,
) -> dict[tuple[int, str], list[int | None]]:
    """C_t sequences of the gated conditions for every image of a decisions path."""
    conditions = {
        n: c for n, c in build_conditions(cfg).items() if n in GATED_CONDITIONS
    }
    limits = ContractLimits.from_config(cfg)
    out: dict[tuple[int, str], list[int | None]] = {}
    for image_id in image_ids:
        rows = score_image(input_dir, image_id, num_levels, conditions, limits, layout)
        for name in conditions:
            seq: list[int | None] = [None] * num_levels
            for r in rows:
                if r["condition"] == name:
                    seq[int(r["t"]) - 1] = r["C"]
            out[(int(image_id), name)] = seq
    return out


def gate_stops(
    states: pd.DataFrame,
    judge_c: Mapping[tuple[int, str], Sequence[int | None]],
    image_ids: Sequence[int],
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """stops for the gated policies and for the ungated reference conditions."""
    conditions = build_conditions(cfg)
    rows = []
    for name in GATED_CONDITIONS:
        cond = conditions[name]
        for image_id in image_ids:
            base = list(judge_c[(int(image_id), name)])
            for method, seq in (
                (
                    f"{GATE_PREFIX}{name}",
                    gated_c_sequence(states, image_id, base, cond.max_u),
                ),
                (name, base),
            ):
                res = replay_stop(seq, cond.theta, cond.consecutive_k)
                rows.append({"image_id": int(image_id), "method": method,
                             "stop_t": res.stop_t, "natural_stop": res.natural_stop,
                             "forced_terminal": res.forced_terminal,
                             "failure_count": res.failure_count})  # fmt: skip
    return pd.DataFrame(rows)


def _status(method: str) -> str:
    if method in PROBE_POLICIES:
        return PROBE_STATUS
    if method in GATE_POLICIES:
        return GATE_STATUS
    if method in GATED_CONDITIONS:
        return REFERENCE_STATUS
    return FIXED_STATUS


def evaluate_probe_policies(
    states: pd.DataFrame,
    curves: RecallCurves,
    params: EvalParams,
    consecutive_k: int,
    gate: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """(stops, summary, bootstrap) for the probe (and gated) policies vs fixed stops."""
    ids = sorted(curves.q)
    parts = [probe_stops(states, ids, params.num_levels, consecutive_k)]
    if gate is not None:
        parts.append(gate)
    stops = pd.concat([*parts, fixed_stops(ids, params.num_levels)], ignore_index=True)
    adaptive = list(PROBE_POLICIES)
    final = fixed_method_name(params.num_levels)
    comparisons: list[tuple[str, str]] = [
        (PROBE_STOP, final),
        (PROBE_STOP_NEXT, final),
        (PROBE_STOP, PROBE_STOP_NEXT),
    ]
    if gate is not None:
        adaptive += [*GATE_POLICIES, *GATED_CONDITIONS]
        for name in GATED_CONDITIONS:
            gated = f"{GATE_PREFIX}{name}"
            comparisons += [(gated, final), (gated, name)]
        comparisons.append((GATE_POLICIES[0], GATE_POLICIES[1]))
    eval_set = build_eval_set(curves, params.oracle_ratio, params.oracle_gain_delta)
    per_image = per_image_metrics(stops, curves, eval_set)
    summary = summarize(per_image, stops, curves, eval_set, None, adaptive)
    summary.insert(1, "status", [_status(m) for m in summary["method"]])
    boot = paired_bootstrap(
        per_image, curves, eval_set, adaptive, comparisons,
        params.bootstrap_repeats, params.bootstrap_seed, params.ci_level,
    )  # fmt: skip
    boot["policy_status"] = {m: _status(m) for m in adaptive}
    return stops, summary, boot


# --------------------------------------------------------------------------
# One configuration end to end
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeResult:
    cfg: str
    q_metric: str
    levels: pd.DataFrame
    stats: dict[str, Any]
    summary: pd.DataFrame
    bootstrap: dict[str, Any]


def run_probe_diag(
    input_dir: Path,
    probe_dir: Path,
    out_dir: Path,
    cfg: Mapping[str, Any],
    probe_cfg: str,
    q_metric: str,
    gate_layout: RunLayout | None = None,
) -> ProbeResult:
    """Diagnostics + probe-only replay for one probe config and one q metric.

    With ``gate_layout`` (decisions path to gate), also replays
    probe_gate_proposed / probe_gate_observed_only and the ungated references.
    """
    params = EvalParams.from_config(cfg, q_metric)
    k = int(cfg["proposal"]["consecutive_k"])
    stage = load_stage_metrics(input_dir / "stage_metrics.csv", q_metric)
    curves = build_curves(stage, q_metric, params.num_levels)
    raw = load_probe_states(probe_dir, sorted(curves.q), params.num_levels)
    states = attach_q(raw, curves, params.oracle_ratio)
    levels = probe_by_level(states)
    stats = {
        "cfg": probe_cfg,
        "q_metric": q_metric,
        "ratio": params.oracle_ratio,
        "outcomes": states["outcome"].value_counts().to_dict(),
        **probe_stats(states),
        "monotonicity": monotonicity(states),
    }
    gate = None
    if gate_layout is not None:
        ids = sorted(curves.q)
        judge_c = judge_c_table(input_dir, ids, params.num_levels, cfg, gate_layout)
        gate = gate_stops(states, judge_c, ids, cfg)
        stats["gate_decisions"] = gate_layout.decisions
        stats["gate_graphs"] = gate_layout.graphs
    stops, summary, boot = evaluate_probe_policies(states, curves, params, k, gate)
    target = q_output_dir(out_dir, q_metric)
    target.mkdir(parents=True, exist_ok=True)
    states.to_csv(target / "probe_states.csv", index=False)
    levels.to_csv(target / "probe_by_level.csv", index=False)
    stops.to_csv(target / "stops.csv", index=False)
    summary.to_csv(target / "summary.csv", index=False)
    (target / "bootstrap.json").write_text(json.dumps(boot, indent=2))
    (target / "probe_stats.json").write_text(json.dumps(stats, indent=2))
    return ProbeResult(probe_cfg, q_metric, levels, stats, summary, boot)


def compare_row(res: ProbeResult) -> dict[str, Any]:
    """One row of probe_compare.csv (cfg x q metric)."""
    st = res.stats
    outcomes = st["outcomes"]
    n_total = sum(outcomes.values())
    row: dict[str, Any] = {
        "cfg": res.cfg, "q_metric": res.q_metric, "n_states": st["n_states"],
        "failure_rate": 1 - outcomes.get(OUTCOME_OK, 0) / n_total if n_total else math.nan,
        "n_sufficient_gt": st["n_sufficient_gt"],
        "rho_detail_t": st["rho_detail_t"]["rho"],
        "rho_detail_qratio": st["rho_detail_qratio"]["rho"],
        "auc_detail": st["auc_detail"], "auc_not_next": st["auc_not_next"],
        "auc_sufficient_flag": st["auc_sufficient_flag"],
        "frac_monotone": st["monotonicity"]["frac_monotone"],
        "mean_inversions": st["monotonicity"]["mean_inversions"],
    }  # fmt: skip
    summ = res.summary.set_index("method")
    row["gate_decisions"] = res.stats.get("gate_decisions")
    for policy in (*PROBE_POLICIES, *GATE_POLICIES, *GATED_CONDITIONS):
        if policy not in summ.index:
            continue
        stat = (
            res.bootstrap.get("methods", {}).get(policy, {}).get("q_minus_eq_fixed", {})
        )
        row |= {
            f"{policy}_mean_stop": summ.loc[policy, "mean_stop_t"],
            f"{policy}_q_stop": summ.loc[policy, "q_stop"],
            f"{policy}_eq_fixed_q": summ.loc[policy, "eq_fixed_q"],
            f"{policy}_diff": stat.get("mean", math.nan),
            f"{policy}_diff_ci_low": stat.get("ci_low", math.nan),
            f"{policy}_diff_ci_high": stat.get("ci_high", math.nan),
            f"{policy}_forced_rate": summ.loc[policy, "forced_terminal_rate"],
        }
    final = fixed_method_name(int(res.levels["t"].max()))
    for a, b in [(g, final) for g in GATE_POLICIES] + [
        (f"{GATE_PREFIX}{c}", c) for c in GATED_CONDITIONS
    ]:
        stat = res.bootstrap.get("pairwise", {}).get(f"{a}_minus_{b}", {}).get("q_stop")
        if stat:
            row[f"{a}_minus_{b}"] = stat["mean"]
            row[f"{a}_minus_{b}_ci_low"] = stat["ci_low"]
            row[f"{a}_minus_{b}_ci_high"] = stat["ci_high"]
    return row
