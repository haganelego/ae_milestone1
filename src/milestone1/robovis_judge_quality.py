"""Judge-quality diagnostic: does the judge's "resolved" track recovered recall?

Intended judge semantics (user guidance): an item is "resolved" once roughly
``ratio`` (0.9) of what should be visible is visible. Per state we derive
``frac_resolved`` (resolved / judged items), ``frac_lowres`` (share of judged
items labelled low_resolution; a DIAGNOSTIC signal, not a stop condition) and
``any_gate_theta`` (C_proposed >= theta) and compare them with the recovery
ratio ``q_ratio = q_t / qmax`` (same q metric as the signal diagnostic).
Zero model calls; inputs are the signal_per_state rows of robovis_diag.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from milestone1.robovis_stats import auc, spearman_stat
from milestone1.stop_policy import OUTCOME_EMPTY_GRAPH, OUTCOME_OK

OK_OUTCOMES = (OUTCOME_OK, OUTCOME_EMPTY_GRAPH)

# q_ratio bins [0, .5), [.5, ratio), [ratio, 1]; the upper edge is inclusive.
LOW_BIN_EDGE = 0.5
TABLE_LEVEL = "level"
TABLE_CALIBRATION = "calibration"
LOWRES_ROLE = "diagnostic signal, not a stop condition"


def add_judge_columns(states: pd.DataFrame, theta: int) -> pd.DataFrame:
    """frac_resolved, frac_lowres (NaN without judged items), any_gate_theta."""
    n_items = states["n_items"].astype(float)
    has_items = n_items > 0
    c = states["C_proposed"].astype(float)
    return states.assign(
        frac_resolved=(states["n_resolved"] / n_items).where(has_items),
        frac_lowres=(states["n_lowres"] / n_items).where(has_items),
        any_gate_theta=(c >= theta).astype(float).where(c.notna()),
        q_ratio=states["frac_of_qmax"],
    )


def bin_label(lo: float, hi: float, closed_right: bool) -> str:
    return f"[{lo:g},{hi:g}{']' if closed_right else ')'}"


def ratio_bins(ratio: float) -> list[tuple[float, float, bool]]:
    return [
        (0.0, LOW_BIN_EDGE, False),
        (LOW_BIN_EDGE, ratio, False),
        (ratio, 1.0, True),
    ]


def _usable(states: pd.DataFrame) -> pd.DataFrame:
    """Valid, successfully judged states of images with qmax > 0."""
    ok = states["outcome"].isin(OK_OUTCOMES) & states["valid"].astype(bool)
    return states[ok & (states["qmax"] > 0)]


def judge_quality(
    states: pd.DataFrame, theta: int, ratio: float, condition_columns: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-level means + calibration bins (table) and rank statistics (dict)."""
    st = add_judge_columns(states, theta)
    use = _usable(st)
    rows: list[dict[str, Any]] = []
    for t in sorted(int(x) for x in use["t"].unique()):
        grp = use[use["t"] == t]
        rows.append({"table": TABLE_LEVEL, "t": t, "n": len(grp), **_means(grp)})
    for lo, hi, closed in ratio_bins(ratio):
        upper = use["q_ratio"] <= hi if closed else use["q_ratio"] < hi
        grp = use[(use["q_ratio"] >= lo) & upper]
        rows.append({"table": TABLE_CALIBRATION, "q_ratio_bin": bin_label(lo, hi, closed),
                     "n": len(grp), **_means(grp)})  # fmt: skip

    sufficient = (use["q_ratio"] >= ratio).to_numpy()
    resolved = use["frac_resolved"].to_numpy(float)
    has_res = ~np.isnan(resolved)
    lowres = use["frac_lowres"].to_numpy(float)
    has_low = ~np.isnan(lowres)
    stats: dict[str, Any] = {
        "n_states": len(use),
        "n_sufficient": int(sufficient.sum()),
        "ratio": ratio,
        "rho_resolved_t": spearman_stat(use["frac_resolved"], use["t"]),
        "rho_resolved_qratio": spearman_stat(use["frac_resolved"], use["q_ratio"]),
        "auc_sufficient_resolved": auc(resolved[has_res], sufficient[has_res]),
        # Diagnostic signal only (never a stop condition): share of judged items
        # labelled low_resolution; 1 - frac_lowres is the "looks sufficient" score.
        "lowres_role": LOWRES_ROLE,
        "rho_lowres_qratio": spearman_stat(use["frac_lowres"], use["q_ratio"]),
        "auc_sufficient_lowres": auc(1.0 - lowres[has_low], sufficient[has_low]),
        "auc_sufficient_C": {
            col: auc(1.0 - use[col].to_numpy(float), sufficient)
            for col in condition_columns
        },
    }
    return pd.DataFrame(rows), stats


def _means(grp: pd.DataFrame) -> dict[str, float]:
    cols = ("frac_resolved", "frac_lowres", "any_gate_theta", "C_proposed", "q_ratio")
    return {f"{c}_mean": float(grp[c].mean()) if len(grp) else math.nan for c in cols}
