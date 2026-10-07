"""Phase D evaluation of the frozen two-stage U policy (recovery plan §6.3, §7).

Zero model calls. Inputs:

- policy decisions (A3, ``test_policy.csv`` / ``dev_two_stage.csv`` format):
  ``image_id, U, valid, decision_t in {10, 20}, extra_cost_failure``;
- seedA stage metrics (all 20 levels) and, separately, seedB stage metrics
  (levels 10/15/20 only). SeedA and seedB are never combined.

Per metric (R50 primary, relsg_gt secondary):

- policy: mean q(decision_t) and the realized continuation rate p;
- mix: image-independent t10/t20 mixture at the SAME realized p,
  ``mean(q10 + p (q20 - q10))``;
- PRIMARY comparison: policy - mix (R50), paired bootstrap over images that
  recomputes p, the policy mean and the mix in every resample (tau / p_tie are
  frozen upstream; decisions are never re-derived here);
- fixed15 (flagged when the mean level is not 15);
- seedA only: full fixed curve and F(mean level), GT-posterior references.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from milestone1.recovery_diagnostics import best_fixed_mixture, quality_matrix

OBSERVE, CONTINUE, FIXED_MID = 10, 20, 15
DECISION_LEVELS = (OBSERVE, CONTINUE)
SEED_B_LEVELS = (OBSERVE, FIXED_MID, CONTINUE)
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20261008
CI_LEVEL = 0.95
PRIMARY_METRIC = "R50"
TRUE_STRINGS = frozenset({"true", "1", "yes", "t"})
VERDICT_POSITIVE = "positive (95% CI excludes 0)"
VERDICT_UNCONFIRMED = "unconfirmed (positive, CI includes 0)"
VERDICT_NOT_POSITIVE = "not positive"
VERDICT_UNSTABLE = "unstable (seedA positive, seedB negative)"
SCHEDULE_AWARE_NOTE = ("schedule-aware reference (knows the level index; GT-posterior for F) "
                       "— not a comparator")  # fmt: skip
F_NOTE = "F on seedA: schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    return str(value).strip().lower() in TRUE_STRINGS


def load_policy(df: pd.DataFrame) -> pd.DataFrame:
    """image_id, decision_t (10/20), failure flag, U; duplicates / bad levels raise."""
    if df.duplicated("image_id").any():
        raise ValueError("duplicate image_id in policy decisions")
    bad = set(df["decision_t"].astype(int)) - set(DECISION_LEVELS)
    if bad:
        raise ValueError(f"decision_t outside {DECISION_LEVELS}: {sorted(bad)}")
    failure_col = df["extra_cost_failure"] if "extra_cost_failure" in df else False
    return pd.DataFrame({
        "image_id": df["image_id"].astype(int),
        "decision_t": df["decision_t"].astype(int),
        "failure": pd.Series(failure_col, index=df.index).map(_truthy),
        "U": pd.to_numeric(df.get("U"), errors="coerce") if "U" in df else math.nan,
    })  # fmt: skip


@dataclass(frozen=True)
class Arrays:
    q_lo: np.ndarray
    q_mid: np.ndarray
    q_hi: np.ndarray
    cont: np.ndarray  # bool, decision_t == 20
    failure: np.ndarray  # bool
    ids: tuple[int, ...]


def level_arrays(stage: pd.DataFrame, metric: str, policy: pd.DataFrame,
                 levels: Sequence[int], label: str) -> tuple[Arrays, dict[str, Any]]:  # fmt: skip
    """q at t10/15/20 for images with a decision and complete values at ``levels``."""
    if stage.duplicated(["image_id", "t"]).any():
        raise ValueError(f"{label}: duplicate (image_id, t) rows (mixed seeds?)")
    sub = stage[stage["t"].isin(levels)]
    q = sub.assign(v=pd.to_numeric(sub[metric], errors="coerce"))
    q = q[q["valid"].map(_truthy) & q["v"].notna()]
    table = q.pivot(index="image_id", columns="t", values="v")
    complete = (
        table.dropna(subset=list(levels))
        if set(levels) <= set(table.columns)
        else table.iloc[0:0]
    )
    pol = policy.set_index("image_id")
    ids = [i for i in pol.index if i in complete.index]
    info = {"n_decisions": len(pol), "n_evaluated": len(ids),
            "n_excluded_missing_q": len(pol) - len(ids)}  # fmt: skip
    mid = (
        complete.loc[ids, FIXED_MID]
        if FIXED_MID in complete.columns
        else pd.Series(math.nan, index=ids)
    )
    arrays = Arrays(
        complete.loc[ids, OBSERVE].to_numpy(float), mid.to_numpy(float),
        complete.loc[ids, CONTINUE].to_numpy(float),
        (pol.loc[ids, "decision_t"] == CONTINUE).to_numpy(bool),
        pol.loc[ids, "failure"].to_numpy(bool), tuple(int(i) for i in ids),
    )  # fmt: skip
    return arrays, info


def policy_minus_mix(a: Arrays, rows: np.ndarray) -> tuple[float, float, float, float]:
    """(policy mean, mix mean at the realized p of these rows, difference, p)."""
    lo, hi, cont = a.q_lo[rows], a.q_hi[rows], a.cont[rows]
    p = float(cont.mean())
    policy = float(np.where(cont, hi, lo).mean())
    mix = float(np.mean(lo + p * (hi - lo)))
    return policy, mix, policy - mix, p


def bootstrap_policy_minus_mix(a: Arrays, repeats: int = BOOTSTRAP_REPEATS,
                               seed: int = BOOTSTRAP_SEED, ci_level: float = CI_LEVEL,
                               ) -> dict[str, float]:  # fmt: skip
    n = len(a.q_lo)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(repeats, n))
    diffs = np.array([policy_minus_mix(a, rows)[2] for rows in idx])
    alpha = (1 - ci_level) / 2
    lo, hi = np.quantile(diffs, [alpha, 1 - alpha])
    point = policy_minus_mix(a, np.arange(n))[2]
    return {"diff": point, "ci_low": float(lo), "ci_high": float(hi),
            "repeats": repeats, "seed": seed}  # fmt: skip


def verdict(diff: float, ci_low: float) -> str:
    if diff > 0 and ci_low > 0:
        return VERDICT_POSITIVE
    if diff > 0:
        return VERDICT_UNCONFIRMED
    return VERDICT_NOT_POSITIVE


def summarize_seed(a: Arrays, info: dict[str, Any], metric: str, seed_label: str,
                   repeats: int) -> dict[str, Any]:  # fmt: skip
    rows = np.arange(len(a.q_lo))
    policy, mix, diff, p = policy_minus_mix(a, rows)
    boot = bootstrap_policy_minus_mix(a, repeats)
    mean_level = OBSERVE + (CONTINUE - OBSERVE) * p
    n_fail = int(a.failure.sum())
    fixed15 = float(np.nanmean(a.q_mid)) if np.isfinite(a.q_mid).any() else math.nan
    return {
        "seed": seed_label, "metric": metric, "primary": metric == PRIMARY_METRIC,
        **info, "n_t20": int(a.cont.sum()), "p_continue": p,
        "mean_level": mean_level, "policy_q": policy, "mix_q_same_p": mix,
        "policy_minus_mix": diff, "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
        "bootstrap_repeats": boot["repeats"], "verdict": verdict(diff, boot["ci_low"]),
        "fixed15_q": fixed15, "policy_minus_fixed15": policy - fixed15,
        "fixed15_role": SCHEDULE_AWARE_NOTE,
        "fixed15_same_budget": math.isclose(mean_level, FIXED_MID),
        "n_failure_continued": n_fail,
        "failure_extra_levels_per_image": (CONTINUE - OBSERVE) * n_fail / len(rows) if len(rows) else math.nan,
        "pgsg_states_observed": int(len(rows) + a.cont.sum()),
        "v4_calls": int(len(rows)),
    }  # fmt: skip


def seed_a_references(
    stage: pd.DataFrame, metric: str, mean_level: float
) -> dict[str, Any]:
    """Full fixed curve and F(mean level) on seedA (GT-posterior)."""
    qm = quality_matrix(stage, metric, "seedA")
    f = qm.q.mean(axis=0)
    mix = best_fixed_mixture(f, mean_level)
    return {"F_at_mean_level": mix.value, "F_support": ";".join(f"t{t}:{p:.3f}" for t, p in mix.support),
            "fixed_curve": [float(v) for v in f], "n_complete_series": qm.n, "F_note": F_NOTE}  # fmt: skip


def combined_verdict(seed_a: str, seed_b_diff: float | None) -> str:
    """§7: never average seeds; seedA positive and seedB negative -> unstable."""
    if (
        seed_b_diff is not None
        and seed_a in (VERDICT_POSITIVE, VERDICT_UNCONFIRMED)
        and seed_b_diff < 0
    ):
        return VERDICT_UNSTABLE
    return seed_a


# --------------------------------------------------------------------------
# MIRU-style method comparison (presentation; the primary criterion stays
# R50 policy - same-p mix)
# --------------------------------------------------------------------------

# Online comparators only. fixed10/15/20 and F(b) know the level index (schedule-aware)
# and are reported as references, never tested against.
COMPARATORS = ("mix", "conf_two_stage")
PERM_REPEATS = 2000
FIXED_STOPS = {"fixed10": OBSERVE, "fixed15": FIXED_MID, "fixed20": CONTINUE}
SIG_ALPHA = 0.05
# column -> True if higher is better
COLUMN_HIGHER_BETTER = {"mae": False, "agree": True, "q": True}


@dataclass(frozen=True)
class SeedLevels:
    """q at t10/15/20 per metric for the evaluated images of one seed."""

    q: dict[str, np.ndarray]  # metric -> (n, 3) array for levels (10, 15, 20)
    cont: np.ndarray  # policy decision: True = t20
    ids: tuple[int, ...]
    conf_cont: np.ndarray | None = None  # conf_two_stage decision: True = t20


def oracle_cont(q: np.ndarray, k: int) -> np.ndarray:
    """Same-slot two-choice oracle: top-k images by g = q20 - q10 go to t20."""
    g = q[:, 2] - q[:, 0]
    cont = np.zeros(len(g), dtype=bool)
    cont[np.argsort(-g, kind="stable")[:k]] = True
    return cont


def stop_values(
    q: np.ndarray, stops: np.ndarray, oracle: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-image q(stop), |stop - oracle_t| and agreement for stops in {10,15,20}."""
    col = {OBSERVE: 0, FIXED_MID: 1, CONTINUE: 2}
    oracle_t = np.where(oracle, CONTINUE, OBSERVE)
    return {
        "q": q[np.arange(len(stops)), [col[int(s)] for s in stops]],
        "mae": np.abs(stops - oracle_t).astype(float),
        "agree": (stops == oracle_t).astype(float),
    }


def mix_values(q: np.ndarray, p: float, oracle: np.ndarray) -> dict[str, np.ndarray]:
    """Expected per-image values of the image-independent mix at rate p."""
    oracle_t = np.where(oracle, CONTINUE, OBSERVE)
    return {
        "q": q[:, 0] + p * (q[:, 2] - q[:, 0]),
        "mae": p * np.abs(CONTINUE - oracle_t) + (1 - p) * np.abs(OBSERVE - oracle_t),
        "agree": np.where(oracle, p, 1 - p),
    }


def method_rows(
    levels: SeedLevels, oracle_metric: str
) -> dict[str, dict[str, np.ndarray]]:
    """Per-image values of every row, with MAE / agreement vs the oracle of ``oracle_metric``."""
    q = levels.q[oracle_metric]
    k = int(levels.cont.sum())
    orc = oracle_cont(q, k)
    n = len(levels.cont)
    policy_stops = np.where(levels.cont, CONTINUE, OBSERVE)
    rows = {"U policy": stop_values(q, policy_stops, orc),
            "mix (same p)": mix_values(q, float(levels.cont.mean()), orc)}  # fmt: skip
    if levels.conf_cont is not None:
        rows["conf_two_stage"] = stop_values(
            q, np.where(levels.conf_cont, CONTINUE, OBSERVE), orc
        )
    for name, t in FIXED_STOPS.items():
        rows[name] = stop_values(q, np.full(n, t), orc)
    rows["GT two-choice oracle (reference)"] = stop_values(
        q, np.where(orc, CONTINUE, OBSERVE), orc
    )
    return rows


def _comparator_values(levels: SeedLevels, metric: str, oracle: np.ndarray, rows: np.ndarray,
                       comparator: str) -> dict[str, float]:  # fmt: skip
    q = levels.q[metric][rows]
    orc = oracle[rows]
    return {
        c: float(v.mean())
        for c, v in _comparator_arrays(levels, q, orc, rows, comparator).items()
    }


def _comparator_arrays(levels: SeedLevels, q: np.ndarray, orc: np.ndarray, rows: np.ndarray,
                       comparator: str) -> dict[str, np.ndarray]:  # fmt: skip
    if comparator == "mix":
        return mix_values(q, float(levels.cont[rows].mean()), orc)
    if comparator == "conf_two_stage":
        if levels.conf_cont is None:
            raise ValueError("conf_two_stage decisions missing")
        return stop_values(q, np.where(levels.conf_cont[rows], CONTINUE, OBSERVE), orc)
    return stop_values(q, np.full(len(rows), FIXED_STOPS[comparator]), orc)


def _policy_values(levels: SeedLevels, metric: str, oracle: np.ndarray, rows: np.ndarray,
                   cont: np.ndarray | None = None) -> dict[str, float]:  # fmt: skip
    cont = levels.cont[rows] if cont is None else cont
    vals = stop_values(
        levels.q[metric][rows], np.where(cont, CONTINUE, OBSERVE), oracle[rows]
    )
    return {c: float(v.mean()) for c, v in vals.items()}


def significance_rows(levels: SeedLevels, metric: str, oracle_metric: str,
                      repeats: int = BOOTSTRAP_REPEATS, perm_repeats: int = PERM_REPEATS,
                      seed: int = BOOTSTRAP_SEED) -> list[dict[str, Any]]:  # fmt: skip
    """U policy vs the online comparators (same-p mix, conf_two_stage).

    Bootstrap: images resampled; the mix's p is recomputed per resample; oracle labels
    are fixed per image. Permutation: vs the mix, the policy's t20 slots are reassigned
    at random (same count; the null is random assignment = the mix); vs conf_two_stage,
    a paired sign-flip test on the per-image differences.
    """
    n = len(levels.cont)
    oracle = oracle_cont(levels.q[oracle_metric], int(levels.cont.sum()))
    full = np.arange(n)
    q_full = levels.q[metric]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(repeats, n))
    perm_rng = np.random.default_rng(seed + 1)
    perms = [perm_rng.permutation(levels.cont) for _ in range(perm_repeats)]
    signs = np.random.default_rng(seed + 2).choice([-1.0, 1.0], size=(perm_repeats, n))
    observed = _policy_values(levels, metric, oracle, full)
    policy_arr = stop_values(q_full, np.where(levels.cont, CONTINUE, OBSERVE), oracle)
    slot_perm = [_policy_values(levels, metric, oracle, full, c) for c in perms]
    comparators = [
        c for c in COMPARATORS if c != "conf_two_stage" or levels.conf_cont is not None
    ]
    out = []
    for comparator in comparators:
        comp = _comparator_values(levels, metric, oracle, full, comparator)
        comp_arr = _comparator_arrays(levels, q_full, oracle, full, comparator)
        boot = [
            (_policy_values(levels, metric, oracle, r), _comparator_values(levels, metric, oracle, r, comparator))
            for r in idx
        ]  # fmt: skip
        for column, higher_better in COLUMN_HIGHER_BETTER.items():
            if column != "q" and metric != oracle_metric:
                continue  # MAE / agreement are reported once, against the oracle metric
            diff = observed[column] - comp[column]
            diffs = np.array([pv[column] - cv[column] for pv, cv in boot])
            lo, hi = np.quantile(diffs, [(1 - CI_LEVEL) / 2, 1 - (1 - CI_LEVEL) / 2])
            if comparator == "mix":
                null = np.array([pv[column] for pv in slot_perm]) - comp[column]
            else:
                d = policy_arr[column] - comp_arr[column]
                null = (signs * d).mean(axis=1)
            up = float(np.mean(null >= diff - 1e-12))
            down = float(np.mean(null <= diff + 1e-12))
            better_p, worse_p = (up, down) if higher_better else (down, up)
            better = diff > 0 if higher_better else diff < 0
            ci_excludes_0 = bool(lo > 0 or hi < 0)
            sig = ci_excludes_0 or min(better_p, worse_p) < SIG_ALPHA
            out.append({
                "metric": metric, "oracle_metric": oracle_metric, "comparator": comparator,
                "column": column, "policy": observed[column], "comparator_value": comp[column],
                "diff_policy_minus_comparator": diff, "ci_low": float(lo), "ci_high": float(hi),
                "perm_p_better": better_p, "perm_p_worse": worse_p,
                "perm_test": "slot permutation" if comparator == "mix" else "paired sign flip",
                "direction": "better" if better else ("worse" if diff != 0 else "equal"),
                "significant": sig, "applicable": True,
            })  # fmt: skip
    return out


# --------------------------------------------------------------------------
# conf_two_stage: level-agnostic, observation-based comparator
# --------------------------------------------------------------------------

CONF_TOP_K = 8  # same top-K cleaned-G entities as the v4 U policy
CONF_POLICY_NAME = "conf_two_stage"
CONF_TIE_TOL = 1e-12


def top_k_confidence(graph: dict[str, Any] | None, k: int = CONF_TOP_K) -> float:
    """Mean PGSG entity ``score`` of the top-k cleaned-G entities (v4 selection); NaN if none."""
    from milestone1.grounded_judge import top_entities  # local: heavy optional deps

    if not graph:
        return math.nan
    ents = top_entities(graph, k)
    return (
        float(np.mean([float(e.get("score", 0.0)) for e in ents])) if ents else math.nan
    )


def fit_conf_threshold(conf: Sequence[float]) -> dict[str, Any]:
    """tau = dev median; continue iff conf < tau; p_tie keeps the expected rate at half."""
    vals = [c for c in conf if not math.isnan(c)]
    if not vals:
        raise ValueError("no valid confidences")
    tau = float(np.median(vals))
    n = len(vals)
    n_below = sum(c < tau - CONF_TIE_TOL for c in vals)
    n_tie = sum(abs(c - tau) <= CONF_TIE_TOL for c in vals)
    p_tie = min(1.0, max(0.0, (0.5 * n - n_below) / n_tie)) if n_tie else 0.0
    return {
        "tau": tau,
        "p_tie": p_tie,
        "n_valid": n,
        "n_below": n_below,
        "n_tie": n_tie,
    }


def conf_decision(
    image_id: int, conf: float, tau: float, p_tie: float
) -> tuple[int, str]:
    """(stop level, reason): low confidence -> keep observing; ties by the GT-free hash."""
    from milestone1.two_stage_policy import hash_unit

    if math.isnan(conf):
        return CONTINUE, "failure_continue"
    if abs(conf - tau) <= CONF_TIE_TOL:
        return (
            (CONTINUE, "tie_continue")
            if hash_unit(image_id) < p_tie
            else (OBSERVE, "tie_stop")
        )
    return (CONTINUE, "below_tau") if conf < tau else (OBSERVE, "above_tau")
