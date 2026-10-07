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
F_NOTE = "GT-posterior reference on seedA, not a method"


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
