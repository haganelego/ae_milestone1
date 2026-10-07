"""Recovery plan Phase B / C diagnostics (zero model calls).

All quantities are GT-posterior REFERENCES, not stop methods:

- ``best_fixed_mixture`` F(b): best image-independent probability mixture over
  the 20 fixed levels with expected level <= b (upper concave hull of
  (t, f_t), f_t = mean_i q_i(t)). ``legacy_adjacent_mix`` keeps the old
  floor/ceil interpolation as a separate column.
- ``budget_oracle`` O(b): omniscient per-image level choice, one level per
  image, sum t_i <= floor(N*b) (multiple-choice knapsack DP). Integer budget,
  so tiny O < F gaps are integrality, not "negative adaptivity".
- ``two_choice_oracle``: t10/t20 only; the k images with the largest
  g_i = q_i(20) - q_i(10) go to t20 (exactly k, negatives allowed) vs
  B_half = mean(q10 + (k/N) g) (image-independent mixture at the same rate).
- ``shuffle_diagnostic``: permute a method's stop list across images; the
  observed mean q vs the permutation distribution isolates the image
  assignment at an identical stop distribution.
- Cross-seed (plan §5.1): seed differences, Spearman of g across seeds, and
  top-k assignments chosen on one seed and scored on another.

Quality series with any missing / invalid level are excluded (never set to 0)
and the coverage is reported. R50 and relsg_gt are never mixed; datasets are
never pooled (a duplicate (image, level) row raises).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

NUM_LEVELS = 20
BUDGETS = (10, 12, 15, 18)
LOW_LEVEL, HIGH_LEVEL = 10, 20
METRICS = ("R50", "relsg_gt")
SHUFFLE_REPEATS = 1000
SHUFFLE_SEED = 20261008
REFERENCE_NOTE = "schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator"
TRUE_STRINGS = frozenset({"true", "1", "yes", "t"})


# --------------------------------------------------------------------------
# Quality matrix
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityMatrix:
    dataset: str
    metric: str
    ids: tuple[int, ...]  # images with a complete valid series
    q: np.ndarray  # (n, num_levels)
    n_total: int
    excluded: dict[int, str] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def coverage(self) -> float:
        return self.n / self.n_total if self.n_total else math.nan


def _valid(value: Any) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | float | np.integer | np.floating):
        return not math.isnan(float(value)) and bool(value)
    return str(value).strip().lower() in TRUE_STRINGS


def quality_matrix(
    stage: pd.DataFrame, metric: str, dataset: str, num_levels: int = NUM_LEVELS
) -> QualityMatrix:
    """Complete-series matrix of one metric of one dataset (no zero filling)."""
    if metric not in stage.columns:
        raise ValueError(f"{dataset}: metric {metric!r} not in stage metrics")
    dup = stage.duplicated(["image_id", "t"])
    if dup.any():
        raise ValueError(f"{dataset}: duplicate (image_id, t) rows -> mixed inputs?")
    values = pd.to_numeric(stage[metric], errors="coerce")
    ok = stage["valid"].map(_valid) & values.notna()
    table = {(int(i), int(t)): float(v) for i, t, v, g in
             zip(stage["image_id"], stage["t"], values, ok, strict=True) if g}  # fmt: skip
    all_ids = sorted({int(i) for i in stage["image_id"]})
    ids, rows, excluded = [], [], {}
    for image_id in all_ids:
        series = [table.get((image_id, t)) for t in range(1, num_levels + 1)]
        if any(v is None for v in series):
            excluded[image_id] = "incomplete or invalid series"
            continue
        ids.append(image_id)
        rows.append(series)
    q = np.array(rows, dtype=float).reshape(len(ids), num_levels)
    return QualityMatrix(dataset, metric, tuple(ids), q, len(all_ids), excluded)


# --------------------------------------------------------------------------
# F(b): best fixed mixture
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedMixture:
    value: float
    support: tuple[tuple[int, float], ...]  # (level, probability)

    @property
    def mean_level(self) -> float:
        return sum(t * p for t, p in self.support)


def upper_hull(f: np.ndarray) -> list[int]:
    """Indices (0-based) of the upper concave hull of points (t, f_t), t = 1..L."""
    hull: list[int] = []
    for i in range(len(f)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # Drop b if it lies on or below the chord a -> i.
            if (f[b] - f[a]) * (i - a) <= (f[i] - f[a]) * (b - a):
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


def best_fixed_mixture(f: np.ndarray, b: float) -> FixedMixture:
    """max sum_t p_t f_t s.t. p >= 0, sum p = 1, sum p_t t <= b (levels 1..L)."""
    levels = np.arange(1, len(f) + 1)
    if b < levels[0]:
        raise ValueError(f"budget {b} below the first level")
    hull = upper_hull(f)
    best = int(np.argmax(f))  # hull is non-decreasing up to its maximum
    if levels[best] <= b:
        return FixedMixture(float(f[best]), ((int(levels[best]), 1.0),))
    for a, c in zip(hull, hull[1:], strict=False):
        if levels[a] <= b <= levels[c]:
            w = (b - levels[a]) / (levels[c] - levels[a])
            value = (1 - w) * f[a] + w * f[c]
            support = tuple(
                (int(levels[i]), float(p)) for i, p in ((a, 1 - w), (c, w)) if p > 0
            )
            return FixedMixture(float(value), support)
    raise AssertionError("budget not bracketed by the hull")


def legacy_adjacent_mix(f: np.ndarray, b: float) -> float:
    """Old equal-observation mix: floor/ceil levels around b."""
    lo = int(math.floor(b))
    if lo >= len(f):
        return float(f[-1])
    w = b - lo
    return float((1 - w) * f[lo - 1] + w * f[lo]) if w > 0 else float(f[lo - 1])


# --------------------------------------------------------------------------
# O(b): multiple-choice knapsack
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetOracle:
    value: float  # mean q over images
    assignment: np.ndarray  # chosen level per image (1-based)
    budget_total: int


def budget_oracle(q: np.ndarray, b: float) -> BudgetOracle:
    """Choose one level per image maximising sum q, with sum of levels <= floor(N*b)."""
    n, num_levels = q.shape
    total = int(math.floor(n * b))
    if total < n:
        raise ValueError("budget below one level per image")
    neg = -np.inf
    dp = np.full(total + 1, neg)
    dp[0] = 0.0
    choice = np.zeros((n, total + 1), dtype=np.int16)
    for i in range(n):
        new = np.full(total + 1, neg)
        for t in range(1, num_levels + 1):
            cand = np.full(total + 1, neg)
            cand[t:] = dp[: total + 1 - t] + q[i, t - 1]
            better = cand > new
            new[better] = cand[better]
            choice[i, better] = t
        dp = new
    spent = int(np.argmax(dp))
    assignment = np.zeros(n, dtype=int)
    for i in range(n - 1, -1, -1):
        t = int(choice[i, spent])
        assignment[i] = t
        spent -= t
    return BudgetOracle(float(dp.max() / n), assignment, total)


# --------------------------------------------------------------------------
# t10 / t20 two-choice oracle
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoChoice:
    n: int
    k: int
    value: float  # mean q of the assignment
    b_half: float  # image-independent mixture at rate k/n
    chosen: tuple[int, ...]  # row indices sent to t20

    @property
    def gain(self) -> float:
        return self.value - self.b_half


def top_k_by_gain(g: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest g (stable: ties keep row order)."""
    return np.argsort(-g, kind="stable")[:k]


def assignment_value(
    q_lo: np.ndarray, q_hi: np.ndarray, chosen: Sequence[int] | np.ndarray
) -> float:
    """Mean q when rows in ``chosen`` take q_hi and the others q_lo."""
    q = q_lo.astype(float).copy()
    q[list(chosen)] = q_hi[list(chosen)]
    return float(q.mean())


def mixture_value(q_lo: np.ndarray, q_hi: np.ndarray, rate: float) -> float:
    return float(np.mean(q_lo + rate * (q_hi - q_lo)))


def two_choice_oracle(
    q_lo: np.ndarray, q_hi: np.ndarray, k: int | None = None
) -> TwoChoice:
    n = len(q_lo)
    k = n // 2 if k is None else k
    chosen = top_k_by_gain(q_hi - q_lo, k)
    return TwoChoice(n, k, assignment_value(q_lo, q_hi, chosen),
                     mixture_value(q_lo, q_hi, k / n), tuple(int(c) for c in chosen))  # fmt: skip


# --------------------------------------------------------------------------
# Shuffle diagnostic
# --------------------------------------------------------------------------


def permuted_stops(stops: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same multiset of stop levels, randomly re-assigned to images."""
    return rng.permutation(stops)


def shuffle_diagnostic(
    q: np.ndarray,
    stops: np.ndarray,
    repeats: int = SHUFFLE_REPEATS,
    seed: int = SHUFFLE_SEED,
) -> dict[str, float]:
    """Observed mean q(stop) vs permutations of the stop list across images."""
    n = len(stops)
    rows = np.arange(n)
    observed = float(q[rows, stops - 1].mean())
    rng = np.random.default_rng(seed)
    perm_means = np.array(
        [q[rows, permuted_stops(stops, rng) - 1].mean() for _ in range(repeats)]
    )
    return {
        "n": n,
        "mean_stop": float(stops.mean()),
        "observed": observed,
        "perm_mean": float(perm_means.mean()),
        "perm_sd": float(perm_means.std(ddof=1)),
        "perm_p_ge_observed": float((perm_means >= observed - 1e-12).mean()),
        "observed_minus_perm": observed - float(perm_means.mean()),
    }


# --------------------------------------------------------------------------
# Cross-seed (plan §5.1)
# --------------------------------------------------------------------------

MULTISEED_COLUMNS = ("image_id", "seed", "level")


WIDE_LEVEL_RE = re.compile(r"^q(\d+)$")


def wide_to_long(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """A1's wide multiseed_gain.csv (image_id, seed, metric, q5..q20, g, valid) ->
    long rows (image_id, seed, level, <metric>). Invalid levels are blank there,
    so present values are taken as valid; one row per (image, seed, metric)."""
    sel = df[df["metric"] == metric]
    if sel.duplicated(["image_id", "seed"]).any():
        raise ValueError(f"duplicate (image_id, seed) rows for metric {metric}")
    level_cols = {
        c: int(m.group(1)) for c in df.columns if (m := WIDE_LEVEL_RE.match(c))
    }
    long = sel.melt(id_vars=["image_id", "seed"], value_vars=list(level_cols),
                    var_name="col", value_name=metric)  # fmt: skip
    return long.assign(level=long["col"].map(level_cols)).drop(columns="col")


def multiseed_table(df: pd.DataFrame, metric: str) -> dict[int, pd.DataFrame]:
    """seed -> (image x level) table of valid values; duplicates raise.

    Accepts the long format (image_id, seed, level, R50, relsg_gt[, valid]) or
    A1's wide format (image_id, seed, metric, q5, q10, q15, q20, g, valid).
    """
    if "metric" in df.columns:
        df = wide_to_long(df, metric)
    missing = [c for c in (*MULTISEED_COLUMNS, metric) if c not in df.columns]
    if missing:
        raise ValueError(f"multiseed table lacks {missing}")
    if df.duplicated(list(MULTISEED_COLUMNS)).any():
        raise ValueError("duplicate (image_id, seed, level) rows")
    values = pd.to_numeric(df[metric], errors="coerce")
    ok = values.notna() & (df["valid"].map(_valid) if "valid" in df.columns else True)
    clean = df.assign(q=values)[ok]
    return {
        int(seed): grp.pivot(index="image_id", columns="level", values="q")
        for seed, grp in clean.groupby("seed")
    }


def _common_complete(
    tables: Sequence[pd.DataFrame], levels: Sequence[int]
) -> list[int]:
    ids = None
    for tab in tables:
        cols = [lv for lv in levels if lv in tab.columns]
        good = (
            set(tab.index[tab[cols].notna().all(axis=1)])
            if len(cols) == len(levels)
            else set()
        )
        ids = good if ids is None else ids & good
    return sorted(ids or [])


def spearman_or_undefined(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho; NaN (undefined) if either side is constant or n < 3."""
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(spearmanr(x, y).statistic)


def seed_differences(tables: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Per seed pair and level: mean / sd / max |q_s - q_s'| over common images."""
    rows = []
    seeds = sorted(tables)
    for a_i, a in enumerate(seeds):
        for b in seeds[a_i + 1 :]:
            for lv in sorted(set(tables[a].columns) & set(tables[b].columns)):
                both = pd.concat([tables[a][lv], tables[b][lv]], axis=1).dropna()
                d = (both.iloc[:, 0] - both.iloc[:, 1]).to_numpy()
                rows.append({"seed_a": a, "seed_b": b, "level": int(lv), "n": len(d),
                             "mean_abs_diff": float(np.abs(d).mean()) if len(d) else math.nan,
                             "sd_diff": float(d.std(ddof=1)) if len(d) > 1 else math.nan,
                             "max_abs_diff": float(np.abs(d).max()) if len(d) else math.nan})  # fmt: skip
    return pd.DataFrame(rows)


TIE_DRAWS = 2000
TIE_SEED = 20261008
TIE_QUANTILES = (0.05, 0.95)


def tie_order_sensitivity(
    g_sel: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray, k: int, b_half: float,
    draws: int = TIE_DRAWS, seed: int = TIE_SEED,
) -> dict[str, float]:  # fmt: skip
    """Cross-seed gain over random orders among tied g (ranking otherwise fixed).

    The main estimate breaks ties by image order; many R50 gains are exactly 0,
    so the top-k set can depend on that arbitrary order.
    """
    if len(g_sel) == 0:
        return {"tie_random_mean": math.nan, "tie_random_p05": math.nan,
                "tie_random_p95": math.nan}  # fmt: skip
    rng = np.random.default_rng(seed)
    gains = np.array([
        assignment_value(q_lo, q_hi, np.lexsort((rng.random(len(g_sel)), -g_sel))[:k]) - b_half
        for _ in range(draws)
    ])  # fmt: skip
    lo, hi = np.quantile(gains, TIE_QUANTILES)
    return {"tie_random_mean": float(gains.mean()), "tie_random_p05": float(lo),
            "tie_random_p95": float(hi)}  # fmt: skip


def cross_seed_oracle(
    tables: dict[int, pd.DataFrame], select_seed: int, eval_seed: int,
    lo: int = LOW_LEVEL, hi: int = HIGH_LEVEL, k: int | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Top-k t20 assignment chosen on ``select_seed`` g, scored ONLY on ``eval_seed``."""
    ids = _common_complete([tables[select_seed], tables[eval_seed]], (lo, hi))
    sel = tables[select_seed].loc[ids]
    ev = tables[eval_seed].loc[ids]
    g_sel = (sel[hi] - sel[lo]).to_numpy()
    q_lo, q_hi = ev[lo].to_numpy(), ev[hi].to_numpy()
    n = len(ids)
    k = n // 2 if k is None else k
    chosen = top_k_by_gain(g_sel, k)
    value = assignment_value(q_lo, q_hi, chosen)
    b_half = mixture_value(q_lo, q_hi, k / n) if n else math.nan
    same = two_choice_oracle(q_lo, q_hi, k)
    ties = tie_order_sensitivity(g_sel, q_lo, q_hi, k, b_half)
    return {
        "n_gain_pos_select": int((g_sel > 0).sum()),
        "n_gain_zero_select": int((g_sel == 0).sum()),
        **ties,
        "select_seed": select_seed, "eval_seed": eval_seed, "n": n, "k": k,
        "assignment_value": value, "b_half_eval": b_half,
        "cross_seed_gain": value - b_half,
        "same_seed_oracle_gain": same.gain,
        "spearman_g": spearman_or_undefined(g_sel, q_hi - q_lo),
        "chosen_ids": ";".join(str(ids[c]) for c in chosen),
    }  # fmt: skip
