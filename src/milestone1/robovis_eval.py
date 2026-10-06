"""Stop-quality evaluation for ROBOVIS2027 (plan §9).

Inputs are per-image, per-level recall values ``q_t`` (R@50) from
``stage_metrics.csv`` and a long table of stops
(``image_id, method, stop_t, natural_stop, forced_terminal, failure_count``).

Undefined quantities are never filled with the final level: oracle-based MAE
is computed over images with a defined oracle only, and ``n`` is reported.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from milestone1.stop_policy import fixed_method_name

ORACLE_DEFINED = "defined"
ORACLE_UNSTABLE_TAIL = "unstable_tail"
ORACLE_ZERO_RECOVERY = "zero_recovery"
ORACLE_INVALID_LEVELS = "invalid_levels"
FLOAT_TOL = 1e-12

STAGE_REQUIRED_COLUMNS = ("image_id", "t", "valid")
PRED_HITS_BASE_COLUMNS = ("image_id", "t", "pred_id", "n_gt")
PRED_HITS_FILENAME = "stage_predicate_hits.csv"
STOPS_COLUMNS = (
    "image_id",
    "method",
    "stop_t",
    "natural_stop",
    "forced_terminal",
    "failure_count",
)
TRUE_STRINGS = frozenset({"true", "1", "yes", "t"})


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def recall_column(recall_k: int) -> str:
    return f"R{recall_k}"


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | float | np.integer | np.floating):
        return not math.isnan(float(value)) and bool(value)
    return str(value).strip().lower() in TRUE_STRINGS


@dataclass(frozen=True)
class RecallCurves:
    """q_t per image (NaN where the row is missing) and per-level validity."""

    q: dict[int, np.ndarray]
    valid: dict[int, np.ndarray]
    num_levels: int

    def all_valid(self, image_id: int) -> bool:
        v = self.valid[image_id]
        return bool(v.all()) and not bool(np.isnan(self.q[image_id]).any())


def load_stage_metrics(path: str | Path, recall_k: int) -> pd.DataFrame:
    """Read stage_metrics.csv; only image_id, t, valid and R@k are required."""
    df = pd.read_csv(path)
    col = recall_column(recall_k)
    missing = [c for c in (*STAGE_REQUIRED_COLUMNS, col) if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df.copy()
    df["image_id"] = df["image_id"].astype(int)
    df["t"] = df["t"].astype(int)
    df["valid"] = df["valid"].map(_to_bool)
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_curves(stage: pd.DataFrame, recall_k: int, num_levels: int) -> RecallCurves:
    col = recall_column(recall_k)
    q: dict[int, np.ndarray] = {}
    valid: dict[int, np.ndarray] = {}
    for rec in stage.to_dict("records"):
        image_id = int(rec["image_id"])
        if image_id not in q:
            q[image_id] = np.full(num_levels, np.nan)
            valid[image_id] = np.zeros(num_levels, dtype=bool)
        t = int(rec["t"])
        if 1 <= t <= num_levels:
            q[image_id][t - 1] = float(rec[col])
            valid[image_id][t - 1] = bool(rec["valid"])
    return RecallCurves(dict(sorted(q.items())), valid, num_levels)


def load_predicate_hits(path: str | Path, recall_k: int) -> pd.DataFrame | None:
    """Optional per-predicate sidecar (``image_id,t,pred_id,n_gt,hits<k>``).

    Returns columns ``image_id,t,pred_id,n_gt,hits`` or None if absent.
    """
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    hits_col = f"hits{recall_k}"
    missing = [c for c in (*PRED_HITS_BASE_COLUMNS, hits_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df.rename(columns={hits_col: "hits"})
    return df.astype({"image_id": int, "t": int, "pred_id": int})


# --------------------------------------------------------------------------
# Oracle and residual
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleResult:
    status: str
    t: int | None
    qmax: float


def relative_stable_oracle(
    q: Sequence[float] | np.ndarray,
    valid: Sequence[bool] | np.ndarray | None,
    ratio: float,
) -> OracleResult:
    """First t with q_u >= ratio * qmax for all u >= t (1-based).

    ``zero_recovery`` if qmax == 0; ``invalid_levels`` if any level is
    invalid/missing; ``unstable_tail`` if no such t exists.
    """
    qa = np.asarray(q, dtype=float)
    va = np.ones(len(qa), dtype=bool) if valid is None else np.asarray(valid, bool)
    if len(qa) == 0 or np.isnan(qa).any() or not va.all():
        qmax = float(np.nanmax(qa)) if len(qa) and not np.isnan(qa).all() else math.nan
        return OracleResult(ORACLE_INVALID_LEVELS, None, qmax)
    qmax = float(qa.max())
    if qmax <= 0.0:
        return OracleResult(ORACLE_ZERO_RECOVERY, None, qmax)
    threshold = ratio * qmax - FLOAT_TOL
    stable = qa >= threshold
    # suffix_ok[i] is True iff stable[i:] is all True.
    suffix_ok = np.flip(np.logical_and.accumulate(np.flip(stable)))
    if not suffix_ok.any():
        return OracleResult(ORACLE_UNSTABLE_TAIL, None, qmax)
    return OracleResult(ORACLE_DEFINED, int(np.argmax(suffix_ok)) + 1, qmax)


def residual(q: Sequence[float] | np.ndarray, stop_t: int) -> float:
    """max(0, qmax - q_stop)."""
    qa = np.asarray(q, dtype=float)
    return max(0.0, float(qa.max()) - float(qa[stop_t - 1]))


# --------------------------------------------------------------------------
# Eval set
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSet:
    ids: tuple[int, ...]
    excluded: dict[int, str]
    oracle: dict[int, OracleResult]


def build_eval_set(curves: RecallCurves, ratio: float) -> EvalSet:
    """Images with all levels present and valid; oracle per eval image."""
    ids: list[int] = []
    excluded: dict[int, str] = {}
    oracle: dict[int, OracleResult] = {}
    for image_id in sorted(curves.q):
        if not curves.all_valid(image_id):
            excluded[image_id] = ORACLE_INVALID_LEVELS
            continue
        ids.append(image_id)
        oracle[image_id] = relative_stable_oracle(
            curves.q[image_id], curves.valid[image_id], ratio
        )
    return EvalSet(tuple(ids), excluded, oracle)


# --------------------------------------------------------------------------
# Stops and per-image metrics
# --------------------------------------------------------------------------


def fixed_stops(image_ids: Sequence[int], num_levels: int) -> pd.DataFrame:
    rows = [
        {
            "image_id": i,
            "method": fixed_method_name(k),
            "stop_t": k,
            "natural_stop": True,
            "forced_terminal": False,
            "failure_count": 0,
        }
        for k in range(1, num_levels + 1)
        for i in image_ids
    ]
    return pd.DataFrame(rows, columns=list(STOPS_COLUMNS))


def per_image_metrics(
    stops: pd.DataFrame, curves: RecallCurves, eval_set: EvalSet
) -> pd.DataFrame:
    """Stop-level recall, residual, oracle error per (image, method) on the eval set."""
    rows = []
    ids = set(eval_set.ids)
    for r in stops.to_dict("records"):
        image_id = int(r["image_id"])
        if image_id not in ids:
            continue
        q = curves.q[image_id]
        stop_t = int(r["stop_t"])
        orc = eval_set.oracle[image_id]
        rows.append(
            {
                "image_id": image_id,
                "method": r["method"],
                "stop_t": stop_t,
                "q_stop": float(q[stop_t - 1]),
                "q_final": float(q[-1]),
                "qmax": float(q.max()),
                "residual": residual(q, stop_t),
                "oracle_status": orc.status,
                "oracle_t": orc.t,
                "abs_err_oracle": abs(stop_t - orc.t)
                if orc.t is not None
                else math.nan,
                "natural_stop": bool(r["natural_stop"]),
                "forced_terminal": bool(r["forced_terminal"]),
                "failure_count": int(r["failure_count"]),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["oracle_t"] = df["oracle_t"].astype("Int64")
    return df


def mean_recall_at_stop(
    stops: pd.DataFrame, pred_hits: pd.DataFrame | None, image_ids: Sequence[int]
) -> float:
    """Dataset-level mR: mean over predicates of sum(hits)/sum(n_gt) at stop_t."""
    if pred_hits is None:
        return math.nan
    sel = stops[stops["image_id"].isin(image_ids)][["image_id", "stop_t"]]
    merged = sel.merge(
        pred_hits, left_on=["image_id", "stop_t"], right_on=["image_id", "t"]
    )
    if merged.empty:
        return math.nan
    agg = merged.groupby("pred_id")[["hits", "n_gt"]].sum()
    agg = agg[agg["n_gt"] > 0]
    if agg.empty:
        return math.nan
    return float((agg["hits"] / agg["n_gt"]).mean())


# --------------------------------------------------------------------------
# Equal-observation fixed mix
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedMix:
    lower: int
    weight_upper: float  # weight on fixed_{lower+1}


def equal_observation_mix(mean_stop: float, num_levels: int) -> FixedMix:
    """fixed_a / fixed_{a+1} linear mix whose mean level equals ``mean_stop``."""
    if not 1.0 <= mean_stop <= num_levels:
        raise ValueError(f"mean stop {mean_stop} outside [1, {num_levels}]")
    lower = int(math.floor(mean_stop))
    if lower >= num_levels:
        return FixedMix(num_levels, 0.0)
    return FixedMix(lower, mean_stop - lower)


def mix_recall(q_matrix: np.ndarray, mix: FixedMix) -> np.ndarray:
    """Per-image recall of the mix; q_matrix is (n_images, num_levels)."""
    lo = q_matrix[:, mix.lower - 1]
    if mix.weight_upper == 0.0:
        return lo
    hi = q_matrix[:, mix.lower]
    return (1.0 - mix.weight_upper) * lo + mix.weight_upper * hi


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def _q_matrix(curves: RecallCurves, ids: Sequence[int]) -> np.ndarray:
    return np.vstack([curves.q[i] for i in ids])


def summarize(
    per_image: pd.DataFrame,
    stops: pd.DataFrame,
    curves: RecallCurves,
    eval_set: EvalSet,
    pred_hits: pd.DataFrame | None,
    adaptive_methods: Sequence[str],
) -> pd.DataFrame:
    """One row per method with the plan §9 metrics."""
    ids = list(eval_set.ids)
    qm = _q_matrix(curves, ids)
    rows = []
    for method, grp in per_image.groupby("method", sort=False):
        defined = grp["abs_err_oracle"].dropna()
        stop_sum = int(grp["stop_t"].sum())
        row: dict[str, Any] = {
            "method": method,
            "n_images": len(grp),
            "mean_stop_t": float(grp["stop_t"].mean()),
            "R50_stop": float(grp["q_stop"].mean()),
            "mR50_stop": mean_recall_at_stop(
                stops[stops["method"] == method], pred_hits, ids
            ),
            "R50_final": float(grp["q_final"].mean()),
            "mean_residual": float(grp["residual"].mean()),
            "oracle_mae": float(defined.mean()) if len(defined) else math.nan,
            "oracle_n": len(defined),
            "natural_stop_rate": float(grp["natural_stop"].mean()),
            "forced_terminal_rate": float(grp["forced_terminal"].mean()),
            "inference_failure_rate": (
                float(grp["failure_count"].sum()) / stop_sum if stop_sum else math.nan
            ),
            "images_with_failure": int((grp["failure_count"] > 0).sum()),
            "eq_fixed_lower": math.nan,
            "eq_fixed_weight_upper": math.nan,
            "eq_fixed_R50": math.nan,
        }
        if method in adaptive_methods:
            mix = equal_observation_mix(row["mean_stop_t"], curves.num_levels)
            row["eq_fixed_lower"] = mix.lower
            row["eq_fixed_weight_upper"] = mix.weight_upper
            row["eq_fixed_R50"] = float(mix_recall(qm, mix).mean())
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Paired bootstrap
# --------------------------------------------------------------------------


def _ci(values: np.ndarray, ci_level: float) -> tuple[float, float]:
    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def _method_arrays(
    per_image: pd.DataFrame, method: str, ids: Sequence[int]
) -> dict[str, np.ndarray]:
    grp = per_image[per_image["method"] == method].set_index("image_id").loc[list(ids)]
    return {
        "R50_stop": grp["q_stop"].to_numpy(float),
        "stop_t": grp["stop_t"].to_numpy(float),
        "residual": grp["residual"].to_numpy(float),
    }


def paired_bootstrap(
    per_image: pd.DataFrame,
    curves: RecallCurves,
    eval_set: EvalSet,
    adaptive_methods: Sequence[str],
    comparisons: Sequence[tuple[str, str]],
    repeats: int,
    seed: int,
    ci_level: float,
) -> dict[str, Any]:
    """Paired bootstrap over images (same resample indices for every quantity).

    Reports, per adaptive method, the mean and CI of R@50(stop), stop_t and
    residual; the difference vs the equal-observation fixed mix (mix weights
    recomputed inside each resample from that resample's mean stop_t); and the
    requested pairwise method differences.
    """
    ids = list(eval_set.ids)
    n = len(ids)
    if n == 0:
        return {"n_images": 0, "repeats": repeats, "seed": seed, "ci_level": ci_level}
    qm = _q_matrix(curves, ids)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(repeats, n))
    full = np.arange(n)

    methods = sorted(set(adaptive_methods) | {m for pair in comparisons for m in pair})
    arrays = {m: _method_arrays(per_image, m, ids) for m in methods}

    def stat(values: np.ndarray) -> dict[str, float]:
        boot = values[idx].mean(axis=1)
        lo, hi = _ci(boot, ci_level)
        return {"mean": float(values.mean()), "ci_low": lo, "ci_high": hi}

    def eq_diff(arr: dict[str, np.ndarray], rows: np.ndarray) -> float:
        mix = equal_observation_mix(
            float(arr["stop_t"][rows].mean()), curves.num_levels
        )
        return float(arr["R50_stop"][rows].mean() - mix_recall(qm[rows], mix).mean())

    method_stats: dict[str, Any] = {}
    for m in adaptive_methods:
        arr = arrays[m]
        boot_eq = np.array([eq_diff(arr, rows) for rows in idx])
        lo, hi = _ci(boot_eq, ci_level)
        method_stats[m] = {
            **{key: stat(values) for key, values in arr.items()},
            "R50_minus_eq_fixed": {
                "mean": eq_diff(arr, full),
                "ci_low": lo,
                "ci_high": hi,
            },
        }

    pairwise: dict[str, Any] = {}
    for a, b in comparisons:
        pairwise[f"{a}_minus_{b}"] = {
            key: stat(arrays[a][key] - arrays[b][key]) for key in arrays[a]
        }

    return {
        "n_images": n,
        "repeats": repeats,
        "seed": seed,
        "ci_level": ci_level,
        "methods": method_stats,
        "pairwise": pairwise,
    }


def oracle_table(eval_set: EvalSet, curves: RecallCurves) -> pd.DataFrame:
    rows = []
    for image_id in sorted(curves.q):
        orc = eval_set.oracle.get(image_id)
        q = curves.q[image_id]
        rows.append(
            {
                "image_id": image_id,
                "in_eval_set": orc is not None,
                "oracle_status": orc.status if orc else eval_set.excluded[image_id],
                "oracle_t": orc.t if orc else None,
                "qmax": float(np.nanmax(q)) if not np.isnan(q).all() else math.nan,
                "q_final": float(q[-1]),
            }
        )
    df = pd.DataFrame(rows)
    df["oracle_t"] = df["oracle_t"].astype("Int64")
    return df


def oracle_status_counts(eval_set: EvalSet) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for orc in eval_set.oracle.values():
        counts[orc.status] = counts.get(orc.status, 0) + 1
    return counts


# --------------------------------------------------------------------------
# Figure example selection (Fig 3)
# --------------------------------------------------------------------------


def select_figure_examples(
    per_image: pd.DataFrame, method: str, baseline: str
) -> dict[str, Any]:
    """Pick improvement-max / degradation-max / median images.

    ``residual_diff = residual(baseline) - residual(method)`` (positive means
    ``method`` stopped closer to the attainable maximum). Ties are broken by
    image id; the median pick avoids re-using the two extreme images when
    another image is available.
    """
    a = per_image[per_image["method"] == method].set_index("image_id")["residual"]
    b = per_image[per_image["method"] == baseline].set_index("image_id")["residual"]
    diff = (b - a).dropna().sort_index()
    if diff.empty:
        return {"method": method, "baseline": baseline, "n": 0, "examples": {}}
    ordered = sorted(diff.items(), key=lambda kv: (-kv[1], kv[0]))
    improvement_id = int(ordered[0][0])
    degradation = sorted(diff.items(), key=lambda kv: (kv[1], kv[0]))
    degradation_id = int(degradation[0][0])
    median_value = float(diff.median())
    taken = {improvement_id, degradation_id}
    pool = [kv for kv in diff.items() if int(kv[0]) not in taken] or list(diff.items())
    median_id = int(min(pool, key=lambda kv: (abs(kv[1] - median_value), kv[0]))[0])
    return {
        "method": method,
        "baseline": baseline,
        "definition": "residual_diff = residual(baseline) - residual(method)",
        "n": len(diff),
        "median_residual_diff": median_value,
        "examples": {
            "improvement_max": {"image_id": improvement_id,
                                "residual_diff": float(diff[improvement_id])},
            "degradation_max": {"image_id": degradation_id,
                                "residual_diff": float(diff[degradation_id])},
            "median": {"image_id": median_id, "residual_diff": float(diff[median_id])},
        },
    }  # fmt: skip
