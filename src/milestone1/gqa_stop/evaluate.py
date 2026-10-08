"""Scoring, frozen-policy application, fixed references and image bootstrap.

Plan §7 with §16 A5: the primary fixed reference is the upper concave hull
F(b) of the 20 fixed-stop points, recomputed inside every bootstrap replicate;
the adjacent-level mix is kept as a secondary reference.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import atomic_write_json
from .dataset import normalize_text
from .inference import SCORE_COLUMNS
from .policy import (
    REASON_FIXED,
    REASON_MET_AT_LAST,
    REASON_NEVER_MET,
    PolicyParams,
    policy_params,
    replay_stop,
    theta_from_json,
)

PERCENTILE_LOW = 2.5
PERCENTILE_HIGH = 97.5
PRIMARY_COMPARISONS = (
    ("checklist_min", "direct"),
    ("checklist_min", "checklist_mean"),
)


def normalize_answer(text: str) -> str:
    """NFKC -> lowercase -> strip -> collapse whitespace (plan §7.1)."""
    return normalize_text(text)


def is_correct(prediction: str | None, gt: str) -> bool:
    """Exact match after normalisation; an invalid (None) answer is incorrect."""
    if prediction is None:
        return False
    return normalize_answer(prediction) == normalize_answer(gt)


def add_correctness(table: pd.DataFrame, gt: Mapping[str, str]) -> pd.DataFrame:
    flags = [
        is_correct(None if pd.isna(a) else str(a), gt[str(q)])
        for q, a in zip(table["question_id"], table["reader_answer"], strict=True)
    ]
    return table.assign(reader_correct=flags)


# ---- fixed references -------------------------------------------------------
def upper_concave_hull(xs: Sequence[float], ys: Sequence[float]) -> list[tuple]:
    """Upper hull of points sorted by x (monotone chain)."""
    points = sorted(zip(xs, ys, strict=True))
    hull: list[tuple[float, float]] = []
    for point in points:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], point) >= 0:
            hull.pop()
        hull.append(point)
    return hull


def _cross(o: tuple, a: tuple, b: tuple) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def hull_value(curve: Mapping[int, float], b: float) -> float:
    """F(b): linear interpolation on the upper concave hull of (t, Acc_t)."""
    hull = upper_concave_hull(list(curve), list(curve.values()))
    return float(np.interp(b, [p[0] for p in hull], [p[1] for p in hull]))


def adjacent_mix(curve: Mapping[int, float], b: float) -> float:
    """Linear mix of Fixed-floor(b) and Fixed-ceil(b) (secondary reference)."""
    low, high = math.floor(b), math.ceil(b)
    if low == high:
        return float(curve[low])
    return float(curve[low] * (high - b) + curve[high] * (b - low))


# ---- applying the frozen policy ---------------------------------------------
@dataclass(frozen=True)
class EvalData:
    question_ids: list[str]
    group_ids: list[str]
    correct: np.ndarray  # (n_questions, n_levels) bool
    stops: dict[str, np.ndarray]  # policy method -> stop level per question
    reasons: dict[str, np.ndarray]  # policy method -> stop reason per question
    thetas: dict[str, float]
    n_levels: int
    infeasible: list[str]


def method_label(method: str, target: str) -> str:
    return f"{method}@{target}"


def frozen_params(frozen: Mapping[str, Any], n_levels: int) -> PolicyParams:
    stored = frozen.get("policy_params") or {
        "start_level": 1,
        "consecutive_k": 1,
        "force_stop_level": n_levels,
    }
    return policy_params(stored, n_levels)


def build_eval_data(
    table: pd.DataFrame, frozen: Mapping[str, Any], n_levels: int
) -> EvalData:
    params = frozen_params(frozen, n_levels)
    ordered = table.sort_values(["question_id", "level"])
    qids = sorted(ordered["question_id"].unique().tolist())
    first = ordered.groupby("question_id", sort=True).first()
    correct = np.zeros((len(qids), n_levels), dtype=bool)
    scores: dict[str, np.ndarray] = {
        m: np.zeros((len(qids), n_levels)) for m in SCORE_COLUMNS
    }
    index = {q: i for i, q in enumerate(qids)}
    for row in ordered.itertuples(index=False):
        i, t = index[row.question_id], int(row.level) - 1
        correct[i, t] = bool(row.reader_correct)
        for method, column in SCORE_COLUMNS.items():
            scores[method][i, t] = float(getattr(row, column))
    stops: dict[str, np.ndarray] = {}
    reasons: dict[str, np.ndarray] = {}
    thetas: dict[str, float] = {}
    infeasible: list[str] = []
    for target, chosen in frozen["selections"].items():
        for method in frozen["methods"]:
            theta = theta_from_json(chosen[method]["theta"])
            label = method_label(method, target)
            if theta is None:
                infeasible.append(label)  # no dev threshold met this target
                continue
            decisions = [
                replay_stop(list(s), theta, params.start_level, params.max_level)
                for s in scores[method]
            ]
            stops[label] = np.array([d.level for d in decisions])
            reasons[label] = np.array([d.reason for d in decisions])
            thetas[label] = theta
        fixed_dev = chosen.get("fixed_dev")
        label = f"fixed_dev@{target}"
        if fixed_dev is None:
            infeasible.append(label)
            continue
        stops[label] = np.full(len(qids), int(fixed_dev))
        reasons[label] = np.full(len(qids), REASON_FIXED)
    groups = [str(first.loc[q, "group_id"]) for q in qids]
    return EvalData(qids, groups, correct, stops, reasons, thetas, n_levels, infeasible)


def metrics_for(
    correct: np.ndarray, stop: np.ndarray, reasons: np.ndarray | None = None
) -> dict[str, float]:
    """Question-weighted metrics; stop-reason rates only for threshold methods.

    forced20_rate follows §7.2: the threshold was not met at t < 20, so the
    question reached Level 20 (natural pass at 20 included, reason recorded
    separately). stop20_rate is any stop at the last level; with start_level 1
    and k = 1 the two coincide by construction.
    """
    n_levels = correct.shape[1]
    hits = correct[np.arange(len(stop)), stop - 1]
    result = {
        "accuracy": float(hits.mean()),
        "mean_level": float(stop.mean()),
        "median_level": float(np.median(stop)),
        "stop20_rate": float((stop == n_levels).mean()),
    }
    threshold_method = reasons is not None and not (reasons == REASON_FIXED).all()
    if threshold_method:
        result["forced20_rate"] = float((stop == n_levels).mean())
        result["never_met_rate"] = float((reasons == REASON_NEVER_MET).mean())
        result["natural_pass20_rate"] = float((reasons == REASON_MET_AT_LAST).mean())
    return result


def fixed_points(correct: np.ndarray) -> dict[int, float]:
    return {t + 1: float(correct[:, t].mean()) for t in range(correct.shape[1])}


def point_estimates(data: EvalData, idx: np.ndarray | None = None) -> dict[str, float]:
    """Flat metric dict for one (re)sample; references recomputed from it."""
    sel = np.arange(len(data.question_ids)) if idx is None else idx
    correct = data.correct[sel]
    curve = fixed_points(correct)
    out: dict[str, float] = {f"fixed_{t}.accuracy": a for t, a in curve.items()}
    for label, stop in data.stops.items():
        m = metrics_for(correct, stop[sel])
        out[f"{label}.accuracy"] = m["accuracy"]
        out[f"{label}.mean_level"] = m["mean_level"]
        out[f"{label}.hull_gap"] = m["accuracy"] - hull_value(curve, m["mean_level"])
        out[f"{label}.adjacent_gap"] = m["accuracy"] - adjacent_mix(
            curve, m["mean_level"]
        )
    for target in _targets(data):
        for a, b in PRIMARY_COMPARISONS:
            la, lb = method_label(a, target), method_label(b, target)
            if la in data.stops and lb in data.stops:
                key = f"{la}-{lb}"
                out[f"{key}.accuracy_diff"] = (
                    out[f"{la}.accuracy"] - out[f"{lb}.accuracy"]
                )
                out[f"{key}.mean_level_diff"] = (
                    out[f"{la}.mean_level"] - out[f"{lb}.mean_level"]
                )
    return out


def _targets(data: EvalData) -> list[str]:
    return sorted({label.split("@")[1] for label in data.stops})


def cluster_bootstrap_indices(
    group_ids: Sequence[str], repeats: int, seed: int
) -> list[np.ndarray]:
    """Resample unique images with replacement; keep all their questions."""
    unique = sorted(set(group_ids))
    members = {
        g: np.array([i for i, x in enumerate(group_ids) if x == g]) for g in unique
    }
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repeats):
        drawn = rng.integers(0, len(unique), size=len(unique))
        samples.append(np.concatenate([members[unique[k]] for k in drawn]))
    return samples


def bootstrap(data: EvalData, repeats: int, seed: int) -> dict[str, dict[str, float]]:
    point = point_estimates(data)
    draws: dict[str, list[float]] = {k: [] for k in point}
    for idx in cluster_bootstrap_indices(data.group_ids, repeats, seed):
        for key, value in point_estimates(data, idx).items():
            draws[key].append(value)
    return {
        key: {
            "estimate": point[key],
            "ci_low": float(np.percentile(draws[key], PERCENTILE_LOW)),
            "ci_high": float(np.percentile(draws[key], PERCENTILE_HIGH)),
        }
        for key in point
    }


def noninferiority_verdict(
    acc_diff: Mapping[str, float], level_diff: Mapping[str, float], margin: float
) -> str:
    """Conservative §7.3 rule; otherwise distinguish the outcome descriptively."""
    if acc_diff["ci_low"] >= -margin and level_diff["ci_high"] < 0:
        return "supported: earlier with accuracy loss CI above -margin"
    if level_diff["ci_high"] < 0:
        return "trade_off_or_unclear: earlier, accuracy CI crosses -margin"
    if level_diff["ci_low"] > 0:
        return "not_supported: later stop"
    return "unclear: mean-level difference CI includes 0"


def method_summary(data: EvalData, frozen: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for t in range(1, data.n_levels + 1):
        stop = np.full(len(data.question_ids), t)
        rows.append(
            {"method": f"fixed_{t}", "theta": None, **metrics_for(data.correct, stop)}
        )
    for label, stop in data.stops.items():
        m = metrics_for(data.correct, stop, data.reasons[label])
        rows.append(
            {"method": label, "theta": data.thetas.get(label), "status": "ok", **m}
        )
    for label in data.infeasible:
        rows.append({"method": label, "theta": None, "status": "infeasible_on_dev"})
    frame = pd.DataFrame(rows)
    frame["accuracy_loss_vs_l20"] = (
        float(data.correct[:, -1].mean()) - frame["accuracy"]
    )
    frame["mean_observed_level_reduction"] = 1 - frame["mean_level"] / data.n_levels
    frame["primary_target"] = frozen.get("primary_target")
    return frame


def stops_frame(data: EvalData, table: pd.DataFrame) -> pd.DataFrame:
    answers = {
        (r.question_id, int(r.level)): r.reader_answer
        for r in table.itertuples(index=False)
    }
    rows = []
    for label, stop in data.stops.items():
        for i, qid in enumerate(data.question_ids):
            level = int(stop[i])
            rows.append(
                {
                    "method": label,
                    "theta": data.thetas.get(label),
                    "question_id": qid,
                    "group_id": data.group_ids[i],
                    "stop_level": level,
                    "forced20": level == data.n_levels
                    and data.reasons[label][i] != REASON_FIXED,
                    "stop_reason": str(data.reasons[label][i]),
                    "reader_answer": answers.get((qid, level)),
                    "correct": bool(data.correct[i, level - 1]),
                }
            )
    return pd.DataFrame(rows)


def comparisons_frame(
    boot: Mapping[str, Mapping[str, float]], margin: float, primary: str
) -> pd.DataFrame:
    rows = []
    for key, stats in boot.items():
        name, metric = key.rsplit(".", 1)
        if metric not in {
            "accuracy_diff",
            "mean_level_diff",
            "hull_gap",
            "adjacent_gap",
        }:
            continue
        rows.append({"comparison": name, "metric": metric, **stats})
    frame = pd.DataFrame(rows)
    verdicts = []
    for name in frame["comparison"].unique() if not frame.empty else []:
        acc = boot.get(f"{name}.accuracy_diff")
        lvl = boot.get(f"{name}.mean_level_diff")
        if acc and lvl:
            verdicts.append(
                {
                    "comparison": name,
                    "metric": "verdict",
                    "verdict": noninferiority_verdict(acc, lvl, margin),
                    "is_primary_target": name.endswith(f"@{primary}"),
                }
            )
    return pd.concat([frame, pd.DataFrame(verdicts)], ignore_index=True)


def subgroup_summary(data: EvalData, table: pd.DataFrame, primary: str) -> pd.DataFrame:
    meta = table.groupby("question_id", sort=True).first()
    rows = []
    for column in ("attribute", "relate_count"):
        values = np.array([meta.loc[q, column] for q in data.question_ids])
        for value in sorted(set(values.tolist())):
            mask = values == value
            for label, stop in data.stops.items():
                if not label.endswith(f"@{primary}"):
                    continue
                m = metrics_for(data.correct[mask], stop[mask])
                rows.append(
                    {
                        "group_by": column,
                        "value": value,
                        "method": label,
                        "n": int(mask.sum()),
                        **m,
                    }
                )
    return pd.DataFrame(rows)


def diagnostics(table: pd.DataFrame, gt: Mapping[str, str]) -> dict[str, Any]:
    """Secondary diagnostics (plan §5.2, §7.2, §16 A8); never used for selection."""
    both = table[
        (table["reader_status"] == "valid") & (table["direct_status"] == "valid")
    ]
    disagree = [
        normalize_answer(str(a)) != normalize_answer(str(b))
        for a, b in zip(both["reader_answer"], both["direct_answer"], strict=True)
    ]
    planned = table[table["n_items"] > 0]
    variants = 0
    for row in table.itertuples(index=False):
        if (
            row.reader_correct
            or row.reader_answer is None
            or pd.isna(row.reader_answer)
        ):
            continue
        pred, ref = (
            normalize_answer(str(row.reader_answer)),
            normalize_answer(gt[str(row.question_id)]),
        )
        if pred and (pred in ref or ref in pred):
            variants += 1
    per_question = table.groupby("question_id").first()
    return {
        "rows": len(table),
        "direct_reader_disagreement_rate": float(np.mean(disagree))
        if disagree
        else None,
        "checklist_unresolved_item_rate": (
            float((planned["unresolved_items"] / planned["n_items"]).mean())
            if not planned.empty
            else None
        ),
        "planner_failures": int((per_question["planner_status"] != "valid").sum()),
        "n_items_distribution": _counts(per_question["n_items"]),
        "status_counts": {
            role: _counts(table[f"{role}_status"])
            for role in ("reader", "direct", "check")
        },
        "surface_variant_errors": variants,
        "surface_variant_rule": "incorrect and one normalised string contains the other",
    }


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().items()}


def evaluate(
    table: pd.DataFrame,
    frozen: Mapping[str, Any],
    eval_cfg: Mapping[str, Any],
    n_levels: int,
    output_dir: Path,
    gt: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write stops.csv, method_summary.csv, paired_comparisons.csv, bootstrap_summary.json."""
    data = build_eval_data(table, frozen, n_levels)
    repeats = int(eval_cfg["bootstrap_repeats"])
    seed = int(eval_cfg["bootstrap_seed"])
    margin = float(eval_cfg["accuracy_noninferiority_margin"])
    primary = str(frozen["primary_target"])
    boot = bootstrap(data, repeats, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    stops_frame(data, table).to_csv(output_dir / "stops.csv", index=False)
    method_summary(data, frozen).to_csv(output_dir / "method_summary.csv", index=False)
    comparisons_frame(boot, margin, primary).to_csv(
        output_dir / "paired_comparisons.csv", index=False
    )
    subgroup_summary(data, table, primary).to_csv(
        output_dir / "subgroup_summary.csv", index=False
    )
    summary = {
        "n_questions": len(data.question_ids),
        "n_images": len(set(data.group_ids)),
        "bootstrap_unit": "image (duplicate group)",
        "repeats": repeats,
        "seed": seed,
        "ci": [PERCENTILE_LOW, PERCENTILE_HIGH],
        "noninferiority_margin": margin,
        "primary_target": primary,
        "infeasible_targets": data.infeasible,
        "fixed_reference_primary": "upper concave hull of fixed-t points, per replicate",
        "fixed_reference_secondary": "adjacent-level linear mix, per replicate",
        "statistics": boot,
    }
    atomic_write_json(output_dir / "bootstrap_summary.json", summary)
    if gt is not None:
        atomic_write_json(output_dir / "diagnostics.json", diagnostics(table, gt))
    return summary
