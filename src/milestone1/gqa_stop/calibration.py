"""Phase-B confidence calibration gate (plan §16 A4).

Only valid calls contribute confidences: Direct uses its reported confidence,
the checklist uses effective r_j (dependency rule applied). Spearman rho is the
Pearson correlation of average ranks (pandas only; scipy is unavailable).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd

VALID = "valid"


def spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman rho via ranked Pearson; NaN when either side is constant."""
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return math.nan
    return float(x.rank().corr(y.rank()))


def direct_pairs(table: pd.DataFrame) -> pd.DataFrame:
    rows = table[table["direct_status"] == VALID]
    return pd.DataFrame(
        {
            "question_id": rows["question_id"].astype(str),
            "level": rows["level"].astype(int),
            "confidence": rows["direct_confidence"].astype(float),
        }
    )


def checklist_pairs(table: pd.DataFrame) -> pd.DataFrame:
    records = []
    rows = table[table["check_status"] == VALID]
    for row in rows.itertuples(index=False):
        for item_id, value in dict(row.r_values).items():
            records.append(
                {
                    "question_id": str(row.question_id),
                    "item": item_id,
                    "level": int(row.level),
                    "confidence": float(value),
                }
            )
    return pd.DataFrame(records, columns=["question_id", "item", "level", "confidence"])


def first_correct_levels(table: pd.DataFrame) -> dict[str, float]:
    """Earliest pilot level where the common reader is correct (inf if never)."""
    result: dict[str, float] = {}
    for qid, frame in table.groupby("question_id"):
        hits = frame[frame["reader_correct"].astype(bool)]["level"]
        result[str(qid)] = float(hits.min()) if not hits.empty else math.inf
    return result


def summarize_pairs(
    pairs: pd.DataFrame, first_correct: Mapping[str, float], over_threshold: float
) -> dict[str, Any]:
    if pairs.empty:
        return {"n": 0, "per_level": {}, "spearman_pooled": None}
    per_level = {
        str(level): {
            "n": int(len(frame)),
            "mean": float(frame["confidence"].mean()),
            "std": float(frame["confidence"].std(ddof=0)),
        }
        for level, frame in pairs.groupby("level", sort=True)
    }
    per_question = [
        spearman(frame["level"], frame["confidence"])
        for _, frame in pairs.groupby("question_id")
    ]
    finite = [v for v in per_question if not math.isnan(v)]
    before = pairs[
        pairs["level"]
        < pairs["question_id"].map(lambda q: first_correct.get(q, math.inf))
    ]
    early = int((before["confidence"] >= over_threshold).sum())
    pooled = spearman(pairs["level"], pairs["confidence"])
    return {
        "n": int(len(pairs)),
        "per_level": per_level,
        "lowest_level_mean": per_level[min(per_level, key=int)]["mean"],
        "spearman_pooled": None if math.isnan(pooled) else pooled,
        "spearman_per_question_median": (
            float(pd.Series(finite).median()) if finite else None
        ),
        "questions_with_constant_confidence": len(per_question) - len(finite),
        "early_overconfidence_pairs": early,
        "pairs_before_first_correct": int(len(before)),
        "early_overconfidence_rate": early / len(before) if len(before) else None,
    }


def method_fails(summary: Mapping[str, Any], gate: Mapping[str, Any]) -> bool:
    """Fail if rho < min (or undefined) OR lowest-level mean confidence >= max."""
    rho = summary.get("spearman_pooled")
    low_mean = summary.get("lowest_level_mean")
    if rho is None or low_mean is None:
        return True
    return rho < float(gate["min_spearman"]) or low_mean >= float(
        gate["max_level1_mean_confidence"]
    )


def calibration_report(
    table: pd.DataFrame, gate: Mapping[str, Any], model: Mapping[str, Any]
) -> dict[str, Any]:
    """Diagnostics + gate verdict; the gate fails only if BOTH methods fail."""
    first = first_correct_levels(table)
    threshold = float(gate["early_overconfidence_threshold"])
    direct = summarize_pairs(direct_pairs(table), first, threshold)
    checklist = summarize_pairs(checklist_pairs(table), first, threshold)
    direct_fail = method_fails(direct, gate)
    checklist_fail = method_fails(checklist, gate)
    histogram = pd.Series(list(first.values())).value_counts().sort_index()
    return {
        "model": dict(model),
        "gate_config": dict(gate),
        "levels": sorted(int(v) for v in table["level"].unique()),
        "n_questions": int(table["question_id"].nunique()),
        "direct": {**direct, "fails": direct_fail},
        "checklist_effective_r": {**checklist, "fails": checklist_fail},
        "first_correct_level_histogram": {
            ("never" if math.isinf(k) else str(int(k))): int(v)
            for k, v in histogram.items()
        },
        "gate_passed": not (direct_fail and checklist_fail),
        "gate_rule": (
            "fail iff for both Direct and checklist r_j: pooled rho < min_spearman "
            "or lowest-pilot-level mean confidence >= max_level1_mean_confidence"
        ),
    }


def choose_model(reports: Mapping[str, Mapping[str, Any]]) -> str:
    """Between two gate-failed models: higher mean rho, then lower L1 mean."""

    def key(name: str) -> tuple[float, float]:
        rep = reports[name]
        parts = [rep["direct"], rep["checklist_effective_r"]]
        rhos = [
            p["spearman_pooled"] for p in parts if p.get("spearman_pooled") is not None
        ]
        lows = [
            p["lowest_level_mean"]
            for p in parts
            if p.get("lowest_level_mean") is not None
        ]
        mean_rho = sum(rhos) / len(rhos) if rhos else -math.inf
        mean_low = sum(lows) / len(lows) if lows else math.inf
        return (-mean_rho, mean_low)

    return min(sorted(reports), key=key)
