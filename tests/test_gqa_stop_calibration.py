"""Pilot calibration diagnostics and gate (plan §16 A4)."""

from __future__ import annotations

import pandas as pd
import pytest

from milestone1.gqa_stop.calibration import (
    calibration_report,
    choose_model,
    first_correct_levels,
    spearman,
)

GATE = {
    "min_spearman": 0.3,
    "max_level1_mean_confidence": 0.8,
    "early_overconfidence_threshold": 0.8,
}
LEVELS = [1, 5, 10, 15, 20]


def pilot_table(
    direct: dict[int, float], check: dict[int, float], first_correct: int = 10
) -> pd.DataFrame:
    rows = []
    for q in ("a", "b"):
        for t in LEVELS:
            rows.append(
                {
                    "question_id": q,
                    "image_id": q,
                    "level": t,
                    "reader_correct": t >= first_correct,
                    "direct_status": "valid",
                    "direct_confidence": direct[t],
                    "check_status": "valid",
                    "r_values": {"c1": 0.9, "c2": check[t]},
                }
            )
    return pd.DataFrame(rows)


def test_spearman_by_hand() -> None:
    assert spearman(pd.Series([1, 2, 3]), pd.Series([0.1, 0.5, 0.9])) == pytest.approx(
        1.0
    )
    assert spearman(pd.Series([1, 2, 3]), pd.Series([0.9, 0.5, 0.1])) == pytest.approx(
        -1.0
    )
    assert pd.isna(spearman(pd.Series([1, 2, 3]), pd.Series([0.5, 0.5, 0.5])))


def test_report_per_level_and_early_overconfidence() -> None:
    direct = {1: 0.9, 5: 0.9, 10: 0.95, 15: 0.95, 20: 0.95}
    check = {1: 0.0, 5: 0.2, 10: 0.8, 15: 0.9, 20: 0.95}
    rep = calibration_report(pilot_table(direct, check), GATE, {"name": "m"})
    d = rep["direct"]
    assert d["per_level"]["1"]["mean"] == pytest.approx(0.9)
    assert d["per_level"]["1"]["std"] == pytest.approx(0.0)
    assert d["early_overconfidence_rate"] == 1.0  # L1 and L5 before first correct L10
    assert d["pairs_before_first_correct"] == 4
    c = rep["checklist_effective_r"]
    assert c["pairs_before_first_correct"] == 8  # 2 items x 2 levels x 2 questions
    assert c["early_overconfidence_pairs"] == 4  # c1 = 0.9 at L1/L5
    assert d["fails"] is True  # L1 mean 0.9 >= 0.8
    assert c["fails"] is False
    assert rep["gate_passed"] is True  # fails only if BOTH fail
    assert rep["first_correct_level_histogram"] == {"10": 2}


def test_gate_fails_when_both_flat_or_high() -> None:
    flat = dict.fromkeys(LEVELS, 0.9)
    rep = calibration_report(pilot_table(flat, flat), GATE, {"name": "m"})
    assert rep["direct"]["spearman_pooled"] is None
    assert rep["gate_passed"] is False


def test_first_correct_never() -> None:
    table = pilot_table(dict.fromkeys(LEVELS, 0.1), dict.fromkeys(LEVELS, 0.1), 99)
    assert first_correct_levels(table) == {"a": float("inf"), "b": float("inf")}


def test_choose_model_prefers_higher_rho_then_lower_l1() -> None:
    def rep(rho: float, low: float) -> dict:
        part = {"spearman_pooled": rho, "lowest_level_mean": low}
        return {"direct": part, "checklist_effective_r": part}

    assert choose_model({"9b": rep(0.1, 0.9), "4b": rep(0.4, 0.95)}) == "4b"
    assert choose_model({"9b": rep(0.4, 0.5), "4b": rep(0.4, 0.7)}) == "9b"
