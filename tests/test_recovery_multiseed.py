"""Tests for scripts/45 gain rows: g = q20 - q10 per metric, missing never filled with 0."""

import importlib.util
import math
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "s45", REPO_ROOT / "scripts/45_recovery_multiseed_metrics.py"
)
s45 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(s45)


def _stage(rows: list[tuple[int, int, float, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["image_id", "t", "R50", "relsg_gt", "valid"])


def test_gain_is_q20_minus_q10_per_metric() -> None:
    st = _stage(
        [
            (7, 5, 0.0, 0.1, 1),
            (7, 10, 0.2, 0.4, 1),
            (7, 15, 0.3, 0.5, 1),
            (7, 20, 0.5, 0.45, 1),
        ]
    )
    rows = {r["metric"]: r for r in s45.gain_rows(st, seed=1)}
    assert math.isclose(rows["R50"]["g"], 0.3) and math.isclose(
        rows["relsg_gt"]["g"], 0.05
    )
    assert (
        rows["R50"]["seed"] == 1
        and rows["R50"]["valid"] == 1
        and rows["R50"]["q5"] == 0.0
    )


def test_missing_or_invalid_levels_stay_missing() -> None:
    st = _stage([(8, 5, 0.1, 0.2, 1), (8, 10, 0.2, 0.3, 0), (8, 20, 0.4, 0.5, 1)])
    r = {x["metric"]: x for x in s45.gain_rows(st, seed=0)}["R50"]
    assert r["q10"] is None and r["q15"] is None and r["g"] is None and r["valid"] == 0
