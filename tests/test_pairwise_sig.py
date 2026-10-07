"""Tests for the paired per-image significance helpers (milestone1.pairwise_sig)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.pairwise_sig import paired_test, pairwise_rows, per_image  # noqa: E402


def test_paired_test_detects_consistent_difference() -> None:
    d = np.full(30, 0.1) + np.random.default_rng(0).normal(0, 0.01, 30)
    t = paired_test(d, repeats=500)
    assert t["ci_low"] > 0 and t["p_up"] < 0.01 and t["p_down"] > 0.99
    zero = paired_test(np.zeros(10), repeats=200)
    assert zero["diff"] == 0 and zero["ci_low"] == 0 == zero["ci_high"]


def _setup() -> tuple[pd.DataFrame, dict, dict, dict]:
    ids = [1, 2, 3, 4]
    q = np.tile(np.linspace(0, 1, 20), (4, 1))  # q(t) = (t-1)/19 for every image
    stops = pd.DataFrame(
        [(i, "late", 15) for i in ids] + [(i, "early", 5) for i in ids],
        columns=["image_id", "method", "stop_t"],
    )
    return (
        stops,
        {"R50": q, "relsg_gt": q},
        {i: k for k, i in enumerate(ids)},
        dict.fromkeys(ids, 14),
    )


def test_per_image_values() -> None:
    stops, q, index, oracle = _setup()
    vals = per_image(stops, "late", [1, 2], q, index, oracle)
    assert list(vals["abs_err_c09"]) == [1.0, 1.0]
    assert vals["R50"][0] == pytest.approx(14 / 19)


def test_pairwise_rows_directions() -> None:
    stops, q, index, oracle = _setup()
    rows = pairwise_rows(stops, ["late"], {"early": "early"}, {"all": [1, 2, 3, 4]}, q, index,
                         oracle, repeats=200)  # fmt: skip
    by_col = rows.set_index("column")
    assert by_col.loc["abs_err_c09", "direction"] == "better"  # |15-14| < |5-14|
    assert by_col.loc["R50", "direction"] == "better" and by_col.loc["R50", "diff"] > 0
    assert len(rows) == 3
