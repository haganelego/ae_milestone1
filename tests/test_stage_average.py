"""Tests for milestone1.stage_average (per-state mean over PGSG seeds)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milestone1.stage_average import average_stage_tables  # noqa: E402


def _table(
    r50: list[float], valid: list[int], relsg: list[float] | None = None
) -> pd.DataFrame:
    n = len(r50)
    return pd.DataFrame(
        {
            "image_id": [7] * n,
            "split": ["test"] * n,
            "t": list(range(1, n + 1)),
            "scale": [0.05, 1.0][:n],
            "R50": [v if ok else np.nan for v, ok in zip(r50, valid, strict=True)],
            "n_gt_rel": [4] * n,
            "valid": valid,
            "relsg_gt": relsg if relsg is not None else [0.1] * n,
        }
    )


def test_average_is_per_state_mean_with_n_seeds() -> None:
    out = average_stage_tables(
        [_table([0.0, 0.5], [1, 1]), _table([0.25, 0.25], [1, 1], [0.3, 0.3])]
    )
    assert out["R50"].tolist() == pytest.approx([0.125, 0.375])
    assert out["relsg_gt"].tolist() == pytest.approx([0.2, 0.2])
    assert out["n_seeds"].tolist() == [2, 2]
    assert out["valid"].tolist() == [1, 1]
    assert out["n_gt_rel"].tolist() == [4, 4] and out["scale"].tolist() == [0.05, 1.0]


def test_invalid_seed_rows_are_excluded_and_mark_state_invalid() -> None:
    out = average_stage_tables([_table([0.2, 0.4], [1, 1]), _table([0.9, 0.6], [0, 1])])
    assert out.loc[0, "R50"] == pytest.approx(0.2)  # seed 2 invalid at t=1
    assert out.loc[0, "n_seeds"] == 1 and out.loc[0, "valid"] == 0
    assert out.loc[1, "R50"] == pytest.approx(0.5) and out.loc[1, "valid"] == 1


def test_single_table_round_trips_and_empty_input_raises() -> None:
    t = _table([0.1, 0.2], [1, 1])
    out = average_stage_tables([t])
    assert out["R50"].tolist() == pytest.approx([0.1, 0.2]) and out[
        "n_seeds"
    ].tolist() == [1, 1]
    with pytest.raises(ValueError):
        average_stage_tables([])
