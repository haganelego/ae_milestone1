"""Plan §22 dev grid helpers on a hand-made table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from milestone1.gqa_stop import devgrid_v22 as g

LEVELS = 20
COLUMNS = [m[1] for m in g.ALL_METHODS]


def make_table(n: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        onset = int(rng.integers(1, LEVELS + 1))
        for level in range(1, LEVELS + 1):
            score = min(1.0, level / LEVELS + 0.05 * (i % 3))
            rows.append(
                {
                    "question_id": f"q{i:02d}",
                    "group_id": f"g{i // 2}",
                    "level": level,
                    "attribute": ["color", "material"][i % 2],
                    "relate_count": i % 2,
                    "reader_correct": level >= onset,
                    **dict.fromkeys(COLUMNS, score),
                }
            )
    return pd.DataFrame(rows)


def test_selections_meet_target_and_include_fixed() -> None:
    table = make_table()
    grid = [x / 100 for x in range(1, 101)] + [float("inf")]
    sel, chosen = g.selections(table, grid, [0.9, 0.99], "s")
    feasible = sel[(sel["method"] == "checklist_v4_min") & sel["feasible"]]
    assert (feasible["accuracy"] >= 0.9 - 1e-9).all()
    assert set(sel["target"]) == {"0.90", "0.99"}
    assert "fixed_dev" in set(sel["method"])
    assert "checklist_v4_min" in chosen["0.90"]


def test_curves_cover_grid_and_fixed_levels() -> None:
    table = make_table()
    grid = [0.2, 0.5, 0.9]
    frame = g.curves(table, grid, "s")
    assert len(frame[frame["method"] == "direct_v4"]) == len(grid)
    assert len(frame[frame["method"] == "fixed"]) == LEVELS


def test_paired_identical_methods_give_zero_difference() -> None:
    table = make_table()
    chosen = {
        "0.95": {
            "checklist_v4_min": ("checklist_v4_min", 0.5, 1),
            "direct_v4": ("direct_v4", 0.5, 1),
        }
    }
    frame = g.paired(table, chosen, 50, 1, "s")
    row = frame.iloc[0]
    assert row["mean_level_diff"] == 0.0 and row["mean_level_ci_low"] == 0.0
    assert row["n_images"] == 6


def test_breakdown_counts_sum_to_questions() -> None:
    table = make_table()
    chosen = {"0.95": {"checklist_v4_min": ("checklist_v4_min", 0.5, 1)}}
    frame = g.breakdown(table, chosen)
    attr = frame[frame["group_by"] == "attribute"]
    assert attr["n"].sum() == 12
