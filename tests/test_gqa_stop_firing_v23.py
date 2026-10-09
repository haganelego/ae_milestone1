"""Plan §23 firing-rate helpers on a hand-made table."""

from __future__ import annotations

import pandas as pd

from milestone1.gqa_stop import firing_v23 as f


def table() -> pd.DataFrame:
    rows = []
    scores = {"a": [0.1, 0.2, 0.6], "b": [0.1, 0.1, 0.2], "c": [0.5, 0.1, 0.1]}
    for q, values in scores.items():
        for level, s in enumerate(values, start=1):
            rows.append(
                {
                    "question_id": q,
                    "group_id": f"g{q}",
                    "level": level,
                    "reader_correct": level == 1,
                    "s": s,
                    "t": 0.0,
                }
            )
    return pd.DataFrame(rows)


THETAS = {
    "0.95": {
        "checklist_v4_min": {"column": "s", "k": 1, "role": "main", "theta": 0.5},
        "direct_v4": {"column": "t", "k": 1, "role": "comparator", "theta": 0.5},
        "k2": {"column": "s", "k": 2, "role": "logged", "theta": 0.15},
    }
}


def test_firing_counts_met_levels_and_not_forced_stop() -> None:
    fires = f.firing_frame(table(), THETAS, 3)
    main = fires[fires["method"] == "checklist_v4_min"].set_index("question_id")
    assert main.loc["a", "fired"] and main.loc["a", "fire_level"] == 3
    assert not main.loc["b", "fired"]  # only the forced last-level stop
    assert main.loc["c", "fired"] and main.loc["c", "reader_correct_at_fire"]
    k2 = fires[fires["method"] == "k2"].set_index("question_id")
    assert k2.loc["a", "fired"] and not k2.loc["c", "fired"]


def test_summary_and_paired_difference() -> None:
    fires = f.firing_frame(table(), THETAS, 3)
    summary = f.summarize(fires).set_index("method")
    assert summary.loc["checklist_v4_min", "firing_rate"] == 2 / 3
    assert summary.loc["direct_v4", "n_fired"] == 0
    pairs = f.paired(fires, {"a": "ga", "b": "gb", "c": "gc"}, 50, 1)
    row = pairs[pairs["comparison"] == "checklist_v4_min - direct_v4"].iloc[0]
    assert row["firing_rate_diff"] == 2 / 3 and row["only_a_fired"] == 2


def test_frozen_thetas_reads_inf_and_skips_infeasible(tmp_path) -> None:
    path = tmp_path / "t.csv"
    pd.DataFrame(
        [
            {
                "subset": "v22",
                "target": "0.99",
                "method": "direct_v1_verbalized",
                "role": "c",
                "k": 1,
                "feasible": True,
                "theta": "inf",
            },
            {
                "subset": "v22",
                "target": "0.99",
                "method": "pans_min",
                "role": "logged",
                "k": 1,
                "feasible": False,
                "theta": None,
            },
            {
                "subset": "v4c_59",
                "target": "0.99",
                "method": "direct_v4",
                "role": "c",
                "k": 1,
                "feasible": True,
                "theta": "0.5",
            },
        ]
    ).to_csv(path, index=False)
    thetas = f.frozen_thetas(path)
    assert thetas == {
        "0.99": {
            "direct_v1_verbalized": {
                "column": "direct_score",
                "k": 1,
                "role": "c",
                "theta": float("inf"),
            }
        }
    }
