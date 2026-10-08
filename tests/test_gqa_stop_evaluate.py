"""Normalisation and fixed references by hand (test 13), image-cluster
bootstrap (test 12), non-inferiority verdict."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from milestone1.gqa_stop.evaluate import (
    adjacent_mix,
    build_eval_data,
    cluster_bootstrap_indices,
    evaluate,
    hull_value,
    is_correct,
    noninferiority_verdict,
    normalize_answer,
    point_estimates,
    upper_concave_hull,
)

N = 20


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Light   Blue ", "light blue"),
        ("ＲＥＤ", "red"),
        ("Wood.", "wood."),
        ("dark\tbrown\n", "dark brown"),
    ],
)
def test_normalisation(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


def test_exact_match_only() -> None:
    assert is_correct(" Blue", "blue")
    assert not is_correct("blue", "light blue")
    assert not is_correct("wooden", "wood")
    assert not is_correct(None, "red")


def test_adjacent_mix_by_hand() -> None:
    curve = {5: 0.6, 6: 0.8}
    assert adjacent_mix(curve, 5.25) == pytest.approx(0.65)
    assert adjacent_mix(curve, 6.0) == 0.8


def test_concave_hull_by_hand() -> None:
    curve = {1: 0.2, 2: 0.3, 3: 0.9, 4: 0.92, 5: 1.0}
    hull = upper_concave_hull(list(curve), list(curve.values()))
    assert [p[0] for p in hull] == [1, 3, 5]  # (2, 0.3) lies under the hull
    assert hull_value(curve, 2.0) == pytest.approx(0.55)
    assert hull_value(curve, 4.0) == pytest.approx(0.95)
    assert hull_value(curve, 1.0) == pytest.approx(0.2)


def test_bootstrap_draws_whole_images() -> None:
    groups = ["a", "a", "a", "b", "c", "c"]
    for idx in cluster_bootstrap_indices(groups, 50, 7):
        drawn = [groups[i] for i in idx]
        counts = pd.Series(drawn).value_counts()
        assert counts.get("a", 0) % 3 == 0
        assert counts.get("c", 0) % 2 == 0
        for g in set(drawn):
            members = [i for i, x in enumerate(groups) if x == g]
            assert all(
                drawn.count(g) // len(members) == list(idx).count(m) for m in members
            )


def test_bootstrap_is_seeded() -> None:
    a = cluster_bootstrap_indices(list("aabbc"), 5, 1)
    b = cluster_bootstrap_indices(list("aabbc"), 5, 1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_verdicts() -> None:
    assert noninferiority_verdict(
        {"ci_low": -0.03}, {"ci_high": -1.0}, 0.05
    ).startswith("supported")
    assert noninferiority_verdict(
        {"ci_low": -0.08}, {"ci_high": -1.0}, 0.05
    ).startswith("trade_off")
    assert noninferiority_verdict(
        {"ci_low": 0.0}, {"ci_low": 1.0, "ci_high": 2.0}, 0.05
    ).startswith("not_supported")
    assert noninferiority_verdict(
        {"ci_low": 0.0}, {"ci_low": -1.0, "ci_high": 1.0}, 0.05
    ).startswith("unclear")


def table() -> pd.DataFrame:
    rows = []
    for q in range(6):
        for t in range(1, N + 1):
            rows.append(
                {
                    "question_id": f"q{q}",
                    "image_id": f"i{q // 2}",
                    "group_id": f"i{q // 2}",
                    "level": t,
                    "reader_correct": t >= 2 + 2 * q,
                    "reader_answer": "x",
                    "attribute": "color" if q % 2 else "material",
                    "relate_count": q % 2,
                    "direct_score": 0.9 if t >= 3 else 0.0,
                    "checklist_min": 0.9 if t >= 2 + 2 * q else 0.1,
                    "checklist_mean": 0.9 if t >= 1 + q else 0.2,
                    "reader_status": "valid",
                    "direct_status": "valid",
                    "direct_answer": "x" if t >= 3 else "y",
                    "check_status": "valid",
                    "planner_status": "valid",
                    "n_items": 2,
                    "unresolved_items": 0 if t >= 2 + 2 * q else 1,
                }
            )
    return pd.DataFrame(rows)


FROZEN = {
    "primary_target": "0.95",
    "methods": ["direct", "checklist_min", "checklist_mean"],
    "selections": {
        "0.95": {
            "direct": {"theta": 0.5},
            "checklist_min": {"theta": 0.5},
            "checklist_mean": {"theta": "inf"},
            "fixed_dev": 12,
        }
    },
}


def test_frozen_policy_application_by_hand() -> None:
    data = build_eval_data(table(), FROZEN, N)
    assert list(data.stops["direct@0.95"]) == [3] * 6
    assert list(data.stops["checklist_min@0.95"]) == [2, 4, 6, 8, 10, 12]
    assert list(data.stops["checklist_mean@0.95"]) == [20] * 6
    point = point_estimates(data)
    assert point["direct@0.95.accuracy"] == pytest.approx(1 / 6)
    assert point["checklist_min@0.95.accuracy"] == 1.0
    assert point["checklist_min@0.95-direct@0.95.mean_level_diff"] == pytest.approx(
        7 - 3
    )
    curve = {t: sum(t >= 2 + 2 * q for q in range(6)) / 6 for t in range(1, N + 1)}
    assert point["fixed_dev@0.95.accuracy"] == pytest.approx(curve[12])
    assert point["direct@0.95.hull_gap"] == pytest.approx(
        1 / 6 - hull_value(curve, 3.0)
    )


def test_evaluate_writes_outputs(tmp_path) -> None:
    gt = {f"q{q}": "x" for q in range(6)}
    summary = evaluate(
        table(),
        FROZEN,
        {
            "bootstrap_repeats": 30,
            "bootstrap_seed": 1,
            "accuracy_noninferiority_margin": 0.05,
        },
        N,
        tmp_path,
        gt,
    )
    for name in (
        "stops.csv",
        "method_summary.csv",
        "paired_comparisons.csv",
        "bootstrap_summary.json",
        "diagnostics.json",
    ):
        assert (tmp_path / name).exists()
    stats = summary["statistics"]["checklist_min@0.95-direct@0.95.accuracy_diff"]
    assert stats["ci_low"] <= stats["estimate"] <= stats["ci_high"]
    stops = pd.read_csv(tmp_path / "stops.csv")
    assert set(stops.groupby("method")["question_id"].count()) == {6}
