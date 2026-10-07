"""Tests for the recovery-plan §5.2 two-stage policy (plan §9 items 7 and 8).

Run:  uv run --project egtr --with pytest python -m pytest tests/test_two_stage_policy.py -q
"""

import inspect
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1 import two_stage_policy as tsp  # noqa: E402

IDS = ["v1", "v2", "v3", "v4"]


def _objs(*ratings):
    return [{"id": f"v{i + 1}", "clear": c, "note": ""} for i, c in enumerate(ratings)]


def test_u_for_0_1_2_absent_is_0375() -> None:
    rec = tsp.count_ratings(1, IDS, _objs(0, 1, 2, "absent"))
    assert rec.U == pytest.approx(0.375)
    assert (rec.n0, rec.n1, rec.n2, rec.n_absent, rec.n_malformed) == (1, 1, 1, 1, 0)
    assert rec.valid


def test_all_absent_gives_u0_and_reports_absent_rate() -> None:
    rec = tsp.count_ratings(1, IDS, _objs("absent", "absent", "absent", "absent"))
    assert rec.U == 0.0 and rec.absent_rate == 1.0 and rec.valid


def test_all_missing_is_failure_and_continues_to_t20() -> None:
    for objects in (None, []):
        rec = tsp.count_ratings(7, IDS, objects)
        assert not rec.valid and rec.U is None and rec.n_malformed == len(IDS)
        assert tsp.decide(rec, tau=0.5, p_tie=0.0) == (20, "failure_continue")
    no_boxes = tsp.count_ratings(7, [], [])
    assert not no_boxes.valid
    row = tsp.apply_frozen([no_boxes], {"tau": 0.5, "p_tie": 0.0})[0]
    assert row["decision_t"] == 20 and row["extra_cost_failure"] is True


def test_malformed_answers_are_excluded_and_counted_with_ceil_half_rule() -> None:
    objects = [
        {"id": "v1", "clear": 0},
        {"id": "v2", "clear": 3},  # invalid rating
        {"id": "v2", "clear": 1},  # first valid answer for v2 still counts
        {"id": "v9", "clear": 2},  # not targeted
        {"id": "v1", "clear": 2},  # duplicate -> ignored
        {"id": "v3", "clear": True},  # bool is not a rating
    ]
    rec = tsp.count_ratings(1, IDS, objects)
    assert (rec.n0, rec.n1, rec.n_malformed) == (1, 1, 2)
    assert rec.valid and rec.U == pytest.approx(0.75)  # 2 valid >= ceil(4/2)
    five = ["v1", "v2", "v3", "v4", "v5"]
    assert not tsp.count_ratings(1, five, _objs(0, 1)).valid  # 2 < ceil(5/2)=3
    assert tsp.count_ratings(1, five, _objs(0, 1, 2)).valid


def test_fit_threshold_median_and_p_tie() -> None:
    fit = tsp.fit_threshold([0.1, 0.25, 0.25, 0.25, 0.6, 0.8])
    assert fit["tau"] == 0.25 and fit["n_above"] == 2 and fit["n_tie"] == 3
    assert fit["p_tie"] == pytest.approx((0.5 * 6 - 2) / 3)
    no_tie = tsp.fit_threshold([0.1, 0.2, 0.3, 0.4])
    assert no_tie["tau"] == pytest.approx(0.25) and no_tie["p_tie"] == 0.0
    clipped = tsp.fit_threshold([0.5, 0.5, 0.9, 0.9, 0.9])
    assert 0.0 <= clipped["p_tie"] <= 1.0


def test_decision_rule_and_tie_breaking_by_hash() -> None:
    above = tsp.count_ratings(11, IDS, _objs(0, 0, 0, 2))
    below = tsp.count_ratings(12, IDS, _objs(2, 2, 2, 0))
    tie = tsp.count_ratings(13, IDS, _objs(0, 2, 0, 2))
    assert tsp.decide(above, 0.5, 0.0)[0] == 20
    assert tsp.decide(below, 0.5, 1.0)[0] == 10
    assert tsp.decide(tie, 0.5, 1.0) == (20, "tie_continue")
    assert tsp.decide(tie, 0.5, 0.0) == (10, "tie_stop")
    h = tsp.hash_unit(13)
    assert 0.0 <= h < 1.0 and h == tsp.hash_unit("13")
    expected = tsp.decide(tie, 0.5, h + 1e-9)
    assert expected[0] == 20


def test_policy_is_gt_independent_and_reproducible() -> None:
    for fn in (
        tsp.count_ratings,
        tsp.fit_threshold,
        tsp.decide,
        tsp.apply_frozen,
        tsp.hash_unit,
    ):
        params = " ".join(inspect.signature(fn).parameters).lower()
        assert "gt" not in params and "r50" not in params and "metric" not in params
    # decision path only (evaluation reads GT by design, after decisions exist)
    decision_fns = (
        tsp.count_ratings, tsp.fit_threshold, tsp.decide, tsp.apply_frozen,
        tsp.hash_unit, tsp.load_state, tsp.freeze_dev, tsp.apply_to_run,
    )  # fmt: skip
    for fn in decision_fns:
        source = inspect.getsource(fn)
        assert "stage_metrics" not in source and "R50" not in source
        assert "relsg" not in source
    recs = [tsp.count_ratings(i, IDS, _objs(i % 3, 1, 2, "absent")) for i in range(6)]
    frozen = tsp.fit_threshold([r.U for r in recs])
    first = tsp.apply_frozen(recs, frozen)
    again = tsp.apply_frozen(recs, json.loads(json.dumps(frozen)))
    assert first == again


# --- evaluation (gate criterion 3) ------------------------------------------

DEC_FIELDS = ["image_id", "decision_t", "extra_cost_failure"]


def _write_decisions(path: Path, decisions: dict[int, int]) -> Path:
    lines = [",".join(DEC_FIELDS)]
    lines += [f"{i},{t},False" for i, t in decisions.items()]
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_metrics(path: Path, q: dict[int, tuple[float | None, float | None]]) -> Path:
    lines = ["image_id,t,R50,relsg_gt,valid"]
    for i, (q10, q20) in q.items():
        for t, v in ((10, q10), (20, q20)):
            if v is None:
                lines.append(f"{i},{t},,,0")
            else:
                lines.append(f"{i},{t},{v},{v},1")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_evaluate_policy_that_picks_all_gainers_beats_mix(tmp_path: Path) -> None:
    q = {i: (0.2, 0.2 + (0.3 if i < 5 else 0.0)) for i in range(10)}
    dec = {i: 20 if i < 5 else 10 for i in range(10)}
    out = tsp.evaluate_two_stage(
        _write_decisions(tmp_path / "d.csv", dec),
        _write_metrics(tmp_path / "m.csv", q),
        "R50",
        seed=1,
    )
    assert out["p_continue"] == 0.5 and out["mean_level"] == 15
    assert out["policy_q"] == pytest.approx(0.35)
    assert out["mix_q"] == pytest.approx(0.2 + 0.5 * 0.15)
    assert out["policy_minus_mix"] > 0
    assert out["oracle_q_gt_posterior"] == pytest.approx(out["policy_q"])
    assert out["b_half_gt_posterior"] == pytest.approx(out["mix_q"])


def test_evaluate_random_decisions_has_near_zero_diff(tmp_path: Path) -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    n = 4000
    q10 = rng.uniform(0, 0.5, n)
    g = rng.normal(0.05, 0.1, n)
    dec = {i: int(20 if rng.random() < 0.5 else 10) for i in range(n)}
    q = {i: (float(q10[i]), float(q10[i] + g[i])) for i in range(n)}
    out = tsp.evaluate_two_stage(
        _write_decisions(tmp_path / "d.csv", dec),
        _write_metrics(tmp_path / "m.csv", q),
        "R50",
    )
    assert abs(out["policy_minus_mix"]) < 0.005


def test_evaluate_excludes_missing_q_and_rejects_duplicates(tmp_path: Path) -> None:
    q = {1: (0.1, 0.2), 2: (0.1, None), 3: (0.3, 0.3)}
    out = tsp.evaluate_two_stage(
        _write_decisions(tmp_path / "d.csv", {1: 20, 2: 20, 3: 10}),
        _write_metrics(tmp_path / "m.csv", q),
        "R50",
    )
    assert out["n_evaluated"] == 2 and out["n_excluded_missing_q"] == 1
    dup = tmp_path / "dup.csv"
    dup.write_text("image_id,t,R50,relsg_gt,valid\n1,10,0.1,0.1,1\n1,10,0.2,0.2,1\n")
    with pytest.raises(ValueError):
        tsp.read_q_pairs(dup, "R50")


def test_gate3_requires_positive_diff_on_both_seeds() -> None:
    rows = [
        {"seed": "1", "metric": "R50", "policy_q": 0.3, "policy_minus_mix": 0.01},
        {"seed": "2", "metric": "R50", "policy_q": 0.3, "policy_minus_mix": -0.001},
        {"seed": "2", "metric": "relsg_gt", "policy_q": 0.3, "policy_minus_mix": 0.2},
    ]
    assert not tsp.gate3(rows)
    rows[1]["policy_minus_mix"] = 0.002
    assert tsp.gate3(rows)


def test_read_image_ids_formats(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text('[1, 2, {"image_id": 3}]')
    (tmp_path / "b.txt").write_text("4\n5,test\n")
    assert tsp.read_image_ids(tmp_path / "a.json") == [1, 2, 3]
    assert tsp.read_image_ids(tmp_path / "b.txt") == [4, 5]
    (tmp_path / "sel.json").write_text('{"seed": 1, "selected_ids": [9, 7]}')
    assert tsp.read_image_ids(tmp_path / "sel.json") == [9, 7]
