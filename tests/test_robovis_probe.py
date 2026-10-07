"""Tests for the view-sufficiency probe diagnostic (scripts/41, robovis_probe).

Run: uv run --project egtr --with pytest python -m pytest tests/test_robovis_probe.py -q

The probe fixture is generated from the synthetic R50 curves: detail_fraction =
floor(100 * q_t / qmax), sufficient = q_t / qmax >= 0.9, next_step_helps = not
sufficient, with a few failed / missing / malformed records injected.
"""

import copy
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.robovis_data import level_name, load_robovis_config  # noqa: E402
from milestone1.robovis_probe import (  # noqa: E402
    GATE_STATUS,
    OUTCOME_FAILURE,
    OUTCOME_INVALID,
    OUTCOME_MISSING,
    OUTCOME_OK,
    PROBE_STATUS,
    REFERENCE_STATUS,
    compare_row,
    count_inversions,
    gated_c_sequence,
    parse_probe,
    run_probe_diag,
)
from milestone1.robovis_replay import RunLayout  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "robovis_synthetic"
RATIO = 0.9
PERCENT = 100
FAST_REPEATS = 100
# (image_id, level) -> injected record
FAILED = (1001, 5)
MISSING = (1002, 3)
INVALID = (1003, 4)
RESET = (1004, 19)  # failure inside the stopping window -> forced terminal


@pytest.fixture
def fast_cfg() -> dict[str, Any]:
    cfg = copy.deepcopy(load_robovis_config())
    cfg["evaluation"]["bootstrap_repeats"] = FAST_REPEATS
    return cfg


def _write_probe(run_dir: Path, cfg_name: str) -> None:
    stage = pd.read_csv(run_dir / "stage_metrics.csv")
    qmax = stage.groupby("image_id")["R50"].max()
    for r in stage.to_dict("records"):
        key = (int(r["image_id"]), int(r["t"]))
        if key == MISSING:
            continue
        ratio = r["R50"] / qmax[r["image_id"]] if qmax[r["image_id"]] > 0 else 0.0
        rec: dict[str, Any] = {
            "detail_fraction": math.floor(PERCENT * ratio + 1e-9),
            "sufficient": bool(ratio >= RATIO - 1e-12),
            "next_step_helps": bool(ratio < RATIO - 1e-12),
            "reason": "synthetic",
            "failure": False,
        }
        if key in (FAILED, RESET):
            rec = {"failure": True, "error": "synthetic"}
        if key == INVALID:
            rec["detail_fraction"] = 150
        path = run_dir / "probe" / cfg_name / str(key[0]) / f"{level_name(key[1])}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec))


@pytest.fixture
def probe_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    shutil.copytree(FIXTURE, run)
    _write_probe(run, "toy")
    return run


# --------------------------------------------------------------------------
# Unit tests
# --------------------------------------------------------------------------

VALID = {
    "detail_fraction": 55,
    "sufficient": False,
    "next_step_helps": True,
    "reason": "r",
}


@pytest.mark.parametrize(
    ("obj", "outcome"),
    [
        (VALID, OUTCOME_OK),
        ({**VALID, "detail_fraction": 0.0}, OUTCOME_OK),
        (None, OUTCOME_MISSING),
        ({**VALID, "failure": True}, OUTCOME_FAILURE),
        ({**VALID, "detail_fraction": 101}, OUTCOME_INVALID),
        ({**VALID, "detail_fraction": True}, OUTCOME_INVALID),
        ({**VALID, "sufficient": "yes"}, OUTCOME_INVALID),
        ({k: v for k, v in VALID.items() if k != "reason"}, OUTCOME_INVALID),
        ("corrupt", OUTCOME_INVALID),
        ({"model": "m", "failure": False, "probe": VALID}, OUTCOME_OK),
        ({"model": "m", "failure": True, "probe": None}, OUTCOME_FAILURE),
        ({"model": "m", "failure": False, "probe": None}, OUTCOME_INVALID),
    ],
)
def test_parse_probe(obj: Any, outcome: str) -> None:
    rec, got = parse_probe(obj)
    assert got == outcome
    assert (rec is not None) == (outcome == OUTCOME_OK)


def test_count_inversions() -> None:
    assert count_inversions([10, 20, 15, 30]) == 1
    assert count_inversions([30, 20, 10]) == 3
    assert count_inversions([1, 1, 2]) == 0
    assert count_inversions([5, math.nan, 4]) == 1


# --------------------------------------------------------------------------
# End to end on the synthetic probe
# --------------------------------------------------------------------------


def test_probe_diag_end_to_end(probe_run: Path, fast_cfg: dict[str, Any]) -> None:
    out = probe_run / "eval" / "probe" / "toy"
    res = run_probe_diag(
        probe_run, probe_run / "probe" / "toy", out, fast_cfg, "toy", "R50"
    )
    for name in ("probe_states.csv", "probe_by_level.csv", "probe_stats.json",
                 "stops.csv", "summary.csv", "bootstrap.json"):  # fmt: skip
        assert (out / name).exists(), name

    st = res.stats
    assert st["outcomes"] == {OUTCOME_OK: 196, OUTCOME_FAILURE: 2, OUTCOME_MISSING: 1,
                              OUTCOME_INVALID: 1}  # fmt: skip
    # The probe is a monotone function of q_ratio -> perfect ranking.
    assert st["auc_detail"] == pytest.approx(1.0)
    assert st["auc_not_next"] == pytest.approx(1.0)
    assert st["rho_detail_qratio"]["rho"] > 0.9
    # Only 1006 (recall drops at the tail) has inversions.
    assert st["monotonicity"]["frac_monotone"] == pytest.approx(0.9)
    assert st["monotonicity"]["n_images"] == 10

    stops = pd.read_csv(out / "stops.csv").set_index(["image_id", "method"])
    # 1001: q/qmax >= .9 from t9 (.72/.78) -> second consecutive level t10.
    assert stops.loc[(1001, "probe_stop"), "stop_t"] == 10
    # 1006: t8 (.55/.6) and t9 -> t9, despite the later drop.
    assert stops.loc[(1006, "probe_stop"), "stop_t"] == 9
    # 1004: instant at t18, failed t19 resets, t20 alone -> forced terminal.
    row = stops.loc[(1004, "probe_stop")]
    assert (row["stop_t"], bool(row["forced_terminal"]), row["failure_count"]) == (
        20,
        True,
        1,
    )
    # next_step_helps = not sufficient in this fixture -> identical policy.
    assert stops.loc[(1001, "probe_stop_next"), "stop_t"] == 10

    summary = pd.read_csv(out / "summary.csv").set_index("method")
    assert summary.loc["probe_stop", "status"] == PROBE_STATUS
    assert summary.loc["fixed_20", "status"] == "baseline"
    assert summary.loc["probe_stop", "n_images"] == 9  # 1007 excluded (invalid level)
    boot = json.loads((out / "bootstrap.json").read_text())
    assert "probe_stop_minus_fixed_20" in boot["pairwise"]
    assert boot["policy_status"]["probe_stop"] == PROBE_STATUS

    row = compare_row(res)
    assert row["cfg"] == "toy" and row["q_metric"] == "R50"
    assert row["failure_rate"] == pytest.approx(4 / 200)
    assert {"probe_stop_mean_stop", "probe_stop_diff_ci_low", "auc_detail"} <= set(row)


def test_probe_diag_relsg_metric_goes_to_subdir(
    probe_run: Path, fast_cfg: dict[str, Any]
) -> None:
    out = probe_run / "eval" / "probe" / "toy"
    res = run_probe_diag(
        probe_run, probe_run / "probe" / "toy", out, fast_cfg, "toy", "relsg_gt"
    )
    assert (out / "q_relsg_gt" / "summary.csv").exists()
    assert res.stats["q_metric"] == "relsg_gt"
    # relsg_gt = sqrt(R50) in the fixture: still monotone in the probe's R50 ratio.
    assert res.stats["rho_detail_qratio"]["rho"] > 0.9


# --------------------------------------------------------------------------
# Probe gate + judge C (probe_gate_proposed / probe_gate_observed_only)
# --------------------------------------------------------------------------


def test_probe_gate_policies(probe_run: Path, fast_cfg: dict[str, Any]) -> None:
    out = probe_run / "eval" / "probe" / "toy"
    res = run_probe_diag(
        probe_run, probe_run / "probe" / "toy", out, fast_cfg, "toy", "R50", RunLayout()
    )
    stops = pd.read_csv(out / "stops.csv").set_index(["image_id", "method"])

    def stop(image_id: int, method: str) -> int:
        return int(stops.loc[(image_id, method), "stop_t"])

    # 1001: judge alone stops at t9 (C=1 at t8,t9); the probe says "not
    # sufficient" until t8 (.70/.78 < .9) -> gated: t9 (C=1), t10 (C=0) -> t10.
    assert (stop(1001, "proposed"), stop(1001, "probe_gate_proposed")) == (9, 10)
    # 1010: observed_only alone stops at t2 (A=0 from t1); gated waits until the
    # probe is sufficient at t11 -> t11, t12 -> t12. proposed is unchanged (t13).
    assert (stop(1010, "observed_only"), stop(1010, "probe_gate_observed_only")) == (
        2,
        12,
    )
    assert (stop(1010, "proposed"), stop(1010, "probe_gate_proposed")) == (13, 13)
    # 1004: failed probe at t19 is a missing C_t -> reset -> forced terminal.
    row = stops.loc[(1004, "probe_gate_proposed")]
    assert (row["stop_t"], bool(row["forced_terminal"]), row["failure_count"]) == (
        20,
        True,
        1,
    )
    assert not bool(stops.loc[(1004, "proposed"), "forced_terminal"])

    summary = pd.read_csv(out / "summary.csv").set_index("method")
    assert summary.loc["probe_gate_proposed", "status"] == GATE_STATUS
    assert summary.loc["proposed", "status"] == REFERENCE_STATUS
    boot = json.loads((out / "bootstrap.json").read_text())
    assert "probe_gate_proposed_minus_proposed" in boot["pairwise"]
    assert "probe_gate_proposed_minus_fixed_20" in boot["pairwise"]
    assert "probe_gate_proposed_minus_probe_gate_observed_only" in boot["pairwise"]
    assert res.stats["gate_decisions"] == "decisions"

    row = compare_row(res)
    for col in ("probe_gate_proposed_mean_stop", "probe_gate_proposed_diff_ci_low",
                "probe_gate_proposed_minus_proposed", "probe_gate_proposed_minus_fixed_20",
                "proposed_mean_stop", "gate_decisions"):  # fmt: skip
        assert col in row, col


def test_gated_c_sequence_rules() -> None:
    states = pd.DataFrame({
        "image_id": [1, 1, 1], "t": [1, 2, 3],
        "outcome": [OUTCOME_OK, OUTCOME_OK, OUTCOME_FAILURE],
        "sufficient": [0.0, 1.0, math.nan],
    })  # fmt: skip
    assert gated_c_sequence(states, 1, [0, None, 0], max_u=2) == [2, None, None]
    assert gated_c_sequence(states, 1, [1, 1, 1], max_u=2) == [2, 1, None]
