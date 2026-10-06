"""Generate the ROBOVIS2027 synthetic replay fixture (10 images x 20 levels).

Usage:
    uv run --project egtr python tests/fixtures/make_robovis_synthetic.py

Writes ``tests/fixtures/robovis_synthetic/`` (deterministic; safe to re-run):
stage_metrics.csv, stage_predicate_hits.csv, graphs/, checks/, decisions/,
expected_stops.json. Checks files of odd image ids use the check_generator
record layout (payload nested under "checks"); even ids use the bare contract.

Level codes (one per level):
    N      non-empty graph, no checks                      -> C = 0
    aXhY   observed issue u=X, hypothesis u=Y (low_res)    -> A = X, H = Y
    E      empty observed graph (no decision file at all)  -> A = 2
    M      decision file missing                           -> missing
    F      decision with "failure": true                   -> missing
    B      checks anchor not in graph (invalid checks)     -> missing
    J      judgment basis = 3 (invalid judgments)          -> missing
    U      unresolved item, limitation unknown, basis 2    -> 0 (uc variant: 1)
    O      unresolved item, limitation out_of_frame        -> 0
"""

import csv
import json
import shutil
from pathlib import Path
from typing import Any

FIXTURE_DIR = Path(__file__).resolve().parent / "robovis_synthetic"
NUM_LEVELS = 20
SPLIT = "dev"
SCALES = [
    0.05, 0.0585, 0.0685, 0.0802, 0.0940, 0.1100, 0.1288, 0.1508, 0.1765, 0.2066,
    0.2419, 0.2832, 0.3316, 0.3882, 0.4545, 0.5321, 0.6229, 0.7293, 0.8538, 1.0,
]  # fmt: skip
PRED_GT = {0: 4, 1: 3, 2: 3}  # pred_id -> n_gt per image
N_GT_REL = sum(PRED_GT.values())
R20_FACTOR = 0.8
R100_FACTOR = 1.1


def _ramp(peak_t: int, peak: float, start: float = 0.0) -> list[float]:
    """Linear rise from ``start`` at t=1 to ``peak`` at peak_t, flat afterwards."""
    out = []
    for t in range(1, NUM_LEVELS + 1):
        frac = min(1.0, (t - 1) / max(1, peak_t - 1))
        out.append(round(start + (peak - start) * frac, 4))
    return out


def _codes(spec: list[tuple[str, int]]) -> list[str]:
    seq = [code for code, n in spec for _ in range(n)]
    assert len(seq) == NUM_LEVELS, (spec, len(seq))
    return seq


# image_id -> (q curve, level codes, invalid levels, expected stops, note)
IMAGES: dict[int, dict[str, Any]] = {
    1001: {
        "note": "saturates ~t9; hypothesis keeps proposed one level longer",
        "q": [0, 0, .05, .1, .2, .35, .5, .7, .72, .75,
              .75, .76, .76, .77, .77, .78, .78, .78, .78, .78],
        "codes": _codes([("E", 2), ("a2h2", 4), ("a1h2", 1), ("a1h1", 2), ("N", 11)]),
        "expected": {"proposed": 9, "observed_only": 8, "unknown_conservative": 11},
        "oracle": 9,
    },
    1002: {
        "note": "missing + failed decisions reset the counter",
        "q": _ramp(10, 0.6),
        "codes": _codes([("a2h2", 4), ("a1h1", 1), ("M", 1), ("a1h1", 1), ("F", 1),
                         ("a0h1", 1), ("a0h0", 1), ("N", 10)]),
        "expected": {"proposed": 10, "observed_only": 10, "unknown_conservative": 11},
        "failures": {"proposed": 2},
        "oracle": 10,
    },
    1003: {
        "note": "never satisfies -> forced terminal",
        "q": _ramp(20, 0.8),
        "codes": _codes([("a2h2", 20)]),
        "expected": {"proposed": 20, "observed_only": 20, "unknown_conservative": 20},
        "forced": True,
        "oracle": 19,
    },
    1004: {
        "note": "only t19,t20 instant-stop -> natural stop at t20",
        "q": _ramp(19, 0.5),
        "codes": _codes([("a2h2", 18), ("a1h1", 2)]),
        "expected": {"proposed": 20, "observed_only": 20, "unknown_conservative": 20},
        "forced": False,
        "oracle": 18,
    },
    1005: {
        "note": "empty graph everywhere, zero recall -> zero_recovery, forced",
        "q": [0.0] * NUM_LEVELS,
        "codes": _codes([("E", 20)]),
        "expected": {"proposed": 20, "observed_only": 20, "unknown_conservative": 20},
        "forced": True,
        "oracle": None,
    },
    1006: {
        "note": "recall drops at the tail -> unstable_tail; out_of_frame u=0",
        "q": [.05, .1, .2, .3, .4, .45, .5, .55, .58, .6,
              .6, .55, .5, .5, .45, .45, .4, .4, .4, .4],
        "codes": _codes([("a2h2", 5), ("O", 1), ("N", 14)]),
        "expected": {"proposed": 7, "observed_only": 7, "unknown_conservative": 7},
        "oracle": None,
    },
    1007: {
        "note": "level 12 invalid -> excluded from eval set",
        "q": _ramp(8, 0.4),
        "codes": _codes([("a2h2", 3), ("N", 17)]),
        "invalid_levels": [12],
        "expected": {"proposed": 5, "observed_only": 5, "unknown_conservative": 5},
        "oracle": None,
    },
    1008: {
        "note": "invalid checks (bad anchor) and invalid judgments (basis=3)",
        "q": _ramp(6, 0.7),
        "codes": _codes([("a2h2", 3), ("B", 1), ("N", 1), ("J", 1), ("N", 14)]),
        "expected": {"proposed": 8, "observed_only": 8, "unknown_conservative": 8},
        "failures": {"proposed": 2},
        "oracle": 6,
    },
    1009: {
        "note": "unknown limitation: proposed stops, unknown_conservative forced",
        "q": _ramp(5, 0.9),
        "codes": _codes([("a2h2", 3), ("U", 17)]),
        "expected": {"proposed": 5, "observed_only": 5, "unknown_conservative": 20},
        "oracle": 5,
    },
    1010: {
        "note": "only hypotheses are informative -> observed_only stops at t2",
        "q": _ramp(12, 0.65, start=0.1),
        "codes": _codes([("a0h2", 11), ("N", 9)]),
        "expected": {"proposed": 13, "observed_only": 2, "unknown_conservative": 13},
        "oracle": 11,
    },
}  # fmt: skip


# --------------------------------------------------------------------------
# Level payloads
# --------------------------------------------------------------------------


def _graph(image_id: int, t: int, empty: bool) -> dict[str, Any]:
    entities = (
        []
        if empty
        else [
            {"id": "v1", "label_id": 77, "label": "man", "score": 0.8,
             "box_xyxy_orig": [10.0, 10.0, 100.0, 200.0]},
            {"id": "v2", "label_id": 20, "label": "shirt", "score": 0.6,
             "box_xyxy_orig": [20.0, 40.0, 90.0, 120.0]},
            {"id": "v3", "label_id": 11, "label": "table", "score": 0.4,
             "box_xyxy_orig": [150.0, 100.0, 300.0, 250.0]},
        ]
    )  # fmt: skip
    relations = (
        []
        if empty
        else [
            {"id": "e1", "sub": "v1", "obj": "v2", "pred_id": 31, "pred": "wearing",
             "pred_score": 0.7, "trp_score": 0.4, "rank": 0},
        ]
    )  # fmt: skip
    return {
        "image_id": image_id,
        "split": SPLIT,
        "level": t,
        "scale": SCALES[t - 1],
        "entities": entities,
        "relations": relations,
        "stats": {"synthetic": True},
    }


def _observed(cid: str, anchor: str = "v1") -> dict[str, Any]:
    return {
        "id": cid,
        "anchor_ids": [anchor],
        "target": "label of the man",
        "region_anchor_id": anchor,
        "evidence": "box is a few pixels wide",
    }


def _hypothesis(cid: str) -> dict[str, Any]:
    return {
        "id": cid,
        "kind": "missing_relation",
        "candidate": "man holding cup",
        "anchor_ids": ["v1", "v3"],
        "relation_ids": ["e1"],
        "region_anchor_id": "v3",
        "evidence": "hand near table",
    }


def _judgment(
    cid: str, u: int, limitation: str = "low_resolution", basis: int | None = None
) -> dict[str, Any]:
    if u == 0 and basis is None:
        return {"id": cid, "status": "resolved", "basis": 2, "resolution_benefit": 0,
                "limitation": limitation, "reason": "clear enough"}  # fmt: skip
    return {
        "id": cid,
        "status": "unresolved",
        "basis": u if basis is None else basis,
        "resolution_benefit": 2,
        "limitation": limitation,
        "reason": "needs more pixels",
    }


def level_payloads(
    code: str,
) -> tuple[bool, dict[str, Any] | None, dict[str, Any] | None]:
    """(graph_empty, checks, decision) for one level code; None = no file."""
    empty_checks = {"observed_issues": [], "hypotheses": []}
    ok = {"failure": False}
    if code == "N":
        return False, empty_checks, {"items": [], **ok}
    if code == "E":
        return True, None, None
    if code.startswith("a"):
        x, y = int(code[1]), int(code[3])
        checks = {
            "observed_issues": [_observed("o1")],
            "hypotheses": [_hypothesis("h1")],
        }
        decision = {"items": [_judgment("o1", x), _judgment("h1", y)], **ok}
        return False, checks, decision
    one_issue = {"observed_issues": [_observed("o1")], "hypotheses": []}
    if code == "M":
        return False, one_issue, None
    if code == "F":
        return False, one_issue, {"items": [], "failure": True}
    if code == "B":
        bad = {"observed_issues": [_observed("o1", anchor="v99")], "hypotheses": []}
        return False, bad, {"items": [_judgment("o1", 2)], **ok}
    if code == "J":
        return False, one_issue, {"items": [_judgment("o1", 2, basis=3)], **ok}
    if code == "U":
        return False, one_issue, {"items": [_judgment("o1", 2, "unknown")], **ok}
    if code == "O":
        return False, one_issue, {"items": [_judgment("o1", 2, "out_of_frame")], **ok}
    raise ValueError(f"unknown level code {code!r}")


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------


def _checks_file(image_id: int, checks: dict[str, Any]) -> dict[str, Any]:
    """Odd ids use the check_generator record layout (payload under "checks")."""
    if image_id % 2 == 0:
        return checks
    return {"model": "synthetic", "failure": False, "error": None, "checks": checks}


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")


def main() -> None:
    if FIXTURE_DIR.exists():
        shutil.rmtree(FIXTURE_DIR)
    FIXTURE_DIR.mkdir(parents=True)

    stage_rows: list[dict[str, Any]] = []
    hit_rows: list[dict[str, Any]] = []
    expected: dict[str, Any] = {}
    for image_id, spec in IMAGES.items():
        invalid = set(spec.get("invalid_levels", []))
        for t, (q, code) in enumerate(zip(spec["q"], spec["codes"], strict=True), 1):
            stage_rows.append(
                {
                    "image_id": image_id,
                    "split": SPLIT,
                    "t": t,
                    "scale": SCALES[t - 1],
                    "R20": round(q * R20_FACTOR, 4),
                    "R50": q,
                    "R100": round(min(1.0, q * R100_FACTOR), 4),
                    "n_gt_rel": N_GT_REL,
                    "valid": t not in invalid,
                }
            )
            for pred_id, n_gt in PRED_GT.items():
                hits = min(n_gt, int(q * n_gt + 0.5))
                hit_rows.append(
                    {"image_id": image_id, "t": t, "pred_id": pred_id,
                     "n_gt": n_gt, "hits50": hits}
                )  # fmt: skip
            empty, checks, decision = level_payloads(code)
            name = f"level_{t:02d}.json"
            _write_json(FIXTURE_DIR / "graphs" / str(image_id) / name,
                        _graph(image_id, t, empty))  # fmt: skip
            if checks is not None:
                _write_json(
                    FIXTURE_DIR / "checks" / str(image_id) / name,
                    _checks_file(image_id, checks),
                )
            if decision is not None:
                _write_json(FIXTURE_DIR / "decisions" / str(image_id) / name, decision)
        expected[str(image_id)] = {
            "note": spec["note"],
            "codes": spec["codes"],
            "stops": spec["expected"],
            "forced": spec.get("forced"),
            "failures": spec.get("failures", {}),
            "oracle": spec["oracle"],
            "in_eval_set": not invalid,
        }

    for fname, rows in (
        ("stage_metrics.csv", stage_rows),
        ("stage_predicate_hits.csv", hit_rows),
    ):
        with open(FIXTURE_DIR / fname, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    _write_json(FIXTURE_DIR / "expected_stops.json", expected)
    print(f"wrote {len(IMAGES)} images x {NUM_LEVELS} levels -> {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
