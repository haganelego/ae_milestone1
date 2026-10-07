"""ROBOVIS2027: summarize checks + decisions of one run/variant, grouped by level.

Used to judge whether the §6 stop gate is degenerate (e.g. always / never firing).

Reads   outputs/robovis2027/<run_id>/{checks,decisions}/<variant>/<image_id>/level_XX.json
        outputs/robovis2027/<run_id>/graphs/<image_id>/level_XX.json (anchor check)
Writes  outputs/robovis2027/<run_id>/summaries/checks_summary_<variant>.json

§6 gate per decision item: status == "unresolved" AND limitation ==
"low_resolution" AND the check's anchors exist in the graph AND its
region_anchor_id is a non-null entity of the graph. "strong" additionally
requires min(basis, resolution_benefit) >= proposal.theta.

Run:
  uv run --project egtr python scripts/29b_robovis_checks_summary.py \
      --run-id dev_r0 --variant F2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.robovis_data import load_robovis_config  # noqa: E402

LEVEL_FILE_RE = re.compile(r"^level_(\d+)\.json$")
GATE_STATUS = "unresolved"
GATE_LIMITATION = "low_resolution"
ALL_KEY = "all"
ROUND_DIGITS = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/robovis2027.yaml")
    p.add_argument("--run-id", required=True)
    p.add_argument("--variant", required=True, help="F1, F2, F2__9b, ...")
    p.add_argument(
        "--graphs-subdir",
        default="graphs",
        help="graph dir used for the anchor check (graphs_clean for variants *_G)",
    )
    p.add_argument(
        "--checks-variant",
        default=None,
        help="checks dir if different from --variant (judge-only runs, e.g. F2)",
    )
    return p.parse_args()


def iter_records(root: Path) -> list[tuple[int, int, dict[str, Any]]]:
    out: list[tuple[int, int, dict[str, Any]]] = []
    if not root.is_dir():
        return out
    for image_dir in sorted(
        d for d in root.iterdir() if d.is_dir() and d.name.isdigit()
    ):
        for path in sorted(image_dir.glob("level_*.json")):
            match = LEVEL_FILE_RE.match(path.name)
            if match is None:
                continue
            with open(path) as f:
                out.append((int(image_dir.name), int(match.group(1)), json.load(f)))
    return out


def ratio(num: float, den: float) -> float | None:
    return None if den == 0 else round(num / den, ROUND_DIGITS)


def distribution(counter: Counter) -> dict[str, Any]:
    total = sum(counter.values())
    return {
        "counts": {str(k): v for k, v in sorted(counter.items(), key=str)},
        "fractions": {
            str(k): ratio(v, total) for k, v in sorted(counter.items(), key=str)
        },
    }


def summarize_checks(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r["checks"] for r in records if not r.get("failure") and r.get("checks")]
    n_obs = [len(c["observed_issues"]) for c in valid]
    n_hyp = [len(c["hypotheses"]) for c in valid]
    items = [i for c in valid for i in c["observed_issues"] + c["hypotheses"]]
    kinds = Counter(h["kind"] for c in valid for h in c["hypotheses"])
    return {
        "n_states": len(records),
        "n_failures": sum(1 for r in records if r.get("failure")),
        "n_valid": len(valid),
        "mean_observed_issues": ratio(sum(n_obs), len(valid)),
        "mean_hypotheses": ratio(sum(n_hyp), len(valid)),
        "frac_empty_observed_issues": ratio(sum(n == 0 for n in n_obs), len(valid)),
        "frac_empty_hypotheses": ratio(sum(n == 0 for n in n_hyp), len(valid)),
        "frac_both_empty": ratio(
            sum(a == 0 and b == 0 for a, b in zip(n_obs, n_hyp, strict=True)),
            len(valid),
        ),
        "kind": distribution(kinds),
        "region_anchor_null_rate": ratio(
            sum(i["region_anchor_id"] is None for i in items), len(items)
        ),
    }


def item_anchors_ok(check_item: dict[str, Any] | None, entity_ids: set[str]) -> bool:
    """Anchors exist in the graph and the region anchor is a non-null graph entity."""
    if check_item is None:
        return False
    anchors = check_item.get("anchor_ids") or []
    region = check_item.get("region_anchor_id")
    return bool(anchors) and set(anchors) <= entity_ids and region in entity_ids


def summarize_decisions(
    rows: list[tuple[dict[str, Any], dict[str, Any] | None, set[str]]], theta: int
) -> dict[str, Any]:
    """rows: (decision record, matching checks record, graph entity ids)."""
    status, limitation, basis, benefit = Counter(), Counter(), Counter(), Counter()
    n_items = n_gate = n_strong = n_min_theta = 0
    states_gate = states_strong = n_judged = 0
    for decision, checks_rec, entity_ids in rows:
        if decision.get("failure") or decision.get("items") is None:
            continue
        n_judged += 1
        checks = (checks_rec or {}).get("checks") or {}
        by_id = {
            i["id"]: i
            for i in checks.get("observed_issues", []) + checks.get("hypotheses", [])
        }
        any_gate = any_strong = False
        for item in decision["items"]:
            n_items += 1
            status[item["status"]] += 1
            limitation[item["limitation"]] += 1
            basis[item["basis"]] += 1
            benefit[item["resolution_benefit"]] += 1
            min_ok = min(item["basis"], item["resolution_benefit"]) >= theta
            gate = (
                item["status"] == GATE_STATUS
                and item["limitation"] == GATE_LIMITATION
                and item_anchors_ok(by_id.get(item["id"]), entity_ids)
            )
            n_min_theta += min_ok
            n_gate += gate
            n_strong += gate and min_ok
            any_gate |= gate
            any_strong |= gate and min_ok
        states_gate += any_gate
        states_strong += any_strong
    return {
        "n_states": len(rows),
        "n_failures": sum(1 for d, _, _ in rows if d.get("failure")),
        "n_skipped_empty": sum(1 for d, _, _ in rows if d.get("skipped_empty")),
        "n_items": n_items,
        "status": distribution(status),
        "limitation": distribution(limitation),
        "basis": distribution(basis),
        "resolution_benefit": distribution(benefit),
        "frac_items_gate": ratio(n_gate, n_items),
        "frac_items_min_basis_benefit_ge_theta": ratio(n_min_theta, n_items),
        "frac_items_gate_and_min_ge_theta": ratio(n_strong, n_items),
        "frac_states_any_gate": ratio(states_gate, n_judged),
        "frac_states_any_gate_and_min_ge_theta": ratio(states_strong, n_judged),
    }


def load_entity_ids(graphs_root: Path, image_id: int, level: int) -> set[str]:
    path = graphs_root / str(image_id) / f"level_{level:02d}.json"
    if not path.exists():
        return set()
    with open(path) as f:
        return {e["id"] for e in json.load(f).get("entities", [])}


def main() -> None:
    args = parse_args()
    cfg = load_robovis_config(resolve(args.config))
    theta = int(cfg["proposal"]["theta"])
    run_root = resolve(cfg["output_root"]) / args.run_id
    checks_variant = args.checks_variant or args.variant
    checks = iter_records(run_root / "checks" / checks_variant)
    decisions = iter_records(run_root / "decisions" / args.variant)
    if not checks:
        sys.exit(f"no checks under {run_root / 'checks' / checks_variant}")
    checks_by_key = {(i, lv): r for i, lv, r in checks}

    levels = sorted({lv for _, lv, _ in checks})
    groups: dict[str, Any] = {}
    for key in [ALL_KEY, *levels]:
        sel_checks = [r for _, lv, r in checks if key == ALL_KEY or lv == key]
        sel_dec = [
            (
                d,
                checks_by_key.get((i, lv)),
                load_entity_ids(run_root / args.graphs_subdir, i, lv),
            )
            for i, lv, d in decisions
            if key == ALL_KEY or lv == key
        ]
        groups[str(key)] = {
            "checks": summarize_checks(sel_checks),
            "decisions": summarize_decisions(sel_dec, theta) if sel_dec else None,
        }
    summary = {
        "run_id": args.run_id,
        "variant": args.variant,
        "checks_variant": checks_variant,
        "graphs_subdir": args.graphs_subdir,
        "theta": theta,
        "gate": f"status={GATE_STATUS} & limitation={GATE_LIMITATION} & anchors+region valid",
        "by_level": groups,
    }
    out_path = run_root / "summaries" / f"checks_summary_{args.variant}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{args.run_id}/{args.variant} (theta={theta}) -> {out_path}")
    header = (
        f"{'level':>5} {'n':>4} {'fail':>4} {'obs':>5} {'hyp':>5} {'empty':>6} "
        f"{'rnull':>6} {'unres':>6} {'lowres':>6} {'gate':>6} {'g&min':>6} {'st_g&m':>6}"
    )
    print(header)
    for key, g in groups.items():
        c, d = g["checks"], g["decisions"] or {}
        unres = (d.get("status") or {}).get("fractions", {}).get(GATE_STATUS)
        lowres = (d.get("limitation") or {}).get("fractions", {}).get(GATE_LIMITATION)
        cells = [
            c["mean_observed_issues"],
            c["mean_hypotheses"],
            c["frac_both_empty"],
            c["region_anchor_null_rate"],
            unres,
            lowres,
            d.get("frac_items_gate"),
            d.get("frac_items_gate_and_min_ge_theta"),
            d.get("frac_states_any_gate_and_min_ge_theta"),
        ]
        txt = " ".join("     -" if v is None else f"{v:6.2f}" for v in cells)
        print(f"{key:>5} {c['n_states']:>4} {c['n_failures']:>4} {txt}")


if __name__ == "__main__":
    main()
