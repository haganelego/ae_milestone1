"""ROBOVIS2027 recovery plan §5.2: the single v4-based two-stage policy (t10 / t20).

Per image, from the cached v4 ratings at t10 (decisions/F1g_v4_G__9b/<id>/level_10.json):

    U = (n0 + 0.5 * n1) / (n0 + n1 + n2 + n_absent)

- ``absent`` counts in the denominator, not as unclear.
- Malformed box answers (unknown id, invalid rating, duplicate, missing) are
  excluded from the denominator and counted in ``n_malformed``.
- The state is a failure if the targeted box set is empty or the number of valid
  answers is below ceil(n_targeted / 2); a failure continues to t20 and the extra
  cost is flagged (never counted as sufficiency).

tau = median of valid dev U. p_tie = clip((0.5*n - n(U>tau)) / n(U=tau), 0, 1) if
ties exist, else 0. Rule: U > tau -> t20, U < tau -> t10, U == tau -> t20 iff
SHA256("recovery-v1:" + image_id) mapped to [0, 1) < p_tie. No GT is read here.

CLI (dev freeze; refuses to overwrite an existing frozen policy):
  PYTHONPATH=src uv run --project egtr python -m milestone1.two_stage_policy \
      --run-id dev_r0 --split dev --out-run rec_r0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from milestone1.check_generator import write_json_atomic
from milestone1.config import resolve
from milestone1.grounded_judge import (
    CLEAR_ABSENT,
    STRING_HINT_V4,
    V4_FORMAT_NOTE,
    V4_PROMPT,
    top_entities,
)
from milestone1.robovis_data import load_robovis_config

POLICY_VERSION = "recovery-v1"
HASH_SALT = "recovery-v1:"
OBSERVE_LEVEL = 10  # pre-registered observation point (plan §5.2)
CONTINUE_LEVEL = 20  # pre-registered continuation point
HALF = 0.5
UNCLEAR_WEIGHTS = {0: 1.0, 1: 0.5, 2: 0.0, CLEAR_ABSENT: 0.0}
TIE_ABS_TOL = 1e-12  # U values are small-denominator rationals
HASH_BITS = 256
V4_VARIANT = "F1g_v4_G__9b"
CSV_FIELDS = (
    "image_id",
    "n_targeted",
    "n0",
    "n1",
    "n2",
    "n_absent",
    "n_malformed",
    "U",
    "absent_rate",
    "valid",
    "decision_t",
    "reason",
    "extra_cost_failure",
)


@dataclass(frozen=True)
class URecord:
    image_id: int
    n_targeted: int
    n0: int
    n1: int
    n2: int
    n_absent: int
    n_malformed: int
    U: float | None
    absent_rate: float | None
    valid: bool


def count_ratings(
    image_id: int,
    targeted_ids: Sequence[str],
    v4_objects: Iterable[Any] | None,
) -> URecord:
    """Count valid 0/1/2/absent answers for the targeted boxes; compute U."""
    targeted = list(dict.fromkeys(targeted_ids))
    counts = {0: 0, 1: 0, 2: 0, CLEAR_ABSENT: 0}
    seen: set[str] = set()
    for obj in v4_objects or []:
        if not isinstance(obj, dict):
            continue
        obj_id, rating = obj.get("id"), obj.get("clear")
        if obj_id not in targeted or obj_id in seen:
            continue
        if isinstance(rating, bool) or rating not in counts:
            continue
        seen.add(obj_id)
        counts[rating] += 1
    n_valid = sum(counts.values())
    n_targeted = len(targeted)
    valid = n_targeted > 0 and n_valid >= math.ceil(n_targeted / 2)
    u = None
    absent_rate = None
    if valid:
        u = sum(UNCLEAR_WEIGHTS[k] * v for k, v in counts.items()) / n_valid
        absent_rate = counts[CLEAR_ABSENT] / n_valid
    return URecord(
        image_id=image_id,
        n_targeted=n_targeted,
        n0=counts[0],
        n1=counts[1],
        n2=counts[2],
        n_absent=counts[CLEAR_ABSENT],
        n_malformed=n_targeted - n_valid,
        U=u,
        absent_rate=absent_rate,
        valid=valid,
    )


def hash_unit(image_id: int | str) -> float:
    """SHA256(salt + image_id) mapped to [0, 1); GT-independent tie breaker."""
    digest = hashlib.sha256(f"{HASH_SALT}{image_id}".encode()).hexdigest()
    return int(digest, 16) / (1 << HASH_BITS)


def fit_threshold(u_values: Sequence[float]) -> dict[str, Any]:
    """tau = median of valid U; p_tie keeps the expected continuation at half."""
    if not u_values:
        raise ValueError("no valid U values to fit tau")
    tau = statistics.median(u_values)
    n = len(u_values)
    n_above = sum(1 for u in u_values if u > tau and not _tie(u, tau))
    n_tie = sum(1 for u in u_values if _tie(u, tau))
    p_tie = 0.0
    if n_tie:
        p_tie = min(1.0, max(0.0, (HALF * n - n_above) / n_tie))
    return {
        "tau": tau,
        "p_tie": p_tie,
        "n_valid": n,
        "n_above": n_above,
        "n_tie": n_tie,
    }


def _tie(u: float, tau: float) -> bool:
    return math.isclose(u, tau, rel_tol=0.0, abs_tol=TIE_ABS_TOL)


def decide(record: URecord, tau: float, p_tie: float) -> tuple[int, str]:
    """(stop level, reason). Uses only U, frozen tau/p_tie and the image-id hash."""
    if not record.valid or record.U is None:
        return CONTINUE_LEVEL, "failure_continue"
    if _tie(record.U, tau):
        if hash_unit(record.image_id) < p_tie:
            return CONTINUE_LEVEL, "tie_continue"
        return OBSERVE_LEVEL, "tie_stop"
    if record.U > tau:
        return CONTINUE_LEVEL, "above_tau"
    return OBSERVE_LEVEL, "below_tau"


def apply_frozen(
    records: Sequence[URecord], frozen: dict[str, Any]
) -> list[dict[str, Any]]:
    """Stops for any split from a frozen policy (no refit, no GT)."""
    tau, p_tie = float(frozen["tau"]), float(frozen["p_tie"])
    rows = []
    for rec in records:
        level, reason = decide(rec, tau, p_tie)
        rows.append(
            {
                **asdict(rec),
                "decision_t": level,
                "reason": reason,
                "extra_cost_failure": not rec.valid,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Cache I/O (v4 decisions at t10; cleaned graph only to recover the box targets)
# ---------------------------------------------------------------------------


def load_state(
    run_root: Path, image_id: int, k: int, graphs_subdir: str, variant: str
) -> tuple[URecord, dict[str, Any] | None]:
    name = f"level_{OBSERVE_LEVEL:02d}.json"
    graph_path = run_root / graphs_subdir / str(image_id) / name
    targeted: list[str] = []
    if graph_path.exists():
        with open(graph_path) as f:
            targeted = [e["id"] for e in top_entities(json.load(f), k)]
    dec_path = run_root / "decisions" / variant / str(image_id) / name
    decision = None
    if dec_path.exists():
        with open(dec_path) as f:
            decision = json.load(f)
    objects = None if decision is None else decision.get("v4_objects")
    return count_ratings(image_id, targeted, objects), decision


def v4_template_sha256() -> str:
    return hashlib.sha256(
        f"{V4_PROMPT} {V4_FORMAT_NOTE} {STRING_HINT_V4}".encode()
    ).hexdigest()


def freeze_dev(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_robovis_config(resolve(args.config))
    run_root = resolve(cfg["output_root"]) / args.run_id
    out_dir = resolve(args.out_root) / args.out_run
    frozen_path = out_dir / "frozen_policy.json"
    if frozen_path.exists() and not args.force:
        sys.exit(f"{frozen_path} exists; refusing to overwrite (use a new --out-run)")
    with open(resolve(cfg["paths"]["selected_images"])) as f:
        ids = sorted(
            int(r["image_id"]) for r in json.load(f) if r["split"] == args.split
        )
    k = int(cfg["checks"]["v4"]["top_entities"])
    records, decisions = [], []
    for image_id in ids:
        rec, dec = load_state(run_root, image_id, k, args.graphs_subdir, args.variant)
        records.append(rec)
        decisions.append(dec)
    valid_u = [r.U for r in records if r.valid and r.U is not None]
    fit = fit_threshold(valid_u)
    models = sorted({d.get("model") for d in decisions if d})
    num_ctx = sorted(
        {
            ((d.get("options") or {}).get("options") or {}).get("num_ctx")
            for d in decisions
            if d
        },
        key=str,
    )
    with open(resolve(args.config), "rb") as f:
        config_sha = hashlib.sha256(f.read()).hexdigest()
    frozen = {
        "policy_version": POLICY_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tau": fit["tau"],
        "p_tie": fit["p_tie"],
        "fit": fit,
        "u_formula": "(n0 + 0.5*n1) / (n0 + n1 + n2 + n_absent)",
        "failure_rule": "n_targeted == 0 or n_valid < ceil(n_targeted/2) -> t20",
        "tie_rule": f'SHA256("{HASH_SALT}" + image_id) / 2^{HASH_BITS} < p_tie -> t20',
        "observe_level": OBSERVE_LEVEL,
        "continue_level": CONTINUE_LEVEL,
        "gt_used": False,
        "source": {
            "run_id": args.run_id,
            "split": args.split,
            "image_ids": ids,
            "variant": args.variant,
            "graphs_subdir": args.graphs_subdir,
            "top_entities": k,
            "models": models,
            "num_ctx": num_ctx,
            "v4_prompt_template_sha256": v4_template_sha256(),
            "config_sha256": config_sha,
        },
        "dev_counts": {
            "n_images": len(records),
            "n_valid": sum(r.valid for r in records),
            "n_failure": sum(not r.valid for r in records),
            "gate1_valid_ge_18_of_20": sum(r.valid for r in records)
            >= args.gate_min_valid,
        },
    }
    rows = apply_frozen(records, frozen)
    frozen["dev_counts"]["n_continue_t20"] = sum(
        r["decision_t"] == CONTINUE_LEVEL for r in rows
    )
    frozen["dev_counts"]["n_extra_cost_failure"] = sum(
        r["extra_cost_failure"] for r in rows
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "dev_two_stage.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json_atomic(frozen_path, frozen)
    return {"frozen": frozen, "rows": rows, "out_dir": str(out_dir)}


# ---------------------------------------------------------------------------
# Apply a frozen policy to new images (Phase D; no GT, no refit)
# ---------------------------------------------------------------------------


def read_image_ids(path: str | Path) -> list[int]:
    """JSON list (ints or {"image_id": ..}) or one id per line."""
    text = Path(path).read_text().strip()
    if text.startswith("["):
        items = json.loads(text)
        return [int(x["image_id"] if isinstance(x, dict) else x) for x in items]
    return [int(line.split(",")[0]) for line in text.splitlines() if line.strip()]


def apply_to_run(args: argparse.Namespace) -> list[dict[str, Any]]:
    """test_policy.csv from cached v4 t10 outputs + frozen tau/p_tie (no GT)."""
    cfg = load_robovis_config(resolve(args.config))
    with open(resolve(args.frozen)) as f:
        frozen = json.load(f)
    out_path = resolve(args.out_csv)
    if out_path.exists() and not args.force:
        sys.exit(f"{out_path} exists; refusing to overwrite")
    run_root = resolve(cfg["output_root"]) / args.run_id
    k = int(frozen["source"]["top_entities"])
    ids = read_image_ids(resolve(args.image_ids_file))
    records = [
        load_state(run_root, i, k, args.graphs_subdir, args.variant)[0] for i in ids
    ]
    rows = apply_frozen(records, frozen)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# Evaluation against one seed's stage metrics (gate criterion 3; GT is read
# ONLY here, after the decisions are fixed)
# ---------------------------------------------------------------------------

EVAL_METRICS = ("R50", "relsg_gt")
TRUE_STRINGS = {"1", "true", "yes"}
EVAL_FIELDS = (
    "seed",
    "metric",
    "n_decisions",
    "n_evaluated",
    "n_excluded_missing_q",
    "n_t20",
    "p_continue",
    "mean_level",
    "policy_q",
    "mix_q",
    "policy_minus_mix",
    "n_failure_continued",
    "oracle_k",
    "oracle_q_gt_posterior",
    "b_half_gt_posterior",
    "oracle_minus_b_half_gt_posterior",
)


def _is_valid(value: str) -> bool:
    return str(value).strip().lower() in TRUE_STRINGS


def read_q_pairs(
    stage_metrics_csv: str | Path, metric: str
) -> dict[int, tuple[float, float]]:
    """{image_id: (q10, q20)} for images with BOTH levels valid (never zero-filled).

    A duplicate (image_id, t) row raises (mixed seeds / datasets).
    """
    seen: set[tuple[int, int]] = set()
    table: dict[tuple[int, int], float] = {}
    with open(stage_metrics_csv, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["image_id"]), int(row["t"]))
            if key in seen:
                raise ValueError(f"duplicate (image_id, t) row {key} -> mixed inputs?")
            seen.add(key)
            value = row.get(metric, "")
            if not _is_valid(row.get("valid", "")) or value in ("", "nan", "NaN"):
                continue
            table[key] = float(value)
    ids = sorted({i for i, _ in table})
    return {
        i: (table[(i, OBSERVE_LEVEL)], table[(i, CONTINUE_LEVEL)])
        for i in ids
        if (i, OBSERVE_LEVEL) in table and (i, CONTINUE_LEVEL) in table
    }


def evaluate_two_stage(
    decisions_csv: str | Path,
    stage_metrics_csv: str | Path,
    metric: str,
    seed: int | str = "",
) -> dict[str, Any]:
    """Policy mean q vs the image-independent mix at the realized rate p.

    p = n(t20)/n over evaluated images; mix = mean(q10 + p*(q20 - q10)). The
    t10/t20 oracle (top-k gain, k = n//2) and B_half = mean(q10 + (k/n) g) are
    GT-posterior references only (recovery_diagnostics.two_choice_oracle).
    """
    import numpy as np

    from milestone1.recovery_diagnostics import mixture_value, two_choice_oracle

    with open(decisions_csv, newline="") as f:
        decisions = list(csv.DictReader(f))
    pairs = read_q_pairs(stage_metrics_csv, metric)
    rows = [d for d in decisions if int(d["image_id"]) in pairs]
    n = len(rows)
    out: dict[str, Any] = {
        "seed": seed,
        "metric": metric,
        "n_decisions": len(decisions),
        "n_evaluated": n,
        "n_excluded_missing_q": len(decisions) - n,
    }
    if n == 0:
        return out
    q_lo = np.array([pairs[int(d["image_id"])][0] for d in rows])
    q_hi = np.array([pairs[int(d["image_id"])][1] for d in rows])
    cont = np.array([int(d["decision_t"]) == CONTINUE_LEVEL for d in rows])
    p = float(cont.mean())
    policy_q = float(np.where(cont, q_hi, q_lo).mean())
    mix_q = mixture_value(q_lo, q_hi, p)
    oracle = two_choice_oracle(q_lo, q_hi)
    oracle_q, b_half = float(oracle.value), float(oracle.b_half)
    out.update(
        {
            "n_t20": int(cont.sum()),
            "p_continue": p,
            "mean_level": float(np.where(cont, CONTINUE_LEVEL, OBSERVE_LEVEL).mean()),
            "policy_q": policy_q,
            "mix_q": mix_q,
            "policy_minus_mix": policy_q - mix_q,
            "n_failure_continued": sum(
                str(d.get("extra_cost_failure", "")).strip().lower() == "true"
                for d in rows
            ),
            "oracle_k": n // 2,
            "oracle_q_gt_posterior": oracle_q,
            "b_half_gt_posterior": b_half,
            "oracle_minus_b_half_gt_posterior": oracle_q - b_half,
        }
    )
    return out


def evaluate_seeds(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_dir = resolve(args.out_root) / args.out_run
    decisions_csv = out_dir / "dev_two_stage.csv"
    rows = []
    for seed in [s.strip() for s in args.seeds.split(",") if s.strip()]:
        metrics_csv = resolve(args.stage_metrics_pattern.format(seed=seed))
        if not metrics_csv.exists():
            print(f"  seed {seed}: {metrics_csv} missing; skipped")
            continue
        for metric in args.metrics.split(","):
            rows.append(evaluate_two_stage(decisions_csv, metrics_csv, metric, seed))
    out_path = out_dir / "dev_two_stage_eval.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {out_path}")
    return rows


def gate3(rows: Sequence[dict[str, Any]], seeds: Sequence[str] = ("1", "2")) -> bool:
    """§5.3 criterion 3: policy > mix on BOTH seed1 and seed2 for R50."""
    by_seed = {
        str(r["seed"]): r for r in rows if r["metric"] == "R50" and "policy_q" in r
    }
    return all(s in by_seed and by_seed[s]["policy_minus_mix"] > 0 for s in seeds)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recovery §5.2 two-stage policy")
    p.add_argument("--mode", choices=("freeze", "apply", "evaluate"), default="freeze")
    p.add_argument("--config", default="configs/robovis2027.yaml")
    p.add_argument("--run-id", default="dev_r0")
    p.add_argument("--split", default="dev")
    p.add_argument("--variant", default=V4_VARIANT)
    p.add_argument("--graphs-subdir", default="graphs_clean")
    p.add_argument("--out-root", default="outputs/robovis2027_recovery")
    p.add_argument("--out-run", default="rec_r0")
    p.add_argument("--gate-min-valid", type=int, default=18, help="plan §5.3 cond. 1")
    p.add_argument("--force", action="store_true")
    # apply
    p.add_argument(
        "--frozen", default="outputs/robovis2027_recovery/rec_r0/frozen_policy.json"
    )
    p.add_argument("--image-ids-file", default=None)
    p.add_argument("--out-csv", default=None)
    # evaluate
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument(
        "--stage-metrics-pattern",
        default="outputs/robovis2027_recovery/rec_r0/stage_metrics_seed{seed}.csv",
    )
    p.add_argument("--metrics", default=",".join(EVAL_METRICS))
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.mode == "apply":
        if not args.image_ids_file or not args.out_csv:
            sys.exit("--mode apply needs --image-ids-file and --out-csv")
        rows = apply_to_run(args)
        n20 = sum(r["decision_t"] == CONTINUE_LEVEL for r in rows)
        n_fail = sum(r["extra_cost_failure"] for r in rows)
        print(f"-> {args.out_csv}: {len(rows)} images, t20={n20}, failures={n_fail}")
        return
    if args.mode == "evaluate":
        rows = evaluate_seeds(args)
        for r in rows:
            if "policy_q" not in r:
                print(f"  seed {r['seed']} {r['metric']}: no evaluable images")
                continue
            print(
                f"  seed {r['seed']} {r['metric']:8} n={r['n_evaluated']} "
                f"p={r['p_continue']:.2f} policy={r['policy_q']:.4f} "
                f"mix={r['mix_q']:.4f} diff={r['policy_minus_mix']:+.4f} | "
                f"oracle={r['oracle_q_gt_posterior']:.4f} "
                f"B_half={r['b_half_gt_posterior']:.4f} (GT-posterior)"
            )
        print(f"gate criterion 3 (R50, seeds 1 & 2 policy > mix): {gate3(rows)}")
        return
    result = freeze_dev(args)
    frozen, rows = result["frozen"], result["rows"]
    print(f"-> {result['out_dir']}")
    print(
        json.dumps(
            {k: frozen[k] for k in ("tau", "p_tie", "fit", "dev_counts")}, indent=1
        )
    )
    for r in rows:
        u = "-" if r["U"] is None else f"{r['U']:.3f}"
        print(
            f"  {r['image_id']} U={u} n0/1/2/abs={r['n0']}/{r['n1']}/{r['n2']}/"
            f"{r['n_absent']} mal={r['n_malformed']} -> t{r['decision_t']} ({r['reason']})"
        )


if __name__ == "__main__":
    main()
