"""Recovery plan Phase D: evaluate the frozen two-stage U policy (zero model calls).

Usage (real run, inputs under outputs/robovis2027_recovery/<run_id>/):
    uv run --project egtr python scripts/47_recovery_phase_d_eval.py --run-id rec_r0
Dry run on dev (seed0 full curve as "seedA", seed1/seed2 as "seedB" stand-ins):
    uv run --project egtr python scripts/47_recovery_phase_d_eval.py --run-id rec_r0 --dry-run-dev

Writes (real run): test_summary.csv (seedA), seed_replication.csv (seedB, if present),
phase_d_summary.json, figures/phase_d_quality_vs_level.{png,pdf}.
SeedA and seedB are never combined; no hull is built on seedB; verdicts per plan §7.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.phase_d_eval import (  # noqa: E402
    DECISION_LEVELS,
    F_NOTE,
    FIXED_MID,
    PRIMARY_METRIC,
    SEED_B_LEVELS,
    combined_verdict,
    level_arrays,
    load_policy,
    seed_a_references,
    summarize_seed,
)
from milestone1.recovery_diagnostics import METRICS, upper_hull  # noqa: E402

OUT_ROOT = "outputs/robovis2027_recovery"
NUM_LEVELS = 20
REAL_INPUTS = {"policy": "test_policy.csv", "seedA": "stage_metrics_seedA.csv",
               "seedB": "stage_metrics_seedB.csv"}  # fmt: skip
DRY_INPUTS = {"policy": "dev_two_stage.csv",
              "seedA": "../../robovis2027/dev_r0/stage_metrics.csv",
              "seedB": ["stage_metrics_seed1.csv", "stage_metrics_seed2.csv"]}  # fmt: skip
DRY_SUBDIR = "phase_d_dryrun_dev"
FIG_SIZE = (8.4, 3.4)
DPI = 200
COLORS = {
    "curve": "0.55",
    "F": "0.25",
    "policy": "#1f4e79",
    "mix": "#c55a11",
    "fixed15": "#548235",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-id", default="rec_r0")
    p.add_argument("--dry-run-dev", action="store_true")
    p.add_argument("--repeats", type=int, default=2000)
    return p.parse_args()


def evaluate_seed_a(
    policy: pd.DataFrame, stage: pd.DataFrame, repeats: int
) -> list[dict[str, Any]]:
    rows = []
    for metric in METRICS:
        arrays, info = level_arrays(
            stage, metric, policy, (*DECISION_LEVELS, FIXED_MID), "seedA"
        )
        row = summarize_seed(arrays, info, metric, "A", repeats)
        refs = seed_a_references(stage, metric, row["mean_level"])
        row.update({k: v for k, v in refs.items() if k != "fixed_curve"})
        row["policy_minus_F"] = row["policy_q"] - refs["F_at_mean_level"]
        row["fixed_curve"] = ";".join(f"{v:.4f}" for v in refs["fixed_curve"])
        rows.append(row)
    return rows


def evaluate_seed_b(
    policy: pd.DataFrame, stage: pd.DataFrame, label: str, repeats: int
) -> list[dict[str, Any]]:
    extra = sorted(set(stage["t"]) - set(SEED_B_LEVELS))
    stage = stage[stage["t"].isin(SEED_B_LEVELS)]  # seedB uses t10/15/20 only (no hull)
    rows = []
    for metric in METRICS:
        arrays, info = level_arrays(
            stage, metric, policy, SEED_B_LEVELS, f"seed{label}"
        )
        row = summarize_seed(arrays, info, metric, label, repeats)
        row["ignored_levels"] = ";".join(map(str, extra))
        rows.append(row)
    return rows


def plot(stage_a: pd.DataFrame, a_rows: list[dict[str, Any]], out_stem: Path) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=FIG_SIZE)
    for ax, row in zip(axes, a_rows, strict=True):
        f = np.array([float(v) for v in row["fixed_curve"].split(";")])
        t = np.arange(1, NUM_LEVELS + 1)
        ax.plot(t, f, color=COLORS["curve"], marker=".", lw=1, label="fixed level t")
        hull = upper_hull(f)
        ax.plot(t[hull], f[hull], color=COLORS["F"], ls="--", lw=1,
                label="F (best fixed mixture; GT-posterior)")  # fmt: skip
        x = row["mean_level"]
        yerr = [[max(0.0, row["policy_minus_mix"] - row["ci_low"])],
                [max(0.0, row["ci_high"] - row["policy_minus_mix"])]]  # fmt: skip
        ax.errorbar([x], [row["policy_q"]], yerr=yerr, fmt="o", color=COLORS["policy"],
                    capsize=3, label="U policy (CI of policy - mix)")  # fmt: skip
        ax.scatter([x], [row["mix_q_same_p"]], marker="s", color=COLORS["mix"],
                   label="t10/t20 mix, same p", zorder=3)  # fmt: skip
        ax.scatter([FIXED_MID], [row["fixed15_q"]], marker="^", color=COLORS["fixed15"],
                   label="fixed15", zorder=3)  # fmt: skip
        ax.set_title(
            f"seedA {row['metric']}{' (primary)' if row['primary'] else ''}", fontsize=9
        )
        ax.set_xlabel("mean observed level")
        ax.set_ylabel(f"mean {row['metric']}")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=DPI)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = resolve(OUT_ROOT) / args.run_id
    inputs = DRY_INPUTS if args.dry_run_dev else REAL_INPUTS
    out_dir = run_dir / DRY_SUBDIR if args.dry_run_dev else run_dir
    policy_path = run_dir / inputs["policy"]
    seed_a_path = run_dir / inputs["seedA"]
    for path in (policy_path, seed_a_path):
        if not path.exists():
            raise SystemExit(f"waiting for input: {path}")
    policy = load_policy(pd.read_csv(policy_path))
    stage_a = pd.read_csv(seed_a_path)
    a_rows = evaluate_seed_a(policy, stage_a, args.repeats)
    seed_b_paths = (
        inputs["seedB"] if isinstance(inputs["seedB"], list) else [inputs["seedB"]]
    )
    b_rows: list[dict[str, Any]] = []
    for i, name in enumerate(seed_b_paths):
        path = run_dir / name
        if path.resolve() == seed_a_path.resolve():
            raise SystemExit("seedB file is the seedA file: refusing to mix seeds")
        if path.exists():
            b_rows += evaluate_seed_b(policy, pd.read_csv(path), "B" if len(seed_b_paths) == 1
                                      else f"B{i + 1}", args.repeats)  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(a_rows).to_csv(out_dir / "test_summary.csv", index=False)
    if b_rows:
        pd.DataFrame(b_rows).to_csv(out_dir / "seed_replication.csv", index=False)
    primary_a = next(r for r in a_rows if r["metric"] == PRIMARY_METRIC)
    primary_b = [r for r in b_rows if r["metric"] == PRIMARY_METRIC]
    summary = {
        "dry_run_dev": args.dry_run_dev,
        "inputs": {"policy": str(policy_path), "seedA": str(seed_a_path),
                   "seedB": [str(run_dir / n) for n in seed_b_paths if (run_dir / n).exists()]},
        "primary": {"metric": PRIMARY_METRIC, "seedA_policy_minus_mix": primary_a["policy_minus_mix"],
                    "ci": [primary_a["ci_low"], primary_a["ci_high"]],
                    "seedA_verdict": primary_a["verdict"],
                    "seedB": {r["seed"]: r["policy_minus_mix"] for r in primary_b},
                    "combined_verdict": [combined_verdict(primary_a["verdict"], r["policy_minus_mix"])
                                         for r in primary_b] or [primary_a["verdict"]]},
        "rules": "plan §7: seeds never averaged; positive with CI including 0 -> unconfirmed; "
                 "seedA positive and seedB negative -> unstable",
        "F_note": F_NOTE,
    }  # fmt: skip
    (out_dir / "phase_d_summary.json").write_text(json.dumps(summary, indent=2))
    plot(stage_a, a_rows, out_dir / "figures" / "phase_d_quality_vs_level")
    cols = [
        "seed",
        "metric",
        "n_evaluated",
        "p_continue",
        "mean_level",
        "policy_q",
        "mix_q_same_p",
        "policy_minus_mix",
        "ci_low",
        "ci_high",
        "verdict",
        "fixed15_q",
        "n_failure_continued",
    ]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pd.DataFrame(a_rows + b_rows)[cols].round(4).to_string(index=False))
    print(json.dumps(summary["primary"], indent=1))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
