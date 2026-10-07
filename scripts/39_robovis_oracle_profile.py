"""ROBOVIS2027: oracle definitions per q metric, straight from stage_metrics.csv.

Usage:
    uv run --project egtr python scripts/39_robovis_oracle_profile.py --run-id test_r0 \
        --q-metrics R50,relsg_gt,det_recall20_clean

Needs no checks/decisions (works for runs without model outputs). Writes
outputs/robovis2027/<run_id>/eval/oracle_profile.csv: one row per q metric x
definition (relative_stable = main; monotone_stable, gain_based = auxiliary)
with status counts, defined-at-last-level and median oracle level.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.robovis_data import load_robovis_config  # noqa: E402
from milestone1.robovis_eval import (  # noqa: E402
    OracleParams,
    configured_q_metric,
    oracle_profile_from_stage,
)

PROFILE_FILENAME = "oracle_profile.csv"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--q-metrics", default=None, help="comma list; default: config q_metric"
    )
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_robovis_config(args.config)
    run_dir = resolve(cfg["output_root"]) / args.run_id
    metrics = (
        [m.strip() for m in args.q_metrics.split(",") if m.strip()]
        if args.q_metrics
        else [configured_q_metric(cfg)]
    )
    params = OracleParams.from_config(cfg)
    num_levels = len(cfg["resolution"]["scales"])
    table = pd.concat(
        [
            oracle_profile_from_stage(
                run_dir / "stage_metrics.csv", m, params, num_levels
            )
            for m in metrics
        ],
        ignore_index=True,
    )
    out = run_dir / "eval" / PROFILE_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"ratio={params.ratio} gain_delta={params.gain_delta} -> {out}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
