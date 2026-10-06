"""ROBOVIS2027: replay the stop policy over cached outputs and evaluate (model-free).

Usage:
    uv run --project egtr python scripts/30_robovis_replay_eval.py --run-id dev_r0
    uv run --project egtr python scripts/30_robovis_replay_eval.py \
        --run-id synthetic_r0 --input-dir tests/fixtures/robovis_synthetic

Reads ``<input_dir>`` (default ``outputs/robovis2027/<run_id>``) and writes
``outputs/robovis2027/<run_id>/<eval_subdir>/{stops,summary,...}.csv``.
Missing / failed / invalid levels count as a missing C_t (resets the
consecutive counter, counted in failure_count).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.robovis_data import load_robovis_config  # noqa: E402
from milestone1.robovis_replay import RunLayout, run_replay_eval  # noqa: E402

SUMMARY_COLUMNS = [
    "method",
    "n_images",
    "mean_stop_t",
    "R50_stop",
    "mR50_stop",
    "mean_residual",
    "oracle_mae",
    "oracle_n",
    "forced_terminal_rate",
    "inference_failure_rate",
    "eq_fixed_R50",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--input-dir", default=None, help="default: <output_root>/<run_id>")
    p.add_argument("--config", default=None, help="default: configs/robovis2027.yaml")
    p.add_argument(
        "--path",
        default=None,
        help="check path tag as written by script 29 (e.g. F1, F2, F1__tag): "
        "reads checks/<path>, decisions/<path>, writes eval/<path>",
    )
    p.add_argument("--graphs-subdir", default="graphs")
    p.add_argument("--checks-subdir", default=None, help="default: checks[/<path>]")
    p.add_argument(
        "--decisions-subdir", default=None, help="default: decisions[/<path>]"
    )
    p.add_argument("--eval-subdir", default=None, help="default: eval[/<path>]")
    return p.parse_args()


def _subdir(explicit: str | None, base: str, path: str | None) -> str:
    if explicit is not None:
        return explicit
    return base if path is None else f"{base}/{path}"


def main() -> None:
    args = parse_args()
    cfg = load_robovis_config(args.config)
    run_dir = resolve(cfg["output_root"]) / args.run_id
    input_dir = resolve(args.input_dir) if args.input_dir else run_dir
    out_dir = run_dir / _subdir(args.eval_subdir, "eval", args.path)
    layout = RunLayout(
        args.graphs_subdir,
        _subdir(args.checks_subdir, "checks", args.path),
        _subdir(args.decisions_subdir, "decisions", args.path),
    )

    digest = run_replay_eval(input_dir, out_dir, cfg, layout)
    print(f"images={digest['n_images']} eval={digest['n_eval']} -> {out_dir}")
    print(f"level outcomes: {digest['outcomes']}")
    summary: pd.DataFrame = digest["summary"]
    adaptive = summary[~summary["method"].str.startswith("fixed_")]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(adaptive[SUMMARY_COLUMNS].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
