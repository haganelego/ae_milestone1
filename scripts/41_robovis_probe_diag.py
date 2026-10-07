"""ROBOVIS2027: image-only view-sufficiency probe diagnostic (zero model calls).

Usage:
    uv run --project egtr python scripts/41_robovis_probe_diag.py --run-id dev_r0 \
        --cfgs 9b_384,9b_768,4b_384,4b_768

Reads  outputs/robovis2027/<run_id>/{stage_metrics.csv, probe/<cfg>/<image_id>/level_XX.json}
Writes outputs/robovis2027/<run_id>/eval/probe/<cfg>[/q_<metric>]/
         probe_states.csv probe_by_level.csv probe_stats.json stops.csv summary.csv bootstrap.json
       outputs/robovis2027/<run_id>/eval/probe/probe_compare.csv (cfg x q metric; upserted)

probe_stop / probe_stop_next are EXPLORATORY probe-only policies (k-consecutive
replay of sufficient=true / next_step_helps=false). probe_gate_proposed /
probe_gate_observed_only gate the judge C_t of --decisions-path (default
F1v_v3_G__9b, graphs auto) with the probe: not sufficient -> C = 2 (continue).
None of these are plan conditions; the ungated proposed / observed_only of the
decisions path are reported as references.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.robovis_data import load_robovis_config  # noqa: E402
from milestone1.robovis_probe import compare_row, run_probe_diag  # noqa: E402
from milestone1.robovis_replay import RunLayout, resolve_graphs_subdir  # noqa: E402
from milestone1.robovis_report import report_metric_roles  # noqa: E402

COMPARE_FILENAME = "probe_compare.csv"
DEFAULT_DECISIONS_PATH = "F1v_v3_G__9b"
NO_GATE = "none"
SHOW_COLUMNS = ["cfg", "q_metric", "n_states", "failure_rate", "rho_detail_t",
                "rho_detail_qratio", "auc_detail", "auc_not_next", "frac_monotone",
                "mean_inversions", "probe_stop_mean_stop", "probe_stop_diff",
                "probe_stop_next_mean_stop", "probe_stop_next_diff",
                "probe_gate_proposed_mean_stop", "probe_gate_proposed_diff",
                "probe_gate_proposed_minus_proposed", "proposed_mean_stop",
                "probe_gate_observed_only_diff"]  # fmt: skip


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--cfgs", required=True, help="comma list of probe configs")
    p.add_argument(
        "--q-metrics", default=None, help="default: evaluation.report_q_metrics"
    )
    p.add_argument("--input-dir", default=None, help="default: <output_root>/<run_id>")
    p.add_argument("--probe-subdir", default="probe")
    p.add_argument(
        "--decisions-path",
        default=DEFAULT_DECISIONS_PATH,
        help="decisions path gated by the probe (probe_gate_*); 'none' to skip",
    )
    p.add_argument(
        "--graphs-subdir", default="auto", help="graphs | graphs_clean | auto"
    )
    p.add_argument("--config", default=None)
    return p.parse_args()


def write_compare(rows: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Upsert rows keyed by (cfg, q_metric)."""
    if path.exists():
        old = pd.read_csv(path)
        key = set(zip(rows["cfg"], rows["q_metric"], strict=True))
        keep = [k not in key for k in zip(old["cfg"], old["q_metric"], strict=True)]
        rows = pd.concat([old[keep], rows], ignore_index=True)
    rows.to_csv(path, index=False)
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_robovis_config(args.config)
    run_dir = resolve(cfg["output_root"]) / args.run_id
    input_dir = resolve(args.input_dir) if args.input_dir else run_dir
    cfgs = [c.strip() for c in args.cfgs.split(",") if c.strip()]
    metrics = (
        [m.strip() for m in args.q_metrics.split(",") if m.strip()]
        if args.q_metrics
        else list(report_metric_roles(cfg))
    )
    out_root = run_dir / "eval" / "probe"
    gate_layout = None
    if args.decisions_path.lower() != NO_GATE:
        path = args.decisions_path
        if (input_dir / "decisions" / path).exists():
            gate_layout = RunLayout(
                resolve_graphs_subdir(args.graphs_subdir, path),
                f"checks/{path}",
                f"decisions/{path}",
            )
        else:
            print(f"gate skipped: decisions/{path} not found")
    rows = []
    for probe_cfg in cfgs:
        probe_dir = input_dir / args.probe_subdir / probe_cfg
        if not probe_dir.exists():
            print(f"skip {probe_cfg}: {probe_dir} not found")
            continue
        for m in metrics:
            res = run_probe_diag(
                input_dir,
                probe_dir,
                out_root / probe_cfg,
                cfg,
                probe_cfg,
                m,
                gate_layout,
            )
            rows.append(compare_row(res))
            print(f"===== {probe_cfg} (q={m}) outcomes={res.stats['outcomes']}")
            with pd.option_context("display.width", 220, "display.max_columns", 20):
                print(res.levels.round(3).to_string(index=False))
    if not rows:
        raise SystemExit("no probe configs found")
    out_root.mkdir(parents=True, exist_ok=True)
    full = write_compare(pd.DataFrame(rows), out_root / COMPARE_FILENAME)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(
            full[[c for c in SHOW_COLUMNS if c in full.columns]]
            .round(3)
            .to_string(index=False)
        )
    print(f"-> {out_root / COMPARE_FILENAME}")


if __name__ == "__main__":
    main()
