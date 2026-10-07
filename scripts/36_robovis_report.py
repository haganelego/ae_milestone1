"""ROBOVIS2027: assemble outputs/robovis2027/<run_id>/report.md (no model calls).

Usage:
    uv run --project egtr python scripts/36_robovis_report.py --run-id dev_r0 \
        --paths F1,F2

Reads (all optional; missing inputs become a visible "not available" line):
  outputs/robovis2027/preflight.json, <run>/manifest.json, selection_manifest,
  <run>/stage_summary.json, <run>/pilot_report.md, <run>/eval/<P>[/q_<m>]/*,
  <run>/eval/signal_compare.csv, <run>/usage.jsonl, <run>/summaries/*.json,
  outputs/robovis2027/figures/fig1_pipeline.png, report_notes.md
  (--notes, else <run>/report_notes.md, else outputs/robovis2027/report_notes.md).
``--refresh`` first reruns scripts 37, 30, 31 and 32 (model-free) so the whole
report is one command:
    uv run --project egtr python scripts/36_robovis_report.py --run-id dev_r0 \
        --paths F1,F2 --refresh
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.robovis_data import load_robovis_config  # noqa: E402
from milestone1.robovis_eval import OracleParams  # noqa: E402
from milestone1.robovis_report import build_report, report_metric_roles  # noqa: E402

NOTES_FILENAME = "report_notes.md"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--paths", required=True, help="comma list, e.g. F1,F2,F2__9b")
    p.add_argument(
        "--q-metrics",
        default=None,
        help="subset of evaluation.report_q_metrics (default: all of them, R50 first)",
    )
    p.add_argument(
        "--graphs-subdir",
        default="graphs",
        help="for --refresh: graphs | graphs_clean | auto (graphs_clean for paths with a 'G' segment)",
    )
    p.add_argument("--notes", default=None, help="report_notes.md to include verbatim")
    p.add_argument("--out", default=None, help="default: <run>/report.md")
    p.add_argument("--config", default=None)
    p.add_argument(
        "--refresh",
        action="store_true",
        help="first rerun scripts 37 (Fig 1), 30+31 per path x metric, 32 per metric",
    )
    return p.parse_args()


def run_step(script: str, *args: str) -> None:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / script), *args]
    print("$", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def refresh(
    run_id: str,
    paths: list[str],
    q_metrics: list[str],
    config: str | None,
    graphs_subdir: str,
) -> None:
    """Regenerate every eval output and figure the report reads (no model calls)."""
    cfg_args = ["--config", config] if config else []
    run_step("37_robovis_fig1_pipeline.py", *cfg_args)
    for m in q_metrics:
        for path in paths:
            common = ["--run-id", run_id, "--path", path, "--q-metric", m, *cfg_args]
            run_step(
                "30_robovis_replay_eval.py", *common, "--graphs-subdir", graphs_subdir
            )
            run_step("31_robovis_figures.py", *common)
        run_step("32_robovis_signal_diag.py", "--run-id", run_id,
                 "--paths", ",".join(paths), "--q-metric", m,
                 "--graphs-subdir", graphs_subdir, *cfg_args)  # fmt: skip


def git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "n/a"
    return out.stdout.strip()


def find_notes(explicit: str | None, run_dir: Path, output_root: Path) -> Path | None:
    if explicit:
        return resolve(explicit)
    for cand in (run_dir / NOTES_FILENAME, output_root / NOTES_FILENAME):
        if cand.exists():
            return cand
    return None


def main() -> None:
    args = parse_args()
    cfg = load_robovis_config(args.config)
    output_root = resolve(cfg["output_root"])
    run_dir = output_root / args.run_id
    paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    roles = report_metric_roles(cfg)
    q_metrics = (
        [m.strip() for m in args.q_metrics.split(",") if m.strip()]
        if args.q_metrics
        else list(roles)
    )
    unknown = [m for m in q_metrics if m not in roles]
    if unknown:
        raise SystemExit(
            f"--q-metrics {unknown} are not agreed evaluation metrics "
            f"(evaluation.report_q_metrics = {roles})"
        )
    if args.refresh:
        refresh(args.run_id, paths, q_metrics, args.config, args.graphs_subdir)
    notes = find_notes(args.notes, run_dir, output_root)
    header = [
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `scripts/36_robovis_report.py`"
        f" at commit `{git_head()}`.",
        f"Paths: {', '.join(paths)}. q metrics: {', '.join(q_metrics)}."
        " All numbers are read from files in this run directory.",
    ]
    text = build_report(
        run_id=args.run_id,
        output_root=output_root,
        selection_manifest=resolve(cfg["paths"]["selection_manifest"]),
        paths=paths,
        q_metrics=q_metrics,
        num_levels=len(cfg["resolution"]["scales"]),
        oracle_params=OracleParams.from_config(cfg),
        metric_roles=roles,
        notes=notes,
        header_lines=header,
    )
    out = resolve(args.out) if args.out else run_dir / "report.md"
    out.write_text(text)
    n_na = text.count("_not available:")
    print(
        f"wrote {out} ({len(text.splitlines())} lines, {n_na} 'not available' entries)"
    )


if __name__ == "__main__":
    main()
