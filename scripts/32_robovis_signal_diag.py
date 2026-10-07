"""ROBOVIS2027: stop-signal diagnostic per state (no replay, no model calls).

Usage:
    uv run --project egtr python scripts/32_robovis_signal_diag.py --run-id dev_r0 --paths F1,F2

Per path: outputs/robovis2027/<run_id>/eval/<P>/{signal_per_state.csv,
signal_by_level.csv, signal_stats.json, signal_by_level.png/.pdf}.
Non-R50 --q-metric outputs go to eval/<P>/q_<metric>/ (R50 keeps eval/<P>/).
Cross-path: outputs/robovis2027/<run_id>/eval/signal_compare.csv (one row per
path x q_metric x condition; only rerun (path, q_metric) rows are replaced).
Conditions benefit_gated* are exploratory (plan deviation, needs sign-off).

Reads  outputs/robovis2027/<run_id>/{stage_metrics.csv, graphs/, checks/<P>/, decisions/<P>/}
Writes outputs/robovis2027/<run_id>/eval/<P>/{signal_per_state.csv,
       signal_by_level.csv, signal_stats.json, signal_by_level.png/.pdf}

Levels default to those that have check/decision files (e.g. 1,5,10,15,20).
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.robovis_data import load_robovis_config  # noqa: E402
from milestone1.robovis_diag import (  # noqa: E402
    JUDGE_QUALITY_FILENAME,
    DiagResult,
    bootstrap_mean_ci,
    compare_paths,
    run_signal_diag,
)
from milestone1.robovis_eval import (  # noqa: E402
    DEFAULT_Q_METRIC,
    configured_q_metric,
    q_output_dir,
)
from milestone1.robovis_replay import RunLayout, resolve_graphs_subdir  # noqa: E402
from milestone1.stop_policy import (  # noqa: E402
    OUTCOME_EMPTY_GRAPH,
    OUTCOME_OK,
    build_conditions,
)

FIG_SIZE = (4.2, 3.0)
DPI = 200
FONT_SIZE = 8
CAPSIZE = 2.5
X_JITTER = 0.25  # horizontal offset between series at the same level
SERIES = (
    ("A", "A_t (observed)", "#c55a11", "s"),
    ("H", "H_t (hypotheses)", "#548235", "^"),
    ("C_proposed", "C_t (proposed)", "#1f4e79", "o"),
    ("C_benefit_gated", "C_t benefit-gated (expl.)", "#7030a0", "D"),
)
DEFAULT_COMPARE_LEVELS = "1,10,20"
COMPARE_FILENAME = "signal_compare.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--paths", default=None, help="comma list: F1,F2,F2__9b,...")
    p.add_argument("--path", default=None, help="single path (alias of --paths)")
    p.add_argument("--compare-levels", default=DEFAULT_COMPARE_LEVELS)
    p.add_argument(
        "--graphs-subdir",
        default="graphs",
        help="graphs | graphs_clean | auto (graphs_clean for paths with a 'G' segment, per path)",
    )
    p.add_argument(
        "--q-metric",
        default=None,
        help="stage_metrics column used as q_t (default: evaluation.q_metric = R50)",
    )
    p.add_argument("--input-dir", default=None, help="default: <output_root>/<run_id>")
    p.add_argument("--config", default=None)
    p.add_argument("--levels", default=None, help="comma list; default: auto-detect")
    return p.parse_args()


def plot_signal_by_level(
    states: pd.DataFrame, cfg: dict, theta: int, title: str, out_stem: Path
) -> None:
    ev = cfg["evaluation"]
    ok = states[states["outcome"].isin([OUTCOME_OK, OUTCOME_EMPTY_GRAPH])]
    levels = sorted(ok["t"].unique())
    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    offsets = np.linspace(-X_JITTER, X_JITTER, len(SERIES))
    for (col, label, color, marker), dx in zip(SERIES, offsets, strict=True):
        means, lows, highs = [], [], []
        for t in levels:
            values = ok.loc[ok["t"] == t, col].to_numpy(float)
            lo, hi = bootstrap_mean_ci(
                values,
                int(ev["bootstrap_repeats"]),
                int(ev["bootstrap_seed"]),
                float(ev["ci_level"]),
            )
            m = float(np.nanmean(values))
            means.append(m)
            lows.append(m - lo if not np.isnan(lo) else 0.0)
            highs.append(hi - m if not np.isnan(hi) else 0.0)
        ax.errorbar(
            np.asarray(levels, float) + dx,
            means,
            yerr=[lows, highs],
            color=color,
            marker=marker,
            capsize=CAPSIZE,
            linewidth=1.0,
            label=label,
        )
    ax.axhline(theta, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xticks(levels)
    ax.set_xlabel("resolution level t")
    ax.set_ylabel("mean signal (0-2)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, fontsize=FONT_SIZE - 1)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=DPI)
    plt.close(fig)


def run_one(
    path: str,
    run_dir: Path,
    input_dir: Path,
    cfg: dict,
    levels: list[int] | None,
    q_metric: str,
    graphs_subdir: str,
) -> DiagResult:
    out_dir = q_output_dir(run_dir / "eval" / path, q_metric)
    graphs = resolve_graphs_subdir(graphs_subdir, path)
    layout = RunLayout(graphs, f"checks/{path}", f"decisions/{path}")
    res = run_signal_diag(input_dir, out_dir, cfg, layout, levels, q_metric)
    theta = int(cfg["proposal"]["theta"])
    title = f"{run_dir.name} {path} (q={q_metric})"
    plot_signal_by_level(res.states, cfg, theta, title, out_dir / "signal_by_level")
    cols = ["t", "n", "n_missing", "A_mean", "H_mean", "C_proposed_mean",
            "C_benefit_gated_mean", "frac_C_proposed_ge_theta",
            "frac_C_benefit_gated_ge_theta", "q_mean", "gain_left_mean"]  # fmt: skip
    print(f"===== {path} (graphs: {graphs}) -> {out_dir}")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(res.levels[cols].round(3).to_string(index=False))
        print_judge_quality(out_dir, res.stats.get("judge_quality") or {})
    return res


def print_judge_quality(out_dir: Path, jq: dict) -> None:
    """Calibration bins + rank statistics of the judge's 'resolved' (zero-call)."""
    table = pd.read_csv(out_dir / JUDGE_QUALITY_FILENAME)
    cal = table[table["table"] == "calibration"]
    print("judge quality: calibration by q_t/qmax")
    print(cal[["q_ratio_bin", "n", "frac_resolved_mean", "frac_lowres_mean",
               "any_gate_theta_mean", "C_proposed_mean"]].round(3).to_string(index=False))  # fmt: skip
    if jq:
        print(
            f"rho(resolved,t)={jq['rho_resolved_t']['rho']:+.3f} "
            f"rho(resolved,q/qmax)={jq['rho_resolved_qratio']['rho']:+.3f} "
            f"AUC_sufficient(resolved)={jq['auc_sufficient_resolved']:.3f} "
            f"AUC_sufficient(1-C_proposed)={jq['auc_sufficient_C']['C_proposed']:.3f} "
            f"| diagnostic, not a stop rule: rho(frac_lowres,q/qmax)="
            f"{jq['rho_lowres_qratio']['rho']:+.3f} "
            f"AUC_sufficient(1-frac_lowres)={jq['auc_sufficient_lowres']:.3f} "
            f"(n={jq['n_states']}, sufficient={jq['n_sufficient']})"
        )


def write_compare(
    compare: pd.DataFrame, out_path: Path, paths: list[str], q_metric: str
) -> pd.DataFrame:
    """Upsert rows of (``paths``, ``q_metric``) into the cross-path table."""
    if out_path.exists():
        old = pd.read_csv(out_path)
        if "q_metric" not in old.columns:  # tables written before q_metric existed
            old.insert(1, "q_metric", DEFAULT_Q_METRIC)
        replaced = old["path"].isin(paths) & (old["q_metric"] == q_metric)
        compare = pd.concat([old[~replaced], compare], ignore_index=True)
    compare.to_csv(out_path, index=False)
    return compare


def main() -> None:
    args = parse_args()
    raw = args.paths or args.path
    if not raw:
        raise SystemExit("--paths (or --path) is required")
    paths = [x.strip() for x in raw.split(",") if x.strip()]
    cfg = load_robovis_config(args.config)
    run_dir = resolve(cfg["output_root"]) / args.run_id
    input_dir = resolve(args.input_dir) if args.input_dir else run_dir
    levels = [int(x) for x in args.levels.split(",")] if args.levels else None
    compare_levels = [int(x) for x in args.compare_levels.split(",")]

    q_metric = configured_q_metric(cfg, args.q_metric)
    results = {
        p: run_one(p, run_dir, input_dir, cfg, levels, q_metric, args.graphs_subdir)
        for p in paths
    }
    compare = compare_paths(results, build_conditions(cfg), compare_levels)
    out_path = run_dir / "eval" / COMPARE_FILENAME
    full = write_compare(compare, out_path, paths, q_metric)
    show = (
        ["path", "q_metric", "graphs_dir", "condition", "n_valid"]
        + [c for c in full.columns if c.startswith("frac_ge_theta")]
        + ["rho_t", "rho_gain_left", "auc_score", "auc_ge_theta"]
        + [
            c
            for c in (
                "rho_resolved_qratio",
                "auc_sufficient_C",
                "auc_sufficient_resolved",
                "rho_lowres_qratio",
                "auc_sufficient_lowres",
            )
            if c in full.columns
        ]  # fmt: skip
    )
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(full[show].round(3).to_string(index=False))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
