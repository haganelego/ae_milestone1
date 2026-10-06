"""ROBOVIS2027: figures from the replay evaluation (script 30 outputs).

Usage:
    uv run --project egtr python scripts/31_robovis_figures.py --run-id dev_r0

Reads  outputs/robovis2027/<run_id>/<eval_subdir>/{summary.csv,per_image.csv,bootstrap.json}
Writes outputs/robovis2027/<run_id>/<eval_subdir>/figures/fig2_fixed_curve.{png,pdf}
       outputs/robovis2027/<run_id>/<eval_subdir>/figure_examples.json

Fig 2: mean R@50 vs mean stop level for fixed_01..fixed_N (curve) with the
adaptive conditions as points; error bars are bootstrap CIs of the mean
stop level (x) and mean R@50 at stop (y).
Fig 3 (selection only): improvement-max / degradation-max / median images by
residual difference between --example-method and --example-baseline.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.robovis_data import load_robovis_config  # noqa: E402
from milestone1.robovis_eval import select_figure_examples  # noqa: E402
from milestone1.stop_policy import (  # noqa: E402
    FIXED_PREFIX,
    OBSERVED_ONLY,
    PROPOSED,
    UNKNOWN_CONSERVATIVE,
)

FIG_SIZE = (4.2, 3.2)
DPI = 200
FIXED_COLOR = "0.45"
POINT_STYLES: dict[str, dict[str, Any]] = {
    PROPOSED: {"color": "#1f4e79", "marker": "o", "label": "proposed"},
    OBSERVED_ONLY: {"color": "#c55a11", "marker": "s", "label": "observed only"},
    UNKNOWN_CONSERVATIVE: {"color": "#548235", "marker": "^", "label": "unknown-cons."},
}
FIXED_LABEL_EVERY = 5  # annotate every N-th fixed level
CAPSIZE = 2.5
MARKER_SIZE = 6
FONT_SIZE = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--path", default=None, help="check path tag (F1, F2, ...)")
    p.add_argument("--eval-subdir", default=None, help="default: eval[/<path>]")
    p.add_argument("--example-method", default=PROPOSED)
    p.add_argument("--example-baseline", default=OBSERVED_ONLY)
    return p.parse_args()


def _ci_err(stat: dict[str, float]) -> list[list[float]]:
    return [[stat["mean"] - stat["ci_low"]], [stat["ci_high"] - stat["mean"]]]


def plot_fixed_curve(
    summary: pd.DataFrame, boot: dict[str, Any], out_stem: Path
) -> None:
    plt.rcParams.update({"font.size": FONT_SIZE})
    fixed = summary[summary["method"].str.startswith(FIXED_PREFIX)].sort_values(
        "mean_stop_t"
    )
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(
        fixed["mean_stop_t"],
        fixed["R50_stop"],
        color=FIXED_COLOR,
        marker=".",
        linewidth=1.0,
        label="fixed stop level",
        zorder=1,
    )
    for row in fixed.itertuples(index=False):
        level = int(row.mean_stop_t)
        if level % FIXED_LABEL_EVERY == 0:
            ax.annotate(
                str(level),
                (row.mean_stop_t, row.R50_stop),
                textcoords="offset points",
                xytext=(0, -9),
                ha="center",
                color=FIXED_COLOR,
            )
    for method, style in POINT_STYLES.items():
        stats = boot.get("methods", {}).get(method)
        if stats is None:
            continue
        ax.errorbar(
            stats["stop_t"]["mean"],
            stats["R50_stop"]["mean"],
            xerr=_ci_err(stats["stop_t"]),
            yerr=_ci_err(stats["R50_stop"]),
            fmt=style["marker"],
            color=style["color"],
            markersize=MARKER_SIZE,
            capsize=CAPSIZE,
            label=style["label"],
            zorder=3,
        )
    ax.set_xlabel("mean stop level (observations)")
    ax.set_ylabel("mean R@50 at stop")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=DPI)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_robovis_config(args.config)
    eval_subdir = args.eval_subdir or (
        "eval" if args.path is None else f"eval/{args.path}"
    )
    eval_dir = resolve(cfg["output_root"]) / args.run_id / eval_subdir
    summary = pd.read_csv(eval_dir / "summary.csv")
    per_image = pd.read_csv(eval_dir / "per_image.csv")
    boot = json.loads((eval_dir / "bootstrap.json").read_text())

    fig2 = eval_dir / "figures" / "fig2_fixed_curve"
    plot_fixed_curve(summary, boot, fig2)

    examples = select_figure_examples(
        per_image, args.example_method, args.example_baseline
    )
    (eval_dir / "figure_examples.json").write_text(json.dumps(examples, indent=2))
    print(f"wrote {fig2}.png/.pdf and {eval_dir / 'figure_examples.json'}")
    print(json.dumps(examples["examples"]))


if __name__ == "__main__":
    main()
