"""ROBOVIS2027: figures from the replay evaluation (script 30 outputs).

Usage:
    uv run --project egtr python scripts/31_robovis_figures.py --run-id dev_r0

Reads  outputs/robovis2027/<run_id>/<eval_subdir>/{summary.csv,per_image.csv,bootstrap.json}
Writes outputs/robovis2027/<run_id>/<eval_subdir>/figures/fig2_fixed_curve.{png,pdf}
       outputs/robovis2027/<run_id>/<eval_subdir>/figure_examples.json

Fig 2: mean q (R@50 or --q-metric) vs mean stop level for fixed_01..fixed_N
(curve) with the adaptive conditions as points; error bars are bootstrap CIs
of the mean stop level (x) and mean q at stop (y).
Fig 3: improvement-max / degradation-max / median images by residual difference
between --example-method and --example-baseline (figure_examples.json), rendered
as eval/.../figures/fig3_examples.{png,pdf}: level images at 3-4 levels (both
stop levels marked), the q_t curve with both stops, fixed_20 and the oracle, and
the hypotheses at the method's stop level (nearest checks level if sparse).
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.robovis_data import level_name, load_robovis_config  # noqa: E402
from milestone1.robovis_eval import (  # noqa: E402
    RecallCurves,
    build_curves,
    configured_q_metric,
    load_stage_metrics,
    q_output_dir,
    select_figure_examples,
)
from milestone1.robovis_panels import hypotheses_at, panel_levels  # noqa: E402
from milestone1.stop_policy import (  # noqa: E402
    BENEFIT_GATED,
    BENEFIT_GATED_OBS,
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
    # Exploratory (plan deviation): hollow markers.
    BENEFIT_GATED: {
        "color": "#7030a0",
        "marker": "D",
        "label": "benefit-gated (expl.)",
        "mfc": "none",
    },
    BENEFIT_GATED_OBS: {
        "color": "#7030a0",
        "marker": "v",
        "label": "benefit-gated obs (expl.)",
        "mfc": "none",
    },
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
    p.add_argument("--q-metric", default=None, help="default: evaluation.q_metric")
    p.add_argument("--example-method", default=PROPOSED)
    p.add_argument("--example-baseline", default=OBSERVED_ONLY)
    p.add_argument(
        "--input-dir",
        default=None,
        help="run inputs (stage_metrics, checks/, decisions/); default: run dir",
    )
    p.add_argument("--no-fig3", action="store_true", help="skip the Fig 3 panels")
    return p.parse_args()


Q_LABELS = {"R50": "R@50", "relsg_gt": "RelSGScore (GT)"}


def q_label(metric: str | None) -> str:
    return Q_LABELS.get(metric or "R50", metric or "R50")


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
        fixed["q_stop"],
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
                (row.mean_stop_t, row.q_stop),
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
            stats["q_stop"]["mean"],
            xerr=_ci_err(stats["stop_t"]),
            yerr=_ci_err(stats["q_stop"]),
            fmt=style["marker"],
            color=style["color"],
            markersize=MARKER_SIZE,
            markerfacecolor=style.get("mfc", style["color"]),
            capsize=CAPSIZE,
            label=style["label"],
            zorder=3,
        )
    ax.set_xlabel("mean stop level (observations)")
    ax.set_ylabel(f"mean {q_label(boot.get('q_metric'))} at stop")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3: qualitative panels
# ---------------------------------------------------------------------------

FIG3_ROW_HEIGHT = 2.3
FIG3_WIDTH = 14.0
FIG3_WIDTH_RATIOS = (1, 1, 1, 1, 1.5, 2.3)
PLACEHOLDER_GRAY = "0.85"
BORDER_WIDTH = 3.0
TEXT_FONT_SIZE = 6.5
Y_HEADROOM = 1.1
EXAMPLE_TITLES = {
    "improvement_max": "largest improvement",
    "degradation_max": "largest degradation",
    "median": "median",
}


@dataclass(frozen=True)
class PanelSources:
    images: Path
    checks: Path
    decisions: Path


def _stop_of(stops: pd.DataFrame, image_id: int, method: str) -> int | None:
    sel = stops[(stops["image_id"] == image_id) & (stops["method"] == method)]
    return int(sel["stop_t"].iloc[0]) if len(sel) else None


# Fallback colours by role when a method has no entry in POINT_STYLES.
ROLE_COLORS = {"method": "#1f4e79", "baseline": "#c55a11"}


def _style(method: str, role: str) -> dict[str, Any]:
    fallback = {"color": ROLE_COLORS[role], "marker": "x", "label": method}
    return POINT_STYLES.get(method, fallback)


def _draw_thumb(ax: Any, path: Path, title: str, edge: str | None) -> None:
    if path.exists():
        ax.imshow(Image.open(path).convert("RGB"))
    else:
        ax.set_facecolor(PLACEHOLDER_GRAY)
        ax.text(0.5, 0.5, "image n/a", ha="center", va="center",
                transform=ax.transAxes, fontsize=TEXT_FONT_SIZE)  # fmt: skip
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(edge is not None)
        if edge is not None:
            spine.set_edgecolor(edge)
            spine.set_linewidth(BORDER_WIDTH)
    ax.set_title(title, fontsize=TEXT_FONT_SIZE + 0.5)


def plot_examples(
    examples: dict[str, Any],
    stops: pd.DataFrame,
    oracle: pd.DataFrame,
    curves: RecallCurves,
    sources: PanelSources,
    out_stem: Path,
) -> None:
    """One row per selected image: thumbnails, q_t curve, hypotheses text."""
    method, baseline = examples["method"], examples["baseline"]
    rows = list(examples.get("examples", {}).items())
    if not rows:
        print("fig3: no examples selected; skipped")
        return
    n_levels = curves.num_levels
    oracle_t = oracle.set_index("image_id")["oracle_t"]
    plt.rcParams.update({"font.size": FONT_SIZE})
    fig = plt.figure(figsize=(FIG3_WIDTH, FIG3_ROW_HEIGHT * len(rows)))
    grid = fig.add_gridspec(len(rows), len(FIG3_WIDTH_RATIOS),
                            width_ratios=FIG3_WIDTH_RATIOS, wspace=0.3, hspace=0.45)  # fmt: skip
    m_style, b_style = _style(method, "method"), _style(baseline, "baseline")
    for r, (kind, info) in enumerate(rows):
        image_id = int(info["image_id"])
        stop_m = _stop_of(stops, image_id, method) or n_levels
        stop_b = _stop_of(stops, image_id, baseline) or n_levels
        for c, t in enumerate(panel_levels(stop_m, stop_b, n_levels)):
            marks = [
                s["label"]
                for s, st in ((m_style, stop_m), (b_style, stop_b))
                if st == t
            ]
            edge = (
                m_style["color"]
                if t == stop_m
                else b_style["color"]
                if t == stop_b
                else None
            )
            title = f"t={t}" + (f" ({' & '.join(marks)} stop)" if marks else "")
            img = sources.images / str(image_id) / f"{level_name(t)}.jpg"
            _draw_thumb(fig.add_subplot(grid[r, c]), img, title, edge)
        ax = fig.add_subplot(grid[r, 4])
        q = curves.q.get(image_id)
        levels = np.arange(1, n_levels + 1)
        if q is not None:
            ax.plot(levels, q, color="0.35", linewidth=1.0, marker=".", markersize=3)
            ax.scatter([n_levels], [q[-1]], color="0.2", marker="s", s=14,
                       label="fixed_20", zorder=3)  # fmt: skip
        for st, style in ((stop_m, m_style), (stop_b, b_style)):
            ax.axvline(st, color=style["color"], linestyle="--", linewidth=1.0,
                       label=f"{style['label']} stop")  # fmt: skip
        ot = oracle_t.get(image_id)
        if ot is not None and not pd.isna(ot):
            ax.axvline(
                int(ot), color="0.6", linestyle=":", linewidth=1.0, label="oracle"
            )
        ax.set_xlim(0.5, n_levels + 0.5)
        q_top = float(np.nanmax(q)) if q is not None and np.isfinite(q).any() else 0.0
        ax.set_ylim(0.0, q_top * Y_HEADROOM if q_top > 0 else 1.0)
        ax.set_xlabel("level t", fontsize=TEXT_FONT_SIZE)
        ax.set_ylabel(q_label(curves.metric), fontsize=TEXT_FONT_SIZE)
        ax.tick_params(labelsize=TEXT_FONT_SIZE)
        ax.spines[["top", "right"]].set_visible(False)
        if r == 0:
            ax.legend(frameon=False, fontsize=TEXT_FONT_SIZE - 1, loc="lower right")
        hyp = hypotheses_at(sources.checks, sources.decisions, image_id, stop_m)
        tax = fig.add_subplot(grid[r, 5])
        tax.axis("off")
        where = f"t={hyp.level}" if hyp.level is not None else "no checks"
        if hyp.level is not None and hyp.level != stop_m:
            where += f" (nearest to stop t={stop_m})"
        header = (f"{EXAMPLE_TITLES.get(kind, kind)} | image {image_id} | "
                  f"residual diff {info['residual_diff']:+.3f}\n"
                  f"hypotheses at {m_style['label']} stop, {where}:")  # fmt: skip
        body = "\n".join(f"- {line}" for line in hyp.lines) or "- not available"
        tax.text(0.0, 1.0, header + "\n" + body, va="top", ha="left",
                 fontsize=TEXT_FONT_SIZE, wrap=True, transform=tax.transAxes)  # fmt: skip
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_robovis_config(args.config)
    eval_subdir = args.eval_subdir or (
        "eval" if args.path is None else f"eval/{args.path}"
    )
    q_metric = configured_q_metric(cfg, args.q_metric)
    eval_dir = q_output_dir(
        resolve(cfg["output_root"]) / args.run_id / eval_subdir, q_metric
    )
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
    if args.no_fig3:
        return
    run_dir = resolve(cfg["output_root"]) / args.run_id
    input_dir = resolve(args.input_dir) if args.input_dir else run_dir
    stage = load_stage_metrics(input_dir / "stage_metrics.csv", q_metric)
    curves = build_curves(stage, q_metric, len(cfg["resolution"]["scales"]))
    sources = PanelSources(
        images=resolve(cfg["paths"]["degraded_images"]),
        checks=input_dir / ("checks" if args.path is None else f"checks/{args.path}"),
        decisions=input_dir
        / ("decisions" if args.path is None else f"decisions/{args.path}"),
    )
    fig3 = eval_dir / "figures" / "fig3_examples"
    plot_examples(
        examples,
        pd.read_csv(eval_dir / "stops.csv"),
        pd.read_csv(eval_dir / "oracle.csv"),
        curves,
        sources,
        fig3,
    )
    print(f"wrote {fig3}.png/.pdf")


if __name__ == "__main__":
    main()
