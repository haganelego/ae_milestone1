"""ROBOVIS2027 Fig 1: processing pipeline diagram (no data needed).

Usage:
    uv run --project egtr python scripts/37_robovis_fig1_pipeline.py

Writes outputs/robovis2027/figures/fig1_pipeline.{png,pdf} (and the .dot source).
Uses graphviz ``dot`` if it is on PATH, else a matplotlib box-and-arrow fallback.
GT enters only the scoring box (dashed, outside the online loop).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.robovis_data import load_robovis_config  # noqa: E402

FONT = "Helvetica"
ONLINE_FILL = "#eef3f8"
MODEL_FILL = "#dbe7f3"
RULE_FILL = "#e2efda"
SCORE_FILL = "#f2f2f2"
EDGE_COLOR = "#404040"
GT_COLOR = "#8c8c8c"
DPI = 300

# (node id, label, fill). Order = left-to-right online pipeline.
NODES = (
    ("img", "degraded image\nlevel t (scale_t)", ONLINE_FILL),
    ("pgsg", "PGSG\n(fixed, no tuning)", MODEL_FILL),
    ("obsg", "observed graph G_t\nentities + relations", ONLINE_FILL),
    ("checks", "check generator\nF1: VLM (image + G_t)\nF2: text LLM (G_t only)",
     MODEL_FILL),
    ("oh", "O_t: observed issues\nH_t: hypotheses\n(<= 3 each, anchored)", ONLINE_FILL),
    ("judge", "judge (text LLM)\nF1v: + image\nstatus / basis / benefit\n/ limitation",
     MODEL_FILL),
    ("u", "u_j = min(basis, benefit)\nif unresolved & low_resolution\n& anchors + region valid",
     RULE_FILL),
    ("ahc", "A_t = max_O u_j\nH_t = max_H u_j\nC_t = max(A_t, H_t)", RULE_FILL),
    ("stop", "stop rule\nC_t < theta for k = 2\nconsecutive levels", RULE_FILL),
)  # fmt: skip
EDGES = (
    ("img", "pgsg"),
    ("pgsg", "obsg"),
    ("obsg", "checks"),
    ("img", "checks"),
    ("checks", "oh"),
    ("oh", "judge"),
    ("obsg", "judge"),
    ("judge", "u"),
    ("u", "ahc"),
    ("ahc", "stop"),
)
# Edges drawn dashed (optional inputs: image only on F1 / F1v).
OPTIONAL_EDGES = {("img", "checks")}
# Snake layout: top row left-to-right, bottom row right-to-left.
ROW_TOP = ("img", "pgsg", "obsg", "checks", "oh")
ROW_BOTTOM = ("out", "stop", "ahc", "u", "judge")
REVERSED_EDGES = {("judge", "u"), ("u", "ahc"), ("ahc", "stop")}
SCORE_LABEL = (
    "scoring (offline only)\nR@50, mR@50, RelSGScore\nresidual, oracle, fixed curves"
)
CONTINUE_LABEL = "continue: level t+1"
STOP_LABEL = "stop_t"


def _esc(label: str) -> str:
    return label.replace("\n", "\\n")


def build_dot() -> str:
    """Two-row snake layout: perception row, decision row (right to left), scoring."""
    fills = {node_id: (label, fill) for node_id, label, fill in NODES}
    lines = [
        "digraph pipeline {",
        "  graph [rankdir=TB, nodesep=0.3, ranksep=0.45, splines=spline, pad=0.1,"
        " newrank=true];",
        f'  node [shape=box, style="rounded,filled", fontname="{FONT}", fontsize=10,'
        f' color="{EDGE_COLOR}", penwidth=0.8, margin="0.08,0.05"];',
        f'  edge [color="{EDGE_COLOR}", penwidth=0.8, arrowsize=0.6, fontname="{FONT}",'
        " fontsize=9];",
    ]
    for node_id, (label, fill) in fills.items():
        lines.append(f'  {node_id} [label="{_esc(label)}", fillcolor="{fill}"];')
    lines += [
        f'  out [label="{STOP_LABEL}", shape=plaintext, style=""];',
        f'  gt [label="VG ground truth", shape=box, style="filled,dashed",'
        f' fillcolor="{SCORE_FILL}", color="{GT_COLOR}", fontcolor="#404040"];',
        f'  score [label="{_esc(SCORE_LABEL)}", fillcolor="{SCORE_FILL}",'
        f' style="rounded,filled,dashed", color="{GT_COLOR}", fontcolor="#404040"];',
        "  { rank=same; " + "; ".join(ROW_TOP) + "; }",
        "  { rank=same; " + "; ".join(ROW_BOTTOM) + "; }",
        "  { rank=same; gt; score; }",
    ]
    for src, dst in EDGES:
        if (src, dst) in OPTIONAL_EDGES:
            lines.append(f'  {src} -> {dst} [style=dashed, xlabel="F1"];')
        elif (src, dst) in REVERSED_EDGES:
            # Same-rank edge drawn right-to-left: declare reversed, draw back.
            lines.append(f"  {dst} -> {src} [dir=back];")
        else:
            lines.append(f"  {src} -> {dst};")
    lines += [
        "  out -> stop [dir=back];",
        f'  stop -> img [xlabel="{CONTINUE_LABEL}", style=dotted, constraint=false];',
        f'  out -> score [style=dashed, color="{GT_COLOR}"];',
        f'  gt -> score [style=dashed, color="{GT_COLOR}"];',
        "}",
    ]
    return "\n".join(lines) + "\n"


def render_dot(dot_src: Path, out_stem: Path) -> bool:
    dot = shutil.which("dot")
    if dot is None:
        return False
    for fmt in ("png", "pdf"):
        args = [
            dot,
            f"-T{fmt}",
            str(dot_src),
            "-o",
            str(out_stem.with_suffix(f".{fmt}")),
        ]
        if fmt == "png":
            args.insert(1, f"-Gdpi={DPI}")
        subprocess.run(args, check=True)
    return True


def render_matplotlib(out_stem: Path) -> None:
    """Fallback: one row of boxes with arrows (no graphviz)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    n = len(NODES)
    box_w, box_h, gap = 1.6, 1.1, 0.35
    fig_w = n * (box_w + gap)
    fig, ax = plt.subplots(figsize=(fig_w * 0.75, 3.0))
    for i, (_, label, fill) in enumerate(NODES):
        x = i * (box_w + gap)
        ax.add_patch(FancyBboxPatch((x, 1.0), box_w, box_h, boxstyle="round,pad=0.05",
                                    fc=fill, ec=EDGE_COLOR, lw=0.8))  # fmt: skip
        ax.text(x + box_w / 2, 1.0 + box_h / 2, label, ha="center", va="center",
                fontsize=6.5)  # fmt: skip
        if i < n - 1:
            ax.annotate("", xy=(x + box_w + gap, 1.55), xytext=(x + box_w, 1.55),
                        arrowprops={"arrowstyle": "->", "color": EDGE_COLOR, "lw": 0.8})  # fmt: skip
    ax.add_patch(FancyBboxPatch((fig_w - 2 * (box_w + gap), 0.05), 2 * box_w, 0.6,
                                boxstyle="round,pad=0.05", fc=SCORE_FILL, ec=GT_COLOR,
                                ls="--", lw=0.8))  # fmt: skip
    ax.text(fig_w - 2 * (box_w + gap) + box_w, 0.35, SCORE_LABEL.replace("\n", "; "),
            ha="center", va="center", fontsize=6, color="#404040")  # fmt: skip
    ax.set_xlim(-0.2, fig_w)
    ax.set_ylim(0, 2.4)
    ax.axis("off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=DPI)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=None)
    p.add_argument("--force-matplotlib", action="store_true")
    args = p.parse_args()
    cfg = load_robovis_config(args.config)
    out_dir = resolve(cfg["output_root"]) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = out_dir / "fig1_pipeline"
    dot_src = out_stem.with_suffix(".dot")
    dot_src.write_text(build_dot())
    used = "graphviz"
    if args.force_matplotlib or not render_dot(dot_src, out_stem):
        render_matplotlib(out_stem)
        used = "matplotlib"
    print(f"wrote {out_stem}.png/.pdf ({used})")


if __name__ == "__main__":
    main()
