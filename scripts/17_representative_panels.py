"""Generate representative example panel figures for the SGG-saturation PoC.

Selects 3 images spanning the saturation spectrum (early / mid / late)
and produces detailed panel figures plus a summary overlay.
"""

import csv
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import load_config, resolve

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_csv_dict(path: Path) -> list[dict[str, str]]:
    """Load CSV as list of dicts."""
    with open(path) as f:
        return list(csv.DictReader(f))


def safe_float(val: str) -> float | None:
    """Parse a float, returning None for NaN / empty strings."""
    if val in ("", "NaN", "nan", "None", "none"):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def load_gt(gt_dir: Path, image_id: str) -> dict[str, Any]:
    """Load ground-truth scene graph."""
    with open(gt_dir / f"{image_id}.json") as f:
        return json.load(f)


def load_egtr_pred(pred_dir: Path, image_id: str, level: int) -> dict[str, Any]:
    """Load EGTR prediction for a given level."""
    with open(pred_dir / image_id / f"level_{level:02d}.json") as f:
        return json.load(f)


def load_llm_sat(llm_dir: Path, image_id: str, level: int) -> dict[str, Any] | None:
    """Load LLM saturation result (levels 5-20 only)."""
    p = llm_dir / image_id / f"level_{level:02d}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def wrap_text(text: str, width: int = 80) -> str:
    """Wrap long text."""
    return "\n".join(textwrap.wrap(text, width=width))


# ---------------------------------------------------------------------------
# Image selection
# ---------------------------------------------------------------------------


def select_representative_images(
    oracle_path: Path,
    gt_dir: Path,
    pred_dir: Path,
) -> list[tuple[str, int, str]]:
    """Select 3 representative images: early, mid, late saturation.

    Returns list of (image_id, rel_oracle_stop, one_line_description).
    """
    oracle_rows = load_csv_dict(oracle_path)

    candidates: list[dict[str, Any]] = []
    for row in oracle_rows:
        iid = row["image_id"]
        stop = int(row["rel_oracle_0.8"])

        # Load GT to check richness
        gt = load_gt(gt_dir, iid)
        n_obj = len(gt["objects"])
        n_rel = len(gt["relationships"])

        # Require non-trivial GT
        if n_obj < 5 or n_rel < 3:
            continue

        # Check L20 has relations (avoid zero-rel images)
        pred_l20 = load_egtr_pred(pred_dir, iid, 20)
        if len(pred_l20.get("relationships", [])) == 0:
            continue

        obj_labels = list({o["label"] for o in gt["objects"]})
        desc = (
            f"{n_obj} objects ({', '.join(sorted(obj_labels)[:6])}), {n_rel} relations"
        )

        candidates.append(
            {
                "image_id": iid,
                "stop": stop,
                "n_obj": n_obj,
                "n_rel": n_rel,
                "desc": desc,
                "l20_rels": len(pred_l20.get("relationships", [])),
            }
        )

    # Sort by stop level
    candidates.sort(key=lambda c: (c["stop"], -c["n_rel"]))

    # Pick EARLY: smallest stop, prefer rich GT
    early_pool = [c for c in candidates if c["stop"] <= 9]
    early = max(early_pool, key=lambda c: c["n_rel"]) if early_pool else candidates[0]

    # Pick LATE: stop == 20, prefer rich GT
    late_pool = [c for c in candidates if c["stop"] == 20]
    late = max(late_pool, key=lambda c: c["n_rel"]) if late_pool else candidates[-1]

    # Pick MID: stop around 13-15, prefer rich GT
    mid_pool = [c for c in candidates if 12 <= c["stop"] <= 16]
    # Remove already-picked
    mid_pool = [
        c
        for c in mid_pool
        if c["image_id"] not in {early["image_id"], late["image_id"]}
    ]
    mid = (
        max(mid_pool, key=lambda c: c["n_rel"])
        if mid_pool
        else candidates[len(candidates) // 2]
    )

    result = []
    for c, label in [(early, "EARLY"), (mid, "MID"), (late, "LATE")]:
        print(
            f"  [{label}] image_id={c['image_id']}, RelOracleStop@0.8={c['stop']}, "
            f"GT: {c['n_obj']} obj / {c['n_rel']} rel, L20 rels={c['l20_rels']}"
        )
        print(f"         {c['desc']}")
        result.append((c["image_id"], c["stop"], c["desc"]))

    return result


# ---------------------------------------------------------------------------
# Panel figure per image
# ---------------------------------------------------------------------------


def make_panel_figure(
    image_id: str,
    oracle_stop: int,
    gt_desc: str,
    degraded_dir: Path,
    pred_dir: Path,
    gt_dir: Path,
    metrics_path: Path,
    rel_metrics_path: Path,
    llm_dir: Path,
    out_path: Path,
) -> None:
    """Create a detailed panel figure for one image."""

    # --- Load data ---
    gt = load_gt(gt_dir, image_id)

    # All metrics (vs true GT)
    all_rows = [r for r in load_csv_dict(metrics_path) if r["image_id"] == image_id]
    all_rows.sort(key=lambda r: int(r["level"]))
    levels_all = [int(r["level"]) for r in all_rows]
    node_recall = [safe_float(r["node_recall"]) for r in all_rows]
    triplet_recall = [safe_float(r["triplet_recall"]) for r in all_rows]
    sg_score = [safe_float(r["sg_score"]) for r in all_rows]

    # Relative metrics (vs L20)
    rel_rows = [r for r in load_csv_dict(rel_metrics_path) if r["image_id"] == image_id]
    rel_rows.sort(key=lambda r: int(r["level"]))
    levels_rel = [int(r["level"]) for r in rel_rows]
    rel_sg_score = [safe_float(r["RelSGScore"]) for r in rel_rows]

    # LLM saturation scores
    llm_levels: list[int] = []
    llm_scores: list[float] = []
    for lv in range(5, 21):
        sat = load_llm_sat(llm_dir, image_id, lv)
        if sat is not None:
            llm_levels.append(lv)
            llm_scores.append(float(sat["saturation_score"]))

    # EGTR predictions for text blocks (levels 5, 10, 20)
    text_levels = [5, 10, 20]
    pred_texts: dict[int, dict[str, Any]] = {}
    llm_reasons: dict[int, str] = {}
    for lv in text_levels:
        pred_texts[lv] = load_egtr_pred(pred_dir, image_id, lv)
        sat = load_llm_sat(llm_dir, image_id, lv)
        if sat:
            llm_reasons[lv] = sat.get("reason", "")

    # --- Build figure ---
    fig = plt.figure(figsize=(18, 28))
    fig.suptitle(
        f"Image {image_id}  |  RelOracleStop@0.8 = {oracle_stop}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Main grid: 5 rows
    gs = gridspec.GridSpec(
        5,
        1,
        height_ratios=[1.0, 1.5, 0.8, 2.5, 1.0],
        hspace=0.35,
        top=0.98,
        bottom=0.02,
        left=0.06,
        right=0.94,
    )

    # --- Row 0: Thumbnails ---
    gs_thumbs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0], wspace=0.05)
    thumb_levels = [1, 5, 10, 15, 20]
    for idx, lv in enumerate(thumb_levels):
        ax = fig.add_subplot(gs_thumbs[0, idx])
        img_path = degraded_dir / image_id / f"level_{lv:02d}.jpg"
        if img_path.exists():
            img = mpimg.imread(str(img_path))
            ax.imshow(img)
        ax.set_title(f"Level {lv}", fontsize=11, fontweight="bold")
        ax.axis("off")

    # --- Row 1: Metric curves ---
    ax_met = fig.add_subplot(gs[1])

    # Filter out None values for plotting
    def plot_safe(ax: plt.Axes, x: list, y: list, *args: Any, **kwargs: Any) -> None:
        xf = [xi for xi, yi in zip(x, y) if yi is not None]
        yf = [yi for yi in y if yi is not None]
        if xf:
            ax.plot(xf, yf, *args, **kwargs)

    plot_safe(
        ax_met, levels_all, node_recall, "b-o", markersize=4, label="NodeRecall (vs GT)"
    )
    plot_safe(
        ax_met,
        levels_all,
        triplet_recall,
        "r-s",
        markersize=4,
        label="TripletRecall (vs GT)",
    )
    plot_safe(
        ax_met, levels_all, sg_score, "g-^", markersize=4, label="SGScore (vs GT)"
    )
    plot_safe(
        ax_met,
        levels_rel,
        rel_sg_score,
        "m-D",
        markersize=5,
        linewidth=2,
        label="RelSGScore (vs L20)",
    )

    ax_met.axvline(
        oracle_stop,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"RelOracleStop@0.8 = {oracle_stop}",
    )
    ax_met.set_xlabel("Degradation Level", fontsize=11)
    ax_met.set_ylabel("Score", fontsize=11)
    ax_met.set_xlim(0.5, 20.5)
    ax_met.set_ylim(-0.05, 1.05)
    ax_met.set_xticks(range(1, 21))
    ax_met.legend(fontsize=9, loc="lower right")
    ax_met.set_title("Metric Curves", fontsize=12, fontweight="bold")
    ax_met.grid(True, alpha=0.3)

    # --- Row 2: LLM saturation score ---
    ax_llm = fig.add_subplot(gs[2])
    if llm_levels:
        ax_llm.plot(
            llm_levels,
            llm_scores,
            "k-*",
            markersize=8,
            linewidth=2,
            label="LLM Saturation Score (4B)",
        )
        ax_llm.axhline(
            0.8, color="gray", linestyle=":", alpha=0.6, label="Threshold 0.8"
        )
        ax_llm.axvline(
            oracle_stop, color="orange", linestyle="--", linewidth=1.5, alpha=0.7
        )
    ax_llm.set_xlabel("Degradation Level", fontsize=11)
    ax_llm.set_ylabel("Saturation Score", fontsize=11)
    ax_llm.set_xlim(4.5, 20.5)
    ax_llm.set_ylim(-0.05, 1.05)
    ax_llm.set_xticks(range(5, 21))
    ax_llm.legend(fontsize=9, loc="lower right")
    ax_llm.set_title("LLM Saturation Score", fontsize=12, fontweight="bold")
    ax_llm.grid(True, alpha=0.3)

    # --- Row 3: Text blocks for predicted SGs at selected levels ---
    ax_text = fig.add_subplot(gs[3])
    ax_text.axis("off")

    text_lines: list[str] = []
    for lv in text_levels:
        pred = pred_texts[lv]
        objs = pred.get("objects", [])
        rels = pred.get("relationships", [])

        # Sort by confidence, take top 6
        objs_sorted = sorted(objs, key=lambda o: o.get("confidence", 0), reverse=True)[
            :6
        ]
        rels_sorted = sorted(rels, key=lambda r: r.get("confidence", 0), reverse=True)[
            :6
        ]

        obj_str = ", ".join(
            f"{o['label']}({o.get('confidence', 0):.2f})" for o in objs_sorted
        )
        rel_str = "\n    ".join(
            f"{r['subject_label']} --{r['predicate']}-- {r['object_label']} "
            f"({r.get('confidence', 0):.2f})"
            for r in rels_sorted
        )
        if not rel_str:
            rel_str = "(none)"

        reason = llm_reasons.get(lv, "N/A")
        if len(reason) > 160:
            reason = reason[:157] + "..."

        text_lines.append(f"--- Level {lv} ---")
        text_lines.append(f"  Objects (top 6): {obj_str}")
        text_lines.append("  Relations (top 6):")
        text_lines.append(f"    {rel_str}")
        text_lines.append(f"  LLM reason: {wrap_text(reason, width=100)}")
        text_lines.append("")

    full_text = "\n".join(text_lines)
    ax_text.text(
        0.02,
        0.98,
        full_text,
        transform=ax_text.transAxes,
        fontsize=7.5,
        fontfamily="monospace",
        verticalalignment="top",
        horizontalalignment="left",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f0f0f0", "alpha": 0.8},
    )
    ax_text.set_title(
        "Predicted Scene Graphs at Selected Levels", fontsize=12, fontweight="bold"
    )

    # --- Row 4: GT scene graph ---
    ax_gt = fig.add_subplot(gs[4])
    ax_gt.axis("off")

    gt_obj_labels = sorted({o["label"] for o in gt["objects"]})
    gt_rels = gt["relationships"][:10]
    gt_rel_str = "\n    ".join(
        f"{r['subject_label']} --{r['predicate']}-- {r['object_label']}"
        for r in gt_rels
    )
    gt_text = (
        f"GT Objects ({len(gt['objects'])}): {', '.join(gt_obj_labels)}\n\n"
        f"GT Relationships ({len(gt['relationships'])} total, showing up to 10):\n"
        f"    {gt_rel_str}"
    )
    ax_gt.text(
        0.02,
        0.95,
        gt_text,
        transform=ax_gt.transAxes,
        fontsize=8,
        fontfamily="monospace",
        verticalalignment="top",
        horizontalalignment="left",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#ffffdd", "alpha": 0.8},
    )
    ax_gt.set_title("Ground-Truth Scene Graph", fontsize=12, fontweight="bold")

    # --- Save ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------


def make_summary_figure(
    selected: list[tuple[str, int, str]],
    rel_metrics_path: Path,
    out_path: Path,
) -> None:
    """Overlay RelSGScore curves for the 3 selected images."""
    rel_rows = load_csv_dict(rel_metrics_path)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#2196F3", "#FF9800", "#E91E63"]
    markers = ["o", "s", "D"]
    labels_prefix = ["EARLY", "MID", "LATE"]

    for idx, (iid, stop, desc) in enumerate(selected):
        rows = [r for r in rel_rows if r["image_id"] == iid]
        rows.sort(key=lambda r: int(r["level"]))
        levels = [int(r["level"]) for r in rows]
        scores = [safe_float(r["RelSGScore"]) for r in rows]

        # Filter None
        xf = [x for x, y in zip(levels, scores) if y is not None]
        yf = [y for y in scores if y is not None]

        ax.plot(
            xf,
            yf,
            color=colors[idx],
            marker=markers[idx],
            markersize=5,
            linewidth=2,
            label=f"[{labels_prefix[idx]}] {iid} (stop={stop})",
        )
        # Mark the oracle stop
        ax.axvline(
            stop,
            color=colors[idx],
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
        )
        # Add a marker at the stop level if score exists
        stop_scores = [y for x, y in zip(xf, yf) if x == stop]
        if stop_scores:
            ax.plot(
                stop,
                stop_scores[0],
                marker="X",
                markersize=14,
                color=colors[idx],
                zorder=5,
            )

    ax.axhline(0.8, color="gray", linestyle=":", alpha=0.5, label="Threshold 0.8")
    ax.set_xlabel("Degradation Level", fontsize=12)
    ax.set_ylabel("RelSGScore (vs L20)", fontsize=12)
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(range(1, 21))
    ax.legend(fontsize=10, loc="lower right")
    ax.set_title(
        "RelSGScore Saturation: Early vs Mid vs Late", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = load_config()

    # Paths
    gt_dir = resolve(cfg["paths"]["gt_scene_graphs"])
    pred_dir = resolve(cfg["paths"]["egtr_predictions"])
    degraded_dir = resolve(cfg["paths"]["degraded_images"])
    metrics_root = resolve(cfg["paths"]["metrics"])
    figures_root = resolve(cfg["paths"]["figures"])
    llm_dir = REPO_ROOT / "outputs" / "llm_ablation" / "qwen3.5-4b_h5_rel_on_conf_on"

    metrics_path = metrics_root / "all_metrics.csv"
    rel_metrics_path = metrics_root / "relative_metrics.csv"
    oracle_stop_path = metrics_root / "relative_oracle_stop.csv"

    print("=" * 60)
    print("Selecting 3 representative images")
    print("=" * 60)

    selected = select_representative_images(oracle_stop_path, gt_dir, pred_dir)

    print()
    print("=" * 60)
    print("Generating panel figures")
    print("=" * 60)

    for image_id, stop, desc in selected:
        print(f"\n  Processing image {image_id} ...")
        out_path = figures_root / f"representative_{image_id}.png"
        make_panel_figure(
            image_id=image_id,
            oracle_stop=stop,
            gt_desc=desc,
            degraded_dir=degraded_dir,
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            metrics_path=metrics_path,
            rel_metrics_path=rel_metrics_path,
            llm_dir=llm_dir,
            out_path=out_path,
        )

    print()
    print("=" * 60)
    print("Generating summary figure")
    print("=" * 60)

    summary_path = figures_root / "representative_summary.png"
    make_summary_figure(selected, rel_metrics_path, summary_path)

    print()
    print("Done!")


if __name__ == "__main__":
    main()
