"""Shared plotting helpers for the relative-saturation (stop_c09) figures.

The canonical "ideal stop" is `stop_c09`: the first resolution level at which
the *core* RelSGScore (recall of L20's top-N high-confidence objects/triplets)
reaches 0.9.  Both the standalone per-image figure (script 15) and the
representative panels (script 17) draw the same curve via `draw_relative_curve`
so the two stay visually consistent.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# Method/reference colours and styles (shared across scripts).
REF_THRESHOLD = 0.9
COLOR_CURVE = "#2A7F4F"
COLOR_REF = "#1B7837"  # ideal stop (stop_c09)
COLOR_OBJCONF = "#E69F00"
COLOR_RELSAT = "#7B3FA0"
COLOR_HYBRID = "#D11149"

# 4b LLM-SGG ablation slug used for RelSat / Hybrid.
BASE_4B_SLUG = "qwen3.5-4b_h5_rel_on_conf_on"
HYBRID_4B_VARIANT = "hybrid_4b_k2"
BASELINE_COL = "baseline_0.35"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load_core_oracle(
    metrics_root: Path, max_level: int
) -> tuple[dict[int, dict[int, float]], dict[int, int]]:
    """Load per-image core RelSGScore trajectory and stop_c09 from topk oracle.

    Returns (core_sg_map[image_id][level] -> score, stop_c09_map[image_id]).
    """
    path = metrics_root / "topk_oracle" / "oracle_topk_per_image.csv"
    core_sg: dict[int, dict[int, float]] = {}
    stop_c09: dict[int, int] = {}
    for row in _read_csv(path):
        iid = int(row["image_id"])
        stop_c09[iid] = int(row["stop_c09"])
        per_level: dict[int, float] = {}
        for lvl in range(1, max_level + 1):
            val = row.get(f"rel_sgscore_L{lvl:02d}", "")
            per_level[lvl] = (
                float(val) if val not in ("", "NaN", "nan") else float("nan")
            )
        core_sg[iid] = per_level
    return core_sg, stop_c09


def load_method_stops(metrics_root: Path, max_level: int) -> dict[str, dict[int, int]]:
    """Load the three 4b stop predictors: ObjConf, RelSat, Hybrid."""
    objconf = {
        int(r["image_id"]): int(float(r[BASELINE_COL]))
        for r in _read_csv(metrics_root / "baseline_stop.csv")
    }
    relsat = {
        int(r["image_id"]): int(r["stop_k2"])
        for r in _read_csv(metrics_root / "llm_ablation_k2_stop.csv")
        if r["config_slug"] == BASE_4B_SLUG
    }
    hybrid = {
        int(r["image_id"]): int(r["stop_level"])
        for r in _read_csv(metrics_root / "hybrid_stop.csv")
        if r["variant"] == HYBRID_4B_VARIANT
    }
    return {"objconf": objconf, "relsat": relsat, "hybrid": hybrid}


def draw_relative_curve(
    ax: plt.Axes,
    levels: list[int],
    core_sg_vals: list[float],
    stop_c09: int,
    objconf: int,
    relsat: int,
    hybrid: int,
    *,
    max_level: int = 20,
    title: str | None = None,
    compact: bool = False,
) -> None:
    """Draw the core-RelSGScore trajectory with the ideal stop and 3 predictors.

    Overlapping stop lines (RelSat and Hybrid coincide for most images) stay
    visible: Hybrid is drawn thick underneath, RelSat thin dashed on top.
    """
    fs_label = 8 if compact else 11
    fs_tick = 7 if compact else 9
    fs_legend = 6 if compact else 8
    fs_title = 8 if compact else 12
    lw_curve = 1.8 if compact else 2.2
    ms = 3 if compact else 4

    ax.plot(
        levels,
        core_sg_vals,
        "-D",
        color=COLOR_CURVE,
        markersize=ms,
        linewidth=lw_curve,
        label="RelSGScore (core)",
        zorder=2,
    )

    # 0.9 saturation threshold guide
    ax.axhline(
        REF_THRESHOLD,
        color="gray",
        linestyle=":",
        linewidth=1.0,
        alpha=0.7,
        label="Threshold 0.9",
        zorder=1,
    )

    # Vertical stop lines are drawn widest-first so that coincident stops stay
    # visible as concentric bands (ObjConf and Hybrid/RelSat frequently land on
    # the same level as the ideal stop). Widths: ObjConf > Ideal > Hybrid > RelSat.
    ax.axvline(
        objconf,
        color=COLOR_OBJCONF,
        linestyle="-",
        linewidth=3.4,
        alpha=0.85,
        label="ObjConf",
        zorder=2,
    )
    ax.axvline(
        stop_c09,
        color=COLOR_REF,
        linestyle="-",
        linewidth=2.3,
        alpha=0.95,
        label="Ideal stop (stop_c09)",
        zorder=3,
    )
    ax.axvline(
        hybrid,
        color=COLOR_HYBRID,
        linestyle="-",
        linewidth=1.6,
        alpha=0.95,
        label="Hybrid (4b)",
        zorder=4,
    )
    ax.axvline(
        relsat,
        color=COLOR_RELSAT,
        linestyle=(0, (4, 2)),
        linewidth=1.1,
        alpha=0.95,
        label="RelSat (4b)",
        zorder=5,
    )

    ax.set_xlabel("Resolution level", fontsize=fs_label)
    ax.set_ylabel("RelSGScore (core)", fontsize=fs_label)
    ax.set_xlim(0.5, max_level + 0.5)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.tick_params(labelsize=fs_tick)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=fs_legend, loc="upper left", framealpha=0.9)
    if title:
        ax.set_title(title, fontsize=fs_title, fontweight="bold")


def figure_kwargs(**overrides: Any) -> dict[str, Any]:
    """Default single-column paper figure kwargs (3.5in wide, 300 dpi)."""
    base = {"figsize": (3.5, 2.7), "dpi": 300}
    base.update(overrides)
    return base
