"""DESCRIPTIVE θ sweep for the v22 test protocol (team-lead request 2026-10-11).

For each method (k = 1) and θ in {0.50, 0.55, ..., 1.00}: accuracy, mean stop
level, Q1 / median / Q3 of the stop level and the stop20 rate on the answerable
set, and the §23 firing rate on the reader-wrong set (score >= θ at some level
1..20; the forced L20 stop is not a firing). Computed on test (251 / 233) and
on dev (102 / 99) side by side.

A θ read off this sweep is NOT a result: θ chosen on test is in-sample. The
pre-registered result is the frozen-θ evaluation (``evaluate-v22`` /
``firing-test``), which must run before this sweep.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import atomic_write_json, read_id_csv, utc_now
from .context import Context, require
from .devgrid_v22 import ALL_METHODS
from .firing_v23 import firing_frame
from .policy import (
    ConsistencyVariant,
    correctness_matrix,
    question_scores,
    replay_stop_consecutive,
)

OUT_DIR = "sweep_v22"
LABEL = (
    "DESCRIPTIVE θ sweep: θ read off the test sweep is not a result; the frozen-θ "
    "rows (evaluate-v22 / firing-test) are the pre-registered result"
)
THETA_START, THETA_STEP, THETA_COUNT = 0.50, 0.05, 11
SWEEP_THETAS = tuple(round(THETA_START + THETA_STEP * i, 2) for i in range(THETA_COUNT))
SWEEP_METHODS = ("direct_v4", "checklist_v4_min", "direct_pans", "pans_min")
QUARTILES = (25, 50, 75)
SWEEP_K = 1
# split -> (answerable table, reader-wrong table), relative to the output root
TABLES = {
    "test": ("level_table_test_merged.csv", "ext_test_wrong/level_table.csv"),
    "dev": ("level_table_dev_merged.csv", "ext_dev_wrong/level_table.csv"),
}
AXES = (
    ("accuracy", "accuracy (answerable set)"),
    (
        "mean_level",
        "stop level (answerable set): mean; test median dotted, test IQR band",
    ),
    ("stop20_rate", "stop20 rate (answerable set)"),
    ("firing_rate", "firing rate (reader-wrong set)"),
)


def method_columns(methods: Sequence[str] = SWEEP_METHODS) -> dict[str, str]:
    columns = {name: column for name, column, _k, _role in ALL_METHODS}
    return {m: columns[m] for m in methods}


def stop_levels(table: pd.DataFrame, column: str, theta: float) -> dict[str, int]:
    variant = ConsistencyVariant(theta=theta, k=SWEEP_K)
    return {
        q: replay_stop_consecutive(s, variant).level
        for q, s in question_scores(table, column).items()
    }


def answerable_row(
    table: pd.DataFrame, column: str, theta: float, n_levels: int
) -> dict[str, Any]:
    correct = correctness_matrix(table)
    stops = stop_levels(table, column, theta)
    levels = np.array([stops[q] for q in sorted(stops)], dtype=float)
    hits = [bool(correct[q][stops[q] - 1]) for q in sorted(stops)]
    q1, median, q3 = np.percentile(levels, QUARTILES)
    return {
        "n": len(levels),
        "accuracy": float(np.mean(hits)),
        "mean_level": float(levels.mean()),
        "q1_level": float(q1),
        "median_level": float(median),
        "q3_level": float(q3),
        "stop20_rate": float((levels == n_levels).mean()),
        "stopped_but_wrong": int(
            sum(
                1 for h, lv in zip(hits, levels, strict=True) if not h and lv < n_levels
            )
        ),
    }


def sweep_frame(
    answerable: pd.DataFrame,
    wrong: pd.DataFrame,
    split: str,
    n_levels: int,
    methods: Sequence[str] = SWEEP_METHODS,
    thetas: Sequence[float] = SWEEP_THETAS,
) -> pd.DataFrame:
    columns = method_columns(methods)
    firing_thetas = {
        f"{theta:.2f}": {
            m: {"column": c, "k": SWEEP_K, "role": "sweep", "theta": theta}
            for m, c in columns.items()
        }
        for theta in thetas
    }
    fires = firing_frame(wrong, firing_thetas, n_levels)
    rate = fires.groupby(["target", "method"])["fired"].agg(["mean", "size"])
    rows = []
    for method, column in columns.items():
        for theta in thetas:
            key = f"{theta:.2f}"
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "theta": theta,
                    **answerable_row(answerable, column, theta, n_levels),
                    "n_wrong_set": int(rate.loc[(key, method), "size"]),
                    "firing_rate": float(rate.loc[(key, method), "mean"]),
                }
            )
    return pd.DataFrame(rows)


def figures(frames: Mapping[str, pd.DataFrame], out: Path) -> list[Path]:
    """One figure per axis: θ on x, test solid and dev dashed, one colour per method."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    colours = {m: f"C{i}" for i, m in enumerate(SWEEP_METHODS)}
    styles = {"test": "-", "dev": "--"}
    paths = []
    for metric, title in AXES:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for split, frame in frames.items():
            for method, g in frame.groupby("method", sort=False):
                g = g.sort_values("theta")
                ax.plot(
                    g["theta"],
                    g[metric],
                    styles[split],
                    marker="o",
                    ms=3,
                    color=colours[str(method)],
                    label=f"{method} ({split})",
                )
                if metric == "mean_level" and split == "test":
                    ax.fill_between(
                        g["theta"],
                        g["q1_level"],
                        g["q3_level"],
                        color=colours[str(method)],
                        alpha=0.12,
                    )
                    ax.plot(
                        g["theta"],
                        g["median_level"],
                        ":",
                        color=colours[str(method)],
                        lw=1,
                    )
        ax.set_xlabel("θ (k = 1)")
        ax.set_ylabel(metric)
        ax.set_title(f"{title}\nDESCRIPTIVE sweep (not a result)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        path = out / f"sweep_{metric}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def load_split(ctx: Context, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    answerable, wrong = (ctx.paths.out(p) for p in TABLES[split])
    require(answerable.exists(), f"{answerable} missing")
    require(wrong.exists(), f"{wrong} missing")
    return read_id_csv(answerable), read_id_csv(wrong)


def sweep_v22(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("sweep_v22")
    require(
        bool((ctx.state.get("evaluate") or {}).get("done")),
        "run `evaluate-v22` (frozen θ) before the descriptive sweep",
    )
    require(
        bool((ctx.state.get("ext_test_wrong") or {}).get("done")),
        "run `firing-test` (frozen θ) before the descriptive sweep",
    )
    if ctx.dry_run:
        ctx.echo(f"[dry-run] sweep θ {SWEEP_THETAS} x {SWEEP_METHODS} on test and dev")
        return {}
    out = ctx.paths.out(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    sizes: dict[str, dict[str, int]] = {}
    for split in TABLES:
        answerable, wrong = load_split(ctx, split)
        frames[split] = sweep_frame(answerable, wrong, split, ctx.n_levels)
        frames[split].to_csv(out / f"{split}_sweep.csv", index=False)
        sizes[split] = {
            "answerable_questions": int(answerable["question_id"].nunique()),
            "answerable_images": int(answerable["group_id"].nunique()),
            "reader_wrong_questions": int(wrong["question_id"].nunique()),
        }
    figure_paths = figures(frames, out / "figures")
    meta = {
        "label": LABEL,
        "time": utc_now(),
        "thetas": list(SWEEP_THETAS),
        "methods": list(SWEEP_METHODS),
        "k": SWEEP_K,
        "firing_rule": "score >= θ at some level 1..20; forced L20 is not a firing",
        "quartiles": "numpy percentile 25/50/75, linear interpolation",
        "sizes": sizes,
        "figures": [p.name for p in figure_paths],
    }
    atomic_write_json(out / "sweep_meta.json", meta)
    ctx.log(f"sweep-v22 (DESCRIPTIVE) written: {sizes}")
    return meta
