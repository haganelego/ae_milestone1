"""Exploratory probe: θ-sweep and per-type tables (markdown to stdout).

Uses the unchanged v27 evaluator functions on the probe level table.
Usage: uv run python scripts/gqa_probe_multi_rel_tables.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from milestone1.gqa_stop.v27_eval import arrays_from_table, summarize

PROBE = Path("/home/kanaoka/usr/github/milestone_1/outputs/gqa_stop/probe_multi_rel")
V27_STAR = {
    "B1": 0.90,
    "P_min": 0.80,
    "P_mean": 0.90,
    "A1": 0.75,
    "A2": 0.95,
    "A3": 1.0,
}
METHODS = ("B1", "P_min", "P_mean", "A1", "A2")
GRID = [round(0.05 * i, 2) for i in range(21)]
N_LEVELS = 20


def fmt(x: float) -> str:
    return "—" if not np.isfinite(x) else f"{x:.3f}"


def main() -> None:
    table = pd.read_csv(PROBE / "level_table_dev.csv", dtype={"question_id": str})
    arrays = arrays_from_table(table, N_LEVELS)
    print("## sweep: balanced / answerable / unanswerable / mean L (all)")
    print("| θ | " + " | ".join(METHODS) + " |")
    print("|---" * (len(METHODS) + 1) + "|")
    for t in GRID:
        cells = []
        for m in METHODS:
            s = summarize(arrays, m, t)
            cells.append(
                f"{s['balanced_strict']:.2f} / {s['answerable_acc']:.2f} / "
                f"{s['unanswerable_strict']:.2f} / {s['mean_level_all']:.1f}"
            )
        print(f"| {t:.2f} | " + " | ".join(cells) + " |")
    a3 = summarize(arrays, "A3", 1.0)
    print(
        f"A3: {a3['balanced_strict']:.3f} / {a3['answerable_acc']:.3f} / "
        f"{a3['unanswerable_strict']:.3f} / {a3['mean_level_all']:.2f}"
    )
    print("\n## at v27 frozen θ*, all and per type")
    kinds = ["all", *sorted(set(arrays.attributes))]
    print(
        "| set | n ans/unans | "
        + " | ".join(f"{m} (θ {V27_STAR[m]:.2f})" for m in (*METHODS, "A3"))
        + " |"
    )
    print("|---" * (len(METHODS) + 3) + "|")
    for kind in kinds:
        part = arrays if kind == "all" else arrays.subset(arrays.attributes == kind)
        n_a = int(part.answerable.sum())
        cells = []
        for m in (*METHODS, "A3"):
            s = summarize(part, m, V27_STAR[m])
            cells.append(
                f"{fmt(s['answerable_acc'])} / {fmt(s['unanswerable_strict'])} / "
                f"{s['mean_level_all']:.1f}"
            )
        print(
            f"| {kind} | {n_a}/{len(part.answerable) - n_a} | "
            + " | ".join(cells)
            + " |"
        )
    print("\n## stop counts at θ 0.80 (same θ): stopped / stopped-correct per type")
    for kind in kinds:
        part = arrays if kind == "all" else arrays.subset(arrays.attributes == kind)
        cells = []
        for m in ("B1", "P_min", "A1"):
            s = summarize(part, m, 0.80)
            n = len(part.answerable)
            stopped = (
                n
                - round(
                    s["never_stop_answerable"] * s["n_answerable"]
                    if s["n_answerable"]
                    else 0
                )
                - round(
                    s["never_stop_unanswerable"] * s["n_unanswerable"]
                    if s["n_unanswerable"]
                    else 0
                )
            )
            cells.append(f"{m} {stopped}/{n}")
        print(kind, cells)


if __name__ == "__main__":
    main()
