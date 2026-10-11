"""Plan §22: dev devgrid on the enlarged visual subset (no cap) — per-target θ,
full accuracy–mean-level curves, paired bootstrap per target, breakdowns.

Dev only and in-sample: θ is chosen on the same questions it is scored on.
The paired bootstrap resamples image groups with θ held at its dev selection
(conditional on θ; selection noise is not included).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import atomic_write_json, read_id_csv, read_json, resolve_path, utc_now
from .context import Context, require, table_to_csv
from .devgrid import fixed_curve
from .evaluate import cluster_bootstrap_indices, hull_value
from .explore import best_setting, replay
from .policy import (
    ACCURACY_TOLERANCE,
    ConsistencyVariant,
    correctness_matrix,
    question_scores,
    replay_stop_consecutive,
    threshold_grid,
)
from .stages_v4c_test import MAIN, METHODS, V1_ARM_DIR, json_theta, targets, with_v1

LOGGED_V1 = ("checklist_v1_mean", "checklist_v1_mean", 1, "logged")
ALL_METHODS = (*METHODS, LOGGED_V1)
CURVE_METHODS = (
    "checklist_v4_min",
    "direct_v4",
    "checklist_v1_min",
    "direct_v1_verbalized",
)
PAIRS = (("checklist_v4_min", "direct_v4"), ("checklist_v4_min", "checklist_v1_min"))
PERCENTILES = (2.5, 97.5)
FULL, REFERENCE = "v22", "v4c_59"


def load_table(ctx: Context) -> pd.DataFrame:
    table = read_id_csv(ctx.paths.out("level_table_dev.csv"))
    table = with_v1(table, ctx.paths.out(f"{V1_ARM_DIR}/level_table_dev.csv"))
    if "checklist_v1_mean" not in table.columns:
        v1 = read_id_csv(ctx.paths.out(f"{V1_ARM_DIR}/level_table_dev.csv"))
        table = table.merge(
            v1[["question_id", "level", "checklist_mean"]].rename(
                columns={"checklist_mean": "checklist_v1_mean"}
            ),
            on=["question_id", "level"],
            how="left",
        )
    return table


def stops_for(table: pd.DataFrame, column: str, theta: float, k: int) -> dict[str, int]:
    variant = ConsistencyVariant(theta=theta, k=k)
    return {
        q: replay_stop_consecutive(s, variant).level
        for q, s in question_scores(table, column).items()
    }


def curves(
    table: pd.DataFrame,
    grid: Sequence[float],
    subset: str,
    methods: Sequence[tuple[str, str, int, str]] = ALL_METHODS,
) -> pd.DataFrame:
    rows = []
    for name, column, k, role in methods:
        for theta in grid:
            r = replay(table, column, theta, k)
            rows.append(
                {
                    "subset": subset,
                    "method": name,
                    "role": role,
                    "k": k,
                    "theta": json_theta(theta),
                    "accuracy": r["accuracy"],
                    "mean_level": r["mean_level"],
                    "stop20_rate": r["stop20_rate"],
                    "stopped_but_wrong": r["stopped_but_wrong"],
                }
            )
    curve = fixed_curve(table)
    rows += [
        {
            "subset": subset,
            "method": "fixed",
            "role": "reference",
            "k": None,
            "theta": None,
            "accuracy": curve[t],
            "mean_level": float(t),
            "stop20_rate": float(t == max(curve)),
            "stopped_but_wrong": None,
        }
        for t in sorted(curve)
    ]
    return pd.DataFrame(rows)


def selections(
    table: pd.DataFrame,
    grid: Sequence[float],
    target_list: Sequence[float],
    subset: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    curve = fixed_curve(table)
    rows, chosen = [], {}
    for target in target_list:
        key = f"{target:.2f}"
        chosen[key] = {}
        for name, column, k, role in ALL_METHODS:
            best = best_setting(table, column, k, grid, target)
            if best is None:
                rows.append(
                    {
                        "subset": subset,
                        "target": key,
                        "method": name,
                        "role": role,
                        "k": k,
                        "feasible": False,
                    }
                )
                continue
            chosen[key][name] = (column, best["theta"], k)
            rows.append(
                {
                    "subset": subset,
                    "target": key,
                    "method": name,
                    "role": role,
                    "k": k,
                    "feasible": True,
                    "theta": json_theta(best["theta"]),
                    "accuracy": best["accuracy"],
                    "mean_level": best["mean_level"],
                    "stop20_rate": best["stop20_rate"],
                    "stopped_but_wrong": best["stopped_but_wrong"],
                    "stopped_wrong_ids": " ".join(best["stopped_wrong_ids"]),
                    "accuracy_minus_hull": best["accuracy"]
                    - hull_value(curve, best["mean_level"]),
                }
            )
        fixed = next(
            (t for t in sorted(curve) if curve[t] >= target - ACCURACY_TOLERANCE), None
        )
        rows.append(
            {
                "subset": subset,
                "target": key,
                "method": "fixed_dev",
                "role": "reference",
                "feasible": fixed is not None,
                "theta": None,
                "accuracy": None if fixed is None else curve[fixed],
                "mean_level": fixed,
            }
        )
    return pd.DataFrame(rows), chosen


def paired(
    table: pd.DataFrame,
    chosen: Mapping[str, Mapping[str, tuple]],
    repeats: int,
    seed: int,
    subset: str,
) -> pd.DataFrame:
    correct = correctness_matrix(table)
    qids = sorted(correct)
    groups = table.groupby("question_id")["group_id"].first().astype(str)
    indices = cluster_bootstrap_indices([groups[q] for q in qids], repeats, seed)
    c = np.array([correct[q] for q in qids], dtype=bool)
    rows = []
    for target, methods in chosen.items():
        for a, b in PAIRS:
            if a not in methods or b not in methods:
                continue
            level = {}
            for m in (a, b):
                column, theta, k = methods[m]
                s = stops_for(table, column, theta, k)
                level[m] = np.array([s[q] for q in qids])
            hit = {m: c[np.arange(len(qids)), level[m] - 1] for m in (a, b)}
            dl = level[a] - level[b]
            da = hit[a].astype(float) - hit[b].astype(float)
            boot_l = [float(dl[i].mean()) for i in indices]
            boot_a = [float(da[i].mean()) for i in indices]
            rows.append(
                {
                    "subset": subset,
                    "target": target,
                    "comparison": f"{a} - {b}",
                    "mean_level_diff": float(dl.mean()),
                    "mean_level_ci_low": float(np.percentile(boot_l, PERCENTILES[0])),
                    "mean_level_ci_high": float(np.percentile(boot_l, PERCENTILES[1])),
                    "accuracy_diff": float(da.mean()),
                    "accuracy_ci_low": float(np.percentile(boot_a, PERCENTILES[0])),
                    "accuracy_ci_high": float(np.percentile(boot_a, PERCENTILES[1])),
                    "n_questions": len(qids),
                    "n_images": int(groups.nunique()),
                }
            )
    return pd.DataFrame(rows)


def breakdown(
    table: pd.DataFrame, chosen: Mapping[str, Mapping[str, tuple]]
) -> pd.DataFrame:
    correct = correctness_matrix(table)
    meta = table.groupby("question_id")[["attribute", "relate_count"]].first()
    rows = []
    for target, methods in chosen.items():
        for name in MAIN:
            if name not in methods:
                continue
            column, theta, k = methods[name]
            s = stops_for(table, column, theta, k)
            frame = pd.DataFrame({"question_id": list(s), "stop": list(s.values())})
            frame["correct"] = [correct[q][lv - 1] for q, lv in s.items()]
            frame = frame.join(meta, on="question_id")
            for by in ("attribute", "relate_count"):
                for value, g in frame.groupby(by):
                    rows.append(
                        {
                            "target": target,
                            "method": name,
                            "group_by": by,
                            "value": value,
                            "n": len(g),
                            "accuracy": float(g["correct"].mean()),
                            "mean_level": float(g["stop"].mean()),
                            "stop20_rate": float((g["stop"] == 20).mean()),
                        }
                    )
    return pd.DataFrame(rows)


def figure(curve_frame: pd.DataFrame, counts: Mapping[str, str], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    out.parent.mkdir(parents=True, exist_ok=True)
    subsets = list(dict.fromkeys(curve_frame["subset"]))
    fig, axes = plt.subplots(
        1, len(subsets), figsize=(6.5 * len(subsets), 4.6), squeeze=False, sharey=True
    )
    styles = {
        "checklist_v4_min": "C0",
        "direct_v4": "C3",
        "checklist_v1_min": "C2",
        "direct_v1_verbalized": "C1",
    }
    for ax, subset in zip(axes[0], subsets, strict=True):
        frame = curve_frame[curve_frame["subset"] == subset]
        fixed = frame[frame["method"] == "fixed"].sort_values("mean_level")
        ax.plot(
            fixed["mean_level"],
            fixed["accuracy"],
            "o-",
            color="gray",
            ms=3,
            label="fixed level",
        )
        curve = dict(zip(fixed["mean_level"], fixed["accuracy"], strict=True))
        xs = np.linspace(1, 20, 191)
        ax.plot(
            xs,
            [hull_value(curve, x) for x in xs],
            "--",
            color="black",
            lw=0.8,
            label="fixed concave hull",
        )
        for name in CURVE_METHODS:
            pts = frame[frame["method"] == name].sort_values("mean_level")
            if pts.empty:
                continue
            lw, alpha = (1.6, 1.0) if name in CURVE_METHODS[:2] else (1.0, 0.6)
            ax.plot(
                pts["mean_level"],
                pts["accuracy"],
                ".-",
                color=styles[name],
                lw=lw,
                ms=3,
                alpha=alpha,
                label=name,
            )
        for t in (0.90, 0.95, 0.97, 0.99):
            ax.axhline(t, color="lightgray", lw=0.6, zorder=0)
        ax.set_title(f"dev {subset}: {counts.get(subset, '')}", fontsize=9)
        ax.set_xlabel("mean stop level (θ grid)")
        ax.set_xlim(0.5, 20.5)
        ax.set_ylim(0.3, 1.01)
    axes[0][0].set_ylabel("accuracy (common reader)")
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def devgrid_v22(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("devgrid_v22")
    info = (ctx.state.get("infer") or {}).get("dev", {})
    require(bool(info.get("complete")), "dev inference must be complete")
    require(
        not ctx.paths.out("frozen_policy.json").exists(), "§22 dev grid is pre-freeze"
    )
    table = load_table(ctx)
    require(
        set(table["question_id"]) == {r["question_id"] for r in ctx.selected("dev")},
        "level table != enlarged dev subset",
    )
    reference_ids = set(
        read_json(
            resolve_path(ctx.paths.repo_root, ctx.cfg["v22"]["reference_subset"])
        )["dev"]
    )
    reference = table[table["question_id"].isin(reference_ids)]
    require(
        reference["question_id"].nunique() == len(reference_ids),
        "reference ids missing",
    )
    if ctx.dry_run:
        ctx.echo(f"[dry-run] devgrid v22: {table['question_id'].nunique()} questions")
        return {}
    grid = threshold_grid(ctx.cfg["policy"])
    target_list = targets(ctx.cfg["policy"])
    ev = ctx.cfg["evaluation"]
    repeats, seed = int(ev["bootstrap_repeats"]), int(ev["bootstrap_seed"])
    out: dict[str, Any] = {
        "label": "plan §22 dev (in-sample θ; no freeze)",
        "time": utc_now(),
    }
    sel_frames, pair_frames, curve_frames = [], [], []
    chosen_full: dict[str, Any] = {}
    for subset, frame in ((FULL, table), (REFERENCE, reference)):
        sel, chosen = selections(frame, grid, target_list, subset)
        sel_frames.append(sel)
        pair_frames.append(paired(frame, chosen, repeats, seed, subset))
        curve_frames.append(curves(frame, grid, subset))
        out[subset] = {
            "n_questions": int(frame["question_id"].nunique()),
            "n_images": int(frame["group_id"].nunique()),
            "fixed_curve": {str(t): v for t, v in fixed_curve(frame).items()},
        }
        if subset == FULL:
            chosen_full = chosen
    table_to_csv(table, ctx.paths.out("level_table_dev_merged.csv"))
    pd.concat(sel_frames).to_csv(ctx.paths.out("dev_targets.csv"), index=False)
    pd.concat(pair_frames).to_csv(
        ctx.paths.out("dev_paired_by_target.csv"), index=False
    )
    curve_frame = pd.concat(curve_frames)
    curve_frame.to_csv(ctx.paths.out("dev_curves.csv"), index=False)
    breakdown(table, chosen_full).to_csv(
        ctx.paths.out("dev_breakdown.csv"), index=False
    )
    counts = {
        k: f"{out[k]['n_questions']} questions / {out[k]['n_images']} images"
        for k in (FULL, REFERENCE)
    }
    figure(curve_frame, counts, ctx.paths.out("figures/dev_curves.png"))
    atomic_write_json(ctx.paths.out("devgrid_v22_summary.json"), out)
    ctx.state.set("devgrid_v22", {"done": True, "time": utc_now()})
    ctx.log(f"§22 devgrid: {out[FULL]['n_questions']} dev questions (+ reference 59)")
    return out
