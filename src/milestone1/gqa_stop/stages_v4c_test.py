"""Plan §21: freeze on the v4c dev visual subset, one-shot test evaluation, and
the pre-registered extension on reader-wrong test questions (§21.2).

Every method is (score column, k). θ per accuracy target is the v1 rule (min
mean level with accuracy >= target; ties: higher accuracy, larger θ), chosen on
dev only. Logged variants are frozen too but never used for a claim.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .client import RequestEngine, monitor_from_config
from .common import (
    atomic_write_json,
    read_id_csv,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
)
from .context import Context, require, table_to_csv
from .devgrid import fixed_curve
from .evaluate import add_correctness, cluster_bootstrap_indices, hull_value
from .explore import add_soft_existence, best_setting
from .inference import InferenceRunner
from .logprob_v4 import LogprobRunner
from .policy import (
    ACCURACY_TOLERANCE,
    ConsistencyVariant,
    correctness_matrix,
    question_scores,
    replay_stop_consecutive,
    theta_from_json,
    threshold_grid,
)
from .stages_model import LOGPROB, full_table, provenance, require_screening_match

FROZEN_FILE = "frozen_policy.json"
V1_ARM_DIR = "v1arm"
EXT_DIR = "ext_unanswerable"
PRIMARY = "checklist_v4_min"
# name, score column, k, role
METHODS: tuple[tuple[str, str, int, str], ...] = (
    ("checklist_v4_min", "checklist_v4_min", 1, "main"),
    ("direct_v4", "direct_v4", 1, "comparator"),
    ("checklist_v1_min", "checklist_v1_min", 1, "comparator (fair v1)"),
    ("direct_v1_verbalized", "direct_score", 1, "comparator"),
    ("checklist_v4_mean", "checklist_v4_mean", 1, "logged"),
    ("pans_min", "pans_min", 1, "logged"),
    ("pans_mean", "pans_mean", 1, "logged"),
    ("rel_min", "rel_min", 1, "logged"),
    ("rel_mean", "rel_mean", 1, "logged"),
    ("direct_pans", "direct_pans", 1, "logged"),
    ("direct_rel", "direct_rel", 1, "logged"),
    ("soft_existence", "soft_existence", 1, "logged (exploratory)"),
    ("checklist_v4_min_k2", "checklist_v4_min", 2, "logged (exploratory)"),
    ("direct_v4_k2", "direct_v4", 2, "logged (exploratory)"),
    ("soft_existence_k2", "soft_existence", 2, "logged (exploratory)"),
)
COMPARISONS = (
    ("checklist_v4_min", "checklist_v1_min"),
    ("checklist_v4_min", "direct_v4"),
    ("checklist_v4_min", "direct_v1_verbalized"),
)
MAIN = ("checklist_v4_min", "direct_v4", "checklist_v1_min", "direct_v1_verbalized")
PERCENTILES = (2.5, 97.5)


def json_theta(theta: float) -> Any:
    return "inf" if math.isinf(theta) else round(float(theta), 4)


def targets(policy_cfg: Mapping[str, Any]) -> list[float]:
    return [
        float(policy_cfg["primary_dev_accuracy_target"]),
        *map(float, policy_cfg["secondary_dev_accuracy_targets"]),
    ]


def with_v1(table: pd.DataFrame, v1_table: Path) -> pd.DataFrame:
    table = add_soft_existence(table)
    if "checklist_v1_min" in table.columns:
        return table
    v1 = read_id_csv(v1_table)[["question_id", "level", "checklist_min"]]
    merged = table.merge(
        v1.rename(columns={"checklist_min": "checklist_v1_min"}),
        on=["question_id", "level"],
        how="left",
    )
    require(merged["checklist_v1_min"].notna().all(), f"{v1_table} lacks rows")
    return merged


# ---- freeze on dev -------------------------------------------------------------------------
def freeze_v4c(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("freeze")
    require(ctx.family == LOGPROB, "freeze_v4c is for the A13 logprob_item run")
    path = ctx.paths.out(FROZEN_FILE)
    require(not path.exists(), f"{path} exists: the policy is frozen once (§21.1)")
    info = (ctx.state.get("infer") or {}).get("dev", {})
    require(
        bool(info.get("complete")) and info.get("family") == ctx.family,
        "dev inference must be complete",
    )
    table = with_v1(
        read_id_csv(ctx.paths.out("level_table_dev.csv")),
        ctx.paths.out(f"{V1_ARM_DIR}/level_table_dev.csv"),
    )
    require(set(table["split"].astype(str)) == {"dev"}, "freeze uses dev rows only")
    if ctx.dry_run:
        ctx.echo(f"[dry-run] freeze on {table['question_id'].nunique()} dev questions")
        return {}
    grid = threshold_grid(ctx.cfg["policy"])
    curve = fixed_curve(table)
    selections: dict[str, Any] = {}
    for target in targets(ctx.cfg["policy"]):
        key = f"{target:.2f}"
        chosen: dict[str, Any] = {}
        for name, column, k, role in METHODS:
            best = best_setting(table, column, k, grid, target)
            chosen[name] = (
                {
                    "column": column,
                    "k": k,
                    "role": role,
                    "theta": None,
                    "feasible": False,
                }
                if best is None
                else {
                    "column": column,
                    "k": k,
                    "role": role,
                    "feasible": True,
                    "theta": json_theta(best["theta"]),
                    "dev_accuracy": best["accuracy"],
                    "dev_mean_level": best["mean_level"],
                    "dev_stopped_wrong_ids": best["stopped_wrong_ids"],
                }
            )
        chosen["fixed_dev"] = next(
            (t for t in sorted(curve) if curve[t] >= target - ACCURACY_TOLERANCE),
            max(curve),
        )
        selections[key] = chosen
    subset = ctx.cfg["paths"]["subset_file"]
    frozen = {
        "frozen_at": utc_now(),
        "spec": "plan §21.1 (v4c, clean decoding A13)",
        "primary_target": f"{targets(ctx.cfg['policy'])[0]:.2f}",
        "methods": [m[0] for m in METHODS],
        "main_methods": list(MAIN),
        "logged_methods_note": "role 'logged' variants are frozen for the record only",
        "selections": selections,
        "dev_fixed_curve": {str(t): curve[t] for t in sorted(curve)},
        "dev_question_ids": sorted(table["question_id"].unique().tolist()),
        "subset_file": subset,
        "subset_sha256": sha256_file(ctx.paths.repo_root / subset),
        "decoding": dict(ctx.settings.decoding_options),
        **provenance(ctx),
    }
    atomic_write_json(path, frozen)
    ctx.state.set("freeze", {"done": True, "time": utc_now()})
    primary = selections[frozen["primary_target"]]
    ctx.log(
        "frozen (§21.1): "
        + ", ".join(f"{m}={primary[m]['theta']}" for m in MAIN)
        + f", fixed_dev={primary['fixed_dev']}"
    )
    return frozen


# ---- test: v1 arm on the test visual subset ---------------------------------------------------
def v1arm_test(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("v1arm_test")
    require(ctx.paths.out(FROZEN_FILE).exists(), "freeze before any test inference")
    ctx.require_frozen_match(read_json(ctx.paths.out(FROZEN_FILE)))
    require_screening_match(ctx, "test")
    questions = ctx.selected("test")
    levels = list(range(1, ctx.n_levels + 1))
    if ctx.dry_run:
        ctx.echo(f"[dry-run] v1 arm test: {len(questions)} x {len(levels)}")
        return {}
    checklist = ctx.cfg["checklist"]
    runner = InferenceRunner(
        ctx.engine(),
        ctx.series,
        int(checklist["min_items"]),
        int(checklist["max_items"]),
    )
    rows = [row for q in questions for row in runner.run_question(q, levels)]
    table = add_correctness(
        pd.DataFrame(rows), ctx.gt([q["question_id"] for q in questions])
    )
    table_to_csv(table, ctx.paths.out(f"{V1_ARM_DIR}/level_table_test.csv"))
    ctx.state.set("v1arm_test", {"done": True, "time": utc_now()})
    ctx.log(f"v1 arm test: {len(questions)} questions x {len(levels)} levels")
    ctx.note_reuse()
    return {"questions": len(questions)}


# ---- evaluation with frozen θ ---------------------------------------------------------------
def frozen_stops(table: pd.DataFrame, entry: Mapping[str, Any]) -> dict[str, Any]:
    theta = theta_from_json(entry["theta"])
    variant = ConsistencyVariant(theta=theta, k=int(entry["k"]))  # θ may be float/inf
    return {
        q: replay_stop_consecutive(s, variant)
        for q, s in question_scores(table, entry["column"]).items()
    }


def stop_frame(table: pd.DataFrame, frozen: Mapping[str, Any]) -> pd.DataFrame:
    correct = correctness_matrix(table)
    rows = []
    for target, chosen in frozen["selections"].items():
        for name in frozen["methods"]:
            entry = chosen[name]
            if not entry.get("feasible"):
                continue
            for q, d in frozen_stops(table, entry).items():
                rows.append(
                    {
                        "target": target,
                        "method": name,
                        "role": entry["role"],
                        "theta": entry["theta"],
                        "k": entry["k"],
                        "question_id": q,
                        "stop_level": d.level,
                        "forced20": d.never_met,
                        "correct": bool(correct[q][d.level - 1]),
                    }
                )
        fixed = int(chosen["fixed_dev"])
        for q in correct:
            rows.append(
                {
                    "target": target,
                    "method": "fixed_dev",
                    "role": "comparator",
                    "theta": None,
                    "k": None,
                    "question_id": q,
                    "stop_level": fixed,
                    "forced20": False,
                    "correct": bool(correct[q][fixed - 1]),
                }
            )
    return pd.DataFrame(rows)


def summarize(
    stops: pd.DataFrame, curve: Mapping[int, float], n_levels: int
) -> pd.DataFrame:
    rows = []
    for (target, method), g in stops.groupby(["target", "method"], sort=False):
        acc, mean_level = g["correct"].mean(), g["stop_level"].mean()
        wrong = g[(~g["correct"]) & (~g["forced20"]) & (g["stop_level"] < n_levels)]
        rows.append(
            {
                "target": target,
                "method": method,
                "role": g["role"].iloc[0],
                "theta": g["theta"].iloc[0],
                "k": g["k"].iloc[0],
                "n": len(g),
                "accuracy": acc,
                "mean_level": mean_level,
                "median_level": float(g["stop_level"].median()),
                "stop20_rate": float((g["stop_level"] == n_levels).mean()),
                "stopped_but_wrong": len(wrong),
                "stopped_wrong_ids": " ".join(sorted(wrong["question_id"])),
                "accuracy_minus_hull": acc - hull_value(curve, mean_level),
            }
        )
    return pd.DataFrame(rows)


def bootstrap(
    stops: pd.DataFrame,
    table: pd.DataFrame,
    target: str,
    repeats: int,
    seed: int,
    pairs: Sequence[tuple[str, str]] = COMPARISONS,
) -> dict[str, Any]:
    """Paired image-cluster bootstrap at one target (hull recomputed per draw)."""
    qids = sorted(table["question_id"].unique())
    groups = table.groupby("question_id")["group_id"].first().astype(str)
    correct_m = correctness_matrix(table)
    correct = np.array([correct_m[q] for q in qids], dtype=bool)
    sub = stops[stops["target"] == target]
    arrays = {
        m: g.set_index("question_id").loc[qids, "stop_level"].to_numpy()
        for m, g in sub.groupby("method")
    }

    def metrics(idx: np.ndarray) -> dict[str, float]:
        c = correct[idx]
        curve = {t + 1: float(c[:, t].mean()) for t in range(c.shape[1])}
        out: dict[str, float] = {}
        for m, stop in arrays.items():
            s = stop[idx]
            acc = float(c[np.arange(len(s)), s - 1].mean())
            out[f"{m}.accuracy"], out[f"{m}.mean_level"] = acc, float(s.mean())
            out[f"{m}.hull_gap"] = acc - hull_value(curve, float(s.mean()))
        for a, b in pairs:
            if a in arrays and b in arrays:
                out[f"{a}-{b}.accuracy_diff"] = (
                    out[f"{a}.accuracy"] - out[f"{b}.accuracy"]
                )
                out[f"{a}-{b}.mean_level_diff"] = (
                    out[f"{a}.mean_level"] - out[f"{b}.mean_level"]
                )
        return out

    point = metrics(np.arange(len(qids)))
    draws: dict[str, list[float]] = {k: [] for k in point}
    for idx in cluster_bootstrap_indices([groups[q] for q in qids], repeats, seed):
        for key, value in metrics(idx).items():
            draws[key].append(value)
    return {
        key: {
            "estimate": point[key],
            "ci_low": float(np.percentile(draws[key], PERCENTILES[0])),
            "ci_high": float(np.percentile(draws[key], PERCENTILES[1])),
        }
        for key in point
    }


def attribute_breakdown(
    stops: pd.DataFrame,
    table: pd.DataFrame,
    target: str,
    methods: Sequence[str] = MAIN,
) -> pd.DataFrame:
    meta = table.groupby("question_id")[["attribute", "relate_count"]].first()
    sub = stops[
        (stops["target"] == target) & stops["method"].isin([*methods, "fixed_dev"])
    ]
    sub = sub.join(meta, on="question_id")
    rows = []
    for by in ("attribute", "relate_count"):
        for (value, method), g in sub.groupby([by, "method"]):
            rows.append(
                {
                    "group_by": by,
                    "value": value,
                    "method": method,
                    "n": len(g),
                    "accuracy": g["correct"].mean(),
                    "mean_level": g["stop_level"].mean(),
                }
            )
    return pd.DataFrame(rows)


def figures(
    summary: pd.DataFrame,
    stops: pd.DataFrame,
    curve: Mapping[int, float],
    primary: str,
    out: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    xs = sorted(curve)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        xs, [curve[x] for x in xs], "o-", color="gray", ms=3, label="Fixed-t (test)"
    )
    grid = np.linspace(1, xs[-1], 191)
    ax.plot(
        grid,
        [hull_value(curve, b) for b in grid],
        "--",
        color="black",
        lw=1,
        label="concave hull",
    )
    for row in summary[summary["method"].isin(MAIN)].itertuples():
        primary_row = row.target == primary
        ax.scatter(
            row.mean_level,
            row.accuracy,
            s=55 if primary_row else 20,
            marker="o" if primary_row else "x",
            label=f"{row.method}@{row.target}" if primary_row else None,
        )
    ax.set_xlabel("mean stop level")
    ax.set_ylabel("accuracy (common reader)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "test_meanlevel_accuracy.png", dpi=150)
    plt.close(fig)
    sub = stops[(stops["target"] == primary) & stops["method"].isin(MAIN)]
    fig, axes = plt.subplots(1, len(MAIN), figsize=(4 * len(MAIN), 3), squeeze=False)
    for ax, method in zip(axes[0], MAIN, strict=True):
        ax.hist(
            sub[sub["method"] == method]["stop_level"], bins=range(1, 22), align="left"
        )
        ax.set_title(method, fontsize=8)
        ax.set_xlabel("stop level")
    fig.tight_layout()
    fig.savefig(out / "test_stop_hist.png", dpi=150)
    plt.close(fig)


def evaluate_v4c(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("evaluate")
    frozen_path = ctx.paths.out(FROZEN_FILE)
    require(frozen_path.exists(), "no frozen policy")
    frozen = read_json(frozen_path)
    ctx.require_frozen_match(frozen)
    info = (ctx.state.get("infer") or {}).get("test", {})
    require(bool(info.get("complete")), "test inference must be complete")
    require(
        bool((ctx.state.get("v1arm_test") or {}).get("done")),
        "run v1 arm on test first",
    )
    if ctx.dry_run:
        ctx.echo("[dry-run] evaluate v4c test with the frozen policy")
        return {}
    table = with_v1(
        full_table(ctx, "test"), ctx.paths.out(f"{V1_ARM_DIR}/level_table_test.csv")
    )
    table_to_csv(table, ctx.paths.out("level_table_test.csv"))
    curve = fixed_curve(table)
    stops = stop_frame(table, frozen)
    summary = summarize(stops, curve, ctx.n_levels)
    primary = frozen["primary_target"]
    boot = bootstrap(
        stops,
        table,
        primary,
        int(ctx.cfg["evaluation"]["bootstrap_repeats"]),
        int(ctx.cfg["evaluation"]["bootstrap_seed"]),
    )
    stops.to_csv(ctx.paths.out("stops.csv"), index=False)
    summary.to_csv(ctx.paths.out("method_summary.csv"), index=False)
    comps = pd.DataFrame(
        [
            {"comparison": k.rsplit(".", 1)[0], "metric": k.rsplit(".", 1)[1], **v}
            for k, v in boot.items()
        ]
    )
    comps.to_csv(ctx.paths.out("paired_comparisons.csv"), index=False)
    hist = (
        stops[stops["target"] == primary]
        .groupby(["method", "stop_level"])
        .agg(
            n_stopped=("correct", "size"),
            n_wrong=("correct", lambda s: int((~s).sum())),
        )
        .reset_index()
    )
    hist.to_csv(ctx.paths.out("test_stop_hist.csv"), index=False)
    attribute_breakdown(stops, table, primary).to_csv(
        ctx.paths.out("test_attribute_breakdown.csv"), index=False
    )
    figures(summary, stops, curve, primary, ctx.paths.out("figures"))
    result = {
        "n_questions": int(table["question_id"].nunique()),
        "n_images": int(table["group_id"].nunique()),
        "primary_target": primary,
        "bootstrap": {
            "unit": "image group",
            "repeats": int(ctx.cfg["evaluation"]["bootstrap_repeats"]),
            "seed": int(ctx.cfg["evaluation"]["bootstrap_seed"]),
        },
        "test_fixed_curve": {str(t): curve[t] for t in sorted(curve)},
        "statistics": boot,
    }
    atomic_write_json(ctx.paths.out("bootstrap_summary.json"), result)
    ctx.state.set("evaluate", {"done": True, "time": utc_now()})
    ctx.log(f"v4c test evaluated once: {result['n_questions']} questions")
    return result


# ---- §21.2 extension: reader-wrong test candidates --------------------------------------------
def extension_questions(ctx: Context) -> list[dict[str, Any]]:
    wrong = {
        r["question_id"]
        for r in read_jsonl(ctx.paths.selection("selection_results.jsonl"))
        if r["split"] == "test" and r["outcome"] == "incorrect"
    }
    return [r for r in ctx.candidates("test") if r["question_id"] in wrong]


def extension(ctx: Context) -> dict[str, Any]:
    """All methods x 20 levels on reader-wrong test candidates, frozen θ (§21.2)."""
    ctx.set_stage("ext_unanswerable")
    frozen_path = ctx.paths.out(FROZEN_FILE)
    require(frozen_path.exists(), "the extension uses the frozen θ (§21.2)")
    frozen = read_json(frozen_path)
    ctx.require_frozen_match(frozen)
    questions = extension_questions(ctx)
    levels = list(range(1, ctx.n_levels + 1))
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] extension: {len(questions)} reader-wrong test questions x 20"
        )
        return {"questions": len(questions)}
    ctx.verify_live_digest()
    out = ctx.paths.out(EXT_DIR)
    engine = RequestEngine(
        ctx.settings,
        ctx.prompts,
        ctx.transport_factory(ctx.settings),
        out,
        monitor_from_config(ctx.cfg),
    )
    engine.stage = "ext_unanswerable"
    checklist = ctx.cfg["checklist"]
    lo, hi = int(checklist["min_items"]), int(checklist["max_items"])
    v4 = LogprobRunner(engine, ctx.series, lo, hi)
    v1 = InferenceRunner(engine, ctx.series, lo, hi)
    gt = ctx.gt([q["question_id"] for q in questions])
    v4_rows, v1_rows = [], []
    for q in questions:
        v4_rows += v4.run_question(q, levels)
        v1_rows += v1.run_question(q, levels)
    v1_table = add_correctness(pd.DataFrame(v1_rows), gt)
    table_to_csv(v1_table, out / "v1_level_table.csv")
    table = add_correctness(pd.DataFrame(v4_rows), gt)
    table = with_v1(table, out / "v1_level_table.csv")
    table_to_csv(table, out / "level_table.csv")
    result = evaluate_extension(table, frozen, ctx.cfg["evaluation"], ctx.n_levels, out)
    ctx.state.set(
        "ext_unanswerable",
        {"done": True, "time": utc_now(), "questions": len(questions)},
    )
    ctx.log(f"§21.2 extension: {len(questions)} reader-wrong test questions")
    return result


def evaluate_extension(
    table: pd.DataFrame,
    frozen: Mapping[str, Any],
    eval_cfg: Mapping[str, Any],
    n_levels: int,
    out: Path,
) -> dict:
    """False-stop rate, accuracy at the stop, mean level; paired bootstrap."""
    primary = frozen["primary_target"]
    chosen = frozen["selections"][primary]
    correct = correctness_matrix(table)
    qids = sorted(correct)
    groups = table.groupby("question_id")["group_id"].first().astype(str)
    arrays, rows = {}, []
    for name in frozen["methods"]:
        entry = chosen[name]
        if not entry.get("feasible"):
            continue
        decisions = frozen_stops(table, entry)
        stop = np.array([decisions[q].level for q in qids])
        false = np.array([decisions[q].level < n_levels for q in qids])
        hit = np.array([correct[q][decisions[q].level - 1] for q in qids])
        arrays[name] = (stop, false, hit)
        rows.append(
            {
                "method": name,
                "role": entry["role"],
                "theta": entry["theta"],
                "k": entry["k"],
                "n": len(qids),
                "false_stop_rate": float(false.mean()),
                "accuracy_at_stop": float(hit.mean()),
                "accuracy_at_early_stops": float(hit[false].mean())
                if false.any()
                else None,
                "mean_level": float(stop.mean()),
            }
        )
    reader_l20 = float(np.mean([correct[q][-1] for q in qids]))
    pairs = (
        ("checklist_v4_min", "direct_v4"),
        ("checklist_v4_min", "checklist_v1_min"),
        ("checklist_v4_min", "direct_v1_verbalized"),
    )
    boot_keys: dict[str, list[float]] = {}
    point: dict[str, float] = {}
    idx_all = np.arange(len(qids))
    for idx in [
        idx_all,
        *cluster_bootstrap_indices(
            [groups[q] for q in qids],
            int(eval_cfg["bootstrap_repeats"]),
            int(eval_cfg["bootstrap_seed"]),
        ),
    ]:
        values = {}
        for a, b in pairs:
            if a in arrays and b in arrays:
                values[f"{a}-{b}.false_stop_diff"] = float(
                    arrays[a][1][idx].mean() - arrays[b][1][idx].mean()
                )
                values[f"{a}-{b}.mean_level_diff"] = float(
                    arrays[a][0][idx].mean() - arrays[b][0][idx].mean()
                )
        if idx is idx_all:
            point = values
            boot_keys = {k: [] for k in values}
        else:
            for k, v in values.items():
                boot_keys[k].append(v)
    stats = {
        k: {
            "estimate": point[k],
            "ci_low": float(np.percentile(v, PERCENTILES[0])),
            "ci_high": float(np.percentile(v, PERCENTILES[1])),
        }
        for k, v in boot_keys.items()
    }
    pd.DataFrame(rows).to_csv(out / "ext_method_summary.csv", index=False)
    result = {
        "label": "§21.2 pre-registered extension (exploratory)",
        "primary_target": primary,
        "n_questions": len(qids),
        "n_images": int(groups.nunique()),
        "reader_correct_at_L20": reader_l20,
        "methods": rows,
        "paired": stats,
    }
    atomic_write_json(out / "ext_summary.json", result)
    return result
