"""Plan §21/§22/§23 test protocol on the v22 run (user decision 2026-10-10).

1. ``freeze_v22``: θ per target / method copied from the §22 dev-102 grid
   (``dev_targets.csv``), cross-checked by recomputation; immutable afterwards.
2. Test inference uses the existing stages (``v1arm --split test``, ``infer --split test``).
3. ``evaluate_v22``: one evaluation with the frozen θ (no tuning on test).
4. ``infer_test_wrong`` (model calls) then ``firing_test`` (cache only): §23 on the
   reader-wrong test candidates (``ext_test_wrong/``).

v1 arm opt-out (``--skip-v1arm``, user decision 2026-10-11): the v1 arm
(checklist_v1 and verbalised Direct) is not run on test. Its methods are dropped
from the frozen selections at evaluation time (the frozen file is not edited) and
its score columns are NaN; the outputs record the decision.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .common import atomic_write_json, read_id_csv, read_json, sha256_file, utc_now
from .context import Context, require, table_to_csv
from .devgrid import fixed_curve
from .devgrid_v22 import ALL_METHODS, CURVE_METHODS, curves, figure, load_table
from .evaluate import cluster_bootstrap_indices
from .explore import best_setting
from .firing_v23 import firing_frame, run_tables, without_v1_columns, wrong_questions
from .firing_v23 import paired as firing_paired
from .firing_v23 import summarize as firing_summary
from .policy import theta_from_json, threshold_grid
from .stages_model import full_table, provenance
from .stages_v4c_test import (
    MAIN,
    PERCENTILES,
    V1_ARM_DIR,
    attribute_breakdown,
    bootstrap,
    figures,
    json_theta,
    stop_frame,
    summarize,
    targets,
    with_v1,
)

FROZEN_FILE = "frozen_policy.json"
EXT_DIR = "ext_test_wrong"
PRIMARY = "0.95"
V1_METHODS = ("checklist_v1_min", "checklist_v1_mean", "direct_v1_verbalized")
V1_SKIP_NOTE = "v1 arm skipped by user decision 2026-10-11"
V1_SKIPPED = (
    "v1 arm: planner v1 + checklist v2 (checklist_v1_min, checklist_v1_mean)",
    "verbalised Direct (role 'direct', direct_v1_verbalized), including the call "
    "inside the v4 runner; confirmed by team-lead 2026-10-11",
)
CV_CAVEAT = (
    "dev single-θ selection loses about 0.01-0.02 accuracy out of fold "
    "(5-fold image-grouped CV x 10 seeds, box_attrtheta_cv.csv): direct_v4 at the "
    "0.95 target gives 0.934 out of fold"
)
PANS = (
    "direct_pans",
    "pans_min",
)  # H_pans: dev-derived, pre-registered before test eval
PANS_PAIRS = tuple((p, m) for p in PANS for m in ("direct_v4", "checklist_v4_min"))
PAIRS = (
    ("checklist_v4_min", "direct_v4"),
    ("checklist_v4_min", "checklist_v1_min"),
    ("direct_v4", "checklist_v1_min"),
    ("checklist_v4_min", "direct_v1_verbalized"),
    *((m, "fixed_dev") for m in (*MAIN, *PANS)),
    *PANS_PAIRS,
)
FIRING_PAIRS = (
    ("checklist_v4_min", "direct_v4"),
    ("checklist_v4_min", "checklist_v1_min"),
    ("direct_v4", "checklist_v1_min"),
    ("checklist_v4_min", "direct_v1_verbalized"),
    ("direct_v4", "direct_v1_verbalized"),
    *PANS_PAIRS,
)


# ---- 1. freeze ------------------------------------------------------------------------------
def freeze_v22(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("freeze")
    path = ctx.paths.out(FROZEN_FILE)
    require(not path.exists(), f"{path} exists: the policy is frozen once")
    require(
        bool((ctx.state.get("devgrid_v22") or {}).get("done")), "run `curves` first"
    )
    grid22 = pd.read_csv(
        ctx.paths.out("dev_targets.csv"), dtype={"target": str, "theta": str}
    )
    grid22 = grid22[grid22["subset"] == "v22"].set_index(["target", "method"])
    dev = load_table(ctx)
    grid = threshold_grid(ctx.cfg["policy"])
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] freeze v22 on {dev['question_id'].nunique()} dev questions"
        )
        return {}
    selections: dict[str, Any] = {}
    for target in targets(ctx.cfg["policy"]):
        key = f"{target:.2f}"
        chosen: dict[str, Any] = {}
        for name, column, k, role in ALL_METHODS:
            row = grid22.loc[(key, name)]
            best = best_setting(dev, column, k, grid, target)
            require(
                best is not None and str(row["feasible"]) == "True", f"{key} {name}"
            )
            assert best is not None
            require(
                json_theta(best["theta"]) == theta_json(row["theta"]),
                f"{key} {name}: recomputed θ {best['theta']} != dev_targets {row['theta']}",
            )
            chosen[name] = {
                "column": column,
                "k": k,
                "role": role,
                "feasible": True,
                "theta": json_theta(best["theta"]),
                "dev_accuracy": best["accuracy"],
                "dev_mean_level": best["mean_level"],
                "dev_stopped_wrong_ids": best["stopped_wrong_ids"],
            }
        chosen["fixed_dev"] = int(grid22.loc[(key, "fixed_dev"), "mean_level"])
        selections[key] = chosen
    subset = ctx.cfg["paths"]["subset_file"]
    frozen = {
        "frozen_at": utc_now(),
        "spec": "plan §21/§22/§23 test protocol on visapp_gqa_v22 (user decision 2026-10-10)",
        "theta_source": "outputs/gqa_stop/visapp_gqa_v22/dev_targets.csv (subset v22, dev 102)",
        "primary_target": PRIMARY,
        "methods": [m[0] for m in ALL_METHODS],
        "main_methods": list(MAIN),
        "logged_methods_note": "roles 'logged' / 'logged (exploratory)' are recorded only",
        "selections": selections,
        "dev_fixed_curve": {str(t): v for t, v in fixed_curve(dev).items()},
        "dev_question_ids": sorted(dev["question_id"].unique().tolist()),
        "subset_file": subset,
        "subset_sha256": sha256_file(ctx.paths.repo_root / subset),
        "decoding": dict(ctx.settings.decoding_options),
        "think": False,
        "cv_caveat": CV_CAVEAT,
        **provenance(ctx),
    }
    atomic_write_json(path, frozen)
    ctx.state.set(
        "freeze", {"done": True, "time": utc_now(), "sha256": sha256_file(path)}
    )
    primary = selections[PRIMARY]
    ctx.log(
        "frozen (v22): "
        + ", ".join(f"{m}={primary[m]['theta']}" for m in MAIN)
        + f", fixed_dev={primary['fixed_dev']}"
    )
    return frozen


def theta_json(value: Any) -> Any:
    return "inf" if str(value) == "inf" else round(float(value), 4)


def without_v1(frozen: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of the frozen policy without the v1-arm methods (file untouched)."""
    return {
        **frozen,
        "methods": [m for m in frozen["methods"] if m not in V1_METHODS],
        "main_methods": [m for m in frozen["main_methods"] if m not in V1_METHODS],
        "selections": {
            t: {m: e for m, e in chosen.items() if m not in V1_METHODS}
            for t, chosen in frozen["selections"].items()
        },
    }


def v1_status(ctx: Context) -> str:
    return V1_SKIP_NOTE if ctx.skip_v1arm else "run"


def skipped(ctx: Context) -> list[str]:
    return list(V1_SKIPPED) if ctx.skip_v1arm else []


def frozen_checked(ctx: Context) -> dict[str, Any]:
    frozen = read_json(ctx.paths.out(FROZEN_FILE))
    ctx.require_frozen_match(frozen)
    expected = (ctx.state.get("freeze") or {}).get("sha256")
    require(
        expected == sha256_file(ctx.paths.out(FROZEN_FILE)),
        "frozen policy was modified",
    )
    return frozen


# ---- 3. evaluation --------------------------------------------------------------------------
def test_table(ctx: Context) -> pd.DataFrame:
    if ctx.skip_v1arm:
        return without_v1_columns(full_table(ctx, "test"))
    v1_path = ctx.paths.out(f"{V1_ARM_DIR}/level_table_test.csv")
    table = with_v1(full_table(ctx, "test"), v1_path)
    v1 = read_id_csv(v1_path)[["question_id", "level", "checklist_mean"]]
    return table.merge(
        v1.rename(columns={"checklist_mean": "checklist_v1_mean"}),
        on=["question_id", "level"],
        how="left",
    )


def evaluate_v22(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("evaluate")
    frozen = frozen_checked(ctx)
    require(
        bool(((ctx.state.get("infer") or {}).get("test") or {}).get("complete")),
        "test inference must be complete",
    )
    require(
        ctx.skip_v1arm or bool((ctx.state.get("v1arm_test") or {}).get("done")),
        "run v1arm --split test (or pass --skip-v1arm)",
    )
    require(
        not (ctx.state.get("evaluate") or {}).get("done"),
        "the v22 test evaluation was already run once",
    )
    if ctx.dry_run:
        ctx.echo("[dry-run] evaluate v22 test with the frozen policy")
        return {}
    if ctx.skip_v1arm:
        ctx.log(f"evaluate-v22: {V1_SKIP_NOTE}; v1 methods dropped, columns NaN")
        frozen = without_v1(frozen)
    table = test_table(ctx)
    require(
        set(table["question_id"]) == {r["question_id"] for r in ctx.selected("test")},
        "test table != test visual subset",
    )
    table_to_csv(table, ctx.paths.out("level_table_test_merged.csv"))
    curve = fixed_curve(table)
    stops = stop_frame(table, frozen)
    summary = summarize(stops, curve, ctx.n_levels)
    ev = ctx.cfg["evaluation"]
    reps, seed = int(ev["bootstrap_repeats"]), int(ev["bootstrap_seed"])
    boot = {
        t: bootstrap(stops, table, t, reps, seed, PAIRS) for t in frozen["selections"]
    }
    stops.to_csv(ctx.paths.out("test_stops.csv"), index=False)
    summary.to_csv(ctx.paths.out("test_method_summary.csv"), index=False)
    pd.DataFrame(
        [
            {
                "target": t,
                "quantity": k.rsplit(".", 1)[0],
                "metric": k.rsplit(".", 1)[1],
                **v,
            }
            for t, stats in boot.items()
            for k, v in stats.items()
        ]
    ).to_csv(ctx.paths.out("test_paired_comparisons.csv"), index=False)
    hist = (
        stops.groupby(["target", "method", "stop_level"])
        .agg(
            n_stopped=("correct", "size"),
            n_wrong=("correct", lambda s: int((~s.astype(bool)).sum())),
        )
        .reset_index()
    )
    hist.to_csv(ctx.paths.out("test_stop_hist.csv"), index=False)
    pd.concat(
        [
            attribute_breakdown(stops, table, t, (*MAIN, *PANS)).assign(target=t)
            for t in frozen["selections"]
        ]
    ).to_csv(ctx.paths.out("test_attribute_breakdown.csv"), index=False)
    grid = threshold_grid(ctx.cfg["policy"])
    curve_frame = curves(
        table, grid, "test (exploratory θ sweep)", methods=evaluated(frozen)
    )
    curve_frame.to_csv(ctx.paths.out("test_curves_exploratory.csv"), index=False)
    n_q, n_img = int(table["question_id"].nunique()), int(table["group_id"].nunique())
    figure(
        curve_frame,
        {"test (exploratory θ sweep)": f"{n_q} questions / {n_img} images"},
        ctx.paths.out("figures/test_curves_exploratory.png"),
    )
    figures(summary, stops, curve, PRIMARY, ctx.paths.out("figures"))
    result = {
        "label": "v22 test, frozen θ, evaluated once",
        "time": utc_now(),
        "v1_arm": v1_status(ctx),
        "skipped": skipped(ctx),
        "methods_evaluated": list(frozen["methods"]),
        "n_questions": n_q,
        "n_images": n_img,
        "test_fixed_curve": {str(t): v for t, v in curve.items()},
        "curve_methods_exploratory": list(CURVE_METHODS),
        "bootstrap": {"unit": "image group", "repeats": reps, "seed": seed},
        "statistics": boot,
    }
    atomic_write_json(ctx.paths.out("test_summary.json"), result)
    ctx.state.set(
        "evaluate", {"done": True, "time": utc_now(), "v1_arm": v1_status(ctx)}
    )
    ctx.log(f"v22 test evaluated once: {n_q} questions")
    return result


def evaluated(frozen: Mapping[str, Any]) -> tuple[tuple[str, str, int, str], ...]:
    return tuple(m for m in ALL_METHODS if m[0] in frozen["methods"])


# ---- 4. §23 firing on reader-wrong test candidates ----------------------------------------
def frozen_firing_thetas(
    frozen: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        t: {
            m: {
                "column": e["column"],
                "k": int(e["k"]),
                "role": e["role"],
                "theta": theta_from_json(e["theta"]),
            }
            for m, e in chosen.items()
            if m != "fixed_dev" and e.get("feasible")
        }
        for t, chosen in frozen["selections"].items()
    }


def infer_test_wrong(ctx: Context) -> dict[str, Any]:
    """Model calls for the reader-wrong test set (L1..L20); no evaluation."""
    ctx.set_stage("infer_test_wrong")
    frozen_checked(ctx)
    require(
        bool((ctx.state.get("determinism") or {}).get("passed")),
        "run `detgate` first: the A13 determinism gate has not passed",
    )
    questions = wrong_questions(ctx, "test")
    if ctx.dry_run:
        ctx.echo(f"[dry-run] infer {len(questions)} reader-wrong test questions x 20")
        return {"questions": len(questions)}
    run_tables(ctx, questions, ctx.paths.out(EXT_DIR), ctx.skip_v1arm, stage=ctx.stage)
    ctx.state.set(
        "infer_test_wrong",
        {
            "done": True,
            "time": utc_now(),
            "questions": len(questions),
            "v1_arm": v1_status(ctx),
        },
    )
    ctx.log(f"infer test wrong: {len(questions)} questions x 20 ({v1_status(ctx)})")
    return {"questions": len(questions)}


def firing_rate_ci(
    fires: pd.DataFrame, groups: Mapping[str, str], repeats: int, seed: int
) -> pd.DataFrame:
    """Image-cluster bootstrap 95% CI of each method's firing rate."""
    qids = sorted(groups)
    indices = cluster_bootstrap_indices([groups[q] for q in qids], repeats, seed)
    rows = []
    for (target, method), g in fires.groupby(["target", "method"], sort=False):
        flags = g.set_index("question_id").loc[qids, "fired"].to_numpy(dtype=float)
        draws = [float(flags[i].mean()) for i in indices]
        rows.append(
            {
                "target": target,
                "method": method,
                "firing_ci_low": float(np.percentile(draws, PERCENTILES[0])),
                "firing_ci_high": float(np.percentile(draws, PERCENTILES[1])),
            }
        )
    return pd.DataFrame(rows)


def firing_test(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("ext_test_wrong")
    frozen = frozen_checked(ctx)
    require(
        bool((ctx.state.get("infer_test_wrong") or {}).get("done")),
        "run `infer-test-wrong` first",
    )
    require(
        not (ctx.state.get("ext_test_wrong") or {}).get("done"),
        "the §23 test firing evaluation was already run once",
    )
    questions = wrong_questions(ctx, "test")
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] §23 test firing: {len(questions)} reader-wrong test questions"
        )
        return {"questions": len(questions)}
    if ctx.skip_v1arm:
        frozen = without_v1(frozen)
    out = ctx.paths.out(EXT_DIR)
    table = run_tables(
        ctx, questions, out, ctx.skip_v1arm, cache_only=True, stage=ctx.stage
    )
    fires = firing_frame(table, frozen_firing_thetas(frozen), ctx.n_levels)
    meta = table.groupby("question_id")[
        ["attribute", "relate_count", "group_id"]
    ].first()
    fires = fires.join(meta[["attribute", "relate_count"]], on="question_id")
    fires.to_csv(out / "firing_per_question.csv", index=False)
    ev = ctx.cfg["evaluation"]
    groups = meta["group_id"].astype(str).to_dict()
    reps, seed = int(ev["bootstrap_repeats"]), int(ev["bootstrap_seed"])
    firing_summary(fires).merge(
        firing_rate_ci(fires, groups, reps, seed), on=["target", "method"]
    ).to_csv(out / "firing_summary.csv", index=False)
    firing_paired(
        fires,
        groups,
        int(ev["bootstrap_repeats"]),
        int(ev["bootstrap_seed"]),
        FIRING_PAIRS,
    ).to_csv(out / "firing_paired.csv", index=False)
    hist = (
        fires[fires["fired"]]
        .groupby(["target", "method", "fire_level"])
        .agg(
            n=("question_id", "size"), reader_correct=("reader_correct_at_fire", "sum")
        )
        .reset_index()
    )
    hist.to_csv(out / "firing_level_hist.csv", index=False)
    rows = [
        {
            "target": t,
            "method": m,
            "group_by": by,
            "value": v,
            "n": len(g),
            "firing_rate": float(g["fired"].mean()),
        }
        for by in ("attribute", "relate_count")
        for (t, m, v), g in fires.groupby(["target", "method", by])
    ]
    pd.DataFrame(rows).to_csv(out / "firing_breakdown.csv", index=False)
    by_level = table.groupby("level")["reader_correct"].mean()
    result = {
        "label": "§23 test firing (frozen θ)",
        "time": utc_now(),
        "v1_arm": v1_status(ctx),
        "skipped": skipped(ctx),
        "n_questions": len(questions),
        "n_images": int(meta["group_id"].nunique()),
        "reader_accuracy_by_level": {int(k): float(v) for k, v in by_level.items()},
        "l20_reader_correct": float(
            np.mean(table[table["level"] == ctx.n_levels]["reader_correct"])
        ),
    }
    atomic_write_json(out / "firing_meta.json", result)
    ctx.state.set(
        "ext_test_wrong", {"done": True, "time": utc_now(), "questions": len(questions)}
    )
    ctx.log(f"§23 test firing: {len(questions)} reader-wrong test questions")
    return result
