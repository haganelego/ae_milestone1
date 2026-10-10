"""Plan §21/§22/§23 test protocol on the v22 run (user decision 2026-10-10).

1. ``freeze_v22``: θ per target / method copied from the §22 dev-102 grid
   (``dev_targets.csv``), cross-checked by recomputation; immutable afterwards.
2. Test inference uses the existing stages (``v1arm --split test``, ``infer --split test``).
3. ``evaluate_v22``: one evaluation with the frozen θ (no tuning on test).
4. ``firing_test``: §23 on the reader-wrong test candidates (``ext_test_wrong/``).
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
from .explore import best_setting
from .firing_v23 import firing_frame, run_tables, wrong_questions
from .firing_v23 import paired as firing_paired
from .firing_v23 import summarize as firing_summary
from .policy import theta_from_json, threshold_grid
from .stages_model import full_table, provenance
from .stages_v4c_test import (
    MAIN,
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
        bool((ctx.state.get("v1arm_test") or {}).get("done")), "run v1arm --split test"
    )
    if ctx.dry_run:
        ctx.echo("[dry-run] evaluate v22 test with the frozen policy")
        return {}
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
    curve_frame = curves(table, grid, "test (exploratory θ sweep)")
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
        "n_questions": n_q,
        "n_images": n_img,
        "test_fixed_curve": {str(t): v for t, v in curve.items()},
        "curve_methods_exploratory": list(CURVE_METHODS),
        "bootstrap": {"unit": "image group", "repeats": reps, "seed": seed},
        "statistics": boot,
    }
    atomic_write_json(ctx.paths.out("test_summary.json"), result)
    ctx.state.set("evaluate", {"done": True, "time": utc_now()})
    ctx.log(f"v22 test evaluated once: {n_q} questions")
    return result


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


def firing_test(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("ext_test_wrong")
    frozen = frozen_checked(ctx)
    questions = wrong_questions(ctx, "test")
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] §23 test firing: {len(questions)} reader-wrong test questions"
        )
        return {"questions": len(questions)}
    out = ctx.paths.out(EXT_DIR)
    table = run_tables(ctx, questions, out)
    fires = firing_frame(table, frozen_firing_thetas(frozen), ctx.n_levels)
    meta = table.groupby("question_id")[
        ["attribute", "relate_count", "group_id"]
    ].first()
    fires = fires.join(meta[["attribute", "relate_count"]], on="question_id")
    fires.to_csv(out / "firing_per_question.csv", index=False)
    firing_summary(fires).to_csv(out / "firing_summary.csv", index=False)
    ev = ctx.cfg["evaluation"]
    groups = meta["group_id"].astype(str).to_dict()
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
