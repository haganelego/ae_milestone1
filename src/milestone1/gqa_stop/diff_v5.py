"""Plan §24 C: level-difference saturation signal (role ``diff_v5``).

One request per level ℓ = 2..20 with the level ℓ-1 and level ℓ images in one
message; signal = P(exact "No") in the first token's top_logprobs (0 if absent;
L1 = 0). Rules: diff-only (P(No) >= θ for k consecutive levels) and diff∧v4
(additionally the v4 score >= its §22 θ at target 0.95). Dev 102 + the 99
reader-wrong questions; θ chosen in-sample on the 102.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .client import (
    REPAIR_PROMPT,
    PromptSet,
    RequestEngine,
    RequestSpec,
    monitor_from_config,
)
from .common import atomic_write_json, read_id_csv, utc_now
from .context import Context, require
from .evaluate import cluster_bootstrap_indices
from .explore import best_setting, replay
from .logprob_v4 import TOP_LOGPROBS, _accept_text, relevance_from_logprobs
from .policy import (
    ConsistencyVariant,
    question_scores,
    replay_stop_consecutive,
    threshold_grid,
)
from .stages_v4c_test import targets

ROLE = "diff_v5"
NUM_PREDICT = 8  # as relevance_v4
OUT_DIR = "v5"
PILOT_QUESTIONS = 20
PILOT_LEVELS = (2, 5, 10, 15, 20)
GATE_TARGET = 0.95
V4_GATE_TARGET = "0.95"
V4_SCORES = ("direct_v4", "checklist_v4_min")
PERCENTILES = (2.5, 97.5)


def diff_spec(
    question: Mapping[str, Any],
    level: int,
    previous: bytes,
    current: bytes,
    suffix: str = "",
) -> RequestSpec:
    return RequestSpec(
        ROLE,
        str(question["question_id"]),
        level,
        {"question": str(question["question"])},
        previous,
        suffix=suffix,
        logprobs=TOP_LOGPROBS,
        raw_text=True,
        extra_image_bytes=(current,),
    )


def engine_for(ctx: Context, root: Path) -> RequestEngine:
    ctx.verify_live_digest()
    texts = {
        ROLE: (ctx.paths.prompts_dir / f"{ROLE}.txt").read_text("utf-8"),
        REPAIR_PROMPT: ctx.prompts.texts[REPAIR_PROMPT],
    }
    settings = dataclasses.replace(
        ctx.settings, num_predict={**ctx.settings.num_predict, ROLE: NUM_PREDICT}
    )
    engine = RequestEngine(
        settings,
        PromptSet(texts),
        ctx.transport_factory(settings),
        root,
        monitor_from_config(ctx.cfg),
    )
    engine.stage = "v5_diff"
    return engine


def signal(record: Mapping[str, Any]) -> dict[str, Any]:
    ok = record.get("status") == "valid"
    rel = relevance_from_logprobs(record.get("logprobs") if ok else None)
    return {
        "status": record.get("status"),
        "p_no": rel.p_no,
        "p_yes": rel.p_yes,
        "has_yes_no": rel.has_yes_or_no,
        "first_token": rel.first_token,
        "text": (record.get("parsed") or {}).get("text"),
    }


def run_levels(
    ctx: Context,
    engine: RequestEngine,
    questions: Sequence[Mapping[str, Any]],
    levels: Sequence[int],
) -> pd.DataFrame:
    rows = []
    for q in questions:
        image_id = str(q["image_id"])
        for level in levels:
            spec = diff_spec(
                q,
                level,
                ctx.series.bytes_for(image_id, level - 1),
                ctx.series.bytes_for(image_id, level),
            )
            rows.append(
                {
                    "question_id": q["question_id"],
                    "level": level,
                    **signal(engine.run(spec, _accept_text)),
                }
            )
    return pd.DataFrame(rows)


def pilot(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("v5_diff_pilot")
    questions = ctx.selected("dev")[:PILOT_QUESTIONS]
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] diff pilot: {len(questions)} x {PILOT_LEVELS} + 2 controls"
        )
        return {}
    engine = engine_for(ctx, ctx.paths.out(OUT_DIR))
    frame = run_levels(ctx, engine, questions, PILOT_LEVELS)
    controls = []
    top = ctx.n_levels
    for q in questions:  # two-image sanity: identical pair vs extreme pair
        img = str(q["image_id"])
        for name, a, b in (("same_L20", top, top), ("L1_to_L20", 1, top)):
            spec = diff_spec(
                q,
                top,
                ctx.series.bytes_for(img, a),
                ctx.series.bytes_for(img, b),
                suffix=f"_ctrl_{name}",
            )
            controls.append(
                {
                    "question_id": q["question_id"],
                    "control": name,
                    **signal(engine.run(spec, _accept_text)),
                }
            )
    ctrl = pd.DataFrame(controls)
    valid = float((frame["status"] == "valid").mean())
    yes_no = float(frame["has_yes_no"].mean())
    result = {
        "label": "plan §24 C pilot",
        "time": utc_now(),
        "valid_rate": valid,
        "yes_no_present_rate": yes_no,
        "gates_pass": valid >= GATE_TARGET and yes_no >= GATE_TARGET,
        "p_no_by_level": {
            int(k): float(v) for k, v in frame.groupby("level")["p_no"].mean().items()
        },
        "controls_mean_p_no": {
            k: float(v) for k, v in ctrl.groupby("control")["p_no"].mean().items()
        },
        "first_tokens": {
            str(k): int(v) for k, v in frame["first_token"].value_counts().items()
        },
    }
    out = ctx.paths.out(OUT_DIR)
    frame.to_csv(out / "diff_pilot.csv", index=False)
    ctrl.to_csv(out / "diff_pilot_controls.csv", index=False)
    atomic_write_json(out / "diff_pilot_summary.json", result)
    ctx.log(f"§24 C pilot: {result['gates_pass']}")
    return result


# ---- full run and evaluation --------------------------------------------------------------
def with_diff(
    table: pd.DataFrame, diff: pd.DataFrame, theta22: Mapping[str, float]
) -> pd.DataFrame:
    merged = table.merge(
        diff[["question_id", "level", "p_no"]], on=["question_id", "level"], how="left"
    )
    merged["p_no"] = merged["p_no"].fillna(0.0)  # L1 has no previous image
    for s in V4_SCORES:
        merged[f"diff_and_{s}"] = np.where(merged[s] >= theta22[s], merged["p_no"], 0.0)
    return merged


METHODS = (
    ("diff_only", "p_no", 1),
    ("diff_only_k2", "p_no", 2),
    ("diff_and_direct_v4", "diff_and_direct_v4", 1),
    ("diff_and_direct_v4_k2", "diff_and_direct_v4", 2),
    ("diff_and_checklist_v4_min", "diff_and_checklist_v4_min", 1),
    ("diff_and_checklist_v4_min_k2", "diff_and_checklist_v4_min", 2),
    ("direct_v4", "direct_v4", 1),
    ("checklist_v4_min", "checklist_v4_min", 1),
)


def stops(
    table: pd.DataFrame, column: str, theta: float, k: int, qids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    variant = ConsistencyVariant(theta=theta, k=k)
    scores = question_scores(table, column)
    decisions = [replay_stop_consecutive(scores[q], variant) for q in qids]
    return (
        np.array([d.level for d in decisions]),
        np.array([not d.never_met for d in decisions]),
    )


def evaluate(ctx: Context, dev: pd.DataFrame, wrong: pd.DataFrame, out: Path) -> dict:
    grid = threshold_grid(ctx.cfg["policy"])
    ev = ctx.cfg["evaluation"]
    reps, seed = int(ev["bootstrap_repeats"]), int(ev["bootstrap_seed"])
    sel22 = pd.read_csv(ctx.paths.out("dev_targets.csv"), dtype={"target": str})
    sel22 = sel22[sel22["subset"] == "v22"].set_index(["target", "method"])
    qd = sorted(dev["question_id"].unique())
    qw = sorted(wrong["question_id"].unique())
    gd = dev.groupby("question_id")["group_id"].first().astype(str)
    gw = wrong.groupby("question_id")["group_id"].first().astype(str)
    idx_d = cluster_bootstrap_indices([gd[q] for q in qd], reps, seed)
    idx_w = cluster_bootstrap_indices([gw[q] for q in qw], reps, seed)
    correct = (
        dev.pivot(index="question_id", columns="level", values="reader_correct")
        .loc[qd]
        .to_numpy(dtype=bool)
    )
    rows, curves = [], []
    for name, column, k in METHODS:
        curves += [
            {
                "method": name,
                "theta": t,
                **{
                    a: b
                    for a, b in replay(dev, column, t, k).items()
                    if a != "stopped_wrong_ids"
                },
            }
            for t in grid
        ]
        for target in targets(ctx.cfg["policy"]):
            key = f"{target:.2f}"
            best = best_setting(dev, column, k, grid, target)
            if best is None:
                rows.append({"method": name, "target": key, "feasible": False})
                continue
            stop, _ = stops(dev, column, best["theta"], k, qd)
            _, fired = stops(wrong, column, best["theta"], k, qw)
            d_theta = float(sel22.loc[(key, "direct_v4"), "theta"])
            d_stop, _ = stops(dev, "direct_v4", d_theta, 1, qd)
            _, d_fired = stops(wrong, "direct_v4", d_theta, 1, qw)
            dl = stop - d_stop
            df = fired.astype(float) - d_fired.astype(float)
            hit = correct[np.arange(len(qd)), stop - 1]
            rows.append(
                {
                    "method": name,
                    "target": key,
                    "feasible": True,
                    "k": k,
                    "theta": best["theta"],
                    "accuracy": float(hit.mean()),
                    "mean_level": float(stop.mean()),
                    "stop20_rate": float((stop == 20).mean()),
                    "stopped_but_wrong": best["stopped_but_wrong"],
                    "firing_rate_99": float(fired.mean()),
                    "mean_level_diff_vs_direct_v4": float(dl.mean()),
                    "mean_level_ci": [
                        float(np.percentile([dl[i].mean() for i in idx_d], p))
                        for p in PERCENTILES
                    ],
                    "firing_diff_vs_direct_v4": float(df.mean()),
                    "firing_ci": [
                        float(np.percentile([df[i].mean() for i in idx_w], p))
                        for p in PERCENTILES
                    ],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "diff_targets.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "diff_curves.csv", index=False)
    reference = pd.read_csv(ctx.paths.out("dev_curves.csv"))
    figure(pd.DataFrame(curves), out, reference[reference["subset"] == "v22"])
    return {"selections": rows}


def figure(
    curves: pd.DataFrame, out: Path, reference: pd.DataFrame | None = None
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    colors = {
        "direct_v4": "C3",
        "checklist_v4_min": "C0",
        "diff_only": "C2",
        "diff_only_k2": "C8",
        "diff_and_direct_v4": "C1",
        "diff_and_checklist_v4_min": "C4",
    }
    for name, color in colors.items():
        pts = curves[curves["method"] == name].sort_values("mean_level")
        ax.plot(pts["mean_level"], pts["accuracy"], ".-", ms=3, color=color, label=name)
    if reference is not None:
        from .boxes_v5 import add_fixed_hull

        fixed = reference[reference["method"] == "fixed"].sort_values("mean_level")
        ax.plot(
            fixed["mean_level"],
            fixed["accuracy"],
            "o-",
            ms=3,
            color="gray",
            label="fixed level",
        )
        add_fixed_hull(ax, reference)
    for t in (0.90, 0.95, 0.97, 0.99):
        ax.axhline(t, color="lightgray", lw=0.6, zorder=0)
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0.3, 1.01)
    ax.set_xlabel("mean stop level (dev 102, θ grid)")
    ax.set_ylabel("accuracy (common reader)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "diff_curves.png", dpi=150)
    plt.close(fig)


def run_diff(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("v5_diff")
    out = ctx.paths.out(OUT_DIR)
    pilot_summary = out / "diff_pilot_summary.json"
    require(pilot_summary.exists(), "run the diff pilot first")
    from .common import read_json

    require(bool(read_json(pilot_summary)["gates_pass"]), "diff pilot gates failed")
    root = ctx.paths.output_root
    dev = read_id_csv(root / "level_table_dev_merged.csv")
    wrong = read_id_csv(root / "ext_dev_wrong" / "level_table.csv")
    levels = list(range(2, ctx.n_levels + 1))
    if ctx.dry_run:
        n = dev["question_id"].nunique() + wrong["question_id"].nunique()
        ctx.echo(f"[dry-run] diff: {n} questions x {len(levels)} levels")
        return {}
    dev_q = {r["question_id"]: r for r in ctx.selected("dev")}
    from .firing_v23 import wrong_questions

    wrong_q = {r["question_id"]: r for r in wrong_questions(ctx, "dev")}
    diff_dev = run_levels(
        ctx,
        engine_for(ctx, out),
        [dev_q[q] for q in sorted(dev["question_id"].unique())],
        levels,
    )
    diff_wrong = run_levels(
        ctx,
        engine_for(ctx, root / "ext_dev_wrong"),
        [wrong_q[q] for q in sorted(wrong["question_id"].unique())],
        levels,
    )
    diff_dev.to_csv(out / "diff_levels_dev.csv", index=False)
    diff_wrong.to_csv(root / "ext_dev_wrong" / "diff_levels.csv", index=False)
    sel22 = pd.read_csv(root / "dev_targets.csv", dtype={"target": str})
    sel22 = sel22[(sel22["subset"] == "v22") & (sel22["target"] == V4_GATE_TARGET)]
    theta22 = {s: float(sel22.set_index("method").loc[s, "theta"]) for s in V4_SCORES}
    dev_t, wrong_t = (
        with_diff(dev, diff_dev, theta22),
        with_diff(wrong, diff_wrong, theta22),
    )
    stats = {
        "valid_rate": float(
            (pd.concat([diff_dev, diff_wrong])["status"] == "valid").mean()
        ),
        "yes_no_rate": float(pd.concat([diff_dev, diff_wrong])["has_yes_no"].mean()),
        "p_no_by_level_dev": {
            int(k): float(v)
            for k, v in diff_dev.groupby("level")["p_no"].mean().items()
        },
        "p_no_by_level_wrong": {
            int(k): float(v)
            for k, v in diff_wrong.groupby("level")["p_no"].mean().items()
        },
        "v4_gate_theta": theta22,
    }
    result = {
        "label": "plan §24 C (dev; in-sample θ)",
        "time": utc_now(),
        **stats,
        **evaluate(ctx, dev_t, wrong_t, out),
    }
    atomic_write_json(out / "diff_summary.json", result)
    ctx.log("§24 C: diff_v5 evaluated")
    return result
