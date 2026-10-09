"""Plan §24 D: clean-decoding replays of v3 (verbalised item confidence) and v2
(blind identification). Dev 102 only.

D1 rebuilds checklist_min from the cached observer_v3 records' verbalised
`confidence` under the v1 gate (no new calls). D2 re-runs planner_v2 + identify
(prompts unchanged) under the A13 decoding block; attr_blind / recompose are not run.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .client import REPAIR_PROMPT, PromptSet, RequestEngine, monitor_from_config
from .common import atomic_write_json, read_id_csv, utc_now
from .consistency import (
    IDENTIFY_ROLE,
    PLANNER_V2_ROLE,
    image_spec,
    plan_v2_from_record,
    planner_v2_spec,
    target_match,
    validate_plan_v2,
)
from .context import Context, require
from .evaluate import cluster_bootstrap_indices
from .explore import best_setting, replay
from .inference import aggregate, effective_confidences
from .logprob_v4 import LogprobRunner
from .policy import threshold_grid
from .schemas import validate_reader
from .stages_v4c_test import targets

OUT_DIR = "v5"
V3_COLUMN = "checklist_v3verb_min"
MAX_ITEMS = 3
NUM_PREDICT = {
    PLANNER_V2_ROLE: 512,
    IDENTIFY_ROLE: 128,
}  # as in configs/gqa_stop_v2.json
PERCENTILES = (2.5, 97.5)


# ---- D1: v3-style verbalised checklist_min -------------------------------------------------
def verbalised_scores(
    row: Mapping[str, Any], plan_items: Sequence[Mapping[str, Any]]
) -> tuple[float, float]:
    observations = []
    for k, item in enumerate(plan_items, start=1):
        require(
            str(row.get(f"item{k}_question")) == str(item["question"]),
            f"{row['question_id']}: item {k} question mismatch",
        )
        conf = row.get(f"item{k}_confidence")
        valid = str(row.get(f"item{k}_valid")) == "True"
        if not valid or conf is None or (isinstance(conf, float) and math.isnan(conf)):
            continue
        observations.append(
            {
                "id": item["id"],
                "answer": str(row.get(f"item{k}_answer")),
                "resolved": str(row.get(f"item{k}_resolved")) == "True",
                "confidence": float(conf),
            }
        )
    return aggregate(effective_confidences(plan_items, observations))


def add_v3_verbalised(ctx: Context, table: pd.DataFrame) -> pd.DataFrame:
    checklist = ctx.cfg["checklist"]
    runner = LogprobRunner(
        ctx.engine(),
        ctx.series,
        int(checklist["min_items"]),
        int(checklist["max_items"]),
        cache_only=True,
    )
    questions = {r["question_id"]: r for r in ctx.selected("dev")}
    plans = {q: runner.plan(questions[q])[0] for q in table["question_id"].unique()}
    mins, means = [], []
    for row in table.to_dict("records"):
        plan = plans[row["question_id"]]
        lo, hi = (0.0, 0.0) if plan.failed else verbalised_scores(row, plan.items)
        mins.append(lo)
        means.append(hi)
    return table.assign(**{V3_COLUMN: mins, "checklist_v3verb_mean": means})


def v3_replay(ctx: Context, table: pd.DataFrame) -> dict[str, Any]:
    grid = threshold_grid(ctx.cfg["policy"])
    sel22 = pd.read_csv(ctx.paths.out("dev_targets.csv"), dtype={"target": str})
    sel22 = sel22[sel22["subset"] == "v22"].set_index(["target", "method"])
    ev = ctx.cfg["evaluation"]
    groups = table.groupby("question_id")["group_id"].first().astype(str)
    qids = sorted(groups.index)
    idx = cluster_bootstrap_indices(
        [groups[q] for q in qids],
        int(ev["bootstrap_repeats"]),
        int(ev["bootstrap_seed"]),
    )
    from .policy import ConsistencyVariant, question_scores, replay_stop_consecutive

    def stops(column: str, theta: float) -> np.ndarray:
        v = ConsistencyVariant(theta=theta, k=1)
        s = question_scores(table, column)
        return np.array([replay_stop_consecutive(s[q], v).level for q in qids])

    rows = []
    for target in targets(ctx.cfg["policy"]):
        key = f"{target:.2f}"
        for column in (V3_COLUMN, "checklist_v3verb_mean"):
            best = best_setting(table, column, 1, grid, target)
            if best is None:
                rows.append({"target": key, "method": column, "feasible": False})
                continue
            d_theta = float(sel22.loc[(key, "direct_v4"), "theta"])
            diff = stops(column, best["theta"]) - stops("direct_v4", d_theta)
            draws = [float(diff[i].mean()) for i in idx]
            rows.append(
                {
                    "target": key,
                    "method": column,
                    "feasible": True,
                    "theta": best["theta"],
                    "accuracy": best["accuracy"],
                    "mean_level": best["mean_level"],
                    "stop20_rate": best["stop20_rate"],
                    "stopped_but_wrong": best["stopped_but_wrong"],
                    "direct_v4_mean_level": float(
                        sel22.loc[(key, "direct_v4"), "mean_level"]
                    ),
                    "mean_level_diff_vs_direct_v4": float(diff.mean()),
                    "ci_low": float(np.percentile(draws, PERCENTILES[0])),
                    "ci_high": float(np.percentile(draws, PERCENTILES[1])),
                }
            )
    curve = pd.DataFrame(
        [
            {
                "theta": t,
                **{
                    k: v
                    for k, v in replay(table, V3_COLUMN, t, 1).items()
                    if k != "stopped_wrong_ids"
                },
            }
            for t in grid
        ]
    )
    item_conf = pd.concat(
        [table[f"item{k}_confidence"] for k in range(1, MAX_ITEMS + 1)]
    )
    return {
        "selections": rows,
        "curve": curve,
        "distinct_operating_points": int(
            len(curve[["accuracy", "mean_level"]].drop_duplicates())
        ),
        "item_confidence_counts": {
            str(k): int(v)
            for k, v in item_conf.dropna().round(3).value_counts().items()
        },
        "min_score_counts": {
            str(k): int(v) for k, v in table[V3_COLUMN].round(3).value_counts().items()
        },
    }


# ---- D2: v2 blind identification ------------------------------------------------------------
def identify_engine(ctx: Context) -> RequestEngine:
    ctx.verify_live_digest()
    texts = {
        r: (ctx.paths.prompts_dir / f"{r}.txt").read_text("utf-8")
        for r in (PLANNER_V2_ROLE, IDENTIFY_ROLE)
    }
    texts[REPAIR_PROMPT] = ctx.prompts.texts[REPAIR_PROMPT]
    settings = dataclasses.replace(
        ctx.settings, num_predict={**ctx.settings.num_predict, **NUM_PREDICT}
    )
    engine = RequestEngine(
        settings,
        PromptSet(texts),
        ctx.transport_factory(settings),
        ctx.paths.out(OUT_DIR),
        monitor_from_config(ctx.cfg),
    )
    engine.stage = "v5_identify"
    return engine


def run_identify(ctx: Context, table: pd.DataFrame) -> pd.DataFrame:
    engine = identify_engine(ctx)
    questions = {r["question_id"]: r for r in ctx.selected("dev")}
    reader = table.set_index(["question_id", "level"])["reader_correct"]
    rows = []
    for q in sorted(table["question_id"].unique()):
        question = questions[q]
        plan = plan_v2_from_record(
            q, engine.run(planner_v2_spec(question), validate_plan_v2)
        )
        if plan.failed or plan.subquestions is None:
            rows.append({"question_id": q, "plan_status": plan.status})
            continue
        sub, parsed = plan.subquestions, plan.plan or {}
        for level in range(1, ctx.n_levels + 1):
            image = ctx.series.bytes_for(str(question["image_id"]), level)
            record = engine.run(
                image_spec(
                    IDENTIFY_ROLE, question, level, image, sub.identify_question
                ),
                validate_reader,
            )
            answer = (record.get("parsed") or {}).get("answer")
            text = str(answer or "")
            rows.append(
                {
                    "question_id": q,
                    "level": level,
                    "plan_status": plan.status,
                    "attribute": question["attribute"],
                    "relate_count": question["relate_count"],
                    "target": parsed.get("target"),
                    "synonyms": "|".join(parsed.get("synonyms", [])),
                    "identify_question": sub.identify_question,
                    "blind_degenerate": sub.blind_degenerate,
                    "identify_answer": answer,
                    "status": record["status"],
                    "a": bool(answer)
                    and target_match(
                        text, parsed["target"], parsed.get("synonyms", [])
                    ),
                    "a_target_only": bool(answer)
                    and target_match(text, parsed["target"], []),
                    "reader_correct": bool(reader[(q, level)]),
                }
            )
    return pd.DataFrame(rows)


def identify_summary(frame: pd.DataFrame) -> dict[str, Any]:
    ok = frame[frame["plan_status"] == "valid"].copy()
    by_level = ok.groupby("level").agg(
        a=("a", "mean"),
        a_target_only=("a_target_only", "mean"),
        reader_acc=("reader_correct", "mean"),
    )
    wrong = ok[~ok["reader_correct"].astype(bool)].groupby("level")["a"].mean()
    by_level["a_given_reader_wrong"] = wrong
    by_deg = ok.groupby("blind_degenerate")["a"].mean()
    return {
        "n_questions": int(ok["question_id"].nunique()),
        "plan_failures": sorted(
            frame.loc[frame["plan_status"] != "valid", "question_id"].unique().tolist()
        ),
        "invalid_identify": int((ok["status"] != "valid").sum()),
        "a_overall": float(ok["a"].mean()),
        "a_target_only_overall": float(ok["a_target_only"].mean()),
        "a_by_degenerate": {str(k): float(v) for k, v in by_deg.items()},
        "degenerate_questions": int(
            ok.groupby("question_id")["blind_degenerate"].first().sum()
        ),
        "by_level": json_safe(by_level),
    }


def json_safe(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    return {
        str(i): {c: (None if pd.isna(v) else float(v)) for c, v in row.items()}
        for i, row in frame.iterrows()
    }


def run_replays(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("v5_replays")
    table = read_id_csv(ctx.paths.out("level_table_dev_merged.csv"))
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] v5 replays: identify {table['question_id'].nunique()} x 20"
        )
        return {}
    out = ctx.paths.out(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    table = add_v3_verbalised(ctx, table)
    table[
        ["question_id", "level", "reader_correct", V3_COLUMN, "checklist_v3verb_mean"]
    ].to_csv(out / "v3verb_scores.csv", index=False)
    v3 = v3_replay(ctx, table)
    v3["curve"].to_csv(out / "v3verb_curve.csv", index=False)
    pd.DataFrame(v3["selections"]).to_csv(out / "v3verb_targets.csv", index=False)
    ident = run_identify(ctx, table)
    ident.to_csv(out / "identify_levels_dev.csv", index=False)
    result = {
        "label": "plan §24 D (dev 102, clean decoding)",
        "time": utc_now(),
        "v3verb": {k: v for k, v in v3.items() if k != "curve"},
        "identify": identify_summary(ident),
    }
    atomic_write_json(out / "replays_summary.json", result)
    ctx.log("§24 D: v3 verbalised replay + v2 identify written")
    return result
