"""Exploratory dev probe (orchestrator request, 2026-10-09; not pre-registered).

Does a reader that also sees the same-level checklist observations (observer
answer, resolved flag, Rel) answer correctly at lower levels than the plain
reader? Inputs come from the v4c dev level table (plans_v3 / observer_v3_lp /
relevance_v4 records). GT and the direct answer are never shown. One reader
call per dev question x level, in its own record directory (reader_informed/).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
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
from .common import atomic_write_json, read_id_csv, read_json, utc_now
from .context import Context, require, table_to_csv
from .devgrid import fixed_curve
from .evaluate import add_correctness, cluster_bootstrap_indices
from .explore import best_setting, replay
from .logprob_v4 import TOP_LOGPROBS, locate_answer_token
from .policy import threshold_grid
from .schemas import validate_reader

ROLE = "reader_informed"
NUM_PREDICT = 128  # same as answer_only
MAX_ITEMS = 3
NEVER = 21  # onset when the answer is not correct at L20
PERCENTILES = (2.5, 97.5)
OUTPUT_CSV = "reader_informed_levels_dev.csv"
SUMMARY_JSON = "reader_informed_summary.json"
FROZEN_METHOD = "checklist_v4_min"


def observation_lines(row: Mapping[str, Any]) -> list[str]:
    """One line per checklist item at this level (no GT, no direct answer)."""
    lines = []
    for k in range(1, MAX_ITEMS + 1):
        question = row.get(f"item{k}_question")
        if not isinstance(question, str) or not question.strip():
            continue
        answer = row.get(f"item{k}_answer")
        answer = answer if isinstance(answer, str) and answer.strip() else "(none)"
        resolved = row.get(f"item{k}_resolved")
        resolved_text = (
            "true" if resolved is True or str(resolved) == "True" else "false"
        )
        rel = row.get(f"item{k}_rel")
        rel_text = (
            "n/a"
            if rel is None or (isinstance(rel, float) and math.isnan(rel))
            else (f"{float(rel):.2f}")
        )
        lines.append(
            f"Q: {question} | A: {answer} | resolved: {resolved_text} "
            f"| answerability: {rel_text}"
        )
    return lines


def informed_spec(
    question: Mapping[str, Any], level: int, image: bytes, row: Mapping[str, Any]
) -> RequestSpec:
    data = {
        "question": str(question["question"]),
        "auxiliary_observations": observation_lines(row),
    }
    return RequestSpec(
        ROLE, str(question["question_id"]), level, data, image, logprobs=TOP_LOGPROBS
    )


def probe_engine(ctx: Context) -> RequestEngine:
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
        ctx.paths.output_root,
        monitor_from_config(ctx.cfg),
    )
    engine.stage = "reader_informed_probe"
    return engine


def run_probe(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("reader_informed_probe")
    plain = read_id_csv(ctx.paths.out("level_table_dev.csv"))
    questions = {q["question_id"]: q for q in ctx.selected("dev")}
    require(set(plain["question_id"]) == set(questions), "level table != dev subset")
    if ctx.dry_run:
        ctx.echo(f"[dry-run] informed reader: {len(plain)} calls")
        return {}
    engine = probe_engine(ctx)
    rows = []
    for row in plain.sort_values(["question_id", "level"]).to_dict("records"):
        q, level = questions[row["question_id"]], int(row["level"])
        spec = informed_spec(
            q, level, ctx.series.bytes_for(str(q["image_id"]), level), row
        )
        record = engine.run(spec, validate_reader)
        parsed = record.get("parsed") or {}
        token = locate_answer_token(record.get("logprobs"))
        rows.append(
            {
                "question_id": row["question_id"],
                "level": level,
                "group_id": row["group_id"],
                "attribute": row.get("attribute"),
                "reader_answer": parsed.get("answer"),
                "status": record["status"],
                "pans": token.prob if token else None,
                "n_observations": len(spec.input_data["auxiliary_observations"]),
                "plain_answer": row["reader_answer"],
                "plain_correct": bool(row["reader_correct"]),
            }
        )
    table = add_correctness(pd.DataFrame(rows), ctx.gt(sorted(questions)))
    table = table.rename(
        columns={
            "reader_answer": "informed_answer",
            "reader_correct": "informed_correct",
        }
    )
    table_to_csv(table, ctx.paths.out(OUTPUT_CSV))
    summary = analyse(table, plain, ctx)
    atomic_write_json(ctx.paths.out(SUMMARY_JSON), summary)
    ctx.log(f"exploratory informed-reader probe: {len(table)} calls (dev)")
    return summary


def onsets(table: pd.DataFrame, column: str) -> dict[str, int]:
    """First level from which `column` stays True through the last level."""
    out = {}
    for q, g in table.sort_values("level").groupby("question_id"):
        flags = g[column].astype(bool).tolist()
        onset = NEVER
        for t in range(len(flags), 0, -1):
            if not flags[t - 1]:
                break
            onset = t
        out[str(q)] = onset
    return out


def analyse(table: pd.DataFrame, plain: pd.DataFrame, ctx: Context) -> dict[str, Any]:
    by_level = table.groupby("level")[["plain_correct", "informed_correct"]].mean()
    on_plain, on_inf = onsets(table, "plain_correct"), onsets(table, "informed_correct")
    qids = sorted(on_plain)
    a = np.array([on_plain[q] for q in qids], dtype=float)
    b = np.array([on_inf[q] for q in qids], dtype=float)
    groups = table.groupby("question_id")["group_id"].first().astype(str)
    cfg = ctx.cfg["evaluation"]
    draws = [
        float((b[i] - a[i]).mean())
        for i in cluster_bootstrap_indices(
            [groups[q] for q in qids],
            int(cfg["bootstrap_repeats"]),
            int(cfg["bootstrap_seed"]),
        )
    ]
    l20 = table[table["level"] == table["level"].max()].set_index("question_id")
    informed_view = (
        plain.drop(columns=["reader_correct"])
        .merge(
            table[["question_id", "level", "informed_correct"]],
            on=["question_id", "level"],
        )
        .rename(columns={"informed_correct": "reader_correct"})
    )
    frozen = read_json(ctx.paths.out("frozen_policy.json"))["selections"]["0.95"][
        FROZEN_METHOD
    ]
    theta = float(frozen["theta"])
    target = float(ctx.cfg["policy"]["primary_dev_accuracy_target"])
    grid = threshold_grid(ctx.cfg["policy"])
    reselect = best_setting(informed_view, FROZEN_METHOD, 1, grid, target)
    plain_fixed, inf_fixed = fixed_curve(plain), fixed_curve(informed_view)
    return {
        "label": "exploratory (orchestrator request; not pre-registered; dev in-sample)",
        "time": utc_now(),
        "n_questions": len(qids),
        "calls": len(table),
        "invalid": int((table["status"] != "valid").sum()),
        "accuracy_by_level": {
            int(t): {
                "plain": float(r.plain_correct),
                "informed": float(r.informed_correct),
            }
            for t, r in by_level.iterrows()
        },
        "onset": {
            "plain_mean": float(a.mean()),
            "plain_median": float(np.median(a)),
            "informed_mean": float(b.mean()),
            "informed_median": float(np.median(b)),
            "diff_mean": float((b - a).mean()),
            "diff_ci": [float(np.percentile(draws, p)) for p in PERCENTILES],
            "earlier": int((b < a).sum()),
            "later": int((b > a).sum()),
            "same": int((b == a).sum()),
            "never_value": NEVER,
            "plain": on_plain,
            "informed": on_inf,
        },
        "l20": {
            "informed_wrong": sorted(l20.index[~l20["informed_correct"]].tolist()),
            "informed_answers_when_wrong": {
                q: str(l20.loc[q, "informed_answer"])
                for q in l20.index[~l20["informed_correct"]]
            },
        },
        "frozen_stops": {
            "theta": theta,
            "plain": replay(plain, FROZEN_METHOD, theta, 1),
            "informed": replay(informed_view, FROZEN_METHOD, theta, 1),
        },
        "reselected_in_sample": None
        if reselect is None
        else {
            "theta": reselect["theta"],
            "accuracy": reselect["accuracy"],
            "mean_level": reselect["mean_level"],
            "stopped_wrong_ids": reselect["stopped_wrong_ids"],
        },
        "fixed_dev": {
            "plain": next(
                (t for t in sorted(plain_fixed) if plain_fixed[t] >= target), None
            ),
            "informed": next(
                (t for t in sorted(inf_fixed) if inf_fixed[t] >= target), None
            ),
        },
    }
