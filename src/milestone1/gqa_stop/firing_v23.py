"""Plan §23: firing rate on reader-wrong questions (dev first; no abstention output).

Set: candidates whose L20 screening reader answer was incorrect (no textprobe
exclusion). Every method runs over levels 1..20 with the θ selected on the §22
enlarged dev grid (no re-tuning). A question *fires* when the stop rule is met
at some level 1..20 (score >= θ, with k consecutive levels for k=2 variants);
the forced L20 stop is not a firing. Records live in their own directory.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .client import RequestEngine, monitor_from_config
from .common import atomic_write_json, read_id_csv, read_jsonl, utc_now
from .context import Context, require, table_to_csv
from .devgrid_v22 import ALL_METHODS
from .evaluate import add_correctness, cluster_bootstrap_indices
from .inference import InferenceRunner
from .logprob_v4 import LogprobRunner
from .policy import (
    ConsistencyVariant,
    correctness_matrix,
    question_scores,
    replay_stop_consecutive,
    theta_from_json,
)
from .stages_v4c_test import with_v1

EXT_DIR = "ext_dev_wrong"
THETA_FILE = "dev_targets.csv"  # §22 enlarged-dev selections
THETA_SUBSET = "v22"
PAIRS = (
    ("checklist_v4_min", "direct_v4"),
    ("checklist_v4_min", "checklist_v1_min"),
    ("checklist_v4_min", "direct_v1_verbalized"),
)
PERCENTILES = (2.5, 97.5)
REUSE_ROLES = (
    "answer_only",
    "planner",
    "checklist",
    "direct",
    "planner_v3",
    "observer_v3",
    "relevance_v4",
)


def wrong_questions(ctx: Context, split: str) -> list[dict[str, Any]]:
    wrong = {
        r["question_id"]
        for r in read_jsonl(ctx.paths.selection("selection_results.jsonl"))
        if r["split"] == split and r["outcome"] == "incorrect"
    }
    return [r for r in ctx.candidates(split) if r["question_id"] in wrong]


def frozen_thetas(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """target -> method -> {column, k, theta, role} from the §22 dev grid."""
    frame = pd.read_csv(path, dtype={"target": str, "theta": str})
    frame = frame[(frame["subset"] == THETA_SUBSET) & (frame["method"] != "fixed_dev")]
    out: dict[str, dict[str, dict[str, Any]]] = {}
    columns = {name: column for name, column, _k, _role in ALL_METHODS}
    for row in frame.itertuples():
        if str(row.feasible) != "True":
            continue
        out.setdefault(row.target, {})[row.method] = {
            "column": columns[row.method],
            "k": int(row.k),
            "role": row.role,
            "theta": theta_from_json(
                row.theta if row.theta == "inf" else float(row.theta)
            ),
        }
    return out


def engine_for(ctx: Context, out: Path) -> RequestEngine:
    ctx.verify_live_digest()
    sources = [(ctx.paths.output_root, frozenset(REUSE_ROLES))]
    engine = RequestEngine(
        ctx.settings,
        ctx.prompts,
        ctx.transport_factory(ctx.settings),
        out,
        monitor_from_config(ctx.cfg),
        reuse_sources=sources,
    )
    engine.stage = "ext_dev_wrong"
    return engine


def run_tables(
    ctx: Context, questions: list[dict[str, Any]], out: Path
) -> pd.DataFrame:
    engine = engine_for(ctx, out)
    checklist = ctx.cfg["checklist"]
    lo, hi = int(checklist["min_items"]), int(checklist["max_items"])
    levels = list(range(1, ctx.n_levels + 1))
    v4, v1 = (
        LogprobRunner(engine, ctx.series, lo, hi),
        InferenceRunner(engine, ctx.series, lo, hi),
    )
    v4_rows: list[dict] = []
    v1_rows: list[dict] = []
    for q in questions:
        v4_rows += v4.run_question(q, levels)
        v1_rows += v1.run_question(q, levels)
    gt = ctx.gt([q["question_id"] for q in questions])
    table_to_csv(add_correctness(pd.DataFrame(v1_rows), gt), out / "v1_level_table.csv")
    table = add_correctness(pd.DataFrame(v4_rows), gt)
    v1 = read_id_csv(out / "v1_level_table.csv")
    table = with_v1(table, out / "v1_level_table.csv")
    table = table.merge(
        v1[["question_id", "level", "checklist_mean"]].rename(
            columns={"checklist_mean": "checklist_v1_mean"}
        ),
        on=["question_id", "level"],
    )
    table_to_csv(table, out / "level_table.csv")
    return table


def firing_frame(
    table: pd.DataFrame, thetas: Mapping[str, Mapping[str, Any]], n_levels: int
) -> pd.DataFrame:
    correct = correctness_matrix(table)
    rows = []
    for target, methods in thetas.items():
        for name, entry in methods.items():
            variant = ConsistencyVariant(theta=entry["theta"], k=entry["k"])
            for q, scores in question_scores(table, entry["column"]).items():
                d = replay_stop_consecutive(scores, variant)
                fired = not d.never_met
                rows.append(
                    {
                        "target": target,
                        "method": name,
                        "role": entry["role"],
                        "theta": entry["theta"],
                        "k": entry["k"],
                        "question_id": q,
                        "fired": fired,
                        "fire_level": d.level if fired else None,
                        "reader_correct_at_fire": bool(correct[q][d.level - 1])
                        if fired
                        else None,
                    }
                )
    return pd.DataFrame(rows)


def summarize(fires: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, method), g in fires.groupby(["target", "method"], sort=False):
        fired = g[g["fired"]]
        rows.append(
            {
                "target": target,
                "method": method,
                "role": g["role"].iloc[0],
                "theta": g["theta"].iloc[0],
                "k": g["k"].iloc[0],
                "n": len(g),
                "n_fired": len(fired),
                "firing_rate": len(fired) / len(g),
                "median_fire_level": float(fired["fire_level"].median())
                if len(fired)
                else None,
                "mean_fire_level": float(fired["fire_level"].mean())
                if len(fired)
                else None,
                "reader_correct_at_fire": int(fired["reader_correct_at_fire"].sum()),
                "fired_and_wrong": int(
                    (~fired["reader_correct_at_fire"].astype(bool)).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def paired(
    fires: pd.DataFrame,
    groups: Mapping[str, str],
    repeats: int,
    seed: int,
    pairs: Sequence[tuple[str, str]] = PAIRS,
) -> pd.DataFrame:
    qids = sorted(groups)
    indices = cluster_bootstrap_indices([groups[q] for q in qids], repeats, seed)
    rows = []
    for target, sub in fires.groupby("target", sort=False):
        flags = {
            m: g.set_index("question_id").loc[qids, "fired"].to_numpy(dtype=float)
            for m, g in sub.groupby("method")
        }
        for a, b in pairs:
            if a not in flags or b not in flags:
                continue
            diff = flags[a] - flags[b]
            draws = [float(diff[i].mean()) for i in indices]
            rows.append(
                {
                    "target": target,
                    "comparison": f"{a} - {b}",
                    "firing_rate_diff": float(diff.mean()),
                    "ci_low": float(np.percentile(draws, PERCENTILES[0])),
                    "ci_high": float(np.percentile(draws, PERCENTILES[1])),
                    "only_a_fired": int(((flags[a] == 1) & (flags[b] == 0)).sum()),
                    "only_b_fired": int(((flags[a] == 0) & (flags[b] == 1)).sum()),
                }
            )
    return pd.DataFrame(rows)


def firing(ctx: Context, split: str = "dev") -> dict[str, Any]:
    ctx.set_stage("ext_dev_wrong")
    require(split == "dev", "§23 runs on dev first; test waits for the user's decision")
    theta_path = ctx.paths.out(THETA_FILE)
    require(theta_path.exists(), "run `curves` (§22 dev grid) first")
    thetas = frozen_thetas(theta_path)
    questions = wrong_questions(ctx, split)
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] §23 firing: {len(questions)} reader-wrong {split} questions x 20"
        )
        return {"questions": len(questions)}
    out = ctx.paths.out(EXT_DIR)
    table = run_tables(ctx, questions, out)
    fires = firing_frame(table, thetas, ctx.n_levels)
    fires.to_csv(out / "firing_per_question.csv", index=False)
    summary = summarize(fires)
    summary.to_csv(out / "firing_summary.csv", index=False)
    groups = table.groupby("question_id")["group_id"].first().astype(str).to_dict()
    ev = ctx.cfg["evaluation"]
    pairs = paired(
        fires, groups, int(ev["bootstrap_repeats"]), int(ev["bootstrap_seed"])
    )
    pairs.to_csv(out / "firing_paired.csv", index=False)
    hist = (
        fires[fires["fired"]]
        .groupby(["target", "method", "fire_level"])
        .agg(
            n=("question_id", "size"), reader_correct=("reader_correct_at_fire", "sum")
        )
        .reset_index()
    )
    hist.to_csv(out / "firing_level_hist.csv", index=False)
    by_level = table.groupby("level")["reader_correct"].mean()
    result = {
        "label": "plan §23 (dev; θ from §22 dev grid, not re-tuned)",
        "time": utc_now(),
        "n_questions": len(questions),
        "n_images": len(set(groups.values())),
        "theta_source": str(theta_path),
        "thetas": json.loads(json.dumps(thetas, default=str)),
        "reader_accuracy_by_level": {int(k): float(v) for k, v in by_level.items()},
    }
    atomic_write_json(out / "firing_meta.json", result)
    ctx.state.set(
        "ext_dev_wrong", {"done": True, "time": utc_now(), "questions": len(questions)}
    )
    ctx.log(f"§23 firing (dev): {len(questions)} reader-wrong questions")
    return result
