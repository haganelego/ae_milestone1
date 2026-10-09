"""v4 stages (plan §19): pilot with §19.3 gates and distributions, devgrid."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .client import PARSE_VERSION, RequestEngine
from .common import atomic_write_json, read_id_csv, read_json, resolve_path, utc_now
from .context import Context, require, table_to_csv
from .devgrid_v3 import run_devgrid_v3
from .item_checklist import item_spec
from .logprob_v4 import METHOD_COLUMNS, PRIMARY_METHOD, observer_lp_spec
from .stages_model import (
    LOGPROB,
    active_tag,
    full_table,
    level_table,
    planned_requests,
    require_reader_prompt,
    require_screening_match,
    split_table_path,
)

GATE_RATE = 0.95
ITEM_COLUMNS = (1, 2, 3)
SIGNALS = ("pans", "rel", "product")
SENSITIVITY_STEP = 0.05


# ---- pilot metrics ----------------------------------------------------------------------
def item_values(frame: pd.DataFrame, key: str, valid_only: bool = True) -> pd.Series:
    parts = []
    for i in ITEM_COLUMNS:
        col = f"item{i}_{key}"
        if col not in frame:
            continue
        keep = (
            frame[f"item{i}_valid"].fillna(False).astype(bool) if valid_only else True
        )
        parts.append(frame.loc[keep, col] if valid_only else frame[col])
    return pd.concat(parts).dropna() if parts else pd.Series(dtype=float)


def call_rates(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    initial, after, repair = {}, {}, {}
    for role in ("observer", "direct_lp", "relevance"):
        calls = float(table.get(f"{role}_calls", pd.Series(dtype=float)).sum())
        if calls:
            initial[role] = float(table[f"{role}_initial_valid"].sum()) / calls
            after[role] = float(table[f"{role}_valid"].sum()) / calls
            repair[role] = float(table[f"{role}_repaired"].sum()) / calls
    plans = table.groupby("question_id").first()
    initial["planner"] = float(plans["planner_initial_valid"].astype(bool).mean())
    after["planner"] = float((plans["planner_status"] == "valid").mean())
    initial["reader"] = float(table["reader_initial_valid"].astype(bool).mean())
    return {
        "valid_rate_initial": initial,
        "valid_rate_after_repair": after,
        "repair_rate": repair,
    }


def gates(table: pd.DataFrame) -> dict[str, Any]:
    located = pd.concat(
        [item_values(table, "located"), table["direct_located"]]
    ).astype(bool)
    yes_no = pd.concat(
        [
            item_values(table, "rel_has_yes_no", valid_only=False),
            table["direct_rel_has_yes_no"],
        ]
    ).astype(bool)
    located_rate, yes_no_rate = float(located.mean()), float(yes_no.mean())
    return {
        "answer_token_located_rate": located_rate,
        "rel_yes_or_no_present_rate": yes_no_rate,
        "gate_target": GATE_RATE,
        "gates_pass": located_rate >= GATE_RATE and yes_no_rate >= GATE_RATE,
    }


def distribution(values: pd.Series) -> dict[str, Any]:
    return {
        "n": int(len(values)),
        "distinct": int(values.round(6).nunique()),
        "mean": float(values.mean()) if len(values) else None,
        "median": float(values.median()) if len(values) else None,
    }


def by_level(table: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for level, frame in table.groupby("level", sort=True):
        out[str(int(level))] = {
            "reader_accuracy": float(frame["reader_correct"].mean()),
            "item1_resolved_rate": float(frame["item1_resolved"].astype(bool).mean()),
            **{f"item_{s}": distribution(item_values(frame, s)) for s in SIGNALS},
            "direct_pans": distribution(frame["direct_pans"]),
            "direct_rel": distribution(frame["direct_rel"]),
            "direct_v4": distribution(frame["direct_v4"]),
            "checklist_v4_min": distribution(frame["checklist_v4_min"]),
        }
    return out


def monotonic(means: Sequence[float]) -> bool:
    return all(b >= a for a, b in zip(means, means[1:], strict=False))


def text_equality(
    ctx: Context, engine: RequestEngine, table: pd.DataFrame, v3_root: Path
) -> dict[str, Any]:
    """Does adding logprobs change observer_v3's reply? Compare with the v3 cache."""
    same_raw = same_parsed = compared = 0
    for row in table.itertuples(index=False):
        q = {"question_id": row.question_id, "image_id": row.image_id}
        for i in ITEM_COLUMNS:
            question = getattr(row, f"item{i}_question", None)
            if not isinstance(question, str):
                continue
            item = {"id": f"c{i}", "question": question}
            v4_path = engine.record_path(observer_lp_spec(q, int(row.level), b"", item))
            v3_rel = engine.record_path(item_spec(q, int(row.level), b"", item))
            v3_path = v3_root / v3_rel.relative_to(ctx.paths.output_root)
            if not (v4_path.exists() and v3_path.exists()):
                continue
            a, b = read_json(v4_path), read_json(v3_path)
            compared += 1
            same_raw += int(a.get("raw") == b.get("raw"))
            same_parsed += int(a.get("parsed") == b.get("parsed"))
    return {
        "compared": compared,
        "raw_equal_rate": same_raw / compared if compared else None,
        "parsed_equal_rate": same_parsed / compared if compared else None,
    }


def pilot_v4(ctx: Context, levels: Sequence[int]) -> dict[str, Any]:
    ctx.set_stage("pilot")
    require(ctx.family == LOGPROB, "pilot_v4 is for the logprob_item family")
    screening = (ctx.state.get("screening") or {}).get("dev")
    require(bool(screening), "pilot needs `screen --split dev` first")
    require_reader_prompt(ctx, screening)
    questions = ctx.selected("dev")[: int(ctx.cfg["execution"]["pilot_questions"])]
    planned = planned_requests(ctx, len(questions), len(levels))
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] v4 pilot: {len(questions)} visual questions x {list(levels)} "
            f"-> up to {planned} new requests (observer+relevance+direct with logprobs)"
        )
        return {"planned_requests": planned}
    table = level_table(ctx, questions, levels, cache_only=False)
    tag = active_tag(ctx)
    table_to_csv(table, ctx.paths.out(f"pilot/pilot_levels_{tag}.csv"))
    v3_root = resolve_path(ctx.paths.repo_root, ctx.cfg["comparison"]["v3_output_root"])
    levels_info = by_level(table)
    rel_means = [levels_info[k]["item_rel"]["mean"] for k in levels_info]
    direct_rel_means = [levels_info[k]["direct_rel"]["mean"] for k in levels_info]
    gate = gates(table)
    summary = {
        "model": ctx.settings.name,
        "digest": ctx.settings.digest,
        "model_tag": tag,
        "family": ctx.family,
        "prompt_hashes": ctx.prompts.hashes,
        "parse_version": PARSE_VERSION,
        "questions": int(table["question_id"].nunique()),
        "levels": list(levels),
        **call_rates(table),
        **gate,
        "format_valid_ok": gate["gates_pass"],
        "observer_text_vs_v3": text_equality(ctx, ctx.engine(), table, v3_root),
        "by_level": levels_info,
        "item_rel_mean_monotonic": monotonic(rel_means),
        "direct_rel_mean_monotonic": monotonic(direct_rel_means),
    }
    atomic_write_json(ctx.paths.out(f"pilot/pilot_summary_{tag}.json"), summary)
    atomic_write_json(ctx.paths.out("pilot_summary.json"), summary)
    ctx.state.set(
        "pilot",
        {
            "done": True,
            "time": utc_now(),
            "family": ctx.family,
            "gates_pass": gate["gates_pass"],
        },
    )
    ctx.log(f"v4 pilot: gates {gate} rates {summary['valid_rate_initial']}")
    ctx.note_reuse()
    return summary


# ---- devgrid ----------------------------------------------------------------------------
def v4_components(
    table: pd.DataFrame, best: Mapping[str, Any], method_columns: Mapping[str, str]
) -> pd.DataFrame:
    rows = []
    for level, frame in table.groupby("level", sort=True):
        wrong = ~frame["reader_correct"].astype(bool)
        row: dict[str, Any] = {
            "level": int(level),
            "reader_accuracy": float((~wrong).mean()),
            "item1_resolved_rate": float(frame["item1_resolved"].astype(bool).mean()),
            "checklist_v4_min_zero_rate": float(
                (frame["checklist_v4_min"] <= 0).mean()
            ),
            **{f"mean_item_{s}": float(item_values(frame, s).mean()) for s in SIGNALS},
            "distinct_checklist_v4_min": int(
                frame["checklist_v4_min"].round(6).nunique()
            ),
        }
        for method, column in method_columns.items():
            row[f"mean_{column}"] = float(frame[column].mean())
            chosen = best.get(method)
            if chosen is not None and np.isfinite(float(chosen["theta"])):
                row[f"p_{column}_ge_best_and_wrong"] = float(
                    ((frame[column] >= float(chosen["theta"])) & wrong).mean()
                )
        rows.append(row)
    return pd.DataFrame(rows)


def l20_table(table: pd.DataFrame, last_level: int) -> pd.DataFrame:
    rows = table[table["level"] == last_level].sort_values("question_id")
    keep = [
        "question_id",
        "question",
        "attribute",
        "relate_count",
        "item1_question",
        "item1_answer",
        "item1_resolved",
        "item1_pans",
        "item1_rel",
    ]
    for i in (2, 3):
        keep += [
            f"item{i}_answer",
            f"item{i}_resolved",
            f"item{i}_pans",
            f"item{i}_rel",
        ]
    keep += ["checklist_v4_min", "direct_v4", "reader_answer", "reader_correct"]
    return rows[[c for c in keep if c in rows]]


def sensitivity(grid: pd.DataFrame) -> dict[str, Any]:
    own = grid[grid["method"] == PRIMARY_METHOD]
    points = own[["accuracy", "mean_level"]].round(6).drop_duplicates()
    sample = own[
        np.isclose((own["theta"] / SENSITIVITY_STEP) % 1, 0)
        | np.isclose((own["theta"] / SENSITIVITY_STEP) % 1, 1)
    ]
    return {
        "distinct_operating_points": int(len(points)),
        "curve_every_0.05": sample[
            ["theta", "accuracy", "mean_level", "stopped_but_wrong"]
        ]
        .round(4)
        .to_dict("records"),
    }


def merge_v3(table: pd.DataFrame, v3_root: Path) -> pd.DataFrame:
    v3 = read_id_csv(v3_root / "level_table_dev.csv")[
        ["question_id", "level", "checklist_v3_min"]
    ]
    merged = table.merge(v3, on=["question_id", "level"], how="left")
    require(
        merged["checklist_v3_min"].notna().all(),
        "v3 table lacks some (question, level)",
    )
    return merged


def devgrid_v4(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("devgrid")
    require(ctx.family == LOGPROB, "devgrid_v4 is for the logprob_item family")
    require_screening_match(ctx, "dev")
    info = (ctx.state.get("infer") or {}).get("dev", {})
    require(bool(info.get("complete")), "dev `infer --levels all` must complete first")
    require(
        info.get("family") == ctx.family, "dev inference was run for another family"
    )
    if ctx.dry_run:
        ctx.echo("[dry-run] v4 devgrid on the dev visual subset; no model calls")
        return {}
    comparison = ctx.cfg["comparison"]
    root = ctx.paths.repo_root
    table = merge_v3(
        full_table(ctx, "dev"), resolve_path(root, comparison["v3_output_root"])
    )
    table_to_csv(table, split_table_path(ctx, "dev"))
    summary = run_devgrid_v3(
        table,
        ctx.cfg["policy"],
        ctx.cfg["evaluation"],
        ctx.paths.output_root,
        resolve_path(root, comparison["v1_output_root"]),
        resolve_path(root, comparison["v1_fair_frozen"]),
        method_columns=METHOD_COLUMNS,
        components_fn=v4_components,
        abort_methods=[PRIMARY_METHOD],
    )
    l20_table(table, ctx.n_levels).to_csv(
        ctx.paths.out("dev_l20_per_question.csv"), index=False
    )
    grid = pd.read_csv(ctx.paths.out("dev_threshold_grid.csv"))
    sens = sensitivity(grid)
    atomic_write_json(ctx.paths.out("dev_sensitivity.json"), sens)
    summary = {**summary, "sensitivity": sens}
    atomic_write_json(ctx.paths.out("devgrid_summary.json"), summary)
    ctx.state.set(
        "devgrid",
        {
            "done": True,
            "time": utc_now(),
            "family": ctx.family,
            "abort": summary["abort"],
        },
    )
    ctx.log(
        f"v4 devgrid: abort={summary['abort']} points={sens['distinct_operating_points']}"
    )
    ctx.note_reuse()
    return summary
