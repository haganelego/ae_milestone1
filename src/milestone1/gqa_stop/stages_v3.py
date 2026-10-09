"""v3 stages (plan §18): pilot with §18.3 diagnostics + negative probe, devgrid.

Two families share these stages:
- ``item_checklist`` (v3 final, §18.6): one call per item and level, v1 scores;
- ``sampled_checklist`` (withdrawn N = 5 design, kept for its tests only).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pandas as pd

from .client import PARSE_VERSION
from .common import atomic_write_json, resolve_path, utc_now
from .context import Context, require, table_to_csv
from .devgrid_v3 import item_components, run_devgrid_v3
from .item_checklist import METHOD_COLUMNS as ITEM_METHOD_COLUMNS
from .item_checklist import PRIMARY_METHOD as ITEM_PRIMARY
from .pilot_v3 import (
    item_metrics,
    l20_per_question,
    l20_zero_causes,
    negative_probe,
    negative_summary,
    pilot_metrics,
)
from .sampled import sampling_from_config
from .stages_model import (
    FORMAT_VALID_TARGET,
    ITEM,
    SAMPLED,
    active_tag,
    full_table,
    level_table,
    planned_requests,
    require_reader_prompt,
    require_screening_match,
    split_table_path,
)

EXAMPLE_COUNT = 5
V3_FAMILIES = frozenset({SAMPLED, ITEM})


def _first_questions(table: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(table["question_id"]))[:EXAMPLE_COUNT]


def _items(first: pd.Series) -> list[str]:
    return [
        first[f"item{i}_question"]
        for i in (1, 2, 3)
        if f"item{i}_question" in first and isinstance(first[f"item{i}_question"], str)
    ]


def plan_examples(table: pd.DataFrame, levels: Sequence[int]) -> list[dict[str, Any]]:
    """Sampled family examples."""
    low, high = min(levels), max(levels)
    examples = []
    for qid in _first_questions(table):
        rows = table[table["question_id"] == qid].set_index("level")
        first = rows.iloc[0]
        examples.append(
            {
                "question_id": qid,
                "question": first["question"],
                "items": _items(first),
                "item_not_standalone_any": bool(first["item_not_standalone_any"]),
                **{
                    f"L{t}": {
                        "item1_resolved_frac": rows.loc[t, "item1_resolved_frac"],
                        "v3_min_smean": rows.loc[t, "v3_min_smean"],
                        "v3_min_sagree": rows.loc[t, "v3_min_sagree"],
                        "direct_sagree": rows.loc[t, "direct_sagree"],
                        "reader_correct": bool(rows.loc[t, "reader_correct"]),
                    }
                    for t in (low, high)
                },
            }
        )
    return examples


def item_examples(table: pd.DataFrame, levels: Sequence[int]) -> list[dict[str, Any]]:
    """v3 final examples: plan, item-1 answer, stop score at min/max level."""
    low, high = min(levels), max(levels)
    examples = []
    for qid in _first_questions(table):
        rows = table[table["question_id"] == qid].set_index("level")
        first = rows.iloc[0]
        examples.append(
            {
                "question_id": qid,
                "question": first["question"],
                "items": _items(first),
                "last_item_has_attribute": bool(first["last_item_has_attribute"]),
                "item_not_standalone_any": bool(first["item_not_standalone_any"]),
                **{
                    f"L{t}": {
                        "item1": [
                            rows.loc[t, "item1_answer"],
                            bool(rows.loc[t, "item1_resolved"]),
                        ],
                        "checklist_v3_min": rows.loc[t, "checklist_v3_min"],
                        "reader_correct": bool(rows.loc[t, "reader_correct"]),
                    }
                    for t in (low, high)
                },
            }
        )
    return examples


def pilot_v3(ctx: Context, levels: Sequence[int]) -> dict[str, Any]:
    ctx.set_stage("pilot")
    require(ctx.family in V3_FAMILIES, "pilot_v3 is for the v3 families")
    screening = (ctx.state.get("screening") or {}).get("dev")
    require(bool(screening), "pilot needs `screen --split dev` first")
    require_reader_prompt(ctx, screening)
    questions = ctx.selected("dev")[: int(ctx.cfg["execution"]["pilot_questions"])]
    planned = planned_requests(ctx, len(questions), len(levels))
    probe_cfg = ctx.cfg["negative_probe"]
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] v3 pilot ({ctx.family}): {len(questions)} visual questions x "
            f"{list(levels)} -> up to {planned} new requests + negative probe "
            f"({len({q['image_id'] for q in questions})} images x {probe_cfg['levels']})"
        )
        return {"planned_requests": planned}
    table = level_table(ctx, questions, levels, cache_only=False)
    tag = active_tag(ctx)
    table_to_csv(table, ctx.paths.out(f"pilot/pilot_levels_{tag}.csv"))
    sampled = ctx.family == SAMPLED
    sampling = sampling_from_config(ctx.cfg) if sampled else None
    negative = negative_probe(
        ctx.engine(), ctx.series, questions, probe_cfg, sampling, ctx.seed
    )
    negative.to_csv(ctx.paths.out("pilot/negative_probe.csv"), index=False)
    metrics = pilot_metrics(table) if sampled else item_metrics(table)
    examples = plan_examples(table, levels) if sampled else item_examples(table, levels)
    summary = {
        "model": ctx.settings.name,
        "digest": ctx.settings.digest,
        "model_tag": tag,
        "family": ctx.family,
        "prompt_hashes": ctx.prompts.hashes,
        "parse_version": PARSE_VERSION,
        "questions": int(table["question_id"].nunique()),
        "levels": list(levels),
        "sampling": (
            {
                "n": sampling.n,
                "temperature": sampling.temperature,
                "seed_base": sampling.seed_base,
            }
            if sampling is not None
            else "none (N = 1, v1 decoding; plan §18.6)"
        ),
        **metrics,
        "format_gate_uses": "valid_rate_initial",
        "format_valid_target": FORMAT_VALID_TARGET,
        "format_valid_ok": all(
            v >= FORMAT_VALID_TARGET for v in metrics["valid_rate_initial"].values()
        ),
        "negative_probe": negative_summary(negative),
        "negative_probe_decoding": (
            "observer_v3 sample-0 decoding (T, seed_base)"
            if sampled
            else "observer_v3 default v1 decoding (temperature 0, v1 seed)"
        ),
        "examples": examples,
    }
    atomic_write_json(ctx.paths.out(f"pilot/pilot_summary_{tag}.json"), summary)
    atomic_write_json(ctx.paths.out("pilot_summary.json"), summary)
    ctx.state.set(
        "pilot",
        {
            "done": True,
            "time": utc_now(),
            "model": ctx.settings.name,
            "family": ctx.family,
            "calibration": False,
        },
    )
    ctx.log(
        f"v3 pilot ({ctx.family}): valid_rate_initial {metrics['valid_rate_initial']}"
    )
    for example in examples:
        ctx.echo(json.dumps(example, ensure_ascii=False, default=str))
    ctx.note_reuse()
    return summary


def devgrid_v3(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("devgrid")
    require(ctx.family in V3_FAMILIES, "devgrid_v3 is for the v3 families")
    require_screening_match(ctx, "dev")
    info = (ctx.state.get("infer") or {}).get("dev", {})
    require(bool(info.get("complete")), "dev `infer --levels all` must complete first")
    require(
        info.get("family") == ctx.family,
        f"dev inference was run for family {info.get('family')}, not {ctx.family}",
    )
    if ctx.dry_run:
        ctx.echo("[dry-run] v3 devgrid on the dev visual subset; no model calls")
        return {}
    table = full_table(ctx, "dev")
    table_to_csv(table, split_table_path(ctx, "dev"))
    comparison = ctx.cfg.get("comparison", {})
    v1_root, fair = comparison.get("v1_output_root"), comparison.get("v1_fair_frozen")
    roots = (
        resolve_path(ctx.paths.repo_root, v1_root) if v1_root else None,
        resolve_path(ctx.paths.repo_root, fair) if fair else None,
    )
    if ctx.family == ITEM:
        summary = run_devgrid_v3(
            table,
            ctx.cfg["policy"],
            ctx.cfg["evaluation"],
            ctx.paths.output_root,
            *roots,
            method_columns=ITEM_METHOD_COLUMNS,
            components_fn=item_components,
            abort_methods=[ITEM_PRIMARY],
        )
        causes = l20_zero_causes(table, ctx.n_levels)
        atomic_write_json(ctx.paths.out("dev_l20_zero_causes.json"), causes)
        l20_per_question(table, ctx.n_levels).to_csv(
            ctx.paths.out("dev_l20_per_question.csv"), index=False
        )
        summary = {**summary, "l20_zero_causes": causes}
    else:
        summary = run_devgrid_v3(
            table,
            ctx.cfg["policy"],
            ctx.cfg["evaluation"],
            ctx.paths.output_root,
            *roots,
        )
    ctx.state.set(
        "devgrid",
        {
            "done": True,
            "time": utc_now(),
            "family": ctx.family,
            "abort": summary["abort"],
        },
    )
    ctx.log(f"v3 devgrid ({ctx.family}): abort={summary['abort']}")
    ctx.note_reuse()
    return summary
