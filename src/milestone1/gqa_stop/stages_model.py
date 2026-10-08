"""Model stages: preflight, pilot (+A4 calibration), infer, freeze, evaluate."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from . import preflight as preflight_mod
from .calibration import calibration_report
from .client import PARSE_VERSION, model_tag
from .common import (
    StageError,
    atomic_write_json,
    read_json,
    resolve_path,
    sha256_file,
    sha256_json,
    utc_now,
    write_effective_config,
)
from .consistency import ConsistencyRunner, planned_consistency_requests
from .context import Context, require, table_to_csv
from .devgrid import run_devgrid
from .evaluate import add_correctness, evaluate
from .inference import LEVEL_ROLES, READER_ROLE, planned_request_count
from .policy import consistency_variants, freeze_policy
from .stages_data import write_dataset_flow

VALID = "valid"
CONSISTENCY = "consistency"
FORMAT_VALID_TARGET = 0.95
FORMAT_VALID_EXIT = 0.90


# ---- preflight --------------------------------------------------------------
def preflight(ctx: Context, get: Callable[..., Any] | None = None) -> dict[str, Any]:
    record = preflight_mod.collect(ctx.cfg, ctx.paths, get)
    model = record["model"]
    ctx.echo(
        f"preflight: branch={record['repo']['branch']} questions={record['questions']['count']} "
        f"model={model['name']}@{model['digest']} fallback={model['fallback_name']}@"
        f"{model['fallback_digest']} free_gb={record['disk']['free_gb']}"
    )
    if ctx.dry_run:
        ctx.echo(f"[dry-run] would write {ctx.paths.out('preflight.json')}")
        return record
    atomic_write_json(ctx.paths.out("preflight.json"), record)
    if model["digest"] is None:
        raise StageError(f"model {model['name']} not found in ollama /api/tags")
    frozen_path = ctx.paths.out("frozen_policy.json")
    if frozen_path.exists() and model["digest"] != read_json(frozen_path)["digest"]:
        raise StageError(
            f"live digest {model['digest']} differs from the frozen policy digest; "
            "refusing to change the model after freeze (plan §2)"
        )
    overrides = {"model": {"fallback_digest": model["fallback_digest"]}}
    switched = (ctx.state.get("model_switch") or {}).get("active")
    if not switched:
        overrides["model"]["digest"] = model["digest"]
    effective = write_effective_config(
        ctx.config_path, ctx.paths, overrides, "preflight: model digests from /api/tags"
    )
    ctx.refresh(effective)
    ctx.state.set("preflight", {"done": True, "time": utc_now()})
    return record


# ---- level tables ----------------------------------------------------------------
def level_table(
    ctx: Context,
    questions: Sequence[Mapping[str, Any]],
    levels: Sequence[int],
    cache_only: bool,
    roles: Sequence[str] = LEVEL_ROLES,
) -> pd.DataFrame:
    if ctx.family == CONSISTENCY:
        v2 = ConsistencyRunner(ctx.engine(), ctx.series, cache_only=cache_only)
        rows = [row for q in questions for row in v2.run_question(q, levels)]
    else:
        runner = ctx.runner(cache_only=cache_only)
        rows = [row for q in questions for row in runner.run_question(q, levels, roles)]
    table = pd.DataFrame(rows)
    return add_correctness(table, ctx.gt([q["question_id"] for q in questions]))


def split_table_path(ctx: Context, split: str) -> Any:
    return ctx.paths.out(f"level_table_{split}.csv")


# ---- pilot -----------------------------------------------------------------------
def switch_to_fallback(ctx: Context) -> None:
    require(
        not (ctx.state.get("infer") or {}).get("dev"),
        "model switch is allowed only before any dev `infer` (plan §16 A4)",
    )
    model = ctx.cfg["model"]
    require(
        bool(model.get("fallback_digest")), "fallback digest unknown: run preflight"
    )
    primary_tag = model_tag(model["name"], model.get("digest"))
    if ctx.dry_run:
        ctx.echo(f"[dry-run] would switch model to {model['fallback_name']}")
        return
    effective = write_effective_config(
        ctx.config_path,
        ctx.paths,
        {"model": {"name": model["fallback_name"], "digest": model["fallback_digest"]}},
        "pilot calibration gate: switch to fallback model (A4)",
    )
    ctx.refresh(effective)
    ctx.state.set(
        "model_switch",
        {
            "active": True,
            "time": utc_now(),
            "model": model["fallback_name"],
            "primary_tag": primary_tag,
        },
    )
    ctx.log(f"model switched to {model['fallback_name']}; dev screening must be re-run")


def pilot(
    ctx: Context, levels: Sequence[int], calibration: bool, use_fallback: bool
) -> dict[str, Any]:
    ctx.set_stage("pilot")
    screening = (ctx.state.get("screening") or {}).get("dev")
    require(bool(screening), "pilot needs `screen --split dev` first")
    require_reader_prompt(ctx, screening)
    if use_fallback:
        switch_to_fallback(ctx)
    count = int(ctx.cfg["execution"]["pilot_questions"])
    questions = ctx.selected("dev")[:count]
    planned = planned_requests(ctx, len(questions), len(levels))
    if ctx.family == CONSISTENCY and calibration:
        ctx.echo(
            "note: --calibration ignored for the consistency family (no confidence)"
        )
        calibration = False
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] pilot ({ctx.family}): {len(questions)} questions x levels "
            f"{list(levels)} -> up to {planned} requests; calibration={calibration}"
        )
        return {"planned_requests": planned}
    table = level_table(ctx, questions, levels, cache_only=False)
    tag = active_tag(ctx)
    table_to_csv(table, ctx.paths.out(f"pilot/pilot_levels_{tag}.csv"))
    summary = pilot_summary(ctx, table, levels)
    atomic_write_json(ctx.paths.out(f"pilot/pilot_summary_{tag}.json"), summary)
    atomic_write_json(ctx.paths.out("pilot_summary.json"), summary)
    if ctx.family == CONSISTENCY:
        examples = pilot_examples(table, levels)
        atomic_write_json(ctx.paths.out("pilot/pilot_examples.json"), examples)
        for example in examples:
            ctx.echo(json.dumps(example, ensure_ascii=False))
    if calibration:
        report = calibration_report(
            table,
            ctx.cfg["execution"]["calibration_gate"],
            {"name": ctx.settings.name, "digest": ctx.settings.digest},
        )
        atomic_write_json(ctx.paths.out(f"pilot/pilot_calibration_{tag}.json"), report)
        atomic_write_json(ctx.paths.out("pilot_calibration.json"), report)
        summary = {**summary, "calibration_gate_passed": report["gate_passed"]}
    ctx.state.set(
        "pilot",
        {
            "done": True,
            "time": utc_now(),
            "model": ctx.settings.name,
            "calibration": calibration,
        },
    )
    ctx.log(
        f"pilot done ({ctx.settings.name}): valid_rate_initial "
        f"{summary['valid_rate_initial']} after_repair "
        f"{summary['valid_rate_after_repair']} repair_rate {summary['repair_rate']}"
    )
    ctx.note_reuse()
    return summary


def pilot_summary(ctx: Context, table: pd.DataFrame, levels: Sequence[int]) -> dict:
    per_q = table.groupby("question_id").first()
    columns = V2_ROLE_COLUMNS if ctx.family == CONSISTENCY else ROLE_COLUMNS
    initial, after, repair = format_rates(table, per_q, columns)
    reader_acc = table.groupby("level")["reader_correct"].mean()
    return {
        "model": ctx.settings.name,
        "digest": ctx.settings.digest,
        "model_tag": active_tag(ctx),
        "prompt_hashes": ctx.prompts.hashes,
        "questions": int(table["question_id"].nunique()),
        "images": int(table["image_id"].nunique()),
        "levels": list(levels),
        "parse_version": PARSE_VERSION,
        "valid_rate_initial": initial,
        "valid_rate_after_repair": after,
        "repair_rate": repair,
        "format_gate_uses": "valid_rate_initial",
        "format_valid_target": FORMAT_VALID_TARGET,
        "format_valid_ok": all(v >= FORMAT_VALID_TARGET for v in initial.values()),
        "reader_accuracy_by_level": {
            str(int(k)): float(v) for k, v in reader_acc.items()
        },
        "n_items_distribution": (
            {str(k): int(v) for k, v in Counter(per_q["n_items"]).items()}
            if "n_items" in per_q
            else None
        ),
        "family": ctx.family,
        "level1_headroom_note": "L1 reader accuracy >= 0.95 means little room to adapt",
    }


ROLE_COLUMNS = {"reader": "reader", "direct": "direct", "checklist": "check"}
V2_ROLE_COLUMNS = {
    "reader": "reader",
    "identify": "identify",
    "attr_blind": "attr",
    "recompose": "recompose",
}
NOT_CALLED = frozenset({"not_run", "planner_failure", "skipped"})
EXAMPLE_COUNT = 5


def planned_requests(ctx: Context, n_questions: int, n_levels: int) -> int:
    if ctx.family == CONSISTENCY:
        return planned_consistency_requests(n_questions, n_levels)
    return planned_request_count(n_questions, n_levels, LEVEL_ROLES)


def pilot_examples(table: pd.DataFrame, levels: Sequence[int]) -> list[dict[str, Any]]:
    """First questions' planner_v2 output and identify answers at min/max level."""
    low, high = min(levels), max(levels)
    examples = []
    for qid in list(dict.fromkeys(table["question_id"]))[:EXAMPLE_COUNT]:
        rows = table[table["question_id"] == qid].set_index("level")
        first = rows.iloc[0]
        examples.append(
            {
                "question_id": qid,
                "question": first["question"],
                "planner_status": first["planner_status"],
                "target": first["target"],
                "synonyms": first["synonyms"],
                "attribute": first["plan_attribute"],
                "referring_phrase": first["referring_phrase"],
                "identify_question": first["identify_question"],
                "attr_question": first["attr_question"],
                **{
                    f"L{t}": {
                        "identify": rows.loc[t, "identify_answer"],
                        "attr_blind": rows.loc[t, "attr_answer"],
                        "recompose": rows.loc[t, "recompose_answer"],
                        "reader": rows.loc[t, "reader_answer"],
                        "abc": [bool(rows.loc[t, x]) for x in ("a", "b", "c")],
                    }
                    for t in (low, high)
                },
            }
        )
    return examples


def format_rates(
    table: pd.DataFrame,
    per_question: pd.DataFrame,
    role_columns: Mapping[str, str] = ROLE_COLUMNS,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Per role over calls actually made: first-attempt valid rate, valid rate
    after the single repair, and the share of calls that needed a repair."""
    initial: dict[str, float] = {}
    after: dict[str, float] = {}
    repair: dict[str, float] = {}
    frames = {
        role: table[~table[f"{col}_status"].isin(NOT_CALLED)]
        for role, col in role_columns.items()
    }
    for role, col in role_columns.items():
        calls = frames[role]
        if calls.empty:
            continue
        initial[role] = float(calls[f"{col}_initial_valid"].astype(bool).mean())
        after[role] = float((calls[f"{col}_status"] == VALID).mean())
        repair[role] = float((calls[f"{col}_attempts"] > 1).mean())
    planner = per_question[per_question["planner_status"] != "not_run"]
    if not planner.empty:
        initial["planner"] = float(planner["planner_initial_valid"].astype(bool).mean())
        after["planner"] = float((planner["planner_status"] == VALID).mean())
        repair["planner"] = float((planner["planner_attempts"] > 1).mean())
    return initial, after, repair


# ---- infer -----------------------------------------------------------------------
def active_tag(ctx: Context) -> str:
    return model_tag(ctx.settings.name, ctx.settings.digest)


def require_reader_prompt(ctx: Context, screening: Mapping[str, Any]) -> None:
    """The common reader prompt must be the one used for screening (§3.3)."""
    if screening.get("reader_prompt_hash") != ctx.prompts.hashes[READER_ROLE]:
        raise StageError(
            "answer_only prompt changed since screening: re-run `screen` so the "
            "selected set and the Level-20 reader use the same prompt"
        )


def require_screening_match(ctx: Context, split: str) -> None:
    screening = (ctx.state.get("screening") or {}).get(split)
    require(bool(screening), f"`screen --split {split}` has not run")
    if screening.get("digest") != ctx.settings.digest:
        raise StageError(
            "screening model digest differs from the active model; re-screen"
        )
    require_reader_prompt(ctx, screening)
    if screening.get("parse_version") != PARSE_VERSION:
        raise StageError(
            f"screening used parser {screening.get('parse_version')}, active is "
            f"{PARSE_VERSION}: re-run `screen --split {split}`"
        )


def pilot_gate_failures(ctx: Context) -> list[str]:
    """§11 format-valid gate and §16 A4 calibration gate for the active model."""
    tag = active_tag(ctx)
    summary_path = ctx.paths.out(f"pilot/pilot_summary_{tag}.json")
    calib_path = ctx.paths.out(f"pilot/pilot_calibration_{tag}.json")
    needs_calibration = ctx.family != CONSISTENCY  # §17 uses no confidence
    if not summary_path.exists() or (needs_calibration and not calib_path.exists()):
        return [f"no pilot with --calibration for the active model {tag}"]
    summary = read_json(summary_path)
    if summary.get("prompt_hashes") != ctx.prompts.hashes:
        return ["prompts changed since the pilot: re-run `pilot --calibration`"]
    if summary.get("parse_version") != PARSE_VERSION:
        return ["parser changed since the pilot: re-run `pilot --calibration`"]
    failures: list[str] = []
    rate = min(summary["valid_rate_initial"].values())  # §11 gate: first attempt
    revisions = int(read_json(ctx.paths.out("prompt_hashes.json")).get("revisions", 0))
    if rate < FORMAT_VALID_TARGET and revisions == 0:
        failures.append(
            f"format-valid rate {rate:.3f} < {FORMAT_VALID_TARGET}: repair the prompt "
            "first (one dev revision, §11) and re-run the pilot"
        )
    elif rate < FORMAT_VALID_EXIT:
        failures.append(
            f"format-valid rate {rate:.3f} < {FORMAT_VALID_EXIT} after a prompt "
            "revision: §11 exit branch (model/method unsuitable; end with pilot report)"
        )
    if needs_calibration and not read_json(calib_path)["gate_passed"]:
        switch = ctx.state.get("model_switch") or {}
        if not switch.get("active"):
            failures.append(
                "A4 calibration gate failed: re-run `pilot --use-fallback-model "
                "--calibration` before any dev inference"
            )
        else:
            failures.append(
                "A4 calibration gate failed for both models: exit branch "
                "(no resolution signal in self-reported confidence)"
            )
    return failures


def enforce_pilot_gates(ctx: Context, force_continue: bool) -> None:
    failures = pilot_gate_failures(ctx)
    if not failures:
        return
    if not force_continue:
        raise StageError(
            "; ".join(failures)
            + " (use --force-continue only to proceed knowingly past a documented "
            "exit branch; results then become exploratory)"
        )
    ctx.log(f"GATE OVERRIDE (--force-continue): {failures}")
    ctx.state.set("gate_override", {"time": utc_now(), "failures": failures})


def check_infer_preconditions(ctx: Context, split: str) -> None:
    require_screening_match(ctx, split)
    if split == "test":
        ctx.require_frozen_match()


def infer(
    ctx: Context, split: str, levels: Sequence[int], force_continue: bool = False
) -> dict[str, Any]:
    ctx.set_stage(f"infer_{split}")
    check_infer_preconditions(ctx, split)
    if split == "dev":
        enforce_pilot_gates(ctx, force_continue)
    questions = ctx.selected(split)
    planned = planned_requests(ctx, len(questions), len(levels))
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] infer {split}: {len(questions)} questions x {len(levels)} levels "
            f"-> up to {planned} requests (cached ones are skipped); "
            f"table -> {split_table_path(ctx, split)}"
        )
        return {"planned_requests": planned}
    ctx.state.set(
        "infer", {**(ctx.state.get("infer") or {}), split: {"started": utc_now()}}
    )
    table = level_table(ctx, questions, levels, cache_only=False)
    complete = sorted(levels) == list(range(1, ctx.n_levels + 1))
    if complete:
        table_to_csv(table, split_table_path(ctx, split))
    info = {
        "started": True,
        "complete": complete,
        "questions": len(questions),
        "levels": list(levels),
        "time": utc_now(),
    }
    ctx.state.set("infer", {**(ctx.state.get("infer") or {}), split: info})
    ctx.log(f"infer {split}: {len(questions)} questions, levels {list(levels)}")
    ctx.note_reuse()
    return info


def full_table(ctx: Context, split: str) -> pd.DataFrame:
    """Replay every (question, level) from cache only (no model calls)."""
    levels = list(range(1, ctx.n_levels + 1))
    return level_table(ctx, ctx.selected(split), levels, cache_only=True)


# ---- freeze / evaluate -------------------------------------------------------------
def provenance(ctx: Context) -> dict[str, Any]:
    eff = ctx.paths.out("effective_config.json")
    return {
        "model": ctx.settings.name,
        "digest": ctx.settings.digest,
        "prompt_hashes": ctx.prompts.hashes,
        "source_config_sha256": sha256_file(ctx.config_path),
        "effective_config_sha256": sha256_json(read_json(eff)["effective"])
        if eff.exists()
        else None,
        "candidate_manifest_sha256": sha256_file(
            ctx.paths.data("candidate_manifest.jsonl")
        ),
    }


def freeze(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("freeze")
    if ctx.family == CONSISTENCY:
        raise StageError(
            "freeze is not available for the consistency family (plan §17: threshold "
            "freezing and test wait for the user's decision); use `devgrid`"
        )
    require_screening_match(ctx, "dev")
    require(
        bool((ctx.state.get("infer") or {}).get("dev", {}).get("complete")),
        "dev `infer --levels all` must complete before freeze",
    )
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] freeze: grid over {len(ctx.selected('dev'))} dev questions -> "
            f"{ctx.paths.out('frozen_policy.json')}"
        )
        return {}
    table = full_table(ctx, "dev")
    table_to_csv(table, split_table_path(ctx, "dev"))
    frozen = freeze_policy(
        table, ctx.cfg["policy"], ctx.n_levels, ctx.paths.output_root, provenance(ctx)
    )
    ctx.state.set("freeze", {"done": True, "time": utc_now()})
    ctx.log(f"policy frozen: primary={frozen['selections'][frozen['primary_target']]}")
    return frozen


def run_evaluate(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("evaluate")
    frozen_path = ctx.paths.out("frozen_policy.json")
    require(frozen_path.exists(), "evaluate needs a frozen policy")
    ctx.require_frozen_match()
    require_screening_match(ctx, "test")
    require(
        bool((ctx.state.get("infer") or {}).get("test", {}).get("complete")),
        "test `infer --levels all` must complete before evaluate",
    )
    if ctx.dry_run:
        ctx.echo(
            "[dry-run] evaluate: stops.csv, method_summary.csv, paired_comparisons.csv, "
            "bootstrap_summary.json (no model calls)"
        )
        return {}
    table = full_table(ctx, "test")
    table_to_csv(table, split_table_path(ctx, "test"))
    gt = ctx.gt(table["question_id"].unique().tolist())
    summary = evaluate(
        table,
        read_json(frozen_path),
        ctx.cfg["evaluation"],
        ctx.n_levels,
        ctx.paths.output_root,
        gt,
    )
    ctx.state.set("evaluate", {"done": True, "time": utc_now()})
    write_dataset_flow(ctx)
    ctx.log(
        f"evaluate done: {summary['n_questions']} questions, {summary['n_images']} images"
    )
    return summary


# ---- v2 dev grid (plan §17) ----------------------------------------------------------
def devgrid(ctx: Context) -> dict[str, Any]:
    """Dev-only θ x k grid, stop histograms and component rates; never freezes."""
    ctx.set_stage("devgrid")
    require(ctx.family == CONSISTENCY, "devgrid is for the consistency family only")
    require_screening_match(ctx, "dev")
    require(
        bool((ctx.state.get("infer") or {}).get("dev", {}).get("complete")),
        "dev `infer --levels all` must complete before devgrid",
    )
    variants = consistency_variants(ctx.cfg["policy"], ctx.n_levels)
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] devgrid: {[v.label for v in variants]} on dev; no model calls"
        )
        return {}
    table = full_table(ctx, "dev")
    table_to_csv(table, split_table_path(ctx, "dev"))
    comparison = ctx.cfg.get("comparison", {}).get("v1_output_root")
    v1_root = resolve_path(ctx.paths.repo_root, comparison) if comparison else None
    target = float(ctx.cfg["policy"]["primary_dev_accuracy_target"])
    summary = run_devgrid(table, variants, target, ctx.paths.output_root, v1_root)
    ctx.state.set(
        "devgrid", {"done": True, "time": utc_now(), "abort": summary["abort"]}
    )
    ctx.log(
        f"devgrid: best={summary['best_feasible_variant']} abort={summary['abort']}"
    )
    ctx.note_reuse()
    return summary
