"""Model stages: preflight, pilot (+A4 calibration), infer, freeze, evaluate."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from . import preflight as preflight_mod
from .calibration import calibration_report
from .client import model_tag
from .common import (
    StageError,
    atomic_write_json,
    read_json,
    sha256_file,
    sha256_json,
    utc_now,
    write_effective_config,
)
from .context import Context, require, table_to_csv
from .evaluate import add_correctness, evaluate
from .inference import LEVEL_ROLES, READER_ROLE, planned_request_count
from .policy import freeze_policy
from .stages_data import write_dataset_flow

VALID = "valid"
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
    planned = planned_request_count(len(questions), len(levels), LEVEL_ROLES)
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] pilot: {len(questions)} questions x levels {list(levels)} -> "
            f"{planned} requests (planner + reader/direct/checklist); "
            f"calibration={calibration}"
        )
        return {"planned_requests": planned}
    table = level_table(ctx, questions, levels, cache_only=False)
    tag = active_tag(ctx)
    table_to_csv(table, ctx.paths.out(f"pilot/pilot_levels_{tag}.csv"))
    summary = pilot_summary(ctx, table, levels)
    atomic_write_json(ctx.paths.out(f"pilot/pilot_summary_{tag}.json"), summary)
    atomic_write_json(ctx.paths.out("pilot_summary.json"), summary)
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
    ctx.log(f"pilot done ({ctx.settings.name}): valid rates {summary['valid_rates']}")
    ctx.note_reuse()
    return summary


def pilot_summary(ctx: Context, table: pd.DataFrame, levels: Sequence[int]) -> dict:
    per_q = table.groupby("question_id").first()
    rates = {
        "reader": float((table["reader_status"] == VALID).mean()),
        "direct": float((table["direct_status"] == VALID).mean()),
        "checklist": float((table["check_status"] == VALID).mean()),
        "planner": float((per_q["planner_status"] == VALID).mean()),
    }
    reader_acc = table.groupby("level")["reader_correct"].mean()
    return {
        "model": ctx.settings.name,
        "digest": ctx.settings.digest,
        "model_tag": active_tag(ctx),
        "prompt_hashes": ctx.prompts.hashes,
        "questions": int(table["question_id"].nunique()),
        "images": int(table["image_id"].nunique()),
        "levels": list(levels),
        "valid_rates": rates,
        "format_valid_target": FORMAT_VALID_TARGET,
        "format_valid_ok": all(v >= FORMAT_VALID_TARGET for v in rates.values()),
        "reader_accuracy_by_level": {
            str(int(k)): float(v) for k, v in reader_acc.items()
        },
        "n_items_distribution": {
            str(k): int(v) for k, v in Counter(per_q["n_items"]).items()
        },
        "level1_headroom_note": "L1 reader accuracy >= 0.95 means little room to adapt",
    }


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


def pilot_gate_failures(ctx: Context) -> list[str]:
    """§11 format-valid gate and §16 A4 calibration gate for the active model."""
    tag = active_tag(ctx)
    summary_path = ctx.paths.out(f"pilot/pilot_summary_{tag}.json")
    calib_path = ctx.paths.out(f"pilot/pilot_calibration_{tag}.json")
    if not summary_path.exists() or not calib_path.exists():
        return [f"no pilot with --calibration for the active model {tag}"]
    summary = read_json(summary_path)
    if summary.get("prompt_hashes") != ctx.prompts.hashes:
        return ["prompts changed since the pilot: re-run `pilot --calibration`"]
    failures: list[str] = []
    rate = min(summary["valid_rates"].values())
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
    if not read_json(calib_path)["gate_passed"]:
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
    planned = planned_request_count(len(questions), len(levels), LEVEL_ROLES)
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
