"""Oracle-checklist probe runner (exploratory, not pre-registered).

Same 30 questions, level images, model and decoding as ``probe_multi_rel``.
Only the oracle checklist items are new calls: observer_v3 (namespace ``lp``,
top_logprobs 10) per oracle item per level, cached under the NEW run dir
``outputs/gqa_stop/probe_oracle/observer_v3_lp/<model>/parse-v2/<qid>/<LL>_o<k>.json``.
reader_lp, observer_v3_lp_direct, plans_v3 and the planner items are read
(cache only, cache-key checked, never re-requested) from probe_multi_rel.

Subcommands:
  plans   build oracle plans from the probe selection (GQA programs) and write
          data/processed/gqa_stop/probe_oracle/plans.json
  show    print every oracle plan next to the planner_v3 plan (markdown)
  infer   request the oracle items; --max-seconds (checked between levels,
          resumable from the cache); --dry-run prints the planned call count
  table   cache-only level table: B1 / A1 / P_min (planner_v3) from the
          reference records + the oracle scores -> level_table_oracle.csv
Usage (repo root):
  C=configs/gqa_stop_probe_oracle.json
  uv run python scripts/gqa_probe_oracle_run.py plans --config $C
  uv run python scripts/gqa_probe_oracle_run.py infer --config $C --dry-run
  timeout 585 uv run python scripts/gqa_probe_oracle_run.py infer --config $C --max-seconds 540
  uv run python scripts/gqa_probe_oracle_run.py table --config $C
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from milestone1.gqa_stop.client import (
    ROLE_DIRS,
    RequestEngine,
    TransportResult,
    record_tag,
)
from milestone1.gqa_stop.common import (
    StageError,
    atomic_write_json,
    load_effective_config,
    read_json,
    read_jsonl,
    resolve_path,
    sha256_text,
    utc_now,
)
from milestone1.gqa_stop.context import Context, require
from milestone1.gqa_stop.evaluate import is_correct
from milestone1.gqa_stop.images import ImageSeries
from milestone1.gqa_stop.logprob_v4 import item_signals, observer_lp_spec
from milestone1.gqa_stop.oracle_plan import RULES_VERSION, answer_leak, build_plans
from milestone1.gqa_stop.oracle_score import score_level
from milestone1.gqa_stop.planner import plan_from_record
from milestone1.gqa_stop.sampled import (
    OBSERVER_ROLE,
    planner_v3_spec,
    validate_observation,
)
from milestone1.gqa_stop.v27_data import load_gt, load_manifest
from milestone1.gqa_stop.v27_infer import (
    direct_resolved_spec,
    level_row,
    reader_lp_spec,
)

SPLIT = "dev"  # the probe runs as the v27 "dev" split (questions_dev.jsonl)
STAGE = "oracle_infer"
STATE_KEY = "oracle_infer"
OBSERVER_DIR = f"{ROLE_DIRS[OBSERVER_ROLE]}_lp"
CHECK_COLUMNS = ("reader_pans", "dres_score", "pans_min", "reader_answer")
MAX_ORACLE_ITEMS = 4


class NoCallTransport:
    """Transport of the cache-only engines: any send is a bug."""

    def send(self, payload: dict[str, Any]) -> TransportResult:
        raise RuntimeError("cache-only engine attempted a model call")


# ---- setup ---------------------------------------------------------------------------------
def make_context(config: Path, dry_run: bool) -> Context:
    cfg, paths = load_effective_config(config)
    return Context(config, cfg, paths, dry_run)


def oracle_cfg(ctx: Context) -> dict[str, Any]:
    return dict(ctx.cfg["oracle"])


def repo_path(ctx: Context, value: str) -> Path:
    return resolve_path(ctx.paths.repo_root, value)


def plans_path(ctx: Context) -> Path:
    return repo_path(ctx, oracle_cfg(ctx)["plans"])


def reference_context(ctx: Context) -> Context:
    return make_context(repo_path(ctx, oracle_cfg(ctx)["reference_config"]), True)


def readonly_series(ctx: Context) -> ImageSeries:
    """The probe_multi_rel level images; never regenerated here."""
    return ImageSeries(
        ctx.cfg["images"],
        ctx.paths.images_root,
        ctx.paths.levels_dir,
        ctx.paths.data("image_manifest.jsonl"),
        read_only=True,
    )


def cache_engine(ctx: Context) -> RequestEngine:
    """Cache-only engine over ``ctx``'s output root (no digest check, no calls)."""
    require(bool(ctx.settings.digest), f"no model digest in {ctx.config_path}")
    return RequestEngine(
        ctx.settings, ctx.prompts, NoCallTransport(), ctx.paths.output_root
    )


def cached(engine: RequestEngine, spec: Any) -> dict[str, Any]:
    record = engine.lookup(spec)
    if record is None:
        raise StageError(
            f"cache miss: {spec.role}/{spec.namespace} {spec.question_id} "
            f"L{spec.level}{spec.suffix} under {engine.output_root}"
        )
    return record


def load_plans(ctx: Context) -> dict[str, dict[str, Any]]:
    path = plans_path(ctx)
    require(path.exists(), f"{path} missing: run `plans` first")
    return {p["question_id"]: p for p in read_json(path)["plans"]}


def plans_hash(ctx: Context) -> str:
    return sha256_text(plans_path(ctx).read_text())


# ---- plans ---------------------------------------------------------------------------------
def cmd_plans(ctx: Context, force: bool) -> None:
    selection = read_jsonl(repo_path(ctx, oracle_cfg(ctx)["selection"]))
    manifest_ids = [q["question_id"] for q in load_manifest(ctx, SPLIT)]
    by_id = {r["question_id"]: r for r in selection}
    require(set(by_id) == set(manifest_ids), "selection != probe manifest")
    plans = build_plans([by_id[q] for q in manifest_ids])
    leaks = {
        p["question_id"]: answer_leak(p, str(by_id[p["question_id"]]["answer"]))
        for p in plans
    }
    require(not any(leaks.values()), f"answer leak in item text: {leaks}")
    too_long = [p["question_id"] for p in plans if len(p["items"]) > MAX_ORACLE_ITEMS]
    payload = {
        "rules_version": RULES_VERSION,
        "source": oracle_cfg(ctx)["selection"],
        "note": "built from the GQA semantic program only; GT answer used only for "
        "the leak check (no item contains a GT word absent from the question)",
        "plans": plans,
    }
    path = plans_path(ctx)
    text = json.dumps(payload, indent=1) + "\n"
    if path.exists() and path.read_text() != text and not force:
        raise StageError(f"{path} exists with different content (use --force)")
    if ctx.dry_run:
        print(f"[dry-run] would write {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    flagged = [p["question_id"] for p in plans if not p["sensible"]]
    print(
        f"plans: {len(plans)} questions, {sum(len(p['items']) for p in plans)} items, "
        f"not sensible {flagged}, > {MAX_ORACLE_ITEMS} items {too_long} -> {path}"
    )


def cmd_show(ctx: Context) -> None:
    """Markdown: question, GT, oracle plan and planner_v3 plan per question."""
    plans = load_plans(ctx)
    ref = reference_context(ctx)
    engine = cache_engine(ref)
    gt = load_gt(ctx, SPLIT)
    for q in load_manifest(ctx, SPLIT):
        plan = plans[q["question_id"]]
        planner = plan_from_record(q["question_id"], cached(engine, planner_v3_spec(q)))
        flags = f"  flags: {', '.join(plan['flags'])}" if plan["flags"] else ""
        print(
            f"{q['attribute']} {q['question_id']} {q['question']!r} -> {gt[q['question_id']]}"
        )
        print(f"  oracle{flags}")
        for item in plan["items"]:
            tag = item["role"] + (f"/{item['logic']}" if item["logic"] else "")
            req = f" req={item['required_answer']}" if item["required_answer"] else ""
            deps = ",".join(item["depends_on"])
            print(f"    {item['id']} [{deps}] {tag}{req}: {item['question']}")
        print("  planner_v3")
        for item in planner.items:
            deps = ",".join(item["depends_on"])
            print(f"    {item['id']} [{deps}]: {item['question']}")


# ---- inference -----------------------------------------------------------------------------
def planned_counts(
    ctx: Context, plans: Mapping[str, Mapping[str, Any]], digest: str
) -> dict[str, Any]:
    questions = load_manifest(ctx, SPLIT)
    levels = range(1, ctx.n_levels + 1)
    base = ctx.paths.output_root / OBSERVER_DIR / record_tag(ctx.settings.name, digest)
    by_type: Counter[str] = Counter()
    items_by_type: Counter[str] = Counter()
    present = 0
    for q in questions:
        items = plans[q["question_id"]]["items"]
        by_type[q["attribute"]] += len(items) * len(levels)
        items_by_type[q["attribute"]] += len(items)
        for level in levels:
            for item in items:
                path = base / q["question_id"] / f"{level:02d}_{item['id']}.json"
                present += int(path.exists())
    total = sum(by_type.values())
    return {
        "questions": len(questions),
        "levels": len(levels),
        "items": sum(items_by_type.values()),
        "items_by_type": dict(sorted(items_by_type.items())),
        "calls_by_type": dict(sorted(by_type.items())),
        "planned_calls": total,
        "record_files_present": present,
        "remaining_calls": total - present,
    }


def reference_digest(ctx: Context) -> str:
    digest = reference_context(ctx).settings.digest
    require(bool(digest), "reference run has no recorded model digest")
    return str(digest)


def require_ready(ctx: Context, ref_digest: str) -> None:
    require(bool((ctx.state.get("preflight") or {}).get("done")), "run `preflight`")
    require(
        bool((ctx.state.get("v27_envcheck") or {}).get("passed")),
        "run `v27-envcheck` on the probe_oracle config first",
    )
    require(
        bool((ctx.state.get("determinism") or {}).get("passed")),
        "run `v27-detgate` on the probe_oracle config first",
    )
    require(
        ctx.settings.digest == ref_digest,
        f"model digest {ctx.settings.digest} != reference {ref_digest}",
    )


def cmd_infer(ctx: Context, max_seconds: float | None) -> None:
    plans = load_plans(ctx)
    ref_digest = reference_digest(ctx)
    counts = planned_counts(ctx, plans, ref_digest)
    if ctx.dry_run:
        print(f"[dry-run] oracle infer: {json.dumps(counts)}")
        return
    require_ready(ctx, ref_digest)
    current_hash = plans_hash(ctx)
    previous = (ctx.state.get(STATE_KEY) or {}).get("plans_sha256")
    require(previous in (None, current_hash), "plans.json changed after inference")
    ctx.set_stage(STAGE)
    engine = ctx.engine()
    series = readonly_series(ctx)
    start, calls0, hits0 = time.monotonic(), engine.calls, engine.cache_hits
    questions = load_manifest(ctx, SPLIT)
    done, stopped = 0, False
    for q in questions:
        items = plans[q["question_id"]]["items"]
        for level in range(1, ctx.n_levels + 1):
            if max_seconds is not None and time.monotonic() - start > max_seconds:
                stopped = True
                break
            image = series.bytes_for(str(q["image_id"]), level)
            for item in items:
                engine.run(
                    observer_lp_spec(q, level, image, item), validate_observation
                )
        if stopped:
            break
        done += 1
    elapsed = time.monotonic() - start
    new_calls = engine.calls - calls0
    progress = {
        "time": utc_now(),
        "plans_sha256": current_hash,
        "questions_done": done,
        "questions_total": len(questions),
        "new_calls": new_calls,
        "cache_hits": engine.cache_hits - hits0,
        "elapsed_s": round(elapsed, 1),
        "s_per_call": round(elapsed / new_calls, 3) if new_calls else 0.0,
        "complete": done == len(questions),
    }
    ctx.state.set(STATE_KEY, progress)
    log_path = ctx.paths.out("logs/oracle_infer_progress.json")
    history = read_json(log_path) if log_path.exists() else []
    atomic_write_json(log_path, [*history, progress])
    ctx.log(f"oracle infer chunk: {progress}")


# ---- level table ---------------------------------------------------------------------------
def oracle_columns(
    plan: Mapping[str, Any], signals: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    scores = score_level(plan["items"], signals)
    gated = scores.pop("gated", {})
    row: dict[str, Any] = {
        **scores,
        "oracle_n_items": len(plan["items"]),
        "oracle_sensible": bool(plan["sensible"]),
    }
    for k, item in enumerate(plan["items"], start=1):
        sig = signals[item["id"]]
        row |= {
            f"oitem{k}_id": item["id"],
            f"oitem{k}_role": item["role"],
            f"oitem{k}_logic": item["logic"],
            f"oitem{k}_question": item["question"],
            f"oitem{k}_answer": sig["answer"],
            f"oitem{k}_valid": sig["valid"],
            f"oitem{k}_resolved": sig["resolved"] if sig["valid"] else False,
            f"oitem{k}_pans": sig["pans"],
            f"oitem{k}_gated": gated.get(item["id"], 0.0),
        }
    return row


def cmd_table(ctx: Context) -> pd.DataFrame:
    state = ctx.state.get(STATE_KEY) or {}
    require(bool(state.get("complete")), "oracle inference is not complete")
    require(state.get("plans_sha256") == plans_hash(ctx), "plans.json changed")
    plans = load_plans(ctx)
    ref = reference_context(ctx)
    ref_engine, own_engine = cache_engine(ref), cache_engine(ctx)
    series = readonly_series(ctx)
    gt = load_gt(ctx, SPLIT)
    screening = {
        r["question_id"]: r
        for r in read_jsonl(ref.paths.selection("screening_dev.jsonl"))
    }
    rows = []
    for q in load_manifest(ctx, SPLIT):
        qid = q["question_id"]
        plan_record = cached(ref_engine, planner_v3_spec(q))
        plan = plan_from_record(qid, plan_record)
        oplan = plans[qid]
        for level in range(1, ctx.n_levels + 1):
            image = series.bytes_for(str(q["image_id"]), level)
            reader = cached(ref_engine, reader_lp_spec(q, level, image))
            direct = cached(ref_engine, direct_resolved_spec(q, level, image))
            items = {
                i["id"]: cached(ref_engine, observer_lp_spec(q, level, image, i))
                for i in plan.items
            }
            row = level_row(q, level, plan, plan_record, reader, direct, items)
            signals = {
                i["id"]: item_signals(
                    cached(own_engine, observer_lp_spec(q, level, image, i)), None
                )
                for i in oplan["items"]
            }
            rows.append(row | oracle_columns(oplan, signals))
    table = pd.DataFrame(rows)
    table["reader_correct"] = [
        is_correct(None if pd.isna(a) else str(a), gt[qid])
        for qid, a in zip(table["question_id"], table["reader_answer"], strict=True)
    ]
    table["label"] = table["question_id"].map(lambda q: screening[q]["label"])
    table["gt_answer"] = table["question_id"].map(gt)
    check_reference(ctx, ref, table)
    out = ctx.paths.out(oracle_cfg(ctx)["table"])
    table.to_csv(out, index=False)
    ctx.log(f"oracle level table: {len(table)} rows -> {out}")
    return table


def check_reference(ctx: Context, ref: Context, table: pd.DataFrame) -> None:
    """B1 / A1 / P_min must equal the reference run's own level table."""
    ref_table = pd.read_csv(
        ref.paths.out("level_table_dev.csv"), dtype={"question_id": str}
    )
    keys = ["question_id", "level"]
    merged = table[[*keys, *CHECK_COLUMNS]].merge(
        ref_table[[*keys, *CHECK_COLUMNS]], on=keys, suffixes=("", "_ref")
    )
    require(len(merged) == len(table), "reference level table does not cover the probe")
    for column in CHECK_COLUMNS:
        a, b = merged[column], merged[f"{column}_ref"]
        if column == "reader_answer":
            same = (a.astype(str) == b.astype(str)).all()
        else:
            same = ((a.astype(float) - b.astype(float)).abs() < 1e-12).all()
        require(bool(same), f"{column} differs from the reference level table")
    ctx.log(f"reference check: {', '.join(CHECK_COLUMNS)} identical to probe_multi_rel")


# ---- main ----------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gqa_probe_oracle_run")
    parser.add_argument("command", choices=("plans", "show", "infer", "table"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--force", action="store_true", help="plans: overwrite")
    args = parser.parse_args(argv)
    ctx = make_context(args.config, args.dry_run)
    try:
        if args.command == "plans":
            cmd_plans(ctx, args.force)
        elif args.command == "show":
            cmd_show(ctx)
        elif args.command == "infer":
            cmd_infer(ctx, args.max_seconds)
        else:
            cmd_table(ctx)
    except StageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
