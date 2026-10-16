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
          reference records + the oracle scores (+ the planner_v4 arm scores
          when ``oracle.v4_config`` is set) -> level_table_oracle.csv
planner_v4 arm (run with --config configs/gqa_stop_probe_oracle_v4.json, its own
run dir outputs/gqa_stop/probe_oracle_v4/):
  v4-infer    planner_v4 (one text-only call per question, plans_v4/) and then
              observer_v3 (namespace lp_v4) per v4 item per level; --dry-run,
              --max-seconds as for ``infer``
  v4-show     markdown: oracle, planner_v3 and planner_v4 plans per question
  v4-quality  plan quality of v4 vs oracle -> plan_quality_v4.csv / .json
  v4-audit    v27 record audit (timing, argmax) on plans_v4 + observer_v3_lp_v4
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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from milestone1.gqa_stop.audit_v22 import audit_record
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
from milestone1.gqa_stop.planner_v4 import (
    observer_v4_spec,
    plan_quality,
    plan_v4_items,
    plan_v4_settings,
    planner_v4_spec,
    validate_plan_v4,
)
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
V4_STAGE = "v4_infer"
V4_STATE_KEY = "v4_infer"
V4_PREFIX = "v4"
V4_OBSERVER_DIR = f"{OBSERVER_DIR}_v4"


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
def final_answer(items: Sequence[Mapping[str, Any]], signals: Mapping[str, Any]) -> str:
    """Answer implied by the final group: the single final item's answer, or for
    and/or operands "yes"/"no" by the logic ("" when an operand is undecided)."""
    finals = [i for i in items if i["role"] == "final"]
    if not finals:
        return ""
    answers = [str(signals[i["id"]]["answer"] or "").strip().lower() for i in finals]
    logic = finals[0].get("logic")
    if logic not in ("and", "or") or len(finals) == 1:
        return answers[-1]
    decisive = "yes" if logic == "or" else "no"
    if decisive in answers:
        return decisive
    other = "no" if decisive == "yes" else "yes"
    return other if all(a == other for a in answers) else ""


def plan_columns(
    items: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
    prefix: str,
    item_prefix: str,
) -> dict[str, Any]:
    """Aggregated scores (``oracle_*`` keys renamed to ``<prefix>_*``) + items."""
    scores = score_level(items, signals)
    gated = scores.pop("gated", {})
    absent = scores.pop("object_absent")
    row: dict[str, Any] = {
        f"{prefix}_{key.removeprefix('oracle_')}": value
        for key, value in scores.items()
    }
    row[f"{prefix}_object_absent" if prefix != "oracle" else "object_absent"] = absent
    row[f"{prefix}_final_answer"] = final_answer(items, signals) if items else ""
    for k, item in enumerate(items, start=1):
        sig = signals[item["id"]]
        row |= {
            f"{item_prefix}{k}_id": item["id"],
            f"{item_prefix}{k}_role": item["role"],
            f"{item_prefix}{k}_logic": item["logic"],
            f"{item_prefix}{k}_question": item["question"],
            f"{item_prefix}{k}_answer": sig["answer"],
            f"{item_prefix}{k}_valid": sig["valid"],
            f"{item_prefix}{k}_resolved": sig["resolved"] if sig["valid"] else False,
            f"{item_prefix}{k}_pans": sig["pans"],
            f"{item_prefix}{k}_gated": gated.get(item["id"], 0.0),
        }
    return row


def oracle_columns(
    plan: Mapping[str, Any], signals: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    row = plan_columns(plan["items"], signals, "oracle", "oitem")
    return row | {
        "oracle_n_items": len(plan["items"]),
        "oracle_sensible": bool(plan["sensible"]),
    }


class V4Arm:
    """Cache-only access to the planner_v4 arm (its own run dir)."""

    def __init__(self, ctx: Context) -> None:
        path = oracle_cfg(ctx).get("v4_config")
        self.ctx = make_context(repo_path(ctx, str(path)), True) if path else None
        if self.ctx is None:
            return
        state = self.ctx.state.get(V4_STATE_KEY) or {}
        require(bool(state.get("complete")), "planner_v4 inference is not complete")
        self.engine = cache_engine(self.ctx)
        self.plans = {
            q["question_id"]: plan_v4_items(cached(self.engine, planner_v4_spec(q)))
            for q in load_manifest(self.ctx, SPLIT)
        }
        require(
            state.get("plans_sha256") == v4_plans_hash(self.plans),
            "planner_v4 plans differ from the inference state",
        )

    def columns(self, q: Mapping[str, Any], level: int, image: bytes) -> dict[str, Any]:
        if self.ctx is None:
            return {}
        items = self.plans[q["question_id"]]
        signals = {
            i["id"]: item_signals(
                cached(self.engine, observer_v4_spec(q, level, image, i)), None
            )
            for i in items
        }
        row = plan_columns(items, signals, V4_PREFIX, "v4item")
        return row | {"v4_n_items": len(items), "v4_failed": not items}


def cmd_table(ctx: Context) -> pd.DataFrame:
    state = ctx.state.get(STATE_KEY) or {}
    require(bool(state.get("complete")), "oracle inference is not complete")
    require(state.get("plans_sha256") == plans_hash(ctx), "plans.json changed")
    plans = load_plans(ctx)
    ref = reference_context(ctx)
    ref_engine, own_engine = cache_engine(ref), cache_engine(ctx)
    series = readonly_series(ctx)
    v4 = V4Arm(ctx)
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
            rows.append(
                row | oracle_columns(oplan, signals) | v4.columns(q, level, image)
            )
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


# ---- planner_v4 arm --------------------------------------------------------------------------
def v4_plans_hash(plans: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    return sha256_text(json.dumps(plans, sort_keys=True))


def v4_validator(ctx: Context) -> Any:
    settings = plan_v4_settings(ctx.cfg)
    return lambda obj: validate_plan_v4(obj, settings)


def v4_dry_run(ctx: Context, digest: str) -> dict[str, Any]:
    """Planner records present / observer calls planned from the cached plans."""
    questions = load_manifest(ctx, SPLIT)
    tag = record_tag(ctx.settings.name, digest)
    plan_dir = ctx.paths.output_root / "plans_v4" / tag
    obs_dir = ctx.paths.output_root / V4_OBSERVER_DIR / tag
    counts: Counter[str] = Counter()
    for q in questions:
        qid = q["question_id"]
        plan_path = plan_dir / f"{qid}.json"
        if not plan_path.exists():
            counts["planner_calls_remaining"] += 1
            continue
        items = plan_v4_items(read_json(plan_path))
        counts["planned_items"] += len(items)
        for level in range(1, ctx.n_levels + 1):
            for item in items:
                path = obs_dir / qid / f"{level:02d}_{item['id']}.json"
                counts["observer_calls_planned"] += 1
                counts["observer_records_present"] += int(path.exists())
    return {"questions": len(questions), **counts}


def cmd_v4_infer(ctx: Context, max_seconds: float | None) -> None:
    ref_digest = reference_digest(ctx)
    if ctx.dry_run:
        print(f"[dry-run] planner_v4 infer: {json.dumps(v4_dry_run(ctx, ref_digest))}")
        return
    require_ready(ctx, ref_digest)
    ctx.set_stage(V4_STAGE)
    engine = ctx.engine()
    series = readonly_series(ctx)
    start, calls0, hits0 = time.monotonic(), engine.calls, engine.cache_hits
    questions = load_manifest(ctx, SPLIT)
    validator = v4_validator(ctx)
    records = {
        q["question_id"]: engine.run(planner_v4_spec(q), validator) for q in questions
    }
    plans = {qid: plan_v4_items(r) for qid, r in records.items()}
    current_hash = v4_plans_hash(plans)
    previous = (ctx.state.get(V4_STATE_KEY) or {}).get("plans_sha256")
    require(
        previous in (None, current_hash), "planner_v4 plans changed after inference"
    )
    done, stopped = 0, False
    for q in questions:
        for level in range(1, ctx.n_levels + 1):
            if max_seconds is not None and time.monotonic() - start > max_seconds:
                stopped = True
                break
            image = series.bytes_for(str(q["image_id"]), level)
            for item in plans[q["question_id"]]:
                engine.run(
                    observer_v4_spec(q, level, image, item), validate_observation
                )
        if stopped:
            break
        done += 1
    elapsed = time.monotonic() - start
    new_calls = engine.calls - calls0
    progress = {
        "time": utc_now(),
        "plans_sha256": current_hash,
        "planner_failures": sorted(q for q, items in plans.items() if not items),
        "planner_repaired": sorted(
            q for q, r in records.items() if not r.get("initial_valid")
        ),
        "items_total": sum(len(items) for items in plans.values()),
        "questions_done": done,
        "questions_total": len(questions),
        "new_calls": new_calls,
        "cache_hits": engine.cache_hits - hits0,
        "elapsed_s": round(elapsed, 1),
        "s_per_call": round(elapsed / new_calls, 3) if new_calls else 0.0,
        "complete": done == len(questions),
    }
    ctx.state.set(V4_STATE_KEY, progress)
    log_path = ctx.paths.out("logs/v4_infer_progress.json")
    history = read_json(log_path) if log_path.exists() else []
    atomic_write_json(log_path, [*history, progress])
    ctx.log(f"planner_v4 infer chunk: {progress}")


def cmd_v4_audit(ctx: Context) -> None:
    """The v27 per-record audit (non-generation time, argmax) on the v4 records."""
    root = ctx.paths.output_root
    qids = {q["question_id"] for q in load_manifest(ctx, SPLIT)}
    plans = [p for p in sorted((root / "plans_v4").rglob("*.json")) if p.stem in qids]
    observations = [
        p
        for p in sorted((root / V4_OBSERVER_DIR).rglob("*.json"))
        if p.parent.name in qids
    ]
    counts: Counter[str] = Counter()
    flagged, worst = [], 0.0
    for path in [*plans, *observations]:
        record = read_json(path)
        result = audit_record(record)
        role = path.relative_to(root).parts[0]
        counts[f"{role}:records"] += 1
        counts[f"{role}:valid"] += int(record.get("status") == "valid")
        counts["argmax_positions"] += result["argmax_positions"]
        counts["argmax_violations"] += result["argmax_violations"]
        worst = max(worst, result["non_generation_s"])
        if result["slow"] or result["argmax_violations"]:
            flagged.append({"path": str(path.relative_to(root)), **result})
    summary = {
        "time": utc_now(),
        "records": len(plans) + len(observations),
        "max_non_generation_s": worst,
        "counts": dict(sorted(counts.items())),
        "flagged": flagged,
    }
    atomic_write_json(ctx.paths.out("logs/audit_v4.json"), summary)
    brief = {k: v for k, v in summary.items() if k != "flagged"}
    ctx.log(f"v4 audit: {json.dumps(brief)}, flagged {len(flagged)}")


def item_line(item: Mapping[str, Any]) -> str:
    tag = item.get("role", "")
    tag += f"/{item['logic']}" if item.get("logic") else ""
    req = f" req={item['required_answer']}" if item.get("required_answer") else ""
    deps = ",".join(item["depends_on"])
    return f"    {item['id']} [{deps}] {tag}{req}: {item['question']}"


def v4_rows(ctx: Context) -> list[dict[str, Any]]:
    """Per question: oracle plan, planner_v3 items, planner_v4 record + quality."""
    oracle = load_plans(ctx)
    ref_engine, own_engine = cache_engine(reference_context(ctx)), cache_engine(ctx)
    selection = {
        r["question_id"]: r
        for r in read_jsonl(repo_path(ctx, oracle_cfg(ctx)["selection"]))
    }
    gt = load_gt(ctx, SPLIT)
    rows = []
    for q in load_manifest(ctx, SPLIT):
        qid = q["question_id"]
        record = cached(own_engine, planner_v4_spec(q))
        items = plan_v4_items(record)
        quality = plan_quality(items, oracle[qid], selection[qid]["semantic"])
        rows.append(
            {
                "question": q,
                "gt": gt[qid],
                "oracle": oracle[qid],
                "v3": plan_from_record(
                    qid, cached(ref_engine, planner_v3_spec(q))
                ).items,
                "v4": items,
                "v4_record": record,
                "quality": quality,
            }
        )
    return rows


def cmd_v4_show(ctx: Context) -> None:
    for row in v4_rows(ctx):
        q, quality = row["question"], row["quality"]
        print(f"{q['attribute']} {q['question_id']} {q['question']!r} -> {row['gt']}")
        print("  oracle")
        print("\n".join(item_line(i) for i in row["oracle"]["items"]))
        print("  planner_v3")
        print("\n".join(item_line(i) for i in row["v3"]))
        defaults = (row["v4_record"].get("parsed") or {}).get("defaults_applied") or []
        extra = f" defaults={defaults}" if defaults else ""
        print(
            f"  planner_v4 (final type {quality['v4_final_type']} vs oracle "
            f"{quality['oracle_final_type']}; objects {quality['objects_covered']}/"
            f"{quality['objects']}; relation checks {quality['relation_checks_covered']}/"
            f"{quality['relation_checks']}; operands {quality['operands_covered']}/"
            f"{quality['operands']}{extra})"
        )
        print("\n".join(item_line(i) for i in row["v4"]) or "    (planner failure)")


def cmd_v4_quality(ctx: Context) -> None:
    frame = pd.DataFrame(
        [
            {
                "question_id": r["question"]["question_id"],
                "type": r["question"]["attribute"],
                "initial_valid": bool(r["v4_record"].get("initial_valid")),
                **r["quality"],
            }
            for r in v4_rows(ctx)
        ]
    )
    frame.to_csv(ctx.paths.out("plan_quality_v4.csv"), index=False)
    sums = [
        "v4_items",
        "oracle_items",
        "v4_existence",
        "oracle_existence",
        "objects",
        "objects_covered",
        "relation_checks",
        "relation_checks_covered",
        "operands",
        "operands_covered",
        "extra_existence",
        "final_type_preserved",
        "v4_failed",
    ]
    by_type = frame.groupby("type")[sums].sum()
    by_type.loc["all"] = frame[sums].sum()
    by_type["n"] = [*frame.groupby("type").size(), len(frame)]
    by_type["count_equal_oracle"] = [
        *frame.assign(eq=frame["v4_items"] == frame["oracle_items"])
        .groupby("type")["eq"]
        .sum(),
        int((frame["v4_items"] == frame["oracle_items"]).sum()),
    ]
    atomic_write_json(
        ctx.paths.out("plan_quality_v4.json"),
        {"time": utc_now(), "by_type": by_type.astype(int).to_dict(orient="index")},
    )
    print(by_type.astype(int).to_string())


# ---- main ----------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gqa_probe_oracle_run")
    parser.add_argument(
        "command",
        choices=(
            "plans",
            "show",
            "infer",
            "table",
            "v4-infer",
            "v4-show",
            "v4-quality",
            "v4-audit",
        ),
    )
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
        elif args.command == "v4-infer":
            cmd_v4_infer(ctx, args.max_seconds)
        elif args.command == "v4-show":
            cmd_v4_show(ctx)
        elif args.command == "v4-quality":
            cmd_v4_quality(ctx)
        elif args.command == "v4-audit":
            cmd_v4_audit(ctx)
        else:
            cmd_table(ctx)
    except StageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
