"""Plan §27 (v27) model stages: environment check, determinism gate, reader_lp
identity check, L20 screening, full inference and the level table.

Roles (prompts unchanged; no new prompt text):
- reader_lp: answer_only + top_logprobs (namespace ``lp``, dir ``reader_lp``).
  Common scorer and the B1 signal (P_ans of the answer's first token).
- reader (plain answer_only, no logprobs): identity check only.
- planner_v3: one text-only call per question.
- observer_v3_lp: observer_v3 per planner item per level + top_logprobs.
- direct_resolved: observer_v3 prompt applied to the ORIGINAL question + top_logprobs
  (namespace ``lp_direct``, dir ``observer_v3_lp_direct``).
Screening = reader_lp at L20: label answerable (correct) / unanswerable (wrong or
invalid). Nothing is excluded.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from . import v6_common
from .audit_v22 import audit_record
from .client import STATUS_VALID, RequestSpec
from .common import atomic_write_json, read_json, read_jsonl, utc_now, write_jsonl
from .context import Context, require
from .dataset import normalize_text
from .evaluate import is_correct
from .inference import READER_ROLE, CacheMissError, build_level_spec
from .logprob_v4 import (
    LP_NAMESPACE,
    TOP_LOGPROBS,
    gated_scores,
    item_signals,
    locate_answer_token,
    observer_lp_spec,
)
from .planner import PlanOutcome, plan_from_record
from .sampled import (
    MAX_ITEMS_COLUMNS,
    OBSERVER_ROLE,
    planner_v3_spec,
    validate_observation,
    validate_plan_v3,
)
from .schemas import validate_reader
from .stages_v4c import _send, argmax_violations
from .v27_data import load_gt, load_manifest

DIRECT_RESOLVED_NAMESPACE = "lp_direct"
UNKNOWN = "unknown"
ANSWERABLE = "answerable"
UNANSWERABLE = "unanswerable"
SCREEN_FILE = "screening_{split}.jsonl"
TABLE_FILE = "level_table_{split}.csv"
ROLE_DIRS = (
    "reader",
    "reader_lp",
    "plans_v3",
    "observer_v3_lp",
    "observer_v3_lp_direct",
)
NVIDIA_QUERY = "name,driver_version,memory.used,memory.total,utilization.gpu"
JOURNAL_LINES = 400
CLIP_MARKER = "CLIP using"
SUBPROCESS_TIMEOUT_S = 30


# ---- request specs (no GT ever enters a request) -------------------------------------------
def reader_lp_spec(
    question: Mapping[str, Any], level: int, image: bytes
) -> RequestSpec:
    spec = build_level_spec(READER_ROLE, question, level, image)
    return dataclasses.replace(spec, namespace=LP_NAMESPACE, logprobs=TOP_LOGPROBS)


def reader_plain_spec(
    question: Mapping[str, Any], level: int, image: bytes
) -> RequestSpec:
    return build_level_spec(READER_ROLE, question, level, image)


def direct_resolved_spec(
    question: Mapping[str, Any], level: int, image: bytes
) -> RequestSpec:
    """observer_v3 prompt on the original question text (A1)."""
    return RequestSpec(
        role=OBSERVER_ROLE,
        question_id=str(question["question_id"]),
        level=level,
        input_data={"question": str(question["question"])},
        image_bytes=image,
        namespace=DIRECT_RESOLVED_NAMESPACE,
        logprobs=TOP_LOGPROBS,
    )


# ---- pure signal helpers ------------------------------------------------------------------
def reader_signal(record: Mapping[str, Any] | None) -> dict[str, Any]:
    """Answer text and P_ans (first token of the JSON answer value) of reader_lp."""
    valid = record is not None and record.get("status") == STATUS_VALID
    if not valid or record is None:
        return {"valid": False, "answer": None, "pans": 0.0, "located": False}
    token = locate_answer_token(record.get("logprobs"))
    return {
        "valid": True,
        "answer": record["parsed"]["answer"],
        "pans": token.prob if token else 0.0,
        "located": token is not None,
    }


def resolved_ok(signal: Mapping[str, Any]) -> bool:
    """valid, resolved == true and answer != "unknown"."""
    return bool(
        signal["valid"]
        and signal["resolved"] is True
        and normalize_text(str(signal["answer"])) != UNKNOWN
    )


def direct_resolved_score(signal: Mapping[str, Any]) -> float:
    """A1: P_ans if resolved and answer != unknown else 0."""
    return float(signal["pans"]) if resolved_ok(signal) else 0.0


def raw_pans_min(plan: PlanOutcome, items: Mapping[str, Mapping[str, Any]]) -> float:
    """A2: min over items of the raw observer P_ans (no gate); planner failure -> 0."""
    if plan.failed or not plan.items:
        return 0.0
    return min(float(items[i["id"]]["pans"]) for i in plan.items)


def all_resolved(plan: PlanOutcome, items: Mapping[str, Mapping[str, Any]]) -> bool:
    """A3: every item resolved (answer != unknown); planner failure -> False."""
    if plan.failed or not plan.items:
        return False
    return all(resolved_ok(items[i["id"]]) for i in plan.items)


def label_of(correct: bool | None) -> str:
    return ANSWERABLE if correct else UNANSWERABLE


# ---- runner -------------------------------------------------------------------------------
class V27Runner:
    """Runs (or replays from cache) every v27 call of one question."""

    def __init__(self, ctx: Context, cache_only: bool = False) -> None:
        self.ctx = ctx
        self.engine = ctx.engine()
        self.cache_only = cache_only
        checklist = ctx.cfg["checklist"]
        self.min_items = int(checklist["min_items"])
        self.max_items = int(checklist["max_items"])

    def run(self, spec: RequestSpec, validator: Any) -> dict[str, Any]:
        if not self.cache_only:
            return self.engine.run(spec, validator)
        cached = self.engine.lookup(spec)
        if cached is None:
            raise CacheMissError(
                f"{spec.role}/{spec.namespace} {spec.question_id} L{spec.level}{spec.suffix}"
            )
        return cached

    def image(self, question: Mapping[str, Any], level: int) -> bytes:
        return self.ctx.series.bytes_for(str(question["image_id"]), level)

    def reader_lp(self, question: Mapping[str, Any], level: int) -> dict[str, Any]:
        spec = reader_lp_spec(question, level, self.image(question, level))
        return self.run(spec, validate_reader)

    def reader_plain(self, question: Mapping[str, Any], level: int) -> dict[str, Any]:
        spec = reader_plain_spec(question, level, self.image(question, level))
        return self.run(spec, validate_reader)

    def plan(self, question: Mapping[str, Any]) -> tuple[PlanOutcome, dict[str, Any]]:
        record = self.run(
            planner_v3_spec(question),
            lambda o: validate_plan_v3(o, self.min_items, self.max_items),
        )
        return plan_from_record(str(question["question_id"]), record), record

    def level_calls(
        self, question: Mapping[str, Any], level: int, plan: PlanOutcome
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
        image = self.image(question, level)
        reader = self.run(reader_lp_spec(question, level, image), validate_reader)
        direct = self.run(
            direct_resolved_spec(question, level, image), validate_observation
        )
        items = {
            item["id"]: self.run(
                observer_lp_spec(question, level, image, item), validate_observation
            )
            for item in plan.items
        }
        return reader, direct, items


def level_row(
    question: Mapping[str, Any],
    level: int,
    plan: PlanOutcome,
    plan_record: Mapping[str, Any],
    reader: Mapping[str, Any],
    direct: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """One (question, level) row; GT and labels are added by the caller."""
    rsig = reader_signal(reader)
    dsig = item_signals(direct, None)
    items = {i: item_signals(obs, None) for i, obs in observations.items()}
    gated = gated_scores(plan, items)
    parsed = plan_record.get("parsed") or {}
    row: dict[str, Any] = {
        "question_id": str(question["question_id"]),
        "image_id": str(question["image_id"]),
        "group_id": str(question.get("group_id", question["image_id"])),
        "split": question.get("split", ""),
        "attribute": question.get("attribute", ""),
        "relate_count": int(question.get("relate_count", -1)),
        "level": level,
        "reader_answer": rsig["answer"],
        "reader_valid": rsig["valid"],
        "reader_located": rsig["located"],
        "reader_pans": rsig["pans"],
        "dres_answer": dsig["answer"],
        "dres_valid": dsig["valid"],
        "dres_resolved": dsig["resolved"] if dsig["valid"] else False,
        "dres_located": dsig["located"],
        "dres_pans": dsig["pans"],
        "dres_score": direct_resolved_score(dsig),
        "planner_status": plan.status,
        "planner_initial_valid": plan.initial_valid,
        "planner_attempts": plan.attempts,
        "last_item_has_attribute": bool(parsed.get("last_item_has_attribute", False)),
        "n_items": len(plan.items),
        "pans_min": gated["pans_min"],
        "pans_mean": gated["pans_mean"],
        "rawpans_min": raw_pans_min(plan, items),
        "all_resolved": float(all_resolved(plan, items)),
        "observer_valid_all": all(s["valid"] for s in items.values()),
    }
    for index, item in enumerate(plan.items[:MAX_ITEMS_COLUMNS], start=1):
        sig = items[item["id"]]
        row[f"item{index}_question"] = item["question"]
        row[f"item{index}_answer"] = sig["answer"]
        row[f"item{index}_valid"] = sig["valid"]
        row[f"item{index}_resolved"] = sig["resolved"] if sig["valid"] else False
        row[f"item{index}_located"] = sig["located"]
        row[f"item{index}_pans"] = sig["pans"]
    return row


# ---- environment check ---------------------------------------------------------------------
def _command(args: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            check=False,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    return result.stdout.strip()


def clip_lines(since_epoch: float) -> list[str]:
    """ollama journal lines naming the vision-encoder backend since ``since_epoch``."""
    text = _command(
        [
            "journalctl",
            "-u",
            "ollama",
            "--since",
            f"@{int(since_epoch)}",
            "-n",
            str(JOURNAL_LINES),
            "--no-pager",
        ]
    )
    return [line for line in text.splitlines() if CLIP_MARKER in line]


def envcheck(ctx: Context) -> dict[str, Any]:
    """nvidia-smi, ollama ps, isolate + warm the model alone (text + one image call),
    then require 'CLIP using CUDA0' in the ollama journal and 100% GPU residency."""
    ctx.set_stage("v27_envcheck")
    endpoint = ctx.cfg["model"]["endpoint"]
    name = ctx.settings.name
    if ctx.dry_run:
        ctx.echo(f"[dry-run] envcheck: isolate + warm {name}")
        return {}
    gpu = _command(
        ["nvidia-smi", f"--query-gpu={NVIDIA_QUERY}", "--format=csv,noheader"]
    )
    before = v6_common.loaded_models(endpoint)
    started = time.time()
    unloaded = v6_common.isolate_model(endpoint, name)
    v6_common.warm_up(endpoint, name)
    # one image request so that the vision encoder is loaded (no record written)
    manifest = load_manifest(ctx, "dev")
    engine = ctx.engine()
    spec = reader_lp_spec(
        manifest[0], 1, ctx.series.bytes_for(manifest[0]["image_id"], 1)
    )
    _send(engine, spec)
    ps_text = _command(["ollama", "ps"])
    after = v6_common.loaded_models(endpoint)
    clips = clip_lines(started - 1)
    report = {
        "time": utc_now(),
        "nvidia_smi": gpu,
        "kernel_driver": _command(["head", "-1", "/proc/driver/nvidia/version"]),
        "resident_before": before,
        "unloaded": unloaded,
        "resident_after": after,
        "ollama_ps": ps_text,
        "clip_lines": clips,
        "clip_cuda0": bool(clips) and all("CUDA0" in line for line in clips),
        "only_target_resident": after == [name],
        "gpu_100": "100% GPU" in ps_text,
    }
    report["passed"] = bool(
        report["clip_cuda0"] and report["only_target_resident"] and report["gpu_100"]
    )
    stamp = report["time"].replace(":", "")
    atomic_write_json(ctx.paths.out(f"logs/envcheck_{stamp}.json"), report)
    ctx.state.set("v27_envcheck", {"passed": report["passed"], "time": report["time"]})
    ctx.log(
        f"ENV: {gpu}; resident before {before}; after {after}; "
        f"CLIP lines {clips[-1:] if clips else 'none'}; ollama ps 100% GPU="
        f"{report['gpu_100']}; passed={report['passed']}"
    )
    require(report["passed"], "environment check failed: see logs/envcheck_*.json")
    return report


def require_env(ctx: Context) -> None:
    require(
        bool((ctx.state.get("v27_envcheck") or {}).get("passed")),
        "run `v27-envcheck` first",
    )


# ---- determinism gate (no records) ----------------------------------------------------------
def detgate(ctx: Context) -> dict[str, Any]:
    """Repeat pairs (planner_v3 text, reader_lp L20 image) must be byte-identical;
    reader_lp on 50 questions x L1/L20 and direct_resolved on 10 x L1/L20 must emit
    the argmax token at every position (120 calls)."""
    ctx.set_stage("v27_detgate")
    require_env(ctx)
    det = ctx.cfg["determinism"]
    questions = load_manifest(ctx, "dev")[: int(det["argmax_questions"])]
    levels = [int(v) for v in det["argmax_levels"]]
    n_direct = int(det["direct_resolved_questions"])
    planned = 4 + len(questions) * len(levels) + n_direct * len(levels)
    if ctx.dry_run:
        ctx.echo(f"[dry-run] v27 determinism gate: {planned} uncached requests")
        return {"planned_requests": planned}
    engine = ctx.engine()
    first = questions[0]
    top = ctx.n_levels
    pair_specs = {
        "planner_v3_text": dataclasses.replace(
            planner_v3_spec(first), logprobs=TOP_LOGPROBS
        ),
        "reader_lp_L20": reader_lp_spec(
            first, top, ctx.series.bytes_for(first["image_id"], top)
        ),
    }
    pairs = {}
    for label, spec in pair_specs.items():
        a, b = _send(engine, spec), _send(engine, spec)
        pairs[label] = {
            "content_identical": a.content == b.content,
            "logprobs_identical": a.logprobs == b.logprobs,
            "content": a.content[:200],
        }
    positions = bad = calls = 0
    offenders = []
    for index, q in enumerate(questions):
        for level in levels:
            image = ctx.series.bytes_for(q["image_id"], level)
            specs = [reader_lp_spec(q, level, image)]
            if index < n_direct:
                specs.append(direct_resolved_spec(q, level, image))
            for spec in specs:
                result = _send(engine, spec)
                calls += 1
                p, v = argmax_violations(result.logprobs)
                positions, bad = positions + p, bad + v
                if v:
                    offenders.append(
                        {
                            "question_id": q["question_id"],
                            "level": level,
                            "role": spec.namespace,
                        }
                    )
    passed = (
        all(p["content_identical"] and p["logprobs_identical"] for p in pairs.values())
        and calls >= len(questions) * len(levels)
        and bad == 0
        and positions > 0
    )
    report = {
        "time": utc_now(),
        "repeat_pairs": pairs,
        "argmax_calls": calls,
        "argmax_positions": positions,
        "argmax_violations": bad,
        "offenders": offenders[:50],
        "passed": passed,
    }
    atomic_write_json(ctx.paths.out("preflight_determinism.json"), report)
    ctx.state.set("determinism", {"passed": passed, "time": utc_now()})
    ctx.log(
        f"determinism gate: passed={passed} calls={calls} positions={positions} "
        f"violations={bad} pairs={ {k: v['content_identical'] for k, v in pairs.items()} }"
    )
    require(passed, "determinism gate failed: see preflight_determinism.json")
    return report


def require_determinism(ctx: Context) -> None:
    require(
        bool((ctx.state.get("determinism") or {}).get("passed")),
        "run `v27-detgate` first",
    )


# ---- reader_lp identity check ----------------------------------------------------------------
def identity_rows(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Per (plain, lp) record pair: raw text and parsed answer identical?"""
    rows = []
    for plain, lp in pairs:
        rows.append(
            {
                "question_id": plain["question_id"],
                "level": plain["level"],
                "plain_status": plain["status"],
                "lp_status": lp["status"],
                "raw_identical": plain.get("raw") == lp.get("raw"),
                "answer_identical": (plain.get("parsed") or {}).get("answer")
                == (lp.get("parsed") or {}).get("answer"),
                "plain_raw": plain.get("raw"),
                "lp_raw": lp.get("raw"),
            }
        )
    return rows


def identity(ctx: Context) -> dict[str, Any]:
    """Plain reader vs reader_lp on the first N dev questions x configured levels."""
    ctx.set_stage("v27_identity")
    require_determinism(ctx)
    icfg = ctx.cfg["v27"]["identity_check"]
    questions = load_manifest(ctx, "dev")[: int(icfg["questions"])]
    levels = [int(v) for v in icfg["levels"]]
    if ctx.dry_run:
        ctx.echo(f"[dry-run] identity: {len(questions) * len(levels)} pairs")
        return {}
    runner = V27Runner(ctx)
    pairs = [
        (runner.reader_plain(q, level), runner.reader_lp(q, level))
        for q in questions
        for level in levels
    ]
    rows = identity_rows(pairs)
    identical = all(r["raw_identical"] for r in rows)
    report = {
        "time": utc_now(),
        "pairs": len(rows),
        "raw_identical": sum(r["raw_identical"] for r in rows),
        "answer_identical": sum(r["answer_identical"] for r in rows),
        "all_identical": identical,
        "decision": (
            "reader_lp is the common scorer and B1 (one call)"
            if identical
            else "NOT identical: keep a separate plain reader as scorer (report)"
        ),
        "mismatches": [r for r in rows if not r["raw_identical"]][:20],
    }
    atomic_write_json(ctx.paths.out("identity_check.json"), report)
    ctx.state.set("v27_identity", {"identical": identical, "time": report["time"]})
    ctx.log(
        f"reader_lp identity: {report['raw_identical']}/{len(rows)} raw byte-identical, "
        f"{report['answer_identical']}/{len(rows)} answers identical -> {report['decision']}"
    )
    ctx.note_reuse()
    return report


# ---- screening ---------------------------------------------------------------------------
def screen(ctx: Context, split: str) -> dict[str, Any]:
    """reader_lp at L20 on every question; internal label answerable / unanswerable."""
    ctx.set_stage(f"v27_screen_{split}")
    require_determinism(ctx)
    require(
        bool((ctx.state.get("v27_identity") or {}).get("identical")),
        "reader_lp identity check has not passed (`v27-identity`)",
    )
    questions = load_manifest(ctx, split)
    if ctx.dry_run:
        ctx.echo(f"[dry-run] screen {split}: {len(questions)} reader_lp L20 calls")
        return {"questions": len(questions)}
    gt = load_gt(ctx, split)
    runner = V27Runner(ctx)
    top = ctx.n_levels
    rows = []
    for q in questions:
        sig = reader_signal(runner.reader_lp(q, top))
        correct = (
            is_correct(sig["answer"], gt[q["question_id"]]) if sig["valid"] else False
        )
        rows.append(
            {
                "question_id": q["question_id"],
                "image_id": q["image_id"],
                "group_id": q["group_id"],
                "attribute": q["attribute"],
                "relate_count": q["relate_count"],
                "reader_answer": sig["answer"],
                "reader_valid": sig["valid"],
                "reader_pans_l20": sig["pans"],
                "correct_l20": bool(correct),
                "label": label_of(correct),
            }
        )
    write_jsonl(ctx.paths.selection(SCREEN_FILE.format(split=split)), rows)
    counts = Counter(r["label"] for r in rows)
    by_attr = Counter((r["attribute"], r["label"]) for r in rows)
    info = {
        "time": utc_now(),
        "questions": len(rows),
        "invalid": sum(not r["reader_valid"] for r in rows),
        "labels": dict(counts),
        "by_attribute": {f"{a}|{lab}": n for (a, lab), n in sorted(by_attr.items())},
    }
    ctx.state.set("v27_screen", {**(ctx.state.get("v27_screen") or {}), split: info})
    ctx.log(f"v27 screen {split}: {info}")
    return info


def load_screening(ctx: Context, split: str) -> dict[str, dict[str, Any]]:
    path = ctx.paths.selection(SCREEN_FILE.format(split=split))
    require(path.exists(), f"run `v27-screen --split {split}` first")
    return {r["question_id"]: r for r in read_jsonl(path)}


# ---- inference ---------------------------------------------------------------------------
def planned_calls(n_items: int, n_levels: int) -> int:
    """Calls of one question: reader_lp L1..L(n-1), planner, (direct + items) x n."""
    return (n_levels - 1) + 1 + n_levels * (1 + n_items)


def infer(ctx: Context, split: str, max_seconds: float | None) -> dict[str, Any]:
    """All calls for every question of ``split``; stops cleanly after ``max_seconds``
    (checked between levels) and resumes from the cache on the next run."""
    ctx.set_stage(f"v27_infer_{split}")
    require_determinism(ctx)
    screening = load_screening(ctx, split)
    questions = load_manifest(ctx, split)
    require(
        set(screening) == {q["question_id"] for q in questions}, "screening incomplete"
    )
    levels = list(range(1, ctx.n_levels + 1))
    if ctx.dry_run:
        estimate = len(questions) * planned_calls(2, len(levels))
        ctx.echo(f"[dry-run] infer {split}: about {estimate} calls")
        return {"planned": estimate}
    runner = V27Runner(ctx)
    engine = runner.engine
    start, calls0, hits0 = time.monotonic(), engine.calls, engine.cache_hits
    finished = load_done(ctx, split)
    done = 0
    stopped = False
    for q in questions:
        if q["question_id"] in finished:  # every call already cached
            done += 1
            continue
        plan, _ = runner.plan(q)
        for level in levels:
            if max_seconds is not None and time.monotonic() - start > max_seconds:
                stopped = True
                break
            runner.level_calls(q, level, plan)
        if stopped:
            break
        done += 1
        finished.add(q["question_id"])
    save_done(ctx, split, finished)
    elapsed = time.monotonic() - start
    new_calls = engine.calls - calls0
    per_call = elapsed / new_calls if new_calls else 0.0
    complete = done == len(questions)
    progress = {
        "time": utc_now(),
        "questions_done": done,
        "questions_total": len(questions),
        "new_calls": new_calls,
        "cache_hits": engine.cache_hits - hits0,
        "elapsed_s": round(elapsed, 1),
        "s_per_call": round(per_call, 3),
        "complete": complete,
    }
    remaining = (len(questions) - done) * planned_calls(2, len(levels))
    progress["eta_s_estimate"] = round(remaining * per_call, 0) if per_call else None
    state = ctx.state.get("v27_infer") or {}
    ctx.state.set("v27_infer", {**state, split: progress})
    append_progress(ctx, split, progress)
    ctx.log(f"v27 infer {split} chunk: {progress}")
    return progress


def done_path(ctx: Context, split: str) -> Path:
    return ctx.paths.out(f"logs/infer_done_{split}.json")


def load_done(ctx: Context, split: str) -> set[str]:
    """Questions whose calls are all cached (skipped on resume; a speed-up only:
    ``build_table`` replays every record from the cache regardless)."""
    path = done_path(ctx, split)
    return set(read_json(path)) if path.exists() else set()


def save_done(ctx: Context, split: str, finished: set[str]) -> None:
    atomic_write_json(done_path(ctx, split), sorted(finished))


def append_progress(ctx: Context, split: str, progress: Mapping[str, Any]) -> None:
    path = ctx.paths.out(f"logs/infer_progress_{split}.json")
    history = read_json(path) if path.exists() else []
    atomic_write_json(path, [*history, dict(progress)])


def require_inferred(ctx: Context, split: str) -> None:
    state = (ctx.state.get("v27_infer") or {}).get(split) or {}
    require(bool(state.get("complete")), f"inference on {split} is not complete")


# ---- level table (cache only) ------------------------------------------------------------
def build_table(ctx: Context, split: str) -> pd.DataFrame:
    """Every (question, level) row from cached records + GT, label and correctness."""
    ctx.set_stage(f"v27_table_{split}")
    require_inferred(ctx, split)
    screening = load_screening(ctx, split)
    questions = load_manifest(ctx, split)
    gt = load_gt(ctx, split)
    runner = V27Runner(ctx, cache_only=True)
    rows = []
    for q in questions:
        plan, plan_record = runner.plan(q)
        for level in range(1, ctx.n_levels + 1):
            reader, direct, items = runner.level_calls(q, level, plan)
            rows.append(level_row(q, level, plan, plan_record, reader, direct, items))
    table = pd.DataFrame(rows)
    table["reader_correct"] = [
        is_correct(None if pd.isna(a) else str(a), gt[qid])
        for qid, a in zip(table["question_id"], table["reader_answer"], strict=True)
    ]
    table["label"] = table["question_id"].map(lambda q: screening[q]["label"])
    table["gt_answer"] = table["question_id"].map(gt)
    table.to_csv(ctx.paths.out(TABLE_FILE.format(split=split)), index=False)
    ctx.log(f"v27 level table {split}: {len(table)} rows")
    return table


# ---- audit and quarantine ----------------------------------------------------------------
def record_files(output_root: Path, qids: set[str]) -> list[Path]:
    files: list[Path] = []
    for role in ROLE_DIRS:
        base = output_root / role
        if not base.exists():
            continue
        for path in base.glob("*/*/*"):
            if path.is_dir() and path.name in qids:
                files += sorted(path.glob("*.json"))
            elif path.suffix == ".json" and path.stem in qids:
                files.append(path)
    return files


def audit(ctx: Context, split: str) -> dict[str, Any]:
    """Non-generation time > 5 s or argmax violations in any record of ``split``."""
    ctx.set_stage(f"v27_audit_{split}")
    qids = {q["question_id"] for q in load_manifest(ctx, split)}
    root = ctx.paths.output_root
    files = record_files(root, qids)
    if ctx.dry_run:
        ctx.echo(f"[dry-run] audit {len(files)} {split} records")
        return {"records": len(files)}
    counts: Counter[str] = Counter()
    flagged = []
    worst = 0.0
    for path in files:
        record = read_json(path)
        result = audit_record(record)
        role = path.relative_to(root).parts[0]
        counts[f"{role}:records"] += 1
        counts[f"{role}:valid"] += int(record.get("status") == STATUS_VALID)
        counts["argmax_positions"] += result["argmax_positions"]
        counts["argmax_violations"] += result["argmax_violations"]
        worst = max(worst, result["non_generation_s"])
        if result["slow"] or result["argmax_violations"]:
            flagged.append({"path": str(path.relative_to(root)), **result})
    summary = {
        "time": utc_now(),
        "split": split,
        "records": len(files),
        "max_non_generation_s": worst,
        "counts": dict(sorted(counts.items())),
        "flagged": flagged,
    }
    atomic_write_json(ctx.paths.out(f"logs/audit_{split}.json"), summary)
    ctx.log(
        f"v27 audit {split}: {len(files)} records, {len(flagged)} flagged, max "
        f"non-generation {worst:.2f} s, argmax violations "
        f"{counts['argmax_violations']}/{counts['argmax_positions']}"
    )
    return summary


def quarantine(ctx: Context, split: str) -> dict[str, Any]:
    """Move (never delete) the records flagged by the last audit of ``split``."""
    ctx.set_stage(f"v27_quarantine_{split}")
    path = ctx.paths.out(f"logs/audit_{split}.json")
    require(path.exists(), f"run `v27-audit --split {split}` first")
    flagged = read_json(path)["flagged"]
    stamp = utc_now().replace(":", "")
    target_root = ctx.paths.out(f"quarantine/{split}_{stamp}")
    moved = []
    for entry in flagged:
        source = ctx.paths.output_root / entry["path"]
        if not source.exists():
            continue
        if ctx.dry_run:
            moved.append(entry["path"])
            continue
        target = target_root / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        moved.append(entry["path"])
    if not ctx.dry_run:
        ctx.log(f"v27 quarantine {split}: moved {len(moved)} records to {target_root}")
        state = ctx.state.get("v27_infer") or {}
        if moved:
            done_path(ctx, split).unlink(missing_ok=True)  # re-walk via the cache
        if moved and split in state:
            ctx.state.set(
                "v27_infer", {**state, split: {**state[split], "complete": False}}
            )
    return {"moved": moved}
