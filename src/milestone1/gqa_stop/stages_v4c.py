"""A13 (plan §20) clean-decoding re-run: determinism gate, reader-only levels,
and the v1-method arm (fair v1 under clean decoding).

The determinism gate talks to the transport directly (no cache, nothing stored
as a run record): (a) one text-only and one image request, each sent twice,
must return byte-identical content and logprobs; (b) on >= 100 logprob-enabled
calls the emitted token must equal the argmax of top_logprobs at every
position (rate 1.0). Any failure raises after the report is written.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .client import RequestEngine, RequestSpec
from .common import StageError, atomic_write_json, utc_now
from .context import Context, require, table_to_csv
from .evaluate import add_correctness
from .inference import READER_ROLE, InferenceRunner, build_level_spec
from .logprob_v4 import TOP_LOGPROBS, direct_lp_spec, relevance_spec
from .policy import freeze_policy
from .schemas import validate_reader
from .stages_model import LOGPROB, provenance, require_screening_match
from .textprobe import text_only_spec

ARGMAX_EPS = 1e-6
MIN_ARGMAX_CALLS = 100
V1_ARM_DIR = "v1arm"
V1_FROZEN = "frozen_policy_subset.json"


# ---- determinism gate ---------------------------------------------------------------------
def argmax_violations(logprobs: Sequence[Mapping[str, Any]] | None) -> tuple[int, int]:
    """(positions, positions where the emitted token is not a top_logprobs argmax)."""
    positions = bad = 0
    for entry in logprobs or []:
        alternatives = entry.get("top_logprobs") or []
        if not alternatives:
            continue
        positions += 1
        best = max(float(a["logprob"]) for a in alternatives)
        emitted = float(entry["logprob"])
        top_tokens = {
            a["token"] for a in alternatives if float(a["logprob"]) >= best - ARGMAX_EPS
        }
        if emitted < best - ARGMAX_EPS or entry.get("token") not in top_tokens:
            bad += 1
    return positions, bad


def _send(engine: RequestEngine, spec: RequestSpec) -> Any:
    return engine.transport.send(
        engine.build_payload(spec, engine.build_messages(spec))
    )


def detgate(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("detgate")
    require(
        bool(ctx.settings.decoding_options), "model.decoding_options must be set (A13)"
    )
    det = ctx.cfg["determinism"]
    # candidate questions in manifest order (dev first, then test); no GT is used
    questions = ctx.candidates()[: int(det["argmax_questions"])]
    levels = [int(v) for v in det["argmax_levels"]]
    n_rel = int(det["relevance_questions"])
    planned = 4 + len(questions) * len(levels) + n_rel * len(levels)
    if ctx.dry_run:
        ctx.echo(f"[dry-run] determinism gate: {planned} uncached requests")
        return {"planned_requests": planned}
    engine = ctx.engine()
    first = questions[0]
    image = ctx.series.bytes_for(str(first["image_id"]), ctx.n_levels)
    pair_specs = {
        "text_only": dataclasses.replace(text_only_spec(first), logprobs=TOP_LOGPROBS),
        "image_reader_L20": dataclasses.replace(
            build_level_spec(READER_ROLE, first, ctx.n_levels, image),
            logprobs=TOP_LOGPROBS,
        ),
    }
    pairs = {}
    for name, spec in pair_specs.items():
        a, b = _send(engine, spec), _send(engine, spec)
        pairs[name] = {
            "content_identical": a.content == b.content,
            "logprobs_identical": a.logprobs == b.logprobs,
            "content": a.content[:200],
        }
    positions = bad = calls = 0
    offenders = []
    for i, q in enumerate(questions):
        for level in levels:
            image = ctx.series.bytes_for(str(q["image_id"]), level)
            specs = [direct_lp_spec(q, level, image)]
            if i < n_rel:
                specs.append(
                    relevance_spec(
                        str(q["question_id"]),
                        level,
                        image,
                        str(q["question"]),
                        "direct",
                    )
                )
            for spec in specs:
                result = _send(engine, spec)
                calls += 1
                p, v = argmax_violations(result.logprobs)
                positions += p
                bad += v
                if v:
                    offenders.append(
                        {
                            "question_id": q["question_id"],
                            "level": level,
                            "role": spec.role,
                            "violations": v,
                        }
                    )
    rate = 1.0 - bad / positions if positions else 0.0
    passed = (
        all(p["content_identical"] and p["logprobs_identical"] for p in pairs.values())
        and calls >= MIN_ARGMAX_CALLS
        and rate == 1.0
    )
    report = {
        "time": utc_now(),
        "decoding": engine.decoding(pair_specs["text_only"]),
        "repeat_pairs": pairs,
        "argmax_calls": calls,
        "argmax_positions": positions,
        "argmax_violations": bad,
        "argmax_match_rate": rate,
        "offenders": offenders[:50],
        "passed": passed,
    }
    atomic_write_json(ctx.paths.out("preflight_determinism.json"), report)
    ctx.state.set("determinism", {"passed": passed, "time": utc_now()})
    ctx.log(
        f"determinism gate: passed={passed} calls={calls} positions={positions} "
        f"violations={bad} pairs={ {k: v['content_identical'] for k, v in pairs.items()} }"
    )
    if not passed:
        raise StageError(
            "determinism gate failed (A13): see preflight_determinism.json"
        )
    return report


def require_determinism(ctx: Context) -> None:
    require(
        bool((ctx.state.get("determinism") or {}).get("passed")),
        "run `detgate` first: the A13 determinism gate has not passed",
    )


# ---- reader on every level (dev and test; no method on test) --------------------------------
def reader_levels(ctx: Context, split: str, levels: Sequence[int]) -> dict[str, Any]:
    ctx.set_stage(f"readers_{split}")
    require_determinism(ctx)
    require_screening_match(ctx, split)
    questions = ctx.selected(split, use_subset=False)
    if ctx.dry_run:
        ctx.echo(f"[dry-run] reader {split}: {len(questions)} x {len(levels)} levels")
        return {"planned_requests": len(questions) * len(levels)}
    engine = ctx.engine()
    rows = []
    for q in questions:
        for level in levels:
            image = ctx.series.bytes_for(str(q["image_id"]), level)
            record = engine.run(
                build_level_spec(READER_ROLE, q, level, image), validate_reader
            )
            ok = record["status"] == "valid"
            rows.append(
                {
                    "question_id": q["question_id"],
                    "image_id": q["image_id"],
                    "split": split,
                    "level": level,
                    "cell": q["cell"],
                    "reader_answer": record["parsed"]["answer"] if ok else None,
                    "reader_status": record["status"],
                }
            )
    table = add_correctness(
        pd.DataFrame(rows), ctx.gt([q["question_id"] for q in questions])
    )
    table.to_csv(ctx.paths.out(f"reader_levels_{split}.csv"), index=False)
    curve = {
        str(int(k)): float(v)
        for k, v in table.groupby("level")["reader_correct"].mean().items()
    }
    ctx.state.set(
        "readers",
        {
            **(ctx.state.get("readers") or {}),
            split: {"done": True, "time": utc_now(), "curve": curve},
        },
    )
    ctx.log(
        f"reader levels {split}: {len(questions)} questions; accuracy by level {curve}"
    )
    ctx.note_reuse()
    return {"questions": len(questions), "curve": curve}


# ---- v1-method arm on the dev visual subset ---------------------------------------------------
def v1arm(ctx: Context) -> dict[str, Any]:
    """v1 methods (planner v1, checklist v2 prompt, direct, reader) under clean
    decoding on the dev visual subset; thresholds frozen on that subset."""
    ctx.set_stage("v1arm")
    require(ctx.family == LOGPROB, "v1arm runs inside the A13 logprob_item run")
    require_determinism(ctx)
    require_screening_match(ctx, "dev")
    questions = ctx.selected("dev")
    levels = list(range(1, ctx.n_levels + 1))
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] v1 arm: {len(questions)} subset questions x {len(levels)} levels"
        )
        return {}
    checklist = ctx.cfg["checklist"]
    runner = InferenceRunner(
        ctx.engine(),
        ctx.series,
        int(checklist["min_items"]),
        int(checklist["max_items"]),
    )
    rows = [row for q in questions for row in runner.run_question(q, levels)]
    table = add_correctness(
        pd.DataFrame(rows), ctx.gt([q["question_id"] for q in questions])
    )
    out = ctx.paths.out(V1_ARM_DIR)
    table_to_csv(table, out / "level_table_dev.csv")
    policy = {**ctx.cfg["policy"], "methods": list(ctx.cfg["v1_arm"]["policy_methods"])}
    if (out / V1_FROZEN).exists():
        ctx.log("v1 arm: subset policy already frozen; table refreshed only")
    else:
        freeze_policy(
            table,
            policy,
            ctx.n_levels,
            out,
            {
                **provenance(ctx),
                "analysis": "v1 methods under A13 clean decoding, "
                "thresholds frozen on the dev visual subset",
            },
            frozen_name=V1_FROZEN,
        )
    ctx.state.set(
        "v1arm", {"done": True, "time": utc_now(), "questions": len(questions)}
    )
    ctx.log(f"v1 arm: {len(questions)} subset questions x {len(levels)} levels")
    ctx.note_reuse()
    return {"questions": len(questions)}
