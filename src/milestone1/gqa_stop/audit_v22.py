"""Record audit for the v22 test inference (team-lead protocol 2026-10-11).

Every cached record of the test sets (visual subset 251 and reader-wrong 233)
is checked for (a) non-generation time of any attempt, total_duration minus
eval_duration > 5 s (model load or CPU vision encoder), and (b) argmax
violations: an emitted token that is not the top_logprobs maximum (A13 greedy
decoding). Flagged records are listed; moving them to the quarantine
directory is a separate, explicit step (records are never deleted).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .common import atomic_write_json, read_json, utc_now
from .context import Context
from .firing_v23 import wrong_questions
from .stages_v4c import argmax_violations

NON_GENERATION_LIMIT_S = 5.0
NS_PER_S = 1e9
AUDIT_FILE = "logs/audit_test.json"
FLAGGED_FILE = "logs/audit_test_flagged.txt"
ROLE_DIRS = (
    "reader",
    "direct_lp",
    "relevance_v4",
    "observer_v3_lp",
    "plans_v3",
)
ROOTS = ("", "ext_test_wrong")


def non_generation_seconds(record: Mapping[str, Any]) -> float:
    """Max over attempts of (total_duration - eval_duration), in seconds."""
    worst = 0.0
    for attempt in record.get("attempts") or []:
        usage = attempt.get("usage") or {}
        total = float(usage.get("total_duration", 0))
        gap = (total - float(usage.get("eval_duration", 0))) / NS_PER_S
        worst = max(worst, gap)
    return worst


def audit_record(record: Mapping[str, Any]) -> dict[str, Any]:
    positions, bad = argmax_violations(record.get("logprobs"))
    gap = non_generation_seconds(record)
    return {
        "non_generation_s": gap,
        "slow": gap > NON_GENERATION_LIMIT_S,
        "argmax_positions": positions,
        "argmax_violations": bad,
        "has_usage": any((a.get("usage") or {}) for a in record.get("attempts") or []),
    }


def record_files(output_root: Path, qids: Iterable[str]) -> list[Path]:
    """Every record of the given questions under the audited role dirs."""
    wanted = set(qids)
    files = []
    for sub in ROOTS:
        for role in ROLE_DIRS:
            base = output_root / sub / role if sub else output_root / role
            if not base.exists():
                continue
            for path in base.glob("*/*/*"):
                if path.is_dir() and path.name in wanted:
                    files += sorted(path.glob("*.json"))
                elif path.suffix == ".json" and path.stem in wanted:
                    files.append(path)  # level-free records (plans)
    return files


def audit(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("audit_test")
    qids = {r["question_id"] for r in ctx.selected("test")}
    qids |= {r["question_id"] for r in wrong_questions(ctx, "test")}
    root = ctx.paths.output_root
    files = record_files(root, qids)
    if ctx.dry_run:
        ctx.echo(f"[dry-run] audit {len(files)} test records")
        return {"records": len(files)}
    counts: Counter[str] = Counter()
    flagged = []
    worst = 0.0
    for path in files:
        record = read_json(path)
        result = audit_record(record)
        role = next(p for p in path.relative_to(root).parts if p not in ROOTS)
        counts[f"{role}:records"] += 1
        counts[f"{role}:no_usage"] += int(not result["has_usage"])
        counts["argmax_positions"] += result["argmax_positions"]
        counts["argmax_violations"] += result["argmax_violations"]
        worst = max(worst, result["non_generation_s"])
        if result["slow"] or result["argmax_violations"]:
            flagged.append(
                {
                    "path": str(path.relative_to(root)),
                    **result,
                    "stage": record.get("stage"),
                }
            )
    summary = {
        "time": utc_now(),
        "records": len(files),
        "limit_s": NON_GENERATION_LIMIT_S,
        "max_non_generation_s": worst,
        "counts": dict(sorted(counts.items())),
        "flagged": flagged,
    }
    atomic_write_json(ctx.paths.out(AUDIT_FILE), summary)
    ctx.paths.out(FLAGGED_FILE).write_text(
        "".join(f"{f['path']}\n" for f in flagged), encoding="utf-8"
    )
    ctx.log(
        f"audit test records: {len(files)} records, {len(flagged)} flagged, "
        f"max non-generation {worst:.2f} s, argmax violations "
        f"{counts['argmax_violations']}/{counts['argmax_positions']}"
    )
    return summary
