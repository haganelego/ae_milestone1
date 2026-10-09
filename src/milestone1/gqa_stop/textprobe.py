"""`textprobe` (plan §18.1): which adopted questions are answerable without an image.

Every adopted dev/test question of the inherited run is answered once from the
question text alone (role ``text_only``, no image, the run's decoding). A
question whose normalised text-only answer equals the normalised GT is
*prior-answerable* and leaves the main evaluation; the rest form the *visual
subset*. The exclusion is made once, before any method result is seen.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .client import STATUS_VALID, RequestSpec
from .common import (
    StageError,
    atomic_write_json,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
)
from .context import Context, require
from .evaluate import is_correct
from .schemas import validate_reader

TEXT_ONLY_ROLE = "text_only"
SUBSET_FILE = "visual_subset.json"
SPLITS = ("dev", "test")
RULE = (
    "text_only answer != GT after normalisation (NFKC, lowercase, strip, collapse "
    "whitespace); text_only failures stay in the visual subset"
)


def text_only_spec(question: Mapping[str, Any]) -> RequestSpec:
    """Question text only: no image, no GT, no metadata."""
    return RequestSpec(
        role=TEXT_ONLY_ROLE,
        question_id=str(question["question_id"]),
        level=None,
        input_data={"question": str(question["question"])},
    )


def reference_adopted(ctx: Context) -> dict[str, list[dict[str, Any]]]:
    """Adopted rows of the inherited run (shared data_root), in adoption order."""
    require(ctx.inherit_root is not None, "textprobe needs paths.inherit_split_from")
    rows = read_jsonl(ctx.paths.data("selected_manifest.jsonl"))
    return {split: [r for r in rows if r["split"] == split] for split in SPLITS}


def build_subset(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    subset: dict[str, Any] = {"rule": RULE, "created": utc_now()}
    excluded: dict[str, list[str]] = {}
    failures: dict[str, list[str]] = {}
    for split in SPLITS:
        rows = [r for r in results if r["split"] == split]
        subset[split] = [r["question_id"] for r in rows if not r["text_ok"]]
        excluded[split] = [r["question_id"] for r in rows if r["text_ok"]]
        failures[split] = [
            r["question_id"] for r in rows if r["status"] != STATUS_VALID
        ]
    subset["excluded"] = excluded
    subset["text_only_failures_kept"] = failures
    subset["counts"] = {
        split: {
            "adopted": len(subset[split]) + len(excluded[split]),
            "visual": len(subset[split]),
            "prior_answerable": len(excluded[split]),
        }
        for split in SPLITS
    }
    return subset


def textprobe(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("textprobe")
    subset_path = ctx.paths.out(SUBSET_FILE)
    if subset_path.exists():
        raise StageError(
            f"{subset_path} exists: the exclusion is made only once (§18.1)"
        )
    adopted = reference_adopted(ctx)
    questions = [r for split in SPLITS for r in adopted[split]]
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] textprobe: {len(questions)} text-only requests "
            f"({ {s: len(adopted[s]) for s in SPLITS} }), no image"
        )
        return {"planned_requests": len(questions)}
    engine = ctx.engine()
    gt = ctx.gt([q["question_id"] for q in questions])
    results = []
    for row in questions:
        record = engine.run(text_only_spec(row), validate_reader)
        ok = record["status"] == STATUS_VALID
        answer = record["parsed"]["answer"] if ok else None
        results.append(
            {
                "question_id": row["question_id"],
                "split": row["split"],
                "cell": row["cell"],
                "attribute": row["attribute"],
                "question": row["question"],
                "gt": gt[row["question_id"]],
                "text_only": answer,
                "status": record["status"],
                "text_ok": is_correct(answer, gt[row["question_id"]]),
            }
        )
    pd.DataFrame(results).to_csv(ctx.paths.out("textprobe.csv"), index=False)
    subset = build_subset(results)
    atomic_write_json(subset_path, subset)
    ctx.state.set(
        "textprobe",
        {
            "done": True,
            "time": utc_now(),
            "subset_sha256": sha256_file(subset_path),
            "counts": subset["counts"],
        },
    )
    ctx.log(f"textprobe: {subset['counts']}")
    return subset


def load_subset(path: Any) -> dict[str, Any]:
    subset = read_json(path)
    for split in SPLITS:
        require(isinstance(subset.get(split), list), f"subset file lacks '{split}' ids")
    return subset
