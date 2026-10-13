"""Plan §27 (v27) dataset: dev from GQA val_balanced, test from testdev_balanced.

Filter (``dataset.filter.mode == "minimal"``): structural == query, semantic ==
attr, final argument in {color, material, shape}; nothing else is excluded.
Order: image groups (byte-identical duplicates merged) in SHA256(seed|group_id)
order and questions inside a group in SHA256(seed|question_id) order, i.e. the
fixed hash order of every earlier run (``sampling.build_groups``).
- dev: the first ``dev_questions`` questions in that order, at most
  ``dev_max_questions_per_image`` per image group (the first ones in question
  hash order); no stratification.
- test: every eligible testdev question (no per-image cap).
Manifests hold no GT; the answers are written to a separate ``gt_<split>.json``
that only screening and evaluation read.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import dataset, sampling
from .common import (
    atomic_write_json,
    read_json,
    read_jsonl,
    resolve_path,
    sha256_file,
    utc_now,
    write_jsonl,
)
from .context import Context, require

SPLITS = ("dev", "test")
SOURCE_HASH_CACHE = "source_image_sha256.json"
GT_REF = {"dev": "gqa_val_balanced", "test": "gqa_testdev_balanced"}
ALL = "all"


def questions_file(ctx: Context, split: str) -> Path:
    key = "gqa_questions" if split == "dev" else "gqa_questions_test"
    return resolve_path(ctx.paths.repo_root, ctx.cfg["paths"][key])


def manifest_path(ctx: Context, split: str) -> Path:
    return ctx.paths.selection(f"questions_{split}.jsonl")


def gt_path(ctx: Context, split: str) -> Path:
    return ctx.paths.selection(f"gt_{split}.json")


def load_manifest(ctx: Context, split: str) -> list[dict[str, Any]]:
    path = manifest_path(ctx, split)
    require(path.exists(), f"{path} missing: run `v27-extract` first")
    return read_jsonl(path)


def load_gt(ctx: Context, split: str) -> dict[str, str]:
    return {str(k): str(v) for k, v in read_json(gt_path(ctx, split)).items()}


# ---- pure selection ---------------------------------------------------------------------
def select_questions(
    groups: Sequence[sampling.ImageGroup],
    max_questions: int | None,
    max_per_group: int | None,
) -> list[tuple[dataset.QuestionRecord, sampling.ImageGroup, int]]:
    """Walk groups in hash order, take the first ``max_per_group`` questions of
    each (question hash order) until ``max_questions`` (None = all)."""
    chosen: list[tuple[dataset.QuestionRecord, sampling.ImageGroup, int]] = []
    for group in groups:
        take = (
            group.questions
            if max_per_group is None
            else group.questions[:max_per_group]
        )
        for rank, record in enumerate(take):
            if max_questions is not None and len(chosen) >= max_questions:
                return chosen
            chosen.append((record, group, rank))
    return chosen


def manifest_row(
    record: dataset.QuestionRecord, group: sampling.ImageGroup, split: str, rank: int
) -> dict[str, Any]:
    """Public fields only (GT referenced by id)."""
    return {
        "question_id": record.question_id,
        "image_id": record.image_id,
        "group_id": group.group_id,
        "group_image_ids": list(group.image_ids),
        "split": split,
        "question": record.question,
        "attribute": record.attribute,
        "relate_count": record.relate_count,
        "operation_count": record.operation_count,
        "detailed_type": record.detailed_type,
        "cell": record.cell,
        "gt_ref": f"{GT_REF[split]}:{record.question_id}",
        "selection_reason": f"v27_hash_order:rank_in_image={rank}",
    }


def breakdown(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "questions": len(rows),
        "images": len({r["image_id"] for r in rows}),
        "groups": len({r.get("group_id", r["image_id"]) for r in rows}),
        "attribute": dict(sorted(Counter(r["attribute"] for r in rows).items())),
        "relate_count": {
            str(k): v
            for k, v in sorted(Counter(int(r["relate_count"]) for r in rows).items())
        },
        "questions_per_image": {
            str(k): v
            for k, v in sorted(
                Counter(Counter(r["image_id"] for r in rows).values()).items()
            )
        },
    }


# ---- stage ------------------------------------------------------------------------------
def image_groups(
    ctx: Context, records: Sequence[dataset.QuestionRecord], split: str
) -> list[sampling.ImageGroup]:
    cache_path = ctx.paths.data(f"{split}_{SOURCE_HASH_CACHE}")
    cache = read_json(cache_path) if cache_path.exists() else {}
    hashes = dataset.compute_image_hashes(
        [r.image_id for r in records], ctx.paths.images_root, cache
    )
    if not ctx.dry_run and hashes != cache:
        atomic_write_json(cache_path, hashes)
    missing = sorted(i for i, h in hashes.items() if not h)
    require(not missing, f"{split}: {len(missing)} images missing, e.g. {missing[:3]}")
    return sampling.build_groups(records, dataset.duplicate_groups(hashes), ctx.seed)


def split_limits(ctx: Context, split: str) -> tuple[int | None, int | None]:
    dcfg = ctx.cfg["dataset"]
    count = dcfg[f"{split}_questions"]
    per_image = dcfg.get(f"{split}_max_questions_per_image")
    return (
        None if count == ALL else int(count),
        None if per_image is None else int(per_image),
    )


def extract(ctx: Context) -> dict[str, Any]:
    """Write questions_<split>.jsonl and gt_<split>.json for dev and test (once)."""
    ctx.set_stage("v27_extract")
    filter_cfg = ctx.cfg["dataset"]["filter"]
    require(
        filter_cfg.get("mode") == dataset.MINIMAL_FILTER_MODE,
        "v27 needs dataset.filter.mode == 'minimal'",
    )
    existing = [s for s in SPLITS if manifest_path(ctx, s).exists()]
    require(not existing, f"manifests exist for {existing}: extraction is frozen")
    summary: dict[str, Any] = {"time": utc_now(), "seed": ctx.seed}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    gts: dict[str, dict[str, str]] = {}
    for split in SPLITS:
        source = questions_file(ctx, split)
        raw = dataset.load_questions(source)
        pool = dataset.build_eligible_pool(raw, filter_cfg)
        groups = image_groups(ctx, pool.records, split)
        limit, per_image = split_limits(ctx, split)
        chosen = select_questions(groups, limit, per_image)
        rows = [manifest_row(r, g, split, rank) for r, g, rank in chosen]
        if limit is not None:
            require(
                len(rows) == limit, f"{split}: only {len(rows)} of {limit} questions"
            )
        rows_by_split[split] = rows
        gts[split] = {
            r["question_id"]: str(raw[r["question_id"]]["answer"]) for r in rows
        }
        summary[split] = {
            "source": str(source),
            "source_sha256": sha256_file(source),
            "source_questions": len(raw),
            "exclusion_counts": pool.reason_counts,
            "eligible": breakdown([r.to_dict() for r in pool.records]),
            "eligible_groups": len(groups),
            "limit": limit,
            "max_per_image": per_image,
            "selected": breakdown(rows),
            "groups_scanned_for_selection": len({r["group_id"] for r in rows}),
        }
    dev_images = {i for r in rows_by_split["dev"] for i in r["group_image_ids"]}
    test_images = {i for r in rows_by_split["test"] for i in r["group_image_ids"]}
    overlap = sorted(dev_images & test_images)
    require(not overlap, f"dev/test share images: {overlap[:5]}")
    summary["dev_test_image_overlap"] = 0
    if ctx.dry_run:
        for split in SPLITS:
            ctx.echo(f"[dry-run] {split}: {summary[split]['selected']}")
        return summary
    for split in SPLITS:
        write_jsonl(manifest_path(ctx, split), rows_by_split[split])
        atomic_write_json(gt_path(ctx, split), gts[split])
        summary[split]["manifest_sha256"] = sha256_file(manifest_path(ctx, split))
    atomic_write_json(ctx.paths.out("dataset_v27.json"), summary)
    ctx.state.set("v27_extract", {"done": True, "time": utc_now()})
    for split in SPLITS:
        sel = summary[split]["selected"]
        ctx.log(
            f"v27 extract {split}: eligible {summary[split]['eligible']['questions']} q / "
            f"{summary[split]['eligible']['images']} img; selected {sel['questions']} q / "
            f"{sel['images']} img; attribute {sel['attribute']}; relate {sel['relate_count']}"
        )
    return summary
