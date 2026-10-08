"""Data stages: probe (A3), prepare (split freeze), screen (L20 selection)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from . import dataset, sampling
from .client import PARSE_VERSION, STATUS_VALID
from .common import (
    StageError,
    atomic_write_json,
    deep_merge,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_effective_config,
    write_jsonl,
)
from .context import Context, code_hash, require
from .evaluate import add_correctness, is_correct
from .inference import READER_ROLE

SOURCE_HASH_CACHE = "source_image_sha256.json"
OUTCOME_TECHNICAL = "technical_failure"
REPEAT_NAMESPACE = "repeat"
EXPLORATORY_DEV_QUESTIONS = 20
EXPLORATORY_TEST_IMAGES = 30


# ---- probe (plan §16 A3) ----------------------------------------------------
def probe_questions(ctx: Context) -> list[dataset.QuestionRecord]:
    per_cell = int(ctx.cfg["probe"]["questions_per_cell"])
    by_cell: dict[str, list[dataset.QuestionRecord]] = defaultdict(list)
    for record in ctx.pool.records:
        by_cell[record.cell].append(record)
    chosen: list[dataset.QuestionRecord] = []
    for cell in ctx.strata.cells:
        ordered = sorted(
            by_cell.get(cell, []),
            key=lambda r: sampling.question_order_key(ctx.seed, r),
        )
        chosen.extend(ordered[:per_cell])
    return chosen


def probe(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("probe")
    require(
        not ctx.state.get("prepare"),
        "probe must run before `prepare` freezes the split (plan §16 A3)",
    )
    levels = [int(v) for v in ctx.cfg["probe"]["levels"]]
    chosen = probe_questions(ctx)
    plan = {
        "questions": len(chosen),
        "per_cell": dict(Counter(r.cell for r in chosen)),
        "levels": levels,
        "reader_requests": len(chosen) * len(levels),
        "outputs": [
            str(ctx.paths.out("probe")),
            str(ctx.paths.out("probe_summary.json")),
        ],
    }
    if ctx.dry_run:
        ctx.echo(f"[dry-run] probe plan: {plan}")
        return plan
    runner = ctx.runner()
    rows = []
    for record in chosen:
        question = {**record.public(), "split": "probe"}
        rows.extend(runner.run_question(question, levels, roles=(READER_ROLE,)))
    table = add_correctness(pd.DataFrame(rows), ctx.gt([r.question_id for r in chosen]))
    summary = summarize_probe(table, levels)
    out_dir = ctx.paths.out("probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    table.drop(columns=["r_values"]).to_csv(out_dir / "probe_results.csv", index=False)
    pd.DataFrame(summary["cell_level_accuracy"]).T.to_csv(
        out_dir / "probe_cell_level_accuracy.csv"
    )
    atomic_write_json(ctx.paths.out("probe_summary.json"), {**plan, **summary})
    ctx.state.set("probe", {"done": True, "time": utc_now(), "questions": len(chosen)})
    ctx.note_reuse()
    ctx.log(f"probe done: {len(chosen)} questions x {levels}")
    return summary


def summarize_probe(table: pd.DataFrame, levels: Sequence[int]) -> dict[str, Any]:
    acc = (
        table.groupby(["cell", "level"])["reader_correct"].mean().unstack("level")
        if not table.empty
        else pd.DataFrame()
    )
    first: dict[str, float] = {}
    for qid, frame in table.groupby("question_id"):
        hits = frame[frame["reader_correct"]]["level"]
        first[str(qid)] = float(hits.min()) if not hits.empty else math.inf
    histogram = Counter(
        "never" if math.isinf(v) else str(int(v)) for v in first.values()
    )
    cells = {
        cell: {str(int(level)): float(v) for level, v in row.items()}
        for cell, row in acc.iterrows()
    }
    overall = table.groupby("level")["reader_correct"].mean()
    return {
        "cell_level_accuracy": cells,
        "overall_level_accuracy": {str(int(k)): float(v) for k, v in overall.items()},
        "first_correct_level_histogram": dict(histogram),
        "status_counts": dict(Counter(table["reader_status"])),
        "levels": list(levels),
        "decision_hint": (
            "A3: if every cell has L1 accuracy >= 0.9 shift quota to smaller "
            "targets / material / shape; never drop a cell (shrink only)."
        ),
    }


# ---- prepare ------------------------------------------------------------------
def source_hashes(ctx: Context) -> dict[str, str]:
    cache_path = ctx.paths.data(SOURCE_HASH_CACHE)
    cache = read_json(cache_path) if cache_path.exists() else {}
    image_ids = [r.image_id for r in ctx.pool.records]
    hashes = dataset.compute_image_hashes(image_ids, ctx.paths.images_root, cache)
    if not ctx.dry_run and hashes != cache:
        atomic_write_json(cache_path, hashes)
    return hashes


def build_image_groups(ctx: Context) -> list[sampling.ImageGroup]:
    hashes = source_hashes(ctx)
    return sampling.build_groups(
        ctx.pool.records, dataset.duplicate_groups(hashes), ctx.seed
    )


def optional_cap(value: Any) -> int | None:
    """``null`` image cap = no cap (plan §16 A2-bis)."""
    return None if value is None else int(value)


def build_scan(
    ctx: Context,
) -> tuple[list[sampling.ImageGroup], list[sampling.ScanResult]]:
    """Dev scan first; test continues from where dev stopped (same hash order)."""
    dcfg = ctx.cfg["dataset"]
    groups = build_image_groups(ctx)
    dev_quota, test_quota = sampling.split_quotas(
        ctx.strata,
        sampling.cell_counts(groups),
        int(dcfg["candidate_dev_questions"]),
        int(dcfg["candidate_test_questions"]),
        int(dcfg["target_dev_questions"]),
        int(dcfg["target_test_questions"]),
    )
    per_image = int(dcfg["max_questions_per_image"])
    dev = sampling.scan_split(
        groups,
        "dev",
        "main",
        dev_quota,
        optional_cap(dcfg.get("candidate_dev_images_scan")),
        per_image,
        set(),
    )
    test = sampling.scan_split(
        groups,
        "test",
        "main",
        test_quota,
        optional_cap(dcfg.get("candidate_test_images_scan")),
        per_image,
        set(dev.groups_used),
        start_index=dev.end_index,
    )
    return groups, [dev, test]


def scan_report(scan: sampling.ScanResult) -> dict[str, Any]:
    filled = scan.filled()
    return {
        "split": scan.split,
        "batch": scan.batch,
        "quotas": scan.quotas.to_dict(),
        "filled": filled,
        "shortfall": {
            c: q - filled.get(c, 0)
            for c, q in scan.quotas.targets.items()
            if q > filled.get(c, 0)
        },
        "candidates": len(scan.candidates),
        "groups_used": len(scan.groups_used),
        "groups_scanned": scan.groups_scanned,
        "start_index": scan.start_index,
        "end_index": scan.end_index,
    }


def load_revision(
    ctx: Context, revision_file: Path | None, reason: str | None
) -> tuple[dict[str, Any], int]:
    """Validate a probe-based revision; nothing is persisted here."""
    prepared = ctx.state.get("prepare") or {}
    limit = int(ctx.cfg["probe"]["max_filter_revisions"])
    require(bool(prepared), "--revise needs an existing frozen prepare")
    require(prepared.get("revisions", 0) < limit, f"revision limit ({limit}) used")
    require(not ctx.state.get("screening"), "screening has started: revision forbidden")
    require(
        revision_file is not None and bool(reason),
        "--revision-file and --reason required",
    )
    overrides = read_json(revision_file)
    require(
        set(overrides) == {"dataset"}, "a revision may only override `dataset` keys"
    )
    return overrides, int(prepared.get("revisions", 0)) + 1


def prepare(
    ctx: Context,
    revise: bool = False,
    revision_file: Path | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Freeze the candidate split. ``--revise`` re-scans in memory first and
    persists overrides, manifests and the revision count only after success."""
    ctx.set_stage("prepare")
    if ctx.inherit_root is not None:
        raise StageError(
            f"this run inherits the frozen split of {ctx.inherit_root}; `prepare` "
            "is disabled so the shared data_root is never rewritten"
        )
    prepared = ctx.state.get("prepare")
    if prepared and not revise:
        raise StageError(
            "prepare is frozen; only one `--revise` is allowed before screening"
        )
    overrides: dict[str, Any] = {}
    revisions = 0
    before = ctx.cfg["dataset"]
    if revise:
        overrides, revisions = load_revision(ctx, revision_file, reason)
        ctx.refresh(deep_merge(ctx.cfg, overrides))
        ctx.echo(f"revision {revisions} (in memory): {overrides['dataset']}")
    _, scans = build_scan(ctx)
    reports = [scan_report(s) for s in scans]
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] eligible={len(ctx.pool.records)} exclusions={ctx.pool.reason_counts}"
        )
        for report in reports:
            ctx.echo(
                f"[dry-run] {report['split']}: {report['candidates']} candidates, "
                f"{report['groups_used']} images used, "
                f"{report['groups_scanned']} images scanned"
            )
            for cell, quota in report["quotas"]["targets"].items():
                ctx.echo(
                    f"[dry-run]   {cell}: size {report['quotas']['available'][cell]}"
                    f" quota {quota} filled {report['filled'].get(cell, 0)}"
                )
        ctx.echo(
            f"[dry-run] model requests: 0; would write {ctx.paths.data('candidate_manifest.jsonl')}"
        )
        return {"scans": reports}
    if revise:
        effective = write_effective_config(
            ctx.config_path,
            ctx.paths,
            overrides,
            f"probe-based dataset revision {revisions}: {reason}",
        )
        ctx.refresh(effective)
        ctx.log(
            f"dataset revision {revisions}: reason={reason}; before={before}; "
            f"after={effective['dataset']}"
        )
    candidates = [row for s in scans for row in s.candidates]
    write_jsonl(ctx.paths.data("candidate_manifest.jsonl"), candidates)
    mapping = {
        **ctx.pool.schema_mapping,
        "code_sha256": code_hash([dataset, sampling]),
        "seed": ctx.seed,
    }
    atomic_write_json(ctx.paths.out("schema_mapping.json"), mapping)
    atomic_write_json(ctx.paths.out("scan_summary.json"), {"scans": reports})
    ctx.state.set(
        "prepare",
        {
            "frozen": True,
            "time": utc_now(),
            "revisions": revisions,
            "candidate_manifest_sha256": sha256_file(
                ctx.paths.data("candidate_manifest.jsonl")
            ),
            "reserve_batches": 0,
            "scan_cursor": scans[-1].end_index,
        },
    )
    write_dataset_flow(ctx)
    ctx.log(f"candidate quota rule: {sampling.SPLIT_QUOTA_RULE}")
    for report in reports:
        ctx.log(
            f"{report['split']} quotas={report['quotas']['targets']} "
            f"cell_sizes={report['quotas']['available']} filled={report['filled']} "
            f"shortfall={report['shortfall']} images_scanned={report['groups_scanned']} "
            f"images_used={report['groups_used']}"
        )
    ctx.log(f"prepare frozen: {[(r['split'], r['candidates']) for r in reports]}")
    return {"scans": reports}


# ---- screen -------------------------------------------------------------------
def screen_rows(
    ctx: Context, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    runner = ctx.runner()
    top = ctx.n_levels
    gt = ctx.gt([r["question_id"] for r in rows])
    results = []
    for row in rows:
        record = runner.role_record(READER_ROLE, row, top)
        ok = record is not None and record["status"] == STATUS_VALID
        answer = record["parsed"]["answer"] if ok and record else None
        correct = is_correct(answer, gt[row["question_id"]]) if ok else None
        outcome = (
            OUTCOME_TECHNICAL if not ok else ("correct" if correct else "incorrect")
        )
        results.append(
            {
                **{
                    k: row[k]
                    for k in (
                        "question_id",
                        "image_id",
                        "group_id",
                        "split",
                        "batch",
                        "cell",
                        "attribute",
                        "relate_count",
                    )
                },
                "level": top,
                "reader_answer": answer,
                "gt_answer": gt[row["question_id"]],
                "status": record["status"] if record else None,
                "correct": correct,
                "outcome": outcome,
                "prompt_hash": record["prompt_hash"] if record else None,
                "image_sha256": record["image_sha256"] if record else None,
                "cache_key": record["cache_key"] if record else None,
                "model": ctx.settings.name,
                "digest": ctx.settings.digest,
            }
        )
    return results


def add_reserve_batch(ctx: Context, split: str, batch_index: int) -> int:
    """Next reserve batch, continuing the hash order after the last scan."""
    dcfg = ctx.cfg["dataset"]
    groups = build_image_groups(ctx)
    used = {r["group_id"] for r in ctx.candidates()}
    prepared = ctx.state.get("prepare") or {}
    cursor = int(prepared.get("scan_cursor", 0))
    batch_images = int(dcfg["reserve_batch_images"])
    per_image = int(dcfg["max_questions_per_image"])
    remaining = [g for g in groups[cursor:] if g.group_id not in used]
    quotas = sampling.allocate(
        batch_images * per_image, ctx.strata, sampling.cell_counts(remaining)
    )
    scan = sampling.scan_split(
        groups,
        split,
        f"reserve_{batch_index}",
        quotas,
        batch_images,
        per_image,
        used,
        start_index=cursor,
    )
    write_jsonl(
        ctx.paths.data("candidate_manifest.jsonl"),
        [*ctx.candidates(), *scan.candidates],
    )
    ctx.state.set("prepare", {**prepared, "scan_cursor": scan.end_index})
    ctx.log(
        f"reserve batch {batch_index} for {split}: {len(scan.candidates)} candidates, "
        f"{scan.groups_scanned} images scanned from index {cursor}"
    )
    return len(scan.candidates)


def merge_split_rows(path: Path, split: str, rows: Sequence[Mapping[str, Any]]) -> None:
    kept = [r for r in read_jsonl(path) if r["split"] != split]
    write_jsonl(path, [*kept, *rows])


def screen(ctx: Context, split: str) -> dict[str, Any]:
    ctx.set_stage(f"screen_{split}")
    frozen_policy = ctx.paths.out("frozen_policy.json").exists()
    require(ctx.split_frozen, "run `prepare` before screening")
    require(
        ctx.inherit_root is None or ctx.paths.selection_root != ctx.paths.data_root,
        "inherited split: paths.selection_root must differ from the shared data_root",
    )
    if split == "dev":
        require(not frozen_policy, "dev screening after freeze is forbidden")
    else:
        require(frozen_policy, "test screening requires `freeze` first (plan §10)")
        ctx.require_frozen_match()
    candidates = ctx.candidates(split)
    if ctx.dry_run:
        repeat = (
            int(ctx.cfg["dataset"]["screening_repeat_dev_questions"])
            if split == "dev"
            else 0
        )
        ctx.echo(
            f"[dry-run] screen {split}: {len(candidates)} L{ctx.n_levels} reader requests "
            f"(+{repeat} repeat); reserve batches only if below minimum"
        )
        return {"candidates": len(candidates)}
    dcfg = ctx.cfg["dataset"]
    target = int(dcfg[f"target_{split}_questions"])
    min_q, min_img = (
        int(dcfg[f"minimum_{split}_questions"]),
        int(dcfg[f"minimum_{split}_images"]),
    )
    while True:
        candidates = ctx.candidates(split)
        results = screen_rows(ctx, candidates)
        passed = {r["question_id"] for r in results if r["outcome"] == "correct"}
        adopted, allocation = sampling.adopt(
            candidates, passed, target, ctx.strata, ctx.seed
        )
        batches = int((ctx.state.get("prepare") or {}).get("reserve_batches", 0))
        if sampling.meets_minimum(adopted, min_q, min_img):
            break
        if ctx.inherit_root is not None:
            ctx.log(f"{split}: below minimum; inherited split has no reserve batches")
            break
        if batches >= int(dcfg["max_reserve_batches_total"]):
            ctx.log(f"{split}: below minimum after {batches} reserve batches")
            break
        ctx.state.set(
            "prepare", {**ctx.state.get("prepare"), "reserve_batches": batches + 1}
        )
        add_reserve_batch(ctx, split, batches + 1)
    adopted_ids = {r["question_id"] for r in adopted}
    results = [{**r, "adopted": r["question_id"] in adopted_ids} for r in results]
    merge_split_rows(ctx.paths.selection("selection_results.jsonl"), split, results)
    selected = [{**r, "adoption_rank": i} for i, r in enumerate(adopted)]
    merge_split_rows(ctx.paths.selection("selected_manifest.jsonl"), split, selected)
    atomic_write_json(ctx.paths.out(f"adoption_{split}.json"), allocation.to_dict())
    info = screening_info(ctx, split, results, adopted)
    mismatch = reference_mismatch(ctx, split, adopted, allocation.to_dict())
    if ctx.inherit_root is not None:
        info = {**info, "reference_match": mismatch is None}
    screening = {**(ctx.state.get("screening") or {}), split: info}
    ctx.state.set("screening", screening)
    if mismatch is not None:
        ctx.log(f"STOP: {split} adoption differs from the inherited run: {mismatch}")
        raise StageError(
            f"adopted {split} set differs from {ctx.inherit_root}: {mismatch}"
        )
    if split == "dev":
        repeat_diagnostic(ctx, adopted, results)
    write_dataset_flow(ctx)
    ctx.log(f"screen {split}: {info}")
    ctx.note_reuse()
    return info


def reference_mismatch(
    ctx: Context,
    split: str,
    adopted: Sequence[Mapping[str, Any]],
    allocation: Mapping[str, Any],
) -> str | None:
    """For an inherited split, the adopted ids and cell targets must equal the
    original run's (plan §17: same 100 dev questions)."""
    if ctx.inherit_root is None:
        return None
    reference = [
        r["question_id"]
        for r in read_jsonl(ctx.paths.data("selected_manifest.jsonl"))
        if r["split"] == split
    ]
    ours = [r["question_id"] for r in adopted]
    if ours != reference:
        missing = sorted(set(reference) - set(ours))
        extra = sorted(set(ours) - set(reference))
        return f"ids differ (missing {missing[:10]}, extra {extra[:10]}, order equal={set(ours) == set(reference)})"
    ref_alloc = read_json(ctx.inherit_root / f"adoption_{split}.json")
    if ref_alloc.get("targets") != allocation.get("targets"):
        return f"cell targets differ: {ref_alloc.get('targets')} vs {allocation.get('targets')}"
    return None


def screening_info(
    ctx: Context,
    split: str,
    results: Sequence[Mapping[str, Any]],
    adopted: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    images = len({r["group_id"] for r in adopted})
    exploratory = (split == "dev" and len(adopted) < EXPLORATORY_DEV_QUESTIONS) or (
        split == "test" and images < EXPLORATORY_TEST_IMAGES
    )
    return {
        "done": True,
        "time": utc_now(),
        "candidates": len(results),
        "outcomes": dict(Counter(r["outcome"] for r in results)),
        "adopted": len(adopted),
        "adopted_images": images,
        "attribute_distribution": dict(Counter(r["attribute"] for r in adopted)),
        "exploratory_only": exploratory,
        "model": ctx.settings.name,
        "digest": ctx.settings.digest,
        "reader_prompt_hash": ctx.prompts.hashes[READER_ROLE],
        "parse_version": PARSE_VERSION,
        "prompt_hashes": ctx.prompts.hashes,
    }


def repeat_diagnostic(
    ctx: Context,
    adopted: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> None:
    """Second L20 reader run on the first N adopted dev questions (plan §3.3)."""
    count = int(ctx.cfg["dataset"]["screening_repeat_dev_questions"])
    first = {r["question_id"]: r for r in results}
    runner = ctx.runner()
    rows = []
    for row in adopted[:count]:
        record = runner.role_record(
            READER_ROLE, row, ctx.n_levels, namespace=REPEAT_NAMESPACE
        )
        ok = record is not None and record["status"] == STATUS_VALID
        answer = record["parsed"]["answer"] if ok and record else None
        original = first[row["question_id"]]
        rows.append(
            {
                "question_id": row["question_id"],
                "first_answer": original["reader_answer"],
                "repeat_answer": answer,
                "agree": answer is not None
                and dataset.normalize_text(answer)
                == dataset.normalize_text(str(original["reader_answer"])),
                "repeat_correct": is_correct(answer, original["gt_answer"]),
            }
        )
    n = max(len(rows), 1)
    atomic_write_json(
        ctx.paths.out("screening_repeat.json"),
        {
            "n": len(rows),
            "agreement_rate": sum(r["agree"] for r in rows) / n,
            "repeat_accuracy": sum(r["repeat_correct"] for r in rows) / n,
            "backend": "ollama /api/chat, temperature 0, fixed seed (GPU may be non-deterministic)",
            "rows": rows,
            "note": "diagnostic only; never changes the selected set",
        },
    )


# ---- dataset_flow.csv -----------------------------------------------------------
def write_dataset_flow(ctx: Context) -> None:
    rows: list[dict[str, Any]] = []

    def add(
        stage: str, split: str, cell: str, metric: str, value: Any, note: str = ""
    ) -> None:
        rows.append(
            {
                "stage": stage,
                "split": split,
                "cell": cell,
                "metric": metric,
                "value": value,
                "note": note,
            }
        )

    mapping = ctx.pool.schema_mapping
    add("filter", "all", "all", "total_questions", mapping["total_questions"])
    for reason, count in sorted(mapping["exclusion_counts"].items()):
        add("filter", "all", "all", f"reason:{reason}", count)
    for cell, count in sorted(mapping["cell_counts"].items()):
        add("filter", "all", cell, "eligible", count)
    scan_path = ctx.paths.out("scan_summary.json")
    for scan in read_json(scan_path)["scans"] if scan_path.exists() else []:
        split, batch = scan["split"], scan["batch"]
        quotas = scan["quotas"]
        for cell, value in quotas["targets"].items():
            add("candidates", split, cell, "cell_size", quotas["available"][cell])
            add("candidates", split, cell, "share_quota", quotas["initial"][cell])
            add("candidates", split, cell, "quota", value, batch)
            add("candidates", split, cell, "filled", scan["filled"].get(cell, 0))
        add("candidates", split, "all", "images_scanned", scan["groups_scanned"])
        add("candidates", split, "all", "images_used", scan["groups_used"])
        for line in quotas["log"]:
            add("candidates", split, "all", "quota_rule", "", line)
    for row in Counter((c["split"], c["batch"]) for c in ctx.candidates()).items():
        add("candidates", row[0][0], "all", f"candidates:{row[0][1]}", row[1])
    results = read_jsonl(ctx.paths.selection("selection_results.jsonl"))
    for (split, cell, outcome), n in sorted(
        Counter((r["split"], r["cell"], r["outcome"]) for r in results).items()
    ):
        add("screening", split, cell, outcome, n)
    for split in ("dev", "test"):
        path = ctx.paths.out(f"adoption_{split}.json")
        if not path.exists():
            continue
        allocation = read_json(path)
        for cell, value in allocation["targets"].items():
            add("adoption", split, cell, "target_after_reallocation", value)
        for line in allocation["log"]:
            add("adoption", split, "all", "reallocation", "", line)
        for cell, n in Counter(r["cell"] for r in ctx.selected(split)).items():
            add("adoption", split, cell, "adopted", n)
    eff = ctx.paths.out("effective_config.json")
    for item in read_json(eff).get("history", []) if eff.exists() else []:
        if "revision" in item["note"]:
            add(
                "revision", "all", "all", "dataset_revision", item["time"], item["note"]
            )
    pd.DataFrame(rows).to_csv(ctx.paths.out("dataset_flow.csv"), index=False)
