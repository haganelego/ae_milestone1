"""Fair v1 (plan §18.1): re-freeze v1 on the dev visual subset and re-evaluate
on the test visual subset from cached records only (no inference).

This is a RE-ANALYSIS of run v1. Outputs go to ``<v1 output>/fair_subset/``;
v1's original frozen_policy.json and report are never touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import report as report_mod
from .common import atomic_write_json, read_json, sha256_file, utc_now
from .context import Context, require
from .evaluate import evaluate
from .policy import freeze_policy
from .stages_model import full_table, provenance, require_screening_match
from .textprobe import load_subset

FAIR_DIR = "fair_subset"
FROZEN_NAME = "frozen_policy_subset.json"
LABEL = (
    "Re-analysis of run v1 restricted to the visual subset (plan §18.1): thresholds "
    "re-frozen on dev visual questions, test re-evaluated on test visual questions; "
    "cached records only, no new inference."
)


def restrict(table: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
    sub = table[table["question_id"].isin(ids)]
    missing = ids - set(sub["question_id"])
    require(not missing, f"subset ids without records: {sorted(missing)[:10]}")
    return sub


def freeze_on_subset(ctx: Context, subset_path: Path) -> dict[str, Any]:
    ctx.set_stage("freeze_subset")
    require(ctx.family == "confidence", "subset freeze is defined for the v1 family")
    require_screening_match(ctx, "dev")
    require(
        bool((ctx.state.get("infer") or {}).get("dev", {}).get("complete")),
        "dev inference must be complete",
    )
    subset = load_subset(subset_path)
    out = ctx.paths.out(FAIR_DIR)
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] subset freeze on {len(subset['dev'])} dev questions -> {out}"
        )
        return {}
    table = restrict(full_table(ctx, "dev"), set(subset["dev"]))
    frozen = freeze_policy(
        table,
        ctx.cfg["policy"],
        ctx.n_levels,
        out,
        {
            **provenance(ctx),
            "analysis": LABEL,
            "subset_file": str(subset_path),
            "subset_sha256": sha256_file(subset_path),
            "n_dev_subset": int(table["question_id"].nunique()),
        },
        frozen_name=FROZEN_NAME,
    )
    ctx.state.set("fair_subset_freeze", {"done": True, "time": utc_now()})
    ctx.log(f"fair-v1 subset freeze: {frozen['selections'][frozen['primary_target']]}")
    return frozen


def evaluate_on_subset(ctx: Context, subset_path: Path) -> dict[str, Any]:
    ctx.set_stage("evaluate_subset")
    out = ctx.paths.out(FAIR_DIR)
    frozen_path = out / FROZEN_NAME
    require(frozen_path.exists(), "run `freeze --subset` first")
    frozen = read_json(frozen_path)
    ctx.require_frozen_match(frozen)
    require(
        frozen.get("subset_sha256") == sha256_file(subset_path),
        "subset file differs from the one used for the subset freeze",
    )
    require_screening_match(ctx, "test")
    require(
        bool((ctx.state.get("infer") or {}).get("test", {}).get("complete")),
        "test inference must be complete",
    )
    subset = load_subset(subset_path)
    if ctx.dry_run:
        ctx.echo(f"[dry-run] subset evaluate on {len(subset['test'])} test questions")
        return {}
    table = restrict(full_table(ctx, "test"), set(subset["test"]))
    gt = ctx.gt(table["question_id"].unique().tolist())
    summary = evaluate(table, frozen, ctx.cfg["evaluation"], ctx.n_levels, out, gt)
    methods = pd.read_csv(out / "method_summary.csv")
    report_mod.figure_tradeoff(
        methods,
        str(frozen["primary_target"]),
        out / "figures" / "test_meanlevel_accuracy.png",
    )
    atomic_write_json(
        out / "analysis_label.json",
        {
            "label": LABEL,
            "n_test_subset": summary["n_questions"],
            "n_test_images": summary["n_images"],
            "subset_file": str(subset_path),
            "time": utc_now(),
        },
    )
    ctx.state.set("fair_subset_evaluate", {"done": True, "time": utc_now()})
    ctx.log(f"fair-v1 subset evaluate: {summary['n_questions']} test questions")
    return summary
