"""Plan §26: model-swap replication on the v22 test sets, DESCRIPTIVE θ sweep only.

A model-swap run (config block ``model_swap``) reuses the reference run's
question sets (test visual subset and reader-wrong test set, both defined by
the reference model's screening reader), images, prompts and A13 decoding;
only the model changes. Nothing is selected or frozen on test.

- ``swap_provenance``: records sha256 of the shared set / image files and the
  question-id lists once, before inference (refuses to overwrite).
- ``sweep_swap``: cache-only replay of the swapped model's records, the θ sweep
  (``sweep_v22`` rules: k = 1, θ 0.50..1.00 step 0.05, forced L20 not a firing),
  the swapped reader's fixed-level curve and L20 ceiling, and the reference
  model's sweep and fixed curve beside it. Figures: one per axis, both models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .common import (
    atomic_write_json,
    read_id_csv,
    read_json,
    resolve_path,
    sha256_file,
    sha256_json,
    utc_now,
)
from .context import Context, require, table_to_csv
from .devgrid import fixed_curve
from .firing_v23 import run_tables, without_v1_columns, wrong_questions
from .stages_model import full_table
from .stages_v22_test import EXT_DIR
from .sweep_v22 import SWEEP_K, SWEEP_METHODS, SWEEP_THETAS, sweep_frame

OUT_DIR = "sweep_swap"
PROVENANCE_FILE = "model_swap_provenance.json"
REFERENCE_SWEEP = "sweep_v22/test_sweep.csv"
REFERENCE_ANSWERABLE = "level_table_test_merged.csv"
REFERENCE_WRONG = f"{EXT_DIR}/level_table.csv"
LABEL = (
    "DESCRIPTIVE model-swap sweep (plan §26): no θ is selected on test and no "
    "claim is made from the sweep beyond description"
)
AXES = (
    ("accuracy", "accuracy (answerable set)"),
    ("mean_level", "mean stop level (answerable set)"),
    ("median_level", "median stop level (answerable set)"),
    ("stop20_rate", "stop20 rate (answerable set)"),
    ("firing_rate", "firing rate (reader-wrong set)"),
)
SWAP_STYLE, REFERENCE_STYLE = ("-", "o"), ("--", "x")
# Supplementary full grid (team-lead request 2026-10-11): 4b scores rarely reach
# the main grid (direct_v4 max 0.66), so θ 0.05..1.00 is emitted separately.
FULL_START, FULL_STEP, FULL_COUNT = 0.05, 0.05, 20
FULL_THETAS = tuple(round(FULL_START + FULL_STEP * i, 2) for i in range(FULL_COUNT))
SUPP_DIR = "supplementary_full_grid"
SUPP_LABEL = (
    "SUPPLEMENTARY full θ grid 0.05..1.00 (descriptive; the user's main grid is "
    "0.50..1.00)"
)
MAIN_TITLE = "DESCRIPTIVE sweep (not a result)"
SUPP_TITLE = "SUPPLEMENTARY full grid, DESCRIPTIVE (not a result)"
TOLERANCE_ROLE_DIRS = ("plans_v3", "observer_v3_lp")
CHECK_COLUMNS = ("accuracy", "mean_level", "firing_rate")


# ---- provenance ---------------------------------------------------------------------------
def shared_files(ctx: Context) -> dict[str, Path]:
    """Files that define the sets and the images, shared with the reference run."""
    subset = ctx.cfg["paths"]["subset_file"]
    return {
        "selected_manifest": ctx.paths.selection("selected_manifest.jsonl"),
        "selection_results": ctx.paths.selection("selection_results.jsonl"),
        "subset_file": resolve_path(ctx.paths.repo_root, subset),
        "candidate_manifest": ctx.paths.data("candidate_manifest.jsonl"),
        "image_manifest": ctx.paths.data("image_manifest.jsonl"),
    }


def set_ids(ctx: Context) -> dict[str, list[str]]:
    return {
        "answerable": sorted(str(r["question_id"]) for r in ctx.selected("test")),
        "reader_wrong": sorted(
            str(r["question_id"]) for r in wrong_questions(ctx, "test")
        ),
    }


def provenance_record(ctx: Context) -> dict[str, Any]:
    swap = ctx.model_swap or {}
    ids = set_ids(ctx)
    files = shared_files(ctx)
    return {
        "files": {
            name: {
                "path": str(path.relative_to(ctx.paths.repo_root)),
                "sha256": sha256_file(path),
            }
            for name, path in files.items()
        },
        "sets": {
            name: {"n": len(values), "ids_sha256": sha256_json(values)}
            for name, values in ids.items()
        },
        "model": ctx.settings.name,
        "digest": ctx.settings.digest,
        "fallback_name": ctx.cfg["model"].get("fallback_name"),
        "reference_model": swap.get("reference_model"),
        "reference_output_root": swap.get("reference_output_root"),
    }


def swap_provenance(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("swap_provenance")
    require(ctx.model_swap is not None, "config has no `model_swap` block (plan §26)")
    path = ctx.paths.out(PROVENANCE_FILE)
    require(not path.exists(), f"{path} exists: provenance is recorded once")
    record = {"time": utc_now(), "plan": "§26", **provenance_record(ctx)}
    if ctx.dry_run:
        ctx.echo(f"[dry-run] provenance -> {path}: {record['sets']}")
        return record
    atomic_write_json(path, record)
    sizes = {k: v["n"] for k, v in record["sets"].items()}
    ctx.log(f"model-swap provenance recorded (plan §26): {sizes}")
    return record


def check_provenance(ctx: Context) -> dict[str, Any]:
    """The shared files and the sets must be unchanged since inference started."""
    path = ctx.paths.out(PROVENANCE_FILE)
    require(path.exists(), "run `swap-provenance` first")
    recorded = read_json(path)
    current = provenance_record(ctx)
    for key in ("files", "sets", "model", "digest"):
        require(recorded[key] == current[key], f"model-swap provenance changed: {key}")
    return recorded


# ---- frames --------------------------------------------------------------------------------
def fixed_frame(
    answerable: pd.DataFrame, wrong: pd.DataFrame, model: str
) -> pd.DataFrame:
    """Reader accuracy per level on both sets (the answerable one is the fixed curve)."""
    ans, bad = fixed_curve(answerable), fixed_curve(wrong)
    return pd.DataFrame(
        [
            {
                "model": model,
                "level": level,
                "answerable_reader_accuracy": ans[level],
                "reader_wrong_reader_accuracy": bad.get(level),
            }
            for level in sorted(ans)
        ]
    )


def side_by_side(
    swap: pd.DataFrame, reference: pd.DataFrame, swap_model: str, ref_model: str
) -> pd.DataFrame:
    """Long frame (model column) restricted to the shared methods and θ."""
    keep = ["method", "theta"]
    ref = reference.assign(theta=reference["theta"].round(2))
    out = swap.assign(theta=swap["theta"].round(2))
    shared = out[keep].merge(ref[keep], on=keep)
    require(len(shared) == len(out), "reference sweep lacks a method/θ of the swap run")
    return pd.concat(
        [
            out.assign(model=swap_model),
            ref.merge(shared, on=keep).assign(model=ref_model),
        ],
        ignore_index=True,
    )


def same_ids(table: pd.DataFrame, expected: Sequence[str]) -> bool:
    return sorted(table["question_id"].astype(str).unique()) == sorted(expected)


# ---- figures -------------------------------------------------------------------------------
def figures(
    combined: pd.DataFrame,
    fixed: pd.DataFrame | None,
    models: Sequence[str],
    out: Path,
    prefix: str = "swap",
    label: str = MAIN_TITLE,
) -> list[Path]:
    """One figure per axis; swapped model solid, reference model dashed."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    colours = {m: f"C{i}" for i, m in enumerate(SWEEP_METHODS)}
    styles = dict(zip(models, (SWAP_STYLE, REFERENCE_STYLE), strict=True))
    paths = []
    plots = [(m, t, "theta", "θ (k = 1)") for m, t in AXES]
    plots.append(("accuracy", "accuracy vs firing rate", "firing_rate", "firing rate"))
    for metric, title, x, xlabel in plots:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for model in models:
            line, marker = styles[model]
            frame = combined[combined["model"] == model]
            for method, g in frame.groupby("method", sort=False):
                g = g.sort_values("theta")
                ax.plot(
                    g[x],
                    g[metric],
                    line,
                    marker=marker,
                    ms=3,
                    color=colours[str(method)],
                    label=f"{method} ({model})",
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric)
        ax.set_title(f"{title}, test\n{label}", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        name = "accuracy_vs_firing" if x == "firing_rate" else metric
        path = out / f"{prefix}_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    if fixed is not None:
        paths.append(fixed_figure(fixed, models, styles, out))
    return paths


def fixed_figure(
    fixed: pd.DataFrame,
    models: Sequence[str],
    styles: Mapping[str, tuple[str, str]],
    out: Path,
) -> Path:
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    columns = (
        ("answerable_reader_accuracy", "C0", "answerable"),
        ("reader_wrong_reader_accuracy", "C3", "reader-wrong"),
    )
    for model in models:
        line, marker = styles[model]
        g = fixed[fixed["model"] == model].sort_values("level")
        for column, colour, name in columns:
            ax.plot(
                g["level"],
                g[column],
                line,
                marker=marker,
                ms=3,
                color=colour,
                label=f"{name} ({model})",
            )
    ax.set_xlabel("level")
    ax.set_ylabel("reader accuracy")
    ax.set_title("fixed-level reader accuracy, test", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out / "swap_fixed_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---- supplementary grid / tolerance counts --------------------------------------------
def supplementary(
    swap_tables: tuple[pd.DataFrame, pd.DataFrame],
    ref_tables: tuple[pd.DataFrame, pd.DataFrame],
    ref_sweep: pd.DataFrame,
    n_levels: int,
) -> pd.DataFrame:
    """Full-grid sweeps of both models (model = "swap" / "reference"); the
    reference is recomputed from its tables and must reproduce its main sweep."""
    frames = {
        name: sweep_frame(a, w, "test", n_levels, thetas=FULL_THETAS)
        for name, (a, w) in (("swap", swap_tables), ("reference", ref_tables))
    }
    check = frames["reference"].assign(theta=frames["reference"]["theta"].round(2))
    ref = ref_sweep.assign(theta=ref_sweep["theta"].round(2))
    merged = ref.merge(check, on=["method", "theta"], suffixes=("", "_re"))
    require(len(merged) == len(ref), "reference main grid not inside the full grid")
    for column in CHECK_COLUMNS:
        same = (merged[column] - merged[f"{column}_re"]).abs().max() < 1e-9
        require(bool(same), f"recomputed reference sweep differs: {column}")
    return pd.concat(
        [f.assign(model=name) for name, f in frames.items()], ignore_index=True
    )


def tolerance_counts(output_root: Path) -> dict[str, dict[str, int]]:
    """Per role and set: records, valid records, replies where trailing "." were
    stripped (plan §26.1). Raw replies stay in the records."""
    counts: dict[str, dict[str, int]] = {}
    for sub in ("", EXT_DIR):
        for role in TOLERANCE_ROLE_DIRS:
            base = output_root / sub / role if sub else output_root / role
            records = [read_json(p) for p in sorted(base.rglob("*.json"))]
            counts[f"{sub or 'answerable'}/{role}"] = {
                "records": len(records),
                "valid": sum(r.get("status") == "valid" for r in records),
                "trailing_period_stripped": sum(
                    bool(r.get("trailing_period_stripped")) for r in records
                ),
            }
    return counts


# ---- stage ---------------------------------------------------------------------------------
def reference_tables(
    ctx: Context, ids: Mapping[str, Sequence[str]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    swap = ctx.model_swap or {}
    root = resolve_path(ctx.paths.repo_root, str(swap["reference_output_root"]))
    paths = [root / p for p in (REFERENCE_SWEEP, REFERENCE_ANSWERABLE, REFERENCE_WRONG)]
    for path in paths:
        require(path.exists(), f"reference file {path} missing")
    sweep = pd.read_csv(paths[0])
    answerable, wrong = read_id_csv(paths[1]), read_id_csv(paths[2])
    require(same_ids(answerable, ids["answerable"]), "reference answerable set differs")
    require(same_ids(wrong, ids["reader_wrong"]), "reference reader-wrong set differs")
    return sweep, answerable, wrong


def sweep_swap(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("sweep_swap")
    swap = ctx.model_swap
    require(swap is not None, "config has no `model_swap` block (plan §26)")
    assert swap is not None
    require(ctx.skip_v1arm, "pass --skip-v1arm: the model-swap run has no v1 arm")
    require(
        bool(((ctx.state.get("infer") or {}).get("test") or {}).get("complete")),
        "test inference must be complete",
    )
    require(
        bool((ctx.state.get("infer_test_wrong") or {}).get("done")),
        "run `infer-test-wrong` first",
    )
    provenance = check_provenance(ctx)
    ids = set_ids(ctx)
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] model-swap sweep on {len(ids['answerable'])} / "
            f"{len(ids['reader_wrong'])} test questions"
        )
        return {}
    ref_sweep, ref_answerable, ref_wrong = reference_tables(ctx, ids)
    answerable = without_v1_columns(full_table(ctx, "test"))
    require(same_ids(answerable, ids["answerable"]), "answerable table != test subset")
    table_to_csv(answerable, ctx.paths.out(REFERENCE_ANSWERABLE))
    questions = wrong_questions(ctx, "test")
    wrong = run_tables(
        ctx, questions, ctx.paths.out(EXT_DIR), True, cache_only=True, stage=ctx.stage
    )
    require(same_ids(wrong, ids["reader_wrong"]), "reader-wrong table != test set")
    out = ctx.paths.out(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    frame = sweep_frame(answerable, wrong, "test", ctx.n_levels)
    frame.to_csv(out / "test_sweep.csv", index=False)
    model, ref_model = ctx.settings.name, str(swap["reference_model"])
    combined = side_by_side(frame, ref_sweep, model, ref_model)
    combined.to_csv(out / "test_sweep_vs_reference.csv", index=False)
    fixed = pd.concat(
        [
            fixed_frame(answerable, wrong, model),
            fixed_frame(ref_answerable, ref_wrong, ref_model),
        ],
        ignore_index=True,
    )
    fixed.to_csv(out / "fixed_curve.csv", index=False)
    figure_paths = figures(combined, fixed, (model, ref_model), out / "figures")
    supp = supplementary(
        (answerable, wrong), (ref_answerable, ref_wrong), ref_sweep, ctx.n_levels
    )
    supp_out = out / SUPP_DIR
    supp_out.mkdir(parents=True, exist_ok=True)
    supp_frame = side_by_side(
        supp[supp["model"] == "swap"].drop(columns="model"),
        supp[supp["model"] == "reference"].drop(columns="model"),
        model,
        ref_model,
    )
    supp_frame.to_csv(supp_out / "test_sweep_full_vs_reference.csv", index=False)
    supp_paths = figures(
        supp_frame, None, (model, ref_model), supp_out / "figures", "supp", SUPP_TITLE
    )
    ceiling = fixed[(fixed["model"] == model) & (fixed["level"] == ctx.n_levels)]
    meta = {
        "label": LABEL,
        "time": utc_now(),
        "model": model,
        "digest": ctx.settings.digest,
        "reference_model": ref_model,
        "reference_output_root": swap["reference_output_root"],
        "thetas": list(SWEEP_THETAS),
        "methods": list(SWEEP_METHODS),
        "k": SWEEP_K,
        "firing_rule": "score >= θ at some level 1..20; forced L20 is not a firing",
        "quartiles": "numpy percentile 25/50/75, linear interpolation",
        "sizes": {
            "answerable_questions": len(ids["answerable"]),
            "answerable_images": int(answerable["group_id"].nunique()),
            "reader_wrong_questions": len(ids["reader_wrong"]),
        },
        "l20_reader_accuracy_answerable": float(
            ceiling["answerable_reader_accuracy"].iloc[0]
        ),
        "provenance_time": provenance["time"],
        "figures": [p.name for p in figure_paths],
        "supplementary": {
            "label": SUPP_LABEL,
            "thetas": list(FULL_THETAS),
            "csv": f"{SUPP_DIR}/test_sweep_full_vs_reference.csv",
            "figures": [p.name for p in supp_paths],
        },
        "parse_tolerance": {
            "roles": sorted(ctx.settings.trailing_period_roles),
            "counts": tolerance_counts(ctx.paths.output_root),
        },
    }
    atomic_write_json(out / "sweep_meta.json", meta)
    ctx.log(
        f"sweep-swap (DESCRIPTIVE, plan §26) written: {meta['sizes']}, "
        f"{model} L20 reader accuracy {meta['l20_reader_accuracy_answerable']:.3f}"
    )
    return meta
