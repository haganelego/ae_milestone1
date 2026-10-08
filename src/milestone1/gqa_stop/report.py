"""report.md, figures and completion.json from on-disk artifacts (plan §13).

Every number is read from data files; stages that have not run are shown as
explicit placeholders and marked ``not_run`` in completion.json.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

from .client import record_tag
from .common import (
    RunPaths,
    atomic_write_json,
    atomic_write_text,
    read_id_csv,
    read_json,
    read_jsonl,
    seeded_hash,
    sha256_file,
    sha256_text,
    utc_now,
)
from .evaluate import hull_value

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

NOT_RUN = "_Not yet run._"
EXAMPLES_PER_CATEGORY = 2
LATE_MARGIN_LEVELS = 5
FIG_DPI = 150
QHASH_PREFIX = 10
STAGE_FILES = {
    "preflight": ["preflight.json", "effective_config.json"],
    "probe": ["probe_summary.json"],
    "prepare": ["schema_mapping.json", "dataset_flow.csv", "scan_summary.json"],
    "screen_dev": ["adoption_dev.json", "screening_repeat.json"],
    "pilot": ["pilot_summary.json"],
    "pilot_calibration": ["pilot_calibration.json"],
    "infer_dev": ["level_table_dev.csv"],
    "freeze": ["frozen_policy.json", "dev_threshold_grid.csv"],
    "screen_test": ["adoption_test.json"],
    "infer_test": ["level_table_test.csv"],
    "evaluate": [
        "stops.csv",
        "method_summary.csv",
        "paired_comparisons.csv",
        "bootstrap_summary.json",
    ],
}


def _load(paths: RunPaths, name: str) -> Any:
    path = paths.out(name)
    if not path.exists():
        return None
    if name.endswith(".csv"):
        return read_id_csv(path)
    return read_json(path)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def md_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    cols = [c for c in columns if c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in frame[cols].itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


# ---- figures ------------------------------------------------------------------
def figure_tradeoff(summary: pd.DataFrame, primary: str, out: Path) -> None:
    fixed = summary[
        summary["method"].str.startswith("fixed_")
        & ~summary["method"].str.startswith("fixed_dev")
    ]
    curve = {
        int(m.split("_")[1]): a
        for m, a in zip(fixed["method"], fixed["accuracy"], strict=True)
    }
    xs = sorted(curve)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, [curve[x] for x in xs], "o-", color="gray", ms=3, label="Fixed-t")
    grid = [1 + i * 0.1 for i in range(int((xs[-1] - 1) * 10) + 1)]
    ax.plot(
        grid,
        [hull_value(curve, b) for b in grid],
        "--",
        color="black",
        lw=1,
        label="concave hull F(b)",
    )
    for row in summary.itertuples(index=False):
        if "@" not in row.method:
            continue
        name, target = row.method.split("@")
        is_primary = target == primary
        ax.scatter(
            row.mean_level,
            row.accuracy,
            s=50 if is_primary else 25,
            marker="o" if is_primary else "x",
            label=f"{name}@{target}" if is_primary else None,
        )
    ax.scatter(
        [], [], marker="x", color="gray", label="pre-registered secondary targets"
    )
    ax.set_xlabel("MeanLevel")
    ax.set_ylabel("Accuracy (common reader)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)


def figure_stop_hist(
    stops: pd.DataFrame, primary: str, n_levels: int, out: Path
) -> None:
    labels = sorted(m for m in stops["method"].unique() if m.endswith(f"@{primary}"))
    fig, axes = plt.subplots(
        1, max(len(labels), 1), figsize=(4 * max(len(labels), 1), 3), squeeze=False
    )
    for ax, label in zip(axes[0], labels, strict=False):
        values = stops[stops["method"] == label]["stop_level"]
        ax.hist(values, bins=range(1, n_levels + 2), align="left", color="steelblue")
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("stop level")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)


def figure_dev_grid(grid: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, frame in grid.groupby("method"):
        ax.plot(frame["mean_level"], frame["accuracy"], ".-", ms=2, label=method)
    ax.set_xlabel("dev MeanLevel")
    ax.set_ylabel("dev accuracy")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)


# ---- examples -----------------------------------------------------------------
def pick_examples(
    stops: pd.DataFrame, table: pd.DataFrame, primary: str, seed: int
) -> dict[str, list[str]]:
    """Fixed categories; first 2 per category in SHA256(seed|qid) order."""
    wide = stops.pivot_table(
        index="question_id", columns="method", values="stop_level", aggfunc="first"
    )
    ok = stops.pivot_table(
        index="question_id", columns="method", values="correct", aggfunc="first"
    )
    lmin, ldir = f"checklist_min@{primary}", f"direct@{primary}"
    if lmin not in wide or ldir not in wide:
        return {}
    first = table[table["reader_correct"]].groupby("question_id")["level"].min()
    cats: dict[str, list[str]] = {
        "min_early_and_correct": [],
        "direct_early_wrong": [],
        "min_too_late": [],
        "both_wrong": [],
    }
    for qid in sorted(wide.index, key=lambda q: seeded_hash(seed, str(q))):
        smin, sdir = wide.loc[qid, lmin], wide.loc[qid, ldir]
        cmin, cdir = bool(ok.loc[qid, lmin]), bool(ok.loc[qid, ldir])
        checks = {
            "min_early_and_correct": smin < sdir and cmin,
            "direct_early_wrong": sdir < smin and not cdir,
            "min_too_late": qid in first and smin - first[qid] >= LATE_MARGIN_LEVELS,
            "both_wrong": not cmin and not cdir,
        }
        for cat, hit in checks.items():
            if hit and len(cats[cat]) < EXAMPLES_PER_CATEGORY:
                cats[cat].append(str(qid))
    return cats


def example_block(
    paths: RunPaths, qid: str, table: pd.DataFrame, stops: pd.DataFrame, tag: str
) -> str:
    rows = table[table["question_id"] == qid].sort_values("level")
    plan_path = paths.out(f"plans/{tag}/{qid}.json")
    plan = read_json(plan_path).get("parsed") if plan_path.exists() else None
    stop_rows = stops[stops["question_id"] == qid]
    lines = [
        f"- qhash `{sha256_text(qid)[:QHASH_PREFIX]}`: {rows.iloc[0]['question']}"
        if "question" in rows
        else f"- qhash `{sha256_text(qid)[:QHASH_PREFIX]}`"
    ]
    items = plan["items"] if plan else []
    lines.append(f"  - checklist: {json.dumps([i['question'] for i in items])}")
    scores = "; ".join(
        f"L{int(r.level)} d={_fmt(r.direct_score, 2)} min={_fmt(r.checklist_min, 2)} "
        f"mean={_fmt(r.checklist_mean, 2)} ans={r.reader_answer}"
        for r in rows.itertuples(index=False)
    )
    lines.append(f"  - scores: {scores}")
    for r in stop_rows.itertuples(index=False):
        if "@" in r.method and not r.method.startswith("fixed"):
            lines.append(
                f"  - {r.method}: stop L{r.stop_level}, answer={r.reader_answer}, "
                f"correct={r.correct}"
            )
    return "\n".join(lines)


# ---- report -------------------------------------------------------------------
def stage_status(paths: RunPaths, state: Mapping[str, Any]) -> dict[str, Any]:
    screening = state.get("screening") or {}
    infer = state.get("infer") or {}
    done = {
        "preflight": bool(state.get("preflight")),
        "probe": bool(state.get("probe")),
        "prepare": bool(state.get("prepare")),
        "screen_dev": bool(screening.get("dev")),
        "pilot": bool(state.get("pilot")),
        "pilot_calibration": paths.out("pilot_calibration.json").exists(),
        "infer_dev": bool(infer.get("dev", {}).get("complete")),
        "freeze": paths.out("frozen_policy.json").exists(),
        "screen_test": bool(screening.get("test")),
        "infer_test": bool(infer.get("test", {}).get("complete")),
        "evaluate": bool(state.get("evaluate")),
    }
    result = {}
    for stage, ok in done.items():
        files = {
            f: sha256_file(paths.out(f))
            for f in STAGE_FILES[stage]
            if paths.out(f).exists()
        }
        started = stage in ("infer_dev", "infer_test") and bool(
            infer.get(stage.split("_")[1], {}).get("started")
        )
        status = "completed" if ok else ("incomplete" if started else "not_run")
        result[stage] = {"status": status, "artifacts_sha256": files}
    return result


def build_report(cfg: Mapping[str, Any], paths: RunPaths) -> str:
    state = _load(paths, "state.json") or {}
    stages = stage_status(paths, state)
    figures = paths.out("figures")
    parts = [
        "# GQA question-decomposition early stop: experiment report",
        f"Generated {utc_now()} from artifacts in `{paths.output_root}`. Spec v"
        f"{cfg.get('spec_version')} (plan §0-§16).",
        "## 1. Scope",
        "Same-view resolution recovery over 20 levels on GQA val_balanced questions "
        "(research dev/test split by image; not official GQA test). Stop scores are "
        "self-reported confidences, not probabilities. Unlike DeCC (EMNLP 2024), which "
        "checks consistency between direct and reconstructed answers, this uses "
        "sub-question confidences to choose a stop level; DeCC-style stopping is not "
        "evaluated. 'No training' means no parameter updates; dev thresholds were "
        "selected. Levels are not physical distances or compute units.",
    ]
    parts += data_section(paths)
    parts += model_section(cfg, paths)
    parts += pilot_section(paths)
    parts += results_section(cfg, paths, figures)
    parts += limitations_section(paths)
    parts += [
        "## 8. Re-run commands",
        "```bash\n"
        + "\n".join(
            f"uv run python -m milestone1.gqa_stop.cli {c} --config configs/gqa_stop.json"
            for c in (
                "preflight",
                "probe",
                "prepare",
                "screen --split dev",
                "pilot --split dev --levels 1,5,10,15,20 --calibration",
                "infer --split dev --levels all",
                "freeze",
                "screen --split test",
                "infer --split test --levels all",
                "evaluate",
                "report",
            )
        )
        + "\n```",
        "## 9. Completion",
        md_table(
            pd.DataFrame(
                [{"stage": k, "status": v["status"]} for k, v in stages.items()]
            ),
            ["stage", "status"],
        ),
    ]
    text = "\n\n".join(parts) + "\n"
    atomic_write_text(paths.out("report.md"), text)
    atomic_write_json(
        paths.out("completion.json"),
        {
            "time": utc_now(),
            "stages": stages,
            "complete": all(v["status"] == "completed" for v in stages.values()),
            "note": "hypothesis support is not a completion criterion (plan §13)",
            "gate_override": state.get("gate_override"),
        },
    )
    return text


def data_section(paths: RunPaths) -> list[str]:
    mapping = _load(paths, "schema_mapping.json")
    out = ["## 2. Data flow"]
    if mapping is None:
        return [*out, NOT_RUN]
    out.append(
        f"Total questions {mapping['total_questions']}; eligible "
        f"{mapping['eligible_questions']} on {mapping['eligible_images']} images."
    )
    reasons = pd.DataFrame(
        sorted(mapping["exclusion_counts"].items()), columns=["reason", "count"]
    )
    out.append(md_table(reasons, ["reason", "count"]))
    results = read_jsonl(paths.data("selection_results.jsonl"))
    selected = read_jsonl(paths.data("selected_manifest.jsonl"))
    if not results:
        return [*out, "Screening: " + NOT_RUN]
    frame = pd.DataFrame(results)
    flow = (
        frame.groupby(["split", "outcome"]).size().unstack(fill_value=0).reset_index()
    )
    out.append(md_table(flow, list(flow.columns)))
    if selected:
        sel = pd.DataFrame(selected)
        dist = (
            sel.groupby(["split", "attribute", "relate_count"])
            .size()
            .reset_index(name="n")
        )
        out.append(md_table(dist, ["split", "attribute", "relate_count", "n"]))
        out.append(
            "Design sample balanced by attribute and target-identification "
            "difficulty (§16 A2); no claim about the natural GQA distribution."
        )
    return out


def model_section(cfg: Mapping[str, Any], paths: RunPaths) -> list[str]:
    eff = _load(paths, "effective_config.json")
    hashes = _load(paths, "prompt_hashes.json")
    model = (eff or {}).get("effective", cfg).get("model", {})
    return [
        "## 3. Model, prompts, images, decoding",
        f"Model `{model.get('name')}` digest `{model.get('digest')}`; temperature "
        f"{model.get('temperature')}, seed {model.get('seed')}, think={model.get('think')}, "
        f"num_ctx {model.get('num_ctx')}, num_predict {model.get('num_predict')}.",
        f"Prompt hashes: `{json.dumps((hashes or {}).get('hashes'))}`.",
        f"Images: {cfg['images']['levels']} levels, s_t = {cfg['images']['scale_min']} x "
        f"20^((t-1)/19), base max side {cfg['images']['base_max_side']} (Lanczos), "
        "bilinear down/up, PNG bytes sent as stored.",
    ]


def pilot_section(paths: RunPaths) -> list[str]:
    out = ["## 4. Probe, pilot and calibration gate"]
    for name in (
        "probe_summary.json",
        "pilot_summary.json",
        "pilot_calibration.json",
        "screening_repeat.json",
    ):
        data = _load(paths, name)
        if data is None:
            out.append(f"`{name}`: {NOT_RUN}")
            continue
        keep = {
            k: v for k, v in data.items() if k not in {"rows", "cell_level_accuracy"}
        }
        out.append(f"`{name}`:\n```json\n{json.dumps(keep, indent=1)[:3000]}\n```")
    return out


def results_section(
    cfg: Mapping[str, Any], paths: RunPaths, figures: Path
) -> list[str]:
    out = ["## 5. Dev thresholds and frozen policy"]
    grid, frozen = (
        _load(paths, "dev_threshold_grid.csv"),
        _load(paths, "frozen_policy.json"),
    )
    if grid is None or frozen is None:
        out.append(NOT_RUN)
    else:
        figure_dev_grid(grid, figures / "dev_threshold_curves.png")
        rows = [
            {
                "target": t,
                "method": m,
                "theta": v.get("theta") if isinstance(v, dict) else f"fixed L{v}",
                "dev_accuracy": v.get("accuracy") if isinstance(v, dict) else None,
                "dev_mean_level": v.get("mean_level") if isinstance(v, dict) else None,
            }
            for t, sel in frozen["selections"].items()
            for m, v in sel.items()
        ]
        out.append(
            md_table(
                pd.DataFrame(rows),
                ["target", "method", "theta", "dev_accuracy", "dev_mean_level"],
            )
        )
        out.append("![dev curves](figures/dev_threshold_curves.png)")
        out.append(
            "Direct's frozen theta may be inf (always Level 20); then the MeanLevel "
            "difference is trivial (§16 A8)."
        )
    out.append("## 6. Test results")
    summary, comps = (
        _load(paths, "method_summary.csv"),
        _load(paths, "paired_comparisons.csv"),
    )
    if summary is None or frozen is None:
        return [*out, NOT_RUN]
    primary = str(frozen["primary_target"])
    main = summary[
        summary["method"].str.endswith(f"@{primary}")
        | summary["method"].isin(["fixed_5", "fixed_10", "fixed_15", "fixed_20"])
    ]
    out.append(
        md_table(
            main,
            [
                "method",
                "theta",
                "accuracy",
                "mean_level",
                "median_level",
                "forced20_rate",
                "never_met_rate",
                "natural_pass20_rate",
                "stop20_rate",
                "status",
                "accuracy_loss_vs_l20",
                "mean_observed_level_reduction",
            ],
        )
    )
    if comps is not None:
        comps = comps[comps["comparison"].str.contains(f"@{primary}", regex=False)]
        out.append(
            "Primary target only; secondary targets are in paired_comparisons.csv."
        )
        out.append(
            md_table(
                comps,
                ["comparison", "metric", "estimate", "ci_low", "ci_high", "verdict"],
            )
        )
    figure_tradeoff(summary, primary, figures / "test_meanlevel_accuracy.png")
    out.append("![tradeoff](figures/test_meanlevel_accuracy.png)")
    out.append(
        "F(b) is the upper bound of level-agnostic randomised fixed policies, not a "
        "learned policy; falling below it is not by itself a failure (§16 A5)."
    )
    stops, table = _load(paths, "stops.csv"), _load(paths, "level_table_test.csv")
    if stops is not None:
        figure_stop_hist(
            stops,
            primary,
            int(cfg["images"]["levels"]),
            figures / "test_stop_level_hist.png",
        )
        out.append("![stops](figures/test_stop_level_hist.png)")
    sub = _load(paths, "subgroup_summary.csv")
    if sub is not None:
        out.append(
            md_table(
                sub, ["group_by", "value", "method", "n", "accuracy", "mean_level"]
            )
        )
    if stops is not None and table is not None:
        out.append("### Examples (first two per category in question-hash order)")
        cats = pick_examples(stops, table, primary, int(cfg["seed"]))
        for cat, qids in cats.items():
            out.append(f"**{cat}**" + ("" if qids else ": none"))
            tag = record_tag(cfg["model"]["name"], cfg["model"].get("digest"))
            out.extend(example_block(paths, q, table, stops, tag) for q in qids)
    return out


def limitations_section(paths: RunPaths) -> list[str]:
    diag = _load(paths, "diagnostics.json")
    errors = read_jsonl(paths.out("errors.jsonl"))
    state = _load(paths, "state.json") or {}
    override = state.get("gate_override")
    return [
        "## 7. Diagnostics and limitations",
        f"Technical failures logged: {len(errors)}.",
        (
            f"**Pilot gate overridden with --force-continue**: {override['failures']}. "
            "Results after this point are exploratory."
            if override
            else "Pilot gates: no override recorded."
        ),
        f"Diagnostics: `{json.dumps(diag) if diag else NOT_RUN}`",
        "Exact match after fixed normalisation (not the official GQA evaluator); surface "
        "variants such as 'light blue' vs 'blue' count as errors. The screened set is "
        "'questions the reader answered correctly at L20 in this run', so L20 accuracy is "
        "1 by construction. GQA/VG may be in the model's pre-training data. With ~200 "
        "test questions the accuracy CI half-width is about 4 pp, so the -5 pp "
        "non-inferiority check is close to inconclusive by design (§16 A8).",
    ]
