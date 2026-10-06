"""Markdown report assembly for a ROBOVIS2027 run (no model calls).

Every section builder returns a list of markdown lines. Missing inputs produce a
visible ``_not available: ..._`` line; no number is ever filled in or guessed.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from milestone1.robovis_eval import DEFAULT_Q_METRIC, q_output_dir
from milestone1.stop_policy import EXPLORATORY_LABEL, MAIN_LABEL, fixed_method_name

FLOAT_DIGITS = 3
HEADING_DEMOTE = 2  # included docs: "# x" -> "### x"
ORACLE_STATUSES = ("defined", "unstable_tail", "zero_recovery", "invalid_levels")
NA_TEXT = "n/a"
STATUS_SHORT = {
    MAIN_LABEL: "main",
    EXPLORATORY_LABEL: "exploratory*",
    "baseline": "baseline",
}
EXPLORATORY_FOOTNOTE = f"\\* exploratory = {EXPLORATORY_LABEL}."
STATUS_ORDER = {MAIN_LABEL: 0, EXPLORATORY_LABEL: 1, "baseline": 2}
STAGE_PROFILE_COLUMNS = ("scale", "n_valid", "R50", "R100", "mR50_official", "relsg_gt",
                         "node_recall_gt", "trip_recall_gt")  # fmt: skip
METHOD_COLUMNS = ("method", "status", "n_images", "mean_stop_t", "q_stop", "mR50_stop",
                  "mean_residual", "oracle_mae", "oracle_n", "forced_terminal_rate",
                  "inference_failure_rate", "eq_fixed_q")  # fmt: skip
COMPARE_COLUMNS = ("path", "q_metric", "condition", "n_valid", "frac_ge_theta_t01",
                   "frac_ge_theta_t10", "frac_ge_theta_t20", "rho_t", "rho_gain_left",
                   "p_gain_left", "auc_score", "auc_ge_theta")  # fmt: skip
PAIRWISE_KEYS = (
    "proposed_minus_observed_only",
    "benefit_gated_minus_benefit_gated_obs",
)
MISSING_C_OUTCOMES_OK = ("ok", "empty_graph")
# (category, regex) checked in order against the first violation of an error.
ERROR_CAUSES = (
    ("upstream_checks_failure", re.compile(r"upstream checks failure")),
    ("json_parse", re.compile(r"\bjson\b")),
    ("text_too_long", re.compile(r"longer than")),
    ("anchor_not_in_graph", re.compile(r"not in graph")),
    ("enum_invalid", re.compile(r"invalid")),
    ("type_or_missing_field", re.compile(r"must be|missing")),
    ("transport", re.compile(r"timeout|connection|http|refused", re.IGNORECASE)),
)
OTHER_CAUSE = "other"


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------


def na(what: str) -> str:
    return f"_not available: {what}_"


def fmt(value: Any) -> str:
    if value is None:
        return NA_TEXT
    if isinstance(value, bool | int):
        return str(value)
    if isinstance(value, float):
        return NA_TEXT if math.isnan(value) else f"{value:.{FLOAT_DIGITS}f}"
    return str(value).replace("|", "\\|")


def demote_headings(lines: Iterable[str], levels: int = HEADING_DEMOTE) -> list[str]:
    """Push markdown headings down so an included document nests under ours."""
    return [("#" * levels + ln) if ln.startswith("#") else ln for ln in lines]


def md_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> list[str]:
    """GitHub markdown table (no tabulate dependency)."""
    if not rows:
        return [na("no rows")]
    out = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    out += ["| " + " | ".join(fmt(r.get(c)) for c in columns) + " |" for r in rows]
    return out


def df_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(k): (None if pd.isna(v) else v) for k, v in r.items()}
        for r in df.to_dict("records")
    ]


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def rel(path: Path, base: Path) -> str:
    return os.path.relpath(path, base)


def image_line(path: Path, base: Path, caption: str) -> str:
    if not path.exists():
        return na(f"{caption} ({rel(path, base)})")
    return f"![{caption}]({rel(path, base)})"


def eval_dir(run_dir: Path, path: str, q_metric: str) -> Path:
    return q_output_dir(run_dir / "eval" / path, q_metric)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def section_manifest(output_root: Path, run_dir: Path) -> list[str]:
    lines = ["## Run manifest", ""]
    pre = read_json(output_root / "preflight.json")
    if pre is not None:
        repo, cfg = pre.get("repo", {}), pre.get("config", {})
        gpus = pre.get("hardware", {}).get("gpus") or []
        envs = pre.get("envs", {})
        lines += [
            f"Source: `preflight.json` (generated {pre.get('generated_at', NA_TEXT)}).",
            "",
            f"- repo: commit `{repo.get('commit', NA_TEXT)}` on `{repo.get('branch', NA_TEXT)}`,"
            f" {len(repo.get('dirty') or [])} dirty paths",
            f"- config: `{cfg.get('path', NA_TEXT)}` sha256 `{cfg.get('sha256', NA_TEXT)}`",
            "- GPU: " + (", ".join(str(g.get("name")) for g in gpus) or NA_TEXT),
        ]
        lines += [
            f"- env `{name}`: python {e.get('python')}, torch {e.get('torch')}, CUDA {e.get('cuda')}"
            for name, e in envs.items()
            if isinstance(e, dict)
        ]
        for key in ("ollama", "budget"):
            if key in pre:
                lines.append(
                    f"- {key}: `{json.dumps(pre[key], ensure_ascii=False)[:400]}`"
                )
    else:
        lines.append(na("outputs/robovis2027/preflight.json; using the run manifest"))
    man = read_json(run_dir / "manifest.json")
    lines.append("")
    if man is None:
        lines.append(na(f"{rel(run_dir / 'manifest.json', run_dir)}"))
        return lines
    ckpt, code = man.get("checkpoint", {}), man.get("code", {})
    lines += [
        f"PGSG run manifest (`manifest.json`, script `{man.get('script', NA_TEXT)}`):",
        "",
        f"- split `{man.get('split', NA_TEXT)}`, {len(man.get('image_ids') or [])} images",
        f"- checkpoint `{ckpt.get('checkpoint', NA_TEXT)}` sha256 `{ckpt.get('sha256', NA_TEXT)}`",
        f"- code at inference: `{code.get('git_head', NA_TEXT)}`",
        f"- config sha256 `{man.get('robovis_config_sha256', NA_TEXT)}`",
    ]
    return lines


def section_dataset(selection_manifest: Path) -> list[str]:
    lines = ["## Dataset", ""]
    sel = read_json(selection_manifest)
    if sel is None:
        return lines + [na(str(selection_manifest))]
    counts, crit, order = (
        sel.get("counts", {}),
        sel.get("criteria", {}),
        sel.get("ordering", {}),
    )
    src = sel.get("source", {})
    lines += [
        f"- source: VG-150 `{src.get('split', NA_TEXT)}` split ({src.get('n_images_in_split', NA_TEXT)} images)",
        f"- eligibility: >= {crit.get('min_gt_objects')} GT objects and >= {crit.get('min_gt_relations')} GT relations;"
        f" eligible {counts.get('eligible', NA_TEXT)}, excluded {counts.get('excluded_by_reason', {})}",
        f"- ordering: {order.get('procedure', NA_TEXT)} (seed {order.get('seed', NA_TEXT)});"
        f" test = `{order.get('test')}`, dev = `{order.get('dev')}`",
        f"- counts: test {counts.get('test', NA_TEXT)}, dev {counts.get('dev', NA_TEXT)}",
        f"- overlap with the MIRU 100 images: `{json.dumps(sel.get('overlap_with_miru_100', {}))}`",
    ]
    for item in sel.get("limitations") or []:
        lines.append(f"- limitation: {item}")
    return lines


def _oracle_lines(
    run_dir: Path, paths: Sequence[str], q_metrics: Sequence[str]
) -> list[str]:
    rows = []
    for m in q_metrics:
        src = next((eval_dir(run_dir, p, m) for p in paths
                    if (eval_dir(run_dir, p, m) / "oracle.csv").exists()), None)  # fmt: skip
        if src is None:
            rows.append({"q_metric": m, "note": na("oracle.csv")})
            continue
        ids = read_json(src / "eval_set_ids.json") or {}
        oracle = pd.read_csv(src / "oracle.csv")
        defined = oracle[oracle["oracle_status"] == "defined"]
        n_last = int((defined["oracle_t"] == ids.get("num_levels", -1)).sum())
        counts = ids.get("oracle_status_counts")
        # Statuses absent from the counts occurred 0 times in a known eval set.
        status_counts = (
            {k: int(counts.get(k, 0)) for k in ORACLE_STATUSES}
            if counts is not None
            else {}
        )
        rows.append({"q_metric": m, "n_eval": ids.get("n_eval"), **status_counts,
                     "defined_at_last_level": n_last, "note": rel(src, run_dir)})  # fmt: skip
    cols = ["q_metric", "n_eval", "defined", "unstable_tail", "zero_recovery",
            "invalid_levels", "defined_at_last_level", "note"]  # fmt: skip
    return ["Relative-stable oracle status (eval set):", "", *md_table(rows, cols)]


def section_recall_profile(
    run_dir: Path, paths: Sequence[str], q_metrics: Sequence[str]
) -> list[str]:
    lines = ["## PGSG recall profile", ""]
    summ = read_json(run_dir / "stage_summary.json")
    if summ is None:
        lines.append(na("stage_summary.json"))
    else:
        levels = summ.get("levels", {})
        rows = [{"t": int(t), **{c: v.get(c) for c in STAGE_PROFILE_COLUMNS}}
                for t, v in sorted(levels.items(), key=lambda kv: int(kv[0]))]  # fmt: skip
        lines += [f"Mean over {summ.get('n_images', NA_TEXT)} images per level"
                  " (`stage_summary.json`; mR50 = official /50.001 convention).", ""]  # fmt: skip
        lines += md_table(rows, ["t", *STAGE_PROFILE_COLUMNS])
    return lines + ["", *_oracle_lines(run_dir, paths, q_metrics)]


def section_pilot(run_dir: Path) -> list[str]:
    path = run_dir / "pilot_report.md"
    if not path.exists():
        return ["## Pilot report", "", na("pilot_report.md")]
    body = demote_headings(path.read_text().splitlines())
    return ["## Pilot report (verbatim, headings demoted)", "", *body]


def _ordered_methods(summary: pd.DataFrame, num_levels: int) -> pd.DataFrame:
    adaptive = summary[summary["status"] != "baseline"].copy()
    adaptive["_o"] = adaptive["status"].map(STATUS_ORDER).fillna(len(STATUS_ORDER))
    adaptive = adaptive.sort_values(["_o", "method"]).drop(columns="_o")
    final = summary[summary["method"] == fixed_method_name(num_levels)]
    return pd.concat([adaptive, final], ignore_index=True)


def _ci_text(stat: Mapping[str, float] | None) -> str:
    if not stat:
        return NA_TEXT
    return f"{fmt(stat['mean'])} [{fmt(stat['ci_low'])}, {fmt(stat['ci_high'])}]"


def _replay_coverage(edir: Path) -> str | None:
    path = edir / "c_sequences.csv"
    if not path.exists():
        return None
    c = pd.read_csv(path).drop_duplicates(["image_id", "t"])
    missing = float((~c["outcome"].isin(MISSING_C_OUTCOMES_OK)).mean())
    return f"replay coverage: {1 - missing:.1%} of states have a C_t ({missing:.1%} missing)"


def section_methods(run_dir: Path, paths: Sequence[str], q_metrics: Sequence[str],
                    num_levels: int) -> list[str]:  # fmt: skip
    lines = ["## Stop-policy results per path", "", EXPLORATORY_FOOTNOTE]
    for path in paths:
        for m in q_metrics:
            edir = eval_dir(run_dir, path, m)
            lines += ["", f"### {path} (q = {m})", ""]
            summary_path = edir / "summary.csv"
            if not summary_path.exists():
                lines.append(na(rel(summary_path, run_dir)))
                continue
            summary = _ordered_methods(pd.read_csv(summary_path), num_levels)
            boot = read_json(edir / "bootstrap.json") or {}
            rows = []
            for r in df_rows(summary):
                stat = (
                    boot.get("methods", {}).get(r["method"], {}).get("q_minus_eq_fixed")
                )
                rows.append({**r, "status": STATUS_SHORT.get(r["status"], r["status"]),
                             "q_stop - eq_fixed_q [95% CI]": _ci_text(stat)})  # fmt: skip
            lines += md_table(rows, [*METHOD_COLUMNS, "q_stop - eq_fixed_q [95% CI]"])
            lines.append("")
            for key in PAIRWISE_KEYS:
                stat = boot.get("pairwise", {}).get(key, {}).get("q_stop")
                lines.append(f"- {key} (q_stop): {_ci_text(stat)}")
            coverage = _replay_coverage(edir)
            lines.append(f"- {coverage}" if coverage else "- " + na("c_sequences.csv"))
            lines += ["", image_line(edir / "figures" / "fig2_fixed_curve.png", run_dir, f"Fig 2 {path} {m}"),
                      image_line(edir / "figures" / "fig3_examples.png", run_dir, f"Fig 3 {path} {m}"),
                      image_line(edir / "signal_by_level.png", run_dir, f"signal by level {path} {m}")]  # fmt: skip
    return lines


def section_signal_compare(run_dir: Path) -> list[str]:
    path = run_dir / "eval" / "signal_compare.csv"
    lines = ["## Signal diagnostic across paths", ""]
    if not path.exists():
        return lines + [na(rel(path, run_dir))]
    df = pd.read_csv(path)
    if "q_metric" not in df.columns:
        df.insert(1, "q_metric", DEFAULT_Q_METRIC)
    df = df.sort_values(["q_metric", "path", "condition"])
    lines += [
        "AUC = predicting gain_left > 0 on qmax > 0 states; rho over pooled states.",
        "",
    ]
    return lines + md_table(
        df_rows(df), [c for c in COMPARE_COLUMNS if c in df.columns]
    )


def classify_error(error: str | None) -> str:
    first = (error or "").split(";")[0]
    for name, pattern in ERROR_CAUSES:
        if pattern.search(first):
            return name
    return OTHER_CAUSE


def _usage_rows(records: Iterable[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        groups[(r.get("stage"), r.get("path"), r.get("model"))].append(r)
    rows, causes = [], []
    for (stage, path, model), recs in sorted(
        groups.items(), key=lambda kv: tuple(map(str, kv[0]))
    ):
        live = [r for r in recs if not r.get("cached")]
        lat = [float(r["elapsed_s"]) for r in live if r.get("elapsed_s") is not None]
        fails = [r for r in recs if r.get("failure")]
        rows.append({"stage": stage, "path": path, "model": model, "calls": len(recs),
                     "cached": len(recs) - len(live), "failures": len(fails),
                     "failure_rate": len(fails) / len(recs) if recs else math.nan,
                     "median_latency_s": statistics.median(lat) if lat else None,
                     "prompt_tokens": sum(int(r.get("prompt_eval_count") or 0) for r in recs),
                     "output_tokens": sum(int(r.get("eval_count") or 0) for r in recs)})  # fmt: skip
        for cause, n in Counter(
            classify_error(r.get("error")) for r in fails
        ).most_common():
            causes.append({"stage": stage, "path": path, "cause": cause, "n": n})
    return rows, causes


def section_usage(run_dir: Path) -> list[str]:
    lines = ["## Model calls, failures and latency", ""]
    path = run_dir / "usage.jsonl"
    if path.exists():
        records = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        rows, causes = _usage_rows(records)
        lines += ["From `usage.jsonl` (latency over non-cached calls):", ""]
        lines += md_table(rows, list(rows[0]) if rows else ["stage"])
        lines += ["", "Failure causes (first violation of each failed call):", ""]
        lines += (
            md_table(causes, ["stage", "path", "cause", "n"]) if causes else ["- none"]
        )
    else:
        lines.append(na("usage.jsonl"))
    lines += [
        "",
        "Per-variant check/judgment summaries (`summaries/checks_summary_*.json`):",
        "",
    ]
    files = sorted((run_dir / "summaries").glob("checks_summary_*.json"))
    if not files:
        return lines + [na("summaries/checks_summary_*.json")]
    rows = []
    for f in files:
        d = (read_json(f) or {}).get("by_level", {}).get("all", {})
        ch, de = d.get("checks", {}), d.get("decisions", {})
        rows.append({"variant": f.stem.removeprefix("checks_summary_"),
                     "check_states": ch.get("n_states"), "check_failures": ch.get("n_failures"),
                     "mean_hypotheses": ch.get("mean_hypotheses"),
                     "region_null_rate": ch.get("region_anchor_null_rate"),
                     "judge_failures": de.get("n_failures"), "items": de.get("n_items"),
                     "frac_unresolved": de.get("status", {}).get("fractions", {}).get("unresolved"),
                     "frac_low_res": de.get("limitation", {}).get("fractions", {}).get("low_resolution"),
                     "frac_unknown": de.get("limitation", {}).get("fractions", {}).get("unknown")})  # fmt: skip
    return lines + md_table(rows, list(rows[0]))


def section_notes(notes: Path | None) -> list[str]:
    lines = ["## Decisions and deviations from the plan", ""]
    if notes is None or not notes.exists():
        return lines + [na("report_notes.md")]
    return lines + [
        f"(verbatim from `{notes.name}`, headings demoted)",
        "",
        *demote_headings(notes.read_text().splitlines()),
    ]


def section_limitations(
    run_dir: Path, paths: Sequence[str], q_metrics: Sequence[str]
) -> list[str]:
    lines = ["## Not done / limitations", "",
             "- TODO: list analyses that were planned but not run.",
             "- TODO: state the single-seed / single-run nature of PGSG sampling and the LLM calls.",
             "- Oracle and residual use VG GT, which is incomplete (missing GT is counted as error)."]  # fmt: skip
    for path in paths:
        for m in q_metrics:
            cov = _replay_coverage(eval_dir(run_dir, path, m))
            if cov:
                lines.append(f"- {path} (q = {m}): {cov}.")
    return lines


def figure1_lines(output_root: Path, run_dir: Path) -> list[str]:
    fig = output_root / "figures" / "fig1_pipeline.png"
    return ["## Fig 1: pipeline", "", image_line(fig, run_dir, "Fig 1 pipeline")]


def build_report(
    *,
    run_id: str,
    output_root: Path,
    selection_manifest: Path,
    paths: Sequence[str],
    q_metrics: Sequence[str],
    num_levels: int,
    notes: Path | None,
    header_lines: Sequence[str] = (),
) -> str:
    """Assemble the full report markdown for one run directory."""
    run_dir = output_root / run_id
    sections: list[list[str]] = [
        [f"# ROBOVIS2027 report: {run_id}", "", *header_lines],
        section_manifest(output_root, run_dir),
        section_dataset(selection_manifest),
        figure1_lines(output_root, run_dir),
        section_recall_profile(run_dir, paths, q_metrics),
        section_methods(run_dir, paths, q_metrics, num_levels),
        section_signal_compare(run_dir),
        section_usage(run_dir),
        section_pilot(run_dir),
        section_notes(notes),
        section_limitations(run_dir, paths, q_metrics),
    ]
    return "\n".join(line for sec in sections for line in [*sec, ""]).rstrip() + "\n"
