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

from milestone1.robovis_eval import (
    AUX_ORACLE_ROLE,
    AUX_ORACLES,
    DEFAULT_Q_METRIC,
    ORACLE_STATUSES,
    OracleParams,
    oracle_profile_from_stage,
    q_output_dir,
)
from milestone1.robovis_judge_quality import LOWRES_ROLE
from milestone1.stop_policy import EXPLORATORY_LABEL, MAIN_LABEL, fixed_method_name

FLOAT_DIGITS = 3
F_COLUMN = "q_stop - F(mean_stop) [95% CI]"
F_TABLE_NOTE = ("F = GT-posterior best fixed mixture on this set (upper concave hull of the "
                "fixed curve; a reference, not a method). eq_fixed_q is the legacy adjacent "
                "two-level mix.")  # fmt: skip
HEADING_DEMOTE = 2  # included docs: "# x" -> "### x"
DEFAULT_ORACLE_RATIO = 0.9  # only used when build_report gets no OracleParams
NA_TEXT = "n/a"
STATUS_SHORT = {
    MAIN_LABEL: "main",
    EXPLORATORY_LABEL: "exploratory*",
    "baseline": "baseline",
}
EXPLORATORY_FOOTNOTE = f"\\* exploratory = {EXPLORATORY_LABEL}."
STATUS_ORDER = {MAIN_LABEL: 0, EXPLORATORY_LABEL: 1, "baseline": 2}
STAGE_PROFILE_COLUMNS = ("scale", "n_valid", "R50", "R100", "mR50_official", "relsg_gt")
# Observed-graph statistics (raw PGSG graph vs cleaned graph G): descriptive only.
GRAPH_STAT_PAIRS = (("R50", "R50_clean"), ("det_prec20", "det_prec20_clean"))
AUX_ORACLE_NOTE = AUX_ORACLE_ROLE
LOWRES_NOTE = LOWRES_ROLE
JUDGE_QUALITY_COLUMNS = ("path", "q_metric", "rho_resolved_t", "rho_resolved_qratio",
                         "auc_sufficient_resolved", "auc_sufficient_C",
                         "rho_lowres_qratio", "auc_sufficient_lowres", "n_sufficient")  # fmt: skip
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
    run_dir: Path, q_metrics: Sequence[str], params: OracleParams, num_levels: int
) -> list[str]:
    """Oracle status per q metric x definition, computed from stage_metrics.csv."""
    stage = run_dir / "stage_metrics.csv"
    if not stage.exists():
        return [na("stage_metrics.csv (oracle profile)")]
    rows: list[dict[str, Any]] = []
    for m in q_metrics:
        try:
            rows += df_rows(oracle_profile_from_stage(stage, m, params, num_levels))
        except ValueError as exc:  # metric column absent / not numeric
            rows.append({"q_metric": m, "definition": na(str(exc))})
    cols = ["q_metric", "definition", "role", "n_eval", *ORACLE_STATUSES,
            "defined_at_last_level", "median_oracle_t"]  # fmt: skip
    return [
        f"Oracle status on the eval set (main = relative_stable, ratio {params.ratio};"
        f" {AUX_ORACLE_ROLE}: monotone_stable = same rule on the isotonic envelope,"
        f" gain_based = first t with max_(u>t) q_u - q_t < {params.gain_delta}):",
        "",
        *md_table(rows, cols),
    ]


def section_recall_profile(
    run_dir: Path, q_metrics: Sequence[str], params: OracleParams, num_levels: int
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
    return lines + ["", *_oracle_lines(run_dir, q_metrics, params, num_levels)]


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


def _aux_mae_cells(row: Mapping[str, Any]) -> dict[str, str]:
    """'MAE (n)' cells for the auxiliary oracle definitions."""
    cells = {}
    for d in AUX_ORACLES:
        mae, n = row.get(f"oracle_mae_{d}"), row.get(f"oracle_n_{d}")
        cells[f"MAE {d} (n)"] = NA_TEXT if n is None else f"{fmt(mae)} ({fmt(n)})"
    return cells


def _replay_coverage(edir: Path) -> str | None:
    path = edir / "c_sequences.csv"
    if not path.exists():
        return None
    c = pd.read_csv(path).drop_duplicates(["image_id", "t"])
    missing = float((~c["outcome"].isin(MISSING_C_OUTCOMES_OK)).mean())
    return f"replay coverage: {1 - missing:.1%} of states have a C_t ({missing:.1%} missing)"


def report_metric_roles(cfg: Mapping[str, Any]) -> dict[str, str]:
    """Agreed evaluation metrics and their role (primary / secondary)."""
    roles = cfg["evaluation"].get("report_q_metrics") or {DEFAULT_Q_METRIC: "primary"}
    return {str(k): str(v) for k, v in roles.items()}


def section_methods(run_dir: Path, paths: Sequence[str], q_metrics: Sequence[str],
                    num_levels: int,
                    metric_roles: Mapping[str, str] | None = None) -> list[str]:  # fmt: skip
    lines = ["## Stop-policy results per path", "", EXPLORATORY_FOOTNOTE]
    for path in paths:
        for m in q_metrics:
            edir = eval_dir(run_dir, path, m)
            role = (metric_roles or {}).get(m)
            lines += ["", f"### {path} (q = {m}{', ' + role if role else ''})", ""]
            summary_path = edir / "summary.csv"
            if not summary_path.exists():
                lines.append(na(rel(summary_path, run_dir)))
                continue
            summary = _ordered_methods(pd.read_csv(summary_path), num_levels)
            boot = read_json(edir / "bootstrap.json") or {}
            rows = []
            aux_rows = []
            for r in df_rows(summary):
                aux_rows.append({"method": r["method"], **_aux_mae_cells(r)})
                stat = (
                    boot.get("methods", {}).get(r["method"], {}).get("q_minus_eq_fixed")
                )
                stat_f = boot.get("methods", {}).get(r["method"], {}).get("q_minus_F")
                rows.append({**r, "status": STATUS_SHORT.get(r["status"], r["status"]),
                             "q_stop - eq_fixed_q [95% CI]": _ci_text(stat),
                             F_COLUMN: _ci_text(stat_f)})  # fmt: skip
            graphs_dir = (
                summary["graphs_dir"].iloc[0] if "graphs_dir" in summary else None
            )
            lines += [f"Graphs replayed: `{graphs_dir or NA_TEXT}`.", ""]
            lines += md_table(
                rows, [*METHOD_COLUMNS, "q_stop - eq_fixed_q [95% CI]", F_COLUMN]
            )
            lines += ["", F_TABLE_NOTE]
            lines.append("")
            for key in PAIRWISE_KEYS:
                stat = boot.get("pairwise", {}).get(key, {}).get("q_stop")
                lines.append(f"- {key} (q_stop): {_ci_text(stat)}")
            coverage = _replay_coverage(edir)
            lines.append(f"- {coverage}" if coverage else "- " + na("c_sequences.csv"))
            aux_cols = [f"MAE {d} (n)" for d in AUX_ORACLES]
            lines += ["", f"Oracle MAE under the auxiliary definitions ({AUX_ORACLE_NOTE}):",
                      "", *md_table(aux_rows, ["method", *aux_cols])]  # fmt: skip
            lines += ["", image_line(edir / "figures" / "fig2_fixed_curve.png", run_dir, f"Fig 2 {path} {m}"),
                      image_line(edir / "figures" / "fig3_examples.png", run_dir, f"Fig 3 {path} {m}"),
                      image_line(edir / "signal_by_level.png", run_dir, f"signal by level {path} {m}")]  # fmt: skip
    return lines


def section_graph_stats(run_dir: Path) -> list[str]:
    """Raw PGSG graph vs cleaned graph G, per level (descriptive, not evaluation)."""
    lines = [
        "## Observed graph G statistics",
        "",
        "Raw PGSG graph vs the cleaned graph G given to the check generator, mean over"
        " valid images per level. Descriptive only; these are not evaluation metrics.",
        "",
    ]
    path = run_dir / "stage_metrics.csv"
    if not path.exists():
        return lines + [na("stage_metrics.csv")]
    df = pd.read_csv(path)
    cols = [c for pair in GRAPH_STAT_PAIRS for c in pair]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return lines + [na(f"stage_metrics.csv columns {missing}")]
    valid = df[df["valid"].astype(str).str.lower().isin(["1", "true"])]
    numeric = valid[["t", *cols]].apply(pd.to_numeric, errors="coerce")
    means = numeric.groupby("t")[cols].mean().reset_index()
    rename = {raw: f"{raw} (raw)" for raw, _ in GRAPH_STAT_PAIRS}
    rename |= {g: f"{raw} (G)" for raw, g in GRAPH_STAT_PAIRS}
    rows = [{("t" if k == "t" else rename[k]): v for k, v in r.items()}
            for r in df_rows(means)]  # fmt: skip
    for r in rows:
        r["t"] = int(r["t"])
    return lines + md_table(rows, ["t", *(rename[c] for c in cols)])


def section_signal_compare(run_dir: Path, q_metrics: Sequence[str]) -> list[str]:
    path = run_dir / "eval" / "signal_compare.csv"
    lines = ["## Signal diagnostic across paths", ""]
    if not path.exists():
        return lines + [na(rel(path, run_dir))]
    df = pd.read_csv(path)
    if "q_metric" not in df.columns:
        df.insert(1, "q_metric", DEFAULT_Q_METRIC)
    df = df[df["q_metric"].isin(q_metrics)]
    order = {m: i for i, m in enumerate(q_metrics)}
    df = df.sort_values(
        ["q_metric", "path", "condition"],
        key=lambda col: col.map(order) if col.name == "q_metric" else col,
    )
    lines += [
        "AUC = predicting gain_left > 0 on qmax > 0 states; rho over pooled states.",
        "",
    ]
    lines += md_table(df_rows(df), [c for c in COMPARE_COLUMNS if c in df.columns])
    return lines + _judge_quality_lines(df)


def _judge_quality_lines(df: pd.DataFrame) -> list[str]:
    """Path-level judge-quality columns (one row per path x q metric)."""
    cols = [c for c in JUDGE_QUALITY_COLUMNS if c in df.columns]
    if len(cols) <= len(("path", "q_metric")):
        return ["", na("judge-quality columns in signal_compare.csv")]
    rows = df[df["condition"] == "proposed"]
    if "n_sufficient" in rows.columns:
        rows = rows.assign(n_sufficient=rows["n_sufficient"].astype("Int64"))
    value_cols = [c for c in cols if c not in ("path", "q_metric")]
    has_values = rows[value_cols].notna().any(axis=1)
    missing = sorted(set(rows.loc[~has_values, "path"]))
    out = [
        "",
        "Judge quality: does `resolved` / `low_resolution` track recovery q_t/qmax?"
        f" sufficient = q_t/qmax >= oracle ratio. `frac_lowres` is a {LOWRES_NOTE}.",
        "",
        *md_table(df_rows(rows[has_values]), cols),
    ]
    if missing:
        out += [
            "",
            na(f"judge-quality columns for {', '.join(missing)} (rerun script 32)"),
        ]
    return out


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
    oracle_params: OracleParams | None = None,
    metric_roles: Mapping[str, str] | None = None,
    header_lines: Sequence[str] = (),
) -> str:
    """Assemble the full report markdown for one run directory."""
    run_dir = output_root / run_id
    oracle_params = oracle_params or OracleParams(DEFAULT_ORACLE_RATIO)
    sections: list[list[str]] = [
        [f"# ROBOVIS2027 report: {run_id}", "", *header_lines],
        section_manifest(output_root, run_dir),
        section_dataset(selection_manifest),
        figure1_lines(output_root, run_dir),
        section_recall_profile(run_dir, q_metrics, oracle_params, num_levels),
        section_graph_stats(run_dir),
        section_methods(run_dir, paths, q_metrics, num_levels, metric_roles),
        section_signal_compare(run_dir, q_metrics),
        section_usage(run_dir),
        section_pilot(run_dir),
        section_notes(notes),
        section_limitations(run_dir, paths, q_metrics),
    ]
    return "\n".join(line for sec in sections for line in [*sec, ""]).rstrip() + "\n"
