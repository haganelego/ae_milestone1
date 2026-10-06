"""Model-free replay of the stop policy over cached run outputs + evaluation.

Reads (under ``input_dir``)::

    stage_metrics.csv
    stage_predicate_hits.csv            (optional, for mR@50)
    graphs/<image_id>/level_XX.json
    checks/<image_id>/level_XX.json
    decisions/<image_id>/level_XX.json  (judgment JSON + "failure": bool)

Writes (under ``out_dir``)::

    c_sequences.csv  stops.csv  per_image.csv  oracle.csv  summary.csv
    bootstrap.json   eval_set_ids.json

No network or model client is touched.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from milestone1.robovis_data import level_name
from milestone1.robovis_eval import (
    PRED_HITS_FILENAME,
    STOPS_COLUMNS,
    build_curves,
    build_eval_set,
    configured_q_metric,
    fixed_stops,
    load_predicate_hits,
    load_stage_metrics,
    oracle_status_counts,
    oracle_table,
    paired_bootstrap,
    per_image_metrics,
    summarize,
)
from milestone1.stop_policy import (
    BENEFIT_GATED,
    BENEFIT_GATED_OBS,
    EXPLORATORY_LABEL,
    MAIN_LABEL,
    OBSERVED_ONLY,
    PROPOSED,
    UNKNOWN_CONSERVATIVE,
    Condition,
    build_conditions,
    fixed_method_name,
    replay_stop,
    score_level_outputs,
)
from milestone1.structured_io import ContractLimits

STAGE_METRICS_FILENAME = "stage_metrics.csv"


@dataclass(frozen=True)
class RunLayout:
    """Sub-directory names inside a run directory."""

    graphs: str = "graphs"
    checks: str = "checks"
    decisions: str = "decisions"


@dataclass(frozen=True)
class EvalParams:
    num_levels: int
    recall_k: int
    q_metric: str
    oracle_ratio: float
    bootstrap_repeats: int
    bootstrap_seed: int
    ci_level: float

    @classmethod
    def from_config(
        cls, cfg: Mapping[str, Any], q_metric: str | None = None
    ) -> EvalParams:
        ev = cfg["evaluation"]
        return cls(
            num_levels=len(cfg["resolution"]["scales"]),
            recall_k=int(ev["recall_k"]),
            q_metric=configured_q_metric(cfg, q_metric),
            oracle_ratio=float(ev["oracle_ratio"]),
            bootstrap_repeats=int(ev["bootstrap_repeats"]),
            bootstrap_seed=int(ev["bootstrap_seed"]),
            ci_level=float(ev["ci_level"]),
        )


def read_json_or_none(path: Path) -> tuple[dict[str, Any] | None, bool]:
    """Return (object, parse_ok). Missing file -> (None, True)."""
    if not path.exists():
        return None, True
    try:
        obj = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, False
    if not isinstance(obj, dict):
        return None, False
    return obj, True


def score_image(
    input_dir: Path,
    image_id: int,
    num_levels: int,
    conditions: Mapping[str, Condition],
    limits: ContractLimits,
    layout: RunLayout,
) -> list[dict[str, Any]]:
    """C_t rows (one per level x condition) for one image."""
    rows: list[dict[str, Any]] = []
    for t in range(1, num_levels + 1):
        fname = f"{level_name(t)}.json"
        graph, g_ok = read_json_or_none(
            input_dir / layout.graphs / str(image_id) / fname
        )
        checks, c_ok = read_json_or_none(
            input_dir / layout.checks / str(image_id) / fname
        )
        decision, d_ok = read_json_or_none(
            input_dir / layout.decisions / str(image_id) / fname
        )
        if not g_ok:
            scores, outcome = dict.fromkeys(conditions), "corrupt_graph"
        elif not c_ok:
            scores, outcome = dict.fromkeys(conditions), "corrupt_checks"
        elif not d_ok:
            scores, outcome = dict.fromkeys(conditions), "corrupt_decision"
        else:
            scores, outcome = score_level_outputs(
                graph, checks, decision, conditions, limits
            )
        for name, c in scores.items():
            rows.append(
                {
                    "image_id": image_id,
                    "condition": name,
                    "t": t,
                    "C": c,
                    "outcome": outcome,
                }
            )
    return rows


def replay_conditions(
    c_table: pd.DataFrame, conditions: Mapping[str, Condition], num_levels: int
) -> pd.DataFrame:
    """stops.csv rows for the adaptive conditions."""
    seqs: dict[tuple[int, str], list[int | None]] = {}
    for rec in c_table.to_dict("records"):
        key = (int(rec["image_id"]), str(rec["condition"]))
        seq = seqs.setdefault(key, [None] * num_levels)
        c = rec["C"]
        seq[int(rec["t"]) - 1] = None if pd.isna(c) else int(c)
    rows = []
    for (image_id, name), seq in sorted(seqs.items()):
        cond = conditions[name]
        res = replay_stop(seq, cond.theta, cond.consecutive_k)
        rows.append(
            {
                "image_id": image_id,
                "method": name,
                "stop_t": res.stop_t,
                "natural_stop": res.natural_stop,
                "forced_terminal": res.forced_terminal,
                "failure_count": res.failure_count,
            }
        )
    return pd.DataFrame(rows, columns=list(STOPS_COLUMNS))


def default_comparisons(num_levels: int) -> list[tuple[str, str]]:
    return [
        (PROPOSED, OBSERVED_ONLY),
        (PROPOSED, UNKNOWN_CONSERVATIVE),
        (PROPOSED, fixed_method_name(num_levels)),
        (OBSERVED_ONLY, fixed_method_name(num_levels)),
        # Exploratory (plan deviation, needs sign-off).
        (BENEFIT_GATED, BENEFIT_GATED_OBS),
        (BENEFIT_GATED, fixed_method_name(num_levels)),
    ]


FIXED_STATUS_LABEL = "baseline"


def method_status(method: str, conditions: Mapping[str, Condition]) -> str:
    """main / exploratory label of a method (fixed_k -> baseline)."""
    cond = conditions.get(method)
    return cond.status_label if cond is not None else FIXED_STATUS_LABEL


def run_replay_eval(
    input_dir: Path,
    out_dir: Path,
    cfg: Mapping[str, Any],
    layout: RunLayout | None = None,
    image_ids: Sequence[int] | None = None,
    q_metric: str | None = None,
) -> dict[str, Any]:
    """Score all levels, replay stops, evaluate and write outputs. Returns a digest.

    ``q_metric`` overrides ``evaluation.q_metric`` (stage_metrics column used as
    q_t for R(stop), residual, oracle, fixed curves, mix and bootstrap).
    """
    layout = layout or RunLayout()
    params = EvalParams.from_config(cfg, q_metric)
    conditions = build_conditions(cfg)
    limits = ContractLimits.from_config(cfg)

    stage = load_stage_metrics(input_dir / STAGE_METRICS_FILENAME, params.q_metric)
    curves = build_curves(stage, params.q_metric, params.num_levels)
    ids = sorted(curves.q) if image_ids is None else sorted(image_ids)

    c_rows: list[dict[str, Any]] = []
    for image_id in ids:
        c_rows.extend(
            score_image(
                input_dir, image_id, params.num_levels, conditions, limits, layout
            )
        )
    c_table = pd.DataFrame(c_rows)
    c_table["C"] = c_table["C"].astype("Int64")

    stops = pd.concat(
        [
            replay_conditions(c_table, conditions, params.num_levels),
            fixed_stops(ids, params.num_levels),
        ],
        ignore_index=True,
    )
    eval_set = build_eval_set(curves, params.oracle_ratio)
    pred_hits = load_predicate_hits(input_dir / PRED_HITS_FILENAME, params.recall_k)
    per_image = per_image_metrics(stops, curves, eval_set)
    adaptive = list(conditions)
    summary = summarize(per_image, stops, curves, eval_set, pred_hits, adaptive)
    summary.insert(
        1, "status", [method_status(m, conditions) for m in summary["method"]]
    )
    # Main conditions first, then exploratory, then fixed baselines.
    order = {MAIN_LABEL: 0, EXPLORATORY_LABEL: 1, FIXED_STATUS_LABEL: 2}
    summary = summary.sort_values(
        ["status", "method"],
        key=lambda col: col.map(order) if col.name == "status" else col,
    ).reset_index(drop=True)
    boot = paired_bootstrap(
        per_image,
        curves,
        eval_set,
        adaptive,
        default_comparisons(params.num_levels),
        params.bootstrap_repeats,
        params.bootstrap_seed,
        params.ci_level,
    )

    boot["condition_status"] = {n: c.status_label for n, c in conditions.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    c_table.to_csv(out_dir / "c_sequences.csv", index=False)
    stops.to_csv(out_dir / "stops.csv", index=False)
    per_image.to_csv(out_dir / "per_image.csv", index=False)
    oracle_table(eval_set, curves).to_csv(out_dir / "oracle.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "bootstrap.json").write_text(json.dumps(boot, indent=2))
    eval_ids = {
        "input_dir": str(input_dir),
        "num_levels": params.num_levels,
        "q_metric": params.q_metric,
        "n_eval": len(eval_set.ids),
        "eval_ids": list(eval_set.ids),
        "excluded": {str(k): v for k, v in eval_set.excluded.items()},
        "oracle_status_counts": dict(oracle_status_counts(eval_set)),
        "mR50_available": pred_hits is not None,
    }
    (out_dir / "eval_set_ids.json").write_text(json.dumps(eval_ids, indent=2))
    return {
        "n_images": len(ids),
        "n_eval": len(eval_set.ids),
        "q_metric": params.q_metric,
        "oracle_status_counts": eval_ids["oracle_status_counts"],
        "outcomes": c_table.drop_duplicates(["image_id", "t"])["outcome"]
        .value_counts()
        .to_dict(),
        "summary": summary,
    }
