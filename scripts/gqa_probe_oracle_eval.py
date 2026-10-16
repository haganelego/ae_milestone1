"""Oracle-checklist probe evaluation (exploratory, in-sample, not pre-registered).

Reads ``level_table_oracle.csv`` (from ``gqa_probe_oracle_run.py table``) and
writes ``eval/`` in the probe_oracle run dir:
- sweep.csv / sweep_by_type.csv: strict + unified θ sweep (v27 definitions,
  reader answer scored) for every method below.
- theta_star.json: in-sample argmax of balanced accuracy (strict and unified).
- at_theta.csv: each method at the fixed θ list and at its in-sample θ*.
- agreement.csv: per type, corr with A1 / P_min, stop-level divergence and
  win / loss counts against A1 and P_min at each fixed θ.
- veto.json: per arm (oracle, v4) object_absent rows and the stops the veto
  blocks (P_*_logic vs P_*_logic_noveto), split by label, with the outcome.
- final_agreement.csv: per type and arm, corr with A1, share of rows where the
  final group binds the logic score, and final answer == A1 answer.
- per_question.csv: stop level and strict correctness per method at each fixed θ
  and at each method's in-sample θ*.
The planner_v4 methods (P_v4_*) are included when the level table holds the
``v4_*`` columns (``oracle.v4_config`` set when the table was built).
Usage: uv run python scripts/gqa_probe_oracle_eval.py --config configs/gqa_stop_probe_oracle.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from milestone1.gqa_stop.common import atomic_write_json, load_effective_config, utc_now
from milestone1.gqa_stop.v27_eval import (
    EvalArrays,
    outcomes,
    summarize,
    sweep,
    sweep_by_attribute,
    theta_grid,
    theta_star,
)
from milestone1.gqa_stop.v27_infer import ANSWERABLE

BASE_METHODS: dict[str, str] = {
    "B1": "reader_pans",
    "A1": "dres_score",
    "P_v3_min": "pans_min",
    "P_oracle_min": "oracle_min",
    "P_oracle_logic": "oracle_logic",
    "P_oracle_logic_noveto": "oracle_logic_noveto",
    "P_oracle_kleene": "oracle_kleene",
}
V4_METHODS: dict[str, str] = {
    "P_v4_min": "v4_min",
    "P_v4_logic": "v4_logic",
    "P_v4_logic_noveto": "v4_logic_noveto",
    "P_v4_kleene": "v4_kleene",
}
METHODS: dict[str, str] = dict(BASE_METHODS)  # + V4_METHODS when present (main)
COMPARED = [
    "P_oracle_min",
    "P_oracle_logic",
    "P_oracle_kleene",
    "P_v4_min",
    "P_v4_logic",
    "P_v4_kleene",
]
REFERENCES = ("A1", "P_v3_min")
# arm -> (logic method, no-veto method, veto column, score prefix)
ARMS = {
    "oracle": ("P_oracle_logic", "P_oracle_logic_noveto", "object_absent", "oracle"),
    "v4": ("P_v4_logic", "P_v4_logic_noveto", "v4_object_absent", "v4"),
}
ALL = "all"


def arrays_from(table: pd.DataFrame, n_levels: int) -> EvalArrays:
    ordered = table.sort_values(["question_id", "level"])
    qids = list(dict.fromkeys(ordered["question_id"].astype(str)))
    if len(ordered) != len(qids) * n_levels:
        raise SystemExit("level table is not complete (n x L)")
    meta = ordered.groupby("question_id", sort=True).first()

    def matrix(column: str) -> np.ndarray:
        return ordered[column].to_numpy().reshape(len(qids), n_levels)

    return EvalArrays(
        qids,
        [str(g) for g in meta.loc[qids, "group_id"]],
        meta.loc[qids, "attribute"].to_numpy(),
        (meta.loc[qids, "label"] == ANSWERABLE).to_numpy(),
        matrix("reader_correct").astype(bool),
        {m: matrix(c).astype(float) for m, c in METHODS.items()},
    )


def subsets(arrays: EvalArrays) -> list[tuple[str, EvalArrays]]:
    types = sorted(set(arrays.attributes))
    return [(ALL, arrays)] + [(t, arrays.subset(arrays.attributes == t)) for t in types]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.ravel(), b.ravel()
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compared() -> list[str]:
    return [m for m in COMPARED if m in METHODS]


def agreement(
    arrays: EvalArrays, thetas: Sequence[float], stars: Mapping[str, float]
) -> pd.DataFrame:
    """Stop divergence vs A1 / P_v3_min at each fixed θ (same θ for both) and at
    each method's own in-sample θ* ("star")."""
    rows = []
    for name, part in subsets(arrays):
        for method in compared():
            for ref in REFERENCES:
                row: dict[str, Any] = {
                    "set": name,
                    "method": method,
                    "reference": ref,
                    "n_questions": len(part.question_ids),
                    "corr": corr(part.scores[method], part.scores[ref]),
                }
                settings = [(f"{t:.2f}", t, t) for t in thetas]
                settings.append(("star", stars[method], stars[ref]))
                for label, t_method, t_ref in settings:
                    a, b = outcomes(part, method, t_method), outcomes(part, ref, t_ref)
                    differ = a["stop"] != b["stop"]
                    row[f"stop_differs@{label}"] = int(differ.sum())
                    row[f"method_wins@{label}"] = int(
                        (a["strict"] & ~b["strict"]).sum()
                    )
                    row[f"reference_wins@{label}"] = int(
                        (~a["strict"] & b["strict"]).sum()
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def normalized(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip().str.lower()


def final_agreement(table: pd.DataFrame) -> pd.DataFrame:
    """Per type and arm: corr(score, A1), final group binds the logic score
    (F <= E), final answer == A1 (direct_resolved) answer; rows = question x level."""
    rows = []
    dres = normalized(table["dres_answer"])
    for arm, (_, _, _, prefix) in ARMS.items():
        if f"{prefix}_final" not in table:
            continue
        binds = table[f"{prefix}_final"] <= table[f"{prefix}_existence"]
        same = normalized(table[f"{prefix}_final_answer"]) == dres
        frame = table.assign(binds=binds, same=same)
        for name, part in [(ALL, frame), *frame.groupby("attribute")]:
            rows.append(
                {
                    "arm": arm,
                    "set": name,
                    "rows": len(part),
                    "corr_logic_A1": corr(
                        part[f"{prefix}_logic"].to_numpy(),
                        part["dres_score"].to_numpy(),
                    ),
                    "corr_min_A1": corr(
                        part[f"{prefix}_min"].to_numpy(), part["dres_score"].to_numpy()
                    ),
                    "final_binds": float(part["binds"].mean()),
                    "final_answer_eq_A1": float(part["same"].mean()),
                }
            )
    return pd.DataFrame(rows)


def veto_stats(
    table: pd.DataFrame, arrays: EvalArrays, thetas: Sequence[float], arm: str
) -> dict[str, Any]:
    logic, noveto, column, _ = ARMS[arm]
    flags = table.assign(object_absent=table[column].astype(bool))
    by_q = flags.groupby("question_id")["object_absent"].agg(["sum", "any"])
    meta = flags.groupby("question_id")[["attribute", "label"]].first()
    by_q = by_q.join(meta)
    report: dict[str, Any] = {
        "rows_object_absent": int(flags["object_absent"].sum()),
        "rows_total": len(flags),
        "questions_with_veto": int(by_q["any"].sum()),
        "veto_rows_by_type_label": {
            f"{t}|{lab}": int(n)
            for (t, lab), n in flags.groupby(["attribute", "label"])["object_absent"]
            .sum()
            .items()
        },
        "veto_rows_by_label": {
            str(lab): int(n)
            for lab, n in flags.groupby("label")["object_absent"].sum().items()
        },
        "veto_questions": {
            q: {
                "n_levels": int(r["sum"]),
                "levels": sorted(
                    int(v)
                    for v in flags.loc[
                        (flags["question_id"] == q) & flags["object_absent"], "level"
                    ]
                ),
                "type": r["attribute"],
                "label": r["label"],
            }
            for q, r in by_q[by_q["any"]].iterrows()
        },
        "blocked": {},
    }
    for theta in thetas:
        a, b = outcomes(arrays, logic, theta), outcomes(arrays, noveto, theta)
        blocked = (b["stop"] != 0) & (a["stop"] != b["stop"])
        ans = arrays.answerable
        report["blocked"][f"{theta:.2f}"] = {
            "blocked_stops": int(blocked.sum()),
            "blocked_unanswerable": int((blocked & ~ans).sum()),
            "blocked_answerable": int((blocked & ans).sum()),
            "unanswerable_now_correct": int((blocked & ~ans & a["strict"]).sum()),
            "answerable_lost": int((blocked & ans & b["strict"] & ~a["strict"]).sum()),
            "blocked_stop_was_wrong": int((blocked & ~b["strict"]).sum()),
            "blocked_stop_was_right": int((blocked & b["strict"]).sum()),
            "outcome_wrong_to_right": int((blocked & ~b["strict"] & a["strict"]).sum()),
            "outcome_right_to_wrong": int((blocked & b["strict"] & ~a["strict"]).sum()),
            "questions": [
                q for q, hit in zip(arrays.question_ids, blocked, strict=True) if hit
            ],
        }
    return report


def per_question(
    arrays: EvalArrays, thetas: Sequence[float], stars: Mapping[str, float]
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "question_id": arrays.question_ids,
            "type": arrays.attributes,
            "answerable": arrays.answerable,
        }
    )
    for theta in thetas:
        for method in METHODS:
            out = outcomes(arrays, method, theta)
            frame[f"{method}_stop@{theta:.2f}"] = out["stop"]
            frame[f"{method}_strict@{theta:.2f}"] = out["strict"]
    for method in METHODS:
        out = outcomes(arrays, method, stars[method])
        frame[f"{method}_stop@star"] = out["stop"]
        frame[f"{method}_strict@star"] = out["strict"]
    return frame


def at_thetas(
    arrays: EvalArrays, fixed: Sequence[float], stars: Mapping[str, float]
) -> pd.DataFrame:
    rows = []
    for name, part in subsets(arrays):
        for method in METHODS:
            settings = [("fixed", t) for t in fixed]
            if np.isfinite(stars.get(method, float("nan"))):
                settings.append(("theta_star_strict_in_sample", stars[method]))
            for kind, theta in settings:
                rows.append(
                    {"set": name, "setting": kind, **summarize(part, method, theta)}
                )
    return pd.DataFrame(rows)


def markdown_sweep(frame: pd.DataFrame) -> str:
    pivot = frame.pivot_table(index="theta", columns="method", values="balanced_strict")
    pivot = pivot[[m for m in METHODS if m in pivot.columns]]
    return pivot.round(3).to_string()


def main() -> None:
    parser = argparse.ArgumentParser(prog="gqa_probe_oracle_eval")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--table", type=Path, default=None, help="override (tests)")
    parser.add_argument("--out", type=Path, default=None, help="override (tests)")
    args = parser.parse_args()
    cfg, paths = load_effective_config(args.config)
    table_path = args.table or paths.out(cfg["oracle"]["table"])
    out = args.out or paths.out("eval")
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(table_path, dtype={"question_id": str, "group_id": str})
    if "v4_min" in table:
        METHODS.update(V4_METHODS)
    n_levels = int(cfg["images"]["levels"])
    arrays = arrays_from(table, n_levels)
    grid = theta_grid(cfg["v27"]["theta_grid"])
    fixed = [float(t) for t in cfg["oracle"]["fixed_thetas"]]
    methods = list(METHODS)
    full = sweep(arrays, grid, methods)
    full.to_csv(out / "sweep.csv", index=False)
    sweep_by_attribute(arrays, grid, methods).to_csv(
        out / "sweep_by_type.csv", index=False
    )
    stars = theta_star(full)
    stars_unified = theta_star(full, "balanced_unified")
    atomic_write_json(
        out / "theta_star.json",
        {
            "note": "in-sample argmax on the 30 probe questions (ties -> smaller θ)",
            "strict": stars,
            "unified": stars_unified,
            "time": utc_now(),
        },
    )
    at_thetas(arrays, fixed, stars).to_csv(out / "at_theta.csv", index=False)
    agree = agreement(arrays, fixed, stars)
    agree.to_csv(out / "agreement.csv", index=False)
    final_agreement(table).to_csv(out / "final_agreement.csv", index=False)
    veto = {
        arm: veto_stats(table, arrays, sorted({*fixed, stars[ARMS[arm][0]]}), arm)
        for arm in ARMS
        if ARMS[arm][0] in METHODS
    }
    atomic_write_json(out / "veto.json", veto)
    per_question(arrays, fixed, stars).to_csv(out / "per_question.csv", index=False)
    print("balanced accuracy (strict) by θ:")
    print(markdown_sweep(full))
    print("θ* strict (in-sample):", json.dumps(stars))
    print("θ* unified (in-sample):", json.dumps(stars_unified))
    for arm, stats in veto.items():
        keys = ("rows_object_absent", "questions_with_veto", "blocked")
        print(f"veto {arm}:", json.dumps({k: stats[k] for k in keys}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
