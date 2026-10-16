"""Oracle-checklist probe evaluation (exploratory, in-sample, not pre-registered).

Reads ``level_table_oracle.csv`` (from ``gqa_probe_oracle_run.py table``) and
writes ``eval/`` in the probe_oracle run dir:
- sweep.csv / sweep_by_type.csv: strict + unified θ sweep (v27 definitions,
  reader answer scored) for every method below.
- theta_star.json: in-sample argmax of balanced accuracy (strict and unified).
- at_theta.csv: each method at the fixed θ list and at its in-sample θ*.
- agreement.csv: per type, corr with A1 / P_min, stop-level divergence and
  win / loss counts against A1 and P_min at each fixed θ.
- veto.json: object_absent rows, and stops the veto blocks (P_oracle_logic vs
  P_oracle_logic_noveto), split by label.
- per_question.csv: stop level and strict correctness per method at each fixed θ.
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

METHODS: dict[str, str] = {
    "B1": "reader_pans",
    "A1": "dres_score",
    "P_min": "pans_min",
    "P_oracle_min": "oracle_min",
    "P_oracle_logic": "oracle_logic",
    "P_oracle_logic_noveto": "oracle_logic_noveto",
    "P_oracle_kleene": "oracle_kleene",
}
ORACLE_METHODS = ("P_oracle_min", "P_oracle_logic", "P_oracle_kleene")
REFERENCES = ("A1", "P_min")
VETO_PAIR = ("P_oracle_logic", "P_oracle_logic_noveto")
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


def agreement(arrays: EvalArrays, thetas: Sequence[float]) -> pd.DataFrame:
    rows = []
    for name, part in subsets(arrays):
        for method in ORACLE_METHODS:
            for ref in REFERENCES:
                row: dict[str, Any] = {
                    "set": name,
                    "method": method,
                    "reference": ref,
                    "n_questions": len(part.question_ids),
                    "corr": corr(part.scores[method], part.scores[ref]),
                }
                for theta in thetas:
                    a, b = outcomes(part, method, theta), outcomes(part, ref, theta)
                    differ = a["stop"] != b["stop"]
                    row[f"stop_differs@{theta:.2f}"] = int(differ.sum())
                    row[f"method_wins@{theta:.2f}"] = int(
                        (a["strict"] & ~b["strict"]).sum()
                    )
                    row[f"reference_wins@{theta:.2f}"] = int(
                        (~a["strict"] & b["strict"]).sum()
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def veto_stats(
    table: pd.DataFrame, arrays: EvalArrays, thetas: Sequence[float]
) -> dict[str, Any]:
    flags = table.assign(object_absent=table["object_absent"].astype(bool))
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
        "veto_questions": {
            q: {"levels": int(r["sum"]), "type": r["attribute"], "label": r["label"]}
            for q, r in by_q[by_q["any"]].iterrows()
        },
        "blocked": {},
    }
    logic, noveto = VETO_PAIR
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
            "questions": [
                q for q, hit in zip(arrays.question_ids, blocked, strict=True) if hit
            ],
        }
    return report


def per_question(arrays: EvalArrays, thetas: Sequence[float]) -> pd.DataFrame:
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
    agree = agreement(arrays, fixed)
    agree.to_csv(out / "agreement.csv", index=False)
    veto = veto_stats(table, arrays, [*fixed, stars["P_oracle_logic"]])
    atomic_write_json(out / "veto.json", veto)
    per_question(arrays, fixed).to_csv(out / "per_question.csv", index=False)
    print("balanced accuracy (strict) by θ:")
    print(markdown_sweep(full))
    print("θ* strict (in-sample):", json.dumps(stars))
    print("θ* unified (in-sample):", json.dumps(stars_unified))
    print(
        "veto:",
        json.dumps(
            {
                k: veto[k]
                for k in ("rows_object_absent", "questions_with_veto", "blocked")
            }
        ),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
