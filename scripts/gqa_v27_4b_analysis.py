"""v27 dev model swap (qwen3.5:4b, plan §27.13): secondary view and mechanism numbers.

Cache / level-table only; no model call. Reads (read-only) the 4b dev level table,
the 9b dev and test level tables, both screening files and the 4b records.
Writes outputs/gqa_stop/visapp_gqa_v27_4b/analysis_4b/.

1. Secondary view: the 4b dev level table re-labelled with the 9b screening labels
   (same 300 questions, population identical to the 9b run), scored with the
   unchanged v27_eval functions: nine methods, θ sweep, values at the 4b θ*
   (strict selection of eval/dev and eval_posthoc/dev), and paired bootstraps.
2. Label overlap (4b vs 9b), 2 x 2.
3. Mechanism numbers per (question, level) row, definitions as in
   scripts/gqa_probe_multi_rel_analyze.py:
   - own_k: item k's P_ans if valid, resolved and answer != "unknown", else 0;
   - last binds: pans_min == own_last; earlier binds: pans_min < own_last.
   Existence items = every item before the last one.
4. Answer accuracy per level (reader, last item, direct_resolved), planner
   validity, n_items and trailing-period counts.
Usage: uv run python scripts/gqa_v27_4b_analysis.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from milestone1.gqa_stop import v27_eval as ev
from milestone1.gqa_stop.common import read_id_csv, read_json, read_jsonl
from milestone1.gqa_stop.evaluate import is_correct

REPO = Path("/home/kanaoka/usr/github/milestone_1")
RUN_4B = REPO / "outputs/gqa_stop/visapp_gqa_v27_4b"
RUN_9B = REPO / "outputs/gqa_stop/visapp_gqa_v27"
SCREEN_4B = REPO / "data/processed/gqa_stop/visapp_gqa_v27_4b/screening_dev.jsonl"
SCREEN_9B = REPO / "data/processed/gqa_stop/visapp_gqa_v27/screening_dev.jsonl"
OUT = RUN_4B / "analysis_4b"
N_LEVELS = 20
REPEATS = 2000
SEED = 20261008
UNKNOWN = "unknown"
EPS = 1e-12
REPORT_LEVELS = (1, 5, 10, 15, 20)
PAIRS = (("P_min", "B1"), ("P_min", "A1"), ("P'_min", "A1'"), ("A1", "B1"))
ROLE_DIRS = (
    "reader",
    "reader_lp",
    "plans_v3",
    "observer_v3_lp",
    "observer_v3_lp_direct",
)


def norm(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return " ".join(str(value).lower().strip().split())


def truthy(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def item_ok(row: dict[str, Any], k: int) -> bool:
    return (
        truthy(row[f"item{k}_valid"])
        and truthy(row[f"item{k}_resolved"])
        and norm(row[f"item{k}_answer"]) != UNKNOWN
    )


# ---- 1. secondary view ---------------------------------------------------------------
def strict_stars() -> dict[str, float]:
    """4b θ* (strict rule) for the six confirmatory and three post-hoc methods."""
    main = read_json(RUN_4B / "eval/dev/theta_star.json")["theta_star"]
    post = read_json(RUN_4B / "eval_posthoc/dev/theta_star.json")["theta_star"]
    return {
        **{m: float(main[m]) for m in ev.METHODS},
        **{m: float(post["strict"][m]) for m in ev.POSTHOC_METHODS},
    }


def relabel(table: pd.DataFrame, screening: Path) -> pd.DataFrame:
    labels = {r["question_id"]: r["label"] for r in read_jsonl(screening)}
    return table.assign(label=table["question_id"].map(labels))


def secondary_view(table_4b: pd.DataFrame) -> dict[str, Any]:
    arrays = ev.arrays_from_table(relabel(table_4b, SCREEN_9B), N_LEVELS)
    grid = ev.theta_grid({"start": 0.0, "stop": 1.0, "step": 0.05})
    sweep = ev.sweep(arrays, grid, ev.ALL_METHODS)
    sweep.to_csv(OUT / "secondary_9b_labels_sweep.csv", index=False)
    stars = strict_stars()
    at_star = pd.DataFrame(
        [ev.summarize(arrays, m, ev._finite_or_none(t)) for m, t in stars.items()]
    )
    at_star.to_csv(OUT / "secondary_9b_labels_at_4b_theta_star.csv", index=False)
    resel = {m: ev.theta_star(sweep, "balanced_strict")[m] for m in ev.ALL_METHODS}
    at_resel = pd.DataFrame(
        [ev.summarize(arrays, m, ev._finite_or_none(t)) for m, t in resel.items()]
    )
    at_resel.to_csv(OUT / "secondary_9b_labels_at_reselected_theta.csv", index=False)
    ci = ev.method_ci(arrays, stars, REPEATS, SEED, ev.POSTHOC_DIFF_KEYS)
    ci.to_csv(OUT / "secondary_9b_labels_method_ci.csv", index=False)
    pb = ev.paired_bootstrap_at(
        arrays, PAIRS, stars, REPEATS, SEED, ev.POSTHOC_DIFF_KEYS
    )
    pb.to_csv(OUT / "secondary_9b_labels_paired_at_4b_theta_star.csv", index=False)
    return {"theta_star_4b": stars, "reselected_under_9b_labels": resel}


# ---- 2. labels ---------------------------------------------------------------------
def label_overlap() -> dict[str, Any]:
    a = {r["question_id"]: r["label"] for r in read_jsonl(SCREEN_4B)}
    b = {r["question_id"]: r["label"] for r in read_jsonl(SCREEN_9B)}
    ans4 = {r["question_id"]: r["reader_answer"] for r in read_jsonl(SCREEN_4B)}
    ans9 = {r["question_id"]: r["reader_answer"] for r in read_jsonl(SCREEN_9B)}
    cells = Counter((a[q], b[q]) for q in a)
    both_wrong = [q for q in a if a[q] == b[q] == "unanswerable"]
    same_wrong = sum(norm(ans4[q]) == norm(ans9[q]) for q in both_wrong)
    return {
        "cells_4b_x_9b": {f"{k[0]}|{k[1]}": v for k, v in sorted(cells.items())},
        "agreement": sum(v for k, v in cells.items() if k[0] == k[1]) / len(a),
        "both_unanswerable_same_wrong_answer": same_wrong,
        "both_unanswerable": len(both_wrong),
    }


# ---- 3. mechanism ------------------------------------------------------------------
def add_derived(table: pd.DataFrame) -> pd.DataFrame:
    rows = table.to_dict("records")
    own_last, last_ans, last_ok = [], [], []
    for row in rows:
        k = int(row["n_items"])
        if k == 0:  # planner failure: no item
            own_last.append(0.0)
            last_ans.append("")
            last_ok.append(False)
            continue
        ok = item_ok(row, k)
        own_last.append(float(row[f"item{k}_pans"]) if ok else 0.0)
        last_ans.append(norm(row[f"item{k}_answer"]))
        last_ok.append(ok)
    dres_ok = (
        table["dres_valid"].map(truthy)
        & table["dres_resolved"].map(truthy)
        & (table["dres_answer"].map(norm) != UNKNOWN)
    )
    return table.assign(
        own_last=own_last,
        last_answer=last_ans,
        last_ok=last_ok,
        dres_ok=dres_ok,
        dres_norm=table["dres_answer"].map(norm),
    )


def agreement(t: pd.DataFrame) -> dict[str, Any]:
    t = t[t["n_items"].astype(int) > 0]
    pm, last = t["pans_min"].astype(float), t["own_last"]
    last_binds = (pm - last).abs() < EPS
    nontrivial = last > 0
    both_valid = t["dres_valid"].map(truthy) & (t["last_answer"] != "")
    a3 = t["all_resolved"].astype(float) > 0
    return {
        "rows": len(t),
        "questions": t["question_id"].nunique(),
        "corr_pmin_a1": float(np.corrcoef(pm, t["dres_score"].astype(float))[0, 1]),
        "last_binds": float(last_binds.mean()),
        "earlier_binds": float((~last_binds).mean()),
        "last_binds_given_last_pos": float(last_binds[nontrivial].mean()),
        "last_ans_eq_a1": float(
            (t["last_answer"] == t["dres_norm"])[both_valid].mean()
        ),
        "last_ok_eq_a1_ok": float((t["last_ok"] == t["dres_ok"]).mean()),
        "a3_eq_a1_ok": float((a3 == t["dres_ok"]).mean()),
        "mean_pmin": float(pm.mean()),
        "mean_a1": float(t["dres_score"].astype(float).mean()),
    }


def existence_by_level(t: pd.DataFrame) -> pd.DataFrame:
    """Items before the last one: resolved-ok rate, mean raw P_ans, 'yes' share."""
    recs = []
    for row in t.to_dict("records"):
        for k in range(1, int(row["n_items"])):
            recs.append(
                {
                    "level": int(row["level"]),
                    "label": row["label"],
                    "ok": item_ok(row, k),
                    "pans": float(row[f"item{k}_pans"]),
                    "yes": norm(row[f"item{k}_answer"]) == "yes",
                }
            )
    frame = pd.DataFrame(recs)
    return (
        frame.groupby("level")
        .agg(
            n=("ok", "size"),
            resolved_ok=("ok", "mean"),
            mean_pans=("pans", "mean"),
            yes_share=("yes", "mean"),
        )
        .reset_index()
    )


def per_level_binds(t: pd.DataFrame) -> pd.DataFrame:
    t = t[t["n_items"].astype(int) > 0]
    binds = (t["pans_min"].astype(float) - t["own_last"]).abs() < EPS
    return (
        t.assign(last_binds=binds, pmin=t["pans_min"].astype(float))
        .groupby("level")
        .agg(last_binds=("last_binds", "mean"), mean_pmin=("pmin", "mean"))
        .reset_index()
    )


def discordance(table: pd.DataFrame, stars: dict[str, float]) -> dict[str, Any]:
    """Per question at θ*: P_min right & A1 wrong vs the reverse (strict)."""
    arrays = ev.arrays_from_table(table, N_LEVELS)
    out_p = ev.outcomes(arrays, "P_min", stars["P_min"])
    out_a = ev.outcomes(arrays, "A1", stars["A1"])
    same_stop = out_p["stop"] == out_a["stop"]
    return {
        "theta_P_min": stars["P_min"],
        "theta_A1": stars["A1"],
        "different_stop_share": float((~same_stop).mean()),
        "P_right_A1_wrong": int((out_p["strict"] & ~out_a["strict"]).sum()),
        "A1_right_P_wrong": int((~out_p["strict"] & out_a["strict"]).sum()),
    }


# ---- 4. accuracy, planner, trailing -------------------------------------------------
def accuracy_by_level(t: pd.DataFrame) -> pd.DataFrame:
    def correct(answers: pd.Series) -> list[bool]:
        return [
            is_correct(a or None, g)
            for a, g in zip(answers, t["gt_answer"].astype(str), strict=True)
        ]

    frame = t.assign(
        reader=[truthy(v) for v in t["reader_correct"]],
        last_item=correct(t["last_answer"]),
        direct=correct(t["dres_norm"]),
    )
    rows = []
    for label in ("all", "answerable", "unanswerable"):
        part = frame if label == "all" else frame[frame["label"] == label]
        grouped = part.groupby("level")[["reader", "last_item", "direct"]].mean()
        rows.append(grouped.reset_index().assign(label=label))
    return pd.concat(rows, ignore_index=True)


def planner_counts(t: pd.DataFrame) -> dict[str, Any]:
    q = t.groupby("question_id").first()
    valid = q["planner_status"] == "valid"
    initial = q["planner_initial_valid"].map(truthy)
    return {
        "questions": len(q),
        "initial_valid": int((valid & initial).sum()),
        "repaired": int((valid & ~initial).sum()),
        "failed": int((~valid).sum()),
        "n_items": {
            str(k): int(v) for k, v in q["n_items"].value_counts().sort_index().items()
        },
        "last_item_has_attribute": int(q["last_item_has_attribute"].map(truthy).sum()),
    }


def trailing_counts(root: Path) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for role in ROLE_DIRS:
        files = list((root / role).glob("**/*.json"))
        recs = [read_json(f) for f in files]
        counts[role] = {
            "records": len(recs),
            "valid": sum(r.get("status") == "valid" for r in recs),
            "trailing_period_stripped": sum(
                bool(r.get("trailing_period_stripped")) for r in recs
            ),
            "initial_invalid": sum(r.get("initial_valid") is False for r in recs),
        }
    return counts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {
        "4b_dev": read_id_csv(RUN_4B / "level_table_dev.csv"),
        "9b_dev": read_id_csv(RUN_9B / "level_table_dev.csv"),
        "9b_test": read_id_csv(RUN_9B / "level_table_test.csv"),
    }
    derived = {k: add_derived(t) for k, t in tables.items()}
    result: dict[str, Any] = {"secondary_view": secondary_view(tables["4b_dev"])}
    result["labels"] = label_overlap()
    result["agreement"] = {k: agreement(t) for k, t in derived.items()}
    stars_4b = strict_stars()
    stars_9b = read_json(RUN_9B / "eval/dev/theta_star.json")["theta_star"]
    result["discordance_at_theta_star"] = {
        "4b_dev": discordance(tables["4b_dev"], stars_4b),
        "9b_dev": discordance(tables["9b_dev"], stars_9b),
        "9b_test": discordance(tables["9b_test"], stars_9b),
    }
    result["planner"] = {k: planner_counts(t) for k, t in tables.items()}
    result["trailing_4b"] = trailing_counts(RUN_4B)
    for name, t in derived.items():
        existence_by_level(t).to_csv(
            OUT / f"existence_by_level_{name}.csv", index=False
        )
        per_level_binds(t).to_csv(OUT / f"binds_by_level_{name}.csv", index=False)
        accuracy_by_level(t).to_csv(OUT / f"accuracy_by_level_{name}.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
