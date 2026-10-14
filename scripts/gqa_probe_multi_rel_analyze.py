"""Exploratory probe analysis: P_min vs A1 agreement on multi-object questions.

Reads (read-only) the probe level table and plans, and the frozen v27 test
level table for the contrast. Writes outputs/gqa_stop/probe_multi_rel/analysis/.
Definitions (per (question, level) row):
- own_k: item k's P_ans if valid, resolved and answer != "unknown", else 0
  (the item's own gate, ignoring depends_on).
- last binds: pans_min == own_last; earlier binds: pans_min < own_last
  (an earlier item, or a failed dependency, holds the min below the last item).
Usage: uv run python scripts/gqa_probe_multi_rel_analyze.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path("/home/kanaoka/usr/github/milestone_1")
PROBE = REPO / "outputs/gqa_stop/probe_multi_rel"
V27_TEST = REPO / "outputs/gqa_stop/visapp_gqa_v27/level_table_test.csv"
OUT = PROBE / "analysis"
UNKNOWN = "unknown"
EPS = 1e-12
THETA_SAME = 0.80  # v27 dev θ* of P_min
THETA_STAR = {"P_min": 0.80, "A1": 0.75}  # v27 dev θ* (frozen)


def norm(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return " ".join(str(value).lower().strip().split())


def truthy(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def add_derived(table: pd.DataFrame) -> pd.DataFrame:
    n = table["n_items"].astype(int)
    own_last, ans_last, ok_last = [], [], []
    for row, k in zip(table.itertuples(index=False), n, strict=True):
        d = row._asdict()
        ok = (
            truthy(d[f"item{k}_valid"])
            and truthy(d[f"item{k}_resolved"])
            and norm(d[f"item{k}_answer"]) != UNKNOWN
        )
        ok_last.append(ok)
        own_last.append(float(d[f"item{k}_pans"]) if ok else 0.0)
        ans_last.append(norm(d[f"item{k}_answer"]))
    dres_ok = (
        table["dres_valid"].map(truthy)
        & table["dres_resolved"].map(truthy)
        & (table["dres_answer"].map(norm) != UNKNOWN)
    )
    return table.assign(
        own_last=own_last,
        last_answer=ans_last,
        last_ok=ok_last,
        dres_ok=dres_ok,
        dres_norm=table["dres_answer"].map(norm),
    )


def agreement(t: pd.DataFrame) -> dict[str, Any]:
    pm, last = t["pans_min"].astype(float), t["own_last"]
    last_binds = (pm - last).abs() < EPS
    nontrivial = last > 0
    both_valid = t["dres_valid"].map(truthy) & (t["last_answer"] != "")
    a3 = t["all_resolved"].astype(float) > 0
    corr = float(np.corrcoef(pm, t["dres_score"].astype(float))[0, 1])
    return {
        "rows": len(t),
        "questions": t["question_id"].nunique(),
        "corr_pmin_a1": round(corr, 3),
        "last_binds": round(float(last_binds.mean()), 3),
        "earlier_binds": round(float((~last_binds).mean()), 3),
        "last_binds_given_last_pos": round(float(last_binds[nontrivial].mean()), 3),
        "rows_last_pos": int(nontrivial.sum()),
        "last_ans_eq_a1": round(
            float((t["last_answer"] == t["dres_norm"])[both_valid].mean()), 3
        ),
        "last_ok_eq_a1_ok": round(float((t["last_ok"] == t["dres_ok"]).mean()), 3),
        "a3_eq_a1_ok": round(float((a3 == t["dres_ok"]).mean()), 3),
        "mean_pmin": round(float(pm.mean()), 3),
        "mean_a1": round(float(t["dres_score"].astype(float).mean()), 3),
    }


def stop_level(scores: np.ndarray, theta: float) -> int:
    hit = np.nonzero(scores >= theta)[0]
    return int(hit[0] + 1) if len(hit) else 0  # 0 = never


def per_question(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for qid, g in t.sort_values("level").groupby("question_id", sort=False):
        pm = g["pans_min"].to_numpy(float)
        a1 = g["dres_score"].to_numpy(float)
        last = g["own_last"].to_numpy(float)
        rc = g["reader_correct"].map(truthy).to_numpy()
        rows.append(
            {
                "question_id": qid,
                "type": g["attribute"].iloc[0],
                "label": g["label"].iloc[0],
                "stop_pmin_same": stop_level(pm, THETA_SAME),
                "stop_a1_same": stop_level(a1, THETA_SAME),
                "stop_last_same": stop_level(last, THETA_SAME),
                "stop_pmin_star": stop_level(pm, THETA_STAR["P_min"]),
                "stop_a1_star": stop_level(a1, THETA_STAR["A1"]),
                "earlier_binds_levels": int(((last - pm) > EPS).sum()),
                "reader_correct_levels": int(rc.sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    probe = add_derived(
        pd.read_csv(PROBE / "level_table_dev.csv", dtype={"question_id": str})
    )
    test = add_derived(pd.read_csv(V27_TEST, dtype={"question_id": str}))
    stats = {"v27_test": agreement(test), "probe_all": agreement(probe)}
    for kind, part in probe.groupby("attribute"):
        stats[f"probe_{kind}"] = agreement(part)
    for label, part in probe.groupby("label"):
        stats[f"probe_label_{label}"] = agreement(part)
    pd.DataFrame(stats).T.to_csv(OUT / "agreement.csv")
    pq = per_question(probe)
    pq.to_csv(OUT / "per_question.csv", index=False)
    tq = per_question(test)
    div = {
        "probe_same_theta_diff_stop": int(
            (pq["stop_pmin_same"] != pq["stop_a1_same"]).sum()
        ),
        "test_same_theta_diff_stop": round(
            float((tq["stop_pmin_same"] != tq["stop_a1_same"]).mean()), 3
        ),
        "probe_same_theta_diff_stop_share": round(
            float((pq["stop_pmin_same"] != pq["stop_a1_same"]).mean()), 3
        ),
        "probe_star_diff_stop_share": round(
            float((pq["stop_pmin_star"] != pq["stop_a1_star"]).mean()), 3
        ),
        "test_star_diff_stop_share": round(
            float((tq["stop_pmin_star"] != tq["stop_a1_star"]).mean()), 3
        ),
        "probe_q_with_earlier_binding": int((pq["earlier_binds_levels"] > 0).sum()),
        "test_q_with_earlier_binding_share": round(
            float((tq["earlier_binds_levels"] > 0).mean()), 3
        ),
    }
    (OUT / "divergence.json").write_text(json.dumps(div, indent=1))
    print(pd.DataFrame(stats).T.to_string())
    print(json.dumps(div, indent=1))
    print(pq.to_string())


if __name__ == "__main__":
    main()
