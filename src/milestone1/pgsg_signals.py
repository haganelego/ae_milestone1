"""PGSG-internal self-consistency signals (``pgsg_internal``, zero model calls).

Computed from one cached graph JSON (contract in
.claude/docs/research/robovis2027-worklog.md). Exploratory alternative stop
signal: how consistently PGSG's 36 sampled segments agree on the scene.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

TOP_ENTITIES = 20
TOP_RELATIONS = 50
MIN_MENTIONS_AGREE = 2

SIGNAL_FIELDS = (
    "n_entities",
    "n_unique_labels",
    "bad_inst_rate",
    "mean_mentions_top20",
    "frac_rel50_agree2",
    "mean_label_maxprob_top20",
    "mean_pred_maxprob_top50",
    "agreement",
)


def normalized_top1(topk: Sequence[Sequence[float]]) -> float:
    """top-1 value / sum of the stored top-k values (NaN if empty or zero sum)."""
    values = [float(v) for _, v in topk]
    total = sum(values)
    return values[0] / total if values and total > 0 else float("nan")


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return (
        float(np.nanmean(arr)) if len(arr) and not np.isnan(arr).all() else float("nan")
    )


def top_entities(graph: dict[str, Any], n: int = TOP_ENTITIES) -> list[dict[str, Any]]:
    return sorted(graph["entities"], key=lambda e: -float(e["score"]))[:n]


def top_relations(
    graph: dict[str, Any], n: int = TOP_RELATIONS
) -> list[dict[str, Any]]:
    return sorted(graph["relations"], key=lambda r: -float(r["trp_score"]))[:n]


def compute_signals(graph: dict[str, Any]) -> dict[str, float]:
    """All ``SIGNAL_FIELDS`` for one state (NaN where undefined)."""
    ents = graph["entities"]
    by_id = {e["id"]: e for e in ents}
    stats = graph.get("stats", {})
    n_raw, n_bad = stats.get("n_raw_triplets"), stats.get("n_bad_inst")
    bad_rate = n_bad / n_raw if n_raw and n_bad is not None else float("nan")

    top_e = top_entities(graph)
    mentions = [e["n_mentions"] for e in top_e if e.get("n_mentions") is not None]
    top_r = top_relations(graph)
    agree = [
        (by_id[r["sub"]].get("n_mentions") or 0) >= MIN_MENTIONS_AGREE
        and (by_id[r["obj"]].get("n_mentions") or 0) >= MIN_MENTIONS_AGREE
        for r in top_r
    ]
    frac_agree = float(np.mean(agree)) if agree else float("nan")
    return {
        "n_entities": float(len(ents)),
        "n_unique_labels": float(len({e["label_id"] for e in ents})),
        "bad_inst_rate": float(bad_rate),
        "mean_mentions_top20": _nanmean(mentions),
        "frac_rel50_agree2": frac_agree,
        "mean_label_maxprob_top20": _nanmean(
            [normalized_top1(e["label_dist_topk"]) for e in top_e]
        ),
        "mean_pred_maxprob_top50": _nanmean(
            [normalized_top1(r["pred_dist_topk"]) for r in top_r]
        ),
        "agreement": frac_agree,
    }
