"""v3 pilot diagnostics (plan §18.3) and the negative existence probe.

The negative probe is analysis only: for each pilot image a common object noun
that the GQA scene graph does NOT list for that image is asked about with the
item-1 form ("Is there a {noun} in the image?"). The scene graph is read here
only to choose that noun; it never enters the main pipeline.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import read_json, seeded_hash
from .consistency import token_forms, tokens
from .dataset import normalize_text
from .item_checklist import item_spec
from .sampled import MAX_ITEMS_COLUMNS, METHOD_COLUMNS, Sampling, observer_spec
from .sampled import validate_observation as _validate_observation

HIST_BINS = np.linspace(0.0, 1.0, 11)
NEGATIVE_NAMESPACE = "negprobe"
NEGATIVE_ITEM_ID = "neg"
YES = "yes"
V1_ITEM1_RESOLVED = {"1": 0.32, "5": 0.61, "10": 0.85, "20": 0.98}


def call_rates(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    """First-attempt / post-repair validity and repair share per role."""
    initial: dict[str, float] = {}
    after: dict[str, float] = {}
    repair: dict[str, float] = {}
    for role in ("observer", "direct"):
        calls = float(table.get(f"{role}_calls", pd.Series(dtype=float)).sum())
        if calls == 0:
            continue
        initial[role] = float(table[f"{role}_initial_valid"].sum()) / calls
        after[role] = float(table[f"{role}_valid"].sum()) / calls
        repair[role] = float(table[f"{role}_repaired"].sum()) / calls
    reader = table[table["reader_status"] != "not_run"]
    initial["reader"] = float(reader["reader_initial_valid"].astype(bool).mean())
    after["reader"] = float((reader["reader_status"] == "valid").mean())
    repair["reader"] = float((reader["reader_attempts"] > 1).mean())
    plans = table.groupby("question_id").first()
    initial["planner"] = float(plans["planner_initial_valid"].astype(bool).mean())
    after["planner"] = float((plans["planner_status"] == "valid").mean())
    repair["planner"] = float((plans["planner_attempts"] > 1).mean())
    return {
        "valid_rate_initial": initial,
        "valid_rate_after_repair": after,
        "repair_rate": repair,
    }


def histogram(values: pd.Series) -> list[int]:
    counts, _ = np.histogram(values.clip(0.0, 1.0), bins=HIST_BINS)
    return [int(c) for c in counts]


def pilot_metrics(table: pd.DataFrame) -> dict[str, Any]:
    by_level: dict[str, Any] = {}
    item_cols = [f"item{i}_resolved_frac" for i in range(1, MAX_ITEMS_COLUMNS + 1)]
    for level, frame in table.groupby("level", sort=True):
        item1 = frame["item1_resolved_frac"].dropna()
        present = [c for c in item_cols if c in frame]
        unresolved = 1.0 - frame[present].stack().mean() if present else None
        by_level[str(int(level))] = {
            "reader_accuracy": float(frame["reader_correct"].mean()),
            "item1_resolved_sample_rate": float(item1.mean()),
            "item1_any_resolved_question_rate": float((item1 > 0).mean()),
            "item_sample_unresolved_rate": None
            if unresolved is None
            else float(unresolved),
            "min_sagree_zero_rate": float((frame["v3_min_sagree"] == 0).mean()),
            "histograms_10_bins": {
                col: histogram(frame[col]) for col in METHOD_COLUMNS.values()
            },
            "means": {col: float(frame[col].mean()) for col in METHOD_COLUMNS.values()},
        }
    return {
        **call_rates(table),
        "by_level": by_level,
        "v1_item1_resolved_reference": V1_ITEM1_RESOLVED,
        "histogram_bins": [round(float(b), 1) for b in HIST_BINS],
    }


# ---- v3 final (N = 1, plan §18.6) ----------------------------------------------------
NO = "no"
CAUSE_A = "a_existence_no_or_unresolved"
CAUSE_B = "b_wording_variance"
CAUSE_C = "c_planner_defect"
CAUSE_OTHER = "other_later_item_unresolved"


def l20_zero_causes(table: pd.DataFrame, last_level: int = 20) -> dict[str, Any]:
    """Why is checklist_v3_min 0 at the last level? Non-exclusive flags + primary.

    (a) item 1 (existence) answered "no" or unresolved/invalid;
    (b) wording variance - defined only for sampled S_agree, so n/a at N = 1;
    (c) planner defect: no attribute item, item reference/pronoun, count item;
    other: a later item unresolved / "unknown" / gated to 0.
    Primary cause order: a, c, other.
    """
    rows = table[(table["level"] == last_level) & (table["checklist_v3_min"] <= 0)]
    flags: dict[str, list[str]] = {CAUSE_A: [], CAUSE_C: [], CAUSE_OTHER: []}
    primary: dict[str, list[str]] = {CAUSE_A: [], CAUSE_C: [], CAUSE_OTHER: []}
    for row in rows.itertuples(index=False):
        answer = normalize_text(str(row.item1_answer)) if row.item1_valid else ""
        a = (not row.item1_valid) or (not row.item1_resolved) or answer == NO
        c = (
            (not row.last_item_has_attribute)
            or bool(row.item_not_standalone_any)
            or bool(row.count_item_any)
        )
        qid = str(row.question_id)
        if a:
            flags[CAUSE_A].append(qid)
        if c:
            flags[CAUSE_C].append(qid)
        if not a and not c:
            flags[CAUSE_OTHER].append(qid)
        primary[CAUSE_A if a else CAUSE_C if c else CAUSE_OTHER].append(qid)
    return {
        "n_zero_at_last_level": int(len(rows)),
        "n_questions": int(table["question_id"].nunique()),
        "flag_counts": {**{k: len(v) for k, v in flags.items()}, CAUSE_B: None},
        "primary_counts": {k: len(v) for k, v in primary.items()},
        "primary_question_ids": primary,
        "note": "(b) wording variance applies only to sampled S_agree; n/a at N = 1",
    }


def item_metrics(table: pd.DataFrame) -> dict[str, Any]:
    plans = table.groupby("question_id").first()
    by_level: dict[str, Any] = {}
    for level, frame in table.groupby("level", sort=True):
        answers = frame["item1_answer"].fillna("").map(normalize_text)
        by_level[str(int(level))] = {
            "reader_accuracy": float(frame["reader_correct"].mean()),
            "item1_resolved_rate": float(frame["item1_resolved"].astype(bool).mean()),
            "item1_yes_rate": float((answers == YES).mean()),
            "min_zero_rate": float((frame["checklist_v3_min"] <= 0).mean()),
            "mean_checklist_v3_min": float(frame["checklist_v3_min"].mean()),
            "mean_direct_score": float(frame["direct_score"].mean()),
            "histograms_10_bins": {
                "checklist_v3_min": histogram(frame["checklist_v3_min"]),
                "direct_score": histogram(frame["direct_score"]),
            },
        }
    return {
        **call_rates(table),
        "plan_quality": {
            "attribute_item_fraction": float(plans["last_item_has_attribute"].mean()),
            "item_not_standalone_rate": float(plans["item_not_standalone_any"].mean()),
            "count_item_rate": float(plans["count_item_any"].mean()),
            "n_items": {
                str(k): int(v) for k, v in plans["n_items"].value_counts().items()
            },
        },
        "by_level": by_level,
        "l20_zero_causes": l20_zero_causes(table, int(table["level"].max())),
        "v1_item1_resolved_reference": V1_ITEM1_RESOLVED,
        "histogram_bins": [round(float(b), 1) for b in HIST_BINS],
    }


# ---- negative existence probe ----------------------------------------------------------
def vocabulary(scene_graphs: Mapping[str, Any], top_n: int) -> list[str]:
    """Most common object names (counted once per image); ties by name."""
    counts: Counter[str] = Counter()
    for graph in scene_graphs.values():
        counts.update({o["name"] for o in graph.get("objects", {}).values()})
    return [n for n, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][
        :top_n
    ]


def _forms(text: str) -> set[str]:
    return {form for tok in tokens(text) for form in token_forms(tok)}


def absent_noun(
    image_id: str,
    graph: Mapping[str, Any],
    question: str,
    vocab: Sequence[str],
    seed: int,
) -> str | None:
    """A common noun sharing no token form with the image's objects or question."""
    present = set().union(
        *(_forms(o["name"]) for o in graph.get("objects", {}).values())
    )
    present |= _forms(question)
    options = [n for n in vocab if not (_forms(n) & present)]
    options.sort(key=lambda n: seeded_hash(seed, f"{image_id}|{n}"))
    return options[0] if options else None


def negative_probe(
    engine: Any,
    series: Any,
    questions: Sequence[Mapping[str, Any]],
    probe_cfg: Mapping[str, Any],
    sampling: Sampling | None,
    seed: int,
) -> pd.DataFrame:
    """One observer_v3 call per pilot image and level (sample-0 decoding when
    sampling is configured, otherwise the default v1 decoding)."""
    graphs = read_json(Path(probe_cfg["scene_graphs"]))
    vocab = vocabulary(graphs, int(probe_cfg["vocabulary_top_n"]))
    rows = []
    seen: set[str] = set()
    for question in questions:
        image_id = str(question["image_id"])
        if image_id in seen:
            continue
        seen.add(image_id)
        noun = absent_noun(
            image_id, graphs.get(image_id, {}), str(question["question"]), vocab, seed
        )
        if noun is None:
            continue
        item = {
            "id": NEGATIVE_ITEM_ID,
            "question": probe_cfg["question_template"].format(noun=noun),
        }
        probe_q = {"question_id": f"neg_{image_id}", "image_id": image_id}
        for level in probe_cfg["levels"]:
            image = series.bytes_for(image_id, int(level))
            spec = (
                item_spec(
                    probe_q, int(level), image, item, namespace=NEGATIVE_NAMESPACE
                )
                if sampling is None
                else observer_spec(
                    probe_q,
                    int(level),
                    image,
                    item,
                    0,
                    sampling,
                    namespace=NEGATIVE_NAMESPACE,
                )
            )
            record = engine.run(spec, _validate_observation)
            parsed = record.get("parsed") or {}
            answer = str(parsed.get("answer", ""))
            rows.append(
                {
                    "image_id": image_id,
                    "noun": noun,
                    "level": int(level),
                    "status": record["status"],
                    "answer": answer,
                    "says_yes": tokens(answer)[:1] == [YES],
                    "resolved": bool(parsed.get("resolved", False)),
                    "confidence": parsed.get("confidence"),
                }
            )
    return pd.DataFrame(rows)


def negative_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    valid = frame[frame["status"] == "valid"]
    return {
        str(int(level)): {
            "n": int(len(g)),
            "yes_rate": float(g["says_yes"].mean()),
            "resolved_rate": float(g["resolved"].mean()),
            "resolved_yes_rate": float((g["says_yes"] & g["resolved"]).mean()),
        }
        for level, g in valid.groupby("level", sort=True)
    }
