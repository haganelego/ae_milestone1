"""MIRU per-image quantities computed directly from EGTR predictions.

Lets the MIRU evaluation run on a new image set that has only EGTR
predictions (no MIRU metric CSVs):

- ``topk_oracle``: the stop_c09 oracle of ``scripts/eval_topk_oracle.py`` --
  reference = L20 top-10 object labels (by confidence) and top-5 triplets (by
  triplet_score); per-level "core" RelSGScore = 0.5 node recall + 0.5 triplet
  recall vs that reference (NaN fallback to the other term); stop = first level
  with score >= coef, else the last level.
- ``object_confidence``: ``mean_object_confidence`` of
  ``scripts/06_compute_sg_metrics.py`` (mean over stored objects, 0 if none,
  rounded to 4 decimals as in all_metrics.csv).
- ``baseline_stops``: ``scripts/08_compute_baseline_stop.py`` -- first level
  (from 1) whose mean object confidence >= theta, else the last level.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOP_N_NODES = 10
TOP_N_TRIPLETS = 5
ORACLE_COEF = 0.9
CONF_DECIMALS = 4
FIRST_LEVEL = 1


def load_prediction(pred_dir: Path, image_id: int, level: int) -> dict[str, Any]:
    return json.loads(
        (pred_dir / str(image_id) / f"level_{level:02d}.json").read_text()
    )


def _labels(objects: Sequence[Mapping[str, Any]]) -> set[str]:
    return {o["label"] for o in objects}


def _triplets(rels: Sequence[Mapping[str, Any]]) -> set[tuple[str, str, str]]:
    return {(r["subject_label"], r["predicate"], r["object_label"]) for r in rels}


def core_relsg(pred: Mapping[str, Any], ref_nodes: set[str],
               ref_trips: set[tuple[str, str, str]]) -> float:  # fmt: skip
    """RelSGScore vs the top-K reference (eval_topk_oracle.compute_rel_sg_score)."""
    node = (len(_labels(pred["objects"]) & ref_nodes) / len(ref_nodes)
            if ref_nodes else math.nan)  # fmt: skip
    trip = (len(_triplets(pred["relationships"]) & ref_trips) / len(ref_trips)
            if ref_trips else math.nan)  # fmt: skip
    if math.isnan(trip):
        return node
    if math.isnan(node):
        return trip
    return 0.5 * node + 0.5 * trip


@dataclass(frozen=True)
class TopkOracle:
    core_scores: dict[int, dict[int, float]]  # image -> level -> core RelSGScore
    stops: dict[int, int]  # image -> stop level (coef)


def first_crossing(scores: Mapping[int, float], coef: float, max_level: int) -> int:
    for lvl in range(FIRST_LEVEL, max_level + 1):
        val = scores.get(lvl, math.nan)
        if not math.isnan(val) and val >= coef:
            return lvl
    return max_level


def topk_oracle(pred_dir: Path, image_ids: Sequence[int], max_level: int,
                coef: float = ORACLE_COEF, top_nodes: int = TOP_N_NODES,
                top_trips: int = TOP_N_TRIPLETS) -> TopkOracle:  # fmt: skip
    core: dict[int, dict[int, float]] = {}
    stops: dict[int, int] = {}
    for image_id in image_ids:
        last = load_prediction(pred_dir, image_id, max_level)
        objs = sorted(last["objects"], key=lambda o: o.get("confidence", 0.0),
                      reverse=True)[:top_nodes]  # fmt: skip
        rels = sorted(last["relationships"], key=lambda r: r.get("triplet_score", 0.0),
                      reverse=True)[:top_trips]  # fmt: skip
        ref_nodes, ref_trips = _labels(objs), _triplets(rels)
        core[image_id] = {
            lvl: core_relsg(
                load_prediction(pred_dir, image_id, lvl), ref_nodes, ref_trips
            )
            for lvl in range(FIRST_LEVEL, max_level + 1)
        }
        stops[image_id] = first_crossing(core[image_id], coef, max_level)
    return TopkOracle(core, stops)


def object_confidence(pred_dir: Path, image_ids: Sequence[int],
                      max_level: int) -> dict[tuple[int, int], float]:  # fmt: skip
    """(image, level) -> mean object confidence, as stored in all_metrics.csv."""
    out: dict[tuple[int, int], float] = {}
    for image_id in image_ids:
        for lvl in range(FIRST_LEVEL, max_level + 1):
            objs = load_prediction(pred_dir, image_id, lvl)["objects"]
            mean = sum(o["confidence"] for o in objs) / len(objs) if objs else 0.0
            out[(image_id, lvl)] = round(mean, CONF_DECIMALS)
    return out


def baseline_stops(conf: Mapping[tuple[int, int], float], image_ids: Sequence[int],
                   theta: float, max_level: int) -> dict[int, int]:  # fmt: skip
    """First level (from 1) with mean object confidence >= theta, else max_level."""
    stops = {}
    for image_id in image_ids:
        stops[image_id] = next(
            (lvl for lvl in range(FIRST_LEVEL, max_level + 1)
             if conf.get((image_id, lvl), 0.0) >= theta),
            max_level,
        )  # fmt: skip
    return stops
