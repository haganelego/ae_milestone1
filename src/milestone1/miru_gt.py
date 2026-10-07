"""GT-based per-level metrics of the cached MIRU EGTR predictions.

EGTR files (``outputs/egtr_predictions/<image_id>/level_XX.json``) hold the
top-20 objects (``bbox`` in original-image pixels, xyxy) and the top-30
relations (relation_conf >= 0.005) with their own subject/object class ids and
boxes and a ``triplet_score``. MIRU images are the original VG images (no
pre-shrink), so boxes are compared to VG-150 GT in original coordinates.

Metrics per (image, level), via the shared robovis_sgdet code:

- ``R20/R50/R100``: official SGDet triplet recall (graph constraint, +1-pixel
  IoU >= 0.5 on subject and object). EGTR stores <= 30 relations, so R@50 and
  R@100 equal recall over all stored relations.
- ``relsg_gt``: MIRU label-set score vs GT (0.5 node recall + 0.5 triplet
  recall, no boxes) with node set = the stored top-20 object labels and
  triplet set = all stored relations (EGTR's own caps; no further cap).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from milestone1 import robovis_sgdet as sg
from milestone1.gt import _load_coco_and_rels, load_gt_scene_graphs

RECALL_KS = sg.TOPK
STAGE_COLUMNS = ("image_id", "split", "t", "scale", "R20", "R50", "R100", "n_gt_rel",
                 "n_pred_rel", "valid", "relsg_gt", "node_recall_gt", "trip_recall_gt")  # fmt: skip
HIT_K = 50


def label_to_id(vg_root: str | Path, split: str) -> dict[str, int]:
    """VG-150 object name -> 0-based class id (same space as milestone1.gt)."""
    coco, _, _ = _load_coco_and_rels(vg_root, split)
    return {info["name"]: cid - 1 for cid, info in coco.cats.items()}


def egtr_to_graph(pred: Mapping[str, Any]) -> dict[str, Any]:
    """EGTR prediction -> graph contract used by robovis_sgdet (relations ranked).

    Entities are the relation endpoints (keyed by EGTR query id), carrying the
    relation's own class id and box, so relations whose endpoint is outside
    the stored top-20 objects are still evaluated.
    """
    entities: dict[str, dict[str, Any]] = {}
    relations = []
    rels = sorted(pred.get("relationships") or [],
                  key=lambda r: -float(r.get("triplet_score", 0.0)))  # fmt: skip
    for rank, r in enumerate(rels):
        for side in ("subject", "object"):
            eid = f"q{r[f'{side}_id']}"
            entities.setdefault(eid, {
                "id": eid,
                "label_id": int(r[f"{side}_class_id"]),
                "box_xyxy_orig": [float(x) for x in r[f"{side}_bbox"]],
            })  # fmt: skip
        relations.append({
            "id": f"e{rank}", "sub": f"q{r['subject_id']}", "obj": f"q{r['object_id']}",
            "pred_id": int(r["predicate_id"]), "trp_score": float(r.get("triplet_score", 0.0)),
        })  # fmt: skip
    return {"entities": list(entities.values()), "relations": relations}


def egtr_label_sets(
    pred: Mapping[str, Any], name2id: Mapping[str, int]
) -> tuple[set[int], set[sg.LabelTriplet]]:
    """Node label ids of the stored objects and (sub, pred, obj) id triplets."""
    nodes = {name2id[o["label"]] for o in pred.get("objects") or []}
    trips = {
        (int(r["subject_class_id"]), int(r["predicate_id"]), int(r["object_class_id"]))
        for r in pred.get("relationships") or []
    }
    return nodes, trips


def evaluate_state(
    pred: Mapping[str, Any],
    gt_graph: Mapping[str, Any],
    name2id: Mapping[str, int],
) -> tuple[dict[str, Any], sg.ImageMetrics]:
    """Recall@K + GT RelSGScore for one EGTR state."""
    metrics = sg.evaluate_image(
        egtr_to_graph(pred), sg.gt_triplets(dict(gt_graph)), RECALL_KS
    )
    gt_nodes, gt_trips = sg.gt_label_sets(dict(gt_graph))
    nodes, trips = egtr_label_sets(pred, name2id)
    node_recall = sg.set_recall(nodes, gt_nodes)
    trip_recall = sg.set_recall(trips, gt_trips)
    row = {
        **{f"R{k}": metrics.recall[k] for k in RECALL_KS},
        "n_gt_rel": metrics.n_gt_rel,
        "n_pred_rel": len(pred.get("relationships") or []),
        "relsg_gt": sg.combine_relsg(node_recall, trip_recall),
        "node_recall_gt": node_recall,
        "trip_recall_gt": trip_recall,
    }
    return row, metrics


def build_stage_metrics(
    pred_dir: Path,
    image_ids: Sequence[int],
    num_levels: int,
    vg_root: str | Path,
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(stage_metrics rows, stage_predicate_hits rows) for all images x levels."""
    gts = load_gt_scene_graphs(vg_root, split, list(image_ids))
    name2id = label_to_id(vg_root, split)
    rows: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    for image_id in image_ids:
        for t in range(1, num_levels + 1):
            path = pred_dir / str(image_id) / f"level_{t:02d}.json"
            base = {"image_id": int(image_id), "split": split, "t": t}
            if not path.exists():
                rows.append({**base, "scale": math.nan, "valid": False})
                continue
            pred = json.loads(path.read_text())
            row, metrics = evaluate_state(pred, gts[image_id], name2id)
            rows.append(
                {**base, "scale": pred.get("scale", math.nan), **row, "valid": True}
            )
            for pred_id, n_gt in metrics.counts.items():
                hits.append({"image_id": int(image_id), "t": t, "pred_id": int(pred_id),
                             "n_gt": int(n_gt),
                             f"hits{HIT_K}": int(metrics.hits[HIT_K].get(pred_id, 0))})  # fmt: skip
    return rows, hits
