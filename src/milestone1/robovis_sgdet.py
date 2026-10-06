"""Per-image SGDet recall (graph constraint) for ROBOVIS2027 graph JSONs.

Numpy re-implementation of the rules in Pix2Grp's evaluator
(``lavis/tasks/evaluation/vg_sgg_eval_tools.py``: ``SGRecall.calculate_recall``,
``_compute_pred_matches``, ``SGMeanRecall``), applied in GT (original image)
coordinates:

* a prediction matches a GT relation iff (subject class, predicate, object
  class) are equal AND subject IoU >= 0.5 AND object IoU >= 0.5; IoU uses the
  +1 pixel convention of ``boxlist_iou`` (``TO_REMOVE = 1``);
* predictions are taken in their stored rank order (``trp_score`` desc), top-K;
* R@K = |union of GT indices matched by the top-K predictions| / #GT, so each GT
  relation counts at most once and duplicate predictions cannot double count;
* mR: per image and predicate, hits / count over the GT relations of that
  predicate; ``mean_recall_official`` then averages each predicate over the
  images containing it and divides the sum by (#predicates + 0.001), treating
  absent predicates as 0 (exactly as ``SGMeanRecall.calculate_mean_recall``).

Validated against the official functions on the same saved predictions by
``scripts/28_robovis_stage_metrics.py --validate-official``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

TOPK = (20, 50, 100)
IOU_THRESHOLD = 0.5
PIXEL_OFFSET = 1.0  # boxlist_iou TO_REMOVE
NUM_PREDICATES = 50
OFFICIAL_MR_EPS = 0.001  # SGMeanRecall divides by (num_rel + 0.001)
BOX_DIM = 4


@dataclass(frozen=True)
class TripletSet:
    """Triplets (sub_cls, pred, obj_cls) with paired boxes (sub xyxy | obj xyxy)."""

    triplets: np.ndarray  # (N, 3) int
    boxes: np.ndarray  # (N, 8) float

    def __len__(self) -> int:
        return len(self.triplets)


@dataclass(frozen=True)
class ImageMetrics:
    recall: dict[int, float]  # K -> R@K (0.0 when no GT)
    n_gt_rel: int
    hits: dict[int, dict[int, int]]  # K -> predicate id -> #GT hit
    counts: dict[int, int]  # predicate id -> #GT relations


def _empty_triplets() -> TripletSet:
    return TripletSet(np.zeros((0, 3), dtype=int), np.zeros((0, 2 * BOX_DIM)))


def gt_triplets(scene_graph: dict[str, Any]) -> TripletSet:
    """From ``milestone1.gt.load_gt_scene_graph`` output (0-based ids, xyxy)."""
    rels = scene_graph["relationships"]
    if not rels:
        return _empty_triplets()
    trip = np.array(
        [
            [r["subject_class_id"], r["predicate_id"], r["object_class_id"]]
            for r in rels
        ],
        dtype=int,
    )
    boxes = np.array(
        [[*r["subject_bbox"], *r["object_bbox"]] for r in rels], dtype=float
    )
    return TripletSet(trip, boxes)


def pred_triplets(graph: dict[str, Any], box_key: str = "box_xyxy_orig") -> TripletSet:
    """From a graph JSON, in stored rank order."""
    rels = graph["relations"]
    if not rels:
        return _empty_triplets()
    ents = {e["id"]: e for e in graph["entities"]}
    trip = np.array(
        [
            [ents[r["sub"]]["label_id"], r["pred_id"], ents[r["obj"]]["label_id"]]
            for r in rels
        ],
        dtype=int,
    )
    boxes = np.array(
        [[*ents[r["sub"]][box_key], *ents[r["obj"]][box_key]] for r in rels],
        dtype=float,
    )
    return TripletSet(trip, boxes)


def iou_plus_one(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes with the +1 pixel convention; (m, n)."""
    a = np.asarray(a, dtype=float).reshape(-1, BOX_DIM)
    b = np.asarray(b, dtype=float).reshape(-1, BOX_DIM)
    area_a = (a[:, 2] - a[:, 0] + PIXEL_OFFSET) * (a[:, 3] - a[:, 1] + PIXEL_OFFSET)
    area_b = (b[:, 2] - b[:, 0] + PIXEL_OFFSET) * (b[:, 3] - b[:, 1] + PIXEL_OFFSET)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt + PIXEL_OFFSET, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def match_predictions(
    gt: TripletSet, pred: TripletSet, iou_threshold: float = IOU_THRESHOLD
) -> list[list[int]]:
    """For each prediction, the GT relation indices it matches (graph constraint)."""
    pred_to_gt: list[list[int]] = [[] for _ in range(len(pred))]
    if len(gt) == 0 or len(pred) == 0:
        return pred_to_gt
    label_eq = (gt.triplets[:, None, :] == pred.triplets[None, :, :]).all(-1)  # (G, P)
    for g in np.where(label_eq.any(1))[0]:
        cand = np.where(label_eq[g])[0]
        sub_iou = iou_plus_one(gt.boxes[g, :BOX_DIM], pred.boxes[cand, :BOX_DIM])[0]
        obj_iou = iou_plus_one(gt.boxes[g, BOX_DIM:], pred.boxes[cand, BOX_DIM:])[0]
        for p in cand[(sub_iou >= iou_threshold) & (obj_iou >= iou_threshold)]:
            pred_to_gt[int(p)].append(int(g))
    return pred_to_gt


def matched_gt_at_k(pred_to_gt: Sequence[Sequence[int]], k: int) -> set[int]:
    """Union of GT indices hit by the top-k predictions."""
    return {g for hits in pred_to_gt[:k] for g in hits}


def evaluate_image(
    graph: dict[str, Any] | None, gt: TripletSet, topk: Iterable[int] = TOPK
) -> ImageMetrics:
    """Per-image R@K and per-predicate hit counts. ``graph=None`` = no prediction."""
    topk = tuple(topk)
    counts: dict[int, int] = {}
    for p in gt.triplets[:, 1]:
        counts[int(p)] = counts.get(int(p), 0) + 1
    pred = pred_triplets(graph) if graph is not None else _empty_triplets()
    pred_to_gt = match_predictions(gt, pred)
    recall, hits = {}, {}
    for k in topk:
        matched = matched_gt_at_k(pred_to_gt, k)
        recall[k] = len(matched) / len(gt) if len(gt) else 0.0
        per_pred: dict[int, int] = {}
        for g in matched:
            p = int(gt.triplets[g, 1])
            per_pred[p] = per_pred.get(p, 0) + 1
        hits[k] = per_pred
    return ImageMetrics(recall, len(gt), hits, counts)


def mean_recall_official(
    metrics: Sequence[ImageMetrics], k: int, num_predicates: int = NUM_PREDICATES
) -> float:
    """PGSG/Motifs mR@K over a set of images (absent predicates count as 0)."""
    per_pred: dict[int, list[float]] = {}
    for m in metrics:
        for p, c in m.counts.items():
            per_pred.setdefault(p, []).append(m.hits[k].get(p, 0) / c)
    total = sum(float(np.mean(v)) for v in per_pred.values())
    return total / (num_predicates + OFFICIAL_MR_EPS)


def mean_recall_present(metrics: Sequence[ImageMetrics], k: int) -> float:
    """mR@K averaged only over predicates present in the GT of these images."""
    per_pred: dict[int, list[float]] = {}
    for m in metrics:
        for p, c in m.counts.items():
            per_pred.setdefault(p, []).append(m.hits[k].get(p, 0) / c)
    if not per_pred:
        return 0.0
    return float(np.mean([np.mean(v) for v in per_pred.values()]))
