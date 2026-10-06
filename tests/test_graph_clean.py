"""Tests for milestone1.graph_clean and the precision-aware metrics."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milestone1 import robovis_sgdet as sg  # noqa: E402
from milestone1.graph_clean import box_iou, clean_graph  # noqa: E402

BOX = [0.0, 0.0, 100.0, 100.0]
NEAR = [5.0, 0.0, 105.0, 100.0]  # IoU 95/105 with BOX
FAR = [300.0, 300.0, 350.0, 350.0]
CUP, TABLE, MAN = 3, 7, 77
ON, NEAR_P, HAS = 21, 30, 20


def _ent(
    eid: str,
    label: int,
    score: float,
    box: list[float],
    mentions: int = 1,
    segs: list[int] | None = None,
) -> dict:
    return {
        "id": eid,
        "label_id": label,
        "label": str(label),
        "score": score,
        "box_xyxy_base": box,
        "box_xyxy_orig": box,
        "n_mentions": mentions,
        "segment_ids": segs or [0],
        "label_dist_topk": [[label, score]],
        "raw_tokens": None,
    }


def _rel(rid: str, sub: str, pred: int, obj: str, trp: float) -> dict:
    return {
        "id": rid,
        "sub": sub,
        "obj": obj,
        "pred_id": pred,
        "pred": str(pred),
        "pred_score": trp,
        "trp_score": trp,
        "pred_dist_topk": [[pred, trp]],
        "rank": 0,
        "segment_id": 0,
    }


def _graph() -> dict:
    ents = [
        _ent("v1", CUP, 0.9, BOX, 2, [0, 1]),
        _ent("v2", CUP, 0.5, NEAR, 3, [1, 5]),  # merges into v1
        _ent("v3", TABLE, 0.8, FAR),
        _ent("v4", TABLE, 0.4, BOX),  # same box as v1 but other label -> kept
    ]
    rels = [
        _rel("e1", "v2", ON, "v3", 0.9),  # -> (v1, on, v3)
        _rel("e2", "v1", ON, "v3", 0.7),  # duplicate after remap, lower score
        _rel("e3", "v1", NEAR_P, "v2", 0.6),  # self-loop after merge
        _rel("e4", "v4", HAS, "v3", 0.3),
        _rel("e5", "v3", HAS, "v4", 0.2),
    ]
    return {"entities": ents, "relations": rels, "stats": {"n_relations_kept": 5}}


def test_box_iou_continuous() -> None:
    assert box_iou(BOX, NEAR) == pytest.approx(9500 / 10500)
    assert box_iou(BOX, FAR) == 0.0


def test_clean_graph_merges_remaps_and_dedups() -> None:
    g = clean_graph(_graph(), merge_iou=0.5, top_relations=20)
    triples = [(r["sub"], r["pred_id"], r["obj"]) for r in g["relations"]]
    assert triples == [("v1", ON, "v3"), ("v4", HAS, "v3"), ("v3", HAS, "v4")]
    assert g["relations"][0]["trp_score"] == pytest.approx(0.9)  # max kept
    assert [r["rank"] for r in g["relations"]] == [0, 1, 2]
    v1 = next(e for e in g["entities"] if e["id"] == "v1")
    assert v1["n_mentions"] == 5 and v1["segment_ids"] == [0, 1, 5]
    assert {e["id"] for e in g["entities"]} == {"v1", "v3", "v4"}
    c = g["stats"]["cleaned"]
    assert c["n_entities_after_merge"] == 3 and c["n_self_loops_dropped"] == 1
    assert c["n_relations_after_dedup"] == 3 and g["stats"]["n_relations_kept"] == 3


def test_clean_graph_top_k_keeps_only_referenced_entities() -> None:
    g = clean_graph(_graph(), merge_iou=0.5, top_relations=1)
    assert len(g["relations"]) == 1
    assert {e["id"] for e in g["entities"]} == {"v1", "v3"}


def test_clean_graph_does_not_mutate_input() -> None:
    raw = _graph()
    clean_graph(raw)
    assert len(raw["entities"]) == 4 and raw["relations"][0]["sub"] == "v2"


def test_det_match_counts_each_gt_once() -> None:
    gt_labels = np.array([CUP, TABLE])
    gt_boxes = np.array([BOX, FAR])
    graph = {
        "entities": [
            _ent("v1", CUP, 0.9, BOX),
            _ent("v2", CUP, 0.8, NEAR),
            _ent("v3", MAN, 0.7, FAR),
        ]
    }
    recall, precision = sg.det_match(graph, gt_labels, gt_boxes, top_k=20)
    assert recall == pytest.approx(0.5)  # cup found once, table missed
    assert precision == pytest.approx(1 / 3)  # duplicate cup is a false positive


def test_det_match_respects_top_k() -> None:
    gt_labels, gt_boxes = np.array([TABLE]), np.array([FAR])
    graph = {"entities": [_ent("v1", CUP, 0.9, BOX), _ent("v2", TABLE, 0.1, FAR)]}
    assert sg.det_match(graph, gt_labels, gt_boxes, top_k=1) == (0.0, 0.0)
    assert sg.det_match(graph, gt_labels, gt_boxes, top_k=2)[0] == pytest.approx(1.0)


def test_triplet_precision_penalizes_duplicates() -> None:
    gt = sg.TripletSet(np.array([[CUP, ON, TABLE]]), np.array([[*BOX, *FAR]]))
    ents = [_ent("v1", CUP, 0.9, BOX), _ent("v2", TABLE, 0.9, FAR)]
    rels = [_rel("e1", "v1", ON, "v2", 0.9), _rel("e2", "v1", ON, "v2", 0.8)]
    out = sg.triplet_precision_f1({"entities": ents, "relations": rels}, gt)
    assert out["trip_prec20"] == pytest.approx(0.5)
    assert out["F1_20"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    empty = sg.triplet_precision_f1({"entities": [], "relations": []}, gt)
    assert np.isnan(empty["trip_prec20"]) and empty["F1_20"] == 0.0
