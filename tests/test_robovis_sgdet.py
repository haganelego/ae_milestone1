"""Tests for milestone1.robovis_sgdet (graph-constrained SGDet matching rules)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milestone1 import robovis_sgdet as sg  # noqa: E402

SUB_BOX = [10.0, 10.0, 109.0, 109.0]
OBJ_BOX = [200.0, 200.0, 299.0, 299.0]
FAR_BOX = [400.0, 400.0, 450.0, 450.0]
MAN, SHIRT, WEARING, HAS = 77, 99, 31, 20


def _gt(rels: list[tuple[int, int, int, list[float], list[float]]]) -> sg.TripletSet:
    return sg.TripletSet(
        np.array([[s, p, o] for s, p, o, _, _ in rels]),
        np.array([[*sb, *ob] for _, _, _, sb, ob in rels], dtype=float),
    )


def _graph(rels: list[tuple[int, int, int, list[float], list[float]]]) -> dict:
    entities, relations = [], []
    for r, (s, p, o, sb, ob) in enumerate(rels):
        for label, box in ((s, sb), (o, ob)):
            entities.append(
                {"id": f"v{len(entities) + 1}", "label_id": label, "box_xyxy_orig": box}
            )
        relations.append(
            {
                "sub": f"v{len(entities) - 1}",
                "obj": f"v{len(entities)}",
                "pred_id": p,
                "rank": r,
            }
        )
    return {"entities": entities, "relations": relations}


GT_ONE = [(MAN, WEARING, SHIRT, SUB_BOX, OBJ_BOX)]


def test_exact_prediction_matches() -> None:
    m = sg.evaluate_image(_graph(GT_ONE), _gt(GT_ONE))
    assert m.recall == {20: 1.0, 50: 1.0, 100: 1.0}
    assert m.hits[50] == {WEARING: 1} and m.counts == {WEARING: 1}


def test_same_labels_different_box_does_not_match() -> None:
    pred = [(MAN, WEARING, SHIRT, SUB_BOX, FAR_BOX)]
    assert sg.evaluate_image(_graph(pred), _gt(GT_ONE)).recall[50] == 0.0


def test_inverted_subject_object_does_not_match() -> None:
    pred = [(SHIRT, WEARING, MAN, OBJ_BOX, SUB_BOX)]
    assert sg.evaluate_image(_graph(pred), _gt(GT_ONE)).recall[50] == 0.0


def test_wrong_predicate_does_not_match() -> None:
    pred = [(MAN, HAS, SHIRT, SUB_BOX, OBJ_BOX)]
    assert sg.evaluate_image(_graph(pred), _gt(GT_ONE)).recall[100] == 0.0


def test_duplicate_predictions_do_not_double_count() -> None:
    gt = GT_ONE + [(MAN, HAS, SHIRT, SUB_BOX, FAR_BOX)]
    pred = [GT_ONE[0]] * 5
    m = sg.evaluate_image(_graph(pred), _gt(gt))
    assert m.recall[50] == pytest.approx(0.5)
    assert m.hits[50] == {WEARING: 1}


def test_top_k_cutoff_uses_rank_order() -> None:
    filler = [(MAN, HAS, SHIRT, FAR_BOX, FAR_BOX)] * 20
    m = sg.evaluate_image(_graph(filler + GT_ONE), _gt(GT_ONE))
    assert m.recall[20] == 0.0
    assert m.recall[50] == 1.0


def test_iou_uses_plus_one_pixel_convention() -> None:
    # boxes [0,0,9,9] and [5,0,14,9]: +1 areas 100 each, inter 5*10=50 -> 1/3
    iou = sg.iou_plus_one(np.array([0, 0, 9, 9]), np.array([[5, 0, 14, 9]]))
    assert iou[0, 0] == pytest.approx(50 / 150)


def test_iou_threshold_is_inclusive_at_half() -> None:
    # +1 convention: [0,0,9,9] vs [0,0,9,19] -> inter 100, union 200 -> 0.5
    gt = [(MAN, WEARING, SHIRT, [0, 0, 9, 9], OBJ_BOX)]
    pred = [(MAN, WEARING, SHIRT, [0, 0, 9, 19], OBJ_BOX)]
    assert sg.evaluate_image(_graph(pred), _gt(gt)).recall[50] == 1.0


def test_no_prediction_and_no_gt() -> None:
    assert sg.evaluate_image(None, _gt(GT_ONE)).recall[50] == 0.0
    empty = sg.TripletSet(np.zeros((0, 3), dtype=int), np.zeros((0, 8)))
    m = sg.evaluate_image(_graph(GT_ONE), empty)
    assert m.recall[50] == 0.0 and m.n_gt_rel == 0


def test_mean_recall_official_divides_by_all_predicates() -> None:
    gt = GT_ONE + [(MAN, HAS, SHIRT, SUB_BOX, FAR_BOX)]
    m = sg.evaluate_image(_graph(GT_ONE), _gt(gt))
    expected = 1.0 / (sg.NUM_PREDICATES + sg.OFFICIAL_MR_EPS)  # wearing 1.0, has 0.0
    assert sg.mean_recall_official([m], 50) == pytest.approx(expected)
    assert sg.mean_recall_present([m], 50) == pytest.approx(0.5)
