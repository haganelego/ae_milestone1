"""Tests for the torch-free parts of milestone1.pgsg_adapter."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milestone1 import pgsg_adapter as pa  # noqa: E402

N_OBJ = 6
N_REL = 4
OBJ_NAMES = [f"obj{i}" for i in range(N_OBJ)]
REL_NAMES = [f"rel{i}" for i in range(N_REL)]
MODEL_INFO = {"checkpoint": "vg_sgg.pth", "sha256": "x"}
GENERATION = {
    "max_length": 1024,
    "min_length": 768,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "num_segments": 3,
}


class FakeBoxList:
    """Minimal stand-in for lavis BoxList (bbox + get_field)."""

    def __init__(self, bbox: np.ndarray, **fields: np.ndarray) -> None:
        self.bbox = bbox
        self._fields = fields

    def get_field(self, name: str) -> np.ndarray:
        return self._fields[name]


def _dist(n: int, hot: int, value: float, fill: float = 0.01) -> np.ndarray:
    d = np.full(n, fill)
    d[hot] = value
    return d


BOX_A = np.array([10.0, 20.0, 110.0, 220.0])  # base coords
BOX_B = np.array([50.0, 60.0, 150.0, 160.0])
BOX_C = np.array([0.0, 0.0, 30.0, 40.0])


def _boxlist() -> FakeBoxList:
    return FakeBoxList(
        np.stack([BOX_A, BOX_B, BOX_C]),
        pred_labels=np.array([2, 4, 2]),
        pred_scores=np.array([0.9, 0.8, 0.3]),
        pred_dists=np.stack(
            [_dist(N_OBJ, 2, 0.9), _dist(N_OBJ, 4, 0.8), _dist(N_OBJ, 2, 0.3)]
        ),
        rel_pair_idxs=np.array([[0, 1], [2, 1]]),
        pred_rel_label=np.array([3, 1]),
        pred_rel_score=np.array([0.7, 0.2]),
        pred_rel_trp_score=np.array([0.5, 0.05]),
        pred_rel_dist=np.stack([_dist(N_REL, 3, 0.7, fill=0.02), _dist(N_REL, 1, 0.2)]),
    )


def _meta() -> pa.StateMeta:
    return pa.StateMeta(
        image_id=7,
        split="dev",
        level=3,
        scale=0.0685,
        base_size=(384, 288),
        orig_size=(500, 400),
        coord_scale=(500 / 384, 400 / 288),
        seed=11,
    )


def _provenance() -> pa.Provenance:
    """3 parsed instances, 1 row each: (A rel3 B), (C rel1 B), (A' rel3 B) where A'
    is a second mention merged into A by NMS (same label, high GIoU)."""
    box_a2 = BOX_A + np.array([1.0, 0.0, 1.0, 0.0])
    sub_boxes = np.stack([BOX_A, BOX_C, box_a2])
    obj_boxes = np.stack([BOX_B, BOX_B, BOX_B])
    nms_boxes = np.concatenate([sub_boxes, obj_boxes])  # rows 0..2 subj, then obj
    nms_labels = np.array([2, 2, 2, 4, 4, 4])
    return pa.Provenance(
        segment_lengths=(10, 10, 10),
        n_rel_markers=5,
        sub_text=("cup", "mug", "cup"),
        obj_text=("table", "table", "desk"),
        sub_pos=np.array([0, 12, 21]),
        obj_pos=np.array([4, 16, 25]),
        rel_pos=np.array([3, 15, 24]),
        sub_boxes=sub_boxes,
        obj_boxes=obj_boxes,
        pred_dists=np.stack(
            [
                _dist(N_REL, 3, 0.35),
                _dist(N_REL, 1, 0.1),
                _dist(N_REL, 3, 0.3, fill=0.02),
            ]
        ),
        nms_boxes=nms_boxes,
        nms_labels=nms_labels,
        nms_keep=np.array([0, 3, 1]),  # -> entities A, B, C (BoxList order)
    )


def _graph() -> dict:
    pred = pa.boxlist_to_arrays(_boxlist(), n_instances=3)
    return pa.build_graph(
        pred, _provenance(), _meta(), OBJ_NAMES, REL_NAMES, MODEL_INFO, GENERATION, 1.5
    )


def test_state_seed_is_deterministic_and_in_range() -> None:
    a = pa.state_seed([1, 2, 3, 4], 5, 42)
    assert a == pa.state_seed([1, 2, 3, 4], 5, 42)
    assert 0 <= a < pa.SEED_MODULUS
    assert a != pa.state_seed([1, 2, 3, 4], 6, 42)
    assert a != pa.state_seed([1, 2, 3, 4], 5, 43)


def test_boxlist_to_arrays_with_no_instances_returns_empty() -> None:
    pred = pa.boxlist_to_arrays(_boxlist(), n_instances=0)
    assert pred.boxes.shape == (0, 4)
    assert pred.rel_pairs.shape == (0, 2)
    assert pred.rel_dists.shape == (0, N_REL)


def test_build_graph_entities_and_relations_follow_boxlist_order() -> None:
    g = _graph()
    assert [e["id"] for e in g["entities"]] == ["v1", "v2", "v3"]
    assert [e["label"] for e in g["entities"]] == ["obj2", "obj4", "obj2"]
    rel = g["relations"]
    assert (rel[0]["sub"], rel[0]["obj"], rel[0]["pred"]) == ("v1", "v2", "rel3")
    assert (rel[1]["sub"], rel[1]["obj"], rel[1]["pred_id"]) == ("v3", "v2", 1)
    assert [r["rank"] for r in rel] == [0, 1]
    assert rel[0]["trp_score"] == pytest.approx(0.5)
    assert rel[0]["pred_dist_topk"][0] == [3, pytest.approx(0.7)]
    assert g["entities"][0]["label_dist_topk"][0] == [2, pytest.approx(0.9)]
    assert len(g["entities"][0]["label_dist_topk"]) == pa.TOPK_DIST


def test_build_graph_maps_boxes_to_original_coordinates_per_axis() -> None:
    g = _graph()
    ent = g["entities"][0]
    sx, sy = 500 / 384, 400 / 288
    assert ent["box_xyxy_base"] == pytest.approx(list(BOX_A))
    assert ent["box_xyxy_orig"] == pytest.approx([10 * sx, 20 * sy, 110 * sx, 220 * sy])
    assert g["base_size"] == [384, 288] and g["orig_size"] == [500, 400]


def test_build_graph_provenance_fields() -> None:
    g = _graph()
    a, b, c = g["entities"]
    assert a["raw_tokens"] == "cup"
    assert a["n_mentions"] == 2  # instance 0 and the NMS-merged instance 2
    assert a["segment_ids"] == [0, 2]
    assert b["n_mentions"] == 3 and b["segment_ids"] == [0, 1, 2]
    assert c["raw_tokens"] == "mug" and c["segment_ids"] == [1]
    # relation 0 (A rel3 B): candidates instances 0 and 2; only instance 2's
    # predicate scores equal the relation's (non-amplified) entries -> segment 2.
    # relation 1 (C rel1 B) -> instance 1 -> segment 1.
    assert [r["segment_id"] for r in g["relations"]] == [2, 1]
    assert g["stats"] == {
        "n_bad_inst": 2,
        "n_raw_triplets": 3,
        "n_relations_kept": 2,
        "elapsed_s": 1.5,
    }


def test_build_graph_without_provenance_sets_nulls() -> None:
    pred = pa.boxlist_to_arrays(_boxlist(), n_instances=3)
    g = pa.build_graph(
        pred, None, _meta(), OBJ_NAMES, REL_NAMES, MODEL_INFO, GENERATION, 0.0
    )
    assert g["entities"][0]["raw_tokens"] is None
    assert g["entities"][0]["segment_ids"] is None
    assert g["relations"][0]["segment_id"] is None
    assert g["stats"]["n_bad_inst"] is None


def test_segment_ids_of_uses_cumulative_boundaries() -> None:
    got = pa.segment_ids_of(np.array([0, 9, 10, 19, 20, 29]), [10, 10, 10])
    assert got.tolist() == [0, 0, 1, 1, 2, 2]


def test_assign_rows_splits_exact_duplicate_instances() -> None:
    # two identical instances (same boxes) with different sampled labels, 1 row each
    prov = _provenance()
    dup = pa.Provenance(
        **{
            **prov.__dict__,
            "sub_boxes": np.stack([BOX_A, BOX_A]),
            "obj_boxes": np.stack([BOX_B, BOX_B]),
            "sub_text": ("x", "x"),
            "obj_text": ("y", "y"),
            "sub_pos": np.array([0, 10]),
            "obj_pos": np.array([2, 12]),
            "rel_pos": np.array([1, 11]),
            "pred_dists": np.zeros((2, N_REL)),
            "nms_boxes": np.stack([BOX_A, BOX_A, BOX_B, BOX_B]),
            "nms_labels": np.array([2, 3, 4, 4]),
            "nms_keep": np.array([0, 1, 2]),
        },
    )
    rows = pa.assign_rows_to_instances(dup)
    assert sorted(rows.tolist()) == [0, 1]


def test_check_graph_consistency_passes_on_consistent_input() -> None:
    pred = pa.boxlist_to_arrays(_boxlist(), n_instances=3)
    assert pa.check_graph_consistency(pred, _provenance()) == []
