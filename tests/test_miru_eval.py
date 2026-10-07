"""Tests for the MIRU method registry (miru_methods) and EGTR GT metrics (miru_gt).

Run: uv run --project egtr --with pytest python -m pytest tests/test_miru_eval.py -q
Data-dependent parity tests skip when the MIRU outputs are not present.
"""

import copy
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import load_config, resolve  # noqa: E402
from milestone1.miru_gt import (  # noqa: E402
    egtr_label_sets,
    egtr_to_graph,
    evaluate_state,
)
from milestone1.miru_methods import (  # noqa: E402
    StopParams,
    consecutive_k_stop,
    fuse_series,
    fused_scores,
    load_existing_method_stops,
    load_object_confidence,
    load_scores,
    registered_score_methods,
    score_method_stops,
)
from milestone1.miru_oracle import (  # noqa: E402
    baseline_stops,
    object_confidence,
    topk_oracle,
)

START, MAX = 5, 20
THR = 0.7
PARAMS = StopParams(START, MAX, THR, 0.5, 0.5, 0.5, (1, 2))


# --------------------------------------------------------------------------
# Stop rule (same semantics as scripts/13_aggregate_evaluation.py)
# --------------------------------------------------------------------------


def test_consecutive_k_stop_rules() -> None:
    scores = {5: 0.8, 6: 0.2, 7: 0.75, 8: 0.9}
    assert consecutive_k_stop(scores, THR, 1, START, MAX) == 5
    assert consecutive_k_stop(scores, THR, 2, START, MAX) == 8
    assert consecutive_k_stop({}, THR, 1, START, MAX) == MAX  # never -> 20
    # A missing level counts as 0 and breaks the run.
    assert consecutive_k_stop({7: 0.9, 9: 0.9, 10: 0.9}, THR, 2, START, MAX) == 10
    # No window may reach below start_level.
    assert consecutive_k_stop({4: 0.9, 5: 0.9}, THR, 2, START, MAX) == MAX


def test_load_scores_skips_failed_and_malformed(tmp_path: Path) -> None:
    d = tmp_path / "1"
    d.mkdir()
    (d / "level_05.json").write_text(json.dumps({"saturation_score": 0.8}))
    (d / "level_06.json").write_text(
        json.dumps({"saturation_score": 0.9, "failure": True})
    )
    (d / "level_07.json").write_text("{not json")
    (d / "level_08.json").write_text(json.dumps({"saturation_score": True}))
    assert load_scores(tmp_path, 1, START, MAX) == {5: 0.8}


def test_fused_scores_zero_when_input_missing() -> None:
    conf = {(1, 5): 0.4, (1, 6): 0.6}
    fused = fused_scores({5: 0.8, 7: 0.9}, conf, 1, PARAMS)
    assert fused[5] == pytest.approx(0.6)
    assert fused[6] == 0.0 and fused[7] == 0.0  # no score / no confidence


def _write_scores(root: Path, image_id: int, scores: dict[int, float]) -> None:
    d = root / str(image_id)
    d.mkdir(parents=True)
    for lvl, sc in scores.items():
        (d / f"level_{lvl:02d}.json").write_text(json.dumps({"saturation_score": sc}))


def test_score_method_stops_names_and_coverage(tmp_path: Path) -> None:
    _write_scores(tmp_path, 1, {5: 0.8, 6: 0.8})
    stops, cov = score_method_stops(
        "x", tmp_path, [1], PARAMS, {(1, 5): 0.2, (1, 6): 0.2}
    )
    assert stops["x_k1"][1] == 5 and stops["x_k2"][1] == 6
    # Hybrid: 0.5*0.2 + 0.5*0.8 = 0.5 >= 0.5 at levels 5 and 6.
    assert stops["hybrid_x_k1"][1] == 5 and stops["hybrid_x_k2"][1] == 6
    assert cov == pytest.approx(2 / (MAX - START + 1))


def test_registry_threshold_override_and_missing_dir(tmp_path: Path) -> None:
    _write_scores(tmp_path / "a", 1, {5: 0.75, 6: 0.75})
    cfg = copy.deepcopy(load_config())
    cfg["evaluation_extra"] = {
        "score_methods": {
            "a": str(tmp_path / "a"),
            "a_strict": {"dir": str(tmp_path / "a"), "threshold": 0.8},
            "gone": str(tmp_path / "missing"),
        },
        "hybrid_methods": [],
        "k_options": [1],
    }
    res = registered_score_methods(cfg, [1], tmp_path)
    assert res.stops["a_k1"][1] == 5
    assert res.stops["a_strict_k1"][1] == MAX
    assert "gone_k1" not in res.stops


# --------------------------------------------------------------------------
# Parity with the existing MIRU results (data-dependent)
# --------------------------------------------------------------------------

METRICS = REPO_ROOT / "outputs" / "metrics"
ABLATION = REPO_ROOT / "outputs" / "llm_ablation"


@pytest.mark.skipif(
    not (METRICS / "hybrid_stop.csv").exists(), reason="MIRU outputs absent"
)
def test_recomputed_llm_and_hybrid_stops_match_miru_csvs() -> None:
    cfg = load_config()
    ids = json.loads(resolve(cfg["paths"]["selected_images"]).read_text())
    params = StopParams.from_config(cfg)
    conf = load_object_confidence(METRICS / "all_metrics.csv")
    existing = load_existing_method_stops(METRICS, ids)
    for size in ("4b", "9b"):
        stops, cov = score_method_stops(
            size, ABLATION / f"qwen3.5-{size}_h5_rel_on_conf_on", ids, params, conf
        )
        assert cov == 1.0
        for k in (1, 2):
            assert stops[f"{size}_k{k}"] == existing[f"llm_sgg_{size}_h5_k{k}"]
            assert stops[f"hybrid_{size}_k{k}"] == existing[f"hybrid_{size}_k{k}"]


# --------------------------------------------------------------------------
# EGTR -> GT metrics
# --------------------------------------------------------------------------

NAME2ID = {"man": 77, "shirt": 111, "dog": 39}


def _gt() -> dict[str, Any]:
    return {
        "objects": [
            {"class_id": 77, "bbox": [10, 10, 100, 200]},
            {"class_id": 111, "bbox": [20, 40, 90, 120]},
            {"class_id": 39, "bbox": [200, 150, 260, 220]},
        ],
        "relationships": [
            {"subject_class_id": 77, "predicate_id": 47, "object_class_id": 111,
             "subject_bbox": [10, 10, 100, 200], "object_bbox": [20, 40, 90, 120]},
        ],
    }  # fmt: skip


def _pred(shift: float = 0.0) -> dict[str, Any]:
    return {
        "objects": [{"label": "man", "confidence": 0.5}, {"label": "shirt", "confidence": 0.4}],
        "relationships": [
            {"subject_id": 3, "object_id": 7, "subject_class_id": 77, "object_class_id": 111,
             "predicate_id": 47, "triplet_score": 0.2,
             "subject_bbox": [10 + shift, 10, 100 + shift, 200],
             "object_bbox": [20 + shift, 40, 90 + shift, 120]},
            {"subject_id": 3, "object_id": 9, "subject_class_id": 77, "object_class_id": 39,
             "predicate_id": 20, "triplet_score": 0.5,
             "subject_bbox": [10, 10, 100, 200], "object_bbox": [200, 150, 260, 220]},
        ],
    }  # fmt: skip


def test_egtr_to_graph_ranks_by_triplet_score_and_keys_by_query() -> None:
    g = egtr_to_graph(_pred())
    assert [r["pred_id"] for r in g["relations"]] == [20, 47]
    assert {e["id"] for e in g["entities"]} == {"q3", "q7", "q9"}


def test_evaluate_state_matches_boxes_and_labels() -> None:
    row, _ = evaluate_state(_pred(), _gt(), NAME2ID)
    assert row["R50"] == pytest.approx(1.0) and row["n_gt_rel"] == 1
    # Nodes {man, shirt} of GT {man, shirt, dog}; triplet hit -> 0.5*2/3 + 0.5*1.
    assert row["relsg_gt"] == pytest.approx(0.5 * 2 / 3 + 0.5)
    assert row["n_pred_rel"] == 2


def test_evaluate_state_box_shift_breaks_recall_not_relsg() -> None:
    row, _ = evaluate_state(_pred(shift=300.0), _gt(), NAME2ID)
    assert row["R50"] == 0.0
    assert row["relsg_gt"] == pytest.approx(0.5 * 2 / 3 + 0.5)


def test_egtr_label_sets() -> None:
    nodes, trips = egtr_label_sets(_pred(), NAME2ID)
    assert nodes == {77, 111}
    assert trips == {(77, 47, 111), (77, 20, 39)}


# --------------------------------------------------------------------------
# Late fusion (zero-call)
# --------------------------------------------------------------------------


def test_fuse_series_mean_min_and_missing_levels() -> None:
    a = {1: {5: 0.9, 6: 0.6, 7: 0.8}}
    b = {1: {5: 0.5, 6: 0.9}}  # level 7 missing in b -> left out (scores 0)
    assert fuse_series([a, b], "mean") == {
        1: {5: pytest.approx(0.7), 6: pytest.approx(0.75)}
    }
    assert fuse_series([a, b], "min") == {1: {5: 0.5, 6: 0.6}}


def test_registry_fusion_methods(tmp_path: Path) -> None:
    _write_scores(tmp_path / "img", 1, {5: 0.9, 6: 0.9, 7: 0.9})
    _write_scores(tmp_path / "txt", 1, {5: 0.5, 6: 0.6, 7: 0.9})
    cfg = copy.deepcopy(load_config())
    cfg["evaluation_extra"] = {
        "score_methods": {"img": str(tmp_path / "img"), "txt": str(tmp_path / "txt")},
        "hybrid_methods": [],
        "fusion_methods": {
            "fuse_mean": {"sources": ["img", "txt"], "op": "mean", "hybrid": False},
            "fuse_min": {"sources": ["img", "txt"], "op": "min", "hybrid": False},
            "fuse_gone": {"sources": ["img", "nope"], "op": "mean", "hybrid": False},
        },
        "k_options": [1, 2],
    }
    res = registered_score_methods(cfg, [1], tmp_path)
    # mean: .70, .75, .90 -> k1 at 5, k2 at 6; min: .5, .6, .9 -> k1 at 7, k2 never.
    assert (res.stops["fuse_mean_k1"][1], res.stops["fuse_mean_k2"][1]) == (5, 6)
    assert (res.stops["fuse_min_k1"][1], res.stops["fuse_min_k2"][1]) == (7, MAX)
    assert "fuse_gone_k1" not in res.stops
    assert "hybrid_fuse_mean_k1" not in res.stops


# --------------------------------------------------------------------------
# Held-out support: oracle / confidence / baseline from predictions, selection
# --------------------------------------------------------------------------


def _write_pred(root: Path, image_id: int, level: int, labels: list[str],
                trips: list[tuple[str, str, str]], conf: float) -> None:  # fmt: skip
    d = root / str(image_id)
    d.mkdir(parents=True, exist_ok=True)
    pred = {
        "objects": [{"label": lab, "confidence": conf} for lab in labels],
        "relationships": [{"subject_label": s, "predicate": p, "object_label": o,
                           "triplet_score": 0.1} for s, p, o in trips],
    }  # fmt: skip
    (d / f"level_{level:02d}.json").write_text(json.dumps(pred))


def test_topk_oracle_conf_and_baseline_from_predictions(tmp_path: Path) -> None:
    max_level = 4
    full = ["man", "shirt"]
    trip = [("man", "wearing", "shirt")]
    _write_pred(tmp_path, 1, 1, [], [], 0.0)
    _write_pred(tmp_path, 1, 2, ["man"], [], 0.2)  # core = node 1/2 only
    _write_pred(tmp_path, 1, 3, full, [], 0.4)  # 0.5*1 + 0.5*0 = 0.5
    _write_pred(tmp_path, 1, 4, full, trip, 0.36)  # 1.0 (L20 reference)
    orc = topk_oracle(tmp_path, [1], max_level)
    assert orc.core_scores[1][2] == pytest.approx(0.25)
    assert orc.core_scores[1][3] == pytest.approx(0.5)
    assert orc.stops[1] == 4
    conf = object_confidence(tmp_path, [1], max_level)
    assert conf[(1, 1)] == 0.0 and conf[(1, 3)] == 0.4
    assert baseline_stops(conf, [1], 0.35, max_level) == {1: 3}
    assert baseline_stops(conf, [1], 0.9, max_level) == {1: max_level}


@pytest.mark.skipif(
    not (METRICS / "all_metrics.csv").exists(), reason="MIRU outputs absent"
)
def test_prediction_based_quantities_match_miru_csvs() -> None:
    cfg = load_config()
    ids = json.loads(resolve(cfg["paths"]["selected_images"]).read_text())
    pred = resolve(cfg["paths"]["egtr_predictions"])
    conf = object_confidence(pred, ids, MAX)
    assert conf == load_object_confidence(METRICS / "all_metrics.csv")
    assert (
        baseline_stops(conf, ids, 0.35, MAX)
        == load_existing_method_stops(METRICS, ids)["baseline"]
    )
    canon = {}
    with open(METRICS / "topk_oracle" / "oracle_topk_per_image.csv") as f:
        for r in csv.DictReader(f):
            canon[int(r["image_id"])] = int(r["stop_c09"])
    assert topk_oracle(pred, ids, MAX).stops == canon


def _script42() -> Any:
    spec = importlib.util.spec_from_file_location(
        "miru42", REPO_ROOT / "scripts" / "42_miru_gt_eval.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_heldout_selection_and_completeness(tmp_path: Path) -> None:
    s42 = _script42()
    src = tmp_path / "robovis.json"
    src.write_text(json.dumps([{"image_id": i, "split": sp} for i, sp in
                               [(1, "test"), (2, "test"), (3, "dev"), (4, "test")]]))  # fmt: skip
    excl = tmp_path / "miru.json"
    excl.write_text(json.dumps([2]))
    listed = tmp_path / "listed.json"
    block = {"selection": {"from": str(src), "split": "test", "exclude": str(excl)}}
    cfg = {"paths": {"selected_images": str(listed)}}
    assert s42.resolve_image_ids(cfg, block) == [1, 4]  # list file absent -> derived
    listed.write_text(json.dumps([4, 1]))
    assert s42.resolve_image_ids(cfg, block) == [1, 4]  # same set -> ok
    listed.write_text(json.dumps([1, 2]))
    with pytest.raises(SystemExit):
        s42.resolve_image_ids(cfg, block)
    _write_pred(tmp_path / "pred", 1, 1, ["man"], [], 0.5)
    with pytest.raises(SystemExit):
        s42.require_complete_predictions(tmp_path / "pred", [1], 2)
    s42.require_complete_predictions(tmp_path / "pred", [1], 1)


def test_registry_block_and_conf_override(tmp_path: Path) -> None:
    _write_scores(tmp_path / "t", 1, {5: 0.2, 6: 0.2})
    block = {"score_methods": {"t": str(tmp_path / "t")}, "hybrid_methods": ["t"],
             "k_options": [2]}  # fmt: skip
    conf = {(1, 5): 0.9, (1, 6): 0.9}  # hybrid = 0.5*0.9 + 0.5*0.2 = 0.55 >= 0.5
    res = registered_score_methods(load_config(), [1], tmp_path, block, conf)
    assert set(res.stops) == {"t_k2", "hybrid_t_k2"}
    assert (res.stops["t_k2"][1], res.stops["hybrid_t_k2"][1]) == (MAX, 6)
