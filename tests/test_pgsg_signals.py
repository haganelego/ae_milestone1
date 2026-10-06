"""Tests for milestone1.pgsg_signals (PGSG-internal self-consistency signals)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milestone1.pgsg_signals import (  # noqa: E402
    SIGNAL_FIELDS,
    compute_signals,
    normalized_top1,
)


def _graph() -> dict:
    entities = [
        {
            "id": "v1",
            "label_id": 5,
            "score": 0.9,
            "n_mentions": 3,
            "label_dist_topk": [[5, 0.6], [1, 0.2], [2, 0.2]],
        },
        {
            "id": "v2",
            "label_id": 7,
            "score": 0.5,
            "n_mentions": 1,
            "label_dist_topk": [[7, 0.5], [1, 0.5]],
        },
        {
            "id": "v3",
            "label_id": 5,
            "score": 0.7,
            "n_mentions": 2,
            "label_dist_topk": [[5, 1.0]],
        },
    ]
    relations = [
        {
            "sub": "v1",
            "obj": "v3",
            "trp_score": 0.8,
            "pred_dist_topk": [[3, 0.8], [4, 0.2]],
        },
        {
            "sub": "v1",
            "obj": "v2",
            "trp_score": 0.4,
            "pred_dist_topk": [[4, 0.5], [3, 0.5]],
        },
    ]
    return {
        "entities": entities,
        "relations": relations,
        "stats": {"n_bad_inst": 5, "n_raw_triplets": 20},
    }


def test_compute_signals_on_synthetic_graph() -> None:
    s = compute_signals(_graph())
    assert set(s) == set(SIGNAL_FIELDS)
    assert s["n_entities"] == 3 and s["n_unique_labels"] == 2
    assert s["bad_inst_rate"] == pytest.approx(0.25)
    assert s["mean_mentions_top20"] == pytest.approx(2.0)
    # (v1,v3): both >= 2 mentions; (v1,v2): v2 has 1 -> 1/2
    assert s["frac_rel50_agree2"] == pytest.approx(0.5)
    assert s["agreement"] == s["frac_rel50_agree2"]
    assert s["mean_label_maxprob_top20"] == pytest.approx((0.6 + 0.5 + 1.0) / 3)
    assert s["mean_pred_maxprob_top50"] == pytest.approx((0.8 + 0.5) / 2)


def test_compute_signals_empty_graph_is_nan_not_error() -> None:
    s = compute_signals(
        {
            "entities": [],
            "relations": [],
            "stats": {"n_bad_inst": None, "n_raw_triplets": 0},
        }
    )
    assert s["n_entities"] == 0
    for key in (
        "bad_inst_rate",
        "mean_mentions_top20",
        "frac_rel50_agree2",
        "mean_pred_maxprob_top50",
    ):
        assert np.isnan(s[key])


def test_normalized_top1_handles_amplified_scores() -> None:
    assert normalized_top1([[1, 4.0], [2, 1.0]]) == pytest.approx(0.8)
    assert np.isnan(normalized_top1([]))
