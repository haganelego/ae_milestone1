"""Error and saturation analysis utilities for SGG-saturation PoC."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from milestone1.config import resolve
from milestone1.metrics import _to_label_set, _to_triplet_set


# --------------------------------------------------------------------------- #
# Data loading helpers
# --------------------------------------------------------------------------- #


def load_gt(gt_root: Path, image_id: int) -> dict[str, Any]:
    """Load a GT scene graph JSON."""
    with open(gt_root / f"{image_id}.json") as f:
        return json.load(f)


def load_pred(pred_root: Path, image_id: int, level: int) -> dict[str, Any]:
    """Load an EGTR prediction JSON for a given level."""
    with open(pred_root / str(image_id) / f"level_{level:02d}.json") as f:
        return json.load(f)


def load_stop_levels(
    metrics_root: Path,
) -> dict[str, dict[int, int]]:
    """Load stop levels for oracle@0.5, baseline@0.4, and llm.

    Returns {method_name: {image_id: stop_level}}.
    """
    stops: dict[str, dict[int, int]] = {}

    # Oracle stop (use oracle_0.5 column = main_oracle_threshold)
    oracle_path = metrics_root / "oracle_stop.csv"
    if oracle_path.exists():
        df = pd.read_csv(oracle_path)
        stops["oracle"] = dict(zip(df["image_id"], df["oracle_0.5"]))

    # Baseline stop (use baseline_0.4 column)
    baseline_path = metrics_root / "baseline_stop.csv"
    if baseline_path.exists():
        df = pd.read_csv(baseline_path)
        stops["baseline"] = dict(zip(df["image_id"], df["baseline_0.4"]))

    # LLM stop
    llm_path = metrics_root / "llm_stop.csv"
    if llm_path.exists():
        df = pd.read_csv(llm_path)
        stops["llm"] = dict(zip(df["image_id"], df["llm_stop"]))

    return stops


# --------------------------------------------------------------------------- #
# Analysis 1: What was missing at the stop level
# --------------------------------------------------------------------------- #


def compute_missing_at_stop(
    gt: dict[str, Any],
    pred: dict[str, Any],
) -> dict[str, Any]:
    """Compare GT vs prediction at a specific level.

    Returns counts and lists of missing/extra nodes and triplets.
    """
    gt_nodes = _to_label_set(gt["objects"])
    pred_nodes = _to_label_set(pred["objects"])
    gt_trips = _to_triplet_set(gt["relationships"])
    pred_trips = _to_triplet_set(pred["relationships"])

    missing_nodes = gt_nodes - pred_nodes
    extra_nodes = pred_nodes - gt_nodes
    missing_trips = gt_trips - pred_trips
    extra_trips = pred_trips - gt_trips

    return {
        "gt_node_count": len(gt_nodes),
        "pred_node_count": len(pred_nodes),
        "missing_node_count": len(missing_nodes),
        "extra_node_count": len(extra_nodes),
        "missing_nodes": ";".join(sorted(missing_nodes)),
        "extra_nodes": ";".join(sorted(extra_nodes)),
        "gt_triplet_count": len(gt_trips),
        "pred_triplet_count": len(pred_trips),
        "missing_triplet_count": len(missing_trips),
        "extra_triplet_count": len(extra_trips),
        "missing_triplets": ";".join(
            f"{s}-{p}-{o}" for s, p, o in sorted(missing_trips)
        ),
        "extra_triplets": ";".join(
            f"{s}-{p}-{o}" for s, p, o in sorted(extra_trips)
        ),
    }


# --------------------------------------------------------------------------- #
# Analysis 2 & 3: Per-category / per-predicate recall by level
# --------------------------------------------------------------------------- #


def compute_category_recall_by_level(
    image_ids: list[int],
    gt_root: Path,
    pred_root: Path,
    num_levels: int,
) -> pd.DataFrame:
    """Compute per-object-category recall at each level across all images.

    For each (category, level): recall = fraction of images that contain
    that GT category where it was detected (i.e. in the predicted label set).
    """
    rows: list[dict[str, Any]] = []

    for img_id in image_ids:
        gt = load_gt(gt_root, img_id)
        gt_cats = _to_label_set(gt["objects"])

        for level in range(1, num_levels + 1):
            pred = load_pred(pred_root, img_id, level)
            pred_cats = _to_label_set(pred["objects"])

            for cat in gt_cats:
                rows.append(
                    {
                        "image_id": img_id,
                        "level": level,
                        "category": cat,
                        "detected": 1 if cat in pred_cats else 0,
                    }
                )

    df = pd.DataFrame(rows)
    # Aggregate: per (category, level), fraction detected
    agg = (
        df.groupby(["category", "level"])["detected"]
        .mean()
        .reset_index()
        .rename(columns={"detected": "recall"})
    )
    return agg


def compute_predicate_recall_by_level(
    image_ids: list[int],
    gt_root: Path,
    pred_root: Path,
    num_levels: int,
) -> pd.DataFrame:
    """Compute per-predicate recall at each level across all images.

    For each GT triplet, we check if that exact (subject, predicate, object)
    is present in the predicted triplet set. Then aggregate by predicate.
    """
    rows: list[dict[str, Any]] = []

    for img_id in image_ids:
        gt = load_gt(gt_root, img_id)
        gt_trips = _to_triplet_set(gt["relationships"])

        for level in range(1, num_levels + 1):
            pred = load_pred(pred_root, img_id, level)
            pred_trips = _to_triplet_set(pred["relationships"])

            for s, p, o in gt_trips:
                rows.append(
                    {
                        "image_id": img_id,
                        "level": level,
                        "predicate": p,
                        "triplet": f"{s}-{p}-{o}",
                        "detected": 1 if (s, p, o) in pred_trips else 0,
                    }
                )

    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["predicate", "level"])["detected"]
        .mean()
        .reset_index()
        .rename(columns={"detected": "recall"})
    )
    return agg


def identify_hardest(
    recall_df: pd.DataFrame,
    entity_col: str,
    num_levels: int,
    top_k: int = 10,
) -> list[str]:
    """Identify the hardest categories/predicates by final-level recall.

    Lower recall at the final level = harder.
    """
    final = recall_df[recall_df["level"] == num_levels].copy()
    final = final.sort_values("recall")
    return final[entity_col].head(top_k).tolist()


# --------------------------------------------------------------------------- #
# Analysis 4: Image feature extraction
# --------------------------------------------------------------------------- #


def compute_image_features(
    image_id: int,
    gt: dict[str, Any],
    vg_images_dir: Path,
    all_metrics_df: pd.DataFrame,
    oracle_stop: int,
) -> dict[str, Any]:
    """Compute per-image features for difficulty characterization."""
    # Image dimensions
    img_path = vg_images_dir / f"{image_id}.jpg"
    if img_path.exists():
        with Image.open(img_path) as im:
            img_w, img_h = im.size
    else:
        # Fallback: use bbox extents from GT
        all_x2 = [o["bbox"][2] for o in gt["objects"]]
        all_y2 = [o["bbox"][3] for o in gt["objects"]]
        img_w = max(all_x2) if all_x2 else 500
        img_h = max(all_y2) if all_y2 else 500

    img_area = img_w * img_h

    # GT object areas (from bbox)
    obj_areas = []
    for obj in gt["objects"]:
        x1, y1, x2, y2 = obj["bbox"]
        obj_areas.append(max(0, (x2 - x1)) * max(0, (y2 - y1)))

    gt_node_cats = _to_label_set(gt["objects"])
    gt_trip_set = _to_triplet_set(gt["relationships"])

    mean_obj_area_frac = (
        (sum(obj_areas) / len(obj_areas) / img_area) if obj_areas else 0.0
    )
    small_threshold = 0.01 * img_area
    frac_small = (
        sum(1 for a in obj_areas if a < small_threshold) / len(obj_areas)
        if obj_areas
        else 0.0
    )

    # Final-level metrics
    img_metrics = all_metrics_df[all_metrics_df["image_id"] == image_id]
    final_row = img_metrics[img_metrics["level"] == 20]
    final_sg = float(final_row["sg_score"].values[0]) if len(final_row) > 0 else 0.0
    final_node_r = (
        float(final_row["node_recall"].values[0]) if len(final_row) > 0 else 0.0
    )
    final_trip_r = (
        float(final_row["triplet_recall"].values[0]) if len(final_row) > 0 else 0.0
    )

    return {
        "image_id": image_id,
        "img_width": img_w,
        "img_height": img_h,
        "img_area": img_area,
        "num_gt_objects": len(gt["objects"]),
        "num_gt_object_categories": len(gt_node_cats),
        "num_gt_relationships": len(gt["relationships"]),
        "num_gt_unique_triplets": len(gt_trip_set),
        "mean_obj_area_frac": round(mean_obj_area_frac, 4),
        "frac_small_objects": round(frac_small, 4),
        "final_sg_score": final_sg,
        "final_node_recall": final_node_r,
        "final_triplet_recall": final_trip_r,
        "oracle_stop_level": oracle_stop,
    }


# --------------------------------------------------------------------------- #
# Analysis 5: Node vs Relation saturation gap
# --------------------------------------------------------------------------- #


def compute_saturation_gap(
    image_id: int,
    all_metrics_df: pd.DataFrame,
    threshold_frac: float = 0.9,
) -> dict[str, Any]:
    """Compute the gap between node and triplet saturation levels.

    Saturation = level where recall first reaches threshold_frac * max(recall).
    """
    img_df = all_metrics_df[all_metrics_df["image_id"] == image_id].sort_values(
        "level"
    )

    max_node_r = img_df["node_recall"].max()
    max_trip_r = img_df["triplet_recall"].max()

    node_sat_level = 20
    trip_sat_level = 20

    if max_node_r > 0:
        target = threshold_frac * max_node_r
        hits = img_df[img_df["node_recall"] >= target]
        if len(hits) > 0:
            node_sat_level = int(hits["level"].min())

    if max_trip_r > 0:
        target = threshold_frac * max_trip_r
        hits = img_df[img_df["triplet_recall"] >= target]
        if len(hits) > 0:
            trip_sat_level = int(hits["level"].min())

    return {
        "image_id": image_id,
        "max_node_recall": round(max_node_r, 4),
        "max_triplet_recall": round(max_trip_r, 4),
        "node_90pct_level": node_sat_level,
        "triplet_90pct_level": trip_sat_level,
        "gap": trip_sat_level - node_sat_level,
    }
