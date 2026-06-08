"""Scene graph evaluation metrics with category-set semantics."""

from typing import Any


def _to_label_set(objects: list[dict[str, Any]]) -> set[str]:
    """Extract unique label set from objects."""
    return {obj["label"] for obj in objects}


def _to_triplet_set(
    relationships: list[dict[str, Any]],
) -> set[tuple[str, str, str]]:
    """Extract unique triplet set from relationships."""
    return {
        (rel["subject_label"], rel["predicate"], rel["object_label"])
        for rel in relationships
    }


def node_precision_recall_f1(
    pred_objects: list[dict[str, Any]],
    gt_objects: list[dict[str, Any]],
) -> tuple[float, float, float]:
    """Compute node (object label) precision, recall, F1.

    Uses category-set semantics: dedup to sets of labels.
    """
    pred_set = _to_label_set(pred_objects)
    gt_set = _to_label_set(gt_objects)

    if not pred_set and not gt_set:
        return 1.0, 1.0, 1.0
    if not pred_set:
        return 0.0, 0.0, 0.0
    if not gt_set:
        return 0.0, 0.0, 0.0

    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def triplet_precision_recall_f1(
    pred_relationships: list[dict[str, Any]],
    gt_relationships: list[dict[str, Any]],
) -> tuple[float, float, float]:
    """Compute triplet (subject, predicate, object) precision, recall, F1.

    Uses category-set semantics: dedup to sets of triplets.
    """
    pred_set = _to_triplet_set(pred_relationships)
    gt_set = _to_triplet_set(gt_relationships)

    if not pred_set and not gt_set:
        return 1.0, 1.0, 1.0
    if not pred_set:
        return 0.0, 0.0, 0.0
    if not gt_set:
        return 0.0, 0.0, 0.0

    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def sg_score(
    node_recall: float,
    triplet_recall: float,
    node_weight: float = 0.5,
    relation_weight: float = 0.5,
) -> float:
    """Compute weighted scene graph score."""
    return node_weight * node_recall + relation_weight * triplet_recall


def mean_confidence(items: list[dict[str, Any]], key: str = "confidence") -> float:
    """Compute mean confidence from a list of items."""
    if not items:
        return 0.0
    return sum(item[key] for item in items) / len(items)
