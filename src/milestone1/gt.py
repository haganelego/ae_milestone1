"""Ground truth scene graph loading from Visual Genome annotations."""

import json
from pathlib import Path
from typing import Any

from pycocotools.coco import COCO


def _load_coco_and_rels(
    vg_root: str | Path, split: str
) -> tuple[COCO, dict[str, Any], dict[str, Any]]:
    """Load COCO annotations and relationship data."""
    vg_root = Path(vg_root)
    # Suppress COCO stdout
    import io
    import sys

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        coco = COCO(str(vg_root / f"{split}.json"))
    finally:
        sys.stdout = old_stdout

    with open(vg_root / "rel.json") as f:
        rel_data = json.load(f)

    return coco, rel_data, rel_data[split]


def count_gt_objects_and_relationships(
    vg_root: str | Path, split: str
) -> list[dict[str, Any]]:
    """Count GT objects and relationships per image.

    Returns list of {image_id, num_objects, num_relationships}.
    """
    coco, rel_data, rels_split = _load_coco_and_rels(vg_root, split)

    results = []
    for img_info in coco.dataset["images"]:
        image_id = img_info["id"]
        ann_ids = coco.getAnnIds(imgIds=image_id)
        num_objects = len(ann_ids)
        rel_list = rels_split.get(str(image_id), [])
        num_relationships = len(rel_list)
        results.append(
            {
                "image_id": image_id,
                "num_objects": num_objects,
                "num_relationships": num_relationships,
            }
        )
    return results


def load_gt_scene_graph(
    vg_root: str | Path, split: str, image_id: int
) -> dict[str, Any]:
    """Load GT scene graph for a single image.

    Returns {
        image_id: int,
        objects: [{object_id, label, class_id, bbox: [x1,y1,x2,y2]}],
        relationships: [{subject_id, subject_label, subject_class_id, subject_bbox,
                         predicate, predicate_id,
                         object_id, object_label, object_class_id, object_bbox}]
    }

    object_id on relationships indexes into `objects`. class_id / predicate_id are
    0-indexed integer labels; bbox is [x1,y1,x2,y2]. These fields make standard
    SGDet R@K (IoU-based GT-vs-prediction triplet matching) computable post-hoc.
    """
    coco, rel_data, rels_split = _load_coco_and_rels(vg_root, split)

    # Object categories (0-indexed)
    id2name = {cid - 1: info["name"] for cid, info in coco.cats.items()}

    # Predicate names (0-indexed)
    pred_names = rel_data["rel_categories"][1:]

    # Objects
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)

    objects = []
    for i, ann in enumerate(anns):
        cls_id = ann["category_id"] - 1
        cls_name = id2name[cls_id]
        # Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]
        x, y, w, h = ann["bbox"]
        bbox = [x, y, x + w, y + h]
        objects.append(
            {
                "object_id": i,
                "label": cls_name,
                "class_id": cls_id,
                "bbox": bbox,
            }
        )

    # Relationships. subject_id / object_id index into `objects`; bbox + class_id
    # + predicate_id are carried so SGDet R@K can match GT triplets to predictions.
    rel_list = rels_split.get(str(image_id), [])
    relationships = []
    for subj_idx, obj_idx, pred_id_1indexed in rel_list:
        if subj_idx < len(anns) and obj_idx < len(anns):
            relationships.append(
                {
                    "subject_id": subj_idx,
                    "subject_label": objects[subj_idx]["label"],
                    "subject_class_id": objects[subj_idx]["class_id"],
                    "subject_bbox": objects[subj_idx]["bbox"],
                    "predicate": pred_names[pred_id_1indexed - 1],
                    "predicate_id": pred_id_1indexed - 1,
                    "object_id": obj_idx,
                    "object_label": objects[obj_idx]["label"],
                    "object_class_id": objects[obj_idx]["class_id"],
                    "object_bbox": objects[obj_idx]["bbox"],
                }
            )

    return {
        "image_id": image_id,
        "objects": objects,
        "relationships": relationships,
    }
