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
        results.append({
            "image_id": image_id,
            "num_objects": num_objects,
            "num_relationships": num_relationships,
        })
    return results


def load_gt_scene_graph(
    vg_root: str | Path, split: str, image_id: int
) -> dict[str, Any]:
    """Load GT scene graph for a single image.

    Returns {
        image_id: int,
        objects: [{object_id, label, bbox: [x1,y1,x2,y2]}],
        relationships: [{subject_label, predicate, object_label}]
    }
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
        cls_name = id2name[ann["category_id"] - 1]
        # Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]
        x, y, w, h = ann["bbox"]
        bbox = [x, y, x + w, y + h]
        objects.append({
            "object_id": i,
            "label": cls_name,
            "bbox": bbox,
        })

    # Relationships
    rel_list = rels_split.get(str(image_id), [])
    relationships = []
    for subj_idx, obj_idx, pred_id_1indexed in rel_list:
        if subj_idx < len(anns) and obj_idx < len(anns):
            subj_name = id2name[anns[subj_idx]["category_id"] - 1]
            obj_name = id2name[anns[obj_idx]["category_id"] - 1]
            pred_name = pred_names[pred_id_1indexed - 1]
            relationships.append({
                "subject_label": subj_name,
                "predicate": pred_name,
                "object_label": obj_name,
            })

    return {
        "image_id": image_id,
        "objects": objects,
        "relationships": relationships,
    }
