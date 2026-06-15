"""EGTR model wrapper for scene graph generation inference."""

from typing import Any

import torch
from PIL import Image

from milestone1.config import setup_egtr_path
from milestone1.vocab import load_obj_names, load_pred_names

# Ensure EGTR is importable
setup_egtr_path()

from glob import glob

from lib.pytorch_misc import argsort_desc
from model.deformable_detr import (
    DeformableDetrConfig,
    DeformableDetrFeatureExtractor,
)
from model.egtr import DetrForSceneGraphGeneration
from util.box_ops import rescale_bboxes


class EgtrModel:
    """Wrapper around EGTR for single-image scene graph inference."""

    def __init__(
        self,
        artifact_path: str,
        architecture: str,
        vg_root: str,
        object_conf_threshold: float = 0.3,
        relation_conf_threshold: float = 0.2,
        top_k_objects: int = 20,
        top_k_relationships: int = 30,
    ) -> None:
        self.object_conf_threshold = object_conf_threshold
        self.relation_conf_threshold = relation_conf_threshold
        self.top_k_objects = top_k_objects
        self.top_k_relationships = top_k_relationships

        # Load vocabularies
        self.obj_names = load_obj_names(vg_root)
        self.pred_names = load_pred_names(vg_root)

        # Load model
        config = DeformableDetrConfig.from_pretrained(artifact_path)
        self.num_labels = config.num_labels  # 150

        self.model = DetrForSceneGraphGeneration.from_pretrained(
            architecture, config=config, ignore_mismatched_sizes=True
        )

        # Load finetuned checkpoint
        ckpt_path = sorted(
            glob(f"{artifact_path}/checkpoints/epoch=*.ckpt"),
            key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
        )[-1]
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        # Strip "model." prefix from PyTorch Lightning keys
        for k in list(state_dict.keys()):
            state_dict[k[6:]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        self.model.cuda()
        self.model.eval()

        # Feature extractor
        self.feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
            architecture, size=800, max_size=1333
        )

    def predict(self, pil_image: Image.Image) -> dict[str, Any]:
        """Run inference on a single PIL image.

        Returns {
            objects: [{object_id, query_id, label, confidence, bbox: [x1,y1,x2,y2]}],
            relationships: [{subject_id, subject_label, subject_class_id, subject_bbox,
                             predicate, predicate_id,
                             object_id, object_label, object_class_id, object_bbox,
                             confidence, subject_conf, object_conf, triplet_score}]
        }

        subject_id / object_id are query indices. *_bbox / *_class_id / predicate_id
        / triplet_score are persisted so standard SGDet R@K (IoU triplet matching vs
        GT) is computable post-hoc, independent of the confidence-gated objects list.
        """
        img_w, img_h = pil_image.size

        # Preprocess
        encoding = self.feature_extractor(images=pil_image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].cuda()
        pixel_mask = encoding["pixel_mask"].cuda()

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                output_attentions=False,
                output_attention_states=True,
                output_hidden_states=True,
            )

        # Extract objects
        logits = outputs.logits[0]  # (200, 150)
        pred_boxes = outputs.pred_boxes[0]  # (200, 4) cx,cy,w,h normalized
        probs = logits.softmax(-1)
        obj_scores, pred_classes = torch.max(probs[:, : self.num_labels], dim=-1)

        # Convert boxes to absolute [x1, y1, x2, y2]
        abs_boxes = rescale_bboxes(pred_boxes.cpu(), (img_w, img_h))  # (200, 4)

        # Filter and rank objects by confidence
        valid_mask = obj_scores > self.object_conf_threshold
        valid_indices = torch.where(valid_mask)[0]

        # Sort by confidence descending, take top_k
        if len(valid_indices) > 0:
            sorted_idx = obj_scores[valid_indices].argsort(descending=True)
            valid_indices = valid_indices[sorted_idx[: self.top_k_objects]]
        else:
            # Fallback: take top_k by score regardless of threshold
            sorted_idx = obj_scores.argsort(descending=True)
            valid_indices = sorted_idx[: self.top_k_objects]

        objects = []
        for rank, idx in enumerate(valid_indices):
            idx_int = idx.item()
            objects.append(
                {
                    "object_id": rank,
                    "query_id": idx_int,
                    "label": self.obj_names[pred_classes[idx_int].item()],
                    "confidence": round(obj_scores[idx_int].item(), 4),
                    "bbox": [round(x, 1) for x in abs_boxes[idx_int].tolist()],
                }
            )

        # Extract relationships (Method A: single predicate per pair).
        # Proper top-K SGG usage: rank ALL query pairs by triplet score and take
        # the top-K directly, WITHOUT requiring both endpoints to pass the object
        # confidence threshold. This avoids dropping valid relations whose endpoint
        # (e.g. a "person"/"man" box at conf ~0.25) falls just below
        # object_conf_threshold, which previously zeroed out relations for ~23% of
        # images. The object LIST above stays confidence-gated so that
        # mean_object_confidence (node track / baseline / hybrid) is unchanged;
        # only the relationship track is decoupled.
        pred_rel = torch.clamp(outputs.pred_rel[0], 0.0, 1.0)  # (200,200,50)
        pred_conn = torch.clamp(outputs.pred_connectivity[0], 0.0, 1.0)  # (200,200,1)
        pred_rel = pred_rel * pred_conn  # gate by connectivity

        # Object score outer product (weights triplets by both endpoints' conf)
        sub_ob_scores = torch.outer(obj_scores, obj_scores)  # (200, 200)
        n_queries = sub_ob_scores.shape[0]
        sub_ob_scores[torch.arange(n_queries), torch.arange(n_queries)] = 0.0

        # Rank triplets over all pairs; buffer extra in case of the conf floor
        triplet_scores = pred_rel.max(-1)[0] * sub_ob_scores  # (200, 200)
        top_pairs = argsort_desc(triplet_scores.cpu().numpy())[
            : self.top_k_relationships * 3
        ]

        relationships = []
        for s_q, o_q in top_pairs:
            s_q = int(s_q)
            o_q = int(o_q)

            best_pred = pred_rel[s_q, o_q].argmax().item()
            rel_score = pred_rel[s_q, o_q, best_pred].item()

            if rel_score < self.relation_conf_threshold:
                continue

            # subject_id/object_id are query indices (endpoints may be below the
            # object_conf_threshold and thus absent from the exported objects list).
            # bbox / class-id / triplet_score are persisted on every relationship so
            # that standard SGDet R@K (IoU-based triplet matching against GT) can be
            # computed post-hoc directly from these JSONs, independent of the
            # confidence-gated `objects` list. abs_boxes is indexed by query id.
            relationships.append(
                {
                    "subject_id": s_q,
                    "subject_label": self.obj_names[pred_classes[s_q].item()],
                    "subject_class_id": int(pred_classes[s_q].item()),
                    "subject_bbox": [round(x, 1) for x in abs_boxes[s_q].tolist()],
                    "predicate": self.pred_names[best_pred],
                    "predicate_id": int(best_pred),
                    "object_id": o_q,
                    "object_label": self.obj_names[pred_classes[o_q].item()],
                    "object_class_id": int(pred_classes[o_q].item()),
                    "object_bbox": [round(x, 1) for x in abs_boxes[o_q].tolist()],
                    "confidence": round(rel_score, 4),
                    "subject_conf": round(obj_scores[s_q].item(), 4),
                    "object_conf": round(obj_scores[o_q].item(), 4),
                    "triplet_score": round(triplet_scores[s_q, o_q].item(), 6),
                }
            )

            if len(relationships) >= self.top_k_relationships:
                break

        return {"objects": objects, "relationships": relationships}
