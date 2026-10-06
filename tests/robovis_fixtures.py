"""Synthetic PGSG-contract graphs for ROBOVIS2027 check/judge tests and smoke runs.

Contents are hand-built from the dev images 2412319 (motorcycle), 2319328
(street sign / tree) and 2322043 (surfer) so F1 smoke runs see a matching image.
"""

from __future__ import annotations

from typing import Any

Entity = tuple[str, float, tuple[float, float, float, float]]
Relation = tuple[int, str, int, float]

SYNTHETIC_SPECS: dict[int, dict[str, Any]] = {
    2412319: {
        "base_size": [384, 308],
        "orig_size": [500, 401],
        "entities": [
            ("motorcycle", 0.91, (5.0, 10.0, 380.0, 300.0)),
            ("wheel", 0.84, (250.0, 140.0, 382.0, 285.0)),
            ("wheel", 0.79, (0.0, 160.0, 85.0, 295.0)),
            ("seat", 0.72, (0.0, 95.0, 110.0, 140.0)),
            ("tree", 0.66, (20.0, 0.0, 240.0, 90.0)),
            ("man", 0.41, (0.0, 55.0, 22.0, 105.0)),
            ("sign", 0.33, (70.0, 75.0, 90.0, 98.0)),
            ("light", 0.38, (140.0, 35.0, 165.0, 62.0)),
            ("street", 0.52, (0.0, 230.0, 384.0, 308.0)),
        ],
        "relations": [
            (0, "has", 1, 0.44),
            (0, "has", 2, 0.41),
            (3, "on", 0, 0.37),
            (0, "parked on", 8, 0.29),
            (7, "on", 0, 0.18),
            (5, "near", 4, 0.07),
            (6, "near", 4, 0.05),
            (4, "behind", 0, 0.04),
        ],
    },
    2319328: {
        "base_size": [384, 288],
        "orig_size": [500, 375],
        "entities": [
            ("tree", 0.88, (0.0, 0.0, 250.0, 288.0)),
            ("sign", 0.81, (283.0, 165.0, 384.0, 195.0)),
            ("pole", 0.63, (330.0, 190.0, 360.0, 288.0)),
            ("letter", 0.45, (300.0, 170.0, 365.0, 190.0)),
            ("wire", 0.39, (225.0, 0.0, 285.0, 250.0)),
            ("sign", 0.47, (290.0, 195.0, 384.0, 288.0)),
        ],
        "relations": [
            (1, "on", 2, 0.39),
            (3, "on", 1, 0.33),
            (5, "on", 2, 0.21),
            (0, "near", 2, 0.12),
            (4, "above", 1, 0.06),
            (4, "near", 0, 0.05),
        ],
    },
    2322043: {
        "base_size": [384, 256],
        "orig_size": [500, 333],
        "entities": [
            ("man", 0.89, (145.0, 100.0, 190.0, 192.0)),
            ("surfboard", 0.58, (150.0, 185.0, 200.0, 198.0)),
            ("wave", 0.83, (0.0, 80.0, 384.0, 200.0)),
            ("short", 0.61, (150.0, 140.0, 178.0, 160.0)),
            ("hand", 0.31, (150.0, 122.0, 160.0, 135.0)),
            ("arm", 0.36, (150.0, 112.0, 165.0, 132.0)),
            ("hair", 0.29, (168.0, 102.0, 180.0, 110.0)),
        ],
        "relations": [
            (0, "on", 1, 0.46),
            (0, "wearing", 3, 0.43),
            (0, "riding", 2, 0.21),
            (0, "has", 5, 0.19),
            (0, "has", 4, 0.15),
            (0, "has", 6, 0.11),
            (1, "in", 2, 0.06),
        ],
    },
}
SYNTHETIC_IMAGE_IDS = tuple(SYNTHETIC_SPECS)


def make_graph(
    image_id: int,
    level: int = 20,
    scale: float = 1.0,
    split: str = "dev",
    entities: list[Entity] | None = None,
    relations: list[Relation] | None = None,
    base_size: list[int] | None = None,
    orig_size: list[int] | None = None,
) -> dict[str, Any]:
    """Build one graph dict following the shared graph JSON contract."""
    spec = SYNTHETIC_SPECS.get(image_id, SYNTHETIC_SPECS[SYNTHETIC_IMAGE_IDS[0]])
    entities = entities if entities is not None else spec["entities"]
    relations = relations if relations is not None else spec["relations"]
    base_size = base_size or spec["base_size"]
    orig_size = orig_size or spec["orig_size"]
    ratio = orig_size[0] / base_size[0]
    ents = []
    for idx, (label, score, box) in enumerate(entities):
        ents.append(
            {
                "id": f"v{idx + 1}",
                "label_id": idx,
                "label": label,
                "score": score,
                "label_dist_topk": [[idx, score]],
                "raw_tokens": label,
                "box_xyxy_orig": [round(v * ratio, 1) for v in box],
                "box_xyxy_base": list(box),
                "segment_ids": [0],
                "n_mentions": 1,
            }
        )
    rels = []
    for rank, (sub, pred, obj, trp) in enumerate(relations):
        rels.append(
            {
                "id": f"e{rank + 1}",
                "sub": f"v{sub + 1}",
                "obj": f"v{obj + 1}",
                "pred_id": rank,
                "pred": pred,
                "pred_score": min(1.0, trp * 2),
                "trp_score": trp,
                "pred_dist_topk": [[rank, trp]],
                "rank": rank,
                "segment_id": 0,
            }
        )
    return {
        "image_id": image_id,
        "split": split,
        "level": level,
        "scale": scale,
        "base_size": base_size,
        "orig_size": orig_size,
        "coord_scale": [ratio, ratio],
        "seed": 123456,
        "model": {"checkpoint": "synthetic", "sha256": "synthetic"},
        "generation": {},
        "entities": ents,
        "relations": rels,
        "stats": {"synthetic": True},
    }
