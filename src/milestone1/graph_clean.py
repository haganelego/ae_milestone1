"""Canonical "cleaned observed graph" for the ROBOVIS2027 stop study.

PGSG emits ~100 entities / ~300 relations per state, many of them near-duplicate.
``clean_graph`` produces the official observed graph G:

1. merge entities with the same label and box IoU > ``merge_iou`` (continuous
   IoU on ``box_xyxy_base``): greedy in descending ``score``, each entity joins
   the FIRST already-kept entity it overlaps, so the highest-scoring member is
   kept (same rule as check_generator.merge_entities, implemented separately);
   the kept entity's ``n_mentions`` becomes the sum and ``segment_ids`` the union
   over its merged members;
2. remap relations to kept entities, drop self-loops, drop duplicate
   (sub, pred_id, obj) keeping the highest ``trp_score``;
3. keep the top-``top_relations`` relations by ``trp_score`` and only the
   entities they reference.

The output follows the graph JSON contract (entity ids are the raw ids of the
kept entities; relation ``rank`` is re-numbered) with
``stats["cleaned"]`` recording the parameters and counts.
"""

from __future__ import annotations

from typing import Any

DEFAULT_MERGE_IOU = 0.5
DEFAULT_TOP_RELATIONS = 20


def box_iou(a: list[float] | None, b: list[float] | None) -> float:
    """Continuous IoU of two xyxy boxes; 0 if either is missing or degenerate."""
    if not a or not b:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_entities(
    entities: list[dict[str, Any]], merge_iou: float
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Greedy same-label merge. Returns (entity id -> kept id, kept id -> members)."""
    kept: list[dict[str, Any]] = []
    alias: dict[str, str] = {}
    members: dict[str, list[str]] = {}
    for ent in sorted(entities, key=lambda e: -float(e["score"])):
        match = next(
            (
                k
                for k in kept
                if k["label_id"] == ent["label_id"]
                and box_iou(k["box_xyxy_base"], ent["box_xyxy_base"]) > merge_iou
            ),
            None,
        )
        target = ent if match is None else match
        if match is None:
            kept.append(ent)
        alias[ent["id"]] = target["id"]
        members.setdefault(target["id"], []).append(ent["id"])
    return alias, members


def _merged_entity(
    ent: dict[str, Any], member_ids: list[str], by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    group = [by_id[i] for i in member_ids]
    mentions = [m["n_mentions"] for m in group if m.get("n_mentions") is not None]
    segs = [m["segment_ids"] for m in group if m.get("segment_ids") is not None]
    return {
        **ent,
        "n_mentions": sum(mentions) if mentions else ent.get("n_mentions"),
        "segment_ids": sorted({s for seg in segs for s in seg})
        if segs
        else ent.get("segment_ids"),
    }


def clean_graph(
    graph: dict[str, Any],
    merge_iou: float = DEFAULT_MERGE_IOU,
    top_relations: int = DEFAULT_TOP_RELATIONS,
) -> dict[str, Any]:
    """Return the cleaned observed graph (new dict; the input is not modified)."""
    entities = graph["entities"]
    by_id = {e["id"]: e for e in entities}
    alias, members = merge_entities(entities, merge_iou)

    n_self = 0
    best: dict[tuple[str, int, str], dict[str, Any]] = {}
    for rel in graph["relations"]:
        sub, obj = alias[rel["sub"]], alias[rel["obj"]]
        if sub == obj:
            n_self += 1
            continue
        key = (sub, int(rel["pred_id"]), obj)
        if key not in best or float(rel["trp_score"]) > float(best[key]["trp_score"]):
            best[key] = {**rel, "sub": sub, "obj": obj}
    deduped = sorted(best.values(), key=lambda r: -float(r["trp_score"]))
    kept_rels = [{**r, "rank": i} for i, r in enumerate(deduped[:top_relations])]

    referenced = {r["sub"] for r in kept_rels} | {r["obj"] for r in kept_rels}
    kept_ents = [
        _merged_entity(e, members[e["id"]], by_id)
        for e in entities
        if e["id"] in referenced
    ]
    stats = {
        **graph.get("stats", {}),
        "n_relations_kept": len(kept_rels),
        "cleaned": {
            "merge_iou": merge_iou,
            "top_relations": top_relations,
            "n_entities_raw": len(entities),
            "n_entities_after_merge": len(members),
            "n_entities_kept": len(kept_ents),
            "n_relations_raw": len(graph["relations"]),
            "n_self_loops_dropped": n_self,
            "n_relations_after_dedup": len(deduped),
        },
    }
    return {**graph, "entities": kept_ents, "relations": kept_rels, "stats": stats}
