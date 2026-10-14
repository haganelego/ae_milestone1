"""Exploratory probe (not pre-registered): select multi-object GQA val questions.

Writes the v27-style manifest (questions_dev.jsonl, gt_dev.json) for the
probe run ``probe_multi_rel`` so that the unchanged v27 CLI stages can run on it,
plus ``selection.jsonl`` (types + program + GT, never sent to the model).

Types (5 each, seed 20261008, <= 1 question per image, v27 dev images excluded):
  a_relO        query x rel, detailed relO (final query on the related object)
  b_relS        query x rel, detailed relS
  c_choose_rel  choose x rel
  d_verify_rel  verify x rel
  e_logical     logical with an and/or op over two selected objects
  f_compare     compare with two selected objects
All types require >= 2 object references in the program (select / relate /
verify rel / choose rel ops, non-existent "(-)" referents included).
Usage: uv run python scripts/gqa_probe_multi_rel_select.py
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

SEED = 20261008
PER_TYPE = 5
REPO = Path("/home/kanaoka/usr/github/milestone_1")
GQA_VAL = Path("/home/kanaoka/usr/dataset/gqa/val_balanced_questions.json")
IMAGES = Path("/home/kanaoka/usr/dataset/gqa/images")
V27_DATA = REPO / "data/processed/gqa_stop/visapp_gqa_v27"
DATA_ROOT = REPO / "data/processed/gqa_stop/probe_multi_rel"
OUT_ROOT = REPO / "outputs/gqa_stop/probe_multi_rel"
OBJECT_OPS = ("select", "relate", "verify rel", "choose rel")
MIN_OBJECT_REFS = 2


def ops(q: dict[str, Any]) -> list[str]:
    return [op["operation"] for op in q["semantic"]]


def object_refs(q: dict[str, Any]) -> int:
    return sum(1 for op in ops(q) if op in OBJECT_OPS)


def n_relate(q: dict[str, Any]) -> int:
    return sum(1 for op in ops(q) if op.startswith("relate"))


def two_selects(q: dict[str, Any]) -> bool:
    return sum(1 for op in ops(q) if op == "select") >= MIN_OBJECT_REFS


TYPE_FILTERS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "a_relO": lambda q: q["types"]["detailed"] == "relO"
    and q["semantic"][-1]["operation"] == "query",
    "b_relS": lambda q: q["types"]["detailed"] == "relS"
    and q["semantic"][-1]["operation"] == "query",
    "c_choose_rel": lambda q: q["types"]["structural"] == "choose"
    and q["types"]["semantic"] == "rel",
    "d_verify_rel": lambda q: q["types"]["structural"] == "verify"
    and q["types"]["semantic"] == "rel",
    "e_logical": lambda q: q["types"]["structural"] == "logical"
    and any(op in ("and", "or") for op in ops(q))
    and two_selects(q),
    "f_compare": lambda q: q["types"]["structural"] == "compare" and two_selects(q),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dev_exclusions() -> tuple[set[str], set[str]]:
    rows = [json.loads(x) for x in (V27_DATA / "questions_dev.jsonl").open()]
    ids = {i for r in rows for i in r["group_image_ids"]}
    hashes = json.loads((V27_DATA / "dev_source_image_sha256.json").read_text())
    return ids, {h for i, h in hashes.items() if i in ids}


def main() -> None:
    questions = json.loads(GQA_VAL.read_text())
    dev_ids, dev_hashes = dev_exclusions()
    rng = random.Random(SEED)
    used: set[str] = set()
    picked: list[dict[str, Any]] = []
    for label, keep in TYPE_FILTERS.items():
        pool = sorted(
            k
            for k, q in questions.items()
            if keep(q) and object_refs(q) >= MIN_OBJECT_REFS
        )
        rng.shuffle(pool)
        got = 0
        for k in pool:
            if got == PER_TYPE:
                break
            q = questions[k]
            image = str(q["imageId"])
            if image in used or image in dev_ids:
                continue
            if sha256_file(IMAGES / f"{image}.jpg") in dev_hashes:
                continue
            used.add(image)
            got += 1
            picked.append({"question_id": k, "type": label, "pool": len(pool), **q})
        print(label, "pool", len(pool), "picked", got)
    manifest, gt, selection = [], {}, []
    for r in picked:
        manifest.append(
            {
                "question_id": r["question_id"],
                "image_id": str(r["imageId"]),
                "group_id": str(r["imageId"]),
                "group_image_ids": [str(r["imageId"])],
                "split": "dev",
                "question": r["question"],
                "attribute": r["type"],  # probe type label (per-type eval columns)
                "relate_count": n_relate(r),
                "operation_count": len(r["semantic"]),
                "detailed_type": r["types"]["detailed"],
                "cell": r["type"],
                "gt_ref": f"gqa_val_balanced:{r['question_id']}",
                "selection_reason": f"probe_multi_rel:{r['type']}:seed{SEED}",
            }
        )
        gt[r["question_id"]] = str(r["answer"])
        selection.append(
            {
                "question_id": r["question_id"],
                "image_id": str(r["imageId"]),
                "type": r["type"],
                "pool_size": r["pool"],
                "types": r["types"],
                "question": r["question"],
                "answer": r["answer"],
                "fullAnswer": r.get("fullAnswer"),
                "semanticStr": r["semanticStr"],
                "semantic": r["semantic"],
                "object_refs": object_refs(r),
                "n_relate": n_relate(r),
            }
        )
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for path in (DATA_ROOT / "questions_dev.jsonl", DATA_ROOT / "gt_dev.json"):
        if path.exists():
            raise SystemExit(f"{path} exists: selection is frozen")
    (DATA_ROOT / "questions_dev.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in manifest)
    )
    (DATA_ROOT / "gt_dev.json").write_text(json.dumps(gt, indent=1))
    (OUT_ROOT / "selection.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in selection)
    )
    print("selected", len(manifest), "images", len(used))


if __name__ == "__main__":
    main()
