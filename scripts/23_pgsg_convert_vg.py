"""Convert the local EGTR-format VG-150 into the Tang-style files PGSG's loader expects.

Outputs (in --out-dir, default data/processed/pgsg_vg/):
  VG-SGG-test.h5            Tang-style h5 (split, img_to_*, labels, boxes_1024, ...)
  VG-SGG-val.h5             symlink to VG-SGG-test.h5 (the val builder entry must exist)
  image_data.json           108,073 rows, row-aligned with the h5 per-image arrays
  categories_dict.json      copy of PGSG's VG category dict (obj: 150, rel: 50)
  zeroshot_triplet.pytorch  [M, 3] 0-based (sub_cls, obj_cls, pred) test triplets unseen in train
  selected_test_ids.json    the target test image ids, in evaluation order

PGSG loader quirks handled here (Pix2Grp_CVPR2024/lavis/datasets/datasets/vg_rel_detection.py):
  * the split is inferred from substrings of the h5 path -> out-dir must not contain
    'train'/'val'/'test'/'zs_pred' (checked below);
  * every non-val split drops its first 5,000 split-matching rows -> we prepend 5,000
    "sacrificial" split-2 rows that share a single dummy box/relation and are never opened;
  * load_image_filenames asserts exactly 108,073 rows -> pad with split=-1 dummy rows;
  * val takes the first 5,000 split-0 rows -> the target images are duplicated as split 0
    so the (unused) val dataset is tiny and non-empty.

Usage:
  uv run --project envs/pgsg python scripts/23_pgsg_convert_vg.py --num-test-images 20
  uv run --project envs/pgsg python scripts/23_pgsg_convert_vg.py --num-test-images 0   # all 26,446
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
VG_ROOT = Path("/home/kanaoka/usr/dataset/vg")
PGSG_CATEGORIES = (
    REPO_ROOT
    / "Pix2Grp_CVPR2024/all_categories_dict/vg/vg_motif_anno/categories_dict.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / "data/processed/pgsg_vg"

TOTAL_ROWS = 108_073  # asserted by PGSG's load_image_filenames
NUM_DROPPED_ROWS = 5_000  # PGSG drops the first 5,000 rows of every non-val split
BOX_SCALE = 1024
MIN_BOX_SIDE = 1e-3
NUM_ATTRIBUTES = 10
SPLIT_TRAIN = 0
SPLIT_TEST = 2
SPLIT_UNUSED = -1
DUMMY_IMAGE_ID_BASE = 10**9  # ids that never exist on disk (rows are never opened)
CORRUPTED_IDS = {1592, 1722, 4616, 4617}  # skipped by the loader -> must not appear
FORBIDDEN_PATH_TOKENS = ("train", "val", "test", "zs_pred")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--num-test-images",
        type=int,
        default=20,
        help="first N test.json images to evaluate; 0 = all test images",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--vg-root", type=Path, default=VG_ROOT)
    return parser.parse_args()


def load_split(vg_root: Path, split: str) -> tuple[list[dict], dict[int, list[dict]]]:
    """Return (images, anns_by_image sorted by ann id) for a COCO-style split json."""
    coco = json.loads((vg_root / f"{split}.json").read_text())
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    for anns in anns_by_image.values():
        anns.sort(key=lambda a: a["id"])
    return coco["images"], anns_by_image


def coco_box_to_cxcywh_1024(bbox: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = bbox
    scale = BOX_SCALE / max(width, height)
    return [
        max((x + w / 2) * scale, 0.0),
        max((y + h / 2) * scale, 0.0),
        max(w * scale, MIN_BOX_SIDE),
        max(h * scale, MIN_BOX_SIDE),
    ]


class H5Builder:
    """Accumulates Tang-style per-image rows and global box/relation arrays."""

    def __init__(self) -> None:
        self.split: list[int] = []
        self.first_box: list[int] = []
        self.last_box: list[int] = []
        self.first_rel: list[int] = []
        self.last_rel: list[int] = []
        self.labels: list[int] = []
        self.boxes: list[list[float]] = []
        self.relationships: list[list[int]] = []
        self.predicates: list[int] = []
        self.image_rows: list[dict] = []

    def add_image(
        self,
        image: dict,
        anns: list[dict],
        rels: list[list[int]],
        split_flag: int,
    ) -> None:
        box_offset = len(self.boxes)
        rel_offset = len(self.relationships)
        for ann in anns:
            self.boxes.append(
                coco_box_to_cxcywh_1024(ann["bbox"], image["width"], image["height"])
            )
            self.labels.append(ann["category_id"])  # 1-based, loader subtracts 1
        for sub_idx, obj_idx, pred in rels:
            self.relationships.append([box_offset + sub_idx, box_offset + obj_idx])
            self.predicates.append(pred)  # 1-based, loader subtracts 1
        self._add_row(
            image["id"],
            image["width"],
            image["height"],
            split_flag,
            (box_offset, len(self.boxes) - 1) if anns else (-1, -1),
            (rel_offset, len(self.relationships) - 1) if rels else (-1, -1),
        )

    def add_sacrificial_rows(self, count: int) -> None:
        """Split-2 rows that the loader drops before ever reading them."""
        box_idx = len(self.boxes)
        rel_idx = len(self.relationships)
        self.boxes.append([1.0, 1.0, 1.0, 1.0])
        self.labels.append(1)
        self.relationships.append([box_idx, box_idx])
        self.predicates.append(1)
        for k in range(count):
            self._add_row(
                DUMMY_IMAGE_ID_BASE + k,
                1,
                1,
                SPLIT_TEST,
                (box_idx, box_idx),
                (rel_idx, rel_idx),
            )

    def pad_rows(self, total_rows: int) -> None:
        start = len(self.split)
        for k in range(total_rows - start):
            self._add_row(
                DUMMY_IMAGE_ID_BASE + NUM_DROPPED_ROWS + k,
                1,
                1,
                SPLIT_UNUSED,
                (-1, -1),
                (-1, -1),
            )

    def _add_row(
        self,
        image_id: int,
        width: int,
        height: int,
        split_flag: int,
        box_range: tuple[int, int],
        rel_range: tuple[int, int],
    ) -> None:
        self.split.append(split_flag)
        self.first_box.append(box_range[0])
        self.last_box.append(box_range[1])
        self.first_rel.append(rel_range[0])
        self.last_rel.append(rel_range[1])
        self.image_rows.append({"image_id": image_id, "width": width, "height": height})

    def write(self, h5_path: Path) -> None:
        int_dtype = np.int32
        with h5py.File(h5_path, "w") as f:
            f["split"] = np.asarray(self.split, dtype=int_dtype)
            f["img_to_first_box"] = np.asarray(self.first_box, dtype=int_dtype)
            f["img_to_last_box"] = np.asarray(self.last_box, dtype=int_dtype)
            f["img_to_first_rel"] = np.asarray(self.first_rel, dtype=int_dtype)
            f["img_to_last_rel"] = np.asarray(self.last_rel, dtype=int_dtype)
            f["labels"] = np.asarray(self.labels, dtype=int_dtype).reshape(-1, 1)
            f["attributes"] = np.zeros(
                (len(self.labels), NUM_ATTRIBUTES), dtype=int_dtype
            )
            f[f"boxes_{BOX_SCALE}"] = np.asarray(self.boxes, dtype=np.float32).reshape(
                -1, 4
            )
            f["relationships"] = np.asarray(
                self.relationships, dtype=int_dtype
            ).reshape(-1, 2)
            f["predicates"] = np.asarray(self.predicates, dtype=int_dtype).reshape(
                -1, 1
            )


def zeroshot_triplets(
    rel: dict, test_anns: dict[int, list[dict]], train_anns: dict[int, list[dict]]
) -> torch.Tensor:
    """0-based (sub_cls, obj_cls, pred) triplets present in test but not in train."""

    def triplet_set(split: str, anns_by_image: dict[int, list[dict]]) -> set[tuple]:
        triplets = set()
        for image_id, rels in rel[split].items():
            anns = anns_by_image[int(image_id)]
            for sub_idx, obj_idx, pred in rels:
                triplets.add(
                    (
                        anns[sub_idx]["category_id"] - 1,
                        anns[obj_idx]["category_id"] - 1,
                        pred - 1,
                    )
                )
        return triplets

    unseen = sorted(triplet_set("test", test_anns) - triplet_set("train", train_anns))
    return torch.tensor(unseen, dtype=torch.long).reshape(-1, 3)


def validate(out_dir: Path, targets: list[dict], test_anns: dict, rel: dict) -> None:
    """Re-read the h5 like the PGSG loader does and round-trip a few images."""
    with h5py.File(out_dir / "VG-SGG-test.h5", "r") as f:
        split = f["split"][:]
        labels = f["labels"][:, 0]
        boxes = f[f"boxes_{BOX_SCALE}"][:]
        preds = f["predicates"][:, 0]
        rels = f["relationships"][:]
        first_box = f["img_to_first_box"][:]
        first_rel = f["img_to_first_rel"][:]
        last_rel = f["img_to_last_rel"][:]
    image_rows = json.loads((out_dir / "image_data.json").read_text())
    assert len(split) == len(image_rows) == TOTAL_ROWS
    assert labels.min() >= 1 and labels.max() <= 150
    assert preds.min() >= 1 and preds.max() <= 50
    assert np.all(boxes[:, :2] >= 0) and np.all(boxes[:, 2:] > 0)
    assert boxes.max() <= BOX_SCALE * 1.01, boxes.max()

    # Mimic the loader: split-2 rows with box+rel, drop first NUM_DROPPED_ROWS.
    mask = (split == SPLIT_TEST) & (first_box >= 0) & (first_rel >= 0)
    kept_rows = np.where(mask)[0][NUM_DROPPED_ROWS:]
    kept_ids = [image_rows[i]["image_id"] for i in kept_rows]
    assert kept_ids == [t["id"] for t in targets], "evaluated rows != target images"

    for row, image in list(zip(kept_rows, targets, strict=True))[:5]:
        anns = test_anns[image["id"]]
        offset = first_box[row]
        got = [
            [int(s - offset), int(o - offset), int(p)]
            for (s, o), p in zip(
                rels[first_rel[row] : last_rel[row] + 1],
                preds[first_rel[row] : last_rel[row] + 1],
                strict=True,
            )
        ]
        assert got == [list(r) for r in rel["test"][str(image["id"])]], image["id"]
        # Box round trip (loader: cxcywh -> xyxy, then / 1024 * max(w, h)).
        cx, cy, w, h = boxes[offset]
        scale = max(image["width"], image["height"]) / BOX_SCALE
        x1 = (cx - w / 2) * scale
        y1 = (cy - h / 2) * scale
        bx, by, _, _ = anns[0]["bbox"]
        assert abs(x1 - bx) < 1e-2 and abs(y1 - by) < 1e-2, (x1, y1, anns[0]["bbox"])
    print(f"validation OK: {len(kept_ids)} target images, triplets/boxes round-trip")


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir.resolve()
    bad = [t for t in FORBIDDEN_PATH_TOKENS if t in str(out_dir)]
    if bad:
        raise ValueError(f"out-dir must not contain {bad}: PGSG infers split from path")
    out_dir.mkdir(parents=True, exist_ok=True)

    rel = json.loads((args.vg_root / "rel.json").read_text())
    test_images, test_anns = load_split(args.vg_root, "test")
    _, train_anns = load_split(args.vg_root, "train")

    test_images = [img for img in test_images if rel["test"].get(str(img["id"]))]
    assert not CORRUPTED_IDS & {img["id"] for img in test_images}
    num = args.num_test_images or len(test_images)
    targets = test_images[:num]

    builder = H5Builder()
    builder.add_sacrificial_rows(NUM_DROPPED_ROWS)
    for image in targets:
        builder.add_image(
            image, test_anns[image["id"]], rel["test"][str(image["id"])], SPLIT_TEST
        )
    for image in targets:  # tiny, unused val split (first split-0 rows)
        builder.add_image(
            image, test_anns[image["id"]], rel["test"][str(image["id"])], SPLIT_TRAIN
        )
    if len(builder.split) > TOTAL_ROWS:
        raise ValueError(f"too many rows: {len(builder.split)} > {TOTAL_ROWS}")
    builder.pad_rows(TOTAL_ROWS)

    builder.write(out_dir / "VG-SGG-test.h5")
    val_link = out_dir / "VG-SGG-val.h5"
    val_link.unlink(missing_ok=True)
    val_link.symlink_to("VG-SGG-test.h5")
    (out_dir / "image_data.json").write_text(json.dumps(builder.image_rows))
    shutil.copyfile(PGSG_CATEGORIES, out_dir / "categories_dict.json")
    zs = zeroshot_triplets(rel, test_anns, train_anns)
    torch.save(zs, out_dir / "zeroshot_triplet.pytorch")
    (out_dir / "selected_test_ids.json").write_text(
        json.dumps([t["id"] for t in targets])
    )
    print(
        f"wrote {out_dir}: {len(targets)} target images, {len(builder.boxes)} boxes, "
        f"{len(builder.relationships)} rels, {len(zs)} zero-shot triplets"
    )
    validate(out_dir, targets, test_anns, rel)


if __name__ == "__main__":
    main()
