"""Compute scene graph metrics per image per level."""

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import load_config, resolve
from milestone1.metrics import (
    mean_confidence,
    node_precision_recall_f1,
    sg_score,
    triplet_precision_recall_f1,
)


def main() -> None:
    cfg = load_config()
    node_w = cfg["metrics"]["sg_score"]["node_weight"]
    rel_w = cfg["metrics"]["sg_score"]["relation_weight"]
    num_levels = cfg["resolution_levels"]["num_levels"]

    selected_path = resolve(cfg["paths"]["selected_images"])
    with open(selected_path) as f:
        image_ids = json.load(f)

    pred_root = resolve(cfg["paths"]["egtr_predictions"])
    gt_root = resolve(cfg["paths"]["gt_scene_graphs"])
    metrics_root = resolve(cfg["paths"]["metrics"])
    metrics_root.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id", "level",
        "node_precision", "node_recall", "node_f1",
        "triplet_precision", "triplet_recall", "triplet_f1",
        "sg_score", "mean_object_confidence", "mean_relation_confidence",
    ]

    all_rows = []

    for img_id in image_ids:
        # Load GT
        gt_path = gt_root / f"{img_id}.json"
        with open(gt_path) as f:
            gt = json.load(f)

        # Per-image CSV
        img_csv_path = metrics_root / f"{img_id}.csv"
        img_rows = []

        for level_idx in range(1, num_levels + 1):
            level_name = f"level_{level_idx:02d}"
            pred_path = pred_root / str(img_id) / f"{level_name}.json"

            with open(pred_path) as f:
                pred = json.load(f)

            np_, nr, nf = node_precision_recall_f1(
                pred["objects"], gt["objects"]
            )
            tp, tr, tf = triplet_precision_recall_f1(
                pred["relationships"], gt["relationships"]
            )
            sgs = sg_score(nr, tr, node_w, rel_w)
            moc = mean_confidence(pred["objects"])
            mrc = mean_confidence(pred["relationships"])

            row = {
                "image_id": img_id,
                "level": level_idx,
                "node_precision": round(np_, 4),
                "node_recall": round(nr, 4),
                "node_f1": round(nf, 4),
                "triplet_precision": round(tp, 4),
                "triplet_recall": round(tr, 4),
                "triplet_f1": round(tf, 4),
                "sg_score": round(sgs, 4),
                "mean_object_confidence": round(moc, 4),
                "mean_relation_confidence": round(mrc, 4),
            }
            img_rows.append(row)

        # Write per-image CSV
        with open(img_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(img_rows)

        all_rows.extend(img_rows)

        # Print summary for last level
        last = img_rows[-1]
        print(
            f"  {img_id}: sg_score L20={last['sg_score']:.3f} "
            f"(node_R={last['node_recall']:.3f}, "
            f"trip_R={last['triplet_recall']:.3f})"
        )

    # Write combined CSV
    combined_path = metrics_root / "all_metrics.csv"
    with open(combined_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved {len(all_rows)} metric rows to {combined_path}")


if __name__ == "__main__":
    main()
