"""Evaluate SGDet R@20 per level and compute oracle stopping points.

Standard SGDet evaluation with graph-constraint (top-20 predictions per image).
Match requires class-id equality for subject, predicate, object AND IoU >= 0.5
for both subject and object bounding boxes.

Oracle stop: first level where R@20 >= coef * R@20(level=20).
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IOU_THRESHOLD: float = 0.5
TOP_K: int = 20
NUM_LEVELS: int = 20
ORACLE_COEFS: tuple[float, ...] = (0.8, 0.9)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def box_iou(a: list[float], b: list[float]) -> float:
    """Compute IoU between two axis-aligned boxes [x1, y1, x2, y2]."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def triplet_matches(pred: dict, gt: dict, iou_thresh: float = IOU_THRESHOLD) -> bool:
    """Check if a predicted triplet matches a GT triplet."""
    if pred["subject_class_id"] != gt["subject_class_id"]:
        return False
    if pred["object_class_id"] != gt["object_class_id"]:
        return False
    if pred["predicate_id"] != gt["predicate_id"]:
        return False
    if box_iou(pred["subject_bbox"], gt["subject_bbox"]) < iou_thresh:
        return False
    if box_iou(pred["object_bbox"], gt["object_bbox"]) < iou_thresh:
        return False
    return True


def recall_at_k(
    predictions: list[dict],
    gt_triplets: list[dict],
    top_k: int = TOP_K,
) -> float:
    """Compute R@K for a single image-level pair.

    Returns the fraction of GT triplets matched by the top-K predictions.
    Predictions must already be sorted by triplet_score descending.
    """
    if not gt_triplets:
        return float("nan")

    top_preds = predictions[:top_k]
    matched = 0
    for gt in gt_triplets:
        for pred in top_preds:
            if triplet_matches(pred, gt):
                matched += 1
                break
    return matched / len(gt_triplets)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_gt(gt_path: Path) -> list[dict]:
    """Load GT relationships for one image."""
    with open(gt_path) as f:
        data = json.load(f)
    return data.get("relationships", [])


def load_predictions(pred_path: Path) -> list[dict]:
    """Load predictions for one image-level, sorted by triplet_score desc."""
    with open(pred_path) as f:
        data = json.load(f)
    rels = data.get("relationships", [])
    rels.sort(key=lambda r: r.get("triplet_score", 0.0), reverse=True)
    return rels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="SGDet R@20 oracle evaluation")
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "runs" / "nearest" / "egtr_predictions",
        help="Directory with per-image prediction subdirs",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "runs" / "nearest" / "sgdet",
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Load config for GT path
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from milestone1.config import load_config, resolve

    cfg = load_config()
    gt_dir = resolve(cfg["paths"]["gt_scene_graphs"])
    selected_path = resolve(cfg["paths"]["selected_images"])

    with open(selected_path) as f:
        image_ids: list[int] = json.load(f)

    pred_dir: Path = args.pred_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-image evaluation
    level_cols = [f"r20_L{lv:02d}" for lv in range(1, NUM_LEVELS + 1)]
    coef_cols = [f"stop_c{int(c * 100):02d}" for c in ORACLE_COEFS]
    fieldnames = (
        ["image_id", "n_gt_triplets"]
        + level_cols
        + ["ref_r20_L20", "degenerate"]
        + coef_cols
    )

    rows: list[dict] = []
    excluded_count = 0

    # Accumulators for corpus-level micro-average
    corpus_matched_per_level: list[int] = [0] * NUM_LEVELS
    corpus_gt_total = 0

    for img_id in image_ids:
        gt_path = gt_dir / f"{img_id}.json"
        gt_rels = load_gt(gt_path)
        n_gt = len(gt_rels)

        if n_gt == 0:
            excluded_count += 1
            continue

        row: dict[str, object] = {
            "image_id": img_id,
            "n_gt_triplets": n_gt,
        }

        corpus_gt_total += n_gt

        r20_values: list[float] = []
        for lv in range(1, NUM_LEVELS + 1):
            lv_path = pred_dir / str(img_id) / f"level_{lv:02d}.json"
            preds = load_predictions(lv_path)
            r20 = recall_at_k(preds, gt_rels)
            r20_values.append(r20)
            row[f"r20_L{lv:02d}"] = round(r20, 6)

            # Count matched GT triplets for corpus micro-average
            top_preds = preds[:TOP_K]
            matched = 0
            for gt in gt_rels:
                for pred in top_preds:
                    if triplet_matches(pred, gt):
                        matched += 1
                        break
            corpus_matched_per_level[lv - 1] += matched

        ref = r20_values[-1]  # R@20 at level 20
        row["ref_r20_L20"] = round(ref, 6)

        is_degenerate = ref == 0.0
        row["degenerate"] = 1 if is_degenerate else 0

        for coef in ORACLE_COEFS:
            col = f"stop_c{int(coef * 100):02d}"
            if is_degenerate:
                row[col] = ""
            else:
                threshold = coef * ref
                stop_level = NUM_LEVELS  # default: last level
                for lv_idx, r20_val in enumerate(r20_values):
                    if r20_val >= threshold:
                        stop_level = lv_idx + 1
                        break
                row[col] = stop_level

        rows.append(row)

    # Save per_image.csv
    csv_path = out_dir / "per_image.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-image results to {csv_path}")

    # ---------------------------------------------------------------------------
    # Aggregate summary
    # ---------------------------------------------------------------------------
    total_images = len(image_ids)
    evaluated_images = len(rows)  # n_gt > 0
    valid_rows = [r for r in rows if r["degenerate"] == 0]
    degenerate_rows = [r for r in rows if r["degenerate"] == 1]
    n_valid = len(valid_rows)
    n_degenerate = len(degenerate_rows)

    # Corpus-level R@20 (micro-average) per level
    corpus_r20_per_level: dict[str, float] = {}
    for lv in range(NUM_LEVELS):
        key = f"L{lv + 1:02d}"
        corpus_r20_per_level[key] = (
            round(corpus_matched_per_level[lv] / corpus_gt_total, 6)
            if corpus_gt_total > 0
            else 0.0
        )

    # Oracle stop distributions (valid images only)
    oracle_stats: dict[str, dict] = {}
    for coef in ORACLE_COEFS:
        col = f"stop_c{int(coef * 100):02d}"
        stops = [int(r[col]) for r in valid_rows if r[col] != ""]
        if stops:
            hist: dict[int, int] = {}
            for s in stops:
                hist[s] = hist.get(s, 0) + 1
            sorted_hist = dict(sorted(hist.items()))
            mean_val = round(statistics.mean(stops), 2)
            median_val = round(statistics.median(stops), 2)
        else:
            sorted_hist = {}
            mean_val = float("nan")
            median_val = float("nan")

        # Sanity-check baselines: always-L19, always-L20
        mae_19 = (
            round(statistics.mean(abs(s - 19) for s in stops), 2)
            if stops
            else float("nan")
        )
        mae_20 = (
            round(statistics.mean(abs(s - 20) for s in stops), 2)
            if stops
            else float("nan")
        )

        oracle_stats[col] = {
            "coef": coef,
            "n_valid": len(stops),
            "mean": mean_val,
            "median": median_val,
            "histogram": sorted_hist,
            "baseline_always_19_mae": mae_19,
            "baseline_always_20_mae": mae_20,
        }

    # GT triplet distribution
    gt_counts = [r["n_gt_triplets"] for r in rows]
    gt_dist = {
        "min": min(gt_counts) if gt_counts else 0,
        "median": round(statistics.median(gt_counts), 1) if gt_counts else 0,
        "max": max(gt_counts) if gt_counts else 0,
        "count_lt5": sum(1 for c in gt_counts if c < 5),
    }

    summary = {
        "image_counts": {
            "total": total_images,
            "excluded_no_gt": excluded_count,
            "evaluated": evaluated_images,
            "valid_ref_gt0": n_valid,
            "degenerate_ref_eq0": n_degenerate,
        },
        "corpus_r20_per_level": corpus_r20_per_level,
        "oracle_stop": oracle_stats,
        "gt_triplets_distribution": gt_dist,
    }

    # Save summary.json
    json_path = out_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved summary JSON to {json_path}")

    # Save summary.md
    md_lines: list[str] = []
    md_lines.append("# SGDet R@20 Oracle Evaluation Summary\n")

    md_lines.append("## Image Counts\n")
    ic = summary["image_counts"]
    md_lines.append(f"- Total selected: {ic['total']}")
    md_lines.append(f"- Excluded (0 GT triplets): {ic['excluded_no_gt']}")
    md_lines.append(f"- Evaluated: {ic['evaluated']}")
    md_lines.append(f"- Valid (ref > 0): {ic['valid_ref_gt0']}")
    md_lines.append(f"- Degenerate (ref == 0): {ic['degenerate_ref_eq0']}\n")

    md_lines.append("## Corpus-level R@20 (micro-average)\n")
    md_lines.append("| Level | R@20 |")
    md_lines.append("|-------|------|")
    for key, val in corpus_r20_per_level.items():
        md_lines.append(f"| {key} | {val:.4f} |")
    md_lines.append("")

    md_lines.append("## Oracle Stop Distributions (valid images only)\n")
    for _col, stats in oracle_stats.items():
        coef_label = stats["coef"]
        md_lines.append(f"### Coefficient {coef_label}\n")
        md_lines.append(f"- N valid: {stats['n_valid']}")
        md_lines.append(f"- Mean stop level: {stats['mean']}")
        md_lines.append(f"- Median stop level: {stats['median']}")
        md_lines.append(f"- Baseline always-L19 MAE: {stats['baseline_always_19_mae']}")
        md_lines.append(f"- Baseline always-L20 MAE: {stats['baseline_always_20_mae']}")
        md_lines.append("")
        md_lines.append("| Level | Count |")
        md_lines.append("|-------|-------|")
        for level, count in stats["histogram"].items():
            md_lines.append(f"| {level} | {count} |")
        md_lines.append("")

    md_lines.append("## GT Triplets Distribution\n")
    md_lines.append(f"- Min: {gt_dist['min']}")
    md_lines.append(f"- Median: {gt_dist['median']}")
    md_lines.append(f"- Max: {gt_dist['max']}")
    md_lines.append(f"- Count with < 5 triplets: {gt_dist['count_lt5']}")
    md_lines.append("")

    md_path = out_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved summary markdown to {md_path}")


if __name__ == "__main__":
    main()
