"""Evaluate a top-K oracle that uses L20's top-N objects/triplets as reference.

Instead of using *all* L20 predictions as pseudo-GT (current RelOracle), this
script restricts the reference to the top-N objects (by confidence) and top-N
triplets (by triplet_score).  This produces a more focused oracle that targets
only the most salient semantic elements.

Outputs (all under --out-dir):
  - oracle_topk_per_image.csv
  - method_comparison_topk_c08.csv / c09.csv
  - summary.json, summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import load_config, resolve  # noqa: E402, I001


# ── Constants ──

COEF_08 = 0.8
COEF_09 = 0.9
DEFAULT_TOP_N_NODES = 10
DEFAULT_TOP_N_TRIPLETS = 5


# ── Helpers (self-contained, no heavy imports) ──


def _to_label_set(objects: list[dict]) -> set[str]:
    """Extract unique label set from objects."""
    return {obj["label"] for obj in objects}


def _to_triplet_set(
    relationships: list[dict],
) -> set[tuple[str, str, str]]:
    """Extract unique triplet set from relationships."""
    return {
        (rel["subject_label"], rel["predicate"], rel["object_label"])
        for rel in relationships
    }


def load_csv_dict(path: Path) -> list[dict]:
    """Load a CSV file as a list of dicts."""
    with open(path) as f:
        return list(csv.DictReader(f))


def save_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    """Write a list of dicts to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {path}")


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation; returns NaN for degenerate cases."""
    n = len(xs)
    if n < 3:
        return float("nan")
    ax = np.array(xs, dtype=float)
    ay = np.array(ys, dtype=float)
    if np.std(ax) == 0 or np.std(ay) == 0:
        return float("nan")
    return float(np.corrcoef(ax, ay)[0, 1])


def median_val(vals: list[float]) -> float:
    """Compute median of a list."""
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def load_egtr_prediction(pred_dir: Path, image_id: int, level: int) -> dict:
    """Load a single EGTR prediction JSON."""
    p = pred_dir / str(image_id) / f"level_{level:02d}.json"
    with open(p) as f:
        return json.load(f)


def top_n_objects(objects: list[dict], n: int) -> list[dict]:
    """Return top-N objects sorted by confidence descending."""
    sorted_objs = sorted(objects, key=lambda o: o.get("confidence", 0.0), reverse=True)
    return sorted_objs[:n]


def top_n_relationships(relationships: list[dict], n: int) -> list[dict]:
    """Return top-N relationships sorted by triplet_score descending."""
    sorted_rels = sorted(
        relationships, key=lambda r: r.get("triplet_score", 0.0), reverse=True
    )
    return sorted_rels[:n]


def compute_rel_sg_score(
    pred_objects: list[dict],
    pred_relationships: list[dict],
    ref_node_labels: set[str],
    ref_triplets: set[tuple[str, str, str]],
) -> dict[str, float]:
    """Compute RelSGScore (node_recall + triplet_recall) against reference sets."""
    pred_nodes = _to_label_set(pred_objects)
    pred_trips = _to_triplet_set(pred_relationships)

    if len(ref_node_labels) == 0:
        node_recall = float("nan")
    else:
        node_recall = len(pred_nodes & ref_node_labels) / len(ref_node_labels)

    if len(ref_triplets) == 0:
        trip_recall = float("nan")
    else:
        trip_recall = len(pred_trips & ref_triplets) / len(ref_triplets)

    # Combine: if one is NaN, use the other; if both NaN, NaN
    node_nan = isinstance(node_recall, float) and math.isnan(node_recall)
    trip_nan = isinstance(trip_recall, float) and math.isnan(trip_recall)

    if trip_nan:
        rel_sg_score = node_recall if not node_nan else float("nan")
    elif node_nan:
        rel_sg_score = trip_recall
    else:
        rel_sg_score = 0.5 * node_recall + 0.5 * trip_recall

    return {
        "node_recall": node_recall,
        "trip_recall": trip_recall,
        "rel_sg_score": rel_sg_score,
    }


def first_crossing_stop(
    scores: dict[int, float], threshold: float, max_level: int
) -> int:
    """Find the first level where score >= threshold."""
    for lvl in range(1, max_level + 1):
        val = scores.get(lvl, float("nan"))
        if not math.isnan(val) and val >= threshold:
            return lvl
    return max_level


def compute_comparison(
    methods_dict: dict[str, dict[int, int]],
    ref_map: dict[int, int],
    image_ids: list[int],
    max_level: int,
) -> list[dict]:
    """Compare each method's stops against a reference oracle."""
    ref_stops = [ref_map[i] for i in image_ids]
    rows: list[dict] = []
    for mname, stop_map in methods_dict.items():
        method_stops = [stop_map.get(i, max_level) for i in image_ids]
        errors = [m - o for m, o in zip(method_stops, ref_stops, strict=True)]
        abs_errors = [abs(e) for e in errors]
        mean_stop = float(np.mean(method_stops))
        mean_ae = float(np.mean(abs_errors))
        med_ae = median_val(abs_errors)
        early = sum(1 for e in errors if e < 0) / len(errors)
        late = sum(1 for e in errors if e > 0) / len(errors)
        corr = pearson_r(
            [float(s) for s in method_stops],
            [float(o) for o in ref_stops],
        )
        rows.append(
            {
                "method": mname,
                "mean_stop": round(mean_stop, 2),
                "mean_oracle_stop": round(float(np.mean(ref_stops)), 2),
                "mean_abs_error": round(mean_ae, 2),
                "median_abs_error": round(med_ae, 1),
                "early_stop_rate": round(early, 2),
                "late_stop_rate": round(late, 2),
                "pearson_r": round(corr, 3) if not math.isnan(corr) else "NaN",
            }
        )
    rows.sort(key=lambda r: float(r["mean_abs_error"]))
    return rows


# ── Method loading ──


def load_method_stops(
    metrics_dir: Path, image_ids: list[int], max_level: int
) -> dict[str, dict[int, int]]:
    """Load all method stop maps from existing CSVs, skipping missing ones."""
    methods: dict[str, dict[int, int]] = {}
    image_set = set(image_ids)

    # baseline
    baseline_path = metrics_dir / "baseline_stop.csv"
    if baseline_path.exists():
        rows = load_csv_dict(baseline_path)
        methods["baseline"] = {
            int(r["image_id"]): int(float(r["baseline_0.35"]))
            for r in rows
            if int(r["image_id"]) in image_set
        }
    else:
        print(f"  WARN: {baseline_path} not found, skipping baseline")

    # LLM ablation k=1
    abl_path = metrics_dir / "llm_ablation_stop.csv"
    if abl_path.exists():
        rows = load_csv_dict(abl_path)
        for slug, method_name in [
            ("qwen3.5-4b_h5_rel_on_conf_on", "llm_sgg_4b_h5_k1"),
            ("qwen3.5-9b_h5_rel_on_conf_on", "llm_sgg_9b_h5_k1"),
        ]:
            slug_rows = [
                r
                for r in rows
                if r["config_slug"] == slug and int(r["image_id"]) in image_set
            ]
            if slug_rows:
                methods[method_name] = {
                    int(r["image_id"]): int(r["stop_level"]) for r in slug_rows
                }
            else:
                print(f"  WARN: no rows for {slug} in ablation stop, skipping")
    else:
        print(f"  WARN: {abl_path} not found, skipping LLM ablation k1")

    # LLM ablation k=2
    abl_k2_path = metrics_dir / "llm_ablation_k2_stop.csv"
    if abl_k2_path.exists():
        rows = load_csv_dict(abl_k2_path)
        for slug, method_name in [
            ("qwen3.5-4b_h5_rel_on_conf_on", "llm_sgg_4b_h5_k2"),
            ("qwen3.5-9b_h5_rel_on_conf_on", "llm_sgg_9b_h5_k2"),
        ]:
            slug_rows = [
                r
                for r in rows
                if r["config_slug"] == slug and int(r["image_id"]) in image_set
            ]
            if slug_rows:
                methods[method_name] = {
                    int(r["image_id"]): int(r["stop_k2"]) for r in slug_rows
                }
            else:
                print(f"  WARN: no rows for {slug} in ablation k2, skipping")
    else:
        print(f"  WARN: {abl_k2_path} not found, skipping LLM ablation k2")

    # VLM single mode
    vlm_path = metrics_dir / "vlm_stop.csv"
    if vlm_path.exists():
        rows = load_csv_dict(vlm_path)
        vlm_configs = [
            ("qwen3.5:4b", "single", 1, "vlm_4b_single_k1"),
            ("qwen3.5:4b", "single", 2, "vlm_4b_single_k2"),
            ("qwen3.5:9b", "single", 1, "vlm_9b_single_k1"),
            ("qwen3.5:9b", "single", 2, "vlm_9b_single_k2"),
        ]
        for model, mode, k, method_name in vlm_configs:
            matching = [
                r
                for r in rows
                if r["model"] == model
                and r["mode"] == mode
                and int(r["k"]) == k
                and int(r["image_id"]) in image_set
            ]
            if matching:
                methods[method_name] = {
                    int(r["image_id"]): int(r["stop_level"]) for r in matching
                }
            else:
                print(f"  WARN: no VLM rows for {method_name}, skipping")
    else:
        print(f"  WARN: {vlm_path} not found, skipping VLM methods")

    # Hybrid
    hybrid_path = metrics_dir / "hybrid_stop.csv"
    if hybrid_path.exists():
        rows = load_csv_dict(hybrid_path)
        hybrid_variants = [
            "hybrid_4b_k1",
            "hybrid_4b_k2",
            "hybrid_9b_k1",
            "hybrid_9b_k2",
        ]
        for variant in hybrid_variants:
            matching = [
                r
                for r in rows
                if r["variant"] == variant and int(r["image_id"]) in image_set
            ]
            if matching:
                methods[variant] = {
                    int(r["image_id"]): int(r["stop_level"]) for r in matching
                }
            else:
                print(f"  WARN: no hybrid rows for {variant}, skipping")
    else:
        print(f"  WARN: {hybrid_path} not found, skipping hybrid methods")

    return methods


# ── Main ──


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate top-K oracle stop against existing method stops."
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        default="outputs/egtr_predictions",
        help="Root dir for EGTR predictions.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="outputs/metrics",
        help="Dir containing per-method stop CSVs.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/metrics/topk_oracle",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--top-n-nodes",
        type=int,
        default=DEFAULT_TOP_N_NODES,
        help="Number of top objects (by confidence) for reference.",
    )
    parser.add_argument(
        "--top-n-triplets",
        type=int,
        default=DEFAULT_TOP_N_TRIPLETS,
        help="Number of top relationships (by triplet_score) for reference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config()
    max_level: int = cfg["resolution_levels"]["num_levels"]

    pred_dir = resolve(args.pred_dir)
    metrics_dir = resolve(args.metrics_dir)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    top_n_nodes: int = args.top_n_nodes
    top_n_triplets: int = args.top_n_triplets

    selected_path = resolve(cfg["paths"]["selected_images"])
    with open(selected_path) as f:
        image_ids: list[int] = json.load(f)

    n_images = len(image_ids)
    print(
        f"TopK Oracle evaluation: {n_images} images, "
        f"top_n_nodes={top_n_nodes}, top_n_triplets={top_n_triplets}"
    )
    print(f"  pred_dir:    {pred_dir}")
    print(f"  metrics_dir: {metrics_dir}")
    print(f"  out_dir:     {out_dir}")

    # ── STEP 1: Build top-K reference sets from L20 ──
    print("\n" + "=" * 60)
    print("STEP 1: Build top-K reference sets from L20 predictions")
    print("=" * 60)

    ref_node_labels: dict[int, set[str]] = {}
    ref_triplets: dict[int, set[tuple[str, str, str]]] = {}
    ref_node_sizes: dict[int, int] = {}
    ref_trip_sizes: dict[int, int] = {}

    for img_id in image_ids:
        l20 = load_egtr_prediction(pred_dir, img_id, max_level)

        top_objs = top_n_objects(l20["objects"], top_n_nodes)
        top_rels = top_n_relationships(l20["relationships"], top_n_triplets)

        node_set = _to_label_set(top_objs)
        trip_set = _to_triplet_set(top_rels)

        ref_node_labels[img_id] = node_set
        ref_triplets[img_id] = trip_set
        ref_node_sizes[img_id] = len(node_set)
        ref_trip_sizes[img_id] = len(trip_set)

    mean_node_ref = float(np.mean(list(ref_node_sizes.values())))
    mean_trip_ref = float(np.mean(list(ref_trip_sizes.values())))
    print(
        f"  Mean ref node labels: {mean_node_ref:.1f}, "
        f"mean ref triplets: {mean_trip_ref:.1f}"
    )

    # ── STEP 2: Compute RelSGScore for each (image, level) ──
    print("\n" + "=" * 60)
    print("STEP 2: Compute RelSGScore for each (image, level)")
    print("=" * 60)

    # scores_map[img_id][level] = RelSGScore
    scores_map: dict[int, dict[int, float]] = {}

    for img_id in image_ids:
        scores_map[img_id] = {}
        for lvl in range(1, max_level + 1):
            pred = load_egtr_prediction(pred_dir, img_id, lvl)
            m = compute_rel_sg_score(
                pred["objects"],
                pred["relationships"],
                ref_node_labels[img_id],
                ref_triplets[img_id],
            )
            scores_map[img_id][lvl] = m["rel_sg_score"]

    # Verify L20 == 1.0 for all (since ref is subset of L20)
    l20_scores = [scores_map[i][max_level] for i in image_ids]
    l20_ones = sum(1 for s in l20_scores if not math.isnan(s) and abs(s - 1.0) < 1e-9)
    print(f"  L20 RelSGScore == 1.0: {l20_ones}/{n_images}")

    # ── STEP 3: Build oracle stop maps ──
    print("\n" + "=" * 60)
    print("STEP 3: Build top-K oracle stop maps (coef=0.8, 0.9)")
    print("=" * 60)

    oracle_stops_c08: dict[int, int] = {}
    oracle_stops_c09: dict[int, int] = {}

    for img_id in image_ids:
        oracle_stops_c08[img_id] = first_crossing_stop(
            scores_map[img_id], COEF_08, max_level
        )
        oracle_stops_c09[img_id] = first_crossing_stop(
            scores_map[img_id], COEF_09, max_level
        )

    sat_c08 = sum(1 for i in image_ids if oracle_stops_c08[i] < max_level)
    sat_c09 = sum(1 for i in image_ids if oracle_stops_c09[i] < max_level)
    mean_c08 = float(np.mean([oracle_stops_c08[i] for i in image_ids]))
    mean_c09 = float(np.mean([oracle_stops_c09[i] for i in image_ids]))
    med_c08 = median_val([float(oracle_stops_c08[i]) for i in image_ids])
    med_c09 = median_val([float(oracle_stops_c09[i]) for i in image_ids])

    print(
        f"  coef=0.8: saturated={sat_c08}/{n_images} ({sat_c08 / n_images:.0%}), "
        f"mean={mean_c08:.1f}, median={med_c08:.0f}"
    )
    print(
        f"  coef=0.9: saturated={sat_c09}/{n_images} ({sat_c09 / n_images:.0%}), "
        f"mean={mean_c09:.1f}, median={med_c09:.0f}"
    )

    # Stop histogram
    hist_c08: dict[int, int] = {}
    hist_c09: dict[int, int] = {}
    for i in image_ids:
        s08 = oracle_stops_c08[i]
        s09 = oracle_stops_c09[i]
        hist_c08[s08] = hist_c08.get(s08, 0) + 1
        hist_c09[s09] = hist_c09.get(s09, 0) + 1

    print("\n  Stop histogram (coef=0.8):")
    for lvl in sorted(hist_c08.keys()):
        bar = "#" * hist_c08[lvl]
        print(f"    L{lvl:02d}: {hist_c08[lvl]:3d} {bar}")

    print("\n  Stop histogram (coef=0.9):")
    for lvl in sorted(hist_c09.keys()):
        bar = "#" * hist_c09[lvl]
        print(f"    L{lvl:02d}: {hist_c09[lvl]:3d} {bar}")

    # ── STEP 4: Save per-image oracle CSV ──
    print("\n" + "=" * 60)
    print("STEP 4: Save per-image oracle CSV")
    print("=" * 60)

    per_image_rows: list[dict] = []
    for img_id in image_ids:
        row: dict = {"image_id": img_id}
        for lvl in range(1, max_level + 1):
            val = scores_map[img_id][lvl]
            row[f"rel_sgscore_L{lvl:02d}"] = (
                round(val, 4) if not math.isnan(val) else "NaN"
            )
        row["stop_c08"] = oracle_stops_c08[img_id]
        row["stop_c09"] = oracle_stops_c09[img_id]
        row["node_ref_size"] = ref_node_sizes[img_id]
        row["trip_ref_size"] = ref_trip_sizes[img_id]
        per_image_rows.append(row)

    per_image_fields = (
        ["image_id"]
        + [f"rel_sgscore_L{lvl:02d}" for lvl in range(1, max_level + 1)]
        + ["stop_c08", "stop_c09", "node_ref_size", "trip_ref_size"]
    )
    save_csv(per_image_rows, out_dir / "oracle_topk_per_image.csv", per_image_fields)

    # ── STEP 5: Load method stops ──
    print("\n" + "=" * 60)
    print("STEP 5: Load method stops from existing CSVs")
    print("=" * 60)

    methods = load_method_stops(metrics_dir, image_ids, max_level)
    print(f"  Loaded {len(methods)} methods: {list(methods.keys())}")

    # ── STEP 6: Compute comparisons ──
    print("\n" + "=" * 60)
    print("STEP 6: Method comparison vs TopK Oracle")
    print("=" * 60)

    comp_c08 = compute_comparison(methods, oracle_stops_c08, image_ids, max_level)
    comp_c09 = compute_comparison(methods, oracle_stops_c09, image_ids, max_level)

    save_csv(
        comp_c08,
        out_dir / "method_comparison_topk_c08.csv",
        list(comp_c08[0].keys()) if comp_c08 else [],
    )
    save_csv(
        comp_c09,
        out_dir / "method_comparison_topk_c09.csv",
        list(comp_c09[0].keys()) if comp_c09 else [],
    )

    # Print comparison table (coef=0.8)
    print(
        f"\n  === coef=0.8 (mean oracle stop={mean_c08:.1f}) ===\n"
        f"  {'Method':<28s} {'MeanStop':>8s} {'MAE':>6s} "
        f"{'MedAE':>6s} {'Early':>6s} {'Late':>6s} {'Corr':>7s}"
    )
    print("  " + "-" * 75)
    for r in comp_c08:
        print(
            f"  {r['method']:<28s} {r['mean_stop']:>8} {r['mean_abs_error']:>6} "
            f"{r['median_abs_error']:>6} {r['early_stop_rate']:>6} "
            f"{r['late_stop_rate']:>6} {str(r['pearson_r']):>7}"
        )

    print(
        f"\n  === coef=0.9 (mean oracle stop={mean_c09:.1f}) ===\n"
        f"  {'Method':<28s} {'MeanStop':>8s} {'MAE':>6s} "
        f"{'MedAE':>6s} {'Early':>6s} {'Late':>6s} {'Corr':>7s}"
    )
    print("  " + "-" * 75)
    for r in comp_c09:
        print(
            f"  {r['method']:<28s} {r['mean_stop']:>8} {r['mean_abs_error']:>6} "
            f"{r['median_abs_error']:>6} {r['early_stop_rate']:>6} "
            f"{r['late_stop_rate']:>6} {str(r['pearson_r']):>7}"
        )

    # ── STEP 7: Trivial baselines (always-19, always-20) ──
    print("\n" + "=" * 60)
    print("STEP 7: Trivial baselines (always-19, always-20)")
    print("=" * 60)

    trivial_methods: dict[str, dict[int, int]] = {
        "always_19": {i: 19 for i in image_ids},
        "always_20": {i: 20 for i in image_ids},
    }

    trivial_c08 = compute_comparison(
        trivial_methods, oracle_stops_c08, image_ids, max_level
    )
    trivial_c09 = compute_comparison(
        trivial_methods, oracle_stops_c09, image_ids, max_level
    )

    trivial_mae_c08: dict[str, float] = {}
    for r in trivial_c08:
        trivial_mae_c08[r["method"]] = float(r["mean_abs_error"])
        print(f"  coef=0.8 | {r['method']}: MAE={r['mean_abs_error']}")

    trivial_mae_c09: dict[str, float] = {}
    for r in trivial_c09:
        trivial_mae_c09[r["method"]] = float(r["mean_abs_error"])
        print(f"  coef=0.9 | {r['method']}: MAE={r['mean_abs_error']}")

    # ── STEP 8: Summary ──
    print("\n" + "=" * 60)
    print("STEP 8: Summary")
    print("=" * 60)

    top3_c08 = comp_c08[:3] if len(comp_c08) >= 3 else comp_c08
    top3_c09 = comp_c09[:3] if len(comp_c09) >= 3 else comp_c09

    summary_data = {
        "config": {
            "top_n_nodes": top_n_nodes,
            "top_n_triplets": top_n_triplets,
            "n_images": n_images,
            "max_level": max_level,
        },
        "reference_stats": {
            "mean_node_ref_labels": round(mean_node_ref, 1),
            "mean_trip_ref_triplets": round(mean_trip_ref, 1),
        },
        "coef_08": {
            "saturated": sat_c08,
            "saturation_rate": round(sat_c08 / n_images, 2),
            "mean_stop": round(mean_c08, 2),
            "median_stop": round(med_c08, 1),
            "histogram": {str(k): v for k, v in sorted(hist_c08.items())},
            "trivial_always_19_mae": trivial_mae_c08.get("always_19", 0.0),
            "trivial_always_20_mae": trivial_mae_c08.get("always_20", 0.0),
            "top3_methods": [
                {"method": r["method"], "mae": float(r["mean_abs_error"])}
                for r in top3_c08
            ],
        },
        "coef_09": {
            "saturated": sat_c09,
            "saturation_rate": round(sat_c09 / n_images, 2),
            "mean_stop": round(mean_c09, 2),
            "median_stop": round(med_c09, 1),
            "histogram": {str(k): v for k, v in sorted(hist_c09.items())},
            "trivial_always_19_mae": trivial_mae_c09.get("always_19", 0.0),
            "trivial_always_20_mae": trivial_mae_c09.get("always_20", 0.0),
            "top3_methods": [
                {"method": r["method"], "mae": float(r["mean_abs_error"])}
                for r in top3_c09
            ],
        },
    }

    # Save summary.json
    summary_json_path = out_dir / "summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  Saved {summary_json_path}")

    # Save summary.md
    summary_md_path = out_dir / "summary.md"
    with open(summary_md_path, "w") as f:
        f.write("# TopK Oracle Evaluation Summary\n\n")
        f.write(f"- **Top-N nodes**: {top_n_nodes} (by confidence)\n")
        f.write(f"- **Top-N triplets**: {top_n_triplets} (by triplet_score)\n")
        f.write(f"- **Images**: {n_images}\n")
        f.write(f"- **Mean ref node labels**: {mean_node_ref:.1f}\n")
        f.write(f"- **Mean ref triplets**: {mean_trip_ref:.1f}\n\n")

        f.write("## coef=0.8\n\n")
        f.write(f"- Saturated: {sat_c08}/{n_images} ({sat_c08 / n_images:.0%})\n")
        f.write(f"- Mean stop: {mean_c08:.1f}, Median: {med_c08:.0f}\n")
        f.write(f"- always-19 MAE: {trivial_mae_c08.get('always_19', 0.0)}\n")
        f.write(f"- always-20 MAE: {trivial_mae_c08.get('always_20', 0.0)}\n\n")
        f.write("### Top methods (MAE ascending)\n\n")
        f.write("| Method | MeanStop | MAE | MedAE | Early | Late | Corr |\n")
        f.write("|--------|---------|-----|-------|-------|------|------|\n")
        for r in comp_c08:
            f.write(
                f"| {r['method']} | {r['mean_stop']} | {r['mean_abs_error']} "
                f"| {r['median_abs_error']} | {r['early_stop_rate']} "
                f"| {r['late_stop_rate']} | {r['pearson_r']} |\n"
            )

        f.write("\n## coef=0.9\n\n")
        f.write(f"- Saturated: {sat_c09}/{n_images} ({sat_c09 / n_images:.0%})\n")
        f.write(f"- Mean stop: {mean_c09:.1f}, Median: {med_c09:.0f}\n")
        f.write(f"- always-19 MAE: {trivial_mae_c09.get('always_19', 0.0)}\n")
        f.write(f"- always-20 MAE: {trivial_mae_c09.get('always_20', 0.0)}\n\n")
        f.write("### Top methods (MAE ascending)\n\n")
        f.write("| Method | MeanStop | MAE | MedAE | Early | Late | Corr |\n")
        f.write("|--------|---------|-----|-------|-------|------|------|\n")
        for r in comp_c09:
            f.write(
                f"| {r['method']} | {r['mean_stop']} | {r['mean_abs_error']} "
                f"| {r['median_abs_error']} | {r['early_stop_rate']} "
                f"| {r['late_stop_rate']} | {r['pearson_r']} |\n"
            )

    print(f"  Saved {summary_md_path}")

    # Final print
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\n  Files created in {out_dir}:")
    print(f"    {out_dir / 'oracle_topk_per_image.csv'}")
    print(f"    {out_dir / 'method_comparison_topk_c08.csv'}")
    print(f"    {out_dir / 'method_comparison_topk_c09.csv'}")
    print(f"    {out_dir / 'summary.json'}")
    print(f"    {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
