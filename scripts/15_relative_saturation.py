"""Relative saturation analysis (Plan B) for Milestone-1 SGG-saturation PoC.

Reference = each image's own Level-20 EGTR output (pseudo-GT), NOT the
dataset's true GT.  This removes the EGTR-ceiling confound (absolute SGScore
only reaches ~0.25-0.30 vs true GT) and matches the robot stopping problem.

Steps:
  1. Build pseudo-GT per image from level_20.json
  2. Compute relative metrics per (image, level 1..20)
  3. Compute RelOracleStop (threshold + gain-based)
  4. Re-evaluate all stop methods vs RelOracle(0.8)
  5. Compare RelOracle vs old true-GT Oracle
  6. Generate figures

Run:
  cd /home/kanaoka/usr/github/milestone_1 && \
  uv run --project egtr python scripts/15_relative_saturation.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import load_config, resolve  # noqa: E402, I001
from milestone1.metrics import _to_label_set, _to_triplet_set  # noqa: E402


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def load_csv_dict(path: Path) -> list[dict]:
    """Load CSV as list of dicts."""
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
    """Pearson correlation; returns NaN if degenerate."""
    n = len(xs)
    if n < 3:
        return float("nan")
    ax = np.array(xs, dtype=float)
    ay = np.array(ys, dtype=float)
    if np.std(ax) == 0 or np.std(ay) == 0:
        return float("nan")
    return float(np.corrcoef(ax, ay)[0, 1])


def median(vals: list[float]) -> float:
    """Simple median."""
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def load_egtr_prediction(pred_dir: Path, image_id: int, level: int) -> dict:
    """Load EGTR prediction JSON for (image_id, level)."""
    p = pred_dir / str(image_id) / f"level_{level:02d}.json"
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Relative metric computation
# ---------------------------------------------------------------------------


def compute_relative_metrics(
    pred_objects: list[dict],
    pred_relationships: list[dict],
    ref_objects: list[dict],
    ref_relationships: list[dict],
) -> dict[str, float]:
    """Compute relative metrics for a prediction vs pseudo-GT reference.

    Returns dict with: RelNodeRecall, RelTripletRecall, RelSGScore,
    RelNodeF1, RelTripletF1, RelSGScore_f1.
    """
    pred_nodes = _to_label_set(pred_objects)
    ref_nodes = _to_label_set(ref_objects)

    pred_trips = _to_triplet_set(pred_relationships)
    ref_trips = _to_triplet_set(ref_relationships)

    # Node recall (relative)
    if len(ref_nodes) == 0:
        node_recall = float("nan")
        node_prec = float("nan")
        node_f1 = float("nan")
    else:
        tp_nodes = len(pred_nodes & ref_nodes)
        node_recall = tp_nodes / len(ref_nodes)
        node_prec = tp_nodes / len(pred_nodes) if len(pred_nodes) > 0 else 0.0
        node_f1 = (
            2 * node_prec * node_recall / (node_prec + node_recall)
            if (node_prec + node_recall) > 0
            else 0.0
        )

    # Triplet recall (relative)
    if len(ref_trips) == 0:
        trip_recall = float("nan")
        trip_prec = float("nan")
        trip_f1 = float("nan")
    else:
        tp_trips = len(pred_trips & ref_trips)
        trip_recall = tp_trips / len(ref_trips)
        trip_prec = tp_trips / len(pred_trips) if len(pred_trips) > 0 else 0.0
        trip_f1 = (
            2 * trip_prec * trip_recall / (trip_prec + trip_recall)
            if (trip_prec + trip_recall) > 0
            else 0.0
        )

    # RelSGScore (recall-based): 0.5 * node_recall + 0.5 * trip_recall
    # If triplet is NaN (no ref triplets), use node_recall only
    if math.isnan(trip_recall):
        rel_sg_score = node_recall if not math.isnan(node_recall) else float("nan")
    elif math.isnan(node_recall):
        rel_sg_score = trip_recall
    else:
        rel_sg_score = 0.5 * node_recall + 0.5 * trip_recall

    # RelSGScore_f1 (F1-based)
    if math.isnan(trip_f1):
        rel_sg_score_f1 = node_f1 if not math.isnan(node_f1) else float("nan")
    elif math.isnan(node_f1):
        rel_sg_score_f1 = trip_f1
    else:
        rel_sg_score_f1 = 0.5 * node_f1 + 0.5 * trip_f1

    return {
        "RelNodeRecall": round(node_recall, 6)
        if not math.isnan(node_recall)
        else "NaN",
        "RelTripletRecall": round(trip_recall, 6)
        if not math.isnan(trip_recall)
        else "NaN",
        "RelSGScore": round(rel_sg_score, 6) if not math.isnan(rel_sg_score) else "NaN",
        "RelNodeF1": round(node_f1, 6) if not math.isnan(node_f1) else "NaN",
        "RelTripletF1": round(trip_f1, 6) if not math.isnan(trip_f1) else "NaN",
        "RelSGScore_f1": round(rel_sg_score_f1, 6)
        if not math.isnan(rel_sg_score_f1)
        else "NaN",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = load_config()
    max_level = cfg["resolution_levels"]["num_levels"]  # 20
    gain_epsilon = cfg["metrics"]["gain_epsilon"]  # 0.02
    gain_k = cfg["metrics"]["gain_k"]  # 3
    main_baseline_t = cfg["baseline"]["mean_object_conf_threshold"]  # 0.4

    # For relative oracle, use threshold=0.8 as primary (stated in task)
    rel_oracle_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    main_rel_threshold = 0.8

    metrics_root = resolve(cfg["paths"]["metrics"])
    figures_root = resolve(cfg["paths"]["figures"])
    pred_root = resolve(cfg["paths"]["egtr_predictions"])
    figures_root.mkdir(parents=True, exist_ok=True)

    selected_path = resolve(cfg["paths"]["selected_images"])
    with open(selected_path) as f:
        image_ids: list[int] = json.load(f)

    # ===================================================================
    # STEP 1: Build pseudo-GT from level_20 predictions
    # ===================================================================
    print("=" * 60)
    print("STEP 1: Build pseudo-GT from level_20 predictions")
    print("=" * 60)

    pseudo_gt: dict[int, dict] = {}
    for img_id in image_ids:
        data = load_egtr_prediction(pred_root, img_id, max_level)
        pseudo_gt[img_id] = data
        node_set = _to_label_set(data["objects"])
        trip_set = _to_triplet_set(data["relationships"])
        print(
            f"  {img_id}: {len(node_set)} unique nodes, {len(trip_set)} unique triplets"
        )

    # ===================================================================
    # STEP 2: Compute relative metrics per (image, level)
    # ===================================================================
    print()
    print("=" * 60)
    print("STEP 2: Compute relative metrics per (image, level 1..20)")
    print("=" * 60)

    rel_metrics_rows: list[dict] = []
    # Also store for easy lookup: {(img_id, level): RelSGScore}
    rel_sg_map: dict[tuple[int, int], float] = {}

    for img_id in image_ids:
        ref = pseudo_gt[img_id]
        for lvl in range(1, max_level + 1):
            pred = load_egtr_prediction(pred_root, img_id, lvl)
            m = compute_relative_metrics(
                pred["objects"],
                pred["relationships"],
                ref["objects"],
                ref["relationships"],
            )
            row = {"image_id": img_id, "level": lvl, **m}
            rel_metrics_rows.append(row)

            # Parse RelSGScore for internal use
            sg_val = m["RelSGScore"]
            rel_sg_map[(img_id, lvl)] = (
                float(sg_val) if sg_val != "NaN" else float("nan")
            )

        # Print level-20 sanity check
        sg20 = rel_sg_map[(img_id, max_level)]
        print(f"  {img_id}: RelSGScore(20)={sg20:.4f} (should be 1.0)")

    save_csv(
        rel_metrics_rows,
        metrics_root / "relative_metrics.csv",
        [
            "image_id",
            "level",
            "RelNodeRecall",
            "RelTripletRecall",
            "RelSGScore",
            "RelNodeF1",
            "RelTripletF1",
            "RelSGScore_f1",
        ],
    )

    # ===================================================================
    # STEP 3: RelOracleStop (threshold-based + gain-based)
    # ===================================================================
    print()
    print("=" * 60)
    print("STEP 3: Compute RelOracleStop")
    print("=" * 60)

    rel_oracle_rows: list[dict] = []
    rel_oracle_map: dict[int, int] = {}  # main threshold -> {img_id: stop}

    for img_id in image_ids:
        row: dict[str, object] = {"image_id": img_id}

        # Threshold-based
        for t in rel_oracle_thresholds:
            stop = max_level
            for lvl in range(1, max_level + 1):
                val = rel_sg_map.get((img_id, lvl), float("nan"))
                if not math.isnan(val) and val >= t:
                    stop = lvl
                    break
            row[f"rel_oracle_{t}"] = stop

        # Main threshold stop
        rel_oracle_map[img_id] = int(row[f"rel_oracle_{main_rel_threshold}"])

        # Gain-based saturation
        sg_scores = [
            rel_sg_map.get((img_id, lvl), float("nan"))
            for lvl in range(1, max_level + 1)
        ]
        gains = [0.0]
        for i in range(1, len(sg_scores)):
            if math.isnan(sg_scores[i]) or math.isnan(sg_scores[i - 1]):
                gains.append(0.0)
            else:
                gains.append(sg_scores[i] - sg_scores[i - 1])

        gain_stop = max_level
        consecutive = 0
        for i, g in enumerate(gains):
            if abs(g) < gain_epsilon:
                consecutive += 1
                if consecutive >= gain_k:
                    gain_stop = i + 1 - (gain_k - 1)
                    break
            else:
                consecutive = 0
        row["gain_based"] = gain_stop

        rel_oracle_rows.append(row)
        print(
            f"  {img_id}: rel_oracle_{main_rel_threshold}="
            f"{row[f'rel_oracle_{main_rel_threshold}']}, "
            f"gain_based={row['gain_based']}"
        )

    fieldnames = (
        ["image_id"]
        + [f"rel_oracle_{t}" for t in rel_oracle_thresholds]
        + ["gain_based"]
    )
    save_csv(rel_oracle_rows, metrics_root / "relative_oracle_stop.csv", fieldnames)

    # Non-degeneracy check
    saturated_main = sum(
        1 for img_id in image_ids if rel_oracle_map[img_id] < max_level
    )
    print(
        f"\n  Non-degeneracy: {saturated_main}/{len(image_ids)} images saturate "
        f"at RelSGScore >= {main_rel_threshold} (vs 4/10 before with true-GT oracle_0.5)"
    )

    # ===================================================================
    # STEP 4: Re-evaluate all stop methods vs RelOracle(0.8)
    # ===================================================================
    print()
    print("=" * 60)
    print("STEP 4: Re-evaluate all methods vs RelOracle(0.8)")
    print("=" * 60)

    # Load all stop data
    baseline_rows = load_csv_dict(metrics_root / "baseline_stop.csv")
    ablation_rows = load_csv_dict(metrics_root / "llm_ablation_stop.csv")
    vlm_rows = load_csv_dict(metrics_root / "vlm_stop.csv")
    hybrid_rows_csv = load_csv_dict(metrics_root / "hybrid_stop.csv")
    abl_k2_rows = load_csv_dict(metrics_root / "llm_ablation_k2_stop.csv")

    # Build lookup maps
    baseline_map = {
        int(r["image_id"]): int(float(r[f"baseline_{main_baseline_t}"]))
        for r in baseline_rows
    }

    # Ablation map: config_slug -> {image_id: stop}
    abl_map: dict[str, dict[int, int]] = {}
    for r in ablation_rows:
        slug = r["config_slug"]
        abl_map.setdefault(slug, {})[int(r["image_id"])] = int(r["stop_level"])

    # Ablation k2 map: config_slug -> {image_id: stop_k2}
    abl_k2_map: dict[str, dict[int, int]] = {}
    for r in abl_k2_rows:
        slug = r["config_slug"]
        abl_k2_map.setdefault(slug, {})[int(r["image_id"])] = int(r["stop_k2"])

    # VLM map: (model, mode, k) -> {image_id: stop}
    vlm_map: dict[tuple[str, str, int], dict[int, int]] = {}
    for r in vlm_rows:
        key = (r["model"], r["mode"], int(r["k"]))
        vlm_map.setdefault(key, {})[int(r["image_id"])] = int(r["stop_level"])

    # Hybrid map: variant -> {image_id: stop}
    hybrid_map: dict[str, dict[int, int]] = {}
    for r in hybrid_rows_csv:
        hybrid_map.setdefault(r["variant"], {})[int(r["image_id"])] = int(
            r["stop_level"]
        )

    # Assemble all methods
    base_4b_slug = "qwen3.5-4b_h5_rel_on_conf_on"
    base_9b_slug = "qwen3.5-9b_h5_rel_on_conf_on"

    methods: dict[str, dict[int, int]] = {}
    methods["baseline"] = baseline_map

    # LLM-SGG from ablation: 4b and 9b base, k=1 and k=2
    methods["llm_sgg_4b_h5_k1"] = abl_map[base_4b_slug]
    methods["llm_sgg_4b_h5_k2"] = abl_k2_map[base_4b_slug]
    methods["llm_sgg_9b_h5_k1"] = abl_map[base_9b_slug]
    methods["llm_sgg_9b_h5_k2"] = abl_k2_map[base_9b_slug]

    # Ablation variants (9b, k=1)
    abl_variant_slugs = {
        "abl_9b_h1": "qwen3.5-9b_h1_rel_on_conf_on",
        "abl_9b_h3": "qwen3.5-9b_h3_rel_on_conf_on",
        "abl_9b_rel_off": "qwen3.5-9b_h5_rel_off_conf_on",
        "abl_9b_conf_off": "qwen3.5-9b_h5_rel_on_conf_off",
    }
    for mname, slug in abl_variant_slugs.items():
        methods[mname] = abl_map[slug]

    # VLM methods
    vlm_method_names = {
        ("qwen3.5:4b", "single", 1): "vlm_4b_single_k1",
        ("qwen3.5:4b", "single", 2): "vlm_4b_single_k2",
        ("qwen3.5:9b", "single", 1): "vlm_9b_single_k1",
        ("qwen3.5:9b", "single", 2): "vlm_9b_single_k2",
        ("qwen3.5:4b", "sequence", 1): "vlm_4b_sequence_k1",
        ("qwen3.5:4b", "sequence", 2): "vlm_4b_sequence_k2",
        ("qwen3.5:9b", "sequence", 1): "vlm_9b_sequence_k1",
        ("qwen3.5:9b", "sequence", 2): "vlm_9b_sequence_k2",
    }
    for vlm_key, mname in vlm_method_names.items():
        if vlm_key in vlm_map:
            methods[mname] = vlm_map[vlm_key]

    # Hybrid methods
    for v, stops in hybrid_map.items():
        methods[v] = stops

    # Compute comparison stats vs RelOracle(0.8)
    oracle_stops = [rel_oracle_map[i] for i in image_ids]
    comparison_rows: list[dict] = []

    for mname, stop_map in methods.items():
        method_stops = [stop_map.get(i, max_level) for i in image_ids]
        errors = [m - o for m, o in zip(method_stops, oracle_stops, strict=True)]
        abs_errors = [abs(e) for e in errors]

        mean_stop = float(np.mean(method_stops))
        mean_ae = float(np.mean(abs_errors))
        median_ae = median(abs_errors)
        early = sum(1 for e in errors if e < 0) / len(errors)
        late = sum(1 for e in errors if e > 0) / len(errors)
        corr = pearson_r(
            [float(s) for s in method_stops],
            [float(o) for o in oracle_stops],
        )

        comparison_rows.append(
            {
                "method": mname,
                "mean_stop": round(mean_stop, 2),
                "mean_rel_oracle_stop": round(float(np.mean(oracle_stops)), 2),
                "mean_abs_error": round(mean_ae, 2),
                "median_abs_error": round(median_ae, 1),
                "early_stop_rate": round(early, 2),
                "late_stop_rate": round(late, 2),
                "pearson_r": round(corr, 3) if not math.isnan(corr) else "NaN",
            }
        )

    # Sort by MAE ascending
    comparison_rows.sort(key=lambda r: float(r["mean_abs_error"]))

    save_csv(
        comparison_rows,
        metrics_root / "method_comparison_relative.csv",
        list(comparison_rows[0].keys()),
    )

    print(
        f"\n  {'Method':<28s} {'MeanStop':>8s} {'OracleM':>7s} {'MAE':>6s} "
        f"{'MedAE':>6s} {'Early':>6s} {'Late':>6s} {'Corr':>7s}"
    )
    print("  " + "-" * 80)
    for r in comparison_rows:
        print(
            f"  {r['method']:<28s} {r['mean_stop']:>8} {r['mean_rel_oracle_stop']:>7} "
            f"{r['mean_abs_error']:>6} {r['median_abs_error']:>6} "
            f"{r['early_stop_rate']:>6} {r['late_stop_rate']:>6} "
            f"{str(r['pearson_r']):>7}"
        )

    # ===================================================================
    # STEP 5: Compare RelOracle vs old true-GT OracleStop
    # ===================================================================
    print()
    print("=" * 60)
    print("STEP 5: Compare RelOracle(0.8) vs old GT Oracle(0.5)")
    print("=" * 60)

    old_oracle_rows = load_csv_dict(metrics_root / "oracle_stop.csv")
    old_oracle_map = {
        int(r["image_id"]): int(float(r["oracle_0.5"])) for r in old_oracle_rows
    }

    print(
        f"\n  {'image_id':<12s} {'GT_Oracle_0.5':>14s} {'RelOracle_0.8':>14s} "
        f"{'Diff(Rel-GT)':>13s}"
    )
    print("  " + "-" * 55)
    diffs = []
    for img_id in image_ids:
        gt_stop = old_oracle_map[img_id]
        rel_stop = rel_oracle_map[img_id]
        diff = rel_stop - gt_stop
        diffs.append(diff)
        print(f"  {img_id:<12d} {gt_stop:>14d} {rel_stop:>14d} {diff:>+13d}")

    gt_saturated = sum(1 for i in image_ids if old_oracle_map[i] < max_level)
    rel_saturated = sum(1 for i in image_ids if rel_oracle_map[i] < max_level)
    gt_mean = float(np.mean([old_oracle_map[i] for i in image_ids]))
    rel_mean = float(np.mean([rel_oracle_map[i] for i in image_ids]))

    n_images = len(image_ids)
    print(
        f"\n  GT Oracle(0.5):  mean={gt_mean:.1f}, saturated={gt_saturated}/{n_images}"
    )
    print(
        f"  RelOracle(0.8):  mean={rel_mean:.1f}, saturated={rel_saturated}/{n_images}"
    )
    print(f"  Mean diff (Rel - GT): {np.mean(diffs):+.1f}")

    # ===================================================================
    # STEP 5b: Node-vs-relation saturation gap under relative reference
    # ===================================================================
    print()
    print("=" * 60)
    print("STEP 5b: Node vs Relation saturation gap (relative ref)")
    print("=" * 60)

    node_sat_threshold = 0.9
    trip_sat_threshold = 0.9

    node_first_levels = []
    trip_first_levels = []
    gaps = []

    for img_id in image_ids:
        # Find first level where RelNodeRecall >= 0.9
        node_first = max_level
        trip_first = max_level
        for lvl in range(1, max_level + 1):
            # Parse from rel_metrics_rows
            row = next(
                r
                for r in rel_metrics_rows
                if int(r["image_id"]) == img_id and int(r["level"]) == lvl
            )
            nr = row["RelNodeRecall"]
            tr = row["RelTripletRecall"]

            if (
                nr != "NaN"
                and float(nr) >= node_sat_threshold
                and node_first == max_level
            ):
                node_first = lvl
            if (
                tr != "NaN"
                and float(tr) >= trip_sat_threshold
                and trip_first == max_level
            ):
                trip_first = lvl

        node_first_levels.append(node_first)
        trip_first_levels.append(trip_first)
        gap = trip_first - node_first
        gaps.append(gap)

        # Check if image has any triplets at level 20
        ref_trips = _to_triplet_set(pseudo_gt[img_id]["relationships"])
        has_trips = len(ref_trips) > 0
        print(
            f"  {img_id}: node_first={node_first}, trip_first={trip_first}, "
            f"gap={gap:+d}, has_trips={has_trips}"
        )

    mean_gap = float(np.mean(gaps))
    print(f"\n  Mean gap (trip_first - node_first at 0.9): {mean_gap:+.1f} levels")
    print(
        "  Interpretation: relations saturate LATER than nodes by "
        f"~{abs(mean_gap):.1f} levels on average"
        if mean_gap > 0
        else "  Interpretation: relations saturate at similar/earlier levels"
    )

    # ===================================================================
    # STEP 6: Figures
    # ===================================================================
    print()
    print("=" * 60)
    print("STEP 6: Generate figures")
    print("=" * 60)

    # Build per-image relative metric lookup for plotting
    # {img_id: {level: {metric_name: value}}}
    img_rel_data: dict[int, dict[int, dict[str, float]]] = {}
    for r in rel_metrics_rows:
        iid = int(r["image_id"])
        lvl = int(r["level"])
        img_rel_data.setdefault(iid, {})[lvl] = {
            "RelNodeRecall": float(r["RelNodeRecall"])
            if r["RelNodeRecall"] != "NaN"
            else float("nan"),
            "RelTripletRecall": float(r["RelTripletRecall"])
            if r["RelTripletRecall"] != "NaN"
            else float("nan"),
            "RelSGScore": float(r["RelSGScore"])
            if r["RelSGScore"] != "NaN"
            else float("nan"),
        }

    # LLM-4b stop map (for vertical lines)
    llm_4b_stop = abl_map.get(base_4b_slug, {})

    # --- Figure 6a: Per-image relative curves ---
    levels_range = list(range(1, max_level + 1))

    for img_id in image_ids:
        fig, ax = plt.subplots(figsize=(10, 6))

        nr_vals = [img_rel_data[img_id][lv]["RelNodeRecall"] for lv in levels_range]
        tr_vals = [img_rel_data[img_id][lv]["RelTripletRecall"] for lv in levels_range]
        sg_vals = [img_rel_data[img_id][lv]["RelSGScore"] for lv in levels_range]

        ax.plot(
            levels_range,
            nr_vals,
            "o-",
            label="RelNodeRecall",
            color="#4C72B0",
            markersize=4,
        )
        ax.plot(
            levels_range,
            tr_vals,
            "s-",
            label="RelTripletRecall",
            color="#DD8452",
            markersize=4,
        )
        ax.plot(
            levels_range,
            sg_vals,
            "D-",
            label="RelSGScore",
            color="#55A868",
            markersize=4,
            linewidth=2,
        )

        # Horizontal threshold line
        ax.axhline(
            y=main_rel_threshold,
            color="gray",
            linestyle=":",
            alpha=0.7,
            label=f"Threshold={main_rel_threshold}",
        )

        # Vertical lines for stops
        rel_stop = rel_oracle_map[img_id]
        ax.axvline(
            x=rel_stop,
            color="green",
            linestyle="--",
            alpha=0.8,
            label=f"RelOracle(0.8)={rel_stop}",
        )

        baseline_stop_val = baseline_map.get(img_id, max_level)
        ax.axvline(
            x=baseline_stop_val,
            color="orange",
            linestyle="--",
            alpha=0.8,
            label=f"Baseline={baseline_stop_val}",
        )

        llm4b_stop_val = llm_4b_stop.get(img_id, max_level)
        ax.axvline(
            x=llm4b_stop_val,
            color="red",
            linestyle="--",
            alpha=0.8,
            label=f"LLM-4b={llm4b_stop_val}",
        )

        ax.set_xlabel("Resolution Level")
        ax.set_ylabel("Relative Metric Value")
        ax.set_title(f"Relative Saturation Curves - Image {img_id}")
        ax.set_xlim(0.5, max_level + 0.5)
        ax.set_ylim(-0.05, 1.10)
        ax.set_xticks(levels_range)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_root / f"relative_curve_{img_id}.png", dpi=150)
        plt.close(fig)

    print(f"  Saved {len(image_ids)} per-image relative curve figures")

    # --- Figure 6b: Method comparison MAE (horizontal bar) ---
    fig, ax = plt.subplots(figsize=(12, 8))
    method_names = [r["method"] for r in reversed(comparison_rows)]
    maes = [float(r["mean_abs_error"]) for r in reversed(comparison_rows)]
    early_rates = [float(r["early_stop_rate"]) for r in reversed(comparison_rows)]
    late_rates = [float(r["late_stop_rate"]) for r in reversed(comparison_rows)]

    colors = []
    for er, lr in zip(early_rates, late_rates, strict=True):
        if er > lr:
            colors.append("#e8746a")  # red-ish = tends early
        elif lr > er:
            colors.append("#6a9fe8")  # blue-ish = tends late
        else:
            colors.append("#999999")  # neutral

    y_pos = np.arange(len(method_names))
    bars = ax.barh(y_pos, maes, color=colors, edgecolor="black", linewidth=0.5)

    for bar, er, lr in zip(bars, early_rates, late_rates, strict=True):
        w = bar.get_width()
        label = f"E{er:.0%}/L{lr:.0%}"
        ax.text(
            w + 0.15,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=7,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_names, fontsize=8)
    ax.set_xlabel("Mean Absolute Error (levels)")
    ax.set_title(f"Method Comparison: MAE vs RelOracle({main_rel_threshold})")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#e8746a", edgecolor="black", label="Tends early"),
        Patch(facecolor="#6a9fe8", edgecolor="black", label="Tends late"),
        Patch(facecolor="#999999", edgecolor="black", label="Balanced"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(figures_root / "relative_method_comparison_mae.png", dpi=150)
    plt.close(fig)
    print("  Saved relative_method_comparison_mae.png")

    # --- Figure 6c: RelOracle vs GT Oracle scatter ---
    fig, ax = plt.subplots(figsize=(8, 8))

    gt_stops = [old_oracle_map[i] for i in image_ids]
    rel_stops = [rel_oracle_map[i] for i in image_ids]

    ax.scatter(gt_stops, rel_stops, s=80, c="#4C72B0", edgecolors="black", zorder=5)

    # Annotate points with image_id
    for img_id, gx, ry in zip(image_ids, gt_stops, rel_stops, strict=True):
        ax.annotate(
            str(img_id),
            (gx, ry),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
            alpha=0.8,
        )

    # Reference line (y=x)
    ax.plot([0, max_level + 1], [0, max_level + 1], "k--", alpha=0.3, label="y=x")

    # Mean lines
    ax.axvline(
        x=gt_mean,
        color="orange",
        linestyle=":",
        alpha=0.7,
        label=f"GT mean={gt_mean:.1f}",
    )
    ax.axhline(
        y=rel_mean,
        color="green",
        linestyle=":",
        alpha=0.7,
        label=f"Rel mean={rel_mean:.1f}",
    )

    ax.set_xlabel("GT Oracle(0.5) Stop Level")
    ax.set_ylabel(f"RelOracle({main_rel_threshold}) Stop Level")
    ax.set_title("Relative vs Absolute Oracle Stop Comparison")
    ax.set_xlim(0, max_level + 1)
    ax.set_ylim(0, max_level + 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_root / "relative_vs_absolute_oracle.png", dpi=150)
    plt.close(fig)
    print("  Saved relative_vs_absolute_oracle.png")

    # ===================================================================
    # Write research report
    # ===================================================================
    print()
    print("=" * 60)
    print("Writing research report")
    print("=" * 60)

    report_path = (
        REPO_ROOT / ".claude" / "docs" / "research" / "m1-relative-saturation.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Milestone-1: Relative Saturation Analysis (Plan B)")
    lines.append("")
    lines.append("Generated by `scripts/15_relative_saturation.py`")
    lines.append("")

    lines.append("## Definition")
    lines.append("")
    lines.append(
        "**Relative saturation** measures convergence of EGTR output across "
        "resolution levels, using each image's own Level-20 (full-resolution) "
        "output as the reference (pseudo-GT), instead of the dataset's true "
        "ground truth scene graph."
    )
    lines.append("")
    lines.append("- `RelNodeRecall(t) = |nodes(t) ∩ nodes(20)| / |nodes(20)|`")
    lines.append(
        "- `RelTripletRecall(t) = |triplets(t) ∩ triplets(20)| / |triplets(20)|`"
    )
    lines.append("- `RelSGScore(t) = 0.5 * RelNodeRecall + 0.5 * RelTripletRecall`")
    lines.append("- By construction, `RelSGScore(20) = 1.0`")
    lines.append("")
    lines.append(
        "**Caveat**: This measures CONVERGENCE/STABILITY of extraction, "
        "NOT correctness. Correctness is reported separately via the "
        "paper-protocol R@K (see `egtr-repro.md`)."
    )
    lines.append("")

    lines.append("## Non-degeneracy Check")
    lines.append("")
    lines.append(
        f"- GT Oracle(0.5): {gt_saturated}/{len(image_ids)} images saturate "
        f"(mean stop = {gt_mean:.1f})"
    )
    lines.append(
        f"- RelOracle(0.8): {rel_saturated}/{len(image_ids)} images saturate "
        f"(mean stop = {rel_mean:.1f})"
    )
    lines.append(
        f"- With relative reference, "
        f"{'ALL' if rel_saturated == len(image_ids) else str(rel_saturated)} "
        f"images now have a reachable and meaningful saturation point."
    )
    lines.append("")

    lines.append("## RelOracle vs GT Oracle Comparison")
    lines.append("")
    lines.append("| image_id | GT_Oracle_0.5 | RelOracle_0.8 | Diff |")
    lines.append("|----------|---------------|---------------|------|")
    for img_id in image_ids:
        gt_s = old_oracle_map[img_id]
        rel_s = rel_oracle_map[img_id]
        lines.append(f"| {img_id} | {gt_s} | {rel_s} | {rel_s - gt_s:+d} |")
    lines.append("")
    lines.append(
        f"Mean diff (Rel - GT): {np.mean(diffs):+.1f} levels. "
        f"Relative saturation generally occurs "
        f"{'EARLIER' if np.mean(diffs) < 0 else 'LATER'} than absolute."
    )
    lines.append("")

    lines.append("## Method Comparison vs RelOracle(0.8) (sorted by MAE)")
    lines.append("")
    lines.append(
        "| Method | Mean Stop | Mean Oracle | MAE | Median AE | Early% | Late% | Pearson r |"
    )
    lines.append(
        "|--------|-----------|-------------|-----|-----------|--------|-------|-----------|"
    )
    for r in comparison_rows:
        lines.append(
            f"| {r['method']} | {r['mean_stop']} | {r['mean_rel_oracle_stop']} | "
            f"{r['mean_abs_error']} | {r['median_abs_error']} | "
            f"{r['early_stop_rate']} | {r['late_stop_rate']} | {r['pearson_r']} |"
        )
    lines.append("")

    # Find best method
    best = comparison_rows[0]
    lines.append(
        f"**Best method**: `{best['method']}` with MAE={best['mean_abs_error']}"
    )
    lines.append("")

    lines.append("## Node vs Relation Saturation Gap (Relative Reference)")
    lines.append("")
    lines.append(
        f"Threshold for both: 0.9. Mean gap (trip_first - node_first): "
        f"{mean_gap:+.1f} levels."
    )
    lines.append("")
    lines.append("| image_id | NodeFirst@0.9 | TripFirst@0.9 | Gap |")
    lines.append("|----------|---------------|---------------|-----|")
    for img_id, nf, tf, g in zip(
        image_ids, node_first_levels, trip_first_levels, gaps, strict=True
    ):
        lines.append(f"| {img_id} | {nf} | {tf} | {g:+d} |")
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    lines.append("- `outputs/figures/relative_curve_{image_id}.png` - per-image curves")
    lines.append(
        "- `outputs/figures/relative_method_comparison_mae.png` - MAE bar chart"
    )
    lines.append(
        "- `outputs/figures/relative_vs_absolute_oracle.png` - scatter comparison"
    )
    lines.append("")

    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text)
    print(f"  Saved {report_path}")

    # ===================================================================
    # Summary
    # ===================================================================
    print()
    print("=" * 60)
    print("DONE. Files created:")
    print("=" * 60)
    print(f"  {metrics_root / 'relative_metrics.csv'}")
    print(f"  {metrics_root / 'relative_oracle_stop.csv'}")
    print(f"  {metrics_root / 'method_comparison_relative.csv'}")
    for img_id in image_ids:
        print(f"  {figures_root / f'relative_curve_{img_id}.png'}")
    print(f"  {figures_root / 'relative_method_comparison_mae.png'}")
    print(f"  {figures_root / 'relative_vs_absolute_oracle.png'}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
