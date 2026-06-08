"""Post-hoc aggregation for 100-image focused evaluation.

Computes:
1. Consecutive-k=2 stops for focused LLM configs (4b, 9b)
2. Hybrid stop (confidence + LLM fusion)
3. Method comparison vs true-GT OracleStop(0.5) -> method_comparison_100.csv
4. Method comparison vs RelOracleStop(0.8) -> method_comparison_relative_100.csv
5. Saturation rate table (RelOracle at 0.6/0.7/0.8/0.9)
6. Figures: relative_method_comparison_mae_100.png, stop_distribution_all_100.png,
   saturation_rate_100.png

Methods included:
  baseline; llm_sgg_4b_h5 (k1,k2); llm_sgg_9b_h5 (k1,k2);
  vlm_4b_single (k1,k2); vlm_9b_single (k1,k2);
  hybrid_4b (k1,k2); hybrid_9b (k1,k2).
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


# ── Utilities ──


def load_csv_dict(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def save_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {path}")


def pearson_r(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    ax = np.array(xs, dtype=float)
    ay = np.array(ys, dtype=float)
    if np.std(ax) == 0 or np.std(ay) == 0:
        return float("nan")
    return float(np.corrcoef(ax, ay)[0, 1])


def median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def load_saturation_scores(
    base_dir: Path, image_id: int, start_level: int, max_level: int
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for lvl in range(start_level, max_level + 1):
        p = base_dir / str(image_id) / f"level_{lvl:02d}.json"
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            scores[lvl] = float(data["saturation_score"])
    return scores


def consecutive_k_stop(
    scores: dict[int, float],
    threshold: float,
    k: int,
    start_level: int,
    max_level: int,
) -> int:
    for lvl in range(start_level, max_level + 1):
        all_above = True
        for offset in range(k):
            check_lvl = lvl - offset
            if check_lvl < start_level:
                all_above = False
                break
            if scores.get(check_lvl, 0.0) < threshold:
                all_above = False
                break
        if all_above:
            return lvl
    return max_level


def load_egtr_prediction(pred_dir: Path, image_id: int, level: int) -> dict:
    p = pred_dir / str(image_id) / f"level_{level:02d}.json"
    with open(p) as f:
        return json.load(f)


def compute_relative_metrics(
    pred_objects: list[dict],
    pred_relationships: list[dict],
    ref_objects: list[dict],
    ref_relationships: list[dict],
) -> dict[str, float]:
    pred_nodes = _to_label_set(pred_objects)
    ref_nodes = _to_label_set(ref_objects)
    pred_trips = _to_triplet_set(pred_relationships)
    ref_trips = _to_triplet_set(ref_relationships)

    if len(ref_nodes) == 0:
        node_recall = float("nan")
    else:
        node_recall = len(pred_nodes & ref_nodes) / len(ref_nodes)

    if len(ref_trips) == 0:
        trip_recall = float("nan")
    else:
        trip_recall = len(pred_trips & ref_trips) / len(ref_trips)

    if math.isnan(trip_recall) if isinstance(trip_recall, float) else False:
        rel_sg_score = node_recall if not math.isnan(node_recall) else float("nan")
    elif math.isnan(node_recall) if isinstance(node_recall, float) else False:
        rel_sg_score = trip_recall
    else:
        rel_sg_score = 0.5 * node_recall + 0.5 * trip_recall

    return {
        "RelNodeRecall": node_recall,
        "RelTripletRecall": trip_recall,
        "RelSGScore": rel_sg_score,
    }


# ── Main ──


def main() -> None:
    cfg = load_config()
    max_level = cfg["resolution_levels"]["num_levels"]
    start_level = cfg["llm"]["start_level"]
    sat_threshold = cfg["llm"]["saturation_threshold"]
    main_oracle_t = cfg["metrics"]["main_oracle_threshold"]
    main_baseline_t = cfg["baseline"]["mean_object_conf_threshold"]
    gain_epsilon = cfg["metrics"]["gain_epsilon"]
    gain_k = cfg["metrics"]["gain_k"]

    metrics_root = resolve(cfg["paths"]["metrics"])
    figures_root = resolve(cfg["paths"]["figures"])
    pred_root = resolve(cfg["paths"]["egtr_predictions"])
    figures_root.mkdir(parents=True, exist_ok=True)

    selected_path = resolve(cfg["paths"]["selected_images"])
    with open(selected_path) as f:
        image_ids: list[int] = json.load(f)

    n_images = len(image_ids)
    print(f"Processing {n_images} images")

    # ── Load base data ──
    all_metrics = load_csv_dict(metrics_root / "all_metrics.csv")
    oracle_rows = load_csv_dict(metrics_root / "oracle_stop.csv")
    baseline_rows = load_csv_dict(metrics_root / "baseline_stop.csv")
    ablation_rows = load_csv_dict(metrics_root / "llm_ablation_stop.csv")
    vlm_rows = load_csv_dict(metrics_root / "vlm_stop.csv")

    oracle_map = {
        int(r["image_id"]): int(float(r[f"oracle_{main_oracle_t}"]))
        for r in oracle_rows
    }
    baseline_map = {
        int(r["image_id"]): int(float(r[f"baseline_{main_baseline_t}"]))
        for r in baseline_rows
    }

    # Ablation map
    abl_map: dict[str, dict[int, int]] = {}
    for r in ablation_rows:
        slug = r["config_slug"]
        abl_map.setdefault(slug, {})[int(r["image_id"])] = int(r["stop_level"])

    # VLM map
    vlm_map: dict[tuple[str, str, int], dict[int, int]] = {}
    for r in vlm_rows:
        key = (r["model"], r["mode"], int(r["k"]))
        vlm_map.setdefault(key, {})[int(r["image_id"])] = int(r["stop_level"])

    # Metrics index
    metrics_by_img_lvl: dict[tuple[int, int], dict] = {}
    for r in all_metrics:
        metrics_by_img_lvl[(int(r["image_id"]), int(r["level"]))] = r

    base_4b_slug = "qwen3.5-4b_h5_rel_on_conf_on"
    base_9b_slug = "qwen3.5-9b_h5_rel_on_conf_on"
    ablation_dir = resolve("outputs/llm_ablation")

    # ── STEP 1: k=2 stops ──
    print("\n" + "=" * 60)
    print("STEP 1: Consecutive-k=2 post-hoc stops for focused LLM configs")
    print("=" * 60)

    abl_k2_map: dict[str, dict[int, int]] = {}
    for slug in [base_4b_slug, base_9b_slug]:
        abl_k2_map[slug] = {}
        slug_dir = ablation_dir / slug
        for img_id in image_ids:
            scores = load_saturation_scores(slug_dir, img_id, start_level, max_level)
            stop_k2 = consecutive_k_stop(
                scores, sat_threshold, 2, start_level, max_level
            )
            abl_k2_map[slug][img_id] = stop_k2

    # Save k2 CSV
    abl_k2_rows = []
    for slug in [base_4b_slug, base_9b_slug]:
        for img_id in image_ids:
            abl_k2_rows.append(
                {
                    "config_slug": slug,
                    "image_id": img_id,
                    "stop_k1": abl_map[slug].get(img_id, max_level),
                    "stop_k2": abl_k2_map[slug].get(img_id, max_level),
                }
            )
    save_csv(
        abl_k2_rows,
        metrics_root / "llm_ablation_k2_stop.csv",
        ["config_slug", "image_id", "stop_k1", "stop_k2"],
    )

    # ── STEP 2: Hybrid stop ──
    print("\n" + "=" * 60)
    print("STEP 2: Hybrid stop (confidence + LLM fusion)")
    print("=" * 60)

    hybrid_cfg = cfg["ablation"]["hybrid"]
    conf_w = hybrid_cfg["conf_weight"]
    llm_w = hybrid_cfg["llm_weight"]
    hybrid_threshold = hybrid_cfg["threshold"]

    hybrid_rows: list[dict] = []
    for variant_label, slug in [
        ("hybrid_4b", base_4b_slug),
        ("hybrid_9b", base_9b_slug),
    ]:
        slug_dir = ablation_dir / slug
        for k_val in [1, 2]:
            variant_name = f"{variant_label}_k{k_val}"
            for img_id in image_ids:
                sat_scores = load_saturation_scores(
                    slug_dir, img_id, start_level, max_level
                )
                fused: dict[int, float] = {}
                for lvl in range(start_level, max_level + 1):
                    key = (img_id, lvl)
                    if key in metrics_by_img_lvl and lvl in sat_scores:
                        obj_conf = float(
                            metrics_by_img_lvl[key]["mean_object_confidence"]
                        )
                        fused[lvl] = conf_w * obj_conf + llm_w * sat_scores[lvl]
                    else:
                        fused[lvl] = 0.0
                stop = consecutive_k_stop(
                    fused, hybrid_threshold, k_val, start_level, max_level
                )
                hybrid_rows.append(
                    {
                        "image_id": img_id,
                        "variant": variant_name,
                        "stop_level": stop,
                    }
                )

    save_csv(
        hybrid_rows,
        metrics_root / "hybrid_stop.csv",
        ["image_id", "variant", "stop_level"],
    )

    hybrid_map: dict[str, dict[int, int]] = {}
    for r in hybrid_rows:
        hybrid_map.setdefault(r["variant"], {})[r["image_id"]] = r["stop_level"]

    # ── STEP 3: Relative metrics + RelOracle ──
    print("\n" + "=" * 60)
    print("STEP 3: Relative metrics + RelOracle")
    print("=" * 60)

    rel_oracle_thresholds = [0.6, 0.7, 0.8, 0.9]
    main_rel_threshold = 0.8

    # Build pseudo-GT from level_20
    pseudo_gt: dict[int, dict] = {}
    for img_id in image_ids:
        pseudo_gt[img_id] = load_egtr_prediction(pred_root, img_id, max_level)

    # Compute relative metrics
    rel_sg_map: dict[tuple[int, int], float] = {}
    rel_node_map: dict[tuple[int, int], float] = {}
    rel_trip_map: dict[tuple[int, int], float] = {}

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
            rel_sg_map[(img_id, lvl)] = (
                m["RelSGScore"] if not math.isnan(m["RelSGScore"]) else float("nan")
            )
            rel_node_map[(img_id, lvl)] = (
                m["RelNodeRecall"]
                if not math.isnan(m["RelNodeRecall"])
                else float("nan")
            )
            rel_trip_map[(img_id, lvl)] = (
                m["RelTripletRecall"]
                if not math.isnan(m["RelTripletRecall"])
                else float("nan")
            )

    # RelOracle stops
    rel_oracle_map: dict[int, int] = {}
    rel_oracle_all: dict[float, dict[int, int]] = {t: {} for t in rel_oracle_thresholds}

    for img_id in image_ids:
        for t in rel_oracle_thresholds:
            stop = max_level
            for lvl in range(1, max_level + 1):
                val = rel_sg_map.get((img_id, lvl), float("nan"))
                if not math.isnan(val) and val >= t:
                    stop = lvl
                    break
            rel_oracle_all[t][img_id] = stop

        rel_oracle_map[img_id] = rel_oracle_all[main_rel_threshold][img_id]

    # Saturation rates
    print("\n  Saturation rates:")
    for t in rel_oracle_thresholds:
        sat = sum(1 for i in image_ids if rel_oracle_all[t][i] < max_level)
        print(f"    RelOracle({t}): {sat}/{n_images} ({sat / n_images:.0%})")

    gt_saturated = sum(1 for i in image_ids if oracle_map[i] < max_level)
    rel_saturated = sum(1 for i in image_ids if rel_oracle_map[i] < max_level)
    gt_mean = float(np.mean([oracle_map[i] for i in image_ids]))
    rel_mean = float(np.mean([rel_oracle_map[i] for i in image_ids]))
    print(
        f"\n  GT Oracle(0.5): saturated={gt_saturated}/{n_images}, mean={gt_mean:.1f}"
    )
    print(
        f"  RelOracle(0.8): saturated={rel_saturated}/{n_images}, mean={rel_mean:.1f}"
    )

    # ── STEP 4: Build all method maps ──
    print("\n" + "=" * 60)
    print("STEP 4: Build method maps")
    print("=" * 60)

    methods: dict[str, dict[int, int]] = {}
    methods["baseline"] = baseline_map
    methods["llm_sgg_4b_h5_k1"] = abl_map.get(base_4b_slug, {})
    methods["llm_sgg_4b_h5_k2"] = abl_k2_map.get(base_4b_slug, {})
    methods["llm_sgg_9b_h5_k1"] = abl_map.get(base_9b_slug, {})
    methods["llm_sgg_9b_h5_k2"] = abl_k2_map.get(base_9b_slug, {})

    vlm_method_names = {
        ("qwen3.5:4b", "single", 1): "vlm_4b_single_k1",
        ("qwen3.5:4b", "single", 2): "vlm_4b_single_k2",
        ("qwen3.5:9b", "single", 1): "vlm_9b_single_k1",
        ("qwen3.5:9b", "single", 2): "vlm_9b_single_k2",
    }
    for vlm_key, mname in vlm_method_names.items():
        if vlm_key in vlm_map:
            methods[mname] = vlm_map[vlm_key]

    for v, stops in hybrid_map.items():
        methods[v] = stops

    # ── STEP 5: Comparison vs true-GT Oracle(0.5) ──
    print("\n" + "=" * 60)
    print("STEP 5: Method comparison vs true-GT Oracle(0.5)")
    print("=" * 60)

    def compute_comparison(
        methods_dict: dict[str, dict[int, int]],
        ref_map: dict[int, int],
        ref_label: str,
    ) -> list[dict]:
        ref_stops = [ref_map[i] for i in image_ids]
        rows = []
        for mname, stop_map in methods_dict.items():
            method_stops = [stop_map.get(i, max_level) for i in image_ids]
            errors = [m - o for m, o in zip(method_stops, ref_stops, strict=True)]
            abs_errors = [abs(e) for e in errors]
            mean_stop = float(np.mean(method_stops))
            mean_ae = float(np.mean(abs_errors))
            median_ae = median(abs_errors)
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
                    f"mean_{ref_label}_stop": round(float(np.mean(ref_stops)), 2),
                    "mean_abs_error": round(mean_ae, 2),
                    "median_abs_error": round(median_ae, 1),
                    "early_stop_rate": round(early, 2),
                    "late_stop_rate": round(late, 2),
                    "pearson_r": round(corr, 3) if not math.isnan(corr) else "NaN",
                }
            )
        rows.sort(key=lambda r: float(r["mean_abs_error"]))
        return rows

    # vs GT Oracle(0.5)
    gt_comparison = compute_comparison(methods, oracle_map, "oracle")
    save_csv(
        gt_comparison,
        metrics_root / "method_comparison_100.csv",
        list(gt_comparison[0].keys()),
    )

    print(
        f"\n  {'Method':<28s} {'MeanStop':>8s} {'MAE':>6s} "
        f"{'MedAE':>6s} {'Early':>6s} {'Late':>6s} {'Corr':>7s}"
    )
    print("  " + "-" * 75)
    for r in gt_comparison:
        print(
            f"  {r['method']:<28s} {r['mean_stop']:>8} {r['mean_abs_error']:>6} "
            f"{r['median_abs_error']:>6} {r['early_stop_rate']:>6} "
            f"{r['late_stop_rate']:>6} {str(r['pearson_r']):>7}"
        )

    # vs RelOracle(0.8)
    print("\n" + "=" * 60)
    print("STEP 6: Method comparison vs RelOracle(0.8)")
    print("=" * 60)

    rel_comparison = compute_comparison(methods, rel_oracle_map, "rel_oracle")
    save_csv(
        rel_comparison,
        metrics_root / "method_comparison_relative_100.csv",
        list(rel_comparison[0].keys()),
    )

    print(
        f"\n  {'Method':<28s} {'MeanStop':>8s} {'MAE':>6s} "
        f"{'MedAE':>6s} {'Early':>6s} {'Late':>6s} {'Corr':>7s}"
    )
    print("  " + "-" * 75)
    for r in rel_comparison:
        print(
            f"  {r['method']:<28s} {r['mean_stop']:>8} {r['mean_abs_error']:>6} "
            f"{r['median_abs_error']:>6} {r['early_stop_rate']:>6} "
            f"{r['late_stop_rate']:>6} {str(r['pearson_r']):>7}"
        )

    # ── STEP 7: Node vs Relation gap ──
    print("\n" + "=" * 60)
    print("STEP 7: Node vs Relation saturation gap")
    print("=" * 60)

    node_sat_threshold = 0.9
    trip_sat_threshold = 0.9
    gaps = []
    for img_id in image_ids:
        node_first = max_level
        trip_first = max_level
        for lvl in range(1, max_level + 1):
            nr = rel_node_map.get((img_id, lvl), float("nan"))
            tr = rel_trip_map.get((img_id, lvl), float("nan"))
            if (
                not math.isnan(nr)
                and nr >= node_sat_threshold
                and node_first == max_level
            ):
                node_first = lvl
            if (
                not math.isnan(tr)
                and tr >= trip_sat_threshold
                and trip_first == max_level
            ):
                trip_first = lvl
        gaps.append(trip_first - node_first)
    mean_gap = float(np.mean(gaps))
    print(f"  Mean gap (trip_first - node_first at 0.9): {mean_gap:+.1f} levels")

    # ── STEP 8: Figures ──
    print("\n" + "=" * 60)
    print("STEP 8: Figures")
    print("=" * 60)

    # Figure 1: Method comparison MAE vs RelOracle(0.8) horizontal bar
    fig, ax = plt.subplots(figsize=(12, 8))
    method_names = [r["method"] for r in reversed(rel_comparison)]
    maes = [float(r["mean_abs_error"]) for r in reversed(rel_comparison)]
    early_rates = [float(r["early_stop_rate"]) for r in reversed(rel_comparison)]
    late_rates = [float(r["late_stop_rate"]) for r in reversed(rel_comparison)]

    colors = []
    for er, lr in zip(early_rates, late_rates, strict=True):
        if er > lr:
            colors.append("#e8746a")
        elif lr > er:
            colors.append("#6a9fe8")
        else:
            colors.append("#999999")

    y_pos = np.arange(len(method_names))
    bars = ax.barh(y_pos, maes, color=colors, edgecolor="black", linewidth=0.5)
    for bar, er, lr in zip(bars, early_rates, late_rates, strict=True):
        w = bar.get_width()
        label = f"E{er:.0%}/L{lr:.0%}"
        ax.text(
            w + 0.15, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=7
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_names, fontsize=8)
    ax.set_xlabel("Mean Absolute Error (levels)")
    ax.set_title(f"Method Comparison: MAE vs RelOracle(0.8) [N={n_images}]")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#e8746a", edgecolor="black", label="Tends early"),
        Patch(facecolor="#6a9fe8", edgecolor="black", label="Tends late"),
        Patch(facecolor="#999999", edgecolor="black", label="Balanced"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(figures_root / "relative_method_comparison_mae_100.png", dpi=150)
    plt.close(fig)
    print("  Saved relative_method_comparison_mae_100.png")

    # Figure 2: Stop distribution box plot
    plot_methods_order = [
        "baseline",
        "llm_sgg_4b_h5_k1",
        "llm_sgg_4b_h5_k2",
        "llm_sgg_9b_h5_k1",
        "llm_sgg_9b_h5_k2",
        "vlm_4b_single_k1",
        "vlm_4b_single_k2",
        "vlm_9b_single_k1",
        "vlm_9b_single_k2",
        "hybrid_4b_k1",
        "hybrid_4b_k2",
        "hybrid_9b_k1",
        "hybrid_9b_k2",
    ]
    plot_methods = [m for m in plot_methods_order if m in methods]

    fig, ax = plt.subplots(figsize=(14, 7))
    box_data = []
    box_labels = []
    for mname in plot_methods:
        stop_map = methods[mname]
        stops = [stop_map.get(i, max_level) for i in image_ids]
        box_data.append(stops)
        box_labels.append(mname)

    oracle_stops = [oracle_map[i] for i in image_ids]
    rel_oracle_stops = [rel_oracle_map[i] for i in image_ids]
    box_data.extend([oracle_stops, rel_oracle_stops])
    box_labels.extend(["oracle(0.5)", "rel_oracle(0.8)"])

    bp = ax.boxplot(box_data, vert=True, patch_artist=True, widths=0.6)
    n_methods = len(box_labels)
    cmap = plt.cm.tab20
    for i, patch in enumerate(bp["boxes"]):
        if box_labels[i] in ("oracle(0.5)", "rel_oracle(0.8)"):
            patch.set_facecolor("#90EE90")
            patch.set_alpha(0.8)
        else:
            patch.set_facecolor(cmap(i / max(n_methods - 1, 1)))
            patch.set_alpha(0.6)

    rng = np.random.default_rng(42)
    for i, data in enumerate(box_data):
        jitter = rng.uniform(-0.15, 0.15, len(data))
        ax.scatter(
            np.full(len(data), i + 1) + jitter,
            data,
            alpha=0.3,
            s=10,
            zorder=5,
            color="black",
        )

    oracle_mean = np.mean(oracle_stops)
    rel_oracle_mean = np.mean(rel_oracle_stops)
    ax.axhline(
        y=oracle_mean,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Oracle mean={oracle_mean:.1f}",
    )
    ax.axhline(
        y=rel_oracle_mean,
        color="blue",
        linestyle=":",
        alpha=0.7,
        label=f"RelOracle mean={rel_oracle_mean:.1f}",
    )

    ax.set_xticks(range(1, n_methods + 1))
    ax.set_xticklabels(box_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Stop Level")
    ax.set_title(f"Stop Level Distribution by Method [N={n_images}]")
    ax.set_ylim(0, max_level + 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures_root / "stop_distribution_all_100.png", dpi=150)
    plt.close(fig)
    print("  Saved stop_distribution_all_100.png")

    # Figure 3: Saturation rate comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    thresholds = [0.6, 0.7, 0.8, 0.9]
    sat_rates = []
    for t in thresholds:
        sat = sum(1 for i in image_ids if rel_oracle_all[t][i] < max_level)
        sat_rates.append(sat / n_images)

    gt_sat_rate = gt_saturated / n_images

    ax.bar(
        [str(t) for t in thresholds],
        sat_rates,
        color="#4C72B0",
        edgecolor="black",
        label="RelOracle saturation rate",
    )
    ax.axhline(
        y=gt_sat_rate,
        color="red",
        linestyle="--",
        alpha=0.8,
        label=f"GT Oracle(0.5) rate={gt_sat_rate:.0%}",
    )
    for i, (t, r) in enumerate(zip(thresholds, sat_rates)):
        ax.text(i, r + 0.02, f"{r:.0%}", ha="center", fontsize=10)

    ax.set_xlabel("RelOracle Threshold")
    ax.set_ylabel("Saturation Rate")
    ax.set_title(f"Saturation Rate by RelOracle Threshold [N={n_images}]")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures_root / "saturation_rate_100.png", dpi=150)
    plt.close(fig)
    print("  Saved saturation_rate_100.png")

    # ── STEP 9: Count parse failures ──
    print("\n" + "=" * 60)
    print("STEP 9: Parse failure counts")
    print("=" * 60)

    for slug in [base_4b_slug, base_9b_slug]:
        slug_dir = ablation_dir / slug
        total = 0
        failures = 0
        for img_id in image_ids:
            for lvl in range(start_level, max_level + 1):
                p = slug_dir / str(img_id) / f"level_{lvl:02d}.json"
                if p.exists():
                    with open(p) as f:
                        data = json.load(f)
                    total += 1
                    reason = data.get("reason", "")
                    if "parse" in reason or "error" in reason:
                        failures += 1
        print(
            f"  LLM {slug}: {failures}/{total} failures ({failures / max(total, 1) * 100:.1f}%)"
        )

    vlm_root = resolve("outputs/vlm_scores")
    for model_name in ["qwen3.5:4b", "qwen3.5:9b"]:
        slug = model_name.replace(":", "-")
        mode_dir = vlm_root / slug / "single"
        total = 0
        failures = 0
        for img_id in image_ids:
            for lvl in range(start_level, max_level + 1):
                p = mode_dir / str(img_id) / f"level_{lvl:02d}.json"
                if p.exists():
                    with open(p) as f:
                        data = json.load(f)
                    total += 1
                    reason = data.get("reason", "")
                    if "parse" in reason or "error" in reason:
                        failures += 1
        print(
            f"  VLM {model_name} single: {failures}/{total} failures ({failures / max(total, 1) * 100:.1f}%)"
        )

    # ── Summary data for report ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Images: {n_images}")
    print(
        f"  GT Oracle(0.5): saturated={gt_saturated}/{n_images} ({gt_saturated / n_images:.0%}), mean={gt_mean:.1f}"
    )
    print(
        f"  RelOracle(0.8): saturated={rel_saturated}/{n_images} ({rel_saturated / n_images:.0%}), mean={rel_mean:.1f}"
    )
    print(f"  Node-vs-Relation gap: {mean_gap:+.1f} levels")
    print("\n  Top-3 methods vs RelOracle(0.8):")
    for r in rel_comparison[:3]:
        print(f"    {r['method']}: MAE={r['mean_abs_error']}")

    print("\n  Files created:")
    print(f"    {metrics_root / 'method_comparison_100.csv'}")
    print(f"    {metrics_root / 'method_comparison_relative_100.csv'}")
    print(f"    {figures_root / 'relative_method_comparison_mae_100.png'}")
    print(f"    {figures_root / 'stop_distribution_all_100.png'}")
    print(f"    {figures_root / 'saturation_rate_100.png'}")


if __name__ == "__main__":
    main()
