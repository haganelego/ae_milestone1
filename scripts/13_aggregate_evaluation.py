"""Post-hoc aggregation for Milestone-1 SGG-saturation evaluation.

Reads cached scores and stop CSVs, computes:
1. Consecutive-k=2 post-hoc stops for LLM ablation configs
2. SG-score weight ablation (balanced vs rel-heavy oracle stops)
3. Hybrid stop (confidence + LLM fusion) for 4b and 9b
4. Unified method comparison vs Oracle(0.5)
5. Three summary figures

Run:
  cd /home/kanaoka/usr/github/milestone_1 && \
  uv run --project egtr python scripts/13_aggregate_evaluation.py
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


def load_saturation_scores(
    base_dir: Path, image_id: int, start_level: int, max_level: int
) -> dict[int, float]:
    """Load per-level saturation_score from cached JSON files.

    Returns {level: score} for levels that have cached files.
    """
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
    """First level t where the previous k scores (t-k+1..t) all >= threshold.

    For k=1 this is just the first level >= threshold.
    For k=2 this is the first level t where score(t)>=thr AND score(t-1)>=thr.
    Returns max_level if never triggered.
    """
    for lvl in range(start_level, max_level + 1):
        # Check k consecutive levels ending at lvl
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = load_config()
    max_level = cfg["resolution_levels"]["num_levels"]  # 20
    start_level = cfg["llm"]["start_level"]  # 5
    sat_threshold = cfg["llm"]["saturation_threshold"]  # 0.7
    main_oracle_t = cfg["metrics"]["main_oracle_threshold"]  # 0.5
    main_baseline_t = cfg["baseline"]["mean_object_conf_threshold"]  # 0.4

    metrics_root = resolve(cfg["paths"]["metrics"])
    figures_root = resolve(cfg["paths"]["figures"])
    figures_root.mkdir(parents=True, exist_ok=True)

    selected_path = resolve(cfg["paths"]["selected_images"])
    with open(selected_path) as f:
        image_ids: list[int] = json.load(f)

    # Load base CSVs
    all_metrics = load_csv_dict(metrics_root / "all_metrics.csv")
    oracle_rows = load_csv_dict(metrics_root / "oracle_stop.csv")
    baseline_rows = load_csv_dict(metrics_root / "baseline_stop.csv")
    # llm_stop.csv is the legacy 4b run; ablation CSV covers it more precisely.
    vlm_rows = load_csv_dict(metrics_root / "vlm_stop.csv")
    ablation_rows = load_csv_dict(metrics_root / "llm_ablation_stop.csv")

    # Build lookup maps
    oracle_map = {
        int(r["image_id"]): int(float(r[f"oracle_{main_oracle_t}"]))
        for r in oracle_rows
    }
    baseline_map = {
        int(r["image_id"]): int(float(r[f"baseline_{main_baseline_t}"]))
        for r in baseline_rows
    }

    # VLM map: (model, mode, k) -> {image_id: stop}
    vlm_map: dict[tuple[str, str, int], dict[int, int]] = {}
    for r in vlm_rows:
        key = (r["model"], r["mode"], int(r["k"]))
        vlm_map.setdefault(key, {})[int(r["image_id"])] = int(r["stop_level"])

    # Ablation map: config_slug -> {image_id: stop}
    abl_map: dict[str, dict[int, int]] = {}
    for r in ablation_rows:
        slug = r["config_slug"]
        abl_map.setdefault(slug, {})[int(r["image_id"])] = int(r["stop_level"])

    # All-metrics indexed by (image_id, level)
    metrics_by_img_lvl: dict[tuple[int, int], dict] = {}
    for r in all_metrics:
        metrics_by_img_lvl[(int(r["image_id"]), int(r["level"]))] = r

    print("=" * 60)
    print("STEP 1: Consecutive-k=2 post-hoc stops for LLM ablation")
    print("=" * 60)

    ablation_dir = resolve("outputs/llm_ablation")
    config_slugs = sorted(abl_map.keys())

    # Build k=2 stops for each ablation config
    abl_k2_map: dict[str, dict[int, int]] = {}
    for slug in config_slugs:
        abl_k2_map[slug] = {}
        slug_dir = ablation_dir / slug
        for img_id in image_ids:
            scores = load_saturation_scores(slug_dir, img_id, start_level, max_level)
            stop_k2 = consecutive_k_stop(
                scores, sat_threshold, 2, start_level, max_level
            )
            abl_k2_map[slug][img_id] = stop_k2

    # Save unified ablation table with k=1 and k=2
    abl_unified_rows = []
    for slug in config_slugs:
        for img_id in image_ids:
            k1_stop = abl_map[slug].get(img_id, max_level)
            k2_stop = abl_k2_map[slug].get(img_id, max_level)
            abl_unified_rows.append(
                {
                    "config_slug": slug,
                    "image_id": img_id,
                    "stop_k1": k1_stop,
                    "stop_k2": k2_stop,
                }
            )

    save_csv(
        abl_unified_rows,
        metrics_root / "llm_ablation_k2_stop.csv",
        ["config_slug", "image_id", "stop_k1", "stop_k2"],
    )

    # Identify the base configs for 4b and 9b (h5, rel_on, conf_on)
    base_4b_slug = "qwen3.5-4b_h5_rel_on_conf_on"
    base_9b_slug = "qwen3.5-9b_h5_rel_on_conf_on"

    print(
        f"  LLM 4b base k=1 mean stop: {np.mean([abl_map[base_4b_slug][i] for i in image_ids]):.1f}"
    )
    print(
        f"  LLM 4b base k=2 mean stop: {np.mean([abl_k2_map[base_4b_slug][i] for i in image_ids]):.1f}"
    )
    print(
        f"  LLM 9b base k=1 mean stop: {np.mean([abl_map[base_9b_slug][i] for i in image_ids]):.1f}"
    )
    print(
        f"  LLM 9b base k=2 mean stop: {np.mean([abl_k2_map[base_9b_slug][i] for i in image_ids]):.1f}"
    )

    print()
    print("=" * 60)
    print("STEP 2: SG-Score weight ablation")
    print("=" * 60)

    weight_configs = {
        "balanced": (0.5, 0.5),
        "rel_heavy": (0.4, 0.6),
    }

    weight_abl_rows = []
    for wname, (nw, rw) in weight_configs.items():
        oracle_stops_w = []
        for img_id in image_ids:
            # Find first level where weighted score >= main_oracle_threshold
            stop = max_level
            for lvl in range(1, max_level + 1):
                key = (img_id, lvl)
                if key in metrics_by_img_lvl:
                    nr = float(metrics_by_img_lvl[key]["node_recall"])
                    tr = float(metrics_by_img_lvl[key]["triplet_recall"])
                    score = nw * nr + rw * tr
                    if score >= main_oracle_t:
                        stop = lvl
                        break
            oracle_stops_w.append(stop)

        saturated = sum(1 for s in oracle_stops_w if s < max_level)
        weight_abl_rows.append(
            {
                "weighting": wname,
                "node_weight": nw,
                "relation_weight": rw,
                "mean_oracle_stop": round(np.mean(oracle_stops_w), 2),
                "median_oracle_stop": round(median(oracle_stops_w), 1),
                "images_saturated": saturated,
                "images_total": len(image_ids),
                "saturation_rate": round(saturated / len(image_ids), 2),
            }
        )
        print(
            f"  {wname}: mean_oracle_stop={np.mean(oracle_stops_w):.2f}, "
            f"saturated={saturated}/{len(image_ids)}"
        )

    save_csv(
        weight_abl_rows,
        metrics_root / "sgscore_weight_ablation.csv",
        list(weight_abl_rows[0].keys()),
    )

    print()
    print("=" * 60)
    print("STEP 3: Hybrid stop (confidence + LLM fusion)")
    print("=" * 60)

    hybrid_cfg = cfg["ablation"]["hybrid"]
    conf_w = hybrid_cfg["conf_weight"]
    llm_w = hybrid_cfg["llm_weight"]
    hybrid_threshold = hybrid_cfg["threshold"]

    hybrid_rows = []

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

                # Build fused series
                fused: dict[int, float] = {}
                for lvl in range(start_level, max_level + 1):
                    key = (img_id, lvl)
                    if key in metrics_by_img_lvl and lvl in sat_scores:
                        obj_conf = float(
                            metrics_by_img_lvl[key]["mean_object_confidence"]
                        )
                        sat_sc = sat_scores[lvl]
                        fused[lvl] = conf_w * obj_conf + llm_w * sat_sc
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

    # Build hybrid lookup: variant -> {image_id: stop}
    hybrid_map: dict[str, dict[int, int]] = {}
    for r in hybrid_rows:
        hybrid_map.setdefault(r["variant"], {})[r["image_id"]] = r["stop_level"]

    print(f"  Hybrid variants saved: {sorted(hybrid_map.keys())}")
    for v, stops in sorted(hybrid_map.items()):
        mean_s = np.mean([stops[i] for i in image_ids])
        print(f"    {v}: mean_stop={mean_s:.1f}")

    print()
    print("=" * 60)
    print("STEP 4: Unified method comparison vs Oracle(0.5)")
    print("=" * 60)

    # Collect all methods: name -> {image_id: stop_level}
    methods: dict[str, dict[int, int]] = {}

    # Baseline
    methods["baseline"] = baseline_map

    # Legacy LLM (from llm_stop.csv = 4b h5 rel+conf k=1)
    # Note: llm_stop.csv is the legacy run; ablation has the same config more precisely.
    # Use ablation for consistency.

    # LLM-SGG from ablation: 4b and 9b base, k=1 and k=2
    methods["llm_sgg_4b_h5_k1"] = abl_map[base_4b_slug]
    methods["llm_sgg_4b_h5_k2"] = abl_k2_map[base_4b_slug]
    methods["llm_sgg_9b_h5_k1"] = abl_map[base_9b_slug]
    methods["llm_sgg_9b_h5_k2"] = abl_k2_map[base_9b_slug]

    # Ablation variants (9b, k=1 only)
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

    # Compute comparison stats
    oracle_stops = [oracle_map[i] for i in image_ids]
    comparison_rows = []

    for mname, stop_map in methods.items():
        method_stops = [stop_map.get(i, max_level) for i in image_ids]
        errors = [m - o for m, o in zip(method_stops, oracle_stops, strict=True)]
        abs_errors = [abs(e) for e in errors]

        mean_stop = np.mean(method_stops)
        mean_ae = np.mean(abs_errors)
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
                "mean_stop": round(float(mean_stop), 2),
                "mean_oracle_stop": round(float(np.mean(oracle_stops)), 2),
                "mean_abs_error": round(float(mean_ae), 2),
                "median_abs_error": round(float(median_ae), 1),
                "early_stop_rate": round(float(early), 2),
                "late_stop_rate": round(float(late), 2),
                "pearson_r": round(float(corr), 3) if not math.isnan(corr) else "NaN",
            }
        )

    # Sort by MAE ascending
    comparison_rows.sort(key=lambda r: float(r["mean_abs_error"]))

    save_csv(
        comparison_rows,
        metrics_root / "method_comparison.csv",
        list(comparison_rows[0].keys()),
    )

    print(
        f"\n  {'Method':<28s} {'MeanStop':>8s} {'MAE':>6s} {'MedAE':>6s} "
        f"{'Early':>6s} {'Late':>6s} {'Corr':>7s}"
    )
    print("  " + "-" * 75)
    for r in comparison_rows:
        print(
            f"  {r['method']:<28s} {r['mean_stop']:>8} {r['mean_abs_error']:>6} "
            f"{r['median_abs_error']:>6} {r['early_stop_rate']:>6} "
            f"{r['late_stop_rate']:>6} {str(r['pearson_r']):>7}"
        )

    print()
    print("=" * 60)
    print("STEP 5: Figures")
    print("=" * 60)

    # --- Figure 1: Method comparison MAE (horizontal bar) ---
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

    # Annotate bars
    for _idx, (bar, er, lr) in enumerate(
        zip(bars, early_rates, late_rates, strict=True)
    ):
        w = bar.get_width()
        label = f"E{er:.0%}/L{lr:.0%}"
        ax.text(
            w + 0.15, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=7
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_names, fontsize=8)
    ax.set_xlabel("Mean Absolute Error (levels)")
    ax.set_title("Method Comparison: MAE vs Oracle(0.5)")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#e8746a", edgecolor="black", label="Tends early"),
        Patch(facecolor="#6a9fe8", edgecolor="black", label="Tends late"),
        Patch(facecolor="#999999", edgecolor="black", label="Balanced"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(figures_root / "method_comparison_mae.png", dpi=150)
    plt.close(fig)
    print("  Saved method_comparison_mae.png")

    # --- Figure 2: Stop distribution box/strip plot ---
    # Select a subset of representative methods for readability
    plot_methods_order = [
        "baseline",
        "llm_sgg_4b_h5_k1",
        "llm_sgg_4b_h5_k2",
        "llm_sgg_9b_h5_k1",
        "llm_sgg_9b_h5_k2",
        "vlm_4b_single_k1",
        "vlm_9b_single_k1",
        "vlm_4b_sequence_k1",
        "vlm_9b_sequence_k1",
        "hybrid_4b_k1",
        "hybrid_4b_k2",
        "hybrid_9b_k1",
        "hybrid_9b_k2",
    ]
    # Filter to methods that exist
    plot_methods = [m for m in plot_methods_order if m in methods]

    fig, ax = plt.subplots(figsize=(14, 7))
    box_data = []
    box_labels = []
    for mname in plot_methods:
        stop_map = methods[mname]
        stops = [stop_map.get(i, max_level) for i in image_ids]
        box_data.append(stops)
        box_labels.append(mname)

    # Add oracle for reference
    box_data.append(oracle_stops)
    box_labels.append("oracle(0.5)")

    bp = ax.boxplot(box_data, vert=True, patch_artist=True, widths=0.6)

    # Color boxes
    n_methods = len(box_labels)
    cmap = plt.cm.tab20
    for i, patch in enumerate(bp["boxes"]):
        if i == n_methods - 1:  # oracle
            patch.set_facecolor("#90EE90")
            patch.set_alpha(0.8)
        else:
            patch.set_facecolor(cmap(i / max(n_methods - 1, 1)))
            patch.set_alpha(0.6)

    # Strip (jittered scatter) overlay
    rng = np.random.default_rng(42)
    for i, data in enumerate(box_data):
        jitter = rng.uniform(-0.15, 0.15, len(data))
        ax.scatter(
            np.full(len(data), i + 1) + jitter,
            data,
            alpha=0.5,
            s=20,
            zorder=5,
            color="black",
        )

    # Oracle mean reference line
    oracle_mean = np.mean(oracle_stops)
    ax.axhline(
        y=oracle_mean,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Oracle mean={oracle_mean:.1f}",
    )

    ax.set_xticks(range(1, n_methods + 1))
    ax.set_xticklabels(box_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Stop Level")
    ax.set_title("Stop Level Distribution by Method")
    ax.set_ylim(0, max_level + 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures_root / "stop_distribution_all.png", dpi=150)
    plt.close(fig)
    print("  Saved stop_distribution_all.png")

    # --- Figure 3: Ablation axes (4 panels) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    oracle_mean_val = float(np.mean(oracle_stops))
    baseline_mean_val = float(
        np.mean([baseline_map.get(i, max_level) for i in image_ids])
    )

    def add_reference_lines(ax_panel: plt.Axes) -> None:
        ax_panel.axhline(
            y=oracle_mean_val,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Oracle mean={oracle_mean_val:.1f}",
        )
        ax_panel.axhline(
            y=baseline_mean_val,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label=f"Baseline mean={baseline_mean_val:.1f}",
        )

    # Panel A: Model size (4b vs 9b at base config)
    ax_a = axes[0, 0]
    model_labels = ["4b", "9b"]
    model_means = [
        np.mean([abl_map[base_4b_slug][i] for i in image_ids]),
        np.mean([abl_map[base_9b_slug][i] for i in image_ids]),
    ]
    bars_a = ax_a.bar(
        model_labels, model_means, color=["#4C72B0", "#DD8452"], width=0.5
    )
    for bar, val in zip(bars_a, model_means, strict=True):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.1f}",
            ha="center",
            fontsize=9,
        )
    add_reference_lines(ax_a)
    ax_a.set_title("(A) Model Size", fontsize=10)
    ax_a.set_ylabel("Mean LLM Stop Level")
    ax_a.set_ylim(0, max_level + 1)
    ax_a.legend(fontsize=7)
    ax_a.grid(True, alpha=0.3, axis="y")

    # Panel B: History length (1, 3, 5 at 9b)
    ax_b = axes[0, 1]
    hist_slugs = {
        1: "qwen3.5-9b_h1_rel_on_conf_on",
        3: "qwen3.5-9b_h3_rel_on_conf_on",
        5: base_9b_slug,
    }
    hist_labels = ["h=1", "h=3", "h=5"]
    hist_means = [
        float(np.mean([abl_map[hist_slugs[h]][i] for i in image_ids]))
        for h in [1, 3, 5]
    ]
    bars_b = ax_b.bar(
        hist_labels, hist_means, color=["#55A868", "#C44E52", "#8172B2"], width=0.5
    )
    for bar, val in zip(bars_b, hist_means, strict=True):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.1f}",
            ha="center",
            fontsize=9,
        )
    add_reference_lines(ax_b)
    ax_b.set_title("(B) History Length (9b)", fontsize=10)
    ax_b.set_ylabel("Mean LLM Stop Level")
    ax_b.set_ylim(0, max_level + 1)
    ax_b.legend(fontsize=7)
    ax_b.grid(True, alpha=0.3, axis="y")

    # Panel C: Relation on/off (9b, h5)
    ax_c = axes[1, 0]
    rel_slugs = {
        "on": base_9b_slug,
        "off": "qwen3.5-9b_h5_rel_off_conf_on",
    }
    rel_labels = ["rel=on", "rel=off"]
    rel_means = [
        float(np.mean([abl_map[rel_slugs["on"]][i] for i in image_ids])),
        float(np.mean([abl_map[rel_slugs["off"]][i] for i in image_ids])),
    ]
    bars_c = ax_c.bar(rel_labels, rel_means, color=["#4C72B0", "#CCB974"], width=0.5)
    for bar, val in zip(bars_c, rel_means, strict=True):
        ax_c.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.1f}",
            ha="center",
            fontsize=9,
        )
    add_reference_lines(ax_c)
    ax_c.set_title("(C) Relation Input (9b, h=5)", fontsize=10)
    ax_c.set_ylabel("Mean LLM Stop Level")
    ax_c.set_ylim(0, max_level + 1)
    ax_c.legend(fontsize=7)
    ax_c.grid(True, alpha=0.3, axis="y")

    # Panel D: Confidence on/off (9b, h5)
    ax_d = axes[1, 1]
    conf_slugs = {
        "on": base_9b_slug,
        "off": "qwen3.5-9b_h5_rel_on_conf_off",
    }
    conf_labels = ["conf=on", "conf=off"]
    conf_means = [
        float(np.mean([abl_map[conf_slugs["on"]][i] for i in image_ids])),
        float(np.mean([abl_map[conf_slugs["off"]][i] for i in image_ids])),
    ]
    bars_d = ax_d.bar(conf_labels, conf_means, color=["#4C72B0", "#64B5CD"], width=0.5)
    for bar, val in zip(bars_d, conf_means, strict=True):
        ax_d.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.1f}",
            ha="center",
            fontsize=9,
        )
    add_reference_lines(ax_d)
    ax_d.set_title("(D) Confidence Input (9b, h=5)", fontsize=10)
    ax_d.set_ylabel("Mean LLM Stop Level")
    ax_d.set_ylim(0, max_level + 1)
    ax_d.legend(fontsize=7)
    ax_d.grid(True, alpha=0.3, axis="y")

    fig.suptitle("LLM-SGG Ablation Axes (k=1)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(figures_root / "ablation_axes.png", dpi=150)
    plt.close(fig)
    print("  Saved ablation_axes.png")

    # ---------------------------------------------------------------
    # Write evaluation report
    # ---------------------------------------------------------------
    print()
    print("=" * 60)
    print("Writing evaluation report")
    print("=" * 60)

    report_path = REPO_ROOT / ".claude" / "docs" / "research" / "m1-evaluation.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Milestone-1 SGG Saturation Evaluation Report")
    lines.append("")
    lines.append("Generated by `scripts/13_aggregate_evaluation.py`")
    lines.append("")

    # Table 1: Method comparison
    lines.append("## 1. Method Comparison (sorted by MAE vs Oracle 0.5)")
    lines.append("")
    lines.append(
        "| Method | Mean Stop | MAE | Median AE | Early% | Late% | Pearson r |"
    )
    lines.append(
        "|--------|-----------|-----|-----------|--------|-------|-----------|"
    )
    for r in comparison_rows:
        lines.append(
            f"| {r['method']} | {r['mean_stop']} | {r['mean_abs_error']} | "
            f"{r['median_abs_error']} | {r['early_stop_rate']} | "
            f"{r['late_stop_rate']} | {r['pearson_r']} |"
        )
    lines.append("")

    # Table 2: Ablation axis effects
    lines.append("## 2. Ablation Axis Effects (LLM-SGG, k=1)")
    lines.append("")
    lines.append("| Axis | Setting | Mean Stop | MAE vs Oracle |")
    lines.append("|------|---------|-----------|---------------|")

    # Compute MAE for each ablation variant
    def mae_for(stop_map: dict[int, int]) -> float:
        errors = [abs(stop_map.get(i, max_level) - oracle_map[i]) for i in image_ids]
        return round(float(np.mean(errors)), 2)

    lines.append(
        f"| Model | 4b (base) | {np.mean([abl_map[base_4b_slug][i] for i in image_ids]):.1f} | {mae_for(abl_map[base_4b_slug])} |"
    )
    lines.append(
        f"| Model | 9b (base) | {np.mean([abl_map[base_9b_slug][i] for i in image_ids]):.1f} | {mae_for(abl_map[base_9b_slug])} |"
    )
    for h in [1, 3, 5]:
        slug = hist_slugs[h]
        lines.append(
            f"| History | h={h} | {np.mean([abl_map[slug][i] for i in image_ids]):.1f} | {mae_for(abl_map[slug])} |"
        )
    lines.append(
        f"| Relation | on | {rel_means[0]:.1f} | {mae_for(abl_map[rel_slugs['on']])} |"
    )
    lines.append(
        f"| Relation | off | {rel_means[1]:.1f} | {mae_for(abl_map[rel_slugs['off']])} |"
    )
    lines.append(
        f"| Confidence | on | {conf_means[0]:.1f} | {mae_for(abl_map[conf_slugs['on']])} |"
    )
    lines.append(
        f"| Confidence | off | {conf_means[1]:.1f} | {mae_for(abl_map[conf_slugs['off']])} |"
    )
    lines.append("")

    # Table 3: SG-Score weight ablation
    lines.append("## 3. SG-Score Weight Ablation")
    lines.append("")
    lines.append("| Weighting | Node W | Rel W | Mean Oracle Stop | Saturated | Rate |")
    lines.append("|-----------|--------|-------|-----------------|-----------|------|")
    for r in weight_abl_rows:
        lines.append(
            f"| {r['weighting']} | {r['node_weight']} | {r['relation_weight']} | "
            f"{r['mean_oracle_stop']} | {r['images_saturated']}/{r['images_total']} | "
            f"{r['saturation_rate']} |"
        )
    lines.append("")

    # Table 4: Hybrid vs components
    lines.append("## 4. Hybrid Stop vs Components")
    lines.append("")
    lines.append("| Method | Mean Stop | MAE |")
    lines.append("|--------|-----------|-----|")
    hybrid_component_methods = [
        "baseline",
        "llm_sgg_4b_h5_k1",
        "llm_sgg_9b_h5_k1",
        "hybrid_4b_k1",
        "hybrid_4b_k2",
        "hybrid_9b_k1",
        "hybrid_9b_k2",
    ]
    for mname in hybrid_component_methods:
        r = next((x for x in comparison_rows if x["method"] == mname), None)
        if r:
            lines.append(
                f"| {r['method']} | {r['mean_stop']} | {r['mean_abs_error']} |"
            )
    lines.append("")

    # Table 5: Consecutive-k effect
    lines.append("## 5. Consecutive-k Effect")
    lines.append("")
    lines.append("| Method | k=1 Mean Stop | k=1 MAE | k=2 Mean Stop | k=2 MAE |")
    lines.append("|--------|---------------|---------|---------------|---------|")
    for label, slug in [("LLM 4b", base_4b_slug), ("LLM 9b", base_9b_slug)]:
        k1_stops = [abl_map[slug][i] for i in image_ids]
        k2_stops = [abl_k2_map[slug][i] for i in image_ids]
        k1_mae = np.mean(
            [abs(s - oracle_map[i]) for s, i in zip(k1_stops, image_ids, strict=True)]
        )
        k2_mae = np.mean(
            [abs(s - oracle_map[i]) for s, i in zip(k2_stops, image_ids, strict=True)]
        )
        lines.append(
            f"| {label} | {np.mean(k1_stops):.1f} | {k1_mae:.2f} | "
            f"{np.mean(k2_stops):.1f} | {k2_mae:.2f} |"
        )
    lines.append("")

    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text)
    print(f"  Saved {report_path}")

    print()
    print("=" * 60)
    print("DONE. Files created:")
    print("=" * 60)
    print(f"  {metrics_root / 'llm_ablation_k2_stop.csv'}")
    print(f"  {metrics_root / 'sgscore_weight_ablation.csv'}")
    print(f"  {metrics_root / 'hybrid_stop.csv'}")
    print(f"  {metrics_root / 'method_comparison.csv'}")
    print(f"  {figures_root / 'method_comparison_mae.png'}")
    print(f"  {figures_root / 'stop_distribution_all.png'}")
    print(f"  {figures_root / 'ablation_axes.png'}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
