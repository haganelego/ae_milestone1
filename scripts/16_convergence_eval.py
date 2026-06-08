"""Evaluate convergence-based stopping criteria (Plan B-1).

Computes deployable convergence stops for multiple signals and compares
them against RelOracle(0.8) and GT Oracle(0.5) references.

Run:
  cd /home/kanaoka/usr/github/milestone_1 && \
  uv run --project egtr python scripts/16_convergence_eval.py
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
from milestone1.convergence import convergence_stop  # noqa: E402


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


def median_val(vals: list[float]) -> float:
    """Simple median."""
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def load_llm_saturation_series(
    ablation_dir: Path, config_slug: str, image_id: int, max_level: int
) -> list[float]:
    """Load saturation_score series for an image from cached LLM ablation data.

    Returns a list of length max_level (indexed 0..max_level-1 for levels 1..max_level).
    Missing levels are NaN.
    """
    series = [float("nan")] * max_level
    img_dir = ablation_dir / config_slug / str(image_id)
    if not img_dir.exists():
        return series
    for lvl in range(1, max_level + 1):
        fpath = img_dir / f"level_{lvl:02d}.json"
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            score = data.get("saturation_score")
            if score is not None and not (
                isinstance(score, float) and math.isnan(score)
            ):
                series[lvl - 1] = float(score)
    return series


def compute_method_stats(
    method_stops: list[int],
    oracle_stops: list[int],
    max_level: int,
) -> dict:
    """Compute comparison statistics for a method vs an oracle reference."""
    errors = [m - o for m, o in zip(method_stops, oracle_stops, strict=True)]
    abs_errors = [abs(e) for e in errors]
    n = len(errors)

    mean_stop = float(np.mean(method_stops))
    frac_stop = sum(1 for s in method_stops if s < max_level) / n
    mean_ae = float(np.mean(abs_errors))
    med_ae = median_val([float(e) for e in abs_errors])
    early = sum(1 for e in errors if e < 0) / n
    late = sum(1 for e in errors if e > 0) / n
    corr = pearson_r(
        [float(s) for s in method_stops],
        [float(o) for o in oracle_stops],
    )

    return {
        "mean_stop": round(mean_stop, 2),
        "frac_stop": round(frac_stop, 3),
        "mean_abs_error": round(mean_ae, 2),
        "median_abs_error": round(med_ae, 1),
        "early_rate": round(early, 3),
        "late_rate": round(late, 3),
        "pearson_r": round(corr, 3) if not math.isnan(corr) else "NaN",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = load_config()
    max_level = cfg["resolution_levels"]["num_levels"]  # 20

    # Convergence parameters (from config, with defaults)
    conv_cfg = cfg.get("convergence", {})
    epsilon = conv_cfg.get("epsilon", 0.02)
    k = conv_cfg.get("k", 3)
    min_level = conv_cfg.get("min_level", 5)

    metrics_root = resolve(cfg["paths"]["metrics"])
    figures_root = resolve(cfg["paths"]["figures"])
    figures_root.mkdir(parents=True, exist_ok=True)

    selected_path = resolve(cfg["paths"]["selected_images"])
    with open(selected_path) as f:
        image_ids: list[int] = json.load(f)

    n_images = len(image_ids)
    print(
        f"Convergence eval: {n_images} images, epsilon={epsilon}, k={k}, min_level={min_level}"
    )

    # ===================================================================
    # Load reference oracles
    # ===================================================================
    print("\n--- Loading reference oracles ---")

    # RelOracle(0.8) - primary reference
    rel_oracle_rows = load_csv_dict(metrics_root / "relative_oracle_stop.csv")
    rel_oracle_map = {
        int(r["image_id"]): int(r["rel_oracle_0.8"]) for r in rel_oracle_rows
    }

    # GT Oracle(0.5) - secondary reference
    gt_oracle_rows = load_csv_dict(metrics_root / "oracle_stop.csv")
    gt_oracle_map = {
        int(r["image_id"]): int(float(r["oracle_0.5"])) for r in gt_oracle_rows
    }

    rel_oracle_stops = [rel_oracle_map[i] for i in image_ids]
    gt_oracle_stops = [gt_oracle_map[i] for i in image_ids]

    # ===================================================================
    # Load signal data
    # ===================================================================
    print("\n--- Loading signal data ---")

    # 1. mean_object_confidence from all_metrics.csv
    all_metrics_rows = load_csv_dict(metrics_root / "all_metrics.csv")
    # Build lookup: {(image_id, level): mean_object_confidence}
    conf_map: dict[tuple[int, int], float] = {}
    for r in all_metrics_rows:
        iid = int(r["image_id"])
        lvl = int(r["level"])
        val = float(r["mean_object_confidence"])
        conf_map[(iid, lvl)] = val

    # 2. LLM saturation scores from cached ablation data
    ablation_dir = REPO_ROOT / "outputs" / "llm_ablation"
    slug_4b = "qwen3.5-4b_h5_rel_on_conf_on"
    slug_9b = "qwen3.5-9b_h5_rel_on_conf_on"

    # ===================================================================
    # Compute convergence stops for each signal
    # ===================================================================
    print("\n--- Computing convergence stops ---")

    # Storage: method_name -> {image_id: stop_level}
    conv_methods: dict[str, dict[int, int]] = {
        "conf_convergence": {},
        "llm4b_convergence": {},
        "llm9b_convergence": {},
        "hybrid_convergence": {},
    }

    for img_id in image_ids:
        # Signal 1: mean_object_confidence
        conf_series = [
            conf_map.get((img_id, lvl), float("nan")) for lvl in range(1, max_level + 1)
        ]
        conv_methods["conf_convergence"][img_id] = convergence_stop(
            conf_series, epsilon, k, min_level, max_level
        )

        # Signal 2: LLM 4b saturation_score
        llm4b_series = load_llm_saturation_series(
            ablation_dir, slug_4b, img_id, max_level
        )
        conv_methods["llm4b_convergence"][img_id] = convergence_stop(
            llm4b_series, epsilon, k, min_level, max_level
        )

        # Signal 3: LLM 9b saturation_score
        llm9b_series = load_llm_saturation_series(
            ablation_dir, slug_9b, img_id, max_level
        )
        conv_methods["llm9b_convergence"][img_id] = convergence_stop(
            llm9b_series, epsilon, k, min_level, max_level
        )

        # Signal 4: Hybrid = 0.5*conf + 0.5*llm4b (aligned levels; NaN where LLM missing)
        hybrid_series: list[float] = []
        for lvl in range(1, max_level + 1):
            c = conf_series[lvl - 1]
            l4 = llm4b_series[lvl - 1]
            if math.isnan(c) or math.isnan(l4):
                hybrid_series.append(float("nan"))
            else:
                hybrid_series.append(0.5 * c + 0.5 * l4)
        conv_methods["hybrid_convergence"][img_id] = convergence_stop(
            hybrid_series, epsilon, k, min_level, max_level
        )

    # Print per-image summary
    for img_id in image_ids[:10]:
        stops_str = ", ".join(f"{m}={conv_methods[m][img_id]}" for m in conv_methods)
        print(f"  {img_id}: {stops_str}")
    if n_images > 10:
        print(f"  ... ({n_images - 10} more)")

    # ===================================================================
    # Load existing absolute methods for comparison
    # ===================================================================
    print("\n--- Loading existing absolute methods for comparison ---")

    abs_methods: dict[str, dict[int, int]] = {}

    # Baseline (absolute conf >= 0.4)
    baseline_path = metrics_root / "baseline_stop.csv"
    if baseline_path.exists():
        baseline_rows = load_csv_dict(baseline_path)
        abs_methods["baseline_abs_conf04"] = {
            int(r["image_id"]): int(float(r["baseline_0.4"])) for r in baseline_rows
        }
        print("  Loaded baseline_stop.csv")
    else:
        print("  SKIP: baseline_stop.csv not found")

    # LLM-SGG 4b/9b from ablation (absolute threshold stop)
    abl_stop_path = metrics_root / "llm_ablation_stop.csv"
    if abl_stop_path.exists():
        abl_rows = load_csv_dict(abl_stop_path)
        for r in abl_rows:
            slug = r["config_slug"]
            if slug == slug_4b:
                abs_methods.setdefault("llm_sgg_4b_abs", {})[int(r["image_id"])] = int(
                    r["stop_level"]
                )
            elif slug == slug_9b:
                abs_methods.setdefault("llm_sgg_9b_abs", {})[int(r["image_id"])] = int(
                    r["stop_level"]
                )
        print("  Loaded llm_ablation_stop.csv (4b and 9b)")
    else:
        print("  SKIP: llm_ablation_stop.csv not found")

    # Hybrid absolute stop
    hybrid_path = metrics_root / "hybrid_stop.csv"
    if hybrid_path.exists():
        hybrid_rows = load_csv_dict(hybrid_path)
        for r in hybrid_rows:
            variant = r["variant"]
            if variant in ("hybrid_4b_k1", "hybrid_9b_k1"):
                abs_methods.setdefault(f"{variant}_abs", {})[int(r["image_id"])] = int(
                    r["stop_level"]
                )
        print("  Loaded hybrid_stop.csv")
    else:
        print("  SKIP: hybrid_stop.csv not found")

    # ===================================================================
    # Build comparison table
    # ===================================================================
    print("\n--- Building comparison table ---")

    # Combine all methods
    all_methods: dict[str, dict[int, int]] = {}
    all_methods.update(conv_methods)
    all_methods.update(abs_methods)

    # Compute stats vs RelOracle(0.8) (primary) and GT Oracle(0.5) (secondary)
    comparison_rows: list[dict] = []

    for mname, stop_map in all_methods.items():
        method_stops = [stop_map.get(i, max_level) for i in image_ids]

        stats_rel = compute_method_stats(method_stops, rel_oracle_stops, max_level)
        stats_gt = compute_method_stats(method_stops, gt_oracle_stops, max_level)

        comparison_rows.append(
            {
                "method": mname,
                "mean_stop": stats_rel["mean_stop"],
                "frac_images_stop": stats_rel["frac_stop"],
                "MAE_vs_RelOracle08": stats_rel["mean_abs_error"],
                "MedAE_vs_RelOracle08": stats_rel["median_abs_error"],
                "early_vs_RelOracle08": stats_rel["early_rate"],
                "late_vs_RelOracle08": stats_rel["late_rate"],
                "pearson_r_vs_RelOracle08": stats_rel["pearson_r"],
                "MAE_vs_GTOracle05": stats_gt["mean_abs_error"],
                "MedAE_vs_GTOracle05": stats_gt["median_abs_error"],
                "early_vs_GTOracle05": stats_gt["early_rate"],
                "late_vs_GTOracle05": stats_gt["late_rate"],
                "pearson_r_vs_GTOracle05": stats_gt["pearson_r"],
            }
        )

    # Sort by MAE vs RelOracle ascending
    comparison_rows.sort(key=lambda r: float(r["MAE_vs_RelOracle08"]))

    save_csv(
        comparison_rows,
        metrics_root / "method_comparison_convergence_100.csv",
        list(comparison_rows[0].keys()),
    )

    # Print table
    print(
        f"\n  {'Method':<25s} {'MeanStop':>8s} {'Frac%':>6s} "
        f"{'MAE_Rel':>7s} {'MedAE':>6s} {'Early':>6s} {'Late':>6s} {'r_Rel':>6s} "
        f"{'MAE_GT':>7s} {'r_GT':>6s}"
    )
    print("  " + "-" * 100)
    for r in comparison_rows:
        print(
            f"  {r['method']:<25s} {r['mean_stop']:>8} {r['frac_images_stop']:>6} "
            f"{r['MAE_vs_RelOracle08']:>7} {r['MedAE_vs_RelOracle08']:>6} "
            f"{r['early_vs_RelOracle08']:>6} {r['late_vs_RelOracle08']:>6} "
            f"{str(r['pearson_r_vs_RelOracle08']):>6} "
            f"{r['MAE_vs_GTOracle05']:>7} {str(r['pearson_r_vs_GTOracle05']):>6}"
        )

    # ===================================================================
    # Figure: Convergence vs Absolute MAE comparison
    # ===================================================================
    print("\n--- Generating figure ---")

    fig, ax = plt.subplots(figsize=(12, 7))

    method_names = [r["method"] for r in reversed(comparison_rows)]
    maes_rel = [float(r["MAE_vs_RelOracle08"]) for r in reversed(comparison_rows)]

    # Color: convergence methods in green, absolute in orange
    colors = []
    for mn in method_names:
        if "convergence" in mn:
            colors.append("#55A868")
        else:
            colors.append("#DD8452")

    y_pos = np.arange(len(method_names))
    bars = ax.barh(y_pos, maes_rel, color=colors, edgecolor="black", linewidth=0.5)

    # Annotate with frac_stop
    frac_stops = [r["frac_images_stop"] for r in reversed(comparison_rows)]
    for bar, frac in zip(bars, frac_stops, strict=True):
        w = bar.get_width()
        ax.text(
            w + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"stop={frac:.0%}",
            va="center",
            fontsize=7,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_names, fontsize=8)
    ax.set_xlabel("Mean Absolute Error vs RelOracle(0.8) [levels]")
    ax.set_title("Convergence vs Absolute Stop Methods: MAE Comparison (100 images)")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="#55A868",
            edgecolor="black",
            label="Convergence-based (deployable)",
        ),
        Patch(facecolor="#DD8452", edgecolor="black", label="Absolute threshold-based"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()

    fig_path = figures_root / "convergence_vs_absolute_mae_100.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_path}")

    # ===================================================================
    # Key findings summary
    # ===================================================================
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Q1: Does conf_convergence make ALL 100 images stop?
    conf_frac = sum(
        1 for i in image_ids if conv_methods["conf_convergence"][i] < max_level
    )
    baseline_frac = sum(
        1
        for i in image_ids
        if abs_methods.get("baseline_abs_conf04", {}).get(i, max_level) < max_level
    )
    print(
        f"\n  Q1: conf_convergence stops {conf_frac}/{n_images} images "
        f"(vs baseline_abs_conf04: {baseline_frac}/{n_images})"
    )

    # Q2: MAE comparison
    conf_row = next(r for r in comparison_rows if r["method"] == "conf_convergence")
    baseline_row = next(
        (r for r in comparison_rows if r["method"] == "baseline_abs_conf04"), None
    )
    print(
        f"\n  Q2: conf_convergence MAE vs RelOracle = {conf_row['MAE_vs_RelOracle08']}"
    )
    if baseline_row:
        print(
            f"      baseline_abs_conf04 MAE vs RelOracle = {baseline_row['MAE_vs_RelOracle08']}"
        )
        if float(conf_row["MAE_vs_RelOracle08"]) < float(
            baseline_row["MAE_vs_RelOracle08"]
        ):
            print("      -> conf_convergence IS closer to RelOracle than baseline")
        else:
            print("      -> conf_convergence is NOT closer (baseline is better)")

    # Q4: Best convergence method
    conv_only = [r for r in comparison_rows if "convergence" in r["method"]]
    if conv_only:
        best_conv = conv_only[0]
        print(
            f"\n  Q4: Best convergence method: {best_conv['method']} "
            f"(MAE={best_conv['MAE_vs_RelOracle08']}, r={best_conv['pearson_r_vs_RelOracle08']})"
        )

    print(
        f"\n  Overall best: {comparison_rows[0]['method']} "
        f"(MAE={comparison_rows[0]['MAE_vs_RelOracle08']})"
    )

    # ===================================================================
    # Done
    # ===================================================================
    print("\n" + "=" * 60)
    print("DONE. Files created:")
    print("=" * 60)
    print(f"  {metrics_root / 'method_comparison_convergence_100.csv'}")
    print(f"  {fig_path}")


if __name__ == "__main__":
    main()
