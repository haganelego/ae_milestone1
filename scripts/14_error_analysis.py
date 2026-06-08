"""Error and saturation analysis for the SGG-saturation PoC.

Analyses:
1. What was missing at each method's stop level
2. Per-object-category recall vs level (hardest categories)
3. Per-predicate recall vs level (hardest predicates)
4. Dataset-weakness characterization (image difficulty features)
5. Node-vs-relation saturation gap

Usage:
    cd /home/kanaoka/usr/github/milestone_1 && uv run --project egtr python scripts/14_error_analysis.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.analysis import (
    compute_category_recall_by_level,
    compute_image_features,
    compute_missing_at_stop,
    compute_predicate_recall_by_level,
    compute_saturation_gap,
    identify_hardest,
    load_gt,
    load_pred,
    load_stop_levels,
)
from milestone1.config import load_config, resolve


def main() -> None:
    cfg = load_config()
    num_levels = cfg["resolution_levels"]["num_levels"]

    # Paths
    selected_path = resolve(cfg["paths"]["selected_images"])
    with open(selected_path) as f:
        image_ids: list[int] = json.load(f)

    gt_root = resolve(cfg["paths"]["gt_scene_graphs"])
    pred_root = resolve(cfg["paths"]["egtr_predictions"])
    metrics_root = resolve(cfg["paths"]["metrics"])
    vg_images_dir = Path(cfg["paths"]["vg_root"]) / "images"
    analysis_dir = resolve("outputs/analysis")
    figures_dir = resolve(cfg["paths"]["figures"])
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_metrics_df = pd.read_csv(metrics_root / "all_metrics.csv")
    stop_levels = load_stop_levels(metrics_root)

    print("=" * 60)
    print("Analysis 1: What was missing at the stop level")
    print("=" * 60)

    missing_rows: list[dict] = []
    for method, stops in stop_levels.items():
        for img_id in image_ids:
            stop_lv = stops.get(img_id, 20)
            # Clamp to valid range
            stop_lv = max(1, min(stop_lv, num_levels))

            gt = load_gt(gt_root, img_id)
            pred = load_pred(pred_root, img_id, stop_lv)
            result = compute_missing_at_stop(gt, pred)
            result["image_id"] = img_id
            result["method"] = method
            result["stop_level"] = stop_lv
            missing_rows.append(result)

    missing_df = pd.DataFrame(missing_rows)
    col_order = [
        "image_id",
        "method",
        "stop_level",
        "gt_node_count",
        "pred_node_count",
        "missing_node_count",
        "extra_node_count",
        "missing_nodes",
        "extra_nodes",
        "gt_triplet_count",
        "pred_triplet_count",
        "missing_triplet_count",
        "extra_triplet_count",
        "missing_triplets",
        "extra_triplets",
    ]
    missing_df = missing_df[col_order]
    missing_df.to_csv(analysis_dir / "missing_at_stop.csv", index=False)

    # Summarize by method
    summary = (
        missing_df.groupby("method")
        .agg(
            mean_stop_level=("stop_level", "mean"),
            mean_missing_nodes=("missing_node_count", "mean"),
            mean_missing_triplets=("missing_triplet_count", "mean"),
            mean_extra_nodes=("extra_node_count", "mean"),
            mean_extra_triplets=("extra_triplet_count", "mean"),
        )
        .round(2)
    )
    print(summary)
    print()

    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Analysis 2: Per-object-category recall vs level")
    print("=" * 60)

    cat_recall_df = compute_category_recall_by_level(
        image_ids, gt_root, pred_root, num_levels
    )
    cat_recall_df.to_csv(
        analysis_dir / "category_recall_by_level.csv", index=False
    )

    hardest_cats = identify_hardest(cat_recall_df, "category", num_levels, top_k=10)
    print(f"Hardest object categories (by L20 recall): {hardest_cats}")

    # Plot hardest categories
    fig, ax = plt.subplots(figsize=(12, 6))
    for cat in hardest_cats:
        cat_data = cat_recall_df[cat_recall_df["category"] == cat]
        ax.plot(cat_data["level"], cat_data["recall"], marker="o", markersize=3, label=cat)
    ax.set_xlabel("Resolution Level")
    ax.set_ylabel("Recall (fraction of images detecting category)")
    ax.set_title("Hardest Object Categories: Recall vs Resolution Level")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_xticks(range(1, num_levels + 1))
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "category_recall_hardest.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {figures_dir / 'category_recall_hardest.png'}")

    # Print recall table for hardest categories at key levels
    pivot = cat_recall_df.pivot(index="category", columns="level", values="recall")
    hardest_pivot = pivot.loc[
        [c for c in hardest_cats if c in pivot.index],
        [1, 5, 10, 15, 20],
    ].round(2)
    print(hardest_pivot)
    print()

    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Analysis 3: Per-predicate recall vs level")
    print("=" * 60)

    pred_recall_df = compute_predicate_recall_by_level(
        image_ids, gt_root, pred_root, num_levels
    )
    pred_recall_df.to_csv(
        analysis_dir / "predicate_recall_by_level.csv", index=False
    )

    hardest_preds = identify_hardest(
        pred_recall_df, "predicate", num_levels, top_k=10
    )
    print(f"Hardest predicates (by L20 recall): {hardest_preds}")

    # Plot hardest predicates
    fig, ax = plt.subplots(figsize=(12, 6))
    for pred_name in hardest_preds:
        p_data = pred_recall_df[pred_recall_df["predicate"] == pred_name]
        ax.plot(
            p_data["level"], p_data["recall"], marker="o", markersize=3, label=pred_name
        )
    ax.set_xlabel("Resolution Level")
    ax.set_ylabel("Recall (fraction of GT triplets detected)")
    ax.set_title("Hardest Predicates: Recall vs Resolution Level")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_xticks(range(1, num_levels + 1))
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "predicate_recall_hardest.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {figures_dir / 'predicate_recall_hardest.png'}")

    # Print recall table for hardest predicates at key levels
    pivot_p = pred_recall_df.pivot(
        index="predicate", columns="level", values="recall"
    )
    hardest_pivot_p = pivot_p.loc[
        [p for p in hardest_preds if p in pivot_p.index],
        [1, 5, 10, 15, 20],
    ].round(2)
    print(hardest_pivot_p)
    print()

    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Analysis 4: Image difficulty features")
    print("=" * 60)

    oracle_stops = stop_levels.get("oracle", {})
    feature_rows = []
    for img_id in image_ids:
        gt = load_gt(gt_root, img_id)
        oracle_lv = oracle_stops.get(img_id, 20)
        features = compute_image_features(
            img_id, gt, vg_images_dir, all_metrics_df, oracle_lv
        )
        feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)
    features_df = features_df.sort_values("final_sg_score")
    features_df.to_csv(analysis_dir / "image_features.csv", index=False)

    print("Image features (sorted by difficulty, hardest first):")
    print(
        features_df[
            [
                "image_id",
                "num_gt_objects",
                "num_gt_unique_triplets",
                "frac_small_objects",
                "final_sg_score",
                "final_node_recall",
                "final_triplet_recall",
                "oracle_stop_level",
            ]
        ].to_string(index=False)
    )
    print()

    # Correlations
    numeric_cols = [
        "num_gt_objects",
        "num_gt_object_categories",
        "num_gt_relationships",
        "num_gt_unique_triplets",
        "mean_obj_area_frac",
        "frac_small_objects",
    ]
    print("Correlations with final_sg_score:")
    for col in numeric_cols:
        corr = features_df["final_sg_score"].corr(features_df[col])
        print(f"  {col}: r={corr:.3f}")
    print()

    print("Correlations with oracle_stop_level:")
    for col in numeric_cols:
        corr = features_df["oracle_stop_level"].corr(features_df[col])
        print(f"  {col}: r={corr:.3f}")
    print()

    # Difficulty scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.scatter(
        features_df["frac_small_objects"],
        features_df["final_sg_score"],
        s=60,
        c="steelblue",
        edgecolors="black",
    )
    for _, row in features_df.iterrows():
        ax.annotate(
            str(int(row["image_id"])),
            (row["frac_small_objects"], row["final_sg_score"]),
            fontsize=6,
            ha="center",
            va="bottom",
        )
    ax.set_xlabel("Fraction of Small Objects (area < 1% image)")
    ax.set_ylabel("Final SGScore (L20)")
    ax.set_title("SGScore vs Small Object Fraction")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(
        features_df["num_gt_unique_triplets"],
        features_df["final_sg_score"],
        s=60,
        c="coral",
        edgecolors="black",
    )
    for _, row in features_df.iterrows():
        ax.annotate(
            str(int(row["image_id"])),
            (row["num_gt_unique_triplets"], row["final_sg_score"]),
            fontsize=6,
            ha="center",
            va="bottom",
        )
    ax.set_xlabel("Num GT Unique Triplets")
    ax.set_ylabel("Final SGScore (L20)")
    ax.set_title("SGScore vs GT Triplet Count")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.scatter(
        features_df["num_gt_object_categories"],
        features_df["final_sg_score"],
        s=60,
        c="seagreen",
        edgecolors="black",
    )
    for _, row in features_df.iterrows():
        ax.annotate(
            str(int(row["image_id"])),
            (row["num_gt_object_categories"], row["final_sg_score"]),
            fontsize=6,
            ha="center",
            va="bottom",
        )
    ax.set_xlabel("Num GT Object Categories")
    ax.set_ylabel("Final SGScore (L20)")
    ax.set_title("SGScore vs Object Category Count")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Image Difficulty vs Features", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "difficulty_vs_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {figures_dir / 'difficulty_vs_features.png'}")

    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Analysis 5: Node vs Relation saturation gap")
    print("=" * 60)

    gap_rows = []
    for img_id in image_ids:
        gap = compute_saturation_gap(img_id, all_metrics_df, threshold_frac=0.9)
        gap_rows.append(gap)

    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(analysis_dir / "saturation_gap.csv", index=False)

    print(gap_df.to_string(index=False))
    mean_gap = gap_df["gap"].mean()
    print(f"\nMean node-vs-relation saturation gap: {mean_gap:.1f} levels")
    print(
        f"  (Nodes reach 90% of max recall {abs(mean_gap):.1f} levels "
        f"{'before' if mean_gap > 0 else 'after'} triplets)"
    )

    print()
    print("=" * 60)
    print("All analyses complete.")
    print("=" * 60)
    print(f"  outputs/analysis/missing_at_stop.csv")
    print(f"  outputs/analysis/category_recall_by_level.csv")
    print(f"  outputs/analysis/predicate_recall_by_level.csv")
    print(f"  outputs/analysis/image_features.csv")
    print(f"  outputs/analysis/saturation_gap.csv")
    print(f"  outputs/figures/category_recall_hardest.png")
    print(f"  outputs/figures/predicate_recall_hardest.png")
    print(f"  outputs/figures/difficulty_vs_features.png")


if __name__ == "__main__":
    main()
