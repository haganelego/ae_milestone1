"""Plot scene graph metric curves and stop point distributions."""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import load_config, resolve


def load_csv_dict(path: Path) -> list[dict]:
    """Load CSV as list of dicts."""
    with open(path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    cfg = load_config()
    main_oracle_t = cfg["metrics"]["main_oracle_threshold"]
    main_baseline_t = cfg["baseline"]["mean_object_conf_threshold"]
    sat_t = cfg["llm"]["saturation_threshold"]

    selected_path = resolve(cfg["paths"]["selected_images"])
    with open(selected_path) as f:
        image_ids = json.load(f)

    metrics_root = resolve(cfg["paths"]["metrics"])
    figures_root = resolve(cfg["paths"]["figures"])
    figures_root.mkdir(parents=True, exist_ok=True)

    # Load stop points
    oracle_rows = load_csv_dict(metrics_root / "oracle_stop.csv")
    baseline_rows = load_csv_dict(metrics_root / "baseline_stop.csv")
    llm_rows = load_csv_dict(metrics_root / "llm_stop.csv")

    oracle_map = {int(r["image_id"]): r for r in oracle_rows}
    baseline_map = {int(r["image_id"]): r for r in baseline_rows}
    llm_map = {int(r["image_id"]): r for r in llm_rows}

    # Per-image plots
    for img_id in image_ids:
        img_csv = metrics_root / f"{img_id}.csv"
        rows = load_csv_dict(img_csv)

        levels = [int(r["level"]) for r in rows]
        node_recall = [float(r["node_recall"]) for r in rows]
        triplet_recall = [float(r["triplet_recall"]) for r in rows]
        sg_scores = [float(r["sg_score"]) for r in rows]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(levels, node_recall, "b-o", markersize=3, label="Node Recall")
        ax.plot(levels, triplet_recall, "r-s", markersize=3, label="Triplet Recall")
        ax.plot(levels, sg_scores, "g-^", markersize=3, label="SG Score")

        # Vertical lines for stop points
        oracle_stop = int(float(oracle_map[img_id][f"oracle_{main_oracle_t}"]))
        baseline_stop = int(float(baseline_map[img_id][f"baseline_{main_baseline_t}"]))
        llm_stop = int(llm_map[img_id]["llm_stop"])

        ax.axvline(
            x=oracle_stop, color="green", linestyle="--", alpha=0.7,
            label=f"Oracle({main_oracle_t})={oracle_stop}",
        )
        ax.axvline(
            x=baseline_stop, color="orange", linestyle="--", alpha=0.7,
            label=f"Baseline({main_baseline_t})={baseline_stop}",
        )
        ax.axvline(
            x=llm_stop, color="purple", linestyle="--", alpha=0.7,
            label=f"LLM({sat_t})={llm_stop}",
        )

        ax.set_xlabel("Resolution Level")
        ax.set_ylabel("Score")
        ax.set_title(f"Image {img_id}: SG Metrics vs Resolution Level")
        ax.set_xlim(0.5, 20.5)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(figures_root / f"{img_id}_curve.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {img_id}_curve.png")

    # Stop distribution histogram
    oracle_stops = [
        int(float(oracle_map[img_id][f"oracle_{main_oracle_t}"]))
        for img_id in image_ids
    ]
    baseline_stops = [
        int(float(baseline_map[img_id][f"baseline_{main_baseline_t}"]))
        for img_id in image_ids
    ]
    llm_stops = [int(llm_map[img_id]["llm_stop"]) for img_id in image_ids]

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.25
    x = np.arange(1, 21)

    oracle_hist = [oracle_stops.count(i) for i in range(1, 21)]
    baseline_hist = [baseline_stops.count(i) for i in range(1, 21)]
    llm_hist = [llm_stops.count(i) for i in range(1, 21)]

    ax.bar(x - width, oracle_hist, width, label=f"Oracle({main_oracle_t})", color="green", alpha=0.7)
    ax.bar(x, baseline_hist, width, label=f"Baseline({main_baseline_t})", color="orange", alpha=0.7)
    ax.bar(x + width, llm_hist, width, label=f"LLM({sat_t})", color="purple", alpha=0.7)

    ax.set_xlabel("Stop Level")
    ax.set_ylabel("Count")
    ax.set_title("Stop Level Distribution")
    ax.set_xticks(range(1, 21))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(figures_root / "stop_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved stop_distribution.png")

    # Stop error summary
    summary_rows = []
    for method_name, method_stops in [
        ("baseline", baseline_stops),
        ("llm", llm_stops),
    ]:
        errors = [m - o for m, o in zip(method_stops, oracle_stops)]
        abs_errors = [abs(e) for e in errors]
        mean_abs_error = sum(abs_errors) / len(abs_errors) if abs_errors else 0
        sorted_abs = sorted(abs_errors)
        n = len(sorted_abs)
        median_abs_error = (
            sorted_abs[n // 2]
            if n % 2 == 1
            else (sorted_abs[n // 2 - 1] + sorted_abs[n // 2]) / 2
        )
        early_rate = sum(1 for e in errors if e < 0) / len(errors) if errors else 0
        late_rate = sum(1 for e in errors if e > 0) / len(errors) if errors else 0
        mean_stop = sum(method_stops) / len(method_stops) if method_stops else 0
        mean_oracle = sum(oracle_stops) / len(oracle_stops) if oracle_stops else 0

        summary_rows.append({
            "method": method_name,
            "mean_abs_error": round(mean_abs_error, 2),
            "median_abs_error": round(median_abs_error, 2),
            "early_stop_rate": round(early_rate, 2),
            "late_stop_rate": round(late_rate, 2),
            "mean_stop_level": round(mean_stop, 2),
            "mean_oracle_level": round(mean_oracle, 2),
        })

    summary_path = metrics_root / "stop_error_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "mean_abs_error", "median_abs_error",
                "early_stop_rate", "late_stop_rate",
                "mean_stop_level", "mean_oracle_level",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nStop Error Summary:")
    for r in summary_rows:
        print(
            f"  {r['method']:10s}: MAE={r['mean_abs_error']:.2f}, "
            f"MedAE={r['median_abs_error']:.2f}, "
            f"early={r['early_stop_rate']:.0%}, "
            f"late={r['late_stop_rate']:.0%}, "
            f"mean_stop={r['mean_stop_level']:.1f} "
            f"(oracle={r['mean_oracle_level']:.1f})"
        )

    print(f"\nSaved stop error summary to {summary_path}")


if __name__ == "__main__":
    main()
