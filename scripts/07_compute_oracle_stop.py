"""Compute oracle stopping points based on sg_score thresholds and gain-based saturation."""

import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import load_config, resolve


def main() -> None:
    cfg = load_config()
    thresholds = cfg["metrics"]["oracle_thresholds"]
    gain_epsilon = cfg["metrics"]["gain_epsilon"]
    gain_k = cfg["metrics"]["gain_k"]

    selected_path = resolve(cfg["paths"]["selected_images"])
    with open(selected_path) as f:
        image_ids = json.load(f)

    metrics_root = resolve(cfg["paths"]["metrics"])

    fieldnames = ["image_id"] + [f"oracle_{t}" for t in thresholds] + ["gain_stop"]
    rows = []

    for img_id in image_ids:
        img_csv = metrics_root / f"{img_id}.csv"
        with open(img_csv) as f:
            reader = csv.DictReader(f)
            levels = list(reader)

        row: dict[str, object] = {"image_id": img_id}

        # Threshold-based oracle: min level with sg_score >= threshold
        for t in thresholds:
            stop_level = float("nan")
            for lv in levels:
                if float(lv["sg_score"]) >= t:
                    stop_level = int(lv["level"])
                    break
            if math.isnan(stop_level):
                stop_level = 20
            row[f"oracle_{t}"] = stop_level

        # Gain-based saturation: min level where gain < epsilon for k consecutive
        # REAL transitions only (level 2..N); no artificial zero at the start.
        sg_scores = [float(lv["sg_score"]) for lv in levels]
        # gains[i] is the transition FROM level i+1 TO level i+2 (0-indexed)
        gains = [sg_scores[i] - sg_scores[i - 1] for i in range(1, len(sg_scores))]

        gain_stop = float("nan")
        consecutive = 0
        for i, g in enumerate(gains):
            if abs(g) < gain_epsilon:
                consecutive += 1
                if consecutive >= gain_k:
                    # The run of k low-gain transitions ends at level (i+2).
                    # The first level of that run started at level (i+2 - k + 1).
                    gain_stop = i + 2 - (gain_k - 1)
                    break
            else:
                consecutive = 0
        if math.isnan(gain_stop):
            gain_stop = 20
        row["gain_stop"] = int(gain_stop)

        rows.append(row)
        main_t = cfg["metrics"]["main_oracle_threshold"]
        print(
            f"  {img_id}: oracle_{main_t}={row[f'oracle_{main_t}']}, "
            f"gain_stop={row['gain_stop']}"
        )

    # Save
    out_path = metrics_root / "oracle_stop.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved oracle stop points to {out_path}")


if __name__ == "__main__":
    main()
