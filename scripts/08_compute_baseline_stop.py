"""Compute baseline stopping points based on mean object confidence thresholds."""

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
    thetas = cfg["baseline"]["mean_object_conf_thresholds"]

    selected_path = resolve(cfg["paths"]["selected_images"])
    with open(selected_path) as f:
        image_ids = json.load(f)

    metrics_root = resolve(cfg["paths"]["metrics"])

    fieldnames = ["image_id"] + [f"baseline_{t}" for t in thetas]
    rows = []

    for img_id in image_ids:
        img_csv = metrics_root / f"{img_id}.csv"
        with open(img_csv) as f:
            reader = csv.DictReader(f)
            levels = list(reader)

        row: dict[str, object] = {"image_id": img_id}

        for theta in thetas:
            stop_level = float("nan")
            for lv in levels:
                if float(lv["mean_object_confidence"]) >= theta:
                    stop_level = int(lv["level"])
                    break
            if math.isnan(stop_level):
                stop_level = 20
            row[f"baseline_{theta}"] = stop_level

        rows.append(row)
        main_t = cfg["baseline"]["mean_object_conf_threshold"]
        print(
            f"  {img_id}: baseline_{main_t}={row[f'baseline_{main_t}']}"
        )

    out_path = metrics_root / "baseline_stop.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved baseline stop points to {out_path}")


if __name__ == "__main__":
    main()
