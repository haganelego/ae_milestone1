"""Verify a gate=0.1 + threshold=0.4 node-confidence baseline (isolated).

Re-runs EGTR inference with object_conf_threshold=0.1 (instead of 0.3) and
recomputes ONLY the baseline stop signal (mean object confidence first-crossing
0.4).  The stop_c09 oracle and RelSGScore are NOT touched -- they remain derived
from the gate=0.3 predictions in outputs/egtr_predictions -- so this is an
apples-to-apples baseline comparison against the SAME oracle.

Outputs:
  - outputs/metrics/baseline_g01_mean_conf.csv   (per image/level mean conf)
  - prints stop distribution + MAE vs stop_c09 vs the current gate0.3 baseline

Run:
  uv run --project egtr python scripts/verify_gate01_baseline.py
"""

from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import load_config, resolve  # noqa: E402
from milestone1.egtr_infer import EgtrModel  # noqa: E402

GATE = 0.1
# Inference (gate) is the expensive step and is run once; thresholds are swept
# cheaply from the saved per-level mean confidence.
STOP_THRESHOLDS = [0.30, 0.32, 0.35, 0.38, 0.40]
MAX_LEVEL = 20


def first_crossing(series: dict[int, float], thr: float) -> int:
    """First level (1..MAX_LEVEL) where mean confidence >= thr, else MAX_LEVEL."""
    for lvl in range(1, MAX_LEVEL + 1):
        if series.get(lvl, 0.0) >= thr:
            return lvl
    return MAX_LEVEL


def main() -> None:
    cfg = load_config()
    degraded_root = resolve(cfg["paths"]["degraded_images"])
    metrics_root = resolve(cfg["paths"]["metrics"])
    selected_path = resolve(cfg["paths"]["selected_images"])
    image_ids: list[int] = json.load(open(selected_path))

    print(f"Loading EGTR with object_conf_threshold={GATE} ...")
    egtr = EgtrModel(
        artifact_path=str(resolve(cfg["paths"]["egtr_artifact"])),
        architecture=cfg["paths"]["egtr_architecture"],
        vg_root=cfg["paths"]["vg_root"],
        object_conf_threshold=GATE,
        relation_conf_threshold=cfg["egtr"]["relation_conf_threshold"],
        top_k_objects=cfg["egtr"]["top_k_objects"],
        top_k_relationships=cfg["egtr"]["top_k_relationships"],
    )

    mean_conf: dict[int, dict[int, float]] = {}
    rows: list[dict] = []
    total = len(image_ids) * MAX_LEVEL
    done = 0
    for img_id in image_ids:
        mean_conf[img_id] = {}
        for lvl in range(1, MAX_LEVEL + 1):
            img_path = degraded_root / str(img_id) / f"level_{lvl:02d}.jpg"
            pil = Image.open(img_path).convert("RGB")
            result = egtr.predict(pil)
            objs = result["objects"]
            moc = st.mean([o["confidence"] for o in objs]) if objs else 0.0
            mean_conf[img_id][lvl] = moc
            rows.append(
                {
                    "image_id": img_id,
                    "level": lvl,
                    "mean_object_confidence": round(moc, 4),
                    "n_objects": len(objs),
                }
            )
            done += 1
        if image_ids.index(img_id) % 10 == 0:
            print(f"  {done}/{total} ...")

    out_csv = metrics_root / "baseline_g01_mean_conf.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["image_id", "level", "mean_object_confidence", "n_objects"]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out_csv}")

    # Reference oracle (FIXED, gate0.3) + current real baseline (gate0.3 + thr0.4)
    oc = {
        int(r["image_id"]): int(r["stop_c09"])
        for r in csv.DictReader(
            open(metrics_root / "topk_oracle" / "oracle_topk_per_image.csv")
        )
    }
    b04 = {
        int(r["image_id"]): int(float(r["baseline_0.4"]))
        for r in csv.DictReader(open(metrics_root / "baseline_stop.csv"))
    }

    def stats(stops: dict[int, int]) -> dict[str, float]:
        sv = [stops[i] for i in image_ids]
        return {
            "mean": st.mean(sv),
            "median": st.median(sv),
            "max": max(sv),
            "never": st.mean([stops[i] == MAX_LEVEL for i in image_ids]),
            "mae": st.mean([abs(stops[i] - oc[i]) for i in image_ids]),
            "medae": st.median([abs(stops[i] - oc[i]) for i in image_ids]),
            "early": st.mean([stops[i] < oc[i] for i in image_ids]),
            "late": st.mean([stops[i] > oc[i] for i in image_ids]),
        }

    print("\n=== gate=0.1 baseline vs stop_c09 (n=100) ===")
    print("Level-mean confidence (gate0.1):")
    for lvl in [1, 5, 10, 16, 20]:
        vals = [mean_conf[i][lvl] for i in image_ids]
        print(
            f"  L{lvl:2d}: mean={st.mean(vals):.3f} min={min(vals):.3f} "
            f"max={max(vals):.3f} %>=0.35={st.mean([v >= 0.35 for v in vals]):.2f}"
        )

    print(f"\n  stop_c09 oracle: mean={st.mean([oc[i] for i in image_ids]):.1f}")
    print("\n  threshold     meanStop median max never%  MAE  MedAE early late")
    print("  " + "-" * 68)
    for thr in STOP_THRESHOLDS:
        s = stats({i: first_crossing(mean_conf[i], thr) for i in image_ids})
        print(
            f"  gate0.1/{thr:.2f}   {s['mean']:>6.1f}  {s['median']:>5}  "
            f"{s['max']:>3}  {s['never']:>4.0%}  {s['mae']:>4.2f}  "
            f"{s['medae']:>4.1f}  {s['early']:>4.2f} {s['late']:>4.2f}"
        )
    sb = stats(b04)
    print(
        f"  gate0.3/0.40* {sb['mean']:>6.1f}  {sb['median']:>5}  {sb['max']:>3}  "
        f"{sb['never']:>4.0%}  {sb['mae']:>4.2f}  {sb['medae']:>4.1f}  "
        f"{sb['early']:>4.2f} {sb['late']:>4.2f}  (current real)"
    )


if __name__ == "__main__":
    main()
