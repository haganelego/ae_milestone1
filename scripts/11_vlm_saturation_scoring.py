"""Run VLM-based semantic saturation scoring via Ollama vision models.

Scores each image at levels start_level..20 using both single-image and
sequence modes, for each configured VLM model. Outputs per-level JSON
scores and a tidy stop-level CSV.

Global ordering: all 4b work first, then all 9b work, to minimize VRAM reloads.
"""

import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import load_config, resolve
from milestone1.vlm import (
    score_vlm_sequence,
    score_vlm_single,
    smoke_test_vlm,
)


def model_slug(model: str) -> str:
    """Convert model name to filesystem-safe slug."""
    return model.replace(":", "-").replace("/", "-")


def run_vlm_scoring(
    image_ids: list[int],
    model: str,
    mode: str,
    chat_endpoint: str,
    degraded_root: Path,
    output_root: Path,
    start_level: int,
    num_levels: int,
    sequence_history: int,
    max_image_side: int,
    timeout: int,
    sat_threshold: float,
) -> list[dict]:
    """Run VLM scoring for one model+mode combo across all images and levels.

    Returns list of stop-level rows for the CSV.
    """
    slug = model_slug(model)
    total_calls = 0
    failed_calls = 0
    stop_rows = []

    for img_id in image_ids:
        out_dir = output_root / slug / mode / str(img_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        stop_k1 = num_levels  # default: never stopped
        stop_k2 = num_levels
        consecutive_count = 0

        for level_idx in range(start_level, num_levels + 1):
            level_name = f"level_{level_idx:02d}"
            out_path = out_dir / f"{level_name}.json"

            # Check cache
            if out_path.exists():
                with open(out_path) as f:
                    result = json.load(f)
                print(
                    f"  [cache] {img_id} L{level_idx:02d} {mode}: "
                    f"sat={result['saturation_score']:.2f}"
                )
            else:
                t0 = time.time()

                if mode == "single":
                    img_path = str(
                        degraded_root / str(img_id) / f"{level_name}.jpg"
                    )
                    result = score_vlm_single(
                        img_path, model, chat_endpoint, max_image_side, timeout
                    )
                else:  # sequence
                    # Collect recent history images
                    hist_start = max(1, level_idx - sequence_history + 1)
                    img_paths = []
                    for lv in range(hist_start, level_idx + 1):
                        p = degraded_root / str(img_id) / f"level_{lv:02d}.jpg"
                        if p.exists():
                            img_paths.append(str(p))
                    result = score_vlm_sequence(
                        img_paths, model, chat_endpoint, max_image_side, timeout
                    )

                elapsed = time.time() - t0
                total_calls += 1

                if "error" in result.get("reason", ""):
                    failed_calls += 1

                # Save result
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)

                print(
                    f"  {img_id} L{level_idx:02d} {mode}: "
                    f"sat={result['saturation_score']:.2f} "
                    f"stop={result['stop_decision']} "
                    f"({elapsed:.1f}s)"
                )

            # Track stop levels
            score = result["saturation_score"]
            if score >= sat_threshold:
                if stop_k1 == num_levels:
                    stop_k1 = level_idx
                consecutive_count += 1
                if consecutive_count >= 2 and stop_k2 == num_levels:
                    # k=2 stop is at the level where 2nd consecutive hit occurs
                    stop_k2 = level_idx
            else:
                consecutive_count = 0

        # Record stop rows for k=1 and k=2
        stop_rows.append(
            {
                "image_id": img_id,
                "model": model,
                "mode": mode,
                "k": 1,
                "stop_level": stop_k1,
            }
        )
        stop_rows.append(
            {
                "image_id": img_id,
                "model": model,
                "mode": mode,
                "k": 2,
                "stop_level": stop_k2,
            }
        )
        print(
            f"  {img_id} {mode}: stop_k1={stop_k1}, stop_k2={stop_k2}"
        )

    print(
        f"  Model={model} mode={mode}: "
        f"total_calls={total_calls}, failed={failed_calls}"
    )
    return stop_rows


def main() -> None:
    cfg = load_config()
    vlm_cfg = cfg["vlm"]
    chat_endpoint = cfg["ollama_chat_endpoint"]
    models = vlm_cfg["models"]
    modes = vlm_cfg["modes"]
    start_level = vlm_cfg["start_level"]
    sequence_history = vlm_cfg["sequence_history"]
    sat_threshold = vlm_cfg["saturation_threshold"]
    max_image_side = vlm_cfg["max_image_side"]
    timeout = vlm_cfg["request_timeout"]
    num_levels = cfg["resolution_levels"]["num_levels"]

    selected_path = resolve(cfg["paths"]["selected_images"])
    with open(selected_path) as f:
        image_ids = json.load(f)

    degraded_root = resolve(cfg["paths"]["degraded_images"])
    vlm_output_root = resolve("outputs/vlm_scores")
    metrics_root = resolve(cfg["paths"]["metrics"])

    # Ordered: 4b models first, then 9b, to minimize VRAM reloads
    ordered_models = sorted(models, key=lambda m: "9b" in m)

    all_stop_rows: list[dict] = []

    for model in ordered_models:
        print(f"\n{'='*60}")
        print(f"VLM Model: {model}")
        print(f"{'='*60}")

        # Smoke test
        print(f"Smoke-testing VLM ({model}) ...")
        if not smoke_test_vlm(model, chat_endpoint, timeout=60):
            print(f"ERROR: VLM smoke test failed for {model}. Aborting this model.")
            # Record failure rows
            for img_id in image_ids:
                for mode in modes:
                    for k in [1, 2]:
                        all_stop_rows.append(
                            {
                                "image_id": img_id,
                                "model": model,
                                "mode": mode,
                                "k": k,
                                "stop_level": -1,
                            }
                        )
            continue
        print(f"VLM smoke test passed for {model}.")

        for mode in modes:
            print(f"\n--- Mode: {mode} ---")
            rows = run_vlm_scoring(
                image_ids=image_ids,
                model=model,
                mode=mode,
                chat_endpoint=chat_endpoint,
                degraded_root=degraded_root,
                output_root=vlm_output_root,
                start_level=start_level,
                num_levels=num_levels,
                sequence_history=sequence_history,
                max_image_side=max_image_side,
                timeout=timeout,
                sat_threshold=sat_threshold,
            )
            all_stop_rows.extend(rows)

    # Save tidy stop CSV
    metrics_root.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_root / "vlm_stop.csv"
    fieldnames = ["image_id", "model", "mode", "k", "stop_level"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stop_rows)

    print(f"\nSaved VLM stop CSV to {csv_path}")

    # Print summary
    print("\n--- VLM Stop Summary ---")
    for model in ordered_models:
        for mode in modes:
            for k in [1, 2]:
                subset = [
                    r["stop_level"]
                    for r in all_stop_rows
                    if r["model"] == model
                    and r["mode"] == mode
                    and r["k"] == k
                    and r["stop_level"] > 0
                ]
                if subset:
                    mean_stop = sum(subset) / len(subset)
                    print(
                        f"  {model} {mode} k={k}: "
                        f"mean_stop={mean_stop:.1f} "
                        f"(n={len(subset)})"
                    )


if __name__ == "__main__":
    main()
