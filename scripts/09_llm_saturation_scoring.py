"""Run LLM-based semantic saturation scoring via Ollama."""

import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import load_config, resolve
from milestone1.llm import score_saturation, smoke_test_ollama


def build_sequence(
    image_id: int,
    current_level: int,
    history_length: int,
    pred_root: Path,
    input_obj_threshold: float,
    input_rel_threshold: float,
    input_top_k_objects: int,
    input_top_k_relationships: int,
) -> list[dict]:
    """Build the recent-history scene graph sequence for LLM input."""
    start = max(1, current_level - history_length + 1)
    sequence = []

    for lv in range(start, current_level + 1):
        level_name = f"level_{lv:02d}"
        pred_path = pred_root / str(image_id) / f"{level_name}.json"

        if not pred_path.exists():
            continue

        with open(pred_path) as f:
            pred = json.load(f)

        # Filter objects by threshold and top-k
        objects = [
            o for o in pred["objects"]
            if o["confidence"] >= input_obj_threshold
        ][:input_top_k_objects]

        # Filter relationships by threshold and top-k
        relationships = [
            r for r in pred["relationships"]
            if r["confidence"] >= input_rel_threshold
        ][:input_top_k_relationships]

        entry = {
            "level": lv,
            "objects": [
                {
                    "object_id": o["object_id"],
                    "label": o["label"],
                    "confidence": o["confidence"],
                }
                for o in objects
            ],
            "relationships": [
                {
                    "subject_id": r["subject_id"],
                    "subject_label": r["subject_label"],
                    "predicate": r["predicate"],
                    "object_id": r["object_id"],
                    "object_label": r["object_label"],
                    "confidence": r["confidence"],
                }
                for r in relationships
            ],
        }
        sequence.append(entry)

    return sequence


def main() -> None:
    cfg = load_config()
    llm_cfg = cfg["llm"]
    model = llm_cfg["model"]
    endpoint = llm_cfg["endpoint"]
    start_level = llm_cfg["start_level"]
    history_length = llm_cfg["history_length"]
    sat_threshold = llm_cfg["saturation_threshold"]
    timeout = llm_cfg["request_timeout"]

    input_obj_threshold = llm_cfg["input_object_conf_threshold"]
    input_rel_threshold = llm_cfg["input_relation_conf_threshold"]
    input_top_k_objects = llm_cfg["input_top_k_objects"]
    input_top_k_relationships = llm_cfg["input_top_k_relationships"]

    num_levels = cfg["resolution_levels"]["num_levels"]

    selected_path = resolve(cfg["paths"]["selected_images"])
    with open(selected_path) as f:
        image_ids = json.load(f)

    pred_root = resolve(cfg["paths"]["egtr_predictions"])
    llm_inputs_root = resolve(cfg["paths"]["llm_inputs"])
    llm_scores_root = resolve(cfg["paths"]["llm_scores"])
    metrics_root = resolve(cfg["paths"]["metrics"])

    # Smoke test
    print(f"Smoke-testing Ollama ({model} at {endpoint}) ...")
    if not smoke_test_ollama(model, endpoint, timeout=30):
        print("ERROR: Ollama is unreachable or model not loaded. Aborting.")
        sys.exit(1)
    print("Ollama smoke test passed.")

    llm_stop_rows = []
    total_calls = 0
    failed_calls = 0

    for img_id in image_ids:
        llm_in_dir = llm_inputs_root / str(img_id)
        llm_out_dir = llm_scores_root / str(img_id)
        llm_in_dir.mkdir(parents=True, exist_ok=True)
        llm_out_dir.mkdir(parents=True, exist_ok=True)

        llm_stop_level = 20  # default if never triggered

        for level_idx in range(start_level, num_levels + 1):
            level_name = f"level_{level_idx:02d}"

            # Build sequence
            sequence = build_sequence(
                image_id=img_id,
                current_level=level_idx,
                history_length=history_length,
                pred_root=pred_root,
                input_obj_threshold=input_obj_threshold,
                input_rel_threshold=input_rel_threshold,
                input_top_k_objects=input_top_k_objects,
                input_top_k_relationships=input_top_k_relationships,
            )

            # Save input
            input_path = llm_in_dir / f"{level_name}.json"
            with open(input_path, "w") as f:
                json.dump(sequence, f, indent=2)

            # Call LLM
            t0 = time.time()
            result = score_saturation(
                sequence, model, endpoint, timeout=timeout
            )
            elapsed = time.time() - t0
            total_calls += 1

            if "error" in result.get("reason", ""):
                failed_calls += 1

            # Save output
            output_path = llm_out_dir / f"{level_name}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            # Check for stop
            if (
                result["saturation_score"] >= sat_threshold
                and llm_stop_level == 20
            ):
                llm_stop_level = level_idx

            print(
                f"  {img_id} L{level_idx:02d}: "
                f"sat={result['saturation_score']:.2f} "
                f"stop={result['stop_decision']} "
                f"({elapsed:.1f}s)"
            )

        llm_stop_rows.append({
            "image_id": img_id,
            "llm_stop": llm_stop_level,
        })
        print(f"  {img_id}: LLM stop = level {llm_stop_level}")

    # Save LLM stop CSV
    out_path = metrics_root / "llm_stop.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "llm_stop"])
        writer.writeheader()
        writer.writerows(llm_stop_rows)

    print(f"\nTotal LLM calls: {total_calls}, failed: {failed_calls}")
    print(f"Saved LLM stop points to {out_path}")


if __name__ == "__main__":
    main()
