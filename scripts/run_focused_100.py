"""Focused LLM+VLM evaluation for 100-image scale-up.

Runs ONLY the focused configs:
  - LLM-SGG: 4b h5 rel=on conf=on, 9b h5 rel=on conf=on
  - VLM: single mode, 4b and 9b

Order: all 4b (LLM + VLM single) then all 9b (LLM + VLM single).
Reuses cached per-level JSON files (skip-existing).
"""

import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import load_config, resolve
from milestone1.llm import smoke_test_ollama
from milestone1.llm_engine import (
    build_prompt,
    build_sequence,
    make_config_slug,
    score_with_config,
)
from milestone1.vlm import score_vlm_single


def model_slug(model: str) -> str:
    return model.replace(":", "-").replace("/", "-")


def run_llm_focused(
    config_tuple: tuple[str, int, bool, bool],
    image_ids: list[int],
    chat_endpoint: str,
    pred_root: Path,
    output_root: Path,
    start_level: int,
    num_levels: int,
    sat_threshold: float,
    timeout: int,
    input_obj_threshold: float,
    input_rel_threshold: float,
    input_top_k_objects: int,
    input_top_k_relationships: int,
) -> list[dict]:
    """Run one LLM config across all images. Returns stop-level rows."""
    model, history_length, relation_on, confidence_on = config_tuple
    slug = make_config_slug(model, history_length, relation_on, confidence_on)
    prompt = build_prompt(relation_on, confidence_on)

    print(f"\n--- LLM Config: {slug} ---")

    total_calls = 0
    cached_calls = 0
    failed_calls = 0
    stop_rows = []

    for img_idx, img_id in enumerate(image_ids):
        out_dir = output_root / slug / str(img_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        stop_level = num_levels

        for level_idx in range(start_level, num_levels + 1):
            level_name = f"level_{level_idx:02d}"
            out_path = out_dir / f"{level_name}.json"

            if out_path.exists():
                with open(out_path) as f:
                    result = json.load(f)
                cached_calls += 1
            else:
                sequence = build_sequence(
                    image_id=img_id,
                    current_level=level_idx,
                    history_length=history_length,
                    pred_root=pred_root,
                    relation_on=relation_on,
                    confidence_on=confidence_on,
                    input_obj_threshold=input_obj_threshold,
                    input_rel_threshold=input_rel_threshold,
                    input_top_k_objects=input_top_k_objects,
                    input_top_k_relationships=input_top_k_relationships,
                )

                t0 = time.time()
                result = score_with_config(
                    sequence, prompt, model, chat_endpoint, timeout
                )
                elapsed = time.time() - t0
                total_calls += 1

                if "error" in result.get("reason", "") or "parse" in result.get(
                    "reason", ""
                ):
                    failed_calls += 1

                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)

                print(
                    f"  [{img_idx + 1}/{len(image_ids)}] {img_id} L{level_idx:02d}: "
                    f"sat={result['saturation_score']:.2f} ({elapsed:.1f}s)"
                )

            if result["saturation_score"] >= sat_threshold and stop_level == num_levels:
                stop_level = level_idx

        stop_rows.append(
            {
                "config_slug": slug,
                "model": model,
                "history_length": history_length,
                "relation_on": relation_on,
                "confidence_on": confidence_on,
                "image_id": img_id,
                "stop_level": stop_level,
            }
        )

    print(
        f"  {slug}: new_calls={total_calls}, cached={cached_calls}, "
        f"failed={failed_calls}"
    )
    return stop_rows


def run_vlm_focused(
    image_ids: list[int],
    model: str,
    chat_endpoint: str,
    degraded_root: Path,
    output_root: Path,
    start_level: int,
    num_levels: int,
    max_image_side: int,
    timeout: int,
    sat_threshold: float,
) -> list[dict]:
    """Run VLM single-mode scoring for one model across all images."""
    slug = model_slug(model)
    mode = "single"
    total_calls = 0
    cached_calls = 0
    failed_calls = 0
    stop_rows = []

    print(f"\n--- VLM {model} single ---")

    for img_idx, img_id in enumerate(image_ids):
        out_dir = output_root / slug / mode / str(img_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        stop_k1 = num_levels
        stop_k2 = num_levels
        consecutive_count = 0

        for level_idx in range(start_level, num_levels + 1):
            level_name = f"level_{level_idx:02d}"
            out_path = out_dir / f"{level_name}.json"

            if out_path.exists():
                with open(out_path) as f:
                    result = json.load(f)
                cached_calls += 1
            else:
                t0 = time.time()
                img_path = str(degraded_root / str(img_id) / f"{level_name}.jpg")
                result = score_vlm_single(
                    img_path, model, chat_endpoint, max_image_side, timeout
                )
                elapsed = time.time() - t0
                total_calls += 1

                if "error" in result.get("reason", ""):
                    failed_calls += 1

                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)

                print(
                    f"  [{img_idx + 1}/{len(image_ids)}] {img_id} L{level_idx:02d} "
                    f"single: sat={result['saturation_score']:.2f} ({elapsed:.1f}s)"
                )

            score = result["saturation_score"]
            if score >= sat_threshold:
                if stop_k1 == num_levels:
                    stop_k1 = level_idx
                consecutive_count += 1
                if consecutive_count >= 2 and stop_k2 == num_levels:
                    stop_k2 = level_idx
            else:
                consecutive_count = 0

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
        f"  {model} single: new_calls={total_calls}, cached={cached_calls}, "
        f"failed={failed_calls}"
    )
    return stop_rows


def main() -> None:
    cfg = load_config()
    llm_cfg = cfg["llm"]
    vlm_cfg = cfg["vlm"]
    chat_endpoint = cfg["ollama_chat_endpoint"]
    start_level = llm_cfg["start_level"]
    sat_threshold = llm_cfg["saturation_threshold"]
    timeout_llm = llm_cfg["request_timeout"]
    timeout_vlm = vlm_cfg["request_timeout"]
    num_levels = cfg["resolution_levels"]["num_levels"]
    max_image_side = vlm_cfg["max_image_side"]

    input_obj_threshold = llm_cfg["input_object_conf_threshold"]
    input_rel_threshold = llm_cfg["input_relation_conf_threshold"]
    input_top_k_objects = llm_cfg["input_top_k_objects"]
    input_top_k_relationships = llm_cfg["input_top_k_relationships"]

    selected_path = resolve(cfg["paths"]["selected_images"])
    with open(selected_path) as f:
        image_ids = json.load(f)

    pred_root = resolve(cfg["paths"]["egtr_predictions"])
    degraded_root = resolve(cfg["paths"]["degraded_images"])
    ablation_root = resolve("outputs/llm_ablation")
    vlm_output_root = resolve("outputs/vlm_scores")
    metrics_root = resolve(cfg["paths"]["metrics"])

    # Focused configs: only 4b and 9b base (h5, rel_on, conf_on)
    focused_configs = [
        ("qwen3.5:4b", 5, True, True),
        ("qwen3.5:9b", 5, True, True),
    ]

    all_llm_stop_rows: list[dict] = []
    all_vlm_stop_rows: list[dict] = []

    for config_tuple in focused_configs:
        model = config_tuple[0]

        # Smoke test
        print(f"\n{'=' * 60}")
        print(f"Model: {model} (LLM + VLM single)")
        print(f"{'=' * 60}")
        print(f"Smoke-testing Ollama ({model}) ...")

        generate_endpoint = chat_endpoint.replace("/api/chat", "/api/generate")
        if not smoke_test_ollama(model, generate_endpoint, timeout=30):
            print(f"ERROR: smoke test failed for {model}. Skipping.")
            continue
        print(f"Smoke test passed for {model}.")

        # LLM first (lighter calls)
        llm_rows = run_llm_focused(
            config_tuple=config_tuple,
            image_ids=image_ids,
            chat_endpoint=chat_endpoint,
            pred_root=pred_root,
            output_root=ablation_root,
            start_level=start_level,
            num_levels=num_levels,
            sat_threshold=sat_threshold,
            timeout=timeout_llm,
            input_obj_threshold=input_obj_threshold,
            input_rel_threshold=input_rel_threshold,
            input_top_k_objects=input_top_k_objects,
            input_top_k_relationships=input_top_k_relationships,
        )
        all_llm_stop_rows.extend(llm_rows)

        # VLM single (heavier calls)
        vlm_rows = run_vlm_focused(
            image_ids=image_ids,
            model=model,
            chat_endpoint=chat_endpoint,
            degraded_root=degraded_root,
            output_root=vlm_output_root,
            start_level=start_level,
            num_levels=num_levels,
            max_image_side=max_image_side,
            timeout=timeout_vlm,
            sat_threshold=sat_threshold,
        )
        all_vlm_stop_rows.extend(vlm_rows)

    # Save focused LLM stop CSV (append-safe: overwrite with all data)
    # We need to also preserve existing ablation configs not in focused set
    existing_ablation_path = metrics_root / "llm_ablation_stop.csv"
    existing_llm_rows: list[dict] = []
    if existing_ablation_path.exists():
        with open(existing_ablation_path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                # Keep non-focused configs
                slug = r["config_slug"]
                if slug not in [make_config_slug(*c) for c in focused_configs]:
                    existing_llm_rows.append(r)

    combined_llm_rows = existing_llm_rows + all_llm_stop_rows
    fieldnames_llm = [
        "config_slug",
        "model",
        "history_length",
        "relation_on",
        "confidence_on",
        "image_id",
        "stop_level",
    ]
    with open(existing_ablation_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_llm)
        writer.writeheader()
        writer.writerows(combined_llm_rows)
    print(f"\nSaved LLM ablation stop CSV to {existing_ablation_path}")

    # Save focused VLM stop CSV
    # Preserve existing non-single or non-focused model entries
    existing_vlm_path = metrics_root / "vlm_stop.csv"
    existing_vlm_rows: list[dict] = []
    focused_models = {c[0] for c in focused_configs}
    if existing_vlm_path.exists():
        with open(existing_vlm_path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                # Keep entries NOT in (focused_model, single)
                if not (r["model"] in focused_models and r["mode"] == "single"):
                    existing_vlm_rows.append(r)

    combined_vlm_rows = existing_vlm_rows + all_vlm_stop_rows
    fieldnames_vlm = ["image_id", "model", "mode", "k", "stop_level"]
    with open(existing_vlm_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_vlm)
        writer.writeheader()
        writer.writerows(combined_vlm_rows)
    print(f"Saved VLM stop CSV to {existing_vlm_path}")

    # Print summary
    print("\n--- Summary ---")
    for slug_label in [make_config_slug(*c) for c in focused_configs]:
        subset = [r for r in all_llm_stop_rows if r["config_slug"] == slug_label]
        if subset:
            stops = [r["stop_level"] for r in subset]
            print(f"  LLM {slug_label}: mean_stop={sum(stops) / len(stops):.1f}")

    for model in ["qwen3.5:4b", "qwen3.5:9b"]:
        for k_val in [1, 2]:
            subset = [
                r for r in all_vlm_stop_rows if r["model"] == model and r["k"] == k_val
            ]
            if subset:
                stops = [r["stop_level"] for r in subset]
                print(
                    f"  VLM {model} single k={k_val}: "
                    f"mean_stop={sum(stops) / len(stops):.1f}"
                )


if __name__ == "__main__":
    main()
