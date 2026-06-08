"""Run LLM-SGG ablation study: OFAT configs scored via ollama.

Configs (OFAT from base = 9b, h5, rel=on, conf=on):
  1. base: 9b, h5, rel=on, conf=on
  2. model axis: 4b, h5, rel=on, conf=on
  3. history: h1 (9b), h3 (9b)
  4. relation: rel=off (9b, h5, conf=on)
  5. confidence: conf=off (9b, h5, rel=on)

Ordering: 4b config first, then all 9b configs, to minimize VRAM reloads.
Caches per-level scores so reruns are free.
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

# OFAT configs: (model, history_length, relation_on, confidence_on)
OFAT_CONFIGS = [
    # 4b first (to group with VLM 4b in global ordering)
    ("qwen3.5:4b", 5, True, True),
    # 9b configs
    ("qwen3.5:9b", 5, True, True),     # base
    ("qwen3.5:9b", 1, True, True),     # history axis: h1
    ("qwen3.5:9b", 3, True, True),     # history axis: h3
    ("qwen3.5:9b", 5, False, True),    # relation axis: rel=off
    ("qwen3.5:9b", 5, True, False),    # confidence axis: conf=off
]


def run_ablation_config(
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
    """Run one ablation config across all images. Returns stop-level rows."""
    model, history_length, relation_on, confidence_on = config_tuple
    slug = make_config_slug(model, history_length, relation_on, confidence_on)
    prompt = build_prompt(relation_on, confidence_on)

    print(f"\n--- Config: {slug} ---")

    total_calls = 0
    failed_calls = 0
    stop_rows = []

    for img_id in image_ids:
        out_dir = output_root / slug / str(img_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        stop_level = num_levels  # default

        for level_idx in range(start_level, num_levels + 1):
            level_name = f"level_{level_idx:02d}"
            out_path = out_dir / f"{level_name}.json"

            # Check cache
            if out_path.exists():
                with open(out_path) as f:
                    result = json.load(f)
                print(
                    f"  [cache] {img_id} L{level_idx:02d}: "
                    f"sat={result['saturation_score']:.2f}"
                )
            else:
                # Build sequence
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

                # Save
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)

                print(
                    f"  {img_id} L{level_idx:02d}: "
                    f"sat={result['saturation_score']:.2f} "
                    f"stop={result['stop_decision']} "
                    f"({elapsed:.1f}s)"
                )

            # Track stop level (k=1)
            if (
                result["saturation_score"] >= sat_threshold
                and stop_level == num_levels
            ):
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
        print(f"  {img_id}: stop_level={stop_level}")

    print(
        f"  Config {slug}: total_calls={total_calls}, failed={failed_calls}"
    )
    return stop_rows


def main() -> None:
    cfg = load_config()
    llm_cfg = cfg["llm"]
    chat_endpoint = cfg["ollama_chat_endpoint"]
    start_level = llm_cfg["start_level"]
    sat_threshold = llm_cfg["saturation_threshold"]
    timeout = llm_cfg["request_timeout"]
    num_levels = cfg["resolution_levels"]["num_levels"]

    input_obj_threshold = llm_cfg["input_object_conf_threshold"]
    input_rel_threshold = llm_cfg["input_relation_conf_threshold"]
    input_top_k_objects = llm_cfg["input_top_k_objects"]
    input_top_k_relationships = llm_cfg["input_top_k_relationships"]

    selected_path = resolve(cfg["paths"]["selected_images"])
    with open(selected_path) as f:
        image_ids = json.load(f)

    pred_root = resolve(cfg["paths"]["egtr_predictions"])
    ablation_root = resolve("outputs/llm_ablation")
    metrics_root = resolve(cfg["paths"]["metrics"])

    all_stop_rows: list[dict] = []

    # Group configs by model to minimize reloads
    current_model = None
    for config_tuple in OFAT_CONFIGS:
        model = config_tuple[0]

        # Smoke test when switching models
        if model != current_model:
            print(f"\n{'='*60}")
            print(f"LLM Ablation Model: {model}")
            print(f"{'='*60}")
            print(f"Smoke-testing Ollama ({model}) ...")

            # Use chat endpoint directly
            if not smoke_test_ollama(model, chat_endpoint.replace("/api/chat", "/api/generate"), timeout=30):
                print(f"ERROR: Ollama smoke test failed for {model}. Skipping.")
                # Record failure rows
                slug = make_config_slug(*config_tuple)
                for img_id in image_ids:
                    all_stop_rows.append(
                        {
                            "config_slug": slug,
                            "model": model,
                            "history_length": config_tuple[1],
                            "relation_on": config_tuple[2],
                            "confidence_on": config_tuple[3],
                            "image_id": img_id,
                            "stop_level": -1,
                        }
                    )
                continue
            print(f"Smoke test passed for {model}.")
            current_model = model

        rows = run_ablation_config(
            config_tuple=config_tuple,
            image_ids=image_ids,
            chat_endpoint=chat_endpoint,
            pred_root=pred_root,
            output_root=ablation_root,
            start_level=start_level,
            num_levels=num_levels,
            sat_threshold=sat_threshold,
            timeout=timeout,
            input_obj_threshold=input_obj_threshold,
            input_rel_threshold=input_rel_threshold,
            input_top_k_objects=input_top_k_objects,
            input_top_k_relationships=input_top_k_relationships,
        )
        all_stop_rows.extend(rows)

    # Save tidy stop CSV
    metrics_root.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_root / "llm_ablation_stop.csv"
    fieldnames = [
        "config_slug",
        "model",
        "history_length",
        "relation_on",
        "confidence_on",
        "image_id",
        "stop_level",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stop_rows)

    print(f"\nSaved LLM ablation stop CSV to {csv_path}")

    # Print summary
    print("\n--- LLM Ablation Stop Summary ---")
    seen_slugs: list[str] = []
    for row in all_stop_rows:
        s = row["config_slug"]
        if s not in seen_slugs:
            seen_slugs.append(s)
    for slug in seen_slugs:
        subset = [
            r["stop_level"]
            for r in all_stop_rows
            if r["config_slug"] == slug and r["stop_level"] > 0
        ]
        if subset:
            mean_stop = sum(subset) / len(subset)
            print(f"  {slug}: mean_stop={mean_stop:.1f} (n={len(subset)})")
        else:
            print(f"  {slug}: no valid stops")


if __name__ == "__main__":
    main()
