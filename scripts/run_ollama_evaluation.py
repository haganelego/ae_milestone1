"""Unified runner for all ollama-based evaluation tasks.

Enforces global model ordering to minimize VRAM reloads:
  Phase 1 (qwen3.5:4b): VLM single + sequence, LLM ablation 4b config
  Phase 2 (qwen3.5:9b): VLM single + sequence, LLM ablation 9b configs

This script orchestrates the VLM and LLM ablation scoring by directly
calling the underlying functions rather than spawning subprocesses.
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
from milestone1.vlm import (
    score_vlm_sequence,
    score_vlm_single,
    smoke_test_vlm,
)


# ── VLM scoring ──────────────────────────────────────────────────────

def run_vlm_for_model(
    model: str,
    image_ids: list[int],
    modes: list[str],
    chat_endpoint: str,
    degraded_root: Path,
    vlm_output_root: Path,
    start_level: int,
    num_levels: int,
    sequence_history: int,
    max_image_side: int,
    timeout: int,
    sat_threshold: float,
) -> list[dict]:
    """Run VLM scoring for one model, all modes."""
    slug = model.replace(":", "-").replace("/", "-")
    all_rows: list[dict] = []

    for mode in modes:
        print(f"\n  --- VLM {model} / {mode} ---")
        total_calls = 0
        failed_calls = 0

        for img_id in image_ids:
            out_dir = vlm_output_root / slug / mode / str(img_id)
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
                else:
                    t0 = time.time()
                    if mode == "single":
                        img_path = str(
                            degraded_root / str(img_id) / f"{level_name}.jpg"
                        )
                        result = score_vlm_single(
                            img_path, model, chat_endpoint, max_image_side, timeout
                        )
                    else:
                        hist_start = max(1, level_idx - sequence_history + 1)
                        img_paths = [
                            str(degraded_root / str(img_id) / f"level_{lv:02d}.jpg")
                            for lv in range(hist_start, level_idx + 1)
                            if (degraded_root / str(img_id) / f"level_{lv:02d}.jpg").exists()
                        ]
                        result = score_vlm_sequence(
                            img_paths, model, chat_endpoint, max_image_side, timeout
                        )
                    elapsed = time.time() - t0
                    total_calls += 1
                    if "error" in result.get("reason", ""):
                        failed_calls += 1
                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=2)
                    print(
                        f"    {img_id} L{level_idx:02d} {mode}: "
                        f"sat={result['saturation_score']:.2f} ({elapsed:.1f}s)"
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

            all_rows.append({"image_id": img_id, "model": model, "mode": mode, "k": 1, "stop_level": stop_k1})
            all_rows.append({"image_id": img_id, "model": model, "mode": mode, "k": 2, "stop_level": stop_k2})
            print(f"    {img_id} {mode}: k1={stop_k1}, k2={stop_k2}")

        print(f"  VLM {model}/{mode}: calls={total_calls}, failed={failed_calls}")
    return all_rows


# ── LLM ablation scoring ────────────────────────────────────────────

def run_llm_ablation_configs(
    configs: list[tuple[str, int, bool, bool]],
    image_ids: list[int],
    chat_endpoint: str,
    pred_root: Path,
    ablation_root: Path,
    start_level: int,
    num_levels: int,
    sat_threshold: float,
    timeout: int,
    input_obj_threshold: float,
    input_rel_threshold: float,
    input_top_k_objects: int,
    input_top_k_relationships: int,
) -> list[dict]:
    """Run LLM ablation for a list of configs (assumed same model)."""
    all_rows: list[dict] = []

    for config_tuple in configs:
        model, history_length, relation_on, confidence_on = config_tuple
        slug = make_config_slug(model, history_length, relation_on, confidence_on)
        prompt = build_prompt(relation_on, confidence_on)

        print(f"\n  --- LLM Ablation: {slug} ---")
        total_calls = 0
        failed_calls = 0

        for img_id in image_ids:
            out_dir = ablation_root / slug / str(img_id)
            out_dir.mkdir(parents=True, exist_ok=True)

            stop_level = num_levels

            for level_idx in range(start_level, num_levels + 1):
                level_name = f"level_{level_idx:02d}"
                out_path = out_dir / f"{level_name}.json"

                if out_path.exists():
                    with open(out_path) as f:
                        result = json.load(f)
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
                    if "error" in result.get("reason", "") or "parse" in result.get("reason", ""):
                        failed_calls += 1

                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=2)

                    print(
                        f"    {img_id} L{level_idx:02d}: "
                        f"sat={result['saturation_score']:.2f} ({elapsed:.1f}s)"
                    )

                if (
                    result["saturation_score"] >= sat_threshold
                    and stop_level == num_levels
                ):
                    stop_level = level_idx

            all_rows.append({
                "config_slug": slug,
                "model": model,
                "history_length": history_length,
                "relation_on": relation_on,
                "confidence_on": confidence_on,
                "image_id": img_id,
                "stop_level": stop_level,
            })
            print(f"    {img_id}: stop={stop_level}")

        print(f"  LLM {slug}: calls={total_calls}, failed={failed_calls}")
    return all_rows


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    vlm_cfg = cfg["vlm"]
    llm_cfg = cfg["llm"]
    chat_endpoint = cfg["ollama_chat_endpoint"]
    num_levels = cfg["resolution_levels"]["num_levels"]

    selected_path = resolve(cfg["paths"]["selected_images"])
    with open(selected_path) as f:
        image_ids = json.load(f)

    degraded_root = resolve(cfg["paths"]["degraded_images"])
    pred_root = resolve(cfg["paths"]["egtr_predictions"])
    vlm_output_root = resolve("outputs/vlm_scores")
    ablation_root = resolve("outputs/llm_ablation")
    metrics_root = resolve(cfg["paths"]["metrics"])
    metrics_root.mkdir(parents=True, exist_ok=True)

    # LLM ablation configs split by model
    configs_4b = [("qwen3.5:4b", 5, True, True)]
    configs_9b = [
        ("qwen3.5:9b", 5, True, True),     # base
        ("qwen3.5:9b", 1, True, True),     # h1
        ("qwen3.5:9b", 3, True, True),     # h3
        ("qwen3.5:9b", 5, False, True),    # rel=off
        ("qwen3.5:9b", 5, True, False),    # conf=off
    ]

    vlm_stop_rows: list[dict] = []
    llm_stop_rows: list[dict] = []

    # ── PHASE 1: qwen3.5:4b ──────────────────────────────────────
    model_4b = "qwen3.5:4b"
    print(f"\n{'='*70}")
    print(f"  PHASE 1: {model_4b}  (VLM + LLM ablation)")
    print(f"{'='*70}")

    # Smoke tests
    print(f"\nSmoke-testing VLM ({model_4b}) ...")
    vlm_4b_ok = smoke_test_vlm(model_4b, chat_endpoint, timeout=60)
    print(f"VLM smoke test: {'PASS' if vlm_4b_ok else 'FAIL'}")

    print(f"Smoke-testing LLM ({model_4b}) ...")
    generate_endpoint = chat_endpoint.replace("/api/chat", "/api/generate")
    llm_4b_ok = smoke_test_ollama(model_4b, generate_endpoint, timeout=30)
    print(f"LLM smoke test: {'PASS' if llm_4b_ok else 'FAIL'}")

    if vlm_4b_ok:
        rows = run_vlm_for_model(
            model=model_4b,
            image_ids=image_ids,
            modes=vlm_cfg["modes"],
            chat_endpoint=chat_endpoint,
            degraded_root=degraded_root,
            vlm_output_root=vlm_output_root,
            start_level=vlm_cfg["start_level"],
            num_levels=num_levels,
            sequence_history=vlm_cfg["sequence_history"],
            max_image_side=vlm_cfg["max_image_side"],
            timeout=vlm_cfg["request_timeout"],
            sat_threshold=vlm_cfg["saturation_threshold"],
        )
        vlm_stop_rows.extend(rows)

    if llm_4b_ok:
        rows = run_llm_ablation_configs(
            configs=configs_4b,
            image_ids=image_ids,
            chat_endpoint=chat_endpoint,
            pred_root=pred_root,
            ablation_root=ablation_root,
            start_level=llm_cfg["start_level"],
            num_levels=num_levels,
            sat_threshold=llm_cfg["saturation_threshold"],
            timeout=llm_cfg["request_timeout"],
            input_obj_threshold=llm_cfg["input_object_conf_threshold"],
            input_rel_threshold=llm_cfg["input_relation_conf_threshold"],
            input_top_k_objects=llm_cfg["input_top_k_objects"],
            input_top_k_relationships=llm_cfg["input_top_k_relationships"],
        )
        llm_stop_rows.extend(rows)

    # ── PHASE 2: qwen3.5:9b ──────────────────────────────────────
    model_9b = "qwen3.5:9b"
    print(f"\n{'='*70}")
    print(f"  PHASE 2: {model_9b}  (VLM + LLM ablation)")
    print(f"{'='*70}")

    print(f"\nSmoke-testing VLM ({model_9b}) ...")
    vlm_9b_ok = smoke_test_vlm(model_9b, chat_endpoint, timeout=120)
    print(f"VLM smoke test: {'PASS' if vlm_9b_ok else 'FAIL'}")

    print(f"Smoke-testing LLM ({model_9b}) ...")
    llm_9b_ok = smoke_test_ollama(model_9b, generate_endpoint, timeout=60)
    print(f"LLM smoke test: {'PASS' if llm_9b_ok else 'FAIL'}")

    if vlm_9b_ok:
        rows = run_vlm_for_model(
            model=model_9b,
            image_ids=image_ids,
            modes=vlm_cfg["modes"],
            chat_endpoint=chat_endpoint,
            degraded_root=degraded_root,
            vlm_output_root=vlm_output_root,
            start_level=vlm_cfg["start_level"],
            num_levels=num_levels,
            sequence_history=vlm_cfg["sequence_history"],
            max_image_side=vlm_cfg["max_image_side"],
            timeout=vlm_cfg["request_timeout"],
            sat_threshold=vlm_cfg["saturation_threshold"],
        )
        vlm_stop_rows.extend(rows)

    if llm_9b_ok:
        rows = run_llm_ablation_configs(
            configs=configs_9b,
            image_ids=image_ids,
            chat_endpoint=chat_endpoint,
            pred_root=pred_root,
            ablation_root=ablation_root,
            start_level=llm_cfg["start_level"],
            num_levels=num_levels,
            sat_threshold=llm_cfg["saturation_threshold"],
            timeout=llm_cfg["request_timeout"],
            input_obj_threshold=llm_cfg["input_object_conf_threshold"],
            input_rel_threshold=llm_cfg["input_relation_conf_threshold"],
            input_top_k_objects=llm_cfg["input_top_k_objects"],
            input_top_k_relationships=llm_cfg["input_top_k_relationships"],
        )
        llm_stop_rows.extend(rows)

    # ── Save CSVs ────────────────────────────────────────────────
    # VLM stop CSV
    vlm_csv = metrics_root / "vlm_stop.csv"
    with open(vlm_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "model", "mode", "k", "stop_level"])
        writer.writeheader()
        writer.writerows(vlm_stop_rows)
    print(f"\nSaved VLM stop CSV: {vlm_csv}")

    # LLM ablation stop CSV
    llm_csv = metrics_root / "llm_ablation_stop.csv"
    fieldnames = [
        "config_slug", "model", "history_length", "relation_on",
        "confidence_on", "image_id", "stop_level",
    ]
    with open(llm_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(llm_stop_rows)
    print(f"Saved LLM ablation stop CSV: {llm_csv}")

    # ── Print summary ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")

    print("\nVLM Stop Levels (mean):")
    for model in [model_4b, model_9b]:
        for mode in vlm_cfg["modes"]:
            for k in [1, 2]:
                subset = [
                    r["stop_level"] for r in vlm_stop_rows
                    if r["model"] == model and r["mode"] == mode and r["k"] == k
                ]
                if subset:
                    mean_val = sum(subset) / len(subset)
                    print(f"  {model} {mode} k={k}: mean={mean_val:.1f}")

    print("\nLLM Ablation Stop Levels (mean):")
    seen: list[str] = []
    for r in llm_stop_rows:
        s = r["config_slug"]
        if s not in seen:
            seen.append(s)
    for slug in seen:
        subset = [r["stop_level"] for r in llm_stop_rows if r["config_slug"] == slug and r["stop_level"] > 0]
        if subset:
            mean_val = sum(subset) / len(subset)
            print(f"  {slug}: mean={mean_val:.1f}")

    # Parse failure counts
    print("\nParse failure analysis:")
    vlm_parse_failures = 0
    vlm_total = 0
    for model in [model_4b, model_9b]:
        for mode in vlm_cfg["modes"]:
            slug_m = model.replace(":", "-").replace("/", "-")
            for img_id in image_ids:
                score_dir = vlm_output_root / slug_m / mode / str(img_id)
                if not score_dir.exists():
                    continue
                for f in sorted(score_dir.glob("level_*.json")):
                    vlm_total += 1
                    with open(f) as fh:
                        d = json.load(fh)
                    if "error" in d.get("reason", "") or "parse" in d.get("reason", ""):
                        vlm_parse_failures += 1
    print(f"  VLM: {vlm_parse_failures}/{vlm_total} parse/error failures")

    llm_parse_failures = 0
    llm_total = 0
    for slug in seen:
        for img_id in image_ids:
            score_dir = ablation_root / slug / str(img_id)
            if not score_dir.exists():
                continue
            for f in sorted(score_dir.glob("level_*.json")):
                llm_total += 1
                with open(f) as fh:
                    d = json.load(fh)
                if "error" in d.get("reason", "") or "parse" in d.get("reason", ""):
                    llm_parse_failures += 1
    print(f"  LLM ablation: {llm_parse_failures}/{llm_total} parse/error failures")


if __name__ == "__main__":
    main()
