"""MIRU follow-up: saturation scoring from the 5-level EGTR SGG history, with or
without the CURRENT level image attached (scaled-down 2x2: {text, img_single} x
{qwen3.5:4b, qwen3.5:9b}).

Identical to scripts/09 except for the transport: same prompt
(milestone1.llm.SATURATION_PROMPT), same sequence (09 build_sequence ==
llm_engine.build_sequence with relations + confidences, history_length,
input gates/top-k from configs/config.yaml `llm`), same JSON contract.
Condition ``img_single`` adds ONE sentence (IMAGE_NOTE) and attaches the current
level's degraded image preprocessed exactly like scripts/11 single mode
(milestone1.vlm._load_and_downscale_image: longest side <= vlm.max_image_side,
JPEG q85). Both conditions run through milestone1.ollama_client with the
`vlm_sgg` settings (JSON mode, think false, temperature 0, fixed seed, one
schema retry at temperature 0.3, transient retries) -- unlike llm.py, a failed
call is NEVER converted into a 0.0 score.

Layout (same per-level JSON layout as outputs/llm_ablation/<slug>/, so the MIRU
loaders `load_saturation_scores(base_dir, image_id, ...)` read it unchanged):
  outputs/llm_scores_vlm/<slug>/<image_id>/level_XX.json
      {"saturation_score", "stop_decision", "reason", "raw": {...provenance}}
  outputs/llm_scores_vlm/<slug>/_failures/<image_id>/level_XX.json  (no score file)
  outputs/llm_scores_vlm/<slug>/usage.jsonl, manifest.json
  slug = <model>_h<history>_rel_on_conf_on_<condition>, e.g.
         qwen3.5-9b_h5_rel_on_conf_on_img_single, qwen3.5-4b_h5_rel_on_conf_on_text

Usage:
  uv run --project egtr python scripts/40_vlm_sgg_saturation_scoring.py \
      --plan qwen3.5:9b=img_single,qwen3.5:4b=img_single,qwen3.5:9b=text,qwen3.5:4b=text
  ... --limit 2 --levels 5,12,20      # smoke
  ... --summarize                     # stop metrics of finished slugs (no calls)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import load_config, resolve  # noqa: E402
from milestone1.llm import SATURATION_PROMPT  # noqa: E402
from milestone1.llm_engine import build_sequence, make_config_slug  # noqa: E402
from milestone1.ollama_client import (  # noqa: E402
    OllamaClient,
    OllamaSettings,
    sha256_json,
    sha256_text,
    structured_chat,
)
from milestone1.vlm import _load_and_downscale_image  # noqa: E402

CONDITIONS = ("text", "img_single")
IMG_CONDITION = "img_single"
SCENE_MARKER = "Scene graph sequence:\n"
IMAGE_NOTE = (
    "An image of the scene at the CURRENT resolution level (the last entry of the "
    "sequence) is attached. Use it together with the scene graphs.\n\n"
)
STOP_DECISION_THRESHOLD = 0.8  # as stated in the prompt's JSON contract
SCRIPT11_JPEG_QUALITY = 85  # milestone1.vlm._load_and_downscale_image
STOP_KS = (1, 2)
FAILURES_DIR = "_failures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--plan",
        default="qwen3.5:9b=img_single,qwen3.5:4b=img_single,qwen3.5:9b=text,qwen3.5:4b=text",
        help="ordered comma list of model=condition",
    )
    parser.add_argument("--limit", type=int, default=None, help="first N images")
    parser.add_argument(
        "--levels", default=None, help="comma list (default start_level..20)"
    )
    parser.add_argument(
        "--summarize", action="store_true", help="no calls; print stop metrics"
    )
    return parser.parse_args()


def build_messages(
    sequence: list[dict[str, Any]], image_b64: str | None
) -> list[dict[str, Any]]:
    """Exactly scripts/09's user content, plus IMAGE_NOTE + image for img_single."""
    prompt = SATURATION_PROMPT
    if image_b64 is not None:
        if prompt.count(SCENE_MARKER) != 1:
            raise RuntimeError("prompt layout changed: cannot place IMAGE_NOTE")
        prompt = prompt.replace(SCENE_MARKER, IMAGE_NOTE + SCENE_MARKER)
    message: dict[str, Any] = {
        "role": "user",
        "content": prompt + json.dumps(sequence, indent=2),
    }
    if image_b64 is not None:
        message["images"] = [image_b64]
    return [message]


def validate_score(data: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """MIRU contract: saturation_score in [0,1] (clamped), stop_decision, reason."""
    if not isinstance(data, dict):
        return None, ["not a JSON object"]
    try:
        score = float(data["saturation_score"])
    except (KeyError, TypeError, ValueError):
        return None, ["saturation_score missing or not numeric"]
    if not math.isfinite(score):
        return None, ["saturation_score not finite"]
    clamped = min(1.0, max(0.0, score))
    stop = data.get("stop_decision")
    if isinstance(stop, str):
        stop = stop.strip().lower() == "true"
    if not isinstance(stop, bool):
        stop = clamped >= STOP_DECISION_THRESHOLD
    reason = data.get("reason")
    return {
        "saturation_score": clamped,
        "stop_decision": bool(stop),
        "reason": reason if isinstance(reason, str) else "",
        "score_clamped": clamped != score,
    }, []


def image_payload(path: Path, max_side: int) -> tuple[str, dict[str, Any]]:
    b64 = _load_and_downscale_image(str(path), max_side)
    raw = base64.b64decode(b64)
    with Image.open(path) as src, Image.open(io.BytesIO(raw)) as sent:
        info = {
            "path": str(path.relative_to(REPO_ROOT)),
            "stored_size": list(src.size),
            "sent_size": list(sent.size),
            "sent_format": "JPEG",
            "sent_quality": SCRIPT11_JPEG_QUALITY,
            "max_image_side": max_side,
            "sent_sha256": sha256_text(b64),
        }
    return b64, info


def settings(cfg: dict[str, Any]) -> tuple[OllamaSettings, int]:
    v = cfg["vlm_sgg"]
    return OllamaSettings(
        endpoint=cfg["ollama_chat_endpoint"],
        think=bool(v["think"]),
        json_mode=bool(v["json_mode"]),
        options=dict(v["options"]),
        timeout_s=float(v["request_timeout_s"]),
        transient_retries=int(v["transient_retries"]),
        backoff_base_s=float(v["backoff_base_s"]),
        backoff_max_s=float(v["backoff_max_s"]),
        retry_temperature=float(v["retry_temperature"]),
    ), int(v["schema_retries"])


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def run_slug(
    cfg: dict[str, Any],
    model: str,
    condition: str,
    image_ids: list[int],
    levels: list[int],
) -> None:
    llm = cfg["llm"]
    history = int(llm["history_length"])
    slug = f"{make_config_slug(model, history, True, True)}_{condition}"
    out_root = resolve(cfg["vlm_sgg"]["output_root"]) / slug
    pred_root = resolve(cfg["paths"]["egtr_predictions"])
    img_root = resolve(cfg["paths"]["degraded_images"])
    max_side = int(cfg["vlm"]["max_image_side"])
    st, schema_retries = settings(cfg)
    client = OllamaClient(st)
    manifest_path = out_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(
        {
            "slug": slug,
            "model": model,
            "condition": condition,
            "script": "scripts/40_vlm_sgg_saturation_scoring.py",
            "git_head": git_head(),
            "settings": st.record(),
            "schema_retries": schema_retries,
            "retry_temperature": st.retry_temperature,
            "prompt_sha256_text": sha256_text(SATURATION_PROMPT),
            "image_note": IMAGE_NOTE if condition == IMG_CONDITION else None,
            "sequence": {
                "history_length": history,
                "relation_on": True,
                "confidence_on": True,
                "input_object_conf_threshold": llm["input_object_conf_threshold"],
                "input_relation_conf_threshold": llm["input_relation_conf_threshold"],
                "input_top_k_objects": llm["input_top_k_objects"],
                "input_top_k_relationships": llm["input_top_k_relationships"],
            },
            "image_preprocessing": (
                "scripts/11 single mode: vlm._load_and_downscale_image(max_side="
                f"{max_side}), JPEG q{SCRIPT11_JPEG_QUALITY}"
            )
            if condition == IMG_CONDITION
            else None,
        }
    )
    manifest.setdefault("created", datetime.now().isoformat(timespec="seconds"))
    write_json(manifest_path, manifest)
    n_new = n_fail = n_cache = 0
    t_start = time.perf_counter()
    for image_id in image_ids:
        for level in levels:
            name = f"level_{level:02d}.json"
            out_path = out_root / str(image_id) / name
            sequence = build_sequence(
                image_id,
                level,
                history,
                pred_root,
                True,
                True,
                llm["input_object_conf_threshold"],
                llm["input_relation_conf_threshold"],
                llm["input_top_k_objects"],
                llm["input_top_k_relationships"],
            )
            image_b64, image_info = (None, None)
            if condition == IMG_CONDITION:
                image_b64, image_info = image_payload(
                    img_root / str(image_id) / f"level_{level:02d}.jpg", max_side
                )
            messages = build_messages(sequence, image_b64)
            cache_key = sha256_json(
                {
                    "model": model,
                    "condition": condition,
                    "content": messages[0]["content"],
                    "image": image_info["sent_sha256"] if image_info else None,
                    "settings": st.record(),
                    "schema_retries": schema_retries,
                }
            )
            if out_path.exists():
                cached = json.loads(out_path.read_text())
                if cached.get("raw", {}).get("cache_key") != cache_key:
                    raise RuntimeError(
                        f"{out_path}: cached result has a different cache key (inputs/settings changed)"
                    )
                n_cache += 1
                continue
            outcome = structured_chat(
                client, model, messages, validate_score, schema_retries
            )
            raw = {
                "model": model,
                "condition": condition,
                "image_id": image_id,
                "level": level,
                "cache_key": cache_key,
                "sequence_sha256": sha256_json(sequence),
                "sequence_levels": [e["level"] for e in sequence],
                "image": image_info,
                "attempts": outcome.attempts,
                "seeds": outcome.seeds,
                "seed_used": outcome.seed_used,
                "temperatures": outcome.temperatures,
                "temperature_used": outcome.temperature_used,
                "http_attempts": outcome.http_attempts,
                "elapsed_s": round(outcome.elapsed_s, 3),
                "usage": outcome.usage,
                "raw_responses": outcome.raw_responses,
                "attempt_errors": outcome.attempt_errors,
                "error": outcome.error,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            with open(out_root / "usage.jsonl", "a") as f:
                f.write(
                    json.dumps(
                        {
                            "image_id": image_id,
                            "level": level,
                            "failure": outcome.failure,
                            "elapsed_s": raw["elapsed_s"],
                            "attempts": outcome.attempts,
                            "usage": outcome.usage,
                        }
                    )
                    + "\n"
                )
            n_new += 1
            if outcome.failure:
                n_fail += 1
                write_json(
                    out_root / FAILURES_DIR / str(image_id) / name,
                    {"failure": True, "raw": raw},
                )
                print(
                    f"  FAIL {slug} {image_id} L{level:02d}: {outcome.error}",
                    flush=True,
                )
                continue
            parsed = outcome.parsed
            write_json(
                out_path,
                {
                    "saturation_score": parsed["saturation_score"],
                    "stop_decision": parsed["stop_decision"],
                    "reason": parsed["reason"],
                    "raw": {**raw, "score_clamped": parsed["score_clamped"]},
                },
            )
            print(
                f"  {slug} {image_id} L{level:02d}: sat={parsed['saturation_score']:.2f} ({raw['elapsed_s']:.1f}s)",
                flush=True,
            )
    wall = time.perf_counter() - t_start
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("runs", []).append(
        {
            "finished": datetime.now().isoformat(timespec="seconds"),
            "n_new": n_new,
            "n_failures": n_fail,
            "n_cached": n_cache,
            "wall_s": round(wall, 1),
            "s_per_call": round(wall / n_new, 2) if n_new else None,
        }
    )
    write_json(manifest_path, manifest)
    print(
        f"{slug}: new {n_new} (failures {n_fail}), cached {n_cache}, {wall:.0f}s",
        flush=True,
    )


def consecutive_k_stop(
    scores: dict[int, float], threshold: float, k: int, start: int, max_level: int
) -> int:
    """Same rule as scripts/13 (missing level = below threshold)."""
    for lvl in range(start, max_level + 1):
        if all(
            lvl - o >= start and scores.get(lvl - o, 0.0) >= threshold for o in range(k)
        ):
            return lvl
    return max_level


def summarize(cfg: dict[str, Any], image_ids: list[int]) -> None:
    """Quick MIRU-style stop metrics (MAE vs oracle_0.5 and stop_c09); no calls."""
    import csv

    start, max_level = (
        int(cfg["llm"]["start_level"]),
        int(cfg["resolution_levels"]["num_levels"]),
    )
    thr = float(cfg["llm"]["saturation_threshold"])
    metrics = resolve(cfg["paths"]["metrics"])
    oracle = {
        int(r["image_id"]): int(r["oracle_0.5"])
        for r in csv.DictReader(open(metrics / "oracle_stop.csv"))
    }
    c09 = {
        int(r["image_id"]): int(r["stop_c09"])
        for r in csv.DictReader(open(metrics / "topk_oracle/oracle_topk_per_image.csv"))
    }
    roots = [
        ("legacy", resolve("outputs/llm_ablation")),
        ("new", resolve(cfg["vlm_sgg"]["output_root"])),
    ]
    for tag, root in roots:
        for slug_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if tag == "legacy" and not slug_dir.name.endswith("_h5_rel_on_conf_on"):
                continue
            n_scores = sum(1 for _ in slug_dir.glob("*/level_*.json"))
            for k in STOP_KS:
                stops = {}
                for i in image_ids:
                    scores = {}
                    for lvl in range(start, max_level + 1):
                        p = slug_dir / str(i) / f"level_{lvl:02d}.json"
                        if p.exists():
                            scores[lvl] = float(
                                json.loads(p.read_text())["saturation_score"]
                            )
                    stops[i] = consecutive_k_stop(scores, thr, k, start, max_level)
                mae_o = sum(abs(stops[i] - oracle[i]) for i in image_ids) / len(
                    image_ids
                )
                mae_c = sum(abs(stops[i] - c09[i]) for i in image_ids) / len(image_ids)
                mean = sum(stops.values()) / len(stops)
                print(
                    f"{tag:6s} {slug_dir.name:45s} k={k} scores={n_scores:4d} mean_stop={mean:5.2f} "
                    f"MAE_oracle0.5={mae_o:5.2f} MAE_stop_c09={mae_c:5.2f}"
                )


def main() -> None:
    args = parse_args()
    cfg = load_config()
    image_ids = json.loads(resolve(cfg["paths"]["selected_images"]).read_text())
    image_ids = image_ids[: args.limit] if args.limit else image_ids
    if args.summarize:
        summarize(cfg, image_ids)
        return
    start, max_level = (
        int(cfg["llm"]["start_level"]),
        int(cfg["resolution_levels"]["num_levels"]),
    )
    levels = (
        [int(x) for x in args.levels.split(",")]
        if args.levels
        else list(range(start, max_level + 1))
    )
    for item in [p.strip() for p in args.plan.split(",") if p.strip()]:
        model, condition = item.split("=")
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition}")
        print(
            f"=== {model} / {condition}: {len(image_ids)} images x {len(levels)} levels",
            flush=True,
        )
        run_slug(cfg, model, condition, image_ids, levels)


if __name__ == "__main__":
    main()
