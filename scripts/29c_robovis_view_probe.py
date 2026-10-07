"""ROBOVIS2027: image-only view-sufficiency probe (no checks, no graph).

One VLM call per (image, level) with ONLY the state JPEG and a fixed prompt, to
test whether the VLM can read resolution at all and whether more pixels help.

Reads   data/processed/robovis2027/degraded_images/<id>/level_XX.jpg
Writes  outputs/robovis2027/<run_id>/probe/<model_slug>_<size>/<id>/level_XX.json
        outputs/robovis2027/<run_id>/probe/<model_slug>_<size>/summary.json (--summarize)

--image-size 384 sends the stored 384-base JPEG bytes as-is; a larger size
upsamples it (bicubic) so max side == size and re-encodes it at
resolution.jpeg_quality before sending (more visual tokens for the VLM).

Run:
  uv run --project egtr python scripts/29c_robovis_view_probe.py \
      --run-id dev_r0 --split dev --model qwen3.5:9b --image-size 768 --levels all
  ... --summarize   (no model calls; aggregates existing records)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.check_generator import (  # noqa: E402
    load_cached,
    next_step_ratio,
    outcome_fields,
    write_json_atomic,
)
from milestone1.config import resolve  # noqa: E402
from milestone1.ollama_client import (  # noqa: E402
    OllamaClient,
    settings_from_config,
    sha256_json,
    sha256_text,
    structured_chat,
)
from milestone1.robovis_data import load_robovis_config  # noqa: E402

PERCENT = 100
DETAIL_MIN, DETAIL_MAX = 0, 100
ROUND_DIGITS = 4
PROBE_PROMPT_TEMPLATE = (
    "Rate how much of the fine detail of this scene is visible in this image. "
    'Answer JSON {{"detail_fraction": <0-100 integer, 100 = everything a sharp '
    'photo would show>, "sufficient": true|false (true if at least ~90 % of the '
    'detail is visible), "next_step_helps": true|false (whether an image about '
    '{pct} % sharper per side would reveal noticeably more), "reason": "<short>"}}'
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/robovis2027.yaml")
    p.add_argument("--run-id", required=True)
    p.add_argument("--split", default="dev")
    p.add_argument("--model", required=True)
    p.add_argument("--image-size", type=int, required=True, help="max side sent")
    p.add_argument("--levels", default="all", help="'all' or comma-separated")
    p.add_argument("--limit", type=int, default=None, help="max number of images")
    p.add_argument("--summarize", action="store_true", help="aggregate only")
    return p.parse_args()


def probe_prompt(cfg: dict[str, Any]) -> str:
    pct = round((next_step_ratio(cfg) - 1.0) * PERCENT)
    return PROBE_PROMPT_TEMPLATE.format(pct=pct)


def validate_probe(
    data: Any, max_chars: int
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(data, dict):
        return None, ["top level must be a JSON object"]
    errors: list[str] = []
    frac = data.get("detail_fraction")
    if (
        not isinstance(frac, int)
        or isinstance(frac, bool)
        or not DETAIL_MIN <= frac <= DETAIL_MAX
    ):
        errors.append(f"detail_fraction must be an integer 0-100: {frac!r}")
    for key in ("sufficient", "next_step_helps"):
        if not isinstance(data.get(key), bool):
            errors.append(f"{key} must be true/false: {data.get(key)!r}")
    reason = data.get("reason")
    if not isinstance(reason, str):
        errors.append("reason must be a string")
    elif len(reason) > max_chars:
        errors.append(f"reason longer than {max_chars} chars")
    if errors:
        return None, errors
    keys = ("detail_fraction", "sufficient", "next_step_helps", "reason")
    return {k: data[k] for k in keys}, []


def encode_for_probe(
    path: Path, image_size: int, jpeg_quality: int
) -> tuple[str, str, list[int]]:
    """(base64, sha256 of sent bytes, sent [w, h]); raw bytes if already that size."""
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as img:
        width, height = img.size
        if max(width, height) == image_size:
            return (
                base64.b64encode(raw).decode("ascii"),
                hashlib.sha256(raw).hexdigest(),
                [width, height],
            )
        factor = image_size / max(width, height)
        size = (round(width * factor), round(height * factor))
        resized = img.convert("RGB").resize(size, Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=jpeg_quality)
    data = buf.getvalue()
    return (
        base64.b64encode(data).decode("ascii"),
        hashlib.sha256(data).hexdigest(),
        list(size),
    )


def dev_image_ids(cfg: dict[str, Any], split: str) -> list[int]:
    with open(resolve(cfg["paths"]["selected_images"])) as f:
        rows = json.load(f)
    return sorted(int(r["image_id"]) for r in rows if r["split"] == split)


def parse_levels(text: str, n_levels: int) -> list[int]:
    if text == "all":
        return list(range(1, n_levels + 1))
    return [int(x) for x in text.split(",") if x]


def run_probe(args: argparse.Namespace, cfg: dict[str, Any], out_root: Path) -> None:
    settings = settings_from_config(cfg)
    client = OllamaClient(settings)
    schema_retries = int(cfg["execution"]["schema_retries"])
    max_chars = int(cfg["checks"]["max_string_chars"])
    jpeg_quality = int(cfg["resolution"]["jpeg_quality"])
    prompt = probe_prompt(cfg)
    prompt_sha = sha256_text(prompt)
    degraded = resolve(cfg["paths"]["degraded_images"])
    ids = dev_image_ids(cfg, args.split)[: args.limit]
    levels = parse_levels(args.levels, len(cfg["resolution"]["scales"]))
    print(
        f"{len(ids)} images x {len(levels)} levels | model={args.model} "
        f"size={args.image_size} options={settings.record()}",
        flush=True,
    )
    n_calls = n_fail = 0
    for image_id in ids:
        for level in levels:
            name = f"level_{level:02d}"
            src = degraded / str(image_id) / f"{name}.jpg"
            if not src.exists():
                print(f"  missing {src}; skip", flush=True)
                continue
            image_b64, sent_sha, sent_size = encode_for_probe(
                src, args.image_size, jpeg_quality
            )
            cache_key = sha256_json(
                {
                    "stage": "view_probe",
                    "model": args.model,
                    "prompt_sha256": prompt_sha,
                    "options": settings.record(),
                    "image_sha256": sent_sha,
                }
            )
            out_path = out_root / str(image_id) / f"{name}.json"
            if load_cached(out_path, cache_key) is not None:
                continue
            messages = [{"role": "user", "content": prompt, "images": [image_b64]}]
            outcome = structured_chat(
                client,
                args.model,
                messages,
                lambda data: validate_probe(data, max_chars),
                schema_retries,
            )
            record = {
                "model": args.model,
                "image_id": image_id,
                "level": level,
                "image_size": args.image_size,
                "sent_size": sent_size,
                "image_sha256": sent_sha,
                "prompt_sha256": prompt_sha,
                "options": settings.record(),
                "cache_key": cache_key,
                "probe": outcome.parsed,
                **outcome_fields(outcome),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            record["resample"] = (
                "as_stored"
                if max(sent_size) == args.image_size
                and sent_sha == hashlib.sha256(src.read_bytes()).hexdigest()
                else "bicubic_upsample"
            )
            write_json_atomic(out_path, record)
            n_calls += outcome.http_attempts
            n_fail += int(outcome.failure)
            p = outcome.parsed or {}
            print(
                f"  {image_id} L{level:02d} df={p.get('detail_fraction')} "
                f"suf={p.get('sufficient')} next={p.get('next_step_helps')} "
                f"att={outcome.attempts} {outcome.elapsed_s:.1f}s"
                f"{' FAIL ' + str(outcome.error) if outcome.failure else ''}",
                flush=True,
            )
    print(f"done: http_calls={n_calls} failures={n_fail}", flush=True)


def _mean(xs: list[float]) -> float | None:
    return round(statistics.fmean(xs), ROUND_DIGITS) if xs else None


def _median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), ROUND_DIGITS) if xs else None


def summarize(out_root: Path) -> dict[str, Any]:
    by_level: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(out_root.glob("*/level_*.json")):
        with open(path) as f:
            rec = json.load(f)
        by_level.setdefault(int(rec["level"]), []).append(rec)
    rows: dict[str, Any] = {}
    for level in sorted(by_level):
        recs = by_level[level]
        ok = [r["probe"] for r in recs if not r.get("failure") and r.get("probe")]
        called = [r for r in recs if r.get("attempts", 0) > 0]
        tokens = [
            u["prompt_eval_count"]
            for r in recs
            for u in (r.get("usage") or [])[:1]
            if "prompt_eval_count" in u
        ]
        rows[str(level)] = {
            "n": len(recs),
            "n_failures": sum(1 for r in recs if r.get("failure")),
            "mean_detail_fraction": _mean([p["detail_fraction"] for p in ok]),
            "frac_sufficient": _mean([float(p["sufficient"]) for p in ok]),
            "frac_next_step_helps": _mean([float(p["next_step_helps"]) for p in ok]),
            "median_latency_s": _median(
                [r["elapsed_s"] / r["attempts"] for r in called]
            ),
            "median_prompt_eval_count": _median(tokens),
        }
    summary = {"probe_dir": str(out_root), "by_level": rows}
    write_json_atomic(out_root / "summary.json", summary)
    print(f"{out_root.name}")
    print("level   n fail  detail  suff  next  lat_s  tokens")
    for level, r in rows.items():
        cells = [
            r["mean_detail_fraction"],
            r["frac_sufficient"],
            r["frac_next_step_helps"],
            r["median_latency_s"],
            r["median_prompt_eval_count"],
        ]
        txt = " ".join("     -" if v is None else f"{v:6.2f}" for v in cells)
        print(f"{level:>5} {r['n']:3} {r['n_failures']:4} {txt}")
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_robovis_config(resolve(args.config))
    slug = args.model.replace(":", "-").replace("/", "-")
    out_root = (
        resolve(cfg["output_root"])
        / args.run_id
        / "probe"
        / f"{slug}_{args.image_size}"
    )
    if not args.summarize:
        run_probe(args, cfg, out_root)
    summarize(out_root)


if __name__ == "__main__":
    main()
