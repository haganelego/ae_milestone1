"""ROBOVIS2027 judge v4 (bbox-grounded visibility): one VLM call per state.

Reads   outputs/robovis2027/<run_id>/<graphs_subdir>/<id>/level_XX.json   (G)
        data/processed/robovis2027/degraded_images/<id>/level_XX.jpg
        checks/<hyp_checks>/... and decisions/<hyp_decisions>/...   (reused H)
Writes  outputs/robovis2027/<run_id>/overlays/<id>/level_XX.jpg
        outputs/robovis2027/<run_id>/checks/<variant>/<id>/level_XX.json
        outputs/robovis2027/<run_id>/decisions/<variant>/<id>/level_XX.json

Phase D (t10 only, given images, then the frozen policy; no GT):
  uv run --project egtr python scripts/29d_robovis_grounded_v4.py --run-id <test_run> \
      --model qwen3.5:9b --levels 10 --image-ids-file <ids.json> --num-ctx 16384
  PYTHONPATH=src uv run --project egtr python -m milestone1.two_stage_policy \
      --mode apply --run-id <test_run> --image-ids-file <ids.json> \
      --out-csv outputs/robovis2027_recovery/<rec_run>/test_policy.csv
  (the cleaned graphs must exist under outputs/robovis2027/<test_run>/graphs_clean;
   hypotheses are reused only if --hyp-checks/--hyp-decisions exist for that run)

Run (smoke):
  uv run --project egtr python scripts/29d_robovis_grounded_v4.py --run-id dev_r0 \
      --model qwen3.5:9b --levels 1,5,10,15,20 --limit 5
  ... --summarize   (no model calls; per-level mean clear)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.check_generator import (  # noqa: E402
    PROMPT_V2,
    limits_from_config,
    load_cached,
    outcome_fields,
    render_graph,
    validate_checks,
    write_json_atomic,
)
from milestone1.config import resolve  # noqa: E402
from milestone1.grounded_judge import (  # noqa: E402
    CLEAR_ABSENT,
    CLEAR_GRADES,
    PROMPT_V4,
    build_v4_prompt,
    derive_observed,
    draw_overlay,
    overlay_style_from_config,
    reuse_hypotheses,
    top_entities,
    validate_v4,
)
from milestone1.judge import validate_judgment  # noqa: E402
from milestone1.ollama_client import (  # noqa: E402
    OllamaClient,
    settings_from_config,
    sha256_json,
    sha256_text,
    structured_chat,
)
from milestone1.robovis_data import load_robovis_config  # noqa: E402
from milestone1.two_stage_policy import read_image_ids  # noqa: E402

ROUND_DIGITS = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/robovis2027.yaml")
    p.add_argument("--run-id", required=True)
    p.add_argument("--split", default="dev")
    p.add_argument("--model", required=True)
    p.add_argument("--levels", default="1,5,10,15,20")
    p.add_argument("--limit", type=int, default=None, help="max number of images")
    p.add_argument(
        "--image-ids-file",
        default=None,
        help="JSON list / one id per line; overrides --split selection (Phase D)",
    )
    p.add_argument("--graphs-subdir", default="graphs_clean")
    p.add_argument(
        "--degraded-root",
        default=None,
        help="override paths.degraded_images (<root>/<id>/level_XX.jpg, 384-base)",
    )
    p.add_argument("--variant-name", default=None, help="default F1g_v4_G__<model>")
    p.add_argument("--hyp-checks", default="F1v_v2_G__9b")
    p.add_argument("--hyp-decisions", default="F1v_v3_G__9b")
    p.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="override checks.options.num_ctx (e.g. 16384 to share the ollama "
        "runner with other jobs); recorded in options / cache key",
    )
    p.add_argument("--summarize", action="store_true")
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def dev_ids(cfg: dict[str, Any], split: str) -> list[int]:
    with open(resolve(cfg["paths"]["selected_images"])) as f:
        rows = json.load(f)
    return sorted(int(r["image_id"]) for r in rows if r["split"] == split)


def run_state(
    *,
    graph: dict[str, Any],
    image_path: Path,
    name: str,
    image_id: int,
    run_root: Path,
    variant: str,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    client: OllamaClient,
) -> dict[str, Any]:
    limits = limits_from_config(cfg, PROMPT_V2)  # G renders identically under v2
    style = overlay_style_from_config(cfg)
    k = int(cfg["checks"]["v4"]["top_entities"])
    hint = int(cfg["checks"]["v2"]["string_hint_chars"])
    entities = top_entities(graph, k)
    prompt = build_v4_prompt(entities, graph["base_size"], limits.box_decimals, hint)
    overlay = draw_overlay(image_path.read_bytes(), entities, style)
    overlay_path = run_root / "overlays" / str(image_id) / name.replace(".json", ".jpg")
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_bytes(overlay)
    image_sha = hashlib.sha256(overlay).hexdigest()
    hyp_c = load_json(run_root / "checks" / args.hyp_checks / str(image_id) / name)
    hyp_d = load_json(
        run_root / "decisions" / args.hyp_decisions / str(image_id) / name
    )
    hyps, hyp_items, dropped = reuse_hypotheses(hyp_c, hyp_d)
    cache_key = sha256_json(
        {
            "stage": "judge_v4",
            "model": args.model,
            "prompt_sha256": sha256_text(prompt),
            "options": client.settings.record(),
            "image_sha256": image_sha,
            "hyp": sha256_json([hyps, hyp_items]),
        }
    )
    dec_path = run_root / "decisions" / variant / str(image_id) / name
    cached = load_cached(dec_path, cache_key)
    if cached is not None:
        return cached
    ids = [e["id"] for e in entities]
    max_chars = limits.max_string_chars
    if ids:
        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": [base64.b64encode(overlay).decode("ascii")],
            }
        ]
        outcome = structured_chat(
            client,
            args.model,
            messages,
            lambda d: validate_v4(d, ids, max_chars),
            int(cfg["execution"]["schema_retries"]),
        )
        objects = None if outcome.parsed is None else outcome.parsed["objects"]
        call = outcome_fields(outcome)
    else:
        objects, call = [], {"attempts": 0, "failure": False, "error": None}
    provenance = {
        "path": "F1",
        "model": args.model,
        "options": client.settings.record(),  # think/format/options incl. num_ctx
        "prompt_version": PROMPT_V4,
        "image_id": image_id,
        "level": graph.get("level"),
        "graphs_subdir": args.graphs_subdir,
        "prompt_sha256": sha256_text(prompt),
        "image_sha256": image_sha,
        "overlay": {**style.record(), "top_entities": k, "file": str(overlay_path)},
        "hypotheses_from": {"checks": args.hyp_checks, "decisions": args.hyp_decisions},
        "hypotheses_dropped_unjudged": dropped,
        "cache_key": cache_key,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    checks = items = None
    failure, error = bool(call.get("failure")), call.get("error")
    if objects is not None:
        issues, obs_items = derive_observed(
            objects, entities, limits.max_observed_issues
        )
        checks = {"observed_issues": issues, "hypotheses": hyps}
        items = obs_items + hyp_items
        rendered = render_graph(graph, limits)
        _, c_err = validate_checks(checks, rendered, limits)
        expected = [i["id"] for i in issues] + [h["id"] for h in hyps]
        _, j_err = validate_judgment({"items": items}, expected, limits)
        if c_err or j_err:  # derived records must satisfy the plan contract
            failure, error = True, "derived contract: " + "; ".join((c_err + j_err)[:5])
            checks = items = None
    checks_rec = {
        **provenance,
        "checks": checks,
        "failure": failure,
        "error": error,
        "derived_from": "v4_objects",
    }
    dec_rec = {
        **provenance,
        "judge_saw_image": True,
        "skipped_empty": not ids and not hyps,
        "items": items,
        "v4_objects": objects,
        **call,
        "failure": failure,
        "error": error,
    }
    write_json_atomic(run_root / "checks" / variant / str(image_id) / name, checks_rec)
    write_json_atomic(dec_path, dec_rec)
    return dec_rec


def summarize(run_root: Path, variant: str) -> None:
    """Per-level rating distribution (0/1/2/absent) + mean of numeric ratings."""
    rows: dict[int, list[dict[str, Any]]] = {}
    for path in sorted((run_root / "decisions" / variant).glob("*/level_*.json")):
        rec = load_json(path) or {}
        rows.setdefault(int(rec.get("level", 0)), []).append(rec)
    out: dict[str, Any] = {}
    print(f"{variant}\nlevel  n fail mean_num  frac0  frac1  frac2 absent  median_s")
    for level in sorted(rows):
        recs = rows[level]
        ratings = [o["clear"] for r in recs for o in (r.get("v4_objects") or [])]
        numeric = [c for c in ratings if c != CLEAR_ABSENT]
        lat = [r["elapsed_s"] / r["attempts"] for r in recs if r.get("attempts")]
        n_r = len(ratings) or 1
        dist = {
            str(g): round(sum(c == g for c in ratings) / n_r, ROUND_DIGITS)
            for g in CLEAR_GRADES
        }
        mean = round(statistics.fmean(numeric), ROUND_DIGITS) if numeric else None
        out[str(level)] = {
            "n_states": len(recs),
            "n_failures": sum(bool(r.get("failure")) for r in recs),
            "n_ratings": len(ratings),
            "rating_fractions": dist,
            "mean_numeric_clear": mean,
            "median_latency_s": round(statistics.median(lat), ROUND_DIGITS)
            if lat
            else None,
        }
        cells = [dist[str(g)] for g in CLEAR_GRADES]
        print(
            f"{level:5} {len(recs):2} {out[str(level)]['n_failures']:4} "
            f"{'   -' if mean is None else f'{mean:8.3f}'} "
            + " ".join(f"{c:6.2f}" for c in cells)
            + f" {statistics.median(lat) if lat else 0:9.1f}"
        )
    summary_path = run_root / "summaries" / f"v4_summary_{variant}.json"
    write_json_atomic(summary_path, {"variant": variant, "by_level": out})
    print(f"-> {summary_path}")


def main() -> None:
    args = parse_args()
    cfg = load_robovis_config(resolve(args.config))
    if args.num_ctx is not None:
        cfg["checks"]["options"]["num_ctx"] = args.num_ctx
    slug = args.model.split(":")[-1]
    variant = args.variant_name or f"F1g_v4_G__{slug}"
    run_root = resolve(cfg["output_root"]) / args.run_id
    if not args.summarize:
        client = OllamaClient(settings_from_config(cfg))
        degraded = resolve(args.degraded_root or cfg["paths"]["degraded_images"])
        levels = [int(x) for x in args.levels.split(",") if x]
        ids = (
            read_image_ids(resolve(args.image_ids_file))
            if args.image_ids_file
            else dev_ids(cfg, args.split)
        )
        for image_id in ids[: args.limit]:
            for level in levels:
                name = f"level_{level:02d}.json"
                graph = load_json(run_root / args.graphs_subdir / str(image_id) / name)
                image_path = degraded / str(image_id) / name.replace(".json", ".jpg")
                if graph is None or not image_path.exists():
                    print(f"  {image_id} L{level:02d} missing input; skip", flush=True)
                    continue
                rec = run_state(
                    graph=graph,
                    image_path=image_path,
                    name=name,
                    image_id=image_id,
                    run_root=run_root,
                    variant=variant,
                    args=args,
                    cfg=cfg,
                    client=client,
                )
                clears = [o["clear"] for o in (rec.get("v4_objects") or [])]
                print(
                    f"  {image_id} L{level:02d} clear={clears} att={rec.get('attempts')}"
                    f"{' FAIL ' + str(rec['error']) if rec.get('failure') else ''}",
                    flush=True,
                )
    summarize(run_root, variant)


if __name__ == "__main__":
    main()
