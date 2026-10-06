"""ROBOVIS2027: generate F1/F2 checks (and optional judgments) for cached graphs.

Reads   outputs/robovis2027/<run_id>/graphs/<image_id>/level_XX.json
Writes  outputs/robovis2027/<run_id>/checks/<PATH>[__tag]/<image_id>/level_XX.json
        outputs/robovis2027/<run_id>/decisions/<PATH>[__tag]/<image_id>/level_XX.json
        outputs/robovis2027/<run_id>/usage.jsonl

F1 sends the state JPEG data/processed/robovis2027/degraded_images/<id>/level_XX.jpg
as stored (384-base). The prompt never contains image id / step / scale.

Run:
  uv run --project egtr python scripts/29_robovis_checks.py \
      --config configs/robovis2027.yaml --run-id dev_r0 --split dev --path F2 --judge
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.check_generator import (  # noqa: E402
    PATH_F1,
    VALID_PATHS,
    generate_checks,
    limits_from_config,
)
from milestone1.config import resolve  # noqa: E402
from milestone1.judge import judge_checks  # noqa: E402
from milestone1.ollama_client import OllamaClient, settings_from_config  # noqa: E402
from milestone1.robovis_data import load_robovis_config  # noqa: E402

LEVEL_FILE_RE = re.compile(r"^level_(\d+)\.json$")


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.replace(" ", "").split(",") if x]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/robovis2027.yaml")
    p.add_argument("--run-id", required=True)
    p.add_argument("--split", default="dev")
    p.add_argument("--path", choices=VALID_PATHS, required=True)
    p.add_argument(
        "--levels",
        type=parse_int_list,
        default=None,
        help="comma-separated, e.g. 1,5,10,15,20",
    )
    p.add_argument("--image-ids", type=parse_int_list, default=None)
    p.add_argument("--limit", type=int, default=None, help="max number of images")
    p.add_argument("--judge", action="store_true", help="also run the judgment step")
    p.add_argument("--check-model", default=None, help="override checks.check_model")
    p.add_argument("--judge-model", default=None, help="override checks.judge_model")
    p.add_argument("--json-mode", choices=("on", "off"), default=None)
    p.add_argument("--out-tag", default="", help="suffix for checks/decisions dirs")
    return p.parse_args()


def iter_states(
    graphs_root: Path,
    split: str,
    levels: list[int] | None,
    image_ids: list[int] | None,
    limit: int | None,
) -> list[tuple[int, int, Path]]:
    """(image_id, level, graph_path) for graphs of the requested split."""
    states: list[tuple[int, int, Path]] = []
    image_dirs = sorted(
        (d for d in graphs_root.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    n_images = 0
    for image_dir in image_dirs:
        image_id = int(image_dir.name)
        if image_ids is not None and image_id not in image_ids:
            continue
        found = []
        for graph_path in sorted(image_dir.glob("level_*.json")):
            match = LEVEL_FILE_RE.match(graph_path.name)
            if match is None:
                continue
            level = int(match.group(1))
            if levels is not None and level not in levels:
                continue
            found.append((image_id, level, graph_path))
        if not found:
            continue
        with open(found[0][2]) as f:
            if json.load(f).get("split", split) != split:
                continue
        if limit is not None and n_images >= limit:
            break
        n_images += 1
        states.extend(found)
    return states


def append_usage(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def usage_row(
    stage: str,
    path: str,
    image_id: int,
    level: int,
    record: dict[str, Any],
    cached: bool,
) -> dict[str, Any]:
    usage = record.get("usage") or []
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage": stage,
        "path": path,
        "model": record.get("model"),
        "image_id": image_id,
        "level": level,
        "cached": cached,
        "skipped_empty": record.get("skipped_empty", False),
        "attempts": 0 if cached else record.get("attempts", 0),
        "http_attempts": 0 if cached else record.get("http_attempts", 0),
        "elapsed_s": 0.0 if cached else record.get("elapsed_s", 0.0),
        "prompt_eval_count": sum(u.get("prompt_eval_count", 0) for u in usage),
        "eval_count": sum(u.get("eval_count", 0) for u in usage),
        "failure": record.get("failure", False),
        "error": record.get("error"),
    }


def main() -> None:
    args = parse_args()
    cfg = load_robovis_config(resolve(args.config))
    if args.json_mode is not None:
        cfg["checks"]["json_mode"] = args.json_mode == "on"
    settings = settings_from_config(cfg)
    client = OllamaClient(settings)
    limits = limits_from_config(cfg)
    schema_retries = int(cfg["execution"]["schema_retries"])
    max_calls = int(cfg["execution"]["max_model_calls"])
    check_model = args.check_model or cfg["checks"]["check_model"]
    judge_model = args.judge_model or cfg["checks"]["judge_model"]

    run_root = resolve(cfg["output_root"]) / args.run_id
    graphs_root = run_root / "graphs"
    if not graphs_root.is_dir():
        sys.exit(f"no graphs under {graphs_root}")
    variant = args.path + (f"__{args.out_tag}" if args.out_tag else "")
    checks_root = run_root / "checks" / variant
    decisions_root = run_root / "decisions" / variant
    usage_path = run_root / "usage.jsonl"
    degraded_root = resolve(cfg["paths"]["degraded_images"])

    states = iter_states(
        graphs_root, args.split, args.levels, args.image_ids, args.limit
    )
    print(
        f"{len(states)} states | path={args.path} check_model={check_model} "
        f"judge={'on:' + judge_model if args.judge else 'off'} "
        f"options={settings.record()}"
    )
    n_calls = 0
    n_fail = {"checks": 0, "judge": 0}
    for image_id, level, graph_path in states:
        if n_calls >= max_calls:
            print(f"model-call budget {max_calls} reached; stopping")
            break
        with open(graph_path) as f:
            graph = json.load(f)
        name = graph_path.name
        image_path = None
        if args.path == PATH_F1:
            image_path = degraded_root / str(image_id) / name.replace(".json", ".jpg")
            if not image_path.exists():
                print(f"  missing image {image_path}; skip")
                continue
        rec, cached = generate_checks(
            graph,
            client=client,
            model=check_model,
            path=args.path,
            limits=limits,
            schema_retries=schema_retries,
            out_path=checks_root / str(image_id) / name,
            image_path=image_path,
        )
        n_calls += 0 if cached else rec["http_attempts"]
        n_fail["checks"] += int(rec["failure"])
        append_usage(
            usage_path, usage_row("checks", variant, image_id, level, rec, cached)
        )
        n_items = "-"
        if rec["checks"] is not None:
            n_items = (
                f"{len(rec['checks']['observed_issues'])}o/"
                f"{len(rec['checks']['hypotheses'])}h"
            )
        print(
            f"  {image_id} L{level:02d} checks {n_items} att={rec['attempts']} "
            f"{rec['elapsed_s']:.1f}s{' [cache]' if cached else ''}"
            f"{' FAIL ' + str(rec['error']) if rec['failure'] else ''}"
        )
        if not args.judge:
            continue
        jrec, jcached = judge_checks(
            graph,
            rec,
            client=client,
            model=judge_model,
            limits=limits,
            schema_retries=schema_retries,
            out_path=decisions_root / str(image_id) / name,
        )
        n_calls += 0 if jcached else jrec.get("http_attempts", 0)
        n_fail["judge"] += int(jrec["failure"])
        append_usage(
            usage_path, usage_row("judge", variant, image_id, level, jrec, jcached)
        )
        print(
            f"  {image_id} L{level:02d} judge items="
            f"{'-' if jrec['items'] is None else len(jrec['items'])} "
            f"att={jrec.get('attempts', 0)} {jrec.get('elapsed_s', 0.0):.1f}s"
            f"{' [skip-empty]' if jrec.get('skipped_empty') else ''}"
            f"{' [cache]' if jcached else ''}"
            f"{' FAIL ' + str(jrec['error']) if jrec['failure'] else ''}"
        )
    print(f"done: http_calls={n_calls} failures={n_fail}")


if __name__ == "__main__":
    main()
