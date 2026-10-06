"""ROBOVIS2027: run PGSG on degraded images and cache graph JSONs.

Writes outputs/robovis2027/<run_id>/graphs/<image_id>/level_XX.json (contract in
.claude/docs/research/robovis2027-worklog.md) plus manifest.json (resolved
config, checkpoint sha256, code git hash, per-state seeds/batches, timing).
Existing graph files are skipped (cache).

Batching / determinism: a generation batch is up to --batch-size images of the
split (in selected_images.json order) at ONE level; image chunks are fixed by
split order, so a state's output depends only on (chunk, level, base seed) and
not on --levels. The torch seed of a batch is
``state_seed(chunk_image_ids, level, base_seed)`` (sha256-based). ``--limit``
reproduces full-run outputs only when it is a multiple of --batch-size.

--seed-offset N (N > 0) repeats the run with base seed + N into graphs_seed<N>/
(manifest_seed<N>.json) for multi-seed q_t smoothing; graphs/ is never touched.

Usage (from repo root):
  PYTHONPATH=Pix2Grp_CVPR2024 PYTHONDONTWRITEBYTECODE=1 uv run --project envs/pgsg \
    python scripts/25_pgsg_infer.py --config configs/robovis2027.yaml \
    --run-id dev_r0 --split dev [--levels 1,5,10] [--limit N] [--level-20-only]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1 import pgsg_adapter as pa  # noqa: E402
from milestone1.config import resolve  # noqa: E402
from milestone1.robovis_data import (  # noqa: E402
    level_name,
    load_robovis_config,
    sha256_file,
)

ALLOWED_OUTPUT_ROOT = REPO_ROOT / "outputs" / "robovis2027"
CATEGORIES_PATH = REPO_ROOT / "data/processed/pgsg_vg/categories_dict.json"
NUM_LEVELS = 20
DEFAULT_BATCH_SIZE = 4
SMOKE_LEVEL = 20
CODE_FILES = (
    REPO_ROOT / "src/milestone1/pgsg_adapter.py",
    REPO_ROOT / "scripts/25_pgsg_infer.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/robovis2027.yaml"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--levels", type=str, default=None, help="comma list, 1-based")
    parser.add_argument(
        "--limit", type=int, default=None, help="first N images of the split"
    )
    parser.add_argument("--level-20-only", action="store_true", help="smoke mode")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--base-seed", type=int, default=None, help="default: eval yaml run.seed"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="N > 0: base seed + N, graphs -> <run>/graphs_seed<N>/, "
        "manifest -> manifest_seed<N>.json (graphs/ untouched)",
    )
    return parser.parse_args()


def git_info() -> dict[str, Any]:
    def run(*cmd: str) -> str:
        return subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "git_head": run("git", "rev-parse", "HEAD"),
        "git_dirty_files": [
            ln[3:] for ln in run("git", "status", "--porcelain").splitlines()
        ],
        "pgsg_submodule_head": run(
            "git", "-C", "Pix2Grp_CVPR2024", "rev-parse", "HEAD"
        ),
        "file_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in CODE_FILES
        },
    }


def graphs_subdir(seed_offset: int) -> str:
    """graphs/ for the base seed, graphs_seed<N>/ for offset N > 0."""
    if seed_offset < 0:
        raise ValueError("--seed-offset must be >= 0")
    return "graphs" if seed_offset == 0 else f"graphs_seed{seed_offset}"


def manifest_name(seed_offset: int) -> str:
    return "manifest.json" if seed_offset == 0 else f"manifest_seed{seed_offset}.json"


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def select_levels(args: argparse.Namespace) -> list[int]:
    if args.level_20_only:
        return [SMOKE_LEVEL]
    if args.levels is None:
        return list(range(1, NUM_LEVELS + 1))
    levels = sorted({int(x) for x in args.levels.split(",") if x.strip()})
    if not all(1 <= lv <= NUM_LEVELS for lv in levels):
        raise ValueError(f"levels must be in 1..{NUM_LEVELS}: {levels}")
    return levels


def load_split_images(
    cfg: dict[str, Any], split: str, limit: int | None
) -> list[dict[str, Any]]:
    rows = json.loads(resolve(cfg["paths"]["selected_images"]).read_text())
    rows = sorted((r for r in rows if r["split"] == split), key=lambda r: r["rank"])
    return rows[:limit] if limit is not None else rows


def state_meta(
    geo: dict[str, Any], split: str, level: int, scale: float, seed: int
) -> pa.StateMeta:
    return pa.StateMeta(
        image_id=int(geo["image_id"]),
        split=split,
        level=level,
        scale=scale,
        base_size=(int(geo["base_width"]), int(geo["base_height"])),
        orig_size=(int(geo["orig_width"]), int(geo["orig_height"])),
        coord_scale=(float(geo["coord_scale_x"]), float(geo["coord_scale_y"])),
        seed=seed,
    )


class Runner:
    """Holds the model and the per-run constants."""

    def __init__(
        self, args: argparse.Namespace, cfg: dict[str, Any], out_dir: Path
    ) -> None:
        self.args = args
        self.out_dir = out_dir
        self.scales = [float(s) for s in cfg["resolution"]["scales"]]
        levels_manifest = json.loads(
            resolve(cfg["paths"]["levels_manifest"]).read_text()
        )
        self.geo = {int(g["image_id"]): g for g in levels_manifest["images"]}
        self.img_root = resolve(cfg["paths"]["degraded_images"])
        cats = json.loads(CATEGORIES_PATH.read_text())
        self.obj_names, self.rel_names = cats["obj"], cats["rel"]
        ckpt = resolve(cfg["pgsg"]["checkpoint"])
        print(f"hashing checkpoint {ckpt} ...", flush=True)
        self.model_info = {"checkpoint": ckpt.name, "sha256": sha256_file(ckpt)}
        self.eval_config = resolve(cfg["pgsg"]["eval_config"])
        self.rt = pa.load_pgsg(self.eval_config, device=args.device)
        self.base_seed = (
            args.base_seed if args.base_seed is not None else int(self.rt.run_cfg.seed)
        ) + args.seed_offset
        self.graphs_subdir = graphs_subdir(args.seed_offset)
        self.generation = pa.graph_generation_block(self.rt.resolved)

    def graph_path(self, image_id: int, level: int) -> Path:
        return (
            self.out_dir
            / self.graphs_subdir
            / str(image_id)
            / f"{level_name(level)}.json"
        )

    def run_chunk(self, ids: list[int], level: int) -> dict[str, dict[str, Any]]:
        """Generate one (image chunk, level) batch; returns manifest state records."""
        images = []
        for image_id in ids:
            img = Image.open(
                self.img_root / str(image_id) / f"{level_name(level)}.jpg"
            ).convert("RGB")
            geo = self.geo[image_id]
            if img.size != (int(geo["base_width"]), int(geo["base_height"])):
                raise ValueError(
                    f"{image_id} level {level}: size {img.size} != base size"
                )
            images.append(img)
        seed = pa.state_seed(ids, level, self.base_seed)
        try:
            batch = pa.run_pgsg_batch(self.rt, images, ids, seed)
            results = [
                (p, pv, batch.elapsed_s / len(ids), seed, False)
                for p, pv in zip(batch.predictions, batch.provenance, strict=True)
            ]
        except RuntimeError as err:
            if not pa.is_empty_stack_error(err):
                raise
            print(
                f"  fallback to batch size 1 for {ids} level {level}: {err}", flush=True
            )
            results = [
                self._run_single(img, i, level)
                for img, i in zip(images, ids, strict=True)
            ]
        records = {}
        for image_id, (pred, prov, elapsed, used_seed, fallback) in zip(
            ids, results, strict=True
        ):
            meta = state_meta(
                self.geo[image_id],
                self.args.split,
                level,
                self.scales[level - 1],
                used_seed,
            )
            graph = pa.build_graph(
                pred,
                prov,
                meta,
                self.obj_names,
                self.rel_names,
                self.model_info,
                self.generation,
                elapsed,
            )
            problems = pa.check_graph_consistency(pred, prov)
            write_json_atomic(self.graph_path(image_id, level), graph)
            records[f"{image_id}/{level}"] = {
                "seed": used_seed,
                "batch_image_ids": ids if not fallback else [image_id],
                "elapsed_s": round(elapsed, 4),
                "fallback_single": fallback,
                "n_entities": len(graph["entities"]),
                "n_relations": len(graph["relations"]),
                "n_relations_without_segment": sum(
                    r["segment_id"] is None for r in graph["relations"]
                ),
                "problems": problems,
            }
        return records

    def _run_single(
        self, img: Image.Image, image_id: int, level: int
    ) -> tuple[Any, Any, float, int, bool]:
        seed = pa.state_seed([image_id], level, self.base_seed)
        try:
            res = pa.run_pgsg_batch(self.rt, [img], [image_id], seed)
            return res.predictions[0], res.provenance[0], res.elapsed_s, seed, True
        except RuntimeError as err:
            if not pa.is_empty_stack_error(err):
                raise
            print(
                f"  {image_id} level {level}: no parsable triplet -> empty graph",
                flush=True,
            )
            empty = pa.empty_prediction(len(self.obj_names), len(self.rel_names))
            return empty, None, 0.0, seed, True


def main() -> None:
    args = parse_args()
    cfg = load_robovis_config(args.config)
    out_root = resolve(cfg["output_root"]).resolve()
    if out_root != ALLOWED_OUTPUT_ROOT.resolve():
        raise ValueError(f"output_root must be {ALLOWED_OUTPUT_ROOT}, got {out_root}")
    out_dir = out_root / args.run_id
    levels = select_levels(args)
    rows = load_split_images(cfg, args.split, args.limit)
    ids = [int(r["image_id"]) for r in rows]
    chunks = [ids[i : i + args.batch_size] for i in range(0, len(ids), args.batch_size)]

    manifest_path = out_dir / manifest_name(args.seed_offset)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest and manifest.get("split") != args.split:
        raise ValueError(
            f"run {args.run_id} is split {manifest.get('split')}, not {args.split}"
        )

    runner = Runner(args, cfg, out_dir)
    if manifest:
        prev = (
            manifest.get("checkpoint", {}).get("sha256"),
            manifest.get("batching", {}).get("base_seed"),
            manifest.get("batching", {}).get("batch_size"),
        )
        now = runner.model_info["sha256"], runner.base_seed, args.batch_size
        if prev != now:
            raise ValueError(
                f"cache mismatch (sha256, base_seed, batch_size): {prev} vs {now}"
            )
    states = manifest.get("states", {})
    manifest.update(
        {
            "run_id": args.run_id,
            "split": args.split,
            "script": "scripts/25_pgsg_infer.py",
            "robovis_config": str(args.config),
            "robovis_config_sha256": sha256_file(args.config),
            "eval_config": str(runner.eval_config.relative_to(REPO_ROOT)),
            "eval_config_sha256": sha256_file(runner.eval_config),
            "checkpoint": {"path": cfg["pgsg"]["checkpoint"], **runner.model_info},
            "code": git_info(),
            "generation_resolved": runner.rt.resolved,
            "graph_generation_block": runner.generation,
            "batching": {
                "batch_size": args.batch_size,
                "base_seed": runner.base_seed,
                "seed_offset": args.seed_offset,
                "graphs_subdir": runner.graphs_subdir,
                "rule": "batch = consecutive images of the split (selected_images rank order) at one level",
                "seed_function": "milestone1.pgsg_adapter.state_seed(batch_image_ids, level, base_seed)",
                "fallback": "if PGSG raises on an image with no parsable triplet, the chunk is re-run per image with state_seed([image_id], level, base_seed)",
            },
            "image_ids": ids,
            "coordinates": "box_xyxy_base: 384-base input pixels; box_xyxy_orig = base * (coord_scale_x, coord_scale_y)",
        }
    )
    manifest.setdefault("created", datetime.now().isoformat(timespec="seconds"))

    n_total = len(chunks) * len(levels)
    run_start = time.perf_counter()
    n_run = 0
    for ci, chunk in enumerate(chunks):
        for level in levels:
            if all(runner.graph_path(i, level).exists() for i in chunk):
                continue
            t0 = time.perf_counter()
            states.update(runner.run_chunk(chunk, level))
            n_run += len(chunk)
            done = ci * len(levels) + levels.index(level) + 1
            print(
                f"[{done}/{n_total}] chunk {ci} level {level}: {time.perf_counter() - t0:.1f}s",
                flush=True,
            )
            manifest["states"] = states
            manifest["updated"] = datetime.now().isoformat(timespec="seconds")
            write_json_atomic(manifest_path, manifest)
    total = time.perf_counter() - run_start
    timing = manifest.setdefault("timing", [])
    timing.append(
        {
            "finished": datetime.now().isoformat(timespec="seconds"),
            "levels": levels,
            "n_states_run": n_run,
            "wall_s": round(total, 1),
            "s_per_state": round(total / n_run, 3) if n_run else None,
        }
    )
    manifest["states"] = states
    manifest["updated"] = datetime.now().isoformat(timespec="seconds")
    write_json_atomic(manifest_path, manifest)
    bad = {k: v["problems"] for k, v in states.items() if v["problems"]}
    print(f"done: {n_run} states in {total:.1f}s; states with problems: {len(bad)}")
    for k, v in list(bad.items())[:10]:
        print("  ", k, v)


if __name__ == "__main__":
    main()
