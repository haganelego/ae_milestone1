"""ROBOVIS2027: build the cleaned observed graph G for every cached state.

Reads outputs/robovis2027/<run_id>/graphs/<id>/level_XX.json and writes
outputs/robovis2027/<run_id>/graphs_clean/<id>/level_XX.json (or <subdir>_clean/ with
--graphs-subdir) with
milestone1.graph_clean.clean_graph (merge IoU / top-K relations from
evaluation.clean_merge_iou / evaluation.clean_top_relations). Always rebuilds
(deterministic, ~seconds); raw graphs are not modified.

Usage:
  uv run --project egtr python scripts/35_robovis_clean_graphs.py --runs dev_r0 test_r0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.graph_clean import (  # noqa: E402
    DEFAULT_MERGE_IOU,
    DEFAULT_TOP_RELATIONS,
    clean_graph,
)
from milestone1.robovis_data import load_robovis_config  # noqa: E402

RAW_DIR = "graphs"
CLEAN_SUFFIX = "_clean"


def clean_subdir(graphs_subdir: str) -> str:
    """graphs -> graphs_clean, graphs_seed1 -> graphs_seed1_clean."""
    return f"{graphs_subdir}{CLEAN_SUFFIX}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/robovis2027.yaml"
    )
    parser.add_argument("--runs", nargs="+", default=["dev_r0", "test_r0"])
    parser.add_argument(
        "--graphs-subdir",
        default=RAW_DIR,
        help="raw graph dir inside the run (e.g. graphs_seed1); output -> <subdir>_clean",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_robovis_config(args.config)
    ev = cfg.get("evaluation", {})
    merge_iou = float(ev.get("clean_merge_iou", DEFAULT_MERGE_IOU))
    top_rel = int(ev.get("clean_top_relations", DEFAULT_TOP_RELATIONS))
    for run in args.runs:
        run_dir = resolve(cfg["output_root"]) / run
        n = 0
        for raw in sorted((run_dir / args.graphs_subdir).glob("*/level_*.json")):
            out = (
                run_dir / clean_subdir(args.graphs_subdir) / raw.parent.name / raw.name
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            cleaned = clean_graph(json.loads(raw.read_text()), merge_iou, top_rel)
            tmp = out.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cleaned, indent=1))
            os.replace(tmp, out)
            n += 1
        print(
            f"{run}/{clean_subdir(args.graphs_subdir)}: wrote {n} cleaned graphs (merge_iou={merge_iou}, top_relations={top_rel})"
        )


if __name__ == "__main__":
    main()
