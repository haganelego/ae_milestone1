"""Command line entry point: ``python -m milestone1.gqa_stop.cli <cmd> --config ...``.

Commands: preflight, probe, prepare, screen, pilot, infer, freeze, evaluate,
report. Every command accepts ``--dry-run`` (no model calls, no writes) and is
resumable through the on-disk request cache.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import (
    boxes_v5,
    devgrid_v22,
    explore,
    fair_subset,
    firing_v23,
    informed_reader,
    replays_v5,
    stages_data,
    stages_model,
    stages_v3,
    stages_v4,
    stages_v4c,
    stages_v4c_test,
    textprobe,
)
from . import report as report_mod
from .client import PauseRequired
from .common import StageError, load_effective_config
from .context import Context, TransportFactory, default_transport

EXIT_STAGE_ERROR = 1
EXIT_PAUSED = 2
COMMANDS = (
    "preflight",
    "probe",
    "prepare",
    "screen",
    "pilot",
    "infer",
    "freeze",
    "evaluate",
    "report",
    "devgrid",
    "textprobe",
    "detgate",
    "readers",
    "v1arm",
    "explore",
    "extension",
    "informed-reader",
    "curves",
    "firing",
    "v5-box",
    "v5-replays",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="milestone1.gqa_stop.cli")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split", choices=("dev", "test"))
    parser.add_argument("--levels", default=None, help="'all' or comma list")
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--use-fallback-model", action="store_true")
    parser.add_argument(
        "--force-continue",
        action="store_true",
        help="dev infer only: proceed past a failed pilot gate (documented exit "
        "branch); logged as GATE OVERRIDE and results become exploratory",
    )
    parser.add_argument(
        "--subset",
        type=Path,
        default=None,
        help="freeze/evaluate only: visual_subset.json (fair-v1 re-analysis, §18.1)",
    )
    parser.add_argument("--revise", action="store_true")
    parser.add_argument("--revision-file", type=Path, default=None)
    parser.add_argument("--reason", default=None)
    parser.add_argument(
        "--repo-root", type=Path, default=None, help="override paths.repo_root (tests)"
    )
    return parser


def make_context(
    args: argparse.Namespace, transport_factory: TransportFactory = default_transport
) -> Context:
    cfg, paths = load_effective_config(args.config, args.repo_root)
    return Context(args.config, cfg, paths, args.dry_run, transport_factory)


def require_split(args: argparse.Namespace, allowed: Sequence[str]) -> str:
    if args.split not in allowed:
        raise StageError(f"{args.command} needs --split in {list(allowed)}")
    return str(args.split)


def dispatch(ctx: Context, args: argparse.Namespace) -> object:
    command = args.command
    if command == "preflight":
        return stages_model.preflight(ctx)
    if command == "probe":
        return stages_data.probe(ctx)
    if command == "prepare":
        return stages_data.prepare(ctx, args.revise, args.revision_file, args.reason)
    if command == "screen":
        return stages_data.screen(ctx, require_split(args, ("dev", "test")))
    if command == "pilot":
        require_split(args, ("dev",))
        levels = ctx.parse_levels(args.levels, ctx.cfg["execution"]["pilot_levels"])
        if ctx.family in stages_v3.V3_FAMILIES:
            return stages_v3.pilot_v3(ctx, levels)
        if ctx.family == stages_model.LOGPROB:
            return stages_v4.pilot_v4(ctx, levels)
        return stages_model.pilot(
            ctx, levels, args.calibration, args.use_fallback_model
        )
    if command == "infer":
        split = require_split(args, ("dev", "test"))
        levels = ctx.parse_levels(args.levels or "all", [])
        return stages_model.infer(ctx, split, levels, args.force_continue)
    if command == "freeze":
        if args.subset is not None:
            return fair_subset.freeze_on_subset(ctx, args.subset)
        if ctx.family == stages_model.LOGPROB:
            return stages_v4c_test.freeze_v4c(ctx)
        return stages_model.freeze(ctx)
    if command == "evaluate":
        if args.subset is not None:
            return fair_subset.evaluate_on_subset(ctx, args.subset)
        if ctx.family == stages_model.LOGPROB:
            return stages_v4c_test.evaluate_v4c(ctx)
        return stages_model.run_evaluate(ctx)
    if command == "textprobe":
        return textprobe.textprobe(ctx)
    if command == "detgate":
        return stages_v4c.detgate(ctx)
    if command == "readers":
        split = require_split(args, ("dev", "test"))
        return stages_v4c.reader_levels(
            ctx, split, ctx.parse_levels(args.levels or "all", [])
        )
    if command == "v1arm":
        if args.split == "test":
            return stages_v4c_test.v1arm_test(ctx)
        return stages_v4c.v1arm(ctx)
    if command == "v5-replays":
        return replays_v5.run_replays(ctx)
    if command == "v5-box":
        return boxes_v5.run_box_precheck(ctx)
    if command == "firing":
        return firing_v23.firing(ctx, args.split or "dev")
    if command == "curves":
        return devgrid_v22.devgrid_v22(ctx)
    if command == "informed-reader":
        return informed_reader.run_probe(ctx)
    if command == "extension":
        return stages_v4c_test.extension(ctx)
    if command == "explore":
        return explore_stage(ctx)
    if command == "devgrid":
        if ctx.family in stages_v3.V3_FAMILIES:
            return stages_v3.devgrid_v3(ctx)
        if ctx.family == stages_model.LOGPROB:
            return stages_v4.devgrid_v4(ctx)
        return stages_model.devgrid(ctx)
    if ctx.dry_run:
        ctx.echo(f"[dry-run] report -> {ctx.paths.out('report.md')}")
        return None
    return report_mod.build_report(ctx.cfg, ctx.paths)


def explore_stage(ctx: Context) -> object:
    """Exploratory replays on the dev level table (no model calls)."""
    table = explore.merged_table(
        ctx.paths.out("level_table_dev.csv"), ctx.paths.out("v1arm/level_table_dev.csv")
    )
    if ctx.dry_run:
        ctx.echo(f"[dry-run] explore: {table['question_id'].nunique()} dev questions")
        return None
    summary = explore.run_exploration(table, ctx.cfg["policy"], ctx.paths.output_root)
    ctx.log(f"exploratory replays ({explore.LABEL}) written")
    return summary


def main(
    argv: Sequence[str] | None = None,
    transport_factory: TransportFactory = default_transport,
) -> int:
    args = build_parser().parse_args(argv)
    ctx = make_context(args, transport_factory)
    try:
        dispatch(ctx, args)
    except PauseRequired as exc:
        ctx.log(f"PAUSED: {exc}")
        ctx.state.set("paused", {"reason": str(exc)})
        return EXIT_PAUSED
    except StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_STAGE_ERROR
    return 0


if __name__ == "__main__":
    sys.exit(main())
