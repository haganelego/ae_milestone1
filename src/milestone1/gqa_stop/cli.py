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

from . import report as report_mod
from . import stages_data, stages_model
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
        return stages_model.pilot(
            ctx, levels, args.calibration, args.use_fallback_model
        )
    if command == "infer":
        split = require_split(args, ("dev", "test"))
        levels = ctx.parse_levels(args.levels or "all", [])
        return stages_model.infer(ctx, split, levels, args.force_continue)
    if command == "freeze":
        return stages_model.freeze(ctx)
    if command == "evaluate":
        return stages_model.run_evaluate(ctx)
    if ctx.dry_run:
        ctx.echo(f"[dry-run] report -> {ctx.paths.out('report.md')}")
        return None
    return report_mod.build_report(ctx.cfg, ctx.paths)


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
