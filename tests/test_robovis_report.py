"""Tests for the ROBOVIS2027 report layer (report.md assembly, Fig 3 panel data).

Run: uv run --project egtr --with pytest python -m pytest tests/test_robovis_report.py -q
"""

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.robovis_data import load_robovis_config  # noqa: E402
from milestone1.robovis_panels import hypotheses_at, panel_levels  # noqa: E402
from milestone1.robovis_replay import run_replay_eval  # noqa: E402
from milestone1.robovis_report import (  # noqa: E402
    build_report,
    classify_error,
    demote_headings,
    fmt,
    md_table,
    report_metric_roles,
    section_graph_stats,
)

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "robovis_synthetic"
FAST_REPEATS = 100
NUM_LEVELS = 20


@pytest.fixture
def fast_cfg() -> dict[str, Any]:
    cfg = copy.deepcopy(load_robovis_config())
    cfg["evaluation"]["bootstrap_repeats"] = FAST_REPEATS
    return cfg


# --------------------------------------------------------------------------
# Formatting / classification
# --------------------------------------------------------------------------


def test_fmt_never_prints_nan_and_keeps_ints() -> None:
    assert fmt(float("nan")) == "n/a"
    assert fmt(None) == "n/a"
    assert fmt(3) == "3"
    assert fmt(1.0) == "1.000"
    assert fmt("a|b") == "a\\|b"


def test_md_table_and_empty_rows() -> None:
    lines = md_table([{"a": 1, "b": 0.5}], ["a", "b"])
    assert lines == ["| a | b |", "|---|---|", "| 1 | 0.500 |"]
    assert "not available" in md_table([], ["a"])[0]


@pytest.mark.parametrize(
    ("error", "cause"),
    [
        ("upstream checks failure", "upstream_checks_failure"),
        ("schema: json: Expecting value: line 1", "json_parse"),
        ("schema: observed_issues[0].evidence longer than 240 chars", "text_too_long"),
        (
            "schema: hypotheses[1].anchor_ids not in graph: ['e5']",
            "anchor_not_in_graph",
        ),
        ("schema: items[1].limitation invalid: 'none'", "enum_invalid"),
        (
            "schema: hypotheses[2].region_anchor_id missing (use null)",
            "type_or_missing_field",
        ),
        ("HTTP 500 from server", "transport"),
        ("something else", "other"),
        (None, "other"),
    ],
)
def test_classify_error_uses_first_violation(error: str | None, cause: str) -> None:
    assert classify_error(error) == cause


def test_demote_headings() -> None:
    assert demote_headings(["# a", "text", "## b"]) == ["### a", "text", "#### b"]


# --------------------------------------------------------------------------
# Fig 3 panel data
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("a", "b"), [(9, 8), (20, 20), (2, 2), (1, 20)])
def test_panel_levels_contains_both_stops_and_final(a: int, b: int) -> None:
    levels = panel_levels(a, b, NUM_LEVELS)
    assert {a, b, NUM_LEVELS} <= set(levels)
    assert levels == sorted(set(levels))
    assert 3 <= len(levels) <= 4


def test_panel_levels_small_level_count() -> None:
    assert panel_levels(1, 2, 2) == [1, 2]


def test_hypotheses_at_reads_nested_record_with_judgment() -> None:
    # 1001 t7 = "a1h2": hypothesis u=2 -> basis 2 (odd id -> nested record layout).
    hyp = hypotheses_at(FIXTURE / "checks", FIXTURE / "decisions", 1001, 7)
    assert hyp.level == 7
    assert hyp.lines[0].startswith("[missing_relation] man holding cup")
    assert "unresolved, b=2/r=2, low_resolution" in hyp.lines[0]


def test_hypotheses_at_uses_nearest_available_level() -> None:
    # 1001 t1-t2 are empty graphs without checks files -> first level with checks.
    hyp = hypotheses_at(FIXTURE / "checks", None, 1001, 1)
    assert hyp.level == 3
    missing = hypotheses_at(FIXTURE / "checks", None, 999999, 5)
    assert (missing.level, missing.lines) == (None, ())


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


def test_build_report_marks_missing_inputs(
    tmp_path: Path, fast_cfg: dict[str, Any]
) -> None:
    run_dir = tmp_path / "out" / "syn"
    shutil.copytree(FIXTURE, run_dir)
    run_replay_eval(run_dir, run_dir / "eval" / "X", fast_cfg)
    notes = tmp_path / "notes.md"
    notes.write_text("# My notes\n- kept verbatim\n")

    text = build_report(
        run_id="syn",
        output_root=tmp_path / "out",
        selection_manifest=tmp_path / "missing_selection.json",
        paths=["X", "Y"],
        q_metrics=["R50"],
        num_levels=NUM_LEVELS,
        notes=notes,
    )
    assert "### X (q = R50)" in text
    # Oracle profile from stage_metrics: all three definitions, main first.
    assert "| R50 | relative_stable | main | 9 | 7 | 1 | 1 | 0 |" in text
    assert (
        "| R50 | monotone_stable | auxiliary, not used in main results | 9 | 8 | 0 | 1 | 0 |"
        in text
    )
    # Auxiliary MAE is a separate, labelled table, not a column of the main table.
    assert "Oracle MAE under the auxiliary definitions (auxiliary, not used" in text
    main_header = next(
        ln for ln in text.splitlines() if ln.startswith("| method | status")
    )
    assert "MAE gain_based" not in main_header
    assert "Graphs replayed: `graphs`." in text
    # Fixture has no clean columns -> the G statistics table says so.
    assert "## Observed graph G statistics" in text
    assert "_not available: stage_metrics.csv columns" in text
    assert "| proposed | main |" in text
    assert "| benefit_gated | exploratory* |" in text
    assert "| fixed_20 | baseline |" in text
    assert "_not available: eval/Y/summary.csv_" in text
    assert "_not available: outputs/robovis2027/preflight.json" in text
    assert "_not available: " + str(tmp_path / "missing_selection.json") in text
    assert "### My notes" in text and "- kept verbatim" in text
    assert "nan" not in text.lower().replace("n/a", "")
    # Methods table order: main conditions before exploratory, fixed_20 last.
    assert text.index("| proposed | main |") < text.index("| benefit_gated |")
    assert text.index("| benefit_gated |") < text.index("| fixed_20 | baseline |")


def test_fig1_dot_source_is_valid_graph() -> None:
    spec = importlib.util.spec_from_file_location(
        "fig1", REPO_ROOT / "scripts" / "37_robovis_fig1_pipeline.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dot = module.build_dot()
    assert dot.startswith("digraph pipeline {") and dot.rstrip().endswith("}")
    # GT only feeds the offline scoring box.
    gt_edges = [ln for ln in dot.splitlines() if ln.strip().startswith("gt ->")]
    assert gt_edges == ['  gt -> score [style=dashed, color="#8c8c8c"];']
    json.dumps(module.NODES)  # node table is plain data


def test_report_metric_roles_from_config() -> None:
    roles = report_metric_roles(load_robovis_config())
    assert roles == {"R50": "primary", "relsg_gt": "secondary"}
    assert report_metric_roles({"evaluation": {}}) == {"R50": "primary"}


def test_graph_stats_section_raw_vs_g(tmp_path: Path) -> None:
    (tmp_path / "stage_metrics.csv").write_text(
        "image_id,t,valid,R50,R50_clean,det_prec20,det_prec20_clean\n"
        "1,1,1,0.2,0.1,0.5,0.6\n2,1,1,0.4,0.3,0.7,0.8\n3,1,0,0.9,0.9,0.9,0.9\n"
    )
    lines = section_graph_stats(tmp_path)
    assert "| t | R50 (raw) | R50 (G) | det_prec20 (raw) | det_prec20 (G) |" in lines
    assert "| 1 | 0.300 | 0.200 | 0.600 | 0.700 |" in lines  # invalid row excluded
