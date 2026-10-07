"""ROBOVIS2027 failure analysis of generated checks (plan §4; zero model calls).

GT is used post hoc, for explanation only. For each variant this script reads
checks/<V>/<id>/level_XX.json (payload under "checks") and
decisions/<V>/<id>/level_XX.json, the raw graph, and the VG-150 GT. It
classifies every hypothesis with milestone1.failure_analysis
(bookkeeping / gt_supported / already_observed / off_vocab) and writes:
  failure_analysis/<V>.csv         one row per hypothesis / observed issue
  failure_analysis/<V>_states.csv  per state: GT labels absent from the rendering (headroom)
  failure_analysis/summary.md
Graphs are read from graphs/ or, for G variants (a "G" segment in the
variant name, milestone1.robovis_replay.is_clean_graph_variant), graphs_clean/; --graphs-subdir V=dir overrides per variant.
The rendered entity set (what the model saw) is re-derived with
check_generator.render_graph. The prompt version (v1/v2) is the one whose
rendered-text sha256 equals the stored graph_sha256; mismatches are counted.

Usage:
  uv run --project egtr python scripts/38_robovis_failure_analysis.py --run-id dev_r0 --variants F1,F2,F2__9b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1 import check_generator as cg  # noqa: E402
from milestone1 import failure_analysis as fa  # noqa: E402
from milestone1.config import resolve  # noqa: E402
from milestone1.gt import load_gt_scene_graphs  # noqa: E402
from milestone1.ollama_client import sha256_text  # noqa: E402
from milestone1.robovis_data import load_robovis_config  # noqa: E402
from milestone1.robovis_replay import is_clean_graph_variant  # noqa: E402

CATEGORIES_PATH = REPO_ROOT / "data/processed/pgsg_vg/categories_dict.json"
OUT_DIR = "failure_analysis"
TOP_ANCHOR_K = 5
N_EXAMPLES = 3
HYP, OBS = "hypothesis", "observed_issue"
RAW_GRAPHS = "graphs"
CLEAN_GRAPHS = "graphs_clean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/robovis2027.yaml"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--variants", required=True, help="comma list, e.g. F1,F2,F2__9b"
    )
    parser.add_argument(
        "--graphs-subdir",
        default=None,
        help="per-variant graph dir overrides 'V=dir,V2=dir'; default: "
        f"{CLEAN_GRAPHS} for G variants (is_clean_graph_variant), else {RAW_GRAPHS}",
    )
    parser.add_argument(
        "--out-subdir",
        default=OUT_DIR,
        help=f"output dir inside the run (default {OUT_DIR}); use another name for "
        "partial runs (e.g. a full-level run of a subset) so summaries are not clobbered",
    )
    parser.add_argument(
        "--levels",
        default=None,
        help="comma list of levels to include (default: all present); use "
        "1,5,10,15,20 to compare variants on the common level grid",
    )
    return parser.parse_args()


def parse_overrides(spec: str | None) -> dict[str, str]:
    if not spec:
        return {}
    pairs = [item.split("=", 1) for item in spec.split(",") if item.strip()]
    return {k.strip(): v.strip() for k, v in pairs}


def graphs_subdir_for(variant: str, overrides: dict[str, str]) -> str:
    """Same convention as A3: variants ending in _G were generated from G."""
    if variant in overrides:
        return overrides[variant]
    return CLEAN_GRAPHS if is_clean_graph_variant(variant) else RAW_GRAPHS


def headroom(gt: dict[str, Any], rendered_labels: set[str]) -> dict[str, float]:
    """GT object labels of the image absent from the rendered graph."""
    gt_labels = {o["label"] for o in gt["objects"]}
    absent = gt_labels - rendered_labels
    return {
        "n_gt_labels": len(gt_labels),
        "n_absent_gt_labels": len(absent),
        "frac_absent_gt_labels": len(absent) / len(gt_labels)
        if gt_labels
        else float("nan"),
        "absent_gt_labels": "|".join(sorted(absent)),
    }


def rendered_entity_ids(
    graph: dict[str, Any], record: dict[str, Any], limits: dict[str, Any]
) -> tuple[set[str], str | None]:
    """Entity ids shown to the model and the matching prompt version (None = no match)."""
    for version, lim in limits.items():
        rendered = cg.render_graph(graph, lim)
        if sha256_text(rendered.text) == record.get("graph_sha256"):
            return set(rendered.entity_ids), version
    fallback = cg.render_graph(graph, limits[cg.PROMPT_V1])
    return set(fallback.entity_ids), None


def analyse_state(
    variant: str,
    record: dict[str, Any],
    decision: dict[str, Any] | None,
    graph: dict[str, Any],
    gt: dict[str, Any],
    names: list[str],
    shown: set[str],
    version: str | None,
) -> tuple[list[dict[str, Any]], str | None]:
    label_of = {e["id"]: e["label"] for e in graph["entities"]}
    rendered_labels = {label_of[i] for i in shown if i in label_of}
    graph_labels = set(label_of.values())
    gt_labels = {o["label"] for o in gt["objects"]}
    top5 = {
        e["id"]
        for e in sorted(graph["entities"], key=lambda e: -float(e["score"]))[
            :TOP_ANCHOR_K
        ]
    }
    judged = {it["id"]: it for it in (decision or {}).get("items", []) or []}
    base = {
        "variant": variant,
        "image_id": record["image_id"],
        "level": record["level"],
        "prompt_version": version,
    }
    rows = []
    checks = record.get("checks") or {}
    for h in checks.get("hypotheses", []) or []:
        cls, mentioned = fa.classify_hypothesis(
            h.get("candidate", ""),
            h.get("evidence", ""),
            names,
            gt_labels,
            rendered_labels,
        )
        j = judged.get(h["id"], {})
        rows.append(
            {
                **base,
                "item_id": h["id"],
                "item_type": HYP,
                "kind": h.get("kind"),
                "class": cls,
                "restates_score": None,
                "mentioned_labels": "|".join(sorted(mentioned)),
                "gt_missing_labels": "|".join(
                    sorted(mentioned & (gt_labels - rendered_labels))
                ),
                "mentioned_in_graph_not_rendered": "|".join(
                    sorted(mentioned & (graph_labels - rendered_labels))
                ),
                "region_anchor_id": h.get("region_anchor_id"),
                "anchor_in_top5": h.get("region_anchor_id") in top5,
                "text": h.get("candidate", ""),
                "evidence": h.get("evidence", ""),
                "evidence_len": len(h.get("evidence", "") or ""),
                "status": j.get("status"),
                "limitation": j.get("limitation"),
                "basis": j.get("basis"),
                "resolution_benefit": j.get("resolution_benefit"),
                "placeholder_echo": fa.is_placeholder_echo(
                    h.get("candidate", ""), h.get("evidence", "")
                ),
            }
        )
    for o in checks.get("observed_issues", []) or []:
        j = judged.get(o["id"], {})
        rows.append(
            {
                **base,
                "item_id": o["id"],
                "item_type": OBS,
                "kind": None,
                "class": None,
                "restates_score": fa.restates_score(
                    o.get("target", ""), o.get("evidence", "")
                ),
                "mentioned_labels": "",
                "gt_missing_labels": "",
                "mentioned_in_graph_not_rendered": "",
                "region_anchor_id": o.get("region_anchor_id"),
                "anchor_in_top5": o.get("region_anchor_id") in top5,
                "text": o.get("target", ""),
                "evidence": o.get("evidence", ""),
                "evidence_len": len(o.get("evidence", "") or ""),
                "status": j.get("status"),
                "limitation": j.get("limitation"),
                "basis": j.get("basis"),
                "resolution_benefit": j.get("resolution_benefit"),
                "placeholder_echo": fa.is_placeholder_echo(
                    o.get("target", ""), o.get("evidence", "")
                ),
            }
        )
    return rows, version


def analyse_variant(
    run_dir: Path,
    variant: str,
    gts: dict[int, dict[str, Any]],
    names: list[str],
    limits: dict[str, Any],
    graphs_subdir: str,
    levels: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "states": 0,
        "failed_or_empty": 0,
        "version_unmatched": 0,
        "graphs_subdir": graphs_subdir,
    }
    for path in sorted((run_dir / "checks" / variant).glob("*/level_*.json")):
        if levels is not None and int(path.stem.split("_")[1]) not in levels:
            continue
        record = json.loads(path.read_text())
        stats["states"] += 1
        graph = json.loads(
            (run_dir / graphs_subdir / path.parent.name / path.name).read_text()
        )
        gt = gts[int(record["image_id"])]
        shown, version = rendered_entity_ids(graph, record, limits)
        label_of = {e["id"]: e["label"] for e in graph["entities"]}
        rendered_labels = {label_of[i] for i in shown if i in label_of}
        failed = bool(record.get("failure") or not record.get("checks"))
        states.append(
            {
                "variant": variant,
                "image_id": record["image_id"],
                "level": record["level"],
                "prompt_version": version,
                "failed_or_empty": failed,
                "n_rendered_labels": len(rendered_labels),
                **headroom(gt, rendered_labels),
            }
        )
        if failed:
            stats["failed_or_empty"] += 1
            continue
        stats["version_unmatched"] += version is None
        dec_path = run_dir / "decisions" / variant / path.parent.name / path.name
        decision = json.loads(dec_path.read_text()) if dec_path.exists() else None
        state_rows, _ = analyse_state(
            variant, record, decision, graph, gt, names, shown, version
        )
        rows.extend(state_rows)
    return pd.DataFrame(rows), pd.DataFrame(states), stats


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def headroom_table(hyp: pd.DataFrame, states: pd.DataFrame) -> list[str]:
    """Headroom (GT labels absent from the rendering) vs realized gt_supported."""
    lines = [
        "Headroom vs realized: GT object labels absent from the rendered graph (all states with",
        "a check file), against hypotheses that name such a label (gt_supported).",
        "",
        "| level | states | GT labels absent (frac, mean) | absent GT labels / image (mean) "
        "| gt_supported (share of hyp) | gt_supported hyp / state |",
        "|---|---|---|---|---|---|",
    ]
    for level in [*sorted(states.level.unique()), "all"]:
        st = states if level == "all" else states[states.level == level]
        h = hyp if level == "all" else hyp[hyp.level == level]
        n_ok = int((~st.failed_or_empty).sum())
        gt_sup = int((h["class"] == fa.GT_SUPPORTED).sum())
        share = pct(gt_sup / len(h)) if len(h) else "-"
        per_state = f"{gt_sup / n_ok:.2f}" if n_ok else "-"
        lines.append(
            f"| {level} | {len(st)} | {pct(st.frac_absent_gt_labels.mean())} "
            f"| {st.n_absent_gt_labels.mean():.1f} | {share} | {per_state} |"
        )
    return lines + [""]


def summarize(
    variant: str, df: pd.DataFrame, states: pd.DataFrame, stats: dict[str, Any]
) -> str:
    hyp, obs = df[df.item_type == HYP], df[df.item_type == OBS]
    lines = [
        f"## {variant}",
        "",
        f"Graph source: `{stats['graphs_subdir']}/`. States: {stats['states']}. Failed/empty: {stats['failed_or_empty']}. "
        f"Prompt version not reproduced: {stats['version_unmatched']}. "
        f"Hypotheses: {len(hyp)}. Observed issues: {len(obs)}. "
        f"Items that echo a schema placeholder (e.g. '<short evidence>'): "
        f"{int(df.placeholder_echo.sum())}.",
        "",
        "| level | n hyp | "
        + " | ".join(fa.CLASSES)
        + " | anchor in top-5 (hyp) | mean evidence chars (hyp) | obs restating scores |",
        "|---" * (len(fa.CLASSES) + 5) + "|",
    ]
    for level in sorted(df.level.unique()):
        h, o = hyp[hyp.level == level], obs[obs.level == level]
        dist = " | ".join(
            pct((h["class"] == c).mean()) if len(h) else "-" for c in fa.CLASSES
        )
        lines.append(
            f"| {level} | {len(h)} | {dist} | {pct(h.anchor_in_top5.mean()) if len(h) else '-'} | "
            f"{h.evidence_len.mean():.0f} | {pct(o.restates_score.mean()) if len(o) else '-'} |"
        )
    dist_all = " | ".join(pct((hyp["class"] == c).mean()) for c in fa.CLASSES)
    lines.append(
        f"| all | {len(hyp)} | {dist_all} | {pct(hyp.anchor_in_top5.mean())} | {hyp.evidence_len.mean():.0f} | {pct(obs.restates_score.mean())} |"
    )
    lines += ["", *headroom_table(hyp, states)]
    lines += ["", "Judgments per class (share of hypotheses, status / limitation):", ""]
    for c in fa.CLASSES:
        sub = hyp[hyp["class"] == c]
        if sub.empty:
            continue
        st = ", ".join(
            f"{k} {pct(v)}"
            for k, v in sub.status.fillna("none").value_counts(normalize=True).items()
        )
        li = ", ".join(
            f"{k} {pct(v)}"
            for k, v in sub.limitation.fillna("none")
            .value_counts(normalize=True)
            .items()
        )
        lines.append(f"- `{c}` (n={len(sub)}): status {st}; limitation {li}")
    lines += ["", "Shortest verbatim examples per class (candidate || evidence):", ""]
    for c in fa.CLASSES:
        sub = (
            hyp[hyp["class"] == c]
            .assign(_n=lambda d: d.text.str.len() + d.evidence.str.len())
            .sort_values("_n")
        )
        for _, r in sub.head(N_EXAMPLES).iterrows():
            lines.append(
                f'- `{c}` [{r.image_id} t={r.level}] "{r.text}" || "{r.evidence}"'
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_robovis_config(args.config)
    run_dir = resolve(cfg["output_root"]) / args.run_id
    ids = [
        int(i) for i in json.loads((run_dir / "manifest.json").read_text())["image_ids"]
    ]
    gts = load_gt_scene_graphs(
        cfg["dataset"]["vg_root"], cfg["dataset"]["source_split"], ids
    )
    names = json.loads(CATEGORIES_PATH.read_text())["obj"]
    limits = {v: cg.limits_from_config(cfg, v) for v in cg.PROMPT_VERSIONS}
    overrides = parse_overrides(args.graphs_subdir)
    levels = {int(x) for x in args.levels.split(",")} if args.levels else None
    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = [
        f"# Failure analysis of checks: {args.run_id}",
        "",
        "Zero model calls. GT is used post hoc, for explanation only. The classes are",
        "defined in `src/milestone1/failure_analysis.py` and checked in this order:",
        "bookkeeping (talk about the graph itself), gt_supported (names a GT class missing",
        "from the rendered graph), already_observed (names a rendered class), off_vocab.",
        "Columns other than the class distribution: whether the hypothesis region anchor is",
        "among the top-5 entities by score, the mean evidence length in characters, and",
        "the share of observed issues that only restate scores or confidence.",
        "",
        "Caveat: gt_supported fires when ANY mentioned class is in GT but not rendered, so a",
        "context word can trigger it. Example: 'sign ... the street name' counts because",
        "'street' is unrendered, although 'sign' is rendered. The class is near zero",
        "anyway: candidates almost never name a GT object that PGSG failed to show.",
        "",
    ]
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        subdir = graphs_subdir_for(variant, overrides)
        df, states, stats = analyse_variant(
            run_dir, variant, gts, names, limits, subdir, levels
        )
        df.to_csv(out_dir / f"{variant}.csv", index=False)
        states.to_csv(out_dir / f"{variant}_states.csv", index=False)
        parts.append(summarize(variant, df, states, stats))
        print(
            f"{variant}: {stats}; hypotheses {int((df.item_type == HYP).sum()) if len(df) else 0}"
        )
    (out_dir / "summary.md").write_text("\n".join(parts))
    print(f"wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
