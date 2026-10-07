"""Recovery plan Phase B (+ Phase C stats): adaptive headroom diagnostics, zero calls.

Usage:
    uv run --project egtr python scripts/43_robovis_recovery_diag.py [--run-id rec_r0]

Per dataset x metric (R50, relsg_gt; datasets never pooled):
  fixed_mix.csv          F(b) (upper concave hull) + legacy adjacent mix, b in {10,12,15,18}
  budget_oracle.csv      O(b) (knapsack DP), F(b), O-F, fixed_b, n / coverage
  two_choice_oracle.csv  t10/t20 oracle (top half by g = q20 - q10) vs B_half
  shuffled_assignment.csv  stop-list permutation diagnostic for existing methods
  cross_seed_oracle.csv + seed_differences.csv  (only if multiseed_gain.csv exists)
  audit.md, diagnostics_manifest.json
F and O are GT-posterior references, not methods. test100 is exploratory only.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.recovery_diagnostics import (  # noqa: E402
    BUDGETS,
    HIGH_LEVEL,
    LOW_LEVEL,
    METRICS,
    REFERENCE_NOTE,
    QualityMatrix,
    best_fixed_mixture,
    budget_oracle,
    cross_seed_oracle,
    legacy_adjacent_mix,
    multiseed_table,
    quality_matrix,
    seed_differences,
    shuffle_diagnostic,
    two_choice_oracle,
)

OUT_ROOT = "outputs/robovis2027_recovery"
DEV_EVAL = "outputs/robovis2027/dev_r0/eval"
TWENTY_LEVEL_MIN_COVERAGE = 0.8  # replay paths with C_t on >= 80 % of states
CROSS_SEED_PAIRS = ((1, 2), (2, 1), (0, 1), (1, 0), (0, 2), (2, 0))
PRIMARY_PAIRS = {(1, 2), (2, 1)}
OK_OUTCOMES = ("ok", "empty_graph")

# name -> stage metrics, role, observer, stop sources
DATASETS: dict[str, dict[str, Any]] = {
    "dev20": {"stage": "outputs/robovis2027/dev_r0/stage_metrics.csv",
              "role": "dev (selection set)", "observer": "PGSG 384-base",
              "stops": "dev_paths"},
    "test100": {"stage": "outputs/robovis2027/test_r0/stage_metrics.csv",
                "role": "exploratory only (already viewed)", "observer": "PGSG 384-base",
                "stops": None},
    "miru100": {"stage": "outputs/miru_gt_eval/stage_metrics.csv",
                "role": "MIRU old 100", "observer": "EGTR original-size",
                "stops": "outputs/miru_gt_eval/eval/stops.csv"},
    "heldout96": {"stage": "outputs/heldout_eval/stage_metrics.csv",
                  "role": "pre-registered held-out", "observer": "EGTR original-size",
                  "stops": "outputs/heldout_eval/eval/stops.csv"},
}  # fmt: skip


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-id", default="rec_r0")
    p.add_argument("--shuffle-repeats", type=int, default=1000)
    return p.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def support_text(support: tuple[tuple[int, float], ...]) -> str:
    return ";".join(f"t{t}:{p:.4f}" for t, p in support)


# --------------------------------------------------------------------------
# Phase B tables
# --------------------------------------------------------------------------


def budget_rows(name: str, spec: dict, qm: QualityMatrix) -> tuple[list, list]:
    f = qm.q.mean(axis=0)
    fixed_rows, oracle_rows = [], []
    base = {"dataset": name, "role": spec["role"], "observer": spec["observer"],
            "metric": qm.metric, "n": qm.n, "n_total": qm.n_total,
            "coverage": qm.coverage}  # fmt: skip
    for b in BUDGETS:
        mix = best_fixed_mixture(f, b)
        legacy = legacy_adjacent_mix(f, b)
        fixed_b = float(f[b - 1])
        fixed_rows.append({**base, "b": b, "F": mix.value, "F_support": support_text(mix.support),
                           "F_mean_level": mix.mean_level, "legacy_adjacent": legacy,
                           "F_minus_legacy": mix.value - legacy, "fixed_b": fixed_b,
                           "note": REFERENCE_NOTE})  # fmt: skip
        orc = budget_oracle(qm.q, b)
        oracle_rows.append({**base, "b": b, "O": orc.value, "F": mix.value,
                            "O_minus_F": orc.value - mix.value, "fixed_b": fixed_b,
                            "legacy_adjacent": legacy, "budget_total": orc.budget_total,
                            "O_mean_level": float(orc.assignment.mean()),
                            "note": f"{REFERENCE_NOTE}; integer budget"})  # fmt: skip
    return fixed_rows, oracle_rows


def two_choice_row(name: str, spec: dict, qm: QualityMatrix) -> dict[str, Any]:
    q_lo, q_hi = qm.q[:, LOW_LEVEL - 1], qm.q[:, HIGH_LEVEL - 1]
    res = two_choice_oracle(q_lo, q_hi)
    return {"dataset": name, "role": spec["role"], "metric": qm.metric, "n": res.n,
            "k_t20": res.k, "mean_q10": float(q_lo.mean()), "mean_q20": float(q_hi.mean()),
            "oracle": res.value, "b_half": res.b_half, "oracle_minus_b_half": res.gain,
            "note": REFERENCE_NOTE}  # fmt: skip


def dev_method_stops() -> list[tuple[str, pd.DataFrame]]:
    """(source, stops) of dev20 replay paths with ~20-level coverage."""
    out = []
    for path_dir in sorted(resolve(DEV_EVAL).iterdir()):
        stops_csv, cseq = path_dir / "stops.csv", path_dir / "c_sequences.csv"
        if not (stops_csv.exists() and cseq.exists()):
            continue
        c = pd.read_csv(cseq).drop_duplicates(["image_id", "t"])
        coverage = float(c["outcome"].isin(OK_OUTCOMES).mean())
        if coverage >= TWENTY_LEVEL_MIN_COVERAGE:
            out.append((f"dev_r0/{path_dir.name} (C_t coverage {coverage:.0%})",
                        pd.read_csv(stops_csv)))  # fmt: skip
    return out


def method_stop_sources(spec: dict) -> list[tuple[str, pd.DataFrame]]:
    src = spec["stops"]
    if src is None:
        return []
    if src == "dev_paths":
        return dev_method_stops()
    path = resolve(src)
    return [(src, pd.read_csv(path))] if path.exists() else []


def shuffle_rows(name: str, qm: QualityMatrix, repeats: int) -> list[dict[str, Any]]:
    rows = []
    index = {i: r for r, i in enumerate(qm.ids)}
    for source, stops in method_stop_sources(DATASETS[name]):
        for method, grp in stops.groupby("method", sort=True):
            if str(method).startswith("fixed_"):
                continue
            sel = grp[grp["image_id"].isin(index)]
            rows_idx = sel["image_id"].map(index).to_numpy()
            res = shuffle_diagnostic(
                qm.q[rows_idx], sel["stop_t"].to_numpy(int), repeats
            )
            rows.append({"dataset": name, "metric": qm.metric, "source": source,
                         "method": method, "n_dropped_incomplete": len(grp) - len(sel),
                         **res})  # fmt: skip
    return rows


# --------------------------------------------------------------------------
# Phase C cross-seed
# --------------------------------------------------------------------------


def multiseed_consistency(df: pd.DataFrame, dev_stage: pd.DataFrame) -> list[str]:
    """g == q20 - q10 and seed 0 == dev_r0 stage metrics (seed 0 is a re-score)."""
    lines = ["", "## 7. Multiseed input consistency", ""]
    if "metric" in df.columns and "g" in df.columns:
        both = df[["q10", "q20", "g"]].apply(pd.to_numeric, errors="coerce").dropna()
        bad = int(((both["q20"] - both["q10"] - both["g"]).abs() > 1e-9).sum())
        lines.append(f"- g column vs q20 - q10: {bad} mismatching rows of {len(both)}")
    for metric in METRICS:
        tabs = multiseed_table(df, metric)
        if 0 not in tabs:
            lines.append(f"- {metric}: seed 0 absent")
            continue
        ref = dev_stage.assign(q=pd.to_numeric(dev_stage[metric], errors="coerce"))
        ref = ref.pivot(index="image_id", columns="t", values="q")
        s0 = tabs[0]
        cols = [c for c in s0.columns if c in ref.columns]
        diff = (s0[cols] - ref.loc[s0.index, cols]).abs().stack()
        lines.append(f"- {metric}: seed 0 vs dev_r0/stage_metrics.csv max |diff| "
                     f"{diff.max():.2e} over {len(diff)} cells")  # fmt: skip
    return lines


def cross_seed_tables(path: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    rows, diffs = [], []
    for metric in METRICS:
        tables = multiseed_table(df, metric)
        d = seed_differences(tables).assign(metric=metric)
        diffs.append(d)
        for a, b in CROSS_SEED_PAIRS:
            if a in tables and b in tables:
                res = cross_seed_oracle(tables, a, b)
                rows.append({"metric": metric, "primary": (a, b) in PRIMARY_PAIRS, **res,
                             "note": REFERENCE_NOTE})  # fmt: skip
    return pd.DataFrame(rows), pd.concat(diffs, ignore_index=True)


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


def audit_lines(qms: dict[tuple[str, str], QualityMatrix], fixed: pd.DataFrame,
                cross_present: bool) -> list[str]:  # fmt: skip
    lines = ["# Phase B audit (generated by scripts/43_robovis_recovery_diag.py)", ""]
    lines += [
        "## 1. Bootstrap recomputes the matched fixed mix per resample",
        "",
        "- `milestone1.robovis_eval.paired_bootstrap` (used by scripts 30, 41, 42) draws image "
        "indices once (`rng.integers(0, n, (repeats, n))`) and, inside `eq_diff`, recomputes the "
        "resample's mean stop and `equal_observation_mix(mean_stop)` for EVERY resample; images are "
        "the resampling unit, levels are never resampled independently.",
        "- Verified by `tests/test_recovery_diagnostics.py::"
        "test_bootstrap_recomputes_equal_observation_mix_per_resample` (repeats + 1 mix evaluations, "
        "varying mean stop).",
        "- Caveat: the point estimate `eq_fixed_q` in summary.csv uses the full-sample mix; the "
        "comparator is the LEGACY adjacent two-level mix, not F(b). Max F(b) - legacy over b in "
        f"{list(BUDGETS)} per dataset x metric:",
        "",
    ]
    gap = fixed.groupby(["dataset", "metric"])["F_minus_legacy"].max().reset_index()
    lines += [
        f"  - {r.dataset} / {r.metric}: {r.F_minus_legacy:.4f}"
        for r in gap.itertuples()
    ]
    val_path = resolve("outputs/robovis2027/dev_r0/official_validation.json")
    lines += ["", "## 2. R50 official match", ""]
    if val_path.exists():
        val = json.loads(val_path.read_text())
        lv = val["levels"]
        lines.append(
            f"- dev20 (PGSG): `{val_path.relative_to(resolve('.'))}`: all_match="
            f"{val.get('all_match')}, {len(lv)} levels, max per-image recall diff "
            f"{max(v['max_abs_diff_per_image_recall'] for v in lv.values())}."
        )
    else:
        lines.append("- dev20: _not available: official_validation.json_")
    lines += [
        "- test100 (PGSG): no separate official validation file; same code path "
        "(`scripts/28_robovis_stage_metrics.py`) as dev20.",
        "- MIRU100 / held-out96 (EGTR): same `robovis_sgdet` matching code, but EGTR stored only "
        "<= 30 relations per state (conf >= 0.005), so R@50 is recall over the stored relations, "
        "not EGTR's official R@50.",
        "",
        "## 3. Missing GT, coverage, no zero filling",
        "",
    ]
    for (name, metric), qm in sorted(qms.items()):
        stage = pd.read_csv(resolve(DATASETS[name]["stage"]))
        no_gt = int((stage.groupby("image_id")["n_gt_rel"].max() == 0).sum())
        nan_metric = int(pd.to_numeric(stage[metric], errors="coerce").isna().sum())
        lines.append(f"- {name} / {metric}: complete series {qm.n}/{qm.n_total} "
                     f"({qm.coverage:.1%}); images without GT relations {no_gt}; "
                     f"missing/NaN {metric} cells {nan_metric}")  # fmt: skip
    lines += [
        "- Incomplete series are excluded from F, O, the two-choice oracle and the shuffle "
        "(never set to 0); zero-recovery images (all q = 0) are kept.",
        "",
        "## 4. Forced terminal vs failure",
        "",
        "- Replay (`stop_policy.replay_stop`, scripts 30 / 41) records `forced_terminal` (k-rule "
        "never satisfied by t20) and `failure_count` (missing C_t up to the stop) separately.",
    ]
    for source, stops in dev_method_stops():
        main = stops[
            stops["method"].isin(["proposed", "observed_only", "unknown_conservative"])
        ]
        g = main.groupby("method").agg(forced=("forced_terminal", "mean"),
                                       failures=("failure_count", "mean"))  # fmt: skip
        cells = ", ".join(f"{m}: forced {r.forced:.2f}, mean failures {r.failures:.2f}"
                          for m, r in g.iterrows())  # fmt: skip
        lines.append(f"  - {source}: {cells}")
    lines += [
        "- MIRU stop CSVs do not record whether a stop at 20 was triggered or a fallback; "
        "script 42 reports `at_final_level` instead of a forced rate.",
        "",
        "## 5. Integer vs expected budget",
        "",
        "- O(b) uses an integer budget (sum t_i <= floor(N*b)); F(b) is an expected-level mixture. "
        "Small O < F differences are integrality, not negative adaptivity.",
        "",
        "## 6. Cross-seed inputs",
        "",
        "- multiseed_gain.csv present: " + ("yes" if cross_present else
                                            "no (cross_seed_oracle.csv not produced yet)"),
    ]  # fmt: skip
    return lines


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "n/a"


def main() -> None:
    args = parse_args()
    out = resolve(OUT_ROOT) / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    fixed_rows, oracle_rows, two_rows, shuf_rows = [], [], [], []
    qms: dict[tuple[str, str], QualityMatrix] = {}
    inputs = {}
    for name, spec in DATASETS.items():
        path = resolve(spec["stage"])
        if not path.exists():
            print(f"skip {name}: {path} not found")
            continue
        inputs[name] = {"path": spec["stage"], "sha256": sha256(path)}
        stage = pd.read_csv(path)
        for metric in METRICS:
            qm = quality_matrix(stage, metric, name)
            qms[(name, metric)] = qm
            f_rows, o_rows = budget_rows(name, spec, qm)
            fixed_rows += f_rows
            oracle_rows += o_rows
            two_rows.append(two_choice_row(name, spec, qm))
            shuf_rows += shuffle_rows(name, qm, args.shuffle_repeats)
            print(f"{name}/{metric}: n={qm.n}/{qm.n_total}")
    fixed = pd.DataFrame(fixed_rows)
    fixed.to_csv(out / "fixed_mix.csv", index=False)
    pd.DataFrame(oracle_rows).to_csv(out / "budget_oracle.csv", index=False)
    pd.DataFrame(two_rows).to_csv(out / "two_choice_oracle.csv", index=False)
    pd.DataFrame(shuf_rows).to_csv(out / "shuffled_assignment.csv", index=False)
    cross = cross_seed_tables(out / "multiseed_gain.csv")
    if cross is not None:
        cross[0].to_csv(out / "cross_seed_oracle.csv", index=False)
        cross[1].to_csv(out / "seed_differences.csv", index=False)
    audit = audit_lines(qms, fixed, cross is not None)
    if cross is not None:
        audit += multiseed_consistency(
            pd.read_csv(out / "multiseed_gain.csv"),
            pd.read_csv(resolve(DATASETS["dev20"]["stage"])),
        )
    (out / "audit.md").write_text("\n".join(audit) + "\n")
    manifest = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_head": git_head(),
                "inputs": inputs, "budgets": list(BUDGETS), "metrics": list(METRICS),
                "shuffle_repeats": args.shuffle_repeats, "note": REFERENCE_NOTE}  # fmt: skip
    (out / "diagnostics_manifest.json").write_text(json.dumps(manifest, indent=2))
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        bo = pd.DataFrame(oracle_rows)
        print(bo[["dataset", "metric", "b", "O", "F", "O_minus_F", "fixed_b", "legacy_adjacent",
                  "n"]].round(4).to_string(index=False))  # fmt: skip
        print(pd.DataFrame(two_rows)[["dataset", "metric", "n", "oracle", "b_half",
                                      "oracle_minus_b_half"]].round(4).to_string(index=False))  # fmt: skip
    print(f"-> {out}")
    if any(math.isnan(r["coverage"]) for r in oracle_rows):
        print("warning: empty dataset")


if __name__ == "__main__":
    main()
