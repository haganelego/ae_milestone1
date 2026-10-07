"""MIRU-style method comparison with significance marks (presentation only, zero calls).

Usage:
    uv run --project egtr python scripts/48_recovery_miru_style_table.py [--run-id rec_r0]

For Phase D (seedA, seedB replication) and dev (seeds 0/1/2):
rows = U policy (frozen), same-p mix (expected value + permutation 95 % range),
fixed10/15/20, GT two-choice oracle (reference; same number of t20 slots as the policy);
columns = mean level, MAE in levels vs the oracle stop, agreement with the oracle,
R50(stop), relsg_gt(stop). Significance: paired bootstrap 2000 (images; the mix's p
recomputed per resample) and, vs the mix, a permutation test (policy's t20 slots
reassigned at random 2000x). The plan's primary (R50 policy - same-p mix) is the headline.
Writes rec_r0/method_comparison_miru_style.{csv,md}.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.phase_d_eval import (  # noqa: E402
    PERM_REPEATS,
    PRIMARY_METRIC,
    SCHEDULE_AWARE_NOTE,
    SEED_B_LEVELS,
    SeedLevels,
    _policy_values,
    load_policy,
    method_rows,
    oracle_cont,
    significance_rows,
)
from milestone1.recovery_diagnostics import METRICS  # noqa: E402

OUT_ROOT = "outputs/robovis2027_recovery"
DEV_FULL_STAGE = "outputs/robovis2027/dev_r0/stage_metrics.csv"
# label -> (policy csv, stage csv) relative to the run dir (or repo for dev seed0)
# (label, U decisions, conf_two_stage decisions, stage metrics)
SEED_SETS = (
    ("Phase D seedA", "test_policy.csv", "conf_test_policy.csv", "stage_metrics_seedA.csv"),
    ("Phase D seedB (replication)", "test_policy.csv", "conf_test_policy.csv",
     "stage_metrics_seedB.csv"),
    ("dev seed0", "dev_two_stage.csv", "conf_dev_two_stage.csv", DEV_FULL_STAGE),
    ("dev seed1", "dev_two_stage.csv", "conf_dev_two_stage.csv", "stage_metrics_seed1.csv"),
    ("dev seed2", "dev_two_stage.csv", "conf_dev_two_stage.csv", "stage_metrics_seed2.csv"),
)  # fmt: skip
ROLE = {
    "U policy": "proposed (frozen U)",
    "mix (same p)": "online comparator (primary)",
    "conf_two_stage": "online comparator (level-agnostic, observation-based)",
    "fixed10": SCHEDULE_AWARE_NOTE,
    "fixed15": SCHEDULE_AWARE_NOTE,
    "fixed20": SCHEDULE_AWARE_NOTE,
    "GT two-choice oracle (reference)": "GT-posterior reference (upper bound)",
}
TRUE_STRINGS = {"1", "true", "yes", "t"}
COLUMN_LABEL = {"mae": "MAE vs oracle stop (levels)", "agree": "agreement with oracle",
                "q": "q(stop)"}  # fmt: skip


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-id", default="rec_r0")
    p.add_argument("--repeats", type=int, default=2000)
    return p.parse_args()


def seed_levels(stage: pd.DataFrame, policy: pd.DataFrame,
                conf: pd.DataFrame | None = None) -> SeedLevels:  # fmt: skip
    """q at t10/15/20 for images complete in BOTH metrics (no zero filling)."""
    if stage.duplicated(["image_id", "t"]).any():
        raise ValueError("duplicate (image_id, t) rows")
    sub = stage[stage["t"].isin(SEED_B_LEVELS)]
    valid = sub["valid"].astype(str).str.strip().str.lower().isin(TRUE_STRINGS)
    tables = {}
    for metric in METRICS:
        vals = sub.assign(v=pd.to_numeric(sub[metric], errors="coerce"))[valid]
        tables[metric] = vals.pivot(index="image_id", columns="t", values="v").dropna()
    pol = policy.set_index("image_id")
    ids = [i for i in pol.index if all(i in t.index for t in tables.values())]
    conf_cont = None
    if conf is not None:
        c = conf.set_index("image_id")
        missing = [i for i in ids if i not in c.index]
        if missing:
            raise ValueError(f"conf_two_stage decisions missing for {missing[:5]}")
        conf_cont = (c.loc[ids, "decision_t"] == 20).to_numpy(bool)
    return SeedLevels(
        {m: tables[m].loc[ids, list(SEED_B_LEVELS)].to_numpy(float) for m in METRICS},
        (pol.loc[ids, "decision_t"] == 20).to_numpy(bool), tuple(int(i) for i in ids),
        conf_cont,
    )  # fmt: skip


def method_table(
    label: str, levels: SeedLevels, perm_repeats: int, seed: int
) -> list[dict[str, Any]]:
    p = float(levels.cont.mean())
    rows_by_metric = {m: method_rows(levels, m) for m in METRICS}
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(levels.cont) for _ in range(perm_repeats)]
    perm_q = {}
    for m in METRICS:
        orc = oracle_cont(levels.q[m], int(levels.cont.sum()))
        vals = [
            _policy_values(levels, m, orc, np.arange(len(levels.cont)), c)
            for c in perms
        ]
        perm_q[m] = np.quantile([v["q"] for v in vals], [0.025, 0.975])
    out = []
    for name in rows_by_metric[PRIMARY_METRIC]:
        r50 = rows_by_metric["R50"][name]
        rel = rows_by_metric["relsg_gt"][name]
        p_conf = (
            float(levels.conf_cont.mean()) if levels.conf_cont is not None else math.nan
        )
        level = {"U policy": 10 + 10 * p, "mix (same p)": 10 + 10 * p, "fixed10": 10,
                 "fixed15": 15, "fixed20": 20, "conf_two_stage": 10 + 10 * p_conf,
                 "GT two-choice oracle (reference)": 10 + 10 * p}[name]  # fmt: skip
        row = {"set": label, "method": name, "role": ROLE[name], "n": len(levels.cont),
               "mean_level": level,
               "mae_vs_R50_oracle": r50["mae"].mean(), "agree_R50_oracle": r50["agree"].mean(),
               "R50": r50["q"].mean(), "mae_vs_relsg_oracle": rel["mae"].mean(),
               "agree_relsg_oracle": rel["agree"].mean(), "relsg_gt": rel["q"].mean()}  # fmt: skip
        if name == "mix (same p)":
            row["R50_perm95"] = f"[{perm_q['R50'][0]:.4f}, {perm_q['R50'][1]:.4f}]"
            row["relsg_perm95"] = (
                f"[{perm_q['relsg_gt'][0]:.4f}, {perm_q['relsg_gt'][1]:.4f}]"
            )
        out.append(row)
    return out


def fmt(v: Any, digits: int = 4) -> str:
    if isinstance(v, float):
        return "—" if np.isnan(v) else f"{v:.{digits}f}"
    return str(v)


def markdown(methods: pd.DataFrame, sig: pd.DataFrame) -> str:
    lines = [
        "# MIRU-style method comparison (presentation; plan primary = R50 policy − same-p mix)",
        "",
        "Oracle = GT two-choice oracle on that seed (top-k by g = q20 − q10, k = the policy's "
        "number of t20 slots); a GT-posterior reference, not a method. With stops in {10, 20}, "
        "MAE = 10 × (1 − agreement). Comparators are ONLINE only: the same-p image-independent "
        "mix (primary) and conf_two_stage (level-agnostic: continue iff the top-8 cleaned-G "
        "mean entity score at t10 is below the frozen dev median). fixed10/15/20 are "
        f"{SCHEDULE_AWARE_NOTE}; they are shown for scale and never tested. "
        "✓ = 95 % paired-bootstrap CI excludes 0 or permutation p < .05 (slot permutation vs "
        "the mix, paired sign flip vs conf_two_stage); the direction says whether the U policy "
        "is better or worse.",
    ]
    for label, grp in methods.groupby("set", sort=False):
        head = sig[(sig["set"] == label) & (sig["comparator"] == "mix") & (sig["metric"] == "R50")
                   & (sig["column"] == "q")].iloc[0]  # fmt: skip
        lines += ["", f"## {label} (n = {int(grp['n'].iloc[0])})", "",
                  f"**Headline (plan primary): R50 policy − same-p mix = "
                  f"{head.diff_policy_minus_comparator:+.4f} [{head.ci_low:+.4f}, {head.ci_high:+.4f}], "
                  f"permutation p(better) = {head.perm_p_better:.3f}.**", "",
                  "| method | role | mean level | MAE vs R50-oracle | agree (R50-oracle) | R50 | "
                  "MAE vs relsg-oracle | agree (relsg-oracle) | relsg_gt |",
                  "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for r in grp.itertuples():
            r50 = fmt(r.R50) + (
                f" (perm95 {r.R50_perm95})"
                if isinstance(getattr(r, "R50_perm95", None), str)
                else ""
            )
            rel = fmt(r.relsg_gt) + (
                f" (perm95 {r.relsg_perm95})"
                if isinstance(getattr(r, "relsg_perm95", None), str)
                else ""
            )
            lines.append(f"| {r.method} | {r.role} | {r.mean_level:.2f} | {r.mae_vs_R50_oracle:.2f} | "
                         f"{r.agree_R50_oracle:.0%} | {r50} | {r.mae_vs_relsg_oracle:.2f} | "
                         f"{r.agree_relsg_oracle:.0%} | {rel} |")  # fmt: skip
        lines += ["", "| U policy vs | column | Δ (policy − comparator) [95% CI] | perm p (better / worse) | sig |",
                  "|---|---|---|---|---|"]  # fmt: skip
        for s in sig[sig["set"] == label].itertuples():
            col = f"{COLUMN_LABEL[s.column]} ({s.metric if s.column == 'q' else s.oracle_metric + '-oracle'})"
            perm = (
                "—"
                if np.isnan(s.perm_p_better)
                else f"{s.perm_p_better:.3f} / {s.perm_p_worse:.3f}"
            )
            mark = f"✓ {s.direction}" if s.significant else f"✗ ({s.direction})"
            lines.append(f"| {s.comparator} | {col} | {s.diff_policy_minus_comparator:+.4f} "
                         f"[{s.ci_low:+.4f}, {s.ci_high:+.4f}] | {perm} | {mark} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_dir = resolve(OUT_ROOT) / args.run_id
    method_out, sig_out = [], []
    for label, policy_name, conf_name, stage_name in SEED_SETS:
        stage_path = (
            resolve(stage_name)
            if stage_name.startswith("outputs/")
            else run_dir / stage_name
        )
        policy_path = run_dir / policy_name
        if not (stage_path.exists() and policy_path.exists()):
            print(f"skip {label}: input missing")
            continue
        conf_path = run_dir / conf_name
        conf = pd.read_csv(conf_path) if conf_path.exists() else None
        levels = seed_levels(
            pd.read_csv(stage_path), load_policy(pd.read_csv(policy_path)), conf
        )
        method_out += method_table(label, levels, PERM_REPEATS, 20261008)
        for metric in METRICS:
            for oracle_metric in METRICS:
                rows = significance_rows(levels, metric, oracle_metric, args.repeats)
                sig_out += [{"set": label, **r} for r in rows]
        print(f"{label}: n={len(levels.cont)} t20={int(levels.cont.sum())}")
    methods = pd.DataFrame(method_out)
    sig = pd.DataFrame(sig_out).drop_duplicates(
        ["set", "metric", "oracle_metric", "comparator", "column"]
    )
    # q rows do not depend on the oracle metric: keep one copy (R50 oracle).
    sig = sig[(sig["column"] != "q") | (sig["oracle_metric"] == PRIMARY_METRIC)]
    out = run_dir / "method_comparison_miru_style"
    methods.to_csv(out.with_suffix(".csv"), index=False)
    sig.to_csv(
        out.parent / "method_comparison_miru_style_significance.csv", index=False
    )
    out.with_suffix(".md").write_text(markdown(methods, sig))
    print(f"-> {out}.md")


if __name__ == "__main__":
    main()
