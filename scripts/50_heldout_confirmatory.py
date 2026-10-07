"""Phase A held-out confirmatory table, significance and figure (zero calls).

Usage (after scripts/42_miru_gt_eval.py --dataset heldout):
    uv run --project egtr python scripts/50_heldout_confirmatory.py
Dry run on the MIRU set:
    uv run --project egtr python scripts/50_heldout_confirmatory.py --eval-dir outputs/miru_gt_eval \
        --oracle-csv outputs/metrics/topk_oracle/oracle_topk_per_image.csv

Reads <eval_dir>/{miru_method_table.csv, stage_metrics.csv, eval/stops.csv} and the
stop_c09 oracle; writes <eval_dir>/heldout_registered_table.{csv,md},
heldout_significance.csv and figures/heldout_quality_vs_level.{png,pdf}.

Significance (MIRU style): pre-registered methods vs the level-agnostic comparators
(text 4b, image-only 9b, node-confidence baseline), per image: |stop - stop_c09|,
R50(stop), relsg_gt(stop); paired bootstrap 2000 (95 % CI of the mean difference) and a
paired sign-flip permutation test 2000x; all 96 and the stop_c09 <= 18 subset.
Fixed levels / F are schedule-aware references (shown, never tested).
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.pairwise_sig import pairwise_rows, significance_markdown  # noqa: E402
from milestone1.recovery_diagnostics import (  # noqa: E402
    METRICS,
    quality_matrix,
    upper_hull,
)

PRIMARY = "hybrid_fuse_mean_vlmimg9b_text4b_k2"
SECONDARY = "fuse_min_vlmimg9b_text4b_k2"
COMPARATORS = {"llm_sgg_text_4b_k2": "text 4b", "vlm_single_9b_k2": "image-only 9b",
               "baseline": "node-confidence baseline"}  # fmt: skip
OTHER_ROWS = ("fuse_mean_vlmimg9b_text4b_k2", "hybrid_llm_sgg_text_4b_k2")
SAT_MAX = 18
REPEATS = 2000
SEED = 20261008
ALPHA = 0.05
NUM_LEVELS = 20
SCHEDULE_AWARE = ("schedule-aware reference (knows the level index; GT-posterior for F) "
                  "— not a comparator")  # fmt: skip
FIG_SIZE = (8.6, 6.4)
DPI = 200
STYLE = {PRIMARY: ("#1f4e79", "o", "hybrid_fuse_mean (PRIMARY)"),
         SECONDARY: ("#2e75b6", "D", "fuse_min (secondary)"),
         "fuse_mean_vlmimg9b_text4b_k2": ("#9dc3e6", "d", "fuse_mean"),
         "llm_sgg_text_4b_k2": ("#c55a11", "s", "text 4b"),
         "hybrid_llm_sgg_text_4b_k2": ("#f4b183", "s", "hybrid text 4b"),
         "vlm_single_9b_k2": ("#548235", "^", "image-only 9b"),
         "baseline": ("#7f7f7f", "v", "node-conf baseline")}  # fmt: skip


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--eval-dir", default="outputs/heldout_eval")
    p.add_argument(
        "--oracle-csv",
        default=None,
        help="default: <eval_dir>/oracle_topk_per_image.csv",
    )
    p.add_argument("--repeats", type=int, default=REPEATS)
    return p.parse_args()


def significance(stops, ids_all, ids_sat, q, index, oracle, repeats) -> pd.DataFrame:  # noqa: ANN001
    subsets = {"all": ids_all, f"stop_c09<={SAT_MAX}": ids_sat}
    return pairwise_rows(stops, (PRIMARY, SECONDARY), COMPARATORS, subsets, q, index, oracle,
                         repeats)  # fmt: skip


def registered_table(table: pd.DataFrame) -> pd.DataFrame:
    order = [PRIMARY, SECONDARY, *OTHER_ROWS, *COMPARATORS, "fixed_20"]
    t = table.set_index("method").reindex(
        [m for m in order if m in set(table["method"])]
    )
    role = {PRIMARY: "PRIMARY (pre-registered)", SECONDARY: "secondary (pre-registered)",
            "fixed_20": SCHEDULE_AWARE}  # fmt: skip
    t.insert(
        0,
        "role",
        [role.get(m, "comparator" if m in COMPARATORS else "other") for m in t.index],
    )
    return t.reset_index()


def fmt(v: Any, d: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def markdown(reg: pd.DataFrame, sig: pd.DataFrame, n: int, n_sat: int) -> str:
    lines = [
        "# Held-out confirmatory results (pre-registered 2026-10-08 00:40; no tuning)",
        "",
        f"n = {n} images (all) and {n_sat} with stop_c09 <= {SAT_MAX} (MIRU's selection rule). "
        "MAE = |stop - stop_c09| (EGTR-L20 top-K oracle). Δ legacy mix = q(stop) minus the "
        "adjacent two-level fixed mix at the method's mean stop (uses the level index; previous "
        "plan's registered comparison) with 95% CI; Δ F = point difference to the best fixed "
        f"mixture ({SCHEDULE_AWARE}).",
        "",
        "| method | role | mean stop | MAE (all) | MAE (≤18) | R50 | Δ R50 vs legacy mix [CI] | "
        "Δ R50 vs F | relsg | Δ relsg vs legacy mix [CI] | Δ relsg vs F |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reg.itertuples():
        lines.append(
            f"| {r.method} | {r.role} | {fmt(r.mean_stop, 2)} | {fmt(r.mae_c09, 2)} | "
            f"{fmt(r.mae_c09_sat, 2)} | {fmt(r.R50_stop)} | {fmt(getattr(r, 'R50_minus_eq_fixed_ci', None))} | "
            f"{fmt(getattr(r, 'R50_minus_F', None))} | {fmt(r.relsg_gt_stop)} | "
            f"{fmt(getattr(r, 'relsg_gt_minus_eq_fixed_ci', None))} | {fmt(getattr(r, 'relsg_gt_minus_F', None))} |"
        )  # fmt: skip
    lines += ["", "## Significance vs level-agnostic comparators (paired bootstrap 95% CI; "
              "paired sign-flip permutation)", "",
              "✓ = CI excludes 0 or permutation p < .05; direction = whether the pre-registered "
              "method is better (lower |stop − c09|, higher q).", ""]  # fmt: skip
    lines += significance_markdown(sig)
    return "\n".join(lines) + "\n"


def figure(stage: pd.DataFrame, stops: pd.DataFrame, ids_sets: dict[str, list[int]],
           out_stem: Path) -> None:  # fmt: skip
    fig, axes = plt.subplots(
        len(ids_sets), len(METRICS), figsize=FIG_SIZE, squeeze=False
    )
    t = np.arange(1, NUM_LEVELS + 1)
    for r, (label, ids) in enumerate(ids_sets.items()):
        for c, metric in enumerate(METRICS):
            ax = axes[r][c]
            qm = quality_matrix(stage, metric, label)
            index = {i: k for k, i in enumerate(qm.ids)}
            rows = np.array([index[i] for i in ids])
            f = qm.q[rows].mean(axis=0)
            hull = upper_hull(f)
            ax.plot(
                t,
                f,
                color="0.6",
                marker=".",
                lw=1,
                label="fixed level (schedule-aware ref.)",
            )
            ax.plot(t[hull], f[hull], color="0.3", ls="--", lw=0.9,
                    label="F (schedule-aware, GT-posterior ref.)")  # fmt: skip
            for method, (color, marker, mlabel) in STYLE.items():
                sel = stops[(stops["method"] == method) & stops["image_id"].isin(ids)]
                if sel.empty:
                    continue
                st = sel.set_index("image_id").loc[ids, "stop_t"].to_numpy(int)
                ax.scatter(st.mean(), qm.q[rows, st - 1].mean(), color=color, marker=marker,
                           s=46 if method == PRIMARY else 30, zorder=3, label=mlabel,
                           edgecolors="k" if method == PRIMARY else "none")  # fmt: skip
            ax.set_title(f"{label} — {metric}", fontsize=9)
            ax.set_xlabel("mean observed level (stop)", fontsize=8)
            ax.set_ylabel(f"mean {metric} at stop", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=DPI)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ev = resolve(args.eval_dir)
    oracle_path = (
        resolve(args.oracle_csv)
        if args.oracle_csv
        else ev / "oracle_topk_per_image.csv"
    )
    oracle = {
        int(r["image_id"]): int(r["stop_c09"])
        for r in pd.read_csv(oracle_path).to_dict("records")
    }
    stage = pd.read_csv(ev / "stage_metrics.csv")
    stops = pd.read_csv(ev / "eval" / "stops.csv")
    table = pd.read_csv(ev / "miru_method_table.csv")
    missing = [
        m for m in (PRIMARY, SECONDARY, *COMPARATORS) if m not in set(stops["method"])
    ]
    if missing:
        raise SystemExit(
            f"methods not available yet: {missing} (rerun script 42 when scores are complete)"
        )
    qms = {m: quality_matrix(stage, m, "eval") for m in METRICS}
    if qms["R50"].ids != qms["relsg_gt"].ids:
        raise SystemExit("R50 / relsg_gt complete-series sets differ")
    ids_all = [i for i in qms["R50"].ids if i in oracle]
    ids_sat = [i for i in ids_all if oracle[i] <= SAT_MAX]
    index = {i: k for k, i in enumerate(qms["R50"].ids)}
    q = {m: qms[m].q for m in METRICS}
    sig = significance(stops, ids_all, ids_sat, q, index, oracle, args.repeats)
    reg = registered_table(table)
    reg.to_csv(ev / "heldout_registered_table.csv", index=False)
    sig.to_csv(ev / "heldout_significance.csv", index=False)
    (ev / "heldout_registered_table.md").write_text(
        markdown(reg, sig, len(ids_all), len(ids_sat))
    )
    figure(stage, stops, {f"all (n={len(ids_all)})": ids_all,
                          f"stop_c09<={SAT_MAX} (n={len(ids_sat)})": ids_sat},
           ev / "figures" / "heldout_quality_vs_level")  # fmt: skip
    print(f"-> {ev / 'heldout_registered_table.md'}")


if __name__ == "__main__":
    main()
