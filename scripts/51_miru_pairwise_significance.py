"""MIRU 100: pairwise MIRU-style significance vs level-agnostic comparators (zero calls).

Usage (after scripts/42_miru_gt_eval.py):
    uv run --project egtr python scripts/51_miru_pairwise_significance.py

Each method in {vlm_sgg_single_4b/9b, llm_sgg_text_4b/9b, fuse_mean, fuse_min,
hybrid_fuse_mean} vs each comparator in {llm_sgg_text_4b (new control), vlm_single_9b
(theta 0.70), node-confidence baseline, hybrid_4b (legacy)}: per-image |stop - stop_c09|,
R50(stop), relsg_gt(stop); paired bootstrap 2000 CI + paired sign-flip permutation.
All rows are in-sample; the fusion rules were designed on this same set (post-hoc).
Writes outputs/miru_gt_eval/miru_pairwise_significance.{csv,md}.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.pairwise_sig import (  # noqa: E402
    REPEATS,
    pairwise_rows,
    significance_markdown,
)
from milestone1.recovery_diagnostics import METRICS, quality_matrix  # noqa: E402

EVAL_DIR = "outputs/miru_gt_eval"
ORACLE_CSV = "outputs/metrics/topk_oracle/oracle_topk_per_image.csv"
METHODS = {
    "vlm_sgg_single_4b_k2": "in-sample (2x2: image + SGG, 4b)",
    "vlm_sgg_single_9b_k2": "in-sample (2x2: image + SGG, 9b)",
    "llm_sgg_text_4b_k2": "in-sample (2x2: text control, 4b)",
    "llm_sgg_text_9b_k2": "in-sample (2x2: text control, 9b)",
    "fuse_mean_vlmimg9b_text4b_k2": "POST-HOC (fusion designed on this set)",
    "fuse_min_vlmimg9b_text4b_k2": "POST-HOC (fusion designed on this set)",
    "hybrid_fuse_mean_vlmimg9b_text4b_k2": "POST-HOC (fusion designed on this set)",
}
COMPARATORS = {
    "llm_sgg_text_4b_k2": "text 4b (new control)",
    "vlm_single_9b_k2": "image-only 9b (θ 0.70)",
    "baseline": "node-confidence baseline",
    "hybrid_4b_k2": "hybrid 4b (legacy)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repeats", type=int, default=REPEATS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ev = resolve(EVAL_DIR)
    stops = pd.read_csv(ev / "eval" / "stops.csv")
    missing = [m for m in (*METHODS, *COMPARATORS) if m not in set(stops["method"])]
    if missing:
        raise SystemExit(f"methods missing from {ev / 'eval/stops.csv'}: {missing}")
    oracle = {int(r["image_id"]): int(r["stop_c09"])
              for r in pd.read_csv(resolve(ORACLE_CSV)).to_dict("records")}  # fmt: skip
    stage = pd.read_csv(ev / "stage_metrics.csv")
    qms = {m: quality_matrix(stage, m, "miru100") for m in METRICS}
    ids = [i for i in qms["R50"].ids if i in oracle]
    index = {i: k for k, i in enumerate(qms["R50"].ids)}
    sig = pairwise_rows(stops, list(METHODS), COMPARATORS, {"MIRU100": ids},
                        {m: qms[m].q for m in METRICS}, index, oracle, args.repeats)  # fmt: skip
    sig.insert(3, "status", sig["method"].map(METHODS))
    sig.to_csv(ev / "miru_pairwise_significance.csv", index=False)
    lines = [
        "# MIRU 100 — pairwise significance vs level-agnostic comparators",
        "",
        "IN-SAMPLE: all methods were run on these 100 images; the fusion rules (fuse_mean, "
        "fuse_min, hybrid_fuse_mean) were designed after seeing them (POST-HOC). Their "
        "pre-registered confirmation is the held-out 96 set. abs_err_c09 = |stop − stop_c09| "
        "(lower is better); R50 / relsg_gt = q(stop) (higher is better). ✓ = paired bootstrap "
        "95% CI excludes 0 or paired sign-flip permutation p < .05.",
        "",
        *significance_markdown(sig),
    ]
    (ev / "miru_pairwise_significance.md").write_text("\n".join(lines) + "\n")
    ticks = sig[sig["significant"]]
    print(f"{len(ticks)} ✓ cells of {len(sig)}")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(ticks[["method", "comparator", "column", "diff", "ci_low", "ci_high", "p_better",
                     "p_worse", "direction"]].round(3).to_string(index=False))  # fmt: skip


if __name__ == "__main__":
    main()
