"""Stage-1b Bayesian saturating-curve stop: relative-epsilon sweep.

Sweeps relative epsilon c in {0.10, 0.15, 0.20, 0.25} with two priors
(narrow and wide) and compares against conf_convergence heuristic.

Run:
  uv run --project egtr python scripts/18_bayesian_stop_stage1.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.bayesian_stop import bayesian_stop, fit_mle_curve  # noqa: E402, I001
from milestone1.config import load_config, resolve  # noqa: E402
from milestone1.convergence import convergence_stop  # noqa: E402


# -------------------------------------------------------------------
# Utilities (reused from script 16)
# -------------------------------------------------------------------


def load_csv_dict(path: Path) -> list[dict]:
    """Load CSV as list of dicts."""
    import csv

    with open(path) as f:
        return list(csv.DictReader(f))


def median_val(vals: list[float]) -> float:
    """Simple median."""
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def compute_method_stats(
    method_stops: list[int],
    oracle_stops: list[int],
    max_level: int,
) -> dict:
    """Compute comparison statistics for a method vs an oracle reference."""
    errors = [m - o for m, o in zip(method_stops, oracle_stops, strict=True)]
    abs_errors = [abs(e) for e in errors]
    n = len(errors)

    mean_stop = float(np.mean(method_stops))
    frac_stop = sum(1 for s in method_stops if s < max_level) / n
    mean_ae = float(np.mean(abs_errors))
    med_ae = median_val([float(e) for e in abs_errors])
    early = sum(1 for e in errors if e < 0) / n
    late = sum(1 for e in errors if e > 0) / n

    return {
        "mean_stop": round(mean_stop, 2),
        "frac_stop": round(frac_stop, 3),
        "MAE": round(mean_ae, 2),
        "MedAE": round(med_ae, 1),
        "early_rate": round(early, 3),
        "late_rate": round(late, 3),
    }


# -------------------------------------------------------------------
# Synthetic sanity check
# -------------------------------------------------------------------


def sanity_check() -> None:
    """Feed a known synthetic series and verify posterior recovery."""
    print("=" * 60)
    print("SANITY CHECK: synthetic saturating curve")
    print("=" * 60)

    true_s_inf = 0.9
    true_lam = 0.4
    true_s0 = 0.1
    rng = np.random.default_rng(42)

    t_vals = list(range(1, 21))
    clean = [
        true_s_inf - (true_s_inf - true_s0) * math.exp(-true_lam * t)
        for t in t_vals
    ]
    noisy = [v + rng.normal(0, 0.005) for v in clean]

    # Fit MLE
    mle = fit_mle_curve(noisy)
    print(
        f"  MLE fit: s_inf={mle['s_inf']:.3f}, lam={mle['lam']:.3f}, "
        f"s0={mle['s0']:.3f}, sigma={mle['sigma']:.4f}"
    )

    # Absolute mode
    result_abs = bayesian_stop(
        noisy,
        prior={"s_inf": (0.8, 0.15), "s0": (0.1, 0.05),
               "lam": (math.log(0.4), 0.6)},
        epsilon=0.03, delta=0.90, min_level=3, max_level=20,
    )
    post_mean, post_std = result_abs["posterior_s_inf"]
    print(f"  [absolute] s_inf={post_mean:.4f}+/-{post_std:.4f}, "
          f"stop={result_abs['stop_level']}")

    assert abs(post_mean - true_s_inf) < 0.15, (
        f"Posterior mean {post_mean:.3f} too far from true {true_s_inf}"
    )
    assert result_abs["stop_level"] < 20, (
        f"Should stop before max_level on clean data, got {result_abs['stop_level']}"
    )

    # Relative mode (c=0.10 => stop when gap < 0.10*s_inf = 0.09)
    result_rel = bayesian_stop(
        noisy,
        prior={"s_inf": (0.8, 0.15), "s0": (0.1, 0.05),
               "lam": (math.log(0.4), 0.6)},
        epsilon=0.10, delta=0.90, min_level=3, max_level=20,
        relative_epsilon=True,
    )
    post_mean_r, post_std_r = result_rel["posterior_s_inf"]
    print(f"  [relative c=0.10] s_inf={post_mean_r:.4f}+/-{post_std_r:.4f}, "
          f"stop={result_rel['stop_level']}")

    assert result_rel["stop_level"] < 20, (
        f"Relative mode should stop on clean data, got {result_rel['stop_level']}"
    )
    print("  [PASS] Both modes work on synthetic data.\n")


# -------------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------------

NARROW_PRIOR: dict[str, tuple[float, float]] = {
    "s_inf": (0.39, 0.06),
    "s0": (0.10, 0.03),
    "lam": (-1.9, 0.5),
}

WIDE_PRIOR: dict[str, tuple[float, float]] = {
    "s_inf": (0.45, 0.10),
    "s0": (0.10, 0.04),
    "lam": (-1.9, 0.6),
}

C_VALUES = [0.10, 0.15, 0.20, 0.25]


def main() -> None:
    sanity_check()

    cfg = load_config()
    max_level: int = cfg["resolution_levels"]["num_levels"]  # 20

    metrics_root = resolve(cfg["paths"]["metrics"])
    selected_path = resolve(cfg["paths"]["selected_images"])
    with open(selected_path) as f:
        image_ids: list[int] = json.load(f)

    n_images = len(image_ids)
    print("=" * 60)
    print(f"STAGE-1b RELATIVE-EPSILON SWEEP: {n_images} images")
    print("=" * 60)

    # -----------------------------------------------------------------
    # Load reference oracles
    # -----------------------------------------------------------------
    rel_oracle_rows = load_csv_dict(metrics_root / "relative_oracle_stop.csv")
    rel_oracle_map = {
        int(r["image_id"]): int(r["rel_oracle_0.8"]) for r in rel_oracle_rows
    }
    gt_oracle_rows = load_csv_dict(metrics_root / "oracle_stop.csv")
    gt_oracle_map = {
        int(r["image_id"]): int(float(r["oracle_0.5"])) for r in gt_oracle_rows
    }

    rel_oracle_stops = [rel_oracle_map[i] for i in image_ids]
    gt_oracle_stops = [gt_oracle_map[i] for i in image_ids]

    # -----------------------------------------------------------------
    # Load conf signal for all images
    # -----------------------------------------------------------------
    all_metrics_rows = load_csv_dict(metrics_root / "all_metrics.csv")
    conf_map: dict[tuple[int, int], float] = {}
    for r in all_metrics_rows:
        iid = int(r["image_id"])
        lvl = int(r["level"])
        val = float(r["mean_object_confidence"])
        conf_map[(iid, lvl)] = val

    conf_series_all: dict[int, list[float]] = {}
    for img_id in image_ids:
        conf_series_all[img_id] = [
            conf_map.get((img_id, lvl), float("nan"))
            for lvl in range(1, max_level + 1)
        ]

    # -----------------------------------------------------------------
    # Convergence baseline (run once)
    # -----------------------------------------------------------------
    convergence_stops: list[int] = []
    for img_id in image_ids:
        cstop = convergence_stop(
            conf_series_all[img_id],
            epsilon=0.02, k=3, min_level=5, max_level=max_level,
        )
        convergence_stops.append(cstop)

    # -----------------------------------------------------------------
    # Sweep: c x prior
    # -----------------------------------------------------------------
    priors = {
        "narrow(0.39,0.06)": NARROW_PRIOR,
        "wide(0.45,0.10)": WIDE_PRIOR,
    }

    # Storage: method_label -> list[int] of stops
    all_methods: dict[str, list[int]] = {
        "conf_convergence": convergence_stops,
    }

    for prior_label, prior in priors.items():
        for c in C_VALUES:
            label = f"bayes_c{c:.2f}_{prior_label}"
            stops: list[int] = []
            print(f"\n  Running {label} ...")
            for idx, img_id in enumerate(image_ids):
                bres = bayesian_stop(
                    conf_series_all[img_id],
                    prior=prior,
                    epsilon=c,
                    delta=0.90,
                    min_level=5,
                    max_level=max_level,
                    relative_epsilon=True,
                )
                stops.append(bres["stop_level"])
                if idx < 3:
                    pm, ps = bres["posterior_s_inf"]
                    print(
                        f"    img={img_id}: stop={bres['stop_level']:2d}, "
                        f"s_inf={pm:.3f}+/-{ps:.3f}"
                    )
            all_methods[label] = stops

    # -----------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON TABLE (all methods x both oracles)")
    print("=" * 70)

    oracles = {
        "vs RelOracle(0.8)": rel_oracle_stops,
        "vs GTOracle(0.5)": gt_oracle_stops,
    }

    header = (
        f"  {'Method':<35s} {'Oracle':<18s} "
        f"{'MeanStop':>8s} {'Frac%':>6s} "
        f"{'MAE':>6s} {'MedAE':>6s} {'Early':>6s} {'Late':>6s}"
    )
    print(header)
    print("  " + "-" * 95)

    results_table: list[dict] = []
    for mname, mstops in all_methods.items():
        for oname, ostops in oracles.items():
            stats = compute_method_stats(mstops, ostops, max_level)
            print(
                f"  {mname:<35s} {oname:<18s} "
                f"{stats['mean_stop']:>8} {stats['frac_stop']:>6} "
                f"{stats['MAE']:>6} {stats['MedAE']:>6} "
                f"{stats['early_rate']:>6} {stats['late_rate']:>6}"
            )
            results_table.append({"method": mname, "oracle": oname, **stats})
        # Blank line between methods for readability
        if mname != list(all_methods.keys())[-1]:
            print()

    # -----------------------------------------------------------------
    # Prior comparison summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PRIOR COMPARISON (narrow vs wide, MAE vs RelOracle)")
    print("=" * 70)

    for c in C_VALUES:
        narrow_key = f"bayes_c{c:.2f}_narrow(0.39,0.06)"
        wide_key = f"bayes_c{c:.2f}_wide(0.45,0.10)"
        nr = next(
            r for r in results_table
            if r["method"] == narrow_key and "RelOracle" in r["oracle"]
        )
        wr = next(
            r for r in results_table
            if r["method"] == wide_key and "RelOracle" in r["oracle"]
        )
        winner = "narrow" if nr["MAE"] <= wr["MAE"] else "wide"
        print(
            f"  c={c:.2f}: narrow MAE={nr['MAE']} frac={nr['frac_stop']} | "
            f"wide MAE={wr['MAE']} frac={wr['frac_stop']} -> {winner}"
        )

    # -----------------------------------------------------------------
    # GO / NO-GO verdict (best c from the better prior)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GO / NO-GO VERDICT")
    print("=" * 70)

    # Find the best Bayesian variant vs RelOracle
    bayes_rows = [
        r for r in results_table
        if r["method"].startswith("bayes_") and "RelOracle" in r["oracle"]
    ]
    # Sort by MAE, then prefer higher frac_stop
    bayes_rows.sort(key=lambda r: (r["MAE"], -r["frac_stop"]))
    best = bayes_rows[0]

    conv_rel = next(
        r for r in results_table
        if r["method"] == "conf_convergence" and "RelOracle" in r["oracle"]
    )

    print(f"  Best Bayesian: {best['method']}")
    print(f"    MAE vs RelOracle(0.8) = {best['MAE']}")
    print(f"    frac_stop = {best['frac_stop']}")
    print(f"    early_rate = {best['early_rate']}")
    print(f"    late_rate = {best['late_rate']}")
    print("  Baseline (conf_convergence):")
    print(f"    MAE vs RelOracle(0.8) = {conv_rel['MAE']}")
    print(f"    frac_stop = {conv_rel['frac_stop']}")
    print(f"    early_rate = {conv_rel['early_rate']}")

    mae_ok = best["MAE"] < conv_rel["MAE"]
    frac_ok = best["frac_stop"] > 0.80
    early_better = best["early_rate"] < conv_rel["early_rate"] * 0.7

    if frac_ok and mae_ok and early_better:
        verdict = "GO"
        reason = (
            f"frac_stop={best['frac_stop']} > 0.80, "
            f"MAE={best['MAE']} < conv {conv_rel['MAE']}, "
            f"early_rate={best['early_rate']} substantially < conv {conv_rel['early_rate']}"
        )
    elif best["frac_stop"] < 0.50 or best["MAE"] > 6.0:
        verdict = "NO-GO"
        reason = f"frac_stop={best['frac_stop']} or MAE={best['MAE']} out of range"
    else:
        verdict = "BORDERLINE"
        parts = []
        if not frac_ok:
            parts.append(f"frac_stop={best['frac_stop']} <= 0.80")
        if not mae_ok:
            parts.append(f"MAE={best['MAE']} >= conv {conv_rel['MAE']}")
        if not early_better:
            parts.append(
                f"early_rate={best['early_rate']} not substantially < "
                f"conv {conv_rel['early_rate']}"
            )
        reason = "; ".join(parts)

    print(f"\n  Verdict: **{verdict}** ({reason})")

    # Show the best c for each oracle
    print("\n  Best c per oracle:")
    for oname in oracles:
        orows = [
            r for r in results_table
            if r["method"].startswith("bayes_") and r["oracle"] == oname
        ]
        orows.sort(key=lambda r: (r["MAE"], -r["frac_stop"]))
        ob = orows[0]
        print(f"    {oname}: {ob['method']} (MAE={ob['MAE']}, frac={ob['frac_stop']})")

    print("\nDone.")


if __name__ == "__main__":
    main()
