"""Paired per-image significance of stop methods (MIRU-style ✓/✗ tables), zero calls.

For method A vs comparator B on the same images: per-image differences of
|stop - stop_c09| (lower is better) and q(stop) for R50 / relsg_gt (higher is better);
paired bootstrap of the mean difference (95 % CI) and a paired sign-flip permutation
test. ✓ = CI excludes 0 or permutation p < alpha, with the direction reported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from milestone1.recovery_diagnostics import METRICS

REPEATS = 2000
SEED = 20261008
ALPHA = 0.05
COLUMNS = (
    ("abs_err_c09", False),
    ("R50", True),
    ("relsg_gt", True),
)  # (column, higher better)


def per_image(stops: pd.DataFrame, method: str, ids: Sequence[int], q: Mapping[str, np.ndarray],
              index: Mapping[int, int], oracle: Mapping[int, int]) -> dict[str, np.ndarray]:  # fmt: skip
    """|stop - stop_c09| and q(stop) per image for one method (ids in the given order)."""
    st = stops[stops["method"] == method].set_index("image_id").loc[list(ids), "stop_t"]
    st = st.to_numpy(int)
    rows = np.array([index[i] for i in ids])
    out = {"abs_err_c09": np.abs(st - np.array([oracle[i] for i in ids])).astype(float),
           "stop": st.astype(float)}  # fmt: skip
    for m in METRICS:
        out[m] = q[m][rows, st - 1]
    return out


def paired_test(
    d: np.ndarray, repeats: int = REPEATS, seed: int = SEED
) -> dict[str, float]:
    """Mean difference, bootstrap 95 % CI, sign-flip one-sided p-values (up / down)."""
    rng = np.random.default_rng(seed)
    n = len(d)
    boot = d[rng.integers(0, n, size=(repeats, n))].mean(axis=1)
    lo, hi = np.quantile(boot, [ALPHA / 2, 1 - ALPHA / 2])
    signs = np.random.default_rng(seed + 1).choice([-1.0, 1.0], size=(repeats, n))
    null = (signs * d).mean(axis=1)
    obs = float(d.mean())
    return {"diff": obs, "ci_low": float(lo), "ci_high": float(hi),
            "p_up": float(np.mean(null >= obs - 1e-12)),
            "p_down": float(np.mean(null <= obs + 1e-12))}  # fmt: skip


def pairwise_rows(stops: pd.DataFrame, methods: Sequence[str], comparators: Mapping[str, str],
                  subsets: Mapping[str, Sequence[int]], q: Mapping[str, np.ndarray],
                  index: Mapping[int, int], oracle: Mapping[int, int],
                  repeats: int = REPEATS) -> pd.DataFrame:  # fmt: skip
    """One row per subset x method x comparator x column (method == comparator skipped)."""
    rows = []
    for subset, ids in subsets.items():
        needed = set(methods) | set(comparators)
        vals = {m: per_image(stops, m, ids, q, index, oracle) for m in needed}
        for method in methods:
            for comp, comp_label in comparators.items():
                if comp == method:
                    continue
                for column, higher_better in COLUMNS:
                    t = paired_test(vals[method][column] - vals[comp][column], repeats)
                    better_p, worse_p = ((t["p_up"], t["p_down"]) if higher_better
                                         else (t["p_down"], t["p_up"]))  # fmt: skip
                    better = t["diff"] > 0 if higher_better else t["diff"] < 0
                    sig = (t["ci_low"] > 0 or t["ci_high"] < 0) or min(
                        better_p, worse_p
                    ) < ALPHA
                    rows.append({"subset": subset, "n": len(ids), "method": method,
                                 "comparator": comp, "comparator_label": comp_label,
                                 "column": column, **t, "p_better": better_p,
                                 "p_worse": worse_p,
                                 "direction": "better" if better else
                                 ("worse" if t["diff"] != 0 else "equal"),
                                 "significant": bool(sig)})  # fmt: skip
    return pd.DataFrame(rows)


def significance_markdown(sig: pd.DataFrame) -> list[str]:
    lines = ["| subset | method | vs | column | Δ (method − comparator) [95% CI] | "
             "p better / worse | sig |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for s in sig.to_dict("records"):
        mark = f"✓ {s['direction']}" if s["significant"] else f"✗ ({s['direction']})"
        lines.append(f"| {s['subset']} (n={s['n']}) | {s['method']} | {s['comparator_label']} | "
                     f"{s['column']} | {s['diff']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}] | "
                     f"{s['p_better']:.3f} / {s['p_worse']:.3f} | {mark} |")  # fmt: skip
    return lines
