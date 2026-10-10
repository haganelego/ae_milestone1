"""Plan §25.6 evaluation shared by V6-A and V6-B (no model calls).

Sets: dev visual subset (102: accuracy and mean stop level) and dev
reader-wrong (99: firing). A question *fires* when the stop rule is met at some
level 1..20; the forced L20 stop is not a firing (§23).

Stop rules on a score matrix s (questions x levels) and an optional boolean
gate g (same shape):
- k = 1: first ℓ with s(ℓ) >= θ and g(ℓ).
- k = 2: first ℓ >= 2 with s >= θ at ℓ-1 and ℓ, and g at both levels
  (gate_mode "both", primary) or at either level (gate_mode "either", logged).

θ selection (§22): accuracy >= target on the 102, minimum mean level; ties ->
higher accuracy, larger θ. Out-of-fold (§25.6): the images of 102 ∪ 99 are
split 5-fold (image-grouped, seeds 0..9, same assignment as the d876011
cross-fit); θ is chosen on the train-fold 102 only and applied to the held-out
102 and to the 99 of the held-out images.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .evaluate import cluster_bootstrap_indices

N_LEVELS = 20
GRID: tuple[float, ...] = (*(round(0.01 * i, 2) for i in range(1, 101)), math.inf)
TARGETS = (0.90, 0.95, 0.97, 0.99)
PRIMARY_TARGET = 0.95
BOUNDS = (14, 15, 16, 17)
FOLDS, SEEDS = 5, 10
TOLERANCE = 1e-12
PERCENTILES = (2.5, 97.5)
# §25.11: two V6-C primaries -> Bonferroni 97.5% two-sided CI for the firing check
PERCENTILES_BONFERRONI = (1.25, 98.75)
# firing-level bands on the 99: plan §25.11 bands and the team-lead bands
PLAN_BANDS = ((1, 5), (6, 10), (11, 15), (16, 19), (20, 20))
LEAD_BANDS = ((1, 5), (6, 9), (10, 15), (16, 19))
GATE_BOTH, GATE_EITHER = "both", "either"


@dataclass(frozen=True)
class Policy:
    name: str
    score: str
    gate: str | None = None
    k: int = 1
    gate_mode: str = GATE_BOTH
    role: str = "logged"


@dataclass
class SetArrays:
    """One question set as level matrices (rows sorted by question_id)."""

    qids: list[str]
    groups: list[str]
    correct: np.ndarray
    attribute: np.ndarray
    relate: np.ndarray
    columns: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.qids)


def wide(table: pd.DataFrame, column: str) -> np.ndarray:
    pivot = table.pivot(index="question_id", columns="level", values=column)
    return pivot.sort_index()[list(range(1, N_LEVELS + 1))].to_numpy()


def build_arrays(table: pd.DataFrame, columns: Sequence[str]) -> SetArrays:
    first = table.sort_values("level").groupby("question_id").first().sort_index()
    return SetArrays(
        qids=[str(q) for q in first.index],
        groups=[str(g) for g in first["group_id"]],
        correct=wide(table, "reader_correct").astype(bool),
        attribute=first["attribute"].astype(str).to_numpy(),
        relate=first["relate_count"].astype(int).to_numpy(),
        columns={c: wide(table, c) for c in columns},
    )


# ---- stop rule --------------------------------------------------------------------------
def met_matrix(
    score: np.ndarray, gate: np.ndarray | None, theta: float, k: int, mode: str
) -> np.ndarray:
    above = score >= theta
    g = np.ones_like(above) if gate is None else gate.astype(bool)
    if k == 1:
        return above & g
    if k != 2:
        raise ValueError(f"k must be 1 or 2, got {k}")
    met = np.zeros_like(above)
    both = g[:, 1:] & g[:, :-1] if mode == GATE_BOTH else g[:, 1:] | g[:, :-1]
    met[:, 1:] = above[:, 1:] & above[:, :-1] & both
    return met


def stops(
    score: np.ndarray,
    gate: np.ndarray | None,
    theta: float,
    k: int = 1,
    mode: str = GATE_BOTH,
) -> tuple[np.ndarray, np.ndarray]:
    """(stop level 1..L, fired) per question; unmet -> forced L, not fired."""
    met = met_matrix(score, gate, theta, k, mode)
    fired = met.any(axis=1)
    level = np.where(fired, met.argmax(axis=1) + 1, score.shape[1])
    return level, fired


def policy_inputs(
    arr: SetArrays, policy: Policy
) -> tuple[np.ndarray, np.ndarray | None]:
    score = arr.columns[policy.score].astype(float)
    gate = None if policy.gate is None else arr.columns[policy.gate].astype(bool)
    return score, gate


@dataclass
class GridStops:
    """Stop level / fired / hit for every θ of the grid (rows = θ)."""

    level: np.ndarray
    fired: np.ndarray
    hit: np.ndarray


def grid_stops(arr: SetArrays, policy: Policy) -> GridStops:
    score, gate = policy_inputs(arr, policy)
    levels, fires = [], []
    for theta in GRID:
        lv, fi = stops(score, gate, theta, policy.k, policy.gate_mode)
        levels.append(lv)
        fires.append(fi)
    level = np.array(levels)
    hit = arr.correct[np.arange(arr.n)[None, :], level - 1]
    return GridStops(level, np.array(fires), hit)


def select_index(
    gs: GridStops, target: float, rows: np.ndarray | None = None
) -> int | None:
    """Index into GRID of the §22 selection on the (subset of) 102 questions."""
    cols = slice(None) if rows is None else rows
    acc = gs.hit[:, cols].mean(axis=1)
    mean = gs.level[:, cols].mean(axis=1)
    best, best_key = None, None
    for i, theta in enumerate(GRID):
        if acc[i] < target - TOLERANCE:
            continue
        key = (mean[i], -acc[i], -(theta if math.isfinite(theta) else 9.0))
        if best_key is None or key < best_key:
            best, best_key = i, key
    return best


# ---- in-sample ----------------------------------------------------------------------------
def curve_frame(policy: Policy, dev: GridStops, wrong: GridStops) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": policy.name,
            "theta": list(GRID),
            "accuracy": dev.hit.mean(axis=1),
            "mean_level": dev.level.mean(axis=1),
            "stop20_rate": (dev.level == N_LEVELS).mean(axis=1),
            "stopped_but_wrong": (dev.fired & ~dev.hit).sum(axis=1),
            "firing_99": wrong.fired.mean(axis=1),
        }
    )


def frontier(
    curve: pd.DataFrame, target: float, bounds: Sequence[int] = BOUNDS
) -> dict:
    """Minimum firing over θ with accuracy >= target and mean level <= bound."""
    ok = curve[curve["accuracy"] >= target - TOLERANCE]
    out = {}
    for b in bounds:
        sub = ok[ok["mean_level"] <= b + TOLERANCE]
        out[f"<={b}"] = float(sub["firing_99"].min()) if len(sub) else None
    return out


def ci(
    values: np.ndarray,
    indices: Sequence[np.ndarray],
    percentiles: Sequence[float] = PERCENTILES,
) -> list[float]:
    draws = [float(values[i].mean()) for i in indices]
    return [float(np.percentile(draws, p)) for p in percentiles]


def band_rates(
    fired: np.ndarray, fire_level: np.ndarray, prefix: str = "fire"
) -> dict[str, float]:
    """Fraction of ALL questions that fire within each level band (both band sets)."""
    out = {}
    for lo, hi in dict.fromkeys(PLAN_BANDS + LEAD_BANDS):
        hit = fired & (fire_level >= lo) & (fire_level <= hi)
        out[f"{prefix}_L{lo}_{hi}"] = float(hit.mean()) if len(hit) else 0.0
    return out


@dataclass
class Bootstrap:
    dev: list[np.ndarray]
    wrong: list[np.ndarray]


def make_bootstrap(
    dev: SetArrays, wrong: SetArrays, repeats: int, seed: int
) -> Bootstrap:
    return Bootstrap(
        cluster_bootstrap_indices(dev.groups, repeats, seed),
        cluster_bootstrap_indices(wrong.groups, repeats, seed),
    )


def paired_diff(
    a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray], boot: Bootstrap
) -> dict[str, float | list[float]]:
    """a - b for level / correct (102 clusters) and fired (99 clusters)."""
    dl = a["level"] - b["level"]
    dc = a["hit"].astype(float) - b["hit"].astype(float)
    df = a["fired"].astype(float) - b["fired"].astype(float)
    return {
        "d_mean_level": float(dl.mean()),
        "d_mean_level_ci": ci(dl, boot.dev),
        "d_accuracy": float(dc.mean()),
        "d_accuracy_ci": ci(dc, boot.dev),
        "d_firing": float(df.mean()),
        "d_firing_ci": ci(df, boot.wrong),
        "d_firing_ci975": ci(df, boot.wrong, PERCENTILES_BONFERRONI),
    }


def insample(
    policies: Sequence[Policy],
    dev: SetArrays,
    wrong: SetArrays,
    comparators: Mapping[str, Sequence[str]],
    boot: Bootstrap,
    targets: Sequence[float] = TARGETS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    """Target table, curves, frontier, and the selected stops per (policy, target)."""
    grids = {p.name: (grid_stops(dev, p), grid_stops(wrong, p)) for p in policies}
    curves = pd.concat([curve_frame(p, *grids[p.name]) for p in policies])
    selected: dict[str, dict] = {}
    rows, fronts = [], []
    for p in policies:
        gd, gw = grids[p.name]
        curve = curves[curves["method"] == p.name]
        for t in targets:
            fronts.append({"method": p.name, "target": t, **frontier(curve, t)})
            i = select_index(gd, t)
            if i is None:
                rows.append({"method": p.name, "target": t, "feasible": False})
                continue
            sel = {
                "theta": GRID[i],
                "level": gd.level[i],
                "hit": gd.hit[i],
                "fired_dev": gd.fired[i],
                "fired": gw.fired[i],
                "fire_level_99": np.where(gw.fired[i], gw.level[i], 0),
            }
            selected[f"{p.name}@{t:.2f}"] = sel
            rows.append(
                {
                    "method": p.name,
                    "role": p.role,
                    "score": p.score,
                    "gate": p.gate or "none",
                    "k": p.k,
                    "gate_mode": p.gate_mode if p.gate and p.k == 2 else "",
                    "target": t,
                    "feasible": True,
                    "theta": GRID[i],
                    "accuracy": float(gd.hit[i].mean()),
                    "mean_level": float(gd.level[i].mean()),
                    "stop20_rate": float((gd.level[i] == N_LEVELS).mean()),
                    "stopped_but_wrong": int((gd.fired[i] & ~gd.hit[i]).sum()),
                    "firing_99": float(gw.fired[i].mean()),
                    "n_fired_99": int(gw.fired[i].sum()),
                    **band_rates(gw.fired[i], gw.level[i]),
                }
            )
    table = pd.DataFrame(rows)
    pairs = []
    for name, refs in comparators.items():
        for ref in refs:
            for t in targets:
                a, b = selected.get(f"{name}@{t:.2f}"), selected.get(f"{ref}@{t:.2f}")
                if a is None or b is None:
                    continue
                pairs.append(
                    {"method": name, "vs": ref, "target": t, **paired_diff(a, b, boot)}
                )
    return table, curves, pd.DataFrame(fronts), {"selected": selected, "pairs": pairs}


# ---- out-of-fold --------------------------------------------------------------------------
def union_folds(
    dev_groups: Sequence[str], wrong_groups: Sequence[str], seed: int, k: int = FOLDS
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.array(sorted(set(dev_groups) | set(wrong_groups)))
    perm = np.random.default_rng(seed).permutation(len(unique))
    fold = {g: i % k for i, g in enumerate(unique[perm])}
    return (
        np.array([fold[g] for g in dev_groups]),
        np.array([fold[g] for g in wrong_groups]),
    )


def out_of_fold(
    policies: Sequence[Policy],
    dev: SetArrays,
    wrong: SetArrays,
    targets: Sequence[float] = TARGETS,
    seeds: int = SEEDS,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Seed-averaged OOF metrics and per-question seed means (level, hit, fired)."""
    rows, per_q = [], {}
    folds = [union_folds(dev.groups, wrong.groups, s) for s in range(seeds)]
    for p in policies:
        gd, gw = grid_stops(dev, p), grid_stops(wrong, p)
        for t in targets:
            lv = np.zeros((seeds, dev.n))
            hit = np.zeros((seeds, dev.n))
            fired = np.zeros((seeds, wrong.n))
            fire_lv = np.zeros((seeds, wrong.n), dtype=int)
            thetas = []
            for s, (fd, fw) in enumerate(folds):
                for f in range(FOLDS):
                    i = select_index(gd, t, fd != f)
                    i = len(GRID) - 1 if i is None else i  # θ = ∞ (never stop)
                    thetas.append(GRID[i])
                    te, tw = fd == f, fw == f
                    lv[s, te] = gd.level[i, te]
                    hit[s, te] = gd.hit[i, te]
                    fired[s, tw] = gw.fired[i, tw]
                    fire_lv[s, tw] = gw.level[i, tw]
            finite = [x for x in thetas if math.isfinite(x)]
            rows.append(
                {
                    "method": p.name,
                    "role": p.role,
                    "target": t,
                    "oof_accuracy": float(hit.mean()),
                    "oof_accuracy_min_seed": float(hit.mean(axis=1).min()),
                    "oof_mean_level": float(lv.mean()),
                    "oof_firing_99": float(fired.mean()),
                    "seeds_meeting_target": int(
                        (hit.mean(axis=1) >= t - TOLERANCE).sum()
                    ),
                    "theta_median": float(np.median(finite)) if finite else math.inf,
                    "theta_inf_folds": len(thetas) - len(finite),
                    "oof_stop20_rate": float((lv == N_LEVELS).mean()),
                    **band_rates(
                        fired.astype(bool).ravel(), fire_lv.ravel(), "oof_fire"
                    ),
                }
            )
            per_q[f"{p.name}@{t:.2f}"] = {
                "level": lv.mean(axis=0),
                "hit": hit.mean(axis=0),
                "fired": fired.mean(axis=0),
            }
    return pd.DataFrame(rows), per_q


def oof_pairs(
    per_q: Mapping[str, Mapping[str, np.ndarray]],
    comparators: Mapping[str, Sequence[str]],
    boot: Bootstrap,
    targets: Sequence[float] = TARGETS,
) -> pd.DataFrame:
    rows = []
    for name, refs in comparators.items():
        for ref in refs:
            for t in targets:
                a, b = per_q.get(f"{name}@{t:.2f}"), per_q.get(f"{ref}@{t:.2f}")
                if a is None or b is None:
                    continue
                rows.append(
                    {"method": name, "vs": ref, "target": t, **paired_diff(a, b, boot)}
                )
    return pd.DataFrame(rows)


def pass_rule(
    oof: pd.DataFrame,
    pairs: pd.DataFrame,
    method: str,
    refs: Sequence[str],
    pans_ref: str = "direct_pans",
    target: float = PRIMARY_TARGET,
    min_accuracy: float = 0.93,
    level_margin: float = 0.5,
    ci_key: str = "d_firing_ci",
) -> dict[str, object]:
    """§25.7: (i) firing diff CI upper < 0 vs every ref; (ii) OOF acc >= 0.93;
    (iii) OOF mean level <= OOF mean level of direct_pans + 0.5."""
    row = oof[(oof["method"] == method) & np.isclose(oof["target"], target)].iloc[0]
    ref_level = oof[(oof["method"] == pans_ref) & np.isclose(oof["target"], target)][
        "oof_mean_level"
    ].iloc[0]
    checks = {}
    for ref in refs:
        pr = pairs[
            (pairs["method"] == method)
            & (pairs["vs"] == ref)
            & np.isclose(pairs["target"], target)
        ].iloc[0]
        checks[f"firing_ci_upper_lt0_vs_{ref}"] = bool(pr[ci_key][1] < 0)
    checks["oof_accuracy_ge_0.93"] = bool(row["oof_accuracy"] >= min_accuracy)
    checks[f"oof_mean_level_le_{pans_ref}+0.5"] = bool(
        row["oof_mean_level"] <= ref_level + level_margin + TOLERANCE
    )
    return {
        "method": method,
        "refs": list(refs),
        "oof_accuracy": float(row["oof_accuracy"]),
        "oof_mean_level": float(row["oof_mean_level"]),
        "oof_firing_99": float(row["oof_firing_99"]),
        "ref_oof_mean_level": float(ref_level),
        "ci_key": ci_key,
        "checks": checks,
        "passed": all(checks.values()),
    }


# ---- breakdown ----------------------------------------------------------------------------
def breakdown(
    selected: Mapping[str, Mapping[str, np.ndarray]],
    names: Sequence[str],
    dev: SetArrays,
    wrong: SetArrays,
    target: float = PRIMARY_TARGET,
) -> pd.DataFrame:
    rows = []
    for name in names:
        sel = selected.get(f"{name}@{target:.2f}")
        if sel is None:
            continue
        for kind, dkey, wkey in (
            ("attribute", dev.attribute, wrong.attribute),
            ("relate", dev.relate, wrong.relate),
        ):
            for value in sorted(set(dkey.tolist()) | set(wkey.tolist())):
                md, mw = dkey == value, wkey == value
                rows.append(
                    {
                        "method": name,
                        "target": target,
                        "by": kind,
                        "value": str(value),
                        "n_102": int(md.sum()),
                        "accuracy": float(sel["hit"][md].mean()) if md.any() else None,
                        "mean_level": float(sel["level"][md].mean())
                        if md.any()
                        else None,
                        "stop20_rate": float((sel["level"][md] == N_LEVELS).mean())
                        if md.any()
                        else None,
                        "n_99": int(mw.sum()),
                        "firing_99": float(sel["fired"][mw].mean())
                        if mw.any()
                        else None,
                    }
                )
    return pd.DataFrame(rows)


def flatten_ci(frame: pd.DataFrame) -> pd.DataFrame:
    """Split list-valued CI columns into _low/_high for CSV output."""
    out = frame.copy()
    for col in [c for c in out.columns if c.endswith(("_ci", "_ci975"))]:
        out[f"{col}_low"] = out[col].map(lambda v: v[0])
        out[f"{col}_high"] = out[col].map(lambda v: v[1])
        out = out.drop(columns=col)
    return out


def curves_figure(
    curves: pd.DataFrame, names: Sequence[str], path: str, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for i, name in enumerate(names):
        pts = curves[curves["method"] == name].sort_values("mean_level")
        color = f"C{i % 10}"
        axes[0].plot(
            pts["mean_level"], pts["accuracy"], ".-", ms=2, color=color, label=name
        )
        feas = pts[pts["accuracy"] >= PRIMARY_TARGET - TOLERANCE]
        axes[1].plot(
            feas["mean_level"], feas["firing_99"], ".-", ms=2, color=color, label=name
        )
    for t in TARGETS:
        axes[0].axhline(t, color="lightgray", lw=0.6, zorder=0)
    axes[0].set_xlabel("mean stop level (dev 102)")
    axes[0].set_ylabel("accuracy (dev 102)")
    axes[0].set_ylim(0.3, 1.01)
    axes[1].set_xlabel("mean stop level (dev 102), θ with acc >= 0.95")
    axes[1].set_ylabel("firing rate (dev reader-wrong 99)")
    for ax in axes:
        ax.set_xlim(0.5, 20.5)
    axes[0].legend(fontsize=6, loc="lower right")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
