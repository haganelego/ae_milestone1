"""Plan §27 (v27) mixed evaluation over answerable and unanswerable questions.

Stop rule per method: the first level ℓ in 1..20 whose score ≥ θ (L20 included);
"never stop" when no level reaches θ (counted as level 20 for mean levels).
Strict scoring (primary):
- answerable (reader_lp correct at L20): correct iff the method stops and the
  reader is correct at the stop level; never stopping is WRONG.
- unanswerable: correct iff the method never stops.
Unified scoring (sensitivity): a stop is correct iff the reader is correct at that
level; never stopping is correct iff the question is unanswerable.
Balanced accuracy = mean of the answerable accuracy and the unanswerable correct
rate. θ* per method = argmax of dev balanced accuracy (strict), ties -> smaller θ.
A3 has no threshold: its score is 1 when every item is resolved, so it is
evaluated at the single setting θ = 1 (reported with θ = NaN).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import atomic_write_json, read_id_csv, read_json, utc_now
from .context import Context, require
from .evaluate import cluster_bootstrap_indices
from .v27_infer import ANSWERABLE, TABLE_FILE, build_table

METHODS: dict[str, str] = {
    "B1": "reader_pans",
    "P_min": "pans_min",
    "P_mean": "pans_mean",
    "A1": "dres_score",
    "A2": "rawpans_min",
    "A3": "all_resolved",
}
NO_THRESHOLD = frozenset({"A3"})
NO_THRESHOLD_THETA = 1.0
NEVER = 0
BANDS = ((1, 5), (6, 10), (11, 15), (16, 19), (20, 20))
QUARTILES = (25, 50, 75)
CI_PERCENTILES = (2.5, 97.5)
SCORINGS = ("strict", "unified")
DIFF_KEYS = (
    "balanced_strict",
    "balanced_unified",
    "answerable_acc",
    "unanswerable_strict",
    "pooled_strict",
    "mean_level_all",
    "mean_level_answerable",
)


@dataclass(frozen=True)
class EvalArrays:
    """Question-major arrays (n_questions x n_levels) for one split."""

    question_ids: list[str]
    group_ids: list[str]
    attributes: np.ndarray
    answerable: np.ndarray  # bool (n,)
    reader_correct: np.ndarray  # bool (n, L)
    scores: dict[str, np.ndarray]  # method -> float (n, L)

    @property
    def n_levels(self) -> int:
        return int(self.reader_correct.shape[1])

    def subset(self, mask: np.ndarray) -> EvalArrays:
        ids = [q for q, keep in zip(self.question_ids, mask, strict=True) if keep]
        groups = [g for g, keep in zip(self.group_ids, mask, strict=True) if keep]
        return EvalArrays(
            ids,
            groups,
            self.attributes[mask],
            self.answerable[mask],
            self.reader_correct[mask],
            {m: s[mask] for m, s in self.scores.items()},
        )


def arrays_from_table(table: pd.DataFrame, n_levels: int) -> EvalArrays:
    ordered = table.sort_values(["question_id", "level"])
    qids = list(dict.fromkeys(ordered["question_id"].astype(str)))
    meta = ordered.groupby("question_id", sort=True).first()
    require(len(ordered) == len(qids) * n_levels, "level table is not complete (n x L)")

    def matrix(column: str) -> np.ndarray:
        return ordered[column].to_numpy().reshape(len(qids), n_levels)

    return EvalArrays(
        qids,
        [str(g) for g in meta.loc[qids, "group_id"]],
        meta.loc[qids, "attribute"].to_numpy(),
        (meta.loc[qids, "label"] == ANSWERABLE).to_numpy(),
        matrix("reader_correct").astype(bool),
        {m: matrix(c).astype(float) for m, c in METHODS.items()},
    )


# ---- per-question outcomes ------------------------------------------------------------------
def stop_levels(scores: np.ndarray, theta: float) -> np.ndarray:
    """First level (1-based) with score ≥ θ; 0 (NEVER) when no level reaches θ."""
    hit = scores >= theta
    first = hit.argmax(axis=1) + 1
    return np.where(hit.any(axis=1), first, NEVER)


def outcomes(arrays: EvalArrays, method: str, theta: float) -> dict[str, np.ndarray]:
    """Per-question stop level, effective level and strict / unified correctness."""
    stop = stop_levels(arrays.scores[method], theta)
    stopped = stop != NEVER
    rows = np.arange(len(stop))
    at_stop = np.where(
        stopped, arrays.reader_correct[rows, np.maximum(stop, 1) - 1], False
    )
    ans = arrays.answerable
    strict = np.where(ans, stopped & at_stop, ~stopped)
    unified = np.where(stopped, at_stop, ~ans)
    return {
        "stop": stop,
        "effective": np.where(stopped, stop, arrays.n_levels),
        "strict": strict,
        "unified": unified,
        "stopped": stopped,
    }


def theta_for(method: str, theta: float) -> float:
    return NO_THRESHOLD_THETA if method in NO_THRESHOLD else theta


def _rate(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else float("nan")


def summarize(arrays: EvalArrays, method: str, theta: float) -> dict[str, Any]:
    out = outcomes(arrays, method, theta_for(method, theta))
    ans, una = arrays.answerable, ~arrays.answerable
    row: dict[str, Any] = {
        "method": method,
        "theta": float("nan") if method in NO_THRESHOLD else theta,
        "n_answerable": int(ans.sum()),
        "n_unanswerable": int(una.sum()),
        "answerable_acc": _rate(out["strict"][ans]),
        "unanswerable_strict": _rate(out["strict"][una]),
        "unanswerable_unified": _rate(out["unified"][una]),
        "pooled_strict": _rate(out["strict"]),
        "pooled_unified": _rate(out["unified"]),
        "mean_level_all": _rate(out["effective"]),
        "mean_level_answerable": _rate(out["effective"][ans]),
        "mean_level_unanswerable": _rate(out["effective"][una]),
        "never_stop_answerable": _rate(~out["stopped"][ans]),
        "never_stop_unanswerable": _rate(~out["stopped"][una]),
        "answerable_stopped_wrong": int((out["stopped"] & ~out["strict"] & ans).sum()),
    }
    row["balanced_strict"] = (row["answerable_acc"] + row["unanswerable_strict"]) / 2
    row["balanced_unified"] = (row["answerable_acc"] + row["unanswerable_unified"]) / 2
    for side, mask in (("answerable", ans), ("all", np.ones_like(ans))):
        values = out["effective"][mask]
        for q, name in zip(QUARTILES, ("q1", "median", "q3"), strict=True):
            row[f"level_{name}_{side}"] = (
                float(np.percentile(values, q)) if len(values) else float("nan")
            )
    fired = out["stop"][una]
    for low, high in BANDS:
        row[f"unans_fire_L{low}_{high}"] = int(((fired >= low) & (fired <= high)).sum())
    row["unans_never"] = int((fired == NEVER).sum())
    return row


def theta_grid(cfg: Mapping[str, Any]) -> list[float]:
    start, stop, step = (float(cfg[k]) for k in ("start", "stop", "step"))
    count = int(round((stop - start) / step))
    return [round(start + i * step, 2) for i in range(count + 1)]


def sweep(arrays: EvalArrays, grid: Sequence[float]) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        thetas = [NO_THRESHOLD_THETA] if method in NO_THRESHOLD else grid
        rows += [summarize(arrays, method, t) for t in thetas]
    return pd.DataFrame(rows)


def sweep_by_attribute(arrays: EvalArrays, grid: Sequence[float]) -> pd.DataFrame:
    frames = []
    for attribute in sorted(set(arrays.attributes)):
        part = arrays.subset(arrays.attributes == attribute)
        frames.append(sweep(part, grid).assign(attribute=attribute))
    return pd.concat(frames, ignore_index=True)


def theta_star(table: pd.DataFrame) -> dict[str, float]:
    """argmax of balanced accuracy (strict) per method; ties -> smaller θ."""
    chosen: dict[str, float] = {}
    for method, part in table.groupby("method", sort=False):
        if method in NO_THRESHOLD:
            chosen[str(method)] = float("nan")
            continue
        finite = part[np.isfinite(part["balanced_strict"].astype(float))]
        if finite.empty:  # one label side is empty: balanced accuracy undefined
            chosen[str(method)] = float("nan")
            continue
        best = finite["balanced_strict"].max()
        chosen[str(method)] = float(
            finite[finite["balanced_strict"] == best]["theta"].min()
        )
    return chosen


# ---- bootstrap ---------------------------------------------------------------------------
def _stats(out: Mapping[str, np.ndarray], ans: np.ndarray) -> dict[str, float]:
    una = ~ans
    a_acc = _rate(out["strict"][ans])
    u_strict = _rate(out["strict"][una])
    u_unified = _rate(out["unified"][una])
    return {
        "balanced_strict": (a_acc + u_strict) / 2,
        "balanced_unified": (a_acc + u_unified) / 2,
        "answerable_acc": a_acc,
        "unanswerable_strict": u_strict,
        "pooled_strict": _rate(out["strict"]),
        "mean_level_all": _rate(out["effective"]),
        "mean_level_answerable": _rate(out["effective"][ans]),
    }


def ci(draws: Sequence[float]) -> tuple[float, float]:
    """Percentile 95% interval over the finite draws (NaN when there are none)."""
    values = np.asarray(draws, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    return (
        float(np.percentile(values, CI_PERCENTILES[0])),
        float(np.percentile(values, CI_PERCENTILES[1])),
    )


def _indexed(out: Mapping[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[idx] for k, v in out.items()}


def paired_bootstrap(
    arrays: EvalArrays,
    pairs: Sequence[tuple[str, str]],
    grid: Sequence[float],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Image-cluster paired bootstrap of (A − B) at every θ (same θ for both)."""
    indices = cluster_bootstrap_indices(arrays.group_ids, repeats, seed)
    rows = []
    for a, b in pairs:
        for theta in grid:
            out_a = outcomes(arrays, a, theta_for(a, theta))
            out_b = outcomes(arrays, b, theta_for(b, theta))
            point_a = _stats(out_a, arrays.answerable)
            point_b = _stats(out_b, arrays.answerable)
            draws: dict[str, list[float]] = {k: [] for k in DIFF_KEYS}
            for idx in indices:
                ans = arrays.answerable[idx]
                sa = _stats(_indexed(out_a, idx), ans)
                sb = _stats(_indexed(out_b, idx), ans)
                for k in DIFF_KEYS:
                    draws[k].append(sa[k] - sb[k])
            for k in DIFF_KEYS:
                low, high = ci(draws[k])
                rows.append(
                    {
                        "pair": f"{a} - {b}",
                        "theta": theta,
                        "quantity": k,
                        "estimate": point_a[k] - point_b[k],
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return pd.DataFrame(rows)


def method_ci(
    arrays: EvalArrays, thetas: Mapping[str, float], repeats: int, seed: int
) -> pd.DataFrame:
    """Image-cluster bootstrap CI of each method's own metrics at its θ*."""
    indices = cluster_bootstrap_indices(arrays.group_ids, repeats, seed)
    rows = []
    for method, theta in thetas.items():
        out = outcomes(arrays, method, theta_for(method, theta))
        point = _stats(out, arrays.answerable)
        for k in DIFF_KEYS:
            low, high = ci(
                [_stats(_indexed(out, i), arrays.answerable[i])[k] for i in indices]
            )
            rows.append(
                {
                    "method": method,
                    "theta": theta,
                    "quantity": k,
                    "estimate": point[k],
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


# ---- figures -----------------------------------------------------------------------------
def figures(table: pd.DataFrame, out_dir: Path, title: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    specs = (
        (
            "balanced_vs_theta",
            "theta",
            "balanced_strict",
            "θ",
            "balanced accuracy (strict)",
        ),
        (
            "unanswerable_vs_theta",
            "theta",
            "unanswerable_strict",
            "θ",
            "unanswerable correct rate (never stop)",
        ),
        (
            "answerable_acc_vs_level",
            "mean_level_answerable",
            "answerable_acc",
            "mean stop level on answerable (never = 20)",
            "answerable accuracy (strict)",
        ),
    )
    for name, x, y, xlabel, ylabel in specs:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for method, part in table.groupby("method", sort=False):
            if method in NO_THRESHOLD:
                if x != "theta":
                    ax.scatter(part[x], part[y], marker="*", s=90, label=str(method))
                else:
                    ax.axhline(
                        float(part[y].iloc[0]), ls=":", lw=1, label=f"{method} (no θ)"
                    )
                continue
            ax.plot(part[x], part[y], marker="o", ms=3, label=str(method))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        path = out_dir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(str(path))
    return written


# ---- stage ------------------------------------------------------------------------------
def load_or_build_table(ctx: Context, split: str) -> pd.DataFrame:
    path = ctx.paths.out(TABLE_FILE.format(split=split))
    return read_id_csv(path) if path.exists() else build_table(ctx, split)


def evaluate(ctx: Context, split: str, confirm_test: bool = False) -> dict[str, Any]:
    """Mixed evaluation (strict + unified), θ sweep, θ*, bootstrap, figures.

    Test evaluation needs ``--confirm-test-eval`` (team-lead go) and uses the
    dev θ* (``eval/dev/theta_star.json``) as the confirmatory number."""
    ctx.set_stage(f"v27_evaluate_{split}")
    if split == "test":
        require(
            confirm_test, "test evaluation needs --confirm-test-eval (team-lead go)"
        )
        require(
            ctx.paths.out("eval/dev/theta_star.json").exists(), "evaluate dev first"
        )
    if ctx.dry_run:
        ctx.echo(f"[dry-run] v27 evaluate {split}")
        return {}
    v27 = ctx.cfg["v27"]
    table = load_or_build_table(ctx, split)
    arrays = arrays_from_table(table, ctx.n_levels)
    grid = theta_grid(v27["theta_grid"])
    out_dir = ctx.paths.out(f"eval/{split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    main = sweep(arrays, grid)
    main.to_csv(out_dir / "sweep.csv", index=False)
    sweep_by_attribute(arrays, grid).to_csv(
        out_dir / "sweep_by_attribute.csv", index=False
    )
    if split == "dev":
        stars = theta_star(main)
        atomic_write_json(
            out_dir / "theta_star.json",
            {"rule": v27["confirmatory_rule"], "theta_star": stars, "time": utc_now()},
        )
    else:
        stars = read_json(ctx.paths.out("eval/dev/theta_star.json"))["theta_star"]
    ev = ctx.cfg["evaluation"]
    reps, seed = int(ev["bootstrap_repeats"]), int(ev["bootstrap_seed"])
    pairs = [tuple(p) for p in v27["bootstrap_pairs"]]
    boot = paired_bootstrap(arrays, pairs, grid, reps, seed)  # type: ignore[arg-type]
    boot.to_csv(out_dir / "paired_bootstrap.csv", index=False)
    ci = method_ci(arrays, stars, reps, seed)
    ci.to_csv(out_dir / "method_ci_at_theta_star.csv", index=False)
    at_star = pd.DataFrame(
        [
            summarize(arrays, m, t if np.isfinite(t) else NO_THRESHOLD_THETA)
            for m, t in stars.items()
        ]
    )
    at_star.to_csv(out_dir / "at_theta_star.csv", index=False)
    by_attr_star = pd.concat(
        [
            pd.DataFrame(
                [
                    summarize(
                        arrays.subset(arrays.attributes == a),
                        m,
                        t if np.isfinite(t) else NO_THRESHOLD_THETA,
                    )
                    for m, t in stars.items()
                ]
            ).assign(attribute=a)
            for a in sorted(set(arrays.attributes))
        ],
        ignore_index=True,
    )
    by_attr_star.to_csv(out_dir / "at_theta_star_by_attribute.csv", index=False)
    n_ans, n_q = int(arrays.answerable.sum()), len(arrays.question_ids)
    title = (
        f"v27 {split}: {n_q} q / {len(set(arrays.group_ids))} images; "
        f"answerable {n_ans}, unanswerable {n_q - n_ans}"
    )
    figs = figures(main, ctx.paths.out(f"figures/{split}"), title)
    reader_curve = {
        int(level): float(v)
        for level, v in enumerate(arrays.reader_correct.mean(axis=0), start=1)
    }
    summary = {
        "time": utc_now(),
        "split": split,
        "questions": n_q,
        "images": len(set(arrays.group_ids)),
        "answerable": n_ans,
        "unanswerable": n_q - n_ans,
        "theta_star": stars,
        "theta_star_source": "dev" if split == "dev" else "dev (frozen before test)",
        "reader_accuracy_by_level": reader_curve,
        "bootstrap": {"unit": "image group", "repeats": reps, "seed": seed},
        "figures": figs,
    }
    atomic_write_json(out_dir / "summary.json", summary)
    ctx.state.set(
        "v27_evaluate",
        {
            **(ctx.state.get("v27_evaluate") or {}),
            split: {"done": True, "time": utc_now()},
        },
    )
    ctx.log(f"v27 evaluate {split}: θ* {stars}")
    return summary
