"""Plan §24 A: GT-box precheck of query-conditioned effective resolution (no VLM calls).

Target object = the object id carried by the step the final ``query`` depends on
(following filter steps back to the step with ``name (id)``; relate questions:
the relate step's id). eff_px(level) = box area fraction x pixels of the degraded
level image (manifest ``degraded_size``). Rules (all dev-chosen, in-sample):
box-only (τ per attribute) and gated (eff_px >= τ_attr and v4 score >= θ).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import atomic_write_json, read_id_csv, read_jsonl, utc_now
from .context import Context, require
from .evaluate import cluster_bootstrap_indices
from .informed_reader import onsets
from .policy import ACCURACY_TOLERANCE, threshold_grid

SCENE_GRAPHS = "data/processed/gqa_stop/scenegraphs_v5/val_sceneGraphs.json"
GQA_QUESTIONS_KEY = "gqa_questions"
OUT_DIR = "v5"
ATTRIBUTES = ("color", "material", "shape")
N_TAU = 40
ID_RE = re.compile(r"\(([^)]*)\)")
SCORES = ("direct_v4", "checklist_v4_min")
PERCENTILES = (2.5, 97.5)
NEVER = np.inf


# ---- target boxes ----------------------------------------------------------------------------
def target_ids(semantic: Sequence[Mapping[str, Any]]) -> list[str]:
    """Object ids of the step the final query reads (filters are followed back)."""
    query = semantic[-1]
    deps = list(query.get("dependencies") or [])
    while deps:
        step = semantic[deps[0]]
        found = ID_RE.search(str(step.get("argument", "")))
        if found:
            return [
                x.strip()
                for x in found.group(1).split(",")
                if x.strip() not in ("", "-")
            ]
        deps = list(step.get("dependencies") or [])
    return []


def area_fraction(graph: Mapping[str, Any] | None, ids: Sequence[str]) -> float | None:
    """Sum of target box areas / image area (capped at 1); None if absent."""
    if graph is None:
        return None
    objects = graph.get("objects", {})
    boxes = [objects[i] for i in ids if i in objects]
    if not boxes:
        return None
    total = sum(float(b["w"]) * float(b["h"]) for b in boxes)
    return min(1.0, total / (float(graph["width"]) * float(graph["height"])))


def level_pixels(manifest: Mapping[str, Mapping[str, Any]], image_id: str) -> list[int]:
    levels = sorted(manifest[image_id]["levels"], key=lambda r: r["level"])
    return [int(r["degraded_size"][0]) * int(r["degraded_size"][1]) for r in levels]


def box_table(ctx: Context, tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    questions = json.loads(Path(ctx.cfg["paths"][GQA_QUESTIONS_KEY]).read_text())
    graphs = json.loads((ctx.paths.repo_root / SCENE_GRAPHS).read_text())
    manifest = {
        r["image_id"]: r for r in read_jsonl(ctx.paths.data("image_manifest.jsonl"))
    }
    rows = []
    for name, table in tables.items():
        meta = table.groupby("question_id")[
            ["image_id", "attribute", "relate_count"]
        ].first()
        for q, m in meta.iterrows():
            gqa = questions[q]
            ids = target_ids(gqa["semantic"])
            frac = area_fraction(graphs.get(str(gqa["imageId"])), ids)
            pixels = level_pixels(manifest, str(m["image_id"]))
            rows.append(
                {
                    "set": name,
                    "question_id": q,
                    "attribute": m["attribute"],
                    "relate_count": m["relate_count"],
                    "target_ids": " ".join(ids),
                    "n_ids": len(ids),
                    "absent": frac is None,
                    "area_fraction": frac,
                    **{
                        f"eff_px_{lv}": (np.nan if frac is None else frac * px)
                        for lv, px in enumerate(pixels, start=1)
                    },
                }
            )
    return pd.DataFrame(rows)


# ---- rule evaluation ------------------------------------------------------------------------
def eff_matrix(boxes: pd.DataFrame, qids: Sequence[str]) -> np.ndarray:
    cols = [c for c in boxes.columns if c.startswith("eff_px_")]
    frame = boxes.set_index("question_id").loc[list(qids), cols]
    return frame.to_numpy(dtype=float)  # NaN (absent) never meets a threshold


def first_level(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First True level (1-based) per row, else last level; and whether met."""
    met = mask.any(axis=1)
    level = np.where(met, mask.argmax(axis=1) + 1, mask.shape[1])
    return level, met


def tau_grid(eff: np.ndarray) -> np.ndarray:
    finite = eff[np.isfinite(eff) & (eff > 0)]
    inner = np.geomspace(finite.min(), finite.max(), N_TAU - 2)
    return np.concatenate([[0.0], inner, [NEVER]])


def per_attribute_stats(
    eff: np.ndarray,
    score: np.ndarray | None,
    theta: float,
    correct: np.ndarray,
    attr: np.ndarray,
    taus: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """attr -> (n_correct[τ], level_sum[τ]) for stop = first eff>=τ (and score>=θ)."""
    out = {}
    gate = np.ones_like(eff, dtype=bool) if score is None else score >= theta
    for a in ATTRIBUTES:
        rows = attr == a
        n_c, s_l = np.zeros(len(taus)), np.zeros(len(taus))
        for i, tau in enumerate(taus):
            with np.errstate(invalid="ignore"):
                mask = (eff[rows] >= tau) & gate[rows]
            level, _ = first_level(mask)
            n_c[i] = correct[rows][np.arange(rows.sum()), level - 1].sum()
            s_l[i] = level.sum()
        out[a] = (n_c, s_l)
    return out


def combine(
    stats: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    (c0, s0), (c1, s1), (c2, s2) = (stats[a] for a in ATTRIBUTES)
    total_c = c0[:, None, None] + c1[None, :, None] + c2[None, None, :]
    total_s = s0[:, None, None] + s1[None, :, None] + s2[None, None, :]
    return total_c, total_s


def search(
    eff: np.ndarray,
    correct: np.ndarray,
    attr: np.ndarray,
    taus: np.ndarray,
    score: np.ndarray | None,
    thetas: Sequence[float],
    target_list: Sequence[float],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Best (τ_attr, θ) per target and the Pareto frontier over all combinations."""
    n = len(correct)
    best: dict[str, dict[str, Any]] = {}
    frontier: dict[int, float] = {}
    for theta in thetas:
        stats = per_attribute_stats(eff, score, theta, correct, attr, taus)
        total_c, total_s = combine(stats)
        flat_c, flat_s = total_c.ravel(), total_s.ravel()
        for c in np.unique(flat_c):
            m = float(flat_s[flat_c == c].min())
            frontier[int(c)] = min(frontier.get(int(c), np.inf), m)
        for target in target_list:
            ok = flat_c / n >= target - ACCURACY_TOLERANCE
            if not ok.any():
                continue
            # min level sum; ties: higher accuracy
            key = np.where(ok, flat_s * (n + 1) - flat_c, np.inf)
            idx = int(np.argmin(key))
            cand = {
                "theta": float(theta),
                "level_sum": float(flat_s[idx]),
                "n_correct": float(flat_c[idx]),
                "tau": dict(
                    zip(
                        ATTRIBUTES,
                        (float(taus[i]) for i in np.unravel_index(idx, total_c.shape)),
                        strict=True,
                    )
                ),
            }
            prev = best.get(f"{target:.2f}")
            if prev is None or (cand["level_sum"], -cand["n_correct"]) < (
                prev["level_sum"],
                -prev["n_correct"],
            ):
                best[f"{target:.2f}"] = cand
    for entry in best.values():
        entry["accuracy"] = entry["n_correct"] / n
        entry["mean_level"] = entry["level_sum"] / n
    pareto = pd.DataFrame(
        [{"accuracy": c / n, "mean_level": s / n} for c, s in sorted(frontier.items())]
    )
    pareto = pareto.sort_values("accuracy")
    pareto = pareto[pareto["mean_level"][::-1].cummin()[::-1] == pareto["mean_level"]]
    return best, pareto


def rule_stops(
    eff: np.ndarray,
    attr: np.ndarray,
    tau: Mapping[str, float],
    score: np.ndarray | None,
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    taus = np.array([tau[a] for a in attr])[:, None]
    gate = np.ones_like(eff, dtype=bool) if score is None else score >= theta
    with np.errstate(invalid="ignore"):
        mask = (eff >= taus) & gate
    return first_level(mask)


def score_stops(score: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    return first_level(score >= theta)


# ---- analysis ---------------------------------------------------------------------------------
def spearman(x: pd.Series, y: pd.Series) -> float:
    return float(x.rank().corr(y.rank()))


def matrices(table: pd.DataFrame, column: str) -> tuple[list[str], np.ndarray]:
    wide = table.pivot(index="question_id", columns="level", values=column).sort_index()
    return list(wide.index), wide.to_numpy(dtype=float)


def run_box_precheck(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("v5_box_precheck")
    root = ctx.paths.output_root
    dev = read_id_csv(root / "level_table_dev_merged.csv")
    wrong = read_id_csv(root / "ext_dev_wrong" / "level_table.csv")
    require((ctx.paths.repo_root / SCENE_GRAPHS).exists(), f"missing {SCENE_GRAPHS}")
    if ctx.dry_run:
        ctx.echo("[dry-run] v5 box precheck (no model calls)")
        return {}
    out = root / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    boxes = box_table(ctx, {"dev102": dev, "wrong99": wrong})
    boxes.to_csv(out / "box_targets.csv", index=False)
    result: dict[str, Any] = {
        "label": "plan §24 A (GT boxes; dev in-sample)",
        "time": utc_now(),
        "absent": {
            s: sorted(g.loc[g["absent"], "question_id"])
            for s, g in boxes.groupby("set")
        },
        "multi_id": int((boxes["n_ids"] > 1).sum()),
    }
    result["onset"] = onset_analysis(dev, boxes[boxes["set"] == "dev102"], out)
    result["rules"] = rule_analysis(ctx, dev, wrong, boxes, out)
    atomic_write_json(out / "box_precheck_summary.json", result)
    ctx.log("§24 A: GT-box precheck written")
    return result


def onset_analysis(dev: pd.DataFrame, boxes: pd.DataFrame, out: Path) -> dict[str, Any]:
    on = onsets(dev, "reader_correct")
    frame = boxes.set_index("question_id").assign(onset=pd.Series(on))
    frame = frame[~frame["absent"]]
    frame["eff_at_onset"] = [
        row[f"eff_px_{int(row.onset)}"] if row.onset <= 20 else np.nan
        for _, row in frame.iterrows()
    ]
    res: dict[str, Any] = {
        "n": len(frame),
        "spearman_effL20_onset": spearman(frame["eff_px_20"], frame["onset"]),
        "spearman_area_onset": spearman(frame["area_fraction"], frame["onset"]),
    }
    for a, g in frame.groupby("attribute"):
        q = g["eff_at_onset"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        res[a] = {
            "n": len(g),
            "spearman_effL20_onset": spearman(g["eff_px_20"], g["onset"]),
            "eff_at_onset_quantiles": {str(k): float(v) for k, v in q.items()},
            "onset_mean": float(g["onset"].mean()),
        }
    frame.reset_index()[
        [
            "question_id",
            "attribute",
            "area_fraction",
            "eff_px_20",
            "onset",
            "eff_at_onset",
        ]
    ].to_csv(out / "box_onset.csv", index=False)
    return res


def rule_analysis(
    ctx: Context, dev: pd.DataFrame, wrong: pd.DataFrame, boxes: pd.DataFrame, out: Path
) -> dict[str, Any]:
    from .stages_v4c_test import targets

    target_list = targets(ctx.cfg["policy"])
    grid = [t for t in threshold_grid(ctx.cfg["policy"]) if np.isfinite(t)]
    ev = ctx.cfg["evaluation"]
    reps, seed = int(ev["bootstrap_repeats"]), int(ev["bootstrap_seed"])
    qids, correct = matrices(dev, "reader_correct")
    correct = correct.astype(bool)
    attr = dev.groupby("question_id")["attribute"].first().loc[qids].to_numpy()
    eff = eff_matrix(boxes[boxes["set"] == "dev102"], qids)
    taus = tau_grid(
        np.concatenate(
            [
                eff.ravel(),
                eff_matrix(
                    boxes[boxes["set"] == "wrong99"],
                    sorted(wrong["question_id"].unique()),
                ).ravel(),
            ]
        )
    )
    scores = {s: matrices(dev, s)[1] for s in SCORES}
    wq, _ = matrices(wrong, "reader_correct")
    w_attr = wrong.groupby("question_id")["attribute"].first().loc[wq].to_numpy()
    w_eff = eff_matrix(boxes[boxes["set"] == "wrong99"], wq)
    w_scores = {s: matrices(wrong, s)[1] for s in SCORES}
    sel22 = pd.read_csv(
        ctx.paths.output_root / "dev_targets.csv", dtype={"target": str}
    )
    sel22 = sel22[(sel22["subset"] == "v22") & sel22["method"].isin(SCORES)]
    theta22 = {(r.target, r.method): float(r.theta) for r in sel22.itertuples()}
    variants = {"box_only": (None, [0.0])}
    variants.update({f"box_gated_{s}": (s, grid) for s in SCORES})
    groups = (
        dev.groupby("question_id")["group_id"].first().astype(str).loc[qids].tolist()
    )
    w_groups = (
        wrong.groupby("question_id")["group_id"].first().astype(str).loc[wq].tolist()
    )
    idx_dev = cluster_bootstrap_indices(groups, reps, seed)
    idx_wrong = cluster_bootstrap_indices(w_groups, reps, seed)
    rows, pareto_frames, summary = [], [], {}
    for name, (score_name, thetas) in variants.items():
        score = None if score_name is None else scores[score_name]
        best, pareto = search(eff, correct, attr, taus, score, thetas, target_list)
        pareto_frames.append(pareto.assign(method=name))
        summary[name] = best
        for target, entry in best.items():
            stop, _ = rule_stops(eff, attr, entry["tau"], score, entry["theta"])
            w_score = None if score_name is None else w_scores[score_name]
            _, w_fired = rule_stops(
                w_eff, w_attr, entry["tau"], w_score, entry["theta"]
            )
            d_stop, _ = score_stops(scores["direct_v4"], theta22[(target, "direct_v4")])
            _, d_fired = score_stops(
                w_scores["direct_v4"], theta22[(target, "direct_v4")]
            )
            hit = correct[np.arange(len(qids)), stop - 1]
            d_hit = correct[np.arange(len(qids)), d_stop - 1]
            dl, da = stop - d_stop, hit.astype(float) - d_hit.astype(float)
            df = w_fired.astype(float) - d_fired.astype(float)
            ci = {
                k: [
                    float(np.percentile([v[i].mean() for i in ix], p))
                    for p in PERCENTILES
                ]
                for k, v, ix in (
                    ("level", dl, idx_dev),
                    ("acc", da, idx_dev),
                    ("fire", df, idx_wrong),
                )
            }
            rows.append(
                {
                    "method": name,
                    "target": target,
                    "theta": entry["theta"],
                    **{f"tau_{a}": entry["tau"][a] for a in ATTRIBUTES},
                    "accuracy": entry["accuracy"],
                    "mean_level": entry["mean_level"],
                    "stop20_rate": float((stop == 20).mean()),
                    "firing_rate_99": float(w_fired.mean()),
                    "direct_v4_mean_level": float(d_stop.mean()),
                    "direct_v4_firing_99": float(d_fired.mean()),
                    "mean_level_diff": float(dl.mean()),
                    "mean_level_ci": ci["level"],
                    "accuracy_diff": float(da.mean()),
                    "accuracy_ci": ci["acc"],
                    "firing_diff": float(df.mean()),
                    "firing_ci": ci["fire"],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "box_rules.csv", index=False)
    pd.concat(pareto_frames).to_csv(out / "box_pareto.csv", index=False)
    figure(pd.concat(pareto_frames), ctx.paths.output_root / "dev_curves.csv", out)
    return {
        "selections": json.loads(frame.to_json(orient="records")),
        "n_tau": int(len(taus)),
    }


def figure(pareto: pd.DataFrame, curves_csv: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    curves = pd.read_csv(curves_csv)
    curves = curves[curves["subset"] == "v22"]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for name, color in (
        ("checklist_v4_min", "C0"),
        ("direct_v4", "C3"),
        ("fixed", "gray"),
    ):
        pts = curves[curves["method"] == name].sort_values("mean_level")
        ax.plot(pts["mean_level"], pts["accuracy"], ".-", ms=3, color=color, label=name)
    for name, color in (
        ("box_only", "C2"),
        ("box_gated_direct_v4", "C1"),
        ("box_gated_checklist_v4_min", "C4"),
    ):
        pts = pareto[pareto["method"] == name].sort_values("mean_level")
        ax.step(
            pts["mean_level"],
            pts["accuracy"],
            where="post",
            color=color,
            lw=1.4,
            label=f"{name} (Pareto, GT box)",
        )
    add_fixed_hull(ax, curves)
    for t in (0.90, 0.95, 0.97, 0.99):
        ax.axhline(t, color="lightgray", lw=0.6, zorder=0)
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0.3, 1.01)
    ax.set_xlabel("mean stop level (dev 102)")
    ax.set_ylabel("accuracy (common reader)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "box_curves.png", dpi=150)
    plt.close(fig)


def add_fixed_hull(ax: Any, curves: pd.DataFrame) -> None:
    """Dashed upper concave hull of the fixed-level curve (§22 dev_curves.csv rows)."""
    from .evaluate import hull_value

    fixed = curves[curves["method"] == "fixed"]
    curve = dict(zip(fixed["mean_level"], fixed["accuracy"], strict=True))
    xs = np.linspace(1, 20, 191)
    ax.plot(
        xs,
        [hull_value(curve, x) for x in xs],
        "--",
        color="black",
        lw=0.8,
        label="fixed concave hull",
    )


# ---- control: per-attribute θ without boxes (team-lead request, 2026-10-10) -------------------
CONTROL_SCORES = ("direct_v4", "checklist_v4_min")


def attr_theta_control(ctx: Context) -> dict[str, Any]:
    """Grid a per-attribute θ (3 params, in-sample) for the v4 scores; compare with
    single-θ direct_v4 (§22) and the box ∧ direct_v4 rows of ``box_rules.csv``."""
    ctx.set_stage("v5_attrtheta")
    from .stages_v4c_test import targets

    root = ctx.paths.output_root
    out = root / OUT_DIR
    require((out / "box_rules.csv").exists(), "run v5-box first")
    if ctx.dry_run:
        ctx.echo("[dry-run] per-attribute θ control (no model calls)")
        return {}
    dev = read_id_csv(root / "level_table_dev_merged.csv")
    wrong = read_id_csv(root / "ext_dev_wrong" / "level_table.csv")
    grid = np.array(threshold_grid(ctx.cfg["policy"]), dtype=float)
    ev = ctx.cfg["evaluation"]
    reps, seed = int(ev["bootstrap_repeats"]), int(ev["bootstrap_seed"])
    qids, correct = matrices(dev, "reader_correct")
    correct = correct.astype(bool)
    attr = dev.groupby("question_id")["attribute"].first().loc[qids].to_numpy()
    wq, _ = matrices(wrong, "reader_correct")
    w_attr = wrong.groupby("question_id")["attribute"].first().loc[wq].to_numpy()
    groups = (
        dev.groupby("question_id")["group_id"].first().astype(str).loc[qids].tolist()
    )
    w_groups = (
        wrong.groupby("question_id")["group_id"].first().astype(str).loc[wq].tolist()
    )
    idx_dev = cluster_bootstrap_indices(groups, reps, seed)
    idx_wrong = cluster_bootstrap_indices(w_groups, reps, seed)
    sel22 = pd.read_csv(root / "dev_targets.csv", dtype={"target": str})
    sel22 = sel22[(sel22["subset"] == "v22") & (sel22["method"] == "direct_v4")]
    theta22 = {r.target: float(r.theta) for r in sel22.itertuples()}
    d_score, w_d_score = matrices(dev, "direct_v4")[1], matrices(wrong, "direct_v4")[1]
    rows, pareto_frames = [], []
    for column in CONTROL_SCORES:
        score, w_score = matrices(dev, column)[1], matrices(wrong, column)[1]
        best, pareto = search(
            score, correct, attr, grid, None, [0.0], targets(ctx.cfg["policy"])
        )
        pareto_frames.append(pareto.assign(method=f"{column}_attr_theta"))
        for target, entry in best.items():
            stop, _ = rule_stops(score, attr, entry["tau"], None, 0.0)
            _, fired = rule_stops(w_score, w_attr, entry["tau"], None, 0.0)
            d_stop, _ = score_stops(d_score, theta22[target])
            _, d_fired = score_stops(w_d_score, theta22[target])
            hit = correct[np.arange(len(qids)), stop - 1]
            dl, df = stop - d_stop, fired.astype(float) - d_fired.astype(float)
            rows.append(
                {
                    "method": f"{column}_attr_theta",
                    "target": target,
                    **{f"theta_{a}": entry["tau"][a] for a in ATTRIBUTES},
                    "accuracy": float(hit.mean()),
                    "mean_level": float(stop.mean()),
                    "stop20_rate": float((stop == 20).mean()),
                    "firing_rate_99": float(fired.mean()),
                    "mean_level_diff": float(dl.mean()),
                    "mean_level_ci": [
                        float(np.percentile([dl[i].mean() for i in idx_dev], p))
                        for p in PERCENTILES
                    ],
                    "firing_diff": float(df.mean()),
                    "firing_ci": [
                        float(np.percentile([df[i].mean() for i in idx_wrong], p))
                        for p in PERCENTILES
                    ],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "box_attrtheta_targets.csv", index=False)
    pd.concat(pareto_frames).to_csv(out / "box_attrtheta_pareto.csv", index=False)
    ctx.log("§24 A control: per-attribute θ written")
    return {"rows": json.loads(frame.to_json(orient="records"))}
