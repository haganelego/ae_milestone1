"""MIRU stop methods: oracle MAE (stop_c09) + GT-based secondary evaluation.

Usage:
    uv run --project egtr python scripts/42_miru_gt_eval.py [--rebuild-stage] [--repeats N]
    uv run --project egtr python scripts/42_miru_gt_eval.py --dataset heldout

``--dataset heldout`` loads configs/config_heldout.yaml as an overlay and uses its
``heldout_eval`` block: pre-registered image selection, stop_c09 recomputed from
the new EGTR L20 predictions, baseline / node confidence from the predictions,
score dirs under outputs/heldout_scores, outputs under outputs/heldout_eval.

1. Stage metrics of the cached EGTR predictions vs VG-150 GT (original image
   coordinates): R@K (stored <= 30 relations) and GT RelSGScore per image x level
   -> <gt_eval_dir>/stage_metrics.csv (+ stage_predicate_hits.csv). Cached.
2. Stop maps: existing MIRU CSVs (outputs/metrics) + evaluation_extra.score_methods
   (score dirs, same threshold / consecutive-k / hybrid machinery as script 13).
3. MIRU primary: MAE / MedAE / early / late vs stop_c09
   (outputs/metrics/topk_oracle/oracle_topk_per_image.csv), as eval_topk_oracle.py.
4. GT secondary per q in {R50, relsg_gt}: q(stop), fixed_01..20 curve,
   equal-observation fixed mix and paired bootstrap (robovis_eval)
   -> <gt_eval_dir>/eval[/q_relsg_gt]/{stops,summary,per_image}.csv, bootstrap.json
5. One table: <gt_eval_dir>/miru_method_table.{csv,md}.

MIRU stop CSVs do not record whether a stop at level 20 was triggered or a
fallback; ``at_final_level`` (stop_t == 20) is reported instead of a forced rate.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import CONFIG_OVERLAY_ENV, load_config, resolve  # noqa: E402
from milestone1.miru_gt import build_stage_metrics  # noqa: E402
from milestone1.miru_methods import (  # noqa: E402
    HYBRID_PREFIX,
    load_existing_method_stops,
    registered_score_methods,
)
from milestone1.miru_oracle import (  # noqa: E402
    baseline_stops,
    object_confidence,
    topk_oracle,
)
from milestone1.robovis_eval import (  # noqa: E402
    PRED_HITS_FILENAME,
    STOPS_COLUMNS,
    build_curves,
    build_eval_set,
    fixed_stops,
    load_predicate_hits,
    load_stage_metrics,
    paired_bootstrap,
    per_image_metrics,
    q_output_dir,
    summarize,
)
from milestone1.stop_policy import fixed_method_name  # noqa: E402

STAGE_FILENAME = "stage_metrics.csv"
SCHEDULE_AWARE = "schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator"
STAGE_META_FILENAME = (
    "stage_metrics.meta.json"  # cache key: prediction dir + file count
)
# dataset -> (config block, overlay). The held-out overlay redirects paths and
# defines its own ``heldout_eval`` block (a dict deep-merge would otherwise keep
# MIRU's evaluation_extra score dirs).
DATASETS: dict[str, dict[str, str | None]] = {
    "miru": {"block": "evaluation_extra", "overlay": None},
    "heldout": {"block": "heldout_eval", "overlay": "configs/config_heldout.yaml"},
}
ORACLE_CSV = "outputs/metrics/topk_oracle/oracle_topk_per_image.csv"
ORACLE_COLUMN = "stop_c09"
Q_METRICS = ("R50", "relsg_gt")
RECALL_K = 50
ORACLE_RATIO = 0.9  # relative-stable GT oracle (plan rule; auxiliary here)
DEFAULT_REPEATS = 2000
BOOTSTRAP_SEED = 20271007
CI_LEVEL = 0.95
LEGACY_LLM_NOTE = "legacy cache (llm_ablation: temp 0.3, gate 0.3)"
LEGACY_VLM_NOTE = "legacy cache (vlm_stop.csv: threshold 0.8)"
LEGACY_VLM_SCORES_NOTE = "legacy cache (vlm_scores, recomputed at 0.70)"
LEGACY_SCORE_KEYS = ("vlm_single_4b", "vlm_single_9b")
FUSION_PREFIX = "fuse_"
FUSION_NOTE = "late fusion (zero-call)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=sorted(DATASETS), default="miru",
                   help="miru: evaluation_extra block; heldout: heldout_eval block "
                        "of configs/config_heldout.yaml (overlay)")  # fmt: skip
    p.add_argument("--overlay", default=None,
                   help="config overlay (default: the dataset's overlay, if any)")  # fmt: skip
    p.add_argument("--rebuild-stage", action="store_true")
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    p.add_argument("--all-methods", action="store_true",
                   help="table rows for every method, not just the block's final_methods")  # fmt: skip
    return p.parse_args()


def registry_key(method: str) -> str:
    """Score-method key of a registry method name (strip hybrid_ and _k<k>)."""
    name = method.removeprefix(HYBRID_PREFIX)
    return name.rsplit("_k", 1)[0]


def ensure_stage_metrics(
    cfg: dict, pred_dir: Path, ids: list[int], out_dir: Path, rebuild: bool
) -> Path:
    path = out_dir / STAGE_FILENAME
    meta_path = out_dir / STAGE_META_FILENAME
    meta = {"pred_dir": str(pred_dir), "n_images": len(ids),
            "n_files": count_prediction_files(pred_dir, ids)}  # fmt: skip
    cached = json.loads(meta_path.read_text()) if meta_path.exists() else None
    if path.exists() and not rebuild and cached == meta:
        return path
    rows, hits = build_stage_metrics(
        pred_dir, ids,
        int(cfg["resolution_levels"]["num_levels"]),
        cfg["paths"]["vg_root"], cfg["paths"]["vg_split"],
    )  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    pd.DataFrame(hits).to_csv(out_dir / PRED_HITS_FILENAME, index=False)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"built {path} ({len(rows)} rows)")
    return path


def count_prediction_files(pred_dir: Path, ids: list[int]) -> int:
    return sum(1 for i in ids for _ in (pred_dir / str(i)).glob("level_*.json"))


def require_complete_predictions(
    pred_dir: Path, ids: list[int], num_levels: int
) -> None:
    """Stop early (no partial caches) if any image x level prediction is missing."""
    missing = [(i, t) for i in ids for t in range(1, num_levels + 1)
               if not (pred_dir / str(i) / f"level_{t:02d}.json").exists()]  # fmt: skip
    if missing:
        n_img = len({i for i, _ in missing})
        raise SystemExit(
            f"EGTR predictions incomplete under {pred_dir}: {len(missing)} files missing "
            f"({n_img} images), e.g. {missing[:3]}; rerun when inference has finished"
        )


def oracle_mae(
    stops: dict[int, int], oracle: dict[int, int], ids: list[int]
) -> dict[str, Any]:
    """MIRU comparison vs stop_c09 (same statistics as eval_topk_oracle.py)."""
    m = np.array([stops[i] for i in ids], dtype=float)
    o = np.array([oracle[i] for i in ids], dtype=float)
    err = m - o
    corr = float(np.corrcoef(m, o)[0, 1]) if m.std() > 0 and o.std() > 0 else math.nan
    return {"mean_stop": m.mean(), "mean_oracle_stop": o.mean(),
            "mae_c09": np.abs(err).mean(), "medae_c09": float(np.median(np.abs(err))),
            "early_rate": float((err < 0).mean()), "late_rate": float((err > 0).mean()),
            "pearson_r": corr}  # fmt: skip


def stops_frame(
    methods: dict[str, dict[int, int]], ids: list[int], max_level: int
) -> pd.DataFrame:
    rows = [
        {"image_id": i, "method": name, "stop_t": int(stop_map[i]),
         "natural_stop": int(stop_map[i]) < max_level,
         "forced_terminal": int(stop_map[i]) == max_level, "failure_count": 0}
        for name, stop_map in methods.items() for i in ids
    ]  # fmt: skip
    return pd.DataFrame(rows, columns=list(STOPS_COLUMNS))


def gt_secondary(
    stage_path: Path, methods: dict[str, dict[int, int]], ids: list[int], q_metric: str,
    out_dir: Path, num_levels: int, repeats: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:  # fmt: skip
    curves = build_curves(
        load_stage_metrics(stage_path, q_metric), q_metric, num_levels
    )
    adaptive = list(methods)
    stops = pd.concat([stops_frame(methods, ids, num_levels), fixed_stops(ids, num_levels)],
                      ignore_index=True)  # fmt: skip
    eval_set = build_eval_set(curves, ORACLE_RATIO)
    pred_hits = load_predicate_hits(stage_path.parent / PRED_HITS_FILENAME, RECALL_K)
    per_image = per_image_metrics(stops, curves, eval_set)
    summary = summarize(per_image, stops, curves, eval_set, pred_hits, adaptive)
    final = fixed_method_name(num_levels)
    boot = paired_bootstrap(per_image, curves, eval_set, adaptive,
                            [(m, final) for m in adaptive], repeats, BOOTSTRAP_SEED,
                            CI_LEVEL)  # fmt: skip
    target = q_output_dir(out_dir / "eval", q_metric)
    target.mkdir(parents=True, exist_ok=True)
    stops.to_csv(target / "stops.csv", index=False)
    per_image.to_csv(target / "per_image.csv", index=False)
    summary.to_csv(target / "summary.csv", index=False)
    (target / "bootstrap.json").write_text(json.dumps(boot, indent=2))
    return summary, boot


def ci_text(stat: dict[str, float] | None) -> str:
    if not stat:
        return "n/a"
    return f"{stat['mean']:+.3f} [{stat['ci_low']:+.3f}, {stat['ci_high']:+.3f}]"


def build_table(
    methods, sources, ids, oracle, results, num_levels, sat_max
) -> pd.DataFrame:  # noqa: ANN001
    final = fixed_method_name(num_levels)
    rows = []
    for name in [*methods, final]:
        row: dict[str, Any] = {"method": name, "source": sources.get(name, SCHEDULE_AWARE),
                               "n": len(ids)}  # fmt: skip
        stops = methods.get(name) or dict.fromkeys(ids, num_levels)
        row.update(oracle_mae(stops, oracle, ids))
        # MIRU's image set was restricted to stop_c09 <= 18 ("saturating"); report
        # the same subset of a new image set separately.
        sat_ids = [i for i in ids if oracle[i] <= sat_max]
        row["n_sat"] = len(sat_ids)
        row["mae_c09_sat"] = (
            oracle_mae(stops, oracle, sat_ids)["mae_c09"] if sat_ids else math.nan
        )
        row["at_final_level"] = float(np.mean([stops[i] == num_levels for i in ids]))
        for q, (summary, boot) in results.items():
            s = summary.set_index("method")
            row[f"{q}_stop"] = s.loc[name, "q_stop"]
            if name == final:
                continue
            row[f"{q}_eq_fixed"] = s.loc[name, "eq_fixed_q"]
            st = boot["methods"][name]["q_minus_eq_fixed"]
            row[f"{q}_minus_eq_fixed"] = st["mean"]
            row[f"{q}_minus_eq_fixed_ci"] = ci_text(st)
            row[f"{q}_minus_fixed20_ci"] = ci_text(
                boot["pairwise"][f"{name}_minus_{final}"]["q_stop"]
            )  # F = GT-posterior best fixed mixture on this set, at the method's mean stop.
            row[f"{q}_F_at_mean_stop"] = s.loc[name, "F_at_mean_stop"]
            stf = boot["methods"][name]["q_minus_F"]
            row[f"{q}_minus_F"] = stf["mean"]
            row[f"{q}_minus_F_ci"] = ci_text(stf)

        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame) -> str:
    cols = ["method", "source", "mean_stop", "mae_c09", "medae_c09", "mae_c09_sat", "n_sat",
            "at_final_level",
            "R50_stop", "R50_minus_eq_fixed_ci", "R50_minus_F", "relsg_gt_stop",
            "relsg_gt_minus_eq_fixed_ci", "relsg_gt_minus_F"]  # fmt: skip
    cols = [c for c in cols if c in table.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in table.to_dict("records"):
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append("n/a" if v is None or (isinstance(v, float) and math.isnan(v))
                         else f"{v:.3f}" if isinstance(v, float) else str(v))  # fmt: skip
        lines.append("| " + " | ".join(cells) + " |")
    notes = [
        "",
        "- `*_minus_eq_fixed_ci`: legacy equal-observation fixed mix (adjacent two levels at "
        "the method's mean stop) — uses the level index; previous plan's registered comparison.",
        "- `*_minus_F`: q_stop - F(mean_stop), point estimate only; F = best fixed mixture "
        "on this set: schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator.",
        "- fixed_20: schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator.",
    ]
    return "\n".join(lines + notes) + "\n"


def load_dataset_config(dataset: str, overlay: str | None) -> dict[str, Any]:
    """Base config + optional overlay (MILESTONE1_CONFIG_OVERLAY mechanism)."""
    overlay = overlay or DATASETS[dataset]["overlay"]
    if overlay:
        os.environ[CONFIG_OVERLAY_ENV] = overlay
    else:
        os.environ.pop(CONFIG_OVERLAY_ENV, None)
    return load_config()


def resolve_image_ids(cfg: dict[str, Any], block: dict[str, Any]) -> list[int]:
    """Image ids: derived from ``block.selection`` (cross-checked against
    ``paths.selected_images`` if that file exists), else paths.selected_images."""
    listed_path = resolve(cfg["paths"]["selected_images"])
    listed = (
        [int(i) for i in json.loads(listed_path.read_text())]
        if listed_path.exists() else None
    )  # fmt: skip
    sel = block.get("selection")
    if not sel:
        if listed is None:
            raise SystemExit(f"{listed_path} not found")
        return listed
    source = json.loads(resolve(sel["from"]).read_text())
    exclude = {int(i) for i in json.loads(resolve(sel["exclude"]).read_text())}
    derived = [int(r["image_id"]) for r in source
               if r.get("split") == sel["split"] and int(r["image_id"]) not in exclude]  # fmt: skip
    if listed is not None and set(listed) != set(derived):
        raise SystemExit(
            f"{listed_path} ({len(listed)}) differs from the pre-registered selection "
            f"({len(derived)}): {sorted(set(listed) ^ set(derived))[:10]}"
        )
    return derived


def oracle_stops(
    cfg: dict[str, Any], block: dict[str, Any], pred_dir: Path, ids: list[int],
    num_levels: int, out_dir: Path,
) -> dict[int, int]:  # fmt: skip
    """stop_c09 from the MIRU CSV (``oracle: csv``) or recomputed from EGTR L20."""
    if block.get("oracle", "csv") == "csv":
        df = pd.read_csv(resolve(block.get("oracle_csv", ORACLE_CSV)))
        return {
            int(r["image_id"]): int(r[ORACLE_COLUMN]) for r in df.to_dict("records")
        }
    orc = topk_oracle(pred_dir, ids, num_levels)
    rows = [{"image_id": i, ORACLE_COLUMN: orc.stops[i],
             **{f"rel_sgscore_L{lv:02d}": orc.core_scores[i][lv]
                for lv in range(1, num_levels + 1)}} for i in ids]  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "oracle_topk_per_image.csv", index=False)
    return orc.stops


def main() -> None:
    args = parse_args()
    cfg = load_dataset_config(args.dataset, args.overlay)
    block = cfg.get(DATASETS[args.dataset]["block"]) or {}
    num_levels = int(cfg["resolution_levels"]["num_levels"])
    paths = {**cfg["paths"], **(block.get("paths") or {})}
    cfg = {**cfg, "paths": paths}
    ids = resolve_image_ids(cfg, block)
    pred_dir = resolve(paths["egtr_predictions"])
    out_dir = resolve(block.get("gt_eval_dir", "outputs/miru_gt_eval"))
    metrics_dir = resolve(paths["metrics"])
    print(f"dataset={args.dataset} overlay={cfg.get('_overlay')} images={len(ids)} "
          f"predictions={pred_dir} -> {out_dir}")  # fmt: skip

    require_complete_predictions(pred_dir, ids, num_levels)
    stage_path = ensure_stage_metrics(cfg, pred_dir, ids, out_dir, args.rebuild_stage)
    sources: dict[str, str] = {}
    conf = None
    if block.get("legacy_csv_methods", True):
        existing = load_existing_method_stops(metrics_dir, ids)
        sources = {m: LEGACY_LLM_NOTE for m in existing}
        sources.update({m: LEGACY_VLM_NOTE for m in existing if m.startswith("vlm_")})
        sources["baseline"] = "MIRU CSV (node confidence)"
    else:
        # Node confidence and the baseline straight from the EGTR predictions.
        conf = object_confidence(pred_dir, ids, num_levels)
        theta = float(cfg["baseline"]["mean_object_conf_threshold"])
        existing = {"baseline": baseline_stops(conf, ids, theta, num_levels)}
        sources["baseline"] = f"node confidence >= {theta} (from predictions)"
    registry = registered_score_methods(cfg, ids, metrics_dir, block, conf)
    for m in registry.stops:
        key = registry_key(m)
        cov = registry.coverage.get(key, math.nan)
        label = (
            LEGACY_VLM_SCORES_NOTE if key in LEGACY_SCORE_KEYS and args.dataset == "miru"
            else FUSION_NOTE if key.startswith(FUSION_PREFIX) else "score dir"
        )  # fmt: skip
        sources[m] = f"{label}, coverage {cov:.0%}" if cov < 1 else label
    if block.get("require_complete_scores", False):
        # Pre-registered evaluation: never score from partial score dirs.
        partial = {k for k, cov in registry.coverage.items() if cov < 1.0}
        dropped = [m for m in registry.stops if registry_key(m) in partial]
        for m in dropped:
            registry.stops.pop(m)
        if partial:
            print(
                f"pending (incomplete score dirs {sorted(partial)}): {sorted(dropped)}"
            )
    methods = {**existing, **registry.stops}
    print(f"methods: {len(existing)} baseline/CSV, {len(registry.stops)} from score dirs; "
          f"coverage {registry.coverage}")  # fmt: skip
    wanted = block.get("final_methods") or list(methods)
    if not args.all_methods:
        missing = [m for m in wanted if m not in methods]
        if missing:
            print(f"not available yet (skipped): {missing}")
        methods = {m: methods[m] for m in wanted if m in methods}

    oracle = oracle_stops(cfg, block, pred_dir, ids, num_levels, out_dir)
    sat_max = int(block.get("saturating_max_stop", num_levels))
    results = {q: gt_secondary(stage_path, methods, ids, q, out_dir, num_levels, args.repeats)
               for q in Q_METRICS}  # fmt: skip
    table = build_table(methods, sources, ids, oracle, results, num_levels, sat_max)
    table.to_csv(out_dir / "miru_method_table.csv", index=False)
    (out_dir / "miru_method_table.md").write_text(to_markdown(table))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table[["method", "mean_stop", "mae_c09", "mae_c09_sat", "R50_stop",
                     "R50_minus_eq_fixed_ci", "relsg_gt_stop",
                     "relsg_gt_minus_eq_fixed_ci"]].round(3).to_string(index=False))  # fmt: skip
    print(f"-> {out_dir / 'miru_method_table.md'}")


if __name__ == "__main__":
    main()
