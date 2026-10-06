"""ROBOVIS2027: per-image, per-level SGDet R@20/50/100 (+ predicate hits for mR).

Reads outputs/robovis2027/<run_id>/graphs/<image_id>/level_XX.json and the VG-150
GT (test.json + rel.json via milestone1.gt), writes into the run dir:
  stage_metrics.csv           image_id,split,t,scale,R20,R50,R100,n_gt_rel,valid,
                              node_recall_gt,trip_recall_gt,relsg_gt (GT-RelSGScore on
                              capped label sets: evaluation.relsg_top_entities/relations),
                              relsg_gt_uncapped,node_prec_gt,trip_prec_gt
                              + det_recall20,det_prec20,trip_prec50,trip_prec20,F1_20
                              (one-to-one, see robovis_sgdet) on the raw graph, and
                              R20/R50/R100 + the same with suffix _clean on the cleaned
                              observed graph G (graphs_clean/, scripts/35)
  stage_predicate_hits.jsonl  per (image, t): GT predicate counts and hits@K (for mR)
  stage_predicate_hits.csv    image_id,t,pred_id,n_gt,hits50 (valid states, GT predicates present)
  stage_summary.json          per level: mean R@K, mR@K (official and present-only)

--avg-seeds graphs,graphs_seed1,graphs_seed2: computes every metric per graph dir
(raw <dir>/ and cleaned <dir>_clean/) -> stage_metrics_seed<N>.csv, then the
per-state mean over seeds (valid rows only, + n_seeds) -> stage_metrics_avg.csv and
per-level means -> stage_summary_avg.json (milestone1.stage_average).

--validate-official (envs/pgsg + PYTHONPATH=Pix2Grp_CVPR2024 only): feeds the SAME
saved predictions to Pix2Grp's own evaluate_relation_of_one_image (SGRecall +
SGMeanRecall) and compares per-image R@K and aggregate mR@K with ours; writes
official_validation.json.

Usage:
  uv run --project egtr python scripts/28_robovis_stage_metrics.py --run-id dev_r0
  PYTHONPATH=Pix2Grp_CVPR2024 PYTHONDONTWRITEBYTECODE=1 uv run --project envs/pgsg \
    python scripts/28_robovis_stage_metrics.py --run-id dev_r0 --validate-official
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1 import robovis_sgdet as sg  # noqa: E402
from milestone1.config import resolve  # noqa: E402
from milestone1.gt import load_gt_scene_graphs  # noqa: E402
from milestone1.robovis_data import level_name, load_robovis_config  # noqa: E402
from milestone1.stage_average import average_stage_tables  # noqa: E402

NUM_LEVELS = 20
HITS_CSV_FIELDS = ["image_id", "t", "pred_id", "n_gt", "hits50"]
HITS_K = 50
RELSG_FIELDS = [
    "node_recall_gt",
    "trip_recall_gt",
    "relsg_gt",
    "relsg_gt_uncapped",
    "node_prec_gt",
    "trip_prec_gt",
]
PREC_FIELDS = ["det_recall20", "det_prec20", "trip_prec50", "trip_prec20", "F1_20"]
CLEAN_SUFFIX = "_clean"
RAW_DIR = "graphs"
SEED_DIR_PREFIX = "graphs_seed"
CLEAN_FIELDS = [f"{f}{CLEAN_SUFFIX}" for f in ("R20", "R50", "R100", *PREC_FIELDS)]
EXTRA_FIELDS = [*RELSG_FIELDS, *PREC_FIELDS, *CLEAN_FIELDS]
CSV_FIELDS = [
    "image_id",
    "split",
    "t",
    "scale",
    "R20",
    "R50",
    "R100",
    "n_gt_rel",
    "valid",
    *RELSG_FIELDS,
    *PREC_FIELDS,
    *CLEAN_FIELDS,
]
CATEGORIES_PATH = REPO_ROOT / "data/processed/pgsg_vg/categories_dict.json"
VALIDATION_TOL = 1e-9
PERCENT = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/robovis2027.yaml"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validate-official", action="store_true")
    parser.add_argument(
        "--avg-seeds",
        default=None,
        help="comma list of graph dirs, e.g. graphs,graphs_seed1,graphs_seed2: "
        "writes stage_metrics_seed<N>.csv per dir and stage_metrics_avg.csv",
    )
    return parser.parse_args()


def predicate_hit_rows(hits_rows: list[dict[str, Any]]) -> list[dict[str, int]]:
    """Long form for mR@50: one row per (image, t, GT predicate present)."""
    out = []
    for row in hits_rows:
        hits = row["hits"][str(HITS_K)]
        for pred_id in sorted(row["counts"]):
            out.append(
                {
                    "image_id": row["image_id"],
                    "t": row["t"],
                    "pred_id": pred_id,
                    "n_gt": row["counts"][pred_id],
                    "hits50": hits.get(pred_id, 0),
                }
            )
    return out


def relsg_caps(cfg: dict[str, Any]) -> tuple[int, int]:
    ev = cfg.get("evaluation", {})
    return (
        int(ev.get("relsg_top_entities", sg.RELSG_DEFAULT_TOP_ENTITIES)),
        int(ev.get("relsg_top_relations", sg.RELSG_DEFAULT_TOP_RELATIONS)),
    )


def relsg_row(
    graph: dict[str, Any],
    gt_sets: tuple[set[int], set[sg.LabelTriplet]],
    caps: tuple[int, int],
) -> dict[str, float]:
    """GT-RelSGScore columns (capped sets + uncapped score) for one state."""
    capped = sg.relsg_score(graph, *gt_sets, caps[0], caps[1])
    uncapped = sg.relsg_score(graph, *gt_sets, None, None)
    return {
        "node_recall_gt": capped.node_recall,
        "trip_recall_gt": capped.trip_recall,
        "relsg_gt": capped.relsg,
        "relsg_gt_uncapped": uncapped.relsg,
        "node_prec_gt": capped.node_prec,
        "trip_prec_gt": capped.trip_prec,
    }


def precision_row(
    graph: dict[str, Any],
    clean: dict[str, Any] | None,
    gt: sg.TripletSet,
    gt_objs: tuple[np.ndarray, np.ndarray],
) -> dict[str, float]:
    """Precision-aware metrics on the raw graph and (suffix _clean) on G."""
    out = sg.precision_metrics(graph, gt, *gt_objs)
    if clean is None:
        return {**out, **dict.fromkeys(CLEAN_FIELDS, float("nan"))}
    m = sg.evaluate_image(clean, gt)
    out.update({f"R{k}{CLEAN_SUFFIX}": m.recall[k] for k in sg.TOPK})
    out.update(
        {
            f"{k}{CLEAN_SUFFIX}": v
            for k, v in sg.precision_metrics(clean, gt, *gt_objs).items()
        }
    )
    return out


def load_graph(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def compute(
    run_dir: Path, cfg: dict[str, Any], graphs_subdir: str = RAW_DIR
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    split, ids = manifest["split"], [int(i) for i in manifest["image_ids"]]
    scales = [float(s) for s in cfg["resolution"]["scales"]]
    gts = load_gt_scene_graphs(
        cfg["dataset"]["vg_root"], cfg["dataset"]["source_split"], ids
    )
    gt_sets = {i: sg.gt_triplets(gts[i]) for i in ids}
    gt_label_sets = {i: sg.gt_label_sets(gts[i]) for i in ids}
    gt_objs = {i: sg.gt_objects(gts[i]) for i in ids}
    caps = relsg_caps(cfg)
    relsg_vals: dict[int, list[dict[str, float]]] = {
        t: [] for t in range(1, NUM_LEVELS + 1)
    }
    rows, hits_rows = [], []
    per_level: dict[int, dict[int, sg.ImageMetrics]] = {
        t: {} for t in range(1, NUM_LEVELS + 1)
    }
    for image_id in ids:
        for t in range(1, NUM_LEVELS + 1):
            graph = load_graph(
                run_dir / graphs_subdir / str(image_id) / f"{level_name(t)}.json"
            )
            gt = gt_sets[image_id]
            valid = graph is not None and len(gt) > 0
            row = {
                "image_id": image_id,
                "split": split,
                "t": t,
                "scale": scales[t - 1],
                "n_gt_rel": len(gt),
                "valid": int(valid),
            }
            if not valid:
                blank = dict.fromkeys(["R20", "R50", "R100", *EXTRA_FIELDS], "")
                rows.append({**row, **blank})
                continue
            m = sg.evaluate_image(graph, gt)
            per_level[t][image_id] = m
            clean = load_graph(
                run_dir
                / f"{graphs_subdir}{CLEAN_SUFFIX}"
                / str(image_id)
                / f"{level_name(t)}.json"
            )
            rel = {
                **relsg_row(graph, gt_label_sets[image_id], caps),
                **precision_row(graph, clean, gt, gt_objs[image_id]),
            }
            relsg_vals[t].append(rel)
            rows.append(
                {
                    **row,
                    **{f"R{k}": round(m.recall[k], 6) for k in sg.TOPK},
                    **{k: round(v, 6) for k, v in rel.items()},
                }
            )
            hits_rows.append(
                {
                    "image_id": image_id,
                    "t": t,
                    "counts": m.counts,
                    "hits": {str(k): m.hits[k] for k in sg.TOPK},
                }
            )
    summary = {
        "run_id": run_dir.name,
        "split": split,
        "n_images": len(ids),
        "relsg_caps": {"top_entities": caps[0], "top_relations": caps[1]},
        "levels": {},
    }
    for t, by_id in per_level.items():
        ms = list(by_id.values())
        if not ms:
            continue
        summary["levels"][str(t)] = {
            "scale": scales[t - 1],
            "n_valid": len(ms),
            **{f"R{k}": float(np.mean([m.recall[k] for m in ms])) for k in sg.TOPK},
            **{f"mR{k}_official": sg.mean_recall_official(ms, k) for k in sg.TOPK},
            **{f"mR{k}_present": sg.mean_recall_present(ms, k) for k in sg.TOPK},
            **{
                key: float(np.nanmean([v[key] for v in relsg_vals[t]]))
                for key in EXTRA_FIELDS
            },
        }
    return (
        rows,
        hits_rows,
        summary,
        {"ids": ids, "gts": gts, "gt_sets": gt_sets, "per_level": per_level},
    )


def validate_official(run_dir: Path, ctx: dict[str, Any]) -> dict[str, Any]:
    """Run Pix2Grp's own per-image evaluator on the same saved predictions."""
    import torch
    from lavis.tasks.evaluation import vg_sgg_eval_tools as tools
    from lavis.tasks.evaluation.boxlist import BoxList
    from lavis.tasks.evaluation.sgg_vg_eval import evaluate_relation_of_one_image

    cats = json.loads(CATEGORIES_PATH.read_text())
    n_rel = len(cats["rel"])
    mode = "sgdet"
    report: dict[str, Any] = {"levels": {}}
    for t, ours in ctx["per_level"].items():
        if not ours:
            continue
        result: dict[str, Any] = {}
        recall = tools.SGRecall(result)
        recall.register_container(mode)
        mean_recall = tools.SGMeanRecall(result, cats["obj"], cats["rel"])
        mean_recall.register_container(mode)
        evaluator = {"eval_recall": recall, "eval_mean_recall": mean_recall}
        container = {"mode": mode, "iou_thres": sg.IOU_THRESHOLD}
        max_diff, n_eval, n_skipped, ours_eval = 0.0, 0, 0, []
        for image_id in ctx["ids"]:
            graph = load_graph(
                run_dir / RAW_DIR / str(image_id) / f"{level_name(t)}.json"
            )
            sgraph = ctx["gts"][image_id]
            if graph is None or not sgraph["relationships"]:
                continue
            w, h = graph["orig_size"]
            gt_box = BoxList(
                torch.tensor(
                    [o["bbox"] for o in sgraph["objects"]], dtype=torch.float32
                ),
                (w, h),
                mode="xyxy",
            )
            gt_box.add_field(
                "labels", torch.tensor([o["class_id"] for o in sgraph["objects"]])
            )
            gt_box.add_field(
                "relation_tuple",
                torch.tensor(
                    [
                        [r["subject_id"], r["object_id"], r["predicate_id"]]
                        for r in sgraph["relationships"]
                    ]
                ),
            )
            ents, rels = graph["entities"], graph["relations"]
            index = {e["id"]: k for k, e in enumerate(ents)}
            boxes = torch.tensor(
                [e["box_xyxy_orig"] for e in ents], dtype=torch.float32
            ).reshape(-1, 4)
            pred = BoxList(boxes, (w, h), mode="xyxy")
            pred.add_field(
                "pred_labels",
                torch.tensor([e["label_id"] for e in ents], dtype=torch.long),
            )
            pred.add_field("pred_scores", torch.tensor([e["score"] for e in ents]))
            pred.add_field(
                "rel_pair_idxs",
                torch.tensor(
                    [[index[r["sub"]], index[r["obj"]]] for r in rels], dtype=torch.long
                ).reshape(-1, 2),
            )
            pred.add_field(
                "pred_rel_label",
                torch.tensor([r["pred_id"] for r in rels], dtype=torch.long),
            )
            pred.add_field(
                "pred_rel_score", torch.tensor([r["pred_score"] for r in rels])
            )
            pred.add_field(
                "pred_rel_dist", torch.zeros((len(rels), n_rel))
            )  # unused by SGRecall/SGMeanRecall
            before = len(result[f"{mode}_recall"][50])
            evaluate_relation_of_one_image(gt_box, pred, container, evaluator)
            if len(result[f"{mode}_recall"][50]) == before:
                n_skipped += 1  # official skips images without predictions
                continue
            mine = ours[image_id]
            ours_eval.append(mine)
            n_eval += 1
            for k in sg.TOPK:
                max_diff = max(
                    max_diff, abs(result[f"{mode}_recall"][k][-1] - mine.recall[k])
                )
        mean_recall.calculate_mean_recall(mode)
        official = {
            f"R{k}": float(np.mean(result[f"{mode}_recall"][k])) for k in sg.TOPK
        }
        official.update(
            {f"mR{k}": float(result[f"{mode}_mean_recall"][k]) for k in sg.TOPK}
        )
        mine_agg = {
            f"R{k}": float(np.mean([m.recall[k] for m in ours_eval])) for k in sg.TOPK
        }
        mine_agg.update(
            {f"mR{k}": sg.mean_recall_official(ours_eval, k) for k in sg.TOPK}
        )
        agg_diff = max(abs(official[key] - mine_agg[key]) for key in official)
        report["levels"][str(t)] = {
            "n_evaluated": n_eval,
            "n_skipped_no_prediction": n_skipped,
            "max_abs_diff_per_image_recall": max_diff,
            "max_abs_diff_aggregate": agg_diff,
            "official": official,
            "ours": mine_agg,
            "match": bool(max_diff <= VALIDATION_TOL and agg_diff <= VALIDATION_TOL),
        }
        print(
            f"level {t:2d}: n={n_eval} per-image max|diff|={max_diff:.2e} agg max|diff|={agg_diff:.2e} official R@50={PERCENT * official['R50']:.2f} ours={PERCENT * mine_agg['R50']:.2f}",
            flush=True,
        )
    report["all_match"] = all(v["match"] for v in report["levels"].values())
    return report


def seed_index(graphs_subdir: str) -> int:
    """graphs -> 0, graphs_seed<N> -> N."""
    if graphs_subdir == RAW_DIR:
        return 0
    if not graphs_subdir.startswith(SEED_DIR_PREFIX):
        raise ValueError(f"unexpected graph dir {graphs_subdir}")
    return int(graphs_subdir[len(SEED_DIR_PREFIX) :])


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_avg_seeds(run_dir: Path, cfg: dict[str, Any], subdirs: list[str]) -> None:
    """Per-seed stage metrics + per-state mean over seeds (stage_metrics_avg.csv)."""
    tables = []
    for subdir in subdirs:
        rows = compute(run_dir, cfg, subdir)[0]
        path = run_dir / f"stage_metrics_seed{seed_index(subdir)}.csv"
        write_rows(path, rows)
        tables.append(pd.read_csv(path))
        print(f"wrote {path} ({sum(r['valid'] for r in rows)} valid)")
    avg = average_stage_tables(tables)
    avg.to_csv(run_dir / "stage_metrics_avg.csv", index=False)
    valid = avg[avg["valid"] == 1]
    numeric = [c for c in avg.columns if c not in ("image_id", "split", "t", "valid")]
    summary = {
        "run_id": run_dir.name,
        "seeds": subdirs,
        "n_states": len(avg),
        "n_valid_all_seeds": int(len(valid)),
        "levels": {
            str(t): {c: float(g[c].mean()) for c in numeric}
            for t, g in valid.groupby("t")
        },
    }
    (run_dir / "stage_summary_avg.json").write_text(json.dumps(summary, indent=1))
    print(
        f"wrote {run_dir / 'stage_metrics_avg.csv'} ({len(valid)}/{len(avg)} valid in all seeds)"
    )


def main() -> None:
    args = parse_args()
    cfg = load_robovis_config(args.config)
    run_dir = resolve(cfg["output_root"]) / args.run_id
    if args.avg_seeds:
        run_avg_seeds(
            run_dir, cfg, [d.strip() for d in args.avg_seeds.split(",") if d.strip()]
        )
        return
    rows, hits_rows, summary, ctx = compute(run_dir, cfg)
    with open(run_dir / "stage_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(run_dir / "stage_predicate_hits.jsonl", "w") as f:
        for row in hits_rows:
            f.write(json.dumps(row) + "\n")
    with open(run_dir / "stage_predicate_hits.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HITS_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(predicate_hit_rows(hits_rows))
    (run_dir / "stage_summary.json").write_text(json.dumps(summary, indent=1))
    print(
        f"wrote {len(rows)} rows ({sum(r['valid'] for r in rows)} valid) to {run_dir / 'stage_metrics.csv'}"
    )
    for t, s in summary["levels"].items():
        print(
            f"t={int(t):2d} n={s['n_valid']:3d} R@20={PERCENT * s['R20']:.2f} R@50={PERCENT * s['R50']:.2f} R@100={PERCENT * s['R100']:.2f} mR@50={PERCENT * s['mR50_official']:.2f}"
        )
    if args.validate_official:
        report = validate_official(run_dir, ctx)
        (run_dir / "official_validation.json").write_text(json.dumps(report, indent=1))
        print(f"official validation all_match={report['all_match']}")


if __name__ == "__main__":
    main()
