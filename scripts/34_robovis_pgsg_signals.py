"""ROBOVIS2027: PGSG-internal self-consistency signals (``pgsg_internal``).

Zero model calls. For every cached state writes
outputs/robovis2027/<run_id>/pgsg_signals.csv (image_id,split,t,scale + signals,
see milestone1.pgsg_signals.SIGNAL_FIELDS). With --analyze, also correlates the
signals (Spearman) with level t, gain_left = qmax - q_t and q_t / qmax for
q = R50 and q = relsg_gt (from stage_metrics.csv) and writes
pgsg_signals_corr.csv into the analyzed run dir.

Correlations: ``pooled`` = Spearman over all (image, t) states; ``within`` =
mean over images of the per-image Spearman across the 20 levels (images with a
constant series are skipped), which is what a per-image stop rule can exploit.

Usage:
  uv run --project egtr python scripts/34_robovis_pgsg_signals.py --runs dev_r0 test_r0 --analyze test_r0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.pgsg_signals import SIGNAL_FIELDS, compute_signals  # noqa: E402
from milestone1.robovis_data import level_name, load_robovis_config  # noqa: E402

NUM_LEVELS = 20
Q_METRICS = ("R50", "relsg_gt")
CANDIDATE_RHO = 0.3
MIN_POINTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/robovis2027.yaml"
    )
    parser.add_argument("--runs", nargs="+", default=["dev_r0", "test_r0"])
    parser.add_argument(
        "--analyze", default=None, help="run id to correlate (e.g. test_r0)"
    )
    return parser.parse_args()


def signals_table(run_dir: Path, scales: list[float]) -> pd.DataFrame:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = []
    for image_id in manifest["image_ids"]:
        for t in range(1, NUM_LEVELS + 1):
            path = run_dir / "graphs" / str(image_id) / f"{level_name(t)}.json"
            if not path.exists():
                continue
            graph = json.loads(path.read_text())
            rows.append(
                {
                    "image_id": int(image_id),
                    "split": manifest["split"],
                    "t": t,
                    "scale": scales[t - 1],
                    **compute_signals(graph),
                }
            )
    return pd.DataFrame(rows)


def _rho(x: pd.Series, y: pd.Series) -> float:
    ok = x.notna() & y.notna()
    if ok.sum() < MIN_POINTS or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def correlate(signals: pd.DataFrame, stage: pd.DataFrame) -> pd.DataFrame:
    df = signals.merge(
        stage[["image_id", "t", *Q_METRICS]], on=["image_id", "t"], how="inner"
    )
    targets: dict[str, pd.Series] = {"t": df["t"].astype(float)}
    for q in Q_METRICS:
        qmax = df.groupby("image_id")[q].transform("max")
        targets[f"gain_left_{q}"] = qmax - df[q]
        targets[f"q_over_qmax_{q}"] = (df[q] / qmax).where(qmax > 0)
    rows = []
    for sig in SIGNAL_FIELDS:
        row: dict[str, Any] = {"signal": sig}
        for name, target in targets.items():
            row[f"{name}_pooled"] = _rho(df[sig], target)
            per_image = [
                _rho(g[sig], target.loc[g.index]) for _, g in df.groupby("image_id")
            ]
            arr = np.asarray(per_image, dtype=float)
            row[f"{name}_within"] = (
                float(np.nanmean(arr)) if not np.isnan(arr).all() else float("nan")
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["n_states"] = len(df)
    return out


def print_table(corr: pd.DataFrame) -> None:
    cols = ["t"] + [f"{p}_{q}" for q in Q_METRICS for p in ("gain_left", "q_over_qmax")]
    print(f"n_states={corr.attrs.get('n_states')}")
    print("| signal | " + " | ".join(f"{c} pooled / within" for c in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for _, r in corr.iterrows():
        cells = [f"{r[f'{c}_pooled']:+.2f} / {r[f'{c}_within']:+.2f}" for c in cols]
        print(f"| {r['signal']} | " + " | ".join(cells) + " |")
    for q in Q_METRICS:
        for kind in ("pooled", "within"):
            col = f"gain_left_{q}_{kind}"
            hits = corr.loc[corr[col].abs() >= CANDIDATE_RHO, ["signal", col]]
            for _, r in hits.iterrows():
                print(
                    f"CANDIDATE |rho|>={CANDIDATE_RHO}: {r['signal']} vs gain_left({q}) [{kind}] rho={r[col]:+.3f}"
                )


def main() -> None:
    args = parse_args()
    cfg = load_robovis_config(args.config)
    root = resolve(cfg["output_root"])
    scales = [float(s) for s in cfg["resolution"]["scales"]]
    tables = {}
    for run in args.runs:
        table = signals_table(root / run, scales)
        table.to_csv(root / run / "pgsg_signals.csv", index=False)
        tables[run] = table
        print(f"wrote {root / run / 'pgsg_signals.csv'} ({len(table)} states)")
    if args.analyze:
        run_dir = root / args.analyze
        signals = tables.get(args.analyze)
        if signals is None:
            signals = pd.read_csv(run_dir / "pgsg_signals.csv")
        stage = pd.read_csv(run_dir / "stage_metrics.csv")
        corr = correlate(signals, stage)
        corr.to_csv(run_dir / "pgsg_signals_corr.csv", index=False)
        print_table(corr)


if __name__ == "__main__":
    main()
