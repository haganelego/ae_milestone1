"""ROBOVIS2027 recovery plan §5 (Phase C.1): per-seed stage metrics + multiseed_gain.csv.

For each PGSG graph dir of the dev run (graphs = seed 0 / base 42, graphs_seed1 = base 43,
graphs_seed2 = base 44; identical batch composition, see manifest*_json) the official-
matching scorer of scripts/28 (milestone1.robovis_sgdet + GT-RelSGScore + cleaned G) is
run unchanged; rows are restricted to the Phase C levels and written to the recovery run:
  stage_metrics_seed{N}.csv   seed + all scripts/28 stage_metrics columns, t in LEVELS
  multiseed_gain.csv          image_id, seed, metric, q5, q10, q15, q20, g (= q20 - q10), valid
                              metric in {R50, relsg_gt}, never mixed; missing -> empty, not 0
Seed 0 is re-scored with the same code and must equal dev_r0/stage_metrics.csv (checked).

Usage:
  uv run --project egtr python scripts/45_recovery_multiseed_metrics.py --run-id rec_r0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.robovis_data import load_robovis_config  # noqa: E402

DEV_RUN = REPO_ROOT / "outputs/robovis2027/dev_r0"
OUT_ROOT = REPO_ROOT / "outputs/robovis2027_recovery"
SEED_DIRS = {0: "graphs", 1: "graphs_seed1", 2: "graphs_seed2"}
LEVELS = (5, 10, 15, 20)
METRICS = ("R50", "relsg_gt")
GAIN_FROM, GAIN_TO = 10, 20
TOL = 1e-6

_spec = importlib.util.spec_from_file_location(
    "s28", REPO_ROOT / "scripts/28_robovis_stage_metrics.py"
)
s28 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s28)


def gain_rows(stage: pd.DataFrame, seed: int) -> list[dict]:
    rows = []
    for image_id, g in stage.groupby("image_id"):
        by_t = g.set_index("t")
        for metric in METRICS:
            q = {
                t: (
                    by_t.loc[t, metric]
                    if t in by_t.index and by_t.loc[t, "valid"] == 1
                    else None
                )
                for t in LEVELS
            }
            q = {
                t: (None if v is None or pd.isna(v) else float(v)) for t, v in q.items()
            }
            valid = all(v is not None for v in q.values())
            gain = (
                q[GAIN_TO] - q[GAIN_FROM]
                if q[GAIN_TO] is not None and q[GAIN_FROM] is not None
                else None
            )
            rows.append(
                {
                    "image_id": int(image_id),
                    "seed": seed,
                    "metric": metric,
                    **{f"q{t}": q[t] for t in LEVELS},
                    "g": gain,
                    "valid": int(valid),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-id", default="rec_r0")
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/robovis2027.yaml"
    )
    args = parser.parse_args()
    cfg = load_robovis_config(args.config)
    out = OUT_ROOT / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    gains = []
    for seed, subdir in SEED_DIRS.items():
        rows = s28.compute(DEV_RUN, cfg, subdir)[0]
        stage = pd.DataFrame(rows)
        stage = stage[stage.t.isin(LEVELS)].copy()
        for col in stage.columns:
            if col not in ("split",):
                stage[col] = pd.to_numeric(stage[col], errors="coerce")
        stage.insert(0, "seed", seed)
        stage.to_csv(out / f"stage_metrics_seed{seed}.csv", index=False)
        print(
            f"seed {seed} ({subdir}): {len(stage)} rows, valid {int(stage.valid.sum())}"
        )
        if seed == 0:
            ref = pd.read_csv(DEV_RUN / "stage_metrics.csv")
            ref = ref[ref.t.isin(LEVELS)].set_index(["image_id", "t"])
            cur = stage.set_index(["image_id", "t"])
            shared = [
                c for c in ref.columns if c in cur.columns and c not in ("split",)
            ]
            diff = (
                (cur[shared].astype(float) - ref.loc[cur.index, shared].astype(float))
                .abs()
                .max()
                .max()
            )
            if not diff <= TOL:
                raise RuntimeError(
                    f"seed 0 re-score differs from dev_r0/stage_metrics.csv (max |diff| {diff})"
                )
            print(
                f"  seed 0 matches dev_r0/stage_metrics.csv (max |diff| {diff:.1e} over {len(shared)} columns)"
            )
        gains.extend(gain_rows(stage, seed))
    pd.DataFrame(gains).to_csv(out / "multiseed_gain.csv", index=False)
    manifest = {
        "script": "scripts/45_recovery_multiseed_metrics.py",
        "dev_run": str(DEV_RUN.relative_to(REPO_ROOT)),
        "seed_dirs": SEED_DIRS,
        "base_seeds": {0: 42, 1: 43, 2: 44},
        "levels": LEVELS,
        "metrics": METRICS,
        "gain": f"q{GAIN_TO} - q{GAIN_FROM}",
        "missing": "empty cell (never 0)",
        "pgsg_manifests": [
            "manifest.json",
            "manifest_seed1.json",
            "manifest_seed2.json",
        ],
    }
    (out / "multiseed_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out / 'multiseed_gain.csv'} ({len(gains)} rows)")


if __name__ == "__main__":
    main()
