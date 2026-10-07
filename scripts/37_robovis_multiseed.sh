#!/usr/bin/env bash
# ROBOVIS2027 multi-seed q_t smoothing: PGSG seeds +1 and +2 (bs4) into graphs_seed{1,2}/,
# clean all three graph dirs, then per-seed + averaged stage metrics (stage_metrics_avg.csv).
# graphs/ (seed offset 0) must already exist. Usage: bash scripts/37_robovis_multiseed.sh <run_id> <split>
set -euo pipefail
RUN="${1:?run_id}"; SPLIT="${2:?split (dev|test)}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_ROOT}"
LOG="outputs/robovis2027/logs/${RUN}_multiseed.log"; mkdir -p "$(dirname "${LOG}")"
{
for N in 1 2; do
  PYTHONPATH=Pix2Grp_CVPR2024 PYTHONDONTWRITEBYTECODE=1 uv run --project envs/pgsg python scripts/25_pgsg_infer.py \
    --config configs/robovis2027.yaml --run-id "${RUN}" --split "${SPLIT}" --seed-offset "${N}" --batch-size 4 \
    2>&1 | grep --line-buffered -v "instance det stat\|seq_len"
done
for SUB in graphs graphs_seed1 graphs_seed2; do
  uv run --project egtr python scripts/35_robovis_clean_graphs.py --runs "${RUN}" --graphs-subdir "${SUB}"
done
uv run --project egtr python scripts/28_robovis_stage_metrics.py --run-id "${RUN}" --avg-seeds graphs,graphs_seed1,graphs_seed2
} 2>&1 | tee -a "${LOG}"
