#!/usr/bin/env bash
# Run PGSG (Pix2Grp CVPR 2024) SGDet evaluation on the converted local VG-150 subset.
# Prereq: uv run --project envs/pgsg python scripts/23_pgsg_convert_vg.py --num-test-images N
# Usage:  bash scripts/24_pgsg_eval.sh [JOB_NAME] [extra --options k=v ...]
# Example: bash scripts/24_pgsg_eval.sh greedy20 run.generation_mode=search
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${PGSG_CFG:-${REPO_ROOT}/configs/pgsg/rel_det_vg_eval_local.yaml}"
JOB_NAME="${1:-local}"
shift || true
LOG_DIR="${REPO_ROOT}/outputs/pgsg/logs"
LOG_FILE="${LOG_DIR}/$(date +%Y%m%d-%H%M%S)-${JOB_NAME}.log"
mkdir -p "${LOG_DIR}"

OPTIONS=()
if [[ $# -gt 0 ]]; then
  OPTIONS=(--options "$@")
fi

cd "${REPO_ROOT}"  # evaluate.py calls git.Git("."), so cwd must be inside a git repo
# PYTHONDONTWRITEBYTECODE keeps __pycache__ out of the Pix2Grp_CVPR2024 submodule.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${REPO_ROOT}/Pix2Grp_CVPR2024" \
  uv run --project "${REPO_ROOT}/envs/pgsg" \
  torchrun --nproc_per_node=1 --master_port="${MASTER_PORT:-29531}" \
  "${REPO_ROOT}/Pix2Grp_CVPR2024/evaluate.py" \
  --cfg-path "${CFG}" --job-name "${JOB_NAME}" "${OPTIONS[@]}" 2>&1 | tee "${LOG_FILE}"
echo "log: ${LOG_FILE}"
