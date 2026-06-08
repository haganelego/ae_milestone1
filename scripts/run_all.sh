#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Milestone 1 — full reproducible pipeline driver.
#
# Runs the entire SGG-saturation experiment end to end, in dependency order,
# from configs/config.yaml. Re-runnable at any time:
#   - `bash scripts/run_all.sh`          : run/refresh (skip-existing where supported)
#   - `bash scripts/run_all.sh --force`  : wipe regenerable artifacts, run from scratch
#                                          (REQUIRED after changing scales / degradation
#                                           / EGTR export, since images+preds+scores change)
#
# Reproducing a PAST experiment: `git checkout <commit>` (restores config + tracked
# results), then `bash scripts/run_all.sh --force`.
#
# All Python runs through EGTR's uv env from the repo root (cwd stays at root).
# The ollama stage needs `ollama` serving the models named in config.yaml
# (qwen3.5:4b, qwen3.5:9b). EGTR inference needs the finetuned checkpoint + GPU.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
R="uv run --project egtr python"
FORCE="${1:-}"

log() { printf '\n========== %s ==========\n' "$*"; }

if [ "$FORCE" = "--force" ]; then
  log "FORCE: wiping regenerable artifacts (degraded images, EGTR preds, all scores)"
  rm -rf data/processed/degraded_images \
         outputs/egtr_predictions \
         outputs/llm_ablation outputs/vlm_scores outputs/llm_scores outputs/llm_inputs
fi

# --- Stage 1: data preparation (no GPU / no ollama) -----------------------
log "01 select VG subset";            $R scripts/01_select_vg_subset.py
log "02 generate resolution levels";  $R scripts/02_generate_resolution_levels.py
log "05 convert VG GT";               $R scripts/05_convert_vg_gt.py

# --- Stage 2: EGTR inference (GPU) ----------------------------------------
log "03 run EGTR inference";          $R scripts/03_run_egtr_inference.py
log "04 convert/validate EGTR out";   $R scripts/04_convert_egtr_outputs.py || true

# --- Stage 3: metrics + GT/relative oracle + baseline (no ollama) ---------
log "06 compute SG metrics";          $R scripts/06_compute_sg_metrics.py
log "07 compute oracle stop";         $R scripts/07_compute_oracle_stop.py
log "08 compute baseline stop";       $R scripts/08_compute_baseline_stop.py
log "15 relative saturation (Plan B)";$R scripts/15_relative_saturation.py

# --- Stage 4: ollama scoring (LLM-SGG 4b/9b + VLM single 4b/9b) -----------
# Ordered internally 4b -> 9b to avoid model thrashing. Skip-existing: only
# missing score files are computed. Use --force above to re-score from scratch.
log "ollama scoring: LLM + VLM (run_focused_100)"; $R scripts/run_focused_100.py

# --- Stage 5: post-hoc aggregation, convergence, analysis, figures --------
log "16 convergence eval";            $R scripts/16_convergence_eval.py
log "aggregate_100 (method compare)"; $R scripts/aggregate_100.py
log "14 error / dataset analysis";    $R scripts/14_error_analysis.py
log "17 representative panels";        $R scripts/17_representative_panels.py

log "DONE — see outputs/metrics/method_comparison_relative_100.csv and outputs/figures/"
