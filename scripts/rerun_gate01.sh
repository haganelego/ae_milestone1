#!/usr/bin/env bash
# Full re-derive after adopting gate=0.1 / baseline threshold=0.35.
# Order respects deps: predictions -> metrics -> oracle/baseline ->
# aggregate_100 (writes hybrid_stop) -> eval_topk_oracle (stop_c09) ->
# 15 (relative figs) -> 16 -> 14 -> 17. LLM/VLM are NOT re-run (input is
# re-filtered at 0.3, so the LLM-visible scene graph is unchanged).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="uv run --project egtr python"
log() { printf '\n========== %s ==========\n' "$*"; }

log "WIPE gate0.3 predictions"; rm -rf outputs/egtr_predictions
log "03 EGTR inference (gate0.1)"; $R scripts/03_run_egtr_inference.py
log "04 validate";                $R scripts/04_convert_egtr_outputs.py || true
log "06 SG metrics";             $R scripts/06_compute_sg_metrics.py
log "07 oracle stop";            $R scripts/07_compute_oracle_stop.py
log "08 baseline stop (0.35)";   $R scripts/08_compute_baseline_stop.py
log "aggregate_100 (+hybrid)";   $R scripts/aggregate_100.py
log "eval_topk_oracle (c09)";    $R scripts/eval_topk_oracle.py
log "15 relative figures";       $R scripts/15_relative_saturation.py
log "16 convergence";            $R scripts/16_convergence_eval.py
log "14 error analysis";         $R scripts/14_error_analysis.py
log "17 representative x100";     $R scripts/17_representative_panels.py
log "DONE"
