# Nearest-Neighbor Re-run — Authoritative Data Sheet (n=100)

This sheet holds the **verified new numbers** from the full `run_all.sh --force`
re-run after switching degradation interpolation **bilinear → nearest-neighbor**.
Use these to update MILESTONE1_REPORT.md and MILESTONE1_PAPER.md. All values are
the mean over the 100-image test subset unless noted. Source CSVs are under
`outputs/metrics/` and `outputs/analysis/`.

> **Key invariant**: Level 20 = original image (scale 1.0, plain copy, no
> interpolation), so **all L20 absolute values are UNCHANGED** by this switch
> (NodeRecall@L20 0.569, TripletRecall@L20 0.286, SGScore@L20 0.427, L20
> zero-relation count 4/100). Only the degraded levels L1–L19 changed.

## Narrative changes vs the old bilinear results (must be reflected, not just numbers)

1. **All images now saturate.** RelOracle(0.8) is reached by **100/100** images
   (old bilinear: 70/100). Nearest-neighbor degradation is fully recoverable as
   resolution returns.
2. **Saturation happens LATER.** RelOracle(0.8) mean stop **18.96** (median 19),
   corresponding scale **≈0.85** (old bilinear: L16.71, scale ≈0.60). Reason:
   nearest-neighbor preserves blocky aliasing that suppresses fine-structure /
   semantic recovery until near-full resolution; bilinear's smooth blur lets EGTR
   recover earlier. This is the expected and intended effect of the switch
   ("closer to real degradation").
3. **The proposed method is the best — as it already was under n=100 bilinear; do
   NOT claim a baseline→proposed flip caused by this switch.** `hybrid_4b_k2` is
   the single best method, **MAE 1.29** vs RelOracle(0.8), beating the simple
   confidence baseline (MAE 2.93). For accurate framing: under **n=100 bilinear**
   `hybrid_4b_k2` was ALREADY rank 1 (MAE 3.25; baseline rank 7, MAE 5.65). The
   "simple node-conf baseline had the lowest MAE" result belongs ONLY to the
   earlier **10-image PoC** (baseline MAE 4.0 vs llm 8.4; see
   `outputs/metrics/stop_error_summary.csv`, CLAUDE.md status, PROGRESS.md). Under
   nearest-neighbor both methods improve (hybrid 3.25→1.29, baseline 5.65→2.93) and
   hybrid_4b_k2 stays best; the baseline→hybrid MAE gap is 1.64 (it was 2.40 under
   bilinear). The takeaway is the consistent superiority of structured-SGG + LLM +
   confidence-anchor + consecutive-k2, NOT a reversal of conclusions.
4. **Q2 (node-before-relation) is now WEAK / near-simultaneous** — report honestly.
   node-90% mean level 17.63, triplet-90% mean level 18.09, gap mean **+0.46**
   (median 0); node saturates at-or-before relation in **69/100** images. The
   early-level "objects visible, relations not yet" signal still exists
   (TripletRecall@L1 = 0.000 while NodeRecall@L1 = 0.034) but the *saturation-gap*
   is marginal under nearest-neighbor.
5. **§5.2 scale-invariance (paper)**: the cross-schedule invariance (~0.58–0.60)
   was a *bilinear* finding. We did NOT re-run the manual schedule under
   nearest-neighbor, so do not claim cross-schedule invariance for nearest. Reframe
   to: under nearest-neighbor the geometric schedule saturates at L≈19 (scale
   ≈0.85); re-validating cross-schedule invariance under nearest is future work.

## Q1 — level curves (mean over 100; from all_metrics.csv)

| Level | scale | NodeRecall | TripletRecall | SGScore | mean_obj_conf | mean_rel_conf |
|-------|-------|-----------|---------------|---------|---------------|---------------|
| 1  | 0.050 | 0.034 | 0.000 | 0.017 | 0.162 | 0.000 |
| 5  | 0.094 | 0.063 | 0.000 | 0.031 | 0.116 | 0.002 |
| 10 | 0.242 | 0.122 | 0.002 | 0.062 | 0.158 | 0.003 |
| 15 | 0.455 | 0.223 | 0.062 | 0.143 | 0.354 | 0.013 |
| 18 | 0.729 | 0.463 | 0.188 | 0.325 | 0.389 | 0.022 |
| 19 | 0.854 | 0.539 | 0.235 | 0.387 | 0.402 | 0.029 |
| 20 | 1.000 | 0.569 | 0.286 | 0.427 | 0.405 | 0.031 |

(obj_conf rises from ≈0.16 at L1 to a ≈0.40 plateau by L20 — plateau argument unchanged.)

## RelSGScore curve (relative_metrics.csv, mean over 100)

| Level | RelSGScore | RelNodeRecall |
|-------|-----------|---------------|
| 1  | 0.016 | 0.032 |
| 5  | 0.038 | 0.071 |
| 10 | 0.068 | 0.131 |
| 15 | 0.232 | 0.307 |
| 18 | 0.570 | 0.659 |
| 19 | 0.750 | 0.827 |
| 20 | 1.000 | 1.000 |

(RelTripletRecall aggregates to NaN — many images have 0 reference triplets at L20; keep using RelSGScore.)

## Oracle / RelOracle stop levels (n=100, all images reach; from oracle_stop.csv, relative_oracle_stop.csv)

GT-based Oracle (absolute SGScore threshold). **IMPORTANT**: the "stop level" in
oracle_stop.csv falls back to L20 when the threshold is never genuinely reached, so
the stop-level mean below is fallback-inclusive. The meaningful "genuinely saturate"
count is `#images whose max SGScore over levels ≥ θ` (computed from all_metrics.csv):
| θ | **genuinely saturate** | stop-level mean (L20-fallback incl.) | median |
|---|------------------------|--------------------------------------|--------|
| 0.3 | **82/100** | 17.57 | 18 |
| 0.4 | **59/100** | 18.66 | 20 |
| 0.5 | **37/100** | 19.20 | 20 |
| 0.6 | **20/100** | 19.65 | 20 |
| gain_stop | — | 5.87 | 2 |

This GT-oracle degeneracy (only 37/100 genuinely reach SGScore≥0.5) is the
motivation for RelOracle. Contrast in docs: "RelOracle(0.8) 100/100 genuinely
saturate (RelSGScore=1.0 at L20 by construction) vs true GT Oracle(0.5) only
**37/100**". The old bilinear paper said 28/100 — that is STALE; use 37/100.

RelOracle (relative-to-L20 threshold) — the main reference:
| θ | reach | mean stop | median | scale@mean |
|---|-------|-----------|--------|-----------|
| 0.7 | 100/100 | 18.40 | 19 | 0.779 |
| **0.8** | **100/100** | **18.96** | **19** | **0.849** |
| 0.9 | 100/100 | 19.49 | 20 | 0.925 |

RelOracle(0.8) stop histogram: {13:1, 14:1, 15:1, 16:5, 17:2, 18:14, 19:32, 20:44}.

## Method comparison — Table 1 (vs RelOracle(0.8), n=100; method_comparison_relative_100.csv)

| Rank | Method | Mean Stop | MAE | MedAE | Early% | Late% | Pearson r |
|------|--------|-----------|-----|-------|--------|-------|-----------|
| 1 | **hybrid_4b_k2** | 19.35 | **1.29** | 1.0 | 0.11 | 0.49 | 0.034 |
| 2 | hybrid_9b_k2 | 18.30 | 2.16 | 1.0 | 0.23 | 0.37 | −0.091 |
| 3 | llm_sgg_4b_h5_k2 | 17.77 | 2.53 | 1.0 | 0.31 | 0.37 | −0.011 |
| 4 | baseline (conf≥0.4) | 16.69 | 2.93 | 2.0 | 0.63 | 0.17 | 0.009 |
| 5 | hybrid_4b_k1 | 16.30 | 3.38 | 2.0 | 0.55 | 0.20 | −0.052 |
| 6 | hybrid_9b_k1 | 15.58 | 4.04 | 3.0 | 0.61 | 0.17 | −0.106 |
| 7 | vlm_9b_single_k2 | 12.84 | 6.18 | 6.0 | 0.97 | 0.01 | −0.005 |
| 8 | llm_sgg_4b_h5_k1 | 12.17 | 7.07 | 7.0 | 0.81 | 0.08 | 0.088 |
| 9 | vlm_9b_single_k1 | 10.74 | 8.22 | 8.0 | 1.00 | 0.00 | 0.048 |
| 10 | vlm_4b_single_k2 | 8.10 | 10.98 | 12.5 | 0.96 | 0.03 | 0.096 |
| 11 | llm_sgg_9b_h5_k2 | 6.25 | 12.71 | 13.0 | 1.00 | 0.00 | 0.092 |
| 12 | vlm_4b_single_k1 | 5.22 | 13.74 | 14.0 | 1.00 | 0.00 | 0.068 |
| 13 | llm_sgg_9b_h5_k1 | 5.08 | 13.88 | 14.0 | 1.00 | 0.00 | 0.031 |

## Convergence / B-1 deployable — Table 2 (vs RelOracle(0.8); method_comparison_convergence_100.csv)

| Method | Mean Stop | Trigger% | MAE | MedAE | Early% | Late% |
|--------|-----------|----------|-----|-------|--------|-------|
| baseline_abs_conf04 | 16.69 | 73% | 2.93 | 2.0 | 0.63 | 0.17 |
| hybrid_convergence | 17.28 | 32% | 3.04 | 1.0 | 0.29 | 0.35 |
| llm4b_convergence | 16.88 | 34% | 3.46 | 1.0 | 0.33 | 0.36 |
| llm9b_convergence | 13.89 | 56% | 5.67 | 6.0 | 0.56 | 0.20 |
| hybrid_4b_k1_abs | 12.54 | 95% | 6.56 | 6.5 | 0.92 | 0.02 |
| llm_sgg_4b_abs | 12.17 | 88% | 7.07 | 7.0 | 0.81 | 0.08 |
| hybrid_9b_k1_abs | 10.83 | 96% | 8.25 | 9.0 | 0.95 | 0.02 |
| conf_convergence | 8.60 | 90% | 10.52 | 12.0 | 0.86 | 0.06 |
| llm_sgg_9b_abs | 5.08 | 100% | 13.88 | 14.0 | 1.00 | 0.00 |

## Model-size asymmetry (from Table 1)

VLM single MAE: 4b_k1=13.74, 4b_k2=10.98, 9b_k1=8.22, 9b_k2=6.18 → VLM needs the big model.
LLM-SGG MAE: 4b_k1=7.07, 4b_k2=2.53, 9b_k1=13.88, 9b_k2=12.71 → small 4b is BEST; 9b overconfident → far too early.
Capability-controlled 4b: hybrid_4b_k2 (SGG+LLM) MAE 1.29 ≫ vlm_4b_single_k2 (image) MAE 10.98.

## consecutive-k effect (k1→k2, MAE)
hybrid_4b: 3.38→1.29; hybrid_9b: 4.04→2.16; llm_sgg_4b: 7.07→2.53; llm_sgg_9b: 13.88→12.71. k2 helps consistently and strongly.

## Confidence plateau / baseline firing (baseline_stop.csv; trigger% from convergence table)
mean_obj_conf plateaus ≈0.40 (L1 0.162 → L20 0.405). Absolute-threshold firing:
θ=0.3 trigger 100% (mean 10.45), θ=0.4 trigger 73% (mean 16.69), θ=0.5 trigger ~7% (plateau rarely exceeds 0.5). Plateau brittleness argument unchanged.

## Error / dataset analysis (outputs/analysis, n=100)
- Categories never recalled at any level: **24/137 (17.5%)**.
- Predicates never recalled at any level: **20/32 (62.5%)** — VG150 long-tail; structural, not an EGTR defect.
- Saturation gap (Q2): see narrative point 4.

## UNCHANGED — keep these as-is (interpolation-independent)
- EGTR validity: author eval JSON R@20/50/100 = 23.54/30.21/34.35, mR@50 = 7.94, AP50 = 30.83; our 500-image RTX5080 = 20.4/26.3/30.5. KEEP.
- Relation decouple table (L20 zero-rel 23→4/100, L20 mean rel 4.1→7.9, TripletRecall@L20 0.250→0.286, NodeRecall@L20 0.569): L20 is the original image → UNCHANGED, KEEP.
- num_ctx fix narrative: KEEP.
- EGTR hyperparameters (obj_conf 0.3, rel_conf 0.005, top_k_obj 20, top_k_rel 30): KEEP.

## New SGDet@20-enabling output schema (already documented by the methodology pass — do not duplicate, just ensure consistency)
EGTR prediction JSON relationships now carry: subject_class_id, subject_bbox, predicate_id, object_class_id, object_bbox, triplet_score (+ existing subject_id/object_id/labels/confidence/subject_conf/object_conf). GT JSON objects carry class_id; GT relationships carry subject_id/object_id/*_class_id/*_bbox/predicate_id. Main metric stays SGScore; SGDet R@K is computable post-hoc but NOT yet computed.
