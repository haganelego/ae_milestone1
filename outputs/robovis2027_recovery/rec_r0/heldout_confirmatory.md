# Phase A: pre-registered held-out confirmation (MIRU pipeline, 96 images)

(2026-10-08 03:20 JST; A1 draft + A2 results, assembled by the orchestrator.)

The single source of truth is `configs/config_heldout.yaml`: its header comment and the `heldout_eval` block, registered at commit 48b964b before any held-out score existed. The method definitions live in `src/milestone1/miru_methods.py`. The evaluation is `scripts/42_miru_gt_eval.py --dataset heldout`, which writes to `outputs/heldout_eval/`.

## 1. Registered conditions
- **Primary rule.** `hybrid_fuse_mean_vlmimg9b_text4b_k2`.
  - Fused score = mean(image-only VLM 9b, SGG-text 4b) per level.
  - Hybrid score = 0.5 · mean object confidence (held-out EGTR) + 0.5 · fused score.
  - Stop at the first level t ≥ 5 at which the hybrid score is ≥ θ_hybrid = 0.5 at both t−1 and t (k = 2), else 20. θ_hybrid is `ablation.hybrid.threshold`, the same machinery that produced the registered MIRU-100 row (MAE 2.59).
- **Secondary rule.** `fuse_min_vlmimg9b_text4b_k2`: min(image-only 9b, text 4b), θ = 0.70, k = 2.
- **Registration-text discrepancy (disclosed).** The header comment summarises the primary as "theta 0.70". For the hybrid key this is inaccurate: the method key, as registered and as implemented, uses θ_hybrid = 0.5. θ 0.70 applies to the non-hybrid fusion rules. The method key is reported unchanged, as registered. Nothing was re-tuned. A dated correction comment (02:40 JST) was appended to the config header before any held-out result for the registered rules or any LLM/VLM method was computed; at that time only the zero-call node-confidence baseline row existed (00:57) and the text-4b scores existed unevaluated.
- **Score sources and decoding.**
  - **Image-only VLM 9b.** `scripts/11` single mode with the MIRU decoding unchanged: temperature 0.3, num_predict 1024, no JSON mode, unseeded. The image is resized so its longest side is ≤ 512 and saved as JPEG q85, with the MIRU VLM prompt. `vlm.num_ctx` = 16384 only sizes the context. Output: `outputs/robovis2027_miru/vlm_scores/qwen3.5-9b/single/`.
  - **SGG-text 4b.** `scripts/40` text condition with the `vlm_sgg` settings: JSON mode, think off, temperature 0, seed 42, num_ctx 16384, num_predict 1024, and one schema retry at temperature 0.3. The prompt is `llm.SATURATION_PROMPT` with a 5-level history. Output: `outputs/robovis2027_miru/llm_scores_vlm/qwen3.5-4b_h5_rel_on_conf_on_text/`.
- **Primary set.** All 96 images, scored by MAE against stop_c09. stop_c09 is recomputed from the held-out EGTR L20 (top-10 / top-5, coefficient 0.9; same definition as MIRU).
- **Auxiliary set.** The images with stop_c09 ≤ 18 (n = 73), which reproduces MIRU's selection condition.
- **Comparators.** Text 4b alone, image-only 9b alone, hybrid text 4b (+ node confidence), the node-confidence baseline, and the fixed-level curve.
- **GT metrics.** R50 and relsg_gt equal-observation Δ on both image sets.

## 2. Image set and the MIRU selection history
- **Held-out set.** The 100 ROBOVIS2027 test images (GT-only selection, seed 42) minus the 4 that overlap the current MIRU 100, leaving 96 (`data/processed/robovis2027_miru/selection_note.json`). A2 recomputed the set independently and it matches.
- **MIRU 100 history ("29/35").** Commit `2b038b2` (2026-06-17) reconstructed the MIRU set as a saturating subset. Its message reads: "removed 35 (29@20, 6@19), refilled to 100 via deterministic seed-42 shuffle continuation (35 new, all stop_c09<=18)". So 35 images were removed: 29 were right-censored at stop_c09 = 20 and 6 had stop_c09 = 19. 35 replacements were added.
  - `git diff 2b038b2^ 2b038b2 -- data/processed/selected_images.json` reconciles with this: 35 removed, 35 added, 65 kept.
  - The 29 / 6 split comes from the commit message. The per-image stop_c09 of the removed images cannot be recomputed from the current caches, because their EGTR predictions were not kept.

## 3. Disclosures (from the preflight, `data_manifest.json`)
1. One held-out image was in the original MIRU 100 (commit `e1b8e56`), before the `2b038b2` re-selection. It is not in the current MIRU 100.
2. Three held-out images are among the first 1000 VG-150 test images used only for aggregate PGSG validation runs. No per-image curves were examined.

The registered held-out set is unchanged.

## 4. Execution record and incident
- **Inputs (zero-call).** MIRU scripts 02–08 and `eval_topk_oracle` ran under the overlay. The EGTR predictions number 1920/1920, with 0 schema errors.
- **Text 4b.** Generated once, from 2026-10-08 01:00:14 to 01:57:47: 1536/1536 with 0 failures. These scores were never regenerated.
- **Image-only 9b, first attempt.** At 01:57:53 the scripts/11 start-up smoke test (an 8×8 image) got HTTP 500 from ollama. scripts/11 then skips the model by design, writes stop_level = −1 rows to `vlm_stop.csv` and exits. As a result 0 score files were written.
  - Most likely cause: GPU memory pressure. Two ollama runners were resident (4b and 9b) while PGSG Phase D used about 7 GB, and the ollama log shows repeated CUDA OOM from 02:01.
  - The incident is recorded in `rec_r0/failures.jsonl`. The failed `vlm_stop.csv` is kept in `rec_r0/logs/heldout_vlm_stop_failed_20261008_0157.csv`.
- **Image-only 9b, rerun.** Started 02:35:51 with scripts/11, the overlay and the decoding all unchanged, after the GPU was free and the smoke test passed. Result: 1536 / 1536 files (96 images × 16 levels), 0 errors, 0 parse errors; wall time 1451 s (02:35:51–03:00:02), median 0.92 s per call; script-11 log: total_calls=1536, failed=0 (`outputs/robovis2027_miru/logs/heldout_vlm9b_rerun.log`). Resolution recorded in `rec_r0/failures.jsonl`.
- **No mixing.** No file from a different condition was used.

## 5. Results (A2, `outputs/heldout_eval/`; registered conditions, no tuning)

Sources: `heldout_registered_table.{csv,md}`, `heldout_significance.csv`, `figures/heldout_quality_vs_level.{png,pdf}` (scripts 42 `--dataset heldout` and 50). Both score dirs complete (1536/1536); coverage 1.0 for all 7 score methods.

| method | mean stop | MAE c09 all 96 / ≤18 (n=73) | R50@stop | ΔR50 vs legacy mix [CI] | ΔR50 vs F (pt) | relsg@stop | Δrelsg vs legacy mix [CI] | Δrelsg vs F (pt) |
|---|---|---|---|---|---|---|---|---|
| **PRIMARY hybrid_fuse_mean** | 13.79 | **3.35 / 2.93** | .109 | −.008 [−.026, +.007] | −.018 | .360 | −.016 [−.033, +.004] | −.018 |
| secondary fuse_min | 16.04 | **2.54 / 2.47** | .138 | −.005 [−.016, +.004] | −.009 | .398 | −.001 [−.013, +.011] | −.004 |
| fuse_mean | 13.51 | 3.39 / 2.82 | .110 | −.002 [−.022, +.017] | −.014 | .356 | −.011 [−.033, +.012] | −.017 |
| hybrid text 4b | 13.46 | 3.77 / 3.14 | .113 | +.001 [−.017, +.019] | −.011 | .357 | −.008 [−.030, +.015] | −.015 |
| text 4b (comparator) | 13.73 | 3.56 / 3.04 | .110 | −.007 [−.026, +.011] | −.017 | .357 | −.016 [−.039, +.008] | −.020 |
| image-only 9b (comparator) | 14.17 | 3.21 / 2.67 | .115 | −.010 [−.032, +.011] | −.016 | .360 | −.024 [−.049, −.001] | −.024 |
| baseline, node conf (comparator) | 14.73 | 3.46 / 3.40 | .108 | −.026 [−.050, −.004] | −.028 | .361 | −.029 [−.048, −.010] | −.029 |
| fixed_20 (schedule-aware reference, not a comparator) | 20 | 4.10 / 5.37 | .159 | — | — | .430 | — | — |

**Significance** (paired bootstrap 2000 CI + paired sign-flip permutation; ✓ = CI excludes 0 or p < .05), method vs the three level-agnostic comparators (text 4b / image-only 9b / baseline), columns |stop − c09|, R50, relsg, subsets all-96 and ≤18:
- **PRIMARY hybrid_fuse_mean: 0/18 ✓ — the pre-registered MAE improvement does NOT replicate.** MAE Δ (all 96): −.21 [−.82, +.34] vs text 4b; +.15 [−.37, +.62] vs image-only 9b (worse); −.10 vs baseline. Best cell (≤18 vs baseline): −.47 [−1.06, +.08], p = .062. All R50/relsg deltas |Δ| ≤ .005. Point MAE 3.35 is worse than image-only 9b alone (3.21).
- **Secondary fuse_min: all-96 9/9 ✓ better** — |stop − c09| −1.02 / −.67 / −.92; R50 +.029 / +.024 / +.030; relsg +.041 / +.038 / +.038 (vs text 4b / image-only 9b / baseline). **≤18 subset 8/9 ✓** (the one ✗: |stop − c09| vs image-only 9b, −.21 [−.69, +.34]).
  - Caveat 1 (budget): fuse_min stops ≈2 levels later (16.0 vs 13.7–14.7), so its q(stop) advantage over the comparators is not budget-matched; at equal observation it sits on the fixed curve (Δ vs legacy mix −.005 / −.001, CIs include 0; Δ vs F −.009 / −.004).
  - Caveat 2 (set mismatch): 21/96 images never saturate (c09 = 20), which rewards late stopping; on MIRU-100 (all ≤18) fuse_min's MAE 3.12 was worse than hybrid_4b's 2.82. On the held-out ≤18 subset it still beats text 4b (−.58 [−1.14, −.03]) and baseline (−.93) on |stop − c09|.
- **Reporting rule (plan §3):** MAE and GT efficiency are both stated. No method improves GT efficiency vs the legacy equal-observation mix; all Δ vs F are negative (−.004 to −.029).

**Figure.** `figures/heldout_quality_vs_level.png`: 2×2 panels (all-96 / ≤18 × R50 / relsg_gt), fixed curve and F as schedule-aware references, method points. All methods sit on or below the fixed curve; fuse_min sits further along it (later stop).

## 6. Non-reach rate
On the held-out set, 21 of the 96 images have stop_c09 = 20, meaning the c09 criterion is never crossed. 23 have stop_c09 ≥ 19, and 73 have stop_c09 ≤ 18. These are reported as-is: the definition is unchanged and 20 still encodes non-reach. In contrast, MIRU's 100 has 0 images at 20, because it was selected to have stop_c09 ≤ 18.
