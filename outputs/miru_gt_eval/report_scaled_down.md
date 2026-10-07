# Scaled-down follow-up on the MIRU pipeline: does the current image help the SGG-sequence saturation judge?

This report is zero-call: it only reads cached scores and evaluation outputs. Drafted 2026-10-08 from repo HEAD `8b6e6b2` plus uncommitted outputs. Every number below is traceable to the file named next to it.

## 1. Setup
- **Pipeline.** MIRU as-is (`configs/config.yaml`). The data are the old 100 VG test images (`data/processed/selected_images.json`) and their 20 bilinear degradation levels (`data/processed/degraded_images/`). The scene graphs are cached EGTR predictions (`outputs/egtr_predictions/`).
- **Judge input.** The input is the SGG sequence of the last `history_length` = 5 levels, scored at levels `start_level` = 5..20. Gates come from `llm.input_object_conf_threshold` 0.3 and `input_relation_conf_threshold` 0.005. Each level keeps the top 20 objects and 30 relations, with confidences. The sequence builder is the one in `scripts/09`: it is identical to `llm_engine.build_sequence(rel_on, conf_on)`, checked on 60/60 sampled states.
- **Prompt.** The prompt is `milestone1.llm.SATURATION_PROMPT`, unchanged, with the JSON contract `saturation_score`, `stop_decision`, `reason`. The img_single condition adds one sentence: "An image of the scene at the CURRENT resolution level (the last entry of the sequence) is attached. Use it together with the scene graphs." It also attaches the current level's degraded image.
  - The image is preprocessed exactly as in `scripts/11` single mode (`vlm._load_and_downscale_image`: longest side ≤ 512, JPEG q85).
  - The stored images are 500 px on the long side, so they are only re-encoded, not resized.
  - The image adds about 220 prompt tokens.
- **Conditions (2×2).** The two factors are {text, img_single} × {qwen3.5:4b, qwen3.5:9b}. Each condition covers 100 images × 16 levels = 1600 states.
  - Code: `scripts/40_vlm_sgg_saturation_scoring.py`.
  - Scores: `outputs/llm_scores_vlm/<slug>/<image_id>/level_XX.json`.
  - Per-slug `manifest.json` / `usage.jsonl`, plus the run log `outputs/llm_scores_vlm/run.log`.
- **Decoding** (`configs/config.yaml` `vlm_sgg`, identical for all four conditions):
  - ollama `/api/chat` through `milestone1.ollama_client`, with JSON mode and `think: false`.
  - temperature 0, seed 42, num_ctx 16384, num_predict 1024.
  - One schema retry at temperature 0.3, plus transient retries.
  - Failed calls are recorded under `_failures/` and never converted to a score.
  - Failures: 3 of 1600 for img 9b, and 0 for the other conditions. Under the MIRU rule a missing level counts as 0.
- **Why the text-only controls were re-run.** The MIRU text cache (`outputs/llm_ablation/qwen3.5-{4b,9b}_h5_rel_on_conf_on`, 2026-06-17) predates the current EGTR predictions, which were regenerated on 2026-06-20 with the object gate changed from 0.3 to 0.1. The inputs were not saved, so whether the relation sets still match cannot be verified. That cache also used different decoding: temperature 0.3, no JSON mode, and a prompt variant that differs only in a blank line and criterion numbering. Re-running text-only with the same client, settings and inputs isolates the effect of the image. The legacy rows are kept as a reproducibility reference.
- **Stop rule** (unchanged MIRU, `scripts/13` `consecutive_k_stop`): θ = `llm.saturation_threshold` 0.70, k ∈ {1, 2}, start level 5, and no stop means level 20.
  - The primary metric is MAE against `stop_c09` (`outputs/metrics/topk_oracle/oracle_topk_per_image.csv`, mean 13.48).
  - The GT-based secondary metric is computed by A2 (`scripts/42_miru_gt_eval.py`).

## 2. Method table (k = 2)
Source: `outputs/miru_gt_eval/miru_method_table.md` (`.csv`). `R50_stop` and `relsg_gt_stop` are the mean q at the stop. `− eq-fixed` is the paired difference to the equal-observation fixed-level mix, with a bootstrap 95 % CI. That mix stops every image at a fixed level that is interpolated to the same mean stop. `− fixed20` is the difference to always stopping at level 20.

| method | source | mean stop | MAE vs stop_c09 | R50 at stop | R50 − eq-fixed | relsg_gt at stop | relsg_gt − eq-fixed | relsg_gt − fixed20 |
|---|---|---|---|---|---|---|---|---|
| llm_sgg_text_4b_k2 | new text control | 13.76 | **3.00** | 0.167 | +0.003 [−0.024, +0.032] | 0.404 | −0.005 [−0.026, +0.015] | −0.074 [−0.103, −0.047] |
| llm_sgg_text_9b_k2 | new text control | 10.32 | 4.40 | 0.098 | +0.011 [−0.021, +0.042] | 0.287 | −0.033 [−0.059, −0.005] | −0.192 [−0.230, −0.152] |
| vlm_sgg_single_4b_k2 | new img + SGG | 9.34 | 5.04 | 0.090 | +0.014 [−0.004, +0.034] | 0.301 | +0.017 [−0.006, +0.040] | −0.177 [−0.209, −0.147] |
| vlm_sgg_single_9b_k2 | new img + SGG | 9.44 | 4.78 | 0.087 | +0.010 [−0.017, +0.040] | 0.281 | −0.008 [−0.032, +0.017] | −0.198 [−0.237, −0.159] |
| llm_sgg_4b_h5_k2 | legacy text cache | 14.21 | 3.27 | 0.170 | −0.002 [−0.027, +0.025] | 0.416 | −0.004 [−0.023, +0.016] | −0.062 [−0.090, −0.037] |
| llm_sgg_9b_h5_k2 | legacy text cache | 10.31 | 4.57 | 0.103 | +0.016 [−0.016, +0.048] | 0.292 | −0.028 [−0.053, −0.001] | −0.187 [−0.225, −0.148] |
| vlm_single_9b_k2 | legacy image-only VLM, θ 0.70 | 14.32 | 3.04 | 0.159 | −0.014 [−0.036, +0.008] | 0.406 | −0.017 [−0.035, +0.002] | −0.073 [−0.104, −0.044] |
| vlm_single_4b_k2 | legacy image-only VLM, θ 0.70 | 12.66 | 5.20 | 0.122 | −0.021 [−0.056, +0.014] | 0.331 | −0.048 [−0.075, −0.021] | −0.147 [−0.187, −0.106] |
| vlm_9b_single_k2 | legacy `vlm_stop.csv` (θ 0.8, stale) | 16.14 | 3.76 | 0.198 | −0.002 [−0.024, +0.019] | 0.437 | −0.012 [−0.029, +0.006] | −0.041 [−0.065, −0.019] |
| vlm_4b_single_k2 | legacy `vlm_stop.csv` (θ 0.8, stale) | 14.90 | 6.34 | 0.152 | −0.028 [−0.061, +0.002] | 0.362 | −0.072 [−0.099, −0.042] | −0.116 [−0.155, −0.079] |
| baseline | MIRU node confidence | 14.31 | 3.49 | 0.159 | −0.014 [−0.038, +0.015] | 0.391 | −0.032 [−0.054, −0.007] | −0.088 [−0.117, −0.060] |
| fixed_20 | always level 20 | 20.00 | 6.52 | 0.240 | n/a | 0.478 | n/a | n/a |

Notes:
- **The stale vlm_stop.csv threshold.** `outputs/metrics/vlm_stop.csv` was computed at θ = 0.8, although `configs/config.yaml` `vlm.saturation_threshold` now says 0.7. The θ 0.70 rows are recomputed by A2 from the same score files (`outputs/vlm_scores/qwen3.5-{4b,9b}/single`). MIRU's earlier `method_comparison_*.csv` used the θ 0.8 stops.
- **Hybrid variants.** The hybrid variants (0.5 node confidence + 0.5 score, θ 0.5) are in the source table. They track their score-only rows within 0.15 MAE.
- **Consistency check.** The MAE values were cross-checked with `scripts/40_vlm_sgg_saturation_scoring.py --summarize` and match to two decimals.
- Late-fusion rows (post hoc) are reported separately in §6.

## 3. GT-based equal-observation results
Under both GT q metrics, no method's stop is significantly better than the equal-observation fixed mix. Every `− eq-fixed` CI in §2 contains 0 or is negative. The largest point estimate is img + SGG 4b, at +0.014 (R50) and +0.017 (relsg_gt). Both CIs contain 0, and that method stops very early (mean 9.3). Every method is significantly below `fixed_20` on relsg_gt.

The fixed-stop curve (`outputs/miru_gt_eval/eval/summary.csv`, R50; `eval/q_relsg_gt/summary.csv`, relsg_gt) rises monotonically:

| fixed level | 5 | 10 | 14 | 15 | 20 |
|---|---|---|---|---|---|
| mean R50 | 0.005 | 0.083 | 0.169 | 0.181 | 0.240 |
| mean relsg_gt | 0.134 | 0.313 | 0.416 | 0.436 | 0.478 |

Per-image and bootstrap details are in `outputs/miru_gt_eval/eval/{stops,per_image}.csv` and `bootstrap.json`.

## 4. Why attaching the image makes stops earlier (diagnosis)
Full tables are in `outputs/llm_scores_vlm/diag_img_vs_text.md`. Only score files were read.
- **Paired shift.** The mean paired shift, img − text, on the same state and model is +0.13 for 4b and +0.05 for 9b.
  - States where img ≥ 0.70 but text < 0.70: 30 % (484) for 4b and 16 % (255) for 9b. The reverse occurs in 1 % and 3 %.
  - The shift peaks at t = 8–11, with 4b at +0.21 to +0.27 and 48–60 % of states flipping. These are the levels where a good stop must wait.
- **Share of states at θ (img vs text).** At t = 9, 74 % vs 16 % for 4b and 48 % vs 29 % for 9b. Over all levels, 79 % vs 49 % (4b) and 72 % vs 59 % (9b).
- **The image is not cited; graph stability is confabulated.** In flipped states, image+SGG reasons mention visual words (image, blur, sharp, clarity …) in only 1 % (4b) and 7 % (9b) of cases, against 82 % for the image-only VLM. They claim that the scene graph has stabilized in 99 % and 89 % of cases. On the identical sequence, text-only reasons call it unstable in 95 % of cases.
- **The blur cue is lost.** Asked about the image alone, the 9b VLM is level-sensitive. Its share of states with a score ≥ 0.70 is 2 % at t = 5 and 14 % at t = 9. With image + SGG it is 45 % and 48 % (4b: 27 % and 74 %).

Typical example: image 2315531, levels 6–8, 4b. Text-only scores 0.12–0.15:
> "The scene graph shows significant instability between levels (e.g., object sets changing from non-empty to empty) …"

Image + SGG scores 0.95–1.00:
> "The scene graph has reached semantic saturation as the object and relationship sets have stabilized across resolution levels with no new categories or relationships detected."

Reasons are post hoc, so the mechanism cannot be identified from text. One possibility is that the image tokens dilute attention to the long JSON sequence. Another is a "complete-looking scene" prior.

## 5. What this means
- **(a) Text-only 4b reproduces MIRU.** The re-run SGG-text 4b (k = 2) gives a mean stop of 13.76 and MAE 3.00, against 14.21 / 3.27 for the legacy cache. Per-level mean scores agree within about 0.04 (`diag_img_vs_text.md` §1). The MIRU result is robust to the newer EGTR predictions and to greedy JSON-mode decoding.
- **(b) Attaching the image to the SGG prompt hurts.** It moves stops much earlier (about 9.4 for both sizes) and worsens MAE (4b: 3.00 → 5.04; 9b: 4.40 → 4.78). The mechanism visible in the data is confabulated sequence stability plus loss of the blur cue (§4). A single added sentence is not enough to make the model use the image as evidence of missing detail.
- **(c) The image-only VLM 9b at θ 0.70 roughly equals SGG-text 4b.** The image-only 9b stops at a mean of 14.32 with MAE 3.04, against SGG-text 4b at 13.76 / 3.00 (and 3.27 for the legacy run). The two have similar GT q at the stop: R50 0.159 vs 0.167 and relsg_gt 0.406 vs 0.404. MIRU's earlier "VLM 9b is worse" reading used the stale θ 0.8 stops (MAE 3.76).
- **(d) Under GT scoring, no method beats the fixed-stop curve.** Every method's GT q at its stop is within the equal-observation fixed-mix CI (§3). Adaptive stopping matches oracle timing to within about 3 levels (MAE) but does not buy more GT recall per observed level than stopping every image at the same level. This also holds for the post-hoc late-fusion rules (§6).

## 6. Late fusion of the image-only VLM and SGG-text scores (post hoc; A2)

Source: `outputs/miru_gt_eval/miru_method_table.md`, rows marked "late fusion (zero-call)". The rules are defined in `configs/config.yaml` `evaluation_extra` (`fuse_*`) and implemented in `src/milestone1/miru_methods.py` (`fused_scores`, `fuse_series`). A fused method combines the per-level scores of the legacy image-only VLM 9b (`outputs/vlm_scores/qwen3.5-9b/single`) and the new SGG-text control, using a mean or a min. It then applies the MIRU stop rule (θ 0.70, k = 2). The `hybrid_` variants also mix in node confidence (0.5 / 0.5, θ 0.5), as in §2.

| method (k = 2) | mean stop | MAE vs stop_c09 | R50 at stop | R50 − eq-fixed | relsg_gt at stop | relsg_gt − eq-fixed | relsg_gt − fixed20 |
|---|---|---|---|---|---|---|---|
| hybrid_fuse_mean_vlmimg9b_text4b | 13.49 | **2.59** | 0.160 | +0.001 [−0.022, +0.023] | 0.397 | −0.005 [−0.022, +0.012] | −0.081 [−0.109, −0.056] |
| fuse_mean_vlmimg9b_text4b | 13.44 | 2.84 | 0.167 | +0.009 [−0.014, +0.033] | 0.407 | +0.006 [−0.010, +0.023] | −0.071 [−0.101, −0.044] |
| fuse_min_vlmimg9b_text4b | 16.16 | 3.12 | 0.201 | +0.001 [−0.017, +0.018] | 0.455 | +0.006 [−0.007, +0.020] | **−0.023** [−0.042, −0.004] |
| hybrid_fuse_min_vlmimg9b_text4b | 15.28 | 2.92 | 0.185 | −0.001 [−0.025, +0.022] | 0.435 | −0.005 [−0.023, +0.013] | −0.044 [−0.068, −0.020] |
| fuse_mean_vlmimg9b_text9b | 12.41 | 2.93 | 0.128 | −0.011 [−0.033, +0.011] | 0.357 | −0.016 [−0.033, +0.001] | −0.121 [−0.153, −0.091] |
| hybrid_fuse_mean_vlmimg9b_text9b | 12.93 | 2.71 | 0.140 | −0.007 [−0.034, +0.018] | 0.376 | −0.010 [−0.030, +0.009] | −0.102 [−0.134, −0.072] |

Reading:
- **Mean fusion.** Fusing the two complementary single-modality judges at the score level, instead of in one prompt, gives the lowest MAE observed. hybrid_fuse_mean_vlmimg9b_text4b reaches 2.59, against 3.00 for SGG-text 4b alone and 3.04 for the image-only 9b alone, at about the same mean stop (13.5).
- **Min fusion.** It stops later (16.2) and comes closest to the full-observation GT quality: relsg_gt 0.455, only −0.023 below `fixed_20`, at 81 % of the observation levels. Its MAE is 3.12.
- **Equal-observation comparison.** Fusion still does not beat the equal-observation fixed mix under GT scoring. All `− eq-fixed` CIs contain 0, so (d) in §5 also holds for fusion.

**Caveat (post hoc).** The fusion operators, the source pairs (image 9b with text 4b), the hybrid weighting and θ were all chosen after seeing these same 100 images and their oracle stops. Six fusion variants were inspected. The MAE gains (2.59 vs 3.00) are therefore optimistic in-sample estimates. They are not evidence of a better method until they are confirmed on held-out images. The pre-registered rule for that confirmation is `hybrid_fuse_mean(vlm image-only 9b, llm_sgg text 4b)`, θ 0.70, k = 2, with `fuse_min` as the secondary rule. The held-out set is ROBOVIS test100 minus the 4 images that overlap the MIRU 100.

## 7. Files
- Scoring code: `scripts/40_vlm_sgg_saturation_scoring.py`, with tests in `tests/test_vlm_sgg_scoring.py`. Settings are in the `configs/config.yaml` `vlm_sgg` block.
- Scores: `outputs/llm_scores_vlm/qwen3.5-{4b,9b}_h5_rel_on_conf_on_{text,img_single}/`. Legacy scores: `outputs/llm_ablation/`, `outputs/vlm_scores/`.
- Evaluation (A2): `scripts/42_miru_gt_eval.py` → `outputs/miru_gt_eval/{miru_method_table.md,eval/}`.
- Diagnosis: `outputs/llm_scores_vlm/diag_img_vs_text.md`.
