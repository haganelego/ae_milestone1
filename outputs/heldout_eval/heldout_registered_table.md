# Held-out confirmatory results (pre-registered 2026-10-08 00:40; no tuning)

n = 96 images (all) and 73 with stop_c09 <= 18 (MIRU's selection rule). MAE = |stop - stop_c09| (EGTR-L20 top-K oracle). Δ legacy mix = q(stop) minus the adjacent two-level fixed mix at the method's mean stop (uses the level index; previous plan's registered comparison) with 95% CI; Δ F = point difference to the best fixed mixture (schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator).

| method | role | mean stop | MAE (all) | MAE (≤18) | R50 | Δ R50 vs legacy mix [CI] | Δ R50 vs F | relsg | Δ relsg vs legacy mix [CI] | Δ relsg vs F |
|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_fuse_mean_vlmimg9b_text4b_k2 | PRIMARY (pre-registered) | 13.79 | 3.35 | 2.93 | 0.109 | -0.008 [-0.026, +0.007] | -0.018 | 0.360 | -0.016 [-0.033, +0.004] | -0.018 |
| fuse_min_vlmimg9b_text4b_k2 | secondary (pre-registered) | 16.04 | 2.54 | 2.47 | 0.138 | -0.005 [-0.016, +0.004] | -0.009 | 0.398 | -0.001 [-0.013, +0.011] | -0.004 |
| fuse_mean_vlmimg9b_text4b_k2 | other | 13.51 | 3.39 | 2.82 | 0.110 | -0.002 [-0.022, +0.017] | -0.014 | 0.356 | -0.011 [-0.033, +0.012] | -0.017 |
| hybrid_llm_sgg_text_4b_k2 | other | 13.46 | 3.77 | 3.14 | 0.113 | +0.001 [-0.017, +0.019] | -0.011 | 0.357 | -0.008 [-0.030, +0.015] | -0.015 |
| llm_sgg_text_4b_k2 | comparator | 13.73 | 3.56 | 3.04 | 0.110 | -0.007 [-0.026, +0.011] | -0.017 | 0.357 | -0.016 [-0.039, +0.008] | -0.020 |
| vlm_single_9b_k2 | comparator | 14.17 | 3.21 | 2.67 | 0.115 | -0.010 [-0.032, +0.011] | -0.016 | 0.360 | -0.024 [-0.049, -0.001] | -0.024 |
| baseline | comparator | 14.73 | 3.46 | 3.40 | 0.108 | -0.026 [-0.050, -0.004] | -0.028 | 0.361 | -0.029 [-0.048, -0.010] | -0.029 |
| fixed_20 | schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator | 20.00 | 4.10 | 5.37 | 0.159 | — | — | 0.430 | — | — |

## Significance vs level-agnostic comparators (paired bootstrap 95% CI; paired sign-flip permutation)

✓ = CI excludes 0 or permutation p < .05; direction = whether the pre-registered method is better (lower |stop − c09|, higher q).

| subset | method | vs | column | Δ (method − comparator) [95% CI] | p better / worse | sig |
|---|---|---|---|---|---|---|
| all (n=96) | hybrid_fuse_mean_vlmimg9b_text4b_k2 | text 4b | abs_err_c09 | -0.208 [-0.823, +0.344] | 0.262 / 0.761 | ✗ (better) |
| all (n=96) | hybrid_fuse_mean_vlmimg9b_text4b_k2 | text 4b | R50 | -0.000 [-0.017, +0.018] | 0.517 / 0.484 | ✗ (worse) |
| all (n=96) | hybrid_fuse_mean_vlmimg9b_text4b_k2 | text 4b | relsg_gt | +0.003 [-0.018, +0.025] | 0.413 / 0.587 | ✗ (better) |
| all (n=96) | hybrid_fuse_mean_vlmimg9b_text4b_k2 | image-only 9b | abs_err_c09 | +0.146 [-0.365, +0.615] | 0.716 / 0.308 | ✗ (worse) |
| all (n=96) | hybrid_fuse_mean_vlmimg9b_text4b_k2 | image-only 9b | R50 | -0.005 [-0.026, +0.017] | 0.682 / 0.318 | ✗ (worse) |
| all (n=96) | hybrid_fuse_mean_vlmimg9b_text4b_k2 | image-only 9b | relsg_gt | +0.000 [-0.025, +0.028] | 0.497 / 0.503 | ✗ (better) |
| all (n=96) | hybrid_fuse_mean_vlmimg9b_text4b_k2 | node-confidence baseline | abs_err_c09 | -0.104 [-0.615, +0.406] | 0.359 / 0.667 | ✗ (better) |
| all (n=96) | hybrid_fuse_mean_vlmimg9b_text4b_k2 | node-confidence baseline | R50 | +0.001 [-0.018, +0.022] | 0.470 / 0.530 | ✗ (better) |
| all (n=96) | hybrid_fuse_mean_vlmimg9b_text4b_k2 | node-confidence baseline | relsg_gt | -0.001 [-0.018, +0.017] | 0.527 / 0.473 | ✗ (worse) |
| all (n=96) | fuse_min_vlmimg9b_text4b_k2 | text 4b | abs_err_c09 | -1.021 [-1.625, -0.448] | 0.000 / 1.000 | ✓ better |
| all (n=96) | fuse_min_vlmimg9b_text4b_k2 | text 4b | R50 | +0.029 [+0.010, +0.049] | 0.001 / 1.000 | ✓ better |
| all (n=96) | fuse_min_vlmimg9b_text4b_k2 | text 4b | relsg_gt | +0.041 [+0.021, +0.063] | 0.000 / 1.000 | ✓ better |
| all (n=96) | fuse_min_vlmimg9b_text4b_k2 | image-only 9b | abs_err_c09 | -0.667 [-1.167, -0.156] | 0.006 / 0.996 | ✓ better |
| all (n=96) | fuse_min_vlmimg9b_text4b_k2 | image-only 9b | R50 | +0.024 [+0.007, +0.042] | 0.005 / 0.995 | ✓ better |
| all (n=96) | fuse_min_vlmimg9b_text4b_k2 | image-only 9b | relsg_gt | +0.038 [+0.016, +0.064] | 0.001 / 0.999 | ✓ better |
| all (n=96) | fuse_min_vlmimg9b_text4b_k2 | node-confidence baseline | abs_err_c09 | -0.917 [-1.594, -0.260] | 0.004 / 0.997 | ✓ better |
| all (n=96) | fuse_min_vlmimg9b_text4b_k2 | node-confidence baseline | R50 | +0.030 [+0.007, +0.053] | 0.006 / 0.994 | ✓ better |
| all (n=96) | fuse_min_vlmimg9b_text4b_k2 | node-confidence baseline | relsg_gt | +0.038 [+0.016, +0.060] | 0.001 / 1.000 | ✓ better |
| stop_c09<=18 (n=73) | hybrid_fuse_mean_vlmimg9b_text4b_k2 | text 4b | abs_err_c09 | -0.110 [-0.658, +0.384] | 0.348 / 0.696 | ✗ (better) |
| stop_c09<=18 (n=73) | hybrid_fuse_mean_vlmimg9b_text4b_k2 | text 4b | R50 | +0.001 [-0.022, +0.026] | 0.482 / 0.520 | ✗ (better) |
| stop_c09<=18 (n=73) | hybrid_fuse_mean_vlmimg9b_text4b_k2 | text 4b | relsg_gt | -0.002 [-0.028, +0.026] | 0.564 / 0.436 | ✗ (worse) |
| stop_c09<=18 (n=73) | hybrid_fuse_mean_vlmimg9b_text4b_k2 | image-only 9b | abs_err_c09 | +0.260 [-0.247, +0.767] | 0.832 / 0.198 | ✗ (worse) |
| stop_c09<=18 (n=73) | hybrid_fuse_mean_vlmimg9b_text4b_k2 | image-only 9b | R50 | -0.005 [-0.032, +0.023] | 0.628 / 0.373 | ✗ (worse) |
| stop_c09<=18 (n=73) | hybrid_fuse_mean_vlmimg9b_text4b_k2 | image-only 9b | relsg_gt | -0.000 [-0.031, +0.034] | 0.515 / 0.485 | ✗ (worse) |
| stop_c09<=18 (n=73) | hybrid_fuse_mean_vlmimg9b_text4b_k2 | node-confidence baseline | abs_err_c09 | -0.466 [-1.055, +0.082] | 0.062 / 0.948 | ✗ (better) |
| stop_c09<=18 (n=73) | hybrid_fuse_mean_vlmimg9b_text4b_k2 | node-confidence baseline | R50 | +0.004 [-0.019, +0.029] | 0.387 / 0.614 | ✗ (better) |
| stop_c09<=18 (n=73) | hybrid_fuse_mean_vlmimg9b_text4b_k2 | node-confidence baseline | relsg_gt | +0.002 [-0.018, +0.023] | 0.404 / 0.596 | ✗ (better) |
| stop_c09<=18 (n=73) | fuse_min_vlmimg9b_text4b_k2 | text 4b | abs_err_c09 | -0.575 [-1.137, -0.027] | 0.029 / 0.975 | ✓ better |
| stop_c09<=18 (n=73) | fuse_min_vlmimg9b_text4b_k2 | text 4b | R50 | +0.040 [+0.016, +0.067] | 0.000 / 1.000 | ✓ better |
| stop_c09<=18 (n=73) | fuse_min_vlmimg9b_text4b_k2 | text 4b | relsg_gt | +0.042 [+0.018, +0.067] | 0.000 / 1.000 | ✓ better |
| stop_c09<=18 (n=73) | fuse_min_vlmimg9b_text4b_k2 | image-only 9b | abs_err_c09 | -0.205 [-0.685, +0.342] | 0.268 / 0.766 | ✗ (better) |
| stop_c09<=18 (n=73) | fuse_min_vlmimg9b_text4b_k2 | image-only 9b | R50 | +0.034 [+0.015, +0.057] | 0.000 / 1.000 | ✓ better |
| stop_c09<=18 (n=73) | fuse_min_vlmimg9b_text4b_k2 | image-only 9b | relsg_gt | +0.044 [+0.016, +0.075] | 0.003 / 0.997 | ✓ better |
| stop_c09<=18 (n=73) | fuse_min_vlmimg9b_text4b_k2 | node-confidence baseline | abs_err_c09 | -0.932 [-1.630, -0.233] | 0.011 / 0.991 | ✓ better |
| stop_c09<=18 (n=73) | fuse_min_vlmimg9b_text4b_k2 | node-confidence baseline | R50 | +0.043 [+0.014, +0.072] | 0.003 / 0.997 | ✓ better |
| stop_c09<=18 (n=73) | fuse_min_vlmimg9b_text4b_k2 | node-confidence baseline | relsg_gt | +0.047 [+0.022, +0.073] | 0.000 / 1.000 | ✓ better |
