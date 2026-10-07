| method | source | mean_stop | mae_c09 | medae_c09 | mae_c09_sat | n_sat | at_final_level | R50_stop | R50_minus_eq_fixed_ci | R50_minus_F | relsg_gt_stop | relsg_gt_minus_eq_fixed_ci | relsg_gt_minus_F |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | MIRU CSV (node confidence) | 14.310 | 3.490 | 2.500 | 3.490 | 100 | 0.220 | 0.159 | -0.014 [-0.038, +0.015] | -0.019 | 0.391 | -0.032 [-0.054, -0.007] | -0.032 |
| llm_sgg_4b_h5_k2 | legacy cache (llm_ablation: temp 0.3, gate 0.3) | 14.210 | 3.270 | 2.000 | 3.270 | 100 | 0.070 | 0.170 | -0.002 [-0.027, +0.025] | -0.007 | 0.416 | -0.004 [-0.023, +0.016] | -0.004 |
| llm_sgg_9b_h5_k2 | legacy cache (llm_ablation: temp 0.3, gate 0.3) | 10.310 | 4.570 | 4.000 | 4.570 | 100 | 0.010 | 0.103 | +0.016 [-0.016, +0.048] | -0.021 | 0.292 | -0.028 [-0.053, -0.001] | -0.029 |
| hybrid_4b_k2 | legacy cache (llm_ablation: temp 0.3, gate 0.3) | 13.920 | 2.820 | 2.000 | 2.820 | 100 | 0.050 | 0.164 | -0.004 [-0.030, +0.025] | -0.009 | 0.407 | -0.007 [-0.027, +0.015] | -0.007 |
| hybrid_9b_k2 | legacy cache (llm_ablation: temp 0.3, gate 0.3) | 10.140 | 4.400 | 4.000 | 4.400 | 100 | 0.010 | 0.099 | +0.014 [-0.019, +0.047] | -0.024 | 0.286 | -0.029 [-0.054, +0.001] | -0.030 |
| vlm_4b_single_k2 | legacy cache (vlm_stop.csv: threshold 0.8) | 14.900 | 6.340 | 6.000 | 6.340 | 100 | 0.570 | 0.152 | -0.028 [-0.061, +0.002] | -0.034 | 0.362 | -0.072 [-0.099, -0.042] | -0.072 |
| vlm_9b_single_k2 | legacy cache (vlm_stop.csv: threshold 0.8) | 16.140 | 3.760 | 3.000 | 3.760 | 100 | 0.170 | 0.198 | -0.002 [-0.024, +0.019] | -0.005 | 0.437 | -0.012 [-0.029, +0.006] | -0.012 |
| vlm_single_4b_k2 | legacy cache (vlm_scores, recomputed at 0.70) | 12.660 | 5.200 | 5.000 | 5.200 | 100 | 0.180 | 0.122 | -0.021 [-0.056, +0.014] | -0.034 | 0.331 | -0.048 [-0.075, -0.021] | -0.051 |
| vlm_single_9b_k2 | legacy cache (vlm_scores, recomputed at 0.70) | 14.320 | 3.040 | 2.000 | 3.040 | 100 | 0.020 | 0.159 | -0.014 [-0.036, +0.008] | -0.019 | 0.406 | -0.017 [-0.035, +0.002] | -0.017 |
| llm_sgg_text_4b_k2 | score dir | 13.760 | 3.000 | 2.000 | 3.000 | 100 | 0.030 | 0.167 | +0.003 [-0.024, +0.032] | -0.004 | 0.404 | -0.005 [-0.026, +0.015] | -0.006 |
| llm_sgg_text_9b_k2 | score dir | 10.320 | 4.400 | 4.000 | 4.400 | 100 | 0.010 | 0.098 | +0.011 [-0.021, +0.042] | -0.027 | 0.287 | -0.033 [-0.059, -0.005] | -0.034 |
| vlm_sgg_single_4b_k2 | score dir | 9.340 | 5.040 | 6.000 | 5.040 | 100 | 0.010 | 0.090 | +0.014 [-0.004, +0.034] | -0.022 | 0.301 | +0.017 [-0.006, +0.040] | 0.008 |
| vlm_sgg_single_9b_k2 | score dir, coverage 100% | 9.440 | 4.780 | 4.000 | 4.780 | 100 | 0.000 | 0.087 | +0.010 [-0.017, +0.040] | -0.026 | 0.281 | -0.008 [-0.032, +0.017] | -0.016 |
| hybrid_llm_sgg_text_4b_k2 | score dir | 13.580 | 2.980 | 2.000 | 2.980 | 100 | 0.030 | 0.165 | +0.004 [-0.023, +0.033] | -0.004 | 0.399 | -0.005 [-0.027, +0.016] | -0.006 |
| hybrid_llm_sgg_text_9b_k2 | score dir | 10.170 | 4.250 | 3.000 | 4.250 | 100 | 0.010 | 0.091 | +0.006 [-0.025, +0.036] | -0.032 | 0.280 | -0.036 [-0.061, -0.006] | -0.037 |
| hybrid_vlm_sgg_single_4b_k2 | score dir | 9.320 | 5.060 | 6.000 | 5.060 | 100 | 0.010 | 0.090 | +0.014 [-0.004, +0.035] | -0.022 | 0.301 | +0.018 [-0.006, +0.040] | 0.008 |
| hybrid_vlm_sgg_single_9b_k2 | score dir, coverage 100% | 9.530 | 4.690 | 4.000 | 4.690 | 100 | 0.000 | 0.088 | +0.010 [-0.018, +0.040] | -0.027 | 0.284 | -0.009 [-0.033, +0.017] | -0.015 |
| fuse_mean_vlmimg9b_text4b_k2 | late fusion (zero-call) | 13.440 | 2.840 | 2.000 | 2.840 | 100 | 0.000 | 0.167 | +0.009 [-0.014, +0.033] | 0.000 | 0.407 | +0.006 [-0.010, +0.023] | 0.005 |
| fuse_min_vlmimg9b_text4b_k2 | late fusion (zero-call) | 16.160 | 3.120 | 2.000 | 3.120 | 100 | 0.080 | 0.201 | +0.001 [-0.017, +0.018] | -0.002 | 0.455 | +0.006 [-0.007, +0.020] | 0.006 |
| fuse_mean_vlmimg9b_text9b_k2 | late fusion (zero-call) | 12.410 | 2.930 | 2.000 | 2.930 | 100 | 0.000 | 0.128 | -0.011 [-0.033, +0.011] | -0.024 | 0.357 | -0.016 [-0.033, +0.001] | -0.018 |
| hybrid_fuse_mean_vlmimg9b_text4b_k2 | late fusion (zero-call) | 13.490 | 2.590 | 2.000 | 2.590 | 100 | 0.010 | 0.160 | +0.001 [-0.022, +0.023] | -0.008 | 0.397 | -0.005 [-0.022, +0.012] | -0.006 |
| hybrid_fuse_min_vlmimg9b_text4b_k2 | late fusion (zero-call) | 15.280 | 2.920 | 2.000 | 2.920 | 100 | 0.070 | 0.185 | -0.001 [-0.025, +0.022] | -0.006 | 0.435 | -0.005 [-0.023, +0.013] | -0.005 |
| hybrid_fuse_mean_vlmimg9b_text9b_k2 | late fusion (zero-call) | 12.930 | 2.710 | 2.000 | 2.710 | 100 | 0.000 | 0.140 | -0.007 [-0.034, +0.018] | -0.020 | 0.376 | -0.010 [-0.030, +0.009] | -0.012 |
| fixed_20 | schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator | 20.000 | 6.520 | 6.000 | 6.520 | 100 | 1.000 | 0.240 | n/a | n/a | 0.478 | n/a | n/a |

- `*_minus_eq_fixed_ci`: legacy equal-observation comparator (adjacent two-level fixed mix at the method's mean stop; uses the level index).
- `*_minus_F`: q_stop - F(mean_stop), point estimate only; F = best fixed mixture on this set: schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator.
- fixed_20: schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator.
