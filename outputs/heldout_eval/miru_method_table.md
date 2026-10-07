| method | source | mean_stop | mae_c09 | medae_c09 | mae_c09_sat | n_sat | at_final_level | R50_stop | R50_minus_eq_fixed_ci | R50_minus_F | relsg_gt_stop | relsg_gt_minus_eq_fixed_ci | relsg_gt_minus_F |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_fuse_mean_vlmimg9b_text4b_k2 | late fusion (zero-call) | 13.792 | 3.354 | 3.000 | 2.932 | 73 | 0.062 | 0.109 | -0.008 [-0.026, +0.007] | -0.018 | 0.360 | -0.016 [-0.033, +0.004] | -0.018 |
| fuse_min_vlmimg9b_text4b_k2 | late fusion (zero-call) | 16.042 | 2.542 | 2.000 | 2.466 | 73 | 0.135 | 0.138 | -0.005 [-0.016, +0.004] | -0.009 | 0.398 | -0.001 [-0.013, +0.011] | -0.004 |
| fuse_mean_vlmimg9b_text4b_k2 | late fusion (zero-call) | 13.510 | 3.385 | 3.000 | 2.822 | 73 | 0.031 | 0.110 | -0.002 [-0.022, +0.017] | -0.014 | 0.356 | -0.011 [-0.033, +0.012] | -0.017 |
| llm_sgg_text_4b_k2 | score dir | 13.729 | 3.562 | 3.000 | 3.041 | 73 | 0.083 | 0.110 | -0.007 [-0.026, +0.011] | -0.017 | 0.357 | -0.016 [-0.039, +0.008] | -0.020 |
| vlm_single_9b_k2 | score dir | 14.167 | 3.208 | 3.000 | 2.671 | 73 | 0.021 | 0.115 | -0.010 [-0.032, +0.011] | -0.016 | 0.360 | -0.024 [-0.049, -0.001] | -0.024 |
| hybrid_llm_sgg_text_4b_k2 | score dir | 13.458 | 3.771 | 3.000 | 3.137 | 73 | 0.062 | 0.113 | +0.001 [-0.017, +0.019] | -0.011 | 0.357 | -0.008 [-0.030, +0.015] | -0.015 |
| baseline | node confidence >= 0.35 (from predictions) | 14.729 | 3.458 | 2.000 | 3.397 | 73 | 0.302 | 0.108 | -0.026 [-0.050, -0.004] | -0.028 | 0.361 | -0.029 [-0.048, -0.010] | -0.029 |
| fixed_20 | schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator | 20.000 | 4.104 | 4.000 | 5.370 | 73 | 1.000 | 0.159 | n/a | n/a | 0.430 | n/a | n/a |

- `*_minus_eq_fixed_ci`: legacy equal-observation fixed mix (adjacent two levels at the method's mean stop) — uses the level index; previous plan's registered comparison.
- `*_minus_F`: q_stop - F(mean_stop), point estimate only; F = best fixed mixture on this set: schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator.
- fixed_20: schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator.
