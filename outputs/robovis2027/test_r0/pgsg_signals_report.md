# PGSG-internal self-consistency signals (`pgsg_internal`), test_r0

The signals are computed from the cached graphs with zero model calls, by
`scripts/34_robovis_pgsg_signals.py` (code in `src/milestone1/pgsg_signals.py`).
Per-state values are in `pgsg_signals.csv` for dev_r0 and test_r0. The correlations
below are in `pgsg_signals_corr.csv`. The test set has n = 2000 states (100 images × 20
levels).

**Conclusion.** No signal is a useful zero-call stop signal beyond the level index
itself. Several signals reach |rho| ≥ 0.3 with gain_left, but only as weaker proxies of
t. At a fixed level they carry almost no information across images.

## Spearman rho, shown as pooled / within

`pooled` uses all 2000 states. `within` is the mean over images of the per-image rho
across the 20 levels. gain_left = qmax − q_t.

| signal | t | gain_left R50 | q/qmax R50 | gain_left relsg_gt | q/qmax relsg_gt |
|---|---|---|---|---|---|
| n_entities | -0.31 / -0.40 | -0.02 / +0.32 | -0.46 / -0.32 | +0.31 / +0.36 | -0.45 / -0.36 |
| n_unique_labels | -0.22 / -0.26 | +0.01 / +0.19 | -0.29 / -0.19 | +0.22 / +0.23 | -0.36 / -0.23 |
| bad_inst_rate | +0.06 / +0.07 | -0.02 / -0.06 | +0.11 / +0.06 | -0.05 / -0.06 | +0.06 / +0.06 |
| mean_mentions_top20 | +0.23 / +0.28 | +0.07 / -0.25 | +0.41 / +0.25 | -0.25 / -0.31 | +0.42 / +0.31 |
| frac_rel50_agree2 (= agreement) | +0.29 / +0.31 | +0.01 / -0.26 | +0.36 / +0.26 | -0.29 / -0.31 | +0.40 / +0.31 |
| mean_label_maxprob_top20 | -0.04 / -0.03 | -0.11 / +0.05 | -0.22 / -0.05 | +0.14 / +0.13 | -0.27 / -0.13 |
| mean_pred_maxprob_top50 | +0.10 / +0.11 | +0.02 / -0.02 | +0.07 / +0.02 | -0.06 / -0.09 | +0.07 / +0.09 |

These pairs reach |rho| ≥ 0.3 with gain_left:

| signal | q | basis | rho |
|---|---|---|---|
| n_entities | R50 | within | +0.32 |
| n_entities | relsg_gt | pooled | +0.31 |
| n_entities | relsg_gt | within | +0.36 |
| agreement | relsg_gt | within | -0.31 |
| mean_mentions_top20 | relsg_gt | within | -0.31 |

## Level confound check

Every one of these hits is weaker than the level index alone:

| target | rho with t (pooled) | rho with t (within) |
|---|---|---|
| gain_left R50 | -0.28 | -0.59 |
| gain_left relsg_gt | -0.60 | -0.68 |

At a fixed level, the cross-image Spearman between signal and gain_left is small and
unstable. Averaged over t = 1..19, |rho| ≤ 0.16 for every signal. The sign flips
between levels. For example, agreement vs relsg gain_left is -0.32 at t = 5, +0.08 at
t = 10 and -0.35 at t = 15. mean_mentions_top20 vs R50 gain_left is +0.25 at t = 5, which
is the wrong direction for a stop cue.

## Reading

- **Main trend.** PGSG becomes more self-consistent as resolution recovers. It emits
  fewer entities (n_entities, rho -0.40 within with t), and its top entities get more
  repeated mentions across the 36 segments.
- **What the stop-relevant hits mean.** The within-image hits mostly re-express t.
  Within an image, a signal that only tracks t is as good as a fixed schedule.
- **No extra stop information.** The signals do not add cross-image information about
  which image has saturated at a given level.
- **Unhelpful signals.** bad_inst_rate and predicate confidence are flat.
- **Recommendation.** Keep `pgsg_internal` as a reported negative control or diagnostic
  only. Do not use it as a stop condition. If a zero-call baseline is wanted, a fixed
  level is the stronger and simpler choice.
