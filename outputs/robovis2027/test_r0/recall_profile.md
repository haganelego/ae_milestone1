# ROBOVIS2027 test_r0 recall profile

Source files are `outputs/robovis2027/test_r0/stage_summary.json` and `stage_metrics.csv`.
dev_r0 is shown for comparison. No model calls were made.
The oracle is `milestone1.robovis_eval.relative_stable_oracle`, imported read-only, with
`oracle_ratio` = 0.9 from `configs/robovis2027.yaml`. A level counts as defined
at the first t where q_u >= 0.9 * qmax holds for every u >= t.
All 100 test images and all 20 dev images have all 20 levels valid.

## 1. test100 recall per level (%, graph-constrained SGDet, GT coordinates, n = 100)

| t | scale | R@50 | R@100 | mR@50 (official, /50) |
|---|---|---|---|---|
| 1 | 0.05 | 0.42 | 0.84 | 0.03 |
| 2 | 0.0585 | 1.48 | 1.81 | 0.14 |
| 3 | 0.0685 | 1.43 | 2.22 | 0.20 |
| 4 | 0.0802 | 2.03 | 2.36 | 0.25 |
| 5 | 0.094 | 3.76 | 4.07 | 0.68 |
| 6 | 0.11 | 3.78 | 3.97 | 0.71 |
| 7 | 0.1288 | 4.88 | 5.38 | 1.40 |
| 8 | 0.1508 | 6.44 | 7.03 | 1.27 |
| 9 | 0.1765 | 6.87 | 8.33 | 1.34 |
| 10 | 0.2066 | 9.10 | 9.81 | 2.82 |
| 11 | 0.2419 | 10.76 | 12.15 | 4.08 |
| 12 | 0.2832 | 11.24 | 12.54 | 4.11 |
| 13 | 0.3316 | 11.10 | 13.27 | 4.10 |
| 14 | 0.3882 | 10.89 | 11.64 | 3.95 |
| 15 | 0.4545 | 12.29 | 13.40 | 4.87 |
| 16 | 0.5321 | 12.55 | 13.72 | 4.72 |
| 17 | 0.6229 | 14.46 | 15.57 | 7.01 |
| 18 | 0.7293 | 13.24 | 15.83 | 4.93 |
| 19 | 0.8538 | 14.24 | 16.52 | 6.02 |
| 20 | 1.0 | 14.04 | 16.22 | 4.72 |

The curve rises almost monotonically up to t = 11 (scale 0.24) and is nearly flat after
that. Mean R@50 is 10.8 at t = 11 and 14.0 at t = 20, with a local peak of 14.5 at t = 17.

## 2. Per-image qmax (max over the 20 levels)

| run | k | qmax = 0 | (0, 0.1] | (0.1, 0.2] | (0.2, 0.4] | (0.4, 1.0] | mean qmax | median qmax | mean q at t=20 |
|---|---|---|---|---|---|---|---|---|---|
| test_r0 | R@50 | 28 | 11 | 21 | 24 | 16 | 0.205 | 0.143 | 0.140 |
| test_r0 | R@100 | 23 | 12 | 18 | 29 | 18 | 0.234 | 0.183 | 0.162 |
| dev_r0 | R@50 | 7 | 1 | 0 | 6 | 6 | 0.286 | 0.323 | 0.196 |
| dev_r0 | R@100 | 5 | 2 | 1 | 6 | 6 | 0.298 | 0.323 | 0.249 |

For test, qmax > 0.2 holds for 40/100 images at R@50 and
47/100 at R@100. The median level of the per-image argmax is
t = 12. The mean qmax is above the mean of q at t = 20
(R@50 0.205 vs 0.140). Each image's curve is one stochastic sample per level, so qmax is
a maximum over 20 noisy draws and is biased upward.

## 3–4. Oracle status and saturation level (oracle t among defined images)

| run | k | n | defined | unstable_tail | zero_recovery | mean t | median t | range |
|---|---|---|---|---|---|---|---|---|
| test_r0 | R@50 | 100 | 37 | 35 | 28 | 16.38 | 18.0 | 7–20 |
| test_r0 | R@100 | 100 | 36 | 41 | 23 | 17.94 | 19.5 | 7–20 |
| dev_r0 | R@50 | 20 | 6 | 7 | 7 | 16.83 | 18.5 | 10–20 |
| dev_r0 | R@100 | 20 | 9 | 6 | 5 | 17.33 | 19.0 | 10–20 |

## Reading

- **Defined fraction.** Only about a third of test images have a defined oracle:
  37/100 at R@50 and 36/100 at R@100. dev20 is lower at R@50 (6/20, 30%) and higher at
  R@100 (9/20, 45%). The test result is consistent with dev, and the larger n makes the
  estimate tighter.
- **Zero recovery.** 28 test images never recover any GT relation in the top 50 at any
  level. 23 never do in the top 100. Raising K mostly converts zero_recovery images
  into unstable_tail images, not into defined ones.
- **Unstable tail.** 35 test images (R@50) and 41 (R@100) have qmax > 0, but their tail
  drops below 0.9 × qmax again. With a few GT relations per image, one recalled triplet
  is worth 0.1–0.3 of recall. Per-level sampling noise is of the same size, as shown by
  the dev20 seed spread at level 20 in `dev_r0/official_eval/summary.json`.
- **Late saturation.** Among defined images the oracle saturation level is late. The
  mean is t = 16.4 at R@50 (median 18) and t = 17.9 at R@100 (median 19.5). Many defined
  images reach the threshold only in the last 2–3 levels, so the "defined" set is
  partly an artifact of the strict all-suffix rule on a noisy curve.


## 5. GT-referenced RelSGScore (q = relsg_gt)

The score is MIRU's label-set RelSGScore, referenced to VG-150 GT instead of EGTR L20.
It is computed as 0.5 · node_recall + 0.5 · triplet_recall over label sets, with no
boxes. If a GT set is empty, the other component is used alone. The GT sets are the
object class set and the (sub, pred, obj) class-triplet set of `test.json` + `rel.json`.
The prediction sets are capped at the top 20 entities by `score` and the top 50
relations by `trp_score`. The caps are set by `evaluation.relsg_top_entities` and
`evaluation.relsg_top_relations`. `relsg_gt_uncapped` uses every entity and relation.
The precision columns are computed on the capped sets. The values are in
`stage_metrics.csv` and in the `stage_summary.json` levels. The code is
`milestone1.robovis_sgdet.relsg_score`. Raw profile numbers are in `relsg_profile.json`.

### test100 per level (means over 100 images)

| t | R@50 | relsg_gt | relsg_gt_uncapped | node_recall_gt | trip_recall_gt | node_prec_gt | trip_prec_gt |
|---|---|---|---|---|---|---|---|
| 1 | 0.004 | 0.139 | 0.358 | 0.235 | 0.043 | 0.231 | 0.006 |
| 2 | 0.015 | 0.176 | 0.402 | 0.281 | 0.072 | 0.244 | 0.009 |
| 3 | 0.014 | 0.213 | 0.440 | 0.334 | 0.091 | 0.286 | 0.011 |
| 4 | 0.020 | 0.237 | 0.465 | 0.359 | 0.115 | 0.305 | 0.015 |
| 5 | 0.038 | 0.276 | 0.483 | 0.412 | 0.140 | 0.341 | 0.018 |
| 6 | 0.038 | 0.318 | 0.532 | 0.455 | 0.181 | 0.375 | 0.023 |
| 7 | 0.049 | 0.349 | 0.557 | 0.503 | 0.196 | 0.412 | 0.024 |
| 8 | 0.064 | 0.378 | 0.588 | 0.516 | 0.240 | 0.418 | 0.030 |
| 9 | 0.069 | 0.401 | 0.610 | 0.549 | 0.253 | 0.468 | 0.032 |
| 10 | 0.091 | 0.405 | 0.614 | 0.548 | 0.263 | 0.465 | 0.034 |
| 11 | 0.108 | 0.448 | 0.655 | 0.581 | 0.315 | 0.491 | 0.041 |
| 12 | 0.112 | 0.447 | 0.646 | 0.580 | 0.314 | 0.490 | 0.039 |
| 13 | 0.111 | 0.461 | 0.667 | 0.603 | 0.319 | 0.503 | 0.040 |
| 14 | 0.109 | 0.468 | 0.661 | 0.604 | 0.332 | 0.505 | 0.042 |
| 15 | 0.123 | 0.473 | 0.670 | 0.605 | 0.342 | 0.514 | 0.043 |
| 16 | 0.125 | 0.484 | 0.677 | 0.617 | 0.350 | 0.522 | 0.044 |
| 17 | 0.145 | 0.491 | 0.670 | 0.627 | 0.355 | 0.523 | 0.045 |
| 18 | 0.132 | 0.491 | 0.681 | 0.640 | 0.342 | 0.533 | 0.044 |
| 19 | 0.142 | 0.501 | 0.694 | 0.643 | 0.358 | 0.538 | 0.046 |
| 20 | 0.140 | 0.492 | 0.690 | 0.639 | 0.345 | 0.543 | 0.044 |

### dev20 (selected levels)

| t | R@50 | relsg_gt | relsg_gt_uncapped | node_recall_gt | trip_recall_gt | node_prec_gt | trip_prec_gt |
|---|---|---|---|---|---|---|---|
| 1 | 0.000 | 0.132 | 0.398 | 0.205 | 0.060 | 0.171 | 0.006 |
| 5 | 0.007 | 0.274 | 0.475 | 0.417 | 0.132 | 0.312 | 0.014 |
| 10 | 0.134 | 0.457 | 0.637 | 0.613 | 0.300 | 0.421 | 0.027 |
| 15 | 0.201 | 0.538 | 0.700 | 0.662 | 0.414 | 0.492 | 0.038 |
| 20 | 0.196 | 0.596 | 0.754 | 0.726 | 0.465 | 0.542 | 0.046 |

### Per-image qmax of relsg

| run | q | = 0 | (0, 0.2] | (0.2, 0.4] | (0.4, 0.6] | (0.6, 1.0] | mean qmax | mean q at t=20 |
|---|---|---|---|---|---|---|---|---|
| test_r0 | relsg_gt | 0 | 0 | 18 | 36 | 46 | 0.604 | 0.492 |
| test_r0 | relsg_gt_uncapped | 0 | 0 | 0 | 8 | 92 | 0.800 | 0.690 |
| dev_r0 | relsg_gt | 0 | 0 | 0 | 9 | 11 | 0.647 | 0.596 |
| dev_r0 | relsg_gt_uncapped | 0 | 0 | 0 | 1 | 19 | 0.845 | 0.754 |

### Relative-stable oracle (`robovis_eval.relative_stable_oracle`, ratio 0.9)

| run | q | n | defined | unstable_tail | zero_recovery | mean t | median t | t = 20 | t >= 18 |
|---|---|---|---|---|---|---|---|---|---|
| test_r0 | relsg_gt | 100 | 36 | 64 | 0 | 18.61 | 20.0 | 19 | 32 |
| test_r0 | relsg_gt_uncapped | 100 | 44 | 56 | 0 | 18.25 | 19.0 | 17 | 35 |
| test_r0 | R50 | 100 | 37 | 35 | 28 | 16.38 | 18.0 | 15 | 20 |
| dev_r0 | relsg_gt | 20 | 14 | 6 | 0 | 18.43 | 20.0 | 9 | 12 |
| dev_r0 | relsg_gt_uncapped | 20 | 11 | 9 | 0 | 17.82 | 19.0 | 4 | 7 |
| dev_r0 | R50 | 20 | 6 | 7 | 7 | 16.83 | 18.5 | 2 | 4 |

### Reading

- **No zero recovery.** relsg_gt removes zero_recovery entirely. Node recall alone is
  already 0.24 at t = 1, so q never stays at 0. The defined count is 36/100, about the
  same as R@50's 37/100. The 28 zero_recovery images move almost entirely into
  unstable_tail, which grows from 35 to 64.
- **The capped score keeps rising.** Test relsg_gt rises from 0.14 at t = 1 to 0.40 at
  t = 10, 0.47 at t = 15 and 0.49 at t = 20, and is still increasing at t = 20.
  Defined images therefore saturate very late: median t = 20, and 19/36 are defined only
  at the last level. That is the degenerate case of the all-suffix rule, where the
  oracle simply equals the final level.
- **The uncapped score is inflated by the dense output.** relsg_gt_uncapped is 0.36 at
  t = 1, where R@50 is 0.004. PGSG emits about 100 entities and about 300 relations even
  at the lowest level. The caps matter for this reason.
- **Hallucination is high.** Triplet precision of the capped top-50 set is low at every
  level: 0.006 at t = 1 and 0.044 at t = 20. Node precision of the top-20 entity set
  rises from 0.23 to 0.54. Most top-ranked triplets are not in the sparse VG GT, even at
  full resolution.
- **dev20.** dev20 has more defined images under relsg_gt (14/20 vs 6/20 for R@50).
  9 of those 14 are defined only at t = 20, so the extra defined images are mostly
  final-level definitions. The test100 numbers are the more reliable ones.


## 6. Cleaned observed graph G and precision-aware metrics

**What G is.** G is built by `milestone1.graph_clean.clean_graph` (`scripts/35_robovis_clean_graphs.py`)
and stored in `graphs_clean/`. Same-label entities with base-box IoU > 0.5 are merged,
greedily by score. Relations are remapped, self-loops are dropped, and duplicate
(sub, pred, obj) triplets are dropped, keeping the max trp_score. Then the top 20
relations by trp_score are kept, with the entities they reference. On test, a raw state
averages 115 entities and 337 relations. After merging it has 74 entities and 326
relations, and G keeps about 10 entities and 20 relations.

**Metrics.** These are computed in `robovis_sgdet`, and all matches are one-to-one.
- **det_recall20 / det_prec20.** The top 20 entities by score are matched to GT
  objects, greedily in score order. A match needs the same label and IoU ≥ 0.5 (+1
  convention), and each GT object is counted once.
- **trip_prec@K.** Under the graph constraint, each prediction in rank order may claim
  one unclaimed GT relation. Duplicates therefore count as false positives.
- **F1_20.** F1 uses that precision and the one-to-one recall at K = 20.
- **Clean variants.** The `_clean` columns apply the same metrics to G. Since G has at
  most 20 relations, R50_clean = R20_clean.

### test100 per level (mean, raw / clean)

| t | R50 raw / clean | det_recall20 raw / clean | det_prec20 raw / clean | trip_prec20 raw / clean | F1_20 raw / clean |
|---|---|---|---|---|---|
| 1 | 0.004 / 0.004 | 0.048 / 0.037 | 0.035 / 0.066 | 0.001 / 0.001 | 0.001 / 0.001 |
| 2 | 0.015 / 0.009 | 0.065 / 0.054 | 0.046 / 0.087 | 0.004 / 0.003 | 0.005 / 0.004 |
| 3 | 0.014 / 0.014 | 0.087 / 0.063 | 0.060 / 0.099 | 0.005 / 0.005 | 0.007 / 0.007 |
| 4 | 0.020 / 0.013 | 0.104 / 0.086 | 0.070 / 0.130 | 0.005 / 0.005 | 0.006 / 0.006 |
| 5 | 0.038 / 0.023 | 0.129 / 0.109 | 0.089 / 0.169 | 0.009 / 0.009 | 0.013 / 0.012 |
| 6 | 0.038 / 0.028 | 0.148 / 0.119 | 0.101 / 0.191 | 0.011 / 0.012 | 0.015 / 0.016 |
| 7 | 0.049 / 0.038 | 0.183 / 0.147 | 0.124 / 0.225 | 0.014 / 0.014 | 0.019 / 0.019 |
| 8 | 0.064 / 0.050 | 0.198 / 0.168 | 0.138 / 0.248 | 0.019 / 0.018 | 0.026 / 0.025 |
| 9 | 0.069 / 0.056 | 0.218 / 0.189 | 0.152 / 0.275 | 0.020 / 0.021 | 0.028 / 0.029 |
| 10 | 0.091 / 0.073 | 0.227 / 0.197 | 0.158 / 0.298 | 0.026 / 0.025 | 0.036 / 0.036 |
| 11 | 0.108 / 0.079 | 0.244 / 0.215 | 0.168 / 0.311 | 0.029 / 0.029 | 0.040 / 0.041 |
| 12 | 0.112 / 0.086 | 0.242 / 0.216 | 0.166 / 0.319 | 0.030 / 0.030 | 0.042 / 0.042 |
| 13 | 0.111 / 0.088 | 0.259 / 0.230 | 0.179 / 0.340 | 0.032 / 0.031 | 0.044 / 0.043 |
| 14 | 0.109 / 0.084 | 0.260 / 0.229 | 0.177 / 0.334 | 0.031 / 0.030 | 0.043 / 0.042 |
| 15 | 0.123 / 0.098 | 0.261 / 0.232 | 0.178 / 0.349 | 0.033 / 0.034 | 0.047 / 0.048 |
| 16 | 0.125 / 0.101 | 0.277 / 0.247 | 0.189 / 0.344 | 0.035 / 0.036 | 0.049 / 0.051 |
| 17 | 0.145 / 0.117 | 0.275 / 0.253 | 0.187 / 0.358 | 0.043 / 0.041 | 0.060 / 0.058 |
| 18 | 0.132 / 0.110 | 0.285 / 0.245 | 0.195 / 0.350 | 0.035 / 0.037 | 0.050 / 0.053 |
| 19 | 0.142 / 0.108 | 0.291 / 0.253 | 0.195 / 0.353 | 0.037 / 0.036 | 0.053 / 0.052 |
| 20 | 0.140 / 0.115 | 0.291 / 0.253 | 0.198 / 0.360 | 0.037 / 0.039 | 0.053 / 0.055 |

### Relative-stable oracle (ratio 0.9) on these q

| run | q | defined | unstable_tail | zero_recovery | mean t | median t | defined only at t = 20 |
|---|---|---|---|---|---|---|---|
| test_r0 | det_recall20_clean | 40 | 60 | 0 | 17.5 | 19.0 | 18 |
| test_r0 | F1_20_clean | 34 | 29 | 37 | 15.71 | 17.0 | 10 |
| test_r0 | R20_clean | 34 | 29 | 37 | 15.53 | 16.5 | 10 |
| test_r0 | det_recall20 | 44 | 55 | 1 | 18.11 | 20.0 | 25 |
| test_r0 | F1_20 | 33 | 30 | 37 | 15.67 | 17.0 | 10 |
| test_r0 | R50 | 37 | 35 | 28 | 16.38 | 18.0 | 15 |
| dev_r0 | det_recall20_clean | 9 | 11 | 0 | 16.44 | 19.0 | 3 |
| dev_r0 | F1_20_clean | 9 | 4 | 7 | 16.11 | 18.0 | 1 |
| dev_r0 | R20_clean | 8 | 5 | 7 | 15.75 | 17.5 | 1 |
| dev_r0 | det_recall20 | 11 | 9 | 0 | 18.27 | 19.0 | 5 |
| dev_r0 | F1_20 | 9 | 4 | 7 | 15.89 | 17.0 | 1 |
| dev_r0 | R50 | 6 | 7 | 7 | 16.83 | 18.5 | 2 |

### Reading

- **Recall.** Cleaning costs some R@50: 0.140 → 0.115 at t = 20, because G keeps only
  20 relations. R@20 improves slightly (0.111 → 0.115), because deduplication frees
  slots.
- **Entities.** G keeps about 10 entities, so det_recall20 drops (0.291 → 0.253 at
  t = 20). Detection precision almost doubles (0.198 → 0.360). G is a much less
  hallucinated observation.
- **Triplet precision.** Triplet precision stays very low in both forms (0.037 / 0.039
  at t = 20). Most of PGSG's confident triplets are not in the sparse VG GT, even with
  correct boxes and labels.
- **det_recall20_clean as q.** There is no zero recovery, but defined is only 40/100.
  It saturates late (median t = 19, 18 defined only at t = 20). The q curve is smooth
  but keeps rising.
- **F1_20_clean as q.** The statuses are 34 defined, 29 unstable_tail and 37
  zero_recovery. They are identical to R20_clean, because at K = 20 F1 is zero exactly
  when no GT relation is recovered. Its saturation is earlier (median t = 17), like
  R@50's (37 defined, median 18).
