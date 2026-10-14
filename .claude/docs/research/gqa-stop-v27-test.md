# gqa_stop v27 test (plan §27): mixed answerable / unanswerable stopping, robot-approach framing

Status: **test evaluated once, 2026-10-15 (JST), by the team lead at the dev θ*.** The evaluation record is `eval/test/summary.json`, stamped 2026-10-14T15:16:24 UTC. The pre-registration is plan §27. The θ* per method was frozen in `eval/dev/theta_star.json` (2026-10-13T11:36:46 UTC) before any test call. Companion dev report: `gqa-stop-v27-dev.md`.

Run: `configs/gqa_stop_v27.json`, output `outputs/gqa_stop/visapp_gqa_v27/`, data `data/processed/gqa_stop/visapp_gqa_v27/`. Model qwen3.5:9b (digest 6488c96fa5fa), A13 clean decoding, think=false, top_logprobs 10. The prompts answer_only, planner_v3 and observer_v3 are unchanged from dev.

All paths below are relative to `outputs/gqa_stop/visapp_gqa_v27/` unless they start with `.claude/`.

## 0. Terms (as in plan §27 and the dev report)

- **Level.** Image resolution step 1 (smallest) to 20 (base, long side 512). It stands in for a robot approaching the object.
- **reader.** reader_lp: the answer_only prompt with top_logprobs. It answers the question at each level. Its answer is the one that is scored.
- **P_ans.** The probability of the first generated token of the JSON answer value, over the full vocabulary.
- **answerable / unanswerable.** These are internal labels from screening. A question is answerable if the reader is correct at L20, and unanswerable otherwise. The labels are used only for scoring and are never shown to a method.
- **Stop rule (all methods).** A method stops at the first level whose score is ≥ θ. If no level reaches θ, the method **never stops**, which means "this question cannot be answered".
- **strict scoring (primary).** On an answerable question, a method is correct if it stops and the reader is right at that level. Never stopping is wrong. On an unanswerable question, a method is correct only if it never stops.
- **unified scoring (sensitivity).** Same as strict, except that a stop on an unanswerable question also counts as correct when the reader happens to be right at that level.
- **answerable acc.** The fraction of answerable questions scored correct.
- **unanswerable correct.** The fraction of unanswerable questions scored correct. Under strict scoring, this equals the never-stop rate on that side.
- **balanced accuracy.** The mean of answerable acc and unanswerable correct. "Never stop" and "always stop at L20" both score 0.500.
- **pooled acc.** Accuracy over all questions, which depends on the answerable / unanswerable mix.
- **mean stop level.** The mean of the stop level, with never stopping counted as 20.
- **θ\*.** For each method, the θ on the grid 0.00, 0.05, …, 1.00 that maximises dev balanced accuracy (strict). Ties go to the smaller θ.
- **Methods.**
  - **B1.** The reader's own P_ans.
  - **P_min / P_mean.** Proposal P: planner_v3 splits the question into items, and observer_v3_lp answers each item at each level. An item scores its P_ans if it is resolved, its answer is not "unknown", and all its depends_on items pass the same test. Otherwise it scores 0. P_min takes the minimum over items and P_mean the mean.
  - **A1.** direct_resolved: the observer prompt applied to the original question, gated by resolved / not unknown.
  - **A2.** The minimum raw item P_ans, with no gate.
  - **A3.** Stop when all items are resolved. A3 has no θ.
- **Image-cluster bootstrap.** Resampling of images (2000 draws, seed 20261008), with θ held fixed. "Paired" means both methods are scored on the same resample. The CIs do not include θ-selection error.

## 1. Bottom line (confirmatory, pre-registered)

The confirmatory number is each method's test balanced accuracy (strict) at its dev θ*. The primary comparisons are P_min vs B1 and P_min vs A1 (§27.7).

| comparison at θ* | strict balanced acc, difference [95% CI] | unified balanced acc, difference [95% CI] |
|---|---|---|
| P_min − B1 | **+0.036 [+0.002, +0.067]**, CI excludes 0 | +0.030 [−0.004, +0.061], CI includes 0 |
| P_min − A1 | +0.020 [−0.001, +0.042], CI includes 0 | +0.011 [−0.009, +0.031], CI includes 0 |

- **P_min beats B1 on strict balanced accuracy, but only narrowly.** The lower CI bound is +0.002. Under unified scoring the difference is no longer separated from 0. §27 did not pre-register a correction for the two primary comparisons; none was applied.
- **P_min vs A1 is not separated from 0.** At the same θ, the two methods are indistinguishable at every θ from 0.50 to 0.95 (§5.2). The +0.020 at θ* comes from the θ* choices: A1's dev θ* (0.75) was a tie-break, and 0.80 does better on test (§9).
- **The gain over B1 is a trade, not a dominance.** P_min − B1 at θ*:

| quantity | P_min − B1 [95% CI] |
|---|---|
| answerable acc | −0.064 [−0.108, −0.024] |
| unanswerable correct | +0.136 [+0.093, +0.181] |
| mean stop level, answerable | +4.39 [+3.81, +4.99] levels |

  P_min abstains on 13.6 points more unanswerable questions. It pays with 6.4 points of answerable accuracy and stops 4.4 levels later on answerable questions.
- **Absolute levels are low.** The best method at θ* reaches 0.597, against the 0.500 floor of "never stop". Every thresholded method lost 0.02–0.06 from dev to test.

## 2. Run facts (test)

- **Environment.** Inference started on 2026-10-13. Env check passed: only qwen3.5:9b was resident, 100% GPU, and the ollama log showed "CLIP using CUDA0 backend". The determinism gate passed: 120 calls, 811 positions, 0 violations, and both repeat pairs were byte-identical. Source: `worklog.md`.
- **Overnight break and resume.** Inference paused after 147 questions on 2026-10-13 and resumed on 2026-10-14. Before resuming, the env check and the determinism gate were repeated and both passed. The resume check re-sent 39 cached requests without the cache, and all 39 were byte-identical in raw reply and logprobs. Source: `worklog.md`.
- **Calls and time.** There were 79 foreground chunks of ≤ 585 s, with 71,681 new calls taking 12.17 h of chunk time. Screening added 886 calls. The rate was 0.61 s per call.
- **Records** (`logs/audit_test.json`). There are 72,566 records in total.
  - reader_lp: 17,720.
  - observer_v3_lp: 36,240.
  - direct_resolved (observer_v3_lp_direct): 17,720.
  - plans_v3: 886.
  - All are valid. In addition, 0 had non-generation time > 5 s (max 0.46 s), and there were 0 argmax violations in 905,068 positions. Nothing was flagged or quarantined.
- **Planner.** All 886 plans are valid: 885 on the first attempt and 1 after one schema repair. 846 plans have 2 items and 40 have 3, a mean of 2.045 items. In 883 of 886 plans the last item names the attribute. Source: `level_table_test.csv`.
- **Level table.** There are 17,720 rows, which is 886 questions × 20 levels.

## 3. Dataset and labels (`dataset_v27.json`, `level_table_test.csv`)

- **Test set.** Test is every eligible GQA testdev_balanced question: 886 questions on 327 images, with no per-image cap. The eligible questions are single-object attribute queries about color, material or shape. Test shares no image with dev.
- **Labels from screening.** The reader_lp L20 call gave 0 invalid replies.

| attribute | answerable | unanswerable | total |
|---|---|---|---|
| color | 270 | 308 | 578 |
| material | 154 | 80 | 234 |
| shape | 43 | 31 | 74 |
| **total** | **467** | **419** | **886** |

  On test, 52.7% of questions are answerable, against 58.3% on dev. Material is a much larger share on test (234 of 886) than on dev (44 of 300).
- **Reader accuracy by level.**

| level | 1 | 5 | 10 | 15 | 19 | 20 |
|---|---|---|---|---|---|---|
| answerable | 0.591 | 0.709 | 0.857 | 0.921 | 0.968 | 1.000 |
| unanswerable | 0.086 | 0.103 | 0.105 | 0.081 | 0.048 | 0.000 |

- **The labels are not clean, as on dev.**
  - 24% of unanswerable questions (102 of 419) are reader-correct at some level before L20 (dev: 22%).
  - 49% of answerable questions (231 of 467) are correct at every level (dev: 43%).

## 4. Results at θ* (strict; `eval/test/at_theta_star.csv`, `method_ci_at_theta_star.csv`)

| method | θ* (dev) | balanced acc [95% CI] | answerable acc [95% CI] | unanswerable correct [95% CI] | pooled acc | mean L (all) | mean L (answerable) |
|---|---|---|---|---|---|---|---|
| B1 reader P_ans | 0.90 | 0.561 [0.528, 0.595] | 0.597 [0.553, 0.643] | 0.525 [0.474, 0.581] | 0.563 | 11.34 | 10.08 |
| **P pans_min** | **0.80** | **0.597 [0.565, 0.629]** | 0.533 [0.483, 0.580] | 0.661 [0.615, 0.707] | 0.594 | 15.42 | 14.48 |
| P pans_mean | 0.90 | 0.581 [0.548, 0.612] | 0.469 [0.420, 0.517] | 0.692 [0.647, 0.737] | 0.574 | 16.06 | 15.36 |
| A1 direct_resolved | 0.75 | 0.577 [0.542, 0.609] | 0.617 [0.570, 0.660] | 0.537 [0.485, 0.590] | 0.579 | 13.89 | 12.93 |
| A2 raw item P_ans, no gate | 0.95 | 0.523 [0.495, 0.552] | 0.300 [0.260, 0.342] | 0.747 [0.705, 0.787] | 0.511 | 16.67 | 16.37 |
| A3 all resolved, no θ | — | 0.467 [0.439, 0.494] | 0.756 [0.713, 0.793] | 0.179 [0.138, 0.220] | 0.483 | 9.87 | 9.44 |

The pooled accuracy is over all 886 questions (467 answerable / 419 unanswerable).

- **Stopping detail at θ*.** "Stopped wrong" counts answerable questions that stopped at a level where the reader was wrong. The quartiles are of the answerable stop level, with never stopping counted as 20.

| method | answerable never-stop | answerable stopped wrong | answerable stop-level Q1 / median / Q3 |
|---|---|---|---|
| B1 | 0.340 | 29 | 1 / 8 / 20 |
| P_min | 0.450 | 8 | 9 / 18 / 20 |
| P_mean | 0.518 | 6 | 10 / 20 / 20 |
| A1 | 0.349 | 16 | 7 / 14 / 20 |
| A2 | 0.670 | 14 | 14 / 20 / 20 |
| A3 | 0.116 | 60 | 4 / 8 / 14 |

  P_min's answerable loss against B1 comes from never stopping: its never-stop rate is 10.9 points higher. It is partly offset by fewer wrong stops (8 vs 29).
- **Unified scoring (sensitivity).**

| method | balanced (unified) [95% CI] | unanswerable correct (unified) |
|---|---|---|
| B1 | 0.572 [0.539, 0.606] | 0.547 |
| P_min | 0.602 [0.570, 0.633] | 0.671 |
| P_mean | 0.584 [0.552, 0.615] | 0.699 |
| A1 | 0.591 [0.557, 0.623] | 0.566 |
| A2 | 0.532 [0.504, 0.560] | 0.764 |
| A3 | 0.509 [0.479, 0.538] | 0.263 |

  Unified scoring narrows P_min's lead. B1 and A1 stop early on more unanswerable questions, and some of those stops are lucky-correct (B1 +0.021, A1 +0.029 on the unanswerable side, against P_min +0.010).

## 5. Paired comparisons (image-cluster paired bootstrap, 2000 draws, seed 20261008)

### 5.1 Each method at its own θ* (the confirmatory comparison; `eval/test/paired_bootstrap_at_theta_star.csv`)

P_min is at θ 0.80, B1 at 0.90 and A1 at 0.75.

| quantity | P_min − B1 | P_min − A1 |
|---|---|---|
| balanced acc (strict) | **+0.036 [+0.002, +0.067]** | +0.020 [−0.001, +0.042] |
| balanced acc (unified) | +0.030 [−0.004, +0.061] | +0.011 [−0.009, +0.031] |
| answerable acc | −0.064 [−0.108, −0.024] | −0.084 [−0.114, −0.054] |
| unanswerable correct | +0.136 [+0.093, +0.181] | +0.124 [+0.095, +0.154] |
| pooled acc (strict) | +0.030 [−0.003, +0.061] | +0.015 [−0.006, +0.036] |
| mean level, all | +4.08 [+3.60, +4.58] | +1.53 [+1.28, +1.80] |
| mean level, answerable | +4.39 [+3.81, +4.99] | +1.55 [+1.22, +1.93] |

### 5.2 Same θ for both methods (descriptive; `eval/test/paired_bootstrap.csv`)

Balanced accuracy (strict), P_min − method:

| θ | 0.50 | 0.60 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|---|
| − B1 | +0.191 [+0.155, +0.224] | +0.174 [+0.138, +0.208] | +0.146 [+0.108, +0.185] | +0.107 [+0.067, +0.144] | +0.102 [+0.064, +0.135] | +0.042 [+0.008, +0.074] | +0.027 [−0.005, +0.060] | −0.020 [−0.050, +0.009] |
| − A1 | +0.006 [−0.013, +0.024] | +0.011 [−0.007, +0.030] | +0.002 [−0.016, +0.020] | −0.001 [−0.020, +0.019] | +0.005 [−0.013, +0.023] | −0.007 [−0.025, +0.011] | −0.010 [−0.026, +0.005] | −0.010 [−0.025, +0.006] |

- **The same-θ gap to B1 is a scale effect, as on dev.** The reader's P_ans is high from L1: its mean at L1 is 0.715 on answerable and 0.665 on unanswerable questions. So at moderate θ, B1 stops almost everything early.
- **P_min and A1 coincide at every θ.** Every CI from 0.50 to 0.95 includes 0, and the point estimates are within ±0.011. This replicates dev: decomposition adds nothing beyond the resolved / unknown gate applied to the original question.

## 6. θ sweep (descriptive; `eval/test/sweep.csv`, figures in `figures/test/`)

The test θ curves are descriptive only. Per §27.7, no θ is re-read from them.

**Balanced accuracy (strict)** (A3, which has no θ: 0.467).

| θ | B1 | P_min | P_mean | A1 | A2 |
|---|---|---|---|---|---|
| 0.00 | 0.296 | 0.296 | 0.296 | 0.296 | 0.296 |
| 0.05 | 0.296 | 0.467 | 0.346 | 0.459 | 0.296 |
| 0.10 | 0.296 | 0.467 | 0.346 | 0.459 | 0.296 |
| 0.15 | 0.296 | 0.467 | 0.345 | 0.459 | 0.296 |
| 0.20 | 0.296 | 0.467 | 0.349 | 0.458 | 0.296 |
| 0.25 | 0.296 | 0.467 | 0.360 | 0.458 | 0.296 |
| 0.30 | 0.297 | 0.470 | 0.366 | 0.458 | 0.296 |
| 0.35 | 0.303 | 0.473 | 0.387 | 0.470 | 0.296 |
| 0.40 | 0.312 | 0.492 | 0.402 | 0.487 | 0.296 |
| 0.45 | 0.318 | 0.509 | 0.421 | 0.501 | 0.297 |
| 0.50 | 0.332 | 0.522 | 0.468 | 0.517 | 0.304 |
| 0.55 | 0.348 | 0.539 | 0.474 | 0.534 | 0.310 |
| 0.60 | 0.376 | 0.550 | 0.479 | 0.538 | 0.331 |
| 0.65 | 0.411 | 0.555 | 0.489 | 0.556 | 0.352 |
| 0.70 | 0.426 | 0.572 | 0.511 | 0.571 | 0.370 |
| 0.75 | 0.469 | 0.576 | 0.527 | *0.577* | 0.399 |
| 0.80 | 0.496 | ***0.597*** | 0.561 | 0.592 | 0.436 |
| 0.85 | 0.543 | 0.585 | 0.571 | 0.592 | 0.463 |
| 0.90 | *0.561* | 0.588 | ***0.581*** | **0.598** | 0.518 |
| 0.95 | **0.576** | 0.555 | 0.576 | 0.566 | ***0.523*** |
| 1.00 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |

(Italic: dev θ*. Bold: test argmax. Bold italic: both.)

**Answerable acc (strict)** (A3: 0.756).

| θ | B1 | P_min | P_mean | A1 | A2 |
|---|---|---|---|---|---|
| 0.00 | 0.591 | 0.591 | 0.591 | 0.591 | 0.591 |
| 0.05–0.25 | 0.591 | 0.756 | 0.679–0.702 | 0.749–0.752 | 0.591 |
| 0.30 | 0.591 | 0.758 | 0.713 | 0.749 | 0.591 |
| 0.40 | 0.621 | 0.764 | 0.749 | 0.764 | 0.591 |
| 0.50 | 0.645 | 0.754 | 0.754 | 0.769 | 0.604 |
| 0.60 | 0.692 | 0.698 | 0.754 | 0.724 | 0.636 |
| 0.70 | 0.711 | 0.612 | 0.728 | 0.660 | 0.657 |
| 0.75 | 0.722 | 0.563 | 0.702 | 0.617 | 0.645 |
| 0.80 | 0.702 | 0.533 | 0.645 | 0.578 | 0.636 |
| 0.85 | 0.664 | 0.445 | 0.567 | 0.512 | 0.561 |
| 0.90 | 0.597 | 0.377 | 0.469 | 0.430 | 0.482 |
| 0.95 | 0.473 | 0.240 | 0.338 | 0.289 | 0.300 |
| 1.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**Unanswerable correct (strict)** (A3: 0.179).

| θ | B1 | P_min | P_mean | A1 | A2 |
|---|---|---|---|---|---|
| 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.05–0.25 | 0.000 | 0.179 | 0.012–0.017 | 0.167 | 0.000 |
| 0.30 | 0.002 | 0.181 | 0.019 | 0.167 | 0.000 |
| 0.40 | 0.002 | 0.220 | 0.055 | 0.210 | 0.000 |
| 0.50 | 0.019 | 0.291 | 0.181 | 0.265 | 0.005 |
| 0.60 | 0.060 | 0.401 | 0.205 | 0.353 | 0.026 |
| 0.70 | 0.141 | 0.532 | 0.294 | 0.482 | 0.084 |
| 0.75 | 0.217 | 0.589 | 0.351 | 0.537 | 0.153 |
| 0.80 | 0.289 | 0.661 | 0.477 | 0.606 | 0.236 |
| 0.85 | 0.422 | 0.726 | 0.575 | 0.673 | 0.365 |
| 0.90 | 0.525 | 0.800 | 0.692 | 0.766 | 0.554 |
| 0.95 | 0.678 | 0.871 | 0.814 | 0.842 | 0.747 |
| 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**Mean stop level on answerable questions** (never stopping counted as 20; A3: 9.44).

| θ | B1 | P_min | P_mean | A1 | A2 |
|---|---|---|---|---|---|
| 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.05–0.25 | 1.00 | 9.44 | 4.08–4.65 | 8.95 | 1.00 |
| 0.30 | 1.03 | 9.47 | 5.28 | 8.99 | 1.00 |
| 0.40 | 1.25 | 9.68 | 6.59 | 9.21 | 1.01 |
| 0.50 | 1.85 | 10.54 | 9.54 | 9.97 | 1.29 |
| 0.60 | 2.99 | 11.85 | 9.95 | 11.03 | 2.18 |
| 0.70 | 4.52 | 13.19 | 10.90 | 12.29 | 4.32 |
| 0.75 | 5.47 | 13.93 | 11.73 | 12.93 | 5.80 |
| 0.80 | 6.65 | 14.48 | 12.90 | 13.57 | 7.73 |
| 0.85 | 7.98 | 15.48 | 14.07 | 14.58 | 10.21 |
| 0.90 | 10.08 | 16.49 | 15.36 | 15.57 | 12.78 |
| 0.95 | 12.77 | 17.85 | 17.07 | 17.08 | 16.37 |
| 1.00 | 20.00 | 20.00 | 20.00 | 20.00 | 20.00 |

**Never-stop rate on answerable questions** (A3: 0.116).

| θ | B1 | P_min | P_mean | A1 | A2 |
|---|---|---|---|---|---|
| 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.05–0.25 | 0.000 | 0.116 | 0.009–0.013 | 0.103–0.105 | 0.000 |
| 0.30 | 0.000 | 0.116 | 0.019 | 0.105 | 0.000 |
| 0.40 | 0.000 | 0.124 | 0.036 | 0.116 | 0.000 |
| 0.50 | 0.004 | 0.154 | 0.118 | 0.139 | 0.000 |
| 0.60 | 0.030 | 0.240 | 0.126 | 0.208 | 0.006 |
| 0.70 | 0.084 | 0.349 | 0.178 | 0.298 | 0.047 |
| 0.75 | 0.116 | 0.411 | 0.218 | 0.349 | 0.099 |
| 0.80 | 0.167 | 0.450 | 0.306 | 0.396 | 0.161 |
| 0.85 | 0.231 | 0.542 | 0.403 | 0.475 | 0.287 |
| 0.90 | 0.340 | 0.619 | 0.518 | 0.565 | 0.422 |
| 0.95 | 0.497 | 0.758 | 0.660 | 0.709 | 0.670 |
| 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The θ grid rows 0.05–0.25 are collapsed where the values are constant or nearly so. Every 0.05 step is in `sweep.csv`, and the per-attribute grid is in `sweep_by_attribute.csv`.

- **θ = 0.00 is the degenerate stop-at-L1 row.** Its answerable accuracy is the L1 reader accuracy (0.591).
- **For P_min, θ 0.05–0.25 equals A3, as on dev.**

**Best over θ on test (oracle θ chosen on test; descriptive only, NOT confirmatory).**

| method | test-argmax θ | best test balanced (strict) | at dev θ* | loss from transferring dev θ* |
|---|---|---|---|---|
| B1 | 0.95 | 0.576 | 0.561 | 0.014 |
| P_min | 0.80 | 0.597 | 0.597 | 0 |
| P_mean | 0.90 | 0.581 | 0.581 | 0 |
| A1 | 0.90 | 0.598 | 0.577 | 0.021 |
| A2 | 0.95 | 0.523 | 0.523 | 0 |

With an oracle θ, A1 (0.598) and P_min (0.597) are equal, and B1 (0.576) is about 0.02 below them.

## 7. Firing on unanswerable questions at θ* (419 questions; counts by stop level)

| method | L1–5 | L6–10 | L11–15 | L16–19 | L20 | never |
|---|---|---|---|---|---|---|
| B1 | 135 | 33 | 21 | 8 | 2 | 220 |
| P_min | 43 | 37 | 37 | 16 | 9 | 277 |
| P_mean | 34 | 39 | 30 | 17 | 9 | 290 |
| A1 | 62 | 56 | 46 | 17 | 13 | 225 |
| A2 | 47 | 24 | 14 | 13 | 8 | 313 |
| A3 | 120 | 105 | 80 | 30 | 9 | 75 |

- **B1 again fires very early.** 135 of its 199 firings are at L1–5.
- **P_min fires less, and later.** Its firings are spread over L1–19, with 43 at L1–5.

## 8. Per attribute at θ* (`eval/test/at_theta_star_by_attribute.csv`)

| attribute | method | n ans / unans | answerable acc | unanswerable correct | balanced (strict) | mean L (answerable) |
|---|---|---|---|---|---|---|
| color | B1 | 270 / 308 | 0.556 | 0.542 | 0.549 | 10.95 |
| color | P_min | 270 / 308 | 0.581 | 0.640 | 0.611 | 13.66 |
| color | A1 | 270 / 308 | 0.659 | 0.506 | 0.583 | 11.54 |
| material | B1 | 154 / 80 | 0.610 | 0.512 | 0.561 | 9.42 |
| material | P_min | 154 / 80 | 0.396 | 0.800 | 0.598 | 16.31 |
| material | A1 | 154 / 80 | 0.494 | 0.725 | 0.609 | 15.68 |
| shape | B1 | 43 / 31 | 0.814 | 0.387 | 0.601 | 7.00 |
| shape | P_min | 43 / 31 | 0.721 | 0.516 | 0.619 | 13.07 |
| shape | A1 | 43 / 31 | 0.791 | 0.355 | 0.573 | 11.81 |

P_mean, A2 and A3 are in the CSV.

- **Color drives the P_min − B1 gain.** The balanced difference is +0.062 on color, +0.037 on material and +0.018 on shape. Color is 65% of the test set.
- **On material, P_min and A1 abstain heavily.** P_min never stops on 59% of answerable material questions, against 33% for B1. That gives high unanswerable correct (0.80) but low answerable acc (0.40).
- **Shape (n = 74: 43 / 31) is too small to rank methods.** No per-attribute bootstrap was run. A rough unpaired binomial approximation gives about ±0.11 at 95% for a single method's balanced accuracy at these counts. The spread among B1, P_min and A1 on shape is 0.05. Shape supports no claim either way.

## 9. Comparison with dev

| method | dev θ* | test-argmax θ | dev balanced at θ* | test balanced at θ* | change |
|---|---|---|---|---|---|
| B1 | 0.90 | 0.95 | 0.614 | 0.561 | −0.053 |
| P_min | 0.80 | 0.80 | 0.630 | 0.597 | −0.033 |
| P_mean | 0.90 | 0.90 | 0.618 | 0.581 | −0.037 |
| A1 | 0.75 | 0.90 | 0.633 | 0.577 | −0.056 |
| A2 | 0.95 | 0.95 | 0.540 | 0.523 | −0.017 |
| A3 | — | — | 0.427 | 0.467 | +0.040 |

Dev values are from `eval/dev/sweep.csv` and the dev report. Test values are from `eval/test/sweep.csv`.

- **The peak stayed at the dev θ* for P_min, P_mean and A2.** It moved up one grid step for B1 (0.90 to 0.95) and three steps for A1 (0.75 to 0.90).
- **A1's dev θ* was a tie.** On dev, A1 scored 0.63314 at both θ 0.75 and θ 0.80, and the pre-registered tie rule picked 0.75. On test, θ 0.80 gives 0.592, which is 0.015 higher than at 0.75. Most of the P_min − A1 point estimate at θ* (+0.020) is this tie-break.
- **The drop from dev was predicted.** The dev report expected a v22-like drop of 0.01–0.05. The actual drops are 0.017–0.056. P_min dropped least among the main methods. One reason may be that its dev peak sits at the same θ on test.
- **The dev direction held for P_min vs B1, and the size grew.** On dev the difference was +0.015 [−0.035, +0.068]. On test it is +0.036 [+0.002, +0.067], with a tighter CI on 3× the questions. For P_min vs A1, dev gave −0.003 and test gives +0.020; neither is separated from 0.
- **Overlap of the signals at L20 is similar to dev.**

| mean at L20 (test) | answerable | unanswerable |
|---|---|---|
| reader P_ans | 0.820 | 0.683 |
| pans_min | 0.668 | 0.518 |
| direct_resolved score | 0.697 | 0.550 |

## 10. Cost (latency per call from `requests.jsonl`, test questions, initial calls)

| role | calls | mean s per call |
|---|---|---|
| reader_lp | 17,720 | 0.486 |
| planner_v3 (text only, once per question) | 886 | 0.716 |
| observer_v3_lp (per item) | 36,240 | 0.641 |
| observer_v3_lp_direct (A1) | 17,720 | 0.644 |

These per-role means are from test questions only. They differ by at most 0.004 s from the figures the team lead supplied (0.483 / 0.712 / 0.641 / 0.644), which may have used a different record subset.

Expected latency per question at θ* is one-off cost + per-level cost × levels visited. Levels visited is the stop level, or 20 if the method never stops. The answer is scored from the reader at the stop level. So every method except B1 adds one reader call (0.486 s) when it stops.

| method | per-level calls | per-level s | one-off s | mean levels visited | stop fraction | expected s per question | if all 20 levels |
|---|---|---|---|---|---|---|---|
| B1 | 1 reader_lp | 0.486 | 0 | 11.34 | 0.572 | 5.51 | 9.72 |
| P_min | 2.045 observer items | 1.311 | 0.716 planner + reader at stop | 15.42 | 0.450 | 21.19 | 26.93 |
| P_mean | as P_min | 1.311 | as P_min | 16.06 | 0.400 | 21.97 | 26.93 |
| A1 | 1 direct_resolved | 0.644 | reader at stop | 13.89 | 0.562 | 9.22 | 12.88 |
| A2 | as P_min | 1.311 | as P_min | 16.67 | 0.293 | 22.78 | 26.93 |
| A3 | as P_min | 1.311 | as P_min | 9.87 | 0.854 | 14.09 | 26.93 |

The expected costs were computed by re-applying the stop rule to `level_table_test.csv`. The levels visited match `mean_level_all` in `at_theta_star.csv` for all six methods.

- **P_min costs about 3.8–3.9× B1 and 2.3× A1 per question at θ*.** It is slower per level (two observer calls) and visits more levels.
- **This is sequential, single-GPU latency.** The setting is an RTX 5080 with ollama and no batching. The observer item calls within a level could run in parallel, which is not measured here.

## 11. Observations

1. **The pre-registered P_min − B1 win is real but small and scoring-dependent.** It holds under strict scoring with a lower bound of +0.002. It is lost under unified scoring. The CIs do not include θ-selection error, and the oracle-θ gap between them is about 0.02 (§6).
2. **The gate, not the decomposition, does the work, again.** P_min and A1 are equal at every same-θ comparison and at the oracle θ. A2 (no gate) is the worst thresholded method on test, as on dev.
3. **P_min trades answerable accuracy and speed for abstention.** On answerable questions, it stops 4.4 levels later than B1 and never stops on 45%.
4. **A3 confirms that "resolved" alone does not abstain.** It stops on 82% of unanswerable questions (balanced 0.467). This is better than on dev (0.427). One likely reason is that material, where A3 abstains more (unanswerable correct 0.450), is a larger share of test; this was not tested.
5. **Material behaves differently from color.** The observer-based methods abstain far more on material. Material is 26% of test but only 15% of dev, so the dev θ* was tuned mostly on color.

## 12. Caveats

- **Label noise.** "Unanswerable" means only that the reader was wrong at L20.
  - 24% of these questions are reader-correct at some earlier level.
  - §25 found that two models often give the same wrong answer at L20, which points to GQA annotation ambiguity.
  - Part of the unanswerable side is therefore not detectable by any stopping signal. This caps achievable balanced accuracy below 1.
- **Strict scoring asymmetry.** Strict scoring counts a lucky-correct early stop on an unanswerable question as wrong. This favours late-stopping methods such as P_min. Under unified scoring, P_min − B1 drops from +0.036 to +0.030 and its CI includes 0.
- **The same reader defines the labels and scores the stops.** B1 uses the same call's P_ans, as noted in §27.10. This could favour B1, yet B1 still ranks below P_min.
- **Single model.** Only qwen3.5:9b was used, with one decoding setting and one seed. No other model was run on v27.
- **One-shot test.** Test was evaluated once. θ* was fixed on 300 dev questions, about one third of the test size. There is no second test set to check the +0.036.
- **Ties.**
  - A1's dev θ* came from a tie (0.75 = 0.80 on dev). The pre-registered smaller-θ rule cost A1 0.015 on test (§9).
  - B1, P_min and P_mean had no dev tie at their θ*.
  - The coarse 0.05 grid itself makes the θ* placement noisy.
- **Two primary comparisons, no multiplicity correction.** None was pre-registered. A correction for two comparisons would widen the P_min − B1 interval. Whether its lower bound would stay above 0 was not computed, because the bootstrap draws are not saved.
- **θ-free CIs.** All bootstrap CIs hold θ fixed. They understate uncertainty about what a freshly tuned θ would give.

## 13. Non-claims (per §27.10)

- No target accuracy (for example 0.95) is claimed or used.
- The numbers are not comparable to standard GQA testdev metrics. The set covers single-object attribute queries only, on degraded images, with exact-match scoring that includes stopping.
- Nothing is claimed for verify, compare, logical, count or relation-answer questions.
- "Answerable" is a reader-at-L20 label, not ground-truth answerability.
- The numbers are not comparable to v22 and earlier results. The set (no textprobe, mixed population), the scoring (forced L20 is wrong) and the methods (no Rel) all differ.
- No claim is made that decomposition (planning) is what helps. A1, without a planner, matches P_min at equal θ.

## 14. Files and commands

```
# inference (2026-10-13 .. 2026-10-14), after env check + determinism gate
uv run python -m milestone1.gqa_stop.cli v27-screen   --split test --config configs/gqa_stop_v27.json
timeout 585 uv run python -m milestone1.gqa_stop.cli v27-infer --split test --max-seconds 560 --config configs/gqa_stop_v27.json   # x79
uv run python -m milestone1.gqa_stop.cli v27-audit    --split test --config configs/gqa_stop_v27.json
uv run python -m milestone1.gqa_stop.cli v27-table    --split test --config configs/gqa_stop_v27.json
# test evaluation, run once by the team lead (requires the explicit go flag)
uv run python -m milestone1.gqa_stop.cli v27-evaluate --split test --confirm-test-eval --config configs/gqa_stop_v27.json
```

- **Outputs** (`outputs/gqa_stop/visapp_gqa_v27/`):
  - `level_table_test.csv` (17,720 rows) and `logs/audit_test.json`.
  - `eval/test/`: `sweep.csv`, `sweep_by_attribute.csv`, `at_theta_star.csv`, `at_theta_star_by_attribute.csv`, `method_ci_at_theta_star.csv`, `paired_bootstrap.csv`, `paired_bootstrap_at_theta_star.csv` and `summary.json`.
  - `eval/dev/theta_star.json`: the frozen θ*.
  - `figures/test/`: `balanced_vs_theta.png`, `answerable_acc_vs_level.png` and `unanswerable_vs_theta.png`.
- **Derived numbers in this report.** The following were computed read-only with `uv run python -I` from `level_table_test.csv`, `requests.jsonl` and the eval CSVs. No inference or evaluation was re-run.
  - Label statistics (§3).
  - Signal means at L20 (§9).
  - Per-role latency and expected cost (§10).
  - Test argmax θ (§6, §9).
