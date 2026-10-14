# gqa_stop v27 POST-HOC: observer-answer methods P'_min / P'_mean / A1' (plan §27.12)

Status: **post-hoc analysis, 2026-10-15, requested by the user via the team lead after the one confirmatory test evaluation.** Nothing here was pre-registered. The confirmatory numbers in `gqa-stop-v27-test.md` (plan §27.11) are unchanged: `eval/dev/` and `eval/test/` were not rewritten (sha256 of every file identical before and after). No inference was run. Every number is computed from the cached level tables `level_table_{dev,test}.csv`.

**Test-set reuse.** The test set has already been scored once for the confirmatory comparison. Comparing new methods on it after seeing that result is a second look at the same data. Treat every test number below as descriptive. Its CIs are optimistic because they ignore this selection step and θ-selection error.

All paths are relative to `outputs/gqa_stop/visapp_gqa_v27/` unless they start with `.claude/` or `src/`.

## 0. Definitions

- **Existing methods** (B1, P_min, P_mean, A1, A2, A3). Unchanged. The final answer is the **reader's** answer (reader_lp) at the stop level. Definitions as in `gqa-stop-v27-test.md` §0.
- **New methods.** Same stop rule as the base method. Only the final answer changes.

| method | stops like | final answer at the stop level |
|---|---|---|
| P'_min | P_min (gated pans_min ≥ θ) | the plan's **last** checklist item's observer answer, `item{n_items}_answer` |
| P'_mean | P_mean (gated pans_mean ≥ θ) | same as P'_min |
| A1' | A1 (dres_score ≥ θ) | direct_resolved's answer, `dres_answer` |

- **Correctness.** `is_correct(answer, gt)` from `src/milestone1/gqa_stop/evaluate.py`, the same function as `reader_correct`: NFKC, lowercase, strip, collapse whitespace, then exact match. "unknown" or a missing answer is wrong. Recomputing `reader_correct` this way from the table reproduces the stored column on all rows of both splits.
- **strict scoring.** Answerable: correct iff the method stops and *its final answer* is correct at the stop level. Unanswerable: correct iff the method never stops.
- **unified scoring ("never stop OR correct").** A stop is correct iff the final answer at the stop level is correct. A never-stop is correct iff the question is unanswerable. The difference from strict is only on unanswerable questions: a stop counts when the final answer happens to be right.
- **The answerable / unanswerable label is unchanged.** It is still the reader's L20 correctness (plan §27.3) for all methods.
- **θ\* rules.**
  - **θ\*_strict.** Argmax of dev balanced strict, ties to the smaller θ. This is the pre-registered rule, applied to all nine methods.
  - **θ\*_unified.** Argmax of dev balanced unified, ties to the smaller θ. This is a secondary selection.
- **Other metrics.** Mean stop level, never-stop rate, quartiles, per attribute and the firing histogram have the same definitions. They depend only on the stop rule, so they are identical between X and X'.
- **Bootstrap.** Image-cluster paired bootstrap, 2000 draws, seed 20261008, the same indices as the confirmatory run. Each method is at its own θ\*.

## 1. Bottom line

- **The observer's answer is almost always the reader's answer where the method stops.** Final answers agree on these stopped questions:

| split | P_min θ 0.80 stops | agree | A1 θ 0.75 stops | agree |
|---|---|---|---|---|
| dev | 152 | 97.4% | 191 | 97.4% |
| test | 399 | 99.5% | 498 | 99.8% |

  The few disagreements go both ways (§6). So P' ≈ P and A1' ≈ A1 at every θ\* (test: +0.001 and +0.000 balanced strict).
- **θ\*_unified = θ\*_strict for all nine methods on dev.** The secondary rule changes nothing. Test numbers at "both" θ\* are therefore one set of numbers.
- **Test at θ\* (descriptive), balanced accuracy.**

| method | B1 | P_min | P_mean | A1 | A2 | A3 | P'_min | P'_mean | A1' |
|---|---|---|---|---|---|---|---|---|---|
| θ\* (both rules) | 0.90 | 0.80 | 0.90 | 0.75 | 0.95 | — | 0.80 | 0.90 | 0.75 |
| strict | 0.561 | 0.597 | 0.581 | 0.577 | 0.523 | 0.467 | 0.598 | 0.582 | 0.577 |
| unified | 0.572 | 0.602 | 0.584 | 0.591 | 0.532 | 0.509 | 0.603 | 0.585 | 0.592 |

- **Unified credit on unanswerable questions is small for every thresholded method.** It adds 0.003–0.015 to balanced accuracy. It is largest for A3, which stops early on unanswerable questions (+0.042). The observer is right on only a few reader-wrong questions at L20 (§6).

## 2. θ\* on dev (`eval_posthoc/dev/theta_star.json`)

Both rules give the same θ\*. For the six existing methods, θ\*_strict equals the frozen `eval/dev/theta_star.json` (checked in `eval_posthoc/dev/summary.json`).

| method | B1 | P_min | P_mean | A1 | A2 | A3 | P'_min | P'_mean | A1' |
|---|---|---|---|---|---|---|---|---|---|
| θ\*_strict | 0.90 | 0.80 | 0.90 | 0.75 | 0.95 | — | 0.80 | 0.90 | 0.75 |
| θ\*_unified | 0.90 | 0.80 | 0.90 | 0.75 | 0.95 | — | 0.80 | 0.90 | 0.75 |

Near-ties on dev, as the top two values:
- **A1 / A1' strict.** 0.75 and 0.80 tie exactly (A1 0.6331 at both, A1' 0.6303 at both), and the smaller θ wins. Under unified, 0.75 wins outright (A1 0.6411 vs 0.6371).
- **P'_min unified.** 0.80 gives 0.6291 and 0.75 gives 0.6286.

### 2.1 Dev at θ\* (identical for both rules; `eval_posthoc/dev/at_theta_star.csv`, CIs from `method_ci_at_theta_star.csv`)

| method | final answer | θ* | answerable acc | unans strict (never stop) | unans unified | balanced strict [CI] | balanced unified [CI] | pooled strict | pooled unified | mean L (all) | mean L (ans) | never-stop ans |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | reader | 0.90 | 0.669 | 0.560 | 0.560 | 0.614 [0.559, 0.667] | 0.614 [0.559, 0.667] | 0.623 | 0.623 | 11.12 | 9.64 | 0.274 |
| P_min | reader | 0.80 | 0.611 | 0.648 | 0.656 | 0.630 [0.575, 0.683] | 0.634 [0.579, 0.686] | 0.627 | 0.630 | 14.35 | 13.27 | 0.383 |
| P_mean | reader | 0.90 | 0.571 | 0.664 | 0.672 | 0.618 [0.562, 0.673] | 0.622 [0.567, 0.677] | 0.610 | 0.613 | 14.87 | 13.97 | 0.429 |
| A1 | reader | 0.75 | 0.714 | 0.552 | 0.568 | 0.633 [0.578, 0.688] | 0.641 [0.587, 0.696] | 0.647 | 0.653 | 12.61 | 10.97 | 0.229 |
| A2 | reader | 0.95 | 0.360 | 0.720 | 0.728 | 0.540 [0.483, 0.593] | 0.544 [0.487, 0.597] | 0.510 | 0.513 | 15.99 | 15.14 | 0.594 |
| A3 | reader | — | 0.743 | 0.112 | 0.168 | 0.427 [0.386, 0.472] | 0.455 [0.410, 0.502] | 0.480 | 0.503 | 8.19 | 7.95 | 0.103 |
| P'_min | last_item | 0.80 | 0.594 | 0.648 | 0.664 | 0.621 [0.566, 0.676] | 0.629 [0.574, 0.683] | 0.617 | 0.623 | 14.35 | 13.27 | 0.383 |
| P'_mean | last_item | 0.90 | 0.554 | 0.664 | 0.680 | 0.609 [0.553, 0.665] | 0.617 [0.561, 0.673] | 0.600 | 0.607 | 14.87 | 13.97 | 0.429 |
| A1' | direct | 0.75 | 0.709 | 0.552 | 0.576 | 0.630 [0.577, 0.686] | 0.642 [0.589, 0.698] | 0.643 | 0.653 | 12.61 | 10.97 | 0.229 |

## 3. Test at θ\* (identical for both rules; `eval_posthoc/test/at_theta_star.csv`)

| method | final answer | θ* | answerable acc | unans strict (never stop) | unans unified | balanced strict [CI] | balanced unified [CI] | pooled strict | pooled unified | mean L (all) | mean L (ans) | never-stop ans |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | reader | 0.90 | 0.597 | 0.525 | 0.547 | 0.561 [0.528, 0.595] | 0.572 [0.539, 0.606] | 0.563 | 0.573 | 11.34 | 10.08 | 0.340 |
| P_min | reader | 0.80 | 0.533 | 0.661 | 0.671 | 0.597 [0.565, 0.629] | 0.602 [0.570, 0.633] | 0.594 | 0.598 | 15.42 | 14.48 | 0.450 |
| P_mean | reader | 0.90 | 0.469 | 0.692 | 0.699 | 0.581 [0.548, 0.612] | 0.584 [0.552, 0.615] | 0.574 | 0.578 | 16.06 | 15.36 | 0.518 |
| A1 | reader | 0.75 | 0.617 | 0.537 | 0.566 | 0.577 [0.542, 0.609] | 0.591 [0.557, 0.623] | 0.579 | 0.593 | 13.89 | 12.93 | 0.349 |
| A2 | reader | 0.95 | 0.300 | 0.747 | 0.764 | 0.523 [0.495, 0.552] | 0.532 [0.504, 0.560] | 0.511 | 0.519 | 16.67 | 16.37 | 0.670 |
| A3 | reader | — | 0.756 | 0.179 | 0.263 | 0.467 [0.439, 0.494] | 0.509 [0.479, 0.538] | 0.483 | 0.523 | 9.87 | 9.44 | 0.116 |
| P'_min | last_item | 0.80 | 0.535 | 0.661 | 0.671 | 0.598 [0.566, 0.630] | 0.603 [0.571, 0.634] | 0.595 | 0.599 | 15.42 | 14.48 | 0.450 |
| P'_mean | last_item | 0.90 | 0.471 | 0.692 | 0.699 | 0.582 [0.549, 0.613] | 0.585 [0.552, 0.617] | 0.576 | 0.579 | 16.06 | 15.36 | 0.518 |
| A1' | direct | 0.75 | 0.617 | 0.537 | 0.568 | 0.577 [0.542, 0.609] | 0.592 [0.558, 0.624] | 0.579 | 0.594 | 13.89 | 12.93 | 0.349 |

For the six existing methods, the strict-θ\* rows equal `eval/test/at_theta_star.csv` exactly (pandas frame equality).

## 4. Paired bootstrap at θ\* (`eval_posthoc/<split>/paired_bootstrap_at_theta_star.csv`; identical for both rules)

Differences are A − B, as estimate [95% CI].

### 4.1 Test (post-hoc; see the test-reuse caveat)

| pair | θ_a / θ_b | balanced strict | balanced unified | answerable acc | unans strict | unans unified | mean L ans |
|---|---|---|---|---|---|---|---|
| P'_min - P_min | 0.80 / 0.80 | +0.001 [+0.000, +0.003] | +0.001 [+0.000, +0.003] | +0.002 [+0.000, +0.007] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.00 [+0.00, +0.00] |
| A1' - A1 | 0.75 / 0.75 | +0.000 [+0.000, +0.000] | +0.001 [+0.000, +0.004] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.002 [+0.000, +0.008] | +0.00 [+0.00, +0.00] |
| P'_min - A1' | 0.80 / 0.75 | +0.021 [+0.001, +0.043] | +0.011 [-0.010, +0.031] | -0.081 [-0.111, -0.052] | +0.124 [+0.095, +0.154] | +0.103 [+0.073, +0.132] | +1.55 [+1.22, +1.93] |
| P'_min - B1 | 0.80 / 0.90 | +0.037 [+0.003, +0.067] | +0.031 [-0.004, +0.062] | -0.062 [-0.106, -0.022] | +0.136 [+0.093, +0.181] | +0.124 [+0.079, +0.169] | +4.39 [+3.81, +4.99] |
| P_min - B1 | 0.80 / 0.90 | +0.036 [+0.002, +0.067] | +0.030 [-0.004, +0.061] | -0.064 [-0.108, -0.024] | +0.136 [+0.093, +0.181] | +0.124 [+0.079, +0.169] | +4.39 [+3.81, +4.99] |
| A1 - B1 | 0.75 / 0.90 | +0.016 [-0.016, +0.047] | +0.019 [-0.012, +0.048] | +0.019 [-0.022, +0.057] | +0.012 [-0.035, +0.061] | +0.019 [-0.027, +0.066] | +2.85 [+2.25, +3.47] |

- **P'_min − P_min and A1' − A1.** Essentially zero, because the answers agree (§1).
- **P'_min − B1.** +0.037 [+0.003, +0.067] strict and +0.031 [−0.004, +0.062] unified. This is the confirmatory P_min − B1 picture (+0.036 / +0.030) plus one question.
- **P'_min − A1'.** +0.021 [+0.001, +0.043] strict. The CI lower bound sits just above 0, while the confirmatory P_min − A1 was +0.020 [−0.001, +0.042].
  - The shift is one answerable question that P'_min gets right and P_min does not.
  - This is a post-hoc pair chosen after the confirmatory result. Do **not** read it as "P' separates from A1". Under unified it is +0.011 [−0.010, +0.031].
- **A1 − B1.** +0.016 [−0.016, +0.047] strict and +0.019 [−0.012, +0.048] unified. Neither CI excludes 0.

### 4.2 Dev

| pair | θ_a / θ_b | balanced strict | balanced unified | answerable acc | unans strict | unans unified | mean L ans |
|---|---|---|---|---|---|---|---|
| P'_min - P_min | 0.80 / 0.80 | -0.009 [-0.019, +0.000] | -0.005 [-0.017, +0.008] | -0.017 [-0.038, +0.000] | +0.000 [+0.000, +0.000] | +0.008 [+0.000, +0.026] | +0.00 [+0.00, +0.00] |
| A1' - A1 | 0.75 / 0.75 | -0.003 [-0.014, +0.006] | +0.001 [-0.012, +0.013] | -0.006 [-0.028, +0.012] | +0.000 [+0.000, +0.000] | +0.008 [+0.000, +0.025] | +0.00 [+0.00, +0.00] |
| P'_min - A1' | 0.80 / 0.75 | -0.009 [-0.048, +0.029] | -0.013 [-0.052, +0.025] | -0.114 [-0.173, -0.061] | +0.096 [+0.045, +0.155] | +0.088 [+0.038, +0.148] | +2.30 [+1.73, +2.97] |
| P'_min - B1 | 0.80 / 0.90 | +0.007 [-0.042, +0.059] | +0.015 [-0.036, +0.067] | -0.074 [-0.143, +0.000] | +0.088 [+0.016, +0.167] | +0.104 [+0.033, +0.180] | +3.63 [+2.71, +4.54] |
| P_min - B1 | 0.80 / 0.90 | +0.015 [-0.035, +0.068] | +0.019 [-0.031, +0.072] | -0.057 [-0.127, +0.017] | +0.088 [+0.016, +0.167] | +0.096 [+0.025, +0.171] | +3.63 [+2.71, +4.54] |
| A1 - B1 | 0.75 / 0.90 | +0.019 [-0.029, +0.072] | +0.027 [-0.019, +0.077] | +0.046 [-0.012, +0.111] | -0.008 [-0.086, +0.071] | +0.008 [-0.065, +0.084] | +1.33 [+0.43, +2.14] |

On dev, P' is slightly worse than P: 3 answerable questions where the reader is right and the last item is not, against 1 unanswerable unified gain.

## 5. Full θ sweeps (balanced accuracy; `eval_posthoc/<split>/sweep.csv`)

The P'/A1' columns are far below P/A1 at low θ. There the methods stop at L1–L3, where the observer often answers "unknown" while the reader still guesses (at θ = 0 all methods stop at L1). From θ = 0.50 up, X' stays within 0.015 of X.

### 5.1 Dev, balanced strict

| θ | B1 | P_min | P_mean | A1 | A2 | P'_min | P'_mean | A1' |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.271 | 0.271 | 0.271 | 0.271 | 0.271 | 0.074 | 0.074 | 0.077 |
| 0.05 | 0.271 | 0.427 | 0.337 | 0.440 | 0.271 | 0.419 | 0.106 | 0.423 |
| 0.10 | 0.271 | 0.427 | 0.337 | 0.440 | 0.271 | 0.419 | 0.106 | 0.423 |
| 0.15 | 0.271 | 0.427 | 0.337 | 0.440 | 0.271 | 0.419 | 0.106 | 0.423 |
| 0.20 | 0.271 | 0.427 | 0.340 | 0.440 | 0.271 | 0.419 | 0.109 | 0.423 |
| 0.25 | 0.271 | 0.427 | 0.354 | 0.440 | 0.271 | 0.419 | 0.129 | 0.423 |
| 0.30 | 0.271 | 0.427 | 0.363 | 0.444 | 0.271 | 0.419 | 0.154 | 0.427 |
| 0.35 | 0.284 | 0.435 | 0.385 | 0.455 | 0.277 | 0.430 | 0.185 | 0.443 |
| 0.40 | 0.287 | 0.450 | 0.398 | 0.477 | 0.277 | 0.445 | 0.232 | 0.465 |
| 0.45 | 0.284 | 0.469 | 0.405 | 0.482 | 0.274 | 0.463 | 0.302 | 0.473 |
| 0.50 | 0.298 | 0.494 | 0.431 | 0.494 | 0.277 | 0.488 | 0.420 | 0.494 |
| 0.55 | 0.337 | 0.518 | 0.442 | 0.513 | 0.297 | 0.518 | 0.428 | 0.513 |
| 0.60 | 0.362 | 0.554 | 0.446 | 0.549 | 0.297 | 0.551 | 0.432 | 0.546 |
| 0.65 | 0.378 | 0.575 | 0.447 | 0.566 | 0.335 | 0.567 | 0.436 | 0.561 |
| 0.70 | 0.411 | 0.606 | 0.475 | 0.614 | 0.369 | 0.601 | 0.470 | 0.609 |
| 0.75 | 0.496 | 0.618 | 0.511 | 0.633 | 0.414 | 0.613 | 0.511 | 0.630 |
| 0.80 | 0.533 | 0.630 | 0.582 | 0.633 | 0.470 | 0.621 | 0.576 | 0.630 |
| 0.85 | 0.554 | 0.599 | 0.610 | 0.606 | 0.492 | 0.596 | 0.601 | 0.603 |
| 0.90 | 0.614 | 0.586 | 0.618 | 0.594 | 0.530 | 0.583 | 0.609 | 0.591 |
| 0.95 | 0.606 | 0.556 | 0.573 | 0.593 | 0.540 | 0.553 | 0.570 | 0.590 |
| 1.00 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| A3 (no θ) | 0.427 |

### 5.2 Dev, balanced unified

| θ | B1 | P_min | P_mean | A1 | A2 | P'_min | P'_mean | A1' |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.307 | 0.307 | 0.307 | 0.307 | 0.307 | 0.078 | 0.078 | 0.093 |
| 0.05 | 0.307 | 0.455 | 0.393 | 0.460 | 0.307 | 0.459 | 0.110 | 0.451 |
| 0.10 | 0.307 | 0.455 | 0.393 | 0.460 | 0.307 | 0.459 | 0.110 | 0.451 |
| 0.15 | 0.307 | 0.455 | 0.393 | 0.460 | 0.307 | 0.459 | 0.110 | 0.451 |
| 0.20 | 0.307 | 0.455 | 0.392 | 0.460 | 0.307 | 0.459 | 0.117 | 0.451 |
| 0.25 | 0.307 | 0.455 | 0.390 | 0.460 | 0.307 | 0.459 | 0.145 | 0.451 |
| 0.30 | 0.307 | 0.455 | 0.403 | 0.464 | 0.307 | 0.455 | 0.174 | 0.455 |
| 0.35 | 0.320 | 0.459 | 0.429 | 0.475 | 0.313 | 0.458 | 0.209 | 0.475 |
| 0.40 | 0.319 | 0.478 | 0.442 | 0.493 | 0.313 | 0.477 | 0.264 | 0.497 |
| 0.45 | 0.324 | 0.493 | 0.445 | 0.502 | 0.314 | 0.483 | 0.338 | 0.501 |
| 0.50 | 0.334 | 0.522 | 0.459 | 0.518 | 0.313 | 0.512 | 0.456 | 0.526 |
| 0.55 | 0.377 | 0.538 | 0.470 | 0.529 | 0.337 | 0.538 | 0.456 | 0.537 |
| 0.60 | 0.402 | 0.570 | 0.466 | 0.565 | 0.333 | 0.571 | 0.460 | 0.570 |
| 0.65 | 0.410 | 0.587 | 0.467 | 0.582 | 0.367 | 0.583 | 0.460 | 0.581 |
| 0.70 | 0.439 | 0.614 | 0.487 | 0.626 | 0.401 | 0.613 | 0.482 | 0.625 |
| 0.75 | 0.520 | 0.626 | 0.527 | 0.641 | 0.446 | 0.629 | 0.527 | 0.642 |
| 0.80 | 0.549 | 0.634 | 0.590 | 0.637 | 0.494 | 0.629 | 0.588 | 0.634 |
| 0.85 | 0.570 | 0.603 | 0.618 | 0.610 | 0.512 | 0.604 | 0.609 | 0.607 |
| 0.90 | 0.614 | 0.590 | 0.622 | 0.598 | 0.538 | 0.591 | 0.617 | 0.595 |
| 0.95 | 0.606 | 0.556 | 0.573 | 0.593 | 0.544 | 0.553 | 0.570 | 0.590 |
| 1.00 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| A3 (no θ) | 0.455 |

### 5.3 Test, balanced strict (descriptive; θ is not re-read from it)

| θ | B1 | P_min | P_mean | A1 | A2 | P'_min | P'_mean | A1' |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.296 | 0.296 | 0.296 | 0.296 | 0.296 | 0.055 | 0.055 | 0.055 |
| 0.05 | 0.296 | 0.467 | 0.346 | 0.459 | 0.296 | 0.461 | 0.096 | 0.455 |
| 0.10 | 0.296 | 0.467 | 0.346 | 0.459 | 0.296 | 0.461 | 0.096 | 0.455 |
| 0.15 | 0.296 | 0.467 | 0.345 | 0.459 | 0.296 | 0.461 | 0.096 | 0.455 |
| 0.20 | 0.296 | 0.467 | 0.349 | 0.458 | 0.296 | 0.461 | 0.097 | 0.454 |
| 0.25 | 0.296 | 0.467 | 0.360 | 0.458 | 0.296 | 0.461 | 0.119 | 0.454 |
| 0.30 | 0.297 | 0.470 | 0.366 | 0.458 | 0.296 | 0.463 | 0.143 | 0.454 |
| 0.35 | 0.303 | 0.473 | 0.387 | 0.470 | 0.296 | 0.469 | 0.181 | 0.464 |
| 0.40 | 0.312 | 0.492 | 0.402 | 0.487 | 0.296 | 0.483 | 0.226 | 0.479 |
| 0.45 | 0.318 | 0.509 | 0.421 | 0.501 | 0.297 | 0.504 | 0.295 | 0.494 |
| 0.50 | 0.332 | 0.522 | 0.468 | 0.517 | 0.304 | 0.521 | 0.456 | 0.513 |
| 0.55 | 0.348 | 0.539 | 0.474 | 0.534 | 0.310 | 0.538 | 0.463 | 0.530 |
| 0.60 | 0.376 | 0.550 | 0.479 | 0.538 | 0.331 | 0.550 | 0.470 | 0.535 |
| 0.65 | 0.411 | 0.555 | 0.489 | 0.556 | 0.352 | 0.557 | 0.477 | 0.557 |
| 0.70 | 0.426 | 0.572 | 0.511 | 0.571 | 0.370 | 0.574 | 0.509 | 0.572 |
| 0.75 | 0.469 | 0.576 | 0.527 | 0.577 | 0.399 | 0.577 | 0.524 | 0.577 |
| 0.80 | 0.496 | 0.597 | 0.561 | 0.592 | 0.436 | 0.598 | 0.562 | 0.592 |
| 0.85 | 0.543 | 0.585 | 0.571 | 0.592 | 0.463 | 0.587 | 0.573 | 0.592 |
| 0.90 | 0.561 | 0.588 | 0.581 | 0.598 | 0.518 | 0.588 | 0.582 | 0.598 |
| 0.95 | 0.576 | 0.555 | 0.576 | 0.566 | 0.523 | 0.555 | 0.576 | 0.566 |
| 1.00 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| A3 (no θ) | 0.467 |

### 5.4 Test, balanced unified (descriptive)

| θ | B1 | P_min | P_mean | A1 | A2 | P'_min | P'_mean | A1' |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.338 | 0.338 | 0.338 | 0.338 | 0.338 | 0.058 | 0.058 | 0.058 |
| 0.05 | 0.338 | 0.509 | 0.395 | 0.502 | 0.338 | 0.504 | 0.102 | 0.493 |
| 0.10 | 0.338 | 0.509 | 0.395 | 0.502 | 0.338 | 0.504 | 0.102 | 0.493 |
| 0.15 | 0.338 | 0.509 | 0.394 | 0.502 | 0.338 | 0.504 | 0.102 | 0.493 |
| 0.20 | 0.338 | 0.509 | 0.399 | 0.501 | 0.338 | 0.504 | 0.103 | 0.492 |
| 0.25 | 0.340 | 0.509 | 0.406 | 0.501 | 0.338 | 0.504 | 0.125 | 0.492 |
| 0.30 | 0.341 | 0.513 | 0.411 | 0.501 | 0.338 | 0.506 | 0.152 | 0.492 |
| 0.35 | 0.350 | 0.517 | 0.437 | 0.515 | 0.340 | 0.512 | 0.194 | 0.503 |
| 0.40 | 0.359 | 0.531 | 0.457 | 0.527 | 0.338 | 0.519 | 0.243 | 0.512 |
| 0.45 | 0.366 | 0.545 | 0.474 | 0.538 | 0.341 | 0.531 | 0.323 | 0.524 |
| 0.50 | 0.378 | 0.555 | 0.509 | 0.553 | 0.350 | 0.548 | 0.498 | 0.542 |
| 0.55 | 0.394 | 0.567 | 0.516 | 0.563 | 0.355 | 0.564 | 0.503 | 0.555 |
| 0.60 | 0.419 | 0.571 | 0.524 | 0.560 | 0.375 | 0.571 | 0.512 | 0.559 |
| 0.65 | 0.449 | 0.571 | 0.532 | 0.573 | 0.397 | 0.573 | 0.513 | 0.575 |
| 0.70 | 0.458 | 0.589 | 0.547 | 0.588 | 0.415 | 0.591 | 0.540 | 0.590 |
| 0.75 | 0.496 | 0.589 | 0.553 | 0.591 | 0.438 | 0.592 | 0.550 | 0.592 |
| 0.80 | 0.512 | 0.602 | 0.574 | 0.603 | 0.470 | 0.603 | 0.576 | 0.603 |
| 0.85 | 0.554 | 0.588 | 0.584 | 0.600 | 0.488 | 0.589 | 0.588 | 0.600 |
| 0.90 | 0.572 | 0.591 | 0.584 | 0.603 | 0.531 | 0.591 | 0.585 | 0.603 |
| 0.95 | 0.580 | 0.555 | 0.578 | 0.567 | 0.532 | 0.555 | 0.578 | 0.567 |
| 1.00 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| A3 (no θ) | 0.509 |

## 6. Mechanism

- **The observer answers almost always match the reader where the gated methods stop.** These are the disagreements among stopped questions at θ\*. Final answers are compared after normalisation.

| split / method | stopped | disagree | reader right, observer wrong | observer right, reader wrong | both wrong |
|---|---|---|---|---|---|
| dev P_min 0.80 | 152 | 4 | 3 (all answerable) | 1 (unanswerable) | 0 |
| dev A1 0.75 | 191 | 5 | 2 | 2 (1 ans, 1 unans) | 1 |
| test P_min 0.80 | 399 | 2 | 0 | 1 (answerable) | 1 |
| test A1 0.75 | 498 | 1 | 0 | 1 (unanswerable) | 0 |

  The gate requires resolved = true and answer ≠ unknown with P_ans ≥ θ. By the time it passes, the image is clear enough that the observer and the reader say the same thing.
- **Unified credit on unanswerable is small because the observer is also wrong on reader-wrong questions.** Observer correct at L20 on unanswerable (= reader wrong at L20) questions:

| split | last item | direct_resolved |
|---|---|---|
| dev (125) | 4.8% (6) | 2.4% (3) |
| test (419) | 4.3% (18) | 4.8% (20) |

  On unanswerable questions at L20, the observer usually gives the same wrong answer as the reader. The other wrong answers are mostly "unknown".

| split | last item = reader | last item "unknown" | direct = reader | direct "unknown" |
|---|---|---|---|---|
| dev | 71% | 15% | 77% | 14% |
| test | 68% | 21% | 69% | 19% |

  This fits the GQA label-granularity and reference-ambiguity reading of "unanswerable" (§27.10). An unanswerable question earns unified credit only if a method stops at a level where the final answer matches GT. Observer answers match GT at *some* level on only 12–15% of unanswerable questions, against 22–24% for the reader. Few of those levels are stop levels.
- **Why X' falls below X at low θ.** Early levels have many observer "unknown" answers, scored wrong, while the reader guesses. This only matters below the θ\* region.

## 7. Caveats

- **Post-hoc.** The P'/A1' definitions and the θ\*_unified rule were added after the confirmatory test evaluation. Test numbers are descriptive and optimistic: they are a repeated look at the same test set, and the CIs ignore θ-selection error. The confirmatory claims stay those of `gqa-stop-v27-test.md` §1.
- **The label is still the reader's L20 correctness.** A1' and P' are scored against GT, but "answerable" is defined by the reader. So the observer methods are not on an equal footing for the answerable side: a question the observer gets right but the reader does not is labelled unanswerable.
- **No multiple-comparison correction.** Six pairs × two scorings × two splits.

## 8. Files and commands

- **Code.**
  - `src/milestone1/gqa_stop/v27_eval.py` adds `POSTHOC_METHODS`, `ALL_METHODS`, observer correctness in `EvalArrays.answer_correct`, per-method `correct_matrix`, and `evaluate_posthoc`. The existing method outputs are byte-identical.
  - `src/milestone1/gqa_stop/cli.py` adds the `--posthoc` flag.
- **Tests.** `tests/test_gqa_stop_v27.py` has 7 new tests.
- **Outputs.** `eval_posthoc/{dev,test}/`:
  - `sweep.csv`, `sweep_by_attribute.csv`
  - `at_theta_star.csv`, `at_theta_star_by_attribute.csv`
  - `method_ci_at_theta_star.csv`
  - `paired_bootstrap.csv` (same θ, full grid) and `paired_bootstrap_at_theta_star.csv`
  - `summary.json`, `figures/`
  - `theta_star.json` (dev only)
  - The selection column (strict / unified) and the answer_source column are in every per-method file.

```bash
uv run python -m milestone1.gqa_stop.cli v27-evaluate --config configs/gqa_stop_v27.json --split dev --posthoc
uv run python -m milestone1.gqa_stop.cli v27-evaluate --config configs/gqa_stop_v27.json --split test --posthoc --confirm-test-eval
```

- **Identity check of existing methods.** The existing `v27-evaluate` (no `--posthoc`) was run with the modified code into a scratch `--repo-root` on dev and test. All seven CSVs in `eval/dev/` and `eval/test/` were byte-identical. `theta_star.json` differed only in the time field.
