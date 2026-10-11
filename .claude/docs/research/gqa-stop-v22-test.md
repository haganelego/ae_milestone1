# gqa_stop v22 test (plan §21/§22/§23 protocol): frozen dev-102 θ on the test visual subset (251) and reader-wrong test set (233)

Status: **evaluated once, 2026-10-11.** The skeleton (frozen θ table, hypotheses including H_pans) was written 2026-10-10T07:45:06Z at HEAD d2c7161, before any test metric existed. Test inference resumed 2026-10-11 on GPU (user decision), and the frozen-θ evaluation (`evaluate-v22`, `firing-test`) ran exactly once, at 2026-10-11T05:09Z. The θ sweep added afterwards is DESCRIPTIVE only.
- **v1 arm skipped by user decision 2026-10-11.** checklist_v1_min, checklist_v1_mean and verbalised Direct were not run on test (`--skip-v1arm`; the verbalised Direct call inside the v4 runner was skipped too). Their columns are NaN and their test rows are absent. The frozen file still lists them and was not edited.
- **Methods compared on test.** direct_v4, checklist_v4_min, direct_pans, pans_min. The other v4-derived logged variants were evaluated at no extra cost; they are in the CSVs and not discussed.

Run: `configs/gqa_stop_v22.json`, output `outputs/gqa_stop/visapp_gqa_v22/`. A13 clean decoding, think=false.
Frozen policy: `frozen_policy.json`, sha256 `05a84b224ba8a5886b7d497e2628a764fec4ae453379fa39d7d9bcafcb4e2bf5`, read-only. θ was copied from the §22 dev-102 grid (`dev_targets.csv`) and verified by recomputation.

| target | checklist_v4_min | direct_v4 | checklist_v1_min | direct verbalised | direct_pans (logged) | pans_min (logged) | Fixed-dev |
|---|---|---|---|---|---|---|---|
| 0.90 | 0.14 | 0.22 | 0.90 | 0.95 | 0.78 | 0.53 | L16 |
| **0.95** | **0.28** | **0.30** | **0.90** | **0.95** | **0.83** | **0.58** | **L20** |
| 0.97 | 0.36 | 0.36 | 1.00 | 0.95 | 0.85 | 0.78 | L20 |
| 0.99 | 0.42 | 0.39 | 1.00 | ∞ | 0.88 | 0.84 | L20 |

CV caveat, recorded in the frozen file: dev single-θ selection loses about 0.01–0.02 accuracy out of fold. direct_v4 at the 0.95 target gives 0.934 out of fold (`box_attrtheta_cv.csv`).

## Hypotheses

### Pre-registered before dev (plan §19/§21/§22/§23)

Verdicts are in Results → Verdict and H_pans verdict.
- **Main.** checklist_v4_min reaches accuracy ≥ 0.95 at a lower mean level than fair v1 (checklist_v1_min) and than Fixed-dev, with the paired image-cluster bootstrap CI below 0.
- **Comparator.** checklist_v4_min vs direct_v4: mean-level and accuracy difference. Dev §22 showed no difference.
- **§23 firing.** The firing rate on the 233 reader-wrong test questions at the frozen θ, compared between methods.

### Dev-derived hypothesis H_pans (formed after seeing dev-99 firing, pre-registered here before test evaluation)

- **Origin.** The cache-only probes (commit d876011, `gqa-stop-unanswerable-probes.md`) found a pattern on dev, and it held out of fold. P_ans alone fires far less on reader-wrong questions than P_ans × Rel at equal accuracy and mean level. P_ans alone is the logged direct_pans / pans_min variants; Rel is the self-assessed confidence.
- **Why it is pre-registered, but weaker.** This hypothesis was formed **after** looking at the dev-99 firing results, so it is not part of the original pre-registration. It is registered here before any test metric is computed. Both variants were already frozen as logged variants, with θ selected on dev 102 in the same grid; nothing is re-tuned.
- **H_pans.** On test, at their frozen 0.95 θ (direct_pans 0.83, pans_min 0.58), direct_pans and pans_min have a **lower firing rate on the 233 reader-wrong test questions** than direct_v4 (θ 0.30) and checklist_v4_min (θ 0.28).
  - Test: paired image-cluster bootstrap, 2000 draws, seed 20261008; the upper CI bound of the firing-rate difference must be < 0.
  - Condition: their accuracy on the 251 test visual questions must be **≥ 0.95 − 0.02 = 0.93**.
- **Rows reported.** direct_pans and pans_min rows appear in every test table and in the 233-question firing table, with paired bootstrap against direct_v4 and checklist_v4_min.

## Results (evaluated once at the frozen θ, 2026-10-11)

### Run facts
- **Environment.** RTX 5080, driver 580.178.04 (userland = kernel). qwen3.5:9b loaded alone after `isolate_model`: 34/34 layers on GPU, ollama log "CLIP using CUDA0 backend". Determinism gate re-run before inference: passed, 120 calls, 0 argmax violations, repeat pairs identical. A fresh re-send of 88 cached dev requests (planner_v3, reader, direct_lp, relevance_v4, observer_v3_lp; 4 questions × L1/L10/L20) was byte-identical in content and logprobs.
- **Calls.** 251 visual subset: 33,152 new calls in 4 h 35 min. 233 reader-wrong: 33,100 new calls in 4 h 26 min. Mean 0.47–0.48 s per call. Foreground chunks of ≤ 585 s, resumed from cache.
- **Validity.** Planner, reader, direct_lp, relevance and observer records are all valid. The answer token was located in 100% of direct_lp records and Rel had Yes/No in 100%. Planner items: 2 items for 241 of the 251 and 3 for 10.
- **Audit** (`audit-test`, every test record incl. reused ones, 69,357 records): 0 records with non-generation time > 5 s (max 0.78 s), 0 argmax violations in 445,901 logprob positions. Nothing was quarantined.
- **Sets.** 251 questions / 213 images (color 127, material 105, shape 19). 233 reader-wrong questions / 204 images (color 143, material 60, shape 30). Reader accuracy on the 233 is 0.05–0.12 at L1–L19 and 0 at L20 (dev 99: 0.05–0.13).

### Verdict
- **Test accuracy falls short of every dev target.** At the 0.95 target, checklist_v4_min reaches 0.940 [0.910, 0.967] and direct_v4 0.924 [0.893, 0.956]. This matches the out-of-fold CV caveat (direct_v4 0.934). The P_ans-only variants lose more: direct_pans 0.904, pans_min 0.896.
- **checklist_v4_min ≈ direct_v4 on test, as on dev.** At 0.95 the mean-level difference is −0.27 [−0.64, +0.07] and the accuracy difference +0.016 [+0.000, +0.035]. At 0.97 and 0.99 both differences include 0. At 0.90 the checklist stops 0.87 levels earlier [−1.18, −0.57] but fires more on the 233 (+0.056 [+0.008, +0.107]) and is 0.020 less accurate (CI includes 0). That is a lower operating point on the same curve, not dominance.
- **Both v4 signals stay above the fixed concave hull.** At 0.95 checklist_v4_min is +0.056 [+0.018, +0.084] above the test hull; direct_v4 is +0.034 [−0.003, +0.064].
- **Firing on reader-wrong questions is unchanged from dev.** At 0.95 checklist_v4_min fires on 0.652 and direct_v4 on 0.644 of the 233 (dev 0.646 / 0.677). The difference is +0.009 [−0.034, +0.055].
- **H_pans is not supported.** direct_pans does fire less than both v4 methods at 0.95, by about 0.23–0.24 with CIs below 0. But its test accuracy is 0.904, below the pre-registered condition of 0.93. pans_min fails both parts: it fires as often as the v4 methods (+0.009 and +0.000, CIs include 0) and its accuracy is 0.896.
- **Main hypothesis.** The fair-v1 comparison was not evaluated because the v1 arm was skipped. Against Fixed-dev (L20 at the 0.95 target, accuracy 1.000 by construction), checklist_v4_min stops 6.30 levels earlier [−6.99, −5.58] at −0.060 accuracy [−0.090, −0.033]. It does not reach the 0.95 target as a point estimate, so the main hypothesis is not met as stated.

### Frozen-θ table, answerable set (dev 102 vs test 251)

θ and dev values are copied from the frozen policy and `dev_targets.csv` (in-sample on dev). The test CI is a paired image-cluster bootstrap, 2000 draws, seed 20261008, with θ held fixed. "s&w" is the number of questions stopped before L20 with a wrong reader answer. Fixed-dev at 0.95 and above is L20, which is correct for every question by construction.

| target | method | θ | dev acc | **test acc** [95% CI] | dev mean L | **test mean L** [95% CI] | dev stop20 | test stop20 | test s&w | test acc − hull [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.90 | direct_v4 | 0.22 | 0.902 | 0.900 [0.863, 0.935] | 12.25 | 12.44 [11.69, 13.17] | 0.17 | 0.16 | 25 | +0.047 [+0.004, +0.080] |
| 0.90 | checklist_v4_min | 0.14 | 0.912 | 0.880 [0.838, 0.918] | 11.87 | 11.57 [10.81, 12.33] | 0.18 | 0.16 | 30 | +0.050 [+0.006, +0.086] |
| 0.90 | direct_pans | 0.78 | 0.912 | 0.849 [0.804, 0.891] | 11.52 | 11.23 [10.20, 12.22] | 0.36 | 0.35 | 38 | +0.028 [-0.023, +0.068] |
| 0.90 | pans_min | 0.53 | 0.902 | 0.876 [0.835, 0.914] | 11.03 | 10.93 [10.11, 11.74] | 0.19 | 0.19 | 31 | +0.064 [+0.019, +0.099] |
| 0.90 | Fixed-dev L16 | | | 0.928 | | 16.00 | | | | |
| **0.95** | direct_v4 | 0.3 | 0.951 | 0.924 [0.893, 0.956] | 13.83 | 13.98 [13.27, 14.68] | 0.25 | 0.26 | 19 | +0.034 [-0.003, +0.064] |
| **0.95** | checklist_v4_min | 0.28 | 0.951 | 0.940 [0.910, 0.967] | 14.15 | 13.71 [13.01, 14.42] | 0.29 | 0.27 | 15 | +0.056 [+0.018, +0.084] |
| **0.95** | direct_pans | 0.83 | 0.951 | 0.904 [0.866, 0.939] | 13.48 | 12.79 [11.73, 13.77] | 0.44 | 0.45 | 24 | +0.042 [-0.005, +0.080] |
| **0.95** | pans_min | 0.58 | 0.951 | 0.896 [0.857, 0.930] | 11.73 | 11.52 [10.73, 12.33] | 0.24 | 0.22 | 26 | +0.068 [+0.024, +0.101] |
| 0.95 | Fixed-dev L20 | | | 1.000 | | 20.00 | | | | |
| 0.97 | direct_v4 | 0.36 | 0.971 | 0.956 [0.930, 0.980] | 14.77 | 14.92 [14.21, 15.60] | 0.31 | 0.33 | 11 | +0.046 [+0.012, +0.070] |
| 0.97 | checklist_v4_min | 0.36 | 0.971 | 0.964 [0.940, 0.984] | 15.01 | 14.93 [14.25, 15.60] | 0.34 | 0.36 | 9 | +0.053 [+0.019, +0.078] |
| 0.97 | direct_pans | 0.85 | 0.971 | 0.924 [0.888, 0.956] | 14.25 | 13.48 [12.46, 14.48] | 0.49 | 0.51 | 19 | +0.045 [-0.000, +0.080] |
| 0.97 | pans_min | 0.78 | 0.971 | 0.952 [0.925, 0.976] | 14.56 | 14.43 [13.66, 15.19] | 0.47 | 0.43 | 12 | +0.052 [+0.016, +0.078] |
| 0.97 | Fixed-dev L20 | | | 1.000 | | 20.00 | | | | |
| 0.99 | direct_v4 | 0.39 | 0.990 | 0.964 [0.940, 0.984] | 15.46 | 15.35 [14.67, 16.01] | 0.39 | 0.36 | 9 | +0.046 [+0.012, +0.069] |
| 0.99 | checklist_v4_min | 0.42 | 0.990 | 0.972 [0.950, 0.992] | 15.76 | 15.54 [14.89, 16.18] | 0.44 | 0.42 | 7 | +0.050 [+0.016, +0.072] |
| 0.99 | direct_pans | 0.88 | 0.990 | 0.944 [0.910, 0.971] | 15.86 | 14.47 [13.52, 15.38] | 0.68 | 0.55 | 14 | +0.043 [+0.002, +0.074] |
| 0.99 | pans_min | 0.84 | 0.990 | 0.968 [0.945, 0.988] | 15.73 | 15.58 [14.82, 16.32] | 0.56 | 0.55 | 8 | +0.046 [+0.008, +0.070] |
| 0.99 | Fixed-dev L20 | | | 1.000 | | 20.00 | | | | |

Breakdown at 0.95 on test (`test_attribute_breakdown.csv`), accuracy @ mean level:

| method | color (127) | material (105) | shape (19) | relate 0 (147) | relate 1 (104) |
|---|---|---|---|---|---|
| direct_v4 | 0.898 @ 11.22 | 0.952 @ 17.11 | 0.947 @ 15.05 | 0.912 @ 13.25 | 0.942 @ 15.00 |
| checklist_v4_min | 0.898 @ 10.95 | 0.990 @ 17.34 | 0.947 @ 12.00 | 0.939 @ 13.16 | 0.942 @ 14.47 |
| direct_pans | 0.866 @ 9.55 | 0.933 @ 15.87 | 1.000 @ 17.42 | 0.912 @ 13.00 | 0.894 @ 12.49 |
| pans_min | 0.890 @ 9.14 | 0.905 @ 14.96 | 0.895 @ 8.37 | 0.884 @ 11.22 | 0.913 @ 11.93 |

The accuracy shortfall sits in colour questions, where every method is at 0.87–0.90.

### §23 firing on the 233 reader-wrong test questions (frozen θ)

A question fires when the score is ≥ θ at some level 1–20 (k = 1); the forced L20 stop is not a firing. The test CI is an image-cluster bootstrap (2000 draws, seed 20261008). Dev values are from `ext_dev_wrong/firing_summary.csv` at the same θ.

| target | method | θ | dev firing (n=99) | **test firing** (n=233) [95% CI] | test fired | test median firing L | test reader correct at firing |
|---|---|---|---|---|---|---|---|
| 0.90 | direct_v4 | 0.22 | 0.778 | 0.742 [0.686, 0.799] | 173 | L12 | 18 |
| 0.90 | checklist_v4_min | 0.14 | 0.788 | 0.798 [0.745, 0.850] | 186 | L11 | 21 |
| 0.90 | direct_pans | 0.78 | 0.556 | 0.545 [0.475, 0.613] | 127 | L5 | 15 |
| 0.90 | pans_min | 0.53 | 0.758 | 0.700 [0.638, 0.765] | 163 | L8 | 12 |
| 0.95 | direct_v4 | 0.3 | 0.677 | 0.644 [0.580, 0.710] | 150 | L14 | 12 |
| 0.95 | checklist_v4_min | 0.28 | 0.646 | 0.652 [0.592, 0.714] | 152 | L13 | 8 |
| 0.95 | direct_pans | 0.83 | 0.465 | 0.412 [0.346, 0.477] | 96 | L5 | 9 |
| 0.95 | pans_min | 0.58 | 0.697 | 0.652 [0.591, 0.719] | 152 | L8 | 12 |
| 0.97 | direct_v4 | 0.36 | 0.545 | 0.545 [0.477, 0.615] | 127 | L14 | 7 |
| 0.97 | checklist_v4_min | 0.36 | 0.556 | 0.545 [0.478, 0.613] | 127 | L14 | 7 |
| 0.97 | direct_pans | 0.85 | 0.424 | 0.352 [0.288, 0.414] | 82 | L5 | 9 |
| 0.97 | pans_min | 0.78 | 0.384 | 0.382 [0.316, 0.450] | 89 | L9 | 5 |
| 0.99 | direct_v4 | 0.39 | 0.535 | 0.524 [0.458, 0.593] | 122 | L14 | 7 |
| 0.99 | checklist_v4_min | 0.42 | 0.495 | 0.498 [0.431, 0.568] | 116 | L15 | 2 |
| 0.99 | direct_pans | 0.88 | 0.303 | 0.266 [0.209, 0.325] | 62 | L5.5 | 7 |
| 0.99 | pans_min | 0.84 | 0.283 | 0.313 [0.255, 0.377] | 73 | L10 | 4 |

- **Firing level.** At 0.95, direct_pans fires early (median L5; 73 of its 96 firings at L ≤ 9), and so does pans_min (median L8; 86 of 152). The v4 methods fire around L13–L14.
- **Breakdown at 0.95** (`firing_breakdown.csv`; color 143, material 60, shape 30, relate 0: 123, relate 1: 110):

| method | color | material | shape | relate 0 | relate 1 |
|---|---|---|---|---|---|
| direct_v4 | 0.713 | 0.500 | 0.600 | 0.691 | 0.591 |
| checklist_v4_min | 0.720 | 0.433 | 0.767 | 0.683 | 0.618 |
| direct_pans | 0.448 | 0.417 | 0.233 | 0.390 | 0.436 |
| pans_min | 0.671 | 0.500 | 0.867 | 0.667 | 0.636 |

### Paired comparisons on test (image-cluster bootstrap, 2000 draws, seed 20261008)

Accuracy and mean level are on the 251; firing is on the 233 (`test_paired_comparisons.csv`, `ext_test_wrong/firing_paired.csv`).

| target | comparison (a − b) | Δ acc [95% CI] | Δ mean L [95% CI] | Δ firing [95% CI] | only a / only b fired |
|---|---|---|---|---|---|
| 0.90 | checklist_v4_min − direct_v4 | -0.020 [-0.052, +0.008] | -0.87 [-1.18, -0.57] | +0.056 [+0.008, +0.107] | 23 / 10 |
| 0.90 | direct_pans − direct_v4 | -0.052 [-0.098, -0.008] | -1.21 [-2.08, -0.39] | -0.197 [-0.268, -0.126] | 18 / 64 |
| 0.90 | direct_pans − checklist_v4_min | -0.032 [-0.082, +0.019] | -0.34 [-1.27, +0.53] | -0.253 [-0.327, -0.179] | 14 / 73 |
| 0.90 | pans_min − direct_v4 | -0.024 [-0.061, +0.012] | -1.51 [-2.02, -1.02] | -0.043 [-0.098, +0.017] | 18 / 28 |
| 0.90 | pans_min − checklist_v4_min | -0.004 [-0.036, +0.028] | -0.64 [-1.04, -0.25] | -0.099 [-0.149, -0.047] | 6 / 29 |
| 0.95 | checklist_v4_min − direct_v4 | +0.016 [+0.000, +0.035] | -0.27 [-0.64, +0.07] | +0.009 [-0.034, +0.055] | 14 / 12 |
| 0.95 | direct_pans − direct_v4 | -0.020 [-0.066, +0.020] | -1.19 [-2.04, -0.37] | -0.232 [-0.308, -0.158] | 20 / 74 |
| 0.95 | direct_pans − checklist_v4_min | -0.036 [-0.078, +0.004] | -0.92 [-1.82, -0.04] | -0.240 [-0.319, -0.164] | 22 / 78 |
| 0.95 | pans_min − direct_v4 | -0.028 [-0.068, +0.008] | -2.46 [-3.00, -1.99] | +0.009 [-0.056, +0.072] | 27 / 25 |
| 0.95 | pans_min − checklist_v4_min | -0.044 [-0.079, -0.012] | -2.19 [-2.60, -1.80] | +0.000 [-0.056, +0.058] | 20 / 20 |
| 0.97 | checklist_v4_min − direct_v4 | +0.008 [-0.012, +0.029] | +0.01 [-0.37, +0.35] | +0.000 [-0.053, +0.055] | 18 / 18 |
| 0.97 | direct_pans − direct_v4 | -0.032 [-0.070, +0.000] | -1.44 [-2.27, -0.63] | -0.193 [-0.267, -0.123] | 18 / 63 |
| 0.97 | direct_pans − checklist_v4_min | -0.040 [-0.078, -0.004] | -1.45 [-2.36, -0.56] | -0.193 [-0.267, -0.121] | 21 / 66 |
| 0.97 | pans_min − direct_v4 | -0.004 [-0.031, +0.020] | -0.49 [-1.05, +0.07] | -0.163 [-0.236, -0.095] | 13 / 51 |
| 0.97 | pans_min − checklist_v4_min | -0.012 [-0.031, +0.004] | -0.50 [-0.92, -0.09] | -0.163 [-0.220, -0.107] | 5 / 43 |
| 0.99 | checklist_v4_min − direct_v4 | +0.008 [-0.012, +0.028] | +0.19 [-0.19, +0.54] | -0.026 [-0.083, +0.033] | 18 / 24 |
| 0.99 | direct_pans − direct_v4 | -0.020 [-0.055, +0.011] | -0.88 [-1.67, -0.10] | -0.258 [-0.325, -0.191] | 10 / 70 |
| 0.99 | direct_pans − checklist_v4_min | -0.028 [-0.065, +0.004] | -1.07 [-1.92, -0.22] | -0.232 [-0.303, -0.164] | 16 / 70 |
| 0.99 | pans_min − direct_v4 | +0.004 [-0.023, +0.031] | +0.23 [-0.29, +0.78] | -0.210 [-0.281, -0.144] | 11 / 60 |
| 0.99 | pans_min − checklist_v4_min | -0.004 [-0.024, +0.016] | +0.04 [-0.40, +0.50] | -0.185 [-0.243, -0.133] | 4 / 47 |

### H_pans verdict (pre-registered before test evaluation)

| variant (θ at 0.95) | test acc ≥ 0.93? | firing − direct_v4 [95% CI] | firing − checklist_v4_min [95% CI] | verdict |
|---|---|---|---|---|
| direct_pans (0.83) | no: 0.904 [0.866, 0.939] | −0.232 [−0.308, −0.158] | −0.240 [−0.319, −0.164] | **not supported**: lower firing, but the accuracy condition fails |
| pans_min (0.58) | no: 0.896 [0.857, 0.930] | +0.009 [−0.056, +0.072] | +0.000 [−0.056, +0.058] | **not supported**: no firing reduction and the accuracy condition fails |

The dev-selected θ of the P_ans-only variants does not transfer. On dev both reached 0.951 accuracy at these θ; on test they lose 0.05.

### θ sweep, DESCRIPTIVE (not a result)

**Read this as a description only.** θ read off the test sweep is chosen on test and is not a result. The frozen-θ tables above are the pre-registered result. Rule: k = 1, θ ∈ {0.50, 0.55, …, 1.00}. Accuracy and stop-level quartiles (numpy linear percentiles) are on the answerable set; firing is on the reader-wrong set with the same k = 1 rule, forced L20 not a firing. The frozen θ of direct_v4 (0.22–0.39) and checklist_v4_min (0.14–0.42) lie below this range. Where a frozen θ is on the grid, the sweep reproduces the frozen row: direct_pans θ 0.85 gives 0.924 @ 13.48, firing 0.352.

Descriptive observations:
- **Above θ 0.5 the v4 products are already conservative.** direct_v4 and checklist_v4_min have accuracy ≥ 0.98 with median stop L20 at every θ in the range.
- **P_ans-only variants cover the middle of the range.** direct_pans needs θ ≥ 0.90 for accuracy ≥ 0.95 on test (0.976 @ 15.58, firing 0.215); on dev θ 0.85 sufficed (0.971 @ 14.25). pans_min reaches 0.952 @ 13.93 at θ 0.75 with firing 0.446 on test.
- **The dev pattern holds descriptively.** At similar answerable mean level, P_ans-only signals fire less: test direct_pans θ 0.85 at 13.48 fires 0.352, while the frozen direct_v4 at 13.98 fires 0.644. The pre-registered test of this pattern (H_pans) failed on the accuracy condition, so this remains an observation.

Figures, one per axis, in `sweep_v22/figures/`: `sweep_accuracy.png`, `sweep_mean_level.png` (mean; the test median is dotted and the test IQR is a band), `sweep_stop20_rate.png`, `sweep_firing_rate.png`. Test is solid, dev dashed.

**Test** (answerable 251, reader-wrong 233; `sweep_v22/test_sweep.csv`)

| method | θ | acc | mean L | Q1 | median | Q3 | stop20 | firing |
|---|---|---|---|---|---|---|---|---|
| direct_v4 | 0.50 | 0.980 | 16.67 | 14 | 20 | 20 | 0.53 | 0.391 |
| direct_v4 | 0.55 | 0.996 | 17.19 | 15 | 20 | 20 | 0.62 | 0.313 |
| direct_v4 | 0.60 | 0.996 | 17.56 | 16 | 20 | 20 | 0.65 | 0.249 |
| direct_v4 | 0.65 | 0.996 | 17.99 | 17.5 | 20 | 20 | 0.70 | 0.197 |
| direct_v4 | 0.70 | 0.996 | 18.33 | 19 | 20 | 20 | 0.74 | 0.163 |
| direct_v4 | 0.75 | 1.000 | 18.69 | 20 | 20 | 20 | 0.80 | 0.124 |
| direct_v4 | 0.80 | 1.000 | 19.00 | 20 | 20 | 20 | 0.82 | 0.090 |
| direct_v4 | 0.85 | 1.000 | 19.37 | 20 | 20 | 20 | 0.89 | 0.052 |
| direct_v4 | 0.90 | 1.000 | 19.73 | 20 | 20 | 20 | 0.94 | 0.039 |
| direct_v4 | 0.95 | 1.000 | 19.95 | 20 | 20 | 20 | 0.99 | 0.017 |
| direct_v4 | 1.00 | 1.000 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 |
| checklist_v4_min | 0.50 | 0.980 | 16.39 | 14 | 20 | 20 | 0.50 | 0.403 |
| checklist_v4_min | 0.55 | 0.988 | 16.94 | 14 | 20 | 20 | 0.55 | 0.348 |
| checklist_v4_min | 0.60 | 0.992 | 17.33 | 15.5 | 20 | 20 | 0.61 | 0.279 |
| checklist_v4_min | 0.65 | 0.996 | 17.71 | 16 | 20 | 20 | 0.65 | 0.219 |
| checklist_v4_min | 0.70 | 1.000 | 18.06 | 17.5 | 20 | 20 | 0.72 | 0.180 |
| checklist_v4_min | 0.75 | 1.000 | 18.38 | 20 | 20 | 20 | 0.76 | 0.155 |
| checklist_v4_min | 0.80 | 1.000 | 18.82 | 20 | 20 | 20 | 0.80 | 0.120 |
| checklist_v4_min | 0.85 | 1.000 | 19.11 | 20 | 20 | 20 | 0.84 | 0.090 |
| checklist_v4_min | 0.90 | 1.000 | 19.43 | 20 | 20 | 20 | 0.89 | 0.064 |
| checklist_v4_min | 0.95 | 1.000 | 19.82 | 20 | 20 | 20 | 0.95 | 0.021 |
| checklist_v4_min | 1.00 | 1.000 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 |
| direct_pans | 0.50 | 0.534 | 4.39 | 1 | 2 | 6 | 0.04 | 0.936 |
| direct_pans | 0.55 | 0.594 | 5.65 | 1 | 3 | 8 | 0.09 | 0.901 |
| direct_pans | 0.60 | 0.649 | 6.58 | 1 | 4 | 10 | 0.13 | 0.845 |
| direct_pans | 0.65 | 0.685 | 7.34 | 1 | 5 | 11 | 0.16 | 0.777 |
| direct_pans | 0.70 | 0.741 | 8.59 | 1.5 | 6 | 16 | 0.22 | 0.704 |
| direct_pans | 0.75 | 0.813 | 10.02 | 2.5 | 9 | 20 | 0.29 | 0.592 |
| direct_pans | 0.80 | 0.876 | 11.92 | 4 | 13 | 20 | 0.41 | 0.502 |
| direct_pans | 0.85 | 0.924 | 13.48 | 6 | 20 | 20 | 0.51 | 0.352 |
| direct_pans | 0.90 | 0.976 | 15.58 | 10.5 | 20 | 20 | 0.63 | 0.215 |
| direct_pans | 0.95 | 0.992 | 18.04 | 20 | 20 | 20 | 0.84 | 0.090 |
| direct_pans | 1.00 | 1.000 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 |
| pans_min | 0.50 | 0.865 | 10.65 | 5.5 | 10 | 15.5 | 0.16 | 0.725 |
| pans_min | 0.55 | 0.888 | 11.17 | 6 | 11 | 16.5 | 0.20 | 0.687 |
| pans_min | 0.60 | 0.908 | 11.75 | 6 | 12 | 18 | 0.22 | 0.639 |
| pans_min | 0.65 | 0.932 | 12.40 | 7 | 13 | 20 | 0.27 | 0.592 |
| pans_min | 0.70 | 0.940 | 13.20 | 7.5 | 14 | 20 | 0.33 | 0.545 |
| pans_min | 0.75 | 0.952 | 13.93 | 9 | 15 | 20 | 0.37 | 0.446 |
| pans_min | 0.80 | 0.952 | 14.67 | 10 | 18 | 20 | 0.48 | 0.365 |
| pans_min | 0.85 | 0.976 | 15.83 | 12 | 20 | 20 | 0.57 | 0.296 |
| pans_min | 0.90 | 0.984 | 16.78 | 15 | 20 | 20 | 0.66 | 0.176 |
| pans_min | 0.95 | 1.000 | 17.80 | 20 | 20 | 20 | 0.78 | 0.099 |
| pans_min | 1.00 | 1.000 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 |

**Dev** (answerable 102, reader-wrong 99; `sweep_v22/dev_sweep.csv`)

| method | θ | acc | mean L | Q1 | median | Q3 | stop20 | firing |
|---|---|---|---|---|---|---|---|---|
| direct_v4 | 0.50 | 0.990 | 17.23 | 15.25 | 20 | 20 | 0.57 | 0.343 |
| direct_v4 | 0.55 | 0.990 | 17.64 | 16 | 20 | 20 | 0.62 | 0.313 |
| direct_v4 | 0.60 | 0.990 | 18.12 | 17 | 20 | 20 | 0.68 | 0.283 |
| direct_v4 | 0.65 | 0.990 | 18.59 | 19.25 | 20 | 20 | 0.75 | 0.212 |
| direct_v4 | 0.70 | 1.000 | 18.86 | 20 | 20 | 20 | 0.80 | 0.182 |
| direct_v4 | 0.75 | 1.000 | 19.11 | 20 | 20 | 20 | 0.85 | 0.101 |
| direct_v4 | 0.80 | 1.000 | 19.44 | 20 | 20 | 20 | 0.91 | 0.061 |
| direct_v4 | 0.85 | 1.000 | 19.63 | 20 | 20 | 20 | 0.94 | 0.040 |
| direct_v4 | 0.90 | 1.000 | 19.79 | 20 | 20 | 20 | 0.97 | 0.010 |
| direct_v4 | 0.95 | 1.000 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 |
| direct_v4 | 1.00 | 1.000 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 |
| checklist_v4_min | 0.50 | 1.000 | 16.22 | 12 | 19.5 | 20 | 0.50 | 0.354 |
| checklist_v4_min | 0.55 | 0.990 | 16.63 | 14 | 20 | 20 | 0.54 | 0.323 |
| checklist_v4_min | 0.60 | 0.990 | 16.93 | 14 | 20 | 20 | 0.57 | 0.273 |
| checklist_v4_min | 0.65 | 1.000 | 17.53 | 15 | 20 | 20 | 0.64 | 0.202 |
| checklist_v4_min | 0.70 | 1.000 | 17.97 | 16.25 | 20 | 20 | 0.69 | 0.182 |
| checklist_v4_min | 0.75 | 1.000 | 18.36 | 18.25 | 20 | 20 | 0.73 | 0.172 |
| checklist_v4_min | 0.80 | 1.000 | 18.66 | 20 | 20 | 20 | 0.76 | 0.131 |
| checklist_v4_min | 0.85 | 1.000 | 19.08 | 20 | 20 | 20 | 0.82 | 0.081 |
| checklist_v4_min | 0.90 | 1.000 | 19.52 | 20 | 20 | 20 | 0.88 | 0.061 |
| checklist_v4_min | 0.95 | 1.000 | 19.85 | 20 | 20 | 20 | 0.97 | 0.010 |
| checklist_v4_min | 1.00 | 1.000 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 |
| direct_pans | 0.50 | 0.520 | 4.40 | 1 | 2 | 5 | 0.07 | 0.990 |
| direct_pans | 0.55 | 0.569 | 5.02 | 1 | 2 | 7 | 0.08 | 0.939 |
| direct_pans | 0.60 | 0.657 | 6.29 | 1 | 3 | 9.75 | 0.13 | 0.869 |
| direct_pans | 0.65 | 0.706 | 7.09 | 2 | 4.5 | 10 | 0.16 | 0.808 |
| direct_pans | 0.70 | 0.784 | 8.60 | 2 | 6 | 17 | 0.24 | 0.737 |
| direct_pans | 0.75 | 0.882 | 10.49 | 4 | 9 | 20 | 0.29 | 0.646 |
| direct_pans | 0.80 | 0.931 | 12.35 | 5.25 | 12 | 20 | 0.39 | 0.515 |
| direct_pans | 0.85 | 0.971 | 14.25 | 8 | 18.5 | 20 | 0.49 | 0.424 |
| direct_pans | 0.90 | 1.000 | 16.76 | 17.75 | 20 | 20 | 0.75 | 0.222 |
| direct_pans | 0.95 | 1.000 | 18.74 | 20 | 20 | 20 | 0.90 | 0.071 |
| direct_pans | 1.00 | 1.000 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 |
| pans_min | 0.50 | 0.892 | 10.71 | 6 | 10 | 16 | 0.17 | 0.778 |
| pans_min | 0.55 | 0.931 | 11.35 | 6.25 | 11 | 16.75 | 0.21 | 0.747 |
| pans_min | 0.60 | 0.951 | 12.07 | 7 | 12 | 19 | 0.25 | 0.677 |
| pans_min | 0.65 | 0.941 | 12.64 | 7.25 | 13 | 20 | 0.30 | 0.606 |
| pans_min | 0.70 | 0.951 | 13.23 | 8 | 13.5 | 20 | 0.34 | 0.545 |
| pans_min | 0.75 | 0.961 | 14.08 | 9 | 15 | 20 | 0.41 | 0.444 |
| pans_min | 0.80 | 0.971 | 14.96 | 10 | 18.5 | 20 | 0.49 | 0.313 |
| pans_min | 0.85 | 0.990 | 15.98 | 12 | 20 | 20 | 0.60 | 0.253 |
| pans_min | 0.90 | 1.000 | 16.75 | 14 | 20 | 20 | 0.67 | 0.202 |
| pans_min | 0.95 | 1.000 | 17.69 | 20 | 20 | 20 | 0.75 | 0.152 |
| pans_min | 1.00 | 1.000 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 |

### Caveats
- **v1 arm skipped (user decision 2026-10-11).** No checklist_v1_min, checklist_v1_mean or verbalised Direct on test. The fair-v1 part of the main hypothesis is untested. The frozen file is unchanged; the evaluation drops those methods in memory and records "v1 arm skipped by user decision 2026-10-11" in `test_summary.json`, `ext_test_wrong/firing_meta.json` and `state.json`.
- **CV caveat (recorded before test).** A single dev θ loses about 0.01–0.02 accuracy out of fold. At the 0.95 target the v4 methods lost 0.011–0.027 on test, in line with it (0.002–0.032 over all four targets). The P_ans-only variants lost 0.047–0.055 at 0.95, more than the caveat.
- **Bootstrap CIs hold θ fixed.** They do not include θ-selection noise.
- **Reader-wrong set includes GQA label and referent ambiguity** (see the dev report §6). Firing therefore over-states false stops. The bias is shared by all methods, so paired comparisons are unaffected.
- **The fixed-level reference is degenerate at ≥ 0.95.** The answerable set keeps only L20-correct questions, so L20 is always correct (test fixed curve: L19 0.944, L20 1.000).
- **Granularity.** On 251 questions, 0.95 allows about 12 wrong stops.

### Reproduction

```bash
# environment: no resident model, qwen3.5:9b alone on GPU (v6_common.isolate_model / warm_up), then
uv run python -m milestone1.gqa_stop.cli detgate --config configs/gqa_stop_v22.json
# inference, repeated until complete (cache resumes); foreground chunks of <= 585 s
timeout 585 uv run python -m milestone1.gqa_stop.cli infer --split test --skip-v1arm --config configs/gqa_stop_v22.json
timeout 585 uv run python -m milestone1.gqa_stop.cli infer-test-wrong --skip-v1arm --config configs/gqa_stop_v22.json
uv run python -m milestone1.gqa_stop.cli audit-test --config configs/gqa_stop_v22.json
# one-shot frozen-θ evaluation (both refuse a second run)
uv run python -m milestone1.gqa_stop.cli evaluate-v22 --skip-v1arm --config configs/gqa_stop_v22.json
uv run python -m milestone1.gqa_stop.cli firing-test --skip-v1arm --config configs/gqa_stop_v22.json
# DESCRIPTIVE θ sweep (requires both evaluations first)
uv run python -m milestone1.gqa_stop.cli sweep-v22 --config configs/gqa_stop_v22.json
```

### Files
- **Answerable set.** `test_method_summary.csv`, `test_stops.csv`, `test_paired_comparisons.csv`, `test_attribute_breakdown.csv`, `test_stop_hist.csv`, `test_summary.json`, `level_table_test.csv`, `level_table_test_merged.csv`, `test_curves_exploratory.csv`, `figures/test_*.png`.
- **Reader-wrong set** (`ext_test_wrong/`). `level_table.csv`, `firing_per_question.csv`, `firing_summary.csv` (with firing CIs), `firing_paired.csv`, `firing_level_hist.csv`, `firing_breakdown.csv`, `firing_meta.json`.
- **Sweep** (`sweep_v22/`). `test_sweep.csv`, `dev_sweep.csv`, `sweep_meta.json`, `figures/sweep_*.png`.
- **Audit.** `logs/audit_test.json`, `logs/audit_test_flagged.txt` (empty).
- **Code.** `src/milestone1/gqa_stop/sweep_v22.py`, `audit_v22.py`, and the `--skip-v1arm` path in `stages_v22_test.py`, `firing_v23.py`, `logprob_v4.py`, `stages_model.py`, `context.py`, `cli.py`. Tests: `tests/test_gqa_stop_v22_test.py`.
