# gqa_stop v27 dev, model swap: qwen3.5:4b (plan §27.13, DEV ONLY)

Status: **dev inference, audit and evaluation done, 2026-10-15.** Team-lead request: replicate the v27 dev experiment with qwen3.5:4b, to test the user's hypothesis "with a weaker model, decomposition (P) may beat A1". Nothing was run on the test split. All numbers are **in-sample on dev**: θ* is chosen on the same 300 questions it is evaluated on.

Run: `configs/gqa_stop_v27_4b.json`, output `outputs/gqa_stop/visapp_gqa_v27_4b/`, data `data/processed/gqa_stop/visapp_gqa_v27_4b/`. Model qwen3.5:4b (digest 2a654d98e6fb), A13 clean decoding, think=false, top_logprobs 10. Prompts answer_only, planner_v3 and observer_v3 are unchanged (same hashes). No module under `src/` was changed. `visapp_gqa_v27/` (9b) was only read.

## 1. Bottom line

- **P does not beat A1 with 4b.** At each method's own θ*, P_min − A1 is −0.007 [−0.046, +0.030] (strict). At the same θ, the difference stays within ±0.022 for every θ from 0.50 to 0.95, and every CI includes 0. This matches 9b dev (−0.003) and 9b test (+0.020, n.s.).
- **The gap to B1 grows, and in B1's favour.** With 4b, B1 is the best method on dev. P_min − B1 is −0.042 [−0.101, +0.018], against +0.015 [−0.035, +0.068] with 9b dev. The CI includes 0, so the direction is not established, but nothing points toward P gaining with a weaker model.

| balanced acc (strict) at θ* | B1 | P_min | P_mean | A1 | A2 | A3 |
|---|---|---|---|---|---|---|
| 9b dev | 0.614 | **0.630** | 0.618 | **0.633** | 0.540 | 0.427 |
| 4b dev | **0.628** | 0.586 | 0.593 | 0.592 | 0.567 | 0.490 |

- **Why the weaker model does not help P.** With 4b, earlier (existence) items bind the min more often: 19% of rows, against 14% with 9b dev. They are resolved later and with lower P_ans. That makes P stop later and lose answerable accuracy, but it adds no veto. The last item still mirrors A1: its answer equals A1's answer on 96% of rows, and the scores correlate at 0.945.
- **The secondary view gives the same result.** Re-scored with the 9b labels, so that the population is identical to the 9b run, P_min − B1 is −0.029 [−0.082, +0.029] and P_min − A1 is −0.001 [−0.040, +0.035].

## 2. Setup and deviations

| item | value |
|---|---|
| config | `configs/gqa_stop_v27_4b.json` |
| output root | `outputs/gqa_stop/visapp_gqa_v27_4b/` |
| model | qwen3.5:4b, digest `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`; fallback is qwen3.5:4b itself |
| decoding | A13 (temperature 0, top_k 1, top_p 1, penalties neutral, mirostat 0), seed 20261008, think false, num_ctx 8192, same num_predict |
| dataset | same v27 extraction, re-run into the new data root |
| labels | re-screened by the 4b reader_lp at L20, as the protocol requires |

The config differs from `configs/gqa_stop_v27.json` in these places only:
- `run_id`;
- `paths.data_root`, `paths.selection_root` and `paths.output_root`, so the 9b screening and manifests are never overwritten;
- `model.name` and `model.fallback_name`, with a wording change in `decoding_note`;
- a new `model_swap` block. It carries `trailing_period_roles: [planner_v3, observer_v3]`, the same tolerance as `configs/gqa_stop_v22_4b.json` (plan §26.1). The observer_v3 role covers both observer_v3_lp and direct_resolved, because both use the observer_v3 prompt.

**Dataset identity.** `v27-extract` was run fresh with the 4b config. The outputs are byte-identical to the 9b run.

| file | 9b sha256 (prefix) | 4b sha256 (prefix) |
|---|---|---|
| `questions_dev.jsonl` | 73817d8a1e9d1bc9 | 73817d8a1e9d1bc9 |
| `gt_dev.json` | 01213080b2de17c8 | 01213080b2de17c8 |
| `dev_source_image_sha256.json` | d47a075d091b8ed6 | d47a075d091b8ed6 |

- The 300 dev question ids are the same and in the same order.
- `dataset_v27.json` matches on source sha256, exclusion counts, eligible and selected breakdowns, and manifest sha256.
- The extraction does not depend on the model. It reads only the source files, the filter, the image hashes and the seed.
- Extraction also writes the test manifest and GT, as in the 9b run. No model call and no evaluation touched test.

**Deviation: none beyond the pre-agreed tolerance.** The trailing-period strip was needed on every valid plan and never elsewhere (§3).

## 3. Run facts

- **Environment.** RTX 5080, driver 580.178.04, ollama 0.30.4. No model was resident before the run. After `isolate_model` and warm-up, only qwen3.5:4b was resident: 100% GPU, context 8192, ollama log "CLIP using CUDA0 backend". It stayed the only resident model at every check between chunks.
- **Determinism gate.** Passed: 120 argmax calls, 801 logprob positions, 0 violations. Both repeat pairs, planner_v3 text-only and reader_lp at L20, were byte-identical in content and logprobs.
- **reader_lp identity check.** 120 of 120 pairs were raw byte-identical. reader_lp is therefore the common scorer and B1, from one call.
- **Calls and time.** Inference took 23,582 new calls in 3.52 h at 0.537 s per call, in 23 logged foreground chunks of ≤ 585 s, between 21:20Z and 00:57Z. 4b is faster than 9b here (0.61 s/call) but slower than v22-4b (0.455 s/call), because the observer's JSON reply is longer.
- **One interrupted chunk.** One chunk was killed with exit 137 about two minutes in. It coincided with a harness session change; there was no OOM and no leftover process. The next chunk resumed from the cache. Records are written atomically, and the audit found nothing abnormal.
- **Audit** (`logs/audit_dev.json`). 0 records flagged, so nothing was quarantined.

| audit item | value |
|---|---|
| records | 24,300 |
| non-generation time > 5 s | 0 (max 0.43 s) |
| argmax violations | 0 of 287,776 positions |

- **Records and validity.**

| role | records | valid | trailing "." stripped |
|---|---|---|---|
| reader (identity only) | 120 | 120 | 0 |
| reader_lp | 6,000 | 6,000 | 0 |
| plans_v3 | 300 | 287 | 287 |
| observer_v3_lp | 11,880 | 11,880 | 0 |
| direct_resolved | 6,000 | 6,000 | 0 |

- **Planner.**

| model | initial valid | repaired | failed | 2 items | 3 items | last item names attribute |
|---|---|---|---|---|---|---|
| 4b dev | 287 | 0 | 13 | 267 | 20 | 286 |
| 9b dev | 299 | 1 | 0 | 288 | 12 | 300 |
| 9b test | 885 | 1 | 0 | 846 | 40 | 883 |

- **The 13 planner failures are genuinely malformed.** Ten lack the final closing brace, and three end with a stray backtick. The schema repair returned the same reply each time. All are relational plans with a third item that depends on c2 or c3. On these questions P scores 0 at every level and never stops.
  - 9 of the 13 are unanswerable, so the failures help P under strict scoring.
  - On the 287 questions with a valid plan, at the same θ*, P_min − B1 is −0.050 [−0.108, +0.008] and P_min − A1 is −0.011 [−0.047, +0.025]. The failures do not explain P's deficit.

## 4. Labels (screening = 4b reader_lp at L20, 0 invalid)

| attribute | answerable | unanswerable |
|---|---|---|
| color | 139 | 111 |
| material | 36 | 8 |
| shape | 3 | 3 |
| **total** | **178** | **122** |

The 9b labels were 175 answerable and 125 unanswerable. Overlap on the same 300 questions:

| | 9b answerable | 9b unanswerable |
|---|---|---|
| **4b answerable** | 160 | 18 |
| **4b unanswerable** | 15 | 107 |

- **The labels agree on 89% of questions.** On 82 of the 107 questions both models get wrong at L20, they give the same wrong answer. This repeats the §25 finding that much of the unanswerable side is GQA annotation ambiguity.
- **4b reader accuracy by level.**

| level | 1 | 10 | 20 |
|---|---|---|---|
| all | 0.323 | 0.490 | 0.593 |
| answerable | 0.500 | 0.781 | 1.000 |
| unanswerable | 0.066 | 0.066 | 0.000 |

- **The labels are not clean, as with 9b.** 32 of 122 unanswerable questions are reader-correct at some level before L20 (9b: 28 of 125). 74 of 178 answerable questions are correct at every level (9b: 43%).

## 5. θ* (strict rule: argmax dev balanced accuracy, ties go to the smaller θ)

| method | B1 | P_min | P_mean | A1 | A2 | A3 | P'_min | P'_mean | A1' |
|---|---|---|---|---|---|---|---|---|---|
| 4b, strict rule | 0.90 | 0.70 | 0.90 | 0.70 | 0.95 | — | 0.70 | 0.90 | 0.70 |
| 4b, unified rule | 0.90 | 0.60 | 0.90 | 0.70 | 0.95 | — | 0.60 | 0.90 | 0.70 |
| 9b, strict rule | 0.90 | 0.80 | 0.90 | 0.75 | 0.95 | — | 0.80 | 0.90 | 0.75 |

The θ* of P_min and A1 moves down by 0.05–0.10 with 4b, because 4b's observer P_ans is lower. B1, P_mean and A2 keep the 9b values. Unlike 9b, the strict and unified rules disagree for P_min and P'_min.

## 6. Results at θ* (strict θ*)

CIs are image-cluster bootstraps (2000 draws, seed 20261008) with θ held fixed, so selection error is not included. "Never" stops count as L20 in the mean levels.

| method | θ* | balanced strict [95% CI] | balanced unified [95% CI] | answerable acc [95% CI] | unanswerable correct [95% CI] | pooled strict | mean L (all) | mean L (answerable) |
|---|---|---|---|---|---|---|---|---|
| **B1** | 0.90 | **0.628 [0.575, 0.680]** | **0.632 [0.579, 0.684]** | 0.640 [0.573, 0.706] | 0.615 [0.530, 0.703] | 0.630 | 12.30 | 10.63 |
| P_min | 0.70 | 0.586 [0.527, 0.640] | 0.594 [0.536, 0.648] | 0.573 [0.502, 0.647] | 0.598 [0.509, 0.685] | 0.583 | 14.78 | 13.99 |
| P_mean | 0.90 | 0.593 [0.541, 0.641] | 0.593 [0.541, 0.641] | 0.416 [0.341, 0.491] | 0.770 [0.694, 0.843] | 0.560 | 16.95 | 16.24 |
| A1 | 0.70 | 0.592 [0.535, 0.646] | 0.600 [0.544, 0.654] | 0.652 [0.579, 0.725] | 0.533 [0.446, 0.619] | 0.603 | 13.33 | 12.30 |
| A2 | 0.95 | 0.567 [0.529, 0.604] | 0.567 [0.529, 0.604] | 0.208 [0.150, 0.269] | 0.926 [0.876, 0.968] | 0.500 | 18.57 | 18.07 |
| A3 | — | 0.490 [0.441, 0.540] | 0.518 [0.467, 0.570] | 0.742 [0.679, 0.803] | 0.238 [0.167, 0.316] | 0.537 | 9.86 | 9.03 |
| P'_min | 0.70 | 0.586 [0.527, 0.640] | 0.594 [0.536, 0.648] | 0.573 | 0.598 | 0.583 | 14.78 | 13.99 |
| P'_mean | 0.90 | 0.593 [0.541, 0.641] | 0.593 [0.541, 0.641] | 0.416 | 0.770 | 0.560 | 16.95 | 16.24 |
| A1' | 0.70 | 0.592 [0.535, 0.646] | 0.609 [0.553, 0.662] | 0.652 | 0.533 | 0.603 | 13.33 | 12.30 |

The pooled accuracy is over 300 questions (178 answerable / 122 unanswerable = 59% / 41%).

- **Unified-rule θ\*.** Only P_min and P'_min change, to 0.60. P_min then scores 0.586 strict and 0.598 unified. Its answerable accuracy rises to 0.663 and its unanswerable correct rate falls to 0.508.
- **The post-hoc methods equal their bases under strict scoring.** P'_min = P_min and A1' = A1 under strict scoring with 4b. A1' gains +0.008 under unified scoring. With 9b dev, P'_min was −0.009 below P_min.
- **Stopping detail at θ\*.**

| method | answerable never-stop | answerable stopped wrong | answerable stop level Q1 / median / Q3 |
|---|---|---|---|
| B1 | 0.287 | 13 | 3 / 10 / 20 |
| P_min | 0.399 | 5 | 8 / 18 / 20 |
| P_mean | 0.584 | 0 | 13 / 20 / 20 |
| A1 | 0.309 | 7 | 6 / 13 / 20 |
| A2 | 0.787 | 1 | 20 / 20 / 20 |
| A3 | 0.096 | 29 | 4 / 7.5 / 14 |

- **Firing on unanswerable questions at θ\*** (122 questions, counts by stop level).

| method | L1–5 | L6–10 | L11–15 | L16–19 | L20 | never |
|---|---|---|---|---|---|---|
| B1 | 24 | 12 | 8 | 2 | 1 | 75 |
| P_min | 9 | 19 | 13 | 5 | 3 | 73 |
| P_mean | 4 | 10 | 8 | 4 | 2 | 94 |
| A1 | 17 | 18 | 15 | 5 | 2 | 65 |
| A2 | 0 | 5 | 3 | 1 | 0 | 113 |
| A3 | 33 | 30 | 18 | 8 | 4 | 29 |

B1 fires early less often with 4b than with 9b: 24 of 122 at L1–5, against 38 of 125. That is where B1 gains most of its unanswerable accuracy (0.615, against 0.560 with 9b).

- **Per attribute at θ\*** (descriptive; shape is 3 + 3 questions, material unanswerable is 8).

| attribute | method | answerable acc | unanswerable correct | balanced | mean L (answerable) |
|---|---|---|---|---|---|
| color | B1 | 0.619 | 0.631 | 0.625 | 10.56 |
| color | P_min | 0.583 | 0.604 | 0.593 | 13.46 |
| color | A1 | 0.669 | 0.541 | 0.605 | 11.44 |
| material | B1 | 0.722 | 0.625 | 0.674 | 11.00 |
| material | P_min | 0.500 | 0.625 | 0.562 | 16.44 |
| material | A1 | 0.556 | 0.625 | 0.590 | 16.03 |

## 7. Paired comparisons (image-cluster paired bootstrap, 2000 draws, seed 20261008)

### 7.1 Each method at its own θ* (strict θ*)
From `eval/dev/paired_bootstrap_at_theta_star.csv` and `eval_posthoc/dev/paired_bootstrap_at_theta_star.csv`.

| quantity | P_min − B1 | P_min − A1 | P'_min − A1' | A1 − B1 |
|---|---|---|---|---|
| balanced (strict) | −0.042 [−0.101, +0.018] | −0.007 [−0.046, +0.030] | −0.007 [−0.046, +0.030] | −0.035 [−0.091, +0.018] |
| balanced (unified) | −0.038 [−0.094, +0.019] | −0.007 [−0.046, +0.030] | −0.015 [−0.054, +0.023] | −0.031 [−0.081, +0.020] |
| answerable acc | −0.067 [−0.130, +0.000] | −0.079 [−0.131, −0.029] | −0.079 [−0.131, −0.029] | +0.011 [−0.045, +0.068] |
| unanswerable correct (strict) | −0.016 [−0.117, +0.078] | +0.066 [+0.014, +0.122] | +0.066 [+0.014, +0.122] | −0.082 [−0.176, +0.008] |
| pooled (strict) | −0.047 [−0.101, +0.010] | −0.020 [−0.060, +0.017] | −0.020 [−0.060, +0.017] | −0.027 [−0.077, +0.024] |
| mean level, all | +2.48 [+1.65, +3.28] | +1.45 [+1.00, +1.94] | +1.45 [+1.00, +1.94] | +1.02 [+0.30, +1.73] |
| mean level, answerable | +3.37 [+2.43, +4.35] | +1.70 [+1.04, +2.44] | +1.70 [+1.04, +2.44] | +1.67 [+0.84, +2.48] |

The same pairs with 9b, balanced strict:

| pair | 9b dev | 9b test (§27.11, §27.12) | 4b dev |
|---|---|---|---|
| P_min − B1 | +0.015 [−0.035, +0.068] | +0.036 [+0.002, +0.067] | −0.042 [−0.101, +0.018] |
| P_min − A1 | −0.003 [−0.041, +0.034] | +0.020 [−0.001, +0.042] | −0.007 [−0.046, +0.030] |
| P'_min − A1' | −0.009 [−0.048, +0.029] | +0.021 [+0.001, +0.043] | −0.007 [−0.046, +0.030] |
| A1 − B1 | +0.019 [−0.029, +0.072] | — | −0.035 [−0.091, +0.018] |

- **P_min vs A1 is the same trade as with 9b.** P_min stops 1.5 levels later and is correct on more unanswerable questions (+0.066). It loses as much answerable accuracy (−0.079). The balanced difference is 0.
- **P_min vs B1 changed sign.** With 9b, P bought unanswerable accuracy from B1 (+0.088). With 4b, B1 abstains about as well as P (−0.016) and is more accurate on answerable questions (+0.067).

### 7.2 Same θ for both methods (`eval/dev/paired_bootstrap.csv`, balanced strict)

| θ | 0.50 | 0.60 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|---|
| P_min − A1 | +0.013 [−0.025, +0.054] | +0.018 [−0.023, +0.061] | −0.007 [−0.046, +0.030] | −0.008 [−0.043, +0.027] | −0.018 [−0.051, +0.015] | −0.002 [−0.038, +0.031] | −0.004 [−0.041, +0.034] | −0.022 [−0.052, +0.009] |
| P_min − B1 | +0.212 [+0.157, +0.269] | +0.207 [+0.152, +0.267] | +0.113 [+0.058, +0.169] | +0.063 [+0.007, +0.121] | +0.020 [−0.037, +0.079] | −0.005 [−0.057, +0.050] | −0.062 [−0.119, −0.008] | −0.053 [−0.097, −0.005] |

As with 9b, same-θ differences against B1 mostly reflect the scale of reader P_ans, not the method. P_min and A1 do not differ at any θ.

## 8. θ sweep (strict and unified; `eval_posthoc/dev/sweep.csv`)

Balanced accuracy, strict (bold = θ*):

| θ | B1 | P_min | P_mean | A1 | A2 | P'_min | A1' |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.250 | 0.250 | 0.250 | 0.250 | 0.250 | 0.067 | 0.067 |
| 0.05 | 0.250 | 0.490 | 0.371 | 0.475 | 0.278 | 0.481 | 0.472 |
| 0.50 | 0.324 | 0.536 | 0.501 | 0.522 | 0.311 | 0.533 | 0.522 |
| 0.60 | 0.379 | 0.586 | 0.512 | 0.567 | 0.334 | 0.586 | 0.567 |
| 0.65 | 0.429 | 0.572 | 0.529 | 0.571 | 0.348 | 0.572 | 0.571 |
| 0.70 | 0.473 | **0.586** | 0.523 | **0.592** | 0.401 | **0.586** | **0.592** |
| 0.75 | 0.514 | 0.577 | 0.557 | 0.584 | 0.442 | 0.577 | 0.584 |
| 0.80 | 0.538 | 0.558 | 0.561 | 0.576 | 0.480 | 0.558 | 0.576 |
| 0.85 | 0.583 | 0.578 | 0.582 | 0.580 | 0.548 | 0.578 | 0.580 |
| 0.90 | **0.628** | 0.566 | **0.593** | 0.570 | 0.549 | 0.566 | 0.570 |
| 0.95 | 0.620 | 0.567 | 0.568 | 0.589 | **0.567** | 0.567 | 0.589 |
| 1.00 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |

Balanced accuracy, unified:

| θ | B1 | P_min | P_mean | A1 | A2 | P'_min | A1' |
|---|---|---|---|---|---|---|---|
| 0.50 | 0.344 | 0.556 | 0.525 | 0.543 | 0.348 | 0.553 | 0.551 |
| 0.60 | 0.403 | 0.598 | 0.549 | 0.584 | 0.359 | 0.598 | 0.588 |
| 0.70 | 0.485 | 0.594 | 0.544 | 0.600 | 0.426 | 0.594 | 0.609 |
| 0.75 | 0.530 | 0.585 | 0.577 | 0.593 | 0.466 | 0.585 | 0.601 |
| 0.80 | 0.550 | 0.566 | 0.569 | 0.584 | 0.500 | 0.566 | 0.588 |
| 0.85 | 0.591 | 0.582 | 0.590 | 0.585 | 0.553 | 0.582 | 0.585 |
| 0.90 | 0.632 | 0.566 | 0.593 | 0.570 | 0.549 | 0.566 | 0.570 |
| 0.95 | 0.624 | 0.567 | 0.568 | 0.589 | 0.567 | 0.567 | 0.589 |

Answerable accuracy / unanswerable correct (strict) / mean stop level (all, never = 20):

| θ | B1 | P_min | A1 | P_mean | A2 |
|---|---|---|---|---|---|
| 0.50 | 0.607 / 0.041 / 2.5 | 0.719 / 0.352 / 11.6 | 0.742 / 0.303 / 10.5 | 0.764 / 0.238 / 10.0 | 0.539 / 0.082 / 2.5 |
| 0.60 | 0.618 / 0.139 / 3.8 | 0.663 / 0.508 / 13.4 | 0.725 / 0.410 / 11.9 | 0.753 / 0.270 / 11.0 | 0.562 / 0.107 / 4.5 |
| 0.70 | 0.708 / 0.238 / 5.9 | 0.573 / 0.598 / 14.8 | 0.652 / 0.533 / 13.3 | 0.685 / 0.361 / 12.7 | 0.556 / 0.246 / 8.3 |
| 0.75 | 0.725 / 0.303 / 7.1 | 0.506 / 0.648 / 15.5 | 0.579 / 0.590 / 14.1 | 0.646 / 0.467 / 13.7 | 0.539 / 0.344 / 10.6 |
| 0.80 | 0.691 / 0.385 / 8.9 | 0.444 / 0.672 / 16.2 | 0.528 / 0.623 / 14.8 | 0.556 / 0.566 / 14.8 | 0.500 / 0.459 / 13.0 |
| 0.85 | 0.674 / 0.492 / 10.5 | 0.410 / 0.746 / 16.9 | 0.489 / 0.672 / 15.4 | 0.500 / 0.664 / 15.8 | 0.449 / 0.648 / 15.4 |
| 0.90 | 0.640 / 0.615 / 12.3 | 0.320 / 0.811 / 17.7 | 0.410 / 0.730 / 16.4 | 0.416 / 0.770 / 17.0 | 0.320 / 0.779 / 17.1 |
| 0.95 | 0.478 / 0.762 / 14.7 | 0.208 / 0.926 / 18.6 | 0.309 / 0.869 / 17.4 | 0.275 / 0.861 / 18.1 | 0.208 / 0.926 / 18.6 |

A3 (no θ): balanced 0.490 strict and 0.518 unified, answerable 0.742, unanswerable 0.238, mean level 9.86.

- **P_min, P_mean and A1 peak lower with 4b; B1, A2 and A3 score higher.** The P_min and A1 curves are flat over θ 0.60–0.75 (within 0.02 of their peak). B1 peaks at 0.90–0.95.
- **For P_min, θ 0.05–0.20 equals A3** (0.490), as with 9b. Gated item scores are non-zero exactly when the item is resolved.

## 9. Secondary view: 4b level table scored with the 9b labels

`scripts/gqa_v27_4b_analysis.py`, outputs `analysis_4b/secondary_9b_labels_*.csv`. The population is identical to the 9b run (175 answerable / 125 unanswerable). The reader correctness, signals and stops are 4b's. The θ is the 4b strict θ*. This view is descriptive.

| method | θ (4b θ*) | balanced strict [95% CI] | balanced unified [95% CI] | answerable acc | unanswerable correct (strict) | mean L (answerable) |
|---|---|---|---|---|---|---|
| **B1** | 0.90 | **0.604 [0.549, 0.656]** | **0.616 [0.562, 0.667]** | 0.640 | 0.568 | 11.06 |
| P_min | 0.70 | 0.575 [0.516, 0.632] | 0.595 [0.539, 0.650] | 0.566 | 0.584 | 14.00 |
| P_mean | 0.90 | 0.590 [0.538, 0.641] | 0.598 [0.547, 0.649] | 0.411 | 0.768 | 16.22 |
| A1 | 0.70 | 0.576 [0.519, 0.634] | 0.600 [0.544, 0.657] | 0.640 | 0.512 | 12.36 |
| A2 | 0.95 | 0.570 [0.532, 0.609] | 0.570 [0.532, 0.609] | 0.211 | 0.928 | 18.03 |
| A3 | — | 0.489 [0.441, 0.540] | 0.529 [0.476, 0.581] | 0.737 | 0.240 | 8.98 |
| P'_min | 0.70 | 0.575 | 0.595 | 0.566 | 0.584 | 14.00 |
| P'_mean | 0.90 | 0.590 | 0.598 | 0.411 | 0.768 | 16.22 |
| A1' | 0.70 | 0.576 | 0.608 | 0.640 | 0.512 | 12.36 |

| pair (balanced strict, at 4b θ*) | estimate [95% CI] |
|---|---|
| P_min − B1 | −0.029 [−0.082, +0.029] |
| P_min − A1 | −0.001 [−0.040, +0.035] |
| P'_min − A1' | −0.001 [−0.040, +0.035] |
| A1 − B1 | −0.028 [−0.080, +0.028] |

Balanced strict sweep with the 9b labels:

| θ | B1 | P_min | P_mean | A1 | A2 |
|---|---|---|---|---|---|
| 0.50 | 0.326 | 0.541 | 0.493 | 0.518 | 0.307 |
| 0.60 | 0.378 | 0.556 | 0.514 | 0.553 | 0.314 |
| 0.70 | 0.471 | 0.575 | 0.532 | 0.576 | 0.396 |
| 0.75 | 0.515 | 0.582 | 0.563 | 0.587 | 0.457 |
| 0.80 | 0.534 | 0.563 | 0.568 | 0.582 | 0.505 |
| 0.85 | 0.561 | 0.578 | 0.587 | 0.586 | 0.562 |
| 0.90 | 0.604 | 0.571 | 0.590 | 0.574 | 0.555 |
| 0.95 | 0.611 | 0.570 | 0.572 | 0.593 | 0.570 |

- **Balanced accuracy changes by −0.024 to +0.003 under the 9b labels.** 33 questions switch side: 15 answerable-by-9b questions that 4b gets wrong at L20 now count as answerable, and 18 that 4b gets right now count as unanswerable. B1 loses most (−0.024). The order and the conclusions do not change.
- **Re-selecting θ under the 9b labels**, which is descriptive only, gives B1 0.611 at 0.95, A1 0.593 at 0.95, P_min 0.582 at 0.75 and P_mean 0.590 at 0.90.

## 10. Mechanism: does decomposition diverge from A1 with a weaker model?

Definitions follow `scripts/gqa_probe_multi_rel_analyze.py`. They reproduce the 9b test figures exactly (0.933 / 0.887 / 0.944).
- **own_last.** The last item's P_ans if that item is valid, resolved and its answer is not "unknown"; otherwise 0.
- **Last binds.** pans_min = own_last.
- **Existence items.** Every item before the last one.

The 4b rows exclude the 13 planner-failure questions, where every score is 0.

| statistic per (question, level) | 9b dev | 9b test | 4b dev |
|---|---|---|---|
| rows | 6,000 | 17,720 | 5,740 |
| corr(pans_min, dres_score) | 0.943 | 0.933 | 0.945 |
| last item binds the min | 0.855 | 0.887 | 0.814 |
| earlier item binds the min | 0.145 | 0.113 | 0.186 |
| last binds, own_last > 0 | 0.766 | 0.791 | 0.667 |
| last-item answer = A1 answer | 0.944 | 0.944 | 0.959 |
| last-item resolved-ok = A1 resolved-ok | 0.964 | 0.959 | 0.966 |
| A3 = A1 resolved-ok | 0.954 | 0.950 | 0.958 |
| mean pans_min / mean A1 score | 0.443 / 0.472 | 0.372 / 0.402 | 0.377 / 0.412 |

Stop decisions at θ* (per question, strict):

| quantity | 9b dev | 9b test | 4b dev |
|---|---|---|---|
| θ* P_min / A1 | 0.80 / 0.75 | 0.80 / 0.75 | 0.70 / 0.70 |
| P_min and A1 stop at a different level | 0.427 | 0.397 | 0.333 |
| P_min right, A1 wrong | 18 | 63 | 14 |
| A1 right, P_min wrong | 24 | 50 | 20 |

Existence items by level (resolved-ok rate / mean raw P_ans / share answering "yes"):

| level | 9b dev | 9b test | 4b dev |
|---|---|---|---|
| 1 | 0.397 / 0.721 / 0.356 | 0.328 / 0.722 / 0.272 | 0.267 / 0.650 / 0.241 |
| 5 | 0.792 / 0.770 / 0.692 | 0.725 / 0.749 / 0.571 | 0.671 / 0.659 / 0.599 |
| 10 | 0.949 / 0.859 / 0.830 | 0.931 / 0.841 / 0.771 | 0.925 / 0.773 / 0.824 |
| 15 | 0.984 / 0.901 / 0.894 | 0.973 / 0.893 / 0.859 | 0.990 / 0.835 / 0.886 |
| 20 | 0.990 / 0.925 / 0.913 | 0.985 / 0.919 / 0.883 | 0.990 / 0.884 / 0.909 |

Last item binds the min, by level:

| level | 1 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| 9b dev | 0.920 | 0.897 | 0.853 | 0.820 | 0.840 |
| 9b test | 0.937 | 0.909 | 0.886 | 0.870 | 0.853 |
| 4b dev | 0.885 | 0.836 | 0.798 | 0.756 | 0.815 |

Answer accuracy against GT by level. The observer's "unknown" counts as wrong. The last item and A1 are the observer.

| set | label | reader L1 / L10 / L20 | last item L1 / L10 / L20 | A1 L1 / L10 / L20 |
|---|---|---|---|---|
| 4b dev | all | 0.323 / 0.490 / 0.593 | 0.083 / 0.320 / 0.500 | 0.083 / 0.330 / 0.520 |
| 4b dev | answerable | 0.500 / 0.781 / 1.000 | 0.135 / 0.522 / 0.820 | 0.135 / 0.534 / 0.854 |
| 4b dev | unanswerable | 0.066 / 0.066 / 0.000 | 0.008 / 0.025 / 0.033 | 0.008 / 0.033 / 0.033 |
| 9b dev | answerable | 0.543 / 0.863 / 1.000 | 0.149 / 0.634 / 0.811 | 0.154 / 0.640 / 0.829 |
| 9b dev | unanswerable | 0.072 / 0.056 / 0.000 | 0.008 / 0.024 / 0.048 | 0.032 / 0.024 / 0.024 |
| 9b test | answerable | 0.591 / 0.857 / 1.000 | 0.109 / 0.550 / 0.827 | 0.109 / 0.557 / 0.839 |
| 9b test | unanswerable | 0.086 / 0.105 / 0.000 | 0.007 / 0.043 / 0.043 | 0.007 / 0.043 / 0.048 |

Mean signals by label (4b dev; 9b dev in parentheses):

| signal | L1 answerable | L1 unanswerable | L20 answerable | L20 unanswerable |
|---|---|---|---|---|
| reader P_ans | 0.66 (0.67) | 0.60 (0.66) | 0.84 (0.86) | 0.68 (0.71) |
| pans_min | 0.08 | 0.04 | 0.65 (0.73) | 0.49 (0.56) |
| dres_score | 0.13 | 0.10 | 0.70 (0.75) | 0.54 (0.58) |

**Reading.**
- **Decomposition diverges a little more with 4b, through the existence items.** 4b's existence items are resolved later (L5: 0.67 vs 0.79) and with lower P_ans (L10: 0.77 vs 0.86). So an earlier item holds the min on 19% of rows (9b dev 14%, 9b test 11%). When the last item is positive, it binds only 67% of the time (9b 77–79%).
- **That divergence only delays stopping.** The existence items answer "yes" as often as with 9b (L10: 0.82 vs 0.83). A resolved "no" passes the gate anyway, so they cannot veto. Their lower confidence pushes P_min below A1, so P stops later. That buys unanswerable accuracy (+0.066 vs A1) and costs the same amount of answerable accuracy (−0.079).
- **The last item still is A1.** Its answer equals A1's answer on 96% of rows, its resolved flag agrees on 97%, and the scores correlate at 0.945. The planner rewrites the question into "Is there X?" plus the original question. The observer answers that last item like the original question.
- **The discordant questions favour A1 slightly with 4b (14 vs 20), as on 9b dev (18 vs 24).** On 9b test they favoured P (63 vs 50).
- **The weaker model's reader P_ans separates the two sides a little better at L1** (0.66 vs 0.60, against 0.67 vs 0.66 with 9b). This is the main reason B1 improves with 4b: it fires less at L1–5 on unanswerable questions.
- **Observer answers are no better than the reader's.** Among answerable questions at L20, the 4b observer is right on 82–85%, and the reader is right on all of them by definition. On the unanswerable side, the observer is right on only 3%.

## 11. Answer to the team lead's question

- **Does P beat A1 with 4b? No.** P_min − A1 is −0.007 [−0.046, +0.030] at θ*. It is within ±0.022 at every same θ, and the post-hoc variants agree (P'_min − A1' −0.007). The result is the same as with 9b: the resolved / unknown gate does the work, and decomposition adds nothing measurable.
- **Is the gap to B1 larger or smaller than with 9b? It is larger, and it points the other way.** P_min − B1 moves from +0.015 (9b dev) and +0.036 (9b test) to −0.042 [−0.101, +0.018] (4b dev). Two things drive it:
  - B1 improves with 4b (0.614 → 0.628), because the 4b reader is less confident at far levels on unanswerable questions.
  - P_min falls (0.630 → 0.586), because 4b's observer and existence items are less confident, so P stops later and loses answerable accuracy.
- **With the 9b labels** the picture is the same: P_min − B1 is −0.029 and P_min − A1 is −0.001.
- **The hypothesis is not supported on dev.** A weaker model makes the decomposition diverge more from A1, but in the direction of more delay, not of better abstention.

## 12. Caveats

- **Dev only, in-sample θ\*.** θ* is chosen on the evaluated questions. The CIs do not include θ-selection error. No test number exists for 4b.
- **n = 300.** Every CI for a balanced-accuracy difference is about ±0.04 to ±0.06 wide. A real difference of 0.02–0.03 cannot be resolved.
- **Labels are model-dependent.** The protocol defines answerable as "the model under test is right at L20". The two label sets disagree on 33 of 300 questions. The secondary view (§9) controls for this and agrees.
- **B1 has a self-consistency advantage.** The same reader_lp call defines the label at L20 and supplies B1's score (§27.10). That advantage exists with 9b too. It does not explain the change between models.
- **Planner failures (13, 4.3%)** are a 4b-specific source of noise. They help P under strict scoring, and excluding them leaves P slightly worse (§3).
- **Trailing-period tolerance.** Every valid 4b plan needed it. Without it, every plan would fail and P would never stop.

## 13. Files and commands

```
uv run python -m milestone1.gqa_stop.cli preflight    --config configs/gqa_stop_v27_4b.json
uv run python -m milestone1.gqa_stop.cli v27-extract  --config configs/gqa_stop_v27_4b.json
uv run python -m milestone1.gqa_stop.cli v27-envcheck --config configs/gqa_stop_v27_4b.json
timeout 585 uv run python -m milestone1.gqa_stop.cli v27-detgate  --config configs/gqa_stop_v27_4b.json
timeout 585 uv run python -m milestone1.gqa_stop.cli v27-identity --config configs/gqa_stop_v27_4b.json
timeout 585 uv run python -m milestone1.gqa_stop.cli v27-screen --split dev --config configs/gqa_stop_v27_4b.json
timeout 585 uv run python -m milestone1.gqa_stop.cli v27-infer --split dev --max-seconds 560 --config configs/gqa_stop_v27_4b.json   # x24 (1 killed, resumed)
uv run python -m milestone1.gqa_stop.cli v27-audit    --split dev --config configs/gqa_stop_v27_4b.json
uv run python -m milestone1.gqa_stop.cli v27-table    --split dev --config configs/gqa_stop_v27_4b.json
uv run python -m milestone1.gqa_stop.cli v27-evaluate --split dev --config configs/gqa_stop_v27_4b.json
uv run python -m milestone1.gqa_stop.cli v27-evaluate --split dev --posthoc --config configs/gqa_stop_v27_4b.json
uv run python scripts/gqa_v27_4b_analysis.py
```

- **Outputs** (`outputs/gqa_stop/visapp_gqa_v27_4b/`):
  - `level_table_dev.csv`, `dataset_v27.json`, `identity_check.json`, `preflight_determinism.json`;
  - `eval/dev/` (confirmatory six methods) and `eval_posthoc/dev/` (nine methods, both θ* rules);
  - `figures/dev/` and `eval_posthoc/dev/figures/`;
  - `analysis_4b/`: `summary.json` (labels, agreement, discordance, planner, trailing counts), `secondary_9b_labels_*.csv`, `existence_by_level_*.csv`, `binds_by_level_*.csv`, `accuracy_by_level_*.csv`;
  - `logs/`: envcheck, audit, infer progress.
- **New files.** `configs/gqa_stop_v27_4b.json` and `scripts/gqa_v27_4b_analysis.py`, which reads caches and tables only.
