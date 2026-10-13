# gqa_stop v27 dev (plan §27): mixed answerable / unanswerable stopping, robot-approach framing

Status: **dev evaluated, 2026-10-13.** Pre-registration is plan §27 (written before any v27 call). θ* per method is fixed in `outputs/gqa_stop/visapp_gqa_v27/eval/dev/theta_star.json` before any test call. Test inference follows; test evaluation waits for the team-lead go.

Run: `configs/gqa_stop_v27.json`, output `outputs/gqa_stop/visapp_gqa_v27/`, data `data/processed/gqa_stop/visapp_gqa_v27/`. Model qwen3.5:9b (digest 6488c96fa5fa), A13 clean decoding, think=false, top_logprobs 10. Prompts answer_only, planner_v3 and observer_v3 are unchanged; no new prompt text.

All numbers below are **in-sample on dev**: θ* is chosen on the same 300 questions it is evaluated on.

## 1. Bottom line

- **No method separates answerable from unanswerable questions well.** The trivial policies "never stop" and "always stop at L20" both score a balanced accuracy of 0.500. The best methods reach only 0.61–0.63.
- **P (decomposition, pans_min) ≈ B1 ≈ A1 at their own θ*.**

| comparison at θ* | balanced accuracy difference [95% CI] |
|---|---|
| P_min − B1 | +0.015 [−0.035, +0.068] |
| P_min − A1 | −0.003 [−0.041, +0.034] |

- **P trades answerable accuracy and speed for not stopping on unanswerable questions.** Against B1 and A1 at θ*, P_min is correct on more unanswerable questions (+0.09 vs both, CI > 0). It also stops 3.2 levels later than B1 and 1.7 later than A1. Its answerable accuracy is lower: −0.06 vs B1 (CI includes 0) and −0.10 vs A1 (CI < 0).
- **The gate matters, decomposition does not.** A1 (observer prompt on the original question, gated by resolved / unknown) is as good as P. A2 (raw item P_ans, no gate) is clearly worst among the thresholded methods (0.540). A3 (resolved flags only, no θ) fires on 89% of unanswerable questions (0.427).

## 2. Run facts

- **Environment.** RTX 5080, driver 580.178.04 (userland = kernel), ollama 0.30.4. No model was resident before the run. After `isolate_model` + warm-up, only qwen3.5:9b was resident: 100% GPU, context 8192, ollama log "CLIP using CUDA0 backend" (`logs/envcheck_*.json`).
- **Determinism gate.** Passed: 120 argmax calls, 811 logprob positions, 0 violations. Both repeat pairs were byte-identical in content and logprobs: planner_v3 text-only, and reader_lp at L20.
- **reader_lp identity check.** The plain reader and reader_lp were compared on 40 dev questions × L1 / L10 / L20. The raw reply was byte-identical in 120 of 120 pairs, so reader_lp is the common scorer and B1, from one call (`identity_check.json`).
- **Calls and time.** 24,660 records: reader 120, reader_lp 6,000, plans_v3 300, observer_v3_lp 12,240, direct_resolved 6,000. Inference took 24,161 new calls in 4.09 h at 0.61 s per call, in 27 foreground chunks of ≤ 585 s resumed from the cache. This is slower than v22's 0.47 s/call, because the observer's three-key JSON reply is longer than v22's relevance reply.
- **Audit** (`logs/audit_dev.json`). Of 24,660 records, 0 had non-generation time > 5 s (max 0.46 s), and there were 0 argmax violations in 305,008 positions. Nothing was quarantined.
- **Validity.** All of the following are 100%: reader_lp, observer and direct_resolved records are valid, and the answer token is located in reader_lp, direct_resolved and every observer item.
- **Planner (300 plans).** 300 are valid; 299 were valid on the first attempt and 1 after one schema repair. 288 plans have 2 items and 12 have 3. In all 300 the last item names the attribute.

## 3. Dataset (`dataset_v27.json`)

| set | questions | images | color | material | shape | relate 0 | relate 1 |
|---|---|---|---|---|---|---|---|
| val eligible (minimal filter) | 9,290 | 5,512 | 7,790 | 1,387 | 113 | 6,187 | 3,103 |
| **dev** (first 300 in hash order, ≤ 2 per image) | 300 | 222 | 250 | 44 | 6 | 202 | 98 |
| testdev eligible = test (all) | 886 | 327 | 578 | 234 | 74 | 623 | 263 |

- **Images.** The 222 dev images in `gqa/images/` are byte-identical to the VG copies (checked). Dev and test share no image.
- **Labels from screening** (reader_lp at L20, 0 invalid replies).

| attribute | answerable | unanswerable |
|---|---|---|
| color | 138 | 112 |
| material | 34 | 10 |
| shape | 3 | 3 |
| **total** | **175** | **125** |

- **Reader accuracy by level.**

| level | 1 | 5 | 10 | 15 | 19 | 20 |
|---|---|---|---|---|---|---|
| answerable | 0.543 | 0.737 | 0.863 | 0.931 | 0.966 | 1.000 |
| unanswerable | 0.072 | 0.064 | 0.056 | 0.080 | 0.064 | 0.000 |

- **The labels are not clean.**
  - 22% of the unanswerable questions (28 of 125) are answered correctly at some level before L20, because the reader is not monotone.
  - 43% of the answerable questions are correct at every level, i.e. already at L1.

## 4. Results at θ* (strict scoring, the pre-registered primary)

θ* is the argmax of dev balanced accuracy, with ties going to the smaller θ. The CI is an image-cluster bootstrap (2000 draws, seed 20261008) with θ held fixed, so selection error is not included. "never" means the method never stops; it counts as L20 in the mean levels.

| method | θ* | balanced acc [95% CI] | answerable acc [95% CI] | unanswerable correct [95% CI] | pooled acc | mean L (all) | mean L (answerable) |
|---|---|---|---|---|---|---|---|
| B1 reader P_ans | 0.90 | 0.614 [0.559, 0.667] | 0.669 [0.594, 0.735] | 0.560 [0.476, 0.649] | 0.623 | 11.12 | 9.64 |
| **P pans_min** | **0.80** | **0.630 [0.575, 0.683]** | 0.611 [0.542, 0.683] | 0.648 [0.561, 0.731] | 0.627 | 14.35 | 13.27 |
| P pans_mean | 0.90 | 0.618 [0.562, 0.673] | 0.571 [0.500, 0.645] | 0.664 [0.579, 0.746] | 0.610 | 14.87 | 13.97 |
| A1 direct_resolved | 0.75 | 0.633 [0.578, 0.688] | 0.714 [0.648, 0.780] | 0.552 [0.466, 0.639] | 0.647 | 12.61 | 10.97 |
| A2 raw item P_ans, no gate | 0.95 | 0.540 [0.483, 0.593] | 0.360 [0.289, 0.433] | 0.720 [0.643, 0.796] | 0.510 | 15.99 | 15.14 |
| A3 all resolved, no θ | — | 0.427 [0.386, 0.472] | 0.743 [0.678, 0.806] | 0.112 [0.059, 0.173] | 0.480 | 8.19 | 7.95 |

The pooled accuracy is over all 300 questions (175 answerable / 125 unanswerable = 58% / 42%).

- **Stopping detail at θ*.** "Stopped wrong" counts answerable questions that stopped at a level where the reader was wrong. The quartiles are of the answerable stop level, with never stopping counted as 20.

| method | answerable never-stop | answerable stopped wrong | answerable stop-level Q1 / median / Q3 |
|---|---|---|---|
| B1 | 0.274 | 10 | 1 / 8 / 20 |
| P_min | 0.383 | 1 | 7 / 14 / 20 |
| P_mean | 0.429 | 0 | 8 / 17 / 20 |
| A1 | 0.229 | 10 | 5 / 10 / 19.5 |
| A2 | 0.594 | 8 | 8 / 20 / 20 |
| A3 | 0.103 | 27 | 3 / 6 / 12 |

- **Unified scoring (sensitivity).** A stop on an unanswerable question counts as correct when the reader happens to be right at that level. This changes balanced accuracy by at most 0.03 and does not change the order.

| method | balanced (unified) | unanswerable correct (unified) |
|---|---|---|
| B1 | 0.614 | 0.560 |
| P_min | 0.634 | 0.656 |
| P_mean | 0.622 | 0.672 |
| A1 | 0.641 | 0.568 |
| A2 | 0.544 | 0.728 |
| A3 | 0.455 | 0.168 |

## 5. Paired comparisons (image-cluster paired bootstrap, 2000 draws, seed 20261008)

### 5.1 Each method at its own θ* (the comparison the confirmatory rule uses on test)
`eval/dev/paired_bootstrap_at_theta_star.csv`. P_min is at θ 0.80, B1 at 0.90 and A1 at 0.75. This is in-sample: the θ* choice is not resampled.

| quantity | P_min − B1 | P_min − A1 |
|---|---|---|
| balanced acc (strict) | +0.015 [−0.035, +0.068] | −0.003 [−0.041, +0.034] |
| balanced acc (unified) | +0.019 [−0.031, +0.072] | −0.007 [−0.045, +0.030] |
| answerable acc | −0.057 [−0.127, +0.017] | −0.103 [−0.160, −0.049] |
| unanswerable correct | +0.088 [+0.016, +0.167] | +0.096 [+0.045, +0.155] |
| pooled acc (strict) | +0.003 [−0.048, +0.057] | −0.020 [−0.061, +0.017] |
| mean level, all | +3.23 [+2.51, +3.91] | +1.74 [+1.36, +2.18] |
| mean level, answerable | +3.63 [+2.71, +4.54] | +2.30 [+1.73, +2.97] |

### 5.2 Same θ for both methods (pre-registered, every θ; `eval/dev/paired_bootstrap.csv`)
Balanced accuracy (strict), P_min − method:

| θ | 0.50 | 0.60 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|---|
| − B1 | +0.195 [+0.142, +0.248] | +0.191 [+0.140, +0.246] | +0.195 [+0.133, +0.258] | +0.122 [+0.066, +0.185] | +0.097 [+0.042, +0.155] | +0.045 [−0.009, +0.102] | −0.029 [−0.078, +0.021] | −0.050 [−0.093, −0.004] |
| − A1 | +0.000 [−0.024, +0.024] | +0.005 [−0.021, +0.032] | −0.008 [−0.040, +0.024] | −0.015 [−0.046, +0.015] | −0.003 [−0.031, +0.023] | −0.007 [−0.030, +0.013] | −0.009 [−0.037, +0.018] | −0.037 [−0.064, −0.013] |

- **Same-θ differences against B1 are a scale effect, not a method effect.** The reader's P_ans sits near 1 at most levels: the L1 mean is 0.67 on both sides. So below θ 0.85, B1 stops almost everything early. Each method's own θ* is the fair comparison (§5.1).
- **P_min and A1 have nearly the same scale and the same curve.** They differ at no θ except 0.95.

## 6. θ sweep (strict; `eval/dev/sweep.csv`, figures in `figures/dev/`)

| θ | B1 bal | P_min bal | P_mean bal | A1 bal | A2 bal | B1 mean L | P_min mean L | A1 mean L |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.271 | 0.271 | 0.271 | 0.271 | 0.271 | 1.00 | 1.00 | 1.00 |
| 0.05 | 0.271 | 0.427 | 0.337 | 0.440 | 0.271 | 1.00 | 8.19 | 7.88 |
| 0.50 | 0.298 | 0.494 | 0.431 | 0.494 | 0.277 | 1.98 | 9.63 | 9.10 |
| 0.70 | 0.411 | 0.606 | 0.475 | 0.614 | 0.369 | 4.80 | 12.68 | 11.86 |
| 0.75 | 0.496 | 0.618 | 0.511 | **0.633** | 0.414 | 6.33 | 13.56 | 12.61 |
| 0.80 | 0.533 | **0.630** | 0.582 | 0.633 | 0.470 | 7.92 | 14.35 | 13.58 |
| 0.85 | 0.554 | 0.599 | 0.610 | 0.606 | 0.492 | 9.10 | 14.98 | 14.39 |
| 0.90 | **0.614** | 0.586 | **0.618** | 0.594 | 0.530 | 11.12 | 15.86 | 15.06 |
| 0.95 | 0.606 | 0.556 | 0.573 | 0.593 | **0.540** | 13.23 | 17.36 | 16.57 |
| 1.00 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 20.00 | 20.00 | 20.00 |

(Mean L is over all 300 questions, with never stopping counted as 20. Bold marks each method's θ*.)

- **θ = 0.00 is the degenerate stop-at-L1 row.** Its answerable accuracy is the L1 reader accuracy (0.543), and its unanswerable correct rate is 0.
- **For P_min, θ 0.05–0.30 equals A3.** A gated item score is non-zero exactly when the item is resolved, and resolved items have P_ans above 0.3.
- **The curves are flat-topped and close.** Every method's peak lies 0.11–0.13 above the 0.5 floor, except A2.

## 7. Firing on unanswerable questions at θ* (125 questions; counts by stop level)

| method | L1–5 | L6–10 | L11–15 | L16–19 | L20 | never |
|---|---|---|---|---|---|---|
| B1 | 38 | 11 | 4 | 2 | 0 | 70 |
| P_min | 11 | 22 | 8 | 3 | 0 | 81 |
| P_mean | 10 | 19 | 10 | 3 | 0 | 83 |
| A1 | 14 | 26 | 9 | 5 | 2 | 69 |
| A2 | 12 | 4 | 13 | 5 | 1 | 90 |
| A3 | 51 | 35 | 19 | 5 | 1 | 14 |

- **When B1 stops wrongly, it stops very early.** 38 of its 55 firings are at L1–5. It is confident from the far view, and it was already so on dev 99 under v22.
- **P and A1 fire mostly at L6–10.** That is after the observer has started to mark the target resolved.

## 8. Per attribute at θ* (descriptive; dev shape is 3 + 3 questions)

| attribute | method | n ans / unans | answerable acc | unanswerable correct | balanced | mean L (answerable) |
|---|---|---|---|---|---|---|
| color | B1 | 138 / 112 | 0.659 | 0.598 | 0.629 | 9.84 |
| color | P_min | 138 / 112 | 0.638 | 0.670 | 0.654 | 12.81 |
| color | A1 | 138 / 112 | 0.732 | 0.571 | 0.652 | 10.22 |
| material | B1 | 34 / 10 | 0.706 | 0.100 | 0.403 | 9.44 |
| material | P_min | 34 / 10 | 0.471 | 0.400 | 0.435 | 15.88 |
| material | A1 | 34 / 10 | 0.647 | 0.400 | 0.524 | 14.62 |
| shape | B1 | 3 / 3 | 0.667 | 0.667 | 0.667 | 2.67 |
| shape | P_min | 3 / 3 | 1.000 | 0.667 | 0.833 | 4.67 |
| shape | A1 | 3 / 3 | 0.667 | 0.333 | 0.500 | 4.00 |

The other methods are in `eval/dev/at_theta_star_by_attribute.csv`, and the full θ grid by attribute is in `sweep_by_attribute.csv`.
- **Material on the unanswerable side has only 10 questions.** B1 stops on 9 of the 10. P and A1 stop later on material (mean answerable L about 15–16) and lose answerable accuracy.

## 9. Observations

1. **The mixed-population task is hard for every signal tried.** The best balanced accuracy is 0.63 against a 0.50 floor. The signals overlap heavily between the two sides.

| mean at L20 | answerable | unanswerable |
|---|---|---|
| reader P_ans | 0.86 | 0.71 |
| pans_min | 0.73 | 0.56 |
| direct_resolved score | 0.75 | 0.58 |

   At L1 the reader P_ans is 0.67 and 0.66. This is the same pattern as the §25 Rel-free probes: no single-model confidence separates the two sets well.
2. **The observer's resolved / unknown gate is the useful ingredient.** Planner decomposition is not. With the gate on the original question, A1 is as good as P_min (balanced −0.003 [−0.041, +0.034] at θ*) and stops 1.7 levels earlier. Without the gate, A2 is worst among the thresholded methods.
3. **P_min vs B1 is an operating-point trade, not a dominance.** P_min stops later and fires less on unanswerable questions (+0.09). B1 stops earlier and is more accurate on answerable questions (+0.06, CI includes 0). Their balanced accuracies are indistinguishable.
4. **A3 shows that "resolved" alone does not abstain.** The observer marks all items resolved on 89% of unanswerable questions at some level, usually by L10. The P_ans threshold on top of the gate is what holds P and A1 back.
5. **The label definition limits what can be shown.**
   - 22% of unanswerable questions are reader-correct at some level. Under unified scoring, a stop there would count as correct.
   - §25 found that both models give the same wrong answer at L20 on most reader-wrong questions, a sign of GQA label noise.
   - Part of the "unanswerable" side is therefore annotation ambiguity that no stopping signal can detect.
6. **θ* lies at the high end for all methods (0.75–0.95).** The curves are fairly flat near the top: P_min stays within 0.03 of its peak over θ 0.70–0.80, A1 over 0.70–0.85, and B1 over 0.90–0.95. So the test balanced accuracy at θ* should be about as stable as the curve, but a dev-to-test drop like v22's (−0.01 to −0.05) is plausible.

## 10. Files and commands

```
uv run python -m milestone1.gqa_stop.cli preflight      --config configs/gqa_stop_v27.json
uv run python -m milestone1.gqa_stop.cli v27-extract    --config configs/gqa_stop_v27.json
uv run python -m milestone1.gqa_stop.cli v27-envcheck   --config configs/gqa_stop_v27.json
uv run python -m milestone1.gqa_stop.cli v27-detgate    --config configs/gqa_stop_v27.json
uv run python -m milestone1.gqa_stop.cli v27-identity   --config configs/gqa_stop_v27.json
uv run python -m milestone1.gqa_stop.cli v27-screen     --split dev --config configs/gqa_stop_v27.json
timeout 585 uv run python -m milestone1.gqa_stop.cli v27-infer --split dev --max-seconds 560 --config configs/gqa_stop_v27.json   # x27
uv run python -m milestone1.gqa_stop.cli v27-audit      --split dev --config configs/gqa_stop_v27.json
uv run python -m milestone1.gqa_stop.cli v27-table      --split dev --config configs/gqa_stop_v27.json
uv run python -m milestone1.gqa_stop.cli v27-evaluate   --split dev --config configs/gqa_stop_v27.json
```

- **Outputs** (`outputs/gqa_stop/visapp_gqa_v27/`):
  - `level_table_dev.csv`
  - `eval/dev/`: `sweep.csv`, `sweep_by_attribute.csv`, `at_theta_star*.csv`, `method_ci_at_theta_star.csv`, `paired_bootstrap.csv`, `paired_bootstrap_at_theta_star.csv`, `theta_star.json`, `summary.json`
  - `figures/dev/`: `balanced_vs_theta.png`, `answerable_acc_vs_level.png`, `unanswerable_vs_theta.png`
  - `logs/`: audit, env check, infer progress
- **Code.** `src/milestone1/gqa_stop/v27_data.py`, `v27_infer.py` and `v27_eval.py`; the minimal filter in `dataset.py`; tests in `tests/test_gqa_stop_v27.py`.
