# gqa_stop: Rel-free signals as a method table (dev, cache-only)

Date: 2026-10-10. Cache-only: no model call was made. Inputs are the v22 dev level tables and raw `direct_lp/` and `observer_v3_lp/` records (top_logprobs, 10 per token) for the dev visual subset (102 questions, 89 images) and the dev reader-wrong set (99 questions, 85 images).
Outputs: `outputs/gqa_stop/visapp_gqa_v22/probe_unans/relfree/`. Scripts there: `extract.py` (signals, re-using `probe_unans/signals.py`), `evaluate.py` (all tables). Re-running both reproduces every CSV byte-for-byte.
Definitions follow §22/§23. θ grid 0.01–1.00 plus ∞, selection = minimum mean level with accuracy ≥ target (ties: higher accuracy, larger θ). Firing = score ≥ θ at some level 1–20 on the 99; the forced L20 stop is not a firing. Bootstrap = image-cluster paired, 2000 draws, seed 20261008, θ held at its selection.
Reproduction: recomputed P_ans, direct_v4, checklist_v4_min, pans_min, pans_mean and checklist_v4_mean match the pipeline columns to ≤ 3.4e-16. The direct_v4 and checklist_v4_min rows reproduce §22/§23 exactly.

## Verdict

- **Rel-free direct signals are one method, not three.** At 0.95 out of fold, direct P_ans, margin and negent all give accuracy 0.947–0.949 at mean level 13.35–13.87 with firing 0.457–0.459. All three beat both references on firing at no level cost: direct_pans − direct_v4 = −0.188 [−0.301, −0.073]; − checklist_v4_min = −0.190 [−0.307, −0.069]. The merged margin is slightly worse (firing 0.519; +0.061 [+0.014, +0.113] vs direct_pans).
- **The direct gain comes from an "early or never" stopping mode.** At 0.95, direct_pans needs θ 0.83: it forces L20 on 45 of the 102 (stop20 0.44 vs 0.25 for direct_v4) and stops 21 of them at L1–5. On the 99 it fires early too: 36 of its 46 firings are at L ≤ 9 (median L4.5), against 18 of 67 for direct_v4 (median L13). Equal mean level is reached by a bimodal stop distribution, and its remaining false stops are confident low-level answers.
- **Rel-free checklist signals do not beat the references at 0.95.** pans_min, margin_min and negent_min convert the Rel removal into levels instead (out of fold: −1.0 to −1.3 levels vs checklist_v4_min, CI excludes 0). Their firing stays at 0.51–0.61 (pans_min − direct_v4: −0.063 [−0.158, +0.025]). At 0.97 they do lower firing by 0.19–0.23 at equal level.
- **Checklist vs direct for the same Rel-free signal is a level-for-firing trade, and it sits in the 15 shape questions.** pans_min − direct_pans at 0.95: mean level −1.75 [−3.31, −0.22], firing +0.232 [+0.112, +0.343]. On shape: −9.87 levels at accuracy 0.867 vs 1.000, firing +0.69 on the 13 shape questions. Shape accounts for −1.45 of the −1.75. The same pattern holds for margin and negent; at 0.97 and 0.99 the overall difference vanishes.
- **Frontier (in-sample, acc ≥ 0.95).** At mean level ≤ 15 the minimum firing is 0.545 for direct_v4 and 0.566 for checklist_v4_min. The Rel-free signals reach 0.29–0.41, best negent_min 0.293 and pans_min 0.313. At ≤ 17 every Rel-free signal reaches 0.19–0.24, against 0.384 and 0.273.
- **At 0.99 Rel-free direct saves levels rather than firing.** Out of fold, direct_pans gives 16.45 against 17.92 for direct_v4 (−1.47 [−2.57, −0.41]). Its firing is 0.237 against 0.277 (−0.039 [−0.129, +0.053]).
- **Bar for V6-C.** The Rel-free baseline to beat is direct_pans out of fold at 0.95: 0.947 @ 13.57, firing 0.459. Rel does not see answerability: its mean by level is nearly identical on the two sets (§8), and so is P_ans.

## 1. Signals

All signals are taken at the first answer-value token, located with the pipeline's `locate_answer_token`. Every record has 10 top_logprobs, and the greedy token is always the top-1 (4020/4020).

| signal | definition |
|---|---|
| P_ans | probability of the greedy answer token (full-vocab, from logprobs) |
| margin | p1 − p2 over the raw top-10 |
| margin_merged | same after summing case, quote and whitespace variants ("metal" / "Metal" / " metal") |
| negent | 1 − H / log 10 over the renormalised top-10 |

- **Direct.** The signal from the `direct_lp` call on the original question, used alone.
- **Checklist (min).** For each planner item, the observer_v3 signal goes through the existing v1 existence gate: unresolved items and items with an unresolved ancestor score 0. The score is the min over items, as for checklist_v4_min.
- **Checklist (mean, logged).** Same gate, mean over items.
- **References.** direct_v4 = P_ans × Rel and checklist_v4_min = min over gated items of P_ans × Rel.

## 2. Per-target selection on the 102 (in-sample; `targets.csv`)

At 0.95 every method stops wrong on 5 questions with accuracy 0.951. Δ columns are paired bootstraps against the reference at the same target. "s20" is the stop20 rate.

| method | θ | mean L | s20 | firing99 | Δ mean L vs direct_v4 | Δ firing vs direct_v4 | Δ mean L vs checklist_v4_min | Δ firing vs checklist_v4_min |
|---|---|---|---|---|---|---|---|---|
| direct_v4 | 0.30 | 13.83 | 0.25 | 0.677 | | | −0.31 [−0.83, +0.18] | +0.030 [−0.030, +0.095] |
| checklist_v4_min | 0.28 | 14.15 | 0.29 | 0.646 | +0.31 [−0.18, +0.83] | −0.030 [−0.095, +0.030] | | |
| direct_pans | 0.83 | 13.48 | 0.44 | 0.465 | −0.35 [−1.79, +1.22] | **−0.212 [−0.324, −0.095]** | −0.67 [−2.24, +1.00] | **−0.182 [−0.305, −0.053]** |
| direct_margin | 0.79 | 14.14 | 0.49 | 0.465 | +0.30 [−1.14, +1.81] | −0.212 [−0.330, −0.093] | −0.01 [−1.53, +1.64] | −0.182 [−0.310, −0.052] |
| direct_margin_merged | 0.79 | 13.43 | 0.44 | 0.505 | −0.40 [−1.83, +1.12] | −0.172 [−0.291, −0.052] | −0.72 [−2.25, +0.95] | −0.141 [−0.268, −0.011] |
| direct_negent | 0.71 | 13.25 | 0.44 | 0.465 | −0.59 [−2.01, +1.00] | −0.212 [−0.333, −0.089] | −0.90 [−2.48, +0.78] | −0.182 [−0.308, −0.050] |
| pans_min | 0.58 | 11.73 | 0.24 | 0.697 | −2.11 [−2.97, −1.24] | +0.020 [−0.074, +0.108] | −2.42 [−3.12, −1.75] | +0.051 [−0.030, +0.130] |
| margin_min | 0.40 | 12.43 | 0.30 | 0.677 | −1.40 [−2.31, −0.50] | 0.000 [−0.101, +0.097] | −1.72 [−2.43, −1.01] | +0.030 [−0.050, +0.108] |
| margin_merged_min | 0.42 | 12.50 | 0.30 | 0.677 | −1.33 [−2.24, −0.44] | 0.000 [−0.101, +0.097] | −1.65 [−2.34, −0.96] | +0.030 [−0.050, +0.108] |
| negent_min | 0.58 | 12.32 | 0.26 | 0.636 | −1.51 [−2.46, −0.56] | −0.040 [−0.144, +0.064] | −1.82 [−2.62, −1.05] | −0.010 [−0.108, +0.085] |
| pans_mean (logged) | 0.79 | 12.69 | 0.29 | 0.606 | −1.15 [−1.97, −0.35] | −0.071 [−0.170, +0.029] | −1.46 [−2.10, −0.86] | −0.040 [−0.119, +0.031] |
| margin_mean (logged) | 0.66 | 12.58 | 0.30 | 0.626 | −1.25 [−2.07, −0.45] | −0.051 [−0.147, +0.041] | −1.57 [−2.18, −0.97] | −0.020 [−0.091, +0.043] |
| negent_mean (logged) | 0.78 | 13.79 | 0.37 | 0.485 | −0.04 [−0.86, +0.74] | −0.192 [−0.298, −0.089] | −0.35 [−0.96, +0.22] | −0.162 [−0.252, −0.077] |

margin_merged_mean equals margin_mean to two decimals at every target. The accuracy differences are 0.000 at every target, because all methods hit the same allowed number of wrong stops; the CIs are in `paired.csv`.

Other targets, as θ: accuracy @ mean level, s20, firing99:

| method | 0.90 | 0.97 | 0.99 |
|---|---|---|---|
| direct_v4 | 0.22: 0.902 @ 12.25, 0.17, 0.778 | 0.36: 0.971 @ 14.77, 0.31, 0.545 | 0.39: 0.990 @ 15.46, 0.39, 0.535 |
| checklist_v4_min | 0.14: 0.912 @ 11.87, 0.18, 0.788 | 0.36: 0.971 @ 15.01, 0.34, 0.556 | 0.42: 0.990 @ 15.76, 0.44, 0.495 |
| direct_pans | 0.78: 0.912 @ 11.52, 0.36, 0.556 | 0.85: 0.971 @ 14.25, 0.49, 0.424 | 0.88: 0.990 @ 15.86, 0.68, 0.303 |
| direct_margin | 0.69: 0.902 @ 11.34, 0.36, 0.566 | 0.80: 0.971 @ 14.47, 0.52, 0.404 | 0.81: 0.990 @ 14.90, 0.55, 0.394 |
| direct_margin_merged | 0.70: 0.902 @ 10.97, 0.33, 0.616 | 0.81: 0.971 @ 14.20, 0.49, 0.485 | 0.84: 0.990 @ 15.09, 0.59, 0.364 |
| direct_negent | 0.65: 0.912 @ 11.30, 0.34, 0.566 | 0.74: 0.971 @ 14.39, 0.51, 0.434 | 0.80: 0.990 @ 16.23, 0.69, 0.253 |
| pans_min | 0.53: 0.902 @ 11.03, 0.19, 0.758 | 0.78: 0.971 @ 14.56, 0.47, 0.384 | 0.84: 0.990 @ 15.73, 0.56, 0.283 |
| margin_min | 0.24: 0.902 @ 11.05, 0.21, 0.778 | 0.58: 0.971 @ 13.89, 0.38, 0.505 | 0.77: 0.990 @ 16.17, 0.62, 0.242 |
| negent_min | 0.46: 0.902 @ 10.50, 0.14, 0.727 | 0.70: 0.971 @ 14.58, 0.47, 0.343 | 0.76: 0.990 @ 15.71, 0.57, 0.283 |
| pans_mean | 0.69: 0.902 @ 10.77, 0.20, 0.788 | 0.84: 0.971 @ 13.97, 0.41, 0.485 | 0.91: 0.990 @ 16.03, 0.60, 0.263 |
| negent_mean | 0.63: 0.902 @ 10.69, 0.19, 0.798 | 0.81: 0.971 @ 14.33, 0.45, 0.394 | 0.84: 0.990 @ 15.39, 0.53, 0.313 |

Stopped-but-wrong ids:
- **0.95, all four direct Rel-free signals.** 07949044, 0813387, 14848534, 18486270, 18486310.
- **0.95, pans_min / margin_min / negent_min / pans_mean.** 11619091, 17888876, 18486270, 18486310, 19315625.
- **0.95, references.** direct_v4: 03295873, 0813387, 11213386, 13386996, 18486270. checklist_v4_min: 11213386, 11619091, 18486270, 18486310, 19315625.
- **0.97.** direct_pans: 07949044, 0813387, 14848534. pans_min: 18486270, 18486310, 19315625. negent_min: 11619091, 18486270, 19315625.
- **0.99.** direct_pans: 07949044. pans_min and negent_min: 19315625. direct_v4: 11213386. checklist_v4_min: 02957790.
- **11213386 drops out.** The unstable umbrella question from §22 is not a wrong stop for any Rel-free min or direct signal at 0.95.

## 3. Firing on the 99 (`firing_summary.csv`, `firing_level_hist.csv`)

Each cell gives the firing rate, median firing level, number fired at L ≤ 9, and number where the reader was correct at the firing level.

| method | 0.90 | 0.95 | 0.97 | 0.99 |
|---|---|---|---|---|
| direct_v4 | 0.778 / L12 / 26 / 6 | 0.677 / L13 / 18 / 1 | 0.545 / L13.5 / 14 / 1 | 0.535 / L14 / 13 / 1 |
| checklist_v4_min | 0.788 / L11 / 33 / 4 | 0.646 / L12.5 / 22 / 3 | 0.556 / L14 / 14 / 1 | 0.495 / L14 / 11 / 1 |
| direct_pans | 0.556 / L4 / 43 / 4 | 0.465 / L4.5 / 36 / 4 | 0.424 / L5 / 32 / 4 | 0.303 / L5 / 25 / 3 |
| direct_margin | 0.566 / L4 / 47 / 4 | 0.465 / L5 / 36 / 4 | 0.404 / L4 / 33 / 4 | 0.394 / L5 / 33 / 3 |
| direct_margin_merged | 0.616 / L4 / 49 / 4 | 0.505 / L4 / 40 / 4 | 0.485 / L5 / 38 / 4 | 0.364 / L5 / 30 / 3 |
| direct_negent | 0.566 / L4 / 43 / 4 | 0.465 / L4 / 37 / 4 | 0.434 / L5 / 33 / 4 | 0.253 / L6 / 21 / 3 |
| pans_min | 0.758 / L8 / 40 / 4 | 0.697 / L8 / 37 / 4 | 0.384 / L8.5 / 23 / 2 | 0.283 / L9 / 17 / 0 |
| margin_min | 0.778 / L9 / 43 / 4 | 0.677 / L9 / 36 / 3 | 0.505 / L9 / 27 / 2 | 0.242 / L9 / 14 / 0 |
| negent_min | 0.727 / L7 / 47 / 5 | 0.636 / L8 / 36 / 4 | 0.343 / L7.5 / 22 / 2 | 0.283 / L9 / 18 / 0 |
| pans_mean | 0.788 / L8 / 47 / 4 | 0.606 / L8 / 34 / 3 | 0.485 / L8.5 / 27 / 2 | 0.263 / L9 / 15 / 0 |
| negent_mean | 0.798 / L8 / 48 / 3 | 0.485 / L8.5 / 27 / 2 | 0.394 / L9 / 22 / 2 | 0.313 / L9 / 16 / 0 |

Firing-level histogram at 0.95, as counts per level band:

| method | L1–5 | L6–10 | L11–15 | L16–19 | L20 |
|---|---|---|---|---|---|
| direct_v4 | 3 | 17 | 21 | 17 | 9 |
| checklist_v4_min | 6 | 18 | 24 | 10 | 6 |
| direct_pans | 26 | 12 | 6 | 0 | 2 |
| direct_negent | 26 | 12 | 6 | 1 | 1 |
| pans_min | 19 | 22 | 22 | 6 | 0 |
| negent_min | 16 | 22 | 16 | 7 | 2 |

The stop levels on the 102 at 0.95 show the same bimodality. direct_pans stops 21 / 15 / 16 / 5 / 45 questions in the same bands; direct_v4 stops 5 / 26 / 30 / 16 / 25.

- **Reader correct at the firing level.** The direct Rel-free signals give 00943538, 05748941, 07833741 and 19415660 at 0.90–0.97. pans_min gives 00943538, 07833741, 091042275 and 19415660 at 0.90/0.95, and 091042275 and 17931122 at 0.97.
- **Reading.** Without Rel, the direct score is high either at low levels, where the model answers confidently from little evidence, or not at all. On the 102 those early stops are mostly correct, because the reader is already right at L1 on 34 of them. On the 99 they are confident wrong answers, often the label-ambiguity cases of §23.

## 4. Frontier (`frontier.csv`; in-sample, minimum firing over θ s.t. accuracy ≥ target and mean level ≤ b)

At target 0.95:

| method | ≤ 14 | ≤ 15 | ≤ 16 | ≤ 17 |
|---|---|---|---|---|
| direct_v4 | 0.677 | 0.545 | 0.475 | 0.384 |
| checklist_v4_min | — | 0.566 | 0.434 | 0.273 |
| direct_pans | 0.444 | 0.384 | 0.303 | 0.222 |
| direct_margin | — | 0.394 | 0.293 | 0.192 |
| direct_margin_merged | 0.505 | 0.414 | 0.333 | 0.232 |
| direct_negent | 0.444 | 0.374 | 0.263 | 0.242 |
| pans_min | 0.485 | 0.313 | 0.253 | 0.202 |
| margin_min | 0.485 | 0.404 | 0.253 | 0.202 |
| negent_min | 0.556 | **0.293** | 0.283 | 0.192 |
| pans_mean | 0.485 | 0.343 | 0.293 | 0.212 |
| negent_mean | 0.465 | 0.364 | 0.283 | 0.212 |

- **0.90.** Only the ≤ 14 column changes: checklist_v4_min (0.687) and direct_margin (0.465) become feasible, and negent_min drops to 0.434. Every other cell equals the 0.95 table.
- **0.97.** Only margin_min (0.485), pans_mean (0.485), margin_mean (0.525) and their merged twins reach ≤ 14. At ≤ 15–17 the values equal the 0.95 row, and checklist_v4_min needs ≤ 16.
- **0.99.** At ≤ 16 the values are direct_v4 0.475, checklist_v4_min 0.434, direct_pans 0.303, pans_min 0.253, negent_min 0.283. At ≤ 17 all Rel-free signals give 0.19–0.24.

## 5. Breakdown at 0.95 (`breakdown.csv`; θ as in §2)

Each cell gives accuracy @ mean level (stop20) on the 102, and firing on the 99. The counts per cell are color 41/56, material 46/30, shape 15/13, relate 0 53/55, relate 1 49/44.

| method | color | material | shape | relate 0 | relate 1 |
|---|---|---|---|---|---|
| direct_v4 | 0.927 @ 11.29 (0.07); 0.804 | 0.957 @ 16.72 (0.43); 0.500 | 1.000 @ 11.93 (0.13); 0.538 | 0.925 @ 12.94 (0.19); 0.655 | 0.980 @ 14.80 (0.31); 0.705 |
| checklist_v4_min | 0.951 @ 11.61 (0.07); 0.732 | 0.957 @ 17.76 (0.57); 0.467 | 0.933 @ 10.00 (0.07); 0.692 | 0.943 @ 13.51 (0.26); 0.655 | 0.959 @ 14.84 (0.33); 0.636 |
| direct_pans | 0.951 @ 10.34 (0.32); 0.518 | 0.935 @ 15.07 (0.43); 0.500 | 1.000 @ 17.20 (0.80); 0.154 | 0.943 @ 14.13 (0.51); 0.491 | 0.959 @ 12.78 (0.37); 0.432 |
| direct_margin | 0.951 @ 10.88 (0.37); 0.500 | 0.935 @ 15.85 (0.48); 0.533 | 1.000 @ 17.80 (0.87); 0.154 | 0.943 @ 14.66 (0.57); 0.473 | 0.959 @ 13.57 (0.41); 0.455 |
| direct_negent | 0.951 @ 10.54 (0.34); 0.500 | 0.935 @ 14.72 (0.43); 0.533 | 1.000 @ 16.13 (0.73); 0.154 | 0.943 @ 13.91 (0.51); 0.473 | 0.959 @ 12.53 (0.37); 0.455 |
| pans_min | 0.976 @ 9.46 (0.10); 0.679 | 0.957 @ 15.17 (0.41); 0.667 | 0.867 @ 7.33 (0.07); 0.846 | 0.943 @ 11.64 (0.25); 0.655 | 0.959 @ 11.82 (0.22); 0.750 |
| margin_min | 0.976 @ 10.02 (0.12); 0.643 | 0.957 @ 16.11 (0.52); 0.667 | 0.867 @ 7.73 (0.13); 0.846 | 0.943 @ 12.62 (0.32); 0.655 | 0.959 @ 12.22 (0.29); 0.705 |
| negent_min | 0.976 @ 10.66 (0.17); 0.571 | 0.957 @ 15.22 (0.39); 0.667 | 0.867 @ 8.00 (0.13); 0.846 | 0.943 @ 12.64 (0.32); 0.673 | 0.959 @ 11.98 (0.20); 0.591 |
| pans_mean | 0.976 @ 10.49 (0.15); 0.589 | 0.957 @ 16.11 (0.48); 0.533 | 0.867 @ 8.20 (0.13); 0.846 | 0.943 @ 12.79 (0.32); 0.600 | 0.959 @ 12.57 (0.27); 0.614 |
| negent_mean | 0.976 @ 12.12 (0.24); 0.446 | 0.957 @ 16.93 (0.57); 0.500 | 0.867 @ 8.73 (0.13); 0.615 | 0.943 @ 13.87 (0.38); 0.564 | 0.959 @ 13.71 (0.37); 0.386 |

- **Shape splits direct and checklist.** Direct Rel-free signals almost never stop on shape (stop20 0.73–0.87) and fire on 2 of 13 shape questions. The min signals stop shape at L7–8 with 2 wrong of 15 and fire on 11 of 13. A likely cause, unverified: the observer prompt normalises shape words ("round, square, rectangular"), which concentrates the item answer token, while the direct answer spreads over synonyms.
- **Color firing drops most without Rel.** direct_v4 fires on 0.804 of the colour questions, against 0.50–0.52 for direct Rel-free signals.

## 6. Out-of-fold (`crossfit.csv`; image-grouped 5-fold × 10 seeds over 102 ∪ 99, θ from train-fold 102 only)

Δ = mean of per-question seed-averaged differences, with the image-cluster bootstrap CI.

At 0.95:

| method | OOF acc (seed min) | OOF mean L | OOF firing | Δ L vs direct_v4 | Δ firing vs direct_v4 | Δ firing vs direct_pans | Δ firing vs checklist_v4_min |
|---|---|---|---|---|---|---|---|
| direct_v4 | 0.945 (0.931) | 13.98 | 0.646 | | | +0.188 [+0.073, +0.301] | −0.002 [−0.060, +0.054] |
| checklist_v4_min | 0.946 (0.931) | 14.16 | 0.648 | +0.18 [−0.33, +0.70] | +0.002 [−0.054, +0.060] | +0.190 [+0.069, +0.307] | |
| direct_pans | 0.947 (0.931) | 13.57 | **0.459** | −0.41 [−1.83, +1.12] | **−0.188 [−0.301, −0.073]** | | **−0.190 [−0.307, −0.069]** |
| direct_margin | 0.949 (0.941) | 13.87 | 0.457 | −0.12 [−1.55, +1.44] | −0.190 [−0.305, −0.076] | −0.002 [−0.043, +0.028] | −0.192 [−0.309, −0.067] |
| direct_margin_merged | 0.946 (0.931) | 13.42 | 0.519 | −0.57 [−2.01, +0.95] | −0.127 [−0.244, −0.014] | +0.061 [+0.014, +0.113] | −0.129 [−0.250, −0.009] |
| direct_negent | 0.949 (0.922) | 13.35 | 0.459 | −0.64 [−2.05, +0.90] | −0.188 [−0.302, −0.069] | 0.000 [−0.022, +0.018] | −0.190 [−0.307, −0.065] |
| pans_min | 0.950 (0.941) | 12.85 | 0.584 | −1.14 [−2.02, −0.29] | −0.063 [−0.158, +0.025] | +0.125 [+0.019, +0.225] | −0.065 [−0.134, +0.007] |
| margin_min | 0.944 (0.922) | 12.92 | 0.609 | −1.07 [−1.96, −0.21] | −0.037 [−0.133, +0.053] | +0.151 [+0.036, +0.260] | −0.039 [−0.111, +0.034] |
| margin_merged_min | 0.944 (0.922) | 12.93 | 0.609 | −1.05 [−1.94, −0.20] | −0.037 [−0.133, +0.053] | +0.151 [+0.036, +0.260] | −0.039 [−0.111, +0.034] |
| negent_min | 0.943 (0.931) | 13.16 | 0.508 | −0.82 [−1.76, +0.08] | −0.138 [−0.234, −0.041] | +0.049 [−0.053, +0.150] | −0.140 [−0.225, −0.053] |
| pans_mean | 0.945 (0.931) | 12.95 | 0.605 | −1.03 [−1.85, −0.25] | −0.041 [−0.135, +0.045] | +0.146 [+0.027, +0.258] | −0.043 [−0.113, +0.023] |
| margin_mean | 0.946 (0.931) | 12.70 | 0.616 | −1.28 [−2.07, −0.51] | −0.030 [−0.121, +0.052] | +0.158 [+0.034, +0.269] | −0.032 [−0.100, +0.029] |
| negent_mean | 0.936 (0.922) | 13.80 | 0.463 | −0.19 [−1.00, +0.57] | −0.184 [−0.281, −0.087] | +0.004 [−0.104, +0.109] | −0.186 [−0.267, −0.108] |

Other targets, as OOF accuracy @ mean level, firing:

| method | 0.90 | 0.97 | 0.99 |
|---|---|---|---|
| direct_v4 | 0.887 @ 11.92, 0.793 | 0.962 @ 14.80, 0.553 | 0.990 @ 17.92, 0.277 |
| checklist_v4_min | 0.889 @ 11.59, 0.793 | 0.951 @ 15.02, 0.569 | 0.988 @ 16.05, 0.381 |
| direct_pans | 0.895 @ 11.09, 0.577 | 0.963 @ 14.31, 0.408 | 0.989 @ 16.45, 0.237 |
| direct_margin | 0.892 @ 11.19, 0.570 | 0.963 @ 14.46, 0.404 | 0.990 @ 16.88, 0.220 |
| direct_margin_merged | 0.890 @ 10.97, 0.612 | 0.963 @ 14.15, 0.481 | 0.989 @ 16.14, 0.289 |
| direct_negent | 0.897 @ 11.06, 0.569 | 0.960 @ 14.18, 0.423 | 0.989 @ 16.78, 0.229 |
| pans_min | 0.892 @ 10.73, 0.774 | 0.967 @ 14.58, 0.365 | 0.988 @ 16.47, 0.222 |
| margin_min | 0.887 @ 10.94, 0.784 | 0.966 @ 14.87, 0.359 | 0.988 @ 16.80, 0.204 |
| negent_min | 0.891 @ 10.65, 0.722 | 0.960 @ 14.56, 0.335 | 0.988 @ 16.15, 0.241 |
| pans_mean | 0.890 @ 10.86, 0.775 | 0.964 @ 14.45, 0.392 | 0.988 @ 16.77, 0.217 |
| negent_mean | 0.891 @ 10.94, 0.752 | 0.966 @ 14.54, 0.383 | 0.989 @ 16.57, 0.231 |

- **0.97.** The Rel-free signals lower firing against both references at equal level, with all mean-level CIs including 0. The exceptions are direct_margin_merged against both references and direct_negent against direct_v4 (CI upper ≥ 0). Examples: negent_min − checklist_v4_min = −0.233 [−0.326, −0.145]; pans_min − direct_v4 = −0.188 [−0.298, −0.085]; direct_pans − direct_v4 = −0.144 [−0.267, −0.019].
- **0.99.** checklist_v4_min OOF accuracy is 0.988 at 16.05; direct_v4 needs 17.92. Rel-free direct signals sit at 16.1–16.9 with firing 0.22–0.29.
- **Against the V6 pass rule.** No Rel-free variant beats direct_pans on firing out of fold at 0.95. The closest are direct_margin (−0.002) and direct_negent (0.000).

## 7. Checklist vs direct for the same signal (`paired.csv`, kind "checklist vs direct")

Rows give min − direct at the same target, each with its own in-sample θ.

| pair | target | Δ mean L [CI] | Δ firing [CI] | shape only: Δ mean L; Δ acc; Δ firing |
|---|---|---|---|---|
| pans_min − direct_pans | 0.90 | −0.49 [−2.14, +0.93] | +0.202 [+0.082, +0.317] | |
| pans_min − direct_pans | **0.95** | **−1.75 [−3.31, −0.22]** | **+0.232 [+0.112, +0.343]** | −9.87 [−13.27, −6.40]; −0.133; +0.692 [+0.308, +1.000] |
| pans_min − direct_pans | 0.97 | +0.30 [−1.24, +1.76] | −0.040 [−0.147, +0.061] | −8.67; −0.067; +0.308 |
| pans_min − direct_pans | 0.99 | −0.14 [−1.54, +1.25] | −0.020 [−0.115, +0.073] | |
| margin_min − direct_margin | 0.95 | −1.71 [−3.29, −0.11] | +0.212 [+0.088, +0.330] | −10.07; −0.133; +0.692 |
| margin_min − direct_margin | 0.99 | +1.26 [−0.24, +2.82] | −0.152 [−0.265, −0.040] | |
| margin_merged_min − direct_margin_merged | 0.95 | −0.93 [−2.56, +0.57] | +0.172 [+0.053, +0.289] | −10.07; −0.133; +0.692 |
| negent_min − direct_negent | 0.95 | −0.92 [−2.53, +0.57] | +0.172 [+0.052, +0.289] | −8.13; −0.133; +0.692 |
| negent_min − direct_negent | 0.97 | +0.19 [−1.37, +1.63] | −0.091 [−0.194, +0.010] | −8.67; −0.067; +0.231 |
| checklist_v4_min − direct_v4 (ref.) | 0.95 | +0.31 [−0.18, +0.83] | −0.030 [−0.095, +0.030] | −1.93 [−3.00, −1.00]; −0.067; +0.154 |

- **pans_min − direct_pans at 0.95 by attribute.** Color: −0.88 [−3.35, +1.50] levels, firing +0.161 [+0.017, +0.321]. Material: +0.11 [−1.80, +2.11], firing +0.167 [0.000, +0.345]. Shape: −9.87 levels, accuracy 0.867 vs 1.000.
- **Weighted contributions.** Shape contributes −1.45 of the −1.75, color −0.35 and material +0.05. Without shape, the checklist would save about 0.36 levels.
- **By relate.** Relate 0: −2.49 [−4.49, −0.39] levels, firing +0.164. Relate 1: −0.96 [−3.18, +1.25], firing +0.318 [+0.136, +0.490].
- **CI note.** The overall CI here is [−3.31, −0.22]. An earlier figure of [−3.25, −0.20] has the same point value; the small difference is presumably bootstrap-draw order (not verified).
- **Per-attribute rows for every pair and for 0.97** are in `paired.csv`.

## 8. Rel and P_ans by level (`rel_by_level.csv`; motivation for plan §25.11 V6-C)

Mean over questions at each level, given as 102 / 99:

| level | Rel (direct) | rel_min (checklist) | P_ans (direct) |
|---|---|---|---|
| 1 | 0.039 / 0.036 | 0.015 / 0.014 | 0.488 / 0.528 |
| 5 | 0.112 / 0.102 | 0.075 / 0.078 | 0.582 / 0.590 |
| 10 | 0.296 / 0.258 | 0.241 / 0.231 | 0.621 / 0.638 |
| 15 | 0.498 / 0.436 | 0.446 / 0.410 | 0.687 / 0.631 |
| 20 | 0.668 / 0.593 | 0.629 / 0.559 | 0.707 / 0.655 |

- **Rel tracks the view, not answerability.** It rises about 17-fold from L1 to L20 on both sets, with a gap of at most 0.08.
- **P_ans is flat and also not separating.** It starts higher on the 99. This is why V6-C conditions the Yes/No question on the proposed answer or on the target's visibility instead of on generic confidence.

## Caveats

- **Many variants, in-sample θ tables.** Only §6 is out-of-fold, and it covers θ selection, not the choice among 14 variants.
- **Bimodality cost is invisible in mean level.** direct_pans matches direct_v4's mean level while forcing L20 on 44% of answerable questions. Any convex level cost, or a cost on the deployed prior-answerable questions, would change the comparison. The 64 dev prior-answerable questions have no L1–L19 records.
- **The 99 include label and referent ambiguity (§23 §6).** Early firings by Rel-free direct signals are partly confident answers to ambiguous labels. This bias is shared by all methods.
- **Granularity.** 0.95 on 102 allows 5 wrong stops. Mean-level CIs for the direct Rel-free signals are wide (about ±1.5), because θ sits on a steep part of a bimodal distribution.
