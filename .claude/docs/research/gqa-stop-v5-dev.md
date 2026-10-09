# gqa_stop v5 (plan §24): box effective resolution, level-difference signal, v2/v3 clean replays — dev

Date: 2026-10-10. Run: `configs/gqa_stop_v22.json` (A13 clean decoding, think=false). Outputs are in `outputs/gqa_stop/visapp_gqa_v22/v5/`.
Scope: dev only, 102 visual questions plus the 99 reader-wrong questions from §23. θ and τ are chosen in-sample. There was no test inference and no commit.
Terminology follows plan A14.

## Verdict

- **C, level-difference signal: no gain.** Asked whether level ℓ adds relevant detail over ℓ−1, the model says "No" almost always.
  - Mean P(No) is 0.91 at L2 and falls to 0.76 at L20, on both sets: the opposite direction to a saturation signal, and not discriminative.
  - Diff-only at 0.95 stops at 18.2 (+4.4 vs direct_v4); its low firing (0.05) comes from hardly stopping at all.
  - diff ∧ direct_v4 at 0.95 is identical to direct_v4 (13.84, firing 0.677).
  - The k=2 AND variants trade levels for firing along the same line as raising direct_v4's own θ, for example 14.76 / 0.545 vs direct_v4 θ0.36 at 14.78 / 0.545.
- **B is not run**, because the A gate failed.
- **Overall.** None of the four v5 directions beats direct_v4 on dev: GT-box effective resolution, verbalised v3 under clean decoding, v2 identification, and the level-difference question.

## Verdict so far (A + D)

- **A, GT-box precheck: the gate fails, so B is skipped.** Even with ground-truth boxes, no box rule beats direct_v4 at 0.95 on either axis. The best variant gates direct_v4 on box size: −0.96 levels [−2.14, +0.19], firing +0.03 [−0.09, +0.15]. Box size is only weakly related to when the reader becomes stably correct (Spearman −0.29).
- **D1, v3 verbalised checklist under clean decoding: still quantised.** Item confidences are mostly 0.9 (63%) or 0, giving 6 operating points. At 0.95 the result is 1.000 @ 19.95, against direct_v4 at 13.83. The v3 quantisation was not a penalty artifact; only the values moved, from {0, 0.85, 1.0} to {0, 0.8, 0.9, 0.95}.
- **D2, v2 blind identification under clean decoding.** The a-rate rose from about 0.30 to **0.66**, so the old low value was largely a presence_penalty artifact. It is still **flat across levels** (0.68 at L1, 0.66 at L20), and a given a wrong reader is about the same. The identify question contains the target noun, so the model repeats it.

## A. GT-box precheck (§24.1; no VLM calls)

- **Setup.**
  - Scene graphs are `val_sceneGraphs.json`, extracted to `data/processed/gqa_stop/scenegraphs_v5/`. They were read only as JSON under `python -I`.
  - The target object is found by following the final query's dependency back through filter steps to the step carrying `name (id)`; for relate questions that is the relate step.
  - All 201 questions resolved to exactly one object present in the scene graph: no target is absent and none has multiple ids.
- **Effective pixels.** eff_px(ℓ) = box area fraction × pixels of the level-ℓ degraded image (manifest `degraded_size`). The base long side is 512, the scale is 0.05·20^((ℓ−1)/19), and sizes are floored. The degraded image is upsampled back to base size before sending, so the information content is the degraded size.
- **Target size.** The median box area fraction is 0.059 / 0.088 / 0.031 for color / material / shape on dev 102. It is about 0.028 on the 99 reader-wrong questions, so reader-wrong targets are smaller.

### Onset vs effective resolution (`box_onset.csv`)

The onset is the first level from which the reader stays correct through L20.

| set | n | Spearman(eff_px at L20, onset) | mean onset | eff_px at onset: median [IQR] |
|---|---|---|---|---|
| all | 102 | −0.29 (area fraction: −0.28) | | |
| color | 41 | −0.36 | 5.5 | 60 [26, 208] |
| material | 46 | −0.17 | 8.6 | 195 [69, 1116] |
| shape | 15 | −0.56 | 5.7 | 128 [28, 277] |

The pixel count at onset spans two to three orders of magnitude within each attribute, so a single τ per attribute cannot be sharp. Material needs the most pixels and has the weakest relation.

### Rules on dev 102 and firing on the 99 (`box_rules.csv`, `box_pareto.csv`, `box_curves.png`)

- **Search.** τ_attr comes from a 40-value grid: 0, 38 geometric values over the observed eff_px range, and ∞. The gated variants add θ from the 0.01 grid, giving 4 parameters, all chosen in-sample.
- **Bootstrap.** Paired image-cluster bootstrap with 2000 draws, against direct_v4 at its §22 θ for the same target.

| target | variant | τ color / material / shape | θ | acc | mean L | Δ mean L vs direct_v4 [CI] | firing (99) | Δ firing vs direct_v4 [CI] |
|---|---|---|---|---|---|---|---|---|
| 0.95 | box-only | 222 / 82382 / 1147 | | 0.951 | 14.52 | +0.69 [−0.44, +1.80] | 0.677 | 0.00 [−0.11, +0.12] |
| 0.95 | box ∧ direct_v4 | 31 / 82382 / 308 | 0.06 | 0.951 | 12.87 | −0.96 [−2.14, +0.19] | 0.707 | +0.03 [−0.09, +0.15] |
| 0.95 | box ∧ checklist_v4_min | 31 / 11459 / 308 | 0.09 | 0.951 | 13.31 | −0.52 [−1.37, +0.32] | 0.707 | +0.03 [−0.05, +0.11] |
| 0.90 | box-only | 60 / 11459 / 1147 | | 0.902 | 12.34 | +0.09 [−1.12, +1.35] | 0.768 | −0.01 [−0.10, +0.08] |
| 0.90 | box ∧ direct_v4 | 0 / 0 / 222 | 0.12 | 0.902 | 10.48 | −1.78 [−2.21, −1.33] | 0.899 | **+0.12 [+0.06, +0.19]** |
| 0.90 | box ∧ checklist_v4_min | 31 / 0 / 308 | 0.01 | 0.902 | 10.55 | −1.71 [−2.50, −0.87] | 0.859 | +0.08 [0.00, +0.16] |
| 0.97 | box-only | 1147 / 30725 / 1147 | | 0.971 | 16.22 | +1.44 [+0.53, +2.33] | 0.576 | +0.03 [−0.08, +0.14] |
| 0.97 | box ∧ direct_v4 | 160 / 0 / 0 | 0.34 | 0.971 | 14.72 | −0.06 [−0.27, +0.18] | 0.576 | +0.03 [0.00, +0.07] |
| 0.99 | box-only | 30725 / 30725 / 1147 | | 0.990 | 18.62 | +3.16 [+2.17, +4.17] | 0.212 | −0.32 [−0.45, −0.20] |
| 0.99 | box ∧ direct_v4 | 0 / 0 / 0 | 0.39 | 0.990 | 15.46 | 0 | 0.535 | 0 |

- **Gate (§24.1): not passed.** At 0.95 no variant has a mean-level CI below 0, and no variant has a firing CI below 0.
  - At 0.90 the gated variants stop earlier, but the direct_v4 gate fires significantly more on the 99, so the other-axis condition fails.
  - At 0.99 box-only fires less, but only because it stops 3.2 levels later.
- **Optimism.** These are in-sample optima of 3–4 parameters against direct_v4's single θ, so even the near-miss at 0.95 is optimistic.
- **Curve.** On the accuracy–mean-level plane, the box-only Pareto frontier lies below direct_v4 and checklist_v4_min from about mean level 7 upward. It beats fixed stopping, but it is not better than the VLM's own answer probability × self-assessed confidence.
- **Consequence.** Idea 1 ends here and B (`ground_v5`) is not run.

## D1. v3-style verbalised checklist_min under clean decoding (`v3verb_*`)

- **Construction.** The score is rebuilt from the cached observer_v3 records' verbalised `confidence` under the v1 gate (unresolved or ancestor-unresolved items count as 0, then min). There are no new calls.
- **Check.** The same code applied to each item's P_ans × Rel product reproduces checklist_v4_min exactly (max |Δ| = 0).
- **Item confidence values** (all items × levels): 0.9 ×2571, 0 ×1227, 0.8 ×221, 0.95 ×88, 0.99 ×22, 0.6 ×9, other 2. The min score takes 0.9 ×964 and 0 ×914. The θ grid yields only **6 distinct operating points**.

| target | θ | acc | mean L | stop20 | direct_v4 mean L | Δ vs direct_v4 [CI] |
|---|---|---|---|---|---|---|
| 0.90 | 0.90 | 0.902 | 11.05 | 0.13 | 12.25 | −1.21 [−1.91, −0.47] |
| 0.95 | 0.95 | 1.000 | 19.95 | 0.98 | 13.83 | +6.12 [+5.12, +7.17] |
| 0.97 | 0.95 | 1.000 | 19.95 | 0.98 | 14.78 | +5.18 [+4.20, +6.17] |
| 0.99 | 0.95 | 1.000 | 19.95 | 0.98 | 15.46 | +4.49 [+3.51, +5.47] |

- **Comparison with old v3.** The old v3 gave 0.983 @ 17.78 on dev 60 under presence_penalty 1.5, with values {0, 0.85, 1.0}. Clean decoding keeps the same cliff: θ ≤ 0.90 gives 0.90 @ 11; θ > 0.90 gives about L20. The cliff just moves to 0.9 / 0.95.
- **Conclusion.** Verbalised confidence is intrinsically quantised for qwen3.5:9b, and the penalty only shifted the levels. The 0.90 point stops earlier than direct_v4, which matches the §22 note on fair v1 and is a lucky placement of one of 6 points. checklist_v3verb_mean behaves the same: 1.000 @ 19.14 at 0.95.

## D2. v2 blind identification under clean decoding (`identify_levels_dev.csv`)

- **Calls.** planner_v2 (102) and identify (2,040), with prompts unchanged, at num_predict 512 / 128 as in v2. There were 0 planner failures and 0 invalid identify answers. attr_blind and recompose were not run.
- **a.** The target or a planner synonym appears in the identify answer (v2 token-form match). "Target only" excludes synonyms.

| level | 1 | 3 | 5 | 10 | 15 | 17 | 20 |
|---|---|---|---|---|---|---|---|
| reader acc | 0.33 | 0.50 | 0.66 | 0.76 | 0.88 | 0.90 | 1.00 |
| **a (clean)** | 0.68 | 0.63 | 0.65 | 0.64 | 0.64 | 0.67 | 0.66 |
| a, target only | 0.66 | 0.59 | 0.61 | 0.59 | 0.56 | 0.60 | 0.58 |
| a given reader wrong | 0.63 | 0.55 | 0.60 | 0.63 | 0.67 | 0.90 | – |
| a (old v2, penalty 1.5, dev 100) | 0.32 | 0.25 | 0.25 | 0.29 | 0.30 | 0.32 | 0.30 |

- **Overall.** a is 0.66 over all levels, and 0.60 for the target only.
- **By question type.** a is 0.76 when the identify question carries the referring phrase, against 0.35 for the 24 relate-0 questions that fall back to "List the main objects". By relate: relate 0 0.63, relate 1 0.70.
- **Stability.** Per question, a at L1 equals a at L20 for 80% of questions.
- **Interpretation.**
  - Clean decoding doubles a. The old 0.30 was depressed by presence_penalty 1.5, which penalises repeating prompt tokens, and the target noun is in the prompt.
  - a is still flat in level and does not separate correct from wrong reader answers. It mostly measures whether the model echoes the noun in the identify question ("What is the bag on the left...?" → "bag"). The rest are plausible paraphrases: "helmet" for hat, "bread box" for container.
  - The v2 conclusion stands: a carries no resolution signal. Only its level, not its flatness, was an artifact.

## C. Level-difference saturation signal (`diff_v5`; §24.3)

- **Setup.**
  - The prompt is `prompts/gqa_stop/diff_v5.txt`, an inline template: "The two images show the same scene; the second image is a higher-quality version of the first image. Consider the question: '{question}'. Does the second image reveal any new detail relevant to answering the question that the first image did not? Answer with Yes or No."
  - Each request carries two images in one user message: level ℓ−1, then level ℓ, for ℓ = 2..20.
  - The client gained an `extra_image_bytes` field. It enters the cache key only when non-empty, so all old keys are unchanged.
  - Decoding is plain text with top_logprobs 10, num_predict 8 and the A13 block. The signal is P(exact "No") in the first token's top_logprobs; L1 = 0.
- **Pilot** (20 dev questions × L2/5/10/15/20, `diff_pilot_summary.json`): valid rate 1.0, and "Yes"/"No" present in the top-10 1.0, so the gates pass. The first token was "No" 99/100 times.
- **Two-image control.** It confirms the model reads both images. An identical L20/L20 pair gives mean P(No) 0.87; an L1→L20 pair gives 0.26.
- **Full run.** 102 dev + 99 reader-wrong questions × 19 levels gave 3,819 calls, all valid. Records are in `v5/diff_v5/` and `ext_dev_wrong/diff_v5/`.

### P(No) by level (mean)

| level | 2 | 5 | 8 | 10 | 12 | 15 | 17 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| dev 102 | 0.91 | 0.91 | 0.87 | 0.84 | 0.82 | 0.81 | 0.82 | 0.83 | 0.76 |
| reader-wrong 99 | 0.90 | 0.87 | 0.85 | 0.82 | 0.81 | 0.79 | 0.81 | 0.83 | 0.74 |

Adjacent levels look alike to the model at every scale. The "new detail" judgement grows slightly with level rather than vanishing at saturation.

### Rules (`diff_targets.csv`, `diff_curves.{csv,png}`; θ in-sample; Δ vs direct_v4 at its §22 θ, paired image-cluster bootstrap 2000)

The AND variants require the v4 score ≥ its §22 θ at 0.95: direct_v4 0.30, checklist_v4_min 0.28.

| target | method | θ | acc | mean L | stop20 | firing (99) | Δ mean L [CI] | Δ firing [CI] |
|---|---|---|---|---|---|---|---|---|
| 0.95 | direct_v4 (ref) | 0.30 | 0.951 | 13.83 | 0.25 | 0.677 | – | – |
| 0.95 | diff-only | 0.99 | 0.971 | 18.23 | 0.87 | 0.051 | +4.39 [+2.86, +5.80] | −0.63 [−0.73, −0.52] |
| 0.95 | diff-only k=2 | 0.98 | 0.971 | 17.90 | 0.83 | 0.091 | +4.07 [+2.58, +5.46] | −0.59 [−0.70, −0.47] |
| 0.95 | diff ∧ direct_v4 | 0.21 | 0.951 | 13.84 | 0.25 | 0.677 | +0.01 [0.00, +0.03] | 0.00 |
| 0.95 | diff ∧ direct_v4 k=2 | 0.21 | 0.961 | 14.76 | 0.30 | 0.545 | +0.92 [+0.77, +1.08] | −0.13 [−0.20, −0.07] |
| 0.95 | diff ∧ checklist_v4_min | 0.09 | 0.951 | 14.16 | 0.29 | 0.646 | +0.32 [−0.17, +0.84] | −0.03 [−0.09, +0.03] |
| 0.95 | diff ∧ checklist_v4_min k=2 | 0.09 | 0.961 | 15.07 | 0.33 | 0.586 | +1.24 [+0.74, +1.76] | −0.09 [−0.17, −0.01] |
| 0.97 | diff ∧ direct_v4 | 0.85 | 0.971 | 15.43 | 0.41 | 0.535 | +0.66 [+0.05, +1.32] | −0.01 [−0.10, +0.07] |
| 0.97 | diff ∧ direct_v4 k=2 | 0.52 | 0.971 | 15.03 | 0.32 | 0.545 | +0.26 [−0.07, +0.62] | 0.00 [−0.05, +0.05] |
| 0.99 | diff ∧ checklist_v4_min | 0.89 | 0.990 | 16.50 | 0.56 | 0.455 | +1.04 [+0.16, +1.84] | −0.08 [−0.17, +0.01] |
| 0.99 | diff ∧ checklist_v4_min k=2 | 0.77 | 0.990 | 16.15 | 0.48 | 0.475 | +0.69 [0.00, +1.39] | −0.06 [−0.15, +0.03] |

- **At 0.90.** The AND variants are bound by the fixed 0.95 v4 gate, so they stop 1.6–2.8 levels later than direct_v4. All rows are in `diff_targets.csv`.
- **No variant lowers both axes.** Every reduction in firing is paid for in mean level.
- **The trade is not new.** It is the same trade direct_v4 makes on its own θ curve:
  - direct_v4 at θ 0.36 gives 0.971 @ 14.78 with firing 0.545.
  - diff ∧ direct_v4 k=2 at 0.95 gives 0.961 @ 14.76 with firing 0.545.
  - At 0.97, diff ∧ direct_v4 k=2 is 15.03 / 0.545 against direct_v4's 14.78 / 0.545.
- **Matched-mean-level firing (`diff_two_axis.csv`) is misleading without accuracy.** diff-only k=2 reaches firing 0.36 at mean level 13.8, but its answerable accuracy there is 0.80, against 0.95 for direct_v4. The target-based table above is the fair comparison.
- **Operating points.** diff-only has 28 distinct points over the θ grid; the AND variants have 53–56.

## B. Predicted-box grounding (`ground_v5`)

Not run. The §24.1 gate failed (section A), so per the pre-registration Idea 1 ends.

## Files

- **Code.** `src/milestone1/gqa_stop/boxes_v5.py` (CLI `v5-box`), `replays_v5.py` (`v5-replays`), `diff_v5.py` (`v5-diff-pilot`, `v5-diff`). The client change is the `extra_image_bytes` field.
- **Tests.** `tests/test_gqa_stop_v5.py`.
- **Prompt.** `prompts/gqa_stop/diff_v5.txt`.
- **Data.** Scene graphs are in `data/processed/gqa_stop/scenegraphs_v5/`: untracked, about 44 MB, do not commit.
