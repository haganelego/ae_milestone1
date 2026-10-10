# Consultation brief: resolution-progressive VQA early stopping (GQA) — where we are and what to do next

Date: 2026-10-11. Author: orchestrator session, on behalf of the project owner.
Audience: a senior advisor agent asked to recommend the research direction. This document is self-contained; repository paths are given only for reference.

## 0. What we are asking

We have a working early-stopping pipeline whose intended contribution (question-decomposition checklist) turned out to be no better than a trivial baseline, and whose fallback contribution (controlling false stops on unanswerable questions) has not been achieved after six method families. We need a recommendation on:

1. Is there a principled stopping rule that can keep the false-stop rate on unanswerable questions below a chosen bound while holding accuracy and mean stop level on answerable questions, given the evidence in §5–§7? Or is the goal ill-posed under our data?
2. If it is ill-posed, which reframing of the contribution is most defensible for a VISAPP-level paper: (a) the negative result plus the "self-assessment causes false stops" finding, (b) a changed evaluation population, (c) a changed task framing (robot approach), or something else?
3. Should the frozen test evaluation be resumed now, or only after the contribution is settled?

Constraints on any proposal are listed in §8. Dead ends that should not be re-proposed are in §6.

## 1. Task and setting

- **Task.** Visual question answering on GQA (attribute questions: colour, material, shape, plus existence/relation items inside decompositions). A vision-language model answers a question about an image presented at increasing resolution levels L1 (coarsest) to L20 (full). The system must decide, online, at which level to stop and commit to an answer.
- **Intended future setting.** The resolution increase stands in for a robot approaching the object. Methods must therefore be phrased as "approach / get closer" decisions and must not rely on the fact that the images are downscaled copies of one photograph. Prompts never mention resolution, blur, downscaling or image quality.
- **Model.** qwen3.5:9b via ollama, deterministic decoding (temperature 0, top_k 1, seeded, no penalties, thinking off). Every call is cached and byte-reproducible; a determinism gate is run before each batch. Second readers used in one experiment: gemma3 (other family) and qwen3.5:4b (same family).
- **Hardware.** One RTX 5080. About 0.6 s per call; a 200-question × 20-level × 1-role sweep is ~4,000 calls, ~40 min.
- **Evaluation protocol.** Dev-only development; thresholds frozen on dev; a single test evaluation (frozen policy already exists, sha256 05a84b22…, 251 answerable + 233 reader-wrong test questions). Test inference was started, then paused by the owner after a GPU driver incident; the owner decided to finish verification first. No test metrics exist.
- **Targets.** Accuracy ≥ 0.95 on answerable questions with mean stop level ≤ 17; secondary targets 0.90 / 0.97 / 0.99.

## 2. Definitions

- **Reader.** The common answering role: given the question and the image at level L, outputs a short answer. Accuracy is measured against GQA labels with exact match after normalisation (no synonym merging).
- **Answerable set (dev 102 questions, 89 images).** Questions the reader answers correctly at L20 and not already at L1 ("prior-answerable" questions, 64 on dev, are excluded from this set; they are a known gap, see §8).
- **Reader-wrong set (dev 99 questions, 85 images).** Questions the reader answers wrongly at L20. These are the "unanswerable" questions for this reader. Many are suspected GQA label or referent problems (§7).
- **Direct signal.** From one call on the original question: P_ans = probability of the first answer-value token (full-vocabulary logprob). Rel = P("Yes") to "Are you confident about answering the question with the current view?". direct_v4 = P_ans × Rel. direct_pans = P_ans alone.
- **Checklist signal.** A planner decomposes the question into items (existence, attribute, relation); an observer answers each item per level with its own P_ans and Rel. checklist_v4_min = min over items of P_ans × Rel with an existence gate (items whose referent is unresolved score 0). pans_min = same with P_ans alone.
- **Stopping rule.** Stop at the first level where score ≥ θ (k = 1); forced stop at L20. θ is chosen on a 0.01 grid (plus ∞) as the smallest mean level meeting the accuracy target.
- **Firing rate.** On the reader-wrong set, the fraction of questions where score ≥ θ at some level ≤ 20 (the forced L20 stop does not count). This is the false-stop rate we want to bound. There is no abstention output: "not firing" means the system keeps approaching.
- **Statistics.** Image-cluster paired bootstrap (2,000 draws) for differences; image-grouped 5-fold × 10-seed cross-validation for out-of-fold (OOF) θ selection. Frontier = minimum firing over θ subject to accuracy ≥ target and mean level ≤ b, for b = 14/15/16/17.
- **Pass rule for a new method (pre-registered).** At 0.95 OOF: firing difference CI upper bound < 0 against both direct_v4 and direct_pans; accuracy ≥ 0.93; mean level ≤ direct_pans + 0.5.

## 3. Result 1: the intended contribution (checklist) equals the baseline

In-sample, dev 102, θ chosen per target. Differences are checklist − direct with 95% CI.

| target | checklist_v4_min | direct_v4 | Δ mean level |
|---|---|---|---|
| 0.90 | 0.912 @ 11.87 (θ 0.14) | 0.902 @ 12.25 (θ 0.22) | CI includes 0 |
| 0.95 | 0.951 @ 14.15 (θ 0.28) | 0.951 @ 13.83 (θ 0.30) | +0.31 [−0.18, +0.83] |
| 0.97 / 0.99 | equal | equal | CI includes 0 |

- Both beat fixed-level stopping by about +0.06 accuracy at equal mean level (fixed-level concave hull reaches 0.95 only at mean level 17.2). So adaptive stopping itself works.
- An earlier apparent checklist advantage at 0.99 on a 59-question set (15.05 vs 18.51) disappeared on 102 questions; it rested on one question.
- Material questions force the checklist to L20 on 57% of cases because the min over items is held down by the existence item.
- Firing on the reader-wrong set at 0.95: checklist 0.646 vs direct 0.677, difference −0.03 [−0.10, +0.03]. No difference at any target.

Literature check (Explore until Confident, RSS 2024; ZoomEye, EMNLP 2025; Kadavath et al. P(IK)/P(True)): the direct signal is the standard confidence-based stopping rule. The progressive-resolution VQA setting is only partially novel (related: CARES one-shot sufficient resolution; RANet/GFNet/PICM-Net for classification; VisionThink/AwaRes via RL; Evidence-Order Calibration on GQA). We found no prior work on checklist-min stopping along an input-quality axis, but we cannot claim it since it does not help.

## 4. Result 2: the reframed goal (bound the false-stop rate) and the one positive finding

Goal: keep accuracy ≥ 0.95 and mean level ≤ 17 on answerable questions while driving firing on reader-wrong questions below a stated bound.

The only intervention that lowered firing without a level cost was removing the self-assessment factor Rel.

| method (0.95, OOF) | accuracy | mean level | firing | Δ firing vs direct_v4 |
|---|---|---|---|---|
| direct_v4 (P_ans × Rel) | 0.945 | 13.98 | 0.646 | — |
| direct_pans (P_ans alone) | 0.947 | 13.57 | 0.459 | −0.188 [−0.301, −0.073] |

- Margin (p1 − p2) and normalised entropy behave identically to P_ans (accuracy 0.947–0.949, firing 0.457–0.459); they are one method.
- The gain comes from a bimodal stopping distribution, not from a sharper signal: at 0.95 direct_pans needs θ = 0.83, forces L20 on 44% of answerable questions (vs 25% for direct_v4) and stops 21 of 102 at L1–5. On the reader-wrong set 36 of its 46 firings are at L ≤ 9 (median L4.5). The remaining false stops are early, confident wrong answers. Raising θ to remove them also removes the early correct stops.
- Checklist with P_ans alone (pans_min) converts the Rel removal into earlier stops (−1.0 to −1.3 levels) rather than lower firing; the difference to direct_pans sits in 15 shape questions.
- Frontier (in-sample, accuracy ≥ 0.95): minimum firing at mean level ≤ 15 is 0.545 (direct_v4) / 0.566 (checklist) vs 0.29–0.41 for Rel-free signals; at ≤ 17, 0.19–0.24 for Rel-free. This is the best any threshold rule on these signals can do.
- At 0.99, direct_pans saves 1.47 levels [−2.57, −0.41] over direct_v4 with no firing gain.

Why Rel hurts: mean Rel by level is the same on the answerable and reader-wrong sets (L1/L10/L20: 0.04/0.30/0.67 vs 0.04/0.26/0.59). It tracks image sharpness, not answerability, so multiplying by it adds a resolution-dependent ramp that lets confident wrong answers through once the image is sharp.

## 5. Methods tried for firing control, with outcomes

All dev-only, pre-registered in the plan before running, judged by the §2 pass rule. Out-of-fold unless stated.

| # | method | idea | outcome | verdict |
|---|---|---|---|---|
| 1 | Ground-truth box effective resolution | condition stopping on the target object's apparent size (ideal grounding) | Spearman(effective pixels, correct-onset level) = −0.29; box ∧ direct −0.96 levels [−2.14, +0.19], firing unchanged | fail; predicted-box version not run |
| 1' | Per-attribute θ (colour / material / shape) | the apparent gain of #1 was attribute-specific thresholds | in-sample −1.73 levels, but OOF 0.896 @ 11.95, target met in 0/10 seeds | overfitting |
| 2 | Adjacent-level difference, P("No new relevant detail") | saturation detection on consecutive levels | P(No) 0.91 at L2 → 0.76 at L20, non-discriminative; diff ∧ direct = direct | fail |
| 3 | Remove Rel (direct_pans, margin, entropy) | self-assessment as the false-stop source | firing 0.646 → 0.459 at equal level (§4) | only positive finding |
| 3' | Answer-changed-from-L1 gate | allow stopping only after the answer moved | firing −0.16, but tied to how the answerable set was built (prior-answerable questions would be forced to L20) | not usable as is |
| 3'' | Risk-controlled θ on the mixed population | conformal-style bound | forces ~L20 | fail |
| 4 | Cross-model agreement gate (gemma3 or qwen3.5:4b must give the same answer) | independent reader as a check | gemma3: firing 0.459 → 0.382 but mean level +0.97 [+0.17, +1.85]; equal-cost frontier identical (0.364 vs 0.384 at ≤15, 0.303 vs 0.303 at ≤16, 0.232 vs 0.222 at ≤17). gemma3 agreement is 0.509 on both sets; gemma3 L20 accuracy only 0.55. qwen3.5:4b gate is a no-op (same θ, same OOF) | fail on mean level; no gain at equal cost |
| 5 | Approach-framed legibility ("close enough, or need to get closer? Enough / Closer") | ask the approach question directly | pilot stopped: option-order recency bias +0.22 in P(Enough) (limit 0.15, Spearman 0.944 between orders). Descriptively mean P(Enough) is 0.404 vs 0.405 on the two sets, i.e. no answerability information either | pilot stop |
| 6a | Self-verification P(True) of the proposed answer | reject wrong answers | L20 AUROC correct-vs-wrong proposals 0.391 [0.187, 0.598]; wrong proposals get higher P(True) (0.82 vs 0.78); "red", "wood", "triangle" accepted at ≥ 0.98 | pilot stop, below chance |
| 6b | Attribute visibility P("Yes") ("can you see the object clearly enough to tell its colour?") | per-attribute legibility | Spearman with Rel 0.974; same curve on both sets | pilot stop, paraphrase of Rel |

Also: no cached signal (P_ans, margin, entropy, Rel, mass, level-to-level change) separates answerable from reader-wrong questions; AUROC 0.51–0.62 at every level.

## 6. Dead ends (please do not re-propose unless with a specific reason)

Verbalised confidence (quantised to {0, 0.8, 0.9, 0.95}; intrinsic, not a decoding artefact), sampling consistency, cross-path consistency, blind object identification ("what is in the image?" is flat across levels), per-item observer self-assessment, ground-truth-box effective resolution, adjacent-level difference questions, per-attribute thresholds (overfit), cross-model agreement gates, approach-framed Enough/Closer, self-verification P(True), attribute-visibility Yes/No, and any Yes/No self-assessment of the current view.

## 7. Diagnosis

1. **Every Yes/No self-assessment tracks resolution only.** Rel, legibility, visibility and P(True) all rise monotonically with level and have the same mean curve on answerable and reader-wrong questions. The model reports how sharp the view is, not whether it can answer this question.
2. **Most reader-wrong questions are not detectable in principle.** At L20, gemma3 gives the same wrong answer as the reader on 51/99 and qwen3.5:4b on 61/99 (at some level: 87 and 94). Of 39 gate-passing false stops in #4, 37 had both readers on the same wrong answer. These look like GQA label or referent errors, or genuinely ambiguous attributes, which no agreement or self-check can catch. Manual inspection of a sample supports this but has not been done systematically.
3. **The remaining controllable set is small.** Removing the shared-wrong cases leaves roughly 40 of 99 questions on which a stopping rule could in principle act. With dev of this size the CIs on firing differences are ±0.07–0.10, so only large effects are detectable.
4. **All firing reductions found so far are equivalent to raising θ.** The equal-cost frontier is unchanged by every gate. direct_pans is the exception only because its score distribution is bimodal, and that is a property of the reader's confidence calibration, not a controllable knob.

## 8. Constraints and resources for any proposal

- Single GPU, ~0.6 s/call; a full dev sweep of one new role over 201 questions × 20 levels is ~40 min. Budget for new dev experiments is a few sweeps, not dozens.
- Dev sets are fixed: 102 answerable, 99 reader-wrong, 64 prior-answerable (never run through methods; §25.9 of the plan proposes it to measure gate costs). Test: 251 + 233, frozen, untouched.
- Pre-registration discipline: every new arm needs a written pass rule before running; test is evaluated once.
- Reader prompt and decoding are fixed and common to all methods; no abstention output; no spelling-variant merging in accuracy.
- Approach framing only (no resolution / downscaling language in prompts), because the follow-up work is a robot approaching the object.
- GQA labels cannot be changed, but a re-labelled or filtered evaluation subset is allowed if the procedure is stated.

## 9. Directions considered but not yet run (for the advisor to weigh)

- **Separate label noise from true unanswerability.** Manually classify the 99 reader-wrong questions (and the 233 test ones) into label/referent problems vs genuinely hard; report firing on the latter only. Risk: subjective; reduces the set further.
- **Mixed-population evaluation.** Include the 64 prior-answerable questions so that "early or never" rules pay their real cost, and evaluate a single rule on the full distribution with a risk bound.
- **Object-first stopping.** Require the referent to be identified (an open-vocabulary "what object is at the question's focus?" answer that stabilises across levels) before any attribute answer may stop. Different from blind identification because it is query-conditioned.
- **Order-averaged legibility.** Average both option orders to remove the recency bias in #5. Probably useless given the identical means on both sets.
- **Reframe the contribution.** Main result: adaptive stopping beats fixed stopping, self-assessment (the standard Explore-until-Confident ingredient) raises false stops on unanswerable questions, and answer-probability alone is the better stopping statistic; decomposition does not help. Supporting result: a characterisation of why firing cannot be bounded by thresholding (label noise + resolution-tracking self-assessment). This needs the test run.
- **Change the population or task.** Move to a robot-approach image sequence (real distance change rather than downscaling) or to a dataset with reliable attribute labels; this restarts data collection.

## 10. Reference material

Research notes (English, in the repository): `gqa-stop-v22-dev.md` (§3), `lit-progressive-resolution-stopping.md`, `gqa-stop-v5-dev.md` (#1, #1', #2), `gqa-stop-unanswerable-probes.md` and `gqa-stop-relfree-dev.md` (§4, #3–3''), `gqa-stop-v6-dev.md` (#4–#6b, §7). Plan: `.claude/docs/VISAPP2027_GQA_EXPERIMENT_PLAN.md` §22–§25. Commits: 024fc5e, 344cb26, f224e18, 9879dbe, 8881835, 4e5ed1d, 6d33ab4, d876011, f139ee9, be7457c, 9f5d93b.
