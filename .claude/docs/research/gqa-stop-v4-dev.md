# gqa_stop v4 (plan §19): P_ans × Rel item confidence — dev results

Date: 2026-10-09. Run: `configs/gqa_stop_v4.json` (family `logprob_item`), output `outputs/gqa_stop/visapp_gqa_v4/`.
Dev only (60 visual-subset questions × 20 levels). No freeze, no test, nothing committed.

## Verdict

**§19.4: do not abort.** At dev accuracy ≥ 0.95, checklist_v4_min (θ=0.30) stops at mean Level 15.50, against 18.45 for fair v1. The difference is −2.95 [−4.48, −1.41], which is significant.
- The user's target, accuracy ≥ 0.95 with mean Level ≤ 17, is met on dev. This is in-sample: θ was chosen on the same 60 questions, and 0.95 means at most 3 wrong.
- **θ is now a smooth trade-off.** checklist_v4_min has 90 distinct (accuracy, mean Level) operating points over the θ grid, against 2 in v3. The stop score takes 355 distinct values on dev.
- **On dev, the decomposition is not better than Direct.**
  - Direct v4 (P_ans × Rel on the original question) stops at mean Level 12.82 with 0.95 accuracy, earlier than checklist_v4_min.
  - Direct P_ans alone stops at 12.55.
  - Rel-only min (14.85) is also earlier than the product min.
  - Part of the cause: the v3 existence gate still zeroes 35% of questions at L20 (the same zeros as v3).

## Pilot (§19.3; 20 dev visual questions × L1/5/10/15/20)

**Gates.**
- Answer token located: 1.00.
- Rel top_logprobs contain Yes or No: 1.00.
- Initial valid rate: {'observer': 1.0, 'direct_lp': 1.0, 'relevance': 1.0, 'planner': 1.0, 'reader': 1.0}.

**Effect of logprobs on the observer.** observer_v3 output with logprobs equals the v3 cache in 215/215 comparisons, for both raw text and parsed form. Adding logprobs does not change the output.

**Rel monotonic in level.** Items: True. Direct: True.

| L | reader acc | item-1 resolved | mean P_ans (items) | mean Rel (items) | mean product | direct P_ans | direct Rel | distinct min r |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.35 | 0.05 | 0.73 | 0.08 | 0.04 | 0.44 | 0.04 | 1 |
| 5 | 0.70 | 0.35 | 0.65 | 0.22 | 0.13 | 0.59 | 0.12 | 2 |
| 10 | 0.80 | 0.65 | 0.72 | 0.47 | 0.37 | 0.65 | 0.32 | 7 |
| 15 | 0.95 | 0.80 | 0.82 | 0.66 | 0.57 | 0.71 | 0.56 | 14 |
| 20 | 1.00 | 0.85 | 0.77 | 0.80 | 0.63 | 0.69 | 0.70 | 16 |

P_ans is roughly flat (0.65–0.82) and not monotonic. Rel carries the level signal.

## Dev devgrid (60 × 20; no freeze)

- **Calls.** 2,600 observer (logprobs), 3,800 relevance and 1,200 direct (logprobs). All were valid on the first attempt.
- **Reused by cache key.** Planner_v3 plans and reader records were copied, and so were the v1 verbalised-direct records used as a comparator.

Best feasible θ per method at dev accuracy ≥ 0.95. The CI is a paired image-cluster bootstrap with 2000 reps.

| method | θ | acc | mean L | median | stop20 | stopped & wrong | acc − hull | mean L − fair v1 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| **checklist_v4_min** (pre-registered) | 0.3 | 0.950 | 15.50 | 18.5 | 0.48 | 3 | +0.034 | -2.95 [-4.48, -1.41] |
| checklist_v4_mean (logged) | 0.59 | 0.950 | 15.68 | 17.5 | 0.47 | 3 | +0.031 | -2.77 [-4.27, -1.21] |
| pans_min (logged) | 0.84 | 0.950 | 16.83 | 20.0 | 0.65 | 3 | +0.009 | -1.62 [-3.21, -0.05] |
| pans_mean (logged) | 0.91 | 0.950 | 16.73 | 20.0 | 0.63 | 3 | +0.011 | -1.72 [-3.35, -0.12] |
| rel_min (logged) | 0.34 | 0.950 | 14.85 | 16.0 | 0.40 | 3 | +0.047 | -3.60 [-5.05, -2.14] |
| rel_mean (logged) | 0.62 | 0.950 | 15.42 | 16.0 | 0.43 | 3 | +0.036 | -3.03 [-4.53, -1.51] |
| direct_v4 (logged) | 0.26 | 0.950 | 12.82 | 12.0 | 0.20 | 3 | +0.085 | -5.63 [-7.05, -4.17] |
| direct_pans (logged) | 0.8 | 0.950 | 12.55 | 13.0 | 0.45 | 3 | +0.090 | -5.90 [-7.88, -3.89] |
| direct_rel (logged) | 0.59 | 0.950 | 15.30 | 16.0 | 0.37 | 3 | +0.038 | -3.15 [-4.53, -1.69] |
| direct_v1_verbalized (comparator) | inf | 1.000 | 20.00 | 20.0 | 1.00 | 0 | +0.000 | +1.55 [+0.70, +2.54] |
| checklist_v3_min (comparator) | 1.0 | 0.983 | 17.78 | 20.0 | 0.53 | 1 | +0.025 | -0.67 [-1.80, +0.53] |
| fair v1 Checklist-min (subset-frozen) | 0.85 | 0.983 | 18.45 | 20.0 | 0.75 | 1 | +0.012 | – |
| subset Fixed-15 / Fixed-dev (L20) | – | 0.883 / 1.000 | 15 / 20 | | | 7 / 0 | −0.023 / 0 | – |

**Sensitivity of checklist_v4_min to θ** (`dev_sensitivity.json`; θ every 0.05):

| θ | acc | mean L | stopped & wrong |
|---|---|---|---|
| 0.05 | 0.900 | 13.97 | 6 |
| 0.10 | 0.900 | 14.12 | 6 |
| 0.15 | 0.900 | 14.38 | 6 |
| 0.20 | 0.917 | 14.63 | 5 |
| 0.25 | 0.917 | 14.97 | 5 |
| 0.30 | 0.950 | 15.50 | 3 |
| 0.35 | 0.950 | 15.87 | 3 |
| 0.40 | 0.950 | 16.18 | 3 |
| 0.45 | 0.967 | 16.53 | 2 |
| 0.50 | 0.967 | 16.72 | 2 |
| 0.55 | 0.967 | 17.07 | 2 |
| 0.60 | 0.983 | 17.30 | 1 |
| 0.65 | 1.000 | 17.62 | 0 |
| 0.70 | 1.000 | 18.08 | 0 |
| 0.75 | 1.000 | 18.35 | 0 |
| 0.80 | 1.000 | 18.62 | 0 |
| 0.85 | 1.000 | 19.03 | 0 |
| 0.90 | 1.000 | 19.38 | 0 |
| 0.95 | 1.000 | 19.78 | 0 |
| 1.00 | 1.000 | 20.00 | 0 |

**Stop histograms** (level stopped/wrong):
- checklist_v4_min@0.3: L1 1/0, L5 1/1, L6 1/0, L7 3/0, L9 5/1, L10 4/1, L11 1/0, L12 3/0, L13 1/0, L14 3/0, L15 3/0, L16 3/0, L18 1/0, L19 1/0, L20 29/0
- rel_min@0.34: L1 1/0, L5 1/1, L6 1/1, L7 5/0, L9 6/0, L10 2/1, L11 3/0, L12 2/0, L13 2/0, L14 3/0, L15 3/0, L16 4/0, L18 1/0, L19 2/0, L20 24/0
- direct_v4@0.26: L1 1/0, L4 1/0, L5 2/0, L6 3/1, L7 4/0, L8 4/0, L9 5/0, L10 3/0, L11 4/0, L12 4/0, L13 3/0, L14 4/0, L15 1/1, L16 2/0, L17 2/0, L18 2/1, L19 3/0, L20 12/0
- checklist_v3_min@1.0: L7 1/0, L10 1/0, L12 4/0, L13 2/1, L15 6/0, L16 4/0, L17 1/0, L18 6/0, L19 3/0, L20 32/0
- fair_v1_checklist_min@0.85: L4 1/0, L6 1/0, L7 1/0, L9 1/0, L10 1/0, L14 1/0, L15 1/0, L16 1/0, L17 3/0, L18 1/1, L19 3/0, L20 45/0

**Component trends** (`dev_components.csv`):

| L | reader acc | item-1 resolved | min r = 0 | mean P_ans | mean Rel | mean product | distinct min r | mean min r | mean direct v4 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.35 | 0.05 | 0.98 | 0.71 | 0.09 | 0.05 | 2 | 0.01 | 0.03 |
| 3 | 0.55 | 0.17 | 0.98 | 0.68 | 0.14 | 0.08 | 2 | 0.01 | 0.05 |
| 5 | 0.68 | 0.33 | 0.93 | 0.65 | 0.23 | 0.15 | 5 | 0.02 | 0.09 |
| 7 | 0.70 | 0.45 | 0.82 | 0.64 | 0.33 | 0.23 | 12 | 0.07 | 0.15 |
| 10 | 0.77 | 0.62 | 0.73 | 0.68 | 0.47 | 0.35 | 17 | 0.13 | 0.23 |
| 12 | 0.85 | 0.65 | 0.58 | 0.69 | 0.55 | 0.41 | 26 | 0.21 | 0.29 |
| 15 | 0.88 | 0.75 | 0.50 | 0.73 | 0.65 | 0.50 | 31 | 0.28 | 0.36 |
| 17 | 0.93 | 0.78 | 0.47 | 0.74 | 0.71 | 0.54 | 33 | 0.34 | 0.42 |
| 20 | 1.00 | 0.80 | 0.35 | 0.76 | 0.80 | 0.62 | 40 | 0.40 | 0.50 |

- P(score ≥ best θ ∧ reader wrong) is at most 0.03 at any level, for both checklist_v4_min and direct_v4.
- The v1/v3 gate still gives r=0 to 35% of questions at L20. The cause is the existence item answering "no" or being unresolved; see the v3 cause attribution. v4 does not change the observer output, so these zeros are identical to v3's.

### L20 per question (`dev_l20_per_question.csv`)

`(y/n)` is the resolved flag; "last item" is the attribute item.

| qid | question | item 1 → answer (resolved) | P_ans₁ | Rel₁ | last item → answer (resolved) | P_ans | Rel | min r | direct v4 |
|---|---|---|---|---|---|---|---|---|---|
| 00271801 | What is the plate to the right of the cabinet made of? | no (n) | 0.40 | 0.47 | metal (n) | 0.86 | 0.19 | 0.00 | 0.29 |
| 00434791 | What's the fence made of? | no (n) | 0.03 | 0.94 | metal (n) | 0.45 | 0.13 | 0.00 | 0.15 |
| 01316734 | Which shape is the plate to the left of the vase? | no (n) | 0.03 | 0.95 | square (y) | 0.57 | 0.75 | 0.00 | 0.37 |
| 01601414 | What is the shape of the sink that is in the bathroom? | no (n) | 0.03 | 0.96 | round (y) | 0.66 | 0.46 | 0.00 | 0.19 |
| 05827270 | What is the floor made of? | yes (y) | 0.99 | 1.00 | unknown (n) | 0.67 | 0.91 | 0.00 | 0.28 |
| 05831816 | What material makes up the utensil to the right of the food? | no (n) | 0.03 | 0.88 | metal (n) | 0.69 | 0.50 | 0.00 | 0.32 |
| 06572517 | What is the ceiling made of? | yes (y) | 1.00 | 1.00 | glass (n) | 0.68 | 0.81 | 0.00 | 0.79 |
| 07785408 | What is the chair to the right of the monitor made of? | yes (y) | 0.98 | 0.95 | unknown (n) | 0.67 | 0.19 | 0.00 | 0.13 |
| 07949045 | What is the shelf made of? | yes (y) | 0.98 | 0.96 | unknown (n) | 0.44 | 0.41 | 0.00 | 0.27 |
| 08254993 | What material is the mailbox made of? | no (n) | 0.19 | 0.48 | unknown (n) | 0.72 | 0.13 | 0.00 | 0.05 |
| 09488485 | What makes up the utensils in the box? | no (y) | 0.23 | 0.67 | unknown (n) | 0.87 | 0.04 | 0.00 | 0.04 |
| 10979790 | What is the sidewalk made of? | no (n) | 0.19 | 0.46 | unknown (n) | 0.73 | 0.14 | 0.00 | 0.07 |
| 12411054 | The car on the left has what color? | no (n) | 0.02 | 0.96 | gray (y) | 0.96 | 0.95 | 0.00 | 0.49 |
| 13127787 | Which material is the cup to the right of the other cup made of? | yes (y) | 1.00 | 0.98 | plastic (n) | 0.95 | 0.20 | 0.00 | 0.10 |
| 1390846 | What is the shelter made of? | no (n) | 0.02 | 0.96 | stone (y) | 0.86 | 0.93 | 0.00 | 0.79 |
| 13985783 | What material was used to make the bag that is to the right of the bowl? | no (n) | 0.04 | 0.85 | plastic (y) | 0.97 | 0.44 | 0.00 | 0.36 |
| 15172866 | What is the color of the train car that is to the left of the other train car? | yes (y) | 1.00 | 1.00 | yellow (y) | 0.93 | 0.72 | 0.00 | 0.62 |
| 15174086 | What material is the container to the right of the man? | no (y) | 0.52 | 0.47 | unknown (n) | 0.62 | 0.03 | 0.00 | 0.02 |
| 17184616 | What is the coffee table made of? | no (n) | 0.03 | 0.94 | glass (y) | 0.99 | 0.87 | 0.00 | 0.77 |
| 17888876 | What shape is the plate? | no (n) | 0.03 | 0.99 | square (y) | 0.95 | 0.88 | 0.00 | 0.65 |
| 19452749 | What is the bench to the left of the van made out of? | no (y) | 0.32 | 0.80 | concrete (n) | 0.56 | 0.26 | 0.00 | 0.36 |
| 18752666 | Which material is the shelf to the left of the tools made of? | no (y) | 0.02 | 0.90 | metal (y) | 0.79 | 0.51 | 0.02 | 0.46 |
| 07512969 | What is the house to the left of the air conditioner made of? | no (y) | 0.42 | 0.86 | wood (y) | 0.99 | 0.22 | 0.02 | 0.37 |
| 16756887 | What color does the large pot have? | no (y) | 0.06 | 0.93 | brown (y) | 0.76 | 0.77 | 0.06 | 0.72 |
| 15172776 | What color is the train to the left of the car? | yes (y) | 1.00 | 1.00 | yellow (y) | 0.91 | 0.53 | 0.15 | 0.46 |
| 18486270 | What material is the fence near the sheep made of? | yes (y) | 0.99 | 0.94 | wood (y) | 0.44 | 0.50 | 0.22 | 0.21 |
| 09887053 | Which shape is the mirror the street sign is reflected in? | no (y) | 0.91 | 0.25 | round (y) | 0.93 | 0.50 | 0.22 | 0.38 |
| 00943538 | What is the color of the clear sky? | no (y) | 0.53 | 0.52 | white (y) | 0.53 | 0.58 | 0.27 | 0.47 |
| 12454186 | What color do the shorts have? | yes (y) | 0.99 | 0.97 | khaki (y) | 0.32 | 0.91 | 0.29 | 0.26 |
| 10780377 | What color is the chair next to the other chair? | yes (y) | 1.00 | 0.99 | black (y) | 0.94 | 0.74 | 0.39 | 0.72 |
| 00947257 | What color is the bear that is to the right of the other bear? | yes (y) | 1.00 | 0.99 | black (y) | 0.97 | 0.84 | 0.41 | 0.79 |
| 141013932 | What color is the shirt? | yes (y) | 0.98 | 0.85 | white (y) | 0.74 | 0.67 | 0.49 | 0.55 |
| 08924895 | What shape do you think the chair is? | yes (y) | 0.97 | 0.97 | triangular (y) | 0.64 | 0.84 | 0.53 | 0.17 |
| 07833741 | The sky is of what color? | yes (y) | 1.00 | 0.98 | gray (y) | 0.64 | 0.84 | 0.54 | 0.66 |
| 06614339 | Which color are the pants? | yes (y) | 0.99 | 0.94 | orange, gray (y) | 0.73 | 0.77 | 0.57 | 0.70 |
| 11213423 | What color is the umbrella in the middle? | yes (y) | 1.00 | 1.00 | red yellow blue (y) | 0.63 | 0.92 | 0.58 | 0.48 |
| 17108363 | What color is the cup that is to the right of the other cup? | yes (y) | 1.00 | 0.99 | orange (y) | 0.98 | 0.66 | 0.60 | 0.64 |
| 03954206 | Which color is the curtain to the left of the lamp? | yes (y) | 1.00 | 1.00 | yellow (y) | 0.75 | 0.83 | 0.63 | 0.66 |
| 11213386 | What color is the umbrella on the right? | yes (y) | 0.99 | 0.94 | yellow and red (y) | 0.67 | 0.97 | 0.65 | 0.44 |
| 14848534 | What color is the bicycle? | yes (y) | 1.00 | 0.99 | white (y) | 0.82 | 0.83 | 0.68 | 0.64 |
| 03189870 | The mirror is of which shape? | yes (y) | 0.99 | 0.94 | round (y) | 0.82 | 0.83 | 0.69 | 0.30 |
| 06572516 | What material is the ceiling? | yes (y) | 1.00 | 1.00 | glass (y) | 0.80 | 0.87 | 0.69 | 0.70 |
| 06726789 | What is the color of the mask the catcher is wearing? | yes (y) | 0.99 | 0.98 | red (y) | 0.85 | 0.82 | 0.70 | 0.61 |
| 11619091 | What do you think is the color of this fence? | yes (y) | 1.00 | 0.99 | black (y) | 0.78 | 0.94 | 0.73 | 0.64 |
| 0778325 | Which color is the plane that is flying in the sky? | yes (y) | 1.00 | 1.00 | white and green (y) | 0.78 | 0.97 | 0.75 | 0.56 |
| 16820186 | The brown table is which shape? | yes (y) | 0.99 | 0.96 | round (y) | 0.98 | 0.77 | 0.76 | 0.53 |
| 11844870 | The container that is sitting on the dresser has what color? | yes (y) | 0.99 | 0.97 | black (y) | 0.96 | 0.80 | 0.77 | 0.71 |
| 18660252 | What is the shape of the tiles that look white? | yes (y) | 1.00 | 0.98 | rectangular (y) | 0.90 | 0.86 | 0.78 | 0.63 |
| 17891388 | What is the color of the candles on the table? | yes (y) | 0.99 | 0.98 | black (y) | 0.91 | 0.87 | 0.79 | 0.77 |
| 09672864 | What shape is the table? | yes (y) | 1.00 | 1.00 | triangular (y) | 0.92 | 0.89 | 0.82 | 0.73 |
| 09149294 | What is the shape of the flags that are on the rope? | yes (y) | 0.99 | 0.97 | triangular (y) | 0.99 | 0.86 | 0.86 | 0.57 |
| 011012564 | What shape does the food which is on the plate have? | yes (y) | 1.00 | 0.99 | triangular (y) | 0.99 | 0.87 | 0.86 | 0.64 |
| 13808031 | What's the ground made of? | yes (y) | 0.99 | 0.98 | brick (y) | 0.98 | 0.90 | 0.88 | 0.50 |
| 14276984 | What is the color of the tiles that are behind the toilet? | yes (y) | 0.99 | 0.98 | blue and white (y) | 0.93 | 0.95 | 0.88 | 0.80 |
| 03774985 | What is the shape of the dessert the toy car is on? | yes (y) | 1.00 | 0.99 | rectangular (y) | 0.97 | 0.95 | 0.93 | 0.46 |
| 19315625 | What shape is the window? | yes (y) | 1.00 | 1.00 | round (y) | 0.96 | 0.97 | 0.93 | 0.52 |
| 12416576 | What material is the sidewalk near the car? | yes (y) | 1.00 | 0.99 | brick (y) | 1.00 | 0.94 | 0.94 | 0.86 |
| 07111669 | What is the color of the coat that the girl is wearing? | yes (y) | 1.00 | 1.00 | pink (y) | 0.97 | 0.98 | 0.95 | 0.95 |
| 121044380 | Which color is the shirt? | yes (y) | 0.99 | 0.99 | gray (y) | 1.00 | 0.96 | 0.96 | 0.66 |
| 01420620 | Which color is the car? | yes (y) | 1.00 | 1.00 | white (y) | 1.00 | 0.99 | 0.99 | 0.96 |

## Notes and choices

- **P_ans.**
  - Definition: the probability of the first token whose bytes contain the first byte after `"answer"\s*:\s*"` in the concatenated token bytes.
  - If the opening quote and the value share a token, that token's probability is used unchanged.
  - No case or tokenisation variants are merged (§19.2).
  - If the token cannot be located, P_ans = 0. This never happened.
- **Rel.** The exact token "Yes" in the first generated token's top_logprobs (top 10); 0 if absent. The relevance prompt has the question substituted inline (`prompts/gqa_stop/relevance_v4.txt`), with no JSON format, temperature 0 and the v1 seed.
- **Gate.** The gate is v1's `effective_confidences` with confidence := P_ans × Rel. The P_ans-only and Rel-only variants use the same gate.
- **Records.** New namespaces `observer_v3_lp/`, `direct_lp/` and `relevance_v4/`. Each record stores the raw `logprobs` list, and the cache key includes the logprobs flag and plain-text mode. Older keys are unchanged: v1/v2/v3 replays still hit.
- **In-sample caveat.** Every θ above is chosen on the same 60 dev questions. Test performance is unknown until freeze and test, which wait for the user's decision.

## Files

- Code (new):
  - `src/milestone1/gqa_stop/logprob_v4.py`: token location, Rel, runner, gated scores.
  - `src/milestone1/gqa_stop/stages_v4.py`: pilot gates and distributions, devgrid, sensitivity, L20 table.
- Code (changed):
  - `client.py`: logprobs request and record, plain-text replies, inline template, multiple reuse sources.
  - `context.py`: reuse source list.
  - `stages_model.py`: `logprob_item` family and the §19.3 gate in dev infer.
  - `cli.py`: dispatch.
- Prompt: `prompts/gqa_stop/relevance_v4.txt`.
- Config: `configs/gqa_stop_v4.json`.
- Tests: `tests/test_gqa_stop_v4.py` (21 tests, including the fenced-JSON, merged-quote and Yes-absent fixtures); `tests/gqa_stop_helpers.py` gains a fake logprob transport.
- Outputs in `outputs/gqa_stop/visapp_gqa_v4/`: `dev_threshold_grid.csv`, `dev_stop_hist.csv`, `dev_components.csv`, `dev_comparison.csv`, `dev_fixed_curve.csv`, `dev_sensitivity.json`, `devgrid_summary.json`, `dev_l20_per_question.csv`, `level_table_dev.csv`, `pilot_summary.json`, `pilot/`.
