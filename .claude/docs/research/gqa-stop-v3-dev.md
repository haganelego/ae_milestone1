# gqa_stop v3 (plan §18 with §18.6): dev results

Date: 2026-10-09. Run: `configs/gqa_stop_v3.json` (family `item_checklist`), output `outputs/gqa_stop/visapp_gqa_v3/`.
Dev only. v3 was not frozen and no v3 test inference was run. Nothing is committed.

## Verdict

**The §18.4 rule says abort.** At dev accuracy ≥ 0.95, checklist_v3_min stops at mean Level 17.78, against 18.45 for fair v1. The difference is −0.67 [−1.80, +0.53], which is not significant.
- The logged mean aggregation does reach significance (17.22, −1.23 [−2.36, −0.07]). It is not the pre-registered method, so it does not decide the outcome.
- **Main cause: discrete confidence.** With one call at temperature 0, item confidences are quantised to about {0, 0.1, 0.15, 0.85, 1.0}. The stop score therefore only takes {0, 0.85, 1.0}, and the 101-point θ grid collapses to three settings. This is defect 6, which §18.6 leaves unaddressed.
- **Second cause: L20 zeros.** At L20, 35% of questions (21 of 60) still have a stop score of 0, so 53% of questions are forced to L20.

## A. textprobe and the visual subset

- **Method.** All 300 v1-adopted questions were answered once from text alone, using `prompts/gqa_stop/text_only.txt`. The run reproduces the scratch probe exactly on dev (100/100). The exclusion was made once.

| split | adopted | prior-answerable | visual subset | color | material | shape |
|---|---|---|---|---|---|---|
| dev | 100 | 40 | 60 | 15/40 | 18/40 | 7/20 |
| test | 200 | 78 | 122 | 21/80 | 36/80 | 21/40 |

The attribute columns show prior-answerable / adopted.

## B. Fair v1 (re-analysis of run v1 on the visual subset, cached records only)

- **Outputs.** Results are in `outputs/gqa_stop/visapp_gqa_v1/fair_subset/`. v1's original files are unchanged (md5 verified).
- **Dev subset.** Checklist-min θ=0.85 (acc 0.983, mean L 18.45), Direct θ=∞, Fixed-dev L20.

Test subset (122 questions):

| method | acc | mean L |
|---|---|---|
| Checklist-min | 0.959 | 17.62 |
| Direct / Fixed-dev | 1.000 | 20.00 |
| Fixed-15 / 17 / 19 | 0.910 / 0.926 / 0.959 | 15 / 17 / 19 |

- Checklist-min minus the hull: −0.007 [−0.049, 0.023].

## C. Prompt changes used by v3 final

- **planner_v3, revision 1/1** (the v3 dev revision; §11 reason: incomplete decomposition).
  - The last item must ask for the attribute using the attribute word.
  - No "If yes"/"If so", no pronouns, no references to other items, no counting items.
  - A two-item example was added.
  - The N=5 pilot had 3/20 plans without an attribute item and 8/20 flagged `item_not_standalone`.
- **observer_v3 A12** (format only, not a revision; plan §16 A12). It now has the reader-v2 answer-format sentence plus "yes/no items answer exactly yes or no".
- **Asymmetry.** Direct keeps the v1 prompt unchanged.

## D. Pilot (20 dev visual questions × L1/5/10/15/20; one call per item, temperature 0, v1 seed)

- **Initial valid rate.** planner, observer, reader and direct are all 1.0. Reader and direct were copied from v1.
- **Plans.** Attribute item 1.00, item_not_standalone 0.00, count item 0.05.

| L | reader acc | item-1 resolved (v3) | item-1 resolved (v1) | min r = 0 |
|---|---|---|---|---|
| 1 | 0.35 | 0.05 | 0.32 | 1.00 |
| 5 | 0.70 | 0.35 | 0.61 | 0.95 |
| 10 | 0.80 | 0.65 | 0.85 | 0.70 |
| 15 | 0.95 | 0.80 | – | 0.35 |
| 20 | 1.00 | 0.85 | 0.98 | 0.25 |

- **L20 zeros in the pilot.** 5 of 20, all cause (a).
- **Negative existence probe** (one call per image and level, v1 decoding):

| L | yes rate | resolved rate |
|---|---|---|
| 1 | 0.00 | 0.25 |
| 5 | 0.00 | 0.50 |
| 20 | 0.05 | 0.85 |

## E. Dev (60 visual questions × 20 levels) and devgrid (no freeze)

- **Calls.** 60 planner_v3 and 2,660 observer_v3 calls, all valid on the first attempt. Reader and direct were copied from v1.
- **Plans over 60.** Attribute item 1.00, item_not_standalone 0.00, count item 0.05. 50 plans have 2 items and 10 have 3.

Best feasible θ per method at dev accuracy ≥ 0.95:

| method | θ | acc | mean L | median | stop20 | stopped & wrong | acc − hull | mean L − fair v1 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| **checklist_v3_min** (pre-registered) | 1.00 | 0.983 | 17.78 | 20 | 0.53 | 1 | +0.025 | −0.67 [−1.80, +0.53] |
| checklist_v3_mean (logged) | 0.95 | 0.983 | 17.22 | 20 | 0.52 | 1 | +0.036 | −1.23 [−2.36, −0.07] |
| direct (v1 cache) | ∞ | 1.000 | 20.00 | 20 | 1.00 | 0 | 0 | +1.55 [0.70, 2.54] |
| fair v1 Checklist-min | 0.85 | 0.983 | 18.45 | 20 | 0.75 | 1 | +0.012 | – |
| subset Fixed-15 | – | 0.883 | 15 | 15 | 0 | 7 | −0.023 | – |
| subset Fixed-dev (L20) | – | 1.000 | 20 | 20 | 1 | 0 | 0 | – |

- "acc − hull" uses the dev-subset Fixed-t upper hull.
- The CI is a paired image-cluster bootstrap with 2000 reps. θ was chosen on the same dev questions, so the numbers are in-sample.

**θ collapse (checklist_v3_min).**
- Any θ ≤ 0.85 gives acc 0.900 / mean L 13.85 / 6 stopped-and-wrong.
- Any θ from 0.86 to 1.00 gives 0.983 / 17.78 / 1.
- Over all 1,200 dev rows the stop score takes 0 (846 rows), 0.85 (202) and 1.0 (152).

**Stop histograms** (level: stopped / wrong):
- checklist_v3_min θ1.0: L7 1/0, L10 1/0, L12 4/0, L13 2/1, L15 6/0, L16 4/0, L17 1/0, L18 6/0, L19 3/0, L20 32/0.
- checklist_v3_mean θ0.95: L7 2/0, L10 2/0, L11 1/0, L12 5/0, L13 2/1, L14 1/0, L15 4/0, L16 4/0, L17 1/0, L18 6/0, L19 1/0, L20 31/0.
- fair v1 θ0.85: 15 stops at L4–L19 with 1 wrong; L20 45/0.

**Component trends:**

| L | reader acc | item-1 resolved | min = 0 | mean min | mean direct conf |
|---|---|---|---|---|---|
| 1 | 0.35 | 0.05 | 0.98 | 0.01 | 0.58 |
| 5 | 0.68 | 0.33 | 0.93 | 0.06 | 0.69 |
| 10 | 0.77 | 0.62 | 0.73 | 0.23 | 0.86 |
| 15 | 0.88 | 0.75 | 0.50 | 0.46 | 0.96 |
| 20 | 1.00 | 0.80 | 0.35 | 0.62 | 0.98 |

- P(min ≥ θ* ∧ reader wrong) is at most 0.02 at every level.

**L20 zeros on dev.** 21 of 60 questions; full lists are in `dev_l20_zero_causes.json`.

| primary cause | count |
|---|---|
| (a) existence item "no" or unresolved for the GQA target | 15 |
| (c) planner defect | 1 |
| other: a later attribute item unresolved/"unknown" | 5 |
| (b) wording variance | n/a (it only applies to sampled S_agree) |

- Cause (a) examples, existence item answered "no": utensils in the box, the car on the left, the plate to the right of the cabinet, the shelter, the coffee table, the mailbox, the sidewalk.
- These targets are visible to the reader at L20; the reader is right on all of them by construction of the screening. The existence item is therefore stricter than the reader. It sometimes fails on stuff-like or partly visible targets and on relational existence ("Is there a plate to the right of the cabinet?").

### L20 per question (dev, 60 visual questions; `dev_l20_per_question.csv`)

At L20 the existence item answered "no" for 20 of 60 GQA targets: 12 of them unresolved and 8 resolved as "no". The reader is correct on all 60 by construction. The causes of a zero stop score at L20 are 15 (a), 5 other and 1 (c), as summarised above.

| qid | question | existence item → answer (resolved) | min r | reader correct | cause |
|---|---|---|---|---|---|
| 00271801 | What is the plate to the right of the cabinet made of? | Is there a plate to the right of the cabinet in the image? → no (no) | 0.00 | True | a |
| 00434791 | What's the fence made of? | Is there a fence in the image? → no (no) | 0.00 | True | a |
| 01316734 | Which shape is the plate to the left of the vase? | Is there a plate to the left of the vase in the image? → no (no) | 0.00 | True | a |
| 01601414 | What is the shape of the sink that is in the bathroom? | Is there a sink in the image? → no (no) | 0.00 | True | a |
| 05827270 | What is the floor made of? | Is there a floor in the image? → yes (yes) | 0.00 | True | other |
| 05831816 | What material makes up the utensil to the right of the food? | Is there a utensil to the right of the food in the image? → no (no) | 0.00 | True | a |
| 06572517 | What is the ceiling made of? | Is there a ceiling in the image? → yes (yes) | 0.00 | True | other |
| 07785408 | What is the chair to the right of the monitor made of? | Is there a chair in the image? → yes (yes) | 0.00 | True | other |
| 07949045 | What is the shelf made of? | Is there a shelf in the image? → yes (yes) | 0.00 | True | other |
| 08254993 | What material is the mailbox made of? | Is there a mailbox in the image? → no (no) | 0.00 | True | a |
| 09488485 | What makes up the utensils in the box? | Is there a box with utensils in the image? → no (yes) | 0.00 | True | a |
| 10979790 | What is the sidewalk made of? | Is there a sidewalk in the image? → no (no) | 0.00 | True | a |
| 12411054 | The car on the left has what color? | Is there a car on the left in the image? → no (no) | 0.00 | True | a |
| 13127787 | Which material is the cup to the right of the other cup made of? | Is there a cup in the image? → yes (yes) | 0.00 | True | other |
| 1390846 | What is the shelter made of? | Is there a shelter in the image? → no (no) | 0.00 | True | a |
| 13985783 | What material was used to make the bag that is to the right of the bowl? | Is there a bag to the right of the bowl in the image? → no (no) | 0.00 | True | a |
| 15172866 | What is the color of the train car that is to the left of the other train car? | Is there a train in the image? → yes (yes) | 0.00 | True | c |
| 15174086 | What material is the container to the right of the man? | Is there a container to the right of the man in the image? → no (yes) | 0.00 | True | a |
| 17184616 | What is the coffee table made of? | Is there a coffee table in the image? → no (no) | 0.00 | True | a |
| 17888876 | What shape is the plate? | Is there a plate in the image? → no (no) | 0.00 | True | a |
| 19452749 | What is the bench to the left of the van made out of? | Is there a bench to the left of the van in the image? → no (yes) | 0.00 | True | a |
| 00943538 | What is the color of the clear sky? | Is there a clear sky in the image? → no (yes) | 0.85 | True |  |
| 06572516 | What material is the ceiling? | Is there a ceiling in the image? → yes (yes) | 0.85 | True |  |
| 06726789 | What is the color of the mask the catcher is wearing? | Is there a mask worn by a catcher in the image? → yes (yes) | 0.85 | True |  |
| 07512969 | What is the house to the left of the air conditioner made of? | Is there a house in the image? → no (yes) | 0.85 | True |  |
| 13808031 | What's the ground made of? | Is there a ground in the image? → yes (yes) | 0.85 | True |  |
| 14848534 | What color is the bicycle? | Is there a bicycle in the image? → yes (yes) | 0.85 | True |  |
| 16756887 | What color does the large pot have? | Is there a large pot in the image? → no (yes) | 0.85 | True |  |
| 16820186 | The brown table is which shape? | Is there a brown table in the image? → yes (yes) | 0.85 | True |  |
| 18486270 | What material is the fence near the sheep made of? | Is there a fence near the sheep in the image? → yes (yes) | 0.85 | True |  |
| 18752666 | Which material is the shelf to the left of the tools made of? | Is there a shelf to the left of the tools in the image? → no (yes) | 0.85 | True |  |
| 00947257 | What color is the bear that is to the right of the other bear? | Is there a bear in the image? → yes (yes) | 1.00 | True |  |
| 011012564 | What shape does the food which is on the plate have? | Is there a food on the plate in the image? → yes (yes) | 1.00 | True |  |
| 01420620 | Which color is the car? | Is there a car in the image? → yes (yes) | 1.00 | True |  |
| 03189870 | The mirror is of which shape? | Is there a mirror in the image? → yes (yes) | 1.00 | True |  |
| 03774985 | What is the shape of the dessert the toy car is on? | Is there a dessert in the image? → yes (yes) | 1.00 | True |  |
| 03954206 | Which color is the curtain to the left of the lamp? | Is there a curtain in the image? → yes (yes) | 1.00 | True |  |
| 06614339 | Which color are the pants? | Is there a pair of pants in the image? → yes (yes) | 1.00 | True |  |
| 07111669 | What is the color of the coat that the girl is wearing? | Is there a girl in the image? → yes (yes) | 1.00 | True |  |
| 0778325 | Which color is the plane that is flying in the sky? | Is there a plane flying in the sky in the image? → yes (yes) | 1.00 | True |  |
| 07833741 | The sky is of what color? | Is there a sky in the image? → yes (yes) | 1.00 | True |  |
| 08924895 | What shape do you think the chair is? | Is there a chair in the image? → yes (yes) | 1.00 | True |  |
| 09149294 | What is the shape of the flags that are on the rope? | Is there a flag on a rope in the image? → yes (yes) | 1.00 | True |  |
| 09672864 | What shape is the table? | Is there a table in the image? → yes (yes) | 1.00 | True |  |
| 09887053 | Which shape is the mirror the street sign is reflected in? | Is there a mirror reflecting a street sign in the image? → no (yes) | 1.00 | True |  |
| 10780377 | What color is the chair next to the other chair? | Is there a chair in the image? → yes (yes) | 1.00 | True |  |
| 11213386 | What color is the umbrella on the right? | Is there an umbrella on the right in the image? → yes (yes) | 1.00 | True |  |
| 11213423 | What color is the umbrella in the middle? | Is there an umbrella in the image? → yes (yes) | 1.00 | True |  |
| 11619091 | What do you think is the color of this fence? | Is there a fence in the image? → yes (yes) | 1.00 | True |  |
| 11844870 | The container that is sitting on the dresser has what color? | Is there a container sitting on the dresser in the image? → yes (yes) | 1.00 | True |  |
| 121044380 | Which color is the shirt? | Is there a shirt in the image? → yes (yes) | 1.00 | True |  |
| 12416576 | What material is the sidewalk near the car? | Is there a sidewalk near the car in the image? → yes (yes) | 1.00 | True |  |
| 12454186 | What color do the shorts have? | Is there a pair of shorts in the image? → yes (yes) | 1.00 | True |  |
| 141013932 | What color is the shirt? | Is there a shirt in the image? → yes (yes) | 1.00 | True |  |
| 14276984 | What is the color of the tiles that are behind the toilet? | Is there a tile wall behind the toilet in the image? → yes (yes) | 1.00 | True |  |
| 15172776 | What color is the train to the left of the car? | Is there a train in the image? → yes (yes) | 1.00 | True |  |
| 17108363 | What color is the cup that is to the right of the other cup? | Is there a cup in the image? → yes (yes) | 1.00 | True |  |
| 17891388 | What is the color of the candles on the table? | Is there a candle on the table in the image? → yes (yes) | 1.00 | True |  |
| 18660252 | What is the shape of the tiles that look white? | Is there a tile in the image? → yes (yes) | 1.00 | True |  |
| 19315625 | What shape is the window? | Is there a window in the image? → yes (yes) | 1.00 | True |  |

## Appendix: Exploratory (withdrawn design §18.6), the N=5 sampling run

- **Status.** The N=5 pilot, dev inference (~19.5k calls) and devgrid had completed before the stop instruction arrived, so nothing was stopped mid-run.
- **Outputs.** They are in `outputs/gqa_stop/visapp_gqa_v3/superseded_n5/`, together with the rev-0 plans.
- **Records.** The `*_s{i}` records (13,800 observer, 6,000 direct-sampled, 60 negative-probe) were moved to their own namespace, `superseded_n5/records/`, on 2026-10-09. They never mix with the N=1 records.
- **Planner revision 0.** It was used only by this run. Its plans had 7 of 60 with no attribute item and 20 of 60 flagged `item_not_standalone`. Revision 1, used by v3 final, has 0 and 0; the count-item rate is 0.05. This is a known issue for the user.
- **Exploratory numbers.** These come from the withdrawn design, so they are not v3 results. The best dev θ at accuracy ≥ 0.95, as mean L with the difference to fair v1 and its CI:
  - checklist min S_agree θ0.6: 16.05, −2.40 [−4.02, −0.85].
  - checklist min S_mean θ0.75: 16.87, −1.58 [−3.03, −0.13].
  - Direct-sampled S_mean θ1.0: 14.87, −3.58 [−4.95, −2.25].
  - Direct-sampled S_agree: θ=∞.

## Deviations and notes

- **Dependency gate.** v3 uses v1's `effective_confidences`, so a child scores 0 unless every ancestor is resolved and its answer is not "unknown". The orchestrator's later wording, "parent resolved and r_parent > 0", changes no row on dev, and it cannot change checklist_v3_min: the min includes the parent itself, so r_parent = 0 already makes the min 0.

- The planner revision also forbids counting items. That was not in the instruction, but a "How many umbrellas…" item appeared in the N=5 pilot. One count item remains (count item rate 0.05).
- **Rules I fixed in code (§18 did not specify them):**
  - `last_item_has_attribute` uses the regex color/colour/material/made of/shape/shaped and is logged only.
  - The L20 cause order is (a), then (c), then other. Flags are non-exclusive; one question has both (a) and (c).
- **Asymmetry.** Direct uses the v1 prompt without A12's format sentence, while observer_v3 has it.

## Files

- Code:
  - `src/milestone1/gqa_stop/item_checklist.py` (new): N=1 runner using v1's `effective_confidences` and `aggregate`.
  - `src/milestone1/gqa_stop/pilot_v3.py`: `item_metrics`, `l20_zero_causes`, and a negative probe without sampling.
  - `src/milestone1/gqa_stop/devgrid_v3.py`: method columns, components and the abort method are now parameters.
  - `src/milestone1/gqa_stop/stages_v3.py`: both v3 families; devgrid refuses an inference run of another family.
  - `src/milestone1/gqa_stop/stages_model.py`: wiring for the `item_checklist` family.
  - `src/milestone1/gqa_stop/sampled.py`: the `last_item_has_attribute` flag. The sampling code is kept for its tests only.
- Prompts: `prompts/gqa_stop/planner_v3.txt` (revision 1) and `observer_v3.txt` (A12).
- Config: `configs/gqa_stop_v3.json`, with no sampling keys and methods `checklist_v3_min` plus logged `checklist_v3_mean`.
- Tests: `tests/test_gqa_stop_v3_item.py` (14 tests). The constants in `tests/test_gqa_stop_v3.py` are pinned to the withdrawn family. `tests/gqa_stop_helpers.py` gains `run_v1_pipeline`.
- Outputs:
  - `outputs/gqa_stop/visapp_gqa_v3/`: `dev_threshold_grid.csv`, `dev_stop_hist.csv`, `dev_components.csv`, `dev_comparison.csv`, `dev_fixed_curve.csv`, `devgrid_summary.json`, `dev_l20_zero_causes.json`, `level_table_dev.csv`, `pilot_summary.json`, `pilot/`.
  - `outputs/gqa_stop/visapp_gqa_v1/fair_subset/`.
