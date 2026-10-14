# Probe: does decomposition (P_min) diverge from A1 on multi-object GQA questions? (2026-10-15)

Status: **exploratory, not pre-registered.** 30 GQA val_balanced questions, 6 multi-object types × 5. All numbers are descriptive and in-sample. Nothing here is a confirmatory result.

Run: `configs/gqa_stop_probe_multi_rel.json` (a copy of `configs/gqa_stop_v27.json` with run_id `probe_multi_rel`, new data/output roots and an identity check on 10 instead of 40 questions). Output `outputs/gqa_stop/probe_multi_rel/`, data `data/processed/gqa_stop/probe_multi_rel/`. Model qwen3.5:9b (digest 6488c96fa5fa), A13 clean decoding, think=false, top_logprobs 10, 20 levels. Prompts answer_only, planner_v3 and observer_v3 unchanged. No module under `src/` was changed. `visapp_gqa_v27/` and `visapp_gqa_v22*/` were only read.

## 0. Question and bottom line

On v27 test (query × attr, single target object), P_min and A1 are nearly the same signal. The user's hypothesis: this is because the target is a single object, and on questions that need several objects or relations, decomposition might diverge from A1.

| statistic (per question × level) | v27 test (886 q) | probe, all 30 q |
|---|---|---|
| corr(pans_min, A1 score) | 0.933 | 0.741 |
| last item binds the min | 0.887 | 0.840 |
| last-item answer = A1 answer | 0.944 | 0.803 |
| last item resolved-ok = A1 resolved-ok | 0.959 | 0.902 |
| P_min and A1 stop at a different level, same θ 0.80 | 0.257 | 0.400 |

- **Yes, P_min diverges from A1 on multi-object questions, but only on some types.** On query × rel (relO, relS) and verify × rel, it is as close to A1 as on v27 test, or closer. It diverges on choose × rel, logical and compare.
- **Where it diverges, it is worse.** At the same θ 0.80, P_min and A1 stop differently on 12 of 30 questions. A1 is right and P_min wrong on 8; P_min is right and A1 wrong on 1; 3 are ties. At the in-sample argmax θ 0.55 for both, the count is 4 for A1 and 0 for P_min. On v27 test the same split is even (36 vs 34 at θ 0.80).
- **The cause is the plan, not a useful check.** The divergence comes from two things. First, planner_v3 rewrites the last item into a different question on compare and logical (its attribute rule). Second, existence items for the second object are less confident and less stable than the answer to the whole question, so they hold the min below θ. The planner never adds a check that would veto a wrong answer, and a resolved "no" on an existence item passes the gate anyway.

## 1. Selection (`scripts/gqa_probe_multi_rel_select.py`, `selection.jsonl`)

- **Source.** `/home/kanaoka/usr/dataset/gqa/val_balanced_questions.json`, the file v27 dev used. Not testdev.
- **Filters.** Every type requires ≥ 2 object references in the program, counting select, relate, verify rel and choose rel ops. Referents that do not exist, written "(-)", are included.

| type | filter on GQA `types` / program | pool |
|---|---|---|
| a_relO | detailed relO, last op query | 11,993 |
| b_relS | detailed relS, last op query | 13,268 |
| c_choose_rel | structural choose, semantic rel | 5,754 |
| d_verify_rel | structural verify, semantic rel | 15,602 |
| e_logical | structural logical, has an and / or op, ≥ 2 select ops | 12,685 |
| f_compare | structural compare, ≥ 2 select ops | 3,688 |

- **Sampling.** Seed 20261008: sorted ids, one `random.Random` shuffle per type in the order a to f, first 5 per type.
- **Exclusions.** At most 1 question per image across the probe, so there are 30 images. The 222 v27 dev images are excluded by id and by source SHA-256.
- **Effect of the ≥ 2-select rule.** It drops single-object logical questions ("Is the shirt red and long?") and the same-category compare types (sameAnimals, diffAnimals, sameGender).
- **Labels.** Answerable means the reader is correct at L20, with the v27 normalisation and exact match. Screening gave 23 answerable and 7 unanswerable.

| type | answerable / unanswerable | GQA detailed types drawn |
|---|---|---|
| a_relO | 2 / 3 | relO ×5 |
| b_relS | 3 / 2 | relS ×5 |
| c_choose_rel | 5 / 0 | relChooser ×5 |
| d_verify_rel | 5 / 0 | relVerify, existRelS, relVerifyCo ×3 |
| e_logical | 4 / 1 | existThatOr, existOrC ×2, existOr, existAttrOrC |
| f_compare | 4 / 1 | twoSameC, twoSameMaterialC, twoDifferent, twoCommon, twoSameMaterial |

The 7 unanswerable questions are mostly open-name label mismatches or GQA ambiguity:
- forest vs "background"
- bag vs "nothing"
- woman vs "tennis player"
- sign vs "flowers"
- letters vs "text"
- no vs "stove"
- color vs "blue"

## 2. Run facts

- **Environment.** Env check passed on 2026-10-14 17:34 UTC: no model resident before, then qwen3.5:9b alone at 100% GPU, and the ollama log showed "CLIP using CUDA0 backend".
- **Determinism and identity.** The determinism gate passed: 80 argmax calls, 618 positions, 0 violations. Both repeat pairs were byte-identical (planner_v3 text, reader_lp L20). The reader_lp identity check gave 30 of 30 raw replies byte-identical, on 10 questions × L1/L10/L20.
- **Records.** 2,700 records, all valid: reader_lp 600, plans_v3 30, observer_v3_lp 1,440 (72 items × 20 levels), direct 600, plain reader 30.
- **Plans.** 30 of 30 valid. 24 were valid on the first attempt and 6 after one schema repair (3 logical, 3 compare). v27 dev needed 1 repair in 300.
- **Audit.** 0 records flagged, max non-generation time 0.45 s, 0 argmax violations in 33,927 positions.
- **Calls and time.** Inference took 2,626 new calls in 3 foreground chunks of ≤ 585 s. That is 27 min at 0.62 s per call.

```
uv run python scripts/gqa_probe_multi_rel_select.py
C=configs/gqa_stop_probe_multi_rel.json
uv run python -m milestone1.gqa_stop.cli preflight --config $C
uv run python -m milestone1.gqa_stop.cli v27-envcheck --config $C
uv run python -m milestone1.gqa_stop.cli v27-detgate --config $C
uv run python -m milestone1.gqa_stop.cli v27-identity --config $C
uv run python -m milestone1.gqa_stop.cli v27-screen --split dev --config $C
timeout 585 uv run python -m milestone1.gqa_stop.cli v27-infer --split dev --max-seconds 540 --config $C   # x3
uv run python -m milestone1.gqa_stop.cli v27-audit --split dev --config $C
uv run python -m milestone1.gqa_stop.cli v27-table --split dev --config $C
uv run python -m milestone1.gqa_stop.cli v27-evaluate --split dev --config $C   # writes eval/dev/ in the probe dir
uv run python scripts/gqa_probe_multi_rel_analyze.py   # analysis/agreement.csv, per_question.csv, divergence.json
uv run python scripts/gqa_probe_multi_rel_tables.py    # θ tables below
```

The probe is run as the v27 "dev" split, because the CLI reads `questions_dev.jsonl`. The `attribute` column holds the probe type label, so the evaluator's per-attribute outputs are per type.

## 3. Plans (verbatim; `[deps]` after the item id)

**a_relO.** The planner checks the anchor in 4 of 5 plans and the related object (the answer) in none. The last item is a verbatim copy of the question in 3 plans and a close paraphrase in 2.
```
19404920 "What is the man wearing?" -> hat
  c1 []    Is there a man in the image?
  c2 [c1]  What is the man wearing?
16403783 "Where are the trees?" -> forest
  c1 []    Is there a tree in the image?
  c2 [c1]  Where is the tree in the image?
19687543 "What is the woman holding?" -> bag
  c1 []    Is there a woman in the image?
  c2 [c1]  What is the woman holding?
09212473 "The crowd is watching who?" -> woman
  c1 []    Is there a crowd in the image?
  c2 [c1]  Who is the crowd watching?
00259214 "What is the sign hanging off of?" -> pole
  c1 []    Is there a sign hanging off of something in the image?
  c2 [c1]  What is the sign hanging off of?
```
**b_relS.** The anchor is checked in 4 plans. The related object is checked in 1, through "container full of donuts". The last item is a copy of the question in 4 plans. In 1 it is hijacked by the attribute rule ("... made of?").
```
14863272 "What is full of donuts?" -> tray
  c1 []    Is there a container full of donuts in the image?
  c2 [c1]  What is the container full of donuts made of?
13365742 "What's on the wire?" -> sign
  c1 []    Is there a wire in the image?
  c2 [c1]  What is on the wire?
06943437 "What's on the oven?" -> kettle
  c1 []    Is there an oven in the image?
  c2 [c1]  What is on the oven?
08900610 "What is on the ski on the left?" -> letters
  c1 []    Is there a ski on the left in the image?
  c2 [c1]  What is on the ski on the left?
11464769 "What is in front of the green building?" -> fence
  c1 []    Is there a green building in the image?
  c2 [c1]  What is in front of the green building?
```
**c_choose_rel.** This is the type with the most real decomposition. Both objects are checked as separate parallel items in 3 plans. One 2-hop plan checks the anchor and the intermediate object but not the subject (counter). The last item is a copy of the question in 3 plans. One plan rewrites it to a binary yes/no question. One plan has a garbled item.
```
02584384 "Is the counter above or below the cabinet the microwave is to the left of?" -> below
  c1 []      Is there a microwave in the image?
  c2 [c1]    Is there a cabinet to the left of the microwave in the image?
  c3 [c2]    Is the counter above or below the cabinet to the left of the microwave in the image?
19419643 "Are the ripe oranges to the left or to the right of the bag made of plastic?" -> right
  c1 []      Is there a bag made of plastic in the image?
  c2 []      Is there a ripe orange in the image?
  c3 [c1,c2] Are the ripe oranges to the left or to the right of the bag made of plastic?
19151786 "Is the bench to the right or to the left of the blue vehicle?" -> left
  c1 []      Is there a bench in the image?
  c2 []      Is there a blue vehicle in the image?
  c3 [c1,c2] Is the bench to the right or to the left of the blue vehicle?
05480434 "Is the elephant to the left or to the right of the car?" -> right
  c1 []      Is there an elephant in the image?
  c2 []      Is there a car in the image?
  c3 [c1,c2] Is the elephant to the left of the car?
16880810 "Is the pillow to the left or to the right of the woman on the right?" -> left
  c1 []      Is there a woman on the right in the image?
  c2 [c1]    Is there a pillow to the left or to the right of the woman on the right?
```
**d_verify_rel.** Item 1 is the whole proposition in existence form in 4 plans; plan 07883808 checks the knife only. So in 4 plans, items 1 and 2 ask the same thing. The last item is a copy of the question in 4 plans, and in the fifth the pronoun is resolved.
```
16654472 "Is the ottoman to the left of a bed?" -> yes
  c1 []    Is there an ottoman to the left of a bed in the image?
  c2 [c1]  Is the ottoman to the left of a bed?
13877235 "Are there any books on the floor?" -> yes
  c1 []    Is there a book on the floor in the image?
  c2 [c1]  Are there any books on the floor?
00777462 "Is the bird on the rock watching a boar?" -> no
  c1 []    Is there a bird on a rock in the image?
  c2 [c1]  Is the bird on the rock watching a boar?
07883808 "Is the knife on a cutting board?" -> no
  c1 []    Is there a knife in the image?
  c2 [c1]  Is the knife on a cutting board?
09249815 "Is he carrying a surfboard?" -> no
  c1 []    Is there a person carrying a surfboard in the image?
  c2 [c1]  Is the person carrying a surfboard?
```
**e_logical.** In 3 plans, each disjunct is checked as a separate item and the last item copies the question. Two plans are nonsense: the attribute rule turns "or" existence questions into "What color is the door or car?". Both of those needed a repair.
```
121053010 "Do you see any doors or cars that are white?" -> yes
  c1 []      Is there a door or a car in the image?
  c2 [c1]    What color is the door or car?
03848528 "Are there any hairbrushes or bottle caps?" -> no
  c1 []      Is there a hairbrush or a bottle cap in the image?
  c2 [c1]    What color is the hairbrush or bottle cap?
0598390 "Are there either pillows or trays?" -> yes
  c1 []      Is there a pillow in the image?
  c2 []      Is there a tray in the image?
  c3 [c1,c2] Are there either pillows or trays?
08206123 "Are there any glasses or stoves in the scene?" -> no
  c1 []      Is there a glass in the image?
  c2 []      Is there a stove in the image?
  c3 [c1,c2] Are there any glasses or stoves in the image?
061043477 "Are there unpeeled bananas or apples?" -> no
  c1 []      Is there an unpeeled banana in the image?
  c2 []      Is there an unpeeled apple in the image?
  c3 [c1,c2] Are there unpeeled bananas or apples in the image?
```
**f_compare.** Both objects are checked in all 5 plans. The last item asks one object's attribute in 3 plans, the wrong attribute for both objects in 1 plan, and copies the question in 1 plan. The comparison itself is never asked in 4 of 5 plans.
```
00768430 "Do the van and the hydrant have the same color?" -> no
  c1 []      Is there a van in the image?
  c2 []      Is there a hydrant in the image?
  c3 [c1]    What color is the van?
04727485 "Are the barrier and the pole made of the same material?" -> no
  c1 []      Is there a barrier in the image?
  c2 []      Is there a pole in the image?
  c3 [c1]    What material is the barrier made of?
13831520 "Is the color of the scarf different than the roof?" -> yes
  c1 []      Is there a scarf in the image?
  c2 []      Is there a roof in the image?
  c3 [c1]    What color is the scarf?
17401329 "What is common to the umbrella and the scarf?" -> color
  c1 []      Is there an umbrella in the image?
  c2 []      Is there a scarf in the image?
  c3 [c1,c2] What material are the umbrella and the scarf made of?
1129358 "Is the frame made out of the same material as the bookcase?" -> yes
  c1 []      Is there a frame in the image?
  c2 []      Is there a bookcase in the image?
  c3 [c1,c2] Is the frame made out of the same material as the bookcase?
```

**Plan quality summary** (judged by reading):

| type | items 2 / 3 | 2nd object or relation checked as its own item | last item is a copy or paraphrase of the question | last item is a different question |
|---|---|---|---|---|
| a_relO | 5 / 0 | 0 / 5 (relation in 1) | 5 | 0 |
| b_relS | 5 / 0 | 1 / 5 | 4 | 1 (material) |
| c_choose_rel | 1 / 4 | 3 / 5 (+1 hop-1 only) | 3 | 2 (binary rewrite, garbled) |
| d_verify_rel | 5 / 0 | 0 / 5 separately (item 1 = whole proposition in 4) | 5 | 0 |
| e_logical | 2 / 3 | 3 / 5 (each disjunct) | 3 | 2 (color hijack) |
| f_compare | 0 / 5 | 5 / 5 | 1 | 4 (one-object attribute, wrong attribute) |

## 4. Agreement between P_min and A1 (`analysis/agreement.csv`)

Definitions per (question, level) row. They reproduce the v27 test figures quoted by the team lead (0.93 / 0.89 / 0.94).
- **own_last.** The last item's P_ans if the item is valid, resolved and its answer is not "unknown", and 0 otherwise. This uses the item's own gate and ignores depends_on.
- **Last binds.** pans_min = own_last.
- **Earlier binds.** pans_min < own_last. Either an earlier item, or a failed dependency, holds the min below the last item.

| set | rows | corr(pans_min, A1) | last binds | last binds, own_last > 0 | last ans = A1 ans | last ok = A1 ok | A3 = A1 ok | mean pans_min / A1 |
|---|---|---|---|---|---|---|---|---|
| **v27 test** | 17,720 | **0.933** | **0.887** | 0.791 | **0.944** | 0.959 | 0.950 | 0.372 / 0.402 |
| probe, all | 600 | 0.741 | 0.840 | 0.703 | 0.803 | 0.902 | 0.882 | 0.324 / 0.435 |
| a_relO | 100 | 0.972 | 1.000 | 1.000 | 0.950 | 0.990 | 0.990 | 0.299 / 0.275 |
| b_relS | 100 | 0.795 | 1.000 | 1.000 | 0.840 | 0.930 | 0.930 | 0.257 / 0.252 |
| c_choose_rel | 100 | 0.820 | 0.620 | 0.433 | 0.720 | 0.980 | 0.920 | 0.354 / 0.489 |
| d_verify_rel | 100 | 0.968 | 0.720 | 0.641 | 1.000 | 1.000 | 0.980 | 0.583 / 0.610 |
| e_logical | 100 | 0.418 | 0.700 | 0.400 | 0.650 | 0.650 | 0.610 | 0.298 / 0.676 |
| f_compare | 100 | 0.731 | 1.000 | 1.000 | 0.660 | 0.860 | 0.860 | 0.156 / 0.309 |

Divergence of the stop decision, per question (`analysis/divergence.json`):

| quantity | v27 test | probe |
|---|---|---|
| P_min and A1 stop at a different level (or one never), same θ 0.80 | 0.257 | 0.400 (12 / 30) |
| same, each at its v27 θ* (P_min 0.80, A1 0.75) | 0.397 | 0.433 |
| questions with an earlier item binding at ≥ 1 level | 0.287 | 0.400 (12 / 30) |

- **relO and relS stay A1-like.** The last item is the question itself, so it binds 100% of the time. The b_relS correlation (0.80) is lower only because of the one plan hijacked to "made of?".
- **verify × rel is A1-like in score.** The correlation is 0.97 and the last answer equals A1's answer 100% of the time. Item 1 binds on 28% of rows, but item 1 is the same proposition, so the min hardly moves.
- **choose × rel diverges through the existence items.** Earlier items bind on 38% of rows, and their scores are lower than the relation answer.
- **logical diverges the most.** The correlation is 0.42. The causes are the two hijacked plans and the "no" disjunct items that bind the min.
- **compare diverges through the last item.** The last item always binds, but it answers a different question: one object's color or material.
- **Existence items cannot veto.** The gate only zeros "unknown" or unresolved items. A resolved "no" on a non-last "Is there …?" item passes. This happens on 33% of probe rows and 13% of v27 test rows. So an existence item can only lower the min through its confidence, never through its content.

## 5. θ sweep (strict; `eval/dev/sweep.csv`, in-sample, n = 23 / 7)

Each cell is balanced / answerable acc / unanswerable correct / mean stop level (never = 20).

| θ | B1 | P_min | P_mean | A1 | A2 |
|---|---|---|---|---|---|
| 0.00 | 0.30 / 0.61 / 0.00 / 1.0 | 0.30 / 0.61 / 0.00 / 1.0 | 0.30 / 0.61 / 0.00 / 1.0 | 0.30 / 0.61 / 0.00 / 1.0 | 0.30 / 0.61 / 0.00 / 1.0 |
| 0.05 | 0.30 / 0.61 / 0.00 / 1.0 | 0.63 / 0.70 / 0.57 / 10.3 | 0.35 / 0.70 / 0.00 / 2.8 | 0.51 / 0.74 / 0.29 / 7.9 | 0.30 / 0.61 / 0.00 / 1.0 |
| 0.10 | 0.30 / 0.61 / 0.00 / 1.0 | 0.63 / 0.70 / 0.57 / 10.3 | 0.35 / 0.70 / 0.00 / 2.8 | 0.51 / 0.74 / 0.29 / 7.9 | 0.30 / 0.61 / 0.00 / 1.0 |
| 0.15 | 0.30 / 0.61 / 0.00 / 1.1 | 0.63 / 0.70 / 0.57 / 10.3 | 0.33 / 0.65 / 0.00 / 3.0 | 0.51 / 0.74 / 0.29 / 7.9 | 0.30 / 0.61 / 0.00 / 1.0 |
| 0.20 | 0.30 / 0.61 / 0.00 / 1.2 | 0.63 / 0.70 / 0.57 / 10.3 | 0.33 / 0.65 / 0.00 / 3.1 | 0.51 / 0.74 / 0.29 / 7.9 | 0.30 / 0.61 / 0.00 / 1.0 |
| 0.25 | 0.30 / 0.61 / 0.00 / 1.3 | 0.66 / 0.74 / 0.57 / 10.4 | 0.33 / 0.65 / 0.00 / 3.5 | 0.51 / 0.74 / 0.29 / 7.9 | 0.30 / 0.61 / 0.00 / 1.0 |
| 0.30 | 0.30 / 0.61 / 0.00 / 1.5 | 0.66 / 0.74 / 0.57 / 10.4 | 0.42 / 0.70 / 0.14 / 4.6 | 0.51 / 0.74 / 0.29 / 8.0 | 0.30 / 0.61 / 0.00 / 1.0 |
| 0.35 | 0.30 / 0.61 / 0.00 / 1.6 | 0.63 / 0.70 / 0.57 / 10.6 | 0.44 / 0.74 / 0.14 / 5.6 | 0.58 / 0.74 / 0.43 / 8.1 | 0.30 / 0.61 / 0.00 / 1.0 |
| 0.40 | 0.38 / 0.61 / 0.14 / 2.3 | 0.66 / 0.74 / 0.57 / 11.2 | 0.44 / 0.74 / 0.14 / 6.1 | 0.61 / 0.78 / 0.43 / 8.7 | 0.28 / 0.57 / 0.00 / 1.1 |
| 0.45 | 0.38 / 0.61 / 0.14 / 2.4 | 0.63 / 0.70 / 0.57 / 11.6 | 0.40 / 0.65 / 0.14 / 7.1 | 0.61 / 0.78 / 0.43 / 9.5 | 0.30 / 0.61 / 0.00 / 1.2 |
| 0.50 | 0.42 / 0.70 / 0.14 / 2.9 | 0.63 / 0.70 / 0.57 / 11.9 | 0.52 / 0.61 / 0.43 / 8.9 | 0.68 / 0.78 / 0.57 / 10.1 | 0.30 / 0.61 / 0.00 / 1.3 |
| 0.55 | 0.42 / 0.70 / 0.14 / 4.1 | **0.68** / 0.65 / 0.71 / 12.2 | 0.52 / 0.61 / 0.43 / 9.3 | **0.82** / 0.78 / 0.86 / 10.5 | 0.30 / 0.61 / 0.00 / 2.2 |
| 0.60 | 0.42 / 0.70 / 0.14 / 4.9 | 0.66 / 0.61 / 0.71 / 12.7 | 0.56 / 0.70 / 0.43 / 10.0 | 0.80 / 0.74 / 0.86 / 10.7 | 0.30 / 0.61 / 0.00 / 2.5 |
| 0.65 | 0.56 / 0.70 / 0.43 / 6.0 | 0.62 / 0.52 / 0.71 / 13.8 | 0.56 / 0.70 / 0.43 / 11.0 | 0.78 / 0.70 / 0.86 / 11.5 | 0.30 / 0.61 / 0.00 / 2.8 |
| 0.70 | 0.66 / 0.74 / 0.57 / 7.3 | 0.60 / 0.48 / 0.71 / 14.4 | 0.66 / 0.74 / 0.57 / 12.1 | 0.78 / 0.70 / 0.86 / 11.7 | 0.26 / 0.52 / 0.00 / 4.7 |
| 0.75 | 0.63 / 0.70 / 0.57 / 8.9 | 0.53 / 0.35 / 0.71 / 15.7 | **0.66** / 0.61 / 0.71 / 13.0 | 0.71 / 0.57 / 0.86 / 13.5 | 0.33 / 0.52 / 0.14 / 7.7 |
| 0.80 | **0.69** / 0.52 / 0.86 / 12.2 | 0.56 / 0.26 / 0.86 / 17.1 | 0.64 / 0.57 / 0.71 / 13.9 | 0.71 / 0.57 / 0.86 / 14.1 | 0.29 / 0.43 / 0.14 / 11.0 |
| 0.85 | 0.67 / 0.48 / 0.86 / 13.7 | 0.54 / 0.22 / 0.86 / 17.9 | 0.55 / 0.39 / 0.71 / 15.4 | 0.71 / 0.57 / 0.86 / 14.8 | 0.27 / 0.39 / 0.14 / 13.1 |
| 0.90 | 0.60 / 0.35 / 0.86 / 16.0 | 0.57 / 0.13 / 1.00 / 18.8 | 0.54 / 0.22 / 0.86 / 17.3 | 0.60 / 0.35 / 0.86 / 17.0 | 0.39 / 0.22 / 0.57 / 16.1 |
| 0.95 | 0.65 / 0.30 / 1.00 / 17.8 | 0.54 / 0.09 / 1.00 / 19.2 | 0.57 / 0.13 / 1.00 / 18.9 | 0.61 / 0.22 / 1.00 / 18.3 | 0.49 / 0.13 / 0.86 / 18.7 |
| 1.00 | 0.50 / 0.00 / 1.00 / 20.0 | 0.50 / 0.00 / 1.00 / 20.0 | 0.50 / 0.00 / 1.00 / 20.0 | 0.50 / 0.00 / 1.00 / 20.0 | 0.50 / 0.00 / 1.00 / 20.0 |

- **A3** (all items resolved, no θ) scores 0.634 / 0.696 / 0.571 / 10.3. Bold marks each method's in-sample argmax θ. A2's argmax is θ 1.00.
- **A1 is above P_min at every θ from 0.50 to 0.95.** Below 0.50, P_min is higher, because there it acts like A3 (it stops once all items are resolved). A1 reaches the best in-sample balanced accuracy, 0.82 at θ 0.55, against 0.68 for P_min. One unanswerable question moves the unanswerable rate by 0.14, so these gaps are within noise. The direction matches the per-question count in §0.
- **P_min's answerable accuracy collapses faster as θ rises.** At θ 0.80 it is 0.26, against 0.57 for A1. P_min stops on 8 of 30 questions there and A1 on 17. On v27 test, P_min paid only −0.06 in answerable accuracy against B1.

At the frozen v27 θ* (B1 0.90, P_min 0.80, P_mean 0.90, A1 0.75, A2 0.95), each cell is answerable acc / unanswerable correct / mean L. A dash means the type has no unanswerable question.

| set | n ans / unans | B1 | P_min | P_mean | A1 | A2 | A3 |
|---|---|---|---|---|---|---|---|
| all | 23 / 7 | 0.35 / 0.86 / 16.0 | 0.26 / 0.86 / 17.1 | 0.22 / 0.86 / 17.3 | **0.57** / 0.86 / 13.5 | 0.13 / 0.86 / 18.7 | 0.70 / 0.57 / 10.3 |
| a_relO | 2 / 3 | 0.00 / 1.00 / 20.0 | 0.50 / 1.00 / 16.4 | 0.50 / 1.00 / 16.6 | 0.50 / 1.00 / 16.4 | 0.00 / 1.00 / 20.0 | 0.50 / 0.67 / 11.4 |
| b_relS | 3 / 2 | 0.00 / 1.00 / 20.0 | 0.00 / 1.00 / 20.0 | 0.00 / 1.00 / 20.0 | 0.00 / 1.00 / 20.0 | 0.00 / 1.00 / 20.0 | 0.67 / 0.50 / 11.0 |
| c_choose_rel | 5 / 0 | 0.80 / — / 14.4 | 0.20 / — / 17.0 | 0.20 / — / 17.6 | 1.00 / — / 12.2 | 0.00 / — / 20.0 | 0.80 / — / 8.6 |
| d_verify_rel | 5 / 0 | 0.40 / — / 10.2 | 0.60 / — / 11.8 | 0.60 / — / 13.6 | 0.60 / — / 9.2 | 0.40 / — / 15.4 | 0.80 / — / 5.2 |
| e_logical | 4 / 1 | 0.25 / 1.00 / 16.2 | 0.25 / 0.00 / 17.2 | 0.00 / 0.00 / 16.2 | 0.50 / 0.00 / 7.4 | 0.25 / 1.00 / 17.0 | 0.25 / 0.00 / 12.0 |
| f_compare | 4 / 1 | 0.25 / 0.00 / 15.4 | 0.00 / 1.00 / 20.0 | 0.00 / 1.00 / 20.0 | 0.50 / 1.00 / 15.6 | 0.00 / 0.00 / 19.8 | 1.00 / 1.00 / 13.8 |

- **Open-name types are hard for every θ-method.** On relO and relS, every thresholded method almost never stops. Open names give a first-token P_ans of about 0.25–0.75 at L20, so the v27 θ* (tuned on color and material) is out of range. This is a scale problem shared by all methods, not a decomposition effect.

## 6. Worked examples (θ 0.80 unless stated; R = resolved)

1. **00768430 (compare): the last item is a different question, so P_min never stops.** "Do the van and the hydrant have the same color?" The GT is no. The reader says "no" at every level.
   - A1 answers "no" with resolved true from L5, scores 0.88 at L7 and stops there, correct.
   - c3 "What color is the van?" is unknown until L9. Then it says gray or black with P 0.36–0.62, so P_min never stops.
   - The binding item is the last one, and it should not bind: it is not the question. The two existence items are yes 0.6–1.0 and harmless.
2. **121053010 and 03848528 (logical): the attribute rule makes P_min never stop.**
   - 121053010, "Do you see any doors or cars that are white?" (yes): c2 "What color is the door or car?" is unknown at all 20 levels, so pans_min = 0 everywhere. A1 stops at L10, correct.
   - 03848528, "Are there any hairbrushes or bottle caps?" (no): c1 says "no" correctly with P up to 0.99. c2 "What color is the hairbrush or bottle cap?" is correctly unknown, because there is nothing to color. A1 stops at L4, correct.
   - A negative existence question can never pass a gate that needs the missing object's color.
3. **0598390 (logical, OR): a correct "no" on one disjunct binds the min.** "Are there either pillows or trays?" The GT is yes.
   - From L13, c1 (pillow) is yes 0.99 and c3, the question copy, is yes 0.98–1.00.
   - c2 "Is there a tray?" is "no" (correct: the tray is "(-)") with P 0.55–0.74, and it binds the min throughout.
   - The min over items is the wrong aggregator for a disjunction. A sure "yes" on one disjunct should be enough. A1 stops at L12, where the reader happens to say "pillows", so both methods are wrong here.
4. **19151786 (choose × rel): checking both objects is the real decomposition, and it costs.** "Is the bench to the right or to the left of the blue vehicle?" The GT is left.
   - The relation item c3 and A1 both say "left" with P 0.78–0.94 from L13.
   - c1 "Is there a bench?" is unknown until L11. After that it flips yes, no, yes with P 0.36–0.77.
   - c2 "Is there a blue vehicle?" is yes until L9 and no from L11 to L19.
   - P_min tops out at 0.71 at L20 and never stops. A1 stops at L14, correct.
   - The existence checks are noisier than the relational answer. They are not reliable evidence that the objects are present.
5. **16880810 (choose × rel): a garbled item binds.** c2 "Is there a pillow to the left or to the right of the woman on the right?" mixes existence and choice. Its answer flips between yes, right and left with P 0.31–0.60. A1 says "left" from L11 with a score of 0.64–0.82, crosses 0.80 only at L16 and stops there, correct. P_min never stops.
6. **061043477 (logical): the only case where P_min wins, and it is timing luck.** "Are there unpeeled bananas or apples?" The GT is no.
   - All three items say "no" from L5.
   - A1 scores 0.85 at L6 and stops there. At L6 the reader answers "banana", a format slip, so A1's stop is wrong.
   - pans_min is 0.78 at L6 and 0.79 at L7, then 0.85 at L8. It stops at L8, the only level from L5 to L9 where the reader says "no", so P_min is correct.
   - No item vetoed anything. P_min stayed just under θ for two levels and crossed it at a level where the reader happened to be right.

## 7. Conclusion on the hypothesis

- **Partly confirmed.** P_min does diverge from A1 on multi-object questions. The correlation falls from 0.93 to 0.74, and the stop level differs on 40% of questions against 26% on v27 test. The divergence is limited to choose × rel, logical and compare. On query × rel (relO, relS) and verify × rel, planner_v3's plans are still "existence check + the question verbatim", so P_min ≈ A1 as on v27 test.
- **Where it diverges, it is worse, not better.** At the same θ, A1 wins 8 diverging questions and P_min wins 1. At the in-sample argmax θ, the count is 4 to 0, and A1's balanced accuracy is 0.82 against P_min's 0.68. All of these are on 30 questions.
- **Three mechanisms drive the divergence**, and none is a useful check:
  1. **The last item is replaced** by an attribute question about one object (compare, logical). It answers something other than the question.
  2. **Existence items are less confident** than the whole-question answer for small or ambiguous objects. They hold the min below θ without adding information (choose × rel).
  3. **Min over items is the wrong aggregator for "or" questions.** A confident "no" on one disjunct should not block a "yes".
- **Missing-object vetoes never happen.** A resolved "no" on an existence item passes the gate, and that happens on 33% of probe rows. The planner never checks the related (answer) object on relO, as the text-only probe also found.
- **Implication.** Decomposition can matter on multi-object questions, but the current planner_v3 and min gate make it a source of noise. A fair test of "decomposition helps on multi-object questions" would need four changes: a type-aware planner (no attribute rule on compare and logical, checks of the related object), a gate that treats an existence "no" as a failure where the question presupposes the object, a logical-structure-aware aggregator (max over disjuncts), and a θ scale for open-name answers. None of this was tried here.

## 8. Caveats

- **n = 30, 5 per type.** Two types have no unanswerable question, and 7 unanswerable questions in total. One question moves an unanswerable rate by 0.14. No CIs are reported, and none would be meaningful.
- **Val images, exploratory, in-sample.** The θ argmax is taken on the same 30 questions. The questions come from val_balanced (the v27 dev source, with dev images excluded), so they are not comparable to the v27 test numbers except as a descriptive contrast.
- **The planner was designed for attribute questions.** planner_v3 is unchanged. Its attribute-word rule causes 7 of 30 hijacked or wrong last items. Six plans needed a schema repair, against 1 in 300 on v27 dev.
- **Labels use exact match.** Open-name answers (relO and relS) carry synonym noise, such as letters vs text or woman vs tennis player. Choose answers matched exactly here: the reader said "left" or "right", not "to the left".
- **Not a GQA metric.** Single model, one seed, one decoding.
- **Same definitions as v27.** The agreement statistics use exactly the v27 test definitions, which reproduce 0.933 / 0.887 / 0.944 from `level_table_test.csv` (read-only).

## 9. Files

- `configs/gqa_stop_probe_multi_rel.json`: the probe config, new.
- `scripts/gqa_probe_multi_rel_select.py`, `scripts/gqa_probe_multi_rel_analyze.py`, `scripts/gqa_probe_multi_rel_tables.py`: new.
- `data/processed/gqa_stop/probe_multi_rel/`: `questions_dev.jsonl`, `gt_dev.json`, `screening_dev.jsonl` and the level images.
- `outputs/gqa_stop/probe_multi_rel/`:
  - `selection.jsonl`: types, programs and GT.
  - `level_table_dev.csv`: 600 rows.
  - `eval/dev/`: v27 evaluator outputs. Here "dev" is the probe set, and `theta_star.json` is in-sample.
  - `analysis/`: `agreement.csv`, `per_question.csv` and `divergence.json`.
  - `logs/`: env check, audit and progress logs.
