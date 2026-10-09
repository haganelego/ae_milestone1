# gqa_stop v4c (plan §20, A13): clean decoding re-run — dev results

> **Superseded numbers (A13).** Every v1–v4 number in `gqa-stop-v1-report.md`, `gqa-stop-v2-dev.md`, `gqa-stop-v3-dev.md`
> and `gqa-stop-v4-dev.md` came from runs where the qwen3.5:9b Modelfile defaults were active: presence_penalty 1.5,
> top_k 20, top_p 0.95 and temperature 1 (overridden to 0). Those numbers are superseded by this run.
> The old outputs stay untouched in `outputs/gqa_stop/visapp_gqa_v{1..4}/`.

Date: 2026-10-09. Run: `configs/gqa_stop_v4c.json` (run_id visapp_gqa_v4c, family `logprob_item`), output `outputs/gqa_stop/visapp_gqa_v4c/`.
Dev only for methods; test was screened and reader-run only. No freeze, nothing committed.

## Verdict

At dev accuracy ≥ 0.95, checklist_v4_min (θ=0.39) stops at mean Level **14.64**, with accuracy 0.966. Fair v1 under the same clean decoding stops at 19.97; the difference is **-5.32 [-6.76, -3.95]**.
- §19.4 / §18.4 abort rule: **abort = False**.
- Direct v4 (θ=0.34) stops at 14.02.
- θ is a smooth trade-off: 98 distinct operating points.
- In-sample caveat: θ is chosen on the same 59 dev questions, and 0.95 means at most 3 wrong.

## 1. Decoding block and determinism gate (preflight)

- **Decoding block.** It is sent with every request: `{"temperature": 0.0, "seed": 20261008, "num_ctx": 8192, "num_predict": 128, "frequency_penalty": 0.0, "mirostat": 0, "presence_penalty": 0.0, "repeat_penalty": 1.0, "top_k": 1, "top_p": 1.0}`, think=False. It is part of every cache key, so no pre-A13 record is reused.
- **Modelfile (`ollama show qwen3.5:9b`, ollama 0.30.4).** The parameters are temperature 1, top_k 20, top_p 0.95 and presence_penalty 1.5, and all four are overridden. Only RENDERER/PARSER qwen3.5 remain; they are the chat template and the output parser, not sampling parameters.
- **Gate (a).** The same request sent twice returned byte-identical content and logprobs, for both a text-only and an image (reader L20) request: True.
- **Gate (b).** 120 logprob calls (direct at L1/L20 on 50 candidates, plus 20 relevance calls) covered 1164 token positions. 0 positions emitted a token that was not the argmax; match rate **1.0**.

## 2. Screening and adoption (same candidates and quota rules as v1)

| split | candidates | correct / incorrect at L20 | outcome changed vs v1 | adopted (images) | kept / dropped / added vs v1 |
|---|---|---|---|---|---|
| dev | 265 | 166 / 99 | 8 (4 c→i, 4 i→c) | 100 (94) | 95 / 5 / 5 |
| test | 639 | 406 / 233 | 28 (13 c→i, 15 i→c) | 200 (183) | 194 / 6 / 6 |

- The L20 repeat diagnostic agrees 1.0.
- The diffs are in `adoption_diff_{dev,test}.json`.
- Reader accuracy by level (all adopted):
  - dev: L1 0.57, L5 0.76, L10 0.86, L15 0.91.
  - test: L1 0.53, L5 0.69, L10 0.82, L15 0.94.

## 3. textprobe and visual subset

| split | adopted | prior-answerable | visual |
|---|---|---|---|
| dev | 100 | 41 | 59 |
| test | 200 | 75 | 125 |

## 4. Fair v1 under clean decoding (v1 arm)

- **Method.** The v1 planner, checklist prompt v2, direct and reader were run on the 59 dev visual questions × 20 levels, and the thresholds were frozen on that subset (`v1arm/frozen_policy_subset.json`).
- **Result.** The thresholds frozen on the dev subset are Checklist-min θ=1.0 (dev acc 0.983, mean L 19.97), Checklist-mean θ=0.95 and Direct (verbalised) θ=0.99. Fixed-dev is L20.
- **Why fair v1 barely stops.** Under clean decoding the verbalised checklist confidence takes essentially three values, counted over all 1180 rows: {0.0: 525, 0.9: 414, 0.8: 238, 1.0: 2, 0.95: 1}. Any θ ≤ 0.9 also stops early wrong answers at 0.8/0.9 and falls below 0.95. Only θ=1.0 remains, and almost no row reaches 1.0, so fair v1 stops at about L20. The comparison against fair v1 is therefore close to a comparison against Fixed-20.
- **Failure.** One v1 planner call (question 05827270) remained a schema failure after repair, so that question scores 0 at every level in the v1 arm.

## 5. v4 pilot (§19.3)

**Gates.**
- Answer token located: 1.00.
- Rel contains Yes or No: 1.00.
- Initial valid rate: {'observer': 1.0, 'direct_lp': 1.0, 'relevance': 1.0, 'planner': 1.0, 'reader': 1.0}.
- Observer raw output equal to the v3 (penalty 1.5) cache: 0.27 of 190 comparisons.
- Rel monotonic: items True, direct True.

| L | reader acc | item-1 resolved | mean P_ans | mean Rel | mean product | direct P_ans | direct Rel | distinct min r |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.50 | 0.40 | 0.81 | 0.08 | 0.06 | 0.46 | 0.05 | 3 |
| 5 | 0.75 | 0.85 | 0.79 | 0.22 | 0.18 | 0.61 | 0.14 | 11 |
| 10 | 0.85 | 0.90 | 0.85 | 0.46 | 0.43 | 0.66 | 0.35 | 14 |
| 15 | 0.95 | 0.95 | 0.89 | 0.67 | 0.62 | 0.69 | 0.60 | 19 |
| 20 | 1.00 | 1.00 | 0.90 | 0.82 | 0.75 | 0.70 | 0.75 | 20 |

## 6. Dev devgrid (59 visual questions × 20; no freeze)

Best feasible θ per method at dev accuracy ≥ 0.95. The CI is a paired image-cluster bootstrap with 2000 reps; "acc − hull" uses the subset Fixed-t upper hull.

| method | θ | acc | mean L | median | stop20 | stopped & wrong | acc − hull | mean L − fair v1 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| checklist_v4_min (**pre-registered**) | 0.39 | 0.966 | 14.64 | 16.0 | 0.34 | 2 | +0.057 | -5.32 [-6.76, -3.95] |
| checklist_v4_mean (logged) | 0.61 | 0.966 | 14.85 | 16.0 | 0.36 | 2 | +0.053 | -5.12 [-6.46, -3.84] |
| pans_min (logged) | 0.78 | 0.966 | 13.24 | 14.0 | 0.37 | 2 | +0.083 | -6.73 [-8.41, -5.24] |
| pans_mean (logged) | 0.84 | 0.966 | 12.85 | 12.0 | 0.36 | 2 | +0.091 | -7.12 [-8.80, -5.51] |
| rel_min (logged) | 0.55 | 0.966 | 14.97 | 16.0 | 0.36 | 2 | +0.051 | -5.00 [-6.32, -3.77] |
| rel_mean (logged) | 0.74 | 0.966 | 15.51 | 17.0 | 0.36 | 2 | +0.042 | -4.46 [-5.66, -3.34] |
| direct_v4 (comparator) | 0.34 | 0.966 | 14.02 | 14.0 | 0.31 | 2 | +0.068 | -5.95 [-7.32, -4.69] |
| direct_pans (logged) | 0.85 | 0.966 | 14.41 | 20.0 | 0.56 | 2 | +0.061 | -5.56 [-7.51, -3.68] |
| direct_rel (logged) | 0.53 | 0.966 | 14.29 | 14.0 | 0.27 | 2 | +0.063 | -5.68 [-6.97, -4.47] |
| checklist_v1_min (comparator (fair v1)) | 1.0 | 0.983 | 19.97 | 20.0 | 0.98 | 1 | -0.016 | +0.00 [+0.00, +0.00] |
| checklist_v1_mean (logged) | 0.95 | 0.983 | 19.95 | 20.0 | 0.98 | 1 | -0.016 | -0.02 [-0.05, +0.00] |
| direct_v1_verbalized (comparator) | 0.99 | 0.966 | 19.32 | 20.0 | 0.92 | 2 | -0.022 | -0.64 [-1.40, -0.10] |
| fair v1 Checklist-min (clean, subset-frozen) | 1.0 | 0.983 | 19.97 | 20.0 | 0.98 | 1 | -0.016 | – |

Subset fixed stops: fixed_5 0.695@5; fixed_10 0.797@10; fixed_15 0.881@15; fixed_20 (Fixed-dev) 1.000@20.

**Sensitivity of checklist_v4_min to θ** (θ every 0.05):

| θ | acc | mean L | stopped & wrong |
|---|---|---|---|
| 0.05 | 0.864 | 9.17 | 8 |
| 0.10 | 0.881 | 10.29 | 7 |
| 0.15 | 0.898 | 11.36 | 6 |
| 0.20 | 0.915 | 12.14 | 5 |
| 0.25 | 0.915 | 13.03 | 5 |
| 0.30 | 0.949 | 13.63 | 3 |
| 0.35 | 0.949 | 14.15 | 3 |
| 0.40 | 0.983 | 14.80 | 1 |
| 0.45 | 1.000 | 15.31 | 0 |
| 0.50 | 1.000 | 15.59 | 0 |
| 0.55 | 0.983 | 15.98 | 1 |
| 0.60 | 0.983 | 16.29 | 1 |
| 0.65 | 1.000 | 16.85 | 0 |
| 0.70 | 1.000 | 17.44 | 0 |
| 0.75 | 1.000 | 17.93 | 0 |
| 0.80 | 1.000 | 18.27 | 0 |
| 0.85 | 1.000 | 18.80 | 0 |
| 0.90 | 1.000 | 19.27 | 0 |
| 0.95 | 1.000 | 19.75 | 0 |
| 1.00 | 1.000 | 20.00 | 0 |

**Stop histograms** (level stopped/wrong):
- checklist_v4_min@0.39: L1 1/0, L4 1/0, L5 1/0, L6 1/0, L7 2/0, L8 5/0, L9 4/1, L10 3/0, L11 1/0, L12 3/0, L13 1/0, L14 3/0, L15 2/0, L16 2/0, L17 2/0, L18 3/1, L19 4/0, L20 20/0
- direct_v4@0.34: L1 1/0, L4 2/0, L5 1/1, L6 2/0, L8 2/0, L9 4/1, L10 6/0, L11 4/0, L12 3/0, L13 3/0, L14 3/0, L15 4/0, L16 1/0, L17 1/0, L18 1/0, L19 3/0, L20 18/0
- fair_v1_checklist_min@1.0: L18 1/1, L20 58/0

**Component trends:**

| L | reader acc | item-1 resolved | min r = 0 | mean P_ans | mean Rel | mean product | distinct min r | mean min r | mean direct v4 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.36 | 0.32 | 0.85 | 0.81 | 0.09 | 0.07 | 10 | 0.02 | 0.03 |
| 3 | 0.58 | 0.44 | 0.78 | 0.79 | 0.14 | 0.12 | 14 | 0.04 | 0.05 |
| 5 | 0.69 | 0.71 | 0.63 | 0.76 | 0.22 | 0.19 | 23 | 0.08 | 0.10 |
| 7 | 0.71 | 0.83 | 0.51 | 0.79 | 0.32 | 0.28 | 30 | 0.15 | 0.16 |
| 10 | 0.80 | 0.92 | 0.44 | 0.81 | 0.44 | 0.39 | 34 | 0.23 | 0.23 |
| 12 | 0.85 | 0.95 | 0.31 | 0.83 | 0.53 | 0.48 | 42 | 0.30 | 0.29 |
| 15 | 0.88 | 0.98 | 0.20 | 0.85 | 0.63 | 0.57 | 48 | 0.41 | 0.36 |
| 17 | 0.93 | 1.00 | 0.14 | 0.86 | 0.69 | 0.63 | 52 | 0.47 | 0.42 |
| 20 | 1.00 | 1.00 | 0.12 | 0.88 | 0.79 | 0.71 | 53 | 0.55 | 0.50 |

### L20 per question (`dev_l20_per_question.csv`)

At L20 the existence item answers "no" for 4 of 59 questions and is unresolved for 0. min r = 0 for 7. Under the penalty run (v3/v4) the corresponding counts were 20 "no", 12 unresolved and 21 zeros.

| qid | question | item 1 → answer (resolved) | P_ans₁ | Rel₁ | last item → answer (resolved) | P_ans | Rel | min r | direct v4 |
|---|---|---|---|---|---|---|---|---|---|
| 05827270 | What is the floor made of? | yes (y) | 0.99 | 1.00 | unknown (n) | 0.67 | 0.91 | 0.00 | 0.28 |
| 07785408 | What is the chair to the right of the monitor made of? | yes (y) | 0.83 | 0.44 | unknown (n) | 0.67 | 0.19 | 0.00 | 0.13 |
| 07949045 | What is the shelf made of? | yes (y) | 0.98 | 0.96 | unknown (n) | 0.44 | 0.41 | 0.00 | 0.27 |
| 08254993 | What material is the mailbox made of? | yes (y) | 0.73 | 0.48 | unknown (n) | 0.72 | 0.13 | 0.00 | 0.05 |
| 09488485 | What makes up the utensils in the box? | yes (y) | 0.72 | 0.67 | unknown (n) | 0.99 | 0.14 | 0.00 | 0.04 |
| 10979790 | What is the sidewalk made of? | yes (y) | 0.75 | 0.46 | unknown (n) | 0.86 | 0.20 | 0.00 | 0.07 |
| 15174086 | What material is the container to the right of the man? | no (y) | 0.52 | 0.47 | unknown (n) | 0.62 | 0.04 | 0.00 | 0.02 |
| 00434791 | What's the fence made of? | yes (y) | 0.97 | 0.94 | metal (y) | 0.45 | 0.13 | 0.06 | 0.15 |
| 19452749 | What is the bench to the left of the van made out of? | yes (y) | 0.63 | 0.80 | concrete (y) | 0.56 | 0.26 | 0.15 | 0.36 |
| 15172776 | What color is the train to the left of the car? | no (y) | 0.62 | 0.25 | yellow (y) | 0.91 | 0.53 | 0.15 | 0.46 |
| 00271801 | What is the plate to the right of the cabinet made of? | yes (y) | 0.48 | 0.47 | metal (y) | 0.86 | 0.19 | 0.16 | 0.29 |
| 13127787 | Which material is the cup to the right of the other cup made of? | yes (y) | 0.83 | 0.66 | plastic (y) | 0.95 | 0.20 | 0.19 | 0.10 |
| 05831816 | What material makes up the utensil to the right of the food? | yes (y) | 0.97 | 0.88 | metal (y) | 0.45 | 0.49 | 0.22 | 0.32 |
| 18486270 | What material is the fence near the sheep made of? | yes (y) | 0.99 | 0.94 | wood (y) | 0.44 | 0.50 | 0.22 | 0.21 |
| 01601414 | What is the shape of the sink that is in the bathroom? | yes (y) | 0.98 | 0.95 | round (y) | 0.64 | 0.44 | 0.28 | 0.19 |
| 15172866 | What is the color of the train car that is to the left of the other train car? | no (y) | 0.88 | 0.37 | yellow (y) | 0.92 | 0.66 | 0.33 | 0.62 |
| 05609323 | What is the color of the bag the logo is on? | yes (y) | 0.91 | 0.96 | black (y) | 0.46 | 0.77 | 0.36 | 0.50 |
| 18752666 | Which material is the shelf to the left of the tools made of? | yes (y) | 0.98 | 0.90 | metal (y) | 0.79 | 0.51 | 0.40 | 0.46 |
| 07512969 | What is the house to the left of the air conditioner made of? | yes (y) | 0.82 | 0.65 | wood (y) | 0.98 | 0.41 | 0.40 | 0.37 |
| 13985783 | What material was used to make the bag that is to the right of the bowl? | yes (y) | 0.96 | 0.85 | plastic (y) | 0.97 | 0.44 | 0.43 | 0.36 |
| 01316734 | Which shape is the plate to the left of the vase? | yes (y) | 0.97 | 0.95 | square (y) | 0.57 | 0.75 | 0.43 | 0.37 |
| 17108363 | What color is the cup that is to the right of the other cup? | no (y) | 0.85 | 0.57 | orange (y) | 0.98 | 0.66 | 0.48 | 0.64 |
| 141013932 | What color is the shirt? | yes (y) | 0.98 | 0.85 | white (y) | 0.74 | 0.67 | 0.49 | 0.55 |
| 17891388 | What is the color of the candles on the table? | yes (y) | 0.99 | 0.98 | black (y) | 0.85 | 0.62 | 0.53 | 0.77 |
| 08924895 | What shape do you think the chair is? | yes (y) | 0.97 | 0.97 | triangular (y) | 0.64 | 0.84 | 0.53 | 0.17 |
| 06572517 | What is the ceiling made of? | yes (y) | 1.00 | 1.00 | glass (y) | 0.68 | 0.81 | 0.55 | 0.79 |
| 11213423 | What color is the umbrella in the middle? | yes (y) | 0.99 | 0.91 | red and yellow (y) | 0.63 | 0.92 | 0.58 | 0.48 |
| 16756887 | What color does the large pot have? | yes (y) | 0.93 | 0.93 | brown (y) | 0.76 | 0.77 | 0.59 | 0.72 |
| 06614339 | Which color are the pants? | yes (y) | 0.99 | 0.94 | orange (y) | 0.74 | 0.84 | 0.61 | 0.70 |
| 03954206 | Which color is the curtain to the left of the lamp? | yes (y) | 0.95 | 0.92 | yellow (y) | 0.75 | 0.83 | 0.63 | 0.66 |
| 11213386 | What color is the umbrella on the right? | yes (y) | 0.99 | 0.94 | yellow and red (y) | 0.67 | 0.97 | 0.65 | 0.44 |
| 14848534 | What color is the bicycle? | yes (y) | 1.00 | 0.99 | white (y) | 0.82 | 0.83 | 0.68 | 0.64 |
| 03189870 | The mirror is of which shape? | yes (y) | 0.99 | 0.94 | round (y) | 0.82 | 0.83 | 0.69 | 0.30 |
| 10780377 | What color is the chair next to the other chair? | yes (y) | 0.96 | 0.89 | black (y) | 0.94 | 0.74 | 0.69 | 0.72 |
| 06572516 | What material is the ceiling? | yes (y) | 1.00 | 1.00 | glass (y) | 0.80 | 0.87 | 0.69 | 0.70 |
| 11619091 | What do you think is the color of this fence? | yes (y) | 1.00 | 0.99 | black (y) | 0.78 | 0.94 | 0.73 | 0.64 |
| 06726789 | What is the color of the mask the catcher is wearing? | yes (y) | 0.99 | 0.97 | red (y) | 0.88 | 0.84 | 0.74 | 0.61 |
| 0778325 | Which color is the plane that is flying in the sky? | yes (y) | 1.00 | 1.00 | white and green (y) | 0.78 | 0.97 | 0.75 | 0.56 |
| 11844870 | The container that is sitting on the dresser has what color? | yes (y) | 0.99 | 0.97 | black (y) | 0.96 | 0.79 | 0.76 | 0.71 |
| 16820186 | The brown table is which shape? | yes (y) | 0.99 | 0.96 | round (y) | 0.98 | 0.77 | 0.76 | 0.53 |
| 18660252 | What is the shape of the tiles that look white? | yes (y) | 1.00 | 0.99 | rectangular (y) | 0.89 | 0.86 | 0.76 | 0.63 |
| 1390846 | What is the shelter made of? | yes (y) | 0.98 | 0.96 | stone (y) | 0.86 | 0.93 | 0.80 | 0.79 |
| 00947257 | What color is the bear that is to the right of the other bear? | yes (y) | 1.00 | 0.99 | black (y) | 0.97 | 0.84 | 0.81 | 0.79 |
| 09672864 | What shape is the table? | yes (y) | 1.00 | 1.00 | triangular (y) | 0.92 | 0.89 | 0.82 | 0.73 |
| 09887053 | Which shape is the mirror the street sign is reflected in? | yes (y) | 1.00 | 1.00 | round (y) | 0.95 | 0.87 | 0.82 | 0.38 |
| 17888876 | What shape is the plate? | yes (y) | 0.96 | 0.99 | square (y) | 0.95 | 0.88 | 0.84 | 0.65 |
| 09149294 | What is the shape of the flags that are on the rope? | yes (y) | 0.99 | 0.97 | triangular (y) | 0.99 | 0.86 | 0.86 | 0.57 |
| 011012564 | What shape does the food which is on the plate have? | yes (y) | 1.00 | 0.99 | triangular (y) | 0.99 | 0.87 | 0.86 | 0.64 |
| 17184616 | What is the coffee table made of? | yes (y) | 0.96 | 0.94 | glass (y) | 0.99 | 0.87 | 0.87 | 0.77 |
| 13808031 | What's the ground made of? | yes (y) | 0.99 | 0.98 | brick (y) | 0.98 | 0.90 | 0.88 | 0.50 |
| 03774985 | What is the shape of the dessert the toy car is on? | yes (y) | 0.98 | 0.96 | rectangular (y) | 0.97 | 0.94 | 0.91 | 0.46 |
| 12411054 | The car on the left has what color? | yes (y) | 0.97 | 0.96 | gray (y) | 0.96 | 0.95 | 0.91 | 0.49 |
| 0813387 | What color is the sky? | yes (y) | 0.99 | 0.99 | gray (y) | 0.95 | 0.97 | 0.92 | 0.50 |
| 19315625 | What shape is the window? | yes (y) | 1.00 | 1.00 | round (y) | 0.96 | 0.97 | 0.93 | 0.52 |
| 12416576 | What material is the sidewalk near the car? | yes (y) | 1.00 | 0.99 | brick (y) | 0.99 | 0.94 | 0.93 | 0.86 |
| 07111669 | What is the color of the coat that the girl is wearing? | yes (y) | 1.00 | 1.00 | pink (y) | 0.97 | 0.98 | 0.95 | 0.95 |
| 121044380 | Which color is the shirt? | yes (y) | 0.99 | 0.99 | gray (y) | 1.00 | 0.96 | 0.96 | 0.66 |
| 05579796 | Which color do you think the umbrella is? | yes (y) | 1.00 | 1.00 | white (y) | 0.98 | 0.99 | 0.97 | 0.89 |
| 01420620 | Which color is the car? | yes (y) | 1.00 | 1.00 | white (y) | 1.00 | 0.99 | 0.99 | 0.96 |

## 7. Exploratory replays (orchestrator request; NOT pre-registered; logged only)

These are computed from the same v4c dev records with no new inference (`explore` command; outputs `dev_exploratory.{csv,json}`).
- **soft_existence.** The stop score is P_ans × Rel of the last (attribute) item only, gated only by that item's own `resolved` flag. The existence item is excluded from the min and from the ancestor gate.
- **k=2.** A stop needs score ≥ θ at 2 consecutive levels. The forced stop at L20 is unchanged.
- **Selection.** The best θ is the minimum mean level with accuracy ≥ 0.95 (ties: higher accuracy, larger θ). It is chosen in-sample on the 59 dev questions.
- **Consistency check.** The checklist_v4_min, direct_v4 and checklist_v1_min rows match the devgrid exactly.

| method | role | k | θ | acc | mean L | stop20 | stopped & wrong | acc − hull | stopped-and-wrong question ids |
|---|---|---|---|---|---|---|---|---|---|
| checklist_v4_min | pre-registered | 1 | 0.39 | 0.966 | 14.64 | 0.34 | 2 | +0.057 | 11213386, 11619091 |
| direct_v4 | comparator | 1 | 0.34 | 0.966 | 14.02 | 0.31 | 2 | +0.068 | 0813387, 11213386 |
| checklist_v1_min (v1 arm) | comparator (fair v1) | 1 | 1.0 | 0.983 | 19.97 | 0.98 | 1 | -0.016 | 11213386 |
| soft_existence | exploratory (not pre-registered; logged only) | 1 | 0.39 | 0.966 | 14.49 | 0.34 | 2 | +0.059 | 11213386, 11619091 |
| checklist_v4_min k=2 | exploratory (not pre-registered; logged only) | 2 | 0.27 | 0.966 | 14.22 | 0.29 | 2 | +0.064 | 11213386, 18486270 |
| direct_v4 k=2 | exploratory (not pre-registered; logged only) | 2 | 0.3 | 0.966 | 14.22 | 0.32 | 2 | +0.064 | 0813387, 11213386 |
| soft_existence k=2 | exploratory (not pre-registered; logged only) | 2 | 0.27 | 0.966 | 14.03 | 0.29 | 2 | +0.067 | 11213386, 18486270 |

- **Subset fixed curve (dev, 59 questions).** Fixed-5 0.695, Fixed-10 0.797, Fixed-15 0.881, Fixed-17 0.932, Fixed-19 0.915, Fixed-20 1.000. Fixed-dev is L20.
- **Recurring error.** Question 11213386 is stopped-and-wrong under every method, including the v1 arm.
- **Effect of the variants.** soft_existence and k=2 each move the mean level by less than one level (14.64 → 14.03–14.49). They do not change the picture.

**v1-arm planner failure.** In the v1 arm, the v1 planner for question 05827270 stayed a schema failure after one repair (invalid JSON: "Extra data"). That question therefore scores 0 at every level in the v1 arm and stops at L20 under checklist_v1_min. The v4 methods use planner_v3, which is unaffected.

## Notes

- **Unchanged.** Prompts are the same as in v1–v4 and no revision counter moved. P_ans, Rel and the gate follow §19 exactly.
- **New stages.** `detgate`, `screen --split test` before freeze (A13 flag), `readers` (reader on every level for dev and test; no method on test), `textprobe` on this run's own adoption, and `v1arm`. `devgrid` merges the v1-arm checklist columns.
- **Reproducibility.** Every record of this run was produced under the same explicit decoding block. A replay from cache needs no model calls.
