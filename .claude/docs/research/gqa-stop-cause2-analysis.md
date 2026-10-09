# gqa_stop v4 dev: analysis of "cause 2" (L20 existence item says no or stays unresolved)

Date: 2026-10-09. Scope: read-only analysis of `outputs/gqa_stop/visapp_gqa_v4/` (dev visual subset, 60 questions × 20 levels, model `qwen3.5:9b@6488c96fa5fa`).
Repository code, configs, prompts and outputs were not modified.
Counterfactual model calls were sent to the local Ollama server; their records live only in the session scratchpad (`.../scratchpad/cause2/*.jsonl`), not in `outputs/`.

## 1. Verdict

1. **Cause 2 is mostly a decoding artifact, not model strictness.**
   - The `qwen3.5:9b` Modelfile sets `presence_penalty 1.5` (also `temperature 1`, `top_k 20`, `top_p 0.95`).
   - Ollama also applies its default `repeat_penalty`, presumably 1.1 over the last 64 tokens (inferred, see §2.3).
   - Our requests override only `temperature`, `seed`, `num_ctx` and `num_predict`, so both penalties stay active at temperature 0.
   - The penalties hit tokens that sit at the end of the prompt. For the observer, that is the format example `{"answer":"yes","resolved":true,"confidence":0.9}`. The model is therefore pushed from `yes` to `no` or `unknown`, from `true` to `false`, and from `0`/`9` to `1`/`8`/`5` in confidence digits.
2. **Of the 22 cause-2 questions at L20, 18 are artifacts.** Replaying the exact requests with a neutral sampler makes the existence item answer "yes" and resolved. The neutral sampler sets `presence_penalty 0`, `repeat_penalty 1.0` and `frequency_penalty 0`.
   - The 22 questions are 20 failing item-1 existence checks plus 2 failing item-2 existence checks.
   - **4 cases are genuine.** These are "clear sky" on an overcast sky, "mirror reflecting a street sign", the "container" that GQA calls a box (0.6% of the image), and the reference "car" that is a rail car.
   - In all 4 genuine cases the GQA annotation or wording is dubious or ambiguous.
   - None of the 22 is a clearly visible object that the unbiased model denies.
3. **The model "knows" the objects.** For the artifact cases, the pre-sampler P(yes) of the existence token at L20 is 0.86 on average. For 11 cases it is ≥ 0.93, yet "no" was emitted.
   - A "yes" is emitted only when log P(yes) − log P(no) > about 3.8, that is P(yes) ≳ 0.975.
   - "true" is emitted only when the true/false gap is > about 4.2.
4. **Fixing the decoding removes the zeros but does not by itself move the operating point much.**
   - L20 zero scores drop from 21 to 7. All 7 remaining are material questions where the attribute item answers "unknown".
   - At dev accuracy ≥ 0.95, checklist_v4_min goes from 15.50 (θ 0.30) to 15.28 (θ 0.39). The paired image-cluster CI is −0.22 [−0.82, +0.30].
   - The reason is that the neutral observer passes the existence gate much earlier. At L5, item 1 passes the gate for 77% of questions under the neutral sampler against 33% by default. That adds early false stops, so θ has to rise.
   - The best decomposed variant under neutral decoding, attribute item only, reaches 14.77. That is still +1.95 [+1.08, +2.90] levels behind Direct v4 at 12.82.
5. **The artifact contaminates more than cause 2. This should be treated as a protocol defect for v1–v4.**
   - Every verbalised confidence in v1–v3 is distorted. The {0, 0.1, 0.15, 0.85, 1.0} quantisation is largely "0.9" with the `9` (and `0.`) penalised.
   - The resolved flags of attribute items are affected too: 6 of the 13 L20 unresolved attribute items had `true` as argmax.
   - Planner wording changes for 25 of 60 plans under the neutral sampler.
   - Reader answers change in 56 of 1200 calls. Under the neutral sampler, the reader says "gray" far more often, and L20 reader accuracy on this subset drops from 60/60 to 57/60.
   - Direct v4 P_ans and all Rel values are not affected. Their logprobs are pre-sampler, and their argmax token was always emitted (0 mismatches).

## 2. Evidence that cause 2 is a sampler artifact

### 2.1 Emitted token vs. argmax across all v4 records

The logprobs returned by Ollama are pre-penalty. With temperature 0 the emitted token should be the argmax. It often is not:

| record set | records | tokens | emitted ≠ argmax | most frequent (emitted ← argmax) |
|---|---|---|---|---|
| observer_v3_lp | 2600 | 40,122 | 3,500 | `1`←`0` 1309, `false`←`true` 621, `5`←`}` 559, `8`←`9` 405, `no`←`yes` 377, `unknown`←`yes` 93 |
| direct_lp | 1200 | 15,437 | 1,261 | `5`←`}` 510, `1`←`0` 489, then later words of free-text answers |
| relevance_v4 | 3800 | 3,801 | 0 | – (single token; `Yes` and `No` both appear in the prompt) |
| neutral replay (observer, all 2600) | 2600 | – | **0** | – |

The answer token (the first token of the JSON `answer` value) breaks down by item role as follows:

| item role | n | answer token ≠ argmax | resolved token `false` while `true` is argmax |
|---|---|---|---|
| existence (item 1) | 1200 | 430 (35.8%): `no`←`yes` 342, `unknown`←`yes` 85 | 301 of 1157 (L20: 12 of 12 unresolved) |
| middle (item 2 of 3) | 200 | 43 (21.5%) | 47 of 200 (L20: 2 of 2) |
| attribute (last) | 1200 | 0 | 273 of 1200 (L20: 6 of 13 unresolved) |
| Direct v4 | 1200 | 0 | n/a |

Decision boundary at the answer token (observer, yes/no in top-10): "yes" was emitted only for log-gap ≥ 3.75 and "no" for gaps up to 3.82. For `true`/`false` the boundary is about 4.1–4.3.
Of the 917 item-1 observations whose argmax is "yes", 397 (43%) were emitted as "no", "unknown" or unresolved.

### 2.2 Controlled replay of the 60 L20 existence items

The replay sends the exact v4 request (same rendered prompt, image, `format: json`, `think: false`, seed, `num_ctx`, `num_predict`, `logprobs`) and varies only the sampler options.

| variant | reproduces cache | yes / resolved | no / resolved | no / unresolved | answer-token argmax mismatches |
|---|---|---|---|---|---|
| default (as in v4) | 60/60 identical raw text | 40 | 8 | 12 | 20 |
| `presence_penalty 0` | – | 51 | 9 | 0 | 6 |
| neutral (`presence_penalty 0`, `repeat_penalty 1.0`, `frequency_penalty 0`) | – | **57** | **3** | **0** | **0** |

### 2.3 Mechanism (partly inferred)

- **Verified:**
  - `ollama show qwen3.5:9b --parameters` gives `presence_penalty 1.5`, `temperature 1`, `top_k 20`, `top_p 0.95` (Ollama 0.30.4).
  - The client's options (`ModelSettings.options` in `src/milestone1/gqa_stop/client.py`) contain only temperature, seed, num_ctx and num_predict.
  - Removing the penalties restores argmax decoding exactly.
- **Inferred:**
  - The penalties apply only to tokens in a recent window, which matches Ollama's `repeat_last_n` default of 64. Only tokens near the end of the prompt are penalised: `yes`, `true`, `0`, `9` and `}` from the observer example.
  - Attribute words earlier in the prompt (`gray`, `round`, `brick`) show no answer-token mismatches in the observer.
  - In the reader prompt, `"gray"` and `brick or wood` sit near the end. The neutral reader says "gray" much more often, which suggests the default sampler was suppressing it.
  - `presence_penalty 0` alone left 6 flips, so a second penalty (most likely `repeat_penalty 1.1`) is also active.

## 3. Definition and the L20 list (Q1)

**Definition.** A question is a cause-2 case at L20 if its existence check for the GQA target fails. Item 1 counts always. Item 2 counts when it is an existence check ("Is there…", "…present…"). The check fails when the answer is not "yes" or `resolved` is false. GQA and the reader treat the object as present (reader correct at L20 on all 60).

- 22 of 60 questions qualify: 20 through item 1, of which 12 are unresolved "no" and 8 are resolved "no", and 2 through item 2 only (15172776 and 07785408).
- **Effect on the score.**
  - The 12 unresolved item-1 cases give a hard zero through the v1 gate.
  - A resolved "no" passes the gate, but its r = P_ans("no") × Rel is tiny when the "no" was penalty-induced: 0.017, 0.023 and 0.056.
  - In 3 resolved-"no" cases the zero comes from the attribute item ("unknown").
  - 2 genuine cases have r of 0.22 and 0.27.
  - All 22 are below θ = 0.30 at L20, so all are forced to L20 under checklist_v4_min.
- v3 attributed 15 of the 21 L20 zeros to "existence no or unresolved". The 22 here differ because this list also counts near-zero r (< 0.30) and the two item-2 cases.

Column key:
- **rel**: GQA relate_count.
- **area %**: bbox area of the GQA target object (last object-id step in the `semantic` program) relative to the image.
- **P_ans**: probability of the emitted first answer token, as used in v4.
- **P(yes)**: pre-sampler probability of `yes` at that position.
- **Rel**: v4 relevance P("Yes").
- **neutral**: answer (resolved) under the neutral sampler.
- **flags**: taxonomy from §4.

| qid | question | GQA target | rel | area % | item 1 question | ans (res) | P_ans | P(yes) | Rel | item 2 (existence) ans (res) | neutral item 1 | neutral item 2 | reader | cause 2 | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 00943538 | What is the color of the clear sky? | sky | 0 | 21.3 | Is there a clear sky in the image? | no (y) | 0.53 | 0.39 | 0.52 |  | no (y) |  | white | genuine | a,c,e |
| 09887053 | Which shape is the mirror the street sign is reflected in? | mirror | 1 | 45.8 | Is there a mirror reflecting a street sign in the image? | no (y) | 0.91 | 0.07 | 0.25 |  | no (y) |  | round | genuine | b,e |
| 15172776 | What color is the train to the left of the car? | train | 1 | 31.8 | Is there a train in the image? | yes (y) | 1.00 | 1.00 | 1.00 | no (y): "Is there a car in the image?" | yes (y) | no (y) | yellow | genuine | a,e |
| 15174086 | What material is the container to the right of the man? | box | 1 | 0.6 | Is there a container to the right of the man in the image? | no (y) | 0.52 | 0.43 | 0.47 |  | no (y) |  | metal | genuine | a,b,d,e |
| 00271801 | What is the plate to the right of the cabinet made of? | plate | 1 | 10.8 | Is there a plate to the right of the cabinet in the image? | no (n) | 0.40 | 0.48 | 0.47 |  | yes (y) |  | metal | artifact | a,b,e |
| 00434791 | What's the fence made of? | fence | 0 | 21.9 | Is there a fence in the image? | no (n) | 0.03 | 0.97 | 0.94 |  | yes (y) |  | metal | artifact | c,f |
| 01316734 | Which shape is the plate to the left of the vase? | plate | 1 | 1.2 | Is there a plate to the left of the vase in the image? | no (n) | 0.03 | 0.97 | 0.95 |  | yes (y) |  | square | artifact | b,f |
| 01601414 | What is the shape of the sink that is in the bathroom? | sink | 1 | 0.3 | Is there a sink in the image? | no (n) | 0.03 | 0.97 | 0.96 |  | yes (y) |  | round | artifact | d,f |
| 05831816 | What material makes up the utensil to the right of the food? | fork | 1 | 5.4 | Is there a utensil to the right of the food in the image? | no (n) | 0.03 | 0.97 | 0.88 |  | yes (y) |  | plastic | artifact | b,f |
| 07512969 | What is the house to the left of the air conditioner made of? | house | 1 | 10.9 | Is there a house in the image? | no (y) | 0.42 | 0.57 | 0.86 | no (y): "Is there an air conditioner in the image?" | yes (y) | yes (y) | wood | artifact | a |
| 07785408 | What is the chair to the right of the monitor made of? | chair | 1 | 2.1 | Is there a chair in the image? | yes (y) | 0.98 | 0.98 | 0.95 | no (n): "Is the chair to the right of the monitor present in the image?" | yes (y) | yes (y) | leather | artifact | b,f |
| 08254993 | What material is the mailbox made of? | mailbox | 0 | 0.8 | Is there a mailbox in the image? | no (n) | 0.19 | 0.73 | 0.48 |  | yes (y) |  | brick | artifact | d |
| 09488485 | What makes up the utensils in the box? | utensils | 1 | 1.3 | Is there a box with utensils in the image? | no (y) | 0.23 | 0.72 | 0.67 |  | yes (y) |  | plastic | artifact | a,b |
| 10979790 | What is the sidewalk made of? | sidewalk | 0 | 19.4 | Is there a sidewalk in the image? | no (n) | 0.19 | 0.75 | 0.46 |  | yes (y) |  | brick | artifact | c |
| 12411054 | The car on the left has what color? | car | 0 | 3.9 | Is there a car on the left in the image? | no (n) | 0.02 | 0.97 | 0.96 |  | yes (y) |  | gray | artifact | b,f |
| 1390846 | What is the shelter made of? | shelter | 0 | 60.0 | Is there a shelter in the image? | no (n) | 0.02 | 0.98 | 0.96 |  | yes (y) |  | brick | artifact | c,f |
| 13985783 | What material was used to make the bag that is to the right of the bowl? | bag | 1 | 18.5 | Is there a bag to the right of the bowl in the image? | no (n) | 0.04 | 0.96 | 0.85 |  | yes (y) |  | plastic | artifact | b,f |
| 16756887 | What color does the large pot have? | pot | 0 | 1.0 | Is there a large pot in the image? | no (y) | 0.06 | 0.93 | 0.93 |  | yes (y) |  | brown | artifact | f |
| 17184616 | What is the coffee table made of? | coffee table | 0 | 6.8 | Is there a coffee table in the image? | no (n) | 0.03 | 0.96 | 0.94 |  | yes (y) |  | glass | artifact | f |
| 17888876 | What shape is the plate? | plate | 0 | 10.8 | Is there a plate in the image? | no (n) | 0.03 | 0.96 | 0.99 |  | yes (y) |  | square | artifact | f |
| 18752666 | Which material is the shelf to the left of the tools made of? | shelf | 1 | 7.1 | Is there a shelf to the left of the tools in the image? | no (y) | 0.02 | 0.98 | 0.90 |  | yes (y) |  | metal | artifact | b,f |
| 19452749 | What is the bench to the left of the van made out of? | bench | 1 | 0.7 | Is there a bench to the left of the van in the image? | no (y) | 0.32 | 0.63 | 0.80 |  | yes (y) |  | concrete | artifact | a,b,d |
| 00947257 | What color is the bear that is to the right of the other bear? | bear | 1 | 5.9 | Is there a bear in the image? | yes (y) | 1.00 | 1.00 | 0.99 |  | yes (y) |  | black | – |  |
| 011012564 | What shape does the food which is on the plate have? | pizza | 1 | 84.7 | Is there a food on the plate in the image? | yes (y) | 1.00 | 1.00 | 0.99 |  | yes (y) |  | triangular | – |  |
| 01420620 | Which color is the car? | car | 0 | 36.1 | Is there a car in the image? | yes (y) | 1.00 | 1.00 | 1.00 |  | yes (y) |  | white | – |  |
| 03189870 | The mirror is of which shape? | mirror | 0 | 2.9 | Is there a mirror in the image? | yes (y) | 0.99 | 0.99 | 0.94 |  | yes (y) |  | round | – |  |
| 03774985 | What is the shape of the dessert the toy car is on? | cake | 1 | 15.8 | Is there a dessert in the image? | yes (y) | 1.00 | 1.00 | 0.99 | yes (y): "Is there a toy car on the dessert in the image?" | yes (y) | yes (y) | rectangular | – |  |
| 03954206 | Which color is the curtain to the left of the lamp? | curtain | 1 | 4.5 | Is there a curtain in the image? | yes (y) | 1.00 | 1.00 | 1.00 |  | yes (y) |  | yellow | – |  |
| 05827270 | What is the floor made of? | floor | 0 | 47.5 | Is there a floor in the image? | yes (y) | 0.99 | 0.99 | 1.00 |  | yes (y) |  | concrete | – |  |
| 06572516 | What material is the ceiling? | ceiling | 0 | 35.5 | Is there a ceiling in the image? | yes (y) | 1.00 | 1.00 | 1.00 |  | yes (y) |  | glass | – |  |
| 06572517 | What is the ceiling made of? | ceiling | 0 | 35.5 | Is there a ceiling in the image? | yes (y) | 1.00 | 1.00 | 1.00 |  | yes (y) |  | glass | – |  |
| 06614339 | Which color are the pants? | pants | 0 | 1.1 | Is there a pair of pants in the image? | yes (y) | 0.99 | 0.99 | 0.94 |  | yes (y) |  | orange | – |  |
| 06726789 | What is the color of the mask the catcher is wearing? | mask | 1 | 0.5 | Is there a mask worn by a catcher in the image? | yes (y) | 0.99 | 0.99 | 0.98 |  | yes (y) |  | red | – |  |
| 07111669 | What is the color of the coat that the girl is wearing? | coat | 1 | 47.7 | Is there a girl in the image? | yes (y) | 1.00 | 1.00 | 1.00 |  | yes (y) |  | pink | – |  |
| 0778325 | Which color is the plane that is flying in the sky? | airplane | 1 | 39.2 | Is there a plane flying in the sky in the image? | yes (y) | 1.00 | 1.00 | 1.00 |  | yes (y) |  | white | – |  |
| 07833741 | The sky is of what color? | sky | 0 | 37.5 | Is there a sky in the image? | yes (y) | 1.00 | 1.00 | 0.98 |  | yes (y) |  | white | – |  |
| 07949045 | What is the shelf made of? | shelf | 0 | 1.4 | Is there a shelf in the image? | yes (y) | 0.98 | 0.98 | 0.96 |  | yes (y) |  | glass | – |  |
| 08924895 | What shape do you think the chair is? | chair | 0 | 3.1 | Is there a chair in the image? | yes (y) | 0.97 | 0.97 | 0.97 |  | yes (y) |  | triangular | – |  |
| 09149294 | What is the shape of the flags that are on the rope? | flags | 1 | 0.5 | Is there a flag on a rope in the image? | yes (y) | 0.99 | 0.99 | 0.97 |  | yes (y) |  | triangular | – |  |
| 09672864 | What shape is the table? | table | 0 | 56.7 | Is there a table in the image? | yes (y) | 1.00 | 1.00 | 1.00 |  | yes (y) |  | triangular | – |  |
| 10780377 | What color is the chair next to the other chair? | chair | 1 | 12.4 | Is there a chair in the image? | yes (y) | 1.00 | 1.00 | 0.99 |  | yes (y) |  | black | – |  |
| 11213386 | What color is the umbrella on the right? | umbrella | 0 | 19.4 | Is there an umbrella on the right in the image? | yes (y) | 0.99 | 0.99 | 0.94 |  | yes (y) |  | yellow | – |  |
| 11213423 | What color is the umbrella in the middle? | umbrella | 0 | 13.3 | Is there an umbrella in the image? | yes (y) | 1.00 | 1.00 | 1.00 |  | yes (y) |  | red | – |  |
| 11619091 | What do you think is the color of this fence? | fence | 0 | 21.6 | Is there a fence in the image? | yes (y) | 1.00 | 1.00 | 0.99 |  | yes (y) |  | black | – |  |
| 11844870 | The container that is sitting on the dresser has what color? | box | 1 | 3.3 | Is there a container sitting on the dresser in the image? | yes (y) | 0.99 | 0.99 | 0.97 |  | yes (y) |  | black | – |  |
| 121044380 | Which color is the shirt? | shirt | 0 | 8.9 | Is there a shirt in the image? | yes (y) | 0.99 | 0.99 | 0.99 |  | yes (y) |  | gray | – |  |
| 12416576 | What material is the sidewalk near the car? | sidewalk | 1 | 23.9 | Is there a sidewalk near the car in the image? | yes (y) | 1.00 | 1.00 | 0.99 |  | yes (y) |  | brick | – |  |
| 12454186 | What color do the shorts have? | shorts | 0 | 3.0 | Is there a pair of shorts in the image? | yes (y) | 0.99 | 0.99 | 0.97 |  | yes (y) |  | khaki | – |  |
| 13127787 | Which material is the cup to the right of the other cup made of? | cup | 1 | 0.9 | Is there a cup in the image? | yes (y) | 1.00 | 1.00 | 0.98 |  | yes (y) |  | plastic | – |  |
| 13808031 | What's the ground made of? | ground | 0 | 35.8 | Is there a ground in the image? | yes (y) | 0.99 | 0.99 | 0.98 |  | yes (y) |  | brick | – |  |
| 141013932 | What color is the shirt? | shirt | 0 | 1.2 | Is there a shirt in the image? | yes (y) | 0.98 | 0.98 | 0.85 |  | yes (y) |  | white | – |  |
| 14276984 | What is the color of the tiles that are behind the toilet? | tiles | 1 | 13.2 | Is there a tile wall behind the toilet in the image? | yes (y) | 0.99 | 0.99 | 0.98 |  | yes (y) |  | blue | – |  |
| 14848534 | What color is the bicycle? | bicycle | 0 | 8.6 | Is there a bicycle in the image? | yes (y) | 1.00 | 1.00 | 0.99 |  | yes (y) |  | white | – |  |
| 15172866 | What is the color of the train car that is to the left of the other train car? | train car | 1 | 23.1 | Is there a train in the image? | yes (y) | 1.00 | 1.00 | 1.00 |  | yes (y) |  | yellow | – |  |
| 16820186 | The brown table is which shape? | table | 0 | 34.1 | Is there a brown table in the image? | yes (y) | 0.99 | 0.99 | 0.96 |  | yes (y) |  | round | – |  |
| 17108363 | What color is the cup that is to the right of the other cup? | cup | 1 | 4.9 | Is there a cup in the image? | yes (y) | 1.00 | 1.00 | 0.99 |  | yes (y) |  | orange | – |  |
| 17891388 | What is the color of the candles on the table? | candles | 1 | 2.4 | Is there a candle on the table in the image? | yes (y) | 0.99 | 0.99 | 0.98 |  | yes (y) |  | black | – |  |
| 18486270 | What material is the fence near the sheep made of? | fence | 1 | 12.2 | Is there a fence near the sheep in the image? | yes (y) | 0.99 | 0.99 | 0.94 |  | yes (y) |  | metal | – |  |
| 18660252 | What is the shape of the tiles that look white? | tiles | 0 | 2.1 | Is there a tile in the image? | yes (y) | 1.00 | 1.00 | 0.98 |  | yes (y) |  | rectangular | – |  |
| 19315625 | What shape is the window? | window | 0 | 5.9 | Is there a window in the image? | yes (y) | 1.00 | 1.00 | 1.00 |  | yes (y) |  | round | – |  |

## 4. Taxonomy of the 22 cause-2 cases (Q2)

The primary cause is decided by the neutral replay. Flags are non-exclusive and describe the default run.

| primary cause | n | questions |
|---|---|---|
| **(g) sampler artifact**: existence item becomes yes/resolved under the neutral sampler | **18** | 00271801, 00434791, 01316734, 01601414, 05831816, 07512969, 07785408, 08254993, 09488485, 10979790, 12411054, 1390846, 13985783, 16756887, 17184616, 17888876, 18752666, 19452749 |
| genuine "no" under the neutral sampler | **4** | 00943538, 09887053, 15174086, 15172776 (item 2) |

| flag | all 22 | of which genuine (4) | examples / notes |
|---|---|---|---|
| (a) wording mismatch between the planner's noun phrase and the GQA noun or sense | 7 | 3 | "box with utensils" (GQA: utensils in a box); "clear sky" (sky is overcast white); "container" (GQA `box`, a post-like object); "bench" (a concrete skate ledge); "plate" (a metal tool plate, not a dish); "house" (a cabin); reference "car" (a rail car) |
| (b) relational qualifier in the existence item | 11 | 2 | "plate to the right of the cabinet", "car on the left", "bag to the right of the bowl", "mirror reflecting a street sign". Regex base rate of relational item-1 wording: 9 of 22 cause-2 vs 10 of 38 others. 9 of the 11 become "yes" under the neutral sampler. |
| (c) stuff-like or amorphous target | 4 | 1 | sidewalk, fence, sky, shelter (stone wall). floor, ceiling and ground are not cause-2. |
| (d) small target (< 1% of image) | 4 | 1 | sink 0.3%, container 0.6%, bench 0.7%, mailbox 0.8%. Under 2%: 7 of 22 vs 6 of 38 others. 3 of the 4 become "yes" under the neutral sampler. |
| (e) GQA annotation dubious or ambiguous | 5 | 4 | `filter weather: clear` on a white overcast sky; the street sign is not identifiable in the mirror; `container` is a GQA `box`; `car` is a rail car; `plate` is a machine plate |
| (f) object clearly visible but answered "no" (model strictness) | 12 in the default run, **0 under the neutral sampler** | 0 | plate (17888876), fence, shelter, coffee table, bag, car on the left, fork, shelf, pot, plate left of the vase, sink, chair (item 2); every one has pre-sampler P(yes) ≥ 0.93 (chair item 2: 0.945) |

- Visual checks of the L20 images with the GQA box drawn confirm that the plate (17888876), fence (00434791) and shelter (1390846) are large and unambiguous.
- In the genuine cases the model's "no" is defensible.
  - The sky in 00943538 is white and overcast, not clear.
  - The mirror in 09887053 shows a bus and street, not a recognisable sign.
  - The "car" in 15172776 is a freight rail car.
  - The "container" in 15174086 is a tiny post-like object.
- Cause-2 questions are skewed to material: 14 of 22 vs 8 of 38 for the others. This overlaps with the separate "material → unknown" abstention (§7).

## 5. Trajectories over levels (Q3)

Groups: artifact (18), genuine (4), non-cause-2 (38). "yes-ok" means answer "yes" and resolved. "gate-ok" means resolved and not "unknown". P(yes) is pre-sampler.

| metric | group | L1 | L3 | L5 | L7 | L10 | L12 | L15 | L17 | L19 | L20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| yes-ok, default | artifact | 0.00 | 0.00 | 0.06 | 0.00 | 0.11 | 0.11 | 0.22 | 0.28 | 0.22 | 0.06 |
| yes-ok, neutral | artifact | 0.11 | 0.28 | 0.56 | 0.56 | 0.78 | 0.83 | 0.89 | 0.89 | 0.89 | 1.00 |
| gate-ok, default | artifact | 0.00 | 0.00 | 0.11 | 0.06 | 0.28 | 0.22 | 0.44 | 0.50 | 0.50 | 0.33 |
| gate-ok, neutral | artifact | 0.11 | 0.28 | 0.67 | 0.72 | 0.89 | 0.89 | 1.00 | 1.00 | 1.00 | 1.00 |
| P(yes) | artifact | 0.20 | 0.32 | 0.39 | 0.52 | 0.66 | 0.75 | 0.80 | 0.81 | 0.84 | 0.86 |
| Rel | artifact | 0.08 | 0.11 | 0.16 | 0.25 | 0.41 | 0.56 | 0.65 | 0.73 | 0.78 | 0.83 |
| yes-ok, default | genuine | 0.00 | 0.00 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| yes-ok, neutral | genuine | 0.75 | 0.75 | 0.75 | 0.75 | 0.50 | 0.50 | 0.50 | 0.50 | 0.25 | 0.25 |
| P(yes) | genuine | 0.36 | 0.64 | 0.62 | 0.72 | 0.59 | 0.55 | 0.54 | 0.50 | 0.46 | 0.47 |
| Rel | genuine | 0.11 | 0.21 | 0.29 | 0.38 | 0.42 | 0.42 | 0.47 | 0.47 | 0.50 | 0.56 |
| yes-ok, default | non-cause-2 | 0.08 | 0.21 | 0.37 | 0.53 | 0.66 | 0.76 | 0.84 | 0.84 | 0.89 | 1.00 |
| yes-ok, neutral | non-cause-2 | 0.50 | 0.61 | 0.76 | 0.89 | 0.92 | 0.97 | 0.97 | 0.97 | 1.00 | 1.00 |
| P(yes) | non-cause-2 | 0.45 | 0.57 | 0.71 | 0.83 | 0.90 | 0.94 | 0.96 | 0.96 | 0.98 | 0.99 |
| Rel | non-cause-2 | 0.18 | 0.28 | 0.43 | 0.59 | 0.75 | 0.83 | 0.90 | 0.93 | 0.96 | 0.98 |

Pre-sampler P(yes) of the item-1 answer token by level (`.` means not yes-ok; `Yr` means yes and resolved):

| qid | L12 | L13 | L14 | L15 | L16 | L17 | L18 | L19 | L20 |
|---|---|---|---|---|---|---|---|---|---|
| 12411054 | Yr 0.98 | Yr 0.99 | Yr 0.99 | Yr 0.99 | Yr 0.99 | Yr 0.98 | Yr 0.99 | Yr 0.99 | . 0.97 |
| 17888876 | Yr 0.99 | Yr 0.99 | Yr 0.99 | Yr 0.98 | Yr 0.99 | Yr 0.98 | Yr 0.99 | Yr 0.99 | . 0.96 |
| 16756887 | . 0.97 | Yr 0.98 | Yr 0.98 | Yr 0.98 | Yr 0.98 | Yr 0.98 | Yr 0.98 | Yr 0.98 | . 0.93 |

- **Artifact group.**
  - Pre-sampler P(yes) rises monotonically, from 0.20 to 0.86, like the non-cause-2 group, only somewhat lower. Rel rises too, from 0.08 to 0.83.
  - The emitted answer flickers because P(yes) sits on the ≈0.975 knife-edge. Some questions are yes-ok over L10–L19 and flip at L20 when P(yes) dips from 0.99 to 0.97, as in the table above. That is not "over-strictness as detail appears".
  - Questions with a yes-ok followed by a later not-ok are equally common in the default run (5 of 18 artifact, 6 of 38 others) and under the neutral sampler (4 of 18, 5 of 38).
  - Consecutive-level yes→not-yes transitions are 14 by default and 13 under the neutral sampler.
- **Genuine group.** This is the only place where a high-resolution flip is real. Item 1 says "yes" at low resolution (0.75 at L1, neutral) and turns to "no" as detail appears. P(yes) falls from 0.72 at L7 to 0.47 at L20. That is the expected behaviour for "clear sky" or "mirror reflecting a street sign" once the image shows that the qualifier does not hold.

## 6. Relation to the reader and Direct (Q4)

| group | n | Direct v4 stop L (θ 0.26) | Direct accuracy | checklist_v4_min stop L (θ 0.30) | checklist accuracy | L20 direct_rel | L20 direct P_ans | L20 direct v4 | L20 item-1 P(yes) | L20 item-1 Rel |
|---|---|---|---|---|---|---|---|---|---|---|
| artifact | 18 | 16.67 | 1.00 | 18.56 | 0.94 | 0.50 | 0.68 | 0.37 | 0.86 | 0.83 |
| genuine | 4 | 16.25 | 1.00 | 20.00 | 1.00 | 0.45 | 0.70 | 0.33 | 0.47 | 0.56 |
| non-cause-2 | 38 | 10.63 | 0.92 | 13.58 | 0.95 | 0.82 | 0.71 | 0.58 | 0.99 | 0.98 |

- **Direct handles every cause-2 question correctly.** It stops on average 1.9 levels (artifact) and 3.75 levels (genuine) before the checklist, and on 13 of 22 before L20.
  - 12411054 (car on the left) is correct under Direct at L11.
  - That same question is one of the three checklist wrong stops (L10). The existence item flickers yes between L10 and L19, so the checklist can stop at L10 when the reader is still wrong there.
- **The model "knows" the object when asked the original question.**
  - For the artifact group at L20, Direct's own P_ans × Rel is 0.37 on average, the existence P(yes) is 0.86, and the existence Rel is 0.83.
  - The denial is entirely in the emitted text and the resolved flag, not in the probabilities.
  - Both cause-2 groups are harder questions overall: Direct Rel at L20 is 0.50 (artifact) and 0.45 (genuine), against 0.82 for the others.
  - For the genuine group the existence P(yes) is also low (0.47), so the model is truly unsure there.
- **Direct is immune to the sampler artifact.**
  - Its prompt has no `yes`/`true` near the end, and its first answer token was always the argmax.
  - Part of the v4 "Direct beats decomposition" result is therefore the artifact, but only part (§7).

Per-question data: `.../scratchpad/cause2/q4_cause2_direct.csv`.

## 7. What the v4 numbers imply (Q5): replay

All rows use the 60 dev questions, the in-sample best θ at accuracy ≥ 0.95 (θ grid 0.00–1.00 by 0.01), and stop at the first level with score ≥ θ (else L20).
The replay reproduces the v4 report exactly: 15.50 / 12.82 / 14.85 / 16.83 for checklist min / Direct / rel_min / pans_min.
On all 1200 level rows, the v1 gate re-implementation, which uses the plan's `depends_on`, matches `checklist_v4_min` with 0 mismatches.

| score (default reader correctness) | θ | acc | mean L | L20 zeros | stop at L20 |
|---|---|---|---|---|---|
| checklist_v4_min (v4, as run) | 0.30 | 0.950 | 15.50 | 21 | 29 |
| attribute item only, ungated product (the lead's figure) | 0.39 | 0.950 | 14.60 | – | – |
| attribute item only, own gate | 0.30 | 0.950 | 14.28 | 13 | – |
| min with existence items dropped (gate among the rest) | 0.30 | 0.950 | 14.70 | 14 | – |
| oracle: drop existence items only on the 22 cause-2 questions | 0.30 | 0.950 | 15.05 | 14 | – |
| item 1 as soft P(yes) × Rel (pre-sampler), no gate from item 1 | 0.30 | 0.950 | 14.85 | 14 | – |
| **neutral observer** (full replay, 2600 calls), checklist min | 0.39 | 0.950 | **15.28** | 7 | 23 |
| neutral observer, existence items dropped | 0.39 | 0.950 | 15.02 | 7 | – |
| neutral observer, attribute item only | 0.39 | 0.950 | **14.77** | 7 | 20 |
| Direct v4 | 0.26 | 0.950 | **12.82** | 0 | 12 |

Paired image-cluster bootstrap (2000 reps, 57 image clusters; θ fixed at the in-sample best):

| comparison | Δ mean L [95% CI] |
|---|---|
| neutral min − v4 min | −0.22 [−0.82, +0.30] |
| neutral attribute-only − v4 min | −0.73 [−1.61, +0.00] |
| neutral attribute-only − Direct v4 | +1.95 [+1.08, +2.90] |
| neutral min − Direct v4 | +2.47 [+1.58, +3.45] |

**The sampler also changes the reader.**
- Replaying the reader with the neutral sampler (1200 calls) changes 56 answers. Most of them go to "gray".
- L20 accuracy falls to 57/60: sky white→gray twice, and shorts khaki→gray.
- With neutral reader correctness, the best mean L at ≥ 0.95 is 15.43 (neutral min, θ 0.40), 14.98 (neutral attribute-only) and 12.82 (Direct).
- The dev subset was screened for "reader correct at L20" under the default sampler. A neutral-sampler v5 would therefore need to re-screen or accept a reader ceiling below 1.0.

**Interpretation.**
- Removing cause-2 zeros is worth about 0.2–1.2 levels, not 2.7. The 2.7 is the gap to Direct.
  - The L20 zeros only matter for questions that otherwise reach θ before L20.
  - Once the existence item stops blocking, the same accuracy constraint binds earlier. The neutral observer is confident earlier (non-cause-2 questions, item 1 yes-ok at L5: 0.76 neutral vs 0.37 default), and θ must rise from 0.30 to 0.39.
- The binding constraint at accuracy ≥ 0.95 is three early false-confident stops, not the L20 zeros.
  - Checklist (both samplers): 11213386 (umbrella on the right, L9), 121044380 (shirt, L5–6) and 12411054 (car on the left, L8–10).
  - Direct: 11213386 (L6), 12454186 (L15) and 18486270 (L18).
- After the decoding fix, the remaining L20 zeros (7) are all material questions whose attribute item answers "unknown":
  - utensils, container, floor, mailbox, sidewalk, shelf, chair.
  - The observer prompt's "Do not infer typical attributes from commonsense" plausibly drives this. It is a separate failure mode ("cause 3").

## 8. Recommendations for v5 (Q6), ranked

| # | change | expected effect (dev, from §7 replays) | cost / risk | needs prompt revision? |
|---|---|---|---|---|
| 1 | **Fix decoding: send `presence_penalty: 0`, `repeat_penalty: 1.0`, `frequency_penalty: 0` (optionally `top_k: 1`) in every request** (planner, observer, reader, direct, relevance). Options are part of the cache key, so this automatically forms new namespaces. | Removes 18 of 22 cause-2 cases and the `true`→`false` resolved flips. L20 zeros go from 21 to 7. Mean L at ≥ 0.95 only −0.2 (CI includes 0) with the current gate. Restores verbalised confidences to unbiased values, though still coarse (0.9 dominant). | A config and client change; full re-run (~2600 observer + 1200 direct + 3800 relevance + 1200 reader + 60 planner calls; about 2 h at the observed rate). Changes the reader ceiling (57/60 at L20) and 25 of 60 plans, so dev must be re-screened and fair-v1 comparators rerun. v1–v4 numbers should carry a caveat. | No (protocol/infrastructure), but it must be re-registered as v5 because every output changes. |
| 2 | **Treat existence items as soft: score the attribute item only, or exclude existence items from the min and the gate.** | v4 data: 14.28 (own gate). Neutral sampler: 14.77. Best decomposed result so far, but still +1.95 [+1.08, +2.90] vs Direct. | Scoring change only; can be applied post hoc to cached outputs. Loses the "identify target first" semantics the decomposition was meant to give. | No |
| 3 | **Address early false confidence** (the binding constraint), e.g. k = 2 consecutive levels ≥ θ (helped in M1), or require attribute-answer agreement across two adjacent levels. | Not measured here (speculative). The three binding wrong stops are flickering early stops (L5–L10). | Cheap post-hoc replay on cached tables. | No |
| 4 | **Material abstention: soften "Do not infer typical attributes from commonsense" for material**, or score "unknown" as low but non-zero. | Targets the 7 neutral-sampler L20 zeros (all material). Size unknown (speculative). | Prompt change to observer_v3; risk of more commonsense guessing at low levels. | Yes (observer) |
| 5 | **Existence-item wording: use the question's noun phrase verbatim, no added qualifiers, visibility phrasing ("Is a plate visible…?").** | Under the neutral sampler only 4 genuine cases remain, each with a dubious annotation (e) or lexical ambiguity (a). Rewording would fix at most about 1–2 (09887053 under the neutral planner asks just "Is there a mirror…"). | Planner revision; the neutral planner already becomes more verbatim on its own (25 of 60 plans change, mostly toward verbatim relational qualifiers). | Yes (planner), and the one dev revision is used, so a v5 must re-register. |
| 6 | Drop relational qualifiers from existence items | Not supported. 9 of 11 relational cases are artifacts and pass under the neutral sampler. Stripping qualifiers weakens target identification. | – | Yes |

**Bottom line.**
- Recommendation 1 is mandatory for validity. It is a measurement bug that affects all versions and both the observer and the reader.
- It does not by itself close the gap to Direct. On dev, decomposition remains about 2 levels behind Direct even with the bug fixed and existence items removed from the min.

## 9. Reproduction (scratchpad scripts; nothing written to the repo except this note)

All files are in `/tmp/claude-1000/-home-kanaoka-usr-github-milestone-1/89f09e7e-066d-41bb-bd4e-992d2f4aa5c2/scratchpad/cause2/`:
- `mismatch.py`, `gap.py`, `ans_mismatch.py`, `res_mismatch.py`: argmax-mismatch census.
- `probe.py`: observer replay, modes `l20c1`, `l20c1_neutral` and `full_neutral` → `probe_*.jsonl`.
- `planner_probe.py` → `planner_neutral.jsonl`; `reader_probe.py` → `reader_neutral.jsonl`.
- `gqa_meta.py` → `gqa_meta.json`: GQA target and bbox.
- `l20table.py`, `tables.py` → `l20_table3.csv`, `tab_l20.md`.
- `traj.py`, `q4q5.py`, `replay.py` (stop/θ replay validated against v4) and `score_neutral.py` → `neutral_scores.csv`.
- Ollama 0.30.4, `qwen3.5:9b` digest 6488c96fa5fa.
