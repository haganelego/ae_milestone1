# Probe (prepared, not yet run): oracle checklist from the GQA program (2026-10-15)

Status: **prepared only; no inference has been run.** Exploratory, not pre-registered. Same 30 questions, level images, model (qwen3.5:9b, digest 6488c96fa5fa), A13 clean decoding, think=false and top_logprobs 10 as the multi-object probe (`gqa-stop-probe-multi-rel.md`). Prompts unchanged: the oracle items go through `observer_v3` exactly like planner_v3 items.

Question the probe answers: **does decomposition help when the plan is correct and the aggregation matches the question logic?** The multi-object probe found P_min loses to A1 for four reasons: the attribute rule hijacks the last item, min is wrong for OR, a resolved "no" on an existence item cannot veto, and relO plans never check the related object. The oracle plan fixes all four by construction. Only the plan and the aggregator change; the scored answer is still the reader's (reader_lp), as in v27.

## 1. Files

| file | role |
|---|---|
| `src/milestone1/gqa_stop/oracle_plan.py` | rule-based plan builder from the GQA `semantic` program (pure) |
| `src/milestone1/gqa_stop/oracle_score.py` | per-level aggregator: gate, veto, logic groups (pure) |
| `tests/test_gqa_stop_oracle_plan.py` | 25 tests (builder + aggregator) |
| `configs/gqa_stop_probe_oracle.json` | copy of the probe config: run_id `probe_oracle`, output `outputs/gqa_stop/probe_oracle/`; data and selection roots stay on `probe_multi_rel` (read only); new `oracle` block |
| `data/processed/gqa_stop/probe_oracle/plans.json` | the 30 oracle plans (written, frozen once inference starts: the runner stores its SHA-256) |
| `scripts/gqa_probe_oracle_run.py` | `plans`, `show`, `infer` (`--dry-run`, `--max-seconds`), `table` |
| `scripts/gqa_probe_oracle_eval.py` | θ sweep, θ*, at-θ table, agreement, veto counts, per-question table |

No existing module was changed. `visapp_gqa_v27*/`, `v22*/` and `probe_multi_rel/` are only read. Nothing has been written under `outputs/gqa_stop/probe_oracle/` yet.

## 2. Plan rules (`oracle_plan.py`, rules version `oracle-v1`)

- **Object nodes.** `select` starts a node. `filter <type>` attaches to the node's noun phrase: color, material and untyped filters become an adjective before the noun ("green building", "unpeeled banana"); hposition becomes "on the left"; vposition becomes "at the top". `relate name,rel,dir` creates a new node tied to its input node (the anchor). With `dir = s` the new node is the subject ("the bird on the rock"); with `dir = o` it is the object ("the cabinet that the microwave is to the left of").
- **Generic targets.** A relate name `_` is an unknown target, phrased "something". `person` becomes "someone" / "Who", and `place` becomes "some place" / "Where". These names are hypernyms implied by the question word, not the answer.
- **Presupposition → existence items.** A node named by the final op (query, verify, choose, same / different / common) is presupposed, and so is every anchor of a presupposed or asserted node. The named subject of `choose rel` is presupposed too. Each presupposed node gets one item with `role = "existence"` and `required_answer = "yes"`. Existence wording is "Is there a X in the image?" for a plain node, "Is the A <rel> something?" for an object-side relation and "Is there something <rel> the A?" for a subject-side relation.
- **Assertion → final items.** The node consumed by `exist` is asserted, not presupposed: it becomes the final item ("Are there books on the floor?"), and only its anchors get existence items.
- **Final item(s)** (`role = "final"`, `required_answer = None`):

| last op | final item |
|---|---|
| query name | "What is the man wearing?", "What is on the wire?", "Who is the crowd watching?", "Where are the trees?" |
| query attr | "What color is the X?" |
| verify attr / verify rel | "Is the X red?" / "Is the X <rel> a Y?" |
| exist | "Is there a X …?" |
| choose attr / choose rel | "Is the X a or b?" / "Is the X a or b the Y?" ("to the left or to the right of") |
| same / different attr | "Do the X and the Y have the same color?", "Are the X and the Y made of the same material?", "… different colors?" |
| common | "What is common to the X and the Y?" |
| and / or | one final item per operand; every one carries `logic = "and"` or `"or"` |

- **depends_on.** An existence item depends on its anchor's existence item. A final item depends on the existence items of the nodes it names. OR / AND operands have no dependency unless their node has an anchor.
- **Item schema.** `id` (o1…), `question`, `depends_on`, `role`, `required_answer`, `logic`, `source_op` (the program op index).
- **Leakage guards.**
  - Object ids and the "(-)" non-existence marker are stripped and never used.
  - Choose options are ordered by where they occur in the question text. GQA lists the correct option first in the program ("below|above" for an "above or below" question), so program order would leak.
  - A relate target name is used only when it is not `_`, and then it is already named in the question.
  - `answer` and `fullAnswer` are never read by the builder. Pronouns map to a fixed noun (he → man, she → woman). The `plans` step reads the GT answer only for a leak check: no item may contain a GT word that the question does not contain. All 30 pass.
- **Unsupported programs** (nested and/or, two-input choose such as "taller", single-input same / different, negated filters, unknown ops) give `items = []`, `sensible = false` and an `unsupported: …` flag. Such a plan scores 0 at every level. None of the 30 probe questions hit this.

## 3. Aggregator (`oracle_score.py`)

Inputs per item and level: the observer_v3_lp signal {valid, answer, resolved, P_ans}, with P_ans the first answer-token probability as in v27.

- **Gated score g_j** (the P rule). g_j = P_ans_j if item j and every item on its `depends_on` chain are valid, resolved and not "unknown"; otherwise g_j = 0.
- **Veto `object_absent`.** An existence item with `required_answer = "yes"` that is valid, resolved and answers "no" sets `object_absent = true`. The question score is then 0 at that level.
- **Existence score E.** The min of g over existence items, or 1 if the plan has none (OR plans).
- **Final score F.** The final items' g combined by their `logic`: OR → max, AND → min, a single final item → its g.

| column | method | definition |
|---|---|---|
| `oracle_min` | P_oracle_min | min of g over all items (the current P rule, no veto) |
| `oracle_logic` | **P_oracle_logic** | 0 if vetoed, else min(E, F) |
| `oracle_logic_noveto` | P_oracle_logic_noveto | min(E, F) without the veto (veto ablation) |
| `oracle_kleene` | P_oracle_kleene (extra) | like `oracle_logic`, but OR = max g over disjuncts answering "yes" if any, else min g over all disjuncts; AND is the dual |

- **Why the extra Kleene variant.** Under the specified OR = max, one confident "no" disjunct reaches θ alone, even when the other disjunct is unresolved. That is logically wrong for a "no" answer: every disjunct must be a confident "no". Three of the five logical questions have GT "no" (03848528, 08206123, 061043477). P_oracle_kleene requires all disjuncts for "no" and one disjunct for "yes". It is reported next to P_oracle_logic, not instead of it.
- **Stop rule and scoring** are those of v27 (§27): stop at the first level with score ≥ θ, the reader's answer is scored, strict scoring is primary and unified scoring is the sensitivity analysis. The answerable / unanswerable labels are probe_multi_rel's screening labels (23 / 7).

## 4. The 30 plans (oracle next to planner_v3)

`[deps]` follows the item id. `req=yes` marks a required-yes existence item. The planner_v3 plans are read from the probe_multi_rel cache.

**a_relO**
```
19404920 'What is the man wearing?' -> hat
  oracle  flags: generic target: _
    o1 [] existence req=yes: Is there a man in the image?
    o2 [o1] existence req=yes: Is the man wearing something?
    o3 [o2] final: What is the man wearing?
  planner_v3
    c1 []: Is there a man in the image?
    c2 [c1]: What is the man wearing?
16403783 'Where are the trees?' -> forest
  oracle  flags: generic target: place
    o1 [] existence req=yes: Are there trees in the image?
    o2 [o1] existence req=yes: Are the trees in some place?
    o3 [o2] final: Where are the trees?
  planner_v3
    c1 []: Is there a tree in the image?
    c2 [c1]: Where is the tree in the image?
19687543 'What is the woman holding?' -> bag
  oracle  flags: generic target: _
    o1 [] existence req=yes: Is there a woman in the image?
    o2 [o1] existence req=yes: Is the woman holding something?
    o3 [o2] final: What is the woman holding?
  planner_v3
    c1 []: Is there a woman in the image?
    c2 [c1]: What is the woman holding?
09212473 'The crowd is watching who?' -> woman
  oracle  flags: generic target: person
    o1 [] existence req=yes: Is there a crowd in the image?
    o2 [o1] existence req=yes: Is the crowd watching someone?
    o3 [o2] final: Who is the crowd watching?
  planner_v3
    c1 []: Is there a crowd in the image?
    c2 [c1]: Who is the crowd watching?
00259214 'What is the sign hanging off of?' -> pole
  oracle  flags: generic target: _
    o1 [] existence req=yes: Is there a sign in the image?
    o2 [o1] existence req=yes: Is the sign hanging off something?
    o3 [o2] final: What is the sign hanging off?
  planner_v3
    c1 []: Is there a sign hanging off of something in the image?
    c2 [c1]: What is the sign hanging off of?
```
**b_relS**
```
14863272 'What is full of donuts?' -> tray
  oracle  flags: generic target: _
    o1 [] existence req=yes: Are there donuts in the image?
    o2 [o1] existence req=yes: Is there something full of the donuts?
    o3 [o2] final: What is full of the donuts?
  planner_v3
    c1 []: Is there a container full of donuts in the image?
    c2 [c1]: What is the container full of donuts made of?
13365742 "What's on the wire?" -> sign
  oracle  flags: generic target: _
    o1 [] existence req=yes: Is there a wire in the image?
    o2 [o1] existence req=yes: Is there something on the wire?
    o3 [o2] final: What is on the wire?
  planner_v3
    c1 []: Is there a wire in the image?
    c2 [c1]: What is on the wire?
06943437 "What's on the oven?" -> kettle
  oracle  flags: generic target: _
    o1 [] existence req=yes: Is there an oven in the image?
    o2 [o1] existence req=yes: Is there something on the oven?
    o3 [o2] final: What is on the oven?
  planner_v3
    c1 []: Is there an oven in the image?
    c2 [c1]: What is on the oven?
08900610 'What is on the ski on the left?' -> letters
  oracle  flags: generic target: _
    o1 [] existence req=yes: Is there a ski on the left in the image?
    o2 [o1] existence req=yes: Is there something on the ski on the left?
    o3 [o2] final: What is on the ski on the left?
  planner_v3
    c1 []: Is there a ski on the left in the image?
    c2 [c1]: What is on the ski on the left?
11464769 'What is in front of the green building?' -> fence
  oracle  flags: generic target: _
    o1 [] existence req=yes: Is there a green building in the image?
    o2 [o1] existence req=yes: Is there something in front of the green building?
    o3 [o2] final: What is in front of the green building?
  planner_v3
    c1 []: Is there a green building in the image?
    c2 [c1]: What is in front of the green building?
```
**c_choose_rel**
```
02584384 'Is the counter above or below the cabinet the microwave is to the left of?' -> below
  oracle
    o1 [] existence req=yes: Is there a microwave in the image?
    o2 [o1] existence req=yes: Is the microwave to the left of a cabinet?
    o3 [] existence req=yes: Is there a counter in the image?
    o4 [o2,o3] final: Is the counter above or below the cabinet that the microwave is to the left of?
  planner_v3
    c1 []: Is there a microwave in the image?
    c2 [c1]: Is there a cabinet to the left of the microwave in the image?
    c3 [c2]: Is the counter above or below the cabinet to the left of the microwave in the image?
19419643 'Are the ripe oranges to the left or to the right of the bag made of plastic?' -> right
  oracle
    o1 [] existence req=yes: Is there a plastic bag in the image?
    o2 [] existence req=yes: Are there oranges in the image?
    o3 [o1,o2] final: Are the oranges to the left or to the right of the plastic bag?
  planner_v3
    c1 []: Is there a bag made of plastic in the image?
    c2 []: Is there a ripe orange in the image?
    c3 [c1,c2]: Are the ripe oranges to the left or to the right of the bag made of plastic?
19151786 'Is the bench to the right or to the left of the blue vehicle?' -> left
  oracle
    o1 [] existence req=yes: Is there a blue vehicle in the image?
    o2 [] existence req=yes: Is there a bench in the image?
    o3 [o1,o2] final: Is the bench to the right or to the left of the blue vehicle?
  planner_v3
    c1 []: Is there a bench in the image?
    c2 []: Is there a blue vehicle in the image?
    c3 [c1,c2]: Is the bench to the right or to the left of the blue vehicle?
05480434 'Is the elephant to the left or to the right of the car?' -> right
  oracle
    o1 [] existence req=yes: Is there a car in the image?
    o2 [] existence req=yes: Is there an elephant in the image?
    o3 [o1,o2] final: Is the elephant to the left or to the right of the car?
  planner_v3
    c1 []: Is there an elephant in the image?
    c2 []: Is there a car in the image?
    c3 [c1,c2]: Is the elephant to the left of the car?
16880810 'Is the pillow to the left or to the right of the woman on the right?' -> left
  oracle
    o1 [] existence req=yes: Is there a woman on the right in the image?
    o2 [] existence req=yes: Is there a pillow in the image?
    o3 [o1,o2] final: Is the pillow to the left or to the right of the woman on the right?
  planner_v3
    c1 []: Is there a woman on the right in the image?
    c2 [c1]: Is there a pillow to the left or to the right of the woman on the right?
```
**d_verify_rel**
```
16654472 'Is the ottoman to the left of a bed?' -> yes
  oracle
    o1 [] existence req=yes: Is there an ottoman in the image?
    o2 [o1] final: Is the ottoman to the left of a bed?
  planner_v3
    c1 []: Is there an ottoman to the left of a bed in the image?
    c2 [c1]: Is the ottoman to the left of a bed?
13877235 'Are there any books on the floor?' -> yes
  oracle
    o1 [] existence req=yes: Is there a floor in the image?
    o2 [o1] final: Are there books on the floor?
  planner_v3
    c1 []: Is there a book on the floor in the image?
    c2 [c1]: Are there any books on the floor?
00777462 'Is the bird on the rock watching a boar?' -> no
  oracle
    o1 [] existence req=yes: Is there a rock in the image?
    o2 [o1] existence req=yes: Is there a bird on the rock?
    o3 [o2] final: Is the bird on the rock watching a boar?
  planner_v3
    c1 []: Is there a bird on a rock in the image?
    c2 [c1]: Is the bird on the rock watching a boar?
07883808 'Is the knife on a cutting board?' -> no
  oracle
    o1 [] existence req=yes: Is there a knife in the image?
    o2 [o1] final: Is the knife on a cutting board?
  planner_v3
    c1 []: Is there a knife in the image?
    c2 [c1]: Is the knife on a cutting board?
09249815 'Is he carrying a surfboard?' -> no
  oracle  flags: pronoun: he
    o1 [] existence req=yes: Is there a man in the image?
    o2 [o1] final: Is the man carrying a surfboard?
  planner_v3
    c1 []: Is there a person carrying a surfboard in the image?
    c2 [c1]: Is the person carrying a surfboard?
```
**e_logical**
```
121053010 'Do you see any doors or cars that are white?' -> yes
  oracle
    o1 [] final/or: Is there a white door in the image?
    o2 [] final/or: Is there a white car in the image?
  planner_v3
    c1 []: Is there a door or a car in the image?
    c2 [c1]: What color is the door or car?
03848528 'Are there any hairbrushes or bottle caps?' -> no
  oracle
    o1 [] final/or: Is there a hairbrush in the image?
    o2 [] final/or: Is there a bottle cap in the image?
  planner_v3
    c1 []: Is there a hairbrush or a bottle cap in the image?
    c2 [c1]: What color is the hairbrush or bottle cap?
0598390 'Are there either pillows or trays?' -> yes
  oracle
    o1 [] final/or: Is there a pillow in the image?
    o2 [] final/or: Is there a tray in the image?
  planner_v3
    c1 []: Is there a pillow in the image?
    c2 []: Is there a tray in the image?
    c3 [c1,c2]: Are there either pillows or trays?
08206123 'Are there any glasses or stoves in the scene?' -> no
  oracle
    o1 [] final/or: Is there a stove in the image?
    o2 [] final/or: Are there glasses in the image?
  planner_v3
    c1 []: Is there a glass in the image?
    c2 []: Is there a stove in the image?
    c3 [c1,c2]: Are there any glasses or stoves in the image?
061043477 'Are there unpeeled bananas or apples?' -> no
  oracle
    o1 [] final/or: Is there an unpeeled banana in the image?
    o2 [] final/or: Is there an unpeeled apple in the image?
  planner_v3
    c1 []: Is there an unpeeled banana in the image?
    c2 []: Is there an unpeeled apple in the image?
    c3 [c1,c2]: Are there unpeeled bananas or apples in the image?
```
**f_compare**
```
00768430 'Do the van and the hydrant have the same color?' -> no
  oracle
    o1 [] existence req=yes: Is there a fire hydrant in the image?
    o2 [] existence req=yes: Is there a van in the image?
    o3 [o1,o2] final: Do the fire hydrant and the van have the same color?
  planner_v3
    c1 []: Is there a van in the image?
    c2 []: Is there a hydrant in the image?
    c3 [c1]: What color is the van?
04727485 'Are the barrier and the pole made of the same material?' -> no
  oracle
    o1 [] existence req=yes: Is there a barrier in the image?
    o2 [] existence req=yes: Is there a pole in the image?
    o3 [o1,o2] final: Are the barrier and the pole made of the same material?
  planner_v3
    c1 []: Is there a barrier in the image?
    c2 []: Is there a pole in the image?
    c3 [c1]: What material is the barrier made of?
13831520 'Is the color of the scarf different than the roof?' -> yes
  oracle
    o1 [] existence req=yes: Is there a scarf in the image?
    o2 [] existence req=yes: Is there a roof in the image?
    o3 [o1,o2] final: Do the scarf and the roof have different colors?
  planner_v3
    c1 []: Is there a scarf in the image?
    c2 []: Is there a roof in the image?
    c3 [c1]: What color is the scarf?
17401329 'What is common to the umbrella and the scarf?' -> color
  oracle
    o1 [] existence req=yes: Is there an umbrella in the image?
    o2 [] existence req=yes: Is there a scarf in the image?
    o3 [o1,o2] final: What is common to the umbrella and the scarf?
  planner_v3
    c1 []: Is there an umbrella in the image?
    c2 []: Is there a scarf in the image?
    c3 [c1,c2]: What material are the umbrella and the scarf made of?
1129358 'Is the frame made out of the same material as the bookcase?' -> yes
  oracle
    o1 [] existence req=yes: Is there a frame in the image?
    o2 [] existence req=yes: Is there a bookcase in the image?
    o3 [o1,o2] final: Are the frame and the bookcase made of the same material?
  planner_v3
    c1 []: Is there a frame in the image?
    c2 []: Is there a bookcase in the image?
    c3 [c1,c2]: Is the frame made out of the same material as the bookcase?
```

Item counts:

| type | oracle items (per question) | planner_v3 items | oracle calls (× 20 levels) |
|---|---|---|---|
| a_relO | 15 (3 each) | 10 | 300 |
| b_relS | 15 (3 each) | 10 | 300 |
| c_choose_rel | 16 (4, 3, 3, 3, 3) | 14 | 320 |
| d_verify_rel | 11 (2, 2, 3, 2, 2) | 10 | 220 |
| e_logical | 10 (2 each) | 13 | 200 |
| f_compare | 15 (3 each) | 15 | 300 |
| **total** | **82** | 72 | **1,640** |

## 5. Flagged questions

No question fails the rules: all 30 plans are `sensible`, have 2 to 4 items and pass the leak check. These plans are sensible but worth a look:

- **16403783 "Where are the trees?" (relO, place target).** The related-object check "Are the trees in some place?" is near-vacuous: it should always be yes. The final item is the question verbatim. GT "forest" is an open label, and this question is unanswerable on screening.
- **19419643 (choose rel).** "ripe" is not in the program, so the items say "oranges", not "ripe oranges". This is faithful to the program but drops a word of the question.
- **061043477 (logical).** The program attaches "unpeeled" to apples as well ("Is there an unpeeled apple?"). The question is ambiguous on this point.
- **03848528, 08206123, 061043477 (logical, GT "no").** Under OR = max, one confident "no" disjunct can stop alone. See P_oracle_kleene in §3.
- **17401329 (common, GT "color").** The final item asks the GQA question itself. The observer may answer "blue" instead of "color". This only matters for the gate (resolved and not "unknown"), because the reader's answer is the one scored.
- **00768430 (compare).** The program noun is "fire hydrant"; the question says "hydrant". This is harmless.
- **Expected A1-likeness.** On relO, relS and verify × rel, the final item is the question verbatim or nearly so, as in planner_v3. There, P_oracle differs from A1 only through the existence checks. Those include the new related-object checks ("Is the man wearing something?"), which planner_v3 never made. The decisive types are choose × rel, logical and compare, where planner_v3 hijacked or garbled the last item.

## 6. Planned calls (dry run)

```
[dry-run] oracle infer: {"questions": 30, "levels": 20, "items": 82, "items_by_type": {"a_relO": 15, "b_relS": 15, "c_choose_rel": 16, "d_verify_rel": 11, "e_logical": 10, "f_compare": 15}, "calls_by_type": {"a_relO": 300, "b_relS": 300, "c_choose_rel": 320, "d_verify_rel": 220, "e_logical": 200, "f_compare": 300}, "planned_calls": 1640, "record_files_present": 0, "remaining_calls": 1640}
```

- **1,640 observer_v3_lp calls.** At the probe's 0.62 s per call this is about 17 min, or 2 chunks of ≤ 585 s. The determinism gate adds 84 calls, about 1 min.
- **Reused, not re-requested.** reader_lp (600 records), observer_v3_lp_direct (600), plans_v3 (30) and planner_v3 items (1,440 records) are read from `probe_multi_rel/` through a cache-only engine. That engine checks every cache key and has a transport that raises on any send.
- **Verified without inference.** The cache-only replay rebuilt all 600 probe rows with 0 model calls. reader_pans, dres_score, pans_min and reader_answer are identical to `probe_multi_rel/level_table_dev.csv`. The `table` step repeats this check every time it runs.
- **Oracle record path.** `outputs/gqa_stop/probe_oracle/observer_v3_lp/qwen3.5:9b@<digest>/parse-v2/<qid>/<LL>_o<k>.json`. The request is role observer_v3, namespace `lp`, input {"question": item text}, suffix `_o<k>`. The suffix differs from planner_v3's `_c<k>`, so an oracle item with the same text as a planner item is still a new call.

## 7. Command sequence (run later, when the GPU is free)

`v27-envcheck` unloads every other resident model. Run it only after the qwen3.5:4b job has finished.

```
C=configs/gqa_stop_probe_oracle.json
# 0. plans are already written; re-running is a no-op (refuses to change them)
uv run python scripts/gqa_probe_oracle_run.py plans --config $C
uv run python scripts/gqa_probe_oracle_run.py infer --config $C --dry-run
# 1. same preflight as the probe, recorded in the probe_oracle run dir
uv run python -m milestone1.gqa_stop.cli preflight --config $C
uv run python -m milestone1.gqa_stop.cli v27-envcheck --config $C
uv run python -m milestone1.gqa_stop.cli v27-detgate --config $C
# 2. inference in foreground chunks (resumable; time checked between levels)
timeout 585 uv run python scripts/gqa_probe_oracle_run.py infer --config $C --max-seconds 540   # x2, until "complete": true
# 3. audit, table, evaluation (no model calls)
uv run python -m milestone1.gqa_stop.cli v27-audit --split dev --config $C
uv run python scripts/gqa_probe_oracle_run.py table --config $C
uv run python scripts/gqa_probe_oracle_eval.py --config $C
```

- **Gates in `infer`.** It refuses to run unless preflight, envcheck and detgate have passed in the probe_oracle state. The live digest must equal probe_multi_rel's digest. `plans.json` must not have changed since the first chunk.
- **Gates in `table`.** It refuses to run unless inference is complete and the plan hash matches.
- **`v27-identity` and `v27-screen` are not needed.** No reader call is made, and the labels come from probe_multi_rel.
- **Evaluation outputs** go to `outputs/gqa_stop/probe_oracle/eval/`. The methods are B1, A1, P_min (planner_v3), P_oracle_min, P_oracle_logic, P_oracle_logic_noveto and P_oracle_kleene.
  - `sweep.csv` and `sweep_by_type.csv`: the strict and unified θ sweep.
  - `theta_star.json`: the in-sample θ*.
  - `at_theta.csv`: results at θ 0.80 and at each in-sample θ*.
  - `agreement.csv`: per type, the correlation with A1 and P_min, the number of questions whose stop level differs at θ 0.80, and wins and losses.
  - `veto.json`: object_absent rows by type and label, and blocked stops. A blocked stop is one where P_oracle_logic_noveto stops and P_oracle_logic does not stop at that level. Blocked stops are split into unanswerable questions now correct and answerable questions lost.
  - `per_question.csv`.
- **Smoke test.** The full infer → table → eval path ran end to end on a fake transport in a scratch directory. That run covered 1,640 fake calls, a 0.5 s chunk stop and resume, and the reference check. Nothing was written to the real run dir.

## 8. Assumptions and caveats

- **Item roles for OR / AND.** The OR / AND operands are `role = "final"` items with a `logic` flag, not existence items, and they have no required answer. The veto therefore never applies to them.
- **Choose subjects are presupposed.** The `choose rel` subject gets a required-yes existence item. Without it, "the counter" in 02584384 would go unchecked.
- **Verify-rel objects are not presupposed.** "a bed" or "a surfboard" is the content of the final yes/no item. It gets no existence item, because a required "yes" there would encode the answer.
- **Dependency gating is kept.** Plain min reproduces the current P rule. The veto acts on top of the gate.
- **n = 30, in-sample θ.** There are 7 unanswerable questions, none of them in choose × rel or verify × rel. One question moves an unanswerable rate by 0.14. The veto's value on unanswerable questions can only be seen where a presupposed object is reported absent. On the 7 unanswerable probe questions, the failure is mostly an open-name label mismatch, not an absent object, so few veto hits are expected there.
