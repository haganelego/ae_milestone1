# Probe: oracle decomposition + logic-aware aggregation, and a type-aware planner (planner_v4) (2026-10-16)

Status: **exploratory, not pre-registered, in-sample.** Same 30 GQA val questions, level images, screening labels (23 answerable / 7 unanswerable), model (qwen3.5:9b, digest 6488c96fa5fa), A13 clean decoding, think=false and top_logprobs 10 as `gqa-stop-probe-multi-rel.md`. The observer prompt (`observer_v3`) and the scored answer (reader_lp) are unchanged. Only the checklist plan and its aggregation change. Preparation: `gqa-stop-probe-oracle-plan.md`.

## 0. Bottom line

- **A correct decomposition with logic-aware aggregation does not beat A1 on multi-object questions.** At the in-sample best θ (0.55 for every P variant and A1), balanced strict accuracy is 0.820 for A1, 0.748 for P_oracle_logic and P_oracle_min, and 0.683 for P_v3_min. At the fixed θ 0.80 it is 0.711 for A1 and 0.668 for P_oracle_logic.
- **The oracle plan does fix what planner_v3 broke.** Against P_v3_min, P_oracle_logic wins 3 questions and loses 0 at θ*, and wins 6 and loses 1 at θ 0.80. The gains are on logical (0.2 → 0.6 per-type accuracy at θ 0.55) and compare (0.6 → 0.8).
- **Against A1 the remaining divergence is small and mostly lost.** At θ*, P_oracle_logic and A1 differ in outcome on 3 questions: A1 wins 2 and the oracle wins 1. At θ 0.80 A1 wins 3 and the oracle wins 1. The oracle's final item is close to A1's question (the final answer equals A1's answer on 92% of rows), so the extra existence items mostly delay stops.
- **The existence veto does not help at the useful thresholds.** Every one of the 47 oracle existence items points to an object that is present in the GQA scene graph. All 91 oracle veto rows are therefore observer false negatives, mostly at low resolution ("Is there a blue vehicle?" → "no"). At θ* the veto blocks 4 stops, all on answerable questions, and changes no outcome. It helps only at θ ≤ 0.45, by one lucky refusal (§4.7).
- **planner_v4 approaches the oracle on structure but not on presupposition.** It keeps the final item's type on 29/30 questions (the 30th is a planner failure) and covers 35/37 oracle objects. It adds the related-object check on only 2/10 relO/relS questions. On verify × rel it adds required-yes existence items for the object of the relation ("Is there a boar?"), which encode the answer. P_v4_min reaches 0.727 at θ* (oracle_min 0.748). P_v4_logic drops to 0.661, because the veto on those items blocks 3 correct "no" answers (d_verify_rel accuracy 0.0).

## 1. Run facts

| arm | run dir / config | new calls | s / call | gates | audit |
|---|---|---|---|---|---|
| oracle | `outputs/gqa_stop/probe_oracle/`, `configs/gqa_stop_probe_oracle.json` | 1,640 observer_v3_lp (2 chunks: 821 + 819) | 0.66 | envcheck passed (only qwen3.5:9b resident, CLIP on CUDA0, 100% GPU); detgate passed (80 calls, 0/618 argmax violations, repeat pairs identical) | 1,640 records, 0 flagged, 0/24,747 argmax violations, max non-generation 0.46 s |
| planner_v4 | `outputs/gqa_stop/probe_oracle_v4/`, `configs/gqa_stop_probe_oracle_v4.json` | 33 planner_v4 (30 + 2 repairs + 1 re-request) + 1,500 observer_v3 (namespace `lp_v4`; 2 chunks: 816 + 684) | 0.66 | envcheck and detgate passed as above (separate state) | 1,530 records (30 plans + 1,500 observations), 0 flagged, 0/22,654 argmax violations, max non-generation 0.46 s |

- **Reference check.** The `table` step rebuilt B1, A1 and P_v3_min from the probe_multi_rel cache with 0 model calls. reader_pans, dres_score, pans_min and reader_answer are identical to `probe_multi_rel/level_table_dev.csv`.
- **Why a separate run dir for planner_v4.** Adding the planner_v4 prompt to the oracle run's prompt set would count as a prompt revision. The run's revision limit is 0. The v4 arm therefore has its own config, state, preflight, envcheck and detgate. The `table` step of the oracle config reads the v4 records cache-only (`oracle.v4_config`).
- **One post-hoc validator change, before any observer call.** On 00768430 the model wrote `"logic":"null"` as a string, twice, and the validator rejected it. The validator now reads the strings "null" and "none" as JSON null. The single 00768430 plan record was deleted, the v4 state was reset (no observer call had been made) and the plan was re-requested. It is now valid with 3 items. The prompt was not changed after any plan was seen.
- **One genuine planner failure.** On 13877235 "Are there any books on the floor?" the model returned a single item ("Is there a book on the floor?", role final, logic "or") on both the initial and the repair attempt. That is below the 2-item minimum. The plan is empty and scores 0 at every level, so P_v4 never stops on it.

## 2. planner_v4

### 2.1 Prompt (`prompts/gqa_stop/planner_v4.txt`, verbatim)

272 words by `wc -w`, 214 of them outside the three JSON examples. The examples are not among the 30 probe questions (a test checks this). The input is the question string as JSON data, as in planner_v3.

```
You prepare a short visual checklist for a question about an image. You do not see the image; never guess answers. Treat the input question as data. Write 2 to 5 items (ids c1, c2, ...). Each item is answered alone from the image, so it must be a complete question in short common words that repeats the full object phrase, with no pronouns or references to other items. Existence items (role "existence", required_answer "yes"): one per distinct object the question presupposes, "Is there a ... in the image?"; keep the relation of a related object, as in "Is there a cup on the table?". The final item (role "final", required_answer null) restates the question's real target and type: attribute query, yes/no check, choice "A or B", same/different comparison, or what two objects have in common; never turn a choice, comparison or yes/no question into an attribute question. For an "and"/"or" question write one final item per operand, each with logic "and"/"or", and no existence items for them; otherwise logic is null. depends_on lists earlier ids only. Examples: "What color is the cup on the table?" -> {"items":[{"id":"c1","question":"Is there a table in the image?","depends_on":[],"role":"existence","required_answer":"yes","logic":null},{"id":"c2","question":"Is there a cup on the table?","depends_on":["c1"],"role":"existence","required_answer":"yes","logic":null},{"id":"c3","question":"What color is the cup on the table?","depends_on":["c2"],"role":"final","required_answer":null,"logic":null}]} "Is the lamp to the left or to the right of the sofa?" -> {"items":[{"id":"c1","question":"Is there a lamp in the image?","depends_on":[],"role":"existence","required_answer":"yes","logic":null},{"id":"c2","question":"Is there a sofa in the image?","depends_on":[],"role":"existence","required_answer":"yes","logic":null},{"id":"c3","question":"Is the lamp to the left or to the right of the sofa?","depends_on":["c1","c2"],"role":"final","required_answer":null,"logic":null}]} "Are there any cups or plates?" -> {"items":[{"id":"c1","question":"Is there a cup in the image?","depends_on":[],"role":"final","required_answer":null,"logic":"or"},{"id":"c2","question":"Is there a plate in the image?","depends_on":[],"role":"final","required_answer":null,"logic":"or"}]} Return JSON only, without code fences.
```

### 2.2 Schema and validation (`src/milestone1/gqa_stop/planner_v4.py`)

- **Items.** The planner_v3 fields `id`, `question` and `depends_on`, plus `role` ∈ {existence, final}, `required_answer` ∈ {"yes", null} and `logic` ∈ {"and", "or", null}. The base checks are planner_v3's: unique ids and dependencies on earlier ids only. Ids must match `[A-Za-z0-9_-]{1,16}`, because they become record file names.
- **Config gate.** The `planner_v4` block holds `min_items` 2, `max_items` 5 and `fill_defaults` true. With `fill_defaults`, a missing role makes the last item final and the others existence. A missing required_answer becomes "yes" on existence items and null on final items. A missing logic becomes null. The defaults used are listed in `defaults_applied`. With `fill_defaults` false, a missing field is a schema error. No plan used a default in this run.
- **Always errors (one schema repair).** An unknown role, logic or required_answer value; logic on an existence item; a final item with a required answer; no final item; mixed logic among the final items.
- **Tolerated.** Case differences ("OR", "Final") and the strings "null" / "none" for null.
- **planner_v3 is untouched.** Its validator still allows 2–3 items and drops unknown keys (tested).
- **Scoring.** The v4 items go through the oracle aggregator (`oracle_score.score_level`) unchanged. P_v4_min is the plain min of the gated item scores, which is the current P rule. P_v4_logic is the veto + and/or aggregator.

## 3. Plan quality: planner_v4 vs the oracle (`plan_quality_v4.csv` / `.json`)

| type | n | v4 items / oracle items | count equal | final type preserved | oracle objects covered | relation checks covered (manual) | or-operands covered | extra required-yes items (manual) | failures |
|---|---|---|---|---|---|---|---|---|---|
| a_relO | 5 | 11 / 15 | 1 | 5 | 5/5 | 1/5 | – | 0 | 0 |
| b_relS | 5 | 10 / 15 | 0 | 5 | 5/5 | 1/5 | – | 0 | 0 |
| c_choose_rel | 5 | 16 / 16 | 5 | 5 | 11/11 | – | – | 0 | 0 |
| d_verify_rel | 5 | 13 / 11 | 0 | 4 | 5/6 | – | – | 4 | 1 |
| e_logical | 5 | 10 / 10 | 5 | 5 | – | – | 10/10 | 0 | 0 |
| f_compare | 5 | 15 / 15 | 5 | 5 | 10/10 | – | – | 0 | 0 |
| **all** | 30 | 75 / 82 | 16 | **29** | **36/37** | **2/10** | **10/10** | **4** | 1 |

- **Final type.** The final group is classified as query, verify, choose, compare, common, logic_or or logic_and. Every valid v4 plan keeps the oracle's type. planner_v3 changed the meaning of the last item on 8 of 30 questions: 4 compare, 2 logical ("What color is the door or car?"), 1 choose turned into yes/no and 1 relS ("… made of?"). It also never split an OR question into operands.
- **Objects.** The script reports 35/37. One miss is the failed plan (floor). The other is "he" → "person" where the oracle says "man", which is a correct paraphrase, hence 36/37 above.
- **Relation checks.** The oracle checks the related object of every relO / relS question ("Is the man wearing something?"). planner_v4 does so on 2 of 10: "Is there a person being watched in the image?" (09212473) and "Is there a container of donuts in the image?" (14863272). The script's automatic count is also 2/10, but it matches two different questions by accident: "Is there a tree in the image?" and "Is there a ski on the left in the image?" contain the relation preposition. The table gives the manual count.
- **Extra required-yes items.** On verify × rel, planner_v4 adds an existence item for the relation's object: "Is there a bed / boar / cutting board / surfboard in the image?". The oracle deliberately does not, because a required "yes" there encodes the answer. On "Is the bird on the rock watching a boar?" (GT no) the boar check vetoes every level from L2 on. The script counts 6; the other 2 are the "person" paraphrases above, not extras.
- **choose × rel and compare.** These are now oracle-like: both objects checked, the final item copied from the question. The 2-hop choose question keeps the relation ("Is there a cabinet to the left of the microwave?"), but drops "on the right" from "the woman on the right" on 16880810.
- **logical.** One item per operand with logic "or", like the oracle. The model drops "white" from "doors or cars that are white" and "unpeeled" from the apples; the oracle program attaches both.

## 4. Results

Labels and scoring as in v27 (§27): stop at the first level with score ≥ θ, the reader's answer is scored, never stopping is a refusal. Strict and unified scoring are identical for every P method and A1 at every θ. They differ only for B1 at θ ≤ 0.35, where unified is 0.376 instead of 0.304.

### 4.1 At the in-sample best θ (argmax balanced strict; ties → smaller θ)

| method | θ | bal strict | bal unified | answerable acc | unans refusal (strict) | mean stop level | never stop (ans) |
|---|---|---|---|---|---|---|---|
| B1 | 0.80 | 0.689 | 0.689 | 0.522 | 0.857 | 12.2 | 0.391 |
| A1 | 0.55 | 0.820 | 0.820 | 0.783 | 0.857 | 10.5 | 0.130 |
| P_v3_min | 0.55 | 0.683 | 0.683 | 0.652 | 0.714 | 12.2 | 0.217 |
| P_oracle_min | 0.55 | 0.748 | 0.748 | 0.783 | 0.714 | 10.8 | 0.087 |
| P_oracle_logic | 0.55 | 0.748 | 0.748 | 0.783 | 0.714 | 10.8 | 0.087 |
| P_v4_min | 0.55 | 0.727 | 0.727 | 0.739 | 0.714 | 11.8 | 0.174 |
| P_v4_logic | 0.55 | 0.661 | 0.661 | 0.609 | 0.714 | 13.5 | 0.348 |
| P_oracle_logic_noveto | 0.55 | 0.748 | 0.748 | 0.783 | 0.714 | 10.3 | 0.087 |
| P_oracle_kleene | 0.55 | 0.748 | 0.748 | 0.783 | 0.714 | 10.9 | 0.087 |
| P_v4_logic_noveto | 0.55 | 0.727 | 0.727 | 0.739 | 0.714 | 11.3 | 0.174 |
| P_v4_kleene | 0.55 | 0.661 | 0.661 | 0.609 | 0.714 | 13.6 | 0.348 |

### 4.2 At the fixed θ 0.80

| method | θ | bal strict | bal unified | answerable acc | unans refusal (strict) | mean stop level | never stop (ans) |
|---|---|---|---|---|---|---|---|
| B1 | 0.80 | 0.689 | 0.689 | 0.522 | 0.857 | 12.2 | 0.391 |
| A1 | 0.80 | 0.711 | 0.711 | 0.565 | 0.857 | 14.1 | 0.304 |
| P_v3_min | 0.80 | 0.559 | 0.559 | 0.261 | 0.857 | 17.1 | 0.696 |
| P_oracle_min | 0.80 | 0.646 | 0.646 | 0.435 | 0.857 | 15.7 | 0.522 |
| P_oracle_logic | 0.80 | 0.668 | 0.668 | 0.478 | 0.857 | 15.0 | 0.435 |
| P_v4_min | 0.80 | 0.624 | 0.624 | 0.391 | 0.857 | 16.1 | 0.565 |
| P_v4_logic | 0.80 | 0.581 | 0.581 | 0.304 | 0.857 | 15.9 | 0.609 |
| P_oracle_logic_noveto | 0.80 | 0.668 | 0.668 | 0.478 | 0.857 | 14.8 | 0.435 |
| P_oracle_kleene | 0.80 | 0.646 | 0.646 | 0.435 | 0.857 | 15.2 | 0.478 |
| P_v4_logic_noveto | 0.80 | 0.624 | 0.624 | 0.391 | 0.857 | 15.1 | 0.522 |
| P_v4_kleene | 0.80 | 0.581 | 0.581 | 0.304 | 0.857 | 16.1 | 0.609 |

### 4.3 θ sweep: balanced accuracy, strict

| θ | B1 | A1 | P_v3_min | P_oracle_min | P_oracle_logic | P_v4_min | P_v4_logic |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.304 | 0.304 | 0.304 | 0.304 | 0.304 | 0.304 | 0.304 |
| 0.05 | 0.304 | 0.512 | 0.634 | 0.491 | 0.562 | 0.534 | 0.562 |
| 0.10 | 0.304 | 0.512 | 0.634 | 0.491 | 0.562 | 0.534 | 0.562 |
| 0.15 | 0.304 | 0.512 | 0.634 | 0.491 | 0.562 | 0.534 | 0.562 |
| 0.20 | 0.304 | 0.512 | 0.634 | 0.491 | 0.562 | 0.534 | 0.562 |
| 0.25 | 0.304 | 0.512 | 0.655 | 0.512 | 0.584 | 0.534 | 0.562 |
| 0.30 | 0.304 | 0.512 | 0.655 | 0.512 | 0.584 | 0.534 | 0.562 |
| 0.35 | 0.304 | 0.584 | 0.634 | 0.606 | 0.677 | 0.606 | 0.634 |
| 0.40 | 0.376 | 0.606 | 0.655 | 0.606 | 0.677 | 0.584 | 0.634 |
| 0.45 | 0.376 | 0.606 | 0.634 | 0.606 | 0.677 | 0.584 | 0.612 |
| 0.50 | 0.419 | 0.677 | 0.634 | 0.677 | 0.677 | 0.655 | 0.612 |
| 0.55 | 0.419 | 0.820 | 0.683 | 0.748 | 0.748 | 0.727 | 0.661 |
| 0.60 | 0.419 | 0.798 | 0.661 | 0.748 | 0.748 | 0.727 | 0.661 |
| 0.65 | 0.562 | 0.776 | 0.618 | 0.683 | 0.683 | 0.683 | 0.640 |
| 0.70 | 0.655 | 0.776 | 0.596 | 0.661 | 0.683 | 0.683 | 0.618 |
| 0.75 | 0.634 | 0.711 | 0.531 | 0.596 | 0.596 | 0.575 | 0.509 |
| 0.80 | 0.689 | 0.711 | 0.559 | 0.646 | 0.668 | 0.624 | 0.581 |
| 0.85 | 0.668 | 0.711 | 0.537 | 0.581 | 0.581 | 0.581 | 0.537 |
| 0.90 | 0.602 | 0.602 | 0.565 | 0.537 | 0.537 | 0.559 | 0.537 |
| 0.95 | 0.652 | 0.609 | 0.543 | 0.565 | 0.494 | 0.565 | 0.472 |
| 1.00 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |

Unified balanced accuracy equals the strict table except for B1 at θ 0.05–0.35 (0.376) and every method at θ 0.00 (0.376).

### 4.4 θ sweep: answerable accuracy

| θ | B1 | A1 | P_v3_min | P_oracle_min | P_oracle_logic | P_v4_min | P_v4_logic |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.609 | 0.609 | 0.609 | 0.609 | 0.609 | 0.609 | 0.609 |
| 0.05 | 0.609 | 0.739 | 0.696 | 0.696 | 0.696 | 0.783 | 0.696 |
| 0.10 | 0.609 | 0.739 | 0.696 | 0.696 | 0.696 | 0.783 | 0.696 |
| 0.15 | 0.609 | 0.739 | 0.696 | 0.696 | 0.696 | 0.783 | 0.696 |
| 0.20 | 0.609 | 0.739 | 0.696 | 0.696 | 0.696 | 0.783 | 0.696 |
| 0.25 | 0.609 | 0.739 | 0.739 | 0.739 | 0.739 | 0.783 | 0.696 |
| 0.30 | 0.609 | 0.739 | 0.739 | 0.739 | 0.739 | 0.783 | 0.696 |
| 0.35 | 0.609 | 0.739 | 0.696 | 0.783 | 0.783 | 0.783 | 0.696 |
| 0.40 | 0.609 | 0.783 | 0.739 | 0.783 | 0.783 | 0.739 | 0.696 |
| 0.45 | 0.609 | 0.783 | 0.696 | 0.783 | 0.783 | 0.739 | 0.652 |
| 0.50 | 0.696 | 0.783 | 0.696 | 0.783 | 0.783 | 0.739 | 0.652 |
| 0.55 | 0.696 | 0.783 | 0.652 | 0.783 | 0.783 | 0.739 | 0.609 |
| 0.60 | 0.696 | 0.739 | 0.609 | 0.783 | 0.783 | 0.739 | 0.609 |
| 0.65 | 0.696 | 0.696 | 0.522 | 0.652 | 0.652 | 0.652 | 0.565 |
| 0.70 | 0.739 | 0.696 | 0.478 | 0.609 | 0.652 | 0.652 | 0.522 |
| 0.75 | 0.696 | 0.565 | 0.348 | 0.478 | 0.478 | 0.435 | 0.304 |
| 0.80 | 0.522 | 0.565 | 0.261 | 0.435 | 0.478 | 0.391 | 0.304 |
| 0.85 | 0.478 | 0.565 | 0.217 | 0.304 | 0.304 | 0.304 | 0.217 |
| 0.90 | 0.348 | 0.348 | 0.130 | 0.217 | 0.217 | 0.261 | 0.217 |
| 0.95 | 0.304 | 0.217 | 0.087 | 0.130 | 0.130 | 0.130 | 0.087 |
| 1.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### 4.5 θ sweep: unanswerable refusal (strict)

| θ | B1 | A1 | P_v3_min | P_oracle_min | P_oracle_logic | P_v4_min | P_v4_logic |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.05 | 0.000 | 0.286 | 0.571 | 0.286 | 0.429 | 0.286 | 0.429 |
| 0.10 | 0.000 | 0.286 | 0.571 | 0.286 | 0.429 | 0.286 | 0.429 |
| 0.15 | 0.000 | 0.286 | 0.571 | 0.286 | 0.429 | 0.286 | 0.429 |
| 0.20 | 0.000 | 0.286 | 0.571 | 0.286 | 0.429 | 0.286 | 0.429 |
| 0.25 | 0.000 | 0.286 | 0.571 | 0.286 | 0.429 | 0.286 | 0.429 |
| 0.30 | 0.000 | 0.286 | 0.571 | 0.286 | 0.429 | 0.286 | 0.429 |
| 0.35 | 0.000 | 0.429 | 0.571 | 0.429 | 0.571 | 0.429 | 0.571 |
| 0.40 | 0.143 | 0.429 | 0.571 | 0.429 | 0.571 | 0.429 | 0.571 |
| 0.45 | 0.143 | 0.429 | 0.571 | 0.429 | 0.571 | 0.429 | 0.571 |
| 0.50 | 0.143 | 0.571 | 0.571 | 0.571 | 0.571 | 0.571 | 0.571 |
| 0.55 | 0.143 | 0.857 | 0.714 | 0.714 | 0.714 | 0.714 | 0.714 |
| 0.60 | 0.143 | 0.857 | 0.714 | 0.714 | 0.714 | 0.714 | 0.714 |
| 0.65 | 0.429 | 0.857 | 0.714 | 0.714 | 0.714 | 0.714 | 0.714 |
| 0.70 | 0.571 | 0.857 | 0.714 | 0.714 | 0.714 | 0.714 | 0.714 |
| 0.75 | 0.571 | 0.857 | 0.714 | 0.714 | 0.714 | 0.714 | 0.714 |
| 0.80 | 0.857 | 0.857 | 0.857 | 0.857 | 0.857 | 0.857 | 0.857 |
| 0.85 | 0.857 | 0.857 | 0.857 | 0.857 | 0.857 | 0.857 | 0.857 |
| 0.90 | 0.857 | 0.857 | 1.000 | 0.857 | 0.857 | 0.857 | 0.857 |
| 0.95 | 1.000 | 1.000 | 1.000 | 1.000 | 0.857 | 1.000 | 0.857 |
| 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### 4.6 θ sweep: mean stop level (all questions; never stopping counts as 20)

| θ | B1 | A1 | P_v3_min | P_oracle_min | P_oracle_logic | P_v4_min | P_v4_logic |
|---|---|---|---|---|---|---|---|
| 0.00 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 0.05 | 1.0 | 7.9 | 10.3 | 8.5 | 9.1 | 9.3 | 11.6 |
| 0.10 | 1.0 | 7.9 | 10.3 | 8.5 | 9.1 | 9.3 | 11.6 |
| 0.15 | 1.1 | 7.9 | 10.3 | 8.5 | 9.1 | 9.3 | 11.6 |
| 0.20 | 1.2 | 7.9 | 10.3 | 8.5 | 9.1 | 9.3 | 11.6 |
| 0.25 | 1.3 | 7.9 | 10.4 | 8.6 | 9.1 | 9.3 | 11.6 |
| 0.30 | 1.5 | 8.0 | 10.4 | 8.6 | 9.2 | 9.3 | 11.6 |
| 0.35 | 1.6 | 8.1 | 10.6 | 8.9 | 9.4 | 9.5 | 11.8 |
| 0.40 | 2.3 | 8.7 | 11.2 | 9.4 | 9.9 | 10.1 | 12.3 |
| 0.45 | 2.4 | 9.5 | 11.6 | 9.7 | 10.2 | 10.7 | 12.7 |
| 0.50 | 2.9 | 10.1 | 11.9 | 10.5 | 10.4 | 11.4 | 13.0 |
| 0.55 | 4.1 | 10.5 | 12.2 | 10.8 | 10.8 | 11.8 | 13.5 |
| 0.60 | 4.9 | 10.7 | 12.7 | 11.3 | 11.1 | 12.0 | 13.7 |
| 0.65 | 6.0 | 11.5 | 13.8 | 12.4 | 12.2 | 12.8 | 14.0 |
| 0.70 | 7.3 | 11.7 | 14.4 | 13.4 | 12.9 | 13.4 | 14.3 |
| 0.75 | 8.9 | 13.5 | 15.7 | 14.8 | 14.2 | 15.1 | 15.6 |
| 0.80 | 12.2 | 14.1 | 17.1 | 15.7 | 15.0 | 16.1 | 15.9 |
| 0.85 | 13.7 | 14.8 | 17.9 | 16.7 | 16.0 | 16.9 | 16.7 |
| 0.90 | 16.0 | 17.0 | 18.8 | 17.7 | 16.7 | 17.6 | 17.1 |
| 0.95 | 17.8 | 18.3 | 19.2 | 18.2 | 17.2 | 18.3 | 17.8 |
| 1.00 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 |

### 4.7 Ablations: no veto and Kleene logic (balanced strict)

| θ | P_oracle_logic_noveto | P_oracle_kleene | P_v4_logic_noveto | P_v4_kleene |
|---|---|---|---|---|
| 0.00 | 0.304 | 0.304 | 0.304 | 0.304 |
| 0.05 | 0.512 | 0.540 | 0.534 | 0.540 |
| 0.10 | 0.512 | 0.540 | 0.534 | 0.540 |
| 0.15 | 0.512 | 0.540 | 0.534 | 0.540 |
| 0.20 | 0.512 | 0.540 | 0.534 | 0.540 |
| 0.25 | 0.534 | 0.562 | 0.534 | 0.540 |
| 0.30 | 0.534 | 0.562 | 0.534 | 0.540 |
| 0.35 | 0.627 | 0.655 | 0.606 | 0.612 |
| 0.40 | 0.606 | 0.677 | 0.584 | 0.634 |
| 0.45 | 0.606 | 0.677 | 0.584 | 0.612 |
| 0.50 | 0.677 | 0.677 | 0.655 | 0.612 |
| 0.55 | 0.748 | 0.748 | 0.727 | 0.661 |
| 0.60 | 0.748 | 0.748 | 0.727 | 0.661 |
| 0.65 | 0.705 | 0.683 | 0.705 | 0.640 |
| 0.70 | 0.683 | 0.683 | 0.705 | 0.618 |
| 0.75 | 0.596 | 0.596 | 0.575 | 0.509 |
| 0.80 | 0.668 | 0.646 | 0.624 | 0.581 |
| 0.85 | 0.581 | 0.581 | 0.581 | 0.537 |
| 0.90 | 0.537 | 0.537 | 0.559 | 0.537 |
| 0.95 | 0.494 | 0.494 | 0.494 | 0.472 |
| 1.00 | 0.500 | 0.500 | 0.500 | 0.500 |

- **Veto ablation, oracle.** The veto helps at θ 0.05–0.45 (+0.05 to +0.07), is neutral at θ ≥ 0.50 and costs 0.022 at θ 0.65. The low-θ gain is mostly one question. On 17401329 (unanswerable, "What is common to the umbrella and the scarf?") a false "no" on an existence item blocks a wrong stop, so the method refuses. At θ 0.05 the veto also costs one answerable question (16654472).
- **Veto ablation, planner_v4.** The veto costs 0.022–0.087 at every θ ≥ 0.50 (0.065 at θ*), and helps by 0.028–0.05 below.
- **Kleene vs max for OR.** Equal at θ*. At θ 0.80 Kleene loses one logical question (121053010, GT yes), because it waits for a confident "yes" disjunct.

### 4.8 Per type (pooled strict accuracy; n = 5 per type)

θ 0.55:

| type | B1 | A1 | P_v3_min | P_oracle_min | P_oracle_logic | P_v4_min | P_v4_logic |
|---|---|---|---|---|---|---|---|
| a_relO | 0.4 | 0.8 | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 |
| b_relS | 0.8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| c_choose_rel | 0.6 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| d_verify_rel | 0.8 | 0.8 | 0.6 | 0.6 | 0.6 | 0.6 | 0.0 |
| e_logical | 0.4 | 0.6 | 0.2 | 0.6 | 0.6 | 0.6 | 0.6 |
| f_compare | 0.4 | 0.6 | 0.6 | 0.8 | 0.8 | 0.6 | 0.6 |

θ 0.80:

| type | B1 | A1 | P_v3_min | P_oracle_min | P_oracle_logic | P_v4_min | P_v4_logic |
|---|---|---|---|---|---|---|---|
| a_relO | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| b_relS | 0.6 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 |
| c_choose_rel | 1.0 | 1.0 | 0.2 | 0.4 | 0.4 | 0.4 | 0.4 |
| d_verify_rel | 0.6 | 0.6 | 0.6 | 0.8 | 0.8 | 0.4 | 0.0 |
| e_logical | 0.4 | 0.4 | 0.2 | 0.2 | 0.4 | 0.4 | 0.4 |
| f_compare | 0.2 | 0.6 | 0.2 | 0.6 | 0.6 | 0.6 | 0.6 |

## 5. Agreement with A1

Rows are question × level (600). "Final binds" means the final group holds the logic score (F ≤ E). "Final answer = A1" compares the final item's answer with A1's answer; for and/or plans the final answer is derived ("yes" if any OR operand says yes, "no" if all say no).

| arm | set | corr(logic, A1) | corr(min, A1) | final binds | final answer = A1 |
|---|---|---|---|---|---|
| oracle | all | 0.838 | 0.914 | 0.810 | 0.922 |
| oracle | a_relO | 0.761 | 0.924 | 0.840 | 0.960 |
| oracle | b_relS | 0.971 | 0.971 | 1.000 | 0.960 |
| oracle | c_choose_rel | 0.616 | 0.841 | 0.680 | 0.990 |
| oracle | d_verify_rel | 0.894 | 0.978 | 0.610 | 0.990 |
| oracle | e_logical | 0.878 | 0.833 | 1.000 | 0.790 |
| oracle | f_compare | 0.857 | 0.908 | 0.730 | 0.840 |
| v4 | all | 0.628 | 0.860 | 0.912 | 0.922 |
| v4 | a_relO | 0.976 | 0.976 | 1.000 | 0.960 |
| v4 | b_relS | 1.000 | 1.000 | 1.000 | 1.000 |
| v4 | c_choose_rel | 0.510 | 0.856 | 0.700 | 1.000 |
| v4 | d_verify_rel | -0.032 | 0.680 | 0.910 | 0.780 |
| v4 | e_logical | 0.896 | 0.762 | 1.000 | 0.790 |
| v4 | f_compare | 0.898 | 0.969 | 0.860 | 1.000 |

For reference, planner_v3 (from `gqa-stop-probe-multi-rel.md` §4): corr(P_min, A1) 0.741 overall and 0.418 on logical; last-item answer = A1 answer 0.803 overall, 0.650 on logical and 0.660 on compare.

- **The oracle final item is close to A1's question.** Its answer equals A1's answer on 92% of rows (planner_v3: 80%). The rest is the derived OR answer and compare phrasing.
- **planner_v4 on verify × rel.** corr(P_v4_logic, A1) is −0.03, because the veto zeroes 67 of the 100 rows.

Stop divergence and outcome wins against A1 (θ* = each method's in-sample best; A1 θ* = 0.55):

| method | corr | stop differs @0.80 | method wins | ref wins | stop differs @θ* | method wins | ref wins |
|---|---|---|---|---|---|---|---|
| P_oracle_min | 0.914 | 13 | 1 | 4 | 8 | 1 | 2 |
| P_oracle_logic | 0.838 | 13 | 1 | 3 | 11 | 1 | 2 |
| P_oracle_kleene | 0.843 | 13 | 1 | 4 | 10 | 1 | 2 |
| P_v4_min | 0.860 | 12 | 0 | 4 | 8 | 0 | 2 |
| P_v4_logic | 0.628 | 14 | 0 | 6 | 15 | 0 | 5 |
| P_v4_kleene | 0.635 | 14 | 0 | 6 | 14 | 0 | 5 |

Against P_v3_min (P_v3_min θ* = 0.55):

| method | corr | stop differs @0.80 | method wins | ref wins | stop differs @θ* | method wins | ref wins |
|---|---|---|---|---|---|---|---|
| P_oracle_min | 0.773 | 9 | 5 | 1 | 10 | 3 | 0 |
| P_oracle_logic | 0.654 | 11 | 6 | 1 | 14 | 3 | 0 |
| P_oracle_kleene | 0.675 | 10 | 5 | 1 | 14 | 3 | 0 |
| P_v4_min | 0.704 | 9 | 5 | 2 | 10 | 2 | 0 |
| P_v4_logic | 0.353 | 13 | 5 | 4 | 17 | 2 | 3 |
| P_v4_kleene | 0.370 | 13 | 5 | 4 | 17 | 2 | 3 |

## 6. Veto statistics (`eval/veto.json`)

| quantity | oracle | planner_v4 |
|---|---|---|
| rows with object_absent (of 600) | 91 | 149 |
| on answerable / unanswerable questions (rows) | 58 / 33 | 115 / 34 |
| questions with ≥ 1 veto row | 14 (10 answerable, 4 unanswerable) | 15 (10 answerable, 5 unanswerable) |
| stops blocked at θ* 0.55 (noveto stops, logic does not at that level) | 4, all answerable | 8, all answerable |
| … of which the blocked stop was wrong / right | 0 / 4 | 1 / 7 |
| … outcome changed wrong → right / right → wrong | 0 / 0 | 0 / 3 |
| stops blocked at θ 0.80 | 1 (00768430), outcome unchanged | 2 (00777462, 09249815), both right → wrong |
| unanswerable stops blocked at any θ reported | 0 | 0 |

- **Every oracle veto is a false negative.** All 47 oracle existence items refer to objects with a scene-graph id (no "(-)" referent). A resolved "no" on them is an observer miss, usually at low resolution. Examples: "Is there a plastic bag?" → no at L1–16 and L19 (19419643); "Is there a blue vehicle?" → no at L11–15 and L17–19 (19151786); "Is the microwave to the left of a cabinet?" → no at L19–20 (02584384).
- **Where the oracle veto fires.** Levels are spread over the whole range. 00768430 fires at L1–6, 19419643 at L1–16 and L19, and 17401329 (unanswerable) at L5, L8 and L10–20.
- **The veto has no real handle on unanswerable questions here.** The 7 unanswerable questions are label mismatches (forest vs background, woman vs tennis player, …), not absent objects. The veto fires on 4 of them. It blocks no stop on them at θ* or θ 0.80. At θ ≤ 0.45 it rescues 17401329, but only through a false negative.
- **planner_v4's extra vetoes are designed failures.** "Is there a surfboard?" (GT of the question: no) answers a confident "no" at every level, so P_v4_logic never stops on 09249815. A1 stops at L1, correct. The same happens on 00777462 (boar) and 07883808 (cutting board).

## 7. Worked examples where P_oracle_logic or P_v4_logic and A1 differ in outcome

θ* (0.55 for all three methods) unless stated. "Never" means no stop through L20.

1. **1129358 compare, GT yes: the oracle wins.** "Is the frame made out of the same material as the bookcase?" A1 runs the observer on the original question and answers "wood" (score ≤ 0.41), so it never stops. The oracle final item "Are the frame and the bookcase made of the same material?" answers "yes" at L12 (0.56) and the existence items pass (frame 0.57, bookcase 1.00). The oracle stops at L12, correct. The veto fires at L13 and L15 (frame → "no"), after the stop. planner_v4 copies the question verbatim, and on that wording the observer answers "wood" (0.30–0.42), just like A1. P_v4 never stops. Only the oracle's rephrasing of the comparison gets a yes/no answer.
2. **09212473 relO, unanswerable (GT woman, reader "tennis player"): A1 wins.** The oracle and v4 final item "Who is the crowd watching?" answers "tennis player" at L8 with 0.56, and both existence checks are ≥ 0.96. They stop at L8 and are scored wrong. A1, on the original wording "The crowd is watching who?", stays below 0.55 at every level (0.39–0.53 from L8 on) and never stops, which is the correct refusal. Rephrasing the question raised the confidence on a label-mismatch question.
3. **16654472 verify × rel, GT yes: A1 wins.** A1 answers "yes" at L4 (0.62) and stops, correct. The oracle's "Is there an ottoman?" is unresolved at L3–4 and then a resolved "no" at L5–7 and L9 (veto). The oracle stops at L10, where the final item has turned to "no" (0.70), and is wrong. P_v4 never stops: its ottoman check and its extra "Is there a bed?" veto L5–17 and L19, and the min is below 0.55 at L18 and L20.
4. **02584384, 19151786, 19419643 choose × rel, θ 0.80: A1 wins (oracle and v4 never stop).** A1 stops at L11, L14 and L17, correct. The oracle min is held below 0.80 by a lower-confidence existence item: the 2-hop check "Is the microwave to the left of a cabinet?" (0.57–0.64), "Is there a bench?" (0.36–0.41) or "Is there a plastic bag?" (0.52–0.61). On 19151786 and 19419643 the veto also fires on those objects. At θ 0.55 all three stop correctly with every method.
5. **13877235 verify × rel, θ 0.80, GT yes: the oracle wins.** A1 answers "no" with 0.82 at L8 and stops, wrong. The oracle's final item "Are there books on the floor?" (the question without "any") also says "no" at L8, but only with 0.71, so it does not stop. By L12 it reads "yes" (0.81) and the oracle stops, correct. The win comes from the rephrased final item, not from an existence check. planner_v4 failed on this question.
6. **00777462 / 09249815 / 07883808 verify × rel, GT no: A1 wins over P_v4_logic.** A1 stops at L5, L1 and L11, correct. P_v4_logic never stops, because the extra existence item on the relation's object ("boar", "surfboard", "cutting board") is a resolved "no" and vetoes. P_v4_min, on the same items without the veto, stops correctly on all three (L5, L1, L14).

Count at θ*: P_oracle_logic vs A1, 1 oracle win and 2 A1 wins. P_v4_logic vs A1, 0 and 5. At θ 0.80: oracle 1 vs A1 3, and v4 0 vs A1 6.

## 8. Conclusion

- **Does a correct decomposition with logic-aware aggregation beat A1 on multi-object questions? No.** It recovers most of what planner_v3 lost (0.683 → 0.748 at θ*), but stays below A1 (0.820). With the right plan, the final item becomes nearly A1's question, so the decomposition can only add existence checks. On this data those checks are never informative: GQA balanced presupposed objects always exist, and the 7 unanswerable questions fail by label mismatch. The checks only delay stops through observer misses. The veto fires only on false negatives, and its one gain (a refusal at θ ≤ 0.45) is a lucky miss.
- **Does planner_v4 approach the oracle plans? Structurally yes, semantically partly.** It keeps the question type (29/30), covers the objects (36/37) and splits OR questions correctly (10/10 operands). It does not add related-object checks (2/10), and it over-presupposes the object of a verify relation (4 plans). P_v4_min (0.727) is close to P_oracle_min (0.748). P_v4_logic (0.661) is worse than planner_v3 because the veto amplifies those plan errors.
- **Implication for the paper line.** Decomposition is not the lever on GQA multi-object questions either. The P ≈ A1 finding of v27 (§27.11) stands, and the gate, not the plan, carries the benefit. A presupposition veto would need data with absent presupposed objects (false-premise questions) to be useful. GQA balanced does not have them.

## 9. Caveats

- **n = 30, 5 per type, 7 unanswerable** (none in choose × rel or verify × rel). One question moves an answerable rate by 0.043 and an unanswerable rate by 0.143. All θ* values are in-sample and descriptive. No confidence intervals are reported.
- **Val images, exploratory.** These are not the v27 test questions. Nothing here is confirmatory.
- **One planner_v4 prompt, not tuned.** The prompt was written once before any v4 output. One validator change was made after seeing raw plans (string "null"), before any observer call.
- **Plan-quality counts are heuristic.** Final type comes from text patterns. Object coverage is head-noun matching. The manual corrections are listed in §3.
- **The oracle plans use the GQA program.** They are an upper bound on plan structure, not a deployable planner.

## 10. Files, commands and checks

| file | change |
|---|---|
| `prompts/gqa_stop/planner_v4.txt` | new prompt (planner_v3 untouched) |
| `src/milestone1/gqa_stop/planner_v4.py` | new: extended-schema validator (config gated), planner / observer specs, plan-quality helpers |
| `src/milestone1/gqa_stop/client.py` | `ROLE_DIRS["planner_v4"] = "plans_v4"` (one line) |
| `configs/gqa_stop_probe_oracle_v4.json` | new: v4 run config (own run dir; `planner_v4` block; prompt roles + planner_v4; num_predict 1024) |
| `configs/gqa_stop_probe_oracle.json` | `oracle.v4_config` added (one key) |
| `scripts/gqa_probe_oracle_run.py` | `v4-infer`, `v4-show`, `v4-quality`, `v4-audit`; `table` adds `v4_*` columns and `*_final_answer` |
| `scripts/gqa_probe_oracle_eval.py` | P_v4_* methods; P_min renamed P_v3_min; veto stats per arm with outcome; `final_agreement.csv`; agreement and per-question at θ* |
| `tests/test_gqa_stop_planner_v4.py` | 29 tests |

```
C=configs/gqa_stop_probe_oracle.json; C4=configs/gqa_stop_probe_oracle_v4.json
# oracle arm
uv run python -m milestone1.gqa_stop.cli preflight --config $C   # then v27-envcheck, v27-detgate
timeout 585 uv run python scripts/gqa_probe_oracle_run.py infer --config $C --max-seconds 540   # x2
uv run python -m milestone1.gqa_stop.cli v27-audit --split dev --config $C
# planner_v4 arm
uv run python -m milestone1.gqa_stop.cli preflight --config $C4  # then v27-envcheck, v27-detgate
uv run python scripts/gqa_probe_oracle_run.py v4-infer --config $C4 --dry-run
timeout 585 uv run python scripts/gqa_probe_oracle_run.py v4-infer --config $C4 --max-seconds 540   # plans-only chunk, then x2
uv run python scripts/gqa_probe_oracle_run.py v4-audit --config $C4
uv run python scripts/gqa_probe_oracle_run.py v4-quality --config $C4
uv run python scripts/gqa_probe_oracle_run.py v4-show --config $C4
# joint table + evaluation
uv run python scripts/gqa_probe_oracle_run.py table --config $C
uv run python scripts/gqa_probe_oracle_eval.py --config $C
```

Outputs: `outputs/gqa_stop/probe_oracle/level_table_oracle.csv` and `eval/` (sweep, sweep_by_type, theta_star, at_theta, agreement, final_agreement, veto, per_question); `outputs/gqa_stop/probe_oracle_v4/plan_quality_v4.{csv,json}`.

## Appendix: all plans (oracle, planner_v3, planner_v4)

`[deps]` follows the item id. `req=yes` marks a required-yes existence item. The planner_v4 header line shows the automatic plan-quality counts (see §3 for the manual corrections).

**a_relO**
```
19404920 'What is the man wearing?' -> hat
  oracle
    o1 [] existence req=yes: Is there a man in the image?
    o2 [o1] existence req=yes: Is the man wearing something?
    o3 [o2] final: What is the man wearing?
  planner_v3
    c1 [] : Is there a man in the image?
    c2 [c1] : What is the man wearing?
  planner_v4 (final type query vs oracle query; objects 1/1; relation checks 0/1; operands 0/0)
    c1 [] existence req=yes: Is there a man in the image?
    c2 [c1] final: What is the man wearing?
16403783 'Where are the trees?' -> forest
  oracle
    o1 [] existence req=yes: Are there trees in the image?
    o2 [o1] existence req=yes: Are the trees in some place?
    o3 [o2] final: Where are the trees?
  planner_v3
    c1 [] : Is there a tree in the image?
    c2 [c1] : Where is the tree in the image?
  planner_v4 (final type query vs oracle query; objects 1/1; relation checks 1/1; operands 0/0)
    c1 [] existence req=yes: Is there a tree in the image?
    c2 [c1] final: Where are the trees?
19687543 'What is the woman holding?' -> bag
  oracle
    o1 [] existence req=yes: Is there a woman in the image?
    o2 [o1] existence req=yes: Is the woman holding something?
    o3 [o2] final: What is the woman holding?
  planner_v3
    c1 [] : Is there a woman in the image?
    c2 [c1] : What is the woman holding?
  planner_v4 (final type query vs oracle query; objects 1/1; relation checks 0/1; operands 0/0)
    c1 [] existence req=yes: Is there a woman in the image?
    c2 [c1] final: What is the woman holding?
09212473 'The crowd is watching who?' -> woman
  oracle
    o1 [] existence req=yes: Is there a crowd in the image?
    o2 [o1] existence req=yes: Is the crowd watching someone?
    o3 [o2] final: Who is the crowd watching?
  planner_v3
    c1 [] : Is there a crowd in the image?
    c2 [c1] : Who is the crowd watching?
  planner_v4 (final type query vs oracle query; objects 1/1; relation checks 0/1; operands 0/0)
    c1 [] existence req=yes: Is there a crowd in the image?
    c2 [c1] existence req=yes: Is there a person being watched in the image?
    c3 [c1,c2] final: Who is the crowd watching?
00259214 'What is the sign hanging off of?' -> pole
  oracle
    o1 [] existence req=yes: Is there a sign in the image?
    o2 [o1] existence req=yes: Is the sign hanging off something?
    o3 [o2] final: What is the sign hanging off?
  planner_v3
    c1 [] : Is there a sign hanging off of something in the image?
    c2 [c1] : What is the sign hanging off of?
  planner_v4 (final type query vs oracle query; objects 1/1; relation checks 0/1; operands 0/0)
    c1 [] existence req=yes: Is there a sign in the image?
    c2 [c1] final: What is the sign hanging off of?
```
**b_relS**
```
14863272 'What is full of donuts?' -> tray
  oracle
    o1 [] existence req=yes: Are there donuts in the image?
    o2 [o1] existence req=yes: Is there something full of the donuts?
    o3 [o2] final: What is full of the donuts?
  planner_v3
    c1 [] : Is there a container full of donuts in the image?
    c2 [c1] : What is the container full of donuts made of?
  planner_v4 (final type query vs oracle query; objects 1/1; relation checks 0/1; operands 0/0)
    c1 [] existence req=yes: Is there a container of donuts in the image?
    c2 [c1] final: What is full of donuts?
13365742 "What's on the wire?" -> sign
  oracle
    o1 [] existence req=yes: Is there a wire in the image?
    o2 [o1] existence req=yes: Is there something on the wire?
    o3 [o2] final: What is on the wire?
  planner_v3
    c1 [] : Is there a wire in the image?
    c2 [c1] : What is on the wire?
  planner_v4 (final type query vs oracle query; objects 1/1; relation checks 0/1; operands 0/0)
    c1 [] existence req=yes: Is there a wire in the image?
    c2 [c1] final: What's on the wire?
06943437 "What's on the oven?" -> kettle
  oracle
    o1 [] existence req=yes: Is there an oven in the image?
    o2 [o1] existence req=yes: Is there something on the oven?
    o3 [o2] final: What is on the oven?
  planner_v3
    c1 [] : Is there an oven in the image?
    c2 [c1] : What is on the oven?
  planner_v4 (final type query vs oracle query; objects 1/1; relation checks 0/1; operands 0/0)
    c1 [] existence req=yes: Is there an oven in the image?
    c2 [c1] final: What's on the oven?
08900610 'What is on the ski on the left?' -> letters
  oracle
    o1 [] existence req=yes: Is there a ski on the left in the image?
    o2 [o1] existence req=yes: Is there something on the ski on the left?
    o3 [o2] final: What is on the ski on the left?
  planner_v3
    c1 [] : Is there a ski on the left in the image?
    c2 [c1] : What is on the ski on the left?
  planner_v4 (final type query vs oracle query; objects 1/1; relation checks 1/1; operands 0/0)
    c1 [] existence req=yes: Is there a ski on the left in the image?
    c2 [c1] final: What is on the ski on the left?
11464769 'What is in front of the green building?' -> fence
  oracle
    o1 [] existence req=yes: Is there a green building in the image?
    o2 [o1] existence req=yes: Is there something in front of the green building?
    o3 [o2] final: What is in front of the green building?
  planner_v3
    c1 [] : Is there a green building in the image?
    c2 [c1] : What is in front of the green building?
  planner_v4 (final type query vs oracle query; objects 1/1; relation checks 0/1; operands 0/0)
    c1 [] existence req=yes: Is there a green building in the image?
    c2 [c1] final: What is in front of the green building?
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
    c1 [] : Is there a microwave in the image?
    c2 [c1] : Is there a cabinet to the left of the microwave in the image?
    c3 [c2] : Is the counter above or below the cabinet to the left of the microwave in the image?
  planner_v4 (final type choose vs oracle choose; objects 3/3; relation checks 0/0; operands 0/0)
    c1 [] existence req=yes: Is there a microwave in the image?
    c2 [c1] existence req=yes: Is there a cabinet to the left of the microwave?
    c3 [] existence req=yes: Is there a counter in the image?
    c4 [c2,c3] final: Is the counter above or below the cabinet the microwave is to the left of?
19419643 'Are the ripe oranges to the left or to the right of the bag made of plastic?' -> right
  oracle
    o1 [] existence req=yes: Is there a plastic bag in the image?
    o2 [] existence req=yes: Are there oranges in the image?
    o3 [o1,o2] final: Are the oranges to the left or to the right of the plastic bag?
  planner_v3
    c1 [] : Is there a bag made of plastic in the image?
    c2 [] : Is there a ripe orange in the image?
    c3 [c1,c2] : Are the ripe oranges to the left or to the right of the bag made of plastic?
  planner_v4 (final type choose vs oracle choose; objects 2/2; relation checks 0/0; operands 0/0)
    c1 [] existence req=yes: Is there a ripe orange in the image?
    c2 [] existence req=yes: Is there a bag made of plastic in the image?
    c3 [c1,c2] final: Are the ripe oranges to the left or to the right of the bag made of plastic?
19151786 'Is the bench to the right or to the left of the blue vehicle?' -> left
  oracle
    o1 [] existence req=yes: Is there a blue vehicle in the image?
    o2 [] existence req=yes: Is there a bench in the image?
    o3 [o1,o2] final: Is the bench to the right or to the left of the blue vehicle?
  planner_v3
    c1 [] : Is there a bench in the image?
    c2 [] : Is there a blue vehicle in the image?
    c3 [c1,c2] : Is the bench to the right or to the left of the blue vehicle?
  planner_v4 (final type choose vs oracle choose; objects 2/2; relation checks 0/0; operands 0/0)
    c1 [] existence req=yes: Is there a bench in the image?
    c2 [] existence req=yes: Is there a blue vehicle in the image?
    c3 [c1,c2] final: Is the bench to the right or to the left of the blue vehicle?
05480434 'Is the elephant to the left or to the right of the car?' -> right
  oracle
    o1 [] existence req=yes: Is there a car in the image?
    o2 [] existence req=yes: Is there an elephant in the image?
    o3 [o1,o2] final: Is the elephant to the left or to the right of the car?
  planner_v3
    c1 [] : Is there an elephant in the image?
    c2 [] : Is there a car in the image?
    c3 [c1,c2] : Is the elephant to the left of the car?
  planner_v4 (final type choose vs oracle choose; objects 2/2; relation checks 0/0; operands 0/0)
    c1 [] existence req=yes: Is there an elephant in the image?
    c2 [] existence req=yes: Is there a car in the image?
    c3 [c1,c2] final: Is the elephant to the left or to the right of the car?
16880810 'Is the pillow to the left or to the right of the woman on the right?' -> left
  oracle
    o1 [] existence req=yes: Is there a woman on the right in the image?
    o2 [] existence req=yes: Is there a pillow in the image?
    o3 [o1,o2] final: Is the pillow to the left or to the right of the woman on the right?
  planner_v3
    c1 [] : Is there a woman on the right in the image?
    c2 [c1] : Is there a pillow to the left or to the right of the woman on the right?
  planner_v4 (final type choose vs oracle choose; objects 2/2; relation checks 0/0; operands 0/0)
    c1 [] existence req=yes: Is there a pillow in the image?
    c2 [] existence req=yes: Is there a woman in the image?
    c3 [c1,c2] final: Is the pillow to the left or to the right of the woman on the right?
```
**d_verify_rel**
```
16654472 'Is the ottoman to the left of a bed?' -> yes
  oracle
    o1 [] existence req=yes: Is there an ottoman in the image?
    o2 [o1] final: Is the ottoman to the left of a bed?
  planner_v3
    c1 [] : Is there an ottoman to the left of a bed in the image?
    c2 [c1] : Is the ottoman to the left of a bed?
  planner_v4 (final type verify vs oracle verify; objects 1/1; relation checks 0/0; operands 0/0)
    c1 [] existence req=yes: Is there an ottoman in the image?
    c2 [] existence req=yes: Is there a bed in the image?
    c3 [c1,c2] final: Is the ottoman to the left of a bed?
13877235 'Are there any books on the floor?' -> yes
  oracle
    o1 [] existence req=yes: Is there a floor in the image?
    o2 [o1] final: Are there books on the floor?
  planner_v3
    c1 [] : Is there a book on the floor in the image?
    c2 [c1] : Are there any books on the floor?
  planner_v4 (final type none vs oracle verify; objects 0/1; relation checks 0/0; operands 0/0)
    (planner failure)
00777462 'Is the bird on the rock watching a boar?' -> no
  oracle
    o1 [] existence req=yes: Is there a rock in the image?
    o2 [o1] existence req=yes: Is there a bird on the rock?
    o3 [o2] final: Is the bird on the rock watching a boar?
  planner_v3
    c1 [] : Is there a bird on a rock in the image?
    c2 [c1] : Is the bird on the rock watching a boar?
  planner_v4 (final type verify vs oracle verify; objects 2/2; relation checks 0/0; operands 0/0)
    c1 [] existence req=yes: Is there a bird in the image?
    c2 [] existence req=yes: Is there a rock in the image?
    c3 [] existence req=yes: Is there a boar in the image?
    c4 [c1,c2,c3] final: Is the bird on the rock watching a boar?
07883808 'Is the knife on a cutting board?' -> no
  oracle
    o1 [] existence req=yes: Is there a knife in the image?
    o2 [o1] final: Is the knife on a cutting board?
  planner_v3
    c1 [] : Is there a knife in the image?
    c2 [c1] : Is the knife on a cutting board?
  planner_v4 (final type verify vs oracle verify; objects 1/1; relation checks 0/0; operands 0/0)
    c1 [] existence req=yes: Is there a knife in the image?
    c2 [] existence req=yes: Is there a cutting board in the image?
    c3 [c1,c2] final: Is the knife on the cutting board?
09249815 'Is he carrying a surfboard?' -> no
  oracle
    o1 [] existence req=yes: Is there a man in the image?
    o2 [o1] final: Is the man carrying a surfboard?
  planner_v3
    c1 [] : Is there a person carrying a surfboard in the image?
    c2 [c1] : Is the person carrying a surfboard?
  planner_v4 (final type verify vs oracle verify; objects 0/1; relation checks 0/0; operands 0/0)
    c1 [] existence req=yes: Is there a person in the image?
    c2 [] existence req=yes: Is there a surfboard in the image?
    c3 [c1,c2] final: Is the person carrying the surfboard?
```
**e_logical**
```
121053010 'Do you see any doors or cars that are white?' -> yes
  oracle
    o1 [] final/or: Is there a white door in the image?
    o2 [] final/or: Is there a white car in the image?
  planner_v3
    c1 [] : Is there a door or a car in the image?
    c2 [c1] : What color is the door or car?
  planner_v4 (final type logic_or vs oracle logic_or; objects 0/0; relation checks 0/0; operands 2/2)
    c1 [] final/or: Is there a door in the image?
    c2 [] final/or: Is there a car in the image?
03848528 'Are there any hairbrushes or bottle caps?' -> no
  oracle
    o1 [] final/or: Is there a hairbrush in the image?
    o2 [] final/or: Is there a bottle cap in the image?
  planner_v3
    c1 [] : Is there a hairbrush or a bottle cap in the image?
    c2 [c1] : What color is the hairbrush or bottle cap?
  planner_v4 (final type logic_or vs oracle logic_or; objects 0/0; relation checks 0/0; operands 2/2)
    c1 [] final/or: Is there a hairbrush in the image?
    c2 [] final/or: Is there a bottle cap in the image?
0598390 'Are there either pillows or trays?' -> yes
  oracle
    o1 [] final/or: Is there a pillow in the image?
    o2 [] final/or: Is there a tray in the image?
  planner_v3
    c1 [] : Is there a pillow in the image?
    c2 [] : Is there a tray in the image?
    c3 [c1,c2] : Are there either pillows or trays?
  planner_v4 (final type logic_or vs oracle logic_or; objects 0/0; relation checks 0/0; operands 2/2)
    c1 [] final/or: Is there a pillow in the image?
    c2 [] final/or: Is there a tray in the image?
08206123 'Are there any glasses or stoves in the scene?' -> no
  oracle
    o1 [] final/or: Is there a stove in the image?
    o2 [] final/or: Are there glasses in the image?
  planner_v3
    c1 [] : Is there a glass in the image?
    c2 [] : Is there a stove in the image?
    c3 [c1,c2] : Are there any glasses or stoves in the image?
  planner_v4 (final type logic_or vs oracle logic_or; objects 0/0; relation checks 0/0; operands 2/2)
    c1 [] final/or: Is there a glass in the image?
    c2 [] final/or: Is there a stove in the image?
061043477 'Are there unpeeled bananas or apples?' -> no
  oracle
    o1 [] final/or: Is there an unpeeled banana in the image?
    o2 [] final/or: Is there an unpeeled apple in the image?
  planner_v3
    c1 [] : Is there an unpeeled banana in the image?
    c2 [] : Is there an unpeeled apple in the image?
    c3 [c1,c2] : Are there unpeeled bananas or apples in the image?
  planner_v4 (final type logic_or vs oracle logic_or; objects 0/0; relation checks 0/0; operands 2/2)
    c1 [] final/or: Is there an unpeeled banana in the image?
    c2 [] final/or: Is there an apple in the image?
```
**f_compare**
```
00768430 'Do the van and the hydrant have the same color?' -> no
  oracle
    o1 [] existence req=yes: Is there a fire hydrant in the image?
    o2 [] existence req=yes: Is there a van in the image?
    o3 [o1,o2] final: Do the fire hydrant and the van have the same color?
  planner_v3
    c1 [] : Is there a van in the image?
    c2 [] : Is there a hydrant in the image?
    c3 [c1] : What color is the van?
  planner_v4 (final type compare vs oracle compare; objects 2/2; relation checks 0/0; operands 0/0)
    c1 [] existence req=yes: Is there a van in the image?
    c2 [] existence req=yes: Is there a hydrant in the image?
    c3 [c1,c2] final: Do the van and the hydrant have the same color?
04727485 'Are the barrier and the pole made of the same material?' -> no
  oracle
    o1 [] existence req=yes: Is there a barrier in the image?
    o2 [] existence req=yes: Is there a pole in the image?
    o3 [o1,o2] final: Are the barrier and the pole made of the same material?
  planner_v3
    c1 [] : Is there a barrier in the image?
    c2 [] : Is there a pole in the image?
    c3 [c1] : What material is the barrier made of?
  planner_v4 (final type compare vs oracle compare; objects 2/2; relation checks 0/0; operands 0/0)
    c1 [] existence req=yes: Is there a barrier in the image?
    c2 [] existence req=yes: Is there a pole in the image?
    c3 [c1,c2] final: Are the barrier and the pole made of the same material?
13831520 'Is the color of the scarf different than the roof?' -> yes
  oracle
    o1 [] existence req=yes: Is there a scarf in the image?
    o2 [] existence req=yes: Is there a roof in the image?
    o3 [o1,o2] final: Do the scarf and the roof have different colors?
  planner_v3
    c1 [] : Is there a scarf in the image?
    c2 [] : Is there a roof in the image?
    c3 [c1] : What color is the scarf?
  planner_v4 (final type compare vs oracle compare; objects 2/2; relation checks 0/0; operands 0/0)
    c1 [] existence req=yes: Is there a scarf in the image?
    c2 [] existence req=yes: Is there a roof in the image?
    c3 [c1,c2] final: Is the color of the scarf different than the roof?
17401329 'What is common to the umbrella and the scarf?' -> color
  oracle
    o1 [] existence req=yes: Is there an umbrella in the image?
    o2 [] existence req=yes: Is there a scarf in the image?
    o3 [o1,o2] final: What is common to the umbrella and the scarf?
  planner_v3
    c1 [] : Is there an umbrella in the image?
    c2 [] : Is there a scarf in the image?
    c3 [c1,c2] : What material are the umbrella and the scarf made of?
  planner_v4 (final type common vs oracle common; objects 2/2; relation checks 0/0; operands 0/0)
    c1 [] existence req=yes: Is there an umbrella in the image?
    c2 [] existence req=yes: Is there a scarf in the image?
    c3 [c1,c2] final: What is common to the umbrella and the scarf?
1129358 'Is the frame made out of the same material as the bookcase?' -> yes
  oracle
    o1 [] existence req=yes: Is there a frame in the image?
    o2 [] existence req=yes: Is there a bookcase in the image?
    o3 [o1,o2] final: Are the frame and the bookcase made of the same material?
  planner_v3
    c1 [] : Is there a frame in the image?
    c2 [] : Is there a bookcase in the image?
    c3 [c1,c2] : Is the frame made out of the same material as the bookcase?
  planner_v4 (final type compare vs oracle compare; objects 2/2; relation checks 0/0; operands 0/0)
    c1 [] existence req=yes: Is there a frame in the image?
    c2 [] existence req=yes: Is there a bookcase in the image?
    c3 [c1,c2] final: Is the frame made out of the same material as the bookcase?
```
