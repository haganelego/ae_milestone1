# Probe: current planner_v3 on GQA "query x rel" questions (2026-10-13)

Text-only dev probe. No images, no test-method inference, no prompt/code edits, no commits.

## Setup

- Prompt: `prompts/gqa_stop/planner_v3.txt` (unchanged), role `planner_v3`, built by
  `sampled.planner_v3_spec` and run through `RequestEngine` with `configs/gqa_stop_v22.json`
  (effective digest `qwen3.5:9b@6488c96f...`), format json, think false, A13 block
  (T 0, seed 20261008, top_k 1, top_p 1.0, penalties 0, repeat 1.0), num_predict 1024,
  one schema-repair attempt, validator `validate_plan_v3` (2-3 items).
- Scratch root `outputs/gqa_stop/probe_planner_rel/` (`sample.jsonl`, `plans.jsonl`, `plans_v3/`);
  `visapp_gqa_v22/` untouched. qwen3.5:9b loaded alone, fully on GPU. 41 calls, 34 s.
- Sample (seed 20261008, 1 question per image): query x rel, final `query: name`; 10 relS, 10 relO,
  8 categoryRelS, 4 categoryRelO (1 relate) and 8 two-relate (2-hop) from the same 4 types.
- Anchor = GQA `select` object; target = related object (the answer); 2-hop: anchor, intermediate, target.

## Counts (n = 40)

| Check | Count |
|---|---|
| Valid plans | 40 (39 first attempt, 1 after repair) |
| 2 items / 3 items | 32 / 8 |
| Item 1 = anchor existence ("Is there a knife in the image?") | 17 |
| Item 1 = target+relation existence ("Is there a person wearing a shirt...") | 13 |
| Item 1 = intermediate+relation existence (2-hop) | 7 |
| Item 1 defective (new object / garbled / bare target category) | 3 |
| Plan has a target-existence item (target + relation) | 20 |
| ...by stratum: relS / relO / catRelS / catRelO / 2-hop | 9/10, 0/10, 8/8, 1/4, 2/8 |
| Plan checks anchor (or hop 1) AND target separately | 7 (5 one-hop, 2 two-hop) |
| Last item asks the name with full referring expression | 37 |
| ...of which verbatim copy of the original question | 28 |
| Last item hijacked by the attribute-word rule (nonsense) | 3 |
| New object introduced | 1 |
| Pronoun / item reference / "if yes" | 0 |
| Counting item | 0 |
| depends_on not a chain where one is expected (parallel, still valid) | 2 |
| 2-hop: both hops checked as separate items | 2/8 (hop 1 in 8/8, target in 2/8) |
| 2-hop: cap truncation (first reply had 4 items, rejected) | 1/8 |

- `last_item_has_attribute` is True in exactly the 3 broken plans: inverted for rel questions.
- `item_not_standalone` flagged 2 items of one plan: regex false positive on "item of furniture".
- "Target" is read inconsistently: relS ("What is on the table?") collapses to one target+relation
  existence item; relO ("What is the knife cutting?") checks the anchor only, never the target.

## Example plans

Good, full chain (categoryRelS, #20). "What vehicle is to the right of the dark car?" -> bus
```
c1 []     Is there a dark car in the image?
c2 [c1]   Is there a vehicle to the right of the dark car in the image?
c3 [c2]   What vehicle is to the right of the dark car?
```
Good, 2-hop with both hops (#33). "What type of vehicle is to the left of the man that is wearing a vest?" -> car
```
c1 []     Is there a man wearing a vest in the image?
c2 [c1]   Is there a vehicle to the left of the man wearing a vest in the image?
c3 [c2]   What type of vehicle is to the left of the man wearing a vest?
```
Typical relS, collapsed (#1). "Who is wearing the shirt?" -> player
```
c1 []     Is there a person wearing a shirt in the image?
c2 [c1]   Who is the person wearing the shirt?
```
Typical relO, anchor only, no target check (#13). "What is the knife cutting?" -> bacon
```
c1 []     Is there a knife in the image?
c2 [c1]   What is the knife cutting?
```
Bad, attribute rule + new object (#11). "What is the wine in?" -> glass
```
c1 []     Is there a wine bottle in the image?
c2 [c1]   What material is the wine bottle made of?
```
Bad, attribute rule + garbled item 1 (#19). "What's the crowd before?" -> fence
```
c1 []     Is there a crowd before in the image?
c2 [c1]   What is the crowd before made of?
```
Bad, attribute rule on a good chain (#21). "What is the drink to the right of the spoons on the left side?" -> beer
```
c1 []     Is there a spoon on the left side in the image?
c2 [c1]   Is there a drink to the right of the spoons on the left side in the image?
c3 [c2]   What is the drink to the right of the spoons on the left side made of?
```
2-hop cut by the cap (#35, -> stove). First reply had 4 items (faucet, plates, appliance, name); repair dropped the target check:
```
c1 []     Is there a faucet in the image?
c2 [c1]   Are there plates to the left of the faucet in the image?
c3 [c2]   What is the appliance to the left of the plates that are to the left of the faucet called?
```

## Verdict

Partly yes, structurally no. The prompt yields valid, standalone, pronoun-free plans. In
37/40 the last item asks the right name question, so the expected attribute-rule failure is
real but rare (3/40, 7.5%). The checklist part is weak, though. 28/40 last items are a
verbatim copy of the question, so most plans are "Direct + one existence check". Only half
of the plans check that the related object exists, and none of the 10 relO plans do. Only
2/8 two-hop plans check both hops, and the 3-item cap blocks the natural 4-item chain. As
is, rel plans would test "existence + Direct" rather than a decomposition, and
`last_item_has_attribute` would mislabel every good plan.

## Minimal changes that would be needed (not implemented)

1. Make the last-item rule conditional. Attribute word only when an attribute is asked.
   When the question asks what/who/which object stands in a relation, the last item asks
   for the name or kind of that object, repeating the full referring expression.
2. Name the roles. Item 1 is the existence of the object named in the question, with its
   own filters. One more item asks "Is there a <category or 'something'> <relation> <that
   object> in the image?". This fixes the relS/relO asymmetry.
3. One existence item per hop for 2-hop questions. That needs max_items 4 for rel
   (config `checklist.max_items` plus the "2 or 3" text), or a rule that merges the
   intermediate and target checks.
4. Add one rel example (preferably an object-side one such as "What is the knife cutting?")
   next to the cup example.
5. Code side: make `last_item_has_attribute` type-aware; stop the regex flagging "item of furniture".

Caveat: rel answers are open names (guy/man, people); exact match and first-token P_ans get noisier.
