# Decisions and deviations from the plan (ROBOVIS2027 v1.0) — orchestrator notes

Status as of 2026-10-07 (JST morning). Updated as the run proceeds; see `.claude/docs/research/robovis2027-worklog.md` for the full log.

## Decisions taken with the user
- Path P (PGSG co-generation of OBSERVED + CHECKS) was dropped after an architecture review instead of a 25-state pilot: PGSG's decoder is BLIP's BERT-base, trained only on `sub [OBJ] pred [REL] obj [OBJ]` sequences with a fixed 40-token prompt; it has no instruction-following ability and grounding comes from a position adapter over entity hidden states. Rationale recorded in `dev_r0/pilot_report.md`. Main paths: F1 (PGSG fixed + local VLM `qwen3.5` with image) and F2 (text LLM, graph only).
- Images are pre-shrunk to max side 384 px before the 20 levels (PGSG resizes inputs to 384×384; without this, levels 19–20 would be indistinguishable to the model). Predicted boxes are mapped back to GT coordinates by `coord_scale` (1.3021 for all 120 images). GT is untouched.
- Evaluation set re-selected GT-only (objects ≥ 5, relations ≥ 3, sha256 dedupe, id-sorted then seed-42 shuffle): test 100 / dev 20 from the VG-150 test split (same split PGSG was evaluated on). The old MIRU 100 were not reused because they had been filtered with EGTR-dependent stop times (29 images right-censored removed) — an observer-dependent selection. Overlap with the old set: 4 images.
- A GT-referenced RelSGScore (MIRU's label-set score, 0.5·node recall + 0.5·triplet recall, no boxes; capped to top-20 entities / top-50 relations) was added as a SECONDARY q_t at the user's request. Primary stays official SGDet R@50.

## Implementation deviations (recorded, not hidden)
- PGSG sampling seed is per (batch of 4 consecutive images, level), not per image: batch 4 is 2.4× faster than batch 1 and the batches are fixed, so results are fully reproducible. Seeds are in `manifest.json`.
- The effective repetition_penalty is 1.1 (hard-coded in PGSG's `med.py` sampling branch; the task's 1.5 is ignored) and `min_length` is unused. The eval yaml sets seg_len 32 (36 segments) and top-3 predicates. All resolved values are stored in each graph's `generation` block.
- Schema retry: with temperature 0 and a fixed seed the "same input" retry was a byte-identical no-op; the retry now uses seed+1 and temperature 0.3 (first attempt unchanged). Recorded per record (`seeds`, `temperatures`, `seed_used`).
- Check-generator rendering caps the graph (v1: 60 entities / 50 relations; v2: duplicate same-label boxes with IoU > 0.5 merged, 30 entities / 20 relations) because PGSG emits ~120 entities and ~340 relations per state regardless of resolution. Anchors are validated against the rendered subset.
- Exploratory stop-rule variant `benefit_gated` (u = resolution_benefit if gate ∧ basis ≥ 1) was added for diagnosis only; it is labelled "plan deviation, needs sign-off" in all outputs and is not a main condition.
- F1v (judge also receives the image) and prompt v2 (scene-focused hypotheses, independent definitions of basis/benefit with a both-2 few-shot, <200-char hint) are separately named variants evaluated on dev only; prompts are hashed per variant.

## Findings that constrain the claims
- PGSG never returns an empty graph (0/2400 states), so the plan's "empty graph → A_t = 2" clause never fires.
- Per-image q_t from a single sampling run is noisy (qmax is upward-biased; one recovered triplet ≈ 0.1–0.3 recall on small-GT images). The relative-stable oracle is defined for 37/100 test images under R@50 (28 zero_recovery) and 36/100 under relsg_gt (0 zero_recovery but 19/36 defined only at t=20 because label recall is still rising at full resolution). Oracle MAE is therefore auxiliary; the main evidence is population-level q(stop) vs the fixed-stop curve with the equal-observation mix and paired bootstrap.
- With the plan's judge contract and `qwen3.5:4b`, the §6 rule is degenerate: items reaching basis = 2 AND resolution_benefit = 2 are ≈ 0 % at every level, because the judge scores basis and benefit anti-correlated; C_t carries no resolution-sufficiency signal (Spearman with remaining gain ≈ 0.2 for F1, ≈ 0 for F2). Changing the aggregation rule does not recover a signal; the content of checks/judgments must change (variants above).

## User-approved deviation (2026-10-07, judge v3)
- `resolved` is relaxed to "about 90 % of what should be visible is visible" (early resolution accepted). The judge is told only how much sharper the next observation will be (a constant ≈17 % per side, since the schedule is geometric); the level index, absolute scale and remaining-step count are never passed. This is the only departure from plan §5's input restrictions and was approved by the user because the stop rule targets approach-driven resolution gain.

## User decision (2026-10-07 evening): grounded visibility instead of resolution reading
- A judge that estimates image resolution directly was rejected as task-specific. Judge v4 instead asks, per claimed object (top-8 entities of G, boxes drawn on the image), whether the object is clearly visible and identifiable inside its box; unclear objects become observed issues with basis 2 / benefit 2 deterministically. Hypotheses are unchanged (reused). Named variant `F1g_v4_G__9b`.
