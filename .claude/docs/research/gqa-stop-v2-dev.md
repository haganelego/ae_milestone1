# gqa_stop v2 consistency-stop: dev results (plan §17)

Date: 2026-10-09. Run: `configs/gqa_stop_v2.json`, output `outputs/gqa_stop/visapp_gqa_v2/`.
Dev only. No freeze was made and no test stage was run. Nothing is committed.

## Verdict

**No consistency variant reaches dev accuracy 0.95, so the §17 abort rule applies.**
- The best variant, θ=3 with k=2, reaches 0.91 at mean Level 14.3. That is below the Fixed-t hull (−0.012).
- Low thresholds stop early, at a median of L1–L2, but they are wrong on 34–42 of 100 questions.
- `devgrid_summary.json` records `abort: true` and `best_feasible_variant: null`.
- The §17 rule reads "abort if every setting that reaches 0.95 averages Level ≥ 18". Here no setting reaches 0.95 at all, which I treat as the stronger form of the same outcome.

## Dev grid (100 questions; v1 dev set, verified identical)

| variant | accuracy | mean L | median L | stop20 | never met | stopped & wrong | acc − hull |
|---|---|---|---|---|---|---|---|
| θ1 k1 | 0.58 | 2.28 | 1 | 0.04 | 0.03 | 42 | −0.053 |
| θ1 k2 | 0.66 | 3.97 | 2 | 0.06 | 0.06 | 34 | −0.068 |
| θ2 k1 | 0.62 | 3.64 | 1 | 0.10 | 0.10 | 38 | −0.090 |
| θ2 k2 | 0.72 | 5.09 | 2 | 0.12 | 0.12 | 28 | −0.037 |
| θ3 k1 | 0.87 | 13.02 | 20 | 0.58 | 0.58 | 13 | −0.034 |
| θ3 k2 | 0.91 | 14.31 | 20 | 0.62 | 0.62 | 9 | −0.012 |
| v1 Checklist-min (θ=0.85, frozen) | 0.98 | 18.91 | – | 0.83 | – | 2 | −0.005 |
| v1 Fixed-dev (L17) | 0.95 | 17.00 | 17 | 0 | – | 5 | −0.009 |
| Fixed-10 / Fixed-15 / Fixed-20 | 0.84 / 0.91 / 1.00 | 10 / 15 / 20 | | | | 16 / 9 / 0 | |

"acc − hull" is the accuracy minus F(mean Level), where F is the upper concave hull of the dev Fixed-t curve. The v2 reader curve equals the v1 curve, because the reader records are copied.

## Stop histograms (no variant meets 0.95; the two most accurate are shown)

Each entry reads level: questions stopped / of which wrong.
- θ3 k2: L2 21/7, L3 2/1, L4 2/0, L5 1/1, L7–L18 12/0, L20 62/0.
- θ3 k1: L1 25/10, L2 3/2, L3 1/0, L4 2/1, L6–L14 11/0, L20 58/0.

The distribution is bimodal. Questions either stop at L1 or L2, where about 40% are wrong, or are forced to L20.

## Component rates by level (dev, 100 questions)

| L | reader acc | a | b | c | a∧b∧c | P(a\|wrong) | P(b\|wrong) | P(c\|wrong) | P(s≥1∧wrong) | P(s≥2∧wrong) | P(s≥3∧wrong) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.56 | 0.32 | 0.73 | 0.72 | 0.25 | 0.30 | 0.77 | 0.73 | 0.37 | 0.32 | 0.10 |
| 3 | 0.67 | 0.25 | 0.77 | 0.76 | 0.22 | 0.18 | 0.79 | 0.73 | 0.26 | 0.24 | 0.06 |
| 5 | 0.75 | 0.25 | 0.80 | 0.79 | 0.22 | 0.28 | 0.76 | 0.72 | 0.20 | 0.18 | 0.06 |
| 10 | 0.84 | 0.29 | 0.82 | 0.83 | 0.27 | 0.38 | 0.75 | 0.75 | 0.13 | 0.12 | 0.05 |
| 15 | 0.91 | 0.30 | 0.82 | 0.80 | 0.28 | 0.33 | 0.56 | 0.44 | 0.06 | 0.04 | 0.02 |
| 17 | 0.95 | 0.32 | 0.85 | 0.84 | 0.31 | 0.40 | 0.40 | 0.40 | 0.03 | 0.02 | 0.01 |
| 20 | 1.00 | 0.30 | 0.86 | 0.82 | 0.28 | – | – | – | 0 | 0 | 0 |

All 20 levels are in `dev_components.csv`.

## Why it fails (diagnostics from `level_table_dev.csv`)

1. **b and c are "consistent but wrong".** When the reader is wrong at L1–L5, attr_blind agrees with it 63% of the time for color, 85% for material and 84% for shape. Low-resolution answers share the model's prior; this is the caveat in decc-paper §6.
2. **c is a copy of b.** The recompose answer equals the attr_blind answer in 93% of calls. Text-only recomposition adds almost no independent evidence, as decc-paper §6 point 5 predicted.
3. **The relate-0 fallback is degenerate in practice.** For the 31 questions with no referring phrase, b and c are 0.94–0.97 at every level, while the reader is right only 45% of the time at L1. "What color is the {target}?" is the reader's own question asked again.
4. **a carries no resolution signal.** It is flat at about 0.3 from L1 to L20, and P(a|wrong) ≈ P(a). Two things cause this.
   - The list fallback rarely names the target: a is 0.13–0.19 for those 31 questions.
   - Referring phrases are often ambiguous over the whole image: "the object that is in the middle" gives "sign", "the object that is to the right of the man" gives "refrigerator".
   - The planner also leaks the noun through adjectives: "the ripe object" gives "apple" at L1.
5. **Planner behaviour.** All 100 plans are valid.
   - 31 have a null phrase.
   - 21 relate-0 questions still got a phrase built from a filter ("the brown object", "the object that is made of wood").
   - 18 phrases name the target noun. Most are landmarks of the same class; a few, such as "the fence near …", hide nothing.

## Pilot (20 questions × L1/5/10/15/20)

- The initial valid rate is 1.0 for reader, identify, attr_blind, recompose and planner_v2. The repair rate is 0.
- Reader accuracy by level: L1 0.55, L5 0.70, L10 0.85, L15 0.90, L20 1.00.
- Five examples are in `pilot/pilot_examples.json`. They include the cup/other-cup, bicycle, ripe apple, umbrella and fence cases quoted above.

## Run facts

- **Screening.** `screen --split dev` used the inherited v1 split, which is read-only. It adopted the same 100 ids in the same order (94 images) with the same cell targets as `outputs/gqa_stop/visapp_gqa_v1/adoption_dev.json`; `reference_match` is true.
- **Reader reuse.** All reader records (285 from screening and repeat, then 2,000 for the 100×20 table) were copied from v1 by identical cache key and keep `copied_from` provenance. The v2 run made 0 reader calls.
- **New calls.** There were 6,105 requests: 104 planner_v2 attempts and 2,000 each for identify, attr_blind and recompose. All 6,000 level calls were valid on the first attempt. 
- **Data protection.** Nothing in the shared `data/processed/gqa_stop/visapp_gqa_v1` was written. v2 selection files are in `data/processed/gqa_stop/visapp_gqa_v2/`, and image generation is read-only for inherited splits.

## Deviations and implementation choices (not specified in §17)

- **planner_v2 validator v1→v2 during the first pilot attempt.** The first attempt was stopped by PID after 48 requests.
  - Originally I rejected any referring phrase that contained the target noun. That rejected a legitimate same-class landmark ("…to the right of the other cup").
  - Such phrases are now accepted and flagged `phrase_names_target`. `SCHEMA_VERSIONS["planner_v2"]="2"` is part of the cache key, so plans judged under the old rule were re-called.
  - The change is logged in worklog.md.
- **Recompose skip.** When identify or attr_blind is invalid, recompose is skipped and c=0. A planner failure gives s=0 at every level. Neither happened in this run.
- **"Stopped but wrong".** This counts questions whose stop condition was met and whose reader answer at that level was wrong. Forced L20 stops are excluded; they are always correct on dev by construction.
- **Tie-breaks for "best feasible".** Ties go to the minimum mean Level, then higher accuracy, then larger θ, then larger k. No variant was feasible.
- **Command guards.** `freeze` refuses for the consistency family. Test commands refuse because there is no frozen policy. `prepare` refuses because the split is inherited.

## Files

- Code:
  - `src/milestone1/gqa_stop/consistency.py`: roles, sub-questions, token match, runner.
  - `src/milestone1/gqa_stop/devgrid.py`.
  - `policy.py`: consecutive-k stop and v2 variants; the v1 k=1 check is unchanged.
  - Shared plumbing in `client.py`, `context.py`, `images.py`, `common.py`, `stages_data.py`, `stages_model.py` and `cli.py`.
- Prompts: `prompts/gqa_stop/{planner_v2,identify,attr_blind,recompose}.txt`. The reader prompt is unchanged.
- Config: `configs/gqa_stop_v2.json`.
- Tests: `tests/test_gqa_stop_v2.py` (40 tests) plus `tests/gqa_stop_helpers.py`, where the fake model now handles the v2 roles.
- Outputs in `outputs/gqa_stop/visapp_gqa_v2/`:
  - Grid and analysis: `dev_threshold_grid.csv`, `dev_stop_hist.csv`, `dev_components.csv`, `dev_comparison.csv`, `dev_fixed_curve.csv`, `devgrid_summary.json`, `level_table_dev.csv`.
  - Pilot: `pilot_summary.json`, `pilot/`.
  - Logs: `worklog.md`.
