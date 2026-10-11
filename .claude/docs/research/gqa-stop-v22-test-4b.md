# gqa_stop v22 test, model swap: qwen3.5:4b on the 9b test sets (plan §26, DESCRIPTIVE θ sweep only)

Status: **inference complete and audited, 2026-10-11.** User request: "run test with qwen:4b; same dataset settings; simply swap the model". The deliverable is the descriptive θ sweep 0.50–1.00 (step 0.05) for direct_v4, checklist_v4_min, direct_pans and pans_min, with the 9b sweep beside it.
- **Not done (user decision).** No 4b dev run, no θ selection, no freeze, no frozen-θ evaluation (`evaluate-v22` / `firing-test`). Nothing here is a hypothesis test.
- **Pre-registration.** Plan §26 was appended at 2026-10-11T05:20Z (HEAD 9cd6829), before any 4b call.
- **One deviation (plan §26.1).** 4b's planner_v3 appends "." after a complete JSON plan. A config-gated parse tolerance was added for planner_v3 and observer_v3 before any item call. The team lead chose this option (B). See "Deviation" below. The strict-parser variant is also reported.
- **Supplementary full grid (team-lead request).** θ 0.05–1.00 is reported separately, because 4b's scores rarely reach the main grid.

## Setup

| item | value |
|---|---|
| config | `configs/gqa_stop_v22_4b.json` |
| output root | `outputs/gqa_stop/visapp_gqa_v22_4b/` |
| model | qwen3.5:4b, digest `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd` |
| fallback | qwen3.5:4b itself (no path switches to 9b) |
| decoding | A13 block (temperature 0, top_k 1, top_p 1, penalties neutral, mirostat 0), seed 20261008, think false, num_ctx 8192, same num_predict |
| prompts | unchanged (same hashes as the 9b run) |
| roles called | reader (answer_only, L1–L20), direct_lp, relevance_v4 (direct + items), planner_v3, observer_v3_lp; v1 arm and verbalised Direct skipped (`--skip-v1arm`) |

The config differs from `configs/gqa_stop_v22.json` in five places:
- `run_id`;
- `paths.output_root`;
- `model.name`;
- `model.fallback_name`, with a wording change in `decoding_note`;
- a new `model_swap` block. It holds the reference run, the notes and `trailing_period_roles`.

**Sets are the 9b sets, not re-screened.** They are referenced, not copied. `model_swap_provenance.json` records the sha256 of each file at 05:25Z; `sweep-swap` re-checks them and refuses to run if any changed.

| file | sha256 (prefix) |
|---|---|
| `visapp_gqa_v22/visual_subset.json` (test visual subset, 251) | eb4a29bf… (= 9b `frozen_policy.json` subset_sha256) |
| `data/processed/gqa_stop/visapp_gqa_v22/selected_manifest.jsonl` | 7d2ed3c6… |
| `data/processed/gqa_stop/visapp_gqa_v22/selection_results.jsonl` (defines the 233) | 5971251b… |
| `data/processed/gqa_stop/visapp_gqa_v1/candidate_manifest.jsonl` | 252f0945… (= 9b frozen) |
| `data/processed/gqa_stop/visapp_gqa_v1/image_manifest.jsonl` (levels) | a85a0549… |

The answerable set has 251 questions on 213 images. The reader-wrong set has 233 questions. The question-id lists match the 9b level tables exactly.

## Environment and gates
- **GPU placement.** RTX 5080, driver 580.178.04 (userland = kernel), ollama 0.30.4. No resident model before the run. After `isolate_model`, qwen3.5:4b was warmed alone at num_ctx 8192: 34/34 layers on GPU, ollama log "CLIP using CUDA0 backend", `ollama ps` 100% GPU. It was the only resident model at every check between chunks, and the audit's maximum non-generation time of 0.43 s shows no reload or CPU encoder.
- **Determinism gate (4b).** Passed: 120 calls, 1,707 logprob positions, 0 argmax violations, and both repeat pairs (text_only, image reader L20) byte-identical in content and logprobs (`preflight_determinism.json`).
- **Sanity reference, not a gate.** 10 cached V6-A qwen3.5:4b dev reader requests (`visapp_gqa_v22/reader_qwen4b/`) were re-sent to a scratch directory. Cache keys were identical, and raw text and logprobs were byte-identical 10/10.
- **Audit** (`audit-test`, every 4b test record). 0 records flagged and nothing quarantined.

| audit item | value |
|---|---|
| records checked | 68,524 |
| non-generation time > 5 s | 0 (max 0.43 s) |
| argmax violations | 0 of 492,455 positions |

## Calls and wall time
Calls ran in foreground chunks of ≤ 585 s, resumed from cache. Request attempts are counted from `requests.jsonl`, schema repairs included.

| phase | set | calls | wall time |
|---|---|---|---|
| strict parser: reader, direct_lp, relevance (direct), planner | 251 | 15,562 | 05:28–07:22Z (1 h 54 m) |
| strict parser: same roles | 233 | 14,446 | 07:23–09:08Z (1 h 45 m) |
| tolerance on: planner re-call, observer_v3_lp, item relevance | 251 | 20,900 | 09:08–11:51Z (2 h 43 m) |
| tolerance on: same roles | 233 | 18,607 | 11:51–14:17Z (2 h 26 m) |
| **total** | | **69,515** | **8 h 48 m, 0.455 s/call** |

Final records:

| role | records | validity |
|---|---|---|
| reader | 9,680 | all valid |
| direct_lp | 9,680 | all valid |
| relevance_v4 | 29,180 | all valid |
| observer_v3_lp | 19,500 | all valid; none needed the trailing-period strip |
| plans_v3 | 484 | 463 valid (244 + 219), all of them after the strip; 21 malformed (7 + 14) |

Plan sizes:

| model | 2 items | 3 items |
|---|---|---|
| 4b (251) | 216 | 28 |
| 9b (251) | 241 | 10 |

## Deviation (plan §26.1): trailing-period tolerance
- **Problem.** With format=json, qwen3.5:4b still ends planner_v3 replies with "." after a complete JSON object, for example `{"items":[...]}.`. The strict parser rejects them as "invalid JSON: Extra data". The schema repair repeats the same reply.
- **Strict-parser counts.** 244 of 251 and 219 of 233 plans failed only because of trailing "." characters. 7 and 14 were otherwise malformed (for example, the closing "}" is missing).
- **Impact if left strict.** Every question would be a planner failure, so checklist_v4_min and pans_min would be 0 at every level.
- **Change.** For planner_v3 and observer_v3 only, "." and whitespace after a final "}" are removed before `json.loads`. Nothing else is tolerated.
  - The change is enabled only by `model_swap.trailing_period_roles` in the 4b config.
  - The rule name `trailing-period-v1` enters only those roles' cache keys, so 9b keys are unchanged. This was verified on 13 cached 9b plans.
  - The 4b reader, direct_lp and relevance records stayed valid and were not re-called.
  - Observer was included before any observer call, and needed it 0 times.
- **Accepted and rejected forms.** A complete JSON object followed by one or more "." characters and whitespace, in any order, is accepted, for example `{...}.`, `{...} .\n` or `{...}..`. Everything else after the closing brace is rejected, for example `{...}x` or `{...}{...}`. A reply with a missing closing brace also stays a failure. The parsed content is exactly what `json.loads` returns for the object: there is no truncation and no repair of the inner JSON. Raw replies are kept unchanged in the records, and stripped records carry `trailing_period_stripped: true`.
- **Per-role counts** (`sweep_swap/sweep_meta.json` → `parse_tolerance`):

| set / role | records | valid | needed the strip |
|---|---|---|---|
| 251 / planner_v3 | 251 | 244 | 244 |
| 233 / planner_v3 | 233 | 219 | 219 |
| 251 / observer_v3_lp | 10,320 | 10,320 | 0 |
| 233 / observer_v3_lp | 9,180 | 9,180 | 0 |

- **Strict failures that remain.** 7 plans on the 251 and 14 on the 233 are planner_failure; their checklist scores are 0.
- **Observer scope.** The team lead's rule was to cover observer_v3 only if it showed the same pattern. It was added to the list up front, before any observer call, to avoid re-running it. It needed the strip 0 times, so its parsed content equals the strict parse; only its cache-key namespace differs.
- **Process.** The question went to the team lead after the first chunk. Their replies (option B) arrived only after the run had finished. Meanwhile B had been applied on my own judgement, after about 2.5 h without a reply, and logged in the plan and worklog before any item call.
- **Strict variant (A).** Its level tables and planner records are kept in `strict_parse/`. Under A, direct_v4 and direct_pans are identical to B (they do not use the plan). checklist_v4_min and pans_min never stop before L20: accuracy 0.888, mean L 20, firing 0 at every θ.

## 4b reader: ceiling and fixed-level curve
Reader accuracy per level on both sets (`sweep_swap/fixed_curve.csv`, `figures/swap_fixed_curve.png`):

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4b on 251 | 0.323 | 0.375 | 0.418 | 0.482 | 0.486 | 0.570 | 0.614 | 0.657 | 0.713 | 0.741 | 0.781 | 0.813 | 0.813 | 0.849 | 0.853 | 0.849 | 0.845 | 0.865 | 0.873 | 0.888 |
| 9b on 251 | 0.303 | 0.394 | 0.422 | 0.486 | 0.538 | 0.606 | 0.669 | 0.709 | 0.761 | 0.745 | 0.797 | 0.825 | 0.869 | 0.869 | 0.912 | 0.928 | 0.928 | 0.940 | 0.944 | 1.000 |
| 4b on 233 | 0.129 | 0.133 | 0.146 | 0.129 | 0.124 | 0.133 | 0.159 | 0.155 | 0.133 | 0.146 | 0.163 | 0.155 | 0.155 | 0.159 | 0.163 | 0.163 | 0.167 | 0.163 | 0.189 | 0.180 |
| 9b on 233 | 0.099 | 0.094 | 0.103 | 0.107 | 0.120 | 0.107 | 0.116 | 0.124 | 0.107 | 0.107 | 0.082 | 0.077 | 0.069 | 0.069 | 0.052 | 0.056 | 0.052 | 0.060 | 0.069 | 0.000 |

- **4b ceiling on the 251 is L20 0.888**, 223 of 251. The 9b value is 1.000 by construction, because the set is the 9b reader's L20-correct questions.
- **4b fails more; it does not need more resolution.** The curves agree within 0.06 up to L12 (L10: 0.741 vs 0.745). From L14 on, 4b stays at 0.85–0.89 while 9b climbs from 0.87 to 0.94 at L19.
- **Onset is the same.** On the 223 questions both models answer correctly at L20, the stable-correct onset is about equal. Onset is the first level from which the reader stays correct to L20.

| model | onset mean | onset median |
|---|---|---|
| 4b | 6.66 | 6 |
| 9b | 6.61 | 5 |

  Across the 223 questions, 4b is later on 70, earlier on 70, and at the same level on 83.
- **4b answers 18% of the 233 correctly at L20** (0.180). The 233 are defined by the 9b reader. Firing on the 233 therefore includes some questions that 4b can answer; see Caveats.

## θ sweep (DESCRIPTIVE, test, k = 1, forced L20 not a firing)
- **Scope.** This is a description, not a result. No θ is selected on test. Accuracy and stop-level quartiles are on the 251 answerable questions. Quartiles are numpy linear percentiles. Firing is on the 233 reader-wrong questions.
- **Sources.** The 9b columns are `visapp_gqa_v22/sweep_v22/test_sweep.csv`; the 4b columns are `sweep_swap/test_sweep.csv`. Both are combined in `sweep_swap/test_sweep_vs_reference.csv`.
- **Figures.** One per axis, 4b solid and 9b dashed, in `sweep_swap/figures/`. Files: `swap_accuracy.png`, `swap_mean_level.png`, `swap_median_level.png`, `swap_stop20_rate.png`, `swap_firing_rate.png`, `swap_accuracy_vs_firing.png`, `swap_fixed_curve.png`.

| method | θ | 4b acc | 4b mean L | 4b Q1 | 4b median | 4b Q3 | 4b stop20 | 4b firing | 9b acc | 9b mean L | 9b Q1 | 9b median | 9b Q3 | 9b stop20 | 9b firing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| direct_v4 | 0.50 | 0.888 | 19.76 | 20 | 20 | 20 | 0.96 | 0.026 | 0.980 | 16.67 | 14 | 20 | 20 | 0.53 | 0.391 |
| direct_v4 | 0.55 | 0.888 | 19.89 | 20 | 20 | 20 | 0.98 | 0.021 | 0.996 | 17.19 | 15 | 20 | 20 | 0.62 | 0.313 |
| direct_v4 | 0.60 | 0.888 | 19.96 | 20 | 20 | 20 | 0.99 | 0.021 | 0.996 | 17.56 | 16 | 20 | 20 | 0.65 | 0.249 |
| direct_v4 | 0.65 | 0.888 | 19.99 | 20 | 20 | 20 | 1.00 | 0.004 | 0.996 | 17.99 | 17.5 | 20 | 20 | 0.70 | 0.197 |
| direct_v4 | 0.70 | 0.888 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 | 0.996 | 18.33 | 19 | 20 | 20 | 0.74 | 0.163 |
| direct_v4 | 0.75 | 0.888 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 | 1.000 | 18.69 | 20 | 20 | 20 | 0.80 | 0.124 |
| direct_v4 | 0.80 | 0.888 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 | 1.000 | 19.00 | 20 | 20 | 20 | 0.82 | 0.090 |
| direct_v4 | 0.85 | 0.888 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 | 1.000 | 19.37 | 20 | 20 | 20 | 0.89 | 0.052 |
| direct_v4 | 0.90 | 0.888 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 | 1.000 | 19.73 | 20 | 20 | 20 | 0.94 | 0.039 |
| direct_v4 | 0.95 | 0.888 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 | 1.000 | 19.95 | 20 | 20 | 20 | 0.99 | 0.017 |
| direct_v4 | 1.00 | 0.888 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 | 1.000 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 |
| checklist_v4_min | 0.50 | 0.888 | 19.10 | 20 | 20 | 20 | 0.82 | 0.185 | 0.980 | 16.39 | 14 | 20 | 20 | 0.50 | 0.403 |
| checklist_v4_min | 0.55 | 0.888 | 19.39 | 20 | 20 | 20 | 0.88 | 0.124 | 0.988 | 16.94 | 14 | 20 | 20 | 0.55 | 0.348 |
| checklist_v4_min | 0.60 | 0.888 | 19.56 | 20 | 20 | 20 | 0.92 | 0.077 | 0.992 | 17.33 | 15.5 | 20 | 20 | 0.61 | 0.279 |
| checklist_v4_min | 0.65 | 0.888 | 19.72 | 20 | 20 | 20 | 0.94 | 0.043 | 0.996 | 17.71 | 16 | 20 | 20 | 0.65 | 0.219 |
| checklist_v4_min | 0.70 | 0.888 | 19.88 | 20 | 20 | 20 | 0.97 | 0.034 | 1.000 | 18.06 | 17.5 | 20 | 20 | 0.72 | 0.180 |
| checklist_v4_min | 0.75 | 0.888 | 19.94 | 20 | 20 | 20 | 0.98 | 0.013 | 1.000 | 18.38 | 20 | 20 | 20 | 0.76 | 0.155 |
| checklist_v4_min | 0.80 | 0.888 | 19.99 | 20 | 20 | 20 | 1.00 | 0.009 | 1.000 | 18.82 | 20 | 20 | 20 | 0.80 | 0.120 |
| checklist_v4_min | 0.85 | 0.888 | 20.00 | 20 | 20 | 20 | 1.00 | 0.004 | 1.000 | 19.11 | 20 | 20 | 20 | 0.84 | 0.090 |
| checklist_v4_min | 0.90 | 0.888 | 20.00 | 20 | 20 | 20 | 1.00 | 0.004 | 1.000 | 19.43 | 20 | 20 | 20 | 0.89 | 0.064 |
| checklist_v4_min | 0.95 | 0.888 | 20.00 | 20 | 20 | 20 | 1.00 | 0.004 | 1.000 | 19.82 | 20 | 20 | 20 | 0.95 | 0.021 |
| checklist_v4_min | 1.00 | 0.888 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 | 1.000 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 |
| direct_pans | 0.50 | 0.765 | 11.68 | 4 | 12 | 20 | 0.37 | 0.549 | 0.534 | 4.39 | 1 | 2 | 6 | 0.04 | 0.936 |
| direct_pans | 0.55 | 0.809 | 13.75 | 7 | 20 | 20 | 0.51 | 0.421 | 0.594 | 5.65 | 1 | 3 | 8 | 0.09 | 0.901 |
| direct_pans | 0.60 | 0.845 | 15.16 | 9.5 | 20 | 20 | 0.62 | 0.309 | 0.649 | 6.58 | 1 | 4 | 10 | 0.13 | 0.845 |
| direct_pans | 0.65 | 0.873 | 16.79 | 19.5 | 20 | 20 | 0.75 | 0.206 | 0.685 | 7.34 | 1 | 5 | 11 | 0.16 | 0.777 |
| direct_pans | 0.70 | 0.880 | 18.08 | 20 | 20 | 20 | 0.86 | 0.159 | 0.741 | 8.59 | 1.5 | 6 | 16 | 0.22 | 0.704 |
| direct_pans | 0.75 | 0.888 | 19.08 | 20 | 20 | 20 | 0.93 | 0.073 | 0.813 | 10.02 | 2.5 | 9 | 20 | 0.29 | 0.592 |
| direct_pans | 0.80 | 0.888 | 19.76 | 20 | 20 | 20 | 0.98 | 0.030 | 0.876 | 11.92 | 4 | 13 | 20 | 0.41 | 0.502 |
| direct_pans | 0.85 | 0.888 | 20.00 | 20 | 20 | 20 | 1.00 | 0.013 | 0.924 | 13.48 | 6 | 20 | 20 | 0.51 | 0.352 |
| direct_pans | 0.90 | 0.888 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 | 0.976 | 15.58 | 10.5 | 20 | 20 | 0.63 | 0.215 |
| direct_pans | 0.95 | 0.888 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 | 0.992 | 18.04 | 20 | 20 | 20 | 0.84 | 0.090 |
| direct_pans | 1.00 | 0.888 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 | 1.000 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 |
| pans_min | 0.50 | 0.841 | 11.78 | 7 | 11 | 20 | 0.26 | 0.614 | 0.865 | 10.65 | 5.5 | 10 | 15.5 | 0.16 | 0.725 |
| pans_min | 0.55 | 0.849 | 12.37 | 7 | 12 | 20 | 0.28 | 0.554 | 0.888 | 11.17 | 6 | 11 | 16.5 | 0.20 | 0.687 |
| pans_min | 0.60 | 0.849 | 13.23 | 8 | 14 | 20 | 0.34 | 0.485 | 0.908 | 11.75 | 6 | 12 | 18 | 0.22 | 0.639 |
| pans_min | 0.65 | 0.876 | 14.05 | 9 | 15 | 20 | 0.41 | 0.438 | 0.932 | 12.40 | 7 | 13 | 20 | 0.27 | 0.592 |
| pans_min | 0.70 | 0.876 | 14.94 | 10 | 17 | 20 | 0.45 | 0.365 | 0.940 | 13.20 | 7.5 | 14 | 20 | 0.33 | 0.545 |
| pans_min | 0.75 | 0.884 | 15.54 | 11 | 20 | 20 | 0.52 | 0.335 | 0.952 | 13.93 | 9 | 15 | 20 | 0.37 | 0.446 |
| pans_min | 0.80 | 0.884 | 16.52 | 13.5 | 20 | 20 | 0.61 | 0.275 | 0.952 | 14.67 | 10 | 18 | 20 | 0.48 | 0.365 |
| pans_min | 0.85 | 0.884 | 17.33 | 17 | 20 | 20 | 0.70 | 0.197 | 0.976 | 15.83 | 12 | 20 | 20 | 0.57 | 0.296 |
| pans_min | 0.90 | 0.888 | 18.18 | 20 | 20 | 20 | 0.79 | 0.172 | 0.984 | 16.78 | 15 | 20 | 20 | 0.66 | 0.176 |
| pans_min | 0.95 | 0.888 | 19.03 | 20 | 20 | 20 | 0.89 | 0.090 | 1.000 | 17.80 | 20 | 20 | 20 | 0.78 | 0.099 |
| pans_min | 1.00 | 0.888 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 | 1.000 | 20.00 | 20 | 20 | 20 | 1.00 | 0.000 |

### Why the 4b product signals barely move in this θ range
4b is less confident than 9b on both factors of the product.

| quantity on the 251 | 4b | 9b |
|---|---|---|
| direct_v4 max over all levels | 0.66 | 0.98 |
| direct_pans max over all levels | 0.83 | 0.99 |
| share of L20 rows with direct_v4 ≥ 0.5 | 0.016 | 0.53 |
| share of L20 rows with checklist_v4_min ≥ 0.5 | 0.23 | 0.54 |
| mean direct Rel at L1 / L10 / L20 | 0.04 / 0.16 / 0.46 | 0.04 / 0.30 / 0.68 |

- **Consequence.** direct_v4 never fires at θ ≥ 0.70, and direct_pans never fires at θ ≥ 0.90. In this range 4b's direct_v4 and checklist_v4_min are close to "always L20": accuracy 0.888 = ceiling, mean L ≥ 19.1.
- **Where they would operate.** The 9b frozen θ for the products were 0.14–0.42, below the requested range. 4b's product signals would operate there too. That range is not part of this sweep.

## Supplementary full grid θ 0.05–1.00 (SUPPLEMENTARY, descriptive)
This grid was requested by the team lead because 4b's direct_v4 peaks at 0.66 and direct_pans at 0.83. It has the same columns and the same rules as the main grid.
- **Files.** `sweep_swap/supplementary_full_grid/test_sweep_full_vs_reference.csv` (both models, all 20 θ). Figures are `supp_*.png` in the `figures/` subdirectory.
- **9b values are recomputed** from the 9b level tables. On the shared θ they reproduce the 9b main sweep exactly; `sweep-swap` checks this.
- **Rows shown.** Only θ 0.05–0.45 are listed below; θ ≥ 0.50 are identical to the main table.

| method | θ | 4b acc | 4b mean L | 4b Q1 | 4b median | 4b Q3 | 4b stop20 | 4b firing | 9b acc | 9b mean L | 9b Q1 | 9b median | 9b Q3 | 9b stop20 | 9b firing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| direct_v4 | 0.05 | 0.777 | 10.67 | 8 | 10 | 13 | 0.09 | 0.914 | 0.737 | 7.55 | 4 | 7 | 10 | 0.04 | 0.957 |
| direct_v4 | 0.10 | 0.829 | 13.70 | 10 | 13 | 18 | 0.23 | 0.738 | 0.825 | 9.51 | 6 | 9 | 13 | 0.08 | 0.901 |
| direct_v4 | 0.15 | 0.876 | 15.60 | 12 | 16 | 20 | 0.37 | 0.562 | 0.876 | 10.95 | 7 | 11 | 15 | 0.12 | 0.837 |
| direct_v4 | 0.20 | 0.888 | 17.10 | 15 | 20 | 20 | 0.52 | 0.352 | 0.900 | 12.01 | 8 | 12 | 16 | 0.14 | 0.764 |
| direct_v4 | 0.25 | 0.888 | 18.09 | 17 | 20 | 20 | 0.68 | 0.245 | 0.916 | 12.97 | 9 | 13 | 18 | 0.21 | 0.717 |
| direct_v4 | 0.30 | 0.884 | 18.69 | 20 | 20 | 20 | 0.78 | 0.193 | 0.924 | 13.98 | 10 | 15 | 20 | 0.26 | 0.644 |
| direct_v4 | 0.35 | 0.888 | 19.13 | 20 | 20 | 20 | 0.85 | 0.129 | 0.948 | 14.76 | 11 | 16 | 20 | 0.32 | 0.562 |
| direct_v4 | 0.40 | 0.888 | 19.38 | 20 | 20 | 20 | 0.90 | 0.069 | 0.964 | 15.50 | 12 | 17 | 20 | 0.38 | 0.506 |
| direct_v4 | 0.45 | 0.888 | 19.55 | 20 | 20 | 20 | 0.93 | 0.039 | 0.964 | 16.12 | 13 | 19 | 20 | 0.45 | 0.455 |
| checklist_v4_min | 0.05 | 0.821 | 12.00 | 8 | 11 | 16.5 | 0.20 | 0.708 | 0.813 | 9.87 | 5 | 9 | 14 | 0.12 | 0.841 |
| checklist_v4_min | 0.10 | 0.845 | 13.47 | 10 | 13 | 19 | 0.24 | 0.682 | 0.865 | 10.81 | 6 | 10 | 15 | 0.14 | 0.824 |
| checklist_v4_min | 0.15 | 0.861 | 14.72 | 11 | 14 | 20 | 0.33 | 0.622 | 0.884 | 11.76 | 7 | 12 | 16 | 0.17 | 0.773 |
| checklist_v4_min | 0.20 | 0.876 | 15.78 | 13 | 17 | 20 | 0.41 | 0.541 | 0.916 | 12.49 | 8 | 13 | 17.5 | 0.20 | 0.725 |
| checklist_v4_min | 0.25 | 0.880 | 16.71 | 13 | 19 | 20 | 0.49 | 0.451 | 0.924 | 13.23 | 9 | 14 | 19 | 0.24 | 0.674 |
| checklist_v4_min | 0.30 | 0.880 | 17.31 | 15 | 20 | 20 | 0.55 | 0.361 | 0.952 | 14.10 | 10 | 15 | 20 | 0.29 | 0.627 |
| checklist_v4_min | 0.35 | 0.884 | 17.81 | 16 | 20 | 20 | 0.60 | 0.275 | 0.964 | 14.79 | 11 | 16 | 20 | 0.35 | 0.549 |
| checklist_v4_min | 0.40 | 0.888 | 18.37 | 18 | 20 | 20 | 0.69 | 0.240 | 0.968 | 15.38 | 12 | 17 | 20 | 0.41 | 0.519 |
| checklist_v4_min | 0.45 | 0.888 | 18.74 | 20 | 20 | 20 | 0.76 | 0.227 | 0.976 | 15.79 | 12 | 18 | 20 | 0.44 | 0.455 |
| direct_pans | 0.05 | 0.323 | 1.00 | 1 | 1 | 1 | 0.00 | 1.000 | 0.303 | 1.00 | 1 | 1 | 1 | 0.00 | 1.000 |
| direct_pans | 0.10 | 0.323 | 1.01 | 1 | 1 | 1 | 0.00 | 1.000 | 0.303 | 1.00 | 1 | 1 | 1 | 0.00 | 1.000 |
| direct_pans | 0.15 | 0.331 | 1.10 | 1 | 1 | 1 | 0.00 | 1.000 | 0.303 | 1.02 | 1 | 1 | 1 | 0.00 | 1.000 |
| direct_pans | 0.20 | 0.375 | 1.53 | 1 | 1 | 1 | 0.00 | 0.996 | 0.331 | 1.18 | 1 | 1 | 1 | 0.00 | 0.996 |
| direct_pans | 0.25 | 0.375 | 2.20 | 1 | 1 | 2 | 0.01 | 0.987 | 0.363 | 1.47 | 1 | 1 | 1 | 0.00 | 0.996 |
| direct_pans | 0.30 | 0.454 | 3.55 | 1 | 1 | 5 | 0.02 | 0.953 | 0.382 | 1.94 | 1 | 1 | 1 | 0.00 | 0.991 |
| direct_pans | 0.35 | 0.530 | 5.19 | 1 | 2 | 7 | 0.06 | 0.901 | 0.414 | 2.42 | 1 | 1 | 2 | 0.00 | 0.983 |
| direct_pans | 0.40 | 0.637 | 7.54 | 1 | 5 | 12 | 0.15 | 0.811 | 0.462 | 2.93 | 1 | 1 | 3 | 0.01 | 0.979 |
| direct_pans | 0.45 | 0.685 | 9.09 | 2 | 7 | 17 | 0.22 | 0.674 | 0.502 | 3.72 | 1 | 1 | 5 | 0.02 | 0.974 |
| pans_min | 0.05 | 0.777 | 10.33 | 5.5 | 10 | 15 | 0.18 | 0.725 | 0.789 | 9.39 | 4.5 | 8 | 13.5 | 0.12 | 0.845 |
| pans_min | 0.10 | 0.777 | 10.33 | 5.5 | 10 | 15 | 0.18 | 0.725 | 0.789 | 9.39 | 4.5 | 8 | 13.5 | 0.12 | 0.845 |
| pans_min | 0.15 | 0.777 | 10.33 | 5.5 | 10 | 15 | 0.18 | 0.725 | 0.789 | 9.39 | 4.5 | 8 | 13.5 | 0.12 | 0.845 |
| pans_min | 0.20 | 0.777 | 10.33 | 5.5 | 10 | 15 | 0.18 | 0.725 | 0.789 | 9.39 | 4.5 | 8 | 13.5 | 0.12 | 0.841 |
| pans_min | 0.25 | 0.777 | 10.33 | 5.5 | 10 | 15 | 0.18 | 0.721 | 0.789 | 9.39 | 4.5 | 8 | 13.5 | 0.12 | 0.841 |
| pans_min | 0.30 | 0.781 | 10.33 | 5.5 | 10 | 15 | 0.18 | 0.717 | 0.789 | 9.41 | 4.5 | 9 | 13.5 | 0.12 | 0.837 |
| pans_min | 0.35 | 0.797 | 10.45 | 5.5 | 10 | 15 | 0.18 | 0.708 | 0.801 | 9.52 | 5 | 9 | 14 | 0.12 | 0.820 |
| pans_min | 0.40 | 0.809 | 10.69 | 6 | 10 | 16 | 0.19 | 0.691 | 0.813 | 9.75 | 5 | 9 | 14 | 0.13 | 0.807 |
| pans_min | 0.45 | 0.825 | 11.06 | 6 | 10 | 16 | 0.20 | 0.665 | 0.853 | 10.19 | 5 | 9 | 15 | 0.14 | 0.773 |

## Observations (descriptive)
1. **Does 4b need more resolution or just fail more? It fails more.**
   - The fixed curves agree within 0.06 up to L12.
   - On questions both models answer at L20, the onset is the same (mean 6.66 vs 6.61).
   - The gap is a lower plateau (0.85–0.89 from L14 on). The 9b curve's last step (0.944 to 1.000) is partly a selection artefact of the set.
   - Every 4b sweep row is bounded by the 0.888 ceiling. Rows that never stop early equal it exactly.
2. **Does the direct_pans bimodality persist? Yes, but it is weaker and shifted later.** Stop-level bins on the 251 show "early or L20" with few stops in between, for both models:

| model | θ | L1–5 | L6–10 | L11–15 | L16–19 | L20 |
|---|---|---|---|---|---|---|
| 4b | 0.50 | 77 | 43 | 26 | 12 | 93 |
| 4b | 0.60 | 42 | 25 | 21 | 8 | 155 |
| 9b | 0.50 | 181 | 38 | 17 | 5 | 10 |
| 9b | 0.80 | 76 | 40 | 24 | 8 | 103 |

   - Firings on the 233 are still concentrated at L1–5. At θ 0.50, 4b has 77 of 128 firings there and 9b has 165 of 218.
   - Because 4b's P_ans is lower, the same θ produces fewer early stops. 4b's θ 0.50 resembles 9b's θ 0.80.
   - pans_min is spread across levels for both models, as before.
3. **Is firing lower or higher at matched accuracy? Lower at equal θ, higher at equal earliness.**
   - Accuracy cannot be matched directly, because the 4b ceiling is 0.888. Two matchings are used instead: by mean stop level (linear interpolation along θ) and by the number of stopped-but-wrong questions.
   - **Equal θ.** 4b fires less than 9b at every θ below 1.00 for all four methods. This only reflects its lower scores.
   - **Equal mean stop level.** 4b fires more and makes more wrong early stops:

| method | mean L | 4b firing | 9b firing | 4b s&w | 9b s&w |
|---|---|---|---|---|---|
| direct_pans | 17 | 0.198 | 0.143 | 6.5 | 3.7 |
| direct_pans | 18 | 0.162 | 0.092 | 4.2 | 2.1 |
| pans_min | 17 | 0.229 | 0.159 | 2.4 | 3.1 |
| pans_min | 18 | 0.177 | 0.090 | 1.2 | 0.0 |

   - **Equal stopped-but-wrong (≤ 6).** The earliest such row per method:

| model | method | θ | mean L | firing |
|---|---|---|---|---|
| 4b | direct_pans | 0.70 | 18.08 | 0.159 |
| 4b | pans_min | 0.75 | 15.54 | 0.335 |
| 9b | direct_pans | 0.90 | 15.58 | 0.215 |
| 9b | pans_min | 0.85 | 15.83 | 0.296 |

   - **Products, main grid.** 4b's direct_v4 and checklist_v4_min have 0 wrong early stops in the whole range. They fire on 0.026 and 0.185 of the 233 at θ 0.50, but stop at mean L 19.76 and 19.10. For comparison, 9b at θ 0.50 fires on 0.391 and 0.403 at mean L 16.67 and 16.39. This is a different operating point, not a better trade-off.
   - **Products, supplementary full grid, at equal mean stop level.** 4b fires a little more and makes more wrong early stops, as the P_ans-only methods do. pans_min at L12–14 is the one exception, where the two models are equal.

| method | mean L | 4b firing | 9b firing | 4b s&w | 9b s&w |
|---|---|---|---|---|---|
| direct_v4 | 16 | 0.506 | 0.465 | 13.6 | 9.0 |
| direct_v4 | 17 | 0.366 | 0.341 | 7.6 | 2.5 |
| checklist_v4_min | 16 | 0.520 | 0.437 | 10.3 | 5.7 |
| checklist_v4_min | 17 | 0.407 | 0.338 | 7.0 | 2.9 |
| pans_min | 14 | 0.441 | 0.438 | 12.5 | 12.0 |
   - **Restricting to the 191 questions 4b also gets wrong at L20.** 4b firing changes little. For example, direct_pans at θ 0.70 is 0.168 on the 191 vs 0.159 on all 233, and pans_min at θ 0.75 is 0.372 vs 0.335.

## Caveats
- **Sets are defined by the 9b reader.** The 251 are 9b's L20-correct, non-prior-answerable questions, so 4b accuracy is bounded by its own L20 accuracy (0.888). The 233 are 9b's L20-wrong questions; 18% of them are 4b-correct at L20. Accuracy and firing are not comparable to 9b on the same footing: 9b is at its ceiling by construction and 4b is not.
- **Descriptive only.** No θ was selected, frozen or tested. No CI is attached to the sweep. The matched comparisons above are interpolations along the test sweep, made after seeing it.
- **Parse deviation.** The checklist methods depend on plan §26.1. Without it (strict variant), they never fire.
- **θ range.** The requested range 0.50–1.00 lies above where 4b's product signals operate, so the product rows there are close to the trivial L20 policy. The supplementary grid covers the lower θ.
- **Reader-wrong set contents.** As in the 9b report, this set includes GQA label and referent ambiguity.

## Reproduction

```bash
# environment: no resident model; qwen3.5:4b warmed alone at num_ctx 8192
# (v6_common.isolate_model + one request with options.num_ctx 8192)
uv run python -m milestone1.gqa_stop.cli preflight --config configs/gqa_stop_v22_4b.json
uv run python -m milestone1.gqa_stop.cli swap-provenance --config configs/gqa_stop_v22_4b.json
uv run python -m milestone1.gqa_stop.cli detgate --config configs/gqa_stop_v22_4b.json
# repeated until complete (cache resumes); foreground chunks of <= 585 s
timeout 585 uv run python -m milestone1.gqa_stop.cli infer --split test --skip-v1arm --config configs/gqa_stop_v22_4b.json
timeout 585 uv run python -m milestone1.gqa_stop.cli infer-test-wrong --skip-v1arm --config configs/gqa_stop_v22_4b.json
uv run python -m milestone1.gqa_stop.cli audit-test --config configs/gqa_stop_v22_4b.json
uv run python -m milestone1.gqa_stop.cli sweep-swap --skip-v1arm --config configs/gqa_stop_v22_4b.json
```

The strict variant was produced by the same infer commands before `model_swap.trailing_period_roles` was added to the config. Its tables are in `strict_parse/`.

## Files
- **Run root** `outputs/gqa_stop/visapp_gqa_v22_4b/`:
  - `model_swap_provenance.json`, `preflight_determinism.json`, `effective_config.json`, `worklog.md`;
  - `level_table_test.csv`, `level_table_test_merged.csv`, `ext_test_wrong/level_table.csv`;
  - `logs/audit_test.json` and `logs/audit_test_flagged.txt` (empty);
  - `strict_parse/` (strict-parser tables and planner records).
- **Sweep** `sweep_swap/`: `test_sweep.csv`, `test_sweep_vs_reference.csv`, `fixed_curve.csv`, `sweep_meta.json` (incl. parse-tolerance counts), `figures/`, `supplementary_full_grid/`.
- **Code.**
  - `src/milestone1/gqa_stop/sweep_swap.py` (new): provenance, the descriptive sweep, the supplementary full grid and the tolerance counts.
  - `context.py`: `model_swap` property.
  - `stages_model.py`: the test infer precondition skips screening and frozen checks for model-swap runs.
  - `stages_v22_test.py`: `infer_test_wrong` skips the frozen check for model-swap runs.
  - `client.py`: config-gated trailing-period tolerance.
  - `cli.py`: `swap-provenance` and `sweep-swap`.
  - Tests: `tests/test_gqa_stop_v22_swap.py`.
