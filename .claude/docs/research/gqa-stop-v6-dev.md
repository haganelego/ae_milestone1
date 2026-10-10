# gqa_stop v6 (plan §25): cross-model agreement gate and approach-framed legibility signal (dev)

Date: 2026-10-10. Run: `configs/gqa_stop_v22.json` (A13 clean decoding, think=false). Outputs: `outputs/gqa_stop/visapp_gqa_v22/v6/` (102) and `ext_dev_wrong/v6/` (99).
Scope: dev only. The sets are the 102 visual questions (§22) and the 99 reader-wrong questions (§23). No test inference, no freeze, no commit. `frozen_policy.json`, `effective_config.json` and the quarantined test records were not touched.
Terminology follows A14. "Firing" means the stop rule is met at some level 1–20; the forced L20 stop is not a firing.

## Verdict

- **V6-A, the agreement gate, fails §25.7 on the mean-level condition.** The primary arm is direct_pans ∧ agree_gemma3 (k=1) compared with direct_pans at 0.95, out of fold (OOF).
  - Accuracy is 0.954 at mean level 14.54 with firing 0.382, against 0.947 @ 13.57 with firing 0.459.
  - (i) Firing difference: −0.077 [−0.143, −0.017], pass.
  - (ii) Accuracy 0.954 ≥ 0.93, pass.
  - (iii) Mean level difference: +0.97 [+0.17, +1.85], above the +0.5 allowance, **fail**.
- **The gate only moves along the existing trade-off.** At equal accuracy and mean level, gated and ungated firing are the same: in-sample frontier 0.364 vs 0.384 at ≤ 15, 0.303 vs 0.303 at ≤ 16, 0.232 vs 0.222 at ≤ 17. This holds for every base score, for k=2, and for both second readers.
- **H-A1 (gemma3) and H-A2 are not supported.**
  - gemma3 agrees with the primary reader equally often on the 102 and the 99 (0.509 vs 0.509).
  - The same-family qwen3.5:4b separates the sets slightly better (+0.076 [+0.005, +0.150]), which is the opposite of H-A2.
  - gemma3 is a weak reader on these questions: GT accuracy on the 102 is 0.22 at L1 and 0.55 at L20.
- **Many of the 99 look like label problems, not invisibility.** Both readers give the same wrong answer at L20 for 51/99 (gemma3) and 61/99 (qwen4b). 37 of the 39 gemma3-gated fires at 0.95 stop with both models on the same wrong answer. No agreement gate can catch these.
- **V6-B, the legibility signal, stopped at the pilot (§25.5).** It failed the option-order gate: the mean absolute difference between orders is 0.224, above the 0.15 limit.
  - Ranks are stable (Spearman 0.944). The failure is a recency bias: listing "Enough" last raises P(Enough) by about +0.22 at every level.
  - The dev legibility run, the policy evaluation and the V6-B pass test were not run.
- **Overall.** Neither v6 arm beats direct_pans on dev, so no test run follows from §25. direct_pans (P_ans alone) remains the best direct signal for not firing on the 99 at a given cost.

## Environment and controls (task 0)

- **GPU and determinism gate.** qwen3.5:9b loads 100% on the GPU after the reboot (driver 580.178). `detgate` passed: 120 calls, 1164 positions, 0 argmax violations, both repeat pairs byte-identical.
  - 92 of those calls have a cached record from before the outage. Their raw text and logprobs are byte-identical 92/92 (`v6/detgate_cache_compare.json`).
  - The previous report is kept as `v6/preflight_determinism_prev_20261009.json`.
- **gemma3 vision encoder on the CPU (found and fixed).** `ollama ps` reports 100% GPU, but that covers only the language-model weights.
  - When gemma3 loaded while qwen3.5:9b was resident, ollama placed its vision encoder on the CPU (log: `CLIP using CPU backend`). Calls took about 19.5 s instead of about 0.6 s.
  - The 29 records made that way are in `logs/cpu_quarantine_v6/`. All pilot records were regenerated.
  - Since then, every model batch first unloads all resident models (keep_alive 0) and warms the target up alone.
  - Every call is audited: non-generation time over 5 s counts as a suspect. Result: 0 suspects in 8,040 second-reader calls (max 3.8 s). The ollama log shows only `CLIP using CUDA0` since 17:20 JST.
- **Second-reader settings** (`v6/effective_config_v6.json`).
  - gemma3's Modelfile sets temperature 1, top_k 64, top_p 0.95. The A13 block overrides them.
  - `think:false` is accepted by both models (no API error), so it is sent.
  - Records are in `reader_gemma3/` and `reader_qwen4b/`. The model digest is in the path and the cache key (gemma3 a2af6cc3eb7f, qwen3.5:4b 2a654d98e6fb).
  - The run's `effective_config.json` was not rewritten: `frozen_policy.json` stores its sha256.
- **Greedy check.** Second-reader calls also request top-10 logprobs.
  - On answer-value tokens, the emitted token is the argmax in every call (0 violations).
  - gemma3 also has 1,394 mismatching token positions over its 4,020 calls, all at JSON structure. With format=json the grammar forces the opening `{"` where gemma3's unconstrained argmax is "Gray" or a code fence. This is format=json, not sampling.
- **Determinism (pilot, 20 requests sent twice).** The generated text is identical 20/20 for both models and equals the stored record.
  - gemma3's logprobs differ in the 3rd–4th decimal between a fresh call and an ollama prompt-cache hit. qwen's are identical.
  - The gate uses the text, as §25.5 specifies.

## V6-A pilot (§25.5; mechanics only)

20 + 20 questions × L1/5/10/15/20 per reader (`v6/agree_pilot_summary.json`).

| check | gemma3 | qwen3.5:4b |
|---|---|---|
| JSON valid after repair (first attempt) | 1.00 (1.00) | 1.00 (1.00) |
| think:false accepted | yes | yes |
| determinism, text identical (logprobs identical) | 20/20 (0/20) | 20/20 (20/20) |
| value-token argmax violations | 0 / 205 | 0 / 246 |
| CPU-encoder suspects | 0 | 0 |
| 30 sampled disagreements: spelling variants (wooden/wood, grey/gray) | 0 | 0 |
| 30 sampled disagreements: near-synonym or granularity | 3 (white/beige ×2, maroon/purple) | 2 (white/beige, maroon/red) |

The other disagreements are substantive, for example metal vs gray, round vs rectangular, or black vs orange. Synonym merging would not change the picture.

## V6-A agreement by level (`agree_by_level.csv`, `agree_by_level.png`)

| set | level | primary acc | agree gemma3 | gemma3 acc | agree qwen4b | qwen4b acc |
|---|---|---|---|---|---|---|
| 102 | 1 | 0.33 | 0.47 | 0.22 | 0.67 | 0.30 |
| 102 | 5 | 0.66 | 0.51 | 0.36 | 0.82 | 0.56 |
| 102 | 10 | 0.76 | 0.52 | 0.41 | 0.81 | 0.67 |
| 102 | 15 | 0.88 | 0.51 | 0.53 | 0.86 | 0.81 |
| 102 | 20 | 1.00 | 0.55 | 0.55 | 0.83 | 0.83 |
| 99 | 1 | 0.12 | 0.47 | 0.13 | 0.75 | 0.11 |
| 99 | 5 | 0.09 | 0.47 | 0.15 | 0.79 | 0.09 |
| 99 | 10 | 0.06 | 0.51 | 0.11 | 0.84 | 0.06 |
| 99 | 15 | 0.08 | 0.50 | 0.15 | 0.69 | 0.15 |
| 99 | 20 | 0.00 | 0.52 | 0.13 | 0.62 | 0.17 |

- **gemma3 agreement is flat, about 0.5, at every level on both sets.** On the 102 it agrees almost only when it is correct; at L20 agreement and accuracy are both 0.549. On the 99 it agrees on the primary's wrong answer about half the time at every level.
- **qwen3.5:4b agrees much more (0.6–0.86), on wrong answers as well.** Its agreement on the 99 falls only after L13.

Hypotheses (§25.3). Per-question mean agreement over L1–L20. The bootstrap resamples images over 102 ∪ 99, so the 10 shared images keep their questions in both sets.

| test | value [95% CI] | result |
|---|---|---|
| H-A1 gemma3: mean agreement 102 − 99 | −0.000 [−0.099, +0.100] (0.509 vs 0.509) | not supported |
| H-A1 qwen4b (control): 102 − 99 | +0.076 [+0.005, +0.150] (0.816 vs 0.739) | positive |
| H-A2: (102−99)_gemma3 − (102−99)_qwen4b | −0.077 [−0.177, +0.020] | not supported (wrong sign) |

Both readers on the same wrong answer, on the 99:

| | gemma3 | qwen3.5:4b |
|---|---|---|
| at L20 | 51 / 99 | 61 / 99 |
| at some level | 87 / 99 | 94 / 99 |
| gated fires at 0.95 (direct_pans ∧ agree) | 39 fires, 37 on the same wrong answer | 46 fires, 42 on the same wrong answer |

These are the candidates for §23's label and referent ambiguity. Two models from different families agree with each other against the GQA label.

## V6-A policies (`agree_targets.csv`, `agree_oof.csv`, `agree_paired_*.csv`, `agree_frontier.csv`, `agree_curves.png`)

- **Setup.** θ grid 0.01–1.00 plus ∞. Selection: accuracy ≥ target on the 102, minimum mean level, ties to higher accuracy and then larger θ.
- **Out of fold.** The images of 102 ∪ 99 are split 5-fold (image-grouped), seeds 0–9. θ is chosen on the train-fold 102 only.
- **Bootstrap.** Image-cluster paired, 2000 draws, seed 20261008. Level and accuracy use the 102 clusters; firing uses the 99 clusters.
- **Reproduction.** The ungated rows reproduce d876011 exactly: direct_pans in-sample 0.951 @ 13.48, firing 0.465; OOF 0.947 @ 13.57, firing 0.459. direct_v4 OOF is 0.945 @ 13.98, firing 0.646.

### Target 0.95 (in-sample θ; OOF = seed mean)

| policy | θ | acc | mean L | firing 99 | OOF acc | OOF mean L | OOF firing |
|---|---|---|---|---|---|---|---|
| direct_pans (comparator) | 0.83 | 0.951 | 13.48 | 0.465 | 0.947 | 13.57 | 0.459 |
| **direct_pans ∧ agree_gemma3 (primary)** | 0.79 | 0.951 | 14.38 | 0.394 | 0.954 | 14.54 | 0.382 |
| direct_pans ∧ agree_gemma3, k=2 both | 0.77 | 0.951 | 15.33 | 0.354 | 0.946 | 15.30 | 0.352 |
| direct_pans ∧ agree_gemma3, k=2 either | 0.78 | 0.951 | 14.94 | 0.374 | 0.950 | 14.96 | 0.373 |
| direct_pans ∧ agree_qwen4b | 0.83 | 0.951 | 13.51 | 0.465 | 0.947 | 13.60 | 0.459 |
| direct_v4 (comparator) | 0.30 | 0.951 | 13.83 | 0.677 | 0.945 | 13.98 | 0.646 |
| direct_v4 ∧ agree_gemma3 | 0.26 | 0.951 | 14.90 | 0.495 | 0.949 | 14.99 | 0.487 |
| direct_v4 ∧ agree_qwen4b | 0.30 | 0.951 | 14.06 | 0.566 | 0.945 | 14.19 | 0.537 |
| checklist_v4_min | 0.28 | 0.951 | 14.15 | 0.646 | 0.946 | 14.16 | 0.648 |
| checklist_v4_min ∧ agree_gemma3 | 0.26 | 0.951 | 15.17 | 0.515 | 0.945 | 15.18 | 0.495 |
| pans_min | 0.58 | 0.951 | 11.73 | 0.697 | 0.950 | 12.85 | 0.584 |
| pans_min ∧ agree_gemma3 | 0.69 | 0.951 | 14.41 | 0.384 | 0.941 | 14.29 | 0.404 |

### Primary comparison, OOF paired (gated − ungated)

| target | Δ mean L [CI] | Δ acc [CI] | Δ firing [CI] |
|---|---|---|---|
| 0.90 | +1.90 [+1.02, +2.78] | −0.005 [−0.028, +0.022] | −0.112 [−0.186, −0.044] |
| **0.95** | **+0.97 [+0.17, +1.85]** | +0.007 [−0.024, +0.041] | **−0.077 [−0.143, −0.017]** |
| 0.97 | +1.17 [+0.51, +1.89] | +0.006 [−0.008, +0.030] | −0.067 [−0.131, −0.010] |
| 0.99 | +0.45 [+0.17, +0.81] | 0.000 | −0.020 [−0.045, −0.003] |

Against direct_v4 (reported, not part of the V6-A rule): at 0.95 OOF, Δ firing is −0.265 [−0.384, −0.146] and Δ mean level +0.56 [−0.96, +2.09]. Most of that gain is direct_pans' own gain over direct_v4.

### Frontier: minimum firing with accuracy ≥ 0.95 and mean level ≤ b (in-sample)

| policy | ≤ 14 | ≤ 15 | ≤ 16 | ≤ 17 |
|---|---|---|---|---|
| direct_pans | 0.444 | 0.384 | 0.303 | 0.222 |
| direct_pans ∧ agree_gemma3 | — | 0.364 | 0.303 | 0.232 |
| direct_pans ∧ agree_qwen4b | 0.444 | 0.384 | 0.303 | 0.212 |
| direct_v4 | 0.677 | 0.545 | 0.475 | 0.384 |
| direct_v4 ∧ agree_gemma3 | — | 0.495 | 0.404 | 0.333 |
| checklist_v4_min | — | 0.566 | 0.434 | 0.273 |
| checklist_v4_min ∧ agree_gemma3 | — | — | 0.414 | 0.253 |
| pans_min | 0.485 | 0.313 | 0.253 | 0.202 |
| pans_min ∧ agree_gemma3 | — | 0.343 | 0.242 | 0.172 |

- **Reading.** On direct_v4 the gemma3 gate lowers firing at equal cost by 0.05–0.07. That is still worse than simply dropping Rel (direct_pans).
- **On direct_pans and pans_min, the gate is within ±0.03 of the ungated frontier.** The right panel of `agree_curves.png` shows the gated and ungated direct_pans curves on top of each other.

### Attribute and relate breakdown (0.95, in-sample; `agree_breakdown.csv`)

| policy | color: acc / L / firing | material | shape | relate 0 | relate 1 |
|---|---|---|---|---|---|
| direct_pans | 0.95 / 10.3 / 0.52 | 0.93 / 15.1 / 0.50 | 1.00 / 17.2 / 0.15 | 0.94 / 14.1 / 0.49 | 0.96 / 12.8 / 0.43 |
| direct_pans ∧ agree_gemma3 | 0.98 / 12.2 / 0.39 | 0.93 / 15.3 / 0.47 | 0.93 / 17.4 / 0.23 | 0.94 / 15.0 / 0.42 | 0.96 / 13.7 / 0.36 |

The gate's effect is mostly on colour: firing falls by 0.13 and the mean level rises by 1.9. On material and shape it does almost nothing, and shape firing rises.

## V6-B pilot (§25.5; `v6/leg_pilot_summary.json`, `leg_pilot.csv`, `leg_pilot_order.png`)

- **Calls.** 20 + 20 questions × L1/5/10/15/20, with the original question and each planner_v3 item, in both option orders. That is 1,230 calls, all valid, in `legibility_v6/` and `legibility_v6_swap/`.
- **Canonical first tokens.** "Enough" (348/348 replies starting with Enough) and "Clo" (246/246; "Closer" is split as Clo + ser). They are recorded in the pilot summary only. They were not fixed in `effective_config_v6.json`, because the gates failed.

| gate (§25.5) | value | result |
|---|---|---|
| reply starts with Enough / Closer ≥ 0.95 | 0.966 (swap 0.992). The rest start "The image is extremely blurry…" or "Looking at the image…" | pass |
| has_option (canonical token in top-10) ≥ 0.95 | 1.00 (swap 1.00) | pass |
| mean Leg on the 102, L20 > L1 | direct 0.146 → 0.668; items 0.226 → 0.763 | pass |
| order swap: Spearman ≥ 0.8 | 0.944 | pass |
| order swap: mean absolute difference ≤ 0.15 | **0.224** | **fail** |
| Spearman with Rel < 0.95 | 0.901 (direct 0.891) | pass |

- **The failure is a systematic recency bias, not rank noise.** Swap − original P(Enough) is +0.20 / +0.23 / +0.24 / +0.23 / +0.21 at L1/5/10/15/20.
  - 26% of calls flip their argmax option.
  - First tokens are Enough 348 vs 487 and Clo 246 vs 123 between the two orders.
- **Descriptive only (no firing measured).** Mean direct Leg is nearly the same on the 102 and the 99 at every pilot level in both orders (0.404 vs 0.405 in the original order), and the curves overlap in `leg_pilot_order.png`. This hints that Leg would not separate the sets either, but it is a pilot observation, not a test.
- **The rule says stop, so V6-B was stopped.** Possible amendments would each need a new pre-registration and a user decision. One is to average both orders. The other is to accept a level-independent offset, since θ is re-selected and ranks are stable.

## Caveats

- **In-sample θ, many logged variants.** Only the primary comparison is confirmatory, and its OOF result is what the verdict uses. The 32 logged V6-A variants are descriptive.
- **The 99 are noisy as "unanswerable" questions.** More than half have two models from different families agreeing on the same non-GQA answer at L20. Firing on the 99 therefore partly measures label disagreement.
- **gemma3:latest is a 4B model and weak on GQA colour/material.** A stronger different-family reader could agree more selectively. That was not pre-registered and was not tried.
- **ollama prompt cache.** For gemma3 it shifts logprobs, not the text. Second-reader signals use the text only.

## Reproduction

```bash
uv run python -m milestone1.gqa_stop.cli detgate --config configs/gqa_stop_v22.json
uv run python -m milestone1.gqa_stop.cli v6-agree-pilot --config configs/gqa_stop_v22.json
timeout 585 uv run python -m milestone1.gqa_stop.cli v6-agree --config configs/gqa_stop_v22.json   # repeat until done (cache resumes)
uv run python -m milestone1.gqa_stop.cli v6-agree-eval --config configs/gqa_stop_v22.json          # cache-only
uv run python -m milestone1.gqa_stop.cli v6-leg-pilot --config configs/gqa_stop_v22.json
# v6-leg / v6-leg-eval exist but refuse to run: the V6-B pilot gates failed
```

Code: `src/milestone1/gqa_stop/v6_common.py` (sets, v6 config, model isolation), `v6_agree.py`, `v6_eval.py` (gated stop rules, selection, out-of-fold, paired bootstrap, frontier, pass rule) and `v6_legibility.py`. Tests: `tests/test_gqa_stop_v6.py` (16). The full `tests/test_gqa_stop*.py` suite passes (341).
