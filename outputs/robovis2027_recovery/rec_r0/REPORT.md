# ROBOVIS2027 recovery run `rec_r0` — report (plan of record 2026-10-08 v1.0)

(2026-10-08 03:25 JST. All Phases A–D complete. Presentation follows the user's guidance: significance of differences vs level-agnostic comparators is what matters; fixed-level stops are schedule-aware references only.)

## 要約（日本語）
- 計画書（`.claude/docs/research/robovis2027-recovery-plan.md`）のPhase A・B・C・Dはすべて完了した。
- Phase A（事前登録held-out、96枚、調整なし）：**主規則 hybrid_fuse_mean は再現しなかった**（有意セル 0/18、MAE 3.35 は画像のみ9b単独の 3.21 より悪い）。副規則 fuse_min は MAE 2.54 で、text 4b・画像のみ9b・信頼度baselineに対し MAE/R50/relsg のほぼ全セルで有意に優位（全96で9/9、≤18で8/9）。ただし停止が約2段階遅く観測量が揃っていない。同観測量では固定曲線上（Δ≈0）で、GT効率の改善はない。image-only 9bの初回採点は01:57に0件（HTTP 500、VRAM競合）で、同一条件の再実行（02:35–03:00、1536/1536、エラー0）で完了した。
- Phase B：GTを見た上限は大きい（dev20 R50、b=15でO−F=+0.085）が、単一seedに由来する楽観を含む。二択oracleの余地はdev R50で+0.047。監査の結果、従来の「固定と同等」という比較は、F(b)より弱い隣接混合を相手にしていたと判明した（差は最大0.050）。
- Phase C：seed間で選び合うと、二択oracleの余地は+0.016/+0.038（R50）に縮む。採否4条件はすべて満たした（GO）。ただし条件3の差は+0.004/+0.009と僅少で、relsg_gtは全seedで負だった。
- Phase D（新60画像）：主比較であるR50のU方策と同継続率混合の差は、seedAで−0.007 [−0.025, +0.012]、seedBで−0.011 [−0.032, +0.010]。**両seedとも「正でない」**。fixed15とF(b)も下回った。MIRU流の「oracle停止との一致率」でも40–43%と偶然（50%）以下で、有意に優位な比較は一つもない。
- 副指標relsg_gtは両seedで+0.023（CIは0を含まず、置換p 0.038/0.017）。ただしdevでは逆符号であり、計画§7に従い主結果には昇格しない。
- 結論：物体別視認性Uによる「t10で止める／t20まで進む」の二択は、R50で画像に依存しない配分を上回らなかった。主張できる範囲は「物体別視認性による追加観測割当」の否定的・分析的結果に限られ、共起仮説やscene graph推論の新規性の証拠にはならない。
- 旧MIRU集合の除外件数「29/35」の相違は、commit `2b038b2` の記録で解決した。c09=20が29件、c09=19が6件で、計35件を除外していた。
- 計画§8の成果物はすべて揃った（`heldout_confirmatory.md`、Phase A図、`usage.jsonl`、`failures.jsonl`、`reproduce.md` 更新済み）。

---

## 0. Status

| item | value |
|---|---|
| date | 2026-10-08, ≈03:00 JST (draft) |
| HEAD | `1b565f8` (Phase D), branch `feature/robovis2027`; working tree has untracked outputs |
| complete | §1 preflight; Phase A (held-out registered evaluation, `heldout_confirmatory.md`); Phase B (cache-only diagnostics + audit); Phase C (multiseed C.1, frozen U policy C.2, gate = GO); Phase D (seedA 1200 + seedB 180 PGSG states, v4 60 states, evaluation, MIRU-style significance table, conf_two_stage comparator) |
| pending | Phase E freeze (due 10/13): figures polish, manuscript |

Every number below cites a file under `outputs/robovis2027_recovery/rec_r0/` unless another path is given. The earlier consolidated report is `outputs/robovis2027/REPORT.md`; it is cited here, not repeated.

## 1. Purpose, questions, completion criteria

Plan §0 limits the work to three questions:
1. Does the pre-registered late-fusion improvement reproduce on the designated held-out set?
2. Does the current observer plus GT metrics leave reproducible headroom for image-dependent allocation of observations?
3. If so, can the existing v4 per-object visibility decide "stop at t10 / continue to t20"?

Fixed throughout: no model training, 20 resolution levels, primary metric R50, secondary relsg_gt. v4 is the policy's perceiver, not a GT scorer.

**Completion (§10):** A, B and C plus the gate record; D if Go; figures, reproduction steps and limitations. Success of the new method is not a completion condition.

| criterion | state |
|---|---|
| A complete | Yes (`heldout_confirmatory.md`, `outputs/heldout_eval/heldout_registered_table.md`) |
| B complete | Yes |
| C + gate record | Yes (`gate_decision.json`, GO) |
| D (Go) | Yes |
| figures / reproduce / limitations | Yes (Phase D and Phase A figures; `reproduce.md` finalized; limitations §7.3) |

## 2. Preflight (plan §1)

Sources: `recovery_preflight.md`, `frozen_config.json`, `model_hashes.json`, `data_manifest.json`.

**Environment.**
- The preflight was taken at HEAD `079ee22` with 60 dirty paths, listed in `frozen_config.json`.
- Local RTX 5080 16 GB with ollama; no external API cost.
- Robovis check/judge calls: 6,031 attempts at preflight against the `execution.max_model_calls` cap of 9000.

**Weights and models** (`model_hashes.json`):

| component | identity |
|---|---|
| PGSG `vg_sgg.pth` | sha256 `d90c63ca…e299eb` (equals the dev_r0 manifest), submodule `cff7420` |
| EGTR ckpt `epoch=03-validation_loss=1.71` | sha256 `8926640f…1f84d5`, submodule `811217d` |
| qwen3.5:4b / 9b | digests `2a654d98…` / `6488c96f…` |

**PGSG generation (effective values).**
- Sampling: nucleus top_p 0.9, HF top_k 50.
- repetition_penalty 1.1. The task passes 1.5, but `med.py` hard-codes 1.1.
- Segments: 36 segments, seg_len 32.
- Inputs: 384-base images, 20 geometric levels (0.05→1.0), bilinear, JPEG q95. `levels_manifest.json` sha256 is `4bf406ee…`.

**Seeds and batches.**
- Seed 0 uses base 42, batch size 4. A batch is the next 4 images in rank order at one level, seeded by `state_seed(batch_ids, level, base)`.
- Seeds 1 and 2 use bases 43 and 44 with identical batch ids and order (`multiseed_manifest.json`, `reproduce.md` §C.1).
- No fallbacks occurred.

**GT.** `test.json` / `rel.json` / `train.json` sha256 are in `data_manifest.json`. The local train/val/test splits share 0 ids.

**Id sets** (`data_manifest.json`):

| set | n | notes |
|---|---|---|
| ROBOVIS dev20 | 20 | no overlap with any other set (smoke_a3 ⊂ dev20) |
| ROBOVIS test100 | 100 | already viewed → exploratory only |
| MIRU100 (current, `2b038b2`) | 100 | test100 ∩ MIRU100 = 4 |
| held-out96 | 96 | = test100 minus those 4 (`selection_note.json`) |
| PGSG pilot (first 1000 test images) | 1000 | aggregate validation only |
| new test60 (Phase D) | 60 | see §6.1 |

**Disclosures** (the registered held-out set is unchanged):
1. One held-out96 image was in the original MIRU100 (`e1b8e56`), before the `2b038b2` re-selection. It is not in the current MIRU100.
2. Three held-out96 images are among the first-1000 PGSG pilot images. Those were used only for aggregate validation; no curves were examined.

Additional note: held-out96 ⊂ test100. Their PGSG (ROBOVIS) curves had been viewed, so held-out96 is held out only with respect to the MIRU fusion rule (EGTR pipeline).

**Preflight state of cached outputs.**
- Held-out EGTR: 1920/1920. Text-4b scores: 0/1536. Image-only 9b scores: 0/1536.
- One GO-waiting runner existed (PID 127475).
- v4: 400 checks, 400 decisions, 20 images × 20 levels.
- h3: 100 states (20 × 5 levels).

## 3. Phase A — pre-registered held-out late fusion (**COMPLETE — primary not confirmed**)

See `heldout_confirmatory.md` for the full record (registered conditions verbatim, selection history, incident, results placeholders).

**Registered conditions** (header of `configs/config_heldout.yaml`, recorded ≈00:40 before any held-out score existed): PRIMARY `hybrid_fuse_mean(vlm image-only 9b, llm_sgg text 4b)`, k=2; secondary `fuse_min(...)`, θ 0.70, k=2; primary set all 96 images (MAE vs stop_c09 recomputed from held-out EGTR L20); comparative set stop_c09 ≤ 18 (73 images); comparators text 4b, image-only 9b, hybrid text 4b, node-confidence baseline, fixed curve; GT R50 / relsg_gt equal-observation deltas on both sets; inputs = scripts/11 single mode with MIRU decoding unchanged (temperature 0.3, no JSON mode, unseeded) and scripts/40 text condition.

- **Registration-text discrepancy (disclosed).** The header says "theta 0.70" for the primary; the hybrid method key, as registered and implemented, applies `ablation.hybrid.threshold` = 0.5 to 0.5·node confidence + 0.5·fused score (the same machinery as the MIRU-100 reference row MAE 2.59). θ 0.70 applies to the non-hybrid fusion rules. The method key governs; a dated correction comment (02:40 JST) was appended before any held-out result for the registered rules or any LLM/VLM method was computed (only the zero-call baseline row existed at 00:57; text-4b scores existed unevaluated). Nothing was re-tuned.
- The overlay also sets `vlm.num_ctx: 16384`, asserted to only size the context (recorded, not verified).

**Execution status.**
- Text-4b held-out scores are complete (1536/1536, 0 failures; generated once, never regenerated).
- The first image-only 9b run ended at 01:57:53 with **0 score files** (`failures.jsonl`): the script-11 start-up smoke test got HTTP 500 from ollama (likely VRAM exhaustion: two resident runners + PGSG seedA ≈7 GB; CUDA OOM in the ollama log from 02:01); the script aborted the model by design and wrote `stop_level=-1` rows (preserved at `rec_r0/logs/heldout_vlm_stop_failed_20261008_0157.csv`).
- The re-run uses unchanged settings and started at 02:35:51 (`outputs/robovis2027_miru/logs/heldout_vlm9b_rerun.log`, smoke test passed).

**Held-out results available now** (`outputs/heldout_eval/miru_method_table.md`, baseline only):

| method | mean stop | MAE c09 (96) | MAE c09 (≤18, n=73) | at final | R50@stop | R50 Δ legacy mix | R50 Δ F | relsg@stop | relsg Δ legacy mix | relsg Δ F | relsg Δ fixed20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (node conf ≥ 0.35) | 14.729 | 3.458 | 3.397 | 0.302 | 0.108 | −0.026 [−0.050, −0.004] | −0.028 [−0.053, −0.009] | 0.361 | −0.029 [−0.048, −0.010] | −0.029 [−0.049, −0.013] | −0.069 [−0.092, −0.046] |
| fixed_20 | 20.000 | 4.104 | 5.370 | 1.000 | 0.159 | n/a | n/a | 0.430 | n/a | n/a | n/a |

**Unreached rate (c09 = 20).** 21/96 images (21.9%) never saturate; 23/96 have c09 ≥ 19. The registered definition (unreached = 20) is kept; the 73-image subset is reported alongside. The held-out set differs from MIRU100: c09 mean 15.90 vs 13.48; t20 R50 0.159 vs 0.240; t20 relsg 0.430 vs 0.478.

**MIRU100 in-sample reference** (`outputs/miru_gt_eval/miru_method_table.md`; fusion was selected post hoc on these 100 images, so these numbers are optimistic):

| method (k=2) | mean stop | MAE c09 | R50 Δ F | relsg Δ F | relsg Δ fixed20 |
|---|---|---|---|---|---|
| hybrid_fuse_mean(vlmimg9b, text4b) — registered primary | 13.49 | 2.59 | −0.008 [−0.038, +0.007] | −0.006 [−0.026, +0.007] | −0.081 [−0.109, −0.056] |
| fuse_min(vlmimg9b, text4b) — registered secondary | 16.16 | 3.12 | −0.002 [−0.026, +0.011] | +0.006 [−0.010, +0.016] | −0.023 [−0.042, −0.004] |
| hybrid_llm_sgg_text_4b | 13.58 | 2.98 | −0.004 [−0.034, +0.013] | −0.006 [−0.030, +0.010] | −0.079 |
| llm_sgg_text_4b | 13.76 | 3.00 | −0.004 [−0.034, +0.013] | −0.006 [−0.028, +0.011] | −0.074 |
| vlm_single_9b (θ 0.70) | 14.32 | 3.04 | −0.019 [−0.048, −0.005] | −0.017 [−0.037, −0.001] | −0.073 |
| baseline | 14.31 | 3.49 | −0.019 [−0.045, −0.004] | −0.032 [−0.055, −0.011] | −0.088 |

Even in-sample, no fusion row has a Δ F CI wholly above 0. The registered claim is about MAE; the GT-efficiency check is reported alongside, per plan §3.

**The 29/35 exclusion-count discrepancy is resolved.** The `2b038b2` commit message states "removed 35 (29@20, 6@19), refilled to 100 via deterministic seed-42 shuffle continuation"; `data_manifest.json` agrees (`miru_history@e1b8e56 ∩ miru100_current = 65`, so 35 replaced). 29 counts only the c09 = 20 images; 35 = 29 + 6 with c09 = 19.

**Results (registered conditions, no tuning; full table and significance in `heldout_confirmatory.md` §5 and `outputs/heldout_eval/heldout_registered_table.md`).**

| method | stop | MAE all 96 / ≤18 | R50 | relsg | GT Δ vs legacy mix (R50 / relsg) | Δ vs F (pt) |
|---|---|---|---|---|---|---|
| **PRIMARY hybrid_fuse_mean** | 13.79 | **3.35 / 2.93** | .109 | .360 | −.008 [−.026, +.007] / −.016 [−.033, +.004] | −.018 / −.018 |
| secondary fuse_min | 16.04 | **2.54 / 2.47** | .138 | .398 | −.005 [−.016, +.004] / −.001 [−.013, +.011] | −.009 / −.004 |
| text 4b | 13.73 | 3.56 / 3.04 | .110 | .357 | −.007 / −.016 | −.017 / −.020 |
| image-only 9b | 14.17 | 3.21 / 2.67 | .115 | .360 | −.010 / −.024 [−.049, −.001] | −.016 / −.024 |
| baseline (node conf) | 14.73 | 3.46 / 3.40 | .108 | .361 | −.026 [−.050, −.004] / −.029 [−.048, −.010] | −.028 / −.029 |

- **PRIMARY not confirmed:** 0/18 significance cells vs the level-agnostic comparators (text 4b, image-only 9b, baseline); its MAE is worse than image-only 9b alone. The in-sample MIRU-100 edge (2.59 vs 3.00; p ≈ .03 with CI touching 0 in the pairwise table) did not survive.
- **Secondary fuse_min:** significantly better than all three comparators on |stop − c09|, R50 and relsg (all-96 9/9 ✓; ≤18 8/9 ✓), **but** it stops ≈2 levels later (not budget-matched) and at equal observation sits on the fixed curve (Δ ≈ 0); 21/96 never-saturating images reward late stopping. On ≤18 it still beats text 4b and baseline on MAE.
- No method improves GT efficiency vs the legacy mix; all Δ F < 0. Figure: `outputs/heldout_eval/figures/heldout_quality_vs_level.png`.

## 4. Phase B — cache-only headroom (plan §4)

Inputs and hashes: `diagnostics_manifest.json`. dev20 and test100 use PGSG on 384-base images; MIRU100 and held-out96 use EGTR on original-size images. All sets have complete series (coverage 100%), no missing cells, no zero-filling (`audit.md` §3). Only dev20 is a selection set; test100 is exploratory (already viewed).

### 4.1 F(b), O(b), O−F

Source: `budget_oracle.csv`, `fixed_mix.csv`. F = best image-independent fixed mixture over all 20 levels (upper hull; GT-posterior reference). O = omniscient per-image multiple-choice knapsack DP with integer budget floor(N·b); infeasible and optimistic (single seed).

| set (observer) | metric | O−F b=10 | O−F b=12 | O−F b=15 | O−F b=18 | F(15) | O(15) | O mean level |
|---|---|---|---|---|---|---|---|---|
| dev20 (PGSG) | R50 | 0.148 | 0.118 | 0.085 | 0.067 | 0.201 | 0.286 | 9.05 |
| dev20 (PGSG) | relsg_gt | 0.136 | 0.121 | 0.100 | 0.071 | 0.547 | 0.647 | 14.4 |
| test100 (PGSG, exploratory) | R50 | 0.108 | 0.091 | 0.073 | 0.060 | 0.132 | 0.205 | 9.06 |
| test100 (PGSG, exploratory) | relsg_gt | 0.146 | 0.140 | 0.127 | 0.108 | 0.476 | 0.604 | 13.62 |
| MIRU100 (EGTR) | R50 | 0.153 | 0.133 | 0.093 | 0.055 | 0.187 | 0.280 | 10.93 |
| MIRU100 (EGTR) | relsg_gt | 0.139 | 0.130 | 0.089 | 0.054 | 0.436 | 0.525 | 14.74 |
| held-out96 (EGTR) | R50 | 0.100 | 0.080 | 0.050 | 0.028 | 0.139 | 0.189 | 8.72 |
| held-out96 (EGTR) | relsg_gt | 0.135 | 0.112 | 0.083 | 0.054 | 0.393 | 0.475 | 14.47 |

For R50, O already reaches mean qmax at b=10 (mean level ≈9): images with zero recovery take t1, which frees budget. This is single-seed optimism (plan §4.3), not evidence of practical adaptivity.

### 4.2 t10/t20 two-choice oracle vs B_half

Source: `two_choice_oracle.csv`. Top-k by g = q20 − q10 (k = N/2, negatives included); mean budget 15. GT-posterior.

| set | metric | mean q10 | mean q20 | oracle | B_half | oracle − B_half |
|---|---|---|---|---|---|---|
| dev20 | R50 | 0.134 | 0.196 | 0.213 | 0.165 | +0.047 |
| dev20 | relsg_gt | 0.457 | 0.596 | 0.587 | 0.526 | +0.061 |
| test100 | R50 | 0.091 | 0.140 | 0.163 | 0.116 | +0.047 |
| test100 | relsg_gt | 0.405 | 0.492 | 0.497 | 0.449 | +0.048 |
| MIRU100 | R50 | 0.083 | 0.240 | 0.245 | 0.162 | +0.084 |
| MIRU100 | relsg_gt | 0.313 | 0.478 | 0.462 | 0.396 | +0.066 |
| held-out96 | R50 | 0.048 | 0.159 | 0.163 | 0.103 | +0.060 |
| held-out96 | relsg_gt | 0.273 | 0.430 | 0.416 | 0.351 | +0.065 |

On dev20 R50 the two-choice frame is itself weak: F(15) = fixed15 = 0.201, above the two-choice B_half of 0.165 and close to the oracle (0.213). Dev R50 jumps at t13–15.

### 4.3 Stop-assignment shuffle

Source: `shuffled_assignment.csv`: 108 rows, 1000 permutations each. 9/108 rows have p < 0.05 (about what multiplicity alone would give): dev relsg 4 rows (F1v_v2_G__9b benefit_gated/proposed; F1v_v3_G__j9b observed_only/proposed), MIRU relsg 5 rows. No R50 row reaches p < 0.05. On dev R50 the diagnostic is uninformative (earlier policies stop at t2–5 where q ≈ 0). MIRU registered primary: R50 p = 0.261, relsg p = 0.356. Held-out baseline: R50 p = 0.695, relsg p = 0.525. Exploratory only.

### 4.4 Audit findings (`audit.md`)

1. **Bootstrap.** `paired_bootstrap` resamples images; each resample recomputes the mean stop and the matched fixed mix; levels are never resampled independently.
2. **Comparator correction (material).** The earlier "Δ vs eq_fixed" used the legacy adjacent two-level mix, weaker than F(b) by up to: dev20 R50 0.050 / relsg 0.025; held-out R50 0.041 / relsg 0.017; MIRU R50 0.037 / relsg 0.002; test100 R50 0.012 / relsg 0.019. Consequences: dev20 all 36 rows < 0 against F; the j9b "+0.056" positive CI disappears (−0.009 [−0.069, +0.038]); MIRU only fuse_min/fuse_mean (relsg +0.005/+0.006) and vlm_sgg_single_4b (relsg +0.008) have point estimates ≥ 0, none with a CI wholly > 0; held-out baseline −0.028 (R50) / −0.029 (relsg). Earlier "≈0 vs fixed" statements are optimistic relative to F. Old and corrected values are both kept.
3. **R50 official match.** dev20 PGSG: `all_match=True`, maximum per-image difference 0.0. test100 uses the same code path. EGTR sets: R@50 is recall over the stored relations (≤ 30 per state), not EGTR's official R@50.
4. **Forced vs failure.** Replay records them separately; MIRU stop CSVs cannot distinguish them (`at_final_level` reported).
5. **Integer vs expected budget.** Small negative O−F differences would be integrality effects.
6. **Multiseed consistency.** g equals q20 − q10 in all 120 rows; the seed-0 re-score equals `dev_r0/stage_metrics.csv` (maximum |diff| 0).

## 5. Phase C — multiseed reproducibility and the two-stage U policy (plan §5)

### 5.1 Additional seeds (160 PGSG states)

dev20 × {5, 10, 15, 20}, seeds 1 and 2 (bases 43/44), same batch composition and order as seed 0. Seed 1: 80 states in 250.8 s (3.14 s/state); seed 2: 260.7 s (3.26 s/state); 0 problems.

Per-seed dev20 means (from `multiseed_gain.csv`, n = 20 per cell):

| metric | seed | q5 | q10 | q15 | q20 | mean g | #g>0 / =0 / <0 |
|---|---|---|---|---|---|---|---|
| R50 | 0 | 0.007 | 0.134 | 0.201 | 0.196 | 0.062 | 6 / 13 / 1 |
| R50 | 1 | 0.017 | 0.138 | 0.174 | 0.207 | 0.069 | 7 / 11 / 2 |
| R50 | 2 | 0.017 | 0.150 | 0.150 | 0.219 | 0.069 | 7 / 10 / 3 |
| relsg_gt | 0 | 0.274 | 0.457 | 0.538 | 0.596 | 0.139 | 13 / 5 / 2 |
| relsg_gt | 1 | 0.241 | 0.467 | 0.527 | 0.569 | 0.102 | 14 / 1 / 5 |
| relsg_gt | 2 | 0.256 | 0.452 | 0.516 | 0.582 | 0.130 | 14 / 1 / 5 |

Seed differences for the same image and level (`seed_differences.csv`, mean |Δ| / maximum |Δ|):

| pair | R50 t10 | R50 t20 | relsg t10 | relsg t20 |
|---|---|---|---|---|
| 0–1 | 0.026 / 0.22 | 0.022 / 0.33 | 0.084 / 0.25 | 0.056 / 0.20 |
| 0–2 | 0.027 / 0.20 | 0.077 / 0.67 | 0.113 / 0.33 | 0.053 / 0.21 |
| 1–2 | 0.012 / 0.13 | 0.055 / 0.33 | 0.101 / 0.33 | 0.079 / 0.33 |

Fixed-level means shift by up to ≈0.05 R50 between seeds (q15: 0.201 vs 0.150), larger than the policy margins in §5.3.

### 5.2 Cross-seed two-choice oracle (`cross_seed_oracle.csv`)

| metric | select→eval | cross-seed gain | same-seed oracle gain (eval seed) | Spearman g | tie-random mean [p05, p95] |
|---|---|---|---|---|---|
| R50 (gate) | 1→2 | **+0.016** | +0.056 | 0.59 | 0.026 [0.016, 0.049] |
| R50 (gate) | 2→1 | **+0.038** | +0.049 | 0.59 | 0.039 [0.038, 0.043] |
| R50 | 0→1 / 1→0 | +0.039 / +0.047 | +0.049 / +0.047 | 0.79 | — |
| R50 | 0→2 / 2→0 | +0.006 / +0.013 | +0.056 / +0.047 | 0.23 | — |
| relsg_gt | 1→2 / 2→1 | −0.010 / +0.007 | +0.061 / +0.063 | 0.08 | — |
| relsg_gt | 0→1 / 1→0 | +0.051 / +0.021 | +0.063 / +0.061 | 0.56 | — |
| relsg_gt | 0→2 / 2→0 | +0.008 / +0.024 | +0.061 / +0.061 | 0.22 | — |

Tie-order robustness: only 7 images per seed have g > 0 for R50 (10–11 have g = 0); over 2000 random tie orders the R50 gain was positive in 100% of cases and the reported image-id order equals the 5th percentile. Sampling noise inflates the same-seed R50 headroom by roughly half; relsg headroom does not transfer between seeds 1 and 2.

### 5.3 Two-stage U policy (frozen before test)

Source: `frozen_policy.json`, `dev_two_stage.csv`, `dev_two_stage_eval.csv`.

- U = (n0 + 0.5·n1) / (n0 + n1 + n2 + n_absent), on the cached v4 at t10 (`F1g_v4_G__9b`: top-8 cleaned G boxes, qwen3.5:9b). Malformed answers excluded from the denominator; a state is a failure if it has 0 boxes or fewer than ceil(n_targeted/2) valid answers → t20.
- τ = dev median of valid U = **0.5** (itself a tie value: 6 above, 6 tied). p_tie = (0.5·20 − 6)/6 = **0.667**. Tie rule: SHA256("recovery-v1:" + id)/2^256 < p_tie → t20. No GT used. Frozen at 00:51, before the multiseed outputs and before test.
- Dev decisions: 20/20 valid; 11 → t20 (6 above τ, 5 tie_continue), 9 → t10. U is discrete (K ≤ 8), so 30% of images tie.
- Provenance: v4 decision records did not store ollama options at the time; both dev and test used `--num-ctx 16384` (temperature 0, seed 20271007). A fix recording options in 29d was added afterwards (no re-run).

Policy vs same-p image-independent mix (p = 0.55, mean level 15.5):

| eval seed | R50 policy | R50 mix | R50 Δ | relsg Δ | role |
|---|---|---|---|---|---|
| 0 | 0.158 | 0.168 | −0.010 | −0.032 | reference only (seed whose t10 G fed v4) |
| 1 | 0.180 | 0.176 | **+0.004** | −0.017 | gate |
| 2 | 0.197 | 0.188 | **+0.009** | −0.026 | gate |

U decisions are identical across seed rows (computed once from the seed-0 t10 graph); criterion 3 tests robustness to sampling noise in q, not in U. On dev seed 0, fixed15 (0.201) and F(15.5) lie well above both the policy and the mix.

### 5.4 Gate decision (`gate_decision.json`, 2026-10-08 01:05, config sha256[:16] `bdc43fa9efea865d`)

| # | criterion | value | threshold | pass |
|---|---|---|---|---|
| 1 | v4 valid at dev t10 | 20/20 | ≥ 18/20 | yes |
| 2 | cross-seed two-choice oracle, R50 | +0.0160 (1→2), +0.0376 (2→1), mean +0.0268 | both > 0, mean ≥ 0.01 | yes |
| 3 | frozen U policy − same-p mix, R50 | +0.0038 (seed 1), +0.0092 (seed 2) | both > 0 | yes |
| 4 | budget to finish D by 10/12 | ≈1.2 h GPU PGSG + ≈15 min v4, local | feasible | yes |

**Decision: GO.** Caveats recorded: criterion-3 margins tiny and sign-only (n = 20); relsg negative for the policy on all seeds; discrete U with 6/20 ties; cross-seed shrinkage of the headroom; seed 0 (not a gate seed) negative (−0.010). No threshold, U weight or observation time was searched to pass the gate.

## 6. Phase D — independent 60-image confirmation (plan §6)

### 6.1 Selection (`test_selection.json`, `scripts/44_recovery_test_selection.py`)

VG-150 test split 26,446 → GT-only eligible (≥ 5 objects, ≥ 3 relations) 18,887 → minus all ids in 12 used sets (1,243 ids, 1,039 eligible) → pool 17,848 → sorted ascending, `random.Random(20261008).shuffle`, first 60 (ordered sha256 `bdca3b28…`). No result-based filtering, no replacement of failed images. Independence from the PGSG training split rests on the VG-150 split protocol (authors' h5 not available locally), so the set is called "new", not a guaranteed held-out set. Sequencing: selection (00:53) and degradation (01:03) were zero-model preparation and preceded the GO record (01:05); PGSG inference started at 01:08.

### 6.2 Execution and cost

| item | count | time / notes |
|---|---|---|
| PGSG seedA, t10 first | 60 states | 264.7 s, 0 problems |
| PGSG seedA, remaining levels | 1140 states | 4607.5 s, 0 problems (shared GPU with ollama) |
| PGSG seedB (base 43), t ∈ {10, 15, 20} | 180 states | 554.0 s, 0 problems; batch ids identical to seedA |
| PGSG total | 1380 | = plan upper bound (§6.2) |
| v4 at t10 (9b) | 60 states | median latency 9.4 s (p90 13.3 s); 422 ratings: 0 41.9%, 1 13.3%, 2 41.7%, absent 3.1% |
| v4 failures | 2/60 (2322283, 2336446: HTTP 500 "unexpected EOF" from the runner on all 6 transient attempts at runner start under VRAM pressure; not re-run per the frozen procedure) | → t20; +0.33 levels/image extra cost; not counted as improvement |
| states the policy itself observes | 91 (60 at t10 + 31 at t20) | vs 1200 for a 20-level policy; the reached level is a movement/resolution proxy, not compute |

**Test decisions** (`test_policy.csv`): 58/60 valid; 18/58 tie at U = 0.5 (31%, dev 30%); 9 images have absent > 0. Reasons: below_tau 22, above_tau 18, tie_continue 11, tie_stop 7, failure_continue 2. 31/60 → t20 (p = 0.517), mean level 15.17. Test U values: 0 (1), .125 (1), .214 (1), .25 (2), .286 (1), .312 (1), .333 (2), .375 (3), .4 (2), .417 (2), .438 (6), .5 (18), .562 (2), .583 (2), .6 (3), .625 (6), .643 (1), .688 (2), .857 (1), .875 (1). Dev vs test: median U 0.5 in both; share above/at/below τ .30/.30/.40 vs .31/.31/.38.

### 6.3 Primary comparison — seedA (`test_summary.csv`; paired image bootstrap, 2000 resamples, p recomputed per resample)

| metric | policy | same-p t10/t20 mix | policy − mix [95% CI] | verdict | fixed15 (different budget) | policy − fixed15 | F(15.17) seedA (GT-posterior) | policy − F |
|---|---|---|---|---|---|---|---|---|
| **R50 (primary)** | 0.1290 | 0.1361 | **−0.0071 [−0.0246, +0.0121]** | **not positive** | 0.1419 | −0.0129 | 0.1453 (t12 0.604; t20 0.396) | −0.0163 |
| relsg_gt (secondary) | 0.4852 | 0.4622 | +0.0230 [+0.0003, +0.0468] | positive (CI excl. 0) | 0.4555 | +0.0297 | 0.4801 (t11 0.537; t20 0.463) | +0.0052 (point only) |

SeedA fixed curve: R50 t10 0.099, t15 0.142, t20 0.171; relsg t10 0.401, t15 0.456, t20 0.520.

### 6.4 Replication — seedB (`seed_replication.csv`; same stop times, no hull built from seedB)

| metric | policy | same-p mix | policy − mix [95% CI] | verdict | fixed15 | policy − fixed15 |
|---|---|---|---|---|---|---|
| **R50 (primary)** | 0.1265 | 0.1372 | **−0.0107 [−0.0323, +0.0101]** | **not positive** | 0.1294 | −0.0029 |
| relsg_gt (secondary) | 0.4822 | 0.4596 | +0.0226 [+0.0033, +0.0424] | positive (CI excl. 0) | 0.4501 | +0.0320 |

Figure: `figures/phase_d_quality_vs_level.{png,pdf}` (seedA fixed curve, F, same-p mix, fixed15, U policy with CI). R50: policy below mix, fixed15 and F. relsg: policy above mix and slightly above F.

### 6.5 Verdicts under plan §7 (`phase_d_summary.json`)

- **Primary (R50): "not positive" on seedA and on seedB.** Seeds are not averaged. The U policy does not beat the image-independent t10/t20 mix at the same continuation rate. (fixed15 and F(b) are schedule-aware references only; see §6.6.)
- **Secondary (relsg_gt): positive on both seeds** (+0.023, CIs exclude 0). Not promotable: plan §5.3/§7 forbid swapping the primary metric; on dev the same policy was negative for relsg on all three seeds; the margin above F is a point estimate only (+0.005). Treat as an exploratory observation.

### 6.6 MIRU-style significance view (`method_comparison_miru_style.{md,csv}`)

**Comparator note (user guidance 2026-10-08).** Fixed-level rows (fixed10/15/20) and F(b) assume knowledge of the level index, i.e. of how much resolution has already been recovered; an online observer does not have that. They are therefore *schedule-aware references*, not comparators, and their ✓/✗ marks below are to be read as descriptive only. The fair comparators are the same-p random assignment (same two-stage protocol, no image information) and level-agnostic observation-based baselines (a confidence-based two-stage rule is being added as `conf_two_stage`).

Per the user's guidance, the question is whether any method's score became *significantly* superior. Oracle = GT two-choice oracle on that seed (top-k by g, k = the policy's number of t20 slots); MAE in levels = 10 × (1 − agreement).

| U policy vs (seedA / seedB) | R50 agreement with oracle | R50 MAE vs oracle stop | R50 q(stop) | relsg q(stop) |
|---|---|---|---|---|
| same-p mix | 40% / 43% vs 50% — ✗ (worse; perm p worse 0.09 / 0.21) | 6.00 / 5.67 vs 4.99 — ✗ (worse) | −0.007 / −0.011 — ✗ | **+0.023 / +0.023 — ✓ better** (perm p 0.038 / 0.017) |
| fixed10 | ✗ (worse) | ✗ (worse) | +0.030 ✓ (A) / +0.020 ✗ (B) | **✓ better** (+0.085 / +0.057) |
| fixed15 | n/a (0% by construction) | ✗ | −0.013 / −0.003 ✗ | +0.030 / +0.032 ✗ (CI lower −0.002 / −0.001) |
| fixed20 | ✗ (worse) | ✗ (worse) | **✓ worse** (−0.042 / −0.039) | ✓ worse (A −0.035) / ✗ (B) |

On the MIRU axis (agreement with the oracle stop) the policy is at or below chance for R50 (40–43% vs 50%) and at chance for relsg (50% / 57%). No comparison is significantly superior on the primary metric; the only significant superiority is relsg_gt vs the same-p mix and vs fixed10, on both seeds.

**Level-agnostic observation-based comparator `conf_two_stage`** (`scripts/49_recovery_conf_two_stage.py`, zero-call, no GT): at t10 continue to t20 iff the mean PGSG entity score of the top-8 cleaned-G entities is below τ_conf = dev-seed0 median 3.0702 (no ties; frozen in `frozen_conf_policy.json`). Phase D: 28/60 continue (p = .467), 0 failures.

| set | conf_two_stage: p / R50 / relsg / R50-oracle agreement | U − conf, R50 [CI], perm p | U − conf, relsg [CI], perm p |
|---|---|---|---|
| Phase D seedA | .467 / .1328 / .4618 / 45% | −.004 [−.029, +.022], .62 ✗ | +.023 [−.010, +.058], .09 ✗ |
| Phase D seedB | .467 / .1245 / .4687 / 38% | +.002 [−.016, +.021], .43 ✗ | +.014 [−.008, +.036], .11 ✗ |
| dev seed0 / 1 / 2 | .50 / — / — / 55–65% | −.005 / −.004 / +.005 ✗ | −.022 / −.021 / −.016 ✗ |

Against this simple level-agnostic confidence rule, the U policy shows no significant advantage on either metric in any seed set. The legacy `eq_fixed` adjacent mix in the dev/MIRU/held-out tables also uses the level index; it is kept (with its CI) only because it was the previous plan's registered comparison, and is labelled accordingly.

## 7. Synthesis

### 7.1 Answers to the three questions

1. **Held-out reproduction of late fusion: the pre-registered PRIMARY (hybrid mean fusion) did NOT replicate** (0/18 significant cells; MAE 3.35 vs image-only 9b 3.21). The secondary fuse_min was significantly better than every level-agnostic comparator on MAE, R50 and relsg (all-96 9/9), but it observes ≈2 more levels, sits on the fixed curve at equal observation, and benefits from 21/96 never-saturating images; its gain is "later stopping that matches a late oracle", not GT efficiency.
2. **Headroom for image-dependent allocation: small and partly noise.** GT-posterior oracles show large single-seed headroom (dev R50 O−F +0.085 at b=15; two-choice +0.047). Across seeds the two-choice R50 headroom is +0.016 / +0.038, positive in both directions and robust to tie order, concentrated in about 7/20 images with g > 0. relsg headroom does not transfer across seeds.
3. **Can v4 visibility decide t10 vs t20? Not on the primary metric.** The frozen U policy loses to the same-p mix for R50 on 60 new images in both seeds (−0.007, −0.011; CIs include 0), is below fixed15 and F, and agrees with the two-choice oracle no better than chance. The dev gate pass (+0.004 / +0.009) did not transfer.

### 7.2 What is and is not claimable

**Claimable (negative/analytical):** PGSG scoring matches the official evaluator; single-seed oracle headroom overstates adaptivity (about half of the two-choice R50 headroom is sampling noise); the earlier "≈0 vs fixed" results used a weak comparator and against F(b) no dev20 rule is ≥ 0; a pre-specified, frozen visibility-based two-stage rule did not improve R50 over image-independent allocation on 60 new images (2 seeds), and its oracle agreement is at chance.

**Not claimable:** any R50 improvement; relsg_gt improvement as a main result; superiority over fixed observation in general; generalisation of the negative result to "adaptive observation is impossible" (plan §10).

**Framing (plan §7).** U uses no relations and no co-occurrence. Even a positive result would be "object-visibility-based allocation of additional observation", not evidence for the co-occurrence hypothesis or for scene-graph-reasoning novelty.

### 7.3 Limitations

- **Sample sizes.** dev n = 20 (gate criteria sign-only, margins ≤ 0.01); test n = 60 (R50 CI half-width ≈ ±0.02, so effects below ≈0.02 cannot be resolved).
- **Discrete U.** K ≤ 8 boxes produce many ties (30% dev, 31% test) decided by the GT-independent hash; τ is the tied median; v4 rates blurry regions as "absent" at low levels, opposite to the intended definition (3.1% at t10).
- **Seeds.** Stage curves are single samples; dev seeds differ by up to ≈0.05 in mean q15. In Phase D, U came from seedA t10 only; seedB re-scores q at the same stops.
- **References.** F(b) and all oracles are GT-posterior references, not methods; O is integer-budgeted and optimistic; EGTR R@50 is recall over ≤ 30 stored relations.
- **Multiple comparisons.** Phase B evaluated 108 shuffle rows, 4 budgets × 4 sets × 2 metrics, 6 cross-seed directions; only the pre-specified gate quantities were used for decisions. Phase D has one primary comparison; relsg results are secondary.
- **Independence.** New-60 independence from PGSG training is protocol-based, not byte-verified. held-out96 ⊂ test100, whose PGSG curves had been viewed.
- **Phase A conditions.** First image-only 9b run failed under GPU contention; the re-run uses unchanged settings; the overlay adds `num_ctx 16384` (asserted output-neutral, unverified); decoding is unseeded at temperature 0.3, as registered; registration header said θ 0.70 for the hybrid primary whereas the method key uses θ_hybrid 0.5 (disclosed).

## 8. Artefact checklist (plan §8) and reproduction

| required | state | location / note |
|---|---|---|
| preflight.md | present (named `recovery_preflight.md`) | `rec_r0/` |
| frozen_config.json / model_hashes.json / data_manifest.json | present | `rec_r0/` |
| heldout_confirmatory.md | present, complete | `outputs/heldout_eval/heldout_registered_table.{csv,md}`, `heldout_significance.csv`, `figures/heldout_quality_vs_level.{png,pdf}` |
| budget_oracle.csv / fixed_mix.csv / shuffled_assignment.csv | present | plus `two_choice_oracle.csv`, `audit.md`, `diagnostics_manifest.json` |
| multiseed_gain.csv / cross_seed_oracle.csv | present | plus `seed_differences.csv`, `stage_metrics_seed{0,1,2}.csv`, `multiseed_manifest.json` |
| dev_two_stage.csv / gate_decision.json | present | plus `dev_two_stage_eval.csv`, `frozen_policy.json` |
| test_selection.json / test_policy.csv | present | plus `selected_images_test_rec.json`, `levels_manifest_test_rec.json` |
| test_summary.csv / seed_replication.csv | present | plus `phase_d_summary.json`, `method_comparison_miru_style.{md,csv}`, `stage_metrics_seed{A,B}.csv` |
| figures/ | present | Phase D figure (`rec_r0/figures/`), Phase A figure (`outputs/heldout_eval/figures/`) |
| usage.jsonl | present | 8 stages (calls, failures, median/wall times) |
| failures.jsonl | present, complete | Phase A incident + resolution; Phase D v4 HTTP-500 states (2) |
| REPORT.md | this document | complete |
| reproduce.md | present, finalized | all Phase A–D commands with actual counts/times |

**Reproduction.** Follow `rec_r0/reproduce.md`. Entry points: `scripts/43_recovery_preflight.py`, `scripts/43_robovis_recovery_diag.py`, `scripts/45_recovery_multiseed_metrics.py`, `python -m milestone1.two_stage_policy --mode freeze|evaluate|apply`, `scripts/44_recovery_test_selection.py`, `scripts/46_recovery_test_rec.py`, `scripts/47_recovery_phase_d_eval.py`, and Phase A `scripts/40` / `scripts/11` / `scripts/42` under `MILESTONE1_CONFIG_OVERLAY=configs/config_heldout.yaml`. Commits: `48b964b` (A–C), `1b565f8` (D). Earlier context: `outputs/robovis2027/REPORT.md`.
