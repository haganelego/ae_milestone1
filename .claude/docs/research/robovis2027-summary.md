# ROBOVIS2027 line — consolidated experiment summary (as of 2026-10-08 01:30 JST)

Index of all experiments run since 2026-10-07 on branch `feature/robovis2027`, their results and where the evidence lives. Chronological detail: `robovis2027-worklog.md`. Plan of record: user's "ROBOVIS2027 実験計画書 v1.0".

## 0. Document map
| Topic | File |
|---|---|
| Chronological log + decisions | `.claude/docs/research/robovis2027-worklog.md` |
| Feasibility assessment of the plan | `.claude/docs/research/robovis2027-feasibility.md` |
| PGSG internals / how it runs | `.claude/docs/research/pgsg-pipeline.md`, `pgsg-run-log.md`, `pix2grp-vg-compat.md` |
| Decisions & plan deviations (for the paper) | `outputs/robovis2027/report_notes.md` |
| Auto-generated full dev report (16 variants) | `outputs/robovis2027/dev_r0/report.md` (regenerate: `scripts/36_robovis_report.py --run-id dev_r0 --paths ... --refresh --graphs-subdir auto`) |
| Path P not run — rationale | `outputs/robovis2027/dev_r0/pilot_report.md` |
| Hypothesis failure analysis (13 variants) | `outputs/robovis2027/dev_r0/failure_analysis/summary.md`, `failure_analysis_full20/summary.md` |
| PGSG recall profile on test100 | `outputs/robovis2027/test_r0/recall_profile.md` |
| PGSG-internal signals (negative control) | `outputs/robovis2027/test_r0/pgsg_signals_report.md` |
| Image-only resolution probe | `outputs/robovis2027/dev_r0/probe/*/summary.json`, `eval/probe/probe_compare.csv` |
| Scaled-down MIRU experiment write-up | `outputs/miru_gt_eval/report_scaled_down.md` |
| MIRU method table (GT-based secondary + fusion) | `outputs/miru_gt_eval/miru_method_table.md` |
| Why the image hurts (MIRU 2×2) | `outputs/llm_scores_vlm/diag_img_vs_text.md` |
| Environment / data / weights hashes | `outputs/robovis2027/preflight.json` |

## 1. Infrastructure built (all committed, 327 tests)
- PGSG (Pix2Grp CVPR2024) runs on local VG-150 in its own uv env (`envs/pgsg`), no submodule patches; per-image SGDet scorer matches the official evaluator exactly; wrapper saves per-state graphs (`scripts/25`, `28`).
- GT-only image re-selection (test 100 / dev 20), 384-px-base 20-level degradation (`scripts/26`, `27`).
- Full ROBOVIS pipeline: check generator (F1/F2), judge (text / image; prompt v1–v3; v4 grounded), stop policy §6, replay/eval/bootstrap/figures (`scripts/29–39`), report generator.
- Cleaned observed graph G (same-label IoU>0.5 merge, top-20 relations).
- MIRU evaluation registry + GT-based secondary for EGTR predictions, late fusion, held-out dataset path (`scripts/40–42`).

## 2. Experiments and results

### 2.1 ROBOVIS plan as specified (PGSG + Qwen3.5 checks/judge), dev20
| Variant | What changed | Stop signal? |
|---|---|---|
| F2 4b / 9b (text LLM, graph only) | plan §4 F2 | none (AUC 0.45–0.50) |
| F1 4b (VLM checks + text judge) | plan §4 F1 | none (AUC 0.60, n.s.) |
| F1v (judge also sees image), v2 prompts, G input, 4b | variants | none |
| F1v_v2_G__9b (9b checks + 9b image judge, G) | model size | first level-tracking signal (gate∧θ 0.50→0.15), but proposed ≡ observed_only |
| Judge v3 (lenient resolved + next-step gain), 9b | user-approved | resolved informative (AUC 0.64); stops still at 3–4; j9b +0.056 relsg vs matched fixed (1 of ~20 comparisons) |
| v3c (region-anchor check prompt) | — | no hypothesis-quality gain |
| Rule variants (benefit_gated, θ=1) | exploratory | do not recover signal |
| PGSG-internal self-consistency signals | zero-call | proxies for level only |
| Image-only probe (detail_fraction) | diagnostic | ordinal reading works (AUC 0.72), "sufficient" never true; 768 px no help |
| v4 grounded visibility (bbox overlays), h3 graph history | running | smoke: mean clear 0.62→1.23 |

20-level replays: every rule stops at mean t ≈ 2.4 (4b) / 4–5 (9b) and sits on the fixed-stop curve (no q_stop − eq_fixed CI excludes 0 except j9b relsg). Hypotheses H contribute ≈ 0 everywhere.

Root causes (failure analysis): hypotheses are graph bookkeeping or already-observed objects; genuinely missing GT objects named 0 % (raw) / 4–9 % (G) while 56 % of GT labels are absent at t1; judge answers resolution_benefit from text, not pixels; "resolved" essentially unused under v1/v2; absolute single-snapshot sufficiency judgment is calibration-bound.

### 2.2 Evaluation facts learned
- PGSG emits ~120 entities / 340 relations per state regardless of resolution; per-image R@50 is single-sample noisy; relative-stable oracle defined for 37/100 test images (R50) and degenerates to t=20 under relsg_gt; gain-based oracle (δ=0.05) defines all qmax>0 images with median 12–13.
- G halves hallucination (det_prec20 0.20→0.36) at −2.5 pt R@50.

### 2.3 Scaled-down MIRU experiment (EGTR, old 100 images, 5-level SGG history, θ 0.70, k=2)
| Condition | mean stop | MAE vs stop_c09 | relsg_gt(stop) |
|---|---|---|---|
| text 4b (re-run) | 13.76 | 3.00 | 0.404 |
| text 9b | 10.32 | 4.40 | 0.287 |
| image + text 4b | 9.34 | 5.04 | 0.301 |
| image + text 9b | 9.44 | 4.78 | 0.281 |
| image-only VLM 9b (legacy scores, θ 0.70) | 14.32 | 3.04 | 0.406 |
| hybrid_4b (MIRU published best) | 13.92 | 2.82 | 0.407 |
| late fusion mean(img9b, text4b) + hybrid (post hoc) | 13.49 | 2.59 | 0.397 |
| late fusion min (post hoc) | 16.16 | 3.12 | 0.455 |
| fixed_20 | 20 | 6.52 | 0.478 |

Findings: (a) text 4b reproduces MIRU; (b) early fusion (image in the SGG prompt) hurts via confabulated "graph stable" reasons (99 % of flipped states) and loss of the blur cue; (c) MIRU's published VLM baseline used θ 0.8 (stale) — at 0.70 it ties SGG-text 4b; (d) under GT scoring no method, including fusion, beats the equal-observation fixed stop (all CIs include 0).

### 2.4 Pre-registered held-out test (pending)
96 GT-only images (ROBOVIS test100 minus 4 overlap), MIRU conditions. Primary: `hybrid_fuse_mean(vlm image-only 9b, llm_sgg text 4b)`, θ 0.70, k=2; secondary `fuse_min`. Primary set = all 96; comparative = stop_c09 ≤ 18 subset (73). Distribution shift noted: 21/96 never saturate under c09 (MIRU 100 had been filtered to ≤ 18).

## 3. Status of the hypotheses
- H1 (unconfirmed co-occurrence candidates improve recall at equal observation): **not supported** in any variant.
- H2 (trade-off with over-observation): not testable (no improvement).
- "Adding the current image to the SGG judge improves stopping": **refuted** (hurts).
- "Adaptive stopping beats fixed stopping per observation under GT scoring": **not supported** for any method, incl. MIRU's.
- Only positive, in-sample number: late-fusion MAE 2.59 (post hoc) — awaiting held-out confirmation.

## 4. What is defensible for a paper (negative/analytical)
1. GT-referenced, equal-observation evaluation framework for resolution-aware stopping (fixed-stop curve + paired bootstrap), applied to EGTR and PGSG.
2. Finding: adaptive stopping methods (confidence, LLM-on-SGG, image VLM, hybrids, fusion) match oracle timing but do not exceed the fixed-stop curve in GT recall per observation.
3. Failure analysis: single-snapshot VLM sufficiency judgments are calibration-bound; early fusion of image + SGG induces confabulated stability; co-occurrence hypotheses from local VLMs rarely name genuinely missing objects.
4. Correction to MIRU: VLM-vs-LLM comparison was threshold-inconsistent; at equal θ they tie.

## 5. Addendum — recovery plan run `rec_r0` (2026-10-08)
Plan of record: `robovis2027-recovery-plan.md`. Full report: `outputs/robovis2027_recovery/rec_r0/REPORT.md` (Phase A pending at the time of writing). Key outcomes:
- Phase B (cache-only): single-seed omniscient headroom is large but ≈ half of the two-choice R50 headroom is sampling noise (cross-seed +0.016/+0.038 vs same-seed +0.05). Audit: earlier "≈0 vs fixed" comparisons used a weak adjacent mix; against the best fixed mixture F(b) no dev20 rule is ≥ 0 and the j9b "+0.056" CI vanishes.
- Phase C: gate GO on all four criteria (margins tiny: policy − mix R50 +0.004/+0.009 on dev seeds 1/2).
- Phase D (60 new images, frozen two-stage U policy from v4 visibility at t10): **R50 policy − same-p random mix = −0.007 [−0.025, +0.012] (seedA), −0.011 [−0.032, +0.010] (seedB) → not positive**; oracle agreement 40–43 % (chance 50 %). relsg_gt +0.023 on both seeds (significant) but secondary and reversed vs dev → not promotable.
- Comparator framing (user): fixed-level stops and F(b) are schedule-aware references, not comparators; fair comparators are same-protocol random assignment and level-agnostic observation-based baselines.
- Phase A (pre-registered late fusion on held-out 96, EGTR pipeline): pending the image-only 9b scores (first run failed with 0 files under VRAM contention; rerun unchanged).
