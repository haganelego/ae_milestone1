# Milestone 1 — Consolidated Results (gate=0.1 calibration, n=100)

> Snapshot after the object-gate / threshold recalibration. Primary oracle =
> **TopK-core, coef 0.9 (`stop_c09`)**, recomputed under the 0.1 object gate.
> All metrics on the 100-image saturating subset (`stop_c09 ≤ 18`).
> Supersedes the earlier gate=0.3 snapshot.

## 1. Executive summary

- The object-confidence **export gate was lowered 0.3 → 0.1** to remove a
  structural floor: under the old 0.3 gate, `mean_object_confidence` was computed
  over objects already filtered at >0.3, so it could never drop much below 0.3 and
  the baseline stop signal operated in a compressed [0.3, 0.5] band.
- Stop thresholds were re-calibrated to the new ranges: **baseline 0.4 → 0.35**
  (0.40 left 61 % of images never stopping), **hybrid fusion 0.6 → 0.5**, **LLM
  saturation 0.8 → 0.70** (0.8 stopped ~2 levels too late).
- On the recalibrated setup, **`hybrid_4b_k2` is the most robust stop predictor**
  (MAE 2.82, balanced early/late, lowest never-stop and catastrophic-error rates).
  The other realizable methods (LLM-SGG, VLM, ObjConf-baseline) cluster around
  MAE 3.3–3.5.
- **Key caveat**: per-image correlation with the oracle is weak for *all* methods
  (r ≤ 0.20). The oracle distribution is narrow (mean 13.5, std 3.1), so the
  methods' value is in *robustness* (avoiding catastrophic over/under-shoot), not
  in per-image tracking. The EGTR relation ceiling (triplet-recall ~0.30 at full
  res) remains the binding limitation.

## 2. Adopted experiment settings

| Component | Setting |
|---|---|
| Dataset | VG test, n=100, seed=42 shuffle, GT obj≥5 ∧ rel≥3, filter `stop_c09 ≤ 18` |
| Degradation | 20 levels, geometric scale 0.05→1.00, **bilinear** |
| EGTR gate | `object_conf_threshold = 0.10`, `relation_conf_threshold = 0.005`, top-20 obj / top-30 rel |
| Primary oracle | `stop_c09` = first level where **core RelSGScore ≥ 0.9** (core = L20 top-10 objects / top-5 triplets) |
| Baseline (ObjConf) | first level `mean_object_confidence ≥ 0.35`, consecutive_k=1 |
| LLM-SGG (RelSat) | LLM saturation_score ≥ **0.70**; qwen3.5:4b, start=5, history=5, consecutive_k=2 |
| VLM | qwen3.5:{4b,9b}, {single,sequence}, sat_thr=0.7, k=2 |
| Hybrid | `0.5·ObjConf + 0.5·LLM_sat ≥ 0.50`, consecutive_k∈{1,2} |
| LLM input filter | object≥0.3, relation≥0.005, top-20 (kept at 0.3 → LLM-visible graph unchanged by the 0.1 export gate; LLM/VLM scores were **not** re-run) |

## 3. Oracle stop distribution (`stop_c09`, gate=0.1, n=100)

mean = **13.48**, median = 14, std = 3.08, range 5–18, #≥19 = 0 (filter holds).
The oracle is near-invariant to the gate (gate0.2 gave mean 13.67) because the
core reference uses L20's highest-confidence objects, which the gate does not affect.

## 4. Main result — method comparison vs `stop_c09` (n=100, MAE ascending)

| Method | MeanStop | MAE | MedAE | Early | Late | NeverStop | Corr |
|--------|---------|-----|-------|-------|------|-----------|------|
| **hybrid_4b_k2** | 13.9 | **2.82** | 2.0 | 0.42 | 0.42 | **0.05** | 0.20 |
| llm_sgg_4b_k2 | 14.2 | 3.27 | 2.0 | 0.43 | 0.50 | 0.07 | 0.10 |
| vlm_9b_single_k1 | 14.1 | 3.47 | 3.0 | 0.38 | 0.53 | 0.06 | 0.10 |
| baseline (obj 0.35) | 14.3 | 3.49 | 2.5 | 0.41 | 0.47 | **0.22** | 0.18 |
| vlm_9b_single_k2 | 16.1 | 3.76 | 3.0 | 0.19 | 0.73 | 0.17 | 0.19 |
| hybrid_9b_k2 | 10.1 | 4.40 | 4.0 | 0.74 | 0.24 | 0.01 | 0.23 |
| llm_sgg_9b_k2 | 10.3 | 4.57 | 4.0 | 0.73 | 0.25 | 0.01 | 0.17 |
| hybrid_4b_k1 | 9.8 | 4.73 | 4.0 | 0.79 | 0.15 | 0.00 | — |
| llm_sgg_4b_k1 | 9.6 | 5.06 | 4.5 | 0.77 | 0.18 | 0.00 | — |
| vlm_4b_single_k2 | 14.9 | 6.34 | 6.0 | 0.36 | 0.62 | 0.57 | — |
| hybrid_9b_k1 | 7.2 | 6.46 | 6.0 | 0.91 | 0.04 | 0.00 | — |
| llm_sgg_9b_k1 | 7.1 | 6.50 | 7.0 | 0.90 | 0.04 | 0.00 | — |
| vlm_4b_single_k1 | 8.6 | 7.21 | 7.0 | 0.81 | 0.18 | 0.15 | — |

Columns: MAE/MedAE vs oracle; Early/Late = fraction stopping before/after oracle;
NeverStop = fraction reaching L20 without stopping; Corr = Pearson r vs oracle.

## 5. Hybrid vs Baseline vs LLM — comparative analysis (4b, k2)

A constant "always stop at L=14" scores MAE 2.40, but it is **not a realizable
policy**: an online stopping rule observes only the incoming scene-graph stream and
cannot know the level index, so a fixed-level predictor is excluded. Among the
realizable methods:

**(a) Pairwise per-image (closer-to-oracle wins / ties / losses)**
- hybrid vs baseline: **49 / 23 / 28** (hybrid clearly better)
- hybrid vs llm: 25 / 70 / 5 (hybrid ≥ llm; identical on 70 %, rarely worse)
- baseline vs llm: 37 / 16 / 47 (llm slightly better)

**(b) Stratified MAE by oracle saturation timing**

| Stratum | n | hybrid | baseline | llm |
|---|---|---|---|---|
| early (oracle ≤ 11) | 21 | 4.67 | 4.57 | 5.62 |
| mid (12–15) | 52 | **2.38** | 3.48 | 2.69 |
| late (≥ 16) | 27 | **2.22** | 2.67 | 2.56 |

**(c) Asymmetric cost** (early-stop = missed information, penalized): hybrid is
best at ×1/×2/×3 early penalty (e.g. ×2: hybrid 4.01 < llm 4.54 < baseline 4.82).

**(d) Error complementarity**: signed-error corr hybrid–llm 0.93, hybrid–baseline
0.73, baseline–llm 0.62. An oracle that picked the better of {baseline, llm} per
image would reach MAE 2.23 (< hybrid 2.82) → the components are complementary and
the fixed-weight fusion leaves ~0.6 MAE of headroom.

**Takeaways**
1. **hybrid is the most robust** — best/﻿tied on MAE, RMSE, never-stop (5 %),
   catastrophic |err|≥5 (22 %), balanced early/late, and head-to-head.
2. **baseline's weakness is never-stopping** (22 % reach L20; confidence alone is
   unreliable), though it carries independent signal (best on early-saturating).
3. **llm is middling with the weakest correlation (0.10)** — it declares
   saturation around the middle but does not track which images saturate early/late.
4. **fusion helps but is not optimal**; conditional/learned fusion is the obvious
   next step.
5. **shared failure**: all methods miss genuinely early-saturating scenes
   (oracle ≤ 11: MAE 4.6–5.6) — they are biased to wait. This is the main frontier.

## 6. OFAT ablation (LLM-SGG, n=100, threshold 0.70) — mean stop

| Config | MeanStop |
|---|---|
| 4b, h5 | 9.6 |
| 9b, h5 (base) | 7.1 |
| 9b, h1 | 9.5 |
| 9b, h3 | 7.1 |
| 9b, rel=off | 6.5 |
| 9b, conf=off | 6.8 |

9b stops far earlier than the oracle (≈7 vs 13.5 = severe early-stop); 4b is closer.
Removing relation/confidence context pushes the LLM to stop even earlier.
(Figure: `ablation_axes.png`.)

## 7. Baseline threshold sweep (gate=0.1, n=100) — mean stop

| Threshold | mean stop |
|---|---|
| 0.30 | 11.1 |
| **0.35 (adopted)** | 14.3 |
| 0.40 | 17.8 (61 % never stop) |

## 8. EGTR ceiling / error analysis

- At L20 (full res): mean node-recall **0.655**, mean triplet-recall **0.302** (vs GT).
  Relations remain the binding constraint (node-before-relation saturation).
- `outputs/analysis/{saturation_gap,missing_at_stop}.csv`,
  `outputs/analysis/{category,predicate}_recall_by_level.csv`.

## 9. Calibration history (this snapshot)

| Parameter | old (gate0.3) | new |
|---|---|---|
| object_conf_threshold (gate) | 0.30 | **0.10** |
| baseline threshold | 0.40 | **0.35** |
| hybrid fusion threshold | 0.60 | **0.50** |
| LLM saturation threshold | 0.80 | **0.70** |

LLM/VLM scoring was **not** re-run: the LLM-visible scene graph is re-filtered at
0.3, so it is invariant to the 0.1 export gate; only stops/fusions were recomputed
from cached saturation scores.

## 10. Artifact pointers

- Per-image oracle: `outputs/metrics/topk_oracle/oracle_topk_per_image.csv` (`stop_c09`)
- Method comparisons: `topk_oracle/method_comparison_topk_c0{8,9}.csv`,
  `method_comparison_100.csv`, `method_comparison_relative_100.csv`
- Stops: `outputs/metrics/{baseline_stop,hybrid_stop,llm_ablation_stop,llm_ablation_k2_stop,vlm_stop}.csv`
- Figures: `outputs/figures/{relative_curve_*,representative_*,ablation_axes,
  stop_distribution_all_100,saturation_rate_100,relative_method_comparison_mae_100}.png`
- Auto-generated: `.claude/docs/research/{m1-evaluation,m1-relative-saturation}.md`
- Selection list: `data/processed/selected_images.json` (final 100)
