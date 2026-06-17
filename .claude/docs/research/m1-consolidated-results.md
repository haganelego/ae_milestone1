# Milestone 1 — Consolidated Results (Reconstructed n=100, saturating subset)

> Snapshot after dataset reconstruction. Primary oracle = **TopK-core, coef 0.9 (`stop_c09`)**.
> All metrics computed on the final 100-image saturating subset (`stop_c09 ≤ 18`).
> Supersedes pre-reconstruction figures.

## 1. Executive summary

- The dataset was rebuilt to **exclude right-censored images** (`stop_c09 ≥ 19`, i.e. scene-graph semantics that do not saturate within the 20-level window) and refilled to 100 with newly-sampled images that *do* saturate.
- On this saturating subset, **structured-SGG stop predictors beat the trivial "always-19/20" baselines** — the headline best (`llm_sgg_4b_h5_k2`, MAE 3.02) is clearly below always-19 (MAE 4.54). This signal was masked before, when the censored images made the oracle partly degenerate.
- **k=2 (consecutive) variants dominate**: the top 4 methods are all k=2. 4b ≥ 9b for LLM-SGG (9b is overconfident → stops too early).
- The fundamental bottleneck remains **EGTR's relation-prediction ceiling**: at full resolution (L20) mean node-recall = 0.605 but mean triplet-recall = only 0.302.

## 2. Dataset reconstruction

| Item | Value |
|---|---|
| Selection | seed=42 shuffle of VG-test eligible (≥5 obj, ≥3 rel); deterministic, reproducible |
| Filter | exclude `stop_c09 ≥ 19` (right-censored / non-saturating within 20 levels) |
| Excluded from original 100 | 35 (29 at stop=20, 6 at stop=19) |
| Retained | 65 |
| New candidates evaluated (shuffle order continued) | 60 → 36 passed (`≤18`), first 35 adopted |
| **Final dataset** | **100 = 65 retained + 35 new, all `stop_c09 ≤ 18`** |

Paper framing: *"Evaluation is restricted to images whose scene-graph semantics saturate within the 20-level resolution window (`stop_c09 ≤ 18`); right-censored images are excluded."*

## 3. Oracle stop distributions (n=100)

| Oracle definition | mean | median | range |
|---|---|---|---|
| **TopK `stop_c09` (primary)** | **14.46** | 15 | 5–18 |
| TopK `stop_c08` | 13.13 | 14 | 5–18 |
| Absolute `oracle_0.4` | 15.28 | 16 | — |
| Absolute `oracle_0.5` | 17.21 | 20 | — |
| Relative `rel_oracle_0.8` | 15.21 | 15 | — |
| Relative `rel_oracle_0.9` | 17.31 | 18 | — |

Trivial references vs `stop_c09`: always-19 MAE = 4.54, always-20 MAE = 5.54.

## 4. Main result — method comparison vs primary oracle `stop_c09` (n=100, MAE ascending)

| Method | MeanStop | MAE | MedAE | Early | Late | Corr |
|--------|---------|-----|-------|-------|------|------|
| **llm_sgg_4b_h5_k2** | 15.66 | **3.02** | 2.0 | 0.31 | 0.59 | 0.174 |
| hybrid_4b_k2 | 15.85 | 3.07 | 2.0 | 0.30 | 0.63 | 0.184 |
| hybrid_9b_k2 | 15.41 | 3.07 | 2.0 | 0.30 | 0.61 | 0.222 |
| vlm_9b_single_k2 | 16.14 | 3.22 | 3.0 | 0.27 | 0.63 | 0.160 |
| vlm_9b_single_k1 | 14.07 | 3.33 | 3.0 | 0.50 | 0.41 | 0.089 |
| hybrid_4b_k1 | 11.99 | 4.09 | 3.5 | 0.65 | 0.26 | 0.144 |
| llm_sgg_9b_h5_k2 | 11.73 | 4.69 | 4.0 | 0.60 | 0.32 | 0.141 |
| llm_sgg_4b_h5_k1 | 10.67 | 5.07 | 4.0 | 0.72 | 0.21 | 0.019 |
| hybrid_9b_k1 | 10.31 | 5.07 | 4.5 | 0.76 | 0.19 | 0.130 |
| baseline (conf 0.4) | 12.15 | 5.31 | 5.0 | 0.65 | 0.30 | 0.036 |
| vlm_4b_single_k2 | 14.90 | 6.04 | 6.0 | 0.40 | 0.59 | 0.045 |
| llm_sgg_9b_h5_k1 | 8.10 | 6.72 | 7.0 | 0.87 | 0.09 | 0.145 |
| vlm_4b_single_k1 | 8.57 | 7.85 | 8.0 | 0.83 | 0.17 | -0.012 |

**Cross-oracle robustness** (best methods by MAE):
- `stop_c08`: vlm_9b_single_k1 (3.36), llm_sgg_4b_h5_k2 (3.53), hybrid_9b_k2 (3.56) — top group stable.
- Absolute `oracle_0.5` (`method_comparison_100.csv`): vlm_9b_single_k2 (3.37), hybrid_4b_k2 (3.64), llm_sgg_4b_h5_k2 (3.85).
- Relative `rel_oracle_0.8` (`method_comparison_relative_100.csv`): vlm_9b_single_k2 (3.29), hybrid_4b_k2 (3.46), vlm_9b_single_k1 (3.54).

→ Across all oracle definitions, the winners are consistently **k=2 hybrid / llm_sgg_4b / vlm_9b**; baseline and k=1 LLM-9b are consistently worst.

## 5. OFAT ablation (LLM-SGG, n=100) — mean stop level

| Config | MeanStop | Effect vs base(9b,h5,rel,conf) |
|---|---|---|
| 4b, h5 | 10.7 | smaller model → stops later (less overconfident) |
| **9b, h5 (base)** | **8.1** | — |
| 9b, h1 | 9.5 | shorter history → stops slightly later |
| 9b, h3 | 8.1 | ≈ base |
| 9b, rel=off | 7.3 | removing relation info → stops earlier |
| 9b, conf=off | 7.3 | removing confidence info → stops earlier |

→ 9b "saturates" judgments far earlier than the oracle (mean 8.1 vs 14.46 = severe early-stop). 4b is closer. Stripping relation/confidence context pushes the LLM to stop even earlier. (Figure: `ablation_axes.png`.)

## 6. Baseline threshold sweep (node-confidence, n=100) — mean stop level

| Threshold | mean | median |
|---|---|---|
| 0.3 | 4.53 | 4 |
| **0.4 (adopted)** | 12.15 | 11 |
| 0.5 | 19.27 | 20 |

→ 0.5+ collapses to "never stop" (degenerate, EGTR confidences rarely exceed it); 0.4 is the practical operating point.

## 7. EGTR ceiling / error analysis

- **Node-before-relation saturation**: at L20, mean node-recall = 0.605 vs mean triplet-recall = 0.302. Relations are the binding constraint.
- `outputs/analysis/saturation_gap.csv` (n=100): per-image node-90% vs triplet-90% levels and gap.
- `outputs/analysis/missing_at_stop.csv`: per-method missing/extra nodes & triplets at the stop level (oracle/baseline/hybrid).
- `outputs/analysis/{category,predicate}_recall_by_level.csv`: per-class recall curves (hardest classes never recalled → degenerate per-class oracle).

## 8. Key takeaways

1. **Filtering fixed the evaluation**: on the saturating subset, learned stop predictors beat trivial baselines — the core PoC claim now holds with a clean (non-censored) oracle.
2. **Consecutive-k=2 + 4b LLM-SGG / 9b-VLM** is the most reliable stop signal across every oracle definition.
3. **9b is overconfident** (stops ~6 levels before oracle); larger model ≠ better here.
4. **Relation prediction is the ceiling** (triplet-recall ~0.30 at full res); this caps how well *any* method can do and is the main threat to validity.

## 9. Artifact pointers

- Per-image oracle: `outputs/metrics/topk_oracle/oracle_topk_per_image.csv` (cols `stop_c08`,`stop_c09`)
- Method comparisons: `topk_oracle/method_comparison_topk_c0{8,9}.csv`, `method_comparison_100.csv`, `method_comparison_relative_100.csv`, `method_comparison_convergence_100.csv`
- Ablation: `outputs/metrics/llm_ablation_stop.csv`, fig `outputs/figures/ablation_axes.png`
- Baseline/oracle: `outputs/metrics/{baseline_stop,oracle_stop,relative_oracle_stop}.csv`
- Figures: `outputs/figures/{stop_distribution_all_100,saturation_rate_100,relative_method_comparison_mae_100}.png`
- Auto-generated report: `.claude/docs/research/m1-evaluation.md`
- Selection list: `data/processed/selected_images.json` (final 100)
