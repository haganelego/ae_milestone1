# Milestone 1 — 100-image scaled evaluation

Scaled from 10 → 100 images (seeded superset; the 10-image set is a strict subset).
Focused method set (full OFAT ablation and VLM-sequence skipped this round).

## Saturation reference: degeneracy check (holds at scale)

| Reference | Saturated images | Mean stop |
|---|---|---|
| **True-GT Oracle (SGScore≥0.5)** | **28/100 (28%)** — still largely degenerate | 17.4 |
| **RelOracle (RelSGScore≥0.8, Plan B)** | **80/100 (80%)** — non-degenerate | 13.8 |

The absolute true-GT oracle remains degenerate at scale (72% never reach 0.5, bounded by
EGTR's recall ceiling). Plan B (self-reference to each image's own Level-20 output) stays
non-degenerate (80% saturate; 10-image was 90%). Confirms the methodological fix.

## Method comparison vs RelOracle(0.8), n=100 (sorted by MAE) — AFTER num_ctx fix

> **Important:** the first 100-image scoring had a high LLM JSON parse-failure rate
> (4b 26.3% / 9b 27.8%) caused by ollama's default `num_ctx=4096` truncating long
> prompts to 4095 tokens, leaving zero room to generate (1-token "Based" stub).
> Fixed with `num_ctx=12288`; the 830 failed LLM calls were re-scored (now ~1%, the
> residue being legitimate low-level 0.0 scores). Table below is the CORRECTED result.

| Method | MeanStop | MAE | MedAE | Early | Late | Corr |
|---|---|---|---|---|---|---|
| **hybrid_9b_k2** | 14.29 | **4.18** | 4.0 | 0.38 | 0.48 | 0.28 |
| **hybrid_4b_k2** | 14.58 | **4.35** | 3.0 | 0.39 | 0.51 | 0.18 |
| **llm_sgg_4b_h5_k2** | 13.18 | 4.47 | 3.0 | 0.51 | 0.40 | 0.11 |
| llm_sgg_9b_h5_k2 | 10.61 | 4.76 | 4.0 | 0.67 | 0.22 | 0.31 |
| hybrid_4b_k1 | 10.89 | 4.88 | 4.0 | 0.63 | 0.23 | 0.18 |
| vlm_9b_single_k2 | 11.45 | 4.96 | 4.5 | 0.62 | 0.31 | 0.05 |
| hybrid_9b_k1 | 9.97 | 5.30 | 4.0 | 0.72 | 0.22 | 0.25 |
| vlm_9b_single_k1 | 9.20 | 5.61 | 5.0 | 0.79 | 0.13 | 0.23 |
| vlm_4b_single_k2 | 12.56 | 5.67 | 5.0 | 0.58 | 0.33 | -0.05 |
| llm_sgg_4b_h5_k1 | 9.18 | 5.97 | 5.0 | 0.79 | 0.16 | 0.09 |
| llm_sgg_9b_h5_k1 | 7.37 | 6.90 | 6.0 | 0.87 | 0.10 | 0.21 |
| baseline | 11.59 | 7.46 | 7.0 | 0.61 | 0.33 | -0.08 |
| vlm_4b_single_k1 | 7.82 | 7.57 | 7.5 | 0.80 | 0.17 | -0.16 |

(vs the degenerate true-GT Oracle(0.5), the ranking inverts and `baseline` looks "best"
— an artifact of the unreachable oracle, see `method_comparison_100.csv`.)

## 10 → 100 conclusions: what held / what changed

- **Plan B non-degeneracy holds** (90%→80% saturate; far above the GT-oracle's 28%).
- **Research thesis restored after the LLM fix**: the top of the table is now **hybrid
  (structured-SGG + LLM + confidence) and LLM-SGG**, ahead of VLM (image-direct). The
  pre-fix "VLM is best" result was an artifact of the LLM context bug (27% of LLM calls
  failing → scored 0.0 → unfair handicap). Best now: `hybrid_9b_k2` (4.18), `hybrid_4b_k2`
  (4.35), `llm_sgg_4b_h5_k2` (4.47); best VLM `vlm_9b_single_k2` is 6th (4.96).
- **Hybrid (confidence + LLM) is the strongest family** — confidence anchors the LLM.
- **consecutive-k=2 consistently improves MAE** over k=1 across all method families.
- **node-confidence baseline stays poor** under the proper relative oracle (12th); best
  only under the degenerate GT oracle.
- **Bigger LLM still over-stops** at k=1 (9b mean 7.37 vs 4b 9.18); the gap is what k=2
  and the confidence anchor (hybrid) correct.
- **Node-vs-relation convergence gap small**: +0.5 levels, high per-image variance.

## Resolved bug

- **LLM JSON parse failures were an ollama `num_ctx` overflow, not a model/prompt issue.**
  Default `num_ctx=4096` truncated the ~6k–9.2k-token history-5/top-20-30 prompts, leaving
  no generation room. Fixed in `src/milestone1/{llm,llm_engine}.py` with `num_ctx=12288`
  (largest prompt ≈9.2k tokens + 1k generation). Parse failures 27% → ~1%.

## Remaining items

- **20% don't reach RelSGScore 0.8** even vs their own L20 — model keeps changing up to
  full res; consider threshold 0.7 or gain-based stop for the tail.

## Artifacts

- `outputs/metrics/method_comparison_100.csv` (vs true-GT Oracle 0.5)
- `outputs/metrics/method_comparison_relative_100.csv` (vs RelOracle 0.8)
- `outputs/metrics/{relative_metrics,relative_oracle_stop,all_metrics,oracle_stop,baseline_stop}.csv` (updated for 100)
- `outputs/figures/{relative_method_comparison_mae_100,stop_distribution_all_100,saturation_rate_100}.png`
- Scripts: `scripts/run_focused_100.py`, `scripts/aggregate_100.py`
