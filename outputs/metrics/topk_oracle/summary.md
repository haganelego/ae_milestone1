# TopK Oracle Evaluation Summary

- **Top-N nodes**: 10 (by confidence)
- **Top-N triplets**: 5 (by triplet_score)
- **Images**: 100
- **Mean ref node labels**: 6.3
- **Mean ref triplets**: 3.6

## coef=0.8

- Saturated: 100/100 (100%)
- Mean stop: 12.2, Median: 13
- always-19 MAE: 6.81
- always-20 MAE: 7.81

### Top methods (MAE ascending)

| Method | MeanStop | MAE | MedAE | Early | Late | Corr |
|--------|---------|-----|-------|-------|------|------|
| hybrid_4b_k2 | 13.92 | 3.19 | 2.0 | 0.31 | 0.62 | 0.209 |
| llm_sgg_4b_h5_k2 | 14.21 | 3.48 | 3.0 | 0.3 | 0.64 | 0.155 |
| vlm_9b_single_k1 | 14.07 | 3.78 | 4.0 | 0.28 | 0.63 | 0.09 |
| baseline | 14.31 | 3.84 | 3.0 | 0.28 | 0.6 | 0.165 |
| hybrid_9b_k2 | 10.14 | 3.93 | 3.0 | 0.63 | 0.31 | 0.162 |
| llm_sgg_9b_h5_k2 | 10.31 | 4.06 | 3.0 | 0.61 | 0.31 | 0.133 |
| hybrid_4b_k1 | 9.85 | 4.16 | 4.0 | 0.66 | 0.27 | 0.021 |
| llm_sgg_4b_h5_k1 | 9.62 | 4.33 | 4.0 | 0.69 | 0.25 | 0.012 |
| vlm_9b_single_k2 | 16.14 | 4.65 | 4.0 | 0.12 | 0.84 | 0.126 |
| hybrid_9b_k1 | 7.2 | 5.43 | 5.0 | 0.85 | 0.09 | 0.073 |
| llm_sgg_9b_h5_k1 | 7.12 | 5.45 | 5.5 | 0.84 | 0.09 | 0.069 |
| vlm_4b_single_k1 | 8.57 | 6.42 | 7.0 | 0.77 | 0.19 | 0.01 |
| vlm_4b_single_k2 | 14.9 | 6.55 | 6.0 | 0.34 | 0.64 | 0.082 |

## coef=0.9

- Saturated: 100/100 (100%)
- Mean stop: 13.5, Median: 14
- always-19 MAE: 5.52
- always-20 MAE: 6.52

### Top methods (MAE ascending)

| Method | MeanStop | MAE | MedAE | Early | Late | Corr |
|--------|---------|-----|-------|-------|------|------|
| hybrid_4b_k2 | 13.92 | 2.82 | 2.0 | 0.42 | 0.42 | 0.195 |
| llm_sgg_4b_h5_k2 | 14.21 | 3.27 | 2.0 | 0.43 | 0.5 | 0.103 |
| vlm_9b_single_k1 | 14.07 | 3.47 | 3.0 | 0.38 | 0.53 | 0.104 |
| baseline | 14.31 | 3.49 | 2.5 | 0.41 | 0.47 | 0.185 |
| vlm_9b_single_k2 | 16.14 | 3.76 | 3.0 | 0.19 | 0.73 | 0.186 |
| hybrid_9b_k2 | 10.14 | 4.4 | 4.0 | 0.74 | 0.24 | 0.227 |
| llm_sgg_9b_h5_k2 | 10.31 | 4.57 | 4.0 | 0.73 | 0.25 | 0.171 |
| hybrid_4b_k1 | 9.85 | 4.73 | 4.0 | 0.79 | 0.15 | 0.066 |
| llm_sgg_4b_h5_k1 | 9.62 | 5.06 | 4.5 | 0.77 | 0.18 | -0.016 |
| vlm_4b_single_k2 | 14.9 | 6.34 | 6.0 | 0.36 | 0.62 | 0.034 |
| hybrid_9b_k1 | 7.2 | 6.46 | 6.0 | 0.91 | 0.04 | 0.136 |
| llm_sgg_9b_h5_k1 | 7.12 | 6.5 | 7.0 | 0.9 | 0.04 | 0.124 |
| vlm_4b_single_k1 | 8.57 | 7.21 | 7.0 | 0.81 | 0.18 | 0.018 |
