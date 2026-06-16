# TopK Oracle Evaluation Summary

- **Top-N nodes**: 10 (by confidence)
- **Top-N triplets**: 5 (by triplet_score)
- **Images**: 100
- **Mean ref node labels**: 5.8
- **Mean ref triplets**: 3.5

## coef=0.8

- Saturated: 80/100 (80%)
- Mean stop: 15.0, Median: 14
- always-19 MAE: 4.39
- always-20 MAE: 4.99

### Top methods (MAE ascending)

| Method | MeanStop | MAE | MedAE | Early | Late | Corr |
|--------|---------|-----|-------|-------|------|------|
| llm_sgg_4b_h5_k2 | 16.17 | 3.26 | 3.0 | 0.3 | 0.55 | 0.251 |
| hybrid_4b_k2 | 16.34 | 3.31 | 3.0 | 0.29 | 0.56 | 0.248 |
| vlm_9b_single_k2 | 14.41 | 3.34 | 3.0 | 0.52 | 0.41 | 0.248 |
| hybrid_9b_k2 | 15.08 | 4.07 | 3.5 | 0.4 | 0.52 | 0.076 |
| vlm_9b_single_k1 | 11.95 | 4.12 | 3.5 | 0.7 | 0.21 | 0.197 |
| hybrid_4b_k1 | 12.77 | 4.22 | 3.0 | 0.59 | 0.27 | 0.1 |
| baseline | 13.38 | 4.97 | 4.0 | 0.5 | 0.36 | 0.171 |
| hybrid_9b_k1 | 10.76 | 5.31 | 5.0 | 0.78 | 0.18 | 0.203 |
| llm_sgg_4b_h5_k1 | 9.95 | 5.84 | 6.0 | 0.79 | 0.14 | 0.012 |
| llm_sgg_9b_h5_k2 | 11.19 | 6.1 | 5.0 | 0.67 | 0.29 | -0.01 |
| vlm_4b_single_k2 | 12.98 | 6.23 | 6.0 | 0.53 | 0.38 | -0.17 |
| llm_sgg_9b_h5_k1 | 6.95 | 8.08 | 8.0 | 0.92 | 0.01 | 0.011 |
| vlm_4b_single_k1 | 7.0 | 8.47 | 9.0 | 0.91 | 0.06 | -0.09 |

## coef=0.9

- Saturated: 71/100 (71%)
- Mean stop: 16.4, Median: 17
- always-19 MAE: 3.22
- always-20 MAE: 3.64

### Top methods (MAE ascending)

| Method | MeanStop | MAE | MedAE | Early | Late | Corr |
|--------|---------|-----|-------|-------|------|------|
| llm_sgg_4b_h5_k2 | 16.17 | 2.99 | 2.0 | 0.44 | 0.4 | 0.227 |
| hybrid_4b_k2 | 16.34 | 3.06 | 2.0 | 0.42 | 0.43 | 0.193 |
| vlm_9b_single_k2 | 14.41 | 3.63 | 3.0 | 0.64 | 0.26 | 0.176 |
| hybrid_9b_k2 | 15.08 | 3.84 | 3.0 | 0.47 | 0.4 | 0.071 |
| hybrid_4b_k1 | 12.77 | 4.91 | 4.0 | 0.69 | 0.19 | 0.079 |
| vlm_9b_single_k1 | 11.95 | 5.03 | 4.5 | 0.82 | 0.13 | 0.12 |
| baseline | 13.38 | 5.4 | 5.0 | 0.58 | 0.26 | 0.133 |
| vlm_4b_single_k2 | 12.98 | 6.22 | 5.0 | 0.61 | 0.32 | -0.076 |
| hybrid_9b_k1 | 10.76 | 6.26 | 6.0 | 0.82 | 0.13 | 0.173 |
| llm_sgg_9b_h5_k2 | 11.19 | 6.33 | 6.0 | 0.72 | 0.18 | 0.089 |
| llm_sgg_4b_h5_k1 | 9.95 | 6.93 | 7.0 | 0.84 | 0.09 | -0.065 |
| llm_sgg_9b_h5_k1 | 6.95 | 9.43 | 10.0 | 0.96 | 0.01 | 0.034 |
| vlm_4b_single_k1 | 7.0 | 9.66 | 10.0 | 0.94 | 0.04 | -0.115 |
