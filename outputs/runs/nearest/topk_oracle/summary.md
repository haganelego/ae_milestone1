# TopK Oracle Evaluation Summary

- **Top-N nodes**: 10 (by confidence)
- **Top-N triplets**: 5 (by triplet_score)
- **Images**: 100
- **Mean ref node labels**: 5.8
- **Mean ref triplets**: 3.5

## coef=0.8

- Saturated: 71/100 (71%)
- Mean stop: 18.5, Median: 19
- always-19 MAE: 1.12
- always-20 MAE: 1.54

### Top methods (MAE ascending)

| Method | MeanStop | MAE | MedAE | Early | Late | Corr |
|--------|---------|-----|-------|-------|------|------|
| hybrid_4b_k2 | 19.35 | 1.43 | 1.0 | 0.08 | 0.63 | 0.416 |
| hybrid_9b_k2 | 18.3 | 2.26 | 1.0 | 0.19 | 0.53 | 0.064 |
| baseline | 16.69 | 2.59 | 2.0 | 0.57 | 0.23 | 0.232 |
| llm_sgg_4b_h5_k2 | 17.77 | 2.65 | 2.0 | 0.28 | 0.52 | 0.083 |
| hybrid_4b_k1 | 16.3 | 3.18 | 2.0 | 0.51 | 0.34 | 0.118 |
| hybrid_9b_k1 | 15.58 | 3.66 | 3.0 | 0.58 | 0.21 | 0.14 |
| vlm_9b_single_k2 | 12.84 | 5.68 | 6.0 | 0.97 | 0.01 | 0.254 |
| llm_sgg_4b_h5_k1 | 12.17 | 6.71 | 7.0 | 0.79 | 0.13 | 0.12 |
| vlm_9b_single_k1 | 10.74 | 7.72 | 8.0 | 1.0 | 0.0 | 0.187 |
| vlm_4b_single_k2 | 8.1 | 10.56 | 12.0 | 0.94 | 0.04 | -0.042 |
| llm_sgg_9b_h5_k2 | 6.25 | 12.21 | 13.0 | 1.0 | 0.0 | 0.113 |
| vlm_4b_single_k1 | 5.22 | 13.24 | 14.0 | 1.0 | 0.0 | 0.034 |
| llm_sgg_9b_h5_k1 | 5.08 | 13.38 | 14.0 | 1.0 | 0.0 | 0.107 |

## coef=0.9

- Saturated: 58/100 (58%)
- Mean stop: 18.8, Median: 19
- always-19 MAE: 1.0
- always-20 MAE: 1.16

### Top methods (MAE ascending)

| Method | MeanStop | MAE | MedAE | Early | Late | Corr |
|--------|---------|-----|-------|-------|------|------|
| hybrid_4b_k2 | 19.35 | 1.27 | 1.0 | 0.09 | 0.52 | 0.137 |
| hybrid_9b_k2 | 18.3 | 2.14 | 1.0 | 0.21 | 0.42 | -0.041 |
| llm_sgg_4b_h5_k2 | 17.77 | 2.57 | 1.0 | 0.31 | 0.43 | -0.03 |
| baseline | 16.69 | 2.79 | 2.0 | 0.59 | 0.17 | 0.064 |
| hybrid_4b_k1 | 16.3 | 3.24 | 2.0 | 0.54 | 0.23 | 0.028 |
| hybrid_9b_k1 | 15.58 | 3.92 | 2.5 | 0.61 | 0.18 | 0.007 |
| vlm_9b_single_k2 | 12.84 | 6.06 | 6.0 | 0.97 | 0.01 | 0.121 |
| llm_sgg_4b_h5_k1 | 12.17 | 6.97 | 7.0 | 0.81 | 0.09 | 0.085 |
| vlm_9b_single_k1 | 10.74 | 8.1 | 8.0 | 1.0 | 0.0 | 0.088 |
| vlm_4b_single_k2 | 8.1 | 10.86 | 12.0 | 0.95 | 0.03 | 0.053 |
| llm_sgg_9b_h5_k2 | 6.25 | 12.59 | 13.0 | 1.0 | 0.0 | 0.077 |
| vlm_4b_single_k1 | 5.22 | 13.62 | 14.0 | 1.0 | 0.0 | 0.007 |
| llm_sgg_9b_h5_k1 | 5.08 | 13.76 | 14.0 | 1.0 | 0.0 | 0.095 |
