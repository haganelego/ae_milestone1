# TopK Oracle Evaluation Summary

- **Top-N nodes**: 10 (by confidence)
- **Top-N triplets**: 5 (by triplet_score)
- **Images**: 100
- **Mean ref node labels**: 6.2
- **Mean ref triplets**: 3.6

## coef=0.8

- Saturated: 100/100 (100%)
- Mean stop: 13.1, Median: 14
- always-19 MAE: 5.87
- always-20 MAE: 6.87

### Top methods (MAE ascending)

| Method | MeanStop | MAE | MedAE | Early | Late | Corr |
|--------|---------|-----|-------|-------|------|------|
| vlm_9b_single_k1 | 14.07 | 3.36 | 3.0 | 0.34 | 0.58 | 0.103 |
| llm_sgg_4b_h5_k2 | 15.66 | 3.53 | 3.0 | 0.19 | 0.73 | 0.213 |
| hybrid_9b_k2 | 15.41 | 3.56 | 3.0 | 0.24 | 0.71 | 0.269 |
| hybrid_4b_k1 | 11.99 | 3.6 | 3.0 | 0.52 | 0.36 | 0.092 |
| hybrid_4b_k2 | 15.85 | 3.62 | 3.0 | 0.17 | 0.76 | 0.237 |
| vlm_9b_single_k2 | 16.14 | 3.89 | 4.0 | 0.16 | 0.76 | 0.13 |
| hybrid_9b_k1 | 10.31 | 4.24 | 4.0 | 0.7 | 0.26 | 0.165 |
| llm_sgg_4b_h5_k1 | 10.67 | 4.32 | 4.0 | 0.63 | 0.27 | 0.012 |
| llm_sgg_9b_h5_k2 | 11.73 | 4.5 | 4.0 | 0.55 | 0.41 | 0.13 |
| baseline | 12.15 | 4.72 | 4.5 | 0.58 | 0.37 | 0.131 |
| llm_sgg_9b_h5_k1 | 8.1 | 5.81 | 6.0 | 0.84 | 0.13 | 0.078 |
| vlm_4b_single_k2 | 14.9 | 6.39 | 6.0 | 0.37 | 0.61 | 0.003 |
| vlm_4b_single_k1 | 8.57 | 6.98 | 7.0 | 0.81 | 0.18 | -0.013 |

## coef=0.9

- Saturated: 100/100 (100%)
- Mean stop: 14.5, Median: 15
- always-19 MAE: 4.54
- always-20 MAE: 5.54

### Top methods (MAE ascending)

| Method | MeanStop | MAE | MedAE | Early | Late | Corr |
|--------|---------|-----|-------|-------|------|------|
| llm_sgg_4b_h5_k2 | 15.66 | 3.02 | 2.0 | 0.31 | 0.59 | 0.174 |
| hybrid_4b_k2 | 15.85 | 3.07 | 2.0 | 0.3 | 0.63 | 0.184 |
| hybrid_9b_k2 | 15.41 | 3.07 | 2.0 | 0.3 | 0.61 | 0.222 |
| vlm_9b_single_k2 | 16.14 | 3.22 | 3.0 | 0.27 | 0.63 | 0.16 |
| vlm_9b_single_k1 | 14.07 | 3.33 | 3.0 | 0.5 | 0.41 | 0.089 |
| hybrid_4b_k1 | 11.99 | 4.09 | 3.5 | 0.65 | 0.26 | 0.144 |
| llm_sgg_9b_h5_k2 | 11.73 | 4.69 | 4.0 | 0.6 | 0.32 | 0.141 |
| llm_sgg_4b_h5_k1 | 10.67 | 5.07 | 4.0 | 0.72 | 0.21 | 0.019 |
| hybrid_9b_k1 | 10.31 | 5.07 | 4.5 | 0.76 | 0.19 | 0.13 |
| baseline | 12.15 | 5.31 | 5.0 | 0.65 | 0.3 | 0.036 |
| vlm_4b_single_k2 | 14.9 | 6.04 | 6.0 | 0.4 | 0.59 | 0.045 |
| llm_sgg_9b_h5_k1 | 8.1 | 6.72 | 7.0 | 0.87 | 0.09 | 0.145 |
| vlm_4b_single_k1 | 8.57 | 7.85 | 8.0 | 0.83 | 0.17 | -0.012 |
