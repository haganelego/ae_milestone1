# SGDet R@20 Oracle Evaluation Summary

## Image Counts

- Total selected: 100
- Excluded (0 GT triplets): 0
- Evaluated: 100
- Valid (ref > 0): 69
- Degenerate (ref == 0): 31

## Corpus-level R@20 (micro-average)

| Level | R@20 |
|-------|------|
| L01 | 0.0000 |
| L02 | 0.0000 |
| L03 | 0.0000 |
| L04 | 0.0000 |
| L05 | 0.0000 |
| L06 | 0.0000 |
| L07 | 0.0000 |
| L08 | 0.0000 |
| L09 | 0.0000 |
| L10 | 0.0000 |
| L11 | 0.0027 |
| L12 | 0.0053 |
| L13 | 0.0120 |
| L14 | 0.0213 |
| L15 | 0.0453 |
| L16 | 0.0719 |
| L17 | 0.0626 |
| L18 | 0.1092 |
| L19 | 0.1505 |
| L20 | 0.1864 |

## Oracle Stop Distributions (valid images only)

### Coefficient 0.8

- N valid: 69
- Mean stop level: 18.1
- Median stop level: 19
- Baseline always-L19 MAE: 1.59
- Baseline always-L20 MAE: 1.9

| Level | Count |
|-------|-------|
| 11 | 1 |
| 12 | 1 |
| 13 | 1 |
| 14 | 2 |
| 15 | 5 |
| 16 | 6 |
| 17 | 3 |
| 18 | 11 |
| 19 | 15 |
| 20 | 24 |

### Coefficient 0.9

- N valid: 69
- Mean stop level: 18.1
- Median stop level: 19
- Baseline always-L19 MAE: 1.59
- Baseline always-L20 MAE: 1.9

| Level | Count |
|-------|-------|
| 11 | 1 |
| 12 | 1 |
| 13 | 1 |
| 14 | 2 |
| 15 | 5 |
| 16 | 6 |
| 17 | 3 |
| 18 | 11 |
| 19 | 15 |
| 20 | 24 |

## GT Triplets Distribution

- Min: 3
- Median: 6.0
- Max: 21
- Count with < 5 triplets: 26
