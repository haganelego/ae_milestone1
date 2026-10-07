# MIRU-style method comparison (presentation; plan primary = R50 policy − same-p mix)

Oracle = GT two-choice oracle on that seed (top-k by g = q20 − q10, k = the policy's number of t20 slots); a GT-posterior reference, not a method. With stops in {10, 20}, MAE = 10 × (1 − agreement); fixed15 has MAE 5 and agreement 0 by construction. ✓ = 95 % paired-bootstrap CI excludes 0 (or, vs the mix, permutation p < .05); the direction says whether the U policy is better or worse.

## Phase D seedA (n = 60)

**Headline (plan primary): R50 policy − same-p mix = -0.0071 [-0.0246, +0.0121], permutation p(better) = 0.767.**

| method | mean level | MAE vs R50-oracle | agree (R50-oracle) | R50 | MAE vs relsg-oracle | agree (relsg-oracle) | relsg_gt |
|---|---|---|---|---|---|---|---|
| U policy | 15.17 | 6.00 | 40% | 0.1290 | 5.00 | 50% | 0.4852 |
| mix (same p) | 15.17 | 4.99 | 50% | 0.1361 (perm95 [0.1166, 0.1550]) | 4.99 | 50% | 0.4622 (perm95 [0.4364, 0.4871]) |
| fixed10 | 10.00 | 5.17 | 48% | 0.0990 | 5.17 | 48% | 0.4006 |
| fixed15 | 15.00 | 5.00 | 0% | 0.1419 | 5.00 | 0% | 0.4555 |
| fixed20 | 20.00 | 4.83 | 52% | 0.1709 | 4.83 | 52% | 0.5198 |
| GT two-choice oracle (reference) | 15.17 | 0.00 | 100% | 0.1846 | 0.00 | 100% | 0.5298 |

| U policy vs | column | Δ (policy − comparator) [95% CI] | perm p (better / worse) | sig |
|---|---|---|---|---|
| mix | MAE vs oracle stop (levels) (R50-oracle) | +1.0056 [-0.2333, +2.1667] | 0.968 / 0.090 | ✗ (worse) |
| mix | agreement with oracle (R50-oracle) | -0.1006 [-0.2167, +0.0233] | 0.968 / 0.090 | ✗ (worse) |
| mix | q(stop) (R50) | -0.0071 [-0.0246, +0.0121] | 0.767 / 0.233 | ✗ (worse) |
| fixed10 | MAE vs oracle stop (levels) (R50-oracle) | +0.8333 [-1.0000, +2.6667] | — | ✗ (worse) |
| fixed10 | agreement with oracle (R50-oracle) | -0.0833 [-0.2667, +0.1000] | — | ✗ (worse) |
| fixed10 | q(stop) (R50) | +0.0300 [+0.0032, +0.0628] | — | ✓ better |
| fixed15 | MAE vs oracle stop (levels) (R50-oracle) | +1.0000 [-0.1667, +2.1667] | — | ✗ (worse) |
| fixed15 | agreement with oracle (R50-oracle) | +0.4000 [+0.2833, +0.5167] | — | n/a (0 % by construction) |
| fixed15 | q(stop) (R50) | -0.0129 [-0.0464, +0.0229] | — | ✗ (worse) |
| fixed20 | MAE vs oracle stop (levels) (R50-oracle) | +1.1667 [-0.5000, +3.0000] | — | ✗ (worse) |
| fixed20 | agreement with oracle (R50-oracle) | -0.1167 [-0.3000, +0.0500] | — | ✗ (worse) |
| fixed20 | q(stop) (R50) | -0.0419 [-0.0679, -0.0169] | — | ✓ worse |
| mix | q(stop) (relsg_gt) | +0.0230 [+0.0003, +0.0468] | 0.038 / 0.962 | ✓ better |
| fixed10 | q(stop) (relsg_gt) | +0.0846 [+0.0460, +0.1316] | — | ✓ better |
| fixed15 | q(stop) (relsg_gt) | +0.0297 [-0.0023, +0.0647] | — | ✗ (better) |
| fixed20 | q(stop) (relsg_gt) | -0.0346 [-0.0633, -0.0043] | — | ✓ worse |
| mix | MAE vs oracle stop (levels) (relsg_gt-oracle) | +0.0056 [-1.2724, +1.1946] | 0.612 / 0.589 | ✗ (worse) |
| mix | agreement with oracle (relsg_gt-oracle) | -0.0006 [-0.1195, +0.1272] | 0.612 / 0.589 | ✗ (worse) |
| fixed10 | MAE vs oracle stop (levels) (relsg_gt-oracle) | -0.1667 [-2.0000, +1.6667] | — | ✗ (better) |
| fixed10 | agreement with oracle (relsg_gt-oracle) | +0.0167 [-0.1667, +0.2000] | — | ✗ (better) |
| fixed15 | MAE vs oracle stop (levels) (relsg_gt-oracle) | +0.0000 [-1.3333, +1.1667] | — | ✗ (equal) |
| fixed15 | agreement with oracle (relsg_gt-oracle) | +0.5000 [+0.3833, +0.6333] | — | n/a (0 % by construction) |
| fixed20 | MAE vs oracle stop (levels) (relsg_gt-oracle) | +0.1667 [-1.6667, +2.0000] | — | ✗ (worse) |
| fixed20 | agreement with oracle (relsg_gt-oracle) | -0.0167 [-0.2000, +0.1667] | — | ✗ (worse) |

## Phase D seedB (replication) (n = 60)

**Headline (plan primary): R50 policy − same-p mix = -0.0107 [-0.0323, +0.0101], permutation p(better) = 0.820.**

| method | mean level | MAE vs R50-oracle | agree (R50-oracle) | R50 | MAE vs relsg-oracle | agree (relsg-oracle) | relsg_gt |
|---|---|---|---|---|---|---|---|
| U policy | 15.17 | 5.67 | 43% | 0.1265 | 4.33 | 57% | 0.4822 |
| mix (same p) | 15.17 | 4.99 | 50% | 0.1372 (perm95 [0.1161, 0.1589]) | 4.99 | 50% | 0.4596 (perm95 [0.4385, 0.4799]) |
| fixed10 | 10.00 | 5.17 | 48% | 0.1067 | 5.17 | 48% | 0.4253 |
| fixed15 | 15.00 | 5.00 | 0% | 0.1294 | 5.00 | 0% | 0.4501 |
| fixed20 | 20.00 | 4.83 | 52% | 0.1658 | 4.83 | 52% | 0.4917 |
| GT two-choice oracle (reference) | 15.17 | 0.00 | 100% | 0.1822 | 0.00 | 100% | 0.5174 |

| U policy vs | column | Δ (policy − comparator) [95% CI] | perm p (better / worse) | sig |
|---|---|---|---|---|
| mix | MAE vs oracle stop (levels) (R50-oracle) | +0.6722 [-0.6224, +1.8889] | 0.911 / 0.207 | ✗ (worse) |
| mix | agreement with oracle (R50-oracle) | -0.0672 [-0.1889, +0.0622] | 0.911 / 0.207 | ✗ (worse) |
| mix | q(stop) (R50) | -0.0107 [-0.0323, +0.0101] | 0.820 / 0.180 | ✗ (worse) |
| fixed10 | MAE vs oracle stop (levels) (R50-oracle) | +0.5000 [-1.3333, +2.3333] | — | ✗ (worse) |
| fixed10 | agreement with oracle (R50-oracle) | -0.0500 [-0.2333, +0.1333] | — | ✗ (worse) |
| fixed10 | q(stop) (R50) | +0.0199 [-0.0073, +0.0520] | — | ✗ (better) |
| fixed15 | MAE vs oracle stop (levels) (R50-oracle) | +0.6667 [-0.6667, +1.8375] | — | ✗ (worse) |
| fixed15 | agreement with oracle (R50-oracle) | +0.4333 [+0.3163, +0.5667] | — | n/a (0 % by construction) |
| fixed15 | q(stop) (R50) | -0.0029 [-0.0334, +0.0309] | — | ✗ (worse) |
| fixed20 | MAE vs oracle stop (levels) (R50-oracle) | +0.8333 [-0.8333, +2.6667] | — | ✗ (worse) |
| fixed20 | agreement with oracle (R50-oracle) | -0.0833 [-0.2667, +0.0833] | — | ✗ (worse) |
| fixed20 | q(stop) (R50) | -0.0392 [-0.0755, -0.0105] | — | ✓ worse |
| mix | q(stop) (relsg_gt) | +0.0226 [+0.0033, +0.0424] | 0.017 / 0.984 | ✓ better |
| fixed10 | q(stop) (relsg_gt) | +0.0569 [+0.0239, +0.0954] | — | ✓ better |
| fixed15 | q(stop) (relsg_gt) | +0.0320 [-0.0008, +0.0710] | — | ✗ (better) |
| fixed20 | q(stop) (relsg_gt) | -0.0095 [-0.0315, +0.0124] | — | ✗ (worse) |
| mix | MAE vs oracle stop (levels) (relsg_gt-oracle) | -0.6611 [-1.8667, +0.5336] | 0.221 / 0.913 | ✗ (better) |
| mix | agreement with oracle (relsg_gt-oracle) | +0.0661 [-0.0534, +0.1867] | 0.221 / 0.913 | ✗ (better) |
| fixed10 | MAE vs oracle stop (levels) (relsg_gt-oracle) | -0.8333 [-2.6667, +1.0000] | — | ✗ (better) |
| fixed10 | agreement with oracle (relsg_gt-oracle) | +0.0833 [-0.1000, +0.2667] | — | ✗ (better) |
| fixed15 | MAE vs oracle stop (levels) (relsg_gt-oracle) | -0.6667 [-1.8333, +0.5000] | — | ✗ (better) |
| fixed15 | agreement with oracle (relsg_gt-oracle) | +0.5667 [+0.4500, +0.6833] | — | n/a (0 % by construction) |
| fixed20 | MAE vs oracle stop (levels) (relsg_gt-oracle) | -0.5000 [-2.3333, +1.3333] | — | ✗ (better) |
| fixed20 | agreement with oracle (relsg_gt-oracle) | +0.0500 [-0.1333, +0.2333] | — | ✗ (better) |

## dev seed0 (n = 20)

**Headline (plan primary): R50 policy − same-p mix = -0.0100 [-0.0414, +0.0244], permutation p(better) = 0.736.**

| method | mean level | MAE vs R50-oracle | agree (R50-oracle) | R50 | MAE vs relsg-oracle | agree (relsg-oracle) | relsg_gt |
|---|---|---|---|---|---|---|---|
| U policy | 15.50 | 5.00 | 50% | 0.1582 | 6.00 | 40% | 0.5010 |
| mix (same p) | 15.50 | 4.95 | 50% | 0.1682 (perm95 [0.1339, 0.2026]) | 4.95 | 50% | 0.5331 (perm95 [0.5009, 0.5634]) |
| fixed10 | 10.00 | 5.50 | 45% | 0.1343 | 5.50 | 45% | 0.4565 |
| fixed15 | 15.00 | 5.00 | 0% | 0.2008 | 5.00 | 0% | 0.5381 |
| fixed20 | 20.00 | 4.50 | 55% | 0.1959 | 4.50 | 55% | 0.5957 |
| GT two-choice oracle (reference) | 15.50 | 0.00 | 100% | 0.2126 | 0.00 | 100% | 0.5922 |

| U policy vs | column | Δ (policy − comparator) [95% CI] | perm p (better / worse) | sig |
|---|---|---|---|---|
| mix | MAE vs oracle stop (levels) (R50-oracle) | +0.0500 [-2.0000, +2.2000] | 0.688 / 0.663 | ✗ (worse) |
| mix | agreement with oracle (R50-oracle) | -0.0050 [-0.2200, +0.2000] | 0.688 / 0.663 | ✗ (worse) |
| mix | q(stop) (R50) | -0.0100 [-0.0414, +0.0244] | 0.736 / 0.275 | ✗ (worse) |
| fixed10 | MAE vs oracle stop (levels) (R50-oracle) | -0.5000 [-3.5000, +3.0000] | — | ✗ (better) |
| fixed10 | agreement with oracle (R50-oracle) | +0.0500 [-0.3000, +0.3500] | — | ✗ (better) |
| fixed10 | q(stop) (R50) | +0.0239 [-0.0333, +0.0868] | — | ✗ (better) |
| fixed15 | MAE vs oracle stop (levels) (R50-oracle) | +0.0000 [-2.0000, +2.0125] | — | ✗ (equal) |
| fixed15 | agreement with oracle (R50-oracle) | +0.5000 [+0.2988, +0.7000] | — | n/a (0 % by construction) |
| fixed15 | q(stop) (R50) | -0.0426 [-0.0978, +0.0028] | — | ✗ (worse) |
| fixed20 | MAE vs oracle stop (levels) (R50-oracle) | +0.5000 [-2.5000, +3.5000] | — | ✗ (worse) |
| fixed20 | agreement with oracle (R50-oracle) | -0.0500 [-0.3500, +0.2500] | — | ✗ (worse) |
| fixed20 | q(stop) (R50) | -0.0378 [-0.0822, +0.0000] | — | ✗ (worse) |
| mix | q(stop) (relsg_gt) | -0.0321 [-0.0623, -0.0010] | 0.975 / 0.025 | ✓ worse |
| fixed10 | q(stop) (relsg_gt) | +0.0444 [+0.0077, +0.0865] | — | ✓ better |
| fixed15 | q(stop) (relsg_gt) | -0.0371 [-0.0894, +0.0093] | — | ✗ (worse) |
| fixed20 | q(stop) (relsg_gt) | -0.0947 [-0.1663, -0.0378] | — | ✓ worse |
| mix | MAE vs oracle stop (levels) (relsg_gt-oracle) | +1.0500 [-1.0500, +3.0000] | 0.919 / 0.305 | ✗ (worse) |
| mix | agreement with oracle (relsg_gt-oracle) | -0.1050 [-0.3000, +0.1050] | 0.919 / 0.305 | ✗ (worse) |
| fixed10 | MAE vs oracle stop (levels) (relsg_gt-oracle) | +0.5000 [-3.0000, +4.0000] | — | ✗ (worse) |
| fixed10 | agreement with oracle (relsg_gt-oracle) | -0.0500 [-0.4000, +0.3000] | — | ✗ (worse) |
| fixed15 | MAE vs oracle stop (levels) (relsg_gt-oracle) | +1.0000 [-1.0000, +3.0000] | — | ✗ (worse) |
| fixed15 | agreement with oracle (relsg_gt-oracle) | +0.4000 [+0.2000, +0.6000] | — | n/a (0 % by construction) |
| fixed20 | MAE vs oracle stop (levels) (relsg_gt-oracle) | +1.5000 [-1.5000, +4.5000] | — | ✗ (worse) |
| fixed20 | agreement with oracle (relsg_gt-oracle) | -0.1500 [-0.4500, +0.1500] | — | ✗ (worse) |

## dev seed1 (n = 20)

**Headline (plan primary): R50 policy − same-p mix = +0.0038 [-0.0266, +0.0349], permutation p(better) = 0.424.**

| method | mean level | MAE vs R50-oracle | agree (R50-oracle) | R50 | MAE vs relsg-oracle | agree (relsg-oracle) | relsg_gt |
|---|---|---|---|---|---|---|---|
| U policy | 15.50 | 6.00 | 40% | 0.1798 | 5.00 | 50% | 0.5057 |
| mix (same p) | 15.50 | 4.95 | 50% | 0.1761 (perm95 [0.1437, 0.2071]) | 4.95 | 50% | 0.5228 (perm95 [0.4851, 0.5569]) |
| fixed10 | 10.00 | 5.50 | 45% | 0.1382 | 5.50 | 45% | 0.4670 |
| fixed15 | 15.00 | 5.00 | 0% | 0.1741 | 5.00 | 0% | 0.5269 |
| fixed20 | 20.00 | 4.50 | 55% | 0.2071 | 4.50 | 55% | 0.5685 |
| GT two-choice oracle (reference) | 15.50 | 0.00 | 100% | 0.2221 | 0.00 | 100% | 0.5869 |

| U policy vs | column | Δ (policy − comparator) [95% CI] | perm p (better / worse) | sig |
|---|---|---|---|---|
| mix | MAE vs oracle stop (levels) (R50-oracle) | +1.0500 [-1.0500, +3.0000] | 0.917 / 0.312 | ✗ (worse) |
| mix | agreement with oracle (R50-oracle) | -0.1050 [-0.3000, +0.1050] | 0.917 / 0.312 | ✗ (worse) |
| mix | q(stop) (R50) | +0.0038 [-0.0266, +0.0349] | 0.424 / 0.582 | ✗ (better) |
| fixed10 | MAE vs oracle stop (levels) (R50-oracle) | +0.5000 [-3.0000, +3.5000] | — | ✗ (worse) |
| fixed10 | agreement with oracle (R50-oracle) | -0.0500 [-0.3500, +0.3000] | — | ✗ (worse) |
| fixed10 | q(stop) (R50) | +0.0416 [-0.0078, +0.1022] | — | ✗ (better) |
| fixed15 | MAE vs oracle stop (levels) (R50-oracle) | +1.0000 [-1.0000, +3.0000] | — | ✗ (worse) |
| fixed15 | agreement with oracle (R50-oracle) | +0.4000 [+0.2000, +0.6000] | — | n/a (0 % by construction) |
| fixed15 | q(stop) (R50) | +0.0057 [-0.0425, +0.0508] | — | ✗ (better) |
| fixed20 | MAE vs oracle stop (levels) (R50-oracle) | +1.5000 [-1.5000, +4.5000] | — | ✗ (worse) |
| fixed20 | agreement with oracle (R50-oracle) | -0.1500 [-0.4500, +0.1500] | — | ✗ (worse) |
| fixed20 | q(stop) (R50) | -0.0272 [-0.0700, +0.0050] | — | ✗ (worse) |
| mix | q(stop) (relsg_gt) | -0.0172 [-0.0522, +0.0182] | 0.807 / 0.193 | ✗ (worse) |
| fixed10 | q(stop) (relsg_gt) | +0.0386 [-0.0075, +0.0895] | — | ✗ (better) |
| fixed15 | q(stop) (relsg_gt) | -0.0212 [-0.0904, +0.0417] | — | ✗ (worse) |
| fixed20 | q(stop) (relsg_gt) | -0.0628 [-0.1278, -0.0076] | — | ✓ worse |
| mix | MAE vs oracle stop (levels) (relsg_gt-oracle) | +0.0500 [-2.0500, +2.0525] | 0.681 / 0.659 | ✗ (worse) |
| mix | agreement with oracle (relsg_gt-oracle) | -0.0050 [-0.2052, +0.2050] | 0.681 / 0.659 | ✗ (worse) |
| fixed10 | MAE vs oracle stop (levels) (relsg_gt-oracle) | -0.5000 [-3.5000, +2.5000] | — | ✗ (better) |
| fixed10 | agreement with oracle (relsg_gt-oracle) | +0.0500 [-0.2500, +0.3500] | — | ✗ (better) |
| fixed15 | MAE vs oracle stop (levels) (relsg_gt-oracle) | +0.0000 [-2.0000, +2.0000] | — | ✗ (equal) |
| fixed15 | agreement with oracle (relsg_gt-oracle) | +0.5000 [+0.3000, +0.7000] | — | n/a (0 % by construction) |
| fixed20 | MAE vs oracle stop (levels) (relsg_gt-oracle) | +0.5000 [-2.5000, +3.5000] | — | ✗ (worse) |
| fixed20 | agreement with oracle (relsg_gt-oracle) | -0.0500 [-0.3500, +0.2500] | — | ✗ (worse) |

## dev seed2 (n = 20)

**Headline (plan primary): R50 policy − same-p mix = +0.0092 [-0.0240, +0.0412], permutation p(better) = 0.318.**

| method | mean level | MAE vs R50-oracle | agree (R50-oracle) | R50 | MAE vs relsg-oracle | agree (relsg-oracle) | relsg_gt |
|---|---|---|---|---|---|---|---|
| U policy | 15.50 | 5.00 | 50% | 0.1969 | 7.00 | 30% | 0.4972 |
| mix (same p) | 15.50 | 4.95 | 50% | 0.1877 (perm95 [0.1543, 0.2232]) | 4.95 | 50% | 0.5232 (perm95 [0.4916, 0.5557]) |
| fixed10 | 10.00 | 5.50 | 45% | 0.1500 | 5.50 | 45% | 0.4517 |
| fixed15 | 15.00 | 5.00 | 0% | 0.1496 | 5.00 | 0% | 0.5163 |
| fixed20 | 20.00 | 4.50 | 55% | 0.2186 | 4.50 | 55% | 0.5817 |
| GT two-choice oracle (reference) | 15.50 | 0.00 | 100% | 0.2398 | 0.00 | 100% | 0.5842 |

| U policy vs | column | Δ (policy − comparator) [95% CI] | perm p (better / worse) | sig |
|---|---|---|---|---|
| mix | MAE vs oracle stop (levels) (R50-oracle) | +0.0500 [-2.0000, +2.1500] | 0.698 / 0.647 | ✗ (worse) |
| mix | agreement with oracle (R50-oracle) | -0.0050 [-0.2150, +0.2000] | 0.698 / 0.647 | ✗ (worse) |
| mix | q(stop) (R50) | +0.0092 [-0.0240, +0.0412] | 0.318 / 0.691 | ✗ (better) |
| fixed10 | MAE vs oracle stop (levels) (R50-oracle) | -0.5000 [-3.5000, +3.0000] | — | ✗ (better) |
| fixed10 | agreement with oracle (R50-oracle) | +0.0500 [-0.3000, +0.3500] | — | ✗ (better) |
| fixed10 | q(stop) (R50) | +0.0469 [-0.0096, +0.1105] | — | ✗ (better) |
| fixed15 | MAE vs oracle stop (levels) (R50-oracle) | +0.0000 [-2.0000, +2.0000] | — | ✗ (equal) |
| fixed15 | agreement with oracle (R50-oracle) | +0.5000 [+0.3000, +0.7000] | — | n/a (0 % by construction) |
| fixed15 | q(stop) (R50) | +0.0473 [+0.0027, +0.0998] | — | ✓ better |
| fixed20 | MAE vs oracle stop (levels) (R50-oracle) | +0.5000 [-2.5000, +3.5000] | — | ✗ (worse) |
| fixed20 | agreement with oracle (R50-oracle) | -0.0500 [-0.3500, +0.2500] | — | ✗ (worse) |
| fixed20 | q(stop) (R50) | -0.0217 [-0.0617, +0.0100] | — | ✗ (worse) |
| mix | q(stop) (relsg_gt) | -0.0260 [-0.0533, +0.0032] | 0.942 / 0.059 | ✗ (worse) |
| fixed10 | q(stop) (relsg_gt) | +0.0455 [+0.0008, +0.0947] | — | ✓ better |
| fixed15 | q(stop) (relsg_gt) | -0.0190 [-0.0905, +0.0508] | — | ✗ (worse) |
| fixed20 | q(stop) (relsg_gt) | -0.0845 [-0.1442, -0.0344] | — | ✓ worse |
| mix | MAE vs oracle stop (levels) (relsg_gt-oracle) | +2.0500 [-0.0500, +3.8000] | 0.991 / 0.073 | ✗ (worse) |
| mix | agreement with oracle (relsg_gt-oracle) | -0.2050 [-0.3800, +0.0050] | 0.991 / 0.073 | ✗ (worse) |
| fixed10 | MAE vs oracle stop (levels) (relsg_gt-oracle) | +1.5000 [-2.0000, +4.5000] | — | ✗ (worse) |
| fixed10 | agreement with oracle (relsg_gt-oracle) | -0.1500 [-0.4500, +0.2000] | — | ✗ (worse) |
| fixed15 | MAE vs oracle stop (levels) (relsg_gt-oracle) | +2.0000 [+0.0000, +4.0000] | — | ✗ (worse) |
| fixed15 | agreement with oracle (relsg_gt-oracle) | +0.3000 [+0.1000, +0.5000] | — | n/a (0 % by construction) |
| fixed20 | MAE vs oracle stop (levels) (relsg_gt-oracle) | +2.5000 [-0.0125, +5.0000] | — | ✗ (worse) |
| fixed20 | agreement with oracle (relsg_gt-oracle) | -0.2500 [-0.5000, +0.0012] | — | ✗ (worse) |
