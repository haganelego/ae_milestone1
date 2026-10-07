# MIRU-style method comparison (presentation; plan primary = R50 policy − same-p mix)

Oracle = GT two-choice oracle on that seed (top-k by g = q20 − q10, k = the policy's number of t20 slots); a GT-posterior reference, not a method. With stops in {10, 20}, MAE = 10 × (1 − agreement). Comparators are ONLINE only: the same-p image-independent mix (primary) and conf_two_stage (level-agnostic: continue iff the top-8 cleaned-G mean entity score at t10 is below the frozen dev median). fixed10/15/20 are schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator; they are shown for scale and never tested. ✓ = 95 % paired-bootstrap CI excludes 0 or permutation p < .05 (slot permutation vs the mix, paired sign flip vs conf_two_stage); the direction says whether the U policy is better or worse.

## Phase D seedA (n = 60)

**Headline (plan primary): R50 policy − same-p mix = -0.0071 [-0.0246, +0.0121], permutation p(better) = 0.767.**

| method | role | mean level | MAE vs R50-oracle | agree (R50-oracle) | R50 | MAE vs relsg-oracle | agree (relsg-oracle) | relsg_gt |
|---|---|---|---|---|---|---|---|---|
| U policy | proposed (frozen U) | 15.17 | 6.00 | 40% | 0.1290 | 5.00 | 50% | 0.4852 |
| mix (same p) | online comparator (primary) | 15.17 | 4.99 | 50% | 0.1361 (perm95 [0.1166, 0.1550]) | 4.99 | 50% | 0.4622 (perm95 [0.4364, 0.4871]) |
| conf_two_stage | online comparator (level-agnostic, observation-based) | 14.67 | 5.50 | 45% | 0.1328 | 5.50 | 45% | 0.4618 |
| fixed10 | schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator | 10.00 | 5.17 | 48% | 0.0990 | 5.17 | 48% | 0.4006 |
| fixed15 | schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator | 15.00 | 5.00 | 0% | 0.1419 | 5.00 | 0% | 0.4555 |
| fixed20 | schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator | 20.00 | 4.83 | 52% | 0.1709 | 4.83 | 52% | 0.5198 |
| GT two-choice oracle (reference) | GT-posterior reference (upper bound) | 15.17 | 0.00 | 100% | 0.1846 | 0.00 | 100% | 0.5298 |

| U policy vs | column | Δ (policy − comparator) [95% CI] | perm p (better / worse) | sig |
|---|---|---|---|---|
| mix | MAE vs oracle stop (levels) (R50-oracle) | +1.0056 [-0.2333, +2.1667] | 0.968 / 0.090 | ✗ (worse) |
| mix | agreement with oracle (R50-oracle) | -0.1006 [-0.2167, +0.0233] | 0.968 / 0.090 | ✗ (worse) |
| mix | q(stop) (R50) | -0.0071 [-0.0246, +0.0121] | 0.767 / 0.233 | ✗ (worse) |
| conf_two_stage | MAE vs oracle stop (levels) (R50-oracle) | +0.5000 [-1.1667, +2.3333] | 0.782 / 0.350 | ✗ (worse) |
| conf_two_stage | agreement with oracle (R50-oracle) | -0.0500 [-0.2333, +0.1167] | 0.782 / 0.350 | ✗ (worse) |
| conf_two_stage | q(stop) (R50) | -0.0038 [-0.0285, +0.0220] | 0.615 / 0.386 | ✗ (worse) |
| mix | q(stop) (relsg_gt) | +0.0230 [+0.0003, +0.0468] | 0.038 / 0.962 | ✓ better |
| conf_two_stage | q(stop) (relsg_gt) | +0.0234 [-0.0098, +0.0581] | 0.093 / 0.907 | ✗ (better) |
| mix | MAE vs oracle stop (levels) (relsg_gt-oracle) | +0.0056 [-1.2724, +1.1946] | 0.612 / 0.589 | ✗ (worse) |
| mix | agreement with oracle (relsg_gt-oracle) | -0.0006 [-0.1195, +0.1272] | 0.612 / 0.589 | ✗ (worse) |
| conf_two_stage | MAE vs oracle stop (levels) (relsg_gt-oracle) | -0.5000 [-2.1667, +1.1667] | 0.345 / 0.775 | ✗ (better) |
| conf_two_stage | agreement with oracle (relsg_gt-oracle) | +0.0500 [-0.1167, +0.2167] | 0.345 / 0.775 | ✗ (better) |

## Phase D seedB (replication) (n = 60)

**Headline (plan primary): R50 policy − same-p mix = -0.0107 [-0.0323, +0.0101], permutation p(better) = 0.820.**

| method | role | mean level | MAE vs R50-oracle | agree (R50-oracle) | R50 | MAE vs relsg-oracle | agree (relsg-oracle) | relsg_gt |
|---|---|---|---|---|---|---|---|---|
| U policy | proposed (frozen U) | 15.17 | 5.67 | 43% | 0.1265 | 4.33 | 57% | 0.4822 |
| mix (same p) | online comparator (primary) | 15.17 | 4.99 | 50% | 0.1372 (perm95 [0.1161, 0.1589]) | 4.99 | 50% | 0.4596 (perm95 [0.4385, 0.4799]) |
| conf_two_stage | online comparator (level-agnostic, observation-based) | 14.67 | 6.17 | 38% | 0.1245 | 5.17 | 48% | 0.4687 |
| fixed10 | schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator | 10.00 | 5.17 | 48% | 0.1067 | 5.17 | 48% | 0.4253 |
| fixed15 | schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator | 15.00 | 5.00 | 0% | 0.1294 | 5.00 | 0% | 0.4501 |
| fixed20 | schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator | 20.00 | 4.83 | 52% | 0.1658 | 4.83 | 52% | 0.4917 |
| GT two-choice oracle (reference) | GT-posterior reference (upper bound) | 15.17 | 0.00 | 100% | 0.1822 | 0.00 | 100% | 0.5174 |

| U policy vs | column | Δ (policy − comparator) [95% CI] | perm p (better / worse) | sig |
|---|---|---|---|---|
| mix | MAE vs oracle stop (levels) (R50-oracle) | +0.6722 [-0.6224, +1.8889] | 0.911 / 0.207 | ✗ (worse) |
| mix | agreement with oracle (R50-oracle) | -0.0672 [-0.1889, +0.0622] | 0.911 / 0.207 | ✗ (worse) |
| mix | q(stop) (R50) | -0.0107 [-0.0323, +0.0101] | 0.820 / 0.180 | ✗ (worse) |
| conf_two_stage | MAE vs oracle stop (levels) (R50-oracle) | -0.5000 [-2.3333, +1.3333] | 0.353 / 0.778 | ✗ (better) |
| conf_two_stage | agreement with oracle (R50-oracle) | +0.0500 [-0.1333, +0.2333] | 0.353 / 0.778 | ✗ (better) |
| conf_two_stage | q(stop) (R50) | +0.0020 [-0.0160, +0.0213] | 0.430 / 0.571 | ✗ (better) |
| mix | q(stop) (relsg_gt) | +0.0226 [+0.0033, +0.0424] | 0.017 / 0.984 | ✓ better |
| conf_two_stage | q(stop) (relsg_gt) | +0.0135 [-0.0080, +0.0356] | 0.112 / 0.888 | ✗ (better) |
| mix | MAE vs oracle stop (levels) (relsg_gt-oracle) | -0.6611 [-1.8667, +0.5336] | 0.221 / 0.913 | ✗ (better) |
| mix | agreement with oracle (relsg_gt-oracle) | +0.0661 [-0.0534, +0.1867] | 0.221 / 0.913 | ✗ (better) |
| conf_two_stage | MAE vs oracle stop (levels) (relsg_gt-oracle) | -0.8333 [-2.5000, +1.0000] | 0.228 / 0.872 | ✗ (better) |
| conf_two_stage | agreement with oracle (relsg_gt-oracle) | +0.0833 [-0.1000, +0.2500] | 0.228 / 0.872 | ✗ (better) |

## dev seed0 (n = 20)

**Headline (plan primary): R50 policy − same-p mix = -0.0100 [-0.0414, +0.0244], permutation p(better) = 0.736.**

| method | role | mean level | MAE vs R50-oracle | agree (R50-oracle) | R50 | MAE vs relsg-oracle | agree (relsg-oracle) | relsg_gt |
|---|---|---|---|---|---|---|---|---|
| U policy | proposed (frozen U) | 15.50 | 5.00 | 50% | 0.1582 | 6.00 | 40% | 0.5010 |
| mix (same p) | online comparator (primary) | 15.50 | 4.95 | 50% | 0.1682 (perm95 [0.1339, 0.2026]) | 4.95 | 50% | 0.5331 (perm95 [0.5009, 0.5634]) |
| conf_two_stage | online comparator (level-agnostic, observation-based) | 15.00 | 4.50 | 55% | 0.1630 | 4.50 | 55% | 0.5227 |
| fixed10 | schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator | 10.00 | 5.50 | 45% | 0.1343 | 5.50 | 45% | 0.4565 |
| fixed15 | schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator | 15.00 | 5.00 | 0% | 0.2008 | 5.00 | 0% | 0.5381 |
| fixed20 | schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator | 20.00 | 4.50 | 55% | 0.1959 | 4.50 | 55% | 0.5957 |
| GT two-choice oracle (reference) | GT-posterior reference (upper bound) | 15.50 | 0.00 | 100% | 0.2126 | 0.00 | 100% | 0.5922 |

| U policy vs | column | Δ (policy − comparator) [95% CI] | perm p (better / worse) | sig |
|---|---|---|---|---|
| mix | MAE vs oracle stop (levels) (R50-oracle) | +0.0500 [-2.0000, +2.2000] | 0.688 / 0.663 | ✗ (worse) |
| mix | agreement with oracle (R50-oracle) | -0.0050 [-0.2200, +0.2000] | 0.688 / 0.663 | ✗ (worse) |
| mix | q(stop) (R50) | -0.0100 [-0.0414, +0.0244] | 0.736 / 0.275 | ✗ (worse) |
| conf_two_stage | MAE vs oracle stop (levels) (R50-oracle) | +0.5000 [-3.0000, +3.5000] | 0.717 / 0.489 | ✗ (worse) |
| conf_two_stage | agreement with oracle (R50-oracle) | -0.0500 [-0.3500, +0.3000] | 0.717 / 0.489 | ✗ (worse) |
| conf_two_stage | q(stop) (R50) | -0.0049 [-0.0333, +0.0187] | 0.748 / 0.491 | ✗ (worse) |
| mix | q(stop) (relsg_gt) | -0.0321 [-0.0623, -0.0010] | 0.975 / 0.025 | ✓ worse |
| conf_two_stage | q(stop) (relsg_gt) | -0.0218 [-0.1015, +0.0413] | 0.682 / 0.338 | ✗ (worse) |
| mix | MAE vs oracle stop (levels) (relsg_gt-oracle) | +1.0500 [-1.0500, +3.0000] | 0.919 / 0.305 | ✗ (worse) |
| mix | agreement with oracle (relsg_gt-oracle) | -0.1050 [-0.3000, +0.1050] | 0.919 / 0.305 | ✗ (worse) |
| conf_two_stage | MAE vs oracle stop (levels) (relsg_gt-oracle) | +1.5000 [-1.5000, +4.5000] | 0.897 / 0.257 | ✗ (worse) |
| conf_two_stage | agreement with oracle (relsg_gt-oracle) | -0.1500 [-0.4500, +0.1500] | 0.897 / 0.257 | ✗ (worse) |

## dev seed1 (n = 20)

**Headline (plan primary): R50 policy − same-p mix = +0.0038 [-0.0266, +0.0349], permutation p(better) = 0.424.**

| method | role | mean level | MAE vs R50-oracle | agree (R50-oracle) | R50 | MAE vs relsg-oracle | agree (relsg-oracle) | relsg_gt |
|---|---|---|---|---|---|---|---|---|
| U policy | proposed (frozen U) | 15.50 | 6.00 | 40% | 0.1798 | 5.00 | 50% | 0.5057 |
| mix (same p) | online comparator (primary) | 15.50 | 4.95 | 50% | 0.1761 (perm95 [0.1437, 0.2071]) | 4.95 | 50% | 0.5228 (perm95 [0.4851, 0.5569]) |
| conf_two_stage | online comparator (level-agnostic, observation-based) | 15.00 | 3.50 | 65% | 0.1841 | 4.50 | 55% | 0.5265 |
| fixed10 | schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator | 10.00 | 5.50 | 45% | 0.1382 | 5.50 | 45% | 0.4670 |
| fixed15 | schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator | 15.00 | 5.00 | 0% | 0.1741 | 5.00 | 0% | 0.5269 |
| fixed20 | schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator | 20.00 | 4.50 | 55% | 0.2071 | 4.50 | 55% | 0.5685 |
| GT two-choice oracle (reference) | GT-posterior reference (upper bound) | 15.50 | 0.00 | 100% | 0.2221 | 0.00 | 100% | 0.5869 |

| U policy vs | column | Δ (policy − comparator) [95% CI] | perm p (better / worse) | sig |
|---|---|---|---|---|
| mix | MAE vs oracle stop (levels) (R50-oracle) | +1.0500 [-1.0500, +3.0000] | 0.917 / 0.312 | ✗ (worse) |
| mix | agreement with oracle (R50-oracle) | -0.1050 [-0.3000, +0.1050] | 0.917 / 0.312 | ✗ (worse) |
| mix | q(stop) (R50) | +0.0038 [-0.0266, +0.0349] | 0.424 / 0.582 | ✗ (better) |
| conf_two_stage | MAE vs oracle stop (levels) (R50-oracle) | +2.5000 [-0.5000, +5.5000] | 0.964 / 0.117 | ✗ (worse) |
| conf_two_stage | agreement with oracle (R50-oracle) | -0.2500 [-0.5500, +0.0500] | 0.964 / 0.117 | ✗ (worse) |
| conf_two_stage | q(stop) (R50) | -0.0043 [-0.0317, +0.0207] | 0.623 / 0.431 | ✗ (worse) |
| mix | q(stop) (relsg_gt) | -0.0172 [-0.0522, +0.0182] | 0.807 / 0.193 | ✗ (worse) |
| conf_two_stage | q(stop) (relsg_gt) | -0.0208 [-0.0912, +0.0428] | 0.721 / 0.281 | ✗ (worse) |
| mix | MAE vs oracle stop (levels) (relsg_gt-oracle) | +0.0500 [-2.0500, +2.0525] | 0.681 / 0.659 | ✗ (worse) |
| mix | agreement with oracle (relsg_gt-oracle) | -0.0050 [-0.2052, +0.2050] | 0.681 / 0.659 | ✗ (worse) |
| conf_two_stage | MAE vs oracle stop (levels) (relsg_gt-oracle) | +0.5000 [-2.5000, +3.5000] | 0.718 / 0.498 | ✗ (worse) |
| conf_two_stage | agreement with oracle (relsg_gt-oracle) | -0.0500 [-0.3500, +0.2500] | 0.718 / 0.498 | ✗ (worse) |

## dev seed2 (n = 20)

**Headline (plan primary): R50 policy − same-p mix = +0.0092 [-0.0240, +0.0412], permutation p(better) = 0.318.**

| method | role | mean level | MAE vs R50-oracle | agree (R50-oracle) | R50 | MAE vs relsg-oracle | agree (relsg-oracle) | relsg_gt |
|---|---|---|---|---|---|---|---|---|
| U policy | proposed (frozen U) | 15.50 | 5.00 | 50% | 0.1969 | 7.00 | 30% | 0.4972 |
| mix (same p) | online comparator (primary) | 15.50 | 4.95 | 50% | 0.1877 (perm95 [0.1543, 0.2232]) | 4.95 | 50% | 0.5232 (perm95 [0.4916, 0.5557]) |
| conf_two_stage | online comparator (level-agnostic, observation-based) | 15.00 | 3.50 | 65% | 0.1915 | 4.50 | 55% | 0.5130 |
| fixed10 | schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator | 10.00 | 5.50 | 45% | 0.1500 | 5.50 | 45% | 0.4517 |
| fixed15 | schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator | 15.00 | 5.00 | 0% | 0.1496 | 5.00 | 0% | 0.5163 |
| fixed20 | schedule-aware reference (knows the level index; GT-posterior for F) — not a comparator | 20.00 | 4.50 | 55% | 0.2186 | 4.50 | 55% | 0.5817 |
| GT two-choice oracle (reference) | GT-posterior reference (upper bound) | 15.50 | 0.00 | 100% | 0.2398 | 0.00 | 100% | 0.5842 |

| U policy vs | column | Δ (policy − comparator) [95% CI] | perm p (better / worse) | sig |
|---|---|---|---|---|
| mix | MAE vs oracle stop (levels) (R50-oracle) | +0.0500 [-2.0000, +2.1500] | 0.698 / 0.647 | ✗ (worse) |
| mix | agreement with oracle (R50-oracle) | -0.0050 [-0.2150, +0.2000] | 0.698 / 0.647 | ✗ (worse) |
| mix | q(stop) (R50) | +0.0092 [-0.0240, +0.0412] | 0.318 / 0.691 | ✗ (better) |
| conf_two_stage | MAE vs oracle stop (levels) (R50-oracle) | +1.5000 [-1.5000, +4.5000] | 0.882 / 0.279 | ✗ (worse) |
| conf_two_stage | agreement with oracle (R50-oracle) | -0.1500 [-0.4500, +0.1500] | 0.882 / 0.279 | ✗ (worse) |
| conf_two_stage | q(stop) (R50) | +0.0054 [-0.0325, +0.0500] | 0.414 / 0.641 | ✗ (better) |
| mix | q(stop) (relsg_gt) | -0.0260 [-0.0533, +0.0032] | 0.942 / 0.059 | ✗ (worse) |
| conf_two_stage | q(stop) (relsg_gt) | -0.0158 [-0.0731, +0.0448] | 0.695 / 0.306 | ✗ (worse) |
| mix | MAE vs oracle stop (levels) (relsg_gt-oracle) | +2.0500 [-0.0500, +3.8000] | 0.991 / 0.073 | ✗ (worse) |
| mix | agreement with oracle (relsg_gt-oracle) | -0.2050 [-0.3800, +0.0050] | 0.991 / 0.073 | ✗ (worse) |
| conf_two_stage | MAE vs oracle stop (levels) (relsg_gt-oracle) | +2.5000 [-0.5000, +5.5000] | 0.964 / 0.109 | ✗ (worse) |
| conf_two_stage | agreement with oracle (relsg_gt-oracle) | -0.2500 [-0.5500, +0.0500] | 0.964 / 0.109 | ✗ (worse) |
