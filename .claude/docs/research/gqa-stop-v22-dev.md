# gqa_stop §22: enlarged dev visual subset and accuracy–level curves (dev only)

Date: 2026-10-10. Run: `configs/gqa_stop_v22.json` (run_id visapp_gqa_v22, family `logprob_item`, A13 clean decoding), output `outputs/gqa_stop/visapp_gqa_v22/`.
Scope: dev only, θ chosen in-sample. There was no freeze, no test method inference and no commit.
Terminology follows plan A14: P_ans is the answer probability, and Rel is the self-assessed confidence. Code identifiers keep `rel`.

## Verdict

- **0.90 to 0.99 targets.** On the enlarged dev set (102 questions, 89 images), checklist_v4_min and direct_v4 are **indistinguishable**. The mean-level difference is within ±0.4 at every target, and every CI includes 0.
- **The 59-question 0.99 gap does not survive.** On the 59 questions it was 15.05 vs 18.51, with 0 wrong stops required. On the 102 questions, 0.99 allows one wrong stop. Both methods use that slack, giving 15.76 vs 15.46.
- **Zero wrong stops (supplementary).** The gap reappears: checklist_v4_min 16.14 vs direct_v4 18.64, difference −2.50 [−3.33, −1.72]. But it rests entirely on **one unstable question**, 11213386. Without it, direct_v4 is better (15.52 vs 16.18). checklist_v4_min also avoids that question only inside a non-monotone θ window (0.48–0.51). This is not evidence for the decomposition.
- **Both v4 signals beat fixed stopping.** Both lie about 0.06 above the fixed-level concave hull at every target. At 0.95 they stop at mean level about 14, while the fixed hull needs 17.2.
- **Fair v1 becomes competitive at 0.95.** Its verbalised confidences give only 5 operating points. With 102 questions, the θ=0.90 point lands at accuracy 0.961, mean level 14.45. At 0.97/0.99 it collapses to L20.
- **Bottom line.** The data do not support "decomposition stops earlier than Direct at equal accuracy" on dev, at any target.

## 1. Enlarged visual subset (§22.2)

The adoption cap is removed: the target equals every L20-correct candidate. The A2-bis cell rules and the `max_questions_per_image=2` scan rule are unchanged. textprobe excludes prior-answerable questions. Screening and the determinism gate were fully reused from v4c records with identical keys; the gate passed with 120 calls and 0 argmax violations.

| split | candidates | L20-correct = adopted | prior-answerable | **visual** | images | original subset kept | additions |
|---|---|---|---|---|---|---|---|
| dev | 265 | 166 | 64 | **102** | 89 | 59 / 59 | 43 |
| test | 639 | 406 | 155 | **251** | 213 | 125 / 125 | 126 (count only; no method inference) |

The dev visual subset breaks down as follows:
- **By attribute.** color 41, material 46, shape 15.
- **By relate.** relate 0: 53, relate 1: 49.
- **By cell.** color 20/21, material 24/22, shape 9/6.

The test visual subset breaks down as follows:
- **By attribute.** color 127, material 105, shape 19.
- **By relate.** relate 0: 147, relate 1: 104.

Provenance, with sha, per-cell counts and a containment check, is in `visual_subset_provenance.json`.

Inference on the 43 additions covered reader, planner_v3, observer_v3 with logprobs, relevance_v4, direct_lp and the v1 arm (planner v1, checklist v2, direct verbalised). All 59 originals were reused from cache. The v1 planner had a schema failure on 2 additions, 05611349 and 071027361, in addition to 05827270 known from v4c §5. Their checklist_v1 scores are 0 at all levels, so they force L20.

## 2. Per-target selection on 102 dev questions (`dev_targets.csv`)

Columns: θ is chosen in-sample. "s&w" is the number of questions stopped and wrong. acc−hull is accuracy minus the fixed concave hull.

| target | method | θ | acc | mean L | stop20 | s&w | acc−hull |
|---|---|---|---|---|---|---|---|
| 0.90 | checklist_v4_min | 0.14 | 0.912 | 11.87 | 0.18 | 9 | +0.072 |
| 0.90 | direct_v4 | 0.22 | 0.902 | 12.25 | 0.17 | 10 | +0.053 |
| 0.90 | checklist_v1_min | 0.90 | 0.961 | 14.45 | 0.31 | 4 | +0.061 |
| 0.90 | direct verbalised | 0.95 | 0.980 | 19.50 | 0.93 | 2 | −0.011 |
| 0.90 | Fixed-dev | L16 | 0.902 | 16 | | | |
| **0.95** | **checklist_v4_min** | 0.28 | 0.951 | **14.15** | 0.29 | 5 | +0.056 |
| **0.95** | **direct_v4** | 0.30 | 0.951 | **13.83** | 0.25 | 5 | +0.063 |
| 0.95 | checklist_v1_min | 0.90 | 0.961 | 14.45 | 0.31 | 4 | +0.061 |
| 0.95 | direct verbalised | 0.95 | 0.980 | 19.50 | 0.93 | 2 | −0.011 |
| 0.95 | Fixed-dev | L20 | 1.000 | 20 | | | |
| 0.97 | checklist_v4_min | 0.36 | 0.971 | 15.01 | 0.34 | 3 | +0.060 |
| 0.97 | direct_v4 | 0.36 | 0.971 | 14.77 | 0.31 | 3 | +0.065 |
| 0.97 | checklist_v1_min | 1.00 | 0.990 | 19.98 | 0.99 | 1 | −0.009 |
| 0.97 | direct verbalised | 0.95 | 0.980 | 19.50 | 0.93 | 2 | −0.011 |
| 0.99 | checklist_v4_min | 0.42 | 0.990 | 15.76 | 0.44 | 1 | +0.066 |
| 0.99 | direct_v4 | 0.39 | 0.990 | 15.46 | 0.39 | 1 | +0.072 |
| 0.99 | checklist_v1_min | 1.00 | 0.990 | 19.98 | 0.99 | 1 | −0.009 |
| 0.99 | direct verbalised | ∞ | 1.000 | 20.00 | 1.00 | 0 | 0 |
| 1.00 (suppl.) | checklist_v4_min | 0.48 | 1.000 | 16.14 | 0.49 | 0 | |
| 1.00 (suppl.) | direct_v4 | 0.66 | 1.000 | 18.64 | 0.75 | 0 | |

- **Fixed curve on 102.** L1 0.333, L5 0.657, L10 0.765, L14 0.892, L16–L19 0.902, L20 1.000. The jump at L20 is by construction, because the set keeps only L20-correct questions. The fixed concave hull reaches 0.90 / 0.95 / 0.97 / 0.99 at mean level 14.4 / 17.2 / 18.3 / 19.5.
- **Stopped-and-wrong ids, checklist_v4_min.**
  - 0.95: 11213386, 11619091, 18486270, 18486310, 19315625.
  - 0.97: 11213386, 11619091, 18486270.
  - 0.99: 02957790.
- **Stopped-and-wrong ids, direct_v4.**
  - 0.95: 03295873, 0813387, 11213386, 13386996, 18486270.
  - 0.97: 0813387, 11213386, 13386996.
  - 0.99: 11213386.
- **Logged variants at 0.95.** Accuracy is 0.951 unless noted.

| variant | mean L | note |
|---|---|---|
| pans_min | 11.73 | lowest of all |
| pans_mean | 12.69 | |
| direct_pans | 13.48 | |
| soft_existence | 13.93 | |
| soft_existence k=2 | 14.10 | acc 0.961 |
| checklist_v4_min k=2 | 14.28 | acc 0.961 |
| checklist_v1_mean | 14.31 | |
| direct_v4 k=2 | 14.59 | |
| rel_mean | 14.76 | |
| checklist_v4_mean | 14.89 | |
| direct_rel | 15.08 | acc 0.961 |
| rel_min | 15.35 | |

- **Operating points over the θ grid.** checklist_v4_min has 99, direct_v4 95, checklist_v1_min 5 and direct verbalised 13.

## 3. Paired image-cluster bootstrap per target (`dev_paired_by_target.csv`)

The bootstrap uses 2000 draws with seed 20261008. θ is held at its in-sample selection, so the CI is conditional on θ and does not include selection noise.

| subset | target | checklist_v4_min − direct_v4: mean L [95% CI] | checklist_v4_min − checklist_v1_min: mean L [95% CI] |
|---|---|---|---|
| 102 | 0.90 | −0.38 [−0.90, +0.17] | −2.58 [−3.57, −1.62] |
| 102 | 0.95 | **+0.31 [−0.18, +0.83]** | −0.30 [−1.24, +0.67] |
| 102 | 0.97 | +0.24 [−0.29, +0.78] | −4.97 [−6.10, −3.92] |
| 102 | 0.99 | +0.30 [−0.21, +0.82] | −4.22 [−5.30, −3.22] |
| 102 | 1.00 (suppl.) | −2.50 [−3.33, −1.72] | −3.86 [−4.87, −2.92] |
| 59 (ref.) | 0.90 | −0.14 [−0.85, +0.62] | −1.64 [−2.95, −0.30] |
| 59 (ref.) | 0.95 | +0.63 [−0.17, +1.48] | −5.32 [−6.76, −3.95] |
| 59 (ref.) | 0.97 | +0.37 [−0.38, +1.12] | −5.17 [−6.59, −3.83] |
| 59 (ref.) | 0.99 | −3.46 [−4.58, −2.37] | −4.95 [−6.33, −3.66] |

Accuracy differences are within ±0.05 everywhere, and every CI includes 0.

### Why the zero-wrong gap is one question

- **The question.** 11213386 asks "What color is the umbrella on the right?", with GT yellow. The common reader answers yellow at L11, L12 and L20, but red at L13–L19.
- **direct_v4.** Its score reaches 0.58 at L15 and L18, where the reader is wrong. Any θ below 0.66 stops it wrong there.
- **checklist_v4_min.** It escapes that question only for θ in 0.48–0.51. At θ 0.52–0.63 it stops wrong on it again, so the zero-wrong set is non-monotone in θ.
- **Without 11213386 (101 questions).** The zero-wrong points are direct_v4 θ0.39 at 15.52 and checklist_v4_min θ0.48 at 16.18.
- **Conclusion.** The 0.99 / zero-wrong advantage seen on the 59 questions is not robust. It depends on a question whose L20 correctness is itself unstable across levels.

## 4. Breakdown at 0.95 (`dev_breakdown.csv`; 102 questions, θ as in §2)

Each cell gives accuracy @ mean level.

| method | color (41) | material (46) | shape (15) | relate 0 (53) | relate 1 (49) |
|---|---|---|---|---|---|
| checklist_v4_min | 0.951 @ 11.61 | 0.957 @ 17.76 | 0.933 @ 10.00 | 0.943 @ 13.51 | 0.959 @ 14.84 |
| direct_v4 | 0.927 @ 11.29 | 0.957 @ 16.72 | 1.000 @ 11.93 | 0.925 @ 12.94 | 0.980 @ 14.80 |
| checklist_v1_min | 0.927 @ 11.37 | 0.978 @ 17.96 | 1.000 @ 12.13 | 0.925 @ 14.43 | 1.000 @ 14.47 |
| direct verbalised | 1.000 @ 19.76 | 0.957 @ 19.13 | 1.000 @ 19.93 | 0.962 @ 19.30 | 1.000 @ 19.71 |

- **Material is the costly attribute.** checklist_v4_min forces L20 on 57% of material questions, against 7% of color questions. direct_v4 forces L20 on 43% of material questions.
- **Relate 1.** The checklist does not stop later on relate-1 questions than direct_v4 does: the mean levels are 14.84 and 14.80.

## 5. Curves (`dev_curves.csv`, `figures/dev_curves.png`)

The θ-grid curves of checklist_v4_min and direct_v4 overlap from mean level about 10 to 20 on both the 102 and the 59 questions. Both lie above the fixed-level curve and its concave hull. direct_v4 additionally reaches lower operating points, at mean level 3.5–10. checklist_v4_min's lowest point is at about 9.7, because the min over items is held back by the existence item.

## Notes

- **In-sample.** θ is selected and scored on the same 102 questions.
- **Granularity.** 0.95 on 102 allows 5 wrong stops, and 0.99 allows 1.
- **Test enlargement.** It was counted only: no planner, observer, direct or v1-arm call was made on test additions.
- **Files.**
  - Code: `src/milestone1/gqa_stop/devgrid_v22.py` (CLI `curves`), tested in `tests/test_gqa_stop_devgrid_v22.py`.
  - Supplementary zero-wrong rows: `dev_targets_zero_wrong.csv`, `dev_paired_zero_wrong.csv`.

## 6. §23 firing rate on reader-wrong dev questions (pre-registered; dev only)

- **Set.** The 99 dev candidates whose L20 screening reader answer was incorrect (85 images), with no textprobe exclusion.
- **Runs.** reader, planner_v3, observer_v3 with logprobs, relevance_v4, direct_lp and the v1 arm, all over L1–L20. Records are in `outputs/gqa_stop/visapp_gqa_v22/ext_dev_wrong/`; only the L20 screening reader records were copied from the v22 root.
- **θ.** Taken from the §22 enlarged-dev grid (`dev_targets.csv`, subset v22) without re-tuning.
- **Firing.** A question fires when the stop rule is met at some level 1–20: score ≥ θ, with k consecutive levels for k=2. The forced L20 stop is not a firing.
- **Reader accuracy on this set.** About 0.05–0.13 at L1–L19 and 0 at L20, by construction.

### Verdict

- **Firing rate.** At 0.95 the main methods fire on about **two thirds** of the questions the reader gets wrong at L20: checklist_v4_min 0.646, direct_v4 0.677. The difference is −0.030 [−0.095, +0.030].
- **No difference between them.** checklist_v4_min and direct_v4 do not differ at any target. Neither signal detects "cannot be answered with this image".
- **Low firing comes with never stopping.** Direct verbalised fires on only 0.091, and the v1 arm at 0.97/0.99 fires on 0. Both pay for it by stopping at about L20 on answerable questions too (§2).
- **Matched answerable cost.** Across the θ grid, the answer-probability variant pans_min (logged) fires least. With answerable mean level ≤ 15 it reaches 0.31, against 0.57 for checklist_v4_min and 0.55 for direct_v4. This is in-sample and logged-only.

### Firing rate per target (`firing_summary.csv`; n = 99)

Each cell gives the firing rate, the median firing level, and the number of fired questions where the reader was correct at the firing level.

| method | 0.90 | 0.95 | 0.97 | 0.99 |
|---|---|---|---|---|
| checklist_v4_min | 0.788 / L11 / 4 | **0.646** / L12.5 / 3 | 0.556 / L14 / 1 | 0.495 / L14 / 1 |
| direct_v4 | 0.778 / L12 / 6 | **0.677** / L13 / 1 | 0.545 / L13.5 / 1 | 0.535 / L14 / 1 |
| checklist_v1_min | 0.717 / L14 / 5 | 0.717 / L14 / 5 | 0.000 | 0.000 |
| direct verbalised | 0.091 / L19 / 0 | 0.091 / L19 / 0 | 0.091 / L19 / 0 | 0.000 |
| pans_min (logged) | 0.758 / L8 / 4 | 0.697 / L8 / 4 | 0.384 / L8.5 / 2 | 0.283 / L9 / 0 |
| direct_pans (logged) | 0.556 / L4 / 4 | 0.465 / L4.5 / 4 | 0.424 / L5 / 4 | 0.303 / L5 / 3 |
| soft_existence (exploratory) | 0.798 | 0.667 | 0.566 | 0.495 |
| checklist_v4_min k=2 (exploratory) | 0.828 | 0.667 | 0.525 | 0.455 |

The other logged variants are in `firing_summary.csv`. At 0.95 they fire on between 0.566 (direct_v4 k=2) and 0.717 (checklist_v1_mean).

### Paired image-cluster bootstrap of firing rate (`firing_paired.csv`; 2000 draws, seed 20261008)

| target | checklist_v4_min − direct_v4 [95% CI] (only-a / only-b) | − checklist_v1_min | − direct verbalised |
|---|---|---|---|
| 0.90 | +0.010 [−0.053, +0.080] (6 / 5) | +0.071 [−0.011, +0.154] | +0.697 [+0.596, +0.794] |
| 0.95 | −0.030 [−0.095, +0.030] (4 / 7) | −0.071 [−0.170, +0.030] | +0.556 [+0.438, +0.673] |
| 0.97 | +0.010 [−0.069, +0.082] (8 / 7) | +0.556 [+0.453, +0.653] | +0.465 [+0.347, +0.581] |
| 0.99 | −0.040 [−0.109, +0.022] (4 / 8) | +0.495 [+0.392, +0.594] | +0.495 [+0.392, +0.594] |

### Firing-level histogram at 0.95 (`firing_level_hist.csv`)

Firing levels spread over the whole range. checklist_v4_min fires 21 of its 64 times at L ≤ 9; direct_v4 fires 18 of its 67 times there. The firing levels do not concentrate at L20: checklist_v4_min fires at L20 6 times and direct_v4 9 times.

### Two-axis view (`ext_dev_wrong/two_axis_curve.{csv,png}`; θ grid, in-sample)

Each cell gives the lowest firing rate on the reader-wrong set among θ values whose answerable mean level (102 questions) is at most the bound.

| answerable mean L ≤ | checklist_v4_min | direct_v4 | pans_min (logged) |
|---|---|---|---|
| 12 | 0.788 | 0.788 | 0.687 |
| 14 | 0.687 | 0.677 | 0.485 |
| 15 | 0.566 | 0.545 | 0.313 |
| 16 | 0.434 | 0.475 | 0.253 |
| 18 | 0.182 | 0.293 | 0.121 |

### What the reader-wrong set contains (caveat for interpretation)

Many of these questions are not "unanswerable from the image". They are GQA label or referent ambiguity, so a confident stop is not necessarily a false stop:
- **Label granularity.** 00945894 has GT brown, and the reader and direct both say "wood". 01210580 has GT silver and the reader says gray. 01672446 has GT "dark" and the reader says gray.
- **Referent ambiguity.** In 00940841, 01584227 and 02206413 the reader names the colour of a different shirt, and direct agrees.
- **Material.** 01179300/01179301 have GT glass while the reader and direct say wood: a glass-fronted wooden cabinet.

The firing rate therefore over-states the false-stop rate. That bias is the same for all methods, so the paired comparison is unaffected.

The v1 planner had a schema failure on 1 of the 99 questions, 02292567. Its checklist_v1 score is 0, so it never fires under the v1 arm. Every v4 call parsed.

Code: `src/milestone1/gqa_stop/firing_v23.py` (CLI `firing`), tested in `tests/test_gqa_stop_firing_v23.py`. The test-side 233 questions were not touched.
