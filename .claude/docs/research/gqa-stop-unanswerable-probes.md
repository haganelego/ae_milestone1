# gqa_stop: cache-only probes for firing on reader-wrong questions (dev)

Date: 2026-10-10. Cache-only: no model call was made. Inputs are the v22 dev level tables and raw records (`level_table_dev.csv`, `ext_dev_wrong/level_table.csv`, `direct_lp/`, `observer_v3_lp/`, `plans_v3/`).
Outputs: `outputs/gqa_stop/visapp_gqa_v22/probe_unans/`. Scripts there: `signals.py` (extraction), `analysis.py` (P1–P4), `extras.py` (gate × P_ans-alone), `crossfit.py` (out-of-fold θ).

Sets: the dev visual subset (102 questions, 89 images) and the dev reader-wrong set (99 questions, 85 images); 10 images are shared.
Firing follows §23: score ≥ θ at some level 1–20; the forced L20 stop is not a firing.
Bootstrap: image-cluster paired, 2000 draws, seed 20261008, θ held at its selection.
Reproduction: recomputed P_ans, direct_v4, checklist_v4_min and pans_min match the pipeline columns to 2e-16. Ungated rows reproduce §22/§23 exactly (direct_v4 θ0.30: 13.83 / 0.677; checklist_v4_min θ0.28: 14.15 / 0.646).

## Verdict

- **Dropping Rel is the cheapest real gain.** Replacing P_ans × Rel by P_ans alone (direct) lowers firing at 0.95 from 0.677 to 0.465 at the same mean level, and the gain holds out-of-fold (0.646 → 0.459 at 13.98 → 13.57). This is the already-logged `direct_pans` variant; Rel adds confident stops on reader-wrong questions.
- **Distribution-shape signals add nothing over P_ans.** Margin, normalised entropy and candidate mass behave like P_ans alone; inside X × Rel they change firing by at most ±0.06.
- **Answer-trajectory gate (a) lowers firing, but leans on the subset construction.** At 0.90/0.95 it lowers firing by 0.16–0.18 (direct) and 0.11–0.13 (checklist) at matched cost, in-sample. Out-of-fold, direct_v4 + gate (a) loses accuracy (0.933 at 0.95). At 0.97/0.99 it costs +1.4 to +2.0 levels.
- **Best combination: P_ans alone + gate (a).** In-sample at 0.95: 0.961 @ 14.36, firing 0.313. Out-of-fold: 0.950 @ 14.34, firing 0.312, against direct_v4 at 0.945 @ 13.98, firing 0.646. Exploratory, chosen after looking.
- **Risk control on the mixed population collapses to never stopping (P3).** At the 0.95 θ, only about 52% of early stops on 102 ∪ 99 are correct. Precision ≥ 0.90 needs θ ≈ 0.91 (mean level 19.8); LTT certifies nothing below L20.
- **No cached signal separates the sets where it matters (P4).** At the firing level, out-of-fold AUROC is 0.61 [0.52, 0.71] for direct_v4 and 0.51 for checklist_v4_min. Separation grows with level only because the 102 become correct by construction.

## P1. Answer-trajectory gates

Reader answers use the pipeline's normalisation (NFKC, lowercase, strip, collapse whitespace). All gates are prefix-only (online-valid).

| fact | visual 102 | reader-wrong 99 |
|---|---|---|
| reader answer never changes L1..L20 | 26 | 43 |
| L1 answer equals L20 answer | 34 | 60 |
| L1 correct | 34 | 12 |
| L1 reader answer equals the (wrong) text-only answer | 43 | n/a (no text-only run) |

**Gate (a) is tied to the visual-subset construction.** The 102 exclude prior-answerable questions, so 43 of them start at the wrong text-prior answer and must change. The 64 prior-answerable L20-correct dev questions would mostly keep their L1 answer. Gate (a) would force them to L20, and that cost is not measured here: they have no L1–L19 records. 26 of the 102 never change either (L1 already correct), and gate (a) forces all of them to L20.

Selected rows (`p1p2_targets.csv`; in-sample θ). "matched" = ungated minimum firing over θ with accuracy ≥ target and mean level ≤ the gated mean level.

| method | gate | target | θ | mean L | Δ mean L [95% CI] | firing | Δ firing [95% CI] | Δ vs matched |
|---|---|---|---|---|---|---|---|---|
| direct_v4 | none | 0.95 | 0.30 | 13.83 | | 0.677 | | |
| direct_v4 | a | 0.90 | 0.07 | 12.75 | +0.49 [−0.88, +1.93] | 0.535 | −0.242 [−0.373, −0.108] | −0.242 |
| direct_v4 | a | 0.95 | 0.13 | 14.07 | +0.24 [−1.02, +1.47] | 0.495 | −0.182 [−0.304, −0.053] | −0.162 |
| direct_v4 | a | 0.97 | 0.29 | 16.15 | +1.37 [+0.44, +2.34] | 0.394 | −0.152 [−0.265, −0.041] | −0.081 |
| direct_v4 | a | 0.99 | 0.39 | 17.39 | +1.93 [+1.14, +2.79] | 0.303 | −0.232 [−0.323, −0.144] | −0.030 |
| direct_v4 | a ∧ stable2 | 0.95 | 0.12 | 14.05 | +0.22 [−1.05, +1.51] | 0.475 | −0.202 [−0.324, −0.074] | −0.182 |
| direct_v4 | a ∧ stable3 | 0.95 | 0.22 | 15.46 | +1.63 [+0.54, +2.75] | 0.394 | −0.283 [−0.389, −0.184] | −0.141 |
| direct_v4 | changes ≤ 1 | 0.95 | 0.26 | 15.19 | +1.35 [+0.60, +2.19] | 0.444 | −0.232 [−0.326, −0.140] | −0.101 |
| direct_v4 | changes ≤ 2 | 0.95 | 0.26 | 14.16 | +0.32 [−0.20, +0.89] | 0.535 | −0.141 [−0.228, −0.059] | −0.121 |
| checklist_v4_min | none | 0.95 | 0.28 | 14.15 | | 0.646 | | |
| checklist_v4_min | a | 0.95 | 0.13 | 15.02 | +0.87 [−0.26, +2.09] | 0.444 | −0.202 [−0.330, −0.074] | −0.111 |
| checklist_v4_min | a | 0.99 | 0.26 | 16.57 | +0.80 [−0.13, +1.80] | 0.384 | −0.111 [−0.229, +0.010] | +0.051 |
| checklist_v4_min | a ∧ stable2 | 0.95 | 0.13 | 15.12 | +0.97 [−0.17, +2.19] | 0.424 | −0.222 [−0.347, −0.096] | −0.131 |
| checklist_v4_min | changes ≤ 1 | 0.95 | 0.28 | 16.00 | +1.85 [+1.19, +2.61] | 0.434 | −0.212 [−0.297, −0.131] | 0.000 |

- **Stability alone (stable2/stable3) does nothing.** Firing changes by −0.03 to +0.01 at every target.
- **Answer-change caps (c) mostly buy firing with levels.** At matched cost they gain −0.10 to −0.12 on direct and 0 to +0.10 on checklist; m = 0 is worse than ungated on checklist.
- **The gain is confined to mean level 10–15.** Above about 16 all gated and ungated firing curves coincide (`p1_curves.png`).
- **Extra gates.** reader == direct agreement gives 0 to −0.09 at matched cost. Gate (a) on the direct answer's own trajectory behaves like gate (a) but costs more levels.

## P2. Distribution-shape signals at the answer token

Every direct_lp and observer record stores **10** top_logprobs per token, and the greedy token is always the top-1 (4020/4020). Signals at the first answer-value token:
- **margin.** p1 − p2 over the raw top-10. The merged variant first sums case/whitespace variants ("metal"/"Metal").
- **negent.** 1 − H/log 10 over the renormalised top-10.
- **mass.** The sum of the top-10 probabilities.
- **Checklist variants.** They go through the same v1 gate and min over items as checklist_v4_min.

At 0.95, in-sample θ (`p1p2_targets.csv`):

| variant | mean L | Δ mean L [CI] | firing | Δ firing [CI] |
|---|---|---|---|---|
| direct_v4 (P_ans × Rel) | 13.83 | | 0.677 | |
| margin × Rel | 13.98 | +0.15 [−0.35, +0.66] | 0.667 | −0.010 [−0.062, +0.041] |
| negent × Rel | 14.01 | +0.18 [+0.02, +0.38] | 0.646 | −0.030 [−0.080, +0.010] |
| mass × Rel | 15.11 | +1.28 [+0.88, +1.70] | 0.626 | −0.051 [−0.120, +0.019] |
| **P_ans alone** (= direct_pans) | 13.48 | −0.35 [−1.79, +1.22] | **0.465** | −0.212 [−0.324, −0.095] |
| margin alone | 14.14 | +0.30 [−1.14, +1.81] | 0.465 | −0.212 [−0.330, −0.093] |
| negent alone | 13.25 | −0.59 [−2.01, +1.00] | 0.465 | −0.212 [−0.333, −0.089] |
| mass alone | 14.72 | +0.88 [+0.00, +1.79] | 0.384 | −0.293 [−0.402, −0.181] (infeasible at 0.97+) |
| checklist_v4_min | 14.15 | | 0.646 | |
| ck margin × Rel | 13.11 | −1.04 [−1.41, −0.67] | 0.667 | +0.020 [−0.022, +0.071] |
| ck negent × Rel | 14.54 | +0.39 [+0.22, +0.58] | 0.606 | −0.040 [−0.102, +0.020] |
| ck P_ans alone (= pans_min) | 11.73 | −2.42 [−3.12, −1.75] | 0.697 | +0.051 [−0.030, +0.130] |

Frontier: minimum firing over θ with accuracy ≥ 0.95 and mean level ≤ bound (`p1p2_frontier.csv`, `extras_frontier.csv`):

| variant | ≤ 14 | ≤ 15 | ≤ 16 | ≤ 17 |
|---|---|---|---|---|
| direct_v4 | 0.677 | 0.545 | 0.475 | 0.384 |
| direct P_ans alone | 0.444 | 0.384 | 0.303 | 0.222 |
| direct negent alone | 0.444 | 0.374 | 0.263 | 0.242 |
| direct margin alone | — | 0.394 | 0.293 | 0.192 |
| direct P_ans alone + gate (a) | — | 0.263 | 0.222 | 0.141 |
| checklist_v4_min | — | 0.566 | 0.434 | 0.273 |
| ck P_ans alone | 0.485 | 0.313 | 0.253 | 0.202 |
| ck P_ans alone + gate (a) | — | 0.374 | 0.242 | 0.141 |

**Reading.** At equal accuracy and mean level, removing Rel lowers firing by 0.15–0.25. No shape signal beats P_ans alone. Folding a shape signal into X × Rel does not help, because Rel dominates the firing.

## Out-of-fold check (`crossfit.csv`)

θ is chosen on train-fold 102 questions and applied to held-out 102 and to the 99 in the same image folds. The folds are image-grouped 5-fold over the union, averaged over 10 seeds.

| variant | target | OOF acc | OOF mean L | OOF firing |
|---|---|---|---|---|
| direct_v4 | 0.95 | 0.945 | 13.98 | 0.646 |
| direct_v4 + a | 0.95 | 0.933 | 14.55 | 0.468 |
| direct P_ans alone | 0.95 | 0.947 | 13.57 | 0.459 |
| direct negent alone | 0.95 | 0.949 | 13.35 | 0.459 |
| direct P_ans alone + a | 0.95 | 0.950 | 14.34 | 0.312 |
| direct P_ans alone + a ∧ stable2 | 0.95 | 0.934 | 13.66 | 0.383 |
| checklist_v4_min | 0.95 | 0.946 | 14.16 | 0.648 |
| checklist_v4_min + a | 0.95 | 0.944 | 14.93 | 0.449 |
| ck P_ans alone | 0.95 | 0.950 | 12.85 | 0.584 |
| ck P_ans alone + a | 0.95 | 0.942 | 14.11 | 0.434 |

The 0.90 and 0.97 rows are in the CSV. At 0.97, direct P_ans alone + gate (a) gives 0.953 @ 14.65 with firing 0.274, against direct_v4 at 0.962 @ 14.80 with firing 0.553.

## P3. Risk-controlled θ on 102 ∪ 99 (`p3_risk_control.csv`, `extras_p3.csv`)

Precision is the fraction of early stops (level < 20) that are correct; the 99 always count as wrong. The fired-including-L20 version gives the same picture.

| method | rule | θ | acc102 | mean L 102 | firing99 | stops | wrong | precision | CP lower (δ=0.1) |
|---|---|---|---|---|---|---|---|---|---|
| direct_v4 | θ@0.95 on 102 alone | 0.30 | 0.951 | 13.83 | 0.677 | 135 | 63 | 0.533 | 0.475 |
| direct_v4 | empirical, α = 0.05 or 0.10 | 0.91 | 1.000 | 19.82 | 0.010 | 2 | 0 | 1.000 | 0.316 |
| direct_v4 | LTT fixed-sequence (binomial or Hoeffding), α = 0.05/0.10 | 0.95 | 1.000 | 20.00 | 0 | 0 | 0 | — | — |
| checklist_v4_min | θ@0.95 on 102 alone | 0.28 | 0.951 | 14.15 | 0.646 | 130 | 63 | 0.515 | 0.456 |
| checklist_v4_min | empirical, α = 0.05 or 0.10 | 0.97 | 1.000 | 19.92 | 0 | 2 | 0 | 1.000 | 0.316 |
| checklist_v4_min | LTT, α = 0.05/0.10 | 0.99 | 1.000 | 20.00 | 0 | 0 | 0 | — | — |
| P_ans alone + a | θ@0.95 on 102 alone | 0.71 | 0.961 | 14.36 | 0.313 | 74 | 30 | 0.595 | 0.513 |

- **This is a framing probe.** It chooses a population θ and does not detect unanswerability per question.
- **The mix is pessimistic.** The real dev candidate pool also has 64 prior-answerable L20-correct questions that would add correct early stops. They have no per-level records.
- **Approximations.** The binomial treatment conditions on the number of stops and ignores image clustering.

## P4. Does any cached signal separate the 99 from the 102?

Features at a level ℓ:
- **Direct signals.** P_ans, Rel, P_ans × Rel, margin, negent, mass.
- **Checklist signals.** checklist_v4_min, pans_min, rel_min, checklist margin min.
- **Answer trajectory.** Answer-change count, changed-from-L1, current answer run length, reader == direct.
- **Prefix maxima.** Running maxima of both scores.

Model: L2 logistic regression (numpy), image-grouped 5-fold CV over the union, 10 seeds. AUROC is out-of-fold. The CI is a cluster bootstrap on seed-averaged out-of-fold predictions.

| setting | n (102 / 99) | signals | attribute only | signals + attribute |
|---|---|---|---|---|
| at firing level, direct_v4 θ0.30 (fired only) | 81 / 67 | **0.609** (seed-avg 0.618 [0.522, 0.713]) | 0.520 | 0.620 |
| at firing level, checklist_v4_min θ0.28 (fired only) | 78 / 64 | **0.515** (0.510 [0.416, 0.611]) | 0.477 | 0.500 |
| fixed L5, all questions | 102 / 99 | 0.550 | 0.530 | 0.592 |
| fixed L10 | 102 / 99 | 0.616 | 0.530 | 0.632 |
| fixed L15 | 102 / 99 | 0.660 | 0.530 | 0.666 |
| fixed L20 | 102 / 99 | 0.716 | 0.530 | 0.724 |
| offline full trajectory (not online) | 102 / 99 | 0.615 | 0.530 | 0.647 |

- **At the firing level the two sets look alike.** The medians of P_ans (0.81 vs 0.75), Rel (0.47 vs 0.47), the v4 score (0.33 vs 0.33), level (12 vs 13) and answer changes (1 vs 1) are nearly equal. The best single in-sample AUROC is about 0.58 (rel_min on direct firing). See `p4_firing_level_distributions.csv`.
- **AUROC by level is construction-driven.** It rises from about 0.46 at L1–L3 to 0.72 at L20 (`p4_auroc_by_level.csv`). The 102 become reader-correct as resolution increases while the 99 stay wrong, so late-level separation is "is the reader correct now", not "is this unanswerable".
- **Label noise caps any detector.** §6 shows many of the 99 are label- or referent-ambiguous, which bounds achievable separation.

## Caveats

- **In-sample, many variants.** All target tables pick θ in-sample, and many gates and signals were tried. Only `crossfit.csv` is out-of-fold, and it covers the θ choice, not the choice of variant.
- **Gate (a) depends on the subset construction.** Its benefit on the deployed population (including prior-answerable questions) is unmeasured and likely smaller.
- **Granularity.** 0.95 on 102 allows 5 wrong stops. Mean-level CIs for the P_ans-alone variants are wide (±1.5) because θ sits on a steep region near 0.8.
