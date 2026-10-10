# gqa_stop v22 test (plan §21/§22/§23 protocol): frozen dev-102 θ on the test visual subset (251) and reader-wrong test set (233)

Status: **skeleton written before any test evaluation.** Written 2026-10-10T07:45:06Z, at HEAD d2c7161.
- **Test inference has not completed.** It is paused because ollama fell back to CPU after an NVIDIA driver update; see the worklog.
- **No test metric has been computed.** No test method table exists yet.

Run: `configs/gqa_stop_v22.json`, output `outputs/gqa_stop/visapp_gqa_v22/`. A13 clean decoding, think=false.
Frozen policy: `frozen_policy.json`, sha256 `05a84b224ba8a5886b7d497e2628a764fec4ae453379fa39d7d9bcafcb4e2bf5`, read-only. θ was copied from the §22 dev-102 grid (`dev_targets.csv`) and verified by recomputation.

| target | checklist_v4_min | direct_v4 | checklist_v1_min | direct verbalised | direct_pans (logged) | pans_min (logged) | Fixed-dev |
|---|---|---|---|---|---|---|---|
| 0.90 | 0.14 | 0.22 | 0.90 | 0.95 | 0.78 | 0.53 | L16 |
| **0.95** | **0.28** | **0.30** | **0.90** | **0.95** | **0.83** | **0.58** | **L20** |
| 0.97 | 0.36 | 0.36 | 1.00 | 0.95 | 0.85 | 0.78 | L20 |
| 0.99 | 0.42 | 0.39 | 1.00 | ∞ | 0.88 | 0.84 | L20 |

CV caveat, recorded in the frozen file: dev single-θ selection loses about 0.01–0.02 accuracy out of fold. direct_v4 at the 0.95 target gives 0.934 out of fold (`box_attrtheta_cv.csv`).

## Hypotheses

### Pre-registered before dev (plan §19/§21/§22/§23)

The verdict paragraphs will be written after test evaluation.
- **Main.** checklist_v4_min reaches accuracy ≥ 0.95 at a lower mean level than fair v1 (checklist_v1_min) and than Fixed-dev, with the paired image-cluster bootstrap CI below 0.
- **Comparator.** checklist_v4_min vs direct_v4: mean-level and accuracy difference. Dev §22 showed no difference.
- **§23 firing.** The firing rate on the 233 reader-wrong test questions at the frozen θ, compared between methods.

### Dev-derived hypothesis H_pans (formed after seeing dev-99 firing, pre-registered here before test evaluation)

- **Origin.** The cache-only probes (commit d876011, `gqa-stop-unanswerable-probes.md`) found a pattern on dev, and it held out of fold. P_ans alone fires far less on reader-wrong questions than P_ans × Rel at equal accuracy and mean level. P_ans alone is the logged direct_pans / pans_min variants; Rel is the self-assessed confidence.
- **Why it is pre-registered, but weaker.** This hypothesis was formed **after** looking at the dev-99 firing results, so it is not part of the original pre-registration. It is registered here before any test metric is computed. Both variants were already frozen as logged variants, with θ selected on dev 102 in the same grid; nothing is re-tuned.
- **H_pans.** On test, at their frozen 0.95 θ (direct_pans 0.83, pans_min 0.58), direct_pans and pans_min have a **lower firing rate on the 233 reader-wrong test questions** than direct_v4 (θ 0.30) and checklist_v4_min (θ 0.28).
  - Test: paired image-cluster bootstrap, 2000 draws, seed 20261008; the upper CI bound of the firing-rate difference must be < 0.
  - Condition: their accuracy on the 251 test visual questions must be **≥ 0.95 − 0.02 = 0.93**.
- **Rows reported.** direct_pans and pans_min rows appear in every test table and in the 233-question firing table, with paired bootstrap against direct_v4 and checklist_v4_min.

## Results

Pending test inference and the single evaluation.
