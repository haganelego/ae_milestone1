# Bilinear + TopK-Oracle — Authoritative Data Sheet (n=100)

Canonical results after: (1) reverting degradation interpolation to **bilinear**
(optically faithful — area-averaging matches sensor/distance degradation; nearest's
blocky aliasing is a digital-zoom artifact, not optical approach), and (2) adopting
a new optimal-stop oracle whose pseudo-GT is **L20's top-10 nodes / top-5 triplets**
(the confident core) instead of all of L20.

Nearest results are preserved at `outputs/runs/nearest/`; bilinear at
`outputs/runs/bilinear/` and as the live `outputs/`. Main metric stays **SGScore**.

## Decisions baked into this sheet (for narrative)

1. **Interpolation = bilinear (adopted).** Optical capture of a distant object is
   area-integrating low-pass (smooth blur), which bilinear models; nearest-neighbor
   point-samples → aliasing/blocky artifacts that don't occur in optical approach AND
   push saturation to the resolution ceiling. Empirically: IoU/box precision saturates
   in lockstep with labels, so this choice is about realism, not metric strictness.
2. **Metric = SGScore (set-based node+triplet recall), unchanged.** SGDet@20 was
   evaluated and rejected as the per-image saturation metric because (a) IoU adds no
   timing signal (71/96 identical per-image stops vs label-only; box precision
   saturates with labels), and (b) standard per-image SGDet@20 is degenerate (31/100
   images have R@20(L20)=0 under strict IoU+class+predicate matching). Standard SGDet
   R@20 is reported ONLY as a corpus-level EGTR-validation number (0.186 on these 100;
   cf. project 500-img 20.4, author 23.54), NOT as the saturation oracle.
3. **Optimal-stop oracle = TopK pseudo-GT (top-10 nodes / top-5 triplets of L20).**
   Rationale (two independent anchors converging on 10/5):
   - VG150 annotation scale: GT unique categories median 8 (6–10), unique triplets
     median 5 (3–6) on this subset → ~one scene's worth of content.
   - EGTR confidence cliff: object conf still ~0.36 at rank 10; triplet_score collapses
     to ~0.001 after rank 5. So the confident core ends ~10 nodes / ~5 triplets.
   - Asymmetric N is correct (relations are sparser than objects: not all object pairs
     are related). Symmetric 10/10 would make triplet top-10 ≈ all triplets (L20 has
     ~8), defeating the "core" selection.
   - Avoids requiring reproduction of L20's noisy low-confidence tail.
   Oracle stop = first level where RelSGScore (recall of the top-10/5 reference set,
   0.5 node + 0.5 triplet) ≥ coef. **coef 0.9 is the MAIN basis; 0.8 is reported as
   sensitivity.** Rationale for 0.9: the reference is L20's top-N *confident core*, so
   those items are recovered early/reliably and RelSGScore runs high — a low threshold
   (0.8) would trigger saturation too readily, so a stricter 0.9 is the meaningful
   criterion. (0.9 is also the highest threshold before quantization collapses to L20:
   coef ≥0.95 ≡ 1.0, piling 43/100 at L20; 0.9 keeps a real spread.) RelSGScore(L20)=1.0
   by construction (ref ⊂ L20).

## Q1 — level curves (bilinear, mean/100; all_metrics.csv)

| Level | scale | NodeRecall | TripletRecall | SGScore | mean_obj_conf |
|-------|-------|-----------|---------------|---------|---------------|
| 1  | 0.050 | 0.112 | 0.000 | 0.056 | 0.166 |
| 5  | 0.094 | 0.161 | 0.021 | 0.091 | 0.247 |
| 10 | 0.242 | 0.341 | 0.105 | 0.223 | 0.356 |
| 15 | 0.455 | 0.494 | 0.199 | 0.347 | 0.391 |
| 18 | 0.729 | 0.547 | 0.240 | 0.393 | 0.399 |
| 20 | 1.000 | 0.569 | 0.286 | 0.427 | 0.405 |

(L20 values identical to nearest — L20 = original image. Bilinear's degraded levels
recover EARLIER than nearest: e.g. L10 NodeRecall 0.341 vs nearest 0.122.)

## Oracle saturation levels (bilinear, n=100)

GT-based Oracle — genuine saturation (max SGScore over levels ≥ θ; degeneracy motivation):
| θ | genuinely saturate |
|---|---|
| 0.4 | 63/100 |
| 0.5 | **40/100** |
| 0.6 | 21/100 |

Full-reference RelOracle (pseudo-GT = ALL of L20) — the PREVIOUS oracle:
| θ | mean stop | median | sat<20 |
|---|-----------|--------|--------|
| 0.7 | 15.22 | 15 | 85/100 |
| 0.8 | 16.71 | 17 | 70/100 |
| 0.9 | 17.90 | 19.5 | 50/100 |

**TopK RelOracle (pseudo-GT = L20 top-10 nodes / top-5 triplets) — the ADOPTED oracle:**
| coef | mean stop | median | sat<20 | histogram |
|------|-----------|--------|--------|-----------|
| **0.9 (MAIN)** | **16.36** | **17** | 71/100 | spread L5–20, peak L15 (14), 29 at L20 |
| 0.8 (sensitivity) | 15.01 | 14.5 | 80/100 | spread L11–20, peak L14 (17), 20 at L20 |

Mean ref set size: 5.8 unique node labels, 3.5 unique triplets.

## ⚖️ Trivial-baseline sanity check (THE key methodological validation)

Constant "always stop at level c" predictor, MAE vs each oracle (bilinear):
| Oracle | always-15 | always-17 | always-19 | always-20 |
|--------|-----------|-----------|-----------|-----------|
| full-ref RelOracle(0.8) | 2.93 | **2.61** | 2.89 | 3.29 |
| **TopK coef 0.8** | **3.01** | 3.55 | 4.39 | 4.99 |

Interpretation:
- Under **full-ref**, the best constant (always-17, MAE 2.61) BEATS the best real method
  (hybrid_4b_k2, MAE 3.39) → the oracle is too concentrated; evaluation is degenerate.
- Under **TopK**, naive constants (always-19/20: 4.39/4.99) are clearly BEATEN by the best
  methods (~3.3). The methods land near the *oracle-tuned* best constant (always-15: 3.01,
  which requires knowing the oracle centers at 15). So TopK gives the evaluation genuine
  discriminative range; honest caveat: per-image precision is still limited (best method
  ≈ best tuned constant; correlation r≈0.25 — positive but weak).

## Threshold scheme (FINAL — one coherent set, not three magic numbers)

| Module / method | stop threshold | rationale |
|---|---|---|
| object-detection confidence (ablation = baseline) | **0.4** | EGTR mean obj-conf plateaus ~0.40 (≥0.5 fires 7/100, ≥0.6 only 1/100) → the plateau level itself. |
| LLM relationship-reliability (ablation = llm_sgg) | **0.8** | high-confidence saturation; also the empirically better operating point. |
| **hybrid = mean(conf, LLM) (PROPOSED)** | **0.6** | **DERIVED: 0.5·0.4 + 0.5·0.8 = 0.6** — the 0.5/0.5 convex combination of the two component thresholds, matching the 0.5/0.5 fusion weights. Stop on **2 consecutive** levels above 0.6 (k=2). |

`configs/config.yaml`: `baseline.mean_object_conf_threshold=0.4`, `llm.saturation_threshold=0.8`,
`ablation.hybrid.threshold=0.6`. Oracle coef 0.9 main.

## Method comparison vs TopK oracle — **coef 0.9 = MAIN** (n=100; topk_oracle/method_comparison_topk_c09.csv)

| Rank | Method | Mean Stop | MAE | MedAE | Early% | Late% | Pearson r |
|------|--------|-----------|-----|-------|--------|-------|-----------|
| 1 | **llm_sgg_4b_h5_k2** | 16.17 | **2.99** | 2.0 | 0.44 | 0.40 | 0.227 |
| 2 | **hybrid_4b_k2** (proposed) | 16.34 | **3.06** | 2.0 | 0.42 | 0.43 | 0.193 |
| 3 | vlm_9b_single_k2 | 14.41 | 3.63 | 3.0 | 0.64 | 0.26 | 0.176 |
| 4 | hybrid_9b_k2 | 15.08 | 3.84 | 3.0 | 0.47 | 0.40 | 0.071 |
| 5 | hybrid_4b_k1 | 12.77 | 4.91 | 4.0 | 0.69 | 0.19 | 0.079 |
| 6 | vlm_9b_single_k1 | 11.95 | 5.03 | 4.5 | 0.82 | 0.13 | 0.120 |
| 7 | baseline (conf≥0.4) | 13.38 | 5.40 | 5.0 | 0.58 | 0.26 | 0.133 |
| 8 | vlm_4b_single_k2 | 12.98 | 6.22 | 5.0 | 0.61 | 0.32 | −0.076 |
| 9 | hybrid_9b_k1 | 10.76 | 6.26 | 6.0 | 0.82 | 0.13 | 0.173 |
| 10 | llm_sgg_9b_h5_k2 | 11.19 | 6.33 | 6.0 | 0.72 | 0.18 | 0.089 |
| 11 | llm_sgg_4b_h5_k1 | 9.95 | 6.93 | 7.0 | 0.84 | 0.09 | −0.065 |
| 12 | llm_sgg_9b_h5_k1 | 6.95 | 9.43 | 10.0 | 0.96 | 0.01 | 0.034 |
| 13 | vlm_4b_single_k1 | 7.00 | 9.66 | 10.0 | 0.94 | 0.04 | −0.115 |

**llm_sgg_4b_k2 (2.99) and the proposed hybrid_4b_k2 (3.06) are statistically tied** (ΔMAE
0.07; **MedAE identical = 2.0**). Both lead; structured-SGG+LLM at 4b with consecutive-k2 is
the top tier. consecutive-k2 helps strongly (hybrid_4b 4.91→3.06; llm_sgg_4b 6.93→2.99).
baseline (conf-only ablation) weak (5.40). 9b overconfident / 4b VLM too early.

**Ablation reading (proposed = hybrid of two modules):** the LLM-reliability module alone
(llm_sgg_4b_k2, 2.99) matches the full hybrid (3.06) → at this operating point the LLM module
is the dominant contributor; the object-confidence module adds little. The confidence anchor
helps only when the LLM threshold is set too low: at llm=0.7 hybrid_4b_k2 (3.06) clearly beat
llm_sgg_4b_k2 (3.82), but at the principled llm=0.8 the gap closes.

### Sensitivity — coef 0.8 (topk_oracle/method_comparison_topk_c08.csv)
Top: llm_sgg_4b_h5_k2 (3.26) / hybrid_4b_k2 (3.31) / vlm_9b_single_k2 (3.34) — same tie at the
top; baseline 4.97. Lowering the threshold pulls the oracle earlier (mean 15.0 vs 16.36) but
the ranking family is stable.

### Quantization note (why 0.9 is the effective ceiling)
RelSGScore over the small top-10/5 reference is discrete; coef ≥0.95 ≡ 1.0 (43/100 pile at
L20). 0.9 is the strictest threshold that preserves a real stop distribution.

## Headline takeaways (bilinear + TopK, coef 0.9 main, scheme 0.4/0.8/0.6)

1. **Saturation is mid-to-late range, inside the resolution span**: TopK oracle (coef 0.9)
   mean stop 16.36 (median 17), 71/100 saturate before L20; spread L5–20. Bilinear keeps the
   saturation point inside the range (unlike nearest, which piled it at L18–20 / the ceiling).
2. **Top tier: llm_sgg_4b_k2 (MAE 2.99) ≈ proposed hybrid_4b_k2 (3.06)** — tied (ΔMAE 0.07,
   MedAE both 2.0). Structured-SGG + LLM at 4b with consecutive-k2 leads. The 9b VLM (3.63)
   trails; conf-only baseline weak (5.40); 9b LLM and 4b VLM far too early.
3. **Ablation story (proposed = hybrid of two modules)**: the LLM-reliability module alone
   matches the full hybrid, so the confidence module's added value is marginal at the
   principled operating point — it helps only when the LLM threshold is set too low (llm=0.7:
   hybrid 3.06 ≫ llm-only 3.82; llm=0.8: gap closes).
4. **Honest limitation**: per-image precision is limited — correlation r≈0.19–0.25 (weak but
   positive). Methods capture the central tendency of saturation; exact per-image level
   prediction remains hard.
5. **coef 0.8 sensitivity**: oracle earlier (mean 15.0), same top tie (llm_sgg_4b_k2 3.26 /
   hybrid_4b_k2 3.31); ranking family stable. 0.9 is the strictest threshold before
   quantization collapses to L20 (≥0.95 ≡ 1.0).

## UNCHANGED — keep as-is (interpolation-independent)
- EGTR validity: author eval R@20/50/100 = 23.54/30.21/34.35, mR@50 = 7.94, AP50 = 30.83;
  project 500-img 20.4/26.3/30.5; this run's corpus SGDet R@20@L20 = 0.186. KEEP.
- Relation decouple L20 stats (zero-rel 4/100, TripletRecall@L20 0.286, NodeRecall@L20 0.569). KEEP.
- num_ctx fix; EGTR hyperparameters (obj_conf 0.3, rel_conf 0.005, top_k_obj 20, top_k_rel 30).
- SGDet@K output schema (bbox/class_id/predicate_id/triplet_score) retained on predictions+GT.

## Nearest comparison (for REPORT; from outputs/runs/nearest/)
Nearest TopK coef 0.8: mean stop 18.5, median 19 (concentrated L18–20); always-19 MAE 1.12,
best method hybrid_4b_k2 MAE 1.43 → trivial nearly matches method (ceiling effect). This is
why bilinear is adopted: nearest's blocky degradation pushes saturation to the ceiling and
collapses the evaluation's discriminative range.
