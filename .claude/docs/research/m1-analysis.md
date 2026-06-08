# Milestone-1 SGG-Saturation PoC: Error & Saturation Analysis

**Date**: 2026-06-05
**Scope**: 10-image Visual Genome subset, 20 resolution levels, category-set matching semantics.

---

## (a) What Tends to Be Missing When Each Method Stops

Three stop methods compared: **oracle** (SGScore >= 0.5), **baseline** (mean object confidence >= 0.4), **llm** (LLM saturation scoring).

| Method   | Mean Stop Level | Mean Missing Nodes | Mean Missing Triplets | Mean Extra Nodes | Mean Extra Triplets |
|----------|:-:|:-:|:-:|:-:|:-:|
| oracle   | 16.0 | 3.5 | 4.2 | 2.1 | 2.0 |
| baseline | 12.0 | 4.6 | 4.7 | 1.6 | 1.8 |
| llm      | 10.2 | 4.6 | 4.8 | 1.2 | 1.1 |

**Key findings:**

- **No method achieves zero missing triplets on any image.** Even oracle (which waits the longest, mean L16) leaves 4.2 missing triplets on average. This indicates that the fundamental EGTR detector fails to produce many GT triplets even at full resolution.
- **The LLM method stops earliest (mean L10.2) but has similar residual missing counts to baseline**, because the missing items are detector-ceiling gaps, not information still arriving.
- **6/10 images under LLM (and 5/10 under baseline) have ALL GT triplets missing at the stop level.** These are images where the EGTR model never predicts the correct triplets at any level.
- **Commonly missing node categories at stop**: `person`, `coat`, `cap`, `hat`, `glass`, `letter`, `pizza`, `street`, `pant` -- clothing items, small objects, and scene-level labels.
- **Commonly missing triplets involve predicates**: `wearing`, `carrying`, `standing on`, `walking on`, `sitting on`, `in` -- action and spatial relation predicates that require both correct subject/object detection AND relation prediction.
- The **extra (false positive) items are moderate** (1-2 on average), indicating EGTR has reasonable precision but poor recall for the GT vocabulary.

---

## (b) Hardest Object Categories & Predicates

### Object Categories

Out of **43 unique GT categories** across the 10 images:

- **15 categories (35%) are NEVER recalled at any level** (L1-L20): `beach`, `box`, `cap`, `coat`, `glass`, `hat`, `letter`, `light`, `mouth`, `person`, `pizza`, `pole`, `post`, `sign`, `street`.
- These are predominantly **small/wearable objects** (cap, glass, hat, coat), **scene parts** (street, beach), and **abstract labels** (person vs. man/woman distinction, letter, sign).
- Categories with **partial recall**: `pant` (0.33), `head` (0.33), `window` (0.50), `short` (0.50), `shirt` (0.60), `hair` (0.67).
- **Always recalled**: `elephant`, `horse`, `building`, `bench`, `door`, `finger`, `kite`, `sidewalk`, `skateboard`, `table`, `tree`, `truck`, `woman`, `eye`, `leg`, `neck`, `nose`, `tail`, `trunk`.

The pattern: EGTR reliably detects **large, salient foreground objects** but fails on **small accessories, clothing, scene-level labels, and VG-specific fine-grained categories**.

### Predicates

Out of **13 unique GT predicates**:

- **9 predicates (69%) are NEVER recalled at L20**: `attached to`, `between`, `carrying`, `in`, `near`, `sitting on`, `standing on`, `walking on`, `with`.
- Only **4 predicates ever produce correct triplets**: `of` (0.50 recall), `on` (0.36), `has` (0.33), `wearing` (0.09).
- This severe predicate recall failure is the dominant bottleneck: even when both subject and object nodes are detected, the EGTR model rarely predicts the correct predicate label for the GT triplet.

---

## (c) What Characterizes Hard / Never-Saturating Images

**6 of 10 images never reach oracle threshold (SGScore >= 0.5) and have oracle_stop=20.**

| Image ID | GT Objects | GT Unique Triplets | Frac Small | Final SGScore | Final NodeR | Final TripR | Oracle Stop |
|----------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2322708 | 23 | 6 | 0.13 | 0.250 | 0.50 | 0.00 | 20 |
| 2331313 | 7 | 4 | 0.00 | 0.250 | 0.50 | 0.00 | 20 |
| 2334997 | 28 | 12 | 0.32 | 0.275 | 0.47 | 0.08 | 20 |
| 2333129 | 11 | 5 | 0.27 | 0.314 | 0.43 | 0.20 | 20 |
| 2341363 | 8 | 3 | 0.13 | 0.333 | 0.67 | 0.00 | 20 |
| 2335659 | 19 | 6 | 0.21 | 0.397 | 0.46 | 0.33 | 20 |

**Correlation with difficulty (Pearson r with final_sg_score):**

| Feature | r with SGScore | r with Oracle Stop |
|---------|:-:|:-:|
| num_gt_objects | -0.431 | +0.500 |
| num_gt_relationships | -0.385 | +0.423 |
| num_gt_unique_triplets | -0.246 | +0.246 |
| num_gt_object_categories | -0.212 | +0.341 |
| frac_small_objects | +0.106 | +0.377 |
| mean_obj_area_frac | +0.044 | -0.169 |

**Interpretation:**

- **More GT objects and relationships correlate with lower SGScore** (r = -0.43 / -0.39). Complex scenes are harder for EGTR.
- **Higher fraction of small objects correlates with later oracle stops** (r = +0.38), meaning small objects delay saturation.
- The two hardest images (2322708 with 23 GT objects and 2334997 with 28 GT objects) have the most complex scenes with many categories (8 and 15 respectively) and high triplet counts (6 and 12).
- Image 2341363 is unusual: only 3 unique categories but 0% triplet recall because its GT predicates (`attached to`, `on`, `has` involving `finger`/`person`) are never predicted by EGTR.
- Image 2331313: no small objects but all GT predicates are `wearing`/`sitting on` which EGTR never predicts correctly.

**Root cause for never-saturating images is a mix of**: (1) too many GT categories for EGTR to cover, (2) GT predicates that EGTR systematically fails to predict, and (3) VG-specific category distinctions (person vs man) that EGTR conflates.

---

## (d) Node-vs-Relation Saturation Gap

| Image | MaxNodeR | MaxTripR | Node 90% Level | Trip 90% Level | Gap |
|-------|:-:|:-:|:-:|:-:|:-:|
| 2318597 | 0.75 | 0.50 | 6 | 6 | 0 |
| 2321559 | 0.88 | 0.60 | 12 | 9 | -3 |
| 2322708 | 0.75 | 0.17 | 3 | 9 | +6 |
| 2329909 | 0.86 | 0.60 | 20 | 19 | -1 |
| 2331313 | 0.50 | 0.00 | 20 | 20 | 0 |
| 2332990 | 1.00 | 0.40 | 12 | 7 | -5 |
| 2333129 | 0.71 | 0.40 | 18 | 15 | -3 |
| 2334997 | 0.60 | 0.17 | 16 | 17 | +1 |
| 2335659 | 0.54 | 0.33 | 12 | 11 | -1 |
| 2341363 | 0.67 | 0.00 | 1 | 20 | +19 |

**Mean gap: +1.3 levels** (nodes reach 90% of their max recall 1.3 levels before triplets, on average).

However, the **variance is high** (std ~ 7 levels), driven by:
- Image 2341363: extreme +19 gap (nodes detected at L1 but triplets never detected).
- Several images show **negative gaps** (triplets saturate before nodes), which occurs when the few detectable triplets appear early but additional node categories keep appearing at higher levels.

The practical takeaway: **node saturation does NOT reliably predict relation saturation**. A stop criterion based purely on node convergence would be insufficient; relation-aware scoring (as in SGScore) is necessary.

---

## Summary of Output Files

### CSV Outputs (outputs/analysis/)
- `missing_at_stop.csv` -- per-(image, method) missing/extra nodes and triplets at the chosen stop level
- `category_recall_by_level.csv` -- per-(category, level) recall fraction across images
- `predicate_recall_by_level.csv` -- per-(predicate, level) recall fraction across images
- `image_features.csv` -- per-image difficulty features (GT counts, areas, final scores)
- `saturation_gap.csv` -- per-image node vs triplet saturation levels and gap

### Figures (outputs/figures/)
- `category_recall_hardest.png` -- recall curves for the 10 hardest object categories
- `predicate_recall_hardest.png` -- recall curves for the 10 hardest predicates
- `difficulty_vs_features.png` -- scatter plots of SGScore vs image features (small objects, triplet count, category count)
