# Image + SGG vs SGG-only: why the image makes MIRU stops earlier (zero-call diagnosis)

Scores: 100 images x levels 5-20 per condition. The stop threshold is 0.7.
`stop_decision` is the model's own boolean; its prompt states saturation_score >= 0.8.
Legacy rows come from the MIRU caches: llm_ablation (text, temperature 0.3, older
EGTR predictions) and vlm_scores single (image only, VLM prompt, no graph).

States per condition: img_4b 1600, img_9b 1597, legacy_text_4b 1600, legacy_text_9b 1600, legacy_vlm_img_only_9b 1600, text_4b 1600, text_9b 1600

## 1. Per level

Mean saturation_score:

| level | text_4b | text_9b | img_4b | img_9b | legacy_text_4b | legacy_text_9b | legacy_vlm_img_only_9b |
|---|---|---|---|---|---|---|---|
| 5 | 0.36 | 0.65 | 0.42 | 0.63 | 0.39 | 0.66 | 0.32 |
| 6 | 0.46 | 0.65 | 0.53 | 0.68 | 0.45 | 0.67 | 0.38 |
| 7 | 0.45 | 0.66 | 0.61 | 0.64 | 0.46 | 0.65 | 0.44 |
| 8 | 0.49 | 0.63 | 0.69 | 0.66 | 0.50 | 0.65 | 0.50 |
| 9 | 0.51 | 0.61 | 0.77 | 0.63 | 0.51 | 0.62 | 0.58 |
| 10 | 0.58 | 0.65 | 0.84 | 0.69 | 0.57 | 0.66 | 0.61 |
| 11 | 0.60 | 0.68 | 0.80 | 0.74 | 0.57 | 0.67 | 0.67 |
| 12 | 0.66 | 0.69 | 0.85 | 0.78 | 0.64 | 0.70 | 0.71 |
| 13 | 0.70 | 0.74 | 0.87 | 0.80 | 0.66 | 0.75 | 0.74 |
| 14 | 0.73 | 0.75 | 0.88 | 0.84 | 0.70 | 0.78 | 0.80 |
| 15 | 0.79 | 0.78 | 0.88 | 0.86 | 0.77 | 0.79 | 0.84 |
| 16 | 0.80 | 0.79 | 0.87 | 0.86 | 0.78 | 0.80 | 0.87 |
| 17 | 0.82 | 0.82 | 0.88 | 0.88 | 0.82 | 0.83 | 0.88 |
| 18 | 0.84 | 0.84 | 0.89 | 0.88 | 0.83 | 0.85 | 0.92 |
| 19 | 0.85 | 0.84 | 0.89 | 0.89 | 0.83 | 0.84 | 0.92 |
| 20 | 0.80 | 0.82 | 0.90 | 0.90 | 0.80 | 0.84 | 0.94 |
| mean | 0.65 | 0.72 | 0.79 | 0.77 | 0.64 | 0.73 | 0.70 |

Fraction with score >= 0.7, the share that triggers a k=1 stop:

| level | text_4b | text_9b | img_4b | img_9b | legacy_text_4b | legacy_text_9b | legacy_vlm_img_only_9b |
|---|---|---|---|---|---|---|---|
| 5 | 11% | 43% | 27% | 45% | 17% | 45% | 2% |
| 6 | 7% | 35% | 31% | 41% | 7% | 36% | 3% |
| 7 | 11% | 38% | 44% | 44% | 12% | 40% | 3% |
| 8 | 12% | 34% | 58% | 47% | 16% | 38% | 6% |
| 9 | 16% | 29% | 74% | 48% | 16% | 33% | 14% |
| 10 | 27% | 34% | 85% | 59% | 29% | 41% | 24% |
| 11 | 34% | 45% | 80% | 64% | 34% | 47% | 29% |
| 12 | 53% | 53% | 88% | 75% | 48% | 52% | 40% |
| 13 | 58% | 60% | 94% | 72% | 51% | 62% | 51% |
| 14 | 59% | 67% | 95% | 88% | 56% | 71% | 68% |
| 15 | 77% | 76% | 97% | 91% | 76% | 77% | 84% |
| 16 | 81% | 80% | 95% | 93% | 80% | 78% | 91% |
| 17 | 83% | 85% | 96% | 97% | 82% | 86% | 90% |
| 18 | 90% | 91% | 99% | 96% | 88% | 92% | 97% |
| 19 | 91% | 91% | 97% | 97% | 90% | 91% | 95% |
| 20 | 80% | 89% | 98% | 95% | 84% | 87% | 100% |
| mean | 49% | 59% | 79% | 72% | 49% | 61% | 50% |

Fraction with stop_decision = true, the model's own flag:

| level | text_4b | text_9b | img_4b | img_9b | legacy_text_4b | legacy_text_9b | legacy_vlm_img_only_9b |
|---|---|---|---|---|---|---|---|
| 5 | 11% | 43% | 27% | 44% | 17% | 45% | 1% |
| 6 | 7% | 35% | 31% | 39% | 6% | 36% | 3% |
| 7 | 7% | 29% | 44% | 36% | 9% | 29% | 2% |
| 8 | 9% | 22% | 54% | 32% | 10% | 27% | 3% |
| 9 | 13% | 16% | 68% | 28% | 9% | 20% | 8% |
| 10 | 21% | 21% | 81% | 44% | 17% | 25% | 11% |
| 11 | 28% | 20% | 75% | 40% | 23% | 24% | 15% |
| 12 | 42% | 25% | 86% | 53% | 30% | 29% | 21% |
| 13 | 48% | 35% | 87% | 56% | 34% | 37% | 28% |
| 14 | 55% | 38% | 90% | 72% | 45% | 50% | 41% |
| 15 | 65% | 50% | 88% | 76% | 58% | 54% | 54% |
| 16 | 71% | 51% | 86% | 78% | 66% | 61% | 60% |
| 17 | 75% | 67% | 90% | 82% | 71% | 73% | 71% |
| 18 | 77% | 67% | 89% | 81% | 70% | 72% | 85% |
| 19 | 84% | 70% | 94% | 88% | 77% | 73% | 87% |
| 20 | 76% | 63% | 95% | 90% | 68% | 72% | 95% |
| mean | 43% | 41% | 74% | 59% | 38% | 45% | 37% |

## 2. Paired difference img - text (same model, same state)

| level | 4b: mean d / img>=thr & text<thr / reverse | 9b: mean d / img>=thr & text<thr / reverse |
|---|---|---|
| 5 | +0.06 / 19% / 3% | -0.02 / 10% / 8% |
| 6 | +0.07 / 28% / 4% | +0.03 / 10% / 4% |
| 7 | +0.16 / 34% / 1% | -0.01 / 15% / 9% |
| 8 | +0.21 / 48% / 2% | +0.03 / 19% / 6% |
| 9 | +0.27 / 58% / 0% | +0.01 / 27% / 8% |
| 10 | +0.26 / 60% / 2% | +0.05 / 30% / 5% |
| 11 | +0.21 / 46% / 0% | +0.06 / 24% / 5% |
| 12 | +0.18 / 35% / 0% | +0.09 / 23% / 1% |
| 13 | +0.17 / 36% / 0% | +0.06 / 17% / 5% |
| 14 | +0.15 / 37% / 1% | +0.09 / 23% / 2% |
| 15 | +0.09 / 21% / 1% | +0.07 / 16% / 1% |
| 16 | +0.06 / 16% / 2% | +0.07 / 11% / 0% |
| 17 | +0.06 / 13% / 0% | +0.06 / 12% / 0% |
| 18 | +0.05 / 9% / 0% | +0.05 / 5% / 0% |
| 19 | +0.04 / 6% / 0% | +0.05 / 6% / 0% |
| 20 | +0.09 / 18% / 0% | +0.07 / 7% / 1% |
| all (4b) | mean d +0.134, median +0.02; img>=thr & text<thr 30.2% (484); reverse 1.0% (16); Spearman(img,text) 0.36 | |
| all (9b) | mean d +0.048, median +0.00; img>=thr & text<thr 16.0% (255); reverse 3.4% (55); Spearman(img,text) 0.55 | |

## 3. Reasons where img says saturated (>= thr) at level <= 8 but text does not

### 4b: 129 such states (levels 5-8)

- [2315531 t=6] img 1.00: "The scene graph has reached semantic saturation as the object and relationship sets have stabilized across resolution levels with no new categories or relationships detected."
  - text 0.12: "The scene graph has not stabilized; new object categories (building) and relationships appear at level 4, followed by additional distinct entities (person riding motorcycle, man wearing shirt) appearing in subsequent levels with varying IDs and labels."
- [2315531 t=7] img 1.00: "The scene graph has reached semantic saturation as the object and relationship sets have stabilized across resolution levels with no new categories or relationships detected."
  - text 0.12: "The scene graph shows significant instability across resolution levels with fluctuating object counts (from empty to one), inconsistent relationship sets involving different subjects and predicates, and no convergence in detected categories or structures."
- [2315531 t=8] img 0.95: "The scene graph has reached semantic saturation as the object and relationship sets have stabilized across resolution levels (Level 7 onwards show no new objects or relationships), with confidence scores remaining consistent."
  - text 0.15: "The scene graph shows significant instability between levels (e.g., object sets changing from non-empty to empty), and confidence scores remain very low across all entries, indicating the detection process has not converged."

### 9b: 54 such states (levels 5-8)

- [2316000 t=8] img 0.95: "The scene graph has reached semantic saturation: object categories (bus, wheels, windows, poles, buildings) and relationships have stabilized across levels; confidence scores are consistent with no new meaningful objects or relations emerging at higher resolution."
  - text 0.65: "New object categories (street, sidewalk, tree) and relationships have appeared at level 8 compared to previous levels. While confidence scores are low for some new detections, the semantic content is still evolving with the introduction of environmental context objects."
- [2316954 t=6] img 0.95: "The scene graph has reached saturation: object categories have stabilized (only 'window' detected at levels 4-5), relationships remain empty throughout, and confidence scores are low but consistent. Further resolution increases yield no new meaningful detections or structural changes."
  - text 0.20: "The scene graph is unstable and has not converged; objects appear at level 4 but disappear completely at level 6, indicating that the detection process is still fluctuating with resolution changes rather than stabilizing."
- [2316954 t=8] img 0.95: "The scene graph has stabilized with consistent object categories (beach, wave) and relationships across levels 7-8. Confidence scores are converging without significant changes in structure or new meaningful objects appearing."
  - text 0.65: "The scene graph is not yet saturated because the object count and categories are still changing significantly between levels (e.g., level 7 introduces 'wheel' and multiple instances of 'beach', while level 8 adds many more 'wave' objects). Additionally, new relationships like 'wearing shirt' and 'carrying surfboard' appear at higher resolutions. Confidence scores also fluctuate substantially across these changes."

Broad keyword rates over ALL reasons, shown as image-words / sequence-words (share of reasons with >= 1 hit).
These lists are too broad: 'resolution', 'appear' and 'levels' also occur in text-only reasons. The refined analysis follows below.

- text_4b: image-words 91% | sequence-words 100% | 'image' literally 0%
- text_9b: image-words 76% | sequence-words 100% | 'image' literally 1%
- img_4b: image-words 99% | sequence-words 99% | 'image' literally 6%
- img_9b: image-words 97% | sequence-words 98% | 'image' literally 11%
- legacy_text_4b: image-words 90% | sequence-words 100% | 'image' literally 1%
- legacy_text_9b: image-words 75% | sequence-words 100% | 'image' literally 1%
- legacy_vlm_img_only_9b: image-words 100% | sequence-words 41% | 'image' literally 38%

## 4. Score distributions (share of states per bin)

| cond | 0-0.2 | 0.2-0.4 | 0.4-0.5 | 0.5-0.6 | 0.6-0.7 | 0.7-0.8 | 0.8-0.9 | 0.9-1.0 | distinct values |
|---|---|---|---|---|---|---|---|---|---|
| text_4b | 5% | 11% | 21% | 0% | 14% | 6% | 22% | 21% | 28 |
| text_9b | 4% | 3% | 7% | 2% | 23% | 19% | 14% | 27% | 23 |
| img_4b | 7% | 4% | 5% | 0% | 4% | 5% | 19% | 55% | 37 |
| img_9b | 7% | 4% | 2% | 5% | 11% | 12% | 10% | 49% | 25 |
| legacy_text_4b | 5% | 13% | 20% | 0% | 13% | 12% | 17% | 20% | 30 |
| legacy_text_9b | 4% | 5% | 6% | 2% | 22% | 16% | 15% | 31% | 24 |
| legacy_vlm_img_only_9b | 3% | 20% | 1% | 8% | 18% | 13% | 0% | 37% | 12 |

## 5. Refined reason analysis (narrow regexes)

- **visual** is /image|visual(ly)|blur*|sharp*|pixel*|photo*|picture|clarity/.
- **claim stable** is /stabiliz*|stable|consistent*|converg*/.
- **claim unstable** is /unstable|instability|fluctuat*|new object|new categor*|not (yet) stabiliz/converg*|still changing/.
- A "flip" is a state where img >= 0.70 and text < 0.70, for the same model and the same SGG sequence.

```
4b all states n=1600: img reasons visual-words 6% | claim stable 88% | claim unstable 27% ;; text reasons: visual 0% stable 81% unstable 78%
4b flip (img>=0.7, text<0.7) n=484: img reasons visual-words 1% | claim stable 99% | claim unstable 19% ;; text reasons: visual 1% stable 59% unstable 95%
   VIS-EX 4b (2316770, 10): img 0.85 "The scene graph has reached semantic saturation as the object categories (zebra parts like head, ear, leg) and relationships have stabilized across levels 9-10 with consistent confidence scores around 2-3%. The primary entity is i"
   VIS-EX 4b (2325568, 9): img 0.95 "The scene graph has reached semantic saturation as the object set and relationships have stabilized across resolution levels (Level 8 to Level 9). The primary objects detected are 'pole' and 'light', with no new categories appeari"
   VIS-EX 4b (2329268, 8): img 0.95 "The scene graph has reached semantic saturation as all levels from level 4 onwards show no objects detected and only a single low-confidence relationship at level 5 with very weak evidence (confidence 0.01). The lack of consistent"
9b all states n=1597: img reasons visual-words 12% | claim stable 82% | claim unstable 55% ;; text reasons: visual 2% stable 90% unstable 80%
9b flip (img>=0.7, text<0.7) n=255: img reasons visual-words 7% | claim stable 89% | claim unstable 45% ;; text reasons: visual 1% stable 80% unstable 95%
   VIS-EX 9b (2319832, 7): img 0.95 "The scene graph has stabilized with only one object detected ('beach') and no relationships across all resolution levels; further increases in resolution are unlikely to reveal meaningful new objects or relationships given the ima"
   VIS-EX 9b (2320110, 5): img 0.95 "The scene graph has reached semantic saturation as the object categories have stabilized (only 'train' and unrelated detections like 'hat/head') with low confidence scores that are not significantly changing across levels. The rel"
   VIS-EX 9b (2321244, 8): img 1.00 "The scene graph has reached semantic saturation: object categories (dog parts like paw, mouth, nose, head) and relationships have stabilized across levels; confidence scores are low but consistent with the blurry image quality; no"
legacy VLM image-only 9b: visual-words 82% stable 0%
```

## Conclusions

- **The image makes the model say "saturated" much more often, mainly for 4b.** The mean paired shift
  img - text is +0.13 for 4b and +0.05 for 9b. Flips (img >= 0.70 while text < 0.70) occur in 30% of
  4b states and 16% of 9b states; the reverse occurs in only 1% and 3%. The shift peaks at the mid
  levels, t = 8-11: +0.21 to +0.27 for 4b, with 48-60% of states flipping. These are exactly the
  levels where MIRU's stop has to wait.
- **The hypothesis is not supported in its verbal form.** In flipped states the image+SGG model
  almost never cites the image: visual words appear in 1% (4b) and 7% (9b) of reasons, against 82%
  for the image-only VLM. Instead it asserts that the SGG sequence is stable (99% / 89%), while the
  text-only model on the identical sequence calls it unstable (95%). Typical case: 2315531 t = 6-8.
  The text-only reason says object sets go from non-empty to empty; the image reason says "sets
  have stabilized ... no new categories".
- **The stability claims are confabulated.** The attached image shifts the model toward a
  "saturated" prior, and the reason rationalizes it as graph stability that the graph does not
  show. Reasons are post hoc, so the mechanism (image tokens diluting attention to the long JSON
  sequence, or a "this looks like a complete scene" prior) cannot be identified from text alone.
- **The blur cue is lost.** Asked about the image alone, the same 9b VLM is level-sensitive: 2% of
  states score >= 0.70 at t = 5 and 14% at t = 9. With image + SGG, 45% / 48% score >= 0.70 at t = 5
  and t = 9 (4b: 27% / 74%). The combined prompt makes the model judge the graph and stop judging
  the image.
- **Score distributions shift toward the top.** For img_4b, 55% of scores fall in 0.9-1.0, against
  21% for text_4b. For img_9b it is 49% against 27%. The legacy text caches reproduce the new text
  runs closely (per-level means within about 0.04), so the text-only control is valid.
- **Failures.** img_9b has 3 failed states. They have no score file and count as 0 under the MIRU
  rule.
