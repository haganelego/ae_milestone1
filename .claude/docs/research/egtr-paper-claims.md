# EGTR Paper: Published Claims & Evaluation Protocol Reference

**Paper**: "EGTR: Extracting Graph from Transformer for Scene Graph Generation"
**Venue**: CVPR 2024 (Best Paper Award Candidate)
**arXiv**: 2404.02072 (v1: Apr 2024, latest: v5)
**Authors**: Jinbae Im, JeongYeon Nam, Nokyung Park, Hyungmin Lee, Seunghyun Park (NAVER AI Lab)
**Code**: https://github.com/naver-ai/egtr

---

## 1. Visual Genome SGDet Results (Graph-Constraint) -- Table 1 (Main Paper)

### EGTR (Deformable-DETR + ResNet-50, 200 queries)

| Variant              | Params | FPS  | AP50 | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 |
|----------------------|--------|------|------|------|------|-------|-------|-------|--------|
| **EGTR (baseline)**  | 42.5M  | 14.7 | 30.8 | 23.5 | 30.2 | 34.3  | 5.5   | 7.9   | 10.1   |
| EGTR (LA, tau=0.3)   | 42.5M  | 14.7 | 30.8 | 22.4 | 28.2 | 31.7  | 8.8   | 14.0  | 18.3   |
| EGTR (LA, tau=0.5)   | 42.5M  | 14.7 | 30.8 | 19.7 | 24.2 | 26.7  | 11.0  | 17.1  | 21.4   |
| EGTR (LA, tau=0.7)   | 42.5M  | 14.7 | 30.8 | 15.7 | 18.7 | 20.5  | 12.1  | 17.8  | 21.7   |

> LA = Logit Adjustment (post-hoc debiasing). Baseline = no debiasing. AP50 is unchanged by LA (object detector is the same).

### Key Comparison Methods (SGDet, Graph-Constraint)

| Model              | Type     | Params  | FPS  | AP50 | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 |
|--------------------|----------|---------|------|------|------|------|-------|-------|-------|--------|
| IMP (EBM)          | 2-stage  | 322.2M  | 2.0  | 28.1 | 18.1 | 25.9 | 31.2  | 2.8   | 4.2   | 5.4    |
| VTransE            | 2-stage  | 312.3M  | 3.5  | --   | 24.5 | 31.3 | 35.5  | 5.1   | 6.8   | 8.0    |
| Motifs             | 2-stage  | 369.9M  | 1.9  | 28.1 | 25.1 | 32.1 | 36.9  | 4.1   | 5.5   | 6.8    |
| VCTree             | 2-stage  | 361.5M  | 0.8  | 28.1 | 24.8 | 31.8 | 36.1  | 4.9   | 6.6   | 7.7    |
| VCTree (TDE)       | 2-stage  | 361.3M  | 0.8  | 28.1 | 14.0 | 19.4 | 23.2  | 6.9   | 9.3   | 11.1   |
| VCTree (EBM)       | 2-stage  | 372.5M  | --   | 28.1 | 24.2 | 31.4 | 35.9  | 5.7   | 7.7   | 9.1    |
| BGNN               | 2-stage  | 341.9M  | 1.7  | 29.0 | 23.3 | 31.0 | 35.8  | 7.5   | 10.7  | 12.6   |
| FCSGG              | 1-stage  | 87.1M   | 6.0  | 28.5 | 16.1 | 21.3 | 25.1  | 2.7   | 3.6   | 4.2    |
| RelTR              | 1-stage  | 63.7M   | 13.4 | 26.4 | 21.2 | 27.5 | --    | 6.8   | 10.8  | --     |
| SGTR               | 1-stage  | 117.1M  | 6.2  | 25.4 | --   | 24.6 | 28.4  | --    | 12.0  | 15.2   |
| Relationformer     | 1-stage  | 92.9M   | 8.5  | 26.3 | 22.2 | 28.4 | 31.3  | 4.6   | 9.3   | 10.7   |
| Iterative SGG      | 1-stage  | 93.5M   | 6.0  | 27.7 | --   | 29.7 | 32.1  | --    | 8.0   | 8.8    |
| SSR-CNN            | 1-stage  | 274.3M  | 4.0  | 23.8 | 25.8 | 32.7 | 36.9  | 6.1   | 8.4   | 10.0   |
| SSR-CNN (LA,t=0.3) | 1-stage  | 274.3M  | 4.0  | 23.8 | 18.4 | 23.3 | 26.5  | 13.5  | 17.9  | 21.4   |

---

## 2. PredCls & SGCls Results -- Table 9 (Supplementary)

Comparison with one-stage SGG models. Format: PredCls / SGCls / SGDet.

| Model         | AP50 | R@50                | mR@50              |
|---------------|------|---------------------|--------------------|
| FCSGG         | 28.5 | 41.0 / 23.5 / 21.3 | 6.3 / 3.7 / 3.6   |
| RelTR         | 26.4 | *36.0 / 30.5 / 25.2* | *10.8 / 9.3 / 8.5* |
| **EGTR (Ours)** | **30.8** | **54.3 / 39.8 / 30.2** | **16.6 / 11.9 / 7.9** |

> *Italic* for RelTR = re-measured using publicly available checkpoint (ground truth objects used rather than ground truth triplets, for fairness).

---

## 3. Object Detection Performance

| Model          | AP50  | AP50_rel | AP50_no-rel |
|----------------|-------|----------|-------------|
| Iterative SGG  | 27.7  | 24.3     | 7.8         |
| SSR-CNN        | 23.8  | 20.2     | 7.4         |
| **EGTR**       | **30.8** | **24.3** | **10.7**    |

> AP50_no-rel = AP50 for objects without any relations (42%+ of VG objects). EGTR excels here.

### ResNet-101 backbone ablation (Supplementary Sec. 4.3):
- AP50: 32.3 (+1.5), R@50: 30.8 (+0.6), mR@50: 8.1 (+0.2)

---

## 4. Unconstrained (No Graph Constraint) Results

The main paper reports ONLY graph-constraint results. However, the supplementary mentions ng-R@50 in the context of applying adaptive smoothing to other models:
- Relationformer with smoothing: ng-R@50 = 28.84 (+0.76)

**EGTR's own ng-R@K values are NOT reported** in either the main paper or supplementary.

---

## 5. Evaluation Protocol Details

### 5.1 Dataset: Visual Genome (VG150 Split)

- **Split**: Following Xu et al. [17] (CVPR 2017, "Scene graph generation by iterative message passing")
- **Object categories**: 150 most frequent
- **Predicate categories**: 50 most frequent
- **Training images**: 57,723 (57K)
- **Validation images**: 5,000 (5K)
- **Test images**: 26,443 (26K)

### 5.2 Task: SGDet (Scene Graph Detection)

From Supplementary Section 2.1:
> "Among various evaluation methods for the scene graph generation task, including scene graph detection (SGDET), scene graph classification (SGCLS), and predicate classification (PRDCLS) [17], we adopt the SGDET evaluation protocol due to its rigorous and comprehensive nature compared to other methods. Unlike SGCLS or PRDCLS, where ground truth categories or box coordinates of objects are given, SGDET evaluates the performance of entity categories and box coordinates for each subject and object, as well as predicate categories collectively."

### 5.3 IoU Threshold

From Supplementary Section 2.1:
> "We use the widely adopted value of 50% as the thresholding parameter for IoU incorporated in SGDET."

**A predicted triplet is correct when:**
1. Subject box IoU >= 0.5 with a GT subject box
2. Object box IoU >= 0.5 with a GT object box
3. Subject class label matches GT
4. Object class label matches GT
5. Predicate class label matches GT

### 5.4 Graph Constraint

From Supplementary Section 2.1:
> "We also adopt the graph constraint evaluation method proposed in Zellers et al. [18], which enforces a limit of one predicted predicate between a given subject and object entity. We select the top 1 predicate for each object pair, as determined by multiplying the predicate score G_ijk by the connectivity score E_ij and multiplying the corresponding class scores of the subject v_i^c and object entity v_j^c."

**Meaning**: For each (subject, object) pair, only the single highest-scoring predicate is kept. This is the standard "with constraint" / "graph constraint" setting.

**No graph constraint (ng)**: Allows multiple predicates per object pair (as long as predicates differ). Not evaluated for EGTR in the paper.

### 5.5 Recall@K and Mean Recall@K

From Supplementary Section 2.1:
> "We use recall at k (R@k) and mean recall at k (mR@k) [14] as evaluation metrics."

**R@K**: Top-K scored triplets per image; compute fraction of GT relations matched; average over images.
- Formula: R@k = |G ∩ X_k| / |G| where G = GT triplets, X_k = top-K predictions
- K counts **triplets** (not pairs)
- K ∈ {20, 50, 100}

**mR@K**: Average of per-predicate R@K across all 50 predicate categories (equal weight per predicate).
- Formula: mR@k = (1/|P|) * sum_p R@k^(p)
- Compensates for long-tail bias by giving rare predicates equal influence

---

## 6. Long-Tail Predicate Distribution

### From Supplementary Figure 1:
The predicate distribution histogram (sorted by training frequency) shows:
- **"on"** dominates at ~40% of GT relations
- Next predicates: "has", "wears/wearing", "near", "in", etc.
- Many tail predicates ("walking on", "between", "walking in", "flying in") have near-zero frequency

### From Main Paper (Section 4.1):
> "mR@k is a balanced version of R@k in that mR@k can compensate for the bias of predicate categories by aggregating for all predicate categories."

### Implication for R@K vs mR@K gap:
- R@K is dominated by frequent predicates ("on", "has", "wearing", "near", "in") which are easy to predict
- mR@K gives equal weight to all 50 predicates, including tail predicates with near-zero recall
- **This explains why mR@50 (~7.9) is much lower than R@50 (~30.2)** -- this ~4x gap is EXPECTED and CONSISTENT across all SGG methods in the literature
- Even strong two-stage models like Motifs show R@50=32.1 vs mR@50=5.5 (6x gap)

### Long-tail is a known characteristic:
The SGG literature universally acknowledges this. Methods using logit adjustment or debiasing (TDE, EBM, LA) trade R@K for mR@K. For example:
- EGTR baseline: R@50=30.2, mR@50=7.9
- EGTR LA(tau=0.3): R@50=28.2, mR@50=14.0 (traded ~2 R@50 points for ~6 mR@50 points)

---

## 7. Training Details (from Supplementary Section 3)

### Architecture
- **Backbone**: Deformable DETR [20] with ResNet-50 [3]
- **Pre-training**: Object detector backbone pre-trained on target dataset (VG), then SGG task trained with pre-trained detector
- **Object queries**: N = 200

### Hyperparameters (from Supplementary + code)
- **GPUs**: 8x V100
- **Batch size**: 32 (for object detector pre-training); 64 for EGTR (batch_size=4 x gpus=8 x accumulate=2)
- **Optimizer**: AdamW
- **Learning rates**:
  - Object detector: default LR = 10^-4, ResNet-50 LR = 10^-5
  - EGTR: relation extractor LR = 2 x 10^-4, object detector LR scaled down by 100x, ResNet-50 by 1000x
- **LR schedule**: Reduce by factor of 10 after initial training; adaptive schedule via early stopping (patience=15)
- **Epochs**:
  - Object detector pre-training: max 150 (first schedule) + 50 (second schedule)
  - EGTR training: max 50 (first schedule) + 25 (second schedule)
- **Loss weights**: lambda_rel=15, lambda_con=30 (VG), lambda_IoU / lambda_L1 / lambda_c / lambda_b follow Deformable DETR defaults (lambda_c=2, lambda_b implicit, lambda_IoU=2, lambda_L1=5)
- **Adaptive smoothing**: alpha = 10^-14
- **Hard sampling**: k_neg=80, k_non=80
- **Gradient clipping**: 0.1
- **Weight decay**: 10^-4
- **Image size**: min 600, max 1000 (for FPS measurement)

---

## 8. Released Checkpoints

### Available Downloads (from GitHub README):
- **Visual Genome checkpoint**: [Google Drive link](https://drive.google.com/file/d/18phcRxbrEI7HqIuM2OLAPuwAF5k3pUC2/view?usp=drive_link)
- **Open Image V6 checkpoint**: [Google Drive link](https://drive.google.com/file/d/1JqWNwf1QvDsTbGFigEXXN_8qv3VF-lGP/view?usp=drive_link)

### Expected Metrics:
The README does NOT specify expected R@K / mR@K / validation loss / epoch for the released checkpoint. The paper's Table 1 values (R@50=30.2, mR@50=7.9, AP50=30.8) should be reproducible with the released checkpoint.

### Evaluation Command:
```bash
python evaluate_egtr.py --data_path dataset/visual_genome --artifact_path $ARTIFACT_PATH
```

---

## 9. Ablation Tables Summary (Supplementary)

### Table 1 (Supp): Relation function f
| f type                | Params | R@50 | mR@50 |
|-----------------------|--------|------|-------|
| dot product attention | 41.3M  | 25.9 | 6.2   |
| dot product           | 41.3M  | 27.4 | 6.8   |
| Hadamard product      | 41.5M  | 29.1 | 7.2   |
| sum                   | 41.5M  | 29.5 | 7.3   |
| **concat**            | 41.6M  | **29.9** | **7.9** |

### Table 3 (Supp): lambda_rel tuning (lambda_con=0)
| lambda_rel | 5    | 10   | **15**   | 20   |
|------------|------|------|----------|------|
| R@50       | 29.1 | 29.1 | **29.4** | 28.8 |
| mR@50      | 7.3  | 7.3  | **7.3**  | 7.0  |

### Table 4 (Supp): lambda_con tuning (lambda_rel=15)
| lambda_con | 0    | 15   | **30**   | 45   |
|------------|------|------|----------|------|
| R@50       | 29.4 | 29.7 | **30.2** | 29.6 |
| mR@50      | 7.3  | 7.4  | **7.9**  | 7.4  |

### Table 5 (Supp): alpha for adaptive smoothing
| alpha    | R@50 | mR@50 |
|----------|------|-------|
| 10^-13   | 30.1 | 7.8   |
| **10^-14** | **30.2** | **7.9** |
| 10^-15   | 30.0 | 7.8   |

### Table 6 (Supp): Sampling k_neg = k_non
| k      | R@50 | mR@50 |
|--------|------|-------|
| 10     | 29.6 | 8.2   |
| 20     | 29.9 | 8.2   |
| 40     | 30.0 | 8.1   |
| **80** | **30.2** | 7.9   |
| 160    | 29.9 | 7.7   |

---

## 10. Key Observations for PoC Comparison

1. **R@K is inherently dominated by head predicates**: "on" alone accounts for ~40% of GT. A model predicting mostly "on"/"has"/"wearing" can achieve high R@K while having near-zero recall on 30+ tail predicates.

2. **mR@K is EXPECTED to be much lower than R@K**: The 4-6x gap (e.g., R@50=30.2 vs mR@50=7.9) is universal across ALL SGG methods and is a known artifact of the extreme long-tail distribution in VG.

3. **Debiasing trades R@K for mR@K**: Logit adjustment improves mR@K dramatically but drops R@K. This is NOT a bug -- it's the expected tradeoff.

4. **AP50=30.8 is the highest among one-stage methods**: Object detection quality directly impacts SGDet (boxes must match with IoU>=0.5). A detector with lower AP50 will produce lower R@K regardless of relation prediction quality.

5. **The evaluation is on the TEST set** (26K images), not validation (5K).

---

## Sources

- arXiv HTML: https://arxiv.org/html/2404.02072v4
- arXiv abstract: https://arxiv.org/abs/2404.02072
- CVPR Open Access: https://openaccess.thecvf.com/content/CVPR2024/html/Im_EGTR_Extracting_Graph_from_Transformer_for_Scene_Graph_Generation_CVPR_2024_paper.html
- Supplementary PDF: https://openaccess.thecvf.com/content/CVPR2024/supplemental/Im_EGTR_Extracting_Graph_CVPR_2024_supplemental.pdf
- GitHub: https://github.com/naver-ai/egtr
- SGG Metrics Review: https://arxiv.org/html/2404.09616v1
