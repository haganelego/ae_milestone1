# Literature survey: query-conditioned early stopping under progressive input quality (VLM / VQA)

Date: 2026-10-10. Purpose: related-work + novelty check for the VISAPP 2027 gqa_stop paper
(qwen3.5:9b, GQA attribute questions, 20-level same-view resolution recovery, "Direct" = P(first answer token) x P("Yes" | "Are you confident about answering with the current view?"), checklist-decomposition variant with min-item stopping, fixed-level concave-hull baseline).

Verification policy: every entry below was opened (arXiv abs/HTML, CVF/ACL/PMLR page, or PDF) during this session unless marked **(unverified)**. "Verified" means title/authors/venue and the specific claim attributed here were seen in the source; details beyond the cited claim were not checked.

---

## Q1. Prior work that stops / early-exits as INPUT quality progressively improves

### 1a. Classic anytime / input-adaptive recognition (classification only, no questions)

| Paper | Input progression | Stop signal | Notes |
|---|---|---|---|
| Karayev, Fritz, Darrell. *Anytime Recognition of Objects and Scenes*. CVPR 2014. https://mlanthology.org/cvpr/2014/karayev2014cvpr-anytime | sequential feature computation (not pixels) | learned dynamic policy | foundational "anytime" framing |
| Teerapittayanon, McDanel, Kung. *BranchyNet*. ICPR 2016. https://arxiv.org/abs/1709.01686 | depth (side branches) | softmax entropy threshold | architecture early-exit, not input |
| Huang et al. *Multi-Scale Dense Networks (MSDNet)*. ICLR 2018. https://arxiv.org/abs/1703.09844 | depth, multi-scale features | confidence threshold; anytime + budgeted batch | architecture early-exit |
| Yang, Han, Chen, Song, Dai, Huang. *Resolution Adaptive Networks (RANet)*. CVPR 2020. https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Resolution_Adaptive_Networks_for_Efficient_Inference_CVPR_2020_paper.html | **low -> high resolution sub-networks** | "samples with high prediction confidence will exit early" | closest classic analogue of a resolution ladder; classification only (CIFAR, ImageNet) |
| Wang et al. *Glance and Focus (GFNet)*. NeurIPS 2020. https://arxiv.org/abs/2010.05300 | glance at downscaled image, then sequential crops | confidence threshold, tuned online; better than entropy criterion | classification only |
| Kim, Kim, Lee. *PICM-Net: Progressive Learned Image Compression for Machine Perception*. arXiv 2512.20070 (v2 Aug 2026). https://arxiv.org/abs/2512.20070 | **progressive bitstream decoding levels** | learned logistic regression over 12 classifier-output features (max prob, entropy, energy...) estimating P(correct); stop when > tau | compares **against fixed decoding levels at matched mean bitrate** (~21.6% BD-rate gain at tau=0.70). ResNet-50 / ConvNeXt classification; no VLM, no question |

### 1b. VLM / MLLM: resolution or token budget chosen per query

| Paper | Mechanism | Sequential stopping? | Signal |
|---|---|---|---|
| Cai, Yang, Gao, Lee. *Matryoshka Multimodal Models (M3)*. ICLR 2025. https://arxiv.org/abs/2405.17430 | nested coarse-to-fine visual token sets, granularity chosen per instance | no (manual/oracle choice) | reports a **large per-sample oracle gap** vs fixed token count; COCO-style benchmarks need ~9 of 576 tokens |
| Kimhi, Shabtay, Giryes, Baskin, Schwartz. *CARES: Context-Aware Resolution Selector for VLMs*. arXiv 2510.19496 (v2 Mar 2026). https://arxiv.org/abs/2510.19496 | separate 350M classifier predicts **minimal sufficient resolution per image-query pair** (3 discrete levels 384/768/1024, interpolated) | **no, one-shot preprocessing** | external trained predictor. Labels = lowest resolution whose ANLS >= 0.85 with <= 0.1 gain above it (a "sufficiency rollout", analogous to our oracle level). Baselines: fixed low/native/high. Doc/chart/OCR benchmarks, not GQA |
| Yang, Li, Lai, Yu, Zhao, Jia. *VisionThink*. NeurIPS 2025. https://arxiv.org/abs/2507.13348 | start at 1/4-resolution image; model emits a special token to request full resolution | single escalation (2 levels) | RL-trained policy (LLM-as-judge reward), not a confidence threshold |
| Lin, Liu, Yang, Tao, Ye. *AdaptVision*. CVPR 2026. https://arxiv.org/abs/2512.03794 | low-res compressed tokens, then bbox-crop tool calls "when necessary" | multi-turn, learned | RL policy (DTPO) |
| Shabtay, Kimhi, et al. *Look Where It Matters (AwaRes)*. arXiv 2603.16932 (v2 Sep 2026). https://arxiv.org/abs/2603.16932 | low-res global view; either answer or request high-res crops | single escalation decision | learned coupled-decision policy (SFT + GRPO); baselines include VisionThink, fixed token pruning |
| Hsu, Cheng, Papagianni. *Progressive Semantic Communication for Efficient Edge-Cloud VLMs*. IEEE GLOBECOM 2026, arXiv 2604.26508. https://arxiv.org/abs/2604.26508 | **progressively refinable latent of visual tokens transmitted level by level** | **yes, stop when quality q(l) >= epsilon** | learned **reconstruction-error predictor**, fixed user threshold, *not query-conditioned*, not VLM self-confidence. SmolVLM-256M, COCO captioning + POPE |
| Liu, Li, Deng, Quek. *Goal-Oriented Semantic Communication for Wireless VQA*. arXiv 2411.02452. https://arxiv.org/abs/2411.02452 | transmit question-relevant boxes / scene-graph content | no progressive stopping | question-type prioritisation |

### 1c. Zoom-in / visual-search MLLM agents (crop sequence, not resolution recovery of the whole view)

| Paper | Stop criterion | Fixed-budget comparison |
|---|---|---|
| Wu, Xie. *V\*: Guided Visual Search as a Core Mechanism in Multimodal LLMs (SEAL)*. CVPR 2024. https://arxiv.org/abs/2312.14135 | VQA LLM first judges if global features suffice; if not, **lists missing target objects**; per-target recursive search until target located (detection confidence / heatmap threshold) or patch below min size | no (vs random/sequential search only) |
| Shen, Zhao, Zhao, Xu, Zhang, Zhu, Yin. *ZoomEye*. EMNLP 2025. https://aclanthology.org/2025.emnlp-main.335/ | tree search; **answering confidence c_a from "Could you answer [question] now?" Yes/No logit ratio, stop if c_a >= tau (0.8 / 0.6)**; also existing / latent confidences from Yes/No logits | no direct fixed-depth baseline |
| Li, Xu, Zhao, Peng. *DyFo*. CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Li_DyFo_A_Training-Free_Dynamic_Focus_Visual_Search_for_Enhancing_LMMs_CVPR_2025_paper.html | MCTS over focus actions with a visual expert | stop rule not checked **(unverified)** |
| Zhang, Khayatkhoei, Chhikara, Ilievski. *MLLMs Know Where to Look (ViCrop)*. ICLR 2025. https://arxiv.org/abs/2502.17422 | attention/gradient-guided crop added to input | no stopping decision found in abstract **(full paper unverified)** |
| *Zoom-Refine* (author list not checked). arXiv 2506.01663. https://arxiv.org/abs/2506.01663 | one localized zoom + self-refinement | no stopping |
| Wang et al. *Active Video Perception (AVP)*. arXiv 2512.05774 (CVPR 2026 Findings per CVF URL). https://arxiv.org/abs/2512.05774 | plan-observe-reflect; reflector judges evidence sufficiency, stops with answer or observes more | vs agentic baselines; exact reflector signal not verified **(partially unverified)** |
| Toh, Majumder, Liu, Chen, Poria. *MNIST-PRO*. arXiv 2608.31022. https://arxiv.org/abs/2608.31022 | glimpse-based partially observable benchmark; agents must decide when to stop | finds **premature stopping** and failure to revise early beliefs |

### 1d. Analyses of VLM confidence under progressive evidence loss (diagnostic, no stopping policy)

- Ali Aacaas Muhamath. *Evidence-Order Calibration for Selective Visual Reasoning under Progressive Loss of Question-Critical Evidence*. arXiv 2609.09184 (Aug 2026). https://arxiv.org/abs/2609.09184 — **GQA balanced-val**, Qwen2.5-VL-3B, 5 severity levels of masking (blur for transfer) of scene-graph-localised question-critical regions, 176 trajectories. Native sequence confidence violates evidence ordering in **92.0%** of trajectories; post-hoc head reduces violations but does not beat native confidence on AURC. No stopping, no decomposition. Closest dataset/question match to ours, but degradation direction and goal differ.
- Zhang, Wang, Lin, Bitton, Szpektor, Bansal. *Seeing Isn't Knowing (SpatialUncertain)*. arXiv 2605.30557. https://arxiv.org/abs/2605.30557 — 8 VLMs; "models do not reliably become more cautious as evidence disappears" (occlusion); verbalized "Cannot determine" abstention.
- Borszukovszki, de Jong, Valdenegro-Toro. *Know What You do Not Know: Verbalized Uncertainty Estimation Robustness on Corrupted Images in VLMs*. TrustNLP @ NAACL 2025. https://arxiv.org/abs/2504.03440 — corruption severity degrades verbalized uncertainty; overconfidence in most settings.
- Ferdous. *Bigger or Cheaper? Scale and Quantization Effects on Uncertainty Signals in VLMs Under Image Degradation*. arXiv 2607.24440. https://arxiv.org/abs/2607.24440 — includes resampling (downscale) degradation; **mean token probability** error-detection AUROC rises 0.80 -> 0.98 from 2B to 7B, while **verbalized confidence stays near chance**; quantization hurts confidence more than accuracy (Food101 MCQ, small study).

**Answer to Q1.** Yes for classification (RANet, GFNet, PICM-Net: confidence-based stop over increasing resolution/bitrate, PICM-Net even against fixed levels). For VLMs, query-conditioned resolution choice exists (CARES one-shot; VisionThink/AwaRes/AdaptVision learned escalation; progressive semantic-comm with a non-query quality stop). I found **no paper that runs a training-free, VLM-self-confidence stopping rule over a multi-level same-view resolution ladder for VQA and compares it to a fixed-level frontier**. Absence of evidence, not proof.

---

## Q2. Which use a confidence signal derived from the VLM itself as the stopping criterion?

| Paper | Self-derived signal | Fixed-budget comparison |
|---|---|---|
| Ren et al. *Explore until Confident*, RSS 2024 | letter-softmax x relevance P(Yes) "Are you confident about answering the question with the current view?", multi-step conformal prediction | entropy / relevance-threshold baselines, step budget |
| ZoomEye, EMNLP 2025 | P(Yes) vs P(No) to "Could you answer [question] now?", threshold tau | no |
| SEAL / V\*, CVPR 2024 | VLM verbal judgement "global features suffice?" + localisation confidence | no |
| IdealGPT, EMNLP Findings 2023 | reasoner verbal "We are not sure" | max-iteration cap only |
| GraphEQA (Saxena et al.), CoRL 2025, https://arxiv.org/abs/2412.14480 | binary + numeric **verbalized** confidence for episode termination | EQA baselines |
| Rendering-Free Lookahead (Sakamoto et al.), arXiv 2610.11039 (8 Oct 2026), https://arxiv.org/abs/2610.11039 | answerability sigma(l_Yes - l_No) for "can [question] be answered from this view?" used to **rank motions**; stopping left to base VLM and named a limitation | fixed T_max = 25 |
| RANet, GFNet (classification) | softmax confidence | budgeted / anytime curves |
| PICM-Net | learned head over classifier logits (self-derived features, learned map) | **yes, fixed decoding levels** |
| Not self-derived | CARES (external predictor), VisionThink / AwaRes / AdaptVision (RL policy), Progressive SemCom (reconstruction-error head), EfficientEQA (BLIP relevance outlier, Z=3) | |

Note: EfficientEQA (Cheng et al., arXiv 2410.20263, https://arxiv.org/abs/2410.20263) is sometimes summarised as using "verbalized confidence"; that is for **exploration** scoring. Its **stopping** module is BLIP image-question similarity outlier detection (mu + 3 sigma, min 10 steps).

---

## Q3. Question decomposition combined with a stopping decision

| Paper | Decomposition | What is stopped / triggered | Input fixed? |
|---|---|---|---|
| ZoomEye, EMNLP 2025 | question -> "visual cues" o_1..o_k (single-instance / all-instance types) via in-context examples | zoom search; stop on **whole-question** answering confidence, cues guide where to look | image fixed, views = crops |
| SEAL / V\*, CVPR 2024 | VLM lists missing target objects | per-target search loop ends when each target is located (checklist-like over targets, localisation not answer confidence) | crops |
| You et al. *IdealGPT*. EMNLP Findings 2023. https://arxiv.org/abs/2305.14985 | LLM generates sub-questions, VLM answers | iterate until reasoner is confident (verbal) or cap; avg 1.8 passes; **no per-sub-question score aggregation** | yes |
| Khan, BG, Schulter, Fu, Chandraker. *Exploring Question Decomposition for Zero-Shot VQA*. NeurIPS 2023. https://arxiv.org/abs/2310.17050 | model-written sub-questions | **selective decomposition**: decompose only if initial answer confidence p(a) < tau (selective prediction framing) | yes |
| Yang, Yan, Agrawal. *Decompose and Compare Consistency (DeCC)*. EMNLP 2024. https://arxiv.org/abs/2407.07840 | sub-questions -> indirect answer | reliability = consistency of direct vs decomposed answer (no stopping) | yes |
| Cho et al. *Davidsonian Scene Graph (DSG)*. ICLR 2024. https://arxiv.org/abs/2310.18235 ; Yarom et al. *What You See is What You Read? (SeeTRUE / VQ2)*. NeurIPS 2023. https://arxiv.org/abs/2305.10400 | atomic yes/no questions (DSG with dependency graph) | evaluation scoring of text-image alignment, not stopping | yes |
| Zheng et al. *DDCoT*. NeurIPS 2023. https://arxiv.org/abs/2310.16436 ; Liu et al. *Chain-of-Spot*. arXiv 2403.12966 | sub-questions / ROI step | no stopping rule | yes |

**Answer to Q3.** Decomposition + confidence is common (IdealGPT stop-on-confident loop, Khan et al. confidence-triggered decomposition, DeCC decomposition-as-reliability, DSG atomic-question checklists). Decomposition guiding a sequential perception loop with a confidence stop exists (ZoomEye, SEAL). I did **not** find a method that stops an input-quality sequence when the **minimum** over decomposed checklist-item scores clears a threshold. The aggregation (min over items) and the stopped axis (resolution level) appear new; the ingredients do not.

---

## Q4. Explore until Confident (Ren et al., RSS 2024) and follow-ups

Base: Ren, Clark, Dixit, Itkina, Majumdar, Sadigh. *Explore until Confident: Efficient Exploration for Embodied Question Answering*. RSS 2024. https://arxiv.org/abs/2403.15941 . Details already verified in `explore-eqa-confidence.md` (Prismatic VLM, letter softmax x relevance P(Yes), multi-step CP = min over steps / intersection of sets, stop when singleton). Predecessor: Ren et al. *Robots That Ask For Help (KnowNo)*. CoRL 2023, PMLR v229. https://proceedings.mlr.press/v229/ren23a.html (CP on LLM MCQ-option scores, ask for help when set not singleton).

Foundational self-assessment: Kadavath et al. *Language Models (Mostly) Know What They Know*. arXiv 2207.05221 (2022). https://arxiv.org/abs/2207.05221 — P(True) = prompted self-evaluation of a proposed answer; P(IK) = probability "I know", **trained** (value head), without reference to a proposed answer.

Follow-ups (EQA / active perception) and whether they reuse Rel x answer-logit:

| Paper | Reuses ETC Rel x logit stop? | What they do instead |
|---|---|---|
| Cheng et al. *EfficientEQA*. arXiv 2410.20263. | no | BLIP relevance outlier stopping; verbalized confidence for exploration |
| Saxena et al. *GraphEQA*. CoRL 2025 (PMLR v305). | no | verbalized binary + numeric confidence for termination |
| Frahm, Patel, Zhang, Yu, Bansal, Sengupta. *Prune-Then-Plan*. arXiv 2511.19768. https://arxiv.org/abs/2511.19768 | no | step-level calibration of frontier scores (ECDF + Holm-Bonferroni), not stopping |
| Ginting et al. *Enter the Mind Palace (LA-EQA)*. arXiv 2507.12846. https://arxiv.org/abs/2507.12846 | no | value-of-information + single-area prediction set to stop *memory retrieval*; ETC appears as a baseline |
| Zhang et al. *FAST-EQA*. WACV 2026. https://arxiv.org/abs/2602.15813 | not verified | "until it can answer confidently" via CoT over memory **(signal unverified)** |
| Zhai, Gao, Wu, Jia. *Memory-Centric EQA*. arXiv 2505.13948. https://arxiv.org/abs/2505.13948 | not verified | has a stopping module; details not checked **(unverified)** |
| Sakamoto et al. *Rendering-Free Lookahead (RFL)*. arXiv 2610.11039. | partially (same Yes/No-logit answerability family, own prompt) | ranks camera motions by answerability; does not solve stopping |
| "Robot-EQA", "Embodied-RAG" | not found as ETC-signal reusers in this search **(unverified, not investigated further)** | |

Takeaway: the relevance x answer-logit product with CP is mostly cited, not reused verbatim. The **Yes/No-logit "can you answer now?" probe** is the reused idea (ZoomEye, RFL). The verbalized variant is common in GPT-based EQA (GraphEQA).

---

## Q5. Novelty assessment

| Claim | Novel? | Closest prior work (how it differs) |
|---|---|---|
| (a) **Direct signal** P(answer token) x P(Yes \| "confident with current view?") | **No.** Direct re-application. | 1. Explore until Confident (RSS 2024): identical product (letter softmax x relevance Yes-prob), plus CP and multi-step intersection; embodied views. 2. ZoomEye (EMNLP 2025): Yes/No-logit "Could you answer now?" with threshold stop over zoom steps. 3. Kadavath et al. 2022: P(True)/P(IK) self-assessment origin (text-only). |
| (b) **Setting**: same-view resolution-recovery ladder (20 levels), query-conditioned stop of an off-the-shelf VLM, vs fixed-level frontier | **Partially novel** (no exact match found; neighbours are close). Frame as an *evaluation setting/benchmark* rather than a new mechanism. | 1. CARES (2025/26): query-conditioned minimal sufficient resolution, but **one-shot** selection by a **separate trained predictor** over 3 levels; doc/OCR data. 2. PICM-Net (2025/26): stop progressive decoding when estimated correctness > tau, **vs fixed levels**, but **image classification**, no question, learned head. 3. VisionThink (NeurIPS 2025): low-res -> request high-res, but **2 levels**, **RL-trained** escalation, no sequential stopping curve. (Also: Progressive SemCom GLOBECOM 2026 = progressive VLM input with stop, but non-query reconstruction-error signal; Evidence-Order Calibration 2026 = GQA + progressive evidence loss, but diagnostic, no stopping.) |
| (c) **Decomposition-based stopping** (checklist items, stop when min item score passes threshold) | **Partially novel**: the min-over-items rule on an input-quality axis was not found; decomposition + confidence-gated control is well established. | 1. ZoomEye: decomposes into visual cues, stops on whole-question answer confidence (no per-item gate). 2. SEAL/V\*: lists missing targets, per-target search until each is found (checklist over targets, localisation not answer reliability). 3. IdealGPT / Khan et al. 2023: sub-questions with confidence-triggered iteration/decomposition on a **fixed** image. |

Risks for reviewers:
- RANet/GFNet/PICM-Net make "confidence-based stop over increasing resolution" a known recipe; the contribution must be the VQA/query-conditioned VLM setting and the analysis, not the stop rule.
- CARES's "sufficiency rollout" labels are conceptually our oracle level; cite it and contrast one-shot learned selection vs sequential training-free stopping.
- Several 2026 diagnostics (Evidence-Order Calibration on GQA, SpatialUncertain, Bigger-or-Cheaper) report that VLM confidence does not track evidence degradation and that verbalized confidence is weak; this supports using token probabilities and calibrated thresholds, and explains failure modes.
- Coverage limits: arXiv-heavy search; several 2026 entries are preprints. Items marked (unverified) should be checked before citing.
