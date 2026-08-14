# Literature Survey: Semantic Saturation Detection in Progressive Image Transmission via Scene Graph Generation

**Date:** 2026-07-13
**Purpose:** Conference-paper positioning (WACV/BMVC/ICIP/ICASSP/CVPR-workshop / SemCom venue)

---

## 1. Semantic Communication & Progressive Image Transmission with Semantic Stopping

### Key Papers

1. **"Task-Oriented Scene Graph-Based Semantic Communications With Adaptive Channel Coding" (GRACE)**
   - Authors: Various (IEEE TWC, 2024)
   - Venue: IEEE Transactions on Wireless Communications
   - Relevance: Directly uses scene graphs as semantic representations for task-oriented communication. Proposes adaptive channel coding based on scene graph semantics for image retrieval. Closest prior work to our setting, but focuses on channel coding adaptation rather than stopping criterion for progressive transmission.

2. **"Task-Oriented Explainable Semantic Communications Based on Structured Scene Graphs" (DeepSC-SG)**
   - Authors: Various (IEEE TWC, 2023; ICASSP 2024)
   - Relevance: Extracts explainable scene graph semantics from images and encodes them into graph embeddings for transmission. Demonstrates that scene graphs are a viable semantic representation for communication, but does not address progressive refinement or saturation detection.

3. **"Scene Graph-Aided Probabilistic Semantic Communication for Image Transmission"**
   - Authors: Chen Zhu, Siyun Liang, Zhouxiang Zhao, et al. (arXiv 2507.11913, July 2025)
   - Relevance: Uses probability graphs (learned from Visual Genome) as a shared semantic knowledge base. Filters redundant relations at the transmitter and reconstructs omitted information at the receiver using conditional/co-occurrence probabilities. Very close to our setting (VG dataset, scene graphs, transmission efficiency), but focuses on static compression rather than progressive transmission with a stopping rule.

4. **"SC-GIR: Goal-oriented Semantic Communication via Invariant Representation Learning for Image Transmission"**
   - Authors: Various (2025)
   - Relevance: Grounded in information bottleneck theory; selectively extracts and transmits semantically relevant information. Relevant framework for understanding what "sufficient semantics" means.

5. **"Progressive Semantic Communication for Efficient Edge-Cloud Vision-Language Models"**
   - Authors: Various (arXiv 2604.26508, 2026)
   - Relevance: Encodes visual inputs into refinable representations transmitted as partial latent segments with quality-aware transmission control that adaptively requests additional information until a target quality threshold is met. Closest to our "stop when sufficient" paradigm, but uses VLM confidence rather than scene graph saturation.

6. **"Transmit What You Need: Task-Adaptive Semantic Communications for Visual Information"**
   - Authors: Various (arXiv 2412.13646, 2024)
   - Relevance: Task-adaptive approach that transmits only what is needed for the downstream task. Related to our motivation but does not use scene graphs or study saturation curves.

7. **"Goal-Oriented Semantic Communication for Wireless Image Transmission via Stable Diffusion"**
   - Authors: Various (arXiv 2408.00428, 2024)
   - Relevance: Uses generative models to reduce bandwidth while maintaining semantic quality. Different mechanism (generative reconstruction vs. progressive refinement).

### Gap Assessment
No existing work studies **progressive image recovery with scene-graph-based saturation detection as the stopping criterion**. Existing SemCom papers either (a) use scene graphs for static compression/retrieval, (b) use pixel-level or VLM-level quality metrics for progressive stopping, or (c) focus on channel adaptation rather than semantic convergence monitoring. The intersection of progressive transmission + SGG + saturation detection is novel.

---

## 2. Bayesian / Statistical Prediction of Saturating Curves & Optimal Stopping

### Key Papers

1. **"Speeding Up Automatic Hyperparameter Optimization of Deep Neural Networks by Extrapolation of Learning Curves"**
   - Authors: Domhan, Springenberg, Hutter (IJCAI 2015)
   - Relevance: Foundational work on learning curve extrapolation using parametric curve families. Could be adapted to predict semantic saturation curves from early resolution levels.

2. **"Freeze-Thaw Bayesian Optimization"**
   - Authors: Swersky, Snoek, Adams (2014)
   - Relevance: Uses Gaussian processes with exponential decay kernels to model learning curves and decide whether to continue or switch. The "freeze-thaw" decision (continue vs. stop) is analogous to our "transmit more vs. stop" decision.

3. **"Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks" (LC-PFN)**
   - Authors: Adriaensen et al. (NeurIPS 2023)
   - Relevance: Approximate Bayesian inference for learning curve prediction in a single forward pass (10,000x faster than MCMC). Achieves 2-6x speedups via predictive early stopping. Could be applied to predict saturation point from partial semantic curves.

4. **"Bayesian Online Changepoint Detection"**
   - Authors: Adams & MacKay (2007; widely extended 2023-2025)
   - Relevance: Online detection of regime changes in sequential data. Saturation onset is a changepoint (transition from "improving" to "stable" regime). BOCPD provides a principled online framework for detecting this transition.

5. **"Kneedle: Knee Point Detection in Python"** / **"Kneeliverse: A Universal Knee-Detection Library"**
   - Authors: Satopaa et al. (2011) / Recent 2025 extension
   - Relevance: Detects the "knee" (point of maximum curvature) in performance curves. Directly applicable to finding the saturation elbow in semantic score vs. resolution curves. Simple but lacks uncertainty quantification.

6. **"Learning the Optimal Stopping for Early Classification within Finite Horizons via SPRT" (FIRMBOUND)**
   - Authors: Various (arXiv 2501.18059, 2025)
   - Relevance: SPRT-based framework for optimal early classification in finite-horizon settings. Bridges optimal stopping theory and practical deployment. Our 20-level progressive recovery is a finite-horizon sequential decision.

7. **"Optimal Stopping for Sequential Bayesian Experimental Design"**
   - Authors: Various (arXiv 2509.21734, 2025)
   - Relevance: Addresses when to stop gathering information in Bayesian experimental design. The trade-off between additional information value and acquisition cost directly parallels our transmission cost vs. semantic gain trade-off.

8. **"Stopping Bayesian Optimization with Probabilistic Regret Bounds"**
   - Authors: Wilson et al. (arXiv 2402.16811, 2024)
   - Relevance: Principled stopping criteria for BO based on probabilistic regret bounds. Framework could inform when the expected semantic improvement falls below a threshold.

### Gap Assessment
Bayesian learning-curve extrapolation (Domhan, LC-PFN) has been applied to training curves but **never to semantic saturation curves** from progressive image recovery. BOCPD is used in time-series but not in the SGG context. No work combines these statistical tools with scene graph semantics.

---

## 3. Early-Exit / Anytime Inference in Vision Models

### Key Papers

1. **"FREE: Fast and Robust Vision Language Models with Early Exits"**
   - Authors: Various (ACL Findings 2025)
   - Relevance: Early exit for VLMs based on confidence thresholds. Validates that early stopping in vision models is viable, but operates at the layer level within a single model, not across resolution levels.

2. **"Dynamic Early Exit in Reasoning Models"**
   - Authors: Various (arXiv 2504.15898, 2025)
   - Relevance: Dynamic early exit using confidence-based stopping in reasoning chains. Analogous to our sequential stopping across resolution levels.

3. **"PABEE: Pre-trained Model-Based Early Exiting"**
   - Authors: Zhou et al. (2020; widely cited)
   - Relevance: Exit when K consecutive internal classifiers agree. Our "consecutive-k" stopping criterion is conceptually similar.

4. **"MuE: Multimodal Early Exit for Vision-Language Models"**
   - Authors: Various (2024)
   - Relevance: Extends early exit to multimodal architectures using layer-wise input similarity as exit criterion. Related to our use of semantic stability across levels.

5. **"From Inheritance to Saturation: Disentangling the Evolution of Visual Redundancy for Architecture-Aware MLLM Inference Acceleration"**
   - Authors: Various (arXiv 2604.16462, 2026)
   - Relevance: Identifies a three-stage lifecycle (Modality Alignment, Global Aggregation, Visual Saturation) in MLLMs. Uses truncated matrix entropy to detect saturation. Conceptually close but operates within model layers, not across input resolution.

6. **"HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit"**
   - Authors: Various (arXiv 2602.23699, 2026)
   - Relevance: Combines token pruning with early exit strategies for efficiency. Shows that semantic saturation of visual tokens can be detected and exploited.

### Gap Assessment
Early-exit literature focuses on **within-model layer-level** stopping. Our problem is **across-input resolution-level** stopping, which is structurally different (each resolution produces a full SGG output, not a partial internal representation). No early-exit work addresses resolution-progressive scene graph saturation.

---

## 4. Scene Graph Similarity & Semantic Stability Metrics

### Key Papers

1. **"SPICE: Semantic Propositional Image Caption Evaluation"**
   - Authors: Anderson et al. (ECCV 2016)
   - Relevance: Evaluates caption quality by comparing scene graph tuples (objects, attributes, relations) via F1-score. Foundational metric for scene graph comparison. Our saturation detection requires a similar tuple-matching framework.

2. **"S^2Match: Soft Similarity Match for Scene Graph Comparison"**
   - Authors: Opitz, Parcalabescu, Frank (2020)
   - Relevance: Implements graded semantic matching using GloVe embedding distances instead of exact string matching. Directly addresses our synonym-matching problem (e.g., "on" vs. "on top of", "person" vs. "man").

3. **"Graph R-CNN for Scene Graph Generation"**
   - Authors: Yang et al. (ECCV 2018)
   - Relevance: Establishes scene graph structure and evaluation protocols. Baseline for graph-level comparison methods.

4. **"GEDAN: Learning the Edit Costs for Graph Edit Distance"**
   - Authors: Various (2024)
   - Relevance: Learns task-specific edit costs for GED computation. Could be adapted for semantic-aware scene graph distance.

5. **"SCENIR: Visual Semantic Clarity through Unsupervised Scene Graph Retrieval"**
   - Authors: Various (arXiv 2505.15867, 2025)
   - Relevance: Uses scene graph embeddings for retrieval, requiring effective graph-level similarity computation. Shows GNN-based approaches to scene graph comparison.

### Gap Assessment
SPICE provides the closest framework, but is designed for caption evaluation, not for measuring **semantic stability across a sequence of scene graphs**. No metric specifically measures "how much new semantic information does this scene graph add over the previous one?" — a delta-semantic metric for progressive SGG is missing.

---

## 5. LLM-as-Judge Reliability & Calibration

### Key Papers

1. **"Analyzing Uncertainty of LLM-as-a-Judge: Interval Evaluations with Conformal Prediction"**
   - Authors: Various (EMNLP 2025)
   - Relevance: Constructs prediction intervals for LLM judge scores using conformal prediction. Directly applicable to our LLM-based saturation scoring — could provide coverage guarantees on saturation scores.

2. **"Diagnosing LLM Judge Reliability: Conformal Prediction Sets and Transitivity Violations"**
   - Authors: Various (arXiv 2604.15302, 2026)
   - Relevance: Diagnoses reliability issues in LLM judges through conformal prediction sets and transitivity analysis. Relevant for understanding when our qwen3.5-based judge can be trusted.

3. **"Calibrating LLM Judges: Linear Probes for Fast and Reliable Uncertainty Estimation"**
   - Authors: Various (arXiv 2512.22245, 2025)
   - Relevance: Uses linear probes on LLM hidden states for uncertainty estimation. Could replace our ad-hoc LLM scoring with calibrated confidence.

4. **"VERDI: Single-Call Confidence Estimation for Verification-Based LLM Judges via Decomposed Inference"**
   - Authors: Various (arXiv 2605.11334, 2026)
   - Relevance: Single-call confidence estimation for LLM judges. Efficiency-oriented approach to getting reliable confidence from LLM scoring.

5. **"Bias and Uncertainty in LLM-as-a-Judge Estimation"**
   - Authors: Various (arXiv 2605.06939, 2026)
   - Relevance: Systematic study of bias and uncertainty in LLM judging. Important for understanding limitations of our LLM-as-saturation-judge approach.

6. **"Conformal Prediction and MLLM aided Uncertainty Quantification in Scene Graph Generation"**
   - Authors: Nag, Ghosh, Ta, Bose, Li, Roy-Chowdhury (CVPR 2025)
   - Relevance: **Highly relevant.** First work combining conformal prediction with MLLM for uncertainty quantification in SGG. Constructs calibrated prediction sets over scene graphs with coverage guarantees. Could provide principled uncertainty quantification for our SGG outputs.

### Gap Assessment
LLM-judge calibration research is booming but **no work applies it to semantic saturation scoring for progressive transmission**. The CVPR 2025 conformal-SGG paper addresses SGG uncertainty but not sequential saturation detection. Combining conformal prediction with LLM-based saturation judgment is an open direction.

---

## 6. Resolution/Degradation Robustness of SGG and Detection Models

### Key Papers

1. **"On the Effect of Image Resolution on Semantic Segmentation"**
   - Authors: Khoshsirat & Kambhamettu (WACV 2024)
   - Relevance: Studies how image resolution affects semantic segmentation performance, finding non-linear relationships with diminishing returns at higher resolutions. Closest existing study to our resolution-vs-semantics question, but uses segmentation, not SGG.

2. **"RestoreDet: Degradation Equivariant Representation for Object Detection in Low Resolution Images"**
   - Authors: Various (2022)
   - Relevance: Learns representations equivariant to degradation for robust detection. Addresses the same underlying problem (detection robustness under resolution change) but does not study scene graphs or saturation.

3. **"Super-Resolution and Object Detection Compatibility"**
   - Authors: Various (PMC 2024)
   - Relevance: Studies whether super-resolution helps downstream detection. Finds mixed results: transformer-based SR can even decrease DETR performance for bicubic degradation. Relevant to understanding why bilinear degradation affects EGTR performance.

4. **"EGTR: Extracting Graph from Transformer for Scene Graph Generation"**
   - Authors: Im et al. (CVPR 2024, Best Paper Award Candidate)
   - Relevance: The model we use. Published at CVPR 2024. Does not study robustness to resolution degradation — this is a gap our work fills.

5. **"HiKER-SGG: Hierarchical Knowledge Enhanced Robust Scene Graph Generation"**
   - Authors: Various (arXiv 2403.12033, 2024)
   - Relevance: Addresses robustness in SGG but through knowledge enhancement, not resolution robustness. Shows that SGG robustness is an active concern.

6. **"Statistical Confidence Rescoring for Robust 3D Scene Graph Generation from Multi-View Images"**
   - Authors: Various (arXiv 2508.06546, 2025)
   - Relevance: Uses statistical confidence rescoring for robust 3D SGG. Relevant methodology for confidence-based approaches to SGG reliability.

### Gap Assessment
**No prior work systematically studies how SGG output degrades with image resolution.** Object detection resolution studies exist but stop at bounding boxes; segmentation-resolution studies exist but produce pixel-level output. The entire curve of "SGG semantic content vs. resolution" is uncharted territory.

---

## 7. Conformal Prediction for Early Stopping (Cross-Cutting)

### Key Papers

1. **"Conformal Prediction for Early Stopping in Mixed Integer Optimization"**
   - Authors: Various (arXiv 2602.01476, 2026)
   - Relevance: Uses conformal prediction to decide when to stop an optimization process. Directly analogous to our problem: when to stop progressive refinement.

2. **"Selective Conformal Risk Control"**
   - Authors: Various (2024)
   - Relevance: Provides selection-conditional coverage guarantees. Could be applied to selectively apply stopping rules based on image complexity.

3. **"Online Conformal Prediction via Universal Portfolio Algorithms"**
   - Authors: Various (arXiv 2602.03168, 2026)
   - Relevance: Parameter-free online conformal prediction. Relevant for streaming/online saturation detection.

---

## Novelty Positioning

### Candidate Claims of Novelty

1. **First systematic study of SGG output vs. image resolution degradation curves**
   - Closest prior: Khoshsirat & Kambhamettu (WACV 2024) study segmentation vs. resolution; EGTR paper (CVPR 2024) does not study resolution robustness. No one has characterized the semantic saturation curve of a scene graph generator under progressive resolution recovery.
   - Strength: Strong — entirely new empirical contribution.

2. **Principled semantic-saturation stopping criterion for progressive image transmission using scene graphs**
   - Closest prior: GRACE (IEEE TWC 2024) uses scene graphs for SemCom but not for stopping; Progressive SemCom (2026) uses quality thresholds but not scene graph semantics; Zhu et al. (2025) use SG probability graphs for compression but not progressive stopping.
   - Strength: Strong — bridges SGG and SemCom in a novel way. The "semantic sufficiency" stopping criterion has no direct precedent.

3. **Application of Bayesian curve extrapolation / changepoint detection to predict semantic saturation points**
   - Closest prior: LC-PFN (NeurIPS 2023), BOCPD (Adams & MacKay 2007), FIRMBOUND (2025) provide the statistical tools but have never been applied to semantic saturation curves.
   - Strength: Moderate-strong — novel application domain for established methods. A strong paper would contribute a saturation-specific prior or kernel.

4. **Conformal-calibrated LLM-as-judge for semantic saturation scoring with coverage guarantees**
   - Closest prior: EMNLP 2025 conformal-LLM-judge and CVPR 2025 conformal-SGG. No one has combined conformal prediction + LLM judge + SGG saturation.
   - Strength: Moderate — incremental combination of existing techniques, but practically valuable for trust/reliability.

5. **Delta-semantic metric for measuring scene graph information gain across progressive refinement levels**
   - Closest prior: SPICE (ECCV 2016), S^2Match (2020). These are single-comparison metrics, not designed for sequential gain measurement.
   - Strength: Moderate — defines a new evaluation framework needed for the saturation problem.

### Recommended Primary Narrative

The strongest positioning is claims 1 + 2 as the primary contribution (new problem definition: "semantic saturation detection for progressive transmission via SGG") with claim 3 as the methodological contribution (principled statistical framework replacing ad-hoc LLM scoring). Claims 4 and 5 can serve as supplementary contributions or ablation studies.

### Venue Fit
- **WACV / BMVC**: Good fit for the empirical SGG-degradation study + practical stopping criterion
- **ICIP / ICASSP**: Good fit for the progressive transmission / SemCom angle
- **SemCom workshops (e.g., at ICC, Globecom, ISIT)**: Strong fit for the semantic communication framing
- **CVPR workshop (e.g., SG2RL, Vision & Language)**: Good fit if scene graph evaluation methodology is emphasized
