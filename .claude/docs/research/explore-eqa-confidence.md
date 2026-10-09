# Explore until Confident (Ren et al., RSS 2024, arXiv 2403.15941): how VLM confidence is obtained

Date: 2026-10-09. Source: arXiv HTML + `Stanford-ILIAD/explore-eqa` and the `allenzren/prismatic-vlms` fork (clones in session scratchpad).

## Verified facts

- VLM: Prismatic VLM (`prism-dinosiglip+7b`). Paper §VII-A: "Since our setup requires probability outputs from the VLM, we cannot use ... GPT4-V." Verbalized confidence is never prompted for or compared.
- Answer confidence f̂(x) ∈ [0,1]^|Y| = softmax over the multiple-choice letter logits (A/B/C/D) only, read from the FIRST generated token (`get_loss(image, prompt, ["A","B","C","D"])`, `max_new_tokens=1`, `output_scores=True`). T=1, no scaling.
- Relevance Rel(x) = P("Yes") / (P("Yes")+P("No")) for the prompt "Consider the question: '{q}'. Are you confident about answering the question with the current view? Answer with Yes or No." (Eq. 5).
- Single-step conformal prediction (Eq. 7): score ρ_y = Rel·(f̂_y − 1); nonconformity κ = 1 − ρ_{y_true}; q̂ = ⌈(N+1)(1−ε)⌉/N quantile on a calibration set; C(x) = {y : ρ_y ≥ 1 − q̂}.
- Multi-step CP (Eqs. 8–9, App. A, following KnowNo): episode score = min over steps; the episode-level set equals the intersection of per-step sets. Stop when the running intersection has exactly one answer; otherwise answer argmax f̂ at the highest-Rel step at the step budget.
- Calibration: 300 scenarios for calibration, 200 for test (HM-EQA). Hardware ε=0.5, Rel 0.4, entropy 0.1.
- Baselines: entropy threshold (stops prematurely on irrelevant views: "VLM still outputs highly confident, biased answers") and relevance threshold (needs more steps). CP: 58% success at ~71% of max steps vs ~85% for the relevance threshold.
- Why CP: "softmax scores from VLMs are often miscalibrated ... inherited from the underlying LLMs". App. C Fig. A5: top Rel in successful episodes is mostly < 0.5, so raw scores alone are not a reliable indicator. No reliability diagram / ECE reported.
- Exploration scores (LSV/GSV) are also letter / Yes-No token probabilities. Code combines them multiplicatively (`sv = lsv * gsv`), not as Eq. 3.
- GPT-4V became usable once its API exposed logprobs (App. C: 73.9% success).

## Not verified / inconsistent

- CP code is not in the public repo (only per-step `smx_vlm_pred` / `smx_vlm_rel` are logged; calibration presumably offline).
- Eq. 7 as printed gives ρ ∈ [−1, 0]; wording ("large when highly relevant") suggests a missing +1 or deliberate design.
- Fig. 9 caption's reading of ε contradicts standard CP notation.

## Mapping to gqa_stop

- Their signal is token-logit based (candidate-token softmax), the same family as the ollama `logprobs` option verified but declined in v3 (§18.6).
- Their stopping rule is sequence-level (min over steps / intersection of prediction sets) with thresholds set by conformal calibration on a dev set, i.e. the same "freeze on dev" structure as ours but with a coverage guarantee instead of a grid search.
- Applicable pieces without logprobs: the "is this view sufficient?" relevance question as a separate yes/no item; intersection-of-candidates stopping over levels (answer-set stability).
