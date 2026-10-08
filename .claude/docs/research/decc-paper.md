# DeCC (EMNLP 2024): how sub-questions are generated and used, and what it implies for gqa_stop checklists

Date: 2026-10-09. Scope: paper reading only, with no code changes and no model inference.

## Sources

- ACL Anthology: https://aclanthology.org/2024.emnlp-main.211/ (PDF https://aclanthology.org/2024.emnlp-main.211.pdf, pp. 3613-3627)
- arXiv: https://arxiv.org/abs/2407.07840 (v3, 9 Oct 2024; HTML https://arxiv.org/html/2407.07840v3). The ACL PDF and arXiv v3 agree on every passage checked: the implementation details and Tables 7 and 8.
- Code: https://github.com/MyLittleChange/DeCC. **No code has been released.** The repo has only a 2-line README. GitHub API on 2026-10-09 reports `size: 0`, one branch (`main`), and last push 2024-09-24.
- Related: Khan et al., "Exploring Question Decomposition for Zero-Shot VQA", NeurIPS 2023, arXiv 2310.17050. Srinivasan et al., "Selective 'Selective Prediction'" (ReCoVERR), arXiv 2402.15610. Dhuliawala et al., "Chain-of-Verification", ACL Findings 2024, arXiv 2309.11495.

Authors: Qian Yang, Weixiang Yan, Aishwarya Agrawal (Mila / UdeM / UCSB).

---

## 1. Facts: sub-question generation

- **Decomposer model.** The decomposer is InternVL-1.5 (25.5B). It is a VLM and it **sees the image**. Section 3.1 says: "the decomposer, which could be any VLM, decomposes the question Q into a sequence of sub-questions conditioned on I". Appendix A.2 says: "We use InternVL-1.5 ... as the decomposer for decomposition and question paraphrasing". The same decomposer output is reused for all three candidate VLMs: "this process needs to be performed only once per sample before evaluation and can then be reused to evaluate multiple VLMs" (A.5).
- **Prompting style.** Decomposition is few-shot with 4 text-only exemplars. Appendix A.2 says: "we employ few-shot prompting by randomly selecting four samples from SNLI-VE and ScienceQA, with manually written decomposition processes as guidance ... Only text is used in the few-shot prompt, without images. The decomposer determines the number of sub-questions needed." The few-shot exemplars come from SNLI-VE and ScienceQA, and A.2 adds that "The remaining datasets are approached with a zero-shot strategy." **Unclear:** A.2 does not say whether this means the exemplars are shared across all benchmarks or that the other benchmarks get no exemplars.
- **Count.** There is no fixed number of sub-questions. Measured averages are 4.1 to 5.9 sub-questions per sample (Table 3).
- **Constraints.** The prompt asks for these properties: "short and easy to understand", "focus on context visual clues of the image", and "provide clues to answer the main question". The paper states **no atomicity, verifiability, dependency, or ordering constraint.** It also describes **no filtering** of sub-questions. The only post-hoc analysis is a question-type count, by string matching over yes/no, color, number, how, why, what/which, when, where, who, and others. That analysis shows ≥2.1 distinct types per sample. The authors motivate it as a guard against VLM yes-bias (Section 4.5, citing Wang et al. 2023, arXiv 2308.15126): VLMs "respond 'yes' to over 80% of queries about non-existent objects, such as in prompts like 'Is there an object in this image?' ... If the decomposed sub-questions retain the same type as the original question, they may inherit the same bias."
- **Existence checks are used.** DeCC's own exemplars and figures use existence-check sub-questions, for example "Are there any children in the image?", "Is the umpire visible in the image?" and "Is there a whiteboard in the image?". DeCC does not forbid them. Instead it never uses their confidence; see Section 3 below.
- **Second iteration.** The second iteration is optional. It runs for all samples in the single-agent "2 iterations" variants. In Multi-Agent mode it runs only when the VLM and LLM consistency checks disagree. That happened on 25.3% of A-OKVQA and 36.6% of VCR samples. The decomposer then sees the image, Q, and the first-round sub-QA pairs, and adds more sub-questions.

### Verbatim prompt: Table 7, "Few-Shot Prompt for Decomposition"

> Given an image and an associated main question, design pre-questions that focus on important contextual information in the image useful for answering the main question. Pre-questions should provide clues to answer the main question. Each pre-question should be short and easy to understand. Pre-questions should focus on context visual clues of the image. Pre-questions should provide clues to answer the main question.
>
> Example scenario to illustrate the expected interaction pattern:
> Main Question: Is this statement entailment, neutral or contradiction based on the image? Statement: 'A professor is late to class' Options: A: entailment, B: neutral, C: contradiction.
> Pre-question 1: Is there a person in the image wearing clothing typically associated with a professor?
> Pre-question 2: Is the person in the image displaying any behavior that could be interpreted as being late to class, such as being out of breath or looking at a clock?
> Pre-question 3: Is there a classroom setting in the image, such as desks or a blackboard?
>
> Example scenario to illustrate the expected interaction pattern:
> Context: Below is a food web from a tundra ecosystem in Nunavut, a territory in Northern Canada. A food web models how the matter eaten by organisms moves through an ecosystem. The arrows in a food web represent how matter moves between organisms in an ecosystem. Main Question: Based on the arrows, which of the following organisms is a decomposer? Choices: A: mushroom, B: lichen
> Pre-question 1: Does the mushroom eat any other organisms in the food web?
> Pre-question 2: Does the lichen eat any other organisms in the food web?
> Pre-question 3: Does the lichen produce any material that other organisms can use?
> Pre-question 4: Does the mushroom produce any material that other organisms can use?
> Pre-question 5: Does a decomposer produce any material that other organisms can use?
>
> Example scenario to illustrate the expected interaction pattern:
> Main Question: Is this statement entailment, neutral or contradiction based on the image? Statement: 'Two children play in the park.' Options: A: entailment, B: neutral, C: contradiction.
> Pre-question 1: Are there any children in the image?
> Pre-question 2: Are the two children playing in the park?
>
> Example scenario to illustrate the expected interaction pattern:
> User: Context: Use the graph to answer the question below. Main Question: Which month has the highest average precipitation in Santiago? Choices: A: March, B: October, C: June
> Pre-question 1: What kind of graph is shown?
> Pre-question 2: Does the graph show the average precipitation for each month in Santiago?
> Pre-question 3: For which month is the bar highest in the graph?

The "pre-questions should provide clues" sentence appears twice in the original.

### Verbatim prompt: Table 8, "Few-Shot Prompt for Second-Iteration Decomposition" (instruction part)

> You will be given an image and an associated main question, and some sub-question-answer pairs. However, these sub-questions might not be sufficient to answer the main question due to lack of detail or conflicting answers. You need to design additional sub-questions that focus on important contextual information in the image useful for answering the main question. Each pre-question should be short, easy to understand, and provide clues to answer the main question.

The exemplars repeat the professor and food-web cases with sub-answers, then list "Additional Sub-question k: ...". Examples include "What is the person's age in the image?", "Is the person holding any books or papers?" and "Is there any arrow pointing towards the mushroom?".

Table 9 is the paraphrase prompt, used only by the paraphrase self-consistency baseline: "Your goal is to paraphrase the given question into 4 questions. Each question should only change the wording of the original question slightly or just replace a few words. ..."

## 2. Facts: sub-answers and recomposition

- **Sub-answers.** The candidate VLM answers the sub-questions with the image, producing "a sequence of sub-QA pairs". In iteration 2, new sub-questions are answered "conditioned on I and previous sub-QA pairs".
- **Recomposition.** Two independent reasoners recompose the final answer.
  - The **VLM agent** is the candidate VLM. It reasons over the sub-QA pairs and Q, and the paper implies it also has the image. Its output is A'_V.
  - The **LLM agent** is OpenHermes-2.5-Mistral-7B (https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B), run with **no image**. It reasons over the sub-QA pairs and Q. Its output is A'_L.
- **Not published.** The paper gives none of these prompts: sub-question answering, VLM-agent reasoning, or LLM-agent reasoning. The code is not released. It is also not stated whether the reasoners see the direct answer. The figures suggest they do not.
- **Decoding.** Decoding is greedy by default. Table 2 also tests decomposer sampling at T=0.8 and 0.9 with top-p=0.9, and the conclusions are unchanged.

## 3. Facts: consistency, reliability score, and verbalized-confidence baselines

- **Consistency is plain string matching.** Footnote 3 and A.2 say: "Since all datasets are multiple-choice QA tasks or short answers, we use string matching for answer consistency." No LLM judge is used.
- **Reliability R is binary.** Single-agent mode sets R = 1 if A' equals A, otherwise 0. That is "VLM Agent Consistency" or "LLM Agent Consistency".
- **Multi-agent mode works as follows (Algorithm 1).**
  - If Cons'_V = Cons'_L, then R takes that common value.
  - Otherwise the second iteration runs.
  - If Cons''_V = Cons''_L, then R takes that value.
  - If neither agent changed its consistency outcome, R = Cons_L. The rationale is that the LLM is "more objective" and the VLM "might suffer from its inherent biases".
  - If both agents changed, R = Cons''_V. The rationale is that the VLM "potentially overcame its biases with additional sub-QA pairs".
- **No continuous score.** No threshold is tuned. The authors list this as an advantage because baselines need about 1,000 labelled samples to tune thresholds.
- **Metrics.** The paper reports the Brier Score on binary R versus accuracy, and Effective Reliability (ER; Whitehead et al. 2022). **It reports no AUROC and no ECE.** ECE and Coverage@Risk are explicitly rejected because R is binary (A.3).
- **Baselines.**
  - "Perplexity of Direct Answer" uses a per-VLM threshold: 1.10, 1.25 or 1.40.
  - "Generated Numerical Confidence" prompts for 'Answer: X. Confidence: X%' and treats >80% as reliable.
  - "Generated Linguistic Confidence" prompts for 'I am confident/not confident in this answer.'
  - "Self-Consistency based on Paraphrase" uses 4 paraphrases. R = 0 if at least n disagree, with n = 0 for LLaVA and Idefics2 and n = 2 for InternVL.

## 4. Numbers

Setup:
- **Tasks and sample counts.**
  - SNLI-VE, VCR and A-OKVQA: 1,000 validation samples each.
  - Winoground: 800 samples.
  - MMMU: 900 validation samples.
  - MathVista: 1,000 testmini samples.
- **Format.** All tasks except MMMU and MathVista are treated as multiple choice. **GQA is not used.**
- **Candidate VLMs.** LLaVA-1.5-7B, Idefics2-8B and InternVL-1.5-25.5B, all zero-shot.

Table 1, mean over 6 tasks. Brier Score (BS) is lower-is-better and ER is higher-is-better, both in %.

| Candidate VLM (mean Acc) | Perplexity BS / ER | Numerical conf. BS / ER | Linguistic conf. BS / ER | Paraphrase SC BS / ER | Best DeCC variant BS / ER |
|---|---|---|---|---|---|
| LLaVA-1.5-7B (49.9) | 37.4 / 15.1 | 50.1 / -0.2 | 50.8 / -1.5 | 34.3 / 18.2 | LLM-agent 2-iter **31.3 / 21.2** |
| Idefics2-8B (59.8) | 35.6 / 21.5 | 42.8 / 12.6 | 38.3 / 14.7 | 36.4 / 20.7 | Multi-agent **30.5 / 27.0** |
| InternVL-1.5 (67.9) | 27.6 / 40.6 | 37.4 / 27.3 | 43.9 / 10.2 | 32.0 / 36.4 | VLM-agent **26.8 / 41.3** |

The paper states these relative gains over the best baseline:
- **Brier Score.** Mean BS falls by 8.7% for LLaVA, 14.3% for Idefics2 and 2.9% for InternVL.
- **Effective Reliability.** Mean ER rises by 16.5% for LLaVA, 25.6% for Idefics2 and 1.7% for InternVL.

Which DeCC variant is best depends on VLM strength:
- **Weak VLM (LLaVA).** LLM-agent consistency is best, because the VLM "struggle[s] to reason over the sub-QA pairs and suffer[s] from confirmation biases".
- **Mid-strength VLM (Idefics2).** Multi-agent consistency is best.
- **Strong VLM (InternVL).** VLM-agent self-consistency is best.

**On hard tasks, verbalized confidence behaves like "always reliable" or worse.** This is my own arithmetic. If R = 1 everywhere, then BS = 1 - Acc.

| Case | Acc | 1 - Acc | Numerical-confidence BS |
|---|---|---|---|
| LLaVA, MMMU | 34.3 | 65.7 | 67.3 |
| LLaVA, MathVista | 24.5 | 75.5 | 75.8 |
| Idefics2, MMMU | 39.9 | 60.1 | 67.7 |
| InternVL, MMMU | 43.7 | 56.3 | 64.6 |

So self-reported confidence ≥80% is emitted almost regardless of correctness when the model is weak on the task. DeCC variants on the same cells score 30.7 to 52.3 BS.

Table 3, sub-questions per sample (question types per sample):

| Benchmark | Sub-questions | Question types |
|---|---|---|
| SNLI | 4.1 | 2.1 |
| VCR | 5.2 | 2.2 |
| AOKVQA | 4.4 | 2.1 |
| Winoground | 4.3 | 2.5 |
| MMMU | 5.9 | 2.4 |
| MathVista | 4.4 | 2.3 |

Table 6, cost with Idefics2:

| Method | VCR (s/sample) | AOKVQA (s/sample) |
|---|---|---|
| Direct answer | 0.22 | 0.16 |
| DeCC multi-agent, expected total | 6.89 | 5.16 |

The decomposition step dominates the DeCC cost: 3.96 s on VCR and 3.36 s on AOKVQA.

### Failure modes the paper does and does not cover

- **Not studied.** The paper does not study low-resolution or degraded images, unanswerable questions, or sub-questions answered confidently without evidence. Its Limitations section mentions only decomposition quality, which is "not fully explored", and the use of a single LLM agent.
- **Hallucinated sub-answers.** These appear only qualitatively. In Fig. 5 (VCR), sub-answers such as "Married" are hallucinated, all three answers disagree, and DeCC labels the direct answer unreliable, which is correct because it was wrong. In Fig. 4 (A-OKVQA), the VLM correctly answers "Geese" as a sub-answer but its own reasoner still returns the original wrong answer C. Only the image-free LLM reasoner flips to the correct B. The paper uses this case to show VLM confirmation bias.
- **Yes-bias on existence questions.** This is acknowledged via Wang et al. (2023). It is mitigated only by having diverse question types, not by any verification step.

## 5. Facts: related work on verifiable or perception-level sub-questions

- **ReCoVERR (Srinivasan et al. 2024, cited by DeCC).** It runs only when VLM confidence is below the abstention threshold. An LLM generates K sub-questions conditioned on Q and the evidence collected so far. The VLM answers each one, and each QA pair becomes a declarative "evidence". An evidence is kept only if it is:
  - reliable: VLM confidence ≥ 1 - r, where r is the risk tolerance; and
  - relevant: |P_NLI(H | S_j) - P_NLI(H | not S_j)| ≥ δ_min, a defeasible-reasoning test with an NLI model.
  The answer is accepted if the concatenated evidence entails the hypothesis under the NLI model, and abstained after N rounds otherwise. The scheme needs **calibrated** VLM confidence. The authors measure calibration and use token-probability or SelfPrompting confidence, Platt-scaled, not free-form verbalized confidence. The evidence pool is also seeded with "vision tools": LVIS object detection and Qwen-VL region captions. Code: https://github.com/tejas1995/ReCoVERR.
- **Khan et al. 2023, NeurIPS (not cited by DeCC).**
  - **Sub-question type.** Each decomposition is one "Perception Question" for a "Reasoning Question". The perception/reasoning split comes from VQA-Introspect (Selvaraju et al.).
  - **Verbatim decomposition template:**
    ```
    template = 'Reasoning Question: is the banana ripe enough to eat? Perception Question: is the banana yellow?\nReasoning Question: is it cold outside? Perception Question: are any people wearing jackets?\nReasoning Question: {question} Perception Question: '
    ```
    The text says three handcrafted demonstrations, but the printed template has two.
  - **Verbatim recomposition template:**
    ```
    exemplar = "Context: is the sky blue? no. are there clouds in the sky? yes. Question: what weather is likely? Short answer: rain"
    template = exemplar + " Context: {subquestion}? {subanswer}. Question: {question}? Short answer:"
    ```
  - **Findings.** Naive decomposition on every question hurts, because it flips correct answers. They propose *selective* decomposition: decompose only when p(â) ≤ τ, using token-probability confidence. Decomposition "may be a primarily linguistic ability": text-only FLAN-T5 decomposers are only about 10% of the base error rate worse than image+text BLIP-2 decomposers. They also note that datasets testing "low-level perception skills" (as opposed to reasoning) are not suitable for decomposition.
- **Chain-of-Verification (Dhuliawala et al., ACL Findings 2024; not cited by DeCC).** The model plans verification questions, then answers them **independently** ("factored"), so that the answers are not biased by the draft response or by other answers. Then it revises the draft. This is the clearest published argument for answering each check in a separate call without the original answer in context.

---

## 6. Assessment for gqa_stop (my interpretation, not from the paper)

Current design, as read from `prompts/gqa_stop/planner.txt`, `prompts/gqa_stop/checklist.txt` and `src/milestone1/gqa_stop/inference.py`:
1. A text-only planner makes 2-3 items: "identify the correct target, then inspect the requested attribute". The last item asks for the requested attribute.
2. **One** VLM call answers all items together, with the original question supplied "as context".
3. Each item returns `answer`, `resolved` and a self-reported `confidence`. The score is r_j = confidence_j if j and its ancestors are resolved, else 0. The stop score is min_j r_j (mean also logged), and the run stops once it passes a dev-tuned threshold.

### Likely causes of "resolved=true, conf 0.9 at Level 1"

1. **The stop signal is verbalized confidence, which DeCC identifies as the worst signal.** DeCC uses decomposition to create a *second, independent answer path* and compares it with the direct answer. It never trusts sub-answer confidence. In Table 1, generated numerical confidence is the worst or second-worst method for every VLM. On hard tasks its BS is at or above the "always reliable" level. Very low resolution is our hard regime, so the observed behaviour is the expected one. Splitting the question into items multiplies the number of overconfident self-reports but adds no independent evidence.
2. **Presupposition leakage plus yes-bias on the existence item.** GQA attribute questions presuppose the target, as in "What color is the car left of the man?". The checklist call supplies that question as context. "Is there a car?" can then be answered "yes" from the question text alone. VLMs also answer "yes" to over 80% of non-existent-object probes, per Wang et al. 2023 as cited by DeCC. The item is therefore not a perception test. DeCC itself uses existence items, but it neutralises their weakness by not trusting them and by requiring type diversity. We use exactly their confidence.
3. **Joint answering in one call.** All items and the original question are answered in a single generation. The items' answers and confidences are therefore not independent of each other. That violates both CoVe's "factored" principle and the multi-agent independence DeCC relies on.
4. **Possibly not the cause: the text-only planner.** DeCC's decomposer sees the image, but Khan et al. find decomposition is mostly linguistic. For GQA, whose questions are compositional with explicit referents, a text-only planner is likely adequate. Showing the planner a low-resolution image would not fix points 1 to 3. It could also leak hallucinated content into the items.
5. **Structural near-tautology.** Our last item is essentially the original question, which differs from DeCC's "contextual clue" sub-questions. A DeCC-style recomposition would therefore mostly copy the last sub-answer. Consistency would then degenerate into paraphrase self-consistency. That is still better than verbalized confidence: its mean BS is 34.3 versus 50.1 for LLaVA. It is weaker than full DeCC.

### Concrete changes suggested by DeCC and adjacent work

These are ordered by expected impact for least effort.

1. **Replace the confidence-min score with a consistency score.** At each level, compare the direct answer, from the existing `answer_only` or `direct` call, with an answer recomposed from sub-answers.
   - Recompose with a text-only LLM agent that sees Q and the sub-QA pairs but not the image and not the direct answer, as in DeCC's LLM agent. Use string match after the existing GQA normalisation.
   - DeCC found the LLM agent best for weak VLMs. qwen3.5:9b at Level 1 is effectively a weak VLM.
   - The rule could be binary (stop when consistent), or k-of-n across items and agents, or a consecutive-k across levels. The last option fits our online stop rule and needs no confidence threshold.
2. **Make the target item an open-ended identification, not an existence check, and keep it blind to the answer and the presupposition.** For example, use "What object is to the left of the man?" rather than "Is there a car?".
   - The answer is verified automatically against the target noun parsed from the question. This makes it a perception task with a *checkable expected value*, a grounding check rather than a self-report.
   - At Level 1, an unrecognisable image will yield wrong or varied nouns, so the check fails naturally.
   - This goes beyond DeCC, which has no such check. It follows from DeCC's own yes-bias argument and from Khan et al.'s "perception question" framing.
3. **Make the attribute item refer to the object by its relation, not by name.** "What color is the object to the left of the man?" does not let the VLM borrow priors from "car". Its answer is then compared with the direct answer.
4. **Answer each item in a separate call without the original question in context.** This applies CoVe's factored verification and removes presupposition leakage. It costs about 2-3 extra calls per level. DeCC's measured overhead is about 31-32x a direct answer (6.89/0.22 s on VCR, 5.16/0.16 s on AOKVQA), so this is cheap by comparison.
5. **Keep `resolved` and `confidence` only as an auxiliary gate or diagnostic.** Do not make them the stop signal. If a confidence is needed, follow ReCoVERR: use token-probability confidence (ollama logprobs, if exposed) calibrated on dev, not free-form verbalized confidence.
6. **Alternative for GQA specifically (background knowledge, not verified in this session).** GQA questions ship with a functional program, the `semantic` field (select → relate → query). Deriving items from it gives exact, non-hallucinated sub-questions and gold intermediate targets for analysis. This would remove planner variance entirely.

### Caveats for this assessment

- **Consistent but wrong.** DeCC was never tested on degraded images. At very low resolution, the direct and recomposed answers can agree on the same prior-driven guess, such as a typical color. Changes 2 to 4 above are meant to break that shared prior, but whether they do is untested.
- **Binary score versus our threshold grid.** DeCC's R is binary, while our protocol freezes a threshold on dev. A consistency-based score must be made ordinal to fit that protocol. Options are the fraction of agreeing agents or items, or a run length across levels.
- **Unpublished details.** The DeCC sub-answer and reasoning prompts are unpublished, so any reimplementation of those steps is our own design, not a faithful replication.
