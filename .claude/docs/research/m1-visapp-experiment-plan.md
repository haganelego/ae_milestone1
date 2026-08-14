# VISAPP Experiment Plan — Query-Conditioned Bayesian Stopping

> Execution resource for the **Linux experiment machine** (where the VG dataset,
> EGTR env, and cached per-level predictions live). Self-contained: method,
> data, formulas, step-by-step checklist, risks, evaluation.
>
> Target: VISAPP 2027 (deadline ~Oct 2026). Date: 2026-08-14.
> Related: `m1-bayesian-methods-explainer.md`, `m1-conference-survey.md`,
> `m1-vgqa-query-conditioned.md`. Memory: `visapp-query-conditioned-pivot`.

---

## 0. Why this pivot (one paragraph)

The original goal — "detect when the FULL scene graph's semantics saturate as
resolution recovers" — is ill-posed: the deployable object-confidence signal does
NOT saturate within 20 levels for ~half the images, so a correct Bayesian stop
refuses to fire (stage-1: frac_stop 0.08 absolute-ε; 0.51 best relative-ε →
BORDERLINE). Fix: make the target **query-conditioned**. A question touches only a
few scene-graph nodes; those saturate early and cleanly, so stopping becomes
well-posed AND the ideal stop varies with question difficulty (a coarse question
saturates early, a fine-detail question late). This also sharpens novelty
(goal-oriented / task-oriented semantics: "stop when the semantics suffice for the
task").

## 1. Method overview

For each (image, question) pair, across 20 resolution levels t=1..20:

- **Signal (deployable, causal)** = **B+C**:
  - **B** = confidence of the *query-relevant* detected objects/relations (EGTR).
  - **C** = stability of the *query-relevant* predicted subgraph across levels.
- **Oracle (ideal stop, may peek GT)** = **O2**: earliest level where the
  query-relevant subgraph's recall (vs the GQA GT scene graph) saturates.
- **Stop rule**: fit the Bayesian saturating curve to the fused signal y_t; stop at
  the first level with `P(residual gain < ε | y_{1:t}) ≥ δ` (relative ε).
- **Evaluation**: MAE of predicted stop vs O2, stratified by question difficulty
  (easy / medium / hard terciles).

**Key feasibility facts:**
- **No new EGTR inference** — per-level predictions are already cached in
  `outputs/egtr_predictions/`. Query-conditioning only changes *which nodes we
  aggregate over*, so it is a re-aggregation of existing outputs.
- **No LLM / ollama** — the signal is detector confidence + subgraph stability, not
  an LLM score. (The old LLM-saturation-score path is dropped for the core method;
  it may return only as an optional ablation.)
- The whole thing reuses the existing harness (`convergence.py`, `bayesian_stop.py`,
  `scripts/16`, `scripts/18`) *restricted to query-relevant nodes*.

## 2. Data acquisition (GQA)

GQA questions are grounded to VG scene-graph node IDs → query grounding is free.
Raw VG QA is NOT used (free-form, ungrounded, noisy).

Download to the Linux box (`downloads.cs.stanford.edu/nlp/data/gqa/`):
- `sceneGraphs.zip` (~43 MB) → `train_sceneGraphs.json`, `val_sceneGraphs.json`
- `questions1.2.zip` (~1.4 GB) → balanced question files (use the *balanced* split)

Images already exist on the VG disk (`/home/kanaoka/usr/dataset/vg/VG_100K/`); GQA
uses the same VG image IDs. GQA scene graphs are cleaned/normalized VG graphs.

**GQA question fields we need** (per question):
- `imageId` — join to VG image.
- `question`, `answer`, `fullAnswer`.
- `semantic[]` — functional program; `argument` strings embed object IDs, e.g.
  `"wall (722332)"`.
- `annotations.question` / `annotations.answer` — maps text spans → scene-graph
  object IDs.
- `types.structural` — verify / query / compare / choose / logical (used in
  difficulty scoring).

**GQA scene-graph fields** (per image): `objects{ id: {name, x,y,w,h, attributes,
relations:[{object, name}] } }`. Boxes here are the GT for O2.

## 3. Query grounding → query-relevant subgraph

For a question q on image i:
1. **Query object IDs** `N_q`: regex `\((\d+)\)` over each `semantic[].argument`,
   plus all object IDs in `annotations.question` and `annotations.answer`.
2. **Query relations** `R_q`: from the GQA scene graph of image i, keep every
   relation whose subject and object are both in `N_q` (optionally expand to a
   1-hop neighborhood of `N_q` to include the relations that answer the question).
3. The GT query-relevant subgraph = (`N_q` nodes with their boxes/categories,
   `R_q` relations). This is the GT reference for both grounding and O2.

> Recommended: use the **GQA scene graph itself as the GT** for query-relevant
> recall (O2). This sidesteps node-ID alignment with the old VG train/val/test GT
> — see Risk R2. EGTR predictions are matched to GQA GT boxes by IoU+category
> (predictions carry no GT IDs anyway).

## 4. Signal computation (B+C), per level t

Let the query-relevant GT node set be `N_q` (size n) and relation set `R_q` (size m).
Match EGTR predictions at level t to GT query nodes by category + box IoU ≥ 0.5.

**B — query-relevant confidence** (rises as required nodes become confidently
detected; missing nodes contribute 0, so coverage is folded in):
```
B_node(t) = (1/n) * Σ_{v in N_q} [ matched(v,t) ? conf_pred(v,t) : 0 ]
B_rel(t)  = (1/m) * Σ_{e in R_q} [ matched(e,t) ? conf_pred(e,t) : 0 ]   (m>0)
B(t)      = w_bn * B_node(t) + w_br * B_rel(t)          # default 0.5/0.5
```

**C — query-relevant subgraph stability** (rises to 1 as the query view stops
changing; no QA executor needed):
```
S_t = { query-relevant predicted triplets at level t }          # (subj,pred,obj) on N_q
C(t) = Jaccard(S_t, S_{t-1})   for t≥2;   C(1) = 0
```
(Optionally smooth C over the last 2 transitions to reduce single-level flicker.)

**Fused readiness signal** the Bayesian curve is fit to:
```
y_t = w_B * B(t) + w_C * C(t)          # both in [0,1]; default w_B=w_C=0.5
```
Both B and C rise-and-plateau, so `f(t)=s∞−(s∞−s₀)e^{−λt}` is an appropriate model.
Run per-signal ablations too (B only, C only) since the harness makes it free.

## 5. Oracle O2 — query-relevant recall saturation (ideal stop)

```
recall_node(t) = |matched query nodes at t| / n
recall_rel(t)  = |matched query relations at t| / m           (m>0)
QR(t)          = w_rn * recall_node(t) + w_rr * recall_rel(t)  # default 0.5/0.5
O2(q) = first level t≥min_level with QR(t) ≥ τ · max_{t'} QR(t')   # τ=0.8, RelOracle-style
        (or: QR stable within η for k consecutive levels — report both variants)
```
O2 is per-question; it peeks at the full curve (legitimate for an oracle). It is the
existing RelOracle(0.8) **restricted to query-relevant nodes** → reuse that code.

## 6. Bayesian stop (recap; reuse `src/milestone1/bayesian_stop.py`)

- Model `y_t = f(t;θ)+noise`, `f=s∞−(s∞−s₀)e^{−λt}`, Student-t(df=5) noise.
- **Relative ε**: stop when `P(s∞−f(t) < c·s∞) ≥ δ` (c≈0.2, δ=0.9, min_level=5).
  (Absolute ε failed; relative ε is the causal analogue of RelOracle — see explainer.)
- **Empirical-Bayes prior via K-fold cross-fitting** over (image,question) pairs:
  each pair gets a prior from the OTHER folds (no peeking). Pure-numpy grid over
  (s∞, s0, λ); profile/gridded σ.
- Output per pair: `stop_level`, `prob_trace`, posterior `s∞`, and the early
  **forecast** `t_sat` (predict the saturation level before reaching it).

## 7. Resolution-difficulty curation (novelty N2)

Composite per-question difficulty score (higher = needs more resolution):
```
diff(q) = 0.5 * (1 − min_rel_object_size)     # smallest query object's box area / image area
        + 0.3 * norm(|N_q|)                    # more query objects → harder
        + 0.2 * struct_weight(types.structural) # verify<query<compare/logical
```
Bin into **easy / medium / hard terciles**. Validate empirically: hard questions'
query objects should first become detectable at higher levels than easy ones
(sanity plot: detection level vs difficulty). This taxonomy is a stated contribution.

## 8. Evaluation protocol

**Primary**: MAE(predicted stop, O2) overall AND per difficulty tercile. Show the
ideal stop O2 shifts later with difficulty (headline figure), and that the Bayesian
predictor tracks it.

**Also report**: frac_stop, early/late rates, MedAE; forecast MAE (t_sat from level
5); calibration curve (claimed δ vs empirical coverage); ε–δ Pareto
(transmission-saved vs semantic-loss).

**Baselines / ablations**:
- Convergence heuristic (consecutive-k gap) on the same fused y_t.
- Node-confidence baseline (existing).
- **Generic (non-query) saturation** on global conf — the borderline result — as the
  contrast that motivates query-conditioning (expect it to under-stop; query version
  fixes it).
- B-only, C-only vs B+C.

## 9. Integration risks & pre-checks (run these first)

- **R1 — Image-set intersection.** The current 100 selected images may not all have
  balanced-GQA questions. First intersect the 100 image IDs with GQA; if too few,
  reselect images that have BOTH a rich GT scene graph (min_gt_objects/relationships)
  AND ≥1 balanced question. Log the final count.
- **R2 — Node-ID alignment.** Do NOT assume the old VG GT (train/val/test.json) node
  IDs match GQA. **Mitigation**: use the GQA scene graph as the GT for O2 (grounding
  + recall come from the same GQA graph); match EGTR predictions to GQA boxes by
  IoU+category. Verify a few images visually.
- **R3 — Relations may be sparse.** Some questions touch objects with no GT relation
  among them (m=0). Handle m=0: fall back to node-only B/QR, and flag the fraction.
- **R4 — Difficulty score validity.** Confirm diff(q) correlates with the actual
  level at which EGTR first detects the query objects; adjust weights if not.
- **R5 — Small N for empirical Bayes.** (image,question) pairs are more numerous than
  images (many Q per image) → this actually *helps* the prior. Keep folds at the
  IMAGE level (not pair level) to avoid leakage between Q on the same image.

## 10. Step-by-step checklist (Linux box)

1. **Data**: download+unzip GQA `sceneGraphs.zip` + balanced `questions1.2.zip`.
2. **Intersect** GQA images with the working image set; finalize (image,question)
   list; write `data/processed/gqa_questions.json` (R1).
3. **Grounding**: parse each question → `N_q`, `R_q`; build GT query-relevant
   subgraphs; cache to `data/processed/query_subgraphs/` (§3).
4. **Difficulty**: compute diff(q), assign terciles; save (§7); run R4 sanity plot.
5. **Signals**: from CACHED `outputs/egtr_predictions/` (no re-inference), compute
   B(t), C(t), fused y_t per (image,question,level) (§4). New module, e.g.
   `src/milestone1/query_signal.py`.
6. **Oracle**: compute O2(q) per pair from GQA GT (§5). New: `scripts/2x_query_oracle.py`.
7. **Bayesian stop**: cross-fit priors (image-level folds), run `bayesian_stop` on
   y_t (§6); new runner `scripts/2x_query_bayesian_stop.py`.
8. **Eval**: MAE vs O2 per tercile + all diagnostics/baselines (§8); CSVs+figures.
9. **Sanity gates**: frac_stop > 0.8 now (query nodes saturate), O2 shifts with
   difficulty, calibration ≈ diagonal.

## 11. Paper framing (from survey)

- **N1 (core)**: query-conditioned optimal stopping for single-image resolution
  recovery via SGG semantics — no prior work combines resolution-ladder × SGG ×
  question-grounding × optimal stopping. Closest: EQA (stops navigation), task-
  oriented SemCom (stops transmission), variable-res VQA (allocates pixels).
- **N2**: resolution-difficulty taxonomy for visual questions (SG-grounded).
- **N3**: dual-signal (confidence+stability) stop benchmarked against a
  query-conditioned oracle.

## 12. Open decisions (confirm before/while running)

- Fusion weights `w_B,w_C` and `w_bn,w_br` — start 0.5/0.5, tune on calibration fold.
- O2 threshold τ (0.8) vs stability-k variant — report both.
- Whether to expand beyond 100 images for statistical power (more Q per image helps).
- Balanced vs full GQA split; cap #questions per image to avoid image imbalance.
