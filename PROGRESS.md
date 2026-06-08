# PROGRESS

Rolling micro-progress (latest 5 checkpoints). Newest first.

---

## 2026-06-08 — Relation-decouple (proper top-K SGG usage)

### Change
- `egtr_infer.py`: relationship export now takes top-K triplets by ranking score WITHOUT the endpoint object_conf>0.3 gate (object LIST unchanged → mean_object_confidence / baseline / B-1 convergence unchanged). Fixes person-relation suppression (person boxes conf ~0.2 were dropping all their relations).
- Re-exported 100×20 EGTR preds; recomputed metrics/relative/convergence/aggregate; re-scored LLM 4b-h5 & 9b-h5 (relation inputs changed); VLM unchanged (image input). Parse failures 0.

### Impact
- L20 zero-relation images **23/100 → 4/100** (residual genuine: rel scores <0.005); mean L20 rels 4.1→7.9; true-GT triplet recall@L20 0.25→0.286; node_recall 0.569 unchanged.
- RelOracle(0.8): 74/100 saturate, mean 14.6 (slightly later than 80/13.8 — relations add honest late info).
- **Conclusions robust**: vs RelOracle(0.8) MAE — hybrid_4b_k2 4.15 (#1), hybrid_9b_k2 4.41, llm_sgg_4b_h5_k2 4.81, vlm_9b_single_k2 5.17; conf_convergence 5.99 (LLM-free, 99/100); baseline 7.03. Structured-SGG+LLM (hybrid k2) still beats image-VLM; k2 helps; baseline weakest.
- Checkpoints: `e1b8e56` (pre-B1), `c3441fc` (B-1). Decouple committed next.
- Refs: `.claude/docs/research/m1-scale100.md` (decouple section).

---

## 2026-06-08 — Git checkpoint + convergence-based stop (Plan B-1)

### Git
- Initial checkpoint commit `e1b8e56` (357 files; bulky regenerable artifacts gitignored: degraded_images, egtr_predictions, llm/vlm score JSONs). Restore point before B-1 dev.

### Codex review (no plugin; danger-full-access bypasses broken bwrap)
- bwrap (codex's command sandbox) can't create namespaces in this env → codex can't read files via its shell. Workaround: `codex exec --sandbox danger-full-access` (verified) or feed code via stdin. `/codex:review` plugin is installed but would hit the same bwrap wall unless configured.
- Findings: #1 relation-ranking truncation = **no practical impact** (L20 rels avg 4.1, never hits the 90-pair buffer). Real issue = relation export over-filters: person nodes score ~0.2-0.25 < object_conf 0.3, so person relations are dropped → 23/100 images have 0 rels at L20. **EGTR is fine** (ranks "man-wearing-shirt" 0.13); it's our export gating. Also gain-based artificial-zero bug (#2/#3), never-vs-20 sentinel (#5), bool("false") (#7), VLM missing num_ctx.

### Plan B-1 (deployable convergence stop) — DONE
- New `src/milestone1/convergence.py`: stop when last-k transitions all < epsilon (past-only; deployable). Oracle-only `relative_to_max_stop` kept separate. Fixed gain-based artificial-zero in scripts 07/15.
- Results (vs RelOracle 0.8, n=100): **conf_convergence stops 99/100** (vs absolute baseline 70/100) and MAE **7.46 → 5.22** (−30%). LLM-convergence too noisy at epsilon=0.02/k=3 (only 9-25% trigger). Best overall still the LLM/hybrid threshold methods (~4.2-4.9, need LLM each level); conf_convergence is the best LLM-free deployable baseline.
- Confidence signal insight: `mean_object_confidence` plateaus ~0.40 → absolute thresholds ≥0.5 dead (orig baseline=0.7 → 0/100). Convergence removes the absolute-threshold fragility.
- Refs: `.claude/docs/research/m1-convergence.md`, `m1-codex-review.md`; `outputs/metrics/method_comparison_convergence_100.csv`.

### Next actions
1. (Optional) relation-export decouple (top-K, endpoints below 0.3) to fix the 23 zero-rel images — needs EGTR re-export + recompute (and LLM re-score for relation inputs). Keeps mean_object_confidence (node track) unchanged.
2. (Optional) tune LLM-convergence epsilon/k; or adopt hybrid threshold as primary.
3. Commit B-1 if accepted (currently uncommitted; revert point = e1b8e56).

---

## 2026-06-05 (100-image scale-up)

### Done
- Scaled to 100 images (seeded superset of the 10; reused cache, computed 90 new). EGTR 2000 preds; focused methods only (LLM-SGG 4b-h5/9b-h5, VLM-single 4b/9b; full OFAT ablation + VLM-sequence skipped). Note: the focused scoring background process died once (subagent exit) at VLM-9b 160/1600; resumed via harness-tracked Bash bg, finished clean.

### Results (n=100) — after LLM num_ctx bug fix
- Degeneracy: true-GT Oracle(0.5) saturates only 28/100 (still degenerate); **RelOracle(0.8) 80/100** (non-degenerate, was 90% at n=10). Plan B confirmed at scale.
- 🐛→✅ **LLM parse-failure bug fixed**: 27% failures were ollama default `num_ctx=4096` truncating long history-5/top-20-30 prompts (~6k–9.2k tok) to 4095, leaving 0 generation room (1-token "Based" stub). Fixed with `num_ctx=12288` in `src/milestone1/{llm,llm_engine}.py`; deleted+re-scored 830 failures → ~1% now.
- **CORRECTED top vs RelOracle(0.8)**: **hybrid_9b_k2 MAE 4.18 (#1)**, **hybrid_4b_k2 4.35 (#2)**, **llm_sgg_4b_h5_k2 4.47 (#3)**; best VLM `vlm_9b_single_k2` 6th (4.96); baseline 12th (7.46).
- **Research thesis RESTORED**: structured-SGG+LLM (esp. confidence+LLM hybrid) now beats image-direct VLM. The pre-fix "VLM best" was an artifact of the LLM context bug.
- consecutive-k=2 consistently helps; 9b over-stops vs 4b at k1 (corrected by k2/hybrid); node-vs-relation gap +0.5.
- Refs: `.claude/docs/research/m1-scale100.md`; `outputs/metrics/method_comparison_{100,relative_100}.csv`; figures `*_100.png`.

### Next actions
1. For the 20% non-saturating tail (RelSGScore<0.8 vs own L20), try threshold 0.7 or gain-based stop.
2. Optional: full OFAT ablation + VLM-sequence at 100 (now that LLM scoring is clean); synonym relaxation for GT-correctness reporting.
3. Consider this milestone's core PoC question answered (structured-SGG+LLM > VLM for saturation; relative reference non-degenerate); write up.

---

## 2026-06-05 (validation + relative saturation)

### Checkpoint/paper validation (user asked: is the ckpt actually loaded? do results conflict with the paper?)
- ckpt load is CORRECT: strict load 0 missing/0 unexpected; finetuned relation-head weights == ckpt tensors.
- Results do NOT conflict with EGTR paper. ckpt dir ships authors' own eval JSON == paper Table 1 (R@20/50/100=23.54/30.21/34.35, mR@50=7.94, AP50=30.83). Our 500-img RTX5080 run via `egtr/scripts/eval_egtr_subset.py`: R@20/50/100=20.4/26.3/30.5 (in-range). PoC's low label-only triplet recall (~0.25) is a metric/protocol/sample artifact + degradation + VG predicate long-tail (mR@50≈7.9 ⇒ most predicates near-zero recall is EXPECTED, universal across SGG). Refs: `.claude/docs/research/{egtr-paper-claims,egtr-repro}.md`.

### Relative saturation (Plan B) — replaces degenerate GT oracle
- New reference = each image's own Level-20 EGTR output (pseudo-GT); RelSGScore = 0.5*RelNodeRecall+0.5*RelTripletRecall vs L20 (label-only). Measures CONVERGENCE/stability, not correctness.
- Non-degeneracy FIXED: 9/10 images saturate at RelSGScore≥0.8 (vs 4/10 before). RelOracle(0.8) mean stop 12.2 vs GT-oracle(0.5) 16.0 (earlier).
- Ranking changed: best now `vlm_9b_single_k2` (MAE 4.0), `hybrid_4b_k1` (4.1), `llm_sgg_4b_h5_k1` (4.4). node-conf baseline dropped to 14th (MAE 6.2) — it was only accidentally calibrated to the degenerate GT oracle.
- Node-vs-relation convergence gap small (+0.8) but high variance per image.
- Refs: `.claude/docs/research/m1-relative-saturation.md`; `outputs/metrics/{relative_metrics,relative_oracle_stop,method_comparison_relative}.csv`; figures `relative_*`.

### Next actions
1. Scale to 100 images (now that the framework is non-degenerate) for statistical power.
2. Confirm `vlm_9b_single_k2` / `hybrid_4b_k1` advantage holds at scale.
3. Optional: synonym matching relaxation; report L20 "correctness" via paper-protocol R@K alongside relative convergence.

---

## 2026-06-05 (eval) — Rigorous evaluation: VLM baseline + ablation + analysis

### Done (10 images, scripts 11–14 + aggregation)
- **VLM baseline** (qwen3.5 4b/9b, single + sequence; image-direct saturation judging) vs **LLM-on-SGG** (no image).
- **LLM-SGG ablation** (OFAT): model 4b/9b, history 1/3/5, relation on/off, confidence on/off (cached scores).
- **Post-hoc**: consecutive-k, SGScore weights, **confidence+LLM hybrid stop**, 21-method comparison.
- **Error/dataset analysis**: missing-at-stop, per-category & per-predicate recall, hard-image characterization, node-vs-relation gap.

### Headline findings
- ⚠️ **Bottleneck is EGTR's relation-prediction ceiling, not resolution**: 69% of GT predicates & 35% of GT object categories NEVER recalled; 6/10 images never reach oracle(0.5). Evaluation of "stop accuracy" is confounded by this degenerate oracle (n=10).
- **Structured SGG > raw image**: all VLM variants stop far too early (mean 5–8.7), sequence mode degenerate (always L5). Supports the thesis that structured SGG output is more usable than image-direct for saturation judging.
- **Bigger LLM is worse**: 9b overconfident → stops earlier (mean 7.8) than 4b (12.8); 4b closer to oracle(16.0).
- **consecutive-k=2 and confidence+LLM hybrid clearly help** (e.g. LLM-4b MAE 7.0→5.3 with k2; hybrid_4b_k2 MAE 4.8). Simple node-conf baseline still lowest MAE (4.0) but early-biased.
- relation on/off & SGScore weights: ~negligible — but relation effect is confounded (EGTR barely produces correct relations).

### Reports/outputs
- `.claude/docs/research/m1-analysis.md` (errors/weakness), `.claude/docs/research/m1-evaluation.md` (method comparison).
- `outputs/metrics/{method_comparison,hybrid_stop,llm_ablation_stop,vlm_stop,sgscore_weight_ablation}.csv`; figures (`method_comparison_mae`, `stop_distribution_all`, `ablation_axes`, `category/predicate_recall_hardest`, `difficulty_vs_features`).
- Root `pyproject.toml` fixed (broken `ty>=0.1` constraint; added Pillow/requests/pyyaml).

### Next actions
1. **Address the oracle-confound** before trusting stop metrics: relax predicate/category synonym matching (e.g. person/man), and/or report stop accuracy ONLY on the subset of images that actually saturate.
2. Scale to 100 images for statistical power.
3. Given VLM-direct is weak and relations barely recalled, consider a stronger SGG model or relation-specific confidence; reconsider whether qwen3.5 sizes are the right LLM judges.

---

## 2026-06-05 — Milestone 1 PoC: full pipeline running on 10 images

### Done
- Phase 1 (understand): confirmed env ready — VG at `/home/kanaoka/usr/dataset/vg`, EGTR finetuned ckpt (`epoch=03`), egtr `.venv` (torch 2.7.1+cu128), ollama `qwen3.5:4b`. Reverse-engineered + verified EGTR inference I/O → `.claude/docs/research/egtr-inference.md`.
- Built `src/milestone1/` (config, vocab, gt, egtr_infer, metrics, llm) + `scripts/01..10` + `configs/config.yaml`.
- Ran end-to-end on 10 images: 200 degraded jpgs, 200 EGTR preds, 10 GT graphs, all metrics CSVs, 160 LLM scores, 11 figures.

### Results (10 images)
- Q1 info rises w/ resolution: YES (NodeRecall 0.17→0.64, TripletRecall 0.00→0.25, SGScore 0.08→0.44, L1→L20).
- Q2 node saturates before relation: YES (the key PoC signal).
- Q3 baseline vs oracle: MAE 4.0, 40% early.
- Q4 LLM vs oracle: MAE 8.4, 80% early — 4b model noisy (`<think>` mode handled via `/api/chat think:false`); many parse failures.

### Calibration (vs original plan)
- EGTR rel scores tiny (<0.1) → thresholds lowered: rel_conf 0.2→0.005, oracle 0.8→0.5, baseline 0.7→0.4, llm 0.8→0.7.

### Next actions
1. Scale to 100 images (`dataset.num_images: 100`), re-run.
2. Improve LLM stage: try larger model (qwen3:14b / gpt-oss:20b) and/or refine prompt + JSON robustness to reduce early-stop bias and parse failures.
3. Consider consecutive-k stop conditions; add attribute info (optional); bbox-aware matching as a later extension.
4. Inspect representative-example figures qualitatively (node-visible-but-relation-missing cases).
