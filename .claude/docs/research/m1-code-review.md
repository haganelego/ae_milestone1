# Milestone-1 Code Review & Documentation Consistency Audit

Generated 2026-06-08 by Opus subagent (manual review; Codex CLI sandbox failed).

---

## Part A: Code Review (Fallback -- Own Review)

> **Note:** Codex CLI was attempted but sandbox failed with `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`. This review was conducted manually by reading all source files.

### Correctness Bugs

1. **`egtr-inference.md` logits shape inconsistency (doc-only):** Line 106 comment says `probs = logits.softmax(-1)  # (200, 151)` but verified output shapes (line 356) show `logits: torch.Size([1, 200, 150])`. The 151-class note is inherited from standard DETR (which has a "no-object" class), but EGTR's config has `num_labels=150` and logits are `(200, 150)`. The code in `egtr_infer.py:107-109` is correct: `probs[:, :self.num_labels]` with `num_labels=150` takes all 150 columns. **Impact: None in code, misleading in doc only.**

2. **`_parse_llm_json` regex may miss nested JSON:** `llm.py:42` uses `r"\{[^{}]*\}"` which rejects any JSON containing nested braces (e.g., `{"key": {"nested": 1}}`). Since the expected output is `{saturation_score, stop_decision, reason}` with no nesting, this works in practice. But if the LLM returns reasoning with nested JSON in the `reason` field, it would fail and fall through to the full-text parse (line 50). **Impact: Low; fallback path handles it.**

3. **`15_relative_saturation.py` hardcoded "/10" in print statements:** Lines 516-517 print `saturated={gt_saturated}/10` and `saturated={rel_saturated}/10` instead of using `len(image_ids)`. When run with 100 images, the print is wrong. Not a data-output bug (CSVs are correct), but misleading console output. Same issue with the report writer at lines 846-847 which writes `{gt_saturated}/10` to the markdown report.

4. **`03_run_egtr_inference.py` potential UnboundLocalError:** Line 84 references `result` outside the inner loop. If `img_path.exists()` is False for all levels, `result` would be unbound. Protected by the fact that `out_path.exists()` check skips but still increments `done`, but the `result` variable used in the print at line 84 could reference a stale value from the previous image. **Impact: Cosmetic print only, not a data bug.**

### Metric/Logic Issues

1. **Set-based metrics are correct.** `metrics.py` properly deduplicates to sets of labels (`_to_label_set`) and triplets (`_to_triplet_set`). `sg_score()` correctly computes `0.5 * node_recall + 0.5 * triplet_recall`. `06_compute_sg_metrics.py` correctly passes `node_w, rel_w` from config. No bugs found.

2. **Relative saturation logic is correct.** `15_relative_saturation.py` and `aggregate_100.py` both use Level-20 as pseudo-GT, compute `RelNodeRecall = |pred_nodes & ref_nodes| / |ref_nodes|`, and `RelSGScore = 0.5 * RelNodeRecall + 0.5 * RelTripletRecall`. Edge case when ref_trips is empty handled correctly (falls back to node_recall only). By construction RelSGScore(20) = 1.0.

3. **EGTR triplet ranking is correct.** `egtr_infer.py:142-159` implements Method A correctly: `pred_rel * pred_connectivity` gating, then `pred_rel.max(-1)[0] * sub_ob_scores` for ranking, `argsort_desc` for top pairs, then `pred_rel[s_q, o_q].argmax()` for single best predicate per pair. Diagonal zeroed correctly.

4. **Stop-level computations are correct.** OracleStop (`07`): first level where `sg_score >= threshold`. BaselineStop (`08`): first level where `mean_object_confidence >= theta`. LLMStop (`09`): first level where `saturation_score >= threshold`. All default to 20 if never triggered. `consecutive_k_stop` in `aggregate_100.py` correctly checks `k` consecutive scores ending at the candidate level.

5. **Hybrid fusion is correct.** `aggregate_100.py:272-283` computes `conf_w * obj_conf + llm_w * sat_scores[lvl]` and applies `consecutive_k_stop` with `hybrid_threshold`. Matches the config definition.

6. **`consecutive_k_stop` semantics for k=2:** The function checks that levels `lvl` and `lvl-1` (going back by offset) both meet threshold. For `lvl = start_level (5)`, offset=1 checks `lvl-1=4 < start_level=5`, so `all_above=False`. This means k=2 can never trigger at the start level, which is correct (need two consecutive). The stop level returned is the SECOND level where threshold is met, not the first. This is documented behavior.

7. **Potential concern: `probs[:, :num_labels]` softmax scope.** Since logits shape is `(200, 150)`, softmax is computed over all 150 classes. There is no "no-object" class in the output. This means every query is forced to predict some object class with some confidence. However, this matches EGTR's architecture -- it uses a different mechanism (object confidence threshold) to filter out non-object queries. The `object_conf_threshold` filter handles this.

### Robustness

1. **No retry logic for Ollama calls.** If ollama returns a transient error, the score is 0.0 with an error reason. The `run_focused_100.py` and `12_llm_ablation.py` scripts have caching (skip if JSON exists), so a re-run picks up from where it left off. Acceptable for PoC.

2. **File I/O has no atomic writes.** JSON writes use `open(path, "w")` directly. A crash mid-write could corrupt a file. For PoC, acceptable.

3. **VLM `call_vlm` does not set `num_ctx`.** `vlm.py:97-112` does not include `num_ctx` in options (unlike `llm.py`). VLM prompts are short (just the text prompt + images), so 4096 default is likely sufficient. But worth documenting the asymmetry.

4. **`_parse_llm_json` fallback returns `saturation_score=0.0` on parse failure.** This biases failed calls toward "not saturated" (conservative). Correct for the stopping problem (fail-safe = keep going).

### Style/Convention

1. **Type hints are comprehensive.** All functions in `src/milestone1/` have proper type annotations. Good.
2. **English comments and docstrings throughout.** Matches repo conventions.
3. **Some code duplication between `15_relative_saturation.py` and `aggregate_100.py`:** Both implement `compute_relative_metrics`, `load_csv_dict`, `save_csv`, `pearson_r`, `median`, `load_egtr_prediction`. Could be refactored into `src/milestone1/`. Acceptable for PoC.
4. **`from __future__ import annotations`** used in newer scripts but not in `src/milestone1/` modules. Minor inconsistency.
5. **Private function imports:** `metrics.py` functions `_to_label_set` and `_to_triplet_set` are used outside the module (in `analysis.py`, `15_relative_saturation.py`, `aggregate_100.py`). The underscore prefix conventionally signals private; consider renaming to `to_label_set` / `to_triplet_set`.

### Top Fixes (Prioritized)

1. **[HIGH] Fix hardcoded `/10` in `15_relative_saturation.py` lines 516-517 and 846-847.** Use `len(image_ids)` instead. The generated `m1-relative-saturation.md` report will contain wrong counts when run on 100 images. (Note: the 10-image report in repo was generated when n=10, so it was correct at that time; but re-running on 100 would produce wrong markdown.)

2. **[MEDIUM] Refactor duplicated utilities** (`compute_relative_metrics`, `pearson_r`, `median`, `load_csv_dict`, `save_csv`) from scripts into `src/milestone1/` to reduce drift.

3. **[LOW] Fix `egtr-inference.md` logits shape comment** from `(200, 151)` to `(200, 150)`.

4. **[LOW] Consider adding `num_ctx` to VLM options** in `vlm.py` for consistency, even if default suffices.

5. **[LOW] Rename `_to_label_set` / `_to_triplet_set`** to public names since they're used outside `metrics.py`.

---

## Part B: Documentation vs Code Consistency Audit

### Claim 1: SGScore = 0.5*NodeRecall + 0.5*TripletRecall, label-only category-SET matching

| Source | Evidence |
|--------|----------|
| `MILESTONE1.md:74` | "SGScore(t) = 0.5 * NodeRecall(t) + 0.5 * TripletRecall(t) (label-only, category-set matching)" |
| `src/milestone1/metrics.py:79-86` | `sg_score(node_recall, triplet_recall, node_weight=0.5, relation_weight=0.5)` returns `node_weight * node_recall + relation_weight * triplet_recall` |
| `src/milestone1/metrics.py:6-8,14-18` | `_to_label_set` returns `{obj["label"] for obj in objects}`; `_to_triplet_set` returns set of `(subject_label, predicate, object_label)` |
| `scripts/06_compute_sg_metrics.py:22-23` | Reads `node_w = cfg["metrics"]["sg_score"]["node_weight"]`, `rel_w = cfg["metrics"]["sg_score"]["relation_weight"]` from config |
| `configs/config.yaml:59-61` | `node_weight: 0.5`, `relation_weight: 0.5` |

**Verdict: MATCH.** Code exactly implements the documented formula with set-based label-only matching.

---

### Claim 2: RelSGScore = 0.5*RelNodeRecall + 0.5*RelTripletRecall vs Level-20 pseudo-GT

| Source | Evidence |
|--------|----------|
| `.claude/docs/research/m1-relative-saturation.md:9-11` | `RelNodeRecall(t) = \|nodes(t) & nodes(20)\| / \|nodes(20)\|`, `RelSGScore(t) = 0.5 * RelNodeRecall + 0.5 * RelTripletRecall` |
| `.claude/docs/research/m1-scale100.md:11` | "RelOracle (RelSGScore>=0.8, Plan B)" |
| `scripts/15_relative_saturation.py:144-151` | `rel_sg_score = 0.5 * node_recall + 0.5 * trip_recall` (when both valid), reference from `load_egtr_prediction(pred_root, img_id, max_level)` where `max_level=20` |
| `scripts/aggregate_100.py:143-144` | Same formula: `rel_sg_score = 0.5 * node_recall + 0.5 * trip_recall` |

**Verdict: MATCH.** Both scripts and docs agree on the formula and the Level-20 reference.

---

### Claim 3: ollama num_ctx=12288 fix present in BOTH llm.py AND llm_engine.py

| Source | Evidence |
|--------|----------|
| `src/milestone1/llm.py:85-86` | `"num_ctx": 12288` in options dict of `call_ollama()` |
| `src/milestone1/llm_engine.py:194-196` | `"num_ctx": 12288` in options dict of `_do_ollama_call()` |
| `.claude/docs/research/m1-scale100.md:21-23` | "Fixed with `num_ctx=12288`" and describes the root cause (ollama default 4096 truncating to 4095 tokens) |
| `MILESTONE1.md` | Does NOT mention `num_ctx` fix explicitly, but the calibration notes at lines 91-95 discuss the endpoint and `think: false`. The num_ctx issue was discovered after MILESTONE1.md was written. |

**Verdict: MATCH.** Both files have `num_ctx: 12288`. The `m1-scale100.md` correctly describes the fix. `MILESTONE1.md` does not mention it, but this is acceptable (MILESTONE1.md was written before the fix).

---

### Claim 4: Degradation = downscale by scale then bilinear upscale to ORIGINAL size; 20 scales; level_20 = scale 1.0

| Source | Evidence |
|--------|----------|
| `MILESTONE1.md:7` | "20 levels, far -> near" |
| `scripts/02_generate_resolution_levels.py:54-57` | `small = img.resize((small_w, small_h), Image.BILINEAR)` then `degraded = small.resize((orig_w, orig_h), Image.BILINEAR)` |
| `scripts/02_generate_resolution_levels.py:49-51` | `if scale >= 1.0: degraded = img.copy()` (original for scale=1.0) |
| `configs/config.yaml:28-50` | `num_levels: 20`, 20 scales listed from 0.05 to 1.00, `interpolation: bilinear` |

**Verdict: MATCH.** Code downscales then upscales to original size with bilinear interpolation. 20 scales, level_20 = scale 1.0 (original). The config `interpolation: bilinear` field is declared but not actually read by the script (it hardcodes `Image.BILINEAR`), which is a minor coupling gap but not a mismatch.

---

### Claim 5: EGTR object extraction (softmax over [:150]) and triplet ranking "Method A: single predicate per pair"

| Source | Evidence |
|--------|----------|
| `.claude/docs/research/egtr-inference.md:106-108` | "Object scores via softmax over classes (NOT sigmoid)", `probs[:, :num_labels]`, "Method A: single preds (with constraint) -- one predicate per pair" |
| `src/milestone1/egtr_infer.py:107-109` | `probs = logits.softmax(-1)`, `obj_scores, pred_classes = torch.max(probs[:, :self.num_labels], dim=-1)` |
| `src/milestone1/egtr_infer.py:141-143` | Comment "# Extract relationships (Method A: single pred per pair)" |
| `src/milestone1/egtr_infer.py:156` | `triplet_scores = pred_rel.max(-1)[0] * sub_ob_scores` |
| `src/milestone1/egtr_infer.py:172` | `best_pred = pred_rel[s_q, o_q].argmax().item()` (single pred per pair) |
| `egtr-inference.md:138-139` | `triplet_scores = torch.mul(pred_rel.max(-1)[0], sub_ob_scores)` matches code |

**Verdict: MATCH.** Code implements Method A exactly as documented. One minor doc issue: `egtr-inference.md:106` comment says shape `(200, 151)` but actual verified shape is `(200, 150)`. The code is correct regardless.

---

### Claim 6: Stop definitions

| Source | Evidence |
|--------|----------|
| `MILESTONE1.md:74-77` | "OracleStop = min level with SGScore >= threshold", "BaselineStop = min level with mean_object_confidence >= theta", "LLMStop = min level with saturation_score >= threshold" |
| `scripts/07_compute_oracle_stop.py:39-44` | Iterates levels, checks `float(lv["sg_score"]) >= t`, sets `stop_level = int(lv["level"])`, breaks |
| `scripts/08_compute_baseline_stop.py:37-44` | Iterates levels, checks `float(lv["mean_object_confidence"]) >= theta` |
| `scripts/09_llm_saturation_scoring.py:160-164` | Checks `result["saturation_score"] >= sat_threshold` |
| `configs/config.yaml:123-126` | `ablation.hybrid: conf_weight: 0.5, llm_weight: 0.5, threshold: 0.6` |
| `scripts/aggregate_100.py:278` | `fused[lvl] = conf_w * obj_conf + llm_w * sat_scores[lvl]` with threshold from config |
| VLM stop: `scripts/11_vlm_saturation_scoring.py:116-122` | Single: `if score >= sat_threshold: stop_k1 = level_idx`. Sequence: same logic with `consecutive_count` for k=2 |

**Verdict: MATCH.** All stop definitions match docs. Hybrid formula matches config. VLM single/sequence both use same threshold logic with k=1 and k=2 variants.

---

### Claim 7: Calibration table in MILESTONE1.md vs current config.yaml values

| Param | MILESTONE1.md calibrated | config.yaml actual | Match? |
|-------|--------------------------|-------------------|--------|
| `egtr.relation_conf_threshold` | 0.005 | 0.005 (line 54) | YES |
| `metrics.oracle_thresholds` | 0.3-0.6 | [0.3, 0.4, 0.5, 0.6] (line 65) | YES |
| `metrics.main_oracle_threshold` | 0.5 | 0.5 (line 66) | YES |
| `baseline.mean_object_conf_threshold` | 0.4 | 0.4 (line 71) | YES |
| `llm.saturation_threshold` | 0.7 | 0.7 (line 81) | YES |

**Staleness check:** `MILESTONE1.md:96` says "Current run is on **10 images** (`dataset.num_images`); raise to 100 after validation." But `config.yaml:21` now says `num_images: 100`. MILESTONE1.md is stale on this point -- it still says "10 images".

**Verdict: PARTIAL.** Calibration values MATCH. But `MILESTONE1.md` is stale regarding `num_images` (says 10, config says 100). The "First-run results" section (lines 98-103) correctly labels itself as "10 images" results, so the data there is historically accurate, but the "Current run is on 10 images" statement at line 96 is outdated.

---

### Claim 8: Run commands `uv run --project egtr python scripts/XX.py`, paths resolved against REPO_ROOT and egtr added to sys.path

| Source | Evidence |
|--------|----------|
| `MILESTONE1.md:40-41` | `cd /home/kanaoka/usr/github/milestone_1` then `uv run --project egtr python scripts/01_select_vg_subset.py` |
| All scripts `01-15, run_focused_100.py, aggregate_100.py` | `REPO_ROOT = Path(__file__).resolve().parents[1]` then `sys.path.insert(0, str(REPO_ROOT / "src"))` |
| `src/milestone1/config.py:9` | `REPO_ROOT = Path(__file__).resolve().parents[2]` (2 levels up from `src/milestone1/config.py` = repo root) |
| `src/milestone1/config.py:28-32` | `setup_egtr_path()` adds `REPO_ROOT / "egtr"` to `sys.path` |
| `scripts/03_run_egtr_inference.py:11-12` | Also manually does `sys.path.insert(0, str(REPO_ROOT / "egtr"))` |

**Verdict: MATCH.** Run commands match the documented pattern. `REPO_ROOT` resolution is correct. EGTR is added to `sys.path` either via `setup_egtr_path()` (called at module import of `egtr_infer.py:15`) or directly in scripts that need it.

---

### Claim 9: Headline numbers in m1-scale100.md match output CSVs

| Claim (m1-scale100.md) | CSV file | CSV value | Match? |
|------------------------|----------|-----------|--------|
| hybrid_9b_k2 MAE 4.18 | `method_comparison_relative_100.csv` row 1 | `mean_abs_error=4.18` | YES |
| hybrid_4b_k2 MAE 4.35 | row 2 | `mean_abs_error=4.35` | YES |
| llm_sgg_4b_h5_k2 MAE 4.47 | row 3 | `mean_abs_error=4.47` | YES |
| vlm_9b_single_k2 MAE 4.96 | row 6 | `mean_abs_error=4.96` | YES |
| hybrid_9b_k2 MeanStop 14.29 | row 1 | `mean_stop=14.29` | YES |
| mean_rel_oracle_stop 13.8 | all rows | `mean_rel_oracle_stop=13.79` | YES (rounded) |
| RelOracle saturates 80/100 | `m1-scale100.md:11` | Cannot directly verify from CSV (aggregate_100.py prints it but does not save a 100-image relative_oracle_stop.csv; the file `relative_oracle_stop.csv` has only 10 rows from the 10-image run). The mean_rel_oracle_stop=13.79 is consistent with 80/100 saturation (if 20 never saturate, mean would be higher). | PLAUSIBLE but NOT directly verifiable from saved CSVs |
| True-GT Oracle: 28/100 (28%) | `m1-scale100.md:10` | Not saved to a separate CSV for 100 images. `oracle_stop.csv` would need to be counted. | NOT directly verifiable without counting rows |

**Verdict: PARTIAL.** All MAE/MeanStop numbers in `method_comparison_relative_100.csv` match exactly. The "80/100" and "28/100" saturation counts are computed transiently in `aggregate_100.py` and printed to console but not saved to a CSV, so they cannot be independently verified from saved artifacts alone. They are consistent with the mean_rel_oracle_stop value.

---

### Claim 10: Config knobs claimed used (top_k_objects=20, top_k_relationships=30, history_length=5, start_level=5, vlm.max_image_side, sequence_history=3)

| Knob | Config value | Code that consumes it | Verified? |
|------|-------------|----------------------|-----------|
| `egtr.top_k_objects=20` | `config.yaml:56` | `egtr_infer.py:43`, `03_run_egtr_inference.py:39` | YES |
| `egtr.top_k_relationships=30` | `config.yaml:57` | `egtr_infer.py:44`, `03_run_egtr_inference.py:40` | YES |
| `llm.history_length=5` | `config.yaml:80` | `09_llm_saturation_scoring.py:85`, `llm_engine.py:70` | YES |
| `llm.start_level=5` | `config.yaml:79` | `09_llm_saturation_scoring.py:84`, `run_focused_100.py:233`, `aggregate_100.py:159` | YES |
| `vlm.max_image_side=512` | `config.yaml:104` | `vlm.py:52` default=512, `11_vlm_saturation_scoring.py:165`, `run_focused_100.py:237` | YES |
| `vlm.sequence_history=3` | `config.yaml:100` | `11_vlm_saturation_scoring.py:163`, used in sequence mode `hist_start = max(1, level_idx - sequence_history + 1)` | YES |
| `llm.input_object_conf_threshold=0.3` | `config.yaml:84` | `llm_engine.py:74`, `09_llm_saturation_scoring.py:89` | YES |
| `llm.input_relation_conf_threshold=0.005` | `config.yaml:85` | `llm_engine.py:75`, `09_llm_saturation_scoring.py:90` | YES |
| `llm.input_top_k_objects=20` | `config.yaml:86` | `llm_engine.py:76`, `09_llm_saturation_scoring.py:91` | YES |
| `llm.input_top_k_relationships=30` | `config.yaml:87` | `llm_engine.py:77`, `09_llm_saturation_scoring.py:92` | YES |

**Verdict: MATCH.** All config knobs are consumed by the corresponding code paths.

---

## Staleness Flags

| Document | Stale item | Severity |
|----------|-----------|----------|
| `MILESTONE1.md:96` | Says "Current run is on **10 images**" but config now has `num_images: 100` | MEDIUM -- misleading for new readers |
| `MILESTONE1.md:98-103` | "First-run results (10 images)" section -- historically correct, labeled as 10-image results. OK as historical record. | LOW -- acceptable if labeled |
| `.claude/docs/research/m1-relative-saturation.md` | Entire report was generated from the 10-image run (tables have 10 rows, "/10" denominators). Superseded by `m1-scale100.md` for 100-image results. | LOW -- the 100-image relative comparison lives in `m1-scale100.md` instead |
| `15_relative_saturation.py` lines 516-517, 846-847 | Hardcoded `/10` in print and markdown generation -- would produce wrong output if re-run with 100 images | MEDIUM -- code bug |
| `relative_oracle_stop.csv` | Contains only 10 rows (10-image run), not regenerated for 100 images. `aggregate_100.py` computes 100-image relative oracle internally but does not overwrite this file. | LOW -- the 100-image oracle data is in `method_comparison_relative_100.csv` |

---

## Summary: Consistency Table

| # | Claim | Verdict | Key Evidence |
|---|-------|---------|--------------|
| 1 | SGScore formula + set matching | **MATCH** | metrics.py:79-86, config.yaml:59-61 |
| 2 | RelSGScore formula + L20 pseudo-GT | **MATCH** | 15_relative_saturation.py:144-151, aggregate_100.py:143-144 |
| 3 | num_ctx=12288 in both LLM files | **MATCH** | llm.py:86, llm_engine.py:196 |
| 4 | Degradation procedure + 20 scales | **MATCH** | 02_generate_resolution_levels.py:54-57, config.yaml:28-50 |
| 5 | EGTR extraction + Method A ranking | **MATCH** | egtr_infer.py:107-109,141-188, egtr-inference.md |
| 6 | Stop definitions (all methods) | **MATCH** | scripts/07,08,09; aggregate_100.py:272-283; config.yaml:123-126 |
| 7 | Calibration table values | **PARTIAL** | Values match; MILESTONE1.md stale on num_images (says 10, config=100) |
| 8 | Run commands + path resolution | **MATCH** | All scripts use REPO_ROOT pattern; MILESTONE1.md documents correctly |
| 9 | Headline numbers vs CSVs | **PARTIAL** | MAE/MeanStop numbers match CSV exactly; saturation counts (80/100, 28/100) not saved to CSV |
| 10 | Config knobs consumed by code | **MATCH** | All 10 checked knobs traced to consuming code |

**Totals: 8 MATCH, 2 PARTIAL, 0 MISMATCH**

---

## Top Issues to Fix (Ranked by Severity)

1. **[HIGH] `15_relative_saturation.py` hardcoded `/10`** (lines 516-517, 846-847) -- will produce wrong markdown report if re-run on 100 images. Should use `len(image_ids)`.

2. **[MEDIUM] `MILESTONE1.md` stale on num_images** -- line 96 says "Current run is on **10 images**" but config is now 100. Should be updated to reflect current state, or restructured to clearly separate the 10-image historical section from the current 100-image state.

3. **[MEDIUM] `relative_oracle_stop.csv` not regenerated for 100 images** -- `aggregate_100.py` computes relative oracle stops for 100 images but only saves them implicitly via `method_comparison_relative_100.csv`. Consider saving a `relative_oracle_stop_100.csv` for auditability.

4. **[LOW] Code duplication across `15_relative_saturation.py`, `aggregate_100.py`, `13_aggregate_evaluation.py`** -- `compute_relative_metrics`, `pearson_r`, `median`, `load_csv_dict`, `save_csv` are copy-pasted. Refactor into `src/milestone1/`.

5. **[LOW] `egtr-inference.md` logits shape comment** says `(200, 151)` at line 106, actual shape is `(200, 150)`. Fix the doc comment.

6. **[LOW] `_to_label_set` / `_to_triplet_set` underscore naming** -- used widely outside `metrics.py`, should be public.

7. **[LOW] `vlm.py` does not set `num_ctx`** in ollama options, unlike `llm.py` and `llm_engine.py`. Probably fine for VLM prompts but worth documenting.

8. **[INFO] `03_run_egtr_inference.py` potential stale `result` variable** in final print -- cosmetic only.
