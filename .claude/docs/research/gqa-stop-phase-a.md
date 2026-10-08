# gqa_stop Phase A (model-free implementation + tests) — 2026-10-08

Spec: `.claude/docs/VISAPP2027_GQA_EXPERIMENT_PLAN.md` v1.1 (§16 overrides). No inference was run.
Only `GET /api/tags` was called (preflight). Nothing committed.

## Files created

- `src/milestone1/gqa_stop/` (package, 18 modules, ~4.6k lines, every file < 530 lines)
  - `common.py` hashing, atomic writes, config/effective-config, `state.json`, worklog
  - `dataset.py` §16 A1 filter + §3.1 exclusions, schema mapping, source-jpg sha256 duplicate groups
  - `sampling.py` §16 A2 stratified scan, quota allocation + shortfall reallocation, adoption
  - `images.py` §4 level series (EXIF→RGB, Lanczos ≤512, bilinear down/up, PNG, manifest)
  - `schemas.py` strict validators (reader/direct/planner/checklist)
  - `client.py` ollama `/api/chat`, cache key, 2 transport retries, 1 schema repair, `requests.jsonl`/`errors.jsonl`, pause monitor
  - `planner.py`, `inference.py` request builders (question text only), r_j / min / mean, runner (live or cache-only replay)
  - `policy.py` online prefix stop, grid, dev-only selection + freeze guard
  - `evaluate.py` exact match, frozen-policy application, concave hull + adjacent mix, image-cluster bootstrap, diagnostics
  - `calibration.py` §16 A4 gate; `preflight.py`; `report.py` (report.md, figures, completion.json)
  - `context.py`, `stages_data.py` (probe/prepare/screen), `stages_model.py` (preflight/pilot/infer/freeze/evaluate), `cli.py`
- `prompts/gqa_stop/{answer_only,direct,planner,checklist,schema_repair}.txt` (§8 fixed text; `Input data: <json>` appended by code)
- `tests/gqa_stop_helpers.py` (synthetic GQA world, fake ollama transport) and 10 test files `tests/test_gqa_stop_*.py` (158 tests)
- `configs/gqa_stop.json`: two keys added under `dataset.filter` (`exclude_negation`, `require_question_text_attribute_match`, both true). `candidate_dev_images_scan` and `candidate_test_images_scan` are set to `null` per A2-bis.
- Runtime output (new dir only): `outputs/gqa_stop/visapp_gqa_v1/{preflight.json,effective_config.json,state.json}`

## Test results

```
uv run ruff check src/milestone1/gqa_stop tests/test_gqa_stop_*.py   -> All checks passed!
uv run ruff format --check ...                                          -> 27 files already formatted
uv run pytest tests/test_gqa_stop_*.py -q                               -> 158 passed in 11.99s
```

Existing suite: `uv run pytest -q` stops at collection with 9 errors in pre-existing tests
(`ModuleNotFoundError: scipy` / `pycocotools`; root env lacks them). Ignoring those 9 files: 349 passed,
3 failed (`test_two_stage_policy.py`, same scipy import at runtime). The 9 files run in the egtr env:
145 passed; `test_two_stage_policy.py` in the egtr env: 12 passed. None are touched by this change.

Mandatory §12 coverage: 1 duplicates (dataset), 2 determinism (dataset), 3 GT leak (client), 4 image size /
L20 base / sent hash (images), 5 threshold boundary (policy), 6 dependency zeroing (scoring), 7 min/mean
cache replay (scoring), 8 schema errors (schemas), 9 planner/reader failure (scoring), 10 common reader +
L20 screening reuse (cli pipeline), 11 freeze guard (policy, cli), 12 cluster bootstrap (evaluate),
13 normalisation / hull / adjacent mix by hand (evaluate), 14 resume + invalidation (client, cli),
15 prefix policy (policy). §16 additions: quotas/shortfall (dataset), hull (evaluate), calibration
(calibration), probe command (cli).

## Preflight summary

- Branch `feature/visapp-gqa`, HEAD 4ee6d1f.
- `qwen3.5:9b` digest `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` (written to effective_config).
- `qwen3.5:4b` digest `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd` (fallback).
- Questions: 132,062; sha256 `2f675643c0cf…0779`. Referenced val images missing: 0. Free disk 411.9 GB.
- Eligible after all rules: 9,148 questions / 5,467 images (A1 filter 9,290, minus negation 112,
  question-text/attribute mismatch 27, answer word in question 3). No byte-duplicate images found.
- Cells: color r0 4,750 / r1 2,909; material r0 1,231 / r1 146; shape r0 76 / r1 36.
- `prepare --dry-run` under §16 A2-bis (no image cap; split-aware quotas; test continues from the dev cursor):

| cell | eligible | dev quota | dev filled | test quota | test filled |
|---|---|---|---|---|---|
| color r0 | 4,750 | 60 | 60 | 160 | 160 |
| color r1 | 2,909 | 60 | 60 | 160 | 160 |
| material r0 | 1,231 | 60 | 60 | 160 | 160 |
| material r1 | 146 | 48 | 48 | 98 | 89 |
| shape r0 | 76 | 25 | 25 | 51 | 46 |
| shape r1 | 36 | 12 | 12 | 24 | 24 |
| total | 9,148 | 265 | 265 | 653 | 639 |

  Dev used 210 images and scanned 1,936. Test used 480 images and scanned 3,531, which exhausts the pool.
  The test shortfall comes from questions that lie inside the dev scan range but were not taken by dev,
  plus the 2-per-image limit. Before A2-bis, the 150/400 caps gave dev 203 and test 524 candidates,
  with material r1 at 12/14 and shape at 11/11.
- `probe --dry-run`: 198 questions (33 per cell), 792 reader requests at L1/5/10/20.

## Deviations / open issues

1. **Rare-cell starvation was resolved by orchestrator decision §16 A2-bis.** The image caps are now `null`
   and serve only as diagnostics. Quotas are split-aware: rare cells are split 1/3 to dev and the rest to test.
   The test scan continues from the dev cursor. Counts are in the table above.
2. **Negation and label-noise exclusions.** §3.1 rules are applied after the A1 filter, so the pool is
   9,148, not 9,290. Negated filters (`not(...)`) are excluded. Questions whose text does not match the
   queried attribute are excluded too ("What is she doing?" labelled color). Both can be switched off by config keys.
3. **Duplicate groups** are scan units; "≤2 questions per image" is applied per group (identical bytes = one image).
4. **Model switch.** The pilot fallback switch is a recorded override. `infer` refuses while the screening
   digest differs from the active model, so dev screening must be re-run after a switch. That can change the dev set.
5. **Fallback choice rule** when both models fail the gate: higher mean pooled ρ wins, then lower lowest-level mean confidence.
6. **Calibration** uses valid calls only (Direct confidence; checklist effective r_j). An undefined ρ (flat
   confidence) counts as failing.
7. Record files are `<role>/<model>@<digest12>/<qid>/<NN>.json`, zero-padded, and plans are `plans/<tag>/<qid>.json`. The repeat diagnostic is in `reader_repeat/`.

## Summary

- Full Phase A package, prompts, CLI and 127 model-free tests are in place; lint and format are clean.
- Preflight ran against the real environment; both model digests were recorded in `effective_config.json`.
- Dry-runs for probe and prepare work on real data; later stages refuse correctly until their prerequisites exist.
- §16 A2-bis is implemented. Dev fills every cell; test reaches 89/98 for material r1 and 46/51 for shape r0 after exhausting the pool.
- Next: `probe` (792 reader calls), then `prepare`, `screen --split dev`, and `pilot --calibration`.

## Review fixes (2026-10-08, `.claude/docs/research/gqa-stop-phase-a-review.md`)

All 12 items from the orchestrator's list are done. `tests/test_gqa_stop_review_fixes.py` adds 25 regression tests for them.

1. **Leading-zero ids.** `common.read_id_csv` reads `question_id`, `image_id` and `group_id` as str. `report.py` is the only CSV reader in the package and now uses it.
2. **Reader prompt drift.** The screening state stores `reader_prompt_hash` and all `prompt_hashes`.
   - `pilot`, `infer` and `freeze` refuse if the answer_only hash changed, and ask for a re-screen.
   - `infer`, `freeze` and `evaluate` also require the screening digest to equal the active digest.
   - `build_eval_data` skips targets whose theta is None and reports them as `infeasible_targets`. They appear in `method_summary.csv` with status `infeasible_on_dev`.
3. **Live digest.** `Context.engine()` calls `GET /api/tags` and refuses if the server is unreachable or its digest differs from the recorded one.
   - `screen --split test`, `infer --split test` and `evaluate` require the model, digest and all prompt hashes to equal `frozen_policy.json`.
   - `preflight` refuses a digest change after freeze.
4. **Transport errors.** `requests.RequestException` (including ChunkedEncodingError and ContentDecodingError), a non-JSON 200 body and a non-object 200 body are all retried. After the retries they become a transport_failure.
5. **Repair prompt in the cache key.** The `schema_repair` prompt hash is part of the cache key whenever repair is enabled.
6. **`prepare --revise`.** The revision is merged in memory and the scan runs first.
   - A dry-run prints the revised scan.
   - The effective config, manifests and revision count are written only after the scan succeeds.
7. **Images.** All 20 levels of an image are generated in one pass, and one manifest line is appended per image. On load, the last line for an image wins. The bytes are re-hashed on every send and regenerated once on a mismatch.
8. **Forced20 per §7.2.** `forced20_rate` is the share of questions that reached L20 without meeting the threshold earlier; a natural pass at L20 counts.
   - It comes with `never_met_rate` and `natural_pass20_rate`.
   - `stop20_rate` is added for every method. It equals `forced20_rate` by construction when start=1 and k=1.
   - `stops.csv` has a `stop_reason` column; the dev grid has the same rate columns.
9. **Pilot gates.** `infer --split dev` requires a pilot summary and calibration for the active model tag, made with the current prompts.
   - Format: a valid rate below 0.95 with no prompt revision is refused ("repair first"). Below 0.90 after a revision is refused as the §11 exit branch.
   - A4: a failed gate without a model switch is refused ("run the fallback pilot"). A failed gate after the switch is refused as both-fail.
   - `--force-continue` overrides a refusal, logs a GATE OVERRIDE, and records `gate_override` in state, completion.json and report.md.
10. **Model-tagged records.** Records live under `<role>/<model>@<digest12>/…`, and the model tag is in the cache key, so 9b and 4b records coexist.
11. **Policy parameters.** `policy.start_level`, `consecutive_k` and `force_stop_level` are read through `policy_params`. Anything other than 1 / 1 / images.levels raises. The values are written into `frozen_policy.json` and used by evaluate.
12. **Provenance.** Records carry the `stage` that created them. When a stage reuses records from another stage, `worklog.md` gets a provenance line, with an explicit note for probe records. Shared caching is kept.
