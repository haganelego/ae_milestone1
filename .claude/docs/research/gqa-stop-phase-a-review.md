# gqa_stop Phase A — independent code review (2026-10-08)

Scope: `src/milestone1/gqa_stop/*` on `feature/visapp-gqa` against
`.claude/docs/VISAPP2027_GQA_EXPERIMENT_PLAN.md` (§16 overrides §0–15).
Read-only review; no model inference. Checks run:

```bash
uv run pytest tests/test_gqa_stop_*.py -q          # 133 passed
uv run python -m milestone1.gqa_stop.cli preflight --config configs/gqa_stop.json --dry-run
uv run python -m milestone1.gqa_stop.cli prepare   --config configs/gqa_stop.json --dry-run
# prepare dry-run: eligible 9148; dev 265 cand / 210 img / 1936 scanned;
# test 639 cand / 480 img / 3531 scanned (matches §16 A2-bis numbers)
```

Data fact verified: 66,722 of 132,062 GQA val_balanced question ids have a
leading zero (e.g. `05515938`). Image ids are all-digit without leading zeros.

## Confirmed defects

1. **MEDIUM — question ids lose leading zeros when report.py reads CSVs back.**
   `report.py:64` uses `pd.read_csv(path)` with no dtype, so `stops.csv` and
   `level_table_test.csv` load `question_id` as int64. I checked this: the id
   `05515938` reads back as `5515938`. Consequences for about half of the questions:
   - `report.py:182` orders the §13 examples by `seeded_hash(seed, "5515938")`.
     That is not the pre-registered SHA256(seed|qid) order, so the example
     selection is not the specified one.
   - `report.py:201` looks for `plans/5515938.json`, which does not exist.
     The checklist in the example block is then shown as `[]`.
   - `report.py:205` prints a qhash computed from the wrong string.
   Fix: in `_load`, call `pd.read_csv(path, dtype={"question_id": str, "image_id": str, "group_id": str})`.
   Also add a test with a leading-zero id.

2. **MEDIUM — a reader-prompt revision after dev screening is not caught.**
   `check_infer_preconditions` (`stages_model.py:178-189`) compares only the
   screening digest. `screening_info` (`stages_data.py:448-469`) does not store
   the `answer_only` prompt hash. `record_prompts` (`context.py:415-442`) allows
   one revision any time before freeze. The plan puts that revision after the
   pilot (§11 Phase B), and the pilot runs after `screen --split dev`.
   Scenario:
   1. Run `screen --split dev`, then `pilot`.
   2. Edit `answer_only.txt`, which counts as revision 1.
   3. Run `infer --split dev`. It proceeds.
   Result: the adopted set was screened with the old reader, but the dev L20
   reader is a new call because the prompt hash changed the cache key. This
   breaks §3.3 ("same prompt as the common reader", L20 = 1 by construction).
   If dev L20 accuracy then falls below 0.99, `select_threshold` returns
   `theta: None` for that target. `build_eval_data` (`evaluate.py:119`) then
   calls `theta_from_json(None)`, which does `float(None)` and raises TypeError
   during `evaluate`.
   Fix: store `prompt_hashes["answer_only"]` in the screening state. Require it
   to match in `check_infer_preconditions` and in `pilot`, and force a
   re-screen if it differs. Also make `build_eval_data` skip or flag targets
   with `feasible == False`.

3. **MEDIUM — the model digest is never re-verified, and nothing ties test runs to the frozen digest.**
   The digest is read once from `/api/tags` at `preflight` and written to
   effective_config (`stages_model.py:45-55`). The cache key and record use
   this stored value (`client.py:265-277`); the running server is never
   re-checked. Two scenarios:
   - (a) `ollama pull qwen3.5:9b` updates the tag mid-run. All later requests
     come from the new weights but are labelled with the old digest. The cache
     treats them as consistent.
   - (b) `preflight` is re-run after `freeze` and writes the new digest. Then
     `screen --split test` and `infer --split test` run on the new model.
     `check_infer_preconditions` passes because the test screening digest
     equals the new active digest. Neither `screen` (`stages_data.py:390-396`)
     nor `run_evaluate` (`stages_model.py:265-288`) compares against
     `frozen_policy.json["digest"]`. This violates §2 ("do not change the model
     mid-evaluation").
   Fix:
   - In `Context.engine()`, GET `/api/tags` and abort if the live digest
     differs from the configured one.
   - In `screen --split test`, `infer --split test` and `evaluate`, require
     the active digest and prompt hashes to equal those in `frozen_policy.json`.
   - Refuse a digest change in `preflight` once `freeze` exists.

4. **LOW — some transport errors crash the command instead of being retried or recorded.**
   `client.py:193` catches only `ConnectionError` and `Timeout`. Two cases escape:
   - `requests.exceptions.ChunkedEncodingError` and `ContentDecodingError`
     propagate and abort the whole command. This can happen if ollama dies
     mid-response.
   - At `client.py:198`, `response.json()` raises `JSONDecodeError` on a
     malformed 200 body.
   The cache makes the run resumable, but the pause or failure accounting
   (§9.2) is bypassed.
   Fix: catch `requests.RequestException` and `ValueError` around the post and
   the `.json()` call, and treat them as retryable transport errors.

5. **LOW — the cache key omits the schema-repair prompt hash.**
   `client.py:272` uses only `prompts.hashes[role]`. If `schema_repair.txt`
   is edited, old records that were produced through a repair attempt stay
   valid on resume. The edit does count as a prompt revision.
   Fix: add `prompt_hashes[REPAIR_PROMPT]` to the key, or only for roles where
   repair is enabled.

6. **LOW — `prepare --revise` has two edge-case problems.**
   - `stages_data.py:218-221`: under `--dry-run` it returns before applying
     the overrides. The dry-run therefore prints the unrevised scan.
   - `stages_data.py:222-230`: in a real run the overrides are written to
     effective_config before `build_scan`. If the scan then raises, state
     `revisions` is never incremented, so a second revision becomes possible.
   Fix: build the merged cfg in memory, scan, then persist the overrides and
   the state together.

7. **LOW (performance only) — image generation does one level per call and rewrites the manifest each time.**
   `ImageSeries.bytes_for` triggers `_generate` (`images.py:113-139`) for a
   single level per call. Each call re-decodes the JPEG and rewrites plus
   fsyncs the entire `image_manifest.jsonl`. Over roughly 5k level rows this
   is O(n²): about 5k full rewrites averaging around 1 MB each. It does not
   crash.
   Fix: in `infer` and `pilot`, call `series.ensure(image_id, levels)` once per
   image, or flush once per command.

## Spec deviations not covered by the agent's own report

- **§7.2 Forced20Rate.** The spec defines it as "threshold not met at t<20, so
  Level 20 was reached". That includes a natural pass exactly at t=20, with the
  reason recorded separately. `online_stop` (`policy.py:56-59`) sets
  `forced=False` when the threshold is met at t=20. `evaluate.metrics_for`
  (`evaluate.py:134-145`) reports only that narrower rate as `forced20_rate`,
  and `method_summary.csv` has no reached-L20 column. The dev grid does have
  `reached_last_rate`, which matches the spec definition. The test main table
  therefore under-reports Forced20Rate. Fix: report the `stop == 20` rate as
  `forced20_rate` and keep the current value as `forced_never_met_rate`.
- **Pilot gates are not enforced in code.** Nothing blocks `infer --split dev`
  if the format-valid rate is below 0.90 after a revision (§11), or if both
  models fail the A4 calibration gate. These are only recorded. This is
  acceptable if the orchestrator enforces them, but it should be stated.
- **Probe and main run share one reader cache.** The probe uses namespace `""`,
  so probe records at L1/5/10/20 are reused by screening, pilot and infer for
  overlapping questions. The requests are byte-identical, so results are
  equivalent. Still, the provenance of those records is "probe". Mention this
  in the worklog or report.
- **Fallback pilot overwrites the primary model's records.** The record path
  has no model component, so the fallback pilot overwrites the per-request
  JSON records of the primary model at the same paths, including the 9b
  screening L20 records for the first 20 dev questions. The raw responses
  survive only in `requests.jsonl` and `pilot_levels_<tag>.csv`. §9.1 asks for
  every inference record. Fix: put the model tag in `record_path`.
- **Config keys are ignored.** `policy.start_level`, `consecutive_k` and
  `force_stop_level` are never read; the code hard-codes 1, 1 and 20.
  Behaviour is correct for v1.1, but editing those keys has no effect.

## Verified OK

1. **GT leakage: OK.** `question_text_input`, `checklist_input` and
   `planner_input` copy only the question text plus planner
   id/question/depends_on. Messages contain no image name, level, scale,
   answer, fullAnswer, semantic or imageId. Repair messages contain only the
   model's own raw reply and validator error strings, which hold no GT. The
   `input_data` stored in records is never sent.
2. **Ollama API: OK, apart from defects 3 and 4.** The payload is
   `model`, `messages[{role, content, images:[b64]}]`, `think:false`,
   `format:"json"`, `stream:false`, and `options{temperature, seed, num_ctx, num_predict}`.
   Responses are read from `message.content`. `json.loads` is applied to the
   whole content, with no regex. Retries happen only on connection errors,
   timeouts, 429 and 5xx, with backoff and a timeout. The digest comes from
   `/api/tags` at preflight.
3. **Image pipeline: OK.** The order is EXIF transpose, then RGB, then Lanczos
   only when the long side exceeds 512. The scale is
   `s_t = 0.05·20^((t-1)/19)`, with t=20 fixed at exactly 1.0. Sizes use
   floor with a minimum of 1, then bilinear down and bilinear up to (W,H).
   L20 is `base.copy()`. All levels are PNG. The sent bytes are the stored
   file bytes, and their sha256 is checked against the manifest before
   sending.
4. **Policy: OK.** `online_stop` calls `score_at(t)` lazily in increasing t,
   and offline replay goes through the same function. The test is `>=`.
   A threshold of +inf gives L20. k=1. All-zero scores give L20 (verified by
   hand).
5. **Effective confidence: OK.** `resolved[j]` already includes the
   ancestors, so zeroing is transitive. bool, str, NaN, inf and out-of-range
   values are rejected. The checklist must contain each id exactly once and
   in order; missing, duplicate, extra and out-of-order cases are detected.
   min and mean use the same `r_values`.
6. **Freeze lock: OK, apart from defects 2 and 3.** `guard_selection` rejects
   any non-dev split, non-dev rows and re-freezing. Test screen and infer
   require the freeze. Dev screen after freeze is refused. Prompt changes
   after freeze raise. `prepare --revise` is limited to one use and only
   before screening.
7. **Sampling: OK.** The orders are `SHA256(f"{seed}|{image_id}")`, keyed by
   group id, which is the min image id of a byte-duplicate group, and
   `SHA256(f"{seed}|{qid}")`. Dev is scanned first and test continues from
   dev's `end_index`, skipping dev groups. Reserve batches exclude all
   candidate groups. There are at most 2 questions per image, with a
   different attribute preferred. The quota is
   `min(share, size*100//300)` for dev and `min(share, size-dev)` for test,
   in integer arithmetic. The procedure is deterministic, and the dry-run
   reproduces the A2-bis numbers.
8. **Evaluation: OK.** Normalisation is NFKC, then lower, then strip, then
   whitespace collapse, followed by an exact match. The bootstrap resamples
   unique groups with replacement and keeps all of each group's questions.
   Metrics are question-weighted per replicate. Thresholds and stops stay
   frozen. The upper hull uses a monotone chain with `cross >= 0` pop and is
   rebuilt per replicate. A hand example gives hull
   `[(1,.5),(4,.9),(5,.91)]` and F(2.5)=0.7.
9. **Cache and resume: OK, apart from defects 3 and 5.** The key covers role,
   namespace, model name, digest, role prompt hash, input JSON hash, image
   sha256 and decoding. Writes go through a temp file, fsync and
   `os.replace`. Valid and schema_failure records are reused, and
   transport_failure records are retried.
10. **Scale: OK, apart from defect 7.** The questions file is 114 MB and
    loads fine. Duplicate hashing covers about 5.5k JPEGs once and is cached.
    The bootstrap is 2000 replicates over about 16 labels with 20-point
    hulls, which is cheap. I found no O(n²) loop over the 9k pool apart from
    the manifest rewrite.
