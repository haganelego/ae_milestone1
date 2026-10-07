# Reproducing the ROBOVIS2027 recovery run (rec_r0)

Updated 2026-10-08 ≈03:05 JST (A1 drafts, saved by the orchestrator). All Phase A–D steps below have run.

All commands run from the repo root `/home/kanaoka/usr/github/milestone_1`, on branch `feature/robovis2027`. Hardware is an RTX 5080 16 GB with local ollama; no external API is used.

**Environments.**
- `egtr` uv project: analysis and ollama clients.
- `envs/pgsg`: PGSG, run with `PYTHONPATH=Pix2Grp_CVPR2024 PYTHONDONTWRITEBYTECODE=1`.

**Outputs.**
- Recovery artefacts: `outputs/robovis2027_recovery/rec_r0/`.
- PGSG caches follow the run-directory convention `outputs/robovis2027/<run_id>/`.
- Earlier results are never overwritten.

**Commits.** `git log --oneline main..HEAD`, oldest first:

| commit | contents |
|---|---|
| e820065 | PGSG submodule (Pix2Grp `cff7420`), `envs/pgsg`, VG-150 converter (scripts/23), official eval wrapper (scripts/24) |
| 8f4830d | GT-only image selection (`data/processed/robovis2027/selected_images.json`: test100 / dev20, seed 42); 384-base 20-level degradation (scripts/27) |
| 1dd1ba2 | PGSG inference wrapper (scripts/25, `src/milestone1/pgsg_adapter.py`), per-image SGDet (scripts/28, `robovis_sgdet.py`) |
| 5534d05 | cleaned graph G (scripts/35, `graph_clean.py`), GT-RelSGScore / precision metrics, `--seed-offset` in scripts/25, `--avg-seeds` in scripts/28 |
| c1d3c81 | multiseed launcher (scripts/37_robovis_multiseed.sh, not used in rec_r0), failure analysis |
| bb4e22e, 8b6e6b2, 6a71445 | judge v3, v4 grounded judge (scripts/29d), MIRU VLM+SGG scoring (scripts/40), MIRU eval registry (scripts/42), scaled-down MIRU write-up, config overlay (`src/milestone1/config.py`) |
| 079ee22 | recovery plan of record |
| 48b964b | recovery Phases A–C: preflight (43_recovery), test selection (44), multiseed metrics (45), Phase B diagnostics (43_robovis_recovery_diag, `recovery_diagnostics.py`), two-stage U policy (`two_stage_policy.py`), held-out overlay + pre-registration (`configs/config_heldout.yaml`), scripts/11 output-root / num_ctx options |
| (later commit) | scripts/46_recovery_test_rec.py, `configs/robovis2027_test_rec.yaml` (generated), scripts/47_recovery_phase_d_eval.py, scripts/29d `--degraded-root` |

## §1 Preflight (read-only, < 1 min)
```
uv run --project egtr python scripts/43_recovery_preflight.py --run-id rec_r0
```
This writes `frozen_config.json`, `model_hashes.json` and `data_manifest.json`. `recovery_preflight.md` was written from these files.

## Phase A: pre-registered held-out confirmation (MIRU pipeline, 96 images)
The overlay and pre-registration are in `configs/config_heldout.yaml`. Inputs were built zero-call with the MIRU scripts under the overlay:
```
export MILESTONE1_CONFIG_OVERLAY=configs/config_heldout.yaml
uv run --project egtr python scripts/02_generate_resolution_levels.py   # 1920 jpg, original size, ~3 s
uv run --project egtr python scripts/03_run_egtr_inference.py           # 1920 json, < 10 min
uv run --project egtr python scripts/04_convert_egtr_outputs.py         # schema check, 0 errors
uv run --project egtr python scripts/05_convert_vg_gt.py                # 96 GT graphs
uv run --project egtr python scripts/06_compute_sg_metrics.py
uv run --project egtr python scripts/07_compute_oracle_stop.py
uv run --project egtr python scripts/08_compute_baseline_stop.py
uv run --project egtr python scripts/eval_topk_oracle.py --pred-dir outputs/robovis2027_miru/egtr_predictions \
  --metrics-dir outputs/robovis2027_miru/metrics --out-dir outputs/robovis2027_miru/metrics/topk_oracle
```

**Scoring.** One detached runner, `outputs/robovis2027_miru/logs/run_heldout_scoring.sh`, waits for the flag `outputs/robovis2027_miru/logs/GO`. The flag was created on 2026-10-08 at 00:59:55 (orchestrator), after A3 confirmed ollama was free. The runner then executes, under the overlay:
```
uv run --project egtr python scripts/40_vlm_sgg_saturation_scoring.py --plan qwen3.5:4b=text
uv run --project egtr python scripts/11_vlm_saturation_scoring.py   # overlay: vlm.models [9b], modes [single], num_ctx 16384
```
- Outputs (1536 files each = 96 × levels 5–20): `outputs/robovis2027_miru/llm_scores_vlm/qwen3.5-4b_h5_rel_on_conf_on_text/` and `outputs/robovis2027_miru/vlm_scores/qwen3.5-9b/single/`.
- **Text 4b.** 2026-10-08 01:00:14 to 01:57:47. 1536/1536, 0 failures, 3453 s (median 2.1 s per call). Generated once and never regenerated.
- **Image-only 9b, first attempt (01:57:47).** The start-up smoke test returned HTTP 500 (probable VRAM pressure: 4b and 9b runners resident while PGSG Phase D ran). scripts/11 aborted the model by design, so 0 score files were written (`rec_r0/failures.jsonl`).
- **Image-only 9b, rerun (02:35:51 to 03:00:02).** The same command, overlay and decoding:
  ```
  MILESTONE1_CONFIG_OVERLAY=configs/config_heldout.yaml uv run --project egtr python scripts/11_vlm_saturation_scoring.py
  ```
  1536/1536 (96 × 16), 0 errors, 0 parse errors, 1451 s (median 0.92 s per call). Log: `outputs/robovis2027_miru/logs/heldout_vlm9b_rerun.log`.

**Evaluation** (A2):
```
uv run --project egtr python scripts/42_miru_gt_eval.py --dataset heldout   # outputs/heldout_eval/
```

## Phase B: cached-only diagnostics (A2, zero calls, ≈1 s)
```
uv run --project egtr python scripts/43_robovis_recovery_diag.py --run-id rec_r0
```
This writes `fixed_mix.csv`, `budget_oracle.csv`, `two_choice_oracle.csv`, `shuffled_assignment.csv`, `audit.md` and `diagnostics_manifest.json`. Once `multiseed_gain.csv` exists it also writes `cross_seed_oracle.csv` and `seed_differences.csv`.

## Phase C.1: PGSG seeds 1 and 2, dev20 × levels {5, 10, 15, 20} (160 states)
Runner: `rec_r0/logs/run_c1_pgsg.sh`.
```
for N in 1 2; do
  PYTHONPATH=Pix2Grp_CVPR2024 PYTHONDONTWRITEBYTECODE=1 uv run --project envs/pgsg python scripts/25_pgsg_infer.py \
    --config configs/robovis2027.yaml --run-id dev_r0 --split dev --seed-offset $N --levels 5,10,15,20 --batch-size 4
done
uv run --project egtr python scripts/35_robovis_clean_graphs.py --runs dev_r0 --graphs-subdir graphs_seed1
uv run --project egtr python scripts/35_robovis_clean_graphs.py --runs dev_r0 --graphs-subdir graphs_seed2
uv run --project egtr python scripts/45_recovery_multiseed_metrics.py --run-id rec_r0
```
- **PGSG outputs.** `outputs/robovis2027/dev_r0/graphs_seed{1,2}/` hold 80 files each, plus `manifest_seed{1,2}.json`.
- **Seeds.** Base seeds are 43 and 44 (seed 0 = 42), with `state_seed(batch_ids, level, base)`.
- **Batching.** batch_image_ids are identical to seed 0, including order: at each level, 5 batches of 4 consecutive dev images in rank order.
- **Wall time.** Seed 1 took 250.8 s of inference (3.14 s/state, 259 s with start-up). Seed 2 took 260.7 s (3.26 s/state, 270 s). Problems and fallbacks: 0.
- **Clean graphs.** `graphs_seed{1,2}_clean/` hold 80 files each.
- **Metrics.** `rec_r0/stage_metrics_seed{0,1,2}.csv` have 80 rows each. The seed-0 re-score equals `dev_r0/stage_metrics.csv` (max |diff| 0 over 25 columns). `rec_r0/multiseed_gain.csv` has 120 rows (image_id, seed, metric, q5, q10, q15, q20, g, valid), plus `multiseed_manifest.json`.

## Phase C.2: two-stage U policy freeze (A3, zero new model calls)
The policy uses the cached v4 decisions `outputs/robovis2027/dev_r0/decisions/F1g_v4_G__9b/<id>/level_10.json` (20 states).
```
PYTHONPATH=src uv run --project egtr python -m milestone1.two_stage_policy --mode freeze --run-id dev_r0 --split dev --out-run rec_r0
PYTHONPATH=src uv run --project egtr python -m milestone1.two_stage_policy --mode evaluate --seeds 0,1,2
```
This writes `rec_r0/frozen_policy.json` (τ, p_tie, hash rule), `dev_two_stage.csv`, `dev_two_stage_eval.csv`. The orchestrator wrote `gate_decision.json` (GO) from these and `cross_seed_oracle.csv`.

## Phase D: independent confirmation (60 new images)
**Selection** (zero-call, done):
```
uv run --project egtr python scripts/44_recovery_test_selection.py --run-id rec_r0   # rec_r0/test_selection.json
```
The selection takes VG-150 test images with objects ≥ 5 and relations ≥ 3 (18,887 eligible). It removes 1,039 eligible ids already used (12 recorded sets), sorts the rest ascending, shuffles with `random.Random(20261008)` and keeps the first 60.

**Degradation** (done 2026-10-08 01:03; 1200 jpg, new ids only):
```
uv run --project egtr python scripts/46_recovery_test_rec.py prepare
```
- This runs scripts/27 `process_image` with `configs/robovis2027.yaml` settings and writes to `data/processed/robovis2027/degraded_images/<id>/`.
- It also writes `rec_r0/levels_manifest_test_rec.json` and `rec_r0/selected_images_test_rec.json`.
- It generates `configs/robovis2027_test_rec.yaml`, which differs from robovis2027.yaml only in `paths.selected_images` and `paths.levels_manifest`.

**PGSG chain** (`rec_r0/logs/run_phase_d_pgsg.sh`, `CFG=configs/robovis2027_test_rec.yaml`, run id `test_rec`, log `rec_r0/logs/phase_d_pgsg.log`). Everything below ran.

- **seedA, level 10 first** (base 42, batch size 4, 15 batches per level of consecutive images in rank order). 60 states in 264.7 s, 0 problems, 0 fallbacks. Clean graphs were produced with scripts/35. Marker `T10_CLEAN_READY` at 01:08:21.
  ```
  PYTHONPATH=Pix2Grp_CVPR2024 PYTHONDONTWRITEBYTECODE=1 uv run --project envs/pgsg python scripts/25_pgsg_infer.py \
    --config $CFG --run-id test_rec --split test --batch-size 4 --levels 10
  uv run --project egtr python scripts/35_robovis_clean_graphs.py --config $CFG --runs test_rec
  ```
- **seedA, all 20 levels.** The remaining 1140 states took 4607.5 s (about 17.5 s per batch, GPU shared with ollama), 0 problems, 0 fallbacks. Outputs: `graphs` and `graphs_clean` 1200/1200; `test_rec/stage_metrics.csv` 1200 rows, all valid; `rec_r0/stage_metrics_seedA.csv` 1200 rows. Marker `SEEDA_DONE` at 02:25:29.
  ```
  ... scripts/25_pgsg_infer.py --config $CFG --run-id test_rec --split test --batch-size 4
  uv run --project egtr python scripts/35_robovis_clean_graphs.py --config $CFG --runs test_rec
  uv run --project egtr python scripts/28_robovis_stage_metrics.py --config $CFG --run-id test_rec
  uv run --project egtr python scripts/46_recovery_test_rec.py metrics --graphs-subdir graphs --name seedA --seed-label A
  ```
- **seedB** (base 43, `--seed-offset 1`, levels 10/15/20). 180 states in 554.0 s, 0 problems, 0 fallbacks. batch_image_ids are identical to seedA. Outputs: `graphs_seed1` and `graphs_seed1_clean` 180/180; `rec_r0/stage_metrics_seedB.csv` 180 rows, all valid. Marker `SEEDB_DONE` at 02:34:52. No hull was built from seedB.
  ```
  ... scripts/25_pgsg_infer.py --config $CFG --run-id test_rec --split test --batch-size 4 --seed-offset 1 --levels 10,15,20
  uv run --project egtr python scripts/35_robovis_clean_graphs.py --config $CFG --runs test_rec --graphs-subdir graphs_seed1
  uv run --project egtr python scripts/46_recovery_test_rec.py metrics --graphs-subdir graphs_seed1 --name seedB --seed-label B --levels 10,15,20
  ```

**v4 at t10 and the frozen policy** (A3, ran 02:00:00 to 02:12:05). 60 states with qwen3.5:9b and num_ctx 16384: 58 valid and 2 failures (2336446, 2322283; HTTP 500 after 6 attempts each, during concurrent CUDA OOM). Under the frozen rule, failures continue to t20. Outputs: `outputs/robovis2027/test_rec/{checks,decisions}/F1g_v4_G__9b/` and `rec_r0/test_policy.csv` (02:12:05).
```
uv run --project egtr python scripts/29d_robovis_grounded_v4.py --run-id test_rec --split test --model qwen3.5:9b \
  --levels 10 --image-ids-file <60-id list from rec_r0/test_selection.json> --num-ctx 16384
PYTHONPATH=src uv run --project egtr python -m milestone1.two_stage_policy --mode apply --run-id test_rec \
  --image-ids-file <same list> --out-csv outputs/robovis2027_recovery/rec_r0/test_policy.csv
```

**Evaluation** (A2, ran at 02:35:11 after `SEEDB_DONE`; zero calls). Outputs: `test_summary.csv` (seedA), `seed_replication.csv` (seedB), `phase_d_summary.json`, `method_comparison_miru_style.{md,csv}` (scripts/48), `conf_*` comparator files (scripts/49). seedA and seedB are never combined.
```
uv run --project egtr python scripts/47_recovery_phase_d_eval.py --run-id rec_r0
uv run --project egtr python scripts/48_recovery_miru_style_table.py --run-id rec_r0
uv run --project egtr python scripts/49_recovery_conf_two_stage.py --run-id rec_r0
```
