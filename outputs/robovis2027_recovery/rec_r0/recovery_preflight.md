# ROBOVIS2027 recovery — preflight (plan §1), rec_r0
Generated 2026-10-08 00:5x JST by A1 (saved by the orchestrator). The checks were read-only. Machine-readable versions: `frozen_config.json`, `model_hashes.json`, `data_manifest.json` (scripts/43_recovery_preflight.py).

## Repository
- Branch `feature/robovis2027`, HEAD `079ee22`. The tree has 60 dirty paths, mostly untracked outputs; the list is in `frozen_config.json`.

## Phase A held-out status
- EGTR predictions: 1920/1920. Text-4b scores: 0/1536. Image-only 9b scores: 0/1536.
- Exactly one GO-waiting runner is active (PID 127475, `outputs/robovis2027_miru/logs/run_heldout_scoring.sh`). It waits for `outputs/robovis2027_miru/logs/GO`, which A1 creates after A3 confirms that ollama is free.
- Registered conditions are in `configs/config_heldout.yaml` (pre-registration comment and the `heldout_eval` block).

## h3 / v4 (A3)
| variant | checks | decisions | images | levels | failure flags | last write |
|---|---|---|---|---|---|---|
| F1g_v4_G__9b | 400 | 400 | 20 | 20 | 1 | 00:44 |
| F1v_v3_G__9b_h3 | 100 | 100 | 20 | 5 | 2 | 00:23 |

No A3 process is running and ollama has no model loaded. (A3 confirmed completion at ~00:50.)

## PGSG
- Checkpoint: `Pix2Grp_CVPR2024/PGSG-CVPR2024/vg_sgg.pth`, sha256 `d90c63ca6e0f7ea516c5f0ae23e2520085011fbc2c04ee1c56f23a2e42e299eb`. It equals the value in the dev_r0 manifest. Submodule commit: `cff7420`.
- Generation (effective values): nucleus sampling, top_p 0.9, HF top_k 50; repetition_penalty 1.1 (the task passes 1.5, but `med.py` hard-codes 1.1); 36 segments, seg_len 32, top-1 entity / top-3 predicate; amplification 5 / 2, entity NMS 0.92, at most 512 relations.
- Inputs: 384-base images, 20 geometric levels (0.05 → 1.0), bilinear, JPEG q95 (`levels_manifest.json`, sha in `data_manifest.json`).
- Seeds and batches: base seed 42, batch size 4. A batch is the next 4 split images in rank order at one level. The seed is `state_seed(batch_ids, level, base_seed)` (sha256 mod 2^31). The seed-0 batches are listed in `frozen_config.json`. There were no fallbacks.

## GT and data
- `test.json`, `rel.json` and `train.json` sha256 are in `data_manifest.json`. The local splits share 0 ids: train 57,723, val 5,000, test 26,446.
- Id sets and overlaps are in `data_manifest.json`.
  - test100 ∩ MIRU100 = 4. These are the 4 images excluded from held-out.
  - dev20 has no overlap with any other set (smoke_a3 is a subset of dev20).
  - **Disclosure (1):** 1 held-out96 image was in the original MIRU100 (`e1b8e56`), before the `2b038b2` re-selection. It is not in the current MIRU100.
  - **Disclosure (2):** 3 held-out96 images are among the first 1000 test images used only for aggregate PGSG validation, with no curves examined.
  - The registered held-out set is unchanged; both points are disclosed.

## Budget
- Everything runs locally (RTX 5080 16 GB, ollama), so there is no external API cost.
- robovis check and judge calls: 6,031 attempts so far, against `execution.max_model_calls` 9000.
- MIRU vlm_sgg calls: 6,407, with no cap configured.

## Phase C.1 / D prep status (at preflight time)
- C.1: seeds 1 and 2 for dev20 × {5,10,15,20} running (bs4, same batch rule as seed 0; verification of batch ids against seed 0 pending completion). Cache under `dev_r0/graphs_seed{1,2}/`, derived files under `rec_r0/`.
- D prep: `rec_r0/test_selection.json` — 60 images from the 17,848 eligible VG-test images after excluding 1,039 used ids (12 exclusion sets), id-sorted, shuffled with `random.Random(20261008)`. Training-split independence asserted via the VG-150 test split protocol (author h5 not available locally for byte-level verification).
