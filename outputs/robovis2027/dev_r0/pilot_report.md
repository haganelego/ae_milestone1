# ROBOVIS2027 dev_r0 pilot report

Run: `dev_r0` (dev20 × 20 levels = 400 states), produced 2026-10-07 by
`scripts/25_pgsg_infer.py` + `scripts/28_robovis_stage_metrics.py`.

## Path P not run

Path P (PGSG itself emitting a CHECKS section next to its scene graph) was **not run**.
The user decided this on 2026-10-07, after an architecture review. The 25-state pilot
was replaced by this written rationale (see `.claude/docs/research/robovis2027-worklog.md`
and `robovis2027-feasibility.md`).

- **Architecture.** PGSG is a BLIP model: a ViT-B/16 image encoder plus a BERT-base
  "MED" text decoder (`XBertLMHeadDecoder`). It starts from BLIP captioning weights and
  is fine-tuned only on VG-150 triplet sequences of the form `sub [OBJ] pred [REL] obj [OBJ]`.
  It is not instruction-tuned. The decoder has no channel for free-form instructions.
- **Prompt.** Generation always starts from the fixed prompt `"The visual scene of: "`,
  which is 6 tokens. The class default text budget is `max_txt_len=40`. A multi-sentence
  CHECKS instruction cannot be given as conditioning, and the model was never trained to
  follow one. The eval config raises `max_txt_len` to 512, but that value only bounds
  training targets. It does not create an instruction input.
- **No training signal.** The training data contains no CHECKS section. PGSG's grounding
  and category conversion read per-token hidden states and scores inside `generate`.
  A CHECKS section therefore cannot be separated without changing the decode loop and
  retraining.
- **Replacement.** The main paths are F1 and F2. In F1, PGSG stays fixed and the local
  VLM `qwen3.5:4b` sees the image plus the graph and produces the checks. In F2, a text
  LLM sees the graph only.

## Resolved PGSG generation settings

These come from `configs/pgsg/rel_det_vg_eval_local.yaml` merged over
`lavis/configs/models/blip_vrd_base_vg.yaml` and resolved at runtime. The same block is
stored in `manifest.json` under `generation_resolved`.

| Setting | Value | Note |
|---|---|---|
| decoding | nucleus sampling, `num_beams` 1 | official `generation_mode: sampling` |
| `top_p` | 0.9 | `generate()` default; the task does not override it |
| HF `top_k` | 50 | transformers default; applied together with `top_p` |
| `repetition_penalty` | **1.1 effective** | the task passes 1.5, but `med.py` `generate_from_encoder` hard-codes 1.1 in the sampling branch |
| `max_length` | 1024 | only sets the segment count |
| `min_length` | 768 | passed by the task but unused by PGSG `generate()` |
| segments | 36 | `1024 // int((seg_len 32 + 24) / 2)`; each restarts from the prompt |
| per-segment new tokens | min 24, max 32 | `seg_len: 32` in the eval yaml |
| prompt | `"The visual scene of: "` | 6 tokens incl. `[DEC]` BOS |
| entity labels / predicates per triplet | top-1 / top-3 | `top_k_ent_label_num 1`, `top_k_predicate_label_num 3` |
| score amplification | entity 5.0, relation 2.0 | `post_proc_cfg` |
| entity NMS IoU / max relations | 0.92 / 512 | hard-coded in `_postprocess` |

The graph JSON `generation` block stores the effective values:
`{max_length 1024, min_length 768, top_p 0.9, repetition_penalty 1.1, num_segments 36}`.

## Inference protocol

- **Input.** The inputs are `degraded_images/<id>/level_XX.jpg`, all at the 384-base size.
  PGSG's processor resizes each one to 384×384. `orig_size` is the base size, so boxes
  come out in base pixels. `box_xyxy_orig` equals the base box times
  (`coord_scale_x`, `coord_scale_y`) from `levels_manifest.json`.
- **Batching and seeds.** A batch holds 4 consecutive split images at one level. The
  seed is `state_seed(batch_ids, level, 42)`, a sha256 of the ids and level taken
  mod 2^31. Outputs are bit-reproducible and do not depend on `--levels`. Each state's
  seed and batch are recorded in `manifest.json`.
- **Provenance.** It is captured by wrapping three callables per call:
  `generate_from_encoder`, `seq2instance`, and the module `box_ops`. Every
  `raw_tokens`, `segment_ids`, `n_mentions`, `segment_id`, and `n_bad_inst` field is
  filled, and 0 of 400 states failed the consistency checks. `raw_tokens` is PGSG's own
  decoded span. It sometimes carries junk such as `". board"`. In those cases PGSG
  sampled the category from the 150-way distribution, so `label` can differ from
  `raw_tokens`.
- **Timing.** The run took 1166.8 s for 380 new states, about 3.07 s per state on an
  RTX 5080. GPU memory was about 7.2 GB according to nvidia-smi, with a peak allocation
  of about 6 GiB.

## Evaluation validation

- **Same predictions, official functions.** `--validate-official` fed the saved graphs
  for all 400 states to Pix2Grp's `evaluate_relation_of_one_image`, with `SGRecall` and
  `SGMeanRecall`. Per-image R@20/50/100 matched our implementation exactly
  (max |diff| = 0). Aggregate mR@K matched to within 1e-17. See `official_validation.json`.
- **End to end with the stock runner.** `scripts/24_pgsg_eval.sh` ran on the same 20
  level-20 inputs and the same batch composition, with seeds 42, 1, 2 and 3. The official
  range was R@50 17.0–21.8, R@100 19.4–25.9 and mR@50 4.1–7.5. Ours was R@50 19.6,
  R@100 24.9 and mR@50 6.7, all inside that range. The two differ only in the sampling
  seed and the GT frame (base vs original pixels). See `official_eval/summary.json`.

## dev20 SGDet by level (graph constraint, GT coordinates, n = 20 images)

| t | scale | R@20 | R@50 | R@100 | mR@50 |
|---|---|---|---|---|---|
| 1 | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | 0.094 | 0.71 | 0.71 | 0.71 | 0.04 |
| 10 | 0.2066 | 13.43 | 13.43 | 13.43 | 3.15 |
| 15 | 0.4545 | 16.87 | 20.08 | 20.08 | 6.41 |
| 20 | 1.0 | 17.20 | 19.59 | 24.88 | 6.72 |

mR here follows the official convention, dividing by all 50 predicates. All levels are
in `stage_summary.json`. Per-image rows are in `stage_metrics.csv`. Per-predicate hits
are in `stage_predicate_hits.jsonl`.
