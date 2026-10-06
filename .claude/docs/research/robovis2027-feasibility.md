# ROBOVIS2027 plan v1.0 — feasibility assessment against the current environment

Date: 2026-10-07. Based on: `pix2grp-vg-compat.md`, `pgsg-run-log.md`, repo state at `84fe307` (branch `feature/query-conditioned-bayesian-stop`, uncommitted PGSG setup).

## Verdict

**Feasible on this machine via path F1 (or F2). Path P is almost certainly infeasible and should be budgeted as a short documented pilot, not a day.** No external API or GPU budget is needed; everything runs locally (RTX 5080 16 GB + ollama).

## What is already in place (§3 preflight items)

| Plan requirement | Status |
|---|---|
| VG images + GT (VG-150, test split) | Present and complete at `/home/kanaoka/usr/dataset/vg` (26,446 test imgs). |
| PGSG code + public weights | Submodule `Pix2Grp_CVPR2024` + `PGSG-CVPR2024/vg_sgg.pth` etc. Loads, "Missing keys []". |
| Standard PGSG inference | Runs end to end, no patches. 1000-img SGDet R@50/100 = 16.5/18.4 vs README 17.7/20.7. |
| Separate env / separate process | `envs/pgsg` (torch 2.7.1+cu128, transformers 4.29.2). EGTR env untouched. |
| Downstream text LLM | ollama `qwen3.5:4b` / `9b` local, used in MIRU pipeline. |
| Alternative vision LLM for F1 | `qwen3.5:4b` reports `vision` capability in ollama → F1 possible locally, zero API cost. |
| 20-level degraded images | `data/processed/degraded_images/` exists for the current 100 images (same scale array as plan). |
| Speed | ~3.0 s/img at max_len 1024 (12 GB @ bs8), 1.5 s/img at 512 (R@100 −5 pt). |

## Compute budget (§12)

- PGSG: 120 imgs × 20 levels = 2400 states × 3 s ≈ **2.0 h** (≈1 h at max_len 512). PGSG's internal 36-segment generation is already inside this figure.
- F1 downstream: ~4800 extra local LLM calls (check generation + judgment) × ~3–8 s on 4b ≈ 4–10 h, sequential. Fits in the 10/10–10/12 window.
- Pilot (25 states): minutes.

## Risks and gaps, ranked

1. **Path P will not work as written (high confidence).** PGSG's text decoder is BLIP's BERT-based `XBertLMHeadDecoder` (`lavis/models/med.py`), fine-tuned only on sequences `"<prompt> obj [OBJ] rel [REL] obj [OBJ] <coord tokens> ..."` (`blip_rel_det_pgsg.py` L321–331). The prompt is a fixed string with `max_txt_len=40` (L78); it is not an instruction-following model, and the plan's multi-sentence CHECKS instruction would be truncated at 40 tokens and ignored. Grounding uses per-token scores inside `generate`, so a CHECKS section cannot be separated without changing the decode loop. Recommendation: run the mandated pilot (25 states, initial prompt + 1 revision) in ≤ half a day purely to document failure, then go to F1. The plan already allows "初期pilotが不成立なら追加確認不要でこの代替に進める".
2. **Evaluation image set has EGTR-dependent selection bias.** The current 100 images were re-filtered in commit `2b038b2` to a "saturating subset" using EGTR-based stop times (`stop_c09<=18`): 35 right-censored images were dropped and refilled. The plan's §8 only allows GT criteria (objects ≥5, relations ≥3) plus split checks. Using the old 100 as-is would import an observer-dependent filter into a PGSG experiment. Decision needed: (a) re-select 100 test images from VG test with GT criteria + seed-42 shuffle (plan-conformant, needs regenerating degraded images, ~minutes), or (b) keep the old 100 and disclose the filter as a limitation. I recommend (a).
3. **Top resolution levels are degenerate for PGSG.** All 100 current images have max side 500 px; PGSG resizes input to 384×384. Effective resolution reaches 384 px only at level 19 (0.8538×500 = 427; level 18 = 365). Levels 19–20 are therefore nearly identical inputs for PGSG, and level 18 is close. Expect saturation at ≤18 by construction and `stop_t=20` to be rare. Disclose in limitations; do not reinterpret as a finding.
4. **Stochastic decoding.** Greedy (`generation_mode=search`) collapses to R@50 ≈ 5 and OOMs, because 36 segments share one prompt. Sampling with a fixed seed is bit-reproducible (two runs identical). Plan §11 "seedを画像・段階から決定" is the right approach. Side effect: q_t curves carry sampling noise → the relative-stable oracle may hit `unstable_tail` often. Report the count.
5. **Per-image predictions are not saved by stock `evaluate.py`.** A wrapper (`scripts/25_pgsg_infer.py`, plan in `pgsg-run-log.md` §7) must call `model.generate` directly and dump triplets (xyxy boxes in orig size, 0-based obj labels, predicate index offset to verify, top-512 ranked). Per-image R@50 then needs either the evaluator's per-image mode (`recall@50 ... (per images)` exists in logs) or a re-implementation validated against the official aggregate on a fixture (plan §11 test 10). The 1000-image official run is a usable fixture.
6. **Upstream evaluator bug**: ngR/ng-mR are meaningless (`vg_sgg_eval_tools.py` ~L316). Do not report them; R/mR are unaffected.
7. **Repo state mismatch with the plan.** Plan cites the Mac path and HEAD `869eeee` with a clean tree. Here: HEAD `84fe307`, feature branch, uncommitted PGSG files (`envs/pgsg`, `scripts/23`, `scripts/24`, `configs/pgsg`, `.gitignore`) plus unrelated query-conditioned work. Commit the PGSG setup (and ideally branch for ROBOVIS) before `preflight.json` records a commit.
8. **dev20 does not exist yet** (needs 20 new test images disjoint from the evaluation set + degradation). Trivial to generate.
9. **F1 with `qwen3.5:4b` as VLM**: MIRU found 9b overconfident and worse than 4b for text scoring; untested for vision-grounded check generation. Keep the 4b as default, record model ID and `think: false`.

## Not yet implemented (plan §10)

`pgsg_adapter.py`, `check_generator.py`, `stop_policy.py`, `robovis_eval.py`, `structured_io.py`, `scripts/run_robovis.py` (phases preflight/pilot/infer/freeze/evaluate), `configs/robovis2027.yaml`, acceptance tests 1–10. Reusable from MIRU: GT loader, config loader, 20-level generator (`02_generate_resolution_levels.py`), IoU, LLM client (`/api/chat`, `think:false`), save/resume.

## Recommended sequencing (fits §12)

1. 10/7: commit PGSG setup; write `preflight.json`; decide item 2 (image set). Build `scripts/25_pgsg_infer.py` + adapter; validate per-image R@50 against the official fixture.
2. 10/8 AM: P pilot (25 states) → expected fail → `pilot_report.md`; switch to F1.
3. 10/8–10/9: check generator + stop policy + structured IO + tests 3–9 (model-free); dev20 run; freeze.
4. 10/10–10/12: test 100 × 20 PGSG cache (~2 h) + F1 calls (~4–10 h), proposed / observed_only / fixed_k replay.
5. 10/13: evaluate, bootstrap, figures, report.
