# ROBOVIS2027 work log (autonomous run started 2026-10-07)

Plan of record: user-supplied "ROBOVIS2027 実験計画書・自律実行仕様 v1.0" (2026-10-07). Feasibility: `robovis2027-feasibility.md`. PGSG internals: `pgsg-pipeline.md`, `pgsg-run-log.md`.

## Decisions (user-confirmed)
- Branch `feature/robovis2027` from `main` (= tag `miru2026`). Old query-conditioned branch archived as `archive/query-conditioned-wip`.
- **Path P dropped** (user decision after architecture review: PGSG is BLIP ViT-B + BERT-base, no instruction following, 40-token prompt cap, no training signal for a CHECKS section). `pilot_report.md` must record this rationale instead of a 25-state run.
- **Main paths: F1** (PGSG fixed + local VLM `qwen3.5:4b` sees image + graph → checks) and **F2** (text LLM, graph only). Downstream judgment: text LLM per plan §5. Stop rule per §6.
- Images pre-shrunk to max side 384 before the 20 levels (so no level is degenerate for PGSG's 384 input). Predictions mapped back to GT coords via `coord_scale` in `data/processed/robovis2027/levels_manifest.json`.
- Dataset re-selected GT-only: test 100 / dev 20 (`data/processed/robovis2027/selected_images.json`). Small start: dev20 first; test only after freeze.
- Everything local: PGSG on GPU (`envs/pgsg`), VLM/LLM via ollama `/api/chat` with `think: false`. No external API.

## Shared graph JSON contract (produced by `scripts/25_pgsg_infer.py`, consumed by check generator / stop policy / eval)
Path: `outputs/robovis2027/<run_id>/graphs/<image_id>/level_XX.json`
```json
{
  "image_id": 2335574, "split": "dev", "level": 7, "scale": 0.1288,
  "base_size": [384, 288], "orig_size": [500, 375], "coord_scale": [1.3021, 1.3021],
  "seed": 123456, "model": {"checkpoint": "vg_sgg.pth", "sha256": "..."},
  "generation": {"max_length": 1024, "min_length": 768, "top_p": 0.9, "repetition_penalty": 1.5, "num_segments": 23},
  "entities": [
    {"id": "v1", "label_id": 77, "label": "man", "score": 0.83,
     "label_dist_topk": [[77, 0.83], [111, 0.05]], "raw_tokens": "man",
     "box_xyxy_orig": [12.0, 30.5, 200.1, 370.0], "box_xyxy_base": [9.2, 23.4, 153.7, 284.2],
     "segment_ids": [0, 4, 9], "n_mentions": 3}
  ],
  "relations": [
    {"id": "e1", "sub": "v1", "obj": "v2", "pred_id": 31, "pred": "wearing",
     "pred_score": 0.71, "trp_score": 0.42, "pred_dist_topk": [[31, 0.71], [20, 0.1]],
     "rank": 0, "segment_id": 4}
  ],
  "stats": {"n_bad_inst": 2, "n_raw_triplets": 180, "n_relations_kept": 512, "elapsed_s": 3.1}
}
```
- `label_id` 0-based over the 150 VG object classes, `pred_id` 0-based over the 50 predicates (background excluded). Names from `data/processed/pgsg_vg/categories_dict.json`.
- `relations` sorted by `trp_score` desc (official ranking after entity NMS + relation dedup), up to 512. R@50 uses the first 50.
- Per-level R@50 lives in `outputs/robovis2027/<run_id>/stage_metrics.csv` (`image_id,split,t,scale,R50,mR50_hits,valid`).

## Check / judgment contracts: plan §5 verbatim. `structured_io.py` validates them (pydantic).

## Run ids
- `dev_r0`: dev20 × 20 levels, PGSG cache + F1 + F2 + judgments + stops.
- `test_r0`: test100 × 20, only after dev freeze.

## Log
- 2026-10-07: branch, dataset, config done (commits e820065, 8f4830d). Launched: (A1) PGSG wrapper + per-image eval + dev/test caching; (A2) model-free modules + tests; (A3) check generator + judge against the contract above with synthetic fixtures.
- A3 done (check_generator/judge/ollama_client, script 29, 49 tests). Smoke on 3 synthetic graphs: 4b 12/12 valid after 2 prompt fixes (JSON null wording; judge prompt lists check ids). JSON mode + think:false works; keep JSON mode on. Latency ≈4–5 s/call (4b and 9b similar; 9b 5.6 GB VRAM). Layout: `checks/<F1|F2>[__tag]/<id>/level_XX.json`, `decisions/...` (same).
  - RISKS seen: 4b always fills 3+3, never empty, restates "low score" as observed issues, proposed an existing relation as missing; **judge returned status=unresolved for 39/39 items** → stop rule may be degenerate via status; must inspect `limitation` distribution (rule needs low_resolution) and basis/benefit on real graphs. 9b cleaner (used empty list once).
  - DECISIONS (orchestrator): (1) schema retry under temp 0 + fixed seed is a no-op → the single retry uses seed+1, same prompt (document in report; still "same input"). (2) Model choice (4b vs 9b) deferred to dev20 comparison on levels {1,5,10,15,20}; both allowed on dev, frozen before test. (3) Anchors validated against rendered top-50 relations subset — acceptable, record in limitations.
- A2 done (structured_io/stop_policy/robovis_eval/robovis_replay, scripts 30/31, 91 tests, synthetic run under `outputs/robovis2027/synthetic_r0/eval/`). Eval outputs go to `<run>/eval/<PATH>/`; use `--path F1|F2[__tag]` on scripts 30/31. Missing/invalid/failed level → C_t missing for all conditions; images with any invalid stage level are excluded from the eval set (listed in `eval_set_ids.json`). Validation is plain dataclasses (no pydantic in egtr env). mR@50 needs `stage_predicate_hits.csv` from script 28 → requested from A1.
- A3 follow-ups done: retry uses seed+1 (records `seeds`, `seed_used`); `scripts/29b_robovis_checks_summary.py --run-id X --variant F1|F2|F2__9b` writes `summaries/checks_summary_<variant>.json` with gate fractions. Synthetic readout: all items unresolved, ~56% low_resolution, gate fires in 2/3 states but **0 items reach min(basis,benefit)>=2** → if real graphs look alike, C_t<2 everywhere → every image stops at t=2 (degenerate-early). KEY CHECK on dev_r0: columns "g&min" and "st_g&m" by level; expect them high at low levels and falling with level if the signal exists.
- A1 done: `pgsg_adapter.py`, `robovis_sgdet.py` (per-image SGDet == official `evaluate_relation_of_one_image`, max diff 0 on 400 states), scripts 25/28, dev_r0 graphs 400 states (3.07 s/state, bs4 ≈7.2 GB). Effective generation: nucleus top_p 0.9 / HF top_k 50, 36 segments (eval yaml seg_len 32), **repetition_penalty effectively 1.1 (hard-coded in med.py; task's 1.5 ignored)**, min_length unused, predicates top-3. Seed is per (batch of 4 consecutive images, level) not per image — recorded in manifest. Scores are amplified (>1 possible); rank by trp_score only.
  - dev20 R@50 (%): t1 0.00, t5 0.71, t10 13.43, t15 20.08, t20 19.59 → PGSG recovers nothing until ~t5, rises to t15, flat after. Official runner on same inputs (4 seeds) brackets our numbers. `pilot_report.md` written (Path P not run).
  - test_r0 inference running detached (PID in `outputs/robovis2027/logs/test_r0_infer.pid`), ETA ≈ 04:52 JST; then run `scripts/28_robovis_stage_metrics.py --run-id test_r0`.
  - Next: A3 running checks+judge (F1, F2, 4b) on dev levels {1,5,10,15,20}; A1 adding `stage_predicate_hits.csv` for mR.
