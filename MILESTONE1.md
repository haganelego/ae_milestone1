# Milestone 1 — Semantic Information Saturation Detection from SGG Output

PoC for estimating **when semantic information saturates** as observation quality
improves, using an existing Scene Graph Generation method (**EGTR**) on
**Visual Genome**. Observation-quality improvement is approximated by stepwise
**resolution recovery** (20 levels, far → near). At each level we run EGTR,
compare the generated scene graph against the Visual Genome GT scene graph, and
study where Node / Triplet information stops growing. We compare a GT-based
**Oracle stop**, a **node-confidence baseline stop**, and an **LLM-based stop**
that scores the *sequence* of generated scene graphs (the LLM never sees images).

> Scope note: only the **resolution** aspect of approaching an object is modeled.
> Viewpoint change, occlusion, and motion planning are out of scope. The LLM is
> used only to score the structure/stability/completeness of SGG output
> sequences, not to interpret images.

## Layout

```
configs/config.yaml            # all params, thresholds, paths
src/milestone1/                # shared package
  config.py vocab.py gt.py egtr_infer.py metrics.py llm.py
scripts/                       # 01..10 thin pipeline stages
data/processed/
  selected_images.json         # chosen image_ids
  degraded_images/{id}/level_XX.jpg
  gt_scene_graphs/{id}.json
outputs/
  egtr_predictions/{id}/level_XX.json
  metrics/  llm_inputs/  llm_scores/  figures/
egtr/                          # EGTR submodule (imported, never modified)
```

## Environment

Everything runs through **EGTR's uv env** from the repo root (keeps cwd at root,
uses egtr's `.venv` which has torch 2.7.1+cu128 and all deps):

```bash
cd /home/kanaoka/usr/github/milestone_1
uv run --project egtr python scripts/01_select_vg_subset.py
```

Prereqs:
- EGTR checkpoint present under `egtr/checkpoint/...finetune.../epoch=03-...ckpt`
- Visual Genome at `/home/kanaoka/usr/dataset/vg` (`images/`, `train/val/test.json`, `rel.json`)
- **ollama** running with the model in `config.yaml` (`qwen3.5:4b`)

## Run order

```bash
uv run --project egtr python scripts/01_select_vg_subset.py        # subset (GT obj>=5, rel>=3)
uv run --project egtr python scripts/02_generate_resolution_levels.py  # 20-level degraded images
uv run --project egtr python scripts/03_run_egtr_inference.py      # EGTR per image/level -> JSON
uv run --project egtr python scripts/04_convert_egtr_outputs.py    # schema validation pass
uv run --project egtr python scripts/05_convert_vg_gt.py           # VG GT -> comparison JSON
uv run --project egtr python scripts/06_compute_sg_metrics.py      # Node/Triplet P/R/F1, SGScore
uv run --project egtr python scripts/07_compute_oracle_stop.py     # GT-based Oracle stop
uv run --project egtr python scripts/08_compute_baseline_stop.py   # node-confidence baseline stop
uv run --project egtr python scripts/09_llm_saturation_scoring.py  # ollama SG-sequence scoring
uv run --project egtr python scripts/10_plot_results.py            # curves, histograms, error CSV
```

## Key outputs

- `outputs/metrics/all_metrics.csv` — per image/level: node/triplet P/R/F1, `sg_score`, mean obj/rel confidence
- `outputs/metrics/{oracle,baseline,llm}_stop.csv` — stop level per method
- `outputs/metrics/stop_error_summary.csv` — MAE / median AE / early & late stop rate vs Oracle
- `outputs/figures/{id}_curve.png` — NodeRecall / TripletRecall / SGScore vs level + stop lines
- `outputs/figures/stop_distribution.png` — stop-level histogram across images

## Definitions

- **SGScore(t)** = `0.5 * NodeRecall(t) + 0.5 * TripletRecall(t)` (label-only, category-set matching)
- **OracleStop** = min level with `SGScore >= threshold` (main threshold in config)
- **BaselineStop** = min level with `mean_object_confidence >= theta`
- **LLMStop** = min level with `saturation_score >= threshold` from the LLM SG-sequence judge

## Important calibration notes (PoC findings)

EGTR raw relation scores are small (mostly `<0.1`), and absolute SGScores top out
around `0.4–0.7` under heavy degradation. The default thresholds in `config.yaml`
were therefore calibrated down from the original plan:

| param | plan | calibrated |
|---|---|---|
| `egtr.relation_conf_threshold` | 0.2 | 0.005 |
| `metrics.oracle_thresholds` | 0.6–0.9 | 0.3–0.6 |
| `metrics.main_oracle_threshold` | 0.8 | 0.5 |
| `baseline.mean_object_conf_threshold` | 0.7 | 0.4 |
| `llm.saturation_threshold` | 0.8 | 0.7 |

`qwen3.5:4b` runs in a `<think>` mode; the LLM client uses the `/api/chat`
endpoint with `"think": false` so the JSON answer is actually produced.

Current run is on **100 images** (`dataset.num_images`); the initial PoC validation used 10.
The 100-image evaluation (incl. relative-saturation Plan B and the ollama `num_ctx` fix) is
summarized in `.claude/docs/research/m1-scale100.md`.

## First-run results (10 images)

- **Q1 (info rises with resolution): YES.** Avg NodeRecall 0.17→0.64, TripletRecall 0.00→0.25, SGScore 0.08→0.44 from level 1→20.
- **Q2 (node saturates before relation): YES.** NodeRecall rises earlier/higher; TripletRecall stays ~0 until level 5–9 and plateaus later. This is the key PoC signal: *objects visible but relations not yet sufficient*.
- **Q3 (baseline vs oracle):** MAE 4.0 levels, 40% early / 0% late (mean stop 12.0 vs oracle 16.0). Node-confidence baseline tends to stop a bit early.
- **Q4 (LLM vs oracle):** MAE 8.4 levels, 80% early / 20% late (mean stop 10.2). The 4b model is aggressive/noisy (many parse failures); a larger model or refined prompt is the next lever.
