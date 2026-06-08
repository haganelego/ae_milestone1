# Design Document — 要件定義書 (Requirements & Macro Design)

> **Role:** Macro-level requirements and design — *what* this project builds and *why*.
> Written at `/init`, kept current by `/design-tracker` (also invoked from `/checkpointing`).
>
> **Document map:** Orchestrator contract → [CLAUDE.md](../../CLAUDE.md) ·
> Micro work progress (latest 5 checkpoints) → [PROGRESS.md](../../PROGRESS.md) ·
> Usage → [MILESTONE1.md](../../MILESTONE1.md) ·
> EGTR I/O reference → [research/egtr-inference.md](research/egtr-inference.md)

## 背景・目的 (Background & Purpose)

Long-term goal: a robot that, given a question/request intent, can autonomously
judge "I have observed enough to answer/decide — no need to get closer."
Milestone 1 is the **intent-free base step**: quantify how the semantic
information extractable from an image grows and stabilizes as observation quality
improves. Semantic information is represented as a **scene graph**; we study how
an existing SGG method's output (EGTR) changes as observation quality improves,
and whether the **saturation timing** can be estimated from SGG output alone.

Observation-quality improvement is approximated by **stepwise resolution
recovery** (far → near). This is a PoC for general SGG saturation detection;
person-centric applications are future background only — no person-specific rules.

## スコープ (Scope)

### In Scope

- Visual Genome image subset (GT obj≥5, rel≥3); 10 images first, scale to 100.
- 20-level resolution degradation (downscale→bilinear upscale to original size).
- EGTR inference at each level → objects + relationship triplets (+confidences, bbox).
- Comparison vs VG GT scene graph (label-only, category-set matching): Node & Triplet P/R/F1, SGScore.
- Saturation/stop estimation: GT-based Oracle stop, node-confidence Baseline stop, LLM SG-sequence stop.
- LLM (ollama `qwen3.5:4b`) scores the *sequence* of generated scene graphs (no image input).
- Visualization: per-image curves, stop-level histograms, stop-error summary.

### Out of Scope

- Question/task intent (added in later milestones).
- Viewpoint change, occlusion change, robot motion planning (only resolution is modeled).
- LLM image interpretation / VLM / captioning.
- Attribute evaluation (optional, deferred); bbox-aware (IoU) matching (deferred).
- Weight optimization for SGScore (fixed weights only).

## 機能要件 (Functional Requirements)

| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-1 | Select VG subset (GT obj≥5, rel≥3), deterministic | High | `01`, `selected_images.json` |
| FR-2 | Generate 20-level degraded image series (bilinear) | High | `02` |
| FR-3 | Run EGTR per image/level → standardized JSON (objects, triplets, conf, bbox) | High | `03`/`04` |
| FR-4 | Convert VG GT → comparison JSON (label-only) | High | `05` |
| FR-5 | Compute Node/Triplet P/R/F1 + SGScore + mean conf per level | High | `06` |
| FR-6 | Oracle stop (SGScore≥thr) for multiple thresholds + gain-based | High | `07` |
| FR-7 | Baseline stop (mean object confidence≥theta) | High | `08` |
| FR-8 | LLM SG-sequence saturation scoring → stop; replaceable LLM call | High | `09` |
| FR-9 | Plots (per-image curves + stop histograms) + stop-error summary | Medium | `10` |

## 非機能要件 (Non-Functional Requirements)

| Category | Requirement | Metric / Target |
|----------|-------------|-----------------|
| Performance | EGTR inference on degraded set | 10 img × 20 lvl runs in minutes on RTX 5080 |
| Reproducibility | Deterministic subset + fixed config | seed in config; all params in `config.yaml` |
| Robustness | LLM failures must not crash run | per-call failure → score 0.0, continue |
| Maintainability | Thin scripts over `src/milestone1` package | files 200–400 LOC; typed; English comments |
| Isolation | Do not modify EGTR submodule | import-only; run via `uv run --project egtr` |

## アーキテクチャ (Architecture)

Pipeline (10 stages) over a shared `src/milestone1` package. Data flow:
`select → degrade → EGTR infer → (convert) → GT convert → metrics → {oracle, baseline} stop → LLM stop → plots`.
All config-relative paths resolved against repo root; EGTR added to `sys.path`;
executed in EGTR's uv venv. LLM call isolated in `llm.py` (ollama `/api/chat`).

## 技術選定 (Tech Stack & Rationale)

| Area | Technology | Rationale | Alternatives Considered |
|------|------------|-----------|-------------------------|
| SGG | EGTR (CVPR24) | provided, finetuned VG ckpt, one-stage triplets+conf | other SGG models |
| Dataset | Visual Genome (EGTR split) | GT scene graphs match EGTR vocab directly | raw VG (not needed) |
| LLM | ollama `qwen3.5:4b` | local, free, fast; intent-free sequence scoring | larger local models (next) |
| Runtime | EGTR uv venv (torch 2.7.1+cu128) | all deps present; `.venv` gitignored | root project env (broken `ty>=0.1`) |
| Viz | matplotlib / pandas | simple PoC plots | — |

## 制約 (Constraints)

- EGTR raw relation scores are tiny (<0.1) → thresholds calibrated down (see MILESTONE1.md).
- `qwen3.5:4b` `<think>` mode → must use `/api/chat` with `"think": false`.
- GPU: RTX 5080 16 GB. `egtr/` is a submodule (read-only for this project).

## Key Decisions

| Decision | Rationale | Alternatives Considered | Date |
|----------|-----------|------------------------|------|
| New code at repo root (`src/scripts/configs/outputs`), egtr import-only | keep submodule clean | code inside egtr/ | 2026-06-05 |
| Run via `uv run --project egtr` from root | egtr venv has all deps, cwd stays at root | separate venv / fix root pyproject | 2026-06-05 |
| Label-only category-set matching | PoC simplicity per plan | bbox-aware IoU matching | 2026-06-05 |
| Calibrate thresholds down to data | EGTR rel scores / SGScores low | keep plan thresholds (degenerate) | 2026-06-05 |

## TODO / Open Questions

- [ ] Scale to 100 images and re-run.
- [ ] Reduce LLM early-stop bias / parse failures (larger model or prompt+JSON hardening).
- [ ] Evaluate consecutive-k stop conditions.
- [ ] Optional: attribute evaluation; bbox-aware matching.
