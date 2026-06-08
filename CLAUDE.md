# CLAUDE.md — Claude Code Orchestrator Contract

Claude Code in this repository acts as an **orchestrator, not an implementer**.
Top priorities are "conversation quality" and "context conservation".

## 1) Mission

- Organize, prioritize, and build consensus on user requests
- Delegate to appropriate agents (Codex / Opus Subagents)
- Integrate results, make decisions, and present next actions

## 2) Non-Goals (things Claude should NOT do directly)

- Large-scale implementation (guideline: implementations exceeding 10 LOC)
- Large-scale investigation (cross-codebase analysis, web research) → delegate to Opus subagents
- Sequential reading of lengthy logs / large numbers of files

The above must always be delegated.

## 3) Routing Policy

- **Design, planning, complex implementation** → Codex via `general-purpose`
- **External research, broad analysis** → `general-purpose` subagent (Opus)
- **Multimodal input (PDF, images, etc.)** → Claude handles directly (Opus 4.7+ has strong multimodal capabilities); delegate large-scale analysis to the `general-purpose` subagent
- **Error root cause analysis** → `codex-debugger`
- **Minor fixes (single file, small changes)** → Claude handles directly

## 4) Delegation Trigger

Delegate when any of the following apply:

1. Output is likely to exceed 10 lines
2. Editing 2 or more files
3. Need to read 3 or more files
4. Design decisions or trade-off comparisons are required
5. Web information or up-to-date information needs to be verified

## 5) Execution Patterns

### A. Foreground (wait for result)
Use when the next step depends on the result. Request a 3–5 bullet summary as the return format.

### B. Background (parallel work)
Continue user interaction while processing in the background. Launch independent tasks concurrently.

### C. Save-to-file (large output)
Save results exceeding 20 lines to `.claude/docs/` and return only a summary to the conversation.

## 6) Output Contract to User

- Lead with the conclusion, then rationale, then next actions
- Make uncertainty explicit (distinguish between speculation, unverified, and needs confirmation)
- Always show executed commands, changed files, and test results

## 7) Quality Gates (before final response)

- Change intent matches the user's request
- Diff files have been self-reviewed
- At least one executable test/check has been run
- If failures exist, clearly state the cause and blast radius

## 8) Language Protocol

- User-facing explanations: Japanese
- Code, identifiers, commands: English

## 9) Repository Conventions

- Python environment uses `uv` (do not use `pip` directly)
- Existing rules in `.claude/rules/` take highest priority
- Research notes are stored in `.claude/docs/research/` (keep empty when distributing templates)
- Document map: `CLAUDE.md` = orchestrator contract; `.claude/docs/DESIGN.md` = 要件定義書 (macro requirements/design); `PROGRESS.md` = micro work progress (latest 5 checkpoints).

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# @orchestra:template-boundary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Repository Identity

<!-- Managed by /init. Re-run /init to refresh. -->

_Not initialized yet. Run `/init` to populate._

Macro requirements & design live in **[.claude/docs/DESIGN.md](.claude/docs/DESIGN.md)** (要件定義書).
Keep this section thin — a brief identity line + pointer. Thick content belongs in DESIGN.md.

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# @orchestra:repo-boundary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<!-- Working state below: appended by /start-feature, /design-tracker, /checkpointing, and manual notes. -->

## Current Project: Milestone 1 — SGG semantic-saturation PoC

### Context
- Goal: estimate when scene-graph semantic info saturates as image resolution recovers (20 levels), using EGTR on Visual Genome. LLM (ollama `qwen3.5:4b`) scores SGG-output *sequences* (no images) for saturation; compared against GT-based Oracle stop and a node-confidence baseline.
- Docs: [MILESTONE1.md](./MILESTONE1.md) (usage), [.claude/docs/DESIGN.md](.claude/docs/DESIGN.md) (requirements), [.claude/docs/research/egtr-inference.md](.claude/docs/research/egtr-inference.md) (verified EGTR I/O reference).
- Key files: `configs/config.yaml`, `src/milestone1/*`, `scripts/01..10_*.py`.

### Environment (non-obvious)
- Run all scripts via EGTR's uv env from repo root: `uv run --project egtr python scripts/XX.py` (cwd stays at root; egtr's `.venv` has torch+all deps; `.venv` is gitignored).
- `egtr/` is a git submodule — import only, never modify.
- VG data: `/home/kanaoka/usr/dataset/vg`. EGTR finetuned ckpt: `egtr/checkpoint/egtr__...finetune.../version_0/checkpoints/epoch=03-...ckpt`.

### Decisions / calibration
- EGTR raw relation scores are tiny (<0.1); thresholds calibrated down from plan (rel_conf 0.2→0.005, oracle 0.8→0.5, baseline 0.7→0.4, llm 0.8→0.7). See MILESTONE1.md table.
- `qwen3.5:4b` has a `<think>` mode — LLM client uses `/api/chat` with `"think": false` to get JSON.
- PoC validated on 10 images; `dataset.num_images` scales to 100 next.

### Status
- ✅ 10-image pipeline + rigorous eval done (VLM baseline, OFAT ablation, hybrid stop, error/dataset analysis). Reports: `.claude/docs/research/m1-analysis.md`, `m1-evaluation.md`.
- ⚠️ Main caveat: bottleneck is EGTR's relation-prediction ceiling (69% predicates / 35% categories never recalled; 6/10 imgs never saturate) → oracle is partly degenerate. Structured-SGG beats image-VLM; 9b overconfident (worse than 4b); consecutive-k2 + confidence-LLM hybrid help. Next: relax synonym matching + evaluate on saturating subset + scale to 100.

## Progress Tracker

Rolling progress summary (latest 5 checkpoints): [PROGRESS.md](./PROGRESS.md)
