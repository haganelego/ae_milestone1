Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /home/kanaoka/usr/github/milestone_1
model: gpt-5.5
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: high
reasoning summaries: detailed
session id: 019ea50e-80d5-77b0-8f95-06ce8d7bb88e
--------
user
REVIEW-ONLY TASK. You have full disk access but you MUST NOT modify, create, or delete ANY file — only read and analyze. Do not run git commands that change state. If you are tempted to edit, instead describe the fix in the report.

Review the Milestone-1 SGG-saturation PoC's OWN code (NOT the egtr/ submodule). Read these files with your shell (cat/sed) and review for CORRECTNESS of scientific logic first, then robustness, then style.

Files to read:
- src/milestone1/: config.py, vocab.py, gt.py, egtr_infer.py, metrics.py, llm.py, llm_engine.py, vlm.py, analysis.py
- scripts/: 02_generate_resolution_levels.py, 06_compute_sg_metrics.py, 07_compute_oracle_stop.py, 08_compute_baseline_stop.py, 15_relative_saturation.py, aggregate_100.py, run_focused_100.py
- configs/config.yaml (for thresholds/keys)

Focus:
1. metrics.py set-based label-only node/triplet precision/recall/F1 — correctness, empty-set/zero-division edge cases.
2. egtr_infer.py EGTR triplet ranking: softmax over logits[:150]; pred_rel*pred_connectivity gating; argsort_desc top-K 'single predicate per pair'; query_id->object_id mapping; bbox rescale. Any indexing/off-by-one bug?
3. scripts/15 relative saturation: pseudo-GT = each image's level_20 EGTR output; RelSGScore=0.5*RelNodeRecall+0.5*RelTripletRecall; RelOracleStop=first level>=threshold; gain-based stop. Level range 1..20 off-by-one? empty level-20 triplets handling?
4. aggregate_100.py: hybrid = conf_weight*mean_object_confidence + llm_weight*saturation_score vs threshold; consecutive-k; MAE/early/late vs oracle; 'never saturated -> 20' handling; reading cached scores by config slug.
5. llm.py/llm_engine.py/vlm.py ollama: num_ctx=12288, think=false, JSON parse robustness (_parse_llm_json regex), wall-clock timeout.

Output STRUCTURED markdown ONLY (this is captured to a file):
## Summary (2-3 lines)
## Correctness bugs (each: file:line, severity High/Med/Low, what's wrong, suggested fix)
## Metric/logic issues
## Robustness
## Style/convention
## Top 5 fixes (ranked)
Be concrete and skeptical; cite file:line; prefer real bugs over nits.
codex
Using `context-loader` first as required, then I’ll read only the requested project files and report findings without modifying anything.
exec
/usr/bin/zsh -lc "sed -n '1,220p' /home/kanaoka/usr/github/milestone_1/.codex/skills/context-loader/SKILL.md && pwd" in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
---
name: context-loader
description: ALWAYS activate this skill at the start of every task. Load project context from .claude/ directory including coding rules, design decisions, and documentation before executing any task.
---

# Context Loader Skill

## Purpose

Load shared project context from `.claude/` directory to ensure Codex CLI has the same knowledge as Claude Code.

## When to Activate

**ALWAYS** - This skill must run at the beginning of every task to load project context.

## Workflow

### Step 1: Load Coding Rules

Read all files in `.claude/rules/`:

```
.claude/rules/
├── coding-principles.md   # Simplicity, single responsibility, early return
├── dev-environment.md     # uv, ruff, ty, pytest requirements
├── language.md            # Think in English, respond in Japanese
├── security.md            # Secrets, validation, SQLi/XSS prevention
├── testing.md             # TDD, AAA pattern, 80% coverage
└── codex-delegation.md    # (skip - not relevant for Codex itself)
```

### Step 2: Load Design Documentation

Read `.claude/docs/DESIGN.md` for:
- Architecture decisions
- Implementation patterns
- Library choices and constraints
- TODO items and open questions

### Step 3: Check Library Documentation

If the task involves specific libraries, read relevant files from:
```
.claude/docs/libraries/
```

### Step 4: Execute Task

With the loaded context, execute the requested task following:
- Coding principles from rules
- Design decisions from DESIGN.md
- Library constraints from docs

## Key Rules to Remember

After loading, always follow these principles:

1. **Simplicity first** - Choose readable code over complex
2. **Single responsibility** - One function/class does one thing
3. **Type hints required** - All functions need annotations
4. **Use uv** - Never use pip directly
5. **Security** - No hardcoded secrets, validate input, parameterize SQL

## Language Protocol

- **Thinking/Reasoning**: English
- **Code**: English (variables, functions, comments)
- **User communication**: Japanese (when reporting back through Claude Code)

## Output

After loading context, briefly confirm:
- Rules loaded
- Design document status
- Ready to execute task
/home/kanaoka/usr/github/milestone_1

**Reviewing project requirements**

I see that we only need a review, so the focus will be on loading the .claude rules and design. It sounds like I should stick to using shell commands like cat and sed without making any modifications. There’s a possibility I might use a plan, but I’m not quite sure if Japanese is the right choice since the user likely wants structured markdown in English. I'll ensure I read the files and consider using parallel processing.
codex
Context-loader points to `.claude/rules` and `DESIGN.md`; I’m loading those now, then I’ll inspect the requested PoC files with line numbers so the review can cite exact locations.
exec
/usr/bin/zsh -lc 'rg --files .claude/rules .claude/docs | sort' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
.claude/docs/CODEX_HANDOFF_PLAYBOOK.md
.claude/docs/DESIGN.md
.claude/docs/research/egtr-inference.md
.claude/docs/research/egtr-paper-claims.md
.claude/docs/research/egtr-repro.md
.claude/docs/research/egtr_eval_full_0.json
.claude/docs/research/egtr_eval_subset_500.json
.claude/docs/research/m1-analysis.md
.claude/docs/research/m1-code-review.md
.claude/docs/research/m1-codex-review.md
.claude/docs/research/m1-evaluation.md
.claude/docs/research/m1-relative-saturation.md
.claude/docs/research/m1-scale100.md
.claude/rules/codex-delegation.md
.claude/rules/coding-principles.md
.claude/rules/dev-environment.md
.claude/rules/language.md
.claude/rules/security.md
.claude/rules/testing.md

exec
/usr/bin/zsh -lc "sed -n '1,240p' .claude/docs/DESIGN.md" in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
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

codex
Rules/design are loaded. The design confirms label-only set matching and calibrated thresholds are intentional, so I’ll separate those project choices from implementation bugs.
exec
/usr/bin/zsh -lc "for f in .claude/rules/coding-principles.md .claude/rules/dev-environment.md .claude/rules/language.md .claude/rules/security.md .claude/rules/testing.md; do printf '\\n## %s\\n' \""'$f"; sed -n '"'1,220p' \""'$f"; done' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:

## .claude/rules/coding-principles.md
# Coding Principles

Core coding rules to always follow.

## Simplicity First

- Choose readable code over complex code
- Avoid over-abstraction
- Prioritize "understandable" over "working"

## Single Responsibility

- One function does one thing only
- One class has one responsibility only
- Target 200-400 lines per file (max 800)

## Early Return

```python
# Bad: Deep nesting
def process(value):
    if value is not None:
        if value > 0:
            return do_something(value)
    return None

# Good: Early return
def process(value):
    if value is None:
        return None
    if value <= 0:
        return None
    return do_something(value)
```

## Type Hints Required

All functions must have type annotations:

```python
def call_llm(
    prompt: str,
    model: str = "gpt-4",
    max_tokens: int = 1000
) -> str:
    ...
```

## Immutability

Create new objects instead of mutating existing ones:

```python
# Bad: Mutating existing object
data["new_key"] = value

# Good: Creating new object
new_data = {**data, "new_key": value}
```

## Naming Conventions

- **Variables/Functions**: snake_case (English)
- **Classes**: PascalCase (English)
- **Constants**: UPPER_SNAKE_CASE (English)
- **Meaningful names**: `user_count` over `x`

## No Magic Numbers

```python
# Bad
if retry_count > 3:
    ...

# Good
MAX_RETRIES = 3
if retry_count > MAX_RETRIES:
    ...
```

## .claude/rules/dev-environment.md
# Development Environment

Project development environment and toolchain.

## Package Management: uv

**Do not use pip directly. All commands must go through uv.**

```bash
# Add packages
uv add <package>
uv add --dev <package>    # Dev dependency

# Sync dependencies
uv sync

# Run scripts
uv run <command>
uv run python script.py
uv run pytest
```

### pyproject.toml

Manage dependencies in `pyproject.toml`:

```toml
[project]
dependencies = [
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.8",
]
```

## Linting & Formatting: ruff

```bash
# Check
uv run ruff check .

# Auto-fix
uv run ruff check --fix .

# Format
uv run ruff format .
```

### ruff Configuration (pyproject.toml)

```toml
[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "UP",     # pyupgrade
]
ignore = ["E501"]  # line too long (formatter handles)

[tool.ruff.format]
quote-style = "double"
```

## Type Checking: ty

> Note: `ty` is currently in beta (0.0.x); the API and behavior are unstable. See https://docs.astral.sh/ty/ for status.
>
> Install: `uv tool install ty` (global) or `uv add --dev ty` (project dev dep).

```bash
# Run type check
uv run ty check src/
```

### ty Features

- Fast Rust-based type checker (by Astral)
- Same ecosystem as ruff / uv
- mypy-compatible type annotations

## Notebooks: marimo

Interactive Python notebook environment.

```bash
# Create/edit notebook
uv run marimo edit notebook.py

# Run notebook (CLI)
uv run marimo run notebook.py

# Deploy as app
uv run marimo run notebook.py --host 0.0.0.0 --port 8080
```

### marimo Features

- **Pure Python files** (.py): Git-friendly
- **Reactive**: Auto-tracks cell dependencies
- **Reproducible**: No execution order dependency

### marimo Best Practices

```python
# Bad: Mutating global state
data = []
def add_item(item):
    data.append(item)  # Side effect

# Good: Pure function
def add_item(data: list, item) -> list:
    return [*data, item]
```

## Task Runner

Manage multiple tool executions in `pyproject.toml` scripts or poe:

```toml
[tool.poe.tasks]
lint = "ruff check . && ruff format --check ."
format = "ruff check --fix . && ruff format ."
typecheck = "ty check src/"
test = "pytest -v"
all = ["lint", "typecheck", "test"]
```

## Common Commands

```bash
# Initialize
uv init
uv venv
source .venv/bin/activate

# Install dev dependencies
uv sync --all-extras

# Quality check (all)
uv run ruff check . && uv run ruff format --check . && uv run ty check src/ && uv run pytest

# Or via poe
poe all
```

## Pre-commit Checklist

- [ ] `uv run ruff check .` passes
- [ ] `uv run ruff format --check .` passes
- [ ] `uv run ty check src/` passes
- [ ] `uv run pytest` passes

## .claude/rules/language.md
# Language Rules

## Thinking and Reasoning

- **Always think and reason in English**
- Internal analysis, planning, and problem-solving should be in English
- Code comments, variable names, function names, and docstrings should be in English

## User Communication

- **Always respond to users in Japanese**
- Explanations, questions, and status updates should be in Japanese
- Error messages shown to users should be in Japanese

## Code

- All code should be written in English:
  - Variable names: `user_count`, not `ユーザー数`
  - Function names: `calculate_total()`, not `合計計算()`
  - Comments: `# Check if valid`, not `# 有効かチェック`
  - Docstrings: English descriptions

## Documentation

- Technical documentation: English
- User-facing documentation (README, etc.): English

## .claude/rules/security.md
# Security Rules

Security checklist to always verify when writing code.

## Secrets Management

### Never Do

- Hardcode API keys or passwords
- Log sensitive information
- Commit `.env` files

### Required

```python
# Good: Get from environment variables
import os
API_KEY = os.environ["API_KEY"]

# Good: With existence check
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable is required")
```

## Input Validation

Always validate external input:

```python
from pydantic import BaseModel, EmailStr, Field

class UserInput(BaseModel):
    email: EmailStr
    age: int = Field(ge=0, le=150)
    name: str = Field(min_length=1, max_length=100)
```

## SQL Injection Prevention

```python
# Bad: String concatenation
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")

# Good: Parameterized query
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
```

## XSS Prevention

- Escape user input before embedding in HTML
- Enable template engine auto-escaping

## Error Messages

```python
# Bad: Too detailed (gives attackers information)
raise Exception(f"Database connection failed: {connection_string}")

# Good: Minimal information
raise Exception("Database connection failed")
# Details go to logs (logs are private)
logger.error(f"Database connection failed: {connection_string}")
```

## Dependencies

- Regular vulnerability checks: `pip-audit`, `safety`
- Remove unused dependencies
- Pin versions (`==` over `>=`)

## Code Review Checklist

- [ ] No hardcoded secrets
- [ ] External input is validated
- [ ] SQL queries are parameterized
- [ ] Error messages are not too detailed
- [ ] Logs don't contain sensitive information

## .claude/rules/testing.md
# Testing Rules

Guidelines for writing tests.

## Core Principles

- **TDD recommended**: Write tests first
- **Coverage target**: 80% or higher
- **Execution speed**: Unit tests should be fast (< 100ms per test)

## Test Structure

### AAA Pattern

```python
def test_user_creation():
    # Arrange
    user_data = {"name": "Alice", "email": "alice@example.com"}

    # Act
    user = create_user(user_data)

    # Assert
    assert user.name == "Alice"
    assert user.email == "alice@example.com"
```

### Naming Convention

```python
# test_{target}_{condition}_{expected_result}
def test_create_user_with_valid_data_returns_user():
    ...

def test_create_user_with_invalid_email_raises_error():
    ...
```

## Test Case Coverage

For each feature, consider:

1. **Happy path**: Basic functionality
2. **Boundary values**: Min, max, empty
3. **Error cases**: Invalid input, error conditions
4. **Edge cases**: None, empty string, special characters

## Mocking

Mock external dependencies:

```python
from unittest.mock import Mock, patch

@patch("module.external_api_call")
def test_with_mocked_api(mock_api):
    mock_api.return_value = {"status": "ok"}
    result = function_under_test()
    assert result == expected
```

## Fixtures

Common setup goes in `conftest.py`:

```python
# tests/conftest.py
import pytest

@pytest.fixture
def sample_user():
    return User(name="Test", email="test@example.com")

@pytest.fixture
def db_session():
    session = create_session()
    yield session
    session.rollback()
```

## Commands

```bash
# All tests
uv run pytest -v

# Specific file
uv run pytest tests/test_user.py -v

# Specific test
uv run pytest tests/test_user.py::test_create_user -v

# With coverage
uv run pytest --cov=src --cov-report=term-missing

# Stop on first failure
uv run pytest -x
```

## Checklist

- [ ] Happy path is tested
- [ ] Error cases are tested
- [ ] Boundary values are tested
- [ ] Tests are independent (no order dependency)
- [ ] External dependencies are mocked
- [ ] Tests run fast

exec
/usr/bin/zsh -lc 'nl -ba src/milestone1/config.py src/milestone1/vocab.py src/milestone1/gt.py src/milestone1/egtr_infer.py src/milestone1/metrics.py' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
     1	"""Configuration loading and path resolution."""
     2	
     3	import sys
     4	from pathlib import Path
     5	from typing import Any
     6	
     7	import yaml
     8	
     9	REPO_ROOT = Path(__file__).resolve().parents[2]
    10	
    11	
    12	def load_config(config_path: str | None = None) -> dict[str, Any]:
    13	    """Load the YAML configuration file."""
    14	    if config_path is None:
    15	        config_path = str(REPO_ROOT / "configs" / "config.yaml")
    16	    with open(config_path) as f:
    17	        return yaml.safe_load(f)
    18	
    19	
    20	def resolve(path: str) -> Path:
    21	    """Resolve a config-relative path against REPO_ROOT."""
    22	    p = Path(path)
    23	    if p.is_absolute():
    24	        return p
    25	    return REPO_ROOT / p
    26	
    27	
    28	def setup_egtr_path() -> None:
    29	    """Add EGTR submodule to sys.path for imports."""
    30	    egtr_path = str(REPO_ROOT / "egtr")
    31	    if egtr_path not in sys.path:
    32	        sys.path.insert(0, egtr_path)
    33	"""Load object and predicate vocabularies from Visual Genome JSON files."""
    34	
    35	import json
    36	from pathlib import Path
    37	
    38	
    39	def load_obj_names(vg_root: str | Path) -> dict[int, str]:
    40	    """Load object category names from test.json.
    41	
    42	    Returns a dict mapping 0-indexed class id (0..149) to name.
    43	    Model output pred_classes uses 0-indexed ids.
    44	    """
    45	    with open(Path(vg_root) / "test.json") as f:
    46	        vg_data = json.load(f)
    47	    return {
    48	        c["id"] - 1: c["name"]
    49	        for c in sorted(vg_data["categories"], key=lambda x: x["id"])
    50	    }
    51	
    52	
    53	def load_pred_names(vg_root: str | Path) -> list[str]:
    54	    """Load predicate names from rel.json.
    55	
    56	    Returns a list of 50 predicate names, 0-indexed matching model output.
    57	    """
    58	    with open(Path(vg_root) / "rel.json") as f:
    59	        rel_data = json.load(f)
    60	    return rel_data["rel_categories"][1:]  # skip __background__
    61	"""Ground truth scene graph loading from Visual Genome annotations."""
    62	
    63	import json
    64	from pathlib import Path
    65	from typing import Any
    66	
    67	from pycocotools.coco import COCO
    68	
    69	
    70	def _load_coco_and_rels(
    71	    vg_root: str | Path, split: str
    72	) -> tuple[COCO, dict[str, Any], dict[str, Any]]:
    73	    """Load COCO annotations and relationship data."""
    74	    vg_root = Path(vg_root)
    75	    # Suppress COCO stdout
    76	    import io
    77	    import sys
    78	    old_stdout = sys.stdout
    79	    sys.stdout = io.StringIO()
    80	    try:
    81	        coco = COCO(str(vg_root / f"{split}.json"))
    82	    finally:
    83	        sys.stdout = old_stdout
    84	
    85	    with open(vg_root / "rel.json") as f:
    86	        rel_data = json.load(f)
    87	
    88	    return coco, rel_data, rel_data[split]
    89	
    90	
    91	def count_gt_objects_and_relationships(
    92	    vg_root: str | Path, split: str
    93	) -> list[dict[str, Any]]:
    94	    """Count GT objects and relationships per image.
    95	
    96	    Returns list of {image_id, num_objects, num_relationships}.
    97	    """
    98	    coco, rel_data, rels_split = _load_coco_and_rels(vg_root, split)
    99	
   100	    results = []
   101	    for img_info in coco.dataset["images"]:
   102	        image_id = img_info["id"]
   103	        ann_ids = coco.getAnnIds(imgIds=image_id)
   104	        num_objects = len(ann_ids)
   105	        rel_list = rels_split.get(str(image_id), [])
   106	        num_relationships = len(rel_list)
   107	        results.append({
   108	            "image_id": image_id,
   109	            "num_objects": num_objects,
   110	            "num_relationships": num_relationships,
   111	        })
   112	    return results
   113	
   114	
   115	def load_gt_scene_graph(
   116	    vg_root: str | Path, split: str, image_id: int
   117	) -> dict[str, Any]:
   118	    """Load GT scene graph for a single image.
   119	
   120	    Returns {
   121	        image_id: int,
   122	        objects: [{object_id, label, bbox: [x1,y1,x2,y2]}],
   123	        relationships: [{subject_label, predicate, object_label}]
   124	    }
   125	    """
   126	    coco, rel_data, rels_split = _load_coco_and_rels(vg_root, split)
   127	
   128	    # Object categories (0-indexed)
   129	    id2name = {cid - 1: info["name"] for cid, info in coco.cats.items()}
   130	
   131	    # Predicate names (0-indexed)
   132	    pred_names = rel_data["rel_categories"][1:]
   133	
   134	    # Objects
   135	    ann_ids = coco.getAnnIds(imgIds=image_id)
   136	    anns = coco.loadAnns(ann_ids)
   137	
   138	    objects = []
   139	    for i, ann in enumerate(anns):
   140	        cls_name = id2name[ann["category_id"] - 1]
   141	        # Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]
   142	        x, y, w, h = ann["bbox"]
   143	        bbox = [x, y, x + w, y + h]
   144	        objects.append({
   145	            "object_id": i,
   146	            "label": cls_name,
   147	            "bbox": bbox,
   148	        })
   149	
   150	    # Relationships
   151	    rel_list = rels_split.get(str(image_id), [])
   152	    relationships = []
   153	    for subj_idx, obj_idx, pred_id_1indexed in rel_list:
   154	        if subj_idx < len(anns) and obj_idx < len(anns):
   155	            subj_name = id2name[anns[subj_idx]["category_id"] - 1]
   156	            obj_name = id2name[anns[obj_idx]["category_id"] - 1]
   157	            pred_name = pred_names[pred_id_1indexed - 1]
   158	            relationships.append({
   159	                "subject_label": subj_name,
   160	                "predicate": pred_name,
   161	                "object_label": obj_name,
   162	            })
   163	
   164	    return {
   165	        "image_id": image_id,
   166	        "objects": objects,
   167	        "relationships": relationships,
   168	    }
   169	"""EGTR model wrapper for scene graph generation inference."""
   170	
   171	import sys
   172	from pathlib import Path
   173	from typing import Any
   174	
   175	import numpy as np
   176	import torch
   177	from PIL import Image
   178	
   179	from milestone1.config import REPO_ROOT, setup_egtr_path
   180	from milestone1.vocab import load_obj_names, load_pred_names
   181	
   182	# Ensure EGTR is importable
   183	setup_egtr_path()
   184	
   185	from glob import glob
   186	
   187	from lib.pytorch_misc import argsort_desc
   188	from model.deformable_detr import (
   189	    DeformableDetrConfig,
   190	    DeformableDetrFeatureExtractor,
   191	)
   192	from model.egtr import DetrForSceneGraphGeneration
   193	from util.box_ops import rescale_bboxes
   194	
   195	
   196	class EgtrModel:
   197	    """Wrapper around EGTR for single-image scene graph inference."""
   198	
   199	    def __init__(
   200	        self,
   201	        artifact_path: str,
   202	        architecture: str,
   203	        vg_root: str,
   204	        object_conf_threshold: float = 0.3,
   205	        relation_conf_threshold: float = 0.2,
   206	        top_k_objects: int = 20,
   207	        top_k_relationships: int = 30,
   208	    ) -> None:
   209	        self.object_conf_threshold = object_conf_threshold
   210	        self.relation_conf_threshold = relation_conf_threshold
   211	        self.top_k_objects = top_k_objects
   212	        self.top_k_relationships = top_k_relationships
   213	
   214	        # Load vocabularies
   215	        self.obj_names = load_obj_names(vg_root)
   216	        self.pred_names = load_pred_names(vg_root)
   217	
   218	        # Load model
   219	        config = DeformableDetrConfig.from_pretrained(artifact_path)
   220	        self.num_labels = config.num_labels  # 150
   221	
   222	        self.model = DetrForSceneGraphGeneration.from_pretrained(
   223	            architecture, config=config, ignore_mismatched_sizes=True
   224	        )
   225	
   226	        # Load finetuned checkpoint
   227	        ckpt_path = sorted(
   228	            glob(f"{artifact_path}/checkpoints/epoch=*.ckpt"),
   229	            key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
   230	        )[-1]
   231	        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
   232	        # Strip "model." prefix from PyTorch Lightning keys
   233	        for k in list(state_dict.keys()):
   234	            state_dict[k[6:]] = state_dict.pop(k)
   235	        self.model.load_state_dict(state_dict)
   236	        self.model.cuda()
   237	        self.model.eval()
   238	
   239	        # Feature extractor
   240	        self.feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
   241	            architecture, size=800, max_size=1333
   242	        )
   243	
   244	    def predict(self, pil_image: Image.Image) -> dict[str, Any]:
   245	        """Run inference on a single PIL image.
   246	
   247	        Returns {
   248	            objects: [{object_id, label, confidence, bbox: [x1,y1,x2,y2]}],
   249	            relationships: [{subject_id, subject_label, predicate,
   250	                             object_id, object_label, confidence}]
   251	        }
   252	        """
   253	        img_w, img_h = pil_image.size
   254	
   255	        # Preprocess
   256	        encoding = self.feature_extractor(
   257	            images=pil_image, return_tensors="pt"
   258	        )
   259	        pixel_values = encoding["pixel_values"].cuda()
   260	        pixel_mask = encoding["pixel_mask"].cuda()
   261	
   262	        # Forward pass
   263	        with torch.no_grad():
   264	            outputs = self.model(
   265	                pixel_values=pixel_values,
   266	                pixel_mask=pixel_mask,
   267	                output_attentions=False,
   268	                output_attention_states=True,
   269	                output_hidden_states=True,
   270	            )
   271	
   272	        # Extract objects
   273	        logits = outputs.logits[0]  # (200, 150)
   274	        pred_boxes = outputs.pred_boxes[0]  # (200, 4) cx,cy,w,h normalized
   275	        probs = logits.softmax(-1)
   276	        obj_scores, pred_classes = torch.max(
   277	            probs[:, : self.num_labels], dim=-1
   278	        )
   279	
   280	        # Convert boxes to absolute [x1, y1, x2, y2]
   281	        abs_boxes = rescale_bboxes(
   282	            pred_boxes.cpu(), (img_w, img_h)
   283	        )  # (200, 4)
   284	
   285	        # Filter and rank objects by confidence
   286	        valid_mask = obj_scores > self.object_conf_threshold
   287	        valid_indices = torch.where(valid_mask)[0]
   288	
   289	        # Sort by confidence descending, take top_k
   290	        if len(valid_indices) > 0:
   291	            sorted_idx = obj_scores[valid_indices].argsort(descending=True)
   292	            valid_indices = valid_indices[sorted_idx[: self.top_k_objects]]
   293	        else:
   294	            # Fallback: take top_k by score regardless of threshold
   295	            sorted_idx = obj_scores.argsort(descending=True)
   296	            valid_indices = sorted_idx[: self.top_k_objects]
   297	
   298	        objects = []
   299	        for rank, idx in enumerate(valid_indices):
   300	            idx_int = idx.item()
   301	            objects.append({
   302	                "object_id": rank,
   303	                "query_id": idx_int,
   304	                "label": self.obj_names[pred_classes[idx_int].item()],
   305	                "confidence": round(obj_scores[idx_int].item(), 4),
   306	                "bbox": [round(x, 1) for x in abs_boxes[idx_int].tolist()],
   307	            })
   308	
   309	        # Extract relationships (Method A: single pred per pair)
   310	        pred_rel = torch.clamp(outputs.pred_rel[0], 0.0, 1.0)  # (200,200,50)
   311	        pred_conn = torch.clamp(
   312	            outputs.pred_connectivity[0], 0.0, 1.0
   313	        )  # (200,200,1)
   314	        pred_rel = pred_rel * pred_conn  # gate by connectivity
   315	
   316	        # Object score outer product
   317	        sub_ob_scores = torch.outer(obj_scores, obj_scores)  # (200, 200)
   318	        n_queries = sub_ob_scores.shape[0]
   319	        sub_ob_scores[
   320	            torch.arange(n_queries), torch.arange(n_queries)
   321	        ] = 0.0
   322	
   323	        # Rank triplets
   324	        triplet_scores = pred_rel.max(-1)[0] * sub_ob_scores  # (200, 200)
   325	        top_pairs = argsort_desc(triplet_scores.cpu().numpy())[
   326	            : self.top_k_relationships * 3
   327	        ]
   328	
   329	        # Build query_id -> object_id mapping for valid objects
   330	        query_to_obj = {
   331	            obj["query_id"]: obj["object_id"] for obj in objects
   332	        }
   333	        valid_query_ids = set(query_to_obj.keys())
   334	
   335	        relationships = []
   336	        for s_q, o_q in top_pairs:
   337	            if s_q not in valid_query_ids or o_q not in valid_query_ids:
   338	                continue
   339	
   340	            best_pred = pred_rel[s_q, o_q].argmax().item()
   341	            rel_score = pred_rel[s_q, o_q, best_pred].item()
   342	
   343	            if rel_score < self.relation_conf_threshold:
   344	                continue
   345	
   346	            relationships.append({
   347	                "subject_id": query_to_obj[s_q],
   348	                "subject_label": self.obj_names[pred_classes[s_q].item()],
   349	                "predicate": self.pred_names[best_pred],
   350	                "object_id": query_to_obj[o_q],
   351	                "object_label": self.obj_names[pred_classes[o_q].item()],
   352	                "confidence": round(rel_score, 4),
   353	            })
   354	
   355	            if len(relationships) >= self.top_k_relationships:
   356	                break
   357	
   358	        return {"objects": objects, "relationships": relationships}
   359	"""Scene graph evaluation metrics with category-set semantics."""
   360	
   361	from typing import Any
   362	
   363	
   364	def _to_label_set(objects: list[dict[str, Any]]) -> set[str]:
   365	    """Extract unique label set from objects."""
   366	    return {obj["label"] for obj in objects}
   367	
   368	
   369	def _to_triplet_set(
   370	    relationships: list[dict[str, Any]],
   371	) -> set[tuple[str, str, str]]:
   372	    """Extract unique triplet set from relationships."""
   373	    return {
   374	        (rel["subject_label"], rel["predicate"], rel["object_label"])
   375	        for rel in relationships
   376	    }
   377	
   378	
   379	def node_precision_recall_f1(
   380	    pred_objects: list[dict[str, Any]],
   381	    gt_objects: list[dict[str, Any]],
   382	) -> tuple[float, float, float]:
   383	    """Compute node (object label) precision, recall, F1.
   384	
   385	    Uses category-set semantics: dedup to sets of labels.
   386	    """
   387	    pred_set = _to_label_set(pred_objects)
   388	    gt_set = _to_label_set(gt_objects)
   389	
   390	    if not pred_set and not gt_set:
   391	        return 1.0, 1.0, 1.0
   392	    if not pred_set:
   393	        return 0.0, 0.0, 0.0
   394	    if not gt_set:
   395	        return 0.0, 0.0, 0.0
   396	
   397	    tp = len(pred_set & gt_set)
   398	    precision = tp / len(pred_set) if pred_set else 0.0
   399	    recall = tp / len(gt_set) if gt_set else 0.0
   400	    f1 = (
   401	        2 * precision * recall / (precision + recall)
   402	        if (precision + recall) > 0
   403	        else 0.0
   404	    )
   405	    return precision, recall, f1
   406	
   407	
   408	def triplet_precision_recall_f1(
   409	    pred_relationships: list[dict[str, Any]],
   410	    gt_relationships: list[dict[str, Any]],
   411	) -> tuple[float, float, float]:
   412	    """Compute triplet (subject, predicate, object) precision, recall, F1.
   413	
   414	    Uses category-set semantics: dedup to sets of triplets.
   415	    """
   416	    pred_set = _to_triplet_set(pred_relationships)
   417	    gt_set = _to_triplet_set(gt_relationships)
   418	
   419	    if not pred_set and not gt_set:
   420	        return 1.0, 1.0, 1.0
   421	    if not pred_set:
   422	        return 0.0, 0.0, 0.0
   423	    if not gt_set:
   424	        return 0.0, 0.0, 0.0
   425	
   426	    tp = len(pred_set & gt_set)
   427	    precision = tp / len(pred_set) if pred_set else 0.0
   428	    recall = tp / len(gt_set) if gt_set else 0.0
   429	    f1 = (
   430	        2 * precision * recall / (precision + recall)
   431	        if (precision + recall) > 0
   432	        else 0.0
   433	    )
   434	    return precision, recall, f1
   435	
   436	
   437	def sg_score(
   438	    node_recall: float,
   439	    triplet_recall: float,
   440	    node_weight: float = 0.5,
   441	    relation_weight: float = 0.5,
   442	) -> float:
   443	    """Compute weighted scene graph score."""
   444	    return node_weight * node_recall + relation_weight * triplet_recall
   445	
   446	
   447	def mean_confidence(items: list[dict[str, Any]], key: str = "confidence") -> float:
   448	    """Compute mean confidence from a list of items."""
   449	    if not items:
   450	        return 0.0
   451	    return sum(item[key] for item in items) / len(items)

exec
/usr/bin/zsh -lc 'nl -ba src/milestone1/llm.py src/milestone1/llm_engine.py src/milestone1/vlm.py src/milestone1/analysis.py' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
     1	"""Ollama LLM client for semantic saturation scoring."""
     2	
     3	import json
     4	import re
     5	from typing import Any
     6	
     7	import requests
     8	
     9	SATURATION_PROMPT = """You are a scene graph analysis expert. You are given a sequence of scene graphs extracted from the same image at increasing resolution levels.
    10	
    11	Your task: Judge whether the scene graph has reached "semantic saturation" — i.e., whether further increasing image resolution is unlikely to reveal meaningful new objects or relationships.
    12	
    13	INPUT: A JSON array of scene graphs at consecutive resolution levels. Each entry has:
    14	- "level": resolution level (higher = better resolution)
    15	- "objects": detected objects with labels and confidence scores
    16	- "relationships": detected relationships (subject -> predicate -> object) with confidence scores
    17	
    18	CRITERIA for saturation:
    19	1. The set of detected object categories has stabilized (few or no new categories appearing)
    20	2. The set of detected relationships has stabilized
    21	3. Confidence scores are not significantly changing
    22	4. The scene graph structure is converging
    23	
    24	OUTPUT: Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
    25	{
    26	  "saturation_score": <float 0.0-1.0, where 1.0 = fully saturated>,
    27	  "stop_decision": <boolean, true if saturation_score >= 0.8>,
    28	  "reason": "<brief explanation of your judgment>"
    29	}
    30	
    31	Scene graph sequence:
    32	"""
    33	
    34	
    35	def _parse_llm_json(text: str) -> dict[str, Any]:
    36	    """Parse JSON from LLM response, handling markdown fences and noise."""
    37	    # Strip markdown code fences
    38	    text = re.sub(r"```(?:json)?\s*", "", text)
    39	    text = re.sub(r"```\s*", "", text)
    40	
    41	    # Try to extract first JSON object
    42	    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    43	    if match:
    44	        try:
    45	            return json.loads(match.group())
    46	        except json.JSONDecodeError:
    47	            pass
    48	
    49	    # Try parsing the whole text
    50	    try:
    51	        return json.loads(text.strip())
    52	    except json.JSONDecodeError:
    53	        return {
    54	            "saturation_score": 0.0,
    55	            "stop_decision": False,
    56	            "reason": f"parse-error: {text[:200]}",
    57	        }
    58	
    59	
    60	def _derive_chat_endpoint(generate_endpoint: str) -> str:
    61	    """Convert generate endpoint to chat endpoint."""
    62	    return generate_endpoint.replace("/api/generate", "/api/chat")
    63	
    64	
    65	def call_ollama(
    66	    prompt: str,
    67	    model: str,
    68	    endpoint: str,
    69	    timeout: int = 120,
    70	) -> str:
    71	    """Call Ollama chat API with thinking disabled and return the response text."""
    72	    chat_endpoint = _derive_chat_endpoint(endpoint)
    73	    payload = {
    74	        "model": model,
    75	        "messages": [
    76	            {"role": "user", "content": prompt},
    77	        ],
    78	        "stream": False,
    79	        "think": False,
    80	        "options": {
    81	            "temperature": 0.3,
    82	            "num_predict": 1024,
    83	            # Avoid ollama's default num_ctx=4096 truncating long prompts to
    84	            # 4095 tokens (leaves no room to generate -> 1-token "Based" stub
    85	            # that fails JSON parsing). Fits our largest ~9.2k-token prompts.
    86	            "num_ctx": 12288,
    87	        },
    88	    }
    89	    resp = requests.post(chat_endpoint, json=payload, timeout=timeout)
    90	    resp.raise_for_status()
    91	    return resp.json()["message"]["content"]
    92	
    93	
    94	def smoke_test_ollama(model: str, endpoint: str, timeout: int = 30) -> bool:
    95	    """Quick check if Ollama is reachable and model is loaded."""
    96	    try:
    97	        result = call_ollama("Respond with just: OK", model, endpoint, timeout=timeout)
    98	        return len(result.strip()) > 0
    99	    except Exception as e:
   100	        print(f"Ollama smoke test failed: {e}")
   101	        return False
   102	
   103	
   104	def score_saturation(
   105	    sequence_json: list[dict[str, Any]],
   106	    model: str,
   107	    endpoint: str,
   108	    timeout: int = 120,
   109	) -> dict[str, Any]:
   110	    """Score semantic saturation for a scene graph sequence.
   111	
   112	    Returns {saturation_score, stop_decision, reason}.
   113	    """
   114	    prompt = SATURATION_PROMPT + json.dumps(sequence_json, indent=2)
   115	
   116	    try:
   117	        raw_response = call_ollama(prompt, model, endpoint, timeout=timeout)
   118	        result = _parse_llm_json(raw_response)
   119	
   120	        # Ensure required fields
   121	        if "saturation_score" not in result:
   122	            result["saturation_score"] = 0.0
   123	        if "stop_decision" not in result:
   124	            result["stop_decision"] = result["saturation_score"] >= 0.8
   125	        if "reason" not in result:
   126	            result["reason"] = "no reason provided"
   127	
   128	        # Clamp score to [0, 1]
   129	        result["saturation_score"] = max(
   130	            0.0, min(1.0, float(result["saturation_score"]))
   131	        )
   132	        result["stop_decision"] = bool(result["stop_decision"])
   133	
   134	        return result
   135	
   136	    except Exception as e:
   137	        return {
   138	            "saturation_score": 0.0,
   139	            "stop_decision": False,
   140	            "reason": f"error: {str(e)[:200]}",
   141	        }
   142	"""Parameterized LLM scoring engine for ablation studies.
   143	
   144	Builds scene-graph sequences with configurable features (relation_on,
   145	confidence_on, history_length) and scores them via ollama /api/chat.
   146	"""
   147	
   148	import json
   149	from pathlib import Path
   150	from typing import Any
   151	
   152	from milestone1.llm import _parse_llm_json
   153	
   154	SATURATION_PROMPT_TEMPLATE = """You are a scene graph analysis expert. You are given a sequence of scene graphs extracted from the same image at increasing resolution levels.
   155	
   156	Your task: Judge whether the scene graph has reached "semantic saturation" — i.e., whether further increasing image resolution is unlikely to reveal meaningful new objects or relationships.
   157	
   158	INPUT: A JSON array of scene graphs at consecutive resolution levels. Each entry has:
   159	- "level": resolution level (higher = better resolution)
   160	- "objects": detected objects with labels{conf_note_objects}
   161	{rel_note}
   162	CRITERIA for saturation:
   163	1. The set of detected object categories has stabilized (few or no new categories appearing)
   164	{rel_criteria}3. The scene graph structure is converging
   165	
   166	OUTPUT: Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
   167	{{
   168	  "saturation_score": <float 0.0-1.0, where 1.0 = fully saturated>,
   169	  "stop_decision": <boolean, true if saturation_score >= 0.8>,
   170	  "reason": "<brief explanation of your judgment>"
   171	}}
   172	
   173	Scene graph sequence:
   174	"""
   175	
   176	
   177	def build_prompt(relation_on: bool, confidence_on: bool) -> str:
   178	    """Build the saturation prompt based on ablation flags."""
   179	    conf_note_objects = " and confidence scores" if confidence_on else ""
   180	
   181	    if relation_on:
   182	        if confidence_on:
   183	            rel_note = '- "relationships": detected relationships (subject -> predicate -> object) with confidence scores'
   184	        else:
   185	            rel_note = '- "relationships": detected relationships (subject -> predicate -> object)'
   186	        rel_criteria = "2. The set of detected relationships has stabilized\n"
   187	    else:
   188	        rel_note = "(No relationship information is provided in this evaluation.)"
   189	        rel_criteria = ""
   190	
   191	    if confidence_on and relation_on:
   192	        # Add confidence stability criterion
   193	        rel_criteria += "3. Confidence scores are not significantly changing\n"
   194	        # Renumber the last criterion
   195	        return SATURATION_PROMPT_TEMPLATE.format(
   196	            conf_note_objects=conf_note_objects,
   197	            rel_note=rel_note,
   198	            rel_criteria=rel_criteria.rstrip("\n") + "\n",
   199	        )
   200	
   201	    return SATURATION_PROMPT_TEMPLATE.format(
   202	        conf_note_objects=conf_note_objects,
   203	        rel_note=rel_note,
   204	        rel_criteria=rel_criteria,
   205	    )
   206	
   207	
   208	def build_sequence(
   209	    image_id: int,
   210	    current_level: int,
   211	    history_length: int,
   212	    pred_root: Path,
   213	    relation_on: bool = True,
   214	    confidence_on: bool = True,
   215	    input_obj_threshold: float = 0.3,
   216	    input_rel_threshold: float = 0.005,
   217	    input_top_k_objects: int = 20,
   218	    input_top_k_relationships: int = 30,
   219	) -> list[dict[str, Any]]:
   220	    """Build a scene graph sequence for LLM input with configurable features."""
   221	    start = max(1, current_level - history_length + 1)
   222	    sequence = []
   223	
   224	    for lv in range(start, current_level + 1):
   225	        level_name = f"level_{lv:02d}"
   226	        pred_path = pred_root / str(image_id) / f"{level_name}.json"
   227	
   228	        if not pred_path.exists():
   229	            continue
   230	
   231	        with open(pred_path) as f:
   232	            pred = json.load(f)
   233	
   234	        # Filter objects
   235	        objects = [
   236	            o for o in pred["objects"] if o["confidence"] >= input_obj_threshold
   237	        ][:input_top_k_objects]
   238	
   239	        # Build object entries
   240	        if confidence_on:
   241	            obj_entries = [
   242	                {
   243	                    "object_id": o["object_id"],
   244	                    "label": o["label"],
   245	                    "confidence": o["confidence"],
   246	                }
   247	                for o in objects
   248	            ]
   249	        else:
   250	            obj_entries = [
   251	                {
   252	                    "object_id": o["object_id"],
   253	                    "label": o["label"],
   254	                }
   255	                for o in objects
   256	            ]
   257	
   258	        entry: dict[str, Any] = {
   259	            "level": lv,
   260	            "objects": obj_entries,
   261	        }
   262	
   263	        # Add relationships if enabled
   264	        if relation_on:
   265	            relationships = [
   266	                r
   267	                for r in pred["relationships"]
   268	                if r["confidence"] >= input_rel_threshold
   269	            ][:input_top_k_relationships]
   270	
   271	            if confidence_on:
   272	                rel_entries = [
   273	                    {
   274	                        "subject_id": r["subject_id"],
   275	                        "subject_label": r["subject_label"],
   276	                        "predicate": r["predicate"],
   277	                        "object_id": r["object_id"],
   278	                        "object_label": r["object_label"],
   279	                        "confidence": r["confidence"],
   280	                    }
   281	                    for r in relationships
   282	                ]
   283	            else:
   284	                rel_entries = [
   285	                    {
   286	                        "subject_id": r["subject_id"],
   287	                        "subject_label": r["subject_label"],
   288	                        "predicate": r["predicate"],
   289	                        "object_id": r["object_id"],
   290	                        "object_label": r["object_label"],
   291	                    }
   292	                    for r in relationships
   293	                ]
   294	            entry["relationships"] = rel_entries
   295	
   296	        sequence.append(entry)
   297	
   298	    return sequence
   299	
   300	
   301	def make_config_slug(
   302	    model: str,
   303	    history_length: int,
   304	    relation_on: bool,
   305	    confidence_on: bool,
   306	) -> str:
   307	    """Create a filesystem-safe slug encoding the ablation config."""
   308	    model_tag = model.replace(":", "-").replace("/", "-")
   309	    rel_tag = "rel_on" if relation_on else "rel_off"
   310	    conf_tag = "conf_on" if confidence_on else "conf_off"
   311	    return f"{model_tag}_h{history_length}_{rel_tag}_{conf_tag}"
   312	
   313	
   314	def _do_ollama_call(
   315	    full_prompt: str,
   316	    model: str,
   317	    chat_endpoint: str,
   318	    socket_timeout: int,
   319	) -> str:
   320	    """Make the actual ollama API call (runs in a thread for wall-clock timeout)."""
   321	    import requests as req
   322	
   323	    payload = {
   324	        "model": model,
   325	        "messages": [
   326	            {"role": "user", "content": full_prompt},
   327	        ],
   328	        "stream": False,
   329	        "think": False,
   330	        "options": {
   331	            "temperature": 0.3,
   332	            "num_predict": 1024,
   333	            # Long history-5 / top-20-30 prompts reach ~9.2k tokens; ollama's
   334	            # default num_ctx=4096 truncates the prompt to 4095 and leaves zero
   335	            # room to generate, producing a 1-token stub ("Based") that fails to
   336	            # parse. Size the context to fit the largest prompt + generation.
   337	            "num_ctx": 12288,
   338	        },
   339	    }
   340	    resp = req.post(chat_endpoint, json=payload, timeout=socket_timeout)
   341	    resp.raise_for_status()
   342	    return resp.json()["message"]["content"]
   343	
   344	
   345	def score_with_config(
   346	    sequence_json: list[dict[str, Any]],
   347	    prompt: str,
   348	    model: str,
   349	    chat_endpoint: str,
   350	    timeout: int = 120,
   351	) -> dict[str, Any]:
   352	    """Score saturation for a sequence using the parameterized prompt.
   353	
   354	    Uses ollama /api/chat with think=false, stream=false.
   355	    Enforces a hard wall-clock timeout via ThreadPoolExecutor to handle
   356	    cases where ollama keeps the connection alive with slow token generation.
   357	    Returns {saturation_score, stop_decision, reason}.
   358	    """
   359	    from concurrent.futures import ThreadPoolExecutor
   360	    from concurrent.futures import TimeoutError as FuturesTimeout
   361	
   362	    full_prompt = prompt + json.dumps(sequence_json, indent=2)
   363	
   364	    try:
   365	        with ThreadPoolExecutor(max_workers=1) as pool:
   366	            future = pool.submit(
   367	                _do_ollama_call, full_prompt, model, chat_endpoint, timeout
   368	            )
   369	            # Hard wall-clock timeout: timeout + 30s grace
   370	            raw = future.result(timeout=timeout + 30)
   371	
   372	        result = _parse_llm_json(raw)
   373	
   374	        # Ensure required fields
   375	        if "saturation_score" not in result:
   376	            result["saturation_score"] = 0.0
   377	        if "stop_decision" not in result:
   378	            result["stop_decision"] = result["saturation_score"] >= 0.8
   379	        if "reason" not in result:
   380	            result["reason"] = "no reason provided"
   381	
   382	        result["saturation_score"] = max(
   383	            0.0, min(1.0, float(result["saturation_score"]))
   384	        )
   385	        result["stop_decision"] = bool(result["stop_decision"])
   386	        return result
   387	
   388	    except FuturesTimeout:
   389	        return {
   390	            "saturation_score": 0.0,
   391	            "stop_decision": False,
   392	            "reason": "error: wall-clock timeout exceeded",
   393	        }
   394	    except Exception as e:
   395	        return {
   396	            "saturation_score": 0.0,
   397	            "stop_decision": False,
   398	            "reason": f"error: {str(e)[:200]}",
   399	        }
   400	"""VLM (Vision-Language Model) client for saturation scoring via ollama /api/chat with images."""
   401	
   402	import base64
   403	import io
   404	import json
   405	from typing import Any
   406	
   407	import requests
   408	from PIL import Image
   409	
   410	from milestone1.llm import _parse_llm_json
   411	
   412	
   413	VLM_SINGLE_PROMPT = """You are a visual scene analysis expert. You are shown a single image captured at a certain observation quality level (resolution).
   414	
   415	Your task: Judge whether the scene has reached "semantic saturation" — i.e., whether getting a sharper or closer view of this scene is unlikely to reveal important new objects or spatial relationships that are not already visible.
   416	
   417	Consider:
   418	1. Can you clearly identify the main objects and their arrangement?
   419	2. Are there blurry or indistinct regions that might contain important objects?
   420	3. Is the image detailed enough to understand the key relationships between objects?
   421	4. Would higher resolution likely reveal new semantic content (new object types, new relationships)?
   422	
   423	OUTPUT: Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
   424	{
   425	  "saturation_score": <float 0.0-1.0, where 1.0 = fully saturated, no benefit from higher resolution>,
   426	  "stop_decision": <boolean, true if saturation_score >= 0.8>,
   427	  "reason": "<brief explanation of your judgment>"
   428	}
   429	"""
   430	
   431	VLM_SEQUENCE_PROMPT = """You are a visual scene analysis expert. You are shown a sequence of images of the SAME scene at gradually improving observation quality (resolution). The images go from lower to higher resolution (oldest to newest).
   432	
   433	Your task: Judge whether the scene has reached "semantic saturation" — i.e., whether further improving image quality is unlikely to reveal meaningful new objects or spatial relationships.
   434	
   435	Consider:
   436	1. Comparing consecutive images: are new objects or details becoming visible with each quality improvement?
   437	2. Has the rate of new discoveries slowed down or stopped?
   438	3. Are the most recent images showing essentially the same semantic content?
   439	4. Would even higher resolution likely reveal important new content?
   440	
   441	OUTPUT: Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
   442	{
   443	  "saturation_score": <float 0.0-1.0, where 1.0 = fully saturated, no benefit from higher resolution>,
   444	  "stop_decision": <boolean, true if saturation_score >= 0.8>,
   445	  "reason": "<brief explanation of your judgment>"
   446	}
   447	"""
   448	
   449	
   450	def _load_and_downscale_image(
   451	    image_path: str, max_side: int = 512
   452	) -> str:
   453	    """Load an image, downscale so longest side <= max_side, return base64 JPEG."""
   454	    img = Image.open(image_path)
   455	    # Downscale if needed
   456	    w, h = img.size
   457	    if max(w, h) > max_side:
   458	        scale = max_side / max(w, h)
   459	        new_w = int(w * scale)
   460	        new_h = int(h * scale)
   461	        img = img.resize((new_w, new_h), Image.LANCZOS)
   462	    # Convert to RGB if needed (e.g., RGBA, P mode)
   463	    if img.mode != "RGB":
   464	        img = img.convert("RGB")
   465	    # Encode as JPEG base64
   466	    buf = io.BytesIO()
   467	    img.save(buf, format="JPEG", quality=85)
   468	    return base64.b64encode(buf.getvalue()).decode("utf-8")
   469	
   470	
   471	def _do_vlm_request(
   472	    payload: dict,
   473	    chat_endpoint: str,
   474	    socket_timeout: int,
   475	) -> str:
   476	    """Make the actual VLM API call (runs in a thread for wall-clock timeout)."""
   477	    resp = requests.post(chat_endpoint, json=payload, timeout=socket_timeout)
   478	    resp.raise_for_status()
   479	    return resp.json()["message"]["content"]
   480	
   481	
   482	def call_vlm(
   483	    prompt: str,
   484	    images_b64: list[str],
   485	    model: str,
   486	    chat_endpoint: str,
   487	    timeout: int = 300,
   488	) -> str:
   489	    """Call Ollama chat API with images and return the response text.
   490	
   491	    Uses a ThreadPoolExecutor with wall-clock timeout to handle cases
   492	    where ollama keeps the connection alive with slow token generation.
   493	    """
   494	    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
   495	
   496	    payload = {
   497	        "model": model,
   498	        "messages": [
   499	            {
   500	                "role": "user",
   501	                "content": prompt,
   502	                "images": images_b64,
   503	            },
   504	        ],
   505	        "stream": False,
   506	        "think": False,
   507	        "options": {
   508	            "temperature": 0.3,
   509	            "num_predict": 1024,
   510	        },
   511	    }
   512	    with ThreadPoolExecutor(max_workers=1) as pool:
   513	        future = pool.submit(_do_vlm_request, payload, chat_endpoint, timeout)
   514	        try:
   515	            return future.result(timeout=timeout + 30)
   516	        except FuturesTimeout:
   517	            raise TimeoutError(f"VLM call exceeded wall-clock timeout of {timeout + 30}s")
   518	
   519	
   520	def smoke_test_vlm(
   521	    model: str, chat_endpoint: str, timeout: int = 60
   522	) -> bool:
   523	    """Smoke test: send a tiny 8x8 red image and check we get a response."""
   524	    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
   525	    buf = io.BytesIO()
   526	    img.save(buf, format="JPEG", quality=85)
   527	    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
   528	
   529	    try:
   530	        result = call_vlm(
   531	            "What color is this image? Respond with just the color name.",
   532	            [b64],
   533	            model,
   534	            chat_endpoint,
   535	            timeout=timeout,
   536	        )
   537	        return len(result.strip()) > 0
   538	    except Exception as e:
   539	        print(f"VLM smoke test failed for {model}: {e}")
   540	        return False
   541	
   542	
   543	def score_vlm_single(
   544	    image_path: str,
   545	    model: str,
   546	    chat_endpoint: str,
   547	    max_image_side: int = 512,
   548	    timeout: int = 300,
   549	) -> dict[str, Any]:
   550	    """Score saturation using a single image (VLM single mode)."""
   551	    try:
   552	        b64 = _load_and_downscale_image(image_path, max_image_side)
   553	        raw = call_vlm(VLM_SINGLE_PROMPT, [b64], model, chat_endpoint, timeout)
   554	        result = _parse_llm_json(raw)
   555	        return _normalize_result(result)
   556	    except Exception as e:
   557	        return {
   558	            "saturation_score": 0.0,
   559	            "stop_decision": False,
   560	            "reason": f"error: {str(e)[:200]}",
   561	        }
   562	
   563	
   564	def score_vlm_sequence(
   565	    image_paths: list[str],
   566	    model: str,
   567	    chat_endpoint: str,
   568	    max_image_side: int = 512,
   569	    timeout: int = 300,
   570	) -> dict[str, Any]:
   571	    """Score saturation using a sequence of images (VLM sequence mode)."""
   572	    try:
   573	        b64_list = [
   574	            _load_and_downscale_image(p, max_image_side) for p in image_paths
   575	        ]
   576	        raw = call_vlm(
   577	            VLM_SEQUENCE_PROMPT, b64_list, model, chat_endpoint, timeout
   578	        )
   579	        result = _parse_llm_json(raw)
   580	        return _normalize_result(result)
   581	    except Exception as e:
   582	        return {
   583	            "saturation_score": 0.0,
   584	            "stop_decision": False,
   585	            "reason": f"error: {str(e)[:200]}",
   586	        }
   587	
   588	
   589	def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
   590	    """Ensure required fields exist and values are in range."""
   591	    if "saturation_score" not in result:
   592	        result["saturation_score"] = 0.0
   593	    if "stop_decision" not in result:
   594	        result["stop_decision"] = result["saturation_score"] >= 0.8
   595	    if "reason" not in result:
   596	        result["reason"] = "no reason provided"
   597	    result["saturation_score"] = max(0.0, min(1.0, float(result["saturation_score"])))
   598	    result["stop_decision"] = bool(result["stop_decision"])
   599	    return result
   600	"""Error and saturation analysis utilities for SGG-saturation PoC."""
   601	
   602	import json
   603	from pathlib import Path
   604	from typing import Any
   605	
   606	import pandas as pd
   607	from PIL import Image
   608	
   609	from milestone1.config import resolve
   610	from milestone1.metrics import _to_label_set, _to_triplet_set
   611	
   612	
   613	# --------------------------------------------------------------------------- #
   614	# Data loading helpers
   615	# --------------------------------------------------------------------------- #
   616	
   617	
   618	def load_gt(gt_root: Path, image_id: int) -> dict[str, Any]:
   619	    """Load a GT scene graph JSON."""
   620	    with open(gt_root / f"{image_id}.json") as f:
   621	        return json.load(f)
   622	
   623	
   624	def load_pred(pred_root: Path, image_id: int, level: int) -> dict[str, Any]:
   625	    """Load an EGTR prediction JSON for a given level."""
   626	    with open(pred_root / str(image_id) / f"level_{level:02d}.json") as f:
   627	        return json.load(f)
   628	
   629	
   630	def load_stop_levels(
   631	    metrics_root: Path,
   632	) -> dict[str, dict[int, int]]:
   633	    """Load stop levels for oracle@0.5, baseline@0.4, and llm.
   634	
   635	    Returns {method_name: {image_id: stop_level}}.
   636	    """
   637	    stops: dict[str, dict[int, int]] = {}
   638	
   639	    # Oracle stop (use oracle_0.5 column = main_oracle_threshold)
   640	    oracle_path = metrics_root / "oracle_stop.csv"
   641	    if oracle_path.exists():
   642	        df = pd.read_csv(oracle_path)
   643	        stops["oracle"] = dict(zip(df["image_id"], df["oracle_0.5"]))
   644	
   645	    # Baseline stop (use baseline_0.4 column)
   646	    baseline_path = metrics_root / "baseline_stop.csv"
   647	    if baseline_path.exists():
   648	        df = pd.read_csv(baseline_path)
   649	        stops["baseline"] = dict(zip(df["image_id"], df["baseline_0.4"]))
   650	
   651	    # LLM stop
   652	    llm_path = metrics_root / "llm_stop.csv"
   653	    if llm_path.exists():
   654	        df = pd.read_csv(llm_path)
   655	        stops["llm"] = dict(zip(df["image_id"], df["llm_stop"]))
   656	
   657	    return stops
   658	
   659	
   660	# --------------------------------------------------------------------------- #
   661	# Analysis 1: What was missing at the stop level
   662	# --------------------------------------------------------------------------- #
   663	
   664	
   665	def compute_missing_at_stop(
   666	    gt: dict[str, Any],
   667	    pred: dict[str, Any],
   668	) -> dict[str, Any]:
   669	    """Compare GT vs prediction at a specific level.
   670	
   671	    Returns counts and lists of missing/extra nodes and triplets.
   672	    """
   673	    gt_nodes = _to_label_set(gt["objects"])
   674	    pred_nodes = _to_label_set(pred["objects"])
   675	    gt_trips = _to_triplet_set(gt["relationships"])
   676	    pred_trips = _to_triplet_set(pred["relationships"])
   677	
   678	    missing_nodes = gt_nodes - pred_nodes
   679	    extra_nodes = pred_nodes - gt_nodes
   680	    missing_trips = gt_trips - pred_trips
   681	    extra_trips = pred_trips - gt_trips
   682	
   683	    return {
   684	        "gt_node_count": len(gt_nodes),
   685	        "pred_node_count": len(pred_nodes),
   686	        "missing_node_count": len(missing_nodes),
   687	        "extra_node_count": len(extra_nodes),
   688	        "missing_nodes": ";".join(sorted(missing_nodes)),
   689	        "extra_nodes": ";".join(sorted(extra_nodes)),
   690	        "gt_triplet_count": len(gt_trips),
   691	        "pred_triplet_count": len(pred_trips),
   692	        "missing_triplet_count": len(missing_trips),
   693	        "extra_triplet_count": len(extra_trips),
   694	        "missing_triplets": ";".join(
   695	            f"{s}-{p}-{o}" for s, p, o in sorted(missing_trips)
   696	        ),
   697	        "extra_triplets": ";".join(
   698	            f"{s}-{p}-{o}" for s, p, o in sorted(extra_trips)
   699	        ),
   700	    }
   701	
   702	
   703	# --------------------------------------------------------------------------- #
   704	# Analysis 2 & 3: Per-category / per-predicate recall by level
   705	# --------------------------------------------------------------------------- #
   706	
   707	
   708	def compute_category_recall_by_level(
   709	    image_ids: list[int],
   710	    gt_root: Path,
   711	    pred_root: Path,
   712	    num_levels: int,
   713	) -> pd.DataFrame:
   714	    """Compute per-object-category recall at each level across all images.
   715	
   716	    For each (category, level): recall = fraction of images that contain
   717	    that GT category where it was detected (i.e. in the predicted label set).
   718	    """
   719	    rows: list[dict[str, Any]] = []
   720	
   721	    for img_id in image_ids:
   722	        gt = load_gt(gt_root, img_id)
   723	        gt_cats = _to_label_set(gt["objects"])
   724	
   725	        for level in range(1, num_levels + 1):
   726	            pred = load_pred(pred_root, img_id, level)
   727	            pred_cats = _to_label_set(pred["objects"])
   728	
   729	            for cat in gt_cats:
   730	                rows.append(
   731	                    {
   732	                        "image_id": img_id,
   733	                        "level": level,
   734	                        "category": cat,
   735	                        "detected": 1 if cat in pred_cats else 0,
   736	                    }
   737	                )
   738	
   739	    df = pd.DataFrame(rows)
   740	    # Aggregate: per (category, level), fraction detected
   741	    agg = (
   742	        df.groupby(["category", "level"])["detected"]
   743	        .mean()
   744	        .reset_index()
   745	        .rename(columns={"detected": "recall"})
   746	    )
   747	    return agg
   748	
   749	
   750	def compute_predicate_recall_by_level(
   751	    image_ids: list[int],
   752	    gt_root: Path,
   753	    pred_root: Path,
   754	    num_levels: int,
   755	) -> pd.DataFrame:
   756	    """Compute per-predicate recall at each level across all images.
   757	
   758	    For each GT triplet, we check if that exact (subject, predicate, object)
   759	    is present in the predicted triplet set. Then aggregate by predicate.
   760	    """
   761	    rows: list[dict[str, Any]] = []
   762	
   763	    for img_id in image_ids:
   764	        gt = load_gt(gt_root, img_id)
   765	        gt_trips = _to_triplet_set(gt["relationships"])
   766	
   767	        for level in range(1, num_levels + 1):
   768	            pred = load_pred(pred_root, img_id, level)
   769	            pred_trips = _to_triplet_set(pred["relationships"])
   770	
   771	            for s, p, o in gt_trips:
   772	                rows.append(
   773	                    {
   774	                        "image_id": img_id,
   775	                        "level": level,
   776	                        "predicate": p,
   777	                        "triplet": f"{s}-{p}-{o}",
   778	                        "detected": 1 if (s, p, o) in pred_trips else 0,
   779	                    }
   780	                )
   781	
   782	    df = pd.DataFrame(rows)
   783	    agg = (
   784	        df.groupby(["predicate", "level"])["detected"]
   785	        .mean()
   786	        .reset_index()
   787	        .rename(columns={"detected": "recall"})
   788	    )
   789	    return agg
   790	
   791	
   792	def identify_hardest(
   793	    recall_df: pd.DataFrame,
   794	    entity_col: str,
   795	    num_levels: int,
   796	    top_k: int = 10,
   797	) -> list[str]:
   798	    """Identify the hardest categories/predicates by final-level recall.
   799	
   800	    Lower recall at the final level = harder.
   801	    """
   802	    final = recall_df[recall_df["level"] == num_levels].copy()
   803	    final = final.sort_values("recall")
   804	    return final[entity_col].head(top_k).tolist()
   805	
   806	
   807	# --------------------------------------------------------------------------- #
   808	# Analysis 4: Image feature extraction
   809	# --------------------------------------------------------------------------- #
   810	
   811	
   812	def compute_image_features(
   813	    image_id: int,
   814	    gt: dict[str, Any],
   815	    vg_images_dir: Path,
   816	    all_metrics_df: pd.DataFrame,
   817	    oracle_stop: int,
   818	) -> dict[str, Any]:
   819	    """Compute per-image features for difficulty characterization."""
   820	    # Image dimensions
   821	    img_path = vg_images_dir / f"{image_id}.jpg"
   822	    if img_path.exists():
   823	        with Image.open(img_path) as im:
   824	            img_w, img_h = im.size
   825	    else:
   826	        # Fallback: use bbox extents from GT
   827	        all_x2 = [o["bbox"][2] for o in gt["objects"]]
   828	        all_y2 = [o["bbox"][3] for o in gt["objects"]]
   829	        img_w = max(all_x2) if all_x2 else 500
   830	        img_h = max(all_y2) if all_y2 else 500
   831	
   832	    img_area = img_w * img_h
   833	
   834	    # GT object areas (from bbox)
   835	    obj_areas = []
   836	    for obj in gt["objects"]:
   837	        x1, y1, x2, y2 = obj["bbox"]
   838	        obj_areas.append(max(0, (x2 - x1)) * max(0, (y2 - y1)))
   839	
   840	    gt_node_cats = _to_label_set(gt["objects"])
   841	    gt_trip_set = _to_triplet_set(gt["relationships"])
   842	
   843	    mean_obj_area_frac = (
   844	        (sum(obj_areas) / len(obj_areas) / img_area) if obj_areas else 0.0
   845	    )
   846	    small_threshold = 0.01 * img_area
   847	    frac_small = (
   848	        sum(1 for a in obj_areas if a < small_threshold) / len(obj_areas)
   849	        if obj_areas
   850	        else 0.0
   851	    )
   852	
   853	    # Final-level metrics
   854	    img_metrics = all_metrics_df[all_metrics_df["image_id"] == image_id]
   855	    final_row = img_metrics[img_metrics["level"] == 20]
   856	    final_sg = float(final_row["sg_score"].values[0]) if len(final_row) > 0 else 0.0
   857	    final_node_r = (
   858	        float(final_row["node_recall"].values[0]) if len(final_row) > 0 else 0.0
   859	    )
   860	    final_trip_r = (
   861	        float(final_row["triplet_recall"].values[0]) if len(final_row) > 0 else 0.0
   862	    )
   863	
   864	    return {
   865	        "image_id": image_id,
   866	        "img_width": img_w,
   867	        "img_height": img_h,
   868	        "img_area": img_area,
   869	        "num_gt_objects": len(gt["objects"]),
   870	        "num_gt_object_categories": len(gt_node_cats),
   871	        "num_gt_relationships": len(gt["relationships"]),
   872	        "num_gt_unique_triplets": len(gt_trip_set),
   873	        "mean_obj_area_frac": round(mean_obj_area_frac, 4),
   874	        "frac_small_objects": round(frac_small, 4),
   875	        "final_sg_score": final_sg,
   876	        "final_node_recall": final_node_r,
   877	        "final_triplet_recall": final_trip_r,
   878	        "oracle_stop_level": oracle_stop,
   879	    }
   880	
   881	
   882	# --------------------------------------------------------------------------- #
   883	# Analysis 5: Node vs Relation saturation gap
   884	# --------------------------------------------------------------------------- #
   885	
   886	
   887	def compute_saturation_gap(
   888	    image_id: int,
   889	    all_metrics_df: pd.DataFrame,
   890	    threshold_frac: float = 0.9,
   891	) -> dict[str, Any]:
   892	    """Compute the gap between node and triplet saturation levels.
   893	
   894	    Saturation = level where recall first reaches threshold_frac * max(recall).
   895	    """
   896	    img_df = all_metrics_df[all_metrics_df["image_id"] == image_id].sort_values(
   897	        "level"
   898	    )
   899	
   900	    max_node_r = img_df["node_recall"].max()
   901	    max_trip_r = img_df["triplet_recall"].max()
   902	
   903	    node_sat_level = 20
   904	    trip_sat_level = 20
   905	
   906	    if max_node_r > 0:
   907	        target = threshold_frac * max_node_r
   908	        hits = img_df[img_df["node_recall"] >= target]
   909	        if len(hits) > 0:
   910	            node_sat_level = int(hits["level"].min())
   911	
   912	    if max_trip_r > 0:
   913	        target = threshold_frac * max_trip_r
   914	        hits = img_df[img_df["triplet_recall"] >= target]
   915	        if len(hits) > 0:
   916	            trip_sat_level = int(hits["level"].min())
   917	
   918	    return {
   919	        "image_id": image_id,
   920	        "max_node_recall": round(max_node_r, 4),
   921	        "max_triplet_recall": round(max_trip_r, 4),
   922	        "node_90pct_level": node_sat_level,
   923	        "triplet_90pct_level": trip_sat_level,
   924	        "gap": trip_sat_level - node_sat_level,
   925	    }

**Reviewing file outputs**

It seems there’s an issue with using `nl` on multiple files where it loses the boundaries between them. When I use it on consecutive files, it doesn’t include headers, starting with the first lines from the config and then moving to vocabulary files. I really need to pinpoint the exact file and line, possibly using `nl -ba`. Also, I think I'll need scripts and configuration files for clarity, while maybe inspecting the full relevant files.
codex
The first line-number pass concatenated package files, so I’m doing separate `nl -ba` reads for the review-critical files plus all requested scripts/config. That will avoid ambiguous citations.
exec
/usr/bin/zsh -lc 'nl -ba src/milestone1/metrics.py' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
     1	"""Scene graph evaluation metrics with category-set semantics."""
     2	
     3	from typing import Any
     4	
     5	
     6	def _to_label_set(objects: list[dict[str, Any]]) -> set[str]:
     7	    """Extract unique label set from objects."""
     8	    return {obj["label"] for obj in objects}
     9	
    10	
    11	def _to_triplet_set(
    12	    relationships: list[dict[str, Any]],
    13	) -> set[tuple[str, str, str]]:
    14	    """Extract unique triplet set from relationships."""
    15	    return {
    16	        (rel["subject_label"], rel["predicate"], rel["object_label"])
    17	        for rel in relationships
    18	    }
    19	
    20	
    21	def node_precision_recall_f1(
    22	    pred_objects: list[dict[str, Any]],
    23	    gt_objects: list[dict[str, Any]],
    24	) -> tuple[float, float, float]:
    25	    """Compute node (object label) precision, recall, F1.
    26	
    27	    Uses category-set semantics: dedup to sets of labels.
    28	    """
    29	    pred_set = _to_label_set(pred_objects)
    30	    gt_set = _to_label_set(gt_objects)
    31	
    32	    if not pred_set and not gt_set:
    33	        return 1.0, 1.0, 1.0
    34	    if not pred_set:
    35	        return 0.0, 0.0, 0.0
    36	    if not gt_set:
    37	        return 0.0, 0.0, 0.0
    38	
    39	    tp = len(pred_set & gt_set)
    40	    precision = tp / len(pred_set) if pred_set else 0.0
    41	    recall = tp / len(gt_set) if gt_set else 0.0
    42	    f1 = (
    43	        2 * precision * recall / (precision + recall)
    44	        if (precision + recall) > 0
    45	        else 0.0
    46	    )
    47	    return precision, recall, f1
    48	
    49	
    50	def triplet_precision_recall_f1(
    51	    pred_relationships: list[dict[str, Any]],
    52	    gt_relationships: list[dict[str, Any]],
    53	) -> tuple[float, float, float]:
    54	    """Compute triplet (subject, predicate, object) precision, recall, F1.
    55	
    56	    Uses category-set semantics: dedup to sets of triplets.
    57	    """
    58	    pred_set = _to_triplet_set(pred_relationships)
    59	    gt_set = _to_triplet_set(gt_relationships)
    60	
    61	    if not pred_set and not gt_set:
    62	        return 1.0, 1.0, 1.0
    63	    if not pred_set:
    64	        return 0.0, 0.0, 0.0
    65	    if not gt_set:
    66	        return 0.0, 0.0, 0.0
    67	
    68	    tp = len(pred_set & gt_set)
    69	    precision = tp / len(pred_set) if pred_set else 0.0
    70	    recall = tp / len(gt_set) if gt_set else 0.0
    71	    f1 = (
    72	        2 * precision * recall / (precision + recall)
    73	        if (precision + recall) > 0
    74	        else 0.0
    75	    )
    76	    return precision, recall, f1
    77	
    78	
    79	def sg_score(
    80	    node_recall: float,
    81	    triplet_recall: float,
    82	    node_weight: float = 0.5,
    83	    relation_weight: float = 0.5,
    84	) -> float:
    85	    """Compute weighted scene graph score."""
    86	    return node_weight * node_recall + relation_weight * triplet_recall
    87	
    88	
    89	def mean_confidence(items: list[dict[str, Any]], key: str = "confidence") -> float:
    90	    """Compute mean confidence from a list of items."""
    91	    if not items:
    92	        return 0.0
    93	    return sum(item[key] for item in items) / len(items)

exec
/usr/bin/zsh -lc 'nl -ba src/milestone1/egtr_infer.py' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
     1	"""EGTR model wrapper for scene graph generation inference."""
     2	
     3	import sys
     4	from pathlib import Path
     5	from typing import Any
     6	
     7	import numpy as np
     8	import torch
     9	from PIL import Image
    10	
    11	from milestone1.config import REPO_ROOT, setup_egtr_path
    12	from milestone1.vocab import load_obj_names, load_pred_names
    13	
    14	# Ensure EGTR is importable
    15	setup_egtr_path()
    16	
    17	from glob import glob
    18	
    19	from lib.pytorch_misc import argsort_desc
    20	from model.deformable_detr import (
    21	    DeformableDetrConfig,
    22	    DeformableDetrFeatureExtractor,
    23	)
    24	from model.egtr import DetrForSceneGraphGeneration
    25	from util.box_ops import rescale_bboxes
    26	
    27	
    28	class EgtrModel:
    29	    """Wrapper around EGTR for single-image scene graph inference."""
    30	
    31	    def __init__(
    32	        self,
    33	        artifact_path: str,
    34	        architecture: str,
    35	        vg_root: str,
    36	        object_conf_threshold: float = 0.3,
    37	        relation_conf_threshold: float = 0.2,
    38	        top_k_objects: int = 20,
    39	        top_k_relationships: int = 30,
    40	    ) -> None:
    41	        self.object_conf_threshold = object_conf_threshold
    42	        self.relation_conf_threshold = relation_conf_threshold
    43	        self.top_k_objects = top_k_objects
    44	        self.top_k_relationships = top_k_relationships
    45	
    46	        # Load vocabularies
    47	        self.obj_names = load_obj_names(vg_root)
    48	        self.pred_names = load_pred_names(vg_root)
    49	
    50	        # Load model
    51	        config = DeformableDetrConfig.from_pretrained(artifact_path)
    52	        self.num_labels = config.num_labels  # 150
    53	
    54	        self.model = DetrForSceneGraphGeneration.from_pretrained(
    55	            architecture, config=config, ignore_mismatched_sizes=True
    56	        )
    57	
    58	        # Load finetuned checkpoint
    59	        ckpt_path = sorted(
    60	            glob(f"{artifact_path}/checkpoints/epoch=*.ckpt"),
    61	            key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
    62	        )[-1]
    63	        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    64	        # Strip "model." prefix from PyTorch Lightning keys
    65	        for k in list(state_dict.keys()):
    66	            state_dict[k[6:]] = state_dict.pop(k)
    67	        self.model.load_state_dict(state_dict)
    68	        self.model.cuda()
    69	        self.model.eval()
    70	
    71	        # Feature extractor
    72	        self.feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
    73	            architecture, size=800, max_size=1333
    74	        )
    75	
    76	    def predict(self, pil_image: Image.Image) -> dict[str, Any]:
    77	        """Run inference on a single PIL image.
    78	
    79	        Returns {
    80	            objects: [{object_id, label, confidence, bbox: [x1,y1,x2,y2]}],
    81	            relationships: [{subject_id, subject_label, predicate,
    82	                             object_id, object_label, confidence}]
    83	        }
    84	        """
    85	        img_w, img_h = pil_image.size
    86	
    87	        # Preprocess
    88	        encoding = self.feature_extractor(
    89	            images=pil_image, return_tensors="pt"
    90	        )
    91	        pixel_values = encoding["pixel_values"].cuda()
    92	        pixel_mask = encoding["pixel_mask"].cuda()
    93	
    94	        # Forward pass
    95	        with torch.no_grad():
    96	            outputs = self.model(
    97	                pixel_values=pixel_values,
    98	                pixel_mask=pixel_mask,
    99	                output_attentions=False,
   100	                output_attention_states=True,
   101	                output_hidden_states=True,
   102	            )
   103	
   104	        # Extract objects
   105	        logits = outputs.logits[0]  # (200, 150)
   106	        pred_boxes = outputs.pred_boxes[0]  # (200, 4) cx,cy,w,h normalized
   107	        probs = logits.softmax(-1)
   108	        obj_scores, pred_classes = torch.max(
   109	            probs[:, : self.num_labels], dim=-1
   110	        )
   111	
   112	        # Convert boxes to absolute [x1, y1, x2, y2]
   113	        abs_boxes = rescale_bboxes(
   114	            pred_boxes.cpu(), (img_w, img_h)
   115	        )  # (200, 4)
   116	
   117	        # Filter and rank objects by confidence
   118	        valid_mask = obj_scores > self.object_conf_threshold
   119	        valid_indices = torch.where(valid_mask)[0]
   120	
   121	        # Sort by confidence descending, take top_k
   122	        if len(valid_indices) > 0:
   123	            sorted_idx = obj_scores[valid_indices].argsort(descending=True)
   124	            valid_indices = valid_indices[sorted_idx[: self.top_k_objects]]
   125	        else:
   126	            # Fallback: take top_k by score regardless of threshold
   127	            sorted_idx = obj_scores.argsort(descending=True)
   128	            valid_indices = sorted_idx[: self.top_k_objects]
   129	
   130	        objects = []
   131	        for rank, idx in enumerate(valid_indices):
   132	            idx_int = idx.item()
   133	            objects.append({
   134	                "object_id": rank,
   135	                "query_id": idx_int,
   136	                "label": self.obj_names[pred_classes[idx_int].item()],
   137	                "confidence": round(obj_scores[idx_int].item(), 4),
   138	                "bbox": [round(x, 1) for x in abs_boxes[idx_int].tolist()],
   139	            })
   140	
   141	        # Extract relationships (Method A: single pred per pair)
   142	        pred_rel = torch.clamp(outputs.pred_rel[0], 0.0, 1.0)  # (200,200,50)
   143	        pred_conn = torch.clamp(
   144	            outputs.pred_connectivity[0], 0.0, 1.0
   145	        )  # (200,200,1)
   146	        pred_rel = pred_rel * pred_conn  # gate by connectivity
   147	
   148	        # Object score outer product
   149	        sub_ob_scores = torch.outer(obj_scores, obj_scores)  # (200, 200)
   150	        n_queries = sub_ob_scores.shape[0]
   151	        sub_ob_scores[
   152	            torch.arange(n_queries), torch.arange(n_queries)
   153	        ] = 0.0
   154	
   155	        # Rank triplets
   156	        triplet_scores = pred_rel.max(-1)[0] * sub_ob_scores  # (200, 200)
   157	        top_pairs = argsort_desc(triplet_scores.cpu().numpy())[
   158	            : self.top_k_relationships * 3
   159	        ]
   160	
   161	        # Build query_id -> object_id mapping for valid objects
   162	        query_to_obj = {
   163	            obj["query_id"]: obj["object_id"] for obj in objects
   164	        }
   165	        valid_query_ids = set(query_to_obj.keys())
   166	
   167	        relationships = []
   168	        for s_q, o_q in top_pairs:
   169	            if s_q not in valid_query_ids or o_q not in valid_query_ids:
   170	                continue
   171	
   172	            best_pred = pred_rel[s_q, o_q].argmax().item()
   173	            rel_score = pred_rel[s_q, o_q, best_pred].item()
   174	
   175	            if rel_score < self.relation_conf_threshold:
   176	                continue
   177	
   178	            relationships.append({
   179	                "subject_id": query_to_obj[s_q],
   180	                "subject_label": self.obj_names[pred_classes[s_q].item()],
   181	                "predicate": self.pred_names[best_pred],
   182	                "object_id": query_to_obj[o_q],
   183	                "object_label": self.obj_names[pred_classes[o_q].item()],
   184	                "confidence": round(rel_score, 4),
   185	            })
   186	
   187	            if len(relationships) >= self.top_k_relationships:
   188	                break
   189	
   190	        return {"objects": objects, "relationships": relationships}

exec
/usr/bin/zsh -lc 'nl -ba scripts/aggregate_100.py' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
     1	"""Post-hoc aggregation for 100-image focused evaluation.
     2	
     3	Computes:
     4	1. Consecutive-k=2 stops for focused LLM configs (4b, 9b)
     5	2. Hybrid stop (confidence + LLM fusion)
     6	3. Method comparison vs true-GT OracleStop(0.5) -> method_comparison_100.csv
     7	4. Method comparison vs RelOracleStop(0.8) -> method_comparison_relative_100.csv
     8	5. Saturation rate table (RelOracle at 0.6/0.7/0.8/0.9)
     9	6. Figures: relative_method_comparison_mae_100.png, stop_distribution_all_100.png,
    10	   saturation_rate_100.png
    11	
    12	Methods included:
    13	  baseline; llm_sgg_4b_h5 (k1,k2); llm_sgg_9b_h5 (k1,k2);
    14	  vlm_4b_single (k1,k2); vlm_9b_single (k1,k2);
    15	  hybrid_4b (k1,k2); hybrid_9b (k1,k2).
    16	"""
    17	
    18	from __future__ import annotations
    19	
    20	import csv
    21	import json
    22	import math
    23	import sys
    24	from pathlib import Path
    25	
    26	import matplotlib
    27	
    28	matplotlib.use("Agg")
    29	import matplotlib.pyplot as plt  # noqa: E402
    30	import numpy as np  # noqa: E402
    31	
    32	REPO_ROOT = Path(__file__).resolve().parents[1]
    33	sys.path.insert(0, str(REPO_ROOT / "src"))
    34	
    35	from milestone1.config import load_config, resolve  # noqa: E402, I001
    36	from milestone1.metrics import _to_label_set, _to_triplet_set  # noqa: E402
    37	
    38	
    39	# ── Utilities ──
    40	
    41	
    42	def load_csv_dict(path: Path) -> list[dict]:
    43	    with open(path) as f:
    44	        return list(csv.DictReader(f))
    45	
    46	
    47	def save_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    48	    path.parent.mkdir(parents=True, exist_ok=True)
    49	    with open(path, "w", newline="") as f:
    50	        writer = csv.DictWriter(f, fieldnames=fieldnames)
    51	        writer.writeheader()
    52	        writer.writerows(rows)
    53	    print(f"  Saved {path}")
    54	
    55	
    56	def pearson_r(xs: list[float], ys: list[float]) -> float:
    57	    n = len(xs)
    58	    if n < 3:
    59	        return float("nan")
    60	    ax = np.array(xs, dtype=float)
    61	    ay = np.array(ys, dtype=float)
    62	    if np.std(ax) == 0 or np.std(ay) == 0:
    63	        return float("nan")
    64	    return float(np.corrcoef(ax, ay)[0, 1])
    65	
    66	
    67	def median(vals: list[float]) -> float:
    68	    s = sorted(vals)
    69	    n = len(s)
    70	    if n == 0:
    71	        return 0.0
    72	    if n % 2 == 1:
    73	        return s[n // 2]
    74	    return (s[n // 2 - 1] + s[n // 2]) / 2
    75	
    76	
    77	def load_saturation_scores(
    78	    base_dir: Path, image_id: int, start_level: int, max_level: int
    79	) -> dict[int, float]:
    80	    scores: dict[int, float] = {}
    81	    for lvl in range(start_level, max_level + 1):
    82	        p = base_dir / str(image_id) / f"level_{lvl:02d}.json"
    83	        if p.exists():
    84	            with open(p) as f:
    85	                data = json.load(f)
    86	            scores[lvl] = float(data["saturation_score"])
    87	    return scores
    88	
    89	
    90	def consecutive_k_stop(
    91	    scores: dict[int, float],
    92	    threshold: float,
    93	    k: int,
    94	    start_level: int,
    95	    max_level: int,
    96	) -> int:
    97	    for lvl in range(start_level, max_level + 1):
    98	        all_above = True
    99	        for offset in range(k):
   100	            check_lvl = lvl - offset
   101	            if check_lvl < start_level:
   102	                all_above = False
   103	                break
   104	            if scores.get(check_lvl, 0.0) < threshold:
   105	                all_above = False
   106	                break
   107	        if all_above:
   108	            return lvl
   109	    return max_level
   110	
   111	
   112	def load_egtr_prediction(pred_dir: Path, image_id: int, level: int) -> dict:
   113	    p = pred_dir / str(image_id) / f"level_{level:02d}.json"
   114	    with open(p) as f:
   115	        return json.load(f)
   116	
   117	
   118	def compute_relative_metrics(
   119	    pred_objects: list[dict],
   120	    pred_relationships: list[dict],
   121	    ref_objects: list[dict],
   122	    ref_relationships: list[dict],
   123	) -> dict[str, float]:
   124	    pred_nodes = _to_label_set(pred_objects)
   125	    ref_nodes = _to_label_set(ref_objects)
   126	    pred_trips = _to_triplet_set(pred_relationships)
   127	    ref_trips = _to_triplet_set(ref_relationships)
   128	
   129	    if len(ref_nodes) == 0:
   130	        node_recall = float("nan")
   131	    else:
   132	        node_recall = len(pred_nodes & ref_nodes) / len(ref_nodes)
   133	
   134	    if len(ref_trips) == 0:
   135	        trip_recall = float("nan")
   136	    else:
   137	        trip_recall = len(pred_trips & ref_trips) / len(ref_trips)
   138	
   139	    if math.isnan(trip_recall) if isinstance(trip_recall, float) else False:
   140	        rel_sg_score = node_recall if not math.isnan(node_recall) else float("nan")
   141	    elif math.isnan(node_recall) if isinstance(node_recall, float) else False:
   142	        rel_sg_score = trip_recall
   143	    else:
   144	        rel_sg_score = 0.5 * node_recall + 0.5 * trip_recall
   145	
   146	    return {
   147	        "RelNodeRecall": node_recall,
   148	        "RelTripletRecall": trip_recall,
   149	        "RelSGScore": rel_sg_score,
   150	    }
   151	
   152	
   153	# ── Main ──
   154	
   155	
   156	def main() -> None:
   157	    cfg = load_config()
   158	    max_level = cfg["resolution_levels"]["num_levels"]
   159	    start_level = cfg["llm"]["start_level"]
   160	    sat_threshold = cfg["llm"]["saturation_threshold"]
   161	    main_oracle_t = cfg["metrics"]["main_oracle_threshold"]
   162	    main_baseline_t = cfg["baseline"]["mean_object_conf_threshold"]
   163	    gain_epsilon = cfg["metrics"]["gain_epsilon"]
   164	    gain_k = cfg["metrics"]["gain_k"]
   165	
   166	    metrics_root = resolve(cfg["paths"]["metrics"])
   167	    figures_root = resolve(cfg["paths"]["figures"])
   168	    pred_root = resolve(cfg["paths"]["egtr_predictions"])
   169	    figures_root.mkdir(parents=True, exist_ok=True)
   170	
   171	    selected_path = resolve(cfg["paths"]["selected_images"])
   172	    with open(selected_path) as f:
   173	        image_ids: list[int] = json.load(f)
   174	
   175	    n_images = len(image_ids)
   176	    print(f"Processing {n_images} images")
   177	
   178	    # ── Load base data ──
   179	    all_metrics = load_csv_dict(metrics_root / "all_metrics.csv")
   180	    oracle_rows = load_csv_dict(metrics_root / "oracle_stop.csv")
   181	    baseline_rows = load_csv_dict(metrics_root / "baseline_stop.csv")
   182	    ablation_rows = load_csv_dict(metrics_root / "llm_ablation_stop.csv")
   183	    vlm_rows = load_csv_dict(metrics_root / "vlm_stop.csv")
   184	
   185	    oracle_map = {
   186	        int(r["image_id"]): int(float(r[f"oracle_{main_oracle_t}"]))
   187	        for r in oracle_rows
   188	    }
   189	    baseline_map = {
   190	        int(r["image_id"]): int(float(r[f"baseline_{main_baseline_t}"]))
   191	        for r in baseline_rows
   192	    }
   193	
   194	    # Ablation map
   195	    abl_map: dict[str, dict[int, int]] = {}
   196	    for r in ablation_rows:
   197	        slug = r["config_slug"]
   198	        abl_map.setdefault(slug, {})[int(r["image_id"])] = int(r["stop_level"])
   199	
   200	    # VLM map
   201	    vlm_map: dict[tuple[str, str, int], dict[int, int]] = {}
   202	    for r in vlm_rows:
   203	        key = (r["model"], r["mode"], int(r["k"]))
   204	        vlm_map.setdefault(key, {})[int(r["image_id"])] = int(r["stop_level"])
   205	
   206	    # Metrics index
   207	    metrics_by_img_lvl: dict[tuple[int, int], dict] = {}
   208	    for r in all_metrics:
   209	        metrics_by_img_lvl[(int(r["image_id"]), int(r["level"]))] = r
   210	
   211	    base_4b_slug = "qwen3.5-4b_h5_rel_on_conf_on"
   212	    base_9b_slug = "qwen3.5-9b_h5_rel_on_conf_on"
   213	    ablation_dir = resolve("outputs/llm_ablation")
   214	
   215	    # ── STEP 1: k=2 stops ──
   216	    print("\n" + "=" * 60)
   217	    print("STEP 1: Consecutive-k=2 post-hoc stops for focused LLM configs")
   218	    print("=" * 60)
   219	
   220	    abl_k2_map: dict[str, dict[int, int]] = {}
   221	    for slug in [base_4b_slug, base_9b_slug]:
   222	        abl_k2_map[slug] = {}
   223	        slug_dir = ablation_dir / slug
   224	        for img_id in image_ids:
   225	            scores = load_saturation_scores(slug_dir, img_id, start_level, max_level)
   226	            stop_k2 = consecutive_k_stop(
   227	                scores, sat_threshold, 2, start_level, max_level
   228	            )
   229	            abl_k2_map[slug][img_id] = stop_k2
   230	
   231	    # Save k2 CSV
   232	    abl_k2_rows = []
   233	    for slug in [base_4b_slug, base_9b_slug]:
   234	        for img_id in image_ids:
   235	            abl_k2_rows.append(
   236	                {
   237	                    "config_slug": slug,
   238	                    "image_id": img_id,
   239	                    "stop_k1": abl_map[slug].get(img_id, max_level),
   240	                    "stop_k2": abl_k2_map[slug].get(img_id, max_level),
   241	                }
   242	            )
   243	    save_csv(
   244	        abl_k2_rows,
   245	        metrics_root / "llm_ablation_k2_stop.csv",
   246	        ["config_slug", "image_id", "stop_k1", "stop_k2"],
   247	    )
   248	
   249	    # ── STEP 2: Hybrid stop ──
   250	    print("\n" + "=" * 60)
   251	    print("STEP 2: Hybrid stop (confidence + LLM fusion)")
   252	    print("=" * 60)
   253	
   254	    hybrid_cfg = cfg["ablation"]["hybrid"]
   255	    conf_w = hybrid_cfg["conf_weight"]
   256	    llm_w = hybrid_cfg["llm_weight"]
   257	    hybrid_threshold = hybrid_cfg["threshold"]
   258	
   259	    hybrid_rows: list[dict] = []
   260	    for variant_label, slug in [
   261	        ("hybrid_4b", base_4b_slug),
   262	        ("hybrid_9b", base_9b_slug),
   263	    ]:
   264	        slug_dir = ablation_dir / slug
   265	        for k_val in [1, 2]:
   266	            variant_name = f"{variant_label}_k{k_val}"
   267	            for img_id in image_ids:
   268	                sat_scores = load_saturation_scores(
   269	                    slug_dir, img_id, start_level, max_level
   270	                )
   271	                fused: dict[int, float] = {}
   272	                for lvl in range(start_level, max_level + 1):
   273	                    key = (img_id, lvl)
   274	                    if key in metrics_by_img_lvl and lvl in sat_scores:
   275	                        obj_conf = float(
   276	                            metrics_by_img_lvl[key]["mean_object_confidence"]
   277	                        )
   278	                        fused[lvl] = conf_w * obj_conf + llm_w * sat_scores[lvl]
   279	                    else:
   280	                        fused[lvl] = 0.0
   281	                stop = consecutive_k_stop(
   282	                    fused, hybrid_threshold, k_val, start_level, max_level
   283	                )
   284	                hybrid_rows.append(
   285	                    {
   286	                        "image_id": img_id,
   287	                        "variant": variant_name,
   288	                        "stop_level": stop,
   289	                    }
   290	                )
   291	
   292	    save_csv(
   293	        hybrid_rows,
   294	        metrics_root / "hybrid_stop.csv",
   295	        ["image_id", "variant", "stop_level"],
   296	    )
   297	
   298	    hybrid_map: dict[str, dict[int, int]] = {}
   299	    for r in hybrid_rows:
   300	        hybrid_map.setdefault(r["variant"], {})[r["image_id"]] = r["stop_level"]
   301	
   302	    # ── STEP 3: Relative metrics + RelOracle ──
   303	    print("\n" + "=" * 60)
   304	    print("STEP 3: Relative metrics + RelOracle")
   305	    print("=" * 60)
   306	
   307	    rel_oracle_thresholds = [0.6, 0.7, 0.8, 0.9]
   308	    main_rel_threshold = 0.8
   309	
   310	    # Build pseudo-GT from level_20
   311	    pseudo_gt: dict[int, dict] = {}
   312	    for img_id in image_ids:
   313	        pseudo_gt[img_id] = load_egtr_prediction(pred_root, img_id, max_level)
   314	
   315	    # Compute relative metrics
   316	    rel_sg_map: dict[tuple[int, int], float] = {}
   317	    rel_node_map: dict[tuple[int, int], float] = {}
   318	    rel_trip_map: dict[tuple[int, int], float] = {}
   319	
   320	    for img_id in image_ids:
   321	        ref = pseudo_gt[img_id]
   322	        for lvl in range(1, max_level + 1):
   323	            pred = load_egtr_prediction(pred_root, img_id, lvl)
   324	            m = compute_relative_metrics(
   325	                pred["objects"],
   326	                pred["relationships"],
   327	                ref["objects"],
   328	                ref["relationships"],
   329	            )
   330	            rel_sg_map[(img_id, lvl)] = (
   331	                m["RelSGScore"] if not math.isnan(m["RelSGScore"]) else float("nan")
   332	            )
   333	            rel_node_map[(img_id, lvl)] = (
   334	                m["RelNodeRecall"]
   335	                if not math.isnan(m["RelNodeRecall"])
   336	                else float("nan")
   337	            )
   338	            rel_trip_map[(img_id, lvl)] = (
   339	                m["RelTripletRecall"]
   340	                if not math.isnan(m["RelTripletRecall"])
   341	                else float("nan")
   342	            )
   343	
   344	    # RelOracle stops
   345	    rel_oracle_map: dict[int, int] = {}
   346	    rel_oracle_all: dict[float, dict[int, int]] = {t: {} for t in rel_oracle_thresholds}
   347	
   348	    for img_id in image_ids:
   349	        for t in rel_oracle_thresholds:
   350	            stop = max_level
   351	            for lvl in range(1, max_level + 1):
   352	                val = rel_sg_map.get((img_id, lvl), float("nan"))
   353	                if not math.isnan(val) and val >= t:
   354	                    stop = lvl
   355	                    break
   356	            rel_oracle_all[t][img_id] = stop
   357	
   358	        rel_oracle_map[img_id] = rel_oracle_all[main_rel_threshold][img_id]
   359	
   360	    # Saturation rates
   361	    print("\n  Saturation rates:")
   362	    for t in rel_oracle_thresholds:
   363	        sat = sum(1 for i in image_ids if rel_oracle_all[t][i] < max_level)
   364	        print(f"    RelOracle({t}): {sat}/{n_images} ({sat / n_images:.0%})")
   365	
   366	    gt_saturated = sum(1 for i in image_ids if oracle_map[i] < max_level)
   367	    rel_saturated = sum(1 for i in image_ids if rel_oracle_map[i] < max_level)
   368	    gt_mean = float(np.mean([oracle_map[i] for i in image_ids]))
   369	    rel_mean = float(np.mean([rel_oracle_map[i] for i in image_ids]))
   370	    print(
   371	        f"\n  GT Oracle(0.5): saturated={gt_saturated}/{n_images}, mean={gt_mean:.1f}"
   372	    )
   373	    print(
   374	        f"  RelOracle(0.8): saturated={rel_saturated}/{n_images}, mean={rel_mean:.1f}"
   375	    )
   376	
   377	    # ── STEP 4: Build all method maps ──
   378	    print("\n" + "=" * 60)
   379	    print("STEP 4: Build method maps")
   380	    print("=" * 60)
   381	
   382	    methods: dict[str, dict[int, int]] = {}
   383	    methods["baseline"] = baseline_map
   384	    methods["llm_sgg_4b_h5_k1"] = abl_map.get(base_4b_slug, {})
   385	    methods["llm_sgg_4b_h5_k2"] = abl_k2_map.get(base_4b_slug, {})
   386	    methods["llm_sgg_9b_h5_k1"] = abl_map.get(base_9b_slug, {})
   387	    methods["llm_sgg_9b_h5_k2"] = abl_k2_map.get(base_9b_slug, {})
   388	
   389	    vlm_method_names = {
   390	        ("qwen3.5:4b", "single", 1): "vlm_4b_single_k1",
   391	        ("qwen3.5:4b", "single", 2): "vlm_4b_single_k2",
   392	        ("qwen3.5:9b", "single", 1): "vlm_9b_single_k1",
   393	        ("qwen3.5:9b", "single", 2): "vlm_9b_single_k2",
   394	    }
   395	    for vlm_key, mname in vlm_method_names.items():
   396	        if vlm_key in vlm_map:
   397	            methods[mname] = vlm_map[vlm_key]
   398	
   399	    for v, stops in hybrid_map.items():
   400	        methods[v] = stops
   401	
   402	    # ── STEP 5: Comparison vs true-GT Oracle(0.5) ──
   403	    print("\n" + "=" * 60)
   404	    print("STEP 5: Method comparison vs true-GT Oracle(0.5)")
   405	    print("=" * 60)
   406	
   407	    def compute_comparison(
   408	        methods_dict: dict[str, dict[int, int]],
   409	        ref_map: dict[int, int],
   410	        ref_label: str,
   411	    ) -> list[dict]:
   412	        ref_stops = [ref_map[i] for i in image_ids]
   413	        rows = []
   414	        for mname, stop_map in methods_dict.items():
   415	            method_stops = [stop_map.get(i, max_level) for i in image_ids]
   416	            errors = [m - o for m, o in zip(method_stops, ref_stops, strict=True)]
   417	            abs_errors = [abs(e) for e in errors]
   418	            mean_stop = float(np.mean(method_stops))
   419	            mean_ae = float(np.mean(abs_errors))
   420	            median_ae = median(abs_errors)
   421	            early = sum(1 for e in errors if e < 0) / len(errors)
   422	            late = sum(1 for e in errors if e > 0) / len(errors)
   423	            corr = pearson_r(
   424	                [float(s) for s in method_stops],
   425	                [float(o) for o in ref_stops],
   426	            )
   427	            rows.append(
   428	                {
   429	                    "method": mname,
   430	                    "mean_stop": round(mean_stop, 2),
   431	                    f"mean_{ref_label}_stop": round(float(np.mean(ref_stops)), 2),
   432	                    "mean_abs_error": round(mean_ae, 2),
   433	                    "median_abs_error": round(median_ae, 1),
   434	                    "early_stop_rate": round(early, 2),
   435	                    "late_stop_rate": round(late, 2),
   436	                    "pearson_r": round(corr, 3) if not math.isnan(corr) else "NaN",
   437	                }
   438	            )
   439	        rows.sort(key=lambda r: float(r["mean_abs_error"]))
   440	        return rows
   441	
   442	    # vs GT Oracle(0.5)
   443	    gt_comparison = compute_comparison(methods, oracle_map, "oracle")
   444	    save_csv(
   445	        gt_comparison,
   446	        metrics_root / "method_comparison_100.csv",
   447	        list(gt_comparison[0].keys()),
   448	    )
   449	
   450	    print(
   451	        f"\n  {'Method':<28s} {'MeanStop':>8s} {'MAE':>6s} "
   452	        f"{'MedAE':>6s} {'Early':>6s} {'Late':>6s} {'Corr':>7s}"
   453	    )
   454	    print("  " + "-" * 75)
   455	    for r in gt_comparison:
   456	        print(
   457	            f"  {r['method']:<28s} {r['mean_stop']:>8} {r['mean_abs_error']:>6} "
   458	            f"{r['median_abs_error']:>6} {r['early_stop_rate']:>6} "
   459	            f"{r['late_stop_rate']:>6} {str(r['pearson_r']):>7}"
   460	        )
   461	
   462	    # vs RelOracle(0.8)
   463	    print("\n" + "=" * 60)
   464	    print("STEP 6: Method comparison vs RelOracle(0.8)")
   465	    print("=" * 60)
   466	
   467	    rel_comparison = compute_comparison(methods, rel_oracle_map, "rel_oracle")
   468	    save_csv(
   469	        rel_comparison,
   470	        metrics_root / "method_comparison_relative_100.csv",
   471	        list(rel_comparison[0].keys()),
   472	    )
   473	
   474	    print(
   475	        f"\n  {'Method':<28s} {'MeanStop':>8s} {'MAE':>6s} "
   476	        f"{'MedAE':>6s} {'Early':>6s} {'Late':>6s} {'Corr':>7s}"
   477	    )
   478	    print("  " + "-" * 75)
   479	    for r in rel_comparison:
   480	        print(
   481	            f"  {r['method']:<28s} {r['mean_stop']:>8} {r['mean_abs_error']:>6} "
   482	            f"{r['median_abs_error']:>6} {r['early_stop_rate']:>6} "
   483	            f"{r['late_stop_rate']:>6} {str(r['pearson_r']):>7}"
   484	        )
   485	
   486	    # ── STEP 7: Node vs Relation gap ──
   487	    print("\n" + "=" * 60)
   488	    print("STEP 7: Node vs Relation saturation gap")
   489	    print("=" * 60)
   490	
   491	    node_sat_threshold = 0.9
   492	    trip_sat_threshold = 0.9
   493	    gaps = []
   494	    for img_id in image_ids:
   495	        node_first = max_level
   496	        trip_first = max_level
   497	        for lvl in range(1, max_level + 1):
   498	            nr = rel_node_map.get((img_id, lvl), float("nan"))
   499	            tr = rel_trip_map.get((img_id, lvl), float("nan"))
   500	            if (
   501	                not math.isnan(nr)
   502	                and nr >= node_sat_threshold
   503	                and node_first == max_level
   504	            ):
   505	                node_first = lvl
   506	            if (
   507	                not math.isnan(tr)
   508	                and tr >= trip_sat_threshold
   509	                and trip_first == max_level
   510	            ):
   511	                trip_first = lvl
   512	        gaps.append(trip_first - node_first)
   513	    mean_gap = float(np.mean(gaps))
   514	    print(f"  Mean gap (trip_first - node_first at 0.9): {mean_gap:+.1f} levels")
   515	
   516	    # ── STEP 8: Figures ──
   517	    print("\n" + "=" * 60)
   518	    print("STEP 8: Figures")
   519	    print("=" * 60)
   520	
   521	    # Figure 1: Method comparison MAE vs RelOracle(0.8) horizontal bar
   522	    fig, ax = plt.subplots(figsize=(12, 8))
   523	    method_names = [r["method"] for r in reversed(rel_comparison)]
   524	    maes = [float(r["mean_abs_error"]) for r in reversed(rel_comparison)]
   525	    early_rates = [float(r["early_stop_rate"]) for r in reversed(rel_comparison)]
   526	    late_rates = [float(r["late_stop_rate"]) for r in reversed(rel_comparison)]
   527	
   528	    colors = []
   529	    for er, lr in zip(early_rates, late_rates, strict=True):
   530	        if er > lr:
   531	            colors.append("#e8746a")
   532	        elif lr > er:
   533	            colors.append("#6a9fe8")
   534	        else:
   535	            colors.append("#999999")
   536	
   537	    y_pos = np.arange(len(method_names))
   538	    bars = ax.barh(y_pos, maes, color=colors, edgecolor="black", linewidth=0.5)
   539	    for bar, er, lr in zip(bars, early_rates, late_rates, strict=True):
   540	        w = bar.get_width()
   541	        label = f"E{er:.0%}/L{lr:.0%}"
   542	        ax.text(
   543	            w + 0.15, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=7
   544	        )
   545	
   546	    ax.set_yticks(y_pos)
   547	    ax.set_yticklabels(method_names, fontsize=8)
   548	    ax.set_xlabel("Mean Absolute Error (levels)")
   549	    ax.set_title(f"Method Comparison: MAE vs RelOracle(0.8) [N={n_images}]")
   550	
   551	    from matplotlib.patches import Patch
   552	
   553	    legend_elements = [
   554	        Patch(facecolor="#e8746a", edgecolor="black", label="Tends early"),
   555	        Patch(facecolor="#6a9fe8", edgecolor="black", label="Tends late"),
   556	        Patch(facecolor="#999999", edgecolor="black", label="Balanced"),
   557	    ]
   558	    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
   559	    ax.grid(True, alpha=0.3, axis="x")
   560	    fig.tight_layout()
   561	    fig.savefig(figures_root / "relative_method_comparison_mae_100.png", dpi=150)
   562	    plt.close(fig)
   563	    print("  Saved relative_method_comparison_mae_100.png")
   564	
   565	    # Figure 2: Stop distribution box plot
   566	    plot_methods_order = [
   567	        "baseline",
   568	        "llm_sgg_4b_h5_k1",
   569	        "llm_sgg_4b_h5_k2",
   570	        "llm_sgg_9b_h5_k1",
   571	        "llm_sgg_9b_h5_k2",
   572	        "vlm_4b_single_k1",
   573	        "vlm_4b_single_k2",
   574	        "vlm_9b_single_k1",
   575	        "vlm_9b_single_k2",
   576	        "hybrid_4b_k1",
   577	        "hybrid_4b_k2",
   578	        "hybrid_9b_k1",
   579	        "hybrid_9b_k2",
   580	    ]
   581	    plot_methods = [m for m in plot_methods_order if m in methods]
   582	
   583	    fig, ax = plt.subplots(figsize=(14, 7))
   584	    box_data = []
   585	    box_labels = []
   586	    for mname in plot_methods:
   587	        stop_map = methods[mname]
   588	        stops = [stop_map.get(i, max_level) for i in image_ids]
   589	        box_data.append(stops)
   590	        box_labels.append(mname)
   591	
   592	    oracle_stops = [oracle_map[i] for i in image_ids]
   593	    rel_oracle_stops = [rel_oracle_map[i] for i in image_ids]
   594	    box_data.extend([oracle_stops, rel_oracle_stops])
   595	    box_labels.extend(["oracle(0.5)", "rel_oracle(0.8)"])
   596	
   597	    bp = ax.boxplot(box_data, vert=True, patch_artist=True, widths=0.6)
   598	    n_methods = len(box_labels)
   599	    cmap = plt.cm.tab20
   600	    for i, patch in enumerate(bp["boxes"]):
   601	        if box_labels[i] in ("oracle(0.5)", "rel_oracle(0.8)"):
   602	            patch.set_facecolor("#90EE90")
   603	            patch.set_alpha(0.8)
   604	        else:
   605	            patch.set_facecolor(cmap(i / max(n_methods - 1, 1)))
   606	            patch.set_alpha(0.6)
   607	
   608	    rng = np.random.default_rng(42)
   609	    for i, data in enumerate(box_data):
   610	        jitter = rng.uniform(-0.15, 0.15, len(data))
   611	        ax.scatter(
   612	            np.full(len(data), i + 1) + jitter,
   613	            data,
   614	            alpha=0.3,
   615	            s=10,
   616	            zorder=5,
   617	            color="black",
   618	        )
   619	
   620	    oracle_mean = np.mean(oracle_stops)
   621	    rel_oracle_mean = np.mean(rel_oracle_stops)
   622	    ax.axhline(
   623	        y=oracle_mean,
   624	        color="green",
   625	        linestyle="--",
   626	        alpha=0.7,
   627	        label=f"Oracle mean={oracle_mean:.1f}",
   628	    )
   629	    ax.axhline(
   630	        y=rel_oracle_mean,
   631	        color="blue",
   632	        linestyle=":",
   633	        alpha=0.7,
   634	        label=f"RelOracle mean={rel_oracle_mean:.1f}",
   635	    )
   636	
   637	    ax.set_xticks(range(1, n_methods + 1))
   638	    ax.set_xticklabels(box_labels, rotation=45, ha="right", fontsize=7)
   639	    ax.set_ylabel("Stop Level")
   640	    ax.set_title(f"Stop Level Distribution by Method [N={n_images}]")
   641	    ax.set_ylim(0, max_level + 1)
   642	    ax.legend(fontsize=8)
   643	    ax.grid(True, alpha=0.3, axis="y")
   644	    fig.tight_layout()
   645	    fig.savefig(figures_root / "stop_distribution_all_100.png", dpi=150)
   646	    plt.close(fig)
   647	    print("  Saved stop_distribution_all_100.png")
   648	
   649	    # Figure 3: Saturation rate comparison
   650	    fig, ax = plt.subplots(figsize=(10, 6))
   651	    thresholds = [0.6, 0.7, 0.8, 0.9]
   652	    sat_rates = []
   653	    for t in thresholds:
   654	        sat = sum(1 for i in image_ids if rel_oracle_all[t][i] < max_level)
   655	        sat_rates.append(sat / n_images)
   656	
   657	    gt_sat_rate = gt_saturated / n_images
   658	
   659	    ax.bar(
   660	        [str(t) for t in thresholds],
   661	        sat_rates,
   662	        color="#4C72B0",
   663	        edgecolor="black",
   664	        label="RelOracle saturation rate",
   665	    )
   666	    ax.axhline(
   667	        y=gt_sat_rate,
   668	        color="red",
   669	        linestyle="--",
   670	        alpha=0.8,
   671	        label=f"GT Oracle(0.5) rate={gt_sat_rate:.0%}",
   672	    )
   673	    for i, (t, r) in enumerate(zip(thresholds, sat_rates)):
   674	        ax.text(i, r + 0.02, f"{r:.0%}", ha="center", fontsize=10)
   675	
   676	    ax.set_xlabel("RelOracle Threshold")
   677	    ax.set_ylabel("Saturation Rate")
   678	    ax.set_title(f"Saturation Rate by RelOracle Threshold [N={n_images}]")
   679	    ax.set_ylim(0, 1.1)
   680	    ax.legend(fontsize=9)
   681	    ax.grid(True, alpha=0.3, axis="y")
   682	    fig.tight_layout()
   683	    fig.savefig(figures_root / "saturation_rate_100.png", dpi=150)
   684	    plt.close(fig)
   685	    print("  Saved saturation_rate_100.png")
   686	
   687	    # ── STEP 9: Count parse failures ──
   688	    print("\n" + "=" * 60)
   689	    print("STEP 9: Parse failure counts")
   690	    print("=" * 60)
   691	
   692	    for slug in [base_4b_slug, base_9b_slug]:
   693	        slug_dir = ablation_dir / slug
   694	        total = 0
   695	        failures = 0
   696	        for img_id in image_ids:
   697	            for lvl in range(start_level, max_level + 1):
   698	                p = slug_dir / str(img_id) / f"level_{lvl:02d}.json"
   699	                if p.exists():
   700	                    with open(p) as f:
   701	                        data = json.load(f)
   702	                    total += 1
   703	                    reason = data.get("reason", "")
   704	                    if "parse" in reason or "error" in reason:
   705	                        failures += 1
   706	        print(
   707	            f"  LLM {slug}: {failures}/{total} failures ({failures / max(total, 1) * 100:.1f}%)"
   708	        )
   709	
   710	    vlm_root = resolve("outputs/vlm_scores")
   711	    for model_name in ["qwen3.5:4b", "qwen3.5:9b"]:
   712	        slug = model_name.replace(":", "-")
   713	        mode_dir = vlm_root / slug / "single"
   714	        total = 0
   715	        failures = 0
   716	        for img_id in image_ids:
   717	            for lvl in range(start_level, max_level + 1):
   718	                p = mode_dir / str(img_id) / f"level_{lvl:02d}.json"
   719	                if p.exists():
   720	                    with open(p) as f:
   721	                        data = json.load(f)
   722	                    total += 1
   723	                    reason = data.get("reason", "")
   724	                    if "parse" in reason or "error" in reason:
   725	                        failures += 1
   726	        print(
   727	            f"  VLM {model_name} single: {failures}/{total} failures ({failures / max(total, 1) * 100:.1f}%)"
   728	        )
   729	
   730	    # ── Summary data for report ──
   731	    print("\n" + "=" * 60)
   732	    print("SUMMARY")
   733	    print("=" * 60)
   734	    print(f"  Images: {n_images}")
   735	    print(
   736	        f"  GT Oracle(0.5): saturated={gt_saturated}/{n_images} ({gt_saturated / n_images:.0%}), mean={gt_mean:.1f}"
   737	    )
   738	    print(
   739	        f"  RelOracle(0.8): saturated={rel_saturated}/{n_images} ({rel_saturated / n_images:.0%}), mean={rel_mean:.1f}"
   740	    )
   741	    print(f"  Node-vs-Relation gap: {mean_gap:+.1f} levels")
   742	    print("\n  Top-3 methods vs RelOracle(0.8):")
   743	    for r in rel_comparison[:3]:
   744	        print(f"    {r['method']}: MAE={r['mean_abs_error']}")
   745	
   746	    print("\n  Files created:")
   747	    print(f"    {metrics_root / 'method_comparison_100.csv'}")
   748	    print(f"    {metrics_root / 'method_comparison_relative_100.csv'}")
   749	    print(f"    {figures_root / 'relative_method_comparison_mae_100.png'}")
   750	    print(f"    {figures_root / 'stop_distribution_all_100.png'}")
   751	    print(f"    {figures_root / 'saturation_rate_100.png'}")
   752	
   753	
   754	if __name__ == "__main__":
   755	    main()

exec
/usr/bin/zsh -lc 'nl -ba configs/config.yaml' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
     1	# Milestone 1: Semantic information saturation detection from SGG output
     2	# All paths are relative to repository root (milestone_1/).
     3	
     4	paths:
     5	  vg_root: /home/kanaoka/usr/dataset/vg          # images/, train.json, val.json, test.json, rel.json
     6	  vg_split: test                                  # which VG split to draw GT + images from
     7	  egtr_repo: egtr                                 # EGTR submodule (imported, never modified)
     8	  egtr_artifact: egtr/checkpoint/egtr__pretrained_detr__SenseTime__deformable-detr__batch__32__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002__visual_genome__finetune/version_0
     9	  egtr_architecture: SenseTime/deformable-detr
    10	  selected_images: data/processed/selected_images.json
    11	  degraded_images: data/processed/degraded_images
    12	  gt_scene_graphs: data/processed/gt_scene_graphs
    13	  egtr_predictions: outputs/egtr_predictions
    14	  metrics: outputs/metrics
    15	  llm_inputs: outputs/llm_inputs
    16	  llm_scores: outputs/llm_scores
    17	  figures: outputs/figures
    18	
    19	dataset:
    20	  name: visual_genome
    21	  num_images: 100           # scaled up after 10-image validation
    22	  min_gt_objects: 5
    23	  min_gt_relationships: 3
    24	  use_person_subset: false
    25	  seed: 42
    26	
    27	resolution_levels:
    28	  num_levels: 20
    29	  scales:
    30	    - 0.05
    31	    - 0.075
    32	    - 0.10
    33	    - 0.125
    34	    - 0.15
    35	    - 0.175
    36	    - 0.20
    37	    - 0.25
    38	    - 0.30
    39	    - 0.35
    40	    - 0.40
    41	    - 0.45
    42	    - 0.50
    43	    - 0.55
    44	    - 0.60
    45	    - 0.65
    46	    - 0.70
    47	    - 0.80
    48	    - 0.90
    49	    - 1.00
    50	  interpolation: bilinear
    51	
    52	egtr:
    53	  object_conf_threshold: 0.3
    54	  relation_conf_threshold: 0.005
    55	  top_k_objects: 20
    56	  top_k_relationships: 30
    57	
    58	metrics:
    59	  sg_score:
    60	    node_weight: 0.5
    61	    relation_weight: 0.5
    62	  sg_score_rel_heavy:
    63	    node_weight: 0.4
    64	    relation_weight: 0.6
    65	  oracle_thresholds: [0.3, 0.4, 0.5, 0.6]
    66	  main_oracle_threshold: 0.5
    67	  gain_epsilon: 0.02
    68	  gain_k: 3
    69	
    70	baseline:
    71	  mean_object_conf_threshold: 0.4
    72	  mean_object_conf_thresholds: [0.3, 0.4, 0.5]
    73	  consecutive_k: 1
    74	
    75	llm:
    76	  provider: ollama
    77	  model: qwen3.5:4b
    78	  endpoint: http://localhost:11434/api/generate
    79	  start_level: 5
    80	  history_length: 5
    81	  saturation_threshold: 0.7
    82	  use_consecutive_stop: false
    83	  consecutive_k: 2
    84	  input_object_conf_threshold: 0.3
    85	  input_relation_conf_threshold: 0.005
    86	  input_top_k_objects: 20
    87	  input_top_k_relationships: 30
    88	  request_timeout: 120
    89	
    90	# ── Evaluation extensions (VLM baseline / ablation / analysis) ──
    91	
    92	# ollama chat endpoint (supports text + images via "images" field, "think": false)
    93	ollama_chat_endpoint: http://localhost:11434/api/chat
    94	
    95	vlm:
    96	  # qwen3.5:4b and :9b are both vision-capable -> same model family as the LLM-on-SGG
    97	  # approach, giving a capability-controlled "image vs structured-SG-text" comparison.
    98	  models: [qwen3.5:4b, qwen3.5:9b]
    99	  modes: [single, sequence]      # single = current-level image; sequence = recent history images
   100	  sequence_history: 3            # number of recent level-images for sequence mode (cost-bounded)
   101	  start_level: 5
   102	  saturation_threshold: 0.7
   103	  consecutive_k: 2
   104	  max_image_side: 512            # downscale images sent to VLM for transport speed (blur already baked in)
   105	  request_timeout: 300
   106	
   107	ablation:
   108	  # One-factor-at-a-time (OFAT) from this base config; LLM-input factors require new
   109	  # scored runs, post-hoc factors (consecutive_k, sg weights, hybrid) are computed from cache.
   110	  base:
   111	    model: qwen3.5:9b
   112	    history_length: 5
   113	    relation_on: true
   114	    confidence_on: true
   115	  llm_models: [qwen3.5:4b, qwen3.5:9b]       # model-size axis (at base config)
   116	  history_lengths: [1, 3, 5]                  # history axis (at base, 9b)
   117	  relation_on_options: [true, false]          # node-only vs node+relation (at base, 9b)
   118	  confidence_on_options: [true, false]        # include confidences in LLM input (at base, 9b)
   119	  consecutive_k_options: [1, 2]               # post-hoc stop rule
   120	  sg_score_weights:                           # post-hoc SGScore recomputation
   121	    balanced: [0.5, 0.5]
   122	    rel_heavy: [0.4, 0.6]
   123	  hybrid:                                      # fusion of node-confidence + LLM saturation_score
   124	    conf_weight: 0.5
   125	    llm_weight: 0.5
   126	    threshold: 0.6
   127	
   128	analysis:
   129	  # which methods to analyze "what was missing at the stop level" for
   130	  stop_methods: [oracle, baseline, llm, vlm_single, vlm_sequence]

exec
/usr/bin/zsh -lc 'nl -ba scripts/15_relative_saturation.py' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
     1	"""Relative saturation analysis (Plan B) for Milestone-1 SGG-saturation PoC.
     2	
     3	Reference = each image's own Level-20 EGTR output (pseudo-GT), NOT the
     4	dataset's true GT.  This removes the EGTR-ceiling confound (absolute SGScore
     5	only reaches ~0.25-0.30 vs true GT) and matches the robot stopping problem.
     6	
     7	Steps:
     8	  1. Build pseudo-GT per image from level_20.json
     9	  2. Compute relative metrics per (image, level 1..20)
    10	  3. Compute RelOracleStop (threshold + gain-based)
    11	  4. Re-evaluate all stop methods vs RelOracle(0.8)
    12	  5. Compare RelOracle vs old true-GT Oracle
    13	  6. Generate figures
    14	
    15	Run:
    16	  cd /home/kanaoka/usr/github/milestone_1 && \
    17	  uv run --project egtr python scripts/15_relative_saturation.py
    18	"""
    19	
    20	from __future__ import annotations
    21	
    22	import csv
    23	import json
    24	import math
    25	import sys
    26	from pathlib import Path
    27	
    28	import matplotlib
    29	
    30	matplotlib.use("Agg")
    31	import matplotlib.pyplot as plt  # noqa: E402
    32	import numpy as np  # noqa: E402
    33	
    34	REPO_ROOT = Path(__file__).resolve().parents[1]
    35	sys.path.insert(0, str(REPO_ROOT / "src"))
    36	
    37	from milestone1.config import load_config, resolve  # noqa: E402, I001
    38	from milestone1.metrics import _to_label_set, _to_triplet_set  # noqa: E402
    39	
    40	
    41	# ---------------------------------------------------------------------------
    42	# Utilities
    43	# ---------------------------------------------------------------------------
    44	
    45	
    46	def load_csv_dict(path: Path) -> list[dict]:
    47	    """Load CSV as list of dicts."""
    48	    with open(path) as f:
    49	        return list(csv.DictReader(f))
    50	
    51	
    52	def save_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    53	    """Write a list of dicts to CSV."""
    54	    path.parent.mkdir(parents=True, exist_ok=True)
    55	    with open(path, "w", newline="") as f:
    56	        writer = csv.DictWriter(f, fieldnames=fieldnames)
    57	        writer.writeheader()
    58	        writer.writerows(rows)
    59	    print(f"  Saved {path}")
    60	
    61	
    62	def pearson_r(xs: list[float], ys: list[float]) -> float:
    63	    """Pearson correlation; returns NaN if degenerate."""
    64	    n = len(xs)
    65	    if n < 3:
    66	        return float("nan")
    67	    ax = np.array(xs, dtype=float)
    68	    ay = np.array(ys, dtype=float)
    69	    if np.std(ax) == 0 or np.std(ay) == 0:
    70	        return float("nan")
    71	    return float(np.corrcoef(ax, ay)[0, 1])
    72	
    73	
    74	def median(vals: list[float]) -> float:
    75	    """Simple median."""
    76	    s = sorted(vals)
    77	    n = len(s)
    78	    if n == 0:
    79	        return 0.0
    80	    if n % 2 == 1:
    81	        return s[n // 2]
    82	    return (s[n // 2 - 1] + s[n // 2]) / 2
    83	
    84	
    85	def load_egtr_prediction(pred_dir: Path, image_id: int, level: int) -> dict:
    86	    """Load EGTR prediction JSON for (image_id, level)."""
    87	    p = pred_dir / str(image_id) / f"level_{level:02d}.json"
    88	    with open(p) as f:
    89	        return json.load(f)
    90	
    91	
    92	# ---------------------------------------------------------------------------
    93	# Relative metric computation
    94	# ---------------------------------------------------------------------------
    95	
    96	
    97	def compute_relative_metrics(
    98	    pred_objects: list[dict],
    99	    pred_relationships: list[dict],
   100	    ref_objects: list[dict],
   101	    ref_relationships: list[dict],
   102	) -> dict[str, float]:
   103	    """Compute relative metrics for a prediction vs pseudo-GT reference.
   104	
   105	    Returns dict with: RelNodeRecall, RelTripletRecall, RelSGScore,
   106	    RelNodeF1, RelTripletF1, RelSGScore_f1.
   107	    """
   108	    pred_nodes = _to_label_set(pred_objects)
   109	    ref_nodes = _to_label_set(ref_objects)
   110	
   111	    pred_trips = _to_triplet_set(pred_relationships)
   112	    ref_trips = _to_triplet_set(ref_relationships)
   113	
   114	    # Node recall (relative)
   115	    if len(ref_nodes) == 0:
   116	        node_recall = float("nan")
   117	        node_prec = float("nan")
   118	        node_f1 = float("nan")
   119	    else:
   120	        tp_nodes = len(pred_nodes & ref_nodes)
   121	        node_recall = tp_nodes / len(ref_nodes)
   122	        node_prec = tp_nodes / len(pred_nodes) if len(pred_nodes) > 0 else 0.0
   123	        node_f1 = (
   124	            2 * node_prec * node_recall / (node_prec + node_recall)
   125	            if (node_prec + node_recall) > 0
   126	            else 0.0
   127	        )
   128	
   129	    # Triplet recall (relative)
   130	    if len(ref_trips) == 0:
   131	        trip_recall = float("nan")
   132	        trip_prec = float("nan")
   133	        trip_f1 = float("nan")
   134	    else:
   135	        tp_trips = len(pred_trips & ref_trips)
   136	        trip_recall = tp_trips / len(ref_trips)
   137	        trip_prec = tp_trips / len(pred_trips) if len(pred_trips) > 0 else 0.0
   138	        trip_f1 = (
   139	            2 * trip_prec * trip_recall / (trip_prec + trip_recall)
   140	            if (trip_prec + trip_recall) > 0
   141	            else 0.0
   142	        )
   143	
   144	    # RelSGScore (recall-based): 0.5 * node_recall + 0.5 * trip_recall
   145	    # If triplet is NaN (no ref triplets), use node_recall only
   146	    if math.isnan(trip_recall):
   147	        rel_sg_score = node_recall if not math.isnan(node_recall) else float("nan")
   148	    elif math.isnan(node_recall):
   149	        rel_sg_score = trip_recall
   150	    else:
   151	        rel_sg_score = 0.5 * node_recall + 0.5 * trip_recall
   152	
   153	    # RelSGScore_f1 (F1-based)
   154	    if math.isnan(trip_f1):
   155	        rel_sg_score_f1 = node_f1 if not math.isnan(node_f1) else float("nan")
   156	    elif math.isnan(node_f1):
   157	        rel_sg_score_f1 = trip_f1
   158	    else:
   159	        rel_sg_score_f1 = 0.5 * node_f1 + 0.5 * trip_f1
   160	
   161	    return {
   162	        "RelNodeRecall": round(node_recall, 6)
   163	        if not math.isnan(node_recall)
   164	        else "NaN",
   165	        "RelTripletRecall": round(trip_recall, 6)
   166	        if not math.isnan(trip_recall)
   167	        else "NaN",
   168	        "RelSGScore": round(rel_sg_score, 6) if not math.isnan(rel_sg_score) else "NaN",
   169	        "RelNodeF1": round(node_f1, 6) if not math.isnan(node_f1) else "NaN",
   170	        "RelTripletF1": round(trip_f1, 6) if not math.isnan(trip_f1) else "NaN",
   171	        "RelSGScore_f1": round(rel_sg_score_f1, 6)
   172	        if not math.isnan(rel_sg_score_f1)
   173	        else "NaN",
   174	    }
   175	
   176	
   177	# ---------------------------------------------------------------------------
   178	# Main
   179	# ---------------------------------------------------------------------------
   180	
   181	
   182	def main() -> None:
   183	    cfg = load_config()
   184	    max_level = cfg["resolution_levels"]["num_levels"]  # 20
   185	    gain_epsilon = cfg["metrics"]["gain_epsilon"]  # 0.02
   186	    gain_k = cfg["metrics"]["gain_k"]  # 3
   187	    main_baseline_t = cfg["baseline"]["mean_object_conf_threshold"]  # 0.4
   188	
   189	    # For relative oracle, use threshold=0.8 as primary (stated in task)
   190	    rel_oracle_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
   191	    main_rel_threshold = 0.8
   192	
   193	    metrics_root = resolve(cfg["paths"]["metrics"])
   194	    figures_root = resolve(cfg["paths"]["figures"])
   195	    pred_root = resolve(cfg["paths"]["egtr_predictions"])
   196	    figures_root.mkdir(parents=True, exist_ok=True)
   197	
   198	    selected_path = resolve(cfg["paths"]["selected_images"])
   199	    with open(selected_path) as f:
   200	        image_ids: list[int] = json.load(f)
   201	
   202	    # ===================================================================
   203	    # STEP 1: Build pseudo-GT from level_20 predictions
   204	    # ===================================================================
   205	    print("=" * 60)
   206	    print("STEP 1: Build pseudo-GT from level_20 predictions")
   207	    print("=" * 60)
   208	
   209	    pseudo_gt: dict[int, dict] = {}
   210	    for img_id in image_ids:
   211	        data = load_egtr_prediction(pred_root, img_id, max_level)
   212	        pseudo_gt[img_id] = data
   213	        node_set = _to_label_set(data["objects"])
   214	        trip_set = _to_triplet_set(data["relationships"])
   215	        print(
   216	            f"  {img_id}: {len(node_set)} unique nodes, {len(trip_set)} unique triplets"
   217	        )
   218	
   219	    # ===================================================================
   220	    # STEP 2: Compute relative metrics per (image, level)
   221	    # ===================================================================
   222	    print()
   223	    print("=" * 60)
   224	    print("STEP 2: Compute relative metrics per (image, level 1..20)")
   225	    print("=" * 60)
   226	
   227	    rel_metrics_rows: list[dict] = []
   228	    # Also store for easy lookup: {(img_id, level): RelSGScore}
   229	    rel_sg_map: dict[tuple[int, int], float] = {}
   230	
   231	    for img_id in image_ids:
   232	        ref = pseudo_gt[img_id]
   233	        for lvl in range(1, max_level + 1):
   234	            pred = load_egtr_prediction(pred_root, img_id, lvl)
   235	            m = compute_relative_metrics(
   236	                pred["objects"],
   237	                pred["relationships"],
   238	                ref["objects"],
   239	                ref["relationships"],
   240	            )
   241	            row = {"image_id": img_id, "level": lvl, **m}
   242	            rel_metrics_rows.append(row)
   243	
   244	            # Parse RelSGScore for internal use
   245	            sg_val = m["RelSGScore"]
   246	            rel_sg_map[(img_id, lvl)] = (
   247	                float(sg_val) if sg_val != "NaN" else float("nan")
   248	            )
   249	
   250	        # Print level-20 sanity check
   251	        sg20 = rel_sg_map[(img_id, max_level)]
   252	        print(f"  {img_id}: RelSGScore(20)={sg20:.4f} (should be 1.0)")
   253	
   254	    save_csv(
   255	        rel_metrics_rows,
   256	        metrics_root / "relative_metrics.csv",
   257	        [
   258	            "image_id",
   259	            "level",
   260	            "RelNodeRecall",
   261	            "RelTripletRecall",
   262	            "RelSGScore",
   263	            "RelNodeF1",
   264	            "RelTripletF1",
   265	            "RelSGScore_f1",
   266	        ],
   267	    )
   268	
   269	    # ===================================================================
   270	    # STEP 3: RelOracleStop (threshold-based + gain-based)
   271	    # ===================================================================
   272	    print()
   273	    print("=" * 60)
   274	    print("STEP 3: Compute RelOracleStop")
   275	    print("=" * 60)
   276	
   277	    rel_oracle_rows: list[dict] = []
   278	    rel_oracle_map: dict[int, int] = {}  # main threshold -> {img_id: stop}
   279	
   280	    for img_id in image_ids:
   281	        row: dict[str, object] = {"image_id": img_id}
   282	
   283	        # Threshold-based
   284	        for t in rel_oracle_thresholds:
   285	            stop = max_level
   286	            for lvl in range(1, max_level + 1):
   287	                val = rel_sg_map.get((img_id, lvl), float("nan"))
   288	                if not math.isnan(val) and val >= t:
   289	                    stop = lvl
   290	                    break
   291	            row[f"rel_oracle_{t}"] = stop
   292	
   293	        # Main threshold stop
   294	        rel_oracle_map[img_id] = int(row[f"rel_oracle_{main_rel_threshold}"])
   295	
   296	        # Gain-based saturation
   297	        sg_scores = [
   298	            rel_sg_map.get((img_id, lvl), float("nan"))
   299	            for lvl in range(1, max_level + 1)
   300	        ]
   301	        gains = [0.0]
   302	        for i in range(1, len(sg_scores)):
   303	            if math.isnan(sg_scores[i]) or math.isnan(sg_scores[i - 1]):
   304	                gains.append(0.0)
   305	            else:
   306	                gains.append(sg_scores[i] - sg_scores[i - 1])
   307	
   308	        gain_stop = max_level
   309	        consecutive = 0
   310	        for i, g in enumerate(gains):
   311	            if abs(g) < gain_epsilon:
   312	                consecutive += 1
   313	                if consecutive >= gain_k:
   314	                    gain_stop = i + 1 - (gain_k - 1)
   315	                    break
   316	            else:
   317	                consecutive = 0
   318	        row["gain_based"] = gain_stop
   319	
   320	        rel_oracle_rows.append(row)
   321	        print(
   322	            f"  {img_id}: rel_oracle_{main_rel_threshold}="
   323	            f"{row[f'rel_oracle_{main_rel_threshold}']}, "
   324	            f"gain_based={row['gain_based']}"
   325	        )
   326	
   327	    fieldnames = (
   328	        ["image_id"]
   329	        + [f"rel_oracle_{t}" for t in rel_oracle_thresholds]
   330	        + ["gain_based"]
   331	    )
   332	    save_csv(rel_oracle_rows, metrics_root / "relative_oracle_stop.csv", fieldnames)
   333	
   334	    # Non-degeneracy check
   335	    saturated_main = sum(
   336	        1 for img_id in image_ids if rel_oracle_map[img_id] < max_level
   337	    )
   338	    print(
   339	        f"\n  Non-degeneracy: {saturated_main}/{len(image_ids)} images saturate "
   340	        f"at RelSGScore >= {main_rel_threshold} (vs 4/10 before with true-GT oracle_0.5)"
   341	    )
   342	
   343	    # ===================================================================
   344	    # STEP 4: Re-evaluate all stop methods vs RelOracle(0.8)
   345	    # ===================================================================
   346	    print()
   347	    print("=" * 60)
   348	    print("STEP 4: Re-evaluate all methods vs RelOracle(0.8)")
   349	    print("=" * 60)
   350	
   351	    # Load all stop data
   352	    baseline_rows = load_csv_dict(metrics_root / "baseline_stop.csv")
   353	    ablation_rows = load_csv_dict(metrics_root / "llm_ablation_stop.csv")
   354	    vlm_rows = load_csv_dict(metrics_root / "vlm_stop.csv")
   355	    hybrid_rows_csv = load_csv_dict(metrics_root / "hybrid_stop.csv")
   356	    abl_k2_rows = load_csv_dict(metrics_root / "llm_ablation_k2_stop.csv")
   357	
   358	    # Build lookup maps
   359	    baseline_map = {
   360	        int(r["image_id"]): int(float(r[f"baseline_{main_baseline_t}"]))
   361	        for r in baseline_rows
   362	    }
   363	
   364	    # Ablation map: config_slug -> {image_id: stop}
   365	    abl_map: dict[str, dict[int, int]] = {}
   366	    for r in ablation_rows:
   367	        slug = r["config_slug"]
   368	        abl_map.setdefault(slug, {})[int(r["image_id"])] = int(r["stop_level"])
   369	
   370	    # Ablation k2 map: config_slug -> {image_id: stop_k2}
   371	    abl_k2_map: dict[str, dict[int, int]] = {}
   372	    for r in abl_k2_rows:
   373	        slug = r["config_slug"]
   374	        abl_k2_map.setdefault(slug, {})[int(r["image_id"])] = int(r["stop_k2"])
   375	
   376	    # VLM map: (model, mode, k) -> {image_id: stop}
   377	    vlm_map: dict[tuple[str, str, int], dict[int, int]] = {}
   378	    for r in vlm_rows:
   379	        key = (r["model"], r["mode"], int(r["k"]))
   380	        vlm_map.setdefault(key, {})[int(r["image_id"])] = int(r["stop_level"])
   381	
   382	    # Hybrid map: variant -> {image_id: stop}
   383	    hybrid_map: dict[str, dict[int, int]] = {}
   384	    for r in hybrid_rows_csv:
   385	        hybrid_map.setdefault(r["variant"], {})[int(r["image_id"])] = int(
   386	            r["stop_level"]
   387	        )
   388	
   389	    # Assemble all methods
   390	    base_4b_slug = "qwen3.5-4b_h5_rel_on_conf_on"
   391	    base_9b_slug = "qwen3.5-9b_h5_rel_on_conf_on"
   392	
   393	    methods: dict[str, dict[int, int]] = {}
   394	    methods["baseline"] = baseline_map
   395	
   396	    # LLM-SGG from ablation: 4b and 9b base, k=1 and k=2
   397	    methods["llm_sgg_4b_h5_k1"] = abl_map[base_4b_slug]
   398	    methods["llm_sgg_4b_h5_k2"] = abl_k2_map[base_4b_slug]
   399	    methods["llm_sgg_9b_h5_k1"] = abl_map[base_9b_slug]
   400	    methods["llm_sgg_9b_h5_k2"] = abl_k2_map[base_9b_slug]
   401	
   402	    # Ablation variants (9b, k=1)
   403	    abl_variant_slugs = {
   404	        "abl_9b_h1": "qwen3.5-9b_h1_rel_on_conf_on",
   405	        "abl_9b_h3": "qwen3.5-9b_h3_rel_on_conf_on",
   406	        "abl_9b_rel_off": "qwen3.5-9b_h5_rel_off_conf_on",
   407	        "abl_9b_conf_off": "qwen3.5-9b_h5_rel_on_conf_off",
   408	    }
   409	    for mname, slug in abl_variant_slugs.items():
   410	        methods[mname] = abl_map[slug]
   411	
   412	    # VLM methods
   413	    vlm_method_names = {
   414	        ("qwen3.5:4b", "single", 1): "vlm_4b_single_k1",
   415	        ("qwen3.5:4b", "single", 2): "vlm_4b_single_k2",
   416	        ("qwen3.5:9b", "single", 1): "vlm_9b_single_k1",
   417	        ("qwen3.5:9b", "single", 2): "vlm_9b_single_k2",
   418	        ("qwen3.5:4b", "sequence", 1): "vlm_4b_sequence_k1",
   419	        ("qwen3.5:4b", "sequence", 2): "vlm_4b_sequence_k2",
   420	        ("qwen3.5:9b", "sequence", 1): "vlm_9b_sequence_k1",
   421	        ("qwen3.5:9b", "sequence", 2): "vlm_9b_sequence_k2",
   422	    }
   423	    for vlm_key, mname in vlm_method_names.items():
   424	        if vlm_key in vlm_map:
   425	            methods[mname] = vlm_map[vlm_key]
   426	
   427	    # Hybrid methods
   428	    for v, stops in hybrid_map.items():
   429	        methods[v] = stops
   430	
   431	    # Compute comparison stats vs RelOracle(0.8)
   432	    oracle_stops = [rel_oracle_map[i] for i in image_ids]
   433	    comparison_rows: list[dict] = []
   434	
   435	    for mname, stop_map in methods.items():
   436	        method_stops = [stop_map.get(i, max_level) for i in image_ids]
   437	        errors = [m - o for m, o in zip(method_stops, oracle_stops, strict=True)]
   438	        abs_errors = [abs(e) for e in errors]
   439	
   440	        mean_stop = float(np.mean(method_stops))
   441	        mean_ae = float(np.mean(abs_errors))
   442	        median_ae = median(abs_errors)
   443	        early = sum(1 for e in errors if e < 0) / len(errors)
   444	        late = sum(1 for e in errors if e > 0) / len(errors)
   445	        corr = pearson_r(
   446	            [float(s) for s in method_stops],
   447	            [float(o) for o in oracle_stops],
   448	        )
   449	
   450	        comparison_rows.append(
   451	            {
   452	                "method": mname,
   453	                "mean_stop": round(mean_stop, 2),
   454	                "mean_rel_oracle_stop": round(float(np.mean(oracle_stops)), 2),
   455	                "mean_abs_error": round(mean_ae, 2),
   456	                "median_abs_error": round(median_ae, 1),
   457	                "early_stop_rate": round(early, 2),
   458	                "late_stop_rate": round(late, 2),
   459	                "pearson_r": round(corr, 3) if not math.isnan(corr) else "NaN",
   460	            }
   461	        )
   462	
   463	    # Sort by MAE ascending
   464	    comparison_rows.sort(key=lambda r: float(r["mean_abs_error"]))
   465	
   466	    save_csv(
   467	        comparison_rows,
   468	        metrics_root / "method_comparison_relative.csv",
   469	        list(comparison_rows[0].keys()),
   470	    )
   471	
   472	    print(
   473	        f"\n  {'Method':<28s} {'MeanStop':>8s} {'OracleM':>7s} {'MAE':>6s} "
   474	        f"{'MedAE':>6s} {'Early':>6s} {'Late':>6s} {'Corr':>7s}"
   475	    )
   476	    print("  " + "-" * 80)
   477	    for r in comparison_rows:
   478	        print(
   479	            f"  {r['method']:<28s} {r['mean_stop']:>8} {r['mean_rel_oracle_stop']:>7} "
   480	            f"{r['mean_abs_error']:>6} {r['median_abs_error']:>6} "
   481	            f"{r['early_stop_rate']:>6} {r['late_stop_rate']:>6} "
   482	            f"{str(r['pearson_r']):>7}"
   483	        )
   484	
   485	    # ===================================================================
   486	    # STEP 5: Compare RelOracle vs old true-GT OracleStop
   487	    # ===================================================================
   488	    print()
   489	    print("=" * 60)
   490	    print("STEP 5: Compare RelOracle(0.8) vs old GT Oracle(0.5)")
   491	    print("=" * 60)
   492	
   493	    old_oracle_rows = load_csv_dict(metrics_root / "oracle_stop.csv")
   494	    old_oracle_map = {
   495	        int(r["image_id"]): int(float(r["oracle_0.5"])) for r in old_oracle_rows
   496	    }
   497	
   498	    print(
   499	        f"\n  {'image_id':<12s} {'GT_Oracle_0.5':>14s} {'RelOracle_0.8':>14s} "
   500	        f"{'Diff(Rel-GT)':>13s}"
   501	    )
   502	    print("  " + "-" * 55)
   503	    diffs = []
   504	    for img_id in image_ids:
   505	        gt_stop = old_oracle_map[img_id]
   506	        rel_stop = rel_oracle_map[img_id]
   507	        diff = rel_stop - gt_stop
   508	        diffs.append(diff)
   509	        print(f"  {img_id:<12d} {gt_stop:>14d} {rel_stop:>14d} {diff:>+13d}")
   510	
   511	    gt_saturated = sum(1 for i in image_ids if old_oracle_map[i] < max_level)
   512	    rel_saturated = sum(1 for i in image_ids if rel_oracle_map[i] < max_level)
   513	    gt_mean = float(np.mean([old_oracle_map[i] for i in image_ids]))
   514	    rel_mean = float(np.mean([rel_oracle_map[i] for i in image_ids]))
   515	
   516	    n_images = len(image_ids)
   517	    print(
   518	        f"\n  GT Oracle(0.5):  mean={gt_mean:.1f}, saturated={gt_saturated}/{n_images}"
   519	    )
   520	    print(
   521	        f"  RelOracle(0.8):  mean={rel_mean:.1f}, saturated={rel_saturated}/{n_images}"
   522	    )
   523	    print(f"  Mean diff (Rel - GT): {np.mean(diffs):+.1f}")
   524	
   525	    # ===================================================================
   526	    # STEP 5b: Node-vs-relation saturation gap under relative reference
   527	    # ===================================================================
   528	    print()
   529	    print("=" * 60)
   530	    print("STEP 5b: Node vs Relation saturation gap (relative ref)")
   531	    print("=" * 60)
   532	
   533	    node_sat_threshold = 0.9
   534	    trip_sat_threshold = 0.9
   535	
   536	    node_first_levels = []
   537	    trip_first_levels = []
   538	    gaps = []
   539	
   540	    for img_id in image_ids:
   541	        # Find first level where RelNodeRecall >= 0.9
   542	        node_first = max_level
   543	        trip_first = max_level
   544	        for lvl in range(1, max_level + 1):
   545	            # Parse from rel_metrics_rows
   546	            row = next(
   547	                r
   548	                for r in rel_metrics_rows
   549	                if int(r["image_id"]) == img_id and int(r["level"]) == lvl
   550	            )
   551	            nr = row["RelNodeRecall"]
   552	            tr = row["RelTripletRecall"]
   553	
   554	            if (
   555	                nr != "NaN"
   556	                and float(nr) >= node_sat_threshold
   557	                and node_first == max_level
   558	            ):
   559	                node_first = lvl
   560	            if (
   561	                tr != "NaN"
   562	                and float(tr) >= trip_sat_threshold
   563	                and trip_first == max_level
   564	            ):
   565	                trip_first = lvl
   566	
   567	        node_first_levels.append(node_first)
   568	        trip_first_levels.append(trip_first)
   569	        gap = trip_first - node_first
   570	        gaps.append(gap)
   571	
   572	        # Check if image has any triplets at level 20
   573	        ref_trips = _to_triplet_set(pseudo_gt[img_id]["relationships"])
   574	        has_trips = len(ref_trips) > 0
   575	        print(
   576	            f"  {img_id}: node_first={node_first}, trip_first={trip_first}, "
   577	            f"gap={gap:+d}, has_trips={has_trips}"
   578	        )
   579	
   580	    mean_gap = float(np.mean(gaps))
   581	    print(f"\n  Mean gap (trip_first - node_first at 0.9): {mean_gap:+.1f} levels")
   582	    print(
   583	        "  Interpretation: relations saturate LATER than nodes by "
   584	        f"~{abs(mean_gap):.1f} levels on average"
   585	        if mean_gap > 0
   586	        else "  Interpretation: relations saturate at similar/earlier levels"
   587	    )
   588	
   589	    # ===================================================================
   590	    # STEP 6: Figures
   591	    # ===================================================================
   592	    print()
   593	    print("=" * 60)
   594	    print("STEP 6: Generate figures")
   595	    print("=" * 60)
   596	
   597	    # Build per-image relative metric lookup for plotting
   598	    # {img_id: {level: {metric_name: value}}}
   599	    img_rel_data: dict[int, dict[int, dict[str, float]]] = {}
   600	    for r in rel_metrics_rows:
   601	        iid = int(r["image_id"])
   602	        lvl = int(r["level"])
   603	        img_rel_data.setdefault(iid, {})[lvl] = {
   604	            "RelNodeRecall": float(r["RelNodeRecall"])
   605	            if r["RelNodeRecall"] != "NaN"
   606	            else float("nan"),
   607	            "RelTripletRecall": float(r["RelTripletRecall"])
   608	            if r["RelTripletRecall"] != "NaN"
   609	            else float("nan"),
   610	            "RelSGScore": float(r["RelSGScore"])
   611	            if r["RelSGScore"] != "NaN"
   612	            else float("nan"),
   613	        }
   614	
   615	    # LLM-4b stop map (for vertical lines)
   616	    llm_4b_stop = abl_map.get(base_4b_slug, {})
   617	
   618	    # --- Figure 6a: Per-image relative curves ---
   619	    levels_range = list(range(1, max_level + 1))
   620	
   621	    for img_id in image_ids:
   622	        fig, ax = plt.subplots(figsize=(10, 6))
   623	
   624	        nr_vals = [img_rel_data[img_id][lv]["RelNodeRecall"] for lv in levels_range]
   625	        tr_vals = [img_rel_data[img_id][lv]["RelTripletRecall"] for lv in levels_range]
   626	        sg_vals = [img_rel_data[img_id][lv]["RelSGScore"] for lv in levels_range]
   627	
   628	        ax.plot(
   629	            levels_range,
   630	            nr_vals,
   631	            "o-",
   632	            label="RelNodeRecall",
   633	            color="#4C72B0",
   634	            markersize=4,
   635	        )
   636	        ax.plot(
   637	            levels_range,
   638	            tr_vals,
   639	            "s-",
   640	            label="RelTripletRecall",
   641	            color="#DD8452",
   642	            markersize=4,
   643	        )
   644	        ax.plot(
   645	            levels_range,
   646	            sg_vals,
   647	            "D-",
   648	            label="RelSGScore",
   649	            color="#55A868",
   650	            markersize=4,
   651	            linewidth=2,
   652	        )
   653	
   654	        # Horizontal threshold line
   655	        ax.axhline(
   656	            y=main_rel_threshold,
   657	            color="gray",
   658	            linestyle=":",
   659	            alpha=0.7,
   660	            label=f"Threshold={main_rel_threshold}",
   661	        )
   662	
   663	        # Vertical lines for stops
   664	        rel_stop = rel_oracle_map[img_id]
   665	        ax.axvline(
   666	            x=rel_stop,
   667	            color="green",
   668	            linestyle="--",
   669	            alpha=0.8,
   670	            label=f"RelOracle(0.8)={rel_stop}",
   671	        )
   672	
   673	        baseline_stop_val = baseline_map.get(img_id, max_level)
   674	        ax.axvline(
   675	            x=baseline_stop_val,
   676	            color="orange",
   677	            linestyle="--",
   678	            alpha=0.8,
   679	            label=f"Baseline={baseline_stop_val}",
   680	        )
   681	
   682	        llm4b_stop_val = llm_4b_stop.get(img_id, max_level)
   683	        ax.axvline(
   684	            x=llm4b_stop_val,
   685	            color="red",
   686	            linestyle="--",
   687	            alpha=0.8,
   688	            label=f"LLM-4b={llm4b_stop_val}",
   689	        )
   690	
   691	        ax.set_xlabel("Resolution Level")
   692	        ax.set_ylabel("Relative Metric Value")
   693	        ax.set_title(f"Relative Saturation Curves - Image {img_id}")
   694	        ax.set_xlim(0.5, max_level + 0.5)
   695	        ax.set_ylim(-0.05, 1.10)
   696	        ax.set_xticks(levels_range)
   697	        ax.legend(fontsize=8, loc="lower right")
   698	        ax.grid(True, alpha=0.3)
   699	        fig.tight_layout()
   700	        fig.savefig(figures_root / f"relative_curve_{img_id}.png", dpi=150)
   701	        plt.close(fig)
   702	
   703	    print(f"  Saved {len(image_ids)} per-image relative curve figures")
   704	
   705	    # --- Figure 6b: Method comparison MAE (horizontal bar) ---
   706	    fig, ax = plt.subplots(figsize=(12, 8))
   707	    method_names = [r["method"] for r in reversed(comparison_rows)]
   708	    maes = [float(r["mean_abs_error"]) for r in reversed(comparison_rows)]
   709	    early_rates = [float(r["early_stop_rate"]) for r in reversed(comparison_rows)]
   710	    late_rates = [float(r["late_stop_rate"]) for r in reversed(comparison_rows)]
   711	
   712	    colors = []
   713	    for er, lr in zip(early_rates, late_rates, strict=True):
   714	        if er > lr:
   715	            colors.append("#e8746a")  # red-ish = tends early
   716	        elif lr > er:
   717	            colors.append("#6a9fe8")  # blue-ish = tends late
   718	        else:
   719	            colors.append("#999999")  # neutral
   720	
   721	    y_pos = np.arange(len(method_names))
   722	    bars = ax.barh(y_pos, maes, color=colors, edgecolor="black", linewidth=0.5)
   723	
   724	    for bar, er, lr in zip(bars, early_rates, late_rates, strict=True):
   725	        w = bar.get_width()
   726	        label = f"E{er:.0%}/L{lr:.0%}"
   727	        ax.text(
   728	            w + 0.15,
   729	            bar.get_y() + bar.get_height() / 2,
   730	            label,
   731	            va="center",
   732	            fontsize=7,
   733	        )
   734	
   735	    ax.set_yticks(y_pos)
   736	    ax.set_yticklabels(method_names, fontsize=8)
   737	    ax.set_xlabel("Mean Absolute Error (levels)")
   738	    ax.set_title(f"Method Comparison: MAE vs RelOracle({main_rel_threshold})")
   739	
   740	    from matplotlib.patches import Patch
   741	
   742	    legend_elements = [
   743	        Patch(facecolor="#e8746a", edgecolor="black", label="Tends early"),
   744	        Patch(facecolor="#6a9fe8", edgecolor="black", label="Tends late"),
   745	        Patch(facecolor="#999999", edgecolor="black", label="Balanced"),
   746	    ]
   747	    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
   748	    ax.grid(True, alpha=0.3, axis="x")
   749	    fig.tight_layout()
   750	    fig.savefig(figures_root / "relative_method_comparison_mae.png", dpi=150)
   751	    plt.close(fig)
   752	    print("  Saved relative_method_comparison_mae.png")
   753	
   754	    # --- Figure 6c: RelOracle vs GT Oracle scatter ---
   755	    fig, ax = plt.subplots(figsize=(8, 8))
   756	
   757	    gt_stops = [old_oracle_map[i] for i in image_ids]
   758	    rel_stops = [rel_oracle_map[i] for i in image_ids]
   759	
   760	    ax.scatter(gt_stops, rel_stops, s=80, c="#4C72B0", edgecolors="black", zorder=5)
   761	
   762	    # Annotate points with image_id
   763	    for img_id, gx, ry in zip(image_ids, gt_stops, rel_stops, strict=True):
   764	        ax.annotate(
   765	            str(img_id),
   766	            (gx, ry),
   767	            textcoords="offset points",
   768	            xytext=(6, 6),
   769	            fontsize=7,
   770	            alpha=0.8,
   771	        )
   772	
   773	    # Reference line (y=x)
   774	    ax.plot([0, max_level + 1], [0, max_level + 1], "k--", alpha=0.3, label="y=x")
   775	
   776	    # Mean lines
   777	    ax.axvline(
   778	        x=gt_mean,
   779	        color="orange",
   780	        linestyle=":",
   781	        alpha=0.7,
   782	        label=f"GT mean={gt_mean:.1f}",
   783	    )
   784	    ax.axhline(
   785	        y=rel_mean,
   786	        color="green",
   787	        linestyle=":",
   788	        alpha=0.7,
   789	        label=f"Rel mean={rel_mean:.1f}",
   790	    )
   791	
   792	    ax.set_xlabel("GT Oracle(0.5) Stop Level")
   793	    ax.set_ylabel(f"RelOracle({main_rel_threshold}) Stop Level")
   794	    ax.set_title("Relative vs Absolute Oracle Stop Comparison")
   795	    ax.set_xlim(0, max_level + 1)
   796	    ax.set_ylim(0, max_level + 1)
   797	    ax.set_aspect("equal")
   798	    ax.legend(fontsize=9)
   799	    ax.grid(True, alpha=0.3)
   800	    fig.tight_layout()
   801	    fig.savefig(figures_root / "relative_vs_absolute_oracle.png", dpi=150)
   802	    plt.close(fig)
   803	    print("  Saved relative_vs_absolute_oracle.png")
   804	
   805	    # ===================================================================
   806	    # Write research report
   807	    # ===================================================================
   808	    print()
   809	    print("=" * 60)
   810	    print("Writing research report")
   811	    print("=" * 60)
   812	
   813	    report_path = (
   814	        REPO_ROOT / ".claude" / "docs" / "research" / "m1-relative-saturation.md"
   815	    )
   816	    report_path.parent.mkdir(parents=True, exist_ok=True)
   817	
   818	    lines: list[str] = []
   819	    lines.append("# Milestone-1: Relative Saturation Analysis (Plan B)")
   820	    lines.append("")
   821	    lines.append("Generated by `scripts/15_relative_saturation.py`")
   822	    lines.append("")
   823	
   824	    lines.append("## Definition")
   825	    lines.append("")
   826	    lines.append(
   827	        "**Relative saturation** measures convergence of EGTR output across "
   828	        "resolution levels, using each image's own Level-20 (full-resolution) "
   829	        "output as the reference (pseudo-GT), instead of the dataset's true "
   830	        "ground truth scene graph."
   831	    )
   832	    lines.append("")
   833	    lines.append("- `RelNodeRecall(t) = |nodes(t) ∩ nodes(20)| / |nodes(20)|`")
   834	    lines.append(
   835	        "- `RelTripletRecall(t) = |triplets(t) ∩ triplets(20)| / |triplets(20)|`"
   836	    )
   837	    lines.append("- `RelSGScore(t) = 0.5 * RelNodeRecall + 0.5 * RelTripletRecall`")
   838	    lines.append("- By construction, `RelSGScore(20) = 1.0`")
   839	    lines.append("")
   840	    lines.append(
   841	        "**Caveat**: This measures CONVERGENCE/STABILITY of extraction, "
   842	        "NOT correctness. Correctness is reported separately via the "
   843	        "paper-protocol R@K (see `egtr-repro.md`)."
   844	    )
   845	    lines.append("")
   846	
   847	    lines.append("## Non-degeneracy Check")
   848	    lines.append("")
   849	    lines.append(
   850	        f"- GT Oracle(0.5): {gt_saturated}/{len(image_ids)} images saturate "
   851	        f"(mean stop = {gt_mean:.1f})"
   852	    )
   853	    lines.append(
   854	        f"- RelOracle(0.8): {rel_saturated}/{len(image_ids)} images saturate "
   855	        f"(mean stop = {rel_mean:.1f})"
   856	    )
   857	    lines.append(
   858	        f"- With relative reference, "
   859	        f"{'ALL' if rel_saturated == len(image_ids) else str(rel_saturated)} "
   860	        f"images now have a reachable and meaningful saturation point."
   861	    )
   862	    lines.append("")
   863	
   864	    lines.append("## RelOracle vs GT Oracle Comparison")
   865	    lines.append("")
   866	    lines.append("| image_id | GT_Oracle_0.5 | RelOracle_0.8 | Diff |")
   867	    lines.append("|----------|---------------|---------------|------|")
   868	    for img_id in image_ids:
   869	        gt_s = old_oracle_map[img_id]
   870	        rel_s = rel_oracle_map[img_id]
   871	        lines.append(f"| {img_id} | {gt_s} | {rel_s} | {rel_s - gt_s:+d} |")
   872	    lines.append("")
   873	    lines.append(
   874	        f"Mean diff (Rel - GT): {np.mean(diffs):+.1f} levels. "
   875	        f"Relative saturation generally occurs "
   876	        f"{'EARLIER' if np.mean(diffs) < 0 else 'LATER'} than absolute."
   877	    )
   878	    lines.append("")
   879	
   880	    lines.append("## Method Comparison vs RelOracle(0.8) (sorted by MAE)")
   881	    lines.append("")
   882	    lines.append(
   883	        "| Method | Mean Stop | Mean Oracle | MAE | Median AE | Early% | Late% | Pearson r |"
   884	    )
   885	    lines.append(
   886	        "|--------|-----------|-------------|-----|-----------|--------|-------|-----------|"
   887	    )
   888	    for r in comparison_rows:
   889	        lines.append(
   890	            f"| {r['method']} | {r['mean_stop']} | {r['mean_rel_oracle_stop']} | "
   891	            f"{r['mean_abs_error']} | {r['median_abs_error']} | "
   892	            f"{r['early_stop_rate']} | {r['late_stop_rate']} | {r['pearson_r']} |"
   893	        )
   894	    lines.append("")
   895	
   896	    # Find best method
   897	    best = comparison_rows[0]
   898	    lines.append(
   899	        f"**Best method**: `{best['method']}` with MAE={best['mean_abs_error']}"
   900	    )
   901	    lines.append("")
   902	
   903	    lines.append("## Node vs Relation Saturation Gap (Relative Reference)")
   904	    lines.append("")
   905	    lines.append(
   906	        f"Threshold for both: 0.9. Mean gap (trip_first - node_first): "
   907	        f"{mean_gap:+.1f} levels."
   908	    )
   909	    lines.append("")
   910	    lines.append("| image_id | NodeFirst@0.9 | TripFirst@0.9 | Gap |")
   911	    lines.append("|----------|---------------|---------------|-----|")
   912	    for img_id, nf, tf, g in zip(
   913	        image_ids, node_first_levels, trip_first_levels, gaps, strict=True
   914	    ):
   915	        lines.append(f"| {img_id} | {nf} | {tf} | {g:+d} |")
   916	    lines.append("")
   917	
   918	    lines.append("## Figures")
   919	    lines.append("")
   920	    lines.append("- `outputs/figures/relative_curve_{image_id}.png` - per-image curves")
   921	    lines.append(
   922	        "- `outputs/figures/relative_method_comparison_mae.png` - MAE bar chart"
   923	    )
   924	    lines.append(
   925	        "- `outputs/figures/relative_vs_absolute_oracle.png` - scatter comparison"
   926	    )
   927	    lines.append("")
   928	
   929	    report_text = "\n".join(lines) + "\n"
   930	    report_path.write_text(report_text)
   931	    print(f"  Saved {report_path}")
   932	
   933	    # ===================================================================
   934	    # Summary
   935	    # ===================================================================
   936	    print()
   937	    print("=" * 60)
   938	    print("DONE. Files created:")
   939	    print("=" * 60)
   940	    print(f"  {metrics_root / 'relative_metrics.csv'}")
   941	    print(f"  {metrics_root / 'relative_oracle_stop.csv'}")
   942	    print(f"  {metrics_root / 'method_comparison_relative.csv'}")
   943	    for img_id in image_ids:
   944	        print(f"  {figures_root / f'relative_curve_{img_id}.png'}")
   945	    print(f"  {figures_root / 'relative_method_comparison_mae.png'}")
   946	    print(f"  {figures_root / 'relative_vs_absolute_oracle.png'}")
   947	    print(f"  {report_path}")
   948	
   949	
   950	if __name__ == "__main__":
   951	    main()

codex
I’ve got the main saturation and aggregation logic. I’m now checking the upstream generation/metric/stop scripts because some “bug” symptoms can originate from how their inputs are produced or cached.
exec
/usr/bin/zsh -lc 'nl -ba scripts/06_compute_sg_metrics.py' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
     1	"""Compute scene graph metrics per image per level."""
     2	
     3	import csv
     4	import json
     5	import sys
     6	from pathlib import Path
     7	
     8	REPO_ROOT = Path(__file__).resolve().parents[1]
     9	sys.path.insert(0, str(REPO_ROOT / "src"))
    10	
    11	from milestone1.config import load_config, resolve
    12	from milestone1.metrics import (
    13	    mean_confidence,
    14	    node_precision_recall_f1,
    15	    sg_score,
    16	    triplet_precision_recall_f1,
    17	)
    18	
    19	
    20	def main() -> None:
    21	    cfg = load_config()
    22	    node_w = cfg["metrics"]["sg_score"]["node_weight"]
    23	    rel_w = cfg["metrics"]["sg_score"]["relation_weight"]
    24	    num_levels = cfg["resolution_levels"]["num_levels"]
    25	
    26	    selected_path = resolve(cfg["paths"]["selected_images"])
    27	    with open(selected_path) as f:
    28	        image_ids = json.load(f)
    29	
    30	    pred_root = resolve(cfg["paths"]["egtr_predictions"])
    31	    gt_root = resolve(cfg["paths"]["gt_scene_graphs"])
    32	    metrics_root = resolve(cfg["paths"]["metrics"])
    33	    metrics_root.mkdir(parents=True, exist_ok=True)
    34	
    35	    fieldnames = [
    36	        "image_id", "level",
    37	        "node_precision", "node_recall", "node_f1",
    38	        "triplet_precision", "triplet_recall", "triplet_f1",
    39	        "sg_score", "mean_object_confidence", "mean_relation_confidence",
    40	    ]
    41	
    42	    all_rows = []
    43	
    44	    for img_id in image_ids:
    45	        # Load GT
    46	        gt_path = gt_root / f"{img_id}.json"
    47	        with open(gt_path) as f:
    48	            gt = json.load(f)
    49	
    50	        # Per-image CSV
    51	        img_csv_path = metrics_root / f"{img_id}.csv"
    52	        img_rows = []
    53	
    54	        for level_idx in range(1, num_levels + 1):
    55	            level_name = f"level_{level_idx:02d}"
    56	            pred_path = pred_root / str(img_id) / f"{level_name}.json"
    57	
    58	            with open(pred_path) as f:
    59	                pred = json.load(f)
    60	
    61	            np_, nr, nf = node_precision_recall_f1(
    62	                pred["objects"], gt["objects"]
    63	            )
    64	            tp, tr, tf = triplet_precision_recall_f1(
    65	                pred["relationships"], gt["relationships"]
    66	            )
    67	            sgs = sg_score(nr, tr, node_w, rel_w)
    68	            moc = mean_confidence(pred["objects"])
    69	            mrc = mean_confidence(pred["relationships"])
    70	
    71	            row = {
    72	                "image_id": img_id,
    73	                "level": level_idx,
    74	                "node_precision": round(np_, 4),
    75	                "node_recall": round(nr, 4),
    76	                "node_f1": round(nf, 4),
    77	                "triplet_precision": round(tp, 4),
    78	                "triplet_recall": round(tr, 4),
    79	                "triplet_f1": round(tf, 4),
    80	                "sg_score": round(sgs, 4),
    81	                "mean_object_confidence": round(moc, 4),
    82	                "mean_relation_confidence": round(mrc, 4),
    83	            }
    84	            img_rows.append(row)
    85	
    86	        # Write per-image CSV
    87	        with open(img_csv_path, "w", newline="") as f:
    88	            writer = csv.DictWriter(f, fieldnames=fieldnames)
    89	            writer.writeheader()
    90	            writer.writerows(img_rows)
    91	
    92	        all_rows.extend(img_rows)
    93	
    94	        # Print summary for last level
    95	        last = img_rows[-1]
    96	        print(
    97	            f"  {img_id}: sg_score L20={last['sg_score']:.3f} "
    98	            f"(node_R={last['node_recall']:.3f}, "
    99	            f"trip_R={last['triplet_recall']:.3f})"
   100	        )
   101	
   102	    # Write combined CSV
   103	    combined_path = metrics_root / "all_metrics.csv"
   104	    with open(combined_path, "w", newline="") as f:
   105	        writer = csv.DictWriter(f, fieldnames=fieldnames)
   106	        writer.writeheader()
   107	        writer.writerows(all_rows)
   108	
   109	    print(f"Saved {len(all_rows)} metric rows to {combined_path}")
   110	
   111	
   112	if __name__ == "__main__":
   113	    main()

exec
/usr/bin/zsh -lc 'nl -ba scripts/02_generate_resolution_levels.py' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
     1	"""Generate degraded images at multiple resolution levels."""
     2	
     3	import json
     4	import sys
     5	from pathlib import Path
     6	
     7	from PIL import Image
     8	
     9	REPO_ROOT = Path(__file__).resolve().parents[1]
    10	sys.path.insert(0, str(REPO_ROOT / "src"))
    11	
    12	from milestone1.config import load_config, resolve
    13	
    14	
    15	def main() -> None:
    16	    cfg = load_config()
    17	    vg_root = Path(cfg["paths"]["vg_root"])
    18	    scales = cfg["resolution_levels"]["scales"]
    19	    num_levels = cfg["resolution_levels"]["num_levels"]
    20	
    21	    # Load selected images
    22	    selected_path = resolve(cfg["paths"]["selected_images"])
    23	    with open(selected_path) as f:
    24	        image_ids = json.load(f)
    25	
    26	    out_root = resolve(cfg["paths"]["degraded_images"])
    27	    print(f"Generating {num_levels} resolution levels for {len(image_ids)} images")
    28	
    29	    for img_id in image_ids:
    30	        img_path = vg_root / "images" / f"{img_id}.jpg"
    31	        if not img_path.exists():
    32	            print(f"  WARNING: {img_path} not found, skipping")
    33	            continue
    34	
    35	        img = Image.open(img_path).convert("RGB")
    36	        orig_w, orig_h = img.size
    37	
    38	        img_out_dir = out_root / str(img_id)
    39	        img_out_dir.mkdir(parents=True, exist_ok=True)
    40	
    41	        for level_idx, scale in enumerate(scales, start=1):
    42	            level_name = f"level_{level_idx:02d}"
    43	            out_path = img_out_dir / f"{level_name}.jpg"
    44	
    45	            # Skip if already exists (idempotent reuse)
    46	            if out_path.exists():
    47	                continue
    48	
    49	            if scale >= 1.0:
    50	                # Original resolution — just save as-is
    51	                degraded = img.copy()
    52	            else:
    53	                # Downscale then upscale back to original size
    54	                small_w = max(1, int(orig_w * scale))
    55	                small_h = max(1, int(orig_h * scale))
    56	                small = img.resize((small_w, small_h), Image.BILINEAR)
    57	                degraded = small.resize((orig_w, orig_h), Image.BILINEAR)
    58	
    59	            degraded.save(out_path, quality=95)
    60	
    61	        print(f"  {img_id}: {num_levels} levels generated ({orig_w}x{orig_h})")
    62	
    63	    print("Done.")
    64	
    65	
    66	if __name__ == "__main__":
    67	    main()

exec
/usr/bin/zsh -lc 'nl -ba scripts/07_compute_oracle_stop.py' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
     1	"""Compute oracle stopping points based on sg_score thresholds and gain-based saturation."""
     2	
     3	import csv
     4	import json
     5	import math
     6	import sys
     7	from pathlib import Path
     8	
     9	REPO_ROOT = Path(__file__).resolve().parents[1]
    10	sys.path.insert(0, str(REPO_ROOT / "src"))
    11	
    12	from milestone1.config import load_config, resolve
    13	
    14	
    15	def main() -> None:
    16	    cfg = load_config()
    17	    thresholds = cfg["metrics"]["oracle_thresholds"]
    18	    gain_epsilon = cfg["metrics"]["gain_epsilon"]
    19	    gain_k = cfg["metrics"]["gain_k"]
    20	
    21	    selected_path = resolve(cfg["paths"]["selected_images"])
    22	    with open(selected_path) as f:
    23	        image_ids = json.load(f)
    24	
    25	    metrics_root = resolve(cfg["paths"]["metrics"])
    26	
    27	    fieldnames = ["image_id"] + [f"oracle_{t}" for t in thresholds] + ["gain_stop"]
    28	    rows = []
    29	
    30	    for img_id in image_ids:
    31	        img_csv = metrics_root / f"{img_id}.csv"
    32	        with open(img_csv) as f:
    33	            reader = csv.DictReader(f)
    34	            levels = list(reader)
    35	
    36	        row: dict[str, object] = {"image_id": img_id}
    37	
    38	        # Threshold-based oracle: min level with sg_score >= threshold
    39	        for t in thresholds:
    40	            stop_level = float("nan")
    41	            for lv in levels:
    42	                if float(lv["sg_score"]) >= t:
    43	                    stop_level = int(lv["level"])
    44	                    break
    45	            if math.isnan(stop_level):
    46	                stop_level = 20
    47	            row[f"oracle_{t}"] = stop_level
    48	
    49	        # Gain-based saturation: min level where gain < epsilon for k consecutive
    50	        sg_scores = [float(lv["sg_score"]) for lv in levels]
    51	        gains = [0.0] + [
    52	            sg_scores[i] - sg_scores[i - 1] for i in range(1, len(sg_scores))
    53	        ]
    54	
    55	        gain_stop = float("nan")
    56	        consecutive = 0
    57	        for i, g in enumerate(gains):
    58	            if abs(g) < gain_epsilon:
    59	                consecutive += 1
    60	                if consecutive >= gain_k:
    61	                    gain_stop = i + 1 - (gain_k - 1)  # first level of the run
    62	                    break
    63	            else:
    64	                consecutive = 0
    65	        if math.isnan(gain_stop):
    66	            gain_stop = 20
    67	        row["gain_stop"] = int(gain_stop)
    68	
    69	        rows.append(row)
    70	        main_t = cfg["metrics"]["main_oracle_threshold"]
    71	        print(
    72	            f"  {img_id}: oracle_{main_t}={row[f'oracle_{main_t}']}, "
    73	            f"gain_stop={row['gain_stop']}"
    74	        )
    75	
    76	    # Save
    77	    out_path = metrics_root / "oracle_stop.csv"
    78	    with open(out_path, "w", newline="") as f:
    79	        writer = csv.DictWriter(f, fieldnames=fieldnames)
    80	        writer.writeheader()
    81	        writer.writerows(rows)
    82	
    83	    print(f"Saved oracle stop points to {out_path}")
    84	
    85	
    86	if __name__ == "__main__":
    87	    main()

exec
/usr/bin/zsh -lc 'nl -ba scripts/run_focused_100.py' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
     1	"""Focused LLM+VLM evaluation for 100-image scale-up.
     2	
     3	Runs ONLY the focused configs:
     4	  - LLM-SGG: 4b h5 rel=on conf=on, 9b h5 rel=on conf=on
     5	  - VLM: single mode, 4b and 9b
     6	
     7	Order: all 4b (LLM + VLM single) then all 9b (LLM + VLM single).
     8	Reuses cached per-level JSON files (skip-existing).
     9	"""
    10	
    11	import csv
    12	import json
    13	import sys
    14	import time
    15	from pathlib import Path
    16	
    17	REPO_ROOT = Path(__file__).resolve().parents[1]
    18	sys.path.insert(0, str(REPO_ROOT / "src"))
    19	
    20	from milestone1.config import load_config, resolve
    21	from milestone1.llm import smoke_test_ollama
    22	from milestone1.llm_engine import (
    23	    build_prompt,
    24	    build_sequence,
    25	    make_config_slug,
    26	    score_with_config,
    27	)
    28	from milestone1.vlm import score_vlm_single
    29	
    30	
    31	def model_slug(model: str) -> str:
    32	    return model.replace(":", "-").replace("/", "-")
    33	
    34	
    35	def run_llm_focused(
    36	    config_tuple: tuple[str, int, bool, bool],
    37	    image_ids: list[int],
    38	    chat_endpoint: str,
    39	    pred_root: Path,
    40	    output_root: Path,
    41	    start_level: int,
    42	    num_levels: int,
    43	    sat_threshold: float,
    44	    timeout: int,
    45	    input_obj_threshold: float,
    46	    input_rel_threshold: float,
    47	    input_top_k_objects: int,
    48	    input_top_k_relationships: int,
    49	) -> list[dict]:
    50	    """Run one LLM config across all images. Returns stop-level rows."""
    51	    model, history_length, relation_on, confidence_on = config_tuple
    52	    slug = make_config_slug(model, history_length, relation_on, confidence_on)
    53	    prompt = build_prompt(relation_on, confidence_on)
    54	
    55	    print(f"\n--- LLM Config: {slug} ---")
    56	
    57	    total_calls = 0
    58	    cached_calls = 0
    59	    failed_calls = 0
    60	    stop_rows = []
    61	
    62	    for img_idx, img_id in enumerate(image_ids):
    63	        out_dir = output_root / slug / str(img_id)
    64	        out_dir.mkdir(parents=True, exist_ok=True)
    65	
    66	        stop_level = num_levels
    67	
    68	        for level_idx in range(start_level, num_levels + 1):
    69	            level_name = f"level_{level_idx:02d}"
    70	            out_path = out_dir / f"{level_name}.json"
    71	
    72	            if out_path.exists():
    73	                with open(out_path) as f:
    74	                    result = json.load(f)
    75	                cached_calls += 1
    76	            else:
    77	                sequence = build_sequence(
    78	                    image_id=img_id,
    79	                    current_level=level_idx,
    80	                    history_length=history_length,
    81	                    pred_root=pred_root,
    82	                    relation_on=relation_on,
    83	                    confidence_on=confidence_on,
    84	                    input_obj_threshold=input_obj_threshold,
    85	                    input_rel_threshold=input_rel_threshold,
    86	                    input_top_k_objects=input_top_k_objects,
    87	                    input_top_k_relationships=input_top_k_relationships,
    88	                )
    89	
    90	                t0 = time.time()
    91	                result = score_with_config(
    92	                    sequence, prompt, model, chat_endpoint, timeout
    93	                )
    94	                elapsed = time.time() - t0
    95	                total_calls += 1
    96	
    97	                if "error" in result.get("reason", "") or "parse" in result.get(
    98	                    "reason", ""
    99	                ):
   100	                    failed_calls += 1
   101	
   102	                with open(out_path, "w") as f:
   103	                    json.dump(result, f, indent=2)
   104	
   105	                print(
   106	                    f"  [{img_idx + 1}/{len(image_ids)}] {img_id} L{level_idx:02d}: "
   107	                    f"sat={result['saturation_score']:.2f} ({elapsed:.1f}s)"
   108	                )
   109	
   110	            if result["saturation_score"] >= sat_threshold and stop_level == num_levels:
   111	                stop_level = level_idx
   112	
   113	        stop_rows.append(
   114	            {
   115	                "config_slug": slug,
   116	                "model": model,
   117	                "history_length": history_length,
   118	                "relation_on": relation_on,
   119	                "confidence_on": confidence_on,
   120	                "image_id": img_id,
   121	                "stop_level": stop_level,
   122	            }
   123	        )
   124	
   125	    print(
   126	        f"  {slug}: new_calls={total_calls}, cached={cached_calls}, "
   127	        f"failed={failed_calls}"
   128	    )
   129	    return stop_rows
   130	
   131	
   132	def run_vlm_focused(
   133	    image_ids: list[int],
   134	    model: str,
   135	    chat_endpoint: str,
   136	    degraded_root: Path,
   137	    output_root: Path,
   138	    start_level: int,
   139	    num_levels: int,
   140	    max_image_side: int,
   141	    timeout: int,
   142	    sat_threshold: float,
   143	) -> list[dict]:
   144	    """Run VLM single-mode scoring for one model across all images."""
   145	    slug = model_slug(model)
   146	    mode = "single"
   147	    total_calls = 0
   148	    cached_calls = 0
   149	    failed_calls = 0
   150	    stop_rows = []
   151	
   152	    print(f"\n--- VLM {model} single ---")
   153	
   154	    for img_idx, img_id in enumerate(image_ids):
   155	        out_dir = output_root / slug / mode / str(img_id)
   156	        out_dir.mkdir(parents=True, exist_ok=True)
   157	
   158	        stop_k1 = num_levels
   159	        stop_k2 = num_levels
   160	        consecutive_count = 0
   161	
   162	        for level_idx in range(start_level, num_levels + 1):
   163	            level_name = f"level_{level_idx:02d}"
   164	            out_path = out_dir / f"{level_name}.json"
   165	
   166	            if out_path.exists():
   167	                with open(out_path) as f:
   168	                    result = json.load(f)
   169	                cached_calls += 1
   170	            else:
   171	                t0 = time.time()
   172	                img_path = str(degraded_root / str(img_id) / f"{level_name}.jpg")
   173	                result = score_vlm_single(
   174	                    img_path, model, chat_endpoint, max_image_side, timeout
   175	                )
   176	                elapsed = time.time() - t0
   177	                total_calls += 1
   178	
   179	                if "error" in result.get("reason", ""):
   180	                    failed_calls += 1
   181	
   182	                with open(out_path, "w") as f:
   183	                    json.dump(result, f, indent=2)
   184	
   185	                print(
   186	                    f"  [{img_idx + 1}/{len(image_ids)}] {img_id} L{level_idx:02d} "
   187	                    f"single: sat={result['saturation_score']:.2f} ({elapsed:.1f}s)"
   188	                )
   189	
   190	            score = result["saturation_score"]
   191	            if score >= sat_threshold:
   192	                if stop_k1 == num_levels:
   193	                    stop_k1 = level_idx
   194	                consecutive_count += 1
   195	                if consecutive_count >= 2 and stop_k2 == num_levels:
   196	                    stop_k2 = level_idx
   197	            else:
   198	                consecutive_count = 0
   199	
   200	        stop_rows.append(
   201	            {
   202	                "image_id": img_id,
   203	                "model": model,
   204	                "mode": mode,
   205	                "k": 1,
   206	                "stop_level": stop_k1,
   207	            }
   208	        )
   209	        stop_rows.append(
   210	            {
   211	                "image_id": img_id,
   212	                "model": model,
   213	                "mode": mode,
   214	                "k": 2,
   215	                "stop_level": stop_k2,
   216	            }
   217	        )
   218	
   219	    print(
   220	        f"  {model} single: new_calls={total_calls}, cached={cached_calls}, "
   221	        f"failed={failed_calls}"
   222	    )
   223	    return stop_rows
   224	
   225	
   226	def main() -> None:
   227	    cfg = load_config()
   228	    llm_cfg = cfg["llm"]
   229	    vlm_cfg = cfg["vlm"]
   230	    chat_endpoint = cfg["ollama_chat_endpoint"]
   231	    start_level = llm_cfg["start_level"]
   232	    sat_threshold = llm_cfg["saturation_threshold"]
   233	    timeout_llm = llm_cfg["request_timeout"]
   234	    timeout_vlm = vlm_cfg["request_timeout"]
   235	    num_levels = cfg["resolution_levels"]["num_levels"]
   236	    max_image_side = vlm_cfg["max_image_side"]
   237	
   238	    input_obj_threshold = llm_cfg["input_object_conf_threshold"]
   239	    input_rel_threshold = llm_cfg["input_relation_conf_threshold"]
   240	    input_top_k_objects = llm_cfg["input_top_k_objects"]
   241	    input_top_k_relationships = llm_cfg["input_top_k_relationships"]
   242	
   243	    selected_path = resolve(cfg["paths"]["selected_images"])
   244	    with open(selected_path) as f:
   245	        image_ids = json.load(f)
   246	
   247	    pred_root = resolve(cfg["paths"]["egtr_predictions"])
   248	    degraded_root = resolve(cfg["paths"]["degraded_images"])
   249	    ablation_root = resolve("outputs/llm_ablation")
   250	    vlm_output_root = resolve("outputs/vlm_scores")
   251	    metrics_root = resolve(cfg["paths"]["metrics"])
   252	
   253	    # Focused configs: only 4b and 9b base (h5, rel_on, conf_on)
   254	    focused_configs = [
   255	        ("qwen3.5:4b", 5, True, True),
   256	        ("qwen3.5:9b", 5, True, True),
   257	    ]
   258	
   259	    all_llm_stop_rows: list[dict] = []
   260	    all_vlm_stop_rows: list[dict] = []
   261	
   262	    for config_tuple in focused_configs:
   263	        model = config_tuple[0]
   264	
   265	        # Smoke test
   266	        print(f"\n{'=' * 60}")
   267	        print(f"Model: {model} (LLM + VLM single)")
   268	        print(f"{'=' * 60}")
   269	        print(f"Smoke-testing Ollama ({model}) ...")
   270	
   271	        generate_endpoint = chat_endpoint.replace("/api/chat", "/api/generate")
   272	        if not smoke_test_ollama(model, generate_endpoint, timeout=30):
   273	            print(f"ERROR: smoke test failed for {model}. Skipping.")
   274	            continue
   275	        print(f"Smoke test passed for {model}.")
   276	
   277	        # LLM first (lighter calls)
   278	        llm_rows = run_llm_focused(
   279	            config_tuple=config_tuple,
   280	            image_ids=image_ids,
   281	            chat_endpoint=chat_endpoint,
   282	            pred_root=pred_root,
   283	            output_root=ablation_root,
   284	            start_level=start_level,
   285	            num_levels=num_levels,
   286	            sat_threshold=sat_threshold,
   287	            timeout=timeout_llm,
   288	            input_obj_threshold=input_obj_threshold,
   289	            input_rel_threshold=input_rel_threshold,
   290	            input_top_k_objects=input_top_k_objects,
   291	            input_top_k_relationships=input_top_k_relationships,
   292	        )
   293	        all_llm_stop_rows.extend(llm_rows)
   294	
   295	        # VLM single (heavier calls)
   296	        vlm_rows = run_vlm_focused(
   297	            image_ids=image_ids,
   298	            model=model,
   299	            chat_endpoint=chat_endpoint,
   300	            degraded_root=degraded_root,
   301	            output_root=vlm_output_root,
   302	            start_level=start_level,
   303	            num_levels=num_levels,
   304	            max_image_side=max_image_side,
   305	            timeout=timeout_vlm,
   306	            sat_threshold=sat_threshold,
   307	        )
   308	        all_vlm_stop_rows.extend(vlm_rows)
   309	
   310	    # Save focused LLM stop CSV (append-safe: overwrite with all data)
   311	    # We need to also preserve existing ablation configs not in focused set
   312	    existing_ablation_path = metrics_root / "llm_ablation_stop.csv"
   313	    existing_llm_rows: list[dict] = []
   314	    if existing_ablation_path.exists():
   315	        with open(existing_ablation_path) as f:
   316	            reader = csv.DictReader(f)
   317	            for r in reader:
   318	                # Keep non-focused configs
   319	                slug = r["config_slug"]
   320	                if slug not in [make_config_slug(*c) for c in focused_configs]:
   321	                    existing_llm_rows.append(r)
   322	
   323	    combined_llm_rows = existing_llm_rows + all_llm_stop_rows
   324	    fieldnames_llm = [
   325	        "config_slug",
   326	        "model",
   327	        "history_length",
   328	        "relation_on",
   329	        "confidence_on",
   330	        "image_id",
   331	        "stop_level",
   332	    ]
   333	    with open(existing_ablation_path, "w", newline="") as f:
   334	        writer = csv.DictWriter(f, fieldnames=fieldnames_llm)
   335	        writer.writeheader()
   336	        writer.writerows(combined_llm_rows)
   337	    print(f"\nSaved LLM ablation stop CSV to {existing_ablation_path}")
   338	
   339	    # Save focused VLM stop CSV
   340	    # Preserve existing non-single or non-focused model entries
   341	    existing_vlm_path = metrics_root / "vlm_stop.csv"
   342	    existing_vlm_rows: list[dict] = []
   343	    focused_models = {c[0] for c in focused_configs}
   344	    if existing_vlm_path.exists():
   345	        with open(existing_vlm_path) as f:
   346	            reader = csv.DictReader(f)
   347	            for r in reader:
   348	                # Keep entries NOT in (focused_model, single)
   349	                if not (r["model"] in focused_models and r["mode"] == "single"):
   350	                    existing_vlm_rows.append(r)
   351	
   352	    combined_vlm_rows = existing_vlm_rows + all_vlm_stop_rows
   353	    fieldnames_vlm = ["image_id", "model", "mode", "k", "stop_level"]
   354	    with open(existing_vlm_path, "w", newline="") as f:
   355	        writer = csv.DictWriter(f, fieldnames=fieldnames_vlm)
   356	        writer.writeheader()
   357	        writer.writerows(combined_vlm_rows)
   358	    print(f"Saved VLM stop CSV to {existing_vlm_path}")
   359	
   360	    # Print summary
   361	    print("\n--- Summary ---")
   362	    for slug_label in [make_config_slug(*c) for c in focused_configs]:
   363	        subset = [r for r in all_llm_stop_rows if r["config_slug"] == slug_label]
   364	        if subset:
   365	            stops = [r["stop_level"] for r in subset]
   366	            print(f"  LLM {slug_label}: mean_stop={sum(stops) / len(stops):.1f}")
   367	
   368	    for model in ["qwen3.5:4b", "qwen3.5:9b"]:
   369	        for k_val in [1, 2]:
   370	            subset = [
   371	                r for r in all_vlm_stop_rows if r["model"] == model and r["k"] == k_val
   372	            ]
   373	            if subset:
   374	                stops = [r["stop_level"] for r in subset]
   375	                print(
   376	                    f"  VLM {model} single k={k_val}: "
   377	                    f"mean_stop={sum(stops) / len(stops):.1f}"
   378	                )
   379	
   380	
   381	if __name__ == "__main__":
   382	    main()

exec
/usr/bin/zsh -lc 'nl -ba scripts/08_compute_baseline_stop.py' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
     1	"""Compute baseline stopping points based on mean object confidence thresholds."""
     2	
     3	import csv
     4	import json
     5	import math
     6	import sys
     7	from pathlib import Path
     8	
     9	REPO_ROOT = Path(__file__).resolve().parents[1]
    10	sys.path.insert(0, str(REPO_ROOT / "src"))
    11	
    12	from milestone1.config import load_config, resolve
    13	
    14	
    15	def main() -> None:
    16	    cfg = load_config()
    17	    thetas = cfg["baseline"]["mean_object_conf_thresholds"]
    18	
    19	    selected_path = resolve(cfg["paths"]["selected_images"])
    20	    with open(selected_path) as f:
    21	        image_ids = json.load(f)
    22	
    23	    metrics_root = resolve(cfg["paths"]["metrics"])
    24	
    25	    fieldnames = ["image_id"] + [f"baseline_{t}" for t in thetas]
    26	    rows = []
    27	
    28	    for img_id in image_ids:
    29	        img_csv = metrics_root / f"{img_id}.csv"
    30	        with open(img_csv) as f:
    31	            reader = csv.DictReader(f)
    32	            levels = list(reader)
    33	
    34	        row: dict[str, object] = {"image_id": img_id}
    35	
    36	        for theta in thetas:
    37	            stop_level = float("nan")
    38	            for lv in levels:
    39	                if float(lv["mean_object_confidence"]) >= theta:
    40	                    stop_level = int(lv["level"])
    41	                    break
    42	            if math.isnan(stop_level):
    43	                stop_level = 20
    44	            row[f"baseline_{theta}"] = stop_level
    45	
    46	        rows.append(row)
    47	        main_t = cfg["baseline"]["mean_object_conf_threshold"]
    48	        print(
    49	            f"  {img_id}: baseline_{main_t}={row[f'baseline_{main_t}']}"
    50	        )
    51	
    52	    out_path = metrics_root / "baseline_stop.csv"
    53	    with open(out_path, "w", newline="") as f:
    54	        writer = csv.DictWriter(f, fieldnames=fieldnames)
    55	        writer.writeheader()
    56	        writer.writerows(rows)
    57	
    58	    print(f"Saved baseline stop points to {out_path}")
    59	
    60	
    61	if __name__ == "__main__":
    62	    main()

exec
/usr/bin/zsh -lc 'nl -ba src/milestone1/llm_engine.py' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
     1	"""Parameterized LLM scoring engine for ablation studies.
     2	
     3	Builds scene-graph sequences with configurable features (relation_on,
     4	confidence_on, history_length) and scores them via ollama /api/chat.
     5	"""
     6	
     7	import json
     8	from pathlib import Path
     9	from typing import Any
    10	
    11	from milestone1.llm import _parse_llm_json
    12	
    13	SATURATION_PROMPT_TEMPLATE = """You are a scene graph analysis expert. You are given a sequence of scene graphs extracted from the same image at increasing resolution levels.
    14	
    15	Your task: Judge whether the scene graph has reached "semantic saturation" — i.e., whether further increasing image resolution is unlikely to reveal meaningful new objects or relationships.
    16	
    17	INPUT: A JSON array of scene graphs at consecutive resolution levels. Each entry has:
    18	- "level": resolution level (higher = better resolution)
    19	- "objects": detected objects with labels{conf_note_objects}
    20	{rel_note}
    21	CRITERIA for saturation:
    22	1. The set of detected object categories has stabilized (few or no new categories appearing)
    23	{rel_criteria}3. The scene graph structure is converging
    24	
    25	OUTPUT: Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
    26	{{
    27	  "saturation_score": <float 0.0-1.0, where 1.0 = fully saturated>,
    28	  "stop_decision": <boolean, true if saturation_score >= 0.8>,
    29	  "reason": "<brief explanation of your judgment>"
    30	}}
    31	
    32	Scene graph sequence:
    33	"""
    34	
    35	
    36	def build_prompt(relation_on: bool, confidence_on: bool) -> str:
    37	    """Build the saturation prompt based on ablation flags."""
    38	    conf_note_objects = " and confidence scores" if confidence_on else ""
    39	
    40	    if relation_on:
    41	        if confidence_on:
    42	            rel_note = '- "relationships": detected relationships (subject -> predicate -> object) with confidence scores'
    43	        else:
    44	            rel_note = '- "relationships": detected relationships (subject -> predicate -> object)'
    45	        rel_criteria = "2. The set of detected relationships has stabilized\n"
    46	    else:
    47	        rel_note = "(No relationship information is provided in this evaluation.)"
    48	        rel_criteria = ""
    49	
    50	    if confidence_on and relation_on:
    51	        # Add confidence stability criterion
    52	        rel_criteria += "3. Confidence scores are not significantly changing\n"
    53	        # Renumber the last criterion
    54	        return SATURATION_PROMPT_TEMPLATE.format(
    55	            conf_note_objects=conf_note_objects,
    56	            rel_note=rel_note,
    57	            rel_criteria=rel_criteria.rstrip("\n") + "\n",
    58	        )
    59	
    60	    return SATURATION_PROMPT_TEMPLATE.format(
    61	        conf_note_objects=conf_note_objects,
    62	        rel_note=rel_note,
    63	        rel_criteria=rel_criteria,
    64	    )
    65	
    66	
    67	def build_sequence(
    68	    image_id: int,
    69	    current_level: int,
    70	    history_length: int,
    71	    pred_root: Path,
    72	    relation_on: bool = True,
    73	    confidence_on: bool = True,
    74	    input_obj_threshold: float = 0.3,
    75	    input_rel_threshold: float = 0.005,
    76	    input_top_k_objects: int = 20,
    77	    input_top_k_relationships: int = 30,
    78	) -> list[dict[str, Any]]:
    79	    """Build a scene graph sequence for LLM input with configurable features."""
    80	    start = max(1, current_level - history_length + 1)
    81	    sequence = []
    82	
    83	    for lv in range(start, current_level + 1):
    84	        level_name = f"level_{lv:02d}"
    85	        pred_path = pred_root / str(image_id) / f"{level_name}.json"
    86	
    87	        if not pred_path.exists():
    88	            continue
    89	
    90	        with open(pred_path) as f:
    91	            pred = json.load(f)
    92	
    93	        # Filter objects
    94	        objects = [
    95	            o for o in pred["objects"] if o["confidence"] >= input_obj_threshold
    96	        ][:input_top_k_objects]
    97	
    98	        # Build object entries
    99	        if confidence_on:
   100	            obj_entries = [
   101	                {
   102	                    "object_id": o["object_id"],
   103	                    "label": o["label"],
   104	                    "confidence": o["confidence"],
   105	                }
   106	                for o in objects
   107	            ]
   108	        else:
   109	            obj_entries = [
   110	                {
   111	                    "object_id": o["object_id"],
   112	                    "label": o["label"],
   113	                }
   114	                for o in objects
   115	            ]
   116	
   117	        entry: dict[str, Any] = {
   118	            "level": lv,
   119	            "objects": obj_entries,
   120	        }
   121	
   122	        # Add relationships if enabled
   123	        if relation_on:
   124	            relationships = [
   125	                r
   126	                for r in pred["relationships"]
   127	                if r["confidence"] >= input_rel_threshold
   128	            ][:input_top_k_relationships]
   129	
   130	            if confidence_on:
   131	                rel_entries = [
   132	                    {
   133	                        "subject_id": r["subject_id"],
   134	                        "subject_label": r["subject_label"],
   135	                        "predicate": r["predicate"],
   136	                        "object_id": r["object_id"],
   137	                        "object_label": r["object_label"],
   138	                        "confidence": r["confidence"],
   139	                    }
   140	                    for r in relationships
   141	                ]
   142	            else:
   143	                rel_entries = [
   144	                    {
   145	                        "subject_id": r["subject_id"],
   146	                        "subject_label": r["subject_label"],
   147	                        "predicate": r["predicate"],
   148	                        "object_id": r["object_id"],
   149	                        "object_label": r["object_label"],
   150	                    }
   151	                    for r in relationships
   152	                ]
   153	            entry["relationships"] = rel_entries
   154	
   155	        sequence.append(entry)
   156	
   157	    return sequence
   158	
   159	
   160	def make_config_slug(
   161	    model: str,
   162	    history_length: int,
   163	    relation_on: bool,
   164	    confidence_on: bool,
   165	) -> str:
   166	    """Create a filesystem-safe slug encoding the ablation config."""
   167	    model_tag = model.replace(":", "-").replace("/", "-")
   168	    rel_tag = "rel_on" if relation_on else "rel_off"
   169	    conf_tag = "conf_on" if confidence_on else "conf_off"
   170	    return f"{model_tag}_h{history_length}_{rel_tag}_{conf_tag}"
   171	
   172	
   173	def _do_ollama_call(
   174	    full_prompt: str,
   175	    model: str,
   176	    chat_endpoint: str,
   177	    socket_timeout: int,
   178	) -> str:
   179	    """Make the actual ollama API call (runs in a thread for wall-clock timeout)."""
   180	    import requests as req
   181	
   182	    payload = {
   183	        "model": model,
   184	        "messages": [
   185	            {"role": "user", "content": full_prompt},
   186	        ],
   187	        "stream": False,
   188	        "think": False,
   189	        "options": {
   190	            "temperature": 0.3,
   191	            "num_predict": 1024,
   192	            # Long history-5 / top-20-30 prompts reach ~9.2k tokens; ollama's
   193	            # default num_ctx=4096 truncates the prompt to 4095 and leaves zero
   194	            # room to generate, producing a 1-token stub ("Based") that fails to
   195	            # parse. Size the context to fit the largest prompt + generation.
   196	            "num_ctx": 12288,
   197	        },
   198	    }
   199	    resp = req.post(chat_endpoint, json=payload, timeout=socket_timeout)
   200	    resp.raise_for_status()
   201	    return resp.json()["message"]["content"]
   202	
   203	
   204	def score_with_config(
   205	    sequence_json: list[dict[str, Any]],
   206	    prompt: str,
   207	    model: str,
   208	    chat_endpoint: str,
   209	    timeout: int = 120,
   210	) -> dict[str, Any]:
   211	    """Score saturation for a sequence using the parameterized prompt.
   212	
   213	    Uses ollama /api/chat with think=false, stream=false.
   214	    Enforces a hard wall-clock timeout via ThreadPoolExecutor to handle
   215	    cases where ollama keeps the connection alive with slow token generation.
   216	    Returns {saturation_score, stop_decision, reason}.
   217	    """
   218	    from concurrent.futures import ThreadPoolExecutor
   219	    from concurrent.futures import TimeoutError as FuturesTimeout
   220	
   221	    full_prompt = prompt + json.dumps(sequence_json, indent=2)
   222	
   223	    try:
   224	        with ThreadPoolExecutor(max_workers=1) as pool:
   225	            future = pool.submit(
   226	                _do_ollama_call, full_prompt, model, chat_endpoint, timeout
   227	            )
   228	            # Hard wall-clock timeout: timeout + 30s grace
   229	            raw = future.result(timeout=timeout + 30)
   230	
   231	        result = _parse_llm_json(raw)
   232	
   233	        # Ensure required fields
   234	        if "saturation_score" not in result:
   235	            result["saturation_score"] = 0.0
   236	        if "stop_decision" not in result:
   237	            result["stop_decision"] = result["saturation_score"] >= 0.8
   238	        if "reason" not in result:
   239	            result["reason"] = "no reason provided"
   240	
   241	        result["saturation_score"] = max(
   242	            0.0, min(1.0, float(result["saturation_score"]))
   243	        )
   244	        result["stop_decision"] = bool(result["stop_decision"])
   245	        return result
   246	
   247	    except FuturesTimeout:
   248	        return {
   249	            "saturation_score": 0.0,
   250	            "stop_decision": False,
   251	            "reason": "error: wall-clock timeout exceeded",
   252	        }
   253	    except Exception as e:
   254	        return {
   255	            "saturation_score": 0.0,
   256	            "stop_decision": False,
   257	            "reason": f"error: {str(e)[:200]}",
   258	        }

exec
/usr/bin/zsh -lc 'nl -ba src/milestone1/llm.py' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
     1	"""Ollama LLM client for semantic saturation scoring."""
     2	
     3	import json
     4	import re
     5	from typing import Any
     6	
     7	import requests
     8	
     9	SATURATION_PROMPT = """You are a scene graph analysis expert. You are given a sequence of scene graphs extracted from the same image at increasing resolution levels.
    10	
    11	Your task: Judge whether the scene graph has reached "semantic saturation" — i.e., whether further increasing image resolution is unlikely to reveal meaningful new objects or relationships.
    12	
    13	INPUT: A JSON array of scene graphs at consecutive resolution levels. Each entry has:
    14	- "level": resolution level (higher = better resolution)
    15	- "objects": detected objects with labels and confidence scores
    16	- "relationships": detected relationships (subject -> predicate -> object) with confidence scores
    17	
    18	CRITERIA for saturation:
    19	1. The set of detected object categories has stabilized (few or no new categories appearing)
    20	2. The set of detected relationships has stabilized
    21	3. Confidence scores are not significantly changing
    22	4. The scene graph structure is converging
    23	
    24	OUTPUT: Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
    25	{
    26	  "saturation_score": <float 0.0-1.0, where 1.0 = fully saturated>,
    27	  "stop_decision": <boolean, true if saturation_score >= 0.8>,
    28	  "reason": "<brief explanation of your judgment>"
    29	}
    30	
    31	Scene graph sequence:
    32	"""
    33	
    34	
    35	def _parse_llm_json(text: str) -> dict[str, Any]:
    36	    """Parse JSON from LLM response, handling markdown fences and noise."""
    37	    # Strip markdown code fences
    38	    text = re.sub(r"```(?:json)?\s*", "", text)
    39	    text = re.sub(r"```\s*", "", text)
    40	
    41	    # Try to extract first JSON object
    42	    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    43	    if match:
    44	        try:
    45	            return json.loads(match.group())
    46	        except json.JSONDecodeError:
    47	            pass
    48	
    49	    # Try parsing the whole text
    50	    try:
    51	        return json.loads(text.strip())
    52	    except json.JSONDecodeError:
    53	        return {
    54	            "saturation_score": 0.0,
    55	            "stop_decision": False,
    56	            "reason": f"parse-error: {text[:200]}",
    57	        }
    58	
    59	
    60	def _derive_chat_endpoint(generate_endpoint: str) -> str:
    61	    """Convert generate endpoint to chat endpoint."""
    62	    return generate_endpoint.replace("/api/generate", "/api/chat")
    63	
    64	
    65	def call_ollama(
    66	    prompt: str,
    67	    model: str,
    68	    endpoint: str,
    69	    timeout: int = 120,
    70	) -> str:
    71	    """Call Ollama chat API with thinking disabled and return the response text."""
    72	    chat_endpoint = _derive_chat_endpoint(endpoint)
    73	    payload = {
    74	        "model": model,
    75	        "messages": [
    76	            {"role": "user", "content": prompt},
    77	        ],
    78	        "stream": False,
    79	        "think": False,
    80	        "options": {
    81	            "temperature": 0.3,
    82	            "num_predict": 1024,
    83	            # Avoid ollama's default num_ctx=4096 truncating long prompts to
    84	            # 4095 tokens (leaves no room to generate -> 1-token "Based" stub
    85	            # that fails JSON parsing). Fits our largest ~9.2k-token prompts.
    86	            "num_ctx": 12288,
    87	        },
    88	    }
    89	    resp = requests.post(chat_endpoint, json=payload, timeout=timeout)
    90	    resp.raise_for_status()
    91	    return resp.json()["message"]["content"]
    92	
    93	
    94	def smoke_test_ollama(model: str, endpoint: str, timeout: int = 30) -> bool:
    95	    """Quick check if Ollama is reachable and model is loaded."""
    96	    try:
    97	        result = call_ollama("Respond with just: OK", model, endpoint, timeout=timeout)
    98	        return len(result.strip()) > 0
    99	    except Exception as e:
   100	        print(f"Ollama smoke test failed: {e}")
   101	        return False
   102	
   103	
   104	def score_saturation(
   105	    sequence_json: list[dict[str, Any]],
   106	    model: str,
   107	    endpoint: str,
   108	    timeout: int = 120,
   109	) -> dict[str, Any]:
   110	    """Score semantic saturation for a scene graph sequence.
   111	
   112	    Returns {saturation_score, stop_decision, reason}.
   113	    """
   114	    prompt = SATURATION_PROMPT + json.dumps(sequence_json, indent=2)
   115	
   116	    try:
   117	        raw_response = call_ollama(prompt, model, endpoint, timeout=timeout)
   118	        result = _parse_llm_json(raw_response)
   119	
   120	        # Ensure required fields
   121	        if "saturation_score" not in result:
   122	            result["saturation_score"] = 0.0
   123	        if "stop_decision" not in result:
   124	            result["stop_decision"] = result["saturation_score"] >= 0.8
   125	        if "reason" not in result:
   126	            result["reason"] = "no reason provided"
   127	
   128	        # Clamp score to [0, 1]
   129	        result["saturation_score"] = max(
   130	            0.0, min(1.0, float(result["saturation_score"]))
   131	        )
   132	        result["stop_decision"] = bool(result["stop_decision"])
   133	
   134	        return result
   135	
   136	    except Exception as e:
   137	        return {
   138	            "saturation_score": 0.0,
   139	            "stop_decision": False,
   140	            "reason": f"error: {str(e)[:200]}",
   141	        }

exec
/usr/bin/zsh -lc 'nl -ba src/milestone1/config.py src/milestone1/vocab.py src/milestone1/gt.py' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
     1	"""Configuration loading and path resolution."""
     2	
     3	import sys
     4	from pathlib import Path
     5	from typing import Any
     6	
     7	import yaml
     8	
     9	REPO_ROOT = Path(__file__).resolve().parents[2]
    10	
    11	
    12	def load_config(config_path: str | None = None) -> dict[str, Any]:
    13	    """Load the YAML configuration file."""
    14	    if config_path is None:
    15	        config_path = str(REPO_ROOT / "configs" / "config.yaml")
    16	    with open(config_path) as f:
    17	        return yaml.safe_load(f)
    18	
    19	
    20	def resolve(path: str) -> Path:
    21	    """Resolve a config-relative path against REPO_ROOT."""
    22	    p = Path(path)
    23	    if p.is_absolute():
    24	        return p
    25	    return REPO_ROOT / p
    26	
    27	
    28	def setup_egtr_path() -> None:
    29	    """Add EGTR submodule to sys.path for imports."""
    30	    egtr_path = str(REPO_ROOT / "egtr")
    31	    if egtr_path not in sys.path:
    32	        sys.path.insert(0, egtr_path)
    33	"""Load object and predicate vocabularies from Visual Genome JSON files."""
    34	
    35	import json
    36	from pathlib import Path
    37	
    38	
    39	def load_obj_names(vg_root: str | Path) -> dict[int, str]:
    40	    """Load object category names from test.json.
    41	
    42	    Returns a dict mapping 0-indexed class id (0..149) to name.
    43	    Model output pred_classes uses 0-indexed ids.
    44	    """
    45	    with open(Path(vg_root) / "test.json") as f:
    46	        vg_data = json.load(f)
    47	    return {
    48	        c["id"] - 1: c["name"]
    49	        for c in sorted(vg_data["categories"], key=lambda x: x["id"])
    50	    }
    51	
    52	
    53	def load_pred_names(vg_root: str | Path) -> list[str]:
    54	    """Load predicate names from rel.json.
    55	
    56	    Returns a list of 50 predicate names, 0-indexed matching model output.
    57	    """
    58	    with open(Path(vg_root) / "rel.json") as f:
    59	        rel_data = json.load(f)
    60	    return rel_data["rel_categories"][1:]  # skip __background__
    61	"""Ground truth scene graph loading from Visual Genome annotations."""
    62	
    63	import json
    64	from pathlib import Path
    65	from typing import Any
    66	
    67	from pycocotools.coco import COCO
    68	
    69	
    70	def _load_coco_and_rels(
    71	    vg_root: str | Path, split: str
    72	) -> tuple[COCO, dict[str, Any], dict[str, Any]]:
    73	    """Load COCO annotations and relationship data."""
    74	    vg_root = Path(vg_root)
    75	    # Suppress COCO stdout
    76	    import io
    77	    import sys
    78	    old_stdout = sys.stdout
    79	    sys.stdout = io.StringIO()
    80	    try:
    81	        coco = COCO(str(vg_root / f"{split}.json"))
    82	    finally:
    83	        sys.stdout = old_stdout
    84	
    85	    with open(vg_root / "rel.json") as f:
    86	        rel_data = json.load(f)
    87	
    88	    return coco, rel_data, rel_data[split]
    89	
    90	
    91	def count_gt_objects_and_relationships(
    92	    vg_root: str | Path, split: str
    93	) -> list[dict[str, Any]]:
    94	    """Count GT objects and relationships per image.
    95	
    96	    Returns list of {image_id, num_objects, num_relationships}.
    97	    """
    98	    coco, rel_data, rels_split = _load_coco_and_rels(vg_root, split)
    99	
   100	    results = []
   101	    for img_info in coco.dataset["images"]:
   102	        image_id = img_info["id"]
   103	        ann_ids = coco.getAnnIds(imgIds=image_id)
   104	        num_objects = len(ann_ids)
   105	        rel_list = rels_split.get(str(image_id), [])
   106	        num_relationships = len(rel_list)
   107	        results.append({
   108	            "image_id": image_id,
   109	            "num_objects": num_objects,
   110	            "num_relationships": num_relationships,
   111	        })
   112	    return results
   113	
   114	
   115	def load_gt_scene_graph(
   116	    vg_root: str | Path, split: str, image_id: int
   117	) -> dict[str, Any]:
   118	    """Load GT scene graph for a single image.
   119	
   120	    Returns {
   121	        image_id: int,
   122	        objects: [{object_id, label, bbox: [x1,y1,x2,y2]}],
   123	        relationships: [{subject_label, predicate, object_label}]
   124	    }
   125	    """
   126	    coco, rel_data, rels_split = _load_coco_and_rels(vg_root, split)
   127	
   128	    # Object categories (0-indexed)
   129	    id2name = {cid - 1: info["name"] for cid, info in coco.cats.items()}
   130	
   131	    # Predicate names (0-indexed)
   132	    pred_names = rel_data["rel_categories"][1:]
   133	
   134	    # Objects
   135	    ann_ids = coco.getAnnIds(imgIds=image_id)
   136	    anns = coco.loadAnns(ann_ids)
   137	
   138	    objects = []
   139	    for i, ann in enumerate(anns):
   140	        cls_name = id2name[ann["category_id"] - 1]
   141	        # Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]
   142	        x, y, w, h = ann["bbox"]
   143	        bbox = [x, y, x + w, y + h]
   144	        objects.append({
   145	            "object_id": i,
   146	            "label": cls_name,
   147	            "bbox": bbox,
   148	        })
   149	
   150	    # Relationships
   151	    rel_list = rels_split.get(str(image_id), [])
   152	    relationships = []
   153	    for subj_idx, obj_idx, pred_id_1indexed in rel_list:
   154	        if subj_idx < len(anns) and obj_idx < len(anns):
   155	            subj_name = id2name[anns[subj_idx]["category_id"] - 1]
   156	            obj_name = id2name[anns[obj_idx]["category_id"] - 1]
   157	            pred_name = pred_names[pred_id_1indexed - 1]
   158	            relationships.append({
   159	                "subject_label": subj_name,
   160	                "predicate": pred_name,
   161	                "object_label": obj_name,
   162	            })
   163	
   164	    return {
   165	        "image_id": image_id,
   166	        "objects": objects,
   167	        "relationships": relationships,
   168	    }

exec
/usr/bin/zsh -lc 'nl -ba src/milestone1/vlm.py' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
     1	"""VLM (Vision-Language Model) client for saturation scoring via ollama /api/chat with images."""
     2	
     3	import base64
     4	import io
     5	import json
     6	from typing import Any
     7	
     8	import requests
     9	from PIL import Image
    10	
    11	from milestone1.llm import _parse_llm_json
    12	
    13	
    14	VLM_SINGLE_PROMPT = """You are a visual scene analysis expert. You are shown a single image captured at a certain observation quality level (resolution).
    15	
    16	Your task: Judge whether the scene has reached "semantic saturation" — i.e., whether getting a sharper or closer view of this scene is unlikely to reveal important new objects or spatial relationships that are not already visible.
    17	
    18	Consider:
    19	1. Can you clearly identify the main objects and their arrangement?
    20	2. Are there blurry or indistinct regions that might contain important objects?
    21	3. Is the image detailed enough to understand the key relationships between objects?
    22	4. Would higher resolution likely reveal new semantic content (new object types, new relationships)?
    23	
    24	OUTPUT: Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
    25	{
    26	  "saturation_score": <float 0.0-1.0, where 1.0 = fully saturated, no benefit from higher resolution>,
    27	  "stop_decision": <boolean, true if saturation_score >= 0.8>,
    28	  "reason": "<brief explanation of your judgment>"
    29	}
    30	"""
    31	
    32	VLM_SEQUENCE_PROMPT = """You are a visual scene analysis expert. You are shown a sequence of images of the SAME scene at gradually improving observation quality (resolution). The images go from lower to higher resolution (oldest to newest).
    33	
    34	Your task: Judge whether the scene has reached "semantic saturation" — i.e., whether further improving image quality is unlikely to reveal meaningful new objects or spatial relationships.
    35	
    36	Consider:
    37	1. Comparing consecutive images: are new objects or details becoming visible with each quality improvement?
    38	2. Has the rate of new discoveries slowed down or stopped?
    39	3. Are the most recent images showing essentially the same semantic content?
    40	4. Would even higher resolution likely reveal important new content?
    41	
    42	OUTPUT: Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
    43	{
    44	  "saturation_score": <float 0.0-1.0, where 1.0 = fully saturated, no benefit from higher resolution>,
    45	  "stop_decision": <boolean, true if saturation_score >= 0.8>,
    46	  "reason": "<brief explanation of your judgment>"
    47	}
    48	"""
    49	
    50	
    51	def _load_and_downscale_image(
    52	    image_path: str, max_side: int = 512
    53	) -> str:
    54	    """Load an image, downscale so longest side <= max_side, return base64 JPEG."""
    55	    img = Image.open(image_path)
    56	    # Downscale if needed
    57	    w, h = img.size
    58	    if max(w, h) > max_side:
    59	        scale = max_side / max(w, h)
    60	        new_w = int(w * scale)
    61	        new_h = int(h * scale)
    62	        img = img.resize((new_w, new_h), Image.LANCZOS)
    63	    # Convert to RGB if needed (e.g., RGBA, P mode)
    64	    if img.mode != "RGB":
    65	        img = img.convert("RGB")
    66	    # Encode as JPEG base64
    67	    buf = io.BytesIO()
    68	    img.save(buf, format="JPEG", quality=85)
    69	    return base64.b64encode(buf.getvalue()).decode("utf-8")
    70	
    71	
    72	def _do_vlm_request(
    73	    payload: dict,
    74	    chat_endpoint: str,
    75	    socket_timeout: int,
    76	) -> str:
    77	    """Make the actual VLM API call (runs in a thread for wall-clock timeout)."""
    78	    resp = requests.post(chat_endpoint, json=payload, timeout=socket_timeout)
    79	    resp.raise_for_status()
    80	    return resp.json()["message"]["content"]
    81	
    82	
    83	def call_vlm(
    84	    prompt: str,
    85	    images_b64: list[str],
    86	    model: str,
    87	    chat_endpoint: str,
    88	    timeout: int = 300,
    89	) -> str:
    90	    """Call Ollama chat API with images and return the response text.
    91	
    92	    Uses a ThreadPoolExecutor with wall-clock timeout to handle cases
    93	    where ollama keeps the connection alive with slow token generation.
    94	    """
    95	    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    96	
    97	    payload = {
    98	        "model": model,
    99	        "messages": [
   100	            {
   101	                "role": "user",
   102	                "content": prompt,
   103	                "images": images_b64,
   104	            },
   105	        ],
   106	        "stream": False,
   107	        "think": False,
   108	        "options": {
   109	            "temperature": 0.3,
   110	            "num_predict": 1024,
   111	        },
   112	    }
   113	    with ThreadPoolExecutor(max_workers=1) as pool:
   114	        future = pool.submit(_do_vlm_request, payload, chat_endpoint, timeout)
   115	        try:
   116	            return future.result(timeout=timeout + 30)
   117	        except FuturesTimeout:
   118	            raise TimeoutError(f"VLM call exceeded wall-clock timeout of {timeout + 30}s")
   119	
   120	
   121	def smoke_test_vlm(
   122	    model: str, chat_endpoint: str, timeout: int = 60
   123	) -> bool:
   124	    """Smoke test: send a tiny 8x8 red image and check we get a response."""
   125	    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
   126	    buf = io.BytesIO()
   127	    img.save(buf, format="JPEG", quality=85)
   128	    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
   129	
   130	    try:
   131	        result = call_vlm(
   132	            "What color is this image? Respond with just the color name.",
   133	            [b64],
   134	            model,
   135	            chat_endpoint,
   136	            timeout=timeout,
   137	        )
   138	        return len(result.strip()) > 0
   139	    except Exception as e:
   140	        print(f"VLM smoke test failed for {model}: {e}")
   141	        return False
   142	
   143	
   144	def score_vlm_single(
   145	    image_path: str,
   146	    model: str,
   147	    chat_endpoint: str,
   148	    max_image_side: int = 512,
   149	    timeout: int = 300,
   150	) -> dict[str, Any]:
   151	    """Score saturation using a single image (VLM single mode)."""
   152	    try:
   153	        b64 = _load_and_downscale_image(image_path, max_image_side)
   154	        raw = call_vlm(VLM_SINGLE_PROMPT, [b64], model, chat_endpoint, timeout)
   155	        result = _parse_llm_json(raw)
   156	        return _normalize_result(result)
   157	    except Exception as e:
   158	        return {
   159	            "saturation_score": 0.0,
   160	            "stop_decision": False,
   161	            "reason": f"error: {str(e)[:200]}",
   162	        }
   163	
   164	
   165	def score_vlm_sequence(
   166	    image_paths: list[str],
   167	    model: str,
   168	    chat_endpoint: str,
   169	    max_image_side: int = 512,
   170	    timeout: int = 300,
   171	) -> dict[str, Any]:
   172	    """Score saturation using a sequence of images (VLM sequence mode)."""
   173	    try:
   174	        b64_list = [
   175	            _load_and_downscale_image(p, max_image_side) for p in image_paths
   176	        ]
   177	        raw = call_vlm(
   178	            VLM_SEQUENCE_PROMPT, b64_list, model, chat_endpoint, timeout
   179	        )
   180	        result = _parse_llm_json(raw)
   181	        return _normalize_result(result)
   182	    except Exception as e:
   183	        return {
   184	            "saturation_score": 0.0,
   185	            "stop_decision": False,
   186	            "reason": f"error: {str(e)[:200]}",
   187	        }
   188	
   189	
   190	def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
   191	    """Ensure required fields exist and values are in range."""
   192	    if "saturation_score" not in result:
   193	        result["saturation_score"] = 0.0
   194	    if "stop_decision" not in result:
   195	        result["stop_decision"] = result["saturation_score"] >= 0.8
   196	    if "reason" not in result:
   197	        result["reason"] = "no reason provided"
   198	    result["saturation_score"] = max(0.0, min(1.0, float(result["saturation_score"])))
   199	    result["stop_decision"] = bool(result["stop_decision"])
   200	    return result

exec
/usr/bin/zsh -lc "sed -n '1,220p' .claude/docs/research/egtr-inference.md" in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
# EGTR Inference & GT Loading Reference

Verified by actual inference run on 2026-06-05.

## 1. Model Loading

### Config + Checkpoint

```python
from model.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor
from model.egtr import DetrForSceneGraphGeneration
from glob import glob

ARTIFACT_PATH = "checkpoint/egtr__pretrained_detr__SenseTime__deformable-detr__batch__32__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002__visual_genome__finetune/version_0"
ARCHITECTURE = "SenseTime/deformable-detr"

# 1. Load config from artifact_path (reads config.json)
config = DeformableDetrConfig.from_pretrained(ARTIFACT_PATH)
# Key config values:
#   config.num_labels = 150  (object classes, derived from id2label in config.json)
#   config.num_rel_labels = 50  (predicate classes)
#   config.num_queries = 200  (object queries)
#   config.use_freq_bias = True
#   config.with_box_refine = False

# 2. Load model architecture from HuggingFace pretrained weights (backbone only)
model = DetrForSceneGraphGeneration.from_pretrained(
    ARCHITECTURE, config=config, ignore_mismatched_sizes=True
)

# 3. Overlay finetuned checkpoint (Lightning state_dict with "model." prefix)
ckpt_path = sorted(
    glob(f"{ARTIFACT_PATH}/checkpoints/epoch=*.ckpt"),
    key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
)[-1]
# ckpt_path = ".../epoch=03-validation_loss=1.71.ckpt"

state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
for k in list(state_dict.keys()):
    state_dict[k[6:]] = state_dict.pop(k)  # strip "model." prefix from PL keys
model.load_state_dict(state_dict)

model.cuda()
model.eval()
```

### Image Preprocessing

```python
# DeformableDetrFeatureExtractor inherits from DetrFeatureExtractor
# Default: resize shortest edge to 800, max longest edge 1333
# Normalization: ImageNet mean/std (applied internally by feature_extractor)
feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
    ARCHITECTURE, size=800, max_size=1333
)

# For a single image:
from PIL import Image
img = Image.open("path/to/image.jpg").convert("RGB")
encoding = feature_extractor(images=img, return_tensors="pt")
pixel_values = encoding["pixel_values"]  # (1, 3, H, W) - resized+normalized
pixel_mask = encoding["pixel_mask"]      # (1, H, W)

# For batching: use collate_fn from train_egtr.py
# collate_fn calls feature_extractor.pad_and_create_pixel_mask() or feature_extractor.pad()
```

## 2. Forward Pass & Output Dict

```python
with torch.no_grad():
    outputs = model(
        pixel_values=pixel_values.cuda(),
        pixel_mask=pixel_mask.cuda(),
        output_attentions=False,
        output_attention_states=True,
        output_hidden_states=True,
    )
```

### Output Keys (DetrSceneGraphGenerationOutput dataclass):

| Key | Shape | Description |
|-----|-------|-------------|
| `logits` | `(B, 200, 150)` | Raw object class logits (no sigmoid/softmax applied) |
| `pred_boxes` | `(B, 200, 4)` | Normalized boxes in `(cx, cy, w, h)` format, values in [0,1] |
| `pred_rel` | `(B, 200, 200, 50)` | Relation scores **after sigmoid** (already in [0,1]) |
| `pred_connectivity` | `(B, 200, 200, 1)` | Connectivity scores **after sigmoid** (already in [0,1]) |
| `last_hidden_state` | `(B, 200, 256)` | Final decoder hidden states |
| `decoder_hidden_states` | tuple of 7 tensors | Per-layer decoder states |
| `encoder_last_hidden_state` | `(B, seq_len, 256)` | Encoder output |
| `encoder_hidden_states` | tuple of 7 tensors | Per-layer encoder states |

**Important**: `pred_rel` and `pred_connectivity` have sigmoid already applied in `forward()` (lines 522-523 of egtr.py). `logits` are raw (no activation).

## 3. Extracting Object Labels, Scores, Boxes

```python
num_labels = config.num_labels  # 150

# Per-image (index j in batch):
logits = outputs.logits[j]        # (200, 150)
pred_boxes = outputs.pred_boxes[j] # (200, 4) in cx,cy,w,h normalized

# Object scores via softmax over classes (NOT sigmoid)
probs = logits.softmax(-1)  # (200, 151) but only take [:, :num_labels]
obj_scores, pred_classes = torch.max(probs[:, :num_labels], dim=-1)
# obj_scores: (200,) confidence per query
# pred_classes: (200,) predicted class index (0-149)

# Convert boxes to [x1,y1,x2,y2] absolute coordinates:
from util.box_ops import rescale_bboxes  # or box_cxcywh_to_xyxy
# rescale_bboxes(pred_boxes, (img_w, img_h)) converts cx,cy,w,h -> x1,y1,x2,y2 * img_size

# Filter high-confidence objects:
mask = obj_scores > threshold  # e.g. 0.1
```

## 4. Extracting Ranked Relationship Triplets

The core logic from `evaluate_batch()` in `train_egtr.py` (lines 43-174):

```python
import numpy as np
from lib.pytorch_misc import argsort_desc

pred_rel = torch.clamp(outputs.pred_rel[j], 0.0, 1.0)      # (200, 200, 50)
pred_connectivity = torch.clamp(outputs.pred_connectivity[j], 0.0, 1.0)  # (200, 200, 1)

# Multiply relation scores by connectivity gate
pred_rel = torch.mul(pred_rel, pred_connectivity)  # (200, 200, 50)

# Object score outer product (weighting triplets by both subject & object confidence)
sub_ob_scores = torch.outer(obj_scores, obj_scores)  # (200, 200)
sub_ob_scores[torch.arange(200), torch.arange(200)] = 0.0  # no self-connections

### Method A: "single preds" (with constraint) - one predicate per pair
triplet_scores = torch.mul(pred_rel.max(-1)[0], sub_ob_scores)  # (200, 200)
top_inds = argsort_desc(triplet_scores.cpu().numpy())[:100]  # (K, 2) = (subj_idx, obj_idx)
# For each (s, o): best predicate = pred_rel[s, o].argmax(), score = pred_rel[s, o, best_pred]
rel_scores = pred_rel.cpu().numpy()[top_inds[:, 0], top_inds[:, 1]]  # (K, 50)

### Method B: "multiple preds" (without constraint) - any predicate per pair
triplet_scores_m = torch.mul(pred_rel, sub_ob_scores.unsqueeze(-1))  # (200, 200, 50)
pred_rel_inds = argsort_desc(triplet_scores_m.cpu().numpy())[:100]  # (K, 3) = (s, o, p)
rel_scores_m = pred_rel.cpu().numpy()[pred_rel_inds[:, 0], pred_rel_inds[:, 1], pred_rel_inds[:, 2]]
# Each row is a (subj_idx, obj_idx, pred_id) triplet with its score
```

### argsort_desc (from lib/pytorch_misc.py)

```python
def argsort_desc(scores):
    """Returns indices that sort scores descending. Output shape: (numel, ndim)"""
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))
```

### Converting to Named Triplets

```python
# subj_label = obj_categories[pred_classes[subj_idx]]
# pred_label = rel_categories[pred_id]
# obj_label  = obj_categories[pred_classes[obj_idx]]
# confidence = rel_scores_m[k]  (or rel_scores[k, pred_id] for single pred)
```

## 5. Label Vocabularies

### Object Categories (150 classes, 1-indexed in JSON, 0-indexed in model output)

**Source**: `/home/kanaoka/usr/dataset/vg/{train,val,test}.json` -> `categories` field

```python
import json

with open("/home/kanaoka/usr/dataset/vg/test.json") as f:
    vg_data = json.load(f)

# categories is a list of {"id": 1..150, "name": "...", "supercategory": "..."}
# Model output pred_classes is 0-indexed (0..149), JSON category_id is 1-indexed (1..150)
obj_categories = ["__background__"] + [c["name"] for c in sorted(vg_data["categories"], key=lambda x: x["id"])]
# obj_categories[0] = "__background__" (unused by model)
# obj_categories[1] = "airplane", ..., obj_categories[150] = "zebra"
# Model class index i corresponds to obj_categories[i+1]
# OR simply:
id2name = {c["id"] - 1: c["name"] for c in vg_data["categories"]}  # 0-indexed
```

Also built in evaluate_egtr.py:
```python
id2label = {k - 1: v["name"] for k, v in test_dataset.coco.cats.items()}  # 0..149 -> name
```

### Predicate Categories (50 classes, 1-indexed in rel.json, 0-indexed in model output)

**Source**: `/home/kanaoka/usr/dataset/vg/rel.json` -> `rel_categories` field

```python
with open("/home/kanaoka/usr/dataset/vg/rel.json") as f:
    rel_data = json.load(f)

# rel_categories[0] = "__background__" (removed in dataset code)
# rel_categories[1..50] = actual predicates
rel_categories = rel_data["rel_categories"]  # 51 items including __background__
pred_names = rel_categories[1:]  # 50 predicates, 0-indexed matching model output

# Full list: above, across, against, along, and, at, attached to, behind, belonging to,
# between, carrying, covered in, covering, eating, flying in, for, from, growing on,
# hanging from, has, holding, in, in front of, laying on, looking at, lying on, made of,
# mounted on, near, of, on, on back of, over, painted on, parked on, part of, playing,
# riding, says, sitting on, standing on, to, under, using, walking in, walking on,
# watching, wearing, wears, with
```

## 6. GT Annotation Loading

### File Structure

- `/home/kanaoka/usr/dataset/vg/train.json` / `val.json` / `test.json` - COCO-format with `images`, `annotations`, `categories`
- `/home/kanaoka/usr/dataset/vg/rel.json` - relationship annotations, keyed by split and image_id

codex
A few potential issues are emerging around threshold semantics, wall-clock timeout implementation, and “level 20 as sentinel” handling. I’m doing one more targeted pass over analysis/config helpers and searching for hard-coded assumptions before I write the structured report.
exec
/usr/bin/zsh -lc 'nl -ba src/milestone1/analysis.py' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
     1	"""Error and saturation analysis utilities for SGG-saturation PoC."""
     2	
     3	import json
     4	from pathlib import Path
     5	from typing import Any
     6	
     7	import pandas as pd
     8	from PIL import Image
     9	
    10	from milestone1.config import resolve
    11	from milestone1.metrics import _to_label_set, _to_triplet_set
    12	
    13	
    14	# --------------------------------------------------------------------------- #
    15	# Data loading helpers
    16	# --------------------------------------------------------------------------- #
    17	
    18	
    19	def load_gt(gt_root: Path, image_id: int) -> dict[str, Any]:
    20	    """Load a GT scene graph JSON."""
    21	    with open(gt_root / f"{image_id}.json") as f:
    22	        return json.load(f)
    23	
    24	
    25	def load_pred(pred_root: Path, image_id: int, level: int) -> dict[str, Any]:
    26	    """Load an EGTR prediction JSON for a given level."""
    27	    with open(pred_root / str(image_id) / f"level_{level:02d}.json") as f:
    28	        return json.load(f)
    29	
    30	
    31	def load_stop_levels(
    32	    metrics_root: Path,
    33	) -> dict[str, dict[int, int]]:
    34	    """Load stop levels for oracle@0.5, baseline@0.4, and llm.
    35	
    36	    Returns {method_name: {image_id: stop_level}}.
    37	    """
    38	    stops: dict[str, dict[int, int]] = {}
    39	
    40	    # Oracle stop (use oracle_0.5 column = main_oracle_threshold)
    41	    oracle_path = metrics_root / "oracle_stop.csv"
    42	    if oracle_path.exists():
    43	        df = pd.read_csv(oracle_path)
    44	        stops["oracle"] = dict(zip(df["image_id"], df["oracle_0.5"]))
    45	
    46	    # Baseline stop (use baseline_0.4 column)
    47	    baseline_path = metrics_root / "baseline_stop.csv"
    48	    if baseline_path.exists():
    49	        df = pd.read_csv(baseline_path)
    50	        stops["baseline"] = dict(zip(df["image_id"], df["baseline_0.4"]))
    51	
    52	    # LLM stop
    53	    llm_path = metrics_root / "llm_stop.csv"
    54	    if llm_path.exists():
    55	        df = pd.read_csv(llm_path)
    56	        stops["llm"] = dict(zip(df["image_id"], df["llm_stop"]))
    57	
    58	    return stops
    59	
    60	
    61	# --------------------------------------------------------------------------- #
    62	# Analysis 1: What was missing at the stop level
    63	# --------------------------------------------------------------------------- #
    64	
    65	
    66	def compute_missing_at_stop(
    67	    gt: dict[str, Any],
    68	    pred: dict[str, Any],
    69	) -> dict[str, Any]:
    70	    """Compare GT vs prediction at a specific level.
    71	
    72	    Returns counts and lists of missing/extra nodes and triplets.
    73	    """
    74	    gt_nodes = _to_label_set(gt["objects"])
    75	    pred_nodes = _to_label_set(pred["objects"])
    76	    gt_trips = _to_triplet_set(gt["relationships"])
    77	    pred_trips = _to_triplet_set(pred["relationships"])
    78	
    79	    missing_nodes = gt_nodes - pred_nodes
    80	    extra_nodes = pred_nodes - gt_nodes
    81	    missing_trips = gt_trips - pred_trips
    82	    extra_trips = pred_trips - gt_trips
    83	
    84	    return {
    85	        "gt_node_count": len(gt_nodes),
    86	        "pred_node_count": len(pred_nodes),
    87	        "missing_node_count": len(missing_nodes),
    88	        "extra_node_count": len(extra_nodes),
    89	        "missing_nodes": ";".join(sorted(missing_nodes)),
    90	        "extra_nodes": ";".join(sorted(extra_nodes)),
    91	        "gt_triplet_count": len(gt_trips),
    92	        "pred_triplet_count": len(pred_trips),
    93	        "missing_triplet_count": len(missing_trips),
    94	        "extra_triplet_count": len(extra_trips),
    95	        "missing_triplets": ";".join(
    96	            f"{s}-{p}-{o}" for s, p, o in sorted(missing_trips)
    97	        ),
    98	        "extra_triplets": ";".join(
    99	            f"{s}-{p}-{o}" for s, p, o in sorted(extra_trips)
   100	        ),
   101	    }
   102	
   103	
   104	# --------------------------------------------------------------------------- #
   105	# Analysis 2 & 3: Per-category / per-predicate recall by level
   106	# --------------------------------------------------------------------------- #
   107	
   108	
   109	def compute_category_recall_by_level(
   110	    image_ids: list[int],
   111	    gt_root: Path,
   112	    pred_root: Path,
   113	    num_levels: int,
   114	) -> pd.DataFrame:
   115	    """Compute per-object-category recall at each level across all images.
   116	
   117	    For each (category, level): recall = fraction of images that contain
   118	    that GT category where it was detected (i.e. in the predicted label set).
   119	    """
   120	    rows: list[dict[str, Any]] = []
   121	
   122	    for img_id in image_ids:
   123	        gt = load_gt(gt_root, img_id)
   124	        gt_cats = _to_label_set(gt["objects"])
   125	
   126	        for level in range(1, num_levels + 1):
   127	            pred = load_pred(pred_root, img_id, level)
   128	            pred_cats = _to_label_set(pred["objects"])
   129	
   130	            for cat in gt_cats:
   131	                rows.append(
   132	                    {
   133	                        "image_id": img_id,
   134	                        "level": level,
   135	                        "category": cat,
   136	                        "detected": 1 if cat in pred_cats else 0,
   137	                    }
   138	                )
   139	
   140	    df = pd.DataFrame(rows)
   141	    # Aggregate: per (category, level), fraction detected
   142	    agg = (
   143	        df.groupby(["category", "level"])["detected"]
   144	        .mean()
   145	        .reset_index()
   146	        .rename(columns={"detected": "recall"})
   147	    )
   148	    return agg
   149	
   150	
   151	def compute_predicate_recall_by_level(
   152	    image_ids: list[int],
   153	    gt_root: Path,
   154	    pred_root: Path,
   155	    num_levels: int,
   156	) -> pd.DataFrame:
   157	    """Compute per-predicate recall at each level across all images.
   158	
   159	    For each GT triplet, we check if that exact (subject, predicate, object)
   160	    is present in the predicted triplet set. Then aggregate by predicate.
   161	    """
   162	    rows: list[dict[str, Any]] = []
   163	
   164	    for img_id in image_ids:
   165	        gt = load_gt(gt_root, img_id)
   166	        gt_trips = _to_triplet_set(gt["relationships"])
   167	
   168	        for level in range(1, num_levels + 1):
   169	            pred = load_pred(pred_root, img_id, level)
   170	            pred_trips = _to_triplet_set(pred["relationships"])
   171	
   172	            for s, p, o in gt_trips:
   173	                rows.append(
   174	                    {
   175	                        "image_id": img_id,
   176	                        "level": level,
   177	                        "predicate": p,
   178	                        "triplet": f"{s}-{p}-{o}",
   179	                        "detected": 1 if (s, p, o) in pred_trips else 0,
   180	                    }
   181	                )
   182	
   183	    df = pd.DataFrame(rows)
   184	    agg = (
   185	        df.groupby(["predicate", "level"])["detected"]
   186	        .mean()
   187	        .reset_index()
   188	        .rename(columns={"detected": "recall"})
   189	    )
   190	    return agg
   191	
   192	
   193	def identify_hardest(
   194	    recall_df: pd.DataFrame,
   195	    entity_col: str,
   196	    num_levels: int,
   197	    top_k: int = 10,
   198	) -> list[str]:
   199	    """Identify the hardest categories/predicates by final-level recall.
   200	
   201	    Lower recall at the final level = harder.
   202	    """
   203	    final = recall_df[recall_df["level"] == num_levels].copy()
   204	    final = final.sort_values("recall")
   205	    return final[entity_col].head(top_k).tolist()
   206	
   207	
   208	# --------------------------------------------------------------------------- #
   209	# Analysis 4: Image feature extraction
   210	# --------------------------------------------------------------------------- #
   211	
   212	
   213	def compute_image_features(
   214	    image_id: int,
   215	    gt: dict[str, Any],
   216	    vg_images_dir: Path,
   217	    all_metrics_df: pd.DataFrame,
   218	    oracle_stop: int,
   219	) -> dict[str, Any]:
   220	    """Compute per-image features for difficulty characterization."""
   221	    # Image dimensions
   222	    img_path = vg_images_dir / f"{image_id}.jpg"
   223	    if img_path.exists():
   224	        with Image.open(img_path) as im:
   225	            img_w, img_h = im.size
   226	    else:
   227	        # Fallback: use bbox extents from GT
   228	        all_x2 = [o["bbox"][2] for o in gt["objects"]]
   229	        all_y2 = [o["bbox"][3] for o in gt["objects"]]
   230	        img_w = max(all_x2) if all_x2 else 500
   231	        img_h = max(all_y2) if all_y2 else 500
   232	
   233	    img_area = img_w * img_h
   234	
   235	    # GT object areas (from bbox)
   236	    obj_areas = []
   237	    for obj in gt["objects"]:
   238	        x1, y1, x2, y2 = obj["bbox"]
   239	        obj_areas.append(max(0, (x2 - x1)) * max(0, (y2 - y1)))
   240	
   241	    gt_node_cats = _to_label_set(gt["objects"])
   242	    gt_trip_set = _to_triplet_set(gt["relationships"])
   243	
   244	    mean_obj_area_frac = (
   245	        (sum(obj_areas) / len(obj_areas) / img_area) if obj_areas else 0.0
   246	    )
   247	    small_threshold = 0.01 * img_area
   248	    frac_small = (
   249	        sum(1 for a in obj_areas if a < small_threshold) / len(obj_areas)
   250	        if obj_areas
   251	        else 0.0
   252	    )
   253	
   254	    # Final-level metrics
   255	    img_metrics = all_metrics_df[all_metrics_df["image_id"] == image_id]
   256	    final_row = img_metrics[img_metrics["level"] == 20]
   257	    final_sg = float(final_row["sg_score"].values[0]) if len(final_row) > 0 else 0.0
   258	    final_node_r = (
   259	        float(final_row["node_recall"].values[0]) if len(final_row) > 0 else 0.0
   260	    )
   261	    final_trip_r = (
   262	        float(final_row["triplet_recall"].values[0]) if len(final_row) > 0 else 0.0
   263	    )
   264	
   265	    return {
   266	        "image_id": image_id,
   267	        "img_width": img_w,
   268	        "img_height": img_h,
   269	        "img_area": img_area,
   270	        "num_gt_objects": len(gt["objects"]),
   271	        "num_gt_object_categories": len(gt_node_cats),
   272	        "num_gt_relationships": len(gt["relationships"]),
   273	        "num_gt_unique_triplets": len(gt_trip_set),
   274	        "mean_obj_area_frac": round(mean_obj_area_frac, 4),
   275	        "frac_small_objects": round(frac_small, 4),
   276	        "final_sg_score": final_sg,
   277	        "final_node_recall": final_node_r,
   278	        "final_triplet_recall": final_trip_r,
   279	        "oracle_stop_level": oracle_stop,
   280	    }
   281	
   282	
   283	# --------------------------------------------------------------------------- #
   284	# Analysis 5: Node vs Relation saturation gap
   285	# --------------------------------------------------------------------------- #
   286	
   287	
   288	def compute_saturation_gap(
   289	    image_id: int,
   290	    all_metrics_df: pd.DataFrame,
   291	    threshold_frac: float = 0.9,
   292	) -> dict[str, Any]:
   293	    """Compute the gap between node and triplet saturation levels.
   294	
   295	    Saturation = level where recall first reaches threshold_frac * max(recall).
   296	    """
   297	    img_df = all_metrics_df[all_metrics_df["image_id"] == image_id].sort_values(
   298	        "level"
   299	    )
   300	
   301	    max_node_r = img_df["node_recall"].max()
   302	    max_trip_r = img_df["triplet_recall"].max()
   303	
   304	    node_sat_level = 20
   305	    trip_sat_level = 20
   306	
   307	    if max_node_r > 0:
   308	        target = threshold_frac * max_node_r
   309	        hits = img_df[img_df["node_recall"] >= target]
   310	        if len(hits) > 0:
   311	            node_sat_level = int(hits["level"].min())
   312	
   313	    if max_trip_r > 0:
   314	        target = threshold_frac * max_trip_r
   315	        hits = img_df[img_df["triplet_recall"] >= target]
   316	        if len(hits) > 0:
   317	            trip_sat_level = int(hits["level"].min())
   318	
   319	    return {
   320	        "image_id": image_id,
   321	        "max_node_recall": round(max_node_r, 4),
   322	        "max_triplet_recall": round(max_trip_r, 4),
   323	        "node_90pct_level": node_sat_level,
   324	        "triplet_90pct_level": trip_sat_level,
   325	        "gap": trip_sat_level - node_sat_level,
   326	    }

exec
/usr/bin/zsh -lc 'nl -ba src/milestone1/gt.py' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
     1	"""Ground truth scene graph loading from Visual Genome annotations."""
     2	
     3	import json
     4	from pathlib import Path
     5	from typing import Any
     6	
     7	from pycocotools.coco import COCO
     8	
     9	
    10	def _load_coco_and_rels(
    11	    vg_root: str | Path, split: str
    12	) -> tuple[COCO, dict[str, Any], dict[str, Any]]:
    13	    """Load COCO annotations and relationship data."""
    14	    vg_root = Path(vg_root)
    15	    # Suppress COCO stdout
    16	    import io
    17	    import sys
    18	    old_stdout = sys.stdout
    19	    sys.stdout = io.StringIO()
    20	    try:
    21	        coco = COCO(str(vg_root / f"{split}.json"))
    22	    finally:
    23	        sys.stdout = old_stdout
    24	
    25	    with open(vg_root / "rel.json") as f:
    26	        rel_data = json.load(f)
    27	
    28	    return coco, rel_data, rel_data[split]
    29	
    30	
    31	def count_gt_objects_and_relationships(
    32	    vg_root: str | Path, split: str
    33	) -> list[dict[str, Any]]:
    34	    """Count GT objects and relationships per image.
    35	
    36	    Returns list of {image_id, num_objects, num_relationships}.
    37	    """
    38	    coco, rel_data, rels_split = _load_coco_and_rels(vg_root, split)
    39	
    40	    results = []
    41	    for img_info in coco.dataset["images"]:
    42	        image_id = img_info["id"]
    43	        ann_ids = coco.getAnnIds(imgIds=image_id)
    44	        num_objects = len(ann_ids)
    45	        rel_list = rels_split.get(str(image_id), [])
    46	        num_relationships = len(rel_list)
    47	        results.append({
    48	            "image_id": image_id,
    49	            "num_objects": num_objects,
    50	            "num_relationships": num_relationships,
    51	        })
    52	    return results
    53	
    54	
    55	def load_gt_scene_graph(
    56	    vg_root: str | Path, split: str, image_id: int
    57	) -> dict[str, Any]:
    58	    """Load GT scene graph for a single image.
    59	
    60	    Returns {
    61	        image_id: int,
    62	        objects: [{object_id, label, bbox: [x1,y1,x2,y2]}],
    63	        relationships: [{subject_label, predicate, object_label}]
    64	    }
    65	    """
    66	    coco, rel_data, rels_split = _load_coco_and_rels(vg_root, split)
    67	
    68	    # Object categories (0-indexed)
    69	    id2name = {cid - 1: info["name"] for cid, info in coco.cats.items()}
    70	
    71	    # Predicate names (0-indexed)
    72	    pred_names = rel_data["rel_categories"][1:]
    73	
    74	    # Objects
    75	    ann_ids = coco.getAnnIds(imgIds=image_id)
    76	    anns = coco.loadAnns(ann_ids)
    77	
    78	    objects = []
    79	    for i, ann in enumerate(anns):
    80	        cls_name = id2name[ann["category_id"] - 1]
    81	        # Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]
    82	        x, y, w, h = ann["bbox"]
    83	        bbox = [x, y, x + w, y + h]
    84	        objects.append({
    85	            "object_id": i,
    86	            "label": cls_name,
    87	            "bbox": bbox,
    88	        })
    89	
    90	    # Relationships
    91	    rel_list = rels_split.get(str(image_id), [])
    92	    relationships = []
    93	    for subj_idx, obj_idx, pred_id_1indexed in rel_list:
    94	        if subj_idx < len(anns) and obj_idx < len(anns):
    95	            subj_name = id2name[anns[subj_idx]["category_id"] - 1]
    96	            obj_name = id2name[anns[obj_idx]["category_id"] - 1]
    97	            pred_name = pred_names[pred_id_1indexed - 1]
    98	            relationships.append({
    99	                "subject_label": subj_name,
   100	                "predicate": pred_name,
   101	                "object_label": obj_name,
   102	            })
   103	
   104	    return {
   105	        "image_id": image_id,
   106	        "objects": objects,
   107	        "relationships": relationships,
   108	    }

exec
/usr/bin/zsh -lc 'nl -ba src/milestone1/vocab.py' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
     1	"""Load object and predicate vocabularies from Visual Genome JSON files."""
     2	
     3	import json
     4	from pathlib import Path
     5	
     6	
     7	def load_obj_names(vg_root: str | Path) -> dict[int, str]:
     8	    """Load object category names from test.json.
     9	
    10	    Returns a dict mapping 0-indexed class id (0..149) to name.
    11	    Model output pred_classes uses 0-indexed ids.
    12	    """
    13	    with open(Path(vg_root) / "test.json") as f:
    14	        vg_data = json.load(f)
    15	    return {
    16	        c["id"] - 1: c["name"]
    17	        for c in sorted(vg_data["categories"], key=lambda x: x["id"])
    18	    }
    19	
    20	
    21	def load_pred_names(vg_root: str | Path) -> list[str]:
    22	    """Load predicate names from rel.json.
    23	
    24	    Returns a list of 50 predicate names, 0-indexed matching model output.
    25	    """
    26	    with open(Path(vg_root) / "rel.json") as f:
    27	        rel_data = json.load(f)
    28	    return rel_data["rel_categories"][1:]  # skip __background__

exec
/usr/bin/zsh -lc 'nl -ba src/milestone1/config.py' in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
     1	"""Configuration loading and path resolution."""
     2	
     3	import sys
     4	from pathlib import Path
     5	from typing import Any
     6	
     7	import yaml
     8	
     9	REPO_ROOT = Path(__file__).resolve().parents[2]
    10	
    11	
    12	def load_config(config_path: str | None = None) -> dict[str, Any]:
    13	    """Load the YAML configuration file."""
    14	    if config_path is None:
    15	        config_path = str(REPO_ROOT / "configs" / "config.yaml")
    16	    with open(config_path) as f:
    17	        return yaml.safe_load(f)
    18	
    19	
    20	def resolve(path: str) -> Path:
    21	    """Resolve a config-relative path against REPO_ROOT."""
    22	    p = Path(path)
    23	    if p.is_absolute():
    24	        return p
    25	    return REPO_ROOT / p
    26	
    27	
    28	def setup_egtr_path() -> None:
    29	    """Add EGTR submodule to sys.path for imports."""
    30	    egtr_path = str(REPO_ROOT / "egtr")
    31	    if egtr_path not in sys.path:
    32	        sys.path.insert(0, egtr_path)

exec
/usr/bin/zsh -lc "rg -n \"20|max_level|num_levels|oracle_0\\.5|baseline_0\\.4|0\\.8|saturation_threshold|stop_decision|num_ctx|think|timeout|ThreadPoolExecutor|write_text|mkdir|open\\(["'^)]*, '"\\\"w\\\"\" src/milestone1 scripts configs/config.yaml" in /home/kanaoka/usr/github/milestone_1
 succeeded in 0ms:
configs/config.yaml:28:  num_levels: 20
configs/config.yaml:36:    - 0.20
configs/config.yaml:47:    - 0.80
configs/config.yaml:55:  top_k_objects: 20
configs/config.yaml:81:  saturation_threshold: 0.7
configs/config.yaml:86:  input_top_k_objects: 20
configs/config.yaml:88:  request_timeout: 120
configs/config.yaml:92:# ollama chat endpoint (supports text + images via "images" field, "think": false)
configs/config.yaml:102:  saturation_threshold: 0.7
configs/config.yaml:105:  request_timeout: 300
src/milestone1/vlm.py:27:  "stop_decision": <boolean, true if saturation_score >= 0.8>,
src/milestone1/vlm.py:45:  "stop_decision": <boolean, true if saturation_score >= 0.8>,
src/milestone1/vlm.py:75:    socket_timeout: int,
src/milestone1/vlm.py:77:    """Make the actual VLM API call (runs in a thread for wall-clock timeout)."""
src/milestone1/vlm.py:78:    resp = requests.post(chat_endpoint, json=payload, timeout=socket_timeout)
src/milestone1/vlm.py:88:    timeout: int = 300,
src/milestone1/vlm.py:92:    Uses a ThreadPoolExecutor with wall-clock timeout to handle cases
src/milestone1/vlm.py:95:    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
src/milestone1/vlm.py:107:        "think": False,
src/milestone1/vlm.py:113:    with ThreadPoolExecutor(max_workers=1) as pool:
src/milestone1/vlm.py:114:        future = pool.submit(_do_vlm_request, payload, chat_endpoint, timeout)
src/milestone1/vlm.py:116:            return future.result(timeout=timeout + 30)
src/milestone1/vlm.py:118:            raise TimeoutError(f"VLM call exceeded wall-clock timeout of {timeout + 30}s")
src/milestone1/vlm.py:122:    model: str, chat_endpoint: str, timeout: int = 60
src/milestone1/vlm.py:136:            timeout=timeout,
src/milestone1/vlm.py:149:    timeout: int = 300,
src/milestone1/vlm.py:154:        raw = call_vlm(VLM_SINGLE_PROMPT, [b64], model, chat_endpoint, timeout)
src/milestone1/vlm.py:160:            "stop_decision": False,
src/milestone1/vlm.py:161:            "reason": f"error: {str(e)[:200]}",
src/milestone1/vlm.py:170:    timeout: int = 300,
src/milestone1/vlm.py:178:            VLM_SEQUENCE_PROMPT, b64_list, model, chat_endpoint, timeout
src/milestone1/vlm.py:185:            "stop_decision": False,
src/milestone1/vlm.py:186:            "reason": f"error: {str(e)[:200]}",
src/milestone1/vlm.py:194:    if "stop_decision" not in result:
src/milestone1/vlm.py:195:        result["stop_decision"] = result["saturation_score"] >= 0.8
src/milestone1/vlm.py:199:    result["stop_decision"] = bool(result["stop_decision"])
src/milestone1/egtr_infer.py:38:        top_k_objects: int = 20,
src/milestone1/egtr_infer.py:105:        logits = outputs.logits[0]  # (200, 150)
src/milestone1/egtr_infer.py:106:        pred_boxes = outputs.pred_boxes[0]  # (200, 4) cx,cy,w,h normalized
src/milestone1/egtr_infer.py:115:        )  # (200, 4)
src/milestone1/egtr_infer.py:142:        pred_rel = torch.clamp(outputs.pred_rel[0], 0.0, 1.0)  # (200,200,50)
src/milestone1/egtr_infer.py:145:        )  # (200,200,1)
src/milestone1/egtr_infer.py:149:        sub_ob_scores = torch.outer(obj_scores, obj_scores)  # (200, 200)
src/milestone1/egtr_infer.py:156:        triplet_scores = pred_rel.max(-1)[0] * sub_ob_scores  # (200, 200)
src/milestone1/analysis.py:40:    # Oracle stop (use oracle_0.5 column = main_oracle_threshold)
src/milestone1/analysis.py:44:        stops["oracle"] = dict(zip(df["image_id"], df["oracle_0.5"]))
src/milestone1/analysis.py:46:    # Baseline stop (use baseline_0.4 column)
src/milestone1/analysis.py:50:        stops["baseline"] = dict(zip(df["image_id"], df["baseline_0.4"]))
src/milestone1/analysis.py:113:    num_levels: int,
src/milestone1/analysis.py:126:        for level in range(1, num_levels + 1):
src/milestone1/analysis.py:155:    num_levels: int,
src/milestone1/analysis.py:168:        for level in range(1, num_levels + 1):
src/milestone1/analysis.py:196:    num_levels: int,
src/milestone1/analysis.py:203:    final = recall_df[recall_df["level"] == num_levels].copy()
src/milestone1/analysis.py:256:    final_row = img_metrics[img_metrics["level"] == 20]
src/milestone1/analysis.py:304:    node_sat_level = 20
src/milestone1/analysis.py:305:    trip_sat_level = 20
scripts/09_llm_saturation_scoring.py:86:    sat_threshold = llm_cfg["saturation_threshold"]
scripts/09_llm_saturation_scoring.py:87:    timeout = llm_cfg["request_timeout"]
scripts/09_llm_saturation_scoring.py:94:    num_levels = cfg["resolution_levels"]["num_levels"]
scripts/09_llm_saturation_scoring.py:107:    if not smoke_test_ollama(model, endpoint, timeout=30):
scripts/09_llm_saturation_scoring.py:119:        llm_in_dir.mkdir(parents=True, exist_ok=True)
scripts/09_llm_saturation_scoring.py:120:        llm_out_dir.mkdir(parents=True, exist_ok=True)
scripts/09_llm_saturation_scoring.py:122:        llm_stop_level = 20  # default if never triggered
scripts/09_llm_saturation_scoring.py:124:        for level_idx in range(start_level, num_levels + 1):
scripts/09_llm_saturation_scoring.py:141:            with open(input_path, "w") as f:
scripts/09_llm_saturation_scoring.py:147:                sequence, model, endpoint, timeout=timeout
scripts/09_llm_saturation_scoring.py:157:            with open(output_path, "w") as f:
scripts/09_llm_saturation_scoring.py:163:                and llm_stop_level == 20
scripts/09_llm_saturation_scoring.py:170:                f"stop={result['stop_decision']} "
scripts/09_llm_saturation_scoring.py:182:    with open(out_path, "w", newline="") as f:
scripts/07_compute_oracle_stop.py:46:                stop_level = 20
scripts/07_compute_oracle_stop.py:66:            gain_stop = 20
scripts/07_compute_oracle_stop.py:78:    with open(out_path, "w", newline="") as f:
src/milestone1/llm.py:27:  "stop_decision": <boolean, true if saturation_score >= 0.8>,
src/milestone1/llm.py:55:            "stop_decision": False,
src/milestone1/llm.py:56:            "reason": f"parse-error: {text[:200]}",
src/milestone1/llm.py:69:    timeout: int = 120,
src/milestone1/llm.py:71:    """Call Ollama chat API with thinking disabled and return the response text."""
src/milestone1/llm.py:79:        "think": False,
src/milestone1/llm.py:83:            # Avoid ollama's default num_ctx=4096 truncating long prompts to
src/milestone1/llm.py:86:            "num_ctx": 12288,
src/milestone1/llm.py:89:    resp = requests.post(chat_endpoint, json=payload, timeout=timeout)
src/milestone1/llm.py:94:def smoke_test_ollama(model: str, endpoint: str, timeout: int = 30) -> bool:
src/milestone1/llm.py:97:        result = call_ollama("Respond with just: OK", model, endpoint, timeout=timeout)
src/milestone1/llm.py:108:    timeout: int = 120,
src/milestone1/llm.py:112:    Returns {saturation_score, stop_decision, reason}.
src/milestone1/llm.py:117:        raw_response = call_ollama(prompt, model, endpoint, timeout=timeout)
src/milestone1/llm.py:123:        if "stop_decision" not in result:
src/milestone1/llm.py:124:            result["stop_decision"] = result["saturation_score"] >= 0.8
src/milestone1/llm.py:132:        result["stop_decision"] = bool(result["stop_decision"])
src/milestone1/llm.py:139:            "stop_decision": False,
src/milestone1/llm.py:140:            "reason": f"error: {str(e)[:200]}",
scripts/06_compute_sg_metrics.py:24:    num_levels = cfg["resolution_levels"]["num_levels"]
scripts/06_compute_sg_metrics.py:33:    metrics_root.mkdir(parents=True, exist_ok=True)
scripts/06_compute_sg_metrics.py:54:        for level_idx in range(1, num_levels + 1):
scripts/06_compute_sg_metrics.py:87:        with open(img_csv_path, "w", newline="") as f:
scripts/06_compute_sg_metrics.py:97:            f"  {img_id}: sg_score L20={last['sg_score']:.3f} "
scripts/06_compute_sg_metrics.py:104:    with open(combined_path, "w", newline="") as f:
scripts/run_focused_100.py:42:    num_levels: int,
scripts/run_focused_100.py:44:    timeout: int,
scripts/run_focused_100.py:64:        out_dir.mkdir(parents=True, exist_ok=True)
scripts/run_focused_100.py:66:        stop_level = num_levels
scripts/run_focused_100.py:68:        for level_idx in range(start_level, num_levels + 1):
scripts/run_focused_100.py:92:                    sequence, prompt, model, chat_endpoint, timeout
scripts/run_focused_100.py:102:                with open(out_path, "w") as f:
scripts/run_focused_100.py:110:            if result["saturation_score"] >= sat_threshold and stop_level == num_levels:
scripts/run_focused_100.py:139:    num_levels: int,
scripts/run_focused_100.py:141:    timeout: int,
scripts/run_focused_100.py:156:        out_dir.mkdir(parents=True, exist_ok=True)
scripts/run_focused_100.py:158:        stop_k1 = num_levels
scripts/run_focused_100.py:159:        stop_k2 = num_levels
scripts/run_focused_100.py:162:        for level_idx in range(start_level, num_levels + 1):
scripts/run_focused_100.py:174:                    img_path, model, chat_endpoint, max_image_side, timeout
scripts/run_focused_100.py:182:                with open(out_path, "w") as f:
scripts/run_focused_100.py:192:                if stop_k1 == num_levels:
scripts/run_focused_100.py:195:                if consecutive_count >= 2 and stop_k2 == num_levels:
scripts/run_focused_100.py:232:    sat_threshold = llm_cfg["saturation_threshold"]
scripts/run_focused_100.py:233:    timeout_llm = llm_cfg["request_timeout"]
scripts/run_focused_100.py:234:    timeout_vlm = vlm_cfg["request_timeout"]
scripts/run_focused_100.py:235:    num_levels = cfg["resolution_levels"]["num_levels"]
scripts/run_focused_100.py:272:        if not smoke_test_ollama(model, generate_endpoint, timeout=30):
scripts/run_focused_100.py:285:            num_levels=num_levels,
scripts/run_focused_100.py:287:            timeout=timeout_llm,
scripts/run_focused_100.py:303:            num_levels=num_levels,
scripts/run_focused_100.py:305:            timeout=timeout_vlm,
scripts/run_focused_100.py:333:    with open(existing_ablation_path, "w", newline="") as f:
scripts/run_focused_100.py:354:    with open(existing_vlm_path, "w", newline="") as f:
scripts/eval_egtr_subset.py:115:    parser.add_argument("--num_queries", type=int, default=200)
scripts/eval_egtr_subset.py:234:    with open(output_path, "w") as f:
src/milestone1/llm_engine.py:28:  "stop_decision": <boolean, true if saturation_score >= 0.8>,
src/milestone1/llm_engine.py:76:    input_top_k_objects: int = 20,
src/milestone1/llm_engine.py:177:    socket_timeout: int,
src/milestone1/llm_engine.py:179:    """Make the actual ollama API call (runs in a thread for wall-clock timeout)."""
src/milestone1/llm_engine.py:188:        "think": False,
src/milestone1/llm_engine.py:192:            # Long history-5 / top-20-30 prompts reach ~9.2k tokens; ollama's
src/milestone1/llm_engine.py:193:            # default num_ctx=4096 truncates the prompt to 4095 and leaves zero
src/milestone1/llm_engine.py:196:            "num_ctx": 12288,
src/milestone1/llm_engine.py:199:    resp = req.post(chat_endpoint, json=payload, timeout=socket_timeout)
src/milestone1/llm_engine.py:209:    timeout: int = 120,
src/milestone1/llm_engine.py:213:    Uses ollama /api/chat with think=false, stream=false.
src/milestone1/llm_engine.py:214:    Enforces a hard wall-clock timeout via ThreadPoolExecutor to handle
src/milestone1/llm_engine.py:216:    Returns {saturation_score, stop_decision, reason}.
src/milestone1/llm_engine.py:218:    from concurrent.futures import ThreadPoolExecutor
src/milestone1/llm_engine.py:224:        with ThreadPoolExecutor(max_workers=1) as pool:
src/milestone1/llm_engine.py:226:                _do_ollama_call, full_prompt, model, chat_endpoint, timeout
src/milestone1/llm_engine.py:228:            # Hard wall-clock timeout: timeout + 30s grace
src/milestone1/llm_engine.py:229:            raw = future.result(timeout=timeout + 30)
src/milestone1/llm_engine.py:236:        if "stop_decision" not in result:
src/milestone1/llm_engine.py:237:            result["stop_decision"] = result["saturation_score"] >= 0.8
src/milestone1/llm_engine.py:244:        result["stop_decision"] = bool(result["stop_decision"])
src/milestone1/llm_engine.py:250:            "stop_decision": False,
src/milestone1/llm_engine.py:251:            "reason": "error: wall-clock timeout exceeded",
src/milestone1/llm_engine.py:256:            "stop_decision": False,
src/milestone1/llm_engine.py:257:            "reason": f"error: {str(e)[:200]}",
scripts/10_plot_results.py:29:    sat_t = cfg["llm"]["saturation_threshold"]
scripts/10_plot_results.py:37:    figures_root.mkdir(parents=True, exist_ok=True)
scripts/10_plot_results.py:84:        ax.set_xlim(0.5, 20.5)
scripts/10_plot_results.py:161:    with open(summary_path, "w", newline="") as f:
scripts/11_vlm_saturation_scoring.py:3:Scores each image at levels start_level..20 using both single-image and
scripts/11_vlm_saturation_scoring.py:40:    num_levels: int,
scripts/11_vlm_saturation_scoring.py:43:    timeout: int,
scripts/11_vlm_saturation_scoring.py:57:        out_dir.mkdir(parents=True, exist_ok=True)
scripts/11_vlm_saturation_scoring.py:59:        stop_k1 = num_levels  # default: never stopped
scripts/11_vlm_saturation_scoring.py:60:        stop_k2 = num_levels
scripts/11_vlm_saturation_scoring.py:63:        for level_idx in range(start_level, num_levels + 1):
scripts/11_vlm_saturation_scoring.py:83:                        img_path, model, chat_endpoint, max_image_side, timeout
scripts/11_vlm_saturation_scoring.py:94:                        img_paths, model, chat_endpoint, max_image_side, timeout
scripts/11_vlm_saturation_scoring.py:104:                with open(out_path, "w") as f:
scripts/11_vlm_saturation_scoring.py:110:                    f"stop={result['stop_decision']} "
scripts/11_vlm_saturation_scoring.py:117:                if stop_k1 == num_levels:
scripts/11_vlm_saturation_scoring.py:120:                if consecutive_count >= 2 and stop_k2 == num_levels:
scripts/11_vlm_saturation_scoring.py:164:    sat_threshold = vlm_cfg["saturation_threshold"]
scripts/11_vlm_saturation_scoring.py:166:    timeout = vlm_cfg["request_timeout"]
scripts/11_vlm_saturation_scoring.py:167:    num_levels = cfg["resolution_levels"]["num_levels"]
scripts/11_vlm_saturation_scoring.py:189:        if not smoke_test_vlm(model, chat_endpoint, timeout=60):
scripts/11_vlm_saturation_scoring.py:217:                num_levels=num_levels,
scripts/11_vlm_saturation_scoring.py:220:                timeout=timeout,
scripts/11_vlm_saturation_scoring.py:226:    metrics_root.mkdir(parents=True, exist_ok=True)
scripts/11_vlm_saturation_scoring.py:229:    with open(csv_path, "w", newline="") as f:
scripts/01_select_vg_subset.py:45:    out_path.parent.mkdir(parents=True, exist_ok=True)
scripts/01_select_vg_subset.py:46:    with open(out_path, "w") as f:
scripts/04_convert_egtr_outputs.py:28:    num_levels = cfg["resolution_levels"]["num_levels"]
scripts/04_convert_egtr_outputs.py:34:        for level_idx in range(1, num_levels + 1):
scripts/14_error_analysis.py:43:    num_levels = cfg["resolution_levels"]["num_levels"]
scripts/14_error_analysis.py:56:    analysis_dir.mkdir(parents=True, exist_ok=True)
scripts/14_error_analysis.py:57:    figures_dir.mkdir(parents=True, exist_ok=True)
scripts/14_error_analysis.py:69:            stop_lv = stops.get(img_id, 20)
scripts/14_error_analysis.py:71:            stop_lv = max(1, min(stop_lv, num_levels))
scripts/14_error_analysis.py:123:        image_ids, gt_root, pred_root, num_levels
scripts/14_error_analysis.py:129:    hardest_cats = identify_hardest(cat_recall_df, "category", num_levels, top_k=10)
scripts/14_error_analysis.py:130:    print(f"Hardest object categories (by L20 recall): {hardest_cats}")
scripts/14_error_analysis.py:141:    ax.set_xticks(range(1, num_levels + 1))
scripts/14_error_analysis.py:153:        [1, 5, 10, 15, 20],
scripts/14_error_analysis.py:164:        image_ids, gt_root, pred_root, num_levels
scripts/14_error_analysis.py:171:        pred_recall_df, "predicate", num_levels, top_k=10
scripts/14_error_analysis.py:173:    print(f"Hardest predicates (by L20 recall): {hardest_preds}")
scripts/14_error_analysis.py:186:    ax.set_xticks(range(1, num_levels + 1))
scripts/14_error_analysis.py:200:        [1, 5, 10, 15, 20],
scripts/14_error_analysis.py:214:        oracle_lv = oracle_stops.get(img_id, 20)
scripts/14_error_analysis.py:282:    ax.set_ylabel("Final SGScore (L20)")
scripts/14_error_analysis.py:303:    ax.set_ylabel("Final SGScore (L20)")
scripts/14_error_analysis.py:324:    ax.set_ylabel("Final SGScore (L20)")
scripts/aggregate_100.py:7:4. Method comparison vs RelOracleStop(0.8) -> method_comparison_relative_100.csv
scripts/aggregate_100.py:8:5. Saturation rate table (RelOracle at 0.6/0.7/0.8/0.9)
scripts/aggregate_100.py:48:    path.parent.mkdir(parents=True, exist_ok=True)
scripts/aggregate_100.py:49:    with open(path, "w", newline="") as f:
scripts/aggregate_100.py:78:    base_dir: Path, image_id: int, start_level: int, max_level: int
scripts/aggregate_100.py:81:    for lvl in range(start_level, max_level + 1):
scripts/aggregate_100.py:95:    max_level: int,
scripts/aggregate_100.py:97:    for lvl in range(start_level, max_level + 1):
scripts/aggregate_100.py:109:    return max_level
scripts/aggregate_100.py:158:    max_level = cfg["resolution_levels"]["num_levels"]
scripts/aggregate_100.py:160:    sat_threshold = cfg["llm"]["saturation_threshold"]
scripts/aggregate_100.py:169:    figures_root.mkdir(parents=True, exist_ok=True)
scripts/aggregate_100.py:225:            scores = load_saturation_scores(slug_dir, img_id, start_level, max_level)
scripts/aggregate_100.py:227:                scores, sat_threshold, 2, start_level, max_level
scripts/aggregate_100.py:239:                    "stop_k1": abl_map[slug].get(img_id, max_level),
scripts/aggregate_100.py:240:                    "stop_k2": abl_k2_map[slug].get(img_id, max_level),
scripts/aggregate_100.py:269:                    slug_dir, img_id, start_level, max_level
scripts/aggregate_100.py:272:                for lvl in range(start_level, max_level + 1):
scripts/aggregate_100.py:282:                    fused, hybrid_threshold, k_val, start_level, max_level
scripts/aggregate_100.py:307:    rel_oracle_thresholds = [0.6, 0.7, 0.8, 0.9]
scripts/aggregate_100.py:308:    main_rel_threshold = 0.8
scripts/aggregate_100.py:310:    # Build pseudo-GT from level_20
scripts/aggregate_100.py:313:        pseudo_gt[img_id] = load_egtr_prediction(pred_root, img_id, max_level)
scripts/aggregate_100.py:322:        for lvl in range(1, max_level + 1):
scripts/aggregate_100.py:350:            stop = max_level
scripts/aggregate_100.py:351:            for lvl in range(1, max_level + 1):
scripts/aggregate_100.py:363:        sat = sum(1 for i in image_ids if rel_oracle_all[t][i] < max_level)
scripts/aggregate_100.py:366:    gt_saturated = sum(1 for i in image_ids if oracle_map[i] < max_level)
scripts/aggregate_100.py:367:    rel_saturated = sum(1 for i in image_ids if rel_oracle_map[i] < max_level)
scripts/aggregate_100.py:374:        f"  RelOracle(0.8): saturated={rel_saturated}/{n_images}, mean={rel_mean:.1f}"
scripts/aggregate_100.py:415:            method_stops = [stop_map.get(i, max_level) for i in image_ids]
scripts/aggregate_100.py:462:    # vs RelOracle(0.8)
scripts/aggregate_100.py:464:    print("STEP 6: Method comparison vs RelOracle(0.8)")
scripts/aggregate_100.py:495:        node_first = max_level
scripts/aggregate_100.py:496:        trip_first = max_level
scripts/aggregate_100.py:497:        for lvl in range(1, max_level + 1):
scripts/aggregate_100.py:503:                and node_first == max_level
scripts/aggregate_100.py:509:                and trip_first == max_level
scripts/aggregate_100.py:521:    # Figure 1: Method comparison MAE vs RelOracle(0.8) horizontal bar
scripts/aggregate_100.py:549:    ax.set_title(f"Method Comparison: MAE vs RelOracle(0.8) [N={n_images}]")
scripts/aggregate_100.py:588:        stops = [stop_map.get(i, max_level) for i in image_ids]
scripts/aggregate_100.py:595:    box_labels.extend(["oracle(0.5)", "rel_oracle(0.8)"])
scripts/aggregate_100.py:599:    cmap = plt.cm.tab20
scripts/aggregate_100.py:601:        if box_labels[i] in ("oracle(0.5)", "rel_oracle(0.8)"):
scripts/aggregate_100.py:603:            patch.set_alpha(0.8)
scripts/aggregate_100.py:641:    ax.set_ylim(0, max_level + 1)
scripts/aggregate_100.py:651:    thresholds = [0.6, 0.7, 0.8, 0.9]
scripts/aggregate_100.py:654:        sat = sum(1 for i in image_ids if rel_oracle_all[t][i] < max_level)
scripts/aggregate_100.py:670:        alpha=0.8,
scripts/aggregate_100.py:697:            for lvl in range(start_level, max_level + 1):
scripts/aggregate_100.py:717:            for lvl in range(start_level, max_level + 1):
scripts/aggregate_100.py:739:        f"  RelOracle(0.8): saturated={rel_saturated}/{n_images} ({rel_saturated / n_images:.0%}), mean={rel_mean:.1f}"
scripts/aggregate_100.py:742:    print("\n  Top-3 methods vs RelOracle(0.8):")
scripts/05_convert_vg_gt.py:24:    out_root.mkdir(parents=True, exist_ok=True)
scripts/05_convert_vg_gt.py:29:        with open(out_path, "w") as f:
scripts/15_relative_saturation.py:3:Reference = each image's own Level-20 EGTR output (pseudo-GT), NOT the
scripts/15_relative_saturation.py:8:  1. Build pseudo-GT per image from level_20.json
scripts/15_relative_saturation.py:9:  2. Compute relative metrics per (image, level 1..20)
scripts/15_relative_saturation.py:11:  4. Re-evaluate all stop methods vs RelOracle(0.8)
scripts/15_relative_saturation.py:54:    path.parent.mkdir(parents=True, exist_ok=True)
scripts/15_relative_saturation.py:55:    with open(path, "w", newline="") as f:
scripts/15_relative_saturation.py:184:    max_level = cfg["resolution_levels"]["num_levels"]  # 20
scripts/15_relative_saturation.py:189:    # For relative oracle, use threshold=0.8 as primary (stated in task)
scripts/15_relative_saturation.py:190:    rel_oracle_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
scripts/15_relative_saturation.py:191:    main_rel_threshold = 0.8
scripts/15_relative_saturation.py:196:    figures_root.mkdir(parents=True, exist_ok=True)
scripts/15_relative_saturation.py:203:    # STEP 1: Build pseudo-GT from level_20 predictions
scripts/15_relative_saturation.py:206:    print("STEP 1: Build pseudo-GT from level_20 predictions")
scripts/15_relative_saturation.py:211:        data = load_egtr_prediction(pred_root, img_id, max_level)
scripts/15_relative_saturation.py:224:    print("STEP 2: Compute relative metrics per (image, level 1..20)")
scripts/15_relative_saturation.py:233:        for lvl in range(1, max_level + 1):
scripts/15_relative_saturation.py:250:        # Print level-20 sanity check
scripts/15_relative_saturation.py:251:        sg20 = rel_sg_map[(img_id, max_level)]
scripts/15_relative_saturation.py:252:        print(f"  {img_id}: RelSGScore(20)={sg20:.4f} (should be 1.0)")
scripts/15_relative_saturation.py:285:            stop = max_level
scripts/15_relative_saturation.py:286:            for lvl in range(1, max_level + 1):
scripts/15_relative_saturation.py:299:            for lvl in range(1, max_level + 1)
scripts/15_relative_saturation.py:308:        gain_stop = max_level
scripts/15_relative_saturation.py:336:        1 for img_id in image_ids if rel_oracle_map[img_id] < max_level
scripts/15_relative_saturation.py:340:        f"at RelSGScore >= {main_rel_threshold} (vs 4/10 before with true-GT oracle_0.5)"
scripts/15_relative_saturation.py:344:    # STEP 4: Re-evaluate all stop methods vs RelOracle(0.8)
scripts/15_relative_saturation.py:348:    print("STEP 4: Re-evaluate all methods vs RelOracle(0.8)")
scripts/15_relative_saturation.py:431:    # Compute comparison stats vs RelOracle(0.8)
scripts/15_relative_saturation.py:436:        method_stops = [stop_map.get(i, max_level) for i in image_ids]
scripts/15_relative_saturation.py:490:    print("STEP 5: Compare RelOracle(0.8) vs old GT Oracle(0.5)")
scripts/15_relative_saturation.py:495:        int(r["image_id"]): int(float(r["oracle_0.5"])) for r in old_oracle_rows
scripts/15_relative_saturation.py:499:        f"\n  {'image_id':<12s} {'GT_Oracle_0.5':>14s} {'RelOracle_0.8':>14s} "
scripts/15_relative_saturation.py:511:    gt_saturated = sum(1 for i in image_ids if old_oracle_map[i] < max_level)
scripts/15_relative_saturation.py:512:    rel_saturated = sum(1 for i in image_ids if rel_oracle_map[i] < max_level)
scripts/15_relative_saturation.py:521:        f"  RelOracle(0.8):  mean={rel_mean:.1f}, saturated={rel_saturated}/{n_images}"
scripts/15_relative_saturation.py:542:        node_first = max_level
scripts/15_relative_saturation.py:543:        trip_first = max_level
scripts/15_relative_saturation.py:544:        for lvl in range(1, max_level + 1):
scripts/15_relative_saturation.py:557:                and node_first == max_level
scripts/15_relative_saturation.py:563:                and trip_first == max_level
scripts/15_relative_saturation.py:572:        # Check if image has any triplets at level 20
scripts/15_relative_saturation.py:619:    levels_range = list(range(1, max_level + 1))
scripts/15_relative_saturation.py:669:            alpha=0.8,
scripts/15_relative_saturation.py:670:            label=f"RelOracle(0.8)={rel_stop}",
scripts/15_relative_saturation.py:673:        baseline_stop_val = baseline_map.get(img_id, max_level)
scripts/15_relative_saturation.py:678:            alpha=0.8,
scripts/15_relative_saturation.py:682:        llm4b_stop_val = llm_4b_stop.get(img_id, max_level)
scripts/15_relative_saturation.py:687:            alpha=0.8,
scripts/15_relative_saturation.py:694:        ax.set_xlim(0.5, max_level + 0.5)
scripts/15_relative_saturation.py:770:            alpha=0.8,
scripts/15_relative_saturation.py:774:    ax.plot([0, max_level + 1], [0, max_level + 1], "k--", alpha=0.3, label="y=x")
scripts/15_relative_saturation.py:795:    ax.set_xlim(0, max_level + 1)
scripts/15_relative_saturation.py:796:    ax.set_ylim(0, max_level + 1)
scripts/15_relative_saturation.py:816:    report_path.parent.mkdir(parents=True, exist_ok=True)
scripts/15_relative_saturation.py:828:        "resolution levels, using each image's own Level-20 (full-resolution) "
scripts/15_relative_saturation.py:833:    lines.append("- `RelNodeRecall(t) = |nodes(t) ∩ nodes(20)| / |nodes(20)|`")
scripts/15_relative_saturation.py:835:        "- `RelTripletRecall(t) = |triplets(t) ∩ triplets(20)| / |triplets(20)|`"
scripts/15_relative_saturation.py:838:    lines.append("- By construction, `RelSGScore(20) = 1.0`")
scripts/15_relative_saturation.py:854:        f"- RelOracle(0.8): {rel_saturated}/{len(image_ids)} images saturate "
scripts/15_relative_saturation.py:866:    lines.append("| image_id | GT_Oracle_0.5 | RelOracle_0.8 | Diff |")
scripts/15_relative_saturation.py:880:    lines.append("## Method Comparison vs RelOracle(0.8) (sorted by MAE)")
scripts/15_relative_saturation.py:930:    report_path.write_text(report_text)
scripts/08_compute_baseline_stop.py:43:                stop_level = 20
scripts/08_compute_baseline_stop.py:53:    with open(out_path, "w", newline="") as f:
scripts/run_ollama_evaluation.py:45:    num_levels: int,
scripts/run_ollama_evaluation.py:48:    timeout: int,
scripts/run_ollama_evaluation.py:62:            out_dir.mkdir(parents=True, exist_ok=True)
scripts/run_ollama_evaluation.py:64:            stop_k1 = num_levels
scripts/run_ollama_evaluation.py:65:            stop_k2 = num_levels
scripts/run_ollama_evaluation.py:68:            for level_idx in range(start_level, num_levels + 1):
scripts/run_ollama_evaluation.py:82:                            img_path, model, chat_endpoint, max_image_side, timeout
scripts/run_ollama_evaluation.py:92:                            img_paths, model, chat_endpoint, max_image_side, timeout
scripts/run_ollama_evaluation.py:98:                    with open(out_path, "w") as f:
scripts/run_ollama_evaluation.py:107:                    if stop_k1 == num_levels:
scripts/run_ollama_evaluation.py:110:                    if consecutive_count >= 2 and stop_k2 == num_levels:
scripts/run_ollama_evaluation.py:132:    num_levels: int,
scripts/run_ollama_evaluation.py:134:    timeout: int,
scripts/run_ollama_evaluation.py:154:            out_dir.mkdir(parents=True, exist_ok=True)
scripts/run_ollama_evaluation.py:156:            stop_level = num_levels
scripts/run_ollama_evaluation.py:158:            for level_idx in range(start_level, num_levels + 1):
scripts/run_ollama_evaluation.py:181:                        sequence, prompt, model, chat_endpoint, timeout
scripts/run_ollama_evaluation.py:188:                    with open(out_path, "w") as f:
scripts/run_ollama_evaluation.py:198:                    and stop_level == num_levels
scripts/run_ollama_evaluation.py:224:    num_levels = cfg["resolution_levels"]["num_levels"]
scripts/run_ollama_evaluation.py:235:    metrics_root.mkdir(parents=True, exist_ok=True)
scripts/run_ollama_evaluation.py:258:    vlm_4b_ok = smoke_test_vlm(model_4b, chat_endpoint, timeout=60)
scripts/run_ollama_evaluation.py:263:    llm_4b_ok = smoke_test_ollama(model_4b, generate_endpoint, timeout=30)
scripts/run_ollama_evaluation.py:275:            num_levels=num_levels,
scripts/run_ollama_evaluation.py:278:            timeout=vlm_cfg["request_timeout"],
scripts/run_ollama_evaluation.py:279:            sat_threshold=vlm_cfg["saturation_threshold"],
scripts/run_ollama_evaluation.py:291:            num_levels=num_levels,
scripts/run_ollama_evaluation.py:292:            sat_threshold=llm_cfg["saturation_threshold"],
scripts/run_ollama_evaluation.py:293:            timeout=llm_cfg["request_timeout"],
scripts/run_ollama_evaluation.py:308:    vlm_9b_ok = smoke_test_vlm(model_9b, chat_endpoint, timeout=120)
scripts/run_ollama_evaluation.py:312:    llm_9b_ok = smoke_test_ollama(model_9b, generate_endpoint, timeout=60)
scripts/run_ollama_evaluation.py:324:            num_levels=num_levels,
scripts/run_ollama_evaluation.py:327:            timeout=vlm_cfg["request_timeout"],
scripts/run_ollama_evaluation.py:328:            sat_threshold=vlm_cfg["saturation_threshold"],
scripts/run_ollama_evaluation.py:340:            num_levels=num_levels,
scripts/run_ollama_evaluation.py:341:            sat_threshold=llm_cfg["saturation_threshold"],
scripts/run_ollama_evaluation.py:342:            timeout=llm_cfg["request_timeout"],
scripts/run_ollama_evaluation.py:353:    with open(vlm_csv, "w", newline="") as f:
scripts/run_ollama_evaluation.py:365:    with open(llm_csv, "w", newline="") as f:
scripts/03_run_egtr_inference.py:27:    num_levels = cfg["resolution_levels"]["num_levels"]
scripts/03_run_egtr_inference.py:45:    total = len(image_ids) * num_levels
scripts/03_run_egtr_inference.py:50:        img_out_dir.mkdir(parents=True, exist_ok=True)
scripts/03_run_egtr_inference.py:52:        for level_idx in range(1, num_levels + 1):
scripts/03_run_egtr_inference.py:79:            with open(out_path, "w") as f:
scripts/03_run_egtr_inference.py:87:            f"  {img_id}: {num_levels} levels done "
scripts/12_llm_ablation.py:52:    num_levels: int,
scripts/12_llm_ablation.py:54:    timeout: int,
scripts/12_llm_ablation.py:73:        out_dir.mkdir(parents=True, exist_ok=True)
scripts/12_llm_ablation.py:75:        stop_level = num_levels  # default
scripts/12_llm_ablation.py:77:        for level_idx in range(start_level, num_levels + 1):
scripts/12_llm_ablation.py:106:                    sequence, prompt, model, chat_endpoint, timeout
scripts/12_llm_ablation.py:117:                with open(out_path, "w") as f:
scripts/12_llm_ablation.py:123:                    f"stop={result['stop_decision']} "
scripts/12_llm_ablation.py:130:                and stop_level == num_levels
scripts/12_llm_ablation.py:158:    sat_threshold = llm_cfg["saturation_threshold"]
scripts/12_llm_ablation.py:159:    timeout = llm_cfg["request_timeout"]
scripts/12_llm_ablation.py:160:    num_levels = cfg["resolution_levels"]["num_levels"]
scripts/12_llm_ablation.py:190:            if not smoke_test_ollama(model, chat_endpoint.replace("/api/chat", "/api/generate"), timeout=30):
scripts/12_llm_ablation.py:217:            num_levels=num_levels,
scripts/12_llm_ablation.py:219:            timeout=timeout,
scripts/12_llm_ablation.py:228:    metrics_root.mkdir(parents=True, exist_ok=True)
scripts/12_llm_ablation.py:239:    with open(csv_path, "w", newline="") as f:
scripts/02_generate_resolution_levels.py:19:    num_levels = cfg["resolution_levels"]["num_levels"]
scripts/02_generate_resolution_levels.py:27:    print(f"Generating {num_levels} resolution levels for {len(image_ids)} images")
scripts/02_generate_resolution_levels.py:39:        img_out_dir.mkdir(parents=True, exist_ok=True)
scripts/02_generate_resolution_levels.py:61:        print(f"  {img_id}: {num_levels} levels generated ({orig_w}x{orig_h})")
scripts/13_aggregate_evaluation.py:48:    path.parent.mkdir(parents=True, exist_ok=True)
scripts/13_aggregate_evaluation.py:49:    with open(path, "w", newline="") as f:
scripts/13_aggregate_evaluation.py:80:    base_dir: Path, image_id: int, start_level: int, max_level: int
scripts/13_aggregate_evaluation.py:87:    for lvl in range(start_level, max_level + 1):
scripts/13_aggregate_evaluation.py:101:    max_level: int,
scripts/13_aggregate_evaluation.py:107:    Returns max_level if never triggered.
scripts/13_aggregate_evaluation.py:109:    for lvl in range(start_level, max_level + 1):
scripts/13_aggregate_evaluation.py:122:    return max_level
scripts/13_aggregate_evaluation.py:132:    max_level = cfg["resolution_levels"]["num_levels"]  # 20
scripts/13_aggregate_evaluation.py:134:    sat_threshold = cfg["llm"]["saturation_threshold"]  # 0.7
scripts/13_aggregate_evaluation.py:140:    figures_root.mkdir(parents=True, exist_ok=True)
scripts/13_aggregate_evaluation.py:194:            scores = load_saturation_scores(slug_dir, img_id, start_level, max_level)
scripts/13_aggregate_evaluation.py:196:                scores, sat_threshold, 2, start_level, max_level
scripts/13_aggregate_evaluation.py:204:            k1_stop = abl_map[slug].get(img_id, max_level)
scripts/13_aggregate_evaluation.py:205:            k2_stop = abl_k2_map[slug].get(img_id, max_level)
scripts/13_aggregate_evaluation.py:253:            stop = max_level
scripts/13_aggregate_evaluation.py:254:            for lvl in range(1, max_level + 1):
scripts/13_aggregate_evaluation.py:265:        saturated = sum(1 for s in oracle_stops_w if s < max_level)
scripts/13_aggregate_evaluation.py:310:                    slug_dir, img_id, start_level, max_level
scripts/13_aggregate_evaluation.py:315:                for lvl in range(start_level, max_level + 1):
scripts/13_aggregate_evaluation.py:327:                    fused, hybrid_threshold, k_val, start_level, max_level
scripts/13_aggregate_evaluation.py:408:        method_stops = [stop_map.get(i, max_level) for i in image_ids]
scripts/13_aggregate_evaluation.py:535:        stops = [stop_map.get(i, max_level) for i in image_ids]
scripts/13_aggregate_evaluation.py:547:    cmap = plt.cm.tab20
scripts/13_aggregate_evaluation.py:551:            patch.set_alpha(0.8)
scripts/13_aggregate_evaluation.py:564:            s=20,
scripts/13_aggregate_evaluation.py:583:    ax.set_ylim(0, max_level + 1)
scripts/13_aggregate_evaluation.py:596:        np.mean([baseline_map.get(i, max_level) for i in image_ids])
scripts/13_aggregate_evaluation.py:636:    ax_a.set_ylim(0, max_level + 1)
scripts/13_aggregate_evaluation.py:666:    ax_b.set_ylim(0, max_level + 1)
scripts/13_aggregate_evaluation.py:693:    ax_c.set_ylim(0, max_level + 1)
scripts/13_aggregate_evaluation.py:720:    ax_d.set_ylim(0, max_level + 1)
scripts/13_aggregate_evaluation.py:739:    report_path.parent.mkdir(parents=True, exist_ok=True)
scripts/13_aggregate_evaluation.py:772:        errors = [abs(stop_map.get(i, max_level) - oracle_map[i]) for i in image_ids]
scripts/13_aggregate_evaluation.py:856:    report_path.write_text(report_text)

**Evaluating ranking settings**

I'm thinking about the fallback approach to take the top-k. If there's no specified threshold in the config, it raises questions about the scientific validity of using zero objects. It could be intentional to ensure there's always some input in the process. Plus, should I focus on whether there's a relevant object confidence threshold? I might need to cite this as an important issue. Also, for relationship ranking, it seems that top_pairs are derived from top_k_relationships multiplied by three before filtering.
**Identifying potential bugs**

I’m noticing some potential issues in my code. First, I see that I'm importing `numpy`, `sys`, and `Path`, but they might not be used, which could be wasteful. There’s also a line constructing `id2name` that makes use of category IDs. If `ann["category_id"]` is 1-based, I should note that this means using `category_id - 1` for the key. In `scripts/02`, there’s no validation for the length of `scales` against `num_levels`, which could lead to mismatches. I need to improve robustness there! Also, I need to check the formatting style in `scripts/06`.
**Considering saturation in metrics**

I'm focusing on relative saturation gains and some potential issues. The baseline consecutive_k config is ignored, but maybe that’s alright since it’s set to 1. It seems like the “never saturated -> 20” concept might confuse reaching level 20 with never truly saturating. If an oracle or sentinel is involved, that could balance early and late saturation. But they count saturation rates for levels below max_level, which could be misleading if the first saturation happens at level 20.
**Examining text matching**

I'm trying to figure out how text should match when it contains braces. If the outer text has braces, it could lead to mismatches, especially if the inner section like `{y}` becomes invalid due to noise in the text. If the response includes surrounding prose with valid JSON and a nested object, that's unexpected and may cause failures. I think I'll need to focus on using balanced extraction to decode and scan correctly.
codex
## Summary
The core label-set metrics are internally consistent for the stated PoC choice, but the stop-level logic has several scientific interpretation hazards. The largest risks are relationship under-selection in EGTR postprocessing, premature gain-based stops, and conflating "never saturated" with "first saturated at level 20".

## Correctness bugs
- `src/milestone1/egtr_infer.py:157`, severity High: `top_pairs` keeps only `top_k_relationships * 3` global subject-object pairs before filtering to exported objects at `src/milestone1/egtr_infer.py:168-170`. If many high-ranked pairs involve non-exported queries, valid lower-ranked pairs among the selected objects are never considered, causing relationship undercount and unstable triplet recall. Suggested fix: mask `triplet_scores` to `valid_query_ids` before `argsort_desc`, or keep scanning the full ranked list until `top_k_relationships` valid relationships are found.
- `scripts/07_compute_oracle_stop.py:51`, severity High: gain-based stopping seeds `gains` with an artificial `0.0`, and `scripts/07_compute_oracle_stop.py:57-62` counts it as a real low-gain step. With `gain_k=3`, two actual small gains can produce `gain_stop=1`. Suggested fix: compute gains only for transitions `level 2..N` and define whether the stop is the current level after `k` low transitions or the first level of that plateau.
- `scripts/15_relative_saturation.py:301`, severity High: the relative gain stop repeats the same artificial-zero off-by-one bug, then returns `i + 1 - (gain_k - 1)` at `scripts/15_relative_saturation.py:314`. This can make `gain_based` saturate at level 1 before enough evidence exists. Suggested fix: align with a transition-based gain definition and add tests for flat curves, one-step gains, and NaN-containing curves.
- `scripts/15_relative_saturation.py:130-151`, severity Med: when level-20 pseudo-GT has no triplets, `RelTripletRecall` becomes `"NaN"` and `RelSGScore` silently switches from `0.5 * node + 0.5 * triplet` to node-only. This contradicts the documented formula at `scripts/15_relative_saturation.py:837` and makes images with no level-20 EGTR relations easier to saturate. Suggested fix: choose and document a fixed convention, e.g. exclude no-triplet images from relation-based RelOracle, or set empty-reference triplet recall to 1.0/0.0 explicitly and report counts.
- `scripts/aggregate_100.py:109`, severity Med: `consecutive_k_stop` returns `max_level` both for "never saturated" and "first saturated at the final level". Comparisons at `scripts/aggregate_100.py:415-417` then treat missing/never/final-hit identically. Suggested fix: carry a separate `hit: bool` or `None` sentinel and only cap to `max_level` for policy-level plots.
- `scripts/run_focused_100.py:66` and `scripts/run_focused_100.py:110-111`, severity Med: LLM `stop_level` also uses `num_levels` as both sentinel and valid final stop. The VLM logic does the same at `scripts/run_focused_100.py:158-196`. Suggested fix: store raw `stop_level_nullable` plus `capped_stop_level`.
- `src/milestone1/llm.py:132`, severity Med: `bool(result["stop_decision"])` turns the string `"false"` into `True`; same pattern in `src/milestone1/llm_engine.py:244` and `src/milestone1/vlm.py:199`. Suggested fix: accept real booleans, parse `"true"/"false"` strings explicitly, otherwise derive from the numeric threshold.

## Metric/logic issues
- `src/milestone1/metrics.py:32-37` and `src/milestone1/metrics.py:61-66`: empty-pred/empty-GT returns perfect F1 only when both sets are empty. That is a defensible convention, but if future subsets include empty GT relationships it will inflate aggregate triplet metrics. Given the current config requires `min_gt_relationships: 3` at `configs/config.yaml:23`, this is not currently a primary bug.
- `src/milestone1/metrics.py:6-18`: label-only set matching intentionally deduplicates instances. Correct for the design, but it means duplicate object categories and multiple identical relation labels are ignored, so it measures category coverage rather than scene-graph instance correctness.
- `src/milestone1/egtr_infer.py:172-184`: relationship output confidence is only gated relation confidence, while ranking at `src/milestone1/egtr_infer.py:156` uses relation confidence times subject/object confidence. That is not necessarily wrong, but downstream `mean_relation_confidence` is not the same score used for ranking.
- `src/milestone1/llm.py:27`, `src/milestone1/llm_engine.py:28`, and `src/milestone1/vlm.py:27`: prompts tell the model `stop_decision` means `saturation_score >= 0.8`, while the configured runtime threshold is `0.7` at `configs/config.yaml:81` and `configs/config.yaml:102`. Since code mostly ignores `stop_decision`, this mainly contaminates LLM calibration and explanations.
- `scripts/15_relative_saturation.py:335-340` and `scripts/aggregate_100.py:360-367`: saturation rates count only stops `< max_level`, so images that first cross threshold at level 20 are reported as unsaturated. This is only valid if "saturated" means "could have stopped before the final observation".

## Robustness
- `src/milestone1/llm_engine.py:224-229` and `src/milestone1/llm_engine.py:247-252`: the claimed hard wall-clock timeout is not actually hard. On `FutureTimeout`, exiting the `with ThreadPoolExecutor(...)` block waits for the worker thread, so a slow Ollama call can still block. Same issue in `src/milestone1/vlm.py:113-118`. Suggested fix: manage executor shutdown with `wait=False, cancel_futures=True`, or use a subprocess/process-based timeout.
- `src/milestone1/llm.py:89`: the older LLM path has only the `requests` socket timeout and no wall-clock guard, unlike `llm_engine.py`.
- `src/milestone1/llm.py:42`: `_parse_llm_json` uses a flat-object regex, so JSON with nested objects or braces inside a string can fail when surrounded by prose. Suggested fix: scan from each `{` and use `json.JSONDecoder().raw_decode`.
- `src/milestone1/llm.py:129-130`, `src/milestone1/llm_engine.py:241-243`, and `src/milestone1/vlm.py:198`: `float("NaN")` can slip through clamping behavior incorrectly. Suggested fix: require `math.isfinite(score)` before clamping; non-finite should become 0.0 and be marked invalid.
- `src/milestone1/vlm.py:97-111`: VLM requests set `think=false` but do not set `num_ctx=12288`, unlike LLM calls at `src/milestone1/llm.py:80-87` and `src/milestone1/llm_engine.py:189-197`.
- `scripts/aggregate_100.py:77-87` and `scripts/aggregate_100.py:415`: missing cached LLM score files silently become score `0.0` and then stop `max_level`, which biases methods late and hides incomplete runs. Suggested fix: validate expected cache cardinality per slug/image/level before aggregation.
- `scripts/aggregate_100.py:211-213`: config slugs are hard-coded rather than generated with `make_config_slug`, so changes to model names/history/config conventions can silently miss cache directories.
- `scripts/02_generate_resolution_levels.py:18-19` and `scripts/02_generate_resolution_levels.py:41`: no validation that `len(scales) == num_levels`; generated files can disagree with printed/evaluated level count.
- `src/milestone1/gt.py:93-97`: relationship indices only check `< len(anns)`, not `>= 0`; malformed negative indices would select from the end of the annotation list.

## Style/convention
- `src/milestone1/egtr_infer.py:3-11`: `sys`, `Path`, `numpy as np`, and `REPO_ROOT` are unused.
- `scripts/aggregate_100.py:139-144`: nested conditional expressions around `math.isnan` are hard to read and unnecessary because recall values are always floats.
- `scripts/07_compute_oracle_stop.py:45-46`, `scripts/07_compute_oracle_stop.py:65-66`, `scripts/08_compute_baseline_stop.py:42-43`, and `src/milestone1/analysis.py:256`: hard-coded `20` should use `num_levels` or a passed final-level parameter.
- `scripts/15_relative_saturation.py:161-174`: `compute_relative_metrics` is annotated as returning `dict[str, float]` but returns `"NaN"` strings.
- `src/milestone1/analysis.py:40-50`: hard-coded `oracle_0.5` and `baseline_0.4` duplicate config assumptions.

## Top 5 fixes
1. Fix EGTR relationship ranking by masking/ranking over selected valid queries before truncation, then add a regression test where valid pairs appear below the first `top_k * 3` global pairs.
2. Redefine gain-based stopping on real inter-level transitions and remove the artificial initial zero in both absolute and relative oracle scripts.
3. Represent "never saturated" separately from valid final-level stop, and update MAE/early/late/saturation-rate reporting accordingly.
4. Harden Ollama parsing/normalization/timeouts: balanced JSON extraction, explicit boolean parsing, finite score checks, and a real wall-clock timeout.
5. Make relative no-triplet pseudo-GT behavior explicit and consistent with the reported `RelSGScore` formula.
tokens used
105,594
## Summary
The core label-set metrics are internally consistent for the stated PoC choice, but the stop-level logic has several scientific interpretation hazards. The largest risks are relationship under-selection in EGTR postprocessing, premature gain-based stops, and conflating "never saturated" with "first saturated at level 20".

## Correctness bugs
- `src/milestone1/egtr_infer.py:157`, severity High: `top_pairs` keeps only `top_k_relationships * 3` global subject-object pairs before filtering to exported objects at `src/milestone1/egtr_infer.py:168-170`. If many high-ranked pairs involve non-exported queries, valid lower-ranked pairs among the selected objects are never considered, causing relationship undercount and unstable triplet recall. Suggested fix: mask `triplet_scores` to `valid_query_ids` before `argsort_desc`, or keep scanning the full ranked list until `top_k_relationships` valid relationships are found.
- `scripts/07_compute_oracle_stop.py:51`, severity High: gain-based stopping seeds `gains` with an artificial `0.0`, and `scripts/07_compute_oracle_stop.py:57-62` counts it as a real low-gain step. With `gain_k=3`, two actual small gains can produce `gain_stop=1`. Suggested fix: compute gains only for transitions `level 2..N` and define whether the stop is the current level after `k` low transitions or the first level of that plateau.
- `scripts/15_relative_saturation.py:301`, severity High: the relative gain stop repeats the same artificial-zero off-by-one bug, then returns `i + 1 - (gain_k - 1)` at `scripts/15_relative_saturation.py:314`. This can make `gain_based` saturate at level 1 before enough evidence exists. Suggested fix: align with a transition-based gain definition and add tests for flat curves, one-step gains, and NaN-containing curves.
- `scripts/15_relative_saturation.py:130-151`, severity Med: when level-20 pseudo-GT has no triplets, `RelTripletRecall` becomes `"NaN"` and `RelSGScore` silently switches from `0.5 * node + 0.5 * triplet` to node-only. This contradicts the documented formula at `scripts/15_relative_saturation.py:837` and makes images with no level-20 EGTR relations easier to saturate. Suggested fix: choose and document a fixed convention, e.g. exclude no-triplet images from relation-based RelOracle, or set empty-reference triplet recall to 1.0/0.0 explicitly and report counts.
- `scripts/aggregate_100.py:109`, severity Med: `consecutive_k_stop` returns `max_level` both for "never saturated" and "first saturated at the final level". Comparisons at `scripts/aggregate_100.py:415-417` then treat missing/never/final-hit identically. Suggested fix: carry a separate `hit: bool` or `None` sentinel and only cap to `max_level` for policy-level plots.
- `scripts/run_focused_100.py:66` and `scripts/run_focused_100.py:110-111`, severity Med: LLM `stop_level` also uses `num_levels` as both sentinel and valid final stop. The VLM logic does the same at `scripts/run_focused_100.py:158-196`. Suggested fix: store raw `stop_level_nullable` plus `capped_stop_level`.
- `src/milestone1/llm.py:132`, severity Med: `bool(result["stop_decision"])` turns the string `"false"` into `True`; same pattern in `src/milestone1/llm_engine.py:244` and `src/milestone1/vlm.py:199`. Suggested fix: accept real booleans, parse `"true"/"false"` strings explicitly, otherwise derive from the numeric threshold.

## Metric/logic issues
- `src/milestone1/metrics.py:32-37` and `src/milestone1/metrics.py:61-66`: empty-pred/empty-GT returns perfect F1 only when both sets are empty. That is a defensible convention, but if future subsets include empty GT relationships it will inflate aggregate triplet metrics. Given the current config requires `min_gt_relationships: 3` at `configs/config.yaml:23`, this is not currently a primary bug.
- `src/milestone1/metrics.py:6-18`: label-only set matching intentionally deduplicates instances. Correct for the design, but it means duplicate object categories and multiple identical relation labels are ignored, so it measures category coverage rather than scene-graph instance correctness.
- `src/milestone1/egtr_infer.py:172-184`: relationship output confidence is only gated relation confidence, while ranking at `src/milestone1/egtr_infer.py:156` uses relation confidence times subject/object confidence. That is not necessarily wrong, but downstream `mean_relation_confidence` is not the same score used for ranking.
- `src/milestone1/llm.py:27`, `src/milestone1/llm_engine.py:28`, and `src/milestone1/vlm.py:27`: prompts tell the model `stop_decision` means `saturation_score >= 0.8`, while the configured runtime threshold is `0.7` at `configs/config.yaml:81` and `configs/config.yaml:102`. Since code mostly ignores `stop_decision`, this mainly contaminates LLM calibration and explanations.
- `scripts/15_relative_saturation.py:335-340` and `scripts/aggregate_100.py:360-367`: saturation rates count only stops `< max_level`, so images that first cross threshold at level 20 are reported as unsaturated. This is only valid if "saturated" means "could have stopped before the final observation".

## Robustness
- `src/milestone1/llm_engine.py:224-229` and `src/milestone1/llm_engine.py:247-252`: the claimed hard wall-clock timeout is not actually hard. On `FutureTimeout`, exiting the `with ThreadPoolExecutor(...)` block waits for the worker thread, so a slow Ollama call can still block. Same issue in `src/milestone1/vlm.py:113-118`. Suggested fix: manage executor shutdown with `wait=False, cancel_futures=True`, or use a subprocess/process-based timeout.
- `src/milestone1/llm.py:89`: the older LLM path has only the `requests` socket timeout and no wall-clock guard, unlike `llm_engine.py`.
- `src/milestone1/llm.py:42`: `_parse_llm_json` uses a flat-object regex, so JSON with nested objects or braces inside a string can fail when surrounded by prose. Suggested fix: scan from each `{` and use `json.JSONDecoder().raw_decode`.
- `src/milestone1/llm.py:129-130`, `src/milestone1/llm_engine.py:241-243`, and `src/milestone1/vlm.py:198`: `float("NaN")` can slip through clamping behavior incorrectly. Suggested fix: require `math.isfinite(score)` before clamping; non-finite should become 0.0 and be marked invalid.
- `src/milestone1/vlm.py:97-111`: VLM requests set `think=false` but do not set `num_ctx=12288`, unlike LLM calls at `src/milestone1/llm.py:80-87` and `src/milestone1/llm_engine.py:189-197`.
- `scripts/aggregate_100.py:77-87` and `scripts/aggregate_100.py:415`: missing cached LLM score files silently become score `0.0` and then stop `max_level`, which biases methods late and hides incomplete runs. Suggested fix: validate expected cache cardinality per slug/image/level before aggregation.
- `scripts/aggregate_100.py:211-213`: config slugs are hard-coded rather than generated with `make_config_slug`, so changes to model names/history/config conventions can silently miss cache directories.
- `scripts/02_generate_resolution_levels.py:18-19` and `scripts/02_generate_resolution_levels.py:41`: no validation that `len(scales) == num_levels`; generated files can disagree with printed/evaluated level count.
- `src/milestone1/gt.py:93-97`: relationship indices only check `< len(anns)`, not `>= 0`; malformed negative indices would select from the end of the annotation list.

## Style/convention
- `src/milestone1/egtr_infer.py:3-11`: `sys`, `Path`, `numpy as np`, and `REPO_ROOT` are unused.
- `scripts/aggregate_100.py:139-144`: nested conditional expressions around `math.isnan` are hard to read and unnecessary because recall values are always floats.
- `scripts/07_compute_oracle_stop.py:45-46`, `scripts/07_compute_oracle_stop.py:65-66`, `scripts/08_compute_baseline_stop.py:42-43`, and `src/milestone1/analysis.py:256`: hard-coded `20` should use `num_levels` or a passed final-level parameter.
- `scripts/15_relative_saturation.py:161-174`: `compute_relative_metrics` is annotated as returning `dict[str, float]` but returns `"NaN"` strings.
- `src/milestone1/analysis.py:40-50`: hard-coded `oracle_0.5` and `baseline_0.4` duplicate config assumptions.

## Top 5 fixes
1. Fix EGTR relationship ranking by masking/ranking over selected valid queries before truncation, then add a regression test where valid pairs appear below the first `top_k * 3` global pairs.
2. Redefine gain-based stopping on real inter-level transitions and remove the artificial initial zero in both absolute and relative oracle scripts.
3. Represent "never saturated" separately from valid final-level stop, and update MAE/early/late/saturation-rate reporting accordingly.
4. Harden Ollama parsing/normalization/timeouts: balanced JSON extraction, explicit boolean parsing, finite score checks, and a real wall-clock timeout.
5. Make relative no-triplet pseudo-GT behavior explicit and consistent with the reported `RelSGScore` formula.
