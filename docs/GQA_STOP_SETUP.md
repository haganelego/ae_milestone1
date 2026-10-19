# VISAPP 2027 GQA early-stopping experiment: setup on another machine

This guide is for a collaborator who clones the repository and wants to resume
or replicate the GQA early-stopping experiment (`src/milestone1/gqa_stop/`,
plan §27 "v27"). Large artifacts are not in git, so it explains how to copy or
regenerate them.

Items marked **UNVERIFIED** come from reading the code. They were not tested
on a second machine.

## 1. What is in git and what is not

| Path | In git | How to get it |
|---|---|---|
| `src/milestone1/gqa_stop/`, `scripts/gqa_*.py`, `tests/test_gqa_stop_*.py` | yes | clone |
| `configs/gqa_stop_*.json` | yes | clone |
| `prompts/gqa_stop/*.txt` (prompt hashes are part of every cache key) | yes | clone |
| GQA questions and images | no | download, see §3 |
| `data/processed/gqa_stop/<run>/` (manifests, GT, screening labels, level PNGs) | no | copy (recommended) or regenerate |
| `outputs/gqa_stop/<run>/` (request cache, state, eval, worklog) | no | copy (recommended) or regenerate |

Sizes on the original machine:

| Directory | Size |
|---|---|
| `data/processed/gqa_stop/visapp_gqa_v27/` | 1.7 GB |
| `outputs/gqa_stop/visapp_gqa_v27/` | 3.0 GB |
| all of `data/processed/gqa_stop/` + `outputs/gqa_stop/` | about 13 GB |

These directories are not listed in `.gitignore`. Do not `git add` them.

## 2. Hardware and software

- **GPU.** Verified on an NVIDIA RTX 5080 with 16 GB VRAM. Driver 580.178.04 (open kernel module) was used. `qwen3.5:9b` at `num_ctx 8192` is 5.9 GB resident and runs 100% on GPU. Smaller GPUs are UNVERIFIED.
- **ollama.** Version 0.30.4, installed as the systemd service `ollama` (`/usr/local/bin/ollama serve`). The code calls `http://localhost:11434/api/chat`. Other ollama versions are UNVERIFIED; the determinism gate in §5 detects decoding drift.
- **Models.**

```bash
ollama pull qwen3.5:9b   # main model (configs/gqa_stop_v27.json)
ollama pull qwen3.5:4b   # model-swap replication (configs/gqa_stop_v27_4b.json)
```

- **Model digests.** The run records the digest at `preflight`, and every later stage refuses to continue if the live digest differs. Recorded digests:

| Model | Digest (ollama ID prefix) |
|---|---|
| `qwen3.5:9b` | `6488c96fa5fa…` |
| `qwen3.5:4b` | `2a654d98e6fb…` |

  Check with `ollama list`. Whether a fresh `ollama pull` today returns the same digest is UNVERIFIED. If it differs, either copy the model store from the original machine (`/usr/share/ollama/.ollama/models/` for the systemd service, stop the service first) or treat the run as a new replication (§4.3).
- **Journal access.** `v27-envcheck` reads `journalctl -u ollama`. The user must be able to read the system journal (member of `adm` or `systemd-journal`).
- **Python.** Use the root `uv` environment (Python 3.12, pandas, numpy, pillow, matplotlib, requests). Do not use `--project egtr` for gqa_stop; that env is for the older EGTR/Milestone-1 scripts.

```bash
uv sync --all-extras
uv run python -m milestone1.gqa_stop.cli --help
```

## 3. ollama gotchas the code guards against

- **Sampler penalties are active at temperature 0.** The `qwen3.5` Modelfile sets `presence_penalty 1.5`, and ollama's default `repeat_penalty` is also on. At temperature 0 they still suppress tokens near the end of the prompt, which flips `yes` to `no` and `true` to `false` in observer JSON. The returned logprobs are pre-penalty, so the emitted token stops matching the argmax. Every request therefore sends neutral `decoding_options` from the config: `temperature 0, top_k 1, top_p 1.0, presence_penalty 0, frequency_penalty 0, repeat_penalty 1.0, mirostat 0`, plus `think: false` and `seed 20261008`. Never remove these. Check the Modelfile defaults with `ollama show qwen3.5:9b --parameters`.
- **The vision encoder can fall back to CPU.** If another model is resident when the target loads, ollama may place the CLIP vision encoder on CPU. The journal then shows `clip_ctx: CLIP using CPU backend`, images take about 30x longer, and numerics may differ. `ollama ps` still says "100% GPU" because it only counts LLM weights. `v27-envcheck` unloads every resident model, warms the target alone, sends one image request, and then requires all three of these:
  - `CLIP using CUDA0` in `journalctl -u ollama`
  - only the target model resident
  - `100% GPU` in `ollama ps`
- **Keep the machine to one model.** Do not run other ollama clients during a run. Stop processes by PID, not with `pkill -f`.
- **Line endings.** Prompt files are hashed byte-for-byte. A checkout that rewrites line endings (for example `core.autocrlf=true`) changes the hashes, and `max_dev_prompt_revisions: 0` then stops the run with `prompt revision limit (0) reached`.

## 4. Data

### 4.1 GQA download (needed for a fresh run; see §4.2 for continuing)

Official page: https://cs.stanford.edu/people/dorarad/gqa/download.html

| Archive | Needed by v27 | Files used |
|---|---|---|
| `questions1.2.zip` (1.4 GB) | yes | `val_balanced_questions.json` (dev), `testdev_balanced_questions.json` (test) |
| `images.zip` (image files, 20.3 GB) | yes | `images/<imageId>.jpg` (flat directory) |
| `sceneGraphs.zip` | no | only the older v5 box pre-check (`boxes_v5.py`) |

```bash
mkdir -p ~/dataset/gqa && cd ~/dataset/gqa
wget https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip questions1.2.zip && unzip images.zip   # -> *.json and images/
```

The configs hard-code absolute paths. Point them at your copy in one of two ways:

- **Same path.** Make `/home/kanaoka/usr/dataset/gqa` exist (for example a symlink to your download). The configs then work unchanged.
- **Local config copy.** Copy the config to an untracked file and edit `paths.gqa_questions`, `paths.gqa_questions_test`, `paths.gqa_images_root` and `paths.repo_root`. There are no environment variables for these paths.

`paths.repo_root` can also be overridden on any command with `--repo-root "$PWD"`. Relative paths (`data_root`, `output_root`, `prompts/gqa_stop`) then resolve against the clone.

### 4.2 Continuing the existing experiment: copy, do not regenerate (recommended)

The screening labels (answerable or unanswerable) come from the model's L20 answer. A different GPU, driver or ollama version may not reproduce them byte-for-byte. All results so far depend on those labels and on the request cache. To continue the same experiment, copy both run directories from the original machine.

```bash
SRC=kanaoka@<original-host>:/home/kanaoka/usr/github/milestone_1
rsync -avP "$SRC/data/processed/gqa_stop/visapp_gqa_v27/" data/processed/gqa_stop/visapp_gqa_v27/
rsync -avP "$SRC/outputs/gqa_stop/visapp_gqa_v27/"        outputs/gqa_stop/visapp_gqa_v27/
# Optional: 4b replication and probes (they read records from other runs)
rsync -avP "$SRC/data/processed/gqa_stop/" data/processed/gqa_stop/
rsync -avP "$SRC/outputs/gqa_stop/"        outputs/gqa_stop/
```

**Absolute paths inside the copy.** `image_manifest.jsonl` stores absolute level-PNG paths (`/home/kanaoka/usr/github/milestone_1/data/processed/...`). If the clone lives elsewhere, the image series thinks the levels are missing and regenerates them from the GQA JPEGs. Regenerated PNG bytes from another Pillow version may differ. Then every image request misses the cache and runs new model calls. Avoid this in one of two ways:

- **Same path.** Clone to `/home/kanaoka/usr/github/milestone_1`, or symlink that path to your clone.
- **Rewrite the manifest copy.** Run this on the copy, not on the original:

```bash
sed -i "s#/home/kanaoka/usr/github/milestone_1#$PWD#g" \
  data/processed/gqa_stop/visapp_gqa_v27/image_manifest.jsonl
```

Rewriting only changes the `path` fields. The SHA-256 of every PNG stays in the manifest and is re-checked on every read. This relocation is UNVERIFIED end-to-end.

With the copied data, the cached stages (`v27-table`, `v27-evaluate`) should not read the raw GQA files. Only `preflight` (it counts questions), `v27-extract` and missing-level regeneration read them. This is UNVERIFIED on a fresh machine.

### 4.3 Fresh replication from scratch

Use a new run id so that copied results are never overwritten. Copy `configs/gqa_stop_v27.json` to a new file (for example `configs/gqa_stop_v27_replica.json`). In the copy, change `run_id` and the three roots `data_root`, `selection_root` and `output_root` to `..._replica`, and set your local `paths`. `configs/gqa_stop_v27_4b.json` is the existing example of this pattern.

These parts are deterministic and model-independent:

- **Question selection.** `v27-extract` with seed `20261008` uses a fixed SHA-256 hash order. It gives dev 300 questions (at most 2 per image, 222 images) and test 886 questions (327 images). The 4b run re-extracted with byte-identical manifests and GT.
- **Level images.** There are 20 levels with a geometric scale from 0.05 to 1.0. The base image has a long side of 512 (Lanczos), degradation uses bilinear resizing, and output is PNG. Images are generated lazily on first use. Byte identity across Pillow versions is UNVERIFIED.

These parts depend on the model and the environment:

- **Labels.** Screening labels from `reader_lp` at L20. On the original machine the dev split had 175 answerable and 125 unanswerable questions; 4b agreed with 9b on 89% of them.
- **Records.** All inference records and logprobs.

## 5. Run sequence (v27, `qwen3.5:9b`)

All commands run from the repo root. Add `--repo-root "$PWD"` if the clone is not at the original path. Every command accepts `--dry-run` (no model calls, no writes).

```bash
C=configs/gqa_stop_v27.json
CLI="uv run python -m milestone1.gqa_stop.cli"

# 0. Fresh run only
$CLI preflight   --config $C   # records model digests into effective_config.json
$CLI v27-extract --config $C   # questions_{dev,test}.jsonl, gt_*.json; refuses if they exist

# 1. Before every session (also when resuming)
$CLI v27-envcheck --config $C  # isolate + warm model, CLIP on CUDA0, 100% GPU
$CLI v27-detgate  --config $C  # repeat pairs byte-identical, argmax violations 0 (~120 calls)

# 2. Dev
$CLI v27-identity --config $C              # reader_lp vs plain reader, 120 pairs
$CLI v27-screen   --split dev --config $C  # L20 labels
timeout 585 $CLI v27-infer --split dev --max-seconds 560 --config $C   # repeat until complete
$CLI v27-audit    --split dev --config $C  # timing > 5 s, argmax violations
$CLI v27-quarantine --split dev --config $C  # only if the audit flags records
$CLI v27-table    --split dev --config $C
$CLI v27-evaluate --split dev --config $C  # θ sweep 0.00..1.00 + eval/dev/theta_star.json
$CLI v27-evaluate --split dev --posthoc --config $C   # eval_posthoc/dev/ (exploratory)

# 3. Test inference (same envcheck + detgate first)
$CLI v27-screen --split test --config $C
timeout 585 $CLI v27-infer --split test --max-seconds 560 --config $C  # repeat
$CLI v27-audit  --split test --config $C
$CLI v27-table  --split test --config $C
```

Notes on the sequence:

- **Chunked inference.** `v27-infer` stops cleanly after `--max-seconds` (checked between levels) and resumes from the on-disk cache. Progress goes to `logs/infer_progress_<split>.json`. The original run used foreground chunks of 560 s and no background chaining. At about 0.6 s per call, dev was about 24,700 records (27 chunks) and test about 72,600 records (79 chunks).
- **Gates.** `detgate` needs a passing `envcheck`, and `identity`, `screen` and `infer` need a passing `detgate`. Both are stored in `outputs/.../state.json`, so re-run them at the start of each session.
- **Exit codes.** Exit code 2 means the run paused after repeated technical errors (see `execution.pause_*` in the config). Exit code 1 is a stage error. See `worklog.md` in the output root.
- **The θ sweep.** It is part of `v27-evaluate`, which writes `sweep.csv`, `sweep_by_attribute.csv`, the paired bootstrap and `theta_star.json`. θ* is the argmax of dev strict balanced accuracy, with ties going to the smaller θ.

## 6. Rules that must not be violated

- **Never re-run the confirmatory test evaluation.** It was run once, on 2026-10-14T15:16 UTC, by the team lead with `v27-evaluate --split test --confirm-test-eval`. The command still works if called again. Only this written rule prevents a re-run.
- **Frozen confirmatory files.** `eval/dev/theta_star.json`, `eval/dev/` and `eval/test/` under `outputs/gqa_stop/visapp_gqa_v27/` are frozen. Post-hoc work goes to `eval_posthoc/` or to a new run directory.
- **No test tuning.** Never choose θ, prompts or methods from test numbers (`execution.no_test_tuning`).
- **Do not edit the v27 prompts.** These are `prompts/gqa_stop/answer_only.txt`, `planner_v3.txt` and `observer_v3.txt`. With `max_dev_prompt_revisions: 0`, the engine refuses any change, and a changed prompt also invalidates every cache key. New prompt variants need a new file, a new config and a new run directory, as in `planner_v4.txt` with `configs/gqa_stop_probe_oracle_v4.json`.
- **Do not change the model in place.** A digest change makes the run refuse to continue. Use a new run id instead.
- **Do not edit tracked configs of finished runs.** Make a copy with a new `run_id`.
- **Never delete records.** Flagged records go to `quarantine/` via `v27-quarantine`.

## 7. Probes and later runs

Exploratory probes reuse v27-style records from other run directories. Copy those directories too (§4.2).

```bash
C=configs/gqa_stop_probe_oracle.json      # reads outputs/gqa_stop/probe_multi_rel
uv run python scripts/gqa_probe_oracle_run.py infer --config $C --dry-run
C4=configs/gqa_stop_probe_oracle_v4.json
uv run python scripts/gqa_probe_oracle_run.py v4-infer --config $C4 --dry-run
```

For the full command lists, see the "Commands" sections of the research notes listed in §9.

## 8. Tests and lint

```bash
# gqa_stop suite (468 tests, about 2.5 min, root env, no ollama needed)
uv run pytest -q -p no:cacheprovider tests/test_gqa_stop_*.py
# quick subset
uv run pytest -q tests/test_gqa_stop_v27.py tests/test_gqa_stop_oracle_plan.py \
  tests/test_gqa_stop_planner_v4.py tests/test_gqa_stop_client.py tests/test_gqa_stop_cli.py
uv run ruff check src/milestone1/gqa_stop tests/test_gqa_stop_*.py
```

Do not use `uv run pytest tests -k gqa_stop` or a bare `uv run pytest`. Collection then imports older Milestone-1 and ROBOVIS test files, which need `scipy` and `pycocotools`. The root environment does not have them.

## 9. Pointers

- **Pre-registration and plan.** `.claude/docs/VISAPP2027_GQA_EXPERIMENT_PLAN.md`. See §27 for v27, §27.8 for gates, §27.11 for the test result, and §27.12–§27.14 for post-hoc work, the 4b swap and probes.
- **Dev report and commands.** `.claude/docs/research/gqa-stop-v27-dev.md`
- **Confirmatory test report and commands.** `.claude/docs/research/gqa-stop-v27-test.md`
- **Post-hoc results.** `.claude/docs/research/gqa-stop-v27-posthoc.md`
- **4b dev replication.** `.claude/docs/research/gqa-stop-v27-dev-4b.md`
- **Probes.** `.claude/docs/research/gqa-stop-probe-multi-rel.md` and `gqa-stop-probe-oracle.md`
- **Sampler-penalty analysis.** `.claude/docs/research/gqa-stop-cause2-analysis.md`
- **Per-run log.** `outputs/gqa_stop/<run>/worklog.md` holds the timestamped log of every stage.
