# Review: VISAPP GQA question-decomposition early-stop plan v1.0 (2026-10-08)

Reviewer: Claude (orchestrator). Grounded on this machine's actual data/model state, not only the plan text.

## Verdict

Go, with 4 blocking fixes before Phase A and 5 strong recommendations. The plan is far better pre-registered
than the ROBOVIS plan was; the main scientific risk is not methodology but that the chosen question population
(84 % color) is solvable at very low resolution, leaving no room for any adaptive stop to show a benefit.

## Environment facts verified (preflight items the plan asks for)

| Item | State |
|---|---|
| Repo | `/home/kanaoka/usr/github/milestone_1`, branch `feature/robovis2027`, HEAD 6f7f1de (plan says `/Users/.../ae_milestone1` — wrong path/OS) |
| GQA questions | `/home/kanaoka/usr/dataset/gqa/val_balanced_questions.json` (v1.2, 132,062 q; also testdev_balanced with answers) |
| GQA images | Not downloaded separately. All 10,234 val-balanced image IDs exist in `/home/kanaoka/usr/dataset/vg/images` (VG jpg). Dimension check still required. |
| Ollama | `qwen3.5:9b` digest `6488c96fa5fa…` (9.7B) and `qwen3.5:4b` `2a654d98e6fb…` present; server running; GPU idle (160 MiB / 16 GB) |
| Disk | 412 GB free |
| Running jobs | none (no conflict with old experiments) |

## Blocking fixes (plan text is wrong against real data)

1. **Schema**: the final operation is always `query`; the attribute is in `argument` (`color`/`material`/`shape`).
   `types.detailed` values are `directOf`, `directWhich`, `material`, `how`, `positionQuery`, … — not attribute names.
   Filter must be `structural=="query" and semantic=="attr" and semantic[-1].argument in {color,material,shape}`.
2. **Population size & shape** (val balanced): 9,290 eligible questions over 5,512 images.
   color 7,790 / material 1,387 / shape 113. relate count: 0 → 6,187, 1 → 3,103, 2 → 0. op count 2–4 only.
   So `max_relations=2`, `max_operations=6` are non-binding; shape is too small to be its own stratum.
3. **Self-evident-attribute rule**: only 3/9,290 have the answer word in the question
   ("What is the glass made of? → glass", "…the orange is inside of → orange"). Keep the rule, it is cheap, but do not
   expect it to matter.
4. **Paths/config**: `gqa_questions`, `gqa_images_root`, `repo_root` must be set to the Linux paths above; use a new
   branch (e.g. `feature/visapp-gqa`) off current HEAD, not `feature/robovis2027`.

## Strong recommendations (pre-registered changes, decided before any model call)

A. **Stratify the candidate pool by attribute and relate count** before hashing/splitting.
   Rationale: color is the most low-pass-robust signal; a color question on a large object is answerable at
   Level 1 (25 px side). With 84 % color the Phase-B diagnostic "L1 already ≥ 0.95" is likely to trigger and the
   whole experiment degenerates. Material (texture) and shape are resolution-sensitive. Proposal:
   quota per split ≈ 40 % color / 40 % material / all passing shape (≤ 20 %), and within each 50/50 relate-0 vs relate-1.
   This is metadata-based, not model-performance-based, so it does not violate §3.2 step 7.
B. **Make a cheap reader-only resolution probe mandatory before freezing the filter** (≈ 200 candidates × L1/L5/L10/L20
   = 800 calls, < 1 h). Allow ONE filter revision from it, documented, before dev/test split. The plan currently
   lets you discover "no headroom" only after dev is fully inferred.
C. **Confidence calibration gate in Phase B.** Prior evidence on this machine: qwen3.5:9b is overconfident
   (MIRU: 9b stops earlier than 4b; ROBOVIS judge reported "resolved" 90–98 % regardless of input). Add a pilot
   diagnostic: mean direct/checklist confidence per level and Spearman(level, confidence). If confidence is flat
   across levels the threshold grid collapses to {stop at L1, never stop}. Pre-register 4b as the one allowed
   model fallback *decided at pilot*, not after dev. (Plan currently forbids model change mid-evaluation only.)
D. **Replace the adjacent-level mix reference with the concave hull of the fixed-t curve** (lesson from rec_r0:
   adjacent mix is a weak comparator). Keep Fixed-dev as the frozen comparator; report hull F(b) descriptively.
E. **Dev size**: with dev = 60, the 0.95 accuracy gate is "≤ 3 errors", so threshold choice is very noisy.
   Compute is cheap (3 image calls/question/level; 9b ≈ 3–5 s/call → dev 60 ≈ 3–5 h, test 200 ≈ 10–17 h).
   Raise `target_dev_questions` to 100 if screening yields it; test 200 is acceptable.

## Smaller points

- Exact match after NFKC/lower/strip: GQA answers include "light blue", "dark", "wood", "brick". The reader will
  often say "blue" for "light blue"; screening removes those at L20 but lower levels can flip. Fine as
  pre-registered, but consider a pre-registered *secondary* metric with the closed GQA attribute vocabulary
  (dataset-level, not per-question GT) given to the reader prompt as allowed answers. Decide now, not after test.
- H2 (min vs mean) is nearly vacuous on relate-0 questions: checklist = {find X, attribute of X}, and the min is
  dominated by the identification item. Expect min ≈ direct on 2/3 of unstratified data — another reason for A.
- Direct-confidence with k=1 and an overconfident model will likely get θ = +∞ on dev (i.e. Level 20). Then
  "min beats direct" is trivially true on MeanLevel. State in the report that Direct's frozen θ may be +∞ and
  show its full dev curve.
- `num_ctx 8192`: enough for these prompts (lesson from the earlier 4096 truncation bug still applies — set it explicitly).
- Ollama + `temperature 0` + `seed` is not bit-deterministic on GPU; the 20-question repeat diagnostic is right.
- PIL bilinear downscale is antialiased; this matches `scripts/02_generate_resolution_levels.py` behaviour. Keep PNG.
- `think:false` on qwen3.5 may reduce the quality of the confidence number; pilot will show.
- Bootstrap: with 200 test q the accuracy CI half-width is ≈ ±4 pp, so the −5 pp non-inferiority test will be close
  to inconclusive by construction. Fine, but say so in the plan to avoid "unknown" looking like failure.
- Existing `src/milestone1/vlm.py` indeed re-resizes to 512/JPEG q85/temperature 0.3 — confirmed; do not reuse.
  `src/milestone1/ollama_client.py` is a cleaner base for the HTTP part.

## Cost estimate (9b, single RTX 5080)

| Stage | Calls | Time |
|---|---|---|
| Screening (200 + 800 candidates, L20 reader) | 1,000 | ≈ 1 h |
| Pilot (20 q × 5 levels × 3 roles + planner) | 320 | ≈ 20 min |
| Dev full (60–100 q × 20 × 3) | 3,600–6,000 | 3–7 h |
| Test full (200 q × 20 × 3) | 12,000 | 10–17 h |

Feasible inside the 10/21 working deadline if Phase A implementation takes ≤ 2 days.
