# ROBOVIS2027 line: consolidated experiment report (2026-10-07 → 2026-10-08)

Branch `feature/robovis2027`. All numbers come from the files named next to them. Status is as of the 2026-10-08 early-morning snapshot (files read-only, no model runs). The plan text ("ROBOVIS2027 実験計画書・自律実行仕様 v1.0") is not in the repo, so references to plan sections follow the worklog and report notes.

## 要約（日本語）

- 目的: PGSG（Pix2Grp）の出力シーングラフを「観測器」として使う。ローカル VLM/LLM（qwen3.5 4b/9b）が「未確認だがありそうな物体・関係」（仮説 H）と観測上の問題を挙げ、それを使って解像度回復（20 段階）をどこで止めるか決める。H1 は「仮説が同一観測量での GT 再現率を上げる」。
- 結果: H1 はどの変種でも支持されなかった。proposed と observed_only の差は ≈0（最大 +0.008、CI 下限 0）。H2 は検証できなかった（前提となる改善がないため）。
- 4b でのプラン通りの §6 規則は退化した。basis=2 かつ benefit=2 の項目が ≈0 % で、20 段階リプレイでは平均 t≈2.2–2.4 で停止する。
- 9b と画像付き judge（F1v_v2_G__9b）で初めて段階に追従する信号が出た。それでも停止は t≈4–5 で、fixed-stop 曲線上にとどまる。等観測 fixed mix との差の CI は 0 を含む。
- judge v3（ユーザー承認済みの逸脱）で resolved が情報を持つようになった（AUC 0.57–0.64）。ただしこれはクラスによらない「鮮明になれば resolved」という一様な傾向である。j9b の relsg_gt +0.056 [+0.006, +0.101] は 60 比較中 2 件だけで、多重比較の注意が必要。
- 失敗の根本原因: 仮説の大半は「グラフの帳簿付け」か「既に観測済みの物体」。本当に欠けている GT 物体を挙げた割合は raw で 0 %、G で 4–9 %（t1 では GT ラベルの 56 % が未描画なので、余地はある）。単一スナップショットでの絶対的な十分性判断は較正に縛られる。
- 画像のみプローブ: 9b は解像度を序数的には読める（detail_fraction と t の ρ = 0.82、AUC 0.71）。一方 "sufficient" は 0 % で、768 px にしても改善しない。
- PGSG の観測特性: 解像度によらず約 115 entity / 337 relation を出す。画像ごとの R@50 は単一サンプルのノイズが大きい。relative-stable oracle が定義できるのは test 37/100（R50）だけ。
- 縮小版 MIRU 実験（EGTR, 旧 100 枚）: text 4b は MIRU を再現した（MAE 3.00）。現在画像を SGG プロンプトに入れる early fusion は悪化した（MAE 4.78–5.04、停止 ≈9.4）。原因は「グラフ安定」の作話と、ぼけ手がかりの喪失。
- MIRU の画像 VLM 比較は閾値が不整合だった（θ 0.8）。θ 0.70 に揃えると image-only 9b（MAE 3.04）は text 4b と同等。
- GT 基準の等観測比較では、どの手法（MIRU 手法・late fusion を含む）も fixed-stop 曲線を超えない。
- late fusion（事後選択）は MAE 2.59 で、事前登録した held-out（96 枚）での検証は **pending**（スコアリング未開始）。v4（bbox 接地）全 20 段階と h3 の要約・リプレイも **pending**。
- 論文として主張できるのは否定的・分析的な結果: 等観測評価枠組み、適応停止が fixed 曲線を超えないこと、early fusion の作話、MIRU の閾値不整合の訂正。

---

## 1. Purpose and plan of record

**Plan.** The plan of record is the user-supplied "ROBOVIS2027 実験計画書・自律実行仕様 v1.0" (2026-10-07). Sources: `.claude/docs/research/robovis2027-worklog.md` and `robovis2027-feasibility.md`.
- PGSG stays fixed as the scene-graph observer over 20 resolution levels.
- A check generator emits up to 3 observed issues and 3 hypotheses per state. Hypotheses are objects or relations likely present but not yet confirmed.
- A judge labels each item: status, limitation, basis 0–2, resolution_benefit 0–2.
- The stop policy (plan §6, `src/milestone1/stop_policy.py`) continues only while some item passes the gate (unresolved ∧ limitation = low_resolution ∧ valid anchors/region) and has min(basis, benefit) ≥ θ = 2. It stops after k = 2 consecutive non-continue states. Default is stop.

**Hypotheses** (wording from `robovis2027-summary.md` §3, not the plan text):
- **H1.** Unconfirmed co-occurrence candidates (hypotheses H) improve GT recall at equal observation. Operationally: `proposed` vs `observed_only`, and `proposed` vs the equal-observation fixed-level mix.
- **H2.** There is a trade-off between H-driven gain and over-observation. This is only testable if H1 shows a gain.

**Conditions** (all replays): `proposed` (A_t + H), `observed_only` (A_t only), `unknown_conservative` (θ = 1, plan-named variant), `fixed_20`. The exploratory rules `benefit_gated` / `benefit_gated_obs` are labelled "plan deviation, needs sign-off" and are not main conditions.

**Evaluation (plan §9, kept unchanged by user decision).**
- Primary q_t = SGDet R@50 against VG GT.
- Single agreed secondary = `relsg_gt` (GT-referenced RelSGScore).
- Main evidence: q(stop) vs the fixed-stop curve, using the equal-observation fixed mix and a paired bootstrap (2000 repeats, seed 20271007).
- Relative-stable oracle (ratio 0.9) MAE is auxiliary.

**What "complete" means (plan §13).** The §13 text is not in the repo; this is reconstructed from the worklog and feasibility §sequencing and **needs confirmation**: dev freeze → test100 run (PGSG + checks + judge) → replay of all conditions → bootstrap, figures, report.

Status against that:
- PGSG test100 inference: done.
- dev20 variants: done.
- Checks/judge on test100: **not run**. The test run was deferred to a user decision (≈5.5 h GPU, expected negative; worklog 14:40). The line was then scaled down by the user (≈19:50).
- The ROBOVIS plan is therefore **not complete** in the plan's sense. All ROBOVIS stop results below are dev20 only.

## 2. Environment, data, models

Source: `outputs/robovis2027/preflight.json` (generated 2026-10-07 08:11, commit 1dd1ba2, 24 dirty paths at that time).

| Item | Value |
|---|---|
| GPU / host | NVIDIA RTX 5080 16 GB (driver 580.173.02), 24 CPU, 125 GiB RAM |
| envs | `egtr` and `envs/pgsg`: both Python 3.11.15, torch 2.7.1+cu128, CUDA 12.8 |
| VG | `/home/kanaoka/usr/dataset/vg`, 108,249 images; test.json sha256 `e786550a…`, rel.json `ab5d4357…` |
| PGSG | submodule `cff7420`; `vg_sgg.pth` sha256 `d90c63ca6e0f7ea516c5f0ae23e2520085011fbc2c04ee1c56f23a2e42e299eb`; BSD-3 |
| PGSG generation (effective) | nucleus sampling top_p 0.9 + HF top_k 50; repetition_penalty **1.1 effective** (1.5 passed, hard-coded in `med.py`); min_length unused; 36 segments (seg_len 32); top-1 entity / top-3 predicate labels; amplification 5.0/2.0; entity NMS 0.92; ≤512 relations |
| PGSG seeds / speed | seed per (batch of 4 consecutive images, level); 3.07 s/state, ≈7.2 GB (`dev_r0/pilot_report.md`) |
| ollama | 0.30.4, `/api/chat`, `think: false`, JSON mode; temp 0, seed 20271007, num_ctx 8192 (16384 for 9b/h3/v4 runs), num_predict 1536; schema retry at temp 0.3 + seed+1 |
| models | `qwen3.5:4b` (4.7B, Q4_K_M, digest `2a654d98…`); `qwen3.5:9b` (9.7B, Q4_K_M, digest `6488c96f…`) |
| budget | max_model_calls 9000, concurrency 1 |

**PGSG validity checks.**
- 1000-image official run: R@50/100 = 16.5/18.4, against README 17.7/20.7 (`pgsg-run-log.md`). Greedy decoding collapses (R@50 5.9), so sampling is mandatory.
- Our per-image SGDet equals Pix2Grp `evaluate_relation_of_one_image` with max |diff| = 0 on 400 states.
- On dev20 level 20, the stock runner with seeds 42/1/2/3 gives R@50 17.0–21.8. Ours is 19.6 (`pilot_report.md`, `official_eval/summary.json`).

**Dataset re-selection** (`data/processed/robovis2027/selection_manifest.json`, script 26, commit e820065):
- Source: VG-150 test split (26,446 images; same split PGSG reports on).
- GT-only eligibility: ≥5 objects and ≥3 relations. 18,887 eligible; excluded 1,900 (objects) and 5,659 (relations).
- Ordering: sorted ids, then `random.Random(42).shuffle`. test = shuffled[0:100], dev = shuffled[100:120]. No observer-dependent filter.
- Overlap with the MIRU 100 images: 4 (all in test).
- Why the MIRU 100 were not reused: they had been re-filtered with EGTR-based stop times (commit 2b038b2). `report_notes.md` says 29 right-censored images were removed; `robovis2027-feasibility.md` says 35. **The two sources disagree; needs confirmation.**

**384-base degradation** (`configs/robovis2027.yaml`, script 27):
- All 120 images have max side 500. They are pre-shrunk to max side 384, PGSG's input size. Without this, levels 19–20 would be near-identical to PGSG.
- Schedule: scale_t = 0.05·20^((t−1)/19), bilinear, JPEG q95.
- Boxes are mapped back to GT coordinates with `coord_scale` = 1.3021 for all 120 images (`levels_manifest.json`).
- Level 20 is a scale-1.0 copy + JPEG with no resample round trip. The t19→t20 jump in relsg_gt is therefore partly schedule-inherent (worklog).

## 3. Decisions and deviations from the plan

Sources: `outputs/robovis2027/report_notes.md` and the worklog.

| # | Decision / deviation | Reason | Who |
|---|---|---|---|
| 1 | **Path P dropped** (PGSG co-generating OBSERVED + CHECKS); the 25-state pilot was replaced by a written rationale | PGSG = BLIP ViT-B/16 + BERT-base decoder trained only on `sub [OBJ] pred [REL] obj [OBJ]`; fixed 6-token prompt, `max_txt_len` 40; no instruction following and no CHECKS training signal (`dev_r0/pilot_report.md`) | user, after architecture review |
| 2 | Main paths: F1 (VLM sees image + graph → checks) and F2 (text LLM, graph only); text judge per plan §5 | replacement for P | user |
| 3 | 384-px pre-shrink + `coord_scale` | see §2 | orchestrator, disclosed |
| 4 | GT-only re-selection (test100/dev20) | avoid the EGTR-dependent filter | user |
| 5 | PGSG seed per batch of 4 × level | batch 4 is 2.4× faster; fully reproducible | disclosed |
| 6 | Schema retry = temp 0.3 + seed+1 | a temp-0 same-seed retry is a byte-identical no-op | disclosed |
| 7 | Rendering caps: v1 60 entities / 50 relations; v2 same-label IoU>0.5 merge, 30/20 | PGSG emits ~120 entities / ~340 relations per state | disclosed |
| 8 | `relsg_gt` (0.5·node recall + 0.5·triplet recall vs VG GT, label sets, top-20 entities / top-50 relations) as the single **secondary** q | SGDet R@50 is too sparse at low resolution (28/100 test images all-zero) | user request |
| 9 | **Cleaned observed graph G** (same-label IoU>0.5 merge, top-20 relations) used as the pipeline input; q_t still scored on raw PGSG output | dense, noisy PGSG output corrupts LLM inputs | user decision ≈08:50 |
| 10 | **Evaluation not expanded**: R50 primary, relsg_gt secondary; `monotone_stable` / `gain_based` oracles and precision metrics are auxiliary only; 3-seed smoothing prepared (`scripts/37_robovis_multiseed.sh`), not launched | the root fault is the method's reliance on dense output, not the metric | user |
| 11 | Exploratory rules `benefit_gated(_obs)` | diagnosis only; flagged "needs sign-off" | orchestrator |
| 12 | F1v (judge sees image), prompt v2: separately named dev-only variants, prompts hashed | plan allows named variants | orchestrator |
| 13 | **Judge v3** (user-approved): `resolved` ≈ "about 90 % of what should be visible is visible"; the judge is told the next view is ≈17 % sharper per side; never the level index, scale or remaining steps | the only departure from §5's input restrictions; the stop rule targets approach-driven gain | user |
| 14 | Judge reading resolution directly was **rejected** (task-specific). Instead **judge v4**: bbox-grounded per-object visibility | user decision ≈19:20 | user |
| 15 | **Scale-down** (≈19:50): priority on the MIRU-based experiment; ROBOVIS graph-history (h3) and v4 deferred | absolute single-snapshot judgment is calibration-bound; H had no effect in any variant | user |
| 16 | Held-out late-fusion test **pre-registered** before any held-out score existed (`configs/config_heldout.yaml`) | the fusion rules were chosen post hoc on the MIRU 100 | team-lead approved |

Findings that constrain claims (`report_notes.md`):
- PGSG never returns an empty graph (0/2400 states), so the plan's "empty graph → A_t = 2" clause never fires.
- Per-image q_t is single-sample noisy.
- Oracle MAE is auxiliary.

## 4. PGSG as observer

### 4.1 Recall profiles

dev20 is from `dev_r0/report.md`; test100 is from `test_r0/recall_profile.md`. Means are over images.

| t (scale) | dev R50 | dev R100 | dev relsg_gt | test R@50 % | test R@100 % | test mR@50 % | test relsg_gt | test node_recall_gt | test trip_prec_gt |
|---|---|---|---|---|---|---|---|---|---|
| 1 (0.050) | 0.000 | 0.000 | 0.132 | 0.42 | 0.84 | 0.03 | 0.139 | 0.235 | 0.006 |
| 5 (0.094) | 0.007 | 0.007 | 0.274 | 3.76 | 4.07 | 0.68 | 0.276 | 0.412 | 0.018 |
| 10 (0.207) | 0.134 | 0.134 | 0.457 | 9.10 | 9.81 | 2.82 | 0.405 | 0.548 | 0.034 |
| 15 (0.455) | 0.201 | 0.201 | 0.538 | 12.29 | 13.40 | 4.87 | 0.473 | 0.605 | 0.043 |
| 20 (1.000) | 0.196 | 0.249 | 0.596 | 14.04 | 16.22 | 4.72 | 0.492 | 0.639 | 0.044 |

- test R@50 rises almost monotonically to t ≈ 11 (10.8 %), then stays roughly flat (local peak 14.5 % at t17).
- relsg_gt is still rising at t20: with the 384-px base, PGSG label recall has not saturated by full resolution.
- Per-image qmax (test, R@50): 28 images have qmax = 0. Mean qmax 0.205 vs mean q at t20 0.140, which shows qmax is upward-biased (max of 20 noisy draws). One recovered triplet is worth ≈0.1–0.3 recall on small-GT images.

### 4.2 Density and the cleaned graph G

- **Raw output.** A raw test state averages 115 entities / 337 relations. After merging: 74 / 326. G keeps ≈10 / 20 (`recall_profile.md` §6).
- **G vs raw at test t20.**
  - R50: 0.140 → 0.115 (−2.5 pt).
  - det_prec20: 0.198 → 0.360 (hallucination roughly halved).
  - det_recall20: 0.291 → 0.253.
  - trip_prec20: 0.037 → 0.039 (VG GT sparsity).
- **The cap matters.** relsg_gt_uncapped is 0.358 at t1 while R@50 is 0.004, so the uncapped score is inflated by density.

dev20 G statistics (`dev_r0/report.md`, "Observed graph G statistics"; descriptive only):

| t | 1 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| R50 raw / G | 0.000 / 0.000 | 0.007 / 0.007 | 0.134 / 0.134 | 0.201 / 0.169 | 0.196 / 0.172 |
| det_prec20 raw / G | 0.015 / 0.026 | 0.055 / 0.120 | 0.120 / 0.235 | 0.170 / 0.341 | 0.180 / 0.370 |

### 4.3 PGSG-internal signals (negative control)

Source: `test_r0/pgsg_signals_report.md`, script 34; zero model calls; n = 2000 states.
- The best |ρ| with gain_left is ≈0.3–0.36 (n_entities +, agreement −, mean_mentions −). All are weaker than the level index itself (ρ(t, gain_left) = −0.59 R50 / −0.68 relsg_gt, within-image).
- At a fixed t, the cross-image |ρ| is ≤ 0.16 with sign flips.
- Verdict: these signals are proxies for t only and have no stop value. The interpretable trend: with more resolution, PGSG emits fewer entities and repeats its top entities more.

### 4.4 Oracle definability

Sources: `test_r0/eval/oracle_profile.csv`, `dev_r0/eval/oracle_profile.csv`.

| set | q | definition (role) | defined | unstable_tail | zero_recovery | defined only at t20 | median t |
|---|---|---|---|---|---|---|---|
| test100 | R50 | relative_stable (main) | 37 | 35 | 28 | 15 | 18.0 |
| test100 | R50 | monotone_stable (aux) | 72 | 0 | 28 | 15 | 16.0 |
| test100 | R50 | gain_based δ 0.05 (aux) | 72 | 0 | 28 | 8 | 12.0 |
| test100 | relsg_gt | relative_stable (main) | 36 | 64 | 0 | 19 | 20.0 |
| test100 | relsg_gt | gain_based (aux) | 100 | 0 | 0 | 7 | 13.0 |
| test100 | det_recall20_clean | relative_stable | 40 | 60 | 0 | 18 | 19.0 |
| dev20 | R50 | relative_stable (main) | 6 | 7 | 7 | 2 | 18.5 |
| dev20 | relsg_gt | relative_stable (main) | 14 | 6 | 0 | 9 | 20.0 |
| dev20 | relsg_gt | gain_based (aux) | 20 | 0 | 0 | 5 | 14.0 |

**What this implies.**
1. With PGSG at 384 px, the main oracle is defined for about a third of images. Under relsg_gt it degenerates toward "go to the end" (median 20). Oracle MAE on dev rests on n = 6 (R50) or n = 14 (relsg_gt) and is auxiliary.
2. The defensible main comparison is population-level q(stop) vs the fixed-stop curve with the equal-observation mix.
3. Dense, hallucinated output (trip_prec ≈ 0.04) is invisible to recall-only metrics and corrupts LLM inputs. This motivated G.

## 5. ROBOVIS pipeline experiments on dev20

### 5.1 Variant matrix

"checks" = hypothesis/issue generator; "judge" = item labeller. Levels: "5" = {1, 5, 10, 15, 20}; "20" = all levels.

| Variant | checks model / prompt | judge model / prompt / sees image | graph | levels | notes |
|---|---|---|---|---|---|
| F1 | 4b / v1 (image + graph) | 4b / v1 / no | raw | 5 | plan §4 F1 |
| F2 | 4b / v1 (graph only) | 4b / v1 / no | raw | 5 | plan §4 F2 |
| F2__9b | 9b / v1 | 9b / v1 / no | raw | 5 | |
| F2__j9b | F2 4b checks | 9b / v1 / no | raw | 5 | calls in `usage.jsonl` only; not in the report path list |
| F1v | F1 checks | 4b / v1 / yes | raw | 5 | |
| F1_v2 / F1v_v2 | 4b / v2 | 4b / v2 / no / yes | raw | 5 | |
| F1_v2_G / F1v_v2_G | 4b / v2 | 4b / v2 / no / yes | G | 20 | F1v_v2_G = pre-announced lead |
| F1_v2_G__9b / F1v_v2_G__9b | 9b / v2 | 9b / v2 / no / yes | G | 20 | |
| F1v_v2_G__j9b | 4b v2 checks | 9b / v2 / yes | G | 5 | generator vs judge split |
| F1v_v3_G | 4b v2 checks | 4b / v3 / yes | G | 5 | |
| F1_v3_G__9b | 9b v2 checks | 9b / v3 / no | G | 5 | text-judge ablation |
| F1v_v3_G__9b | 9b v2 checks | 9b / v3 / yes | G | 20 | |
| F1v_v3_G__j9b | 4b v2 checks | 9b / v3 / yes | G | 20 | |
| F1v_v3c_G__9b | 9b / v3c (region-anchor prompt) | 9b / v3 / yes | G | 5 | |
| F1v_v3_G__9b_h3 | 9b, + previous 3 G graphs (labels + triples, no ids/scores/levels) | 9b / v3 / yes (current image) | G | 5 | run finished (203 calls; 2 checks + 2 judge failures, scratch log); **summary/replay pending** |
| F1g_v4_G__9b | H reused from v2 9b | 9b / v4 grounded: top-8 G entities drawn as boxes; clear ∈ {0, 1, 2, absent} | G | smoke 25 states; 20-level run **in progress** (50/400 decision files at snapshot) | **pending** |

5-level replays force the stop to t20 (replay coverage ≈22–25 %), so they carry no stop information. Only the 20-level variants have meaningful replays (coverage 89.8–96.0 %).

### 5.2 Per-level gate fractions (5 sampled levels)

Share of states with ≥1 item passing gate ∧ min(basis, benefit) ≥ 2, from `summaries/checks_summary_*.json` (`frac_states_any_gate_and_min_ge_theta`). The last columns are the item shares of `resolved` and `low_resolution` at t1 → t20.

| Variant | t1 | t5 | t10 | t15 | t20 | all | resolved t1→t20 | low_res t1→t20 |
|---|---|---|---|---|---|---|---|---|
| F1 | 0.00 | 0.00 | 0.00 | 0.06 | 0.00 | 0.01 | 0.05→0.04 | 0.50→0.44 |
| F2 | 0.07 | 0.00 | 0.06 | 0.00 | 0.00 | 0.02 | 0.04→0.07 | 0.53→0.42 |
| F2__9b | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00→0.00 | 0.49→0.62 |
| F1v | 0.00 | 0.07 | 0.05 | 0.00 | 0.00 | 0.02 | 0.09→0.09 | 0.65→0.67 |
| F1_v2 | 0.00 | 0.00 | 0.00 | 0.06 | 0.00 | 0.01 | 0.01→0.01 | 0.15→0.17 |
| F1v_v2 | 0.00 | 0.00 | 0.07 | 0.00 | 0.00 | 0.01 | 0.00→0.07 | 0.49→0.17 |
| F1_v2_G | 0.06 | 0.00 | 0.05 | 0.06 | 0.06 | 0.02 | 0.01→0.01 | 0.29→0.08 |
| F1v_v2_G | 0.06 | 0.00 | 0.05 | 0.00 | 0.06 | 0.01 | 0.00→0.01 | 0.33→0.12 |
| F1_v2_G__9b | 0.30 | 0.25 | 0.32 | 0.25 | 0.20 | 0.26 | 0.00→0.03 | 0.40→0.11 |
| F1v_v2_G__9b | 0.50 | 0.35 | 0.26 | 0.25 | 0.15 | 0.29 | 0.01→0.19 | 0.84→0.31 |
| F1v_v2_G__j9b | 0.35 | 0.17 | 0.32 | 0.06 | 0.06 | 0.19 | 0.03→0.21 | 0.67→0.20 |
| F1v_v3_G (4b) | 0.06 | 0.00 | 0.11 | 0.00 | 0.00 | 0.03 | 0.25→0.52 | 0.39→0.13 |
| F1_v3_G__9b (text) | 0.40 | 0.45 | 0.47 | 0.31 | 0.20 | 0.37 | 0.04→0.09 | 0.66→0.29 |
| F1v_v3_G__9b | 0.55 | 0.25 | 0.37 | 0.12 | 0.05 | 0.25 | 0.08→0.57 | 0.66→0.21 |
| F1v_v3_G__j9b | 0.41 | 0.22 | 0.26 | 0.06 | 0.06 | 0.14 | 0.20→0.72 | 0.54→0.16 |
| F1v_v3c_G__9b | 0.50 | 0.40 | 0.37 | 0.15 | 0.05 | 0.30 | 0.09→0.60 | 0.73→0.16 |

**Reading.**
- **4b is degenerate under §6.** Gate∧θ ≈0 at every level. The root cause is at item level: the 4b judge scores basis and benefit anti-correlated. For F1, u = min(basis, benefit) counts are 516/17/1 for u = 0/1/2 (worklog).
- **Only the 9b image judge tracks level** (0.50 → 0.15 for v2; 0.55 → 0.05 for v3).
- **v3 makes `resolved` rise with level only when the judge sees the image.** The 9b text judge v3 stays at 0.04 → 0.09.

**Ad-hoc tally for h3.** This tally was computed by this report from the raw decision files; it is not a summary file. Comparing `decisions/F1v_v3_G__9b_h3` with `F1v_v3_G__9b` at t1/5/10/15/20, the resolved item share is:
- h3: 0.14 / 0.12 / 0.32 / 0.38 / 0.56
- no history: 0.08 / 0.37 / 0.30 / 0.38 / 0.57

Low_resolution share at t5 is 0.71 (h3) vs 0.54. So history mainly makes the judge more conservative at t5. Stop effect: **pending** (needs 20 levels + replay).

**v4 smoke** (`summaries/v4_summary_F1g_v4_G__9b.json`, 5 states per level, 0 failures, ≈11.7 s/state): mean clear 0.62 / 0.75 / 0.88 / 1.15 / 1.23 at t1/5/10/15/20, and clear = 2 share 0.24 → 0.58. Full-run results: **pending**.

### 5.3 Signal and judge-quality diagnostics

Source: `dev_r0/eval/signal_compare.csv` / `report.md`. Rule = `proposed`. AUC predicts gain_left > 0 on qmax > 0 states; ρ is Spearman with gain_left over pooled states.

| Variant (levels) | n | ρ R50 (p) | AUC R50 | ρ relsg (p) | AUC relsg |
|---|---|---|---|---|---|
| F1 (5) | 90 | 0.176 (0.098) | 0.598 | 0.200 (0.059) | 0.546 |
| F2 (5) | 86 | −0.129 (0.238) | 0.446 | −0.074 (0.497) | 0.459 |
| F2__9b (5) | 92 | −0.032 (0.764) | 0.502 | −0.077 (0.468) | 0.488 |
| F1v (5) | 88 | −0.009 | 0.446 | 0.007 | 0.494 |
| F1v_v2 (5) | 86 | −0.078 | 0.475 | −0.024 | 0.521 |
| F1v_v2_G (20) | 359 | 0.014 (0.784) | 0.488 | 0.052 (0.330) | 0.510 |
| F1_v2_G (20) | 361 | 0.020 (0.704) | 0.501 | 0.054 (0.310) | 0.551 |
| F1v_v2_G__9b (20) | 384 | 0.032 (0.533) | 0.532 | **0.171 (0.001)** | 0.570 |
| F1_v2_G__9b (20) | 383 | −0.095 (0.062) | 0.514 | −0.028 (0.579) | 0.526 |
| F1v_v2_G__j9b (5) | 88 | 0.178 (0.097) | 0.670 | 0.296 (0.005) | 0.599 |
| F1v_v3_G__9b (20) | 384 | −0.007 (0.897) | 0.527 | 0.133 (0.009) | 0.555 |
| F1v_v3_G__j9b (20) | 366 | 0.007 (0.888) | 0.540 | 0.164 (0.002) | 0.541 |
| F1v_v3c_G__9b (5) | 98 | 0.060 (0.555) | 0.565 | 0.337 (0.001) | 0.676 |

The relsg_gt signal is significant only for 9b image-judge variants, and weak (ρ ≤ 0.34). Under R50 every variant is at chance, because per-image R50 is too noisy.

**Judge quality.** Does `resolved` / `low_resolution` track q_t/qmax? sufficient = q/qmax ≥ 0.9; relsg_gt; `report.md` "Judge quality" table.

| Variant | ρ(resolved, q/qmax) | AUC(sufficient \| resolved) | AUC(sufficient \| 1−C) | ρ(low_res, q/qmax) | AUC(sufficient \| 1−low_res) | n_sufficient |
|---|---|---|---|---|---|---|
| F1 | 0.019 | 0.497 | 0.554 | −0.132 | 0.576 | 20 |
| F1v_v2_G | 0.095 | 0.528 | 0.496 | −0.176 | 0.611 | 62 |
| F1v_v2_G__9b | 0.202 | 0.513 | 0.574 | −0.243 | 0.602 | 69 |
| F1v_v2_G__j9b | 0.196 | 0.510 | 0.607 | −0.406 | **0.787** | 18 |
| F1v_v3_G (4b) | 0.215 | **0.641** | 0.569 | −0.407 | 0.724 | 19 |
| F1v_v3_G__9b | 0.252 | 0.573 | 0.567 | −0.241 | 0.588 | 69 |
| F1v_v3_G__j9b | 0.332 | 0.616 | 0.538 | −0.310 | 0.615 | 62 |
| F1v_v3c_G__9b | 0.351 | 0.612 | 0.665 | −0.327 | 0.591 | 21 |
| F1_v3_G__9b (text) | 0.150 | 0.511 | 0.592 | −0.307 | 0.624 | 21 |

- Under v1/v2, `resolved` is essentially unused even when q/qmax ≥ 0.9 (worklog: ≈2 % for 4b, ≈13 % for 9b).
- The strongest single cue is the share of items labelled low_resolution. This is a diagnostic, not a rule.
- v3 makes `resolved` informative (AUC 0.57–0.64), but see §5.6: the effect is uniform across hypothesis classes.

### 5.4 20-level replays (dev20)

Source: `dev_r0/report.md`. q_stop − eq_fixed is the paired difference vs the equal-observation fixed mix with a 95 % bootstrap CI. Oracle MAE uses the relative_stable oracle (n = 6 for R50, 14 for relsg_gt). fixed_20 = 0.196 (R50) / 0.596 (relsg_gt), with oracle MAE 3.167 / 1.571.

| Variant | rule | mean stop | R50 q_stop | R50 Δ eq-fixed [CI] | relsg q_stop | relsg Δ eq-fixed [CI] | MAE R50 / relsg |
|---|---|---|---|---|---|---|---|
| F1v_v2_G (4b) | proposed | 2.35 | 0.000 | 0.000 [0.000, 0.000] | 0.129 | −0.010 [−0.044, 0.015] | 14.33 / 16.07 |
| | unknown_cons. | 3.50 | 0.017 | 0.017 [0.000, 0.050] | 0.171 | −0.005 [−0.039, 0.037] | 12.33 / 14.57 |
| F1_v2_G (4b) | proposed | 2.20 | 0.000 | 0.000 [0.000, 0.000] | 0.139 | 0.004 [−0.012, 0.023] | 14.67 / 16.21 |
| | unknown_cons. | 3.55 | 0.000 | 0.000 [−0.001, 0.000] | 0.154 | −0.024 [−0.077, 0.028] | 13.17 / 14.71 |
| F1v_v2_G__9b | proposed | 4.90 | 0.040 | 0.034 [−0.017, 0.079] | 0.268 | 0.001 [−0.093, 0.089] | 10.00 / 12.50 |
| | observed_only | 4.85 | 0.033 | 0.027 [−0.024, 0.076] | 0.260 | −0.003 [−0.097, 0.084] | 10.17 / 12.57 |
| | unknown_cons. | 5.65 | 0.040 | 0.016 [−0.044, 0.069] | 0.280 | −0.043 [−0.120, 0.070] | 10.00 / 12.14 |
| F1_v2_G__9b | proposed | 4.20 | 0.000 | −0.001 [−0.038, 0.000] | 0.196 | −0.017 [−0.140, 0.042] | 13.50 / 14.14 |
| | unknown_cons. | 8.95 | 0.070 | −0.049 [−0.110, 0.032] | 0.349 | −0.043 [−0.132, 0.015] | 10.33 / 10.79 |
| F1v_v3_G__9b | proposed (= observed_only) | 4.20 | 0.000 | −0.001 [−0.029, 0.000] | 0.229 | 0.016 [−0.082, 0.080] | 12.00 / 13.57 |
| | unknown_cons. | 5.00 | 0.000 | −0.007 [−0.054, 0.000] | 0.225 | −0.050 [−0.143, 0.039] | 11.83 / 13.43 |
| F1v_v3_G__j9b | proposed | 3.20 | 0.000 | 0.000 [0.000, 0.000] | 0.219 | **0.056 [0.006, 0.101]** | 13.83 / 15.07 |
| | observed_only | 3.10 | 0.000 | 0.000 [0.000, 0.000] | 0.212 | **0.053 [0.007, 0.099]** | 13.83 / 15.21 |
| | unknown_cons. | 3.85 | 0.000 | 0.000 [−0.006, 0.000] | 0.222 | 0.031 [−0.007, 0.071] | 13.50 / 14.50 |

**H contribution** (proposed − observed_only, q_stop):
- F1v_v2_G__9b: +0.007 [0.000, 0.021] (R50) and +0.008 [0.000, 0.024] (relsg).
- F1v_v3_G__j9b: +0.007 [0.000, 0.021] (relsg).
- 0.000 for every other 20-level variant.

**Reading.**
- Every rule stops early: mean 2.2–2.35 for 4b, 4–5 for 9b, and up to 8.95 only for unknown_conservative with the 9b text judge. 0 forced stops.
- All rules sit on the fixed-stop curve. Gaps to fixed_20 are large, e.g. F1v_v2_G__9b proposed −0.155 (R50) / −0.328 (relsg) (worklog).
- 2 of 60 Δ-vs-eq-fixed CIs exclude 0 (6 variants × 5 rules incl. exploratory × 2 metrics). Both are in F1v_v3_G__j9b under relsg_gt, and both for proposed and observed_only. That points to the judge, not H. Given n = 20 and the multiplicity, treat it as unconfirmed.
- Auxiliary-oracle MAEs (monotone_stable / gain_based, n = 13 for R50, 20 for relsg) show the same ordering: every adaptive rule is worse than fixed_20 (e.g. relsg gain_based: fixed_20 5.90 vs 9.95–11.90 for adaptive rules).

### 5.5 Rule variants and PGSG-internal signals

- **Rule variants.** `benefit_gated` (exploratory) and θ = 1 (`unknown_conservative`) do not recover a signal (§5.3 and §5.4 tables). Example: on 4b F1, benefit_gated AUC is 0.598, the same as proposed; F2 is ≤ chance under every rule. Conclusion (worklog): aggregation-rule changes cannot fix a content problem.
- **PGSG-internal signals** (§4.3) are proxies for level only, so they serve as a negative control.

### 5.6 Failure analysis

Sources: `dev_r0/failure_analysis/summary.md` (5 levels) and `failure_analysis_full20/summary.md`. Zero calls; GT is used post hoc only.

Hypothesis classes:
- **bookkeeping**: talks about the graph itself.
- **gt_supported**: names a GT class absent from the rendered graph. It fires on any mentioned class, so it is an upper bound.
- **already_observed**: names a class already in the rendered graph.
- **off_vocab**.

| Checks source (shared by) | graph | n hyp | bookkeep. | gt_supp. | already_obs. | off_vocab | anchor in top-5 | obs. restating scores | GT labels absent (headroom) | gt_supp. hyp/state |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 (F1v) 4b v1 | raw | 276 | 43 % | 0 % | 50 % | 7 % | 28 % | 40 % | 28 % (2.4/img) | 0.01 |
| F2 4b v1 | raw | 270 | 41 % | 0 % | 58 % | 1 % | 29 % | 37 % | 28 % | 0.00 |
| F2__9b | raw | 264 | 20 % | 0 % | 71 % | 9 % | 22 % | 24 % | 28 % | 0.01 |
| F1_v2 (F1v_v2) 4b | raw | 240 | 52 % | 0 % | 45 % | 3 % | 10 % | 53 % | 27 % | 0.01 |
| F1_v2_G (F1v_v2_G, j9b, v3_G, v3_G__j9b) 4b | G | 192 | 36 % | 4 % | 55 % | 5 % | 16 % | 36 % | 51 % (4.3/img) | 0.09 |
| F1v_v2_G__9b (= v3_G__9b) 9b | G | 268 | 9 % | 9 % | 66 % | 16 % | 54 % | 17 % | 51 % | 0.25 |
| F1v_v3c_G__9b 9b | G | 273 | 14 % | 5 % | 64 % | 16 % | 58 % | 18 % | 51 % | 0.15 |
| full20: F1v_v3_G__9b | G | 1061 | 10 % | 7 % | 67 % | 16 % | 54 % | 15 % | 50 % (4.2/img) | 0.19 |
| full20: F1v_v3_G__j9b | G | 831 | 34 % | 6 % | 52 % | 8 % | 10 % | 35 % | 50 % | 0.14 |

Headroom by level:
- Raw rendering (F1): GT labels absent 56 % (t1) → 40 % (t5) → 18 % (t10) → 15 % (t15) → 11 % (t20). Realized gt_supported ≈ 0 at every level.
- G rendering: 75 % (t1) → 34 % (t20).

Omission is a generation failure, not a lack of headroom.

How the judge treats each class (status shares):

| Variant | bookkeeping | gt_supported | already_observed | off_vocab |
|---|---|---|---|---|
| F1 (4b v1) | 89 % unresolved / 2 % resolved | n = 1, unresolved | 86 % / 1 % | 67 % / 0 % |
| F1v_v2_G__9b (v2) | — | n = 24: **100 % unresolved** | — | — |
| F1v_v3_G__9b (v3, 5 levels) | — | n = 24: 54 % unresolved / 46 % resolved | — | — |
| F1v_v3_G__9b (v3, full20) | 58 % / 41 % | n = 74: 51 % / 47 % | 64 % / 33 % | 46 % / 54 % |
| F1v_v3_G__j9b (v3, full20) | 65 % / 32 % | n = 51: 63 % / 33 % | 69 % / 27 % | 62 % / 33 % |

**Reading.**
- 4b → 9b (G, v2) cuts bookkeeping 36 → 9 % and raises gt_supported 4 → 9 %.
- Under v2 the judge never resolves gt_supported hypotheses (0/24).
- Under v3, resolved rises with level in **every** class, including bookkeeping. With 9b checks, gt_supported items are ≈50 % resolved already at t1–t10 with no level dependence (worklog). These are "already visible at low res" cases, not confirmations by resolution.
- So v3 gives a resolution-sufficiency estimate that is not content-aware. v3c brings no hypothesis-quality gain (gt_supported 9 → 5 %).

### 5.7 Image-only resolution probe (diagnostic)

Sources: `dev_r0/probe/*/summary.json`, `eval/probe/probe_compare.csv`. Script `29c`; 20 images × 20 levels per config.

| cfg | failures / 400 | prompt tokens (median) | detail_fraction t1 / t5 / t10 / t15 / t20 | sufficient at t20 | next_step_helps at t20 |
|---|---|---|---|---|---|
| 4b @384 | 2 | 207 | 5 / 13 / 24 / 39 / 65 | 0.20 | 0.80 |
| 4b @768 | 29 | 495 | 5 / 5 / 22.9 / 35.5 / 61.8 | 0.05 | 0.95 |
| 9b @384 | 10 | 207 | 5 / 6 / 16.6 / 32.4 / 79.7 | 0.00 | 1.00 |
| 9b @768 | 13 | 495 | 4.5 / 5 / 10 / 23.9 / 78.8 | 0.00 | 1.00 |

- `sufficient` is 0 % at every level below t20 for all configs.
- 9b@384 (`probe_compare.csv`): ρ(detail, t) = 0.82; ρ(detail, q/qmax) = 0.48 (R50) / 0.58 (relsg); AUC(detail → sufficient) = 0.714 / 0.716 (higher than every judge signal); 65 % of images monotone.
- Probe-based stop policies collapse to fixed_20 (mean stop 20.0), because "sufficient" is never true and "next step helps" is always true.
- Implication (worklog): the VLM perceives blur ordinally even at 384 px. The item judge's "benefit = 0" at low levels is a reasoning/prompt failure, not a perception failure. Going to 768 px (2.4× tokens) does not help and adds format failures. Resolution-threshold policies were not pursued (user: task-specific).

### 5.8 Model calls and failures

Source: `report.md` "Model calls".
- **Failure rates.** 4b checks failed 6–11 % per variant; 9b checks 2–5 %. The dominant cause is `anchor_not_in_graph`: G keeps sparse original ids, and a dense-renumbering fix was proposed but not implemented. Judge failures are mostly inherited from check failures.
- **Latency.** Median latency was ≈3.2–5.3 s per call (v2/v3), against ≈10–13 s for the v1 4b prompts with 60/50 caps.
- **Retry.** The temp-0.3 retry recovers about half of schema failures (worklog).

## 6. Scaled-down MIRU experiment (EGTR, old MIRU 100 images)

### 6.1 Setup

Source: `outputs/miru_gt_eval/report_scaled_down.md` §1.
- **Pipeline.** MIRU unchanged: 100 images, 20 bilinear levels (original size), cached EGTR predictions. Gates: object 0.3 input / relation 0.005; top 20 objects / 30 relations per level. The SGG history covers the last 5 levels, scored at levels 5–20. The prompt is `SATURATION_PROMPT`.
- **New factor.** The current-level image is attached (one added sentence; ≤512 px, JPEG q85, ≈220 tokens). 2×2 = {text, img_single} × {4b, 9b}, 1600 states each (`scripts/40`).
- **Decoding.** JSON mode, `think: false`, temp 0, seed 42, num_ctx 16384. Failures: 3/1600 for img 9b, 0 elsewhere.
- **Why text-only was re-run.** The legacy cache predates the EGTR gate change (0.3 → 0.1, 2026-06-20) and used temp 0.3 without JSON mode.
- **Stop rule and metrics.** MIRU `consecutive_k_stop`, θ 0.70, k = 2. Primary MAE is against `stop_c09` (mean 13.48). The GT-based secondary (R50 and relsg_gt against VG GT) comes from `scripts/42`.
- **Caveat.** EGTR stored ≤23 relations per state, so "R50" here is recall over the stored relations.

### 6.2 Method table (k = 2)

Source: `outputs/miru_gt_eval/miru_method_table.md`.

| method | source | mean stop | MAE c09 | R50 @stop | R50 Δ eq-fixed | relsg @stop | relsg Δ eq-fixed | relsg Δ fixed20 |
|---|---|---|---|---|---|---|---|---|
| llm_sgg_text_4b | new text | 13.76 | 3.00 | 0.167 | +0.003 [−0.024, +0.032] | 0.404 | −0.005 [−0.026, +0.015] | −0.074 [−0.103, −0.047] |
| llm_sgg_text_9b | new text | 10.32 | 4.40 | 0.098 | +0.011 [−0.021, +0.042] | 0.287 | **−0.033 [−0.059, −0.005]** | −0.192 |
| vlm_sgg_single_4b | img + SGG | 9.34 | 5.04 | 0.090 | +0.014 [−0.004, +0.034] | 0.301 | +0.017 [−0.006, +0.040] | −0.177 |
| vlm_sgg_single_9b | img + SGG | 9.44 | 4.78 | 0.087 | +0.010 [−0.017, +0.040] | 0.281 | −0.008 [−0.032, +0.017] | −0.198 |
| hybrid_llm_sgg_text_4b | new text + node conf. | 13.58 | 2.98 | 0.165 | +0.004 | 0.399 | −0.005 | −0.079 |
| llm_sgg_4b_h5 | legacy text cache | 14.21 | 3.27 | 0.170 | −0.002 | 0.416 | −0.004 | −0.062 |
| hybrid_4b | legacy (MIRU published best) | 13.92 | 2.82 | 0.164 | −0.004 [−0.030, +0.025] | 0.407 | −0.007 [−0.027, +0.015] | −0.071 |
| vlm_single_9b | legacy image-only, θ 0.70 | 14.32 | 3.04 | 0.159 | −0.014 [−0.036, +0.008] | 0.406 | −0.017 [−0.035, +0.002] | −0.073 |
| vlm_9b_single | legacy `vlm_stop.csv`, θ 0.8 (stale) | 16.14 | 3.76 | 0.198 | −0.002 | 0.437 | −0.012 | −0.041 |
| vlm_single_4b | legacy image-only, θ 0.70 | 12.66 | 5.20 | 0.122 | −0.021 | 0.331 | **−0.048 [−0.075, −0.021]** | −0.147 |
| baseline | node confidence | 14.31 | 3.49 | 0.159 | −0.014 [−0.038, +0.015] | 0.391 | **−0.032 [−0.054, −0.007]** | −0.088 |
| fixed_20 | — | 20.00 | 6.52 | 0.240 | n/a | 0.478 | n/a | n/a |

Fixed-stop curve (`miru_gt_eval/eval/summary.csv`):

| fixed level | 5 | 10 | 14 | 15 | 20 |
|---|---|---|---|---|---|
| mean R50 | 0.005 | 0.083 | 0.169 | 0.181 | 0.240 |
| mean relsg_gt | 0.134 | 0.313 | 0.416 | 0.436 | 0.478 |

### 6.3 Why the image hurts

Source: `outputs/llm_scores_vlm/diag_img_vs_text.md`; `report_scaled_down.md` §4.
- **Scores shift up.** Mean paired shift img − text on the same state is +0.13 (4b) and +0.05 (9b). img ≥ 0.70 while text < 0.70 in 30 % of states (484) for 4b and 16 % (255) for 9b; the reverse occurs in 1 % / 3 %.
- **The shift peaks exactly where stops should wait.** At t8–t11 the 4b shift is +0.21 to +0.27, with 48–60 % of states flipping. Share ≥ θ at t9: 4b 74 % (img) vs 16 % (text); 9b 48 % vs 29 %.
- **Confabulated stability.** In flipped states, image+SGG reasons mention visual words in only 1 % (4b) / 7 % (9b) of cases, against 82 % for the image-only VLM. They claim the graph has stabilized in 99 % / 89 %, while text-only calls the identical sequence unstable in 95 % of cases.
- **The blur cue is lost.** Image-only 9b has 2 % (t5) / 14 % (t9) of states ≥ θ; image + SGG 9b has 45 % / 48 %.
- **Mechanism.** Not identifiable from post-hoc reasons. Candidates: attention dilution, or a "complete-looking scene" prior.

### 6.4 Stale-threshold finding

- MIRU's `outputs/metrics/vlm_stop.csv` (and the earlier `method_comparison_*.csv`) used θ = 0.8, while LLM/hybrid methods used 0.70 and the config now says 0.7.
- Recomputed at θ 0.70 from the same score files: image-only VLM 9b MAE 3.04 (published 3.76), essentially tied with SGG-text 4b (3.00; legacy 3.27).
- GT q at stop is also similar: R50 0.159 vs 0.167, relsg 0.406 vs 0.404.
- MIRU's "SGG-text beats image-VLM" margin largely disappears at equal θ.

### 6.5 Late fusion (post hoc) and pre-registered held-out test

Post-hoc fusion rows (`miru_method_table.md`, zero-call; `src/milestone1/miru_methods.py`). Six fusion variants were inspected on the same 100 images.

| method (k = 2) | mean stop | MAE c09 | R50 Δ eq-fixed | relsg @stop | relsg Δ eq-fixed | relsg Δ fixed20 |
|---|---|---|---|---|---|---|
| hybrid_fuse_mean(vlmimg9b, text4b) | 13.49 | **2.59** | +0.001 [−0.022, +0.023] | 0.397 | −0.005 [−0.022, +0.012] | −0.081 [−0.109, −0.056] |
| fuse_mean(vlmimg9b, text4b) | 13.44 | 2.84 | +0.009 [−0.014, +0.033] | 0.407 | +0.006 [−0.010, +0.023] | −0.071 |
| fuse_min(vlmimg9b, text4b) | 16.16 | 3.12 | +0.001 [−0.017, +0.018] | 0.455 | +0.006 [−0.007, +0.020] | **−0.023 [−0.042, −0.004]** |
| hybrid_fuse_min(vlmimg9b, text4b) | 15.28 | 2.92 | −0.001 | 0.435 | −0.005 | −0.044 |
| fuse_mean(vlmimg9b, text9b) | 12.41 | 2.93 | −0.011 | 0.357 | −0.016 | −0.121 |
| hybrid_fuse_mean(vlmimg9b, text9b) | 12.93 | 2.71 | −0.007 | 0.376 | −0.010 | −0.102 |

- Fusion gives the lowest in-sample MAE (2.59).
- fuse_min stops later and costs only −0.023 relsg vs fixed_20, at ≈81 % of the observation levels.
- Still, no fusion row separates from the equal-observation fixed mix. The MAE gain is an optimistic in-sample estimate.

**Pre-registration** (`configs/config_heldout.yaml` header, recorded 2026-10-08 ≈00:40 before any held-out score):
- **Image set.** ROBOVIS test100 minus 4 MIRU-overlap images (2417333, 2417743, 2415349, 2415332) = 96 images (`data/processed/robovis2027_miru/selection_note.json`). MIRU conditions throughout: original-size degradation, EGTR gates 0.1/0.005, top-k 20/30, 5-level history.
- **Primary rule.** `hybrid_fuse_mean(vlm image-only 9b, llm_sgg text 4b)`, θ 0.70, k = 2. **Secondary:** `fuse_min`.
- **Primary set.** All 96, scored by MAE vs stop_c09 recomputed from held-out EGTR L20. **Comparative set:** stop_c09 ≤ 18 (73 images).
- **Comparators.** Text 4b, image-only 9b (script 11, MIRU decoding: temp 0.3, no JSON, unseeded), hybrid text 4b, node-confidence baseline, fixed curve.
- **GT check.** Equal-observation Δ on both sets. No tuning.

**Held-out data facts** (worklog):
- EGTR 1920/1920 states; cross-checks exact.
- Distribution shift vs MIRU 100: stop_c09 mean 15.90 vs 13.48; 21/96 never saturate (c09 = 20).
- The held-out GT curve is lower: t20 R50 0.159 vs 0.240; relsg 0.430 vs 0.478.

Current `outputs/heldout_eval/miru_method_table.md`:

| method | mean stop | MAE c09 (all 96) | MAE c09 (≤18, n = 73) | R50 @stop | R50 Δ eq-fixed | relsg @stop | relsg Δ eq-fixed | relsg Δ fixed20 |
|---|---|---|---|---|---|---|---|---|
| baseline (node conf. ≥ 0.35) | 14.729 | 3.458 | 3.397 | 0.108 | −0.026 [−0.050, −0.004] | 0.361 | −0.029 [−0.048, −0.010] | −0.069 [−0.092, −0.046] |
| fixed_20 | 20.000 | 4.104 | 5.370 | 0.159 | n/a | 0.430 | n/a | n/a |

**Status: PENDING.** Only baseline and fixed_20 rows exist. `outputs/robovis2027_miru/logs/run_heldout_scoring.sh` (PID 127475 alive at snapshot) is waiting for a GO flag that is set when the h3/v4 ROBOVIS run frees ollama. Its log is empty: no text-4b or VLM-9b held-out scores exist yet. The primary and secondary pre-registered outcomes are therefore unknown.

## 7. Synthesis

### 7.1 Hypothesis status

| Claim | Status | Key evidence |
|---|---|---|
| H1: hypotheses H improve recall at equal observation | **Not supported** (dev20, every variant) | proposed − observed_only ≤ +0.008 (CI lower bound 0); proposed ≡ observed_only in 4/6 20-level variants (§5.4) |
| H2: trade-off with over-observation | **Not testable** | no H1 gain |
| Judge signals track resolution sufficiency | **Partly**: 9b + image only; weak | ρ_relsg ≤ 0.17 on 20 levels; AUC(resolved) 0.57–0.64 under v3 (§5.3) |
| Adaptive ROBOVIS stop beats the fixed curve | **Not supported** | 58/60 CIs include 0; the 2 exceptions are in one variant (j9b relsg) |
| Adding the current image to the SGG judge (MIRU) improves stopping | **Refuted** (hurts) | MAE 3.00 → 5.04 (4b), 4.40 → 4.78 (9b) (§6.2) |
| Adaptive stopping beats fixed stopping per observation under GT scoring | **Not supported** for any method, incl. MIRU's and fusion | every Δ eq-fixed CI ⊇ 0 or is negative (§6.2, §6.5) |
| Late fusion improves MAE | **In-sample only (post hoc)**; held-out **pending** | 2.59 vs 3.00 (§6.5) |

### 7.2 What worked and what did not

What worked:
- **Infrastructure.** PGSG on local VG with exact per-image SGDet, a fully logged LLM pipeline, the replay/bootstrap/report generator, and a GT-based secondary for EGTR (327 tests per worklog).
- **G halves hallucination** (det_prec20 0.20 → 0.36) at −2.5 pt R@50.
- **9b image judge** gives the first level-tracking signal (gate∧θ 0.50 → 0.15).
- **v3 makes `resolved` informative.**
- **The image-only probe reads blur ordinally** (ρ 0.82).
- **Text-4b reproduces MIRU** (MAE 3.00 vs legacy 3.27; per-level scores within ≈0.04).
- **Late fusion** gives the best in-sample MAE.

What did not:
- **The plan's §6 rule with 4b is degenerate** (basis/benefit anti-correlated; stop ≈2.3).
- **F2 (text-only) is at chance regardless of model size.**
- **v2 prompts** shifted limitation away from low_resolution without producing both-2 items.
- **Rule changes** (benefit_gated, θ = 1) did not help.
- **PGSG-internal signals** proxy only t.
- **Hypotheses** rarely name genuinely missing GT objects (0 % raw, 4–9 % G).
- **The judge cannot convert perception into "sufficient"**: probe sufficient = 0 %; item judge benefit = 0 at low levels.
- **Early image + SGG fusion** confabulates stability.

### 7.3 Defensible claims for a paper (negative/analytical)

1. A GT-referenced, equal-observation evaluation framework for resolution-aware stopping (fixed-stop curve + paired bootstrap), applied to two observers (EGTR, PGSG).
2. Adaptive stopping methods (node confidence, LLM-on-SGG, image VLM, hybrids, fusion, VLM check/judge pipelines) match oracle timing to within ≈3 levels (MIRU) but do not exceed the fixed-stop curve in GT recall per observed level.
3. Failure analysis:
   - Single-snapshot absolute sufficiency judgments by local VLMs are calibration-bound: blur is perceived ordinally, yet "sufficient" is never asserted.
   - Early fusion of image + SGG history induces confabulated "graph stable" reasoning.
   - Co-occurrence hypotheses from local VLMs rarely name genuinely missing objects despite large headroom.
4. Correction to MIRU: the VLM-vs-LLM comparison was threshold-inconsistent; at equal θ they tie.
5. Observer facts: a dense generative SGG (PGSG) makes per-image saturation oracles ill-defined (≈1/3 defined), and recall-only metrics hide hallucination.

### 7.4 Limitations

- **ROBOVIS results are dev20 only**, n = 20 images. Oracle MAE uses n = 6 (R50) / 14 (relsg). The test100 stop experiment was not run.
- **Single sample.** One PGSG sampling run and one LLM decode per state. qmax is upward-biased, and per-image R50 is dominated by sampling noise. 3-seed smoothing was prepared but not run.
- **Multiple comparisons.** About 16 named variants × 5 rules × 2 metrics, plus 6 post-hoc fusion variants. Isolated "significant" cells (j9b +0.056) are not corrected.
- **Scope of variants.** 5-level variants give diagnostics only; their replays are forced to t20.
- **Data overlap and shift.** The held-out set shares 4 → 0 images with MIRU 100 after exclusion, but has a different difficulty distribution (21/96 never saturate; MIRU 100 had been filtered to c09 ≤ 18). This is why both the all-96 and ≤18 sets are pre-registered.
- **GT.** VG GT is incomplete: missing GT counts as error, and trip_prec ≈ 0.04 even at full resolution. EGTR "R50" is recall over ≤23 stored relations.
- **Schedule.** The 384-px base and the level-20 no-resample step make the t20 relsg jump partly schedule-inherent. Label recall is still rising at full resolution.
- **Failure-analysis classes** are heuristic. gt_supported fires on any mentioned GT class (upper bound). Counts per cell are small.
- **Judge v3** received one piece of schedule information (≈17 % per side per step), a user-approved deviation.
- **Dataset record conflict.** The number of right-censored images removed from MIRU 100 differs between sources (29 vs 35).

## 8. Reproduction

All commands run from the repo root. ROBOVIS scripts use `uv run --project egtr python …`; PGSG inference uses the `envs/pgsg` environment as wired in script 25. Check each script's `--help` for the full flag set.

| Step | Command (as recorded in worklog/summary) | Output |
|---|---|---|
| PGSG convert / official eval | `scripts/23_pgsg_convert_vg.py`, `scripts/24_pgsg_eval.sh` | `data/processed/pgsg_vg/`, `outputs/pgsg/` |
| Image selection / levels | `scripts/26_robovis_select_images.py`, `scripts/27_robovis_generate_levels.py` | `data/processed/robovis2027/{selected_images,selection_manifest,levels_manifest}.json`, `degraded_images/` |
| PGSG inference | `scripts/25_pgsg_infer.py --run-id dev_r0 --split dev` (and `test_r0 --split test`) | `outputs/robovis2027/<run>/graphs/`, `manifest.json` |
| Stage metrics | `scripts/28_robovis_stage_metrics.py --run-id <run> [--validate-official]` | `stage_metrics.csv`, `stage_summary.json`, `official_validation.json` |
| Preflight | `scripts/33_robovis_preflight.py` | `outputs/robovis2027/preflight.json` |
| PGSG signals | `scripts/34_robovis_pgsg_signals.py --runs dev_r0,test_r0` | `pgsg_signals.csv`, `test_r0/pgsg_signals_report.md` |
| Clean graph G | `scripts/35_robovis_clean_graphs.py --runs …` | `<run>/graphs_clean/` |
| Checks + judge | e.g. `scripts/29_robovis_checks.py --run-id dev_r0 --split dev --levels 1,5,10,15,20 --path F1 --judge --graphs-subdir graphs_clean --prompt-version v3 --judge-with-image --check-model qwen3.5:9b --judge-model qwen3.5:9b --variant-name F1v_v3_G__9b [--graph-history 3] --num-ctx 16384` | `dev_r0/checks/<variant>/`, `decisions/<variant>/`, `usage.jsonl` |
| Check summaries | `scripts/29b_robovis_checks_summary.py --run-id dev_r0 --variant <V> --graphs-subdir graphs_clean` | `dev_r0/summaries/checks_summary_<V>.json` |
| Probe | `scripts/29c_robovis_view_probe.py --run-id dev_r0 --model qwen3.5:9b --image-size 384`; diag `scripts/41_robovis_probe_diag.py` | `dev_r0/probe/<cfg>/`, `eval/probe/` |
| Judge v4 | `scripts/29d_robovis_grounded_v4.py --run-id dev_r0 --model qwen3.5:9b --levels 1..20 --num-ctx 16384` | `decisions/F1g_v4_G__9b/`, `overlays/` |
| Replay / eval | `scripts/30_robovis_replay_eval.py --run-id dev_r0 --path <V> [--q-metric relsg_gt]` | `dev_r0/eval/<V>/[q_relsg_gt/]` |
| Figures | `scripts/31_robovis_figures.py`, `scripts/37_robovis_fig1_pipeline.py` | `eval/<V>/figures/`, `outputs/robovis2027/figures/` |
| Signal diag | `scripts/32_robovis_signal_diag.py --run-id dev_r0 --paths …` | `eval/<V>/signal_*`, `eval/signal_compare.csv` |
| Report | `scripts/36_robovis_report.py --run-id dev_r0 --paths … --q-metrics R50,relsg_gt --refresh --graphs-subdir auto` | `dev_r0/report.md` |
| Failure analysis | `scripts/38_robovis_failure_analysis.py --run-id dev_r0` | `dev_r0/failure_analysis*/summary.md` |
| Oracle profile | `scripts/39_robovis_oracle_profile.py` | `<run>/eval/oracle_profile.csv` |
| Multiseed (not run) | `scripts/37_robovis_multiseed.sh <run> <split>` | — |
| MIRU 2×2 scoring | `scripts/40_vlm_sgg_saturation_scoring.py` (`--plan qwen3.5:4b=text` etc.) | `outputs/llm_scores_vlm/<slug>/` |
| MIRU GT eval / fusion | `scripts/42_miru_gt_eval.py` (`--dataset heldout` for held-out) | `outputs/miru_gt_eval/`, `outputs/heldout_eval/` |
| Held-out scoring | `MILESTONE1_CONFIG_OVERLAY=configs/config_heldout.yaml bash outputs/robovis2027_miru/logs/run_heldout_scoring.sh` (scripts 40 + 11) | `outputs/robovis2027_miru/{llm_scores_vlm,vlm_scores}/` (pending) |

Git commits on the branch (`git log --oneline main..HEAD`):

```
995d289 Add consolidated ROBOVIS2027 experiment summary and document map
6a71445 Scaled-down MIRU experiment: 2x2 img/text x 4b/9b results, late fusion, GT-based secondary, write-up
8b6e6b2 MIRU-based VLM+SGG saturation scoring (2x2), MIRU eval registry + GT-based secondary, probe diag, v4 grounded judge, graph-history input
bb4e22e ROBOVIS2027: judge v3 (lenient resolved, next-step gain), judge-quality diag, probe diag, failure analysis full-level, graph-dir auto routing
c1d3c81 ROBOVIS2027: auxiliary oracles, report scope per plan, graphs-subdir auto routing, failure analysis + headroom, multiseed prep, config key repair
5534d05 ROBOVIS2027: cleaned observed graph G, GT-RelSGScore, precision metrics, q_metric switch, signal diag, PGSG-internal signals, preflight, report generator, Fig1/Fig3
1dd1ba2 ROBOVIS2027: PGSG inference wrapper, per-image SGDet, check generator/judge (ollama), stop policy, replay eval
8f4830d ROBOVIS2027: GT-only image re-selection (test100/dev20, seed 42) and 384px-base 20-level degradation
e820065 Add PGSG (Pix2Grp CVPR2024) submodule, uv env, VG-150 converter and eval wrapper
```

The working tree also holds uncommitted outputs and code changes: held-out config/eval, `miru_methods.py`, `vlm.py`, scripts 11/42, the h3/v4 decision directories and `outputs/heldout_eval/`.

## Appendix A: Glossary of variant names

| Token | Meaning |
|---|---|
| F1 | PGSG fixed; local VLM sees image + graph → checks (≤3 observed issues + ≤3 hypotheses); text judge (plan §5) |
| F2 | text LLM sees graph only → checks; text judge |
| P | PGSG co-generates CHECKS: dropped, not run |
| F1v | same checks as F1, but the judge also receives the image |
| F1g | judge v4 grounded visibility (boxes drawn) |
| _v2 | prompt v2: scene-focused hypotheses, independent basis/benefit definitions with a both-2 few-shot, <200-char hint; rendering merges same-label IoU>0.5, caps 30 entities / 20 relations |
| _v3 | judge v3 (checks unchanged = v2): lenient `resolved` (≈90 % visible), told next view ≈17 % sharper per side; no level/scale/steps |
| _v3c | v3 check prompt (region-anchor emphasis) + v3 judge |
| _v4 | per-entity clear ∈ {0, 1, 2, absent} for the top-8 G entities; clear ≤ 1 → observed issue (unresolved, low_resolution, basis 2, benefit 2/1); clear 2 → resolved; H reused from v2 9b |
| _G | input graph = cleaned observed graph G (`graphs_clean/`); q_t still scored on raw output |
| __9b | checks and judge both `qwen3.5:9b` (default: 4b) |
| __j9b | 4b checks + 9b judge |
| _h3 | judge/checks also get the previous 3 G graphs (labels + relation triples, oldest first, no ids/scores/levels); image = current only |
| proposed / observed_only | C_t from issues + hypotheses / issues only (plan §6) |
| unknown_conservative | θ = 1 variant (plan-named) |
| benefit_gated(_obs) | exploratory: u = benefit if gate ∧ basis ≥ 1 (needs sign-off) |
| eq-fixed | equal-observation fixed-level mix with the same mean stop |
| relsg_gt | 0.5·node recall + 0.5·triplet recall vs VG GT label sets, top-20 entities / top-50 relations |
| stop_c09 | MIRU EGTR-referenced oracle (top-10/top-5, coef 0.9) |
| MIRU slugs | `llm_sgg_text_{4b,9b}` (SGG-text), `vlm_sgg_single_*` (image + SGG), `vlm_single_*` (image-only VLM), `hybrid_*` (0.5 node confidence + 0.5 score, θ 0.5), `fuse_{mean,min}_vlmimg9b_text4b` (late fusion) |

## Appendix B: File map

| Topic | Path |
|---|---|
| Index / verdicts | `.claude/docs/research/robovis2027-summary.md` |
| Chronology / decisions | `.claude/docs/research/robovis2027-worklog.md` |
| Feasibility | `.claude/docs/research/robovis2027-feasibility.md` |
| PGSG internals / run log / VG compat | `.claude/docs/research/{pgsg-pipeline,pgsg-run-log,pix2grp-vg-compat}.md` |
| Deviations | `outputs/robovis2027/report_notes.md` |
| Preflight | `outputs/robovis2027/preflight.json` |
| Path P rationale + dev SGDet | `outputs/robovis2027/dev_r0/pilot_report.md` |
| Full dev report (16 paths) | `outputs/robovis2027/dev_r0/report.md` |
| Signal table | `outputs/robovis2027/dev_r0/eval/signal_compare.csv` |
| Per-variant gate tables | `outputs/robovis2027/dev_r0/summaries/checks_summary_*.json`, `v4_summary_F1g_v4_G__9b.json` |
| Failure analysis | `outputs/robovis2027/dev_r0/failure_analysis{,_full20}/summary.md` |
| Probe | `outputs/robovis2027/dev_r0/probe/*/summary.json`, `eval/probe/probe_compare.csv` |
| Test recall / oracle | `outputs/robovis2027/test_r0/recall_profile.md`, `eval/oracle_profile.csv` |
| PGSG-internal signals | `outputs/robovis2027/test_r0/pgsg_signals_report.md`, `pgsg_signals_corr.csv` |
| Configs | `configs/robovis2027.yaml`, `configs/config.yaml` (`vlm_sgg`, `evaluation_extra`), `configs/config_heldout.yaml` |
| Selection | `data/processed/robovis2027/selection_manifest.json`, `data/processed/robovis2027_miru/selection_note.json` |
| MIRU scaled-down | `outputs/miru_gt_eval/report_scaled_down.md`, `miru_method_table.{md,csv}`, `eval/` |
| Image-vs-text diagnosis | `outputs/llm_scores_vlm/diag_img_vs_text.md` |
| Held-out (pending) | `outputs/heldout_eval/miru_method_table.md`, `outputs/robovis2027_miru/logs/` |
| Pending ROBOVIS runs | `outputs/robovis2027/dev_r0/decisions/F1v_v3_G__9b_h3/` (complete, unsummarized), `decisions/F1g_v4_G__9b/` (in progress) |

---
**Addendum (2026-10-08):** the follow-up recovery plan and its results (Phases A–D) are in `outputs/robovis2027_recovery/rec_r0/REPORT.md`. Note the comparator correction there: the "Δ vs eq_fixed" columns in this report use the legacy adjacent fixed mix, which is weaker than the best fixed mixture F(b) by up to 0.05 R50; against F no dev20 rule is ≥ 0. Also, per later guidance, fixed-level stops are schedule-aware references rather than fair comparators for an online observer.
