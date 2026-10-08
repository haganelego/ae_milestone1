# GQA question-decomposition early stop: experiment report

Generated 2026-10-08T18:09:46+00:00 from artifacts in `/home/kanaoka/usr/github/milestone_1/outputs/gqa_stop/visapp_gqa_v1`. Spec v1.1 (plan §0-§16).

## 1. Scope

Same-view resolution recovery over 20 levels on GQA val_balanced questions (research dev/test split by image; not official GQA test). Stop scores are self-reported confidences, not probabilities. Unlike DeCC (EMNLP 2024), which checks consistency between direct and reconstructed answers, this uses sub-question confidences to choose a stop level; DeCC-style stopping is not evaluated. 'No training' means no parameter updates; dev thresholds were selected. Levels are not physical distances or compute units.

## 2. Data flow

Total questions 132062; eligible 9148 on 5467 images.

| reason | count |
|---|---|
| answer_word_in_question | 3 |
| eligible | 9148 |
| final_argument_not_target_attribute | 8802 |
| negation | 112 |
| not_semantic_attr | 50075 |
| not_structural_query | 63895 |
| question_text_attribute_mismatch | 27 |

| split | correct | incorrect |
|---|---|---|
| dev | 166 | 99 |
| test | 404 | 235 |

| split | attribute | relate_count | n |
|---|---|---|---|
| dev | color | 0 | 20 |
| dev | color | 1 | 20 |
| dev | material | 0 | 20 |
| dev | material | 1 | 20 |
| dev | shape | 0 | 12 |
| dev | shape | 1 | 8 |
| test | color | 0 | 40 |
| test | color | 1 | 40 |
| test | material | 0 | 40 |
| test | material | 1 | 40 |
| test | shape | 0 | 27 |
| test | shape | 1 | 13 |

Design sample balanced by attribute and target-identification difficulty (§16 A2); no claim about the natural GQA distribution.

## 3. Model, prompts, images, decoding

Model `qwen3.5:9b` digest `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`; temperature 0.0, seed 20261008, think=False, num_ctx 8192, num_predict {'answer_only': 128, 'direct': 256, 'planner': 1024, 'checklist': 1024}.

Prompt hashes: `{"answer_only": "fc5deb1de883fbfb6f0c608105bc5ba38f8a526fe097c81a90d5f72cab23e2e6", "direct": "5d4362425d503ec550ff3891523ae4351ce3c5e5d06bc23a2c0b808571e9de76", "planner": "f76b4e6c1588e0b7a509aab4d47bd57cdd5bed258039d3a395f5886aadc37d53", "checklist": "301066505171b6e8e1d15c5ee989495cddb10fed6e34d7930ba27fd6e04d86a5", "schema_repair": "479b1aec010dba0f6c281eccdec3e538340607aa17a518b764966082336877d8"}`.

Images: 20 levels, s_t = 0.05 x 20^((t-1)/19), base max side 512 (Lanczos), bilinear down/up, PNG bytes sent as stored.

## 4. Probe, pilot and calibration gate

`probe_summary.json`:
```json
{
 "questions": 198,
 "per_cell": {
  "color|r0": 33,
  "color|r1": 33,
  "material|r0": 33,
  "material|r1": 33,
  "shape|r0": 33,
  "shape|r1": 33
 },
 "levels": [
  1,
  5,
  10,
  20
 ],
 "reader_requests": 792,
 "outputs": [
  "/home/kanaoka/usr/github/milestone_1/outputs/gqa_stop/visapp_gqa_v1/probe",
  "/home/kanaoka/usr/github/milestone_1/outputs/gqa_stop/visapp_gqa_v1/probe_summary.json"
 ],
 "overall_level_accuracy": {
  "1": 0.3686868686868687,
  "5": 0.4393939393939394,
  "10": 0.5252525252525253,
  "20": 0.6363636363636364
 },
 "first_correct_level_histogram": {
  "never": 64,
  "5": 22,
  "1": 73,
  "20": 23,
  "10": 16
 },
 "status_counts": {
  "valid": 792
 },
 "decision_hint": "A3: if every cell has L1 accuracy >= 0.9 shift quota to smaller targets / material / shape; never drop a cell (shrink only)."
}
```

`pilot_summary.json`:
```json
{
 "model": "qwen3.5:9b",
 "digest": "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7",
 "model_tag": "qwen3.5-9b@6488c96fa5fa",
 "prompt_hashes": {
  "answer_only": "fc5deb1de883fbfb6f0c608105bc5ba38f8a526fe097c81a90d5f72cab23e2e6",
  "direct": "5d4362425d503ec550ff3891523ae4351ce3c5e5d06bc23a2c0b808571e9de76",
  "planner": "f76b4e6c1588e0b7a509aab4d47bd57cdd5bed258039d3a395f5886aadc37d53",
  "checklist": "301066505171b6e8e1d15c5ee989495cddb10fed6e34d7930ba27fd6e04d86a5",
  "schema_repair": "479b1aec010dba0f6c281eccdec3e538340607aa17a518b764966082336877d8"
 },
 "questions": 20,
 "images": 19,
 "levels": [
  1,
  5,
  10,
  15,
  20
 ],
 "parse_version": "parse-v2",
 "valid_rate_initial": {
  "reader": 1.0,
  "direct": 1.0,
  "checklist": 0.99,
  "planner": 1.0
 },
 "valid_rate_after_repair": {
  "reader": 1.0,
  "direct": 1.0,
  "checklist": 1.0,
  "planner": 1.0
 },
 "repair_rate": {
  "reader": 0.0,
  "direct": 0.0,
  "checklist": 0.01,
  "planner": 0.0
 },
 "format_gate_uses": "valid_rate_initial",
 "format_valid_target": 0.95,
 "format_valid_ok": true,
 "reader_accuracy_by_level": {
  "1": 0.55,
  "5": 0.7,
  "10": 0.85,
  "15": 0.9,
  "20": 1.0
 },
 "n_items_distribution": {
  "2": 14,
  "3": 6
 },
 "level1_headroom_note": "L1 reader accuracy >= 0.95 means little room to adapt"
}
```

`pilot_calibration.json`:
```json
{
 "model": {
  "name": "qwen3.5:9b",
  "digest": "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
 },
 "gate_config": {
  "min_spearman": 0.3,
  "max_level1_mean_confidence": 0.8,
  "early_overconfidence_threshold": 0.8,
  "fallback_model_allowed_at_pilot_only": true
 },
 "levels": [
  1,
  5,
  10,
  15,
  20
 ],
 "n_questions": 20,
 "direct": {
  "n": 100,
  "per_level": {
   "1": {
    "n": 20,
    "mean": 0.5275000000000001,
    "std": 0.26574188604734483
   },
   "5": {
    "n": 20,
    "mean": 0.6925,
    "std": 0.2471209218176397
   },
   "10": {
    "n": 20,
    "mean": 0.8725000000000002,
    "std": 0.21935986415021322
   },
   "15": {
    "n": 20,
    "mean": 0.97,
    "std": 0.05338539126015657
   },
   "20": {
    "n": 20,
    "mean": 0.9974999999999999,
    "std": 0.010897247358851692
   }
  },
  "lowest_level_mean": 0.5275000000000001,
  "spearman_pooled": 0.8146726247835906,
  "spearman_per_question_median": 0.9215552445252148,
  "questions_with_constant_confidence": 0,
  "early_overconfidence_pairs": 9,
  "pairs_before_first_correct": 18,
  "early_overconfidence_rate": 0.5,
  "fails": false
 },
 "checklist_effective_r": {
  "n": 230,
  "per_level": {
   "1": {
    "n": 46,
    "mean": 0.058695652173913045,
    "std": 0.22221815266174985
   },
   "5": {
    "n": 46,
    "mean": 0.2782608695652174,
    "std": 0.40266127931564694
   },
   "10": {
    "n": 46,
    "mean": 0.601086956521739,
    "std": 0.3996587036398495
   },
   "15": {
    "n": 46,
    "mean": 0.6336956521739131,
    "std": 0.37950974493790746
   },
   "20": {
    "n": 46,
    "mean": 0.7271739130434783,
    "std": 0.3120246857812087
   }
  },
  "lowest_level_mean": 0.058695652173913045,
  "spearman_pooled": 0.501341058862959,
  "spearman_per_question_median": 0.5270462766947298,
  "questions_with_constant_confidence": 1,
  "early_overconfidence_pairs": 14,
  "pairs_before_first_correct": 43,
  "early_overconfidence_rate": 0.32558139534883723,
  "fails": false
 },
 "first_correct_level_histogram": {
  "1": 11,
  "5": 4,
  "10": 2,
  "15": 2,
  "20": 1
 },
 "gate_passed": true,
 "gate_rule": "fail iff for both Direct and checklist r_j: pooled rho < min_spearman or lowest-pilot-level mean confidence >= max_level1_mean_confidence"
}
```

`screening_repeat.json`:
```json
{
 "n": 20,
 "agreement_rate": 1.0,
 "repeat_accuracy": 1.0,
 "backend": "ollama /api/chat, temperature 0, fixed seed (GPU may be non-deterministic)",
 "note": "diagnostic only; never changes the selected set"
}
```

## 5. Dev thresholds and frozen policy

| target | method | theta | dev_accuracy | dev_mean_level |
|---|---|---|---|---|
| 0.95 | direct | inf | 1.000 | 20.000 |
| 0.95 | checklist_min | 0.850 | 0.980 | 18.910 |
| 0.95 | checklist_mean | 0.870 | 0.980 | 18.790 |
| 0.95 | fixed_dev | fixed L17 | n/a | n/a |
| 0.90 | direct | 1.000 | 0.910 | 11.350 |
| 0.90 | checklist_min | 0.850 | 0.980 | 18.910 |
| 0.90 | checklist_mean | 0.870 | 0.980 | 18.790 |
| 0.90 | fixed_dev | fixed L14 | n/a | n/a |
| 0.97 | direct | inf | 1.000 | 20.000 |
| 0.97 | checklist_min | 0.850 | 0.980 | 18.910 |
| 0.97 | checklist_mean | 0.870 | 0.980 | 18.790 |
| 0.97 | fixed_dev | fixed L20 | n/a | n/a |
| 0.99 | direct | inf | 1.000 | 20.000 |
| 0.99 | checklist_min | 0.900 | 1.000 | 19.970 |
| 0.99 | checklist_mean | 0.880 | 1.000 | 19.340 |
| 0.99 | fixed_dev | fixed L20 | n/a | n/a |

![dev curves](figures/dev_threshold_curves.png)

Direct's frozen theta may be inf (always Level 20); then the MeanLevel difference is trivial (§16 A8).

## 6. Test results

| method | theta | accuracy | mean_level | median_level | forced20_rate | never_met_rate | natural_pass20_rate | stop20_rate | status | accuracy_loss_vs_l20 | mean_observed_level_reduction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_5 | n/a | 0.690 | 5.000 | 5.000 | n/a | n/a | n/a | 0.000 | n/a | 0.310 | 0.750 |
| fixed_10 | n/a | 0.825 | 10.000 | 10.000 | n/a | n/a | n/a | 0.000 | n/a | 0.175 | 0.500 |
| fixed_15 | n/a | 0.940 | 15.000 | 15.000 | n/a | n/a | n/a | 0.000 | n/a | 0.060 | 0.250 |
| fixed_20 | n/a | 1.000 | 20.000 | 20.000 | n/a | n/a | n/a | 1.000 | n/a | 0.000 | 0.000 |
| direct@0.95 | inf | 1.000 | 20.000 | 20.000 | 1.000 | 1.000 | 0.000 | 1.000 | ok | 0.000 | 0.000 |
| checklist_min@0.95 | 0.850 | 0.970 | 17.985 | 20.000 | 0.745 | 0.690 | 0.055 | 0.745 | ok | 0.030 | 0.101 |
| checklist_mean@0.95 | 0.870 | 0.970 | 17.985 | 20.000 | 0.745 | 0.690 | 0.055 | 0.745 | ok | 0.030 | 0.101 |
| fixed_dev@0.95 | n/a | 0.955 | 17.000 | 17.000 | n/a | n/a | n/a | 0.000 | ok | 0.045 | 0.150 |

Primary target only; secondary targets are in paired_comparisons.csv.

| comparison | metric | estimate | ci_low | ci_high | verdict |
|---|---|---|---|---|---|
| direct@0.95 | hull_gap | 0.000 | 0.000 | 0.000 | n/a |
| direct@0.95 | adjacent_gap | 0.000 | 0.000 | 0.000 | n/a |
| checklist_min@0.95 | hull_gap | -0.010 | -0.040 | 0.010 | n/a |
| checklist_min@0.95 | adjacent_gap | -0.005 | -0.035 | 0.031 | n/a |
| checklist_mean@0.95 | hull_gap | -0.010 | -0.040 | 0.010 | n/a |
| checklist_mean@0.95 | adjacent_gap | -0.005 | -0.035 | 0.031 | n/a |
| fixed_dev@0.95 | hull_gap | -0.015 | -0.044 | 0.000 | n/a |
| fixed_dev@0.95 | adjacent_gap | 0.000 | 0.000 | 0.000 | n/a |
| checklist_min@0.95-direct@0.95 | accuracy_diff | -0.030 | -0.055 | -0.010 | n/a |
| checklist_min@0.95-direct@0.95 | mean_level_diff | -2.015 | -2.680 | -1.446 | n/a |
| checklist_min@0.95-checklist_mean@0.95 | accuracy_diff | 0.000 | 0.000 | 0.000 | n/a |
| checklist_min@0.95-checklist_mean@0.95 | mean_level_diff | 0.000 | 0.000 | 0.000 | n/a |
| checklist_min@0.95-direct@0.95 | verdict | n/a | n/a | n/a | trade_off_or_unclear: earlier, accuracy CI crosses -margin |
| checklist_min@0.95-checklist_mean@0.95 | verdict | n/a | n/a | n/a | unclear: mean-level difference CI includes 0 |

![tradeoff](figures/test_meanlevel_accuracy.png)

F(b) is the upper bound of level-agnostic randomised fixed policies, not a learned policy; falling below it is not by itself a failure (§16 A5).

![stops](figures/test_stop_level_hist.png)

| group_by | value | method | n | accuracy | mean_level |
|---|---|---|---|---|---|
| attribute | color | direct@0.95 | 80 | 1.000 | 20.000 |
| attribute | color | checklist_min@0.95 | 80 | 0.963 | 18.062 |
| attribute | color | checklist_mean@0.95 | 80 | 0.963 | 18.062 |
| attribute | color | fixed_dev@0.95 | 80 | 0.963 | 17.000 |
| attribute | material | direct@0.95 | 80 | 1.000 | 20.000 |
| attribute | material | checklist_min@0.95 | 80 | 0.975 | 18.462 |
| attribute | material | checklist_mean@0.95 | 80 | 0.975 | 18.462 |
| attribute | material | fixed_dev@0.95 | 80 | 0.938 | 17.000 |
| attribute | shape | direct@0.95 | 40 | 1.000 | 20.000 |
| attribute | shape | checklist_min@0.95 | 40 | 0.975 | 16.875 |
| attribute | shape | checklist_mean@0.95 | 40 | 0.975 | 16.875 |
| attribute | shape | fixed_dev@0.95 | 40 | 0.975 | 17.000 |
| relate_count | 0 | direct@0.95 | 107 | 1.000 | 20.000 |
| relate_count | 0 | checklist_min@0.95 | 107 | 0.981 | 17.981 |
| relate_count | 0 | checklist_mean@0.95 | 107 | 0.981 | 17.981 |
| relate_count | 0 | fixed_dev@0.95 | 107 | 0.972 | 17.000 |
| relate_count | 1 | direct@0.95 | 93 | 1.000 | 20.000 |
| relate_count | 1 | checklist_min@0.95 | 93 | 0.957 | 17.989 |
| relate_count | 1 | checklist_mean@0.95 | 93 | 0.957 | 17.989 |
| relate_count | 1 | fixed_dev@0.95 | 93 | 0.935 | 17.000 |

### Examples (first two per category in question-hash order)

**min_early_and_correct**

- qhash `2515e5304b`: What's the floor made of?
  - checklist: ["Is there a visible floor in the image?", "If yes, what material is the identified floor made of?"]
  - scores: L1 d=0.85 min=0.00 mean=0.45 ans=wood; L2 d=0.95 min=0.70 mean=0.80 ans=wood; L3 d=0.95 min=0.70 mean=0.80 ans=wood; L4 d=0.95 min=0.80 mean=0.85 ans=wood; L5 d=0.95 min=0.00 mean=0.45 ans=wood; L6 d=0.95 min=0.85 mean=0.88 ans=wood; L7 d=0.95 min=0.85 mean=0.88 ans=wood; L8 d=0.95 min=0.85 mean=0.88 ans=wood; L9 d=0.95 min=0.85 mean=0.88 ans=wood; L10 d=0.95 min=0.85 mean=0.88 ans=wood; L11 d=1.00 min=0.85 mean=0.88 ans=wood; L12 d=1.00 min=0.85 mean=0.88 ans=wood; L13 d=0.95 min=0.85 mean=0.88 ans=wood; L14 d=1.00 min=0.85 mean=0.88 ans=wood; L15 d=1.00 min=0.85 mean=0.88 ans=wood; L16 d=1.00 min=0.85 mean=0.88 ans=wood; L17 d=1.00 min=0.85 mean=0.88 ans=wood; L18 d=1.00 min=0.85 mean=0.88 ans=wood; L19 d=1.00 min=0.85 mean=0.88 ans=wood; L20 d=1.00 min=0.85 mean=0.88 ans=wood
  - direct@0.95: stop L20, answer=wood, correct=True
  - checklist_min@0.95: stop L6, answer=wood, correct=True
  - checklist_mean@0.95: stop L6, answer=wood, correct=True
  - direct@0.90: stop L11, answer=wood, correct=True
  - checklist_min@0.90: stop L6, answer=wood, correct=True
  - checklist_mean@0.90: stop L6, answer=wood, correct=True
  - direct@0.97: stop L20, answer=wood, correct=True
  - checklist_min@0.97: stop L6, answer=wood, correct=True
  - checklist_mean@0.97: stop L6, answer=wood, correct=True
  - direct@0.99: stop L20, answer=wood, correct=True
  - checklist_min@0.99: stop L20, answer=wood, correct=True
  - checklist_mean@0.99: stop L20, answer=wood, correct=True

- qhash `d0105978ec`: What is the floor made of?
  - checklist: ["Is there a visible floor in the image?", "What material is the identified floor made of?"]
  - scores: L1 d=0.40 min=0.00 mean=0.00 ans=wood; L2 d=0.40 min=0.00 mean=0.00 ans=wood; L3 d=0.10 min=0.00 mean=0.00 ans=wood; L4 d=0.10 min=0.00 mean=0.00 ans=wood; L5 d=0.65 min=0.00 mean=0.45 ans=wood; L6 d=0.15 min=0.00 mean=0.45 ans=tile; L7 d=0.50 min=0.00 mean=0.45 ans=tile; L8 d=0.80 min=0.00 mean=0.45 ans=tile; L9 d=0.85 min=0.00 mean=0.45 ans=wood; L10 d=0.85 min=0.00 mean=0.45 ans=tile; L11 d=0.85 min=0.00 mean=0.45 ans=wood; L12 d=0.50 min=0.00 mean=0.45 ans=wood; L13 d=0.85 min=0.00 mean=0.45 ans=wood; L14 d=0.85 min=0.70 mean=0.80 ans=wood; L15 d=0.85 min=0.70 mean=0.80 ans=wood; L16 d=0.85 min=0.80 mean=0.85 ans=wood; L17 d=0.85 min=0.80 mean=0.85 ans=wood; L18 d=1.00 min=0.85 mean=0.88 ans=wood; L19 d=1.00 min=0.85 mean=0.88 ans=wood; L20 d=1.00 min=0.85 mean=0.88 ans=wood
  - direct@0.95: stop L20, answer=wood, correct=True
  - checklist_min@0.95: stop L18, answer=wood, correct=True
  - checklist_mean@0.95: stop L18, answer=wood, correct=True
  - direct@0.90: stop L18, answer=wood, correct=True
  - checklist_min@0.90: stop L18, answer=wood, correct=True
  - checklist_mean@0.90: stop L18, answer=wood, correct=True
  - direct@0.97: stop L20, answer=wood, correct=True
  - checklist_min@0.97: stop L18, answer=wood, correct=True
  - checklist_mean@0.97: stop L18, answer=wood, correct=True
  - direct@0.99: stop L20, answer=wood, correct=True
  - checklist_min@0.99: stop L20, answer=wood, correct=True
  - checklist_mean@0.99: stop L20, answer=wood, correct=True

**direct_early_wrong**: none

**min_too_late**

- qhash `66ca7d645c`: What is the floor made of?
  - checklist: ["Is there a visible floor in the image?", "What material is the identified floor made of?"]
  - scores: L1 d=0.85 min=0.00 mean=0.45 ans=concrete; L2 d=0.85 min=0.00 mean=0.45 ans=concrete; L3 d=0.85 min=0.00 mean=0.45 ans=concrete; L4 d=0.85 min=0.00 mean=0.45 ans=concrete; L5 d=0.85 min=0.00 mean=0.45 ans=concrete; L6 d=0.85 min=0.00 mean=0.45 ans=concrete; L7 d=0.85 min=0.00 mean=0.45 ans=concrete; L8 d=0.95 min=0.00 mean=0.45 ans=brick; L9 d=0.95 min=0.00 mean=0.45 ans=brick; L10 d=0.85 min=0.00 mean=0.45 ans=brick; L11 d=0.95 min=0.70 mean=0.80 ans=brick; L12 d=0.90 min=0.80 mean=0.85 ans=brick; L13 d=0.95 min=0.70 mean=0.80 ans=brick; L14 d=0.95 min=0.80 mean=0.85 ans=brick; L15 d=0.95 min=0.80 mean=0.85 ans=brick; L16 d=1.00 min=0.80 mean=0.85 ans=brick; L17 d=1.00 min=0.80 mean=0.85 ans=brick; L18 d=1.00 min=0.80 mean=0.85 ans=brick; L19 d=1.00 min=0.80 mean=0.85 ans=brick; L20 d=1.00 min=0.80 mean=0.85 ans=brick
  - direct@0.95: stop L20, answer=brick, correct=True
  - checklist_min@0.95: stop L20, answer=brick, correct=True
  - checklist_mean@0.95: stop L20, answer=brick, correct=True
  - direct@0.90: stop L16, answer=brick, correct=True
  - checklist_min@0.90: stop L20, answer=brick, correct=True
  - checklist_mean@0.90: stop L20, answer=brick, correct=True
  - direct@0.97: stop L20, answer=brick, correct=True
  - checklist_min@0.97: stop L20, answer=brick, correct=True
  - checklist_mean@0.97: stop L20, answer=brick, correct=True
  - direct@0.99: stop L20, answer=brick, correct=True
  - checklist_min@0.99: stop L20, answer=brick, correct=True
  - checklist_mean@0.99: stop L20, answer=brick, correct=True

- qhash `2515e5304b`: What's the floor made of?
  - checklist: ["Is there a visible floor in the image?", "If yes, what material is the identified floor made of?"]
  - scores: L1 d=0.85 min=0.00 mean=0.45 ans=wood; L2 d=0.95 min=0.70 mean=0.80 ans=wood; L3 d=0.95 min=0.70 mean=0.80 ans=wood; L4 d=0.95 min=0.80 mean=0.85 ans=wood; L5 d=0.95 min=0.00 mean=0.45 ans=wood; L6 d=0.95 min=0.85 mean=0.88 ans=wood; L7 d=0.95 min=0.85 mean=0.88 ans=wood; L8 d=0.95 min=0.85 mean=0.88 ans=wood; L9 d=0.95 min=0.85 mean=0.88 ans=wood; L10 d=0.95 min=0.85 mean=0.88 ans=wood; L11 d=1.00 min=0.85 mean=0.88 ans=wood; L12 d=1.00 min=0.85 mean=0.88 ans=wood; L13 d=0.95 min=0.85 mean=0.88 ans=wood; L14 d=1.00 min=0.85 mean=0.88 ans=wood; L15 d=1.00 min=0.85 mean=0.88 ans=wood; L16 d=1.00 min=0.85 mean=0.88 ans=wood; L17 d=1.00 min=0.85 mean=0.88 ans=wood; L18 d=1.00 min=0.85 mean=0.88 ans=wood; L19 d=1.00 min=0.85 mean=0.88 ans=wood; L20 d=1.00 min=0.85 mean=0.88 ans=wood
  - direct@0.95: stop L20, answer=wood, correct=True
  - checklist_min@0.95: stop L6, answer=wood, correct=True
  - checklist_mean@0.95: stop L6, answer=wood, correct=True
  - direct@0.90: stop L11, answer=wood, correct=True
  - checklist_min@0.90: stop L6, answer=wood, correct=True
  - checklist_mean@0.90: stop L6, answer=wood, correct=True
  - direct@0.97: stop L20, answer=wood, correct=True
  - checklist_min@0.97: stop L6, answer=wood, correct=True
  - checklist_mean@0.97: stop L6, answer=wood, correct=True
  - direct@0.99: stop L20, answer=wood, correct=True
  - checklist_min@0.99: stop L20, answer=wood, correct=True
  - checklist_mean@0.99: stop L20, answer=wood, correct=True

**both_wrong**: none

## 7. Diagnostics and limitations

Technical failures logged: 22.

Pilot gates: no override recorded.

Diagnostics: `{"rows": 4000, "direct_reader_disagreement_rate": 0.16725, "checklist_unresolved_item_rate": 0.37383333333333335, "planner_failures": 0, "n_items_distribution": {"2": 174, "3": 26}, "status_counts": {"reader": {"valid": 4000}, "direct": {"valid": 4000}, "check": {"valid": 4000}}, "surface_variant_errors": 6, "surface_variant_rule": "incorrect and one normalised string contains the other"}`

Exact match after fixed normalisation (not the official GQA evaluator); surface variants such as 'light blue' vs 'blue' count as errors. The screened set is 'questions the reader answered correctly at L20 in this run', so L20 accuracy is 1 by construction. GQA/VG may be in the model's pre-training data. With ~200 test questions the accuracy CI half-width is about 4 pp, so the -5 pp non-inferiority check is close to inconclusive by design (§16 A8).

## 8. Re-run commands

```bash
uv run python -m milestone1.gqa_stop.cli preflight --config configs/gqa_stop.json
uv run python -m milestone1.gqa_stop.cli probe --config configs/gqa_stop.json
uv run python -m milestone1.gqa_stop.cli prepare --config configs/gqa_stop.json
uv run python -m milestone1.gqa_stop.cli screen --split dev --config configs/gqa_stop.json
uv run python -m milestone1.gqa_stop.cli pilot --split dev --levels 1,5,10,15,20 --calibration --config configs/gqa_stop.json
uv run python -m milestone1.gqa_stop.cli infer --split dev --levels all --config configs/gqa_stop.json
uv run python -m milestone1.gqa_stop.cli freeze --config configs/gqa_stop.json
uv run python -m milestone1.gqa_stop.cli screen --split test --config configs/gqa_stop.json
uv run python -m milestone1.gqa_stop.cli infer --split test --levels all --config configs/gqa_stop.json
uv run python -m milestone1.gqa_stop.cli evaluate --config configs/gqa_stop.json
uv run python -m milestone1.gqa_stop.cli report --config configs/gqa_stop.json
```

## 9. Completion

| stage | status |
|---|---|
| preflight | completed |
| probe | completed |
| prepare | completed |
| screen_dev | completed |
| pilot | completed |
| pilot_calibration | completed |
| infer_dev | completed |
| freeze | completed |
| screen_test | completed |
| infer_test | completed |
| evaluate | completed |
