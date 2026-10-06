# PGSG (Pix2Grp, CVPR 2024) inference pipeline — as implemented in the VG checkpoint

Source: `Pix2Grp_CVPR2024/lavis/models/blip_models/blip_rel_det_pgsg.py` (class `BlipRelDetectionPGSG`), model yaml `lavis/configs/models/blip_vrd_base_vg.yaml`. Line numbers refer to that file at submodule commit cff7420.

## Architecture
- Image encoder: BLIP ViT-B/16, input 384×384 (aspect ratio not preserved; `blip_det_image_eval`).
- Text decoder: BLIP "MED" = BERT-base (12 layers, 768 d, `med_config_det.json`) with cross-attention to image tokens, wrapped as `XBertLMHeadDecoderDetHead` (L1720). Vocabulary = BERT 30,522 + ~10 special tokens (`[DEC]`, `[ENC]`, `[OBJ]`, `[REL]`, ...). `num_coord_bin: 640` coordinate tokens exist in the base class, but **the VG checkpoint runs with `pos_adapter: True`, so the generated sequence contains no coordinate tokens** (target format at L321: `"{sub} [OBJ] {pred} [REL] {obj} [OBJ]"`).
- Position adapter (L1729–1756, forward L1771): entity hidden states → linear proj → 6-layer transformer encoder over image features (+ sine pos. emb.) → 6-layer decoder with entity states as queries → 3-layer MLP → sigmoid box (normalized cx,cy,w,h). This is the paper's "entity grounding" module.
- Pretrained from `model_base_capfilt_large.pth` (BLIP captioning), then fine-tuned on VG-150 sequences. Not instruction-tuned.

## Inference flow (`generate`, L495–680)
1. Encode image → `image_embeds`.
2. Fixed prompt `"The visual scene of: "` tokenized, `[DEC]` as BOS, trailing `[SEP]` stripped (L545–549). `max_txt_len=40`.
3. Segmented sampling (L564–590): loop `max_length // int((seg_len + 24)/2)` times (seg_len 64 in the model yaml; eval configs may override; with max_length 1024 this is a few dozen segments). Each iteration calls `generate_from_encoder` **from the same prompt** with nucleus sampling (top_p 0.9 default, `LogitsWarper(top_k=60)`), `max_length = seg_len + prompt_len`, `min_length = 24 + prompt_len`, `repetition_penalty`, returning scores and hidden states. Segments are concatenated (prompt stripped) into one long pseudo-sequence; hidden states and per-step score tensors are concatenated too. Greedy/beam (`generation_mode=search`) makes every segment identical → collapse (observed R@50 ≈ 5).
4. `seq2instance` (L756): scans tokens; each `[REL]` marker anchors a triplet; it searches backward (≤15 tokens) for the subject span ending in `[OBJ]`, forward for the object span. Keeps, per role, the token ids, the decoder hidden states (`dec_hs`), the per-step softmax over the vocabulary (`pred_scores`), and token positions. Malformed instances are counted as `bad_inst` and skipped.
5. `vocab2category` (L118): for each role, re-projects the hidden states through the LM head, applies the same top-k warper, softmax, then `get_classifier_scores` sums probability mass over the label-token ranges of each VG category (synonyms averaged, "多个词对应一个类别"); produces `scores_mapped` (150-d for entities, 50-d for predicates) and `labels_mapped`. This is the "category conversion" step.
6. Grounding (L636–660): subject/object hidden states (`int_tok_hs[-1] + dec_hs[-1]`) → `pos_adapter` → boxes written back into each instance (`['boxes']`).
7. Optional `aux_close_set_classifier` (off: `aux_close_classifier: False`).
8. `_postprocess` (L1011–1620): per triplet, entity label = top-1 of `scores_mapped` (`top_k_ent_label_num: 1`), predicate expanded to top-5 (`top_k_predicate_label_num: 5`), scores amplified (`ent_ampl_scale 5.0`, `rel_ampl_scale 2.0`), triplet score = f(sub, pred, obj scores); sort by triplet score; entity NMS at IoU 0.92 (L1196); relation dedup via `rel_prediction_filtering` (same labels and both boxes IoU > 0.8 → drop); keep top ranked (512 in logs). Boxes → xyxy in `orig_size` via `xywh2xyxy`. Output is a `BoxList` with fields `pred_labels`, `pred_scores`, `rel_pair_idxs`, `pred_rel_label`, `pred_rel_score`, `pred_rel_trp_score`, plus distributions.
9. `_gtpostprocess` builds GT `BoxList` from `samples['targets']`; evaluator computes SGDet R@K / mR@K with graph constraint (IoU ≥ 0.5, label match).

## Signals available without changing the model (relevant to a P′ "uncertainty-as-hypothesis" variant)
- Per-entity 150-way and per-predicate 50-way distributions (`scores_mapped`), their max/entropy.
- Which segment (sample) each triplet came from (token positions), hence cross-sample agreement among the segments.
- Free-vocabulary entity tokens before conversion (`label_tkn`), including nouns outside the 150 classes (more so with `vg_ov_sgg.pth`).
- Box regression comes from the adapter, not from token probabilities; there is no explicit box confidence. Agreement of boxes across segments is the only grounding-uncertainty proxy.
- `bad_inst` counts (malformed triplets) per image.

## Practical notes
- Stock `evaluate.py` does not save per-image predictions; `dump_pred` exists (`dump_prediction_res`, L682) but is off in the eval config.
- Time ≈ 3 s/img at max_length 1024 (RTX 5080), 1.5 s at 512.
- Fixed `torch.manual_seed` makes sampling reproducible.
