# PGSG (Pix2Grp, CVPR 2024) on local VG-150: compatibility report

Date: 2026-10-07. Scope: read-only investigation of `Pix2Grp_CVPR2024/` (nothing in the submodule or in `egtr/` was modified).
Abbreviation: `P = /home/kanaoka/usr/github/milestone_1/Pix2Grp_CVPR2024`.

## 0. Verdict

**It can run, after a conversion step.** The local EGTR-format VG-150 holds everything PGSG needs: the same 150 object classes and 50 predicates in the same order, the same Xu/Tang split (26,446 test images), boxes and image sizes. Only the file format is different: PGSG wants Tang-style `VG-SGG-*.h5` plus `image_data.json`. A conversion script of about 120 LOC, written outside the submodule, closes that gap. **All six fine-tuned PGSG checkpoints are already on disk.** The BLIP pretrained weights are not needed for evaluation.

Two blockers have nothing to do with the data:
1. **The GPU is unusable right now.** The NVIDIA kernel module is 580.159.03, but the userspace library is 580.173.02 (`nvidia-smi` reports "Driver/library version mismatch"). This blocks EGTR too. Fix: reboot, or reload the nvidia kernel modules.
2. **The pinned environment cannot run on this GPU.** PGSG pins torch 2.0.1 and transformers 4.29.2. The GPU is PCI `10de:2c02`, very likely an RTX 5080 (Blackwell, sm_120), which needs a CUDA 12.8 build (torch ≥ 2.7). So PGSG needs a **separate** uv env with torch 2.7.x+cu128 and transformers 4.29.2. Nobody has tested that combination, so there is some risk.

---

## 1. Data PGSG expects for VG-150

### Docs
- `P/DATASET.MD` L4-31: "same protocol as BGNN / Tang's Scene-Graph-Benchmark". You download the images plus an annotation bundle `vg_motif_anno` from the author's ShanghaiTech SharePoint. **That link now returns HTTP 403.** You then copy `all_categories_dict/vg/vg_motif_anno/categories_dict.json` into it and symlink `cache/vg`.

### Dataset config (`P/lavis/configs/datasets/vg/defaults_rel_det_eval.yaml`, builder `vg_rel_detection_eval`)
- `val`: `vg/vg_motif_anno/VG-SGG-val.h5` + `vg/vg_motif_anno/image_data.json`
- `test`: `vg/vg_motif_anno/VG-SGG-test.h5` + `vg/vg_motif_anno/image_data.json`
- images: `vg/VG_100k_images`. This is a flat folder with `<image_id>.jpg`, the same layout as the local `/home/kanaoka/usr/dataset/vg/images`.
- Relative paths are prefixed with `cache_root = P/cache` (`P/lavis/__init__.py`, `P/lavis/configs/default.yaml`). Absolute paths are used as-is (`base_dataset_builder.py` L124, L203, L211).
- `categories_dict.json` is read **from the same directory as the h5** (`vg_rel_detection.py` L51-55).
- The PGSG code never opens `VG-SGG-dicts-with-attri.json`. The class names come from `categories_dict.json`, which ships in the repo.

### Loader (`P/lavis/datasets/datasets/vg_rel_detection.py`)
- **The split is inferred from substrings of the h5 path** (L39-45): `'train'`, then `'val'`, then `'test'`, and the last match wins. A `"zs_pred"` substring turns on open-vocabulary filtering (L47-49). **Pitfall:** a directory name containing "val" (such as `eval/` or `interval/`), "test" or "zs_pred" silently changes the split. The three `VG-SGG-{train,val,test}.h5` files can all be the same file, or symlinks to one file.
- The h5 schema is the standard `VG-SGG-with-attri.h5` (L246-299):
  - `split` (per image; 0 = train/val, 2 = test)
  - `img_to_first_box`, `img_to_last_box`, `img_to_first_rel`, `img_to_last_rel` (per image; −1 = none)
  - `labels` [Nbox,1], 1-based (the loader subtracts 1)
  - `attributes` [Nbox,10], 1-based; zeros are fine because attributes are unused
  - `boxes_1024` [Nbox,4], **cx,cy,w,h** at a scale where the longer image side = 1024. The loader asserts cx,cy ≥ 0 and w,h > 0, converts to xyxy, then rescales with `bbox / 1024 * max(w,h)` (L179) using the actual size of the image file.
  - `relationships` [Nrel,2], **global** box indices
  - `predicates` [Nrel,1], 1-based
- Filtering: it keeps images with `split==flag`, at least 1 box and at least 1 relation (L247-258). That reproduces the 26,446-image test set.
- **Quirk:** L262-276 sets `num_val_im = 5000`. `val` takes the first 5,000 split-0 images. **Every other split, including `test`, drops its first 5,000 images** (`image_index[num_im_split:]`). Tang's original code does this only for `train`. With an unmodified loader, PGSG's VG "test" is therefore the last 21,446 of the 26,446 images (unverified whether the authors' h5 differed). Any subset trick has to work around this.
- `load_image_filenames` (L351-383) reads `image_data.json` and only uses the `image_id` field (filename = `<image_id>.jpg`). It skips 4 corrupted ids (1592, 1722, 4616, 4617). It **asserts exactly 108,073 entries**, and these must line up row by row with the h5 per-image arrays.

### Evaluator needs
- `run.zeroshot_cfg.zs_triplets` must point to a `torch.load`-able [N,3] tensor of zero-shot triplets (`sgg_oi_eval.py` L155-170 → `sgg_vg_eval.py` L519-528). If it is absent or None, `torch.load(None)` crashes. Neither the repo nor the checkpoint repo has a VG copy (`PGSG-CVPR2024/psg_datasplit/zeroshot_triplet.pytorch` is for PSG). You can generate it (test triplets not seen in train, 0-based (sub_cls, obj_cls, pred)) or pass a dummy; it only affects the zero-shot recall line.

## 2. Converting the local EGTR VG-150 to PGSG format

What I checked against the local files:

| Item | EGTR local | PGSG | Match? |
|---|---|---|---|
| Object classes | `test.json` categories id 1..150, alphabetical | `categories_dict.json["obj"]` (150, alphabetical, 0-based) | **Identical names and order** (checked in code) |
| Predicates | `rel.json["rel_categories"]` = `['__background__', above, …]` (51) | `categories_dict.json["rel"]` (50, no bg) | **`rel_categories[1:] == rel`** (checked). EGTR pred ids are 1-based, which is what the h5 `predicates` field wants |
| Relation format | `rel[split][str(image_id)] = [[sub_idx, obj_idx, pred], …]`, indices into that image's annotation list in `coco.getAnnIds` order (= ascending ann id; verified sorted for all 26,446 test images) | h5 `relationships` global box idx + `predicates` | Direct mapping (add the image's first-box offset) |
| Boxes | COCO `[x,y,w,h]`, int pixels in original image coords (177 of 325,570 test boxes overflow the image edge by 1-2 px; no zero-size boxes) | `boxes_1024` = cx,cy,w,h × 1024/max(W,H) | Simple transform. Store float32 and clamp w,h ≥ 1e-3 |
| Image size | `images[].width/height` (300 sampled images all match the actual files) | taken from the actual image at load time | OK |
| Splits | train 57,723 / val 5,000 / test 26,446 | Tang/BGNN split, test = 26,446 after "has rel" filter | Same split (EGTR's json came from the Tang h5 via RelationFormer). Whether EGTR val equals PGSG's "first 5,000 train rows" depends on row order and is unverified, but it does not matter for test |
| Test stats | 26,446 imgs, 325,570 boxes, 152,226 rels, every image has ≥ 1 rel | — | — |

**Pitfalls**
- The 108,073-row alignment between `image_data.json` and the h5 is mandatory. Because we write both files, we can make up the order: real images first, then dummy padding rows with `split=-1` and `img_to_first_box=-1`. Padded rows get filtered out before any image is opened. You can also download the official `image_data.json` (URL still works, 1.78 MB): https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip
- The "drop the first 5,000" quirk applies to test. To evaluate N chosen images through the stock loader, mark 5,000 "sacrificial" images with split 2 (any test images; they are never opened) **before** the N target rows, and set every other row to split −1.
- Do not use an `eval`-style directory name (see the split-substring pitfall).
- Do not include `zs_pred` in any path.

### Conversion script sketch (`scripts/pgsg_make_vg_h5.py`, in our repo, egtr env is enough: needs h5py → or the new PGSG env)
```python
# Inputs: /home/kanaoka/usr/dataset/vg/{train,val,test}.json, rel.json
# Output dir (no 'val'/'test'/'train'/'zs_pred' substrings in the dir path!):
#   data/pgsg_vg/vg_motif_anno/{VG-SGG-test.h5, VG-SGG-val.h5 (symlink), image_data.json,
#                               categories_dict.json (copy), zeroshot_triplet_lavis.pytorch}
N_ROWS = 108_073; BOX_SCALE = 1024
rows = []  # list of (image_id, W, H, split_flag, anns_sorted_by_id, rels)
# 1) choose images:  full mode -> train+val as split 0, test as split 2 (keeps PGSG quirk semantics)
#                    subset mode -> 5000 sacrificial test imgs (split 2) + N target test imgs (split 2), rest split -1
# 2) per image: anns = sorted(anns_of(img), key=id); first=len(all_boxes)
#      for a in anns: x,y,w,h=a["bbox"]; s=BOX_SCALE/max(W,H)
#          boxes.append([(x+w/2)*s, (y+h/2)*s, max(w*s,1e-3), max(h*s,1e-3)]); labels.append([a["category_id"]])
#      for sub,obj,pred in rel[split][str(id)]: relationships.append([first+sub, first+obj]); predicates.append([pred])
#      img_to_first_box/last_box/first_rel/last_rel (or -1)
# 3) pad to N_ROWS with dummy ids (e.g. 10**9+k), split=-1, first/last=-1
# 4) h5: split(int), img_to_*(int), labels(int,[N,1]), attributes(zeros int,[N,10]),
#        boxes_1024(float32,[N,4]), relationships(int,[R,2]), predicates(int,[R,1])
# 5) image_data.json = [{"image_id": id, "width": W, "height": H} ...] in the SAME row order
# 6) zeroshot triplets: set((c_s-1, c_o-1, p-1) in test) - set(... in train) -> torch.tensor [M,3]
# 7) symlink VG-SGG-val.h5 -> VG-SGG-test.h5 (val builder entry must exist; val split will just be empty/unused)
```

## 3. Checkpoints

- **Already present locally** in `P/PGSG-CVPR2024/`, a git-lfs clone of https://huggingface.co/rj979797/PGSG-CVPR2024 (real files of about 967 MB each, not LFS pointers):
  - `vg_sgg.pth`: closed-vocabulary VG; README reports R50/100 17.7/20.7, mR 9.0/11.5. epoch 40.
  - `vg_sgg_close_clser.pth`: "VG-c", has extra `close_classifier.*` weights, needs `model.close_clser: True`. README reports R 20.3/23.6. epoch 70.
  - `vg_ov_sgg.pth`: open-vocabulary VG.
  - `psg_*.pth`: not relevant here.
- All of them load with `torch.load(..., map_location='cpu')` under torch 2.7, including the default `weights_only=True`. Keys are `model, config, scaler, epoch`. There are 666 tensors; ViT-B/16 `pos_embed` (1,577,768) means a 384 px input; word embeddings are (30534,768).
- **BLIP pretrained weights (`model_base_capfilt_large.pth`, 2.1 GB) are only for training** (`model.pretrained`). Evaluation loads `model.finetuned` because `load_finetuned: True` in `P/lavis/configs/models/blip_vrd_base_vg.yaml`. Code path: `base_model.py` L82-101; `blip_rel_det_pgsg.py` L1703-1714 loads once, resizes the vocab and reloads, and swallows the first error.
- The tokenizer is `StageBertTokenizer.from_pretrained("bert-base-uncased")` (`blip_rel_det.py` L315-316), downloaded from the HF hub on first run (reachable). It is not cached locally yet; `~/.cache/huggingface/hub` only has deformable-detr and resnet50. The hardcoded `/public/home/lirj2/...` tokenizer paths in `blip.py:24` and `blip_det.py:88` are **not** on the PGSG code path, because `BlipRelDetection.init_tokenizer` overrides them.
- **Prompt mismatch (open question).** The config embedded in `vg_sgg.pth` was trained with arch `blip_rel_detection_xdecoder` and prompt `"A visual scene of "`. `P/lavis/projects/blip/train/vrd_vg_ft_pgsg.yaml:26` uses `"A visual scene of: "`. The eval path falls back to the model default `"The visual scene of: "` (`blip_vrd_base_vg.yaml`). It is unclear which one reproduces the paper. A/B-test `model.prompt` on the first run.

## 4. Environment

### What PGSG pins
- `P/README.md`: python 3.8, torch 2.0.0 (cu118) or 1.10.2.
- `P/requirements_pix2sgg.txt`: torch==2.0.1, torchvision==0.15.2, transformers pinned to git commit `ba70545…` (**= the v4.29.2 release commit**, checked via the GitHub API), tokenizers==0.13.3, huggingface-hub==0.16.4, timm==0.4.12, peft==0.5.0, accelerate==0.22.0, fairscale==0.4.4 (sdist only), h5py, omegaconf, webdataset, decord, iopath, pycocotools, opencv, numpy==1.24.3 (from `requirements-dev.txt`), plus scikit-learn, scikit-image, msgspec, GitPython, ftfy, einops, contexttimer, tabulate, termcolor and others. It also carries many unrelated Jupyter/streamlit pins and conda `@ file://` lines, so do not install the file verbatim.
- `P/setup.py` installs from `requirements-dev.txt`. `setup.py develop` would write `lavis.egg-info` into the submodule; use `PYTHONPATH=Pix2Grp_CVPR2024` instead.

### EGTR env (`egtr/.venv`)
- Python 3.11.15, torch 2.7.1+cu128, torchvision 0.22.1, **transformers 4.41.2**, tokenizers 0.19.1, **timm 1.0.27**, numpy 1.26.4, pycocotools 2.0.11.
- Missing for PGSG: omegaconf, peft, fairscale, webdataset, decord, iopath, h5py, contexttimer, ftfy, einops, cv2, sentencepiece, plus others.

### Recommendation: separate uv project, do not share the EGTR env
- EGTR needs transformers 4.41 and timm 1.x. PGSG's `med.py` and BLIP generation code were written against transformers 4.29 and timm 0.4.12. Sharing one env means changing EGTR's dependencies, which conflicts with "egtr is import-only".
- Put the env outside the submodule, for example `envs/pgsg/pyproject.toml`, and run with `uv run --project envs/pgsg` and `PYTHONPATH=Pix2Grp_CVPR2024`.
  - python 3.10 or 3.11. Wheels exist for tokenizers 0.13.3, h5py 3.9 and pycocotools 2.0.7 on cp310/cp311; decord 0.6.0 is `py3-none`.
  - **torch==2.7.1, torchvision==0.22.1 from the cu128 index.** This is mandatory for sm_120; torch 2.0.1 has no Blackwell kernels.
  - transformers==4.29.2, tokenizers==0.13.3, huggingface-hub==0.16.4 (or the newest hub version that 4.29.2 accepts), timm==0.4.12, peft==0.5.0 (needed for `prepare_model_for_int8_training`, imported at `blip_rel_det_pgsg.py` L22-28; later peft versions removed it), accelerate==0.22.0, fairscale==0.4.13 (or 0.4.4 with `--no-build-isolation`), numpy<2, plus omegaconf, h5py, webdataset, decord, iopath, opencv-python-headless, pycocotools, scikit-learn, scikit-image, pandas, matplotlib, msgspec, GitPython, ftfy, einops, contexttimer, tabulate, termcolor, sentencepiece, regex, portalocker, pycocoevalcap.
- **Risk (not yet tested):** transformers 4.29.2 and timm 0.4.12 running on torch 2.7. I expect them to work because PGSG only uses `BertConfig`/`BertTokenizer`, `transformers.modeling_utils` helpers and `GenerationMixin.generate`. If not, fall back to timm 0.6.13 and transformers 4.31-4.33.
- `lavis/__init__.py` imports *every* LAVIS model, dataset and task (BLIP-2/Vicuna, CLIP, PNP-VQA, …), so the whole import chain has to resolve even though only PGSG is used. `cvpods`, `apex`, `horovod` and `tensorflow` are not on the import path (they are only in unused files).

### GPU
- `lspci` shows NVIDIA `10de:2c02`, most likely a GeForce RTX 5080 (Blackwell, 16 GB; model name and VRAM not confirmed because NVML is down).
- `nvidia-smi` currently fails with **"Driver/library version mismatch"**: the kernel module is 580.159.03 (`/proc/driver/nvidia/version`), the userspace NVML is 580.173.02, and the host has been up 48 days, consistent with an unattended driver upgrade. `torch.cuda.is_available()` is False in the EGTR env as well ("Error 804"). **A reboot is required before any GPU work.**
- The official eval uses `batch_size_eval: 44` on 4 GPUs. On a single 16 GB card, expect a batch size of about 4-8, to be tuned.

## 5. Inference / evaluation entrypoint

- Command (`P/README.md` "Standard SGG → Evaluation"; `P/run_scripts/blip/eval/eval_vg_pgsg.sh` points to an older `rel_det_vg_eval.yaml`):
  ```
  python -m torch.distributed.run --nproc_per_node=4 evaluate.py \
      --cfg-path lavis/projects/blip/eval/rel_det_vg_pgsg_eval.yaml --job-name VG-pgsg_stdsgg-eval
  ```
  - For one GPU use `--nproc_per_node=1`. The torchrun launcher is still required, because `RelationDetectionTask.after_evaluation` calls `torch.distributed.barrier()` unconditionally (`P/lavis/tasks/relation_detection.py` L93), and `init_distributed_mode` only sets up the process group when `RANK`/`WORLD_SIZE` are set (`dist_utils.py` L57-89).
  - `evaluate.py` runs `Git(".")`, so the working directory must be inside a git repo. `commit_code()` exists but is never called.
- Config variants in `P/lavis/projects/blip/eval/`:
  - `rel_det_vg_pgsg_eval.yaml`: standard SGDet
  - `rel_det_vg_pgsg_eval_cls_clser.yaml`: VG-c, `close_clser: True`
  - `rel_det_vg_pgsg_eval_ov.yaml`, `_ov_sgcls.yaml`, `_ov_predcls.yaml`
- Values to override, using a copy of the yaml outside the submodule or `--options k=v`; all of them are hardcoded `/public/home/lirj2/...` paths:
  - `model.finetuned` → `P/PGSG-CVPR2024/vg_sgg.pth`
  - `model.cate_dict_url` and `run.cate_dict_url` → `categories_dict.json`
  - `run.zeroshot_cfg.zs_triplets` → the generated triplets file
  - `datasets.vg_rel_detection_eval.build_info.annotations.{val,test}.{url,storage}` → absolute h5 and `image_data.json` paths. Set `url == storage` so `_download_data` just logs "Using existing file" (`base_dataset_builder.py` L132-136); a relative non-existent url would trigger a download attempt.
  - `...build_info.images.storage` → `/home/kanaoka/usr/dataset/vg/images`
  - `run.output_dir` and `model.dump_dir` → outside the submodule
  - `run.batch_size_eval`, `run.num_workers`, possibly `model.prompt`
- Key run settings:
  - `max_len 1024`, `seg_len 32`. Generation is `1024 // 28 = 36` chained segment generations per batch (`blip_rel_det_pgsg.py` L563-610), which is slow.
  - Input is resized to 384×384 without keeping the aspect ratio (`blip_processors.py` L296-314).
  - `generation_mode: sampling` (nucleus/top-k 60) means outputs are **stochastic**. Use `generation_mode: search` (greedy) or a fixed per-image seed for resolution sweeps.
- Output:
  - `classic_vg_sgg_evaluation` reports SGDet R@20/50/100, mR@K, no-graph-constraint R/mR, zero-shot R, pair accuracy, stage-wise recall, plus COCO-style detection AP and OpenImages-style metrics (`sgg_oi_eval.py` L123-186). Logged to `output_dir`.
  - **Per-image predictions are not saved by default.** `save_output` exists but is not called. `dump_pred: True` pickles raw decoder buffers per batch (`blip_rel_det_pgsg.py` L682-750).
  - Predictions are `BoxList`s in original-image xyxy with fields `pred_labels`, `pred_scores`, `rel_pair_idxs`, `pred_rel_label`, `pred_rel_score`, `pred_rel_dist`, `pred_rel_trp_score` (top 512 triplets; `blip_rel_det_pgsg.py` L1575-1600).

## 6. Minimal plan: PGSG on 10-100 local VG test images

1. **(Blocker) Fix the GPU.** Reboot, or `sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia`. Check with `nvidia-smi`. *About 0.2 h, needs the user / sudo.*
2. **Env.** Create the `envs/pgsg` uv project (torch 2.7.1+cu128, transformers 4.29.2, timm 0.4.12, peft 0.5.0, …, see §4). Smoke test:
   ```
   PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Pix2Grp_CVPR2024 uv run --project envs/pgsg python -c "import lavis; from lavis.common.registry import registry; print(registry.get_model_class('blip_rel_detection_pgsg'))"
   ```
   Then build the model from config and load `vg_sgg.pth` on the GPU, and check that "Missing keys" is empty. *1-3 h, depending on fairscale/timm/transformers friction.*
3. **Convert** with `scripts/pgsg_make_vg_h5.py` (§2) into `data/pgsg_vg/vg_motif_anno/` (copy `categories_dict.json`; generate `zeroshot_triplet_lavis.pytorch`). Validate by loading `VGVRDEvalDataset` directly and checking that `len == N` and that a few GT boxes overlay correctly on the image. *1-2 h.*
4. **Official-path eval on N images.** Copy `rel_det_vg_pgsg_eval.yaml` to `configs/pgsg/vg_pgsg_local.yaml` with the overrides from §5, `batch_size_eval` 4-8 and `test_splits: ["test"]`. Then run:
   ```
   cd <repo root>
   PYTHONPATH=Pix2Grp_CVPR2024 uv run --project envs/pgsg torchrun --nproc_per_node=1 Pix2Grp_CVPR2024/evaluate.py --cfg-path configs/pgsg/vg_pgsg_local.yaml --job-name local100
   ```
   Check R@50/100 against the README (about 17.7/20.7 on the full set; a 100-image subset will be noisy). A/B the prompt string. *1-2 h.*
5. **(For M1 integration) Our own inference wrapper**, `scripts/pgsg_infer.py`, instead of the LAVIS runner:
   - import `lavis`, build the model through the registry from the merged model config, load the eval processor `blip_det_image_eval(384)`.
   - For each (image, resolution level): build `samples={'image', 'targets':[{orig_size, boxes(empty), det_labels, rel_tripets, image_id, ...}], 'image_id'}`, call `model.generate(..., use_nucleus_sampling=False, num_beams=1, max_length=1024, min_length=768, repetition_penalty=1.5)`, and convert the `BoxList` into the same triplet JSON our EGTR pipeline writes.
   - This avoids the h5/108,073/drop-5,000 hacks, works on degraded images, and saves predictions, which the stock runner does not. *3-5 h.*

**Effort:** about 4-8 h to a first SGDet number on 100 images (steps 1-4). Add about 3-5 h for the M1-ready wrapper (step 5). Runtime per image is unmeasured; my guess is roughly 1-5 s/image at max_len 1024 on one RTX 5080. 100 images × 20 resolution levels could take hours, so consider lowering `max_len` (fewer chained segments).

**Blockers / risks, ranked**
1. Driver/library mismatch: GPU unusable until reboot (affects everything).
2. sm_120 GPU forces torch ≥ 2.7, which has not been tested with transformers 4.29.2 / timm 0.4.12 / peft 0.5.0.
3. The annotation bundle link is dead (403), so local conversion is required. The data itself matches; this is low risk.
4. Loader quirks: split chosen by path substring, hard 108,073 assert, test drops its first 5,000 images.
5. Prompt string ambiguity and stochastic decoding affect reproducibility.
6. No per-image prediction dump in the stock runner. The wrapper (step 5) is needed for M1 anyway.
