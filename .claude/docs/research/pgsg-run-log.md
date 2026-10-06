# PGSG (Pix2Grp, CVPR 2024) on local VG-150: run log

Date: 2026-10-07. Prior investigation: [pix2grp-vg-compat.md](pix2grp-vg-compat.md).
Hardware: RTX 5080 16 GB (sm_120), driver 580.173.02 (the driver/library mismatch reported earlier has been resolved).

## 1. Status

| Step | State |
|---|---|
| Env (`envs/pgsg`, torch 2.7.1+cu128) | OK. `import lavis` works, CUDA OK, no source patches needed |
| Model build + `vg_sgg.pth` load on GPU | OK (`Missing keys []`) |
| VG -> PGSG conversion (`scripts/23_pgsg_convert_vg.py`) | OK. Validated: ranges, loader-equivalent row selection, triplet and box round trip |
| Official SGDet eval via `evaluate.py` + torchrun | OK on 20 / 100 / 1000 test images |
| Patches inside `Pix2Grp_CVPR2024/` | **None.** `git -C Pix2Grp_CVPR2024 status` shows only the pre-existing untracked `PGSG-CVPR2024/` checkpoint dir |

## 2. Environment (`envs/pgsg/pyproject.toml` + `uv.lock`)

Python 3.11.15. Key pins (all resolved and installed without build issues):

| Package | Ours | PGSG pin | Note |
|---|---|---|---|
| torch / torchvision | **2.7.1+cu128 / 0.22.1+cu128** | 2.0.1 / 0.15.2 | Required for Blackwell sm_120 (cu128 index, `explicit = true`) |
| transformers | 4.29.2 (PyPI) | git `ba70545` (= v4.29.2) | same code |
| tokenizers / huggingface-hub | 0.13.3 / 0.16.4 | same | |
| timm / peft / accelerate | 0.4.12 / 0.5.0 / 0.22.0 | same | |
| fairscale | **0.4.13** | 0.4.4 | 0.4.4 is sdist-only |
| numpy | **1.26.4** | 1.24.3 | numpy<2 required |
| h5py / pycocotools / scipy | 3.11.0 / 2.0.8 / 1.13.1 | 3.9.0 / 2.0.7 / 1.10.1 | cp311 wheels |
| opencv | opencv-python-headless 4.8.1.78 | opencv-python 4.8.0.76 | headless |
| omegaconf / iopath / webdataset / decord | 2.3.0 / 0.1.10 / 0.48 / 0.6.0 | same | |

Other packages in the env: pandas, einops, ftfy, contexttimer, GitPython, tabulate, termcolor, sentencepiece, regex, portalocker, scikit-learn, scikit-image, matplotlib, msgspec, easydict, pycocoevalcap, safetensors, pyyaml, tqdm. The Jupyter/streamlit/conda lines from `requirements_pix2sgg.txt` were not installed. The torch 2.7, transformers 4.29.2 and timm 0.4.12 combination works for this eval path (sampling generation, the BERT/MED decoder and the ViT encoder).

Cosmetic: every `uv run --project envs/pgsg` prints "Uninstalled/Installed 1 package". uv re-checks the `decord` wheel, which carries a cp36 tag. It costs about 1 ms and does no harm.

## 3. Data conversion

`uv run --project envs/pgsg python scripts/23_pgsg_convert_vg.py --num-test-images N` (N=0 means all 26,446). The output goes to `data/processed/pgsg_vg/` (gitignored):
`VG-SGG-test.h5`, `VG-SGG-val.h5` (a symlink), `image_data.json` (108,073 rows), `categories_dict.json` (a copy of PGSG's VG dict), `zeroshot_triplet.pytorch` (5,426 test triplets of the form 0-based (sub, obj, pred) that are unseen in train), and `selected_test_ids.json`.

The script uses the first N `test.json` images. It handles the loader quirks as follows:
- **Drop-first-5000:** it prepends 5,000 "sacrificial" split-2 rows that share one dummy box and one dummy relation. The loader drops them before it opens any image.
- **108,073-row assert:** the remaining rows are padded with `split=-1` dummy rows.
- **Split from the path substring:** the out-dir is checked for `train`/`val`/`test`/`zs_pred`.
- **val split:** the val builder needs some split-0 rows, so the N target images are duplicated as split 0. The result is a tiny val split that is never evaluated.

Built-in validation: it re-runs the loader's row selection and asserts `kept ids == targets`, and it round-trips the triplets and the first box of 5 images against EGTR's `rel.json` / `test.json`.

## 4. Config + run

- `configs/pgsg/rel_det_vg_eval_local.yaml` is a copy of `lavis/projects/blip/eval/rel_det_vg_pgsg_eval.yaml`. All `/public/...` paths are overridden: `model.finetuned`, both `cate_dict_url`s, `zs_triplets`, the dataset `build_info` (url == storage, absolute), the images dir, and `output_dir` / `dump_dir` → `outputs/pgsg/`. Other changes: batch_size_eval 4, num_workers 2. Generation is the official `sampling` with `seed: 42`.
- `configs/pgsg/rel_det_vg_eval_local_ckpt_prompt.yaml` is the same config with `model.prompt: "A visual scene of "`, the prompt stored in the checkpoint's training config.
- Runner: `bash scripts/24_pgsg_eval.sh JOB [k=v ...]`. It sets `PYTHONPATH=Pix2Grp_CVPR2024` and `PYTHONDONTWRITEBYTECODE=1` (so no `__pycache__` lands in the submodule), then runs `uv run --project envs/pgsg torchrun --nproc_per_node=1 Pix2Grp_CVPR2024/evaluate.py --cfg-path ...`. Its output is tee'd to `outputs/pgsg/logs/<ts>-JOB.log`. Use `PGSG_CFG=...` to switch configs. Example:
  ```
  uv run --project envs/pgsg python scripts/23_pgsg_convert_vg.py --num-test-images 100
  bash scripts/24_pgsg_eval.sh vg100 run.batch_size_eval=8
  ```
- LAVIS run dirs (`log.txt`, `result/`) are written to `outputs/pgsg/runs/<job_id>/`, which is gitignored.

## 5. Results (SGDet, graph-constrained, first N test images)

| Run | N | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | zR@100 | s/img | GPU mem |
|---|---|---|---|---|---|---|---|---|---|---|
| official (sampling, max_len 1024), bs4 | 20 | 8.4 | 13.6 | 17.0 | 0.4 | 1.1 | 2.1 | 0.0 | 3.12 | 7.1 GB |
| official, bs8 | 100 | 17.0 | 22.1 | 24.8 | 4.6 | 8.7 | 10.7 | 10.0 | 2.93 | 12.0 GB |
| ckpt prompt "A visual scene of ", bs8 | 100 | 15.6 | 20.4 | 24.2 | 6.3 | 8.6 | 10.4 | – | 2.92 | 12.0 GB |
| greedy (`generation_mode=search`), bs2 | 100 | 5.7 | 5.9 | 6.1 | 1.6 | 1.6 | 1.6 | – | 4.30 | 10.3 GB |
| sampling, max_len=min_len=512, bs8 | 100 | 16.3 | 18.5 | 19.0 | 5.2 | 7.8 | 7.8 | – | 1.49 | 7.0 GB |
| **official, bs8** | **1000** | **12.1** | **16.5** | **18.4** | 4.6 | 6.6 | 8.2 | 7.1 | 3.05 | 12.0 GB |

Reference (README, full 26,446-image test): R@50/100 = 17.7/20.7, mR@50/100 = 9.0/11.5. Our subsets come from the start of `test.json` and are not random samples, so small-N numbers are noisy. At N=1000 we get R@50/100 = 16.5/18.4 (1.2/2.3 pt below the README) and mR@50/100 = 6.6/8.2. Box AP is 7.4 (AP50 15.5). This is close enough to confirm that the conversion and the checkpoint are correct. Possible sources of the remaining gap are subset bias and seed, and (unverified) the fact that the stock loader skips the first 5,000 test images. Our first 1,000 images are exactly images the authors' loader would have dropped. Log: `outputs/pgsg/logs/20261007-005258-vg1000_official_bs8.log`. `data/processed/pgsg_vg/` currently holds the 1000-image conversion.

Observations:
- **Sampling is deterministic given the seed.** Two runs of the 20-image job with seed 42 gave identical metrics. Results depend on batch composition and order, so they hold only for the same N and batch size.
- **Greedy decoding collapses (R@50 5.9).** `generate()` chains `max_len // ((seg_len+24)/2)` = 36 independent segment generations, all from the same prompt and image. Under greedy decoding every segment is identical, so there are almost no distinct triplets. The model needs stochastic sampling. Greedy also runs out of memory at bs4/bs8 because each segment reaches its maximum length. **For M1, make runs deterministic by re-seeding before each image, not by using greedy decoding.**
- Per-image cost is about 3 s at max_len 1024 and about 1.5 s at max_len 512. bs8 barely improves throughput over bs4, because decoding is sequential and token-bound.
- **Prompt:** the eval default "The visual scene of: " and the training prompt "A visual scene of " fall within noise of each other. We keep the default, which is the official eval setting.
- **Upstream evaluator bug (not patched):** the "No Graph Constraint Recall" (ngR) values are meaninglessly low (ngR@50 1.6 < R@50 22.1). In `lavis/tasks/evaluation/vg_sgg_eval_tools.py` L316-319, when `rel_dist` has no background column, the code stacks the whole 2-column `nogc_score_inds` instead of `nogc_score_inds[:, 1]`, which misaligns the predicate column. Ignore ngR / ng-mR; the README reports only R and mR.
- The first-ever run spent about 3 minutes before the first log line (a cold start). Later runs start within seconds.

## 6. Remaining issues
- Full-test numbers have not been run: 26,446 × 3 s ≈ 22 h. At max_len 512 it would be about 11 h.
- The stock runner does not save per-image predictions. `dump_pred` pickles raw decoder buffers; it is not a triplet JSON.
- The VG "test" quirk (dropping the first 5,000 images) is worked around in the data. Whether the authors' own numbers excluded the first 5,000 test images is still unverified.

## 7. Next steps: M1 wrapper (`scripts/25_pgsg_infer.py`, run with the `envs/pgsg` env)
1. Build the model without the runner: `cfg = OmegaConf.load(configs/pgsg/rel_det_vg_eval_local.yaml)`, merge it with `lavis/configs/models/blip_vrd_base_vg.yaml`, then call `registry.get_model_class("blip_rel_detection_pgsg").from_config(cfg.model).cuda().eval()`. Use the eval processor `blip_det_image_eval(image_size=384)`.
2. For each (image, resolution level), load the degraded image from `data/processed/degraded_images/...`. Build `samples = {"image": tensor[B,3,384,384], "targets": [{"orig_size": [h,w], "boxes": empty, "det_labels": empty, "rel_tripets": empty [0,3], "image_id": ...}], "image_id": [...]}`. `_gtpostprocess` and `_postprocess` only read `orig_size`, `boxes`, `det_labels`, `rel_tripets` and `image_id`; verify this when implementing.
3. Call `torch.manual_seed(SEED)` per image (or per image × level), then `model.generate(samples, use_nucleus_sampling=True, num_beams=1, max_length=1024 or 512, min_length=768, repetition_penalty=1.5)`.
4. Convert each returned `BoxList` into the EGTR triplet JSON schema. Fields: `pred_labels` (0-based) and `pred_scores`; `rel_pair_idxs`, `pred_rel_label` (0-based over the 50 predicates; check the offset against `categories_dict["rel"]`), `pred_rel_score` and `pred_rel_trp_score` (top 512, already sorted). Boxes are xyxy in `orig_size` coordinates.
5. Budget for 100 images × 20 levels: about 1.7 h at max_len 1024, or about 0.8 h at 512. Consider max_len 512 for sweeps.
