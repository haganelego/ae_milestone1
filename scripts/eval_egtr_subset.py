"""
Subset evaluation of EGTR on Visual Genome test set.
Copied from egtr/evaluate_egtr.py with a --max_images flag added.
Does NOT modify any tracked EGTR files.
"""

import argparse
import json
import os
import sys
from glob import glob

# Add egtr to sys.path so we can import its modules
EGTR_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "egtr"
)
sys.path.insert(0, EGTR_DIR)
os.chdir(EGTR_DIR)  # so relative paths (dataset/visual_genome) work

import torch
from data.visual_genome import VGDataset
from lib.evaluation.sg_eval import (
    BasicSceneGraphEvaluator,
    calculate_mR_from_evaluator_list,
)
from model.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor
from model.egtr import DetrForSceneGraphGeneration
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from train_egtr import collate_fn, evaluate_batch


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    num_labels,
    rel_categories,
    multiple_sgg_evaluator=None,
    single_sgg_evaluator=None,
):
    metric_dict = {}
    model.eval()

    multiple_sgg_evaluator_list = []
    single_sgg_evaluator_list = []
    if multiple_sgg_evaluator is not None:
        for index, name in enumerate(rel_categories):
            multiple_sgg_evaluator_list.append(
                (index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True))
            )
    if single_sgg_evaluator is not None:
        for index, name in enumerate(rel_categories):
            single_sgg_evaluator_list.append(
                (index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=False))
            )

    for batch in tqdm(dataloader):
        outputs = model(
            pixel_values=batch["pixel_values"].cuda(),
            pixel_mask=batch["pixel_mask"].cuda(),
            output_attentions=False,
            output_attention_states=True,
            output_hidden_states=True,
        )
        targets = batch["labels"]
        evaluate_batch(
            outputs,
            targets,
            multiple_sgg_evaluator,
            multiple_sgg_evaluator_list,
            single_sgg_evaluator,
            single_sgg_evaluator_list,
            None,  # oi_evaluator
            num_labels,
        )

    if multiple_sgg_evaluator is not None:
        recall = multiple_sgg_evaluator["sgdet"].print_stats()
        mean_recall = calculate_mR_from_evaluator_list(
            multiple_sgg_evaluator_list, "sgdet", multiple_preds=True
        )
        metric_dict.update(recall)
        metric_dict.update(mean_recall)

    if single_sgg_evaluator is not None:
        recall = single_sgg_evaluator["sgdet"].print_stats()
        mean_recall = calculate_mR_from_evaluator_list(
            single_sgg_evaluator_list, "sgdet", multiple_preds=False
        )
        recall = {f"(single){key}": value for key, value in recall.items()}
        mean_recall = {f"(single){key}": value for key, value in mean_recall.items()}
        metric_dict.update(recall)
        metric_dict.update(mean_recall)

    return metric_dict


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/visual_genome")
    parser.add_argument("--artifact_path", type=str, required=True)
    parser.add_argument("--architecture", type=str, default="SenseTime/deformable-detr")
    parser.add_argument("--num_queries", type=int, default=200)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_single_preds", type=str2bool, default=True)
    parser.add_argument("--eval_multiple_preds", type=str2bool, default=True)
    parser.add_argument("--logit_adjustment", type=str2bool, default=False)
    parser.add_argument("--logit_adj_tau", type=float, default=0.3)
    parser.add_argument("--min_size", type=int, default=800)
    parser.add_argument("--max_size", type=int, default=1333)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="If >0, only evaluate on first N images (for quick smoke test)",
    )
    args = parser.parse_args()

    # Feature extractor
    feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
        args.architecture, size=args.min_size, max_size=args.max_size
    )

    # Dataset
    test_dataset = VGDataset(
        data_folder=args.data_path,
        feature_extractor=feature_extractor,
        split=args.split,
        num_object_queries=args.num_queries,
    )
    id2label = {k - 1: v["name"] for k, v in test_dataset.coco.cats.items()}
    rel_categories = test_dataset.rel_categories

    # Subset if requested
    if args.max_images > 0:
        indices = list(range(min(args.max_images, len(test_dataset))))
        test_dataset_sub = Subset(test_dataset, indices)
        print(f"[SUBSET] Using {len(indices)} / {len(test_dataset)} images")
    else:
        test_dataset_sub = test_dataset
        print(f"[FULL] Using all {len(test_dataset)} images")

    # Dataloader
    test_dataloader = DataLoader(
        test_dataset_sub,
        collate_fn=lambda x: collate_fn(x, feature_extractor),
        batch_size=args.eval_batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    # Evaluator
    multiple_sgg_evaluator = None
    single_sgg_evaluator = None
    if args.eval_multiple_preds:
        multiple_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)
    if args.eval_single_preds:
        single_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)

    # Model
    config = DeformableDetrConfig.from_pretrained(args.artifact_path)
    config.logit_adjustment = args.logit_adjustment
    config.logit_adj_tau = args.logit_adj_tau

    model = DetrForSceneGraphGeneration.from_pretrained(
        args.architecture, config=config, ignore_mismatched_sizes=True
    )
    ckpt_path = sorted(
        glob(f"{args.artifact_path}/checkpoints/epoch=*.ckpt"),
        key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
    )[-1]
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state_dict.keys()):
        state_dict[k[6:]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Evaluate
    metric = evaluate(
        model,
        test_dataloader,
        max(id2label.keys()) + 1,
        rel_categories,
        multiple_sgg_evaluator,
        single_sgg_evaluator,
    )

    # Print summary
    print("\n" + "=" * 60)
    print(
        f"EVALUATION SUMMARY ({len(test_dataloader)} batches, "
        f"max_images={'all' if args.max_images <= 0 else args.max_images})"
    )
    print("=" * 60)
    for k, v in sorted(metric.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
    print("=" * 60)

    # Save
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".claude",
        "docs",
        "research",
        f"egtr_eval_{'subset' if args.max_images > 0 else 'full'}_{args.max_images}.json",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metric["eval_arg"] = args.__dict__
    metric["num_images"] = (
        len(test_dataloader) * args.eval_batch_size
        if args.max_images <= 0
        else min(args.max_images, len(test_dataset))
    )
    with open(output_path, "w") as f:
        json.dump(metric, f, indent=2)
    print(f"Saved to: {output_path}")
