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
- `/home/kanaoka/usr/dataset/vg/images/` - 108,249 images

### Per-Image GT Objects

From `{split}.json` (COCO format):
```python
# Each annotation has: image_id, category_id (1-indexed), bbox [x,y,w,h] (COCO format)
# The feature_extractor converts COCO bbox to DETR format (cx,cy,w,h normalized)
# After VGDataset.__getitem__: target["class_labels"] are 0-indexed (category_id - 1)
# target["boxes"] are in (cx, cy, w, h) normalized format
```

### Per-Image GT Relations

From `rel.json[split][str(image_id)]`:
```python
# Each entry is a list of [subj_annotation_idx, obj_annotation_idx, predicate_id]
# where predicate_id is 1-indexed (1..50), and subj/obj indices refer to the
# order of annotations for that image_id in the COCO annotations list

rel_list = rel_data[split][str(image_id)]
# Example: [[1, 0, 20], [11, 8, 50]] means:
#   annotation[1] --(predicate 20 = "has")--> annotation[0]
#   annotation[11] --(predicate 50 = "with")--> annotation[8]
```

### VGDataset._get_rel_tensor (data/visual_genome.py)

Converts rel_list to a dense tensor:
```python
# rel_tensor shape: (num_object_queries, num_object_queries, 50)
# indices[-1, :] -= 1  # convert predicate_id from 1-indexed to 0-indexed
# rel[subj_idx, obj_idx, pred_id] = 1.0 for each GT relation
```

### Enumerating GT for a Given Image

```python
import json
from pycocotools.coco import COCO

split = "test"
data_path = "/home/kanaoka/usr/dataset/vg"

# Load COCO annotations
coco = COCO(f"{data_path}/{split}.json")

# Load relations
with open(f"{data_path}/rel.json") as f:
    rel_data = json.load(f)
rels = rel_data[split]

# Object categories (0-indexed)
cats = coco.cats  # {1: {"name": "airplane"}, ..., 150: {"name": "zebra"}}
id2name = {cid - 1: info["name"] for cid, info in cats.items()}

# Predicate names (0-indexed)
pred_names = rel_data["rel_categories"][1:]  # skip __background__

# For a specific image:
image_id = 2343729
ann_ids = coco.getAnnIds(imgIds=image_id)
anns = coco.loadAnns(ann_ids)

# GT objects
for ann in anns:
    cls_name = id2name[ann["category_id"] - 1]
    bbox = ann["bbox"]  # [x, y, w, h] COCO format
    print(f"Object: {cls_name}, bbox={bbox}")

# GT relations
rel_list = rels[str(image_id)]
for subj_idx, obj_idx, pred_id_1indexed in rel_list:
    subj_name = id2name[anns[subj_idx]["category_id"] - 1]
    obj_name = id2name[anns[obj_idx]["category_id"] - 1]
    pred_name = pred_names[pred_id_1indexed - 1]  # 1-indexed to 0-indexed
    print(f"  {subj_name} --[{pred_name}]--> {obj_name}")
```

## 7. Minimal Standalone Inference Snippet

```python
import json, torch
from glob import glob
from PIL import Image
from model.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor
from model.egtr import DetrForSceneGraphGeneration
from lib.pytorch_misc import argsort_desc
from util.box_ops import rescale_bboxes
import numpy as np

ARTIFACT = "checkpoint/egtr__pretrained_detr__SenseTime__deformable-detr__batch__32__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002__visual_genome__finetune/version_0"
ARCH = "SenseTime/deformable-detr"

# Load model
config = DeformableDetrConfig.from_pretrained(ARTIFACT)
model = DetrForSceneGraphGeneration.from_pretrained(ARCH, config=config, ignore_mismatched_sizes=True)
sd = torch.load(sorted(glob(f"{ARTIFACT}/checkpoints/epoch=*.ckpt"), key=lambda x: int(x.split("epoch=")[1].split("-")[0]))[-1], map_location="cpu")["state_dict"]
for k in list(sd.keys()): sd[k[6:]] = sd.pop(k)
model.load_state_dict(sd); model.cuda(); model.eval()

# Load vocabularies
vg = json.load(open("/home/kanaoka/usr/dataset/vg/test.json"))
obj_names = {c["id"]-1: c["name"] for c in vg["categories"]}
pred_names = json.load(open("/home/kanaoka/usr/dataset/vg/rel.json"))["rel_categories"][1:]

# Preprocess image
fe = DeformableDetrFeatureExtractor.from_pretrained(ARCH, size=800, max_size=1333)
img = Image.open("/home/kanaoka/usr/dataset/vg/images/2343729.jpg").convert("RGB")
enc = fe(images=img, return_tensors="pt")

# Inference
with torch.no_grad():
    out = model(pixel_values=enc["pixel_values"].cuda(), pixel_mask=enc["pixel_mask"].cuda(),
                output_attentions=False, output_attention_states=True, output_hidden_states=True)

# Objects
probs = out.logits[0].softmax(-1); scores, classes = probs[:, :150].max(-1)
for i in torch.where(scores > 0.2)[0]:
    box = rescale_bboxes(out.pred_boxes[0][i:i+1].cpu(), img.size).squeeze().tolist()
    print(f"  {obj_names[classes[i].item()]:15s} score={scores[i]:.3f}  box={[round(x,1) for x in box]}")

# Triplets (single pred per pair, with constraint)
pr = torch.clamp(out.pred_rel[0], 0, 1) * torch.clamp(out.pred_connectivity[0], 0, 1)
so = torch.outer(scores, scores); so[torch.arange(200), torch.arange(200)] = 0
top = argsort_desc((pr.max(-1)[0] * so).cpu().numpy())[:20]
for s, o in top:
    p = pr[s, o].argmax().item()
    print(f"  {obj_names[classes[s].item()]} --[{pred_names[p]}]--> {obj_names[classes[o].item()]}  "
          f"score={pr[s,o,p]:.3f} subj={scores[s]:.2f} obj={scores[o]:.2f}")
```

## 8. Verified Output Shapes (from actual run)

```
logits:             torch.Size([1, 200, 150])
pred_boxes:         torch.Size([1, 200, 4])
pred_rel:           torch.Size([1, 200, 200, 50])
pred_connectivity:  torch.Size([1, 200, 200, 1])
last_hidden_state:  torch.Size([1, 200, 256])
```

## 9. Key Implementation Notes

1. **pred_rel is post-sigmoid**: The forward() applies sigmoid before returning; values are [0,1]. Do NOT apply sigmoid again.
2. **pred_connectivity gating**: Always multiply pred_rel by pred_connectivity before ranking triplets.
3. **Class indexing**: Model outputs 0-indexed classes (0..149). JSON categories use 1-indexed (1..150). Subtract 1 when reading GT annotations; add 1 when mapping model output to JSON category_id.
4. **Predicate indexing**: Model outputs 0-indexed predicates (0..49). rel.json uses 1-indexed (1..50). The dataset code subtracts 1 in `_get_rel_tensor`.
5. **Box format**: pred_boxes are (cx, cy, w, h) normalized [0,1]. Use `rescale_bboxes(boxes, (W, H))` to get (x1, y1, x2, y2) absolute.
6. **GPU recommended**: CUDA custom kernels are compiled on first import. CPU fallback works but is slow.
7. **Feature extractor**: Uses DetrFeatureExtractor (deprecated warning but works). Resizes shortest edge to 800, max 1333. ImageNet normalization applied internally.
8. **Checkpoint format**: PyTorch Lightning checkpoint with keys prefixed by "model.". Must strip prefix before `load_state_dict`.
