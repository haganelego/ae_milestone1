# VG QA vs GQA for Query-Conditioned Semantic-Saturation Stopping

> Investigation report for Milestone 1 VISAPP paper: can we use Visual Genome QA
> or GQA questions to drive a **query-conditioned optimal-stop** study, where the
> stop level depends on which question the user wants answered?

---

## 1. Visual Genome QA

### 1.1 Structure

| Field | Description |
|-------|-------------|
| `image_id` | Integer, maps to VG image (same ID space as VG objects / scene graphs) |
| `qas[]` | Array of QA pairs per image |
| `qas[].qa_id` | Unique QA identifier |
| `qas[].question` | Free-form natural-language question |
| `qas[].answer` | Free-form natural-language answer (open-ended, variable length) |
| `qas[].question_synsets[]` | WordNet synsets for question entities (`synset_name`, `entity_name`, `entity_idx_start/end`) |
| `qas[].answer_synsets[]` | WordNet synsets for answer entities |

- **Size**: ~1.7 M QA pairs over ~108 K images.
- **Answer format**: Free-form text (not closed-set). Answers range from single words to short phrases; highly variable.
- **Grounding**: VG QA questions are image-level -- there is **no explicit link** from a question to specific scene-graph nodes or bounding boxes. The `question_synsets` give WordNet entries for mentioned entities but do **not** provide object IDs. To ground a VG-QA question to scene-graph nodes, one must perform fuzzy string/synset matching between the question text and object names in the VG scene graph -- a noisy, non-trivial process.
- **Region-QA**: VG also has `region_descriptions` and `region_graphs` tied to bounding boxes, but these are descriptions, not question-answer pairs. The separate Visual7W dataset (a VG subset) adds bounding-box-grounded "which" questions, but it is small (~328 K QA, 47 K images) and uses a pointing/grounding answer format.

### 1.2 Quality issues

- VG annotations are crowdsourced with limited quality control; questions are often trivial ("What is this?"), ambiguous, or answerable without the image.
- Answer normalization is poor: spelling variants, synonyms, different granularity levels.
- Scene graphs themselves have noise: missing objects, incorrect relations, inconsistent attribute labels.
- The ontology is uncontrolled (~33 K unique object names, many near-duplicates).

### 1.3 Verdict

**Not recommended as the primary question source.** The lack of explicit question-to-node grounding is the critical gap. Programmatic extraction of query-relevant scene-graph nodes would require fuzzy matching with many failure modes, making the oracle (O2) computation unreliable. The free-form answer format also makes automatic evaluation harder.

---

## 2. GQA (Hudson & Manning, CVPR 2019)

### 2.1 Structure

| Field | Description |
|-------|-------------|
| `imageId` | String, maps to VG image ID (same ID space) |
| `question` | Natural-language question |
| `answer` | Short canonical answer (single word or short phrase) |
| `fullAnswer` | Extended natural-language answer sentence |
| `semantic[]` | Functional program: array of `{operation, dependencies, argument}` |
| `annotations` | Object-level grounding: `answer: {"0": "objID"}`, `question: {"start:end": "objID"}` |
| `types` | `{detailed, semantic, structural}` -- question taxonomy |
| `entailed` | Set of logically entailed question IDs |
| `equivalent` | Set of semantically equivalent question IDs |

### 2.2 Functional program format

Each question has a **semantic program** -- a chain of atomic operations:

```json
{
  "semantic": [
    {"operation": "select",       "dependencies": [],  "argument": "wall (722332)"},
    {"operation": "filter color",  "dependencies": [0], "argument": "white"},
    {"operation": "relate",        "dependencies": [1], "argument": "pipe,on,s"},
    {"operation": "query",         "dependencies": [2], "argument": "name"}
  ]
}
```

- **Operations**: `select`, `filter <attr>`, `relate`, `verify`, `query`, `choose`, `and`, `or`, `exist`, `compare`, etc.
- **Arguments reference scene-graph object IDs** in the format `"name (objectID)"` (e.g., `"wall (722332)"`).
- **Dependencies** form a DAG (usually a chain) specifying the reasoning flow.

### 2.3 Annotations (direct grounding)

```json
"annotations": {
  "answer": {"0": "722335"},
  "question": {"4:6": "722332"}
}
```

- `question` maps character spans in the question text to scene-graph object IDs.
- `answer` maps character spans in the answer to scene-graph object IDs.
- This provides **direct, programmatic grounding** from question to scene-graph nodes.

### 2.4 Scene graph structure

```json
{
  "imageId": {
    "width": 800, "height": 600,
    "objects": {
      "722332": {
        "name": "wall",
        "x": 0, "y": 0, "w": 756, "h": 537,
        "attributes": ["white", "large"],
        "relations": [
          {"object": "722335", "name": "behind"}
        ]
      }
    }
  }
}
```

- Objects have `name`, bounding box (`x, y, w, h`), `attributes[]`, `relations[]`.
- Relations are directed: `{object: targetObjID, name: predicateName}`.
- **Ontology**: 1,740 objects, 620 attributes, 330 relations (cleaned from VG).

### 2.5 Size and splits

| Split | Questions (balanced) | Questions (all) |
|-------|---------------------|-----------------|
| Train | 943 K | 14.3 M |
| Val   | 132 K | 1.8 M |
| Test  | 12.6 K | -- |
| TestDev | 12.6 K | -- |
| **Total** | **~1.1 M (balanced)** | **22 M (all)** |

- 113 K images total, all from Visual Genome / COCO / Flickr.
- All questions for a given image stay in the same split.

### 2.6 Relation to VG scene graphs used by EGTR

**GQA uses the same Visual Genome images and the same image IDs.** The GQA scene graphs are a **cleaned, normalized subset** of VG scene graphs with a consolidated ontology. EGTR is trained and evaluated on VG scene graphs (the original, noisier version).

**Critical implication**: The object IDs in GQA scene graphs may differ from those in VG's `objects.json` / `relationships.json` because GQA re-annotated and cleaned the graphs. However, they reference the same images. For our pipeline:

- **GT scene graph for oracle (O2)**: We can use either GQA's cleaner scene graphs or VG's original ones. Using GQA's is preferable because the questions are grounded to GQA's object IDs.
- **EGTR predictions**: EGTR outputs detected objects with names and bounding boxes. Matching EGTR output to GQA GT nodes requires spatial (IoU) + semantic (name) matching, which we already do.

### 2.7 Verdict

**Strongly recommended.** GQA is the ideal question source because:

1. **Explicit grounding**: Each question's functional program and annotations contain the exact scene-graph object IDs involved, enabling trivial extraction of the query-relevant node set.
2. **Closed, canonical answers**: Easy automatic evaluation.
3. **Question taxonomy**: Structural types (verify/query/choose/compare/logical) and semantic types (object/attribute/relation/global) enable systematic analysis of which question types benefit most from higher resolution.
4. **Same image space as EGTR/VG**: No domain mismatch.
5. **Balanced splits**: Controlled answer distributions reduce language-prior shortcuts.

---

## 3. Extracting Query-Relevant Node Sets from GQA

### 3.1 From functional programs

For a given GQA question, the **query-relevant scene-graph nodes** are all object IDs appearing in the `semantic` program arguments:

```python
import re

def extract_query_nodes(question_entry: dict) -> set[str]:
    """Extract all scene-graph object IDs referenced by a GQA question."""
    node_ids = set()
    for step in question_entry["semantic"]:
        arg = step["argument"]
        # Arguments can contain object IDs in "name (objID)" format
        match = re.search(r'\((\d+)\)', arg)
        if match:
            node_ids.add(match.group(1))
    # Also from annotations
    for span_map in question_entry.get("annotations", {}).values():
        for obj_id in span_map.values():
            node_ids.add(str(obj_id))
    return node_ids
```

### 3.2 From annotations

The `annotations` field directly maps question/answer text spans to object IDs. This is a complementary (and sometimes more complete) source of grounding.

### 3.3 Expanding to relations

Once we have the set of referenced object IDs, we can look up the GQA scene graph to find all relations between them -- this gives the **query-relevant subgraph**:

```python
def extract_query_subgraph(scene_graph: dict, query_nodes: set[str]) -> dict:
    """Extract the subgraph relevant to a set of query node IDs."""
    subgraph = {"objects": {}, "relations": []}
    for obj_id in query_nodes:
        if obj_id in scene_graph["objects"]:
            obj = scene_graph["objects"][obj_id]
            subgraph["objects"][obj_id] = obj
            for rel in obj.get("relations", []):
                if rel["object"] in query_nodes:
                    subgraph["relations"].append({
                        "subject": obj_id,
                        "predicate": rel["name"],
                        "object": rel["object"]
                    })
    return subgraph
```

### 3.4 Oracle O2 definition

**O2 (query-conditioned oracle stop)** = the earliest resolution level at which the query-relevant subgraph's recall vs the GT subgraph saturates. Concretely:

- At each resolution level r, run EGTR to get predicted objects + relations.
- Match predicted objects to query-relevant GT objects (IoU >= 0.5 + name match).
- Compute node recall = |matched query nodes| / |query nodes|.
- Compute relation recall = |matched query relations| / |query relations|.
- O2(q) = min r such that node_recall(r) >= tau AND relation_recall(r) >= tau (or such that both are stable for k consecutive levels).

---

## 4. Resolution-Graded Question Curation

Neither VG QA nor GQA was designed with resolution in mind. We need a method to **bin questions by resolution difficulty** (i.e., how much resolution they require to be answerable).

### 4.1 Proposed automatable curation criteria

| Criterion | Signal | Proxy metric | Hypothesis |
|-----------|--------|--------------|------------|
| **Object size** | Small objects need more pixels | `sqrt(w * h) / sqrt(img_w * img_h)` for each query-relevant object | Smaller relative area -> higher resolution needed |
| **Number of query objects** | More objects to detect -> harder | `len(query_nodes)` | More nodes -> less likely all are detected at low res |
| **Attribute granularity** | Fine-grained attributes (color, texture, material) need detail | Attribute type from GQA ontology categories | "Material" / "pattern" questions need more resolution than "shape" |
| **Relation type** | Spatial relations between distant/small objects | Relation predicate type + spatial distance between object centers | Relations requiring precise localization need more resolution |
| **Object category** | Some categories are inherently small/fine-grained | Category from GQA ontology hierarchy | "Food items", "text", "utensils" need higher res |

### 4.2 Composite resolution-difficulty score

```python
def resolution_difficulty_score(question_entry: dict, scene_graph: dict) -> float:
    """Estimate how much resolution a question needs. Higher = needs more."""
    query_nodes = extract_query_nodes(question_entry)
    img_area = scene_graph["width"] * scene_graph["height"]
    
    # Factor 1: minimum relative object size
    min_rel_size = 1.0
    for obj_id in query_nodes:
        if obj_id in scene_graph["objects"]:
            obj = scene_graph["objects"][obj_id]
            rel_size = (obj["w"] * obj["h"]) / img_area
            min_rel_size = min(min_rel_size, rel_size)
    size_score = 1.0 - min_rel_size  # smaller -> higher score
    
    # Factor 2: number of query objects
    n_objects = len(query_nodes)
    obj_count_score = min(n_objects / 5.0, 1.0)
    
    # Factor 3: question structural type (verify < query < compare/logical)
    type_weights = {"verify": 0.2, "query": 0.4, "choose": 0.6, "compare": 0.8, "logical": 0.9}
    type_score = type_weights.get(question_entry["types"]["structural"], 0.5)
    
    return 0.5 * size_score + 0.3 * obj_count_score + 0.2 * type_score
```

### 4.3 Binning strategy

- Compute the resolution-difficulty score for all GQA val questions that map to our image set.
- Divide into terciles: **easy** (large objects, verify questions), **medium**, **hard** (small objects, multi-hop reasoning).
- Validate empirically: check that EGTR actually detects the relevant objects at lower resolution for "easy" questions and only at higher resolution for "hard" ones.

### 4.4 Prior work on question difficulty

- **Difficulty-Controllable Visual Question Generation** (Li et al., ACL 2021): Proposes controllable generation of VQA questions at different difficulty levels, using reasoning chain length and answer rarity as proxies.
- **HRVQA** (Lobry et al., 2023): VQA benchmark for high-resolution aerial images, explicitly studies how object size in high-res images affects question difficulty.
- **Variable Resolution Improves VQA Under a Limited Pixel Budget** (Gizdov, Ullman & Harari, ECCV-W 2024): Demonstrates that foveated/variable sampling at 3% of full resolution achieves near-full-res accuracy on GQA, and that different questions benefit from different sampling patterns. Directly relevant precedent.

---

## 5. Data Availability & Format

### 5.1 GQA downloads

| File | URL | Size |
|------|-----|------|
| Scene graphs | `https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip` | 42.7 MB |
| Questions v1.2 | `https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip` | 1.4 GB |
| Images + features | `https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip` | 73.9 GB |

**Files inside questions1.2.zip**:
- `train_balanced_questions.json` (~943 K questions)
- `val_balanced_questions.json` (~132 K)
- `test_balanced_questions.json` (~12.6 K)
- `testdev_balanced_questions.json` (~12.6 K)
- Plus unbalanced variants (14.3 M total)

**Files inside sceneGraphs.zip**:
- `train_sceneGraphs.json`
- `val_sceneGraphs.json`

### 5.2 VG QA downloads

| File | URL |
|------|-----|
| QA pairs | `https://visualgenome.org/static/data/dataset/question_answers.json.zip` |
| Scene graphs | `https://visualgenome.org/static/data/dataset/relationships.json.zip` + `objects.json.zip` |
| Region descriptions | `https://visualgenome.org/static/data/dataset/region_descriptions.json.zip` |

### 5.3 Joining GQA question to VG image and GT nodes

```bash
# On the data machine (/home/kanaoka/usr/dataset/vg):

# 1. Download GQA scene graphs and questions
wget https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip
wget https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
unzip sceneGraphs.zip -d gqa_scenegraphs/
unzip questions1.2.zip -d gqa_questions/

# 2. Inspect a question entry
python3 -c "
import json
with open('gqa_questions/val_balanced_questions.json') as f:
    qs = json.load(f)
    q_id, q = next(iter(qs.items()))
    print(json.dumps({q_id: q}, indent=2))
"

# 3. Check image ID correspondence with VG
python3 -c "
import json
with open('gqa_scenegraphs/val_sceneGraphs.json') as f:
    sg = json.load(f)
    img_id = next(iter(sg.keys()))
    print(f'GQA scene graph image ID: {img_id}')
    print(f'Objects: {list(sg[img_id][\"objects\"].keys())[:5]}')
"

# 4. GQA imageId -> VG image file: images are VG images
# GQA imageId "2375429" -> VG image "2375429.jpg"
# The VG images on disk: /home/kanaoka/usr/dataset/vg/VG_100K/ and VG_100K_2/

# 5. Join: question -> query nodes -> scene graph nodes -> bounding boxes
python3 -c "
import json, re
with open('gqa_questions/val_balanced_questions.json') as f:
    qs = json.load(f)
with open('gqa_scenegraphs/val_sceneGraphs.json') as f:
    sgs = json.load(f)
q_id, q = next(iter(qs.items()))
img_id = q['imageId']
sg = sgs[img_id]
# Extract referenced object IDs from semantic program
for step in q['semantic']:
    m = re.search(r'\((\d+)\)', step['argument'])
    if m:
        obj_id = m.group(1)
        obj = sg['objects'].get(obj_id, {})
        print(f'  Node {obj_id}: {obj.get(\"name\",\"?\")} @ ({obj.get(\"x\")},{obj.get(\"y\")},{obj.get(\"w\")},{obj.get(\"h\")})')
"
```

### 5.4 Filtering GQA questions to our image set

Our pipeline uses a subset of VG images. To select relevant GQA questions:

```python
# our_image_ids: set of VG image IDs we use (from configs/config.yaml dataset)
# Filter GQA questions to those whose imageId is in our set
relevant_qs = {qid: q for qid, q in gqa_questions.items() 
               if q["imageId"] in our_image_ids}
```

---

## 6. Novelty Positioning

### 6.1 Related work survey

| # | Paper | Relevance | How we differ |
|---|-------|-----------|---------------|
| 1 | **Das et al., "Embodied Question Answering" (CVPR 2018)** | Agent navigates a 3D environment to gather views needed to answer a question. Active perception to acquire task-relevant visual info. | We operate on a single image at multiple resolutions (not 3D navigation). Our "action" is choosing a resolution level, not a camera pose. We focus on *when to stop* resolution recovery, not spatial exploration. |
| 2 | **Li et al., "Embodied Semantic Scene Graph Generation" (CoRL 2022)** | Agent explores environment to incrementally build a scene graph; quality improves with more views. | We build SG from a single image across resolution levels. We add a *query-conditioned* stopping criterion (stop once the question is answerable), not global SG completeness. |
| 3 | **Gizdov, Ullman & Harari, "Variable Resolution Improves VQA Under a Limited Pixel Budget" (ECCV-W 2024)** | Shows foveated/variable resolution sampling helps VQA with limited pixels; tested on GQA. | They optimize *spatial* pixel allocation at a fixed budget. We optimize *uniform resolution level* selection with a stopping policy. They do not model question-conditioned stopping or use scene graphs as intermediate representation. |
| 4 | **Rosenfeld et al., "Questioning the Stability of VQA" (arXiv 2511.11206, 2025)** | Studies VLM answer instability under minor image perturbations (shifts, rescaling). Shows stability predicts correctness. | We deliberately degrade resolution (not minor perturbations) and measure answer *saturation* across a structured resolution ladder. We use SGG output, not end-to-end VLM answers. Their stability-as-correctness insight supports our "answer stability" (B) signal design. |
| 5 | **Task-Oriented Semantic Communication** (Zhang et al., IEEE 2023; Xu et al., 2024; multiple) | Transmit only enough image semantics for a downstream task; task-aware bandwidth allocation and stopping. | We share the "stop when task semantics suffice" philosophy but operate at the *recognition/SGG level*, not the communication/channel level. Our resolution levels are an analogue of progressive transmission quality. |
| 6 | **RAPNet: Resolution-Adaptive and Predictive Early Exit** (Ge et al., IEEE 2024) | Jointly adapts input resolution and network depth for efficient image recognition via early exit. | They optimize compute efficiency for classification. We optimize *information acquisition* for a compositional QA task using scene-graph semantics as the stopping signal. |
| 7 | **Next-Best-View for Object Recognition** (various, e.g., Mendoza et al., 2022) | Active vision: keep acquiring views until recognition confidence is sufficient. | Analogous confidence-based stopping, but in 3D viewpoint space. We apply the same principle to 2D resolution space with a structured scene-graph representation and query conditioning. |
| 8 | **Query-Driven Scene Graph Generation** (various, 2024-2025) | Generate scene graph elements conditioned on a specific query, avoiding unnecessary computation. | They filter SGG output by relevance. We run full SGG but *evaluate and stop* based on query-relevant subgraph recall. Complementary: their filtering could be composed with our stopping. |
| 9 | **AGQA: Compositional Spatio-Temporal Reasoning** (Grunde-McLaughlin et al., CVPR 2021) | Compositional QA benchmark with functional programs over spatio-temporal scene graphs (video). | Video domain, no resolution degradation. We adapt the GQA program-grounding idea to a resolution-stopping context. |
| 10 | **Adaptive Visual Scene Understanding: Incremental SGG** (arXiv 2310.01636, 2023) | Incremental scene graph generation that adapts to new observations. | They focus on temporal/streaming addition of new object classes. We focus on resolution-conditioned completeness of a fixed scene, with task-driven stopping. |

### 6.2 Candidate novelty claims

| # | Claim | Closest prior work | Gap we fill |
|---|-------|-------------------|-------------|
| **N1** | **Query-conditioned optimal stopping for resolution recovery via scene-graph semantics.** We define a formal stopping problem: given a question Q and an SGG pipeline, find the minimum resolution level r* such that the query-relevant subgraph is "sufficiently reconstructed" (recall saturated or answer stable). | EQA (Das 2018) -- stops navigation when answer is confident; Task-oriented SemCom -- stops transmission when task quality suffices. | Neither operates on resolution levels of a single image, neither uses SGG as the semantic representation, and neither conditions stopping on a specific VQA question grounded to scene-graph nodes. |
| **N2** | **Resolution-difficulty taxonomy for visual questions.** We propose and validate a computable resolution-difficulty score for GQA questions based on object size, attribute granularity, and relation complexity, enabling systematic study of which question types are resolution-sensitive. | HRVQA (Lobry 2023) -- high-res aerial QA; Variable-res VQA (Gizdov 2024) -- pixel budget. | Neither provides a per-question resolution-difficulty metric grounded in scene-graph node properties. |
| **N3** | **Dual-signal stopping policy (confidence + stability) evaluated against a query-conditioned oracle.** Our B+C policy combines EGTR detection confidence and cross-level answer stability, benchmarked against O2 (query-relevant subgraph recall saturation). | Rosenfeld 2025 -- stability predicts VQA correctness; RAPNet -- confidence-based early exit. | Neither combines both signals for stopping, neither defines an oracle based on query-relevant scene-graph recall, and neither evaluates on the resolution-degradation ladder. |
| **N4** | **Scene-graph-mediated QA under controlled resolution degradation.** We interpose an explicit SGG step between the image and the QA answer, enabling interpretable analysis of *why* a question becomes answerable at a certain resolution (which nodes/relations appear). | End-to-end VQA robustness studies (Rosenfeld 2025); GraphVQA (Li 2021). | End-to-end VQA is a black box; we decompose the pipeline into SGG + graph-based QA, revealing the resolution-to-semantic mapping. |
| **N5** | **Practical stopping for bandwidth-constrained visual analytics.** The resolution ladder models progressive JPEG / scalable codec layers; our policy decides when to stop requesting higher-quality layers for a given query. | Task-oriented SemCom (Zhang 2023, Xu 2024). | We provide a concrete, evaluated stopping algorithm on real images + real SGG, not a theoretical communication framework. |

### 6.3 Strongest claims (recommended for paper)

1. **N1** is the primary novelty: the combination of (resolution levels) x (scene-graph semantics) x (query conditioning) x (optimal stopping) is unexplored.
2. **N3** supports N1 with a concrete, evaluable method.
3. **N2** provides an analytical contribution (resolution-difficulty taxonomy) that is useful beyond our specific pipeline.

---

## 7. Summary of Recommendations

1. **Use GQA as the question source** -- explicit question-to-node grounding via functional programs and annotations makes oracle computation trivial.
2. **Download `sceneGraphs.zip` + `questions1.2.zip`** (total ~1.5 GB); images are already on the VG data disk.
3. **Extract query-relevant node sets** from the `semantic` program arguments (regex for `(objectID)`) and `annotations` field.
4. **Define O2** as the earliest level where query-relevant subgraph recall >= threshold (or is stable for k levels).
5. **Bin questions by resolution difficulty** using the composite score (object size, count, attribute type, question type); validate empirically.
6. **Position the novelty** around "query-conditioned optimal stopping for resolution recovery via scene-graph semantics" (N1), supported by the dual-signal policy (N3) and resolution-difficulty taxonomy (N2).
7. **Check GQA image overlap** with our current VG image set -- most VG images should appear in GQA, but verify on the data machine.
