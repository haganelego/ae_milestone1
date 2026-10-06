"""PGSG (Pix2Grp, CVPR 2024) inference adapter for the ROBOVIS2027 run.

Two layers:

* Runtime (needs the ``envs/pgsg`` env and ``PYTHONPATH=Pix2Grp_CVPR2024``):
  ``load_pgsg`` builds the model exactly like LAVIS does (model yaml merged with
  our eval yaml), ``run_pgsg_batch`` calls the stock ``model.generate`` with the
  official task arguments and captures provenance by wrapping three callables
  on the *instance* / module namespace for the duration of the call (no file in
  ``Pix2Grp_CVPR2024/`` is modified):
    - ``text_decoder.generate_from_encoder``: per-segment generated token counts
      (-> segment boundaries of the concatenated pseudo-sequence),
    - ``model.seq2instance``: decoded tokens and the parsed instance list (token
      positions, free-vocabulary label text, boxes and 150/50-way category
      scores are attached to it later by ``vocab2category`` / the pos adapter),
    - module-global ``box_ops`` (torchvision.ops.boxes): the inputs/outputs of
      the entity ``batched_nms`` inside ``_postprocess``.
* Pure numpy (torch-free, unit tested): ``build_graph`` converts the returned
  ``BoxList`` arrays + provenance into the shared graph JSON contract
  (``.claude/docs/research/robovis2027-worklog.md``).
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from milestone1.config import REPO_ROOT

PGSG_ROOT = REPO_ROOT / "Pix2Grp_CVPR2024"
PGSG_MODEL_YAML = PGSG_ROOT / "lavis/configs/models/blip_vrd_base_vg.yaml"
PGSG_MODEL_MODULE = "lavis.models.blip_models.blip_rel_det_pgsg"
EVAL_DATASET_KEY = "vg_rel_detection_eval"

SEED_MODULUS = 2**31
TOPK_DIST = 5
ROLE_SUB = 0
ROLE_OBJ = 1
NO_MATCH = -1
BOX_MATCH_TOL = 1e-2  # px; pre-NMS boxes are bit-identical copies in practice
REL_MARKER = "[REL]"

# Official call in lavis/tasks/relation_detection.py::valid_step.
OFFICIAL_REPETITION_PENALTY = 1.5
OFFICIAL_TOP_P = 0.9  # generate() default; the task does not override it
# lavis/models/med.py::generate_from_encoder hard-codes this in the sampling
# branch, so the 1.5 passed by the task is ignored when sampling.
EFFECTIVE_SAMPLING_REPETITION_PENALTY = 1.1
# blip_rel_det_pgsg.generate: per-segment min new tokens, segment count formula.
MIN_SEGMENT_TOKENS = 24
SEGMENT_DIVISOR = 2
# blip_rel_det_pgsg._postprocess constants (hard-coded upstream).
ENTITY_NMS_IOU = 0.92
MAX_RELATIONS = 512
EMPTY_STACK_ERROR = "non-empty TensorList"


# --------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------
def state_seed(image_ids: Sequence[int], level: int, base_seed: int) -> int:
    """Deterministic torch seed for one generation batch.

    ``sha256("<base_seed>|<id1,id2,...>|<level>") mod 2**31``. Python's ``hash``
    is avoided so the value is stable across interpreters. With batch size 1
    this is the per-(image_id, level) seed.
    """
    payload = f"{base_seed}|{','.join(str(i) for i in image_ids)}|{level}"
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return int(digest, 16) % SEED_MODULUS


# --------------------------------------------------------------------------
# Pure data containers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PredArrays:
    """Numpy view of PGSG's output ``BoxList`` (input-image coordinates)."""

    boxes: np.ndarray  # (N, 4) xyxy
    labels: np.ndarray  # (N,) 0-based over 150 object classes
    scores: np.ndarray  # (N,)
    dists: np.ndarray  # (N, 150) post-amplification category scores
    rel_pairs: np.ndarray  # (R, 2) indices into entities
    rel_labels: np.ndarray  # (R,) 0-based over 50 predicates
    rel_scores: np.ndarray  # (R,)
    rel_trp_scores: np.ndarray  # (R,)
    rel_dists: np.ndarray  # (R, 50)


@dataclass(frozen=True)
class Provenance:
    """Per-image generation trace needed for raw_tokens / segment ids."""

    segment_lengths: tuple[int, ...]  # generated tokens per segment
    n_rel_markers: int  # '[REL]' tokens in the sequence (all, incl. malformed)
    sub_text: tuple[str, ...]  # per parsed instance
    obj_text: tuple[str, ...]
    sub_pos: np.ndarray  # (I,) token_start of the subject span
    obj_pos: np.ndarray  # (I,)
    rel_pos: np.ndarray  # (I,) position of the instance's '[REL]' token
    sub_boxes: np.ndarray  # (I, 4) xyxy, same transform as _postprocess
    obj_boxes: np.ndarray  # (I, 4)
    pred_dists: np.ndarray  # (I, 50) predicate scores_mapped (pre-amplification)
    nms_boxes: np.ndarray  # (M, 4) entity boxes fed to batched_nms
    nms_labels: np.ndarray  # (M,)
    nms_keep: np.ndarray  # (K,) batched_nms output == BoxList entity order

    @property
    def n_instances(self) -> int:
        return len(self.sub_text)


@dataclass(frozen=True)
class StateMeta:
    """Identity and geometry of one (image, level) state."""

    image_id: int
    split: str
    level: int
    scale: float
    base_size: tuple[int, int]  # (w, h) of the PGSG input image
    orig_size: tuple[int, int]  # (w, h) of the VG original (GT coordinates)
    coord_scale: tuple[float, float]  # (x, y): orig = base * coord_scale
    seed: int


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------
def to_numpy(value: Any) -> np.ndarray:
    """Tensor / array-like -> numpy (no torch import needed)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def boxlist_to_arrays(boxlist: Any, n_instances: int) -> PredArrays:
    """Read the fields of PGSG's prediction BoxList.

    ``n_instances == 0`` means PGSG returned its 1-box/1-relation dummy
    placeholder, which is converted to empty arrays.
    """
    if n_instances == 0:
        n_obj = to_numpy(boxlist.get_field("pred_dists")).shape[-1]
        n_rel = to_numpy(boxlist.get_field("pred_rel_dist")).shape[-1]
        return PredArrays(
            boxes=np.zeros((0, 4)),
            labels=np.zeros(0, dtype=int),
            scores=np.zeros(0),
            dists=np.zeros((0, n_obj)),
            rel_pairs=np.zeros((0, 2), dtype=int),
            rel_labels=np.zeros(0, dtype=int),
            rel_scores=np.zeros(0),
            rel_trp_scores=np.zeros(0),
            rel_dists=np.zeros((0, n_rel)),
        )
    return PredArrays(
        boxes=to_numpy(boxlist.bbox).astype(np.float64),
        labels=to_numpy(boxlist.get_field("pred_labels")).astype(int),
        scores=to_numpy(boxlist.get_field("pred_scores")).astype(np.float64),
        dists=to_numpy(boxlist.get_field("pred_dists")).astype(np.float64),
        rel_pairs=to_numpy(boxlist.get_field("rel_pair_idxs")).astype(int),
        rel_labels=to_numpy(boxlist.get_field("pred_rel_label")).astype(int),
        rel_scores=to_numpy(boxlist.get_field("pred_rel_score")).astype(np.float64),
        rel_trp_scores=to_numpy(boxlist.get_field("pred_rel_trp_score")).astype(
            np.float64
        ),
        rel_dists=to_numpy(boxlist.get_field("pred_rel_dist")).astype(np.float64),
    )


def scale_boxes(boxes: np.ndarray, coord_scale: tuple[float, float]) -> np.ndarray:
    """Base/level coordinates -> original GT coordinates (per-axis)."""
    sx, sy = coord_scale
    return np.asarray(boxes, dtype=np.float64) * np.array([sx, sy, sx, sy])


def generalized_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise GIoU of xyxy boxes (same formula as lavis box_ops, no +1)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    lt_c = np.minimum(a[:, None, :2], b[None, :, :2])
    rb_c = np.maximum(a[:, None, 2:], b[None, :, 2:])
    wh_c = np.clip(rb_c - lt_c, 0, None)
    area_c = wh_c[..., 0] * wh_c[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        giou = inter / union - (area_c - union) / area_c
    return np.nan_to_num(giou, nan=-1.0)


def segment_ids_of(positions: np.ndarray, segment_lengths: Sequence[int]) -> np.ndarray:
    """Map token positions (prompt stripped) to 0-based segment indices."""
    starts = np.concatenate([[0], np.cumsum(segment_lengths)[:-1]])
    return np.searchsorted(starts, np.asarray(positions), side="right") - 1


def relink_to_kept(
    boxes: np.ndarray, labels: np.ndarray, keep: np.ndarray
) -> np.ndarray:
    """PGSG's post-NMS relink: each pre-NMS entity -> kept entity index.

    Same rule as ``_postprocess``: among kept entities with the same label,
    take the one with maximal GIoU. Returns indices into ``keep`` order.
    """
    kept_boxes = boxes[keep]
    kept_labels = labels[keep]
    out = np.full(len(boxes), NO_MATCH, dtype=int)
    for label in np.unique(kept_labels):
        kept_idx = np.where(kept_labels == label)[0]
        all_idx = np.where(labels == label)[0]
        giou = generalized_iou(boxes[all_idx], kept_boxes[kept_idx])
        out[all_idx] = kept_idx[giou.argmax(axis=1)]
    return out


def assign_rows_to_instances(prov: Provenance) -> np.ndarray:
    """Pre-NMS relation row -> parsed instance index (-1 if unmatched).

    ``_postprocess`` lays the pre-NMS entities out as [subject of rows 0..N-1,
    object of rows 0..N-1], each parsed instance expanding into
    ``rows_per_instance`` rows (top-k labels x top-k predicates) that share its
    subject/object boxes. A row is matched to the instances whose (subject box,
    object box) equal the row's boxes. Exact duplicates (same token prefix in
    different segments -> identical hidden states -> identical boxes) are
    indistinguishable: their rows are split by (subject label, object label)
    and handed out in instance order.
    """
    n_rows = len(prov.nms_boxes) // 2
    out = np.full(n_rows, NO_MATCH, dtype=int)
    if prov.n_instances == 0 or n_rows == 0:
        return out
    sub_b, obj_b = prov.nms_boxes[:n_rows], prov.nms_boxes[n_rows:]
    dist = np.maximum(
        np.abs(sub_b[:, None, :] - prov.sub_boxes[None, :, :]).max(-1),
        np.abs(obj_b[:, None, :] - prov.obj_boxes[None, :, :]).max(-1),
    )
    match = dist <= BOX_MATCH_TOL
    rows_per_instance = max(1, n_rows // prov.n_instances)
    groups: dict[tuple[int, ...], list[int]] = {}
    for row in range(n_rows):
        cands = tuple(int(i) for i in np.where(match[row])[0])
        if cands:
            groups.setdefault(cands, []).append(row)
    labels = prov.nms_labels
    for cands, rows in groups.items():
        if len(cands) == 1:
            out[rows] = cands[0]
            continue
        by_label: dict[tuple[int, int], list[int]] = {}
        for row in rows:
            key = (int(labels[row]), int(labels[row + n_rows]))
            by_label.setdefault(key, []).append(row)
        inst_iter = iter(cands)
        for label_rows in by_label.values():
            for start in range(0, len(label_rows), rows_per_instance):
                chunk = label_rows[start : start + rows_per_instance]
                out[chunk] = next(inst_iter, NO_MATCH)
    return out


def topk_pairs(dist: np.ndarray, k: int = TOPK_DIST) -> list[list[float]]:
    """Top-k (index, value) pairs of a 1-D score vector."""
    order = np.argsort(-dist, kind="stable")[:k]
    return [[int(i), float(dist[i])] for i in order]


@dataclass(frozen=True)
class _Linking:
    entity_mentions: list[set[tuple[int, int]]]  # kept entity -> {(inst, role)}
    entity_rep: list[tuple[int, int] | None]  # mention kept by NMS
    inst_kept: np.ndarray  # (I, 2) kept entity of (inst, sub/obj)


def _link(prov: Provenance, n_entities: int) -> _Linking:
    row_inst = assign_rows_to_instances(prov)
    kept_of = relink_to_kept(prov.nms_boxes, prov.nms_labels, prov.nms_keep)
    n_rows = len(row_inst)

    def mention(j: int) -> tuple[int, int] | None:
        inst = row_inst[j % n_rows]
        if inst == NO_MATCH:
            return None
        return int(inst), ROLE_SUB if j < n_rows else ROLE_OBJ

    mentions: list[set[tuple[int, int]]] = [set() for _ in range(n_entities)]
    inst_kept = np.full((prov.n_instances, 2), NO_MATCH, dtype=int)
    for j, kept in enumerate(kept_of):
        m = mention(j)
        if m is None or kept == NO_MATCH:
            continue
        mentions[kept].add(m)
        inst_kept[m] = kept
    reps = [mention(int(prov.nms_keep[k])) for k in range(n_entities)]
    return _Linking(mentions, reps, inst_kept)


def _relation_source(
    prov: Provenance, link: _Linking, pair: np.ndarray, rel_dist: np.ndarray
) -> int | None:
    """Instance that produced a kept relation (None if ambiguous / unknown).

    Candidates are instances whose subject/object relink to the relation's
    entities; ties are broken by how many of the 50 predicate scores are
    identical to the relation's distribution (only the sampled predicate is
    amplified, so the true source matches >= 49 entries).
    """
    cand = np.where(
        (link.inst_kept[:, ROLE_SUB] == pair[0])
        & (link.inst_kept[:, ROLE_OBJ] == pair[1])
    )[0]
    if len(cand) == 0:
        return None
    if len(cand) == 1:
        return int(cand[0])
    same = np.isclose(prov.pred_dists[cand], rel_dist[None, :], rtol=0, atol=1e-7)
    counts = same.sum(axis=1)
    return int(cand[counts.argmax()])


def build_graph(
    pred: PredArrays,
    prov: Provenance | None,
    meta: StateMeta,
    obj_names: Sequence[str],
    rel_names: Sequence[str],
    model_info: dict[str, Any],
    generation: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    """Assemble the graph JSON contract for one state."""
    n_ent = len(pred.labels)
    link = _link(prov, n_ent) if prov is not None and n_ent else None
    seg_of: Callable[[int], int | None]
    if prov is not None:
        lengths = prov.segment_lengths

        def seg_of(pos: int) -> int | None:
            return int(segment_ids_of(np.array([pos]), lengths)[0])
    else:

        def seg_of(pos: int) -> int | None:
            return None

    boxes_orig = scale_boxes(pred.boxes, meta.coord_scale)
    entities = []
    for k in range(n_ent):
        raw_tokens = None
        segs: list[int] | None = None
        n_mentions: int | None = None
        if link is not None and prov is not None:
            rep = link.entity_rep[k]
            if rep is not None:
                texts = prov.sub_text if rep[1] == ROLE_SUB else prov.obj_text
                raw_tokens = texts[rep[0]]
            ment = link.entity_mentions[k]
            n_mentions = len(ment)
            positions = [
                prov.sub_pos[i] if r == ROLE_SUB else prov.obj_pos[i] for i, r in ment
            ]
            segs = (
                sorted(
                    {int(s) for s in segment_ids_of(positions, prov.segment_lengths)}
                )
                if positions
                else []
            )
        label = int(pred.labels[k])
        entities.append(
            {
                "id": f"v{k + 1}",
                "label_id": label,
                "label": obj_names[label],
                "score": float(pred.scores[k]),
                "label_dist_topk": topk_pairs(pred.dists[k]),
                "raw_tokens": raw_tokens,
                "box_xyxy_orig": [float(v) for v in boxes_orig[k]],
                "box_xyxy_base": [float(v) for v in pred.boxes[k]],
                "segment_ids": segs,
                "n_mentions": n_mentions,
            }
        )

    relations = []
    for r in range(len(pred.rel_labels)):
        sub, obj = (int(v) for v in pred.rel_pairs[r])
        segment_id = None
        if link is not None and prov is not None:
            src = _relation_source(prov, link, pred.rel_pairs[r], pred.rel_dists[r])
            segment_id = None if src is None else seg_of(int(prov.rel_pos[src]))
        pred_id = int(pred.rel_labels[r])
        relations.append(
            {
                "id": f"e{r + 1}",
                "sub": f"v{sub + 1}",
                "obj": f"v{obj + 1}",
                "pred_id": pred_id,
                "pred": rel_names[pred_id],
                "pred_score": float(pred.rel_scores[r]),
                "trp_score": float(pred.rel_trp_scores[r]),
                "pred_dist_topk": topk_pairs(pred.rel_dists[r]),
                "rank": r,
                "segment_id": segment_id,
            }
        )

    n_raw = prov.n_instances if prov is not None else None
    n_bad = prov.n_rel_markers - prov.n_instances if prov is not None else None
    return {
        "image_id": meta.image_id,
        "split": meta.split,
        "level": meta.level,
        "scale": meta.scale,
        "base_size": list(meta.base_size),
        "orig_size": list(meta.orig_size),
        "coord_scale": list(meta.coord_scale),
        "seed": meta.seed,
        "model": model_info,
        "generation": generation,
        "entities": entities,
        "relations": relations,
        "stats": {
            "n_bad_inst": n_bad,
            "n_raw_triplets": n_raw,
            "n_relations_kept": len(relations),
            "elapsed_s": round(float(elapsed_s), 4),
        },
    }


def check_graph_consistency(pred: PredArrays, prov: Provenance | None) -> list[str]:
    """Sanity checks of the provenance capture; returns problem strings."""
    problems: list[str] = []
    trp = pred.rel_trp_scores
    if len(trp) > 1 and np.any(np.diff(trp) > 1e-9):
        problems.append("relations not sorted by trp_score")
    if prov is None or len(pred.labels) == 0:
        return problems
    if len(prov.nms_keep) != len(pred.labels):
        problems.append("nms keep size != entity count")
        return problems
    if not np.allclose(prov.nms_boxes[prov.nms_keep], pred.boxes, atol=1e-3):
        problems.append("BoxList boxes != NMS-kept boxes")
    if not np.array_equal(prov.nms_labels[prov.nms_keep], pred.labels):
        problems.append("BoxList labels != NMS-kept labels")
    if len(prov.nms_boxes) % 2:
        problems.append("odd number of pre-NMS entities")
    unmatched = int((assign_rows_to_instances(prov) == NO_MATCH).sum())
    if unmatched:
        problems.append(f"{unmatched} pre-NMS rows without instance match")
    return problems


# --------------------------------------------------------------------------
# Runtime (envs/pgsg only)
# --------------------------------------------------------------------------
@dataclass
class PgsgRuntime:
    model: Any
    processor: Any
    device: str
    run_cfg: Any  # OmegaConf run section of the eval yaml
    resolved: dict[str, Any]


def load_pgsg(eval_config: Path, device: str = "cuda") -> PgsgRuntime:
    """Build PGSG like LAVIS: model yaml merged with our eval yaml's model block."""
    import lavis  # noqa: F401  (registers models / processors)
    from lavis.common.registry import registry
    from omegaconf import OmegaConf

    eval_cfg = OmegaConf.load(eval_config)
    model_cfg = OmegaConf.merge(OmegaConf.load(PGSG_MODEL_YAML).model, eval_cfg.model)
    model_cls = registry.get_model_class(model_cfg.arch)
    model = model_cls.from_config(model_cfg).to(device).eval()
    proc_cfg = eval_cfg.datasets[EVAL_DATASET_KEY].vis_processor.eval
    processor = registry.get_processor_class(proc_cfg.name).from_config(proc_cfg)
    run_cfg = eval_cfg.run
    return PgsgRuntime(
        model, processor, device, run_cfg, resolve_settings(model, run_cfg)
    )


def resolve_settings(model: Any, run_cfg: Any) -> dict[str, Any]:
    """Generation / post-processing settings actually in effect."""
    seg_len = int(model.seg_len)
    max_len = int(run_cfg.max_len)
    num_segments = max_len // int((seg_len + MIN_SEGMENT_TOKENS) / SEGMENT_DIVISOR)
    gen_cfg = getattr(model.text_decoder, "generation_config", None)
    hf_top_k = getattr(gen_cfg, "top_k", None)
    return {
        "generation_mode": str(run_cfg.generation_mode),
        "use_nucleus_sampling": str(run_cfg.generation_mode) == "sampling",
        "num_beams": int(run_cfg.num_beams),
        "max_length": max_len,
        "min_length": int(run_cfg.min_len),
        "min_length_note": "passed by the task but unused by PGSG generate()",
        "top_p": OFFICIAL_TOP_P,
        "repetition_penalty_passed": OFFICIAL_REPETITION_PENALTY,
        "repetition_penalty_effective": EFFECTIVE_SAMPLING_REPETITION_PENALTY,
        "repetition_penalty_note": "med.py generate_from_encoder hard-codes 1.1 when sampling",
        "hf_top_k": hf_top_k,
        "seg_len": seg_len,
        "segment_max_new_tokens": seg_len,
        "segment_min_new_tokens": MIN_SEGMENT_TOKENS,
        "num_segments": num_segments,
        "prompt": str(model.prompt),
        "prompt_length": int(model.prompt_length),
        "top_k_ent_label_num": int(model.top_k_label_num),
        "top_k_predicate_label_num": int(model.top_k_predicate_label_num),
        "post_proc_cfg": {k: float(v) for k, v in dict(model.post_proc_cfg).items()},
        "entity_nms_iou": ENTITY_NMS_IOU,
        "max_relations": MAX_RELATIONS,
    }


def graph_generation_block(resolved: dict[str, Any]) -> dict[str, Any]:
    """The compact ``generation`` block stored in every graph JSON."""
    return {
        "max_length": resolved["max_length"],
        "min_length": resolved["min_length"],
        "top_p": resolved["top_p"],
        "repetition_penalty": resolved["repetition_penalty_effective"],
        "num_segments": resolved["num_segments"],
    }


class _NmsRecorder:
    """Stand-in for ``torchvision.ops.boxes`` that records batched_nms calls."""

    def __init__(self, real: Any) -> None:
        self._real = real
        self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def batched_nms(
        self, boxes: Any, scores: Any, idxs: Any, iou_threshold: float
    ) -> Any:
        keep = self._real.batched_nms(boxes, scores, idxs, iou_threshold)
        self.calls.append(
            (
                to_numpy(boxes).astype(np.float64),
                to_numpy(idxs).astype(int),
                to_numpy(keep).astype(int),
            )
        )
        return keep

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


@dataclass
class _Capture:
    segment_lengths: list[int]
    raw_captions: list[list[str]] | None = None
    instances: list[list[dict[str, Any]]] | None = None
    nms: _NmsRecorder | None = None


@contextmanager
def _capture(model: Any) -> Iterator[_Capture]:
    import importlib

    module = importlib.import_module(PGSG_MODEL_MODULE)
    cap = _Capture(segment_lengths=[])
    decoder = model.text_decoder
    orig_gen = decoder.generate_from_encoder
    orig_s2i = model.seq2instance
    orig_box_ops = module.box_ops
    prompt_len = int(model.prompt_length)

    def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
        out = orig_gen(*args, **kwargs)
        cap.segment_lengths.append(int(out.sequences.shape[1]) - prompt_len)
        return out

    def s2i_wrapper(
        decoder_out: Any, raw_caption: Any, *args: Any, **kwargs: Any
    ) -> Any:
        out = orig_s2i(decoder_out, raw_caption, *args, **kwargs)
        cap.raw_captions = [list(c) for c in raw_caption]
        cap.instances = out
        return out

    cap.nms = _NmsRecorder(orig_box_ops)
    decoder.generate_from_encoder = gen_wrapper
    model.seq2instance = s2i_wrapper
    module.box_ops = cap.nms
    try:
        yield cap
    finally:
        del decoder.generate_from_encoder
        del model.seq2instance
        module.box_ops = orig_box_ops


def _instance_boxes(
    instances: list[dict[str, Any]], role: str, size_hw: list[int]
) -> np.ndarray:
    import torch

    module = __import__(PGSG_MODEL_MODULE, fromlist=["xywh2xyxy"])
    if not instances:
        return np.zeros((0, 4))
    boxes = torch.stack([inst[role]["boxes"].detach().cpu() for inst in instances])
    boxes = module.xywh2xyxy(boxes, size_hw)
    invalid = boxes[:, 2:] < boxes[:, :2]  # same fix as _postprocess
    boxes[:, :2][invalid] = boxes[:, 2:][invalid] - 0.001
    return to_numpy(boxes).astype(np.float64)


def _provenance(
    cap: _Capture,
    bi: int,
    nms_call: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    size_hw: list[int],
    prompt_len: int,
) -> Provenance:
    assert cap.instances is not None and cap.raw_captions is not None
    insts = cap.instances[bi]
    tokens = cap.raw_captions[bi][prompt_len:]
    if sum(cap.segment_lengths) != len(tokens):
        raise RuntimeError("segment lengths do not cover the generated sequence")
    nms_boxes, nms_labels, nms_keep = (
        nms_call
        if nms_call is not None
        else (np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0, dtype=int))
    )
    return Provenance(
        segment_lengths=tuple(cap.segment_lengths),
        n_rel_markers=sum(1 for t in tokens if t.upper() == REL_MARKER),
        sub_text=tuple(str(i["sub"]["label"]) for i in insts),
        obj_text=tuple(str(i["obj"]["label"]) for i in insts),
        sub_pos=np.array([int(i["sub"]["token_start"]) for i in insts], dtype=int),
        obj_pos=np.array([int(i["obj"]["token_start"]) for i in insts], dtype=int),
        rel_pos=np.array([int(i["predicate"]["token_end"]) for i in insts], dtype=int),
        sub_boxes=_instance_boxes(insts, "sub", size_hw),
        obj_boxes=_instance_boxes(insts, "obj", size_hw),
        pred_dists=np.stack(
            [
                to_numpy(i["predicate"]["scores_mapped"]).astype(np.float64)
                for i in insts
            ]
        )
        if insts
        else np.zeros((0, 0)),
        nms_boxes=nms_boxes,
        nms_labels=nms_labels,
        nms_keep=nms_keep,
    )


def seed_everything(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass
class BatchResult:
    predictions: list[PredArrays]
    provenance: list[Provenance]
    elapsed_s: float


def run_pgsg_batch(
    rt: PgsgRuntime, images: Sequence[Any], image_ids: Sequence[int], seed: int
) -> BatchResult:
    """One stock ``model.generate`` call on PIL images (sizes may differ).

    ``orig_size`` is each input image's own size, so returned boxes are in
    input (= 384-base) coordinates. Raises RuntimeError from PGSG when an image
    yields no parsable triplet (upstream ``torch.stack([])`` in the pos
    adapter); callers fall back to batch size 1 / an empty graph.
    """
    import torch

    tensors, targets = [], []
    for img, image_id in zip(images, image_ids, strict=True):
        tensor, _ = rt.processor(img, None)
        tensors.append(tensor)
        targets.append(
            {
                "orig_size": torch.tensor([img.height, img.width]),
                "boxes": torch.zeros((0, 4)),
                "det_labels": torch.zeros(0, dtype=torch.long),
                "rel_tripets": torch.zeros((0, 3), dtype=torch.long),
                "image_id": image_id,
            }
        )
    samples = {
        "image": torch.stack(tensors).to(rt.device),
        "targets": targets,
        "image_id": list(image_ids),
    }
    seed_everything(seed)
    start = time.perf_counter()
    with torch.no_grad(), _capture(rt.model) as cap:
        preds, _, _ = rt.model.generate(
            samples,
            use_nucleus_sampling=rt.resolved["use_nucleus_sampling"],
            num_beams=rt.resolved["num_beams"],
            max_length=rt.resolved["max_length"],
            min_length=rt.resolved["min_length"],
            top_p=OFFICIAL_TOP_P,
            repetition_penalty=OFFICIAL_REPETITION_PENALTY,
            num_captions=1,
        )
    if rt.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert cap.instances is not None and cap.nms is not None
    prompt_len = int(rt.model.prompt_length)
    calls = iter(cap.nms.calls)
    out_preds, out_prov = [], []
    for bi, img in enumerate(images):
        n_inst = len(cap.instances[bi])
        nms_call = next(calls) if n_inst > 0 else None
        out_prov.append(
            _provenance(cap, bi, nms_call, [img.height, img.width], prompt_len)
        )
        out_preds.append(boxlist_to_arrays(preds[bi], n_inst))
    cap.instances = None  # release GPU hidden states
    return BatchResult(out_preds, out_prov, elapsed)


def empty_prediction(n_obj: int, n_rel: int) -> PredArrays:
    """Prediction for an image where PGSG parsed no triplet at all."""
    return PredArrays(
        boxes=np.zeros((0, 4)),
        labels=np.zeros(0, dtype=int),
        scores=np.zeros(0),
        dists=np.zeros((0, n_obj)),
        rel_pairs=np.zeros((0, 2), dtype=int),
        rel_labels=np.zeros(0, dtype=int),
        rel_scores=np.zeros(0),
        rel_trp_scores=np.zeros(0),
        rel_dists=np.zeros((0, n_rel)),
    )


def is_empty_stack_error(err: BaseException) -> bool:
    return isinstance(err, RuntimeError) and EMPTY_STACK_ERROR in str(err)
