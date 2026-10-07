"""MIRU stop methods: existing stop CSVs + score-directory methods.

Score-directory methods use the MIRU ``llm_scores`` file format
(``<dir>/<image_id>/level_XX.json`` with ``saturation_score``) and exactly the
stop machinery of ``scripts/13_aggregate_evaluation.py``:

- ``<key>_k<k>``: first level t >= start_level whose k most recent scores
  (t-k+1..t, none before start_level) are all >= ``llm.saturation_threshold``;
  20 if never. Missing / failed levels score 0.
- ``hybrid_<key>_k<k>``: same rule on
  ``conf_weight * mean_object_confidence + llm_weight * score`` with
  ``ablation.hybrid.threshold`` (0 where either input is missing).

Registered via ``evaluation_extra`` in ``configs/config.yaml``.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from milestone1.config import resolve

MISSING_SCORE = 0.0
HYBRID_PREFIX = "hybrid_"

# Existing MIRU stop CSVs (outputs/metrics) -> method names, as in
# scripts/eval_topk_oracle.py.
BASELINE_COLUMN = "baseline_0.35"
ABLATION_K1 = {
    "qwen3.5-4b_h5_rel_on_conf_on": "llm_sgg_4b_h5_k1",
    "qwen3.5-9b_h5_rel_on_conf_on": "llm_sgg_9b_h5_k1",
}
ABLATION_K2 = {
    "qwen3.5-4b_h5_rel_on_conf_on": "llm_sgg_4b_h5_k2",
    "qwen3.5-9b_h5_rel_on_conf_on": "llm_sgg_9b_h5_k2",
}
VLM_METHODS = {
    ("qwen3.5:4b", "single", 1): "vlm_4b_single_k1",
    ("qwen3.5:4b", "single", 2): "vlm_4b_single_k2",
    ("qwen3.5:9b", "single", 1): "vlm_9b_single_k1",
    ("qwen3.5:9b", "single", 2): "vlm_9b_single_k2",
}
HYBRID_VARIANTS = ("hybrid_4b_k1", "hybrid_4b_k2", "hybrid_9b_k1", "hybrid_9b_k2")


@dataclass(frozen=True)
class StopParams:
    start_level: int
    max_level: int
    threshold: float
    hybrid_threshold: float
    conf_weight: float
    llm_weight: float
    k_options: tuple[int, ...]

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> StopParams:
        hyb = cfg["ablation"]["hybrid"]
        extra = cfg.get("evaluation_extra") or {}
        return cls(
            start_level=int(cfg["llm"]["start_level"]),
            max_level=int(cfg["resolution_levels"]["num_levels"]),
            threshold=float(cfg["llm"]["saturation_threshold"]),
            hybrid_threshold=float(hyb["threshold"]),
            conf_weight=float(hyb["conf_weight"]),
            llm_weight=float(hyb["llm_weight"]),
            k_options=tuple(int(k) for k in extra.get("k_options", (1, 2))),
        )


def consecutive_k_stop(
    scores: Mapping[int, float],
    threshold: float,
    k: int,
    start_level: int,
    max_level: int,
) -> int:
    """MIRU rule (script 13): first t whose k scores ending at t are >= threshold."""
    for lvl in range(start_level, max_level + 1):
        window = [lvl - offset for offset in range(k)]
        if any(w < start_level for w in window):
            continue
        if all(scores.get(w, MISSING_SCORE) >= threshold for w in window):
            return lvl
    return max_level


def load_scores(
    score_dir: Path, image_id: int, start_level: int, max_level: int
) -> dict[int, float]:
    """{level: saturation_score} for present, non-failed files."""
    out: dict[int, float] = {}
    for lvl in range(start_level, max_level + 1):
        path = score_dir / str(image_id) / f"level_{lvl:02d}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict) or data.get("failure") is True:
            continue
        score = data.get("saturation_score")
        if isinstance(score, int | float) and not isinstance(score, bool):
            out[lvl] = float(score)
    return out


def load_object_confidence(all_metrics_csv: Path) -> dict[tuple[int, int], float]:
    """(image_id, level) -> mean_object_confidence from outputs/metrics/all_metrics.csv."""
    with open(all_metrics_csv) as f:
        return {
            (int(r["image_id"]), int(r["level"])): float(r["mean_object_confidence"])
            for r in csv.DictReader(f)
        }


def fused_scores(
    scores: Mapping[int, float],
    conf: Mapping[tuple[int, int], float],
    image_id: int,
    params: StopParams,
) -> dict[int, float]:
    """Hybrid series as in script 13 (0 where the score or the confidence is missing)."""
    out: dict[int, float] = {}
    for lvl in range(params.start_level, params.max_level + 1):
        key = (image_id, lvl)
        if key in conf and lvl in scores:
            out[lvl] = params.conf_weight * conf[key] + params.llm_weight * scores[lvl]
        else:
            out[lvl] = MISSING_SCORE
    return out


@dataclass(frozen=True)
class ScoreMethodResult:
    stops: dict[str, dict[int, int]]
    coverage: dict[str, float]  # key -> fraction of (image, level) score files present


def stops_from_series(
    key: str,
    series: Mapping[int, Mapping[int, float]],
    params: StopParams,
    conf: Mapping[tuple[int, int], float] | None,
) -> dict[str, dict[int, int]]:
    """``<key>_k<k>`` (and ``hybrid_<key>_k<k>`` if conf is given) from score series."""
    stops: dict[str, dict[int, int]] = {}
    for image_id, scores in series.items():
        fused = (
            fused_scores(scores, conf, image_id, params) if conf is not None else None
        )
        for k in params.k_options:
            stops.setdefault(f"{key}_k{k}", {})[image_id] = consecutive_k_stop(
                scores, params.threshold, k, params.start_level, params.max_level
            )
            if fused is not None:
                stops.setdefault(f"{HYBRID_PREFIX}{key}_k{k}", {})[image_id] = (
                    consecutive_k_stop(fused, params.hybrid_threshold, k,
                                       params.start_level, params.max_level)
                )  # fmt: skip
    return stops


def load_series(
    score_dir: Path, image_ids: Sequence[int], params: StopParams
) -> tuple[dict[int, dict[int, float]], float]:
    """Per-image score series and coverage (present files / expected files)."""
    series = {
        int(i): load_scores(score_dir, int(i), params.start_level, params.max_level)
        for i in image_ids
    }
    n_levels = params.max_level - params.start_level + 1
    present = sum(len(v) for v in series.values())
    coverage = present / (n_levels * len(image_ids)) if image_ids else 0.0
    return series, coverage


def score_method_stops(
    key: str,
    score_dir: Path,
    image_ids: Sequence[int],
    params: StopParams,
    conf: Mapping[tuple[int, int], float] | None,
) -> tuple[dict[str, dict[int, int]], float]:
    """Stops of ``<key>_k<k>`` (and ``hybrid_<key>_k<k>`` if conf is given)."""
    series, coverage = load_series(score_dir, image_ids, params)
    return stops_from_series(key, series, params, conf), coverage


FUSION_OPS = {"mean": lambda vals: sum(vals) / len(vals), "min": min, "max": max}


def fuse_series(
    sources: Sequence[Mapping[int, Mapping[int, float]]], op: str
) -> dict[int, dict[int, float]]:
    """Level-wise late fusion; a level missing in any source is left out (-> 0)."""
    fn = FUSION_OPS[op]
    out: dict[int, dict[int, float]] = {}
    for image_id in sources[0]:
        per = [src.get(image_id, {}) for src in sources]
        common = set(per[0]).intersection(*per[1:])
        out[image_id] = {
            lvl: float(fn([p[lvl] for p in per])) for lvl in sorted(common)
        }
    return out


def registered_score_methods(
    cfg: Mapping[str, Any],
    image_ids: Sequence[int],
    metrics_dir: Path,
    block: Mapping[str, Any] | None = None,
    conf: Mapping[tuple[int, int], float] | None = None,
) -> ScoreMethodResult:
    """``score_methods`` (existing dirs) + ``fusion_methods`` of an evaluation block.

    ``block`` defaults to ``cfg["evaluation_extra"]``; ``conf`` (mean object
    confidence per (image, level)) defaults to ``<metrics_dir>/all_metrics.csv``.
    """
    extra = block if block is not None else (cfg.get("evaluation_extra") or {})
    params = replace(
        StopParams.from_config(cfg),
        k_options=tuple(int(k) for k in extra.get("k_options", (1, 2))),
    )
    hybrid_keys = set(extra.get("hybrid_methods") or [])
    fusions = extra.get("fusion_methods") or {}
    hybrid_keys |= {name for name, spec in fusions.items() if spec.get("hybrid", True)}
    if conf is None and hybrid_keys:
        conf = load_object_confidence(metrics_dir / "all_metrics.csv")

    stops: dict[str, dict[int, int]] = {}
    coverage: dict[str, float] = {}
    series_by_key: dict[str, dict[int, dict[int, float]]] = {}
    for key, spec in (extra.get("score_methods") or {}).items():
        # spec: "<dir>" or {"dir": "<dir>", "threshold": float}
        rel_dir = spec["dir"] if isinstance(spec, Mapping) else spec
        method_params = params
        if isinstance(spec, Mapping) and "threshold" in spec:
            method_params = replace(params, threshold=float(spec["threshold"]))
        score_dir = resolve(str(rel_dir))
        if not score_dir.exists():
            print(f"  skip score method {key}: {score_dir} not found")
            continue
        series, cov = load_series(score_dir, image_ids, method_params)
        series_by_key[key] = series
        coverage[key] = cov
        stops.update(stops_from_series(
            key, series, method_params, conf if key in hybrid_keys else None))  # fmt: skip
    for name, spec in fusions.items():
        missing = [src for src in spec["sources"] if src not in series_by_key]
        if missing:
            print(f"  skip fusion {name}: sources {missing} not available")
            continue
        fused = fuse_series([series_by_key[src] for src in spec["sources"]], spec["op"])
        n_levels = params.max_level - params.start_level + 1
        coverage[name] = sum(len(v) for v in fused.values()) / (
            n_levels * len(image_ids)
        )
        fusion_params = replace(
            params, threshold=float(spec.get("threshold", params.threshold))
        )
        stops.update(stops_from_series(
            name, fused, fusion_params, conf if name in hybrid_keys else None))  # fmt: skip
    return ScoreMethodResult(stops, coverage)


# --------------------------------------------------------------------------
# Existing MIRU stop CSVs (same selection as scripts/eval_topk_oracle.py)
# --------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load_existing_method_stops(
    metrics_dir: Path, image_ids: Sequence[int]
) -> dict[str, dict[int, int]]:
    """baseline / llm_sgg_*_h5_k{1,2} / vlm_*_single_k{1,2} / hybrid_*_k{1,2}."""
    ids = set(image_ids)
    methods: dict[str, dict[int, int]] = {}

    def add(name: str, rows: list[dict[str, str]], col: str) -> None:
        sel = {
            int(r["image_id"]): int(float(r[col]))
            for r in rows
            if int(r["image_id"]) in ids
        }
        if sel:
            methods[name] = sel

    path = metrics_dir / "baseline_stop.csv"
    if path.exists():
        add("baseline", _read_csv(path), BASELINE_COLUMN)
    for fname, mapping, col in (
        ("llm_ablation_stop.csv", ABLATION_K1, "stop_level"),
        ("llm_ablation_k2_stop.csv", ABLATION_K2, "stop_k2"),
    ):
        path = metrics_dir / fname
        if path.exists():
            rows = _read_csv(path)
            for slug, name in mapping.items():
                add(name, [r for r in rows if r["config_slug"] == slug], col)
    path = metrics_dir / "vlm_stop.csv"
    if path.exists():
        rows = _read_csv(path)
        for (model, mode, k), name in VLM_METHODS.items():
            add(name, [r for r in rows if r["model"] == model and r["mode"] == mode
                       and int(r["k"]) == k], "stop_level")  # fmt: skip
    path = metrics_dir / "hybrid_stop.csv"
    if path.exists():
        rows = _read_csv(path)
        for variant in HYBRID_VARIANTS:
            add(variant, [r for r in rows if r["variant"] == variant], "stop_level")
    return methods
