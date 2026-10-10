"""Plan §25.3 / §25.5 V6-A: second-reader (cross-model) answer agreement.

Second readers (gemma3:latest = other family, qwen3.5:4b = same-family control)
run the UNCHANGED answer_only prompt with the A13 clean decoding block on the
dev visual subset (102) and the dev reader-wrong set (99), levels 1..20.
Records live in their own role directory ``reader_<key>/`` (the namespace is
part of the cache key; the model digest is part of the cache key and the record
path), so no existing record is touched. Requests also ask for top-10
logprobs so that greedy decoding (emitted token == argmax) can be verified for
models whose Modelfile sets other sampler defaults.

agree_m(ℓ) = normalised primary answer == normalised second-reader answer
(NFKC, lowercase, strip, collapse whitespace; no synonym merging).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from . import v6_eval as ev
from .client import (
    REPAIR_PROMPT,
    STATUS_TRANSPORT_FAILURE,
    STATUS_VALID,
    ModelSettings,
    PromptSet,
    RequestEngine,
    RequestSpec,
    monitor_from_config,
)
from .common import atomic_write_json, read_json, utc_now
from .context import Context, require
from .evaluate import cluster_bootstrap_indices, is_correct, normalize_answer
from .inference import READER_ROLE, build_level_spec
from .logprob_v4 import TOP_LOGPROBS, locate_answer_token
from .preflight import digest_of, ollama_tags
from .schemas import validate_reader
from .stages_v4c import argmax_violations
from .v6_common import (
    CPU_CLIP_PROMPT_EVAL_S,
    EXT_DIR,
    OUT_DIR,
    PILOT_LEVELS,
    PILOT_QUESTIONS,
    V6Sets,
    isolate_model,
    load_sets,
    prompt_eval_seconds,
    update_v6_config,
    warm_up,
)

SECOND_READERS: dict[str, str] = {"gemma3": "gemma3:latest", "qwen4b": "qwen3.5:4b"}
PRIMARY_KEY = "gemma3"
GATE_RATE = 0.95
DETERMINISM_REQUESTS = 20
DISAGREEMENT_SAMPLE = 30
PERCENTILES = (2.5, 97.5)
HTTP_TIMEOUT_S = 10


# ---- engine / specs ---------------------------------------------------------------------
def reader_settings(ctx: Context, model_name: str) -> ModelSettings:
    """Primary settings with another model (same A13 decoding block, think=false)."""
    tags = ollama_tags(ctx.cfg["model"]["endpoint"], ctx.tags_get)
    require(bool(tags["reachable"]), "ollama /api/tags unreachable")
    digest = digest_of(tags, model_name)
    require(digest is not None, f"model {model_name} is not installed in ollama")
    return dataclasses.replace(ctx.settings, name=model_name, digest=digest)


def reader_engine(ctx: Context, key: str, root: Path) -> RequestEngine:
    settings = reader_settings(ctx, SECOND_READERS[key])
    texts = {
        READER_ROLE: ctx.prompts.texts[READER_ROLE],
        REPAIR_PROMPT: ctx.prompts.texts[REPAIR_PROMPT],
    }
    engine = RequestEngine(
        settings,
        PromptSet(texts),
        ctx.transport_factory(settings),
        root,
        monitor_from_config(ctx.cfg),
    )
    engine.stage = f"v6_reader_{key}"
    return engine


def second_reader_spec(
    question: Mapping[str, Any], level: int, image: bytes, key: str
) -> RequestSpec:
    """answer_only prompt unchanged; namespace = reader key (own role dir)."""
    spec = build_level_spec(READER_ROLE, question, level, image, namespace=key)
    return dataclasses.replace(spec, logprobs=TOP_LOGPROBS)


def value_token_entries(
    logprobs: Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    """Tokens of the JSON answer value (first value token up to the closing quote).

    With format=json the grammar forces structural tokens such as the opening
    '{"' even when the unconstrained argmax is another token (gemma3 often
    prefers a bare 'Gray' or a code fence there), so greedy decoding is checked
    on the value tokens only.
    """
    token = locate_answer_token(logprobs)
    if token is None or not logprobs:
        return []
    out = [logprobs[token.index]]
    for entry in logprobs[token.index + 1 :]:
        if '"' in str(entry.get("token", "")):
            break
        out.append(entry)
    return out


def record_row(
    question: Mapping[str, Any], level: int, record: Mapping[str, Any]
) -> dict[str, Any]:
    ok = record.get("status") == STATUS_VALID
    positions, bad = argmax_violations(record.get("logprobs"))
    value_positions, value_bad = argmax_violations(
        value_token_entries(record.get("logprobs"))
    )
    return {
        "question_id": str(question["question_id"]),
        "level": level,
        "answer2": (record.get("parsed") or {}).get("answer") if ok else None,
        "status2": record.get("status"),
        "initial_valid2": bool(record.get("initial_valid")),
        "attempts2": int(record.get("attempt", 0)),
        "argmax_positions": positions,
        "argmax_violations": bad,
        "value_argmax_positions": value_positions,
        "value_argmax_violations": value_bad,
        "prompt_eval_s": prompt_eval_seconds(record),
    }


def run_reader(
    ctx: Context,
    engine: RequestEngine,
    key: str,
    questions: Sequence[Mapping[str, Any]],
    levels: Sequence[int],
) -> pd.DataFrame:
    rows = []
    for q in questions:
        for level in levels:
            image = ctx.series.bytes_for(str(q["image_id"]), level)
            record = engine.run(
                second_reader_spec(q, level, image, key), validate_reader
            )
            rows.append(record_row(q, level, record))
    return pd.DataFrame(rows)


# ---- agreement --------------------------------------------------------------------------
def norm(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return normalize_answer(str(value))


def agreement_flags(primary: Sequence[Any], second: Sequence[Any]) -> np.ndarray:
    """Exact match after normalisation; a missing answer never agrees."""
    out = []
    for a, b in zip(primary, second, strict=True):
        na, nb = norm(a), norm(b)
        out.append(na is not None and nb is not None and na == nb)
    return np.array(out, dtype=bool)


def with_agreement(
    table: pd.DataFrame, reader2: pd.DataFrame, key: str
) -> pd.DataFrame:
    """Adds answer_<key>, agree_<key>, correct_<key> (GT) to a level table."""
    merged = table.merge(
        reader2[["question_id", "level", "answer2"]].rename(
            columns={"answer2": f"answer_{key}"}
        ),
        on=["question_id", "level"],
        how="left",
        validate="one_to_one",
    )
    merged[f"agree_{key}"] = agreement_flags(
        merged["reader_answer"].tolist(), merged[f"answer_{key}"].tolist()
    )
    return merged


def agreement_by_level(frame: pd.DataFrame, key: str) -> dict[int, float]:
    col = f"agree_{key}"
    return {int(k): float(v) for k, v in frame.groupby("level")[col].mean().items()}


def union_bootstrap_diff(
    values_a: Mapping[str, float],
    values_b: Mapping[str, float],
    groups: Mapping[str, str],
    repeats: int,
    seed: int,
) -> dict[str, float]:
    """mean(A) - mean(B) with images resampled over the union (shared images keep
    their questions of both sets together)."""
    qids = sorted(set(values_a) | set(values_b))
    in_a = np.array([q in values_a for q in qids])
    vals = np.array([values_a.get(q, values_b.get(q, np.nan)) for q in qids], float)
    point = float(vals[in_a].mean() - vals[~in_a].mean())
    draws = []
    for idx in cluster_bootstrap_indices([groups[q] for q in qids], repeats, seed):
        a, b = vals[idx][in_a[idx]], vals[idx][~in_a[idx]]
        if len(a) and len(b):
            draws.append(float(a.mean() - b.mean()))
    lo, hi = (float(np.percentile(draws, p)) for p in PERCENTILES)
    return {"diff": point, "ci_low": lo, "ci_high": hi}


# ---- pilot ------------------------------------------------------------------------------
def gpu_placement(endpoint: str, model_name: str) -> dict[str, Any]:
    """/api/ps: fraction of the loaded model held in VRAM (1.0 = 100% GPU)."""
    url = endpoint.replace("/api/chat", "/api/ps")
    try:
        models = requests.get(url, timeout=HTTP_TIMEOUT_S).json().get("models", [])
    except (requests.RequestException, ValueError) as exc:
        return {"loaded": False, "error": str(exc)}
    for m in models:
        if m.get("name") == model_name or m.get("model") == model_name:
            size, vram = float(m.get("size", 0)), float(m.get("size_vram", 0))
            return {"loaded": True, "vram_fraction": vram / size if size else 0.0}
    return {"loaded": False}


def determinism_check(
    ctx: Context, engine: RequestEngine, key: str, specs: Sequence[RequestSpec]
) -> dict[str, Any]:
    """Each request sent twice (no cache): the generated text must be identical
    (plan §25.5) and equal to the stored record. Logprob identity is reported
    only: a repeat served from ollama's prompt cache shifts logprobs in the
    3rd-4th decimal without changing the greedy output."""
    same_pair = same_record = same_logprobs = 0
    for spec in specs:
        payload = engine.build_payload(spec, engine.build_messages(spec))
        a, b = engine.transport.send(payload), engine.transport.send(payload)
        record = engine.lookup(spec) or {}
        same_pair += a.content == b.content
        same_logprobs += a.logprobs == b.logprobs
        same_record += record.get("attempt") == 1 and record.get("raw") == a.content
    return {
        "requests": len(specs),
        "pair_identical": same_pair,
        "record_identical": same_record,
        "pair_logprobs_identical": same_logprobs,
        "passed": same_pair == len(specs) and same_record == len(specs),
    }


def pilot_questions(sets: V6Sets) -> tuple[list[dict], list[dict]]:
    dev = [sets.dev_rows[q] for q in sets.dev_qids[:PILOT_QUESTIONS]]
    wrong = [sets.wrong_rows[q] for q in sets.wrong_qids[:PILOT_QUESTIONS]]
    return dev, wrong


def pilot(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("v6_agree_pilot")
    sets = load_sets(ctx)
    dev_q, wrong_q = pilot_questions(sets)
    if ctx.dry_run:
        n = (len(dev_q) + len(wrong_q)) * len(PILOT_LEVELS) * len(SECOND_READERS)
        ctx.echo(f"[dry-run] V6-A pilot: {n} calls + determinism")
        return {}
    root = ctx.paths.output_root
    out = ctx.paths.out(OUT_DIR)
    results: dict[str, Any] = {}
    for key, model_name in SECOND_READERS.items():
        unloaded = isolate_model(ctx.cfg["model"]["endpoint"], model_name)
        warm_up(ctx.cfg["model"]["endpoint"], model_name)
        engine_dev = reader_engine(ctx, key, root)
        engine_wrong = reader_engine(ctx, key, root / EXT_DIR)
        frame = pd.concat(
            [
                run_reader(ctx, engine_dev, key, dev_q, PILOT_LEVELS).assign(
                    set="dev102"
                ),
                run_reader(ctx, engine_wrong, key, wrong_q, PILOT_LEVELS).assign(
                    set="wrong99"
                ),
            ]
        )
        placement = gpu_placement(ctx.cfg["model"]["endpoint"], model_name)
        det_specs = [
            second_reader_spec(
                q, level, ctx.series.bytes_for(str(q["image_id"]), level), key
            )
            for q in dev_q
            for level in PILOT_LEVELS
        ][:DETERMINISM_REQUESTS]
        det = determinism_check(ctx, engine_dev, key, det_specs)
        primary = pd.concat(
            [
                sets.dev_table[["question_id", "level", "reader_answer"]],
                sets.wrong_table[["question_id", "level", "reader_answer"]],
            ]
        )
        frame = frame.merge(primary, on=["question_id", "level"], how="left")
        frame["agree"] = agreement_flags(
            frame["reader_answer"].tolist(), frame["answer2"].tolist()
        )
        disagree = frame[~frame["agree"] & frame["answer2"].notna()]
        disagree.sample(
            n=min(DISAGREEMENT_SAMPLE, len(disagree)), random_state=ctx.seed
        ).sort_values(["set", "question_id", "level"]).to_csv(
            out / f"agree_pilot_disagreements_{key}.csv", index=False
        )
        frame.to_csv(out / f"agree_pilot_{key}.csv", index=False)
        transport_failures = int((frame["status2"] == STATUS_TRANSPORT_FAILURE).sum())
        valid = float((frame["status2"] == STATUS_VALID).mean())
        positions = int(frame["value_argmax_positions"].sum())
        violations = int(frame["value_argmax_violations"].sum())
        slow = int((frame["prompt_eval_s"] > CPU_CLIP_PROMPT_EVAL_S).sum())
        results[key] = {
            "model": model_name,
            "digest": engine_dev.settings.digest,
            "calls": len(frame),
            "json_valid_rate_after_repair": valid,
            "json_initial_valid_rate": float(frame["initial_valid2"].mean()),
            "think_false_accepted": transport_failures == 0,
            "transport_failures": transport_failures,
            "value_argmax_positions": positions,
            "value_argmax_violations": violations,
            "all_positions_argmax_violations": int(frame["argmax_violations"].sum()),
            "greedy_verified": positions > 0 and violations == 0,
            "gpu": placement,
            "unloaded_before_run": unloaded,
            "prompt_eval_s_median": float(frame["prompt_eval_s"].median()),
            "prompt_eval_s_max": float(frame["prompt_eval_s"].max()),
            "cpu_clip_suspect_calls": slow,
            "determinism": det,
            "agreement_by_set": {
                k: float(v) for k, v in frame.groupby("set")["agree"].mean().items()
            },
            "agreement_by_set_level": {
                f"{s}@L{int(lv)}": float(v)
                for (s, lv), v in frame.groupby(["set", "level"])["agree"]
                .mean()
                .items()
            },
            "gates_pass": valid >= GATE_RATE
            and det["passed"]
            and transport_failures == 0
            and violations == 0
            and slow == 0,
        }
    summary = {
        "label": "plan §25.5 V6-A pilot (mechanics only; no accuracy/firing used)",
        "time": utc_now(),
        "levels": list(PILOT_LEVELS),
        "questions": {"dev102": len(dev_q), "wrong99": len(wrong_q)},
        "readers": results,
        "gates_pass": all(r["gates_pass"] for r in results.values()),
    }
    atomic_write_json(out / "agree_pilot_summary.json", summary)
    update_v6_config(
        ctx,
        {
            "second_readers": {
                key: {
                    "model": r["model"],
                    "digest": r["digest"],
                    "think_sent": False,
                    "think_false_accepted": r["think_false_accepted"],
                    "decoding": dict(ctx.settings.decoding_options),
                    "logprobs": TOP_LOGPROBS,
                    "record_dir": f"reader_{key}/",
                }
                for key, r in results.items()
            }
        },
        "V6-A pilot: second-reader settings",
    )
    ctx.log(f"§25 V6-A pilot: gates_pass={summary['gates_pass']}")
    return summary


# ---- dev run ----------------------------------------------------------------------------
def reader_table_path(ctx: Context, key: str, which: str) -> Path:
    if which == "dev102":
        return ctx.paths.out(OUT_DIR) / f"reader2_{key}_dev102.csv"
    return ctx.paths.out(EXT_DIR) / OUT_DIR / f"reader2_{key}_wrong99.csv"


def run_agree(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("v6_agree")
    summary_path = ctx.paths.out(OUT_DIR) / "agree_pilot_summary.json"
    require(summary_path.exists(), "run v6-agree-pilot first")
    require(bool(read_json(summary_path)["gates_pass"]), "V6-A pilot gates failed")
    sets = load_sets(ctx)
    levels = list(range(1, ctx.n_levels + 1))
    if ctx.dry_run:
        n = (len(sets.dev_qids) + len(sets.wrong_qids)) * len(levels)
        ctx.echo(f"[dry-run] V6-A dev: {n} calls per reader")
        return {}
    root = ctx.paths.output_root
    done = {}
    for key, model_name in SECOND_READERS.items():  # one model at a time
        isolate_model(ctx.cfg["model"]["endpoint"], model_name)
        warm_up(ctx.cfg["model"]["endpoint"], model_name)
        for which, qids, rows, base in (
            ("dev102", sets.dev_qids, sets.dev_rows, root),
            ("wrong99", sets.wrong_qids, sets.wrong_rows, root / EXT_DIR),
        ):
            engine = reader_engine(ctx, key, base)
            frame = run_reader(ctx, engine, key, [rows[q] for q in qids], levels)
            path = reader_table_path(ctx, key, which)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
            done[f"{key}/{which}"] = {
                "rows": len(frame),
                "valid_rate": float((frame["status2"] == STATUS_VALID).mean()),
                "value_argmax_violations": int(frame["value_argmax_violations"].sum()),
                "cpu_clip_suspect_calls": int(
                    (frame["prompt_eval_s"] > CPU_CLIP_PROMPT_EVAL_S).sum()
                ),
            }
    ctx.log(f"§25 V6-A dev readers done: {done}")
    return done


def load_reader2(ctx: Context, key: str, which: str) -> pd.DataFrame:
    return pd.read_csv(reader_table_path(ctx, key, which), dtype={"question_id": str})


# ---- evaluation (cache-only) ------------------------------------------------------------
BASES = ("direct_pans", "direct_v4", "checklist_v4_min", "pans_min")
PRIMARY_POLICY = "direct_pans+agree_gemma3"
FIGURE_POLICIES = (
    "direct_v4",
    "direct_pans",
    "direct_pans+agree_gemma3",
    "direct_pans+agree_qwen4b",
    "direct_v4+agree_gemma3",
    "checklist_v4_min",
    "checklist_v4_min+agree_gemma3",
    "pans_min+agree_gemma3",
)


def agree_policies() -> list[ev.Policy]:
    out: list[ev.Policy] = []
    for base in BASES:
        role = "comparator" if base in ("direct_pans", "direct_v4") else "logged"
        out.append(ev.Policy(base, base, role=role))
        out.append(ev.Policy(f"{base}_k2", base, k=2))
        for key in SECOND_READERS:
            gate = f"agree_{key}"
            name = f"{base}+{gate}"
            out.append(
                ev.Policy(
                    name,
                    base,
                    gate,
                    role="primary" if name == PRIMARY_POLICY else "logged",
                )
            )
            out.append(ev.Policy(f"{name}_k2", base, gate, k=2))
            out.append(
                ev.Policy(f"{name}_k2either", base, gate, k=2, gate_mode=ev.GATE_EITHER)
            )
    return out


def agree_comparators(policies: Sequence[ev.Policy]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in policies:
        if p.gate is None and p.k == 1:
            continue
        refs = [p.score]
        refs += ["direct_pans", "direct_v4"] if p.score.startswith("direct") else []
        refs += (
            ["checklist_v4_min", "pans_min"] if not p.score.startswith("direct") else []
        )
        out[p.name] = list(dict.fromkeys(r for r in refs if r != p.name))
    return out


def agreement_tables(
    ctx: Context, sets: V6Sets
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Level tables (102, 99) with agree_/answer_/correct_ columns for each reader."""
    dev, wrong = sets.dev_table.copy(), sets.wrong_table.copy()
    gt = ctx.gt(sets.dev_qids + sets.wrong_qids)
    raw = {}
    for key in SECOND_READERS:
        for which in ("dev102", "wrong99"):
            frame = load_reader2(ctx, key, which)
            require(
                bool((frame["status2"] == STATUS_VALID).all()),
                f"{key}/{which}: invalid second-reader records",
            )
            raw[f"{key}/{which}"] = frame
        dev = with_agreement(dev, raw[f"{key}/dev102"], key)
        wrong = with_agreement(wrong, raw[f"{key}/wrong99"], key)
        for table in (dev, wrong):
            table[f"correct_{key}"] = [
                is_correct(None if pd.isna(a) else str(a), gt[str(q)])
                for q, a in zip(
                    table["question_id"], table[f"answer_{key}"], strict=True
                )
            ]
    return dev, wrong, raw


def agreement_stats(
    dev: pd.DataFrame, wrong: pd.DataFrame, repeats: int, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for name, table in (("dev102", dev), ("wrong99", wrong)):
        for level, g in table.groupby("level"):
            row = {
                "set": name,
                "level": int(level),
                "primary_accuracy": float(g["reader_correct"].mean()),
            }
            for key in SECOND_READERS:
                row[f"agree_{key}"] = float(g[f"agree_{key}"].mean())
                row[f"accuracy_{key}"] = float(g[f"correct_{key}"].mean())
                row[f"agree_wrong_{key}"] = float(
                    (g[f"agree_{key}"] & ~g["reader_correct"]).mean()
                )
            rows.append(row)
    by_level = pd.DataFrame(rows)
    groups = {
        **dev.groupby("question_id")["group_id"].first().astype(str).to_dict(),
        **wrong.groupby("question_id")["group_id"].first().astype(str).to_dict(),
    }
    per_q = {
        key: (
            dev.groupby("question_id")[f"agree_{key}"].mean().to_dict(),
            wrong.groupby("question_id")[f"agree_{key}"].mean().to_dict(),
        )
        for key in SECOND_READERS
    }
    hyp: dict[str, Any] = {}
    for key, (a, b) in per_q.items():
        hyp[f"H-A1 {key}: mean agreement 102 - 99"] = {
            "mean_102": float(np.mean(list(a.values()))),
            "mean_99": float(np.mean(list(b.values()))),
            **union_bootstrap_diff(a, b, groups, repeats, seed),
        }
    (ga, gb), (qa, qb) = per_q["gemma3"], per_q["qwen4b"]
    hyp["H-A2: (102-99) gemma3 - (102-99) qwen4b"] = union_bootstrap_diff(
        {q: ga[q] - qa[q] for q in ga},
        {q: gb[q] - qb[q] for q in gb},
        groups,
        repeats,
        seed,
    )
    l20 = wrong[wrong["level"] == ev.N_LEVELS]
    for key in SECOND_READERS:
        same_wrong_any = (
            wrong.assign(x=wrong[f"agree_{key}"] & ~wrong["reader_correct"])
            .groupby("question_id")["x"]
            .any()
        )
        hyp[f"99 both readers agree on a wrong answer ({key})"] = {
            "at_L20": int((l20[f"agree_{key}"] & ~l20["reader_correct"]).sum()),
            "at_any_level": int(same_wrong_any.sum()),
            "n": int(l20["question_id"].nunique()),
        }
    return by_level, hyp


def agreement_figure(by_level: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, SECOND_READERS, strict=True):
        for name, style in (("dev102", "-"), ("wrong99", "--")):
            g = by_level[by_level["set"] == name]
            ax.plot(
                g["level"], g[f"agree_{key}"], "o" + style, ms=3, label=f"agree {name}"
            )
            ax.plot(
                g["level"],
                g[f"accuracy_{key}"],
                "s" + style,
                ms=3,
                alpha=0.6,
                label=f"{key} accuracy {name}",
            )
            ax.plot(
                g["level"],
                g["primary_accuracy"],
                style,
                color="gray",
                lw=0.8,
                label=f"primary accuracy {name}",
            )
        ax.set_title(f"second reader {SECOND_READERS[key]}", fontsize=9)
        ax.set_xlabel("level")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fired_same_wrong(
    selected: Mapping[str, Mapping[str, np.ndarray]],
    wrong_table: pd.DataFrame,
    wrong_arr: ev.SetArrays,
) -> dict[str, Any]:
    """At the 0.95 θ: on the 99, fires of each gated policy and how many of them
    stop with both readers on the same wrong answer (= every fire, by the gate,
    unless the primary is correct at that level)."""
    out = {}
    correct = wrong_arr.correct
    for key in SECOND_READERS:
        sel = selected.get(f"{PRIMARY_POLICY.replace('gemma3', key)}@0.95")
        if sel is None:
            continue
        fired = sel["fired"]
        lv = sel["fire_level_99"]
        wrong_at_fire = fired & ~correct[np.arange(wrong_arr.n), np.maximum(lv, 1) - 1]
        out[key] = {
            "fired": int(fired.sum()),
            "fired_same_wrong": int(wrong_at_fire.sum()),
        }
    return out


def evaluate_agree(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("v6_agree_eval")
    sets = load_sets(ctx)
    if ctx.dry_run:
        ctx.echo("[dry-run] V6-A evaluation (cache-only)")
        return {}
    out = ctx.paths.out(OUT_DIR)
    out_w = ctx.paths.out(EXT_DIR) / OUT_DIR
    out_w.mkdir(parents=True, exist_ok=True)
    e = ctx.cfg["evaluation"]
    reps, seed = int(e["bootstrap_repeats"]), int(e["bootstrap_seed"])
    dev_t, wrong_t, _ = agreement_tables(ctx, sets)
    dev_t.to_csv(out / "agree_level_table_dev102.csv", index=False)
    wrong_t.to_csv(out_w / "agree_level_table_wrong99.csv", index=False)
    by_level, hyp = agreement_stats(dev_t, wrong_t, reps, seed)
    by_level.to_csv(out / "agree_by_level.csv", index=False)
    agreement_figure(by_level, out / "agree_by_level.png")
    gates = [f"agree_{k}" for k in SECOND_READERS]
    cols = list(BASES) + gates
    dev, wrong = ev.build_arrays(dev_t, cols), ev.build_arrays(wrong_t, cols)
    policies = agree_policies()
    comps = agree_comparators(policies)
    boot = ev.make_bootstrap(dev, wrong, reps, seed)
    table, curves, fronts, sel = ev.insample(policies, dev, wrong, comps, boot)
    table.to_csv(out / "agree_targets.csv", index=False)
    curves.to_csv(out / "agree_curves.csv", index=False)
    fronts.to_csv(out / "agree_frontier.csv", index=False)
    ev.flatten_ci(pd.DataFrame(sel["pairs"])).to_csv(
        out / "agree_paired_insample.csv", index=False
    )
    oof, per_q = ev.out_of_fold(policies, dev, wrong)
    oof.to_csv(out / "agree_oof.csv", index=False)
    pairs = ev.oof_pairs(per_q, comps, boot)
    ev.flatten_ci(pairs).to_csv(out / "agree_paired_oof.csv", index=False)
    verdict = ev.pass_rule(oof, pairs, PRIMARY_POLICY, ["direct_pans"])
    names = [p.name for p in policies if p.k == 1]
    ev.breakdown(sel["selected"], names, dev, wrong).to_csv(
        out / "agree_breakdown.csv", index=False
    )
    ev.curves_figure(
        curves,
        FIGURE_POLICIES,
        str(out / "agree_curves.png"),
        "V6-A: agreement-gated policies (dev; in-sample θ grid)",
    )
    result = {
        "label": "plan §25 V6-A (dev; in-sample θ + image-grouped 5-fold x 10 seeds)",
        "time": utc_now(),
        "n_102": dev.n,
        "n_99": wrong.n,
        "hypotheses": hyp,
        "fired_same_wrong_at_0.95": fired_same_wrong(sel["selected"], wrong_t, wrong),
        "primary_verdict": verdict,
    }
    atomic_write_json(
        out / "agree_summary.json", json.loads(json.dumps(result, default=float))
    )
    ctx.log(f"§25 V6-A evaluated: primary passed={verdict['passed']}")
    return result
