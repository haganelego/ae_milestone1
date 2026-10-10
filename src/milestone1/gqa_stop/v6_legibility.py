"""Plan §25.4 / §25.5 V6-B: "close enough, or get closer?" signal (Rel replacement).

Role ``legibility_v6`` (pilot control: ``legibility_v6_swap``, options in the
other order): one plain-text call per (question, level, target) with the image
and either the original question (target "direct") or a planner_v3 item
question (target = item id); top-10 logprobs, num_predict 8 (as relevance_v4),
A13 decoding, think false. Records live in ``legibility_v6*/`` under the run
root (dev 102) and under ``ext_dev_wrong/`` (99).

Leg = probability of the canonical first token of "Enough" among the first
generated token's top_logprobs (exact token; 0 if absent). The canonical
tokens are fixed from the pilot (argmax first token of replies that start
with "Enough" / "Closer") and recorded in ``v6/effective_config_v6.json``
before the dev run. No case or tokenisation merging.

Scores: direct_v6 = P_ans(direct_lp) x Leg(direct) (primary);
checklist_v6_min = min over items of P_ans_j x Leg_j under the v1 gate
(secondary); logged: Leg only (direct / min), P_ans x Leg x Rel (direct / min).
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import v6_eval as ev
from .client import (
    REPAIR_PROMPT,
    STATUS_VALID,
    PromptSet,
    RequestEngine,
    RequestSpec,
    monitor_from_config,
)
from .common import atomic_write_json, read_json, utc_now
from .context import Context, require
from .inference import aggregate, effective_confidences
from .logprob_v4 import TOP_LOGPROBS, LogprobRunner, _accept_text
from .planner import PlanOutcome
from .v6_common import (
    EXT_DIR,
    OUT_DIR,
    PILOT_LEVELS,
    PILOT_QUESTIONS,
    V6Sets,
    isolate_model,
    load_sets,
    update_v6_config,
    v6_config,
    warm_up,
)

ROLE = "legibility_v6"
SWAP_ROLE = "legibility_v6_swap"
NUM_PREDICT = 8
ENOUGH, CLOSER = "Enough", "Closer"
DIRECT = "direct"
GATE_RATE = 0.95
MIN_SWAP_SPEARMAN = 0.8
MAX_SWAP_MEAN_ABS_DIFF = 0.15
MAX_REL_SPEARMAN = 0.95
MAX_ITEMS = 3
PRIMARY_POLICY = "direct_v6"
SECONDARY_POLICY = "checklist_v6_min"


# ---- requests -----------------------------------------------------------------------------
def leg_spec(
    question_id: str, level: int, image: bytes, text: str, target: str, role: str = ROLE
) -> RequestSpec:
    return RequestSpec(
        role=role,
        question_id=question_id,
        level=level,
        input_data={"question": text},
        image_bytes=image,
        suffix=f"_{target}",
        logprobs=TOP_LOGPROBS,
        raw_text=True,
    )


def leg_engine(ctx: Context, root: Path, roles: Sequence[str]) -> RequestEngine:
    ctx.verify_live_digest()
    texts = {r: (ctx.paths.prompts_dir / f"{r}.txt").read_text("utf-8") for r in roles}
    texts[REPAIR_PROMPT] = ctx.prompts.texts[REPAIR_PROMPT]
    settings = dataclasses.replace(
        ctx.settings,
        num_predict={**ctx.settings.num_predict, **dict.fromkeys(roles, NUM_PREDICT)},
    )
    engine = RequestEngine(
        settings,
        PromptSet(texts),
        ctx.transport_factory(settings),
        root,
        monitor_from_config(ctx.cfg),
    )
    engine.stage = "v6_legibility"
    return engine


def plan_engine(ctx: Context, root: Path) -> RequestEngine:
    """Primary engine on ``root`` used cache-only for planner_v3 plans."""
    return RequestEngine(
        ctx.settings, ctx.prompts, ctx.transport_factory(ctx.settings), root
    )


def plans_for(
    ctx: Context, root: Path, questions: Sequence[Mapping[str, Any]]
) -> dict[str, PlanOutcome]:
    checklist = ctx.cfg["checklist"]
    runner = LogprobRunner(
        plan_engine(ctx, root),
        ctx.series,
        int(checklist["min_items"]),
        int(checklist["max_items"]),
        cache_only=True,
    )
    return {str(q["question_id"]): runner.plan(q)[0] for q in questions}


def targets_for(
    question: Mapping[str, Any], plan: PlanOutcome
) -> list[tuple[str, str]]:
    """(target, text): the original question, then each item question."""
    items = (
        [] if plan.failed else [(str(i["id"]), str(i["question"])) for i in plan.items]
    )
    return [(DIRECT, str(question["question"])), *items]


# ---- parsing --------------------------------------------------------------------------
def first_token_probs(logprobs: Sequence[Mapping[str, Any]] | None) -> dict[str, float]:
    if not logprobs:
        return {}
    first = logprobs[0]
    alternatives = first.get("top_logprobs") or [first]
    return {str(a.get("token")): math.exp(float(a["logprob"])) for a in alternatives}


def leg_signal(
    record: Mapping[str, Any], enough_token: str, closer_token: str
) -> dict[str, Any]:
    ok = record.get("status") == STATUS_VALID
    logprobs = record.get("logprobs") if ok else None
    probs = first_token_probs(logprobs)
    text = str((record.get("parsed") or {}).get("text", "")) if ok else ""
    return {
        "status": record.get("status"),
        "text": text,
        "first_token": str(logprobs[0].get("token")) if logprobs else None,
        "first_probs": json.dumps(probs, sort_keys=True),
        "leg": probs.get(enough_token, 0.0),
        "p_closer": probs.get(closer_token, 0.0),
        "has_option": enough_token in probs or closer_token in probs,
        "starts_with_option": text.startswith((ENOUGH, CLOSER)),
        "starts_with_option_ci": text.lower().startswith(
            (ENOUGH.lower(), CLOSER.lower())
        ),
    }


def run_targets(
    ctx: Context,
    engine: RequestEngine,
    questions: Sequence[Mapping[str, Any]],
    plans: Mapping[str, PlanOutcome],
    levels: Sequence[int],
    role: str,
    tokens: tuple[str, str],
) -> pd.DataFrame:
    rows = []
    for q in questions:
        qid = str(q["question_id"])
        for level in levels:
            image = ctx.series.bytes_for(str(q["image_id"]), level)
            for target, text in targets_for(q, plans[qid]):
                record = engine.run(
                    leg_spec(qid, level, image, text, target, role), _accept_text
                )
                rows.append(
                    {
                        "question_id": qid,
                        "level": level,
                        "target": target,
                        "role": role,
                        **leg_signal(record, *tokens),
                    }
                )
    return pd.DataFrame(rows)


def canonical_tokens(frame: pd.DataFrame) -> dict[str, Any]:
    """Most frequent first (argmax) token of replies starting with each option."""
    out: dict[str, Any] = {}
    for option in (ENOUGH, CLOSER):
        counts = Counter(
            frame.loc[frame["text"].str.startswith(option), "first_token"].dropna()
        )
        out[option] = {
            "token": counts.most_common(1)[0][0] if counts else option,
            "counts": dict(counts),
        }
    return out


def rel_lookup(table: pd.DataFrame) -> pd.DataFrame:
    """Long (question_id, level, target, rel) from a v4 level table."""
    parts = [
        table[["question_id", "level", "direct_rel"]]
        .rename(columns={"direct_rel": "rel"})
        .assign(target=DIRECT)
    ]
    for i in range(1, MAX_ITEMS + 1):
        parts.append(
            table[["question_id", "level", f"item{i}_rel"]]
            .rename(columns={f"item{i}_rel": "rel"})
            .assign(item_index=i)
        )
    return pd.concat(parts)


def with_item_ids(rel: pd.DataFrame, plans: Mapping[str, PlanOutcome]) -> pd.DataFrame:
    ids = {
        (qid, idx + 1): str(item["id"])
        for qid, plan in plans.items()
        for idx, item in enumerate(plan.items)
    }
    out = rel.copy()
    mask = out["target"].isna()
    out.loc[mask, "target"] = [
        ids.get((q, int(i)))
        for q, i in zip(
            out.loc[mask, "question_id"], out.loc[mask, "item_index"], strict=True
        )
    ]
    return out.dropna(subset=["target"])[["question_id", "level", "target", "rel"]]


# ---- pilot -----------------------------------------------------------------------------
def pilot(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("v6_leg_pilot")
    sets = load_sets(ctx)
    dev_q = [sets.dev_rows[q] for q in sets.dev_qids[:PILOT_QUESTIONS]]
    wrong_q = [sets.wrong_rows[q] for q in sets.wrong_qids[:PILOT_QUESTIONS]]
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] V6-B pilot: {len(dev_q) + len(wrong_q)} q x {PILOT_LEVELS}"
        )
        return {}
    root = ctx.paths.output_root
    isolate_model(ctx.cfg["model"]["endpoint"], ctx.settings.name)
    warm_up(ctx.cfg["model"]["endpoint"], ctx.settings.name)
    provisional = (ENOUGH, CLOSER)
    frames = []
    for name, qs, base in (
        ("dev102", dev_q, root),
        ("wrong99", wrong_q, root / EXT_DIR),
    ):
        plans = plans_for(ctx, base, qs)
        engine = leg_engine(ctx, base, (ROLE, SWAP_ROLE))
        for role in (ROLE, SWAP_ROLE):
            frames.append(
                run_targets(
                    ctx, engine, qs, plans, PILOT_LEVELS, role, provisional
                ).assign(set=name)
            )
    raw = pd.concat(frames)
    tokens = canonical_tokens(raw[raw["role"] == ROLE])
    enough, closer = tokens[ENOUGH]["token"], tokens[CLOSER]["token"]
    frame = reparse(raw, (enough, closer))
    result = pilot_checks(ctx, sets, frame, (enough, closer))
    result["canonical_tokens"] = tokens
    out = ctx.paths.out(OUT_DIR)
    frame.to_csv(out / "leg_pilot.csv", index=False)
    pilot_figure(frame, out / "leg_pilot_order.png")
    atomic_write_json(
        out / "leg_pilot_summary.json", json.loads(json.dumps(result, default=float))
    )
    if result["gates_pass"]:
        update_v6_config(
            ctx,
            {
                "legibility": {
                    "prompt": f"prompts/gqa_stop/{ROLE}.txt",
                    "prompt_swap": f"prompts/gqa_stop/{SWAP_ROLE}.txt",
                    "num_predict": NUM_PREDICT,
                    "top_logprobs": TOP_LOGPROBS,
                    "canonical_enough_token": enough,
                    "canonical_closer_token": closer,
                    "fixed_from": "V6-B pilot (argmax first token of replies starting with the option)",
                }
            },
            "V6-B pilot: canonical first tokens fixed",
        )
    ctx.log(
        f"§25 V6-B pilot: gates_pass={result['gates_pass']} tokens={enough!r}/{closer!r}"
    )
    return result


def pilot_figure(frame: pd.DataFrame, path: Path) -> None:
    """Mean Leg by level: both option orders, direct vs items, 102 vs 99."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    kinds = frame["target"].eq(DIRECT).map({True: "direct", False: "items"})
    for ax, kind in zip(axes, ("direct", "items"), strict=True):
        sub = frame[kinds == kind]
        for (role, name), g in sub.groupby(["role", "set"]):
            means = g.groupby("level")["leg"].mean()
            style = "-" if role == ROLE else "--"
            ax.plot(
                means.index, means.values, "o" + style, ms=3, label=f"{role} {name}"
            )
        ax.set_title(f"P(Enough), {kind}", fontsize=9)
        ax.set_xlabel("level")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6)
    axes[0].set_ylabel("mean Leg (pilot 20+20 q)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def reparse(frame: pd.DataFrame, tokens: tuple[str, str]) -> pd.DataFrame:
    """Recompute Leg / P(Closer) / has_option from the stored first-token probs."""
    enough, closer = tokens
    probs = frame["first_probs"].map(json.loads)
    return frame.assign(
        leg=probs.map(lambda p: p.get(enough, 0.0)),
        p_closer=probs.map(lambda p: p.get(closer, 0.0)),
        has_option=probs.map(lambda p: enough in p or closer in p),
    )


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def pilot_checks(
    ctx: Context, sets: V6Sets, frame: pd.DataFrame, tokens: tuple[str, str]
) -> dict[str, Any]:
    main = frame[frame["role"] == ROLE]
    swap = frame[frame["role"] == SWAP_ROLE]
    keys = ["set", "question_id", "level", "target"]
    paired = main.merge(swap[[*keys, "leg"]], on=keys, suffixes=("", "_swap"))
    root = ctx.paths.output_root
    plans = {
        **plans_for(ctx, root, [sets.dev_rows[q] for q in sets.dev_qids]),
        **plans_for(ctx, root / EXT_DIR, [sets.wrong_rows[q] for q in sets.wrong_qids]),
    }
    rel = with_item_ids(
        pd.concat([rel_lookup(sets.dev_table), rel_lookup(sets.wrong_table)]), plans
    )
    with_rel = main.merge(rel, on=["question_id", "level", "target"], how="left")
    dev_main = main[main["set"] == "dev102"]

    def level_means(df: pd.DataFrame) -> dict[int, float]:
        return {int(k): float(v) for k, v in df.groupby("level")["leg"].mean().items()}

    direct_means = level_means(dev_main[dev_main["target"] == DIRECT])
    item_means = level_means(dev_main[dev_main["target"] != DIRECT])
    checks = {
        "starts_with_option_rate": float(main["starts_with_option"].mean()),
        "starts_with_option_rate_swap": float(swap["starts_with_option"].mean()),
        "has_option_rate": float(main["has_option"].mean()),
        "has_option_rate_swap": float(swap["has_option"].mean()),
        "valid_rate": float((frame["status"] == STATUS_VALID).mean()),
        "leg_by_level_dev102_direct": direct_means,
        "leg_by_level_dev102_items": item_means,
        "leg_by_level_wrong99_direct": level_means(
            main[(main["set"] == "wrong99") & (main["target"] == DIRECT)]
        ),
        "swap_spearman": spearman(paired["leg"], paired["leg_swap"]),
        "swap_mean_abs_diff": float((paired["leg"] - paired["leg_swap"]).abs().mean()),
        "rel_spearman": spearman(with_rel["leg"], with_rel["rel"]),
        "rel_spearman_direct": spearman(
            with_rel.loc[with_rel["target"] == DIRECT, "leg"],
            with_rel.loc[with_rel["target"] == DIRECT, "rel"],
        ),
        "rel_missing": int(with_rel["rel"].isna().sum()),
        "first_tokens": dict(Counter(main["first_token"].dropna())),
        "first_tokens_swap": dict(Counter(swap["first_token"].dropna())),
        "n_calls": len(frame),
    }
    gates = {
        "starts_with_option>=0.95": checks["starts_with_option_rate"] >= GATE_RATE,
        "has_option>=0.95": checks["has_option_rate"] >= GATE_RATE,
        "dev102_direct_L20>L1": direct_means.get(20, 0.0) > direct_means.get(1, 1.0),
        "dev102_items_L20>L1": item_means.get(20, 0.0) > item_means.get(1, 1.0),
        "swap_spearman>=0.8": checks["swap_spearman"] >= MIN_SWAP_SPEARMAN,
        "swap_mean_abs_diff<=0.15": checks["swap_mean_abs_diff"]
        <= MAX_SWAP_MEAN_ABS_DIFF,
        "rel_spearman<0.95": checks["rel_spearman"] < MAX_REL_SPEARMAN,
    }
    return {
        "label": "plan §25.5 V6-B pilot (mechanics only; no accuracy/firing used)",
        "time": utc_now(),
        "tokens": list(tokens),
        "checks": checks,
        "gates": gates,
        "gates_pass": all(gates.values()),
    }


# ---- dev run ------------------------------------------------------------------------------
def fixed_tokens(ctx: Context) -> tuple[str, str]:
    leg = v6_config(ctx).get("settings", {}).get("legibility")
    require(leg is not None, "V6-B canonical tokens not fixed: run v6-leg-pilot first")
    return str(leg["canonical_enough_token"]), str(leg["canonical_closer_token"])


def leg_table_path(ctx: Context, which: str) -> Path:
    if which == "dev102":
        return ctx.paths.out(OUT_DIR) / "leg_levels_dev102.csv"
    return ctx.paths.out(EXT_DIR) / OUT_DIR / "leg_levels_wrong99.csv"


def run_leg(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("v6_leg")
    summary = ctx.paths.out(OUT_DIR) / "leg_pilot_summary.json"
    require(summary.exists(), "run v6-leg-pilot first")
    require(bool(read_json(summary)["gates_pass"]), "V6-B pilot gates failed")
    tokens = fixed_tokens(ctx)
    sets = load_sets(ctx)
    levels = list(range(1, ctx.n_levels + 1))
    if ctx.dry_run:
        ctx.echo(
            f"[dry-run] V6-B dev: {len(sets.dev_qids) + len(sets.wrong_qids)} q x 20"
        )
        return {}
    isolate_model(ctx.cfg["model"]["endpoint"], ctx.settings.name)
    warm_up(ctx.cfg["model"]["endpoint"], ctx.settings.name)
    root = ctx.paths.output_root
    done = {}
    for which, qids, rows, base in (
        ("dev102", sets.dev_qids, sets.dev_rows, root),
        ("wrong99", sets.wrong_qids, sets.wrong_rows, root / EXT_DIR),
    ):
        qs = [rows[q] for q in qids]
        plans = plans_for(ctx, base, qs)
        frame = run_targets(
            ctx, leg_engine(ctx, base, (ROLE,)), qs, plans, levels, ROLE, tokens
        )
        path = leg_table_path(ctx, which)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        done[which] = {
            "rows": len(frame),
            "valid_rate": float((frame["status"] == STATUS_VALID).mean()),
            "has_option_rate": float(frame["has_option"].mean()),
        }
    ctx.log(f"§25 V6-B dev legibility done: {done}")
    return done


# ---- scores -------------------------------------------------------------------------------
def item_columns(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    return {
        "valid": bool(row[f"item{index}_valid"]),
        "answer": row[f"item{index}_answer"],
        "resolved": bool(row[f"item{index}_resolved"]),
        "pans": float(row[f"item{index}_pans"]),
        "rel": float(row[f"item{index}_rel"]),
    }


def gated_min_mean(
    plan: PlanOutcome, items: Mapping[str, Mapping[str, Any]], key: str
) -> tuple[float, float]:
    """v1 gate (unresolved / unresolved ancestor -> 0) with confidence := items[key]."""
    if plan.failed:
        return aggregate({})
    observations = [
        {
            "id": i,
            "answer": ""
            if s["answer"] is None
            or (isinstance(s["answer"], float) and np.isnan(s["answer"]))
            else s["answer"],
            "resolved": s["resolved"],
            "confidence": s[key],
        }
        for i, s in items.items()
        if s["valid"]
    ]
    return aggregate(effective_confidences(plan.items, observations))


def add_v6_scores(
    table: pd.DataFrame, leg: pd.DataFrame, plans: Mapping[str, PlanOutcome]
) -> pd.DataFrame:
    """Level table + Leg columns + v6 scores (and recomputed v4 checks)."""
    lookup = {
        (r.question_id, int(r.level), str(r.target)): float(r.leg)
        for r in leg.itertuples()
    }
    rows = []
    for row in table.to_dict("records"):
        qid, level = str(row["question_id"]), int(row["level"])
        plan = plans[qid]
        leg_direct = lookup[(qid, level, DIRECT)]
        items = {}
        for index, item in enumerate(plan.items, start=1):
            sig = item_columns(row, index)
            sig["leg"] = lookup[(qid, level, str(item["id"]))]
            sig["pans_leg"] = sig["pans"] * sig["leg"]
            sig["pans_leg_rel"] = sig["pans_leg"] * sig["rel"]
            sig["product_v4"] = sig["pans"] * sig["rel"]
            items[str(item["id"])] = sig
        v6_min, v6_mean = gated_min_mean(plan, items, "pans_leg")
        leg_min, _ = gated_min_mean(plan, items, "leg")
        v6rel_min, _ = gated_min_mean(plan, items, "pans_leg_rel")
        v4_min, _ = gated_min_mean(plan, items, "product_v4")
        rows.append(
            {
                **row,
                "leg_direct": leg_direct,
                "direct_v6": float(row["direct_pans"]) * leg_direct,
                "direct_v6rel": float(row["direct_pans"])
                * leg_direct
                * float(row["direct_rel"]),
                "checklist_v6_min": v6_min,
                "checklist_v6_mean": v6_mean,
                "leg_min": leg_min,
                "checklist_v6rel_min": v6rel_min,
                "checklist_v4_min_recomputed": v4_min,
            }
        )
    return pd.DataFrame(rows)


# ---- evaluation ---------------------------------------------------------------------------
def leg_policies() -> list[ev.Policy]:
    return [
        ev.Policy("direct_v4", "direct_v4", role="comparator"),
        ev.Policy("direct_pans", "direct_pans", role="comparator"),
        ev.Policy("checklist_v4_min", "checklist_v4_min", role="comparator"),
        ev.Policy("pans_min", "pans_min", role="comparator"),
        ev.Policy(PRIMARY_POLICY, "direct_v6", role="primary"),
        ev.Policy(SECONDARY_POLICY, "checklist_v6_min", role="secondary"),
        ev.Policy("checklist_v6_mean", "checklist_v6_mean"),
        ev.Policy("leg_direct", "leg_direct"),
        ev.Policy("leg_min", "leg_min"),
        ev.Policy("direct_v6rel", "direct_v6rel"),
        ev.Policy("checklist_v6rel_min", "checklist_v6rel_min"),
    ]


LEG_COMPARATORS = {
    PRIMARY_POLICY: ["direct_pans", "direct_v4"],
    SECONDARY_POLICY: ["checklist_v4_min", "pans_min"],
    "checklist_v6_mean": ["checklist_v4_min", "pans_min"],
    "leg_direct": ["direct_pans", "direct_v4"],
    "leg_min": ["checklist_v4_min", "pans_min"],
    "direct_v6rel": ["direct_pans", "direct_v4"],
    "checklist_v6rel_min": ["checklist_v4_min", "pans_min"],
}
SCORE_COLUMNS = (
    "direct_v4",
    "direct_pans",
    "checklist_v4_min",
    "pans_min",
    "direct_v6",
    "checklist_v6_min",
    "checklist_v6_mean",
    "leg_direct",
    "leg_min",
    "direct_v6rel",
    "checklist_v6rel_min",
)


def leg_level_stats(dev: pd.DataFrame, wrong: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, t in (("dev102", dev), ("wrong99", wrong)):
        for level, g in t.groupby("level"):
            rows.append(
                {
                    "set": name,
                    "level": int(level),
                    "leg_direct": float(g["leg_direct"].mean()),
                    "rel_direct": float(g["direct_rel"].mean()),
                    "pans_direct": float(g["direct_pans"].mean()),
                    "direct_v6": float(g["direct_v6"].mean()),
                    "direct_v4": float(g["direct_v4"].mean()),
                    "leg_min": float(g["leg_min"].mean()),
                    "checklist_v6_min": float(g["checklist_v6_min"].mean()),
                    "primary_accuracy": float(g["reader_correct"].mean()),
                }
            )
    return pd.DataFrame(rows)


def evaluate_leg(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("v6_leg_eval")
    sets = load_sets(ctx)
    if ctx.dry_run:
        ctx.echo("[dry-run] V6-B evaluation (cache-only)")
        return {}
    root = ctx.paths.output_root
    out = ctx.paths.out(OUT_DIR)
    out_w = ctx.paths.out(EXT_DIR) / OUT_DIR
    out_w.mkdir(parents=True, exist_ok=True)
    plans_dev = plans_for(ctx, root, [sets.dev_rows[q] for q in sets.dev_qids])
    plans_wrong = plans_for(
        ctx, root / EXT_DIR, [sets.wrong_rows[q] for q in sets.wrong_qids]
    )
    leg_dev = pd.read_csv(
        leg_table_path(ctx, "dev102"), dtype={"question_id": str, "target": str}
    )
    leg_wrong = pd.read_csv(
        leg_table_path(ctx, "wrong99"), dtype={"question_id": str, "target": str}
    )
    dev_t = add_v6_scores(sets.dev_table, leg_dev, plans_dev)
    wrong_t = add_v6_scores(sets.wrong_table, leg_wrong, plans_wrong)
    recompute_gap = float(
        max(
            (dev_t["checklist_v4_min"] - dev_t["checklist_v4_min_recomputed"])
            .abs()
            .max(),
            (wrong_t["checklist_v4_min"] - wrong_t["checklist_v4_min_recomputed"])
            .abs()
            .max(),
        )
    )
    require(
        recompute_gap < 1e-9,
        f"v1-gate recompute differs from the table: {recompute_gap}",
    )
    dev_t.to_csv(out / "leg_level_table_dev102.csv", index=False)
    wrong_t.to_csv(out_w / "leg_level_table_wrong99.csv", index=False)
    stats = leg_level_stats(dev_t, wrong_t)
    stats.to_csv(out / "leg_by_level.csv", index=False)
    e = ctx.cfg["evaluation"]
    reps, seed = int(e["bootstrap_repeats"]), int(e["bootstrap_seed"])
    dev, wrong = (
        ev.build_arrays(dev_t, SCORE_COLUMNS),
        ev.build_arrays(wrong_t, SCORE_COLUMNS),
    )
    policies = leg_policies()
    boot = ev.make_bootstrap(dev, wrong, reps, seed)
    table, curves, fronts, sel = ev.insample(
        policies, dev, wrong, LEG_COMPARATORS, boot
    )
    table.to_csv(out / "leg_targets.csv", index=False)
    curves.to_csv(out / "leg_curves.csv", index=False)
    fronts.to_csv(out / "leg_frontier.csv", index=False)
    ev.flatten_ci(pd.DataFrame(sel["pairs"])).to_csv(
        out / "leg_paired_insample.csv", index=False
    )
    oof, per_q = ev.out_of_fold(policies, dev, wrong)
    oof.to_csv(out / "leg_oof.csv", index=False)
    pairs = ev.oof_pairs(per_q, LEG_COMPARATORS, boot)
    ev.flatten_ci(pairs).to_csv(out / "leg_paired_oof.csv", index=False)
    verdict = ev.pass_rule(oof, pairs, PRIMARY_POLICY, ["direct_pans", "direct_v4"])
    secondary = ev.pass_rule(
        oof,
        pairs,
        SECONDARY_POLICY,
        ["checklist_v4_min", "pans_min"],
        pans_ref="direct_pans",
    )
    ev.breakdown(sel["selected"], [p.name for p in policies], dev, wrong).to_csv(
        out / "leg_breakdown.csv", index=False
    )
    ev.curves_figure(
        curves,
        [
            "direct_v4",
            "direct_pans",
            PRIMARY_POLICY,
            "leg_direct",
            "checklist_v4_min",
            "pans_min",
            SECONDARY_POLICY,
        ],
        str(out / "leg_curves.png"),
        "V6-B: legibility (Enough/Closer) policies (dev; in-sample θ grid)",
    )
    result = {
        "label": "plan §25 V6-B (dev; in-sample θ + image-grouped 5-fold x 10 seeds)",
        "time": utc_now(),
        "tokens": list(fixed_tokens(ctx)),
        "v1_gate_recompute_max_abs_diff": recompute_gap,
        "primary_verdict": verdict,
        "secondary_verdict_logged": secondary,
    }
    atomic_write_json(
        out / "leg_summary.json", json.loads(json.dumps(result, default=float))
    )
    ctx.log(f"§25 V6-B evaluated: primary passed={verdict['passed']}")
    return result
