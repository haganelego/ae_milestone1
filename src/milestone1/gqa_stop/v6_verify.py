"""Plan §25.11 V6-C: answer- / object-conditioned Yes/No (Rel replacement).

(a) ``verify_v6`` (P(True)-style): question + proposed answer of the SAME
level's existing call (direct_lp answer for Direct; observer_v3 item answer for
Checklist, only for valid, resolved, non-"unknown" items), normalised with the
pipeline normalisation. (b) ``visible_v6``: question + the question's attribute
word (Direct only). Plain text, top-10 logprobs, num_predict 8, A13 decoding,
think false; signal = P(exact "Yes") of the first generated token (0 if
absent), as relevance_v4.

Team-lead addition (beyond §25.11, which has no order control): the pilot also
sends ``*_swap`` prompts ("Answer with No or Yes.") and applies the V6-B order
gate (mean |P(Yes) - P(Yes)_swap| <= 0.15) to each form.

Scores: direct_v6t = P_ans x P(True), direct_v6v = P_ans x P(Vis) (primaries);
P(True) alone and checklist_v6t_min (secondary); logged: P(True) x Rel
(direct / min), P(Vis) alone, P(True) min / mean, checklist_v6t_mean.
"""

from __future__ import annotations

import base64
import dataclasses
import json
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
from .evaluate import cluster_bootstrap_indices, is_correct, normalize_answer
from .inference import UNKNOWN_ANSWER
from .logprob_v4 import TOP_LOGPROBS, _accept_text, relevance_from_logprobs
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
    warm_up,
)
from .v6_legibility import gated_min_mean, item_columns, plans_for

VERIFY, VISIBLE = "verify_v6", "visible_v6"
SWAP = "_swap"
FORMS = (VERIFY, VISIBLE)
NUM_PREDICT = 8
DIRECT = "direct"
YES, NO = "Yes", "No"
GATE_RATE = 0.95
MIN_AUROC = 0.65
MAX_REL_SPEARMAN = 0.95
MAX_SWAP_MEAN_ABS_DIFF = 0.15
MAX_ITEMS = 3
PRIMARIES = {VERIFY: "direct_v6t", VISIBLE: "direct_v6v"}
REPEATS_PILOT = 2000


class TemplateEngine(RequestEngine):
    """Inline template with several ``{key}`` placeholders filled from input_data
    (the values are part of the cache key through input_data)."""

    def build_messages(self, spec: RequestSpec) -> list[dict[str, Any]]:
        content = self.prompts.texts[spec.role].strip()
        for key, value in spec.input_data.items():
            content = content.replace("{" + key + "}", str(value))
        if "{" in content and "}" in content.split("{", 1)[1]:
            raise ValueError(f"unfilled placeholder in {spec.role} prompt")
        message: dict[str, Any] = {"role": "user", "content": content}
        if spec.image_bytes is not None:
            message["images"] = [base64.b64encode(spec.image_bytes).decode("ascii")]
        return [message]


def yesno_spec(
    role: str,
    question_id: str,
    level: int,
    image: bytes,
    data: Mapping[str, str],
    target: str,
) -> RequestSpec:
    return RequestSpec(
        role=role,
        question_id=question_id,
        level=level,
        input_data=dict(data),
        image_bytes=image,
        suffix=f"_{target}",
        logprobs=TOP_LOGPROBS,
        raw_text=True,
    )


def engine_for(ctx: Context, root: Path, roles: Sequence[str]) -> TemplateEngine:
    ctx.verify_live_digest()
    texts = {r: (ctx.paths.prompts_dir / f"{r}.txt").read_text("utf-8") for r in roles}
    texts[REPAIR_PROMPT] = ctx.prompts.texts[REPAIR_PROMPT]
    settings = dataclasses.replace(
        ctx.settings,
        num_predict={**ctx.settings.num_predict, **dict.fromkeys(roles, NUM_PREDICT)},
    )
    engine = TemplateEngine(
        settings,
        PromptSet(texts),
        ctx.transport_factory(settings),
        root,
        monitor_from_config(ctx.cfg),
    )
    engine.stage = "v6_verify"
    return engine


def yes_signal(record: Mapping[str, Any]) -> dict[str, Any]:
    ok = record.get("status") == STATUS_VALID
    rel = relevance_from_logprobs(record.get("logprobs") if ok else None)
    text = str((record.get("parsed") or {}).get("text", "")) if ok else ""
    return {
        "status": record.get("status"),
        "text": text,
        "first_token": rel.first_token,
        "p_yes": rel.p_yes,
        "p_no": rel.p_no,
        "has_yes_no": rel.has_yes_or_no,
        "starts_yes_no": text.startswith((YES, NO)),
    }


# ---- targets ------------------------------------------------------------------------------
def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def item_called(row: Mapping[str, Any], index: int) -> bool:
    """§25.11: verify only valid, resolved, non-"unknown" observer items."""
    answer = row.get(f"item{index}_answer")
    return (
        bool(row.get(f"item{index}_valid"))
        and bool(row.get(f"item{index}_resolved"))
        and not is_missing(answer)
        and normalize_answer(str(answer)) != UNKNOWN_ANSWER
    )


def level_targets(
    row: Mapping[str, Any], plan: PlanOutcome, form: str, attribute: str
) -> list[tuple[str, dict[str, str]]]:
    """(target, input data) for one (question, level) table row."""
    question = str(row["question"])
    if form == VISIBLE:
        return [(DIRECT, {"question": question, "attribute": attribute})]
    out = []
    if not is_missing(row.get("direct_lp_answer")):
        answer = normalize_answer(str(row["direct_lp_answer"]))
        out.append((DIRECT, {"question": question, "answer": answer}))
    for index, item in enumerate(plan.items[:MAX_ITEMS], start=1):
        if not item_called(row, index):
            continue
        require(
            str(row[f"item{index}_question"]) == str(item["question"]),
            f"item question mismatch for {row['question_id']} item {index}",
        )
        answer = normalize_answer(str(row[f"item{index}_answer"]))
        out.append(
            (str(item["id"]), {"question": str(item["question"]), "answer": answer})
        )
    return out


def run_form(
    ctx: Context,
    engine: RequestEngine,
    table: pd.DataFrame,
    rows: Mapping[str, Mapping[str, Any]],
    plans: Mapping[str, PlanOutcome],
    levels: Sequence[int],
    role: str,
) -> pd.DataFrame:
    form = role.removesuffix(SWAP)
    sub = table[table["level"].isin(levels)].sort_values(["question_id", "level"])
    out = []
    for row in sub.to_dict("records"):
        qid, level = str(row["question_id"]), int(row["level"])
        q = rows[qid]
        image = ctx.series.bytes_for(str(q["image_id"]), level)
        for target, data in level_targets(row, plans[qid], form, str(q["attribute"])):
            record = engine.run(
                yesno_spec(role, qid, level, image, data, target), _accept_text
            )
            out.append(
                {
                    "question_id": qid,
                    "level": level,
                    "target": target,
                    "role": role,
                    "proposed": data.get("answer", data.get("attribute")),
                    **yes_signal(record),
                }
            )
    return pd.DataFrame(out)


# ---- pilot --------------------------------------------------------------------------------
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC (ties = 0.5); NaN when one class is empty."""
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float(
        (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
        / (len(pos) * len(neg))
    )


def auroc_ci(
    scores: np.ndarray, labels: np.ndarray, groups: Sequence[str], seed: int
) -> list[float]:
    draws = []
    for idx in cluster_bootstrap_indices(list(groups), REPEATS_PILOT, seed):
        value = auroc(scores[idx], labels[idx])
        if not np.isnan(value):
            draws.append(value)
    return [float(np.percentile(draws, p)) for p in ev.PERCENTILES]


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def rel_long(table: pd.DataFrame, plans: Mapping[str, PlanOutcome]) -> pd.DataFrame:
    rows = []
    for r in table.to_dict("records"):
        qid = str(r["question_id"])
        rows.append((qid, int(r["level"]), DIRECT, float(r["direct_rel"])))
        for i, item in enumerate(plans[qid].items[:MAX_ITEMS], start=1):
            rows.append(
                (qid, int(r["level"]), str(item["id"]), float(r[f"item{i}_rel"]))
            )
    return pd.DataFrame(rows, columns=["question_id", "level", "target", "rel"])


def form_checks(
    frame: pd.DataFrame,
    form: str,
    rel: pd.DataFrame,
    truth: pd.DataFrame,
    groups: Mapping[str, str],
    seed: int,
) -> dict[str, Any]:
    main = frame[frame["role"] == form]
    swap = frame[frame["role"] == form + SWAP]
    keys = ["set", "question_id", "level", "target"]
    paired = main.merge(swap[[*keys, "p_yes"]], on=keys, suffixes=("", "_swap"))
    with_rel = main.merge(rel, on=["question_id", "level", "target"], how="left")
    direct = main[main["target"] == DIRECT]

    def means(df: pd.DataFrame) -> dict[str, float]:
        return {
            f"{s}@L{int(lv)}": float(v)
            for (s, lv), v in df.groupby(["set", "level"])["p_yes"].mean().items()
        }

    dev_direct = direct[direct["set"] == "dev102"].groupby("level")["p_yes"].mean()
    checks: dict[str, Any] = {
        "n_calls": len(main),
        "n_calls_swap": len(swap),
        "starts_yes_no_rate": float(main["starts_yes_no"].mean()),
        "has_yes_no_rate": float(main["has_yes_no"].mean()),
        "starts_yes_no_rate_swap": float(swap["starts_yes_no"].mean()),
        "has_yes_no_rate_swap": float(swap["has_yes_no"].mean()),
        "valid_rate": float((frame["status"] == STATUS_VALID).mean()),
        "mean_p_yes_direct_by_set_level": means(direct),
        "mean_p_yes_items_by_set_level": means(main[main["target"] != DIRECT]),
        "rel_spearman": spearman(with_rel["p_yes"], with_rel["rel"]),
        "rel_spearman_direct": spearman(
            with_rel.loc[with_rel["target"] == DIRECT, "p_yes"],
            with_rel.loc[with_rel["target"] == DIRECT, "rel"],
        ),
        "rel_missing": int(with_rel["rel"].isna().sum()),
        "swap_mean_abs_diff": float(
            (paired["p_yes"] - paired["p_yes_swap"]).abs().mean()
        ),
        "swap_mean_diff": float((paired["p_yes_swap"] - paired["p_yes"]).mean()),
        "swap_spearman": spearman(paired["p_yes"], paired["p_yes_swap"]),
        "first_tokens": main["first_token"].value_counts().to_dict(),
    }
    gates = {
        "starts_yes_no>=0.95": checks["starts_yes_no_rate"] >= GATE_RATE,
        "has_yes_no>=0.95": checks["has_yes_no_rate"] >= GATE_RATE,
        "dev102_direct_L20>L1": float(dev_direct.get(20, 0.0))
        > float(dev_direct.get(1, 1.0)),
        "rel_spearman<0.95": checks["rel_spearman"] < MAX_REL_SPEARMAN,
        "swap_mean_abs_diff<=0.15 (team-lead)": checks["swap_mean_abs_diff"]
        <= MAX_SWAP_MEAN_ABS_DIFF,
    }
    if form == VERIFY:
        d = direct.merge(truth, on=["question_id", "level"])
        l20 = d[d["level"] == ev.N_LEVELS]
        lab, sc = l20["proposal_correct"].to_numpy(bool), l20["p_yes"].to_numpy(float)
        checks["auroc_L20"] = auroc(sc, lab)
        checks["auroc_L20_n_correct_wrong"] = [int(lab.sum()), int((~lab).sum())]
        checks["auroc_L20_ci"] = auroc_ci(
            sc, lab, [groups[q] for q in l20["question_id"]], seed
        )
        for name in ("dev102", "wrong99"):
            s = l20[l20["set"] == name]
            checks[f"auroc_L20_{name}"] = auroc(
                s["p_yes"].to_numpy(float), s["proposal_correct"].to_numpy(bool)
            )
        checks["auroc_all_levels"] = auroc(
            d["p_yes"].to_numpy(float), d["proposal_correct"].to_numpy(bool)
        )
        gates["auroc_L20>=0.65"] = checks["auroc_L20"] >= MIN_AUROC
    return {
        "checks": checks,
        "gates": gates,
        "gates_pass_plan": all(v for k, v in gates.items() if "team-lead" not in k),
        "gates_pass": all(gates.values()),
    }


def proposal_truth(ctx: Context, sets: V6Sets) -> pd.DataFrame:
    table = pd.concat([sets.dev_table, sets.wrong_table])
    gt = ctx.gt(sets.dev_qids + sets.wrong_qids)
    return pd.DataFrame(
        {
            "question_id": table["question_id"].astype(str),
            "level": table["level"].astype(int),
            "proposal_correct": [
                is_correct(None if is_missing(a) else str(a), gt[str(q)])
                for q, a in zip(
                    table["question_id"], table["direct_lp_answer"], strict=True
                )
            ],
        }
    )


def set_inputs(ctx: Context, sets: V6Sets, n: int | None = None) -> list[tuple]:
    root = ctx.paths.output_root
    out = []
    for name, qids, rows, table, base in (
        ("dev102", sets.dev_qids, sets.dev_rows, sets.dev_table, root),
        ("wrong99", sets.wrong_qids, sets.wrong_rows, sets.wrong_table, root / EXT_DIR),
    ):
        keep = qids[:n] if n else qids
        sub = table[table["question_id"].isin(keep)]
        out.append((name, keep, rows, sub, base))
    return out


def pilot(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("v6_verify_pilot")
    sets = load_sets(ctx)
    if ctx.dry_run:
        ctx.echo(f"[dry-run] V6-C pilot: 40 q x {PILOT_LEVELS}, 2 forms x 2 orders")
        return {}
    isolate_model(ctx.cfg["model"]["endpoint"], ctx.settings.name)
    warm_up(ctx.cfg["model"]["endpoint"], ctx.settings.name)
    roles = [r for f in FORMS for r in (f, f + SWAP)]
    frames, plans_all = [], {}
    for name, qids, rows, sub, base in set_inputs(ctx, sets, PILOT_QUESTIONS):
        plans = plans_for(ctx, base, [rows[q] for q in qids])
        plans_all.update(plans)
        engine = engine_for(ctx, base, roles)
        for role in roles:
            frames.append(
                run_form(ctx, engine, sub, rows, plans, PILOT_LEVELS, role).assign(
                    set=name
                )
            )
    frame = pd.concat(frames)
    out = ctx.paths.out(OUT_DIR)
    frame.to_csv(out / "verify_pilot.csv", index=False)
    pilot_tables = pd.concat([sets.dev_table, sets.wrong_table])
    pilot_tables = pilot_tables[pilot_tables["question_id"].isin(plans_all)]
    rel = rel_long(pilot_tables, plans_all)
    truth = proposal_truth(ctx, sets)
    groups = {
        **{q: str(sets.dev_rows[q]["group_id"]) for q in sets.dev_qids},
        **{q: str(sets.wrong_rows[q]["group_id"]) for q in sets.wrong_qids},
    }
    seed = int(ctx.cfg["evaluation"]["bootstrap_seed"])
    forms = {f: form_checks(frame, f, rel, truth, groups, seed) for f in FORMS}
    result = {
        "label": "plan §25.11 V6-C pilot (mechanics only; no accuracy/firing used)",
        "time": utc_now(),
        "forms": forms,
        "note": "swap gate added by team-lead (not in §25.11); gates_pass_plan "
        "excludes it, gates_pass includes it and decides",
    }
    atomic_write_json(
        out / "verify_pilot_summary.json", json.loads(json.dumps(result, default=float))
    )
    update_v6_config(
        ctx,
        {
            "v6c": {
                "prompts": {r: f"prompts/gqa_stop/{r}.txt" for r in roles},
                "num_predict": NUM_PREDICT,
                "top_logprobs": TOP_LOGPROBS,
                "signal": "P(exact 'Yes') in first-token top_logprobs",
                "forms_passed": [f for f in FORMS if forms[f]["gates_pass"]],
            }
        },
        "V6-C pilot",
    )
    ctx.log(
        "§25.11 V6-C pilot: "
        + ", ".join(f"{f} gates_pass={forms[f]['gates_pass']}" for f in FORMS)
    )
    return result


# ---- dev run ------------------------------------------------------------------------------
def passed_forms(ctx: Context) -> list[str]:
    path = ctx.paths.out(OUT_DIR) / "verify_pilot_summary.json"
    require(path.exists(), "run v6-verify-pilot first")
    forms = read_json(path)["forms"]
    return [f for f in FORMS if forms[f]["gates_pass"]]


def table_path(ctx: Context, form: str, which: str) -> Path:
    if which == "dev102":
        return ctx.paths.out(OUT_DIR) / f"{form}_levels_dev102.csv"
    return ctx.paths.out(EXT_DIR) / OUT_DIR / f"{form}_levels_wrong99.csv"


def run_verify(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("v6_verify")
    forms = passed_forms(ctx)
    require(bool(forms), "no V6-C form passed its pilot gates")
    sets = load_sets(ctx)
    levels = list(range(1, ctx.n_levels + 1))
    if ctx.dry_run:
        ctx.echo(f"[dry-run] V6-C dev: forms {forms}")
        return {}
    isolate_model(ctx.cfg["model"]["endpoint"], ctx.settings.name)
    warm_up(ctx.cfg["model"]["endpoint"], ctx.settings.name)
    done = {}
    for name, qids, rows, sub, base in set_inputs(ctx, sets):
        plans = plans_for(ctx, base, [rows[q] for q in qids])
        engine = engine_for(ctx, base, forms)
        for form in forms:
            frame = run_form(ctx, engine, sub, rows, plans, levels, form)
            path = table_path(ctx, form, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
            done[f"{form}/{name}"] = {
                "rows": len(frame),
                "valid_rate": float((frame["status"] == STATUS_VALID).mean()),
                "has_yes_no_rate": float(frame["has_yes_no"].mean()),
            }
    ctx.log(f"§25.11 V6-C dev done: {done}")
    return done


# ---- scores -------------------------------------------------------------------------------
def add_v6c_scores(
    table: pd.DataFrame,
    verify: pd.DataFrame | None,
    visible: pd.DataFrame | None,
    plans: Mapping[str, PlanOutcome],
) -> pd.DataFrame:
    """Level table + P(True)/P(Vis) + V6-C scores (uncalled targets -> P = 0)."""

    def lookup(frame: pd.DataFrame | None) -> dict[tuple[str, int, str], float]:
        if frame is None:
            return {}
        return {
            (str(r.question_id), int(r.level), str(r.target)): float(r.p_yes)
            for r in frame.itertuples()
        }

    pt, pv = lookup(verify), lookup(visible)
    rows = []
    for row in table.to_dict("records"):
        qid, level = str(row["question_id"]), int(row["level"])
        plan = plans[qid]
        ptrue = pt.get((qid, level, DIRECT), 0.0)
        pvis = pv.get((qid, level, DIRECT), 0.0)
        items = {}
        for index, item in enumerate(plan.items[:MAX_ITEMS], start=1):
            sig = item_columns(row, index)
            sig["ptrue"] = pt.get((qid, level, str(item["id"])), 0.0)
            sig["pans_ptrue"] = sig["pans"] * sig["ptrue"]
            sig["ptrue_rel"] = sig["ptrue"] * sig["rel"]
            items[str(item["id"])] = sig
        v6t_min, v6t_mean = gated_min_mean(plan, items, "pans_ptrue")
        pt_min, pt_mean = gated_min_mean(plan, items, "ptrue")
        ptrel_min, _ = gated_min_mean(plan, items, "ptrue_rel")
        rows.append(
            {
                **row,
                "ptrue_direct": ptrue,
                "pvis_direct": pvis,
                "direct_v6t": float(row["direct_pans"]) * ptrue,
                "direct_v6v": float(row["direct_pans"]) * pvis,
                "ptrue_rel_direct": ptrue * float(row["direct_rel"]),
                "checklist_v6t_min": v6t_min,
                "checklist_v6t_mean": v6t_mean,
                "ptrue_min": pt_min,
                "ptrue_mean": pt_mean,
                "ptrue_rel_min": ptrel_min,
            }
        )
    return pd.DataFrame(rows)


COMPARATORS = ("direct_v4", "direct_pans", "checklist_v4_min", "pans_min")
DIRECT_REFS = ["direct_pans", "direct_v4"]
CHECK_REFS = ["checklist_v4_min", "pans_min"]
FORM_POLICIES: dict[str, list[tuple[str, str, list[str]]]] = {
    VERIFY: [
        ("direct_v6t", "primary", DIRECT_REFS),
        ("ptrue_direct", "secondary", DIRECT_REFS),
        ("checklist_v6t_min", "secondary", CHECK_REFS),
        ("checklist_v6t_mean", "logged", CHECK_REFS),
        ("ptrue_rel_direct", "logged", DIRECT_REFS),
        ("ptrue_rel_min", "logged", CHECK_REFS),
        ("ptrue_min", "logged", CHECK_REFS),
        ("ptrue_mean", "logged", CHECK_REFS),
    ],
    VISIBLE: [
        ("direct_v6v", "primary", DIRECT_REFS),
        ("pvis_direct", "logged", DIRECT_REFS),
    ],
}


def policies_for(forms: Sequence[str]) -> tuple[list[ev.Policy], dict[str, list[str]]]:
    policies = [ev.Policy(c, c, role="comparator") for c in COMPARATORS]
    comps: dict[str, list[str]] = {}
    for form in forms:
        for name, role, refs in FORM_POLICIES[form]:
            policies.append(ev.Policy(name, name, role=role))
            comps[name] = refs
    return policies, comps


def by_level(
    dev: pd.DataFrame, wrong: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    rows = []
    for name, t in (("dev102", dev), ("wrong99", wrong)):
        for level, g in t.groupby("level"):
            rows.append(
                {
                    "set": name,
                    "level": int(level),
                    "primary_accuracy": float(g["reader_correct"].mean()),
                    **{c: float(g[c].mean()) for c in columns},
                }
            )
    return pd.DataFrame(rows)


def evaluate_verify(ctx: Context) -> dict[str, Any]:
    ctx.set_stage("v6_verify_eval")
    forms = passed_forms(ctx)
    require(bool(forms), "no V6-C form passed its pilot gates")
    sets = load_sets(ctx)
    if ctx.dry_run:
        ctx.echo("[dry-run] V6-C evaluation (cache-only)")
        return {}
    out = ctx.paths.out(OUT_DIR)
    out_w = ctx.paths.out(EXT_DIR) / OUT_DIR
    out_w.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name, qids, rows, sub, base in set_inputs(ctx, sets):
        plans = plans_for(ctx, base, [rows[q] for q in qids])
        frames = {
            f: pd.read_csv(
                table_path(ctx, f, name), dtype={"question_id": str, "target": str}
            )
            if f in forms
            else None
            for f in FORMS
        }
        tables[name] = add_v6c_scores(sub, frames[VERIFY], frames[VISIBLE], plans)
    dev_t, wrong_t = tables["dev102"], tables["wrong99"]
    dev_t.to_csv(out / "verify_level_table_dev102.csv", index=False)
    wrong_t.to_csv(out_w / "verify_level_table_wrong99.csv", index=False)
    policies, comps = policies_for(forms)
    cols = [p.score for p in policies]
    by_level(
        dev_t,
        wrong_t,
        ["ptrue_direct", "pvis_direct", "direct_rel", "direct_pans", *cols],
    ).to_csv(out / "verify_by_level.csv", index=False)
    e = ctx.cfg["evaluation"]
    reps, seed = int(e["bootstrap_repeats"]), int(e["bootstrap_seed"])
    dev, wrong = ev.build_arrays(dev_t, cols), ev.build_arrays(wrong_t, cols)
    boot = ev.make_bootstrap(dev, wrong, reps, seed)
    table, curves, fronts, sel = ev.insample(policies, dev, wrong, comps, boot)
    table.to_csv(out / "verify_targets.csv", index=False)
    curves.to_csv(out / "verify_curves.csv", index=False)
    fronts.to_csv(out / "verify_frontier.csv", index=False)
    ev.flatten_ci(pd.DataFrame(sel["pairs"])).to_csv(
        out / "verify_paired_insample.csv", index=False
    )
    oof, per_q = ev.out_of_fold(policies, dev, wrong)
    oof.to_csv(out / "verify_oof.csv", index=False)
    pairs = ev.oof_pairs(per_q, comps, boot)
    ev.flatten_ci(pairs).to_csv(out / "verify_paired_oof.csv", index=False)
    verdicts = {
        PRIMARIES[f]: ev.pass_rule(
            oof, pairs, PRIMARIES[f], DIRECT_REFS, ci_key="d_firing_ci975"
        )
        for f in forms
    }
    ev.breakdown(sel["selected"], [p.name for p in policies], dev, wrong).to_csv(
        out / "verify_breakdown.csv", index=False
    )
    ev.curves_figure(
        curves,
        [p.name for p in policies if p.role in ("comparator", "primary", "secondary")],
        str(out / "verify_curves.png"),
        "V6-C: answer/object-conditioned Yes/No (dev; in-sample θ grid)",
    )
    result = {
        "label": "plan §25.11 V6-C (dev; in-sample θ + image-grouped 5-fold x 10 seeds)",
        "time": utc_now(),
        "forms": forms,
        "primary_verdicts": verdicts,
        "ci_for_rule_i": "97.5% two-sided (Bonferroni over the two primaries)",
    }
    atomic_write_json(
        out / "verify_summary.json", json.loads(json.dumps(result, default=float))
    )
    ctx.log(
        "§25.11 V6-C evaluated: "
        + ", ".join(f"{k} passed={v['passed']}" for k, v in verdicts.items())
    )
    return result
