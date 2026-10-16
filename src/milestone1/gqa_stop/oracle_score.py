"""Per-level scores of an oracle checklist plan (pure functions).

Input per item: the observer signal {valid, answer, resolved, pans} (as returned
by ``logprob_v4.item_signals``). Definitions:
- own gate (as in P): valid, resolved is True and answer != "unknown".
- gated score g_j: P_ans_j if item j and every item in its ``depends_on`` chain
  pass the own gate, else 0 (the P rule of v27).
- veto ``object_absent``: an existence item with ``required_answer == "yes"``
  that is valid, resolved and answers "no". The question score is then 0.
- existence score E: min of g over existence items (1 when there are none).
- final score F: the final items' g combined by their ``logic``: "and" -> min,
  "or" -> max, no logic (single final item) -> min.
- ``oracle_min`` (plain min, the current P rule): min of g over all items.
- ``oracle_logic``: 0 if vetoed, else min(E, F).
- ``oracle_logic_noveto``: min(E, F) without the veto (veto ablation).
- ``oracle_kleene`` (answer-aware logic, an extra variant): like ``oracle_logic``
  but an "or" group scores max g over its disjuncts answering "yes" if any,
  else min g over all disjuncts (every disjunct must be a confident "no");
  "and" is the dual (min over all if none answers "no", else max over the
  "no" conjuncts).
An empty plan (rules failed) scores 0 everywhere.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .dataset import normalize_text

UNKNOWN = "unknown"
YES = "yes"
NO = "no"
EXISTENCE = "existence"
FINAL = "final"
AND = "and"
OR = "or"
EMPTY_SCORE = 0.0
NO_EXISTENCE_SCORE = 1.0
SCORE_KEYS = (
    "oracle_min",
    "oracle_logic",
    "oracle_logic_noveto",
    "oracle_kleene",
    "oracle_existence",
    "oracle_final",
)

Signal = Mapping[str, Any]
Item = Mapping[str, Any]


def answer_of(signal: Signal | None) -> str:
    if signal is None or signal.get("answer") is None:
        return ""
    return normalize_text(str(signal["answer"]))


def own_ok(signal: Signal | None) -> bool:
    """valid, resolved is True and answer != "unknown"."""
    return bool(
        signal is not None
        and signal.get("valid")
        and signal.get("resolved") is True
        and answer_of(signal) not in ("", UNKNOWN)
    )


def gated_scores(
    items: Sequence[Item], signals: Mapping[str, Signal]
) -> dict[str, float]:
    """g_j = P_ans_j if j and all its ancestors pass the own gate, else 0.

    Items must be listed after the items they depend on (plan order)."""
    passed: dict[str, bool] = {}
    scores: dict[str, float] = {}
    for item in items:
        signal = signals.get(item["id"])
        ok = own_ok(signal) and all(passed.get(d, False) for d in item["depends_on"])
        passed[item["id"]] = ok
        scores[item["id"]] = float(signal["pans"]) if ok and signal else 0.0
    return scores


def object_absent(items: Sequence[Item], signals: Mapping[str, Signal]) -> bool:
    """A required-"yes" existence item answered a resolved "no"."""
    for item in items:
        if item["role"] != EXISTENCE or item.get("required_answer") != YES:
            continue
        signal = signals.get(item["id"])
        valid = signal is not None and signal.get("valid")
        if valid and signal.get("resolved") is True and answer_of(signal) == NO:
            return True
    return False


def group_logic(items: Sequence[Item]) -> str | None:
    logics = {item.get("logic") for item in items}
    if len(logics) != 1:
        raise ValueError(f"mixed logic in the final group: {sorted(map(str, logics))}")
    return logics.pop()


def combine(values: Sequence[float], logic: str | None) -> float:
    if not values:
        return EMPTY_SCORE
    return max(values) if logic == OR else min(values)


def kleene(
    items: Sequence[Item],
    signals: Mapping[str, Signal],
    gated: Mapping[str, float],
    logic: str | None,
) -> float:
    """Answer-aware combination of the final group (see module docstring)."""
    values = [gated[i["id"]] for i in items]
    if logic not in (AND, OR):
        return combine(values, logic)
    decisive = YES if logic == OR else NO
    hits = [
        gated[i["id"]]
        for i in items
        if gated[i["id"]] > 0 and answer_of(signals.get(i["id"])) == decisive
    ]
    return max(hits) if hits else min(values)


def score_level(items: Sequence[Item], signals: Mapping[str, Signal]) -> dict[str, Any]:
    """All oracle scores of one (question, level)."""
    if not items:
        return {**dict.fromkeys(SCORE_KEYS, EMPTY_SCORE), "object_absent": False}
    gated = gated_scores(items, signals)
    existence = [gated[i["id"]] for i in items if i["role"] == EXISTENCE]
    finals = [i for i in items if i["role"] == FINAL]
    logic = group_logic(finals) if finals else None
    e_score = min(existence) if existence else NO_EXISTENCE_SCORE
    f_score = combine([gated[i["id"]] for i in finals], logic)
    k_score = kleene(finals, signals, gated, logic) if finals else EMPTY_SCORE
    vetoed = object_absent(items, signals)
    noveto = min(e_score, f_score)
    return {
        "oracle_min": min(gated.values()),
        "oracle_logic": EMPTY_SCORE if vetoed else noveto,
        "oracle_logic_noveto": noveto,
        "oracle_kleene": EMPTY_SCORE if vetoed else min(e_score, k_score),
        "oracle_existence": e_score,
        "oracle_final": f_score,
        "object_absent": vetoed,
        "gated": gated,
    }
