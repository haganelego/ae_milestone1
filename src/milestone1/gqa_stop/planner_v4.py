"""planner_v4: type-aware text-only checklist planner (exploratory probe).

Same input as planner_v3 (the question string as data, no image) and the v3
item schema (``id``, ``question``, ``depends_on``) extended with
- ``role``: "existence" (an object the question presupposes) or "final"
  (the question's actual target, or one operand of an and/or question);
- ``required_answer``: "yes" for an existence item whose object must exist,
  else null (final items are always null);
- ``logic``: "and" / "or" on the operand items of a logical question, else null.

The extended schema is config gated (``planner_v4`` block). With
``fill_defaults`` true, missing fields get defaults and are listed in
``defaults_applied``: missing role -> the last item is final, the others are
existence; missing required_answer -> "yes" for existence, null for final;
missing logic -> null. With ``fill_defaults`` false a missing field is a schema
error. Contradictions are always errors (one schema repair): unknown role /
logic / required_answer values, logic on an existence item, a final item with a
required answer, no final item, mixed logic among the final items.

planner_v3 behaviour is untouched: this module only adds a new role.

Also here: the observer request for v4 items (observer_v3 prompt, own
namespace) and plan-quality helpers that compare a v4 plan with the oracle
plan (item count, oracle object coverage, final-item type).
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .client import STATUS_VALID, RequestSpec
from .dataset import normalize_text
from .logprob_v4 import observer_lp_spec
from .oracle_plan import (
    ProgramWalker,
    is_generic,
    noun_of,
    op_name,
    parse_relation,
    strip_ref,
)
from .sampled import item_not_standalone
from .schemas import validate_plan

PLANNER_V4_ROLE = "planner_v4"
OBSERVER_V4_NAMESPACE = "lp_v4"
EXISTENCE = "existence"
FINAL = "final"
YES = "yes"
LOGICS = ("and", "or")
ITEM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,16}$")
DEFAULT_MIN_ITEMS = 2
DEFAULT_MAX_ITEMS = 5
MISSING = object()
FieldResult = tuple[Any, list[str], bool]  # value, errors, default used


# ---- configuration ----------------------------------------------------------------------
@dataclass(frozen=True)
class PlanV4Settings:
    min_items: int = DEFAULT_MIN_ITEMS
    max_items: int = DEFAULT_MAX_ITEMS
    fill_defaults: bool = True


def plan_v4_settings(cfg: Mapping[str, Any]) -> PlanV4Settings:
    """``planner_v4`` config block; absent keys keep the defaults."""
    block = cfg.get("planner_v4") or {}
    return PlanV4Settings(
        int(block.get("min_items", DEFAULT_MIN_ITEMS)),
        int(block.get("max_items", DEFAULT_MAX_ITEMS)),
        bool(block.get("fill_defaults", True)),
    )


# ---- validation -------------------------------------------------------------------------
NULL_STRINGS = frozenset({"null", "none"})


def _lower(value: Any) -> Any:
    """Case-insensitive strings; the strings "null" / "none" mean JSON null."""
    if not isinstance(value, str):
        return value
    text = value.strip().lower()
    return None if text in NULL_STRINGS else text


def _field(raw: Mapping[str, Any], key: str) -> Any:
    """MISSING when the key is absent; explicit null stays None."""
    return _lower(raw[key]) if key in raw else MISSING


def _role(
    raw: Mapping[str, Any], is_last: bool, settings: PlanV4Settings, where: str
) -> FieldResult:
    role = _field(raw, "role")
    if role is MISSING or role is None:
        if not settings.fill_defaults:
            return None, [f"{where}: role is required"], False
        return (FINAL if is_last else EXISTENCE), [], True
    if role not in (EXISTENCE, FINAL):
        return None, [f"{where}: role must be 'existence' or 'final'"], False
    return role, [], False


def _logic(
    raw: Mapping[str, Any], role: str | None, settings: PlanV4Settings, where: str
) -> FieldResult:
    logic = _field(raw, "logic")
    if logic is MISSING:
        if not settings.fill_defaults:
            return None, [f"{where}: logic is required (null allowed)"], False
        return None, [], True
    if logic is not None and logic not in LOGICS:
        return None, [f"{where}: logic must be 'and', 'or' or null"], False
    if logic is not None and role == EXISTENCE:
        return None, [f"{where}: logic is only allowed on final items"], False
    return logic, [], False


def _required(
    raw: Mapping[str, Any], role: str | None, settings: PlanV4Settings, where: str
) -> FieldResult:
    value = _field(raw, "required_answer")
    if value is MISSING:
        if not settings.fill_defaults:
            return None, [f"{where}: required_answer is required (null allowed)"], False
        return (YES if role == EXISTENCE else None), [], True
    if value not in (YES, None):
        return None, [f"{where}: required_answer must be 'yes' or null"], False
    if value is not None and role == FINAL:
        return None, [f"{where}: a final item has required_answer null"], False
    return value, [], False


def validate_plan_v4(obj: Any, settings: PlanV4Settings) -> tuple[Any, list[str]]:
    """v3 item schema + role / required_answer / logic (see module docstring)."""
    parsed, errors = validate_plan(obj, settings.min_items, settings.max_items)
    if errors:
        return None, errors
    raw_items = obj["items"]
    items: list[dict[str, Any]] = []
    defaults: list[str] = []
    for index, (base, raw) in enumerate(zip(parsed["items"], raw_items, strict=True)):
        where = f"items[{index}]"
        if not ITEM_ID_RE.match(base["id"]):
            errors.append(f"{where}: id must match {ITEM_ID_RE.pattern}")
        role, e1, d1 = _role(raw, index == len(raw_items) - 1, settings, where)
        logic, e2, d2 = _logic(raw, role, settings, where)
        required, e3, d3 = _required(raw, role, settings, where)
        errors.extend([*e1, *e2, *e3])
        defaults.extend(
            f"{base['id']}.{name}"
            for name, used in (("role", d1), ("logic", d2), ("required_answer", d3))
            if used
        )
        items.append(
            {
                **base,
                "role": role,
                "required_answer": required,
                "logic": logic,
                "item_not_standalone": item_not_standalone(base["question"]),
            }
        )
    finals = [i for i in items if i["role"] == FINAL]
    if not errors and not finals:
        errors.append("plan has no final item")
    if not errors and len({i["logic"] for i in finals}) > 1:
        errors.append("final items mix different logic values")
    if errors:
        return None, errors
    return {"items": items, "defaults_applied": defaults}, []


# ---- request specs and records ------------------------------------------------------------
def planner_v4_spec(question: Mapping[str, Any]) -> RequestSpec:
    """Text-only: the question string is the only input (as in planner_v3)."""
    return RequestSpec(
        role=PLANNER_V4_ROLE,
        question_id=str(question["question_id"]),
        level=None,
        input_data={"question": str(question["question"])},
    )


def observer_v4_spec(
    question: Mapping[str, Any], level: int, image: bytes, item: Mapping[str, Any]
) -> RequestSpec:
    """observer_v3 (prompt unchanged) on one v4 item; own namespace ``lp_v4``."""
    spec = observer_lp_spec(question, level, image, item)
    return dataclasses.replace(spec, namespace=OBSERVER_V4_NAMESPACE)


def plan_v4_items(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Items of a valid planner_v4 record; [] on a final planner failure."""
    if record.get("status") != STATUS_VALID or record.get("parsed") is None:
        return []
    return [dict(i) for i in record["parsed"]["items"]]


# ---- plan quality (v4 vs oracle) ----------------------------------------------------------
QUERY_RE = re.compile(r"^(what|who|where|which|how)\b")
COMPARE_RE = re.compile(r"\b(same|different|differ)\b")
COMMON_RE = re.compile(r"\b(common)\b")
CHOICE_RE = re.compile(r"\bor\b")


def final_type(items: Sequence[Mapping[str, Any]]) -> str:
    """Type of the final group: logic_or / logic_and / compare / common / choose /
    query / verify / none (heuristic on the text of the final item)."""
    finals = [i for i in items if i["role"] == FINAL]
    if not finals:
        return "none"
    logic = finals[0].get("logic")
    if logic in LOGICS and len(finals) > 1:
        return f"logic_{logic}"
    text = normalize_text(str(finals[-1]["question"]))
    if COMMON_RE.search(text):
        return "common"
    if COMPARE_RE.search(text):
        return "compare"
    if CHOICE_RE.search(text):
        return "choose"
    if QUERY_RE.match(text):
        return "query"
    return "verify"


def head_word(phrase: str) -> str:
    """Singular-insensitive last word ('fire hydrant' -> 'hydrant')."""
    words = re.findall(r"[a-z]+", phrase.lower())
    return singular(words[-1]) if words else ""


def singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def item_words(text: str) -> set[str]:
    return {singular(w) for w in re.findall(r"[a-z]+", text.lower())}


@dataclass(frozen=True)
class OracleObject:
    """One oracle existence check: a named object, or a relation to a generic one."""

    noun: str  # head word of the named object ("" for a generic relation target)
    anchor: str  # head word of the anchor for relation checks ("" otherwise)
    relation: str  # relation words for relation checks ("" otherwise)

    @property
    def is_relation_check(self) -> bool:
        return not self.noun


def oracle_objects(
    plan: Mapping[str, Any], program: Sequence[Mapping[str, Any]]
) -> list[OracleObject]:
    """Objects behind the oracle existence items (via each item's source op)."""
    walker = ProgramWalker(program)
    out: list[OracleObject] = []
    for item in plan["items"]:
        if item["role"] != "existence":
            continue
        op = program[int(item["source_op"])]
        name, _ = op_name(op)
        if name == "choose":  # 'counter,below|above,s' -> the subject
            out.append(
                OracleObject(head_word(parse_relation(op["argument"])[0]), "", "")
            )
            continue
        if name not in ("select", "filter", "relate"):
            out.append(
                OracleObject(head_word(noun_of(strip_ref(op["argument"]))), "", "")
            )
            continue
        node = walker.node(int(item["source_op"]))
        if is_generic(node.target):
            anchor = walker.node(node.anchor) if node.anchor is not None else None
            anchor_noun = head_word(anchor.phrase.noun) if anchor else ""
            out.append(OracleObject("", anchor_noun, str(node.relation or "")))
        else:
            out.append(OracleObject(head_word(node.phrase.noun), "", ""))
    return out


def covered(obj: OracleObject, texts: Sequence[str]) -> bool:
    """Named object: its head word occurs in one of ``texts``. Relation check:
    one text holds the anchor and every relation word."""
    for text in texts:
        words = item_words(text)
        if obj.is_relation_check:
            rel = item_words(obj.relation)
            if obj.anchor in words and rel <= words:
                return True
        elif obj.noun in words:
            return True
    return False


def operand_heads(plan: Mapping[str, Any]) -> list[str]:
    """Head nouns of the oracle and/or operands ('Is there a white door ...')."""
    heads = []
    for item in plan["items"]:
        if item["role"] != FINAL or item.get("logic") not in LOGICS:
            continue
        match = re.match(
            r"^(?:is|are) there (?:an? )?(.+?) in the image\?$",
            normalize_text(str(item["question"])),
        )
        if match:
            heads.append(head_word(match.group(1)))
    return heads


def plan_quality(
    v4_items: Sequence[Mapping[str, Any]],
    oracle_plan: Mapping[str, Any],
    program: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Per-question comparison of a v4 plan with the oracle plan."""
    existence_texts = [str(i["question"]) for i in v4_items if i["role"] == EXISTENCE]
    final_texts = [str(i["question"]) for i in v4_items if i["role"] == FINAL]
    objects = oracle_objects(oracle_plan, program)
    named = [o for o in objects if not o.is_relation_check]
    relations = [o for o in objects if o.is_relation_check]
    operands = operand_heads(oracle_plan)
    nouns = {o.noun for o in named} | {o.anchor for o in relations}
    extra = [t for t in existence_texts if not item_words(t) & nouns]
    oracle_type = final_type(oracle_plan["items"])
    v4_type = final_type(v4_items)
    return {
        "v4_items": len(v4_items),
        "oracle_items": len(oracle_plan["items"]),
        "v4_existence": len(existence_texts),
        "oracle_existence": sum(
            1 for i in oracle_plan["items"] if i["role"] == EXISTENCE
        ),
        "objects": len(named),
        "objects_covered": sum(covered(o, existence_texts) for o in named),
        "relation_checks": len(relations),
        "relation_checks_covered": sum(covered(o, existence_texts) for o in relations),
        "operands": len(operands),
        "operands_covered": sum(
            covered(OracleObject(h, "", ""), final_texts) for h in operands
        ),
        "extra_existence": len(extra),
        "oracle_final_type": oracle_type,
        "v4_final_type": v4_type,
        "final_type_preserved": bool(v4_items) and v4_type == oracle_type,
        "v4_failed": not v4_items,
    }
