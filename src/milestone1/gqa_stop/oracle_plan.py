"""Oracle checklist plans built mechanically from a GQA ``semantic`` program.

Exploratory probe (not pre-registered). The plan replaces planner_v3 with a
checklist whose structure follows the question's program, so that the probe can
ask "does decomposition help when the plan is correct and the aggregation
matches the question logic?".

Rules (rules version ``oracle-v1``):
- Object nodes. ``select`` starts a node; ``filter <type>`` attaches its value to
  the node's noun phrase (color / material / untyped -> adjective before the
  noun; hposition -> "on the left"; vposition -> "at the top"); ``relate
  name,rel,dir`` creates a new node related to its input node (the anchor).
  ``dir == "s"``: the new node is the subject ("the bird on the rock"); ``dir ==
  "o"``: it is the object ("the cabinet that the microwave is to the left of").
  A relate name "_" is an unknown target (phrased "something"); "person" ->
  "someone" / "who", "place" -> "some place" / "where".
- Presupposition. A node referenced by the final operation (query, verify,
  choose, same / different / common) is presupposed, and so is every anchor of a
  presupposed or asserted node. Each presupposed node gets one existence item
  ``role = "existence"`` with ``required_answer = "yes"``. The node consumed by
  an ``exist`` op is asserted, not presupposed: it becomes the final item. The
  named subject of ``choose rel`` is presupposed too.
- Final item(s), ``role = "final"``, ``required_answer = None``:
  query name -> "What is the man wearing?" / "What is on the wire?";
  query attr -> "What color is the X?"; verify attr -> "Is the X red?";
  verify rel -> "Is the X <rel> a Y?"; exist -> "Is there a X?";
  choose attr -> "Is the X a or b?"; choose rel -> "Is the X a or b the Y?";
  same / different attr -> one yes/no item over the two phrases;
  common -> "What is common to the X and the Y?";
  and / or -> one final item per operand, all carrying ``logic``.
- ``depends_on``: an existence item depends on its anchor's existence item; a
  final item depends on the existence items of the nodes it names.
- Leakage. Object ids and the "(-)" non-existence marker are stripped and never
  used. A relate target name is used only when it is not "_" (it is then named
  in the question). Choose options are ordered as they appear in the question
  text (GQA lists the correct option first in the program). ``answer`` and
  ``fullAnswer`` are never read; pronouns ("he") map to a fixed noun ("man").
Anything outside these rules is flagged and gives an empty plan (score 0).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

RULES_VERSION = "oracle-v1"
EXISTENCE = "existence"
FINAL = "final"
YES = "yes"
LOGIC_OPS = ("and", "or")
SUBJECT = "s"
OBJECT = "o"
UNKNOWN_TARGET = "_"
PERSON_TARGETS = frozenset({"person", "people"})
PLACE_TARGETS = frozenset({"place"})
PRONOUNS = {"he": "man", "him": "man", "she": "woman", "her": "woman"}
ID_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")
WORD_RE = re.compile(r"[a-z]+")
SINGULAR_S_ENDINGS = ("ss", "us", "is")
VOWELS = "aeiou"
ADJECTIVE_FILTERS = frozenset({"color", "material", "shape", "size", "pattern", ""})
COMPARE_OPS = ("same", "different")
MATERIAL = "material"


class OracleRuleError(ValueError):
    """The program uses an operation the rules do not cover."""


# ---- noun phrases ---------------------------------------------------------------------------
def is_plural(noun: str) -> bool:
    last = noun.split()[-1] if noun.split() else noun
    if last in {"people", "men", "women", "children"}:
        return True
    return last.endswith("s") and not last.endswith(SINGULAR_S_ENDINGS)


def be(plural: bool) -> str:
    return "are" if plural else "is"


def article(text: str) -> str:
    return "an" if text[:1].lower() in VOWELS else "a"


@dataclass(frozen=True)
class Phrase:
    """A noun phrase: adjectives + noun + post-modifiers (position, relation)."""

    noun: str
    pre: tuple[str, ...] = ()
    post: tuple[str, ...] = ()

    @property
    def plural(self) -> bool:
        return is_plural(self.noun)

    @property
    def core(self) -> str:
        return " ".join([*self.pre, self.noun, *self.post])

    @property
    def the(self) -> str:
        return f"the {self.core}"

    @property
    def indefinite(self) -> str:
        return self.core if self.plural else f"{article(self.core)} {self.core}"


@dataclass(frozen=True)
class Node:
    """An object node of the program (the end of a select/filter/relate chain)."""

    index: int
    phrase: Phrase
    anchor: int | None = None  # node index of the relate input
    relation: str | None = None
    direction: str | None = None
    target: str | None = None  # "_" / "person" / "place" / a named noun


def strip_ref(argument: str) -> str:
    """'man (884411)' -> 'man'; 'car (-) ' -> 'car' (the marker is never used)."""
    return ID_SUFFIX_RE.sub("", argument.strip()).strip()


def noun_of(name: str) -> str:
    name = name.strip().lower()
    return PRONOUNS.get(name, name)


def is_generic(target: str | None) -> bool:
    return target is not None and (
        target == UNKNOWN_TARGET or target in PERSON_TARGETS | PLACE_TARGETS
    )


def generic_word(target: str | None) -> str:
    if target in PERSON_TARGETS:
        return "someone"
    if target in PLACE_TARGETS:
        return "some place"
    return "something"


def generic_noun(target: str | None) -> str:
    if target in PERSON_TARGETS:
        return "person"
    if target in PLACE_TARGETS:
        return "place"
    return "thing"


def position_text(kind: str, value: str) -> str:
    if kind == "vposition" and value == "middle":
        return "in the middle"
    return f"at the {value}" if kind == "vposition" else f"on the {value}"


# ---- program walk -----------------------------------------------------------------------------
def op_name(op: Mapping[str, Any]) -> tuple[str, str]:
    """('filter', 'color') from 'filter color'; ('exist', '') from 'exist'."""
    parts = str(op["operation"]).split(" ", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def parse_relation(argument: str) -> tuple[str, str, str]:
    """'cabinet,to the left of,o (984369)' -> ('cabinet', 'to the left of', 'o')."""
    fields = [f.strip() for f in strip_ref(argument).split(",")]
    if len(fields) != 3 or fields[2] not in (SUBJECT, OBJECT):
        raise OracleRuleError(f"unparsed relation argument: {argument!r}")
    return noun_of(fields[0]), fields[1], fields[2]


class ProgramWalker:
    """Resolves object nodes of one program (pure, no I/O)."""

    def __init__(self, program: Sequence[Mapping[str, Any]]) -> None:
        self.program = list(program)
        self.nodes: dict[int, Node] = {}

    def node(self, index: int) -> Node:
        if index in self.nodes:
            return self.nodes[index]
        op = self.program[index]
        name, kind = op_name(op)
        if name == "select":
            node = Node(index, Phrase(noun_of(strip_ref(op["argument"]))))
        elif name == "filter":
            node = self._filter(index, kind, op)
        elif name == "relate":
            node = self._relate(index, op)
        else:
            raise OracleRuleError(f"op {index} '{op['operation']}' is not an object")
        self.nodes[index] = node
        return node

    def _filter(self, index: int, kind: str, op: Mapping[str, Any]) -> Node:
        base = self.node(int(op["dependencies"][0]))
        value = strip_ref(op["argument"]).lower()
        if value.startswith("not("):
            raise OracleRuleError(f"negated filter at op {index}: {value}")
        phrase = base.phrase
        if kind in ("hposition", "vposition"):
            phrase = replace(phrase, post=(*phrase.post, position_text(kind, value)))
        elif kind in ADJECTIVE_FILTERS:
            phrase = replace(phrase, pre=(*phrase.pre, value))
        else:
            raise OracleRuleError(f"filter type '{kind}' at op {index}")
        return replace(base, index=index, phrase=phrase)

    def _relate(self, index: int, op: Mapping[str, Any]) -> Node:
        anchor = self.node(int(op["dependencies"][0]))
        target, relation, direction = parse_relation(str(op["argument"]))
        noun = generic_noun(target) if is_generic(target) else target
        if direction == SUBJECT:
            post = (f"{relation} {anchor.phrase.the}",)
        else:
            verb = be(anchor.phrase.plural)
            post = (f"that {anchor.phrase.the} {verb} {relation}",)
        return Node(
            index, Phrase(noun, (), post), anchor.index, relation, direction, target
        )

    def lineage(self, node: Node) -> list[Node]:
        """The node's anchors, outermost first, then the node itself."""
        chain = [node]
        while chain[0].anchor is not None:
            chain.insert(0, self.node(chain[0].anchor))
        return chain


# ---- question templates ------------------------------------------------------------------------
def existence_question(walker: ProgramWalker, node: Node) -> str:
    """'Is there a man in the image?' / 'Is the man wearing something?'."""
    if node.anchor is None:
        phrase = node.phrase
        lead = "Are there" if phrase.plural else "Is there"
        return f"{lead} {phrase.indefinite} in the image?"
    anchor = walker.node(node.anchor).phrase
    if is_generic(node.target):
        what = generic_word(node.target)
        if node.direction == SUBJECT:
            return f"Is there {what} {node.relation} {anchor.the}?"
        return f"{be(anchor.plural).capitalize()} {anchor.the} {node.relation} {what}?"
    target = Phrase(node.phrase.noun, node.phrase.pre)
    if node.direction == SUBJECT:
        lead = "Are there" if target.plural else "Is there"
        return f"{lead} {target.indefinite} {node.relation} {anchor.the}?"
    return (
        f"{be(anchor.plural).capitalize()} {anchor.the} {node.relation} "
        f"{target.indefinite}?"
    )


def query_question(walker: ProgramWalker, node: Node, attribute: str) -> str:
    phrase = node.phrase
    if attribute != "name":
        if attribute in ("hposition", "vposition"):
            return f"On which side of the image {be(phrase.plural)} {phrase.the}?"
        return f"What {attribute} {be(phrase.plural)} {phrase.the}?"
    if node.anchor is None or not is_generic(node.target):
        return f"What {be(phrase.plural)} {phrase.the}?"
    anchor = walker.node(node.anchor).phrase
    if node.target in PLACE_TARGETS and node.direction == OBJECT:
        return f"Where {be(anchor.plural)} {anchor.the}?"
    wh = "Who" if node.target in PERSON_TARGETS else "What"
    if node.direction == SUBJECT:
        return f"{wh} is {node.relation} {anchor.the}?"
    return f"{wh} {be(anchor.plural)} {anchor.the} {node.relation}?"


def option_order(options: Sequence[str], question: str) -> list[str]:
    """Order choose options by where their distinctive words occur in the question."""
    words = WORD_RE.findall(question.lower())
    text = " ".join(words)
    shared = set.intersection(*(set(WORD_RE.findall(o.lower())) for o in options))

    def position(option: str) -> tuple[int, str]:
        own = [w for w in WORD_RE.findall(option.lower()) if w not in shared] or [
            option.lower()
        ]
        found = [text.find(w) for w in own if text.find(w) >= 0]
        return (min(found) if found else len(text) + 1, option)

    return sorted(options, key=position)


def join_options(options: Sequence[str]) -> str:
    """'to the left of' + 'to the right of' -> 'to the left or to the right of'."""
    a, b = (o.split() for o in options)
    suffix: list[str] = []
    while len(a) > 1 and len(b) > 1 and a[-1] == b[-1]:
        suffix.insert(0, a.pop())
        b.pop()
    tail = f" {' '.join(suffix)}" if suffix else ""
    return f"{' '.join(a)} or {' '.join(b)}{tail}"


def verify_attr_question(phrase: Phrase, kind: str, value: str) -> str:
    lead = be(phrase.plural).capitalize()
    if kind in ("hposition", "vposition"):
        return f"{lead} {phrase.the} {position_text(kind, value)}?"
    return f"{lead} {phrase.the} {value}?"


def compare_question(op: str, attribute: str, first: Phrase, second: Phrase) -> str:
    pair = f"{first.the} and {second.the}"
    if op == "common":
        return f"What is common to {pair}?"
    if attribute == MATERIAL:
        same = "the same material" if op == "same" else "different materials"
        return f"Are {pair} made of {same}?"
    same = f"the same {attribute}" if op == "same" else f"different {attribute}s"
    return f"Do {pair} have {same}?"


# ---- plan assembly -------------------------------------------------------------------------------
@dataclass
class PlanBuilder:
    walker: ProgramWalker
    question: str
    items: list[dict[str, Any]] = field(default_factory=list)
    existence_ids: dict[int, str] = field(default_factory=dict)  # node index -> id
    subject_ids: dict[str, str] = field(default_factory=dict)  # choose subject -> id

    def _add(self, question: str, depends_on: Sequence[str], **extra: Any) -> str:
        item_id = f"o{len(self.items) + 1}"
        item = {"id": item_id, "question": question, "depends_on": list(depends_on)}
        self.items.append({**item, **extra})
        return item_id

    def presuppose(self, node: Node) -> str:
        """Existence items for the node and its anchors; returns the node's id."""
        parent: str | None = None
        for link in self.walker.lineage(node):
            key = link.index
            if key not in self.existence_ids:
                self.existence_ids[key] = self._add(
                    existence_question(self.walker, link),
                    [parent] if parent else [],
                    role=EXISTENCE,
                    required_answer=YES,
                    logic=None,
                    source_op=link.index,
                )
            parent = self.existence_ids[key]
        return self.existence_ids[node.index]

    def anchors_of(self, node: Node) -> list[str]:
        """Presuppose the anchors of an asserted node; ids it depends on."""
        if node.anchor is None:
            return []
        return [self.presuppose(self.walker.node(node.anchor))]

    def presuppose_subject(self, noun: str, op_index: int) -> str:
        if noun not in self.subject_ids:
            phrase = Phrase(noun)
            lead = "Are there" if phrase.plural else "Is there"
            self.subject_ids[noun] = self._add(
                f"{lead} {phrase.indefinite} in the image?",
                [],
                role=EXISTENCE,
                required_answer=YES,
                logic=None,
                source_op=op_index,
            )
        return self.subject_ids[noun]

    def final(self, text: str, deps: Sequence[str], op: int, logic: str | None) -> None:
        self._add(
            text, deps, role=FINAL, required_answer=None, logic=logic, source_op=op
        )

    def emit(self, index: int, logic: str | None = None) -> None:
        """Final item(s) of the op at ``index`` (recursive for and / or)."""
        op = self.walker.program[index]
        name, kind = op_name(op)
        deps = [int(d) for d in op["dependencies"]]
        if name in LOGIC_OPS:
            if logic is not None:
                raise OracleRuleError("nested and/or")
            for dep in deps:
                self.emit(dep, logic=name)
            return
        if name == "exist":
            node = self.walker.node(deps[0])
            text = existence_question(self.walker, node)
            self.final(text, self.anchors_of(node), index, logic)
            return
        if name in (*COMPARE_OPS, "common"):
            self._compare(index, name, kind, deps, logic)
            return
        node = self.walker.node(deps[0])
        if name == "query":
            # GQA: "query" carries the attribute in its argument ("name", "color")
            attribute = kind or strip_ref(str(op["argument"])).lower() or "name"
            text = query_question(self.walker, node, attribute)
            self.final(text, [self.presuppose(node)], index, logic)
        elif name == "verify":
            self._verify(index, node, kind, str(op["argument"]), logic)
        elif name == "choose":
            self._choose(index, node, kind, str(op["argument"]), logic)
        else:
            raise OracleRuleError(f"final op '{op['operation']}'")

    def _verify(
        self, index: int, node: Node, kind: str, argument: str, logic: str | None
    ) -> None:
        deps = [self.presuppose(node)]
        phrase = node.phrase
        if kind != "rel":
            text = verify_attr_question(phrase, kind, strip_ref(argument).lower())
            self.final(text, deps, index, logic)
            return
        target, relation, direction = parse_relation(argument)
        other = Phrase(target)
        if direction == OBJECT:
            lead = be(phrase.plural).capitalize()
            text = f"{lead} {phrase.the} {relation} {other.indefinite}?"
        else:
            lead = "Are there" if other.plural else "Is there"
            text = f"{lead} {other.indefinite} {relation} {phrase.the}?"
        self.final(text, deps, index, logic)

    def _choose(
        self, index: int, node: Node, kind: str, argument: str, logic: str | None
    ) -> None:
        if kind == "rel":
            fields = [f.strip() for f in strip_ref(argument).split(",")]
            if len(fields) != 3:
                raise OracleRuleError(f"choose rel argument {argument!r}")
            subject = noun_of(fields[0])
            options = join_options(option_order(fields[1].split("|"), self.question))
            anchor_id = self.presuppose(node)
            subject_id = self.presuppose_subject(subject, index)
            first, second = Phrase(subject), node.phrase
            if fields[2] == OBJECT:
                first, second = second, first
            lead = be(first.plural).capitalize()
            text = f"{lead} {first.the} {options} {second.the}?"
            self.final(text, [anchor_id, subject_id], index, logic)
            return
        values = option_order(strip_ref(argument).lower().split("|"), self.question)
        if len(values) != 2:
            raise OracleRuleError(f"choose argument {argument!r}")
        phrase = node.phrase
        lead = be(phrase.plural).capitalize()
        if kind in ("hposition", "vposition"):
            a, b = (position_text(kind, v) for v in values)
        elif kind == "name":
            a, b = (f"{article(v)} {v}" for v in values)
        else:  # color / material / any attribute value
            a, b = values
        text = f"{lead} {phrase.the} {a} or {b}?"
        self.final(text, [self.presuppose(node)], index, logic)

    def _compare(
        self, index: int, name: str, kind: str, deps: list[int], logic: str | None
    ) -> None:
        if len(deps) != 2:
            raise OracleRuleError(f"'{name}' over {len(deps)} input(s)")
        if name != "common" and not kind:
            raise OracleRuleError(f"'{name}' without an attribute")
        first, second = (self.walker.node(d) for d in deps)
        ids = [self.presuppose(first), self.presuppose(second)]
        text = compare_question(name, kind, first.phrase, second.phrase)
        self.final(text, ids, index, logic)


def build_oracle_plan(record: Mapping[str, Any]) -> dict[str, Any]:
    """Oracle plan for one GQA question record (``semantic`` + ``question``).

    Unsupported programs give ``items = []`` and ``sensible = False``.
    """
    program = list(record["semantic"])
    question = str(record["question"])
    flags: list[str] = []
    builder = PlanBuilder(ProgramWalker(program), question)
    try:
        builder.emit(len(program) - 1)
        items = builder.items
    except (OracleRuleError, KeyError, IndexError, ValueError) as exc:
        flags.append(f"unsupported: {exc}")
        items = []
    flags += plan_flags(program, items)
    return {
        "question_id": str(record["question_id"]),
        "type": record.get("type"),
        "question": question,
        "rules_version": RULES_VERSION,
        "items": items,
        "flags": flags,
        "sensible": bool(items) and not any(f.startswith("unsupported") for f in flags),
    }


def plan_flags(
    program: Sequence[Mapping[str, Any]], items: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Informational flags (pronoun mapped, generic target)."""
    flags = []
    for op in program:
        name, _ = op_name(op)
        if name == "select" and strip_ref(op["argument"]).lower() in PRONOUNS:
            flags.append(f"pronoun: {strip_ref(op['argument'])}")
        if name == "relate":
            target = strip_ref(op["argument"]).split(",")[0].strip()
            if is_generic(target):
                flags.append(f"generic target: {target}")
    if items and not any(i["role"] == FINAL for i in items):
        flags.append("unsupported: no final item")
    return flags


def answer_leak(plan: Mapping[str, Any], answer: str) -> list[str]:
    """Item texts that contain the GT answer word although the question does not."""
    words = set(WORD_RE.findall(answer.lower()))
    in_question = set(WORD_RE.findall(str(plan["question"]).lower()))
    leaked = words - in_question
    return [
        i["question"]
        for i in plan["items"]
        if leaked & set(WORD_RE.findall(i["question"].lower()))
    ]


def build_plans(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [build_oracle_plan(r) for r in records]
