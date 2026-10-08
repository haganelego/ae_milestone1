"""Stratified candidate scan and post-screening adoption (plan §3.2 / §16 A2).

Everything is a pure function of the eligible pool, the image hashes and the
seed. Duplicate images (same image id or same source bytes) form one *group*
that is scanned as a unit, so a group can never straddle dev and test.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .common import seeded_hash
from .dataset import QuestionRecord, cell_name, normalize_text

CAPPED_SHARE_PREFIX = "all_up_to_"


@dataclass(frozen=True)
class Strata:
    """Cell definitions and target shares parsed from ``dataset.stratification``."""

    attributes: tuple[str, ...]
    relates: tuple[int, ...]
    attribute_share: dict[str, float]
    capped: frozenset[str]
    relate_share: dict[int, float]

    @property
    def cells(self) -> list[str]:
        return [cell_name(a, r) for a in self.attributes for r in self.relates]

    def attribute_of(self, cell: str) -> str:
        return cell.split("|")[0]

    def weight(self, cell: str) -> float:
        attribute, relate = cell.split("|")
        return self.attribute_share[attribute] * self.relate_share[int(relate[1:])]


def parse_strata(strat_cfg: Mapping[str, Any]) -> Strata:
    shares: dict[str, float] = {}
    capped: set[str] = set()
    for attribute, value in strat_cfg["attribute_share"].items():
        if isinstance(value, str) and value.startswith(CAPPED_SHARE_PREFIX):
            shares[attribute] = float(value[len(CAPPED_SHARE_PREFIX) :])
            capped.add(attribute)
        else:
            shares[attribute] = float(value)
    relate_share = {
        int(k): float(v) for k, v in strat_cfg["relate_share_within_attribute"].items()
    }
    return Strata(
        attributes=tuple(shares),
        relates=tuple(sorted(relate_share)),
        attribute_share=shares,
        capped=frozenset(capped),
        relate_share=relate_share,
    )


def largest_remainder(amount: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Split ``amount`` integers proportionally; ties go to earlier keys."""
    keys = list(weights)
    total_weight = sum(weights.values())
    if amount <= 0 or total_weight <= 0:
        return dict.fromkeys(keys, 0)
    raw = {k: amount * weights[k] / total_weight for k in keys}
    result = {k: math.floor(raw[k]) for k in keys}
    remaining = amount - sum(result.values())
    order = sorted(keys, key=lambda k: (-(raw[k] - result[k]), keys.index(k)))
    for k in order[:remaining]:
        result[k] += 1
    return result


def distribute(
    amount: int, weights: Mapping[str, float], spare: Mapping[str, int]
) -> tuple[dict[str, int], int]:
    """Proportionally hand out ``amount`` capped by ``spare``; returns (given, leftover)."""
    given = dict.fromkeys(weights, 0)
    left = amount
    while left > 0:
        open_keys = {k: w for k, w in weights.items() if spare[k] - given[k] > 0}
        if not open_keys:
            break
        if sum(open_keys.values()) <= 0:
            open_keys = dict.fromkeys(open_keys, 1.0)
        share = largest_remainder(left, open_keys)
        moved = 0
        for k, units in share.items():
            take = min(units, spare[k] - given[k])
            given[k] += take
            moved += take
        left -= moved
        if moved == 0:
            break
    return given, left


@dataclass
class Allocation:
    targets: dict[str, int]
    initial: dict[str, int]
    available: dict[str, int]
    unfilled: int
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": self.targets,
            "initial": self.initial,
            "available": self.available,
            "unfilled": self.unfilled,
            "log": self.log,
        }


def allocate(total: int, strata: Strata, available: Mapping[str, int]) -> Allocation:
    """Cell targets for ``total`` questions with the §16 A2 shortfall reallocation.

    1. Initial targets = total x attribute share x relate share (largest remainder).
    2. A cell short of supply passes its excess to the same attribute's other
       relate cell(s), proportional to relate share.
    3. What remains goes to the other *uncapped* attributes in proportion to
       their shares (``all_up_to_X`` attributes never receive extra).
    """
    cells = strata.cells
    avail = {c: int(available.get(c, 0)) for c in cells}
    initial = largest_remainder(total, {c: strata.weight(c) for c in cells})
    targets = dict(initial)
    log: list[str] = []
    leftover = 0
    for attribute in strata.attributes:
        own = [c for c in cells if strata.attribute_of(c) == attribute]
        excess = 0
        for c in own:
            if targets[c] > avail[c]:
                excess += targets[c] - avail[c]
                log.append(f"{c}: target {targets[c]} > available {avail[c]}")
                targets[c] = avail[c]
        if excess == 0:
            continue
        spare = {c: avail[c] - targets[c] for c in own}
        weights = {c: strata.weight(c) for c in own}
        given, rest = distribute(excess, weights, spare)
        for c, units in given.items():
            if units:
                targets[c] += units
                log.append(f"{attribute}: +{units} to {c} (same attribute)")
        leftover += rest
    if leftover:
        receivers = [c for c in cells if strata.attribute_of(c) not in strata.capped]
        spare = {c: avail[c] - targets[c] for c in receivers}
        weights = {c: strata.weight(c) for c in receivers}
        given, leftover = distribute(leftover, weights, spare)
        for c, units in given.items():
            if units:
                targets[c] += units
                log.append(f"+{units} to {c} (other attribute, proportional)")
    if leftover:
        log.append(f"unfilled after reallocation: {leftover}")
    return Allocation(targets, initial, avail, leftover, log)


@dataclass(frozen=True)
class ImageGroup:
    """One scan unit: an image id plus any byte-identical duplicates."""

    group_id: str
    image_ids: tuple[str, ...]
    questions: tuple[QuestionRecord, ...]  # sorted by seeded question hash
    order_key: str


def build_groups(
    records: Sequence[QuestionRecord], group_of: Mapping[str, str], seed: int
) -> list[ImageGroup]:
    """Groups in SHA256(seed|group_id) order; questions in SHA256(seed|qid) order."""
    members: dict[str, set[str]] = defaultdict(set)
    questions: dict[str, list[QuestionRecord]] = defaultdict(list)
    for record in records:
        group_id = group_of.get(record.image_id, record.image_id)
        members[group_id].add(record.image_id)
        questions[group_id].append(record)
    groups = [
        ImageGroup(
            group_id=gid,
            image_ids=tuple(sorted(members[gid])),
            questions=tuple(
                sorted(questions[gid], key=lambda q: question_order_key(seed, q))
            ),
            order_key=seeded_hash(seed, gid),
        )
        for gid in members
    ]
    return sorted(groups, key=lambda g: (g.order_key, g.group_id))


def question_order_key(seed: int, record: QuestionRecord) -> tuple[str, str]:
    return seeded_hash(seed, record.question_id), record.question_id


def cell_counts(groups: Sequence[ImageGroup]) -> dict[str, int]:
    return dict(Counter(q.cell for g in groups for q in g.questions))


def pick_from_group(
    group: ImageGroup, open_slots: Mapping[str, int], max_per_image: int
) -> list[QuestionRecord]:
    """Up to ``max_per_image`` questions; prefer different attributes (plan §3.2)."""
    remaining = dict(open_slots)
    picked: list[QuestionRecord] = []
    while len(picked) < max_per_image:
        usable = [
            q
            for q in group.questions
            if q not in picked
            and remaining.get(q.cell, 0) > 0
            and normalize_text(q.question)
            not in {normalize_text(p.question) for p in picked}
        ]
        if not usable:
            break
        used_attributes = {p.attribute for p in picked}
        different = [q for q in usable if q.attribute not in used_attributes]
        choice = different[0] if different else usable[0]
        picked.append(choice)
        remaining[choice.cell] -= 1
    return picked


@dataclass
class ScanResult:
    split: str
    batch: str
    quotas: Allocation
    candidates: list[dict[str, Any]]
    groups_used: list[str]
    groups_scanned: int
    start_index: int = 0
    end_index: int = 0

    def filled(self) -> dict[str, int]:
        return dict(Counter(c["cell"] for c in self.candidates))


def share_quotas(total: int, strata: Strata) -> dict[str, int]:
    """Configured share quota per cell (largest remainder, no reallocation)."""
    return largest_remainder(total, {c: strata.weight(c) for c in strata.cells})


SPLIT_QUOTA_RULE = (
    "A2-bis: dev_quota[c] = min(share quota of candidate_dev_questions, "
    "floor(cell_size * target_dev / (target_dev + target_test))); "
    "test_quota[c] = min(share quota of candidate_test_questions, "
    "cell_size - dev_quota[c]); scan has no image cap"
)


def split_quotas(
    strata: Strata,
    cell_sizes: Mapping[str, int],
    dev_total: int,
    test_total: int,
    target_dev: int,
    target_test: int,
) -> tuple[Allocation, Allocation]:
    """Per-cell candidate quotas (plan §16 A2-bis).

    Dense cells are bound by the configured share quota; rare cells by the
    dev fraction target_dev / (target_dev + target_test), so test keeps the
    remainder of every rare cell. Integer arithmetic (no float rounding).
    """
    sizes = {c: int(cell_sizes.get(c, 0)) for c in strata.cells}
    dev_share = share_quotas(dev_total, strata)
    test_share = share_quotas(test_total, strata)
    denominator = target_dev + target_test
    dev: dict[str, int] = {}
    test: dict[str, int] = {}
    dev_log: list[str] = [SPLIT_QUOTA_RULE]
    test_log: list[str] = [SPLIT_QUOTA_RULE]
    for cell in strata.cells:
        dev_cap = sizes[cell] * target_dev // denominator
        dev[cell] = min(dev_share[cell], dev_cap)
        test[cell] = min(test_share[cell], sizes[cell] - dev[cell])
        dev_bound = "share" if dev[cell] == dev_share[cell] else "split_fraction"
        test_bound = "share" if test[cell] == test_share[cell] else "cell_remainder"
        dev_log.append(
            f"{cell}: size {sizes[cell]}, share {dev_share[cell]}, "
            f"split cap {dev_cap} -> {dev[cell]} ({dev_bound})"
        )
        test_log.append(
            f"{cell}: size {sizes[cell]}, share {test_share[cell]}, remainder "
            f"{sizes[cell] - dev[cell]} -> {test[cell]} ({test_bound})"
        )
    dev_alloc = Allocation(
        dev, dev_share, sizes, sum(dev_share.values()) - sum(dev.values()), dev_log
    )
    test_alloc = Allocation(
        test, test_share, sizes, sum(test_share.values()) - sum(test.values()), test_log
    )
    return dev_alloc, test_alloc


def scan_split(
    groups: Sequence[ImageGroup],
    split: str,
    batch: str,
    quotas: Allocation,
    max_images: int | None,
    max_per_image: int,
    excluded_groups: set[str],
    start_index: int = 0,
) -> ScanResult:
    """Walk groups in hash order from ``start_index`` filling cell quotas.

    Stops when every quota is full or the pool is exhausted; ``max_images``
    (None = no cap) only bounds reserve batches. Groups that offer nothing for
    an open cell stay unused. ``end_index`` is where a later scan continues.
    """
    open_slots = dict(quotas.targets)
    candidates: list[dict[str, Any]] = []
    used: list[str] = []
    scanned = 0
    index = start_index
    while index < len(groups):
        if sum(open_slots.values()) <= 0:
            break
        if max_images is not None and len(used) >= max_images:
            break
        group = groups[index]
        index += 1
        if group.group_id in excluded_groups:
            continue
        scanned += 1
        picked = pick_from_group(group, open_slots, max_per_image)
        if not picked:
            continue
        used.append(group.group_id)
        for rank, record in enumerate(picked):
            open_slots[record.cell] -= 1
            candidates.append(candidate_row(record, group, split, batch, rank))
    return ScanResult(
        split, batch, quotas, candidates, used, scanned, start_index, index
    )


def candidate_row(
    record: QuestionRecord, group: ImageGroup, split: str, batch: str, rank: int
) -> dict[str, Any]:
    """Manifest row. GT is referenced by question id, not copied (plan §9.1)."""
    return {
        "question_id": record.question_id,
        "image_id": record.image_id,
        "group_id": group.group_id,
        "group_image_ids": list(group.image_ids),
        "split": split,
        "batch": batch,
        "question": record.question,
        "attribute": record.attribute,
        "relate_count": record.relate_count,
        "operation_count": record.operation_count,
        "detailed_type": record.detailed_type,
        "cell": record.cell,
        "gt_ref": f"gqa_val_balanced:{record.question_id}",
        "selection_reason": f"stratified_scan:rank_in_image={rank}",
    }


def adopt(
    candidates: Sequence[Mapping[str, Any]],
    passed_ids: set[str],
    target_total: int,
    strata: Strata,
    seed: int,
) -> tuple[list[dict[str, Any]], Allocation]:
    """Adopt screened-correct candidates per cell in hash order (plan §16 A2).

    Returned rows are in global SHA256(seed|qid) order (pilot uses the head).
    """
    by_cell: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row["question_id"] in passed_ids:
            by_cell[row["cell"]].append(row)
    for rows in by_cell.values():
        rows.sort(key=lambda r: (seeded_hash(seed, r["question_id"]), r["question_id"]))
    allocation = allocate(
        target_total, strata, {c: len(by_cell.get(c, [])) for c in strata.cells}
    )
    adopted = [
        dict(row)
        for cell in strata.cells
        for row in by_cell.get(cell, [])[: allocation.targets[cell]]
    ]
    adopted.sort(key=lambda r: (seeded_hash(seed, r["question_id"]), r["question_id"]))
    return adopted, allocation


def meets_minimum(rows: Sequence[Mapping[str, Any]], min_q: int, min_img: int) -> bool:
    return len(rows) >= min_q and len({r["group_id"] for r in rows}) >= min_img
