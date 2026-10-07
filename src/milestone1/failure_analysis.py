"""Post-hoc failure analysis of generated checks (plan §4; GT used for explanation only).

Each hypothesis is put in exactly one class, checked in this order:

* ``bookkeeping``: the candidate or evidence talks about the graph itself
  (missing relations, scores, low confidence, duplicates, connectivity,
  contradictions) rather than about the scene;
* ``gt_supported``: the candidate names a VG-150 object class that is in this
  image's GT but NOT among the rendered graph labels (a genuinely
  missing-but-present object);
* ``already_observed``: the candidate names a class that is among the rendered
  graph labels;
* ``off_vocab``: none of the above (no VG-150 class, or only classes absent from
  both GT and the rendered graph).

Class names are matched case-insensitively as whole words with an optional
plural ``s``/``es`` (no synonym expansion).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

BOOKKEEPING = "bookkeeping"
GT_SUPPORTED = "gt_supported"
ALREADY_OBSERVED = "already_observed"
OFF_VOCAB = "off_vocab"
CLASSES = (BOOKKEEPING, GT_SUPPORTED, ALREADY_OBSERVED, OFF_VOCAB)

BOOKKEEPING_RE = re.compile(
    r"no relations?\b|has no relations|\bscores?\b|low[- ]confidence|duplicat"
    r"|not connected|\black(s|ing)?\b|contradict",
    re.IGNORECASE,
)
PLACEHOLDER_RE = re.compile(r"^\s*<[^<>]*>\s*$")
SCORE_RE = re.compile(r"\bscores?\b|confidence|probabilit", re.IGNORECASE)


def label_pattern(name: str) -> re.Pattern[str]:
    """Whole-word, case-insensitive class name with optional plural."""
    return re.compile(rf"\b{re.escape(name.lower())}(?:s|es)?\b", re.IGNORECASE)


def mentioned_labels(text: str, names: Iterable[str]) -> set[str]:
    """VG-150 class names mentioned in ``text``."""
    return {n for n in names if label_pattern(n).search(text or "")}


def classify_hypothesis(
    candidate: str,
    evidence: str,
    class_names: Iterable[str],
    gt_labels: set[str],
    rendered_labels: set[str],
) -> tuple[str, set[str]]:
    """(class, mentioned class names) for one hypothesis; see module doc."""
    mentioned = mentioned_labels(candidate, class_names)
    if BOOKKEEPING_RE.search(f"{candidate} {evidence}"):
        return BOOKKEEPING, mentioned
    if mentioned & (gt_labels - rendered_labels):
        return GT_SUPPORTED, mentioned
    if mentioned & rendered_labels:
        return ALREADY_OBSERVED, mentioned
    return OFF_VOCAB, mentioned


def restates_score(*texts: str) -> bool:
    """True if an observed issue is about scores / confidence values."""
    return any(SCORE_RE.search(t or "") for t in texts)


def is_placeholder_echo(*texts: str) -> bool:
    """True if a field is a copied schema placeholder such as ``<short evidence>``."""
    return any(PLACEHOLDER_RE.match(t or "") for t in texts)
