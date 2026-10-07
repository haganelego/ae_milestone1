"""Tests for milestone1.failure_analysis (post-hoc hypothesis classes)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milestone1 import failure_analysis as fa  # noqa: E402

NAMES = ["man", "shirt", "ski", "pole", "street", "sign", "woman"]
GT = {"man", "ski", "street"}
RENDERED = {"man", "pole", "sign"}


def _cls(candidate: str, evidence: str = "") -> str:
    return fa.classify_hypothesis(candidate, evidence, NAMES, GT, RENDERED)[0]


def test_bookkeeping_has_priority() -> None:
    assert _cls("ski", "the ski has no relations in the graph") == fa.BOOKKEEPING
    assert _cls("man --near-- pole", "low-confidence edge e3") == fa.BOOKKEEPING
    assert _cls("pole", "pole lacks an 'on' relation") == fa.BOOKKEEPING


def test_gt_supported_needs_gt_label_missing_from_rendering() -> None:
    assert (
        _cls("a pair of skis under the man") == fa.GT_SUPPORTED
    )  # ski: GT, not rendered
    assert (
        _cls("a shirt on the man") == fa.ALREADY_OBSERVED
    )  # shirt not in GT; man rendered


def test_already_observed_and_off_vocab() -> None:
    assert _cls("sign above the pole") == fa.ALREADY_OBSERVED
    assert _cls("a dog near the fence") == fa.OFF_VOCAB


def test_label_matching_is_whole_word_with_plural() -> None:
    assert fa.mentioned_labels("Two Signs and a woman", NAMES) == {
        "sign",
        "woman",
    }  # not 'man'
    assert fa.mentioned_labels("signpost", NAMES) == set()


def test_score_restatement_and_placeholder_echo() -> None:
    assert fa.restates_score("v3 has score 0.12", "")
    assert not fa.restates_score("the man is wearing a shirt", "")
    assert fa.is_placeholder_echo("ok", "<short evidence>")
    assert not fa.is_placeholder_echo("a <b> c", "")
