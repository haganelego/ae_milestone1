"""Data preparation for the Fig 3 qualitative panels (no plotting here).

- :func:`panel_levels` picks 3-4 levels to show, always including both stop levels.
- :func:`hypotheses_at` reads the hypotheses (plus their judgments when present)
  of one image at the checks level closest to a stop level.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from milestone1.robovis_data import level_name

DEFAULT_PANEL_COUNT = 4
FIRST_LEVEL = 1
MAX_HYPOTHESES = 3
TEXT_MAX_CHARS = 70
ELLIPSIS = "..."


def panel_levels(
    stop_a: int, stop_b: int, num_levels: int, n_panels: int = DEFAULT_PANEL_COUNT
) -> list[int]:
    """Sorted distinct levels: both stops, the final level, an early level, then
    the level farthest from those already chosen until ``n_panels`` are picked."""
    chosen = {stop_a, stop_b, num_levels}
    chosen.add(max(FIRST_LEVEL, min(stop_a, stop_b) // 2))
    candidates = range(FIRST_LEVEL, num_levels + 1)
    while len(chosen) < min(n_panels, num_levels):
        chosen.add(max(candidates, key=lambda t: (min(abs(t - c) for c in chosen), -t)))
    return sorted(chosen)


@dataclass(frozen=True)
class HypothesisLines:
    level: int | None  # checks level actually used (None if no checks at all)
    lines: tuple[str, ...]


def _shorten(text: str, limit: int = TEXT_MAX_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - len(ELLIPSIS)] + ELLIPSIS


def _read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _available_levels(checks_dir: Path) -> list[int]:
    out = []
    for p in checks_dir.glob("level_*.json"):
        stem = p.stem.removeprefix("level_")
        if stem.isdigit():
            out.append(int(stem))
    return sorted(out)


def _nearest_level(levels: list[int], target: int) -> int | None:
    """Closest level <= target, else the closest level above it."""
    if not levels:
        return None
    below = [t for t in levels if t <= target]
    return max(below) if below else min(levels)


def hypotheses_at(
    checks_root: Path, decisions_root: Path | None, image_id: int, target_level: int
) -> HypothesisLines:
    """Top hypotheses (with judgment summary) at the checks level nearest target."""
    checks_dir = checks_root / str(image_id)
    level = _nearest_level(_available_levels(checks_dir), target_level)
    if level is None:
        return HypothesisLines(None, ())
    fname = f"{level_name(level)}.json"
    record = _read(checks_dir / fname) or {}
    payload = record.get("checks", record)
    if record.get("failure") is True or not isinstance(payload, dict):
        return HypothesisLines(level, ("(check generation failed)",))
    hyps = payload.get("hypotheses") or []
    judged: dict[str, dict[str, Any]] = {}
    if decisions_root is not None:
        decision = _read(decisions_root / str(image_id) / fname) or {}
        for item in decision.get("items") or []:
            if isinstance(item, dict) and "id" in item:
                judged[str(item["id"])] = item
    lines = []
    for h in hyps[:MAX_HYPOTHESES]:
        if not isinstance(h, dict):
            continue
        text = f"[{h.get('kind', '?')}] {_shorten(h.get('candidate', ''))}"
        j = judged.get(str(h.get("id")))
        if j is not None:
            text += (
                f" | {j.get('status')}, b={j.get('basis')}/"
                f"r={j.get('resolution_benefit')}, {j.get('limitation')}"
            )
        lines.append(text)
    return HypothesisLines(level, tuple(lines) or ("(no hypotheses)",))
