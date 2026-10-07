"""ROBOVIS2027 judge v4: bbox-grounded per-object visibility (variant F1g_v4_*).

One VLM call per state: the current image with the cleaned graph's top-K
entities drawn as labelled boxes, plus the same list as text. The model rates
each object 0/1/2 ("clear"). Observed issues and their judgments are then
derived deterministically (no second call); hypotheses and their judgments are
reused from existing v2 checks / v3 decisions so H is unchanged.

The prompt is generic (no resolution/step/scale words). The observed-issue cap
of the plan contract (proposal.max_observed_issues) is respected: the least
clear objects are kept, ties broken by higher graph score; the raw ratings of
all K objects are stored in ``v4_objects``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROMPT_V4 = "v4"
CLEAR_GRADES = (0, 1, 2)
CLEAR_RESOLVED = 2
BENEFIT_FOR_CLEAR = {0: 1, 1: 2}  # clear 0 -> benefit 1, clear 1 -> benefit 2
BASIS_GROUNDED = 2
RESOLVED_LIMITATION = "other"
UNRESOLVED_LIMITATION = "low_resolution"

V4_PROMPT = (
    "The scene graph claims these objects exist at the drawn boxes. For each "
    "listed id, judge from the image: can you clearly see and identify that object "
    'inside its box? Return JSON {"objects":[{"id":"<listed id>","clear":0|1|2,'
    '"note":"<short>"}]} where 2 = clearly visible and identifiable, 1 = something '
    "is there but its identity or extent is unclear, 0 = cannot confirm any such "
    "object at that box. Judge every listed id exactly once."
)
# Format-only addendum (smoke 1: the model copied the example id and returned bare
# lists); does not change what is judged.
V4_FORMAT_NOTE = (
    "Use exactly the listed ids (copy them verbatim) and return a JSON object with "
    'the key "objects", not a bare list.'
)
STRING_HINT_V4 = "Each note must be under {chars} characters."
TARGET_TEMPLATE = "identity/extent of {label}"


@dataclass(frozen=True)
class OverlayStyle:
    line_width: int
    color: tuple[int, int, int]
    font_size: int
    jpeg_quality: int

    def record(self) -> dict[str, Any]:
        return {
            "line_width": self.line_width,
            "color": list(self.color),
            "font_size": self.font_size,
            "jpeg_quality": self.jpeg_quality,
            "drawn": "rectangle + id text at box top-left",
        }


def overlay_style_from_config(cfg: dict[str, Any]) -> OverlayStyle:
    v4 = cfg["checks"]["v4"]
    return OverlayStyle(
        line_width=int(v4["overlay_line_width"]),
        color=tuple(int(c) for c in v4["overlay_color"]),  # type: ignore[arg-type]
        font_size=int(v4["overlay_font_size"]),
        jpeg_quality=int(cfg["resolution"]["jpeg_quality"]),
    )


def top_entities(graph: dict[str, Any], k: int) -> list[dict[str, Any]]:
    """Top-K entities by score (ties: graph order), with a usable base box."""
    with_box = [e for e in graph.get("entities", []) if e.get("box_xyxy_base")]
    order = {e["id"]: i for i, e in enumerate(with_box)}
    ranked = sorted(
        with_box, key=lambda e: (-float(e.get("score", 0.0)), order[e["id"]])
    )
    return ranked[:k]


def normalized_box(
    entity: dict[str, Any], base_size: Any, decimals: int
) -> list[float]:
    width, height = float(base_size[0]), float(base_size[1])
    x1, y1, x2, y2 = (float(v) for v in entity["box_xyxy_base"])
    norm = [x1 / width, y1 / height, x2 / width, y2 / height]
    return [round(min(1.0, max(0.0, v)), decimals) for v in norm]


def entity_list_text(
    entities: list[dict[str, Any]], base_size: Any, decimals: int
) -> str:
    lines = ["Listed objects (box = normalized [x1, y1, x2, y2], origin top-left):"]
    for e in entities:
        box = ",".join(
            f"{v:.{decimals}f}" for v in normalized_box(e, base_size, decimals)
        )
        lines.append(f"{e['id']}: {e['label']}, box=[{box}]")
    return "\n".join(lines)


def build_v4_prompt(
    entities: list[dict[str, Any]], base_size: Any, decimals: int, hint_chars: int
) -> str:
    hint = STRING_HINT_V4.format(chars=hint_chars)
    listing = entity_list_text(entities, base_size, decimals)
    return f"{V4_PROMPT} {V4_FORMAT_NOTE} {hint}\n\n{listing}"


def draw_overlay(
    image_bytes: bytes, entities: list[dict[str, Any]], style: OverlayStyle
) -> bytes:
    """Draw thin labelled rectangles (base-image coordinates) and return JPEG bytes."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=style.font_size)
    for e in entities:
        x1, y1, x2, y2 = (float(v) for v in e["box_xyxy_base"])
        draw.rectangle([x1, y1, x2, y2], outline=style.color, width=style.line_width)
        tx, ty = x1 + style.line_width, y1 + style.line_width
        bbox = draw.textbbox((tx, ty), e["id"], font=font)
        draw.rectangle(bbox, fill=style.color)
        draw.text((tx, ty), e["id"], fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=style.jpeg_quality)
    return buf.getvalue()


def validate_v4(
    data: Any, expected_ids: list[str], max_chars: int
) -> tuple[dict[str, Any] | None, list[str]]:
    """Every listed id exactly once; clear in {0,1,2}; note a short string."""
    if not isinstance(data, dict) or not isinstance(data.get("objects"), list):
        return None, ["top level must be an object with an 'objects' list"]
    errors: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for idx, obj in enumerate(data["objects"]):
        where = f"objects[{idx}]"
        if not isinstance(obj, dict):
            errors.append(f"{where} must be an object")
            continue
        obj_id = obj.get("id")
        if obj_id not in expected_ids:
            errors.append(f"{where}.id not listed: {obj_id!r}")
            continue
        if obj_id in by_id:
            errors.append(f"duplicate id {obj_id}")
            continue
        clear = obj.get("clear")
        if isinstance(clear, bool) or clear not in CLEAR_GRADES:
            errors.append(f"{where}.clear must be 0, 1 or 2: {clear!r}")
        note = obj.get("note")
        if not isinstance(note, str):
            errors.append(f"{where}.note must be a string")
        elif len(note) > max_chars:
            errors.append(f"{where}.note longer than {max_chars} chars")
        by_id[obj_id] = {"id": obj_id, "clear": clear, "note": note}
    missing = [i for i in expected_ids if i not in by_id]
    if missing:
        errors.append(f"missing ids: {missing}")
    if errors:
        return None, errors
    return {"objects": [by_id[i] for i in expected_ids]}, []


def derive_observed(
    objects: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    max_observed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(observed_issues, judgment items) from per-object ratings, capped.

    Least clear first; ties by higher graph score; ids o1..oN in that order.
    """
    info = {e["id"]: e for e in entities}
    ranked = sorted(
        objects, key=lambda o: (o["clear"], -float(info[o["id"]].get("score", 0.0)))
    )[:max_observed]
    issues: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for n, obj in enumerate(ranked, start=1):
        check_id = f"o{n}"
        resolved = obj["clear"] == CLEAR_RESOLVED
        issues.append(
            {
                "id": check_id,
                "anchor_ids": [obj["id"]],
                "target": TARGET_TEMPLATE.format(label=info[obj["id"]]["label"]),
                "region_anchor_id": obj["id"],
                "evidence": obj["note"],
            }
        )
        items.append(
            {
                "id": check_id,
                "status": "resolved" if resolved else "unresolved",
                "basis": BASIS_GROUNDED,
                "resolution_benefit": 0
                if resolved
                else BENEFIT_FOR_CLEAR[obj["clear"]],
                "limitation": RESOLVED_LIMITATION
                if resolved
                else UNRESOLVED_LIMITATION,
                "reason": obj["note"],
            }
        )
    return issues, items


def reuse_hypotheses(
    hyp_checks: dict[str, Any] | None, hyp_decision: dict[str, Any] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Hypotheses + their judgments from earlier records; drops unjudged ones."""
    if not hyp_checks or hyp_checks.get("failure") or not hyp_checks.get("checks"):
        return [], [], []
    hyps = list(hyp_checks["checks"].get("hypotheses", []))
    judged = {}
    if hyp_decision and not hyp_decision.get("failure"):
        judged = {i["id"]: i for i in (hyp_decision.get("items") or [])}
    kept = [h for h in hyps if h["id"] in judged]
    dropped = [h["id"] for h in hyps if h["id"] not in judged]
    return kept, [judged[h["id"]] for h in kept], dropped
