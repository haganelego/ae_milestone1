"""Image series (plan §4) and mandatory test 4: constant size, L20 == base,
sent bytes hash == stored PNG hash."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from gqa_stop_helpers import CONFIG_PATH, PROMPTS_DIR
from PIL import Image

from milestone1.gqa_stop.client import RequestEngine, load_prompts, settings_from_config
from milestone1.gqa_stop.images import (
    ImageSeries,
    level_scale,
    make_base,
    png_bytes,
)
from milestone1.gqa_stop.inference import build_level_spec

CFG = json.loads(CONFIG_PATH.read_text())
EXIF_ORIENTATION_TAG = 0x0112
ROTATE_90_CW = 6


def series(tmp_path: Path) -> ImageSeries:
    return ImageSeries(
        CFG["images"],
        tmp_path / "src",
        tmp_path / "levels",
        tmp_path / "image_manifest.jsonl",
    )


def write_source(tmp_path: Path, image_id: str, size: tuple[int, int]) -> Path:
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (10, 200, 30))
    for x in range(0, size[0], 7):
        for y in range(0, size[1], 5):
            image.putpixel((x, y), (255, (x * 3) % 255, (y * 5) % 255))
    path = tmp_path / "src" / f"{image_id}.jpg"
    image.save(path, quality=95)
    return path


def test_scales_follow_formula() -> None:
    for t in range(1, 21):
        assert level_scale(t, 20, 0.05, 1.0) == pytest.approx(
            0.05 * 20 ** ((t - 1) / 19)
        )
    assert level_scale(1, 20, 0.05, 1.0) == pytest.approx(0.05)
    assert level_scale(20, 20, 0.05, 1.0) == 1.0
    with pytest.raises(ValueError):
        level_scale(21, 20, 0.05, 1.0)


def test_all_levels_constant_size_and_level20_is_base(tmp_path: Path) -> None:
    source = write_source(tmp_path, "7", (900, 600))
    s = series(tmp_path)
    rows = s.ensure("7", range(1, 21))
    base = make_base(source, 512)
    assert base.size == (512, 341)
    for row in rows:
        with Image.open(row["path"]) as image:
            assert image.size == base.size
        assert row["sent_size"] == list(base.size)
    assert s.bytes_for("7", 20) == png_bytes(base)
    assert rows[0]["degraded_size"] == [25, 17]  # floor(512*0.05), floor(341*0.05)


def test_small_images_are_not_upscaled(tmp_path: Path) -> None:
    write_source(tmp_path, "8", (120, 80))
    rows = series(tmp_path).ensure("8", [1, 20])
    assert rows[1]["base_size"] == [120, 80]
    assert rows[0]["degraded_size"] == [6, 4]


def test_tiny_dimension_never_below_one(tmp_path: Path) -> None:
    write_source(tmp_path, "9", (10, 10))
    rows = series(tmp_path).ensure("9", [1])
    assert rows[0]["degraded_size"] == [1, 1]


def test_identical_adjacent_levels_are_recorded_not_dropped(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    Image.new("RGB", (30, 20), (90, 90, 90)).save(tmp_path / "src" / "u.jpg")
    rows = series(tmp_path).ensure("u", range(1, 21))
    assert len(rows) == 20
    assert any(r["identical_to_previous"] for r in rows)
    assert rows[0]["identical_to_previous"] is False


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    image = Image.new("RGB", (60, 20), (0, 0, 255))
    exif = image.getexif()
    exif[EXIF_ORIENTATION_TAG] = ROTATE_90_CW
    image.save(tmp_path / "src" / "e.jpg", exif=exif)
    assert make_base(tmp_path / "src" / "e.jpg", 512).size == (20, 60)


def test_manifest_hash_matches_file_and_sent_payload(tmp_path: Path) -> None:
    write_source(tmp_path, "7", (300, 200))
    s = series(tmp_path)
    s.ensure("7", [1, 10, 20])
    settings = settings_from_config(CFG)
    engine = RequestEngine(settings, load_prompts(PROMPTS_DIR), None, tmp_path)
    question = {"question_id": "q", "image_id": "7", "question": "What color?"}
    for level in (1, 10, 20):
        spec = build_level_spec("answer_only", question, level, s.bytes_for("7", level))
        payload = engine.build_payload(spec, engine.build_messages(spec))
        sent = base64.b64decode(payload["messages"][0]["images"][0])
        stored = Path(s.row("7", level)["path"]).read_bytes()
        assert sent == stored
        assert hashlib.sha256(sent).hexdigest() == s.row("7", level)["sha256"]


def test_resume_does_not_regenerate_valid_levels(tmp_path: Path) -> None:
    write_source(tmp_path, "7", (100, 100))
    s = series(tmp_path)
    s.ensure("7", [3])
    path = Path(s.row("7", 3)["path"])
    mtime = path.stat().st_mtime_ns
    again = series(tmp_path)
    again.ensure("7", [3])
    assert path.stat().st_mtime_ns == mtime
