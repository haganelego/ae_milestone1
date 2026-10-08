"""Resolution-level image series (plan §4).

Base image B: EXIF transpose -> RGB -> Lanczos downscale to max side 512 only if
larger. Level t < 20: bilinear down to floor(W*s_t), floor(H*s_t) (min 1) then
bilinear back to (W, H). Level 20 = B. Every level is saved as PNG and those
exact bytes are what the client base64-encodes (no re-encode).
"""

from __future__ import annotations

import io
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .common import (
    append_jsonl,
    atomic_write_bytes,
    read_jsonl,
    sha256_bytes,
    utc_now,
)

PNG_FORMAT = "PNG"
LEVEL_FILE_TEMPLATE = "level_{level:02d}.png"


def level_scale(level: int, n_levels: int, scale_min: float, scale_max: float) -> float:
    """s_t = s_min * (s_max/s_min)^((t-1)/(n-1)); default 0.05 * 20^((t-1)/19)."""
    if not 1 <= level <= n_levels:
        raise ValueError(f"level {level} outside 1..{n_levels}")
    if level == n_levels:
        return scale_max
    exponent = (level - 1) / (n_levels - 1)
    return scale_min * (scale_max / scale_min) ** exponent


def make_base(source: Path, max_side: int) -> Image.Image:
    """EXIF-corrected RGB base image, Lanczos-downscaled only when too large."""
    with Image.open(source) as opened:
        transposed = ImageOps.exif_transpose(opened)
        image = transposed.convert("RGB")
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    ratio = max_side / longest
    size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    return image.resize(size, Image.Resampling.LANCZOS)


def degraded_size(width: int, height: int, scale: float) -> tuple[int, int]:
    return max(1, math.floor(width * scale)), max(1, math.floor(height * scale))


def degrade(base: Image.Image, scale: float, is_top_level: bool) -> Image.Image:
    if is_top_level:
        return base.copy()
    small = base.resize(degraded_size(*base.size, scale), Image.Resampling.BILINEAR)
    return small.resize(base.size, Image.Resampling.BILINEAR)


def png_bytes(image: Image.Image) -> bytes:
    """Deterministic PNG encoding (no metadata, default compression)."""
    buffer = io.BytesIO()
    image.save(buffer, format=PNG_FORMAT, optimize=False)
    return buffer.getvalue()


def level_path(levels_dir: Path, image_id: str, level: int) -> Path:
    return levels_dir / image_id / LEVEL_FILE_TEMPLATE.format(level=level)


class ImageSeries:
    """Generates all level PNGs of an image at once; ``image_manifest.jsonl``.

    The manifest holds one appended line per image generation (all levels);
    on load the last line per image wins. Bytes are re-verified against the
    manifest sha256 every time they are read for sending.
    """

    def __init__(
        self,
        images_cfg: Mapping[str, Any],
        images_root: Path,
        levels_dir: Path,
        manifest_path: Path,
        read_only: bool = False,
    ) -> None:
        self.read_only = read_only  # inherited data_root: never write images
        self.n_levels = int(images_cfg["levels"])
        self.scale_min = float(images_cfg["scale_min"])
        self.scale_max = float(images_cfg["scale_max"])
        self.max_side = int(images_cfg["base_max_side"])
        self.images_root = images_root
        self.levels_dir = levels_dir
        self.manifest_path = manifest_path
        self._images = {r["image_id"]: r for r in read_jsonl(manifest_path)}

    def scale(self, level: int) -> float:
        return level_scale(level, self.n_levels, self.scale_min, self.scale_max)

    def source_path(self, image_id: str) -> Path:
        return self.images_root / f"{image_id}.jpg"

    def ensure(self, image_id: str, levels: Iterable[int]) -> list[dict[str, Any]]:
        """Generate every level if any file is missing; returns manifest rows."""
        wanted = sorted(set(levels))
        if not self._complete(image_id):
            self._generate_checked(image_id)
        return [self.row(image_id, t) for t in wanted]

    def _generate_checked(self, image_id: str) -> None:
        if self.read_only:
            raise RuntimeError(
                f"image series is read-only; {image_id} needs generation"
            )
        self._do_generate(image_id)

    def _complete(self, image_id: str) -> bool:
        entry = self._images.get(image_id)
        if entry is None or len(entry["levels"]) != self.n_levels:
            return False
        return all(Path(lv["path"]).exists() for lv in entry["levels"])

    def _do_generate(self, image_id: str) -> None:
        source = self.source_path(image_id)
        base = make_base(source, self.max_side)
        with Image.open(source) as original:
            original_size = original.size
        levels: list[dict[str, Any]] = []
        previous: str | None = None
        for level in range(1, self.n_levels + 1):
            scale = self.scale(level)
            top = level == self.n_levels
            data = png_bytes(degrade(base, scale, top))
            path = level_path(self.levels_dir, image_id, level)
            atomic_write_bytes(path, data)
            digest = sha256_bytes(data)
            levels.append(
                {
                    "level": level,
                    "scale": scale,
                    "degraded_size": list(
                        base.size if top else degraded_size(*base.size, scale)
                    ),
                    "path": str(path),
                    "sha256": digest,
                    "identical_to_previous": digest == previous,
                }
            )
            previous = digest
        entry = {
            "image_id": image_id,
            "source_path": str(source),
            "source_size": list(original_size),
            "base_size": list(base.size),
            "sent_size": list(base.size),
            "pixels": base.size[0] * base.size[1],
            "generated": utc_now(),
            "levels": levels,
        }
        self._images[image_id] = entry
        append_jsonl(self.manifest_path, entry)

    def path(self, image_id: str, level: int) -> Path:
        return level_path(self.levels_dir, image_id, level)

    def bytes_for(self, image_id: str, level: int) -> bytes:
        """The exact PNG bytes to send; regenerated once if the hash mismatches."""
        self.ensure(image_id, [level])
        data = self.path(image_id, level).read_bytes()
        if sha256_bytes(data) == self.row(image_id, level)["sha256"]:
            return data
        self._generate_checked(image_id)
        data = self.path(image_id, level).read_bytes()
        if sha256_bytes(data) != self.row(image_id, level)["sha256"]:
            raise RuntimeError(f"level image hash mismatch: {image_id} L{level}")
        return data

    def row(self, image_id: str, level: int) -> dict[str, Any]:
        """Image-level fields merged with the given level's fields."""
        entry = self._images[image_id]
        common = {k: v for k, v in entry.items() if k != "levels"}
        return {**common, **entry["levels"][level - 1]}

    def rows(self) -> list[dict[str, Any]]:
        return [
            self.row(image_id, lv["level"])
            for image_id, entry in sorted(self._images.items())
            for lv in entry["levels"]
        ]
