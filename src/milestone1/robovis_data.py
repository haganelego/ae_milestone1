"""Dataset selection and resolution-degradation helpers for the ROBOVIS2027 run.

Pure / side-effect-light functions shared by
``scripts/26_robovis_select_images.py``, ``scripts/27_robovis_generate_levels.py``
and ``tests/test_robovis_dataset.py``.
"""

import hashlib
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from milestone1.config import REPO_ROOT, load_config

ROBOVIS_CONFIG_PATH = REPO_ROOT / "configs" / "robovis2027.yaml"
HASH_CHUNK_BYTES = 1 << 20
# Same naming as scripts/02_generate_resolution_levels.py.
LEVEL_NAME_FORMAT = "level_{:02d}"
MIN_PIXELS = 1
FULL_SCALE = 1.0

# Map config interpolation name -> PIL resample filter.
# Copied from scripts/02_generate_resolution_levels.py (not importable: module name
# starts with a digit and the logic lives inside main()).
RESAMPLE_FILTERS: dict[str, Image.Resampling] = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def load_robovis_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configs/robovis2027.yaml."""
    path = Path(config_path) if config_path is not None else ROBOVIS_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def load_miru_scales() -> list[float]:
    """Return the 20-level scale array of the MIRU pipeline (configs/config.yaml)."""
    return [float(s) for s in load_config()["resolution_levels"]["scales"]]


def sha256_file(path: str | Path) -> str:
    """Return the hex sha256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ids(ids: Iterable[int]) -> str:
    """Return a sha256 fingerprint of an integer id list (order-sensitive)."""
    payload = ",".join(str(i) for i in ids).encode()
    return hashlib.sha256(payload).hexdigest()


def meets_gt_criteria(
    n_objects: int, n_relations: int, min_objects: int, min_relations: int
) -> bool:
    """GT-only eligibility rule (plan §8)."""
    return n_objects >= min_objects and n_relations >= min_relations


def shuffle_and_split(
    eligible_ids: Iterable[int], seed: int, test_count: int, dev_count: int
) -> tuple[list[int], list[int], list[int]]:
    """Sort ids ascending, shuffle with ``random.Random(seed)``, split test/dev.

    Returns (shuffled_all, test_ids, dev_ids). The test set keeps the shuffled
    order so it can be truncated to a prefix later; dev follows immediately after.
    """
    ordered = sorted(set(eligible_ids))
    needed = test_count + dev_count
    if len(ordered) < needed:
        raise ValueError(f"Need {needed} eligible images, only {len(ordered)}")
    random.Random(seed).shuffle(ordered)
    test_ids = ordered[:test_count]
    dev_ids = ordered[test_count:needed]
    return ordered, test_ids, dev_ids


def compute_base_size(width: int, height: int, base_max_side: int) -> tuple[int, int]:
    """Aspect-preserving size whose longer side equals ``base_max_side``."""
    ratio = base_max_side / max(width, height)
    base_w = max(MIN_PIXELS, round(width * ratio))
    base_h = max(MIN_PIXELS, round(height * ratio))
    if width >= height:
        base_w = base_max_side
    else:
        base_h = base_max_side
    return base_w, base_h


def effective_size(width: int, height: int, scale: float) -> tuple[int, int]:
    """Downscaled size used at a level (same formula as script 02)."""
    return max(MIN_PIXELS, int(width * scale)), max(MIN_PIXELS, int(height * scale))


def make_base_image(img: Image.Image, base_max_side: int, resample: int) -> Image.Image:
    """Pre-shrink an RGB image so that max(W, H) == base_max_side."""
    base_size = compute_base_size(img.width, img.height, base_max_side)
    if base_size == img.size:
        return img.copy()
    return img.resize(base_size, resample)


def degrade(img: Image.Image, scale: float, resample: int) -> Image.Image:
    """Downscale by ``scale`` then upscale back to the input size.

    Mirrors scripts/02_generate_resolution_levels.py: scale >= 1.0 is a plain
    copy; all levels are later JPEG-encoded identically by the caller.
    """
    if scale >= FULL_SCALE:
        return img.copy()
    small = img.resize(effective_size(img.width, img.height, scale), resample)
    return small.resize(img.size, resample)


def level_name(level_index: int) -> str:
    """1-based level index -> 'level_XX'."""
    return LEVEL_NAME_FORMAT.format(level_index)


def scales_equal(a: Sequence[float], b: Sequence[float], tol: float = 1e-12) -> bool:
    """Element-wise float equality of two scale arrays."""
    return len(a) == len(b) and all(
        abs(x - y) <= tol for x, y in zip(a, b, strict=True)
    )
