"""Environment record (plan §2): repo, data, ollama models, disk, processes.

Only read-only probes: git, file hashes, ``GET /api/tags`` (no inference).
"""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
from PIL import Image

from .common import RunPaths, seeded_hash, sha256_file, utc_now
from .dataset import load_questions

TAGS_TIMEOUT_S = 10.0
IMAGE_SAMPLE_SIZE = 20
GIT_STATUS_LINES_SHOWN = 30
PACKAGES = ("numpy", "pandas", "pillow", "requests", "matplotlib", "pytest")
PROCESS_PATTERNS = ("ollama", "python")
EXCLUDED_PROCESS_MARKERS = ("shell-snapshots", "milestone1.gqa_stop.cli")
PROCESS_LINE_CHARS = 200
BYTES_PER_GB = 1024**3


def run_command(args: Sequence[str], cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            list(args), cwd=cwd, capture_output=True, text=True, check=False
        )
    except OSError as exc:
        return f"error: {exc}"
    return result.stdout.strip()


def git_summary(repo_root: Path) -> dict[str, Any]:
    status = run_command(["git", "status", "--porcelain"], repo_root).splitlines()
    return {
        "path": str(repo_root),
        "branch": run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root),
        "head": run_command(["git", "rev-parse", "HEAD"], repo_root),
        "status_entries": len(status),
        "modified_tracked": sum(1 for s in status if not s.startswith("??")),
        "untracked": sum(1 for s in status if s.startswith("??")),
        "status_head": status[:GIT_STATUS_LINES_SHOWN],
    }


def tags_url(chat_endpoint: str) -> str:
    parts = urlsplit(chat_endpoint)
    return f"{parts.scheme}://{parts.netloc}/api/tags"


def ollama_tags(
    chat_endpoint: str, get: Callable[..., Any] | None = None
) -> dict[str, Any]:
    url = tags_url(chat_endpoint)
    getter = get if get is not None else requests.get
    try:
        response = getter(url, timeout=TAGS_TIMEOUT_S)
        models = response.json().get("models", [])
    except (requests.RequestException, ValueError) as exc:
        return {"url": url, "reachable": False, "error": str(exc), "models": []}
    return {
        "url": url,
        "reachable": True,
        "models": [
            {
                "name": m.get("name"),
                "digest": m.get("digest"),
                "size": m.get("size"),
                "details": m.get("details"),
            }
            for m in models
        ],
    }


def digest_of(tags: Mapping[str, Any], name: str) -> str | None:
    for model in tags.get("models", []):
        if model.get("name") == name:
            return model.get("digest")
    return None


def image_sample(
    image_ids: Sequence[str], images_root: Path, seed: int
) -> dict[str, Any]:
    ordered = sorted(set(image_ids), key=lambda i: seeded_hash(seed, i))
    sample = ordered[:IMAGE_SAMPLE_SIZE]
    rows = []
    for image_id in sample:
        path = images_root / f"{image_id}.jpg"
        if not path.exists():
            rows.append({"image_id": image_id, "exists": False})
            continue
        with Image.open(path) as image:
            rows.append(
                {
                    "image_id": image_id,
                    "exists": True,
                    "size": list(image.size),
                    "mode": image.mode,
                }
            )
    missing = sum(1 for i in set(image_ids) if not (images_root / f"{i}.jpg").exists())
    return {
        "images_root_exists": images_root.exists(),
        "referenced_images": len(set(image_ids)),
        "missing_images": missing,
        "sample": rows,
    }


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def matching_processes() -> list[str]:
    lines = run_command(["ps", "-eo", "pid,etime,args"]).splitlines()[1:]
    return [
        line.strip()[:PROCESS_LINE_CHARS]
        for line in lines
        if any(p in line for p in PROCESS_PATTERNS)
        and not any(x in line for x in EXCLUDED_PROCESS_MARKERS)
    ]


def collect(
    cfg: Mapping[str, Any],
    paths: RunPaths,
    get: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    questions = load_questions(paths.questions_file)
    image_ids = [str(q["imageId"]) for q in questions.values()]
    tags = ollama_tags(cfg["model"]["endpoint"], get)
    model = cfg["model"]
    disk = shutil.disk_usage(paths.repo_root)
    return {
        "time": utc_now(),
        "repo": git_summary(paths.repo_root),
        "agents_files": sorted(
            str(p.relative_to(paths.repo_root))
            for p in [
                *paths.repo_root.glob("CLAUDE.md"),
                *paths.repo_root.glob("AGENTS.md"),
            ]
        ),
        "questions": {
            "path": str(paths.questions_file),
            "sha256": sha256_file(paths.questions_file),
            "count": len(questions),
            "version_note": "GQA v1.2 val_balanced (official answers public)",
        },
        "images": image_sample(image_ids, paths.images_root, int(cfg["seed"])),
        "ollama": tags,
        "model": {
            "name": model["name"],
            "digest": digest_of(tags, model["name"]),
            "fallback_name": model["fallback_name"],
            "fallback_digest": digest_of(tags, model["fallback_name"]),
            "image_preprocessing": (
                "client sends stored PNG bytes; model-internal image processor "
                "is not observable from the API (recorded as unknown)"
            ),
        },
        "disk": {"free_gb": round(disk.free / BYTES_PER_GB, 1)},
        "python": platform.python_version(),
        "packages": package_versions(),
        "processes": matching_processes(),
        "output_root": str(paths.output_root),
        "data_root": str(paths.data_root),
        "output_root_preexisting": paths.output_root.exists(),
    }
