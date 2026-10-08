"""Shared helpers: hashing, atomic file IO, config/paths, run state, worklog."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

HASH_CHUNK_BYTES = 1 << 20
SPLITS = ("dev", "test")


class StageError(RuntimeError):
    """A command was invoked in a state the protocol forbids."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for hashing (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seeded_hash(seed: int, key: str) -> str:
    """SHA256(f"{seed}|{key}") used for every deterministic ordering (plan §3.2)."""
    return sha256_text(f"{seed}|{key}")


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write to a temp file in the same directory, fsync, then rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def _json_default(obj: Any) -> Any:
    """numpy scalars / arrays -> Python values."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def atomic_write_json(path: str | Path, obj: Any) -> None:
    text = json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default)
    atomic_write_text(path, text + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    atomic_write_text(path, "".join(line + "\n" for line in lines))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    return [json.loads(line) for line in iter_lines(target)]


def iter_lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                yield stripped


ID_COLUMNS = ("question_id", "image_id", "group_id")


def read_id_csv(path: str | Path) -> pd.DataFrame:
    """pandas CSV reader that keeps id columns as strings (GQA ids have
    leading zeros, e.g. "05515938")."""
    return pd.read_csv(path, dtype=dict.fromkeys(ID_COLUMNS, str))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict where ``override`` is merged recursively into ``base``."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


@dataclass(frozen=True)
class RunPaths:
    """All on-disk locations of one run (plan §9.1)."""

    repo_root: Path
    data_root: Path
    output_root: Path
    questions_file: Path
    images_root: Path
    prompts_dir: Path
    selection_root: Path  # selection_results / selected_manifest (default data_root)

    @property
    def levels_dir(self) -> Path:
        return self.data_root / "levels"

    def out(self, name: str) -> Path:
        return self.output_root / name

    def data(self, name: str) -> Path:
        return self.data_root / name

    def selection(self, name: str) -> Path:
        return self.selection_root / name


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def build_paths(cfg: dict[str, Any], repo_root: Path | None = None) -> RunPaths:
    paths = cfg["paths"]
    root = repo_root if repo_root is not None else Path(paths["repo_root"])
    prompts = paths.get("prompts_dir", "prompts/gqa_stop")
    return RunPaths(
        repo_root=root,
        data_root=resolve_path(root, paths["data_root"]),
        output_root=resolve_path(root, paths["output_root"]),
        questions_file=resolve_path(root, paths["gqa_questions"]),
        images_root=resolve_path(root, paths["gqa_images_root"]),
        prompts_dir=resolve_path(root, prompts),
        selection_root=resolve_path(
            root, paths.get("selection_root", paths["data_root"])
        ),
    )


def load_config(path: str | Path) -> dict[str, Any]:
    return read_json(path)


EFFECTIVE_CONFIG = "effective_config.json"


def load_effective_config(
    config_path: str | Path, repo_root: Path | None = None
) -> tuple[dict[str, Any], RunPaths]:
    """Source config merged with overrides recorded in effective_config.json.

    The source file is never edited; run-time decisions (model digest, model
    switch at pilot, the single probe-based dataset revision) live in
    ``outputs/.../effective_config.json`` under ``overrides``.
    """
    source = load_config(config_path)
    paths = build_paths(source, repo_root)
    eff_path = paths.out(EFFECTIVE_CONFIG)
    if not eff_path.exists():
        return source, paths
    overrides = read_json(eff_path).get("overrides", {})
    return deep_merge(source, overrides), paths


def write_effective_config(
    config_path: str | Path,
    paths: RunPaths,
    new_overrides: dict[str, Any],
    note: str,
) -> dict[str, Any]:
    """Merge ``new_overrides`` into the recorded overrides and rewrite the file."""
    source = load_config(config_path)
    eff_path = paths.out(EFFECTIVE_CONFIG)
    previous = read_json(eff_path) if eff_path.exists() else {}
    overrides = deep_merge(previous.get("overrides", {}), new_overrides)
    history = [*previous.get("history", []), {"time": utc_now(), "note": note}]
    record = {
        "source_config": str(config_path),
        "source_config_sha256": sha256_file(config_path),
        "overrides": overrides,
        "history": history,
        "effective": deep_merge(source, overrides),
    }
    atomic_write_json(eff_path, record)
    return record["effective"]


class RunState:
    """Small JSON state machine (``state.json``) guarding stage order."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = read_json(path) if path.exists() else {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data = {**self.data, key: value, "updated": utc_now()}
        atomic_write_json(self.path, self.data)


def append_worklog(paths: RunPaths, text: str) -> None:
    log = paths.out("worklog.md")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"- {utc_now()} {text}\n")
