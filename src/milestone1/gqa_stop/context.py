"""Run context shared by all CLI stages (config, paths, state, engine factory)."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from . import dataset, sampling
from .client import (
    PARSE_VERSION,
    ROLES,
    ModelSettings,
    OllamaTransport,
    PromptSet,
    RequestEngine,
    Transport,
    load_prompts,
    monitor_from_config,
    settings_from_config,
)
from .common import (
    RunPaths,
    RunState,
    StageError,
    append_worklog,
    atomic_write_json,
    atomic_write_text,
    read_json,
    read_jsonl,
    resolve_path,
    sha256_text,
)
from .images import ImageSeries
from .inference import InferenceRunner
from .preflight import digest_of, ollama_tags

TransportFactory = Callable[[ModelSettings], Transport]
LEVEL_ALL = "all"


def default_transport(settings: ModelSettings) -> Transport:
    return OllamaTransport(settings)


@dataclass
class Context:
    config_path: Path
    cfg: dict[str, Any]
    paths: RunPaths
    dry_run: bool = False
    transport_factory: TransportFactory = default_transport
    echo: Callable[[str], None] = print
    tags_get: Callable[..., Any] | None = None  # GET for /api/tags (tests inject)
    stage: str = ""
    _engine: RequestEngine | None = field(default=None, init=False, repr=False)

    def set_stage(self, name: str) -> None:
        """Label new records with the stage that created them (provenance)."""
        self.stage = name
        if self._engine is not None:
            self._engine.stage = name

    def refresh(self, cfg: dict[str, Any]) -> None:
        """Swap in a new effective config and drop derived caches."""
        self.cfg = cfg
        for name in ("pool", "strata", "settings", "series"):
            self.__dict__.pop(name, None)
        self._engine = None

    # ---- state / files --------------------------------------------------
    @cached_property
    def state(self) -> RunState:
        return RunState(self.paths.out("state.json"))

    @property
    def n_levels(self) -> int:
        return int(self.cfg["images"]["levels"])

    @property
    def seed(self) -> int:
        return int(self.cfg["seed"])

    def parse_levels(self, text: str | None, default: Sequence[int]) -> list[int]:
        if text is None:
            return list(default)
        if text == LEVEL_ALL:
            return list(range(1, self.n_levels + 1))
        levels = sorted({int(v) for v in text.split(",") if v.strip()})
        if not levels or levels[0] < 1 or levels[-1] > self.n_levels:
            raise StageError(f"levels must be within 1..{self.n_levels}: {text}")
        return levels

    def log(self, text: str) -> None:
        self.echo(text)
        if not self.dry_run:
            append_worklog(self.paths, text)

    # ---- data -----------------------------------------------------------
    @cached_property
    def questions(self) -> dict[str, Any]:
        return dataset.load_questions(self.paths.questions_file)

    @cached_property
    def pool(self) -> dataset.EligibilityResult:
        return dataset.build_eligible_pool(
            self.questions, self.cfg["dataset"]["filter"]
        )

    @cached_property
    def strata(self) -> sampling.Strata:
        return sampling.parse_strata(self.cfg["dataset"]["stratification"])

    def gt(self, question_ids: Sequence[str]) -> dict[str, str]:
        return {q: str(self.questions[q]["answer"]) for q in question_ids}

    def candidates(self, split: str | None = None) -> list[dict[str, Any]]:
        rows = read_jsonl(self.paths.data("candidate_manifest.jsonl"))
        return [r for r in rows if split is None or r["split"] == split]

    def selected(self, split: str) -> list[dict[str, Any]]:
        """Adopted rows of ``split``; restricted to ``paths.subset_file`` if set."""
        rows = read_jsonl(self.paths.selection("selected_manifest.jsonl"))
        rows = [r for r in rows if r["split"] == split]
        keep = self.subset_ids(split)
        if keep is None:
            return rows
        return [r for r in rows if r["question_id"] in keep]

    def subset_ids(self, split: str) -> set[str] | None:
        value = self.cfg["paths"].get("subset_file")
        if not value:
            return None
        path = resolve_path(self.paths.repo_root, value)
        if not path.exists():
            raise StageError(f"evaluation subset {path} missing: run `textprobe` first")
        return set(read_json(path)[split])

    # ---- model ----------------------------------------------------------
    @cached_property
    def settings(self) -> ModelSettings:
        return settings_from_config(self.cfg)

    @property
    def family(self) -> str:
        """Stop-method family: "confidence" (v1, plan §0-16) or "consistency" (§17)."""
        return str(self.cfg["policy"].get("family", "confidence"))

    @property
    def inherit_root(self) -> Path | None:
        """Output root of the run whose frozen split this run inherits (v2)."""
        value = self.cfg["paths"].get("inherit_split_from")
        return resolve_path(self.paths.repo_root, value) if value else None

    @property
    def split_frozen(self) -> bool:
        if self.state.get("prepare"):
            return True
        inherited = self.inherit_root is not None
        return inherited and self.paths.data("candidate_manifest.jsonl").exists()

    @cached_property
    def prompts(self) -> PromptSet:
        roles = self.cfg.get("prompts", {}).get("roles", list(ROLES))
        return load_prompts(self.paths.prompts_dir, roles)

    @cached_property
    def series(self) -> ImageSeries:
        return ImageSeries(
            self.cfg["images"],
            self.paths.images_root,
            self.paths.levels_dir,
            self.paths.data("image_manifest.jsonl"),
            read_only=self.inherit_root is not None,
        )

    def engine(self) -> RequestEngine:
        if self.dry_run:
            raise StageError("internal: engine requested during --dry-run")
        if self._engine is not None:
            return self._engine
        if not self.settings.digest:
            raise StageError("model digest unknown: run `preflight` first")
        self.verify_live_digest()
        self.record_prompts()
        self._engine = RequestEngine(
            self.settings,
            self.prompts,
            self.transport_factory(self.settings),
            self.paths.output_root,
            monitor_from_config(self.cfg),
            *self._reuse_settings(),
        )
        self._engine.stage = self.stage
        return self._engine

    def _reuse_settings(self) -> tuple[Path | None, frozenset[str], list]:
        """paths.reuse_records_from: one {output_root, roles} or a list of them."""
        reuse = self.cfg["paths"].get("reuse_records_from")
        if not reuse:
            return None, frozenset(), []
        entries = reuse if isinstance(reuse, list) else [reuse]
        sources = [
            (
                resolve_path(self.paths.repo_root, e["output_root"]),
                frozenset(e["roles"]),
            )
            for e in entries
        ]
        return None, frozenset(), sources

    def verify_live_digest(self) -> None:
        """The server's current digest for the model must equal the recorded one."""
        tags = ollama_tags(self.cfg["model"]["endpoint"], self.tags_get)
        if not tags["reachable"]:
            raise StageError("ollama /api/tags unreachable: cannot verify model digest")
        live = digest_of(tags, self.settings.name)
        if live != self.settings.digest:
            raise StageError(
                f"live digest of {self.settings.name} is {live}, recorded "
                f"{self.settings.digest}: the model changed; refusing to continue"
            )

    def note_reuse(self) -> None:
        """Worklog provenance note when cached records of another stage are reused."""
        if self._engine is None or not self._engine.reused_from:
            return
        origins = dict(self._engine.reused_from)
        probe = " (probe records: identical request keys)" if "probe" in origins else ""
        self.log(
            f"provenance: stage {self.stage} reused cached records created by "
            f"{origins}{probe}"
        )
        self._engine.reused_from.clear()

    def frozen_policy(self) -> dict[str, Any]:
        path = self.paths.out("frozen_policy.json")
        if not path.exists():
            raise StageError("no frozen policy: run `freeze` first")
        return read_json(path)

    def require_frozen_match(self, frozen: dict[str, Any] | None = None) -> None:
        """Test-side stages must use the frozen model digest and prompts."""
        frozen = self.frozen_policy() if frozen is None else frozen
        if frozen.get("digest") != self.settings.digest:
            raise StageError(
                f"active digest {self.settings.digest} != frozen {frozen.get('digest')}"
            )
        if frozen.get("model") != self.settings.name:
            raise StageError(
                f"active model {self.settings.name} != frozen {frozen.get('model')}"
            )
        if frozen.get("prompt_hashes") != self.prompts.hashes:
            raise StageError("prompt hashes differ from the frozen policy")

    def runner(self, cache_only: bool = False) -> InferenceRunner:
        checklist = self.cfg["checklist"]
        return InferenceRunner(
            self.engine(),
            self.series,
            int(checklist["min_items"]),
            int(checklist["max_items"]),
            cache_only=cache_only,
        )

    def record_prompts(self) -> None:
        """Copy prompts + hashes into the run; a change is a logged revision."""
        hashes = self.prompts.hashes
        hash_path = self.paths.out("prompt_hashes.json")
        previous = read_json(hash_path) if hash_path.exists() else None
        if previous is not None and previous.get("hashes") != hashes:
            self._guard_prompt_revision(previous)
        revisions = (previous or {}).get("revisions", 0)
        if previous is not None and previous.get("hashes") != hashes:
            revisions += 1
            self.log(f"prompt revision {revisions}: hashes changed")
        for name, text in self.prompts.texts.items():
            atomic_write_text(self.paths.out(f"prompts/{name}.txt"), text)
        old_parse = (previous or {}).get("parse_version", "parse-v1")
        if previous is not None and old_parse != PARSE_VERSION:
            self.log(
                f"re-namespace: parser {old_parse} -> {PARSE_VERSION} "
                "(a reply that is exactly one ```json fenced block is now unwrapped "
                "before json.loads; previously such replies were format failures "
                "and went through schema repair). New records live under "
                f"<role>/<model>@<digest>/{PARSE_VERSION}/ and the parser version "
                "is in the cache key, so no older record is reused; old records "
                "stay on disk for provenance."
            )
        atomic_write_json(
            hash_path,
            {
                "parse_version": PARSE_VERSION,
                "hashes": hashes,
                "revisions": revisions,
                "combined": sha256_text("".join(hashes[k] for k in sorted(hashes))),
            },
        )

    def _guard_prompt_revision(self, previous: dict[str, Any]) -> None:
        limit = int(self.cfg["execution"]["max_dev_prompt_revisions"])
        if self.paths.out("frozen_policy.json").exists():
            raise StageError("prompts changed after the policy was frozen")
        if previous.get("revisions", 0) >= limit:
            raise StageError(f"prompt revision limit ({limit}) reached")


def code_hash(modules: Sequence[Any]) -> str:
    sources = "".join(inspect.getsource(m) for m in modules)
    return sha256_text(sources)


def table_to_csv(table: pd.DataFrame, path: Path) -> None:
    out = table.copy()
    if "r_values" in out:
        out["r_values"] = out["r_values"].map(lambda v: _json_dumps(v))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(path)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StageError(message)
