"""Plan §25 (v6) shared pieces: the two dev sets and the v6 settings record.

Sets: dev visual subset (102, ``level_table_dev_merged.csv``) and the dev
reader-wrong set (99, ``ext_dev_wrong/level_table.csv``). Pilot questions are
the first 20 of each in question_id order (§25.5).

v6 run-time settings (second-reader digests, think handling, legibility
canonical token) are recorded in ``v6/effective_config_v6.json``. The run's
``effective_config.json`` is NOT rewritten: the frozen v22 policy stores its
sha256, and the paused test run checks it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from .common import atomic_write_json, deep_merge, read_id_csv, read_json, utc_now
from .context import Context, require

OUT_DIR = "v6"
EXT_DIR = "ext_dev_wrong"
V6_CONFIG = "effective_config_v6.json"
PILOT_QUESTIONS = 20
PILOT_LEVELS = (1, 5, 10, 15, 20)
HTTP_TIMEOUT_S = 60
# A new image normally encodes in < 2 s on the GPU; gemma3's vision encoder
# fell back to the CPU (~19 s) while qwen3.5:9b was resident (ollama log
# "CLIP using CPU backend"). Calls above this prompt-eval time are flagged.
CPU_CLIP_PROMPT_EVAL_S = 5.0


@dataclass(frozen=True)
class V6Sets:
    dev_table: pd.DataFrame
    wrong_table: pd.DataFrame
    dev_rows: dict[str, dict[str, Any]]
    wrong_rows: dict[str, dict[str, Any]]

    @property
    def dev_qids(self) -> list[str]:
        return sorted(self.dev_rows)

    @property
    def wrong_qids(self) -> list[str]:
        return sorted(self.wrong_rows)


def load_sets(ctx: Context) -> V6Sets:
    from .firing_v23 import wrong_questions

    root = ctx.paths.output_root
    dev_table = read_id_csv(root / "level_table_dev_merged.csv")
    wrong_table = read_id_csv(root / EXT_DIR / "level_table.csv")
    dev_rows = {str(r["question_id"]): r for r in ctx.selected("dev")}
    wrong_rows = {str(r["question_id"]): r for r in wrong_questions(ctx, "dev")}
    require(
        set(dev_table["question_id"]) == set(dev_rows),
        "dev level table and the dev visual subset differ",
    )
    require(
        set(wrong_table["question_id"]) == set(wrong_rows),
        "ext_dev_wrong level table and the reader-wrong set differ",
    )
    return V6Sets(dev_table, wrong_table, dev_rows, wrong_rows)


def v6_config(ctx: Context) -> dict[str, Any]:
    path = ctx.paths.out(OUT_DIR) / V6_CONFIG
    return read_json(path) if path.exists() else {}


def update_v6_config(ctx: Context, values: Mapping[str, Any], note: str) -> None:
    path = ctx.paths.out(OUT_DIR) / V6_CONFIG
    previous = read_json(path) if path.exists() else {}
    settings = deep_merge(previous.get("settings", {}), dict(values))
    history = [*previous.get("history", []), {"time": utc_now(), "note": note}]
    atomic_write_json(
        path,
        {
            "plan": "§25 (v6)",
            "primary_model": ctx.settings.name,
            "primary_digest": ctx.settings.digest,
            "decoding_options": dict(ctx.settings.decoding_options),
            "settings": settings,
            "history": history,
        },
    )


def loaded_models(endpoint: str) -> list[str]:
    url = endpoint.replace("/api/chat", "/api/ps")
    models = requests.get(url, timeout=HTTP_TIMEOUT_S).json().get("models", [])
    return [str(m.get("name") or m.get("model")) for m in models]


def isolate_model(endpoint: str, model_name: str) -> list[str]:
    """Unload every resident model, the target included (keep_alive 0), so that
    the target reloads alone and its vision encoder is placed on the GPU (a
    target loaded next to another model may keep a CPU encoder). Returns what
    was unloaded; ``model_name`` only documents the intended next model."""
    url = endpoint.replace("/api/chat", "/api/generate")
    others = loaded_models(endpoint)
    for name in others:
        requests.post(
            url, json={"model": name, "keep_alive": 0}, timeout=HTTP_TIMEOUT_S
        )
    return others


def prompt_eval_seconds(record: Mapping[str, Any]) -> float:
    """Non-generation time of the first attempt: total - eval duration (s).

    The engine stores total/eval durations only; the difference is prompt
    evaluation (image encoding included) plus any model load time, so the
    model is warmed up before a run (``warm_up``)."""
    attempts = record.get("attempts") or [{}]
    usage = attempts[0].get("usage") or {}
    total = float(usage.get("total_duration", 0))
    return max(total - float(usage.get("eval_duration", 0)), 0.0) / 1e9


def warm_up(endpoint: str, model_name: str) -> None:
    """Load ``model_name`` with a one-token text request (no record is written)."""
    requests.post(
        endpoint,
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "options": {"num_predict": 1},
        },
        timeout=HTTP_TIMEOUT_S * 5,
    )
