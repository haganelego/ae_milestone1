"""Synthetic GQA world + fake ollama transport for the gqa_stop tests.

No test touches the real dataset or ollama. The fake model "sees" the level
of an image by looking its sha256 up in the run's image manifest, and reads
the question from the JSON input data appended to the prompt - exactly what a
real model receives.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from milestone1.gqa_stop.client import TransportResult
from milestone1.gqa_stop.common import (
    RunPaths,
    build_paths,
    load_effective_config,
    read_jsonl,
)
from milestone1.gqa_stop.context import Context

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "gqa_stop.json"
PROMPTS_DIR = REPO_ROOT / "prompts" / "gqa_stop"
N_IMAGES = 48
DUPLICATE_PAIR = ("1046", "1047")  # identical bytes
COLORS = ("red", "blue", "green", "white")
MATERIALS = ("wood", "metal", "glass")
SHAPES = ("round", "square")
OBJECTS = ("cup", "box", "sign", "chair", "plate", "bag")
INPUT_MARKER = "Input data: "
FAIL_PLAN_WORD = "kite"  # planner returns broken JSON for these questions
FAIL_SCREEN_ANSWER = "glass"  # reader always wrong -> screening failure
ROLE_PREFIX = {
    "You answer a question": "answer_only",
    "Answer the question about": "direct",
    "You prepare a minimal": "planner",
    "Use the supplied image": "checklist",
}


def semantic(obj: str, attribute: str, relate: int) -> list[dict[str, Any]]:
    ops = [{"operation": "select", "dependencies": [], "argument": f"{obj} (1)"}]
    if relate:
        ops.append(
            {"operation": "relate", "dependencies": [0], "argument": "table,on,s (2)"}
        )
    ops.append(
        {"operation": "query", "dependencies": [len(ops) - 1], "argument": attribute}
    )
    return ops


def raw_question(
    image_id: str, question: str, answer: str, attribute: str, relate: int, obj: str
) -> dict[str, Any]:
    return {
        "question": question,
        "answer": answer,
        "imageId": image_id,
        "fullAnswer": f"The {obj} is {answer}.",
        "semantic": semantic(obj, attribute, relate),
        "types": {"structural": "query", "semantic": "attr", "detailed": "directOf"},
        "annotations": {"answer": {}, "question": {"3": "1"}, "fullAnswer": {"1": "1"}},
        "semanticStr": "select->query",
    }


def question_text(attribute: str, obj: str, relate: int, index: int) -> str:
    where = (" on the table" if relate else "") + f" in scene {index}"
    if attribute == "color":
        return f"What color is the {obj}{where}?"
    if attribute == "material":
        return f"What material is the {obj}{where} made of?"
    return f"What shape is the {obj}{where}?"


def make_questions() -> dict[str, dict[str, Any]]:
    """~100 eligible questions over 48 images plus excluded decoys."""
    questions: dict[str, dict[str, Any]] = {}
    qid = 100000
    for i in range(N_IMAGES):
        image_id = str(1000 + i)
        obj = OBJECTS[i % len(OBJECTS)]
        specs = [("color", i % 2, COLORS[i % len(COLORS)])]
        if i % 3 != 0:
            specs.append(("material", (i // 2) % 2, MATERIALS[i % len(MATERIALS)]))
        if i % 4 == 0:
            specs.append(("shape", (i // 4) % 2, SHAPES[i % len(SHAPES)]))
        if i % 7 == 0:
            specs.append(("color", 1 - i % 2, COLORS[(i + 1) % len(COLORS)]))
        for attribute, relate, answer in specs:
            target = FAIL_PLAN_WORD if (i == 5 and attribute == "color") else obj
            text = question_text(attribute, target, relate, i)
            if attribute == "color" and relate == 1 and i % 7 == 0:
                text = f"Which color is the {target} on the table in scene {i}?"
            questions[str(qid)] = raw_question(
                image_id, text, answer, attribute, relate, target
            )
            qid += 1
    questions.update(decoys())
    return questions


def decoys() -> dict[str, dict[str, Any]]:
    neg = raw_question(
        "1001", "What color is the cup that is not small?", "red", "color", 0, "cup"
    )
    neg["semantic"].insert(
        1, {"operation": "filter size", "dependencies": [0], "argument": "not(small)"}
    )
    neg["semantic"][-1]["dependencies"] = [1]
    yes = raw_question("1002", "Is the cup red?", "yes", "color", 0, "cup")
    yes["types"] = {
        "structural": "verify",
        "semantic": "attr",
        "detailed": "verifyAttr",
    }
    doing = raw_question(
        "1003", "What is the boy doing?", "brushing teeth", "color", 0, "boy"
    )
    evident = raw_question(
        "1004", "What color is the white cup?", "white", "color", 0, "cup"
    )
    name = raw_question("1005", "What is this?", "cup", "name", 0, "cup")
    return {
        "900001": neg,
        "900002": yes,
        "900003": doing,
        "900004": evident,
        "900005": name,
    }


def write_images(root: Path, size: tuple[int, int] = (40, 30)) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(N_IMAGES):
        image_id = str(1000 + i)
        if image_id == DUPLICATE_PAIR[1]:
            continue
        image = Image.new("RGB", size, ((i * 37) % 255, (i * 91) % 255, 120))
        draw = ImageDraw.Draw(image)
        draw.rectangle([2, 2, size[0] // 2, size[1] // 2], fill=(255, 255, 255))
        draw.line([0, size[1] - 1, size[0] - 1, 0], fill=(0, 0, 0))
        image.save(root / f"{image_id}.jpg", quality=95)
    shutil.copyfile(
        root / f"{DUPLICATE_PAIR[0]}.jpg", root / f"{DUPLICATE_PAIR[1]}.jpg"
    )


def small_config(tmp: Path) -> dict[str, Any]:
    cfg = json.loads(CONFIG_PATH.read_text())
    cfg = copy.deepcopy(cfg)
    cfg["paths"] = {
        "repo_root": str(tmp),
        "gqa_questions": str(tmp / "gqa" / "questions.json"),
        "gqa_images_root": str(tmp / "images"),
        "data_root": "data/run",
        "output_root": "outputs/run",
        "prompts_dir": str(PROMPTS_DIR),
    }
    dataset = cfg["dataset"]
    dataset.update(
        candidate_dev_questions=24,
        candidate_dev_images_scan=None,
        candidate_test_questions=40,
        candidate_test_images_scan=None,
        target_dev_questions=12,
        target_test_questions=16,
        minimum_dev_questions=4,
        minimum_dev_images=3,
        minimum_test_questions=4,
        minimum_test_images=3,
        reserve_batch_images=4,
        screening_repeat_dev_questions=3,
    )
    cfg["probe"]["questions_per_cell"] = 2
    cfg["execution"]["pilot_questions"] = 4
    cfg["evaluation"]["bootstrap_repeats"] = 40
    cfg["model"]["backoff_base_seconds"] = 0.0
    return cfg


def make_world(tmp: Path) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    questions = make_questions()
    (tmp / "gqa").mkdir(parents=True, exist_ok=True)
    (tmp / "gqa" / "questions.json").write_text(json.dumps(questions))
    write_images(tmp / "images")
    cfg = small_config(tmp)
    config_path = tmp / "gqa_stop.json"
    config_path.write_text(json.dumps(cfg, indent=1))
    return config_path, cfg, questions


def solve_level(text: str) -> int:
    """Deterministic first level at which the fake reader becomes correct."""
    return 1 + int(hashlib.sha256(text.encode()).hexdigest(), 16) % 14


class FakeTransport:
    """Pretends to be ollama; records every payload it receives."""

    def __init__(self, paths: RunPaths, answers_by_text: dict[str, str]) -> None:
        self.paths = paths
        self.answers = answers_by_text
        self.payloads: list[dict[str, Any]] = []
        self.calls: Counter[str] = Counter()
        self.broken: set[tuple[str, int]] = set()  # (role, level) -> invalid JSON

    def level_of(self, payload: dict[str, Any]) -> int | None:
        images = payload["messages"][0].get("images")
        if not images:
            return None
        digest = hashlib.sha256(base64.b64decode(images[0])).hexdigest()
        for entry in read_jsonl(self.paths.data("image_manifest.jsonl")):
            for level in entry["levels"]:
                if level["sha256"] == digest:
                    return int(level["level"])
        raise AssertionError("image bytes not in manifest")

    def send(self, payload: dict[str, Any]) -> TransportResult:
        self.payloads.append(payload)
        content = payload["messages"][0]["content"]
        role = next(r for p, r in ROLE_PREFIX.items() if content.startswith(p))
        self.calls[role] += 1
        data = json.loads(content.split(INPUT_MARKER, 1)[1])
        level = self.level_of(payload)
        if (role, level) in self.broken:
            return TransportResult("not json", 1, 0.0)
        reply = self.reply(role, data, level)
        return TransportResult(reply, 1, 0.0)

    def reply(self, role: str, data: dict[str, Any], level: int | None) -> str:
        text = data["question"]
        gt = self.answers[text]
        solved = level is not None and level >= solve_level(text)
        answer = gt if solved and gt != FAIL_SCREEN_ANSWER else "purple"
        if role == "answer_only":
            return json.dumps({"answer": answer})
        if role == "direct":
            return json.dumps(
                {"answer": answer, "confidence": min(1.0, level / 20 + 0.1)}
            )
        if role == "planner":
            if FAIL_PLAN_WORD in text:
                return "```json {broken"
            return json.dumps(
                {
                    "items": [
                        {"id": "c1", "question": "Find the target.", "depends_on": []},
                        {
                            "id": "c2",
                            "question": "What is its attribute?",
                            "depends_on": ["c1"],
                        },
                    ]
                }
            )
        return json.dumps(
            {
                "items": [
                    {"id": "c1", "answer": "yes", "resolved": True, "confidence": 0.9},
                    {
                        "id": "c2",
                        "answer": answer if solved else "unknown",
                        "resolved": solved,
                        "confidence": 0.96 if solved else 0.0,
                    },
                ]
            }
        )


def fake_tags_get(url: str, timeout: float) -> Any:
    class Response:
        def json(self) -> dict[str, Any]:
            return {
                "models": [
                    {"name": "qwen3.5:9b", "digest": "d9" * 32},
                    {"name": "qwen3.5:4b", "digest": "d4" * 32},
                ]
            }

    return Response()


def make_context(
    config_path: Path,
    cfg: dict[str, Any],
    transport: FakeTransport | None = None,
    dry_run: bool = False,
) -> Context:
    eff, paths = load_effective_config(config_path)
    factory = (lambda _s: transport) if transport is not None else (lambda _s: None)
    return Context(
        config_path,
        eff,
        paths,
        dry_run,
        factory,
        echo=lambda _t: None,
        tags_get=fake_tags_get,
    )


def answers_by_text(questions: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {q["question"]: q["answer"] for q in questions.values()}


def paths_for(cfg: dict[str, Any]) -> RunPaths:
    return build_paths(cfg)
