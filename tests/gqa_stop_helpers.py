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
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from milestone1.gqa_stop import stages_data, stages_model
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
FAIL_SCREEN_ANSWER = "glass"
FENCED_TEMPLATE = "```json\n{body}\n```"
V3_SEED_BASE = 20261008
PRIOR_SOLVE_LEVEL = 3  # fake text_only "knows" questions solved by L3  # reader always wrong -> screening failure
# v2 roles share the reader's first sentence, so they are recognised by a
# distinctive phrase first (checked before ROLE_PREFIX).
V2_ROLE_MARKERS = (
    ("without any image", "text_only"),
    ("Is there a ... in the image?", "planner_v3"),
    ('three keys "answer"', "observer_v3"),
    ("without seeing the image", "recompose"),
    ("object names as single common nouns", "identify"),
    ("Answer only from what is visible", "attr_blind"),
    ("You extract referring information", "planner_v2"),
)
SCENE_RE = re.compile(r"the (\w+)( on the table)? in scene (\d+)")
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

    def __init__(
        self,
        paths: RunPaths,
        answers_by_text: dict[str, str],
        raw_questions: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.paths = paths
        self.answers = answers_by_text
        self.by_image: dict[str, list[dict[str, Any]]] = {}
        for raw in (raw_questions or {}).values():
            ops = [op["operation"].split(" ")[0] for op in raw["semantic"]]
            self.by_image.setdefault(str(raw["imageId"]), []).append(
                {
                    "text": raw["question"],
                    "attribute": raw["semantic"][-1]["argument"],
                    "relate": "relate" in ops,
                    "object": raw["semantic"][0]["argument"].split(" (")[0],
                }
            )
        self.payloads: list[dict[str, Any]] = []
        self.calls: Counter[str] = Counter()
        self.broken: set[tuple[str, int]] = set()  # (role, level) -> invalid JSON

    def level_of(self, payload: dict[str, Any]) -> int | None:
        return self.image_level_of(payload)[1]

    def image_level_of(self, payload: dict[str, Any]) -> tuple[str | None, int | None]:
        images = payload["messages"][0].get("images")
        if not images:
            return None, None
        digest = hashlib.sha256(base64.b64decode(images[0])).hexdigest()
        for entry in read_jsonl(self.paths.data("image_manifest.jsonl")):
            for level in entry["levels"]:
                if level["sha256"] == digest:
                    return entry["image_id"], int(level["level"])
        raise AssertionError("image bytes not in manifest")

    def role_of(self, content: str) -> str:
        for marker, role in V2_ROLE_MARKERS:
            if marker in content:
                return role
        return next(r for p, r in ROLE_PREFIX.items() if content.startswith(p))

    def send(self, payload: dict[str, Any]) -> TransportResult:
        self.payloads.append(payload)
        content = payload["messages"][0]["content"]
        role = self.role_of(content)
        self.calls[role] += 1
        data = json.loads(content.split(INPUT_MARKER, 1)[1])
        image_id, level = self.image_level_of(payload)
        if (role, level) in self.broken:
            return TransportResult("not json", 1, 0.0)
        if role in {"planner_v2", "identify", "attr_blind", "recompose"}:
            return TransportResult(self.reply_v2(role, data, image_id, level), 1, 0.0)
        if role in {"text_only", "planner_v3", "observer_v3"}:
            seed = int(payload["options"]["seed"]) - V3_SEED_BASE
            return TransportResult(
                self.reply_v3(role, data, image_id, level, seed), 1, 0.0
            )
        reply = self.reply(role, data, level)
        return TransportResult(reply, 1, 0.0)

    def original_for(self, image_id: str, sub_question: str) -> dict[str, Any]:
        """The image's question a sub-question was derived from (fake bookkeeping)."""
        relate = "on the table" in sub_question
        attribute = next(
            (a for a in ("color", "material", "shape") if a in sub_question), None
        )
        options = [
            q
            for q in self.by_image[image_id]
            if q["relate"] == relate
            and (attribute is None or q["attribute"] == attribute)
        ]
        return (options or self.by_image[image_id])[0]

    def reply_v2(
        self, role: str, data: dict[str, Any], image_id: str | None, level: int | None
    ) -> str:
        if role == "planner_v2":
            text = data["question"]
            match = SCENE_RE.search(text)
            attribute = next(a for a in ("color", "material", "shape") if a in text)
            phrase = (
                f"the object that is on the table in scene {match.group(3)}"
                if match and match.group(2)
                else None
            )
            target = match.group(1) if match else "object"
            return json.dumps(
                {
                    "target": target,
                    "synonyms": [],
                    "attribute": attribute,
                    "referring_phrase": phrase,
                }
            )
        if role == "recompose":
            return json.dumps({"answer": data["observations"][1]["answer"]})
        original = self.original_for(str(image_id), data["question"])
        solved = level is not None and level >= solve_level(original["text"])
        if role == "identify":
            noun = original["object"]
            listing = data["question"].startswith("List")
            answer = (f"{noun}s, table" if listing else noun) if solved else "blob"
            return json.dumps({"answer": answer})
        gt = self.answers[original["text"]]
        return json.dumps(
            {"answer": gt if solved and gt != FAIL_SCREEN_ANSWER else "purple"}
        )

    def reply_v3(
        self,
        role: str,
        data: dict[str, Any],
        image_id: str | None,
        level: int | None,
        sample: int,
    ) -> str:
        if role == "text_only":
            text = data["question"]
            prior = solve_level(text) <= PRIOR_SOLVE_LEVEL
            return json.dumps({"answer": self.answers[text] if prior else "purple"})
        if role == "planner_v3":
            text = data["question"]
            match = SCENE_RE.search(text)
            obj, scene = (match.group(1), match.group(3)) if match else ("object", "0")
            attribute = next(a for a in ("color", "material", "shape") if a in text)
            return json.dumps(
                {
                    "items": [
                        {
                            "id": "c1",
                            "question": f"Is there a {obj} in scene {scene} in the image?",
                            "depends_on": [],
                        },
                        {
                            "id": "c2",
                            "question": f"What {attribute} is the {obj} in scene {scene}?",
                            "depends_on": ["c1"],
                        },
                    ]
                }
            )
        question = data["question"]
        original = self.original_for(str(image_id), question)
        solved = level is not None and level >= solve_level(original["text"])
        existence = question.startswith("Is there")
        truth = "yes" if existence else self.answers[original["text"]]
        if "scene" not in question and existence:  # negative probe noun
            truth = "no"
        if solved:
            return json.dumps({"answer": truth, "resolved": True, "confidence": 0.9})
        if sample % 2 == 0:  # samples 0, 2, 4 guess; 1, 3 abstain
            guess = "yes" if existence else "purple"
            return json.dumps({"answer": guess, "resolved": True, "confidence": 0.6})
        return json.dumps({"answer": "unknown", "resolved": False, "confidence": 0.0})

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
        # like qwen3.5:9b, the fake wraps checklist JSON in a ```json fence
        return FENCED_TEMPLATE.format(
            body=json.dumps(
                {
                    "items": [
                        {
                            "id": "c1",
                            "answer": "yes",
                            "resolved": True,
                            "confidence": 0.9,
                        },
                        {
                            "id": "c2",
                            "answer": answer if solved else "unknown",
                            "resolved": solved,
                            "confidence": 0.96 if solved else 0.0,
                        },
                    ]
                },
                indent=2,
            )
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


def run_v1_pipeline(tmp: Path) -> tuple[Path, dict[str, Any], dict[str, Any], Context]:
    """Synthetic v1 run through test inference (used by the v3 tests)."""
    config_path, cfg, questions = make_world(tmp)
    transport = FakeTransport(None, answers_by_text(questions), questions)
    ctx = make_context(config_path, cfg, transport)
    transport.paths = ctx.paths
    stages_model.preflight(ctx, get=fake_tags_get)
    stages_data.prepare(ctx)
    stages_data.screen(ctx, "dev")
    stages_model.pilot(ctx, [1, 5, 10, 15, 20], calibration=True, use_fallback=False)
    stages_model.infer(ctx, "dev", list(range(1, 21)))
    stages_model.freeze(ctx)
    stages_data.screen(ctx, "test")
    stages_model.infer(ctx, "test", list(range(1, 21)))
    return config_path, cfg, questions, ctx
