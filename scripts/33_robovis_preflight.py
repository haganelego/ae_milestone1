"""ROBOVIS2027 preflight record (plan §3): environment, data, model and budget facts.

Writes outputs/robovis2027/preflight.json. Idempotent: re-running overwrites the
file with freshly measured values (only ``generated_at`` and volatile fields such
as free disk change). No secrets are read or stored (no environment variables,
no credentials); ollama is queried only for public model metadata.

Usage:
  uv run --project egtr python scripts/33_robovis_preflight.py --config configs/robovis2027.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.robovis_data import load_robovis_config, sha256_file  # noqa: E402

OUTPUT_NAME = "preflight.json"
CATEGORIES_PATH = REPO_ROOT / "data/processed/pgsg_vg/categories_dict.json"
PGSG_LICENSE = REPO_ROOT / "Pix2Grp_CVPR2024/LICENSE.txt"
ENVS = {"egtr": REPO_ROOT / "egtr", "pgsg": REPO_ROOT / "envs/pgsg"}
OLLAMA_MODELS = ("qwen3.5:4b", "qwen3.5:9b")
GENERATION_SOURCE_RUN = "dev_r0"
HTTP_TIMEOUT_S = 10
SUBPROCESS_TIMEOUT_S = 300
BYTES_PER_GIB = 1024**3
KIB = 1024
ENV_PROBE = (
    "import json, platform, sys\n"
    "out = {'python': platform.python_version()}\n"
    "try:\n"
    "    import torch\n"
    "    out.update(torch=torch.__version__, cuda=torch.version.cuda,\n"
    "               cudnn=torch.backends.cudnn.version(),\n"
    "               cuda_available=torch.cuda.is_available())\n"
    "except Exception as err:\n"
    "    out['torch_error'] = type(err).__name__\n"
    "print('PREFLIGHT ' + json.dumps(out))\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/robovis2027.yaml"
    )
    return parser.parse_args()


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> str:
    res = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
        check=False,
    )
    return res.stdout.strip()


def repo_info() -> dict[str, Any]:
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": [ln[3:] for ln in run(["git", "status", "--porcelain"]).splitlines()],
    }


def env_versions(project: Path) -> dict[str, Any]:
    out = run(["uv", "run", "--project", str(project), "python", "-c", ENV_PROBE])
    for line in out.splitlines():
        if line.startswith("PREFLIGHT "):
            return {
                "project": str(project.relative_to(REPO_ROOT)),
                **json.loads(line[len("PREFLIGHT ") :]),
            }
    return {"project": str(project.relative_to(REPO_ROOT)), "error": "probe failed"}


def hardware() -> dict[str, Any]:
    gpu = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for line in gpu.splitlines():
        name, mem, driver = (x.strip() for x in line.split(","))
        gpus.append({"name": name, "vram_mib": int(mem), "driver": driver})
    mem_kib = next(
        int(ln.split()[1])
        for ln in Path("/proc/meminfo").read_text().splitlines()
        if ln.startswith("MemTotal")
    )
    disk = shutil.disk_usage(REPO_ROOT)
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_gib": round(mem_kib * KIB / BYTES_PER_GIB, 1),
        "gpus": gpus,
        "disk_free_gib": round(disk.free / BYTES_PER_GIB, 1),
        "disk_total_gib": round(disk.total / BYTES_PER_GIB, 1),
    }


def vg_info(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg["dataset"]["vg_root"])
    split = cfg["dataset"]["source_split"]
    images = root / "images"
    return {
        "vg_root": str(root),
        f"{split}.json_sha256": sha256_file(root / f"{split}.json"),
        "rel.json_sha256": sha256_file(root / "rel.json"),
        "images_dir": str(images),
        "images_count": sum(1 for _ in os.scandir(images)),
    }


def dataset_info(cfg: dict[str, Any]) -> dict[str, Any]:
    sel_path = resolve(cfg["paths"]["selected_images"])
    lv_path = resolve(cfg["paths"]["levels_manifest"])
    selected = json.loads(sel_path.read_text())
    counts: dict[str, int] = {}
    for row in selected:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    deg = resolve(cfg["paths"]["degraded_images"])
    jpgs = sum(1 for p in deg.glob("*/*.jpg"))
    return {
        "selected_images": {
            "path": cfg["paths"]["selected_images"],
            "sha256": sha256_file(sel_path),
            "counts": counts,
        },
        "levels_manifest": {
            "path": cfg["paths"]["levels_manifest"],
            "sha256": sha256_file(lv_path),
            "n_images": len(json.loads(lv_path.read_text())["images"]),
        },
        "degraded_images": {
            "path": cfg["paths"]["degraded_images"],
            "n_jpg": jpgs,
            "expected": len(selected) * len(cfg["resolution"]["scales"]),
        },
    }


def pgsg_info(cfg: dict[str, Any]) -> dict[str, Any]:
    ckpt = resolve(cfg["pgsg"]["checkpoint"])
    graphs = sorted(
        (resolve(cfg["output_root"]) / GENERATION_SOURCE_RUN / "graphs").glob(
            "*/level_20.json"
        )
    )
    generation = json.loads(graphs[0].read_text())["generation"] if graphs else None
    manifest_path = (
        resolve(cfg["output_root"]) / GENERATION_SOURCE_RUN / "manifest.json"
    )
    resolved = (
        json.loads(manifest_path.read_text()).get("generation_resolved")
        if manifest_path.exists()
        else None
    )
    return {
        "submodule_commit": run(["git", "-C", "Pix2Grp_CVPR2024", "rev-parse", "HEAD"]),
        "checkpoint": {
            "path": cfg["pgsg"]["checkpoint"],
            "filename": ckpt.name,
            "sha256": sha256_file(ckpt),
        },
        "eval_config": {
            "path": cfg["pgsg"]["eval_config"],
            "sha256": sha256_file(resolve(cfg["pgsg"]["eval_config"])),
        },
        "categories_dict": {
            "path": str(CATEGORIES_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_file(CATEGORIES_PATH),
        },
        "license": {
            "path": str(PGSG_LICENSE.relative_to(REPO_ROOT)),
            "first_line": PGSG_LICENSE.read_text().splitlines()[0].strip(),
        },
        "generation_graph_block": generation,
        "generation_resolved": resolved,
        "generation_source": f"outputs/robovis2027/{GENERATION_SOURCE_RUN} (graph JSON + manifest)",
    }


def ollama_info(cfg: dict[str, Any]) -> dict[str, Any]:
    endpoint = cfg["llm"]["endpoint"]
    parts = urlsplit(endpoint)
    base = f"{parts.scheme}://{parts.netloc}"
    info: dict[str, Any] = {
        "endpoint": endpoint,
        "think": cfg["llm"].get("think"),
        "checks": cfg.get("checks"),
    }
    try:
        tags = requests.get(f"{base}/api/tags", timeout=HTTP_TIMEOUT_S).json()["models"]
        version = (
            requests.get(f"{base}/api/version", timeout=HTTP_TIMEOUT_S)
            .json()
            .get("version")
        )
    except (requests.RequestException, KeyError, ValueError) as err:
        return {**info, "error": f"ollama unreachable: {type(err).__name__}"}
    by_name = {m["name"]: m for m in tags}
    info["ollama_version"] = version
    info["models"] = {
        name: (
            {
                "digest": by_name[name]["digest"],
                "size_bytes": by_name[name]["size"],
                "details": by_name[name].get("details"),
            }
            if name in by_name
            else None
        )
        for name in OLLAMA_MODELS
    }
    return info


def budget_info(cfg: dict[str, Any]) -> dict[str, Any]:
    ex = cfg.get("execution", {})
    return {
        "max_model_calls": ex.get("max_model_calls"),
        "max_gpu_hours": ex.get("max_gpu_hours"),
        "concurrency": ex.get("concurrency"),
        "schema_retries": ex.get("schema_retries"),
        "transient_retries": ex.get("transient_retries"),
    }


def main() -> None:
    args = parse_args()
    cfg = load_robovis_config(args.config)
    out_path = resolve(cfg["output_root"]) / OUTPUT_NAME
    record = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/33_robovis_preflight.py",
        "config": {
            "path": str(args.config),
            "sha256": sha256_file(args.config),
            "spec_version": cfg.get("spec_version"),
        },
        "repo": repo_info(),
        "envs": {name: env_versions(path) for name, path in ENVS.items()},
        "hardware": hardware(),
        "vg": vg_info(cfg),
        "dataset": dataset_info(cfg),
        "pgsg": pgsg_info(cfg),
        "ollama": ollama_info(cfg),
        "budget": budget_info(cfg),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(tmp, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
