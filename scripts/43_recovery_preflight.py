"""ROBOVIS2027 recovery plan §1 preflight (read-only): frozen configs, model hashes, data manifest.

Writes into outputs/robovis2027_recovery/<run_id>/ (never touches earlier results):
  frozen_config.json   config files (content + sha256), resolved PGSG generation, batching rule
  model_hashes.json    PGSG / EGTR checkpoint sha256, submodule commits, ollama model digests
  data_manifest.json   image-id sets (dev20, test100, MIRU100 incl. history, held-out96,
                       pilot/smoke sets), pairwise overlaps, GT / level-manifest hashes,
                       file coverage of the cached artefacts
Re-runnable; only reads files, git metadata and ollama /api/tags.

Usage:
  uv run --project egtr python scripts/43_recovery_preflight.py --run-id rec_r0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.robovis_data import sha256_file  # noqa: E402

OUT_ROOT = REPO_ROOT / "outputs/robovis2027_recovery"
CONFIG_FILES = (
    "configs/robovis2027.yaml",
    "configs/config.yaml",
    "configs/config_heldout.yaml",
    "configs/pgsg/rel_det_vg_eval_local.yaml",
)
ROBOVIS_SELECTED = "data/processed/robovis2027/selected_images.json"
MIRU_SELECTED = "data/processed/selected_images.json"
HELDOUT_SELECTED = "data/processed/robovis2027_miru/selected_images.json"
PGSG_PILOT_IDS = "data/processed/pgsg_vg/selected_test_ids.json"
MIRU_HISTORY_REFS = (
    "e1b8e56",
    "2b038b2^",
    "2b038b2",
    "archive/query-conditioned-wip",
    "origin/feature/query-conditioned-bayesian-stop",
)
DEV_MANIFEST = "outputs/robovis2027/dev_r0/manifest.json"
PGSG_CKPT = "Pix2Grp_CVPR2024/PGSG-CVPR2024/vg_sgg.pth"
OLLAMA_TAGS = "http://localhost:11434/api/tags"
OLLAMA_MODELS = ("qwen3.5:4b", "qwen3.5:9b")
HTTP_TIMEOUT_S = 10


def run(*cmd: str) -> str:
    return subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


def ids_sha(ids: list[int]) -> str:
    return hashlib.sha256(",".join(str(i) for i in sorted(ids)).encode()).hexdigest()


def git_json(ref: str, path: str) -> Any | None:
    out = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        json.loads(out.stdout) if out.returncode == 0 and out.stdout.strip() else None
    )


def id_sets() -> dict[str, dict[str, Any]]:
    robovis = json.loads((REPO_ROOT / ROBOVIS_SELECTED).read_text())
    sets: dict[str, dict[str, Any]] = {
        "robovis_dev20": {
            "ids": [
                r["image_id"]
                for r in sorted(robovis, key=lambda r: r["rank"])
                if r["split"] == "dev"
            ],
            "source": f"{ROBOVIS_SELECTED} split=dev (rank order)",
        },
        "robovis_test100": {
            "ids": [
                r["image_id"]
                for r in sorted(robovis, key=lambda r: r["rank"])
                if r["split"] == "test"
            ],
            "source": f"{ROBOVIS_SELECTED} split=test (rank order)",
        },
        "miru100_current": {
            "ids": json.loads((REPO_ROOT / MIRU_SELECTED).read_text()),
            "source": MIRU_SELECTED,
        },
        "heldout96": {
            "ids": json.loads((REPO_ROOT / HELDOUT_SELECTED).read_text()),
            "source": HELDOUT_SELECTED,
        },
        "pgsg_official_eval_pilot": {
            "ids": json.loads((REPO_ROOT / PGSG_PILOT_IDS).read_text()),
            "source": f"{PGSG_PILOT_IDS} (first 1000 test.json images; aggregate PGSG validation runs)",
        },
    }
    for ref in MIRU_HISTORY_REFS:
        ids = git_json(ref, MIRU_SELECTED)
        if ids is not None:
            sets[f"miru_history@{ref}"] = {
                "ids": ids,
                "source": f"git show {ref}:{MIRU_SELECTED}",
            }
    for name, path in (
        ("robovis_smoke_a3", "outputs/robovis2027/smoke_a3/graphs"),
        ("miru_schedule_compare", "data/processed/schedule_compare/a_geometric"),
    ):
        p = REPO_ROOT / path
        if p.exists():
            sets[name] = {
                "ids": sorted(int(d.name) for d in p.iterdir() if d.name.isdigit()),
                "source": path,
            }
    for v in sets.values():
        v["n"] = len(v["ids"])
        v["sha256_sorted"] = ids_sha(v["ids"])
    return sets


def coverage() -> dict[str, Any]:
    def count(path: str, pattern: str = "*/level_*.json") -> int:
        p = REPO_ROOT / path
        return (
            sum(1 for f in p.glob(pattern) if "_failures" not in f.parts)
            if p.exists()
            else 0
        )

    return {
        "robovis_dev_r0_graphs": count("outputs/robovis2027/dev_r0/graphs"),
        "robovis_test_r0_graphs": count("outputs/robovis2027/test_r0/graphs"),
        "robovis_dev_r0_graphs_seed1": count("outputs/robovis2027/dev_r0/graphs_seed1"),
        "robovis_dev_r0_graphs_seed2": count("outputs/robovis2027/dev_r0/graphs_seed2"),
        "v4_checks_F1g_v4_G__9b": count(
            "outputs/robovis2027/dev_r0/checks/F1g_v4_G__9b"
        ),
        "v4_decisions_F1g_v4_G__9b": count(
            "outputs/robovis2027/dev_r0/decisions/F1g_v4_G__9b"
        ),
        "h3_checks_F1v_v3_G__9b_h3": count(
            "outputs/robovis2027/dev_r0/checks/F1v_v3_G__9b_h3"
        ),
        "h3_decisions_F1v_v3_G__9b_h3": count(
            "outputs/robovis2027/dev_r0/decisions/F1v_v3_G__9b_h3"
        ),
        "heldout_egtr_predictions": count("outputs/robovis2027_miru/egtr_predictions"),
        "heldout_text4b_scores": count(
            "outputs/robovis2027_miru/llm_scores_vlm/qwen3.5-4b_h5_rel_on_conf_on_text"
        ),
        "heldout_vlm9b_scores": count(
            "outputs/robovis2027_miru/vlm_scores/qwen3.5-9b/single"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-id", default="rec_r0")
    args = parser.parse_args()
    out = OUT_ROOT / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat(timespec="seconds")
    git = {
        "head": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": [ln[3:] for ln in run("git", "status", "--porcelain").splitlines()],
    }
    dev_manifest = json.loads((REPO_ROOT / DEV_MANIFEST).read_text())
    robovis_cfg = yaml.safe_load((REPO_ROOT / "configs/robovis2027.yaml").read_text())
    levels_manifest = REPO_ROOT / robovis_cfg["paths"]["levels_manifest"]

    frozen = {
        "generated_at": now,
        "git": git,
        "configs": {
            f: {
                "sha256": sha256_file(REPO_ROOT / f),
                "content": (REPO_ROOT / f).read_text(),
            }
            for f in CONFIG_FILES
        },
        "pgsg_generation_resolved": dev_manifest.get("generation_resolved"),
        "pgsg_graph_generation_block": dev_manifest.get("graph_generation_block"),
        "pgsg_batching": dev_manifest.get("batching"),
        "pgsg_seed0_batches": sorted(
            {tuple(v["batch_image_ids"]) for v in dev_manifest["states"].values()}
        ),
        "resolution": {
            "base_max_side": robovis_cfg["dataset"]["base_max_side"],
            "scales": robovis_cfg["resolution"]["scales"],
            "interpolation": robovis_cfg["resolution"]["interpolation"],
            "jpeg_quality": robovis_cfg["resolution"]["jpeg_quality"],
        },
    }
    egtr_dir = (
        REPO_ROOT
        / yaml.safe_load((REPO_ROOT / "configs/config.yaml").read_text())["paths"][
            "egtr_artifact"
        ]
        / "checkpoints"
    )
    try:
        tags = {
            m["name"]: m
            for m in requests.get(OLLAMA_TAGS, timeout=HTTP_TIMEOUT_S).json()["models"]
        }
    except (requests.RequestException, KeyError, ValueError):
        tags = {}
    models = {
        "generated_at": now,
        "pgsg": {
            "checkpoint": PGSG_CKPT,
            "sha256": sha256_file(REPO_ROOT / PGSG_CKPT),
            "sha256_in_dev_manifest": dev_manifest["checkpoint"]["sha256"],
            "submodule_commit": run(
                "git", "-C", "Pix2Grp_CVPR2024", "rev-parse", "HEAD"
            ),
        },
        "egtr": {
            "checkpoints": {
                p.name: sha256_file(p) for p in sorted(egtr_dir.glob("*.ckpt"))
            },
            "submodule_commit": run("git", "-C", "egtr", "rev-parse", "HEAD"),
        },
        "ollama": {
            m: (
                {"digest": tags[m]["digest"], "size": tags[m]["size"]}
                if m in tags
                else None
            )
            for m in OLLAMA_MODELS
        },
    }
    sets = id_sets()
    overlaps = {
        f"{a}&{b}": len(set(sets[a]["ids"]) & set(sets[b]["ids"]))
        for a, b in combinations(sets, 2)
    }
    data = {
        "generated_at": now,
        "gt": {
            "test.json_sha256": sha256_file(
                Path(robovis_cfg["dataset"]["vg_root"]) / "test.json"
            ),
            "rel.json_sha256": sha256_file(
                Path(robovis_cfg["dataset"]["vg_root"]) / "rel.json"
            ),
            "train.json_sha256": sha256_file(
                Path(robovis_cfg["dataset"]["vg_root"]) / "train.json"
            ),
        },
        "levels_manifest": {
            "path": str(levels_manifest.relative_to(REPO_ROOT)),
            "sha256": sha256_file(levels_manifest),
        },
        "id_sets": sets,
        "overlaps": {k: v for k, v in overlaps.items() if v},
        "coverage": coverage(),
    }
    for name, obj in (
        ("frozen_config.json", frozen),
        ("model_hashes.json", models),
        ("data_manifest.json", data),
    ):
        (out / name).write_text(json.dumps(obj, indent=1, default=list))
    print(
        json.dumps(
            {
                "git": git["head"],
                "dirty": len(git["dirty"]),
                "pgsg_sha_match": models["pgsg"]["sha256"]
                == models["pgsg"]["sha256_in_dev_manifest"],
                "sets": {k: v["n"] for k, v in sets.items()},
                "overlaps": data["overlaps"],
                "coverage": data["coverage"],
                "ollama": {
                    k: (v or {}).get("digest", "")[:12]
                    for k, v in models["ollama"].items()
                },
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
