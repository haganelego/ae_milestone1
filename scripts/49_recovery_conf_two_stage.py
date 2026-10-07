"""conf_two_stage comparator: freeze on dev, apply to dev and Phase D test (zero calls).

Usage:
    uv run --project egtr python scripts/49_recovery_conf_two_stage.py [--run-id rec_r0]

Level-agnostic, observation-based two-stage rule: at t10, compute the mean PGSG
entity score of the top-8 cleaned-G entities (same selection as the v4 U policy);
continue to t20 iff it is below tau. tau = dev median (dev seed0 t10), ties by the
GT-free SHA256 hash rule with p_tie keeping the dev continuation at half. A missing
graph is a failure -> t20. No GT is read.

Writes rec_r0/frozen_conf_policy.json, conf_dev_two_stage.csv, conf_test_policy.csv
(same columns as the U decision files: image_id, U (= confidence), valid,
decision_t, reason, extra_cost_failure).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milestone1.config import resolve  # noqa: E402
from milestone1.phase_d_eval import (  # noqa: E402
    CONF_TOP_K,
    OBSERVE,
    conf_decision,
    fit_conf_threshold,
    top_k_confidence,
)

OUT_ROOT = "outputs/robovis2027_recovery"
DEV_GRAPHS = "outputs/robovis2027/dev_r0/graphs_clean"
TEST_GRAPHS = (
    "outputs/robovis2027/test_rec/graphs_clean"  # seedA (same graphs as the U policy)
)
LEVEL_FILE = f"level_{OBSERVE:02d}.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-id", default="rec_r0")
    return p.parse_args()


def confidences(graph_root: Path, ids: list[int]) -> dict[int, float]:
    out = {}
    for i in ids:
        path = graph_root / str(i) / LEVEL_FILE
        out[i] = top_k_confidence(
            json.loads(path.read_text()) if path.exists() else None
        )
    return out


def decisions(conf: dict[int, float], tau: float, p_tie: float) -> pd.DataFrame:
    rows = []
    for i, c in conf.items():
        t, reason = conf_decision(i, c, tau, p_tie)
        rows.append({"image_id": i, "U": c, "valid": c == c, "decision_t": t,
                     "reason": reason, "extra_cost_failure": reason == "failure_continue"})  # fmt: skip
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    run = resolve(OUT_ROOT) / args.run_id
    dev_ids = [
        int(i)
        for i in json.loads((run / "frozen_policy.json").read_text())["source"][
            "image_ids"
        ]
    ]
    test_ids = [int(i) for i in pd.read_csv(run / "test_policy.csv")["image_id"]]
    dev_conf = confidences(resolve(DEV_GRAPHS), dev_ids)
    fit = fit_conf_threshold(list(dev_conf.values()))
    frozen = {
        "policy": "conf_two_stage", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "signal": f"mean PGSG entity score of top-{CONF_TOP_K} cleaned-G entities at t{OBSERVE}",
        "rule": "continue to t20 iff confidence < tau; ties: SHA256 hash < p_tie; missing -> t20",
        "fit": fit, "fit_source": {"graphs": DEV_GRAPHS, "level": OBSERVE, "image_ids": dev_ids},
        "gt_used": False, "level_agnostic": True,
    }  # fmt: skip
    (run / "frozen_conf_policy.json").write_text(json.dumps(frozen, indent=2))
    dev = decisions(dev_conf, fit["tau"], fit["p_tie"])
    test = decisions(
        confidences(resolve(TEST_GRAPHS), test_ids), fit["tau"], fit["p_tie"]
    )
    dev.to_csv(run / "conf_dev_two_stage.csv", index=False)
    test.to_csv(run / "conf_test_policy.csv", index=False)
    summary: dict[str, Any] = {
        "tau": fit["tau"], "p_tie": fit["p_tie"],
        "dev_continue": int((dev.decision_t == 20).sum()), "dev_n": len(dev),
        "test_continue": int((test.decision_t == 20).sum()), "test_n": len(test),
        "test_failures": int(test.extra_cost_failure.sum()),
    }  # fmt: skip
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
