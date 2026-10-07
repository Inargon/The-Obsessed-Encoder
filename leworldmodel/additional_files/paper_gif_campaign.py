#!/usr/bin/env python3
"""Submit one lightweight job for the protocol and dense-selectivity GIFs."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess

from additional_files.bloop_tagged_diagnostics_campaign import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PYTHON,
    REPO,
)


DEFAULT_DENSE_JSON = REPO / (
    "leworldmodel/results/dense-planner-selectivity-20261006-084358/"
    "artifacts/dense-planner-selectivity.json"
)


def quote(values) -> str:
    return shlex.join(str(value) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dense-json", type=Path, default=DEFAULT_DENSE_JSON)
    parser.add_argument("--partition", default="interactive")
    parser.add_argument("--nodelist", default="SPGL-1-1")
    args = parser.parse_args()

    required = {
        "python": args.python,
        "fixed-goal dataset": args.data_root / "datasets/pusht_expert_train.h5",
        "random-goal dataset": args.data_root / "datasets/pusht_scripted_goal_train.lance",
        "dense diagnostic JSON": args.dense_json,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    plan = {
        "artifact": "paper-ready dataset protocol and dense selectivity GIFs",
        "source": "real PushT datasets and existing matched dense interventions",
        "outputs": [
            "pusht-protocol.gif",
            "dense-context-selectivity.gif",
            "dense-goal-selectivity.gif",
        ],
        "selection_disclosure": "largest JEPA-minus-Ours tag-allocation gaps",
        "missing": missing,
    }
    print(json.dumps(plan, indent=2))
    if not args.submit:
        print("PLAN ONLY: pass --submit --confirm-run to launch.")
        return
    if not args.confirm_run:
        parser.error("refusing to submit without --confirm-run")
    if missing:
        parser.error("missing required inputs: " + "; ".join(missing))

    campaign = REPO / "leworldmodel/results" / (
        "paper-gifs-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    campaign.mkdir(parents=True, exist_ok=False)
    artifacts = campaign / "artifacts"
    protocol_png = artifacts / "pusht-protocol.png"
    commands = [
        "set -euo pipefail",
        f"export PYTHONPATH={shlex.quote(str(REPO / 'leworldmodel'))}:{shlex.quote(str(REPO))}",
        f"export LOCAL_DATASET_DIR={shlex.quote(str(args.data_root))}",
        f"mkdir -p {shlex.quote(str(artifacts))}",
        quote([
            args.python,
            REPO / "leworldmodel/additional_files/graphs/dataset_montage.py",
            "--baseline-dataset", "pusht_expert_train.h5",
            "--randgoal-dataset", "pusht_scripted_goal_train.lance",
            "--out", protocol_png,
        ]),
        f"mv {shlex.quote(str(protocol_png.with_suffix('.gif')))} "
        f"{shlex.quote(str(artifacts / 'pusht-protocol.gif'))}",
        quote([
            args.python,
            REPO / "leworldmodel/additional_files/render_dense_selectivity_gif.py",
            "--input", args.dense_json,
            "--out-dir", artifacts,
            "--samples", 10,
            "--fps", 2,
            "--dpi", 100,
        ]),
        f"cp {shlex.quote(str(campaign / 'manifest.json'))} {shlex.quote(str(artifacts / 'manifest.json'))}",
        f"tar -czf {shlex.quote(str(campaign / 'paper-gifs.tar.gz'))} "
        f"-C {shlex.quote(str(campaign))} artifacts",
        f"find {shlex.quote(str(artifacts))} -type f -printf '%f %s bytes\\n' | sort",
        'echo "PAPER_GIFS_COMPLETE"',
    ]
    batch = [
        "sbatch", "--parsable",
        "--partition", args.partition,
        "--cpus-per-task=4", "--mem=20G", "--time=00:30:00",
        "--job-name=oe-paper-gifs",
        "--chdir", str(REPO),
        "--output", str(campaign / "job-%j.out"),
        "--wrap", "\n".join(commands),
    ]
    if args.nodelist:
        batch[4:4] = ["--nodelist", args.nodelist]
    (campaign / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    job = subprocess.check_output(batch, text=True).strip().split(";")[0]
    print(f"PAPER_GIFS_JOB={job}")
    print(f"CAMPAIGN={campaign}")


if __name__ == "__main__":
    main()
