#!/usr/bin/env python3
"""Submit the no-retraining parameter-space interference audit."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

try:
    from .interface_cycle_campaign import environment
except ImportError:
    from interface_cycle_campaign import environment


REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN_NAME = (
    "interface-cycle-20261002-064431_tagged_pusht_"
    "bloop_cycle_full_seed0"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--partition", default="interactive")
    parser.add_argument("--nodelist", default="SPGL-1-1")
    args = parser.parse_args()
    if not args.submit:
        print("PLAN ONLY. Pass --submit --confirm-run on the cluster.")
        return
    if not args.confirm_run:
        parser.error("submission requires --confirm-run")

    root = REPO / "leworldmodel/results" / (
        "parameter-space-interference-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=False)
    artifact = root / "artifacts" / "parameter-space-interference.json"
    command = [
        sys.executable,
        str(HERE / "diagnose_parameter_space_interference.py"),
        "--run-name",
        RUN_NAME,
        "--checkpoint",
        "weights_epoch_10.pt",
        "--num-batches",
        "16",
        "--batch-size",
        "8",
        "--output",
        str(artifact),
    ]
    wrapped = "\n".join(
        (
            "set -euo pipefail",
            shlex.join(command),
            f"tar -czf {shlex.quote(str(root / 'parameter-space-interference.tar.gz'))} "
            f"-C {shlex.quote(str(root))} artifacts",
        )
    )
    batch = [
        "sbatch",
        "--parsable",
        "--partition",
        args.partition,
        "--nodelist",
        args.nodelist,
        "--gres=gpu:1",
        "--cpus-per-task=8",
        "--mem=48G",
        "--time=01:00:00",
        "--job-name=oe-param-audit",
        "--chdir",
        str(REPO),
        "--output",
        str(root / "job-%j.out"),
        "--wrap",
        wrapped,
    ]
    job = subprocess.check_output(batch, text=True, env=environment()).strip().split(";")[0]
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "job": job,
                "checkpoint": RUN_NAME + "/weights_epoch_10.pt",
                "retraining": False,
                "command": command,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"PARAMETER_SPACE_INTERFERENCE_JOB={job}")
    print(f"CAMPAIGN={root}")


if __name__ == "__main__":
    main()
