#!/usr/bin/env python3
"""Submit one CPU job to package successful-rollout GIFs for all tasks."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess

from additional_files.bloop_tagged_diagnostics_campaign import DEFAULT_PYTHON, REPO


RESULTS = REPO / "leworldmodel/results"
DEFAULT_MODERN = RESULTS / "bloop-success-rollouts-20261005-012438"
DEFAULT_REACHER = RESULTS / "bloop-final-reacher-20261003-044551/reacher"
TASKS = ("clean_pusht", "tagged_pusht", "tworoom", "cube")


def quote(values) -> str:
    return shlex.join(str(value) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--modern-source", type=Path, default=DEFAULT_MODERN)
    parser.add_argument("--reacher-source", type=Path, default=DEFAULT_REACHER)
    parser.add_argument("--partition", default="interactive")
    parser.add_argument("--nodelist", default="SPGL-1-1")
    args = parser.parse_args()
    required = [args.python, args.reacher_source / "reacher-historical.txt"]
    required.extend(args.modern_source / task / "summary.json" for task in TASKS)
    required.extend(args.reacher_source / f"rollout_{index}.mp4" for index in range(50))
    missing = [str(path) for path in required if not path.exists()]
    plan = {
        "artifact": "five successful Ours rollout GIFs",
        "reruns_models_or_evaluation": False,
        "outputs": [
            "tagged-pusht-success.gif",
            "clean-pusht-success.gif",
            "tworoom-success.gif",
            "cube-success.gif",
            "reacher-success.gif",
        ],
        "tagged_pusht": "two successful episodes with exact tag reconstruction and zoom",
        "other_tasks": "two selected simulator-labelled successes per GIF",
        "missing": missing,
    }
    print(json.dumps(plan, indent=2))
    if not args.submit:
        print("PLAN ONLY: pass --submit --confirm-run to launch.")
        return
    if not args.confirm_run:
        parser.error("refusing to submit without --confirm-run")
    if missing:
        parser.error("missing: " + "; ".join(missing))

    campaign = RESULTS / (
        "success-rollout-gifs-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    campaign.mkdir(parents=True, exist_ok=False)
    artifacts = campaign / "artifacts"
    command = quote([
        args.python,
        REPO / "leworldmodel/additional_files/compose_success_rollout_gifs.py",
        "--modern-source", args.modern_source,
        "--reacher-source", args.reacher_source,
        "--out-dir", artifacts,
        "--frames", 36,
        "--fps", 8,
    ])
    shell = "\n".join([
        "set -euo pipefail",
        f"export PYTHONPATH={shlex.quote(str(REPO / 'leworldmodel'))}:{shlex.quote(str(REPO))}",
        command,
        f"cp {shlex.quote(str(campaign / 'manifest.json'))} {shlex.quote(str(artifacts / 'campaign-manifest.json'))}",
        f"tar -czf {shlex.quote(str(campaign / 'success-rollout-gifs.tar.gz'))} "
        f"-C {shlex.quote(str(campaign))} artifacts",
        f"find {shlex.quote(str(artifacts))} -type f -printf '%f %s bytes\\n' | sort",
        'echo "SUCCESS_ROLLOUT_GIF_CAMPAIGN_COMPLETE"',
    ])
    batch = [
        "sbatch", "--parsable", "--partition", args.partition,
        "--cpus-per-task=4", "--mem=16G", "--time=00:30:00",
        "--job-name=oe-success-gifs", "--chdir", str(REPO),
        "--output", str(campaign / "job-%j.out"), "--wrap", shell,
    ]
    if args.nodelist:
        batch[4:4] = ["--nodelist", args.nodelist]
    (campaign / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    job = subprocess.check_output(batch, text=True).strip().split(";")[0]
    print(f"SUCCESS_ROLLOUT_GIFS_JOB={job}")
    print(f"CAMPAIGN={campaign}")


if __name__ == "__main__":
    main()
