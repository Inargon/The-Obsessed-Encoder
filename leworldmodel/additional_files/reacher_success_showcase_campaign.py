#!/usr/bin/env python3
"""Submit the PNG-only historical Reacher success showcase."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess


REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PYTHON = Path("/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python")
SOURCE = REPO / "leworldmodel/results/bloop-final-reacher-20261003-044551/reacher"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--partition", default="interactive")
    parser.add_argument("--nodelist", default="SPGL-1-1")
    args = parser.parse_args()
    required = [args.python, args.source / "reacher-historical.txt"]
    required.extend(args.source / f"rollout_{index}.mp4" for index in range(50))
    missing = [str(path) for path in required if not path.is_file()]
    plan = {
        "artifact": "top-2 successful Ours historical Reacher rollout PNGs",
        "source": str(args.source),
        "selection": "successful episodes ranked by initial fingertip-to-goal distance",
        "reruns_evaluation": False,
        "missing": missing,
    }
    print(json.dumps(plan, indent=2))
    if not args.submit:
        print("PLAN ONLY: pass --submit --confirm-run to launch.")
        return
    if not args.confirm_run:
        parser.error("refusing to submit without --confirm-run")
    if missing:
        parser.error("missing: " + ", ".join(missing))
    subprocess.run(
        [
            str(args.python), "-m", "pytest", "-q",
            str(HERE / "tests/test_reacher_success_showcase.py"),
        ],
        cwd=REPO / "leworldmodel",
        check=True,
    )
    campaign = REPO / "leworldmodel/results" / (
        "ours-reacher-success-showcase-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    campaign.mkdir(parents=True, exist_ok=False)
    command = shlex.join(
        str(value)
        for value in [
            args.python,
            HERE / "compose_reacher_success_showcase.py",
            "--source", args.source,
            "--out-dir", campaign / "artifacts",
            "--examples", 2,
        ]
    )
    shell = "\n".join(
        [
            "set -euo pipefail",
            f"export PYTHONPATH={shlex.quote(str(REPO / 'leworldmodel'))}:{shlex.quote(str(REPO))}",
            command,
            f"tar -czf {shlex.quote(str(campaign / 'ours-reacher-success-showcase.tar.gz'))} "
            f"-C {shlex.quote(str(campaign))} artifacts",
        ]
    )
    batch = [
        "sbatch", "--parsable", "--partition", args.partition,
        "--cpus-per-task=4", "--mem=8G", "--time=00:15:00",
        "--job-name=oe-reacher-showcase", "--chdir", str(REPO),
        "--output", str(campaign / "job-%j.out"), "--wrap", shell,
    ]
    if args.nodelist:
        batch[4:4] = ["--nodelist", args.nodelist]
    job = subprocess.check_output(batch, text=True).strip().split(";")[0]
    plan.update({"job_id": job, "campaign": str(campaign)})
    (campaign / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"OURS_REACHER_SHOWCASE_JOB={job}")
    print(f"CAMPAIGN={campaign}")


if __name__ == "__main__":
    main()
