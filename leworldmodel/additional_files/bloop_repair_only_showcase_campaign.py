#!/usr/bin/env python3
"""Submit a lightweight PNG-only JEPA-versus-repair rollout showcase."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess

from additional_files.bloop_tagged_diagnostics_campaign import DEFAULT_PYTHON, REPO

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = REPO / "leworldmodel/results/bloop-paper-visuals-20261005-055350"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--partition", default="interactive")
    parser.add_argument("--nodelist", default="SPGL-1-1")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()
    required = [
        args.python,
        args.source / "rollouts/jepa/summary.json",
        args.source / "rollouts/repair/summary.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    plan = {
        "artifact": "PNG-only matched repair-success/JEPA-failure showcase",
        "source_campaign": str(args.source),
        "examples": args.examples,
        "selection": "lowest repair terminal visible-goal residual among repair-only successes",
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
    test_env = dict(os.environ)
    test_env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO / "leworldmodel"), str(REPO), test_env.get("PYTHONPATH")])
    )
    subprocess.run(
        [
            str(args.python), "-m", "pytest", "-q",
            str(HERE / "tests/test_repair_only_showcase.py"),
        ],
        cwd=REPO,
        env=test_env,
        check=True,
    )
    campaign = REPO / "leworldmodel/results" / (
        "bloop-repair-only-showcase-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    campaign.mkdir(parents=True, exist_ok=False)
    command = shlex.join(
        str(value)
        for value in [
            args.python,
            HERE / "compose_repair_only_showcase.py",
            "--jepa-summary", args.source / "rollouts/jepa/summary.json",
            "--repair-summary", args.source / "rollouts/repair/summary.json",
            "--out-dir", campaign / "artifacts",
            "--examples", args.examples,
            "--columns", 6,
        ]
    )
    shell = "\n".join(
        [
            "set -euo pipefail",
            f"export PYTHONPATH={shlex.quote(str(REPO / 'leworldmodel'))}:{shlex.quote(str(REPO))}",
            command,
            (
                f"tar -czf {shlex.quote(str(campaign / 'repair-only-showcase.tar.gz'))} "
                f"-C {shlex.quote(str(campaign))} artifacts"
            ),
        ]
    )
    batch = [
        "sbatch", "--parsable", "--partition", args.partition,
        "--cpus-per-task=4", "--mem=16G", "--time=00:30:00",
        "--job-name=oe-repair-showcase", "--chdir", str(REPO),
        "--output", str(campaign / "job-%j.out"), "--wrap", shell,
    ]
    if args.nodelist:
        batch[4:4] = ["--nodelist", args.nodelist]
    job = subprocess.check_output(batch, text=True).strip().split(";")[0]
    plan.update({"job_id": job, "campaign": str(campaign)})
    (campaign / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"BLOOP_REPAIR_ONLY_SHOWCASE_JOB={job}")
    print(f"CAMPAIGN={campaign}")


if __name__ == "__main__":
    main()
