#!/usr/bin/env python3
"""Record fixed-group successful Bloop rollouts for paper and web figures."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess


REPO = Path(__file__).resolve().parents[2]
PYTHON = Path("/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python")
DATA_ROOT = Path("/grp01/ids_compcog/song/swm")
TASKS = {
    "clean_pusht": {
        "run": "interface-cycle-20261003-075323_clean_pusht_bloop_cycle_full_seed0",
        "evaluator": "evaluate_pusht_checkpoint.py",
        "extra": [],
    },
    "tagged_pusht": {
        "run": "interface-cycle-20261002-064431_tagged_pusht_bloop_cycle_full_seed0",
        "evaluator": "evaluate_pusht_checkpoint.py",
        "extra": ["--tagged"],
    },
    "tworoom": {
        "run": "interface-cycle-20261002-203108_clean_tworoom_bloop_cycle_full_seed0",
        "evaluator": "evaluate_clean_checkpoint.py",
        "extra": ["--task", "tworoom"],
    },
    "cube": {
        "run": "interface-cycle-20261002-203108_clean_cube_bloop_cycle_full_seed0",
        "evaluator": "evaluate_clean_checkpoint.py",
        "extra": ["--task", "cube", "--dataset", "ogbench/cube_single_expert.h5"],
    },
}


def task_command(task: str, root: Path, python: Path) -> str:
    spec = TASKS[task]
    output = root / task
    parts = [
        python,
        REPO / "leworldmodel/additional_files" / spec["evaluator"],
        "--run-name", spec["run"],
        "--checkpoint", "weights_epoch_10.pt",
        "--num-eval", "50",
        "--seed", "42",
        "--video-dir", output / "all-rollouts",
        "--output", output / "summary.json",
        *spec["extra"],
    ]
    return shlex.join(str(value) for value in parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--partition", default="interactive")
    parser.add_argument("--nodelist", default="SPGL-1-1")
    args = parser.parse_args()

    missing = []
    for task, spec in TASKS.items():
        checkpoint = args.data_root / "checkpoints" / spec["run"] / "weights_epoch_10.pt"
        if not checkpoint.exists():
            missing.append(f"{task}: {checkpoint}")
    plan = {
        "method": "EMA-guided orthogonal gradient repair (Bloop checkpoint)",
        "artifact": "full task rollouts from the fixed seed-42 evaluation group",
        "tasks": list(TASKS),
        "episodes_per_task": 50,
        "selection_rule": "publish the first three successful episode indices; retain all rollouts",
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

    campaign = REPO / "leworldmodel/results" / (
        "bloop-success-rollouts-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    campaign.mkdir(parents=True, exist_ok=False)
    task_names = list(TASKS)
    commands = [task_command(task, campaign, args.python) for task in task_names]
    quoted_commands = " ".join(shlex.quote(command) for command in commands)
    wrapper = "\n".join([
        "set -euo pipefail",
        f"export PYTHONPATH={shlex.quote(str(REPO / 'leworldmodel'))}:{shlex.quote(str(REPO))}",
        f"export STABLEWM_HOME={shlex.quote(str(args.data_root))}",
        f"export LOCAL_DATASET_DIR={shlex.quote(str(args.data_root))}",
        "export MUJOCO_GL=egl",
        'export TMPDIR="/grp01/ids_compcog/song/tmp/aluo/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"',
        'mkdir -p "$TMPDIR"',
        f"tasks=({' '.join(shlex.quote(task) for task in task_names)})",
        f"commands=({quoted_commands})",
        'task="${tasks[$SLURM_ARRAY_TASK_ID]}"',
        'mkdir -p ' + shlex.quote(str(campaign)) + '/"$task"/all-rollouts',
        'eval "${commands[$SLURM_ARRAY_TASK_ID]}"',
        'echo "BLOOP_SUCCESS_ROLLOUT_COMPLETE task=$task"',
    ])
    batch = [
        "sbatch", "--parsable", "--partition", args.partition,
        f"--array=0-{len(task_names)-1}%2", "--gres=gpu:1",
        "--cpus-per-task=8", "--mem=48G", "--time=01:30:00",
        "--job-name=oe-bloop-rollout", "--chdir", str(REPO),
        "--output", str(campaign / "rollout-%A_%a.out"), "--wrap", wrapper,
    ]
    if args.nodelist:
        batch[4:4] = ["--nodelist", args.nodelist]
    job = subprocess.check_output(batch, text=True).strip().split(";")[0]
    plan["job_id"] = job
    plan["campaign"] = str(campaign)
    (campaign / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"BLOOP_SUCCESS_ROLLOUT_JOB={job}")
    print(f"CAMPAIGN={campaign}")


if __name__ == "__main__":
    main()
