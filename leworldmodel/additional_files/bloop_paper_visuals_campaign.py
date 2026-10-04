#!/usr/bin/env python3
"""Submit the statistical, mechanism, and matched-rollout Bloop figures."""

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


HERE = Path(__file__).resolve().parent
CHECKPOINTS = (
    "jepa=colored_square_episode_seed0/weights_epoch_10.pt",
    (
        "bloop=interface-cycle-20261002-064431_"
        "tagged_pusht_bloop_cycle_full_seed0/weights_epoch_10.pt"
    ),
)
RUNS = {
    "JEPA": "colored_square_episode_seed0",
    "EMA repair": (
        "interface-cycle-20261002-064431_"
        "tagged_pusht_bloop_cycle_full_seed0"
    ),
}
TRAINING_METRICS = {
    "tagged PushT": (
        "interface-cycle-20261002-064431/"
        "interface-cycle-20261002-064431_tagged_pusht_bloop_cycle_full_seed0/"
        "metrics.jsonl"
    ),
    "clean Reacher": (
        "interface-cycle-20261002-064431/"
        "interface-cycle-20261002-064431_clean_reacher_bloop_cycle_full_seed0/"
        "metrics.jsonl"
    ),
    "clean Cube": (
        "interface-cycle-20261002-203108/"
        "interface-cycle-20261002-203108_clean_cube_bloop_cycle_full_seed0/"
        "metrics.jsonl"
    ),
    "clean TwoRoom": (
        "interface-cycle-20261002-203108/"
        "interface-cycle-20261002-203108_clean_tworoom_bloop_cycle_full_seed0/"
        "metrics.jsonl"
    ),
    "clean PushT": (
        "interface-cycle-20261003-075323/"
        "interface-cycle-20261003-075323_clean_pusht_bloop_cycle_full_seed0/"
        "metrics.jsonl"
    ),
}


def quote(parts) -> str:
    return shlex.join(str(value) for value in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--partition", default="interactive")
    parser.add_argument("--nodelist", default="SPGL-1-1")
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()

    required = [args.python, args.data_root / "datasets/pusht_expert_train.h5"]
    for spec in CHECKPOINTS:
        _, location = spec.split("=", 1)
        required.append(args.data_root / "checkpoints" / location)
    for location in TRAINING_METRICS.values():
        required.append(REPO / "leworldmodel/results" / location)
    missing = [str(path) for path in required if not path.exists()]
    plan = {
        "benchmark": "tagged PushT plus five-task Bloop training logs",
        "comparison": "JEPA versus EMA orthogonal repair",
        "artifacts": [
            "128-trajectory paired feature-allocation statistics",
            "training-time EMA gradient geometry",
            "matched closed-loop tagged-PushT rollouts",
        ],
        "protocol": {
            "trajectory_seed": args.seed,
            "trajectories": 128,
            "steps_per_trajectory": 16,
            "rollout_seed": 42,
            "rollout_episodes_per_model": 50,
            "rollout_selection": "first indices by success-category; no visual selection",
        },
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

    campaign = REPO / "leworldmodel/results" / (
        "bloop-paper-visuals-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    campaign.mkdir(parents=True, exist_ok=False)
    checkpoint_args = []
    for spec in CHECKPOINTS:
        checkpoint_args.extend(("--checkpoint", spec))
    mechanism_args = []
    for label, location in TRAINING_METRICS.items():
        mechanism_args.extend((
            "--run",
            f"{label}={REPO / 'leworldmodel/results' / location}",
        ))
    commands = [
        quote([
            args.python,
            HERE / "diagnose_feature_allocation_visuals.py",
            *checkpoint_args,
            "--dataset", "pusht_expert_train.h5",
            "--cache-dir", args.data_root,
            "--frames", 1024,
            "--trajectories", 128,
            "--sweep-steps", 16,
            "--seed", args.seed,
            "--out-dir", campaign / "allocation",
        ]),
        quote([
            args.python,
            HERE / "plot_bloop_training_mechanism.py",
            *mechanism_args,
            "--out-dir", campaign / "mechanism",
        ]),
    ]
    for label, run_name in RUNS.items():
        directory = "jepa" if label == "JEPA" else "repair"
        commands.append(quote([
            args.python,
            HERE / "evaluate_pusht_checkpoint.py",
            "--run-name", run_name,
            "--checkpoint", "weights_epoch_10.pt",
            "--dataset", "pusht_expert_train.h5",
            "--tagged",
            "--num-eval", 50,
            "--seed", 42,
            "--video-dir", campaign / "rollouts" / directory / "all-rollouts",
            "--output", campaign / "rollouts" / directory / "summary.json",
        ]))
    commands.extend([
        quote([
            args.python,
            HERE / "compose_matched_rollouts.py",
            "--jepa-summary", campaign / "rollouts/jepa/summary.json",
            "--repair-summary", campaign / "rollouts/repair/summary.json",
            "--out-dir", campaign / "matched-rollouts",
            "--per-category", 2,
        ]),
        (
            f"tar -czf {shlex.quote(str(campaign / 'bloop-paper-visuals.tar.gz'))} "
            f"-C {shlex.quote(str(campaign))} allocation mechanism matched-rollouts manifest.json"
        ),
        'echo "BLOOP_PAPER_VISUALS_COMPLETE"',
    ])
    shell = "\n".join([
        "set -euo pipefail",
        f"export PYTHONPATH={shlex.quote(str(REPO / 'leworldmodel'))}:{shlex.quote(str(REPO))}",
        f"export STABLEWM_HOME={shlex.quote(str(args.data_root))}",
        f"export LOCAL_DATASET_DIR={shlex.quote(str(args.data_root))}",
        "export MUJOCO_GL=egl",
        'export TMPDIR="/grp01/ids_compcog/song/tmp/aluo/${SLURM_JOB_ID}"',
        'mkdir -p "$TMPDIR"',
        f"mkdir -p {shlex.quote(str(campaign / 'rollouts/jepa/all-rollouts'))}",
        f"mkdir -p {shlex.quote(str(campaign / 'rollouts/repair/all-rollouts'))}",
        *commands,
    ])
    batch = [
        "sbatch", "--parsable", "--partition", args.partition,
        "--gres=gpu:1", "--cpus-per-task=8", "--mem=64G", "--time=04:00:00",
        "--job-name=oe-bloop-paper-viz", "--chdir", str(REPO),
        "--output", str(campaign / "job-%j.out"), "--wrap", shell,
    ]
    if args.nodelist:
        batch[4:4] = ["--nodelist", args.nodelist]
    job = subprocess.check_output(batch, text=True).strip().split(";")[0]
    plan["job_id"] = job
    plan["campaign"] = str(campaign)
    (campaign / "manifest.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    print(f"BLOOP_PAPER_VISUALS_JOB={job}")
    print(f"CAMPAIGN={campaign}")


if __name__ == "__main__":
    main()
