#!/usr/bin/env python3
"""Submit JEPA-versus-Ours dense planner-selectivity visualization pilot."""

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
    checkpoint_path,
)


CHECKPOINTS = (
    "jepa=colored_square_episode_seed0/weights_epoch_10.pt",
    (
        "ours=interface-cycle-20261002-064431_"
        "tagged_pusht_bloop_cycle_full_seed0/weights_epoch_10.pt"
    ),
)


def quote(values) -> str:
    return shlex.join(str(value) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--partition", default="interactive")
    parser.add_argument("--nodelist", default="SPGL-1-1")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--num-clips", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=16)
    args = parser.parse_args()

    required = [args.python, args.data_root / "datasets/pusht_expert_train.h5"]
    required.extend(checkpoint_path(args.data_root, spec) for spec in CHECKPOINTS)
    missing = [str(path) for path in required if not path.exists()]
    plan = {
        "benchmark": "tagged PushT",
        "artifact": "JEPA-versus-Ours dense latent and planner selectivity maps",
        "checkpoints": list(CHECKPOINTS),
        "protocol": {
            "seed": args.seed,
            "clips": args.num_clips,
            "candidates": 32,
            "grid": [args.grid_size, args.grid_size],
            "interventions": ["local blur", "matched real donor patch"],
            "smoke_gate": "2 clips on a 2x2 grid in the same allocation before the full run",
            "outputs": [
                "context latent sensitivity",
                "context planning-cost sensitivity",
                "goal planning-cost sensitivity",
                "candidate pair-order sensitivity",
                "tag-region attribution mass with paired bootstrap intervals",
            ],
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
        "dense-planner-selectivity-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    campaign.mkdir(parents=True, exist_ok=False)
    checkpoint_args = []
    for spec in CHECKPOINTS:
        checkpoint_args.extend(("--checkpoint", spec))
    def diagnostic_command(out_dir: Path, clips: int, grid: int, patch_batch: int, examples: int) -> str:
        return quote([
            args.python,
            REPO / "leworldmodel/additional_files/diagnose_dense_planner_selectivity.py",
            *checkpoint_args,
            "--dataset",
            "pusht_expert_train.h5",
            "--out-dir",
            out_dir,
            "--seed",
            args.seed,
            "--num-clips",
            clips,
            "--candidates",
            32,
            "--history",
            3,
            "--horizon",
            5,
            "--frameskip",
            5,
            "--grid-size",
            grid,
            "--tag-size",
            5,
            "--patch-batch-size",
            patch_batch,
            "--examples",
            examples,
        ])

    common = [
            "set -euo pipefail",
            f"export PYTHONPATH={shlex.quote(str(REPO / 'leworldmodel'))}:{shlex.quote(str(REPO))}",
            f"export LOCAL_DATASET_DIR={shlex.quote(str(args.data_root))}",
            f"export STABLEWM_HOME={shlex.quote(str(args.data_root))}",
            "export MUJOCO_GL=egl",
            'export TMPDIR="/grp01/ids_compcog/song/tmp/aluo/${SLURM_JOB_ID}"',
            'mkdir -p "$TMPDIR"',
    ]
    full_wrap = "\n".join(
        common
        + [
            diagnostic_command(campaign / "smoke-artifacts", 2, 2, 2, 1),
            'echo "DENSE_PLANNER_SELECTIVITY_SMOKE_COMPLETE"',
            diagnostic_command(campaign / "artifacts", args.num_clips, args.grid_size, 4, 4),
            'tar -czf "' + str(campaign / "dense-planner-selectivity-visuals.tar.gz") + '" -C "' + str(campaign) + '" artifacts',
            'echo "DENSE_PLANNER_SELECTIVITY_CAMPAIGN_COMPLETE"',
        ]
    )
    batch = [
        "sbatch",
        "--parsable",
        "--partition",
        args.partition,
        "--gres=gpu:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=04:00:00",
        "--job-name=oe-dense-select",
        "--chdir",
        str(REPO),
        "--output",
        str(campaign / "job-%j.out"),
        "--wrap",
        full_wrap,
    ]
    if args.nodelist:
        batch[4:4] = ["--nodelist", args.nodelist]
    job = subprocess.check_output(batch, text=True).strip().split(";")[0]
    plan["job_id"] = job
    (campaign / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"DENSE_PLANNER_SELECTIVITY_JOB={job}")
    print(f"CAMPAIGN={campaign}")


if __name__ == "__main__":
    main()
