#!/usr/bin/env python3
"""Submit paper-style latent geometry visualizations for tagged PushT."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess

from additional_files.bloop_tagged_diagnostics_campaign import (
    CHECKPOINTS,
    DEFAULT_DATA_ROOT,
    DEFAULT_PYTHON,
    REPO,
    checkpoint_path,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--partition", default="interactive")
    parser.add_argument("--nodelist", default="SPGL-1-1")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--gallery-size", type=int, default=4096)
    parser.add_argument("--pairs", type=int, default=48)
    parser.add_argument("--examples", type=int, default=6)
    args = parser.parse_args()

    required = [args.python, args.data_root / "datasets/pusht_expert_train.h5"]
    required.extend(checkpoint_path(args.data_root, spec) for spec in CHECKPOINTS)
    missing = [str(path) for path in required if not path.exists()]
    plan = {
        "benchmark": "tagged PushT",
        "artifact": "LeJEPA-inspired latent interpolation, retrieval, geometry, and nuisance visualizations",
        "checkpoints": list(CHECKPOINTS),
        "protocol": {
            "gallery_size": args.gallery_size,
            "matched_endpoint_pairs": args.pairs,
            "interpolation_steps": 8,
            "goal_offset": 25,
            "illustrative_pairs": args.examples,
            "seed": args.seed,
            "decoder": "1-NN retrieval from shared gallery",
            "reporting": "all-pair bootstrap CI plus declared illustrative selection",
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
        "bloop-latent-geometry-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    campaign.mkdir(parents=True, exist_ok=False)
    checkpoint_args = []
    for spec in CHECKPOINTS:
        checkpoint_args.extend(("--checkpoint", spec))
    command = shlex.join(str(value) for value in [
        args.python,
        REPO / "leworldmodel/additional_files/diagnose_latent_geometry.py",
        *checkpoint_args,
        "--dataset", "pusht_expert_train.h5",
        "--cache-dir", args.data_root,
        "--gallery-size", args.gallery_size,
        "--pairs", args.pairs,
        "--examples", args.examples,
        "--steps", 8,
        "--goal-offset", 25,
        "--seed", args.seed,
        "--out-dir", campaign / "artifacts",
    ])
    wrapper = "\n".join([
        "set -euo pipefail",
        f"export PYTHONPATH={shlex.quote(str(REPO / 'leworldmodel'))}:{shlex.quote(str(REPO))}",
        f"export LOCAL_DATASET_DIR={shlex.quote(str(args.data_root))}",
        f"export STABLEWM_HOME={shlex.quote(str(args.data_root))}",
        "export MUJOCO_GL=egl",
        'export TMPDIR="/grp01/ids_compcog/song/tmp/aluo/${SLURM_JOB_ID}"',
        'mkdir -p "$TMPDIR"',
        command,
    ])
    batch = [
        "sbatch", "--parsable", "--partition", args.partition,
        "--gres=gpu:1", "--cpus-per-task=8", "--mem=64G", "--time=03:00:00",
        "--job-name=oe-bloop-geometry", "--chdir", str(REPO),
        "--output", str(campaign / "job-%j.out"), "--wrap", wrapper,
    ]
    if args.nodelist:
        batch[4:4] = ["--nodelist", args.nodelist]
    job = subprocess.check_output(batch, text=True).strip().split(";")[0]
    plan["job_id"] = job
    (campaign / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"BLOOP_LATENT_GEOMETRY_JOB={job}")
    print(f"CAMPAIGN={campaign}")


if __name__ == "__main__":
    main()
