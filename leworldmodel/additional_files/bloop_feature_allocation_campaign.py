#!/usr/bin/env python3
"""Submit JEPA-versus-repair feature-allocation and planning visuals."""

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


SELECTED = tuple(
    spec
    for spec in CHECKPOINTS
    if spec.split("=", 1)[0] in {"jepa", "bloop"}
)


def quote(values):
    return shlex.join(str(value) for value in values)


def main():
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
    required.extend(checkpoint_path(args.data_root, spec) for spec in SELECTED)
    missing = [str(path) for path in required if not path.exists()]
    plan = {
        "benchmark": "tagged PushT",
        "comparison": "JEPA versus EMA orthogonal repair",
        "inspiration": "Enigma feature-suppression similarity and smooth-stimulus geometry",
        "checkpoints": list(SELECTED),
        "outputs": [
            "matched input/tag panel",
            "same-content versus same-tag similarity crossover",
            "content-sweep versus tag-sweep latent trajectories",
            "normalized latent path-length allocation",
            "three-row Oracle/JEPA/repair latent planning figures and GIFs",
        ],
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
        "bloop-feature-allocation-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    campaign.mkdir(parents=True, exist_ok=False)
    checkpoint_args = []
    for spec in SELECTED:
        checkpoint_args.extend(("--checkpoint", spec))
    common = [
        "set -euo pipefail",
        f"export PYTHONPATH={shlex.quote(str(REPO / 'leworldmodel'))}:{shlex.quote(str(REPO))}",
        f"export LOCAL_DATASET_DIR={shlex.quote(str(args.data_root))}",
        f"export STABLEWM_HOME={shlex.quote(str(args.data_root))}",
        "export MUJOCO_GL=egl",
        'export TMPDIR="/grp01/ids_compcog/song/tmp/aluo/${SLURM_JOB_ID}"',
        'mkdir -p "$TMPDIR"',
    ]
    commands = [
        quote([
            args.python,
            REPO / "leworldmodel/additional_files/diagnose_feature_allocation_visuals.py",
            *checkpoint_args,
            "--dataset", "pusht_expert_train.h5",
            "--cache-dir", args.data_root,
            "--frames", 1024,
            "--sweep-steps", 32,
            "--seed", args.seed,
            "--out-dir", campaign / "allocation",
        ]),
        quote([
            args.python,
            REPO / "leworldmodel/additional_files/diagnose_latent_geometry.py",
            *checkpoint_args,
            "--models", "jepa,bloop",
            "--dataset", "pusht_expert_train.h5",
            "--cache-dir", args.data_root,
            "--gallery-size", 4096,
            "--pairs", 48,
            "--examples", 6,
            "--steps", 8,
            "--goal-offset", 25,
            "--seed", args.seed,
            "--out-dir", campaign / "planning",
        ]),
        'echo "BLOOP_FEATURE_ALLOCATION_CAMPAIGN_COMPLETE"',
    ]
    batch = [
        "sbatch", "--parsable", "--partition", args.partition,
        "--gres=gpu:1", "--cpus-per-task=8", "--mem=64G", "--time=04:00:00",
        "--job-name=oe-bloop-allocation", "--chdir", str(REPO),
        "--output", str(campaign / "job-%j.out"), "--wrap", "\n".join(common + commands),
    ]
    if args.nodelist:
        batch[4:4] = ["--nodelist", args.nodelist]
    job = subprocess.check_output(batch, text=True).strip().split(";")[0]
    plan["job_id"] = job
    (campaign / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"BLOOP_FEATURE_ALLOCATION_JOB={job}")
    print(f"CAMPAIGN={campaign}")


if __name__ == "__main__":
    main()
