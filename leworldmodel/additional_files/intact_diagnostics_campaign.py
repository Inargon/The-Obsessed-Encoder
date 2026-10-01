#!/usr/bin/env python3
"""Submit representation and planning diagnostics for tagged INTACT PushT.

The checkpoint is frozen.  One GPU job runs, in order, the same physical/tag
linear probe, counterfactual scene--tag geometry, and fixed-candidate tag
intervention used for the other tagged-PushT methods in this repository.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess


REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PINNED_INTACT_COMMIT = "653ee22266a34a74efca21b0b03dfc1fd6fa37ff"
DEFAULT_INTACT_ROOT = Path("/grp01/ids_compcog/song/code/INTACT-JEPA")
DEFAULT_PYTHON = Path("/grp01/ids_compcog/song/envs/intact-py310/bin/python")
DEFAULT_DATA_ROOT = Path("/grp01/ids_compcog/song/swm")
DEFAULT_OUTPUT_ROOT = Path("/grp01/ids_compcog/song/intact")
DEFAULT_RUN = (
    "intact-tagged-pusht-20261001-054708_goal_pusht_tagged_s3072"
)


def quote(parts: list[object]) -> str:
    return shlex.join(str(value) for value in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--intact-root", type=Path, default=DEFAULT_INTACT_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", default="weights_epoch_1.pt")
    parser.add_argument("--partition", default="interactive")
    parser.add_argument("--nodelist", default="SPGL-1-1")
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()

    checkpoint_dir = args.output_root / "checkpoints" / args.run_name
    required = {
        "python": args.python,
        "intact checkout": args.intact_root,
        "checkpoint": checkpoint_dir / args.checkpoint,
        "checkpoint config": checkpoint_dir / "config.json",
        "dataset": args.data_root / "datasets/pusht_expert_train.h5",
    }
    missing = [label for label, path in required.items() if not path.exists()]
    commit = None
    dirty = None
    if args.intact_root.is_dir():
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.intact_root, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--short"], cwd=args.intact_root, text=True
        ).strip()

    plan = {
        "benchmark": "tagged PushT",
        "method": "INTACT goal-displacement",
        "checkpoint": str(checkpoint_dir / args.checkpoint),
        "upstream_commit": commit,
        "expected_upstream_commit": PINNED_INTACT_COMMIT,
        "upstream_dirty": bool(dirty),
        "missing": missing,
        "protocol": {
            "probe_seed": args.seed,
            "linear_probe_clips": 1024,
            "geometry_frames": 512,
            "intervention_clips": 128,
            "intervention_candidates": 32,
            "tag_size": 5,
        },
    }
    print(json.dumps(plan, indent=2))
    if not args.submit:
        print("PLAN ONLY: pass --submit --confirm-run to launch.")
        return
    if not args.confirm_run:
        parser.error("refusing to submit without --confirm-run")
    if missing:
        parser.error("missing: " + ", ".join(missing))
    if commit != PINNED_INTACT_COMMIT or dirty:
        parser.error("INTACT checkout must be pinned and clean")

    root = REPO / "leworldmodel/results" / (
        "intact-tagged-diagnostics-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=False)
    spec = f"intact={args.run_name}/{args.checkpoint}"
    common = [
        "set -euo pipefail",
        # Put upstream first so Hydra resolves INTACT's module.py and utils.py.
        "export PYTHONPATH="
        + shlex.quote(str(args.intact_root))
        + ":"
        + shlex.quote(str(REPO / "leworldmodel"))
        + ":"
        + shlex.quote(str(REPO)),
        f"export LOCAL_DATASET_DIR={shlex.quote(str(args.data_root))}",
        f"export STABLEWM_HOME={shlex.quote(str(args.output_root))}",
        "export MUJOCO_GL=egl",
        "export PYOPENGL_PLATFORM=egl",
        'export TMPDIR="/grp01/ids_compcog/song/tmp/aluo/${SLURM_JOB_ID}"',
        'mkdir -p "$TMPDIR"',
    ]
    commands = [
        quote([
            args.python, HERE / "diagnose_frozen_representation.py",
            "--checkpoint", spec,
            "--dataset", "pusht_expert_train.h5",
            "--cache-dir", args.data_root,
            "--num-clips", 1024,
            "--batch-size", 32,
            "--seed", args.seed,
            "--ridge-alpha", 1.0,
            "--out", root / "linear-probe.json",
        ]),
        'echo "INTACT_DIAGNOSTIC_LINEAR_COMPLETE"',
        quote([
            args.python, HERE / "diagnose_generic_tag_geometry.py",
            "--checkpoint", spec,
            "--dataset", "pusht_expert_train.h5",
            "--cache-dir", args.data_root,
            "--num-frames", 512,
            "--batch-size", 32,
            "--tag-size", 5,
            "--tag-seed", 0,
            "--seed", args.seed,
            "--out", root / "tag-geometry.json",
        ]),
        'echo "INTACT_DIAGNOSTIC_GEOMETRY_COMPLETE"',
        quote([
            args.python, HERE / "diagnose_tag_intervention.py",
            "--checkpoint", spec,
            "--dataset", "pusht_expert_train.h5",
            "--num-clips", 128,
            "--candidates", 32,
            "--history", 3,
            "--horizon", 5,
            "--frameskip", 5,
            "--tag-size", 5,
            "--seed", args.seed,
            "--out", root / "tag-intervention.json",
        ]),
        'echo "INTACT_DIAGNOSTICS_COMPLETE"',
    ]
    shell = "\n".join(common + commands)
    batch = [
        "sbatch", "--parsable",
        "--partition", args.partition,
        "--gres=gpu:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=03:00:00",
        "--job-name=oe-intact-diag",
        "--chdir", str(REPO),
        "--output", str(root / "job-%j.out"),
        "--wrap", shell,
    ]
    if args.nodelist:
        batch[4:4] = ["--nodelist", args.nodelist]
    job = subprocess.check_output(batch, text=True).strip().split(";")[0]
    plan["job_id"] = job
    (root / "manifest.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    print(f"INTACT_DIAGNOSTIC_JOB={job}")
    print(f"CAMPAIGN={root}")


if __name__ == "__main__":
    main()
