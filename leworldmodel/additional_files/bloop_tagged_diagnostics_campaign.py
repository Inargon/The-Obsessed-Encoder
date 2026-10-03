#!/usr/bin/env python3
"""Submit matched tagged-PushT representation diagnostics for Bloop controls.

The four frozen checkpoints are evaluated on identical sampled clips and
interventions: nuisance/physical linear probes, counterfactual scene--tag
geometry, and fixed-candidate planning-cost interventions.  This is a
diagnostic campaign, not a closed-loop success-rate evaluation.
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
DEFAULT_PYTHON = Path(
    "/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python"
)
DEFAULT_DATA_ROOT = Path("/grp01/ids_compcog/song/swm")
CHECKPOINTS = (
    "jepa=colored_square_episode_seed0/weights_epoch_10.pt",
    "full=control_aligned_pred1_seed0/weights_epoch_10.pt",
    (
        "cycle=interface-cycle-20260930-222755_"
        "tagged_pusht_interface_cycle_full_seed0/weights_epoch_10.pt"
    ),
    (
        "bloop=interface-cycle-20261002-064431_"
        "tagged_pusht_bloop_cycle_full_seed0/weights_epoch_10.pt"
    ),
)


def quote(parts: list[object]) -> str:
    return shlex.join(str(value) for value in parts)


def checkpoint_path(data_root: Path, spec: str) -> Path:
    _, location = spec.split("=", 1)
    return data_root / "checkpoints" / location


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

    required = {"python": args.python}
    for spec in CHECKPOINTS:
        label = spec.split("=", 1)[0]
        checkpoint = checkpoint_path(args.data_root, spec)
        required[f"{label} checkpoint"] = checkpoint
        required[f"{label} config"] = checkpoint.parent / "config.json"
    required["dataset"] = args.data_root / "datasets/pusht_expert_train.h5"
    missing = [label for label, path in required.items() if not path.exists()]

    plan = {
        "benchmark": "tagged PushT",
        "diagnostic": "matched frozen representation and cost interventions",
        "checkpoints": list(CHECKPOINTS),
        "missing": missing,
        "protocol": {
            "probe_seed": args.seed,
            "linear_probe_clips": 1024,
            "geometry_frames": 512,
            "intervention_clips": 128,
            "intervention_candidates": 32,
            "tag_size": 5,
            "reporting": "all four models on identical samples",
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

    root = REPO / "leworldmodel/results" / (
        "bloop-tagged-diagnostics-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=False)
    checkpoint_args: list[object] = []
    for spec in CHECKPOINTS:
        checkpoint_args.extend(("--checkpoint", spec))

    common = [
        "set -euo pipefail",
        f"export PYTHONPATH={shlex.quote(str(REPO / 'leworldmodel'))}:"
        f"{shlex.quote(str(REPO))}",
        f"export LOCAL_DATASET_DIR={shlex.quote(str(args.data_root))}",
        f"export STABLEWM_HOME={shlex.quote(str(args.data_root))}",
        "export MUJOCO_GL=egl",
        'export TMPDIR="/grp01/ids_compcog/song/tmp/aluo/${SLURM_JOB_ID}"',
        'mkdir -p "$TMPDIR"',
    ]
    commands = [
        quote([
            args.python,
            HERE / "diagnose_frozen_representation.py",
            *checkpoint_args,
            "--dataset", "pusht_expert_train.h5",
            "--cache-dir", args.data_root,
            "--num-clips", 1024,
            "--batch-size", 32,
            "--seed", args.seed,
            "--ridge-alpha", 1.0,
            "--out", root / "linear-probe.json",
        ]),
        'echo "BLOOP_DIAGNOSTIC_LINEAR_COMPLETE"',
        quote([
            args.python,
            HERE / "diagnose_generic_tag_geometry.py",
            *checkpoint_args,
            "--dataset", "pusht_expert_train.h5",
            "--cache-dir", args.data_root,
            "--num-frames", 512,
            "--batch-size", 32,
            "--tag-size", 5,
            "--tag-seed", 0,
            "--seed", args.seed,
            "--out", root / "tag-geometry.json",
        ]),
        'echo "BLOOP_DIAGNOSTIC_GEOMETRY_COMPLETE"',
        quote([
            args.python,
            HERE / "diagnose_tag_intervention.py",
            *checkpoint_args,
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
        'echo "BLOOP_TAGGED_DIAGNOSTICS_COMPLETE"',
    ]
    batch = [
        "sbatch", "--parsable",
        "--partition", args.partition,
        "--gres=gpu:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=04:00:00",
        "--job-name=oe-bloop-diag",
        "--chdir", str(REPO),
        "--output", str(root / "job-%j.out"),
        "--wrap", "\n".join(common + commands),
    ]
    if args.nodelist:
        batch[4:4] = ["--nodelist", args.nodelist]
    job = subprocess.check_output(batch, text=True).strip().split(";")[0]
    plan["job_id"] = job
    (root / "manifest.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    print(f"BLOOP_TAGGED_DIAGNOSTIC_JOB={job}")
    print(f"CAMPAIGN={root}")


if __name__ == "__main__":
    main()
