#!/usr/bin/env python3
"""Evaluate the completed tagged PushT AC-MTM checkpoint on fixed seeds."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "leworldmodel"))

from additional_files.ac_mtm_tagged_adapter import PINNED_AC_MTM_COMMIT


ADAPTER = HERE / "ac_mtm_tagged_adapter.py"
DEFAULT_AC_ROOT = Path("/grp01/ids_compcog/song/code/AC-MTM")
DEFAULT_PYTHON = Path("/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python")
DEFAULT_STABLEWM_HOME = Path("/grp01/ids_compcog/song/swm")
DEFAULT_TRAINING_RUN = (
    "ac-mtm-tagged-pusht-20261001-164536_s3072"
)
EVALUATION_SEEDS = (0, 1, 42)


def quote(parts: list[object]) -> str:
    return shlex.join(str(value) for value in parts)


def policy_name(training_run: str) -> str:
    return (
        f"baselines/{training_run}/"
        "lewm_masked_action_nce_epoch_10"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--ac-root", type=Path, default=DEFAULT_AC_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--stablewm-home", type=Path, default=DEFAULT_STABLEWM_HOME)
    parser.add_argument("--training-run", default=DEFAULT_TRAINING_RUN)
    parser.add_argument("--partition", default="gpu_shared")
    parser.add_argument("--max-concurrent", type=int, default=2)
    args = parser.parse_args()

    policy = policy_name(args.training_run)
    checkpoint = args.stablewm_home / f"{policy}_object.ckpt"
    required = {
        "python": args.python,
        "AC-MTM eval": args.ac_root / "eval.py",
        "adapter": ADAPTER,
        "epoch-10 evaluation checkpoint": checkpoint,
    }
    missing = [label for label, path in required.items() if not path.is_file()]
    commit = dirty = None
    if args.ac_root.is_dir():
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.ac_root, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--short"], cwd=args.ac_root, text=True
        ).strip()

    manifest = {
        "benchmark": "tagged PushT",
        "method": "AC-MTM action-contrastive inverse dynamics",
        "upstream_commit": commit,
        "expected_upstream_commit": PINNED_AC_MTM_COMMIT,
        "upstream_dirty": bool(dirty),
        "training_seed": 3072,
        "training_run": args.training_run,
        "policy": policy,
        "checkpoint": str(checkpoint),
        "checkpoint_size": checkpoint.stat().st_size if checkpoint.is_file() else None,
        "tag": {"mode": "video", "size": 5},
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "evaluation_episodes_per_seed": 100,
        "evaluation_env_batch_size": 10,
        "missing": missing,
        "jobs": {},
    }
    print(json.dumps(manifest, indent=2))
    if not args.submit:
        print("PLAN ONLY: pass --submit --confirm-run to launch.")
        return
    if not args.confirm_run:
        parser.error("refusing to submit without --confirm-run")
    if missing:
        parser.error("missing: " + ", ".join(missing))
    if commit != PINNED_AC_MTM_COMMIT or dirty:
        parser.error("pinned clean AC-MTM checkout preflight failed")

    root = REPO / "leworldmodel/results" / (
        "ac-mtm-tagged-eval-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=False)

    exports = [
        "set -euo pipefail",
        f"export PYTHONPATH={shlex.quote(str(REPO / 'leworldmodel'))}:{shlex.quote(str(REPO))}",
        f"export STABLEWM_HOME={shlex.quote(str(args.stablewm_home))}",
        f"export XDG_CACHE_HOME={shlex.quote(str(args.stablewm_home / 'cache'))}",
        f"export HF_HOME={shlex.quote(str(args.stablewm_home / 'cache/huggingface'))}",
        "export MUJOCO_GL=egl",
        "export PYOPENGL_PLATFORM=egl",
        "export WANDB_MODE=disabled",
        'export TMPDIR="/grp01/ids_compcog/song/tmp/aluo/${SLURM_JOB_ID}"',
        'mkdir -p "$TMPDIR"',
        f"seeds=({' '.join(str(value) for value in EVALUATION_SEEDS)})",
        'eval_seed="${seeds[$SLURM_ARRAY_TASK_ID]}"',
    ]
    command = [
        args.python,
        ADAPTER,
        "--ac-root", args.ac_root,
        "--expected-commit", PINNED_AC_MTM_COMMIT,
        "--tag-mode", "video",
        "--tag-size", 5,
        "--tag-seed", "$eval_seed",
        "eval",
        "--",
        "--config-name=pusht",
        f"policy={policy}",
        "seed=$eval_seed",
        "eval.num_eval=100",
        "eval.env_batch_size=10",
        "output.save_video=false",
        'output.filename=ac-mtm-tagged-seed${eval_seed}.txt',
    ]
    command_line = quote(command).replace("'$eval_seed'", '"$eval_seed"')
    command_line = command_line.replace(
        "'seed=$eval_seed'", '"seed=$eval_seed"'
    ).replace(
        "'output.filename=ac-mtm-tagged-seed${eval_seed}.txt'",
        '"output.filename=ac-mtm-tagged-seed${eval_seed}.txt"',
    )
    shell = "\n".join(
        exports
        + [
            command_line,
            'echo "AC_MTM_TAGGED_EVAL_COMPLETE seed=$eval_seed"',
        ]
    )
    job = subprocess.check_output(
        [
            "sbatch", "--parsable", "--partition", args.partition,
            f"--array=0-{len(EVALUATION_SEEDS) - 1}%{args.max_concurrent}",
            "--gres=gpu:1", "--cpus-per-task=8", "--mem=48G",
            "--time=04:00:00", "--job-name=oe-acmtm-tag-eval",
            "--chdir", str(REPO), "--output", str(root / "eval-%A_%a.out"),
            "--wrap", shell,
        ],
        text=True,
    ).strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError(f"unexpected sbatch response: {job!r}")
    manifest["jobs"]["evaluation_array"] = job
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"AC_MTM_TAGGED_EVAL_JOB={job}")
    print(f"CAMPAIGN={root}")


if __name__ == "__main__":
    main()
