#!/usr/bin/env python3
"""Prepare or submit pinned AC-MTM on the exact tagged PushT dataset."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
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


def quote(parts: list[object]) -> str:
    return shlex.join(str(value) for value in parts)


def adapter_prefix(args, phase: str, tag_seed: object) -> list[object]:
    return [
        args.python,
        ADAPTER,
        "--ac-root", args.ac_root,
        "--expected-commit", PINNED_AC_MTM_COMMIT,
        "--tag-mode", "video",
        "--tag-size", 5,
        "--tag-seed", tag_seed,
        phase,
        "--",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--ac-root", type=Path, default=DEFAULT_AC_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--stablewm-home", type=Path, default=DEFAULT_STABLEWM_HOME)
    parser.add_argument("--partition", default="gpu_shared")
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--tag-seed", type=int, default=0)
    args = parser.parse_args()

    required = {
        "python": args.python,
        "AC-MTM train": args.ac_root / "train.py",
        "AC-MTM eval": args.ac_root / "eval.py",
        "PushT dataset": (
            args.stablewm_home / "datasets/pusht_expert_train.h5"
        ),
    }
    missing = [label for label, path in required.items() if not path.exists()]
    commit = dirty = None
    if args.ac_root.is_dir():
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.ac_root, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--short"], cwd=args.ac_root, text=True
        ).strip()
    plan = {
        "benchmark": "tagged PushT",
        "method": "AC-MTM action-contrastive inverse dynamics",
        "upstream_commit": commit,
        "expected_upstream_commit": PINNED_AC_MTM_COMMIT,
        "upstream_dirty": bool(dirty),
        "missing": missing,
        "training_seed": args.seed,
        "epochs": 10,
        "batch_size": 128,
        "tag": {"mode": "video", "size": 5, "seed": args.tag_seed},
        "loss": {
            "type": "action_nce",
            "inverse_weight": 0.30,
            "temperature": 0.1,
        },
        "jobs": {},
    }
    print(json.dumps(plan, indent=2))
    if not args.submit:
        print("PLAN ONLY: pass --submit --confirm-run to launch.")
        return
    if not args.confirm_run:
        parser.error("refusing to submit without --confirm-run")
    if missing:
        parser.error("missing: " + ", ".join(missing))
    if commit != PINNED_AC_MTM_COMMIT or dirty:
        parser.error("pinned clean AC-MTM checkout preflight failed")

    subprocess.run(
        [
            str(args.python), "-m", "pytest", "-q",
            str(HERE / "tests/test_pixel_tag.py"),
            str(HERE / "tests/test_ac_mtm_tagged_adapter.py"),
        ],
        cwd=REPO,
        check=True,
    )

    root = REPO / "leworldmodel/results" / (
        "ac-mtm-tagged-pusht-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=False)
    run_subdir = f"baselines/{root.name}_s{args.seed}"
    smoke_subdir = f"baselines/{root.name}_smoke_s{args.seed}"
    exports = [
        "set -euo pipefail",
        f"export PYTHONPATH={shlex.quote(str(REPO / 'leworldmodel'))}:{shlex.quote(str(REPO))}",
        f"export STABLEWM_HOME={shlex.quote(str(args.stablewm_home))}",
        "export MUJOCO_GL=egl",
        "export PYOPENGL_PLATFORM=egl",
        "export WANDB_MODE=disabled",
        'export TMPDIR="/grp01/ids_compcog/song/tmp/aluo/${SLURM_JOB_ID}"',
        'mkdir -p "$TMPDIR"',
    ]

    smoke = adapter_prefix(args, "train", args.tag_seed) + [
        "--config-name=lewm_masked_action_nce",
        "data=pusht",
        f"subdir={smoke_subdir}",
        f"seed={args.seed}",
        "wandb.enabled=false",
        "early_stopping.enabled=false",
        "+trainer.max_steps=2",
        "loader.num_workers=0",
        "loader.persistent_workers=false",
        "loader.prefetch_factor=null",
    ]
    smoke_shell = "\n".join(
        exports + [quote(smoke), 'echo "AC_MTM_TAGGED_SMOKE_COMPLETE"']
    )
    smoke_job = subprocess.check_output(
        [
            "sbatch", "--parsable", "--partition", args.partition,
            "--gres=gpu:1", "--cpus-per-task=8", "--mem=64G",
            "--time=00:30:00", "--job-name=oe-acmtm-smoke",
            "--chdir", str(REPO), "--output", str(root / "smoke-%j.out"),
            "--wrap", smoke_shell,
        ],
        text=True,
    ).strip().split(";")[0]
    plan["jobs"]["smoke"] = smoke_job
    (root / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")

    train = adapter_prefix(args, "train", args.tag_seed) + [
        "--config-name=lewm_masked_action_nce",
        "data=pusht",
        f"subdir={run_subdir}",
        f"seed={args.seed}",
        "trainer.max_epochs=10",
        "early_stopping.enabled=false",
        "wandb.enabled=false",
    ]
    train_shell = "\n".join(
        exports + [quote(train), 'echo "AC_MTM_TAGGED_TRAIN_COMPLETE"']
    )
    train_job = subprocess.check_output(
        [
            "sbatch", "--parsable", "--partition", args.partition,
            "--dependency=afterok:" + smoke_job,
            "--kill-on-invalid-dep=yes", "--gres=gpu:1",
            "--cpus-per-task=12", "--mem=96G", "--time=20:00:00",
            "--job-name=oe-acmtm-train", "--chdir", str(REPO),
            "--output", str(root / "train-%j.out"), "--wrap", train_shell,
        ],
        text=True,
    ).strip().split(";")[0]
    plan["jobs"]["train"] = train_job
    plan["run_subdir"] = run_subdir
    plan["evaluation_submission"] = "deferred until checkpoint audit"
    (root / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"AC_MTM_SMOKE_JOB={smoke_job}")
    print(f"AC_MTM_TRAIN_JOB={train_job}")
    print("AC_MTM_EVAL_JOB=DEFERRED")
    print(f"CAMPAIGN={root}")


if __name__ == "__main__":
    main()
