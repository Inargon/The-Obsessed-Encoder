#!/usr/bin/env python3
"""Prepare and submit the matched clean/tagged H-JEPA PushT pilot."""

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

from additional_files.hjepa_tagged_adapter import PINNED_HJEPA_COMMIT

ADAPTER = HERE / "hjepa_tagged_adapter.py"
DEFAULT_UPSTREAM = Path("/grp01/ids_compcog/song/code/hjepa-repro-2609.33497")
DEFAULT_PYTHON = Path("/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python")
DEFAULT_SWM_HOME = Path("/grp01/ids_compcog/song/swm")
UPSTREAM_URL = "https://github.com/tmz-lab/hjepa.git"


def quote(parts) -> str:
    return shlex.join(str(x) for x in parts)


def prepare_upstream(root: Path) -> str:
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", UPSTREAM_URL, str(root)], check=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if commit != PINNED_HJEPA_COMMIT:
        subprocess.run(["git", "fetch", "origin", PINNED_HJEPA_COMMIT], cwd=root, check=True)
        subprocess.run(["git", "checkout", "--detach", PINNED_HJEPA_COMMIT], cwd=root, check=True)
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--short"], cwd=root, text=True
    ).strip()
    if commit != PINNED_HJEPA_COMMIT or dirty:
        raise RuntimeError("H-JEPA checkout is not at the pinned clean commit")
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--prepare-upstream", action="store_true")
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--stablewm-home", type=Path, default=DEFAULT_SWM_HOME)
    parser.add_argument("--train-partition", default="gpu_shared")
    parser.add_argument("--train-node", default="SPGL-1-11")
    parser.add_argument("--eval-partition", default="interactive")
    parser.add_argument("--eval-node", default="SPGL-1-1")
    parser.add_argument("--training-seed", type=int, default=3072)
    parser.add_argument("--evaluation-seed", type=int, default=42)
    parser.add_argument("--tag-seed", type=int, default=0)
    parser.add_argument("--num-eval", type=int, default=50)
    args = parser.parse_args()

    commit = None
    if args.prepare_upstream or args.submit:
        commit = prepare_upstream(args.upstream)
    elif (args.upstream / ".git").exists():
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.upstream, text=True
        ).strip()

    plan = {
        "benchmark": "PushT clean/predictable-tag matched pilot",
        "method": "official H-JEPA",
        "upstream_url": UPSTREAM_URL,
        "upstream_commit": commit,
        "expected_upstream_commit": PINNED_HJEPA_COMMIT,
        "training_seed": args.training_seed,
        "evaluation_seed": args.evaluation_seed,
        "tag": {"mode": "video", "size": 5, "seed": args.tag_seed},
        "training": {
            "conditions": ["clean", "tagged"],
            "epochs": 10,
            "official_config": "config/train/hjepa.yaml + data/pusht.yaml",
        },
        "evaluation_cells": ["clean_clean", "clean_tagged", "tagged_tagged"],
        "num_eval": args.num_eval,
        "interpretation": {
            "clean_clean": "ordinary H-JEPA control baseline",
            "clean_tagged": "test-time nuisance shift sensitivity",
            "tagged_tagged": "whether predictable tag is exploited during training",
        },
        "jobs": {},
    }
    print(json.dumps(plan, indent=2))
    if not args.submit:
        print("PLAN ONLY: use --prepare-upstream to pin source; add --submit --confirm-run to launch.")
        return
    if not args.confirm_run:
        parser.error("refusing to submit without --confirm-run")
    required = [
        args.python,
        args.upstream / "train.py",
        args.upstream / "eval.py",
        args.stablewm_home / "datasets/pusht_expert_train.h5",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("missing required paths: " + ", ".join(missing))

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO / "leworldmodel"), str(REPO), env.get("PYTHONPATH")])
    )
    subprocess.run(
        [
            str(args.python), "-m", "pytest", "-q",
            str(HERE / "tests/test_hjepa_tagged_adapter.py"),
            str(HERE / "tests/test_hjepa_tagged_campaign.py"),
            str(HERE / "tests/test_pixel_tag.py"),
        ],
        cwd=REPO,
        env=env,
        check=True,
    )
    subprocess.run(
        [str(args.python), ADAPTER, "--upstream", args.upstream, "preflight"],
        cwd=REPO,
        env=env,
        check=True,
    )

    campaign = REPO / "leworldmodel/results" / (
        "hjepa-tagged-pusht-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    campaign.mkdir(parents=True, exist_ok=False)
    checkpoint_root = args.stablewm_home / "checkpoints" / campaign.name
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    plan["campaign"] = str(campaign)
    plan["checkpoint_root"] = str(checkpoint_root)

    exports = [
        "set -euo pipefail",
        f"export PYTHONPATH={shlex.quote(str(REPO / 'leworldmodel'))}:{shlex.quote(str(REPO))}",
        f"export STABLEWM_HOME={shlex.quote(str(args.stablewm_home))}",
        f"export LOCAL_DATASET_DIR={shlex.quote(str(args.stablewm_home))}",
        f"export XDG_CACHE_HOME={shlex.quote(str(args.stablewm_home / 'cache'))}",
        f"export HF_HOME={shlex.quote(str(args.stablewm_home / 'cache/huggingface'))}",
        "export MUJOCO_GL=egl",
        "export PYOPENGL_PLATFORM=egl",
        "export WANDB_MODE=disabled",
        'export TMPDIR="/grp01/ids_compcog/song/tmp/aluo/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"',
        'mkdir -p "$TMPDIR"',
    ]
    train_shell = "\n".join(
        exports
        + [
            "conditions=(clean tagged)",
            'condition="${conditions[$SLURM_ARRAY_TASK_ID]}"',
            f'run_dir={shlex.quote(str(checkpoint_root))}/"$condition"',
            (
                f"{shlex.quote(str(args.python))} {shlex.quote(str(ADAPTER))} "
                f"--upstream {shlex.quote(str(args.upstream))} train "
                f"--condition \"$condition\" --tag-seed {args.tag_seed} "
                "--tag-size 5 -- --config-name=hjepa data=pusht "
                f'"subdir=$run_dir" seed={args.training_seed} '
                "trainer.max_epochs=10 wandb.enabled=false"
            ),
            'echo "HJEPA_TAGGED_TRAIN_COMPLETE condition=$condition"',
        ]
    )
    train_job = subprocess.check_output(
        [
            "sbatch", "--parsable", "--partition", args.train_partition,
            "--nodelist", args.train_node, "--array=0-1%2", "--gres=gpu:1",
            "--cpus-per-task=12", "--mem=96G", "--time=1-12:00:00",
            "--job-name=oe-hjepa-tag-train", "--chdir", str(REPO),
            "--output", str(campaign / "train-%A_%a.out"), "--wrap", train_shell,
        ],
        text=True,
    ).strip().split(";")[0]
    plan["jobs"]["train_array"] = train_job

    eval_shell = "\n".join(
        exports
        + [
            "cells=(clean_clean clean_tagged tagged_tagged)",
            'cell="${cells[$SLURM_ARRAY_TASK_ID]}"',
            'train_condition="${cell%%_*}"',
            'input_condition="${cell##*_}"',
            f'checkpoint={shlex.quote(str(checkpoint_root))}/"$train_condition"/hjepa_epoch_10_object.ckpt',
            (
                f"{shlex.quote(str(args.python))} {shlex.quote(str(ADAPTER))} "
                f"--upstream {shlex.quote(str(args.upstream))} eval "
                '--checkpoint "$checkpoint" --train-condition "$train_condition" '
                '--input-condition "$input_condition" '
                f'--output "{campaign}/$cell.json" --seed {args.evaluation_seed} '
                f"--num-eval {args.num_eval} --tag-seed {args.tag_seed} --tag-size 5"
            ),
        ]
    )
    eval_job = subprocess.check_output(
        [
            "sbatch", "--parsable", "--partition", args.eval_partition,
            "--nodelist", args.eval_node, "--array=0-2%1",
            "--dependency=afterok:" + train_job, "--kill-on-invalid-dep=yes",
            "--gres=gpu:1", "--cpus-per-task=8", "--mem=64G", "--time=02:00:00",
            "--job-name=oe-hjepa-tag-eval", "--chdir", str(REPO),
            "--output", str(campaign / "eval-%A_%a.out"), "--wrap", eval_shell,
        ],
        text=True,
    ).strip().split(";")[0]
    plan["jobs"]["eval_array"] = eval_job
    (campaign / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"HJEPA_TRAIN_ARRAY_JOB={train_job}")
    print(f"HJEPA_EVAL_ARRAY_JOB={eval_job}")
    print(f"CAMPAIGN={campaign}")


if __name__ == "__main__":
    main()
