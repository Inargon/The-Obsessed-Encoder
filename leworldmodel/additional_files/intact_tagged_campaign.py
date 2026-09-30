#!/usr/bin/env python3
"""Prepare or submit the pinned INTACT tagged-PushT baseline campaign.

Dry-run is the default. Submission requires BOTH --submit and --confirm-run.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ADAPTER = HERE / "intact_tagged_adapter.py"
PINNED_COMMIT = "653ee22266a34a74efca21b0b03dfc1fd6fa37ff"
DEFAULT_INTACT_ROOT = Path("/grp01/ids_compcog/song/code/INTACT-JEPA")
DEFAULT_INTACT_PYTHON = Path("/grp01/ids_compcog/song/envs/intact-py310/bin/python")
DEFAULT_DATA_ROOT = Path("/grp01/ids_compcog/song/swm")
DEFAULT_OUTPUT_ROOT = Path("/grp01/ids_compcog/song/intact")
# Official LeWM evaluation calls the actor-disabled broad-search mode ``cem``.
# ``pure_cem`` is the corresponding CLEAR-LeWM adapter spelling and is not a
# valid solver/config name for INTACT's native eval.py entrypoint.
EVAL_MODES = ("direct", "cem")


def validate_clean_audit(path: Path | None) -> tuple[bool, dict | None, str | None]:
    """Validate the local reproduction gate, not merely the file's existence."""
    if path is None:
        return False, None, "clean audit path was not provided"
    if not path.is_file():
        return False, None, f"clean audit does not exist: {path}"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, None, f"cannot read clean audit: {error}"
    expected = {
        "status": "pass",
        "benchmark": "clean PushT",
        "upstream_commit": PINNED_COMMIT,
        "protocol": "official LeWM",
        "inference_mode": "direct",
        "search_enabled": False,
    }
    mismatches = [
        f"{key}={record.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if record.get(key) != value
    ]
    if int(record.get("num_eval", 0)) < 100:
        mismatches.append("num_eval must be at least 100")
    success_rate = record.get("success_rate")
    if not isinstance(success_rate, (int, float)) or not 0.0 <= success_rate <= 1.0:
        mismatches.append("success_rate must be a fraction in [0, 1]")
    checkpoint = Path(str(record.get("checkpoint", "")))
    expected_hash = record.get("checkpoint_sha256")
    if not checkpoint.is_file():
        mismatches.append(f"audited checkpoint is missing: {checkpoint}")
    elif not isinstance(expected_hash, str):
        mismatches.append("checkpoint_sha256 is missing")
    else:
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            mismatches.append("audited checkpoint SHA-256 no longer matches")
    if mismatches:
        return False, record, "; ".join(mismatches)
    return True, record, None


def command_text(parts) -> str:
    return shlex.join(str(value) for value in parts)


def preflight(args) -> dict:
    paths = {
        "intact_root": args.intact_root,
        "intact_python": args.intact_python,
        "lance_dataset": args.data_root / "datasets/pusht_expert_train.lance",
        "hdf5_dataset": args.data_root / "datasets/pusht_expert_train.h5",
        "output_root": args.output_root,
    }
    missing = [
        name for name, path in paths.items()
        if name != "output_root" and not path.exists()
    ]
    commit = dirty = None
    if args.intact_root.is_dir():
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.intact_root, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--short"], cwd=args.intact_root, text=True
        ).strip()
    audit_valid, audit_record, audit_error = validate_clean_audit(args.clean_audit)
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "missing": missing,
        "intact_commit": commit,
        "expected_commit": PINNED_COMMIT,
        "intact_dirty": bool(dirty),
        "clean_audit": str(args.clean_audit) if args.clean_audit else None,
        "clean_audit_exists": bool(args.clean_audit and args.clean_audit.is_file()),
        "clean_audit_valid": audit_valid,
        "clean_audit_error": audit_error,
        "clean_audit_summary": None if audit_record is None else {
            key: audit_record.get(key)
            for key in (
                "training_seed", "evaluation_seed", "num_eval",
                "success_rate", "checkpoint_sha256", "job_id",
            )
        },
    }


def common_env(args) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": os.pathsep.join((str(REPO / "leworldmodel"), str(REPO))),
        "LOCAL_DATASET_DIR": str(args.data_root),
        "STABLEWM_HOME": str(args.output_root),
        "INTACT_OUTPUT_HOME": str(args.output_root),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "WANDB_MODE": "disabled",
    })
    return env


def adapter_prefix(args, phase: str, tag_seed) -> list[str]:
    return [
        # argparse.REMAINDER begins after the positional phase. Global adapter
        # flags must therefore precede ``train``/``eval`` or they are forwarded
        # to Hydra and the required --intact-root appears to be missing.
        str(args.intact_python), str(ADAPTER),
        "--intact-root", str(args.intact_root),
        "--expected-commit", PINNED_COMMIT,
        "--tag-mode", "video", "--tag-size", "5",
        "--tag-seed", str(tag_seed), phase, "--",
    ]


def submit(args, root: Path, manifest: dict) -> None:
    root.mkdir(parents=True, exist_ok=False)

    def save():
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    exports = [
        "set -euo pipefail",
        f"export PYTHONPATH={shlex.quote(str(REPO / 'leworldmodel'))}:{shlex.quote(str(REPO))}",
        f"export LOCAL_DATASET_DIR={shlex.quote(str(args.data_root))}",
        f"export STABLEWM_HOME={shlex.quote(str(args.output_root))}",
        f"export INTACT_OUTPUT_HOME={shlex.quote(str(args.output_root))}",
        "export MUJOCO_GL=egl",
        "export PYOPENGL_PLATFORM=egl",
        "export WANDB_MODE=disabled",
        'export TMPDIR="/grp01/ids_compcog/song/tmp/aluo/${SLURM_JOB_ID}"',
        'mkdir -p "$TMPDIR"',
    ]

    smoke_name = f"{root.name}_smoke_s{args.train_seed}"
    smoke_cmd = adapter_prefix(args, "train", args.tag_seed) + [
        "--config-name=intact_goal", f"output_model_name={smoke_name}",
        f"seed={args.train_seed}", "+trainer.max_steps=2",
        "loader.num_workers=0", "loader.persistent_workers=false",
    ]
    smoke_shell = "\n".join(
        exports + [command_text(smoke_cmd), "echo INTACT_TAGGED_SMOKE_COMPLETE"]
    )
    smoke = subprocess.check_output([
        "sbatch", "--parsable", "--partition", args.partition,
        "--gres=gpu:1", "--cpus-per-task=8", "--mem=64G", "--time=00:30:00",
        "--job-name=oe-intact-tag-smoke", "--chdir", str(REPO),
        "--output", str(root / "smoke-%j.out"), "--wrap", smoke_shell,
    ], text=True).strip().split(";")[0]
    manifest["jobs"]["smoke"] = smoke
    save()

    run_name = f"{root.name}_goal_pusht_tagged_s{args.train_seed}"
    train_cmd = adapter_prefix(args, "train", args.tag_seed) + [
        "--config-name=intact_goal", f"output_model_name={run_name}",
        f"seed={args.train_seed}",
    ]
    train_shell = "\n".join(
        exports + [command_text(train_cmd), "echo INTACT_TAGGED_TRAIN_COMPLETE"]
    )
    training = subprocess.check_output([
        "sbatch", "--parsable", "--partition", args.partition,
        "--dependency=afterok:" + smoke, "--kill-on-invalid-dep=yes",
        "--gres=gpu:1", "--cpus-per-task=12", "--mem=96G", "--time=20:00:00",
        "--job-name=oe-intact-tag-train", "--chdir", str(REPO),
        "--output", str(root / "train-%j.out"), "--wrap", train_shell,
    ], text=True).strip().split(";")[0]
    manifest["jobs"]["train"] = training
    manifest["run_name"] = run_name
    save()

    if args.defer_eval:
        manifest["jobs"]["eval"] = None
        manifest["evaluation_submission"] = "deferred_by_request"
        save()
        print(f"SMOKE_JOB={smoke}")
        print(f"TRAIN_JOB={training}")
        print("EVAL_ARRAY_JOB=DEFERRED")
        print(f"CAMPAIGN={root}")
        return

    cells = [(mode, seed) for mode in EVAL_MODES for seed in args.eval_seeds]
    cell_lines = [f"{mode} {seed}" for mode, seed in cells]
    eval_prefix = command_text(adapter_prefix(args, "eval", "TAG_SEED"))
    eval_prefix = eval_prefix.replace("TAG_SEED", '"$eval_seed"')
    eval_shell = "\n".join(exports + [
        "cells=(" + " ".join(shlex.quote(line) for line in cell_lines) + ")",
        'read -r mode eval_seed <<< "${cells[$SLURM_ARRAY_TASK_ID]}"',
        eval_prefix
        + " --config-name=pusht"
        + ' solver="$mode"'
        + f" policy={shlex.quote(run_name + '/weights_epoch_1.pt')}"
        + ' seed="$eval_seed"'
        + f" eval.num_eval={args.num_eval}"
        + " eval.protocol=official"
        + ' eval.inference_mode="$mode"'
        + f" cache_dir={shlex.quote(str(args.data_root))}"
        + ' output.filename="tagged-${mode}-seed${eval_seed}.txt"',
        'echo "INTACT_TAGGED_EVAL_COMPLETE mode=$mode seed=$eval_seed"',
    ])
    eval_batch = [
        "sbatch", "--parsable", "--partition", args.eval_partition,
        f"--array=0-{len(cells)-1}%{args.max_eval_concurrent}",
        "--dependency=afterok:" + training, "--kill-on-invalid-dep=yes",
        "--gres=gpu:1", "--cpus-per-task=8", "--mem=48G", "--time=03:00:00",
        "--job-name=oe-intact-tag-eval", "--chdir", str(REPO),
        "--output", str(root / "eval-%A_%a.out"), "--wrap", eval_shell,
    ]
    if args.eval_nodelist:
        eval_batch[4:4] = ["--nodelist", args.eval_nodelist]
    evaluation = subprocess.check_output(eval_batch, text=True).strip().split(";")[0]
    manifest["jobs"]["eval"] = evaluation
    save()
    print(f"SMOKE_JOB={smoke}")
    print(f"TRAIN_JOB={training}")
    print(f"EVAL_ARRAY_JOB={evaluation}")
    print(f"CAMPAIGN={root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--intact-root", type=Path, default=DEFAULT_INTACT_ROOT)
    parser.add_argument("--intact-python", type=Path, default=DEFAULT_INTACT_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clean-audit", type=Path)
    parser.add_argument("--partition", default="gpu_shared")
    parser.add_argument("--eval-partition", default="interactive")
    parser.add_argument("--eval-nodelist", default="SPGL-1-1")
    parser.add_argument("--train-seed", type=int, default=3072)
    parser.add_argument("--tag-seed", type=int, default=0)
    parser.add_argument("--eval-seeds", default="0,1,42")
    parser.add_argument("--num-eval", type=int, default=100)
    parser.add_argument("--max-eval-concurrent", type=int, default=2)
    parser.add_argument(
        "--defer-eval", action="store_true",
        help="submit only smoke and training; add the evaluation array later",
    )
    args = parser.parse_args()
    args.eval_seeds = tuple(int(value) for value in args.eval_seeds.split(","))

    checks = preflight(args)
    print(json.dumps(checks, indent=2))
    print("\nPLANNED CONTRACT")
    print(f"  upstream: INTACT {PINNED_COMMIT}")
    print("  train: official intact_goal, seed 3072, 1 epoch, Math SDPA")
    print("  nuisance: exact 5x5 episode-constant PixelTag before preprocessing")
    print(f"  eval: {EVAL_MODES}, seeds={args.eval_seeds}, episodes={args.num_eval}")
    print("  dataset: one clean Lance copy; no duplicated tagged dataset")

    if not args.submit:
        print("\nPLAN ONLY: no jobs submitted.")
        print("Actual launch requires both --submit and --confirm-run.")
        return
    if not args.confirm_run:
        parser.error("refusing to submit without --confirm-run")
    if checks["missing"]:
        parser.error("preflight missing: " + ", ".join(checks["missing"]))
    if checks["intact_commit"] != PINNED_COMMIT or checks["intact_dirty"]:
        parser.error("pinned clean INTACT checkout preflight failed")
    if not checks["clean_audit_valid"]:
        parser.error(
            "tagged training requires a valid clean audit: "
            + str(checks["clean_audit_error"])
        )

    subprocess.run([
        sys.executable, "-m", "pytest", "-q",
        str(HERE / "tests/test_pixel_tag.py"),
        str(HERE / "tests/test_intact_tagged_adapter.py"),
    ], cwd=REPO, env=common_env(args), check=True)
    subprocess.run(
        [str(args.intact_python), str(args.intact_root / "scripts/verify_install.py")],
        cwd=args.intact_root,
        env=common_env(args),
        check=True,
    )
    root = REPO / "leworldmodel/results" / (
        "intact-tagged-pusht-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    manifest = {
        "benchmark": "tagged PushT",
        "method": "INTACT goal-displacement",
        "upstream_commit": PINNED_COMMIT,
        "adapter": str(ADAPTER.relative_to(REPO)),
        "training_seed": args.train_seed,
        "tag": {"mode": "video", "size": 5, "seed": args.tag_seed},
        "evaluation_modes": EVAL_MODES,
        "evaluation_seeds": args.eval_seeds,
        "evaluation_episodes": args.num_eval,
        "clean_audit": str(args.clean_audit),
        "jobs": {},
    }
    submit(args, root, manifest)


if __name__ == "__main__":
    main()
