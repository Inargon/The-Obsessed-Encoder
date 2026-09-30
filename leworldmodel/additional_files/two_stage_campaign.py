#!/usr/bin/env python3
"""Pilot representation-then-dynamics training on matched PushT.

Stage 1 is the existing h=1 checkpoint at the predeclared 60%-budget boundary
(step 85k). Stage 2 uses a fresh optimizer for every arm and four epochs, so
the total number of updates remains approximately matched to the 10-epoch run.
No job is submitted unless --submit and --confirm-run are both present.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

try:
    from .clean_inverse_reach_campaign import environment, source_digest
    from .horizon_ablation_campaign import arm_spec as horizon_arm_spec
except ImportError:
    from clean_inverse_reach_campaign import environment, source_digest
    from horizon_ablation_campaign import arm_spec as horizon_arm_spec


REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STAGE1_RUNS = {
    "clean": "horizon-clean-ablation-20260930-180928_full_h1_pair_seed0",
    "tagged": "horizon-ablation-20260930-025506_full_h1_pair_seed0",
}
ARMS = {
    "unfrozen_reset": {
        "freeze_representation": False,
        "cycle_weight": 0.5,
        "cycle_scope": "joint",
        "description": "optimizer-reset control; all model parameters remain trainable",
    },
    "frozen_no_cycle": {
        "freeze_representation": True,
        "cycle_weight": 0.0,
        "cycle_scope": "joint",
        "description": "freeze encoder/projector; calibrate Forward without Cycle",
    },
    "frozen_predictor_cycle": {
        "freeze_representation": True,
        "cycle_weight": 0.5,
        "cycle_scope": "predictor_only",
        "description": "freeze encoder/projector; Cycle calibrates predictor only",
    },
}


def runtime_environment(create_dirs: bool = True) -> dict[str, str]:
    env = environment()
    if create_dirs:
        for name in ("SPT_CACHE_DIR", "XDG_CACHE_HOME", "TMPDIR"):
            Path(env[name]).mkdir(parents=True, exist_ok=True)
    return env


def source_checkpoint(condition: str, filename: str) -> Path:
    root = Path(os.environ.get("STABLEWM_HOME", "/grp01/ids_compcog/song/swm"))
    return root / "checkpoints" / STAGE1_RUNS[condition] / filename


def _replace(overrides: list[str], prefix: str, value: object) -> None:
    matches = [index for index, item in enumerate(overrides) if item.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one override beginning {prefix!r}, got {len(matches)}")
    overrides[matches[0]] = prefix + str(value).lower()


def stage2_spec(arm: str, condition: str, checkpoint: Path) -> tuple[dict, dict]:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}")
    config, spec = horizon_arm_spec("h1_pair", condition)
    spec = copy.deepcopy(spec)
    overrides = list(spec["overrides"])
    _replace(overrides, "+loss.control.cycle_weight=", ARMS[arm]["cycle_weight"])
    overrides = [
        value for value in overrides
        if not value.startswith("+loss.control.cycle_scope=")
    ]
    if ARMS[arm]["cycle_scope"] != "joint":
        overrides.append(
            "+loss.control.cycle_scope=" + ARMS[arm]["cycle_scope"]
        )
    overrides += [
        "+stage2.enabled=true",
        f"+stage2.init_checkpoint={checkpoint}",
        "+stage2.freeze_representation="
        + str(ARMS[arm]["freeze_representation"]).lower(),
    ]
    spec["overrides"] = overrides
    return config, spec


def train(root: Path, arm: str, condition: str, seed: int, smoke: bool, source: Path) -> None:
    try:
        from . import run as base_runner
    except ImportError:
        import run as base_runner

    config, spec = stage2_spec(arm, condition, source)
    phase = "smoke" if smoke else "full"
    prefix = f"{root.name}_{condition}_{phase}_{arm}"
    run_name = f"{prefix}_seed{seed}"
    checkpoint_dir = Path(os.environ["STABLEWM_HOME"]) / "checkpoints" / run_name
    run_dir = root / run_name
    if checkpoint_dir.exists() or run_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing Stage-2 run: {run_name}")

    options = "++checkpoint.every_n_steps=5000 ++eval.every_n_steps=1000000000"
    if smoke:
        options = (
            "++trainer.max_steps=2 ++checkpoint.every_n_steps=1000000 "
            "++eval.every_n_steps=1000000000 loader.num_workers=0 "
            "loader.persistent_workers=false"
        )
    args = argparse.Namespace(
        results_dir=str(root),
        epochs=1 if smoke else 4,
        extra_opts=options,
        tag=root.name,
    )
    built = base_runner.build_spec(prefix, spec, config, seed, args)
    run_dir.mkdir(parents=True)
    (run_dir / "command.json").write_text(
        json.dumps(built.command, indent=2), encoding="utf-8"
    )
    with (run_dir / "resolved-config.yaml").open("w", encoding="utf-8") as handle:
        subprocess.run(
            built.command + ["--cfg", "job", "--resolve"],
            cwd=built.cwd,
            env={**os.environ, **built.env},
            stdout=handle,
            check=True,
        )
    subprocess.run(
        built.command, cwd=built.cwd, env={**os.environ, **built.env}, check=True
    )
    expected = checkpoint_dir / (
        "weights_epoch_1.pt" if smoke else "weights_epoch_4.pt"
    )
    if not expected.is_file():
        raise RuntimeError(f"Missing Stage-2 checkpoint: {expected}")
    (run_dir / "complete.json").write_text(
        json.dumps({
            "condition": condition,
            "arm": arm,
            "source_checkpoint": str(source),
            "stage2_checkpoint": str(expected),
            **ARMS[arm],
        }, indent=2),
        encoding="utf-8",
    )


def evaluate(root: Path, arm: str, condition: str, seed: int) -> None:
    run_name = f"{root.name}_{condition}_full_{arm}_seed{seed}"
    output = root / f"{condition}-{arm}-epoch4-eval-seed42.json"
    command = [
        sys.executable,
        str(HERE / "evaluate_pusht_checkpoint.py"),
        "--run-name", run_name,
        "--checkpoint", "weights_epoch_4.pt",
        "--dataset", "pusht_expert_train.h5",
        "--num-eval", "50",
        "--seed", "42",
        "--output", str(output),
    ]
    if condition == "tagged":
        command.append("--tagged")
    subprocess.run(
        command,
        cwd=REPO / "leworldmodel",
        env=runtime_environment(),
        check=True,
    )
    print(f"TWO_STAGE_EVAL_COMPLETE condition={condition} arm={arm}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--condition", choices=("clean", "tagged"), default="clean")
    parser.add_argument("--partition", default="gpu_shared")
    parser.add_argument("--eval-partition", default="interactive")
    parser.add_argument("--eval-nodelist", default="SPGL-1-1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-checkpoint", default="weights_step_85000.pt")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--worker", choices=("smoke", "train", "eval"))
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--arm-index", type=int)
    args = parser.parse_args()
    env = runtime_environment(create_dirs=bool(args.worker or args.submit))
    os.environ.update(env)
    source = source_checkpoint(args.condition, args.source_checkpoint)

    if args.worker:
        if args.campaign is None or args.arm_index is None:
            parser.error("worker mode requires --campaign and --arm-index")
        arm = tuple(ARMS)[args.arm_index]
        manifest = json.loads((args.campaign / "manifest.json").read_text())
        if source_digest() != manifest["source_sha256"]:
            raise RuntimeError("Source/config changed since submission; submit fresh")
        if args.worker == "eval":
            evaluate(args.campaign, arm, args.condition, args.seed)
        else:
            train(
                args.campaign, arm, args.condition, args.seed,
                smoke=args.worker == "smoke", source=source,
            )
        print(f"TWO_STAGE_{args.worker.upper()}_{arm.upper()}_COMPLETE", flush=True)
        return

    print(f"Two-stage {args.condition} PushT pilot")
    print(f"  Stage-1 source: {source}")
    print("  Boundary: step 85k (~60%); Stage 2: 4 epochs; fresh optimizer in all arms")
    for index, (name, arm) in enumerate(ARMS.items()):
        print(f"  [{index}] {name}: {arm['description']}")
    if not args.submit:
        print("PLAN ONLY. Actual launch requires --submit --confirm-run.")
        return
    if not args.confirm_run:
        parser.error("refusing to submit without --confirm-run")
    if not source.is_file():
        parser.error(f"missing Stage-1 checkpoint: {source}")

    subprocess.run([
        sys.executable, "-m", "pytest", "-q",
        str(HERE / "tests/test_stage2_training.py"),
        str(HERE / "tests/test_two_stage_campaign.py"),
        str(HERE / "tests/test_control_objectives.py"),
    ], cwd=REPO, env=env, check=True)

    root = REPO / "leworldmodel/results" / (
        f"two-stage-{args.condition}-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_sha256": source_digest(),
        "condition": args.condition,
        "training_seed": args.seed,
        "source_checkpoint": str(source),
        "switch_step": 85000,
        "stage2_epochs": 4,
        "optimizer_state_resumed": False,
        "arms": ARMS,
        "jobs": {},
    }
    manifest_path = root / "manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def submit_array(worker: str, dependency: str | None = None) -> str:
        command = command_text([
            sys.executable, str(Path(__file__).resolve()),
            "--worker", worker,
            "--condition", args.condition,
            "--campaign", str(root),
            "--arm-index", "$SLURM_ARRAY_TASK_ID",
            "--seed", str(args.seed),
            "--source-checkpoint", args.source_checkpoint,
        ]).replace("'$SLURM_ARRAY_TASK_ID'", '"$SLURM_ARRAY_TASK_ID"')
        batch = [
            "sbatch", "--parsable", "--partition",
            args.eval_partition if worker == "eval" else args.partition,
            "--array", f"0-{len(ARMS)-1}%{args.max_concurrent}",
            "--gres=gpu:1",
            "--cpus-per-task=8" if worker == "eval" else "--cpus-per-task=12",
            "--mem=48G" if worker == "eval" else "--mem=64G",
            "--time=01:00:00" if worker == "eval" else (
                "--time=00:30:00" if worker == "smoke" else "--time=08:00:00"
            ),
            f"--job-name=oe-stage2-{worker}",
            "--chdir", str(REPO),
            "--output", str(root / f"{worker}-%A_%a.out"),
        ]
        if worker == "eval" and args.eval_nodelist:
            batch += ["--nodelist", args.eval_nodelist]
        if dependency:
            batch += ["--dependency=afterok:" + dependency, "--kill-on-invalid-dep=yes"]
        batch += ["--wrap", command]
        response = subprocess.check_output(batch, text=True).strip().split(";")[0]
        if not response.isdigit():
            raise RuntimeError(f"Unexpected sbatch response: {response!r}")
        return response

    save()
    smoke = submit_array("smoke")
    training = submit_array("train", smoke)
    evaluation = submit_array("eval", training)
    manifest["jobs"] = {"smoke": smoke, "train": training, "eval": evaluation}
    save()
    print(f"SMOKE_ARRAY_JOB={smoke}")
    print(f"TRAIN_ARRAY_JOB={training}")
    print(f"EVAL_ARRAY_JOB={evaluation}")
    print(f"CAMPAIGN={root}")


def command_text(parts) -> str:
    return shlex.join(str(value) for value in parts)


if __name__ == "__main__":
    main()
