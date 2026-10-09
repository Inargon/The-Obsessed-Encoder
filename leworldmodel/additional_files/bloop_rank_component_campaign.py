#!/usr/bin/env python3
"""Matched tagged-PushT ablations for Bloop rank and control evidence.

The reference Ours checkpoint is the existing rank-one Inverse+Reach run.  This
campaign trains only the four missing arms by default:

* rank0_both: Inverse+Reach, predictor-only Cycle, no gradient routing;
* rank4_both: the same objective with a four-direction EMA residual bank;
* rank1_inverse: rank-one Bloop guided only by inverse action prediction;
* rank1_reach: rank-one Bloop guided only by reachability.

Every other dataset, optimizer, prediction loss, Cycle boundary, seed, and
evaluation protocol is inherited from the released Ours campaign.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

try:
    from . import interface_cycle_campaign as reference
except ImportError:
    import interface_cycle_campaign as reference


REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARMS = (
    "rank0_both",
    "rank4_both",
    "rank1_inverse",
    "rank1_reach",
)
REFERENCE_CHECKPOINT = (
    "interface-cycle-20261002-064431_tagged_pusht_"
    "bloop_cycle_full_seed0/weights_epoch_10.pt"
)


def _replace(overrides: list[str], key: str, value: str) -> list[str]:
    prefix = f"+{key}="
    result = [item for item in overrides if not item.startswith(prefix)]
    result.append(f"+{key}={value}")
    return result


def arm_spec(arm: str) -> tuple[dict, dict]:
    """Return the matched training config for one ablation arm."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    config, spec = reference.benchmark_spec("tagged_pusht", routing="bloop")
    overrides = list(spec["overrides"])

    if arm == "rank0_both":
        overrides = [
            item for item in overrides if not item.startswith("+loss.bloop.")
        ]
    elif arm == "rank4_both":
        overrides = _replace(overrides, "loss.bloop.rank", "4")
    elif arm == "rank1_inverse":
        overrides = _replace(overrides, "loss.control.inverse_weight", "1.0")
        overrides = _replace(overrides, "loss.control.reachability_weight", "0.0")
    elif arm == "rank1_reach":
        overrides = _replace(overrides, "loss.control.inverse_weight", "0.0")
        overrides = _replace(overrides, "loss.control.reachability_weight", "0.1")

    spec["overrides"] = overrides
    return config, spec


def run_name(root: Path, arm: str, seed: int, smoke: bool) -> str:
    phase = "smoke" if smoke else "full"
    return f"{root.name}_tagged_pusht_{arm}_{phase}_seed{seed}"


def train(root: Path, arm: str, seed: int, smoke: bool) -> None:
    try:
        from . import run as base_runner
    except ImportError:
        import run as base_runner

    env = reference.environment()
    os.environ.update(env)
    for name in ("SPT_CACHE_DIR", "XDG_CACHE_HOME", "TMPDIR"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)

    config, spec = arm_spec(arm)
    name = run_name(root, arm, seed, smoke)
    checkpoint_dir = Path(env["STABLEWM_HOME"]) / "checkpoints" / name
    run_dir = root / name
    if checkpoint_dir.exists() or run_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing run: {name}")

    options = "++checkpoint.every_n_steps=5000 ++eval.every_n_steps=1000000000"
    if smoke:
        options = (
            "++trainer.max_steps=2 "
            "++checkpoint.every_n_steps=1000000 "
            "++eval.every_n_steps=1000000"
        )
    args = argparse.Namespace(
        results_dir=str(root),
        epochs=1 if smoke else 10,
        extra_opts=options,
        tag=root.name,
    )
    built = base_runner.build_spec(
        name.rsplit(f"_seed{seed}", 1)[0], spec, config, seed, args
    )
    run_dir.mkdir()
    (run_dir / "command.json").write_text(
        json.dumps(built.command, indent=2), encoding="utf-8"
    )
    with (run_dir / "resolved-config.yaml").open("w", encoding="utf-8") as handle:
        subprocess.run(
            built.command + ["--cfg", "job", "--resolve"],
            cwd=built.cwd,
            env={**env, **built.env},
            stdout=handle,
            check=True,
        )
    subprocess.run(built.command, cwd=built.cwd, env={**env, **built.env}, check=True)
    expected = checkpoint_dir / (
        "weights_epoch_1.pt" if smoke else "weights_epoch_10.pt"
    )
    if not expected.is_file():
        raise RuntimeError(f"Missing expected checkpoint after training: {expected}")
    (run_dir / "complete.json").write_text(
        json.dumps(
            {"arm": arm, "seed": seed, "checkpoint": str(expected)}, indent=2
        ),
        encoding="utf-8",
    )


def evaluate(root: Path, arm: str, seed: int) -> None:
    env = reference.environment()
    command = [
        sys.executable,
        str(HERE / "evaluate_pusht_checkpoint.py"),
        "--run-name",
        run_name(root, arm, seed, smoke=False),
        "--checkpoint",
        "weights_epoch_10.pt",
        "--num-eval",
        "50",
        "--seed",
        "42",
        "--output",
        str(root / f"{arm}-epoch10-eval-seed42.json"),
        "--dataset",
        "pusht_expert_train.h5",
        "--tagged",
    ]
    subprocess.run(command, cwd=REPO / "leworldmodel", env=env, check=True)
    print(f"BLOOP_ABLATION_EVAL_COMPLETE arm={arm}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--partition", default="gpu_shared")
    parser.add_argument("--eval-partition", default="interactive")
    parser.add_argument("--eval-nodelist", default="SPGL-1-1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--defer-eval", action="store_true")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument(
        "--worker", choices=("smoke", "train", "smoke_train", "eval")
    )
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    os.environ.update(reference.environment())

    if args.worker:
        if not args.campaign or not args.arm:
            parser.error("worker mode requires --campaign and --arm")
        manifest = json.loads((args.campaign / "manifest.json").read_text())
        if reference.source_digest() != manifest["source_sha256"]:
            raise RuntimeError("Source/config changed since submission; submit fresh")
        if args.worker == "eval":
            evaluate(args.campaign, args.arm, args.seed)
        elif args.worker == "smoke_train":
            train(args.campaign, args.arm, args.seed, smoke=True)
            print(f"BLOOP_ABLATION_SMOKE_COMPLETE arm={args.arm}")
            train(args.campaign, args.arm, args.seed, smoke=False)
        else:
            train(args.campaign, args.arm, args.seed, args.worker == "smoke")
        print(f"BLOOP_ABLATION_{args.worker.upper()}_COMPLETE arm={args.arm}")
        return

    arms = list(
        dict.fromkeys(value.strip() for value in args.arms.split(",") if value.strip())
    )
    if not arms or any(value not in ARMS for value in arms):
        parser.error(f"arms must be a nonempty subset of {ARMS}")
    print("Matched tagged-PushT Bloop ablation campaign")
    print(f"  missing arms: {', '.join(arms)}")
    print(f"  reused rank-one reference: {REFERENCE_CHECKPOINT}")
    print("  evaluation: 50 episodes, simulator labels, seed 42")
    if not args.submit:
        print("PLAN ONLY. Pass --submit --confirm-run on the cluster.")
        return
    if not args.confirm_run:
        parser.error("submission requires --confirm-run")
    if args.seed < 0 or args.max_concurrent < 1:
        parser.error("seed must be nonnegative and max-concurrent positive")

    reference.preflight_bloop_runtime()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(HERE / "tests/test_bloop_gradient_routing.py"),
            str(HERE / "tests/test_bloop_rank_component_campaign.py"),
        ],
        cwd=REPO,
        env=reference.environment(),
        check=True,
    )

    root = REPO / "leworldmodel/results" / (
        "bloop-rank-components-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_sha256": reference.source_digest(),
        "training_seed": args.seed,
        "evaluation_seed": 42,
        "evaluation_episodes": 50,
        "arms": arms,
        "rank1_reference_checkpoint": REFERENCE_CHECKPOINT,
        "matching": (
            "same data/model/optimizer/prediction loss/predictor-only Cycle; "
            "only router rank or control evidence changes"
        ),
        "jobs": {},
    }

    def save() -> None:
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def submit_array(
        worker: str,
        partition: str,
        dependency: str | None = None,
        nodelist: str | None = None,
    ) -> str:
        command = shlex.join(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                worker,
                "--campaign",
                str(root),
            ]
        )
        command += f' --arm "$arm" --seed {args.seed}'
        shell = "\n".join(
            (
                "set -euo pipefail",
                f"arms=({' '.join(arms)})",
                'arm="${arms[$SLURM_ARRAY_TASK_ID]}"',
                command,
            )
        )
        batch = [
            "sbatch",
            "--parsable",
            "--partition",
            partition,
            f"--array=0-{len(arms) - 1}%{args.max_concurrent}",
            "--gres=gpu:1",
            "--cpus-per-task=12" if worker != "eval" else "--cpus-per-task=8",
            "--mem=64G" if worker != "eval" else "--mem=48G",
            "--time=00:30:00"
            if worker == "smoke"
            else (
                "--time=36:00:00"
                if worker in {"train", "smoke_train"}
                else "--time=01:00:00"
            ),
            f"--job-name=oe-bloop-ab-{worker}",
            "--chdir",
            str(REPO),
            "--output",
            str(root / f"{worker}-%A_%a.out"),
        ]
        if nodelist:
            batch += ["--nodelist", nodelist]
        if dependency:
            batch += [
                "--dependency=afterok:" + dependency,
                "--kill-on-invalid-dep=yes",
            ]
        batch += ["--wrap", shell]
        job = subprocess.check_output(batch, text=True).strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(f"Unexpected sbatch response: {job!r}")
        manifest["jobs"][worker] = job
        save()
        print(f"SUBMITTED {job} {worker}", flush=True)
        return job

    save()
    # One array slot performs its own two-step smoke gate before the full run.
    # This halves the number of submitted array tasks under strict QOS limits.
    training = submit_array("smoke_train", args.partition)
    evaluation = None
    if not args.defer_eval:
        evaluation = submit_array(
            "eval", args.eval_partition, training, args.eval_nodelist
        )
    print(f"BLOOP_ABLATION_TRAIN_JOB={training}")
    print(f"BLOOP_ABLATION_EVAL_JOB={evaluation or 'DEFERRED'}")
    print(f"CAMPAIGN={root}")


if __name__ == "__main__":
    main()
