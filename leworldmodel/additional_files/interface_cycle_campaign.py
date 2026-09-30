#!/usr/bin/env python3
"""Train and evaluate predictor-only Cycle on tagged PushT and clean Reacher.

The matched Full method lets Cycle update the encoder, predictor, and inverse
head together.  This campaign changes only that gradient boundary: Inverse and
Reach remain the representation guide, while Cycle starts from a detached
encoder latent and passes through a frozen, real-transition-trained IDM.  Its
gradient therefore calibrates the predictor/action-conditioning path only.
"""

from __future__ import annotations

import argparse
import copy
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
BENCHMARKS = ("tagged_pusht", "clean_reacher")


def source_digest() -> str:
    digest = hashlib.sha256()
    files = list((REPO / "leworldmodel").glob("*.py"))
    for directory in (HERE, REPO / "leworldmodel/config", REPO / "common"):
        files += [
            path
            for path in directory.rglob("*")
            if path.suffix in (".py", ".yaml")
        ]
    for path in sorted(set(files)):
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + str(REPO / "leworldmodel")
    env["WANDB_MODE"] = "disabled"
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    env.setdefault("STABLEWM_HOME", "/grp01/ids_compcog/song/swm")
    env.setdefault("LOCAL_DATASET_DIR", env["STABLEWM_HOME"])
    env.setdefault(
        "SPT_CACHE_DIR", "/grp01/ids_compcog/song/cache/stable-pretraining"
    )
    env.setdefault("XDG_CACHE_HOME", "/grp01/ids_compcog/song/cache/xdg")
    job_id = env.get("SLURM_JOB_ID")
    if job_id:
        env["TMPDIR"] = f"/grp01/ids_compcog/song/tmp/aluo/{job_id}"
    else:
        env.setdefault("TMPDIR", "/grp01/ids_compcog/song/tmp/aluo/local")
    return env


def benchmark_spec(benchmark: str) -> tuple[dict, dict]:
    if benchmark == "tagged_pusht":
        try:
            from .run_control import load_control_configs
        except ImportError:
            from run_control import load_control_configs

        config = load_control_configs()
        spec = copy.deepcopy(config["arms"]["control_aligned_pred1"])
    elif benchmark == "clean_reacher":
        try:
            from .run_reacher_control import load_reacher_configs
        except ImportError:
            from run_reacher_control import load_reacher_configs

        config = load_reacher_configs()
        spec = copy.deepcopy(config["arms"]["reacher_clean_aligned"])
    else:
        raise ValueError(f"unknown benchmark {benchmark!r}")

    overrides = list(spec["overrides"])
    if not any(value == "+loss.control.cycle_weight=0.5" for value in overrides):
        raise RuntimeError(f"{benchmark} is not the matched Full Cycle arm")
    if not any("aligned_gradient_routing.enabled=true" in value for value in overrides):
        raise RuntimeError(f"{benchmark} does not enable the matched router")
    overrides.append("+loss.control.cycle_scope=predictor_only")
    spec["overrides"] = overrides
    return config, spec


def run_name(root: Path, benchmark: str, seed: int, smoke: bool) -> str:
    phase = "smoke" if smoke else "full"
    return f"{root.name}_{benchmark}_interface_cycle_{phase}_seed{seed}"


def train(root: Path, benchmark: str, seed: int, smoke: bool) -> None:
    try:
        from . import run as base_runner
    except ImportError:
        import run as base_runner

    env = environment()
    os.environ.update(env)
    for name in ("SPT_CACHE_DIR", "XDG_CACHE_HOME", "TMPDIR"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)

    config, spec = benchmark_spec(benchmark)
    name = run_name(root, benchmark, seed, smoke)
    checkpoint_dir = Path(env["STABLEWM_HOME"]) / "checkpoints" / name
    run_dir = root / name
    if checkpoint_dir.exists() or run_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing run: {name}")

    options = "++checkpoint.every_n_steps=5000 ++eval.every_n_steps=2000"
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
    built = base_runner.build_spec(name.rsplit(f"_seed{seed}", 1)[0], spec, config, seed, args)
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
            {
                "benchmark": benchmark,
                "variant": "predictor-only Cycle",
                "seed": seed,
                "checkpoint": str(expected),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def evaluate(root: Path, benchmark: str, seed: int) -> None:
    env = environment()
    name = run_name(root, benchmark, seed, smoke=False)
    output = root / f"{benchmark}-epoch10-eval-seed42.json"
    script = (
        HERE / "evaluate_pusht_checkpoint.py"
        if benchmark == "tagged_pusht"
        else HERE / "evaluate_reacher_checkpoint.py"
    )
    command = [
        sys.executable,
        str(script),
        "--run-name",
        name,
        "--checkpoint",
        "weights_epoch_10.pt",
        "--num-eval",
        "50",
        "--seed",
        "42",
        "--output",
        str(output),
    ]
    if benchmark == "tagged_pusht":
        command += ["--dataset", "pusht_expert_train.h5", "--tagged"]
    else:
        command += ["--dataset", "dmc/reacher_random.h5"]
    subprocess.run(command, cwd=REPO / "leworldmodel", env=env, check=True)
    print(f"INTERFACE_CYCLE_EVAL_COMPLETE benchmark={benchmark}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--partition", default="gpu_shared")
    parser.add_argument("--eval-partition", default="interactive")
    parser.add_argument("--eval-nodelist", default="SPGL-1-1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--worker", choices=("smoke", "train", "eval"))
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--benchmark", choices=BENCHMARKS)
    args = parser.parse_args()
    os.environ.update(environment())

    if args.worker:
        if not args.campaign or not args.benchmark:
            parser.error("worker mode requires --campaign and --benchmark")
        manifest = json.loads((args.campaign / "manifest.json").read_text())
        if source_digest() != manifest["source_sha256"]:
            raise RuntimeError("Source/config changed since submission; submit fresh")
        if args.worker == "eval":
            evaluate(args.campaign, args.benchmark, args.seed)
        else:
            train(
                args.campaign,
                args.benchmark,
                args.seed,
                smoke=args.worker == "smoke",
            )
        print(
            f"INTERFACE_CYCLE_{args.worker.upper()}_COMPLETE "
            f"benchmark={args.benchmark}",
            flush=True,
        )
        return

    print("Predictor-only Cycle campaign:")
    print("  tagged PushT: test whether IR's shortcut robustness is retained")
    print("  clean Reacher: test whether the predicted-latent interface is repaired")
    print("  matched Full differs only in Cycle gradient scope")
    if not args.submit:
        print("PLAN ONLY. Pass --submit on the cluster to launch jobs.")
        return
    if args.seed < 0 or args.max_concurrent < 1:
        parser.error("seed must be nonnegative and max-concurrent positive")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(HERE / "tests/test_control_objectives.py"),
            str(HERE / "tests/test_interface_cycle_campaign.py"),
        ],
        cwd=REPO,
        env=environment(),
        check=True,
    )

    root = REPO / "leworldmodel/results" / (
        "interface-cycle-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_sha256": source_digest(),
        "training_seed": args.seed,
        "evaluation_seed": 42,
        "evaluation_episodes": 50,
        "benchmarks": list(BENCHMARKS),
        "variant": "Inverse+Reach guide; predictor-only Cycle through frozen IDM",
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
        worker_command = shlex.join(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                worker,
                "--campaign",
                str(root),
            ]
        )
        worker_command += f' --benchmark "$benchmark" --seed {args.seed}'
        shell = "\n".join(
            (
                "set -euo pipefail",
                f"benchmarks=({' '.join(BENCHMARKS)})",
                'benchmark="${benchmarks[$SLURM_ARRAY_TASK_ID]}"',
                worker_command,
            )
        )
        batch = [
            "sbatch",
            "--parsable",
            "--partition",
            partition,
            f"--array=0-{len(BENCHMARKS) - 1}%{args.max_concurrent}",
            "--gres=gpu:1",
            "--cpus-per-task=12" if worker != "eval" else "--cpus-per-task=8",
            "--mem=64G" if worker != "eval" else "--mem=48G",
            "--time=00:30:00" if worker == "smoke" else (
                "--time=20:00:00" if worker == "train" else "--time=01:00:00"
            ),
            f"--job-name=oe-ifcycle-{worker}",
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
    smoke = submit_array("smoke", args.partition)
    training = submit_array("train", args.partition, smoke)
    evaluation = submit_array(
        "eval", args.eval_partition, training, args.eval_nodelist
    )
    print(f"SMOKE_ARRAY_JOB={smoke}")
    print(f"TRAIN_ARRAY_JOB={training}")
    print(f"EVAL_ARRAY_JOB={evaluation}")
    print(f"CAMPAIGN={root}")


if __name__ == "__main__":
    main()
