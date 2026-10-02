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
BENCHMARKS = (
    "tagged_pusht",
    "clean_pusht",
    "clean_reacher",
    "clean_cube",
    "clean_tworoom",
)
DEFAULT_BENCHMARKS = ("tagged_pusht", "clean_reacher")
ROUTINGS = ("aligned", "conflict_only", "bloop")


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


def preflight_bloop_runtime() -> None:
    """Fail locally before submission if the installed trainer lacks the hook."""
    import stable_pretraining as spt

    if not hasattr(spt.Module, "after_manual_backward"):
        raise RuntimeError(
            "Bloop requires stable-pretraining.Module.after_manual_backward; "
            "the selected Python environment is incompatible"
        )
    print("BLOOP_AFTER_MANUAL_BACKWARD_PREFLIGHT_PASS", flush=True)


def benchmark_spec(
    benchmark: str, routing: str = "aligned"
) -> tuple[dict, dict]:
    if routing not in ROUTINGS:
        raise ValueError(f"unknown routing {routing!r}")
    if benchmark in ("tagged_pusht", "clean_pusht"):
        try:
            from .run_control import load_control_configs
        except ImportError:
            from run_control import load_control_configs

        config = load_control_configs()
        spec = copy.deepcopy(config["arms"]["control_aligned_pred1"])
        if benchmark == "clean_pusht":
            spec["overrides"] = [
                value
                for value in spec["overrides"]
                if not value.startswith("+pixel_tag.")
            ]
            spec["eval_overrides"] = [
                value
                for value in spec.get("eval_overrides", [])
                if not value.startswith("+eval.tag_")
            ]
    elif benchmark == "clean_reacher":
        try:
            from .run_reacher_control import load_reacher_configs
        except ImportError:
            from run_reacher_control import load_reacher_configs

        config = load_reacher_configs()
        spec = copy.deepcopy(config["arms"]["reacher_clean_aligned"])
    elif benchmark == "clean_cube":
        try:
            from .run_cube_control import load_cube_configs
        except ImportError:
            from run_cube_control import load_cube_configs

        config = load_cube_configs()
        spec = copy.deepcopy(config["arms"]["cube_clean_aligned"])
    elif benchmark == "clean_tworoom":
        try:
            from .run_tworoom_control import load_tworoom_configs
        except ImportError:
            from run_tworoom_control import load_tworoom_configs

        config = load_tworoom_configs()
        spec = copy.deepcopy(config["arms"]["tworoom_clean_aligned"])
    else:
        raise ValueError(f"unknown benchmark {benchmark!r}")

    overrides = list(spec["overrides"])
    if not any(value == "+loss.control.cycle_weight=0.5" for value in overrides):
        raise RuntimeError(f"{benchmark} is not the matched Full Cycle arm")
    if not any("aligned_gradient_routing.enabled=true" in value for value in overrides):
        raise RuntimeError(f"{benchmark} does not enable the matched router")
    if routing == "conflict_only":
        overrides.append(
            "+loss.aligned_gradient_routing.routing_mode=conflict_only"
        )
    elif routing == "bloop":
        # Bloop operates on encoder/projector parameter gradients after the
        # joint backward pass, so the embedding-space router must be absent.
        overrides = [
            value
            for value in overrides
            if not value.startswith("+loss.aligned_gradient_routing.")
        ]
        overrides.extend(
            (
                "+loss.bloop.enabled=true",
                "+loss.bloop.decay=0.9",
                "+loss.bloop.auxiliary_weight=1.0",
            )
        )
    overrides.append("+loss.control.cycle_scope=predictor_only")
    spec["overrides"] = overrides
    return config, spec


def run_name(
    root: Path,
    benchmark: str,
    seed: int,
    smoke: bool,
    routing: str = "aligned",
) -> str:
    phase = "smoke" if smoke else "full"
    variant = "interface_cycle" if routing == "aligned" else f"{routing}_cycle"
    return f"{root.name}_{benchmark}_{variant}_{phase}_seed{seed}"


def train(
    root: Path,
    benchmark: str,
    seed: int,
    smoke: bool,
    routing: str = "aligned",
) -> None:
    try:
        from . import run as base_runner
    except ImportError:
        import run as base_runner

    env = environment()
    os.environ.update(env)
    for name in ("SPT_CACHE_DIR", "XDG_CACHE_HOME", "TMPDIR"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)

    config, spec = benchmark_spec(benchmark, routing)
    name = run_name(root, benchmark, seed, smoke, routing)
    checkpoint_dir = Path(env["STABLEWM_HOME"]) / "checkpoints" / name
    run_dir = root / name
    if checkpoint_dir.exists() or run_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing run: {name}")

    # Environment construction is deliberately deferred to the fixed post-hoc
    # evaluation job. Some gpu_shared nodes can train normally but do not expose
    # a usable EGL runtime; an online callback would otherwise kill a healthy
    # training run when it first evaluates at step 2000.
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
                "routing": routing,
                "seed": seed,
                "checkpoint": str(expected),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def evaluate(
    root: Path,
    benchmark: str,
    seed: int,
    routing: str = "aligned",
) -> None:
    env = environment()
    name = run_name(root, benchmark, seed, smoke=False, routing=routing)
    output = root / f"{benchmark}-epoch10-eval-seed42.json"
    if benchmark in ("tagged_pusht", "clean_pusht"):
        script = HERE / "evaluate_pusht_checkpoint.py"
    elif benchmark == "clean_reacher":
        script = HERE / "evaluate_reacher_checkpoint.py"
    else:
        script = HERE / "evaluate_clean_checkpoint.py"
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
    if benchmark in ("tagged_pusht", "clean_pusht"):
        command += ["--dataset", "pusht_expert_train.h5"]
        if benchmark == "tagged_pusht":
            command.append("--tagged")
    elif benchmark == "clean_reacher":
        command += ["--dataset", "dmc/reacher_random.h5"]
    else:
        command += [
            "--task",
            "cube" if benchmark == "clean_cube" else "tworoom",
        ]
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
    parser.add_argument("--routing", choices=ROUTINGS, default="aligned")
    parser.add_argument(
        "--defer-eval",
        action="store_true",
        help=(
            "Submit smoke and training only. Use this for Reacher so the "
            "checkpoint can be evaluated with the locked historical protocol."
        ),
    )
    parser.add_argument(
        "--benchmarks",
        default=",".join(DEFAULT_BENCHMARKS),
        help=(
            "Comma-separated subset of tagged_pusht,clean_pusht,clean_reacher,"
            "clean_cube,clean_tworoom"
        ),
    )
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
            evaluate(args.campaign, args.benchmark, args.seed, args.routing)
        else:
            train(
                args.campaign,
                args.benchmark,
                args.seed,
                smoke=args.worker == "smoke",
                routing=args.routing,
            )
        print(
            f"INTERFACE_CYCLE_{args.worker.upper()}_COMPLETE "
            f"benchmark={args.benchmark}",
            flush=True,
        )
        return

    print("Predictor-only Cycle campaign:")
    print(f"  routing: {args.routing}")
    print("  tagged PushT: test whether IR's shortcut robustness is retained")
    print("  clean Reacher: test whether the predicted-latent interface is repaired")
    print("  clean Cube/TwoRoom: cross-task test of the same interface repair")
    print("  matched Full differs only in Cycle gradient scope")
    benchmarks = list(
        dict.fromkeys(value.strip() for value in args.benchmarks.split(",") if value.strip())
    )
    if not benchmarks or any(value not in BENCHMARKS for value in benchmarks):
        parser.error(f"benchmarks must be a nonempty subset of {BENCHMARKS}")
    if not args.submit:
        print("PLAN ONLY. Pass --submit on the cluster to launch jobs.")
        return
    if args.seed < 0 or args.max_concurrent < 1:
        parser.error("seed must be nonnegative and max-concurrent positive")
    if args.routing == "bloop":
        preflight_bloop_runtime()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(HERE / "tests/test_control_objectives.py"),
            str(HERE / "tests/test_interface_cycle_campaign.py"),
            str(HERE / "tests/test_bloop_gradient_routing.py"),
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
        "benchmarks": benchmarks,
        "variant": "Inverse+Reach guide; predictor-only Cycle through frozen IDM",
        "routing": args.routing,
        "evaluation_submission": (
            "deferred_by_request" if args.defer_eval else "afterok_train"
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
        worker_command += f" --routing {args.routing}"
        shell = "\n".join(
            (
                "set -euo pipefail",
                f"benchmarks=({' '.join(benchmarks)})",
                'benchmark="${benchmarks[$SLURM_ARRAY_TASK_ID]}"',
                worker_command,
            )
        )
        batch = [
            "sbatch",
            "--parsable",
            "--partition",
            partition,
            f"--array=0-{len(benchmarks) - 1}%{args.max_concurrent}",
            "--gres=gpu:1",
            "--cpus-per-task=12" if worker != "eval" else "--cpus-per-task=8",
            "--mem=64G" if worker != "eval" else "--mem=48G",
            "--time=00:30:00" if worker == "smoke" else (
                (
                    "--time=36:00:00"
                    if args.routing == "bloop"
                    else "--time=20:00:00"
                )
                if worker == "train"
                else "--time=01:00:00"
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
    evaluation = None
    if not args.defer_eval:
        evaluation = submit_array(
            "eval", args.eval_partition, training, args.eval_nodelist
        )
    print(f"SMOKE_ARRAY_JOB={smoke}")
    print(f"TRAIN_ARRAY_JOB={training}")
    print(f"EVAL_ARRAY_JOB={evaluation or 'DEFERRED'}")
    print(f"CAMPAIGN={root}")


if __name__ == "__main__":
    main()
