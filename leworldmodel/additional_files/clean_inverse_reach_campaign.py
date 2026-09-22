"""Train clean Aligned-IR on Reacher, Cube, and TwoRoom.

Aligned-IR keeps the existing aligned gradient router and the inverse and
reachability terms of Lctrl, while disabling only the cycle term.  Each task
has an independent two-step smoke test and a 10-epoch training job, allowing
the three full runs to execute in parallel when resources are available.
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
TASKS = ("reacher", "cube", "tworoom")
TASK_ARMS = {
    "reacher": "reacher_clean_aligned",
    "cube": "cube_clean_aligned",
    "tworoom": "tworoom_clean_aligned",
}


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
    env.pop("PYOPENGL_PLATFORM", None)
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


def load_task_config(task: str) -> dict:
    if task == "reacher":
        try:
            from .run_reacher_control import load_reacher_configs
        except ImportError:
            from run_reacher_control import load_reacher_configs

        return load_reacher_configs()
    if task == "cube":
        try:
            from .run_cube_control import load_cube_configs
        except ImportError:
            from run_cube_control import load_cube_configs

        return load_cube_configs()
    if task == "tworoom":
        try:
            from .run_tworoom_control import load_tworoom_configs
        except ImportError:
            from run_tworoom_control import load_tworoom_configs

        return load_tworoom_configs()
    raise ValueError(f"Unknown task: {task}")


def task_spec(task: str) -> tuple[dict, dict]:
    """Return the clean aligned arm with only cycle loss disabled."""
    config = load_task_config(task)
    spec = copy.deepcopy(config["arms"][TASK_ARMS[task]])
    overrides = list(spec["overrides"])
    found_cycle = False
    found_reach = False
    for index, value in enumerate(overrides):
        if value.startswith("+loss.control.cycle_weight="):
            overrides[index] = "+loss.control.cycle_weight=0.0"
            found_cycle = True
        elif value.startswith("+loss.control.reachability_weight="):
            found_reach = float(value.split("=", 1)[1]) > 0.0
    if not found_cycle:
        raise RuntimeError(f"{task} aligned arm has no cycle weight override")
    if not found_reach:
        raise RuntimeError(f"{task} aligned arm has no positive reachability weight")
    if any("pixel_tag" in value for value in overrides):
        raise RuntimeError(f"{task} clean arm unexpectedly enables pixel tags")
    spec["overrides"] = overrides
    return config, spec


def train(root: Path, task: str, seed: int, smoke: bool) -> None:
    try:
        from . import run as base_runner
    except ImportError:
        import run as base_runner

    env = environment()
    os.environ.update(env)
    for name in ("SPT_CACHE_DIR", "XDG_CACHE_HOME", "TMPDIR"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)

    config, spec = task_spec(task)
    phase = "smoke" if smoke else "full"
    prefix = f"{root.name}_{task}_clean_inverse_reach_{phase}"
    run_name = f"{prefix}_seed{seed}"
    checkpoint_dir = Path(env["STABLEWM_HOME"]) / "checkpoints" / run_name
    run_dir = root / run_name
    if checkpoint_dir.exists() or run_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing run: {run_name}")

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
    built = base_runner.build_spec(prefix, spec, config, seed, args)
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
    subprocess.run(
        built.command,
        cwd=built.cwd,
        env={**env, **built.env},
        check=True,
    )
    expected = checkpoint_dir / (
        "weights_epoch_1.pt" if smoke else "weights_epoch_10.pt"
    )
    if not expected.is_file():
        raise RuntimeError(f"Missing expected checkpoint after training: {expected}")
    (run_dir / "complete.json").write_text(
        json.dumps(
            {
                "task": task,
                "variant": "aligned inverse + reachability",
                "seed": seed,
                "checkpoint": str(expected),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--partition", default="gpu_shared")
    parser.add_argument(
        "--tasks",
        default=",".join(TASKS),
        help="Comma-separated subset of reacher,cube,tworoom",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker", choices=("smoke", "train"))
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--task", choices=TASKS)
    args = parser.parse_args()
    os.environ.update(environment())

    if args.worker:
        if not args.campaign or not args.task:
            parser.error("worker mode requires --campaign and --task")
        manifest = json.loads((args.campaign / "manifest.json").read_text())
        if source_digest() != manifest["source_sha256"]:
            raise RuntimeError("Source/config changed since submission; submit a fresh campaign")
        train(args.campaign, args.task, args.seed, smoke=args.worker == "smoke")
        print(
            f"CLEAN_IR_{args.worker.upper()}_{args.task.upper()}_COMPLETE",
            flush=True,
        )
        return

    tasks = list(dict.fromkeys(value.strip() for value in args.tasks.split(",")))
    if not tasks or any(task not in TASKS for task in tasks):
        parser.error(f"tasks must be a nonempty subset of {TASKS}")
    if args.seed < 0:
        parser.error("seed must be nonnegative")

    print(
        f"Clean Aligned-IR tasks={tuple(tasks)}, seed={args.seed}; "
        "10 epochs, one GPU per independent task."
    )
    print("Each full run depends only on its matching two-step smoke test.")
    print("SR is evaluated every 2000 steps and checkpoints every 5000 steps.")
    if not args.submit:
        print("PLAN ONLY. Pass --submit on the cluster to launch jobs.")
        return

    partitions = (
        subprocess.check_output(["sinfo", "-h", "-o", "%P"], text=True)
        .replace("*", "")
        .split()
    )
    if args.partition not in partitions:
        raise SystemExit(
            f"Unknown partition {args.partition!r}; available: {sorted(set(partitions))}"
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(HERE / "tests/test_clean_inverse_reach_campaign.py"),
            str(HERE / "tests/test_control_objectives.py"),
        ],
        cwd=REPO,
        env=environment(),
        check=True,
    )

    root = REPO / "leworldmodel/results" / (
        "clean-inverse-reach-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_sha256": source_digest(),
        "training_seed": args.seed,
        "tasks": tasks,
        "variant": "clean aligned inverse + reachability; cycle disabled",
        "jobs": [],
    }

    def save() -> None:
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def submit(
        worker: str, task: str, dependency: str | None = None
    ) -> str:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            worker,
            "--campaign",
            str(root),
            "--task",
            task,
            "--seed",
            str(args.seed),
        ]
        batch = [
            "sbatch",
            "--parsable",
            "--partition",
            args.partition,
            "--gres=gpu:1",
            "--cpus-per-task=12",
            "--mem=64G",
            "--time=00:30:00" if worker == "smoke" else "--time=20:00:00",
            f"--job-name=oe-ir-{task[:5]}-{worker}",
            "--chdir",
            str(REPO),
            "--output",
            str(root / "job-%j.out"),
        ]
        if dependency:
            batch += [
                "--dependency=afterok:" + dependency,
                "--kill-on-invalid-dep=yes",
            ]
        batch += ["--wrap", shlex.join(command)]
        job = subprocess.check_output(batch, text=True).strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(f"Unexpected sbatch response: {job!r}")
        manifest["jobs"].append(
            {
                "job_id": job,
                "worker": worker,
                "task": task,
                "seed": args.seed,
                "dependency": dependency,
            }
        )
        save()
        print(f"SUBMITTED {job} {worker} task={task} seed={args.seed}", flush=True)
        return job

    save()
    for task in tasks:
        smoke = submit("smoke", task)
        submit("train", task, smoke)
    print(f"JOB_MANIFEST={root / 'manifest.json'}")


if __name__ == "__main__":
    main()
