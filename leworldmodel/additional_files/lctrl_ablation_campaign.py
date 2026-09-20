"""Submit matched tagged-PushT Lctrl component ablations in parallel.

The four formal runs are independent Slurm jobs. Each depends only on its own
two-step smoke test, so available GPUs can execute all arms concurrently.
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
ARMS = {
    "full": "inverse + cycle + reachability",
    "idm_only": "inverse only",
    "inverse_cycle": "inverse + cycle",
    "inverse_reach": "inverse + reachability",
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
    return env


def arm_spec(arm: str) -> tuple[dict, dict]:
    """Derive every ablation from the existing full arm in memory.

    Keeping these variants out of the shared config grid avoids changing old
    campaign defaults. The returned overrides differ only in active Lctrl
    components; prediction, routing, masking, data, and evaluation stay fixed.
    """
    try:
        from .run_control import load_control_configs
    except ImportError:
        from run_control import load_control_configs

    config = load_control_configs()
    spec = copy.deepcopy(config["arms"]["control_aligned_pred1"])
    overrides = list(spec["overrides"])
    if arm == "full":
        return config, spec
    if arm == "idm_only":
        overrides = [
            "+loss.control.cycle_weight=0.0"
            if item.startswith("+loss.control.cycle_weight=")
            else "+loss.control.reachability_weight=0.0"
            if item.startswith("+loss.control.reachability_weight=")
            else item
            for item in overrides
        ]
    elif arm == "inverse_cycle":
        overrides = [
            "+loss.control.reachability_weight=0.0"
            if item.startswith("+loss.control.reachability_weight=")
            else item
            for item in overrides
        ]
    elif arm == "inverse_reach":
        overrides = [
            "+loss.control.cycle_weight=0.0"
            if item.startswith("+loss.control.cycle_weight=")
            else item
            for item in overrides
        ]
    else:
        raise ValueError(f"Unknown arm: {arm}")
    spec["overrides"] = overrides
    return config, spec


def train(root: Path, arm: str, seed: int, smoke: bool) -> None:
    try:
        from . import run as base_runner
    except ImportError:
        import run as base_runner

    config, spec = arm_spec(arm)
    phase = "smoke" if smoke else "full"
    prefix = f"{root.name}_{phase}_{arm}"
    run_name = f"{prefix}_seed{seed}"
    checkpoint_dir = Path(os.environ["STABLEWM_HOME"]) / "checkpoints" / run_name
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
            env={**os.environ, **built.env},
            stdout=handle,
            check=True,
        )
    subprocess.run(
        built.command,
        cwd=built.cwd,
        env={**os.environ, **built.env},
        check=True,
    )
    expected = checkpoint_dir / (
        "weights_epoch_1.pt" if smoke else "weights_epoch_10.pt"
    )
    if not expected.is_file():
        raise RuntimeError(f"Missing expected checkpoint after training: {expected}")
    (run_dir / "complete.json").write_text(
        json.dumps(
            {"arm": arm, "components": ARMS[arm], "seed": seed, "checkpoint": str(expected)},
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--partition", default="gpu_shared")
    parser.add_argument(
        "--seeds",
        default="0",
        help="Explicit training seeds, for example 0 or 0,1,2 (not a count)",
    )
    parser.add_argument("--worker", choices=("smoke", "train"))
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.environ.update(environment())

    if args.worker:
        if not args.campaign or not args.arm:
            parser.error("worker mode requires --campaign and --arm")
        manifest = json.loads((args.campaign / "manifest.json").read_text())
        if source_digest() != manifest["source_sha256"]:
            raise RuntimeError("Source/config changed since submission; submit a fresh campaign")
        train(args.campaign, args.arm, args.seed, smoke=args.worker == "smoke")
        print(f"LCTRL_{args.worker.upper()}_{args.arm.upper()}_COMPLETE", flush=True)
        return

    seeds = list(dict.fromkeys(int(value) for value in args.seeds.split(",")))
    if not seeds or min(seeds) < 0:
        parser.error("Need nonnegative explicit seeds")
    print(
        f"Matched tagged-PushT Lctrl arms={tuple(ARMS)}, seeds={seeds}; "
        "one GPU per independent job."
    )
    print("Each full run depends only on its matching two-step smoke test.")
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
            str(HERE / "tests/test_lctrl_ablation_campaign.py"),
            str(HERE / "tests/test_control_objectives.py"),
        ],
        cwd=REPO,
        env=environment(),
        check=True,
    )

    root = REPO / "leworldmodel/results" / (
        "lctrl-ablation-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_sha256": source_digest(),
        "training_seeds": seeds,
        "arms": ARMS,
        "jobs": [],
        "comparison": "Only active Lctrl components differ; Full is retrained in campaign.",
    }

    def save() -> None:
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def submit(worker: str, arm: str, seed: int, dependency: str | None = None) -> str:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            worker,
            "--campaign",
            str(root),
            "--arm",
            arm,
            "--seed",
            str(seed),
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
            f"--job-name=oe-lctrl-{arm[:5]}-{worker}",
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
                "arm": arm,
                "seed": seed,
                "dependency": dependency,
            }
        )
        save()
        print(f"SUBMITTED {job} {worker} arm={arm} seed={seed}", flush=True)
        return job

    save()
    for seed in seeds:
        for arm in ARMS:
            smoke = submit("smoke", arm, seed)
            submit("train", arm, seed, smoke)
    print(f"JOB_MANIFEST={root / 'manifest.json'}")


if __name__ == "__main__":
    main()
