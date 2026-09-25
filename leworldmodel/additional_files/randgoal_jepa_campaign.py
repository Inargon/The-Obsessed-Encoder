"""Submit a matched pure-JEPA baseline on the existing RandGoal dataset.

The campaign deliberately changes no data, augmentation, or model settings.
It reuses the published ``randgoal`` arm, gives the run a fresh checkpoint
namespace, disables online environment evaluation during training, and gates
the 10-epoch run behind a two-step smoke test.  Final success rate must be
measured post hoc with the same fixed RandGoal evaluator used for the method
checkpoint.
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
DATASET = "pusht_scripted_goal_train.lance"


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


def setup_environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + str(REPO / "leworldmodel")
    env["WANDB_MODE"] = "disabled"
    env["MUJOCO_GL"] = "egl"
    env.pop("PYOPENGL_PLATFORM", None)
    env.setdefault("STABLEWM_HOME", "/grp01/ids_compcog/song/swm")
    env.setdefault("LOCAL_DATASET_DIR", env["STABLEWM_HOME"])
    env.setdefault(
        "SPT_CACHE_DIR",
        "/grp01/ids_compcog/song/cache/stable-pretraining",
    )
    job_id = env.get("SLURM_JOB_ID", "local")
    tmpdir = Path("/grp01/ids_compcog/song/tmp/aluo") / job_id
    tmpdir.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmpdir)
    return env


def jepa_spec() -> tuple[dict, dict]:
    """Return the canonical RandGoal arm and reject accidental treatments."""
    try:
        from . import run as base_runner
    except ImportError:
        import run as base_runner

    config = base_runner.load_configs()
    spec = config["arms"]["randgoal"]
    if spec["overrides"] != [f"data.dataset.name={DATASET}"]:
        raise RuntimeError(f"Unexpected RandGoal training overrides: {spec['overrides']}")
    if spec["eval_overrides"] != [f"+eval.dataset_name={DATASET}"]:
        raise RuntimeError(f"Unexpected RandGoal eval overrides: {spec['eval_overrides']}")
    forbidden = ("loss.control", "aligned_gradient_routing", "pixel_tag")
    flattened = " ".join(spec["overrides"] + spec["eval_overrides"])
    if any(token in flattened for token in forbidden):
        raise RuntimeError("Pure-JEPA baseline unexpectedly contains a treatment")
    return config, spec


def train(campaign: Path, seed: int, smoke: bool) -> None:
    try:
        from . import run as base_runner
    except ImportError:
        import run as base_runner

    config, spec = jepa_spec()
    phase = "smoke" if smoke else "full"
    prefix = f"{campaign.name}_{phase}_randgoal_jepa"
    run_name = f"{prefix}_seed{seed}"
    checkpoint_dir = Path(os.environ["STABLEWM_HOME"]) / "checkpoints" / run_name
    run_dir = campaign / run_name
    if checkpoint_dir.exists() or run_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing run: {run_name}")

    options = "++checkpoint.every_n_steps=5000 ++eval.every_n_steps=1000000"
    if smoke:
        options = (
            "++trainer.max_steps=2 "
            "++checkpoint.every_n_steps=1000000 "
            "++eval.every_n_steps=1000000"
        )
    args = argparse.Namespace(
        results_dir=str(campaign),
        epochs=1 if smoke else 10,
        extra_opts=options,
        tag=campaign.name,
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
        raise RuntimeError(f"Missing expected checkpoint: {expected}")
    (run_dir / "complete.json").write_text(
        json.dumps(
            {
                "variant": "pure JEPA",
                "dataset": DATASET,
                "seed": seed,
                "checkpoint": str(expected),
                "smoke": smoke,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"RANDGOAL_JEPA_{phase.upper()}_COMPLETE checkpoint={expected}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--partition", default="gpu_shared")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker", choices=("smoke", "train"))
    parser.add_argument("--campaign", type=Path)
    args = parser.parse_args()
    os.environ.update(setup_environment())

    if args.worker:
        if not args.campaign:
            parser.error("--worker requires --campaign")
        manifest = json.loads((args.campaign / "manifest.json").read_text())
        if source_digest() != manifest["source_sha256"]:
            raise RuntimeError(
                "Source/config changed after submission; create a new campaign"
            )
        train(args.campaign, args.seed, smoke=args.worker == "smoke")
        return

    print(
        f"Matched RandGoal JEPA: dataset={DATASET}, seed={args.seed}, "
        "10 epochs; online SR disabled; checkpoint every 5000 steps."
    )
    if not args.submit:
        print("PLAN ONLY. Pass --submit on the cluster to launch.")
        return
    if args.seed < 0:
        raise SystemExit("seed must be nonnegative")
    partitions = (
        subprocess.check_output(["sinfo", "-h", "-o", "%P"], text=True)
        .replace("*", "")
        .split()
    )
    if args.partition not in partitions:
        raise SystemExit(
            f"Unknown partition {args.partition!r}; available={sorted(set(partitions))}"
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(HERE / "tests/test_randgoal_jepa_campaign.py"),
            str(HERE / "tests/test_configs.py"),
            "-q",
        ],
        cwd=REPO,
        check=True,
    )

    root = REPO / "leworldmodel/results" / (
        "randgoal-jepa-matched-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_sha256": source_digest(),
        "variant": "pure JEPA",
        "dataset": DATASET,
        "training_seed": args.seed,
        "epochs": 10,
        "online_evaluation": False,
        "jobs": [],
    }

    def save() -> None:
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    save()

    def submit(worker: str, dependency: str | None = None) -> str:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            worker,
            "--campaign",
            str(root),
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
            "--job-name=oe-rg-jepa-" + worker,
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
            {"job_id": job, "worker": worker, "dependency": dependency}
        )
        save()
        print(f"SUBMITTED {job} {worker}", flush=True)
        return job

    smoke = submit("smoke")
    train_job = submit("train", smoke)
    print(f"SMOKE_JOB={smoke}")
    print(f"TRAIN_JOB={train_job}")
    print(f"JOB_MANIFEST={root / 'manifest.json'}")


if __name__ == "__main__":
    main()
