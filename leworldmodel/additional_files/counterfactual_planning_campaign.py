#!/usr/bin/env python3
"""Submit a smoke-gated full Tagged PushT counterfactual-planning run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def source_digest() -> str:
    digest = hashlib.sha256()
    files = list((REPO / "leworldmodel").glob("*.py"))
    for directory in (HERE, REPO / "leworldmodel/config", REPO / "common"):
        files += [p for p in directory.rglob("*") if p.suffix in (".py", ".yaml")]
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


def arm_spec(ranking_metric: str = "js", relative_cost_only: bool = False) -> dict:
    return {
        "overrides": [
            "data.dataset.name=pusht_expert_train.h5",
            "+pixel_tag.mode=video",
            "+pixel_tag.size=5",
            "+pixel_tag.counterfactual_key=pixels_counterfactual",
            "+loss.pred_weight=1.0",
            "+loss.counterfactual_planning.enabled=true",
            f"+loss.counterfactual_planning.representation_weight={0 if relative_cost_only else 0.1}",
            "+loss.counterfactual_planning.action_weight=1.0",
            "+loss.counterfactual_planning.ranking_weight=0.1",
            "+loss.counterfactual_planning.margin=0.05",
            "+loss.counterfactual_planning.temperature=0.1",
            f"+loss.counterfactual_planning.ranking_metric={ranking_metric}",
        ],
        "pair_suites": ["colour"],
        "eval_overrides": ["+eval.tag_mode=video", "+eval.tag_size=5"],
    }


def training_config() -> dict:
    return {
        "epochs": 10,
        "eval_every_n_steps": 2000,
        "pair_every_n_steps": 100,
        "checkpoint_every_n_steps": 5000,
        "preview_episodes": 2,
    }


def train(
    root: Path, seed: int, phase: str, manifest: dict,
) -> None:
    import run as base_runner

    prefix = f"{root.name}_{phase if phase != 'train' else 'full'}_cfplan"
    name = f"{prefix}_seed{seed}"
    checkpoint = Path(os.environ["STABLEWM_HOME"]) / "checkpoints" / name
    target = root / name
    if checkpoint.exists() or target.exists():
        raise RuntimeError(f"Refusing existing run: {name}; use a fresh campaign")
    options = "++checkpoint.every_n_steps=5000 ++eval.every_n_steps=2000"
    arm = arm_spec(manifest.get("ranking_metric", "js"), manifest.get("relative_cost_only", False))
    if phase == "smoke":
        options = (
            "++trainer.max_steps=2 ++checkpoint.every_n_steps=1000000 "
            "++eval.every_n_steps=1000000"
        )
    elif phase == "pilot":
        options = (
            f"++trainer.max_steps={manifest['pilot_steps']} ++trainer.log_every_n_steps=4 "
            "++trainer.limit_val_batches=0 "
            "++checkpoint.every_n_steps=1000000 ++eval.every_n_steps=1000000 "
            "++pair.every_n_steps=1000000"
        )
        arm["overrides"] += [
            "+stage2.enabled=true",
            f"+stage2.init_checkpoint={manifest['init_checkpoint']}",
            "+stage2.freeze_representation=false",
            "+optimizer_scheduler.type=ConstantLR",
            "+optimizer_scheduler.factor=1.0",
            "optimizer.lr=1e-5",
            "+loss.counterfactual_planning.action_margin_reference_only=true",
        ]
    args = argparse.Namespace(
        results_dir=str(root),
        epochs=10 if phase == "train" else 1,
        extra_opts=options,
        tag=root.name,
    )
    built = base_runner.build_spec(prefix, arm, training_config(), seed, args)
    target.mkdir()
    (target / "command.json").write_text(json.dumps(built.command, indent=2))
    with (target / "resolved-config.yaml").open("w") as stream:
        subprocess.run(
            built.command + ["--cfg", "job", "--resolve"],
            cwd=built.cwd,
            env={**os.environ, **built.env},
            stdout=stream,
            check=True,
        )
    subprocess.run(
        built.command,
        cwd=built.cwd,
        env={**os.environ, **built.env},
        check=True,
    )
    expected = checkpoint / (
        "weights_epoch_10.pt" if phase == "train" else "weights_epoch_1.pt"
    )
    if not expected.is_file():
        raise RuntimeError(f"Missing expected checkpoint: {expected}")
    (target / "complete.json").write_text(
        json.dumps({"checkpoint": str(expected), "seed": seed})
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--smoke-only", action="store_true")
    budget.add_argument("--pilot-steps", type=int)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--ranking-metric", choices=("js", "centered_mse", "margin_guard"), default="js")
    parser.add_argument("--relative-cost-only", action="store_true")
    parser.add_argument("--partition", default="gpu_shared")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--worker", choices=("smoke", "pilot", "train"))
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.environ.update(environment())

    if args.worker:
        if not args.campaign:
            parser.error("--worker requires --campaign")
        manifest = json.loads((args.campaign / "manifest.json").read_text())
        if source_digest() != manifest["source_sha256"]:
            raise RuntimeError("Source/config changed since submission")
        train(args.campaign, args.seed, args.worker, manifest)
        print(f"COUNTERFACTUAL_{args.worker.upper()}_COMPLETE", flush=True)
        return

    seeds = list(dict.fromkeys(int(item) for item in args.seeds.split(",")))
    if not seeds or min(seeds) < 0:
        parser.error("Need explicit nonnegative seeds")
    if args.pilot_steps is not None:
        if args.pilot_steps < 1 or args.init_checkpoint is None:
            parser.error("--pilot-steps requires a positive step count and --init-checkpoint")
        if args.submit and not args.init_checkpoint.is_file():
            parser.error(f"Initialization checkpoint missing: {args.init_checkpoint}")
        budget_description = f"{args.pilot_steps}-step warm-start pilot"
    elif args.init_checkpoint is not None:
        parser.error("--init-checkpoint requires --pilot-steps")
    else:
        budget_description = "2-step smoke only" if args.smoke_only else "10 epochs after smoke"
    print(f"Tagged PushT counterfactual planning: seeds={seeds}, {budget_description}.")
    print(f"Ranking metric: {args.ranking_metric}; relative-cost-only: {args.relative_cost_only}")
    print("No Lctrl, aligned routing, or privileged state is enabled.")
    if not args.submit:
        print("PLAN ONLY. Pass --submit on the cluster.")
        return
    partitions = (
        subprocess.check_output(["sinfo", "-h", "-o", "%P"], text=True)
        .replace("*", "")
        .split()
    )
    if args.partition not in partitions:
        raise SystemExit(f"Unknown partition {args.partition!r}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = REPO / "leworldmodel/results" / f"counterfactual-planning-{args.ranking_metric}-{timestamp}"
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_sha256": source_digest(),
        "training_seeds": seeds,
        "ranking_metric": args.ranking_metric,
        "relative_cost_only": args.relative_cost_only,
        "smoke_only": args.smoke_only,
        "pilot_steps": args.pilot_steps,
        "init_checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
        "jobs": [],
    }

    def save() -> None:
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    save()

    def submit(worker: str, seed: int, dependency: str | None = None) -> str:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            worker,
            "--campaign",
            str(root),
            "--seed",
            str(seed),
        ]
        batch = [
            "sbatch", "--parsable", "--partition", args.partition,
            "--gres=gpu:1", "--cpus-per-task=12", "--mem=64G",
            "--time=20:00:00" if worker == "train" else "--time=00:30:00",
            "--job-name=oe-cfplan-" + worker,
            "--chdir", str(REPO), "--output", str(root / "job-%j.out"),
        ]
        if dependency:
            batch += ["--dependency=afterok:" + dependency, "--kill-on-invalid-dep=yes"]
        batch += ["--wrap", shlex.join(command)]
        job = subprocess.check_output(batch, text=True).strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(f"Unexpected sbatch response {job!r}")
        manifest["jobs"].append(
            {"job_id": job, "worker": worker, "seed": seed, "dependency": dependency}
        )
        save()
        print(f"SUBMITTED {job} {worker} seed={seed}", flush=True)
        return job

    if args.pilot_steps is not None:
        for seed in seeds:
            submit("pilot", seed)
    elif args.smoke_only:
        for seed in seeds:
            submit("smoke", seed)
    else:
        smoke = submit("smoke", 0)
        for seed in seeds:
            submit("train", seed, smoke)
    print(f"JOB_MANIFEST={root / 'manifest.json'}")


if __name__ == "__main__":
    main()
