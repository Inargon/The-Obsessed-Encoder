#!/usr/bin/env python3
"""Run a protocol-matched JEPA-Bisim adaptation on tagged PushT.

The public JEPA-Bisim method learns a compact planning representation whose
distances match action-conditioned transition distances.  This campaign keeps
our tagged PushT data, ViT-tiny backbone, optimizer, ten-epoch budget, CEM
evaluator, and nuisance interventions fixed.  Only the 32-dimensional planning
projector and reward-free bisimulation objective are added.

This is intentionally reported as a *matched adaptation*, not as a verbatim
reproduction of the authors' frozen-DINO/patched-token implementation.
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
DEFAULT_PYTHON = Path(
    "/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python"
)
DEFAULT_DATA_ROOT = Path("/grp01/ids_compcog/song/swm")
REFERENCE_CHECKPOINTS = (
    "jepa=colored_square_episode_seed0/weights_epoch_10.pt",
    (
        "ours=interface-cycle-20261002-064431_"
        "tagged_pusht_bloop_cycle_full_seed0/weights_epoch_10.pt"
    ),
)


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


def environment(data_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "leworldmodel") + os.pathsep + str(REPO)
    env["WANDB_MODE"] = "disabled"
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    env["STABLEWM_HOME"] = str(data_root)
    env["LOCAL_DATASET_DIR"] = str(data_root)
    env.setdefault(
        "SPT_CACHE_DIR", "/grp01/ids_compcog/song/cache/stable-pretraining"
    )
    env.setdefault("XDG_CACHE_HOME", "/grp01/ids_compcog/song/cache/xdg")
    job_id = env.get("SLURM_JOB_ID", "local")
    env["TMPDIR"] = f"/grp01/ids_compcog/song/tmp/aluo/{job_id}"
    return env


def matched_spec() -> dict:
    return {
        "overrides": [
            "data.dataset.name=pusht_expert_train.h5",
            "+pixel_tag.mode=video",
            "+pixel_tag.size=5",
            # ViT-tiny still emits 192 dimensions; the learned projector and
            # all planner-facing dynamics operate in the paper's 32-D state.
            "embed_dim=32",
            "model.projector.input_dim=192",
            "+loss.bisimulation.enabled=true",
            "+loss.bisimulation.discount=0.99",
            "+loss.bisimulation.weight=1.0",
            "+loss.bisimulation.variance_weight=1.0",
            "+loss.bisimulation.covariance_weight=1.0",
            "+loss.bisimulation.variance_target=1.0",
        ],
        "pair_suites": ["colour"],
        "eval_overrides": ["+eval.tag_mode=video", "+eval.tag_size=5"],
    }


def run_name(campaign: Path, seed: int, smoke: bool) -> str:
    phase = "smoke" if smoke else "full"
    return f"{campaign.name}_tagged_pusht_jepa_bisim_{phase}_seed{seed}"


def train(campaign: Path, seed: int, smoke: bool, data_root: Path) -> None:
    import run as base_runner

    env = environment(data_root)
    os.environ.update(env)
    for name in ("SPT_CACHE_DIR", "XDG_CACHE_HOME", "TMPDIR"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    name = run_name(campaign, seed, smoke)
    checkpoint_dir = data_root / "checkpoints" / name
    run_dir = campaign / name
    if checkpoint_dir.exists() or run_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing run: {name}")

    options = "++checkpoint.every_n_steps=5000 ++eval.every_n_steps=1000000000"
    if smoke:
        options = (
            "++trainer.max_steps=2 ++checkpoint.every_n_steps=1000000 "
            "++eval.every_n_steps=1000000"
        )
    args = argparse.Namespace(
        results_dir=str(campaign),
        epochs=1 if smoke else 10,
        extra_opts=options,
        tag=campaign.name,
    )
    config = {
        "epochs": 10,
        "eval_every_n_steps": 1000000000,
        "pair_every_n_steps": 100,
        "checkpoint_every_n_steps": 5000,
        "preview_episodes": 2,
    }
    prefix = name.rsplit(f"_seed{seed}", 1)[0]
    built = base_runner.build_spec(prefix, matched_spec(), config, seed, args)
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
        raise RuntimeError(f"Missing expected checkpoint: {expected}")
    (run_dir / "complete.json").write_text(
        json.dumps(
            {
                "method": "matched JEPA-Bisim adaptation",
                "seed": seed,
                "planning_latent_dim": 32,
                "checkpoint": str(expected),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def postprocess(campaign: Path, seed: int, data_root: Path) -> None:
    env = environment(data_root)
    name = run_name(campaign, seed, smoke=False)
    checkpoint = f"bisim={name}/weights_epoch_10.pt"
    checkpoint_args: list[str] = []
    for spec in (*REFERENCE_CHECKPOINTS, checkpoint):
        checkpoint_args += ["--checkpoint", spec]

    commands = [
        [
            sys.executable,
            str(HERE / "evaluate_pusht_checkpoint.py"),
            "--run-name", name,
            "--checkpoint", "weights_epoch_10.pt",
            "--dataset", "pusht_expert_train.h5",
            "--tagged",
            "--num-eval", "50",
            "--seed", "42",
            "--output", str(campaign / "tagged-pusht-seed42.json"),
        ],
        [
            sys.executable,
            str(HERE / "diagnose_frozen_representation.py"),
            *checkpoint_args,
            "--dataset", "pusht_expert_train.h5",
            "--cache-dir", str(data_root),
            "--num-clips", "1024",
            "--batch-size", "32",
            "--seed", "73",
            "--ridge-alpha", "1.0",
            "--out", str(campaign / "linear-probe.json"),
        ],
        [
            sys.executable,
            str(HERE / "diagnose_generic_tag_geometry.py"),
            *checkpoint_args,
            "--dataset", "pusht_expert_train.h5",
            "--cache-dir", str(data_root),
            "--num-frames", "512",
            "--batch-size", "32",
            "--tag-size", "5",
            "--tag-seed", "0",
            "--seed", "73",
            "--out", str(campaign / "tag-geometry.json"),
        ],
        [
            sys.executable,
            str(HERE / "diagnose_tag_intervention.py"),
            *checkpoint_args,
            "--dataset", "pusht_expert_train.h5",
            "--num-clips", "128",
            "--candidates", "32",
            "--history", "3",
            "--horizon", "5",
            "--frameskip", "5",
            "--tag-size", "5",
            "--seed", "73",
            "--out", str(campaign / "tag-intervention.json"),
        ],
    ]
    for command in commands:
        subprocess.run(command, cwd=REPO / "leworldmodel", env=env, check=True)
    print("JEPA_BISIM_TAGGED_POSTPROCESS_COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--partition", default="gpu_shared")
    parser.add_argument("--post-partition", default="interactive")
    parser.add_argument("--post-nodelist", default="SPGL-1-1")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker", choices=("smoke", "train", "post"))
    parser.add_argument("--campaign", type=Path)
    args = parser.parse_args()
    os.environ.update(environment(args.data_root))

    if args.worker:
        if args.campaign is None:
            parser.error("--worker requires --campaign")
        manifest = json.loads((args.campaign / "manifest.json").read_text())
        if source_digest() != manifest["source_sha256"]:
            raise RuntimeError("Source/config changed after submission; submit fresh")
        if args.worker == "post":
            postprocess(args.campaign, args.seed, args.data_root)
        else:
            train(
                args.campaign,
                args.seed,
                smoke=args.worker == "smoke",
                data_root=args.data_root,
            )
        print(f"JEPA_BISIM_TAGGED_{args.worker.upper()}_COMPLETE", flush=True)
        return

    missing = []
    for spec in REFERENCE_CHECKPOINTS:
        _, location = spec.split("=", 1)
        if not (args.data_root / "checkpoints" / location).is_file():
            missing.append(location)
    plan = {
        "benchmark": "tagged PushT",
        "method": "matched JEPA-Bisim adaptation",
        "paper": "arXiv:2602.18639",
        "training_seed": args.seed,
        "planning_latent_dim": 32,
        "budget": "10 epochs matched",
        "postprocess": {
            "closed_loop": "50 episodes, seed 42, matched CEM",
            "linear_probe_clips": 1024,
            "geometry_frames": 512,
            "intervention_clips": 128,
            "intervention_candidates": 32,
            "comparators": ["jepa", "ours"],
        },
        "missing_reference_checkpoints": missing,
        "warning": (
            "matched adaptation on the local ViT-tiny stack; not a verbatim "
            "frozen-DINO reproduction"
        ),
    }
    print(json.dumps(plan, indent=2))
    if not args.submit:
        print("PLAN ONLY: pass --submit --confirm-run to launch.")
        return
    if not args.confirm_run:
        parser.error("refusing to submit without --confirm-run")
    if args.seed != 0:
        parser.error("the initial primary experiment is intentionally seed 0")
    if missing:
        parser.error("missing reference checkpoints: " + ", ".join(missing))

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(HERE / "tests/test_jepa_bisim.py"),
            str(HERE / "tests/test_jepa_bisim_tagged_campaign.py"),
        ],
        cwd=REPO,
        check=True,
    )
    root = REPO / "leworldmodel/results" / (
        "jepa-bisim-tagged-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        **plan,
        "source_sha256": source_digest(),
        "jobs": [],
    }

    def save() -> None:
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    def submit(worker: str, dependency: str | None = None) -> str:
        command = [
            args.python,
            Path(__file__).resolve(),
            "--worker", worker,
            "--campaign", root,
            "--python", args.python,
            "--data-root", args.data_root,
            "--seed", str(args.seed),
        ]
        is_post = worker == "post"
        batch = [
            "sbatch", "--parsable",
            "--partition", args.post_partition if is_post else args.partition,
            "--gres=gpu:1",
            "--cpus-per-task=8" if is_post else "--cpus-per-task=12",
            "--mem=64G",
            "--time=05:00:00" if is_post else (
                "--time=00:30:00" if worker == "smoke" else "--time=24:00:00"
            ),
            "--job-name=oe-bisim-" + worker,
            "--chdir", str(REPO),
            "--output", str(root / f"{worker}-%j.out"),
        ]
        if is_post and args.post_nodelist:
            batch += ["--nodelist", args.post_nodelist]
        if dependency:
            batch += [
                "--dependency=afterok:" + dependency,
                "--kill-on-invalid-dep=yes",
            ]
        batch += ["--wrap", shlex.join(str(value) for value in command)]
        job = subprocess.check_output(batch, text=True).strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(f"Unexpected sbatch response: {job!r}")
        manifest["jobs"].append(
            {"job_id": job, "worker": worker, "dependency": dependency}
        )
        save()
        print(f"SUBMITTED {job} {worker}", flush=True)
        return job

    save()
    smoke = submit("smoke")
    training = submit("train", smoke)
    post = submit("post", training)
    print(f"JEPA_BISIM_SMOKE_JOB={smoke}")
    print(f"JEPA_BISIM_TRAIN_JOB={training}")
    print(f"JEPA_BISIM_POST_JOB={post}")
    print(f"CAMPAIGN={root}")


if __name__ == "__main__":
    main()
