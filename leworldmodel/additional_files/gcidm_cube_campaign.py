#!/usr/bin/env python3
"""Run a protocol-matched GC-IDM pilot on the frozen Bloop Cube encoder.

This wrapper deliberately leaves the official GC-IDM implementation untouched.
It creates an inference-only LeWM bundle, then submits three dependent jobs:
embedding extraction, GC-IDM training, and closed-loop evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
DEFAULT_UPSTREAM = Path(
    "/grp01/ids_compcog/song/code/"
    "Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models"
)
DEFAULT_STABLEWM = Path("/grp01/ids_compcog/song/swm")
DEFAULT_PYTHON = Path(
    "/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python"
)
RUN_NAME = "interface-cycle-20261002-203108_clean_cube_bloop_cycle_full_seed0"
CHECKPOINT_NAME = "weights_epoch_10.pt"
DATASET_RELATIVE = Path("datasets/ogbench/cube_single_expert.h5")
TRAINING_ONLY_PREFIXES = ("control_objective.", "bloop_router.")
UPSTREAM_URL = (
    "https://github.com/hdnndh/"
    "Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models.git"
)


def make_inference_state_dict(state_dict: dict) -> tuple[dict, list[str]]:
    """Remove Bloop/control buffers that are absent from inference config."""
    removed = [
        key for key in state_dict if key.startswith(TRAINING_ONLY_PREFIXES)
    ]
    inference = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(TRAINING_ONLY_PREFIXES)
    }
    return inference, removed


def prepare_bundle(
    stablewm_home: Path, campaign: Path
) -> tuple[Path, dict]:
    import torch

    source = stablewm_home / "checkpoints" / RUN_NAME
    config = source / "config.json"
    checkpoint = source / CHECKPOINT_NAME
    missing = [str(path) for path in (config, checkpoint) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Bloop Cube artifacts: {missing}")

    bundle = campaign / "bloop-cube-inference"
    bundle.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config, bundle / "config.json")
    source_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    inference_state, removed = make_inference_state_dict(source_state)
    torch.save(inference_state, bundle / "weights.pt")
    provenance = {
        "source_run": RUN_NAME,
        "source_checkpoint": str(checkpoint),
        "source_tensor_count": len(source_state),
        "inference_tensor_count": len(inference_state),
        "removed_tensor_count": len(removed),
        "removed_prefixes": list(TRAINING_ONLY_PREFIXES),
    }
    (bundle / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return bundle, provenance


def upstream_record(upstream: Path) -> dict:
    required = (
        upstream / "train_idm.py",
        upstream / "eval_idm.py",
        upstream / "idm/model.py",
        upstream / "idm/dataset.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing official GC-IDM checkout files: {missing}. Clone {UPSTREAM_URL}"
        )
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(upstream), "status", "--porcelain"], text=True
        ).strip()
    )
    if dirty:
        raise RuntimeError(f"Official GC-IDM checkout is dirty: {upstream}")
    return {"root": str(upstream), "commit": commit, "dirty": dirty}


def parse_eval_summary(text: str) -> dict:
    success = re.search(r"IDM metrics:.*?'success_rate':\s*([0-9.]+)", text, re.S)
    plan_ms = re.search(r"IDM avg plan time:\s*([0-9.]+)\s*ms", text)
    episode_ms = re.search(r"IDM time:.*?,\s*([0-9.]+)\s*ms/episode", text)
    if not success or not plan_ms:
        raise ValueError("GC-IDM evaluation log does not contain final metrics")
    rate_percent = float(success.group(1))
    return {
        "task": "cube",
        "condition": "clean",
        "method": "Bloop frozen encoder + GC-IDM",
        "num_eval": 50,
        "seed": 42,
        "goal_offset": 25,
        "eval_budget": 50,
        "success_rate": rate_percent / 100.0,
        "success_rate_percent": rate_percent,
        "idm_plan_ms": float(plan_ms.group(1)),
        "idm_ms_per_episode": float(episode_ms.group(1)) if episode_ms else None,
        "bloop_cem_reference_success_rate": 0.80,
    }


def submit(
    *,
    name: str,
    output: Path,
    command: str,
    partition: str,
    dependency: str | None = None,
    time_limit: str = "01:00:00",
    memory: str = "48G",
    nodelist: str | None = None,
) -> str:
    batch = [
        "sbatch", "--parsable", "--partition", partition,
        "--gres=gpu:1", "--cpus-per-task=12", "--mem", memory,
        "--time", time_limit, "--job-name", name,
        "--chdir", str(REPO), "--output", str(output),
    ]
    if dependency:
        batch += ["--dependency=afterok:" + dependency, "--kill-on-invalid-dep=yes"]
    if nodelist:
        batch += ["--nodelist", nodelist]
    batch += ["--wrap", command]
    job = subprocess.check_output(batch, text=True).strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError(f"Unexpected sbatch response: {job!r}")
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--stablewm-home", type=Path, default=DEFAULT_STABLEWM)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--partition", default="interactive")
    parser.add_argument("--nodelist", default="SPGL-1-1")
    parser.add_argument("--worker", choices=("summarize",))
    parser.add_argument("--campaign", type=Path)
    args = parser.parse_args()

    if args.worker == "summarize":
        if not args.campaign:
            parser.error("summary worker requires --campaign")
        log = args.campaign / "eval.txt"
        if not log.is_file():
            raise FileNotFoundError(f"Missing completed evaluation log: {log}")
        result = parse_eval_summary(log.read_text(encoding="utf-8", errors="replace"))
        result["evaluation_log"] = str(log)
        (args.campaign / "summary.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        print("GCIDM_CUBE_SUMMARY_COMPLETE", flush=True)
        return

    source_checkpoint = (
        args.stablewm_home / "checkpoints" / RUN_NAME / CHECKPOINT_NAME
    )
    dataset = args.stablewm_home / DATASET_RELATIVE
    plan = {
        "benchmark": "clean Cube",
        "method": "frozen Bloop encoder + official GC-IDM",
        "source_checkpoint": str(source_checkpoint),
        "dataset": str(dataset),
        "protocol": {
            "action_dim": 5,
            "frameskip": 1,
            "max_goal_horizon": 50,
            "goal_offset": 25,
            "eval_budget": 50,
            "num_eval": 50,
            "evaluation_seed": 42,
            "training_seed": 42,
            "epochs": 200,
            "batch_size": 8192,
            "learning_rate": 3e-4,
            "evaluation_group": "exact GoalEvalCallback fixed-group sampler",
        },
    }
    print(json.dumps(plan, indent=2))
    if not args.submit:
        print("PLAN ONLY: pass --submit --confirm-run to launch.")
        return
    if not args.confirm_run:
        parser.error("submission requires --confirm-run")
    if not args.python.is_file():
        raise FileNotFoundError(f"Python runtime not found: {args.python}")
    if not dataset.is_file():
        raise FileNotFoundError(f"Cube dataset not found: {dataset}")
    official = upstream_record(args.upstream)

    campaign = REPO / "leworldmodel/results" / (
        "gcidm-bloop-cube-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    campaign.mkdir(parents=True, exist_ok=False)
    bundle, conversion = prepare_bundle(args.stablewm_home, campaign)
    embeddings = campaign / "cube-bloop-embeddings.npz"
    idm = campaign / "cube-bloop-gcidm.pt"

    exports = "\n".join(
        (
            "set -euo pipefail",
            f"export PYTHONPATH={shlex.quote(str(args.upstream))}:${{PYTHONPATH:-}}",
            f"export STABLEWM_HOME={shlex.quote(str(args.stablewm_home))}",
            f"export LOCAL_DATASET_DIR={shlex.quote(str(args.stablewm_home))}",
            "export MUJOCO_GL=egl",
            "export PYOPENGL_PLATFORM=egl",
            'export TMPDIR="/grp01/ids_compcog/song/tmp/aluo/$SLURM_JOB_ID"',
            'mkdir -p "$TMPDIR"',
        )
    )
    extract_cmd = exports + "\n" + shlex.join(
        [
            str(args.python), str(args.upstream / "train_idm.py"), "extract",
            "--checkpoint", str(bundle), "--h5", str(dataset),
            "--output", str(embeddings), "--batch-size", "512",
            "--num-prefetch", "8", "--device", "cuda:0",
        ]
    )
    train_cmd = exports + "\n" + shlex.join(
        [
            str(args.python), str(args.upstream / "train_idm.py"), "train",
            "--embeddings", str(embeddings), "--output", str(idm),
            "--embed-dim", "192", "--action-dim", "5", "--frameskip", "1",
            "--max-goal-horizon", "50", "--epochs", "200",
            "--lr", "0.0003", "--batch-size", "8192", "--seed", "42",
            "--device", "cuda:0",
        ]
    )
    official_eval = shlex.join(
        [
            str(args.python), str(REPO / "leworldmodel/additional_files/gcidm_matched_eval.py"),
            "--dataset", "cube", "--checkpoint", str(bundle),
            "--idm", str(idm), "--num-eval", "50", "--goal-offset", "25",
            "--eval-budget", "50", "--seed", "42", "--device", "cuda:0",
        ]
    )
    summary_command = shlex.join(
        [
            str(args.python), str(Path(__file__).resolve()), "--worker", "summarize",
            "--campaign", str(campaign),
        ]
    )
    eval_cmd = (
        exports
        + "\n"
        + official_eval
        + " 2>&1 | tee "
        + shlex.quote(str(campaign / "eval.txt"))
        + "\n"
        + summary_command
    )

    jobs: dict[str, str] = {}
    jobs["extract"] = submit(
        name="oe-gcidm-cube-extract", output=campaign / "extract-%j.out",
        command=extract_cmd, partition=args.partition, nodelist=args.nodelist,
        time_limit="04:00:00", memory="64G",
    )
    jobs["train"] = submit(
        name="oe-gcidm-cube-train", output=campaign / "train-%j.out",
        command=train_cmd, partition=args.partition, nodelist=args.nodelist,
        dependency=jobs["extract"], time_limit="02:00:00", memory="32G",
    )
    jobs["eval"] = submit(
        name="oe-gcidm-cube-eval", output=campaign / "eval-%j.out",
        command=eval_cmd, partition=args.partition, nodelist=args.nodelist,
        dependency=jobs["train"], time_limit="01:00:00", memory="48G",
    )
    manifest = {
        **plan,
        "official_upstream": official,
        "inference_bundle": str(bundle),
        "conversion": conversion,
        "jobs": jobs,
    }
    (campaign / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for phase, job in jobs.items():
        print(f"GCIDM_CUBE_{phase.upper()}_JOB={job}")
    print(f"CAMPAIGN={campaign}")


if __name__ == "__main__":
    main()
