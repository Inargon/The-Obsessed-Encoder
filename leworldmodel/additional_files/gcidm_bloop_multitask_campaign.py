#!/usr/bin/env python3
"""Run official GC-IDM on frozen Bloop TwoRoom and clean PushT encoders."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from additional_files.gcidm_cube_campaign import (
    DEFAULT_PYTHON,
    DEFAULT_STABLEWM,
    DEFAULT_UPSTREAM,
    HDF5_BOOTSTRAP,
    REPO,
    make_inference_state_dict,
    parse_eval_summary,
    submit,
    upstream_record,
)
HISTORICAL_ROOT = Path("/grp01/ids_compcog/song/code/le-wm-repro-20260514")
HISTORICAL_PYTHON = Path(
    "/grp01/ids_compcog/song/envs/lewm-repro-py310/bin/python"
)
HISTORICAL_SWM = Path(
    "/grp01/ids_compcog/song/code/stable-worldmodel-repro-20260514"
)


TASKS = {
    "tworoom": {
        "run_name": (
            "interface-cycle-20261002-203108_"
            "clean_tworoom_bloop_cycle_full_seed0"
        ),
        "checkpoint": "weights_epoch_10.pt",
        "dataset": "datasets/tworoom.h5",
        "official_dataset": "tworoom",
        "action_dim": 2,
        "cem_reference": 0.96,
        "runtime": "modern",
    },
    "reacher": {
        "run_name": (
            "interface-cycle-20261002-064431_"
            "clean_reacher_bloop_cycle_full_seed0"
        ),
        "checkpoint": "weights_epoch_10.pt",
        "dataset": "datasets/dmc/reacher_random.h5",
        "official_dataset": "reacher",
        "action_dim": 2,
        "cem_reference": 0.88,
        "runtime": "historical",
    },
    "clean_pusht": {
        "run_name": (
            "interface-cycle-20261003-075323_"
            "clean_pusht_bloop_cycle_full_seed0"
        ),
        "checkpoint": "weights_epoch_10.pt",
        "dataset": "datasets/pusht_expert_train.h5",
        "official_dataset": "pusht",
        "action_dim": 2,
        "cem_reference": None,
        "runtime": "modern",
    },
}


def parse_tasks(value: str) -> list[str]:
    tasks = list(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not tasks or any(task not in TASKS for task in tasks):
        raise ValueError(f"tasks must be a nonempty subset of {tuple(TASKS)}")
    return tasks


def prepare_task_bundle(
    stablewm_home: Path, campaign: Path, task: str
) -> tuple[Path, dict]:
    import torch

    spec = TASKS[task]
    source = stablewm_home / "checkpoints" / spec["run_name"]
    config = source / "config.json"
    checkpoint = source / spec["checkpoint"]
    missing = [str(path) for path in (config, checkpoint) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {task} Bloop artifacts: {missing}")
    task_root = campaign / task
    bundle = task_root / "inference"
    bundle.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config, bundle / "config.json")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if spec["runtime"] == "historical":
        from additional_files.convert_reacher_historical_checkpoint import (
            convert_state_dict as convert_reacher_state_dict,
        )

        inference, removed, renamed = convert_reacher_state_dict(state)
        if len(renamed) != 192:
            raise RuntimeError(
                f"expected 192 historical encoder renames, found {len(renamed)}"
            )
    else:
        inference, removed = make_inference_state_dict(state)
        renamed = []
    torch.save(inference, bundle / "weights.pt")
    provenance = {
        "source_run": spec["run_name"],
        "source_checkpoint": str(checkpoint),
        "source_tensor_count": len(state),
        "inference_tensor_count": len(inference),
        "removed_tensor_count": len(removed),
        "renamed_tensor_count": len(renamed),
        "runtime": spec["runtime"],
    }
    (bundle / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return bundle, provenance


def runtime_environment(manifest: dict, task: str) -> dict[str, str]:
    env = dict(os.environ)
    upstream = manifest["official_upstream"]["root"]
    if TASKS[task]["runtime"] == "historical":
        roots = (
            upstream,
            manifest["historical_swm_root"],
            manifest["historical_root"],
            str(REPO / "leworldmodel"),
            str(REPO),
            env.get("PYTHONPATH", ""),
        )
    else:
        roots = (
            upstream,
            str(REPO / "leworldmodel"),
            str(REPO),
            env.get("PYTHONPATH", ""),
        )
    env["PYTHONPATH"] = os.pathsep.join(roots)
    stablewm = manifest["stablewm_home"]
    env["STABLEWM_HOME"] = stablewm
    env["LOCAL_DATASET_DIR"] = stablewm
    env["MUJOCO_GL"] = "egl"
    if TASKS[task]["runtime"] == "historical":
        env.pop("PYOPENGL_PLATFORM", None)
    else:
        env["PYOPENGL_PLATFORM"] = "egl"
    job_id = env.get("SLURM_JOB_ID", "local")
    env["TMPDIR"] = f"/grp01/ids_compcog/song/tmp/aluo/{job_id}"
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def run_worker(campaign: Path, task: str, phase: str) -> None:
    manifest = json.loads((campaign / "manifest.json").read_text())
    if task not in manifest["tasks"]:
        raise ValueError(f"task {task} is absent from campaign")
    spec = TASKS[task]
    python = (
        manifest["historical_python"]
        if spec["runtime"] == "historical"
        else manifest["python"]
    )
    upstream = Path(manifest["official_upstream"]["root"])
    stablewm = Path(manifest["stablewm_home"])
    task_root = campaign / task
    bundle = task_root / "inference"
    embeddings = task_root / "embeddings.npz"
    idm = task_root / "gcidm.pt"
    env = runtime_environment(manifest, task)

    if phase == "extract":
        command = [
            python, "-c", HDF5_BOOTSTRAP, str(upstream / "train_idm.py"),
            "extract", "--checkpoint", str(bundle),
            "--h5", str(stablewm / spec["dataset"]),
            "--output", str(embeddings), "--batch-size", "512",
            "--num-prefetch", "8", "--device", "cuda:0",
        ]
        subprocess.run(command, env=env, check=True)
    elif phase == "train":
        command = [
            python, str(upstream / "train_idm.py"), "train",
            "--embeddings", str(embeddings), "--output", str(idm),
            "--embed-dim", "192", "--action-dim", str(spec["action_dim"]),
            "--frameskip", "1", "--max-goal-horizon", "50",
            "--epochs", "200", "--lr", "0.0003", "--batch-size", "8192",
            "--seed", "42", "--device", "cuda:0",
        ]
        subprocess.run(command, env=env, check=True)
    elif phase == "eval":
        command = [
            python, str(REPO / "leworldmodel/additional_files/gcidm_matched_eval.py"),
            "--dataset", spec["official_dataset"], "--checkpoint", str(bundle),
            "--idm", str(idm), "--num-eval", "50", "--goal-offset", "25",
            "--eval-budget", "50", "--seed", "42", "--device", "cuda:0",
        ]
        log = task_root / "eval.txt"
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
            return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
        result = parse_eval_summary(log.read_text(encoding="utf-8", errors="replace"))
        result.update(
            {
                "task": task,
                "method": "Bloop frozen encoder + GC-IDM",
                "bloop_cem_reference_success_rate": spec["cem_reference"],
                "evaluation_log": str(log),
            }
        )
        (task_root / "summary.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    else:
        raise ValueError(phase)
    print(f"GCIDM_BLOOP_{phase.upper()}_COMPLETE task={task}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--tasks", default="tworoom,reacher")
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--stablewm-home", type=Path, default=DEFAULT_STABLEWM)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--partition", default="interactive")
    parser.add_argument("--nodelist", default="SPGL-1-1")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--worker", choices=("extract", "train", "eval"))
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--task", choices=tuple(TASKS))
    args = parser.parse_args()

    if args.worker:
        if not args.campaign or not args.task:
            parser.error("worker mode requires --campaign and --task")
        run_worker(args.campaign, args.task, args.worker)
        return

    try:
        tasks = parse_tasks(args.tasks)
    except ValueError as error:
        parser.error(str(error))
    plan = {
        "method": "frozen Bloop encoder + official GC-IDM",
        "tasks": tasks,
        "protocol": {
            "frameskip": 1,
            "max_goal_horizon": 50,
            "goal_offset": 25,
            "eval_budget": 50,
            "num_eval": 50,
            "seed": 42,
            "epochs": 200,
            "batch_size": 8192,
            "learning_rate": 3e-4,
            "evaluation_group": "exact GoalEvalCallback fixed-group sampler",
        },
        "task_specs": {task: TASKS[task] for task in tasks},
    }
    print(json.dumps(plan, indent=2))
    if not args.submit:
        print("PLAN ONLY: pass --submit --confirm-run to launch.")
        return
    if not args.confirm_run:
        parser.error("submission requires --confirm-run")
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    official = upstream_record(args.upstream)
    missing_datasets = [
        str(args.stablewm_home / TASKS[task]["dataset"])
        for task in tasks
        if not (args.stablewm_home / TASKS[task]["dataset"]).is_file()
    ]
    if missing_datasets:
        raise FileNotFoundError(f"Missing datasets: {missing_datasets}")
    missing_checkpoints = []
    for task in tasks:
        source = args.stablewm_home / "checkpoints" / TASKS[task]["run_name"]
        for filename in ("config.json", TASKS[task]["checkpoint"]):
            path = source / filename
            if not path.is_file():
                missing_checkpoints.append(str(path))
    if missing_checkpoints:
        raise FileNotFoundError(
            f"Missing completed Bloop checkpoints: {missing_checkpoints}"
        )
    if "reacher" in tasks:
        historical_required = (
            HISTORICAL_PYTHON,
            HISTORICAL_ROOT / "jepa.py",
            HISTORICAL_SWM / "stable_worldmodel/wm/utils.py",
        )
        historical_missing = [
            str(path) for path in historical_required if not path.is_file()
        ]
        if historical_missing:
            raise FileNotFoundError(
                f"Missing historical Reacher runtime: {historical_missing}"
            )

    campaign = REPO / "leworldmodel/results" / (
        "gcidm-bloop-multitask-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    campaign.mkdir(parents=True, exist_ok=False)
    conversions = {}
    for task in tasks:
        _, conversions[task] = prepare_task_bundle(
            args.stablewm_home, campaign, task
        )
    manifest = {
        **plan,
        "python": str(args.python),
        "stablewm_home": str(args.stablewm_home),
        "historical_root": str(HISTORICAL_ROOT),
        "historical_python": str(HISTORICAL_PYTHON),
        "historical_swm_root": str(HISTORICAL_SWM),
        "official_upstream": official,
        "conversions": conversions,
        "jobs": {},
    }
    if "reacher" in tasks:
        preflight = """
import sys
import stable_worldmodel as swm
import stable_worldmodel.wm.utils
from idm.dataset import load_lewm_model
assert swm.__file__.startswith(sys.argv[1])
model = load_lewm_model(sys.argv[2], "cpu")
jepa = model.model if hasattr(model, "model") else model
assert hasattr(jepa, "encoder") and hasattr(jepa, "projector")
print("GCIDM_HISTORICAL_REACHER_PREFLIGHT_PASS")
"""
        subprocess.run(
            [
                str(HISTORICAL_PYTHON),
                "-c",
                preflight,
                str(HISTORICAL_SWM),
                str(campaign / "reacher/inference"),
            ],
            env=runtime_environment(manifest, "reacher"),
            check=True,
        )

    def save() -> None:
        (campaign / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    def submit_array(phase: str, dependency: str | None = None) -> str:
        script = "\n".join(
            (
                "set -euo pipefail",
                f"tasks=({' '.join(tasks)})",
                'task="${tasks[$SLURM_ARRAY_TASK_ID]}"',
                shlex.join(
                    [
                        str(args.python), str(Path(__file__).resolve()),
                        "--worker", phase, "--campaign", str(campaign),
                    ]
                ) + ' --task "$task"',
            )
        )
        batch = [
            "sbatch", "--parsable", "--partition", args.partition,
            f"--array=0-{len(tasks) - 1}%{args.max_concurrent}",
            "--gres=gpu:1", "--cpus-per-task=12",
            "--mem=64G" if phase == "extract" else "--mem=48G",
            "--time=04:00:00" if phase == "extract" else (
                "--time=02:00:00" if phase == "train" else "--time=01:00:00"
            ),
            f"--job-name=oe-gcidm-{phase}", "--nodelist", args.nodelist,
            "--chdir", str(REPO),
            "--output", str(campaign / f"{phase}-%A_%a.out"),
        ]
        if dependency:
            batch += ["--dependency=afterok:" + dependency, "--kill-on-invalid-dep=yes"]
        batch += ["--wrap", script]
        job = subprocess.check_output(batch, text=True).strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(f"Unexpected sbatch response: {job!r}")
        manifest["jobs"][phase] = job
        save()
        return job

    save()
    extract = submit_array("extract")
    training = submit_array("train", extract)
    evaluation = submit_array("eval", training)
    print(f"GCIDM_BLOOP_EXTRACT_JOB={extract}")
    print(f"GCIDM_BLOOP_TRAIN_JOB={training}")
    print(f"GCIDM_BLOOP_EVAL_JOB={evaluation}")
    print(f"CAMPAIGN={campaign}")


if __name__ == "__main__":
    main()
