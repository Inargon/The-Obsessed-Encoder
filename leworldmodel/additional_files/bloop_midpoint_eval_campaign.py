#!/usr/bin/env python3
"""Evaluate intermediate Bloop checkpoints without disturbing training.

Tagged PushT uses the matched current evaluator. Clean Reacher is converted
and evaluated through the pinned historical stack recovered for the paper.
These midpoint numbers are diagnostic only; epoch-10 results remain primary.
"""

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

try:
    from .reacher_historical_eval_campaign import (
        DEFAULT_HISTORICAL_PYTHON,
        DEFAULT_HISTORICAL_ROOT,
        DEFAULT_HISTORICAL_SWM_ROOT,
        DEFAULT_STABLEWM_HOME,
        HISTORICAL_EVAL_PRELUDE,
        historical_env,
        parse_success_rate,
        preflight_historical_dataset,
        preflight_historical_model_loader,
    )
except ImportError:
    from reacher_historical_eval_campaign import (
        DEFAULT_HISTORICAL_PYTHON,
        DEFAULT_HISTORICAL_ROOT,
        DEFAULT_HISTORICAL_SWM_ROOT,
        DEFAULT_STABLEWM_HOME,
        HISTORICAL_EVAL_PRELUDE,
        historical_env,
        parse_success_rate,
        preflight_historical_dataset,
        preflight_historical_model_loader,
    )


REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def run_names(source_campaign: Path, training_seed: int) -> dict[str, str]:
    prefix = source_campaign.name
    return {
        "tagged_pusht": (
            f"{prefix}_tagged_pusht_bloop_cycle_full_seed{training_seed}"
        ),
        "clean_reacher": (
            f"{prefix}_clean_reacher_bloop_cycle_full_seed{training_seed}"
        ),
    }


def modern_environment(stablewm_home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + str(REPO / "leworldmodel")
    env["STABLEWM_HOME"] = str(stablewm_home)
    env["LOCAL_DATASET_DIR"] = str(stablewm_home)
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    env["WANDB_MODE"] = "disabled"
    return env


def worker(args: argparse.Namespace) -> None:
    names = run_names(args.source_campaign, args.training_seed)
    checkpoint_name = f"weights_epoch_{args.epoch}.pt"
    checkpoint_root = args.stablewm_home / "checkpoints"

    tagged_output = args.output_dir / "tagged-pusht.json"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "evaluate_pusht_checkpoint.py"),
            "--run-name",
            names["tagged_pusht"],
            "--checkpoint",
            checkpoint_name,
            "--dataset",
            "pusht_expert_train.h5",
            "--num-eval",
            str(args.num_eval),
            "--seed",
            str(args.eval_seed),
            "--tagged",
            "--output",
            str(tagged_output),
        ],
        cwd=REPO / "leworldmodel",
        env=modern_environment(args.stablewm_home),
        check=True,
    )

    reacher_source_dir = checkpoint_root / names["clean_reacher"]
    bundle = args.output_dir / "reacher-historical-bundle"
    bundle.mkdir()
    shutil.copy2(reacher_source_dir / "config.json", bundle / "config.json")
    converted = bundle / f"weights_epoch_{args.epoch}_historical_v2.pt"
    provenance = args.output_dir / "reacher-conversion.json"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "convert_reacher_historical_checkpoint.py"),
            "--source",
            str(reacher_source_dir / checkpoint_name),
            "--destination",
            str(converted),
            "--provenance",
            str(provenance),
        ],
        check=True,
    )

    output_name = "reacher-historical.txt"
    historical_command = [
        str(args.historical_python),
        "-c",
        HISTORICAL_EVAL_PRELUDE,
        str(args.historical_root / "eval.py"),
        "--config-name=reacher",
        f"policy={converted}",
        f"+cache_dir={args.stablewm_home}",
        "eval.dataset_name=dmc/reacher_random",
        "dataset.keys_to_cache=[action]",
        f"seed={args.eval_seed}",
        "solver.n_steps=30",
        f"output.filename={output_name}",
    ]
    env = historical_env(args.historical_swm_root)
    env["STABLEWM_HOME"] = str(args.stablewm_home)
    env["LOCAL_DATASET_DIR"] = str(args.stablewm_home)
    env["MUJOCO_GL"] = "egl"
    env.pop("PYOPENGL_PLATFORM", None)
    subprocess.run(
        historical_command,
        cwd=args.historical_root,
        env=env,
        check=True,
    )
    reacher_text = (bundle / output_name).read_text()
    reacher_rate = parse_success_rate(reacher_text)
    tagged = json.loads(tagged_output.read_text())
    summary = {
        "status": "diagnostic_midpoint_only",
        "source_campaign": str(args.source_campaign),
        "checkpoint": checkpoint_name,
        "training_seed": args.training_seed,
        "evaluation_seed": args.eval_seed,
        "episodes_per_task": args.num_eval,
        "tagged_pusht_success_rate": tagged["success_rate"],
        "clean_reacher_historical_success_rate": reacher_rate,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print("BLOOP_MIDPOINT_EVAL_COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--partition", default="interactive")
    parser.add_argument("--nodelist", default="SPGL-1-1")
    parser.add_argument("--stablewm-home", type=Path, default=DEFAULT_STABLEWM_HOME)
    parser.add_argument("--historical-root", type=Path, default=DEFAULT_HISTORICAL_ROOT)
    parser.add_argument(
        "--historical-python", type=Path, default=DEFAULT_HISTORICAL_PYTHON
    )
    parser.add_argument(
        "--historical-swm-root", type=Path, default=DEFAULT_HISTORICAL_SWM_ROOT
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    names = run_names(args.source_campaign, args.training_seed)
    checkpoint_name = f"weights_epoch_{args.epoch}.pt"
    required = {
        f"{task} checkpoint": (
            args.stablewm_home / "checkpoints" / name / checkpoint_name
        )
        for task, name in names.items()
    }
    required["Reacher config"] = (
        args.stablewm_home
        / "checkpoints"
        / names["clean_reacher"]
        / "config.json"
    )
    missing = [label for label, path in required.items() if not path.is_file()]
    if missing:
        parser.error("missing: " + ", ".join(missing))

    if args.worker:
        if args.output_dir is None:
            parser.error("worker mode requires --output-dir")
        worker(args)
        return

    plan = {
        "source_campaign": str(args.source_campaign),
        "checkpoint": checkpoint_name,
        "training_seed": args.training_seed,
        "evaluation_seed": args.eval_seed,
        "episodes_per_task": args.num_eval,
        "tagged_pusht_protocol": "matched current tagged evaluation",
        "clean_reacher_protocol": "clean_reacher_historical_seed42",
        "note": "diagnostic midpoint only; epoch 10 remains primary",
    }
    print(json.dumps(plan, indent=2))
    if not args.submit:
        print("PLAN ONLY: pass --submit to launch.")
        return

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(HERE / "tests/test_convert_reacher_historical_checkpoint.py"),
            str(HERE / "tests/test_bloop_midpoint_eval_campaign.py"),
        ],
        cwd=REPO / "leworldmodel",
        check=True,
    )

    rows = preflight_historical_dataset(
        args.historical_python,
        args.historical_root,
        args.stablewm_home,
        args.historical_swm_root,
    )
    preflight_historical_model_loader(
        args.historical_python, args.historical_swm_root
    )
    print(f"HISTORICAL_REACHER_DATASET_PREFLIGHT_PASS rows={rows}")

    output_dir = args.source_campaign / (
        f"midpoint-eval-epoch{args.epoch}-seed{args.eval_seed}-"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--source-campaign",
        str(args.source_campaign),
        "--output-dir",
        str(output_dir),
        "--epoch",
        str(args.epoch),
        "--training-seed",
        str(args.training_seed),
        "--eval-seed",
        str(args.eval_seed),
        "--num-eval",
        str(args.num_eval),
        "--stablewm-home",
        str(args.stablewm_home),
        "--historical-root",
        str(args.historical_root),
        "--historical-python",
        str(args.historical_python),
        "--historical-swm-root",
        str(args.historical_swm_root),
    ]
    batch = [
        "sbatch",
        "--parsable",
        "--partition",
        args.partition,
        "--nodelist",
        args.nodelist,
        "--gres=gpu:1",
        "--cpus-per-task=8",
        "--mem=48G",
        "--time=01:30:00",
        "--job-name=oe-bloop-mid-eval",
        "--chdir",
        str(REPO),
        "--output",
        str(output_dir / "job-%j.out"),
        "--wrap",
        shlex.join(command),
    ]
    job = subprocess.check_output(batch, text=True).strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError(f"unexpected sbatch response: {job!r}")
    print(f"BLOOP_MIDPOINT_EVAL_JOB={job}")
    print(f"OUTPUT_DIR={output_dir}")


if __name__ == "__main__":
    main()
