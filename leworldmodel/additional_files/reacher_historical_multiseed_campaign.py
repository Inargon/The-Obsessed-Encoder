#!/usr/bin/env python3
"""Paired multi-seed clean Reacher evaluation under the pinned historical stack."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

if __package__:
    from .reacher_historical_eval_campaign import (
        DEFAULT_HISTORICAL_PYTHON,
        DEFAULT_HISTORICAL_ROOT,
        DEFAULT_HISTORICAL_SWM_ROOT,
        DEFAULT_STABLEWM_HOME,
        HISTORICAL_EVAL_PRELUDE,
        historical_cache_root,
        historical_env,
        parse_success_rate,
        preflight_historical_dataset,
        preflight_historical_model_loader,
        quote,
    )
else:
    from reacher_historical_eval_campaign import (
        DEFAULT_HISTORICAL_PYTHON,
        DEFAULT_HISTORICAL_ROOT,
        DEFAULT_HISTORICAL_SWM_ROOT,
        DEFAULT_STABLEWM_HOME,
        HISTORICAL_EVAL_PRELUDE,
        historical_cache_root,
        historical_env,
        parse_success_rate,
        preflight_historical_dataset,
        preflight_historical_model_loader,
        quote,
    )


REPO = Path(__file__).resolve().parents[2]
METHODS = ("full", "predictor_only_cycle")
DEFAULT_SEEDS = (0, 1, 2, 3, 42)
SOURCE_CAMPAIGN = "reacher-historical-matrix-20261002-032925"


def cells(seeds: tuple[int, ...]) -> list[tuple[str, int]]:
    return [(method, seed) for method in METHODS for seed in seeds]


def source_checkpoint(stablewm_home: Path, source_campaign: str, method: str) -> Path:
    return (
        stablewm_home
        / "checkpoints"
        / f"{source_campaign}_{method}"
        / "weights_epoch_10_historical_v2.pt"
    )


def evaluate(args: argparse.Namespace) -> None:
    if args.campaign is None or args.method is None or args.seed is None:
        raise ValueError("worker requires campaign, method, and seed")
    checkpoint = source_checkpoint(
        args.stablewm_home, args.source_campaign, args.method
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    output_name = (
        f"{args.campaign.name}-{args.method}-seed{args.seed}-historical-reacher.txt"
    )
    command = [
        str(args.historical_python),
        "-c",
        HISTORICAL_EVAL_PRELUDE,
        str(args.historical_root / "eval.py"),
        "--config-name=reacher",
        f"policy={checkpoint}",
        f"+cache_dir={historical_cache_root(args.stablewm_home)}",
        "eval.dataset_name=dmc/reacher_random",
        "dataset.keys_to_cache=[action]",
        f"seed={args.seed}",
        "solver.n_steps=30",
        f"output.filename={output_name}",
    ]
    env = historical_env(args.historical_swm_root)
    env["STABLEWM_HOME"] = str(args.stablewm_home)
    env["LOCAL_DATASET_DIR"] = str(historical_cache_root(args.stablewm_home))
    env["MUJOCO_GL"] = "egl"
    env.pop("PYOPENGL_PLATFORM", None)
    subprocess.run(command, cwd=args.historical_root, env=env, check=True)

    output = checkpoint.parent / output_name
    if not output.is_file():
        raise FileNotFoundError(output)
    rate = parse_success_rate(output.read_text())
    record = {
        "method": args.method,
        "evaluation_seed": args.seed,
        "num_eval": 50,
        "success_rate": rate,
        "successes": round(rate * 50),
        "protocol": "clean_reacher_historical_multiseed",
        "checkpoint": str(checkpoint),
    }
    result = args.campaign / f"{args.method}-seed{args.seed}.json"
    result.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    print(
        f"REACHER_HISTORICAL_MULTISEED_COMPLETE method={args.method} "
        f"seed={args.seed}",
        flush=True,
    )


def summarize(campaign: Path, seeds: tuple[int, ...]) -> None:
    summary: dict[str, object] = {
        "protocol": "clean_reacher_historical_multiseed",
        "evaluation_seeds": list(seeds),
        "episodes_per_seed": 50,
        "methods": {},
    }
    for method in METHODS:
        rows = [
            json.loads((campaign / f"{method}-seed{seed}.json").read_text())
            for seed in seeds
        ]
        rates = [float(row["success_rate"]) for row in rows]
        successes = sum(int(row["successes"]) for row in rows)
        total = 50 * len(rows)
        mean = statistics.mean(rates)
        std = statistics.stdev(rates) if len(rates) > 1 else 0.0
        se = math.sqrt(mean * (1.0 - mean) / total)
        summary["methods"][method] = {
            "per_seed_success_rate": {
                str(seed): rate for seed, rate in zip(seeds, rates)
            },
            "mean_success_rate": mean,
            "sample_std_across_seeds": std,
            "pooled_success_rate": successes / total,
            "pooled_successes": successes,
            "pooled_episodes": total,
            "normal_approx_ci95": [
                max(0.0, mean - 1.96 * se),
                min(1.0, mean + 1.96 * se),
            ],
        }
    path = campaign / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"REACHER_HISTORICAL_MULTISEED_SUMMARY_COMPLETE {path}", flush=True)


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(","))
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be unique non-negative integers")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--partition", default="interactive")
    parser.add_argument("--nodelist", default="SPGL-1-1")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--source-campaign", default=SOURCE_CAMPAIGN)
    parser.add_argument("--stablewm-home", type=Path, default=DEFAULT_STABLEWM_HOME)
    parser.add_argument("--historical-root", type=Path, default=DEFAULT_HISTORICAL_ROOT)
    parser.add_argument(
        "--historical-python", type=Path, default=DEFAULT_HISTORICAL_PYTHON
    )
    parser.add_argument(
        "--historical-swm-root", type=Path, default=DEFAULT_HISTORICAL_SWM_ROOT
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if args.worker:
        evaluate(args)
        return
    if args.summarize:
        if args.campaign is None:
            parser.error("--summarize requires --campaign")
        summarize(args.campaign, args.seeds)
        return
    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be positive")

    matrix = cells(args.seeds)
    missing = [
        str(source_checkpoint(args.stablewm_home, args.source_campaign, method))
        for method in METHODS
        if not source_checkpoint(
            args.stablewm_home, args.source_campaign, method
        ).is_file()
    ]
    plan = {
        "protocol": "clean_reacher_historical_multiseed",
        "source_campaign": args.source_campaign,
        "methods": list(METHODS),
        "evaluation_seeds": list(args.seeds),
        "episodes_per_cell": 50,
        "total_cells": len(matrix),
        "total_episodes": 50 * len(matrix),
        "reporting": "mean, sample std, pooled rate, and CI95; never best-seed only",
        "missing": missing,
        "jobs": {},
    }
    print(json.dumps(plan, indent=2))
    if not args.submit:
        print("PLAN ONLY: pass --submit --confirm-run to launch.")
        return
    if not args.confirm_run:
        parser.error("refusing to submit without --confirm-run")
    if missing:
        parser.error("missing historical checkpoints: " + ", ".join(missing))

    rows = preflight_historical_dataset(
        args.historical_python,
        args.historical_root,
        historical_cache_root(args.stablewm_home),
        args.historical_swm_root,
    )
    preflight_historical_model_loader(
        args.historical_python, args.historical_swm_root
    )
    print(f"HISTORICAL_REACHER_DATASET_PREFLIGHT_PASS rows={rows}")

    campaign = REPO / "leworldmodel/results" / (
        "reacher-historical-multiseed-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    campaign.mkdir(parents=True, exist_ok=False)
    (campaign / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")

    method_values = " ".join(f"'{method}'" for method in METHODS)
    seed_values = " ".join(str(seed) for seed in args.seeds)
    worker = quote(
        [
            sys.executable,
            Path(__file__).resolve(),
            "--worker",
            "--campaign",
            campaign,
            "--method",
            "$method",
            "--seed",
            "$eval_seed",
            "--source-campaign",
            args.source_campaign,
            "--stablewm-home",
            args.stablewm_home,
            "--historical-root",
            args.historical_root,
            "--historical-python",
            args.historical_python,
            "--historical-swm-root",
            args.historical_swm_root,
        ]
    ).replace("'$method'", '"$method"').replace("'$eval_seed'", '"$eval_seed"')
    shell = "\n".join(
        (
            "set -euo pipefail",
            f"methods=({method_values})",
            f"seeds=({seed_values})",
            'method="${methods[$SLURM_ARRAY_TASK_ID]}"',
            'for eval_seed in "${seeds[@]}"; do',
            f"  {worker}",
            "done",
        )
    )
    batch = [
        "sbatch",
        "--parsable",
        "--partition",
        args.partition,
        "--nodelist",
        args.nodelist,
        f"--array=0-{len(METHODS) - 1}%{min(args.max_concurrent, len(METHODS))}",
        "--gres=gpu:1",
        "--cpus-per-task=8",
        "--mem=48G",
        "--time=00:45:00",
        "--job-name=oe-reacher-multiseed",
        "--chdir",
        str(REPO),
        "--output",
        str(campaign / "eval-%A_%a.out"),
        "--wrap",
        shell,
    ]
    eval_job = subprocess.check_output(batch, text=True).strip().split(";")[0]

    summary_command = quote(
        [
            sys.executable,
            Path(__file__).resolve(),
            "--summarize",
            "--campaign",
            campaign,
            "--seeds",
            ",".join(str(seed) for seed in args.seeds),
        ]
    )
    summary_batch = [
        "sbatch",
        "--parsable",
        "--partition",
        args.partition,
        "--nodelist",
        args.nodelist,
        f"--dependency=afterok:{eval_job}",
        "--kill-on-invalid-dep=yes",
        "--cpus-per-task=1",
        "--mem=2G",
        "--time=00:05:00",
        "--job-name=oe-reacher-summary",
        "--chdir",
        str(REPO),
        "--output",
        str(campaign / "summary-%j.out"),
        "--wrap",
        summary_command,
    ]
    summary_job = (
        subprocess.check_output(summary_batch, text=True).strip().split(";")[0]
    )
    plan["jobs"] = {"evaluation_array": eval_job, "summary": summary_job}
    (campaign / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"REACHER_HISTORICAL_MULTISEED_JOB={eval_job}")
    print(f"REACHER_HISTORICAL_MULTISEED_SUMMARY_JOB={summary_job}")
    print(f"CAMPAIGN={campaign}")


if __name__ == "__main__":
    main()
