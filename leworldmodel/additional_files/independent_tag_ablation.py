"""Submit the Tagged PushT independent-retagging pilots and paired evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

from counterfactual_planning_campaign import HERE, REPO, environment, source_digest


ARMS = ("both", "context_only", "goal_only", "balanced")
STEPS = 96
TRAIN_SEED = 0
EVAL_SEED = 73
CLIPS = 64
CANDIDATES = 32
TAG_MODES = ("both_tags", "context_tag", "goal_tag")
TRAINING_METRICS = (
    "counterfactual_margin_eligible_rate",
    "counterfactual_margin_active_rate",
    "counterfactual_ranking_loss",
    "counterfactual_action_accuracy",
)
EVAL_METRICS = tuple(
    f"{condition}/{metric}"
    for condition in (*TAG_MODES, "action_sequence")
    for metric in (
        "selected_candidate_changed",
        "pair_order_reversal",
        "cost_rmse_over_reference_std",
    )
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def checkpoint_spec(name: str, path: Path, checkpoint_root: Path) -> str:
    relative = path.resolve().relative_to(checkpoint_root.resolve())
    if len(relative.parts) != 2 or not path.is_file():
        raise ValueError(f"{name} must have a checkpoint at RUN/FILE.pt under {checkpoint_root}: {path}")
    return f"{name}={relative.as_posix()}"


def completed_run(campaign: Path) -> tuple[Path, Path]:
    completions = list(campaign.glob("*/complete.json"))
    if len(completions) != 1:
        raise ValueError(f"Expected one completed pilot in {campaign}, found {len(completions)}")
    completion = read_json(completions[0])
    if completion["seed"] != TRAIN_SEED:
        raise ValueError(f"Expected seed {TRAIN_SEED} in {completions[0]}")
    return Path(completion["checkpoint"]), completions[0].parent


def check_historical_arm(campaign: Path, base: Path, checkpoint_root: Path) -> None:
    manifest = read_json(campaign / "manifest.json")
    required = {
        "ranking_metric": "margin_guard",
        "relative_cost_only": True,
        "pilot_steps": STEPS,
        "training_seeds": [TRAIN_SEED],
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise ValueError(f"Historical pilot {key} must be {value!r}: {campaign}")
    if manifest.get("tag_intervention", "both") != "both":
        raise ValueError(f"Historical pilot must retag both endpoints: {campaign}")
    if Path(manifest["init_checkpoint"]).resolve() != base:
        raise ValueError(f"Historical pilot used a different initialization: {campaign}")
    checkpoint, _ = completed_run(campaign)
    checkpoint_spec("both", checkpoint, checkpoint_root)


def submit(both: Path, base: Path, partition: str, env: dict[str, str]) -> None:
    root = REPO / "leworldmodel/results" / f"independent-tag-{datetime.now():%Y%m%d-%H%M%S-%f}"
    root.mkdir(parents=True, exist_ok=False)
    manifest_path = root / "manifest.json"
    manifest = {
        "source_sha256": source_digest(),
        "init_checkpoint": str(base),
        "campaigns": {"both": str(both)},
        "training_jobs": {},
    }
    write_json(manifest_path, manifest)
    print(f"Run manifest: {manifest_path}", flush=True)

    for mode in ARMS[1:]:
        command = [
            sys.executable, str(HERE / "counterfactual_planning_campaign.py"),
            "--submit", "--pilot-steps", str(STEPS), "--init-checkpoint", str(base),
            "--ranking-metric", "margin_guard", "--tag-intervention", mode,
            "--relative-cost-only", "--partition", partition, "--seeds", str(TRAIN_SEED),
        ]
        receipt = subprocess.run(command, cwd=REPO, env=env, text=True, stdout=subprocess.PIPE)
        print(receipt.stdout, end="", flush=True)
        receipt.check_returncode()
        receipt_manifest, = (
            line.partition("=")[2] for line in receipt.stdout.splitlines()
            if line.startswith("JOB_MANIFEST=")
        )
        pilot_manifest = Path(receipt_manifest)
        pilot_job, = read_json(pilot_manifest)["jobs"]
        manifest["campaigns"][mode] = str(pilot_manifest.parent)
        manifest["training_jobs"][mode] = pilot_job["job_id"]
        write_json(manifest_path, manifest)

    dependency = "afterok:" + ":".join(manifest["training_jobs"].values())
    worker = [sys.executable, "-u", str(Path(__file__).resolve()), "--evaluate", str(manifest_path)]
    response = subprocess.check_output([
        "sbatch", "--parsable", "--partition", partition, "--job-name=oe-tag-sides-eval",
        "--gres=gpu:1", "--cpus-per-task=4", "--mem=24G", "--time=00:30:00",
        "--dependency", dependency, "--kill-on-invalid-dep=yes",
        "--chdir", str(REPO), "--output", str(root / "job-%j.out"), "--wrap", shlex.join(worker),
    ], text=True, env=env, cwd=REPO)
    job_id = response.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected sbatch response: {response!r}")
    manifest["evaluation_job"] = job_id
    write_json(manifest_path, manifest)
    jobs = ",".join([*manifest["training_jobs"].values(), job_id])
    print(f"Evaluation job: {job_id}\n"
          f"Monitor: sacct -j {jobs} --format=JobID,State,Elapsed\n"
          f"Report after evaluation: {root / 'report.json'}", flush=True)


def training_stats(name, checkpoint, run_dir, original):
    import math
    import torch

    trained = torch.load(checkpoint, map_location="cpu", weights_only=True)
    sum_squares, elements, max_change = 0.0, 0, 0.0
    for key, source in original.items():
        if not key.startswith(("encoder.", "projector.")) or not source.is_floating_point():
            continue
        if key.endswith((".running_mean", ".running_var")):
            continue
        delta = trained[key].float() - source.float()
        sum_squares += delta.square().sum().item()
        elements += delta.numel()
        max_change = max(max_change, delta.abs().max().item())
    if not elements or not max_change or not math.isfinite(sum_squares):
        raise RuntimeError(f"{name}: representation parameter change could not be verified")

    log = run_dir / "metrics.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.is_file() else []
    logged = {key: [] for key in TRAINING_METRICS}
    steps = []
    for row in rows:
        for key, values in logged.items():
            for label in (f"fit/{key}_step", f"fit/{key}", f"train/{key}_step", f"train/{key}"):
                if label in row:
                    value = float(row[label])
                    if math.isfinite(value):
                        values.append(value)
                        if row.get("step") is not None:
                            steps.append(row["step"])
                    break
    return {
        "checkpoint": str(checkpoint),
        "representation_parameter_rms_change": math.sqrt(sum_squares / elements),
        "representation_parameter_max_change": max_change,
        "logged_values": {key: len(values) for key, values in logged.items()},
        "last_logged_step": max(steps, default=None),
        "logged_means": {key: sum(values) / len(values) if values else None for key, values in logged.items()},
        "counterfactual_log_keys": sorted({key for row in rows for key in row if "counterfactual" in key}),
    }


def summarize_evaluation(raw: dict) -> tuple[dict, dict]:
    import numpy as np

    runs = raw["runs"]
    if len(raw["protocol"]["indices"]) != CLIPS or set(runs) != {"base", *ARMS}:
        raise ValueError("Diagnostic did not return the expected clips and checkpoints")
    flips = {
        name: np.maximum.reduce([
            np.asarray(run["per_clip"][f"{mode}/selected_candidate_changed"]) for mode in TAG_MODES
        ])
        for name, run in runs.items()
    }
    evaluation = {
        name: {
            "checkpoint_sha256": run["sha256"],
            "metrics": {key: run["metrics"][key] for key in EVAL_METRICS},
            "clips_stable_under_all_tag_modes": int((1 - flips[name]).sum()),
        }
        for name, run in runs.items()
    }
    samples = np.random.default_rng(EVAL_SEED).integers(0, CLIPS, size=(2000, CLIPS))
    paired = {}
    for name in ARMS[1:]:
        differences = {
            key: np.asarray(runs[name]["per_clip"][key]) - np.asarray(runs["both"]["per_clip"][key])
            for key in EVAL_METRICS
        }
        differences["any_tag_flip"] = flips[name] - flips["both"]
        paired[name] = {
            key: {
                "difference": float(delta.mean()),
                "bootstrap_clip_ci95": np.quantile(delta[samples].mean(1), [0.025, 0.975]).tolist(),
            }
            for key, delta in differences.items()
        }
    return evaluation, paired


def evaluate(manifest_path: Path, env: dict[str, str]) -> None:
    import torch

    manifest = read_json(manifest_path)
    if source_digest() != manifest["source_sha256"]:
        raise RuntimeError("Source/config changed since submission")
    root = manifest_path.parent
    base = Path(manifest["init_checkpoint"])
    checkpoint_root = Path(env["STABLEWM_HOME"]) / "checkpoints"
    original = torch.load(base, map_location="cpu", weights_only=True)
    specs = [checkpoint_spec("base", base, checkpoint_root)]
    training = {}
    for name in ARMS:
        checkpoint, run_dir = completed_run(Path(manifest["campaigns"][name]))
        specs.append(checkpoint_spec(name, checkpoint, checkpoint_root))
        training[name] = training_stats(name, checkpoint, run_dir, original)
    del original
    write_json(root / "training.json", training)
    print("GRIDFIRE_TAG_SIDES_TRAINING_READY", flush=True)

    out = root / "tag-intervention.json"
    subprocess.run([
        sys.executable, str(HERE / "diagnose_tag_intervention.py"),
        "--out", str(out), "--num-clips", str(CLIPS), "--candidates", str(CANDIDATES),
        "--seed", str(EVAL_SEED), *[token for spec in specs for token in ("--checkpoint", spec)],
    ], check=True, env=env, cwd=REPO)
    raw = read_json(out)
    evaluation, paired = summarize_evaluation(raw)
    write_json(root / "report.json", {
        "steps": STEPS,
        "seed": TRAIN_SEED,
        "protocol": {
            "clips": CLIPS, "candidates": CANDIDATES, "seed": EVAL_SEED,
            **{key: raw["protocol"][key] for key in ("diagnostic", "sampling", "bootstrap")},
        },
        "training": training,
        "evaluation": evaluation,
        "paired_vs_both": paired,
    })
    print(f"GRIDFIRE_TAG_SIDES_EVALUATION_COMPLETE {root / 'report.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--both-campaign", type=Path, help="Completed both-tags margin_guard pilot")
    operation.add_argument("--evaluate", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--partition", default="gpu_shared")
    args = parser.parse_args()
    env = environment()
    env.update(HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    if args.evaluate:
        evaluate(args.evaluate, env)
        return
    checkpoint_root = Path(env["STABLEWM_HOME"]) / "checkpoints"
    base = (checkpoint_root / "colored_square_episode_seed0/weights_epoch_10.pt").resolve()
    both = args.both_campaign.resolve()
    try:
        checkpoint_spec("base", base, checkpoint_root)
        check_historical_arm(both, base, checkpoint_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    submit(both, base, args.partition, env)


if __name__ == "__main__":
    main()
