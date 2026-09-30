"""Submit the matched tagged-PushT temporal-horizon ablation.

The four arms change only the maximum control horizon and the way Reach is
aggregated across horizons. Data, Lctrl weights, masking, gradient routing,
training budget and online evaluation protocol are inherited from Full Ours.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

try:
    from .clean_inverse_reach_campaign import environment, source_digest
except ImportError:
    from clean_inverse_reach_campaign import environment, source_digest


REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARMS = {
    "h1_pair": {
        "max_horizon": 1,
        "reach_aggregation": "pair",
        "description": "one-step control evidence",
    },
    "h12_pair": {
        "max_horizon": 2,
        "reach_aggregation": "pair",
        "description": "one- and two-step control evidence",
    },
    "h123_pair": {
        "max_horizon": 3,
        "reach_aggregation": "pair",
        "description": "historical pair-uniform Full Ours",
    },
    "h123_balanced": {
        "max_horizon": 3,
        "reach_aggregation": "horizon",
        "description": "three horizons with equal Reach weight",
    },
}


def campaign_environment() -> dict[str, str]:
    """Use group storage for all caches and per-job temporary files."""
    env = environment()
    for name in ("SPT_CACHE_DIR", "XDG_CACHE_HOME", "TMPDIR"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    return env


def _replace_override(overrides: list[str], key: str, value: object) -> None:
    prefix = f"+loss.control.{key}="
    replacement = prefix + str(value).lower()
    matches = [index for index, item in enumerate(overrides) if item.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {key!r} override, found {len(matches)}")
    overrides[matches[0]] = replacement


def arm_spec(arm: str, condition: str = "tagged") -> tuple[dict, dict]:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    if condition == "tagged":
        try:
            from .run_control import load_control_configs
        except ImportError:
            from run_control import load_control_configs

        config = load_control_configs()
        spec = copy.deepcopy(config["arms"]["control_aligned_pred1"])
    elif condition == "clean":
        try:
            from .run_pusht_clean_ours_matched import (
                load_pusht_clean_ours_matched_configs,
            )
        except ImportError:
            from run_pusht_clean_ours_matched import (
                load_pusht_clean_ours_matched_configs,
            )

        config = load_pusht_clean_ours_matched_configs()
        spec = copy.deepcopy(next(iter(config["arms"].values())))
    else:
        raise ValueError(f"Unknown condition: {condition}")
    overrides = list(spec["overrides"])
    _replace_override(overrides, "max_horizon", ARMS[arm]["max_horizon"])
    _replace_override(
        overrides, "reach_aggregation", ARMS[arm]["reach_aggregation"]
    )
    spec["overrides"] = overrides
    return config, spec


def train(
    root: Path, arm: str, seed: int, smoke: bool, condition: str = "tagged"
) -> None:
    try:
        from . import run as base_runner
    except ImportError:
        import run as base_runner

    config, spec = arm_spec(arm, condition)
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
            {
                "arm": arm,
                "seed": seed,
                "checkpoint": str(expected),
                **ARMS[arm],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--partition", default="gpu_shared")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--condition", choices=("tagged", "clean"), default="tagged")
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--worker", choices=("smoke", "train", "combined"))
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--arm-index", type=int)
    args = parser.parse_args()
    runtime_env = campaign_environment()
    os.environ.update(runtime_env)

    if args.worker:
        if args.campaign is None or args.arm_index is None:
            parser.error("worker mode requires --campaign and --arm-index")
        arms = tuple(ARMS)
        if not 0 <= args.arm_index < len(arms):
            parser.error(f"arm-index must be in [0, {len(arms) - 1}]")
        manifest = json.loads((args.campaign / "manifest.json").read_text())
        if source_digest() != manifest["source_sha256"]:
            raise RuntimeError(
                "Source/config changed since submission; submit a fresh campaign"
            )
        arm = arms[args.arm_index]
        if args.worker == "combined":
            train(args.campaign, arm, args.seed, smoke=True, condition=args.condition)
            train(args.campaign, arm, args.seed, smoke=False, condition=args.condition)
        else:
            train(
                args.campaign,
                arm,
                args.seed,
                smoke=args.worker == "smoke",
                condition=args.condition,
            )
        print(f"HORIZON_{args.worker.upper()}_{arm.upper()}_COMPLETE", flush=True)
        return

    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.max_concurrent < 1:
        parser.error("max-concurrent must be positive")
    print(f"Matched {args.condition} PushT temporal-horizon ablation:")
    for index, (arm, spec) in enumerate(ARMS.items()):
        print(f"  [{index}] {arm}: {spec['description']}")
    if not args.submit:
        print("PLAN ONLY. Pass --submit on the cluster to launch jobs.")
        return

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(HERE / "tests/test_horizon_ablation_campaign.py"),
            str(HERE / "tests/test_control_objectives.py"),
        ],
        cwd=REPO,
        env=runtime_env,
        check=True,
    )
    root = REPO / "leworldmodel/results" / (
        f"horizon-{args.condition}-ablation-"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_sha256": source_digest(),
        "training_seed": args.seed,
        "condition": args.condition,
        "arms": ARMS,
        "array_order": list(ARMS),
        "comparison": (
            "Only max_horizon and Reach aggregation differ; data, Lctrl weights, "
            "masking, router, epochs and online evaluation are matched."
        ),
        "jobs": {},
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def submit(worker: str, dependency: str | None = None) -> str:
        command = (
            f"{shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).resolve()))} "
            f"--worker {worker} --condition {args.condition} "
            f"--campaign {shlex.quote(str(root))} "
            f"--arm-index $SLURM_ARRAY_TASK_ID --seed {args.seed}"
        )
        batch = [
            "sbatch",
            "--parsable",
            "--partition",
            args.partition,
            "--array",
            f"0-{len(ARMS) - 1}%{args.max_concurrent}",
            "--gres=gpu:1",
            "--cpus-per-task=12",
            "--mem=64G",
            "--time=00:30:00" if worker == "smoke" else "--time=20:30:00",
            f"--job-name=oe-horizon-{worker}",
            "--chdir",
            str(REPO),
            "--output",
            str(root / f"{worker}-%A_%a.out"),
        ]
        if dependency:
            batch += [
                "--dependency=afterok:" + dependency,
                "--kill-on-invalid-dep=yes",
            ]
        batch += ["--wrap", command]
        response = subprocess.check_output(batch, text=True).strip().split(";")[0]
        if not response.isdigit():
            raise RuntimeError(f"Unexpected sbatch response: {response!r}")
        return response

    if args.combined:
        combined_job = submit("combined")
        manifest["jobs"] = {"combined_array": combined_job}
        print(f"COMBINED_ARRAY_JOB={combined_job}")
    else:
        smoke_job = submit("smoke")
        train_job = submit("train", smoke_job)
        manifest["jobs"] = {"smoke_array": smoke_job, "train_array": train_job}
        print(f"SMOKE_ARRAY_JOB={smoke_job}")
        print(f"TRAIN_ARRAY_JOB={train_job}")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"CAMPAIGN={root}")


if __name__ == "__main__":
    main()
