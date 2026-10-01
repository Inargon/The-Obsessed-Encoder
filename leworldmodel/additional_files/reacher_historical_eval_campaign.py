#!/usr/bin/env python3
"""Re-evaluate clean Reacher checkpoints under the pinned historical stack."""

from __future__ import annotations

import argparse
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONVERTER = HERE / "convert_reacher_historical_checkpoint.py"
DEFAULT_STABLEWM_HOME = Path("/grp01/ids_compcog/song/swm")
DEFAULT_HISTORICAL_ROOT = Path(
    "/grp01/ids_compcog/song/code/le-wm-repro-20260514"
)
DEFAULT_HISTORICAL_PYTHON = Path(
    "/grp01/ids_compcog/song/envs/lewm-repro-py310/bin/python"
)

ARMS = {
    "jepa": {
        "run_name": "reacher_clean_jepa_full_seed0",
        "expected_success_rate": 0.90,
    },
    "full": {
        "run_name": "reacher_clean_aligned_full_seed0",
        "expected_success_rate": 0.86,
    },
    "inverse_reach": {
        "run_name": (
            "clean-inverse-reach-20260923-084931_"
            "reacher_clean_inverse_reach_full_seed0"
        ),
        "expected_success_rate": 0.76,
    },
    "predictor_only_cycle": {
        "run_name": (
            "interface-cycle-20260930-225626_"
            "clean_reacher_interface_cycle_full_seed0"
        ),
        "expected_success_rate": None,
    },
}

HISTORICAL_OVERRIDES = (
    "+cache_dir={stablewm_home}",
    "eval.dataset_name=dmc/reacher_random",
    "dataset.keys_to_cache=[action]",
    "seed=42",
    "solver.n_steps=30",
)


def quote(parts: list[object]) -> str:
    return shlex.join(str(value) for value in parts)


def parse_success_rate(text: str) -> float:
    matches = re.findall(r"['\"]success_rate['\"]\s*:\s*([0-9.]+)", text)
    if not matches:
        raise ValueError("success_rate was not found in historical evaluator output")
    value = float(matches[-1])
    return value / 100.0 if value > 1.0 else value


def historical_dataset_dir(stablewm_home: Path) -> Path:
    return stablewm_home / "datasets"


def compare_reference(actual: Path, reference: Path) -> None:
    import torch

    left = torch.load(actual, map_location="cpu", weights_only=True)
    right = torch.load(reference, map_location="cpu", weights_only=True)
    if list(left) != list(right):
        raise RuntimeError(f"historical key mismatch against {reference}")
    mismatches = [key for key in left if not torch.equal(left[key], right[key])]
    if mismatches:
        raise RuntimeError(
            f"historical tensor mismatch against {reference}: {mismatches[:5]}"
        )


def worker(args) -> None:
    arm = ARMS[args.arm]
    checkpoint_root = args.stablewm_home / "checkpoints"
    source_dir = checkpoint_root / arm["run_name"]
    source = source_dir / "weights_epoch_10.pt"
    config = source_dir / "config.json"
    bundle = checkpoint_root / f"{args.campaign.name}_{args.arm}"
    destination = bundle / "weights_epoch_10_historical_v2.pt"
    provenance = args.campaign / f"{args.arm}-conversion.json"
    result = args.campaign / f"{args.arm}.json"
    for path in (source, config):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (bundle, provenance, result):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite: {path}")

    bundle.mkdir(parents=True)
    try:
        shutil.copy2(config, bundle / "config.json")
        subprocess.run(
            [
                sys.executable,
                str(CONVERTER),
                "--source",
                str(source),
                "--destination",
                str(destination),
                "--provenance",
                str(provenance),
            ],
            check=True,
        )
        reference = source_dir / "weights_epoch_10_historical_v2.pt"
        if arm["expected_success_rate"] is not None:
            if not reference.is_file():
                raise FileNotFoundError(reference)
            compare_reference(destination, reference)

        output_name = f"{args.arm}-historical-reacher.txt"
        command = [
            str(args.historical_python),
            str(args.historical_root / "eval.py"),
            "--config-name=reacher",
            f"policy={destination}",
            *(
                value.format(stablewm_home=args.stablewm_home)
                for value in HISTORICAL_OVERRIDES
            ),
            f"output.filename={output_name}",
        ]
        env = dict(os.environ)
        env["STABLEWM_HOME"] = str(args.stablewm_home)
        # stable-worldmodel 0.0.6 interprets LOCAL_DATASET_DIR as the dataset
        # directory itself (unlike STABLEWM_HOME, which is the cache root).
        # The recovered job 91315 read
        # ``<STABLEWM_HOME>/datasets/dmc/reacher_random.h5``.
        env["LOCAL_DATASET_DIR"] = str(historical_dataset_dir(args.stablewm_home))
        env["MUJOCO_GL"] = "egl"
        env.pop("PYOPENGL_PLATFORM", None)
        subprocess.run(command, cwd=args.historical_root, env=env, check=True)

        output = bundle / output_name
        if not output.is_file():
            raise FileNotFoundError(output)
        success_rate = parse_success_rate(output.read_text())
        expected = arm["expected_success_rate"]
        record = {
            "arm": args.arm,
            "source_run_name": arm["run_name"],
            "evaluation_bundle": bundle.name,
            "checkpoint": destination.name,
            "protocol": "clean_reacher_historical_seed42",
            "num_eval": 50,
            "evaluation_seed": 42,
            "success_rate": success_rate,
            "expected_regression_success_rate": expected,
            "regression_pass": expected is None or success_rate == expected,
        }
        result.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2), flush=True)
        if expected is not None and success_rate != expected:
            raise RuntimeError(
                f"historical regression failed for {args.arm}: "
                f"expected {expected:.2f}, observed {success_rate:.2f}"
            )
        print(f"REACHER_HISTORICAL_EVAL_COMPLETE arm={args.arm}", flush=True)
    except BaseException:
        # Preserve the bundle and provenance for post-mortem inspection.
        raise


def package_versions(python: Path) -> dict:
    code = """
import json, sys
from importlib.metadata import PackageNotFoundError, version
packages = ['stable-worldmodel', 'stable-pretraining', 'torch', 'numpy', 'mujoco']
values = {'python': sys.version}
for package in packages:
    try:
        values[package] = version(package)
    except PackageNotFoundError:
        values[package] = None
print(json.dumps(values))
"""
    return json.loads(subprocess.check_output([str(python), "-c", code], text=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--partition", default="interactive")
    parser.add_argument("--nodelist", default="SPGL-1-1")
    parser.add_argument("--stablewm-home", type=Path, default=DEFAULT_STABLEWM_HOME)
    parser.add_argument("--historical-root", type=Path, default=DEFAULT_HISTORICAL_ROOT)
    parser.add_argument(
        "--historical-python", type=Path, default=DEFAULT_HISTORICAL_PYTHON
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--arm", choices=tuple(ARMS))
    args = parser.parse_args()

    if args.worker:
        if args.campaign is None or args.arm is None:
            parser.error("worker mode requires --campaign and --arm")
        worker(args)
        return

    required = {
        "converter": CONVERTER,
        "historical eval.py": args.historical_root / "eval.py",
        "historical Python": args.historical_python,
        "Reacher dataset": args.stablewm_home / "datasets/dmc/reacher_random.h5",
    }
    for label, arm in ARMS.items():
        directory = args.stablewm_home / "checkpoints" / arm["run_name"]
        required[f"{label} checkpoint"] = directory / "weights_epoch_10.pt"
        required[f"{label} config"] = directory / "config.json"
        if arm["expected_success_rate"] is not None:
            required[f"{label} historical reference"] = (
                directory / "weights_epoch_10_historical_v2.pt"
            )
    missing = [label for label, path in required.items() if not path.is_file()]
    manifest = {
        "protocol": "clean_reacher_historical_seed42",
        "historical_root": str(args.historical_root),
        "historical_python": str(args.historical_python),
        "local_dataset_dir": str(historical_dataset_dir(args.stablewm_home)),
        "historical_versions": (
            package_versions(args.historical_python)
            if args.historical_python.is_file()
            else None
        ),
        "config_name": "reacher",
        "overrides": [
            value.format(stablewm_home=args.stablewm_home)
            for value in HISTORICAL_OVERRIDES
        ],
        "arms": ARMS,
        "missing": missing,
        "invalid_cross_protocol_result": {
            "arm": "predictor_only_cycle",
            "success_rate": 0.62,
            "reason": "evaluated with the modern Python 3.12 stack",
        },
        "jobs": {},
    }
    print(json.dumps(manifest, indent=2))
    if not args.submit:
        print("PLAN ONLY: pass --submit --confirm-run to launch.")
        return
    if not args.confirm_run:
        parser.error("refusing to submit without --confirm-run")
    if missing:
        parser.error("missing: " + ", ".join(missing))

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(HERE / "tests/test_convert_reacher_historical_checkpoint.py"),
            str(HERE / "tests/test_reacher_historical_eval_campaign.py"),
        ],
        cwd=REPO / "leworldmodel",
        check=True,
    )
    root = REPO / "leworldmodel/results" / (
        "reacher-historical-matrix-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=False)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    command = quote(
        [
            sys.executable,
            Path(__file__).resolve(),
            "--worker",
            "--campaign",
            root,
            "--arm",
            "$arm",
            "--stablewm-home",
            args.stablewm_home,
            "--historical-root",
            args.historical_root,
            "--historical-python",
            args.historical_python,
        ]
    ).replace("'$arm'", '"$arm"')
    shell = "\n".join(
        (
            "set -euo pipefail",
            f"arms=({' '.join(ARMS)})",
            'arm="${arms[$SLURM_ARRAY_TASK_ID]}"',
            command,
        )
    )
    batch = [
        "sbatch",
        "--parsable",
        "--partition",
        args.partition,
        "--nodelist",
        args.nodelist,
        f"--array=0-{len(ARMS) - 1}%1",
        "--gres=gpu:1",
        "--cpus-per-task=8",
        "--mem=48G",
        "--time=00:45:00",
        "--job-name=oe-reacher-historical",
        "--chdir",
        str(REPO),
        "--output",
        str(root / "job-%A_%a.out"),
        "--wrap",
        shell,
    ]
    job = subprocess.check_output(batch, text=True).strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError(f"unexpected sbatch response: {job!r}")
    manifest["jobs"]["evaluation_array"] = job
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"REACHER_HISTORICAL_MATRIX_JOB={job}")
    print(f"CAMPAIGN={root}")


if __name__ == "__main__":
    main()
