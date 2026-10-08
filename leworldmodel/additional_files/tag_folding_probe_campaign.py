"""Submit one smoke-gated Slurm job for the controlled tag-folding audit."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--partition", default="interactive")
    parser.add_argument("--nodelist", default="SPGL-1-1")
    parser.add_argument("--python-bin", default="/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python")
    parser.add_argument("--scenes", type=int, default=8)
    parser.add_argument("--colors", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.submit or not args.confirm_run:
        raise SystemExit("Pass --submit --confirm-run to launch the smoke-gated campaign.")
    repo = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    campaign = repo / "leworldmodel" / "results" / f"tag-folding-probe-{timestamp}"
    campaign.mkdir(parents=True, exist_ok=False)
    script = repo / "leworldmodel" / "additional_files" / "tag_folding_probe.py"
    smoke = campaign / "smoke-artifacts"
    artifacts = campaign / "artifacts"

    base = [args.python_bin, str(script)]
    smoke_command = base + [
        "--out-dir", str(smoke),
        "--scenes", "2",
        "--colors", "48",
        "--projections", "32",
        "--batch-size", "24",
    ]
    full_command = base + [
        "--out-dir", str(artifacts),
        "--scenes", str(args.scenes),
        "--colors", str(args.colors),
        "--projections", "512",
        "--batch-size", "64",
    ]
    archive = campaign / "tag-folding-probe-visuals.tar.gz"
    shell = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(repo))}",
            f"export PYTHONPATH={shlex.quote(str(repo / 'leworldmodel'))}:{shlex.quote(str(repo))}${{PYTHONPATH:+:$PYTHONPATH}}",
            "export LOCAL_DATASET_DIR=/grp01/ids_compcog/song/swm",
            "export STABLEWM_HOME=/grp01/ids_compcog/song/swm",
            shlex.join(smoke_command),
            "echo TAG_FOLDING_PROBE_SMOKE_COMPLETE",
            shlex.join(full_command),
            shlex.join(
                [
                    "tar", "-czf", str(archive), "-C", str(artifacts),
                    "tag-folding-probe-summary.png", "tag-folding-probe.json",
                ]
            ),
            "echo TAG_FOLDING_PROBE_CAMPAIGN_COMPLETE",
        ]
    )
    manifest = {
        "artifact": "controlled RGB-tag folding audit",
        "campaign": str(campaign),
        "protocol": {
            "smoke": "2 scenes x 48 colors",
            "full": f"{args.scenes} scenes x {args.colors} colors",
            "models": ["jepa", "ours"],
            "outputs": [
                "absolute tag sensitivity",
                "global participation dimension",
                "local TwoNN dimension",
                "folding ratio",
                "held-out projection Gaussian gap",
            ],
        },
    }
    (campaign / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sbatch = [
        "sbatch", "--parsable",
        "--partition", args.partition,
        "--nodelist", args.nodelist,
        "--gres=gpu:1",
        "--cpus-per-task=4",
        "--mem=24G",
        "--time=00:30:00",
        "--job-name=oe-tag-fold",
        "--chdir", str(repo),
        "--output", str(campaign / "job-%j.out"),
        "--wrap", shell,
    ]
    job = subprocess.check_output(sbatch, text=True).strip().split(";")[0]
    print(json.dumps(manifest, indent=2))
    print(f"TAG_FOLDING_PROBE_JOB={job}")
    print(f"CAMPAIGN={campaign}")


if __name__ == "__main__":
    main()
