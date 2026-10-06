#!/usr/bin/env python3
"""Submit one parallel campaign for the remaining main-paper figures."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess

from additional_files.bloop_paper_visuals_campaign import TRAINING_METRICS
from additional_files.bloop_tagged_diagnostics_campaign import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PYTHON,
    REPO,
)


HERE = Path(__file__).resolve().parent
DEFAULT_DENSE = REPO / (
    "leworldmodel/results/dense-planner-selectivity-20261006-084358/"
    "artifacts/dense-planner-selectivity.json"
)
DEFAULT_SHOWCASE = REPO / (
    "leworldmodel/results/ours-success-showcase-20261005-203312/artifacts"
)
DEFAULT_REACHER = REPO / (
    "leworldmodel/results/ours-reacher-success-showcase-20261006-022115/artifacts"
)
DEFAULT_REPAIR_ONLY = REPO / (
    "leworldmodel/results/bloop-repair-only-showcase-20261005-195538/artifacts"
)


def quote(parts) -> str:
    return shlex.join(str(value) for value in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dense-json", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--showcase", type=Path, default=DEFAULT_SHOWCASE)
    parser.add_argument("--reacher-showcase", type=Path, default=DEFAULT_REACHER)
    parser.add_argument("--repair-only", type=Path, default=DEFAULT_REPAIR_ONLY)
    parser.add_argument("--partition", default="interactive")
    parser.add_argument("--nodelist", default="SPGL-1-1")
    args = parser.parse_args()

    metrics = {
        label: REPO / "leworldmodel/results" / location
        for label, location in TRAINING_METRICS.items()
    }
    required = {
        "python": args.python,
        "dense diagnostic": args.dense_json,
        "multi-task showcase": args.showcase,
        "reacher showcase": args.reacher_showcase,
        "repair-only showcase": args.repair_only,
        **{f"training metrics: {label}": path for label, path in metrics.items()},
    }
    missing = [label for label, path in required.items() if not path.exists()]
    plan = {
        "artifact": "final main-paper figure bundle",
        "parallel_tasks": [
            "two-column JEPA-versus-Ours selectivity plus five-task CEM result plot",
            "training-time EMA gradient geometry",
            "curated Ours successes and matched JEPA-fail/Ours-success examples",
        ],
        "scientific_guards": [
            "GC-IDM is excluded",
            "success bars identify the single fixed evaluation group",
            "relative heatmaps report allocation and retain absolute tag effects in labels",
            "tag-masked residual stays appendix-only",
            "existing fixed outputs are copied without visual re-selection",
        ],
        "missing": missing,
    }
    print(json.dumps(plan, indent=2))
    if not args.submit:
        print("PLAN ONLY: pass --submit --confirm-run to launch.")
        return
    if not args.confirm_run:
        parser.error("refusing to submit without --confirm-run")
    if missing:
        parser.error("missing: " + ", ".join(missing))

    test_env = dict(os.environ)
    test_env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO / "leworldmodel"), str(REPO), test_env.get("PYTHONPATH")])
    )
    subprocess.run(
        [
            str(args.python), "-m", "pytest", "-q",
            str(HERE / "tests/test_render_dense_selectivity_magnitude.py"),
            str(HERE / "tests/test_render_ours_result_summary.py"),
        ],
        cwd=REPO, env=test_env, check=True,
    )

    campaign = REPO / "leworldmodel/results" / (
        "ours-final-paper-figures-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    campaign.mkdir(parents=True, exist_ok=False)
    (campaign / "manifest.json").write_text(
        json.dumps({**plan, "campaign": str(campaign)}, indent=2) + "\n",
        encoding="utf-8",
    )

    metric_args = []
    for label, path in metrics.items():
        metric_args.extend(("--run", f"{label}={path}"))
    dense_command = quote([
        args.python, HERE / "render_dense_selectivity_magnitude.py",
        "--input", args.dense_json,
        "--out-dir", campaign / "selectivity",
        "--example-position", 17,
    ])
    result_command = quote([
        args.python, HERE / "render_ours_result_summary.py",
        "--out-dir", campaign / "results",
    ])
    mechanism_command = quote([
        args.python, HERE / "plot_bloop_training_mechanism.py",
        *metric_args, "--out-dir", campaign / "mechanism",
    ])
    copy_commands = [
        f"mkdir -p {shlex.quote(str(campaign / 'showcases/multitask'))}",
        f"mkdir -p {shlex.quote(str(campaign / 'showcases/reacher'))}",
        f"mkdir -p {shlex.quote(str(campaign / 'showcases/repair-only'))}",
        f"cp -a {shlex.quote(str(args.showcase))}/. {shlex.quote(str(campaign / 'showcases/multitask'))}/",
        f"cp -a {shlex.quote(str(args.reacher_showcase))}/. {shlex.quote(str(campaign / 'showcases/reacher'))}/",
        f"cp -a {shlex.quote(str(args.repair_only))}/. {shlex.quote(str(campaign / 'showcases/repair-only'))}/",
    ]
    shell = "\n".join([
        "set -euo pipefail",
        f"export PYTHONPATH={shlex.quote(str(REPO / 'leworldmodel'))}:{shlex.quote(str(REPO))}",
        f"export STABLEWM_HOME={shlex.quote(str(args.data_root))}",
        f"export LOCAL_DATASET_DIR={shlex.quote(str(args.data_root))}",
        'case "$SLURM_ARRAY_TASK_ID" in',
        f"  0) {dense_command}; {result_command} ;;",
        f"  1) {mechanism_command} ;;",
        "  2) " + "; ".join(copy_commands) + " ;;",
        '  *) echo "invalid task" >&2; exit 2 ;;',
        "esac",
        'echo "FINAL_PAPER_FIGURE_TASK_COMPLETE index=$SLURM_ARRAY_TASK_ID"',
    ])
    batch = [
        "sbatch", "--parsable", "--partition", args.partition,
        "--array=0-2%3", "--cpus-per-task=4", "--mem=24G", "--time=00:45:00",
        "--job-name=oe-paper-figs", "--chdir", str(REPO),
        "--output", str(campaign / "figure-%A_%a.out"), "--wrap", shell,
    ]
    if args.nodelist:
        batch[4:4] = ["--nodelist", args.nodelist]
    figures_job = subprocess.check_output(batch, text=True).strip().split(";")[0]

    post_shell = "\n".join([
        "set -euo pipefail",
        f"tar -czf {shlex.quote(str(campaign / 'ours-final-paper-figures.tar.gz'))} "
        f"-C {shlex.quote(str(campaign))} selectivity results mechanism showcases manifest.json",
        f"find {shlex.quote(str(campaign))} -type f -printf '%P %s bytes\\n' | sort > "
        f"{shlex.quote(str(campaign / 'files.txt'))}",
        'echo "FINAL_PAPER_FIGURES_COMPLETE"',
    ])
    post_batch = [
        "sbatch", "--parsable", "--partition", args.partition,
        "--cpus-per-task=2", "--mem=4G", "--time=00:15:00",
        "--job-name=oe-paper-pack", "--chdir", str(REPO),
        "--dependency", f"afterok:{figures_job}", "--kill-on-invalid-dep=yes",
        "--output", str(campaign / "package-%j.out"), "--wrap", post_shell,
    ]
    if args.nodelist:
        post_batch[4:4] = ["--nodelist", args.nodelist]
    package_job = subprocess.check_output(post_batch, text=True).strip().split(";")[0]

    manifest = {**plan, "campaign": str(campaign), "figures_job": figures_job,
                "package_job": package_job}
    (campaign / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"FINAL_PAPER_FIGURES_JOB={figures_job}")
    print(f"FINAL_PAPER_PACKAGE_JOB={package_job}")
    print(f"CAMPAIGN={campaign}")


if __name__ == "__main__":
    main()
