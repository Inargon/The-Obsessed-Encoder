#!/usr/bin/env python3
"""Submit one CPU job that polishes and packages the final paper figures."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess

from additional_files.bloop_paper_visuals_campaign import TRAINING_METRICS
from additional_files.bloop_tagged_diagnostics_campaign import DEFAULT_PYTHON, REPO


HERE = Path(__file__).resolve().parent
RESULTS = REPO / "leworldmodel/results"
DEFAULT_DENSE = RESULTS / "ours-final-paper-figures-20261006-101015/selectivity"
DEFAULT_MULTITASK = RESULTS / "bloop-success-rollouts-20261005-012438"
DEFAULT_MATCHED = RESULTS / "bloop-paper-visuals-20261005-055350"
DEFAULT_REACHER = RESULTS / "ours-reacher-success-showcase-20261006-022115/artifacts"


def quote(parts) -> str:
    return shlex.join(str(value) for value in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--dense", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--multitask-source", type=Path, default=DEFAULT_MULTITASK)
    parser.add_argument("--matched-source", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--reacher", type=Path, default=DEFAULT_REACHER)
    parser.add_argument("--partition", default="interactive")
    parser.add_argument("--nodelist", default="SPGL-1-1")
    args = parser.parse_args()

    metrics = {
        label: RESULTS / location for label, location in TRAINING_METRICS.items()
    }
    required = [
        args.python,
        args.dense / "dense-example-context-relative-uniform-main.png",
        args.dense / "dense-tag-sensitivity-summary.png",
        args.matched_source / "rollouts/jepa/summary.json",
        args.matched_source / "rollouts/repair/summary.json",
        args.reacher / "reacher-top2-overview.png",
        *[args.multitask_source / task / "summary.json"
          for task in ("clean_pusht", "tagged_pusht", "tworoom", "cube")],
        *metrics.values(),
    ]
    missing = [str(path) for path in required if not path.exists()]
    plan = {
        "artifact": "polished final paper figure bundle",
        "changes": [
            "move fixed-group note outside five-task bars",
            "make two-panel mechanism figure and retain four-panel appendix",
            "use Ours consistently in matched qualitative comparisons",
            "add a labelled nuisance-tag inset to tagged PushT rollouts",
        ],
        "reruns_models": False,
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
            str(HERE / "tests/test_ours_success_showcase.py"),
            str(HERE / "tests/test_repair_only_showcase.py"),
            str(HERE / "tests/test_render_ours_result_summary.py"),
        ],
        cwd=REPO, env=test_env, check=True,
    )

    campaign = RESULTS / (
        "ours-final-paper-polish-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    campaign.mkdir(parents=True, exist_ok=False)
    metric_args = []
    for label, path in metrics.items():
        metric_args.extend(("--run", f"{label}={path}"))
    commands = [
        quote([args.python, HERE / "render_ours_result_summary.py",
               "--out-dir", campaign / "results"]),
        quote([args.python, HERE / "plot_bloop_training_mechanism.py",
               *metric_args, "--out-dir", campaign / "mechanism"]),
        quote([args.python, HERE / "compose_ours_success_showcase.py",
               "--source", args.multitask_source,
               "--out-dir", campaign / "showcases/multitask",
               "--tasks", "clean_pusht,tagged_pusht,tworoom,cube", "--per-task", 2]),
        quote([args.python, HERE / "compose_repair_only_showcase.py",
               "--jepa-summary", args.matched_source / "rollouts/jepa/summary.json",
               "--repair-summary", args.matched_source / "rollouts/repair/summary.json",
               "--out-dir", campaign / "showcases/ours-only", "--examples", 2]),
        f"mkdir -p {shlex.quote(str(campaign / 'selectivity'))} "
        f"{shlex.quote(str(campaign / 'showcases/reacher'))}",
        f"cp {shlex.quote(str(args.dense / 'dense-example-context-relative-uniform-main.png'))} "
        f"{shlex.quote(str(campaign / 'selectivity/'))}",
        f"cp {shlex.quote(str(args.dense / 'dense-tag-sensitivity-summary.png'))} "
        f"{shlex.quote(str(campaign / 'selectivity/'))}",
        f"cp -a {shlex.quote(str(args.reacher))}/. "
        f"{shlex.quote(str(campaign / 'showcases/reacher/'))}",
        f"tar -czf {shlex.quote(str(campaign / 'ours-final-paper-polish.tar.gz'))} "
        f"-C {shlex.quote(str(campaign))} results mechanism selectivity showcases manifest.json",
        'echo "FINAL_PAPER_POLISH_COMPLETE"',
    ]
    shell = "\n".join([
        "set -euo pipefail",
        f"export PYTHONPATH={shlex.quote(str(REPO / 'leworldmodel'))}:{shlex.quote(str(REPO))}",
        *commands,
    ])
    batch = [
        "sbatch", "--parsable", "--partition", args.partition,
        "--cpus-per-task=4", "--mem=16G", "--time=00:30:00",
        "--job-name=oe-paper-polish", "--chdir", str(REPO),
        "--output", str(campaign / "job-%j.out"), "--wrap", shell,
    ]
    if args.nodelist:
        batch[4:4] = ["--nodelist", args.nodelist]
    job = subprocess.check_output(batch, text=True).strip().split(";")[0]
    manifest = {**plan, "job_id": job, "campaign": str(campaign)}
    (campaign / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"FINAL_PAPER_POLISH_JOB={job}")
    print(f"CAMPAIGN={campaign}")


if __name__ == "__main__":
    main()
