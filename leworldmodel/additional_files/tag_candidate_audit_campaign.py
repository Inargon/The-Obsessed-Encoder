"""Launch the frozen Tagged PushT candidate-ranking diagnostic on Slurm."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys

from counterfactual_planning_campaign import HERE, REPO, environment, source_digest
from independent_tag_ablation import ARMS, checkpoint_spec, read_json, write_json


def evaluate(previous: Path, root: Path, dataset: str, env: dict[str, str]) -> None:
    from tag_candidate_audit import summarize_candidate_audit

    manifest = read_json(root / "manifest.json")
    if manifest["source_sha256"] != source_digest():
        raise RuntimeError("Source/config changed since submission")
    old = read_json(previous)
    labels = ("base", *ARMS)
    if set(old["runs"]) != set(labels):
        raise ValueError("Expected the original checkpoint and all four interventions")
    protocol = old["protocol"]
    raw = root / "candidate-costs.json"
    command = [
        sys.executable, "-u", str(HERE / "diagnose_tag_intervention.py"),
        "--out", str(raw), "--candidate-audit", "--dataset", dataset,
        "--num-clips", str(len(protocol["indices"])),
        "--candidates", str(protocol["candidates"]), "--seed", str(protocol["seed"]),
    ]
    for key in ("history", "horizon", "frameskip", "tag_size"):
        command += ["--" + key.replace("_", "-"), str(protocol[key])]
    checkpoint_root = Path(env["STABLEWM_HOME"]) / "checkpoints"
    for name in labels:
        run = old["runs"][name]
        checkpoint = checkpoint_root / run["run_name"] / run["checkpoint"]
        command += ["--checkpoint", checkpoint_spec(name, checkpoint, checkpoint_root)]
    subprocess.run(command, check=True, cwd=REPO, env=env)
    evaluated = read_json(raw)
    audit = {}
    for name in labels:
        run = evaluated["runs"][name]
        if run["sha256"] != old["runs"][name]["sha256"]:
            raise RuntimeError(f"{name}: checkpoint SHA-256 differs from the previous evaluation")
        audit[name] = run["candidate_audit"]
    results = summarize_candidate_audit(audit, protocol["seed"])
    for name, run in evaluated["runs"].items():
        suffix = "/selected_action_normalized_l2"
        results[name]["first_action_block_changed"] = {
            key.removesuffix(suffix): sum(value > 0 for value in values)
            for key, values in run["per_clip"].items()
            if key.endswith(suffix)
        }
    write_json(root / "report.json", {
        "dataset": dataset,
        "query_episodes": len(evaluated["protocol"]["episode_split"]["query"]),
        "bank_episodes": len(evaluated["protocol"]["episode_split"]["candidate"]),
        "training_independent_episodes_verified": False,
        "bootstrap": evaluated["protocol"]["bootstrap"],
        "reference": "same sampled reference color on both endpoints; observed past and bank fixed across tag views",
        "strict_flip": "a candidate defeats the original winner by strictly lower cost",
        "gap": "original runner-up minus winner cost, divided by candidate cost standard deviation",
        "anchors": "frozen replay of the one-step training comparison with one disjoint negative per query",
        "checkpoint_sha256": {name: old["runs"][name]["sha256"] for name in labels},
        "runs": results,
    })
    print(f"TAG_CANDIDATE_AUDIT_COMPLETE {root / 'report.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", required=True, type=Path, help="Previous tag-intervention.json")
    parser.add_argument("--dataset", help="Optional independent evaluation data")
    parser.add_argument("--partition", default="gpu_shared")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    previous = args.previous.resolve()
    if not previous.is_file():
        parser.error(f"Previous evaluation missing: {previous}")
    old = read_json(previous)
    dataset = args.dataset or old["protocol"]["dataset"]
    env = environment()
    env.update(HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    if args.worker:
        evaluate(previous, args.worker, dataset, env)
        return

    root = REPO / "leworldmodel/results" / f"tag-candidate-audit-{datetime.now():%Y%m%d-%H%M%S-%f}"
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"source_sha256": source_digest(), "previous": str(previous), "dataset": dataset}
    write_json(root / "manifest.json", manifest)
    worker = [
        sys.executable, "-u", str(Path(__file__).resolve()), "--previous", str(previous),
        "--dataset", dataset, "--worker", str(root),
    ]
    response = subprocess.check_output([
        "sbatch", "--parsable", "--partition", args.partition, "--job-name=oe-tag-candidate-audit",
        "--gres=gpu:1", "--cpus-per-task=4", "--mem=24G", "--time=00:30:00",
        "--chdir", str(REPO), "--output", str(root / "job-%j.out"), "--wrap", shlex.join(worker),
    ], text=True, env=env, cwd=REPO)
    job_id = response.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected sbatch response: {response!r}")
    manifest["job_id"] = job_id
    write_json(root / "manifest.json", manifest)
    print(f"Evaluation job: {job_id}\n"
          f"Monitor: sacct -j {job_id} --format=JobID,State,Elapsed,ExitCode\n"
          f"Report after evaluation: {root / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
