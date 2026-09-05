#!/usr/bin/env python3
"""Grid runner + figure factory for the LeWM example.

One run = one printed, human-runnable ``train.py`` subprocess (launched from
``leworldmodel/``).  Results land under ``RESULTS_DIR/<config>_seed<k>/``
(``metrics.jsonl`` mirror, ``summary.json`` done-marker); completed runs are
skipped on re-invocation (``--force`` / ``--rerun`` restart from scratch).
``--plot-only`` renders the example's figure set from the local run files
without launching anything.

    python additional_files/run.py --seeds 1 --gpus 0        # all four arms
    python additional_files/run.py --configs baseline,randgoal
    python additional_files/run.py --plot-only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

ADDITIONAL_FILES_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = ADDITIONAL_FILES_DIR.parent  # leworldmodel/
REPO_ROOT = EXAMPLE_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
from common import results_io, runner  # noqa: E402

CASE = "lewm"
CONFIGS_YAML = ADDITIONAL_FILES_DIR / "configs.yaml"
# Checkpoint the published figure 10 is read from. Pinned rather than tracking
# the campaign's last epoch, so re-rendering the figure cannot silently swap the
# encoder under it. Bump this to point the figure at a longer run.
READOUT_EPOCH = 5


def load_configs() -> dict:
    with CONFIGS_YAML.open() as f:
        return yaml.safe_load(f)


def build_spec(config: str, spec: dict, cfg: dict, seed: int,
               args: argparse.Namespace) -> runner.RunSpec:
    name = f"{config}_seed{seed}"
    run_dir = runner.run_dir_for(args.results_dir, name)
    command = [
        sys.executable,
        "train.py",
        f"output_model_name={name}",
        f"subdir={name}",
        f"trainer.max_epochs={args.epochs or cfg['epochs']}",
        f"seed={seed}",
        "+seed_everything=true",
        f"+preview.num_episodes={cfg['preview_episodes']}",
        f"+metrics_jsonl_path={os.path.join(run_dir, 'metrics.jsonl')}",
        f"+eval.every_n_steps={cfg['eval_every_n_steps']}",
        f"+checkpoint.every_n_steps={cfg['checkpoint_every_n_steps']}",
        *spec["overrides"],
        *spec["eval_overrides"],
    ]
    suites = spec.get("pair_suites")
    if suites:
        command += [
            f"+pair.every_n_steps={cfg['pair_every_n_steps']}",
            f"+pair.suites=[{','.join(suites)}]",
        ]
    # wandb is optional monitoring; entity comes from the account.
    # The template pins id=${subdir} with resume=allow -- a deterministic id
    # that collides across machines and campaigns and silently appends to
    # older runs; drop both so every invocation gets a fresh random id.
    # stable_pretraining cannot START a run offline: its wandb init assumes
    # an earlier offline run dir exists to reuse and crashes on a fresh box,
    # so WANDB_MODE=offline/disabled turns wandb off instead of silencing it.
    if os.environ.get("WANDB_MODE", "").lower() in ("offline", "dryrun", "disabled"):
        command += ["wandb.enabled=false"]
    else:
        command += [
            "wandb.enabled=true",
            "wandb.config.project=obsessed-encoder-lewm",
            f"wandb.config.name={CASE}_{name}",
            "~wandb.config.entity",
            "~wandb.config.id",
            "~wandb.config.resume",
        ]
    command += [
        # Per-run sidecar/checkpoint root: stable-pretraining writes its
        # wandb_resume.json under trainer.default_root_dir, which otherwise
        # is the shared CWD -- concurrent arms would overwrite each other's
        # run id there.
        f"++trainer.default_root_dir={run_dir}",
    ]
    env = {
        "PYTHONPATH": f"{REPO_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        "WANDB_DIR": run_dir,
    }
    if getattr(args, "extra_opts", ""):
        command += args.extra_opts.split()
    if args.tag:
        env["WANDB_RUN_GROUP"] = args.tag
    return runner.RunSpec(name=name, config=config, seed=seed, command=command,
                          env=env, cwd=str(EXAMPLE_DIR))


def _stablewm_home() -> str:
    return os.environ.get("STABLEWM_HOME",
                          os.path.join(os.path.expanduser("~"), ".stable_worldmodel"))


def _clear_checkpoint_cache(name: str) -> None:
    """Delete a rerun's stale checkpoints from the stable-worldmodel cache.

    train.py writes checkpoints under STABLEWM_HOME/checkpoints/<name>, outside
    RESULTS_DIR; without this, the encoder-readout figure could load a stale
    higher-step checkpoint from a previous attempt (run names are deterministic
    across campaigns).
    """
    import shutil

    ckpt_dir = os.path.join(_stablewm_home(), "checkpoints", name)
    if os.path.isdir(ckpt_dir):
        print(f"[run] {name}: clearing stale checkpoint cache {ckpt_dir}")
        shutil.rmtree(ckpt_dir)


def write_summary(spec: runner.RunSpec, results_dir: str) -> None:
    run_dir = runner.run_dir_for(results_dir, spec.name)
    rows = results_io.read_metrics(run_dir)
    if not rows:
        raise RuntimeError(f"{spec.name}: training exited 0 but wrote no metrics")
    final: Dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            if (
                key in ("fit/pred_loss", "fit/allocation_loss",
                        "fit/local_effective_rank", "fit/control_loss",
                        "fit/inverse_dynamics_loss", "fit/action_cycle_loss",
                        "fit/reachability_loss", "fit/reachability_accuracy",
                        "fit/reachability_shuffled_accuracy",
                        "fit/reachability_action_margin",
                        "fit/action_router_gate",
                        "eval/success_rate")
                or key.startswith("pair/")
                or key.startswith("fit/inverse_horizon_")
                or key.startswith("fit/context_")
                or key.startswith("fit/dynamic_")
            ):
                final[f"final_{key}"] = value
    steps = [r["step"] for r in rows if isinstance(r.get("step"), int)]
    results_io.write_summary(run_dir, {
        "case": CASE,
        "config": spec.config,
        "seed": spec.seed,
        "steps": max(steps) if steps else None,
        **final,
    })


def render_figures(args) -> None:
    results = os.path.abspath(args.results_dir)
    out_dir = os.path.join(results, "figures")
    env = {**os.environ,
           "PYTHONPATH": f"{REPO_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}

    def step(script: str, *extra: str) -> None:
        cmd = [sys.executable, str(ADDITIONAL_FILES_DIR / script), *extra]
        print(f"[figures] {script}:\n  " + " ".join(cmd))
        subprocess.run(cmd, cwd=EXAMPLE_DIR, env=env, check=True)

    step("graphs/training_curves.py", "--results-dir", results, "--out-dir", out_dir)
    step("graphs/dataset_montage.py", "--out", os.path.join(out_dir, "f7_lewm_datasets.png"))
    readout_ckpt = os.path.join(_stablewm_home(), "checkpoints", "randgoal_seed0",
                                f"weights_epoch_{READOUT_EPOCH}.pt")
    if os.path.exists(readout_ckpt):
        step("graphs/encoder_readout.py", "--run-name", "randgoal_seed0",
             "--epoch", str(READOUT_EPOCH), "--out-dir", out_dir)
    else:
        # Budget-capped runs never reach epoch 5; a hard requirement here
        # failed campaigns at their very last step.
        print(f"[figures] encoder_readout SKIPPED: no {readout_ckpt} "
              f"(run randgoal to epoch {READOUT_EPOCH} to render figure 10)")
    print(f"[figures] -> {out_dir}")


def main(argv: Optional[List[str]] = None) -> int:
    cfg = load_configs()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    runner.add_common_arguments(parser, default_tag="")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"default {cfg['epochs']} (the paper's operating point)")
    parser.add_argument("--extra-opts", default="",
                        help="space-separated hydra overrides appended to every "
                             "run (shakeouts, e.g. '+trainer.max_steps=2000')")
    parser.set_defaults(seeds="3", configs=",".join(sorted(cfg["arms"])))
    args = parser.parse_args(argv)

    if args.plot_only:
        render_figures(args)
        return 0

    unknown = [c for c in args.configs.split(",") if c and c not in cfg["arms"]]
    if unknown:
        raise SystemExit(f"unknown config(s) {unknown}; available: {sorted(cfg['arms'])}")

    # Seed-major order: every completed stretch is a full arm set at one seed,
    # so an interrupted campaign still yields arm-comparable figures.
    specs = [build_spec(config, cfg["arms"][config], cfg, seed, args)
             for seed in runner.parse_seeds(args.seeds)
             for config in args.configs.split(",") if config]
    if args.dry_run:
        runner.print_dry_run(specs, [g.strip() for g in args.gpus.split(",") if g.strip()])
        return 0
    to_run = []
    for spec in specs:
        if runner.should_run(args.results_dir, spec, force=args.force, rerun=args.rerun):
            # LeWM has no mid-run resume: a summary-less dir is a dead partial
            # attempt, and rerunning on top of it would append a second
            # from-scratch trajectory to its metrics.jsonl -- so every rerun
            # (interrupted, --force, or --rerun) starts from a clean dir.
            had_partial = os.path.isdir(
                runner.run_dir_for(args.results_dir, spec.name))
            runner.clear_run_dir(args.results_dir, spec.name)
            if had_partial:
                # Only a rerun of THIS results dir may prune the global
                # stable-worldmodel cache: a fresh campaign reusing run names
                # must not delete another campaign's checkpoints.
                _clear_checkpoint_cache(spec.name)
            to_run.append(spec)
        else:
            print(f"[run] {spec.name}: summary.json present, skipping "
                  "(--force or --rerun to redo)")

    results = runner.execute_pool(
        to_run, [g.strip() for g in args.gpus.split(",") if g.strip()],
        on_success=lambda spec: write_summary(spec, args.results_dir))
    code = runner.summarize_results(results)
    if code == 0:
        render_figures(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
