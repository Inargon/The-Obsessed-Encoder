"""Grid runner + figure factory for the DINOv3 example.

One run = one printed, human-runnable ``torchrun ... dinov3/train/train.py``
subprocess with the arm's ``DINOV3_WM_*`` environment.  Results land under
``RESULTS_DIR/<config>_seed<k>/`` (``train/`` = upstream's output dir,
``metrics.jsonl`` mirror, ``dense_eval_results.csv``, ``summary.json``
done-marker).  Upstream's trainer natively resumes from its output dir, so
re-invoking a killed run continues it; after training, the dense tail runs the
truncated NYU-depth probe on every teacher milestone that has no CSV row yet
(the fill-missing semantics), appending ``dense/nyu_rmse`` rows to the same
``metrics.jsonl``.  ``--plot-only`` renders the figure set from local files.

    uv run python dinov3/additional_files/run.py --seeds 3 --gpus 0,1,2
    uv run python dinov3/additional_files/run.py --plot-only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

_VENDORED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_ROOT = os.path.dirname(_VENDORED_ROOT)
for _p in (ARTIFACT_ROOT, _VENDORED_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402

from common import plotting, results_io, runner  # noqa: E402

from additional_files.dense_eval import (  # noqa: E402
    DEFAULT_HEAD_ITERS,
    RESULT_MARKER,
    watermark_kwargs_from_env,
)

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(_VENDORED_ROOT, "dinov3", "train", "train.py")
DENSE_EVAL_SCRIPT = os.path.join(HERE, "dense_eval.py")
BASE_CONFIG = os.path.join(HERE, "configs", "vitl_im1k.yaml")
ARMS_YAML = os.path.join(HERE, "configs", "arms.yaml")
CASE = "dinov3"

DEFAULT_MAX_ITER = 50_000

_MILESTONE_RE = re.compile(r"training_(\d+)")

# torchrun fabricates these; they must not leak into the dense-eval child (its
# own single-rank store would collide with the parent's).
_TORCHRUN_ENV_VARS = ("TORCHELASTIC_RUN_ID", "MASTER_ADDR", "MASTER_PORT", "RANK",
                      "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")

# Every arm sets ALL of these explicitly (empty string = unset for the shared
# decoder), so an ambient shell export can never leak a watermark into the
# clean arm -- or flip a control -- without appearing in the printed command.
WM_ENV_VARS = ("DINOV3_WM_OPACITY", "DINOV3_WM_MODULUS", "DINOV3_WM_TILE",
               "DINOV3_WM_BITS", "DINOV3_WM_ANCHOR", "DINOV3_WM_RANDOM_ANCHOR",
               "DINOV3_WM_REPEAT")

DENSE_CSV_NAME = "dense_eval_results.csv"
DENSE_CSV_COLUMNS = [
    "config", "seed", "teacher_iteration", "protocol", "head_iters", "tta",
    "n_test_images", "rmse", "abs_rel", "a1", "status", "error",
    "wall_clock_s", "timestamp",
]


def load_arms() -> Dict[str, Dict[str, str]]:
    with open(ARMS_YAML) as f:
        arms = yaml.safe_load(f)
    return {name: dict(env or {}) for name, env in arms.items()}


def build_spec(config: str, arm_env: Dict[str, str], seed: int,
               args: argparse.Namespace) -> runner.RunSpec:
    name = f"{config}_seed{seed}"
    run_dir = runner.run_dir_for(args.results_dir, name)
    data_dir = os.path.abspath(args.data_dir)
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node=1", TRAIN_SCRIPT,
        "--config-file", BASE_CONFIG,
        "--output-dir", os.path.join(run_dir, "train"),
        "--seed", str(seed),
        f"train.seed={seed}",
    ]
    if args.extra_opts:
        command += args.extra_opts.split()
    env = {
        **{var: "" for var in WM_ENV_VARS},
        **arm_env,
        "DINOV3_PROBE": "1",
        "DINOV3_MAX_ITER": str(args.max_iter),
        "DINOV3_RUN_DIR": run_dir,
        "DINOV3_WANDB_NAME": f"{CASE}_{name}",
        "DATA_DIR": data_dir,
        "HF_HOME": os.path.join(data_dir, "hf_cache"),
        # ViT-L at the recipe batch peaks ~76 GiB on an 80 GiB card; without
        # expandable segments the allocator fragments and the first backward
        # OOMs (measured). Allocator config only, no effect on the math.
        # Deliberately the "deprecated" name: torch 2.9.1 warns for it but
        # does not yet honor the replacement PYTORCH_ALLOC_CONF for
        # expandable segments (verified via memory._snapshot).
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONPATH": f"{_VENDORED_ROOT}{os.pathsep}{ARTIFACT_ROOT}",
        # wandb's own files (incl. offline runs awaiting `wandb sync`) belong on
        # the results volume, not the training cwd.
        "WANDB_DIR": run_dir,
    }
    if args.tag:
        env["DINOV3_WANDB_GROUP"] = args.tag
    return runner.RunSpec(name=name, config=config, seed=seed, command=command,
                          env=env, cwd=_VENDORED_ROOT)


# ---------------------------------------------------------------------------- #
# The dense tail: truncated NYU-depth probe per teacher milestone
# ---------------------------------------------------------------------------- #
def local_milestones(run_dir: str) -> List[int]:
    """Milestone iterations dumped by do_test: ``train/eval/training_<it>/``
    holding a teacher_checkpoint.pth (a dir without one is skipped)."""
    eval_dir = os.path.join(run_dir, "train", "eval")
    if not os.path.isdir(eval_dir):
        return []
    out = []
    for name in os.listdir(eval_dir):
        m = _MILESTONE_RE.fullmatch(name)
        if m and os.path.exists(os.path.join(eval_dir, name, "teacher_checkpoint.pth")):
            out.append(int(m.group(1)))
    return sorted(out)


def dense_csv_path(run_dir: str) -> str:
    return os.path.join(run_dir, DENSE_CSV_NAME)


def read_dense_rows(run_dir: str) -> List[Dict[str, str]]:
    path = dense_csv_path(run_dir)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def write_dense_rows(run_dir: str, rows: List[Dict[str, str]]) -> None:
    # Whole-file rewrite every time -- the cheapest resumable bookkeeping.
    path = dense_csv_path(run_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DENSE_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def pending_milestones(milestones: List[int], rows: List[Dict[str, str]], *,
                       head_iters: int) -> List[int]:
    """Milestones with no row for this instrument.  ``failed`` rows count as
    done (a poisoned checkpoint must not re-run forever); delete a row -- or use
    --retry-failed -- to re-evaluate."""
    done = {int(r["teacher_iteration"]) for r in rows
            if r.get("status") in ("ok", "failed")
            and int(r.get("head_iters") or 0) == head_iters}
    return [m for m in milestones if m not in done]


def _dense_child_env(spec_env: Dict[str, str], gpu: Optional[str]) -> Dict[str, str]:
    env = dict(os.environ)
    env.update(spec_env)
    for var in _TORCHRUN_ENV_VARS:
        env.pop(var, None)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_probe_child(command: List[str], env: Dict[str, str]) -> Tuple[str, object]:
    import threading

    timeout = int(os.environ.get("DINOV3_DENSE_EVAL_TIMEOUT_SEC", "7200"))
    print(f"[dense] {' '.join(command)}", flush=True)
    lines: List[str] = []
    try:
        proc = subprocess.Popen(command, env=env, cwd=_VENDORED_ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        assert proc.stdout is not None

        # Drain stdout on a side thread so the deadline binds the PROCESS: a
        # child that hangs silently (producing no output) must still be killed,
        # or one stuck milestone stalls the whole grid.
        def _pump():
            for line in proc.stdout:
                lines.append(line.rstrip("\n"))

        reader = threading.Thread(target=_pump, daemon=True)
        reader.start()
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=60)
            return "failed", f"probe timed out after {timeout}s (killed)"
        reader.join(timeout=30)
    except OSError as e:
        return "failed", f"probe launch failed: {e}"
    if rc != 0:
        return "failed", f"probe exited {rc}: " + " | ".join(lines[-8:])
    for line in reversed(lines):
        if line.startswith(RESULT_MARKER):
            return "ok", json.loads(line[len(RESULT_MARKER):])
    return "failed", "probe exited 0 but printed no DENSE_EVAL_RESULT line"


def run_dense_tail(spec: runner.RunSpec, args: argparse.Namespace,
                   gpu: Optional[str]) -> Dict[str, int]:
    """Evaluate every pending milestone; returns {'ok': n, 'failed': n}.

    A run whose cap is below the milestone period legitimately has zero
    milestones; a run that SHOULD have dumped milestones but shows none is an
    error, never a quiet success.
    """
    run_dir = runner.run_dir_for(args.results_dir, spec.name)
    milestones = local_milestones(run_dir)
    eval_period = _eval_period(args)
    if not milestones:
        if args.max_iter >= eval_period:
            raise RuntimeError(
                f"{spec.name}: trained to {args.max_iter} with milestone period "
                f"{eval_period} but no teacher milestones found under "
                f"{run_dir}/train/eval")
        return {"ok": 0, "failed": 0}

    rows = read_dense_rows(run_dir)
    if args.retry_failed:
        rows = [r for r in rows if not (r.get("status") == "failed"
                                        and int(r.get("head_iters") or 0) == args.head_iters)]
        write_dense_rows(run_dir, rows)
    pending = pending_milestones(milestones, rows, head_iters=args.head_iters)
    if not pending:
        # Tally only this instrument's rows -- a stale failed row from a
        # different head_iters (e.g. an old smoke) must not veto completion.
        mine = [r for r in rows if int(r.get("head_iters") or 0) == args.head_iters]
        return {"ok": sum(r["status"] == "ok" for r in mine),
                "failed": sum(r["status"] == "failed" for r in mine)}

    watermark = watermark_kwargs_from_env(spec.env)
    counts = {"ok": 0, "failed": 0}
    for iteration in pending:
        ckpt = os.path.join(run_dir, "train", "eval", f"training_{iteration}",
                            "teacher_checkpoint.pth")
        probe_out = os.path.join(run_dir, "dense_eval", f"training_{iteration}")
        command = [sys.executable, DENSE_EVAL_SCRIPT,
                   "--ckpt", ckpt, "--output-dir", probe_out,
                   "--head-iters", str(args.head_iters)]
        if args.dense_extra_opts:
            command += ["--extra-opts", args.dense_extra_opts]
        if watermark is not None:
            command += ["--watermark-json", json.dumps(watermark)]
        status, payload = _run_probe_child(command, _dense_child_env(spec.env, gpu))
        row = {
            "config": spec.config, "seed": spec.seed,
            "teacher_iteration": iteration, "protocol": "milestone",
            "head_iters": args.head_iters, "tta": False,
            "n_test_images": "", "rmse": "", "abs_rel": "", "a1": "",
            "status": status, "error": "", "wall_clock_s": "",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if status == "ok":
            row.update({k: payload[k] for k in ("n_test_images", "rmse", "abs_rel",
                                                "a1", "wall_clock_s")})
            results_io.append_metrics(run_dir, {
                "step": iteration,
                "dense/nyu_rmse": float(payload["rmse"]),
                "dense/nyu_abs_rel": float(payload["abs_rel"]),
                "dense/nyu_a1": float(payload["a1"]),
                "dense/teacher_iteration": iteration,
            })
        else:
            row["error"] = str(payload)[:500]
            print(f"[dense] {spec.name} milestone {iteration} FAILED: {payload}",
                  flush=True)
        counts[status] += 1
        rows.append(row)
        write_dense_rows(run_dir, rows)
    return counts


def _eval_period(args: argparse.Namespace) -> int:
    for token in (args.extra_opts or "").split():
        if token.startswith("evaluation.eval_period_iterations="):
            return int(token.split("=", 1)[1])
    with open(BASE_CONFIG) as f:
        return int(yaml.safe_load(f)["evaluation"]["eval_period_iterations"])


# ---------------------------------------------------------------------------- #
# Per-run pipeline: train (native resume) -> dense tail -> summary
# ---------------------------------------------------------------------------- #
def launch_and_tail(spec: runner.RunSpec, gpu: Optional[str],
                    args: argparse.Namespace) -> int:
    code = runner.launch(spec, gpu)
    if code != 0:
        return code
    try:
        counts = run_dense_tail(spec, args, gpu)
    except RuntimeError as e:
        print(f"[dense] {e}", flush=True)
        return 1
    run_dir = runner.run_dir_for(args.results_dir, spec.name)
    rows = results_io.read_metrics(run_dir)
    if not rows:
        print(f"[run] {spec.name}: training exited 0 but wrote no metrics", flush=True)
        return 1
    if counts["failed"]:
        # A run with failed dense milestones is NOT complete: no done-marker, so
        # re-invocation reaches the tail again (failed rows are not retried until
        # --retry-failed, keeping a poisoned checkpoint from looping).
        print(f"[dense] {spec.name}: {counts['failed']} milestone(s) failed -- "
              "rows kept in dense_eval_results.csv; fix the cause and re-run "
              "with --retry-failed", flush=True)
        return 1
    final = {}
    for row in rows:
        for key, value in row.items():
            if key in ("probe/val_top1", "ssl/total_loss", "dense/nyu_rmse") or \
                    key.startswith("pair/"):
                final[f"final_{key}"] = value
    results_io.write_summary(run_dir, {
        "case": CASE,
        "config": spec.config,
        "seed": spec.seed,
        "max_iter": args.max_iter,
        "dense_milestones_ok": counts["ok"],
        **final,
    })
    return 0


# ---------------------------------------------------------------------------- #
# Figures
# ---------------------------------------------------------------------------- #
def render_watermark_panel(args, figures_dir: str) -> List[str]:
    from common.imagenet1k import load_split, locally_available

    if not locally_available("val", os.path.abspath(args.data_dir)):
        raise RuntimeError(
            "ImageNet-1k val not present locally -- refusing to download "
            "~45 GB for a figure panel; run prepare_data.py first")
    from common.watermark import apply_watermark
    from PIL import Image

    arm = load_arms()["watermarked"]
    kwargs = dict(opacity=float(arm["DINOV3_WM_OPACITY"]),
                  modulus=int(arm["DINOV3_WM_MODULUS"]),
                  tile_px=int(arm["DINOV3_WM_TILE"]),
                  bit_capacity=int(arm["DINOV3_WM_BITS"]),
                  anchor=arm["DINOV3_WM_ANCHOR"])
    ds = load_split("val", data_dir=os.path.abspath(args.data_dir))
    n = len(ds)
    indices = [max(1, n // 7), max(1, n // 2), max(1, (5 * n) // 6)]
    samples = [(ds[i]["image"].convert("RGB"), i) for i in indices]
    pil, index = samples[0]

    def center_square(im):
        w, h = im.size if isinstance(im, Image.Image) else (im.shape[1], im.shape[0])
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        if isinstance(im, Image.Image):
            return im.crop((left, top, left + s, top + s))
        return im[top:top + s, left:left + s]

    pattern_args = dict(modulus=kwargs["modulus"], tile_px=kwargs["tile_px"],
                        bit_capacity=kwargs["bit_capacity"], anchor=kwargs["anchor"])
    alone = plotting.pattern_alone(index, pil.size[1], pil.size[0],
                                   kwargs["opacity"], **pattern_args)
    panes = [
        (center_square(pil), "clean"),
        (center_square(alone), "pattern alone (contrast-stretched)"),
        (center_square(apply_watermark(pil, index, repeat=True, **kwargs)),
         "watermarked"),
        (center_square(apply_watermark(pil, index, repeat=False, random_anchor=True,
                                       **kwargs)),
         "random control"),
    ]
    png = plotting.quad_panel_figure(
        panes,
        title=f"The planted feature at opacity {kwargs['opacity']}",
        out_path=os.path.join(figures_dir, "f1_dino_watermark.png"))

    def frame(pil, index, i):
        return dict(
            clean=center_square(pil),
            watermarked=center_square(apply_watermark(pil, index, repeat=True,
                                                      **kwargs)),
            control=center_square(apply_watermark(pil, index, repeat=False,
                                                  random_anchor=True, **kwargs)),
            pattern_wm=center_square(plotting.pattern_alone(
                index, pil.size[1], pil.size[0], kwargs["opacity"], **pattern_args)),
            pattern_ctrl=center_square(plotting.pattern_alone(
                index, pil.size[1], pil.size[0], kwargs["opacity"], repeat=False,
                random_anchor=True, **pattern_args)),
            label=f"image {i + 1} of {len(samples)}",
        )

    gif = plotting.animated_watermark_panel(
        [frame(p, idx, i) for i, (p, idx) in enumerate(samples)],
        title=f"The planted feature at DINOv3's opacity ({kwargs['opacity']})",
        out_path=os.path.join(figures_dir, "f1_dino_watermark.gif"))
    return [png, gif]


def render_figures(args) -> List[str]:
    figures_dir = os.path.join(os.path.abspath(args.results_dir), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    # The stimulus panel is the one figure that needs the dataset; the curve
    # figures need only local run files and must not die with it.
    written = []
    try:
        written.extend(render_watermark_panel(args, figures_dir))
    except Exception as e:
        print(f"[plot] watermark panel skipped (dataset unavailable: {e})")

    history = plotting.history_frame(os.path.abspath(args.results_dir))
    if history.empty:
        print("[plot] no run histories found; nothing further to render")
        return written
    panels = [dict(metric="ssl/total_loss", label="SSL total loss ↓", smooth=25),
              dict(metric="probe/val_top1", label="IN-1k probe top-1 ↑")]
    if history[history["metric"] == "dense/nyu_rmse"].empty:
        print("[plot] WARNING: no dense/nyu_rmse rows -- the crossover will lack "
              "the NYU-depth panel this case's blog figure requires (the dense "
              "tail has not produced results)")
    else:
        panels.append(dict(metric="dense/nyu_rmse",
                           label="NYU-depth RMSE ↓",
                           marker=True))
    written.append(plotting.crossover_figure(
        history, panels,
        title="DINOv3, three arms",
        xlabel="training iteration",
        out_path=os.path.join(figures_dir, "f2_dino_crossover.png")))
    written.append(plotting.pair_series_figure(
        history, rep="tokens_mean",
        title="DINOv3 pair cosine",
        xlabel="training iteration",
        out_path=os.path.join(figures_dir, "f3_dino_pair.png")))
    for path in written:
        print(f"[plot] wrote {path}")
    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    runner.add_common_arguments(parser, default_tag="")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER,
                        help="training step cap (the LR schedule horizon is the "
                             "config's and does not move with this)")
    parser.add_argument("--head-iters", type=int, default=DEFAULT_HEAD_ITERS,
                        help="dense-probe head-training cap")
    parser.add_argument("--retry-failed", action="store_true",
                        help="re-evaluate milestones whose dense rows are 'failed'")
    parser.add_argument("--extra-opts", default="",
                        help="space-separated dinov3 config overrides (smoke runs)")
    parser.add_argument("--dense-extra-opts", default="",
                        help="space-separated depth-config overrides (smoke runs)")
    args = parser.parse_args(argv)
    # datasets' Arrow cache belongs under DATA_DIR (in-process figure rendering;
    # training subprocesses get it via their env).
    os.environ.setdefault("HF_HOME", os.path.join(os.path.abspath(args.data_dir), "hf_cache"))

    if args.plot_only:
        render_figures(args)
        return 0


    arms = load_arms()
    unknown = [c for c in args.configs.split(",") if c and c not in arms]
    if unknown:
        raise SystemExit(f"unknown config(s) {unknown}; available: {sorted(arms)}")

    # Seed-major order: every completed stretch is a full arm set at one seed,
    # so an interrupted campaign still yields arm-comparable figures.
    specs = [build_spec(config, arms[config], seed, args)
             for seed in runner.parse_seeds(args.seeds)
             for config in args.configs.split(",") if config]
    if args.dry_run:
        runner.print_dry_run(specs, [g.strip() for g in args.gpus.split(",") if g.strip()])
        return 0
    to_run = []
    for spec in specs:
        if runner.should_run(args.results_dir, spec, force=args.force, rerun=args.rerun):
            if args.force or spec.name in args.rerun:
                # An explicitly forced rerun must start from scratch: with the
                # stale run dir in place, the trainer's native resume would turn
                # it into a silent (possibly zero-step) continuation.
                runner.clear_run_dir(args.results_dir, spec.name)
            to_run.append(spec)
        else:
            print(f"[run] {spec.name}: summary.json present, skipping "
                  "(--force or --rerun to redo)")

    results = runner.execute_pool(
        to_run, [g.strip() for g in args.gpus.split(",") if g.strip()],
        launch_fn=lambda spec, gpu: launch_and_tail(spec, gpu, args))
    code = runner.summarize_results(results)
    if code == 0:
        render_figures(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
