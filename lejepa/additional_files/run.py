"""Grid runner + figure factory for the LeJEPA example.

One run = one printed, human-runnable ``python lejepa/lejepa_minimal.py +key=value ...``
subprocess.  Results land under ``RESULTS_DIR/<config>_seed<k>/`` (``metrics.jsonl``
mirror, ``samples.png``, ``summary.json`` done-marker); completed runs are skipped on
re-invocation.  ``--plot-only`` renders the example's figure set from the local run
files without launching anything.

    uv run python lejepa/additional_files/run.py --seeds 3 --gpus 0
    uv run python lejepa/additional_files/run.py --plot-only
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import yaml

ARTIFACT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ARTIFACT_ROOT not in sys.path:
    sys.path.insert(0, ARTIFACT_ROOT)

from common import plotting, results_io, runner  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(ARTIFACT_ROOT, "lejepa", "lejepa_minimal.py")
CONFIGS_YAML = os.path.join(HERE, "configs.yaml")
CASE = "lejepa"


def load_configs() -> Dict[str, Dict]:
    with open(CONFIGS_YAML) as f:
        return yaml.safe_load(f)


def _format_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_spec(config: str, overrides: Dict, seed: int,
               args: argparse.Namespace) -> runner.RunSpec:
    if getattr(args, "max_steps", None):
        overrides = dict(overrides, max_steps=args.max_steps)
    name = f"{config}_seed{seed}"
    run_dir = runner.run_dir_for(args.results_dir, name)
    data_dir = os.path.abspath(args.data_dir)
    command = [sys.executable, TRAIN_SCRIPT]
    command += [f"+{key}={_format_value(value)}" for key, value in overrides.items()]
    command += [
        f"+seed={seed}",
        f"+run_dir={run_dir}",
        f"+data_dir={data_dir}",
        f"+wandb_name={CASE}_{name}",
    ]
    if args.tag:
        command.append(f"+wandb_group={args.tag}")
    # Keep hydra's own output inside the run directory (everything the artifact
    # writes lives under RESULTS_DIR).
    command.append(f"hydra.run.dir={os.path.join(run_dir, 'hydra')}")
    env = {
        "HF_HOME": os.path.join(data_dir, "hf_cache"),
        "WANDB_DIR": run_dir,
        # The recipe's training peak sits near a 48 GB card's capacity; the eval
        # passes interleave allocations into the cache and the default allocator
        # then fragments (a mid-run backward OOMs hunting for a contiguous
        # block). Expandable segments remove the fragmentation failure mode;
        # allocator config only, no effect on the math.
        # Deliberately the "deprecated" name: torch 2.9.1 warns for it but
        # does not yet honor the replacement PYTORCH_ALLOC_CONF for
        # expandable segments (verified via memory._snapshot).
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    return runner.RunSpec(name=name, config=config, seed=seed, command=command,
                          env=env, cwd=ARTIFACT_ROOT)


def write_summary(spec: runner.RunSpec, results_dir: str) -> None:
    run_dir = runner.run_dir_for(results_dir, spec.name)
    rows = results_io.read_metrics(run_dir)
    if not rows:
        raise RuntimeError(f"{spec.name}: training exited 0 but wrote no metrics")
    final: Dict[str, float] = {}
    for row in rows:
        for key in ("test/acc", "train/lejepa"):
            if key in row:
                final[f"final_{key}"] = row[key]
        for key, value in row.items():
            if key.startswith("pair/"):
                final[f"final_{key}"] = value
    results_io.write_summary(run_dir, {
        "case": CASE,
        "config": spec.config,
        "seed": spec.seed,
        "steps": rows[-1].get("step"),
        **final,
    })


# ---------------------------------------------------------------------------- #
# Figures -- the case's blog figure set, rendered from local JSONLs only
# ---------------------------------------------------------------------------- #
def _sample_source_images(data_dir: str, k: int = 3):
    """k spread-out val images, deterministic picks; the first is the one the
    static panel has always used."""
    from common.imagenet1k import load_split, locally_available

    if not locally_available("val", os.path.abspath(data_dir)):
        raise RuntimeError(
            "ImageNet-1k val not present locally -- refusing to download "
            "~45 GB for a figure panel; run prepare_data.py first")

    ds = load_split("val", data_dir=os.path.abspath(data_dir))
    n = len(ds)
    indices = [max(1, n // 7), max(1, n // 2), max(1, (5 * n) // 6)][:k]
    return [(ds[i]["image"].convert("RGB"), i) for i in indices]


def _center_square(im):
    from PIL import Image

    w, h = im.size if not hasattr(im, "shape") else (im.shape[1], im.shape[0])
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    if isinstance(im, Image.Image):
        return im.crop((left, top, left + s, top + s))
    return im[top:top + s, left:left + s]


def render_watermark_panel(args, figures_dir: str) -> List[str]:
    """Watermarked/control image pairs at this case's opacity, including a
    looping GIF over several sources. Renderer-only; no runs are needed."""
    from common.watermark import apply_watermark

    wm = load_configs()["watermarked"]
    kwargs = dict(opacity=wm["watermark_opacity"], modulus=wm["watermark_modulus"],
                  tile_px=wm["watermark_tile"], bit_capacity=wm["watermark_bits"],
                  anchor=wm["origin_anchor"])
    samples = _sample_source_images(args.data_dir)
    pil, index = samples[0]
    panes = [
        (_center_square(apply_watermark(pil, index, repeat=True, **kwargs)),
         "watermarked"),
        (_center_square(apply_watermark(pil, index, repeat=False,
                                        random_anchor=wm["random_anchor"], **kwargs)),
         "random control"),
    ]
    png = plotting.image_pair_figure(
        panes,
        title=f"The planted feature at opacity {kwargs['opacity']}",
        out_path=os.path.join(figures_dir, "f4_lejepa_watermark.png"))
    gif = plotting.animated_image_pair_panel(
        [_panel_frame(pil, index, wm, kwargs) for pil, index in samples],
        title=f"The planted feature at LeJEPA's opacity ({kwargs['opacity']})",
        out_path=os.path.join(figures_dir, "f4_lejepa_watermark.gif"))
    return [png, gif]


def _panel_frame(pil, index: int, wm: dict, kwargs: dict) -> dict:
    from common.watermark import apply_watermark

    return dict(
        watermarked=_center_square(apply_watermark(pil, index, repeat=True, **kwargs)),
        control=_center_square(apply_watermark(pil, index, repeat=False,
                                               random_anchor=wm["random_anchor"],
                                               **kwargs)),
    )


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
        print("[plot] no run histories found; rendered the watermark panel only")
        return written
    written.append(plotting.crossover_figure(
        history,
        [dict(metric="train/lejepa", label="LeJEPA loss", smooth=250),
         dict(metric="test/acc", label="online probe top-1")],
        title="LeJEPA, three arms",
        xlabel="training step",
        out_path=os.path.join(figures_dir, "f5_lejepa_crossover.png")))
    written.append(plotting.pair_series_figure(
        history, rep="backbone",
        title="LeJEPA pair cosine",
        xlabel="training step",
        out_path=os.path.join(figures_dir, "f6_lejepa_pair.png")))
    for path in written:
        print(f"[plot] wrote {path}")
    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    runner.add_common_arguments(parser, default_tag="")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="override the configs' step budget (shakeout runs)")
    args = parser.parse_args(argv)
    # datasets' Arrow cache belongs under DATA_DIR (both for in-process figure
    # rendering here and for the training subprocesses, which get it via env).
    os.environ.setdefault("HF_HOME", os.path.join(os.path.abspath(args.data_dir), "hf_cache"))

    if args.plot_only:
        render_figures(args)
        return 0

    configs = load_configs()
    unknown = [c for c in args.configs.split(",") if c and c not in configs]
    if unknown:
        raise SystemExit(f"unknown config(s) {unknown}; available: {sorted(configs)}")

    # Seed-major order: every completed stretch is a full arm set at one seed,
    # so an interrupted campaign still yields arm-comparable figures.
    specs = [build_spec(config, configs[config], seed, args)
             for seed in runner.parse_seeds(args.seeds)
             for config in args.configs.split(",") if config]
    if args.dry_run:
        runner.print_dry_run(specs, [g.strip() for g in args.gpus.split(",") if g.strip()])
        return 0
    to_run = []
    for spec in specs:
        if runner.should_run(args.results_dir, spec, force=args.force, rerun=args.rerun):
            # LeJEPA has no resume: every launch is from scratch, so a leftover
            # partial run directory (appended metrics.jsonl, stale samples.png)
            # must not blend into the new run's mirror.
            runner.clear_run_dir(args.results_dir, spec.name)
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
