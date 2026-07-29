"""Milestone dense probe: the truncated NYU-depth linear probe, one process per
teacher checkpoint.

Standalone CLI over the vendored ``dinov3/eval/depth`` protocol.  The
*milestone dense probe* keeps the vendored config verbatim -- the 38.4K-iteration
LR schedule, bs 16 == the published 2x8 effective batch -- and only caps how far
the head-training loop runs (default 4800 iterations, read mid-warmup), then
reports the single no-TTA pass over the 654-image Eigen test.  The *full
protocol* is the same code path with ``--head-iters 38400 --tta``.

The vendored tree is imported, never modified; every fix lives here:

* head-iters cap -- IterBasedTrainer stops early, the LR schedule stays built for
  the config's full ``scheduler.total_iter`` (a compressed schedule would be a
  different protocol).
* dataloader worker lift + numpy>=2 float32 cast + two vendored resume defects
  (scheduler fast-forward, DDP-prefix load).
* watermarked NYU datasets (``--watermark-json``): the run's frozen watermark
  rendered on the decoded source image pre-resize, payload = dataset index, so
  watermarked runs are measured at their own operating point.

The BTS data gate (manifest counts, pair resolution, uint16-mm scale) runs at
startup, every invocation: the probe is only meaningful on the exact dataset
DINOv3's depth benchmark is defined on, and the commonly-mirrored h5 conversion
of NYU is a different dataset (different train composition, filled instead of
raw test depth), so a wrong root must hard-stop instead of silently shifting
every number.  See additional_files/fetch_nyu.py for obtaining the data.

Runs single-process: it force-fabricates its own one-rank torchrun env (fresh
port), so a parent's stale MASTER_*/RANK vars can never collide with its store.
The final stdout line ``DENSE_EVAL_RESULT {json}`` is the machine-readable
result the runner's dense tail parses.

    uv run python dinov3/additional_files/dense_eval.py \
        --ckpt <run>/train/eval/training_4999/teacher_checkpoint.pth \
        --output-dir /tmp/probe_out
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

_VENDORED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ARTIFACT_ROOT = os.path.dirname(_VENDORED_ROOT)
for _p in (_ARTIFACT_ROOT, _VENDORED_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BACKBONE_CONFIG = os.path.join(_VENDORED_ROOT, "additional_files", "configs",
                               "vitl_im1k.yaml")
DEPTH_CONFIG = os.path.join(_VENDORED_ROOT, "dinov3", "eval", "depth", "configs",
                            "config-nyu.yaml")


def default_data_root() -> str:
    return (os.environ.get("DINOV3_NYU_BTS_ROOT")
            or os.path.join(os.environ.get("DATA_DIR") or "./data", "nyu_depth_v2_bts"))


# Single-process adaptation of the vendored protocol (bs 2 x 8 GPUs); worker
# count is infra, not a protocol knob.
EFFECTIVE_BS = 16
DEPTH_NUM_WORKERS = 8

# The truncated instrument's cap (ordering readable by ~1600 head iters, the
# healthy-vs-degraded gap at full size by ~3200-4800).
DEFAULT_HEAD_ITERS = 4800

# BTS gate expectations.
NYU_TRAIN_SIZE = 24_231
NYU_TEST_SIZE = 654
NYU_DEPTH_WINDOW_M = (0.001, 10.0)
MM_PER_METER = 1000.0

RESULT_MARKER = "DENSE_EVAL_RESULT "


def watermark_kwargs_from_env(env: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """An arm's ``DINOV3_WM_*`` env set -> ``apply_watermark`` kwargs (the
    ``--watermark-json`` payload); None for the clean arm.  Decoding is the one
    shared decoder, so the probe measures at exactly the operating point the
    training dataset rendered."""
    from additional_files.wm_env import watermark_kwargs

    return watermark_kwargs(env)


def nyu_data_gate(data_root: str, *, sample_size: int = 64) -> Dict[str, Any]:
    """Hard-stop assertion that ``data_root`` is the BTS protocol dataset.

    Manifest counts (24,231 / 654), sampled pair resolution, uint16-mm depth
    scale inside the protocol window, plausible indoor means -- a silently-wrong
    root still trains and produces garbage RMSE, so it must fail here instead.
    """
    import numpy as np
    from PIL import Image

    report: Dict[str, Any] = {}
    lo, hi = NYU_DEPTH_WINDOW_M
    for fname, expected in (("nyu_train.txt", NYU_TRAIN_SIZE),
                            ("nyu_test.txt", NYU_TEST_SIZE)):
        manifest = os.path.join(data_root, fname)
        if not os.path.exists(manifest):
            raise SystemExit(
                f"dense_eval: HARD STOP -- manifest missing: {manifest}\n"
                "This root is not the BTS NYU protocol dataset; see "
                "dinov3/additional_files/fetch_nyu.py")
        with open(manifest) as f:
            lines = [ln.split() for ln in f.read().strip().split("\n")]
        if len(lines) != expected:
            raise SystemExit(f"dense_eval: HARD STOP -- {fname}: {len(lines)} entries, "
                             f"expected {expected}")
        rng = np.random.default_rng(0)
        sample = [lines[i] for i in rng.choice(len(lines), size=min(sample_size, len(lines)),
                                               replace=False)]
        means = []
        for img_rel, depth_rel, _focal in sample:
            img_path = os.path.join(data_root, img_rel.strip("/"))
            depth_path = os.path.join(data_root, depth_rel.strip("/"))
            if not (os.path.exists(img_path) and os.path.exists(depth_path)):
                raise SystemExit(f"dense_eval: HARD STOP -- {fname}: missing pair "
                                 f"{img_rel} / {depth_rel}")
            arr = np.array(Image.open(depth_path))
            if arr.dtype not in (np.uint16, np.int32):  # PIL I;16 reads as uint16/int32
                raise SystemExit(f"dense_eval: HARD STOP -- {depth_path}: decoded as "
                                 f"{arr.dtype}, expected 16-bit integer")
            valid = arr[arr > 0].astype(np.float64) / MM_PER_METER
            if valid.size == 0 or valid.min() < lo or valid.max() > hi:
                raise SystemExit(f"dense_eval: HARD STOP -- {depth_path}: depths outside "
                                 f"[{lo}, {hi}] m -- scale is wrong")
            means.append(float(valid.mean()))
        median = float(np.median(means))
        if not (0.5 < median < 6.0):
            raise SystemExit(f"dense_eval: HARD STOP -- {fname}: median mean-depth "
                             f"{median:.3f} m implausible for indoor NYU")
        report[fname] = {"count": len(lines), "n_sampled": len(sample),
                         "median_mean_depth_m": median}
    return report


def _fabricate_torchrun_env() -> None:
    """The single-process torchrun env the vendored distributed init expects.

    Forced, never setdefault'd: a parent process's fabricated MASTER_*/RANK vars
    (its store may still be bound to that port) must not leak in.
    """
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    os.environ.update({
        "TORCHELASTIC_RUN_ID": "dinov3-dense-eval",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "1",
    })


def make_watermarked_dataset_class(base_cls, watermark: Dict[str, Any]):
    """Subclass a vendored dataset so every decoded source image gets the
    watermark rendered pre-resize (before the dataset's transforms), with the
    dataset index as the key.  The render itself is common/watermark.py's -- the
    wrapper only routes (image, index) into make_source_transform's binding."""
    from common.watermark import make_source_transform
    from dinov3.data.datasets.decoders import ImageDataDecoder

    source_transform = make_source_transform(**watermark)

    class Watermarked(base_cls):
        def __init__(self, **kwargs):
            dataset = self

            class _WatermarkedDecoder(ImageDataDecoder):
                def decode(self):
                    return source_transform(super().decode(), dataset._wm_payload_index)

            super().__init__(image_decoder=_WatermarkedDecoder, **kwargs)

        # __getitem__ fetches image data for exactly one index at a time within a
        # worker, so stashing it here hands the decoder its key index.
        def get_image_data(self, index: int):
            self._wm_payload_index = index
            return super().get_image_data(index)

    Watermarked.__name__ = f"Watermarked{base_cls.__name__}"
    Watermarked.__qualname__ = Watermarked.__name__
    return Watermarked


def _patch_nyu_with_watermark(watermark: Dict[str, Any]) -> None:
    """Point the vendored dataset registry at the watermarked NYU subclass.
    Process-wide; one milestone per process, so clean runs never see it."""
    import dinov3.data.loaders as loaders

    loaders.NYU = make_watermarked_dataset_class(loaders.NYU, watermark)
    print(f"[dense_eval] watermark arm: NYU patched with {watermark}")


def _patch_depth_infra(out_dir: str, head_iters: int) -> None:
    """The wrapper-layer fixes around the vendored depth trainer, plus the
    head-iters cap that makes the truncated instrument."""
    import torch
    import dinov3.eval.depth.train as depth_train

    # The cap: stop the head-training loop early while the LR scheduler stays
    # built for config.scheduler.total_iter -- the original schedule, read
    # mid-warmup.
    vendored_trainer_init = depth_train.IterBasedTrainer.__init__

    def capped_init(self, *args, **kwargs):
        vendored_trainer_init(self, *args, **kwargs)
        if head_iters < self.total_iter:
            self.total_iter = head_iters

    depth_train.IterBasedTrainer.__init__ = capped_init
    print(f"[dense_eval] head-training capped at {head_iters} iterations "
          "(LR schedule untouched)")

    vendored_loader = depth_train.build_dataloader

    def build_dataloader(*args, **kwargs):
        kwargs["num_workers"] = max(kwargs.get("num_workers", 0), DEPTH_NUM_WORKERS)
        return vendored_loader(*args, **kwargs)

    depth_train.build_dataloader = build_dataloader
    print(f"[dense_eval] depth dataloader num_workers lifted to {DEPTH_NUM_WORKERS} "
          "(infra-only; protocol knobs unchanged)")

    # Environment fix, values untouched: under numpy>=2 scalar promotion the
    # vendored ColorAug returns float64 images (tensor * np.float64), which
    # autocast refuses to cast and conv2d rejects.  Meta's stack produced float32
    # here; restore that dtype after the vendored transforms run.
    vendored_factory = depth_train.make_depth_train_transforms_from_config

    def make_train_transforms(config):
        transforms = vendored_factory(config)

        def with_float32_images(img, label):
            img, label = transforms(img, label)
            if torch.is_tensor(img) and img.dtype == torch.float64:
                img = img.float()
            return img, label

        return with_float32_images

    depth_train.make_depth_train_transforms_from_config = make_train_transforms

    # Resume correctness (vendored defects): run_epochs restores
    # model/optimizer/global_step but rebuilds the LR scheduler at position 0, and
    # save_checkpoint strips the DDP "module." prefix that the resume load expects.
    # The runner's tail evaluates each milestone in a fresh dir (resume_iter 0,
    # both patches inert); they are kept so a standalone rerun into a populated
    # dir stays correct.
    from dinov3.eval.depth.checkpoint_utils import find_latest_checkpoint

    resume_path = find_latest_checkpoint(out_dir)
    resume_iter = 0
    if resume_path:
        blob = torch.load(resume_path, map_location="cpu")
        resume_iter = int(blob.get("iteration") or 0)
    if resume_iter > 0:
        vendored_scheduler = depth_train.build_scheduler

        def build_scheduler(*args, **kwargs):
            scheduler = vendored_scheduler(*args, **kwargs)
            for _ in range(resume_iter):
                scheduler.step()
            return scheduler

        depth_train.build_scheduler = build_scheduler

        vendored_load = depth_train.load_checkpoint

        def load_checkpoint(path):
            state_dicts, iteration = vendored_load(path)
            state_dicts["model"] = {
                k if k.startswith("module.") else "module." + k: v
                for k, v in state_dicts["model"].items()
            }
            return state_dicts, iteration

        depth_train.load_checkpoint = load_checkpoint
        print(f"[dense_eval] resume at iteration {resume_iter}: LR scheduler "
              "fast-forward + DDP-prefix load fix armed")


def run_probe(
    *,
    ckpt: str,
    output_dir: str,
    head_iters: int = DEFAULT_HEAD_ITERS,
    tta: bool = False,
    watermark: Optional[Dict[str, Any]] = None,
    data_root: Optional[str] = None,
    backbone_config: str = BACKBONE_CONFIG,
    extra_opts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """One checkpoint through the probe; returns the JSON-able result dict."""
    data_root = data_root or default_data_root()
    if not os.path.exists(ckpt):
        raise SystemExit(f"dense_eval: HARD STOP -- checkpoint missing: {ckpt}")
    gate = nyu_data_gate(data_root)
    print(f"[dense_eval] BTS data gate passed: {gate}")
    # The protocol's head training is otherwise unseeded: the same checkpoint
    # re-probed across sessions drifted ~10% RMSE. One fixed seed makes
    # milestone rows comparable; the backbone under probe is untouched.
    import random

    import numpy as np
    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    os.makedirs(output_dir, exist_ok=True)

    _fabricate_torchrun_env()
    if watermark is not None:
        _patch_nyu_with_watermark(watermark)
    _patch_depth_infra(output_dir, head_iters)

    argv = [
        f"config={DEPTH_CONFIG}",
        f"output_dir={output_dir}",
        f"bs={EFFECTIVE_BS}",
        "n_gpus=1",
        f"datasets.root={data_root}",
        f"model.config_file={backbone_config}",
        f"model.pretrained_weights={ckpt}",
        f"eval.use_tta={str(bool(tta)).lower()}",
        *(extra_opts or []),
    ]
    print(f"[dense_eval] launching vendored depth eval: {argv}")

    from dinov3.eval.depth.run import benchmark_launcher
    from dinov3.eval.helpers import cli_parser
    from dinov3.run.init import job_context

    started = time.time()
    eval_args = cli_parser(argv)
    with job_context(output_dir=output_dir):
        results = benchmark_launcher(eval_args=eval_args)
    wall = time.time() - started

    # benchmark_launcher's per-image lists (written before its nanmean reduction)
    # are the test-pass size witness.
    per_image_path = os.path.join(output_dir, "NYU", "results.json")
    n_test_images = None
    if os.path.exists(per_image_path):
        with open(per_image_path) as f:
            per_image = json.load(f)
        n_test_images = max((len(v) for v in per_image.values()), default=None)

    missing = [k for k in ("NYU_rmse", "NYU_abs_rel", "NYU_a1") if k not in results]
    if missing:
        raise SystemExit(f"dense_eval: HARD STOP -- vendored eval returned no {missing}; "
                         f"got keys {sorted(results)}")
    return {
        "rmse": float(results["NYU_rmse"]),
        "abs_rel": float(results["NYU_abs_rel"]),
        "a1": float(results["NYU_a1"]),
        "n_test_images": n_test_images,
        "head_iters": int(head_iters),
        "tta": bool(tta),
        "wall_clock_s": round(wall, 1),
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True,
                        help="teacher checkpoint (.pth, the do_test milestone dump)")
    parser.add_argument("--output-dir", required=True,
                        help="scratch dir for the vendored run (logs, head checkpoints)")
    parser.add_argument("--head-iters", type=int, default=DEFAULT_HEAD_ITERS,
                        help="head-training cap; 38400 == the full protocol")
    parser.add_argument("--tta", action="store_true",
                        help="test-time augmentation on the final pass (full protocol)")
    parser.add_argument("--watermark-json", default=None,
                        help="apply_watermark kwargs as JSON (absent = clean inputs)")
    parser.add_argument("--data-root", default=None,
                        help="BTS NYU root (dir holding nyu_train.txt / nyu_test.txt); "
                             "default DATA_DIR/nyu_depth_v2_bts")
    parser.add_argument("--backbone-config", default=BACKBONE_CONFIG,
                        help="SSL config yaml the backbone is built from")
    parser.add_argument("--extra-opts", default="",
                        help="space-separated depth-config key=value overrides (smoke)")
    args = parser.parse_args(argv)

    watermark = json.loads(args.watermark_json) if args.watermark_json else None
    result = run_probe(
        ckpt=args.ckpt,
        output_dir=args.output_dir,
        head_iters=args.head_iters,
        tta=args.tta,
        watermark=watermark,
        data_root=args.data_root,
        backbone_config=args.backbone_config,
        extra_opts=args.extra_opts.split() or None,
    )
    # The machine-readable contract with the runner's dense tail: exactly one
    # marker line, last on stdout.
    print(RESULT_MARKER + json.dumps(result))


if __name__ == "__main__":
    main()
