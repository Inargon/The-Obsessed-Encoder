#!/usr/bin/env python3
"""Apply the repository's exact tagged-PushT intervention to upstream INTACT.

This is an adapter, not a fork. Training tags decoded uint8 clips before
INTACT's preprocessing; evaluation tags both observation and goal before the
official normalization and resize.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


PINNED_INTACT_COMMIT = "653ee22266a34a74efca21b0b03dfc1fd6fa37ff"


def verify_upstream(root: Path, expected_commit: str) -> str:
    required = ("train.py", "eval.py", "config/train/intact_goal.yaml")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"INTACT root is missing: {', '.join(missing)}")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if commit != expected_commit:
        raise RuntimeError(
            f"INTACT commit mismatch: expected {expected_commit}, found {commit}"
        )
    dirty = subprocess.check_output(
        ["git", "status", "--short"], cwd=root, text=True
    ).strip()
    if dirty:
        raise RuntimeError(
            "INTACT checkout is dirty; keep upstream untouched and put changes "
            "in this adapter:\n" + dirty
        )
    return commit


def load_upstream(root: Path, filename: str, module_name: str):
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(module_name, root / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {root / filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def install_training_tag(module, *, mode: str, size: int, seed: int) -> None:
    from pixel_tag import PixelTag, attach_pixel_tag

    original = module.swm.data.load_dataset

    def load_tagged_dataset(*args, **kwargs):
        dataset = original(*args, **kwargs)
        return attach_pixel_tag(dataset, PixelTag(mode=mode, size=size, seed=seed))

    module.swm.data.load_dataset = load_tagged_dataset


def install_evaluation_tag(module, *, mode: str, size: int, seed: int) -> None:
    from additional_files.callbacks.goal_eval import EvalTagStamp
    from pixel_tag import PixelTag
    from torchvision.transforms import v2 as transforms

    original = module.img_transform

    def tagged_img_transform(cfg):
        # Upstream calls this once for pixels and once for goal. Each stream
        # gets an independent counter but the exact same video-mode color.
        base = original(cfg)
        tag = PixelTag(mode=mode, size=size, seed=seed)
        return transforms.Compose([EvalTagStamp(tag), base])

    module.img_transform = tagged_img_transform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "eval"))
    parser.add_argument("--intact-root", type=Path, required=True)
    parser.add_argument("--expected-commit", default=PINNED_INTACT_COMMIT)
    parser.add_argument("--tag-mode", choices=("video", "frame"), default="video")
    parser.add_argument("--tag-size", type=int, default=5)
    parser.add_argument("--tag-seed", type=int, default=0)
    parser.add_argument("hydra_overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.intact_root.resolve()
    commit = verify_upstream(root, args.expected_commit)
    if args.tag_size < 1:
        parser.error("tag-size must be positive")

    overrides = list(args.hydra_overrides)
    if overrides[:1] == ["--"]:
        overrides = overrides[1:]
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    if args.phase == "train":
        upstream = load_upstream(root, "train.py", "intact_upstream_train")
        install_training_tag(
            upstream, mode=args.tag_mode, size=args.tag_size, seed=args.tag_seed
        )
    else:
        upstream = load_upstream(root, "eval.py", "intact_upstream_eval")
        install_evaluation_tag(
            upstream, mode=args.tag_mode, size=args.tag_size, seed=args.tag_seed
        )

    print(
        "INTACT_TAGGED_ADAPTER "
        f"phase={args.phase} commit={commit} mode={args.tag_mode} "
        f"size={args.tag_size} seed={args.tag_seed}",
        flush=True,
    )
    sys.argv = [str(root / f"{args.phase}.py"), *overrides]
    upstream.run()


if __name__ == "__main__":
    main()
