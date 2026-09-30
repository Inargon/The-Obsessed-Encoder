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
from typing import Sequence


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


def split_hydra_invocation(
    phase: str, overrides: Sequence[str]
) -> tuple[str, list[str]]:
    """Separate Hydra's config-name switch from configuration overrides."""
    config_name = "intact_goal" if phase == "train" else "pusht"
    remaining: list[str] = []
    index = 0
    while index < len(overrides):
        value = overrides[index]
        if value.startswith("--config-name="):
            config_name = value.split("=", 1)[1]
        elif value == "--config-name":
            index += 1
            if index >= len(overrides):
                raise ValueError("--config-name requires a value")
            config_name = overrides[index]
        elif value.startswith("--"):
            raise ValueError(f"Unsupported Hydra launcher flag: {value}")
        else:
            remaining.append(value)
        index += 1
    if not config_name:
        raise ValueError("Hydra config name must not be empty")
    return config_name, remaining


def run_upstream(root: Path, phase: str, upstream, overrides: Sequence[str]) -> None:
    """Compose from the upstream absolute config directory and run its task.

    Loading ``train.py`` or ``eval.py`` as an adapter module changes the module
    name Hydra uses to resolve ``./config/...``.  Calling the decorated entry
    point would therefore reinterpret the filesystem-relative path as a Python
    package path.  Absolute config composition preserves the upstream config
    tree while the ``__wrapped__`` call retains the patched module globals.
    """
    import hydra

    config_name, config_overrides = split_hydra_invocation(phase, overrides)
    config_dir = (root / "config" / phase).resolve()
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Hydra config directory not found: {config_dir}")
    task = getattr(upstream.run, "__wrapped__", None)
    if task is None:
        raise RuntimeError("Upstream Hydra entry point has no __wrapped__ task")
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir),
        job_name=f"intact_tagged_{phase}",
    ):
        cfg = hydra.compose(config_name=config_name, overrides=config_overrides)
        task(cfg)


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
    run_upstream(root, args.phase, upstream, overrides)


if __name__ == "__main__":
    main()
