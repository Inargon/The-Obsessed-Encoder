#!/usr/bin/env python3
"""Apply the exact episode-constant PushT tag to pinned upstream AC-MTM."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from additional_files.intact_tagged_adapter import (
    install_evaluation_tag,
    load_upstream,
    run_upstream,
)


PINNED_AC_MTM_COMMIT = "43c96f66bb12c95208be68b0858ee658c7e1d363"


def verify_upstream(root: Path, expected_commit: str) -> str:
    required = (
        "train.py",
        "eval.py",
        "config/train/lewm_masked_action_nce.yaml",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"AC-MTM root is missing: {', '.join(missing)}")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if commit != expected_commit:
        raise RuntimeError(
            f"AC-MTM commit mismatch: expected {expected_commit}, found {commit}"
        )
    dirty = subprocess.check_output(
        ["git", "status", "--short"], cwd=root, text=True
    ).strip()
    if dirty:
        raise RuntimeError("AC-MTM checkout must remain clean:\n" + dirty)
    return commit


def install_training_tag(module, *, mode: str, size: int, seed: int) -> None:
    """Wrap AC-MTM's direct HDF5Dataset constructor without editing upstream."""
    from pixel_tag import PixelTag, attach_pixel_tag

    original = module.swm.data.HDF5Dataset

    def tagged_dataset(*args, **kwargs):
        dataset = original(*args, **kwargs)
        return attach_pixel_tag(dataset, PixelTag(mode=mode, size=size, seed=seed))

    module.swm.data.HDF5Dataset = tagged_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "eval"))
    parser.add_argument("--ac-root", type=Path, required=True)
    parser.add_argument("--expected-commit", default=PINNED_AC_MTM_COMMIT)
    parser.add_argument("--tag-mode", choices=("video", "frame"), default="video")
    parser.add_argument("--tag-size", type=int, default=5)
    parser.add_argument("--tag-seed", type=int, default=0)
    parser.add_argument("hydra_overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    root = args.ac_root.resolve()
    commit = verify_upstream(root, args.expected_commit)
    overrides = list(args.hydra_overrides)
    if overrides[:1] == ["--"]:
        overrides = overrides[1:]
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    # Upstream imports sibling modules by their top-level names.
    sys.path.insert(0, str(root))
    if args.phase == "train":
        upstream = load_upstream(root, "train.py", "ac_mtm_upstream_train")
        install_training_tag(
            upstream, mode=args.tag_mode, size=args.tag_size, seed=args.tag_seed
        )
    else:
        upstream = load_upstream(root, "eval.py", "ac_mtm_upstream_eval")
        install_evaluation_tag(
            upstream, mode=args.tag_mode, size=args.tag_size, seed=args.tag_seed
        )

    print(
        "AC_MTM_TAGGED_ADAPTER "
        f"phase={args.phase} commit={commit} mode={args.tag_mode} "
        f"size={args.tag_size} seed={args.tag_seed}",
        flush=True,
    )
    run_upstream(root, args.phase, upstream, overrides)


if __name__ == "__main__":
    main()
