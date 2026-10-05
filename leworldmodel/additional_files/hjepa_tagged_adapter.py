#!/usr/bin/env python3
"""Run pinned upstream H-JEPA on clean or predictably tagged PushT.

The adapter deliberately leaves the upstream checkout untouched.  Training
monkey-patches only the HDF5Dataset constructor so the existing PixelTag seam
stamps raw uint8 frames before H-JEPA's official image preprocessing.  Eval
reuses the official model, solver and macro-action policy on a matched fixed
set of PushT initial/goal states.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "leworldmodel"))

from additional_files.callbacks.goal_eval import eval_transforms
from additional_files.pixel_tag import PixelTag, attach_pixel_tag


PINNED_HJEPA_COMMIT = "fd4f7de927daa7831a6d51e95e0959a813c13901"


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _activate_upstream(root: Path) -> None:
    root = root.resolve()
    required = [root / "train.py", root / "eval.py", root / "hamiltonian_jepa.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing H-JEPA source files: " + ", ".join(missing))
    sys.path.insert(0, str(root))


def _install_training_tag(seed: int, size: int) -> None:
    original = swm.data.HDF5Dataset

    def tagged_dataset(*args, **kwargs):
        dataset = original(*args, **kwargs)
        return attach_pixel_tag(
            dataset, PixelTag(mode="video", size=size, seed=seed)
        )

    swm.data.HDF5Dataset = tagged_dataset


def train(args: argparse.Namespace, overrides: list[str]) -> None:
    _activate_upstream(args.upstream)
    if args.condition == "tagged":
        _install_training_tag(args.tag_seed, args.tag_size)
    upstream_train = _load_file("hjepa_upstream_train", args.upstream / "train.py")
    sys.argv = [str(args.upstream / "train.py"), *overrides]
    print(
        "HJEPA_TAGGED_ADAPTER phase=train "
        f"condition={args.condition} tag_seed={args.tag_seed} tag_size={args.tag_size}"
    )
    upstream_train.run()


def _fixed_rows(dataset, seed: int, total: int, offset: int, count: int, goal_offset: int):
    for col in ("episode_idx", "ep_idx"):
        try:
            episode_idx = np.asarray(dataset.get_col_data(col))
            break
        except Exception:
            episode_idx = None
    if episode_idx is None:
        raise ValueError("dataset has neither episode_idx nor ep_idx")
    step_idx = np.asarray(dataset.get_col_data("step_idx"))
    episodes = np.unique(episode_idx)
    lengths = np.array([step_idx[episode_idx == ep].max() + 1 for ep in episodes])
    max_start = dict(zip(episodes, lengths - goal_offset - 1))
    valid = np.nonzero(step_idx <= np.array([max_start[ep] for ep in episode_idx]))[0]
    if total > len(valid):
        raise ValueError(f"requested {total} fixed rows from only {len(valid)} valid rows")
    rng = np.random.default_rng(seed)
    # Match the released H-JEPA evaluator exactly, including its excluded last
    # valid position, so published and tagged arms share the same fixed group.
    selected = np.sort(valid[rng.choice(len(valid) - 1, size=total, replace=False)])
    selected = selected[offset : offset + count]
    return (
        np.asarray(episode_idx[selected]).tolist(),
        np.asarray(step_idx[selected]).tolist(),
        selected.tolist(),
    )


def evaluate(args: argparse.Namespace) -> None:
    _activate_upstream(args.upstream)
    official_eval = _load_file("hjepa_upstream_eval", args.upstream / "eval.py")
    cfg = OmegaConf.load(args.upstream / "config/eval/pusht.yaml")
    cfg.solver = OmegaConf.load(args.upstream / "config/eval/solver/cem.yaml")
    cfg.pop("defaults", None)
    cfg.seed = args.seed
    cfg.eval.num_eval = args.num_eval
    cfg.eval.num_eval_total = args.num_eval_total
    cfg.eval.batch_offset = args.batch_offset
    cfg.world.num_envs = args.num_eval

    checkpoint = args.checkpoint.expanduser().resolve()
    model = torch.load(checkpoint, weights_only=False, map_location="cpu").cuda().eval()
    model.requires_grad_(False)
    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name, keys_to_cache=cfg.dataset.keys_to_cache
    )
    process = {}
    for col in cfg.dataset.keys_to_cache:
        data = dataset.get_col_data(col)
        process[col] = preprocessing.StandardScaler().fit(
            data[~np.isnan(data).any(axis=1)]
        )
        if col != "action":
            process[f"goal_{col}"] = process[col]

    tag = None
    if args.input_condition == "tagged":
        tag = PixelTag(mode="video", size=args.tag_size, seed=args.tag_seed)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(**cfg.plan_config),
        process=process,
        transform=eval_transforms(int(cfg.eval.img_size), tag),
    )
    policy = official_eval.MacroPixelHistoryPolicy(
        policy,
        history_len=cfg.plan_config.history_len,
        action_block=cfg.plan_config.action_block,
    )
    episodes, starts, rows = _fixed_rows(
        dataset,
        args.seed,
        args.num_eval_total,
        args.batch_offset,
        args.num_eval,
        int(cfg.eval.goal_offset_steps),
    )
    world = swm.World(
        **cfg.world,
        max_episode_steps=2 * cfg.eval.eval_budget,
        image_shape=(224, 224),
    )
    world.set_policy(policy)
    started = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=starts,
        goal_offset=int(cfg.eval.goal_offset_steps),
        eval_budget=int(cfg.eval.eval_budget),
        episodes_idx=episodes,
        callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
        video=str(args.video_dir) if args.video_dir else None,
    )
    successes = np.asarray(metrics["episode_successes"], dtype=bool)
    result = {
        "method": "official H-JEPA",
        "checkpoint": str(checkpoint),
        "train_condition": args.train_condition,
        "input_condition": args.input_condition,
        "seed": args.seed,
        "num_eval": args.num_eval,
        "num_eval_total": args.num_eval_total,
        "batch_offset": args.batch_offset,
        "dataset_rows": rows,
        "success_rate": float(successes.mean()),
        "success_rate_percent": float(metrics["success_rate"]),
        "successful_episode_indices": np.flatnonzero(successes).tolist(),
        "evaluation_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(
        "HJEPA_TAGGED_EVAL_COMPLETE "
        f"train={args.train_condition} input={args.input_condition}"
    )


def preflight(args: argparse.Namespace) -> None:
    _activate_upstream(args.upstream)
    import importlib.metadata as metadata

    info = {
        "upstream": str(args.upstream.resolve()),
        "stable-worldmodel": metadata.version("stable-worldmodel"),
        "stable-pretraining": metadata.version("stable-pretraining"),
        "torch": torch.__version__,
        "has_hdf5_dataset": hasattr(swm.data, "HDF5Dataset"),
        "has_dataset_evaluate": "dataset" in __import__("inspect").signature(
            swm.World.evaluate
        ).parameters,
    }
    print(json.dumps(info, indent=2))
    if not info["has_hdf5_dataset"] or not info["has_dataset_evaluate"]:
        raise RuntimeError("modern stable-worldmodel runtime is required")
    print("HJEPA_TAGGED_RUNTIME_PREFLIGHT_PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    sub = parser.add_subparsers(dest="phase", required=True)

    p = sub.add_parser("preflight")
    p.set_defaults(func=preflight)

    p = sub.add_parser("train")
    p.add_argument("--condition", choices=("clean", "tagged"), required=True)
    p.add_argument("--tag-seed", type=int, default=0)
    p.add_argument("--tag-size", type=int, default=5)
    p.set_defaults(func=train)

    p = sub.add_parser("eval")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-condition", choices=("clean", "tagged"), required=True)
    p.add_argument("--input-condition", choices=("clean", "tagged"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--video-dir", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument("--num-eval-total", type=int, default=500)
    p.add_argument("--batch-offset", type=int, default=0)
    p.add_argument("--tag-seed", type=int, default=0)
    p.add_argument("--tag-size", type=int, default=5)
    p.set_defaults(func=evaluate)

    args, rest = parser.parse_known_args()
    if args.phase == "train":
        if rest[:1] == ["--"]:
            rest = rest[1:]
        args.func(args, rest)
    elif rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    else:
        args.func(args)


if __name__ == "__main__":
    main()
