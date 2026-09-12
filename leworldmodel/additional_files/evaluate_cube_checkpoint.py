#!/usr/bin/env python3
"""Evaluate one frozen Cube checkpoint with the training-time Cube protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import stable_worldmodel as swm
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from additional_files.callbacks.goal_eval import GoalEvalCallback


def load_inference_model(run_name: str, checkpoint: str):
    """Load the JEPA core while dropping training-only control heads.

    Control objectives are attached to the model during training and therefore
    appear in aligned checkpoints, but they are not part of the model config and
    are not used by CEM inference.  Keep strict loading for the remaining keys so
    a genuine architecture mismatch still fails loudly.
    """
    root = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))
    checkpoint_dir = root / "checkpoints" / run_name
    with (checkpoint_dir / "config.json").open() as handle:
        config = json.load(handle)
    state_dict = torch.load(
        checkpoint_dir / checkpoint, map_location="cpu", weights_only=True
    )
    auxiliary_prefixes = ("control_objective.",)
    stripped = [
        key for key in state_dict if key.startswith(auxiliary_prefixes)
    ]
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(auxiliary_prefixes)
    }
    model = instantiate(config)
    model.load_state_dict(state_dict, strict=True)
    print(f"loaded {run_name}/{checkpoint}; stripped {len(stripped)} training-only keys")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="weights_step_10000.pt")
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tagged", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = load_inference_model(args.run_name, args.checkpoint).to("cuda")
    model.eval().requires_grad_(False)
    model.interpolate_pos_encoding = True

    eval_cfg = {
        "config": "cube",
        "num_eval": args.num_eval,
        "dataset_name": args.dataset,
    }
    if args.tagged:
        eval_cfg.update(tag_mode="video", tag_size=5, tag_seed=args.seed)

    callback = GoalEvalCallback(
        OmegaConf.create(eval_cfg),
        seed=args.seed,
        default_dataset=args.dataset,
    )
    callback._setup(model)
    with torch.no_grad():
        metrics = callback._world.evaluate(
            dataset=callback._dataset,
            episodes_idx=callback._episodes,
            start_steps=callback._starts,
            goal_offset=callback.cfg.eval.goal_offset_steps,
            eval_budget=callback.cfg.eval.eval_budget,
            callables=OmegaConf.to_container(
                callback.cfg.eval.get("callables"), resolve=True
            ),
        )

    result = {
        "run_name": args.run_name,
        "checkpoint": args.checkpoint,
        "tagged": args.tagged,
        "num_eval": args.num_eval,
        "seed": args.seed,
        "success_rate": float(metrics["success_rate"]) / 100.0,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
