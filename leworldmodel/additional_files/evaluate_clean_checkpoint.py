#!/usr/bin/env python3
"""Evaluate a frozen clean-task checkpoint with one matched protocol.

This is the standalone counterpart of ``GoalEvalCallback``.  It intentionally
uses the same task configs, fixed-row sampler, preprocessing, CEM planner, and
success-rate convention as training-time evaluation.
"""

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


TASK_DEFAULTS = {
    "pusht": "pusht_expert_train.h5",
    "cube": "ogbench/cube_single_expert.h5",
    "tworoom": "tworoom",
}


def load_inference_model(run_name: str, checkpoint: str):
    """Load the JEPA core and discard heads used only during training."""
    root = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))
    checkpoint_dir = root / "checkpoints" / run_name
    with (checkpoint_dir / "config.json").open() as handle:
        config = json.load(handle)
    state_dict = torch.load(
        checkpoint_dir / checkpoint, map_location="cpu", weights_only=True
    )
    auxiliary_prefixes = ("control_objective.",)
    stripped = [key for key in state_dict if key.startswith(auxiliary_prefixes)]
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(auxiliary_prefixes)
    }
    model = instantiate(config)
    model.load_state_dict(state_dict, strict=True)
    print(
        f"loaded {run_name}/{checkpoint}; "
        f"stripped {len(stripped)} training-only keys"
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASK_DEFAULTS), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="weights_epoch_10.pt")
    parser.add_argument("--dataset")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset or TASK_DEFAULTS[args.task]

    model = load_inference_model(args.run_name, args.checkpoint).to("cuda")
    model.eval().requires_grad_(False)
    model.interpolate_pos_encoding = True

    callback = GoalEvalCallback(
        OmegaConf.create(
            {
                "config": args.task,
                "num_eval": args.num_eval,
                "dataset_name": dataset,
            }
        ),
        seed=args.seed,
        default_dataset=dataset,
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
        "task": args.task,
        "condition": "clean",
        "run_name": args.run_name,
        "checkpoint": args.checkpoint,
        "dataset": dataset,
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
