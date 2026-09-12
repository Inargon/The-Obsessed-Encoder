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
from omegaconf import OmegaConf

from additional_files.callbacks.goal_eval import GoalEvalCallback


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

    model = swm.wm.utils.load_pretrained(
        f"{args.run_name}/{args.checkpoint}"
    ).to("cuda")
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
