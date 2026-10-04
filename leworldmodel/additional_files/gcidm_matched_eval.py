#!/usr/bin/env python3
"""Run official GC-IDM evaluation with this project's fixed-group sampler."""

from __future__ import annotations

import importlib
import functools
import inspect
import sys

import numpy as np


def install_world_evaluate_compat(world_class) -> bool:
    """Allow the modern GC-IDM evaluator to call the historical World API."""
    evaluate = world_class.evaluate
    if "dataset" in inspect.signature(evaluate).parameters:
        return False

    @functools.wraps(evaluate)
    def compatible_evaluate(self, *args, **kwargs):
        kwargs.pop("dataset", None)
        return evaluate(self, *args, **kwargs)

    world_class.evaluate = compatible_evaluate
    return True


def sample_matched_eval_episodes(
    dataset,
    num_eval: int,
    goal_offset: int,
    seed: int = 42,
    train_split: float = 1.0,
    split_seed: int = 42,
):
    """Match GoalEvalCallback's exact fixed (episode, start) selection."""
    if train_split != 1.0:
        raise ValueError("matched pilot requires the full-dataset protocol")
    del split_seed
    episode_idx = None
    for column in ("episode_idx", "ep_idx"):
        try:
            episode_idx = np.asarray(dataset.get_col_data(column))
            break
        except Exception:
            continue
    if episode_idx is None:
        raise ValueError("dataset has neither episode_idx nor ep_idx")
    step_idx = np.asarray(dataset.get_col_data("step_idx"))
    last = {
        episode: step_idx[episode_idx == episode].max()
        for episode in np.unique(episode_idx)
    }
    max_start = np.asarray(
        [last[episode] - goal_offset for episode in episode_idx]
    )
    valid = np.nonzero(step_idx <= max_start)[0]
    if len(valid) < num_eval:
        raise RuntimeError(
            f"only {len(valid)} valid starts for num_eval={num_eval}"
        )
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(valid, size=num_eval, replace=False))
    # The official evaluator calls ``.tolist()`` when passing both arrays into
    # World.evaluate, so preserve its ndarray interface here.
    return episode_idx[rows], step_idx[rows]


def main() -> None:
    official = importlib.import_module("eval_idm")
    official.sample_eval_episodes = sample_matched_eval_episodes
    world_class = official.swm.World
    if install_world_evaluate_compat(world_class):
        print("GCIDM_HISTORICAL_WORLD_EVALUATE_COMPAT_ACTIVE")
    official.main()


if __name__ == "__main__":
    main()
