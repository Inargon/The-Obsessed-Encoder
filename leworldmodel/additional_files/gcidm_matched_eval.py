#!/usr/bin/env python3
"""Run official GC-IDM evaluation with this project's fixed-group sampler."""

from __future__ import annotations

import importlib
import functools
import inspect
import sys
from copy import deepcopy

import numpy as np
import torch


def _extract_init_goal(dataset, episodes_idx, start_steps, goal_offset):
    ep_idx_arr = np.asarray(episodes_idx)
    start_arr = np.asarray(start_steps)
    data = dataset.load_chunk(
        ep_idx_arr, start_arr, start_arr + goal_offset + 1
    )
    init_lists: dict[str, list] = {}
    goal_lists: dict[str, list] = {}
    for episode in data:
        for column in dataset.column_names:
            if column.startswith("goal"):
                continue
            value = episode[column]
            if column.startswith("pixels") and isinstance(value, torch.Tensor):
                value = value.permute(0, 2, 3, 1)
            if not isinstance(value, (torch.Tensor, np.ndarray)):
                continue
            array = value.numpy() if isinstance(value, torch.Tensor) else value
            init_lists.setdefault(column, []).append(array[0])
            goal_lists.setdefault(column, []).append(array[-1])
    init_state = {key: np.stack(value) for key, value in init_lists.items()}
    goal_state = {
        "goal" if key == "pixels" else f"goal_{key}": np.stack(value)
        for key, value in goal_lists.items()
    }
    return init_state, goal_state


def _apply_callables(env, callables, init_state) -> None:
    for spec in callables:
        method = spec["method"]
        if not hasattr(env, method):
            continue
        prepared = {}
        for name, data in spec.get("args", {}).items():
            if data.get("in_dataset", True):
                key = data.get("value")
                if key in init_state:
                    prepared[name] = deepcopy(init_state[key])
            else:
                prepared[name] = data.get("value")
        getattr(env, method)(**prepared)


def _historical_evaluate_from_dataset(
    world,
    dataset,
    episodes_idx,
    start_steps,
    goal_offset,
    eval_budget,
    callables,
) -> dict:
    """Backport modern fixed-dataset evaluation to historical World."""
    count = len(episodes_idx)
    if count != world.num_envs:
        raise ValueError(
            f"dataset evaluation has {count} episodes but {world.num_envs} envs"
        )
    init_state, goal_state = _extract_init_goal(
        dataset, episodes_idx, start_steps, goal_offset
    )
    world.reset(seed=init_state.get("seed"))

    if callables:
        merged = {**init_state, **goal_state}
        envs = world.envs.unwrapped.envs
        for index in range(count):
            env_init = {key: value[index] for key, value in merged.items()}
            _apply_callables(envs[index].unwrapped, callables, env_init)

    shape_prefix = world.infos["pixels"].shape[:2]
    for source in (init_state, goal_state):
        for key, value in source.items():
            if key in world.infos or key in goal_state:
                world.infos[key] = np.broadcast_to(
                    value[:, None, ...], shape_prefix + value.shape[1:]
                ).copy()
    goal_snapshot = {key: world.infos[key].copy() for key in goal_state}
    successes = np.zeros(count, dtype=bool)
    alive = np.ones(count, dtype=bool)
    for _ in range(eval_budget):
        world.step()
        terminated = np.asarray(world.terminateds, dtype=bool)
        truncated = np.asarray(world.truncateds, dtype=bool)
        successes |= alive & terminated
        newly_done = alive & (terminated | truncated)
        alive[newly_done] = False

        # Old Gymnasium has no vector ``mask=`` step.  It asserts if a done
        # sub-environment is stepped again, so reset those slots to satisfy
        # its state machine while ``alive`` prevents any second attempt from
        # contributing to the metric (the modern implementation freezes them).
        must_reset = terminated | truncated
        if must_reset.any():
            vector_env = world.envs.unwrapped
            vector_env._autoreset_envs = np.zeros(count, dtype=bool)
            for index in np.flatnonzero(must_reset):
                _, infos = vector_env.envs[index].reset()
                for key, value in infos.items():
                    if key in world.infos:
                        world.infos[key][index] = np.asarray(value)

        world.infos.update(deepcopy(goal_snapshot))
        if not alive.any():
            break

    return {
        "success_rate": float(successes.sum()) / count * 100.0,
        "episode_successes": successes,
        "seeds": init_state.get("seed"),
    }


def install_world_evaluate_compat(world_class) -> bool:
    """Allow the modern GC-IDM evaluator to call the historical World API."""
    evaluate = world_class.evaluate
    if "dataset" in inspect.signature(evaluate).parameters:
        return False

    @functools.wraps(evaluate)
    def compatible_evaluate(self, *args, **kwargs):
        dataset = kwargs.pop("dataset", None)
        if dataset is not None:
            if args:
                raise TypeError("dataset evaluation requires keyword arguments")
            return _historical_evaluate_from_dataset(
                self,
                dataset=dataset,
                episodes_idx=kwargs.pop("episodes_idx"),
                start_steps=kwargs.pop("start_steps"),
                goal_offset=kwargs.pop("goal_offset"),
                eval_budget=kwargs.pop("eval_budget"),
                callables=kwargs.pop("callables", None),
            )
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
