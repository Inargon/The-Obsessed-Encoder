"""Column building + Lance writing. Output loads through swm.data.load_dataset
exactly like the released pusht_expert_train.lance; every frame has a real
action (no NaN padding), so training never sees NaN inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_worldmodel.data import LanceWriter


def build_episode_columns(
    states: np.ndarray,
    actions: np.ndarray,
    frames: np.ndarray,
    goal: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per-frame columns for one episode; length == len(actions) (frames/states trimmed to match).

    states:  (>=L, 7)  [agent_x, agent_y, block_x, block_y, block_angle, vel_x, vel_y]
    actions: (L, 2)
    frames:  (>=L, H, W, 3) uint8
    goal:    optional (3,) commanded destination [x, y, angle], broadcast to every
             frame as a `goal_pose` column.
    """
    n = len(actions)
    states = states[:n]
    frames = frames[:n]
    proprio = states[:, [0, 1, 5, 6]].astype(np.float32)
    cols = {
        "pixels": np.asarray(frames, dtype=np.uint8),
        "action": np.asarray(actions, dtype=np.float32),
        "proprio": proprio,
        "state": states.astype(np.float32),
    }
    if goal is not None:
        cols["goal_pose"] = np.broadcast_to(
            np.asarray(goal, dtype=np.float32), (n, len(goal))
        ).copy()
    return cols


def write_dataset(path: str | Path, episodes) -> None:
    """Stream an iterable of episode column-dicts into one Lance table.

    The whole iterable lands through a single RecordBatchReader, so memory
    stays bounded to one in-flight episode regardless of dataset size.
    LanceWriter type-detects on per-frame value lists, so each column array is
    handed over as a list of its rows.
    """
    as_lists = ({k: list(v) for k, v in ep.items()} for ep in episodes)
    with LanceWriter(str(path), mode="overwrite") as writer:
        writer.write_episodes(as_lists)
