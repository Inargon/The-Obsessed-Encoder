"""Controlled PushT stimulus rendering shared by the pair analysis and the
encoder-readout figures.

Frames are rendered fresh from the simulator — never taken from training data
— so every factor (agent pose, block pose, destination outline, corner colour)
is independently controllable.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
import stable_worldmodel.envs  # noqa: F401  registers swm/PushT-v1

# p1..p99 of the pusht_expert_train `state` column [agent_xy].
AGENT_LO = np.array([38.0, 75.0])
AGENT_HI = np.array([449.0, 482.0])

# Legal tee placement, the scripted collector's convention
# (pusht_random_dest/collect.py --goal-bounds): a tee whose center is
# in [140, 372]^2 stays fully inside the walls at any angle. Used for both the
# block (T-orig) and the destination outline (T-dest), so no stimulus renders
# a partially out-of-field tee.
TEE_LO = np.array([140.0, 140.0])
TEE_HI = np.array([372.0, 372.0])

RENDER_SEED = 0  # fixed so the scene's background/agent/block colours are constant


class StimulusRenderer:
    """Lazily-created swm/PushT-v1 env that renders (state, goal) pairs."""

    def __init__(self):
        self._env = None

    def _env_lazy(self):
        if self._env is None:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            self._env = gym.make("swm/PushT-v1", render_mode="rgb_array")
        return self._env

    def render(self, orig_state: np.ndarray, dest_pose: np.ndarray) -> np.ndarray:
        """One (H, W, 3) uint8 frame: block/agent at orig_state, the target-T
        outline drawn at dest_pose."""
        u = self._env_lazy().unwrapped
        u.reset(seed=RENDER_SEED, options={"state": np.asarray(orig_state, float)})
        u.goal_pose = np.asarray(dest_pose, float)
        return np.ascontiguousarray(u.render())


def sample_orig(rng: np.random.Generator, k: int) -> np.ndarray:
    """k full env states [agent_xy, block_xy, angle, vel_xy(=0)]."""
    a = rng.uniform(AGENT_LO, AGENT_HI, size=(k, 2))
    b = rng.uniform(TEE_LO, TEE_HI, size=(k, 2))
    ang = rng.uniform(0.0, 2 * np.pi, size=(k, 1))
    return np.concatenate([a, b, ang, np.zeros((k, 2))], axis=1)


def sample_dest(rng: np.random.Generator, k: int) -> np.ndarray:
    """k destination poses [x, y, angle]."""
    b = rng.uniform(TEE_LO, TEE_HI, size=(k, 2))
    ang = rng.uniform(0.0, 2 * np.pi, size=(k, 1))
    return np.concatenate([b, ang], axis=1)


def stamp_corner(frame: np.ndarray, color, size: int) -> np.ndarray:
    """A copy of an (H, W, 3) frame with the corner square set to color."""
    out = frame.copy()
    out[:size, :size, :] = np.asarray(color, np.uint8)
    return out
