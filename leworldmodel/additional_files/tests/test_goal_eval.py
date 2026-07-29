"""Destination alignment: the eval env draws the T where the dataset put it."""
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from stable_worldmodel.envs.pusht.env import PushT  # noqa: E402

# Both examples ship an ``additional_files`` package; swap any cached one out,
# import the LeWM one under its real package name (the callbacks package uses
# relative imports, so a bare top-level ``callbacks`` alias cannot host it),
# and restore afterwards (same isolation as test_wandb_rename.py).
_LEWM_DIR = str(Path(__file__).resolve().parents[2])
_saved = {name: sys.modules.pop(name) for name in list(sys.modules)
          if name == "additional_files" or name.startswith("additional_files.")}
sys.path.insert(0, _LEWM_DIR)
from additional_files.callbacks.goal_eval import (  # noqa: E402
    EvalTagStamp, GoalEvalCallback, eval_transforms,
)
from additional_files.pixel_tag import PixelTag  # noqa: E402

for _name in [m for m in sys.modules
              if m == "additional_files" or m.startswith("additional_files.")]:
    del sys.modules[_name]
sys.modules.update(_saved)
sys.path.remove(_LEWM_DIR)


class _Dataset:
    """Column source; raises like the real readers when a column is absent."""

    def __init__(self, cols):
        self.cols = cols

    def get_col_data(self, name):
        if name not in self.cols:
            raise KeyError(name)
        return self.cols[name]


class _World:
    def __init__(self, envs):
        self.envs = type("Pool", (), {"envs": envs})()


def _callback(envs):
    cb = GoalEvalCallback({"every_n_steps": 1}, seed=0)
    cb._world = _World(envs)
    return cb


DEFAULT_POSE = np.array([256.0, 256.0, np.pi / 4])


def test_randomized_destination_reaches_the_env():
    envs, poses = [PushT(), PushT()], np.array([[140.0, 371.0, -2.5], [300.0, 160.0, 1.2]])
    _callback(envs)._align_destinations(_Dataset({"goal_pose": poses}), rows=[0, 1])
    for env, pose in zip(envs, poses):
        env.reset(seed=7)
        assert np.allclose(env.goal_pose, pose)


def test_alignment_survives_repeated_resets():
    env = PushT()
    pose = np.array([[200.0, 300.0, 0.5]])
    _callback([env])._align_destinations(_Dataset({"goal_pose": pose}), rows=[0])
    for seed in (1, 2, 3):
        env.reset(seed=seed)
        assert np.allclose(env.goal_pose, pose[0])


def test_fixed_destination_dataset_keeps_the_env_default():
    env = PushT()
    _callback([env])._align_destinations(_Dataset({"state": np.zeros((4, 7))}), rows=[0])
    env.reset(seed=7)
    assert np.allclose(env.goal_pose, DEFAULT_POSE)


def test_only_the_selected_rows_are_used():
    env = PushT()
    poses = np.array([[100.0, 100.0, 0.0], [340.0, 210.0, -1.0], [150.0, 150.0, 2.0]])
    _callback([env])._align_destinations(_Dataset({"goal_pose": poses}), rows=[1])
    env.reset(seed=7)
    assert np.allclose(env.goal_pose, poses[1])


# ---------------------------------------------------------------------------
# EvalTagStamp through the REAL policy path: WorldModelPolicy._prepare_info
# permutes frames channel-first before the transform Compose runs, which is
# exactly the layout that silently defeated the stamp (260728 review V1).
# ---------------------------------------------------------------------------
def _prepare_through_policy(transforms, buf):
    from stable_worldmodel.policy import WorldModelPolicy

    policy = WorldModelPolicy.__new__(WorldModelPolicy)
    policy.transform = transforms
    return policy._prepare_info({"pixels": buf.copy(), "goal": buf.copy()})


def test_eval_tag_stamp_fires_through_prepare_info():
    rng = np.random.default_rng(0)
    buf = rng.integers(0, 255, size=(2, 1, 224, 224, 3), dtype=np.uint8)
    tag = PixelTag(mode="video", size=5, seed=0)

    tagged = _prepare_through_policy(eval_transforms(224, tag), buf)
    plain = _prepare_through_policy(eval_transforms(224, None), buf)

    for key in ("pixels", "goal"):
        t, p = np.asarray(tagged[key]), np.asarray(plain[key])
        assert t.shape == p.shape
        # the tag corner must differ from the untagged pipeline...
        assert not np.allclose(t[..., :5, :5], p[..., :5, :5]), key
        # ...and everything outside it must be untouched
        assert np.allclose(t[..., 5:, 5:], p[..., 5:, 5:]), key


def test_eval_tag_stamp_covers_both_layouts():
    tag = PixelTag(mode="video", size=5, seed=0)
    color = tag.color_for(EvalTagStamp._VIDEO_KEY)

    hwc = EvalTagStamp(tag)(np.zeros((224, 224, 3), dtype=np.uint8))
    assert np.array_equal(hwc[:5, :5], np.broadcast_to(color, (5, 5, 3)))

    import torch
    chw = EvalTagStamp(tag)(torch.zeros(3, 224, 224, dtype=torch.uint8))
    assert np.array_equal(chw[:, :5, :5].numpy(),
                          np.broadcast_to(color.reshape(3, 1, 1), (3, 5, 5)))
