"""Figure 10 stimuli: the green goal outline is never occluded along either path.

The pixel tests are the real guarantee — they render and count green — so the
geometry check lives here rather than in the figure script.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Both examples ship an ``additional_files`` package and the DINOv3 one may
# already be imported by the time this module collects; swap it out, import
# the LeWM one, and put everything back so later dinov3 imports resolve theirs
# (same isolation as test_wandb_rename.py).
_LEWM_DIR = str(Path(__file__).resolve().parents[2])
_saved = {name: sys.modules.pop(name) for name in list(sys.modules)
          if name == "additional_files" or name.startswith("additional_files.")}
sys.path.insert(0, _LEWM_DIR)
from additional_files.readout import (  # noqa: E402
    PARKED_AGENT, PARKED_STATE, PARKED_TEE, SEED, TEE_REACH, WANDER_HI,
    WANDER_LO, block_states, wander_pose,
)
from additional_files.stimuli import StimulusRenderer  # noqa: E402

for _name in [m for m in sys.modules
              if m == "additional_files" or m.startswith("additional_files.")]:
    del sys.modules[_name]
sys.modules.update(_saved)

GREEN = np.array([144, 238, 144], np.uint8)  # pygame LightGreen, the goal outline
OFFSCREEN = np.array([2000.0, 2000.0])  # past the walls, so nothing is drawn
AGENT_RADIUS = 15.0
POINTS = 800


def paths():
    poses = wander_pose(SEED, POINTS)
    return poses, block_states(poses)


def green_pixels(frame):
    return int((frame == GREEN).all(axis=-1).sum())


def gap_to_parked_tee(points):
    """Distance from each point to the parked tee's footprint, 0 inside. Its
    angle is 0, so the footprint is the axis-aligned box below."""
    x, y = PARKED_TEE[:2]
    box = np.array([x - 60.0, y, x + 60.0, y + 120.0])
    outside = np.maximum(np.maximum(box[:2] - points, points - box[2:]), 0.0)
    return np.hypot(outside[:, 0], outside[:, 1])


@pytest.fixture(scope="module")
def renderer():
    return StimulusRenderer()


def test_moving_tee_clears_the_parked_tee_and_the_agent():
    """A tee at any angle fits inside a TEE_REACH disc about its position, so
    clearing the parked footprint by that much clears it at every angle."""
    poses, _ = paths()
    assert gap_to_parked_tee(poses[:, :2]).min() > TEE_REACH
    assert np.linalg.norm(poses[:, :2] - PARKED_AGENT, axis=1).min() > TEE_REACH + AGENT_RADIUS
    assert gap_to_parked_tee(PARKED_AGENT[None])[0] > AGENT_RADIUS


def test_the_check_would_catch_an_overlap():
    assert gap_to_parked_tee(PARKED_TEE[None, :2])[0] == 0.0
    assert gap_to_parked_tee(np.array([(WANDER_LO + WANDER_HI) / 2]))[0] > TEE_REACH


def test_both_panels_move_along_the_identical_trajectory():
    """One trajectory drives both panels, so the comparison cannot be
    confounded by one tee sweeping further than the other. The block states
    must carry it verbatim, with the agent parked."""
    poses, states = paths()
    assert np.allclose(poses, states[:, [2, 3, 4]])
    assert np.allclose(states[:, :2], PARKED_AGENT)
    assert np.allclose(states[:, 5:], 0.0)
    assert np.all(poses[:, :2] >= WANDER_LO) and np.all(poses[:, :2] <= WANDER_HI)


def test_goal_panel_shows_every_green_pixel(renderer):
    """Closest approach plus the box corners: the parked block and agent hide
    none of the wandering goal, checked against a render with them removed."""
    poses, _ = paths()
    near = int(np.linalg.norm(poses[:, :2] - PARKED_TEE[:2], axis=1).argmin())
    probes = [poses[near], poses[0], poses[-1],
              np.array([WANDER_LO[0], WANDER_LO[1], 0.0])]
    bare = np.array([*OFFSCREEN, *(OFFSCREEN + 400.0), 0.0, 0.0, 0.0])
    for pose in probes:
        full = green_pixels(renderer.render(PARKED_STATE, pose))
        alone = green_pixels(renderer.render(bare, pose))
        assert full == alone, f"goal at {pose[:2]} is occluded ({full} of {alone} px)"
        assert alone > 0


def test_block_panel_shows_every_green_pixel(renderer):
    """The wandering block and the parked agent hide none of the fixed goal."""
    _, states = paths()
    near = int(np.linalg.norm(states[:, 2:4] - PARKED_TEE[:2], axis=1).argmin())
    bare = np.array([*OFFSCREEN, *(OFFSCREEN + 400.0), 0.0, 0.0, 0.0])
    alone = green_pixels(renderer.render(bare, PARKED_TEE))
    assert alone > 0
    for state in (states[near], states[0], states[-1]):
        full = green_pixels(renderer.render(state, PARKED_TEE))
        assert full == alone, f"block at {state[2:4]} occludes the goal ({full} of {alone} px)"
