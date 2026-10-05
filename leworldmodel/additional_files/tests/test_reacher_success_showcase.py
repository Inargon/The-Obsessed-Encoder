import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "compose_reacher_success_showcase", ROOT / "compose_reacher_success_showcase.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_parse_historical_multiline_metrics():
    text = """metrics: {'success_rate': 75.0, 'episode_successes': array([ True,
 False, True, True]), 'seeds': array([1, 2, 3, 4])}"""
    rate, mask = module.parse_historical_metrics(text)
    assert rate == 0.75
    assert mask == [True, False, True, True]


def reacher_tile(current_tip=(3, 3), goal_tip=(6, 6)):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    # Lower-left current and lower-right goal tiles.
    cx, cy = current_tip
    gx, gy = goal_tip
    frame[10 + cy : 12 + cy, cx : 2 + cx] = (230, 100, 20)
    frame[10 + gy : 12 + gy, 10 + gx : 12 + gx] = (230, 100, 20)
    return frame


def test_split_reacher_frame_uses_latest_lower_row():
    current, goal = module.split_reacher_frame(reacher_tile())
    assert current.shape == (10, 10, 3)
    assert goal.shape == (10, 10, 3)
    assert tuple(current[3, 3]) == (230, 100, 20)
    assert tuple(goal[6, 6]) == (230, 100, 20)


def test_pose_geometry_measures_goal_approach():
    frames = [reacher_tile((3, 3), (6, 6)), reacher_tile((6, 6), (6, 6))]
    geometry = module.pose_geometry(frames)
    assert geometry["start_pose_rmse"] > 0
    assert geometry["end_pose_rmse"] == 0


def test_goal_pose_overlay_uses_exact_goal_pixels_without_fake_geometry():
    current = np.full((10, 10, 3), (30, 70, 110), dtype=np.uint8)
    goal = current.copy()
    goal[6:8, 7:9] = (230, 160, 25)
    rendered = module.overlay_goal_pose(current, goal)
    assert np.array_equal(rendered[1, 1], current[1, 1])
    assert rendered[6, 7, 0] > current[6, 7, 0]
    assert rendered[6, 7, 2] > current[6, 7, 2]
