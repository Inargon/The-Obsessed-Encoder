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


def test_fingertip_geometry_measures_goal_approach():
    frames = [reacher_tile((3, 3), (6, 6)), reacher_tile((6, 6), (6, 6))]
    geometry = module.fingertip_geometry(frames)
    assert geometry["start_distance_over_width"] > 0
    assert geometry["end_distance_over_width"] == 0


def test_orange_fingertip_chooses_distal_component_not_elbow():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    orange = (230, 100, 20)
    # Elbow close to the fixed image-centre shoulder.
    frame[9:11, 12:14] = orange
    # Fingertip farther away.
    frame[2:4, 16:18] = orange
    x, y = module.orange_fingertip(frame)
    assert np.isclose(x, 16.5)
    assert np.isclose(y, 2.5)
