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


def test_visual_motion_score_prefers_visible_change():
    still = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    moving = [frame.copy() for frame in still]
    moving[-1][:] = 255
    assert module.visual_motion_score(still) == 0.0
    assert module.visual_motion_score(moving) > module.visual_motion_score(still)
