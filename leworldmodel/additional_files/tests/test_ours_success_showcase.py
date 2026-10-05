import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "compose_ours_success_showcase", ROOT / "compose_ours_success_showcase.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def triptych(agent: int, goal: int) -> np.ndarray:
    frame = np.zeros((36, 96, 3), dtype=np.uint8)
    frame[:32, :32] = agent
    frame[:32, 64:96] = goal
    return frame


def test_terminal_alignment_prefers_matching_agent_and_goal():
    matched = triptych(100, 100)
    mismatched = triptych(20, 200)
    assert module.terminal_agent_goal_rmse([matched]) == 0.0
    assert module.terminal_agent_goal_rmse([matched]) < module.terminal_agent_goal_rmse(
        [mismatched]
    )


def test_all_supported_tasks_have_paper_labels():
    assert set(module.TASK_DISPLAY) == {
        "clean_pusht",
        "tagged_pusht",
        "tworoom",
        "cube",
    }


def tworoom_frame(start_x: int, goal_x: int, side: int = 32) -> np.ndarray:
    frame = np.full((side, 3 * side, 3), 255, dtype=np.uint8)
    frame[15:18, start_x - 1 : start_x + 2] = (255, 0, 0)
    frame[15:18, 2 * side + goal_x - 1 : 2 * side + goal_x + 2] = (255, 0, 0)
    return frame


def test_cross_room_requires_start_and_goal_on_opposite_sides():
    opposite = tworoom_frame(start_x=8, goal_x=24)
    same = tworoom_frame(start_x=8, goal_x=10)
    assert module.is_cross_room([opposite])[0]
    assert not module.is_cross_room([same])[0]


def test_red_agent_centroid_tracks_the_rendered_blob():
    frame = np.full((32, 32, 3), 255, dtype=np.uint8)
    frame[10:13, 21:24] = (255, 0, 0)
    x, y = module.red_agent_centroid(frame)
    assert np.isclose(x, 22.0)
    assert np.isclose(y, 11.0)
