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
