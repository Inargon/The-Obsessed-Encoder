import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "compose_repair_only_showcase", ROOT / "compose_repair_only_showcase.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_green_goal_mask_ignores_tag_corner_and_finds_goal():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[:5, :5] = [0, 255, 0]
    frame[20:25, 20:25] = [100, 180, 110]
    mask = module.green_goal_mask(frame)
    assert not mask[:5, :5].any()
    assert mask[20:25, 20:25].all()


def test_terminal_goal_residual_prefers_tighter_overlap_proxy():
    tight = np.zeros((32, 32, 3), dtype=np.uint8)
    loose = tight.copy()
    tight[20:22, 20:22] = [100, 180, 110]
    loose[20:28, 20:28] = [100, 180, 110]
    assert module.terminal_goal_residual([tight]) < module.terminal_goal_residual([loose])


def test_extract_panel_removes_triptych_and_label_strip():
    frame = np.zeros((36, 96, 3), dtype=np.uint8)
    frame[:32, :32] = 10
    frame[:32, 32:64] = 20
    frame[:32, 64:96] = 30
    assert (module.extract_panel(frame, "agent") == 10).all()
    assert (module.extract_panel(frame, "dataset") == 20).all()
    assert (module.extract_panel(frame, "goal") == 30).all()
