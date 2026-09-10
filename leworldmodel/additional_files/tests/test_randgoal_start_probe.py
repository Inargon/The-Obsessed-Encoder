import numpy as np

from additional_files.diagnose_randgoal_start import pose_target, relative_target


def test_pose_target_wraps_angle_as_sincos():
    pose = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, np.pi / 2]])
    target = pose_target(pose)
    assert target.shape == (2, 4)
    np.testing.assert_allclose(target[0], [1.0, 2.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(target[1], [3.0, 4.0, 1.0, 0.0], atol=1e-6)


def test_relative_target_uses_block_pose():
    state = np.array([[10.0, 20.0, 3.0, 5.0, 0.25]])
    goal = np.array([[8.0, 2.0, 0.75]])
    target = relative_target(state, goal)
    np.testing.assert_allclose(
        target[0], [5.0, -3.0, np.sin(0.5), np.cos(0.5)], atol=1e-6
    )
