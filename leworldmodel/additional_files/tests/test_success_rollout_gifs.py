import numpy as np

from additional_files.compose_success_rollout_gifs import (
    stamp_rgb,
    synchronized_indices,
)


def test_synchronized_indices_include_exact_endpoints():
    np.testing.assert_array_equal(synchronized_indices(10, 4), [0, 3, 6, 9])


def test_synchronized_indices_can_hold_short_streams():
    values = synchronized_indices(2, 5)
    assert values[0] == 0
    assert values[-1] == 1
    assert set(values) == {0, 1}


def test_stamp_rgb_changes_only_requested_corner():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    output = stamp_rgb(frame, np.asarray([10, 20, 30], dtype=np.uint8), size=2)
    assert np.all(output[:2, :2] == [10, 20, 30])
    assert not output[2:].any()
    assert not output[:, 2:].any()
    assert not frame.any()
