from __future__ import annotations

import numpy as np

from leworldmodel.additional_files.tag_folding_probe import sample_rgb, stamp_tag


def test_sample_rgb_is_deterministic_and_varies_all_channels() -> None:
    first = sample_rgb(64, seed=7)
    second = sample_rgb(64, seed=7)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (64, 3)
    assert all(len(np.unique(first[:, channel])) > 8 for channel in range(3))


def test_stamp_tag_changes_only_requested_cell() -> None:
    image = np.arange(20 * 24 * 3, dtype=np.uint8).reshape(20, 24, 3)
    result = stamp_tag(image, np.array([11, 22, 33], dtype=np.uint8), size=5)
    expected = image.copy()
    expected[:5, :5] = np.array([11, 22, 33], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)
    assert not np.shares_memory(result, image)
