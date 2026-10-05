from __future__ import annotations

import numpy as np

from additional_files.hjepa_tagged_adapter import _fixed_rows


class FakeDataset:
    def __init__(self):
        self.cols = {
            "episode_idx": np.repeat(np.arange(4), 40),
            "step_idx": np.tile(np.arange(40), 4),
        }

    def get_col_data(self, name):
        return self.cols[name]


def test_fixed_rows_are_deterministic_sorted_and_have_goal_runway():
    dataset = FakeDataset()
    episodes, starts, rows = _fixed_rows(dataset, 42, 30, 5, 10, 25)
    assert len(episodes) == len(starts) == len(rows) == 10
    assert rows == sorted(rows)
    assert max(starts) <= 14
    again = _fixed_rows(dataset, 42, 30, 5, 10, 25)
    assert (episodes, starts, rows) == again


def test_fixed_rows_changes_with_seed():
    dataset = FakeDataset()
    assert _fixed_rows(dataset, 1, 20, 0, 10, 25)[2] != _fixed_rows(
        dataset, 2, 20, 0, 10, 25
    )[2]
