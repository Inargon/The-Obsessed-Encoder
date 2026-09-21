import numpy as np
import torch

from additional_files.diagnose_tag_intervention import (
    choose_finite_action_indices,
)


class _Dataset:
    def __init__(self, actions):
        self.actions = actions

    def __getitem__(self, index):
        return {"action": self.actions[index]}


def test_finite_action_sampler_skips_episode_boundary_padding():
    dataset = _Dataset(
        [
            torch.tensor([[1.0], [float("nan")]]),
            torch.tensor([[2.0], [3.0]]),
            torch.tensor([[float("nan")], [4.0]]),
            torch.tensor([[5.0], [6.0]]),
        ]
    )

    selected = choose_finite_action_indices(
        dataset, np.array([0, 1, 2, 3]), requested=2
    )

    assert selected == [1, 3]


def test_finite_action_sampler_fails_when_too_few_valid_clips():
    dataset = _Dataset(
        [torch.tensor([[float("nan")]]), torch.tensor([[1.0]])]
    )

    try:
        choose_finite_action_indices(
            dataset, np.array([0, 1]), requested=2
        )
    except ValueError as error:
        assert "Requested 2" in str(error)
        assert "only 1" in str(error)
    else:
        raise AssertionError("expected insufficient-valid-clips failure")
