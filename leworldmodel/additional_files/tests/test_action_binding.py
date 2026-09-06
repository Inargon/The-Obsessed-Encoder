import numpy as np
import torch

from leworldmodel.additional_files.action_binding import binding_losses
from leworldmodel.additional_files.generate_action_binding import select_branches


def test_binding_prefers_the_correct_same_anchor_branch():
    torch.manual_seed(0)
    target = torch.randn(2, 6, 12)
    partner = torch.full((2, 6), -1, dtype=torch.long)
    good = binding_losses(target, target, target, partner)
    bad = binding_losses(target.roll(1, dims=1), target, target, partner)
    assert good["binding_loss"] < bad["binding_loss"]
    assert good["binding_accuracy"] == 1
    assert good["nuisance_effect_loss"] == 0


def test_select_branches_keeps_a_mediator_matched_hard_pair():
    anchor = np.zeros(5)
    finals = np.array([
        [1, 1, 0, 0, 0],
        [2, 2, 30, 0, 0],
        [80, 0, 0, 40, 0.2],
        [-80, 0, 0, -40, -0.2],
    ], dtype=float)
    selected, partner = select_branches(
        finals, anchor, count=4, hard_pairs=1,
        rng=np.random.default_rng(0),
    )
    assert set(selected[:2]) == {0, 1}
    assert tuple(partner[:2]) == (1, 0)
