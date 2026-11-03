from __future__ import annotations

import torch

from leworldmodel.additional_files.action_effect_ratio import ActionEffectRatio


def test_action_effect_ratio_has_finite_encoder_and_head_gradients():
    torch.manual_seed(7)
    objective = ActionEffectRatio(embed_dim=8, action_dim=4, hidden_dim=16)
    start = torch.randn(6, 8, requires_grad=True)
    end = torch.randn(6, 8, requires_grad=True)
    actions = torch.randn(6, 4)
    changed_start = start + 0.1 * torch.randn(6, 8)
    changed_end = end + 0.1 * torch.randn(6, 8)

    result = objective(
        start, end, actions,
        counterfactual_start=changed_start,
        counterfactual_end=changed_end,
        consistency_weight=0.2,
    )
    result["action_effect_loss"].backward()

    assert result["action_effect_score"].shape == (6,)
    assert result["action_effect_consistency_loss"] > 0
    assert torch.isfinite(result["action_effect_loss"])
    assert start.grad is not None and torch.isfinite(start.grad).all()
    assert end.grad is not None and torch.isfinite(end.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in objective.parameters())


def test_identical_counterfactual_has_zero_score_change():
    objective = ActionEffectRatio(embed_dim=5, action_dim=2, hidden_dim=12)
    start, end, actions = torch.randn(3, 5), torch.randn(3, 5), torch.randn(3, 2)
    result = objective(
        start, end, actions,
        counterfactual_start=start,
        counterfactual_end=end,
        consistency_weight=1.0,
    )
    assert result["action_effect_consistency_loss"] == 0
