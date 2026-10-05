from __future__ import annotations

import pytest
import torch

from additional_files.jepa_bisim import RewardFreeBisimulationObjective


def test_bisimulation_objective_is_finite_and_differentiable() -> None:
    torch.manual_seed(13)
    emb = torch.randn(8, 4, 6, requires_grad=True)
    predicted = torch.randn(8, 3, 6, requires_grad=True)
    objective = RewardFreeBisimulationObjective()

    terms = objective(emb, predicted)
    terms["bisim_loss"].backward()

    assert terms["bisim_loss"] > 0
    assert emb.grad is not None and torch.isfinite(emb.grad).all()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()


def test_bisimulation_total_respects_configured_weights() -> None:
    torch.manual_seed(17)
    objective = RewardFreeBisimulationObjective(
        weight=2.0,
        variance_weight=0.5,
        covariance_weight=0.25,
    )
    terms = objective(torch.randn(5, 4, 3), torch.randn(5, 3, 3))
    expected = 2.0 * (
        terms["bisim_metric_loss"]
        + 0.5 * terms["bisim_variance_loss"]
        + 0.25 * terms["bisim_covariance_loss"]
    )
    assert terms["bisim_loss"].item() == pytest.approx(expected.item())


def test_bisimulation_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="discount"):
        RewardFreeBisimulationObjective(discount=1.1)
