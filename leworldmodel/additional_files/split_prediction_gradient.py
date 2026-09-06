"""Separate prediction-loss strength for predictor and encoder parameters."""

from __future__ import annotations

import torch


def encoder_scaled_prediction_surrogate(
    pred_loss: torch.Tensor,
    embedding: torch.Tensor,
    encoder_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Scale only the prediction gradient entering the representation.

    ``pred_loss`` remains in the total loss with coefficient one and therefore
    trains predictor parameters at full strength. The returned zero-valued
    surrogate corrects its gradient at ``embedding`` to ``encoder_weight``
    times the original value, which also scales the gradient received by the
    upstream projector and encoder.
    """
    if not 0.0 <= encoder_weight <= 1.0:
        raise ValueError("encoder_weight must lie in [0, 1]")

    pred_grad = torch.autograd.grad(
        pred_loss, embedding, retain_graph=True, create_graph=False
    )[0].detach()
    correction = (encoder_weight - 1.0) * pred_grad
    surrogate = (correction * (embedding - embedding.detach())).sum()
    diagnostics = {
        "encoder_prediction_weight": embedding.new_tensor(encoder_weight),
        "prediction_grad_removed_fraction": embedding.new_tensor(
            1.0 - encoder_weight
        ),
    }
    return surrogate, diagnostics
