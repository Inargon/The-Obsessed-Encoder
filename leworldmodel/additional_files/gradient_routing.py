"""Representation-level conflict routing for prediction gradients."""

from __future__ import annotations

import torch


def conflict_aware_prediction_surrogate(
    pred_loss: torch.Tensor,
    control_loss: torch.Tensor,
    embedding: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return a zero-valued term that projects conflicting prediction gradients.

    Gradients are compared per sample at the projected encoder representation.
    The original prediction loss continues to train the predictor normally;
    adding the returned surrogate changes only the gradient entering
    ``embedding`` and therefore its encoder/projector ancestors.
    """
    pred_grad = torch.autograd.grad(
        pred_loss, embedding, retain_graph=True, create_graph=False
    )[0].detach()
    control_grad = torch.autograd.grad(
        control_loss, embedding, retain_graph=True, create_graph=False
    )[0].detach()

    # Do the geometry in fp32 even under the bf16 training configuration.
    pred_flat = pred_grad.float().flatten(1)
    control_flat = control_grad.float().flatten(1)
    dot = (pred_flat * control_flat).sum(dim=1)
    control_norm_sq = control_flat.square().sum(dim=1).clamp_min(eps)
    conflict = dot < 0

    # For a conflicting sample, remove the component of g_pred that points
    # opposite g_control. Non-conflicting samples keep the full prediction
    # gradient. This is the asymmetric PCGrad update appropriate here because
    # control is the protected objective and prediction is the routed one.
    coefficient = torch.where(conflict, dot / control_norm_sq, torch.zeros_like(dot))
    coefficient = coefficient.to(dtype=embedding.dtype)
    safe_pred_grad = pred_grad - coefficient.view(
        -1, *([1] * (embedding.ndim - 1))
    ) * control_grad
    correction = safe_pred_grad - pred_grad

    # Exactly zero in the forward pass, with d(surrogate)/d(embedding) equal
    # to the detached correction. Predictor gradients from pred_loss are not
    # touched because the surrogate depends directly on embedding only.
    surrogate = (correction * (embedding - embedding.detach())).sum()

    pred_norm = pred_flat.norm(dim=1)
    control_norm = control_flat.norm(dim=1)
    cosine = dot / (pred_norm * control_norm).clamp_min(eps)
    removed_fraction = correction.flatten(1).norm(dim=1) / pred_norm.clamp_min(eps)
    diagnostics = {
        "prediction_control_grad_cosine": cosine.mean(),
        "prediction_control_conflict_fraction": conflict.float().mean(),
        "prediction_grad_removed_fraction": removed_fraction.mean(),
    }
    return surrogate, diagnostics
