"""Diagnostics for prediction gradients against individual control signals."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def _flat_gradient(
    loss: torch.Tensor,
    embedding: torch.Tensor,
) -> torch.Tensor:
    gradient = torch.autograd.grad(
        loss,
        embedding,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]
    if gradient is None:
        gradient = torch.zeros_like(embedding)
    return gradient.detach().float().flatten(1)


def measure_component_gradient_geometry(
    pred_loss: torch.Tensor,
    component_losses: Mapping[str, torch.Tensor],
    embedding: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Measure per-component alignment and joint control-span coverage.

    ``component_losses`` should already contain the weights with which each
    term enters the control objective.  The function is diagnostic only: all
    gradients are detached and the optimized loss is unchanged.
    """
    pred = _flat_gradient(pred_loss, embedding)
    pred_norm = pred.norm(dim=1)
    metrics: dict[str, torch.Tensor] = {}
    normalized_components = []
    raw_components = []

    for name, loss in component_losses.items():
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            continue
        gradient = _flat_gradient(loss, embedding)
        norm = gradient.norm(dim=1)
        dot = (pred * gradient).sum(dim=1)
        cosine = dot / (pred_norm * norm).clamp_min(eps)
        metrics[f"component_grad/{name}/cosine"] = cosine.mean()
        metrics[f"component_grad/{name}/negative_fraction"] = (
            dot < 0
        ).float().mean()
        metrics[f"component_grad/{name}/norm_ratio"] = (
            norm / pred_norm.clamp_min(eps)
        ).mean()
        normalized_components.append(
            gradient / norm.clamp_min(eps).unsqueeze(1)
        )
        raw_components.append(gradient)

    if not normalized_components:
        return metrics

    # Row-normalized component gradients form a small per-sample matrix.  Its
    # row span is invariant to the original loss scales and exposes how much
    # prediction-gradient energy is covered jointly by all control signals.
    # Lightning trains under mixed precision.  CUDA autocast would turn the
    # small matrix products back into bfloat16 even though `_flat_gradient`
    # explicitly returns float32, while `linalg.pinv` requires float/complex.
    # Keep this diagnostic-only solve in float32.
    with torch.autocast(device_type=embedding.device.type, enabled=False):
        basis = torch.stack(normalized_components, dim=1).float()
        pred_float = pred.float()
        gram = basis @ basis.transpose(1, 2)
        rhs = (basis @ pred_float.unsqueeze(2)).squeeze(2)
        coefficients = (torch.linalg.pinv(gram) @ rhs.unsqueeze(2)).squeeze(2)
        projection = (coefficients.unsqueeze(2) * basis).sum(dim=1)
    metrics["component_grad/joint_span/retained_fraction"] = (
        projection.norm(dim=1) / pred_norm.clamp_min(eps)
    ).mean()

    raw_sum = torch.stack(raw_components, dim=1).sum(dim=1)
    sum_of_norms = torch.stack(
        [gradient.norm(dim=1) for gradient in raw_components], dim=1
    ).sum(dim=1)
    metrics["component_grad/joint_span/cancellation_ratio"] = (
        raw_sum.norm(dim=1) / sum_of_norms.clamp_min(eps)
    ).mean()
    return metrics
