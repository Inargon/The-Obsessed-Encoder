"""Control-aligned routing of prediction gradients at the representation."""

from __future__ import annotations

import torch


def control_aligned_prediction_surrogate(
    pred_loss: torch.Tensor,
    control_loss: torch.Tensor,
    embedding: torch.Tensor,
    eps: float = 1e-12,
    orthogonal_mode: str = "cosine",
    shuffle_control: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Protect control-aligned prediction and attenuate orthogonal capacity use.

    For every sample, decompose the prediction gradient into a component
    parallel to the control gradient and an orthogonal component. A positive
    parallel component is retained in full, a negative one is removed, and the
    orthogonal component is gated by the non-negative gradient cosine. The
    returned term is zero-valued but supplies the correction on backward, so
    the prediction head still receives its complete prediction gradient.
    """
    if orthogonal_mode not in {"cosine", "drop"}:
        raise ValueError(
            "orthogonal_mode must be either 'cosine' or 'drop', "
            f"got {orthogonal_mode!r}"
        )

    pred_grad = torch.autograd.grad(
        pred_loss, embedding, retain_graph=True, create_graph=False
    )[0].detach()
    control_grad = torch.autograd.grad(
        control_loss, embedding, retain_graph=True, create_graph=False
    )[0].detach()
    if shuffle_control and embedding.size(0) > 1:
        # A deliberately wrong guide for the negative-control experiment.
        # Sampling a non-zero cyclic shift preserves the guide distribution
        # and norm while breaking its correspondence with each observation.
        shift = int(torch.randint(1, embedding.size(0), (), device=embedding.device))
        control_grad = control_grad.roll(shift, dims=0)

    pred_flat = pred_grad.float().flatten(1)
    control_flat = control_grad.float().flatten(1)
    dot = (pred_flat * control_flat).sum(dim=1)
    pred_norm = pred_flat.norm(dim=1)
    control_norm = control_flat.norm(dim=1)
    cosine = dot / (pred_norm * control_norm).clamp_min(eps)

    control_norm_sq = control_flat.square().sum(dim=1).clamp_min(eps)
    coefficient = dot / control_norm_sq
    view_shape = (-1, *([1] * (embedding.ndim - 1)))
    coefficient_view = coefficient.to(embedding.dtype).view(view_shape)
    parallel = coefficient_view * control_grad
    orthogonal = pred_grad - parallel

    positive_coefficient = coefficient.clamp_min(0).to(embedding.dtype)
    positive_parallel = positive_coefficient.view(view_shape) * control_grad
    if orthogonal_mode == "drop":
        orthogonal_gate = torch.zeros_like(cosine, dtype=embedding.dtype)
    else:
        orthogonal_gate = cosine.clamp(min=0, max=1).to(embedding.dtype)
    safe_pred_grad = positive_parallel + orthogonal_gate.view(view_shape) * orthogonal
    correction = safe_pred_grad - pred_grad

    # Zero forward value; exact detached gradient correction at embedding.
    surrogate = (correction * (embedding - embedding.detach())).sum()

    safe_norm = safe_pred_grad.float().flatten(1).norm(dim=1)
    diagnostics = {
        "prediction_control_grad_cosine": cosine.mean(),
        "prediction_orthogonal_gate": orthogonal_gate.float().mean(),
        "prediction_grad_retained_fraction": (
            safe_norm / pred_norm.clamp_min(eps)
        ).mean(),
        "prediction_reversed_fraction": (dot < 0).float().mean(),
        "prediction_guide_shuffled": torch.tensor(
            float(shuffle_control), device=embedding.device
        ),
    }
    return surrogate, diagnostics
