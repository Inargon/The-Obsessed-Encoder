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
    routing_mode: str = "aligned",
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
    if routing_mode not in {
        "aligned",
        "norm_matched_scalar",
        "retention_matched_shuffled",
    }:
        raise ValueError(f"unknown routing_mode {routing_mode!r}")

    pred_grad = torch.autograd.grad(
        pred_loss, embedding, retain_graph=True, create_graph=False
    )[0].detach()
    true_control_grad = torch.autograd.grad(
        control_loss, embedding, retain_graph=True, create_graph=False
    )[0].detach()

    view_shape = (-1, *([1] * (embedding.ndim - 1)))
    pred_flat = pred_grad.float().flatten(1)
    pred_norm = pred_flat.norm(dim=1)

    def route_with(guide_grad: torch.Tensor):
        guide_flat = guide_grad.float().flatten(1)
        dot = (pred_flat * guide_flat).sum(dim=1)
        guide_norm = guide_flat.norm(dim=1)
        cosine = dot / (pred_norm * guide_norm).clamp_min(eps)
        coefficient = dot / guide_flat.square().sum(dim=1).clamp_min(eps)
        parallel = coefficient.to(embedding.dtype).view(view_shape) * guide_grad
        orthogonal = pred_grad - parallel
        positive_parallel = (
            coefficient.clamp_min(0).to(embedding.dtype).view(view_shape)
            * guide_grad
        )
        if orthogonal_mode == "drop":
            gate = torch.zeros_like(cosine, dtype=embedding.dtype)
        else:
            gate = cosine.clamp(min=0, max=1).to(embedding.dtype)
        safe = positive_parallel + gate.view(view_shape) * orthogonal
        return safe, cosine, gate, dot

    def match_reference_norm(
        candidate: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        candidate_norm = candidate.float().flatten(1).norm(dim=1)
        reference_norm = reference.float().flatten(1).norm(dim=1)
        fallback_norm = pred_norm.clamp_min(eps)
        candidate_unit = candidate / candidate_norm.clamp_min(eps).to(
            embedding.dtype
        ).view(view_shape)
        fallback_unit = pred_grad / fallback_norm.to(embedding.dtype).view(view_shape)
        direction = torch.where(
            (candidate_norm > eps).view(view_shape), candidate_unit, fallback_unit
        )
        return direction * reference_norm.to(embedding.dtype).view(view_shape)

    reference_safe, true_cosine, true_gate, true_dot = route_with(true_control_grad)
    control_grad = true_control_grad
    guide_shuffled = False

    if routing_mode == "norm_matched_scalar":
        safe_pred_grad = match_reference_norm(pred_grad, reference_safe)
        cosine, orthogonal_gate, dot = true_cosine, true_gate, true_dot
    elif routing_mode == "retention_matched_shuffled" and embedding.size(0) > 1:
        # Use another sample's control-gradient axis. Flip its sign toward the
        # prediction gradient so the candidate is non-zero, then match the
        # exact per-sample norm of the true aligned update. This isolates axis
        # correspondence from the amount of prediction gradient retained.
        shift = int(torch.randint(1, embedding.size(0), (), device=embedding.device))
        control_grad = true_control_grad.roll(shift, dims=0)
        raw_dot = (pred_flat * control_grad.float().flatten(1)).sum(dim=1)
        sign = torch.where(raw_dot < 0, -1.0, 1.0).to(embedding.dtype)
        oriented_guide = control_grad * sign.view(view_shape)
        candidate, cosine, orthogonal_gate, _ = route_with(oriented_guide)
        safe_pred_grad = match_reference_norm(candidate, reference_safe)
        dot = raw_dot
        guide_shuffled = True
    else:
        if shuffle_control and embedding.size(0) > 1:
            # Legacy unmatched negative control.
            shift = int(
                torch.randint(1, embedding.size(0), (), device=embedding.device)
            )
            control_grad = true_control_grad.roll(shift, dims=0)
            guide_shuffled = True
        safe_pred_grad, cosine, orthogonal_gate, dot = route_with(control_grad)

    correction = safe_pred_grad - pred_grad

    # Zero forward value; exact detached gradient correction at embedding.
    surrogate = (correction * (embedding - embedding.detach())).sum()

    safe_norm = safe_pred_grad.float().flatten(1).norm(dim=1)
    reference_norm = reference_safe.float().flatten(1).norm(dim=1)
    reference_flat = reference_safe.float().flatten(1)
    direction_cosine = (safe_pred_grad.float().flatten(1) * reference_flat).sum(1) / (
        safe_norm * reference_norm
    ).clamp_min(eps)
    diagnostics = {
        "prediction_control_grad_cosine": cosine.mean(),
        "prediction_orthogonal_gate": orthogonal_gate.float().mean(),
        "prediction_grad_retained_fraction": (
            safe_norm / pred_norm.clamp_min(eps)
        ).mean(),
        "prediction_reversed_fraction": (dot < 0).float().mean(),
        "prediction_guide_shuffled": torch.tensor(float(guide_shuffled), device=embedding.device),
        "prediction_reference_retained_fraction": (
            reference_norm / pred_norm.clamp_min(eps)
        ).mean(),
        "prediction_norm_match_error": (
            (safe_norm - reference_norm).abs() / reference_norm.clamp_min(eps)
        ).mean(),
        "prediction_direction_cosine_to_aligned": direction_cosine.mean(),
    }
    return surrogate, diagnostics
