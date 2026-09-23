"""History-aware routing through an empirical control-relevance subspace."""

from __future__ import annotations

import math

import torch
from torch import nn


class DecisionSubspaceRouter(nn.Module):
    """Keep a low-rank sketch of recent control gradients and route through it."""

    def __init__(
        self,
        rank: int = 16,
        decay: float = 0.99,
        update_every: int = 20,
        candidates_per_update: int = 8,
        minimum_retention: float = 0.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if update_every <= 0:
            raise ValueError(f"update_every must be positive, got {update_every}")
        if candidates_per_update <= 0:
            raise ValueError(
                "candidates_per_update must be positive, "
                f"got {candidates_per_update}"
            )
        if not 0.0 <= minimum_retention <= 1.0:
            raise ValueError(
                "minimum_retention must be in [0, 1], "
                f"got {minimum_retention}"
            )
        self.rank = int(rank)
        self.decay = float(decay)
        self.update_every = int(update_every)
        self.candidates_per_update = int(candidates_per_update)
        self.minimum_retention = float(minimum_retention)
        self.eps = float(eps)
        # Optimizer-side statistics: excluded from inference checkpoints.
        self.register_buffer("sketch", torch.empty(0), persistent=False)
        self.register_buffer(
            "num_calls", torch.zeros((), dtype=torch.long), persistent=False
        )

    @property
    def basis(self) -> torch.Tensor:
        if self.sketch.numel() == 0:
            return self.sketch
        norms = self.sketch.float().norm(dim=1)
        valid = norms > self.eps
        return self.sketch[valid] / norms[valid, None].to(self.sketch.dtype)

    @torch.no_grad()
    def _update_sketch(self, control_flat: torch.Tensor) -> None:
        rows = control_flat.detach().float()
        norms = rows.norm(dim=1)
        valid = norms > self.eps
        rows, norms = rows[valid], norms[valid]
        if rows.numel() == 0:
            return
        rows = rows / norms[:, None]
        if rows.size(0) > self.candidates_per_update:
            indices = torch.linspace(
                0,
                rows.size(0) - 1,
                self.candidates_per_update,
                device=rows.device,
            ).round().long()
            rows = rows.index_select(0, indices)
        rows = rows / math.sqrt(float(rows.size(0)))
        if self.sketch.numel() == 0 or self.sketch.size(1) != rows.size(1):
            stacked = rows
        else:
            stacked = torch.cat(
                (
                    math.sqrt(self.decay) * self.sketch.float(),
                    math.sqrt(1.0 - self.decay) * rows,
                ),
                dim=0,
            )
        _, singular_values, right = torch.linalg.svd(stacked, full_matrices=False)
        keep = min(self.rank, right.size(0))
        self.sketch = (
            singular_values[:keep, None] * right[:keep]
        ).to(control_flat.device)

    @staticmethod
    def _project(flat: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        if basis.numel() == 0:
            return torch.zeros_like(flat)
        basis = basis.to(device=flat.device, dtype=flat.dtype)
        return (flat @ basis.transpose(0, 1)) @ basis

    def forward(
        self,
        pred_loss: torch.Tensor,
        control_loss: torch.Tensor,
        embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_grad = torch.autograd.grad(
            pred_loss, embedding, retain_graph=True, create_graph=False
        )[0].detach()
        control_grad = torch.autograd.grad(
            control_loss, embedding, retain_graph=True, create_graph=False
        )[0].detach()
        pred_flat = pred_grad.float().flatten(1)
        control_flat = control_grad.float().flatten(1)
        pred_norm = pred_flat.norm(dim=1)

        call = int(self.num_calls.item())
        if call % self.update_every == 0 or self.sketch.numel() == 0:
            self._update_sketch(control_flat)
        self.num_calls.add_(1)

        basis = self.basis
        supported_pred = self._project(pred_flat, basis)
        supported_control = self._project(control_flat, basis)
        # PCGrad inside the learned decision subspace.
        dot = (supported_pred * supported_control).sum(dim=1)
        guide_sq = supported_control.square().sum(dim=1).clamp_min(self.eps)
        conflict_coefficient = (dot / guide_sq).clamp_max(0.0)
        safe_flat = supported_pred - conflict_coefficient[:, None] * supported_control
        if self.minimum_retention:
            safe_flat = (
                self.minimum_retention * pred_flat
                + (1.0 - self.minimum_retention) * safe_flat
            )

        safe_pred_grad = safe_flat.to(embedding.dtype).reshape_as(embedding)
        correction = safe_pred_grad - pred_grad
        surrogate = (correction * (embedding - embedding.detach())).sum()

        safe_norm = safe_flat.norm(dim=1)
        supported_norm = supported_pred.norm(dim=1)
        control_norm = control_flat.norm(dim=1)
        raw_cosine = (pred_flat * control_flat).sum(dim=1) / (
            pred_norm * control_norm
        ).clamp_min(self.eps)
        diagnostics = {
            "prediction_control_grad_cosine": raw_cosine.mean(),
            "prediction_grad_retained_fraction": (
                safe_norm / pred_norm.clamp_min(self.eps)
            ).mean(),
            "prediction_reversed_fraction": (dot < 0.0).float().mean(),
            "decision_subspace_rank": torch.tensor(
                float(basis.size(0)), device=embedding.device
            ),
            "decision_subspace_prediction_coverage": (
                supported_norm / pred_norm.clamp_min(self.eps)
            ).mean(),
            "decision_subspace_control_coverage": (
                supported_control.norm(dim=1)
                / control_norm.clamp_min(self.eps)
            ).mean(),
        }
        return surrogate, diagnostics

