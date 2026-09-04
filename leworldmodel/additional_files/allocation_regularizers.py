"""Capacity-allocation regularizers for the Obsessed Encoder experiments.

The published anti-collapse term constrains the global distribution of the
embedding.  These losses instead look at geometry that an episode-constant,
low-dimensional shortcut cannot satisfy:

* ``conditional`` requires variation and decorrelation *within* trajectories.
* ``local_rank`` requires neighborhoods of the embedding manifold to use more
  than a few local directions.

All neighbor selection is detached. Gradients flow through the selected local
differences, but not through the discrete top-k operation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    n, m = matrix.shape
    if n != m:
        raise ValueError(f"expected a square matrix, got {tuple(matrix.shape)}")
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class AllocationRegularizer(nn.Module):
    """Regularize conditional coverage and/or local intrinsic rank.

    Args:
        mode: ``conditional``, ``local_rank``, or ``hybrid``.
        variance_target: minimum per-coordinate standard deviation after the
            per-episode temporal mean has been removed.
        variance_weight: weight of the conditional variance hinge.
        covariance_weight: weight of conditional off-diagonal correlation.
        local_rank_target: minimum effective rank within a k-NN neighborhood.
        local_rank_weight: weight of the local-rank hinge.
        neighbors: number of neighbors used by the local-rank estimate.
    """

    MODES = {"conditional", "local_rank", "hybrid"}

    def __init__(
        self,
        *,
        mode: str,
        variance_target: float = 0.20,
        variance_weight: float = 1.0,
        covariance_weight: float = 0.01,
        local_rank_target: float = 8.0,
        local_rank_weight: float = 0.10,
        neighbors: int = 16,
        eps: float = 1e-4,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(self.MODES)}")
        if neighbors < 2:
            raise ValueError("neighbors must be at least 2")
        self.mode = mode
        self.variance_target = variance_target
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.local_rank_target = local_rank_target
        self.local_rank_weight = local_rank_weight
        self.neighbors = neighbors
        self.eps = eps

    def conditional_terms(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Variance and correlation terms within trajectories."""
        # Removing each trajectory's temporal mean makes episode-constant tags
        # unable to satisfy this term while leaving the original embedding free
        # to retain static goals and obstacles.
        residual = emb.float() - emb.float().mean(dim=1, keepdim=True)
        flat = residual.reshape(-1, residual.size(-1))
        std = torch.sqrt(flat.var(dim=0, unbiased=False) + self.eps)
        variance = F.relu(self.variance_target - std).mean()

        normalized = (flat - flat.mean(dim=0)) / std.clamp_min(self.eps)
        denom = max(normalized.size(0) - 1, 1)
        corr = normalized.T @ normalized / denom
        covariance = _off_diagonal(corr).square().sum() / corr.size(0)
        return variance, covariance

    def local_rank_term(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Hinge on neighborhood effective rank, plus the measured mean rank."""
        points = emb.float().reshape(-1, emb.size(-1))
        count = points.size(0)
        k = min(self.neighbors, count - 1)
        if k < 2:
            zero = points.sum() * 0.0
            return zero, zero.detach()

        # Pairwise distances are used only to select neighborhoods. Detaching
        # avoids pretending top-k is differentiable and saves its autograd graph.
        with torch.no_grad():
            distances = torch.cdist(points.detach(), points.detach())
            indices = distances.topk(k + 1, largest=False).indices[:, 1:]

        neighbors = points[indices]
        differences = neighbors - points[:, None, :]
        differences = differences - differences.mean(dim=1, keepdim=True)

        # The k x k Gram matrix has the same non-zero spectrum as the much
        # larger D x D local covariance matrix.
        # Autocast may downcast the matmul result to BF16 even though ``points``
        # was explicitly promoted above. CUDA eigendecomposition does not
        # support BF16, and the spectrum is more stable in FP32 in any case.
        with torch.autocast(device_type=emb.device.type, enabled=False):
            differences_fp32 = differences.float()
            gram = differences_fp32 @ differences_fp32.transpose(1, 2)
            gram = gram / max(points.size(-1), 1)
            # Clamp only numerical negative eigenvalues. Giving every null
            # direction a fixed positive floor would make the estimate depend
            # on embedding scale and let a collapsed representation appear
            # full-rank simply by shrinking below that floor.
            eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        spectral_mass = eigenvalues.sum(dim=-1, keepdim=True)
        tiny = torch.finfo(eigenvalues.dtype).tiny
        probabilities = eigenvalues / spectral_mass.clamp_min(tiny)
        entropy = -torch.xlogy(probabilities, probabilities).sum(dim=-1)
        effective_rank = torch.where(
            spectral_mass.squeeze(-1) > tiny,
            entropy.exp(),
            torch.zeros_like(entropy),
        )
        loss = F.relu(self.local_rank_target - effective_rank).mean()
        return loss, effective_rank.mean().detach()

    def forward(self, emb: torch.Tensor) -> dict[str, torch.Tensor]:
        if emb.ndim != 3:
            raise ValueError(f"expected emb shaped (B,T,D), got {tuple(emb.shape)}")

        zero = emb.sum() * 0.0
        variance = covariance = local_rank = zero
        measured_rank = zero.detach()

        if self.mode in {"conditional", "hybrid"}:
            variance, covariance = self.conditional_terms(emb)
        if self.mode in {"local_rank", "hybrid"}:
            local_rank, measured_rank = self.local_rank_term(emb)

        total = (
            self.variance_weight * variance
            + self.covariance_weight * covariance
            + self.local_rank_weight * local_rank
        )
        return {
            "allocation_loss": total,
            "conditional_variance_loss": variance,
            "conditional_covariance_loss": covariance,
            "local_rank_loss": local_rank,
            "local_effective_rank": measured_rank,
        }
