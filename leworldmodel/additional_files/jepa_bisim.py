"""Reward-free bisimulation objective for a matched JEPA-Bisim baseline.

This is the control-relevant metric used by the matched baseline campaign.  A
randomly paired state distance is trained to equal the discounted distance
between the corresponding action-conditioned successor predictions.  The
variance and covariance terms follow the public JEPA-Bisim implementation and
prevent the trivial constant representation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RewardFreeBisimulationObjective(nn.Module):
    """Match latent distances to action-conditioned transition distances."""

    def __init__(
        self,
        *,
        discount: float = 0.99,
        weight: float = 1.0,
        variance_weight: float = 1.0,
        covariance_weight: float = 1.0,
        variance_target: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        if min(weight, variance_weight, covariance_weight) < 0.0:
            raise ValueError("loss weights must be non-negative")
        if variance_target <= 0.0 or eps <= 0.0:
            raise ValueError("variance_target and eps must be positive")
        self.discount = discount
        self.weight = weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.variance_target = variance_target
        self.eps = eps

    def _variance_covariance(
        self, current: torch.Tensor, successor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat((current, successor), dim=1).reshape(
            -1, current.size(-1)
        ).float()
        centered = features - features.mean(dim=0, keepdim=True)
        std = torch.sqrt(centered.var(dim=0, unbiased=False) + self.eps)
        variance = F.relu(self.variance_target - std).mean()

        denominator = max(features.size(0) - 1, 1)
        covariance = centered.T @ centered / denominator
        diagonal = torch.diagonal(covariance)
        off_diagonal = covariance - torch.diag_embed(diagonal)
        covariance_loss = off_diagonal.square().sum() / max(
            covariance.numel() - diagonal.numel(), 1
        )
        return variance, covariance_loss

    def forward(
        self, emb: torch.Tensor, predicted_successor: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if emb.ndim != 3 or predicted_successor.ndim != 3:
            raise ValueError("emb and predicted_successor must have shape (B,T,D)")
        if emb.size(0) != predicted_successor.size(0):
            raise ValueError("embedding and prediction batch sizes differ")
        length = min(emb.size(1) - 1, predicted_successor.size(1))
        if length < 1 or emb.size(0) < 2:
            zero = emb.sum() * 0.0
            return {
                "bisim_metric_loss": zero,
                "bisim_variance_loss": zero,
                "bisim_covariance_loss": zero,
                "bisim_loss": zero,
            }

        current = emb[:, :length]
        successor = predicted_successor[:, :length]
        # Public JEPA-Bisim samples a batch permutation.  Rolling by a random
        # non-zero offset gives the same cross-batch pairing without fixed
        # points, which otherwise contribute a vacuous zero-distance pair.
        offset = int(
            torch.randint(
                1, emb.size(0), (), device=emb.device
            ).item()
        )
        paired_current = current.roll(offset, dims=0)
        paired_successor = successor.roll(offset, dims=0)

        state_distance = F.smooth_l1_loss(
            current, paired_current, reduction="none"
        ).sum(dim=-1)
        transition_distance = torch.sqrt(
            (successor - paired_successor)
            .float()
            .square()
            .sum(dim=-1)
            .mean(dim=-1, keepdim=True)
            + self.eps
        ).expand_as(state_distance)
        target_distance = self.discount * transition_distance
        metric_loss = (state_distance - target_distance).square().mean()
        variance_loss, covariance_loss = self._variance_covariance(
            current, successor
        )
        total = self.weight * (
            metric_loss
            + self.variance_weight * variance_loss
            + self.covariance_weight * covariance_loss
        )
        return {
            "bisim_metric_loss": metric_loss,
            "bisim_variance_loss": variance_loss,
            "bisim_covariance_loss": covariance_loss,
            "bisim_state_distance": state_distance.mean().detach(),
            "bisim_transition_distance": transition_distance.mean().detach(),
            "bisim_loss": total,
        }
