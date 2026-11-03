"""Experimental action-effect density ratio for continuous action chunks.

This is an opt-in pilot, not an estimate of goal-reaching probability.  The
score estimates log q(a | z_start, z_end) - log q(a | z_start).  Bayes' rule
identifies it with a transition likelihood ratio only when both conditional
action densities are calibrated to the same behavior distribution.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _network(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


def diagonal_gaussian_log_prob(
    actions: torch.Tensor, parameters: torch.Tensor
) -> torch.Tensor:
    """Return one log density per action chunk, with bounded log scales."""
    mean, raw_log_scale = parameters.chunk(2, dim=-1)
    if mean.shape != actions.shape:
        raise ValueError("action and Gaussian parameter dimensions must match")
    log_scale = raw_log_scale.clamp(-5.0, 2.0)
    standardized = (actions - mean) * torch.exp(-log_scale)
    return -0.5 * (standardized.square() + math.log(2 * math.pi)).sum(-1) - log_scale.sum(-1)


class ActionEffectRatio(nn.Module):
    """Estimate an action-conditioned transition density ratio in latent space.

    The prior sees a detached start latent, so its likelihood objective cannot
    shape the shared encoder.  The posterior does shape both endpoint latents.
    Optional matched nuisance interventions penalize changes to the *score*,
    not all information in either latent.  They must preserve physical state.
    """

    def __init__(self, embed_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        if embed_dim < 1 or action_dim < 1:
            raise ValueError("embed_dim and action_dim must be positive")
        self.action_dim = action_dim
        self.posterior = _network(2 * embed_dim, hidden_dim, 2 * action_dim)
        self.prior = _network(embed_dim, hidden_dim, 2 * action_dim)

    def score(
        self, start: torch.Tensor, end: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if start.shape != end.shape or start.shape[:-1] != actions.shape[:-1]:
            raise ValueError("start, end, and actions must have matching batch axes")
        if actions.shape[-1] != self.action_dim:
            raise ValueError("incorrect action chunk dimension")
        posterior_logp = diagonal_gaussian_log_prob(
            actions, self.posterior(torch.cat((start, end), dim=-1))
        )
        prior_logp = diagonal_gaussian_log_prob(
            actions, self.prior(start.detach())
        )
        return posterior_logp - prior_logp

    def forward(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        actions: torch.Tensor,
        *,
        counterfactual_start: torch.Tensor | None = None,
        counterfactual_end: torch.Tensor | None = None,
        consistency_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if consistency_weight < 0:
            raise ValueError("consistency_weight must be non-negative")
        posterior_params = self.posterior(torch.cat((start, end), dim=-1))
        prior_params = self.prior(start.detach())
        posterior_nll = -diagonal_gaussian_log_prob(actions, posterior_params).mean()
        prior_nll = -diagonal_gaussian_log_prob(actions, prior_params).mean()
        ratio = diagonal_gaussian_log_prob(actions, posterior_params) - diagonal_gaussian_log_prob(actions, prior_params)
        consistency = ratio.new_zeros(())
        if (counterfactual_start is None) != (counterfactual_end is None):
            raise ValueError("supply both counterfactual endpoints or neither")
        if counterfactual_start is not None:
            changed_ratio = self.score(counterfactual_start, counterfactual_end, actions)
            consistency = F.mse_loss(ratio, changed_ratio)
        return {
            "action_effect_loss": posterior_nll + prior_nll + consistency_weight * consistency,
            "action_effect_posterior_nll": posterior_nll,
            "action_effect_prior_nll": prior_nll,
            "action_effect_consistency_loss": consistency,
            "action_effect_score": ratio,
        }
