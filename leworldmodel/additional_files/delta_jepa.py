"""Latent-difference action decoding for a matched Delta-JEPA baseline.

The decoder follows the defining mechanism of Delta-JEPA: an action sequence
is reconstructed from the endpoint displacement ``z[t+h] - z[t]`` rather
than from concatenated endpoint embeddings.  A bank of learned action queries
is modulated by the displacement and processed by a non-causal Transformer.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LatentDifferenceActionDecoder(nn.Module):
    """Decode one or more coarse action blocks from a latent displacement."""

    def __init__(
        self,
        *,
        embed_dim: int,
        action_dim: int,
        max_horizon: int = 3,
        decoder_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.0,
        action_weight: float = 10.0,
    ) -> None:
        super().__init__()
        if max_horizon < 1:
            raise ValueError("max_horizon must be positive")
        if decoder_dim % num_heads:
            raise ValueError("decoder_dim must be divisible by num_heads")
        if action_weight <= 0:
            raise ValueError("action_weight must be positive")

        self.action_dim = action_dim
        self.max_horizon = max_horizon
        self.action_weight = action_weight
        self.action_queries = nn.Parameter(
            torch.empty(max_horizon, decoder_dim)
        )
        nn.init.trunc_normal_(self.action_queries, std=0.02)
        self.condition = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 2 * decoder_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(decoder_dim),
        )
        self.action_head = nn.Linear(decoder_dim, action_dim)

    @staticmethod
    def _action_sequence(actions: torch.Tensor, horizon: int) -> torch.Tensor:
        """Return aligned targets with shape ``(B, T-h, h, A)``."""
        length = actions.size(1) - horizon
        return torch.stack(
            [actions[:, offset : offset + length] for offset in range(horizon)],
            dim=-2,
        )

    def decode(self, displacement: torch.Tensor, horizon: int) -> torch.Tensor:
        if not 1 <= horizon <= self.max_horizon:
            raise ValueError(
                f"horizon must be in [1, {self.max_horizon}], got {horizon}"
            )
        shape = displacement.shape[:-1]
        flat = displacement.reshape(-1, displacement.size(-1))
        scale, shift = self.condition(flat).chunk(2, dim=-1)
        queries = self.action_queries[:horizon].unsqueeze(0).expand(
            flat.size(0), -1, -1
        )
        # AdaLN-style conditioning: the displacement controls every learned
        # action query while the Transformer reasons jointly over the sequence.
        tokens = queries * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        decoded = self.action_head(self.decoder(tokens))
        return decoded.reshape(*shape, horizon, self.action_dim)

    def forward(
        self, emb: torch.Tensor, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if emb.ndim != 3:
            raise ValueError(f"expected emb shaped (B,T,D), got {tuple(emb.shape)}")
        actions = actions.float().reshape(actions.size(0), actions.size(1), -1)
        if actions.size(-1) != self.action_dim:
            raise ValueError(
                f"expected action dim {self.action_dim}, got {actions.size(-1)}"
            )
        max_horizon = min(
            self.max_horizon,
            emb.size(1) - 1,
            actions.size(1) - 1,
        )
        if max_horizon < 1:
            zero = emb.sum() * 0.0
            return {
                "delta_action_loss": zero,
                "delta_jepa_loss": zero,
            }

        losses = []
        metrics: dict[str, torch.Tensor] = {}
        for horizon in range(1, max_horizon + 1):
            displacement = emb[:, horizon:] - emb[:, :-horizon]
            target = self._action_sequence(actions, horizon)
            prediction = self.decode(displacement, horizon)
            loss = F.mse_loss(prediction, target)
            losses.append(loss)
            metrics[f"delta_horizon_{horizon}_loss"] = loss

            target_variance = target.float().var(unbiased=False).clamp_min(1e-8)
            metrics[f"delta_horizon_{horizon}_mse_over_action_variance"] = (
                loss.detach().float() / target_variance.detach()
            )

        action_loss = torch.stack(losses).mean()
        return {
            "delta_action_loss": action_loss,
            "delta_jepa_loss": self.action_weight * action_loss,
            **metrics,
        }
