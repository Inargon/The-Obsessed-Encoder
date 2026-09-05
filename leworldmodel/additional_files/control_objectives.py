"""Control-sufficiency objectives for shortcut-resistant world models.

The objectives deliberately use within-episode endpoints. Episode-constant
features therefore cannot identify which endpoint follows an action sequence.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class ControlObjective(nn.Module):
    """Multi-horizon inverse dynamics with optional masking and reachability.

    ``multi_horizon_idm`` predicts action chunks from real endpoint pairs.
    ``masked_reachability`` additionally masks a shared random latent subspace,
    closes an action cycle through the JEPA predictor, and identifies the true
    endpoint among hard negatives from the same episode.
    """

    MODES = {
        "multi_horizon_idm",
        "masked_reachability",
        "direct_reachability",
        "factorized_reachability",
    }

    def __init__(
        self,
        *,
        embed_dim: int,
        action_dim: int,
        mode: str,
        max_horizon: int = 3,
        hidden_dim: int = 256,
        inverse_weight: float = 1.0,
        cycle_weight: float = 0.5,
        reachability_weight: float = 0.1,
        mask_keep_prob: float = 0.5,
        temperature: float = 0.1,
        context_dim: int = 32,
        context_weight: float = 0.1,
        static_leak_weight: float = 0.1,
        dynamic_variance_target: float = 0.2,
        dynamic_variance_weight: float = 1.0,
        dynamic_covariance_weight: float = 0.01,
        eps: float = 1e-4,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(self.MODES)}")
        if max_horizon < 1:
            raise ValueError("max_horizon must be positive")
        if not 0.0 < mask_keep_prob <= 1.0:
            raise ValueError("mask_keep_prob must be in (0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if mode == "factorized_reachability" and not 0 < context_dim < embed_dim:
            raise ValueError("context_dim must be between zero and embed_dim")

        self.mode = mode
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.max_horizon = max_horizon
        self.inverse_weight = inverse_weight
        self.cycle_weight = cycle_weight
        self.reachability_weight = reachability_weight
        self.mask_keep_prob = mask_keep_prob
        self.temperature = temperature
        self.context_dim = context_dim if mode == "factorized_reachability" else 0
        self.context_weight = context_weight
        self.static_leak_weight = static_leak_weight
        self.dynamic_variance_target = dynamic_variance_target
        self.dynamic_variance_weight = dynamic_variance_weight
        self.dynamic_covariance_weight = dynamic_covariance_weight
        self.eps = eps

        control_dim = embed_dim - self.context_dim
        self.control_dim = control_dim

        self.inverse_heads = nn.ModuleDict({
            # At longer lags, exact action sequences are not identifiable from
            # endpoints (many paths share an endpoint). Predict the mean
            # control over the interval; h=1 remains the original IDM target.
            str(h): _mlp(2 * control_dim, hidden_dim, action_dim)
            for h in range(1, max_horizon + 1)
        })
        # Learned query/key heads are used only by the masked ablation. The
        # direct method below intentionally has no projection head: its
        # contrastive margins must be realized in the embedding consumed by
        # the planner itself, not quarantined in a small auxiliary subspace.
        if mode == "masked_reachability":
            self.reach_queries = nn.ModuleDict({
                str(h): _mlp(control_dim + h * action_dim, hidden_dim, control_dim)
                for h in range(1, max_horizon + 1)
            })
            self.reach_key = _mlp(control_dim, hidden_dim, control_dim)
        else:
            self.reach_queries = nn.ModuleDict()
            self.reach_key = nn.Identity()

    @property
    def uses_mask(self) -> bool:
        return self.mode == "masked_reachability"

    def _paired_mask(
        self, start: torch.Tensor, end: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (self.uses_mask and self.training):
            return start, end
        keep = self.mask_keep_prob
        mask = (torch.rand_like(start) < keep).to(start.dtype) / keep
        return start * mask, end * mask

    @staticmethod
    def _action_chunk(actions: torch.Tensor, horizon: int) -> torch.Tensor:
        length = actions.size(1) - horizon
        return torch.cat(
            [actions[:, offset : offset + length] for offset in range(horizon)],
            dim=-1,
        )

    def forward(
        self,
        emb: torch.Tensor,
        actions: torch.Tensor,
        pred_emb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if emb.ndim != 3:
            raise ValueError(f"expected emb shaped (B,T,D), got {tuple(emb.shape)}")
        if emb.size(-1) != self.embed_dim:
            raise ValueError(f"expected embedding dim {self.embed_dim}, got {emb.size(-1)}")

        actions = actions.float().reshape(actions.size(0), actions.size(1), -1)
        if actions.size(-1) != self.action_dim:
            raise ValueError(f"expected action dim {self.action_dim}, got {actions.size(-1)}")

        zero = emb.sum() * 0.0
        context_consistency = zero
        dynamic_static = zero
        dynamic_variance = zero
        dynamic_covariance = zero
        control_emb = emb
        control_pred = pred_emb

        if self.mode == "factorized_reachability":
            context = emb[..., : self.context_dim]
            dynamics = emb[..., self.context_dim :]
            context_consistency = (context[:, 1:] - context[:, :-1]).square().mean()

            dynamic_mean = dynamics.mean(dim=1, keepdim=True)
            residual = dynamics - dynamic_mean
            static_var = dynamic_mean.squeeze(1).var(dim=0, unbiased=False)
            changing_var = residual.reshape(-1, residual.size(-1)).var(
                dim=0, unbiased=False
            )
            dynamic_static = (
                static_var / (static_var + changing_var + self.eps)
            ).mean()

            flat = residual.reshape(-1, residual.size(-1)).float()
            std = torch.sqrt(flat.var(dim=0, unbiased=False) + self.eps)
            dynamic_variance = F.relu(self.dynamic_variance_target - std).mean()
            normalized = (flat - flat.mean(dim=0)) / std.clamp_min(self.eps)
            denom = max(normalized.size(0) - 1, 1)
            corr = normalized.T @ normalized / denom
            off_diagonal = corr.flatten()[:-1].view(
                corr.size(0) - 1, corr.size(0) + 1
            )[:, 1:].flatten()
            dynamic_covariance = off_diagonal.square().sum() / corr.size(0)

            control_emb = residual
            if pred_emb is not None:
                control_pred = (
                    pred_emb[..., self.context_dim :]
                    - dynamic_mean.expand(-1, pred_emb.size(1), -1)
                )

        max_horizon = min(self.max_horizon, control_emb.size(1) - 1)
        if max_horizon < 1:
            return {
                "control_loss": zero,
                "inverse_dynamics_loss": zero,
                "action_cycle_loss": zero,
                "reachability_loss": zero,
                "reachability_accuracy": zero.detach(),
                "context_consistency_loss": context_consistency,
                "dynamic_static_leak_loss": dynamic_static,
                "dynamic_variance_loss": dynamic_variance,
                "dynamic_covariance_loss": dynamic_covariance,
            }

        inverse_terms = []
        inverse_by_horizon = {}
        reach_terms = []
        reach_correct = []
        for horizon in range(1, max_horizon + 1):
            start = control_emb[:, :-horizon]
            end = control_emb[:, horizon:]
            target_actions = self._action_chunk(actions, horizon)
            target_mean_action = target_actions.reshape(
                *target_actions.shape[:-1], horizon, self.action_dim
            ).mean(dim=-2)

            masked_start, masked_end = self._paired_mask(start, end)
            predicted_actions = self.inverse_heads[str(horizon)](
                torch.cat((masked_start, masked_end), dim=-1)
            )
            inverse_term = F.smooth_l1_loss(predicted_actions, target_mean_action)
            inverse_terms.append(inverse_term)
            inverse_by_horizon[f"inverse_horizon_{horizon}_loss"] = inverse_term

            if self.mode == "masked_reachability":
                for time_index in range(emb.size(1) - horizon):
                    query_start = emb[:, time_index]
                    candidates = emb
                    if self.training:
                        keep = self.mask_keep_prob
                        reach_mask = (
                            torch.rand_like(query_start) < keep
                        ).to(query_start.dtype) / keep
                        query_start = query_start * reach_mask
                        candidates = candidates * reach_mask[:, None, :]
                    keys = F.normalize(self.reach_key(candidates), dim=-1)
                    query_input = torch.cat(
                        (query_start, target_actions[:, time_index]), dim=-1
                    )
                    query = F.normalize(
                        self.reach_queries[str(horizon)](query_input), dim=-1
                    )
                    # Every candidate is from the same episode, so a constant
                    # colour/goal tag contributes equally and cannot solve it.
                    logits = torch.einsum("bd,btd->bt", query, keys) / self.temperature
                    target_index = torch.full(
                        (emb.size(0),),
                        time_index + horizon,
                        device=emb.device,
                        dtype=torch.long,
                    )
                    reach_terms.append(F.cross_entropy(logits, target_index))
                    reach_correct.append((logits.argmax(dim=-1) == target_index).float().mean())

        inverse_loss = torch.stack(inverse_terms).mean()
        zero = inverse_loss * 0.0
        cycle_loss = zero
        if self.mode == "masked_reachability" and control_pred is not None:
            length = min(control_pred.size(1), control_emb.size(1) - 1, actions.size(1))
            if length:
                start = control_emb[:, :length]
                predicted_end = control_pred[:, :length]
                start, predicted_end = self._paired_mask(start, predicted_end)
                reconstructed = self.inverse_heads["1"](
                    torch.cat((start, predicted_end), dim=-1)
                )
                cycle_loss = F.smooth_l1_loss(reconstructed, actions[:, :length])

        reachability_loss = torch.stack(reach_terms).mean() if reach_terms else zero
        reachability_accuracy = (
            torch.stack(reach_correct).mean().detach() if reach_correct else zero.detach()
        )
        if self.mode in {"direct_reachability", "factorized_reachability"} and control_pred is not None:
            direct_terms = []
            direct_correct = []
            length = min(control_pred.size(1), control_emb.size(1) - 1)
            keys = F.normalize(control_emb, dim=-1)
            for time_index in range(length):
                query = F.normalize(control_pred[:, time_index], dim=-1)
                logits = torch.einsum("bd,btd->bt", query, keys) / self.temperature
                target_index = torch.full(
                    (emb.size(0),),
                    time_index + 1,
                    device=emb.device,
                    dtype=torch.long,
                )
                direct_terms.append(F.cross_entropy(logits, target_index))
                direct_correct.append(
                    (logits.argmax(dim=-1) == target_index).float().mean()
                )
            if direct_terms:
                reachability_loss = torch.stack(direct_terms).mean()
                reachability_accuracy = torch.stack(direct_correct).mean().detach()

        total = self.inverse_weight * inverse_loss
        if self.mode == "masked_reachability":
            total = (
                total
                + self.cycle_weight * cycle_loss
                + self.reachability_weight * reachability_loss
            )
        elif self.mode == "direct_reachability":
            total = total + self.reachability_weight * reachability_loss
        elif self.mode == "factorized_reachability":
            total = (
                total
                + self.reachability_weight * reachability_loss
                + self.context_weight * context_consistency
                + self.static_leak_weight * dynamic_static
                + self.dynamic_variance_weight * dynamic_variance
                + self.dynamic_covariance_weight * dynamic_covariance
            )
        return {
            "control_loss": total,
            "inverse_dynamics_loss": inverse_loss,
            "action_cycle_loss": cycle_loss,
            "reachability_loss": reachability_loss,
            "reachability_accuracy": reachability_accuracy,
            "context_consistency_loss": context_consistency,
            "dynamic_static_leak_loss": dynamic_static,
            "dynamic_variance_loss": dynamic_variance,
            "dynamic_covariance_loss": dynamic_covariance,
            **inverse_by_horizon,
        }
