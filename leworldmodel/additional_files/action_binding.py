"""Counterfactual action binding on same-anchor intervention clouds.

IDM asks whether an action can be decoded from two latents.  This objective
asks the planning model a stronger, forward question: among several futures
from the same state, does each action sequence select its actual outcome?
"""

from __future__ import annotations

from glob import glob
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .effect_geometry import geometry_terms, physical_effects


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


def binding_losses(predicted: torch.Tensor, target_a: torch.Tensor,
                   target_b: torch.Tensor, hard_partner: torch.Tensor,
                   temperature: float = 0.1) -> dict[str, torch.Tensor]:
    """Forward branch retrieval plus invariance of physical effects to tags."""
    target = 0.5 * (target_a + target_b)
    p = F.normalize(predicted.float(), dim=-1)
    t = F.normalize(target.detach().float(), dim=-1)
    logits = torch.einsum("nkd,njd->nkj", p, t) / temperature
    labels = torch.arange(predicted.size(1), device=predicted.device)
    labels = labels[None].expand(predicted.size(0), -1)
    forward_ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    reverse_ce = F.cross_entropy(
        logits.transpose(1, 2).flatten(0, 1), labels.flatten()
    )
    binding = 0.5 * (forward_ce + reverse_ce)
    accuracy = (logits.argmax(-1) == labels).float().mean().detach()

    valid = hard_partner >= 0
    if valid.any():
        n_idx, branch_idx = valid.nonzero(as_tuple=True)
        partner_idx = hard_partner[n_idx, branch_idx]
        positive = logits[n_idx, branch_idx, branch_idx]
        negative = logits[n_idx, branch_idx, partner_idx]
        hard_logits = torch.stack((positive, negative), dim=-1)
        hard = F.cross_entropy(
            hard_logits, torch.zeros_like(positive, dtype=torch.long)
        )
        hard_accuracy = (positive > negative).float().mean().detach()
    else:
        hard = predicted.sum() * 0.0
        hard_accuracy = hard.detach()

    return {
        "binding_loss": binding,
        "hard_binding_loss": hard,
        "binding_accuracy": accuracy,
        "hard_binding_accuracy": hard_accuracy,
        "counterfactual_forward_loss": F.smooth_l1_loss(
            predicted.float(), target.detach().float()
        ),
        "nuisance_effect_loss": F.smooth_l1_loss(
            target_a.float(), target_b.float()
        ),
    }


class CounterfactualActionObjective(nn.Module):
    MODES = {"bank_idm", "binding", "binding_geometry"}

    def __init__(self, *, embed_dim: int, action_dim: int, mode: str,
                 bank_glob: str, horizon: int = 3, hidden_dim: int = 256,
                 clouds_per_step: int = 2, branches_per_cloud: int = 8,
                 inverse_weight: float = 1.0, forward_weight: float = 1.0,
                 binding_weight: float = 0.1, hard_binding_weight: float = 0.1,
                 nuisance_weight: float = 0.5, geometry_weight: float = 0.1,
                 temperature: float = 0.1, tag_size: int = 5,
                 image_size: int = 224, seed: int = 0):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown action-binding mode {mode!r}")
        paths = sorted(glob(str(Path(bank_glob).expanduser())))
        if not paths:
            raise FileNotFoundError(f"no action-binding shards match {bank_glob!r}")
        shards = [np.load(path) for path in paths]
        keys = (
            "history_frames", "history_actions", "anchor_states",
            "branch_frames", "branch_states", "branch_actions", "hard_partner",
        )
        values = {key: np.concatenate([shard[key] for shard in shards]) for key in keys}
        self.action_mean = np.asarray(shards[0]["action_mean"])
        self.action_std = np.asarray(shards[0]["action_std"])
        for shard in shards:
            shard.close()
        self.history_frames = values["history_frames"]
        self.history_actions = values["history_actions"]
        self.anchor_states = values["anchor_states"]
        self.branch_frames = values["branch_frames"]
        self.branch_states = values["branch_states"]
        self.branch_actions = values["branch_actions"]
        self.hard_partner = values["hard_partner"]
        if branches_per_cloud > self.branch_frames.shape[1]:
            raise ValueError("branches_per_cloud exceeds stored branch count")
        if self.branch_actions.shape[-2] != horizon:
            raise ValueError("configured horizon does not match bank")
        if self.branch_actions.shape[-1] != action_dim:
            raise ValueError("configured action_dim does not match bank")

        self.mode = mode
        self.horizon = int(horizon)
        self.clouds_per_step = int(clouds_per_step)
        self.branches_per_cloud = int(branches_per_cloud)
        self.inverse_weight = float(inverse_weight)
        self.forward_weight = float(forward_weight)
        self.binding_weight = float(binding_weight)
        self.hard_binding_weight = float(hard_binding_weight)
        self.nuisance_weight = float(nuisance_weight)
        self.geometry_weight = float(geometry_weight)
        self.temperature = float(temperature)
        self.tag_size = int(tag_size)
        self.image_size = int(image_size)
        self.rng = np.random.default_rng(seed)
        self.inverse_head = _mlp(2 * embed_dim, hidden_dim, horizon * action_dim)

    def _images(self, frames: np.ndarray, device: torch.device) -> torch.Tensor:
        x = torch.as_tensor(frames, device=device).permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, (self.image_size, self.image_size), mode="bilinear",
                              align_corners=False)
        mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return (x - mean) / std

    def _tag(self, frames: np.ndarray, colours: np.ndarray) -> np.ndarray:
        out = frames.copy()
        shape = (len(colours),) + (1,) * (out.ndim - 4) + (1, 1, 3)
        out[..., :self.tag_size, :self.tag_size, :] = colours.reshape(shape)
        return out

    def _encode_frames(self, model, frames: np.ndarray, device: torch.device):
        flat = self._images(frames.reshape(-1, *frames.shape[-3:]), device)
        cls = model.encoder(flat, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        return model.projector(cls).reshape(*frames.shape[:-3], -1)

    def _normalise_actions(self, actions: np.ndarray, device: torch.device):
        mean = torch.as_tensor(self.action_mean, device=device)
        std = torch.as_tensor(self.action_std, device=device)
        return (torch.as_tensor(actions, device=device).float() - mean) / (std + 1e-8)

    def forward(self, model) -> dict[str, torch.Tensor]:
        device = next(model.parameters()).device
        n = min(self.clouds_per_step, len(self.history_frames))
        anchors = self.rng.choice(len(self.history_frames), size=n, replace=False)
        # The first stored branches are mediator-matched hard pairs; always keep
        # them and sample any remaining slots from the physically diverse tail.
        hard_count = min(4, self.branches_per_cloud)
        chosen = []
        for _ in range(n):
            tail = np.arange(hard_count, self.branch_frames.shape[1])
            extra = self.rng.choice(
                tail, size=self.branches_per_cloud - hard_count, replace=False
            ) if self.branches_per_cloud > hard_count else np.empty(0, dtype=int)
            chosen.append(np.concatenate((np.arange(hard_count), extra)))
        chosen = np.stack(chosen)

        histories = self.history_frames[anchors]
        hist_actions = self.history_actions[anchors]
        ends = self.branch_frames[anchors[:, None], chosen]
        end_states = self.branch_states[anchors[:, None], chosen]
        actions = self.branch_actions[anchors[:, None], chosen]
        partners = self.hard_partner[anchors[:, None], chosen].copy()
        # Stored partner ids refer to the stored ordering; remap to this sample.
        for row in range(n):
            remap = {int(old): new for new, old in enumerate(chosen[row])}
            partners[row] = [remap.get(int(old), -1) for old in partners[row]]

        colours_a = self.rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
        colours_b = self.rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
        hist_a = self._tag(histories, colours_a)
        anchor_a = hist_a[:, -1]
        anchor_b = self._tag(histories[:, -1:], colours_b)[:, 0]
        ends_a = self._tag(ends, colours_a)
        ends_b = self._tag(ends, colours_b)

        target_a = self._encode_frames(model, ends_a, device)
        target_b = self._encode_frames(model, ends_b, device)
        anchor_a_z = self._encode_frames(model, anchor_a[:, None], device)[:, 0]
        anchor_b_z = self._encode_frames(model, anchor_b[:, None], device)[:, 0]
        effect_a = target_a - anchor_a_z[:, None]
        effect_b = target_b - anchor_b_z[:, None]

        norm_hist = self._normalise_actions(hist_actions[:, :2], device)
        norm_branch = self._normalise_actions(actions, device)
        sequence = torch.cat(
            (norm_hist[:, None].expand(-1, self.branches_per_cloud, -1, -1),
             norm_branch), dim=2
        )
        hist_px = self._images(hist_a.reshape(-1, *hist_a.shape[-3:]), device)
        hist_px = hist_px.reshape(n, hist_a.shape[1], *hist_px.shape[1:])
        rollout = model.rollout(
            {"pixels": hist_px[:, None]}, sequence, history_size=hist_a.shape[1]
        )
        predicted_end = rollout["predicted_emb"][:, :, -1]
        predicted_effect = predicted_end - anchor_a_z[:, None]

        inverse_pred = self.inverse_head(torch.cat(
            (anchor_a_z[:, None].expand_as(target_a), target_a), dim=-1
        ))
        bank_inverse = F.smooth_l1_loss(
            inverse_pred.float(), norm_branch.flatten(-2).float()
        )
        zero = bank_inverse * 0.0
        terms = {
            "bank_inverse_loss": bank_inverse,
            "binding_loss": zero,
            "hard_binding_loss": zero,
            "counterfactual_forward_loss": zero,
            "nuisance_effect_loss": zero,
            "binding_accuracy": zero.detach(),
            "hard_binding_accuracy": zero.detach(),
            "binding_geometry_real_loss": zero,
            "binding_geometry_pred_loss": zero,
        }
        if self.mode != "bank_idm":
            terms.update(binding_losses(
                predicted_effect, effect_a, effect_b,
                torch.as_tensor(partners, device=device), self.temperature,
            ))
        total = self.inverse_weight * bank_inverse
        if self.mode != "bank_idm":
            total = total + (
                self.forward_weight * terms["counterfactual_forward_loss"]
                + self.binding_weight * terms["binding_loss"]
                + self.hard_binding_weight * terms["hard_binding_loss"]
                + self.nuisance_weight * terms["nuisance_effect_loss"]
            )
        if self.mode == "binding_geometry":
            physical = physical_effects(
                torch.as_tensor(end_states, device=device),
                torch.as_tensor(self.anchor_states[anchors], device=device),
                agent_position_scale=128.0,
                block_position_scale=32.0,
                angular_scale=1.0,
            )
            real_geom = geometry_terms(0.5 * (effect_a + effect_b), physical)
            pred_geom = geometry_terms(predicted_effect, physical)
            terms["binding_geometry_real_loss"] = (
                real_geom["effect_geometry_gram_loss"]
                + real_geom["effect_geometry_scale_loss"]
            )
            terms["binding_geometry_pred_loss"] = (
                pred_geom["effect_geometry_gram_loss"]
                + pred_geom["effect_geometry_scale_loss"]
            )
            total = total + self.geometry_weight * (
                terms["binding_geometry_real_loss"]
                + terms["binding_geometry_pred_loss"]
            )
        terms["action_binding_loss"] = total
        return terms
