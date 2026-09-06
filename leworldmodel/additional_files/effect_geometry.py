"""Privileged reachable-effect geometry objective for PushT.

Each bank row contains several simulator rollouts from exactly the same anchor.
The loss asks the encoder to preserve the physical outcome cloud's centred Gram
matrix.  It is intentionally an oracle experiment: physical state is used only
to define the target geometry and is never passed to the planner.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def physical_effects(states: torch.Tensor, anchors: torch.Tensor,
                     agent_position_scale: float = 128.0,
                     block_position_scale: float = 32.0,
                     angular_scale: float = 1.0) -> torch.Tensor:
    """PushT state differences in a periodic, dimensionless physical metric."""
    if states.ndim != 3 or states.size(-1) < 5:
        raise ValueError("states must have shape (N,K,>=5)")
    if anchors.shape != (states.size(0), states.size(-1)):
        raise ValueError("anchors must have shape (N,state_dim)")
    delta = states[..., :4] - anchors[:, None, :4]
    agent = delta[..., :2] / agent_position_scale
    block = delta[..., 2:4] / block_position_scale
    theta = states[..., 4]
    theta0 = anchors[:, None, 4]
    angular = torch.stack(
        (torch.sin(theta) - torch.sin(theta0),
         torch.cos(theta) - torch.cos(theta0)), dim=-1
    )
    return torch.cat((agent, block, angular / angular_scale), dim=-1)


def centred_normalized_gram(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Centred per-cloud Gram with unit trace; shape (N,K,K)."""
    x = x.float() - x.float().mean(dim=1, keepdim=True)
    gram = x @ x.transpose(1, 2)
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(eps)
    return gram / trace[:, None, None]


def geometry_terms(latent_effects: torch.Tensor, physical: torch.Tensor, *,
                   min_effect_rms: float = 0.05,
                   max_effect_rms: float = 2.0) -> dict[str, torch.Tensor]:
    """Relative geometry plus an explicit non-collapse/non-explosion band."""
    z = latent_effects.float() - latent_effects.float().mean(dim=1, keepdim=True)
    s = physical.float() - physical.float().mean(dim=1, keepdim=True)
    # This is the squared Frobenius norm from the method statement, averaged
    # over clouds (not over K^2 entries, which would silently weaken it as K grows).
    gram_delta = centred_normalized_gram(z) - centred_normalized_gram(s)
    gram_loss = gram_delta.square().sum(dim=(-2, -1)).mean()
    effect_rms = z.square().mean().sqrt()
    lower = F.relu(effect_rms.new_tensor(min_effect_rms) - effect_rms).square()
    upper = F.relu(effect_rms - effect_rms.new_tensor(max_effect_rms)).square()
    return {
        "effect_geometry_gram_loss": gram_loss,
        "effect_geometry_scale_loss": lower + upper,
        "effect_geometry_latent_rms": effect_rms.detach(),
    }


class EffectGeometryOracle(torch.nn.Module):
    """Small in-memory bank sampler and differentiable encoder-side loss."""

    def __init__(self, *, bank_glob: str, weight: float = 0.1,
                 scale_weight: float = 1.0, clouds_per_step: int = 4,
                 branches_per_cloud: int = 8,
                 agent_position_scale: float = 128.0,
                 block_position_scale: float = 32.0,
                 angular_scale: float = 1.0,
                 min_effect_rms: float = 0.05, max_effect_rms: float = 2.0,
                 image_size: int = 224, seed: int = 0):
        super().__init__()
        paths = sorted(glob(str(Path(bank_glob).expanduser())))
        if not paths:
            raise FileNotFoundError(f"no effect-geometry shards match {bank_glob!r}")
        rows = [np.load(path) for path in paths]
        self.anchor_frames = np.concatenate([r["anchor_frames"] for r in rows])
        self.branch_frames = np.concatenate([r["branch_frames"] for r in rows])
        self.anchor_states = np.concatenate([r["anchor_states"] for r in rows])
        self.branch_states = np.concatenate([r["branch_states"] for r in rows])
        for r in rows:
            r.close()
        if len(self.anchor_frames) != len(self.branch_frames):
            raise ValueError("effect-geometry bank has inconsistent anchor counts")
        if branches_per_cloud > self.branch_frames.shape[1]:
            raise ValueError("branches_per_cloud exceeds branches stored in bank")
        self.weight = float(weight)
        self.scale_weight = float(scale_weight)
        self.clouds_per_step = int(clouds_per_step)
        self.branches_per_cloud = int(branches_per_cloud)
        self.agent_position_scale = float(agent_position_scale)
        self.block_position_scale = float(block_position_scale)
        self.angular_scale = float(angular_scale)
        self.min_effect_rms = float(min_effect_rms)
        self.max_effect_rms = float(max_effect_rms)
        self.image_size = int(image_size)
        self.rng = np.random.default_rng(seed)

    def _images(self, frames: np.ndarray, device: torch.device) -> torch.Tensor:
        x = torch.as_tensor(frames, device=device).permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, (self.image_size, self.image_size), mode="bilinear",
                              align_corners=False)
        mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return (x - mean) / std

    def forward(self, model) -> dict[str, torch.Tensor]:
        n = min(self.clouds_per_step, len(self.anchor_frames))
        anchors = self.rng.choice(len(self.anchor_frames), size=n, replace=False)
        branches = np.stack([
            self.rng.choice(self.branch_frames.shape[1], size=self.branches_per_cloud,
                            replace=False) for _ in range(n)
        ])
        branch_px = self.branch_frames[anchors[:, None], branches]
        branch_s = self.branch_states[anchors[:, None], branches]
        anchor_px = self.anchor_frames[anchors]
        anchor_s = self.anchor_states[anchors]
        device = next(model.parameters()).device
        all_px = np.concatenate((anchor_px[:, None], branch_px), axis=1)
        flat = self._images(all_px.reshape(-1, *all_px.shape[-3:]), device)
        encoded = model.encoder(flat, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        encoded = model.projector(encoded).reshape(n, self.branches_per_cloud + 1, -1)
        latent_effects = encoded[:, 1:] - encoded[:, :1]
        physical = physical_effects(
            torch.as_tensor(branch_s, device=device),
            torch.as_tensor(anchor_s, device=device),
            self.agent_position_scale,
            self.block_position_scale,
            self.angular_scale,
        )
        terms = geometry_terms(
            latent_effects, physical,
            min_effect_rms=self.min_effect_rms,
            max_effect_rms=self.max_effect_rms,
        )
        terms["effect_geometry_loss"] = self.weight * (
            terms["effect_geometry_gram_loss"]
            + self.scale_weight * terms["effect_geometry_scale_loss"]
        )
        return terms
