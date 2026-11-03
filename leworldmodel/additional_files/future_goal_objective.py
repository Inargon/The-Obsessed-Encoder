"""Budget-conditioned observed-future contrast, NOT optimal reachability.

Other trajectories supply marginal goal samples, not unreachable labels.
No goal later than H in the query's own clip is used as a negative.
The score is uncalibrated and depends on the collection policy/distribution.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class FutureGoalObjective(nn.Module):
    def __init__(self, embed_dim, max_horizon=3, hidden_dim=128,
                 weight=.1, temperature=.1, detach_representation=True):
        super().__init__()
        if max_horizon < 1 or weight < 0 or temperature <= 0:
            raise ValueError('Invalid future-goal objective configuration')
        self.max_horizon = int(max_horizon)
        self.weight = float(weight)
        self.temperature = float(temperature)
        self.detach_representation = bool(detach_representation)
        self.query = nn.Sequential(nn.Linear(embed_dim + 1, hidden_dim), nn.GELU(),
                                   nn.Linear(hidden_dim, hidden_dim))
        self.key = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim))

    def logits(self, emb, horizon):
        if emb.ndim != 3 or emb.shape[0] < 2:
            raise ValueError('Need B>=2 continuous clips shaped B,T,D')
        if not 1 <= horizon <= min(self.max_horizon, emb.shape[1]-1):
            raise ValueError('Budget exceeds observed clip')
        if self.detach_representation:
            emb = emb.detach()
        budget = emb.new_full((len(emb), 1), horizon / self.max_horizon)
        q = F.normalize(self.query(torch.cat((emb[:, 0], budget), dim=-1)), dim=-1)
        # Only observed future frames 1..H; own later frames are excluded.
        keys = F.normalize(self.key(emb[:, 1:horizon+1].reshape(-1, emb.shape[-1])), dim=-1)
        return q @ keys.T / self.temperature

    def forward(self, emb):
        terms, accuracy = [], []
        for h in range(1, min(self.max_horizon, emb.shape[1]-1)+1):
            logits = self.logits(emb, h)
            # Uniform positive over own H future frames, not a binary success label.
            logp = F.log_softmax(logits.float(), dim=-1).reshape(len(emb), len(emb), h)
            rows = torch.arange(len(emb), device=emb.device)
            terms.append(-logp[rows, rows].mean() - math.log(h))
            accuracy.append((logits.argmax(-1)//h == rows).float().mean())
        if not terms:
            raise ValueError('Need at least one observed successor')
        raw = torch.stack(terms).mean()
        return {'future_goal_loss': self.weight * raw,
                'future_goal_contrast_loss': raw,
                'future_goal_source_accuracy': torch.stack(accuracy).mean().detach()}
