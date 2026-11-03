"""Fast synthetic diagnostic for the action-effect ratio pilot.

Run from the repository root:
    python leworldmodel/additional_files/smoke_action_effect_ratio.py

This is a mechanism check on generated data, not a robotics benchmark.
"""

from __future__ import annotations

import torch
from torch import nn

from action_effect_ratio import ActionEffectRatio


def sample(batch_size: int) -> tuple[torch.Tensor, ...]:
    position = torch.randn(batch_size, 2)
    action = 0.5 * torch.randn(batch_size, 2)
    next_position = position + action + 0.05 * torch.randn_like(action)
    tag = torch.randn(batch_size, 3)
    changed_tag = torch.randn_like(tag)
    start = torch.cat((position, tag), dim=-1)
    end = torch.cat((next_position, tag), dim=-1)
    changed_start = torch.cat((position, changed_tag), dim=-1)
    changed_end = torch.cat((next_position, changed_tag), dim=-1)
    return start, end, changed_start, changed_end, action


def run(seed: int, consistency_weight: float, steps: int = 300) -> tuple[float, float]:
    torch.manual_seed(seed)
    encoder = nn.Linear(5, 8)
    ratio = ActionEffectRatio(embed_dim=8, action_dim=2, hidden_dim=32)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(ratio.parameters()), lr=0.003
    )
    for _ in range(steps):
        start, end, changed_start, changed_end, action = sample(128)
        result = ratio(
            encoder(start), encoder(end), action,
            counterfactual_start=encoder(changed_start),
            counterfactual_end=encoder(changed_end),
            consistency_weight=consistency_weight,
        )
        optimizer.zero_grad()
        result["action_effect_loss"].backward()
        optimizer.step()

    with torch.no_grad():
        start, end, changed_start, changed_end, action = sample(2048)
        original = ratio.score(encoder(start), encoder(end), action)
        changed = ratio.score(encoder(changed_start), encoder(changed_end), action)
        shuffled = ratio.score(encoder(start), encoder(end), action.roll(1, 0))
        tag_shift = (original - changed).abs().mean().item()
        action_margin = (original - shuffled).mean().item()
    return tag_shift, action_margin


if __name__ == "__main__":
    torch.set_num_threads(1)
    print("synthetic test: position changes with action; 3D tag is episode-constant")
    print("weight | tag-only score change, mean +/- SD | action margin, mean +/- SD")
    for weight in (0.0, 0.1):
        values = torch.tensor([run(seed, weight, steps=250) for seed in (7, 8, 9)])
        means = values.mean(dim=0)
        deviations = values.std(dim=0)
        print(
            f"{weight:6.2f} | {means[0]:.4f} +/- {deviations[0]:.4f}"
            f" | {means[1]:.2f} +/- {deviations[1]:.2f}"
        )
