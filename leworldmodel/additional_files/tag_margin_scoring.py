"""Fit a shared frozen-latent scorer using physically ordered candidate pairs."""

import copy

import numpy as np
import torch
from torch import nn


TAG_COMBINATIONS = ((0, 0), (1, 0), (0, 1), (1, 1))
ARMS = ("reference_only", "mean_over_tags", "worst_tag")


def cost_inputs(predictions, goals):
    """Construct all four matched views and their Euclidean baseline costs."""
    predicted = torch.stack([predictions[:, c] for c, _ in TAG_COMBINATIONS], dim=1)
    target = torch.stack([goals[:, g] for _, g in TAG_COMBINATIONS], dim=1).unsqueeze(2)
    difference = predicted - target
    native = difference.square().sum(-1)
    scale = native.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    normalized = (native - native.mean(dim=-1, keepdim=True)) / scale
    inputs = torch.cat((difference, target.expand_as(difference)), dim=-1)
    return inputs, native, normalized


def physical_pairs(baseline, endpoint_distance, starts_outside, hard_negatives=4):
    """Use clear hits and the hardest clear misses in the frozen planner bank."""
    good = endpoint_distance <= 1
    bad = endpoint_distance >= 2
    masked = baseline.masked_fill(~bad[:, None, :], torch.inf)
    hardest = masked.argsort(dim=-1)[:, :, :hard_negatives].reshape(len(baseline), -1)
    selected = torch.zeros_like(bad).scatter_(1, hardest, True) & bad
    return (starts_outside[:, None, None] & good[:, :, None] & selected[:, None, :]).nonzero(as_tuple=True)


def margin_loss(scores, pairs, arm, margin=0.25):
    query, better, worse = pairs
    gaps = scores[query, :, worse] - scores[query, :, better]
    violations = torch.relu(margin - gaps)
    if arm == "reference_only":
        loss = violations[:, 0]
    elif arm == "mean_over_tags":
        loss = violations.mean(dim=1)
    elif arm == "worst_tag":
        loss = violations.amax(dim=1)
    else:
        raise ValueError(f"Unknown training objective {arm}")
    # Give each informative query the same weight regardless of the number of clear pairs.
    counts = torch.bincount(query)
    return (loss / counts[query]).sum() / (counts > 0).sum()


def fit_heads(inputs, baseline, pairs, steps, seed):
    torch.manual_seed(seed)
    template = nn.Sequential(
        nn.LayerNorm(inputs.shape[-1]), nn.Linear(inputs.shape[-1], 32), nn.GELU(), nn.Linear(32, 1),
    ).to(device=inputs.device, dtype=inputs.dtype)
    nn.init.zeros_(template[-1].weight)
    nn.init.zeros_(template[-1].bias)
    fitted = {}
    for arm in ARMS:
        model = copy.deepcopy(template)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = baseline + model(inputs).squeeze(-1)
            loss = margin_loss(prediction, pairs, arm)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{arm}: nonfinite loss at step {step}")
            loss.backward()
            optimizer.step()
        fitted[arm] = model.eval()
        print(f"PHYSICAL_MARGIN_TRAINED {arm} loss={loss.item():.6f}", flush=True)
    return fitted


def heldout_metrics(scores, distances):
    if scores.shape[0] != len(distances) or scores.shape[-1] != distances.shape[-1]:
        raise ValueError("Candidate scores do not match physical measurements")
    if not np.isfinite(scores).all() or not np.isfinite(distances).all():
        raise ValueError("Nonfinite held-out scores or physical distances")
    selected = scores.argmin(axis=-1)
    physical = np.take_along_axis(distances, selected, axis=1)
    regret = physical - distances.min(axis=-1, keepdims=True)
    return {
        "any_tag_flip": (selected[:, 1:] != selected[:, :1]).any(axis=1),
        "all_views_hit": (physical <= 1).all(axis=1),
        "worst_endpoint_regret": regret.max(axis=1),
    }


def compare_holdout(variants, distances, eligible, seed):
    metrics = {name: heldout_metrics(values, distances) for name, values in variants.items()}
    result = {}
    for label, mask in (("all_queries", np.ones(len(distances), dtype=bool)), ("informative_queries", eligible)):
        count = int(mask.sum())
        if count == 0:
            result[label] = {"queries": 0, "variants": {}}
            continue
        sampled = np.random.default_rng(seed).integers(count, size=(2000, count))
        reports = {}
        for name, measures in metrics.items():
            summary = {}
            for key, values in measures.items():
                observed = values[mask].astype(float)
                delta = observed - metrics["native"][key][mask].astype(float)
                summary[key] = {
                    "mean": float(observed.mean()),
                    "difference_vs_native": float(delta.mean()),
                    "paired_episode_ci95": np.quantile(delta[sampled].mean(1), [0.025, 0.975]).tolist(),
                }
                if name == "worst_tag":
                    delta_to_mean = observed - metrics["mean_over_tags"][key][mask].astype(float)
                    summary[key]["paired_vs_mean_over_tags"] = {
                        "difference": float(delta_to_mean.mean()),
                        "paired_episode_ci95": np.quantile(
                            delta_to_mean[sampled].mean(1), [0.025, 0.975],
                        ).tolist(),
                    }
            reports[name] = summary
        result[label] = {"queries": count, "variants": reports}
    return result
