"""Counterfactual tag invariance and action-discriminative prediction.

This objective is deliberately independent of control-gradient routing.  A
second view changes only the synthetic corner tag.  The observed action must
predict the observed future better than another sample's action, while the
resulting two-candidate ranking should be stable across the tag intervention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _per_item_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target.detach()).square().mean(dim=-1)


def counterfactual_planning_objective(
    *,
    reference_emb: torch.Tensor,
    changed_emb: torch.Tensor,
    reference_pred: torch.Tensor,
    changed_pred: torch.Tensor,
    reference_negative_pred: torch.Tensor,
    changed_negative_pred: torch.Tensor,
    reference_target: torch.Tensor,
    changed_target: torch.Tensor,
    representation_weight: float = 0.1,
    action_weight: float = 1.0,
    ranking_weight: float = 0.1,
    margin: float = 0.05,
    temperature: float = 0.1,
    ranking_metric: str = "js",
    margin_fraction: float = 0.25,
    action_margin_reference_only: bool = False,
    tag_intervention: str = "both",
) -> dict[str, torch.Tensor]:
    """Return the loss and diagnostics for a matched pair of visual views."""
    if reference_emb.shape != changed_emb.shape:
        raise ValueError("reference and counterfactual embeddings must match")
    if temperature <= 0 or margin < 0:
        raise ValueError("temperature must be positive and margin nonnegative")
    if ranking_metric not in {"js", "centered_mse", "margin_guard"}:
        raise ValueError("ranking_metric must be 'js', 'centered_mse', or 'margin_guard'")
    if not 0 <= margin_fraction <= 1:
        raise ValueError("margin_fraction must be between zero and one")

    if tag_intervention == "context_only":
        changed_target = reference_target
    elif tag_intervention == "goal_only":
        changed_pred = reference_pred
        changed_negative_pred = reference_negative_pred
    elif tag_intervention == "balanced":
        # Shuffled batch rows cycle through both, context-only, and goal-only.
        # Keep the existing RNG stream and use the same actions in every case.
        mode = torch.arange(changed_pred.shape[0], device=changed_pred.device) % 3
        mode = mode.reshape(-1, *([1] * (changed_pred.ndim - 1)))
        changed_pred = torch.where(mode != 2, changed_pred, reference_pred)
        changed_negative_pred = torch.where(mode != 2, changed_negative_pred, reference_negative_pred)
        changed_target = torch.where(mode != 1, changed_target, reference_target)
    elif tag_intervention != "both":
        raise ValueError(f"Unknown tag intervention {tag_intervention!r}")

    ref_pos = _per_item_mse(reference_pred, reference_target)
    ref_neg = _per_item_mse(reference_negative_pred, reference_target)
    changed_pos = _per_item_mse(changed_pred, changed_target)
    changed_neg = _per_item_mse(changed_negative_pred, changed_target)

    action_margin_loss = F.relu(margin + ref_pos - ref_neg).mean()
    if not action_margin_reference_only:
        action_margin_loss = 0.5 * (
            action_margin_loss + F.relu(margin + changed_pos - changed_neg).mean()
        )
    representation_loss = F.mse_loss(reference_emb, changed_emb)

    ref_scores = torch.stack((ref_pos, ref_neg), dim=-1)
    changed_scores = torch.stack((changed_pos, changed_neg), dim=-1)
    # The observed action anchors the comparison once it beats the shuffled
    # action by the supervised margin. Stop-gradient fixes the required gap.
    reference_gap = (ref_neg.detach().float() - ref_pos.detach().float())
    changed_gap = changed_neg.float() - changed_pos.float()
    margin_eligible = (reference_gap > 0) & (reference_gap >= margin)
    margin_deficit = F.relu(margin_fraction * reference_gap - changed_gap)
    if ranking_metric == "centered_mse":
        # Subtract common cost shifts before comparing the action margins.
        score_delta = ref_scores.float() - changed_scores.float()
        ranking_loss = (score_delta - score_delta.mean(dim=-1, keepdim=True)).square().mean()
    elif ranking_metric == "margin_guard":
        # Match centered MSE's two-candidate scale for equal gap errors.
        ranking_loss = 0.25 * (margin_deficit.square() * margin_eligible).mean()
    else:
        ref_prob = F.softmax(-ref_scores / temperature, dim=-1)
        changed_prob = F.softmax(-changed_scores / temperature, dim=-1)
        midpoint = 0.5 * (ref_prob + changed_prob)
        eps = torch.finfo(ref_prob.dtype).eps
        ranking_loss = 0.5 * (
            (ref_prob * ((ref_prob + eps).log() - (midpoint + eps).log())).sum(-1).mean()
            + (changed_prob * ((changed_prob + eps).log() - (midpoint + eps).log())).sum(-1).mean()
        )

    total = (
        float(representation_weight) * representation_loss
        + float(action_weight) * action_margin_loss
        + float(ranking_weight) * ranking_loss
    )
    return {
        "counterfactual_planning_loss": total,
        "counterfactual_representation_loss": representation_loss,
        "counterfactual_action_margin_loss": action_margin_loss,
        "counterfactual_ranking_loss": ranking_loss,
        "counterfactual_margin_eligible_rate": margin_eligible.float().mean(),
        "counterfactual_margin_active_rate": (margin_eligible & (margin_deficit > 0)).float().mean(),
        "counterfactual_action_accuracy": 0.5
        * ((ref_pos < ref_neg).float().mean() + (changed_pos < changed_neg).float().mean()),
        "counterfactual_ranking_changed": (
            ref_scores.argmin(-1) != changed_scores.argmin(-1)
        ).float().mean(),
        "counterfactual_positive_distance": 0.5 * (ref_pos.mean() + changed_pos.mean()),
        "counterfactual_negative_distance": 0.5 * (ref_neg.mean() + changed_neg.mean()),
    }
