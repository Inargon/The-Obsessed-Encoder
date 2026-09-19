from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from counterfactual_planning import counterfactual_planning_objective  # noqa: E402
from counterfactual_planning_campaign import arm_spec  # noqa: E402
from pixel_tag import PixelTag  # noqa: E402


def test_counterfactual_tag_preserves_content_and_changes_patch():
    tag = PixelTag(
        mode="video", size=2, seed=7, counterfactual_key="pixels_counterfactual"
    )
    steps = {"pixels": torch.arange(3 * 3 * 6 * 6, dtype=torch.uint8).reshape(3, 3, 6, 6)}
    tag.stamp(steps["pixels"], ep_idx=11, start=0, frameskip=1)
    tag.attach_counterfactual(steps, ep_idx=11, start=0, frameskip=1)
    changed = steps["pixels_counterfactual"]
    assert not torch.equal(steps["pixels"][:, :, :2, :2], changed[:, :, :2, :2])
    assert torch.equal(steps["pixels"][:, :, 2:, 2:], changed[:, :, 2:, 2:])


def test_objective_rewards_action_discrimination_and_tag_invariance():
    emb = torch.zeros(2, 3, 4)
    target = torch.zeros(2, 2, 4)
    positive = torch.zeros_like(target)
    negative = torch.ones_like(target)
    result = counterfactual_planning_objective(
        reference_emb=emb,
        changed_emb=emb.clone(),
        reference_pred=positive,
        changed_pred=positive.clone(),
        reference_negative_pred=negative,
        changed_negative_pred=negative.clone(),
        reference_target=target,
        changed_target=target.clone(),
    )
    assert result["counterfactual_planning_loss"].item() == 0.0
    assert result["counterfactual_action_accuracy"].item() == 1.0
    assert result["counterfactual_ranking_changed"].item() == 0.0


def test_campaign_is_independent_of_aligned_and_control_losses():
    overrides = " ".join(arm_spec()["overrides"])
    assert "counterfactual_planning.enabled=true" in overrides
    assert "pixel_tag.counterfactual_key=pixels_counterfactual" in overrides
    assert "aligned_gradient_routing" not in overrides
    assert "loss.control" not in overrides
