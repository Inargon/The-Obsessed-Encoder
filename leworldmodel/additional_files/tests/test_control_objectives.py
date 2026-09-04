from __future__ import annotations

import math

import torch

from leworldmodel.additional_files.control_objectives import ControlObjective


def _inputs():
    torch.manual_seed(10)
    emb = torch.randn(4, 4, 12, requires_grad=True)
    actions = torch.randn(4, 4, 2)
    pred_emb = torch.randn(4, 3, 12, requires_grad=True)
    return emb, actions, pred_emb


def test_multi_horizon_idm_has_finite_gradients():
    emb, actions, pred_emb = _inputs()
    objective = ControlObjective(
        embed_dim=12, action_dim=2, mode="multi_horizon_idm", max_horizon=3
    )

    terms = objective(emb, actions, pred_emb)
    terms["control_loss"].backward()

    assert terms["inverse_dynamics_loss"] > 0
    assert all(terms[f"inverse_horizon_{h}_loss"] > 0 for h in (1, 2, 3))
    assert terms["action_cycle_loss"] == 0
    assert terms["reachability_loss"] == 0
    assert emb.grad is not None and torch.isfinite(emb.grad).all()


def test_masked_reachability_trains_encoder_predictor_and_heads():
    emb, actions, pred_emb = _inputs()
    objective = ControlObjective(
        embed_dim=12,
        action_dim=2,
        mode="masked_reachability",
        max_horizon=3,
        mask_keep_prob=0.5,
    )

    terms = objective(emb, actions, pred_emb)
    terms["control_loss"].backward()

    for key in ("inverse_dynamics_loss", "action_cycle_loss", "reachability_loss"):
        assert torch.isfinite(terms[key]) and terms[key] > 0
    assert 0 <= terms["reachability_accuracy"] <= 1
    assert emb.grad is not None and torch.isfinite(emb.grad).all()
    assert pred_emb.grad is not None and torch.isfinite(pred_emb.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in objective.parameters()
    )


def test_episode_constant_embedding_cannot_solve_within_episode_reachability():
    torch.manual_seed(11)
    episode_tag = torch.randn(5, 1, 12)
    emb = episode_tag.expand(-1, 4, -1).clone()
    actions = torch.randn(5, 4, 2)
    objective = ControlObjective(
        embed_dim=12,
        action_dim=2,
        mode="masked_reachability",
        max_horizon=3,
    ).eval()

    terms = objective(emb, actions)

    assert terms["reachability_accuracy"] == 0
    assert torch.allclose(
        terms["reachability_loss"],
        torch.tensor(math.log(4.0)),
        atol=1e-5,
    )


def test_control_objective_runs_under_bfloat16_autocast():
    emb, actions, pred_emb = _inputs()
    objective = ControlObjective(
        embed_dim=12, action_dim=2, mode="masked_reachability", max_horizon=2
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        terms = objective(emb, actions, pred_emb)
    terms["control_loss"].backward()

    assert torch.isfinite(terms["control_loss"])
    assert emb.grad is not None and torch.isfinite(emb.grad).all()
