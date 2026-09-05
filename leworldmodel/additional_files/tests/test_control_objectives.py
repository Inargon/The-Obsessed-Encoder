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


def test_sequence_inverse_target_predicts_complete_action_chunks():
    emb, actions, pred_emb = _inputs()
    objective = ControlObjective(
        embed_dim=12,
        action_dim=2,
        mode="multi_horizon_idm",
        max_horizon=3,
        inverse_target="sequence",
    )

    terms = objective(emb, actions, pred_emb)
    terms["control_loss"].backward()

    assert objective.inverse_heads["1"][-1].out_features == 2
    assert objective.inverse_heads["2"][-1].out_features == 4
    assert objective.inverse_heads["3"][-1].out_features == 6
    assert torch.isfinite(terms["control_loss"])
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


def test_masked_reachability_reports_action_shuffle_diagnostics():
    emb, actions, pred_emb = _inputs()
    objective = ControlObjective(
        embed_dim=12,
        action_dim=2,
        mode="masked_reachability",
        max_horizon=3,
        inverse_target="sequence",
        action_shuffle_diagnostics=True,
    )

    terms = objective(emb, actions, pred_emb)

    assert 0 <= terms["reachability_shuffled_accuracy"] <= 1
    assert torch.isfinite(terms["reachability_action_margin"])


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


def test_direct_reachability_shapes_the_unprojected_embedding():
    emb, actions, pred_emb = _inputs()
    objective = ControlObjective(
        embed_dim=12,
        action_dim=2,
        mode="direct_reachability",
        max_horizon=1,
        reachability_weight=0.1,
    )

    terms = objective(emb, actions, pred_emb)
    terms["control_loss"].backward()

    assert terms["inverse_horizon_1_loss"] > 0
    assert terms["reachability_loss"] > 0
    assert 0 <= terms["reachability_accuracy"] <= 1
    assert emb.grad is not None and torch.isfinite(emb.grad).all()
    assert pred_emb.grad is not None and torch.isfinite(pred_emb.grad).all()
    # No learned reachability projection exists to hide the signal from the
    # embedding used by downstream planning.
    assert len(objective.reach_queries) == 0
    assert not any("reach_key" in name for name, _ in objective.named_parameters())


def test_factorized_reachability_penalizes_static_leak_in_dynamic_block():
    torch.manual_seed(12)
    batch, time, context_dim, dynamic_dim = 8, 4, 4, 8
    context = torch.randn(batch, 1, context_dim).expand(-1, time, -1)
    static_dynamic = torch.randn(batch, 1, dynamic_dim).expand(-1, time, -1)
    changing_dynamic = torch.randn(batch, time, dynamic_dim)
    actions = torch.randn(batch, time, 2)
    pred = torch.randn(batch, time - 1, context_dim + dynamic_dim)
    objective = ControlObjective(
        embed_dim=context_dim + dynamic_dim,
        action_dim=2,
        mode="factorized_reachability",
        max_horizon=1,
        context_dim=context_dim,
    )

    static_terms = objective(
        torch.cat((context, static_dynamic), dim=-1), actions, pred
    )
    changing_terms = objective(
        torch.cat((context, changing_dynamic), dim=-1), actions, pred
    )

    assert static_terms["dynamic_static_leak_loss"] > 0.9
    assert changing_terms["dynamic_static_leak_loss"] < 0.7
    assert static_terms["context_consistency_loss"] == 0


def test_factorized_reachability_has_finite_gradients():
    emb, actions, pred_emb = _inputs()
    objective = ControlObjective(
        embed_dim=12,
        action_dim=2,
        mode="factorized_reachability",
        max_horizon=1,
        context_dim=4,
    )

    terms = objective(emb, actions, pred_emb)
    terms["control_loss"].backward()

    for key in (
        "context_consistency_loss",
        "dynamic_static_leak_loss",
        "dynamic_variance_loss",
        "dynamic_covariance_loss",
    ):
        assert torch.isfinite(terms[key])
    assert emb.grad is not None and torch.isfinite(emb.grad).all()
    assert pred_emb.grad is not None and torch.isfinite(pred_emb.grad).all()
