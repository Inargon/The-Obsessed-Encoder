from __future__ import annotations

import torch

from leworldmodel.additional_files.allocation_regularizers import AllocationRegularizer


def test_conditional_penalizes_episode_constant_embeddings():
    torch.manual_seed(0)
    static = torch.randn(12, 1, 24).expand(-1, 4, -1).clone()
    dynamic = torch.randn(12, 4, 24)
    regularizer = AllocationRegularizer(mode="conditional")

    static_terms = regularizer(static)
    dynamic_terms = regularizer(dynamic)

    assert static_terms["conditional_variance_loss"] > 0.15
    assert dynamic_terms["conditional_variance_loss"] < 1e-3
    assert static_terms["allocation_loss"] > dynamic_terms["allocation_loss"]


def test_local_rank_penalizes_low_dimensional_manifold():
    torch.manual_seed(1)
    batch, time, dim = 20, 4, 32
    factors = torch.randn(batch * time, 2)
    mixing = torch.randn(2, dim)
    low_dimensional = (factors @ mixing).reshape(batch, time, dim)
    high_dimensional = torch.randn(batch, time, dim)
    regularizer = AllocationRegularizer(
        mode="local_rank", neighbors=12, local_rank_target=6.0
    )

    low_terms = regularizer(low_dimensional)
    high_terms = regularizer(high_dimensional)

    assert low_terms["local_effective_rank"] < 3.0
    assert high_terms["local_effective_rank"] > low_terms["local_effective_rank"]
    assert low_terms["local_rank_loss"] > high_terms["local_rank_loss"]


def test_hybrid_has_finite_gradients():
    torch.manual_seed(2)
    emb = torch.randn(8, 4, 16, requires_grad=True)
    regularizer = AllocationRegularizer(
        mode="hybrid", neighbors=6, local_rank_target=5.0
    )

    terms = regularizer(emb)
    terms["allocation_loss"].backward()

    assert emb.grad is not None
    assert torch.isfinite(emb.grad).all()
