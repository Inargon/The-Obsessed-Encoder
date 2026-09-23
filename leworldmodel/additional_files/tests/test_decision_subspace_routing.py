import torch

from additional_files.decision_subspace_routing import DecisionSubspaceRouter


def routed_prediction_gradient(router, pred_direction, control_direction):
    embedding = torch.zeros_like(pred_direction, requires_grad=True)
    pred_loss = (embedding * pred_direction).sum()
    control_loss = (embedding * control_direction).sum()
    surrogate, diagnostics = router(pred_loss, control_loss, embedding)
    total_grad = torch.autograd.grad(
        pred_loss + control_loss + surrogate, embedding
    )[0]
    return total_grad - control_direction, diagnostics


def test_subspace_basis_is_bounded_and_orthonormal():
    router = DecisionSubspaceRouter(
        rank=2, decay=0.9, update_every=1, candidates_per_update=4
    )
    pred = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    control = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    routed_prediction_gradient(router, pred, control)
    basis = router.basis
    assert basis.shape == (2, 3)
    torch.testing.assert_close(basis @ basis.T, torch.eye(2), atol=1e-5, rtol=1e-5)


def test_router_removes_supported_conflict():
    router = DecisionSubspaceRouter(
        rank=1, decay=0.9, update_every=1, candidates_per_update=1
    )
    routed, diagnostics = routed_prediction_gradient(
        router, torch.tensor([[-1.0, 0.0]]), torch.tensor([[1.0, 0.0]])
    )
    torch.testing.assert_close(routed, torch.zeros_like(routed), atol=1e-6, rtol=0)
    assert diagnostics["prediction_reversed_fraction"].item() == 1.0


def test_router_accumulates_multiple_control_directions():
    router = DecisionSubspaceRouter(
        rank=2, decay=0.5, update_every=1, candidates_per_update=1
    )
    routed_prediction_gradient(
        router, torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[1.0, 0.0, 0.0]])
    )
    routed, diagnostics = routed_prediction_gradient(
        router, torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([[0.0, 1.0, 0.0]])
    )
    assert diagnostics["decision_subspace_rank"].item() == 2.0
    torch.testing.assert_close(
        routed, torch.tensor([[1.0, 1.0, 0.0]]), atol=1e-5, rtol=1e-5
    )


def test_minimum_retention_restores_configured_raw_fraction():
    router = DecisionSubspaceRouter(
        rank=1,
        decay=0.9,
        update_every=1,
        candidates_per_update=1,
        minimum_retention=0.25,
    )
    routed, _ = routed_prediction_gradient(
        router, torch.tensor([[0.0, 4.0]]), torch.tensor([[1.0, 0.0]])
    )
    torch.testing.assert_close(
        routed, torch.tensor([[0.0, 1.0]]), atol=1e-6, rtol=0
    )

